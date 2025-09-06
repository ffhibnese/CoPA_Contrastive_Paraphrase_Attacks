# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import torch
from .model import load_tokenizer, load_model, get_model_fullname, from_pretrained

from .prob_estimator import ProbEstimator
from .detector import Detector

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


class Detect_GPT(Detector):
    def __init__(
        self,
        ref_path,
        pct_words_masked,
        mask_top_p,
        span_length,
        n_perturbations,
        scoring_model_name,
        mask_filling_model_name,
        seed=0,
        device0="cuda:0",
        device1="cuda:0",
        cache_dir="../cache",
    ):
        self.ref_path = ref_path
        self.pct_words_masked = pct_words_masked
        self.mask_top_p = mask_top_p
        self.span_length = span_length
        self.n_perturbations = n_perturbations
        self.scoring_model_name = scoring_model_name
        self.mask_filling_model_name = mask_filling_model_name
        self.seed = seed
        self.device0 = device0
        self.device1 = device0
        self.cache_dir = cache_dir

        # load model
        self.mask_model = self.load_mask_model(
            self.mask_filling_model_name, self.device1, self.cache_dir
        )
        self.mask_model.eval()
        self.scoring_tokenizer = load_tokenizer(
            self.scoring_model_name, None, self.cache_dir
        )
        self.scoring_model = load_model(self.scoring_model_name, "cpu", self.cache_dir)
        self.scoring_model.eval()
        self.scoring_model.to(self.device0)
        try:
            n_positions = self.mask_model.config.n_positions
        except AttributeError:
            n_positions = 512
        self.mask_tokenizer = self.load_mask_tokenizer(
            self.mask_filling_model_name, n_positions, self.cache_dir
        )
        self.prob_estimator = ProbEstimator(self.ref_path)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def load_mask_model(self, model_name, device1, cache_dir):
        model_name = get_model_fullname(model_name)
        # mask filling t5 model
        print(f"Loading mask filling model {model_name}...")
        mask_model = from_pretrained(AutoModelForSeq2SeqLM, model_name, {}, cache_dir)
        mask_model = mask_model.to(device1)
        return mask_model

    def load_mask_tokenizer(self, model_name, max_length, cache_dir):
        model_name = get_model_fullname(model_name)
        tokenizer = from_pretrained(
            AutoTokenizer, model_name, {"model_max_length": max_length}, cache_dir
        )
        return tokenizer

    def tokenize_and_mask(self, text, span_length, pct, ceil_pct=False):
        buffer_size = 1
        tokens = text.split(" ")[:512]
        mask_string = "<<<mask>>>"

        n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)
        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(0, len(tokens) - span_length)
            # start = np.random.randint(0, 512 - span_length)
            end = start + span_length
            search_start = max(0, start - buffer_size)
            search_end = min(len(tokens), end + buffer_size)
            # search_end = min(512, end + buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1

        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f"<extra_id_{num_filled}>"
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = " ".join(tokens)
        return text

    def count_masks(self, texts):
        return [
            len([x for x in text.split() if x.startswith("<extra_id_")])
            for text in texts
        ]

    # replace each masked span with a sample from T5 mask_model
    def replace_masks(self, mask_model, mask_tokenizer, texts):
        n_expected = self.count_masks(texts)
        stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(
            self.device1
        )
        outputs = mask_model.generate(
            **tokens,
            max_length=150,
            do_sample=True,
            top_p=self.mask_top_p,
            num_return_sequences=1,
            eos_token_id=stop_id,
        )
        return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def extract_fills(self, texts):
        # remove <pad> from beginning of each text
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

        # return the text in between each matched mask token
        extracted_fills = [pattern.split(x)[1:-1] for x in texts]

        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        return extracted_fills

    def apply_extracted_fills(self, masked_texts, extracted_fills):
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(" ") for x in masked_texts]

        n_expected = self.count_masks(masked_texts)

        # replace each mask token with the corresponding fill
        for idx, (text, fills, n) in enumerate(
            zip(tokens, extracted_fills, n_expected)
        ):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts

    def perturb_texts_(self, mask_model, mask_tokenizer, texts, ceil_pct=False):
        span_length = self.span_length
        pct = self.pct_words_masked
        masked_texts = [
            self.tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts
        ]
        raw_fills = self.replace_masks(mask_model, mask_tokenizer, masked_texts)
        extracted_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while "" in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == ""]
            print(
                f"WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}]."
            )
            print(texts)
            masked_texts = [
                self.tokenize_and_mask(x, span_length, pct, ceil_pct)
                for idx, x in enumerate(texts)
                if idx in idxs
            ]
            raw_fills = self.replace_masks(mask_model, mask_tokenizer, masked_texts)
            extracted_fills = self.extract_fills(raw_fills)
            new_perturbed_texts = self.apply_extracted_fills(
                masked_texts, extracted_fills
            )
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
        return perturbed_texts

    def perturb_texts(self, mask_model, mask_tokenizer, texts, ceil_pct=False):
        chunk_size = 10
        outputs = []
        for i in range(0, len(texts), chunk_size):
            outputs.extend(
                self.perturb_texts_(
                    mask_model,
                    mask_tokenizer,
                    texts[i : i + chunk_size],
                    ceil_pct=ceil_pct,
                )
            )
        return outputs

    # Get the log likelihood of each text under the base_model
    def get_ll(self, scoring_model, scoring_tokenizer, text):
        with torch.no_grad():
            tokenized = scoring_tokenizer(
                text, return_tensors="pt", return_token_type_ids=False
            ).to(self.device0)
            labels = tokenized.input_ids
            return -scoring_model(**tokenized, labels=labels).loss.item()

    def get_mask_ll(self, text, indexes):
        with torch.no_grad():
            tokenized = self.scoring_tokenizer(
                text, return_tensors="pt", return_token_type_ids=False
            ).to(self.device0)
            labels = tokenized.input_ids
            mask = torch.zeros_like(tokenized["input_ids"])
            for i in range(len(mask[0])):
                mask[0][i] = 1 if i in indexes else 0
            tokenized["attention_mask"] = mask.to(self.device0)
            return -self.scoring_model(**tokenized, labels=labels).loss.item()

    def get_lls(self, scoring_model, scoring_tokenizer, texts):
        return [self.get_ll(scoring_model, scoring_tokenizer, text) for text in texts]

    def generate_perturbs(self, original_text):
        n_perturbations = self.n_perturbations

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # generate perturb samples
        p_original_text = self.perturb_texts(
            self.mask_model,
            self.mask_tokenizer,
            [original_text for _ in range(n_perturbations)],
        )
        assert (
            len(p_original_text) == n_perturbations
        ), f"Expected {n_perturbations} perturbed samples, got {len(p_original_text)}"

        return original_text, p_original_text

    def get_tokens(self, query: str):
        # Find the token indexes of a word to make attention mask
        tokens_id = self.scoring_tokenizer.encode(query)
        tokens_id = tokens_id[: self.scoring_tokenizer.model_max_length - 2]
        tokens_id = (
            [self.scoring_tokenizer.bos_token_id]
            + tokens_id
            + [self.scoring_tokenizer.eos_token_id]
        )
        tokens = self.scoring_tokenizer.convert_ids_to_tokens(tokens_id)

        return tokens

    def run(self, query):
        # generate perturbations
        original_text, perturbed_original = self.generate_perturbs(query)

        # evaluate
        original_ll = self.get_ll(
            self.scoring_model, self.scoring_tokenizer, original_text
        )
        all_perturbed_original_ll = self.get_lls(
            self.scoring_model, self.scoring_tokenizer, perturbed_original
        )
        perturbed_original_ll = np.mean(all_perturbed_original_ll)
        perturbed_original_ll_std = (
            np.std(all_perturbed_original_ll)
            if len(all_perturbed_original_ll) > 1
            else 1
        )

        # compute diffs with perturbed
        if perturbed_original_ll_std == 0:
            perturbed_original_ll_std = 1
        prediction = (original_ll - perturbed_original_ll) / perturbed_original_ll_std

        # estimate likelihood using prediction
        llm_likelihood = self.prob_estimator.crit_to_prob(prediction)
        human_likelihood = 1 - llm_likelihood

        return llm_likelihood, human_likelihood, prediction

    def llm_likelihood(self, query):
        return self.experiment(query)[0]

    def human_likelihood(self, query):
        return self.experiment(query)[1]

    def crit(self, query):
        return self.experiment(query)[2]


# detect_gpt = Detect_GPT("./exp_main/results/*perturbation_100.json", 0.3, 1.0, 2, 10, "gpt2-xl", "t5-3b")

# original = "In 1954, major Serbian and Croatian writers, linguists and literary critics, backed by Matica srpska and Matica In 1954, a group of prominent Serbian and Croatian writers, linguists, and literary critics, led by Matica srpska and Matica hrvatska, launched a controversial campaign aimed at standardizing the Serbo-Croatian language. The campaign sought to unify the Bosnian, Croatian, and Serbian languages into a single, standardized form, in an effort to promote national unity and cultural identity in the region. This move was met with resistance from some quarters, with activists and scholars arguing that the proposed standard was too heavily influenced by Serbian, and that it would result in the loss of unique regional variations and cultural identities. Despite these objections, the standard was ultimately adopted, and it remains the basis for the Serbian, Croatian, and Bosnian languages today."

# sample = "In 1954 major Serbian and Croatian scribes linguists and literary critics, backed by Matica srpska and Matica In 1954 a group of prominent Serbian and Croatian writers, Linguists and literary critics, followed by Matica srpska and Matica hrvatska, launched a controversial campaign aimed at streamlining the SerboCroatian language. The campaign sought to unify the Bosnia Serb and Serbian languages into a single, standardized shape in an initiative to promote nation unity and cultural identity in the regions This move was met with resistance from some quarters, with activists and scholars arguing that the proposed standard was too massively influenced by Serbian, and that it would result in ofthe loss of unique regional variations and cultural names Despite these misgivings that standard was ultimately adopts and it remain in basis for the Serbia Croatian, and Bosnian languages today."

# print(detect_gpt.get_ll(detect_gpt.scoring_model, detect_gpt.scoring_tokenizer, original))
# print(detect_gpt.get_ll(detect_gpt.scoring_model, detect_gpt.scoring_tokenizer, sample))
