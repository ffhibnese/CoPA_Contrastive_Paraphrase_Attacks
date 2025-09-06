import os
import re
import json
import time
import uuid
import datetime
import random
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import torch
import glob
import nltk
from nltk.tokenize import word_tokenize
import logging
import pandas as pd
from openai import OpenAI
from gensim.models import KeyedVectors

from detectors.baselines import Baselines
from detectors.ghostbuster import Ghostbuster
from detectors.detect_gpt import Detect_GPT
from detectors.fast_detect_gpt import Fast_Detect_GPT
from detectors.roberta_gpt2_detector import GPT2RobertaDetector


client = OpenAI(
    api_key=""
)


def openai_backoff(**kwargs):
    retries, wait_time = 0, 10
    while retries < 10:
        try:
            return client.chat.completions.create(**kwargs)
        except Exception:
            time.sleep(wait_time)
            wait_time *= 2
            retries += 1


class Experiment:
    def __init__(
        self,
        dataset,
        data_generator_llm,
        paraphrased_model,
        proxy_model,
        detector,
        output_path,
        proxy_model_device,
        target_detector_device,
        mask_pct,
        top_k,
        candidate_generation,
        dataset_file,
    ):
        self.dataset = dataset
        self.data_generator_llm = data_generator_llm
        self.paraphrased_model = paraphrased_model
        self.proxy_model_str = proxy_model
        self.mask_pct = mask_pct
        self.output_path = output_path
        self.dataset_file = dataset_file
        self.detector = detector
        self.target_detector_device = target_detector_device
        self.proxy_model_device = proxy_model_device

        self.proxy_model_tokenizer = None
        self.proxy_model = None
        self.top_k = top_k

        self.proxy_model_type = (
            "detection"
            if self.proxy_model_str
            in ["roberta-base-detector", "roberta-large-detector"]
            else "next-token-generation"
        )

        self.candidate_generation = candidate_generation
        if self.candidate_generation == "word2vec":
            self.word_vectors = KeyedVectors.load_word2vec_format(
                "./assets/GoogleNews-vectors-negative300.bin.gz", binary=True
            )  # , limit=500000)

    def filter_punctuation(self, string):
        pattern = r"^[\W_]+|[\W_]+$"

        left_punctuation = re.findall(r"^[\W_]+", string)
        right_punctuation = re.findall(r"[\W_]+$", string)
        clean_string = re.sub(pattern, "", string)

        return "".join(left_punctuation), "".join(right_punctuation), clean_string

    def get_pos(self, word):
        tokens = word_tokenize(word)
        tagged = nltk.pos_tag(tokens)
        return tagged[0][1] if tagged else None

    def are_same_pos(self, word1, word2):
        pos1 = self.get_pos(word1)
        pos2 = self.get_pos(word2)
        return pos1 == pos2

    def get_top_similar_words(self, word, n=15):
        try:
            similar_words = self.word_vectors.most_similar(word, topn=n)
            return similar_words
        except (KeyError, AttributeError):
            self.logger.info(f"'{word}' is not in the vocabulary.")
            return []

    def generate_text(self, query):
        response = openai_backoff(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content

    def predict_words(self, paragraph, top_k):
        query = f"""Given some input paragraph, we have highlighted a word using brackets. List {top_k} alternative words for it that ensure grammar correctness and semantic fluency. Output words only.\n{paragraph}"""
        output = self.generate_text(query)
        predicted_words = re.findall(r"\b[a-zA-Z]+\b", output)
        
        if type(predicted_words) == list:
            return predicted_words[:top_k]
        if len(predicted_words) == top_k:
            return predicted_words[:top_k]
        else:
            print(f"OpenAI returned else: {predicted_words}")
            return []

    def load_data(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                return json.load(file)
        else:
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    def remove_punctuation(text):
        return re.sub(r"[^\w\s]", "", text)

    def flatten(self, lst):
        flattened_list = []
        for item in lst:
            if isinstance(item, list):
                flattened_list.extend(self.flatten(item))
            else:
                flattened_list.append(item)
        return flattened_list

    def load_dataset(self) -> None:
        """
        Loads the dataset from the given directory into object self.data.
        """
        if self.dataset not in ["xsum", "squad", "abstract", "writing"]:
            raise ValueError(
                "Selected Dataset is invalid. Valid choices: 'xsum','squad','abstract', 'writing'"
            )
        if self.data_generator_llm not in [
            "gpt-3.5-turbo",
            "gpt-4",
            "davinci",
            "mixtral-8x7B-Instruct",
            "llama-3-70b-chat",
        ]:
            raise ValueError(
                "Selected Data Generator LLM is invalid. Valid choices: 'gpt-3.5-turbo', 'mixtral-8x7B-Instruct', 'llama-3-70b-chat'"
            )
        

        # file_name = os.path.join(
        #     self.dataset_dir, f"{self.dataset}_{self.paraphrased_model}_{self.data_generator_llm}_paraphrase.raw_data.json"
        # )
        file_name = self.dataset_file
        if os.path.exists(file_name):
            self.data = self.load_data(file_name)
            print(
                f"Dataset {self.dataset} generated with {self.data_generator_llm} loaded successfully!"
            )
        else:
            raise ValueError(f"Data filepath {file_name} does not exist")

    def load_proxy_model(self) -> None:
        proxy_model_map = {
            "roberta-base-detector": "roberta-base",
            "roberta-large-detector": "roberta-large",
        }

        proxy_model_checkpoint_map = {
            "roberta-base-detector": "./assets/detector-base.pt",
            "roberta-large-detector": "./assets/detector-large.pt",
        }
        if self.proxy_model_str in proxy_model_map.keys():
            self.proxy_model = GPT2RobertaDetector(
                model_name=proxy_model_map[self.proxy_model_str],
                device=self.proxy_model_device,
                checkpoint=proxy_model_checkpoint_map[self.proxy_model_str],
            )
        elif self.proxy_model == "gpt2":
            from transformers import GPT2LMHeadModel, GPT2Tokenizer

            self.proxy_model_tokenizer = GPT2Tokenizer.from_pretrained(
                self.proxy_model_str
            )
            self.proxy_model = GPT2LMHeadModel.from_pretrained(self.proxy_model_str)
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.proxy_model_tokenizer = AutoTokenizer.from_pretrained(
                self.proxy_model_str, torch_dtype=torch.float16
            )
            self.proxy_model = AutoModelForCausalLM.from_pretrained(
                self.proxy_model_str, torch_dtype=torch.float16
            )

        if self.proxy_model_device != "cpu":
            self.proxy_model.to(self.proxy_model_device)
            print(f"{self.proxy_model_str} model pushed to {self.proxy_model_device}")
        print(
            f"{self.proxy_model_str} model and tokenizer (if applicable) loaded successfully!"
        )

    def load_detector(self) -> None:
        if self.detector == "dgpt":
            self.detector_model = Detect_GPT(
                "./detectors/*sampling_discrepancy.json",
                0.3,
                1.0,
                2,
                10,
                "gpt2-xl",
                "t5-3b",
                device0=self.target_detector_device,
                device1=self.target_detector_device,
            )
        elif self.detector == "fdgpt":
            self.detector_model = Fast_Detect_GPT(
                "gpt2-xl",
                "gpt2-xl",
                "xsum",
                "./detectors/*sampling_discrepancy.json",
                self.target_detector_device,
            )
            # print("I\'m upset")

        elif self.detector == "ghostbuster":
            self.detector_model = Ghostbuster()
        elif self.detector == "logrank":
            self.detector_model = Baselines(
                "logrank", "gpt-neo-2.7B", device=self.target_detector_device
            )
        elif self.detector == "logprob":
            self.detector_model = Baselines(
                "likelihood", "gpt-neo-2.7B", device=self.target_detector_device
            )
        # elif self.detector == "roberta-base":
        #     self.detector_model = GPT2RobertaDetector(
        #         "/mnt/data/kongjiawei/SimLLM/ckpt/checkpoint-376/", self.target_detector_device, "/mnt/data/kongjiawei/SimLLM/ckpt/checkpoint-376/optimizer.pt"
        #     )
        elif self.detector == "roberta-base":
            self.detector_model = GPT2RobertaDetector(
                "/mnt/data/kongjiawei/cache/local.FacebookAI_roberta-base",
                self.target_detector_device,
                "./assets/detector-base.pt",
            )
        elif self.detector == "roberta-large":
            self.detector_model = GPT2RobertaDetector(
                "/mnt/data/kongjiawei/cache/local.FacebookAI_roberta-large",
                self.target_detector_device,
                "./assets/detector-large.pt",
            )

    def create_experiment(self) -> None:
        current_date = datetime.datetime.now()
        formatted_date = current_date.strftime("%Y-%m-%d-%H-%M-%S")
        uid = str(uuid.uuid4())
        self.experiment_name = f"{self.dataset}_{self.data_generator_llm}_{self.proxy_model_str.replace('/','')}_{self.detector}_{formatted_date}"
        self.experiment_path = os.path.join(self.output_path, self.experiment_name)

        os.makedirs(self.experiment_path)

        self.config = {
            "dataset": self.dataset,
            "data_generator_llm": self.data_generator_llm,
            "proxy_model": self.proxy_model_str,
            "proxy_type": self.proxy_model_type,
            "mask_pct": self.mask_pct,
            "detector": self.detector,
            "output_path": self.output_path,
            "dataset_file": self.dataset_file,
            "timestamp_created": str(current_date),
            "candidate_generation": self.candidate_generation,
            "experiment_name": self.experiment_name,
            "experiment_path": self.experiment_path,
        }

        with open(os.path.join(self.experiment_path, "config.json"), "w") as f:
            json.dump(self.config, f)

        logging.basicConfig(
            filename=os.path.join(self.experiment_path, "experiment.log"),
            filemode="w",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info(f"Creating experiment {self.experiment_name}")

        check_path = os.path.join(
            self.output_path,
            f"{self.dataset}_{self.data_generator_llm}_{self.proxy_model_str.replace('/','')}_{self.detector}_*)",
        )
        if len(sorted(glob.glob(check_path))) > 0:
            self.logger.warning("Duplicated experiment detected")

    def raft(self) -> None:
        data = self.data

        # originals / results -> Probability of LLM Generated
        # original_crits / result_crits -> Likelihood of LLM Generated
        originals, results, original_crits, result_crits = [], [], [], []
        original_texts, result_texts = [], []

        n_samples = len(data["sampled"])

        for index in tqdm(range(n_samples)):
            paragraph = data["sampled"][index]
            words = paragraph.split()
            len_paragraph = len(words)
            ranks = {}
            # Proxy scoring model for determining priority of words to replace
            if self.proxy_model_type == "next-token-generation":
                tokens_id = self.proxy_model_tokenizer.encode(
                    paragraph, add_special_tokens=True
                )
                logits = self.proxy_model(
                    torch.tensor(tokens_id).unsqueeze(0).to(self.proxy_model_device)
                ).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                ranks = [(0, -1, "[HEAD]", 0.0)]
                for i in range(1, len(probs[0])):
                    token_id = tokens_id[i]
                    ranks.append(
                        (
                            i,
                            token_id,
                            self.proxy_model_tokenizer.convert_ids_to_tokens(token_id),
                            probs[0][i - 1][token_id].item(),
                        )
                    )  # append as (token position, token id, token, token_prob)
                ranks.sort(key=lambda x: x[3])
                percent_masked = self.mask_pct
                num_masks = int(len(probs[0]) * percent_masked)
                ranks_filter = list(filter(lambda x: "Ġ" in x[2], ranks))
                for rank_filter in ranks_filter:
                    if rank_filter[2].replace("Ġ", "") == "":
                        ranks_filter.remove(rank_filter)
                sorted_keys = [x[0] for x in ranks_filter]
                sorted_words = [x[2].replace("Ġ", "") for x in ranks_filter]

            if self.proxy_model_type == "detection":
                words_original = words.copy()
                for i in range(len_paragraph):
                    paragraph_new = " ".join(
                        words[:i] + ["=", words[i], "="] + words[i + 1 :]
                    )
                    tokens = self.proxy_model.get_tokens(paragraph_new)
                    if tokens[2] == "=":
                        tokens[2] = "Ġ="
                    start_end = [i for i, x in enumerate(tokens) if x == "Ġ="]
                    highlight_indexes = [
                        i for i in range(start_end[0], start_end[1] - 1)
                    ]
                    ranks[i] = self.proxy_model.llm_likelihood(
                        paragraph, highlight_indexes
                    )

                sorted_keys = [
                    k for k, v in sorted(ranks.items(), key=lambda item: item[1])
                ]
                sorted_words = [words[k] for k in sorted_keys]

            mask_keys, num_masks = [], int(len_paragraph * self.mask_pct)

            if self.proxy_model_type == "next-token-generation":
                ctr = 0
                candidates = []
                while ctr < num_masks:
                    token_pos, token_id, token, prob = ranks_filter.pop()
                    candidates.append((token_pos, token_id, token, prob))
                    ctr += 1

                changes = 0
                best_words = []
                for candidate in candidates:
                    token_pos, token_id, token, prob = candidate
                    word = self.proxy_model_tokenizer.decode(token_id).strip()
                    min_score, best_word = self.detector_model.crit(paragraph), word

                    word_to_replace = self.proxy_model_tokenizer.decode(
                        tokens_id[token_pos]
                    ).strip()
                    self.logger.info(f"{index} - Word to replace: {word_to_replace}")
                    paragraph_query = (
                        self.proxy_model_tokenizer.decode(
                            self.flatten(tokens_id[:token_pos])
                        )
                        + f"[{self.proxy_model_tokenizer.decode(tokens_id[token_pos]).strip()}]"
                        + self.proxy_model_tokenizer.decode(
                            self.flatten(tokens_id[token_pos + 1 :])
                        )
                    )

                    similar_words = self.predict_words(paragraph_query, self.top_k)
                    self.logger.info(
                        f"{index} - Returned candidate words: {similar_words}"
                    )
                    for similar_word in similar_words:
                        if self.are_same_pos(word_to_replace, similar_word):
                            paragraph_temp = (
                                self.proxy_model_tokenizer.decode(
                                    self.flatten(tokens_id[:token_pos])
                                )
                                + " "
                                + similar_word
                                + " "
                                + self.proxy_model_tokenizer.decode(
                                    self.flatten(tokens_id[token_pos + 1 :])
                                )
                            )
                            score = self.detector_model.crit(paragraph_temp)
                            if score <= min_score:
                                best_word = similar_word
                                min_score = score
                                changes += 1

                    best_words.append(best_word)
                    if best_word == word:
                        self.logger.info(f"{index} -Word {word} not replaced")
                        continue
                    else:
                        # print(f'Word {word_to_replace} replaced with {best_word}')
                        old_val = tokens_id[token_pos]
                        tokens_id[token_pos] = self.proxy_model_tokenizer.encode(
                            " " + best_word.strip(), add_special_tokens=True
                        )

                paragraph_final = self.proxy_model_tokenizer.decode(
                    self.flatten(tokens_id)
                )

                """
                Result reporting
                """
                original_texts.append(paragraph)
                result_texts.append(paragraph_final)
                if self.detector in ["roberta-base", "roberta-large", "ghostbuster"]:
                    original_crit = self.detector_model.crit(paragraph)
                    original = original_crit
                    result_crit = self.detector_model.crit(paragraph_final)
                    result = result_crit
                else:
                    original, _, original_crit = self.detector_model.run(paragraph)
                    result, _, result_crit = self.detector_model.run(paragraph_final)
                originals.append(original)
                results.append(result)
                original_crits.append(original_crit)
                result_crits.append(result_crit)

                output_json = {
                    "original": paragraph,
                    "sampled": paragraph_final,
                    "replacement_keys": [x[0] for x in candidates],
                    "original_crit": original_crit,
                    "sampled_crit": result_crit,
                    "original_llm_likelihood": original,
                    "sampled_llm_likelihood": result,
                }
                with open(
                    os.path.join(self.experiment_path, f"{self.dataset}_{index}.json"),
                    "w",
                ) as output_file:
                    json.dump(output_json, output_file)
                result_json = {
                    "mean_original": np.mean(originals),
                    "mean_result": np.mean(results),
                    "originals": originals,
                    "results": results,
                    "original_crits": original_crits,
                    "result_crits": result_crits,
                }

                with open(
                    os.path.join(self.experiment_path, "results.json"), "w"
                ) as result_file:
                    json.dump(result_json, result_file)

            if self.proxy_model_type == "detection":
                for key in sorted_keys:
                    if num_masks == 0:
                        break
                    (
                        left_punctuation,
                        right_punctuation,
                        word_to_replace,
                    ) = self.filter_punctuation(words[key])
                    paragraph_query = " ".join(
                        words[:key]
                        + [left_punctuation, f"[{word_to_replace}]", right_punctuation]
                        + words[key + 1 :]
                    )
                    predicted_words = self.predict_words(paragraph_query, self.top_k)
                    min_score, word_best, replaced = float("inf"), words[key], False
                    for predicted_word in predicted_words:
                        if predicted_word not in [
                            "",
                            " ",
                            word_to_replace,
                        ] and self.are_same_pos(word_to_replace, predicted_word):
                            predicted_word = (
                                left_punctuation + predicted_word + right_punctuation
                            )
                            paragraph_new = " ".join(
                                words[:key] + [predicted_word] + words[key + 1 :]
                            )
                            score = self.detector_model.crit(paragraph_new)
                            if score <= min_score:
                                word_best = predicted_word
                                min_score = score
                                replaced = True
                    if replaced:
                        num_masks -= 1
                        mask_keys.append(key)
                        words[key] = word_best
                paragraph_final = " ".join(words)

                """
                Result reporting
                """
                original_texts.append(paragraph)
                result_texts.append(paragraph_final)

                if self.detector in ["roberta-base", "roberta-large", "ghostbuster"]:
                    original_crit = self.detector_model.crit(paragraph)
                    original = original_crit
                    result_crit = self.detector_model.crit(paragraph_final)
                    result = result_crit
                else:
                    original, _, original_crit = self.detector_model.run(paragraph)
                    result, _, result_crit = self.detector_model.run(paragraph_final)

                originals.append(original)
                results.append(result)
                original_crits.append(original_crit)
                result_crits.append(result_crit)

                output_json = {
                    "original": paragraph,
                    "sampled": paragraph_final,
                    "replacement_keys": mask_keys,
                    "original_crit": original_crit,
                    "sampled_crit": result_crit,
                    "original_llm_likelihood": original,
                    "sampled_llm_likelihood": result,
                }
                with open(
                    os.path.join(self.experiment_path, f"{self.dataset}_{index}.json"),
                    "w",
                ) as output_file:
                    json.dump(output_json, output_file)

                print(
                    f"""Wrote to file {(os.path.join(self.experiment_path, f"{self.dataset}_{index}.json"))}"""
                )

                result_json = {
                    "mean_original": np.mean(originals),
                    "mean_sampled": np.mean(results),
                    "originals": originals,
                    "sampled": results,
                    "original_crits": original_crits,
                    "sampled_crits": result_crits,
                }

                with open(
                    os.path.join(self.experiment_path, "results.json"), "w"
                ) as result_file:
                    json.dump(result_json, result_file)

            for handler in self.logger.handlers:
                handler.flush()

        df = pd.DataFrame(
            {
                "original_text": original_texts,
                "sampked_text": result_texts,
                "original_crit": original_crits,
                "sampled_crit": result_crits,
                "original_llm_likelihood": originals,
                "sampled_llm_likelihood": results,
            }
        ).to_csv(os.path.join(self.experiment_path, "results.csv"))

    def hmcd(self) -> None:
        data = self.data

        # originals / results -> Probability of LLM Generated
        # original_crits / result_crits -> Likelihood of LLM Generated
        originals, original_crits = [], []
        samples, sampled_crits = [], []
        paraphrases, paraphrased_crits = [], []
        n_samples = len(data["original"])

        for index in tqdm(range(n_samples)):
            paragraph = data["original"][index]
            sampled_paragraph = data["sampled"][index]
            paraphrased_paragraph = data["paraphrased"][index]
            if self.detector in ["roberta-base", "roberta-large", "ghostbuster"]:
                original_crit = self.detector_model.crit(paragraph)
                sampled_crit = self.detector_model.crit(sampled_paragraph)
                paraphrased_crit = self.detector_model.crit(paraphrased_paragraph)
                original = original_crit
                sampled = sampled_crit
                paraphrased = paraphrased_crit
            else:
                original, _, original_crit = self.detector_model.run(paragraph)
                sampled, _, sampled_crit = self.detector_model.run(sampled_paragraph)
                paraphrased, _, paraphrased_crit = self.detector_model.run(
                    paraphrased_paragraph
                )
            
            originals.append(original)
            original_crits.append(original_crit)
            samples.append(sampled)
            sampled_crits.append(sampled_crit)
            paraphrases.append(paraphrased)
            paraphrased_crits.append(paraphrased_crit)

            output_json = {
                "original": paragraph,
                "sampled": sampled_paragraph,
                "paraphrased": paraphrased_paragraph,
                "original_crit": original_crit,
                "sampled_crit": sampled_crit,
                "paraphrased_crit": paraphrased_crit,
                "original_llm_likelihood": original,
                "sampled_llm_likelihood": sampled,
                "paraphrased_llm_likelihood": paraphrased
            }
            with open(
                os.path.join(self.experiment_path, f"{self.dataset}_{index}.json"),
                "w",
            ) as output_file:
                json.dump(output_json, output_file)

            print(
                    f"""Wrote to file {(os.path.join(self.experiment_path, f"{self.dataset}_{index}.json"))}"""
                )
        
        result_json = {
            "mean_original": np.mean(originals),
            "mean_sampled" : np.mean(samples),
            "mean_paraphrased" : np.mean(paraphrases),
            "originals": originals,
            "sampled" : samples,
            "paraphrased" : paraphrases,
            "original_crits": original_crits,
            "sampled_crits" : sampled_crits,
            "paraphrased_crits" : paraphrased_crits 
        }

        with open(
            os.path.join(self.experiment_path, "results.json"), "w"
        ) as result_file:
            json.dump(result_json, result_file)

        for handler in self.logger.handlers:
            handler.flush()

        df = pd.DataFrame(
            {
                "original_text": paragraph[:n_samples],
                'sampled_text' : sampled_paragraph[:n_samples],
                "paraphrased_text": paraphrased_paragraph[:n_samples],
                "original_crit": original_crits,
                "sampled_crit": sampled_crits,
                "paraphrased_crit": paraphrased_crits,
                "original_llm_likelihood": originals,
                "sampled_llm_likelihood": samples,
                "paraphrased_llm_likelihood": paraphrases
            }
        ).to_csv(os.path.join(self.experiment_path, "results.csv"))

        fpr, tpr, _ = roc_curve([0]*len(originals) + [1]*len(samples), originals+samples)
        tpr_at_fpr = np.interp(0.05, fpr, tpr)
        print(f"Real mean/std: {np.mean(originals[:50]):.2f}/{np.std(originals[:50]):.2f}, Samples mean/std: {np.mean(samples[:50]):.2f}/{np.std(samples[:50]):.2f}, Paraphrase mean/std: {np.mean(paraphrases[:50]):.2f}/{np.std(paraphrases[:50]):.2f}")
        print("sample TPR at 5% FPR:{}".format(tpr_at_fpr))
        tpr_at_fpr = np.interp(0.01, fpr, tpr)
        print("sample TPR at 1% FPR:{}".format(tpr_at_fpr))


        fpr, tpr, _ = roc_curve([0]*len(originals) + [1]*len(paraphrases), originals+paraphrases)
        tpr_at_fpr = np.interp(0.05, fpr, tpr)
        print("paraphrase TPR at 5% FPR:{}".format(tpr_at_fpr))
        tpr_at_fpr = np.interp(0.01, fpr, tpr)
        print("paraphrase TPR at 1% FPR:{}".format(tpr_at_fpr))


    def run(self) -> None:
        self.load_dataset()
        self.load_proxy_model()
        self.load_detector()
        self.create_experiment()
        self.raft()
    
    def run_paraphrase(self) -> None:
        self.load_dataset()
        # self.load_proxy_model()
        self.load_detector()
        self.create_experiment()
        self.hmcd()
        # self.raft()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["xsum", "squad", "abstract", "writing"], default="xsum")
    parser.add_argument("--mask_pct", default=0.1)
    parser.add_argument("--top_k", default=15)
    parser.add_argument("--data_generator_llm", choices=["gpt-3.5-turbo", "mixtral-8x7B-Instruct", "llama-3-70b-chat"], default="gpt-3.5-turbo",)
    parser.add_argument(
        "--proxy_model",
        choices=[
            "roberta-base-detector",
            "roberta-large-detector",
            "gpt2",
            "opt-2.7b",
            "neo-2.7b",
            "gpt-j-6b",
        ],
        default="gpt2",
    )
    parser.add_argument(
        "--detector",
        choices=[
            "logprob",
            "logrank",
            "dgpt",
            "fdgpt",
            "ghostbuster",
            "roberta-base",
            "roberta-large",
        ],
        default="roberta-base",
    )
    parser.add_argument("--paraphrased_model", default="qwen2.5-72b")
    parser.add_argument("--output_path", default="./data/evaluate_result/")
    parser.add_argument("--proxy_model_device", default="cpu")
    parser.add_argument("--target_detector_device", default="cpu")
    parser.add_argument(
        "--candidate_generation",
        choices=["gpt-3.5-turbo", "word2vec"],
        default="gpt-3.5-turbo",
    )
    parser.add_argument("--dataset_file", required=True, type=str, help="Path to dataset file")
    return parser.parse_args()


print("Done Loading!")

if __name__ == "__main__":
    args = get_args()
    print("Running test...")
    print("Initializing")
    experiment = Experiment(
        args.dataset,
        args.data_generator_llm,
        args.paraphrased_model,
        args.proxy_model,
        args.detector,
        args.output_path,
        args.proxy_model_device,
        args.target_detector_device,
        args.mask_pct,
        args.top_k,
        args.candidate_generation,
        args.dataset_file,
    )
    experiment.run_paraphrase()
