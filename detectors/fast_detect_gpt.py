# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
from .model import load_tokenizer, load_model

from .prob_estimator import ProbEstimator
from .detector import Detector


class Fast_Detect_GPT(Detector):
    def __init__(
        self,
        reference_model_name,
        scoring_model_name,
        dataset,
        ref_path,
        device="cuda",
        cache_dir="../cache",
    ):
        self.reference_model_name = reference_model_name
        self.scoring_model_name = scoring_model_name
        self.dataset = dataset
        self.ref_path = ref_path
        self.device = device
        self.cache_dir = cache_dir

        # load model
        self.scoring_tokenizer = load_tokenizer(
            self.scoring_model_name, self.dataset, self.cache_dir
        )
        self.scoring_model = load_model(
            self.scoring_model_name, self.device, self.cache_dir
        )
        self.scoring_model.eval()
        if self.reference_model_name != self.scoring_model_name:
            self.reference_tokenizer = load_tokenizer(
                self.reference_model_name, self.dataset, self.cache_dir
            )
            self.reference_model = load_model(
                self.reference_model_name, self.device, self.cache_dir
            )
            self.reference_model.eval()
        # evaluate criterion
        self.criterion_fn = self.get_sampling_discrepancy_analytic
        self.prob_estimator = ProbEstimator(self.ref_path)

    def get_sampling_discrepancy_analytic(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        labels = (
            labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        )
        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        probs_ref = torch.softmax(logits_ref, dim=-1)
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(
            mean_ref
        )
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(
            dim=-1
        ).sqrt()
        discrepancy = discrepancy.mean()
        return discrepancy.item()

    def get_tokens(self, query):
        tokenized = self.scoring_tokenizer(
            query, return_tensors="pt", padding=True, return_token_type_ids=False
        ).to(self.device)
        input, output = tokenized["input_ids"].detach().cpu().numpy().tolist(), []
        for i in input:
            token = self.scoring_tokenizer.convert_ids_to_tokens(i)
            output.append(token)

        return output[0]

    # def get_mask_ll(self, query, indexes=None):
    #     with torch.no_grad():
    #         tokenized = self.scoring_tokenizer(query, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
    #         labels = tokenized.input_ids
    #         if indexes:
    #             mask = torch.zeros_like(tokenized["input_ids"])
    #             for i in range(len(mask[0])):
    #                 mask[0][i] = 1 if i in indexes else 0
    #             tokenized["attention_mask"] = mask.to(self.device)
    #         return - self.scoring_model(**tokenized, labels=labels).loss.item()

    # run local inference
    def run(self, query, indexes=None):
        # evaluate query
        tokenized = self.scoring_tokenizer(
            query, return_tensors="pt", padding=True, return_token_type_ids=False
        ).to(self.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            if indexes:
                mask = torch.ones_like(tokenized["input_ids"])
                for i in range(len(mask[0])):
                    mask[0][i] = 0 if i in indexes else 1
                tokenized["attention_mask"] = mask.to(self.device)
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            # print(logits_score)
            # print(logits_score.size())
            # exit()
            if self.reference_model_name == self.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.reference_tokenizer(
                    query,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).to(self.device)
                assert torch.all(
                    tokenized.input_ids[:, 1:] == labels
                ), "Tokenizer is mismatch."
                logits_ref = self.reference_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        # estimate the probability of machine generated query
        llm_likelihood = self.prob_estimator.crit_to_prob(crit)
        human_likelihood = 1 - llm_likelihood

        return llm_likelihood, human_likelihood, crit

    def llm_likelihood(self, query, indexes=None):
        return self.run(query, indexes)[0]

    def human_likelihood(self, query: str, indexes=None):
        return self.run(query, indexes)[1]

    def crit(self, query: str, indexes=None):
        return self.run(query, indexes)[2]


if __name__ == "__main__":
    fast_detect_gpt = Fast_Detect_GPT(
        "gpt2-xl", "gpt2-xl", "xsum", "exp_main/results/*sampling_discrepancy.json"
    )

    original = "Hello and welcome to Morgan Stanley’s Investment Community Group. My name is Mike Wilson and I am Morgan Stanley’s Chief U.S. Equity Strategist and Chief Investment Officer. As Chief Investment Officer and Chairman of the Global Investment Committee, I am currently responsible for the economic and statistical analysis of Morgan Stanley’s portfolio data, asset prices and global economic megatrends. To celebrate the 27th anniversary of the new Morgan Stanley Investment Group, we’re bringing you a two-month sharing of investment knowledge and tips from the WhatsApp Group. While everyone is here to profit, I hope you will share Morgan Stanley Group with your friends and join our brokerage team. As a global investment firm, we work together to create long-term value for investors, companies, shareholders, individuals and communities Those who like to invest or want to optimize the allocation of resources to grow their existing wealth, our team of analysts can help you Join us now and get an initial margin worth 200USDT when you register your trading account"

    print(fast_detect_gpt.llm_likelihood(original))
