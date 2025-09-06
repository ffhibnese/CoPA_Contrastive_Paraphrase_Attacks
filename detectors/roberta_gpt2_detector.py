from .detector import Detector


import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer


class GPT2RobertaDetector(Detector):
    def __init__(
        self,
        model_name="roberta-large",
        device="cpu",
        checkpoint="./assets/detector-large.pt",
    ):
        checkpoint_weights = torch.load(checkpoint, map_location=device)

        self.model = RobertaForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.device = device

        self.model.load_state_dict(checkpoint_weights["model_state_dict"], strict=False)
        self.model.to(self.device)
        self.model.eval()
        print("%s detector model loaded on %s" % (model_name, device))

    def get_tokens(self, query: str):
        # Find the token indexes of a word to make attention mask
        tokens_id = self.tokenizer.encode(query)
        tokens_id = tokens_id[: self.tokenizer.model_max_length - 2]
        tokens_id = (
            [self.tokenizer.bos_token_id] + tokens_id + [self.tokenizer.eos_token_id]
        )
        tokens = self.tokenizer.convert_ids_to_tokens(tokens_id)

        return tokens

    def _calculate_likelihood(self, query: str, indexes: any):
        tokens = self.tokenizer.encode(query)
        tokens = tokens[: self.tokenizer.model_max_length - 2]
        tokens = torch.tensor(
            [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        ).unsqueeze(0)

        if indexes:
            mask = torch.zeros_like(tokens)
            for i in range(len(mask[0])):
                mask[0][i] = 1 if i in indexes else 0.5
        else:
            mask = torch.ones_like(tokens)

        with torch.no_grad():
            logits = self.model(
                tokens.to(self.device), attention_mask=mask.to(self.device)
            )[0]
            probs = logits.softmax(dim=-1)

        fake, real = probs.detach().cpu().flatten().numpy().tolist()
        return fake, real

    def llm_likelihood(self, query: str, indexes=None):
        return self._calculate_likelihood(query, indexes)[0]

    def human_likelihood(self, query: str, indexes=None):
        return self._calculate_likelihood(query, indexes)[1]

    def crit(self, query):
        return self.llm_likelihood(query)
