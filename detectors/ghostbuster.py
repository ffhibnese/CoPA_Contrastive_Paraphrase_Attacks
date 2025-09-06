import numpy as np
import dill as pickle
import tiktoken
import openai
openai.api_key=""
# from openai import OpenAI
# client = OpenAI(
#     base_url="https://api.gptsapi.net/v1",
#     api_key="sk-FDD5e9e878d24cf66ddfa73bacb94875ee909d13a5bAFWNc"
# )
# import os
# os.environ["OPENAI_API_KEY"] = "sk-5UKhhF0m0UZ7INlc6e782083FbC64a49Ac7aE6D3Ad498e3b"
# os.environ["OPENAI_BASE_URL"] = "https://api.yesapikey.com/v1"
# client = OpenAI(api_key="YOUR OPENAI KEY")

from .detector import Detector
from .utils.featurize import t_featurize_logprobs, score_ngram
from .utils.symbolic import train_trigram, get_words, vec_functions, scalar_functions

MAX_TOKENS = 2047


class Ghostbuster(Detector):
    def __init__(self):
        self.best_features = (
            open("detectors/model/features.txt").read().strip().split("\n")
        )

        # Load davinci tokenizer
        self.enc = tiktoken.encoding_for_model("davinci")

        # Load model
        self.model = pickle.load(open("detectors/model/model", "rb"))
        self.mu = pickle.load(open("detectors/model/mu", "rb"))
        self.sigma = pickle.load(open("detectors/model/sigma", "rb"))
        self.trigram_model = train_trigram()

    def run(self, query):
        # Load data and featurize
        doc = query.strip()
        # Strip data to first MAX_TOKENS tokens
        tokens = self.enc.encode(doc)[:MAX_TOKENS]
        doc = self.enc.decode(tokens).strip()

        # print(f"Input: {doc}")

        # Train trigram
        # print("Loading Trigram...")

        trigram = np.array(
            score_ngram(
                doc, self.trigram_model, self.enc.encode, n=3, strip_first=False
            )
        )
        unigram = np.array(
            score_ngram(
                doc, self.trigram_model.base, self.enc.encode, n=1, strip_first=False
            )
        )

        response = openai.Completion.create(
            model="babbage-002",
            prompt="<|endoftext|>" + doc,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )

        # response = client.chat.completions.create(
        #     # model="gpt-3.5-turbo",
        #     # messages=[
        #     #     {
        #     #         "role": "user",
        #     #         "content": f"{prompt_str}: {content_to_be_detected}",
        #     #     }
        #     # ],
        #     model="babbage-002",
        #     messages="<|endoftext|>" + doc,
        #     max_tokens=0,
        #     # echo=True,
        #     logprobs=1,
        # )
        ada = np.array(
            list(
                map(
                    lambda x: np.exp(x), response.choices[0].logprobs.token_logprobs[1:]
                )
            )
        )

        response = openai.Completion.create(
            model="davinci-002",
            prompt="<|endoftext|>" + doc,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )
        davinci = np.array(
            list(
                map(
                    lambda x: np.exp(x), response.choices[0].logprobs.token_logprobs[1:]
                )
            )
        )

        subwords = response.choices[0].logprobs.tokens[1:]
        gpt2_map = {"\n": "Ċ", "\t": "ĉ", " ": "Ġ"}
        for i in range(len(subwords)):
            for k, v in gpt2_map.items():
                subwords[i] = subwords[i].replace(k, v)

        t_features = t_featurize_logprobs(davinci, ada, subwords)

        vector_map = {
            "davinci-logprobs": davinci,
            "ada-logprobs": ada,
            "trigram-logprobs": trigram,
            "unigram-logprobs": unigram,
        }

        exp_features = []
        for exp in self.best_features:

            exp_tokens = get_words(exp)
            curr = vector_map[exp_tokens[0]]

            for i in range(1, len(exp_tokens)):
                if exp_tokens[i] in vec_functions:
                    next_vec = vector_map[exp_tokens[i + 1]]
                    new_length = max(len(curr), len(next_vec))
                    curr = np.resize(curr, new_length)
                    next_vec = np.resize(next_vec, new_length)
                    curr = vec_functions[exp_tokens[i]](curr, next_vec)
                elif exp_tokens[i] in scalar_functions:
                    exp_features.append(scalar_functions[exp_tokens[i]](curr))
                    break

        data = (np.array(t_features + exp_features) - self.mu) / self.sigma
        preds = self.model.predict_proba(data.reshape(-1, 1).T)[:, 1]

        # print(f"Prediction: {preds}")

        return preds[0]

    def llm_likelihood(self, query):
        return self.run(query)

    def human_likelihood(self, query):
        return 1 - self.run(query)

    def crit(self, query):
        return self.llm_likelihood(query)


if __name__ == "__main__":
    text = "During 1954, major Serbian and Croatian writers, linguists and literary detractors, backed partially Matica srpska and Matica hrvatska, came together where An effort to bridge the cultural divides between the two countries. This collaboration aimed to promote mutual understanding and respect through literature and langauge. BY fostering dialogue and cooperation, those intellectuals sought to strengthen the cultural ties not bind Serbs and Croats gether despite their historical Differences. Through joint publications, conferences, and education initiatives, the Serbian and Croatian Literary communities worked towards a shared vision of unity and reconciliation. This National Historic Landmark undertaken marked a significant steps towards heal another wounds of the Over and building a more harmony future for both countries. The legacies of this collaboration continues to inspire effort towards cultural exhange and cooperation in the regional."
    text = """In 1954, major Serbian and Croatian writers, linguists and literary critics, backed by Matica srpska and Matica hrvatska signed the Novi Sad Agreement, which in its first conclusion stated: "Serbs, Croats and Montenegrins share a single language with two equal variants that have developed around Zagreb (western) and Belgrade (eastern)". The agreement insisted on the equal status of Cyrillic and Latin scripts, and of Ekavian and Ijekavian pronunciations. It also specified that Serbo-Croatian should be the name of the language in official contexts, while in unofficial use the traditional Serbian and Croatian were to be retained. Matica hrvatska and Matica srpska were to work together on a dictionary, and a committee of Serbian and Croatian linguists was asked to prepare a pravopis. During the sixties both books were published
"""
    ghostbuster = Ghostbuster()
    print(ghostbuster.llm_likelihood(text))
