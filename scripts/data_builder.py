import time
import datetime
import numpy as np
import torch
import random
import argparse
import os
import json
import custom_datasets
from model import load_tokenizer, load_model
from sim_model import load_sim_model, FileSim, batcher
from sacremoses import MosesTokenizer
import google.generativeai as genai

PARAPHRASE = "_paraphrase"
ASSISTANT_SEPERATOR = '\nassistant\n'

SYSTEM_PROMPT_HUMAN = "You are a helpful paraphraser. You are given an input passage 'INPUT'. You should paraphrase 'INPUT' to print 'OUTPUT'. 'OUTPUT' should preserve the meaning and content of 'INPUT'. 'OUTPUT' should not be very shorter than 'INPUT'."
SYSTEM_PROMPT_MACHINE = "You are a helpful assistant."
 
def save_data(output_file, args, data):
    # write args to file
    output_file = output_file + '_lambda_{}_alpha_{}_temperature_{}_max_tries_{}'.format(args.lamda, args.alpha, args.temperature, args.max_tries)
    if args.paraphrase:
        output_file = output_file + PARAPHRASE
    args_file = f"{output_file}.args.json"
    with open(args_file, "w") as fout:
        json.dump(args.__dict__, fout, ensure_ascii=False, indent=4)
        print(f"Args written into {args_file}")

    # write the data to a json file in the save folder
    data_file = f"{output_file}.raw_data.json"
    if args.paraphrase:
        args_file = PARAPHRASE + data_file
    with open(data_file, "w") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)
        print(f"Raw data written into {data_file}")

class DataBuilder:
    def __init__(self, args):
        self.args = args
        self.base_tokenizer = load_tokenizer(args.base_model_name, args.dataset, args.cache_dir)
        self.base_model = None if (args.openai_model or args.gemini_model) else load_model(args.base_model_name, args.device, args.cache_dir)
        self.prompt_human = "Rewrite the following INPUT in the tone of a text message to a friend without any greetings or emojis:"
        self.prompt_machine = "Repeat the following paragraph:"

    def _openai_sample(self, prefix):
        def _drop_last_word(text):
            return ' '.join(text.split(' ')[:-1])

        import openai
        assert self.args.openai_key is not None, "Must provide OpenAI API key as --openai_key"
        openai.api_key = self.args.openai_key
        if self.args.openai_base is not None:
            openai.api_base = self.args.openai_base

        if self.args.dataset != 'pubmed':  # keep Answer: prefix for pubmed
            prefix = _drop_last_word(prefix)

        # sample from the openai model
        kwargs = {"max_tokens": 200}
        if self.args.do_top_p:
            kwargs['top_p'] = self.args.top_p
        elif self.args.do_top_k:
            kwargs['top_k'] = self.args.top_k
        elif self.args.do_temperature:
            kwargs['temperature'] = self.args.temperature

        if self.args.openai_model == 'davinci':
            kwargs["engine"] = self.args.openai_model
            response = openai.Completion.create(prompt=f"{prefix}", **kwargs)
            return prefix + response['choices'][0]['text']

        elif self.args.openai_model in ['gpt-3.5-turbo', 'gpt-4']:
            roles = {'xsum': 'You are a News writer.',
                     'writing': 'You are a Fiction writer.',
                     'pubmed': 'You are a Technical writer.'}
            prompts = {'xsum': 'Please write an article with about 150 words starting exactly with:',
                       'writing': 'Please write an article with about 150 words starting exactly with:',
                       'pubmed': 'Please answer the question in about 50 words.'}
            messages = [
                {'role': 'system', 'content': roles[self.args.dataset]},
                {'role': 'user', 'content': f'{prompts[self.args.dataset]} {prefix}'},
            ]
            kwargs["model"] = self.args.openai_model
            kwargs["messages"] = messages
            response = openai.ChatCompletion.create(**kwargs)
            response = response['choices'][0]['message']['content']
            # ChatGPT may repeat the prefix
            if response.startswith(prefix[:20]):
                return response
            return prefix + ' ' + response

        else:
            raise NotImplementedError
    
    def _gemini_sample(self, prefix):
        def _drop_last_word(text):
            return ' '.join(text.split(' ')[:-1])

        
        assert self.args.gemini_key is not None, "Must provide Gemini API key as --gemini_key"
        genai.configure(api_key=self.args.gemini_key)

        model = genai.GenerativeModel(self.args.gemini_model)

        if self.args.dataset != 'pubmed':  # keep Answer: prefix for pubmed
            prefix = _drop_last_word(prefix)

        prompts = {'xsum': 'You are a News writer. Please write an article with about 150 words starting exactly with:',
                    'writing': 'You are a Fiction writer. Please write an article with about 150 words starting exactly with:',
                    'pubmed': 'You are a Technical writer. Please answer the question in about 50 words.'}
        response = model.generate_content(
            f'{prompts[self.args.data_name]} {prefix}',
            generation_config=genai.types.GenerationConfig(
                # Only one candidate for now.
                candidate_count=1,
                max_output_tokens=200,
                temperature=self.args.temperature,
            ),
        )
        response = response.text
        
        # Gemini may repeat the prefix
        if response.startswith(prefix[:20]):
            return response
        return prefix + ' ' + response
    
    # sample from base_model using ****only**** the first 30 tokens in each example as context
    def _sample_from_model(self, texts, min_words=55, prompt_tokens=30):
        # encode each text as a list of token ids
        if self.args.dataset == 'pubmed':
            texts = [t[:t.index(custom_datasets.SEPARATOR)] for t in texts]
            all_encoded = self.base_tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
        else:
            all_encoded = self.base_tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
            all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

        if self.args.openai_model:
            # decode the prefixes back into text
            prefixes = self.base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)

            decoded = []
            for idx, prefix in enumerate(prefixes):
                while idx >= len(decoded):
                    try:
                        decoded.append(self._openai_sample(prefix))
                    except Exception as ex:
                        print(ex)
                        print('Wait 10 seconds before retry ...')
                        time.sleep(10)

        elif self.args.gemini_model:
            # decode the prefixes back into text
            prefixes = self.base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)

            decoded = []
            for idx, prefix in enumerate(prefixes):
                while idx >= len(decoded):
                    try:
                        decoded.append(self._gemini_sample(prefix))
                    except Exception as ex:
                        print(ex)
                        print('Wait 10 seconds before retry ...')
                        time.sleep(10)
        
        else:
            self.base_model.eval()
            decoded = ['' for _ in range(len(texts))]

            # sample from the model until we get a sample with at least min_words words for each example
            # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
            tries = 0
            m = 0
            while m < min_words:
                if tries != 0:
                    print()
                    print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")
                    prefixes = self.base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
                    for prefix, x in zip(prefixes, decoded):
                        if len(x.split()) == m:
                            print(prefix, '=>', x)

                sampling_kwargs = {}
                if self.args.do_top_p:
                    sampling_kwargs['top_p'] = self.args.top_p
                elif self.args.do_top_k:
                    sampling_kwargs['top_k'] = self.args.top_k
                elif self.args.do_temperature:
                    sampling_kwargs['temperature'] = self.args.temperature
                min_length = 50 if self.args.dataset in ['pubmed'] else 150
                outputs = self.base_model.generate(**all_encoded, min_length=min_length, max_length=200, do_sample=True,
                                                   **sampling_kwargs, pad_token_id=self.base_tokenizer.eos_token_id,
                                                   eos_token_id=self.base_tokenizer.eos_token_id)
                decoded = self.base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                m = min(len(x.split()) for x in decoded)
                tries += 1

        return decoded

    def _paraphrase_from_model(self, text, idx, max_tries):
        model = load_sim_model('paraphrase-at-scale/model.para.lc.100.pt')
        model.eval()
        therehold = 0.76
        from argparse import Namespace
        new_args = Namespace(batch_size=32, entok=MosesTokenizer(lang='en'), sp=model.sp,
                        model=model, lower_case=model.args.lower_case,
                        tokenize=model.args.tokenize)
        s = FileSim()

        model_kwargs = {"max_new_tokens": 512}
        if self.args.do_top_p:
            model_kwargs['top_p'] = self.args.top_p
        elif self.args.do_top_k:
            model_kwargs['top_k'] = self.args.top_k
        elif self.args.do_temperature:
            model_kwargs['temperature'] = self.args.temperature

        # input_text = ' '.join(text.split(' ')[:150])
        input_text = text
        messages_human = [
            {"role": "system", "content": SYSTEM_PROMPT_HUMAN},
            {"role": "user", "content": f'{self.prompt_human} {input_text}'}
        ]
        messages_machine = [
            {"role": "system", "content": SYSTEM_PROMPT_MACHINE},
            {"role": "user", "content": f'{self.prompt_machine} {input_text}'}
        ]

        text_human = self.base_tokenizer.apply_chat_template(
            messages_human,
            tokenize=False,
            add_generation_prompt=True
        )
        text_machine = self.base_tokenizer.apply_chat_template(
            messages_machine,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs_human = self.base_tokenizer([text_human], return_tensors="pt").to(self.base_model.device)
        model_inputs_machine = self.base_tokenizer([text_machine], return_tensors="pt").to(self.base_model.device)

        model_kwargs['model_inputs_machine'] = model_inputs_machine
        model_kwargs['lamda'] = self.args.lamda
        model_kwargs['alpha'] = self.args.alpha

        n, current_len, score, cur_max_score = 0, 0, 0, 0
        input_length = len(input_text.split(' '))
        print(f"The input length of input:{input_length}")
        max_response = ''
        while n < max_tries and cur_max_score <= therehold:
            n += 1
            generated_ids = self.base_model.generate(
                **model_inputs_human,
                **model_kwargs
            )
            response = self.base_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response = response.split(ASSISTANT_SEPERATOR)[1]
            current_len = len(response.split(' '))
            
            score = s.score(new_args, batcher, response, text)[0]
            
            print(f"Current n:{n} Current length:{current_len} Score:{score}")

            if score > cur_max_score:
                cur_max_score = score
                max_response = response
            
            if cur_max_score > therehold:
                break

        return max_response

    def generate_samples(self, raw_data, batch_size):
        # trim to shorter length
        def _trim_to_shorter_length(texta, textb):
            # truncate to shorter of o and s
            shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
            texta = ' '.join(texta.split(' ')[:shorter_length])
            textb = ' '.join(textb.split(' ')[:shorter_length])
            return texta, textb

        def _truncate_to_substring(text, substring, idx_occurrence):
            # truncate everything after the idx_occurrence occurrence of substring
            assert idx_occurrence > 0, 'idx_occurrence must be > 0'
            idx = -1
            for _ in range(idx_occurrence):
                idx = text.find(substring, idx + 1)
                if idx == -1:
                    return text
            return text[:idx]

        data = {
            "original": [],
            "sampled": [],
        }

        for batch in range(len(raw_data) // batch_size):
            print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
            original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
            sampled_text = self._sample_from_model(original_text, min_words=30 if self.args.dataset in ['pubmed'] else 55)

            for o, s in zip(original_text, sampled_text):
                if self.args.dataset == 'pubmed':
                    s = _truncate_to_substring(s, 'Question:', 2)
                    o = o.replace(custom_datasets.SEPARATOR, ' ')

                o, s = _trim_to_shorter_length(o, s)

                # add to the data
                data["original"].append(o)
                data["sampled"].append(s)

        return data

    def paraphrase_samples(self, data, batch_size, max_tries):
        original_data = data['original'][:args.n_samples]
        raw_data = data['sampled'][:args.n_samples]
        updated_data = {'original':[], 'sampled':[], 'paraphrased':[]}

        print(f"Paraphrasing prefixes with lamda {self.args.lamda}")
        for batch in range(len(raw_data) // batch_size):
            print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
            original_text = original_data[batch * batch_size:(batch + 1) * batch_size]
            sampled_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
            original, sampled, paraphrased = [], [], []
            for idx, prefix in enumerate(sampled_text):
                paraphrased_text = self._paraphrase_from_model(prefix, batch*batch_size + idx, max_tries)
                original.append(original_text[idx])
                sampled.append(prefix)
                paraphrased.append(paraphrased_text)
            updated_data['original'].extend(original)
            updated_data['sampled'].extend(sampled) 
            updated_data['paraphrased'].extend(paraphrased)

        return updated_data, self.prompt_human, self.prompt_machine


def generate_data(args, dataset):
    def load_data(file_path):
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                return json.load(file)
        else:
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    # load data
    data = load_data(dataset)

    data_builder = DataBuilder(args)

    print(f"Total number of samples: {len(data['original'])}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data['original']])}")

    return data_builder.generate_samples(data['original'][:args.n_samples], batch_size=args.batch_size)

def paraphrase_data(args, dataset):
    def load_data(file_path):
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                return json.load(file)
        else:
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    # load data
    data = load_data(dataset)
    
    data_builder = DataBuilder(args)

    # print stats about remaining data
    print(f"Total number of samples: {len(data['sampled'])}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data['sampled']])}")

    return data_builder.paraphrase_samples(data, batch_size=args.batch_size, max_tries=args.max_tries)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_gpt3/data/xsum_gpt2")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--data_name', type=str, default="xsum")
    parser.add_argument('--n_samples', type=int, default=150)
    parser.add_argument('--openai_base', type=str, default=None)
    parser.add_argument('--openai_key', type=str, default='add your API key')
    parser.add_argument('--openai_model', type=str, default=None)  # davinci, gpt-3.5-turbo, gpt-4
    parser.add_argument('--gemini_key', type=str, default='add your API key')
    parser.add_argument('--gemini_model', type=str, default=None)
    parser.add_argument('--paraphrase', action="store_true")       # If true, paraphrase the data
    parser.add_argument('--lamda', type=float, default=0.5)        # adjust the contrastive decoding
    parser.add_argument('--alpha', type=float, default=1e-5)
    parser.add_argument('--base_model_name', type=str, default="gpt2")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--do_temperature', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    parser.add_argument('--max_tries', type=int, default=10)
    args = parser.parse_args()

    os.environ["XDG_CACHE_HOME"] = args.cache_dir
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    print(f"Using cache dir {args.cache_dir}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f'Loading dataset {args.dataset}...')
    if args.paraphrase:
        paraphrase_data = paraphrase_data(args, args.dataset)
        data = paraphrase_data[0]
        args.__dict__.update({'system_prompt_human':SYSTEM_PROMPT_HUMAN, 'system_prompt_machine':SYSTEM_PROMPT_MACHINE, 'prompt_human':paraphrase_data[1], 
                                                    'prompt_machine':paraphrase_data[2]})
    else:
        data = generate_data(args, args.dataset)
    save_data(args.output_file, args, data)
