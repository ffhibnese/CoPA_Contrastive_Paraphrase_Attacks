DATA=$1
detector=$2  # logprob | logrank | dgpt | fdgpt | ghostbuster | roberta-base | roberta-large]
generator_llm="gpt-3.5-turbo"

CUDA_VISIBLE_DEVICES=0 python evaluate.py --data_generator_llm $generator_llm --detector $detector --dataset_file ${DATA}