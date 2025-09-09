exp_path=data
data_path=$exp_path/raw_data
res_path=$exp_path/results
source_model=gpt-3.5-turbo  # gpt-3.5-turbo | gpt-4 | gpt-4o | claude3.5 | gemini
D=$1  # xsum | squad | longqa
M=qwen2.5-72b
file_name="${D}_${source_model}.raw_data.json"

CUDA_VISIBLE_DEVICES=0,1,2 python scripts/data_builder.py --data_name $D --dataset $data_path/$file_name \
        --n_samples 150 --lamda 0.5 --paraphrase --do_temperature  --temperature 1 --max_tries 5 \
        --batch_size 1 --alpha 1e-5 --base_model_name $M --output_file $data_path/${D}_${M}_${source_model}
