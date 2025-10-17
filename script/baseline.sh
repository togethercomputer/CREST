steering=False
export STEERING=${steering}

tensor_parallel_size=8
batch_size=16
max_examples=-1

seeds=(
    64
)
model_paths=(
    "Qwen/Qwen3-30B-A3B-Thinking-2507"
)

datasets=(
    "aime25"
)

for seed in "${seeds[@]}"; do
    for model_path in "${model_paths[@]}"; do
        for dataset in "${datasets[@]}"; do
            export MODEL_NAME_OR_PATH=${model_path}
            
            data_dir=results/Steering${steering}/${model_path}/${dataset}/seed${seed}/

            mkdir -p ${data_dir}
            
            python -u main_vllm.py \
                --model_name_or_path  ${model_path} \
                --save_dir ${data_dir} \
                --n_samples 1 \
                --temperature 0.6 \
                --max_tokens 32768 \
                --use_chat_format \
                --dataset ${dataset} \
                --remove_bos \
                --seed ${seed} \
                --tensor_parallel_size ${tensor_parallel_size} \
                --batch_size ${batch_size} \
                --max_examples ${max_examples} \
                --steering ${steering} > ${data_dir}/vllm.log 2>&1
        done
    done
done
