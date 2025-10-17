steering=True

steering_number=512
steering_coef=-4

steering_mode=after_o_proj
steering_mode_name=${steering_mode}_correct_0_100/mix_others_low_rank_1000
steering_real_mode=${steering_mode}_norm_threshold
steering_vector_path=/workspace/charlie/CREST_Internal/probing/steering_vector_zoo/MATH_train/Qwen3-30B-A3B-Thinking-2507/template-t0-n1-10000/hidden${steering_mode_name}

export STEERING=${steering}

export STEERING_NUMBER=${steering_number}
export STEERING_COEF=${steering_coef}

export STEERING_MODE=${steering_real_mode}
export STEERING_VECTOR_PATH=${steering_vector_path}

tensor_parallel_size=8
batch_size=16
max_examples=-1

seeds=(
    64
)

model_paths=(
    "Qwen/Qwen3-30B-A3B-Thinking-2507"
)
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
    # Qwen/Qwen3-4B-Thinking-2507
    # openai/gpt-oss-20b
    
datasets=(
    "aime25"
)
    # "AIME"
    # "amc23"
    # "cp"
    # "lcb"
    # "MATH500"
    # "GSM"
    # "gpqa"

for seed in "${seeds[@]}"; do
    for model_path in "${model_paths[@]}"; do
        for dataset in "${datasets[@]}"; do
            export MODEL_NAME_OR_PATH=${model_path}
            
            data_dir=results/Steering${steering}_numb${steering_number}_coef${steering_coef}_mode${steering_mode_name}/${model_path}/${dataset}

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
                --enforce_eager \
                --steering ${steering} \
                --steering_vector_path ${steering_vector_path} \
                --steering_number ${steering_number} \
                --steering_coef ${steering_coef} \
                --steering_mode ${steering_real_mode} > ${data_dir}/vllm.log 2>&1
        done
    done
done
