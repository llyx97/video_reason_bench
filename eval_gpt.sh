model_name=gpt-4o-2024-11-20
judge_model_name=gpt-4o-2024-11-20
judge_implement=api     # api or huggingface
api_type=openai         # openai or gemini
result_path="predictions/${model_name}"
fps=1.0
max_num_frames=128
resolution=448
max_new_token=8192
thinking_budget=8192
temperature=1.0
num_proc=1

if [ "$model_name" = "gemini-2.0-flash" ]; then
    thinking_budget=0
fi

if [ "$api_type" = "openai" ]; then
    result_prefix="${resolution}p_fps${fps}_${max_num_frames}maxfrm_${max_new_token}token_temperature${temperature}_${num_proc}"
elif [ "$api_type" = "gemini" ]; then
    result_prefix="${max_new_token}token_temperature${temperature}_think${thinking_budget}_${num_proc}"
fi

for IDX in $(seq 0 $(($num_proc-1))); do
    python3 eval_api.py \
        --model_name $model_name \
        --judge_model_name $judge_model_name \
        --judge_implement $judge_implement \
        --result_path $result_path \
        --api_type $api_type \
        --resolution $resolution \
        --max_num_frames $max_num_frames \
        --target_fps $fps \
        --max_new_token $max_new_token \
        --thinking_budget $thinking_budget \
        --temperature $temperature \
        --result_prefix $result_prefix \
        --num_chunks $num_proc \
        --chunk_idx $IDX  &
done

wait

python3 utils/show_results.py \
    --result_path $result_path \
    --result_prefix $result_prefix \
    # --overwrite_merge_result