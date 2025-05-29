model_name=Qwen/Qwen2.5-VL-7B-Instruct
judge_model_name=Qwen/Qwen2.5-72B-Instruct  # gpt-4o-2024-11-20 or Qwen/Qwen2.5-72B-Instruct
judge_implement=huggingface     # api or huggingface
model_basename="${model_name##*/}"
result_path="predictions/${model_basename}"
mode=all
nframes=32
fps=-1
resolution=448
max_new_token=4096
temperature=0.01
num_proc=8

# only set either fps or nframes
if [ "$nframes" -gt 0 ]; then
    result_prefix="${resolution}p_${nframes}frames_${max_new_token}token_temperature${temperature}_${num_proc}"
elif [ "$fps" -gt 0 ]; then
    result_prefix="${resolution}p_${fps}fps_${max_new_token}token_temperature${temperature}_${num_proc}"
fi


for IDX in $(seq 0 $(($num_proc-1))); do
    CUDA_VISIBLE_DEVICES=$IDX python3 eval_huggingface.py \
        --model_name $model_name \
        --judge_model_name $judge_model_name \
        --mode $mode \
        --judge_implement $judge_implement \
        --result_path $result_path \
        --resolution $resolution \
        --nframes $nframes \
        --fps $fps \
        --max_new_token $max_new_token \
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