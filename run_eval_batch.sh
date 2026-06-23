#!/bin/bash
source /home/test1267/test-6/miniconda3/etc/profile.d/conda.sh
conda activate dmz-lc

# set python path if account is test-6
export PYTHONPATH="/home/test1267/test-6/miniconda3/envs/dmz-lc/lib/python3.10/site-packages"
# add local bin to path for srun-helper
export PATH="/home/test/test06/lyq/.local/bin:${PATH}"
# set vllm logging level to ERROR to reduce log verbosity
export VLLM_LOGGING_LEVEL="ERROR"

export mode="exclude"
#export mode="specific"
export use_ssh="0"

export srun_part="TEST1"
export account="test"
nodes="g[80,81,84]"

model_list=(
    'qwen3-1.7b'
)

dataset_list=("aime24" "aime25" "beyond_aime" "math500" "olympiad_bench" "minervamath" "gsm8k")

max_num_tokens=32768
temperature=0.6
tmp_eval_root="/tmp/luoyuqi/eval_results"
final_eval_root="./outputs/eval_results"

for dataset in "${dataset_list[@]}"; do
    for model in "${model_list[@]}"; do
        # due to disk quota, output results to a temporary directory first, and then move to the final directory after evaluation is completed
        echo "Scheduling evaluation for model: ${model} on dataset: ${dataset}"
        filename="${dataset}-$((max_num_tokens / 1024))k-${temperature}.jsonl"
        tmp_output_dir="${tmp_eval_root}/${model}"
        final_output_dir="${final_eval_root}/${model}"
        tmp_filepath="${tmp_output_dir}/${filename}"
        final_filepath="${final_output_dir}/${filename}"

        if [ -f "${final_filepath}" ]; then
            echo "Results already exist at ${final_filepath}. Skipping evaluation."
            continue
        fi

        run_cmd="mkdir -p ${tmp_eval_root} ${final_output_dir} && python3 eval.py --model-name ${model} --dataset ${dataset} --max-num-tokens ${max_num_tokens} --temperature ${temperature} --output-dir ${tmp_eval_root} && mv -f ${tmp_filepath} ${final_filepath}"

        if [ "$use_ssh" -eq 0 ]; then
            CMD=$run_cmd srun-helper $nodes &
        else
            CMD=$run_cmd srun-helper $nodes
        fi

        sleep 0.1
    done
done

wait
echo "All evaluations completed."
