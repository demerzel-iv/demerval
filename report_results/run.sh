#!/bin/bash
set -euo pipefail

source /home/test1267/test-6/miniconda3/etc/profile.d/conda.sh
conda activate dmz-lc

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

model_list=(
    'qwen3-1.7b'
    'qwen3-1.7b-hint'
    'qwen3-8b-hint'
)
dataset_list=("math500" "gsm8k" "aime24" "aime25" "beyond_aime" "olympiad_bench" "minervamath")

model_names=$(IFS=','; echo "${model_list[*]}")
dataset_names=$(IFS=','; echo "${dataset_list[*]}")
output_txt="/home/test/testdata/luoyuqi/outputs_demerval/eval_results.txt"
max_num_tokens=32768
temperature=0.6

run_cmd=(
    python3 "$SCRIPT_DIR/eval_results.py"
    --model-name "$model_names"
    --dataset-name "$dataset_names"
    --output-txt "$output_txt"
    --max-num-tokens "$max_num_tokens"
    --temperature "$temperature"
    #--draw
)

"${run_cmd[@]}"
