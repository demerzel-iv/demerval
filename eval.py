import argparse
import json
import os
import random

from tqdm import tqdm

import task_utils
from metric_utils import add_mv_scores, add_oc_scores
from model_utils import MODEL_PATH, get_tokenizer_path
from solver import HintMathSolver, MathSolver

TASK_CLASS_MAP = {
    'math500': 'Math500TaskGenerator',
    'aime24': 'AimeTaskGenerator',
    'aime25': 'AimeTaskGenerator',
    'beyond_aime': 'AimeTaskGenerator',
    'olympiad_bench': 'OlympiadBenchTaskGenerator',
    'minervamath': 'MinervaMathTaskGenerator',
    'gsm8k': 'GSM8KTaskGenerator',
}

TASK_KWARGS_MAP = {
    'math500': {'repeat_times': 4},
    'aime24': {'repeat_times': 32, 'year': '2024'},
    'aime25': {'repeat_times': 32, 'year': '2025'},
    'beyond_aime': {'repeat_times': 32, 'year': 'beyond'},
    'olympiad_bench': {'repeat_times': 4},
    'minervamath': {'repeat_times': 8},
    'gsm8k': {'repeat_times': 4},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True, help='model name')
    parser.add_argument('--dataset', type=str, default='math500', choices=TASK_KWARGS_MAP.keys(), help='dataset name')
    parser.add_argument('--max-num-tokens', type=int, default=32768, help='max number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.6, help='sampling temperature')

    parser.add_argument('--output-dir', type=str, default='outputs/eval_results', help='directory to save eval results')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='tensor parallel size for the evaluated model')
    parser.add_argument('--batch-size', type=int, default=64, help='generation batch size per scheduled job')

    parser.add_argument('--skip-oc-score', action='store_true', help='skip OpenCompass verifier scoring')
    parser.add_argument('--use-hint-solver', action='store_true', help='use HintMathSolver for hint-augmented solving')

    return parser.parse_args()


def main():
    args = parse_args()

    base_model_name = args.model_name
    model_name = f'{args.model_name}-hint' if args.use_hint_solver else args.model_name
    dataset_name = args.dataset
    max_num_tokens = args.max_num_tokens
    temperature = args.temperature

    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    filename = f'{dataset_name}-{max_num_tokens // 1024}k-{temperature}.jsonl'
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(
            f'Results for model {model_name} on dataset {dataset_name} with max tokens '
            f'{max_num_tokens} already exist at {filepath}. Skipping evaluation.'
        )
        return

    print(f'task: {model_name}-{dataset_name}-{max_num_tokens // 1024}k, temperature={temperature}')

    llm_kwargs = {
        'model': MODEL_PATH[base_model_name],
        'tokenizer': get_tokenizer_path(base_model_name),
        'tensor_parallel_size': args.tensor_parallel_size,
        'gpu_memory_utilization': 0.9,
        'max_model_len': max_num_tokens,
        'hf_overrides': {
            'max_position_embeddings': max_num_tokens + 10000,
        },
    }
    sampling_params_kwargs = {
        'temperature': temperature,
        'top_p': 0.95,
        'max_tokens': max_num_tokens,
        'skip_special_tokens': False,
    }

    task_generator_class = getattr(task_utils, TASK_CLASS_MAP[dataset_name])
    task_generator_kwargs = TASK_KWARGS_MAP[dataset_name]
    task_generator = task_generator_class(**task_generator_kwargs)
    tasks = list(task_generator)
    random.shuffle(tasks)

    solver_class = HintMathSolver if args.use_hint_solver else MathSolver
    solver = solver_class(
        sampling_params_kwargs=sampling_params_kwargs,
        llm_kwargs=llm_kwargs,
        num_device=8,
        batch_size=args.batch_size,
    )
    results = solver.solve(tasks)

    if len(results) == 0:
        print('no results')
        return

    results = add_mv_scores(results)
    mv_acc = sum(result['mv_score'] for result in results) / len(results) * 100
    print(f'Name: {model_name}/{filename}, Math-Verify Accuracy: {mv_acc:.2f}')

    if not args.skip_oc_score:
        results = add_oc_scores(results)
        oc_acc = sum(result['oc_score'] for result in results) / len(results) * 100
        print(f'Name: {model_name}/{filename}, OpenCompass Accuracy: {oc_acc:.2f}')

    with open(filepath, 'w', encoding='utf-8') as f:
        for result in tqdm(results, desc='Writing results'):
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
