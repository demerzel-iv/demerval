import copy
import json
import os
import time
import random
from collections import defaultdict


result_root = '/home/test/testdata/luoyuqi/outputs_demerval/eval_results'
model_prefix = 'qwen3-1.7b-opsd-reproduce-lr1e-6'
steps = [25, 50, 75, 100]
dataset_file = 'aime24-32k-1.0.jsonl'

seed = time.time_ns() % (2**32)
num_samples = 8
max_generated_tokens = 8192*4
score_key = 'oc_score'


def read_by_problem(path):
    by_problem = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            problem_id = result['task_id'].split('-')[0]
            by_problem[problem_id].append(result)
    return by_problem


def apply_length_limit(result):
    new_result = copy.deepcopy(result)
    new_result['original_num_generated_tokens'] = result['num_generated_tokens']
    new_result['sim_max_generated_tokens'] = max_generated_tokens
    new_result['num_generated_tokens'] = min(result['num_generated_tokens'], max_generated_tokens)

    if result['num_generated_tokens'] > max_generated_tokens:
        new_result['mv_score'] = 0
        new_result['oc_score'] = 0.0
        new_result['sim_truncated'] = True
    else:
        new_result['sim_truncated'] = False

    return new_result


def summarize(results):
    total_correct = sum(float(result[score_key]) for result in results)
    return total_correct / len(results) * 100


def expected_accuracy(by_problem):
    correct_rate_sum = 0.0
    for problem_id in sorted(by_problem, key=int):
        results = by_problem[problem_id]
        success = 0
        for result in results:
            if result['num_generated_tokens'] <= max_generated_tokens and float(result[score_key]) == 1.0:
                success += 1

        correct_rate_sum += success / len(results)

    return correct_rate_sum / len(by_problem) * 100


def main():
    rng = random.Random(seed)

    print(f'seed={seed}, samples_per_problem={num_samples}, max_generated_tokens={max_generated_tokens}')
    print('')
    print('step\tsample_acc\texpected_acc')

    for step in steps:
        model_name = f'{model_prefix}-{step}step'
        input_path = os.path.join(result_root, model_name, dataset_file)
        by_problem = read_by_problem(input_path)

        sampled_results = []
        for problem_id in sorted(by_problem, key=int):
            sampled = rng.sample(by_problem[problem_id], num_samples)
            for result in sampled:
                sampled_results.append(apply_length_limit(result))

        sample_acc = summarize(sampled_results)
        expected_acc = expected_accuracy(by_problem)
        print(f'{step}\t{sample_acc:.2f}\t{expected_acc:.2f}')


if __name__ == '__main__':
    main()
