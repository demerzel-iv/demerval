import argparse
import json
import os
from pathlib import Path

import numpy as np


RESULT_ROOT = '/home/test/testdata/luoyuqi/outputs_demerval/eval_results'


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    x = np.arange(size) - (size - 1) / 2
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


def plot_filled_curve(x: np.ndarray, y: np.ndarray, color: str, ax) -> None:
    ax.plot(x, y, color=color)
    ax.fill_between(x, y, color=color, alpha=0.25)


def parse_csv(value: str) -> list[str]:
    items = []
    for item in value.split(','):
        stripped = item.strip()
        if stripped:
            items.append(stripped)
    if not items:
        raise ValueError(f'Empty comma-separated value: {value}')
    return items


def read_results(path: str) -> list[dict]:
    results = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if 'oc_score' not in data:
                raise ValueError(f'Key oc_score not found in result data {path}')
            results.append(data)
    return results


def build_filename(dataset_name: str, max_num_tokens: int, temperature: float) -> str:
    return f'{dataset_name}-{max_num_tokens // 1024}k-{temperature}.jsonl'


def find_result_path(model_name: str, dataset_name: str, max_num_tokens: int, temperature: float) -> str:
    filename = build_filename(dataset_name, max_num_tokens, temperature)
    return os.path.join(RESULT_ROOT, model_name, filename)


def calculate_accuracy(results: list[dict]) -> float:
    if len(results) == 0:
        return 0.0
    return sum(float(result['oc_score']) for result in results) / len(results) * 100


def draw_result(results: list[dict], result_path: str, max_length: int, accuracy: float) -> None:
    import matplotlib

    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    correct_cnt = np.zeros(max_length)
    wrong_cnt = np.zeros(max_length)

    for result in results:
        num_generated_tokens = int(result['num_generated_tokens'])
        if num_generated_tokens >= max_length:
            num_generated_tokens = max_length - 1

        if int(result['oc_score']) == 1:
            correct_cnt[num_generated_tokens] += 1
        else:
            wrong_cnt[num_generated_tokens] += 1

    kernel_size = max(3, int(max_length / 4))
    sigma = max(1.0, max_length / 50)
    kernel = gaussian_kernel(kernel_size, sigma)
    correct = np.convolve(correct_cnt, kernel, mode='same')
    wrong = np.convolve(wrong_cnt, kernel, mode='same')

    _, ax = plt.subplots(figsize=(10, 6))
    plot_filled_curve(np.arange(max_length), correct, 'g', ax)
    plot_filled_curve(np.arange(max_length), wrong, 'r', ax)

    ax.set_xbound(0, max_length)
    ax.set_ybound(0, None)
    ax.set_xlabel('length')
    ax.set_ylabel('num')
    ax.set_title(f'acc {accuracy:.2f} %')
    ax.grid(True)

    fig_path = str(Path(result_path).with_suffix('.png'))
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f'saved figure to {fig_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Draw and summarize Demerval eval results.')
    parser.add_argument('--model-name', type=str, required=True, help='Comma-separated model names')
    parser.add_argument('--dataset-name', type=str, required=True, help='Comma-separated dataset names')
    parser.add_argument('--output-txt', type=str, required=True, help='Path to save tab-separated summary')
    parser.add_argument('--max-num-tokens', type=int, default=32768, help='max number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.6, help='sampling temperature')
    parser.add_argument('--draw', action='store_true', help='draw length distribution figures')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = parse_csv(args.model_name)
    dataset_names = parse_csv(args.dataset_name)
    max_length = int(args.max_num_tokens * 1.1)

    lines = ['model\t' + '\t'.join(dataset_names)]

    for model_name in model_names:
        row = [model_name]
        for dataset_name in dataset_names:
            result_path = find_result_path(
                model_name,
                dataset_name,
                args.max_num_tokens,
                args.temperature,
            )

            if not os.path.exists(result_path):
                print(f'Result file not found: {result_path}')
                row.append('NA')
                continue

            print(f'processing {result_path}')
            results = read_results(result_path)
            accuracy = calculate_accuracy(results)
            row.append(f'{accuracy:.2f}')

            if args.draw:
                draw_result(results, result_path, max_length, accuracy)

        lines.append('\t'.join(row))

    output_txt = os.path.abspath(args.output_txt)
    output_dir = os.path.dirname(output_txt)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print('')
    print('\n'.join(lines))
    print('')
    print(f'saved summary to {output_txt}')


if __name__ == '__main__':
    main()
