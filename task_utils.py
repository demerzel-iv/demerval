from typing import Dict, Iterator, Tuple
from datasets import load_dataset

class MathTaskGenerator:
    def __init__(self, repeat_times: int = 1):
        self.repeat_times = repeat_times
        self.dataset = None

    def get_problem_and_answer(self, data) -> Tuple[str, str]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Dict]:
        for i, data in enumerate(self.dataset):
            problem, answer = self.get_problem_and_answer(data)

            for j in range(self.repeat_times):
                yield {
                    'problem': problem,
                    'answer': answer,
                    'task_id': f'{i}-{j}',
                }

    def __len__(self) -> int:
        return len(self.dataset) * self.repeat_times

class Math500TaskGenerator(MathTaskGenerator):
    def __init__(self, repeat_times: int = 1):
        super().__init__(repeat_times)
        self.dataset = load_dataset('HuggingFaceH4/MATH-500', split='test')

    def get_problem_and_answer(self, data) -> Tuple[str, str]:
        return data['problem'], data['answer']

class AimeTaskGenerator(MathTaskGenerator):
    def __init__(self, repeat_times: int = 1, year: str = '2024'):
        super().__init__(repeat_times)

        if year == '2024':
            self.dataset = load_dataset("HuggingFaceH4/aime_2024", split='train')
        elif year == '2025':
            self.dataset = load_dataset("math-ai/aime25", split='test')
        elif year == 'beyond':
            self.dataset = load_dataset("ByteDance-Seed/BeyondAIME", split='test')
        else:
            raise ValueError(f'year {year} not supported')

    def get_problem_and_answer(self, data) -> Tuple[str, str]:
        answer = int(data['answer'])
        return data['problem'], str(answer)

class OlympiadBenchTaskGenerator(MathTaskGenerator):
    def __init__(self, repeat_times: int = 1):
        super().__init__(repeat_times)
        self.dataset = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split='train')

    def get_problem_and_answer(self, data) -> Tuple[str, str]:
        assert len(data['final_answer']) == 1
        return data['question'], data['final_answer'][0]

class MinervaMathTaskGenerator(MathTaskGenerator):
    def __init__(self, repeat_times: int = 1):
        super().__init__(repeat_times)
        self.dataset = load_dataset("math-ai/minervamath", split="test")

    def get_problem_and_answer(self, data) -> Tuple[str, str]:
        return data['question'], data['answer']

class GSM8KTaskGenerator(MathTaskGenerator):
    def __init__(self, repeat_times: int = 1):
        super().__init__(repeat_times)
        self.dataset = load_dataset("openai/gsm8k", "main", split="test")

    def get_problem_and_answer(self, data) -> Tuple[str, str]:
        answer = data['answer'].split("####")[1].strip()
        return data['question'], answer
