from typing import Dict, List, Optional

from vllm_utils import parallel_generate


MATH_USER_PROMPT = "{problem} Please reason step by step, and put your final answer within \\boxed{{}}."


class MathSolver:
    def __init__(
        self,
        sampling_params_kwargs: Dict,
        llm_kwargs: Dict,
        num_device: int = 8,
        batch_size: int = 64,
    ):
        self.sampling_params_kwargs = sampling_params_kwargs or {}
        self.llm_kwargs = llm_kwargs
        self.num_device = num_device
        self.batch_size = batch_size

    def build_message(self, task: Dict) -> List[Dict[str, str]]:
        return [{
            'role': 'user',
            'content': MATH_USER_PROMPT.format(problem=task['problem']),
        }]

    def solve(self, tasks: List[Dict]) -> List[Dict]:
        messages = [self.build_message(task) for task in tasks]

        task_outputs = parallel_generate(
            messages,
            self.llm_kwargs,
            self.sampling_params_kwargs,
            num_device=self.num_device,
            batch_size=self.batch_size,
        )

        return [
            self.format_result(task, task_output)
            for task, task_output in zip(tasks, task_outputs)
        ]

    def format_result(self, task: Dict, task_output) -> Dict:
        prompt_ids = list(task_output.prompt_token_ids)
        response = task_output.outputs[0].text
        num_generated_tokens = len(task_output.outputs[0].token_ids)

        return {
            'task_id': task['task_id'],
            'problem': task['problem'],
            'answer': task['answer'],
            'prompt_ids': prompt_ids,
            'generated_text': response,
            'num_generated_tokens': num_generated_tokens,
            'num_total_tokens': num_generated_tokens + len(prompt_ids),
        }
