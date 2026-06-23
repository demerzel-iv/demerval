from typing import Dict, List, Optional

from vllm_utils import parallel_generate


MATH_USER_PROMPT = "{problem} Please reason step by step, and put your final answer within \\boxed{{}}."
HINT_MATH_USER_PROMPT = """{problem}

Helpful hint:
{hint}

Please reason step by step, and put your final answer within \\boxed{{}}."""

MAKE_HINT_PROMPT = """Act as my math tutor. I am a student trying to solve this complex math problem on my own.
Provide some high-level problem-solving ideas and explain relevant theorems.

CRITICAL:
- DO NOT solve the problem for me.
- DO NOT provide the final answer.
- NO CONVERSATIONAL FILLER. Do not start with "Sure", "Absolutely", "I'd be happy to help", etc. Jump IMMEDIATELY into the first point.

Please include:
1. Problem Deconstruction: Briefly clarify the key constraints and objective.
2. Relevant Theorems & Tools: Explain mathematical theorems or structural properties that might apply.
3. Potential Strategies: Outline 1-2 promising directions or attack plans I could explore.

Here is the problem:
{problem_text}"""

DEFAULT_HINT_SAMPLING_PARAMS_KWARGS = {
    'temperature': 0.7,
    'top_p': 0.95,
    'max_tokens': 4096,
}


class MathSolver:
    def __init__(
        self,
        sampling_params_kwargs: Dict,
        llm_kwargs: Dict,
        num_device: int = 8,
        batch_size: int = 64,
    ):
        self.sampling_params_kwargs = sampling_params_kwargs
        self.llm_kwargs = llm_kwargs
        self.num_device = num_device
        self.batch_size = batch_size

    def solve(self, tasks: List[Dict]) -> List[Dict]:
        messages = [{
            'role': 'user',
            'content': MATH_USER_PROMPT.format(problem=task['problem']),
        } for task in tasks]

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


class HintMathSolver(MathSolver):
    def __init__(
        self,
        sampling_params_kwargs: Dict,
        llm_kwargs: Dict,
        num_device: int = 8,
        batch_size: int = 64,
        hint_sampling_params_kwargs: Optional[Dict] = None,
    ):
        super().__init__(
            sampling_params_kwargs=sampling_params_kwargs,
            llm_kwargs=llm_kwargs,
            num_device=num_device,
            batch_size=batch_size,
        )
        self.hint_sampling_params_kwargs = dict(DEFAULT_HINT_SAMPLING_PARAMS_KWARGS)
        if hint_sampling_params_kwargs is not None:
            self.hint_sampling_params_kwargs.update(hint_sampling_params_kwargs)
        self.hint_sampling_params_kwargs['n'] = 1

    def solve(self, tasks: List[Dict]) -> List[Dict]:
        hint_messages = [{
            'role': 'user',
            'content': MAKE_HINT_PROMPT.format(
                problem_text=task['problem'],
            ),
        } for task in tasks]

        hint_outputs = parallel_generate(
            hint_messages,
            self.llm_kwargs,
            self.hint_sampling_params_kwargs,
            num_device=self.num_device,
            batch_size=self.batch_size,
            enable_thinking=False,
        )

        solve_messages = []
        hint_infos = []
        for task, hint_output in zip(tasks, hint_outputs):
            hint_response = hint_output.outputs[0]
            hint_generation_issue = None
            hint_text = None

            if hint_response.finish_reason == 'stop':
                hint_text = hint_response.text
                content = HINT_MATH_USER_PROMPT.format(
                    problem=task['problem'],
                    hint=hint_text,
                )
            else:
                hint_generation_issue = hint_response.finish_reason
                content = MATH_USER_PROMPT.format(problem=task['problem'])

            solve_messages.append({
                'role': 'user',
                'content': content,
            })
            hint_infos.append({
                'hint_text': hint_text,
                'hint_generation_issue': hint_generation_issue,
            })

        task_outputs = parallel_generate(
            solve_messages,
            self.llm_kwargs,
            self.sampling_params_kwargs,
            num_device=self.num_device,
            batch_size=self.batch_size,
        )

        return [
            self.format_result(task, task_output, hint_info)
            for task, task_output, hint_info in zip(tasks, task_outputs, hint_infos)
        ]

    def format_result(self, task: Dict, task_output, hint_info: Dict) -> Dict:
        result = super().format_result(task, task_output)
        result['hint_text'] = hint_info['hint_text']
        result['hint_generation_issue'] = hint_info['hint_generation_issue']
        return result
