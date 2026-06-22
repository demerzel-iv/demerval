from typing import Dict, List

from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

parse_cfg = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]

#DEFAULT_OPEN_COMPASS_PATH   = "opencompass/CompassVerifier-3B"
DEFAULT_OPEN_COMPASS_PATH  = "/home/test/test06/.cache/huggingface/hub/models--opencompass--CompassVerifier-3B/snapshots/27d6d660bdf6c2b8a9f45906d4b76c508af65ab0"

OPEN_COMPASS_PROMPT = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 
Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT 
B: INCORRECT
C: INVALID
Just return the letters "A", "B", or "C", with no text around it.
Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
<Original Question Begin>:
{question}
<Original Question End>
<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>
<Candidate's Answer Begin>: 
{llm_response}
<Candidate's Answer End>
Judging the correctness of the candidate's answer:
"""

def math_verify_judger(answer: str, response: str):
    if 'boxed' not in answer:
        answer = '\\boxed{' + answer.strip() + '}'

    pans = parse(answer, parse_cfg)
    psol = parse(response, parse_cfg)

    score = int(verify(pans, psol, timeout_seconds=3))

    return score

def add_mv_scores(results: List[Dict]) -> List[Dict]:
    for result in results:
        response = result['generated_text']
        answer = result['answer']

        if '</think>' in response:
            answer_part = response.split('</think>')[-1].strip()[-512:]
            mv_score = math_verify_judger(answer, answer_part)
        else:
            mv_score = 0

        result['mv_score'] = mv_score

    return results


def add_oc_scores(
    results: List[Dict],
) -> List[Dict]:
    if not results:
        return results

    from vllm_utils import parallel_generate

    model_inputs = []
    for result in results:
        response = result['generated_text']
        if '</think>' in response:
            answer_part = response.split('</think>')[-1].strip()[-512:]
        else:
            answer_part = ''

        model_inputs.append([{
            'role': 'user',
            'content': OPEN_COMPASS_PROMPT.format(
                question=result['problem'],
                gold_answer=result['answer'],
                llm_response=answer_part,
            ),
        }])

    outputs = parallel_generate(
        model_inputs,
        llm_kargs={
            'model': DEFAULT_OPEN_COMPASS_PATH,
            'tensor_parallel_size': 8,
        },
        sampling_params_kwargs={
            'temperature': 0.0,
            'max_tokens': 2048,
        },
        num_device=8,
        batch_size=4096,
    )

    for result, output in zip(results, outputs):
        judgement = output.outputs[0].text.strip()
        result['oc_score'] = 1.0 if judgement == "A" else 0.0

    return results

if __name__ == '__main__':
    answer = r'13.5\times10^{33}'
    sol = r'</think>\boxed{\frac{27}{2}\times10^{33}}'
    print(math_verify_judger(answer, sol))
