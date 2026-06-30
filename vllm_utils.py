import logging
import os
import signal
import sys
import time
import traceback
import multiprocessing as mp
from queue import Empty
from typing import List, Sequence, Tuple

from vllm import LLM, SamplingParams


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[{asctime}]{message}", datefmt="%H:%M:%S", style="{"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _signal_process_group(proc, sig):
    if proc.pid is None:
        return

    try:
        # Each worker calls os.setpgrp(), so its pid is also its process-group id.
        # Sending the signal to the group helps clean up vLLM/torch child
        # processes that may otherwise keep running and holding GPU memory.
        os.killpg(proc.pid, sig)
        logger.info(f"[main] signal pgid={proc.pid} sig={sig}")
    except ProcessLookupError:
        # The worker/process group may have exited between the liveness check
        # and killpg; that race is harmless.
        pass


def _device_groups(num_device: int, tensor_parallel_size: int) -> List[Tuple[int, ...]]:
    assert num_device == 8, "Current code assumes exactly 8 visible devices."
    assert tensor_parallel_size > 0
    assert num_device % tensor_parallel_size == 0

    return [
        tuple(range(start, start + tensor_parallel_size))
        for start in range(0, num_device, tensor_parallel_size)
    ]


def _count_output_tokens(outputs) -> int:
    total_tokens = 0
    for request_output in outputs:
        if not hasattr(request_output, "outputs"):
            continue
        for completion_output in request_output.outputs:
            token_ids = getattr(completion_output, "token_ids", None)
            if token_ids is not None:
                total_tokens += len(token_ids)
    return total_tokens


def _worker(
    worker_rank: int,
    device_ids: Sequence[int],
    llm_kwargs,
    sampling_params_kwargs,
    enable_thinking,
    task_queue,
    result_queue,
):
    # Put this worker into a new process group. Libraries like vLLM/torch may
    # spawn child processes; grouping them with the worker lets the parent send
    # SIGTERM/SIGKILL to the whole group during cleanup.
    os.setpgrp()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device_id) for device_id in device_ids)

    logger.info(
        f"[worker {worker_rank}] start, pid={os.getpid()}, "
        f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
    )

    llm = LLM(**llm_kwargs)
    # Build SamplingParams inside each worker so the parent only needs to pass
    # simple kwargs through multiprocessing.
    sampling_params = SamplingParams(**sampling_params_kwargs)
    tokenizer = None
    if enable_thinking is not None:
        tokenizer = llm.get_tokenizer()

    logger.info(f"[worker {worker_rank}] llm initialized")

    while True:
        task = task_queue.get()
        if task is None:
            logger.info(f"[worker {worker_rank}] received stop signal")
            break

        batch_id, start_idx, messages = task
        end_idx = start_idx + len(messages)
        logger.info(
            f"[worker {worker_rank}] run batch_id={batch_id}, "
            f"range=[{start_idx}:{end_idx})"
        )

        try:
            # vLLM shows a tqdm progress bar by default; each worker logging its
            # own bar makes multi-process terminal output noisy, so keep progress
            # reporting in the main process only.
            if enable_thinking is None:
                outputs = llm.chat(messages, sampling_params, use_tqdm=False)
            else:
                # Offline LLM.chat in this vLLM version does not expose
                # enable_thinking, so apply the chat template here and pass
                # tokenized prompts to generate.
                prompts = []
                for message in messages:
                    prompt_text = tokenizer.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                    prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                    prompts.append({"prompt_token_ids": prompt_token_ids})

                outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        except Exception as exc:
            # Report the traceback before re-raising so the main process can
            # fail fast instead of waiting forever on a missing batch result.
            result_queue.put(
                (
                    "error",
                    worker_rank,
                    batch_id,
                    repr(exc),
                    traceback.format_exc(),
                )
            )
            raise

        result_queue.put(("result", worker_rank, batch_id, start_idx, outputs))
        logger.info(
            f"[worker {worker_rank}] finished batch_id={batch_id}, "
            f"outputs={len(outputs)}"
        )

    result_queue.put(("done", worker_rank))
    logger.info(f"[worker {worker_rank}] exit")


def parallel_generate(
    messages_list: List,
    llm_kargs,
    sampling_params_kwargs,
    num_device: int = 8,
    batch_size: int = 64,
    enable_thinking=None,
):
    """Generate chat completions with dynamic batch scheduling across vLLM workers.

    The previous implementation split the full input into one fixed chunk per GPU.
    Here the main process splits inputs into small batches and workers pull from a
    shared queue, which keeps faster workers busy when some batches are slower.
    """

    if not messages_list:
        return []

    assert batch_size > 0
    # Accept either one-message conversations or already grouped conversations.
    # Workers always receive list[list[message]], matching vLLM's chat API.
    messages_list = [
        [message] if isinstance(message, dict) else message
        for message in messages_list
    ]
    sampling_params_kwargs = sampling_params_kwargs or {}
    tensor_parallel_size = llm_kargs["tensor_parallel_size"]
    device_groups = _device_groups(num_device, tensor_parallel_size)
    num_workers = len(device_groups)

    logger.info(
        f"[main] parallel_generate start, total messages={len(messages_list)}, "
        f"num_device={num_device}, tp={tensor_parallel_size}, "
        f"num_workers={num_workers}, batch_size={batch_size}"
    )

    batches = [
        (batch_id, start_idx, messages_list[start_idx : start_idx + batch_size])
        for batch_id, start_idx in enumerate(range(0, len(messages_list), batch_size))
    ]
    batch_lengths = {batch_id: len(messages) for batch_id, _, messages in batches}
    logger.info(f"[main] total batches={len(batches)}")

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    processes = []

    # Queue all batches up front; each worker pulls the next available batch,
    # which keeps faster GPU groups busy when generation lengths vary.
    for batch in batches:
        task_queue.put(batch)
    for _ in range(num_workers):
        task_queue.put(None)

    for worker_rank, device_ids in enumerate(device_groups):
        process = ctx.Process(
            target=_worker,
            args=(
                worker_rank,
                device_ids,
                llm_kargs,
                sampling_params_kwargs,
                enable_thinking,
                task_queue,
                result_queue,
            ),
        )
        process.start()
        processes.append(process)
        logger.info(
            f"[main] process started: worker={worker_rank}, pid={process.pid}, "
            f"devices={','.join(str(device_id) for device_id in device_ids)}"
        )

    results = [None] * len(messages_list)
    finished_batches = 0
    failed_error = None
    total_output_tokens = 0
    start_time = time.monotonic()

    try:
        while finished_batches < len(batches):
            try:
                message = result_queue.get(timeout=60)
            except Empty:
                logger.info("[main] waiting result timeout, checking worker status")
                failed_workers = [
                    idx for idx, process in enumerate(processes) if process.exitcode not in (None, 0)
                ]
                if failed_workers:
                    failed_error = RuntimeError(f"Worker crashed, failed workers: {failed_workers}")
                    break
                continue

            message_type = message[0]
            if message_type == "result":
                _, worker_rank, batch_id, start_idx, outputs = message
                expected_len = batch_lengths[batch_id]
                if len(outputs) != expected_len:
                    raise RuntimeError(
                        f"Worker {worker_rank} returned {len(outputs)} outputs for "
                        f"batch {batch_id}, expected {expected_len}"
                    )
                end_idx = start_idx + len(outputs)
                # Each worker returns a whole batch; place it back at the
                # original slice so caller-visible ordering is deterministic.
                results[start_idx:end_idx] = outputs
                finished_batches += 1
                total_output_tokens += _count_output_tokens(outputs)
                elapsed = max(time.monotonic() - start_time, 1e-9)
                avg_output_tokens_per_second = total_output_tokens / elapsed
                logger.info(
                    f"[main] received batch_id={batch_id} from worker={worker_rank}, "
                    f"range=[{start_idx}:{end_idx}), "
                    f"finished={finished_batches}/{len(batches)}, "
                    f"toks/s={avg_output_tokens_per_second:.2f}"
                )
            elif message_type == "error":
                _, worker_rank, batch_id, exc_repr, tb = message
                failed_error = RuntimeError(
                    f"Worker {worker_rank} failed on batch {batch_id}: {exc_repr}\n{tb}"
                )
                break
            elif message_type == "done":
                _, worker_rank = message
                logger.info(f"[main] worker={worker_rank} done")

        if failed_error is not None:
            raise failed_error

        missing = [idx for idx, output in enumerate(results) if output is None]
        if missing:
            raise RuntimeError(f"Missing outputs for {len(missing)} messages; first index={missing[0]}")

        logger.info(f"[main] merged output len={len(results)}")
        return results
    finally:
        logger.info("[main] cleaning up workers")

        for process in processes:
            process.join(timeout=10)
            logger.info(
                f"[main] join phase-1: pid={process.pid}, "
                f"alive={process.is_alive()}, exitcode={process.exitcode}"
            )

        for process in processes:
            if process.is_alive():
                logger.info(f"[main] terminate pid={process.pid}")
                _signal_process_group(process, signal.SIGTERM)
                process.terminate()

        for process in processes:
            process.join(timeout=2)
            logger.info(
                f"[main] join phase-2: pid={process.pid}, "
                f"alive={process.is_alive()}, exitcode={process.exitcode}"
            )

        for process in processes:
            if process.is_alive():
                logger.info(f"[main] kill pid={process.pid}")
                _signal_process_group(process, signal.SIGKILL)
                process.kill()

        for process in processes:
            process.join(timeout=1)
            logger.info(
                f"[main] join phase-3: pid={process.pid}, "
                f"alive={process.is_alive()}, exitcode={process.exitcode}"
            )

        for queue in (task_queue, result_queue):
            queue.cancel_join_thread()
            queue.close()
