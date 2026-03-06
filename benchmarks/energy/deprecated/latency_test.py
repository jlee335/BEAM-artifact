import csv
import time
import numpy as np
import torch
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

MODEL_NAME = "Qwen/Qwen2.5-14B"

llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
)

def measure_latency(batch_size, ctx_len, max_tokens, num_iters=5, warmup=2):
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )
    # dummy token ids
    dummy_tokens = np.random.randint(10, 20000, size=(batch_size, ctx_len), dtype=np.int32)
    prompts: list[TokensPrompt] = [
        {"prompt_token_ids": seq.tolist()} for seq in dummy_tokens
    ]

    # warmup
    for _ in range(warmup):
        torch.cuda.synchronize()
        llm.generate(prompts, sampling_params=sampling_params)
        torch.cuda.synchronize()

    # measure
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts, sampling_params=sampling_params)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    return np.mean(times)

batch_sizes = [1, 2, 4, 8]
ctx_len = 512
out_lens = [1, 8, 32, 128]
OUTPUT_CSV = "latency_results.csv"

rows: list[tuple[int, int, float, float]] = []
for bs in batch_sizes:
    # baseline: 1 output token
    base_t = measure_latency(bs, ctx_len, max_tokens=1)
    for out in out_lens:
        t = measure_latency(bs, ctx_len, max_tokens=out)
        if out > 1:
            decode_per_token = (t - base_t) / (out - 1)
        else:
            decode_per_token = 0.0
        rows.append((bs, out, t, decode_per_token))

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["batch_size", "out_len", "total_ms", "approx_decode_ms_per_token"])
    writer.writerows(rows)

print(f"Saved latency results to {OUTPUT_CSV}")
