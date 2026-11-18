#!/usr/bin/env python
"""Profile where a continuous-batching decode step's time actually goes.

Suspect list, measured directly rather than guessed:
1. Per-token GPU->CPU syncs (.item() calls in the finish-check loop)
2. KV cache concat/slice copies on admission/eviction
3. The executor round-trip itself

Steady-state scenario: 8 active sequences (matching max_batch_size in
the benchmark), warmed up, then profiled over a run of real decode
steps using torch.profiler so the breakdown is by actual operation,
not guesswork.
"""

import time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

from flux.decode import ContinuousBatch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def main():
    model, tokenizer, device = load()
    batch = ContinuousBatch(model, tokenizer, device)

    prompts = [f"Request {i}: tell me something interesting about space" for i in range(8)]
    for i, prompt in enumerate(prompts):
        batch.admit(str(i), prompt, 64)  # long enough that steady state holds for the whole profile

    if device == "cuda":
        torch.cuda.synchronize()

    # Warm up (cuDNN autotune, first-call JIT) before measuring anything.
    for _ in range(5):
        batch.step()
    if device == "cuda":
        torch.cuda.synchronize()

    # --- Coarse wall-clock baseline, no profiler overhead ---
    n_steps = 30
    start = time.perf_counter()
    for _ in range(n_steps):
        batch.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"\nBaseline (no profiler): {elapsed / n_steps * 1000:.2f}ms/step over {n_steps} steps\n")

    # Re-admit to replace anyone who finished during the baseline run, so
    # the profiled section also runs at steady state (8 active).
    while batch.active_count < 8:
        idx = len(prompts)
        prompt = f"Request {idx}: tell me something interesting about space"
        prompts.append(prompt)
        batch.admit(str(idx), prompt, 64)

    # --- torch.profiler: where does the time inside step() actually go? ---
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(n_steps):
            batch.step()
        if device == "cuda":
            torch.cuda.synchronize()

    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    print(f"Top operations by {sort_key} over {n_steps} steps:\n")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))


if __name__ == "__main__":
    main()
