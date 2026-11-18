"""Correctness tests for manual decoding (Steps 1 and 2 of continuous batching).

Proves the hand-written token-by-token loops produce identical text to
model.generate() under greedy decoding - not just "close enough." This
loads the real model, so it's slow compared to the queue/scheduler tests,
but mocking the model here would defeat the point.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flux.decode import ContinuousBatch, generate_batch_fixed, generate_one

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def loaded_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _generate_with_hf(model, tokenizer, device, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )

    new_tokens = generated[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _generate_batch_with_hf(model, tokenizer, device, prompts, max_new_tokens_list):
    """Mirrors engine.py's current production _generate(): one generate()
    call at the batch's largest max_new_tokens, then each row truncated
    back down to its own request's max_new_tokens."""
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    input_len = input_ids.shape[1]
    max_new = max(max_new_tokens_list)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )

    texts = []
    for i, max_new_tokens in enumerate(max_new_tokens_list):
        new_tokens = generated[i, input_len : input_len + max_new_tokens]
        texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return texts


class TestManualDecodingMatchesGenerate:
    """Step 1: one sequence at a time."""

    def test_matches_hf_generate_on_several_prompts(self, loaded_model):
        model, tokenizer, device = loaded_model

        prompts = [
            "The capital of France is",
            "Once upon a time, there was a",
            "2 + 2 =",
        ]

        for prompt in prompts:
            manual_text, _ = generate_one(model, tokenizer, prompt, max_new_tokens=16, device=device)
            hf_text = _generate_with_hf(model, tokenizer, device, prompt, max_new_tokens=16)
            assert manual_text == hf_text, f"Mismatch for prompt {prompt!r}: manual={manual_text!r} hf={hf_text!r}"


class TestBatchedDecodingMatchesGenerate:
    """Step 2: several sequences, fixed membership, staggered finish times."""

    def test_matches_hf_generate_with_different_max_new_tokens_per_sequence(self, loaded_model):
        """Different max_new_tokens per sequence means they finish at
        different steps - exactly the case Step 2 has to get right without
        corrupting the sequences that keep running."""
        model, tokenizer, device = loaded_model

        prompts = [
            "The capital of France is",
            "Once upon a time, there was a",
            "2 + 2 =",
        ]
        max_new_tokens_list = [4, 16, 10]  # deliberately staggered

        manual_results = generate_batch_fixed(model, tokenizer, prompts, max_new_tokens_list, device)
        hf_texts = _generate_batch_with_hf(model, tokenizer, device, prompts, max_new_tokens_list)

        for i, (prompt, (manual_text, _), hf_text) in enumerate(zip(prompts, manual_results, hf_texts)):
            assert manual_text == hf_text, (
                f"Mismatch for prompt {prompt!r} (max_new_tokens={max_new_tokens_list[i]}): "
                f"manual={manual_text!r} hf={hf_text!r}"
            )

    def test_output_tokens_matches_each_sequences_own_budget(self, loaded_model):
        """Each sequence's output_tokens should reflect its own finish
        point, not get confused by batchmates finishing earlier or later."""
        model, tokenizer, device = loaded_model

        prompts = ["2 + 2 =", "The capital of France is"]
        max_new_tokens_list = [3, 12]

        results = generate_batch_fixed(model, tokenizer, prompts, max_new_tokens_list, device)

        for (_, output_tokens), max_new_tokens in zip(results, max_new_tokens_list):
            assert output_tokens <= max_new_tokens


class TestContinuousBatchMatchesGenerateOne:
    """Step 3: dynamic membership - the actual continuous batching.

    The invariant that matters here isn't "matches model.generate()" (no
    single generate() call does what this does) - it's that admission and
    eviction TIMING must never change WHAT a sequence generates, only WHEN
    it finishes relative to the others. So the ground truth is Step 1's
    generate_one(), already proven correct, run on each prompt alone.
    """

    def test_sequences_admitted_and_evicted_at_different_times_match_solo_generation(self, loaded_model):
        model, tokenizer, device = loaded_model

        # Staggered on purpose: different prompt lengths (so admission has
        # to pad in both directions) and different max_new_tokens (so
        # eviction happens at different steps for different rows).
        requests = {
            "A": ("The capital of France is", 8),
            "B": ("Once upon a time, there was a brave knight who", 12),
            "C": ("2 + 2 =", 6),
        }

        expected = {
            seq_id: generate_one(model, tokenizer, prompt, max_new_tokens, device)
            for seq_id, (prompt, max_new_tokens) in requests.items()
        }

        batch = ContinuousBatch(model, tokenizer, device)
        finished: dict[str, tuple[str, int]] = {}

        # Admit A alone, step twice so it's mid-generation.
        batch.admit("A", *requests["A"])
        for _ in range(2):
            for seq_id, text, output_tokens, _ in batch.step():
                finished[seq_id] = (text, output_tokens)

        # Admit B while A is still running (tests padding the new, longer
        # prompt against an existing shorter batch and vice versa).
        batch.admit("B", *requests["B"])
        for _ in range(2):
            for seq_id, text, output_tokens, _ in batch.step():
                finished[seq_id] = (text, output_tokens)

        # Admit C while A and/or B may still be running.
        batch.admit("C", *requests["C"])

        # Run until everyone still active has finished.
        while batch.active_count > 0:
            for seq_id, text, output_tokens, _ in batch.step():
                finished[seq_id] = (text, output_tokens)

        assert set(finished.keys()) == set(requests.keys())
        for seq_id in requests:
            expected_text, expected_tokens = expected[seq_id]
            actual_text, actual_tokens = finished[seq_id]
            assert actual_text == expected_text, (
                f"Sequence {seq_id!r} text mismatch: "
                f"continuous={actual_text!r} solo={expected_text!r}"
            )
            assert actual_tokens == expected_tokens
