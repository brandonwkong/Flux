"""Correctness test for manual decoding (Step 1 of continuous batching).

Proves generate_one() produces identical text to model.generate() for the
same prompt under greedy decoding - the hand-written token-by-token loop
with past_key_values has to be equivalent to HF's own loop, not just
"close enough." This loads the real model, so it's slow compared to the
queue/scheduler tests, but mocking the model here would defeat the point.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flux.decode import generate_one

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _load():
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


class TestManualDecodingMatchesGenerate:
    def test_matches_hf_generate_on_several_prompts(self):
        model, tokenizer, device = _load()

        prompts = [
            "The capital of France is",
            "Once upon a time, there was a",
            "2 + 2 =",
        ]

        for prompt in prompts:
            manual_text, _ = generate_one(model, tokenizer, prompt, max_new_tokens=16, device=device)
            hf_text = _generate_with_hf(model, tokenizer, device, prompt, max_new_tokens=16)
            assert manual_text == hf_text, f"Mismatch for prompt {prompt!r}: manual={manual_text!r} hf={hf_text!r}"
