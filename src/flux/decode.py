"""Low-level manual decoding: one token at a time, reusing past_key_values.

model.generate() runs an entire generation to completion internally and
can't be interrupted to let another sequence join or a finished one leave -
that's exactly what continuous batching needs to do between every single
token. Hand-rolling the loop here is what makes stopping after any one
step possible.

Step 1 scope (see docs/continuous_batching_notes.md): ONE sequence, no
batching (Step 2) and no admission/eviction mid-generation (Step 3). This
file exists to prove the token-by-token mechanism is correct - identical
output to model.generate() under greedy decoding - before anything gets
built on top of it.
"""

from __future__ import annotations

import torch


def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> tuple[str, int]:
    """Generate greedily, one token at a time, using past_key_values.

    Returns (text, output_tokens) - same convention as the batched path
    in engine.py: output_tokens counts real generated tokens up to (not
    including) EOS.
    """
    eos_id = tokenizer.eos_token_id

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    generated_ids: list[int] = []

    with torch.no_grad():
        # Prefill: one forward pass over the whole prompt. Builds the KV
        # cache for every prompt position in a single shot - this is the
        # expensive, parallelizable pass.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()

        position = prompt_len  # position index of the next token to generate

        for _ in range(max_new_tokens):
            if next_token_id == eos_id:
                break
            generated_ids.append(next_token_id)

            # Decode step: feed ONLY the newest token. past_key_values
            # already holds keys/values for every earlier position, so the
            # model never re-reads the rest of the sequence - this is the
            # one-token-at-a-time step that makes continuous batching
            # possible, versus generate()'s uninterruptible internal loop.
            next_input_ids = torch.tensor([[next_token_id]], device=device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
            cache_position = torch.tensor([position], device=device)

            outputs = model(
                input_ids=next_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()
            position += 1

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, len(generated_ids)
