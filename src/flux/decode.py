"""Low-level manual decoding: one token at a time, reusing past_key_values.

model.generate() runs an entire generation to completion internally and
can't be interrupted to let another sequence join or a finished one leave -
that's exactly what continuous batching needs to do between every single
token. Hand-rolling the loop here is what makes stopping after any one
step possible.

See docs/continuous_batching_notes.md for the build order. generate_one()
is Step 1 (one sequence). generate_batch_fixed() is Step 2 (several
sequences, fixed membership for the whole generation - no admission or
eviction mid-generation yet, that's Step 3). Both exist to prove each
mechanism is correct - identical output to model.generate() under greedy
decoding - before anything gets built on top of it.
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


def generate_batch_fixed(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens_list: list[int],
    device: str,
) -> list[tuple[str, int]]:
    """Batched decode, fixed membership (Step 2): several sequences run the
    token-by-token loop together, but no one joins or leaves mid-generation -
    that's Step 3. The thing this has to get right is sequences finishing
    (their own max_new_tokens, or EOS) at different steps: a finished
    sequence keeps running through the shared forward pass (wasted but
    harmless compute - each batch row's attention is isolated from every
    other row), it just stops having its tokens recorded.

    Returns (text, output_tokens) per prompt, same convention as the
    single-sequence and static-batched paths.
    """
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    max_new = max(max_new_tokens_list)

    generated: list[list[int]] = [[] for _ in range(batch_size)]
    finished = [False] * batch_size

    with torch.no_grad():
        # Prefill: one forward pass over the whole (padded) batch of prompts.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_ids = outputs.logits[:, -1, :].argmax(dim=-1)  # shape (batch_size,)

        position = prompt_len

        for _ in range(max_new):
            # Decide, per sequence, whether this step's predicted token is
            # real output or the sign that this sequence is done.
            for i in range(batch_size):
                if finished[i]:
                    continue
                token_id = next_token_ids[i].item()
                if token_id == eos_id or len(generated[i]) >= max_new_tokens_list[i]:
                    finished[i] = True
                    continue
                generated[i].append(token_id)

            if all(finished):
                break

            # Finished rows get fed a filler token - it only affects that
            # row's own (already-ignored) future, never anyone else's.
            feed_ids = torch.tensor(
                [[pad_id if finished[i] else next_token_ids[i].item()] for i in range(batch_size)],
                device=device,
            )
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
            cache_position = torch.tensor([position], device=device)

            outputs = model(
                input_ids=feed_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_ids = outputs.logits[:, -1, :].argmax(dim=-1)
            position += 1

    results = []
    for i in range(batch_size):
        text = tokenizer.decode(generated[i], skip_special_tokens=True)
        results.append((text, len(generated[i])))
    return results
