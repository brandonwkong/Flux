"""Low-level manual decoding: one token at a time, reusing past_key_values.

model.generate() runs an entire generation to completion internally and
can't be interrupted to let another sequence join or a finished one leave -
that's exactly what continuous batching needs to do between every single
token. Hand-rolling the loop here is what makes stopping after any one
step possible.

See docs/continuous_batching_notes.md for the build order. generate_one()
is Step 1 (one sequence). generate_batch_fixed() is Step 2 (several
sequences, fixed membership for the whole generation). ContinuousBatch is
Step 3 - the actual continuous batching: sequences can be admitted and
evicted between every single step. All three exist to prove each
mechanism is correct - identical output to model.generate() under greedy
decoding - before anything gets built on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

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


def _pad_cache_left(cache: Any, pad_amount: int) -> None:
    """Prepend pad_amount all-zero positions to every layer's KV cache, in
    place. Used when a sequence (new or already-active) is shorter than
    the batch's current shared length - the shared cache is one tensor,
    so every row must have the same seq_len dimension."""
    if pad_amount <= 0:
        return
    for layer in cache.layers:
        if not layer.is_initialized:
            continue
        zeros_k = torch.zeros(
            (*layer.keys.shape[:-2], pad_amount, layer.keys.shape[-1]),
            dtype=layer.keys.dtype,
            device=layer.keys.device,
        )
        zeros_v = torch.zeros(
            (*layer.values.shape[:-2], pad_amount, layer.values.shape[-1]),
            dtype=layer.values.dtype,
            device=layer.values.device,
        )
        layer.keys = torch.cat([zeros_k, layer.keys], dim=-2)
        layer.values = torch.cat([zeros_v, layer.values], dim=-2)


def _concat_cache_batch(cache: Any, other: Any) -> None:
    """Concatenate other's rows onto cache along the batch dimension, in
    place. Assumes both already share the same sequence length (pad first)."""
    for layer, other_layer in zip(cache.layers, other.layers):
        layer.keys = torch.cat([layer.keys, other_layer.keys], dim=0)
        layer.values = torch.cat([layer.values, other_layer.values], dim=0)


@dataclass
class _ActiveSeq:
    """One sequence currently occupying a row of the continuous batch.

    position is this row's own logical position for its next token - how
    many real (non-padding) tokens it has processed so far. This has to
    be tracked per-row, separately from the batch's shared physical
    column count: once a sequence's cache gets left-padded after the fact
    (because a longer sequence joined later), its physical column count
    and its real position diverge. Its already-cached keys were rotated
    (RoPE) using its real position at compute time, which is fixed and
    doesn't change just because padding was inserted in front of it -
    so every later step must keep using that same real position, not
    the shared tensor's column count.
    """

    seq_id: Any
    max_new_tokens: int
    position: int
    generated: list[int] = field(default_factory=list)


class ContinuousBatch:
    """Dynamic-membership batch: sequences can be admitted and evicted
    between every single decode step - this is the actual continuous
    batching (Step 3). Deliberately knows nothing about QueuedRequest or
    Futures or the HTTP layer; it just tracks "a set of sequences sharing
    one padded KV cache," identified by whatever id the caller hands in.
    engine.py is what maps ids back to real requests and resolves Futures.

    Implementation note, stated plainly rather than hidden: this uses
    plain padded-tensor batching - every row left-padded to the longest
    active sequence's length - not a paged KV cache like vLLM's
    PagedAttention. That wastes some memory and compute on padding. It's
    a deliberate simplification to keep the KV cache manipulation
    tractable to hand-write and verify for this project; a block-based
    paged cache is how production engines avoid that waste, and is out
    of scope here.
    """

    def __init__(self, model: Any, tokenizer: Any, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.eos_id = tokenizer.eos_token_id

        self._cache: Optional[Any] = None
        self._attention_mask: Optional[torch.Tensor] = None
        self._next_token_ids: Optional[torch.Tensor] = None
        self._active: list[_ActiveSeq] = []
        self._length = 0  # shared sequence length (the cache's seq_len dim)

    @property
    def active_count(self) -> int:
        return len(self._active)

    def admit(self, seq_id: Any, prompt: str, max_new_tokens: int) -> None:
        """Prefill one new sequence and splice it into the batch - padding
        whichever side (the new sequence, or the existing batch) is shorter
        so every row ends up at the same shared length before concatenating."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        new_input_ids = inputs["input_ids"].to(self.device)
        new_attention_mask = inputs["attention_mask"].to(self.device)
        new_len = new_input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(input_ids=new_input_ids, attention_mask=new_attention_mask, use_cache=True)
        new_cache = outputs.past_key_values
        new_next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)  # shape (1,)

        new_seq = _ActiveSeq(seq_id=seq_id, max_new_tokens=max_new_tokens, position=new_len)

        if self._cache is None:
            self._cache = new_cache
            self._attention_mask = new_attention_mask
            self._next_token_ids = new_next_token_id
            self._active = [new_seq]
            self._length = new_len
            return

        common_len = max(self._length, new_len)

        if self._length < common_len:
            pad = common_len - self._length
            _pad_cache_left(self._cache, pad)
            self._attention_mask = torch.nn.functional.pad(self._attention_mask, (pad, 0), value=0)
            self._length = common_len

        if new_len < common_len:
            pad = common_len - new_len
            _pad_cache_left(new_cache, pad)
            new_attention_mask = torch.nn.functional.pad(new_attention_mask, (pad, 0), value=0)

        _concat_cache_batch(self._cache, new_cache)
        self._attention_mask = torch.cat([self._attention_mask, new_attention_mask], dim=0)
        self._next_token_ids = torch.cat([self._next_token_ids, new_next_token_id], dim=0)
        self._active.append(new_seq)

    def step(self) -> list[tuple[Any, str, int]]:
        """One iteration: evaluate every active sequence's last-predicted
        token (from admission's prefill, or the previous step's decode),
        evict whoever just finished, then run one decode step for whoever
        is left. Returns (seq_id, text, output_tokens) for each sequence
        that finished this step."""
        finished_results: list[tuple[Any, str, int]] = []
        keep_indices: list[int] = []

        for i, seq in enumerate(self._active):
            token_id = self._next_token_ids[i].item()
            if token_id == self.eos_id or len(seq.generated) >= seq.max_new_tokens:
                text = self.tokenizer.decode(seq.generated, skip_special_tokens=True)
                finished_results.append((seq.seq_id, text, len(seq.generated)))
            else:
                seq.generated.append(token_id)
                keep_indices.append(i)

        if len(keep_indices) != len(self._active):
            if not keep_indices:
                self._cache = None
                self._attention_mask = None
                self._next_token_ids = None
                self._active = []
                self._length = 0
                return finished_results

            keep_tensor = torch.tensor(keep_indices, device=self.device)
            self._cache.batch_select_indices(keep_tensor)
            self._attention_mask = self._attention_mask[keep_tensor]
            self._next_token_ids = self._next_token_ids[keep_tensor]
            self._active = [self._active[i] for i in keep_indices]

        if not self._active:
            return finished_results

        batch_size = len(self._active)
        feed_ids = self._next_token_ids.unsqueeze(-1)
        self._attention_mask = torch.cat(
            [
                self._attention_mask,
                torch.ones((batch_size, 1), dtype=self._attention_mask.dtype, device=self.device),
            ],
            dim=1,
        )
        # cache_position is shared (every row's tensor grows by one column
        # together); position_ids is per-row (each row's own real position,
        # ignoring however much left-padding it happens to have right now) -
        # see _ActiveSeq's docstring for why these must NOT be the same value.
        cache_position = torch.tensor([self._length], device=self.device)
        position_ids = torch.tensor([[seq.position] for seq in self._active], device=self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=feed_ids,
                attention_mask=self._attention_mask,
                past_key_values=self._cache,
                cache_position=cache_position,
                position_ids=position_ids,
                use_cache=True,
            )
        self._cache = outputs.past_key_values
        self._next_token_ids = outputs.logits[:, -1, :].argmax(dim=-1)
        for seq in self._active:
            seq.position += 1
        self._length += 1

        return finished_results
