# Continuous batching: build notes

Running notes for the static-batching -> continuous-batching migration.
One section per step: the concept, the decision made, and the checkpoint
explanation. Written to review later, not as user-facing documentation.

## Step 1 - Manual generation, single sequence

**Concept.** `model.generate()` is a closed box: hand it a prompt and a
token budget, it loops internally until done, hands back the whole
result. Nothing else can get in or out while it's running - which is
exactly the problem, since continuous batching needs to decide, between
*every single token*, whether a new request gets admitted or a finished
one leaves. The fix is to stop using `generate()` for the real engine
loop and hand-write the loop ourselves: one forward pass per token,
pausable after every single one.

**What was built.** `src/flux/decode.py`, `generate_one()`: a single-
sequence, greedy, token-by-token loop using `past_key_values`. Verified
against `model.generate()` in `tests/test_decode.py` - exact text match
on three different prompts, not just "looks similar." Nothing in
`engine.py` or `server.py` changed; the production server still runs
the old static-batching path unchanged, confirmed by booting it and
hitting `/generate` after this step.

**Checkpoint explanation (given, then paused for discussion):**

- **What `past_key_values` contains.** For every attention layer, and
  for every token position the model has already processed, it's the
  cached key and value vectors. Attention scores are `query · key` over
  every past position; those keys/values don't change once computed,
  because what the model "knows" about tokens 0..k never changes just
  because token k+1 exists. The cache is exactly "the keys and values
  from every forward pass so far," handed back to be reused.

- **Why feeding only the last token gives the same result.** Attention
  needs the *new* token's query vector against *every* position's keys
  and values - including all the old ones. If those old keys/values are
  already sitting in the cache, re-computing them by re-feeding the
  whole sequence would recompute numbers that are already known and
  unchanged. Feeding just the newest token computes only what's new
  (that token's own key/value, plus attention against the full cache)
  and skips redundant work - same math, much less of it.

- **Prefill vs decode in this code.** Prefill is the first call in
  `generate_one()`: one forward pass over the *entire* prompt at once,
  building the cache for every prompt position in one shot - this is
  the parallelizable, GPU-efficient pass. Decode is every call after:
  `input_ids` of shape `(1, 1)`, just the newest token, relying on
  `past_key_values` for everything before it. This split - one expensive
  parallel pass, then many cheap sequential single-token passes - is
  the shape every later step builds on.
