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

## Step 2 - Batched decode, fixed membership

**Concept.** Step 1 proved the single-token mechanic for one sequence.
Step 2 runs several sequences through that same mechanic together, in
one batched step loop - but membership is still *fixed* for the whole
generation, exactly like static batching: everyone starts together, no
one joins or leaves until Step 3. The new problem this step has to
solve: sequences finish (hit their own `max_new_tokens`, or EOS) at
different step counts, but the batch dimension is shared across every
forward call - a finished sequence can't just disappear from the
tensor.

**What was built.** `generate_batch_fixed()` in `decode.py`: prefills a
left-padded batch of prompts together, then loops decode steps across
the whole batch. Each step, per sequence: if it's already finished,
skip it; otherwise check whether this step's predicted token is EOS or
pushes it past its own `max_new_tokens` budget, and if so mark it
finished *without* recording that token. Finished sequences still get
fed through the shared forward pass every subsequent step (with a
throwaway filler token) so the tensor shapes stay valid - this costs
wasted compute but causes no corruption, because each batch row's
attention is computed independently; row i's keys/values are never
mixed with row j's. This is exactly the same trick `model.generate()`
already uses internally for a batch.

Verified in `tests/test_decode.py` against `model.generate()` called on
the *same padded batch* (mirroring the current production
`engine.py._generate()`'s own convention: one `generate()` call at the
batch's largest `max_new_tokens`, each row truncated back to its own
request's budget). Deliberately used staggered `max_new_tokens`
(`[4, 16, 10]`) across three prompts so the finish-at-different-times
code path actually gets exercised, not just coincidentally skipped.
Exact text match on every sequence. `engine.py`/`server.py` still
untouched - the production server still runs the old static-batching
path, confirmed unchanged by booting it and hitting `/generate`.

**Checkpoint explanation (given, then paused for discussion):**

- **How the batched KV cache is laid out.** Add a batch dimension to
  what Step 1 already had: for every layer, the cache is shaped roughly
  `(batch_size, num_heads, sequence_length, head_dim)`. Row `i` of that
  batch dimension holds exactly what `generate_one()` would have built
  for sequence `i` alone - there's no cross-talk between rows. That
  independence is *why* padding a finished sequence with garbage can't
  corrupt its neighbors: each row's attention only ever looks at that
  row's own keys/values.

- **What has to happen when sequences have different lengths.** Two
  different kinds of "different length" show up, and they're handled
  differently. (1) Different *prompt* lengths get left-padded to a
  common width before prefill, same as the existing static path - the
  attention_mask marks the padding so it's excluded from real attention.
  (2) Different *finish times* during decode don't shrink the tensor at
  all - the batch dimension stays fixed size for the whole generation
  (that's the "fixed membership" in this step's name); a finished
  sequence just stops being recorded while still occupying its row,
  fed filler tokens, until Step 3 adds the ability to actually remove
  it from the tensor.
