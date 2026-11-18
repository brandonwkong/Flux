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

## Step 3 - Dynamic membership (the actual continuous batching)

**Concept.** Steps 1-2 proved the mechanics when membership is decided
once, up front. Step 3 makes membership dynamic: between every single
step, the engine can admit new requests into free slots and evict
anyone who just finished - this is the whole point of the migration.
Each iteration now does three things: admit (prefill a newly-admitted
request, splice its KV cache into the batch), run one decode step for
everyone active, evict (resolve finished requests' Futures immediately,
slice them out of the cache). The scheduler's job changes shape:
`select_batch` used to pick a whole batch once per generation; now it
decides admission order every iteration, for however many slots are
free (often zero) - same three policies, called far more often, each
call much smaller-stakes.

**What was built.**

1. `ContinuousBatch` in `decode.py` - the low-level mechanics
   (`admit()`, `step()`), deliberately with no knowledge of
   `QueuedRequest` or `Future`. Uses plain padded-tensor batching: every
   row left-padded to the longest active sequence's length, not a paged
   KV cache like vLLM's PagedAttention. That's a real, stated
   simplification - it wastes some memory/compute on padding - made to
   keep the KV cache manipulation tractable to hand-write and verify by
   hand; block-based paging is how production engines avoid that waste
   and is out of scope here.
2. `engine.py` rewritten around it: `_run_loop()` admits-then-steps every
   iteration, `_resolve_finished()` resolves each sequence's Future the
   moment it finishes. `max_wait_ms` is gone from `Engine`'s constructor
   and `server.py`'s config - **flagging this as the one forced config
   change**: it meant "wait for stragglers before running a batch,"
   which has no meaning once admission happens every iteration. The
   `/generate` HTTP contract is unchanged.

**Two real bugs found and fixed while building this** - worth recording
because both are the kind of thing that only shows up once membership
is actually dynamic, not before:

- **Position encoding mismatch.** First version of the correctness test
  failed: three sequences admitted/evicted at staggered times produced
  different text than running each alone, even though Steps 1-2's exact
  same mechanics (minus dynamic admission) had passed cleanly. Root
  cause: when sequence A's cache gets left-padded *after* some of its
  tokens are already cached (because a longer sequence B joins later),
  A's already-cached keys were rotated (RoPE) using A's real position
  at the time they were computed - that's permanent, baked in, and
  doesn't change just because padding was inserted in front of it
  afterward. But the code was deriving every row's position implicitly
  from the shared physical column count (`cache_position`), which
  changes when padding changes. Fix: track each active sequence's own
  logical position explicitly (ignoring its padding) and pass it as a
  separate `position_ids` argument, distinct from the shared
  `cache_position` used only for indexing into the cache tensor. Exact
  match against solo generation after the fix.
- **Queue time measured after the work it was supposed to exclude.**
  Smoke-testing the live server showed `queue_time_ms` of ~760ms for a
  single request on an otherwise idle server - should have been ~0.
  Cause: the admission timestamp was recorded *after* `batch.admit()`
  (the blocking prefill call) returned, not before - so any time spent
  in that first, possibly cold-cache prefill call was silently counted
  as "waiting in queue" instead of "doing the work." Fix: timestamp
  before the blocking call, not after. Confirmed fixed: `queue_time_ms`
  reads 0.0 on an idle server afterward, with the cold-start cost now
  correctly showing up in `compute_time_ms` instead.

**Live proof it actually works** (not just passing tests): fired a
64-token request (A), then 300ms later, mid-A's-generation, an 8-token
request (B). B's `queue_time_ms` was 58.6ms - admitted within about one
loop iteration, not after waiting for A's whole ~2.9s generation - and
B finished in 512ms total while A kept running independently
(`batch_size=2` confirms they were genuinely sharing steps). That's the
exact failure mode static batching had and continuous batching doesn't.

**Checkpoint explanation (given, then paused for discussion):**

- **One full step, in plain language.** At the top of every iteration:
  (1) *Admit* - if there's free capacity, ask the scheduler which
  pending requests should fill it (same FIFO/priority/EDF logic as
  before, just choosing among whoever's waiting right now instead of
  the whole next batch); prefill each one and splice it into the active
  set. (2) *Run* - one decode step, together, for every sequence
  currently active. (3) *Evict* - anyone whose last predicted token was
  EOS or hit their own `max_new_tokens`: decode their text, resolve
  their Future immediately, remove their row from the cache. Repeat.
- **The policy's job changed shape, not its logic.** Under static
  batching, `select_batch` ran once per *generation*, picking a whole
  batch (up to 8) from everyone pending, and that batch was locked for
  however long the slowest sequence in it took. Under continuous
  batching, the exact same function runs every single *iteration*,
  picking however many requests fit in whatever capacity just opened up
  (often zero, sometimes one, rarely several) - a much higher-frequency,
  lower-stakes decision. FIFO/priority/EDF didn't need to change at
  all; what changed is how often, and over how much, they get asked.

## Step 4 - Timing semantics

**Concept.** `queue_time_ms` and `compute_time_ms` already took on their
new per-sequence meanings as a direct consequence of Step 3's design
(arrival->admission, admission->finish) - nothing left to decide there.
The one open question was `batch_size`: under static batching it was
trivially well-defined (the whole batch was fixed for the entire
request), but continuous batching has no single "the batch" a request
belongs to - it can share steps with a different, changing set of
sequences throughout its life.

**The decision.** Three options were on the table: a snapshot at
admission time, a lifetime peak, or a lifetime average. Chose the
**average active-sequence count across every step the request's
generation participated in** - the number that answers "how batched was
this request's compute, overall," which is the closest continuation of
what the field always meant.

**What was built.** `ContinuousBatch` now accumulates, per active
sequence, a running sum of `len(active)` sampled at the top of every
`step()` call, plus a sample count; on finish, `avg_active_count =
sum / samples`. `engine.py` rounds this to an int for the
`GenerateResponse.batch_size` field - same field name, same type, so
the benchmark script needs no changes.

**Verified on the live server**, warm: a 64-token request (A) run with
an 8-token request (B) joining 300ms in reported `batch_size=1` for A
and `batch_size=2` for B. Checked the arithmetic by hand: A spent ~56
of its 64 steps alone and ~8 shared with B, averaging to
(56x1 + 8x2)/64 = 1.125, which rounds to 1 - matches exactly.

## Post-Step-4 investigation: why was continuous batching slower in the final sweep?

The static-vs-continuous comparison (both at the same trimmed config)
showed continuous batching with *higher* `compute_time_ms` in most
cells - 25-65% higher in the contested ones - which ate the queueing
win. Before accepting "continuous batching lost" as the real answer,
profiled where a decode step's time actually goes, in one session,
rather than guessing.

**Suspect list, each measured directly, each ruled out:**

1. *Per-token GPU->CPU syncs* (`.item()` in the finish-check loop).
   Patched to one bulk `.tolist()` transfer instead of N per-sequence
   `.item()` calls. Result: 30.2ms/step before, 31.1ms/step after - no
   measurable difference.
2. *KV cache concat/slice on admission/eviction*. Measured pure decode
   (no admit/evict happening) at 28.4ms/step vs the full loop with
   admission folded in at 30.2ms/step - the cost is real but tiny and
   one-time-per-request, not a recurring per-token tax.
3. *The executor round-trip itself*. `step()` called directly: ~30ms.
   Same call via `await loop.run_in_executor(...)`: 31.5ms. ~1.5ms of
   genuine dispatch overhead, not enough to explain a 25-65% gap.

**Two more checks, beyond the original list:**

- `step()`'s cost is flat across batch size 1 through 8 (26.8-32.6ms
  regardless), and flat across sequence length 36 through 136 tokens.
  Neither scales the way "GPU compute" or "growing tensor copy" would
  predict - this cost looks like fixed per-call overhead (Python
  dispatch + CUDA kernel launch across 22 transformer layers), not
  anything specific to this implementation.
- Compared directly against `model.generate()`'s own internal per-token
  cost on the identical batch: **32.16ms/token** - statistically
  indistinguishable from the hand-rolled loop's ~27-32ms/step. The
  hand-written decode loop is not inherently slower than HF's own
  internal loop.

**Then the real surprise**: re-ran the same single-request scenario
through the actual `Engine` class directly (no HTTP) and got
28-32ms/token - then through the full running server over real HTTP
and got 28-40ms/token. Neither reproduced the original sweep's
consistent 44-46ms/token at concurrency=1. Checked `nvidia-smi`: the
same 4-5 background processes were present in both cases, but GPU
memory usage was lower during this diagnostic session (153MB) than
right after the original sweep finished (266MB) - weak but suggestive
evidence that the gap is more consistent with **GPU contention varying
over the ~45-minute sweep** than with anything fixable in this code.

**Conclusion, stated plainly: no code-level fix is justified by this
evidence.** All three originally-suspected mechanisms were measured
and ruled out individually. The decode primitive itself costs the same
whether hand-rolled or run through `generate()`. The most likely
explanation for the benchmark gap is environmental (background GPU
load fluctuating across a long run), which is a measurement problem,
not an engineering one - the real fix is a cleaner GPU and/or
back-to-back (not sequential) static/continuous runs to control for
contention drift, not a change to `decode.py` or `engine.py`.

Profiling script kept at `scripts/profile_decode_step.py` for anyone
who wants to re-run this investigation under cleaner conditions.
