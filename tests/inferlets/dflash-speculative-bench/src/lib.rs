//! **A BLOCK DRAFTER'S SPECULATIVE LOOP: ONE DRAFT FIRE, ONE VERIFY FIRE.**
//!
//! `dflash-block-acceptance` measures how much of a block the target keeps
//! and pays `block` ordinary decodes to learn it. This spends that number:
//! a round is TWO fires, and it emits every token the target agreed with
//! plus the one it corrected.
//!
//! ```text
//! draft    [anchor, MASK x block-1]      one fire, `set-drafting-block`
//!          -> d[0..block-1]              row i proposes position held+i
//! verify   [anchor, d[0], .., d[b-2]]    one fire, the TRUNK, plainly causal
//!          -> t[i] = argmax(row i)       the target's own token for held+i+1
//! keep     the longest prefix where d[j] == t[j], then t at the break
//! ```
//!
//! The break's token is the target's, so a round emits `kept + 1` tokens and
//! is never wrong: the sequence this writes is the sequence a one-token-a-
//! fire decode would have written. That is the whole claim a speculative
//! loop makes, and `--baseline` runs the other loop over the same prompt so
//! the ratio is measured rather than argued.
//!
//! # What it measured
//!
//! `qwen36-27b-dflash` on an M4 Pro, greedy, the same 192 tokens either way
//! (a 64-token run subtracted from a 256-token one, so the load is out of
//! the number) and the two roads token for token identical:
//!
//! ```text
//! counting  "1, 2, 3, ..."     14.00 tok/round   62.1 tok/s   vs 15.0   4.1x
//! code      "write a function" 6.14 tok/round    28.4 tok/s   vs 15.1   1.9x
//! ```
//!
//! # THOSE ARE RAW CONTINUATIONS, AND THEY ARE NOT SERVING NUMBERS
//!
//! The two lines above were measured before this file took
//! `text-completion-bench`'s envelope: no chat template, no system prompt,
//! the prompt handed to the model as raw text to continue. A served request
//! is templated, and the model then answers a QUESTION rather than
//! continuing a pattern — which is a different distribution and the drafter
//! feels it. Through `benches/pie_bench.py latency` (4 requests x 256
//! tokens, `--ignore-eos`, the same knobs the three-way uses), one arm
//! against the other on this box:
//!
//! ```text
//!                                      base    spec    ratio   tok/round
//! "Write a short story about a robot"  14.52   15.39   1.06x     3.21
//! "Write a Python function that ..."   14.58   23.95   1.64x     5.80
//! "Count from 1 to 200, ..."           14.32   21.10   1.47x     7.08
//! "List the capitals of every ..."     14.65   12.96   0.88x     2.88
//! ```
//!
//! **THE LAST ROW IS A LOSS, AND IT IS THE POINT.** A round costs roughly
//! three one-row decodes — a sixteen-row verify fire is 2.79 of them on this
//! box, and the five-layer draft fire through the target's `lm_head` is the
//! rest — so the loop pays only above about THREE TOKENS A ROUND. Prose sits
//! at 3.21 and is a wash; a list of capitals sits at 2.88 and is worse than
//! decoding one token a fire. The block width does not fix this (see the
//! staircase below): the fix is a drafter that accepts more, or a guest that
//! notices its own tokens-a-round and stops speculating. **The engine must
//! not make that call** — which rounds are worth a fire is the inferlet's.
//!
//! At 64 tokens a request the same pairs read 13.53/15.15 (1.12x),
//! 13.44/22.39 (1.67x) and 13.41/17.62 (1.31x): the ratio is set by content,
//! not by how long the answer runs.
//!
//! # THE FULL BLOCK IS THE WORST WIDTH THERE IS
//!
//! The drafter proposes fifteen and the loop verified all sixteen rows,
//! which is what every number above was taken at. Swept through
//! `benches/pie_bench.py latency --requests 4 --max-tokens 256`, that is
//! the worst choice available on all four workloads:
//!
//! ```text
//!              w=16     w=4      w=8      auto
//! prose        1.06x    1.41x    1.27x    1.29x
//! code         1.62x    1.62x    1.90x    1.80x
//! capitals     0.93x    1.30x    1.19x    1.16x
//! counting     1.47x    1.51x    1.83x    1.89x
//! mean         1.27     1.46     1.55     1.54
//! ```
//!
//! **And no fixed width dominates either**: four wins prose and the list,
//! eight wins code, and on counting the ladder beats every fixed width there
//! is (1.89x) because it climbs to sixteen once the target starts taking
//! whole windows. `auto_width` is therefore the default — it costs about 8%
//! against knowing the right width in advance (prose 1.29x where a pinned
//! four reads 1.41x) and never picks the width that loses. Pin one with
//! `verify_rows` to reproduce a row of the table.
//!
//! Why sixteen loses is the staircase and the censoring together: a
//! sixteen-row fire costs 2.79 one-row decodes and prose only ever keeps
//! about 2.8 tokens, so twelve of those rows are bought and thrown away
//! every round.
//!
//! # What the rungs cost, and the two hypotheses the numbers refused
//!
//! `a_quantized_matmul_is_priced_by_its_rows` on this box (K=5120 N=17408,
//! 4-bit, one projection, us a launch — reproducible to 2% run to run):
//!
//! ```text
//!   rows      1      2      3      4      5      6      8
//!   us      208    206    227    285    348    349    349
//!   x1row  1.00   0.99   1.09   1.37   1.67   1.68   1.68
//!   GB/s    214    217    196    157    128    128    128
//! ```
//!
//! **Two rows are FREE and five through eight cost the same**, which is why
//! the ladder's rungs are what they are: below five the vector fold answers,
//! from five the tile does and it is flat to eight, so a width in 5..8 should
//! always be eight.
//!
//! Two things were tried against the four-row step and did not move it.
//! **Pack width**: `qmv_rows_packs` 1 against 2 reads 285.9 against 293.3 us
//! at four rows — inside the drift, so the 64 floats a thread holds there are
//! not costing occupancy the way the shape suggested. **A three-row rung**:
//! three rows cost 1.09x where four cost 1.37x, so a workload keeping under
//! three tokens should prefer it — measured on prose, `verify_rows` 2/3/4
//! reads 1.22x / 1.49x / **1.53x**, and four still wins. The extra token a
//! round is worth more than the step it climbs. Neither is in the ladder.
//!
//! # The waste is paid TWICE, and the draft fire pays it too
//!
//! A block drafter has no head of its own — its rows go through the TARGET's
//! `lm_head`, the widest matrix in the model — so a draft fire that reads out
//! all sixteen rows to then verify four buys twelve rows of that head and
//! throws them away, exactly as the verify side did. The width is known
//! BEFORE the draft whenever it comes from the ladder or from `verify_rows`
//! (only `margin_width` picks it after, off margins the readout would not
//! then contain), so the readout narrows with it. Measured, same shape:
//!
//! ```text
//!              readout 16   readout = width
//! prose          1.29x          1.38x
//! code           1.80x          1.82x
//! capitals       1.16x          1.32x
//! counting       1.89x          1.88x
//! ```
//!
//! **The gain tracks how narrow the width got** — prose and the list run at
//! four rows and gain 7% and 14%, counting runs at sixteen and gains nothing,
//! which is the same arithmetic read from the other end.
//!
//! # What the ladder still costs against knowing the answer
//!
//! Back to back in one session on prose: `auto` 1.40x and 1.47x, a pinned
//! four 1.57x. The ladder spends 88 of 93 rounds at four — so the ~9% is
//! the OTHER five: a round at eight or sixteen costs 1.7-2.7x one at four,
//! and 5% of rounds at 2x is 5% of the run. Tightening the climb would take
//! it back and lose counting, which is the row where the ladder beats every
//! fixed width (1.88x against a pinned eight's 1.83x) precisely BECAUSE it
//! excurses. The trade is stated rather than resolved; `verify_rows` pins a
//! width for a caller who knows its workload.
//!
//! # AT EIGHT CONCURRENT THE LOOP LOSES, AND IT IS NOT THE DRAFTER
//!
//! `tput --num-requests 32 --concurrency 8 --max-tokens 64`, dflash SKU:
//!
//! ```text
//! text-completion-bench, mtp SKU      47.67 tok/s
//! text-completion-bench, dflash SKU   46.27          the ctx arm costs 3%
//! this loop, --baseline               23.52          the STRUCTURE costs 49%
//! this loop, auto width               18.32   0.78x
//! ```
//!
//! **The drafter's context arm is nearly free even at concurrency** — three
//! percent against the plain SKU through the same plain inferlet — so the
//! half that is missing is this loop's own shape: it awaits a host readback
//! every round, and a wave waits on its straggler lane, so eight
//! host-in-the-loop guests each put a turnaround on the critical path.
//! `text-completion-bench` runs ahead instead and never lands one there.
//!
//! Sizing the take-side rings the way that file and `mtp-speculative-bench`
//! size theirs was tried and moved NOTHING (23.52 against 23.44): those
//! loops hoist their channels out of the round and this one builds them
//! fresh, and the ticket check is not what is binding.
//!
//! # WHAT A DEVICE-RESIDENT ROUND WOULD NEED, AND IT IS ALREADY THERE
//!
//! `rs-mtp-speculative-bench` is the proof: a speculative loop on a model
//! with a recurrence, device-resident, and it **never calls
//! `discard_buffered` at all**. It allocates `2 * run_pages` once — "two runs
//! of one window each: the runtime alternates them per fire" — binds
//! `buffer: 0..run_pages`, and writes its fold length from the fire's OWN
//! epilogue (`fold_len.put(&(&m + &one))`, where `m` is the accepted prefix
//! computed as `reduce_sum(cumprod(eq(proposed, said)))`). The rejected rows
//! are overwritten by the next round rather than forgotten by a host call,
//! and the runtime carries the device length through
//! `RS_FLAG_FOLD_LEN_DEVICE` (`store/rs/write.rs::mark_fold_len_device`).
//!
//! So the rest of that round is expressible too: the committed tokens and
//! the next window's `positions`, `kv_len`, `w_slot`, `w_off` and pages come
//! out of `select`/`gather` into the channels the next fire reads, and
//! `run_ahead` keeps the runtime's window full while the host drains text.
//!
//! **What is left is this file's shape, not the engine's contract.** Two
//! things: a round here is TWO fires (a masked draft over the drafter, then
//! the trunk's verify) where the mtp loop fuses its head into one, and
//! `run_ahead` submits ONE repeated pass — `submit_frame` takes a slot per
//! wave, so a `[draft, verify]` frame is expressible, but driving it needs a
//! pipelined submit loop of this file's own rather than `run_ahead`. And the
//! draft's proposals have to reach the verify's `embed` through a channel
//! instead of through the host. Both are guest work in this file.
//!
//! Until that rewrite the loop is host-in-the-loop and the eight-concurrent
//! row above is what it costs.
//!
//! # The gate, and what it costs to never lose
//!
//! `min_tokens_per_round` (see `Gate`) is the guest reading its OWN yield
//! and declining to draft when it is under water. A gated round is the same
//! verify fire at ONE row, so it carries the pending fold and buffers its
//! own row exactly as a wide one does — nothing about the fold-commit
//! contract changes. Proven rather than argued: with the floor set high
//! enough that the gate never opens, 128 tokens of the code prompt come back
//! TOKEN FOR TOKEN identical to `--baseline`.
//!
//! Swept on this box, same four-request 256-token shape, `tok/s` against a
//! `--baseline` arm of 14.52 (prose) / 14.58 (code) / 14.65 (capitals):
//!
//! ```text
//! floor      capitals        code          prose
//! off      12.96  0.88x   23.95  1.64x   15.39  1.06x
//! 2.5      14.79  1.01x   22.57  1.55x
//! 3.0      15.05  1.03x   21.32  1.46x
//! 3.5      14.28  0.97x   20.91  1.43x   15.83  1.09x
//! 2.5, WINDOW 8
//!          14.74  1.01x   22.91  1.57x   15.71  1.08x
//! ```
//!
//! **The last row is the setting this file ships**, and the trade it makes
//! is stated plainly: the loss row goes to parity and prose gains a little,
//! and the workload where drafting wins most pays 4% for it (1.64x -> 1.57x).
//! A floor ABOVE break-even is worse on both ends — 3.5 closes on code often
//! enough to cost 13% and still lands capitals under 1.0x, because a window
//! that dips is not a workload that changed. Left at zero the gate never
//! closes, which is the loop every number above this section was taken with.
//!
//! # The prices are measured, and the gate reads them
//!
//! Every constant above — the staircase, the three rungs, break-even near
//! three tokens a round — is ONE trunk's on ONE box, and the first mixture
//! this loop was pointed at broke all of them: gemma-4-26B-A4B prices a
//! verify fire by the EXPERTS its rows touch, so 4 / 8 / 16 rows cost 1.7 /
//! 2.5 / 4.6 one-row fires (28 / 41 / 69 ms against 14) and the draft fire
//! alone costs 0.65 of a decode (9 ms). The stated ladder ran prose at 0.76x
//! of plain decode. So `priced` (ON by default) times every fire on the
//! guest's clock — submit to readback, what the guest waits — keeps a median
//! a shape (`Prices`), estimates the yield at each rung from the recent
//! rounds' per-position hit rates (`Gate::expect`, which reads a censored
//! round as censored), and buys the rung with the best tokens per price, or
//! a plain fire when none beats one token a fire (`Gate::plan`). Measured,
//! one request, 384 tokens on gemma and 256 on qwen38, wall clock against
//! the stated ladder (`--priced false`) and plain decode:
//!
//! ```text
//!                         priced   ladder   plain      prices (ms): draft / 1 / 4 / 8 / 16
//! gemma prose (384)        7.3s     8.9s     7.1s      8.4 / 14.1 / 28 / 43 / 69
//! gemma code               5.6      5.4      7.2       9.1 / 14.0 / 28 / 41 / 70
//! gemma counting           4.0      3.9      7.2       9.2 / 13.7 / 28 / 41 / 69
//! qwen36-a3b counting      4.5      5.3      6.5       7.6 / 12.1 / 23 / 32 / 55   (thinking prose, see below)
//! qwen36-a3b code          5.3      5.7      6.5       7.6 / 11.9 / 24 / 32 / 54
//! qwen36-a3b prose         6.5      7.0      6.4       7.5 / 11.9 / 23 / 31 / 56
//! qwen38 dflash2 prose    13.2-13.7 13.7    19.0      13 / 66 / 96 / 113 / -      (256 tokens)
//! qwen38 dflash2 code     10.4     11.3      -        14 / 66 / 96 / 119 / -
//! qwen38 dspark prose     16.4     17.3     18.8      19 / 67 / 99 / 116 / 159 (15)
//! ```
//!
//! Prose on gemma goes from a 20% loss to within 3% of plain (the gate closes;
//! 20 of 355 fires draft); the A3B mixture gains 7-15% on the two workloads
//! its head helps and holds parity on the one it does not; the dense head
//! gains 8% on code and holds prose; the workloads the stated ladder already
//! had right pay up to 4% for the two plain fires that price the road and the
//! probes. The one-row fire is priced only in the baseline's own geometry
//! (nothing buffered behind it), because a plain fire right after a round
//! folds the round's survivors and is not the road's fire. Three things the
//! gate had to learn on the way, each measured before it was written down:
//! a closed gate judges on the probes fired SINCE it closed (a full window of
//! stale losses outvoted every probe and a run closed at round 33 stayed
//! closed to the end); two probes have to pay before it reopens (one is a
//! coin on a 30% prose); and it judges from two rounds, not eight (eight
//! warm-up rounds at eight rows on a prose that closes at once cost 5%).
//!
//! **THE DEFAULTS FOLLOW THE CROWD** (`batch_concurrency`, which
//! `benches/pie_bench.py` passes): alone, the prices gate and no floor; in a
//! crowd, no prices and a floor of five with four narrow probes to reopen and
//! a probe cadence that backs off per probe. Measured at four lanes under
//! strict sealing on qwen38 (aggregate tok/s, plain 40 on every prompt):
//! counting 68, code 39, prose 28 — and prose is the honest number: with the
//! floor at 2.5 it read 31, with probes every sixteen fires 24-28, with the
//! cadence backing off 37 once and 28 the next time. Four lanes of a
//! low-yield prompt lose 25-30% to the warm-up rounds and the probes falling
//! out of the batch, whatever this gate does, and the plain loop is the
//! right one for them; the crowd default keeps the counting gain (1.6x) and
//! pays that. `min_tokens_per_round = 1000` closes the gate for good.
//!
//! **UNDER CONCURRENT LOAD THE PRICES LIE, SO `priced` IS OFF IN A CROWD.**
//! Eight lanes held open (`pie_bench.py tput --num-requests 8 --concurrency
//! 8`, aggregate tok/s, plain / open loop / floor 2.5 / priced): gemma prose
//! 107 / 56 / 87 / 92, gemma counting 110 / 174 / 181 / 149, qwen38 code 27
//! / 34 / 29 / 33, qwen38 prose 27 / 27 / - / 22. A lane's clock prices a
//! fire as the batch it waited for, the plain fire it is compared against is
//! that same batch, and rows that would have ridden a batch nearly free are
//! priced as a second wait — so the gate closes lanes it should not. What
//! ships is the yield floor (2.5, the row above) with the closed gate judging
//! on its probes; the yield is load-blind, which under load is the virtue.
//! (`latency --requests 8` admits one process at a time and measures nothing
//! about concurrency — its numbers are one lane's, eight times.) And the
//! eight-lane numbers above are taken under the scheduler's `ready` seal,
//! which on this box lands 2.2 lanes a batch; `PIE_SEAL_MODE=strict` fills
//! the batch and lifts PLAIN decode to 60.7 / 180 / 221 tok/s on qwen38 /
//! gemma / the A3B (1.7-2.3x), after which the open loop reads 1.02x on
//! qwen38 counting and 1.31x on gemma counting and the floor-gated prose
//! 0.65x: at eight lanes the batch is the lever, and this loop is a
//! one-lane latency tool (1.3-2.5x).
//!
//! **A priced loop is not a deterministic function of its prompt.** The
//! prices come off a clock, the widths follow the prices, and on a trunk
//! whose bf16 bits depend on the fire's width (§ below) two runs of one
//! prompt can part at a near-tie — measured: two priced prose runs on
//! qwen38 came back 256 tokens each, same speed, different at one token.
//! `--priced false` or `--verify_rows` restores a deterministic loop. The
//! Qwen3.6-35B-A3B rows above are thinking prose: that head's chat template
//! opens `<think>` whatever the system prompt says, and `/no_think` did not
//! close it.
//!
//! # One thing the loop does NOT promise
//!
//! Token identity with one-token-a-fire decode holds until the first
//! near-tie, not forever. A sixteen-row verify fire folds its dot products
//! in a different order than the one-row point, and on a list of European
//! capitals at 256 tokens the two roads pick different capitals at index 154
//! — the same bf16 accumulation-order floor this repo already records for
//! whole-prefill against one-token-at-a-time. Short runs (64 and 128 tokens,
//! code and prose) do come back identical, and that is what the gate above
//! was verified against.
//!
//! # The verify width is worth choosing, and nothing chose it well
//!
//! The drafter proposes fifteen whatever the target reads — a block diffusion
//! model is out of distribution at any other block width — but the VERIFY
//! fire is priced by its rows, and the price is a STAIRCASE: the tile point
//! pads a fire up to a row block. `a_fire_is_priced_by_its_width` on this
//! box, in one-row fires: 1.00 / 1.83 / 2.79 / 5.19 / 5.00 at 1 / 8 / 16 /
//! 24 / 32 rows when the table below was taken — twelve then cost MORE than
//! sixteen (3.13) because the tile launched two eight-row blocks over the
//! padded sixteen; `quant::widen_rung` now takes the one sixteen-row tile
//! and twelve reads 2.61, under sixteen. The table stands as measured, and
//! the ladder's rungs stand too: re-priced on the acceptance distributions,
//! a twelve rung never beats eight. 256 tokens, wall clock:
//!
//! ```text
//!                      v=4     v=6     v=8     v=12    v=16
//! counting            9.96s   8.51s   7.14s   8.06s   6.37s
//! code                11.26   10.87    9.83   13.77   11.18
//! ```
//!
//! So the best width is a property of the WORKLOAD — sixteen where the block
//! is nearly all kept, eight where five tokens are — and it is worth up to
//! 20%. THREE rules to pick it per round were written and all three are a
//! wash, which is why the parameter is still the caller's to state:
//!
//! - **A smoothed accepted prefix** against the stated prices RATCHETS: at
//!   eight rows a prefix can never be observed above seven, so nothing ever
//!   argues for going back to sixteen. Recall went 10.37s to 15.19.
//! - **A two-armed bandit on the guest's own clock**, tokens per nanosecond
//!   with a probe floor on each arm, picked the losing arm on two of three
//!   workloads (code 10.67s against 9.31 fixed; recall 14.87 against 10.37).
//!   A run that MIXES widths keeps less per round than either pure run: the
//!   round boundaries move, so the arms do not sample one distribution.
//! - **The drafter's own logit margin** (`--margin_width`), the signal Dspark
//!   spends a trained confidence head on. Pooled it separates cleanly — over
//!   sixty anchors on five prompts a position inside the accepted prefix
//!   carries a margin of 6.86 and one outside 1.10, and the first seven
//!   positions' mean reads 7.95 where the prefix reached eight against 2.16
//!   where it did not. It is MISCALIBRATED exactly where the width matters:
//!   on recall the drafter is unsure and RIGHT — a factual token has many
//!   plausible rivals — so it sends long rounds to the narrow width. At a
//!   threshold of 4 recall goes 10.74s to 15.38; pulled back to 2 the three
//!   workloads come to 28.73s against a fixed sixteen's 29.15 and an oracle's
//!   27.27.
//!
//! What survives is the MEASUREMENT, not a policy. The margin costs 9% of a
//! round (3.28s against 3.01 over 64 tokens) because `top_k(logits, 2)` reads
//! the plane ONCE and returns the runner-up beside the proposal. Two consumers
//! of that plane cost about a second a round — 3.0s to 7.4s for a bare second
//! reduction, 46s with a softmax on it — which is the thing to know before
//! reaching for any other signal off a fire's logits.
//!
//! # A candidate tree does not pay here, and the arithmetic says so
//!
//! The drafter denoises every position from the SAME context in one pass, so
//! its guess at `j + 1` does not depend on its guess at `j` — a second
//! candidate at one position costs a parallel chain, not a second draft.
//! Branching at `j` lays out `j + 2 * (block - j)` rows: 31 for `j = 1`.
//!
//! At the break, the target's token was the drafter's RUNNER-UP 14 times in
//! 45 (31%), and `top_k` already carries it. But 31 rows are 5.00 + 0.27 =
//! 5.27 one-row fires against the linear block's 3.09, so the tree must
//! return 14.1 tokens a round where the block returns 8.27 — a prefix of 13.1
//! against 7.27. Recovering a third of the breaks is worth about one token.
//! Not close, and the reason is the width curve: on a GPU where sixteen rows
//! cost two one-row fires rather than 2.79, the same tree would be arguable.
//!
//! # The recurrence is a fold, not a cell
//!
//! This SKU is hybrid, so a verify fire that folded its rows would fold the
//! REJECTED ones too and no `kv_len` could take them back —
//! `rs-speculative-decoding`'s header is the argument in full. So the verify
//! window is BUFFERED (`fold-len` leaves it unfolded), the rejected tail is
//! forgotten with `discard-buffered`, and the NEXT round's verify folds
//! exactly the accepted prefix ahead of its own rows.
//!
//! The draft fire in between is the trap: a fire must bind one recurrent
//! working set per request and is CHARGED ITS ROWS in the buffer, even though
//! the trunk that owns the recurrence runs over none of a draft fire's rows.
//! Binding a scratch working set instead corrupts the sequence's state; the
//! fix is to fold nothing and forget the fire's own rows again, which leaves
//! the buffer exactly as it was found.
//!
//! **THE KV ROLLBACK IS THE SUBTLE PART.** The verify fire writes a row's
//! keys at `held + i` for all `block` rows, and the rows past the break
//! carried tokens the target rejected — so the next round states
//! `kv_len = held + kept + 1` and the stale rows are simply never read.
//! The drafter's own context rides the same rows (its context arm runs on
//! every trunk fire), so one length rolls both streams back.

use inferlet::eta::hybrid::prelude::*;
use inferlet::{chat, monotonic_now_ns, session};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    /// Pre-tokenized prompt (the harness's, template already applied); wins
    /// over `prompt` when given.
    #[serde(default)]
    prompt_tokens: Option<Vec<u32>>,
    #[serde(default = "default_system")]
    system: String,
    /// Drop the template's stop tokens so the loop runs to `max_tokens`.
    #[serde(default)]
    ignore_eos: bool,
    /// Announce `ready` and wait for the harness's `start` before the clock.
    #[serde(default)]
    wait_for_start: bool,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Decode one token a fire instead — the same sequence by a different
    /// road, and the denominator of the speedup.
    #[serde(default)]
    baseline: bool,
    /// **HOW MANY OF THE BLOCK'S ROWS TO VERIFY**, the anchor's included.
    /// Zero (the default) verifies all of them.
    ///
    /// The drafter is trained at one block width and proposes fifteen
    /// whatever this says — a block diffusion model is out of distribution
    /// at any other width — but the TARGET need not read them all, and a
    /// verify fire is priced by its rows: sixteen cost 2.82 one-row fires on
    /// this box, eight cost 1.83. A workload whose accepted prefix rarely
    /// reaches eight pays for ten rows it was never going to keep. Which
    /// rows are worth a fire is the guest's call, not the engine's.
    #[serde(default)]
    verify_rows: u32,
    /// The block the head was trained at: sixteen for the v1 head
    /// (`qwen36-27b-dflash`), eight for DFlash2 (`qwen38-27b-dflash2`). The
    /// load does not advertise it yet, so the caller states it; absent, v1's.
    #[serde(default)]
    block: Option<u32>,
    /// Bind no mask to the draft fire. DFlash2's layers are all sliding and
    /// causal inside the block, so its plan states no masked arm for the
    /// draft rows and refuses a lane that carries one; v1's last layer is
    /// bidirectional and needs the mask this loop builds.
    #[serde(default)]
    no_mask: bool,
    /// Let the drafter's own logit margin choose the width per round, rather
    /// than verifying the whole block — see `wide_enough`. Off by default,
    /// because on three workloads it is a WASH.
    #[serde(default)]
    margin_width: bool,
    /// **STOP DRAFTING BELOW THIS MANY TOKENS A ROUND.** Absent, the crowd
    /// decides (`default_floor`): none alone, where the prices gate, five in
    /// a crowd. Zero states no floor, which is the loop the header's first
    /// tables were taken with; 2.5 is the header's shipped row. Around 3.0 is
    /// break-even for a dense 27B alone on this box, and in a crowd a round
    /// costs about two batch waits whatever the trunk; see `Gate`.
    #[serde(default)]
    min_tokens_per_round: Option<f64>,
    /// How many requests the caller is holding open beside this one
    /// (`benches/pie_bench.py` passes its `--concurrency`). Absent, one. Read
    /// for the floor's default alone — see `default_floor`.
    #[serde(default = "default_concurrency")]
    batch_concurrency: u32,
    /// **LET THE LOOP'S OWN YIELD PICK THE VERIFY WIDTH.** See `Gate::width`.
    /// **ON by default**, because the full block is the WORST width on every
    /// workload measured — see the table in the header. `verify_rows` pins a
    /// width and wins over this; `--verify_rows 16` is the loop every number
    /// above the width section was taken with.
    #[serde(default = "default_true")]
    auto_width: bool,
    /// **PRICE THE FIRES ON THE GUEST'S OWN CLOCK, AND LET THE PRICES PICK.**
    /// Absent, ON when the caller holds one request and OFF in a crowd
    /// (`batch_concurrency`): it wins alone on the box (see the header) and
    /// LOSES under concurrent load, where a lane's clock prices a fire as the
    /// batch it waited for. Every fire's wall time is kept by its shape — the
    /// one-row fire, the draft fire, each verify width — and a round is
    /// bought at the rung whose recent yield per price is best, or not at
    /// all when one token a fire beats every rung (`Prices`, `Gate::plan`).
    /// The constants the header's tables were taken with are a DENSE trunk's
    /// staircase; a mixture prices a wide fire by the experts its rows touch
    /// (gemma-4-26B-A4B: 1.7 / 2.5 / 4.6 one-row fires at 4 / 8 / 16 rows,
    /// and the draft fire 1.1 — where a 27B dense reads 1.38 / 1.83 / 2.79)
    /// and no stated floor fits both. `min_tokens_per_round` still closes
    /// the gate on top; `verify_rows` still pins the rung, and the gate then
    /// closes when that rung loses. Off, the loop is the header's.
    #[serde(default)]
    priced: Option<bool>,
}

/// The floor this file ships when none is stated. **Alone, none**: the
/// prices gate the lone lane (`priced`), and a yield floor beside them
/// closes what the prices would keep open — measured, qwen38 code alone
/// with both read 15.9 s against the prices' 10.4. **In a crowd, five**:
/// under strict sealing a round is two batch waits whatever the trunk, and a
/// lane that drafts falls out of step with its neighbours and narrows their
/// batches — measured at four lanes on qwen38 with the floor at 2.5,
/// counting 1.60x but code 0.87x and prose 0.77x (gemma: 1.40x and 0.73x),
/// where the break-even yield sits near four or five tokens a round rather
/// than the one-lane three. The crowd is what the caller says it is; the
/// guest has no other way to know it is one of several.
fn default_floor(concurrency: u32) -> f64 {
    if concurrency > 1 { 5.0 } else { 0.0 }
}

fn default_concurrency() -> u32 {
    1
}

fn default_prompt() -> String {
    "The quick brown fox jumps over".into()
}

fn default_max_tokens() -> usize {
    64
}

fn default_true() -> bool {
    true
}

/// **A TAKE-SIDE RING WITH A FRAME OF MARGIN ABOVE THE ADVERTISED
/// CAPACITY**, as `text-completion-bench` and `mtp-speculative-bench` size
/// theirs: sized at exactly `channel_capacity()` the runtime's ticket check
/// skips continuations that land inside its staging margin and the run-ahead
/// collapses.
///
/// **MEASURED NEUTRAL HERE**, and kept for the convention rather than for a
/// gain: 23.52 against 23.44 tok/s at eight concurrent. This loop's channels
/// are built fresh a round rather than hoisted like the mtp loop's, and its
/// concurrency wall is elsewhere — see the header.
fn ring() -> u32 {
    (channel_capacity() + 7 * frame_size()) as u32
}

fn default_system() -> String {
    "You are a helpful benchmarking assistant.".into()
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    tokens: Vec<u32>,
    count: usize,
    /// `text-completion-bench`'s envelope, for `benches/pie_bench.py`.
    num_prompt_tokens: usize,
    num_output_tokens: usize,
    token_ids: Vec<u32>,
    /// Draft+verify fire pairs after the prefill; zero under `--baseline`.
    rounds: usize,
    /// Proposals made, `block - 1` a round.
    drafted: usize,
    /// Proposals the target's own argmax agreed with.
    accepted: usize,
    /// `accepted / drafted`, or zero when nothing was drafted.
    acceptance_rate: f64,
    /// Tokens emitted per round, the number the economics turns on: a round
    /// costs two fires whatever this is.
    tokens_per_round: f64,
    /// The block width this run used.
    block: u32,
    /// The widths the rounds chose, in order.
    verify: Vec<u32>,
    /// Rows replayed through the buffer read path, summed over the rounds —
    /// a run whose `replayed` is zero exercised none of the fold-commit path.
    replayed: usize,
    /// Rows the buffer forgot because the target rejected them.
    discarded: usize,
    /// The recurrent buffer's page width, for reading the grant arithmetic.
    rs_page: u32,
    /// Fires the gate spent on one token because a draft was not worth it.
    /// Zero unless `min_tokens_per_round` was stated or the prices closed
    /// the gate.
    plain_fires: usize,
    /// The fires' measured prices, milliseconds by width — `0` the draft
    /// fire, `1` the plain fire, then each verify width fired.
    prices_ms: Vec<(u32, f64)>,
}

/// **WHAT A FIRE COSTS HERE, MEASURED, BY ITS SHAPE.** A price is the median
/// of the last few wall-clock samples of a fire — submit to readback, what
/// the guest WAITS, which is what it is buying with — kept for the draft
/// fire, the one-row fire and each verify width that has been fired. The
/// header's staircase (1.38 / 1.83 / 2.79) was one box's and one trunk's;
/// this is the trunk under this loop, on this box, under this load, and it
/// is what `Gate::plan` prices a round against.
struct Prices {
    draft: Vec<u64>,
    /// Indexed by width; `[1]` is the plain fire.
    verify: Vec<Vec<u64>>,
}

impl Prices {
    /// Samples kept a shape. A median over five forgets a stall and follows
    /// a load that changes within a request.
    const KEEP: usize = 5;

    fn new(block: u32) -> Self {
        Prices { draft: Vec::new(), verify: vec![Vec::new(); block as usize + 1] }
    }

    /// **A SHAPE'S FIRST FIRE IS NOT ITS PRICE.** The first fire of a shape
    /// compiles its plan, and a one-row fire measured 67 ms on its first two
    /// fires against 34 ms after — so the first sample a bucket sees is
    /// dropped, and a shape has a price from its second fire on.
    fn keep(bucket: &mut Vec<u64>, ns: u64) {
        if bucket.is_empty() && ns != 0 {
            bucket.push(0);
            return;
        }
        if bucket.first() == Some(&0) {
            bucket.remove(0);
        }
        bucket.push(ns);
        if bucket.len() > Self::KEEP {
            bucket.remove(0);
        }
    }

    fn median(bucket: &[u64]) -> Option<f64> {
        if bucket.is_empty() || bucket == [0] {
            return None;
        }
        let mut sorted = bucket.to_vec();
        sorted.sort_unstable();
        Some(sorted[sorted.len() / 2] as f64)
    }

    fn drafted(&mut self, ns: u64) {
        Self::keep(&mut self.draft, ns);
    }

    fn verified(&mut self, width: u32, ns: u64) {
        if let Some(bucket) = self.verify.get_mut(width as usize) {
            Self::keep(bucket, ns);
        }
    }

    fn plain(&self) -> Option<f64> {
        Self::median(&self.verify[1])
    }

    /// What a round at `width` costs: the draft fire and the verify at that
    /// width — or, for a width not fired yet, at the nearest fired width
    /// below it. A fire is never cheaper wider, so that is a LOWER bound:
    /// optimistic for an unexplored rung, which is what lets the ladder try
    /// a rung once and then know its price rather than never try it.
    fn round(&self, width: u32) -> Option<f64> {
        let draft = Self::median(&self.draft)?;
        let verify = (1..=width as usize)
            .rev()
            .find_map(|w| Self::median(&self.verify[w]))?;
        Some(draft + verify)
    }

    fn report(&self) -> Vec<(u32, f64)> {
        let ms = |ns: f64| ns / 1e6;
        Self::median(&self.draft)
            .map(|d| (0, ms(d)))
            .into_iter()
            .chain(
                self.verify
                    .iter()
                    .enumerate()
                    .filter_map(|(w, b)| Self::median(b).map(|p| (w as u32, ms(p)))),
            )
            .collect()
    }
}

/// **IS A DRAFT WORTH ITS FIRE AT ALL? THE LOOP'S OWN YIELD SAYS.**
///
/// A round costs about three one-row decodes on this box — a sixteen-row
/// verify fire is 2.79 of them and the five-layer draft fire, which still
/// goes through the target's `lm_head`, is the rest — so a workload that
/// yields fewer than about three tokens a round is SLOWER than decoding one
/// token a fire. Measured through `benches/pie_bench.py`, a list of European
/// capitals yields 2.88 and runs at 0.88x; prose yields 3.21 and is a wash.
///
/// This reads the loop's own recent yield and stops drafting when it falls
/// under a floor the CALLER states, then probes again after a run of plain
/// fires. **The floor is the guest's number, not the engine's** — the engine
/// dispatches the fires it is given and has no opinion about which rounds
/// were worth one. Left at zero, the gate never closes and the loop is the
/// one every measurement above was taken with.
struct Gate {
    /// Tokens a round below which drafting stops. Zero never closes.
    floor: f64,
    /// What the last rounds that DID draft emitted, newest last, each with
    /// the width it was verified at — a round that emitted its whole width
    /// is CENSORED there; see `Gate::width` and `Gate::expect`.
    recent: Vec<(u32, u32)>,
    /// The width the last drafting round used, the ladder's starting rung.
    last: u32,
    /// Plain fires since the gate closed, counted towards the next probe.
    since: u32,
    /// Plain fires the next probe waits for. `PROBE` when the gate closes;
    /// doubled each time a probe confirms the closure, up to `PROBE_MAX`, so
    /// a request the drafter cannot help stops paying a round in sixteen for
    /// the news, and back to `PROBE` the moment one opens it.
    probe_after: u32,
    /// Whether the priced gate is closed: the window then holds only the
    /// probes fired since it closed, and one probe that pays reopens it.
    closed: bool,
    /// Probes a closed gate judges on before it may reopen: `REOPEN_ALONE`
    /// or `REOPEN_CROWDED` — see those.
    reopen: usize,
    /// Whether the caller holds other requests open. A crowded gate backs its
    /// probe cadence off PER PROBE (16, 32, 64, 128 plain fires): every probe
    /// is a lane stepping out of the batch its neighbours are in, and four
    /// lanes probing every sixteen fires kept the batch two lanes wide (four
    /// lanes of qwen38 prose read 24-28 tok/s against 37 with the cadence
    /// backing off and 40 plain). Alone there is no batch to fall out of, so
    /// the cadence backs off only when a judgement rejects — a lone lane
    /// closed on one bad streak reopens within two probes.
    crowded: bool,
    /// Whether the verify width follows the yield. Off leaves it at `block`.
    auto: bool,
}

impl Gate {
    /// Rounds averaged before the gate will act. Fewer reads noise as a
    /// verdict; more spends the fires it is trying to save.
    const WINDOW: usize = 8;
    /// Plain fires between probes. One probe costs a round; at sixteen the
    /// probe is under 7% of the fires it is deciding about.
    const PROBE: u32 = 16;
    /// Where the priced gate's probe interval stops doubling: a request that
    /// turns list-shaped is still noticed within this many tokens.
    const PROBE_MAX: u32 = 128;
    /// Rounds the priced gate waits for before it judges — not the eight the
    /// stated ladder averages over: the per-position estimate is shrunk
    /// towards its pooled rate and corrects itself a round later, and eight
    /// warm-up rounds at eight rows on a prose that closes at once cost 5%
    /// of a 384-token run (7.5 s against 7.1 plain).
    const WARM: usize = 2;
    /// Probes a closed gate judges on before it may reopen, all at the
    /// narrowest rung. One is a coin: on a 30%-acceptance prose a four-row
    /// probe comes back whole about one time in eight, reopened the gate,
    /// and eight losing rounds followed (7.2 s -> 7.6 s over 384 tokens);
    /// two still churned in a crowd (four lanes of prose read 28 tok/s
    /// against a never-reopening loop's 39). Four pooled probes have to pay.
    const REOPEN_ALONE: usize = 2;
    /// In a crowd, four: two still churned there (four lanes of prose read
    /// 28 tok/s against a never-reopening loop's 39), and four cost the
    /// lone lane a fifth of its code and prose gains (qwen38 code 10.4 ->
    /// 12.9 s, prose 12.1 -> 16.1), so the count follows the crowd.
    const REOPEN_CROWDED: usize = 4;
    /// A round has to beat one token a fire by this much before the priced
    /// gate opens, and fall this far under before it closes — a window of
    /// eight reads a bad streak as a verdict otherwise, and the fires spent
    /// crossing back and forth are the ones the gate is meant to save.
    const HYSTERESIS: f64 = 0.9;
    /// **THE WIDTHS WORTH BUYING, AND THERE ARE ONLY THREE.** A verify fire is
    /// priced by its rows as a STAIRCASE — the vector fold answers 1-3 rows
    /// for about what one costs, 4 steps to 1.38, 5-8 share the tile's first
    /// step at 1.67-1.83, and 9-16 all cost 2.79 — so a width between rungs
    /// pays the rung above it and keeps nothing extra. Twelve rows measured
    /// 0.81x where sixteen measured 0.88x, which is the staircase, not noise.
    ///
    /// **DROPPING THE SIXTEEN RUNG IS A WASH, AND IT STAYS FOR A HEAD THAT
    /// CAN REACH IT.** `heads.py` prices sixteen below eight on every prompt
    /// for the head this SKU carries, which suggested cutting the top rung.
    /// Measured end to end with rungs `[4, 8]`: counting 1.87x against 1.88x,
    /// prose 1.42x against 1.38x, code 1.78x against 1.82x, capitals 1.33x
    /// against 1.32x — the same mean to two places, because the ladder
    /// reaches sixteen on only a handful of rounds anyway. It is kept because
    /// it costs nothing here and another head does want it: DSpark's counting
    /// prices BEST at sixteen (2.94).
    ///
    /// **THE LADDER STARTS AT FOUR AND NEVER GOES BELOW IT.** Two rows are
    /// nearly free but yield about 1.7 tokens a round against four rows' 2.8,
    /// and 1.7/1.00 loses to 2.8/1.38: measured, letting prose sink to two on
    /// a sixth of its rounds cost 1.41x -> 1.30x.
    const RUNGS: [u32; 3] = [4, 8, 16];
    /// The widest rung the warm-up rounds pay for before the window fills:
    /// eight, where the fire's premium over four is 28% — see `width`.
    const WARM_TOP: u32 = 8;

    fn new(floor: f64, auto: bool, crowded: bool) -> Self {
        Gate {
            floor,
            recent: Vec::new(),
            last: 0,
            since: 0,
            probe_after: Self::PROBE,
            closed: false,
            reopen: if crowded { Self::REOPEN_CROWDED } else { Self::REOPEN_ALONE },
            crowded,
            auto,
        }
    }

    /// Whether this round drafts. A probe is allowed through so a workload
    /// that turns list-shaped into code-shaped is noticed.
    fn drafts(&mut self) -> bool {
        if self.floor <= 0.0 {
            return true;
        }
        if self.closed {
            // Judged on the probes fired since closing — see `plan` for why
            // the window is emptied at closure. A rejected judgement empties
            // the window again and backs the probe cadence off; the cadence
            // does NOT back off per probe, or four probes would take 240
            // plain fires to gather and a gate closed on one bad streak
            // would stay closed for the request (measured: qwen38 code alone
            // read 23.8 s against 10.4).
            if self.recent.len() >= self.reopen {
                if self.mean() >= self.floor {
                    self.closed = false;
                    self.since = 0;
                    self.probe_after = Self::PROBE;
                    return true;
                }
                self.recent.clear();
                self.probe_after = (self.probe_after * 2).min(Self::PROBE_MAX);
            }
            return self.probe(0).is_some();
        }
        if self.recent.len() < Self::WINDOW || self.mean() >= self.floor {
            return true;
        }
        self.closed = true;
        self.recent.clear();
        self.probe(0).is_some()
    }

    /// **VERIFY AS MANY ROWS AS THE LOOP HAS BEEN KEEPING, ROUNDED UP TO A
    /// RUNG.** The signal is the loop's own tokens a round, which is exactly
    /// the accepted prefix plus the anchor — no prediction, no second pass
    /// over a logits plane, just what the last rounds actually did.
    ///
    /// The comparison is STRICT, so a width the yield has saturated steps up
    /// rather than pinning the loop under its own ceiling: at width four a
    /// mean of 4.0 asks for eight, and the loop climbs back on a workload
    /// that turns code-shaped. Measured against the shipped sixteen on this
    /// box, a fixed width picked this way is the per-workload optimum in all
    /// three cases (`prose` 4, `code` 8, `capitals` 4).
    fn width(&self, block: u32) -> u32 {
        if !self.auto {
            return block;
        }
        // **START CHEAP AND CLIMB, RATHER THAN WIDE AND FALL — BUT "CHEAP"
        // REACHES EIGHT.** With no history the ladder has nothing to read,
        // and eight rounds at a sixteen-row block is eight fires at 2.79
        // where the first rung costs 1.38 — on a 91-round request that alone
        // was 9% of the rounds spent at the most expensive width the loop
        // offers. Eight rows are another matter: 1.83 against four's 1.43,
        // a 28% premium for the warm-up rounds where sixteen's is 94%, and
        // a block-eight head (DFlash2) that starts at four spends its first
        // window censored under its own block. Measured on `qwen38-27b-dflash2`
        // (4 x 256, ladder): starting at four read 42.8 / 31.5 / 21.9 / 28.4
        // tok/s on counting / code / prose / capitals where a width PINNED
        // at eight read 49.0 / 33.5 / 19.9 / 30.7 — the ladder lost 6-15%
        // on three of four for the warm-up it spent narrow, and won only
        // prose, which the mean rule below still steps down to four once
        // the window fills.
        if self.recent.len() < Self::WINDOW {
            return Self::WARM_TOP.min(block);
        }
        // **A WINDOW THE TARGET TOOK WHOLE SAYS NOTHING ABOUT HOW MUCH MORE
        // IT WOULD HAVE TAKEN.** The yield is CENSORED at the width, so
        // reading the mean alone is a ratchet: narrow once and the mean can
        // never again exceed the width that produced it, and a request that
        // turns code-shaped stays pinned at four rows. Measured: without
        // this the code prompt sits at width 4 and 1.62x where width 8 is
        // 1.86x. So a window that saturated asks for the NEXT RUNG UP
        // instead, and the ladder climbs until the target stops taking
        // everything.
        let saturated = self.recent.iter().filter(|(t, w)| *w > 1 && t == w).count();
        let over = f64::from(self.last);
        // Three quarters, not half: a rung is left only on strong evidence,
        // because the step up costs 1.5-2x and a window that merely brushed
        // its ceiling has not earned it. At half the ladder reached sixteen
        // on a fifth of the prose rounds and gave back most of the win.
        let want = if saturated * 4 >= self.recent.len() * 3 { over } else { self.mean() };
        Self::RUNGS
            .iter()
            .copied()
            .find(|w| f64::from(*w) > want)
            .unwrap_or(block)
            .min(block)
    }

    fn mean(&self) -> f64 {
        self.recent.iter().map(|(t, _)| f64::from(*t)).sum::<f64>() / self.recent.len() as f64
    }

    /// What a drafting round emitted, the anchor's token included, and the
    /// width it was verified at.
    fn yielded(&mut self, tokens: u32, width: u32) {
        self.last = width;
        self.recent.push((tokens, width));
        if self.recent.len() > Self::WINDOW {
            self.recent.remove(0);
        }
    }

    /// **THE TOKENS A ROUND AT `width` WOULD EMIT, READ OFF THE RECENT
    /// ROUNDS POSITION BY POSITION.** A round that kept `k` of its `w - 1`
    /// proposals tried positions `0..=k` and hit the first `k`; positions
    /// past its width it never tried — which is what a mean of the rounds'
    /// yields gets wrong, since a window taken WHOLE at four says nothing
    /// about the fifth position and reading it as sixteen (or as four) is a
    /// guess either way. So this keeps a hit rate a POSITION, and the yield
    /// at `width` is one plus the chain of them: `Σ_i Π_{j<=i} a_j` over the
    /// positions the width shows. A position no recent round reached takes
    /// the window's pooled rate — neither the ceiling that climbed to
    /// sixteen on three saturated fours nor the floor that never leaves them.
    fn expect(&self, width: u32) -> f64 {
        let slots = width.saturating_sub(1) as usize;
        let mut hits = vec![0u32; slots];
        let mut tries = vec![0u32; slots];
        for &(tokens, w) in &self.recent {
            let kept = tokens.saturating_sub(1) as usize;
            for i in 0..kept.min(slots) {
                hits[i] += 1;
                tries[i] += 1;
            }
            if kept + 1 < w as usize && kept < slots {
                tries[kept] += 1;
            }
        }
        // A position tried twice in eight rounds reads 0 or 1 or a half, and
        // a tail rate of 1 extrapolated over eight untried positions is what
        // sent the code prompt to sixteen rows for nothing. Every position's
        // rate is shrunk towards the window's pooled rate by `PRIOR`
        // pseudo-tries, and the pooled rate is what the untried tail takes.
        const PRIOR: f64 = 2.0;
        let pooled = {
            let (h, t) = (hits.iter().sum::<u32>(), tries.iter().sum::<u32>());
            if t == 0 { 1.0 } else { f64::from(h) / f64::from(t) }
        };
        let mut chain = 1.0;
        let mut total = 1.0;
        for i in 0..slots {
            let rate = (f64::from(hits[i]) + PRIOR * pooled) / (f64::from(tries[i]) + PRIOR);
            chain *= rate;
            total += chain;
        }
        total
    }

    /// **WHICH ROUND TO BUY, IF ANY: THE RUNG WHOSE TOKENS PER PRICE IS
    /// BEST, AGAINST ONE TOKEN A FIRE.** `None` is a plain fire. Until the
    /// one-row fire, the draft fire and the window are all measured, this is
    /// the stated gate and ladder above; with them it is one comparison per
    /// rung — `expect(w) / round_price(w)` against `1 / plain_price` — with
    /// the stated floor still able to close it and a probe still let through
    /// after `PROBE` plain fires, so a request that turns list-shaped is
    /// noticed. `pinned` narrows the rungs to one.
    fn plan(&mut self, block: u32, prices: &Prices, pinned: Option<u32>) -> Option<u32> {
        let plain = prices.plain();
        // A closed gate judges on the probes it has fired since closing — the
        // window was emptied when it closed, so `REOPEN` probes that pay are
        // what reopen it. Left full, seven stale rounds outvote every probe
        // and the gate never reopens (measured: closed at round 33 of a
        // 384-token run and closed at the end, probing into the same seven
        // losses).
        let enough = self.recent.len() >= if self.closed { self.reopen } else { Self::WARM };
        let measured = plain.is_some() && prices.round(1).is_some() && enough;
        if !self.drafts() {
            return None;
        }
        if !measured {
            if self.closed {
                return self.probe(pinned.unwrap_or(Self::RUNGS[0]).min(block));
            }
            return Some(pinned.unwrap_or_else(|| self.width(block)));
        }
        let plain = plain.expect("measured");
        let rungs: Vec<u32> = match pinned {
            Some(w) => vec![w.min(block)],
            None => Self::RUNGS.iter().copied().filter(|w| *w < block).chain([block]).collect(),
        };
        let narrowest = rungs[0];
        let best = rungs
            .into_iter()
            .filter_map(|w| prices.round(w).map(|price| (w, self.expect(w) / price)))
            .max_by(|a, b| a.1.total_cmp(&b.1))?;
        let (width, rate) = best;
        // Open while a round beats the plain fire; closing takes a clear
        // loss, and a closed gate probes at the NARROWEST rung — the probe's
        // job is to sample the yield, and the cheap rung samples the first
        // positions, which is where a workload that changed shows first.
        let open = if self.closed { rate * plain >= 1.0 } else { rate * plain >= Self::HYSTERESIS };
        if open {
            self.closed = false;
            self.since = 0;
            self.probe_after = Self::PROBE;
            return Some(width);
        }
        if !self.closed {
            self.closed = true;
        } else {
            // A judgement (REOPEN probes gathered) that kept the gate closed
            // backs the cadence off.
            self.probe_after = (self.probe_after * 2).min(Self::PROBE_MAX);
        }
        self.recent.clear();
        self.probe(narrowest)
    }

    /// A closed gate's turn: a plain fire, or — every `probe_after` of them —
    /// one round at `width`. Alone, the cadence backs off where the judgement
    /// is rejected (`drafts`, `plan`); in a crowd, per probe (see `crowded`).
    fn probe(&mut self, width: u32) -> Option<u32> {
        if self.since >= self.probe_after {
            self.since = 0;
            if self.crowded {
                self.probe_after = (self.probe_after * 2).min(Self::PROBE_MAX);
            }
            return Some(width);
        }
        self.since += 1;
        None
    }
}

/// **IS THIS ROUND WORTH A WIDE VERIFY? THE DRAFTER ALREADY SAID.**
///
/// The signal is the LOGIT MARGIN, top-1 less top-2, because that is what
/// comes free: `top_k` reads the logits plane once and returns both, where a
/// probability would want a softmax over the whole vocabulary and a second
/// pass over the plane.
///
/// A verify fire's price is a staircase — the tile point pads a fire's rows up
/// to a row block, so a round costs 2.10 one-row fires at eight rows and 3.09
/// at sixteen — and the question is whether the prefix will reach eight. The
/// drafter's own top-1 probability answers it: over sixty anchors on five
/// prompts, a position INSIDE the accepted prefix carries 0.936 and one
/// outside 0.383, and the first seven positions' mean reads 0.959 where the
/// prefix reached eight against 0.553 where it did not. At this threshold
/// that calls the wide round on 28 of 30 long anchors and on 2 of 30 short
/// ones.
///
/// This is the signal Dspark spends a trained confidence head on. A block
/// drafter that reads out through the target's own `lm_head` has it for one
/// reduction over a plane the fire already computed.
fn wide_enough(confidence: &[f32]) -> bool {
    const THRESHOLD: f64 = MARGIN_THRESHOLD;
    let head = &confidence[..confidence.len().min(NARROW as usize - 1)];
    let mean = head.iter().map(|c| f64::from(*c)).sum::<f64>() / head.len().max(1) as f64;
    mean >= THRESHOLD
}

/// The narrow rung: the tile is flat from five to eight rows, so eight is
/// the widest width that costs what five does.
const NARROW: u32 = 8;

/// Where the margin separates a round that will reach eight from one that
/// will not — fitted offline, `scratchpad/dflash_ref/confidence.py`.
const MARGIN_THRESHOLD: f64 = 2.0;

/// The pages the recurrent buffer must hold for one fire: the survivors (at
/// most a page's worth of head offset before them), plus the window.
/// A fold releases only the WHOLE head pages its prefix covers and rebases
/// `buffer_head` by the remainder, so a run whose folds are not page-aligned
/// keeps a head offset the grant has to sit above — hence the extra page on
/// top of the survivors and the window.
fn buffer_pages_for(survivors: u32, window: u32, page: u32) -> u32 {
    (page.saturating_sub(1) + survivors + window)
        .div_ceil(page.max(1))
        .max(1)
        + 1
}

/// Rows one draft block carries — the model text's `DFLASH_BLOCK`. A guest
/// cannot read it off the load yet (`mtp_depth` advertises the seam's DEPTH,
/// which is one: a block drafter plants one proposal a ROW), so it is stated
/// in one place until the load advertises it.
const BLOCK_ROWS: u32 = 16;

/// The drafter's own mask token, `dflash_config.mask_token_id`. Stated here
/// for the same reason and with the same caveat.
const MASK_TOKEN: i32 = 248_070;

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::mtp_depth() == 0 {
        return Err("this SKU ships no draft head".into());
    }
    // **THE HEAD'S FACTS COME OFF THE LOAD.** The block the head was trained
    // at, its mask token and whether its block sees itself are advertised by
    // the model text that plants the draft seam (`model::draft_block`); the
    // inputs override for a diagnostic, and the constants below are the last
    // resort for a load that advertises nothing.
    let advertised = model::draft_block();
    let block = input
        .block
        .or(advertised.map(|d| d.rows))
        .unwrap_or(BLOCK_ROWS)
        .max(2);
    let mask_token = advertised.map_or(MASK_TOKEN, |d| d.mask_token as i32);
    let bind_mask = !input.no_mask && advertised.is_none_or(|d| d.bidirectional);
    // The first block row whose pick is a proposal: 1 when the anchor row
    // proposes nothing (DFlash), 0 when it proposes the next token (DSpark).
    let from = advertised.map_or(1, |d| d.proposals_from) as usize;
    let pinned = (input.verify_rows != 0).then(|| input.verify_rows.clamp(2, block));
    let page_size = kv_page_size();
    let rs_page = model::rs_buffer_page_size().max(1);
    // The harness's tokens when it sent them; else the chat template the
    // plain bench applies (system + user + the assistant cue). Both benches
    // must see the same prompt or the ratio is against a different prefill.
    let mut prompt: Vec<u32> = match &input.prompt_tokens {
        Some(tokens) => tokens.clone(),
        None => {
            let mut p = chat::system_user(&input.system, &input.prompt);
            p.extend(chat::cue());
            p
        }
    };
    if prompt.is_empty() {
        prompt.push(0);
    }
    let stop_tokens: Vec<u32> = if input.ignore_eos { Vec::new() } else { chat::stop_tokens() };
    if input.wait_for_start {
        session::send("ready");
        let _ = session::receive().await;
    }
    let n = prompt.len() as u32;
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();

    // The prompt, every token the loop commits, and one block of speculative
    // rows above the committed length that the next fire writes over.
    let ws = WorkingSet::new();
    let max_extent = n + input.max_tokens as u32 + 2 * block;
    let max_pages = max_extent.div_ceil(page_size);
    ws.reserve(max_pages).context("reserve KV")?;
    let pool = max_pages * page_size;
    // A hybrid text (qwen's GDN layers) needs its recurrent state bound and
    // folded round by round; an attention-only one (gemma) binds none, and
    // the fold-commit below has nothing to size — the drafter is the same.
    let rs_set: Vec<RsWorkingSet> = match model::pass_kind() {
        model::ForwardKind::Attention => Vec::new(),
        model::ForwardKind::Hybrid => vec![RsWorkingSet::new()],
        model::ForwardKind::Recurrent => {
            return Err("a block drafter reads attention kv; a recurrent-only text has none".into());
        }
    };

    let pipe = Pipeline::new();

    // ── PREFILL: the prompt, chunked. Every chunk leaves the drafter's
    //    context behind it, because the context arm rides every trunk fire.
    let mut anchor: i32 = 0;
    for &(base, end) in &prefill_chunks(n, None) {
        let len = end - base;
        let toks = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
        let indptr = Channel::from([0u32, len]).named("embed_indptr_p");
        let positions = Channel::from_iter(base..end).named("positions_p");
        let pages = Channel::from_iter(0..max_pages).named("pages_p");
        let page_indptr = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_p");
        let w_slot = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_p");
        let w_off = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_p");
        let kv_len = Channel::from([end]).named("kv_len_p");
        let readout = Channel::from([len - 1]).named("readout_p");
        let next = Channel::new([1], dtype::i32).named("next_p");

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &indptr)?;
        fwd.readout(&readout)?;
        fwd.attention(
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: ..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: None,
                },
            }),
            &rs_set,
            RsGeometry { fold_len: None, buffer: 0..0 },
        )?;
        fwd.epilogue(move || {
            next.put(&reshape(reduce_argmax(intrinsics::logits()), [1]));
        });
        fwd.submit(&pipe).context("prefill submit")?;
        anchor = next.take_host::<Vec<i32>>().await.context("prefill readback")?[0];
    }

    // ── THE LOOP ────────────────────────────────────────────────────────
    // `held` is the committed length: positions `0..held` are in the cache
    // and `anchor` is the token AT position `held`, not yet written.
    let mut held = n;
    let mut generated: Vec<u32> = vec![anchor as u32];
    let mut rounds = 0usize;
    let mut drafted = 0usize;
    let mut accepted = 0usize;
    let mut replayed = 0usize;
    let mut discarded = 0usize;
    // Tokens sitting in the recurrent buffer unfolded. The prefill folded
    // everything and the anchor is the first window's own row, so the first
    // round replays nothing.
    let mut survivors: u32 = 0;
    let mut widths: Vec<u32> = Vec::new();
    let crowded = input.batch_concurrency > 1;
    let floor = input.min_tokens_per_round.unwrap_or_else(|| default_floor(input.batch_concurrency));
    // Alone, the prices; in a crowd, the yield floor — see `priced`.
    let priced = input.priced.unwrap_or(!crowded);
    let mut gate = Gate::new(floor, input.auto_width, crowded);
    let mut prices = Prices::new(block);
    let mut plain_fires: usize = 0;
    // Drafting rounds since the plain fire was last priced, and plain fires
    // still owed to the current re-pricing; see `REPRICE`.
    let mut since_priced: u32 = 0;
    let mut owed_plain: u32 = 0;
    /// **THE PLAIN ROAD IS RE-PRICED EVERY THIS MANY ROUNDS**, with TWO
    /// one-row fires: the first folds the last round's survivors through the
    /// buffer, which is not the baseline's fire (67 ms against 42 on a
    /// hybrid trunk), the second binds the baseline's own geometry and is
    /// the one priced (`Prices::verified` is only fed a fire with nothing
    /// behind it). So the price every rung is compared against follows the
    /// load — the first fires' is the cold one, and at eight concurrent a
    /// fire costs what the batch costs. Two fires in thirty-two rounds is
    /// under a percent of the tokens.
    const REPRICE: u32 = 32;

    // A round commits several tokens at once, so the stop is read off the
    // committed run after the fact rather than one token at a time.
    while generated.len() < input.max_tokens
        && !generated.iter().any(|t| stop_tokens.contains(t))
    {
        if input.baseline {
            // One token a fire, the road the speedup is measured against.
            let toks = Channel::from([anchor]).named("toks_b");
            let indptr = Channel::from([0u32, 1]).named("embed_indptr_b");
            let positions = Channel::from([held]).named("positions_b");
            let pages = Channel::from_iter(0..max_pages).named("pages_b");
            let page_indptr =
                Channel::from([0u32, (held + 1).div_ceil(page_size)]).named("page_indptr_b");
            let w_slot = Channel::from([held / page_size]).named("w_slot_b");
            let w_off = Channel::from([held % page_size]).named("w_off_b");
            let kv_len = Channel::from([held + 1]).named("kv_len_b");
            let next = Channel::new([1], dtype::i32)
                .capacity(ring())
                .named("next_b");

            let fwd = ForwardPass::new();
            fwd.embed(&toks, &indptr)?;
            fwd.attention(
                Some(KvBinding {
                    working_set: &ws,
                    geometry: KvGeometry {
                        readable_pages: ..,
                        writable_pages: ..,
                        kv_len: &kv_len,
                        pages: &pages,
                        page_indptr: &page_indptr,
                        w_slot: &w_slot,
                        w_off: &w_off,
                        positions: &positions,
                        mask: None,
                    },
                }),
                &rs_set,
                RsGeometry { fold_len: None, buffer: 0..0 },
            )?;
            fwd.epilogue(move || {
                next.put(&reshape(reduce_argmax(intrinsics::logits()), [1]));
            });
            let fired_at = monotonic_now_ns();
            fwd.submit(&pipe).context("baseline submit")?;
            anchor = next.take_host::<Vec<i32>>().await.context("baseline readback")?[0];
            // Priced like the loop's own plain fire, so the two roads' one-row
            // fires can be read against each other in `prices_ms`.
            prices.verified(1, monotonic_now_ns().saturating_sub(fired_at));
            held += 1;
            generated.push(anchor as u32);
            continue;
        }

        // The width is chosen AFTER the draft fire, from what the drafter
        // itself says — see `wide_enough`. Pinned, it is stated here.
        // **THE FIRST TWO FIRES ARE PLAIN**, when the prices are asked for:
        // they price the one-row road every rung is compared against (the
        // first compiles and is dropped, `Prices::keep`), and cost the two
        // tokens they emit. Then the rung — or no round — is the prices'
        // choice; without prices, the stated gate's and ladder's.
        let planned = if priced {
            if since_priced >= REPRICE {
                since_priced = 0;
                owed_plain = 2;
            }
            if prices.plain().is_none() || owed_plain > 0 {
                owed_plain = owed_plain.saturating_sub(1);
                None
            } else {
                since_priced += 1;
                gate.plan(block, &prices, pinned)
            }
        } else {
            gate.drafts().then(|| {
                // A closed gate's probe fires at the narrowest rung: cheap,
                // and its yield is the first positions', which is where a
                // workload that changed shows first (`Gate::plan` does the same).
                if gate.closed {
                    pinned.unwrap_or(Gate::RUNGS[0]).min(block)
                } else {
                    pinned.unwrap_or_else(|| gate.width(block))
                }
            })
        };
        let mut verify = planned.unwrap_or(1);
        // The buffer must hold the survivors and this window; the grant is
        // the guest's one allocation decision.
        let buffer_pages = buffer_pages_for(survivors, block, rs_page);
        if let Some(rs) = rs_set.first() {
            let have = rs.buffer_size();
            if have < buffer_pages {
                rs.alloc_buffer(buffer_pages - have)
                    .map_err(|why| format!("alloc {} rs buffer page(s): {why}", buffer_pages - have))?;
            }
        }
        let fold_none = Channel::from([0u32]).named("fold_none");

        // ── the gate: a round the loop is losing on drafts nothing ──────
        //    A gated round is a one-row fire, and it FOLDS EVERYTHING: the
        //    pending survivors and its own row, so nothing stays buffered
        //    behind it. The first version left one row buffered and replayed
        //    it every fire, which put every gated fire on the committed
        //    path — a two-row scan, the conv's replay and a run of buffer
        //    copies a lane — where the baseline's fire folds directly.
        //    Measured at eight concurrent on prose (128 tokens, floor 3.5):
        //    the gate closed on 71 of 86 fires and the loop still read 17.8
        //    tok/s against the baseline's 25.9. Once the buffer is empty a
        //    gated fire binds the baseline's own geometry (no fold, no
        //    buffer) and IS the baseline's fire; the next drafting round
        //    starts from zero survivors, which the contract allows.
        let drafting = planned.is_some();
        let fold_all = survivors + 1;
        let fold_len =
            Channel::from([if drafting { survivors } else { fold_all }]).named("fold_len_v");
        let mut proposals_owned: Vec<i32> = Vec::new();
        if !drafting {
            verify = 1;
            plain_fires += 1;
        } else {
        // ── the draft: ONE pass over `[anchor, MASK x block-1]` ──────────
        let mut ids = vec![mask_token; block as usize];
        ids[0] = anchor;
        let toks = Channel::from(ids.as_slice()).named("toks_d");
        let indptr = Channel::from([0u32, block]).named("embed_indptr_d");
        let positions = Channel::from_iter(held..held + block).named("positions_d");
        let pages = Channel::from_iter(0..max_pages).named("pages_d");
        let page_indptr =
            Channel::from([0u32, (held + block).div_ceil(page_size)]).named("page_indptr_d");
        let w_slot = Channel::from_iter((held..held + block).map(|p| p / page_size)).named("w_slot_d");
        let w_off = Channel::from_iter((held..held + block).map(|p| p % page_size)).named("w_off_d");
        let kv_len = Channel::from([held + block]).named("kv_len_d");
        // The drafter's full-attention layer is BIDIRECTIONAL over the
        // block, which only a stated mask says; every key up to the block's
        // end is visible to every block row.
        let visible: Vec<bool> = (0..block)
            .flat_map(|_| (0..pool).map(move |j| j < held + block))
            .collect();
        let mask = Channel::from_shaped([block, pool], visible).named("mask_d");
        // **ONLY THE ROWS THE VERIFY WILL READ.** A block drafter's rows go
        // through the TARGET's one `lm_head`, so there is no separate draft
        // plane — and that head is the widest matrix in the model. Reading
        // all sixteen rows of it to then verify four is the same waste the
        // width staircase found on the verify side, paid a second time.
        //
        // The width is known BEFORE this fire whenever it comes from the
        // ladder or from `verify_rows`; only `margin_width` picks it after,
        // off margins this readout would otherwise not contain, so that one
        // still reads the block.
        let shown = if input.margin_width { block } else { verify };
        let readout = Channel::from_iter(0..shown).named("readout_d");
        // **THE PROPOSALS ARE THE HEAD'S READOUT, OFF THE `mtp.drafts`
        // SEAM.** A v1 head plants its per-slot argmax there; DFlash2 plants
        // its selector's walk. Reading the seam rather than the logits is
        // what lets one loop drive both — the guest asks the head what it
        // proposes and does not re-derive it.
        let out = Channel::new([shown], dtype::i32)
            .capacity(ring())
            .named("drafts_d");
        // **THE DRAFTER'S OWN CONFIDENCE, OFF THE LOGITS**, only when a
        // policy reads it (`margin_width`): a top-two over the whole plane
        // costs a reduction the plain loop does not need.
        let conf = Channel::new([shown * 2], dtype::f32)
            .capacity(ring())
            .named("conf_d");
        let want_margin = input.margin_width;

        let fwd = ForwardPass::new();
        fwd.set_drafting_block(true)
            .map_err(|why| format!("stating the draft block: {why}"))?;
        fwd.embed(&toks, &indptr)?;
        fwd.readout(&readout)?;
        fwd.attention(
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: ..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: if bind_mask { Some(&mask) } else { None },
                },
            }),
            &rs_set,
            // **FOLDS NOTHING, AND ITS OWN ROWS ARE DISCARDED BELOW.** A
            // fire binds one recurrent working set per request and is
            // charged its rows in the buffer, but the trunk that owns the
            // recurrence runs over NONE of a draft fire's rows — they are
            // the drafter's. Folding "everything" here would fold the
            // previous round's accepted prefix at the wrong instant and the
            // draft rows on top of it; folding nothing and forgetting the
            // rows again leaves the buffer exactly as this fire found it.
            RsGeometry { fold_len: Some(&fold_none), buffer: 0..buffer_pages },
        )?;
        {
            let out = out.clone();
            let conf = conf.clone();
            fwd.epilogue(move || {
                // **ONE OP, ONE CONSUMER, BOTH ANSWERS.** A second reduction
                // over the logits plane costs about a second a round even
                // with the plane bound once (3.0 s to 7.4 s over a 64-token
                // run; 46 s with a softmax on it) — two consumers take it off
                // the device. `top_k` reads it once and returns the values
                // beside the indices, so the proposal and the margin that
                // says how sure of it the drafter is come out together.
                out.put(&reshape(intrinsics::mtp_drafts(shown), [shown]));
                if want_margin {
                    let (value, _) = top_k(intrinsics::logits(), 2);
                    conf.put(&reshape(value, [shown * 2]));
                }
            });
        }
        let drafted_at = monotonic_now_ns();
        fwd.submit(&pipe).context("draft submit")?;
        // **ROW 0 IS THE ANCHOR, NOT A PREDICTION.** A block diffusion model
        // denoises each mask into the token AT ITS OWN POSITION, so row `i`
        // proposes position `held + i` and the anchor's row proposes nothing
        // new. The proposals are rows `1..block`.
        let picks = out.take_host::<Vec<i32>>().await.context("draft readback")?;
        prices.drafted(monotonic_now_ns().saturating_sub(drafted_at));
        proposals_owned = picks[from..shown as usize].to_vec();
        if want_margin {
            let value = conf.take_host::<Vec<f32>>().await.context("margin readback")?;
            // Row `r` occupies `[2r, 2r + 1]`: the best and its runner-up.
            let margin: Vec<f32> = (from..shown as usize)
                .map(|r| value[2 * r] - value[2 * r + 1])
                .collect();
            if pinned.is_none() {
                verify = if wide_enough(&margin) { block } else { NARROW };
            }
        }
        // The buffer is back to the accepted prefix the verify is about to
        // fold — see the draft fire's geometry.
        if let Some(rs) = rs_set.first() {
            rs.discard_buffered(block)
                .map_err(|why| format!("forget the draft fire's {block} row(s): {why}"))?;
        }
        }
        let proposals = proposals_owned.as_slice();
        widths.push(verify);

        // ── the verify: ONE trunk fire over `[anchor, proposals]` ────────
        let mut fed = vec![anchor];
        fed.extend_from_slice(&proposals[..verify as usize - 1]);
        let toks = Channel::from(fed.as_slice()).named("toks_v");
        let indptr = Channel::from([0u32, verify]).named("embed_indptr_v");
        let positions = Channel::from_iter(held..held + verify).named("positions_v");
        let pages = Channel::from_iter(0..max_pages).named("pages_v");
        let page_indptr =
            Channel::from([0u32, (held + verify).div_ceil(page_size)]).named("page_indptr_v");
        let w_slot = Channel::from_iter((held..held + verify).map(|p| p / page_size)).named("w_slot_v");
        let w_off = Channel::from_iter((held..held + verify).map(|p| p % page_size)).named("w_off_v");
        let kv_len = Channel::from([held + verify]).named("kv_len_v");
        let readout = Channel::from_iter(0..verify).named("readout_v");
        let truth = Channel::new([verify], dtype::i32)
            .capacity(ring())
            .named("truth_v");

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &indptr)?;
        fwd.readout(&readout)?;
        fwd.attention(
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: ..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    // No mask: the verify is an ordinary causal prefill of
                    // the rows the drafter guessed.
                    mask: None,
                },
            }),
            &rs_set,
            // A gated fire with nothing buffered behind it is the baseline's
            // fire: direct fold, no buffer. Anything else folds through the
            // buffer — the survivors, or on a gated fire the survivors and
            // its own row.
            if !drafting && survivors == 0 {
                RsGeometry { fold_len: None, buffer: 0..0 }
            } else {
                RsGeometry {
                    fold_len: Some(&fold_len),
                    buffer: 0..buffer_pages,
                }
            },
        )?;
        {
            let truth = truth.clone();
            fwd.epilogue(move || {
                truth.put(&reshape(reduce_argmax(intrinsics::logits()), [verify]));
            });
        }
        let verified_at = monotonic_now_ns();
        fwd.submit(&pipe).context("verify submit")?;
        let truth = truth.take_host::<Vec<i32>>().await.context("verify readback")?;
        // A one-row fire is priced only in the baseline's own geometry — with
        // nothing buffered behind it — because that is the road it stands for.
        if drafting || survivors == 0 {
            prices.verified(verify, monotonic_now_ns().saturating_sub(verified_at));
        }

        // ── what the target kept ────────────────────────────────────────
        // Row `i` of the verify predicts position `held + i + 1`, which is
        // what proposal `i` claims — so the prefix is read off one zip.
        let kept = proposals[..verify as usize - 1]
            .iter()
            .zip(&truth)
            .take_while(|(p, t)| p == t)
            .count();
        if drafting {
            rounds += 1;
            drafted += verify as usize - 1;
            accepted += kept;
            gate.yielded(kept as u32 + 1, verify);
        }
        replayed += survivors as usize;
        // The rejected tail never happened: forget it before the next fire,
        // whose fold reaches exactly the accepted prefix.
        let rejected = (verify as usize - 1 - kept) as u32;
        if rejected > 0 {
            if let Some(rs) = rs_set.first() {
                rs.discard_buffered(rejected)
                    .map_err(|why| format!("discard {rejected} rejected row(s): {why}"))?;
            }
            discarded += rejected as usize;
        }
        // A drafting round leaves its accepted prefix buffered for the next
        // fire to fold; a gated one folded its own row already.
        survivors = if drafting { kept as u32 + 1 } else { 0 };
        for tok in proposals[..kept].iter() {
            generated.push(*tok as u32);
        }
        // The break's token is the TARGET's, so the round is never wrong.
        anchor = truth[kept];
        generated.push(anchor as u32);
        // Positions `held ..= held + kept` carried tokens the target agreed
        // with and stay; everything above them is stale and never read.
        held += kept as u32 + 1;
    }

    // A stop token ends the text; anything the last block committed past it
    // is not the answer.
    if let Some(at) = generated.iter().position(|t| stop_tokens.contains(t)) {
        generated.truncate(at + 1);
    }
    generated.truncate(input.max_tokens);
    let count = generated.len();
    Ok(Output {
        sampler: "dflash-speculative-bench",
        text: model::decode(&generated)?,
        tokens: generated.clone(),
        count,
        num_prompt_tokens: prompt.len(),
        num_output_tokens: count,
        token_ids: generated,
        rounds,
        drafted,
        accepted,
        acceptance_rate: if drafted == 0 {
            0.0
        } else {
            accepted as f64 / drafted as f64
        },
        tokens_per_round: if rounds == 0 {
            0.0
        } else {
            (accepted + rounds) as f64 / rounds as f64
        },
        block,
        verify: widths,
        replayed,
        discarded,
        rs_page,
        plain_fires,
        prices_ms: prices.report(),
    })
}
