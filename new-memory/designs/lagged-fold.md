# Lagged fold: a run-ahead stop on recurrent and window sets

Status: design, for CUDA (L40S), Metal (Mac) and engine-gpu (GPU) to build
from. Lands with `sc/l40s-stop-truncate` (retract, 551d11c13 + 1a212379e),
so a hybrid set is never left unusable by an early stop.

## Problem

A guest decode runs ahead: up to `W` fires are submitted before their tokens
are read. Today every decode fire folds all it writes (`Fold::All`), so on a
set with a recurrent state (Qwen3.5's conv + delta state) or window lists
(Gemma 4, gpt-oss) the rows past a stop are folded into state that cannot be
cut back. `retract` refuses those sets by name, and the set is dead for the
next turn.

## Rule

**The lag is the unaccepted rows.** On a set with recurrent or window state,
the committed state (the folded length the set's KV and rs share) never
includes a row the guest has not accepted. Rows past it stay in the set's
buffer and are replayed by each fire, which the engines already do for beam
and buffer-only eviction on hybrids.

`W` is only the bound: the binding's submit depth,
`W = (frames_in_flight + HOST_COVER) × frame_size`
(`scheduler::run_ahead_window()`, 10 at the defaults), never a constant.

## Definitions (one for every engine)

The verb is the shared `engine::RsVerb::Buffer { pages, base, folded, fold,
lineage }` (`crates/engine/src/fire.rs:643`), planned by
`engine::rs::LanePlan::of` (`crates/engine/src/rs.rs:24`). Its meaning does
not change; only who sets `folded` and `fold`, and when:

- `folded` — the set's committed length as the runtime stages the fire:
  everything below it is in the state. Under the lag it is the accepted
  length the last fold reached, never past an unaccepted row.
- replay range — `[folded, first)`: the lane's buffered rows (accepted but
  not yet folded, and unaccepted), replayed from the committed state in
  order before the fire's own rows. At most `W` rows a lane.
- `fold` — always `FoldLen::Host(n)`: the rows this fire freezes past
  `folded`, `n = accepted_now − folded`, clamped to `first + rows` as today.
  Host-known, so run-ahead-safe on every engine (Metal resolves ports on the
  host; a device `fold` port is never used for this).

Who advances them: the runtime, when it stages a fire.
- **Accepted.** A fire whose input tokens the device carries (a run-ahead
  decode) is accepted once the guest takes its host-read output (the take is
  already a host call per token; no new call). A fire with host-placed
  tokens (a prompt, a host-fed decode) is accepted as it is submitted, as
  today: its rows are the guest's own. A device-token fire with no host-read
  output is accepted when it lands.
- **Fold.** The next fire staged on the set carries `fold = accepted −
  folded`. After a stop there is no next fire on that pipeline: the
  accepted-but-unfolded rows (at most `W`) stay buffered and the set's next
  fire (the next turn, a fork's first fire) folds them. Nothing is copied
  after landing.
- **Bound.** If a guest runs more than `W` fires past what it has read
  (custom windows, e.g. text-completion-bench's ignore_eos depth), the
  oldest rows past the bound are accepted as they land, so the lag never
  exceeds `W`; a retract reaching into them is refused by name.

`Fold::All` on such a set therefore means "fold through what the guest has
accepted". On a set with neither state, nothing changes (decoded rows stay
unsealed until the next prompt fold or release, as 551d11c13 has it).

## Retract under the lag

The rows past the cut are all buffer rows, so retract is a buffer truncate
on every kind of set: KV (`KvStore::truncate`, already in 551d11c13), the rs
buffer pages past the cut, the window head, and the host cursor. No frozen
row and no state is touched.

## Windows

The trim is the runtime's: `RsStore::window_advance` (`store/rs.rs:1543`)
retires each list's slots whose last row lies below `reach − window`.

- Host-positioned fires (the envelope decode) use `reach = committed`, which
  moves with the fold; with `fold = accepted − folded` it never retires a
  slot that unaccepted rows or a stop's rewind still need.
- Device-positioned fires (`positioned = false`) use the fire's first
  writable token as `reach`, which under the lag lies past `committed`, so
  a stop rewinding to `committed` could find its slots retired. **Every
  fire retires below `min(committed, its reach) − window`.**
- At a retract the window head rewinds to `committed`; slots past it that
  hold stray rows are released, or copied before the next write into them
  (the tail rule `checkpoint_at` / `prepare_window` keep: a write into a
  shared slot copies first). A fork or checkpoint at the accepted length
  takes that tail copy and never shares a slot with rows past it.

No extra retirement lag (`committed − (window + W)`) is needed: `committed`
never passes an unaccepted row.

## Reservation

The bound's buffer rows go in the set's reservation: `W` rows a lane on the
rs buffer pages (KV page size, 16 tokens) beyond what the set holds, i.e.
`ceil(W / page) + 1` pages, plus the KV pages those rows take (already
counted as written rows). Qwen3.5-0.8B: about 12 KiB of conv input a token a
layer, 18 layers, so about 2.2 MiB a lane at `W = 10`; 9B about 3.8 MiB.

## Measurement: today's replay path (CUDA, L40S, Qwen3.5-0.8B)

A scratch bench guest folds the prompt to `n − L` and each decode one row,
so every decode replays `L` buffered rows through the existing path.

| lag L | c1 tok/s (ITL) | c64 tok/s |
|---|---|---|
| 0 (twice) | 378.6 / 378.3 (2.61 ms) | 10645 / 10654 |
| 1 | 261.6 (3.76 ms), −31% | 4994, −53% |
| 2 | 254.7 (3.87 ms), −33% | 4970, −53% |

The cost is a fixed path switch, not a cost a row: a replaying fire takes
the `~rsx` body, whose recurrent layers run `conv1d_extended` /
`gated_delta_extended` (`engine-cuda/src/dispatch/ssm.rs:463`), two chunked
prefill passes a layer (committed part, then the tail) plus the gather and
scatter into the extended scratch; `rung_rows(own, carried)` lifts the
fire's rung by its carried rows (c64); some shapes have no `~rsx` body and
walk eagerly. The lagged fold cannot land on this path.

## The decode replay kernel (agreed with GPU)

One pass a recurrent layer, inside the ordinary decode body:

- **Inputs.** The committed state (the set's state slot); the lane's carried
  rows' inputs to the recurrence, read from the set's buffer: for gated
  delta the k / v / gate / beta rows (as the existing replay gathers them),
  for conv1d the last `kernel_size − 1 + W` input rows; the fire's own rows;
  `commit_len` a lane (= `folded + fold − first`, from `LanePlan`), and the
  carried count a lane (≤ W) as data.
- **Outputs.** The fire's own rows' outputs (carried rows produce none:
  their outputs were read when they ran), and the state at `commit_len`,
  never the state past an unaccepted row.
- **Work a layer.** Conv: load the committed conv window, slide over the
  carried rows then the own rows, write the window at `commit_len`.
  Gated delta: a block per (lane, head) loads the committed state once
  (each thread keeps its value column, `K_d` floats, in registers, as
  `gdn_step` streams it today), steps the carried rows and the own row in
  registers, stores the state at `commit_len`. State traffic is one read
  and one write a step, the same as the single-row `gdn_step` /
  `conv1d_update` decode today; the extra is the carried rows' inputs
  (~12 KiB a row a layer on 0.8B) and `L` more delta-rule steps
  (~1 MFLOP a row a layer a lane on 0.8B; ~3.5 GFLOP a step at c64, L = 2).
  Expected: within a few percent of the single-row step for L ≤ 2.
- **Body and rung.** The rung is chosen from the fire's own rows; carried
  rows are a kernel input, not rows of the fire (`rung_rows` stops counting
  them for this kernel's lanes). One body a rung, captured at load, the
  same count as today's decode bodies: the replay kernel replaces the
  single-row recurrent kernels in the decode body, a lane with no carried
  rows replaying none. No `~rsx` body family for decode.
- **Reservation.** The carried rows' buffer is in the set's reservation
  (W rows a lane), so nothing is allocated at fire time.

## Engines

- **CUDA.** Today's replay: `RsBody::Replays` (`serve/arming.rs:1261`),
  extended scratch sized for the widest bucket (`serve/prepare.rs:2039`), a
  partial fold as a `commit_len` split scan (`serve/enqueue.rs:686`,
  `kernels/ssm/gdn_chunk.cu:227`), replayed rows over `budget.max_tokens`
  refused (`serve/prepare.rs:1563`). The decode replay kernel replaces it
  for decode classes.
- **Metal.** `LanePlan::of` with `RsVerb::Buffer` staged per fire
  (`engine-metal/src/serve.rs` ~4488–4680); carried rows bounded by the
  current rung (`serve.rs:4908`). Mamba-2 and biased buffered conv refused
  (`dispatch/ssm.rs:23–31`). M1 replay proxy (Qwen3.5-0.8B, single-lane
  fires of 1 / 2 / 3 rows, as W = 0 / 1 / 2): the recurrent kernels cost
  0.730 / 0.986 / 1.052 ms a fire. Metal switches from the step kernels
  (`causal_conv1d`, `gated_delta_scan_step`) to the chunked prefill kernels
  at ≥ 2 rows: a fixed +0.26 ms a fire at a lag of 1 (+35% of the recurrent
  layers, ~+3.8% of the 6.7 ms c1 decode fire), then ~+0.07 ms a row. Metal
  needs the one-pass decode replay kernel too.
- **engine-gpu.** `LaneFold::of` wraps `LanePlan::of`
  (`store/frozen.rs:34`); replay in `dispatch/rs.rs`, committed kernels in
  `dispatch/ssm.rs:92`. Vulkan / wgpu share the kernel via Slang / WGSL.

## Tests

- Hybrid miniature: lagged vs unlagged logits over 30 decode steps, within
  tolerance.
- Stops at k = 1 / 2 / 3 / 7 on Qwen3.5-0.8B, then the next turn matches one
  drawn a token at a time (`working-set-handles` `stop_after`).
- A window model's cut (gemma-4-E2B), same check.
- The L = 0 / 1 / 2 bench: L = 1 / 2 within ~2% of L = 0, c1 and c64.
