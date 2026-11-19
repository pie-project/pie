# Host-class guest lanes at depth 2: the host's take deferred one step

Status: design for review (GPU). No code. Written by the Metal session, 2026-11-19, on memory-redesign
6071e274e. Facts are cited by file; numbers are measured unless marked otherwise.

## 1. The problem

A guest lane bound `GeometryClass::Host` runs at depth 1. Its next fire is built on the host from what the
last one settled, so the GPU idles while the host:
- takes the step's outputs,
- settles the step,
- stages and encodes the next frame.

Measured idle between frames (each frame's command-buffer start and end, Qwen3-0.6B, home-carve, no
budgets, text_completion_bench decodes):

| device | lanes | busy a frame | idle a frame (median) | idle share |
|---|---|---|---|---|
| M1 Max | 1 | | | 7.3% |
| M1 Max | 4 | 8025 µs | 430 µs | 5.1% |
| M1 Max | 16 | | | 9.5% |
| M1 Max | 64 | | | 15.1% |
| M5 (Mac Pro) | 4 | 5355–5382 µs | 171–185 µs | 3.2–3.4% |

- The M1 rows at 1, 16 and 64 lanes are from F31's audit; the 4-lane rows are from the beam-epilogue
  probes (Metal and Mac Pro, today).
- M5 at 16 and 64 lanes is not measured yet. Ask: Mac Pro, the same probe at c16 / c64.
- The idle share is the upper bound of what this design returns. The share grows with lanes because the
  host's per-step work (takes, settles, staging) grows with lanes and the GPU's doesn't, per lane.

## 2. Which lanes are Host-class today, and why

`runtime::pipeline::fire::host_geometry::classify` binds a pass:
- **Host**, when the program is `host_derivable`: the host derives every geometry port (tokens,
  positions, slots, KV extent, fold, mask) from what it wrote itself.
- **DeviceGeometry** or **DecodeEnvelope**, when the device computes a geometry port and the engine's
  port mask covers it.
- **Host** with a named reason (`HostGeometry::EngineLacks` / `HostGeometry::Program`), otherwise.

On the M1 beam mix (F31), 2080 instances bound Host-class, 32 DecodeEnvelope, 0 DeviceGeometry.

The Host-class bulk is the standard samplers attached to plain decodes. Each samples a token in its
epilogue and puts it on a host-read channel. The host takes it at the landing (`LaneRequest::Landed` →
`publish_landed` → `pump_out`) and writes it as the next fire's `embed_tokens`. The token is computed on
the device but routed through the host, which is what makes the program "host-derivable" and its fires
serial (`host_geometry.rs` module doc: "each fire waits for the one before it to settle").

A lane with no program at all is also Host (`inferlet/host/forward.rs`, the `else` arm).

## 3. What exists to build on

- **DecodeEnvelope already runs ahead where the engine allows it.** Its pages and write slots are the
  host's. The device supplies tokens (`token_device_source`) and, where computed, the mask. So the next
  fire is staged while this one runs (`classify`'s comment).
  - CUDA allows it (`engine-cuda/src/serve/prepare.rs`, `can_runahead`; Host-class instances pass that
    check too).
  - Metal doesn't: `fence_instances` (engine-metal `serve.rs`) holds every attached instance, whatever
    its class, to its last fire's landing before its next is staged. One status word a session, and
    the host reads its cells between fires.
- **Landings are told from the completion handler** (6071e274e, Metal). The runtime takes channels at
  `publish_landed`, after the engine commits the epilogue.
- **Scratch arms are per class** (`engine::GuestState::scratch_arms`, stated by `guest_state(.., class)`).
  Every engine states 1 today (engine-metal `program/guest.rs`, engine-cuda `store.rs`, engine-gpu
  `guest/state.rs`).

## 4. The design

### 4.1 Make the sampled token a device-sourced port

A standard sampler's token goes to the next fire's `embed_tokens` from its epilogue's cell on the device,
as a decode envelope's does, not through the host:
- The sampler programs declare their token output as the `embed_tokens` source.
- `classify` then finds them not host-derivable and binds them DecodeEnvelope where the engine covers
  `embed_tokens` (CUDA, ROCm, Metal's masks all do; `host_geometry.rs` table).
- Positions, slots and pages stay the host's (+1 a step, known before the token).

This is the smallest change that gives Host-class decodes run-ahead. It reuses the envelope path rather
than adding a fourth class. Lanes whose program reads a host input every step (constrained decoding fed
from the host, tool-call gates) stay Host: their next fire really does need the host.

### 4.2 The host's take, deferred one step

At depth 2, step N+1 is submitted before step N's take.
- **The take of N** (the token for streaming, stop checks and the process's history) happens at N's
  landing, while N+1 runs, as envelopes do today.
- **Stop one step late.** If N's take shows the process stopped (EOS, a stop sequence, max tokens), N+1
  is already in flight. Its output is dropped, and its KV row and position are not committed.
  - The cost is one wasted lane-step a finished process, which is about 1/len of the work.
  - Open (GPU): does the runtime already discard an envelope lane's step past its stop, and roll back
    its KV extent and slot? If envelopes are only served for beam today, this is new runtime work.
- **Commit order.** N's channel commit (the engine's epilogue commit at `publish_landed`) happens before
  N+1's, because harvests run in sequence order.
  - N+1's prologue cells are refreshed (`prepare_airborne` → `refresh_cells`) before N commits. So N+1
    is staged on the cursors N will leave: the cursors now, plus N's puts and takes as its plan states
    them.
  - If N faults, N+1 was staged on cursors that never came. It is abandoned with N, its lanes failed
    the same way, which is the failed-buffer path (b32d2f57a) applied to a pair of passes.
  - The host never writes the instance's channels between N's enqueue and its publish (GPU confirmed
    for 6071e274e: the post-submit pump defers when the submit is asynchronous; evacuations move only
    quiescent instances). That stays true at depth 2: an instance with N+1 in flight is not quiescent.
- **Ring capacity.** The token channel needs capacity ≥ 2 (depth + the cell the host has not taken),
  so N+1's epilogue doesn't put over N's untaken cell.
  - The sampler programs' token channels must state it.
  - The prologue gate (`blocked_at`) then holds the depth honestly. An instance whose ring is full
    waits, it doesn't overwrite.

### 4.3 Each engine: what the host writes a fire, doubled by parity (the value scratch is not)

Revised after GPU's note on CUDA's one arm.

- **The value scratch needs no second arm.** On CUDA all guest work is ordered on the context's single
  stream (side streams fork after the wave's pull and join before the land). On Metal, command buffers
  on one queue run in order. So N+1's guest kernels start only after N's finish, and a fire's value
  scratch (GPU-written, dead outside its fire) is per fire in time, not in space.
  - `scratch_arms` stays 1 for every class, as long as the host never reads N's scratch after N+1 is
    enqueued.
  - Metal: it doesn't. The verdict is read from the status word, which is resident, and channel cells
    are in the rings.
  - CUDA: GPU's agent is confirming.
  - This keeps a beam instance at 22.8 MB on Metal, not 45.6.
- **What does need two sets: the state the host writes while staging a fire, which N still reads while
  N+1 is staged.** On Metal (engine-metal `program/launch.rs`) that is:
  - the status word, which `refresh_cells` resets to "ready" before each fire, and from which the fire's
    verdict is read at its harvest;
  - the pending flags, which the host zeroes before each fire (`zero_span`);
  - the instance's grouped slot tables, which `grouped.refresh` rewrites with the fire's bound cells.
  - The pair's batch tables are already one set a frame in flight.
  - These three regions are small (words and tables), so the resident bytes grow by one copy of them. Fire
    N uses set N mod 2, and N's verdict is read from its set at its harvest, through the
    completion-handler check 21e887e7f added (`EpilogueCheck` already carries the status regions it
    reads).
  - `fence_instances` holds an instance only to fire N−1's landing (depth 2), not N's.
- **Ports.** The device token source of N+1 is N's put cell, the pending cell of the token channel's
  ring at the cursor N will leave. Same queue, so N's epilogue has finished before N+1's prologue reads
  it.

### 4.4 Per engine

| engine | needs |
|---|---|
| runtime | samplers as envelopes (4.1); submit a run-ahead lane's N+1 before N's take; stop one step late with N+1 dropped and its KV / slot rolled back (4.2); fault of N abandons N+1; token ring capacity ≥ 2 checked at bind |
| engine-metal | status word, pending flags and grouped slot tables doubled by fire parity (resident grows by those); `fence_instances` to depth 2 for run-ahead classes; the verdict check per parity (the `EpilogueCheck` path); token source = N's pending cell; `scratch_arms` stays 1 (queue order) |
| engine-cuda / ROCm | already runs envelopes ahead (`can_runahead`); one arm is safe by single-stream order (GPU confirming that the host reads no scratch of N after N+1 is enqueued); the same audit for host-written per-fire state (status words, cells) |
| engine-gpu (wgpu / Vulkan) | the same audit: one queue's order for the value scratch, parity sets for what the host writes a fire; its envelope path (`api.rs` binds `DecodeEnvelope`) to run ahead, or stay depth 1 and keep correctness (the runtime asks the engine's depth, it doesn't assume it) |

## 5. Alternatives considered

- **Cohort ping-pong without engine changes.** The runtime splits Host-class lanes into two cohorts
  that alternate frames, each at depth 1. The GPU always has a frame, but each frame has half the lanes.
  - At decode sizes the step is weight-read bound (M1, Qwen3-0.6B: 1 row 5.96 ms, 4 rows 8.21 ms; F33).
  - So two half-frames cost close to two full ones: c64 as 2 × 32 would spend more GPU than the 15%
    idle it returns. Rejected for decode. It could pay at large batch (compute-bound), not where the
    idle is largest today.
- **A fourth geometry class ("Host, run-ahead").** Equivalent to 4.1 but duplicates the envelope's
  plumbing. Rejected unless 4.1 hits a sampler that can't name its token as `embed_tokens`.
- **Reading the take in place (unified memory, Metal).** Skips a copy, not the wait: the host still
  needs the token before it builds the next fire. It doesn't remove the idle.

## 6. What it removes, and how to check

- **Expected:** the depth-1 idle in section 1, minus the host work that doesn't overlap:
  - the take of N runs during N+1;
  - staging N+2 runs during N+1.
  - Upper bound: 5–15% on the M1 and ~3% at 4 lanes on the M5 (c16 / c64 to measure).
- **Gate:**
  - decode c1 / c16 / c64 ABBA, busy and idle per frame, against the tip (M1 and M5);
  - text_completion outputs identical (temperature 0), including stops at EOS, at stop sequences and at
    max_tokens (the one-step-late stop is where it could go wrong);
  - beam --verify clean;
  - the failed-buffer test extended to a pair in flight (N faults, so N+1 is abandoned, the instance
    recovers);
  - memory: instance bytes before and after, per class (expected: resident + the parity copy of the
  status word, flags and slot tables; scratch unchanged).

## 7. Questions for GPU

1. Do envelope lanes already stop one step late in the runtime (drop N+1, roll back its KV extent and
   slot)? If yes, 4.2 is only ring capacity and the fault pairing.
2. Do the standard samplers have a reason to route the token through the host (`HostRole::Reader`),
   beyond the take for streaming? If not, 4.1 is a program declaration change in the sampler library.
3. (Answered in part: single-stream order makes one arm safe for the value scratch.) Does CUDA's host
   write any per-fire state of an instance (status words, cells, tables) that fire N still reads while
   N+1 is staged? On Metal three regions qualify (4.3).
4. Order: Metal's two status words first (needed for beam to run ahead on Metal at all: today it's
   depth 1 there), then 4.1 and 4.2 together.
