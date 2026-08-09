# Cutover: how `driver-metal-new` replaces `driver-metal`

> **Largely obsolete, 2026-08-10.** `driver-metal` is retired: removed from the
> workspace and from `engine`'s graph, source kept for reference only
> (`crates/driver-metal/README.md`). There is no longer a backend to cut over
> *from*, so this document's mechanics — the additive feature, the canary, the
> per-process config key, the one-release overlap, the rollback — describe a
> transition that will not happen. `DIRECTION.md` is the plan.
>
> What survives here and is still worth reading: the six-item gate (items 1 and
> 4 hold), the dependency chain among those items, and the decision to replace
> at the Rust boundary rather than behind the twelve `pie_metal_*` symbols —
> which has now happened by deletion rather than by migration.

Written 2026-08-09, at the point the `m1_runtime.cpp` port closed out. This
is the plan the handoff said must exist before the port finishes: what
"replace" concretely means, what must exist first, what test gate authorises
the flip, and how it rolls back.

## What the serving path is today

The engine reaches Metal through twelve `extern "C"` entry points —
`pie_metal_create`, `destroy`, `load_model`, `register_program`,
`register_channel`, `bind_instance`, `launch`, `copy_kv`, `copy_state`,
`resize_pool`, `close_channel`, `close_instance` — declared in
`crates/engine/src/driver/backend/metal.rs` and defined by the C++
`driver-metal` crate. The backend is chosen at build time by the cargo
feature chain `worker/driver-metal → engine/driver-metal →
extern crate driver_metal as _`.

## The decision: replace at the Rust boundary, not the C one

There are two places the new crate could slot in:

* **Behind the same twelve symbols** — a Rust cdylib exporting `pie_metal_*`.
  Minimal engine churn, and wrong: it rebuilds the exact `void*` boundary
  whose lifetime bugs are this crate's reason for existing. Every handle
  crossing that ABI goes back to being a raw pointer with a comment.
* **As a Rust backend module** — `engine/src/driver/backend/metal.rs` grows a
  twin that calls `driver_metal_new` directly: `Context`, `Runtime`, `Pool`,
  `Tables`, `Stepper`, `Ring`, the fire/M2/M3 calls. The `unsafe` count at
  the boundary drops to zero and the types carry their own lifetimes.

The plan is the second. The C ABI dies with the C++; it is not a fence to
preserve.

## Prerequisites, in order

The launch path (`m1_runtime.cpp`) is fully ported and device-tested. What
the twelve entry points need beyond it, and where each lands:

| entry points | need | port |
|---|---|---|
| `launch` (the serving hot path) | scheduling, fires, tickets, forward composition | `batch/` (~11.6k lines, mostly portable — portable half first) |
| `load_model` | weight loading and staging | `loader/` (~3.2k) |
| `register_program` / `bind_instance` | program/instance registry | **already ported** — `registry.cpp` is `pipeline/registry.rs` and `descriptor_resolve.hpp` is `pipeline/resolve.rs`, ledgered in `PARITY-REGISTRY.md`. One entry outstanding: `translate_kv_pages`, which is frame bookkeeping and belongs to `forward.cpp` |
| `register_channel` / `close_channel` | ring registry over [`Ring`] | small; `Ring` exists, the registry is bookkeeping |
| `copy_kv` / `copy_state` / `resize_pool` | K/V plumbing over `Elastic` | deciding half done (`src/store/control.rs`); the moving half with the Metal KV pool |
| (verification, not serving) | the CPU reference interpreter | **already ported** — `interp.hpp` (1.7k) is `src/pipeline/`, ledgered in `PARITY-INTERP.md`. What is missing is the harness that diffs it against the device, not the interpreter |
| `store/`, `model/` | small glue | ~375 |

Nothing else blocks assembly: the compile, the three execute paths, the
channel rings, the pools, heaps, tables and timing all exist and are tested.

## The gate

The flip is authorised when all of the following hold, and not before:

1. **The crate's own suite is green on device** — `cargo test -p
   driver-metal-new --no-fail-fast` on the Mac, zero red, under load.
   (True as of 2026-08-09; keep it true.)
2. **A/B at the backend seam.** The engine's driver-level tests run against
   both backends behind one harness, and every observable — registration
   outcomes, launch outcomes, channel effects, status reports — is equal.
   Divergences are bugs in one side or unstated behaviour in the other;
   both get written down before the flip.

   *Blocked (2026-08-09), and this was checked rather than assumed.* The
   harness needs a second `DriverBackend` variant, so the new crate must
   answer all fourteen dispatch methods in `engine/src/driver/backend.rs`.
   Most it can:

   | seam method | new crate today |
   |---|---|
   | `create` / `device_facts` | `Context::new` + `facts.rs` |
   | `register_program`, `register_channel`, `bind_instance`, `close_instance`, `close_channel` | `pipeline::Registry` (`PARITY-REGISTRY.md`) |
   | `encode` | both sides refuse; Metal media encode is unsupported |
   | `load_model` | `loader/` (portable done; staging arms deferred) |
   | `copy_kv` / `copy_state` / `resize_pool` | deciding half **done** (`store::plan_kv_copy` etc., `PARITY-STORE.md`); the moving half needs the Metal KV pool |
   | `launch` | **missing** — see below |

   `launch` is the one that decides the item. It takes a `FrameSubmission`:
   an instance roster, a frame-union KV page translation with its CSR
   partition, a required-page high-water for admission, and an ordered list
   of steps. The new crate's top entry is `Decoder::fire(&[Lane])` — a
   model-level call over requests, slots and tokens. Everything between the
   two is `forward.cpp`: instance-to-lane mapping, `translate_kv_pages`,
   step sequencing, the `Exhausted`/`Impossible` admission outcomes, and the
   completion broker. `forward.cpp` (5393) is ledgered `missing` and is
   explicitly the last thing in the port, over everything above it.

   So item 2 is not a harness someone forgot to write. It is downstream of
   the largest remaining slice, and attempting it first produces a twin
   backend that cannot serve the one call the A/B is mostly about.
3. **Token-exact decode.** A real checkpoint (qwen3.5 is the one with
   in-tree semantic coverage), fixed seeds, N ≥ 1000 decoded tokens: the new
   backend's tokens are bit-identical to the old one's. The PTIR channel
   plane is deterministic by contract, so any drift is a defect, not noise.

   *Progress (2026-08-09):* `tests/device_smoke.rs` decodes Qwen3.6-27B
   greedily on four paths — M=1 ring, paged sequential, paged per-row
   stream, and a fleet (equal and mixed lengths) — and all reproduce one
   reference sequence exactly over the tested horizon; staging is
   byte-exact for every tensor, and the golden-tap bisect holds every
   stage function to host arithmetic at cosine 0.999+. The N ≥ 1000 horizon has
   since run: 1000 greedy tokens on the paged path at a flat 18.6 tok/s,
   no faults, no NaNs, no rate creep as the cache grew to position 1006.
   An mlx-lm cross-check on the same prompt matches tokens 1–2 exactly
   and then tie-breaks differently at a step where mlx-lm's own top-two
   gap is 0.0625 — one bf16 ulp at that magnitude — with both backends
   agreeing on the same top-five set (`PIE_SMOKE_TOP5_AT` is the
   instrument). A near-tie amplified by greedy, not a defect signal; the
   BIT-identical standard this item states applies to the comparison
   against the OLD driver, which shares these kernels and remains the
   one open leg.
4. **The interpreter agrees.** The same fires replayed through the CPU
   reference interpreter match the device results within its stated
   tolerance — the oracle the C++ never had wired to Metal.

   *Correction (2026-08-09):* this item was recorded as blocked on porting
   `interp.hpp`. It is not — the interpreter is `src/pipeline/`, closed out
   in `PARITY-INTERP.md`, and `adopt_launch_package` → `make_instance` →
   `step` runs today. The open work was always the harness.

   *Progress (2026-08-09):* the harness's **CPU half is done**.
   `tests/oracle_interp.rs` runs one trace through both this crate's
   interpreter and `tensor_compiler::eval::interp` — the original golden
   model, reached as a dev-dependency exactly as `pipeline/status.rs`
   reaches the fault table — and compares commit verdicts and every
   host-readable channel bit for bit. Seven cases, three of them
   mutation-verified. **They agree**, including at matmul's zero-skip, which
   both implement identically. That settles the oracle question the stronger
   way: `pipeline::step` is now pinned to the original, so the remaining
   device test may diff against the local interpreter and still be claiming
   agreement with the golden model.

   *Done (2026-08-09):* the device half is `tests/device_oracle.rs`, running
   the compiler's **real emitted MSL** against the interpreter on one trace.
   **The tolerance this item names is one ulp, and only transcendentals
   spend it.** Plain arithmetic, the width-32 pairwise reduction and the
   argmax tie-break are all exact on device; `exp` differs by one ulp
   (`exp(0.5)` is `1.6487212` in Rust and `1.6487213` in Metal — two libms,
   both within half an ulp of the truth). The tolerance is compared only
   against float magnitudes: index, integer and boolean lanes are exact
   whatever it is set to, so it can never quietly accept a different argmax.

   Stated boundary for both halves: per-layer tap stages, which the C++ and
   this crate both reject at classification and which are therefore outside
   the claim. The cases are arithmetic-contract probes rather than coverage;
   broadening them is ordinary work now that the construction exists.
5. **Soak without growth.** A sustained decode (hours, not minutes) with
   `PoolStats`, `Memory`, and ring counts sampled: no monotone growth, no
   working-set creep. This is the leak class `release_standalone_buffer`
   existed to hide.
6. **The panic regressions.** The elastic drop-under-mapping and keepalive
   scenarios from 2026-08-08 rerun against the new path. The machine staying
   up is the assertion.

### The gate is a chain, not a checklist

The six items are numbered but not ordered, and reading them as a to-do list
has now cost time twice — item 4 was attempted as a 1.7k-line port that did not
exist to do, and item 2 as a harness that cannot be written yet. The actual
dependency structure:

```
  forward.cpp  ──┬──▶  2. A/B at the seam  ──┬──▶  5. soak without growth
                 │                            └──▶  6. panic regressions
                 └──▶  3's open leg (vs the OLD driver)

  1. suite green      ✅ independent, holds
  4. interpreter      ✅ independent, holds
```

**Four of the six are behind `forward.cpp`.** Items 5 and 6 need a running
backend to soak and to re-run the panic scenarios against, which needs item 2,
which needs `launch`. Item 3's remaining leg — bit-identical against the *old*
driver — needs both backends running the same frames, which is item 2 again.
Item 3's own N ≥ 1000 horizon is already done through this crate's device
smoke, which is why it reads as half-held rather than blocked.

Items 1 and 4 are the two that never depended on it, and both hold.

The consequence for planning: **there is one critical path, and it is
`forward.cpp` plus the `copy_kv`/`copy_state`/`resize_pool` trio beside it.**
Anything else picked up before those is either independent cleanup or work that
will be redone once the frame path exists. Whoever picks this up next should
start at `forward.cpp`'s prerequisites in `PARITY-BATCH.md` — the scratch
schedule, the segment loop, the bind layouts — rather than at the gate.

## Mechanics

1. Land the assembly as feature `driver-metal-new`, additive, alongside
   `driver-metal`. Both can be compiled into one worker; a config key picks
   the backend at create time, so a canary flips per process with no
   rebuild.
2. Canary on the Mac Studio under real traffic; watch the gate's metrics.
3. Default flips: `driver-metal` the feature becomes an alias for the new
   backend; the C++ stays in-tree, buildable, for one full release.
4. Delete `driver-metal` (the crate, `csrc/`, its CMake), and the twelve
   `pie_metal_*` declarations with it. `PARITY.md`, `PARITY-M1.md` and the
   subsystem ledgers are the record of what the C++ was; they stay.

## Rollback

Until step 4, rollback is the config key (minutes) or the feature default
(one rebuild). That is the reason for step 3's one-release overlap: the
window where rollback is cheap must outlive the window where surprises are
likely. After step 4 there is no rollback, only revert — which is why step 4
waits for a full release of silence.
