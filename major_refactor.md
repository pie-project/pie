# Major refactor: `runtime/engine/src` north star

Handoff plan for restructuring the engine crate (`pie-engine`). Self-contained: everything
needed to execute is in this file. Written 2026-07-10 against the uncommitted working tree
on `dev` (parent commit `f2fa7dbd`).

## Goal

PTIR is the endgame execution model. The engine should read as six layers with
dependency arrows pointing strictly down, every module named for the resource it
manages, and no parallel legacy paths. Two modules are the main offenders today:

- `inference/` is a grab-bag: batch scheduler, KV contention, forward-prepare
  projection, legacy execution glue, and a vestigial one-message actor service.
  The name carries no information in a crate that is entirely an inference engine.
- `ptir/` is named after the wire format (the IR lives in `interface/ptir`), every
  file stutters a `ptir_` prefix, and `ptir_host.rs` (2.1k lines) staples WIT glue
  to the run-ahead fire engine. It also leaks upward: `driver/frame.rs` imports
  `crate::ptir`, and the scheduler imports pipeline channel internals.

## Design rules (the invariants that keep it clean)

1. **Imports flow down only.** A module may import modules in layers below it,
   never sideways-up. Checkable with the grep gates in the Verification section.
   `bootstrap.rs` is the sole exception (composition root).
2. **WIT host glue contains resource types and Host impls, nothing else.** The
   moment a host file grows an algorithm, the algorithm moves down a layer.
3. **A file is named for the noun it exports; a module name is a membership
   test** you can answer without opening files. No prefix stutter
   (`ptir::ptir_kv`), no `request.rs` that contains no `Request`.
4. **Placement follows what code manages, not what it reads or who calls it.**
   This puts reclaim in `store/` (manages pages) even though the fire path
   triggers it, and geometry projection in `store/kv/` (a view over the mapping)
   even though `pipeline/fire/` calls it.
5. **One forward path.** New capability lands as PTIR program capability plus,
   at most, store/scheduler generality. Never a second submit path. (The
   dumb-driver principle applied one layer up.)

## Target layout

```
runtime/engine/src
├── driver.rs          L0  dumb backend ABI module root
├── driver/
│   ├── frame.rs           LaunchPlan, LaunchSubmission, ChannelValue (moved down)
│   ├── backend.rs  completion.rs  ffi.rs  ffi/{cuda,metal}.rs
│   └── registry.rs        driver specs only; scheduler handles move OUT
│
├── store.rs           L1  device memory module root
├── store/
│   ├── kv/                page_table, hash, write, project.rs (new home of project_kv,
│   │                      KvProjection, KvWrite, PhysicalPageId)
│   ├── rs.rs  pool.rs  genmap.rs  registry.rs
│   └── reclaim.rs         pressure ladder (from inference/contention.rs)
│
├── scheduler.rs       L2  spawn/shutdown, SchedulerHandle + its own handle registry
├── scheduler/             per-driver batching: when accumulated fires launch
│   ├── worker.rs          per-driver loop (from inference/scheduler.rs)
│   ├── batch.rs           capacity accounting + accumulator
│   ├── wire.rs            owned plans -> borrowed FFI descriptors, page trim
│   │                      (from inference/request.rs)
│   ├── quorum.rs          the wait-all fire rule (from inference/policy.rs)
│   ├── stats.rs           SchedulerStats + cross-driver aggregation (absorbs the
│   │                      InferenceStats family from the inference mod root)
│   └── probe.rs           profile-fire probes (from inference/probe.rs)
│
├── pipeline.rs        L3  THE forward path module root
├── pipeline/              guest-programmed pipelines (from ptir/)
│   ├── program.rs         container bytes -> bind -> price -> cache (ptir_registry.rs;
│   │                      absorbs model_profile() from ptir_host)
│   ├── instance.rs        program + seeds -> Pipeline (ptir_instance.rs)
│   ├── channel.rs         ChannelCell host endpoint, SPSC roles (ptir_channel_store.rs)
│   ├── fire.rs            PendingFire engine, OpSignal, poison, FIFO finalization
│   │                      (the non-glue ~60% of ptir_host.rs)
│   └── fire/              one fire: prepare -> run-ahead submit -> finalize/poison
│       ├── geometry.rs    ports -> LaunchPlan geometry, pure (ptir_geometry.rs)
│       ├── kv.rs          KvTxn prepare/finalize (ptir_kv.rs; absorbs canonical-fire
│       │                  evidence helpers + the descriptor validation from
│       │                  inference/paging.rs)
│       ├── rs.rs          RsTxn prepare/finalize (ptir_rs.rs)
│       └── lease.rs       device-geometry page grants (ptir_lease.rs; absorbs
│                          detect_device_geometry)
│
├── inferlet.rs        L4  guest process runtime module root
├── inferlet/
│   ├── host.rs            WIT bindgen! + add_to_linker (from api.rs)
│   ├── host/              WIT boundary: one thin file per interface (from api/)
│   │   ├── forward.rs     Host impls for pie:inferlet/forward (channel, forward-pass;
│   │   │                  the glue ~40% of ptir_host.rs)
│   │   ├── pipeline.rs    Host impls for pie:inferlet/pipeline
│   │   └── …              types, model, tokenizer, grammar, working-set (kv/rs),
│   │                      session, media, speech, system, chat, tools, reasoning
│   ├── program/  process/  linker/  python/  sandbox.rs   (unchanged)
│
├── server.rs          L5  client edge module root
├── server/                sessions, handler, inbox, data transfer
│                          (unchanged; inbox.rs is already the u2i messaging home)
│
├── service.rs             KEEP: shared actor utility (scheduler, server, program,
│                          linker, process all use it)
├── telemetry.rs            OTLP export (unchanged)
└── bootstrap.rs            composition root; only module allowed to import everything
```

Modules that cease to exist: `inference/` (dissolved across scheduler/store/pipeline,
legacy `execute.rs` deleted), `ptir/` (renamed/split), `api/` (becomes
`inferlet/host/`). Already gone in the working tree: `messaging.rs`, `token.rs`,
`util.rs`, `probe/`.

## Phase 0: already applied, verify and commit

The "small fix" review items are **already applied in the uncommitted working
tree**. Do not redo them; verify and fold into the first commit.

| Fix | State in working tree |
|---|---|
| 1. p2p messaging removal | `interface/inferlet/messaging.wit`, `api/messaging.rs`, `src/messaging.rs` deleted; `world.wit` has no messaging import; `server/inbox.rs` is the per-process inbox; `api/session.rs` uses `server::inbox::receive`; demos (`inferlets/agent-swarm`, `inferlets/demo-messaging`) deleted |
| 2. token removal | `src/token.rs` deleted; no `AuthByToken`/`auth_by_token` anywhere in `client/`, `runtime/`, `bin/`; no `base64` in engine Cargo.toml; clients moved to public-key auth (`client/rust/src/client.rs::authenticate`) |
| 3. util removal | `src/util.rs` deleted; no `get_pie_home` in the workspace; `bootstrap/src/paths.rs` (the bootstrap crate) is the canonical resolver |
| 4. probe relocation | top-level `probe/` deleted; now `inference/probe.rs`, `profile-fire` feature kept (`runtime/engine/Cargo.toml:105`) |

Verification greps (all must return nothing):

```sh
grep -rn "messaging" interface/inferlet/ runtime/engine/src/api.rs
grep -rn "AuthByToken\|auth_by_token\|AuthConfig" client runtime bin bootstrap --include='*.rs' --include='*.py' --include='*.js'
grep -rn "get_pie_home\|engine::util" --include='*.rs' runtime bin bootstrap
```

Then: `cargo check -p pie-engine && cargo test -p pie-engine`, plus a client
connect smoke test (Rust client `authenticate` path). Commit this state before
starting Phase 1 so the mechanical refactor diffs stay reviewable.

Note: `inference/probe.rs` is correct only while `inference/` exists; Phase 3
carries it to `scheduler/probe.rs`.

## Phase 1: kill the upward dependencies (small, safe, mechanical)

1. **`ChannelValue` moves down.** `PtirChannelValue` (a `{channel: u64, bytes}`
   pair, no PTIR semantics) moves from the `ptir` mod root into
   `driver/frame.rs` as `ChannelValue`, next to `LaunchPlan` which carries it.
   Update `driver.rs`, `driver/frame.rs`, and ptir-side users. This flips the
   driver->ptir arrow.
2. **Completion-sink inversion.** `inference/scheduler.rs` imports
   `ptir_channel_store::{ChannelCell, ChannelError::Full}` to deliver results.
   Invert: the pipeline layer installs a completion (extend
   `driver/completion.rs::InstanceCompletion`) that fills its cells; the
   scheduler delivers bytes to the sink and stays ignorant of cell semantics.

Gate after phase: `grep -rn "crate::ptir" runtime/engine/src/driver runtime/engine/src/inference` returns nothing.

## Phase 2: grow `store/`

1. `inference/contention.rs` -> `store/reclaim.rs`. Rename the concept in docs
   from "contention" to the reclaim ladder (idle-lease drop, preempt-youngest,
   wait queue, restore-on-free). `ContentionOrchestrator` et al. keep working
   names; file/module name is the contract. Callers: `ptir_host`,
   `api/{kv,rs}_working_set`, `inferlet/process`, `bootstrap`.
2. Split `inference/paging.rs`:
   - `PhysicalPageId`/`BlockId` type aliases + `project_kv`, `KvProjection`,
     `KvWrite` -> `store/kv/project.rs` (it derives driver geometry from the
     mapping, a view over store state). Heaviest import churn:
     `PhysicalPageId` is referenced across inference/ptir.
   - The WIT-descriptor validation half (`execute()`, `PrepareError`,
     seal-eligibility split) stays put for now; it rides to `pipeline/fire/kv.rs`
     in Phase 4.

Gate: `store/` imports only `driver/` (+ leaf crates); reclaim tests still pass
(`store/kv/tests.rs`, `store/rs/tests.rs`).

## Phase 3: extract `scheduler/`, dissolve the `inference` mod root

1. Moves (drop old names per rule 3):
   - `inference/scheduler.rs` -> `scheduler/worker.rs`
   - `inference/batch.rs` -> `scheduler/batch.rs`
   - `inference/request.rs` -> `scheduler/wire.rs`
   - `inference/policy.rs` -> `scheduler/quorum.rs`
   - `inference/stats.rs` -> `scheduler/stats.rs`
   - `inference/probe.rs` -> `scheduler/probe.rs`
2. **Scheduler owns its handle registry.** Move
   `install_scheduler_handle`/`clear_scheduler_handle`/`scheduler_handle` out of
   `driver/registry.rs` into `scheduler.rs` (or `scheduler/registry.rs`).
   `driver/registry.rs` keeps driver specs only.
3. **Dissolve `InferenceService`.** It is a one-variant actor (`GetStats`).
   `spawn()` becomes `scheduler::spawn` (creates per-driver workers);
   `get_stats()` becomes a plain function in `scheduler/stats.rs` aggregating
   over the registry's `Vec<Arc<SchedulerStats>>` (they are lock-free atomics,
   no actor round-trip needed). The `InferenceStats`/`FireStats`/`QuorumStats`
   structs move to `scheduler/stats.rs`; rename `InferenceStats` ->
   `scheduler::Stats` (or `AggregateStats`).
4. `submit_async`/`submit_prebuilt_async` become `scheduler::submit*` (or
   methods on `SchedulerHandle`). Callers: `ptir_host`, `ptir_kv`, and the
   legacy `execute.rs` path.
5. What remains of `inference/`: `execute.rs` + the paging validation
   remainder. Leave in place, marked `#[deprecated]`-in-spirit via the module
   doc; deleted in Phase 5.

Gate: `scheduler/` imports only `store/` + `driver/`;
`grep -rn "crate::inference::scheduler\|SchedulerHandle" runtime/engine/src/driver` empty;
`cargo test -p pie-engine`; run the cuda e2e suite on the 4090 workstation
(`bin/pie/tests/cuda_plain_gen.rs` first, then the full set) since scheduler
timing paths cannot be validated host-side.

## Phase 4: `ptir/` -> `pipeline/`, `api/` -> `inferlet/host/`

The big rename. Do the two moves in one phase because `ptir_host.rs` splits
across both destinations.

1. **Split `ptir_host.rs` (2132 lines) along the glue/domain seam:**
   - WIT resource types + `Host*` trait impls for `pie:inferlet/forward`
     (Channel, ForwardPass) -> `inferlet/host/forward.rs`; for
     `pie:inferlet/pipeline` (Pipeline) -> `inferlet/host/pipeline.rs`. Thin:
     each impl body should be a call into `pipeline/`.
   - Run-ahead engine (`PendingFire`, `PendingMove`, `OpSignal`, `poison_readers`,
     FIFO finalization, the optimistic `committed_tokens` cursor) ->
     `pipeline/fire.rs`.
   - Helpers distribute: `model_profile()` -> `pipeline/program.rs`;
     `detect_device_geometry`, `DevGeo` -> `pipeline/fire/lease.rs`;
     `canonical_kv_shape`, `canonical_fire_evidence`, `host_known_port_u32s`,
     `decode_le_u32s` -> `pipeline/fire/kv.rs`.
2. **File renames** (mechanical, drop the stutter):
   `ptir_registry.rs` -> `pipeline/program.rs`, `ptir_instance.rs` ->
   `pipeline/instance.rs`, `ptir_channel_store.rs` -> `pipeline/channel.rs`,
   `ptir_geometry.rs` -> `pipeline/fire/geometry.rs`, `ptir_kv.rs` ->
   `pipeline/fire/kv.rs`, `ptir_rs.rs` -> `pipeline/fire/rs.rs`,
   `ptir_lease.rs` -> `pipeline/fire/lease.rs`. Symbol renames ride along:
   `ptir_kv_prepare` -> `fire::kv::prepare`, `PtirKvTxn` -> `fire::KvTxn`, etc.
   The word "ptir" survives only where the IR is touched (imports of
   `pie_ptir::*` in `program.rs`, `fire/geometry.rs`).
3. **Move `api/*` -> `inferlet/host/*`** and `api.rs` (bindgen! macro +
   `add_to_linker`) -> `inferlet/host.rs`. Every `crate::api::pie::` path
   in the crate becomes `crate::inferlet::host::pie::`. To keep the diff
   reviewable, add a temporary `pub use inferlet::host as api;` shim in lib.rs,
   migrate call sites in a follow-up commit, then drop the shim.
4. Fold the paging remainder from Phase 2 (`execute()` descriptor validation,
   `PrepareError`) into `pipeline/fire/kv.rs`.
5. Fix stale docs as you go: `ptir_registry`'s broken `super::program_cache`
   link (the real one is `inferlet/program/`), the mod-root "later refactor
   renames ptir_host->host" note (this is that refactor).

Accepted layering exception, document it in `inferlet/host.rs`: `host/session.rs`
calls `server::send_file` and `server::inbox::receive` (guest-to-client I/O goes
through the server's facade). If it grates later, extract a `client_io` seam;
out of scope here.

Gate: `grep -rn "ptir" runtime/engine/src --include='*.rs' | grep -v "pie_ptir"` returns
nothing; wit-bindgen worlds unchanged (`interface/inferlet/world.wit` untouched);
cuda e2e suite green on the workstation.

## Phase 5: delete the legacy forward path

`inference/execute.rs` self-describes as machinery behind a WIT inference host
that no longer exists in `world.wit@0.2.0` (no `inference` import; the linker
registers no such interface). Audit its remaining consumers (last check:
`api/grammar.rs` pulls from `inference::execute`), relocate what is genuinely
still served (likely grammar-mask plumbing -> `pipeline/fire/` or
`inferlet/host/grammar.rs`), then delete `execute.rs` and the emptied
`inference/` module. Remove `pub mod inference` from lib.rs.

Gate: `grep -rn "crate::inference" runtime/engine/src` empty; full engine test
suite + cuda e2e + one real inferlet run via the workstation serve/client loop.

## Phase 6: visibility and polish

1. **lib.rs surface audit.** Consumers of `pie-engine` are the worker role lib,
   `bin/pie`, and `runtime/engine/tests/`. Keep `pub` only what they import
   (expect: `bootstrap`, `server` spawn surface, driver registration for the
   worker, telemetry init); everything else `pub(crate)`. Compile the workspace,
   not just the crate, to find the true surface:
   `cargo check --workspace && cargo test -p pie-engine --no-run`.
2. Doc sweep: every module root gets its one-line charter matching this file's
   tree; delete relocation fossils ("moved out of api/", thrust/lane codenames
   in doc headers where they no longer aid navigation).
3. Add the layering grep gates to CI (see Verification).

## Move map (one line per current file)

| Current | Target | Phase |
|---|---|---|
| `inference.rs` (mod root) | dissolve: spawn -> `scheduler.rs`, stats structs -> `scheduler/stats.rs`, submit fns -> `scheduler/` | 3 |
| `inference/scheduler.rs` | `scheduler/worker.rs` | 3 |
| `inference/batch.rs` | `scheduler/batch.rs` | 3 |
| `inference/request.rs` | `scheduler/wire.rs` | 3 |
| `inference/policy.rs` | `scheduler/quorum.rs` | 3 |
| `inference/stats.rs` | `scheduler/stats.rs` | 3 |
| `inference/probe.rs` | `scheduler/probe.rs` | 3 |
| `inference/contention.rs` | `store/reclaim.rs` | 2 |
| `inference/paging.rs` | split: projection -> `store/kv/project.rs`, validation -> `pipeline/fire/kv.rs` | 2, 4 |
| `inference/execute.rs` | delete (relocate grammar remnants) | 5 |
| `ptir.rs` (mod root) | `ChannelValue` -> `driver/frame.rs`; rest dissolves | 1, 4 |
| `ptir/ptir_host.rs` | split: glue -> `inferlet/host/{forward,pipeline}.rs`, engine -> `pipeline/fire.rs`, helpers distributed | 4 |
| `ptir/ptir_registry.rs` | `pipeline/program.rs` | 4 |
| `ptir/ptir_instance.rs` | `pipeline/instance.rs` | 4 |
| `ptir/ptir_channel_store.rs` | `pipeline/channel.rs` | 4 |
| `ptir/ptir_geometry.rs` | `pipeline/fire/geometry.rs` | 4 |
| `ptir/ptir_kv.rs` | `pipeline/fire/kv.rs` | 4 |
| `ptir/ptir_rs.rs` | `pipeline/fire/rs.rs` | 4 |
| `ptir/ptir_lease.rs` | `pipeline/fire/lease.rs` | 4 |
| `api.rs` + `api/*` | `inferlet/host.rs` + `inferlet/host/*` | 4 |
| `driver/registry.rs` scheduler handles | `scheduler.rs` | 3 |
| `server/*`, `service.rs`, `telemetry.rs`, `bootstrap.rs`, `store/*`, `inferlet/{program,process,linker,python,sandbox}` | unchanged | - |

## Decision log

- **`pipeline/` over `forward/` for the ex-ptir domain module.** In Gim's call
  (2026-07-10). The lifecycle nouns read naturally (`pipeline::program`,
  `pipeline::instance`, a pipeline's channels), and the WIT `pie:inferlet/pipeline`
  interface owns instantiate/submit. The WIT vocabulary stays fully visible at the
  boundary because the glue files match interface names one-to-one
  (`inferlet/host/forward.rs`, `inferlet/host/pipeline.rs`). "Forward" remains the
  conceptual term for the path itself in docs. Do not flip after Phase 4 lands.
- **Reclaim lives in `store/`, not `scheduler/`.** The batching loop never calls
  it (zero references from scheduler/policy/batch); placement follows what it
  manages (pages), not its trigger seam (forward prepare).
- **Probe under `scheduler/`.** The review's `inference/probe.rs` placement was
  right while `inference/` existed; all consumers are the fire/scheduler path.
  Keep the `profile-fire` feature.
- **`service.rs` stays.** Genuinely shared (scheduler, server, program, linker,
  process actors). Only the vestigial `InferenceService` dies.
- **Token auth is not re-added.** Replaced by public-key auth already in tree;
  `bootstrap::AuthConfig` and the base64 dep are already gone.
- **Session/inbox upward call accepted** (`inferlet/host` -> `server` facade) as
  the one documented exception; revisit only if a second one appears.

## Verification playbook

Per phase: `cargo check -p pie-engine`, `cargo clippy -p pie-engine`,
`cargo test -p pie-engine`, then workspace `cargo check --workspace`.
GPU-dependent phases (3, 4, 5): cuda e2e suite in `bin/pie/tests/` on the 4090
workstation (start `cuda_plain_gen`, then full sweep), plus one live
serve + client inferlet run.

Layering gates (CI-able; each must print nothing):

```sh
# L0 driver imports nothing above it
grep -rn "crate::\(store\|scheduler\|pipeline\|inferlet\|server\|inference\|ptir\|api\)" runtime/engine/src/driver/
# L1 store: only driver below it
grep -rn "crate::\(scheduler\|pipeline\|inferlet\|server\|inference\|ptir\|api\)" runtime/engine/src/store/
# L2 scheduler: only store/driver
grep -rn "crate::\(pipeline\|inferlet\|server\|ptir\|api\)" runtime/engine/src/scheduler/
# L3 pipeline: only scheduler/store/driver
grep -rn "crate::\(inferlet\|server\|api\)" runtime/engine/src/pipeline/
```

(Adjust the lists as phases land; until Phase 5, `inference` still exists and is
allowed only in `inferlet/host` and `bootstrap`.)

## Out of scope

- `interface/ptir` (the IR crate): correctly named, untouched.
- `server/` internals, `inferlet/{linker,python,sandbox}`: no known problems.
- Real authentication hardening beyond the already-landed public-key path.
- Splitting `scheduler/worker.rs` (2.4k lines) into `queue`/`fire`/`membership`:
  worthwhile, but only after a full read; do not do it blind during the move.
