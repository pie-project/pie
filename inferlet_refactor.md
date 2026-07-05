# Inferlet Module Refactor Plan

Consolidate the five scattered inferlet-related modules in `runtime/engine/src/`
(`program.rs`, `linker.rs`, `instance.rs`, `process.rs`, `policy.rs`) into a
single `src/inferlet/` module, and fix the naming debt around
"process" vs "instance" along the way.

## Current state

The five modules together implement the full lifecycle of one inferlet, but
they sit flat at the crate root next to unrelated subsystems (inference
scheduling, driver, working sets, server):

| Module | Lines | Role |
|---|---|---|
| `program.rs` + `program/` | 701 + 520 | Program service actor: metadata, caching, registry download, compilation |
| `program/python/` | 1,574 | CPython runtime modules + memory snapshot pipeline |
| `linker.rs` + `linker/` | 330 + 1,267 | Linker service actor: owns the wasmtime `Engine`, instantiation, dynamic linking |
| `instance.rs` + `instance/` | 276 + 159 | `InstanceState`: per-instance `Store` data (WASI ctx, sandbox state, scratch dir, resource maps) |
| `process.rs` | 611 | Per-instance lifecycle actor: spawn/attach/terminate, registry, admission control |
| `policy.rs` | 346 | `FsPolicy` / `NetworkPolicy` parsing and checks |

### Why grouping is safe

Import analysis shows the cluster is internally dense but externally narrow.
What the rest of the crate (and external crates) actually touches is a handful
of types and actor entry points:

- `policy`: used only by `bootstrap` and `linker`. Pure leaf.
- `linker`: used by `bootstrap`, the cluster itself, and `program/python/snapshot`.
- `instance`: 15 `api/*` files use it, but only because `InstanceState` is the
  `Store<T>` data type that host bindings are implemented on.
- `process`: widest surface (`inference/*`, `api/session`, `server`), but only
  through `ProcessId`, `ProcessEvent`, and the module-level actor functions.
- `program`: used by `server` handlers and `bootstrap`.

External crates (`worker`, `bin/pie`, `sdk/python-server`) reference
`pie_engine::program` (15 uses) and `pie_engine::process` (9 uses). All
in-workspace, so path updates are a mechanical one-shot.

### Existing problems worth fixing (not just moving)

1. **Two words for one entity.** A running inferlet is called a *process* on
   the client wire protocol (`attach_process`, `signal_process`, `process_id`)
   and an *instance* on the guest WIT surface (`system.wit`'s `instance-id`,
   which returns the `ProcessId` verbatim). Inside the engine both `process.rs`
   and `instance.rs` exist, and "instance" additionally collides with
   wasmtime's own `Instance` type, forcing the `Instance as WasmInstance`
   alias in `linker.rs`.
2. **`InstancePolicy` is defined in `linker.rs`**, not `policy.rs`, even
   though it is just the fs+network policy bundle.
3. **`crate::policy` vs `crate::inference::policy`** name collision: sandbox
   policy vs the batch scheduler's fire policy.
4. **`program/python/` points upward.** `snapshot.rs` (1,380 lines, the
   largest file in the cluster) imports `linker` and `instance`, i.e. a child
   of `program` depends on its parent's siblings. Python support is an
   instantiation-time subsystem used by both `program` (install) and `linker`
   (instantiate), not a part of repository management.

## Target structure

```
src/inferlet/
├── mod.rs           docs + re-exports: ProcessId, ProcessEvent, ProcessCtx,
│                    ProgramName, Manifest, InstancePolicy
│
│  ── artifact (at rest) ──
├── program/
│   ├── mod.rs       service actor + public API + ProgramName
│   ├── manifest.rs
│   └── repository.rs
│
│  ── instantiation ──
├── linker/
│   ├── mod.rs       linker actor, instantiate()
│   └── dynamic.rs   (was linker/dynamic_linking.rs)
│
│  ── execution ──
├── process/
│   ├── mod.rs       lifecycle actor: spawn/attach/terminate, registry, admission
│   ├── ctx.rs       ProcessCtx (was InstanceState) + OutputMode
│   └── output.rs    LogStream (was instance/output.rs)
│
│  ── supporting subsystems ──
├── sandbox.rs       FsPolicy + NetworkPolicy + InstancePolicy
└── python/          (promoted from program/python/)
    ├── runtime.rs   shared-module loading
    └── snapshot.rs  memory snapshot pipeline
```

Everything else stays where it is. In particular:

- **`api/` stays top-level.** It is the bridge layer between guest calls and
  engine services (`working_set`, `inference`, `ptir`, `messaging`), not part
  of the inferlet execution machinery.
- **`bootstrap.rs`, `server/` stay.** They orchestrate; they are callers.
- **`ProcessEvent` stays in `process`.** It is the wire protocol event type,
  but the producer is the process actor; `server` only serializes it.

## Design decisions

### 1. `process` is the single word for a running inferlet

The OS metaphor (spawn, terminate, attach, signal, stdout/stderr, registry,
admission) is deliberate and accurate, and the word is already baked into the
client protocol and external crates. Renaming it would be a protocol revision,
not an internal refactor.

Rejected alternatives: `task` (collides with tokio tasks), `session` (client
sessions and `session.wit` exist), `worker` (pie-worker crate), `job` (batch
connotation), `run`/`exec` (awkward call sites, diverges from wire protocol).

### 2. `InstanceState` → `ProcessCtx`, folded into `process/`

`instance.rs` does not describe a separate concept; it is the host-side
execution context of one running process (the `T` in `Store<T>`). In OS terms
`process/mod.rs` is the task lifecycle and `ctx.rs` is the address
space / fd table. The rename unifies the execution vocabulary into one word
family (`ProcessId`, `ProcessEvent`, `ProcessCtx`, `process::spawn()`) and
frees "instance" to mean only wasmtime's `Instance`, removing the
`WasmInstance` alias. Host bindings become `impl model::Host for ProcessCtx`.

Rename scope: 23 files, mechanical.

### 3. `policy.rs` → `sandbox.rs`, absorbing `InstancePolicy`

`InstancePolicy` moves from `linker.rs` to live with the policies it bundles.
`linker` is left purely about assembly. The rename also removes the ambiguity
against `inference::policy` (the scheduler's fire policy):
`inferlet::sandbox` vs `inference::policy`.

### 4. `python/` promoted to a direct child of `inferlet/`

It is a cross-cutting subsystem used at install time (`program`) and at
instantiate time (`linker`). Promotion fixes the only upward-pointing
dependency edge in the cluster: `python ← program, linker`.

### 5. Narrow public surface via `inferlet/mod.rs`

`lib.rs` currently exposes every module as `pub mod` with no boundary.
After the move, outsiders use `inferlet::ProcessId`, `inferlet::ProcessCtx`,
`inferlet::process::spawn(...)`, etc.; internals such as `linker::dynamic` and
`python::snapshot` become `pub(crate)` or tighter.

## Migration plan

Split the move and the redesign into separate commits so each diff stays
reviewable:

1. **Move only.** Relocate the five modules under `src/inferlet/`, shrink
   `lib.rs`, fix `crate::X` → `crate::inferlet::X` paths crate-wide.
2. **External paths.** Update `pie_engine::{program,process}` references in
   `worker`, `bin/pie`, `sdk/python-server` (~24 sites).
3. **Redesign commit(s).**
   - `policy.rs` → `sandbox.rs` + move `InstancePolicy` out of `linker`.
   - `instance.rs`/`instance/output.rs` → `process/ctx.rs`, `process/output.rs`;
     rename `InstanceState` → `ProcessCtx`.
   - Promote `program/python/` → `inferlet/python/`;
     rename `linker/dynamic_linking.rs` → `linker/dynamic.rs`.
   - Tighten visibility in `inferlet/mod.rs`.

## Out of scope

- **WIT `instance-id`.** Full vocabulary unification would rename it to
  `process-id`, but `pie:inferlet@0.2.0` is a published guest interface that
  SDKs depend on. Defer to a future WIT version bump; it is an interface
  change, not an engine refactor.
- **`api/` reorganization.** Its coupling to `ProcessCtx` is type-level only;
  restructuring the bridge layer is an independent effort.
