# Pie Architecture

Pie is a programmable inference engine: it runs small user-supplied WebAssembly
programs (*inferlets*) next to the model, giving them direct access to the KV
cache and forward pass. This document describes how the host-side Rust workspace
is laid out and the rules that keep it composable as Pie scales from a single
laptop to a disaggregated cluster.

The same source tree builds two very different deployments:

- **single-node** — one self-contained `pie` binary that does everything
  (prefill, decode, KV cache, scheduling) in one process.
- **distributed** — many workers, each pinned to a role, coordinated by a
  standalone controller, streaming KV tensors to one another over the network.

Nothing in the layout forks between those two cases. The difference is which
components are *present* and whether they talk in-process or over the wire — not
different code paths bolted onto a monolith.

## Root layout

| Path | Crate | Responsibility |
|---|---|---|
| `protocol/` | `pie-schema` · `pie-schema-derive` · `pie-schema-bindgen` | Shared wire/RPC contracts. `pie-schema` remains the **floor** of the dependency graph and now also carries the gateway↔worker edge-session RPC types. See [The protocol floor](#the-protocol-floor). |
| `runtime/` | `pie` | The inferlet runtime: loads `wasm32-wasip2` inferlets, drives the forward pass, owns scheduling and KV-cache bookkeeping. |
| `worker/` | `pie-worker` | The invariant engine assembly layer — boots runtime + driver(s), serves the gateway-facing edge-rpc session API, and selects single-node vs. distributed topology by launch flag. |
| `driver/` | (several) | Backend drivers plus the runtime↔driver IPC mechanism and weight loader. See [Drivers](#drivers). |
| `driver/transport/` | `pie-transport` | The **data plane**: moves KV-cache tensors worker↔worker, peer-to-peer. |
| `controller/` | `pie-controller` | The **control plane**: cluster membership, role assignment, prefill↔decode pairing, and health. |
| `gateway/` | `pie-gateway` | The **edge plane**: the single client-facing host. Terminates client WebSockets and routes each session to a worker the controller picks. Control-plane client only — never touches tensor data. |

Supporting trees that are not part of the host workspace's core dependency
graph: `client/` (client libraries), `sdk/` (inferlet SDKs), `inferlets/`
(example inferlets, built for `wasm32-wasip2`), `benches/`, `tests/`, and
`website/`.

### The protocol floor

Everything that crosses a process or language boundary is described once, in
`protocol/`. It is split into three crates so the schema *contract*, the *macro
that generates code from it*, and the *header it emits for C++* stay separate:

| Path | Crate | What it is |
|---|---|---|
| `protocol/schema/` | `pie-schema` | The wire vocabulary itself: frame types, KV-cache layout, cluster-coordination messages, driver capabilities, and the schema hash. This is the floor — a pure data-struct crate (no tarpc); every component imports its types. |
| `protocol/schema-derive/` | `pie-schema-derive` | The `#[schema]` proc-macro. From one struct definition it derives the rkyv wire format plus a `#[repr(C)]` C-friendly view, so the same schema source is usable from both Rust and C++. |
| `protocol/schema-bindgen/` | `pie-schema-bindgen` | Generates the committed C/C++ header (`pie_schema.h`) from the schema via cbindgen, so non-Rust drivers compile against the same definitions. |

`pie-schema` is the *floor*: it sits at the bottom of the graph and depends on no
runtime sibling. Its single carve-out is a compile-time one — it uses the
`pie-schema-derive` proc-macro to derive its own wire types. `schema-bindgen` is
a build-time tool that consumes `pie-schema` to emit the header; nothing depends
*on* it at runtime.

### Drivers

A driver owns model weights and the forward pass for one backend; the runtime
owns everything above it. The runtime never links a backend directly — it speaks
to whichever driver is configured through a single in-process mechanism, and the
concrete backend driver is linked by the assembly layer (the worker).

| Path | Crate | What it is |
|---|---|---|
| `driver/ipc/` | `pie-ipc` | The runtime↔driver **mechanism**: a POSIX shared-memory ring, an in-process C-ABI vtable, and the rkyv wire helpers. Layered directly on `pie-schema`. |
| `driver/cuda/` | — | Embedded C++/CUDA driver — the native NVIDIA GPU forward pass. |
| `driver/portable/` | — | Embedded ggml-backed driver — CPU and every non-NVIDIA backend (metal / vulkan / …) ggml supports. |
| `driver/dummy/` | `pie-driver-dummy` | A no-hardware driver for tests and CI. |
| `driver/weight-loader/` | `pie-weight-loader` | Loads HuggingFace safetensors into driver-owned buffers. |

The `ipc` crate is the *mechanism* (how runtime and driver exchange frames); the
*vocabulary* of those frames lives one level down in `pie-schema`. Keeping the
two apart is what lets both a Rust and a C++ driver speak the same wire.

## Three orthogonal axes

The refactor exists to keep three independent questions from leaking into one
another. Any given deployment picks one value on each axis, and the axes do not
constrain each other.

| Axis | The question | Values | Where it lives |
|---|---|---|---|
| **backend** | *How* is the forward pass run? | cuda / rocm / metal / vulkan / CPU | `driver/*` only |
| **role** | *What* does this worker do? | prefill / decode / encode | `controller` (role table) |
| **topology** | *Whether* a controller exists | single-node / distributed | `worker` launch flag + controller deployment form |

On the backend axis, the hardware values map onto two embedded drivers, not one
crate per backend: `driver/cuda` is the native NVIDIA path, and `driver/portable`
covers CPU and every other backend (metal, vulkan, rocm, …) through ggml. Adding
a backend means teaching a driver, never touching the controller or transport.

The discipline is that each axis is invisible to the components that don't own
it:

- The **controller is oblivious to the backend** — it never knows or cares
  whether a node runs CUDA or ggml; it routes on role, load, and KV headroom.
- The **transport is oblivious to the role** — it moves pages in whichever
  direction it is told; "prefill" vs. "decode" is a controller concept, not a
  data-plane one. A backend appears to it only as an opaque memory domain on a
  registered region.
- The **driver is oblivious to topology** — it runs a forward pass and exports a
  KV handle whether or not a controller exists above it.

A backend that is single-node only (metal, vulkan) simply does not implement the
transport registration shim; its driver returns "unsupported" and the
data plane never engages. That is an axis value, not a special case.

## The dependency rule

Edges point **downward only**, and `protocol/schema` is the floor. Each band
below depends only on the bands beneath it:

```
  assembly / entry  │  worker  ·  sdk/python-server
                    │      (link a backend driver, wire the topology)
  ──────────────────┼─────────────────────────────────────────────────────────
  planes + runtime  │  controller        transport        runtime
                    │  (control plane)    (data plane)   (inferlet exec)
  ──────────────────┼─────────────────────────────────────────────────────────
  backend + mech.   │  driver/{cuda,portable,dummy}         driver/ipc
                    │  (forward pass — linked by assembly)  (runtime↔driver)
  ──────────────────┼─────────────────────────────────────────────────────────
  floor             │  protocol/schema   ◄── protocol/schema-derive  (proc-macro)
                    │  (shared types)        protocol/schema-bindgen  → pie_schema.h
```

`controller` and `transport` are siblings of `runtime` under the worker — they
do **not** depend on the runtime, and there is **no edge between the two
planes**. The concrete backend driver is linked by the assembly layer (the
worker, or `sdk/python-server` for the Python build), never by the runtime,
controller, or transport. Every path funnels down to `protocol/schema`.

Concretely:

- `protocol/schema` depends on no runtime sibling. Its one edge is a
  *compile-time* one — the `pie-schema-derive` proc-macro that derives its own
  wire types — so at runtime it is the single source of truth that imports
  nothing back.
- `controller` depends on `pie-schema` and **nothing else** — never on the
  runtime, transport, or any driver.
- `transport` consumes a driver-exported KV handle through a trait, and shares
  the KV layout descriptor via `pie-schema`; it never imports a driver crate.
  Driver and transport meet *only* through the handle type defined in the floor.
- `controller` and `transport` exchange only the small coordination metadata
  both already import from `pie-schema`: the control plane decides, the data
  plane moves bytes, and neither calls into the other.
- backend drivers (`cuda` / `portable` / `dummy`) are depended on **only** by the
  assembly layer; the runtime reaches them through `driver/ipc`, not by linking
  them directly.

Because the floor's only dependency is its own proc-macro, the same `pie-schema`
types are linked by a Rust controller and a C++ CUDA driver without either
depending on the other.

These rules are not just convention: `tools/arch-audit` (`pie-arch-audit`) parses
the workspace manifests in CI and fails the build on any upward edge, dependency
cycle, or backend driver depended on from outside the assembly layer.

## Control plane vs. data plane

The controller and the transport are mirror images split along the
control-plane / data-plane boundary. The controller decides *what, when, and to
whom*; the transport moves the bytes. **The controller never touches a large
tensor, and the transport never makes a policy decision.**

```
worker  → controller :  register + heartbeat + load    (push, periodic, small)
controller → worker  :  placement / pairing decision    (per request, small)
worker  ↔ worker     :  KV tensors, peer-to-peer         (via transport, large)
```

The disaggregated prefill→decode flow makes the split concrete:

1. A prefill worker and a decode worker each **register** with the controller
   and push periodic heartbeats and load.
2. For a request, the controller **pairs** them — "prefill P3's KV goes to
   decode D1" — and then **steps out** of the path.
3. The prefill worker streams its KV cache **directly** to the decode worker
   through `pie-transport`, peer-to-peer. **The KV tensors never transit the
   controller.**

Keeping the planes separate is what lets the controller stay light — soft state
only, foldable in-process on a single node — while the transport stays fast,
moving registered buffers with zero-serialization handle passing.

## Topology: single-node and distributed

Topology is the axis that decides whether a controller exists as a separate
thing at all. The controller is written once as a library with two shells, so
neither the worker nor the controller logic forks:

- **single-node** — the worker constructs the controller in its own address
  space and calls it directly. Routing is trivial (the one node is always the
  target), pairing is same-node, and there is no network hop. The data plane
  uses a same-node device-to-device copy; RDMA never engages.
- **distributed** — the controller runs as its own process, reachable over the
  network. Workers register and pair through control-plane RPC, then move KV
  tensors P2P over the transport's network engine.

The worker selects the form by launch flag (single-node default vs.
`--role=… --controller=addr` for distributed); the controller trait and the
transport interface are identical underneath.

### Maturity: single-node ships, distributed is scaffolded

**Single-node is the supported, exercised topology** — the shipping path today.
The embedded in-proc controller, the same-node `local` transport engine, and the
monolithic all-roles worker are the production configuration.

**Distributed mode is built but not yet production-hardened.** Its seams exist
and compile: the control-RPC (length-prefixed serde over TCP), `RemoteController`,
the standalone `run_as_process` serve loop, and the cross-node `nixl` data-plane
engine. The `nixl` engine is a real implementation over NVIDIA's NIXL C-API — but
gated behind `feature = "nixl"` and **off by default**, so the on-device build
ships NIXL-free with only the `local` engine and carries no distributed-only
weight. What remains is the control-plane *orchestration* hardening — the
distributed path needs a follow-on pass before it can be relied on:

- **worker heartbeat-pump** — a registered worker must periodically `report()`;
  without it the controller ages the worker out of routing (~6s). (Single-node
  doesn't need this: its one local worker never times out.)
- **socket timeouts** on the control-RPC client/server (no unbounded blocking).
- **RPC reconnect** — recover a dropped control connection instead of failing.
- **accept-loop resilience** — the serve loop must survive a bad connection
  rather than tearing down on the first error.
- **clean deregister + state compaction** — workers leaving should be reaped and
  the soft state compacted.
- **thread-pool cap** — the standalone process spawns a thread per connection;
  bound it.
- **routable `control_addr`** — workers must advertise an address peers can
  actually reach (not a loopback placeholder).

Until that pass lands, treat distributed mode as next-phase scaffolding: **run
single-node in production.**

## Further reading

- `protocol/` — the shared wire vocabulary (`schema`), its `#[schema]` macro
  (`schema-derive`), and the C++ header generator (`schema-bindgen`).
- `driver/ipc/` (`README`) — the runtime↔driver mechanism and cross-language
  handshake.
- `controller/` and `driver/transport/` (`README`s) — per-crate scope and the
  three-axes table applied to each plane.
