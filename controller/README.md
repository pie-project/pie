# controller

`pie-controller` — Pie's cluster-coordination **control plane**: a worker
registry that routes requests, pairs prefill↔decode, and tracks liveness/load.

It is **control plane only**. It moves small coordination metadata — worker ids,
roles, addresses, load — and **never touches tensor bytes**. KV blocks and
activations travel on the data plane (`pie-transport`), which this crate has no
knowledge of.

## One trait, two deployments

`Controller` is the seam the worker wires to. Two impls back it, selected by
the worker's `pie serve` flag:

- **on-device** (`--single-node`): the worker constructs an `InProcController`
  in its own address space and calls it directly — no network, no serialization.
- **distributed** (`--role=<…> --controller=<addr>`): the worker dials a
  `RemoteController`, which frames each call over a socket to a standalone
  controller process (`run_as_process`, backed by the same `InProcController`).

```rust
pub trait Controller: Send + Sync {
    fn register(&self, worker: WorkerInfo) -> Result<WorkerId>;
    fn report(&self, worker: WorkerId, load: LoadState) -> Result<()>;
    fn route(&self, req: &RequestMeta) -> Result<Placement>;
    fn pair(&self, req: RequestId) -> Result<(WorkerId, WorkerId)>;
}
```

```bash
cargo test -p pie-controller --all-targets             # lib + RPC round-trip
cargo run  -p pie-controller -- --listen 0.0.0.0:7000  # standalone process
```

## Minimal start (YAGNI)

Per the ratified spec the coordinator is a **registry + round-robin `route`**
with **pushed** load (`report` — workers push, the controller never polls; soft
state, survives restart). `pair` is the trivial **same-node** decision
(monolithic workers). Real PD pairing, least-loaded routing, and placement
scaling are deferred — the trait keeps them swappable with worker code
untouched. `src/pairing.rs` holds the (kept-but-unextended) prefill↔decode A↔B
machinery for that future.

## Control-RPC

`src/rpc.rs` is the minimal concrete control channel: length-prefixed
(`u32` BE) MessagePack (rmp-serde) frames over blocking `std::net` — low-rate
metadata only (register / report / route / pair). `RemoteController` dials it;
`run_as_process` serves it (one thread per connection + a liveness ticker).
Swappable to rkyv later behind the trait with no logic churn.

## Three orthogonal axes

| axis         | question                       | where it lives                       |
|--------------|--------------------------------|--------------------------------------|
| **role**     | prefill / decode / encode?     | `Role` (recorded per worker)         |
| **topology** | single-node vs distributed?    | the two `Controller` impls           |
| **backend**  | cuda / portable / dummy?       | *not here* — invisible to control    |

## Dependency direction

Edges point **downward only**. The controller depends on `pie-schema` for the
shared vocabulary (`pie_schema::cluster`) and nothing else — never on the
runtime, transport, or driver crates. The control-RPC message envelope
(`ControlRequest` / `ControlResponse`) is local in `src/protocol.rs`.

## Status

Implemented and unit-tested (registry + round-robin routing, pushed load,
heartbeat grading, same-node pairing, and an end-to-end control-RPC round-trip
over a real socket). Wired into the worker: `pie serve --single-node` embeds the
controller in-proc; `--role/--controller` dials a standalone process. Deeper
role→stage gating and least-loaded/PD routing are deferred follow-ons.
