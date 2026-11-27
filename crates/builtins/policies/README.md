# Built-in policies

PLEX 1.1 policies built into the pie binary, named in the configuration by
the names here. A policy of your own is named by its `.wasm` path.

## Worker: `[runtime] worker_policy`

A worker runs one policy. Every one but `noop` does what `default` does too:
the cache a process leaves stays on the device alone, its decoded tail goes
before its prompt, and starts wait while the pools cannot hold the running
processes to their ends. Hints are the process's props
(`x-plex-hint-<name>` on the compat routes).

| Name | Beside that | Hints |
|---|---|---|
| `default` | nothing: first come, first served | |
| `deadline` | the soonest due runs first, and those without a deadline after | `deadline-ms`: due that long after arriving |
| `fair` | the tenant that has had the least running time runs first | |
| `priority` | the larger priority runs first; 0 without one. It is the client's alone: nothing a cluster policy notes is read | `priority` |
| `keep-session` | a completed process's KV is kept above the rest, on any tier, then dropped; a session keeps its latest turn alone | `keep-ms` |
| `noop` | writes no rows | |

`deadline`, `fair` and `priority` let waiting starts go in their own order.

## Cluster: `[cluster] cluster_policies`

Every one but `noop` does what `default-cluster` does too: a session's turns
go where its others are, a launch waits while every worker is full, a launch
that can run on one worker alone goes there, every pool keeps a copy of what
the workers cache, and every worker that serves others' passes helps. Each
writes the same columns for it, and no two policies may write one column, so
no two of these can be loaded together.

| Name | Beside that | Hints |
|---|---|---|
| `default-cluster` | nothing | |
| `pin` | a launch goes first to the worker it names | `worker` |
| `drain-migrate` | the turns on a draining worker move to the others, spread over the least loaded | `worker` |
| `rebalance` | while a worker runs two more turns than another, its youngest moves there, each at most once | `worker` |
| `replicate` | every run some worker caches is kept on two decode workers | |
| `noop` | writes no rows | |

**What moves.** `drain-migrate` and `rebalance` move a turn that has sent no
output, which runs again from its input, and also one that has sent some and
can go on from it. A turn that names handles or is tied to its worker stays,
as does any the gateway refuses; none is asked twice.

**What `replicate` promises.** No event says a run was cached, so it looks
once a second, at what the controller reads once a second: a run is asked for
up to about two seconds after it is cached. The copy's tier, and whether it
stays, is the receiving worker's retention to say.

## One of your own

A policy that orders processes by a deadline. It is loaded in place of
`default` and writes `schedule/priority` alone, so what `default` does is not
done.

```toml
[package]
name = "deadline"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib"]

[dependencies]
plex = { path = "<pie>/crates/plex-sdk" }
```

```rust
use plex::prelude::*;

#[derive(Default)]
struct Deadline;

impl Policy for Deadline {
    fn plan(&mut self, _now: Mark) {
        let rows: Vec<(ProcessId, i64)> = snapshot::get_processes()
            .into_iter()
            .filter_map(|p| {
                let ms = hint(&p.id, "deadline-ms").int()?;
                let due = (p.arrival / MILLISECOND) as i64 + ms.max(0);
                Some((p.id, i64::MAX - due))
            })
            .collect();
        schedule::priority::clear();
        schedule::priority::set_list(&rows).ok();
    }
}

plex::export!(Deadline);
```

```sh
cargo build --release --target wasm32-unknown-unknown
wasm-tools component new target/wasm32-unknown-unknown/release/deadline.wasm -o deadline.wasm
```

```toml
[runtime]
worker_policy = "/path/to/deadline.wasm"
```

The component imports only the columns it calls, and an import is the
authority to write. SIGHUP swaps the running policy for the one the file
names.
