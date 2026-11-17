# PLEX 1.1 on pie: design pass

Read-only pass, 2026-11-17. Sources:
- plex `d32445219`: `wit-1.1/` (the contract), `docs/plex-1.1-spec.md`, `host-1.1/src/{host,runtime,tables,py}.rs`, `ports-1.1/` (README, UPSTREAM, RESULTS, `python/plex_port/engine.py`, `e2e/conform.py`, `probes/conform`).
- pie `origin/memory-redesign` = `c3b24dd77`. Every `file:line` below is at that commit.

**Headline.** pie does not need a port in the vLLM/SGLang sense. It already *is* the PLEX 1.0 host, natively, on both planes, with mechanisms 1.1 assumes only pie has:
- a worker plane: `crates/runtime/src/policy/*` (its own wasm host, `program.rs` 642 lines, `service.rs` 1789);
- a cluster plane in the controller: `crates/controller/src/cluster/host.rs` 2619 lines, enacted by the gateways through `crates/gateway/src/policy/service.rs`;
- the cluster-draft verbs 1.1 kept as columns: offload to executors (helpers), migration (`placement.migrate`), replication to pools and host tiers.

So the 1.1 work is mostly a **re-skin**:
- replace pie's 1.0 wasm host (`plex:policy@1.0.0`, WASI 0.3, manifest/grants/capabilities) with `host-1.1` embedded as a Rust crate;
- re-point the existing report sites and the existing table read points at it;
- add four real mechanisms:
  1. `retention/tiers` membership (demote now; keep off a tier);
  2. `paused` by `client`;
  3. cooperative `terminate` that the client sees;
  4. migration of a started compat turn after output, by continuation.

---

## 1. Integration shape

### 1.1 Embed `host-1.1` as a Rust crate, no Python

Yes. `host-1.1` is a plain Rust library (`crate-type = ["cdylib","rlib"]`, `python` feature off by default). `Host` is the whole port surface:
- reports (`arrived` … `tick`, `host.rs:542-846`);
- enactment (`version()` `host.rs:535`, `poll()` `host.rs:855`);
- its own policy thread (`start()` `host.rs:480`).

pie calls it directly. Nothing in `py.rs` is needed.

**Where it lives.** Vendor it as `crates/plex-host` (src + `wit-1.1/` as `crates/plex-host/wit`), with a `PROVENANCE`/sync note pointing at plex `host-1.1@<commit>`.
- Why vendor: plex is private and pie builds standalone, and pie already hosted the 1.0 WIT (`crates/policy/wit`).
- Bump wasmtime 47.0.3 to pie's workspace 49.0.2 (`Cargo.toml:78`). The `bindgen!` and `LinkOptions` API is the same shape.
- Keep `imbl`, `serde_json`.
- It creates its own `wasmtime::Engine` and 1 ms epoch thread (`runtime.rs` `engine()`), separate from the inferlet engine. That is fine, and avoids sharing pie's async WASI 0.3 engine: 1.1 policies are `wasm32-unknown-unknown` components with no WASI.

**Three small upstream additions to `host-1.1`** (do them in plex, then vendor):
1. **Typed delta.** `Host::poll_delta() -> Delta`, the same content as `poll()` but typed (`Vec<(K, Option<V>)>` per column, defaults, terminate, notes). This saves a JSON round-trip on pie's enact path; `poll()` stays as a thin wrapper for py.
2. **Gate support.** `Host::arrived` returns an event sequence number, and `Host::heard() -> u64` gives the highest sequence every policy has heard, counting only events taken by a completed round. pie's gateway gate already waits for "every program has heard this arrival" (`controller-api` `GatewayDecisions.heard`, gateway `service.rs:301 pass`, `:352 next`). The 1.1 spec states the same gate in §2. The host does not expose it today.
3. **A config switch for tiers this worker lacks.** Only needed if pie wants the host, rather than the port, to judge `retention/tiers` rows that name only absent tiers. Optional: the port can call `refuse()` as `plex_port/engine.py judge_tiers` does.

### 1.2 Threads and per-step calls (worker)

Today a worker runs one `pie-policy` thread (`runtime/src/policy/service.rs:31 start`, `:155 run`). Every report site calls `policy::report(Event)` (`policy.rs:165`), and the thread receives events over an mpsc channel. Enactment installs whole tables into globals that the hot paths read under an `RwLock` (`policy/schedule.rs` `CURRENT`, `policy/retention.rs` `CURRENT`).

Keep that shape and swap what is behind it:

```
any runtime thread                 pie-plex (port) thread                plex-policy thread (host-1.1)
─────────────────                  ──────────────────────                ─────────────────────────────
policy::report(Event) ──mpsc──▶    translate Event → Host::arrived/      rounds: snapshot, call each
  (unchanged call sites)             started/paused/…  (µs, lock+append)   policy (50 ms deadline),
                                   every ≤2 ms or after a batch:           commit, bump version
fire path: per-process atomics       if host.due(): host.tick(counters)
  (rows written, tokens sampled)     if host.version() changed:
                                       poll_delta() → install tables
hot paths read tables as today:        (schedule::install, retention::install,
  schedule::key/held, retention::      tiers, terminate, discard, notes→attrs)
  Installed, offload::rule
```

- **No per-step call from the scheduler.** pie's "step" is a fire, and its policy reads already happen there lock-free:
  - `scheduler/frame.rs:553` (admission earmark order) and `:749` (lane order);
  - `inferlet/process/ordered.rs:141` (gate grant);
  - the planner's `proc.key`/`proc.held` (`planner/residency_planner.rs:562,677`) and `reschedule` (`:892`).

  1.1's "one atomic version read per step" is met by the port thread doing it every ≤2 ms. Enactment never runs on a fire thread. That is what the 1.0 service does today.
- **Counters for `tick`.** `generated-tokens` (required by `snapshot.process`, and `conform`'s `cut` needs it), `prompt-tokens` and `cached-tokens` come from a per-process `Arc<AtomicU64>` triple that the fire path bumps. `FireContext::note_cached` (`inferlet/host.rs:31` → `process/ctx.rs:234`) already exists for cached rows. Add `note_rows(new_rows)` and `note_sampled(n)` beside it in `pipeline/fire.rs` (near `:3781`, where `Cached` is reported). Then `first_token` is the first `note_sampled`.
- **Cluster.** In the controller, `cluster/host.rs:332 run` becomes a port thread around a cluster `Host`. It turns gateway `ProcessEvent`s (`controller-api lib.rs:260`) and directory views into reports. It turns `poll_delta()` into the existing level-triggered `GatewayDecisions` per gateway (`cluster/host.rs:220 decisions_of`), which gateways already watch and apply (`gateway/src/policy/service.rs:452 apply`).

### 1.3 What is deleted

The 1.0 policy machinery that `host-1.1` replaces:
- `runtime/src/policy/{program,manifest}.rs` and most of `service.rs` (`Registry`, `Snapshot`/`Plane`);
- `controller/src/cluster/{program,manifest}.rs` and most of `host.rs`;
- `crates/policy` (the 1.0 SDK; 1.1 uses plex `sdk-1.1`);
- 1.0 verbs with no 1.1 column: `install-batch`, `install-scale`, transfer shares, acceptance, cluster `install-retention`, `publish` (replaced by `notes`), `discard` (replaced by `tiers = []`).

This is about 5–6k lines out. The 25 builtin 1.0 policies (5 of them test probes) (`crates/builtins/policies/*`) must be re-ported to `sdk-1.1`, or retired in favour of plex `policies-1.1`.

---

## 2. Mapping (pie `origin/memory-redesign`)

Legend: **E** exists, **S** small change (<~150 lines), **N** new mechanism, **X** cannot.

### 2.1 Worker reports (into a worker `Host`)

| 1.1 report | pie source | Mark | Note |
|---|---|---|---|
| `arrived(id, session, tenant, worker, prompt_tokens, max_tokens, hints)` | `inferlet/process.rs:882` `Event::Arrived(pid, Arrival{tenant,session,props})` | E | Process id = the client pid (`schedule::plex_id`, `policy/schedule.rs:78`; a restart keeps it). `worker = Some(self)`, so the state is `waiting`. Props become hints; `max_tokens` from the `MAX_TOKENS` prop (`gateway/ingress/compat.rs:125`). |
| `started(id)` | `inferlet/process.rs:485` (execution admitted) | E | |
| `paused(id, host\|policy)` | `planner/report.rs:35` `Event::Paused(pid)` in `report_evicted` | S | Add `by`: `proc.held` gives `policy`, else `host`. The `Proc` is in hand at `report.rs:29`. |
| resumed = `started(id)` | `planner/report.rs:152` `Event::Resumed` | E | |
| `paused(id, client)` / `started` | `inferlet/host/session.rs:15` `idle_while` (used at `:205`, `:238`, `receive`/`receive_blocking`) | S | Report `Paused(pid, Client)` on entry and `Resumed` on exit. Optionally wrap outgoing wasi-http in the linker for tool calls (§3.5). |
| `first_token(id)` | none | S | First `note_sampled` (§1.2). |
| `exited(id, outcome, prompt, cached, generated)` | `inferlet/process.rs:1055` | E | Counters from the per-process atomics. |
| `tick(counters)` | `pipeline/fire.rs:3781` (`Cached`), plus new `note_rows`/`note_sampled` | S | |
| `hint(Process, …)` | `inferlet/host/session.rs:97` (`set-props` → `PropsChanged`) | E | |
| `hint(Session, …)`, `closed(session)` | `policy.rs:158,162` (from `gateway/ingress/plex.rs:14,32`, fanned out via `worker-api plex_session` `:61`) | E | |
| `retained(ws, worker)` | `store/kv.rs:1735` `Event::Kept(owner, tokens)` | E | |
| `evicted(ws, worker, by, gone)` | `store/kv.rs:684` `Event::CacheEmptied(owner, by)`; handles `handle/table.rs:128,163,220` `HandleGone` | E | |
| `put_working_set(id, parent, owner, session, tokens, idle, copies)` | `store/kv.rs:688 cached_working_sets()` via `policy/service.rs:469 Cache.list`; resting handles `handle/table.rs:139` | S | `CachedWorkingSet` (`kv.rs:344`) lacks per-tier pages. Add device/host/disk page counts so `copies` gives the highest tier. `parent` is `None`, as on the other ports (spec §12 "shared working sets"). |
| `put_worker(id, draining, tiers, running, waiting, step_ns)` | `policy/service.rs:1265-1300` (running/waiting/`step-ms` from `scheduler::last_batch_us`, `capacity-tokens`/`used-tokens`, `swap-*`), `Cache.memory/swap` (`:469`) | E | Re-expressed as rooms: device, host (swap), disk (`disk_pool`). Periodic from the port thread. |
| `attr(…, "pie/…")` | `policy/pressure.rs:277` `TierPressure`; `offload/prefill.rs:230,249` `Offloaded`/`OffloadFailed`; `restarts`, `folded-tokens`, `recurrent-slots`, `fire-tokens` (`service.rs:1183-1324`) | E | All become `get-attr` values under `pie/`. Tier pressure is no longer an edge: the 1.0 default responder (`service.rs:413 respond`) stays as host behaviour. |
| `note(writer, subject, name, value)` | cluster → worker: `PlacementView.props` → `cluster/<name>` props (`controller-api lib.rs:299`); worker → cluster: `StateReport.published` (`:182`) | E | Carry `notes_out` in both directions over these existing channels. |
| `refuse(column, key, tokens, why)` | none | S | Called by the enactors below. |

### 2.2 Worker enactment (out of the worker `Host`)

| Column / verb | pie mechanism | Mark | Change |
|---|---|---|---|
| `schedule/priority` | `policy/schedule.rs` `Table.keys`, read by `ordered.rs:141` (admission and bind gates), `frame.rs:553,749` (earmark, lane order), planner `proc.key` (`residency_planner.rs:562,677`) feeding `Rank` (`planner/rank.rs`: victim order and restore order) | E | Build `schedule::Table` from the column (rows + default), then `schedule::install` (`schedule.rs:67`: `regrant_admission`, `reschedule`, `notify_schedule_changed`). This covers all four decisions in spec §4.1: admit, run order, victim lowest first, resume highest first. |
| `schedule/hold` | `schedule.rs` `held()` = key below floor; gate filter `ordered.rs:141`; planner `pause_held` (`residency_planner.rs:984`, `pause` `:997`), `vacate_held_seats` (`:907`) | S | Replace `floor: Option<i64>` with `held: HashSet<ProcessId>` plus a default bool. Every reader goes through `schedule::held(pid)`, so readers do not change. A running hold on a worker with no host or disk tier (`can_pause()` false, `:979`) leads to `refuse(hold, p)`. |
| `retention/priority` | `policy/retention.rs` `Table.steps` (owner → `(upto, key)`, highest cover wins); `Installed` read by `store/kv.rs:932 reclaim_order`, `:981 prepare_demote`, `:1130 prepare_spill`, `:755 reclaim_droppable`, `store/rs.rs:1026-1084`; `split_at_steps`; holders' `holder_key` (`residency_planner.rs:1232`) | E | `prefix{ws, tokens}` maps one-for-one to `(owner, upto)`; a working-set id is the owner pid or handle id. Install, then `planner::rekey_holders`, as `service.rs:338` does. **Caveat:** pie only *moves down* cache keyed above `FLOOR = 0` (`retention.rs:17 above_floor`). At or below 0 it drops. 1.1's default priority is 0 = "the engine's order", so under 1.1 defaults pie drops idle cache instead of demoting. That is legal ("the engine's own order") but differs from vLLM and SGLang. Decide: shift the floor below `i64::MIN+1`, or keep it and document it. |
| `retention/tiers` | none as membership. Today pie decides the tier by key plus room (`kv.rs:981 prepare_demote`: device → host, else disk; `:1130 prepare_spill`: host → disk), driven by planner sweeps (`planner/pool.rs:409,421`) | **N** | Extend `retention::Keys` with `tiers_at(owner, at) -> TierSet`. (a) `prepare_demote`/`prepare_spill` skip a group whose set lacks the target tier and drop it instead. (b) A **demote-now sweep** (port-triggered on row change, planner-executed): cache whose row lacks `device` goes host (or disk) without waiting for pressure. (c) `[]` means drop now: `kv.rs:718 discard_cache` / handle discard (`handle/table.rs:155`), and also at keep time (`kv.rs:1735`) so a row written while the process ran drops it at exit. (d) A row that names only tiers the worker lacks leads to `refuse`. (e) `[device]` on host-resident KV comes back at admission (`supply` promotes), not ahead of a hit. That is allowed (§12), and it is reported as an attr `pie/promotes-ahead=false`. |
| `terminate` | `inferlet/process.rs:647 terminate_by(pid, Err(reason), By::Policy)`; gateway `session.rs` terminate | S | Works today, but the wasm is aborted (`process.rs:1043 handle.abort()`), so the compat client gets an `error` event, not a reply. conform needs `finish_reason` `abort` and the reason in the choice. See §4. |
| notes in | `host.note(…)` from cluster props | E | |
| alarm | inside `host-1.1` | E | |

### 2.3 Cluster reports (into the controller's cluster `Host`)

| 1.1 report | pie source | Mark | Note |
|---|---|---|---|
| `arrived` (`worker=None`, pending) | gateway `policy/service.rs:249 admit` → `ProcessEvent::Arrived` (`controller-api lib.rs:262`) | E | Id `<gateway>.<req>` (`cluster/host.rs:101`). `prompt_tokens` is approximate from text blocks (`PROMPT_BLOCK_BYTES`, `TOKENS_PER_BYTE`). |
| `dispatched(id, w)`, again on a move | `service.rs:382 dispatched`, `:418 moved` | E | |
| `started` | none at the gateway | S | Report at first output (`turn.emitted`, gateway `session.rs`), as the open item in spec §12. Better: the worker sends `Started` over `push_tokens` or `report`. |
| `paused` at cluster | none | S (optional) | Workers could forward it; the spec does not require it. |
| `exited` | `service.rs:435` | E | |
| `closed`, session hints | `service.rs:396,412` | E | |
| `put_worker` | `WorkerStatus` (`controller-api lib.rs:85`: running, waiting, capacity_tokens, used_tokens, draining); pools via `StateReport.pool` (`:182`) | S | Add host/disk rooms and `step_ns` to `WorkerStatus`. A pool reports `tiers = [host(, disk)]`, no device, `step = None`. |
| `remove_worker`, `worker-changed` | `ProcessEvent::Workers`, directory forget | E | |
| `put_working_set` + `copies` | directory groups (`GroupView{keys, sites, disk_sites}`, `controller-api`; `cluster/host.rs:1012 apply_state`) | S | A group (a run kept whole from token 0) becomes a working set: id = hex(last key), tokens = keys × page, copies from sites/disk_sites. **Gap:** a run has no owner. Add the owner (gateway req, or session) to `StateReport.runs_added`, so a process's working set at the cluster is its run and `replication` rows can name it. Pending processes' `cached-tokens` stay text-block estimates. |
| `refuse` | none | S | |

### 2.4 Cluster enactment

| Column | pie mechanism | Mark | Change |
|---|---|---|---|
| `placement/workers`: pending | `PlacementView.worker: Option<WorkerId>` (`controller-api:299`); gateway dispatch `session.rs:281`; hold at gate = gate schedule floor (`service.rs:352 next`) | S | `PlacementView.workers: Vec<WorkerId>`, best first. Dispatch tries in order (the first that takes it). `[]` means held at the gate (reuse the `next` hold path). No row means the router's order. |
| `placement/workers`: started, by exclusion | `service.rs:482 move_misplaced` (target ≠ at) → `gateway/src/lib.rs:316` → `session.rs:506 migrate` (restart; refused once `emitted`) | S + N | Move only when `!list.contains(at)` (exclusion, not ranking), to the first in the list that takes it. After output: new continuation migration (§3.2). If neither is possible, `refuse(placement/workers, p, not-supported)`. |
| `placement/helpers` | `PlacementView.executors` → `LaunchProcess.executors` → `process.rs:583 executors::set` → `offload/prefill.rs:176 before_pass` / `:259 rule` → peer prefill with streamed layers (`offload/stream.rs`) | E + S | A row maps to `executors`. 1.1 drops worker `install-offload`, so `rule(pid)` reads the worker config: mode `cost`/`always`, `prefill_min_suffix_tokens` (`offload.rs:20`). A row changed after dispatch needs a new `worker-api` RPC `set_executors(req, list)` (else refuse for started processes). The helper's `prefill` process reports on the helper with attr `pie/helper-for`. |
| `replication/workers` | `cluster/host.rs:1129 replace_replication`, `:1216 orders` → `GatewayDecisions.replicate` → gateway `service.rs:502 carry` (30 s retry) → `worker-api replicate` (`:52`) → serving workers' host tier, or a pool (`worker/src/link/pool.rs:31`, `put_run`) | E + S | A row on `prefix{ws,tokens}` becomes the group's keys truncated to `tokens` (1.0's `upto`). "The table asks; retention disposes": send `retention = 0` and `disk = false`, and let the target worker's retention rank it. Drop pie's cluster `state.retain` (`:1139`, `:1290 retained`): 1.1 has no cluster retention. |
| `terminate` | `GatewayDecisions.terminated` → `service.rs:463` refuse at gate or cancel | E | |
| `alarm`, `notes` | host-1.1 | E | Notes on a process go out as `PlacementView.props` `cluster/note/<writer>/<name>`. |

### 2.5 Load-time support sets (`Config.supported`)

- **Worker:** `schedule/priority`, `schedule/hold`, `retention/priority`, `retention/tiers` always (`[]` and `[device]` work without a host tier), `terminate`.
- **Pool worker** (if it runs a host, §3.3): `retention/*` only.
- **Cluster:** `placement/workers`; `placement/helpers` only when some worker has a peer link (`offload::executors::reachable`); `replication/workers`.

### 2.6 Hook count

- **Report hooks:** 19.
  - 16 existing call sites, all already funnelled through `policy::report` (`git grep 'policy::report('`: 21 lines across `process.rs`, `planner/report.rs`, `store/kv.rs`, `handle/table.rs`, `session.rs`, `offload/prefill.rs`, `fire.rs`, `pressure.rs`, `policy.rs`);
  - 3 new: client pause in `idle_while`, fire counters, per-tier pages.
- **Enact points:** 9 on the worker (schedule install, hold set, retention install, tiers install plus sweep, discard, terminate, notes-in, refusals, executors RPC) and 6 on the cluster (placement list, exclusion move, continuation move, helpers, replicate orders, gate `heard`).

Total ≈ 34; 7 of them are new code points.

---

## 3. The five paper items

### 3.1 `placement/helpers` (disaggregated prefill): feasible; mostly exists

pie already prefills a pass on a peer and streams its pages back layer by layer (`offload/prefill.rs` module doc and `:176`, `offload/stream.rs`). Policy executors flow from the cluster placement to the process (`process.rs:583`), and `Offloaded`/`OffloadFailed` events exist (`prefill.rs:230,249`).

What 1.1 changes:
- the per-process rule (mode, min tokens) leaves the policy and becomes worker config;
- the list shape is the same as `workers`.

Lacking:
1. a helpers change after dispatch (`set_executors` RPC);
2. helper-side visibility: report the executor's `prefill` program as a process with `pie/helper-for` (spec §12 leaves "helpers in a worker's snapshot" open; pie answers yes, as a process);
3. a check, `conform-cluster.py helpers`: pin p to W1 with `helpers=[W2]` and a long cold prompt; expect W1 attr `pie/offloaded-tokens > 0`, W2 sees a `pie/helper-for = p` process, and output equals the unhelped run.

**Limit:** "the last page of a pass is always computed at home". What can go to a helper is prefill of KV-producing passes only, never decode. That is consistent with "what a host can hand to a helper is the host's to say".

### 3.2 Migration of a started process by exclusion: feasible for compat turns; not for arbitrary inferlets

What exists:
- A move before output: `session.rs:506 migrate` ends the turn at A and relaunches it from its own input at B. It is refused once `turn.emitted`, or if the turn uses handles.
- A **continuation path for a lost worker**: compat frames carry `continuation` (the tokens they cover, `compat.rs:266 Kept`, `:470 frames`). The gateway relaunches the same request with them, and the new run drops the frames the client already has.
- **KV by key from another worker's cache**: `prefill.rs:176 before_pass` fetches a pass's pages from whichever worker the directory says holds them (`fetch_held`), before any recompute.

Design for "park on A + restore on B" as pie can do it:
1. The cluster row leaves A out, so the gateway `move_misplaced` (exclusion test) calls `migrate(req, list)`.
2. If not emitted and no handles: the existing restart.
3. If emitted, continuable (compat), and no handles:
   - (a) cancel at A by `By::Policy`; the KV stays as A's cached working set (`Kept`);
   - (b) re-dispatch with the `Kept` continuation to the first listed worker that takes it;
   - (c) B's first pass fetches A's sealed pages by key through the directory (prompt plus generated tokens so far; only the partial tail page is recomputed);
   - (d) report `Moved` → `dispatched(p, B)`.
4. Otherwise, `refuse(placement/workers, p, not-supported)`. That covers a general inferlet with guest state, a turn that holds handles, and a non-continuable API.

About 250 lines: gateway `session.rs` plus the `compat.rs` glue that drives a continuation on demand instead of only on loss. The directory lags by its report period (~1 s), so very fresh decode pages may be recomputed on B.

**Cannot:** move a running wasm guest with its linear memory and host resources (pipelines, handles, inbox) between machines. A paused-by-client agent inferlet stays on its worker. Its *KV* can be replicated (§3.3) and its *next* process placed elsewhere.

### 3.3 `replication/workers`, including a pool as a worker with no device tier: feasible; exists in 1.0 shape

What exists:
- pool workers (`worker/src/link/pool.rs`: loads no model, `engine_cpu::Pool`, takes runs by `replicate`, serves them by key, reports `StateReport.pool = true`);
- serving workers' `push_to_pools` (`link/peer.rs:666`);
- cluster orders with retry (`cluster/host.rs:1216`, gateway `service.rs:502`);
- copies in the directory (`GroupView.sites/disk_sites`).

What changes:
- Rows are keyed by `prefix` of a 1.1 working set, which needs the run ↔ process/session owner link in `StateReport` (S).
- The pool appears in `get-workers` as `tiers=[host]` (plus disk if configured), `running = waiting = 0`, `step = None`. This needs a room report from the pool (S).
- Placement naming a pool is refused at dispatch: it already rejects turns (`pool.rs` `dispatch` → `Accepted::Reject`). The cluster host should refuse at write time instead (S).
- **Pool retention as a worker policy (optional N, ~250 lines):** embed a worker `Host` in `pool.rs` with `supported = retention/*`, working sets = its runs, and ranking fed into `put_run(…, retention)` and the pool's eviction. Without it, a pool keeps runs in its own (key, LRU) order. That is the column default, which is enough for the paper's "pool is a worker" claim. With it, a policy can drop a pool copy now (`tiers=[]`), as spec §4.4 describes.

### 3.4 Worker columns and `terminate` under `conform.py`

| conform test | pie status |
|---|---|
| `priority` (4 seats, high admitted before 6 low) | E: `ordered.rs` grants by key; needs a 4-seat config (`init_admission(max_concurrent)`, `process.rs:347`) |
| `hold` waiting, then running → paused → resumes | S: per-pid hold; the running case needs `can_pause()` (host tier). The pause path is `pause_held`. |
| `terminate` (`cut`: < 200 tokens, `finish_reason` abort/stop, "cut" visible) | S+N: needs `generated-tokens` counters, plus cooperative terminate (§4) |
| `keep` (retention priority under pressure) | E: `reclaim_order` by key |
| `upto` (head survives, tail goes) | E: `split_at_steps` and step keys |
| `drop` (`tiers=[]` drops now) | S: discard at keep time and on row |
| `demote` (`--host`: `[host]` demotes now; next turn served from host; `[device]` keeps it off host) | **N**: demote-now sweep and tier filter; plus host-hit reporting |
| `host-order` (`--host`: higher comes back from host, lower gone) | E/S: host order follows the key via `prepare_spill` / host drop. **Verify** the host pool's own eviction is key-ordered when there is no disk tier. |

**Which of the two 1.0 gaps does 1.1 cover?**
- **Two-tier demotion: yes, it is a 1.1 column.** `retention/tiers` makes device → host → disk placement explicit and testable (`demote`, `host-order`). pie now has host and disk tiers (`kv.rs:981,1130`); under 1.0 it had none ("absent (device only)", `ports-1.0/README.md`). It still lacks tier *membership*, which is the N above.
- **Admission preemption: no, 1.1 has no mechanism for it.** The spec has "no displacement": a policy that wants a waiting process to displace a running one holds the victim (§4.1, `schedule/hold` doc). pie's `schedule.preempt-for-admit` is still not granted (`policy.rs:42` lists it; `service.rs:101 capability` returns false). Under 1.1 it is not needed: hold → `pause_held` → paused(policy) → the seat is vacated (`vacate_held_seats`) → the waiter is admitted. conform does not test displacement. The paper should say "covered as a policy idiom (`hold`), not an engine mechanism" (spec §12 "a free seat").

### 3.5 `paused` with `by: client`: feasible, small; pie is the only engine with it

pie knows when a process waits on its client: `idle_while` (`inferlet/host/session.rs:15`) wraps `receive`/`receive_blocking` (`:205,:238`) and marks the planner `idle` (`residency_planner.rs:871 set_idle`), which makes it the first victim and parks its restore.

Design:
- Report `paused(p, client)` on entry and `started(p)` on return.
- For tool calls inside an inferlet, wrap the outgoing wasi-http request future (inferlet linker) and `sleep` in the same `idle_while`, behind an opt-in such as the inferlet's `declare-tool-wait`, so model-side waits are not misread.
- Retention on a client-paused process: when a `tiers` row lacking `device` names a paused process's working set, the port asks the planner to `pause([pid])` now (`residency_planner.rs:997`), so it demotes during the wait instead of under pressure. When `receive` returns, `set_idle(false)` queues the restore, as today.

Compat (OpenAI API) tool calls end the turn, as on vLLM. There the signal is `retained`, not `paused`. `by: client` therefore shows on WS or `/v1/ws` multi-turn inferlets and agent inferlets, which is the paper's example B on pie.

---

## 4. `conform.py` on pie: what must be served

Endpoints and behaviour:

| Need | pie today | Mark |
|---|---|---|
| `GET /v1/models` with `owned_by` | `compat.rs:558` returns `"owned_by": "pie"` | E |
| `POST /v1/completions` with **token-id prompt** (`prompt: [int…]`), `max_tokens`, `temperature 0`, `ignore_eos`, `stream` SSE | `completions.rs:86-100` **refuses token arrays** (400) | S: `compat::Request::new(tokens)` already takes tokens; accept `[int…]` |
| `usage.prompt_tokens_details.cached_tokens`, `usage.completion_tokens` | `compat-openai/src/lib.rs:39`, `compat/src/run.rs:216,244` | E |
| **from-host count** for `--host` (`h[2]`, `p[2]`) | none; conform reads vLLM metrics or SGLang `sglext` | S: count host-promoted pages in `supply` (`fire.rs:3781` area), surface `prompt_tokens_details.cached_tokens_details.host`, and add a `pie` branch to `conform.py` |
| `x-plex-hint-<name>` headers | only `x-plex-prop-*` (`compat.rs:97`) | S: accept both, as `plex_port/hints.py` does |
| `x-plex-session` | `compat.rs:99` | E |
| `POST /v1/plex/sessions/<s>/hints` | only `/props` and `/close` (`ingress.rs:18-26`, `ingress/plex.rs`) | S: alias `/hints` |
| terminate visible: `finish_reason` `abort`/`stop` and the reason in the choice | wasm aborted → `ProcessEvent::Error` → error body (`compat.rs:429`) | **N** (~150 lines): a cooperative stop. `terminate_by(…, Policy)` sets a per-process stop flag with the reason. A new inferlet `session.stop-requested() -> option<string>` (WIT) is checked per token in `compat/src/run.rs` (the loop at `:192`). A new `Finish::Abort(reason)` (`run.rs:21`) gives `finish_reason: "abort"`, `stop_reason: reason`. Hard abort after a grace (one fire or 200 ms) for non-cooperating inferlets. |
| `generated_tokens` for `cut` | not counted by the runtime | S (§1.2) |
| 4 seats, ~16K-token device pool, host tier on/off | worker config: admission seats, `kv_pages`, swap | E (config) |
| `/metrics` | only read for vLLM | n/a |

Run:

```
pie serve --policy conform.wasm --seats 4 --kv-tokens 16384 [--host-tier …]
python conform.py PORT --device-tokens 16384 [--host]
```

Conform covers worker columns only. Add `conform-cluster.py` for the three pie-only checks: helpers, exclusion migration, and replication to a pool. Use the same probe style: a cluster probe policy driven by hints.

---

## 5. What cannot be done, plainly

1. **Migrating an arbitrary running inferlet.** Guest linear memory plus host resources (pipelines, inbox, handles) do not move between machines. Only continuable compat turns migrate after output, by continuation plus KV fetched by key. Others are refused (`not-supported`).
2. **Restoring on B the exact device state parked on A, as one transfer.** pie moves KV by sealed-page key through the directory. The partial tail page and pages the directory has not yet heard of are recomputed on B.
3. **Helpers for decode,** or for the last page of a pass: always computed at home.
4. **Promotion ahead of a hit for plain cache.** `tiers=[device]` on host-resident cached KV takes effect at admission (spec §12 allows it). Handle holders can be restored early; plain cache cannot, without new work.
5. **Shared working sets (`parent`) at the worker.** Not reported, as on the other ports. pie's trie could, later.
6. **An exact cluster tree for pending processes.** Prompts reach the controller as text-block hashes, not token page keys. `cached-tokens` before dispatch and a pending process's `parent` are estimates.
7. **`started` at the cluster.** It means the first output, unless workers forward starts.
8. **Hold of a running process on a worker without a host or disk tier.** Refused (pie cannot recompute KV, so pausing needs a lower tier).
9. **Admission preemption as an engine mechanism.** Not in 1.1. Use the `hold` idiom.
10. **1.0 verbs with no 1.1 column:** batch token bound, scale, transfer shares, acceptance, cluster retention. These become pie config or attrs; policies lose them.
11. **Exact prompt/generated split for non-compat inferlets.** The runtime counts rows written and tokens sampled. A generic inferlet's "prompt" is its own business.

---

## 6. Work plan

Each phase ends at a gate. "Conflict" means it touches the residency planner that `memory-redesign` follow-ups are changing:
- guest bind as a planner waiter (today `process.rs:447 ensure_bind_admitted` / `ordered.rs`);
- park-victim order from policy, youngest first by default (`planner/eviction.rs:295-317`, `rank.rs`);
- restore waiting (`WaitKind::Restore`, `report.rs:33`, `set_idle :871`, `reseated`).

**Rule that keeps the conflict small:** the port touches the planner only through three reads it already uses, `schedule::key(pid)`, `schedule::held(pid)` and `retention::Keys` (plus the new `tiers_at`), and through `planner.pause(pids)` and `reschedule()`. Whatever the redesign moves, it keeps calling those.

| Phase | Work | Lines (new/changed) | Gate | Planner conflict |
|---|---|---|---|---|
| **P0 Host crate** | Upstream to `host-1.1`: `poll_delta`, `heard` sequence. Vendor as `crates/plex-host`, wasmtime 49. | ~200 | `cargo test -p plex-host`; plex `policies-1.1/check` scenarios (268) pass against the vendored host | none |
| **P1 Worker reports** | New `runtime/src/plex/` port thread replaces `policy/service.rs` `Registry`/`Program`. Map the 16 events. Add `by` on pause, the fire counters, per-tier pages. `put_worker`/`put_working_set` periodic. | ~700 (−2,600) | A probe that only reads (`notes` its snapshot) shows correct states, counters, working sets and rooms on a live CUDA worker | low: one line in `planner/report.rs:35` |
| **P2 Worker enact: schedule, retention priority, terminate** | `schedule::Table` from rows (hold set), `retention::Table` from prefix rows, `terminate`. Compat: token-array prompts, `x-plex-hint-*`, `/hints` alias, cooperative stop (`Finish::Abort`). | ~550 | `conform.py` (no `--host`): priority, hold, terminate, keep, upto, drop | **medium:** hold and resume ride `pause_held`, `vacate_held_seats` and restore order. Land after, or rebase onto, the restore-waiting change; it needs only `held(pid)` semantics kept. |
| **P3 Retention tiers** | `tiers_at` in `Keys`; tier filter in `prepare_demote`/`prepare_spill`/keep; demote-now sweep; refusals; host-hit count in usage; `conform.py` pie branch. | ~450 | `conform.py --host`: demote, host-order, plus all of P2 | **high:** `store/kv.rs` demote/spill and `planner/pool.rs:409,421` sweeps are the redesign's ground. Do it last on the worker, after the victim-order change merges. |
| **P4 Paused by client** | `idle_while` reports; optional tool-wait wrap; a tiers row on an idle pid → `planner.pause`. | ~100 | A WS agent inferlet waiting on `receive` is `paused(client)`; a `[host]` row demotes it during the wait; it resumes with KV restored | **medium:** `set_idle` and restore queueing |
| **P5 Cluster host** | `controller/cluster` port around a cluster `Host`. Directory → working sets/copies (run owner in `StateReport`). `WorkerStatus` rooms. `PlacementView.workers` list; gate on `heard`; exclusion test in `move_misplaced`. | ~900 (−3,300) | `conform-cluster.py placement`: `[]` holds at the gate, `[w]` pins, the list falls through, a reorder does not move, an exclusion moves a not-yet-emitted turn | none (gateway, controller) |
| **P6 Paper items** | (a) continuation migration after output; (b) helpers (`set_executors`, config rule, helper-side process); (c) replication rows → orders with `retention=0`, pool rooms; optional pool worker host. | ~650 (+250 optional) | `conform-cluster.py`: migrate-after-output (stream continues, client sees no duplicate, B fetched pages), helpers (offloaded tokens > 0, output identical), replicate-to-pool (copy appears in `copies` with pool worker tier `host`) | none |
| **P7 Builtins** | Re-port the 25 builtin 1.0 policies to `sdk-1.1`, or retire them for plex `policies-1.1`. Delete `crates/policy`. | ~1,500 (policies) | builtin tests green | none |

**Size.** About 3,500–4,000 lines of new or changed runtime, gateway and controller code, plus the policy re-ports. About 6,000 lines of 1.0 host code deleted.

**Hooks.** ~34: 19 report points (16 existing `policy::report` sites, 3 new) and ~15 enact points (9 worker, 6 cluster).

**Order.** P0, P1 and P5 can start now, in parallel; they do not touch the planner. P2 needs only the `held(pid)` set. P3 and P4 should wait for the planner's victim-order and restore-waiting changes to merge.
