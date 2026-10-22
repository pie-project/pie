# Built-in policies

The policies built into pie (the contract is `crates/policy/wit`). Name one
in the config:

- the worker plane (when a worker's passes run, where its state sits) in
  `[runtime] worker_policy`;
- the cluster plane (where programs and their state live), run by the
  controller: in `pie serve`, `[cluster] cluster_policies`; in
  `pie-controller`'s file, `cluster_policies`. A list: several run side by side so long as no
  two are granted the same write (`cache-aware` places launches while
  `replicate` keeps runs on two workers); `publish` is excepted, for each
  program publishes under its own name.

SIGHUP reloads them from the config file, without a restart:
`pie-controller` its `cluster_policies`, `pie-worker` its `[runtime]
worker_policy`, `pie serve` both; nothing else in the file is read again. The new set loads whole or
the running one goes on, and the error is logged. The old programs stop as
a dead one does: their tables and what they published are withdrawn, and
the others hear it. The new ones start with no state of their own, a policy
in both sets too, and read the cluster as it stands; they act on the next
edge they subscribe to. A controller that started with no cluster policy
starts them on a reload.

A policy reads the props a client sends with a launch (as request headers,
in a compat body, or as `LaunchProcess.props`) and the session it names.

| policy | plane | does | reads |
|---|---|---|---|
| `noop` | worker | nothing: the worker's own order | |
| `priority` | worker | larger priority runs first and is preempted last | `cluster/priority`, else `priority` |
| `fair` | worker | the tenant that has had the least service runs first (after VTC); service is time spent running | the tenant |
| `deadline` | worker | earliest deadline first; one without a deadline runs after every one with | `deadline-ms`, from arrival |
| `offload` | worker | sends a pass's prefill to a peer worker: from 1024 uncached tokens, or as the process's `offload` prop says | `offload`, `offload-min-tokens` |
| `keep-session` | worker | keeps a completed process's cached KV above the rest for `keep-ms`, for its session's next turn, then discards it | `keep-ms`, the session |
| `cache-aware` | cluster | sends a launch to the worker holding the most of its prompt (as the gateway routed it) if that is at least 0.3 of it, else to the least loaded; load wins when one worker runs 8 more turns than another and 1.5 times as many (after SGLang's router) | `worker` |
| `session-affinity` | cluster | sends a session's processes where its last one went | the session |
| `pool-replicate` | cluster | keeps every run some worker caches in every pool too, so it outlives the worker | |
| `persist-prefix` | cluster | keeps on each holding worker's persistent disk tier (`disk_kv_persistent`) the prefix a process asked for: its system turn, or its first tokens | `persist-system` (the compat routes count the turn as `system-tokens`), `persist-prefix-tokens` |
| `retain-all` | cluster | retains every run some worker caches (key 1), so the gateway keeps a copy of each in a pool and the last worker letting it go does not lose it; with no pool it does nothing | |
| `replicate` | cluster | keeps every run some worker caches on two decode workers (the two a hash of the run picks), in their host tier, so losing a worker leaves its runs on another | |
| `disaggregate` | cluster | names the prefill workers as every process's executors, so its passes are prefilled there: those on the host another program's committed placement (`placement/worker`) put the process first, then the least loaded; it writes executors alone, so `cache-aware` places processes beside it | |
| `relocate` | cluster | places a launch by `worker`, then moves it to `move-to` once it has run `move-after-ms`, where it runs again from the start; a turn that has sent output is refused and stays | `worker`, `move-to`, `move-after-ms` |
| `drain-migrate` | cluster | moves the turns on a draining worker that have sent no output to the least loaded one that is not; the rest finish where they run, which the drain waits for; a `worker` prop places a launch, as `pin` does, and a drain still moves it before it runs | `worker` |
| `rebalance` | cluster | keeps running turns even across serving workers by moving the youngest from the busiest to the idlest (after Llumnix); a turn moves only while it has sent no output; a `worker` prop names where a launch starts | `worker` |
| `pin` | cluster | sends a launch to the worker its `worker` prop names | `worker` |
| `scale-by-prop` | cluster | keeps the cluster at the worker counts launches ask for: the last ask of each shape (model and engine kind) stands, as the scale table the controller's actuator converges on (`cluster.scale`); a shape never asked for keeps what it has | `scale-workers`, `scale-engine`, `scale-model` |
| `cluster-priority` | cluster | launches leave the gateway's queue by priority, and pass it on as `cluster/priority` | `priority` |

Pair `cluster-priority` with the worker's `priority` and the two planes order
alike.

Cluster policies running side by side see each other through the host:
what their calls committed reads as attrs (a process's `placement/worker`,
`placement/executors` and `placement/props/<name>`, a group's
`replication/sites` and `retention/key`), and a policy granted `publish`
says a value for the others as `published/<its name>/<name>`. A change
reaches the others as an edge (`placement-changed`, `replication-changed`,
`retention-changed`, `published`) on their next call; see
`crates/policy/wit`.

`test-*` policies misbehave on purpose and are built only for the hosts'
own tests.
