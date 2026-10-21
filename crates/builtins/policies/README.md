# Built-in policies

The policies built into pie (the contract is `crates/policy/wit`). Name one
in the config:

- the worker plane (when a worker's passes run, where its state sits) in
  `[runtime] policy`;
- the cluster plane (where programs and their state live), in `pie serve`,
  in `[server] policies`, a list: several run side by side so long as no
  two are granted the same write (`cache-aware` places launches while
  `replicate` keeps runs on two workers).

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
| `replicate` | cluster | keeps every run some worker caches on two decode workers (the two a hash of the run picks), in their host tier, so losing a worker leaves its runs on another | |
| `disaggregate` | cluster | names the prefill workers (least loaded first) as every process's executors, so its passes are prefilled there; where it runs is the gateway's default | |
| `relocate` | cluster | places a launch by `worker`, then moves it to `move-to` once it has run `move-after-ms`; its stream goes on | `worker`, `move-to`, `move-after-ms` |
| `drain-migrate` | cluster | moves every turn running on a draining worker to the least loaded one that is not, so a worker leaves without ending what it runs | |
| `rebalance` | cluster | keeps running turns even across serving workers by moving the youngest from the busiest to the idlest (after Llumnix); a `worker` prop names where a launch starts | `worker` |
| `pin` | cluster | sends a launch to the worker its `worker` prop names | `worker` |
| `cluster-priority` | cluster | launches leave the gateway's queue by priority, and pass it on as `cluster/priority` | `priority` |

Pair `cluster-priority` with the worker's `priority` and the two planes order
alike. `test-*` policies misbehave on purpose and are built only for the
hosts' own tests.
