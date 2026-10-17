# Built-in PLEX policies

The policies built into pie (`plex:policy@1.0.0`; the contract is
`crates/policy/wit`). Name one in the config:

- the engine plane in `[runtime] policy`;
- the router plane, in `pie serve`, in `[server] policy`.

A policy reads the props a client sends with a launch (`x-plex-prop-<name>`
headers, a `plex` object in a compat body, or `LaunchProcess.props`) and the
session it names (`x-plex-session`).

| policy | plane | does | reads |
|---|---|---|---|
| `noop` | engine | nothing: the engine's own order | |
| `priority` | engine | larger priority runs first and is preempted last | `router/priority`, else `priority` |
| `fair` | engine | the tenant that has had the least service runs first (after VTC); service is time spent running | the tenant |
| `deadline` | engine | earliest deadline first; one without a deadline runs after every one with | `deadline-ms`, from arrival |
| `keep-session` | engine | keeps a completed process's cached KV above the rest for `keep-ms`, for its session's next turn, then discards it | `keep-ms`, the session |
| `session-affinity` | router | sends a session's processes where its last one went | the session |
| `router-priority` | router | launches leave the gateway's queue by priority, and pass it on as `router/priority` | `priority` |

Pair `router-priority` with the engine's `priority` and the two planes order
alike. `test-*` policies misbehave on purpose and are built only for the
hosts' own tests.
