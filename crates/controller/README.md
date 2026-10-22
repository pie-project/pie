# controller

The cluster's control plane: workers and gateways register with it and beat;
it grades their health, evicts the silent, and hands gateways the routing
table. `pie-controller -c controller.toml` runs it (`pie serve` embeds one).

## Config

```toml
listen = "tcp://0.0.0.0:7000"
heartbeat_timeout = "8s"     # a node silent this long is evicted
tick_interval = "2s"         # how often it grades, evicts and scales
cluster_policies = ["persist-prefix"]  # built-in names or .wasm paths;
                                       # SIGHUP reloads them

# Workers the actuator may launch to meet a scale table (`cluster.scale`).
# Each is one worker: a command that starts it, with its own config
# (device, ports), and the shape it serves.
[[scale.slot]]
model = "qwen3-14b"          # the model as workers serve it (`[model] name`);
                             # unset: a worker that loads no model
engine = "cuda"       # as the worker's `[engine] type` spells it
command = ["pie-worker", "-c", "/etc/pie/gpu0.toml"]

[[scale.slot]]
model = "qwen3-14b"
engine = "cuda"
command = ["pie-worker", "-c", "/etc/pie/gpu1.toml"]
```

## Cluster policies

The controller runs every cluster policy, once. Gateways tell it what
happens to their processes and apply what it decides about them (the gate,
the order launches leave in, placement, terminations, moves); workers
report what they hold to its directory. The tables that reach workers
(replication, retention) are carried by one gateway, the routing table's
`author` (the live gateway with the lowest id); when it is evicted, the
next lowest carries them on that tick. A controller dialed again hears each
gateway's live processes and each worker's whole state, so a restarted one
takes up where the last left off.

## Scale

A cluster policy granted `scale` installs a scale table (`install-scale`):
how many workers of each shape, a shape being a model as workers serve it
(their `[model] name`, or none) and an engine kind. The controller's
actuator takes the table whole, again whenever it changes.

With slots configured, the controller's actuator converges on it every tick:

- **Short of a shape:** it launches a free slot of that shape. The worker
  finds its launch token in `PIE_LAUNCH` and registers with it, so the
  controller knows which launch it is.
- **Over a shape:** it stops a worker it launched, by SIGTERM. pie-worker
  drains on the signal: it takes no new turns, the cluster policy may move
  the ones that have sent no output elsewhere, and it waits for the rest to
  finish before it leaves.
- **What it leaves alone:** a worker it did not launch is never stopped, and
  a shape the table does not name keeps what it has.
- **Short of slots:** when the table asks for more than the slots of a shape
  can give, or a slot's command will not start, it says so once in its log,
  and in its status.
- **A launch that never registers** holds its slot until its process exits.
  One that registers as another shape than its slot's is warned about, and
  not counted.
- **Adoption:** a launch token names its slot: a digest of the slot's
  entry and its index, then what makes the launch its own. A worker a
  previous controller launched, registering again with this controller on
  the same config, is adopted into its slot when the slot is free here,
  and stopped by SIGTERM to the pid it registered with. It is not this
  controller's child, so its exit is seen when no registered worker carries
  its token any more (it left, or was evicted). A token no slot of this
  config gave is a worker the actuator did not launch: it counts as
  serving, holds no slot, and is never stopped. A controller launches
  nothing for one heartbeat timeout after it starts, so the workers alive
  already have registered again and are adopted first.

Which worker of a shape is stopped is the host's choice, not a policy's:
one that does not take turns first (not registered yet, draining,
unreachable), then the most recently launched. A SIGTERM that cannot be
sent is tried again on the next tick. When the controller goes, it signals
what it launched to drain; those still draining outlive it.

`cluster.scale` answers true to a policy only while the controller has slots.
Without them, the scale table still reaches the controller, and
`scale_status` (on the `Control` service) reports each shape's desired,
serving, launching, stopping and free counts, so an operator outside pie
(for example a Kubernetes one) can converge on it instead.
