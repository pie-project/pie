# controller

The cluster's control plane: workers and gateways register with it and beat;
it grades their health, evicts the silent, and hands gateways the routing
table. `pie-controller -c controller.toml` runs it (`pie serve` embeds one).

## Config

```toml
listen_addr = "tcp://0.0.0.0:7000"
heartbeat_timeout_secs = 8   # a node silent this long is evicted
tick_interval_secs = 2       # how often it grades, evicts and scales
command_buffer = 256

# Workers the actuator may launch to meet a scale table (`cluster.scale`).
# Each is one worker: a command that starts it, with its own config
# (device, ports), and the shape it serves.
[[scale.slot]]
model = "qwen3-14b"          # the model as workers serve it (`[model] name`);
                             # unset: a worker that loads no model
engine = "cuda_native"       # as the worker's `[engine] type` spells it
command = ["pie-worker", "-c", "/etc/pie/gpu0.toml"]

[[scale.slot]]
model = "qwen3-14b"
engine = "cuda_native"
command = ["pie-worker", "-c", "/etc/pie/gpu1.toml"]
```

## Author

Every gateway runs every cluster policy. Per-request writes (schedule,
placement, executors, moves) are each gateway's own, for its own requests.
The cluster-wide ones (scale, replication, retention) have one author: the
live gateway with the lowest id, which the controller names in the routing
table (`author`). Only the author enacts them; the others' policies run
all the same and their tables are taken, as an absent capability's are,
and not enacted. The controller takes a scale table (`desire_scale`) only
from the author. When the author is evicted, the next lowest is named on
that tick and sends and enacts the tables it has at once: they are
level-triggered, and its directory is built from the workers' reports.

## Scale

A cluster policy granted `scale` installs a scale table (`install-scale`):
how many workers of each shape, a shape being a model as workers serve it
(their `[model] name`, or none) and an engine kind. The author gateway sends the
table to the controller whole, again when it changes and after the
controller restarts or it becomes the author.

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
