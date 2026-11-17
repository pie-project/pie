# WIP, not for merge: the Metal residency gap's repro

A serving Metal process's GPU driver memory that `currentAllocatedSize` does not
count (new-memory/FOLLOWUPS.md F14). This branch carries the experiment code:
- `PIE_PROBE_LIFT=1`: capacity is the recommended working set less what the
  process held, less the 64 MiB floor (no wired check), so serving reaches
  the driver's refusal;
- the refused frame logged as `metal: refused-frame: …`;
- the residency probe (`cargo test -p engine-metal --release --lib --
  --ignored --nocapture where_the_device_refuses_residency`; `PIE_PROBE_SPARSE`,
  `PIE_PROBE_NOCOPY_MB`, `PIE_PROBE_HOLD_GB`, `PIE_PROBE_WARMUP_CMDS`).

The gap is visible right after a load (no traffic needed):
`ioreg -r -c IOAccelerator -l -w0 | grep -o '"Alloc system memory"=[0-9]*'`,
before the server starts and once it says "Server ready", against the load
line's `device N MiB`. On an M1 Max: 0.29 GB idle, 3.17 GB after loading
Qwen3-0.6B, whose own allocation is 1.89 GB.

The refusal (optional): `soak_topp.py BINARY HOME EXAMPLES WASMDIR TAG MINUTES
[WINDOW]` with `SOAK_CAP=8 PIE_PROBE_LIFT=1 PIE_SHARED_SEGMENTS_FILE=<file>`,
writing `600` to the file ~45 s in; the refusal comes ~1 min in.
