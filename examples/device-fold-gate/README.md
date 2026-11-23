# Device-fold gate witnesses

Served witnesses of the fold / stop-truncate contract: `device-fold-gate` (greedy run-ahead, the decode
fires folding by a device-computed or a host-stated count), `device-fold-gate-park` (a set parked
mid-run and resumed), `device-fold-gate-fork` (a set forked mid-run), and `device-fold-give` (a set
given, then stat / fork / save / reopen by the client). What each compares (e.g. the device-fold run's
tokens against the host-fold run's) is in its `src/lib.rs` header.

Run: build with `cargo build --release --target wasm32-wasip3 -p <witness>` in `examples/`, serve
with one of `configs/*.toml` (set `engine.type` / `device` for the backend under test), and launch the
guest. `device-fold-give` has its driver:
`python3 device-fold-give/drive.py <pie> <config> <guest.wasm> <log> packages/python/client/src`.
