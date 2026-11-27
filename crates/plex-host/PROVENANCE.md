# plex-host: provenance

Vendored from the PLEX repository (`plex`), with the user's approval.

Base: plex `d32445219` (the commit the 1.1 contract and
paper are stated against):
- `wit/` is `wit-1.1/` at `d32445219`, byte for byte, except the
  clarifications listed below (each approved by the spec's author); pie
  makes no other edits. It is normative.
- `src/` is `host-1.1/src` at `d32445219`, plus these plex host commits:
  - `84c1b989a` typed `poll_delta` and an event sequence with `heard()`
    (the two additions the pie design asked of the host);
  - `128380ff8` clippy-clean (column macro bindings, a `Counter` alias);
  - `ee2f99186` lazy snapshot and tracked exits, a round costs O(changes);
  - `779667cfa` `host_build` in the stats.

  Not taken: `ae6029d8f` (evaluation tooling: call and age percentiles).

Pie-local changes:
- wasmtime 47.0.3 to the workspace's 49.0.2;
- `bindgen!` reads `wit/` (and `wit-ext/`) here, not `../wit-1.1`;
- the pyo3 module (`py.rs`, the `python` feature, the cdylib) is not
  vendored: pie links the rlib;
- formatted with pie's rustfmt;
- the pie extension `wit-ext/` (`pie:plex-ext`), and its column in
  `tables.rs` and `runtime.rs` (see `EXTENSION.md`).

Nothing else from plex is vendored (no paper, docs, policies, ports, traces
or results). To sync, copy `src/` and `wit/` again and reapply the pie-local
changes. A fix to the host goes to plex first, so the vLLM and SGLang ports
and pie run the same host.

## Clarifications (approved by the spec's author)

- §2 dispatch gate: The gate applies while a loaded cluster policy imports placement/workers or terminate; with none, the gate bound is zero.
