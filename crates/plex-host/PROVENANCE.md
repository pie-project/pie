# plex-host: provenance

Vendored from the PLEX repository:

- `src/` from `host-1.1/src` at plex `128380ff884b9419070a6cce623b359e6eb39e43`;
- `wit/` from `wit-1.1/` at the same commit (the normative PLEX 1.1 contract).

Changes against that commit:

- wasmtime 47.0.3 to the workspace's 49.0.2, and pyo3 0.29 to 0.30;
- `bindgen!` reads `wit/` here instead of `../wit-1.1`;
- formatted with pie's rustfmt;
- plex `ee2f99186` (lazy snapshot and tracked exits) and `779667cfa`
  (`host_build` in the stats), applied by hand; plex `ae6029d8f` (call and
  age percentiles, between the vendored commit and those) is not.

To sync, copy both directories again and reapply these changes. A fix to
the host goes to plex first, so the vLLM and SGLang ports and pie run the
same host.
