# pie-transport

The worker↔worker **P2P KV-tensor data plane** for pie.

This crate moves KV-cache tensors directly between workers. A controller pairs
worker A with worker B and then **steps out** — from that point KV tensors flow
P2P A↔B, *bypassing the controller*. `pie-transport` owns that movement and
nothing else; it never makes policy.

## Structure

```
transport/src/
  core/       backend-agnostic interface: register → send/recv → poll
  engines/
    local/    same-node device-to-device copy (co-located PD, zero network)
    nixl/     cross-node RDMA/TCP/NVMe via NIXL   [feature = "nixl", deferred stub]
  registry/   binds a driver-exported handle to an engine, dispatches
  error.rs, lib.rs
```

**Minimal start = `core` + `local`.** Co-located prefill+decode defers all RDMA
(YAGNI). The `nixl` engine is a stub behind `feature = "nixl"` and is the only
place RDMA lives — enable it for the datacenter build; the on-device build ships
NIXL-free with just the `local` engine.

Backends are asymmetric: cuda/rocm cross-node use NIXL, co-located peers use
`local`, and **metal/vulkan never participate** (single-node; NIXL is
Linux-only).

## Boundaries

| edge | contract |
| --- | --- |
| **↔driver** | driver pins KV buffers + exports a `pie_driver_abi::KvHandle`; transport consumes it without owning/interpreting bytes. The per-backend registration shim lives on the driver's export surface. Transport never imports the driver — they meet only through the handle type. |
| **↔controller** | receives a pairing decision and *executes* it. No routing/scheduling here. |
| **↔runtime** | transfers are async — transport exposes the start + a completion signal; *when* to await is the scheduler's job. |
| **↔interface/driver** | the KV layout + handle type live in `pie_driver_abi`, shared by driver / transport / runtime / controller. |

## Engine seam

Each engine implements `core::Engine` (`register` → `send`/`recv` → `poll`).
The `local` engine issues its device-to-device copy through the `D2dCopier`
seam (a real impl wraps `cudaMemcpyDeviceToDevice` / `hipMemcpy`; tests use a
recording fake), so the crate builds and tests without a device.

The `nixl` engine is the cross-node path. It wraps NVIDIA's NIXL via its C-API
(`libnixl_capi.so`) over UCX — RDMA where a NIC exists, otherwise `shm`/`tcp`.
The remote-access credential is NIXL's opaque agent **metadata blob**
(`get_local_md`/`load_remote_md`), carried at the connect level in
[`PeerConn`]; the floor's `KvRegion` stays backend-neutral (no ibverbs `rkey`).

## NIXL feature

The whole `nixl` engine lives behind `--features nixl`. The default build needs
no NIXL — it compiles only `core` + `local`, so CI stays green with no native
dependency. Building `--features nixl` requires a `NIXL_PREFIX` assembled from
the pip wheel (the only NIXL *source* file vendored here is
`src/engines/nixl/wrapper.h`, the Apache-2.0 C-API header):

```bash
# 1. Assemble NIXL_PREFIX from the pinned wheel (no source build):
pip download nixl-cu12==1.3.0 --no-deps -d /tmp/nx && (cd /tmp/nx && unzip -q *.whl)
mkdir -p "$NIXL_PREFIX"/lib/{plugins,ucx}
cp -a /tmp/nx/.nixl_cu12.mesonpy.libs/*.so*          "$NIXL_PREFIX"/lib/
cp -a /tmp/nx/.nixl_cu12.mesonpy.libs/plugins/*.so*  "$NIXL_PREFIX"/lib/plugins/
cp -a /tmp/nx/nixl_cu12.libs/*.so*                   "$NIXL_PREFIX"/lib/        # bundled UCX (libucp/libucs) + deps
cp -a /tmp/nx/nixl_cu12.libs/ucx/*.so*               "$NIXL_PREFIX"/lib/ucx/    # UCX uct transport modules

# 2. Build + run the single-node UCX shm,tcp e2e (no RDMA NIC / no GPU needed):
export NIXL_PREFIX LD_LIBRARY_PATH="$NIXL_PREFIX/lib:$NIXL_PREFIX/lib/ucx"
cargo test -p pie-transport --features nixl -- --test-threads=1
```

`build.rs` bindgens `wrapper.h` and links `$NIXL_PREFIX/lib/libnixl_capi.so`
(with rpath). At run time the engine sets `NIXL_PLUGIN_DIR`/`UCX_MODULE_DIR` and
defaults `UCX_TLS=shm,tcp`. NIXL is single-threaded (it can deadlock), so the
suite runs with `--test-threads=1`. The e2e test **skips** (does not fail) when
`NIXL_PREFIX` is unset or the agent can't init. First cut is **host-DRAM**;
VRAM/GPUDirect is deferred.

## Status

`core` + `local` are the working baseline; `nixl` is a real engine behind
`--features nixl` (proven with a verified UCX `shm,tcp` DRAM transfer). The
registry mints globally-unique [`TransferId`]s across engines. `WorkerId` is
`pie_ids::WorkerId` (the shared floor vocab).
