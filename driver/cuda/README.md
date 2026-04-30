# driver/cuda

Native CUDA driver for Pie. Sibling to `runtime/` (Rust) and `pie/` (the Python
driver). Replaces the Python forward-pass executor with a C++ + flashinfer
binary that talks to the runtime over the shmem fast path defined in
`runtime/src/shmem_ipc.rs`.

The binary built here (`pie_driver_cuda`) is intended to be spawned and
managed by the Python wrapper module `pie_driver_cuda_native` (forthcoming).

> **Status:** Scaffold only. Currently parses config, opens the shmem region,
> decodes `BatchedForwardPassRequest`s, and replies with an empty payload.
> Model loading and CUDA forward pass via flashinfer are TODO.

## Architecture

```
┌────────────┐  shmem fast path   ┌──────────────────────┐
│ runtime/   │ ◄────────────────► │ driver/cuda (this)   │
│ (Rust,     │   PIE3 control +   │  - decode batch      │
│  client)   │   BPIQ schema      │  - run forward       │
└────────────┘                    │  - sample            │
                                  └──────────────────────┘
```

The Rust runtime owns scheduling, KV-cache page tables, and inferlet
execution. `driver/cuda` owns model weights and per-batch forward-pass
execution on CUDA. They share state only through the shmem channel (no
sockets, no RPC).

## Dependencies

System packages (Debian/Ubuntu):

```bash
sudo apt-get install -y cmake ninja-build libzstd-dev
```

Plus a CUDA toolkit (12.x) and an NVIDIA GPU with SM 8.0+.

CMake-managed via CPM:

- `flashinfer-ai/flashinfer@0.2.7` — paged attention kernels
- `marzer/tomlplusplus@3.4.0` — config parsing
- `CLIUtils/CLI11@2.5.0` — CLI args
- `nlohmann/json@3.12.0` — model metadata
- `facebook/zstd@1.5.7` — zTensor decompression

## Build

```bash
cd driver/cuda
cmake -S . -B build -G Ninja
cmake --build build
```

The binary lands at `build/bin/pie_driver_cuda`.

## Run (scaffold)

```bash
./build/bin/pie_driver_cuda --config dev.toml
```

Then start the Rust runtime pointed at the same shmem name. The runtime will
be the client; `pie_driver_cuda` will log incoming batches and acknowledge
them with empty responses.

## Roadmap

- [x] CMake + CPM dependency wiring
- [x] Shmem IPC server (mirrors `runtime/src/shmem_ipc.rs`)
- [x] BPIQ flat schema decoder
- [ ] HuggingFace weight loading (zTensor / safetensors)
- [ ] L4MA model graph (attention via flashinfer, MLP, RMSNorm, RoPE)
- [ ] KV-cache page allocator
- [ ] Sampler (top-k / top-p / min-p / temperature)
- [ ] Adapter (LoRA) support
- [ ] Speculative decoding
- [ ] Response payload writer matching the runtime's expected layout
- [ ] `pie_driver_cuda_native` Python wrapper (spawn + shmem handshake)
