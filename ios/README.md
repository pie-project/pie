# Pie on iOS (work in progress)

Runs the Pie runtime — wasmtime executing inferlets — inside an iOS app.

- `pie-shim/`: C-ABI staticlib embedding the engine (`pie_ios_run` /
  `pie_ios_run_stream` / `pie_ios_free`), boot path mirrors `pie run`.
  Build with `cargo build --target aarch64-apple-ios-sim` (or
  `aarch64-apple-ios`).
- `demo-app/`: minimal SwiftUI shell + hand-assembled .app bundle for the
  Simulator. See `build-app.sh` (expects a Qwen3 `tokenizer.json` under
  `target/qwen3-tok/` and the helloworld + marketing-tab2-watermark
  inferlets built for `wasm32-wasip2`).
- `voice-app/`: **a voice assistant you hold a conversation with** —
  speech in, model reply spoken back, all on the device. Layered so the
  Pie-facing code, the audio code, and the UI can each be replaced
  independently; see `voice-app/README.md`. Uses the `voice-chat`
  inferlet, which carries the conversation's KV state across turns in a
  named snapshot.

Status (2026-09-21): **RUNS ON A REAL iPHONE 16 PRO** (iOS 26.1) — speech
in, Qwen3-0.6B Q4_K_M through the portable (ggml, CPU) driver, speech out,
all on the device, with inferlets running under wasmtime Pulley. Measured
on the phone (3-turn scripted run, 4 ggml threads, warm relaunch): engine
boot + weight load 0.97 s; turn 1 prefill 83 tokens, 0.57 s to first token,
62.6 tok/s decode; turns 2-3 reuse 107/149 KV tokens with 0.17-0.19 s to
first token at ~60 tok/s; peak footprint 1.0 GiB (1,029 MiB)
(`PieVoice/results/dev-20260921-iphone16pro-qwen3-0.6b.jsonl`). Deploy with
`bash ios/PieVoice/deploy-device.sh`. Build the shim with --release: ggml at
-O0 is unusably slow.

Measured on the phone: the largest single anonymous mmap the kernel grants
is 5.2 GiB without the extended-virtual-addressing entitlement. Two engine
defaults exceeded it and aborted the app until sized for a phone — wasmtime's
1000 x 4 GiB pool reservation (~4 TB) and ggml's scheduler context for a
2^19-node graph budget (~10.8 GiB); see `runtime/src/bootstrap.rs` and
`driver/portable/src/graph_common.hpp`.

iOS-specific changes so far:
- Pulley engine target (runtime/src/bootstrap.rs)
- wasmtime allocator sized for a phone (runtime/src/bootstrap.rs,
  `init_wasmtime_ios`): an iPhone 16 Pro grants a process at most 5.2 GiB of virtual
  address space without the extended-virtual-addressing entitlement, and
  the desktop pooling defaults (1000 slots × 4 GiB) reserve ~4 TB — the
  mmap fails with ENOMEM and the engine aborted on a real iPhone. iOS now
  uses a 4-slot × 128 MiB pool and falls back to on-demand allocation
  instead of panicking.
- file-backed mmap fallback for POSIX shmem (driver/bridge/src/ipc/posix.rs)
- iOS cross-compile support for the portable driver's CMake build
  (server/build.rs: SDK sysroot defines + ios system-libs arm)
- the app sets `$PIE_HOME` to Library/Application Support/pie before boot:
  the default `~/.pie` resolves to the read-only container root on device
  (ios/voice-app/Sources/PieKit/PieRuntimeConfig.swift)

Two things worth flagging to maintainers, found while building the
voice app:

1. `Context::take` fails on the portable driver with
   `take: insufficient GPU pages (got 0, need 1)`, so a multi-turn
   session has to resume with `Context::open` and delete the old
   snapshot by hand before re-saving. `save` also refuses to overwrite an
   existing name.
2. `Generator` truncates the stop token instead of appending it, so a
   context saved right after generation ends on an unterminated
   assistant turn. Inferlets that persist a conversation need an explicit
   `Context::seal()` before `save()`, which is easy to miss —
   `demo-persistent-kv` has the same gap.

Next: model ladder on the phone (1.7B expected to fit; 4B/8B will not map
under the 5.2 GiB ceiling without the entitlement), ggml Metal backend on
iOS, TestFlight (needs an Apple Developer Program membership), Android
(llama.cpp Vulkan or ggml), durable-inferlet migration demo.
