# Pie on iOS (work in progress)

Runs the Pie 0.5 runtime — the Metal engine plus wasmtime executing
inferlets — inside an iOS app.

- `pie-shim/`: C-ABI staticlib embedding the standalone engine
  (`pie_ios_run_stream` / `pie_ios_free`). The first call boots the
  engine from a Pie 0.5 config TOML and keeps it warm for the life of the
  process; each call installs the inferlet component once, launches it
  with a JSON input, streams its stdout to a callback and returns its
  return value. Build with `cargo build --release --target
  aarch64-apple-ios` (or `aarch64-apple-ios-sim`).
- `voice-app/`: **a voice assistant you hold a conversation with** —
  speech in, model reply spoken back, all on the device. Layered so the
  Pie-facing code, the audio code, and the UI can each be replaced
  independently; see `voice-app/README.md`. Uses the `voice-chat`
  inferlet (`examples/voice-chat`): the app sends the whole transcript
  every turn and the engine serves what it can from the KV state of
  earlier turns.
- `PieVoice/`: XcodeGen project, one-command device deploy, benchmark
  driver. See `PieVoice/README.md`.

Status (2026-10-08): **runs on a real iPhone 16 Pro with Pie 0.5** —
the Metal engine, `.zt` model artifacts, Qwen3.5-0.8B (SKU
`qwen35-d0.8b-u4g64-kv-bf16`, a hybrid model: attention every fourth
layer, Gated DeltaNet state otherwise), inferlets under wasmtime Pulley.
Measured on the phone (iOS 26.1), 5-turn scripted conversation, two runs
(`PieVoice/results/dev-20261008-iphone16pro-qwen3.5-0.8b-run{1,2}.jsonl`):
engine boot + warm-up turn 0.3 s / 0.6 s; time to first token 0.11-0.25 s;
decode 59-72 tok/s after the first turn (36 and 62 tok/s on turn 1, the
first run paying for kernel warm-up); prompt tokens reused from earlier
turns 64 → 256 by turn 5, so each turn prefills only the previous reply
plus the new question (54-101 tokens) however long the transcript grows;
peak footprint 977 / 981 MiB. Deploy with
`bash ios/PieVoice/deploy-device.sh`.

History — measured 2026-09-21 with Pie 0.4 / the ggml CPU driver: ran on
a real iPhone 16 Pro (iOS 26.1) with Qwen3-0.6B Q4_K_M, speech in and
out on the device, inferlets under wasmtime Pulley. 3-turn scripted run,
4 ggml threads, warm relaunch: engine boot + weight load 0.97 s; turn 1
prefill 83 tokens, 0.57 s to first token, 62.6 tok/s decode; turns 2-3
reused 107/149 KV tokens with 0.17-0.19 s to first token at ~60 tok/s;
peak footprint 1.0 GiB (1,029 MiB)
(`PieVoice/results/dev-20260921-iphone16pro-qwen3-0.6b.jsonl`). Those
numbers are for the old driver and model and do not carry over.

Device facts that still hold in 0.5:

- The largest single anonymous mmap the kernel grants is 5.2 GiB without
  the extended-virtual-addressing entitlement (measured on the iPhone 16
  Pro). The app logs the measured ceiling at launch; the engine config it
  writes is sized under it (`[sandbox]` pool of 4 x 128 MiB, 256 KV
  pages, 8 recurrent-state slots).
- `$PIE_HOME` defaults to `~/.pie`, which on a device is the read-only
  container root. The app sets it to Library/Application Support/pie
  before boot (`voice-app/Sources/PieKit/PieRuntimeConfig.swift`).
- A failed engine boot is final for the process; the app's Retry button
  quits so the next tap starts clean.
- There is no attached console on a device, so stdout/stderr are mirrored
  to `Documents/pie-console.log` (`deploy-device.sh --log` pulls it).

Next: the 2B and 4B rungs of the ladder (`PieVoice/bench-ladder.sh`),
TestFlight (needs an Apple Developer Program membership), Android.
