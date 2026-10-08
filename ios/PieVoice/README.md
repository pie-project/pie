# PieVoice on a physical iPhone

Two ways to get the app onto a device. Both end with the full app — engine,
model, and inferlet run on the phone; nothing needs a network.

## Fastest: one command (phone plugged in, unlocked, Developer Mode on)

```bash
# from the repo root — builds the engine shim + inferlet, stages the model
# artifact, signs with the team in project.yml, installs, launches, then
# pulls the phone's own log and prints READY / ENGINE FAILED
bash ios/PieVoice/deploy-device.sh
bash ios/PieVoice/deploy-device.sh --no-rust   # Swift-only change, skip cargo
bash ios/PieVoice/deploy-device.sh --log       # re-pull Documents/pie-console.log
```

The model comes from a Pie store on the Mac: `pie model import
Qwen/Qwen3.5-0.8B` first, then `PIE_STORE` (default `$PIE_HOME`, else
`~/random/pie-05-home`) or `MODEL_DIR` tells the script where it is.

A free Apple ID's signing profile lasts 7 days; the script mints a fresh one
every run, so just run it again if the app stops opening after a week.

Three device-only failures the Simulator never showed, all handled by the
app: `$PIE_HOME` defaulting to the read-only container root (now
Library/Application Support/pie), engine boot errors swallowed into a
forever-spinner (now shown with a Retry button that quits for a clean
boot), and a wasm pool sized for a datacenter (the iPhone 16 Pro grants at
most 5.2 GiB in a single reservation; the app's config asks for 4 x 128
MiB). The app mirrors stdout/stderr to `Documents/pie-console.log` on
device and logs the measured virtual-address ceiling at launch.

## A. Build and run with Xcode

Requires: Xcode 16+, a free Apple ID, an iPhone on iOS 17+.

```bash
# from the repo root
rustup target add aarch64-apple-ios wasm32-wasip2
(cd ios/pie-shim && CARGO_TARGET_DIR=$PWD/../../target \
   cargo build --release --target aarch64-apple-ios)
(cd examples && cargo build --release --target wasm32-wasip2 -p voice-chat)
cp examples/target/wasm32-wasip2/release/voice_chat.wasm ios/PieVoice/Resources/
pie model import Qwen/Qwen3.5-0.8B
mkdir -p ios/PieVoice/Resources/models
cp -R "${PIE_HOME:-$HOME/.pie}/models/Qwen--Qwen3.5-0.8B" ios/PieVoice/Resources/models/
bash ios/voice-app/make-samples.sh ios/PieVoice/Resources
brew install xcodegen && (cd ios/PieVoice && xcodegen)
open ios/PieVoice/PieVoice.xcodeproj
```

In Xcode: Signing & Capabilities → select your team (a free Apple ID works;
Xcode creates the certificate), plug in the iPhone, press Run. With a free
Apple ID the install expires after 7 days — re-run to refresh.

## B. Sideload a prebuilt .ipa

Unsigned `PieVoice.ipa` builds are attached to the releases of
[aarushkandukoori/pie-ios](https://github.com/aarushkandukoori/pie-ios/releases);
the ones there today are Pie 0.4 builds. Install with
[AltStore](https://altstore.io) or [Sideloadly](https://sideloadly.io),
which re-sign it with your own Apple ID. Same 7-day refresh rule.

## Notes for device runs

- **Measured 2026-10-08 on an iPhone 16 Pro (iOS 26.1)** with the 0.5
  build: `results/dev-20261008-iphone16pro-qwen3.5-0.8b-run{1,2}.jsonl`
  (5-turn scripted conversation, `-PieBenchmark 1`). Engine boot plus a
  warm-up turn 0.3 s and 0.6 s; time to first token 0.11-0.25 s; decode
  59-72 tok/s after the first turn; reused prompt tokens 64 → 256 (run 1)
  and 64 → 192 (run 2) by turn 5; peak footprint 977 / 981 MiB. `bench-report.py <file>` prints the
  table. `results/dev-20260921-iphone16pro-qwen3-0.6b.jsonl` is the Pie
  0.4 ggml-CPU run with Qwen3-0.6B, kept for comparison. `bench-ladder.sh`
  lists the 2B and 4B rungs but only the 0.8B rung has been deployed.
- Right after an install, the first launch can be refused with a
  "Security" error while iOS finishes registering the app; the deploy
  script waits a few seconds and launches again before giving up.
- If the status line ever says "engine boot failed: …", the button under it
  quits the app; reopen it for a clean boot. A failed boot is deliberately
  final for the process (a half-started engine can't be safely restarted
  in place), and the reason is in `Documents/pie-console.log`
  (`deploy-device.sh --log`).
- Speech falls back from on-device to Apple's server recogniser for the rest
  of the run if the local one fails right after a mic press (assets not
  downloaded yet); the header badge says which is in force. Typing always
  works regardless of speech permissions.
- A free Apple ID cannot use the extended-virtual-addressing entitlement, so
  the largest single reservation the kernel grants is 5.2 GiB (measured on
  an iPhone 16 Pro, iOS 26.1). Whether the 2B and 4B rungs map under it
  with U4 weights is exactly what the ladder is for.
- First launch pays the full model load (~420 MB from flash) — expect a
  noticeably longer warm-up than a relaunch.
- TestFlight distribution needs an Apple Developer Program membership.
