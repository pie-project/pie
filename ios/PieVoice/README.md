# PieVoice on a physical iPhone

Two ways to get the app onto a device. Both end with the full app — engine,
model, and inferlet run on the phone; nothing needs a network.

## Fastest: one command (phone plugged in, unlocked, Developer Mode on)

```bash
# from the repo root — builds the engine shim + inferlet, stages the model,
# signs with the team in project.yml, installs, launches, then pulls the
# phone's own log and prints READY / ENGINE FAILED
bash ios/PieVoice/deploy-device.sh
bash ios/PieVoice/deploy-device.sh --no-rust   # Swift-only change, skip cargo
bash ios/PieVoice/deploy-device.sh --log       # re-pull Documents/pie-console.log
```

A free Apple ID's signing profile lasts 7 days; the script mints a fresh one
every run, so just run it again if the app stops opening after a week.

Three device-only failures the Simulator never showed, all fixed on this
branch: `$PIE_HOME` defaulting to the read-only container root (now
Library/Application Support/pie), engine boot errors swallowed into a
forever-spinner (now shown with a Retry button), and wasmtime's pooling
allocator reserving ~3.9 TiB of address space (the iPhone 16 Pro grants at
most 5.2 GiB in a single reservation; now a 4-slot × 128 MiB pool with an
on-demand fallback,
`runtime/src/bootstrap.rs`). The app mirrors stdout/stderr to
`Documents/pie-console.log` on device and logs the measured virtual-address
ceiling at launch.

## A. Build and run with Xcode (10 minutes)

Requires: Xcode 16+, a free Apple ID, an iPhone on iOS 17+.

```bash
# from the repo root
rustup target add aarch64-apple-ios wasm32-wasip2
(cd ios/pie-shim && cargo build --release --target aarch64-apple-ios)
(cd inferlets/voice-chat && cargo build --release --target wasm32-wasip2)
# put the model at target/qwen3-gguf/Qwen3-0.6B-Q4_K_M.gguf, then:
cp inferlets/voice-chat/target/wasm32-wasip2/release/voice_chat.wasm ios/PieVoice/Resources/
cp inferlets/voice-chat/Pie.toml ios/PieVoice/Resources/voice-chat-Pie.toml
mkdir -p ios/PieVoice/Resources/qwen3-gguf
cp target/qwen3-gguf/Qwen3-0.6B-Q4_K_M.gguf ios/PieVoice/Resources/qwen3-gguf/
bash ios/voice-app/make-samples.sh ios/PieVoice/Resources
brew install xcodegen && (cd ios/PieVoice && xcodegen)
open ios/PieVoice/PieVoice.xcodeproj
```

In Xcode: Signing & Capabilities → select your team (a free Apple ID works;
Xcode creates the certificate), plug in the iPhone, press Run. With a free
Apple ID the install expires after 7 days — re-run to refresh.

## B. Sideload the prebuilt .ipa

An unsigned `PieVoice.ipa` is attached to the releases of
[aarushkandukoori/pie-ios](https://github.com/aarushkandukoori/pie-ios/releases).
Install it with [AltStore](https://altstore.io) or
[Sideloadly](https://sideloadly.io), which re-sign it with your own Apple ID
on install. Same 7-day refresh rule for free accounts.

## Notes for device runs

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
  an iPhone 16 Pro, iOS 26.1): the 0.6B rung fits with
  room; 4B and 8B will not map without the entitlement.

- The simulator build cannot use on-device speech recognition (missing
  assets); a real iPhone can — the header badge should show
  "on-device speech" once the recognizer assets download.
- Decode speed on the CPU driver will differ from the published
  Simulator-on-Mac numbers in either direction; the Metal driver is the
  performance milestone.
- First launch pays the full model load (~400 MB from flash) — expect a
  noticeably longer warm-up than a relaunch.
- TestFlight distribution needs an Apple Developer Program membership and is
  planned once the Metal driver lands; the project page tracks it.
