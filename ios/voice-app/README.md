# Pie Voice — talking to an on-device model served by Pie

An iOS app you hold a spoken conversation with. Speech is transcribed on
the device, answered by Qwen3-0.6B running through the Pie engine
embedded in the app, and spoken back. Nothing leaves the phone.

Built on top of `ios/demo-app`, which established that the runtime and a
real model run in the Simulator at all. This adds the audio interface and
splits the app into layers that can be replaced independently.

```
   microphone ──▶ SFSpeechRecognizer ──▶ transcript
                                             │
                                             ▼
                                   ConversationController
                                             │
                       voice-chat inferlet (wasmtime Pulley)
                                             │
                            Qwen3-0.6B Q4_K_M · ggml driver
                                             │
                       speakable text, streamed sentence by sentence
                                             ▼
                                    AVSpeechSynthesizer ──▶ speaker
```

## Layout

| Directory | Knows about | Depends on |
|---|---|---|
| `Sources/PieKit/` | inferlets, wasm, GGUF, the C shim | `ConversationBackend` |
| `Sources/AudioKit/` | microphones, recognisers, synthesizers | nothing app-specific |
| `Sources/Conversation/` | turn-taking | protocols only |
| `Sources/UI/` | SwiftUI | the controller |
| `Sources/VoiceApp.swift` | all three, once | composition root |

Three protocols hold the seams open:

- **`ConversationBackend`** (`Conversation/ConversationBackend.swift`) —
  "answer this utterance, stream me the text." Says nothing about Pie.
- **`VoiceInput`** / **`VoiceOutput`** (`AudioKit/VoiceIO.swift`) — where
  utterances come from and where replies go.

`ConversationController` holds all three as protocol references and is
the only type that sees more than one layer at a time.

### Upgrading Pie

Everything version-specific is in **`PieKit/PieRuntimeConfig.swift`**:
the engine config TOML, the model path, and the inferlet's
id/wasm/manifest names. A new Pie release that changes the config schema
or the inferlet's parameters should be a diff to that file plus a rebuild
of `ios/pie-shim`. `AudioKit/`, `Conversation/`, and `UI/` do not import
anything Pie-shaped and should not need to change.

The C ABI itself is confined to `PieKit/PieBridge.swift` — three
`@_silgen_name` declarations and a callback trampoline.

### Swapping audio

`MicrophoneInput` and `AudioFileInput` are both `VoiceInput`. The app
ships with the second one wired to the "Sample" segment in the UI, which
plays a bundled recording through the same recogniser, controller, and
model as the microphone. That is how the voice path is exercised in the
Simulator, which has nobody to talk to it — and it is the modularity
claim tested rather than asserted.

## What is Pie-specific about it

A voice assistant is the case where re-prefilling the conversation every
turn hurts most, because the user is sitting there waiting to be spoken
to. The `voice-chat` inferlet (`inferlets/voice-chat`) keeps the
conversation's KV state inside the engine under a named snapshot:

- turn 1 builds a context, generates, `flush()`, `save(session)`
- turn *n* `take`s the snapshot back (falling back to `open`), appends
  only the new user message, generates, and re-saves

So turn *n* prefills one utterance, not the whole transcript. The number
is on screen: each assistant bubble reports KV tokens reused, new prefill
tokens, and decode rate for that turn.

The inferlet also splits its two output channels deliberately — stdout
carries only speakable text with `<think>` blocks stripped as they
stream, so chunks can go straight to the synthesizer, while the KV
accounting comes back in the return value where it can't be spoken aloud.

## Building

Prerequisites, same as `ios/demo-app`:

```bash
# 1. the shim (release — ggml at -O0 is unusably slow)
(cd ios/pie-shim && cargo build --release --target aarch64-apple-ios-sim)

# 2. the inferlet
(cd inferlets/voice-chat && cargo build --release --target wasm32-wasip2)

# 3. the model at target/qwen3-gguf/Qwen3-0.6B-Q4_K_M.gguf
#    (or pass MODEL=/path/to.gguf)

# 4. the app
bash ios/voice-app/build-app.sh
xcrun simctl install booted target/PieVoice.app
xcrun simctl launch booted org.pie-project.voice
```

The sample recordings are synthesised at build time by
`make-samples.sh` with macOS `say`, so no audio is checked in.

## Known limits

- **Simulator speech.** `supportsOnDeviceRecognition` is false until the
  en-US assets are present; the app then uses Apple's server recogniser
  and the header badge says "cloud speech" instead of "on-device speech".
  On a device with the assets installed it stays local. The badge always
  reports which one is actually in force.
- **Barge-in is client-side.** Tapping while the app is speaking stops
  the synthesizer and starts listening, but the inferlet keeps generating
  to `max_tokens`. Cancelling a running turn needs a stop path through
  the shim, which does not exist yet.
- **Hands-free is off by default.** With speakers and a microphone in one
  room the synthesizer talks into the recogniser and the app answers
  itself. It is a toggle, not a default.
- **CPU only.** The ggml portable driver, as in `ios/demo-app`. Metal on
  iOS is the next milestone (M2 in the RFC).
