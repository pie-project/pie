# Pie Voice — talking to an on-device model served by Pie

An iOS app you hold a spoken conversation with. Speech is transcribed on
the device, answered by Qwen3.5-0.8B running through the Pie 0.5 Metal
engine embedded in the app, and spoken back. Nothing leaves the phone.

```
   microphone ──▶ SFSpeechRecognizer ──▶ transcript
                                             │
                                             ▼
                                   ConversationController
                                             │
                       voice-chat inferlet (wasmtime Pulley)
                                             │
                          Qwen3.5-0.8B U4g64 · Metal engine
                                             │
                       speakable text, streamed sentence by sentence
                                             ▼
                                    AVSpeechSynthesizer ──▶ speaker
```

## Layout

| Directory | Knows about | Depends on |
|---|---|---|
| `Sources/PieKit/` | the inferlet, wasm, `.zt` artifacts, the C shim | `ConversationBackend` |
| `Sources/AudioKit/` | microphones, recognisers, synthesizers | nothing app-specific |
| `Sources/Conversation/` | turn-taking, the transcript | protocols only |
| `Sources/UI/` | SwiftUI | the controller |
| `Sources/VoiceApp.swift` | all three, once | composition root |

Three protocols hold the seams open:

- **`ConversationBackend`** (`Conversation/ConversationBackend.swift`) —
  "here is the conversation so far and what was just said; stream me the
  reply." Says nothing about Pie.
- **`VoiceInput`** / **`VoiceOutput`** (`AudioKit/VoiceIO.swift`) — where
  utterances come from and where replies go.

`ConversationController` holds all three as protocol references and is
the only type that sees more than one layer at a time.

### Upgrading Pie

Everything version-specific is in **`PieKit/PieRuntimeConfig.swift`**:
the engine config TOML, the model ladder and artifact lookup, the
inferlet's wasm name and version, and the voice system prompt. A new Pie
release that changes the config schema or the inferlet's inputs should be
a diff to that file plus a rebuild of `ios/pie-shim`. `AudioKit/`,
`Conversation/`, and `UI/` do not import anything Pie-shaped and should
not need to change.

The C ABI itself is confined to `PieKit/PieBridge.swift` — two
`@_silgen_name` declarations and a callback trampoline.

### Swapping audio

`MicrophoneInput` and `AudioFileInput` are both `VoiceInput`. The app
ships with the second one wired to the "Sample" segment in the UI, which
plays a bundled recording through the same recogniser, controller, and
model as the microphone. That is how the voice path is exercised in the
Simulator, which has nobody to talk to it.

## What is Pie-specific about it

A voice assistant is the case where re-prefilling the conversation every
turn hurts most, because the user is sitting there waiting to be spoken
to. The `voice-chat` inferlet (`examples/voice-chat`) is stateless: the
app sends the system prompt, the transcript so far and the new utterance
every turn, and the engine serves the shared prefix from the KV and
recurrent state it published on earlier turns under the session name.
The accounting is on screen: each assistant bubble reports prompt tokens
reused, new prefill tokens, and decode rate for that turn.

The inferlet splits its two output channels deliberately — stdout carries
only speakable text, so chunks go straight to the synthesizer, while the
token accounting comes back in the return value where it can't be spoken
aloud.

## Building

`ios/PieVoice/deploy-device.sh` does all of this for a phone. By hand:

```bash
# 1. the shim, into the repository target directory
(cd ios/pie-shim && CARGO_TARGET_DIR=$PWD/../../target \
   cargo build --release --target aarch64-apple-ios-sim)

# 2. the inferlet
(cd examples && cargo build --release --target wasm32-wasip2 -p voice-chat)
cp examples/target/wasm32-wasip2/release/voice_chat.wasm ios/PieVoice/Resources/

# 3. the model: one .zt artifact, imported by pie on the Mac
pie model import Qwen/Qwen3.5-0.8B
mkdir -p ios/PieVoice/Resources/models
cp -R "$PIE_HOME/models/Qwen--Qwen3.5-0.8B" ios/PieVoice/Resources/models/

# 4. the app
bash ios/voice-app/make-samples.sh ios/PieVoice/Resources
(cd ios/PieVoice && xcodegen) && open ios/PieVoice/PieVoice.xcodeproj
```

The sample recordings are synthesised at build time by
`make-samples.sh` with macOS `say`, so no audio is checked in.

## Known limits

- Measured on an iPhone 16 Pro with Pie 0.5 on 2026-10-08: see
  `ios/README.md` for the numbers and `ios/PieVoice/results/` for the
  raw runs. The 2026-09-21 numbers there are from the 0.4 ggml CPU build.
- **Simulator speech.** `supportsOnDeviceRecognition` is false until the
  en-US assets are present; the app then uses Apple's server recogniser
  and the header badge says "cloud speech" instead of "on-device speech".
  The badge always reports which one is actually in force.
- **Barge-in is client-side.** Tapping while the app is speaking stops
  the synthesizer and starts listening, but the inferlet keeps generating
  to `max_tokens`. Cancelling a running turn needs a stop path through
  the shim, which does not exist yet.
- **Hands-free is off by default.** With speakers and a microphone in one
  room the synthesizer talks into the recogniser and the app answers
  itself. It is a toggle, not a default.
