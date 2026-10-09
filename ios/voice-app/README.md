# Pie Voice — talking to an on-device model served by Pie

An iOS app you hold a spoken conversation with. Speech is transcribed on
the device, answered by Qwen3.5-0.8B running through the Pie 0.5 Metal
engine embedded in the app, and spoken back. Nothing leaves the phone.

```
   microphone ─▶ echo canceller ─▶ voice-activity gate ─▶ SFSpeechRecognizer
        ▲                                                        │ transcript
        │ talking over a reply cancels it                        ▼
        │                                            VoiceModeController
        │                                                        │
        │                                  ChatController ─▶ PieEngine (ConversationBackend)
        │                                                        │
        │                               voice-chat inferlet (wasmtime Pulley)
        │                               Qwen3.5-0.8B U4g64 · Metal engine
        │                                                        │ reply text, streamed
        │                                                        ▼
   speaker ◀── AudioEngineHub ◀── AVSpeechSynthesizer.write, sentence by sentence
```

The same conversations are also a typed chat, laid out like ChatGPT's
iPhone app (sidebar of saved chats, composer with dictation and
attachments, Markdown replies, "Think longer" mode) in the palette of
Lin Zhong's site. Voice mode is a full-screen view over the open chat, and
its turns land in that chat.

## Layout

| Directory | Knows about | Depends on |
|---|---|---|
| `Sources/Core/` | the shared contracts: chat models, the backend and audio protocols | nothing |
| `Sources/PieKit/` | the inferlet, wasm, `.zt` artifacts, the C shim | `Core/` |
| `Sources/AudioKit/` | the audio engine and session, microphones, recognisers, synthesizers | `Core/` |
| `Sources/Conversation/` | chats, their storage, voice mode and dictation turn-taking | `Core/` protocols |
| `Sources/UI/` | SwiftUI: chat, voice mode, settings, the screenshot tour | the controllers |
| `Sources/Benchmark/` | the `-PieBenchmark 1` scripted run | `ConversationBackend` |
| `Sources/VoiceApp.swift` | all of them, once | composition root |

Three protocols hold the seams open:

- **`ConversationBackend`** (`Core/ConversationBackend.swift`): "here is
  the conversation so far; stream me the reply, and stop when I say so."
  Says nothing about Pie.
- **`SpeechInput`** / **`SpeechOutput`** (`Core/AudioContracts.swift`):
  where utterances come from and where replies go.

`ChatController` (typed chat, read-aloud), `VoiceModeController` and
`DictationController` hold these as protocol references; only
`VoiceApp.swift` names the concrete types.

### Upgrading Pie

Everything version-specific is in **`PieKit/PieRuntimeConfig.swift`**:
the engine config TOML, the model ladder and artifact lookup, the
inferlet's wasm name and version, the system prompts and the reply
options. A new Pie release that changes the config schema or the
inferlet's inputs should be a diff to that file plus a rebuild of
`ios/pie-shim`. `AudioKit/`, `Conversation/`, and `UI/` do not import
anything Pie-shaped and should not need to change.

The C ABI itself (version 3) is confined to `PieKit/PieBridge.swift`:
three `@_silgen_name` declarations (`pie_ios_run_stream`,
`pie_ios_cancel`, `pie_ios_free`) and a callback trampoline that tells
reply text from reasoning.

### Talking over a reply

Speech output and the microphone share one `AVAudioEngine`
(`AudioKit/AudioEngineHub.swift`) whose input node has voice processing
on in voice mode, so the reply coming out of the speaker is cancelled out
of what the microphone hears. That lets voice mode keep listening while
it speaks: the user starting to talk is the barge-in signal, which
silences the speech and cancels the engine's turn (`pie_ios_cancel`
terminates the inferlet's process), and what they are saying is already
being transcribed as the next question.

### Sample questions

`MicrophoneInput` and `SampleQuestionInput` are both `SpeechInput`. The
second plays a bundled recording through the same engine, recogniser,
controller and model as the microphone, from voice mode's menu or the
composer's "+" sheet. It finalises only when the recording has finished
playing, so the reply never starts over the question. That is also how
the voice path is exercised in the Simulator, which has nobody to talk
to it.

## What is Pie-specific about it

A voice assistant is the case where re-prefilling the conversation every
turn hurts most, because the user is sitting there waiting to be spoken
to. The `voice-chat` inferlet (`examples/voice-chat`) is stateless: the
app sends the system prompt, the transcript so far and the new utterance
every turn, and the engine serves the shared prefix from the KV and
recurrent state it published on earlier turns under the session name.
The accounting is on screen: each assistant bubble reports prompt tokens
reused, new prefill tokens, and decode rate for that turn.

The inferlet splits its output channels deliberately: stdout carries only
reply text, so chunks go straight to the screen and the synthesizer;
reasoning in "Think longer" mode streams as session messages, shown under
"Thought for Ns" and never spoken; the token accounting comes back in the
return value, where it can't be spoken aloud.

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
  and Settings shows Transcription as "Apple's servers" instead of "On
  this iPhone". It always reports which one is actually in force.
- **Barge-in depends on the echo canceller.** The onset thresholds in
  `AudioKit/VoiceActivityDetector.swift` were tuned on a Mac harness and
  are checked on the phone by `-PieAudioCheck 1` (`no_false_bargein`,
  `external_onset`). With the mic muted there is no barge-in; a tap on the
  orb still interrupts.
