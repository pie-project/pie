# qwen3-tts

Text to speech through Qwen3-TTS (12 Hz, `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`):
the talker speaks each frame's first codebook, the code predictor the other
fifteen, and the codec decoder turns the frames into 24 kHz audio. A frame's
predictor fires, its sampling and the talker's next row all run on the
device, back to back; the host only collects the codes.

## Run it

Import the checkpoint with its speech tokenizer beside it
(`speech_tokenizer/model.safetensors` linked as `speech_tokenizer.safetensors`),
then build the guest and run it:

```bash
pie model import <checkpoint> --deployment qwen3-tts-0.6b-u8g64-u4g64-kv-bf16
cargo build -p qwen3-tts --release --target wasm32-wasip3   # from examples
pie run qwen3-tts -o ./out -- --text "Hello there." --speaker ryan --stream
```

The deployment `u8g64-u4g64` keeps the talker at 8 bits and the predictor,
which runs fifteen times a frame, at 4.

## Inputs

| field | default | what |
| --- | --- | --- |
| `text` | | what to say |
| `speaker`, `language` | `ryan`, `auto` | one of the checkpoint's voices and languages |
| `stream` | false | send `speech.wav` a piece at a time as it decodes |
| `ahead` | 0 | seconds the listener already has queued when a stream starts |
| `greedy`, `seed` | false, 0 | argmax every code, or sample from `seed` |
| `codes` | false | answer the codes too |
| `frames` | | decode these frames as they are, skipping the talker |

It answers the frames made, the seconds of audio and the time spent making
and decoding them, and sends the audio as `speech.wav`.

It runs on Metal. CUDA decodes given frames (`--frames`); its frame loop
does not yet draw the predictor's codes right.
