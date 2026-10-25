# qwen3-asr

Speech to text through Qwen3-ASR (`Qwen/Qwen3-ASR-0.6B`): the clip's log-mel
through the encoder's convolutions and windowed attention, and the encoder's
rows in place of the audio tokens through the dense Qwen3 that writes the
transcript.

## Run it

```bash
pie model import Qwen/Qwen3-ASR-0.6B --deployment qwen3-asr-0.6b-u8g64-kv-bf16
cargo build -p qwen3-asr --release --target wasm32-wasip3   # from examples
pie run qwen3-asr -- --audio-b64 "$(base64 < clip.wav)"
```

The clip is a 16 kHz WAV (16-bit PCM or 32-bit float), base64: the guest
has no filesystem. A shell passes one argument of at most 128 KiB, about
three seconds of 16-bit audio; a longer clip goes through a client. It
answers the transcript and the language Qwen3-ASR names.

The engine's voxel ceiling (`max_voxels`) must hold a convolution pass's
clip: two chunks of 100 frames by 128 bands. It runs on Vulkan, Metal and
CUDA.
