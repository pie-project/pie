# text-to-media

One model-agnostic generation guest: a prompt in, a picture or a clip out. An
image is one latent frame.

```bash
cargo build -p text-to-media --release --target wasm32-wasip3   # from examples
pie --config <config> run text-to-media -o ./out -- \
    --prompt "a red bicycle leaning on a blue wall" --width 1024 --height 1024 --steps 4 --out image
```

Arguments: `prompt`, `width`, `height`, `steps` (from the model's pinned
sigmas when it has them), `frames` (video models), `negative_prompt` (CFG when
`guidance > 1`), `fps`, `seed`, `guidance`, `strength`, `mask_rect`
(`x0,y0,x1,y1`), `format` (`png` for one frame, else `mp4`; also `y4m`,
`rgb8`), `out`. `width`, `height`, `steps` and `frames` are required.

Recommended: Z-Image-Turbo 1024x1024, 8 steps; FLUX.2-klein 1024x1024, 4
steps; Wan2.2-TI2V-5B 1280x704 (832x480 for speed), 121 frames (17 for a
quick clip), 50 steps (8 for a quick look).

Files (`pie run text-to-media --file NAME=PATH -- ...`): `init` (a picture, or
a y4m clip of `frames` frames, to start from) and `mask` (white repaints). A
mask needs an init. An mp4 is not read; convert it with
`ffmpeg -i in.mp4 -pix_fmt yuv420p in.y4m`.

## Standard names

The guest drives the model by convention, not by probing:

| | names |
|---|---|
| readings | `text`, `denoise`, `vae.encode`, `vae.decode` |
| denoise streams | `latents` (the latent lane), `context` (the prompt lane); others are ignored |
| inputs | `latents`, `timestep`, `guidance`, `context`, `length`; `pixels` (the `vae.encode` input) |
| outputs | `hidden`, `velocity`, `pixels` |

The text reading's rows are zero-padded to the rows `context` states and go
straight to `denoise`. The latent comes from `model::latent()`, the sigmas
from `model::schedule()` (a flow schedule).

## The sampler

Everything is on one pipeline: text, the VAE encode of an init, one denoise
pass a step, one decode pass a latent frame. The first denoise prologue draws
the start latent, `normal([seed, 0])`, mixed with the encoded init at the
first sigma for img2img. The epilogue is Euler, `x <- x + (sigma' - sigma) v`,
with `v = u + s (c - u)` under CFG; an inpaint blends the unmasked cells back
to the init noised to `sigma'` with `normal([seed, k + 1])`.
