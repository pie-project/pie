# text-to-media

One model-agnostic generation guest: a prompt in, a picture or a clip out. An
image is one latent frame.

```bash
cargo build -p text-to-media --release --target wasm32-wasip3   # from examples
pie --config <config> run text-to-media -o ./out -- \
    --prompt "a red bicycle leaning on a blue wall" --width 1024 --height 1024 --steps 4 --out image
```

Arguments: `prompt`, `negative_prompt` (CFG when `guidance > 1`), `width`,
`height`, `frames` (default 1 when the latent has no temporal compression,
else 17), `fps`, `steps`, `seed`, `guidance`, `strength`, `mask_rect`
(`x0,y0,x1,y1`), `format` (`png` for one frame, else `mp4`; also `y4m`,
`rgb8`), `out`.

Files (`pie run text-to-media --file NAME=PATH -- ...`): `init` (a picture to
start from) or `init_0`, `init_1`, ... (one still per frame), and `mask` (white
repaints). A mask needs an init.

## Standard names

The guest drives the model by convention, not by probing:

| | names |
|---|---|
| readings | `text`, `denoise`, `vae.encode`, `vae.decode` |
| denoise streams | `latents` (the latent lane), `context` (the prompt lane); others are ignored |
| inputs | `latents`, `timestep`, `guidance`, `context`, `length` |
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
