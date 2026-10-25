# text-to-image

The model-agnostic text-to-image sampler: a prompt in, a picture out, with no
family named anywhere in the guest. Everything it needs to size and drive the
job is a host-answered fact (imagegen design D1, D4, D11, D12).

## Run it

Build the guest once, then run it by name. `pie run` finds a bare name in the
`examples` of the checkout the command runs in, so no `--path`:

```bash
cargo build -p text-to-image --release --target wasm32-wasip3   # from examples
pie --config ~/.pie/config.<yours>.toml run text-to-image -o ./out -- \
    --prompt "a red bicycle leaning on a blue wall" \
    --width 1024 --height 1024 --steps 4 --seed 0 --out image
```

The config must bind a generative row whose `reading` switch has a
`vae.decode` value (`pie model info` says when one binds the inputs this
sampler drives), and `[engine] max_model_len` large enough for the image's
latent tokens plus the prompt's rows.

The picture arrives as `./out/image.png`: the guest fires the model's
`vae.decode` reading on the voxel axis, reads the `pixels` seam with
`intrinsics::pixels()`, and hands the channel's cell to `send_frames`, so the
picture never enters WASM memory. A model with no `vae.decode` reading is
refused by name.

## Standard input names

The sampler binds a model's inputs by name, and knows four:

| input | what this guest puts there |
|---|---|
| `latents` | the latent, the VAE's own `[h, w, C]`, on the voxel axis |
| `timestep` | the step's timestep, one value a lane |
| `guidance` | the guidance scale, one value a lane, when the model takes it |
| `context` | the text encoder's hidden rows, on the prompt lane |

These names are a convention between a model and the model-agnostic guests
that drive it, not runtime vocabulary: the runtime checks a channel against
the input's axis, width and dtype and never reads the name. A lane binds
exactly the inputs `model::lane_of` says it reads, and an input it reads by
any other name is refused by name instead of zero-filled. Positions are not
among them: the model computes its own, and patches its own latent.

## How it finds its way around

The readings are the values of the model's `reading` switch, and what each
one is comes from what its lanes bind (`model::lane_of`):

| role | the fact that says so |
|---|---|
| text encoder | a lane that `embeds` and reads out `Hidden` |
| denoiser | a token-less lane that reads `latents` and reads out `Velocity` |
| the latent / prompt lane | the `stream` whose lane reads `latents` / `context` |
| a bridge | a token-less `Hidden` reading that turns the encoder's rows into the width `context` takes (Z-Image's `refine`) |
| VAE decode / encode | `vae.decode` / `vae.encode` |

The latent's size comes from `model::latent()` (channels and spatial
compression), the step count and trajectory from `model::schedule()`. A model
with no text reading is refused BY NAME ("this model has no text reading"),
which is the right answer for the `mini-dit` fixture.

## One pipeline, nothing back

Every pass goes down one pipeline, in submission order: the text encoder
(its rows stay on the device, zero-padded there by an ETA copy to the rows
`context` states, when it states some), the bridge, the VAE encoder for
img2img, one denoise pass a step, and the decode. The last pass before the
loop draws the latent on the device (`normal` on a keyed state, or the encoded
picture noised to the schedule's first sigma) and seeds the loop's clock;
the denoise epilogue is the sampler, `x <- x + (sigma' - sigma) v`, with `v`
the guided `u + s (c - u)` when a negative prompt adds a second attention
group to the pass. Nothing crosses into WASM until the decoded picture is
handed to the client.

## Gates

```bash
CUDA_VISIBLE_DEVICES=<n> uv run python tests/examples/test_text_to_image.py --config ~/.pie/config.<yours>.toml
CUDA_VISIBLE_DEVICES=<n> uv run python tests/examples/test_generating_images.py --config ~/.pie/config.<yours>.toml
```

`test_text_to_image.py` is the gate on the sampler (the loop, the sigmas, the
refusal). `test_generating_images.py` is the gate on the user path (the facts
`pie model list` reports, the bare-name resolution, the named file, the PNG).

Long form: `website/docs/guide/model/generating-images.mdx`.
