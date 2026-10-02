# text-to-image

The model-agnostic text-to-image sampler: a prompt in, a picture out, with no
family named anywhere in the guest. Everything it needs to size and drive the
job is a host-answered fact (imagegen design D1, D4, D11, D12).

## Run it

Build the guest once, then run it by name. `pie run` finds a bare name in the
`examples` of the checkout the command runs in, so no `--path`:

```bash
cargo build -p text-to-image --release --target wasm32-wasip2   # from examples
pie --config ~/.pie/config.<yours>.toml run text-to-image -o ./out -- \
    --prompt "a red bicycle leaning on a blue wall" \
    --width 1024 --height 1024 --steps 4 --seed 0 --out image
```

The config must bind a generative row (`pie model list` marks one
`text-to-image`) and must set `[engine] max_model_len = 32768`: a 1024²
job is 4096 latent rows a lane and the frame carries two fires, so the 4096
default refuses the second fire by name.

It does NOT need `[runtime] submit_deadline` touched, and that is worth
saying because for a while it did. A denoise frame composes an attention group
out of the image lane and the context lane, and the group forms only when both
are members of one step; that wait used to be leashed by `submit_deadline`
(50 ms), and a 1024² job's first submit carries megabytes of seeded `context`
and `latents` across the guest boundary, so under load the leash fired first
and the frame sealed with the IMAGE LANE ALONE. The model then attended its
image rows with no caption and the run still SUCCEEDED, handing back a
plausible textured field with nothing of the prompt in it — measured here on
2026-09-06 at 1 of 3 correct on the eager path and 4 of 5 bodied, the wrong
ones bit-identical per trajectory, which reads exactly like a numerics
non-determinism and is not one. The runtime now keeps the promise instead (a
stated cohort fires whole or not at all, `runtime::scheduler::frame`'s
`group_short`), so this runs at the 50 ms default: 4 of 4 bit-identical and
correct. If a picture ever comes back as that textured field again, a partial
group is the first thing to check.

Two exits, and which one a model gets is its own fact.

A row that declares a drivable `vae.decode` reading gets `./out/image.png`:
the guest fires the VAE on the voxel axis, reads the `pixels` seam with
`intrinsics::pixels()`, and hands the channel's cell to `send_frames`, so the
picture never enters WASM memory. The report says `"decoded": true`.

A row whose VAE is traced but not declared as a reading (FLUX.2 today: its
decoder's mid-block attention has no kernel arm) gets the final latent under
the name the guest gave it, `./out/image.latent.f32`, plus the JSON report on
stdout, which is its sidecar. `scripts/imagegen/decode_latent.py` finishes the
job with the checkpoint's own diffusers VAE:

```bash
python scripts/imagegen/decode_latent.py --latent ./out/image.latent.f32 \
    --sidecar ./out/image.json --model-dir <the diffusers folder> --out ./out/image.png
```

## How it finds its way around

| role | the fact that says so |
|---|---|
| text encoder | `takes_tokens && readout == Hidden` |
| denoiser | `!takes_tokens && readout == Velocity` |
| VAE decode | `readout == Pixels`, and not the encoder |

The latent grid comes from `model::latent()`, the step count and trajectory
from `model::schedule()`, the row ceiling from `model::max_latent_rows()`, and
the rotary coordinates from the reading's own `positions` convention. A model
with no text reading is refused BY NAME ("this model has no text reading"),
which is the right answer for the `mini-dit` fixture.

The sampler is an eta epilogue: fire 0 seeds the latent with a keyed normal on
the device, and every later fire integrates one Euler step over `velocity()`.
Nothing large crosses into WASM.

## Gates

```bash
CUDA_VISIBLE_DEVICES=<n> uv run python tests/examples/test_text_to_image.py --config ~/.pie/config.<yours>.toml
CUDA_VISIBLE_DEVICES=<n> uv run python tests/examples/test_generating_images.py --config ~/.pie/config.<yours>.toml
```

`test_text_to_image.py` is the gate on the sampler (the loop, the sigmas, the
refusal). `test_generating_images.py` is the gate on the user path (the facts
`pie model list` reports, the bare-name resolution, the named file, the PNG).

Long form: `website/docs/guide/model/generating-images.mdx`.
