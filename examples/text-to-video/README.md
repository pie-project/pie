# text-to-video

The model-agnostic text-to-video sampler: a prompt in, an mp4 out, with no
family named anywhere in the guest. The sibling of `text-to-image`, and
deliberately a sibling rather than a flag on it (imagegen design D1, D4, D8,
D11, D12).

## Run it

```bash
cargo build -p text-to-video --release --target wasm32-wasip3   # from examples
pie --config ~/.pie/config.<yours>.toml run text-to-video -o ./out -- \
    --prompt "a red bicycle leaning on a blue wall" \
    --width 832 --height 480 --frames 17 --steps 20 --seed 0 --out clip
```

The config must bind a row whose `latent()` has a temporal compression and
whose `reading` switch has a `vae.decode` value. On `wan22-ti2v-5b` that means
`max_model_len` well above the image default: 1950 latent rows plus a 512-row
context lane, times the submit depth.

## Standard input names

Like `text-to-image`, this binds a model's inputs by name: `latents` (the
VAE's own `[t, h, w, C]` latent, on the voxel axis), `timestep` and
`guidance` (one value a lane) and `context` (the text encoder's hidden rows,
on the prompt lane). They are a convention between a model and its
model-agnostic guests, not runtime vocabulary. A lane binds exactly the
inputs `model::lane_of` says it reads; one it reads by any other name is
refused by name, never zero-filled. No positions: the model computes its own
and patches its own latent.

## The path

Text encoder, then `steps` denoise fires, down one pipeline. The encoder's
rows stay on the device, zero-padded there by an ETA copy to the rows the
`context` input states (`input-decl.rows`: Wan 2.2's 512, whose reference
zero-pads umT5's answer and attends every pad row). The encoder's epilogue
draws the latent from a keyed `normal` on the device (or noises the encoded
init clip to the schedule's first sigma) and seeds the loop's clock; each
denoise epilogue integrates one Euler step, guided `u + s (c - u)` when a
negative prompt adds a second attention group to the pass. The last fire also
lands the latent on a host-read channel, because the decode below takes it
frame by frame from the guest until the decode reads it on the device.

### The frame lattice is refused, not rounded

A causal video VAE decodes one latent frame a fire on one working set, and
the first fire on it (the model's `fresh` switch) lands ONE output frame while
each later one lands `temporal-compression`. So `T` latent frames are
`1 + tc*(T - 1)` output frames and nothing else — 1, 5, 9, 13, 17, ..., 49 at
`tc = 4`. A `--frames` off that lattice is refused with the two nearest values
that are on it. The decode fires go down one pipeline on one working set (the
conv caches are its state); video2video's encode is the same read backwards.

### The pixels do cross into WASM here

A clip is several decode fires and each chunk is taken with `take_host` and
assembled into one handle with `frames.from-rgb8`.

### `--prompt-ids`, and why the row that needs it needs it

`--prompt` renders the bound model's template and tokenizes with the bound
model's vocabulary. Wan 2.2's text encoder is umT5, whose tokenizer is a
SentencePiece **Unigram** model while `crates/tokenizer` compiles BPE
pipelines alone (`models::wan_2::tokenizer` states exactly what is missing),
so that row's artifact carries somebody else's vocabulary and `--prompt` on it
would condition the DiT on ids it has never seen. `--prompt-ids` is the door
for that: tokenize with the reference tokenizer and hand the ids over — the
ENCODER still runs inside pie. The report says `"prompt_source"`, so a green
run can never be mistaken for a tokenizer that works.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("<Wan2.2-TI2V-5B snapshot>/tokenizer")
print(",".join(str(i) for i in tok("a red bicycle leaning on a blue wall")["input_ids"]))
# 289,4062,188625,346,291,1350,369,289,15258,21006,1
```

### The schedule is the family's FIXED shift

`FlowMatchEuler`'s `rows` argument turns the stated shift into a base `mu`
and bends it by the latent's row count — a resolution heuristic FLUX and
Z-Image were trained with. A video family's rows count FRAMES as well as
pixels, and neither video row in the tree rescales by them (Wan's scheduler
says `use_dynamic_shifting: false` and its 5.0 IS the shift), so this guest
passes `None`. Bending by a 1950-row volume instead put nineteen of twenty
steps above sigma 0.58 and left the last one to do the denoising: it still
produced a recognisable clip, which is exactly why it is written down.

### CFG

A model with no `guidance` input takes a negative prompt as a second
attention group of the same pass; the epilogue reads both video lanes'
velocities and steps the one latent.

## Gate

```bash
CUDA_VISIBLE_DEVICES=<n> python scripts/imagegen/gates.py --only wan-video
```

Two steps: the host-fed VAE parity
(`cargo test -p engine-cuda --test engine_cuda the_wan_2_vae_answers_the_reference`,
which is the numbers), then this guest end to end (which is the path).
