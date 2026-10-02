# text-to-video

The model-agnostic text-to-video sampler: a prompt in, an mp4 out, with no
family named anywhere in the guest. The sibling of `text-to-image`, and
deliberately a sibling rather than a flag on it (imagegen design D1, D4, D8,
D11, D12).

## Run it

```bash
cargo build -p text-to-video --release --target wasm32-wasip2   # from examples
pie --config ~/.pie/config.<yours>.toml run text-to-video -o ./out -- \
    --prompt "a red bicycle leaning on a blue wall" \
    --width 832 --height 480 --frames 17 --steps 20 --seed 0 --out clip
```

The config must bind a row whose `latent()` has a temporal compression and
whose readings include a `vae.decode`. On `wan22-ti2v-5b` that means
`max_model_len` well above the image default: 1950 latent rows plus a 512-row
context lane, times the submit depth.

## What differs from the image sampler

| | image | video |
|---|---|---|
| the latent | an `h x w` grid | a `t x h x w` VOLUME (`LaneRows::Volume`) |
| the decode | one fire over one clip | ONE FIRE PER LATENT FRAME, in order, down one pipeline |
| the way out | a still through `take_frames` | a clip assembled from the fires, `frames.from-rgb8`, `mp4-h264` |

### The frame lattice is refused, not rounded

A causal video VAE decodes one latent frame at a time and treats the first
apart: latent frame 0 lands ONE output frame and each later one lands
`temporal-compression`. So `T` latent frames are `1 + tc*(T - 1)` output
frames and nothing else — 1, 5, 9, 13, 17, ..., 49 at `tc = 4`. A `--frames`
off that lattice is refused with the two nearest values that are on it.

### The decode fires are a sequence

Every causal convolution holds its last input frames in a `CacheRow::State`
slab the SLOT carries between fires, so frame `k` depends on frame `k-1`
having been fired into the same slot. The guest keeps two rules: one pipeline
for the whole decode (a pipeline is serial — the opposite of the denoise loop,
where one pipeline per lane is what lets a group compose), and the head arm's
pass closed before the later arm's is opened, so the later arm binds onto the
seat the head arm warmed. Firing them the other way round gives the later arm
zeroed slabs, which is a decoder with no history: measured cos 0.98 and mean
|err| 0.125 on `[-1, 1]` against the reference, versus 0.99999 warm
(`engine-cuda`'s `the_wan_2_vae_answers_the_reference`, claim 2).

### The pixels DO cross into WASM here

`text-to-image` never lets its picture into linear memory. A clip is several
fires and there is no host-side verb that concatenates their cells, so this
takes each chunk with `take_host` and builds one handle with
`frames.from-rgb8`. A 480x832x17 clip is 81 MB of f32 across the boundary and
20 MB of RGB8 back. The fix is a seam that appends rather than replaces, not
a change here.

### The context pad is a fact now

`port-fact.rows` says "my context lane is exactly this tall, whatever the
prompt was". Wan 2.2 says 512: its reference truncates umT5's answer to the
prompt's real length and zero-pads the EMBEDS back to 512, and the transformer
attends every one of those keys — the pad rows go through `text_embedder` into
a nonzero constant with real attention mass. Handing such a model a context
lane of the prompt's own height is a different model, quietly. Where the fact
is `none` (FLUX.2, Z-Image) the encoder's rows are the lane's rows and nothing
is padded.

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

### CFG does not run yet

A model with no `guidance` port takes a negative prompt as a second lane pair
whose velocity the host combines. That path is written and refuses: four lanes
on `wan22-ti2v-5b` die with `no cell available` at the first readback, because
the two branches' `out` channels are taken in the same turn and one has no
committed cell yet. Measured 2026-09-06. The clip in the gate is the
single-branch path at `guidance: 1.0`, which is why it is more saturated and
less faithful than the reference's UniPC-8-at-CFG-5.

## Gate

```bash
CUDA_VISIBLE_DEVICES=<n> python scripts/imagegen/gates.py --only wan-video
```

Two steps: the host-fed VAE parity
(`cargo test -p engine-cuda --test the_wan_2_vae_answers_the_reference`,
which is the numbers), then this guest end to end (which is the path).
