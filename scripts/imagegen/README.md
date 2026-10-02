# `scripts/imagegen` — verification harness for the image/video generation effort

Reference implementations, golden dumps and a diff tool for the milestones in
[`.wiki/imagegen/design.md`](../../.wiki/imagegen/design.md) (D15 = verification,
§4 = milestone order).  **Nothing here touches the Rust tree.**  Everything is
Python that produces *reference numbers*; the pie side reproduces them.

Large artifacts live **outside the repo**, under
`$PIE_IMAGEGEN_GOLDEN` (default `/root/.cache/pie-imagegen/golden/`).

```
scripts/imagegen/
  mini_dit_ref.py     M0  synthetic mini-DiT reference (no real model, no download)
  mini_dit_parity.py  M0  drives pie's `mini-dit` row against that reference
  zimage_golden.py    M1  Z-Image-Turbo golden + miniature forward
  zimage_vae_parity.py M1 drives pie's `vae.decode` reading FROM A GUEST
  flux2_golden.py     M2  FLUX.2-klein-4B golden + miniature forward
  wan22_golden.py     M3  Wan 2.2 TI2V-5B golden + miniature forwards
  ltx2_golden.py      M4  LTX-2.5 miniature: one joint video+audio step; `--vae`
                          dumps the REAL video VAE decoder over a fixed latent
  ltx2_parity.py      M4  drives pie's `ltx25-mini` row against it
  vendor/ltx_2/       M4  the LTX-2.5 reference, transcribed (see its header)
  h3_golden.py        M5  MiniMax H3 miniature forward (vendored reference)
  h3_parity.py        M5  drives pie's `minimax-h3-mini` row against it
  vendor/minimax_h3/      a dependency-free transcription of H3's DiT --
                          sglang's own package cannot be imported here
  hy3_golden.py       M6  HunyuanImage 3 miniature: one prefill + one denoise step
  hy3_parity.py       M6  drives pie's `hunyuanimage3-mini` row against it
  golden_common.py        shared tap/hook/manifest plumbing
  compare.py              npz-vs-npz diff with tolerance gates
  gates.py                EVERY gate above, one command, one table (§6)
```

---

## 1. Environment

`/root/.venv/imagegen` (uv-created, CPython 3.12.3).  System CUDA 13.0, driver
595.91.07, 4x RTX PRO 6000 Blackwell (sm_120, 97 GB each).

| package | version |
|---|---|
| torch | **2.14.0+cu130** (`torch.version.cuda == 13.0`) |
| torchvision | 0.29.0+cu130 |
| diffusers | **0.40.0** |
| transformers | 5.16.1 |
| tokenizers | 0.23.2 |
| accelerate | 1.14.0 |
| safetensors | 0.8.0 |
| huggingface-hub | 1.30.0 |
| numpy | 2.5.2 |
| Pillow | 12.3.0 |
| imageio | 2.37.4 |
| imageio-ffmpeg | 0.6.0 |
| sentencepiece | 0.2.2 |
| protobuf | 7.36.1 |
| ftfy | 6.3.1 |
| einops | 0.8.2 |
| opencv-python-headless | 5.0.0.93 |

`torch.cuda.get_arch_list()` = `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`,
so sm_120 is covered natively.

Rebuild:

```bash
uv venv --python 3.12 ~/.venv/imagegen
uv pip install --python ~/.venv/imagegen/bin/python \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match torch torchvision
uv pip install --python ~/.venv/imagegen/bin/python \
  diffusers transformers accelerate safetensors "huggingface_hub[cli]" \
  Pillow numpy imageio imageio-ffmpeg sentencepiece protobuf ftfy einops \
  opencv-python-headless
```

**The cuBLAS symlink fix in `/root/SETUP_INSTRUCTIONS.md` was NOT needed.**  The
`cu130` wheel set ships `nvidia-cublas-cu13` that matches the system CUDA 13.0
runtime; a bf16 GEMM plus SDPA both ran clean on GPU 0 with no `LD_PRELOAD` and no
symlink surgery.  If a future wheel bump reintroduces the mismatch, apply the fix
to `~/.venv/imagegen/lib/python3.12/site-packages/nvidia/cu13/lib`.

**No vendoring was required** — diffusers 0.40.0 already ships `ZImagePipeline`,
`ZImageTransformer2DModel`, `Flux2KleinPipeline`, `Flux2Pipeline`,
`Flux2Transformer2DModel`, `AutoencoderKLFlux2`, `WanPipeline`,
`WanTransformer3DModel` and `AutoencoderKLWan`.  `scripts/imagegen/vendor/` does
not exist and is not needed at this diffusers pin.

---

## 2. Model downloads (HF cache `~/.cache/huggingface/hub`)

Downloaded in full (all files, `hf download <repo>`):

| repo | bytes on disk | note |
|---|---:|---|
| `Tongyi-MAI/Z-Image-Turbo` | 32,899,676,972 (32.9 GB) | fp32 transformer + Qwen3-4B encoder + FLUX 16-ch VAE |
| `black-forest-labs/FLUX.2-klein-4B` | 23,740,013,690 (23.7 GB) | `Flux2KleinPipeline`, Qwen3 encoder, `AutoencoderKLFlux2` |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 34,203,030,526 (34.2 GB) | `WanPipeline`, umT5-xxl, `AutoencoderKLWan` (4,16,16) z=48 |

Snapshot revisions: Z-Image `f332072a…`, FLUX.2-klein-4B `e7b7dc27…`,
Wan2.2-TI2V-5B-Diffusers `b8fff731…`.

Probed but **not** downloaded (all readable with the token in
`~/.cache/huggingface/token`; sizes from `files_metadata`):

| repo | gated | files | size |
|---|---|---:|---:|
| `Lightricks/LTX-2.5` | `auto` (accepted) | 17 | 200.9 GB |
| `MiniMaxAI/MiniMax-H3` | no | 280 | 498.5 GB |
| `tencent/HunyuanImage-3.0` | no | 84 | 168.7 GB |
| `black-forest-labs/FLUX.2-dev` | `auto` (accepted) | 39 | 177.6 GB |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | no | 49 | 126.2 GB |

Every one of these returned metadata successfully, i.e. **the token has read access
to all five** — the two `gated: auto` repos are already accepted for this account.
Total if all were pulled: ~1.17 TB against ~1.9 TB free (was 1.9 TB before the
91 GB above).  Diffusers-format siblings worth knowing about:
`Lightricks/LTX-2.5-Diffusers` exists; MiniMax and HunyuanImage ship custom code only.

---

## 3. `mini_dit_ref.py` — the M0 synthetic reference

No real model, no download, CPU-only, ~1 s.  It is the smallest graph that
exercises exactly the M0 substrate patterns and nothing else.

```bash
python mini_dit_ref.py            # --init --dump --euler --dpm2m
python mini_dit_ref.py --out-dir /somewhere/else
```

`--dpm2m` integrates the Euler golden's start and schedule with DPM-Solver++ 2M
(`mini_dit_dpm2m_{fp32,bf16}.npz`); `mini_dit_parity.py all --solver dpm2m` drives
the guest's device loop (`latent::dpm2m_step`) against it, and `gates.py` runs that as
the `dpm-2m` row. `mini_dit_parity.py classes` drives the guest-stated attention
mask (`forward-pass.attention-classes`) through four modes and checks its identities
without a golden; `gates.py` runs that as `attn-classes`.

On a card smaller than the roster assumes, `gates.py` takes `PIE_GATES_ROWS`,
`PIE_GATES_MEM`, `PIE_GATES_VOXELS` (the VAE ladder asks 21 GiB at its 65 536
default) and `PIE_GATES_T2I_BIG` / `PIE_GATES_T2I_SMALL` (the text-to-image gate's
picture sizes). The mini-dit golden dir needs a `tokenizer.json` beside the weights
before `pie model import` (any BPE tokenizer; the runtime insists on one).

### Architecture (see `config.json` for the machine-readable copy)

hidden 256, 4 heads x head_dim 64, SwiGLU ratio 2 (hidden 512), RMSNorm QK-norm
(eps 1e-6), LayerNorm-no-affine (eps 1e-6), 3-axis interleaved RoPE
`dims [16,24,24]`, `theta 10000`.

| block | type | sequence it sees | modulation |
|---|---|---|---|
| 0 | single-stream joint | `[text(8) ‖ image(64)]` = 72 rows, one sequence | 6-param adaLN-Zero, `Linear(SiLU(temb))` |
| 1 | MM-DiT double-stream | text and image in **separate** rectangles, **joint** attention over the same 72-row order | 6-param **per stream**, separate `img_adaLN`/`txt_adaLN` |
| 2 | Wan-style cross-attn | image (64) only: self-attn → cross-attn → FFN | 6-param, `mod_table[6,256] + Linear(SiLU(temb)).view(B,6,256)` |

- Text stream ends after block 1 (block 2 and the head are image-only), so the
  head only ever emits velocity for patch rows.
- RoPE positions: text token `i` at `(i, 0, 0)`; image token at `(0, h, w)`.
  Applied to q and k **after** QK-norm.  Block 2's cross-attention gets **no** RoPE
  (the Wan contract) and its `norm_cross` is a LayerNorm **with** affine and **no**
  modulation; its cross residual is **ungated**.
- Cross-attn context: 16 rows of width 512, projected by one packed
  `Linear(512, 2*256)`.
- Patchify: `latent [B,16,16,16]` → `reshape(B,C,Hp,p,Wp,p)` →
  `permute(0,2,4,1,3,5)` → `[B, 64, 64]`, feature order `(c, ph, pw)`,
  token index `i = h*8 + w`.  Head is `LayerNorm-no-affine → x*(1+scale)+shift →
  Linear(256 → 64)` then the inverse permutation.
- Modulation chunk orders (load-bearing, easy to permute wrongly):
  blocks 0/1 `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`;
  block 2 `(shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn)`;
  final `(shift, scale)`.  Form is `x*(1+scale)+shift`, residual `x + gate*y`.
- Timestep embedding: `sinusoid(t, 256, max_period=10000)` = `[sin(all) ‖ cos(all)]`
  in fp32; each adaLN is `Linear(SiLU(temb))`.

Init is deterministic from seed 0 (parameters visited in **sorted name order**,
one `torch.Generator(0)`):
Linear weight `[out,in]` ~ `N(0, (1.0/sqrt(in))^2)`; bias ~ `N(0, 0.02^2)`;
QK-norm and LayerNorm gains = `1 + N(0, 0.02^2)`; `blocks.2.mod_table` ~ `N(0, 0.5^2)`.
This makes each block move the residual by **28–51 %** (measured), so a broken
block cannot pass a parity gate by accident.

### Tensor naming scheme (`mini_dit.safetensors`, 4,771,264 params, fp32)

```
x_embedder.{weight,bias}                          [256,64] [256]
blocks.0.adaLN.{weight,bias}                      [1536,256] [1536]
blocks.0.attn.qkv.{weight,bias}                   [768,256] [768]
blocks.0.attn.{norm_q,norm_k}                     [64]            # RMSNorm QK gains
blocks.0.attn.out.{weight,bias}                   [256,256] [256]
blocks.0.mlp.{gate_proj,up_proj}.{weight,bias}    [512,256] [512] # SwiGLU w1, w3
blocks.0.mlp.down_proj.{weight,bias}              [256,512] [256] # SwiGLU w2
blocks.1.{img,txt}_adaLN.{weight,bias}            [1536,256] [1536]
blocks.1.{img,txt}_attn.qkv.{weight,bias}         [768,256] [768]
blocks.1.{img,txt}_attn.{norm_q,norm_k}           [64]
blocks.1.{img,txt}_attn.out.{weight,bias}         [256,256] [256]
blocks.1.{img,txt}_mlp.{gate_proj,up_proj,down_proj}.{weight,bias}
blocks.2.mod_table                                [6,256]         # Wan scale_shift_table
blocks.2.adaLN.{weight,bias}                      [1536,256] [1536]
blocks.2.self_attn.qkv.{weight,bias}              [768,256] [768]
blocks.2.self_attn.{norm_q,norm_k}                [64]
blocks.2.self_attn.out.{weight,bias}              [256,256] [256]
blocks.2.norm_cross.{weight,bias}                 [256]           # LayerNorm WITH affine
blocks.2.cross_attn.q.{weight,bias}               [256,256] [256]
blocks.2.cross_attn.kv.{weight,bias}              [512,512] [512] # packed k|v from 512-wide ctx
blocks.2.cross_attn.{norm_q,norm_k}               [64]
blocks.2.cross_attn.out.{weight,bias}             [256,256] [256]
blocks.2.mlp.{gate_proj,up_proj,down_proj}.{weight,bias}
final_adaLN.{weight,bias}                         [512,256] [512]
final_proj.{weight,bias}                          [64,256] [64]
```

`config.json` carries the full `tensors` map, every chunk order, and the schedule.

### Dump keys — `mini_dit_dump_{fp32,bf16}.npz` (103 tensors each)

Fixed inputs: `latent [2,16,16,16]` and `text [2,8,256]` and `context [2,16,512]`
from `torch.Generator(1234)`, `timestep = [500., 250.]` (batch 2, different
timesteps).  Both files hold the same keys; `_bf16` is the bf16-emulated run stored
upcast to fp32.

```
in.latent [2,16,16,16]   in.text [2,8,256]   in.context [2,16,512]
in.timestep [2]          in.txt_pos [8,3]    in.img_pos [64,3]
rope.{cos,sin}_txt [8,32]    rope.{cos,sin}_img [64,32]
temb [2,256]   temb_silu [2,256]   patches [2,64,64]   x_embed [2,64,256]

b0.mod [2,1536]  b0.mod.{shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp} [2,256]
b0.in [2,72,256]        b0.norm1_out [2,72,256]
b0.{q_raw,k_raw,v} [2,4,72,64]      b0.{q_qknorm,k_qknorm} [2,4,72,64]
b0.{q_rope,k_rope} [2,4,72,64]      b0.attn_heads [2,4,72,64]
b0.attn_out [2,72,256]  b0.x_after_attn  b0.norm2_out  b0.mlp_out  b0.out [2,72,256]
b0.out_txt [2,8,256]    b0.out_img [2,64,256]

b1.{img,txt}_mod [2,1536]   b1.{img,txt}_mod.<same six names> [2,256]
b1.{img,txt}_norm1_out      b1.img_{q_rope,k_rope,v} [2,4,64,64]
b1.txt_{q_rope,k_rope,v} [2,4,8,64]     b1.joint_attn_heads [2,4,72,64]
b1.{img,txt}_attn_out       b1.{img,txt}_after_attn
b1.{img,txt}_norm2_out      b1.{img,txt}_mlp_out
b1.out_img [2,64,256]       b1.out_txt [2,8,256]

b2.mod_proj [2,6,256]   b2.mod [2,6,256]
b2.mod.{shift_msa,scale_msa,gate_msa,shift_ffn,scale_ffn,gate_ffn} [2,256]
b2.in  b2.norm1_out     b2.self_{q_rope,k_rope,v} [2,4,64,64]
b2.self_attn_heads [2,4,64,64]  b2.self_attn_out  b2.x_after_self
b2.cross_norm_out   b2.cross_q [2,4,64,64]  b2.cross_{k,v} [2,4,16,64]
b2.cross_attn_heads [2,4,64,64] b2.cross_attn_out  b2.x_after_cross
b2.norm3_out  b2.mlp_out  b2.out [2,64,256]

final.mod [2,512]  final.mod.shift [2,256]  final.mod.scale [2,256]
final.norm_out [2,64,256]   final.tokens [2,64,64]   velocity [2,16,16,16]
```

### Euler keys — `mini_dit_euler_{fp32,bf16}.npz` (427 tensors)

4 Euler flow-matching steps from `torch.Generator(7)` noise, sigmas
`[1.0, 0.75, 0.5, 0.25, 0.0]`, `t = sigma * 1000`, update `x += (s[i+1]-s[i])*v`:

```
euler.x_init [2,16,16,16]   euler.sigmas [5]
euler.t{0..3} [2]           euler.v{0..3} [2,16,16,16]
euler.x{1..4} [2,16,16,16]  euler.latent [2,16,16,16]
euler.s{0..3}.<every key from the dump list above>     # per-step intermediates
```

### bf16 emulation contract

`--dump`/`--euler` write both an exact-fp32 run and a bf16-emulated run.  bf16
rounds at every point a bf16 serving engine would: weights bf16; every Linear
input, weight, bias and output bf16; SwiGLU product bf16.  Reductions stay fp32
and are rounded back: LayerNorm/RMSNorm, the modulation `x*(1+s)+b`, the gated
residual add, RoPE (cos/sin fp32), the timestep sinusoid, and attention scores +
softmax (probabilities are cast to bf16 before `p·v`).

Measured fp32-vs-bf16 drift on the fixed inputs: worst max-abs 0.088, worst
relative 9.7e-3, worst cosine 0.99995 — a usable gate at
`--tol 0.1 --rel-tol 0.02 --cos-tol 0.9999`.

### Artifacts — `/root/.cache/pie-imagegen/golden/mini-dit/`

| file | bytes | md5 |
|---|---:|---|
| `config.json` | 5,995 | `9eeb9e43a9600eac66b233fcc61ac57d` |
| `mini_dit.safetensors` | 19,091,976 | `865dcbe3832b0f9fe31eb8544a738758` |
| `mini_dit_dump_fp32.npz` | 6,640,440 | `61e3f63d836bb820f8bbe0d2dcc37ddd` |
| `mini_dit_dump_bf16.npz` | 6,640,440 | `8ddf3e5b244c0bf9cad3d9266dd1152a` |
| `mini_dit_euler_fp32.npz` | 26,900,586 | `67e5976b6eadf30f78255de0725b6475` |
| `mini_dit_euler_bf16.npz` | 26,900,586 | `b53374ab535517f03d7b98eb25ac15db` |

### The pie side — `mini_dit_parity.py`

The other half of the M0 loop: it turns the golden's *inputs* into the case
JSON the `mini-dit-parity` inferlet takes, runs it, turns its JSON answer back
into an `.npz` under the golden's own key names, and diffs the two with
`compare.py` at the bf16 gate above.

```bash
# the artifact the row serves (the SKU name is `<text>-<weights>-kv-<kv>`)
cargo build -p pie --features cuda
pie model import "$PIE_IMAGEGEN_GOLDEN/mini-dit/" --sku mini-dit-bf16-kv-bf16 \
    --out ~/.cache/pie-imagegen/mini-dit.zt

# one step, then the four-step Euler schedule
python mini_dit_parity.py all --out /tmp/mini-dit-parity
python mini_dit_parity.py all --out /tmp/mini-dit-parity --euler
```

`case`, `collect` and `compare` are separate subcommands so a pie-side answer
produced any other way can be diffed too; `all` is the four in order.  One
batch element is one run (the golden's batch of 2 carries two timesteps), and
the harness stacks them back into the golden's `[2, ...]` shapes.  Patchify is
checked against the reference's own `patches` tensor on every invocation.

Both runs pass the gate (one step: cos 0.99996, max-abs 0.047; four Euler
steps: worst cos 0.99996, max-abs 0.049 on the velocities, 0.024 on the
latents).  The config the row is served under keeps `[engine] graphs = "off"`
so the parity walks eagerly.

**Bisecting a mismatch.**  `--tap <dump key>` on `run`, `collect` and
`compare` reads an INTERMEDIATE out in the velocity's place: `run` sets the
family's `PIE_MINI_DIT_TAP` knob (`crates/models/src/mini_dit/forward.rs`,
`Tap`) — the plan plants its readout seam on that rectangle, so the artifact
must be re-imported under the same environment — `collect` lays the pie rows
out the way the golden's tensor is shaped (a `[B, H, N, DH]` head tensor is
transposed back; a joint `[txt || img]` rectangle takes its caption rows from
the guest's caption-lane readout), and `compare` diffs that one key.

```bash
PIE_MINI_DIT_TAP=b0.norm1_out pie model import "$PIE_IMAGEGEN_GOLDEN/mini-dit/" \
    --sku mini-dit-bf16-kv-bf16 --out ~/.cache/pie-imagegen/mini-dit.zt --force
python mini_dit_parity.py run     --out /tmp/mini-dit-parity --tap b0.norm1_out --config ...
python mini_dit_parity.py collect --out /tmp/mini-dit-parity --tap b0.norm1_out
python mini_dit_parity.py compare --out /tmp/mini-dit-parity --tap b0.norm1_out
```

Every key the reference dumps between `x_embed` and `final.norm_out` is a
tap; `b0.in` and the `q_raw`/`v` heads included.  `run` also keeps the
server's log beside each answer (`pie_<b>.stderr`).

---

## 4. Model goldens

Each script has `--full` (real weights, GPU) and `--mini` (random-init miniature,
CPU).  With no flag it runs both.  Every run writes a `MANIFEST.json` next to the
outputs with sizes, md5s and the torch/diffusers/transformers versions used.

Common taps (in `golden_common.py`): `prepare_latents` → `noise.init*`;
`transformer.forward` call 0 → `dit.step0.in.*` and `dit.step0.out*`;
`scheduler.step` → `sched.x{1..N}`, with `latent.final` aliased to the last.

### `zimage_golden.py` → `/root/.cache/pie-imagegen/golden/z-image/`

Prompt `"a red bicycle leaning on a blue wall"`, seed 0 (CPU generator), 1024x1024,
**8 steps, guidance 0.0**, `max_sequence_length=512`, bf16.  Denoise took ~1.8 s.

Sigmas came out `[1.0, .95455, .900, .83333, .750, .64286, .500, .300, 0]` —
**exactly** the shift-3.0 values the study (`z-image.md` §K.2) predicts.
Prompt embeds are Qwen3 `hidden_states[-2]`, unpadded, `[16, 2560]` for this prompt.

Keys: `prompt_embeds.0 [16,2560]`, `prompt_embeds.lengths`, `noise.init [1,16,128,128]`,
`dit.step0.in.arg0.0 [16,1,128,128]` (latent, list-of-tensor calling convention),
`dit.step0.in.arg1 [1]` (t), `dit.step0.in.arg2.0 [16,2560]` (caption),
`dit.step0.out.0 [16,1,128,128]` (velocity), `sched.x{1..8}`, `latent.final`,
`sigmas [9]`, `timesteps [8]`, `image.rgb [1024,1024,3]`.

`--mini`: `ZImageTransformer2DModel(dim=256, n_layers=2, n_refiner_layers=2,
n_heads=4, axes_dims=[16,24,24], axes_lens=[256,64,64], cap_feat_dim=64)` —
2 noise-refiner + 2 context-refiner + 2 joint layers, 6,416,768 params.  Forward on
`x=[16,1,16,16]` (→ 64 image tokens at p=2), `cap=[8,64]`, `t=[500]`.

| file | bytes | md5 |
|---|---:|---|
| `zimage_golden.npz` | 25,498,702 | `efec906d7c786c18fcb70cc8a8707b97` |
| `zimage_golden.png` | 1,891,324 | `1acc842cdfbbfa719260561a878ed7f7` |
| `zimage_config.json` | 2,641 | `cad26e9472d8316797fb6b8eb054d991` |
| `zimage_mini.npz` | 35,856 | `d72d2751c83cafa3f266abf68049e0e0` |
| `zimage_mini.safetensors` | 25,677,496 | `fd71e61a6955381aff902cf10612c850` |
| `zimage_mini_config.json` | 7,278 | `3afce5520a90b635c009a890b866b64b` |

#### The pie side, from a guest — `zimage_vae_parity.py`

`zimage_golden.py --vae` dumps a 64x64x16 centre crop of `latent.final` beside
the pixels the reference VAE decodes it to
(`golden/z-image/zimage_vae/{latent,pixels,mean}.f32` + `shapes.json`).  The
engine gate `engine-cuda/tests/the_z_image_vae_answers_the_reference` feeds
that latent from the HOST; this script feeds it the way a guest can, through
`examples/zimage-vae-parity`: the latent bound as the reading's
`Voxels` port CHANNEL (whose declared shape is the clip's box), the answer
read off `intrinsics::pixels()`, and the picture out through
`frames.from-channel` + `session.send-frames`.

```bash
cp ~/.pie/config.minidit.toml ~/.pie/config.zimage-vae.toml
# then edit: [model] model = the imported z-image-turbo artifact,
#            [engine] graphs = "on", max_model_len = 32768,
#            [server] port = something nothing else is using
python zimage_vae_parity.py all --out /tmp/zimage-vae \
    --config ~/.pie/config.zimage-vae.toml
```

Measured: **cos 0.999981, mean |err| 0.00225, max |err| 0.1526** against
`pixels.f32` — the same distance the host-fed gate reports (0.99998 / 0.0023),
so the guest road costs nothing.  `/tmp/zimage-vae/zimage-vae.png` is the
picture: the golden's red bicycle against its blue wall, 512x512.

`--no-pixels` takes the zero-copy road only (the PNG, no rows lifted); the two
roads produce a byte-identical PNG.

### `flux2_golden.py` → `/root/.cache/pie-imagegen/golden/flux2/`

`Flux2KleinPipeline` (klein-4B is `_class_name: Flux2KleinPipeline`, **not**
`Flux2Pipeline`), same prompt/seed, 1024x1024, **4 steps** (the distilled default).
klein-4B's `transformer/config.json` has `guidance_embeds: false`, `num_layers: 5`,
`num_single_layers: 20`, `num_attention_heads: 24`, `joint_attention_dim: 7680`
— the guidance scale is explicitly ignored by the pipeline.

Text embeddings are Qwen3 layers **(9, 18, 27)** concatenated → `[1, 512, 7680]`,
with a 4-axis `text_ids [1,512,4]` (text lives on axis 3, `l = arange(L)`).

Keys: `prompt_embeds [1,512,7680]`, `text_ids [1,512,4]`,
`noise.init.0 [1,4096,128]` (packed) and `noise.init.1 [1,4096,4]` (latent ids),
`dit.step0.in.{hidden_states,encoder_hidden_states,timestep,img_ids,txt_ids}`,
`dit.step0.out [1,4096,128]`, `sched.x{1..4}`, `latent.final`, `sigmas [5]`,
`timesteps [4]`, `image.rgb`.

`--mini`: `Flux2Transformer2DModel(num_layers=2, num_single_layers=2,
num_attention_heads=2, attention_head_dim=128, joint_attention_dim=192,
guidance_embeds=True)` — head_dim stays 128 because `sum(axes_dims_rope)=128`.
Runs 64 target tokens + **2 references of 64 tokens** at RoPE `T = 10` and `T = 20`
(`num_ref_tokens=128`), so the reference `T`-offsets are exercised.  6,604,288 params.

| file | bytes | md5 |
|---|---:|---|
| `flux2_golden.npz` | 60,969,856 | `128e9be25bf3a1aa073eb4b90929902b` |
| `flux2_golden.png` | 1,936,947 | `aef3b174e01c79b4444ed283e49a873a` |
| `flux2_config.json` | 2,344 | `3993447f0ab5cd868938f1fc84f26608` |
| `flux2_mini.npz` | 226,946 | `fa365df134ae5775ec3afb1c480e89aa` |
| `flux2_mini.safetensors` | 26,422,928 | `c8127a7a22a40e2c863e48a45ed80b62` |
| `flux2_mini_config.json` | 4,530 | `0a80cb07bd98017049c77edcd9e2f79e` |

#### `--vae` — the autoencoder alone

`AutoencoderKLFlux2` in fp32 (`force_upcast`) over the centre **32x32 tokens**
of `--full`'s `latent.final`.  Both sides cut at the **128-wide /16 grid** the
transformer itself holds, which is where `models::flux_2::vae` puts its port:
the VAE codes at 32 channels on a /8 grid and the pipeline packs a 2x2 block of
those into one 128-channel cell (`_patchify_latents`), normalised by the frozen
`BatchNorm2d(128)` (`(x - running_mean)/sqrt(running_var + eps)`).  There is no
`scaling_factor`/`shift_factor` on this family.

```text
decode: latent * bn_std + bn_mean -> _unpatchify_latents -> vae.decode
encode: vae.encode(...).mean      -> _patchify_latents   -> (x - bn_mean)/bn_std
```

`flux2_vae/{latent,pixels,mean}.f32` + `shapes.json` are the same planes as raw
little-endian f32 rows of channels in `(h, w)` order — what the Rust gate loads:

```bash
CUDA_VISIBLE_DEVICES=2 python flux2_golden.py --vae
# the gate reads the row's ARTIFACT, not the snapshot: a serving load may not
# apply the `Unary` the BatchNorm planes are stated through
pie model import <the FLUX.2-klein-4B snapshot> --sku flux2-klein-4b-bf16-kv-bf16 \
    --out ~/.cache/pie-imagegen/flux2-klein-4b.zt --force
CUDA_VISIBLE_DEVICES=2 cargo test -p engine-cuda --features cuda \
    --test the_flux_2_vae_answers_the_reference -- --nocapture
```

Measured (bf16 pie vs the fp32 golden): decode `[1,32,32] -> [1,512,512]` cos
0.999994, mean |err| 0.0017, max 0.040 (not one of 786 432 values past 0.05);
encode `[1,512,512] -> [1,32,32]` cos 0.999957, mean |err| 0.0069, max 0.115.

#### The pie side — `flux2_parity.py`

`examples/flux2-parity` is the `flux2-mini` row's guest: one denoise
step over three lanes of one group — the text lane (`context` port, the raw
`[32, 192]` stack), the target lane (`latents`, 64 rows) and ONE reference
lane (`latents` again, both references' 128 rows, at their own `T`
offsets) — every lane binding `timestep` (`σ·1000`, i.e. 500) and
`guidance` (4.0, raw), the target lane alone reading out `velocity()`.
`flux2_parity.py` turns `flux2_mini.npz`'s inputs into the case, runs it,
and diffs pie's `[1, 64, 128]` against `mini.out.0[:, :64]` (the reference
discards the reference rows' predictions; pie never computes them).

```bash
# the artifact: the golden dir needs a `config.json` and a tokenizer
# beside the weights (the snapshot's `tokenizer/{tokenizer,tokenizer_config}.json`)
pie model import "$PIE_IMAGEGEN_GOLDEN/flux2/" --sku flux2-mini-bf16-kv-bf16 \
    --out ~/.cache/pie-imagegen/flux2-mini.zt
python flux2_parity.py all --out /tmp/flux2-parity --config ~/.pie/config.flux2-mini.toml
```

Measured (bf16 pie vs the fp32 golden): max-abs 0.0059, rel 0.0044, cos
0.99999 — under the mini-dit gate (`--tol 0.1 --rel-tol 0.02 --cos-tol 0.9999`).

#### The REAL row — `flux2_klein_parity.py`

`examples/flux2-klein-parity` is the `flux2-klein-4b` row against the
same dump: the `text` reading over the family's chat template (the ids are
checked against `text.input_ids` exactly), one `denoise` step over the
golden's own step-0 inputs, one independent step per sigma from the
reference's own latent, and the four-step Euler trajectory, then both final
latents through the diffusers VAE.

```bash
# ~3 min to import, ~2 min to run
pie model import ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/*/ \
    --sku flux2-klein-4b-bf16-kv-bf16 --out ~/.cache/pie-imagegen/flux2-klein-4b.zt
CUDA_VISIBLE_DEVICES=0 python flux2_klein_parity.py all \
    --out /tmp/flux2-klein-parity --config ~/.pie/config.flux2-klein.toml
```

The config needs `[engine] max_model_len = 32768` (rows × submit depth) and
`[model] model` at the artifact. It needs nothing said about `[runtime]
submit_deadline` any more: this parity used to demand `"10s"` because at the
50 ms default the cohort gate sealed the denoise group's FIRST frame with
the image lane alone and the velocity came back unconditioned (cos ≈ 0.535
against the golden, cos 0.9999 against a no-text reference). A stated cohort
is now fired whole or not at all — the deadline is a density knob again, and
this harness runs at the default so that a partial group would show up here.

Measured (bf16 pie vs the bf16 golden, `graphs = "on"`):

| reading | cos | gate |
|---|---|---|
| `text.hidden` (20 rows × 3072) | 0.999993 | ≥ 0.999 ✓ |
| `dit.step0.out` | 0.999552 | ≥ 0.999 ✓ |
| `probe.step{0,1,2,3}.out` | 0.999552 / 0.999125 / 0.999615 / 0.999767 | ≥ 0.999 ✓ |
| `latent.final` (four Euler steps) | 0.997903 | reported |
| PSNR(`pie.png`, `golden.png`) | 34.87 dB | reported |

The trajectory is REPORTED, not gated: klein is distilled to four steps
whose last takes σ 0.767 → 0, which amplifies a velocity difference about
sixfold, and bf16 alone moves the step-0 velocity by 4e-4 in cosine. The
SAME diffusers transformer in fp32 reads cos 0.999618 on the velocity and
**0.997510** on `latent.final` against this bf16 dump — i.e. an exact fp32
computation is FARTHER from the golden than pie is. The per-step velocities
are what the model is gated on.

### `wan22_golden.py` → `/root/.cache/pie-imagegen/golden/wan22/`

`WanPipeline` on TI2V-5B, 480x832, **17 frames**, **8 steps**, seed 0, bf16 DiT with
the **VAE forced to fp32** (parity requires it, `wan22.md` §K.2).
`model_index.json` has `boundary_ratio: null` (single backbone) and
`expand_timesteps: true`.

Confirmed live: the transformer receives `timestep [1, 1950]` — a **per-token**
timestep, 1950 = 5 latent frames x 15 x 26 patch grid — which is the TI2V pattern
D6/D3 must support.  For pure T2V it is uniformly 999 at step 0 (zeros appear only
in the I2V/first-frame-conditioned case).  umT5 embeds are `[1,512,4096]` with
exactly 11 non-zero rows (truncate-then-**zero**-pad, unmasked).
Scheduler is `UniPCMultistepScheduler`; sigmas came out
`[0.999999, .97225, .93758, .89301, .83361, .75050, .62594, .41861, 0]`.

Keys: `prompt_embeds`, `negative_prompt_embeds`, `prompt_embeds.nonzero_rows`,
`noise.init [1,48,5,30,52]`, `dit.step0.in.{hidden_states,timestep,encoder_hidden_states}`,
`dit.step0.out`, `sched.x{1..8}`, `latent.final`, `sigmas [9]`, `timesteps [8]`,
`vae.decode.in [1,48,5,30,52]`, `vae.decode.out [1,3,17,480,832]`, `frames.u8_shape`.

`--mini`: two random-init `WanTransformer3DModel`s, each forwarded twice (scalar
timestep `[1]` and TI2V per-token timestep `[1,320]`):
- `nano` — `attention_head_dim=24` (rope split `d-4*(d//6)` → `[8,8,8]`), 93,584 params
- `d128` — `attention_head_dim=128` (the real `[44,42,42]` split), 2,227,008 params

Both on latent `[1,16,5,16,16]` → S = 320 tokens, context `[1,32,64]`.

| file | bytes | md5 |
|---|---:|---|
| `wan22_golden.npz` | 126,117,934 | `8ded636cff5fffa8c63d0010f7e70439` |
| `wan22_golden.mp4` | 129,186 | `ef55e6377f0426f41d679155d45008a5` |
| `wan22_frames.npy` | 20,367,488 | `6239bd8693fe9c714edf290fd6e647b7` |
| `wan22_frame0.png` | 652,441 | `fd7c3ee53c089709ec80f3f84d492761` |
| `wan22_config.json` | 4,390 | `c86f6f3a472b968de24fe1f0f9f809e6` |
| `wan22_mini.npz` | 513,878 | `4393141452e7ba1c4f4f18ae0879373b` |
| `wan22_mini_nano.safetensors` | 380,704 | `743e316070fbbf918e556b9e43d92b97` |
| `wan22_mini_d128.safetensors` | 8,914,648 | `5707cde610525dbcec5ca1c95f21ec16` |
| `wan22_mini_config.json` | 11,474 | `aab5a938c58f7864cbe2a03fe69bf47d` |

#### `--vae` — the autoencoder alone, both directions

`--vae` runs the fp32 `AutoencoderKLWan` over `latent.final` and writes two
directories of raw little-endian f32 in pie's row-per-voxel `(t, h, w)`
layout, each with a `shapes.json` naming the boxes and the CHUNK boundaries
the reference's own loop produces:

- `wan22_vae/` — the DECODE. `latent.f32` (`[T·30·52, 48]`, the denoiser's
  space), `denorm.f32` (the same times `latents_std` plus `latents_mean`,
  which is what the reference hands its decoder) and `pixels.f32`
  (`[(4T−3)·480·832, 3]` in `[−1, 1]`). Chunk `k` is latent frame `k` and
  lands output frames `[chunks[k], chunks[k+1])`.
- `wan22_vae_encode/` — that decode back IN. `pixels.f32` (the encoder's
  input, the same plane), `mean.f32` (`quant_conv`'s first 48 channels —
  `DiagonalGaussianDistribution`'s MEAN, never a sample) and `latent.f32`,
  the same rows normalised into the denoiser's space, which is what pie's
  `vae.encode` arms answer. Chunk `k` takes pixel frames
  `[chunks[k], chunks[k+1])` — one frame, then four — and lands latent
  frame `k`.

```
CUDA_VISIBLE_DEVICES=3 python wan22_golden.py --vae     # needs `--full` to have run
```

| direction | shape | cos | mean \|err\| | gate |
|---|---|---:|---:|---|
| decode | 5×30×52 → 17×480×832 | 0.999986 | 0.00244 | `the_wan_2_vae_answers_the_reference` |
| encode | 17×480×832 → 5×30×52 | 0.999988 | 0.00492 | `the_wan_2_vae_encodes_the_reference` |

### `ltx2_golden.py` → `/root/.cache/pie-imagegen/golden/ltx25/`

**No `--full`.** The flagship needs the 201 GB snapshot and sglang's serving
environment, and `import sglang` needs the whole stack (starlette, orjson, …)
this box does not have. `vendor/ltx_2/modeling.py` is a self-contained
transcription of the reference classes — provenance at the top of the file,
the HUGGING FACE checkpoint's module names throughout, so ONE
`crates/models/src/ltx_2/import.rs` reads both this miniature and
`Lightricks/LTX-2.5-Diffusers`.

`--mini` writes a random-init miniature (two blocks, two heads a side at the
REAL head widths — video 128, audio 64 — the real 128-channel latents, a
16-wide caption, one connector layer apiece) under the pipeline's own
prefixes (`dit.`, `connectors.`), and dumps one joint video+audio denoise
step (72 video rows over a 3x4x6 latent grid, 8 audio rows, 16 text rows at
σ = 0.909375) plus one connector pass.

**The fixture's initialisation is chosen to DISCRIMINATE**, not to imitate
the reference's: the modulation tables are drawn at unit scale rather than
`randn / sqrt(dim)` so every gate is O(1), and the across-heads QK gains sit
at one rather than at 0.06 — under the reference's own init every gate is a
whisper, every softmax is nearly uniform, and a dropped fold or a wrong rope
hides under the tolerance. The trained tables are O(1) too.

Keys: `mini.dit.in.{latents,audio_latents,context,audio_context,timestep,
audio_timestep,positions,audio_positions}`, `mini.dit.out.{velocity,
audio_velocity}`, `mini.conn.in.{text,positions}`, `mini.conn.out.{video,
audio}`.

**Positions are dumped already normalised** — `(2·midpoint/max − 1)·π/2`, in
seconds and pixels — because that product is exactly what the pie port takes:
`RopeForm::SplitLadder` multiplies it by `theta^(f/(F−1))` and nothing else.

#### The pie side — `ltx2_parity.py`

`examples/ltx2-parity` is the `ltx25-mini` row's guest. A denoise case
is FOUR lanes of one group, each on its own pipeline: `Video` (latents,
three coordinates, timestep), `Audio` (latents, one coordinate, timestep),
`Context` (the video text context, timestep) and `Reference` (the audio text
context, timestep) — the two context lanes carry a timestep because they
modulate their own rows. A `--refine` case is the two connector passes over
one rectangle of packed trunk rows.

```bash
python ltx2_golden.py --mini
# the golden dir needs a `config.json` and a tokenizer beside the weights
pie model import "$PIE_IMAGEGEN_GOLDEN/ltx25/" --sku ltx25-mini-bf16-kv-bf16 \
    --out ~/.cache/pie-imagegen/ltx2-mini.zt
python ltx2_parity.py all          --out /tmp/ltx2-parity --config ~/.pie/config.ltx2-mini.toml
python ltx2_parity.py all --refine --out /tmp/ltx2-parity --config ~/.pie/config.ltx2-mini.toml
python ltx2_parity.py matters      --out /tmp/ltx2-parity --config ~/.pie/config.ltx2-mini.toml
```

Measured (bf16 pie vs the fp32 golden), under the mini-dit gate
(`--tol 0.1 --rel-tol 0.02 --cos-tol 0.9999`):

| case | max-abs | rel | cos |
|---|---:|---:|---|
| joint step, video velocity | 0.0247 | 0.0064 | 0.99998 |
| joint step, audio velocity | 0.0266 | 0.0061 | 0.99998 |
| `refine.video` | 0.0204 | 0.0043 | 0.99999 |
| `refine.audio` | 0.0345 | 0.0047 | 0.99999 |

The FLAGSHIP row's import is checked against the real 201 GB snapshot:
`pie model import <snapshot> --sku ltx25-bf16-kv-bf16 --dry-run` lands every
plane the flagship declares (13.0 GiB decoded — the reordered tables and the
doubled head projection — and 28.3 GiB copied through). Nothing runs the
DiT yet: the arm has no `text` reading. The VIDEO VAE DECODER runs, below.

#### The video VAE — `ltx2_golden.py --vae` → `ltx25/ltx2_vae/`

The real thing, not a miniature: diffusers 0.40's `AutoencoderKLLTX2Video`
over the snapshot's `vae/` (1.4 GB, bf16), run in fp32 over a fixed random
DiT-space latent (`torch.randn(1, 128, 3, 8, 12)` at seed 7, denormalised by
`latents_std`/`latents_mean` the way `_denormalize_latents` does) and decoded
in ONE call — the decoder is non-causal, so there is no per-frame loop and no
cache. Dumps `latent.f32` (`[T·h·w, 128]`, DiT space), `denorm.f32`,
`pixels.f32` (`[(8T−7)·32h·32w, 3]`, UNCLAMPED, rows in `(t, h, w)` order)
and `shapes.json`; `--vae-shape T,H,W` writes another size beside it as
`ltx2_vae_TxHxW/`.

```bash
CUDA_VISIBLE_DEVICES=1 python ltx2_golden.py --vae
CUDA_VISIBLE_DEVICES=1 cargo test -p engine-cuda --features cuda \
    --test the_ltx_2_vae_answers_the_reference -- --nocapture
# or, through the roster:
CUDA_VISIBLE_DEVICES=1 python gates.py --only ltx2-vae
```

The gate (`crates/engine-cuda/tests/the_ltx_2_vae_answers_the_reference.rs`)
reads the decoder's 86 planes straight out of the snapshot through
`models::ltx_2::Model::import_vae` — no artifact — fires the whole clip once
through `vae.decode`, and asserts `cos ≥ 0.9999`, `mean |err| ≤ 0.005` over
the clip AND per output frame (the end frames are where the replicate time
padding and the anchor-frame drop act). Measured, bf16 banks and activations
against the fp32 reference:

| clip | pixels | cos | mean abs err | max abs err | fire |
|---|---|---|---:|---:|---:|
| 3×8×12 (the gate's) | 17 × 256 × 384 | 0.999985 | 0.0019 | 0.034 | 0.19 s |
| 4×17×30 (`--vae-shape 4,17,30`, `PIE_LTX2_VAE_GOLDEN=ltx2_vae_4x17x30`) | 25 × 544 × 960 | 0.999985 | 0.0020 | 0.048 | 1.39 s |

Every frame of both clips sits between 0.999980 and 0.999989 — the same
distance Wan's (0.999986) and FLUX.2's (0.999994) decoders read, i.e. the
bf16 floor. What is NOT covered: the encoder, the audio VAE, the vocoder.

`matters` is the claim a parity gate cannot make on its own: **every
conditioning stream moves the answer.** It perturbs each in turn and demands
the velocity move by more than ten times the gate's own slack — the video by
its text context (1.8e-2) and by the audio latents through the a2v fold
(1.3e-2), the audio by its own text context (1.8e-2) and by the video
latents through v2a (4.8e-3). A lane that never joined the fire's attention
group would show zero there and still pass the tolerance gate.
### `h3_golden.py` → `/root/.cache/pie-imagegen/golden/minimax_h3/`

**Miniature only, and the reference is VENDORED.** `sglang.multimodal_gen`
cannot be imported in this checkout (its `__init__` pulls in starlette and the
rest of the serving stack), so `vendor/minimax_h3/modeling.py` transcribes the
eager arithmetic of `runtime/models/dits/minimax_h3.py` from sglang commit
`6cee9285a3dd43e1f0270818aef7bf01a0568863`, class by class, with the upstream
line numbers in its header. Only eager branches are kept: no TP, no
quantization, no fused kernels, no adaLN cache, no sparse attention.

`--mini` builds the study's §D.3 configuration (2 blocks, 1 refiner, dim 128,
2 heads x 64, ffn 256, text_dim 64, 8 rope frequencies → 48 of 64 rotated;
863,048 params) over a 21-row packed sequence: 3 text rows, 8 video rows
(a 2 x 4 x 4 latent under the (1,2,2) patch), 6 audio rows (3 ticks, stereo,
channel-major) and 4 keyframe rows. The four unique timesteps are
`[0.35, 0.999, 0.62, 1.0]` — video, the pinned visual condition, audio, and the
ref2va audio-reference slot nothing in FL2VA claims.

It also **checks** the claim `crates/models/src/minimax_h3/forward.rs` makes
about the packed row order: pie's `[text | video | audio | reference]` (lanes by
stream code) and the reference's `[text | refs | audio | video]` answer at
cos 1.0, because the joint attention is unmasked and every row's rotary
coordinates travel with it.

The checkpoint it writes uses the OFFICIAL tensor spellings — `qkv_proj`
re-interleaved per head — so pie's import runs the same de-interleave the 66 GB
flagship needs.

Keys: `mini.in.{text_hidden,refined_text,video_rows,audio_rows,reference_rows,
positions,token_tags,inverse_indices,unique_timesteps}`,
`mini.out.{video,audio}`.

### The pie side — `h3_parity.py`

```bash
python h3_golden.py --mini
pie model import <dir with h3_mini.safetensors> --sku minimax-h3-mini-bf16-kv-bf16
python h3_parity.py all --out /tmp/h3-parity --config ~/.pie/config.h3-mini.toml
```

The guest (`examples/h3-parity`) runs the `refine` pass (the text lane
alone) and then one `denoise` step of FOUR lanes in one attention group — text,
video, audio, reference — one pipeline each, one timestep cell per pass. Use a
PRIVATE config with its own `[server] port` and an `[engine] max_model_len` at
least the packed row count times the submit depth.

Measured (fp32 reference vs pie's bf16 trunk, one denoise step on one GPU):

| tensor | shape | max-abs | rel | cos |
|---|---|---:|---:|---:|
| `mini.out.refined_text` | (3, 128) | 0.00025 | 0.0039 | 0.99999254 |
| `mini.out.video` | (8, 96) | 0.00025 | 0.0027 | 0.99999653 |
| `mini.out.audio` | (6, 32) | 0.00020 | 0.0025 | 0.99999699 |

Gate: `--tol 0.1 --rel-tol 0.02 --cos-tol 0.9999`.

---

## 4b. `hy3_golden.py` / `hy3_parity.py` (M6) -> `$PIE_IMAGEGEN_GOLDEN/hy3/`

HunyuanImage 3.0 is an AR-plus-diffusion hybrid: one Hunyuan-A13B MoE trunk
denoises an image *inside* an LLM token sequence.  `--mini` random-inits a
two-layer, eight-expert `HunyuanImage3ForCausalMM` from the HF custom code
(`$HY3_SRC`, default the GitHub package mirror), assembles the T2I sequence by
hand -- no tokenizer, no pipeline, no flash-attn -- and dumps one causal text
prefill and one denoise step tapped at the three seams pie reads back.

```bash
CUDA_VISIBLE_DEVICES=1 python hy3_golden.py --mini      # seconds
# a directory the importer can read: the weights, config.json, tokenizer.json
pie model import $PIE_IMAGEGEN_GOLDEN/hy3/artifact \
    --sku hunyuanimage3-mini-bf16-kv-bf16 --out .../hy3-mini.zt
python hy3_parity.py all --out /tmp/hy3-parity --config ~/.pie/config.hy3-mini.toml
```

The config must be the run's OWN (its own `[server] port`), and
`[engine] max_model_len` at least the sequence.

**A whole denoise step, in three fires, each fed pie's own answer to the one
before**: `image.in` (the conv `patch_embed` on the voxel axis), `denoise`
(the trunk over the frozen prefix pages), `image.out` (the conv
`final_layer`). Plus three more denoise fires that make two claims the golden
diff cannot: `B` prefill(`<cfg>`-masked prompt) then denoise(t0) — THE PREFIX
MATTERS; `C` denoise(t1) over A's pages and `D` denoise(t1) after a fresh
prefill — THE PREFIX K/V IS REUSED EXACTLY.

Measured 2026-09-06 (fp32 golden vs bf16 weights and bf16 activations):

| tensor | shape | cos | max-abs |
|---|---|---|---|
| `image_in.rows` | (64, 256) | 0.999991 | 0.0012 |
| `denoise.hidden.image` | (64, 256) | 0.999989 | 0.0016 |
| `denoise.hidden.timestep_row` | (256,) | 0.999993 | 0.00046 |
| `uncond.hidden.image` | (64, 256) | 0.999989 | 0.0015 |
| `uncond.hidden.timestep_row` | (256,) | 0.999993 | 0.00047 |
| **`image_out.velocity.rows`** | (64, 32) | **0.999997** | **0.00012** |
| `encode.max` | (9,) | 0.999998 | 5.8e-05 |

```
[claim] PASS the prefix conditions the canvas: <cfg> moves the <timestep> row
        rel 0.0214 (reference 0.0213, 0.1% off) and the image rows rel 0.0031
        (reference 0.0020)
[claim] PASS the prefix K/V is reused exactly: max-abs 0.000e+00
```

Two shell rules shape the voxel arms: the CUDA shell seats **one voxel width
a fire**, so the timestep's sinusoid rides packed into the clip's own
rectangle (`[h, w, 32 + 256]` going in, `[h, w, D + 256]` coming out) and the
model text splits the columns; and a voxel-axis lane broadcast does not
exist, which is why the sinusoid is per voxel at all.

**The conditioning gate is stated against the reference, not as a constant.**
On this two-layer random-init miniature the prompt moves the image rows by rel
0.002 — below the bf16 parity floor — while it moves the `<timestep>` row,
which is causal over the prefix and nothing else, by rel 0.021. A guessed
absolute threshold would fail a correct model here and pass an unconditioned
one on a deeper row.

The one host round trip is between `denoise` and `image.out`: the trunk hands
back `[h*w + 1, D]` token rows and the voxel arm wants `[h, w, D]` without the
`<timestep>` row, and dropping a row is not something the guest can spell on
the device today. A production loop would carry it in an epilogue.
#### Measured — `wan22_parity.py`

```bash
pie model import <dir with wan22_mini_d128.safetensors, a config.json and a tokenizer> \
    --sku wan22-mini-d128-bf16-kv-bf16 --out ~/.cache/pie-imagegen/wan22-mini-d128.zt
python wan22_parity.py all --out /tmp/wan22-parity --config ~/.pie/config.wan22-mini.toml
python wan22_parity.py all --pertoken --out /tmp/wan22-parity --config ...
```

`d128`, both forwards: max-abs 0.00949, rel 0.00605, **cos 0.999982** — under
the gate (`--tol 0.1 --rel-tol 0.02 --cos-tol 0.9999`). The per-token forward
(two video lanes of one group, the conditioning frame at timestep 0) lands the
same number. `conditioning` passes at a move of 0.0065.

**The real row passes too.** `wan22-ti2v-5b` against `dit.step0.out` answers
**cos 0.999959** (max-abs 0.0625, rel 0.00914) under its own `--cos-tol 0.999`,
and `conditioning` moves the velocity by **0.3233** where the reference — the
same diffusers forward with the 512-row context zeroed — moves **0.3254**.
Four consecutive flagship runs answered the same cosine to every printed digit.

### The defect that used to hold it at cos 0.2740

One vector, folded into thirty times. `elementwise.add_bias` folds its bias IN
PLACE — the IR aliases `out_out` onto `out`, which is exactly what a biased
projection wants of its own matmul output — and `denoise` computes
`time_proj(silu(temb))` ONCE a fire and hands the same vector to every block.
Each block added its `scale_shift_table` to it, so block `k` modulated by
`timestep_proj + sum(table_0..table_k)` instead of `timestep_proj + table_k`.
`wan_2::forward::copy_of` now hands each block a fresh copy (two elementwise
ops on a `[lanes, 6·dim]` f32 vector), and `cargo test -p models --test
every_wan_2_fold_owns_the_vector_it_folds_into` walks every wan_2 plan on every
platform for another fold whose operand a later node still reads.

The bisection that named it runs one 1×2×2 latent — ONE token, so the
self-attention is the identity on V and DEPTH is the only variable — against
diffusers with the same planes zeroed. "K blocks live" means every block from
K on has its three output projections zeroed: all thirty still run, only the
first K write to the residual.

| live blocks K | before | after |
|---:|---|---|
| 0 (all neutralised) | 0.999998 | — |
| 1 | 0.999991 | 0.999991 |
| 2 | **0.98704** | 0.999991 |
| 4 | 0.96408 | 0.999985 |
| 8 | 0.86953 | 0.999985 |
| 15 | 0.30741 | 0.999980 |
| 30 | 0.16397 | 0.999970 |

The break is at TWO blocks and deepens monotonically — the signature of
something carried BETWEEN blocks, not inside one. It is also why the fixtures
that came first all missed it: one live block has one table to fold, and
"thirty blocks, one constant modulation" and "`time_proj` zeroed" both give
every block the SAME table, which a cumulative sum cannot distinguish from a
per-block one until the tables differ. On four whole-model cases (1, 1, 4 and
390 tokens, context 0/0/11/11 rows) the same fix moves pie from cos
0.16/0.23/0.30/0.30 to 0.99997/0.99996/0.99993/0.99978; repeat runs of those
move the last digits by ~2e-5, and nothing worse was seen in ~20 runs.

Two operational notes for the real row: `[engine] gpu_mem_utilization` must be
0.95+ (every fire — a denoise step included — demands the VAE's whole
causal-conv state watermark, ~6.4 GiB, because `state_slot_bytes` sums EVERY
`Shape::State` row of the plan with no idea which reading's arms touch them:
`crates/engine-cuda/src/store.rs`, consumed as `state_size` in
`crates/runtime/src/bootstrap.rs`. Charging a reading only the state its own
arms read is a real fix and still to be made), and the umT5 tokenizer is a
SentencePiece Unigram model `pie model import` cannot compile, so the import
needs a `tokenizer/` a BPE loader accepts (the row borrows `qwen_3`'s contract
anyway — `models::wan_2::tokenizer`).

---

## 5. `compare.py`

```bash
python compare.py A.npz B.npz [--keys 'b0.*'] [--tol 1e-3] [--rel-tol 1e-4] \
                  [--cos-tol 0.9999] [--sort-by rel] [--quiet] [--allow-missing] \
                  [--floor 0] [--floor-frac 1e-3]
```

Per tensor: `max-abs`, `rel` = `||a-b|| / ||b||`, `max-rel-el` (worst element-wise
relative error over elements of B above `max(--floor, --floor-frac * max|B|)`,
default `max(0, 1e-3 * max|B|)`, so near-zero elements do not dominate it) and
cosine similarity.  Exit 1 on any tripped gate, on a shape mismatch, or on differing key
sets (unless `--allow-missing`).  Example:

```
$ python compare.py mini_dit_dump_bf16.npz mini_dit_dump_fp32.npz \
    --tol 0.1 --rel-tol 0.02 --cos-tol 0.9999 --quiet
compared 103 tensors  worst max-abs 0.08775  worst rel 0.009689  worst cos 0.999953099
PASS
```

For the Z-Image RMSNorm bit-exactness gate (`z-image.md` §K.2 asks for
`torch.equal` on norm outputs, not a tolerance) use `--tol 0 --keys '*norm*'`.

---

## 6. `gates.py` — every gate above, one command

Six families landed with their own harnesses, their own flags and their own
idioms, and for a while nobody had run them together on one tree.  `gates.py`
does: it drives every parity gate this effort built, in a fixed order, and
prints one table plus a machine-readable summary line and an exit code.  A
person should be able to run it and tell at a glance whether `dev` is healthy.

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/imagegen/gates.py            # all of it
python scripts/imagegen/gates.py --list                            # the roster
python scripts/imagegen/gates.py --only flux2-klein --only wan-mini
python scripts/imagegen/gates.py --json /tmp/gates.json            # the table, as data
```

It re-implements nothing.  Every number in the table is printed by a harness
or by `compare.py`, and every PASS/FAIL is a harness's own exit status; this
file is the roster, the plumbing and the reaper.

### The roster, and what each expected number means

| gate | what it wraps | expected | what the number is |
|---|---|---|---|
| `mini-dit` | `mini_dit_parity.py all`, and `--euler` | cos ≥ 0.9999 (0.99996) | the M0 synthetic DiT, one step and the four-step Euler run, against the bf16-emulated reference (§3). The gate is the measured bf16 drift of a three-block trunk. |
| `zimage-mini` | `zimage_parity.py --mode mini`, `--mode mini_pad` | ≥ 0.9999 (0.999985) | the random-init Z-Image miniature, once on exact rows and once on rows that NEED padding (48→64 image, 40→64 caption). |
| `zimage-turbo` | `zimage_parity.py gate` | text 0.99999, turbo 0.9998, steps 0.9961–0.9972, PSNR 32.8–33.3 dB | the flagship: the `text` reading, the step-0 `denoise`, the whole text→refine→denoise `chain`, then the eight-step trajectory, then the VAE decode of both final latents. |
| `flux2-mini` | `flux2_parity.py all` | 0.999991 | the FLUX.2 miniature's one denoise step over three lanes of one group (text, target, references). Re-measured 2026-09-06 after the guest gave each lane a pipeline of its own: the earlier 0.99999 was taken while all three passes rode ONE pipeline, so the group never formed and the image lane denoised alone — the miniature's random-init caption sat under the tolerance and the gate stayed green. `gates.py` caught it once the frame contract began refusing a group that cannot compose. |
| `flux2-klein` | `flux2_klein_parity.py all` | per-step velocities ≥ 0.999, PSNR ≥ 34 dB | the real klein-4B row: `text.hidden`, `dit.step0.out` and `probe.step{0..3}.out` are gated; the four-step trajectory and the picture's PSNR are reported. |
| `flux2-vae` | `cargo test -p engine-cuda --features cuda --test the_flux_2_vae_answers_the_reference` | decode 0.99999, encode 0.99995 | the autoencoder alone, host-fed, against the fp32 diffusers reference. |
| `zimage-vae` | `zimage_vae_parity.py all`, then `the_z_image_vae_answers_the_reference` | 0.99998 | the same VAE reading twice: once fired FROM A GUEST through the real runtime, once from the host. The two roads must agree with the reference to the same distance. |
| `wan-mini` | `wan22_parity.py all`, `all --pertoken` | 0.999982, and the conditioning must MOVE | the `d128` miniature under both timestep forms (scalar, and TI2V's per-token one), each followed by the `conditioning` claim: zeroing the umT5 context must move the velocity by more than the gate's own slack. |
| `h3-mini` | `h3_parity.py all` | refined-text / video / audio ≥ 0.9999 | MiniMax H3's `refine` pass and one four-lane `denoise` step, three readouts. |
| `hy3-mini` | `hy3_parity.py all` | velocity 0.999997, plus two claims | HunyuanImage 3's whole denoise step in three fires, plus `the prefix conditions the canvas` and `the prefix K/V is reused exactly` (D10's claim: after step 0 the prefix is never recomputed). |
| `ltx2-mini` | `ltx2_parity.py all`, `all --refine`, `matters` | 0.999982 | the LTX-2.5 miniature's four-lane joint step, its two connector passes, and the claim that every conditioning stream moves the answer. 0.999982 is the figure measured 2026-09-06, recorded here because the family landed without one; the harness gates at `--cos-tol 0.9999`. |
| `text-to-image` | the model-agnostic guest on `flux2-klein-4b.zt` | a real PNG | 4 steps at 1024², seed 0. FLUX.2 hands back the final latent (its VAE is traced, not a declared reading), so the gate finishes with `decode_latent.py` and then checks the PNG magic, its box and its size. |

**The endpoint gates for distilled trajectories are BELOW the bf16 floor by
design, and are reported rather than gated.**  Both `zimage`'s eight-step
trajectory and `flux2-klein`'s four-step one integrate a schedule whose last
step takes a large σ to 0, which amplifies a per-step velocity difference
several-fold.  Each harness records its own measured **fp32-reference floor** —
the distance the SAME diffusers transformer in fp32 reaches against the
recorded bf16 dump:

- `zimage_parity.py`, `TOLERANCES["steps"]`: replaying the eight steps in fp32
  lands cos **0.99726** against the recorded `latent.final`. The gate is
  `--cos-tol 0.995`; the per-STEP claims (`--stop 1` at 0.9999985, the step-0
  velocity at 0.99966) are where 0.999 belongs.
- `flux2_klein_parity.py`, its NUMERICS header: fp32 reads cos **0.999618** on
  the velocity and **0.997510** on `latent.final` against the bf16 golden —
  i.e. an exact fp32 computation is FARTHER from the golden than pie is. The
  per-step velocities are gated at 0.999 and the endpoint is checked as PSNR
  after the decode.

A trajectory number below those floors is not a bug in the engine; a
per-step velocity below 0.999 is.

### The three rules the runner enforces itself

Each has been paid for once already.

- **Its own config, its own port.**  Every gate gets
  `~/.pie/config.gates-<name>.toml`, written fresh each run, with its own
  `[server] port` (probed free before it is written), its own
  `fs_scratch_dir`, and `[engine] max_model_len` at rows × submit depth —
  never the 4096 default, which kills a 1024² job on its second fire.  Several
  agents share this box and a shared config file has been repointed at another
  artifact mid-run, which reads exactly like a numerics regression.
- **A missing artifact or golden is a `skip`, not a `FAIL`.**  Each gate names
  the files it needs; when one is absent the row says `skip` and the reason
  names the file and the command that would make it (`pie model import …`,
  `python <family>_golden.py --mini`).  A red table means a regression.
- **Every gate reaps its server in a `finally`.**  A failed `pie run` has left
  a process holding 35 GiB of a card, and the NEXT gate then died with a
  misleading elastic-memory message.  Each step runs in its own process
  session, so the whole tree can be killed on a timeout, and afterwards the
  runner sweeps `/proc` for any `pie` still carrying THIS gate's config path.
  The config path is the marker, so a server belonging to another agent on the
  box is never a candidate.

### The output

```
gate      status  measured      expected                        note
--------  ------  ------------  ------------------------------  ----
mini-dit  pass    cos 0.999956  cos >= 0.9999 (landed 0.99996)

gates: verdict=HEALTHY total=1 pass=1 fail=0 skip=0 error=0 seconds=18 mini-dit=pass
```

The last line is the machine-readable one: `key=value` pairs, then one
`name=status` per gate.  Exit is 0 when nothing failed (a skip does not fail
the run) and 1 otherwise.  `--json PATH` writes the same table as data, and the
full transcript of every command — servers' logs included — lands in
`<out>/gates.log`.

### The state the roster found (2026-09-06)

Eleven of the twelve gates pass and reproduce the number their family
recorded, to the last digit the table prints.  The one red row is
`flux2-mini`, and it is not drift: the guest
`examples/flux2-parity/src/lib.rs` submits all THREE passes of its one
attention group down ONE `Pipeline`, and a pipeline is serial — the three
arrive as one lane and count once.  `97bdf6185` (frame-seal) made that a
named refusal ("attention group 0 never composed: 3 live forward passes name
the group and only 1 of them reached the runtime on a lane of its own"); the
0.99999 predates it, when the same shape silently fired the image lane alone
and a random-init miniature's missing caption sat under the tolerance.  Its
siblings — `mini-dit-parity`, `zimage-parity`, `flux2-klein-parity`,
`ltx2-parity` — already give each lane a pipeline of its own.
