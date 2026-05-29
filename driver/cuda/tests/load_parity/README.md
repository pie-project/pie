# End-to-end weight-loading parity tests

Automated regression net for the **C++ materialize path** — the layer previously
validated only by a hand-run 32 GB GLM download + A/B. It catches ABI / copy /
FP8-dequant / FP8→MXFP4 transcode / fusion / split / packing / TP / parallel-
reader bugs in milliseconds, no downloads, on tiny synthetic checkpoints.

The harness is **composable**: a model is built from module primitives (not a
per-family builder), and the oracle derives the expected result from the source
by **byte-reconstruction** (not a hardcoded fusion map) — so it checks both the
real supported layouts *and* arbitrary random compositions with one code path.

## Files
- **`spec.py`** — the module DSL. `Recipe(model_type, dims, attn=<module>, ffn_plan=[…])`
  composes a model from primitives (`mha` / `mla` / `fused_qkv` attention;
  `dense` / `fused_gate_up` / `moe_qwen` / `moe_mixtral` FFN; DSA indexer; block-
  FP8). `named_recipes()` expresses every supported `model_type` as a composition
  (so the model_type-specific loader transforms fire); `random_recipe(rng)`
  generates random valid compositions (random dims / counts / quant) tagged with a
  TP-enabled carrier model_type matching their structure (so they can also be
  TP-materialized — see "Notes on TP").
- **`dtypes.py`** — single source of truth for dtype metadata (tag ⇄ name, element
  bytes, packed-ness, safetensors tag). `gen.py`, `parse_cache.py` and `oracle.py`
  all derive their tables from the one `DTYPES` list here.
- **`gen.py`** — writes a tiny checkpoint (config.json + single-shard safetensors
  + index, deterministic random fill; safetensors written by hand, numpy only).
- **`parse_cache.py`** — reads the materialized artifact cache (PIEWCAC3) and
  source safetensors into `{name: Tensor}`.
- **`oracle.py`** — the **generic** checker. Classifies each materialized tensor
  against the source with no per-model knowledge:
  `direct` (bytes ==), `fusion` (== ordered concat of consumed source siblings),
  `split` (a slice of a consumed source; the slices must *tile* it — catches a
  wrong offset), `skip-quant` (packed weight / derived scale — can't reconstruct
  without the quant kernel). For TP it reassembles the rank shards on their shard
  axis (shape-driven) into the full form, then runs the same classifier.
- **`run.py`** — the unified runner (see below).

## How parity is checked (three oracles, no reimplementation of the loader)
1. **differential** — the loader must agree with *itself* across orthogonal paths
   (reader on/off, fused/unfused FP8→MXFP4): byte-identical. Works for *any*
   composition; needs no reference.
2. **absolute** (tp=1) — the materialized form vs `oracle.py`'s byte-reconstruction
   from the source. Catches a bug present in *every* path (wrong fusion / split /
   placement).
3. **tp** (tp=2, ≥2 GPUs) — per-rank shards must reassemble to the source. Wrong
   shard axis → wrong reassembled bytes → caught. The loader's TP slicing is
   name-based and arch-agnostic; the *engine*, however, only starts a TP group for
   a known model_type (it inserts arch-specific collective comms in the forward
   pass). So random compositions are tagged with a TP-enabled "carrier" model_type
   matching their structure — for dense/MoE the carrier maps to the GENERIC
   ArchProfile, so the loader does exactly the generic slicing; the carrier only
   satisfies the engine gate. (The qwen3.5 named hybrids stay tp-skipped: the
   engine additionally gates them on a `linear_num_key_heads` the layout omits.)

**Quantized output** (FP8↔BF16, →MXFP4 pack + E8M0 scale) is `skip-quant` in the
absolute/tp oracles by design — reconstructing it would mean reimplementing the
quant kernel. It is covered by the **differential** runs (fused vs unfused, bit-
identical) + the dedicated **`tests/test_transcode_fused.cu`** (bit-exact vs the
two-step path).

## Run
```bash
cargo build -p pie-server --release --no-default-features --features driver-cuda
python3 driver/cuda/tests/load_parity/run.py                       # all named, all modes
python3 driver/cuda/tests/load_parity/run.py --random 20 --seed 7  # + 20 random compositions
python3 driver/cuda/tests/load_parity/run.py --mode tp glm_moe_dsa kimi_k2   # one mode/recipe
```
GPU-gated (materialize needs a device; `tp` needs two — auto-skipped on one GPU).
Exit 0 = all checks pass.

## Coverage (named `model_type`s — each a DSL composition)
- **dense (llama-family):** llama, llama3, qwen2, qwen3, mistral, mistral3,
  ministral3, olmo2, olmo3, qwen3_5, qwen3_5_text
- **phi3** — fused `qkv` / `gate_up` (the loader splits them; `split` oracle)
- **MoE:** mixtral (`w1/w2/w3`), qwen3_moe / qwen3_5_moe / qwen3_5_moe_text
- **MLA + MoE:** deepseek_v2 / v3 / v4, kimi_k2 (incl. `language_model.` prefix;
  MLA `q_kv_a` fused joins checked via the `fusion` oracle)
- **glm_moe_dsa** — MLA + DSA indexer + MoE, **block-FP8 → MXFP4 runtime quant**
  (the `#29` fused-transcode validator)
- **gemma 2/3** — pre/post-FFN norms, tied embeddings

## Notes on TP (`--mode tp`)
- TP *disables* the qkv/gate_up fusion the tp=1 path does (fusing would interleave
  per-rank shards); the oracle reassembles shards → source regardless.
- Block-FP8 axes that shard must have an even 128-block count (`dim/128 % tp == 0`);
  the DSL keeps named/random dims TP-safe.
- **qwen3_5 family** is `tp` -skipped: the *engine* (not the loader) gates tp>1 on a
  linear-attention head count (`linear_num_key_heads`) the dense layout doesn't
  model. Their loader slicing is covered by the sibling dense/MoE recipes + tp=1.
- **deepseek_v4** reports `[WARN: nothing sharded]`: `dsv4_shard_axis` matches dsv4's
  *native* `.ffn.experts.w*` naming, while its recipe uses HF-style `.mlp.experts.*`,
  so nothing shards (reassembly trivially holds). Its tp=1 placement is validated;
  native-naming tp=2 sharding needs a faithful recipe.

## Adding coverage
A new model = one `named_recipes()` entry (compose existing modules + a flag).
A new structural primitive = one module function in `spec.py` + a key in `ATTN`/`FFN`.
The oracle needs no change — it reconstructs from the source generically.
Still TODO (need their real layouts): gpt_oss (native-MXFP4), nemotron_h (Mamba),
gemma4 / gemma3n, and a native-naming deepseek_v4 recipe.
