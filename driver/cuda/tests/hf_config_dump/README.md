# The HF config normalizer as a golden oracle *(oracle retired 2026-08-04)*

`pie.model/1` moves HuggingFace `config.json` normalization from serve time to
import time, and from C++ to Rust. `driver/cuda/src/model/config.cpp` was what
that Rust had to agree with: 855 lines, 25 `model_type` conditionals, 219
output fields, and — as its own comment admitted — a pile of rules that existed
only because "until we add per-arch metadata, derive it here."

Two implementations of the same normalization, in two languages, that must
agree by coincidence is exactly the skew the artifact exists to kill. So the
port was checked against the original **field for field on real configs**,
rather than against someone's reading of the source — and then the original
was deleted.

## What is left, now that the oracle is gone

`config.cpp` is deleted, so nothing here can run `parse_hf_config` any more.
**`golden/` is the oracle now**: 56 files, checked in, recording exactly what
that function produced for every corpus input. Two things still read them and
both still pass —

* `check_descriptor.sh`, the round trip below;
* `model/config/tests/differential.rs`, which re-normalizes the same
  inputs in Rust and compares.

Neither needs the C++ normalizer to exist, which is what made deleting it
safe. To run the original again, check out a commit before the harvest.

What builds today is the *read* side — `descriptor.cpp`, generated from
`model/config.hpp`, which is still the schema both languages derive from:

```
./build.sh --descriptor         # -> ./dump_from_descriptor
./dump_from_descriptor some/pie.model.json
```

It prints every `HfConfig` field as JSON, or `{"error": ...}` and exit 1 when
the reader refuses — a refusal is part of the behavior being compared, not a
crash. `./build.sh` without `--descriptor` refuses and says why.

## The pieces

| File | What it is |
|---|---|
| `generate.py` | Writes `dump_hf_config.cpp` *and* `src/model/descriptor.cpp` from `model/config.hpp`. |
| `dump_hf_config.cpp` | **Generated.** Do not edit; regenerate. |
| `dump_from_descriptor.cpp` | The same emitter, fed by the new read path. |
| `build.sh` | Host-compiler build, no CUDA. `--descriptor` builds the read side. |
| `check_descriptor.sh` | The round trip (below). |
| `synthesize.py` | Writes the `synthetic--*` corpus entries. |
| `corpus/` | Inputs: 28 real configs + 27 synthetic. |
| `golden/` | `dump_hf_config` run over `corpus/`, checked in. |

Regenerating everything:

```
python3 generate.py && ./build.sh && python3 synthesize.py
for f in corpus/*.json; do ./dump_hf_config "$f" > "golden/$(basename "$f")"; done
```

**The emitter is generated on purpose.** Hand-maintaining 216 field emissions
across 7 structs would not survive one new architecture, and a field missing
from the oracle is a field missing from every comparison — silently. The first
draft of the regex required a declaration to end its line, which dropped the 24
fields carrying a trailing `//` comment (`head_dim`, `use_qk_norm`,
`torch_dtype` among them) and produced goldens that looked fine. `generate.py`
now refuses to run if any declaration in a struct body goes uncaptured.

## The corpus

`corpus/` is real configs first: 28 of them, straight from a local HF cache,
covering `qwen2_moe`, `qwen3`, `qwen3_moe`, `qwen3_vl`, `qwen3_5`,
`qwen3_5_moe`, `qwen3_next`, `llama`, `mixtral`, `phi3`, `olmo3`, `gpt_oss`,
`gemma3_text`, `gemma3n` and `gemma4`. They are the ground truth; no synthetic
config replaces one.

But a cache only holds what someone downloaded, and what it misses are not the
easy branches. `synthesize.py` covers those, each entry named for the branch it
exists to reach: MLA `head_dim` synthesis (and the `model_type` gate that
gates it), the kimi_k3-vs-kimi_linear disambiguation, the kimi_k25 wrapper,
`deepseek_v4`, `nemotron_h`'s `hybrid_override_pattern`, gemma2's hardcoded
sliding pattern, gemma3's `sliding_window_pattern`, `glm_moe_dsa`'s
`indexer_types`, CSM's nested depth-decoder and codec configs, both RoPE
scaling kinds plus the YaRN mscale derivation, and each defaulting rule in
isolation. None of those had a fixture anywhere in the repo before.

## The round trip: what said `parse_hf_config` could be deleted

`check_descriptor.sh` runs both halves of the replacement against the thing being
replaced:

```
config.json --[C++ parse_hf_config]--------------------------> golden (recorded)
config.json --[Rust normalize]--> pie.model/1 --[C++ read]---> must equal it
```

56 matched, 0 differed — before the deletion against a live oracle, and after it
against the recording, which is the same comparison because the top line was
never re-run per invocation. The reader (`src/model/descriptor.cpp`) is generated
from the same header the oracle was, so a field can only go missing from both at
once — and `generate.py` refuses to run when a declaration goes uncaptured.

The check has been shown to fail for the right reasons: making the reader take
`num_key_value_heads` from `num_attention_heads` reports exactly that field on exactly
the configs where the two differ.

## What the goldens are not

They are a record of what `config.cpp` *does*, not of what it *should* do. The
port must reproduce them, bug for bug; changing behavior is a separate and
deliberate act that regenerates them.

One such bug is already in here. `config.cpp:306` sets
`attention_has_sinks = true` for `deepseek_v4`, and `config.cpp:331` then
overwrites it unconditionally with `(model_type == "gpt_oss")`. The assignment
at 306 is dead and DeepSeek-V4 silently loses its attention sinks — visible in
`golden/synthetic--deepseek-v4.json` as `"attention_has_sinks": false`. The
golden records the bug because the port must not "fix" it by accident; fixing
it means editing `config.cpp` and regenerating.

## `head_dim_kernel` stays out of the artifact

`golden/microsoft--Phi-3-mini-4k-instruct.json` shows `head_dim: 96` and
`head_dim_kernel: 128`. That round-up comes from `kernels.def` — the set of
head dims *this build of this driver* instantiated — and is the one `HfConfig`
field that is not a fact about the checkpoint. `pie.model/1` carries `head_dim`
and the driver recomputes the rest at load; baking it in would couple artifacts
to a driver build.
