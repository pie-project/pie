"""hf_sim.py <real model dir> <out ckpt dir> <golden.json> key=value...: a simulator gate's model:
the real model's HF config (its architecture class, head width, GQA ratio, rope, layer pattern)
with only the size fields given overridden, random weights (seed 0) saved in bf16, and its
golden: greedy decode in f32 from the saved bf16 weights, from random prompt ids.

qwen3-sim (models/qwen3_dense.poem): hf_sim.py <Qwen3-0.6B> <ckpt> <golden.json> hidden_size=256
  num_attention_heads=2 num_key_value_heads=1 head_dim=128 intermediate_size=256
  num_hidden_layers=2 vocab_size=512
qwen3-sim4: the same with num_attention_heads=4
minimax-sim (models/minimax_m2.poem): hf_sim.py <MiniMax-M2.5's config.json alone, its
  quantization_config dropped, attn_type_list cut to 2, use_mtp false> <ckpt> <golden.json>
  hidden_size=256 num_attention_heads=8 num_key_value_heads=2 head_dim=128 intermediate_size=64
  num_local_experts=4 num_experts_per_tok=2 num_hidden_layers=2 vocab_size=512 rotary_dim=64
gemma4-sim (models/gemma_4.poem): hf_sim.py <gemma-4-E2B-it> <ckpt> <golden.json> hidden_size=256
  num_attention_heads=2 num_key_value_heads=1 intermediate_size=256 num_hidden_layers=2
  vocab_size=512 vocab_size_per_layer_input=512 num_kv_shared_layers=0 _vectors=4 (a multimodal release:
  its text model alone, saved under the release's names beside its config)"""
import json, os, sys, torch
from transformers import AutoConfig, AutoModelForCausalLM
torch.set_num_threads(8)
src, out, gold = sys.argv[1:4]
top = AutoConfig.from_pretrained(src)
# a multimodal release (its text model's config `text_config`): the text model alone, saved
# under the release's names (`model.language_model.`) beside the release's config
text = getattr(top, "text_config", None) if not hasattr(top, "num_hidden_layers") else None
cfg = text if text is not None else top
# `_vectors=s`: each vector parameter (norm weights, scalars) moved by N(0, s) from its init (a
# random model whose norms stay at their init can echo its input through tied embeddings)
vectors = 0.0
for kv in sys.argv[4:]:
    k, v = kv.split("=")
    if k == "_vectors":
        vectors = float(v)
        continue
    setattr(cfg, k, type(getattr(cfg, k))(v) if getattr(cfg, k, None) is not None else int(v))
if hasattr(cfg, "layer_types") and cfg.layer_types:
    cfg.layer_types = cfg.layer_types[:cfg.num_hidden_layers]
if getattr(cfg, "per_layer_config", None) is not None:
    # (per-layer overrides past the layers kept: rebuilt from the cut config without them)
    d = cfg.to_dict()
    d["per_layer_config"] = {k: v for k, v in (d.get("per_layer_config") or {}).items() if int(k) < cfg.num_hidden_layers}
    cfg = type(cfg).from_dict(d)
# (the release's first layers' kinds, as the slice the poem cuts reads them: a config's own
# checks may force its last layer's kind otherwise, as Gemma 4's forces a global one)
if hasattr(cfg, "layer_types") and cfg.layer_types:
    kinds = list(getattr(top if text is None else top.text_config, "layer_types"))[:cfg.num_hidden_layers]
    object.__setattr__(cfg, "layer_types", kinds)
torch.manual_seed(0)
cfg.initializer_range = 0.1
m = AutoModelForCausalLM.from_config(cfg)
if vectors > 0:
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() <= 1:
                p.add_(torch.randn_like(p) * vectors)
m = m.to(torch.bfloat16)
m.save_pretrained(out, safe_serialization=True)
# (the saved bf16 weights in f32: the model as built, not reloaded, whose config's checks
# could change its layers' kinds again)
m = m.float().eval()
if text is not None:
    from safetensors.torch import load_file, save_file
    f = os.path.join(out, "model.safetensors")
    w = load_file(f)
    w = {("model.language_model." + k[len("model."):] if k.startswith("model.") else k): v for k, v in w.items()}
    save_file(w, f, metadata={"format": "pt"})
    top.text_config = cfg
    top.save_pretrained(out)
g = torch.Generator().manual_seed(1)
seq = torch.randint(0, cfg.vocab_size, (5,), generator=g).tolist()
prompt = list(seq)
steps = []
with torch.no_grad():
    for _ in range(16):
        lg = m(torch.tensor([seq]), use_cache=False).logits[0, -1]
        top = torch.topk(lg, 5)
        steps.append([[int(i), float(v)] for i, v in zip(top.indices, top.values)])
        seq.append(int(top.indices[0]))
json.dump({"model": out, "prompt": prompt, "tokens": seq, "top5": steps}, open(gold, "w"))
print("margins", [round(s[0][1] - s[1][1], 3) for s in steps])
print(json.dumps({"prompt": prompt, "tokens": seq}))
