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
  num_local_experts=4 num_experts_per_tok=2 num_hidden_layers=2 vocab_size=512 rotary_dim=64"""
import json, sys, torch
from transformers import AutoConfig, AutoModelForCausalLM
torch.set_num_threads(8)
src, out, gold = sys.argv[1:4]
cfg = AutoConfig.from_pretrained(src)
for kv in sys.argv[4:]:
    k, v = kv.split("=")
    setattr(cfg, k, type(getattr(cfg, k))(v) if getattr(cfg, k, None) is not None else int(v))
if hasattr(cfg, "layer_types") and cfg.layer_types:
    cfg.layer_types = cfg.layer_types[:cfg.num_hidden_layers]
torch.manual_seed(0)
cfg.initializer_range = 0.1
m = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16)
m.save_pretrained(out, safe_serialization=True)
m = AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.float32).eval()
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
