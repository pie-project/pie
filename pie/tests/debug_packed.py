"""Test packed weight approach: allocate one big tensor per dtype, copy into it."""
import sys, time, torch
sys.path.insert(0, "/Users/ingim/Workspace/pie-mac/pie/src")
from pie_backend.engine import Engine
from pie_backend.config import RuntimeConfig
import torch.nn.functional as fun
import pie_backend.model.gpt_oss as gpt_oss

def noop_compact(self):
    pass
gpt_oss.ForwardPass.compact_weights = noop_compact

config = RuntimeConfig.from_args(hf_repo="openai/gpt-oss-20b")
engine = Engine.load(config)
fp = engine.forward_pass
cfg = fp.model_config
device = config.device

# Group all weights by dtype and calculate total size per dtype
from collections import defaultdict

dtype_groups = defaultdict(list)  # dtype -> [(layer_idx, key, tensor)]
for i, lw in enumerate(fp._layer_weights):
    for key, val in lw.items():
        if isinstance(val, torch.Tensor):
            dtype_groups[val.dtype].append((i, key, val))

# Also add embed, norm, lm_head
dtype_groups[fp._embed_token.dtype].append((-1, "_embed_token", fp._embed_token))
dtype_groups[fp._norm_last.dtype].append((-1, "_norm_last", fp._norm_last))
dtype_groups[fp._lm_head.dtype].append((-1, "_lm_head", fp._lm_head))

print("Weight groups by dtype:")
for dtype, entries in dtype_groups.items():
    total = sum(t.numel() for _, _, t in entries)
    print(f"  {dtype}: {len(entries)} tensors, {total:,} elements, {total * torch.tensor([], dtype=dtype).element_size() / 1e9:.2f} GB")

# Allocate one big contiguous buffer per dtype
packed = {}
for dtype, entries in dtype_groups.items():
    total_elements = sum(t.numel() for _, _, t in entries)
    packed[dtype] = torch.empty(total_elements, dtype=dtype, device=device)
    print(f"Allocated {dtype} buffer: {packed[dtype].numel():,} elements")

torch.mps.synchronize()

# Copy weights into packed buffer and create views
for dtype, entries in dtype_groups.items():
    buf = packed[dtype]
    offset = 0
    for layer_idx, key, tensor in entries:
        n = tensor.numel()
        buf[offset:offset + n].copy_(tensor.reshape(-1))
        # Replace with view into packed buffer
        new_tensor = buf[offset:offset + n].view(tensor.shape)
        if layer_idx >= 0:
            fp._layer_weights[layer_idx][key] = new_tensor
        elif key == "_embed_token":
            fp._embed_token = new_tensor
        elif key == "_norm_last":
            fp._norm_last = new_tensor
        elif key == "_lm_head":
            fp._lm_head = new_tensor
        offset += n

torch.mps.synchronize()
print("All weights packed and replaced")

# Verify correctness
print("\n=== E2E test ===")
page_size = config.kv_page_size
kv_cache = engine.kv_cache_at_layer
num_pages = 2
kv_page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
kv_page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)

generated = [13225]
for step in range(10):
    current_ids = torch.tensor(generated, dtype=torch.long, device=device)
    embeds = fp.embed_tokens(current_ids)
    pos_ids = torch.arange(len(generated), dtype=torch.long, device=device)

    if step == 0:
        qo_indptr = torch.tensor([0, len(generated)], dtype=torch.int32, device=device)
        inp = embeds
    else:
        qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
        inp = embeds[-1:]
        pos_ids = pos_ids[-1:]

    kv_last = torch.tensor(
        [len(generated) % page_size or page_size], dtype=torch.int32, device=device
    )
    hidden = fp.transform(
        input_embeds=inp, position_ids=pos_ids, qo_indptr=qo_indptr,
        kv_cache_at_layer=kv_cache, kv_page_indices=kv_page_indices,
        kv_page_indptr=kv_page_indptr, kv_last_page_lens=kv_last,
        custom_mask=None, single_token_inference_mode=(step > 0),
        adapter_subpass=None,
    )
    logits = fp.lm_head(hidden)
    next_token = torch.argmax(logits[-1, :]).item()
    generated.append(next_token)
    print(f"  Step {step+1}: token_id={next_token}")

# Benchmark if correct
print("\n=== Performance test ===")
from flashinfer_metal._wrappers import get_seq_lens, get_batch_indices_positions
page_size = int(engine.kv_cache_at_layer[0].shape[2])
kv_page_indices2 = torch.arange(32, dtype=torch.int32, device=device)
kv_page_indptr2 = torch.tensor([0, 32], dtype=torch.int32, device=device)
kv_last_page_lens2 = torch.tensor([11], dtype=torch.int32, device=device)
qo_indptr2 = torch.tensor([0, 1], dtype=torch.int32, device=device)
seq_lens = get_seq_lens(kv_page_indptr2, kv_last_page_lens2, page_size)
batch_indices, batch_positions = get_batch_indices_positions(append_indptr=qo_indptr2, seq_lens=seq_lens, nnz=1)
position_ids_i32 = torch.tensor([42], dtype=torch.int32, device=device)
local_q = cfg.num_q_heads // fp.tp_size
local_kv = cfg.num_kv_heads // fp.tp_size
fp.wrapper_window.plan(qo_indptr2, kv_page_indptr2, kv_page_indices2, kv_last_page_lens2,
    local_q, local_kv, cfg.dim_head, page_size, causal=True, window_left=cfg.sliding_window-1,
    q_data_type=config.activation_dtype, kv_data_type=config.activation_dtype)
fp.wrapper_full.plan(qo_indptr2, kv_page_indptr2, kv_page_indices2, kv_last_page_lens2,
    local_q, local_kv, cfg.dim_head, page_size, causal=True, window_left=-1,
    q_data_type=config.activation_dtype, kv_data_type=config.activation_dtype)

def full_step():
    h = fun.embedding(torch.tensor([42], device=device), fp._embed_token)
    for i in range(cfg.num_layers):
        w = fp.wrapper_window if i % 2 == 0 else fp.wrapper_full
        h = fp.attention(h, i, position_ids_i32, engine.kv_cache_at_layer[i],
            kv_page_indices2, kv_page_indptr2, kv_last_page_lens2,
            batch_indices, batch_positions, None, w)
        h = fp.moe(h, i)
    return fp.lm_head(h)

for _ in range(5):
    full_step()
torch.mps.synchronize()

times = []
for _ in range(10):
    torch.mps.synchronize()
    t0 = time.perf_counter()
    full_step()
    torch.mps.synchronize()
    times.append((time.perf_counter() - t0) * 1000)

times.sort()
print(f"Median: {times[len(times)//2]:.1f} ms, Min: {times[0]:.1f} ms")
