"""Find the maximum safe batch size for MPS clone operations."""
import sys, torch
sys.path.insert(0, "/Users/ingim/Workspace/pie-mac/pie/src")
from pie_backend.engine import Engine
from pie_backend.config import RuntimeConfig
import pie_backend.model.gpt_oss as gpt_oss

def noop_compact(self):
    pass
gpt_oss.ForwardPass.compact_weights = noop_compact

config = RuntimeConfig.from_args(hf_repo="openai/gpt-oss-20b")
engine = Engine.load(config)
fp = engine.forward_pass

def zero_kv_cache(kv_cache):
    for layer_cache in kv_cache:
        layer_cache.zero_()
    torch.mps.synchronize()

def generate_tokens(fp, engine, n_tokens=3):
    device = config.device
    page_size = config.kv_page_size
    kv_cache = engine.kv_cache_at_layer
    zero_kv_cache(kv_cache)

    num_pages = 2
    kv_page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)

    generated = [13225]
    for step in range(n_tokens):
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
    return generated[1:]

# Get baseline (no compact)
baseline = generate_tokens(fp, engine)
print(f"Baseline: {baseline}")

# Save originals
originals = {}
for i in range(24):
    originals[i] = {k: v for k, v in fp._layer_weights[i].items() if isinstance(v, torch.Tensor)}

def restore_all():
    for i in range(24):
        for k, v in originals[i].items():
            fp._layer_weights[i][k] = v

# Test batch sizes: clone N tensors before sync
for batch_size in [1, 2, 3, 4, 5, 6, 7, 8, 10, 14]:
    restore_all()

    count = 0
    for lw in fp._layer_weights:
        for key, val in lw.items():
            if isinstance(val, torch.Tensor):
                lw[key] = val.clone()
                count += 1
                if count % batch_size == 0:
                    torch.mps.synchronize()
    torch.mps.synchronize()  # final sync

    # Also clone embed/norm/lm_head
    fp._embed_token = fp._embed_token.clone()
    fp._norm_last = fp._norm_last.clone()
    fp._lm_head = fp._lm_head.clone()
    torch.mps.synchronize()

    tokens = generate_tokens(fp, engine)
    match = tokens == baseline
    print(f"Batch size {batch_size:3d}: {tokens} {'OK' if match else 'FAIL'}")

    if not match:
        # Restore for next test
        restore_all()
