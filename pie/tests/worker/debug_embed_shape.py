"""
Minimal script to debug embedding weight shapes under tensor parallel loading.
"""

import torch
from pie_backend.backend import Backend, RuntimeConfig

MODEL = "qwen-3-0.6b"
DEVICES = ["cuda:2", "cuda:3"]

print("=" * 60)
print("Embedding Weight Shape Debug")
print("=" * 60)

# Single GPU test
print("\n[1] Single GPU (cuda:2):")
config1 = RuntimeConfig.from_args(model=MODEL, device=DEVICES[0])
runtime1 = Backend(config1)

embed_weight = runtime1.engine.weights.get("embed_token")
print(f"    Config: rank={config1.rank}, world_size={config1.world_size}")
print(f"    Embed weight shape: {embed_weight.shape}")
print(
    f"    Expected full: [vocab_size, hidden_size] = [{runtime1.model_config.num_vocabs}, {runtime1.model_config.dim_hidden}]"
)

# Test embedding
token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device=config1.device)
embeddings = runtime1.engine.embed_tokens(token_ids)
print(f"    embed_tokens output shape: {embeddings.shape}")

# Clean up
del runtime1
torch.cuda.empty_cache()

# Multi-GPU config simulation (without distributed)
print("\n[2] Multi-GPU config (rank=0, world_size=2):")
config2_r0 = RuntimeConfig.from_args(model=MODEL, devices=DEVICES, rank=0)
print(
    f"    Config: rank={config2_r0.rank}, world_size={config2_r0.world_size}, device={config2_r0.device}"
)

# Load model with this config
runtime2 = Backend(config2_r0)
embed_weight2 = runtime2.engine.weights.get("embed_token")
print(f"    Embed weight shape: {embed_weight2.shape}")

# Test embedding - but without distributed, all_reduce won't work
token_ids2 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device=config2_r0.device)

# Skip the all_reduce by calling embedding directly
import torch.nn.functional as fun

raw_embed = fun.embedding(token_ids2, embed_weight2)
print(f"    Raw embedding from sharded weight: {raw_embed.shape}")

print("\n" + "=" * 60)
print("Analysis:")
if embed_weight2.shape[1] == runtime2.model_config.dim_hidden:
    print("    ✓ Embedding weight has FULL hidden_size (row parallel)")
    print("    -> embed_tokens should work with all_reduce to combine vocab shards")
else:
    expected_shard = runtime2.model_config.dim_hidden // config2_r0.world_size
    if embed_weight2.shape[1] == expected_shard:
        print("    ✗ Embedding weight has SHARDED hidden_size (column parallel)")
        print(
            f"    -> hidden_size is {embed_weight2.shape[1]} instead of {runtime2.model_config.dim_hidden}"
        )
        print("    -> Need all_gather after embedding, not all_reduce")
    else:
        print(f"    ? Unexpected shape: {embed_weight2.shape}")
print("=" * 60)
