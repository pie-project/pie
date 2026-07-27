#pragma once

/// What GLM-5.1 binds (`model/registry.cpp` row).
///
/// MLA attention plus a DSA indexer plus routed and shared MoE. Nothing about
/// the checkpoint's layout is unusual; what is unusual is that this is the one
/// family whose routed experts this driver will re-quantize at load time, and
/// the one that ships FP8 experts a runtime FP4 request is allowed to consume.

#include "model/contract.hpp"

namespace pie_cuda_driver::model {

/// glm_moe_dsa. `embed_tokens` is sharded on axis 0 under TP to save per-rank
/// memory; the FP4 path touches routed and shared experts only, because there
/// is no FP4 GEMM for the attention projections on this hardware.
inline void author_glm5_contract(ContractBuilder& b) {
    b.shard_embed_tokens();
    b.allow_bf16_runtime_quant();
    b.allow_mxfp4_runtime_quant();
    author_dense_contract(b);
}

}  // namespace pie_cuda_driver::model
