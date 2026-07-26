#pragma once

/// What GLM-5.1 binds (`model/registry.cpp` row).
///
/// MLA attention plus a DSA indexer plus routed and shared MoE. Nothing about
/// the checkpoint's layout is unusual; what is unusual is that this is the one
/// family whose routed experts this driver will re-quantize at load time, and
/// the one that ships FP8 experts a runtime FP4 request is allowed to consume.

#include "model/contract.hpp"
#include "model/glm5/glm5.hpp"

namespace pie_cuda_driver::model {

/// glm_moe_dsa. `embed_tokens` is sharded on axis 0 under TP to save per-rank
/// memory; the FP4 path touches routed and shared experts only, because there
/// is no FP4 GEMM for the attention projections on this hardware.
inline void author_glm5_contract(ContractBuilder& b) {
    b.shard_embed_tokens();
    b.allow_bf16_runtime_quant();
    b.allow_mxfp4_runtime_quant();
    // GLM-5.2 ships routed experts one tensor per expert; glm5_forward reads
    // the fused 3-D slabs. Float only: this family's quantised checkpoints
    // keep the per-expert layout and take the per-expert forward path.
    //
    // `gate_second` publishes each expert's halves as `[up | gate]`, which is
    // what flashinfer's CUTLASS grouped GEMM reads fc1 as. Stating it here is
    // the whole point: the alternative is a driver-side block swap over the
    // largest tensor in the model, done after the loader has already placed it.
    contract_detail::hf_moe_expert_stacks(b, glm5_moe_gate_up_swapped(),
                                          /*float_only=*/true);
    author_dense_contract(b);
}

}  // namespace pie_cuda_driver::model
