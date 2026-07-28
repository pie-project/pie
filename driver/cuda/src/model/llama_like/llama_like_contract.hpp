#pragma once

/// What the llama-like families bind (`model/registry.cpp` rows).
///
/// Four binders share this forward — plain llama/qwen, Mistral-3, Phi-3 and
/// OLMo-3 — and they differ in the contract only where the checkpoint differs.
/// Phi-3 is the one with real work: it ships `self_attn.qkv_proj` and
/// `mlp.gate_up_proj` pre-fused, and the bind path reads the six pieces.

#include "model/contract.hpp"

namespace pie_cuda_driver::model {
namespace contract_detail {

inline void phi3_qkv_split(ContractBuilder& b, const SourceTensor& raw) {
    if (raw.shape.size() != 2) {
        fail("Phi-3 fused QKV '" + std::string(raw.name) + "' must be 2-D");
    }
    const std::int64_t q_rows = raw.shape[1];
    const std::int64_t kv_rows = (raw.shape[0] - q_rows) / 2;
    if (q_rows <= 0 || kv_rows <= 0 || q_rows + 2 * kv_rows != raw.shape[0]) {
        fail("Phi-3 fused QKV '" + std::string(raw.name) + "' has an unsupported shape");
    }
    const std::int64_t cols = raw.shape[1];
    const std::string base =
        std::string(raw.name.substr(0, raw.name.size() -
                                           std::string_view(".self_attn.qkv_proj.weight").size()));
    const struct {
        std::string_view proj;
        std::int64_t start;
        std::int64_t rows;
    } specs[] = {{"q_proj", 0, q_rows},
                 {"k_proj", q_rows, kv_rows},
                 {"v_proj", q_rows + kv_rows, kv_rows}};
    for (const auto& spec : specs) {
        auto [expr, local_rows] =
            b.band(b.contract().src(std::string(raw.name)), 0, spec.start, spec.rows);
        b.push_expr(base + ".self_attn." + std::string(spec.proj) + ".weight", raw,
                  {local_rows, cols}, expr);
    }
}

inline void phi3_gate_up_split(ContractBuilder& b, const SourceTensor& raw) {
    if (raw.shape.size() != 2 || raw.shape[0] % 2 != 0) {
        fail("Phi-3 fused gate/up '" + std::string(raw.name) + "' has an unsupported shape");
    }
    const std::int64_t half_rows = raw.shape[0] / 2;
    const std::int64_t cols = raw.shape[1];
    const std::string base = std::string(
        raw.name.substr(0, raw.name.size() -
                               std::string_view(".mlp.gate_up_proj.weight").size()));
    const struct {
        std::string_view proj;
        std::int64_t start;
    } specs[] = {{"gate_proj", 0}, {"up_proj", half_rows}};
    for (const auto& spec : specs) {
        auto [expr, local_rows] =
            b.band(b.contract().src(std::string(raw.name)), 0, spec.start, half_rows);
        b.push_expr(base + ".mlp." + std::string(spec.proj) + ".weight", raw,
                  {local_rows, cols}, expr);
    }
}
/// Split Phi-3's fused QKV and gate/up back into the six tensors the
/// llama-like bind path reads.
inline void phi3_fused_splits(ContractBuilder& b) {
    for (const SourceTensor& raw : b.tensors()) {
        if (ends_with(raw.name, ".self_attn.qkv_proj.weight")) {
            phi3_qkv_split(b, raw);
        } else if (ends_with(raw.name, ".mlp.gate_up_proj.weight")) {
            phi3_gate_up_split(b, raw);
        }
    }
}

}  // namespace contract_detail

/// llama, llama3, mistral, qwen2, qwen3: the checkpoint already uses the names
/// the bind path reads. BF16 -> FP8/INT8 runtime quant is wired for these.
inline void author_llama_like_contract(ContractBuilder& b) {
    b.allow_bf16_runtime_quant();
    author_dense_contract(b);
}

/// Phi-3: undo the two source-side fusions first, so the generic tail never
/// sees the fused tensors and the dense join can re-fuse on CUDA's terms.
inline void author_phi3_contract(ContractBuilder& b) {
    contract_detail::phi3_fused_splits(b);
    author_dense_contract(b);
}
}  // namespace pie_cuda_driver::model
