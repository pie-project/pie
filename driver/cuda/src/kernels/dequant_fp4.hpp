#pragma once

// MXFP4 (E2M1 + E8M0 block scales) → bf16 dequantization. Used by
// GPT-OSS — its expert weights ship as packed uint8 (two 4-bit values
// per byte) plus per-32-element fp8-E8M0 block scales.
//
// Layout:
//   * `packed`      — uint8 [out_dim, in_dim/2]; low nibble = element 2k,
//                     high nibble = element 2k+1.
//   * `block_scale` — uint8 [out_dim, in_dim/32]; single E8M0 byte per
//                     32-element row block. E8M0 is a "bias-127 power-of-2",
//                     i.e. `scale = 2^(byte - 127)`.
//   * `out`         — bf16 [out_dim, in_dim].
//
// FP4 codepoints (E2M1, signed): 8 positive + 8 negative values, stored
// in a 16-entry LUT inside the kernel. Standard OCP MX spec values:
//   {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_dequant_mxfp4_to_bf16(
    const std::uint8_t* packed,        // [out_dim, in_dim/2]
    const std::uint8_t* block_scale,   // [out_dim, in_dim/32]
    void*               out,           // [out_dim, in_dim] bf16
    int                 out_dim,
    int                 in_dim,        // must be a multiple of 32
    cudaStream_t        stream);

// ── Fused MXFP4 MoE decode GEMVs ────────────────────────────────────────
//
// GPT-OSS's expert weights are MXFP4, and the only path that existed for
// them materialized each routed expert's full bf16 weight matrix before
// handing it to cuBLAS. At decode that is 63% of the whole step: every
// token rewrites ~1.6 GiB of bf16 it reads exactly once. These GEMVs read
// the packed nibbles directly, so the traffic is the 4-bit weight and
// nothing else.
//
// Routing is taken from `topk_idx` on the device, so the caller needs no
// D2H round trip and the whole MoE block stays enqueue-only.
//
// `gate_up` rows are interleaved as HF ships them: row 2i is gate row i,
// row 2i+1 is up row i (see `deinterleave_rows_bf16_kernel`).

// Expert-grouped gate/up decode: one block per (expert, row slab), walking the
// expert's own routes so its weight slab is streamed once for up to four
// tokens instead of once per route. `sorted_route_ids` / `counts` come from
// `launch_moe_bucket_exact`. Output layout is identical to the per-route
// kernel (indexed by original route id), so the consumers are unchanged.
void launch_mxfp4_moe_gate_up_decode_grouped_bf16(
    const void* act_fp16,
    const std::int32_t* sorted_route_ids,
    const std::int32_t* counts,
    const std::uint8_t* const* gate_up_packed,
    const std::uint8_t* const* gate_up_scales,
    const void* const* gate_bias,
    const void* const* up_bias,
    void* gate_out_bf16,
    void* up_out_bf16,
    int num_experts, int top_k, int hidden, int intermediate,
    cudaStream_t stream);

void launch_mxfp4_moe_gate_up_decode_bf16(
    const void*          act_fp16,      // [num_tokens, hidden] fp16
    const std::int32_t*  topk_idx,      // [num_tokens * top_k]
    const std::uint8_t* const* gate_up_packed,  // per-expert [2I, H/2]
    const std::uint8_t* const* gate_up_scales,  // per-expert [2I, H/32]
    const void* const*   gate_bias,     // per-expert [I] bf16, or null
    const void* const*   up_bias,       // per-expert [I] bf16
    void*                gate_out_bf16, // [routes, I]
    void*                up_out_bf16,   // [routes, I]
    int                  num_tokens,
    int                  top_k,
    int                  hidden,
    int                  intermediate,
    cudaStream_t         stream);

void launch_mxfp4_moe_down_decode_bf16(
    const void*          act_fp16,      // [routes, intermediate] fp16
    const std::int32_t*  topk_idx,      // [num_tokens * top_k]
    const std::uint8_t* const* down_packed,  // per-expert [H, I/2]
    const std::uint8_t* const* down_scales,  // per-expert [H, I/32]
    const void* const*   down_bias,     // per-expert [H] bf16, or null
    void*                out_bf16,      // [routes, hidden]
    int                  num_tokens,
    int                  top_k,
    int                  hidden,
    int                  intermediate,
    cudaStream_t         stream);

}  // namespace pie_cuda_driver::kernels
