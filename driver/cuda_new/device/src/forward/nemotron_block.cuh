#pragma once

// Composed Nemotron-H / Mamba-2 mixer block forward (prefill, bf16). Chains the
// banked + validated primitives, matching driver/cuda's nemotron_h_forward.cpp
// mamba_layer():
//
//   hidden [T,H]
//     -> in_proj GEMM            : projected [T, d_in_proj]   (act@W^T)
//     -> mamba_split             : conv_in [T, conv_dim], dt [T, num_heads]
//                                  (z/gate left in `projected`, read in place)
//     -> causal_conv1d (prefill) : conv_out [T, conv_dim]  (silu'd depthwise conv
//                                  over the packed x|B|C; state = zero pad)
//     -> prepare_mamba_params    : A=-exp(A_log), D, dt_bias -> fp32 [num_heads]
//     -> ssm_selective_scan      : scan_y [T, intermediate]  (state starts zero)
//     -> rmsnorm_gated           : core [T, intermediate] = gated RMSNorm(scan_y, z)
//     -> out_proj GEMM           : [T,H]  (act@W^T), beta=1 fuses residual add
//   = hidden + mixer(hidden)   (in place)
//
// FORMULATION: matches upstream exactly. The in_proj split is the gate==nullptr
// ("conv+dt") path: the gate/z slice is NOT copied out — it stays in `projected`
// and the gated RMSNorm reads it in place via gate_stride = d_in_proj. The scan
// reads x/B/C directly from the conv output with the packing the wired scan
// expects (see ssm_scan.cuh). State (conv + recurrent SSM) starts zero for a
// fresh prefill, which is plain causal zero-padding + zero recurrent init.
//
// =====================================================================
// DIMENSIONS
//   H            = hidden_size
//   num_heads    = mamba_num_heads
//   head_dim     = mamba_head_dim
//   state_size   = mamba_state_size            (SSM state dim per channel)
//   n_groups     = mamba_n_groups              (B/C are shared within a group)
//   conv_kernel  = mamba_conv_kernel (K)
//   intermediate = num_heads * head_dim
//   conv_dim     = intermediate + 2*n_groups*state_size
//   d_in_proj    = intermediate + conv_dim + num_heads
//                = 2*intermediate + 2*n_groups*state_size + num_heads
//
// IN-PROJECTION SPLIT (the d_in_proj breakdown, per token row):
//   [   z (gate)   |   x | B | C   (conv_in)   |   dt   ]
//      intermediate         conv_dim                num_heads
//   where conv_in packs, for the conv channel axis c in [0, conv_dim):
//       x[h,dim] at c = h*head_dim + dim                         (intermediate)
//       B[g,s]   at c = intermediate + g*state_size + s          (n_groups*state)
//       C[g,s]   at c = intermediate + n_groups*state_size
//                       + g*state_size + s                       (n_groups*state)
//   The depthwise conv runs over ALL conv_dim channels (x, B and C), producing
//   conv_out with the identical packing, which the scan consumes directly.
//
// =====================================================================
// WEIGHT LAYOUTS (all device bf16, row-major).
//   in_proj_w    [d_in_proj, H]          in-proj  (HF: hidden@W^T)
//   conv_w       [conv_dim, conv_kernel] depthwise per-channel kernel ([C,K];
//                                        HF conv1d.weight [conv_dim,1,K] squeezed)
//   conv_bias    [conv_dim]              per-channel conv bias (nullable)
//   A_log        [num_heads]             SSD log-decay; A = -exp(A_log)
//   D            [num_heads]             SSM skip term
//   dt_bias      [num_heads]             dt softplus bias
//   norm_weight  [intermediate]          gated RMSNorm gain
//   out_proj_w   [H, intermediate]       out-proj (HF: core@W^T)

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace pie_cuda_device::forward {

// Per-layer Mamba-2 mixer weight pointers (device bf16). See header for layouts.
struct NemotronMambaWeights {
    const void* in_proj_w;    // [d_in_proj, H]
    const void* conv_w;       // [conv_dim, conv_kernel]
    const void* conv_bias;    // [conv_dim]  (nullable)
    const void* A_log;        // [num_heads]
    const void* D;            // [num_heads]
    const void* dt_bias;      // [num_heads]
    const void* norm_weight;  // [intermediate]
    const void* out_proj_w;   // [H, intermediate]
};

// Runs one Nemotron-H Mamba-2 mixer block in place on `hidden`
// [num_tokens, H] bf16 for a single prefill request (state starts zero). All
// work is enqueued on `stream`; this entry synchronizes before returning and
// allocates its own activation scratch (mirrors mla_block_bf16 /
// moe_mlp_block_bf16). Returns the first CUDA error encountered.
//   time_step_min : lower clamp on softplus(dt) (upstream uses 0.f).
cudaError_t nemotron_mamba_block_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    void* hidden, const NemotronMambaWeights& w,
    int num_tokens, int H,
    int num_heads, int head_dim, int state_size, int n_groups, int conv_kernel,
    float rms_eps, float time_step_min);

}  // namespace pie_cuda_device::forward
