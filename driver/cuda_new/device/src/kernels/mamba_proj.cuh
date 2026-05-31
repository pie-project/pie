#pragma once

// Mamba-2 in-projection split, parameter prep, and gated RMSNorm — the
// per-token / per-head bookkeeping that wraps the selective scan
// (ssm_scan.cuh) and the causal conv (causal_conv1d.cuh) into a full
// Nemotron-H / Mamba-2 mixer block.
//
// Lifted VERBATIM (de-branded) from driver/cuda/src/kernels/nemotron_h.cu —
// specifically the device kernels and launcher bodies that the scan lift left
// behind:
//   * mamba_split_kernel / mamba_split_conv_dt_kernel
//                                  -> mamba_split_bf16
//   * prepare_mamba_params_kernel  -> prepare_mamba_params_bf16
//   * prepare_mamba_dt_da_kernel   -> prepare_mamba_dt_da_bf16
//   * zamba_rmsnorm_gated_kernel   -> rmsnorm_gated_bf16
// The only edits are the namespace (pie_cuda_driver::kernels ->
// pie_cuda_device::kernels) and the dropped launch_/nemotron_/zamba_ prefixes.
// The recurrence math, layouts and index arithmetic are unchanged.
//
// =====================================================================
// IN-PROJECTION SPLIT (mamba_split_bf16)
//
// The in_proj GEMM emits `projected [N, projection_dim]` where
//     projection_dim = intermediate + conv_dim + num_heads
// and each row is laid out as the concatenation
//     [  z (gate)   |   x | B | C  (conv_in) |   dt  ]
//        intermediate     conv_dim              num_heads
// with conv_dim = intermediate + 2*n_groups*state_size and
// intermediate = num_heads*head_dim.
//
// The split has two forms (chosen by whether `gate` is null):
//   * gate != nullptr  — full split into 3 buffers (z, conv_in, dt). z is
//                        copied to a contiguous [N, intermediate] gate buffer.
//   * gate == nullptr  — conv+dt only: only conv_in [N, conv_dim] and
//                        dt [N, num_heads] are extracted; the gate slice is
//                        left in `projected` to be read in place by the gated
//                        RMSNorm (gate_stride = projection_dim). This is the
//                        path Nemotron-H uses.
//
// =====================================================================
// PARAM PREP (prepare_mamba_params_bf16)
//   A[h]       = -exp(A_log[h])        (negative; the SSD decay coefficient)
//   D_f32[h]   = (fp32) D[h]
//   dt_bias_f32[h] = (fp32) dt_bias[h]
// A_log / D / dt_bias arrive as bf16 [num_heads]; outputs are fp32 [num_heads].
//
// OPTIONAL dt/dA PRECOMPUTE (prepare_mamba_dt_da_bf16)
//   dt_out[t,h] = max( softplus(dt_raw[t,h] + dt_bias[h]), time_step_min )
//   dA_out[t,h] = exp(dt_out[t,h] * A[h])
// fed to ssm_selective_scan_bf16 via dt_precomputed / dA_precomputed (the scan
// computes these inline if you pass nullptr instead).
//
// =====================================================================
// GATED RMSNORM (rmsnorm_gated_bf16)  — the Zamba/Mamba-2 gated norm
//   For each (row, group) of `group_size` channels:
//       v[i]    = x[i] * silu(gate[i])
//       inv_rms = rsqrt( mean_i(v[i]^2) + eps )
//       y[i]    = v[i] * inv_rms * weight[i]
//   x / y    : [N, hidden] bf16 (hidden = intermediate = num_groups*group_size)
//   gate     : read at gate[row*gate_stride + group*group_size + i]; pass
//              gate_stride = hidden for a contiguous gate buffer, or
//              gate_stride = projection_dim to read z in place from the
//              in_proj output (gate slice is its first `intermediate` cols).
//   weight   : [hidden] bf16 norm gain.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

// In-projection split. If `gate == nullptr`, only conv_in/dt are written
// (the gate slice stays in `projected`); otherwise z is copied to `gate`.
void mamba_split_bf16(
    const void* projected,   // [N, projection_dim] bf16
    void*       gate,        // [N, intermediate] bf16, nullable
    void*       conv_in,     // [N, conv_dim] bf16
    void*       dt,          // [N, num_heads] bf16
    int N,
    int projection_dim,
    int intermediate,
    int conv_dim,
    int num_heads,
    cudaStream_t stream);

// A = -exp(A_log); D, dt_bias cast bf16 -> fp32. All [num_heads].
void prepare_mamba_params_bf16(
    const void* A_log,        // [num_heads] bf16
    const void* D,            // [num_heads] bf16
    const void* dt_bias,      // [num_heads] bf16
    float*      A,            // [num_heads] fp32 out
    float*      D_f32,        // [num_heads] fp32 out
    float*      dt_bias_f32,  // [num_heads] fp32 out
    int num_heads,
    cudaStream_t stream);

// Optional dt/dA precompute (fed to the scan's dt_precomputed/dA_precomputed).
void prepare_mamba_dt_da_bf16(
    const void*  dt,            // [N, num_heads] bf16 raw dt
    const float* A,             // [num_heads] fp32
    const float* dt_bias,       // [num_heads] fp32
    float*       dt_out,        // [N, num_heads] fp32 out
    float*       dA_out,        // [N, num_heads] fp32 out
    int N,
    int num_heads,
    float time_step_min,
    cudaStream_t stream);

// Gated RMSNorm: y = (x * silu(gate)) * rsqrt(mean(.^2)+eps) * weight,
// per group of `group_size` channels.
void rmsnorm_gated_bf16(
    const void* x,        // [N, hidden] bf16
    const void* gate,     // bf16, read with stride gate_stride
    const void* weight,   // [hidden] bf16
    void*       y,        // [N, hidden] bf16 (may alias x)
    int N,
    int hidden,
    int gate_stride,
    int group_size,
    float eps,
    cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
