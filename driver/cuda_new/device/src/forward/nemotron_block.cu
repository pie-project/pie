#include "nemotron_block.cuh"

#include "../kernels/mamba_proj.cuh"      // mamba_split / param-prep / gated rmsnorm
#include "../kernels/ssm_scan.cuh"        // ssm_selective_scan_bf16
#include "../kernels/causal_conv1d.cuh"   // causal_conv1d_prefill_bf16
#include "../ops/gemm.cuh"                // gemm_act_x_wt_bf16

#include <cuda_bf16.h>

// Composition mirrors driver/cuda/src/model/nemotron_h_forward.cpp::mamba_layer
// (single fresh-prefill request, no TP). All math lives in the lifted
// primitives; this file only sequences them and owns scratch.

namespace pie_cuda_device::forward {

cudaError_t nemotron_mamba_block_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    void* hidden, const NemotronMambaWeights& w,
    int num_tokens, int H,
    int num_heads, int head_dim, int state_size, int n_groups, int conv_kernel,
    float rms_eps, float time_step_min) {
    const int T = num_tokens;
    const int intermediate = num_heads * head_dim;
    const int conv_dim = intermediate + 2 * n_groups * state_size;
    const int d_in_proj = intermediate + conv_dim + num_heads;
    const int K = conv_kernel;
    constexpr size_t es = sizeof(std::uint16_t);   // bf16 element
    constexpr size_t fs = sizeof(float);

    // Single prefill request covering all T tokens; SSM state starts zero.
    const std::uint32_t qo_indptr_h[2] = {0u, static_cast<std::uint32_t>(T)};
    const std::int32_t slot_ids_h[1] = {0};

    // Scratch buffers (see enum). projected holds the in_proj output AND the
    // gate/z slice that gated-rmsnorm reads in place (gate_stride = d_in_proj).
    enum {
        PROJECTED,   // [T, d_in_proj] bf16  (z | conv_in | dt)
        CONV_IN,     // [T, conv_dim]  bf16  (x|B|C extracted by split)
        CONV_OUT,    // [T, conv_dim]  bf16  (post depthwise causal conv)
        DT_RAW,      // [T, num_heads] bf16
        CORE,        // [T, intermediate] bf16 (scan_y, then gated norm in place)
        A_F32,       // [num_heads] fp32
        D_F32,       // [num_heads] fp32
        DTB_F32,     // [num_heads] fp32
        SSM_STATE,   // [1, num_heads, head_dim, state_size] bf16
        QO_INDPTR,   // [2] uint32
        SLOT_IDS,    // [1] int32
        NBUF
    };
    const size_t sizes[NBUF] = {
        (size_t)T * d_in_proj * es,
        (size_t)T * conv_dim * es,
        (size_t)T * conv_dim * es,
        (size_t)T * num_heads * es,
        (size_t)T * intermediate * es,
        (size_t)num_heads * fs,
        (size_t)num_heads * fs,
        (size_t)num_heads * fs,
        (size_t)num_heads * head_dim * state_size * es,
        2 * sizeof(std::uint32_t),
        1 * sizeof(std::int32_t),
    };
    void* b[NBUF] = {};
    for (int i = 0; i < NBUF; ++i) {
        if (cudaError_t e = cudaMalloc(&b[i], sizes[i]); e != cudaSuccess) {
            for (int j = 0; j < i; ++j) cudaFree(b[j]);
            return e;
        }
    }

    // SSM recurrent state starts zero for a fresh prefill.
    if (cudaError_t e = cudaMemsetAsync(b[SSM_STATE], 0, sizes[SSM_STATE], stream);
        e != cudaSuccess) {
        for (int i = 0; i < NBUF; ++i) cudaFree(b[i]);
        return e;
    }
    cudaMemcpyAsync(b[QO_INDPTR], qo_indptr_h, sizes[QO_INDPTR],
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(b[SLOT_IDS], slot_ids_h, sizes[SLOT_IDS],
                    cudaMemcpyHostToDevice, stream);

    // 1. in_proj: projected[T, d_in_proj] = hidden @ in_proj_w^T
    ops::gemm_act_x_wt_bf16(cublas, hidden, w.in_proj_w, b[PROJECTED],
                            T, d_in_proj, H, /*beta=*/0.f);

    // 2. split: conv_in (x|B|C) + dt; gate==nullptr leaves z in `projected`.
    kernels::mamba_split_bf16(
        b[PROJECTED], /*gate=*/nullptr, b[CONV_IN], b[DT_RAW],
        T, d_in_proj, intermediate, conv_dim, num_heads, stream);

    // 3. depthwise causal conv over all conv_dim channels (state_out=nullptr =>
    //    fresh prompt zero-padding, no persisted window).
    kernels::causal_conv1d_prefill_bf16(
        b[CONV_IN], w.conv_w, w.conv_bias, b[CONV_OUT], /*state_out=*/nullptr,
        T, conv_dim, K, stream);

    // 4. param prep: A=-exp(A_log), D, dt_bias -> fp32.
    kernels::prepare_mamba_params_bf16(
        w.A_log, w.D, w.dt_bias,
        static_cast<float*>(b[A_F32]), static_cast<float*>(b[D_F32]),
        static_cast<float*>(b[DTB_F32]), num_heads, stream);

    // 5. selective scan: core[T, intermediate] (dt/dA computed inline).
    kernels::ssm_selective_scan_bf16(
        b[CONV_OUT], b[DT_RAW],
        static_cast<const float*>(b[A_F32]),
        static_cast<const float*>(b[D_F32]),
        static_cast<const float*>(b[DTB_F32]),
        /*dt_precomputed=*/nullptr, /*dA_precomputed=*/nullptr,
        b[SSM_STATE],
        static_cast<const std::int32_t*>(b[SLOT_IDS]),
        static_cast<const std::uint32_t*>(b[QO_INDPTR]),
        b[CORE],
        /*R=*/1, num_heads, head_dim, state_size, n_groups,
        conv_dim, intermediate, time_step_min, stream);

    // 6. gated RMSNorm: core = gated_rmsnorm(core, z=projected, norm_weight).
    //    z lives in projected's first `intermediate` cols => gate_stride=d_in_proj.
    kernels::rmsnorm_gated_bf16(
        b[CORE], b[PROJECTED], w.norm_weight, b[CORE],
        T, intermediate, /*gate_stride=*/d_in_proj,
        /*group_size=*/intermediate / n_groups, rms_eps, stream);

    // 7. out_proj with fused residual add: hidden += core @ out_proj_w^T.
    ops::gemm_act_x_wt_bf16(cublas, b[CORE], w.out_proj_w, hidden,
                            T, H, intermediate, /*beta=*/1.f);

    cudaError_t e = cudaStreamSynchronize(stream);
    for (int i = 0; i < NBUF; ++i) cudaFree(b[i]);
    return e;
}

}  // namespace pie_cuda_device::forward
