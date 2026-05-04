// RMS Norm Metal Kernels for Apple Silicon
//
// Two kernels:
//   1. rms_norm_bf16 / rms_norm_f16  — standalone RMS norm with weight multiply
//   2. residual_rms_norm_bf16 / residual_rms_norm_f16  — fused residual-add + RMS norm
//
// For decode (n=1, H=2880), data fits in L1 (~5.7 KB).
// Dispatch overhead dominates, so a direct Metal kernel to pre-allocated output
// beats MPSGraph (which has graph lookup + encoding + output allocation overhead).
//
// Grid: (256, 1, 1), Group: (256, 1, 1) — single threadgroup, 8 simdgroups
//
// Params buffer: [H_float, eps]

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Load bfloat as float (Metal bfloat → float is a simple bit shift)
inline float bf_to_f(bfloat v) { return float(v); }
inline bfloat f_to_bf(float v) { return bfloat(v); }
inline float h_to_f(half v) { return float(v); }
inline half f_to_h(float v) { return half(v); }

// ---------------------------------------------------------------------------
// 1. Standalone RMS norm: output = rms_norm(input) * weight
// ---------------------------------------------------------------------------

template<typename T>
kernel void rms_norm_kernel(
    device const T*     input   [[buffer(0)]],   // [H]
    device const T*     weight  [[buffer(1)]],   // [H]
    device T*           output  [[buffer(2)]],   // [H] pre-allocated
    device const float* params  [[buffer(3)]],   // [H, eps]
    uint tid      [[thread_position_in_grid]],
    uint lane_id  [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]
) {
    const int H = (int)params[0];
    const float eps = params[1];

    // Number of float4 elements (H must be divisible by 4)
    const int H4 = H >> 2;
    // Elements per thread
    const int elems_per_thread = (H4 + 255) / 256;

    // Phase 1: Load input and compute sum of squares
    float sum_sq = 0.0f;
    // Store in registers for reuse in Phase 3 (max 3 float4 for H=2880)
    float4 vals[4];  // supports up to H=4096 (1024 float4 / 256 threads = 4)

    for (int e = 0; e < elems_per_thread; ++e) {
        int idx = tid + e * 256;
        if (idx < H4) {
            float4 v;
            int base = idx * 4;
            v.x = float(input[base]);
            v.y = float(input[base + 1]);
            v.z = float(input[base + 2]);
            v.w = float(input[base + 3]);
            vals[e] = v;
            sum_sq += dot(v, v);
        }
    }

    // Phase 2: Two-stage SIMD reduction
    // Stage 1: reduce within simdgroup
    sum_sq = simd_sum(sum_sq);

    // Stage 2: reduce across simdgroups via shared memory
    threadgroup float shared[8];
    if (lane_id == 0) {
        shared[simd_gid] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total;
    if (simd_gid == 0) {
        // 8 simdgroups → load shared into lanes 0..7
        float v = (lane_id < 8) ? shared[lane_id] : 0.0f;
        total = simd_sum(v);
    }

    // Broadcast rms_scale to all threads via shared memory
    threadgroup float shared_scale[1];
    if (tid == 0) {
        shared_scale[0] = rsqrt(total / float(H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms_scale = shared_scale[0];

    // Phase 3: Apply norm * weight, write output
    for (int e = 0; e < elems_per_thread; ++e) {
        int idx = tid + e * 256;
        if (idx < H4) {
            float4 v = vals[e];
            int base = idx * 4;
            float4 w;
            w.x = float(weight[base]);
            w.y = float(weight[base + 1]);
            w.z = float(weight[base + 2]);
            w.w = float(weight[base + 3]);
            float4 r = v * rms_scale * w;
            output[base]     = T(r.x);
            output[base + 1] = T(r.y);
            output[base + 2] = T(r.z);
            output[base + 3] = T(r.w);
        }
    }
}

// Explicit instantiations
template [[host_name("rms_norm_bf16")]]
kernel void rms_norm_kernel<bfloat>(
    device const bfloat*, device const bfloat*, device bfloat*,
    device const float*, uint, uint, uint);

template [[host_name("rms_norm_f16")]]
kernel void rms_norm_kernel<half>(
    device const half*, device const half*, device half*,
    device const float*, uint, uint, uint);


// ---------------------------------------------------------------------------
// 2. Fused residual-add + RMS norm:
//    res_out = a + b
//    norm_out = rms_norm(res_out) * weight
// ---------------------------------------------------------------------------

template<typename T>
kernel void residual_rms_norm_kernel(
    device const T*     a        [[buffer(0)]],   // [H] residual
    device const T*     b        [[buffer(1)]],   // [H] addend
    device const T*     weight   [[buffer(2)]],   // [H] norm weight
    device T*           res_out  [[buffer(3)]],   // [H] pre-allocated (a + b)
    device T*           norm_out [[buffer(4)]],   // [H] pre-allocated (normalized)
    device const float* params   [[buffer(5)]],   // [H, eps]
    uint tid      [[thread_position_in_grid]],
    uint lane_id  [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]
) {
    const int H = (int)params[0];
    const float eps = params[1];

    const int H4 = H >> 2;
    const int elems_per_thread = (H4 + 255) / 256;

    // Phase 1: Compute residual, write res_out, accumulate sum of squares
    float sum_sq = 0.0f;
    float4 vals[4];  // register storage for reuse

    for (int e = 0; e < elems_per_thread; ++e) {
        int idx = tid + e * 256;
        if (idx < H4) {
            int base = idx * 4;
            float4 va, vb;
            va.x = float(a[base]);     va.y = float(a[base + 1]);
            va.z = float(a[base + 2]); va.w = float(a[base + 3]);
            vb.x = float(b[base]);     vb.y = float(b[base + 1]);
            vb.z = float(b[base + 2]); vb.w = float(b[base + 3]);
            float4 r = va + vb;
            vals[e] = r;
            // Write residual output
            res_out[base]     = T(r.x);
            res_out[base + 1] = T(r.y);
            res_out[base + 2] = T(r.z);
            res_out[base + 3] = T(r.w);
            sum_sq += dot(r, r);
        }
    }

    // Phase 2: Two-stage SIMD reduction
    sum_sq = simd_sum(sum_sq);

    threadgroup float shared[8];
    if (lane_id == 0) {
        shared[simd_gid] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total;
    if (simd_gid == 0) {
        float v = (lane_id < 8) ? shared[lane_id] : 0.0f;
        total = simd_sum(v);
    }

    threadgroup float shared_scale[1];
    if (tid == 0) {
        shared_scale[0] = rsqrt(total / float(H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms_scale = shared_scale[0];

    // Phase 3: Apply norm * weight from registers (no re-read from memory)
    for (int e = 0; e < elems_per_thread; ++e) {
        int idx = tid + e * 256;
        if (idx < H4) {
            float4 v = vals[e];
            int base = idx * 4;
            float4 w;
            w.x = float(weight[base]);
            w.y = float(weight[base + 1]);
            w.z = float(weight[base + 2]);
            w.w = float(weight[base + 3]);
            float4 r = v * rms_scale * w;
            norm_out[base]     = T(r.x);
            norm_out[base + 1] = T(r.y);
            norm_out[base + 2] = T(r.z);
            norm_out[base + 3] = T(r.w);
        }
    }
}

// Explicit instantiations
template [[host_name("residual_rms_norm_bf16")]]
kernel void residual_rms_norm_kernel<bfloat>(
    device const bfloat*, device const bfloat*, device const bfloat*,
    device bfloat*, device bfloat*, device const float*,
    uint, uint, uint);

template [[host_name("residual_rms_norm_f16")]]
kernel void residual_rms_norm_kernel<half>(
    device const half*, device const half*, device const half*,
    device half*, device half*, device const float*,
    uint, uint, uint);
