#include <metal_stdlib>
using namespace metal;

// gemv_demo — stand-in M=1 GEMV so the harness builds + runs before delta's real
// kernel ports land. Mirrors the affine-qmv launch shape (grid = N output cols).
// bfloat is native on Metal 4 (delta's probe) — demo uses float for portability.
//
// Binding order matches decode_abi.hpp bind::Qmv { W=0, Scales=1, Biases=2, X=3,
// Out=4, K=5, N=6 } loosely — here a dense float GEMV (no quant) for the rig smoke.
kernel void gemv_demo(
    device const float* w   [[buffer(0)]],   // [N, K] row-major weights
    device const float* x   [[buffer(3)]],   // [K] input row (M=1)
    device       float* out [[buffer(4)]],   // [N] output
    constant uint&      K   [[buffer(5)]],
    constant uint&      N   [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    float acc = 0.0f;
    device const float* wrow = w + (uint64_t)gid * K;
    for (uint k = 0; k < K; ++k) acc += wrow[k] * x[k];
    out[gid] = acc;
}
