// Per-layer scalar multiply: `out = x * scalar[0]`.
//
// The Metal port's important finding is preserved: the grid is the extent. The
// params struct stays bound for ABI compatibility, but it is not a second bound
// on the dispatch because prefill rows would otherwise be skipped.

#include "common/bf16.slang"
#include "norm/rms_params.slang"

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, scalar);
PIE_BUFFER_RW(2, out_);
PIE_PARAMS(3, LayerScalarParams, p);

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    out_[gid.x] = PIE_STORE(PIE_LOAD(x[gid.x]) * PIE_LOAD(scalar[0]));
}

// pie:instantiate layer_scalar_mul_bfloat16
