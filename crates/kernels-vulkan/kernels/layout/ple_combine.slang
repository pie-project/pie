// Gemma PLE combine: `(proj + token) * inv_sqrt2`.
//
// The count field remains in `PleCombineParams` for ABI parity with Metal, but
// it is not a bounds check: the driver already expressed the true extent in the
// grid, including multi-row prefill tensors.

#include "common/bf16.slang"
#include "norm/rms_params.slang"

PIE_BUFFER_RO(0, proj);
PIE_BUFFER_RO(1, token);
PIE_BUFFER_RW(2, out_);
PIE_PARAMS(3, PleCombineParams, p);

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    out_[gid.x] = PIE_STORE((PIE_LOAD(proj[gid.x]) + PIE_LOAD(token[gid.x])) * p.inv_sqrt2);
}

// pie:instantiate ple_combine_bfloat16
