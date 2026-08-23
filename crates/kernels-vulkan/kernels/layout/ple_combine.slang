// Gemma PLE combine: `(proj + token) * inv_sqrt2`.
//
// THE SCALE IS A PUSH CONSTANT, NOT A STORAGE STRUCT. `PleCombineParams
// { inv_sqrt2, n }` was Metal's buffer-3 layout carried over whole, second word
// and all, and it lived in norm/rms_params.slang beside the RMS family's
// blocks so the two trees could be deleted together. They were: that header is
// gone, its last reader having moved its scalars into a push range. The count field was
// already dead here: it was stated as ONE ROW's element count while the
// elementwise launch dispatches the whole `[n_layers, ple_dim]` rectangle, so
// the driver had already expressed the true extent in the grid -- multi-row
// prefill tensors included -- and the field survived only to hold the struct's
// size for ABI parity with Metal.
//
// With the scale stated as `inv_sqrt2: Const<f32>` there is no struct to keep
// the size of, so the count goes with it. Four live bytes were costing this
// plane a descriptor, a staging slot and the sentinel `Holds::params_block`
// mints for a buffer with no address; four bytes is a push range on every
// device this targets, and the number in it is word 0 of the same statement run
// the struct's first field read.

#include "common/bf16.slang"

PIE_BUFFER_RO(0, proj);
PIE_BUFFER_RO(1, token);
PIE_BUFFER_RW(2, out_);

struct Push { float inv_sqrt2; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    out_[gid.x] = PIE_STORE((PIE_LOAD(proj[gid.x]) + PIE_LOAD(token[gid.x])) * pc.inv_sqrt2);
}

// pie:instantiate ple_combine_bfloat16
