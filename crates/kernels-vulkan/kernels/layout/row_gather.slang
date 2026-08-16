// Compact the rows a fire will actually sample.
//
// Ported from Metal's row gather with the same packed params buffer. The extra
// `count` operand in the Rust row is packed into that buffer, so there is no
// push constant here.
//
// The first kernel of this tree written in Slang, and so the one that had to
// prove the SPIR-V is indistinguishable from what `glslc` produced: same
// entrypoint name, same four bindings in the same order, same workgroup size,
// no push block. `tests/gpu.rs` runs it on the card and compares against the
// same expectation it always did.

#include "common/bf16.slang"
#include "layout/row_gather_params.slang"

PIE_BUFFER_RO(0, input_);
PIE_BUFFER_RW(1, out_);
[[vk::binding(2)]] StructuredBuffer<uint> rows;
PIE_PARAMS(3, RowGatherParams, p);

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    uint i = gid.y;
    if (c >= p.width || i >= p.count) return;
    out_[i * p.width + c] = input_[rows[i] * p.width + c];
}

// pie:instantiate row_gather_bfloat16
