// Add a per-column bias to every row, in place: `out[r][c] += bias[c]`.
//
// The Qwen-2 family's attention biases, and the reason this file exists at
// all: the shared Metal text omitted them because no Metal kernel adds a
// bias, and a Qwen2 computed without its q/k/v biases is not a crash or a
// NaN -- it is fluent, wrong text. `crates/model`'s text states the op when
// the deployment says the backend can launch it, and this is that claim.
//
// # Why the grid is two-dimensional and `residual_add`'s is not
//
// A residual is elementwise against a buffer of the same extent, so a flat
// index over `rows * width` reads both operands correctly. A bias is
// BROADCAST: one vector of `width`, re-read by every row. The column has to
// be recoverable from the invocation, and `LaunchRule::RouteRows` gives it
// directly -- it dispatches `[width, rows, 1]`, so `x` is the column and `y`
// is the row.
//
// # Why two buffers and not three
//
// The op is IN PLACE, and not by accident: `kernels-cuda`'s
// `norm::add_bias_bf16` declares `in_place = &[(0, 0)]` and sources its only
// mutable buffer from `Out(0)`, so the trace hands both backends the SAME
// allocation for the value and the result. A three-buffer shader here would
// bind that one allocation twice and read a row it had already written --
// correct only by luck of the launch order. Mirroring the CUDA row keeps one
// statement meaning one thing on both backends.
//
// A flat grid with a `% width` would work and is worse: the modulus is a
// division per element, and it puts the width in the shader where the launch
// rule already carries it.
//
// # The bounds test is not decoration
//
// The grid is rounded up to whole workgroups, so the last group runs past
// the row. The buffer is WRITTEN, so an unguarded lane does not merely read
// a zero back -- it biases a column of the NEXT row a second time, and on
// the last row it writes off the end of the allocation. Neither is reported:
// this card returns zero for an out-of-bounds storage read and faults on
// nothing, and the validation layer sees a dispatch it approved. `width` is the column bound; the row bound is exact by
// construction (`y` is one row per invocation, and the rule dispatches
// exactly `rows` of them).

#include "common/bf16.slang"

PIE_BUFFER_RW(0, out_);
PIE_BUFFER_RO(1, bias);

struct Push { int width; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    if (c >= uint(pc.width)) {
        return;
    }
    uint i = gid.y * uint(pc.width) + c;
    out_[i] = PIE_STORE(PIE_LOAD(out_[i]) + PIE_LOAD(bias[c]));
}

// pie:instantiate add_bias_bfloat16
