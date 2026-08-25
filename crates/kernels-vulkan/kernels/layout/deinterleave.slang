// The two row cuts: `layout.split_rows` and `layout.select`.
//
// Both are pure COPIES -- no arithmetic, no conversion, not even a widen and a
// narrow. The element is moved as its storage type, so a bf16 that goes in
// comes out as the same sixteen bits. `kernels-cuda`'s
// `layout/deinterleave.cuh` is the semantics both follow, and `kernels-metal`'s
// `layout/deinterleave.metal` is where the two entrypoints were first filed
// together; this is the third telling, and the arithmetic is transcribed
// rather than re-derived.
//
// ## Why this plane's version is SHORTER than the wgpu one
//
// `kernels-wgpu`'s twin moves WHOLE WORDS, refuses an odd row pitch by name,
// and carries an `arrayLength` guard on every access, because WGSL has no
// sub-word store: a bf16 pair shares a `u32` and two adjacent output columns
// belong to two adjacent invocations. None of that applies here. This plane's
// bf16 is `uint16_t` in a `StructuredBuffer<PIE_ACT>` -- ONE element per slot,
// addressable on its own -- so an odd width is not a special case and there is
// no pairing to scramble. The Vulkan port is closer to metal's than to the
// wgsl one it was translated beside.
//
// ## The grids are DIFFERENT rectangles, and that is the whole ABI
//
// `split_rows` walks the SOURCE row, `select` walks the RESULT row.
//
// That is not a stylistic difference; it is what makes each one exactly cover
// its work. A split's source row is `left_dim + right_dim` wide and every
// element of it lands in exactly one of the two outputs, so one invocation per
// source element reads once and writes once, and a grid over either output
// alone would leave the other half unwritten. A select's source row is
// `layers * width` wide and only `width` of it is wanted, so a grid over the
// source would launch `layers` times the invocations and every one of them but
// a `1/layers` slice would have nothing to do.
//
// So `kernels_vulkan::layout` states `elementwise_rows(x.width, x.rows)` for
// the first and `elementwise_rows(y.width, y.rows)` for the second, and the
// guards below are `total` and `width` respectively for the same reason.
//
// ## `stride` and `width` are both parameters and neither is the other
//
// Carried over verbatim from the CUDA header, because it is the thing a reader
// tries to simplify: the packed relay's row is `layers * width` and the
// result's is `width`, and a kernel that recovered one from the other would
// have to be told `layers` -- which is the one number gemma's statement does
// not carry. Its param says WHICH layer, not how many.

#include "common/bf16.slang"

#if defined(PIE_SELECT)
PIE_BUFFER_RO(0, table);
PIE_BUFFER_RW(1, out_);
struct Push { int stride; int offset; int width; };
#else
PIE_BUFFER_RO(0, src);
PIE_BUFFER_RW(1, left);
PIE_BUFFER_RW(2, right);
struct Push { int left_dim; int right_dim; };
#endif
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    uint row = gid.y;
#if defined(PIE_SELECT)
    uint width = uint(pc.width);
    if (c >= width) {
        return;
    }
    out_[row * width + c] = table[row * uint(pc.stride) + uint(pc.offset) + c];
#else
    uint left_dim = uint(pc.left_dim);
    uint total = left_dim + uint(pc.right_dim);
    if (c >= total) {
        return;
    }
    PIE_ACT value = src[row * total + c];
    if (c < left_dim) {
        left[row * left_dim + c] = value;
    } else {
        right[row * uint(pc.right_dim) + (c - left_dim)] = value;
    }
#endif
}

// pie:instantiate split_rows_bfloat16
// pie:instantiate select_slice_bfloat16 PIE_SELECT=1
