// The two column cuts: one packed row out to two, and one layer's slice of a
// laid-out relay.
//
// `kernels-cuda/kernels/layout/deinterleave.cuh` holds the same pair (plus the
// five its own callers need) and this is the WGSL reading of those two.
//
// # `select_slice` and not `select`
//
// `select` is a WGSL builtin, and every body in this tree calls it. An entry
// point named `select` would be a redeclaration the module rejects. The POINT
// is still `layout.select`; only the entrypoint is spelled apart from it, which
// is also what `kernels-metal/kernels/layout/deinterleave.metal` does and for
// the same collision.
//
// # Why a copy and not a binding offset
//
// `select` is a base and an offset -- the whole arithmetic -- so a plane could
// one day answer it at BINDING with a view and never launch at all, which the
// declaration's own doc says. Until the binder can say that, a copy is what
// says it: a slice this kernel wrote is a rectangle the arena owns, with no
// aliasing rule for a later pass to discover. `split_rows` is the same shape of
// statement with both halves kept.
//
// # Both bodies move WORDS, and the host is what makes that safe
//
// A bf16 tensor is an `array<u32>` with two values to a word and WGSL has no
// sub-word atomic, so an invocation that owned one element would
// read-modify-write a word its neighbour is writing at the same moment. The
// unit of ownership is therefore the PAIR, and a pair is whole only when every
// row of every operand starts on a word boundary:
//
//   * `split_rows` needs `left_dim` EVEN (so the cut falls between words, not
//     inside one) and `left_dim + right_dim` even (so the source's rows do).
//     Both halves' rows are then even-pitched too.
//   * `select_slice` needs `width` EVEN and `stride` even, which together make
//     `offset = layer * width` even and every read and write word-aligned.
//
// `kernels_wgpu::layout` checks exactly those and refuses by name, so an odd
// pitch is a MEASURED refusal rather than a scrambled tensor. Nothing is
// widened to f32 anywhere below: a cut moves bits, and a round trip through
// `pie_f32_to_bf16` would canonicalise a NaN for nothing.
//
// The grid is `elementwise_rows` of the row each body WALKS -- the source's for
// `split_rows`, since every element is read once and lands in exactly one of
// the two results, and the result's for `select_slice`, since the source row is
// the wider of the two. One lane per element, of which half do the work.

@group(0) @binding(0) var<storage, read_write> src: array<u32>;

//#if defined(PIE_SELECT)
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;

struct Params { stride: i32, offset: i32, width: i32 }
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = u32(params.width);
    let c = gid.x * 2u;
    if c + 1u >= width {
        return;
    }
    let row = gid.y;
    let at = (row * width + c) >> 1u;
    if at >= arrayLength(&out_) {
        return;
    }
    out_[at] = src[(row * u32(params.stride) + u32(params.offset) + c) >> 1u];
}

//#else
@group(0) @binding(1) var<storage, read_write> left: array<u32>;
@group(0) @binding(2) var<storage, read_write> right: array<u32>;

struct Params { left_dim: i32, right_dim: i32 }
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let left_dim = u32(params.left_dim);
    let right_dim = u32(params.right_dim);
    let total = left_dim + right_dim;
    let c = gid.x * 2u;
    if c + 1u >= total {
        return;
    }
    let row = gid.y;
    // The source word is whole for the same reason both destination words are:
    // `total` is even, so every source row starts on a word boundary, and `c`
    // is even. The pair therefore never straddles the cut either -- `left_dim`
    // is even, so `c < left_dim` and `c + 1 < left_dim` agree.
    let word = src[(row * total + c) >> 1u];
    if c < left_dim {
        let at = (row * left_dim + c) >> 1u;
        if at < arrayLength(&left) {
            left[at] = word;
        }
    } else {
        let at = (row * right_dim + (c - left_dim)) >> 1u;
        if at < arrayLength(&right) {
            right[at] = word;
        }
    }
}

//#endif

// pie:instantiate select_slice_bfloat16 PIE_SELECT=1
// pie:instantiate split_rows_bfloat16
