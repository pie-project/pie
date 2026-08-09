// The three param blocks the MoE routing kernels read.
//
// `moe/params.glsl` states the same three structs, and the field ORDER is the
// contract rather than a style: the host fills one block for every backend, so
// a field moved here is a field read at the wrong offset there, and every value
// after it shifts by four bytes. Nothing reports that -- the block is bytes,
// and neither wgpu nor a validation layer knows what they were meant to be.
//
// ## Why these are STORAGE blocks and not the `@group(1)` uniform
//
// The launch ABI puts a row's SCALAR operands in one uniform block. These five
// rows state no scalars: `route_sort`, `route_gather`, `combine_sorted` and the
// two `router_topk`s each take a `params: Buf` -- a POINTER to where the
// numbers already are -- so the block rides in `@group(0)` like any other
// buffer and the shader reads it as a struct.
//
// That is not a workaround. The routing params are BUILT by the host plan, not
// carried in the statement, so they are already in device memory when the
// launch is assembled; a uniform would be a round trip for numbers that never
// leave the device.
//
// WGSL's storage layout for a struct of `u32`s is std430's -- each member on
// its own four-byte alignment, in declaration order -- so these three are
// byte-for-byte what the GLSL sibling reads.

// `router_topk` / `router_topk_scaled`.
//
// `logits_pitch` of zero means "tightly packed", i.e. the pitch IS `n_experts`.
// The zero is load-bearing: a router reading a slice of a wider activation has
// a pitch that is not its expert count, and a host with no slice writes 0
// rather than having to state the count twice.
struct RouterParams {
    n_experts: u32,
    experts_per_token: u32,
    softmax_over_all: u32,
    logits_pitch: u32,
}

// `combine_sorted`. `out_pitch` of zero means `width`, for the same reason.
struct ExpertCombineParams {
    width: u32,
    experts_per_token: u32,
    out_pitch: u32,
}

// `route_sort` and `route_gather`.
//
// `n` is the number of (row, slot) PAIRS -- one per expert choice -- while
// `padded` is the length of the permutation, `n` rounded up so every expert's
// span is a whole number of `tile_rows` tiles. The two are different numbers
// and the sort reads both; a body that used one for the other would clear a
// permutation shorter than it fills.
struct MoeRouteParams {
    n: u32,
    n_experts: u32,
    experts_per_token: u32,
    tile_rows: u32,
    padded: u32,
    width: u32,
    x_pitch: u32,
}
