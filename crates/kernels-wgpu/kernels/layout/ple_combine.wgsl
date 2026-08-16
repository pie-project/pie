// Gemma PLE combine: `(proj + token) * inv_sqrt2`.
//
// Two streams averaged in the root-mean-square sense, over the whole
// `[n_layers, ple_dim]` block at once. The scale is the JOIN's and not a
// deployment's, which is why it arrives in the params struct rather than as an
// axis.
//
// Both scalars ride in `PleCombineParams` -- binding 3 -- so this row states
// four buffers and no scalars, and the shader declares no `@group(1)` at all.
// The `n` field is kept for ABI parity with Metal and is NOT a bounds check:
// the driver already expressed the extent in the grid, multi-row prefill
// tensors included. The guard is the output's own length, which is the only
// number that is true whatever the grid did.

//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> proj: array<u32>;
@group(0) @binding(1) var<storage, read_write> token: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;
struct PleCombineParams { inv_sqrt2: f32, n: u32 }
@group(0) @binding(3) var<storage, read_write> params: PleCombineParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // A WORD, not an element: two bf16 to a `u32`, so this invocation owns
    // both halves and the store needs no read-modify-write.
    let i = gid.x;
    if (i >= arrayLength(&out_)) { return; }

    let a = proj[i];
    let b = token[i];
    out_[i] = pie_pack_bf16(
        (pie_bf16_to_f32(a & 0xffffu) + pie_bf16_to_f32(b & 0xffffu)) * params.inv_sqrt2,
        (pie_bf16_to_f32(a >> 16u) + pie_bf16_to_f32(b >> 16u)) * params.inv_sqrt2,
    );
}

// pie:instantiate ple_combine_bfloat16
