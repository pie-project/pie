// Gemma PLE combine: `(proj + token) * inv_sqrt2`.
//
// Two streams averaged in the root-mean-square sense, over the whole
// `[n_layers, ple_dim]` block at once. The scale is the JOIN's and not a
// deployment's, which is why it is a number the statement carries rather than
// an axis.
//
// THE SCALE IS A MARK, NOT A STRUCT. Both scalars used to ride
// `PleCombineParams { inv_sqrt2, n }` on `@group(0) @binding(3)` -- Metal's
// buffer-3 layout, ported here through Vulkan -- so this row stated four
// buffers, no scalars, and no `@group(1)` at all. The second word was already
// dead: `n` was stated as ONE ROW's element count while the elementwise launch
// dispatches `width * rows`, so it never was the bound it was named for, and
// the guard that actually holds is the output's own length -- the only number
// that is true whatever the grid did. The field stayed only so the struct kept
// its size and layout.
//
// The routine states `inv_sqrt2: Const<f32>` now and
// `driver-wgpu::lowering::routine::bind` packs it into the `@group(1)` uniform
// block, which is word 0 of the same `Lowered::params` run reached by its index
// instead of by a struct field. With no struct left to keep the size of, `n`
// goes with it.

//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> proj: array<u32>;
@group(0) @binding(1) var<storage, read_write> token: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;

struct Params { inv_sqrt2: f32 }
@group(1) @binding(0) var<uniform> params: Params;

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
