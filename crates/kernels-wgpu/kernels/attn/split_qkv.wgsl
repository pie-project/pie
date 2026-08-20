// The fused QKV projection, split back into three tensors.
//
// THE TWO WIDTHS ARE MARKS, NOT A STRUCT. They used to arrive as
// `SplitQkvParams { q_width, kv_width }` on a fifth `@group(0)` storage
// binding -- MLX's buffer-4 layout, carried here through Metal and Vulkan --
// so the row stated five buffers and no scalars and this file declared no
// `@group(1)` at all. The routine states `q_width: Const<u32>` and
// `kv_width: Const<u32>` now and `driver-wgpu::lowering::routine::bind` packs
// the pair into the `@group(1)` block, which is words 0 and 1 of the same
// `Lowered::params` run the struct was staged from, reached by index instead
// of by field. THE ORDER IS THE STRUCT'S ORDER, because it is the statement's:
// `q_width` is word 0 and `kv_width` is word 1, and swapping the two marks
// would cut both boundaries in the wrong place rather than refuse.

//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> packed: array<u32>;
@group(0) @binding(1) var<storage, read_write> q: array<u32>;
@group(0) @binding(2) var<storage, read_write> k: array<u32>;
@group(0) @binding(3) var<storage, read_write> v: array<u32>;

struct Params { q_width: u32, kv_width: u32 }
@group(1) @binding(0) var<uniform> params: Params;

// The bf16 half-index unpack. `pie_load_bf16(&packed, i)` is the shared answer
// and cannot be CALLED: its `ptr<storage, array<u32>, read>` parameter is
// WGSL's `unrestricted_pointer_parameters`, which naga does not implement, so a
// module that calls it parses and then fails `create_shader_module`. The
// CONVERSION keeps one definition in `common/bf16.inc.wgsl`; only the address
// arithmetic is restated here.
fn packed_at(i: u32) -> f32 {
    let word = packed[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let q_width = params.q_width;
    let kv_width = params.kv_width;
    let packed_width = q_width + 2u * kv_width;

    // One invocation per PAIR of channels, not per channel. A bf16 destination
    // word holds two values and WGSL has no sub-word atomic, so an invocation
    // that wrote one half would read-modify-write a word its neighbour is
    // writing at the same moment and one of the two stores would be lost. The
    // pair is the unit of ownership; the grid may still be a full row wide,
    // and the upper half of it exits here, which is the harmless direction.
    let c = gid.x * 2u;
    if (c >= packed_width) { return; }

    // Both halves come from one packed row, and both land in one destination
    // row, because every width in this kernel is `heads * head_dim` and every
    // point of the head-dim axis is even. A pair therefore never straddles the
    // q|k|v boundary and never straddles a row. An odd width would break that
    // and there is no way to write it safely from inside a shader — it would
    // need one invocation per destination WORD and a different grid.
    let src = row * packed_width + c;
    let lo = packed_at(src);
    let hi = packed_at(src + 1u);

    if (c < q_width) {
        let at = (row * q_width + c) >> 1u;
        if (at < arrayLength(&q)) { q[at] = pie_pack_bf16(lo, hi); }
    } else if (c < q_width + kv_width) {
        let at = (row * kv_width + (c - q_width)) >> 1u;
        if (at < arrayLength(&k)) { k[at] = pie_pack_bf16(lo, hi); }
    } else {
        let at = (row * kv_width + (c - q_width - kv_width)) >> 1u;
        if (at < arrayLength(&v)) { v[at] = pie_pack_bf16(lo, hi); }
    }
}

// pie:instantiate split_qkv_bf16
