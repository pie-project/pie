// Residual add, flat and row-strided.
//
// `out` may alias `x`, and every element is read before its one write. The
// strided variant takes its row pitch from the uniform block.

//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> residual: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;

//#if defined(PIE_STRIDED)
struct Params { row_pitch: i32 }
@group(1) @binding(0) var<uniform> params: Params;
//#endif

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
//#if defined(PIE_STRIDED)
    // The pitch is in ELEMENTS and the word index is half of it, so the pair
    // this invocation owns is `2*i` and `2*i + 1` of the logical row.
    let i = gid.y * u32(params.row_pitch) / 2u + gid.x;
//#else
    let i = gid.x;
//#endif
    // The tail guard is the buffer's own length and not a stated count:
    // `dispatch_workgroups` rounds up, so the last group runs past the data.
    if (i >= arrayLength(&out_)) { return; }

    let a = x[i];
    let b = residual[i];
    out_[i] = pie_pack_bf16(
        pie_bf16_to_f32(a & 0xffffu) + pie_bf16_to_f32(b & 0xffffu),
        pie_bf16_to_f32(a >> 16u) + pie_bf16_to_f32(b >> 16u),
    );
}

// pie:instantiate residual_add_bfloat16
// pie:instantiate residual_add_strided_bfloat16 PIE_STRIDED=1
