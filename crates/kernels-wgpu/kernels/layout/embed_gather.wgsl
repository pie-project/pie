// Quantized tied-embedding gather.
//
// Metal's affine-dequant gather on WebGPU's binding rule: the five tensors are
// `@group(0)` storage buffers in the row's order, and `hidden` -- plus the
// embedding scale the `_scaled` rows carry -- are the fields of the one
// `@group(1) @binding(0)` uniform block. The affine axes are compile-time facts
// (`PIE_GROUP`, `PIE_BITS`) so a `_gs_64_b_4` module cannot be reused for the
// identical-looking `_gs_128_b_8` storage.
//
// ## One invocation owns one WORD of the output, not one element
//
// The Vulkan and Metal bodies write `out[m * hidden + k]` for one `k`, because
// there a bf16 tensor is an array of 16-bit elements. WGSL has no 16-bit
// storage: `out_` is `array<u32>` with two embeddings to a word, so an
// element-per-invocation body would be a read-modify-write of a word its
// neighbour is also writing, and WGSL has no sub-word atomic to make that safe.
//
// So this invocation dequantises the PAIR `(2j, 2j+1)` and writes the whole
// word with `pie_pack_bf16`. The pair is free: `hidden` is a multiple of
// `PIE_GROUP` (32 at the narrowest) or the checkpoint would not pack, so both
// elements land in one packed word and one affine group -- the scale and the
// bias are loaded once and the second code is the next field of the same word.
//
// The grid is unchanged. `LaunchRule::Elementwise` covers `hidden` lanes where
// half that many do the work, and the extra ones fall out of the guard: an
// overshoot is harmless and an undershoot is the host's problem.

//#include "common/bf16.inc.wgsl"
//#include "common/affine.inc.wgsl"

@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> biases: array<u32>;
@group(0) @binding(3) var<storage, read> id: array<i32>;
@group(0) @binding(4) var<storage, read_write> out_: array<u32>;

//#if defined(PIE_SCALED)
struct Params { hidden: i32, embed_scale: f32 }
//#else
struct Params { hidden: i32 }
//#endif
@group(1) @binding(0) var<uniform> params: Params;

// One bf16 out of a packed `array<u32>` at a HALF-index. `pie_load_bf16` says
// this in one call and cannot be used: it takes a `ptr<storage, ...>`, and
// naga's validator allows a pointer argument only in the `private` and
// `function` address spaces -- `unrestricted_pointer_parameters` is
// unimplemented (gfx-rs/wgpu#5158), so a module that CALLS it is rejected by
// `create_shader_module` rather than by the parser.
fn load_half(word: u32, i: u32) -> f32 {
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

//#if defined(PIE_MB)
@compute @workgroup_size(16, 16)
//#else
@compute @workgroup_size(256)
//#endif
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // The word this invocation owns within the row: elements `2j` and `2j+1`.
    let j = gid.x;
//#if defined(PIE_MB)
    let m = gid.y;
//#else
    let m = 0u;
//#endif
    let hidden = u32(params.hidden);
    let k = 2u * j;
    if (k + 1u >= hidden) { return; }

    // The row guard is the OUTPUT's own length, and it is what bounds `m`: the
    // row count is not an operand, so a grid rounded up on the row axis would
    // otherwise read `id` past its end and -- worse -- write a clamped index,
    // which lands on the last element of `out_` rather than nowhere.
    let at = (m * hidden + k) >> 1u;
    if (at >= arrayLength(&out_)) { return; }

    let row = u32(id[m]);
    // Both codes are in this word and both share this group: `PIE_GROUP` and
    // `PIE_CODES_PER_WORD` are even and `k` is even, so `k` and `k+1` cannot
    // straddle either boundary.
    let packed = w[pie_affine_word_of(row, hidden, k)];
    let c = pie_affine_code_of(k);
    let g = pie_affine_scale_of(row, hidden, k);
    let s = load_half(scales[g >> 1u], g);
    let b = load_half(biases[g >> 1u], g);

    var lo = pie_affine_value(packed, c, s, b);
    var hi = pie_affine_value(packed, c + 1u, s, b);
//#if defined(PIE_SCALED)
    // gemma multiplies its embeddings by `sqrt(hidden)`, which is a number the
    // statement carries rather than one a kernel derives from the model.
    lo = lo * params.embed_scale;
    hi = hi * params.embed_scale;
//#endif
    out_[at] = pie_pack_bf16(lo, hi);
}

// pie:instantiate embed_gather_4bit_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4
// pie:instantiate embed_gather_4bit_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8
// pie:instantiate embed_gather_4bit_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4
// pie:instantiate embed_gather_4bit_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8
// pie:instantiate embed_gather_4bit_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate embed_gather_4bit_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8
// pie:instantiate embed_gather_mb_4bit_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4 PIE_MB=1
// pie:instantiate embed_gather_mb_4bit_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8 PIE_MB=1
// pie:instantiate embed_gather_mb_4bit_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4 PIE_MB=1
// pie:instantiate embed_gather_mb_4bit_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8 PIE_MB=1
// pie:instantiate embed_gather_mb_4bit_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_MB=1
// pie:instantiate embed_gather_mb_4bit_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_MB=1
// pie:instantiate embed_gather_scaled_4bit_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_4bit_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_4bit_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_4bit_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_4bit_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_4bit_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4 PIE_MB=1 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8 PIE_MB=1 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4 PIE_MB=1 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8 PIE_MB=1 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_MB=1 PIE_SCALED=1
// pie:instantiate embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_MB=1 PIE_SCALED=1
