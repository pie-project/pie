// Load-time quant transcodes for the affine and MXFP4 formats.
//
// The affine encoder follows MLX's CPU quantizer where it is unusual: the scale
// sign chooses the dominant endpoint, endpoint snapping changes the scale rather
// than the code, `w_max` starts at zero, and ties round away from zero. MXFP4
// decode shares `common/mxfp4.inc.wgsl` with the runtime readers so the two
// paths cannot drift.
//
// ## Bindings
//
// The three rows here state no operands, so `kernels_wgpu::bindings()` has
// nothing to answer and a shell falls back to the lowered plan's own argument
// order (`.wiki/new-driver/vulkan.md` §13). That order is `kernels-metal`'s
// parameter list, and this file numbers by applying the crate's two-run rule to
// it -- buffers densely from 0 in `@group(0)`, the scalar run as fields of the
// one `@group(1) @binding(0)` block:
//
//     mxfp4_dequant_bf16   payload 0, exponents 1, out_ 2       | blocks, block_size
//     affine_encode_u4_*   input 0, codes 1, scales 2, biases 3 | groups, group_size
//
// Metal spells the scalar pair as `constant DequantParams&`, a buffer like any
// other, and Vulkan copied that into a `readonly buffer` at binding 3. Neither
// reading survives here: two words in a storage slot would spend one of
// WebGPU's eight GUARANTEED storage buffers on eight bytes, and the crate's ABI
// already says where a scalar rides.

//#include "common/bf16.inc.wgsl"
//#if defined(PIE_MXFP4)
//#include "common/mxfp4.inc.wgsl"
//#endif

//#if defined(PIE_MXFP4)
// An MXFP4 payload is two codes per byte and an E8M0 plane is one byte per
// block. WGSL's smallest storage element is a WORD, so both are `array<u32>`
// and `common/mxfp4.inc.wgsl` owns the split.
@group(0) @binding(0) var<storage, read_write> payload: array<u32>;
@group(0) @binding(1) var<storage, read_write> exponents: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;

struct Params {
    blocks: u32,
    block_size: u32,
}
//#else
//#if defined(PIE_F32_INPUT)
@group(0) @binding(0) var<storage, read_write> input_f32: array<f32>;
//#else
@group(0) @binding(0) var<storage, read_write> input_bf16: array<u32>;
//#endif
@group(0) @binding(1) var<storage, read_write> codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> biases: array<u32>;

struct Params {
    groups: u32,
    group_size: u32,
}
//#endif

@group(1) @binding(0) var<uniform> params: Params;

// Half away from zero, which is what MLX's `round` is and what `rint` is not.
// The difference was an 8.2% disagreement with `mx.quantize` on an
// MXFP4-derived expert bank -- every mismatch by exactly one, because those
// values sit on half-integers by construction.
fn round_away(x: f32) -> f32 {
    return select(floor(x + 0.5), -floor(-x + 0.5), x < 0.0);
}

//#if defined(PIE_MXFP4)
// One invocation per block: 16 bytes read and 64 written at the format's 32,
// wide enough that the per-invocation setup disappears and narrow enough that
// the exponent stays in a register. 256 is the tree's pointwise workgroup
// width (`norm/residual_add.wgsl` and the rest), and it has to be a number the
// SHELL also knows: `dispatch_workgroups` counts workgroups, so the host
// divides the block count by this and an undershoot is a gap nothing writes.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block = gid.x;
    // The stated count and not `arrayLength(&out_)`: `dispatch_workgroups`
    // rounds up to a whole workgroup, and the output is `blocks * block_size`
    // ELEMENTS in a buffer of half as many words, so the buffer's length is not
    // the bound. Nothing below is a barrier, so an early return is safe here.
    if (block >= params.blocks) { return; }

    let factor = pie_mxfp4_block_scale(pie_mxfp4_byte(exponents[block >> 2u], block));
    let first = block * params.block_size;

    // A whole WORD per iteration -- the pair `(first + i, first + i + 1)` is
    // one word of the bf16 output because `block_size` is even (32, fixed by
    // the format) and so `first` is even too. `pie_store_bf16` would be a
    // read-modify-write of a word this invocation only half owns, and WGSL has
    // no sub-word atomic to make that safe.
    for (var i = 0u; i < params.block_size; i = i + 2u) {
        let at = (first + i) >> 1u;  // the byte, and the output word, both
        let byte_ = pie_mxfp4_byte(payload[at >> 2u], at);
        out_[at] = pie_pack_bf16(
            pie_mxfp4_lo(byte_) * factor,
            pie_mxfp4_hi(byte_) * factor,
        );
    }
}
//#else
fn input_at(i: u32) -> f32 {
//#if defined(PIE_F32_INPUT)
    return input_f32[i];
//#else
    // `pie_load_bf16` inlined, and it has to be: naga 30 refuses a
    // `ptr<storage, ...>` function parameter, so the include's own loader
    // parses and then fails validation on every device. The conversion itself
    // is still the include's -- only the half-index split is here.
    let word = input_bf16[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
//#endif
}

// The (scale, bias) MLX picks for one group, as `vec2(scale, bias)`.
//
// Transcribed from `mlx/backend/cpu/quantized.cpp::quantize`, because a
// checkpoint this produces has to be interchangeable with one `mlx_lm convert`
// produced. Three details are not what an independently written quantizer
// would do, and all three are load-bearing:
//
//   * the scale is NEGATED unless the group's minimum is the larger in
//     magnitude, which puts code 0 on whichever end dominates;
//   * the ENDPOINT is snapped, not the scale, which keeps the largest
//     magnitude in the group exact;
//   * `w_max` starts at ZERO, so an all-negative group is quantized over the
//     range up to zero rather than up to its own largest element.
fn group_params(group: u32) -> vec2<f32> {
    // A group past the end contributes nothing: it exists only because the
    // store below writes a whole word, and the last word of an odd group count
    // has a half nobody owns.
    if (group >= params.groups) { return vec2<f32>(0.0, 0.0); }

    let first = group * params.group_size;
    var w_min = bitcast<f32>(0x7f800000u);  // +inf
    var w_max = 0.0;
    for (var i = 0u; i < params.group_size; i = i + 1u) {
        let v = input_at(first + i);
        w_min = min(w_min, v);
        w_max = max(w_max, v);
    }

    let mask = abs(w_min) > abs(w_max);
    var scale = max((w_max - w_min) / 15.0, 1e-7);
    if (!mask) { scale = -scale; }
    let edge = select(w_max, w_min, mask);
    let q0 = round_away(edge / scale);
    var bias = 0.0;
    if (q0 != 0.0) {
        scale = edge / q0;
        bias = edge;
    }
    return vec2<f32>(scale, bias);
}

// One invocation per group: `group_size` elements in, `group_size / 8` packed
// code words plus one scale and one bias out.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let group = gid.x;
    if (group >= params.groups) { return; }

    let prm = group_params(group);

    // `scales` and `biases` are one bf16 per GROUP, so two adjacent groups --
    // two adjacent invocations -- share one 32-bit word. The EVEN invocation of
    // each pair computes its neighbour's parameters as well and writes the
    // whole word; the odd one writes neither plane. The alternative is
    // `pie_store_bf16`, a read-modify-write of a word two invocations each own
    // half of, and WGSL has no sub-word atomic that could make it safe: one of
    // the two groups' scales would simply be lost, and a lost scale is a
    // wrong-by-a-factor row rather than a crash.
    //
    // The cost is one extra min/max scan per pair, on a kernel that runs once
    // at model load.
    if ((group & 1u) == 0u) {
        let odd = group_params(group + 1u);
        scales[group >> 1u] = pie_pack_bf16(prm.x, odd.x);
        biases[group >> 1u] = pie_pack_bf16(prm.y, odd.y);
    }

    // The codes come from the f32 parameters, not from their bf16 rounding:
    // MLX rounds only on store, and agreeing with MLX matters more than being
    // self-consistent with what the runtime will read back.
    //
    // Eight u4 codes to a word, and `first / 8` starts this invocation's own
    // run of `group_size / 8` words -- no word is shared with a neighbour, so
    // these are plain stores.
    let first = group * params.group_size;
    for (var word = 0u; word < params.group_size / 8u; word = word + 1u) {
        var packed = 0u;
        for (var k = 0u; k < 8u; k = k + 1u) {
            let v = input_at(first + word * 8u + k);
            let q = round_away((v - prm.y) / prm.x);
            packed = packed | (u32(clamp(q, 0.0, 15.0)) << (4u * k));
        }
        codes[first / 8u + word] = packed;
    }
}
//#endif

// pie:instantiate affine_encode_u4_bf16 PIE_ENCODE=1
// pie:instantiate affine_encode_u4_f32 PIE_ENCODE=1 PIE_F32_INPUT=1
// pie:instantiate mxfp4_dequant_bf16 PIE_MXFP4=1
