// Gated RMSNorm for GDN: `out = w * rmsnorm(x) * silu(z)`.
//
// The math is Metal's GDN epilogue, including the numerically stable sigmoid
// and the RAW gate-norm weight -- no gemma `1 + w` here, which is the one thing
// about this kernel that reads like a mistake and is not: `ops/gated_delta.cpp`
// applies `gate_norm_w` directly.
//
// ## The defect this body does NOT inherit
//
// Metal launches `tg = (V_d, 1, 1)`: the threadgroup is exactly as wide as the
// axis, so every lane owns one channel and there is nothing to loop over. A
// WGSL `@workgroup_size` is fixed when the module COMPILES and cannot follow
// `params.vd`, so a body that kept "one lane per channel" would be correct up
// to its own declared width and silently wrong above it -- every channel from
// 256 up left out of the sum AND never written, so the norm divides by a mean
// taken over the first 256 channels and the rest of the head keeps whatever the
// arena held. `kernels-vulkan` records exactly that bug (`.wiki/new-driver/
// vulkan.md` §9); `v_d` is 128 on every GDN checkpoint the tree has seen, which
// is why nothing caught it, but the row states `vd` as a runtime scalar and so
// promises to honour it.
//
// Both loops here therefore walk the axis in strides of the workgroup width,
// and the guards sit on the STORES rather than on an early return: the sum
// between them barriers, and a lane that returned in front of a barrier would
// hang the ones that did not.
//
// The buffers are the sibling kernels' order -- the row is UNSTATED, so the
// Metal signature is the only written description of the call there is -- and
// the strided form's `row_pitch` is the one field of the uniform block, where
// Vulkan puts it in a push block WebGPU does not have.

//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

// One name for the attribute below and for the width `pie_inv_rms` folds.
const PIE_LANES = 256u;

struct GatedRmsParams {
    eps: f32,
    vd: u32,
}

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> z: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
// Atomic for the odd-`vd` edge alone -- `store_half` says why -- and the host
// binds the same read_write storage buffer of 4-byte words either way.
@group(0) @binding(3) var<storage, read_write> out_: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> params: GatedRmsParams;

//#if defined(PIE_STRIDED)
struct Strided { row_pitch: i32 }
@group(1) @binding(0) var<uniform> strided: Strided;
//#endif

// The half-index split, one reader per binding. `pie_bf16_at` takes a WORD
// rather than the buffer because core WGSL allows a pointer parameter only in
// the `function`, `private` and `workgroup` address spaces -- a shared
// `load(&buffer, i)` parses and then fails validation.
fn x_at(i: u32) -> f32 {
    return pie_bf16_at(x[i >> 1u], i);
}

fn z_at(i: u32) -> f32 {
    return pie_bf16_at(z[i >> 1u], i);
}

fn w_at(i: u32) -> f32 {
    return pie_bf16_at(w[i >> 1u], i);
}

// One bf16 of a word this invocation does not own outright.
//
// Only an odd `vd` or `row_pitch` reaches this: the head then shares its edge
// word with the neighbouring head, whose workgroup is writing the other half at
// the same time. A read-modify-write keeps whichever landed second and drops
// the other channel; the device-scoped compare-exchange keeps both, retrying
// the spurious failure `...Weak` is permitted. Every GDN checkpoint has an even
// `v_d`, so this is an edge and not the hot path -- which is exactly why it
// would never have been caught if it were wrong.
fn store_half(i: u32, value: f32) {
    let at = i >> 1u;
    var old = atomicLoad(&out_[at]);
    loop {
        let res = atomicCompareExchangeWeak(&out_[at], old, pie_bf16_into(old, i, value));
        if (res.exchanged) { break; }
        old = res.old_value;
    }
}

// MLX's numerically stable sigmoid: the exponent is taken of `-|v|`, so it
// cannot overflow, and the branch puts the reflection back.
fn stable_sigmoid(v: f32) -> f32 {
    let y = 1.0 / (1.0 + exp(-abs(v)));
    return select(y, 1.0 - y, v < 0.0);
}

// One output element. `abs` addresses the head's data; `at` is the channel,
// which is what indexes the gate-norm weight -- one vector shared by every
// head, so feeding it the absolute index would read the next head's gains.
fn gated(abs: u32, at: u32, inv: f32) -> f32 {
    let zr = z_at(abs);
    let normed = x_at(abs) * inv * w_at(at);
    return normed * (zr * stable_sigmoid(zr));
}

@compute @workgroup_size(PIE_LANES)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
//#if !defined(PIE_STRIDED)
    @builtin(num_workgroups) grid: vec3<u32>,
//#endif
) {
    let lane = lid.x;
    let vd = params.vd;
//#if defined(PIE_STRIDED)
    // `.z` is the token and `.y` the value head, and the token's pitch is a
    // scalar because a prefill's heads are packed inside a wider row.
    let base = wg.z * u32(strided.row_pitch) + wg.y * vd;
//#else
    // Densely packed: the head's index is the grid's own row-major fold, which
    // is why this arm needs `num_workgroups` and the strided one does not.
    let base = (wg.z * grid.y + wg.y) * vd;
//#endif

    var sum_sq = 0.0;
    for (var at = lane; at < vd; at = at + PIE_LANES) {
        let v = x_at(base + at);
        sum_sq = sum_sq + v * v;
    }
    let inv = pie_inv_rms(lane, PIE_LANES, sum_sq, vd, params.eps);

    // One lane per WORD for the stores, for the reason `norm/rms.wgsl` states
    // at length: a half-word write is a read-modify-write, and the two lanes
    // that can share a word are in different WORKGROUPS, where no barrier
    // reaches. The interior of a head is whole words and needs none of that.
    let first = base >> 1u;
    let end = (base + vd + 1u) >> 1u;
    for (var word = first + lane; word < end; word = word + PIE_LANES) {
        let lo = word * 2u;
        let hi = lo + 1u;
        let has_lo = lo >= base && lo < base + vd;
        let has_hi = hi < base + vd;
        if (has_lo && has_hi) {
            // Both channels are this head's, so this lane owns the word: one
            // write, and nothing else in the launch can touch it.
            atomicStore(&out_[word], pie_pack_bf16(
                gated(lo, lo - base, inv),
                gated(hi, hi - base, inv),
            ));
        } else if (has_hi) {
            // Only an odd `vd` or `row_pitch` reaches either of these, and the
            // other half of the word then belongs to the neighbouring head --
            // a different workgroup, writing concurrently, which is why this
            // goes through the compare-exchange and not a plain edit.
            store_half(hi, gated(hi, hi - base, inv));
        } else if (has_lo) {
            store_half(lo, gated(lo, lo - base, inv));
        }
    }
}

// pie:instantiate gated_rms_bfloat16
// pie:instantiate gated_rms_strided_bfloat16 PIE_STRIDED=1
