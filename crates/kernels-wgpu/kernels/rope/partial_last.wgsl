// The rotation that turns the LAST `rotary` channels of a head.
//
// `rope/neox.wgsl` turns a head's leading channels; this one turns its
// trailing ones, and the two differ in three places rather than one. It is a
// separate file for that reason and not folded in as an eighth variant:
// `neox.wgsl` decomposes a pair into two words a fixed `dist` apart, which is
// the wrong decomposition for half of what this file has to do.
//
// ── THE THREE DIFFERENCES ─────────────────────────────────────────────────
//
// **The offset.** Channels `[0, head_dim - rotary)` are untouched and the
// rotation lives in `[head_dim - rotary, head_dim)`. `pie::rope::rotate_partial_last`
// in `kernels-cuda/kernels/rope/rope.cuh` spells that `offset = head_dim - rotary_dim`.
//
// **The denominator.** The exponent divides by `rotary`, NOT by `head_dim`:
//
//     freq(d) = theta ^ (-2d / rotary)
//
// That is the opposite of `rope.partial`, whose kernel divides by `head_dim`
// and whose cuda header records dividing by the rotary as the bug that variant
// exists to not have. The two points are not the same rotation at a different
// offset, and reading one file for the other's formula is the mistake this
// paragraph exists to stop.
//
// Written `exp2(-(d / rh) * base)` with `base = log2(theta)` and `rh = rotary/2`,
// because WGSL has no `pow` on a runtime base and `2d/rotary == d/rh`.
//
// **The pairing is a runtime fact of the checkpoint, and both are here.**
// `ssm`-adjacent DeepSeek-V4 fires this point with `interleaved = true` --
// GPT-J pairing, `(off + 2d, off + 2d + 1)` -- while NeoX pairing puts the
// partner half a ROTARY away, `(off + d, off + d + rh)`. They are two
// variants rather than a branch because they are two different WORD layouts,
// not two different index expressions: see below.
//
// ── WHY THE PAIRING CHANGES THE WORD LAYOUT ───────────────────────────────
//
// A bf16 tensor crosses as `array<u32>`, two channels to a word, and WGSL has
// no sub-word atomic -- so an invocation must own WHOLE WORDS or race its
// neighbour. Under NeoX the two halves of a pair sit `rh` channels apart and
// therefore in two different words, which is `neox.wgsl`'s lo/hi scheme.
// Under GPT-J they are ADJACENT: channels `off + 2d` and `off + 2d + 1` are
// the two halves of ONE word, so a pair is a single load and a single store
// and the partner word does not exist.
//
// Both arms give an invocation TWO pairs, so the grid is the same expression
// for either: `pairs.div_ceil(2)` invocations on x. Under NeoX that is two
// words `rh` apart; under GPT-J it is two adjacent words.
//
// `offset` must be even under both, or a pair straddles a word boundary at a
// half-word and no race-free decomposition exists. `rh` must be even under
// NeoX for the same reason. The claim body refuses both rather than computing
// a wrong address quietly; DeepSeek-V4's 128/64 satisfies each.

//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> position: array<i32>;

struct Params { base: f32, head_dim: i32, rotary: i32 }
@group(1) @binding(0) var<uniform> params: Params;

// `(x1, x2)` turned by `theta`.
fn pie_rotate(x1: f32, x2: f32, theta: f32) -> vec2<f32> {
    let c = cos(theta);
    let s = sin(theta);
    return vec2<f32>(x1 * c - x2 * s, x1 * s + x2 * c);
}

const PIE_LANES = 32u;

@compute @workgroup_size(PIE_LANES)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    // This invocation owns rotary pairs `2t` and `2t + 1`.
    let t = gid.x;
    let h = gid.y;
    let row = gid.z;

    let head_dim = u32(params.head_dim);
    let rotary = u32(params.rotary);
    let rh = rotary >> 1u;
    let n_head = grid.y;

    let d0 = 2u * t;
    if (d0 >= rh) { return; }

    // The rotated slice starts here: the head's base plus the untouched lead.
    let head = row * n_head * head_dim + h * head_dim + (head_dim - rotary);
    let pos = f32(position[row]);
    let theta0 = pos * exp2(-(f32(d0) / f32(rh)) * params.base);

//#if defined(PIE_INTERLEAVED)
    // GPT-J: a pair IS a word. Two pairs are two adjacent words, each read,
    // rotated and written on its own -- no partner word, no tearing.
    let w0 = (head + d0 * 2u) >> 1u;
    let word0 = x[w0];
    let r0 = pie_rotate(
        pie_bf16_to_f32(word0 & 0xffffu),
        pie_bf16_to_f32(word0 >> 16u),
        theta0,
    );
    x[w0] = pie_pack_bf16(r0.x, r0.y);

    if (d0 + 1u < rh) {
        let theta1 = pos * exp2(-(f32(d0 + 1u) / f32(rh)) * params.base);
        let word1 = x[w0 + 1u];
        let r1 = pie_rotate(
            pie_bf16_to_f32(word1 & 0xffffu),
            pie_bf16_to_f32(word1 >> 16u),
            theta1,
        );
        x[w0 + 1u] = pie_pack_bf16(r1.x, r1.y);
    }
//#else
    // NeoX: the partner is half a rotary away, so a pair spans two words.
    // Load both before storing either -- the rotation of the second reads the
    // first, and they live in the same buffer.
    let lo_at = (head + d0) >> 1u;
    let hi_at = (head + d0 + rh) >> 1u;
    let word_lo = x[lo_at];
    let word_hi = x[hi_at];
    let a0 = pie_bf16_to_f32(word_lo & 0xffffu);
    let a1 = pie_bf16_to_f32(word_lo >> 16u);
    let b0 = pie_bf16_to_f32(word_hi & 0xffffu);
    let b1 = pie_bf16_to_f32(word_hi >> 16u);

    let r0 = pie_rotate(a0, b0, theta0);
    // The odd tail: pair `d0 + 1` is past the rotated range, so both its
    // channels keep their values. Unreachable while `rh` is even, which the
    // claim body requires, and cheaper to carry than to argue about.
    var r1 = vec2<f32>(a1, b1);
    if (d0 + 1u < rh) {
        let theta1 = pos * exp2(-(f32(d0 + 1u) / f32(rh)) * params.base);
        r1 = pie_rotate(a1, b1, theta1);
    }
    x[lo_at] = pie_pack_bf16(r0.x, r1.x);
    x[hi_at] = pie_pack_bf16(r0.y, r1.y);
//#endif
}

// pie:instantiate neox_last_mb_bfloat16
// pie:instantiate gptj_last_mb_bfloat16 PIE_INTERLEAVED=1
