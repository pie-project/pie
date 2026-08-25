// YaRN: the rotation whose frequencies ramp from extrapolated to interpolated
// across the head.
//
// `rope/neox.wgsl`'s `PIE_FREQS` arm exists for exactly this shape of problem
// and cannot serve this point on this plane: it reads the ladder out of a
// staged `inv_freq` buffer, and nothing on the wgpu plane stages one. Vulkan
// does -- `self.stream::<f32>("rope.yarn_inv_freq")` -- and cuda and metal
// evaluate the ramp in the kernel instead. This file is the third of those,
// because a `Cache` mark is the only door a wgpu body has to a buffer it did
// not receive, and `rope.yarn` declares none.
//
// ── THE RAMP ──────────────────────────────────────────────────────────────
//
// `pie::yarn_original_freq` in `kernels-cuda/kernels/prelude/rope.cuh`. It is
// a linear ramp in the DIM-INDEX domain -- not the wavelength domain llama-3's
// piecewise interpolation uses -- blending the unscaled frequency into a
// `1/factor`-scaled one:
//
//     base(d) = theta ^ (-2d / head_dim)
//     denom   = high == low ? high + 1e-3 - low : high - low
//     ramp    = clamp((d - low) / denom, 0, 1)
//     freq(d) = base(d) * ((1 - ramp) + ramp / factor)
//
// `low` and `high` are the correction dims, computed ON THE HOST from
// `beta_fast`, `beta_slow` and the checkpoint's original context length --
// `kernels_wgpu::rope::ramp_bounds`, which is `kernels-cuda/src/rope.rs`'s
// `ramp_bounds` transcribed. They are host arithmetic on stated numbers with
// no operand in them, so computing them per invocation would be the same two
// logarithms a few million times.
//
// The `high == low` guard is not decoration: the ramp's denominator is a
// difference of two host-rounded dims and they collapse for a narrow enough
// band, which without the guard divides by zero and paints the whole head with
// a NaN.
//
// ── `mscale` ──────────────────────────────────────────────────────────────
//
// `attention_factor` multiplies cos and sin BEFORE the rotation, which is
// where cuda and metal both put it. Rotation is linear so scaling afterwards
// is the same number in exact arithmetic; it is not the same FLOAT, and this
// family's parity is checked in floats.
//
// ── PAIRING, AND THE WORD LAYOUT IT DECIDES ───────────────────────────────
//
// Two variants for the two pairings, for `rope/partial_last.wgsl`'s reason:
// under NeoX a pair's halves are `head_dim/2` channels apart and so in two
// words, under GPT-J they are adjacent and so in one. That is a different
// decomposition rather than a different index, and a runtime branch over it
// would carry both through every load. gpt-oss fires the NeoX arm; cuda
// supports both and so does this.
//
// The rotation spans the WHOLE head -- `rope.yarn` states no rotary width --
// so `pairs` is `head_dim / 2` and the NeoX partner distance is `pairs`.

//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> position: array<i32>;

struct Params {
    base: f32,
    head_dim: i32,
    factor: f32,
    low_dim: f32,
    high_dim: f32,
    mscale: f32,
}
@group(1) @binding(0) var<uniform> params: Params;

// The angle pair `d` turns through, with the ramp folded in.
fn pie_yarn_theta(d: u32, pairs: u32, pos: f32) -> f32 {
    let base_freq = exp2(-(f32(d) / f32(pairs)) * params.base);
    let low = params.low_dim;
    let high = params.high_dim;
    var denom = high - low;
    if (high == low) {
        denom = high + 1e-3 - low;
    }
    let ramp = clamp((f32(d) - low) / denom, 0.0, 1.0);
    return pos * base_freq * ((1.0 - ramp) + ramp / params.factor);
}

// `(x1, x2)` turned by `theta`, with `mscale` on the cosine and the sine.
fn pie_rotate(x1: f32, x2: f32, theta: f32) -> vec2<f32> {
    let c = cos(theta) * params.mscale;
    let s = sin(theta) * params.mscale;
    return vec2<f32>(x1 * c - x2 * s, x1 * s + x2 * c);
}

const PIE_LANES = 32u;

@compute @workgroup_size(PIE_LANES)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    // This invocation owns pairs `2t` and `2t + 1`.
    let t = gid.x;
    let h = gid.y;
    let row = gid.z;

    let head_dim = u32(params.head_dim);
    let pairs = head_dim >> 1u;
    let n_head = grid.y;

    let d0 = 2u * t;
    if (d0 >= pairs) { return; }

    let head = row * n_head * head_dim + h * head_dim;
    let pos = f32(position[row]);
    let theta0 = pie_yarn_theta(d0, pairs, pos);

//#if defined(PIE_INTERLEAVED)
    let w0 = (head + d0 * 2u) >> 1u;
    let word0 = x[w0];
    let r0 = pie_rotate(
        pie_bf16_to_f32(word0 & 0xffffu),
        pie_bf16_to_f32(word0 >> 16u),
        theta0,
    );
    x[w0] = pie_pack_bf16(r0.x, r0.y);

    if (d0 + 1u < pairs) {
        let word1 = x[w0 + 1u];
        let r1 = pie_rotate(
            pie_bf16_to_f32(word1 & 0xffffu),
            pie_bf16_to_f32(word1 >> 16u),
            pie_yarn_theta(d0 + 1u, pairs, pos),
        );
        x[w0 + 1u] = pie_pack_bf16(r1.x, r1.y);
    }
//#else
    // Load both words before storing either: a pair's two channels live in
    // this same buffer and the rotation of the second reads the first.
    let lo_at = (head + d0) >> 1u;
    let hi_at = (head + d0 + pairs) >> 1u;
    let word_lo = x[lo_at];
    let word_hi = x[hi_at];
    let a0 = pie_bf16_to_f32(word_lo & 0xffffu);
    let a1 = pie_bf16_to_f32(word_lo >> 16u);
    let b0 = pie_bf16_to_f32(word_hi & 0xffffu);
    let b1 = pie_bf16_to_f32(word_hi >> 16u);

    let r0 = pie_rotate(a0, b0, theta0);
    var r1 = vec2<f32>(a1, b1);
    if (d0 + 1u < pairs) {
        r1 = pie_rotate(a1, b1, pie_yarn_theta(d0 + 1u, pairs, pos));
    }
    x[lo_at] = pie_pack_bf16(r0.x, r1.x);
    x[hi_at] = pie_pack_bf16(r0.y, r1.y);
//#endif
}

// pie:instantiate neox_yarn_mb_bfloat16
// pie:instantiate gptj_yarn_mb_bfloat16 PIE_INTERLEAVED=1
