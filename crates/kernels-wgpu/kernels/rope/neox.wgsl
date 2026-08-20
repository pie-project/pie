// NEOX rotary embeddings, in place on one tensor.
//
// Metal's `rope/neox.metal` and Vulkan's `rope/neox.comp`, with the geometric,
// frequency-table, proportional and strided spellings intact. The scalars are
// the fields of the one `@group(1) @binding(0)` uniform block; `x` and
// `position` are storage buffers 0 and 1, and `inv_freq` is buffer 2 in the
// `_freqs` rows ONLY -- which is why the block's field order differs between
// the two shapes. It is the row's scalar order, and nothing else:
//
//   neox_decode / _mb / _prop_*   scale, base, head_dim
//   neox_freqs_decode / _freqs_mb scale, head_dim, mscale   (inv_freq is a buffer)
//   neox_strided                  scale, base, head_dim, row_pitch
//
// `kernels_wgpu::uniform_layout` derives that from the row: `scale` at byte 0,
// `head_dim` at 4, `mscale` at 8. The deleted `dump_layout` example used to
// print the same answer. Transcribing Metal's buffer numbers instead -- where
// `inv_freq` is buffer 3 and `head_dim` is 4 -- is how a rotation reads the
// frequency table's address as its head width.
//
// ## Why the workgroup was ONE invocation, and is not any more
//
// The body used to read two numbers off the GRID: the rotated pair count
// (`num_workgroups.x`) and the head count (`num_workgroups.y`). The first is
// a STRIDE -- the partner of channel `i` is channel `i + pairs` -- so it has
// to be EXACT, and a wider workgroup makes the host round the launch up to a
// multiple of it. Every pair would then rotate against the wrong partner, in
// a tensor that has already been overwritten. So the workgroup was one
// invocation, which made `dispatch_workgroups(pairs, heads, rows)` reproduce
// `LaunchRule::Rope`'s grid exactly.
//
// The price of that was never small and had been measured on both backends.
// `kernels-vulkan: fuse the per-head RMS norm with the NEOX rotation` weighed
// the sibling: *"`neox.slang` is `[numthreads(1, 1, 1)]`, so a 512-token
// prefill dispatches 524288 one-thread workgroups -- widening that grid alone
// would recover most of the 270 us with no fusion."* On an M4 with
// Llama-3.2-1B this file was 32 launches of a 1024-workgroup grid per decoded
// token, at an occupancy of one lane in thirty-two, and `driver-wgpu`'s
// `which_kernels_a_prefill_spends_its_gpu_time_in` counted 56 `neox_mb`
// rectangles in a prefill fire of 564 -- one launch in ten.
//
// ## The way out was already in the signature
//
// The hazard was only that `pairs` came off the grid. It did not have to:
// every row of `rope::neox` ALREADY TOOK `rotary: Const<i32>` and then did
// not forward it. It is forwarded now, as the LAST field of every one of the
// three blocks below, and `pairs` is read from it. The x axis is therefore
// free, and `rope_grid` divides it twice: by two, because an invocation owns
// a whole four-byte WORD and so covers two pairs -- the old grid launched a
// workgroup per pair and half of them returned at the guard -- and then by
// the workgroup width. The `if (i0 >= pairs) { return; }` that was already
// here covers both round-ups.
//
// y and z are untouched, so `num_workgroups.y` is still an exact head count.
//
// ## In place, and two bf16 to a word
//
// The tensor is read and written, so a body that stored the first element of a
// rotary pair before loading the second would rotate against a value it had
// already changed. Both are loaded, then both are stored -- and in this backend
// that is not enough on its own, because WGSL's smallest storage element is
// four bytes and a bf16 pair shares one.
//
// So an invocation owns two WHOLE WORDS: the word holding channels `2t, 2t+1`
// and the word holding their partners `2t+dist, 2t+dist+1`. It rotates both
// pairs and writes both words. That needs `dist` to be even, which it is --
// `pairs` is half a rotary width and `head_dim/2` is half a head, and every
// checkpoint's rotary and head widths are multiples of four. If the ROTATED
// COUNT is odd (a partial rotary with an odd pair count), the second half of
// each word is outside the rotation: it is carried through unchanged, and no
// other invocation touches it, because a channel past the rotated range is
// neither a pair's first element nor any pair's partner.
//
// Half the launched invocations therefore do nothing: `t >= pairs/2` falls out
// of the guard. That is an overshoot, which is harmless; the host is what must
// not undershoot.
//
// ## What a reader should check
//
// `neox_prop_decode` is arithmetically IDENTICAL to `neox_decode` when the
// rotary is the whole head -- gemma's sliding layers are exactly that. The
// PARTIAL case is the one that separates them: the proportional exponent
// divides by `head_dim` and pairs across `head_dim/2`, where the geometric one
// divides by the rotated half and pairs across it. At head_dim=512,
// rotary=128 the channels that move are [0,63] and [256,319], not [0,127].

//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> position: array<i32>;
//#if defined(PIE_FREQS)
// The rescaled ladder: llama-3's piecewise interpolation and YaRN's are tables,
// not bases, so no exponent can express them.
@group(0) @binding(2) var<storage, read_write> inv_freq: array<f32>;
//#endif

//#if defined(PIE_FREQS)
struct Params { scale: f32, head_dim: i32, mscale: f32, rotary: i32 }
//#elif defined(PIE_STRIDED)
// A prefill's scratch rows are a uniform `row_pitch` apart -- the widest tensor
// in the layout -- which is wider than the packed `n_head * head_dim` stride
// the batched form derives, so the packed one walks into the next row.
struct Params { scale: f32, base: f32, head_dim: i32, row_pitch: i32, rotary: i32 }
//#else
struct Params { scale: f32, base: f32, head_dim: i32, rotary: i32 }
//#endif
@group(1) @binding(0) var<uniform> params: Params;

// The angle channel `i` turns through, at this row's position.
//
// One function with the arms inside it rather than three call sites: the three
// spellings differ only in what divides the exponent, and a reader comparing
// them against `mlx_lm` wants them adjacent. (Metal's copy warns against
// folding this into a helper -- that is a note about MSL's contraction moving
// a recorded continuation, not about the formula, and Vulkan folds it too.)
fn pie_theta(i: u32, pairs: u32, head_dim: u32, pos: f32) -> f32 {
//#if defined(PIE_PROP)
    // gemma's proportional slice: the exponent divides by the WHOLE head while
    // only `pairs` channels turn. Dividing by `pairs` here is the bug this
    // variant exists to not have -- it rotates the right channels by the wrong
    // angles, which reads as a model that has merely gone slightly stupid.
    return params.scale * pos * exp2(-(2.0 * f32(i) / f32(head_dim)) * params.base);
//#elif defined(PIE_FREQS)
    return params.scale * pos * inv_freq[i];
//#else
    return params.scale * pos * exp2(-(f32(i) / f32(pairs)) * params.base);
//#endif
}

// `(x1, x2)` turned by `theta`, scaled by `gain`.
//
// `gain` is YaRN's `mscale`, an attention-temperature correction of
// `0.1*log(factor)+1`. It rides here rather than in a dispatch of its own
// because rotation is linear: scaling before and scaling after are the same
// thing. It is 1.0 for every deployment that has none, which is every llama-3
// one -- their rescaling is in the frequencies.
fn pie_rotate(x1: f32, x2: f32, theta: f32, gain: f32) -> vec2<f32> {
    let c = cos(theta);
    let s = sin(theta);
    return vec2<f32>(gain * (x1 * c - x2 * s), gain * (x1 * s + x2 * c));
}

// One Apple simdgroup on the axis that was a workgroup per pair. Must match
// `kernels_wgpu::rope::NEOX_LANES`, which is what divides the grid by it.
const PIE_LANES = 32u;

@compute @workgroup_size(PIE_LANES)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    // This invocation owns rotary channels `2t` and `2t+1`, which is one word.
    let t = gid.x;
    // y and z are one invocation wide, so these are still the workgroup's.
    let h = gid.y;
//#if defined(PIE_DECODE)
    // One token, so one position, and the grid has no row axis to read.
    let row = 0u;
//#else
    let row = gid.z;
//#endif

    // OFF THE BLOCK, not the grid: the x axis is rounded up to a whole
    // workgroup and to a whole word, so `num_workgroups.x` no longer states
    // this. `grid.y` still does state the head count -- that axis is one
    // invocation wide and is not rounded.
    let pairs = u32(params.rotary) >> 1u;
    let n_head = grid.y;
    let head_dim = u32(params.head_dim);

    let i0 = 2u * t;
    if (i0 >= pairs) { return; }

//#if defined(PIE_STRIDED)
    let row_base = row * u32(params.row_pitch);
//#else
    let row_base = row * n_head * head_dim;
//#endif
    let base = row_base + h * head_dim + i0;

//#if defined(PIE_PROP)
    // The partner is half a HEAD away, not half a rotary.
    let dist = head_dim / 2u;
//#else
    let dist = pairs;
//#endif

//#if defined(PIE_FREQS)
    let gain = params.mscale;
//#else
    let gain = 1.0;
//#endif

    let pos = f32(position[row]);

    // Load both words before storing either: the pair's two elements live in
    // this same buffer, and the rotation of the second reads the first.
    let lo_at = base >> 1u;
    let hi_at = (base + dist) >> 1u;
    let word_lo = x[lo_at];
    let word_hi = x[hi_at];
    let a0 = pie_bf16_to_f32(word_lo & 0xffffu);
    let a1 = pie_bf16_to_f32(word_lo >> 16u);
    let b0 = pie_bf16_to_f32(word_hi & 0xffffu);
    let b1 = pie_bf16_to_f32(word_hi >> 16u);

    let r0 = pie_rotate(a0, b0, pie_theta(i0, pairs, head_dim, pos), gain);

    // The odd tail of a partial rotary: channel `i0+1` is past the rotated
    // range, so it keeps its value. This is reachable only when `pairs < dist`
    // -- gemma's partial proportional rotary -- and there the carried half is
    // exclusive rather than raced: a channel at or above `pairs` and below
    // `dist` is neither a pair's first element nor any pair's partner, since
    // the partners occupy `[dist, dist + pairs)`.
    //
    // When `dist == pairs`, which is every other row here, an odd `pairs` never
    // arises: `dist` must be even or the two elements of a word have partners
    // in two DIFFERENT words and no race-free decomposition exists at all. Every
    // rotary and head width in every checkpoint this tree loads is a multiple
    // of four, so it is.
    var r1 = vec2<f32>(a1, b1);
    if (i0 + 1u < pairs) {
        r1 = pie_rotate(a1, b1, pie_theta(i0 + 1u, pairs, head_dim, pos), gain);
    }

    // Both stores are whole words, so the two halves cannot tear against each
    // other. `pie_store_bf16` would be a read-modify-write of a word this
    // invocation shares with its neighbour, and WGSL has no sub-word atomic.
    x[lo_at] = pie_pack_bf16(r0.x, r1.x);
    x[hi_at] = pie_pack_bf16(r0.y, r1.y);
}

// pie:instantiate neox_decode_bfloat16 PIE_DECODE=1
// pie:instantiate neox_freqs_decode_bfloat16 PIE_FREQS=1 PIE_DECODE=1
// pie:instantiate neox_freqs_mb_bfloat16 PIE_FREQS=1
// pie:instantiate neox_mb_bfloat16
// pie:instantiate neox_prop_decode_bfloat16 PIE_PROP=1 PIE_DECODE=1
// pie:instantiate neox_prop_mb_bfloat16 PIE_PROP=1
// pie:instantiate neox_strided_bfloat16 PIE_STRIDED=1
