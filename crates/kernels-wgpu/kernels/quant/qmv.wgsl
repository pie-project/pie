// Affine GEMV projections, ported from `quant/qmv.comp` (itself from
// `quant/qmv.metal`).
//
// The port keeps MLX's little-endian affine packing, the four-rows-per-slot
// launch shape, and the bounds-checked K tail whose absence broke Gemma-4-31B
// at K=5376. WebGPU promises no subgroup width at all in the Baseline tier, so
// the final reduction is through workgroup memory over the 32 local-x lanes --
// the same shape Vulkan's port took, for the same reason.
//
// Two things are WGSL's alone and are the only real divergences from the
// `.comp`:
//
//   * every bf16 tensor is an `array<u32>` holding TWO values per word, low
//     half first, because WGSL has no 16-bit storage type even with `enable
//     f16`. Every index the GLSL uses against `uint16_t[]` is a half-index
//     here. See `common/bf16.inc.wgsl`.
//   * the output store is therefore a read-modify-write of a word this
//     invocation may only half own, and it goes through a CAS loop. See
//     `store_y` for which race that is and why it is not hypothetical.

//#include "common/bf16.inc.wgsl"
//#include "common/affine.inc.wgsl"

// ── Bindings ────────────────────────────────────────────────────────────────
//
// Numbered from the ROW in `src/quant.rs`, densely from zero over its
// buffer-kinded operands, with the scalars going to the uniform block instead.
// `.wiki/new-driver/vulkan.md` §3 records what happens when they are copied
// from Metal instead: Metal numbers scalars in the same run as buffers, so its
// `residual` is buffer 7 while the row puts it at 5, and a shader that
// transcribes the 7 binds a buffer the shell never wrote.

@group(0) @binding(0) var<storage, read_write> w: array<u32>;
@group(0) @binding(1) var<storage, read_write> scales: array<u32>;
@group(0) @binding(2) var<storage, read_write> biases: array<u32>;
@group(0) @binding(3) var<storage, read_write> x: array<u32>;
// `atomic` for the CAS in `store_y`, not because anything here accumulates.
@group(0) @binding(4) var<storage, read_write> y: array<atomic<u32>>;
// Declared only by the variants whose row has a sixth buffer operand:
// `affine_qmv_fast` has five and no sixth binding at all, so this cannot be
// declared unconditionally -- a module that binds past the row's buffer count
// is a pipeline-layout error, while a HOLE in the set is fine (§13).
//#if defined(PIE_RESIDUAL) || defined(PIE_BIAS)
@group(0) @binding(5) var<storage, read_write> extra: array<u32>;
//#endif

// The block is variant-shaped, and it has to be. `uniform_layout()` builds the
// buffer from what the ROW states -- `affine_qmv_fast` states two scalars -- so
// the fields here are exactly that row's scalars in that row's order, and
// `row_stride`/`m` appear only for the strided variant that reads them. The
// Vulkan file carried three more (`x_slot_stride`, `x_row_stride`,
// `slots_per_row`) that were the ROUTED row's shape, copied in and read by
// nothing.
struct Params {
    in_vec_size: i32,
    out_vec_size: i32,
//#if defined(PIE_WIDE_STRIDED)
    row_stride: i32,
    m: i32,
//#endif
}
@group(1) @binding(0) var<uniform> params: Params;

// Activation rows one workgroup carries, and the whole point of the shape.
//
// A workgroup owns FOUR output columns and reads their weights over the whole
// of K -- four rows of the packed matrix, which for a 2048-wide int4 head is
// 4 KiB.

// It owned EIGHT, as two y-slots of four, until the workgroup was narrowed to
// `@workgroup_size(32, 1, 1)`. The launchers state y in INVOCATIONS and the
// driver divides by the module's own width, so dropping the y extent doubled
// the workgroup count at the same total work without touching a launcher.
// Measured per dispatch, decode shapes: q/o 2048x2048 0.025 -> 0.021 ms
// (147 -> 179 GB/s), gate/up 8192x2048 0.063 -> 0.061, down 2048x8192
// 0.047 -> 0.044, lm head 128256x2048 0.793 -> 0.783. End to end tg128
// 104.7 -> 107.6 tok/s and tg256@2048 73.8 -> 75.5, prefill unchanged.
// A decode's k/v plane is 512 rows, which was 64 workgroups on a 20-core
// GPU -- three per core, with no second wave to hide a memory stall behind.
// This is the opposite lever to widening x to 64 lanes, which was tried and
// measured worse at every shape; the parallelism this kernel wants is MORE
// workgroups, not wider ones. One activation row per workgroup meant the lm head read that 8 KiB
// once per token: a 512-token prefill pulled the whole 64 MiB head 512 times,
// 67 GiB, and the measured 617 ms of a 1981 ms prefill is exactly that traffic
// at this device's bandwidth. Four rows to a workgroup retire four times the
// arithmetic per weight word and divide the traffic by four.
//
// FOUR and not more because the accumulators are registers: one `vec4<f32>`
// per activation row per four output columns, plus a `vec4` of activation
// sums. Eight would double that and the k loop is the whole kernel.
//
// A VARIANT AXIS and not a plain constant, because those registers are not
// free where there is nothing to spend them on. A decode fire is ONE row, and
// the four-row body reached it at 61 tok/s where the one-row body reaches 64 --
// the arithmetic is dead there and the allocation is not. `quant::qmv_fast`
// row, which is also exactly when `Rule::Qmv`'s quartered x extent differs
// from the row count.
const PIE_MT = 2u;

// 4 output columns x `PIE_MT` activation rows x the 32 local-x lanes each
// column is split over.
var<workgroup> qmv_partials: array<f32, 4u * PIE_MT * 32u>;

// The `i`-th bf16 of a word already loaded. Not `pie_load_bf16`: that one takes
// a `ptr<storage, ...>`, which naga 30 parses and then refuses to validate
// ("which can't be passed into functions"), so it cannot appear in a module
// that has to reach a device. The conversion is still the include's.
fn qmv_bf16(word: u32, i: u32) -> f32 {
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

// Write one bf16 output element.
//
// The word at `i >> 1` holds outputs `i & ~1` and `i | 1`, and this invocation
// generally owns only one of them: with an odd `out_vec_size` -- a one-column
// gate projection is a real shape -- consecutive outputs of one row land in one
// word while belonging to different y-slots, and consecutive `vec`s land in one
// word while belonging to different WORKGROUPS. A plain read-modify-write
// therefore loses whichever half landed second, and no workgroup barrier can
// fix the cross-workgroup case. The CAS is device-scoped, which is exactly the
// scope of the race; it retries on the spurious failure `...Weak` is allowed.
fn store_y(i: u32, v: f32) {
    let at = i >> 1u;
    let shift = (i & 1u) * 16u;
    let keep = ~(0xffffu << shift);
    let bits = pie_f32_to_bf16(v) << shift;
    var old = atomicLoad(&y[at]);
    loop {
        let r = atomicCompareExchangeWeak(&y[at], old, (old & keep) | bits);
        if r.exchanged {
            break;
        }
        old = r.old_value;
    }
}

fn affine_value(row: u32, k: u32) -> f32 {
    let len = u32(params.in_vec_size);
    let word = w[pie_affine_word_of(row, len, k)];
    let g = pie_affine_scale_of(row, len, k);
    return pie_affine_value(
        word,
        pie_affine_code_of(k),
        qmv_bf16(scales[g >> 1u], g),
        qmv_bf16(biases[g >> 1u], g),
    );
}

fn x_at(vec_: u32, k: u32) -> f32 {
//#if defined(PIE_WIDE_STRIDED)
    let i = vec_ * u32(params.row_stride) + k;
//#else
    let i = vec_ * u32(params.in_vec_size) + k;
//#endif
    return qmv_bf16(x[i >> 1u], i);
}

// The WORD holding activations `k` and `k + 1`, for an even `k`.
//
// `x_at` reads `x[i >> 1u]` and keeps one half, so a loop that walks `k` one at
// a time loads every word TWICE. A lane's block is sixteen activations against
// eight weight-word loads at four bits, so those redundant loads were the
// commonest instruction in the inner loop.
fn x_word(vec_: u32, k: u32) -> u32 {
//#if defined(PIE_WIDE_STRIDED)
    let i = vec_ * u32(params.row_stride) + k;
//#else
    let i = vec_ * u32(params.in_vec_size) + k;
//#endif
    return x[i >> 1u];
}

// Values one lane pulls per pass: FOUR words' worth, so 32 at four bits and 16
// at eight.
//
// It was two words'. Four halves the trip count of the K loop and doubles the
// weight loads in flight per lane, which is what a serial-dispatch GPU has to
// hide a memory stall behind. Measured per dispatch: lm head 128256x2048
// 0.783 -> 0.741 ms (188 -> 199 GB/s), gate/up 8192x2048 0.061 -> 0.058,
// down 2048x8192 0.044 -> 0.042; tg128 107.6 -> 110.2 tok/s. Prefill is
// unchanged -- it goes through `qmm_t`.
//
// FOUR AND NOT EIGHT. The scale and bias are hoisted to once per lane-block,
// which is only sound while a block cannot straddle two quantization groups
// -- `PIE_QMV_VPT <= PIE_GROUP`, and the narrowest group this file stamps is
// 32. Eight words is 64 codes at four bits and would read one scale for two
// groups' weights, which is wrong and not slow. The GLSL passed this as a runtime argument and then wrote the inner
// loop to a constant 16 with `i < values_per_thread` inside; here it is a real
// const-expression -- `PIE_BITS` is a prelude const -- so the loop bound is the
// constant and the redundant guard is gone. The OTHER guard is not redundant:
// see `dot_lane`.
// Whether the nibble fast path above applies: four bits to a code, so a
// 32-bit word is eight of them and two `unpack4x8unorm` calls cover it.
const PIE_NIBBLE_FAST = PIE_BITS == 4;
const PIE_QMV_VPT = PIE_CODES_PER_WORD * 4;

// One lane's block of `PIE_QMV_VPT` values, for FOUR rows at once.
//
// This is `qmv_fast_impl`'s inner loop as MLX writes it, and the shape of it is
// the whole point. The version this replaces called `affine_value(row, k)` once
// PER ELEMENT PER ROW, and that helper re-derived the word index, re-read the
// packed word, and re-read the scale AND the bias every time -- so a lane's
// sixteen values cost sixteen word loads and thirty-two scale loads per row,
// where they need two and one.
//
// Three things are hoisted, and each is hoisted to where the value actually
// changes:
//
// * **The scale and bias, to once per lane-block.** `PIE_QMV_VPT` divides
//   `PIE_GROUP` at every instantiation this file stamps -- sixteen into 32, 64
//   and 128 at four bits, eight into the same at eight -- so a lane's values
//   never straddle two groups and one pair covers them all.
// * **The packed word, to once per `PIE_CODES_PER_WORD` values.** That is what
//   the word IS: eight codes at four bits, four at eight.
// * **The activation, to once per value.** It was read once per value PER ROW.
//
// The four rows ride in a `vec4` rather than a loop, which is not a flourish:
// a `var` array indexed by a loop variable is scratch memory in every backend
// naga targets, so the array that would make this readable would also make it
// slower than what it replaces.
// One lane's block for four output columns and ONE activation row.
//
// The `PIE_MT` form below does the same arithmetic for four rows at a time and
// is what a prefill wants; a DECODE fire is one row, and running it through the
// four-row form spends four times the multiplies to throw three quarters of
// them away. Measured: 64.0 tok/s at ctx 512 with this form, 48.6 without it
// and 51.6 with only the reduction narrowed -- so it is the k loop's
// arithmetic that a decode cannot afford, not the tree. `reduce_store` picks
// between the two on a workgroup-uniform test, which is why neither needs a
// barrier inside it.
fn block_dot1(rows: vec4<u32>, vec_: u32, k0: u32) -> vec4<f32> {
    let n = u32(params.in_vec_size);
    let cpw = u32(PIE_CODES_PER_WORD);
    let mask = (1u << u32(PIE_BITS)) - 1u;
    let g = rows * (n / u32(PIE_GROUP)) + vec4<u32>(k0 / u32(PIE_GROUP));
    let s = vec4<f32>(
        qmv_bf16(scales[g.x >> 1u], g.x),
        qmv_bf16(scales[g.y >> 1u], g.y),
        qmv_bf16(scales[g.z >> 1u], g.z),
        qmv_bf16(scales[g.w >> 1u], g.w),
    );
    let b = vec4<f32>(
        qmv_bf16(biases[g.x >> 1u], g.x),
        qmv_bf16(biases[g.y >> 1u], g.y),
        qmv_bf16(biases[g.z >> 1u], g.z),
        qmv_bf16(biases[g.w >> 1u], g.w),
    );
    let wbase = rows * (n / cpw);
    var acc = vec4<f32>(0.0);
    var xsum = 0.0;
    let whole = k0 + u32(PIE_QMV_VPT) <= n;
    for (var jw = 0u; jw < u32(PIE_QMV_VPT) / u32(PIE_CODES_PER_WORD); jw = jw + 1u) {
        let base = k0 + jw * u32(PIE_CODES_PER_WORD);
        if base >= n {
            break;
        }
        let at = wbase + vec4<u32>(base / u32(PIE_CODES_PER_WORD));
        let word = vec4<u32>(w[at.x], w[at.y], w[at.z], w[at.w]);
        // FOUR CODES AN INSTRUCTION, for the shape that is 92% of a decode.
        //
        // A nibble loop pays a shift, a mask and an integer-to-float convert
        // per code per row -- three instructions for one number, and qmv is
        // ALU-bound: with the unpack stubbed out the kernel reads its weights
        // at 113 GB/s against this machine's ~120 GB/s roofline, so every
        // instruction removed here is time.
        //
        // `unpack4x8unorm` is Metal's `unpack_unorm4x8_to_float` and SPIR-V's
        // `UnpackUnorm4x8`: one instruction for four bytes. Masking the word
        // to `0x0f0f0f0f` puts codes 0, 2, 4 and 6 alone in their bytes and
        // shifting first puts 1, 3, 5 and 7 there, so two unpacks cover the
        // word.
        //
        // **The `255.0` is exact.** The builtin divides each byte by 255 and
        // this multiplies it back; for a byte below 256 both roundings are
        // below half an ulp of a value whose integer neighbours are 2^-8 apart
        // in relative terms, so the round trip is the identity. The codes are
        // therefore the same numbers the loop below produces, multiplied into
        // `acc` in the same order, which is what the cross-backend parity walk
        // compares.
        if PIE_NIBBLE_FAST && whole {
            let lo = vec4<u32>(0x0f0f0f0fu);
            let e0 = unpack4x8unorm(word.x & lo.x) * 255.0;
            let e1 = unpack4x8unorm(word.y & lo.y) * 255.0;
            let e2 = unpack4x8unorm(word.z & lo.z) * 255.0;
            let e3 = unpack4x8unorm(word.w & lo.w) * 255.0;
            let sh = (word >> vec4<u32>(4u)) & lo;
            let o0 = unpack4x8unorm(sh.x) * 255.0;
            let o1 = unpack4x8unorm(sh.y) * 255.0;
            let o2 = unpack4x8unorm(sh.z) * 255.0;
            let o3 = unpack4x8unorm(sh.w) * 255.0;
            let w0 = x_word(vec_, base);
            let w1 = x_word(vec_, base + 2u);
            let w2 = x_word(vec_, base + 4u);
            let w3 = x_word(vec_, base + 6u);
            let x0 = pie_bf16_to_f32(w0 & 0xffffu);
            let x1 = pie_bf16_to_f32(w0 >> 16u);
            let x2 = pie_bf16_to_f32(w1 & 0xffffu);
            let x3 = pie_bf16_to_f32(w1 >> 16u);
            let x4 = pie_bf16_to_f32(w2 & 0xffffu);
            let x5 = pie_bf16_to_f32(w2 >> 16u);
            let x6 = pie_bf16_to_f32(w3 & 0xffffu);
            let x7 = pie_bf16_to_f32(w3 >> 16u);
            xsum = xsum + x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
            acc = acc + x0 * vec4<f32>(e0.x, e1.x, e2.x, e3.x);
            acc = acc + x1 * vec4<f32>(o0.x, o1.x, o2.x, o3.x);
            acc = acc + x2 * vec4<f32>(e0.y, e1.y, e2.y, e3.y);
            acc = acc + x3 * vec4<f32>(o0.y, o1.y, o2.y, o3.y);
            acc = acc + x4 * vec4<f32>(e0.z, e1.z, e2.z, e3.z);
            acc = acc + x5 * vec4<f32>(o0.z, o1.z, o2.z, o3.z);
            acc = acc + x6 * vec4<f32>(e0.w, e1.w, e2.w, e3.w);
            acc = acc + x7 * vec4<f32>(o0.w, o1.w, o2.w, o3.w);
            continue;
        }
        // TWO CODES A TRIP, because both read the same activation word.
        // `base` is a multiple of `PIE_CODES_PER_WORD` and `in_vec_size` is
        // even, so `k` is even here and the word holds `k` in its low half and
        // `k + 1` in its high one. The order the products reach `acc` is the
        // order it was -- c, then c+1 -- so this is the same sum.
        for (var c = 0u; c < u32(PIE_CODES_PER_WORD); c = c + 2u) {
            if !whole && base + c >= n {
                break;
            }
            let xw = x_word(vec_, base + c);
            let xv0 = pie_bf16_to_f32(xw & 0xffffu);
            xsum = xsum + xv0;
            let code0 = (word >> vec4<u32>(u32(PIE_BITS) * c)) & vec4<u32>(mask);
            acc = acc + xv0 * vec4<f32>(code0);
            if !whole && base + c + 1u >= n {
                break;
            }
            let xv1 = pie_bf16_to_f32(xw >> 16u);
            xsum = xsum + xv1;
            let code1 = (word >> vec4<u32>(u32(PIE_BITS) * (c + 1u))) & vec4<u32>(mask);
            acc = acc + xv1 * vec4<f32>(code1);
        }
    }
    return s * acc + b * vec4<f32>(xsum);
}

// WHAT IS LEFT IN HERE, MEASURED, SO THE NEXT PASS STARTS FROM A NUMBER
//
// On an M4 Pro with Llama-3.2-1B this file is most of a decoded token's GPU
// time. The weights it must read are 1.22G parameters at four bits plus
// scales and biases, about 0.67 GB a token.
//
// # The roofline this was measured against was WRONG, and the record is fixed
//
// An earlier version of this note put the machine's peak at 120 GB/s -- the
// base M4's -- and concluded from ~93 GB/s that the kernel was at its
// roofline and worth a few percent at most. The adapter is an M4 PRO, which
// moves 273 GB/s. Anything that reasoned from "we are at the roofline" was
// reasoning from the wrong number.
//
// What replaces it, from `how_long_a_decodes_kernels_take` at this model's own
// shapes. `wgs` is the workgroup count, which is the output height over eight;
// `work` is the dispatch time less the 0.009 ms an 8x64 FLOOR dispatch costs,
// which is what an empty kernel costs and is not this file's to spend:
//
//   shape                 wgs      ms    work      GB/s of work
//   512x2048    k/v        64   0.013   0.004      147
//   2048x2048   q/o       256   0.025   0.016      147
//   8192x2048   gate/up  1024   0.063   0.054      175
//   2048x8192   down      256   0.047   0.038      248
//   128256x2048 head    16032   0.793   0.784      188
//
// Everything but the head fits an M4 Pro's system cache when dispatched two
// hundred times over, so 248 is a cache number; the head at 147.8 MB is the
// only row that is honestly DRAM, and it reads at 188 GB/s -- 69% of peak.
//
// So THE BODY IS NOT THE PROBLEM at any shape. What the table actually shows
// is the 0.009 ms floor: it is a third of a k/v projection and 36% of a q or
// o. A token fires this kernel 113 times and 115 other kernels besides, and
// 228 x 0.009 ms is 2.0 ms of a 8.4 ms token -- a quarter of the GPU, spent
// on dispatch and not on arithmetic. The lever that is left is FEWER AND
// BIGGER dispatches, which is fusion in the authored trace, not WGSL.
//
// Per eight codes and four rows -- 32 multiply-adds -- the fast path above
// spends 32 fused multiply-adds and about as many other instructions again:
// eight `unpack4x8unorm`, eight multiplies by 255, eight bf16 widenings, and
// EIGHT `xsum` ADDS.
//
// That last eighth is the one that is not inherent. `xsum` is a sum of
// ACTIVATIONS over the lane's k block; it does not depend on the row, and yet
// every one of the 512 four-row groups in a 2048-wide projection recomputes
// the same value from the same words. A pre-pass over `x` writing one partial
// sum per `PIE_QMV_VPT` block -- 128 floats for a 2048-wide vector -- would
// turn eight adds into one load here, and summing each block in the order this
// loop does keeps the answer bit-identical.
//
// It is not done because it is not kernel-local: it needs a routine, a buffer
// the driver stages per fire, and a place in the authored trace. Recorded with
// its size (about an eighth of the arithmetic above the transfer, so a few
// percent of a token) so that it is weighed rather than guessed at.
//
// Things that were tried against this loop and MEASURED WORSE, so that they
// are not tried again: one word per lane and four words per lane
// (`PIE_QMV_VPT`), `extractBits` instead of shift-and-mask (naga emits
// clamping guards), folding the `255.0` into the scale (a wash, and it changes
// the rounding), and the `unpack4x8unorm` path in `block_dot` below, where the
// extra live registers cost more than the instructions save.
//
// THE NIBBLE PATH IN `block_dot` WAS TRIED A SECOND TIME, structured the way
// `block_dot2` is -- unpack the eight weight vectors once, then walk the four
// activation rows ONE AT A TIME, so no more than one accumulator and eight
// activation floats are live beside them. That is the exact discipline that
// made `block_dot2` a 26% win at two rows, and at four rows it loses anyway:
//
//   streams        1       4       8      16
//   before       114.0   194.8   250.1   217.5 tok/s
//   after        109.8   137.7   167.1   187.5 tok/s
//
// Two rows is the shape where the unpack pays and four is not, so the entry
// above stands and the arm split is the answer, not a rewrite of `block_dot`.
//
// AND NOTE THE ONE-STREAM COLUMN. `mt == 1` does not go through `block_dot`
// at all, and it still fell 114.0 -> 109.8. The arms share an entrypoint, so
// the register allocation is the WORST arm's, and a fat arm taxes every fire
// that never enters it. That is the reason `block_dot2` is written lean, and
// it is the first thing to weigh before adding a third arm here.
//
// 64 K-LANES INSTEAD OF 32 was tried a second time, on the theory that a
// 512-row plane gives only 64 workgroups and 4096 invocations for a whole GPU,
// and that the table above was reading occupancy. It is not. Doubling the x
// extent to `@workgroup_size(64, 2, 1)` -- same grid, twice the invocations,
// one more level of tree, 8 KiB of partials instead of 4 -- measured worse at
// EVERY shape, including the head, where occupancy was never in question:
//
//   512x2048     45 ->  35 GB/s      8192x2048    149 -> 123 GB/s
//   2048x2048    96 ->  83 GB/s    128256x2048    186 -> 146 GB/s
//
// (Whole-dispatch figures, floor included, which is why they are lower than
// the `work` column above.) A wider workgroup buys memory requests in flight
// and pays for them in resident workgroups and reduction depth, and on this
// adapter the second is dearer. The first attempt at this is recorded in the
// list above; it is repeated here with numbers so the third attempt does not
// happen.

// One lane's block, for four output columns AND `PIE_MT` activation rows.
//
// The four sums ride in named fields rather than an array for the reason the
// four columns ride in a `vec4`: an array indexed by a loop variable is scratch
// memory in every backend naga targets, and the k loop is where it would live.
struct BlockM {
    a0: vec4<f32>,
    a1: vec4<f32>,
    a2: vec4<f32>,
    a3: vec4<f32>,
}

// One lane's block for four output columns and EXACTLY TWO activation rows.
//
// # Why a second special case, when `block_dot` already handles two
//
// It handles two by computing four. `reduce_store` clamps `vecs` to the last
// live row, so a two-row fire runs row 1 three times and discards two of
// them -- and it takes the slow unpack besides, because only `block_dot1`
// has the nibble path. Measured on an M4 Pro, Llama-3.2-1B, two concurrent
// conversations: 12.99 ms a step against 7.52 ms for one conversation, where
// the weights read is the same read. Four rows of arithmetic on three
// instructions a code is exactly the 1.73x that is.
//
// # What this arm bought
//
// M4 Pro, Llama-3.2-1B q4, both sides built and measured back to back on the
// same thermal state (the machine drifted a few percent warmer across the
// day, so only same-sitting pairs mean anything):
//
// ```text
//                          before    after
//   2 streams, aggregate   133.7     168.6 tok/s    +26%
//   2 streams, per-stream   68.7      90.2 tok/s
//   8 streams, aggregate      -      205.2 tok/s
//   1 stream,  tg128       111.1     113.9 tok/s    unchanged (mt == 1)
//   1 stream,  pp512       900.7     902.2 tok/s    unchanged
// ```
//
// The one-row and prefill paths do not come through here and did not move,
// which is the check that the extra arm cost the entrypoint no registers.
//
// Two rows is the shape a SERVED deployment actually decodes in -- the
// second conversation is the common case, not the fourth -- and it is the
// one shape where the nibble path is affordable. The list at the foot of
// this file records `unpack4x8unorm` in `block_dot` as measured worse, and
// that stands: the FOUR-row form keeps four accumulators and four rows of
// activations live across the unpack, and the register pressure costs more
// than the instructions save. This form keeps two accumulators and walks one
// row's activations at a time, so the eight unpacked vectors are the only
// thing held across.
//
// The order the products reach each accumulator is `block_dot1`'s, k
// ascending, so a row summed here and the same row summed alone agree bit
// for bit.
struct BlockM2 {
    a0: vec4<f32>,
    a1: vec4<f32>,
}

fn block_dot2(rows: vec4<u32>, vecs: vec2<u32>, k0: u32) -> BlockM2 {
    let n = u32(params.in_vec_size);
    let cpw = u32(PIE_CODES_PER_WORD);
    let mask = (1u << u32(PIE_BITS)) - 1u;
    let g = rows * (n / u32(PIE_GROUP)) + vec4<u32>(k0 / u32(PIE_GROUP));
    let s = vec4<f32>(
        qmv_bf16(scales[g.x >> 1u], g.x),
        qmv_bf16(scales[g.y >> 1u], g.y),
        qmv_bf16(scales[g.z >> 1u], g.z),
        qmv_bf16(scales[g.w >> 1u], g.w),
    );
    let b = vec4<f32>(
        qmv_bf16(biases[g.x >> 1u], g.x),
        qmv_bf16(biases[g.y >> 1u], g.y),
        qmv_bf16(biases[g.z >> 1u], g.z),
        qmv_bf16(biases[g.w >> 1u], g.w),
    );
    let wbase = rows * (n / cpw);
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var xsum = vec2<f32>(0.0);
    let whole = k0 + u32(PIE_QMV_VPT) <= n;
    for (var jw = 0u; jw < u32(PIE_QMV_VPT) / u32(PIE_CODES_PER_WORD); jw = jw + 1u) {
        let base = k0 + jw * u32(PIE_CODES_PER_WORD);
        if base >= n {
            break;
        }
        let at = wbase + vec4<u32>(base / u32(PIE_CODES_PER_WORD));
        let word = vec4<u32>(w[at.x], w[at.y], w[at.z], w[at.w]);
        if PIE_NIBBLE_FAST && whole {
            let lo = vec4<u32>(0x0f0f0f0fu);
            let e0 = unpack4x8unorm(word.x & lo.x) * 255.0;
            let e1 = unpack4x8unorm(word.y & lo.y) * 255.0;
            let e2 = unpack4x8unorm(word.z & lo.z) * 255.0;
            let e3 = unpack4x8unorm(word.w & lo.w) * 255.0;
            let sh = (word >> vec4<u32>(4u)) & lo;
            let o0 = unpack4x8unorm(sh.x) * 255.0;
            let o1 = unpack4x8unorm(sh.y) * 255.0;
            let o2 = unpack4x8unorm(sh.z) * 255.0;
            let o3 = unpack4x8unorm(sh.w) * 255.0;
            // ONE ROW AT A TIME across the unpack, which is the whole
            // difference from the four-row attempt: eight activation floats
            // are live here, not thirty-two.
            let p0 = x_word(vecs.x, base);
            let p1 = x_word(vecs.x, base + 2u);
            let p2 = x_word(vecs.x, base + 4u);
            let p3 = x_word(vecs.x, base + 6u);
            let u0 = pie_bf16_to_f32(p0 & 0xffffu);
            let u1 = pie_bf16_to_f32(p0 >> 16u);
            let u2 = pie_bf16_to_f32(p1 & 0xffffu);
            let u3 = pie_bf16_to_f32(p1 >> 16u);
            let u4 = pie_bf16_to_f32(p2 & 0xffffu);
            let u5 = pie_bf16_to_f32(p2 >> 16u);
            let u6 = pie_bf16_to_f32(p3 & 0xffffu);
            let u7 = pie_bf16_to_f32(p3 >> 16u);
            xsum.x = xsum.x + u0 + u1 + u2 + u3 + u4 + u5 + u6 + u7;
            a0 = a0 + u0 * vec4<f32>(e0.x, e1.x, e2.x, e3.x);
            a0 = a0 + u1 * vec4<f32>(o0.x, o1.x, o2.x, o3.x);
            a0 = a0 + u2 * vec4<f32>(e0.y, e1.y, e2.y, e3.y);
            a0 = a0 + u3 * vec4<f32>(o0.y, o1.y, o2.y, o3.y);
            a0 = a0 + u4 * vec4<f32>(e0.z, e1.z, e2.z, e3.z);
            a0 = a0 + u5 * vec4<f32>(o0.z, o1.z, o2.z, o3.z);
            a0 = a0 + u6 * vec4<f32>(e0.w, e1.w, e2.w, e3.w);
            a0 = a0 + u7 * vec4<f32>(o0.w, o1.w, o2.w, o3.w);
            let q0 = x_word(vecs.y, base);
            let q1 = x_word(vecs.y, base + 2u);
            let q2 = x_word(vecs.y, base + 4u);
            let q3 = x_word(vecs.y, base + 6u);
            let v0 = pie_bf16_to_f32(q0 & 0xffffu);
            let v1 = pie_bf16_to_f32(q0 >> 16u);
            let v2 = pie_bf16_to_f32(q1 & 0xffffu);
            let v3 = pie_bf16_to_f32(q1 >> 16u);
            let v4 = pie_bf16_to_f32(q2 & 0xffffu);
            let v5 = pie_bf16_to_f32(q2 >> 16u);
            let v6 = pie_bf16_to_f32(q3 & 0xffffu);
            let v7 = pie_bf16_to_f32(q3 >> 16u);
            xsum.y = xsum.y + v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
            a1 = a1 + v0 * vec4<f32>(e0.x, e1.x, e2.x, e3.x);
            a1 = a1 + v1 * vec4<f32>(o0.x, o1.x, o2.x, o3.x);
            a1 = a1 + v2 * vec4<f32>(e0.y, e1.y, e2.y, e3.y);
            a1 = a1 + v3 * vec4<f32>(o0.y, o1.y, o2.y, o3.y);
            a1 = a1 + v4 * vec4<f32>(e0.z, e1.z, e2.z, e3.z);
            a1 = a1 + v5 * vec4<f32>(o0.z, o1.z, o2.z, o3.z);
            a1 = a1 + v6 * vec4<f32>(e0.w, e1.w, e2.w, e3.w);
            a1 = a1 + v7 * vec4<f32>(o0.w, o1.w, o2.w, o3.w);
            continue;
        }
        for (var c = 0u; c < u32(PIE_CODES_PER_WORD); c = c + 2u) {
            if !whole && base + c >= n {
                break;
            }
            let k = base + c;
            let xw = vec2<u32>(x_word(vecs.x, k), x_word(vecs.y, k));
            let xv0 = vec2<f32>(
                pie_bf16_to_f32(xw.x & 0xffffu),
                pie_bf16_to_f32(xw.y & 0xffffu),
            );
            xsum = xsum + xv0;
            let code0 = vec4<f32>((word >> vec4<u32>(u32(PIE_BITS) * c)) & vec4<u32>(mask));
            a0 = a0 + xv0.x * code0;
            a1 = a1 + xv0.y * code0;
            if !whole && k + 1u >= n {
                break;
            }
            let xv1 = vec2<f32>(
                pie_bf16_to_f32(xw.x >> 16u),
                pie_bf16_to_f32(xw.y >> 16u),
            );
            xsum = xsum + xv1;
            let code1 = vec4<f32>((word >> vec4<u32>(u32(PIE_BITS) * (c + 1u))) & vec4<u32>(mask));
            a0 = a0 + xv1.x * code1;
            a1 = a1 + xv1.y * code1;
        }
    }
    return BlockM2(
        s * a0 + b * vec4<f32>(xsum.x),
        s * a1 + b * vec4<f32>(xsum.y),
    );
}

fn block_dot(rows: vec4<u32>, vecs: vec4<u32>, k0: u32) -> BlockM {
    let n = u32(params.in_vec_size);
    let cpw = u32(PIE_CODES_PER_WORD);
    let mask = (1u << u32(PIE_BITS)) - 1u;
    let g = rows * (n / u32(PIE_GROUP)) + vec4<u32>(k0 / u32(PIE_GROUP));
    let s = vec4<f32>(
        qmv_bf16(scales[g.x >> 1u], g.x),
        qmv_bf16(scales[g.y >> 1u], g.y),
        qmv_bf16(scales[g.z >> 1u], g.z),
        qmv_bf16(scales[g.w >> 1u], g.w),
    );
    let b = vec4<f32>(
        qmv_bf16(biases[g.x >> 1u], g.x),
        qmv_bf16(biases[g.y >> 1u], g.y),
        qmv_bf16(biases[g.z >> 1u], g.z),
        qmv_bf16(biases[g.w >> 1u], g.w),
    );
    let wbase = rows * (n / cpw);
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var xsum = vec4<f32>(0.0);
    // CONST BOUNDS, both of them. Written with a `let` copy of the same
    // constant this ran 22% SLOWER than the loop it replaced, measured
    // isolated at [4096, 4096] -- a bound naga cannot fold is a loop it cannot
    // unroll, and an unrolled sixteen-step inner loop is most of what this
    // kernel is.
    //
    // The tail is hoisted out of the inner loop for the same reason: a block
    // wholly inside K takes the unguarded path, and only the last one per row
    // pays a comparison per value.
    let whole = k0 + u32(PIE_QMV_VPT) <= n;
    for (var jw = 0u; jw < u32(PIE_QMV_VPT) / u32(PIE_CODES_PER_WORD); jw = jw + 1u) {
        let base = k0 + jw * u32(PIE_CODES_PER_WORD);
        // The K tail, guarded on the WORD as well as the value: a row is
        // `n / cpw` words and reading one past the last row's last word is off
        // the end of the buffer, not merely off the end of a row.
        if base >= n {
            break;
        }
        let at = wbase + vec4<u32>(base / u32(PIE_CODES_PER_WORD));
        let word = vec4<u32>(w[at.x], w[at.y], w[at.z], w[at.w]);
        // TWO CODES A TRIP. Each activation word holds `k` and `k + 1`, so
        // walking k one at a time loaded every one of them twice -- and here
        // that is FOUR redundant loads a value, one per activation row. `base`
        // is a multiple of `PIE_CODES_PER_WORD` and `in_vec_size` is even, so
        // `k` is even and the low half is `k`. Same products, same order.
        for (var c = 0u; c < u32(PIE_CODES_PER_WORD); c = c + 2u) {
            if !whole && base + c >= n {
                break;
            }
            let k = base + c;
            // The four activation rows this workgroup carries, at one k. The
            // word above is read ONCE for all four, which is the saving.
            let xw = vec4<u32>(
                x_word(vecs.x, k),
                x_word(vecs.y, k),
                x_word(vecs.z, k),
                x_word(vecs.w, k),
            );
            let xv0 = vec4<f32>(
                pie_bf16_to_f32(xw.x & 0xffffu),
                pie_bf16_to_f32(xw.y & 0xffffu),
                pie_bf16_to_f32(xw.z & 0xffffu),
                pie_bf16_to_f32(xw.w & 0xffffu),
            );
            xsum = xsum + xv0;
            let code0 = vec4<f32>((word >> vec4<u32>(u32(PIE_BITS) * c)) & vec4<u32>(mask));
            a0 = a0 + xv0.x * code0;
            a1 = a1 + xv0.y * code0;
            a2 = a2 + xv0.z * code0;
            a3 = a3 + xv0.w * code0;
            if !whole && k + 1u >= n {
                break;
            }
            let xv1 = vec4<f32>(
                pie_bf16_to_f32(xw.x >> 16u),
                pie_bf16_to_f32(xw.y >> 16u),
                pie_bf16_to_f32(xw.z >> 16u),
                pie_bf16_to_f32(xw.w >> 16u),
            );
            xsum = xsum + xv1;
            let code1 = vec4<f32>((word >> vec4<u32>(u32(PIE_BITS) * (c + 1u))) & vec4<u32>(mask));
            a0 = a0 + xv1.x * code1;
            a1 = a1 + xv1.y * code1;
            a2 = a2 + xv1.z * code1;
            a3 = a3 + xv1.w * code1;
        }
    }
    // `sum_i x_i * (s * code_i + b)` regrouped: the bias rides the plain sum of
    // the activations, which is why `xsum` is worth carrying. Per activation
    // row, and in the same order the one-row form summed it.
    return BlockM(
        s * a0 + b * vec4<f32>(xsum.x),
        s * a1 + b * vec4<f32>(xsum.y),
        s * a2 + b * vec4<f32>(xsum.z),
        s * a3 + b * vec4<f32>(xsum.w),
    );
}

//#if !defined(PIE_WIDE_STRIDED)
fn reduce_store(lid: u32, ly: u32, wg: vec3<u32>) {
    let out0 = wg.y * 4u;
    let vec0 = wg.x * PIE_MT;

    // How many activation rows the fire actually bound.
    //
    // `affine_qmv_fast` states two scalars -- `in_vec_size` and
    // `out_vec_size` -- and neither is the batch. It does not have to be:
    // `binding::extent` binds an arena operand as exactly the rectangle the
    // launch covers, `rows * width * bytes`, so the bound length of `x` IS the
    // row count times the width, and `arrayLength` reads it back. Deriving it
    // here rather than adding a scalar keeps the row's operand list and every
    // other backend's uniform block alone.
    let m = max((arrayLength(&x) * 2u) / u32(params.in_vec_size), 1u);

    // Every lane runs all four columns and all `PIE_MT` rows and stores all
    // the partials, out of range or not, because the `barrier()` below is
    // next: an early return in front of a workgroup barrier is a HANG in WGSL,
    // not a wrong number. Out-of-range indices are CLAMPED rather than skipped
    // -- it keeps the lanes on the same trip count and the values are dropped
    // at the store, which already asks.
    let last = max(u32(params.out_vec_size), 1u) - 1u;
    let rows = min(
        vec4<u32>(out0, out0 + 1u, out0 + 2u, out0 + 3u),
        vec4<u32>(last),
    );
    let vecs = min(
        vec4<u32>(vec0, vec0 + 1u, vec0 + 2u, vec0 + 3u),
        vec4<u32>(m - 1u),
    );
    // Rows this workgroup actually has, which is `PIE_MT` everywhere but the
    // last group of a ragged batch -- and ONE for every decode, where the fire
    // is a single row. Workgroup-uniform, so the loops it bounds may hold a
    // barrier; the k loop above cannot use it, because an unrolled `vec4` is
    // what makes that loop fast.
    // `max(m, vec0) - vec0` and not `m - vec0`: the subtraction is unsigned,
    // and a grid that dispatches MORE x groups than the batch needs -- which
    // is legal, and which every launcher in this tree used to do before
    // `quarters()` -- would wrap to four billion and walk the whole of both
    // loops below off the end. Zero is the right answer for a workgroup with
    // no rows, and it leaves every barrier here in uniform control flow.
    let mt = min(PIE_MT, max(m, vec0) - vec0);
    var t0 = vec4<f32>(0.0);
    var t1 = vec4<f32>(0.0);
    var t2 = vec4<f32>(0.0);
    var t3 = vec4<f32>(0.0);
    var k0 = lid * u32(PIE_QMV_VPT);
    // WORKGROUP-UNIFORM, and there is no barrier in either arm -- the one this
    // function has is below, after both have joined. The ragged tail of a
    // batch can still be a single row, and it takes the cheap body too.
    if mt == 1u {
        while k0 < u32(params.in_vec_size) {
            t0 = t0 + block_dot1(rows, vecs.x, k0);
            k0 = k0 + u32(PIE_QMV_VPT) * 32u;
        }
    } else if mt == 2u {
        while k0 < u32(params.in_vec_size) {
            let part = block_dot2(rows, vecs.xy, k0);
            t0 = t0 + part.a0;
            t1 = t1 + part.a1;
            k0 = k0 + u32(PIE_QMV_VPT) * 32u;
        }
    } else {
        while k0 < u32(params.in_vec_size) {
            let part = block_dot(rows, vecs, k0);
            t0 = t0 + part.a0;
            t1 = t1 + part.a1;
            t2 = t2 + part.a2;
            t3 = t3 + part.a3;
            k0 = k0 + u32(PIE_QMV_VPT) * 32u;
        }
    }
    // Slot `(r, mi)` of this y-slot is `((ly * 4 + r) * PIE_MT + mi)`. All
    // sixteen are written whether or not the group has four live rows: the
    // spare ones hold the zeros they were accumulated from, and the tree and
    // the store below walk only `mt` of them.
    let slot0 = 0u;
    qmv_partials[(slot0 + 0u * PIE_MT + 0u) * 32u + lid] = t0.x;
    qmv_partials[(slot0 + 1u * PIE_MT + 0u) * 32u + lid] = t0.y;
    qmv_partials[(slot0 + 2u * PIE_MT + 0u) * 32u + lid] = t0.z;
    qmv_partials[(slot0 + 3u * PIE_MT + 0u) * 32u + lid] = t0.w;
    // The other three rows' slots, written only by a group that HAS them. A
    // decode writes four partials here, which is what it wrote before `PIE_MT`
    // existed; writing all sixteen cost it 3 tok/s of 64 and bought nothing.
    // ONE SLOT INDEX PER `mi`, GUARDED BY `mt` AND NOT BY `mt > 1` ALONE.
    // These used to be twelve stores under a single `if mt > 1u`, which is
    // right only while `PIE_MT` is 4: slot `(r, mi)` is `r * PIE_MT + mi`, so
    // an `mi` at or past `PIE_MT` is not out of bounds -- it is ROW `r + 1`'s
    // slot, written with row `r`'s partial. At `PIE_MT = 2` that silently
    // corrupted every other output column and cost three gpu tests and an
    // afternoon.
    if mt > 1u {
        qmv_partials[(slot0 + 0u * PIE_MT + 1u) * 32u + lid] = t1.x;
        qmv_partials[(slot0 + 1u * PIE_MT + 1u) * 32u + lid] = t1.y;
        qmv_partials[(slot0 + 2u * PIE_MT + 1u) * 32u + lid] = t1.z;
        qmv_partials[(slot0 + 3u * PIE_MT + 1u) * 32u + lid] = t1.w;
    }
    if mt > 2u {
        qmv_partials[(slot0 + 0u * PIE_MT + 2u) * 32u + lid] = t2.x;
        qmv_partials[(slot0 + 1u * PIE_MT + 2u) * 32u + lid] = t2.y;
        qmv_partials[(slot0 + 2u * PIE_MT + 2u) * 32u + lid] = t2.z;
        qmv_partials[(slot0 + 3u * PIE_MT + 2u) * 32u + lid] = t2.w;
    }
    if mt > 3u {
        qmv_partials[(slot0 + 0u * PIE_MT + 3u) * 32u + lid] = t3.x;
        qmv_partials[(slot0 + 1u * PIE_MT + 3u) * 32u + lid] = t3.y;
        qmv_partials[(slot0 + 2u * PIE_MT + 3u) * 32u + lid] = t3.z;
        qmv_partials[(slot0 + 3u * PIE_MT + 3u) * 32u + lid] = t3.w;
    }
    workgroupBarrier();

    // A TREE, where this used to be lane zero adding thirty-two partials for
    // each of four rows -- a hundred and twenty-eight serial adds with
    // thirty-one lanes of the y-slot watching. Five halvings do the same work
    // in five steps.
    //
    // The barrier is OUTSIDE the `if`, which is not a style choice: WGSL
    // requires `workgroupBarrier` in uniform control flow, and a barrier
    // reached by half a workgroup is undefined rather than slow.
    for (var half = 16u; half > 0u; half = half >> 1u) {
        if lid < half {
            for (var r = 0u; r < 4u; r = r + 1u) {
                for (var mi = 0u; mi < mt; mi = mi + 1u) {
                    let at = (slot0 + r * PIE_MT + mi) * 32u + lid;
                    qmv_partials[at] = qmv_partials[at] + qmv_partials[at + half];
                }
            }
        }
        workgroupBarrier();
    }

    if lid == 0u {
        for (var r = 0u; r < 4u; r = r + 1u) {
            let row = out0 + r;
            if row >= u32(params.out_vec_size) {
                continue;
            }
            for (var mi = 0u; mi < mt; mi = mi + 1u) {
                let vec_ = vec0 + mi;
                let sum0 = qmv_partials[(slot0 + r * PIE_MT + mi) * 32u];
                var sum = sum0;
//#if defined(PIE_BIAS)
                // Unconditional, where the GLSL asked `if (tail)`: every
                // variant that defines PIE_BIAS also defines PIE_TAIL, so the
                // runtime flag could only ever be true.
                sum = sum + qmv_bf16(extra[row >> 1u], row);
//#endif
//#if defined(PIE_RESIDUAL)
                let at = vec_ * u32(params.out_vec_size) + row;
                // Rounded to bf16 BEFORE the add, which is not an accident:
                // it makes this bit-identical to the two-kernel path (project,
                // then `residual_add`) that the fused variant replaces.
                let q = pie_bf16_to_f32(pie_f32_to_bf16(sum));
                store_y(at, q + qmv_bf16(extra[at >> 1u], at));
//#else
                store_y(vec_ * u32(params.out_vec_size) + row, sum);
//#endif
            }
        }
    }
}
//#endif

//#if defined(PIE_WIDE_STRIDED)
// One row per lane over the whole of K, four lanes per y-slot. `PIE_K_LANES` is
// in the entrypoint name and read by nothing: Metal splits K over eight lanes
// and finishes with `simd_shuffle_down`, and neither Vulkan's port nor this one
// does -- the name records the shape the launcher picked, and the body is a
// serial K loop that gets the same answer with 4 of 32 lanes busy. There is no
// barrier in here, which is why the early returns below are safe.
fn wide_strided(lid: vec3<u32>, wg: vec3<u32>) {
    let row_slot = lid.x & 3u;
    if lid.x >= 4u {
        return;
    }
    let row = wg.y * 4u + row_slot;
    let vec0 = wg.x * u32(PIE_VEC);
    if row >= u32(params.out_vec_size) {
        return;
    }
    for (var v = 0u; v < u32(PIE_VEC); v = v + 1u) {
        let vec_ = vec0 + v;
        if vec_ < u32(params.m) {
            var acc = 0.0;
            for (var k = 0u; k < u32(params.in_vec_size); k = k + 1u) {
                acc = acc + x_at(vec_, k) * affine_value(row, k);
            }
            store_y(vec_ * u32(params.row_stride) + row, acc);
        }
    }
}
//#endif

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
//#if defined(PIE_WIDE_STRIDED)
    wide_strided(lid, wg);
//#else
    reduce_store(lid.x, lid.y, wg);
//#endif
}

// pie:instantiate affine_qmv_fast_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4
// pie:instantiate affine_qmv_fast_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8
// pie:instantiate affine_qmv_fast_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4
// pie:instantiate affine_qmv_fast_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8
// pie:instantiate affine_qmv_fast_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmv_fast_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_tail_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_TAIL=1
// pie:instantiate affine_qmv_tail_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_TAIL=1
// pie:instantiate affine_qmv_tail_bias_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_BIAS=1 PIE_TAIL=1
// pie:instantiate affine_qmv_tail_bias_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_BIAS=1 PIE_TAIL=1
// pie:instantiate affine_qmv_wide_strided_bfloat16_gs_64_b_4_v_4_kl_8 PIE_GROUP=64 PIE_BITS=4 PIE_WIDE_STRIDED=1 PIE_VEC=4 PIE_K_LANES=8
// pie:instantiate affine_qmv_wide_strided_bfloat16_gs_64_b_8_v_4_kl_8 PIE_GROUP=64 PIE_BITS=8 PIE_WIDE_STRIDED=1 PIE_VEC=4 PIE_K_LANES=8
