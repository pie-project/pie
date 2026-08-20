// Affine tiled GEMM projections, ported from `quant/qmm_t.comp` (itself from
// `quant/qmm_t.metal`).
//
// One body, three kernels: the GEMM itself, the `PIE_CAST_INPUT` staging pass
// that rounds an activation to fp16 once instead of once per output tile, and
// the `PIE_REDUCE` second half of a split-K projection. They share nothing but
// the file -- each arm declares its own bindings, its own uniform block and its
// own `main` -- and they are one file because the Metal and Vulkan trees put
// them in one file, so a reader diffing the three backends finds them together.
//
// ## What this is NOT
//
// Metal drives Apple's simdgroup matrix unit; Vulkan adds a `@coopmat` tier on
// `VK_KHR_cooperative_matrix`. WebGPU's equivalent is
// `wgpu::Features::SUBGROUP_MATRIX`, which is deliberately not in this crate's
// tier list (`src/capability.rs`: Baseline / Fp16 / Subgroup, where Subgroup is
// `subgroupAdd`-class reductions only). So there is exactly one tier here, and
// `_fp16_precast` means what it means in the Vulkan baseline: the staged
// activation carries fp16 STORAGE precision and the multiply is fp32. It is an
// emulation of the Metal path's numerics, not a half-precision matrix unit, and
// the distinction is intentional.
//
// ## What is new here
//
// Vulkan's baseline gives each invocation one output element and a full K loop,
// re-fetching `x` and the packed weight from global memory for every element of
// the tile. This body stages a BM x PIE_BK slab of the activation and a
// PIE_BK x BN slab of the dequantised weight into workgroup memory first, so
// each fetch is amortised over BN and BM lanes respectively -- the same shape
// the coopmat tier stages for its matrix unit, minus the matrix unit. The
// accumulation is fp32 either way, and `affine_quad`/`load_x` are called
// exactly as Vulkan calls them, so the arithmetic is unchanged.
//
// The two barriers per K block are the hazard: `workgroupBarrier()` must be
// reached by every invocation in the workgroup, so NOTHING in the K loop may
// return early and every bound is applied to a VALUE or to a STORE, never to
// whether an invocation arrives. An early return in front of a barrier is a
// hang, not a wrong number.

//#include "common/bf16.inc.wgsl"
// The codec is only meaningful to the GEMM arm, and the other two arms define
// neither `PIE_GROUP` nor `PIE_BITS` -- splicing it there would fail to compile
// on the include's own first line rather than on anything they wrote.
//#if !defined(PIE_REDUCE) && !defined(PIE_CAST_INPUT)
//#include "common/affine.inc.wgsl"
//#endif

// ── Bindings ────────────────────────────────────────────────────────────────
//
// Every number below is derived from the ROW in `src/quant.rs` -- buffer-kinded
// operands numbered densely from zero in the row's order, scalars moved to the
// uniform block -- and NONE is transcribed from Metal. `.wiki/new-driver/vulkan.md`
// §3 is the record of what transcribing costs: Metal numbers scalars in the same
// run as buffers, so its `residual` is buffer 7 where the row puts it at 5, and
// across 54 entrypoints of this file's Vulkan sibling that 7 was a descriptor
// the shell never wrote.
//
// For the rows the table states no operands for (everything but `affine_qmm_t`
// and `affine_qmm_t_residual`), the order is the lowered plan's, which is
// `qmm_t.metal`'s parameter list; applying the same two-run rule to that list
// reproduces the two stated rows exactly, which is the only evidence available
// that it is the right list. A variant declares only what it READS, so the sets
// have holes where a flag is off -- legitimate, per §13 -- but never a binding
// past the row's buffer count.

//#if defined(PIE_CAST_INPUT)
@group(0) @binding(0) var<storage, read_write> cast_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> half_out: array<atomic<u32>>;
struct Params {
//#if defined(PIE_STRIDED)
    k: i32,
    row_stride: i32,
//#else
    count: i32,
//#endif
}
//#elif defined(PIE_REDUCE)
@group(0) @binding(0) var<storage, read_write> reduce_y: array<atomic<u32>>;
//#if defined(PIE_PARTIAL_F32)
@group(0) @binding(1) var<storage, read_write> partial_f32: array<f32>;
//#else
@group(0) @binding(1) var<storage, read_write> partial_bf16: array<u32>;
//#endif
// `n`, the partition stride and the partition COUNT. Metal's `split_k` is
// buffer 11 and not 9 on purpose -- 9 is the GEMM half's `k_partition_size`,
// the partition LENGTH -- and the two halves of a split projection share one
// argument table, which is why that distinction has a comment there too.
struct Params {
    n: i32,
    split_k_partition_stride: i32,
    split_k: i32,
}
//#else
@group(0) @binding(0) var<storage, read_write> w: array<u32>;
@group(0) @binding(1) var<storage, read_write> scales: array<u32>;
@group(0) @binding(2) var<storage, read_write> biases: array<u32>;
// The activation. `_fp16_precast` does not have one: its input is the fp16
// staging buffer `half_in`, which the plan passes LAST (Metal buffer 12), so
// every later binding in a precast variant shifts down by one. That shift is
// the whole reason these are written out arm by arm instead of once.
//#if !defined(PIE_FP16_PRECAST)
@group(0) @binding(3) var<storage, read_write> x: array<u32>;
//#endif
// The result. Split-K writes its `[split_k, M, N]` partial through the same
// operand slot -- Metal gives it buffer 8 and the plain output buffer 4, but
// no variant has both, so densely numbered they are the same slot.
//#if defined(PIE_FP16_PRECAST) && defined(PIE_OUT_F32)
@group(0) @binding(3) var<storage, read_write> out_f32: array<f32>;
//#elif defined(PIE_FP16_PRECAST)
@group(0) @binding(3) var<storage, read_write> out_: array<atomic<u32>>;
//#elif defined(PIE_OUT_F32)
@group(0) @binding(4) var<storage, read_write> out_f32: array<f32>;
//#else
@group(0) @binding(4) var<storage, read_write> out_: array<atomic<u32>>;
//#endif
// The bias vector and the residual block are the same operand slot: no variant
// has both, and a variant with neither declares nothing here at all -- which is
// what `affine_qmm_t`'s five-operand row requires. (The condition is in
// disjunctive form because the directive grammar has no parentheses and `&&`
// binds tighter than `||`.)
//#if defined(PIE_FP16_PRECAST) && defined(PIE_BIAS) || defined(PIE_FP16_PRECAST) && defined(PIE_RESIDUAL)
@group(0) @binding(4) var<storage, read_write> extra: array<u32>;
//#elif defined(PIE_BIAS) || defined(PIE_RESIDUAL)
@group(0) @binding(5) var<storage, read_write> extra: array<u32>;
//#endif
//#if defined(PIE_FP16_PRECAST) && defined(PIE_BIAS) || defined(PIE_FP16_PRECAST) && defined(PIE_RESIDUAL)
@group(0) @binding(5) var<storage, read_write> half_in: array<u32>;
//#elif defined(PIE_FP16_PRECAST)
@group(0) @binding(4) var<storage, read_write> half_in: array<u32>;
//#endif
// Variant-shaped, and it has to be: `uniform_layout()` sizes the buffer from
// the row's scalars, so a field the row does not state is a field the shell
// never writes. The Vulkan sibling declared `row_stride` and `split_k` for the
// split-K GEMM; the plan passes neither to that half -- only `k_partition_size`
// and `split_k_partition_stride` -- and the body reads neither.
struct Params {
    k: i32,
    n: i32,
//#if defined(PIE_STRIDED)
    row_stride: i32,
//#endif
//#if defined(PIE_SPLITK)
    k_partition_size: i32,
    split_k_partition_stride: i32,
//#endif
    // THE ROW COUNT, and the reason `write_out` can bound its rows.
    //
    // Last on purpose: every other field's offset is what it was before this
    // one existed, so adding it could not move a field a variant already
    // reads. It is unconditional because the guard must hold for every
    // variant -- a form that omitted it would be the one form that still
    // overruns.
    m: i32,
}
//#endif

@group(1) @binding(0) var<uniform> params: Params;

// The `i`-th bf16 of a word already loaded. Not `pie_load_bf16`: that takes a
// `ptr<storage, ...>`, which naga 30 parses and then refuses to validate
// ("which can't be passed into functions"), so a module using it never reaches
// a device. The conversion is still the include's.
fn qmm_bf16(word: u32, i: u32) -> f32 {
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

//#if defined(PIE_FP16_PRECAST) || defined(PIE_CAST_INPUT)
// ── fp16 as a storage format, with no device capability asked for ───────────
//
// WGSL has `pack2x16float`/`unpack2x16float` and this was written with them
// first. naga refuses both without `Capabilities::SHADER_FLOAT16_IN_FLOAT32`,
// which wgpu grants only from `DownlevelFlags::SHADER_F16_IN_F32` -- a flag
// whose own documentation says Vulkan on Mesa does not set it when
// `Features::SHADER_F16` is absent. That would leave these 41 entrypoints
// failing at `create_shader_module` on a plausible Linux target while all 262
// others built, and the Baseline tier's whole promise is that it asks for
// nothing. So the conversion is integer arithmetic, and it is exactly
// round-to-nearest-even -- which is what `half(x)` is on the Metal side these
// numbers have to match.
//
// Both are Fabian Giesen's. Before they were written down here the round trip
// over all 63488 finite half patterns, the tie at all 31744 midpoints between
// neighbouring halves, and ~2000 awkward values against an independent
// nearest-value search were checked in Rust; all three were exact.

// f16 bits -> f32. The shift moves sign+exponent+mantissa into f32's fields at
// once; the two corrections are for the exponent encodings that do not scale --
// 31 means infinity rather than 2^16, and 0 means a subnormal whose implicit
// leading 1 the float add below subtracts back off.
fn qmm_f16_to_f32(h: u32) -> f32 {
    let shifted_exp = 0x7c00u << 13u;
    var o = (h & 0x7fffu) << 13u;
    let exp_ = shifted_exp & o;
    o = o + ((127u - 15u) << 23u);
    if exp_ == shifted_exp {
        o = o + ((128u - 16u) << 23u);
    } else if exp_ == 0u {
        o = o + (1u << 23u);
        o = bitcast<u32>(bitcast<f32>(o) - bitcast<f32>(113u << 23u));
    }
    return bitcast<f32>(o | ((h & 0x8000u) << 16u));
}

// f32 -> f16 bits. `0xc8000000` is `(15 - 127) << 23` in two's complement, the
// exponent rebias; `+ 0xfff + mant_odd` is the round-to-nearest-EVEN, which a
// plain `+ 0x1000` would not be. Anything at or above 65520 becomes infinity
// because 65520 is the midpoint of the last representable step and its even
// neighbour IS infinity.
fn qmm_f32_to_f16(x: f32) -> u32 {
    var f = bitcast<u32>(x);
    let sign = f & 0x80000000u;
    f = f ^ sign;
    var o = 0u;
    if f >= (127u + 16u) << 23u {
        o = select(0x7c00u, 0x7e00u, f > (255u << 23u));
    } else if f < (113u << 23u) {
        // Subnormal, and zero with it: adding 0.5 forces the value's bits into
        // the mantissa field at the fixed exponent a half-subnormal has, so the
        // hardware's own rounding does the work.
        let magic = bitcast<f32>(126u << 23u);
        o = bitcast<u32>(bitcast<f32>(f) + magic) - bitcast<u32>(magic);
    } else {
        let mant_odd = (f >> 13u) & 1u;
        o = (f + 0xc8000000u + 0xfffu + mant_odd) >> 13u;
    }
    return o | (sign >> 16u);
}
//#endif

//#if !defined(PIE_OUT_F32)
// Write one 16-bit element of the output -- bf16 everywhere but the cast pass,
// which stages fp16; the packing hazard is the same either way and the bits
// arrive already encoded.
//
// The word at `i >> 1` holds elements `i & ~1` and `i | 1`, and this invocation
// owns only one of them: adjacent COLUMNS of an output tile belong to adjacent
// lanes, so every even/odd column pair is a word two lanes both want, and with
// an odd output row pitch the pairing shifts by one on every row so no lane
// assignment can fix it. A plain read-modify-write drops whichever half landed
// second. The CAS is device-scoped, which is the scope of the race -- the two
// lanes may be in different workgroups when the pitch is odd -- and it retries
// on the spurious failure `...Weak` is permitted.
//#if defined(PIE_CAST_INPUT)
fn store_half(i: u32, bits: u32) {
    let at = i >> 1u;
    let shift = (i & 1u) * 16u;
    let keep = ~(0xffffu << shift);
    let put = bits << shift;
    var old = atomicLoad(&half_out[at]);
    loop {
        let r = atomicCompareExchangeWeak(&half_out[at], old, (old & keep) | put);
        if r.exchanged {
            break;
        }
        old = r.old_value;
    }
}
//#elif defined(PIE_REDUCE)
fn store_half(i: u32, bits: u32) {
    let at = i >> 1u;
    let shift = (i & 1u) * 16u;
    let keep = ~(0xffffu << shift);
    let put = bits << shift;
    var old = atomicLoad(&reduce_y[at]);
    loop {
        let r = atomicCompareExchangeWeak(&reduce_y[at], old, (old & keep) | put);
        if r.exchanged {
            break;
        }
        old = r.old_value;
    }
}
//#else
fn store_half(i: u32, bits: u32) {
    let at = i >> 1u;
    let shift = (i & 1u) * 16u;
    let keep = ~(0xffffu << shift);
    let put = bits << shift;
    var old = atomicLoad(&out_[at]);
    loop {
        let r = atomicCompareExchangeWeak(&out_[at], old, (old & keep) | put);
        if r.exchanged {
            break;
        }
        old = r.old_value;
    }
}
//#endif
//#endif

//#if defined(PIE_CAST_INPUT)
// Stage the projection source as fp16 ONCE. Casting inside every output tile
// repeated the same conversion N/BN times -- 128 times for gate/up on a dense
// g64/b4 model -- and this pass removes that multiplier; it is not a numeric
// change, the tile body would round to the same fp16 either way.
//
// fp16 here is a STORAGE format and nothing more: `qmm_f32_to_f16` is integer
// arithmetic, every arithmetic use of the result widens back to f32, and no
// part of this asks the device for a half-precision anything.
@compute @workgroup_size(32, 2, 2)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
//#if defined(PIE_STRIDED)
    let col = gid.x;
    let row = gid.y;
    if col < u32(params.k) {
        let i = row * u32(params.row_stride) + col;
        store_half(i, qmm_f32_to_f16(qmm_bf16(cast_in[i >> 1u], i)));
    }
//#else
    // The grid is 2-D and the z dimension of the workgroup is not folded in, so
    // each element is visited twice (z = 0 and z = 1). A cast is idempotent and
    // the store writes the same bits both times, so the duplicate is waste and
    // not a race; the alternative is a flat index that disagrees with the
    // launcher's grid, which is a wrong answer.
    let idx = gid.x + gid.y * groups.x * 32u;
    if idx < u32(params.count) {
        store_half(idx, qmm_f32_to_f16(qmm_bf16(cast_in[idx >> 1u], idx)));
    }
//#endif
}
//#elif defined(PIE_REDUCE)
fn partial_at(at: u32) -> f32 {
//#if defined(PIE_PARTIAL_F32)
    return partial_f32[at];
//#else
    return qmm_bf16(partial_bf16[at >> 1u], at);
//#endif
}

// The second half of a split-K projection: sum the `split_k` slices the GEMM
// left at `[split_k, M, N]` and write the activation type. No barriers in here,
// which is why the early return is safe.
@compute @workgroup_size(32, 2, 2)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= u32(params.n) {
        return;
    }
    let base = row * u32(params.n) + col;
    var acc = 0.0;
    for (var s = 0u; s < u32(params.split_k); s = s + 1u) {
        acc = acc + partial_at(base + s * u32(params.split_k_partition_stride));
    }
    store_half(base, pie_f32_to_bf16(acc));
}
//#else

// ── The GEMM ────────────────────────────────────────────────────────────────

// K elements staged per pass. The workgroup storage a tile costs is
// `(BM + BN) * PIE_BK * 4` bytes and wgpu's downlevel default caps that at
// 16352, so the wide shapes -- BM=64/BN=64, BM=128/BN=32 -- have to stage 16.
// The narrow ones do not, and paying two barriers and a fresh staging pass for
// every 16 columns of K when 32 fit is half the loop's overhead spent on
// nothing: at (32, 32) the tile costs 8192 bytes at PIE_BK=32, and a 512-token
// projection through K=2048 drops from 128 staging passes to 64.
const PIE_BK = select(16, 32, PIE_BM + PIE_BN <= 127);
// The workgroup is 32x2x2 = 128 invocations, matching the Vulkan body's shape.
const PIE_LANES = 128u;

// A LANE OWNS FOUR COLUMNS, not one.
//
// Both staged slabs are read vectorised, which is what keeps the inner product
// off the workgroup memory's throughput ceiling: four columns at a time makes
// `ws` a vec4 whose index carries the lane in the SLOW position, so consecutive
// lanes still touch consecutive words and nothing strides the banks, and four K
// columns at a time makes `xs` a vec4 whose index does not carry the lane at
// all. Five vector loads then retire sixteen multiplies where the scalar shape
// spent thirty-two loads on them.
//
// The terms stay separate `+` operands rather than a `dot`: the accumulation
// order is what the parity walk against the sibling backends compares.
const PIE_BK4 = PIE_BK / 4;
const PIE_BN4 = PIE_BN / 4;
// vec4 accumulators per lane. Ceiling rather than a quotient because BM=16 with
// BN=16 gives 64 of them across 128 lanes -- the only declared shape that does
// not fill the workgroup, and the reason `r` is bounds-checked below.
const PIE_ACCV = (PIE_BM * PIE_BN4 + 127) / 128;

var<workgroup> xs: array<vec4<f32>, PIE_BM * PIE_BK4>;
// Stored K-major (`ws[kk * BN4 + c4]`) though it is read column-major, so that
// consecutive lanes -- which hold consecutive `c4` -- touch consecutive words in
// the inner loop instead of striding by PIE_BK.
var<workgroup> ws: array<vec4<f32>, PIE_BN4 * PIE_BK>;

fn input_stride() -> u32 {
//#if defined(PIE_STRIDED)
    return u32(params.row_stride);
//#else
    return u32(params.k);
//#endif
}

fn output_stride() -> u32 {
//#if defined(PIE_STRIDED)
    return u32(params.row_stride);
//#else
    return u32(params.n);
//#endif
}

fn load_x(row: u32, kk: u32) -> f32 {
    let i = row * input_stride() + kk;
//#if defined(PIE_FP16_PRECAST)
    // The value was rounded to fp16 by the `cast_qmm_input` pass; the multiply
    // it feeds is still fp32, which is what makes this a precast and not an
    // fp16 GEMM.
    let word = half_in[i >> 1u];
    return qmm_f16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
//#else
    return qmm_bf16(x[i >> 1u], i);
//#endif
}

// Four consecutive K of one column, dequantised from ONE packed word.
//
// `k` and not `row_stride` indexes the packed planes even in the strided
// variants: the stride is the ACTIVATION's, the weight is always densely
// packed by K.
//
// A one-value-at-a-time dequant reads the word, the scale and the bias for
// every single code, so staging a (BN, BK) weight tile that way issues three
// global loads per value -- 3072 of them for the 1024 values a (32, 32) tile
// stages. Four codes at a time is the widest step every declared width allows
// (eight fit in a word at four bits, four at eight), and since `PIE_GROUP` is
// never below 32 the four share a scale and a bias too. Twelve loads then
// retire sixteen values where they used to retire four.
//
// `k` must be a multiple of four, which the caller's `k4 * 4` and `PIE_BK`
// being a multiple of four together guarantee.
fn affine_quad(col: u32, k: u32) -> vec4<f32> {
    let len = u32(params.k);
    let word = w[pie_affine_word_of(col, len, k)];
    let g = pie_affine_scale_of(col, len, k);
    return pie_affine_dequant4(
        word,
        pie_affine_code_of(k),
        qmm_bf16(scales[g >> 1u], g),
        qmm_bf16(biases[g >> 1u], g),
    );
}

// A tile is BM x BN whatever the matrix is, so the last tile in each direction
// runs off the end of the real output.
//
// The COLUMN overhang is the dangerous one and this guard is the fix. The
// output is row-major, so a lane at `col >= n` does not write past the buffer,
// it writes over `(row + 1, col - n)` -- a live element of the NEXT row, with a
// value computed from weights that are themselves out of range. A GPU sweep
// over the whole `{16,32,64}^2` tile grid at `n = 47` caught this in the Vulkan
// tree: every row after the first began with the zero an out-of-range weight
// fetch produced, and it was invisible at the tile-aligned shapes the earlier
// tests happened to use. `n` is in the uniform block, so the guard is exact.
//
// THE ROW OVERHANG IS GUARDED TOO, AND USED NOT TO BE.
//
// It used to say the row overhang *could not* be guarded here, because no
// entrypoint's block carried `m` -- the launch named the grid and nothing
// else -- so the contract was that the output allocation is a whole number of
// `BM` rows and the extra rows were written with garbage and ignored, exactly
// as `qmm_t.metal` does by calling MLX's `store_result` where a
// `store_result_safe` sits beside it.
//
// That was never a shortage of information. Every one of this file's
// entrypoints already computed `m` -- `ctx.ask::<i32, keys::Rows>()` in
// `kernels-wgpu::quant`, which is what `qmm_grid` rounds up with `div_ceil`
// -- and simply did not pass it. Now `Params` ends with `m` and this is the
// whole of what that buys.
//
// What it is FOR: a fire whose row count the tile does not divide. Refusing
// one (`Ungeometric::PartialTile`, and `GuardPred::TokensMultipleOf` above
// it) costs a prefill 2.34x on thirty-one prompt lengths in thirty-two,
// because the fire falls to the matvec. `driver-wgpu`'s `Serving::prefill`
// carries the numbers and what letting it through did before this guard
// existed: an unrelated answer AND a twelvefold slowdown, the second being
// the first's consequence and not a cost of padding.
//
// `gpu.rs`'s `a_tiled_gemm_agrees_over_every_tile_shape_and_quantization_point`
// is the proof. It fires m = 33 over all nine tiles and six codecs and now
// asserts the overhang still holds its sentinel; delete the `row` term below
// and it reports 705 of 705 overhang values written.
fn write_out(row: u32, col: u32, value: f32, slice: u32) {
    if col >= u32(params.n) || row >= u32(params.m) {
        return;
    }
    var v = value;
//#if defined(PIE_BIAS)
    v = v + qmm_bf16(extra[col >> 1u], col);
//#endif
//#if defined(PIE_RESIDUAL)
    let at = row * output_stride() + col;
    // Rounded to bf16 BEFORE the add, deliberately: it makes the fused variant
    // bit-identical to the two-kernel path (project, then `residual_add`) it
    // replaces, which is what makes swapping them a pure performance change.
    let q = pie_bf16_to_f32(pie_f32_to_bf16(v));
    store_half(at, pie_f32_to_bf16(q + qmm_bf16(extra[at >> 1u], at)));
//#elif defined(PIE_SPLITK) && defined(PIE_OUT_F32)
    out_f32[row * u32(params.n) + col + slice * u32(params.split_k_partition_stride)] = v;
//#elif defined(PIE_SPLITK)
    store_half(row * u32(params.n) + col + slice * u32(params.split_k_partition_stride), pie_f32_to_bf16(v));
//#else
    store_half(row * output_stride() + col, pie_f32_to_bf16(v));
//#endif
}

@compute @workgroup_size(32, 2, 2)
fn main(
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let tile_row = wg.y * u32(PIE_BM);
    let tile_col = wg.x * u32(PIE_BN);
//#if defined(PIE_SPLITK)
    let k0 = wg.z * u32(params.k_partition_size);
    let k1 = min(u32(params.k), k0 + u32(params.k_partition_size));
//#else
    let k0 = 0u;
    let k1 = u32(params.k);
//#endif

    var acc: array<vec4<f32>, PIE_ACCV>;
    for (var a = 0u; a < u32(PIE_ACCV); a = a + 1u) {
        acc[a] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // `k0` and `k1` come from the workgroup id and the uniform block and from
    // nothing else, so every invocation runs this loop the same number of times
    // and both barriers below are reached by all 128 of them. That is the whole
    // discipline: no `return`, no `continue`, no lane-dependent bound anywhere
    // between here and the epilogue.
    var kb = k0;
    while kb < k1 {
        // K is NOT a whole number of PIE_BK blocks. Assuming it was is what
        // broke gemma-4-31b at K=5376, per the Metal comments, so the tail is
        // measured here and staged as ZERO below -- which is what lets the
        // inner product run the full PIE_BK with no bound of its own.
        let kn = min(u32(PIE_BK), k1 - kb);

        // Before overwriting the tiles, wait for the previous iteration's reads
        // of them to finish. On the first iteration this costs one barrier and
        // protects nothing, which is cheaper than proving it can be skipped.
        workgroupBarrier();
        for (var e = local; e < u32(PIE_BM * PIE_BK4); e = e + PIE_LANES) {
            let r = e / u32(PIE_BK4);
            let k4 = (e - r * u32(PIE_BK4)) * 4u;
            var v = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            if k4 + 0u < kn { v.x = load_x(tile_row + r, kb + k4 + 0u); }
            if k4 + 1u < kn { v.y = load_x(tile_row + r, kb + k4 + 1u); }
            if k4 + 2u < kn { v.z = load_x(tile_row + r, kb + k4 + 2u); }
            if k4 + 3u < kn { v.w = load_x(tile_row + r, kb + k4 + 3u); }
            xs[e] = v;
        }
        // FOUR K OF FOUR COLUMNS PER LANE, not one element. See `affine_quad`:
        // the packed word and the scale/bias pair a code needs are shared by
        // the four codes around it, so reading them once and expanding four
        // cuts this pass's global traffic by four. The tile is written in the
        // same K-major layout either way -- one lane's sixteen values are four
        // `ws` vec4s, at four consecutive `kk` and one `c4`.
        for (var e = local; e < u32(PIE_BN4 * PIE_BK4); e = e + PIE_LANES) {
            let c4 = e / u32(PIE_BK4);
            let kk = (e - c4 * u32(PIE_BK4)) * 4u;
            var q0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var q1 = q0;
            var q2 = q0;
            var q3 = q0;
            if kk < kn {
                let c = tile_col + c4 * 4u;
                q0 = affine_quad(c + 0u, kb + kk);
                q1 = affine_quad(c + 1u, kb + kk);
                q2 = affine_quad(c + 2u, kb + kk);
                q3 = affine_quad(c + 3u, kb + kk);
            }
            // K is not required to be a multiple of four, so the staged tail
            // is zeroed component-wise rather than by the block guard above.
            let base = kk * u32(PIE_BN4) + c4;
            let n4 = u32(PIE_BN4);
            ws[base + 0u * n4] = select(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(q0.x, q1.x, q2.x, q3.x), kk + 0u < kn);
            ws[base + 1u * n4] = select(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(q0.y, q1.y, q2.y, q3.y), kk + 1u < kn);
            ws[base + 2u * n4] = select(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(q0.z, q1.z, q2.z, q3.z), kk + 2u < kn);
            ws[base + 3u * n4] = select(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(q0.w, q1.w, q2.w, q3.w), kk + 3u < kn);
        }
        workgroupBarrier();

        // EVERY ROW THE LANE OWNS, OVER ONE COLUMN GROUP -- so the four
        // `ws` vec4s an inner step needs are read ONCE and retire every
        // accumulator, instead of once per accumulator.
        //
        // `c4` is `idx % BN4` and `idx` differs between accumulators by
        // exactly PIE_LANES, so where `BN4` divides 128 -- it is 4, 8 or 16 at
        // every declared shape, since BN is 16, 32 or 64 -- every
        // accumulator a lane holds sits in the SAME column group and differs
        // only in its row, by a fixed `PIE_LANES / BN4`.
        //
        // That is the whole of this loop's arithmetic intensity. Per K step a
        // lane reads `4 + ACCV` workgroup vec4s and retires `16 * ACCV`
        // multiplies, where the per-accumulator sweep this replaces read
        // `5 * ACCV` for the same work: 3.2 multiplies a load at every shape,
        // against 5.3 at two accumulators and 10.7 at eight.
        //
        // # Why this was worth writing
        //
        // It used to be a two-accumulator special case with a general loop
        // beside it, and the special case was the only fast path. Sweeping
        // the tile at 512 rows of llama-3.2-1B's own shapes, the three tiles
        // where `BM * BN4` is exactly 256 -- (16, 64), (32, 32), (64, 16) --
        // read 1.56, 1.70 and 1.85 TFLOP/s, and the other six ALL read about
        // 1.0. The tile did not predict the rate; whether it hit the special
        // case did. (64, 64) has four times (32, 32)'s accumulators and the
        // best intensity available, and it was the SLOWEST shape measured.
        //
        // After, on the same sweep -- the ACC2 shapes reproduce to the third
        // digit, so this arm is the old special case at ACCV = 2 and the
        // change is what happens above it:
        //
        // ```text
        //            bn=16   bn=32   bn=64        (TFLOP/s, q/o at m=512)
        //   bm=16     1.07    0.90    1.56
        //   bm=32     1.04    1.68    2.54   <- was 0.99
        //   bm=64     1.82    2.66    2.94   <- was 1.07, 0.98
        // ```
        //
        // # WHY THE ACCUMULATORS ARE NAMED AND NOT AN ARRAY
        //
        // The first attempt at this was one loop over `acc[a]`, which is the
        // obvious spelling and LOST: (32, 32) fell 1.68 -> 1.40 and (64, 16)
        // 1.85 -> 1.24, while the wide tiles gained only to 1.2. A dynamically
        // indexed `array<vec4<f32>, N>` local is not registers -- MSL puts it
        // in thread-local memory -- so the loop traded four workgroup reads
        // for ACCV stack round-trips and paid on every shape that already had
        // a fast path.
        //
        // It is worth knowing which suspect it was NOT. The inner `r < BM`
        // guard was replaced by a clamp, on the theory that the branch cost
        // it: (32, 32) moved 1.40 -> 1.44 and (16, 16) fell 1.14 -> 0.52,
        // since its dead half then did the full sweep. The branch was never
        // the problem.
        //
        // So the arms below are unrolled by hand at 1, 2, 4 and 8, which is
        // every `ceil(BM * BN4 / 128)` the declared shapes produce. The cost
        // is a fifth arm to add if a tile ever needs one, so the
        // per-accumulator sweep they replaced stays below them as the
        // fallback -- an unlisted `ACCV` is then SLOW rather than silently
        // short by however many accumulators the arms did not name.
        //
        // The accumulation order is UNCHANGED, which the parity walk against
        // `kernels-metal` and `kernels-vulkan` requires: each accumulator
        // still folds `xv.x * w0 + xv.y * w1 + ...` in that order, over
        // ascending `k4`.
        let c4 = local % u32(PIE_BN4);
        let r0 = local / u32(PIE_BN4);
        let rstep = PIE_LANES / u32(PIE_BN4);
        // Clamped so an arm naming more accumulators than this shape holds
        // still COMPILES. Every arm below is guarded by a const comparison,
        // so only one survives folding and the clamps in it are identities.
        let n0 = min(0u, u32(PIE_ACCV) - 1u);
        let n1 = min(1u, u32(PIE_ACCV) - 1u);
        let n2 = min(2u, u32(PIE_ACCV) - 1u);
        let n3 = min(3u, u32(PIE_ACCV) - 1u);
        let n4 = min(4u, u32(PIE_ACCV) - 1u);
        let n5 = min(5u, u32(PIE_ACCV) - 1u);
        let n6 = min(6u, u32(PIE_ACCV) - 1u);
        let n7 = min(7u, u32(PIE_ACCV) - 1u);

        if u32(PIE_ACCV) == 8u {
            var s0 = acc[n0];
            var s1 = acc[n1];
            var s2 = acc[n2];
            var s3 = acc[n3];
            var s4 = acc[n4];
            var s5 = acc[n5];
            var s6 = acc[n6];
            var s7 = acc[n7];
            for (var k4 = 0u; k4 < u32(PIE_BK4); k4 = k4 + 1u) {
                let kk = k4 * 4u;
                let w0 = ws[(kk + 0u) * u32(PIE_BN4) + c4];
                let w1 = ws[(kk + 1u) * u32(PIE_BN4) + c4];
                let w2 = ws[(kk + 2u) * u32(PIE_BN4) + c4];
                let w3 = ws[(kk + 3u) * u32(PIE_BN4) + c4];
                let x0 = xs[(r0) * u32(PIE_BK4) + k4];
                let x1 = xs[(r0 + 1u * rstep) * u32(PIE_BK4) + k4];
                let x2 = xs[(r0 + 2u * rstep) * u32(PIE_BK4) + k4];
                let x3 = xs[(r0 + 3u * rstep) * u32(PIE_BK4) + k4];
                let x4 = xs[(r0 + 4u * rstep) * u32(PIE_BK4) + k4];
                let x5 = xs[(r0 + 5u * rstep) * u32(PIE_BK4) + k4];
                let x6 = xs[(r0 + 6u * rstep) * u32(PIE_BK4) + k4];
                let x7 = xs[(r0 + 7u * rstep) * u32(PIE_BK4) + k4];
                s0 = s0 + x0.x * w0 + x0.y * w1 + x0.z * w2 + x0.w * w3;
                s1 = s1 + x1.x * w0 + x1.y * w1 + x1.z * w2 + x1.w * w3;
                s2 = s2 + x2.x * w0 + x2.y * w1 + x2.z * w2 + x2.w * w3;
                s3 = s3 + x3.x * w0 + x3.y * w1 + x3.z * w2 + x3.w * w3;
                s4 = s4 + x4.x * w0 + x4.y * w1 + x4.z * w2 + x4.w * w3;
                s5 = s5 + x5.x * w0 + x5.y * w1 + x5.z * w2 + x5.w * w3;
                s6 = s6 + x6.x * w0 + x6.y * w1 + x6.z * w2 + x6.w * w3;
                s7 = s7 + x7.x * w0 + x7.y * w1 + x7.z * w2 + x7.w * w3;
            }
            acc[n0] = s0;
            acc[n1] = s1;
            acc[n2] = s2;
            acc[n3] = s3;
            acc[n4] = s4;
            acc[n5] = s5;
            acc[n6] = s6;
            acc[n7] = s7;
        } else if u32(PIE_ACCV) == 4u {
            var s0 = acc[n0];
            var s1 = acc[n1];
            var s2 = acc[n2];
            var s3 = acc[n3];
            for (var k4 = 0u; k4 < u32(PIE_BK4); k4 = k4 + 1u) {
                let kk = k4 * 4u;
                let w0 = ws[(kk + 0u) * u32(PIE_BN4) + c4];
                let w1 = ws[(kk + 1u) * u32(PIE_BN4) + c4];
                let w2 = ws[(kk + 2u) * u32(PIE_BN4) + c4];
                let w3 = ws[(kk + 3u) * u32(PIE_BN4) + c4];
                let x0 = xs[(r0) * u32(PIE_BK4) + k4];
                let x1 = xs[(r0 + 1u * rstep) * u32(PIE_BK4) + k4];
                let x2 = xs[(r0 + 2u * rstep) * u32(PIE_BK4) + k4];
                let x3 = xs[(r0 + 3u * rstep) * u32(PIE_BK4) + k4];
                s0 = s0 + x0.x * w0 + x0.y * w1 + x0.z * w2 + x0.w * w3;
                s1 = s1 + x1.x * w0 + x1.y * w1 + x1.z * w2 + x1.w * w3;
                s2 = s2 + x2.x * w0 + x2.y * w1 + x2.z * w2 + x2.w * w3;
                s3 = s3 + x3.x * w0 + x3.y * w1 + x3.z * w2 + x3.w * w3;
            }
            acc[n0] = s0;
            acc[n1] = s1;
            acc[n2] = s2;
            acc[n3] = s3;
        } else if u32(PIE_ACCV) == 2u {
            var s0 = acc[n0];
            var s1 = acc[n1];
            for (var k4 = 0u; k4 < u32(PIE_BK4); k4 = k4 + 1u) {
                let kk = k4 * 4u;
                let w0 = ws[(kk + 0u) * u32(PIE_BN4) + c4];
                let w1 = ws[(kk + 1u) * u32(PIE_BN4) + c4];
                let w2 = ws[(kk + 2u) * u32(PIE_BN4) + c4];
                let w3 = ws[(kk + 3u) * u32(PIE_BN4) + c4];
                let x0 = xs[(r0) * u32(PIE_BK4) + k4];
                let x1 = xs[(r0 + 1u * rstep) * u32(PIE_BK4) + k4];
                s0 = s0 + x0.x * w0 + x0.y * w1 + x0.z * w2 + x0.w * w3;
                s1 = s1 + x1.x * w0 + x1.y * w1 + x1.z * w2 + x1.w * w3;
            }
            acc[n0] = s0;
            acc[n1] = s1;
        } else if u32(PIE_ACCV) == 1u {
            // ONE accumulator, and the only shape that reaches it -- (16, 16)
            // -- has 64 of them across 128 lanes, so half the workgroup owns a
            // row past the tile and must not read `xs` at all. Leaving early
            // is safe: no barrier stands between here and the epilogue, which
            // makes the same test before it writes.
            if r0 < u32(PIE_BM) {
                var s0 = acc[n0];
                for (var k4 = 0u; k4 < u32(PIE_BK4); k4 = k4 + 1u) {
                    let kk = k4 * 4u;
                    let w0 = ws[(kk + 0u) * u32(PIE_BN4) + c4];
                    let w1 = ws[(kk + 1u) * u32(PIE_BN4) + c4];
                    let w2 = ws[(kk + 2u) * u32(PIE_BN4) + c4];
                    let w3 = ws[(kk + 3u) * u32(PIE_BN4) + c4];
                    let x0 = xs[(r0) * u32(PIE_BK4) + k4];
                    s0 = s0 + x0.x * w0 + x0.y * w1 + x0.z * w2 + x0.w * w3;
                }
                acc[n0] = s0;
            }
        } else {
            // NO DECLARED SHAPE REACHES THIS, and it exists so that one added
            // later is slow rather than wrong. The arms above are unrolled at
            // the four `ACCV` the nine declared tiles produce; a tenth tile
            // with, say, 16 accumulators would otherwise fall into the ACCV
            // == 1 arm and leave fifteen of them at zero, which is a fluent
            // wrong answer and the failure this file exists to avoid.
            //
            // It is the per-accumulator sweep the arms above replaced, kept
            // for its correctness rather than its rate: `5 * ACCV` workgroup
            // reads for `16 * ACCV` multiplies, and a dynamically indexed
            // `acc`, which is about 1.0 TFLOP/s where the arms reach 2.9.
            for (var a = 0u; a < u32(PIE_ACCV); a = a + 1u) {
                let r = r0 + a * rstep;
                if r >= u32(PIE_BM) { continue; }
                var s = acc[a];
                for (var k4 = 0u; k4 < u32(PIE_BK4); k4 = k4 + 1u) {
                    let xv = xs[r * u32(PIE_BK4) + k4];
                    let kk = k4 * 4u;
                    s = s + xv.x * ws[(kk + 0u) * u32(PIE_BN4) + c4]
                          + xv.y * ws[(kk + 1u) * u32(PIE_BN4) + c4]
                          + xv.z * ws[(kk + 2u) * u32(PIE_BN4) + c4]
                          + xv.w * ws[(kk + 3u) * u32(PIE_BN4) + c4];
                }
                acc[a] = s;
            }
        }

        kb = kb + u32(PIE_BK);
    }

    for (var a = 0u; a < u32(PIE_ACCV); a = a + 1u) {
        let idx = local + a * PIE_LANES;
        let r = idx / u32(PIE_BN4);
        let c4 = idx - r * u32(PIE_BN4);
        if r >= u32(PIE_BM) { continue; }
        let c = c4 * 4u;
        write_out(tile_row + r, tile_col + c + 0u, acc[a].x, wg.z);
        write_out(tile_row + r, tile_col + c + 1u, acc[a].y, wg.z);
        write_out(tile_row + r, tile_col + c + 2u, acc[a].z, wg.z);
        write_out(tile_row + r, tile_col + c + 3u, acc[a].w, wg.z);
    }
}
//#endif

// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_4_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_4_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_4_bm_16_bn_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_4_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_4_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_4_bm_32_bn_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_4_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_4_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_4_bm_64_bn_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_8_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_8_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_8_bm_16_bn_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_8_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_8_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_8_bm_32_bn_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_8_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_8_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_8_bm_64_bn_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_4_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_4_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_4_bm_16_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_4_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_4_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_4_bm_32_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_4_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_4_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_4_bm_64_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_8_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_8_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_8_bm_16_bn_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_8_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_8_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_8_bm_32_bn_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_8_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_8_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_8_bm_64_bn_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4 PIE_GROUP=64 PIE_BITS=4 PIE_BM=128 PIE_BN=32 PIE_PROBE_SHAPE=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_PROBE_SHAPE=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_PROBE_SHAPE=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_PROBE_SHAPE=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_PROBE_SHAPE=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_8_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_8_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_8_bm_16_bn_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_8_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_8_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_8_bm_64_bn_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=64
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_16_bn_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_32_bn_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_64_bn_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_16_bn_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_32_bn_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_64_bn_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_16_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_32_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_64_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_16_bn_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_32_bn_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_64_bn_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_16_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_32_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_64_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_16_bn_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_32_bn_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=16 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_64_bn_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=64 PIE_BIAS=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_BIAS=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_BIAS=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=64 PIE_BIAS=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_BIAS=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_BIAS=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_BIAS=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_BIAS=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_BIAS=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_BIAS=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=64 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_16_bn_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_32_bn_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_64_bn_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_16_bn_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_32_bn_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_64_bn_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_16_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_32_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_64_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_16_bn_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_32_bn_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_64_bn_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_16_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_32_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_64_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_16_bn_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_32_bn_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=16 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_64_bn_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=64 PIE_RESIDUAL=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=64 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_128_b_8_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_128_b_8_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_128_b_8_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_32_b_4_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_32_b_4_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_32_b_4_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_32_b_8_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_32_b_8_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_32_b_8_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_64_b_8_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_64_b_8_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_64_b_8_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_128_b_4_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_128_b_4_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_128_b_4_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_128_b_8_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_128_b_8_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_128_b_8_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_32_b_4_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_32_b_4_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_32_b_4_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_32_b_8_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_32_b_8_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_32_b_8_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_64_b_8_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_64_b_8_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_64_b_8_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1
// pie:instantiate affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_SPLITK=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_128_b_4_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_128_b_4_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_128_b_4_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_128_b_8_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_128_b_8_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_128_b_8_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_32_b_4_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_32_b_4_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_32_b_4_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_32_b_8_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_32_b_8_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_32_b_8_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_64_b_8_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_64_b_8_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_64_b_8_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_STRIDED=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_STRIDED=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_STRIDED=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_FP16_PRECAST=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_128_b_4_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_128_b_4_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_128_b_4_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_128_b_8_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_128_b_8_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_128_b_8_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_32_b_4_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_32_b_4_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_32_b_4_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_32_b_8_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_32_b_8_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_32_b_8_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_64_b_8_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_64_b_8_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_64_b_8_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1
// pie:instantiate cast_qmm_input_bfloat16_to_float16 PIE_CAST_INPUT=1
// pie:instantiate cast_qmm_input_strided_bfloat16_to_float16 PIE_STRIDED=1 PIE_CAST_INPUT=1
// pie:instantiate qmm_splitk_reduce_bfloat16 PIE_SPLITK=1 PIE_REDUCE=1
// pie:instantiate qmm_splitk_reduce_f32_bfloat16 PIE_SPLITK=1 PIE_REDUCE=1 PIE_PARTIAL_F32=1
