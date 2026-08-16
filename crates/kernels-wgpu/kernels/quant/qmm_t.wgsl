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
// accumulation is fp32 either way, and `affine_value`/`load_x` are called
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

// K elements staged per pass. 16 and not 32 because the workgroup storage a
// tile costs is `(BM + BN) * PIE_BK * 4` bytes and wgpu's downlevel default
// caps that at 16352: BM=64/BN=64 at PIE_BK=32 is 16384, which is over by 32
// bytes and fails at `create_shader_module` on exactly the devices this
// backend exists for. At 16 the widest shape here (BM=128, BN=32) costs 10240.
const PIE_BK = 16;
// The workgroup is 32x2x2 = 128 invocations, matching the Vulkan body's shape.
const PIE_LANES = 128u;
// Output elements per lane. Integral for every declared (BM, BN): the smallest
// tile is 16x16 = 256, and all ten shapes are multiples of 128.
const PIE_ACC = PIE_BM * PIE_BN / 128;

var<workgroup> xs: array<f32, PIE_BM * PIE_BK>;
// Stored K-major (`ws[kk * BN + c]`) though it is read column-major, so that
// consecutive lanes -- which hold consecutive `c` -- touch consecutive words in
// the inner loop instead of striding by PIE_BK.
var<workgroup> ws: array<f32, PIE_BN * PIE_BK>;

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

// One dequantised weight. `k` and not `row_stride` indexes the packed planes
// even in the strided variants: the stride is the ACTIVATION's, the weight is
// always densely packed by K.
fn affine_value(col: u32, kk: u32) -> f32 {
    let len = u32(params.k);
    let word = w[pie_affine_word_of(col, len, kk)];
    let g = pie_affine_scale_of(col, len, kk);
    return pie_affine_value(
        word,
        pie_affine_code_of(kk),
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
// The ROW overhang cannot be guarded here, because no entrypoint's block
// carries `m` -- the launch names the grid and nothing else. So the contract is
// that the output allocation is a whole number of `BM` rows; those extra rows
// are written with garbage and ignored. That is a real requirement on the
// caller, not an oversight: `qmm_t.metal` has the same overhang deliberately,
// calling MLX's `store_result` where a `store_result_safe` sits right beside it.
fn write_out(row: u32, col: u32, value: f32, slice: u32) {
    if col >= u32(params.n) {
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

    var acc: array<f32, PIE_ACC>;
    for (var a = 0u; a < u32(PIE_ACC); a = a + 1u) {
        acc[a] = 0.0;
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
        for (var e = local; e < u32(PIE_BM * PIE_BK); e = e + PIE_LANES) {
            let r = e / u32(PIE_BK);
            let kk = e - r * u32(PIE_BK);
            var v = 0.0;
            if kk < kn {
                v = load_x(tile_row + r, kb + kk);
            }
            xs[e] = v;
        }
        for (var e = local; e < u32(PIE_BN * PIE_BK); e = e + PIE_LANES) {
            let c = e / u32(PIE_BK);
            let kk = e - c * u32(PIE_BK);
            var v = 0.0;
            if kk < kn {
                v = affine_value(tile_col + c, kb + kk);
            }
            ws[kk * u32(PIE_BN) + c] = v;
        }
        workgroupBarrier();

        for (var a = 0u; a < u32(PIE_ACC); a = a + 1u) {
            let idx = local + a * PIE_LANES;
            let r = idx / u32(PIE_BN);
            let c = idx - r * u32(PIE_BN);
            var s = acc[a];
            for (var kk = 0u; kk < u32(PIE_BK); kk = kk + 1u) {
                s = s + xs[r * u32(PIE_BK) + kk] * ws[kk * u32(PIE_BN) + c];
            }
            acc[a] = s;
        }

        kb = kb + u32(PIE_BK);
    }

    for (var a = 0u; a < u32(PIE_ACC); a = a + 1u) {
        let idx = local + a * PIE_LANES;
        let r = idx / u32(PIE_BN);
        let c = idx - r * u32(PIE_BN);
        write_out(tile_row + r, tile_col + c, acc[a], wg.z);
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
