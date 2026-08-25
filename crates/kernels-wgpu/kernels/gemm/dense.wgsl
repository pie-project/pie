// The DENSE bf16 projection: `y[M, N] = act[M, K] @ w[N, K]^T`.
//
// This is `gemm.matmul`, `gemm.lm_head` and `gemm.attention_landing` -- one
// arithmetic under three names, exactly as `kernels-cuda/src/gemm.rs` answers
// all three with one `act_x_wt_bf16`. The weight is stored TRANSPOSED
// (`[N, K]`, output-major, K contiguous), which is what the point declaration
// states by sizing the result `[act.rows, w.axis(0)]`: axis 0 of the weight is
// the output width, so the contraction runs down each weight ROW.
//
// # Why a shader and not a vendor call
//
// On CUDA this point is cuBLAS. There is no equivalent road on this plane:
// `kernels::plane::Fire` names a FILE, an ENTRYPOINT and a GRID, and
// `driver-wgpu`'s baker turns exactly that into `dispatch_workgroups`. WebGPU
// has no BLAS to call and no vendor path to name -- the only matrix hardware it
// exposes at all is `wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX`, which is
// a WGSL type (`coop_mat8x8`) written inside a shader, not a library called
// beside one. So the choice here was own-shader or nothing, and this file is
// the shader.
//
// A `@matrix` tier over these entry points is open and deliberately not taken
// yet: `quant/qmm_t.wgsl`'s cooperative arm is what it would look like, and the
// tier's cost there is a hand-unrolled accumulator block per tile shape. The
// baseline has to exist and be right first, and a tier is an ADDITIONAL variant
// of an entrypoint that already exists (`build.rs`'s `check_tiers`), so nothing
// here has to move for one to arrive.
//
// # Two entry points, and the rule that picks between them
//
//   * `dense_gemm_t_bfloat16` -- the staged tile loop, for M >= PIE_BM.
//   * `dense_gemv_t_bfloat16` -- 32 lanes splitting K per output column, for
//     M < PIE_BM.
//
// The split is not a tuning knob, it is the tile's own arithmetic. The tile
// path computes a whole BM x BN block whatever M is, so at M = 1 it stages BM
// rows of activation to use one of them and does BM times the multiplies the
// answer needs; the vector path does exactly M*N*K and no more. BM is therefore
// the crossing point BY CONSTRUCTION, and the host states it in
// `kernels_wgpu::quant::TILE_M`. Where inside that range the two actually cross
// on a given adapter is unmeasured, and nothing here is a performance claim.
//
// # The tile, and where its shape comes from
//
// BM = BN = BK = 32 over a 128-invocation workgroup, which is
// `quant/qmm_t.wgsl`'s (32, 32) point exactly. That file swept the whole
// `{16,32,64}^2` grid on this plane and this is the shape whose accumulator
// count -- `BM * BN4 / LANES` = 2 -- is the one its inner loop was written for.
// The staged tiles cost `(BM + BN) * BK * 4` = 8192 bytes against the 16352
// wgpu's downlevel default allows.
//
// What is NOT carried over from that file is its dequantisation: a dense weight
// is bf16 in memory, so the four-codes-share-a-scale argument that makes
// `qmm_t` stage its weight tile through `affine_quad` has nothing to buy here.
// The staging loops below issue plain scalar `load_x` / `load_w`, each of which
// reads a whole `u32` and keeps one half. That is 2x the bytes the tile needs,
// and it is the deliberate simple choice: consecutive lanes read consecutive
// half-indices, so both halves of every word are read by the same lane or by
// its neighbour and the second read is an L1 hit. A pair-at-a-time stage is
// open and would need the flat index's parity checked at every load, which is
// the complexity `qmm_t`'s `load_x_quad` carries and pays for with a codec this
// file does not have.
//
// # Honest duplication
//
// `kernels-metal/kernels/gemm/dense.metal` answers the same three points on
// MLX's steel primitives and shares nothing with this but the arithmetic. Two
// shaders, on purpose.

//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> w: array<u32>;
// bf16 behind an `array<atomic<u32>>`, for the reason `store_half` gives.
@group(0) @binding(2) var<storage, read_write> out_: array<atomic<u32>>;

// The three extents, in the order the body passes them: `m` then `n` then `k`,
// which is rows, columns, contraction. A body that swapped `n` and `k` would
// index a rectangle of the right size the wrong way round and report nothing.
struct Params { m: i32, n: i32, k: i32 }
@group(1) @binding(0) var<uniform> params: Params;

// One bf16 of the activation row. `pie_bf16_at` takes a WORD and a half-index
// because WGSL has no `ptr<storage, ...>` parameter (`common/bf16.inc.wgsl`
// says why at length), so the split is stated at every load.
fn load_x(row: u32, kk: u32) -> f32 {
    let i = row * u32(params.k) + kk;
    return pie_bf16_at(x[i >> 1u], i);
}

// One bf16 of a weight ROW, which is one output column's whole contraction.
fn load_w(col: u32, kk: u32) -> f32 {
    let i = col * u32(params.k) + kk;
    return pie_bf16_at(w[i >> 1u], i);
}

// Write one 16-bit element of the output.
//
// The word at `i >> 1` holds elements `i & ~1` and `i | 1` and this invocation
// owns only one of them: adjacent output COLUMNS belong to adjacent lanes, and
// when the output pitch `n` is odd the even/odd pairing shifts by one on every
// row, so no lane assignment can make the pairs whole. A plain
// read-modify-write drops whichever half landed second. The CAS is
// device-scoped, which is the scope of the race -- the two lanes may be in
// different workgroups -- and it retries on the spurious failure `...Weak` is
// permitted. This is `quant/qmm_t.wgsl`'s `store_half` restated rather than
// shared: that file's copy is one of three arms behind a `//#if`, chosen by
// which of its buffers the variant writes.
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

// One output element, guarded on both axes.
//
// The COLUMN overhang is the dangerous one. The output is row-major, so a lane
// at `col >= n` does not write past the buffer, it writes over
// `(row + 1, col - n)` -- a live element of the NEXT row, computed from a
// weight fetch that was itself out of range. `qmm_t.wgsl` records the GPU sweep
// in the Vulkan tree where exactly that made every row after the first begin
// with a zero, and how invisible it was at the tile-aligned shapes.
fn write_at(row: u32, col: u32, value: f32) {
    if col >= u32(params.n) || row >= u32(params.m) {
        return;
    }
    store_half(row * u32(params.n) + col, pie_f32_to_bf16(value));
}

//#if defined(PIE_GEMV)

// ── THE VECTOR ARM ──────────────────────────────────────────────────────────
//
// M below the row tile. One workgroup is 32 K-lanes by 8 columns: the 32 lanes
// split one column's contraction between them and fold it in workgroup memory,
// and the 8 columns are there so the fold's log2(32) barriers are paid once for
// eight results instead of once for one.
//
// K-SPLITTING RATHER THAN ONE ELEMENT PER INVOCATION, because the element-wise
// shape has no parallelism to give at decode: `y` is then `1 x N`, so a grid
// over the result is N invocations each walking the whole of K, and at
// N = K = 4096 that is 4096 threads on a card with 142 multiprocessors. The
// split multiplies the available work by 32 and turns each lane's walk down K
// into a strided read its 31 neighbours coalesce with.
//
// The subgroup tier is the obvious next step and is not taken here:
// `subgroupAdd` over 32 lanes is exactly this fold with no workgroup memory and
// no barriers, and it is an ADDITIONAL variant this file can gain without
// moving anything else.
const PIE_KLANES = 32u;
const PIE_NLANES = 8u;

var<workgroup> partial: array<f32, PIE_KLANES * PIE_NLANES>;

@compute @workgroup_size(32, 8, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let lane = lid.x;
    let col = gid.y;
    let row = gid.z;
    let base = lid.y * PIE_KLANES;

    // The guard is on the ACCUMULATION and not on the invocation: every lane
    // has to reach the barriers below, so an out-of-range column contributes a
    // zero and is dropped at the store instead of returning early.
    var acc = 0.0;
    if col < u32(params.n) && row < u32(params.m) {
        let k = u32(params.k);
        var kk = lane;
        while kk < k {
            acc = acc + load_x(row, kk) * load_w(col, kk);
            kk = kk + PIE_KLANES;
        }
    }
    partial[base + lane] = acc;

    // The fold. `s` starts at a constant and is stepped by a constant, so every
    // invocation runs this loop the same number of times and the barrier is in
    // uniform control flow, which is what WGSL requires of one.
    for (var s = PIE_KLANES >> 1u; s > 0u; s = s >> 1u) {
        workgroupBarrier();
        if lane < s {
            partial[base + lane] = partial[base + lane] + partial[base + lane + s];
        }
    }
    workgroupBarrier();
    if lane == 0u {
        write_at(row, col, partial[base]);
    }
}

//#else

// ── THE TILE ARM ────────────────────────────────────────────────────────────

const PIE_BM = 32u;
const PIE_BN = 32u;
const PIE_BK = 32u;
// A lane owns FOUR columns, which is what lets both staged slabs be read as
// vec4 and keeps the inner product off workgroup memory's throughput ceiling.
const PIE_BK4 = PIE_BK / 4u;
const PIE_BN4 = PIE_BN / 4u;
// 32 x 2 x 2, the workgroup every tiled body in this tree declares.
const PIE_LANES = 128u;
// The row a lane's second accumulator sits at. Accumulators per lane are
// `BM * BN4 / LANES` = 2, and they are NAMED rather than an array:
// `qmm_t.wgsl` measured the array spelling and it LOST -- a dynamically indexed
// `array<vec4<f32>, N>` local is thread-local memory and not registers, and its
// (32, 32) point fell 1.68 -> 1.40 TFLOP/s for it.
const PIE_RSTEP = PIE_LANES / PIE_BN4;

var<workgroup> xs: array<vec4<f32>, PIE_BM * PIE_BK4>;
// Stored K-major (`ws[kk * BN4 + c4]`) though it is read column-major, so that
// consecutive lanes -- which hold consecutive `c4` -- touch consecutive words
// in the inner loop instead of striding by PIE_BK.
var<workgroup> ws: array<vec4<f32>, PIE_BK * PIE_BN4>;

// Four consecutive K of one output column. Reads past K happen in the tail
// block and are DISCARDED by the `select` at the call site; WGSL bounds them to
// the binding, so the worst one can be is the next column's data.
fn load_w_quad(col: u32, k0: u32) -> vec4<f32> {
    return vec4<f32>(
        load_w(col, k0 + 0u),
        load_w(col, k0 + 1u),
        load_w(col, k0 + 2u),
        load_w(col, k0 + 3u),
    );
}

// Two adjacent output columns in ONE store.
//
// `store_half` is a compare-exchange loop, and none of it is needed when one
// lane owns both halves of the word. This lane's four columns start at a
// multiple of four, so its two pairs are whole whenever the row's own base
// `row * n` is even -- which is every row when `n` is even and half of them
// when it is odd. The fallback is the CAS, taken on the overhang and on the odd
// pitch.
fn write_pair(row: u32, col: u32, v0: f32, v1: f32) {
    let n = u32(params.n);
    let at = row * n + col;
    if row >= u32(params.m) || col + 1u >= n || (at & 1u) != 0u {
        write_at(row, col, v0);
        write_at(row, col + 1u, v1);
        return;
    }
    atomicStore(&out_[at >> 1u], pie_f32_to_bf16(v0) | (pie_f32_to_bf16(v1) << 16u));
}

@compute @workgroup_size(32, 2, 2)
fn main(
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let tile_row = wg.y * PIE_BM;
    let tile_col = wg.x * PIE_BN;
    let k = u32(params.k);

    // (r0, c4) partitions the 32 x 32 block over 128 lanes exactly: 16 row
    // starts by 8 column quads, two rows apiece, no lane idle and no element
    // owned twice.
    let c4 = local % PIE_BN4;
    let r0 = local / PIE_BN4;

    var acc0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // `kb` and `k` come from the uniform block and from nothing else, so every
    // invocation runs this loop the same number of times and both barriers are
    // reached by all 128 of them. That is the whole discipline: no `return`, no
    // `continue`, no lane-dependent bound anywhere between here and the
    // epilogue.
    var kb = 0u;
    while kb < k {
        // K is NOT a whole number of PIE_BK blocks -- gemma's 5376 is not -- so
        // the tail is measured here and staged as ZERO below, which is what
        // lets the inner product run the full PIE_BK with no bound of its own.
        let kn = min(PIE_BK, k - kb);

        // Before overwriting the tiles, wait for the previous iteration's reads
        // of them to finish. On the first iteration this costs one barrier and
        // protects nothing, which is cheaper than proving it can be skipped.
        workgroupBarrier();
        for (var e = local; e < PIE_BM * PIE_BK4; e = e + PIE_LANES) {
            let r = e / PIE_BK4;
            let k4 = (e - r * PIE_BK4) * 4u;
            let row = tile_row + r;
            var v = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            if k4 + 0u < kn { v.x = load_x(row, kb + k4 + 0u); }
            if k4 + 1u < kn { v.y = load_x(row, kb + k4 + 1u); }
            if k4 + 2u < kn { v.z = load_x(row, kb + k4 + 2u); }
            if k4 + 3u < kn { v.w = load_x(row, kb + k4 + 3u); }
            xs[e] = v;
        }
        // The weight slab is read K-fast and written K-major, so one lane's
        // sixteen values become four `ws` entries at four consecutive `kk` and
        // one `c4`. Reading it K-fast is what makes the global load coalesce:
        // consecutive lanes hold consecutive `kk` of the SAME column.
        for (var e = local; e < PIE_BN4 * PIE_BK4; e = e + PIE_LANES) {
            let cc = e / PIE_BK4;
            let kk = (e - cc * PIE_BK4) * 4u;
            let col = tile_col + cc * 4u;
            let zero = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var q0 = zero;
            var q1 = zero;
            var q2 = zero;
            var q3 = zero;
            if kk < kn {
                q0 = load_w_quad(col + 0u, kb + kk);
                q1 = load_w_quad(col + 1u, kb + kk);
                q2 = load_w_quad(col + 2u, kb + kk);
                q3 = load_w_quad(col + 3u, kb + kk);
            }
            // K is not required to be a multiple of four, so the staged tail is
            // zeroed component-wise rather than by the block guard above.
            let base = kk * PIE_BN4 + cc;
            ws[base + 0u * PIE_BN4] = select(zero, vec4<f32>(q0.x, q1.x, q2.x, q3.x), kk + 0u < kn);
            ws[base + 1u * PIE_BN4] = select(zero, vec4<f32>(q0.y, q1.y, q2.y, q3.y), kk + 1u < kn);
            ws[base + 2u * PIE_BN4] = select(zero, vec4<f32>(q0.z, q1.z, q2.z, q3.z), kk + 2u < kn);
            ws[base + 3u * PIE_BN4] = select(zero, vec4<f32>(q0.w, q1.w, q2.w, q3.w), kk + 3u < kn);
        }
        workgroupBarrier();

        // BOTH ROWS THE LANE OWNS, OVER ONE COLUMN QUAD -- so the four `ws`
        // vec4s an inner step needs are read ONCE and retire both accumulators
        // instead of once per accumulator. Six workgroup vec4 reads retire
        // thirty-two multiplies.
        //
        // The terms stay separate `+` operands rather than a `dot`: the
        // accumulation order is the thing a parity walk against the sibling
        // backends compares.
        for (var k4 = 0u; k4 < PIE_BK4; k4 = k4 + 1u) {
            let kk = k4 * 4u;
            let w0 = ws[(kk + 0u) * PIE_BN4 + c4];
            let w1 = ws[(kk + 1u) * PIE_BN4 + c4];
            let w2 = ws[(kk + 2u) * PIE_BN4 + c4];
            let w3 = ws[(kk + 3u) * PIE_BN4 + c4];
            let x0 = xs[r0 * PIE_BK4 + k4];
            acc0 = acc0 + x0.x * w0 + x0.y * w1 + x0.z * w2 + x0.w * w3;
            let x1 = xs[(r0 + PIE_RSTEP) * PIE_BK4 + k4];
            acc1 = acc1 + x1.x * w0 + x1.y * w1 + x1.z * w2 + x1.w * w3;
        }
        kb = kb + PIE_BK;
    }

    let col = tile_col + c4 * 4u;
    let row0 = tile_row + r0;
    let row1 = row0 + PIE_RSTEP;
    write_pair(row0, col + 0u, acc0.x, acc0.y);
    write_pair(row0, col + 2u, acc0.z, acc0.w);
    write_pair(row1, col + 0u, acc1.x, acc1.y);
    write_pair(row1, col + 2u, acc1.z, acc1.w);
}

//#endif

// pie:instantiate dense_gemm_t_bfloat16
// pie:instantiate dense_gemv_t_bfloat16 PIE_GEMV=1
