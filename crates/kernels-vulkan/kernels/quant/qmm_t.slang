// Affine tiled GEMM projections, ported from `quant/qmm_t.metal`.
//
// The Metal sibling uses Apple's simdgroup matrix unit and MLX steel loaders.
// This portable body keeps the same affine packing, BM/BN entrypoint
// matrix, split-K two-pass contract, and bias/residual epilogues, but performs
// scalar fp32 multiply-accumulate inside each tile so it does not require an
// optional cooperative-matrix extension. `_fp16_precast` variants use an fp16
// storage round for the staged activation and fp32 math; they are not a native
// fp16-MMA path, and the distinction is intentional.
//
// Two capability tiers sit alongside that baseline now, both optional and
// both compiled as separate modules so a device without the extension they
// need still gets every entrypoint from the baseline above.
//
// `@coopmat` (`PIE_COOPMAT`) replaces the inner scalar multiply-accumulate
// of the main GEMM (the `#else` branch below -- not `PIE_REDUCE`, not
// `PIE_CAST_INPUT`) with `coopMatMulAdd`, at a fixed 16x16x16 shape. It is
// only worth adding where a 16-wide tile amortises: entrypoints whose BM
// and BN are both 32 or 64. The bm_16/bn_16 shapes do not get a coopmat
// module -- a missing tier just falls back to the scalar baseline, which
// is the whole point of tiers being additive rather than replacing
// anything. See the coopmat `main` below for how the dequant math is kept
// byte-for-byte identical to the scalar tier's `affine_value`/`load_x`, and
// why the accumulation is still done as an ordinary float add in shared
// memory rather than a persistent coopmat accumulator.
//
// `@fp16` (`PIE_FP16_MATH`) is for the `_fp16_precast` entrypoints only. The
// baseline's `_fp16_precast` path rounds the staged activation through fp16
// storage and then does the multiply in fp32 -- an emulation, as the
// comment above says. This tier does the multiply itself in a real
// `float16_t`, widening only the per-term product back to `float` before
// it is added into the running sum, so the sum itself stays fp32.

#include "common/bf16.slang"
#if !defined(PIE_REDUCE) && !defined(PIE_CAST_INPUT)
#include "common/affine.slang"
#endif
#if defined(PIE_COOPMAT)
using namespace linalg;
typealias MatA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias MatAcc = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
// A and B are `float16_t` and NOT `float`, which is not a performance choice:
// `VK_KHR_cooperative_matrix` promises no configuration at all, a device
// advertises a LIST, and a `coopmat` outside that list is undefined behaviour
// rather than a slow path. This tier declared fp32 A/B until a GPU run
// compared it against its own baseline -- an RTX 4090 offers no fp32 A/B
// config whatsoever, accepted the pipeline anyway, and returned -9.5 where the
// answer was -0.0618. `16x16x16` with fp16 A/B and an fp32 accumulator is on
// every list that has anything, and is what llama.cpp uses for the same
// reason.
#endif

/// The workgroup, stated once. Every walk in this file strides by
/// `PIE_THREADS`, and `PIE_LOCAL_X` is the row width the flat cast index is
/// built from; Slang has no workgroup-size builtin to read them back off.
#define PIE_LOCAL_X 32u
#define PIE_THREADS 128u

#if defined(PIE_CAST_INPUT)
PIE_BUFFER_RO(3, cast_in);
[[vk::binding(12)]] RWStructuredBuffer<uint16_t> half_out;
struct PushCast { int k; int n; int row_stride; int count; };
[[vk::push_constant]] ConstantBuffer<PushCast> pc;
#elif defined(PIE_REDUCE)
PIE_BUFFER_RW(4, reduce_y);
#if defined(PIE_PARTIAL_F32)
[[vk::binding(8)]] StructuredBuffer<float> partial_f32;
#else
PIE_BUFFER_RO(8, partial_bf16);
#endif
struct PushReduce { int k; int n; int row_stride; int split_k_partition_stride; int split_k; };
[[vk::push_constant]] ConstantBuffer<PushReduce> pc;
#else
[[vk::binding(0)]] StructuredBuffer<uint> w;
PIE_BUFFER_RO(1, scales);
PIE_BUFFER_RO(2, biases);
PIE_BUFFER_RO(3, x);
PIE_BUFFER_RW(4, y);
// `Extra`/`OutF32`/`OutBf16`/`HalfIn` used to sit at bindings 7/8/8/12,
// unconditionally declared regardless of which macros actually read them.
// Those numbers were transcribed from the Metal tree, which numbers every
// operand -- including the scalars that Vulkan moves to a push-constant
// block -- in one combined sequence. `kernels_vulkan::bindings` numbers
// only the buffer operands, densely from zero in the row's operand order,
// so binding 7 was never a descriptor the shell wrote; it just read
// whatever garbage sat at slot 7 of a 5- or 6-buffer set. A SPIR-V-level
// audit that compiles every variant and diffs its `OpDecorate Binding`
// set against the table's ABI caught this (`scripts/vulkan-kernel-audit.py
// --bindings`) across 54 entrypoints in this file. That mode is deleted --
// it read the table's half by running `examples/dump_layout.rs`, which went
// with the rest of `examples/` -- so the numbers below are now held by
// reading alone.
//
// `affine_qmm_t`'s row has 5 buffer operands (0..4) and `affine_qmm_t_residual`'s
// has 6, with `residual` at buffer 5 -- which was confirmed against the ABI
// the audit derived, and which `kernels_vulkan::bindings` still derives from
// the row in `src/quant.rs`.
// `Extra` doubles as that residual buffer and as the bias vector, so it is
// declared only when one of those flags is live, at binding 5, and a plain
// `affine_qmm_t` module (neither flag set) now declares nothing past 4, as
// its row requires.
//
// `OutF32`/`OutBf16`/`HalfIn` serve `_splitk`/`_fp16_precast` rows the table
// states no operands for at all (`kernels_vulkan::push_layout` returns nothing
// for them, and an UNSTATED row is what that means), so there is no ABI to
// check these two
// numbers against -- 6 and 7 are just the next dense slots after 5, not
// verified numbers. Gating each on the macro that actually reads it (rather
// than declaring all of them unconditionally, which is what created the
// out-of-range bindings in the first place) keeps every compiled variant's
// descriptor set exactly as wide as the buffers it uses.
#if defined(PIE_BIAS) || defined(PIE_RESIDUAL)
PIE_BUFFER_RO(5, extra);
#endif
#if defined(PIE_SPLITK)
#if defined(PIE_OUT_F32)
[[vk::binding(6)]] RWStructuredBuffer<float> out_f32;
#else
PIE_BUFFER_RW(6, out_bf16);
#endif
#endif
#if defined(PIE_FP16_PRECAST)
[[vk::binding(7)]] StructuredBuffer<uint16_t> half_in;
#endif
// Only the split-K variants read the last four, and a push block may not
// over-declare: it has to fit the pipeline layout's range, which is built
// from the row's scalars. `affine_qmm_t` states `k` and `n` and nothing else.
struct Push {
    int k;
    int n;
#if defined(PIE_STRIDED) || defined(PIE_SPLITK)
    int row_stride;
#endif
#if defined(PIE_SPLITK)
    int k_partition_size;
    int split_k_partition_stride;
    int split_k;
#endif
};
[[vk::push_constant]] ConstantBuffer<Push> pc;
#endif

#if defined(PIE_CAST_INPUT)
/// `f32tof16` is Slang's spelling of the same rounding `packHalf2x16` did,
/// and it returns the ONE half in the low sixteen bits rather than a packed
/// pair, so the mask GLSL needed is gone rather than forgotten.
uint16_t f32_to_f16_bits(float v) { return uint16_t(f32tof16(v) & 0xffffu); }
[numthreads(32, 2, 2)]
void main(uint3 thread: SV_DispatchThreadID) {
#if defined(PIE_STRIDED)
    uint col = thread.x;
    uint row = thread.y;
    if (col < uint(pc.k)) half_out[row * uint(pc.row_stride) + col] = f32_to_f16_bits(PIE_LOAD(cast_in[row * uint(pc.row_stride) + col]));
#else
    uint idx = thread.x + thread.y * gl_NumWorkGroups.x * PIE_LOCAL_X;
    if (idx < uint(pc.count)) half_out[idx] = f32_to_f16_bits(PIE_LOAD(cast_in[idx]));
#endif
}
#elif defined(PIE_REDUCE)
float partial_at(uint at) {
#if defined(PIE_PARTIAL_F32)
    return partial_f32[at];
#else
    return PIE_LOAD(partial_bf16[at]);
#endif
}
[numthreads(32, 2, 2)]
void main(uint3 thread: SV_DispatchThreadID) {
    uint col = thread.x;
    uint row = thread.y;
    if (col >= uint(pc.n)) return;
    uint base = row * uint(pc.n) + col;
    float acc = 0.0;
    for (uint s = 0u; s < uint(pc.split_k); s++) acc += partial_at(base + s * uint(pc.split_k_partition_stride));
    reduce_y[base] = PIE_STORE(acc);
}
#else
uint weight_words_per_row() { return uint(pc.k) * uint(PIE_BITS) / 32u; }
uint groups_per_row() { return uint(pc.k) / uint(PIE_GROUP); }
float half_bits_to_f32(uint16_t v) { return f16tof32(uint(v)); }
uint input_stride() {
#if defined(PIE_STRIDED)
    return uint(pc.row_stride);
#else
    return uint(pc.k);
#endif
}
uint output_stride() {
#if defined(PIE_STRIDED)
    return uint(pc.row_stride);
#else
    return uint(pc.n);
#endif
}
float load_x(uint row, uint kk) {
#if defined(PIE_FP16_PRECAST)
    return half_bits_to_f32(half_in[row * input_stride() + kk]);
#else
    return PIE_LOAD(x[row * input_stride() + kk]);
#endif
}
#if defined(PIE_FP16_MATH)
/// The true-fp16 stand-in for `load_x`: same bits, kept as `float16_t`
/// instead of being widened to `float` before the multiply. This is what
/// makes the `@fp16` tier different from the baseline's `_fp16_precast`
/// path, which widens first and does the whole multiply in fp32.
half load_x_fp16(uint row, uint kk) {
    return half(half_bits_to_f32(half_in[row * input_stride() + kk]));
}
#endif
float affine_value(uint col, uint kk) {
    uint word = w[col * weight_words_per_row() + kk / uint(PIE_CODES_PER_WORD)];
    float code = pie_affine_code(word, kk % uint(PIE_CODES_PER_WORD));
    uint g = kk / uint(PIE_GROUP);
    return code * PIE_LOAD(scales[col * groups_per_row() + g])
         + PIE_LOAD(biases[col * groups_per_row() + g]);
}
// A tile is BM x BN whatever the matrix is, so the last tile in each
// direction runs off the end of the real output. The column overhang is the
// dangerous one: the output is row-major, so a lane at `col >= n` does not
// write past the buffer, it writes over `(row + 1, col - n)` -- a live
// element of the NEXT row, with a value computed from weights that are
// themselves out of range. A GPU sweep over the whole `{16,32,64}^2` tile
// grid at `n = 47` caught this: every row after the first began with the
// zero that a robust out-of-range weight fetch produced, and the corruption
// was invisible at the tile-aligned shapes the earlier tests happened to use.
// `n` is in the push block, so the guard is exact and free.
//
// The ROW overhang cannot be guarded here, because no entrypoint's push block
// carries `m` -- the launch names the grid and nothing else. So the contract
// is that the output allocation is a whole number of `BM` rows; those extra
// rows are written with garbage and ignored. That is a real requirement on
// the caller, not an oversight to fix in the shader -- `qmm_t.metal` has the
// same overhang deliberately, calling MLX's `store_result` when a
// `store_result_safe` sits right beside it, so both backends want a
// tile-aligned output and this guard only makes the column half of that
// contract free to get wrong.
// `group_z` is the split-K partition, passed in rather than read off a
// builtin: Slang helpers in this tree do not read thread indices, and this
// one is called from two different `main`s.
void write_out(uint row, uint col, float acc, uint group_z) {
    if (col >= uint(pc.n)) return;
#if defined(PIE_BIAS)
    acc += PIE_LOAD(extra[col]);
#endif
#if defined(PIE_RESIDUAL)
    uint at = row * output_stride() + col;
    PIE_ACT q = PIE_STORE(acc);
    y[at] = PIE_STORE(PIE_LOAD(q) + PIE_LOAD(extra[at]));
#elif defined(PIE_SPLITK)
    uint at = row * uint(pc.n) + col + group_z * uint(pc.split_k_partition_stride);
#if defined(PIE_OUT_F32)
    out_f32[at] = acc;
#else
    out_bf16[at] = PIE_STORE(acc);
#endif
#else
    y[row * output_stride() + col] = PIE_STORE(acc);
#endif
}

#if defined(PIE_COOPMAT)

// The coopmat tier of the main GEMM (see the file header for which
// entrypoints get it and why). `affine_value`/`load_x` above are called
// exactly as the scalar tier calls them -- the dequant is not touched --
// they are just staged into a shared fp32 tile first so `coopMatLoad` has
// something it understands to read from.
//
// The workgroup is 128 invocations (32x2x2); a coopMatMulAdd call only
// spans one subgroup, and nothing here can assume a particular subgroup
// size or that the driver's subgroup partition lines up with the
// y/z dimensions. Splitting the BM/BN output tile's 16x16 sub-blocks
// across subgroups would need every invocation to agree on which
// sub-block "belongs" to its subgroup, which is exactly the kind of
// data-dependent branch a coopmat call cannot safely sit behind. Instead,
// every invocation in the workgroup walks the same sequence of sub-blocks
// and K-chunks, so every coopMatMulAdd call happens under whole-workgroup
// (and therefore whole-subgroup) uniform control flow; the redundancy
// across subgroups is the price of that safety, not a correctness
// requirement.
//
// Each K-chunk's 16x16 partial product is accumulated into `cm_psum` with
// an ordinary shared-memory float add rather than a persistent coopmat
// accumulator carried across the K loop, because a coopmat accumulator
// only lives for as long as its subgroup keeps calling into it and this
// tier does not assume anything about how many (mi, ni) sub-blocks a given
// subgroup will end up covering.
uint coopmat_bm_tiles() { return uint(PIE_BM) / 16u; }
uint coopmat_bn_tiles() { return uint(PIE_BN) / 16u; }

groupshared half cm_x[PIE_BM * 16];
groupshared half cm_w[PIE_BN * 16];
groupshared float cm_tmp[16 * 16];
groupshared float cm_psum[PIE_BM * PIE_BN];

[numthreads(32, 2, 2)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    uint flat = local.x + 32u * local.y + 64u * local.z;
    uint tile_row = group.y * uint(PIE_BM);
    uint tile_col = group.x * uint(PIE_BN);
#if defined(PIE_SPLITK)
    uint k0 = group.z * uint(pc.k_partition_size);
    uint k1 = min(uint(pc.k), k0 + uint(pc.k_partition_size));
#else
    uint k0 = 0u;
    uint k1 = uint(pc.k);
#endif

    for (uint e = flat; e < uint(PIE_BM * PIE_BN); e += PIE_THREADS) cm_psum[e] = 0.0;
    GroupMemoryBarrierWithGroupSync();

    for (uint kb = k0; kb < k1; kb += 16u) {
        uint kn = min(16u, k1 - kb);
        GroupMemoryBarrierWithGroupSync();
        for (uint e = flat; e < uint(PIE_BM * 16); e += PIE_THREADS) {
            uint r = e / 16u;
            uint kk = e - r * 16u;
            cm_x[e] = half((kk < kn) ? load_x(tile_row + r, kb + kk) : 0.0);
        }
        for (uint e = flat; e < uint(PIE_BN * 16); e += PIE_THREADS) {
            uint c = e / 16u;
            uint kk = e - c * 16u;
            cm_w[e] = half((kk < kn) ? affine_value(tile_col + c, kb + kk) : 0.0);
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint mi = 0u; mi < coopmat_bm_tiles(); mi++) {
            for (uint ni = 0u; ni < coopmat_bn_tiles(); ni++) {
                MatAcc acc = MatAcc(0.0);
                MatA matX = MatA.Load<CoopMatMatrixLayout.RowMajor>(cm_x, mi * 16u * 16u, 16u);
                MatB matW = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(cm_w, ni * 16u * 16u, 16u);
                acc = coopMatMulAdd<float, false>(matX, matW, acc);
                GroupMemoryBarrierWithGroupSync();
                acc.Store<CoopMatMatrixLayout.RowMajor>(cm_tmp, 0, 16);
                GroupMemoryBarrierWithGroupSync();
                for (uint e = flat; e < 256u; e += PIE_THREADS) {
                    uint r = e / 16u;
                    uint c = e - r * 16u;
                    cm_psum[(mi * 16u + r) * uint(PIE_BN) + (ni * 16u + c)] += cm_tmp[e];
                }
                GroupMemoryBarrierWithGroupSync();
            }
        }
    }

    for (uint idx = flat; idx < uint(PIE_BM * PIE_BN); idx += PIE_THREADS) {
        uint r = idx / uint(PIE_BN);
        uint c = idx - r * uint(PIE_BN);
        write_out(tile_row + r, tile_col + c, cm_psum[idx], group.z);
    }
}

#else

[numthreads(32, 2, 2)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    uint flat = local.x + 32u * local.y + 64u * local.z;
    uint tile_row = group.y * uint(PIE_BM);
    uint tile_col = group.x * uint(PIE_BN);
    for (uint idx = flat; idx < uint(PIE_BM * PIE_BN); idx += PIE_THREADS) {
        uint r = idx / uint(PIE_BN);
        uint c = idx - r * uint(PIE_BN);
        uint row = tile_row + r;
        uint col = tile_col + c;
#if defined(PIE_SPLITK)
        uint k0 = group.z * uint(pc.k_partition_size);
        uint k1 = min(uint(pc.k), k0 + uint(pc.k_partition_size));
#else
        uint k0 = 0u;
        uint k1 = uint(pc.k);
#endif
        float acc = 0.0;
#if defined(PIE_FP16_MATH)
        for (uint kk = k0; kk < k1; kk++) acc += float(load_x_fp16(row, kk) * half(affine_value(col, kk)));
#else
        for (uint kk = k0; kk < k1; kk++) acc += load_x(row, kk) * affine_value(col, kk);
#endif
        write_out(row, col, acc, group.z);
    }
}

#endif
#endif

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
// pie:instantiate cast_qmm_input_strided_bfloat16_to_float16 PIE_STRIDED=1 PIE_CAST_INPUT=1 PIE_STRIDED=1
// pie:instantiate qmm_splitk_reduce_bfloat16 PIE_SPLITK=1 PIE_REDUCE=1
// pie:instantiate qmm_splitk_reduce_f32_bfloat16 PIE_SPLITK=1 PIE_REDUCE=1 PIE_PARTIAL_F32=1

// Coopmat tier: the main GEMM path's bm_32/bm_64 x bn_32/bn_64 shapes,
// where a 16-wide coopMatMulAdd amortises. See the file header.
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_4_bm_32_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_4_bm_32_bn_64 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_4_bm_64_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_4_bm_64_bn_64 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_8_bm_32_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_8_bm_32_bn_64 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=64 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_8_bm_64_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_128_b_8_bm_64_bn_64 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=64 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_4_bm_32_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_4_bm_32_bn_64 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_4_bm_64_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_4_bm_64_bn_64 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_8_bm_32_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_8_bm_32_bn_64 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=64 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_8_bm_64_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_32_b_8_bm_64_bn_64 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=64 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_PROBE_SHAPE=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_PROBE_SHAPE=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_PROBE_SHAPE=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_PROBE_SHAPE=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=64 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_8_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bfloat16_gs_64_b_8_bm_64_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=64 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_32_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_32_bn_64 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_64_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_64_bn_64 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_32_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_32_bn_64 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=64 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_64_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_64_bn_64 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=64 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_32_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_32_bn_64 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_64_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_64_bn_64 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_32_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_32_bn_64 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=64 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_64_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_64_bn_64 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=64 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_32_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_64_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_32_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=64 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_64_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=64 PIE_BIAS=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_BIAS=1 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_BIAS=1 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_BIAS=1 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_BIAS=1 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_32_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_32_bn_64 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_64_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_64_bn_64 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_32_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_32_bn_64 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=64 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_64_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_64_bn_64 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=64 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_32_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_32_bn_64 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_64_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_64_bn_64 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_32_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_32_bn_64 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=64 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_64_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_64_bn_64 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=64 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_32_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_64_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_32_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=64 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_64_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=64 PIE_RESIDUAL=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_32_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_64_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_128_b_8_bm_32_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_128_b_8_bm_64_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_32_b_4_bm_32_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_32_b_4_bm_64_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_32_b_8_bm_32_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_32_b_8_bm_64_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_64_b_4_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_64_b_4_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_64_b_8_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_bfloat16_gs_64_b_8_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_128_b_4_bm_32_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_128_b_4_bm_64_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_128_b_8_bm_32_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_128_b_8_bm_64_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_32_b_4_bm_32_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_32_b_4_bm_64_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_32_b_8_bm_32_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_32_b_8_bm_64_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_64_b_4_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_64_b_4_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_64_b_8_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_f32_bfloat16_gs_64_b_8_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_128_b_4_bm_32_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_128_b_4_bm_64_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_128_b_8_bm_32_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_128_b_8_bm_64_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_32_b_4_bm_32_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_32_b_4_bm_64_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_32_b_8_bm_32_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_32_b_8_bm_64_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_64_b_4_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_64_b_4_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_64_b_8_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_bfloat16_gs_64_b_8_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_STRIDED=1 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_STRIDED=1 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_FP16_PRECAST=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_128_b_4_bm_32_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_128_b_4_bm_64_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_128_b_8_bm_32_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_128_b_8_bm_64_bn_32 @coopmat PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_32_b_4_bm_32_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_32_b_4_bm_64_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_32_b_8_bm_32_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_32_b_8_bm_64_bn_32 @coopmat PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_64_b_4_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_64_b_4_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_64_b_8_bm_32_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_strided_residual_bfloat16_gs_64_b_8_bm_64_bn_32 @coopmat PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_COOPMAT=1

// Fp16 tier: true float16_t multiply for the _fp16_precast entrypoints.
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_16 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_BIAS=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_BIAS=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=64 PIE_BIAS=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_16 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_BIAS=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_BIAS=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_BIAS=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_16 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_BIAS=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_BIAS=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_BIAS=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_16 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=64 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_16 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_16 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_16 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=64 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_16 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_16 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_RESIDUAL=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_SPLITK=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_16_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_32_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_64_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_SPLITK=1 PIE_SPLITK=1 PIE_OUT_F32=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_STRIDED=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_STRIDED=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_STRIDED=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_16_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_32_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
// pie:instantiate affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_64_bn_32 @fp16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_RESIDUAL=1 PIE_STRIDED=1 PIE_FP16_PRECAST=1 PIE_FP16_MATH=1
