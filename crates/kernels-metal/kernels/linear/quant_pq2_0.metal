#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

// PQ2_0 positional-2-bit decode-in-dot (Dtype::Pq2_0, spelled `g128_u2_f16_n`).
//
// A block is 34 bytes for 128 weights (2.125 bpw): a LEADING fp16 scale `d` at
// bytes 0-1 (not a trailing one, as PTQ1_0 has) + `qs[32]` holding positional
// 2-bit codes, 4 per byte. A weight is `value = (code - 1) * d`, code in
// {0,1,2,3} -> {-1,0,+1,+2}
// (an asymmetric quaternary with a `+2` outlier code, no zero-point). The scale
// is INLINE per block; there is NO separate scales plane. row_bytes =
// ceil(K/128) * 34.
//
// This mirrors the host oracle `checkpoint::codec::pq2_0::decode_block` byte for
// byte, which is itself bit-exact against the fork's own `dequantize_row_pq2_0`
// (PrismML-Eng llama.cpp, branch `prism`). Unlike PTQ1_0 there is NO base-3
// staging: element `j` lives in byte `j / 4` at bit offset `(j % 4) * 2`, so the
// layout is purely positional and decodes in natural element order 0..127. We
// decode a block the same way, inside the dot, so
// `y[row] = sum_i decoded_W[row][i] * x[i]`.
//
// We compute each decoded weight as `float(code - 1) * d` and then multiply by
// the activation, exactly as the host forms `(f32::from(code) - 1.0) * d`. With
// a one-hot activation only one term survives and the read-back is bit-exact
// against the host decoder (an integer in {-1,0,1,2} times a float is exact, and
// half->float is exact for every finite scale).

#define PQ2_0_BLOCK 128
#define PQ2_0_BLOCK_BYTES 34
#define PQ2_0_QS_OFFSET 2
#define SIMD_SIZE 32

// Dot one 34-byte block against the 128 activations at `xblk`, decoding weights
// positionally (element `j` -> byte `qs[j / 4]`, bit `(j % 4) * 2`). Returns
// `sum_j (code_j - 1) * d * x_j`.
template <typename Ti>
inline float pq2_0_block_dot(const device uint8_t* blk, const device Ti* xblk) {
  ushort dbits = ushort(uint(blk[0]) | (uint(blk[1]) << 8));
  half dh = as_type<half>(dbits);
  float d = float(dh);
  const device uint8_t* qs = blk + PQ2_0_QS_OFFSET;

  float acc = 0.0f;
  for (int j = 0; j < PQ2_0_BLOCK; j++) {
    int code = (int(qs[j >> 2]) >> ((j & 3) * 2)) & 0x03;
    float wv = float(code - 1) * d;
    acc += wv * float(xblk[j]);
  }
  return acc;
}

// Matrix-vector (and small-batch) decode-in-dot. Shape mirrors the affine
// `affine_qmv_fast` (and the PTQ1_0 `ptq1_0_qmv`): 2 simdgroups per threadgroup,
// 4 output rows per simdgroup, the 32 lanes split the K blocks and `simd_sum`
// reduces. Grid is `qmv_grid`: lanes [vecs*32, ceil(out/4), 1], group [32, 2, 1].
template <typename Ti, typename To>
METAL_FUNC void pq2_0_qmv_impl(
    const device uint8_t* w,
    const device Ti* x,
    device To* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;

  const int num_blocks = in_vec_size / PQ2_0_BLOCK;
  const int row_bytes = num_blocks * PQ2_0_BLOCK_BYTES;
  const int out_row = int(tid.y) * (num_simdgroups * results_per_simdgroup) +
      int(simd_gid) * results_per_simdgroup;

  const device Ti* x_vec = x + int(tid.x) * in_vec_size;

  float result[results_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int bk = int(simd_lid); bk < num_blocks; bk += SIMD_SIZE) {
    const device Ti* xblk = x_vec + bk * PQ2_0_BLOCK;
    for (int row = 0; row < results_per_simdgroup; row++) {
      int r = out_row + row;
      if (r >= out_vec_size) {
        continue;
      }
      const device uint8_t* blk = w + r * row_bytes + bk * PQ2_0_BLOCK_BYTES;
      result[row] += pq2_0_block_dot<Ti>(blk, xblk);
    }
  }

  device To* y_vec = y + int(tid.x) * out_vec_size + out_row;
  for (int row = 0; row < results_per_simdgroup; row++) {
    float v = simd_sum(result[row]);
    if (simd_lid == 0 && out_row + row < out_vec_size) {
      y_vec[row] = static_cast<To>(v);
    }
  }
}

template <typename Ti, typename To>
[[kernel]] void pq2_0_qmv(
    const device uint8_t* w          [[buffer(0)]],
    const device Ti* x               [[buffer(1)]],
    device To* y                     [[buffer(2)]],
    const constant int& in_vec_size  [[buffer(3)]],
    const constant int& out_vec_size [[buffer(4)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  pq2_0_qmv_impl<Ti, To>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

#define instantiate_pq2_0_qmv(name, itype, otype)                          \
  template [[host_name("pq2_0_qmv_" #name)]]                              \
  [[kernel]] void pq2_0_qmv<itype, otype>(                                 \
      const device uint8_t*, const device itype*, device otype*,           \
      const constant int&, const constant int&, uint3, uint, uint);

// Production path: bf16 activation in, bf16 out (pie's qwen_3 d27b decode).
instantiate_pq2_0_qmv(bfloat16, bfloat, bfloat)
// Bit-exact read-back path: bf16 activation in, f32 out (the M2b oracle test).
instantiate_pq2_0_qmv(bfloat16_f32, bfloat, float)

// ─────────────────────────────────────────────────────────────────────────
// PQ2_0 GEMM-tiled qmm (prefill path).
//
// The qmv kernel above reads the whole weight matrix once per activation
// vector: at m prefill rows its us/row is flat (no weight reuse), so prefill
// costs ~m x decode. This tiled qmm amortises each decoded weight tile across
// BM activation rows using the same MLX "steel" simdgroup-matrix machinery the
// affine `quant_qmm_t.metal` uses. The ONLY codec-specific piece is the weight
// loader `Pq2_0BlockLoader`, which decodes 34-byte positional-2-bit blocks into
// the `Ws` threadgroup tile as half; the activation loader, the BlockMMA and the
// store are the proven generic steel code. A tile walks BK=32 contraction
// columns per step (a quarter of a 128-weight block; 4 steps cross one block and
// its inline leading fp16 scale), BN output rows, BM activation rows.
//
// Decode matches `pq2_0_block_dot` / the host oracle byte for byte: element j of
// a block lives in code byte `j/4` at bit `(j%4)*2`, value `(code-1)*d`. The
// loader writes the 4 codes of a byte in positional order p=0..3, so a decoded
// weight lands at Ws column `(bj+i)*4 + p`, i.e. its true contraction index — the
// same index the activation loader places x at, so the MMA dots matching pairs.

#define MLX_MTL_CONST static constant constexpr const

template <int BM>
inline constexpr int pq2_0_qmm_wm() {
  return BM < 16 ? 1 : 2;
}

// The vendored `mlx_steel_transforms.metal` carries an MXFP4 block loader that
// references these helpers; `quant_qmm_t.metal` defines them before the same
// include. We do not use that loader, but the header must compile, so define
// them here too (verbatim from `quant_qmm_t.metal`).
constant float kMxfp4Lut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
inline float mxfp4_lo(uint8_t byte) { return kMxfp4Lut[byte & 0xf]; }
inline float mxfp4_hi(uint8_t byte) { return kMxfp4Lut[byte >> 4]; }
inline float mxfp4_block_scale(uint8_t code) {
  return code == 0xff ? NAN : metal::ldexp(1.0f, int(code) - 127);
}

#include "../third_party/mlx_steel_prelude.metal"
#include "../third_party/mlx_steel_transforms.metal"
#include "../third_party/mlx_steel_mma.metal"
#include "../third_party/mlx_steel_loader.metal"

// Decode a PQ2_0 weight tile into the `Ws` half tile. Mirrors the affine
// QuantizedBlockLoader's thread layout (pack_factor codes per byte, bi/bj thread
// map) but the scale is INLINE (leading each 34-byte block) rather than a
// separate plane, and the value is `(code-1)*d` rather than affine `s*q+bias`.
template <typename D, short BROWS, short BCOLS, short dst_ld, short tgp_size>
struct Pq2_0BlockLoader {
  static_assert(BCOLS == 32, "a PQ2_0 K-tile is 32 contraction columns");
  MLX_MTL_CONST short pack_factor = 4;  // 4 positional 2-bit codes per byte
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;    // 8 code bytes/sub
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short subs_per_block = PQ2_0_BLOCK / BCOLS;  // 4 sub-tiles/block

  const int row_bytes;  // bytes per weight row = (K/128)*34
  const short thread_idx;
  const short bi;  // output row within the BN tile this thread decodes
  const short bj;  // first code byte within the sub-tile this thread reads
  short sub_cnt;   // which sub-tile of the current block (0..subs_per_block-1)
  threadgroup D* dst;
  const device uint8_t* src;        // -> current code byte for (bi, sub, bj)
  const device uint8_t* scale_src;  // -> current block's leading fp16 for bi

  Pq2_0BlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup D* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : row_bytes((src_ld_ / PQ2_0_BLOCK) * PQ2_0_BLOCK_BYTES),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        sub_cnt(0),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * row_bytes + PQ2_0_QS_OFFSET + bj),
        scale_src(src_ + bi * row_bytes) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }
    const ushort dbits =
        ushort(uint(scale_src[0]) | (uint(scale_src[1]) << 8));
    const D d = D(as_type<half>(dbits));
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_reads; i++) {
      const uint8_t byte = src[i];
      dst[i * 4 + 0] = D(int(byte & 0x3) - 1) * d;
      dst[i * 4 + 1] = D(int((byte >> 2) & 0x3) - 1) * d;
      dst[i * 4 + 2] = D(int((byte >> 4) & 0x3) - 1) * d;
      dst[i * 4 + 3] = D(int((byte >> 6) & 0x3) - 1) * d;
    }
  }

  void next() {
    sub_cnt++;
    if (sub_cnt == subs_per_block) {
      // Cross into the next block: step past this sub's code bytes AND the next
      // block's leading fp16 scale, and advance the scale pointer one block.
      sub_cnt = 0;
      src += BCOLS_PACKED + PQ2_0_QS_OFFSET;
      scale_src += PQ2_0_BLOCK_BYTES;
    } else {
      src += BCOLS_PACKED;
    }
  }
};

// The tiled driver: load x (bf16 -> half) and the decoded W tile, MMA, store
// bf16. A copy of `quant_qmm_t.metal`'s `qmm_t_cast_loaded_impl` specialised to
// the no-bias / no-residual PQ2_0 case with our loader. `w` is W^T: BN output
// rows, each `row_bytes` of 34-byte blocks; `x` is M x K bf16; `y` is M x N bf16.
template <int BM, int BK, int BN, int WM = pq2_0_qmm_wm<BM>(), int WN = 2>
METAL_FUNC void pq2_0_qmm_t_impl(
    const device uint8_t* w,
    const device bfloat* x,
    device bfloat* y,
    threadgroup half* Xs,
    threadgroup half* Ws,
    const constant int& K,
    const constant int& N,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int BK_padded = BK + 16 / sizeof(half);
  using loader_x_t = mlx::steel::
      BlockLoaderCast<bfloat, half, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = Pq2_0BlockLoader<half, BN, BK, BK_padded, WM * WN * SIMD_SIZE>;
  using mma_t = mlx::steel::
      BlockMMA<half, bfloat, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;

  const int y_row = int(tid.y) * BM;
  const int y_col = int(tid.x) * BN;
  const int row_bytes = (K / PQ2_0_BLOCK) * PQ2_0_BLOCK_BYTES;

  loader_x_t loader_x(x + size_t(y_row) * size_t(K), K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(
      w + size_t(y_col) * size_t(row_bytes), K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_unsafe();
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    loader_x.next();
    loader_w.next();
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  device bfloat* yp = y + size_t(y_row) * size_t(N) + y_col;
  mma_op.store_result(yp, N);
}

template <int BM, int BK, int BN>
[[kernel]] void pq2_0_qmm_t(
    const device uint8_t* w          [[buffer(0)]],
    const device bfloat* x           [[buffer(1)]],
    device bfloat* y                 [[buffer(2)]],
    const constant int& in_vec_size  [[buffer(3)]],
    const constant int& out_vec_size [[buffer(4)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = BK + 16 / sizeof(half);
  threadgroup half Xs[BM * BK_padded];
  threadgroup half Ws[BN * BK_padded];
  pq2_0_qmm_t_impl<BM, BK, BN>(
      w, x, y, Xs, Ws, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Stamp one tile point: `PIE_STAMP_pq2_0_qmm("entry", bm, bn)` mints
// `pq2_0_qmm_t<bm, 32, bn>` under host_name `entry`. BK is fixed at 32.
#define PIE_STAMP_pq2_0_qmm(entry, bm, bn)                                     \
  template [[host_name(entry)]]                                                \
  [[kernel]] void pq2_0_qmm_t<bm, 32, bn>(                                     \
      const device uint8_t*, const device bfloat*, device bfloat*,             \
      const constant int&, const constant int&, uint3, uint, uint);
