// quant_qmv_rows.metal — the MULTI-ROW vector point: one threadgroup fetches a
// weight block ONCE and spends it on R activation rows.
//
// # What this file is for, and what it is NOT for
//
// `quant_qmv.metal`'s `affine_qmv_fast` is a one-row kernel launched R times
// side by side. This one folds R rows into a threadgroup. It was written on
// the hypothesis that the fold would collapse a four-lane decode toward the
// cost of a one-lane one, because four lanes read the same bank four times —
// and THAT HYPOTHESIS IS FALSE ON THIS MACHINE. The measurement is below,
// because a kernel that is kept for a smaller reason than it was written for
// should say so at the top.
//
// # The band, and what it actually costs
//
// The dense ladder in `linear::quant::act_x_wt` has two arms, and at two to
// four rows neither of them shares a weight read: the vector point is one row
// per threadgroup, and the tiled point's narrowest row block is eight, so a
// four-row fire pays for eight rows of dequantize. Measured on qwen3.6-27B,
// ms per warm decode fire (`throughput_probe`, 16 fires, M1 Max):
//
//   lanes        1       2       3       4       5       6       7       8
//   vector   62.19  107.51  159.33  207.70       —       —       —  266.63
//   tile         —       —       —       —  239.62  246.57  259.36  267.11
//
// The vector column is nearly linear in the row count, and 62.19 ms is
// 14.25 GiB of bank at 245 GB/s — which reads like a kernel repeating a
// read it could share. It is not.
//
// # THE READ WAS NEVER THE BILL
//
// `what_the_vector_point_is_bound_by` fires both points over a 72 MiB bank
// (past this machine's 48 MiB system cache, so every byte is DRAM) and then
// over a 4 MiB slice of the same bank (so every byte is cache), and scales the
// second back to the first. Per row, ms:
//
//   rows            1      2      4      8     16
//   one-row DRAM  0.359  0.215  0.211  0.208  0.207     — 365 GB/s at 16
//   one-row cache 0.472  0.321  0.266  0.237  0.223     — the read is free
//
// **Taking the memory traffic away does not make the kernel faster.** The
// one-row point is ARITHMETIC-bound: `x * (q & mask)` is a mask, an integer
// convert and an FMA for every four-bit code, three issue slots per multiply-
// accumulate, and no amount of sharing the fetch removes any of them. That
// the DRAM number and the ALU number land within a few percent of each other
// on this machine is a coincidence of the M1 Max's balance, and it is exactly
// the coincidence that makes the roofline argument for a fold look right.
//
// # What the fold IS worth, which is the reason this file exists
//
// A fold does not remove arithmetic, but it does remove the LOAD INSTRUCTIONS
// and the address and scale arithmetic that go with them: one pack fetch, one
// scale, one zero point per two rows instead of per row. That is around a
// tenth of the issue, and a tenth is what it measures. `throughput_probe`,
// production ladder, aggregate tok/s:
//
//                       N=1    N=2    N=3    N=4    N=5..8
//   qwen36-27b  before  16.1   18.6   18.8   19.3   unchanged
//               after   16.1   20.5   18.8   21.5   unchanged
//   gemma4-31b  before  16.9   18.9   19.3   19.5   unchanged
//               after   16.9   22.0   19.3   23.3   unchanged
//
// N=1 and N=3 are the CONTROLS: neither folds (see `qmv_rows_fold` — one row
// is not a fold, and three rows no rung divides), and both reproduce to the
// third digit.
//
// # The shape, and why it is not the deleted narrow kernel
//
// `quant_narrow.metal` (removed with the fast-ladder restore) was also a
// shared-read narrow point and it lost to the vector point at every width —
// 107.8 ms against 61.99 at one row on this checkpoint. Its defect was the k
// axis: one thread per output column walked the whole contraction serially,
// so the simdgroup's k split was gone.
//
// This kernel keeps the vector point's k split EXACTLY — thirty-two lanes
// each own a slice of the contraction and `simd_sum` folds them. The launch
// is the vector point's with its x extent divided by R:
//
//   threadgroups   M / R  x  out_vec_size / 8
//   threads        [32, 2, 1] per threadgroup, as `affine_qmv_fast`
//
// # THE REGISTER BUDGET IS THE DESIGN, and the first shape lost on it
//
// The obvious fold — hold R activation slices live and sweep one weight
// through them — was written first and measured. `R x values_per_thread`
// floats live across a k step is 64 registers at four rows and 128 at eight,
// which on this machine is past where a thread keeps its arrays in registers;
// the kernel then reads its "registers" out of device memory once per term.
// Over the 72 MiB bank that shape ran a four-row fold at 65 GB/s and an
// eight-row fold at 10, against the one-row point's 365. So the live set is
// INVERTED here: what is held across the row loop is the WEIGHT — four output
// rows' packs and their two factors — and the activation slice is one row's,
// loaded and spent inside the loop exactly as `affine_qmv_fast` loads it.
//
// # Bit identity with the one-row point — AT ONE PACK WIDTH, AND ONLY ONE
//
// This section used to promise identity outright. It does not hold, and the
// sentence cost two campaigns: `throughput_probe`'s
// `gemma4_26b_a4b_decodes_on_many_lanes_at_once` was triaged first as a
// dequantization width and then as a suspected window / page-table defect,
// and it was this file all along. `.wiki/macos-bench.md` §23.
//
// `qdot_staged` IS `quant_qmv.metal`'s `qdot` with its pack handed in rather
// than read in: same accumulator, same `scale * SUM + bias * sum` factoring,
// same `simd_sum` over the same thirty-two lanes. What the two files do not
// share is HOW K IS DEALT OUT to those lanes, because that is
//
//   block_size = pack_factor * packs_per_thread * SIMD_SIZE
//
// and `packs_per_thread` is a template parameter on both — fixed at 2 in
// `qmv_fast_impl` (its only stamp) and set from `DeviceTuning::qmv_rows_packs`
// here, which is 1 on the M1 Max. At four bits that is an eight-code slice of
// a 256-code block against a sixteen-code slice of a 512-code one: the same
// products, thirty-two different partial sums, a different fold.
//
// So row `r` of an R-row group lands `affine_qmv_fast`'s bits when this point
// is minted at TWO packs, and reassociates at one. The reassociation is one
// bfloat16 ulp in about one element in eight thousand — nothing on a dense
// stack, and a different expert on a checkpoint that routes a top-k on it.
// `affine_floor`'s `the_folded_vector_point_lands_the_one_row_bits` measures
// both sides and is the regression that keeps the distinction.
//
// # The tail is clamped on the read and guarded on the write
//
// `qmv_rows_fold` only offers a rung that divides the batch, so the tail is
// empty in practice. The guards are kept anyway — a clamp on the read, a
// predicate on the store — so that a caller which selected its own fold
// cannot walk off the end of the activation.
//
// # Why the three helpers below are copied and not included
//
// `get_pack_factor`, `get_bytes_per_pack` and `load_vector` are
// `quant_qmv.metal`'s, verbatim. The driver's include flattener would resolve
// `#include "quant_qmv.metal"` happily, but it emits a header WHOLE: pulling
// that file in would compile its thirty-odd instantiated points into every
// library minted for one stamp here. Thirty lines of duplication against that
// is the cheaper of the two, and the duplication is INERT — these three are
// format arithmetic over the packing mlx_lm ships, and a change to them is a
// change to the checkpoint format rather than to either kernel.

// # ONE UNROLL PRAGMA IS THE FOLD'S SPEED ON THIS MACHINE (M4 Pro, 2026-09)
//
// The compute-side `for (int r = 0; r < R; r++)` carries
// `#pragma clang loop unroll(full)`. Without it the backend leaves that loop
// rolled and the thread-local arrays (`pack`, `x_thread`, `result`) in
// memory, and a folded row costs ~4 issue slots a code where its arithmetic
// is under 2. Measured with `a_quantized_matmul_is_priced_by_its_rows`
// (K=5120 N=17408 4-bit, us a launch; one row = 207 at 215 GB/s):
//
//   rows            2     3     4     5     6
//   before        205   340   378   604   652     three one-row launches: 297; tile at bm=8: 345
//   r loop unrolled 208   227   286   344   413
//
// A three-row fire is 1.10x a one-row fire (was 1.44x, as three one-row
// launches) and a four-row fire 1.37x (was 1.55x); from five rows the tile
// is the arm. On this GPU matrix and vector work share one ALU budget (the
// tile at bm=8 sits at exactly dequant + MMA), so what a fold saves is issue
// slots, and the loads and sums a rolled row loop re-issues are the slots.
//
// THE PRAGMA GOES ON THAT LOOP AND NO OTHER. With it on the inner `row`
// loops as well, `r_4_p_2` at 2-bit lands a wrong answer in every column;
// with it on the store loops too, `r_5_p_2` at 4-bit does — deterministic,
// per shape, and gone when the pragma is: this OS's Metal compiler
// miscompiles those unrollings. `r_5_p_2` at 4-bit is wrong even with the
// one pragma here, so the pack-2 points are fenced to the rungs that were
// bit-checked before it (`QMV_ROW_RUNGS_PACK2` in `linear::quant`), and
// `every_folded_point_answers_the_one_row_point` sweeps every point the
// ladder can fire against the one-row point — equality at pack 2, one bf16
// ulp of reassociation at pack 1 — so the next such shape fails a test.

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

#define MLX_MTL_CONST static constant constexpr const
MLX_MTL_CONST int SIMD_SIZE = 32;

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}
template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(bits == 2 || bits == 4 || bits == 8,
                "port covers the widths mlx affine ships this box");
  U sum = 0;
  if (bits == 2) {
    // Verbatim `quant_qmv.metal`'s two-bit arm — eight codes a uint16, so the
    // activation is pre-divided by 4^j over each run of eight to cancel the
    // `<< 2j` the unshifted mask in `qdot_staged` leaves on code j.
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
      x_thread[i + 4] = x[i + 4] / 256.0f;
      x_thread[i + 5] = x[i + 5] / 1024.0f;
      x_thread[i + 6] = x[i + 6] / 4096.0f;
      x_thread[i + 7] = x[i + 7] / 16384.0f;
    }
  } else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  return sum;
}

/// `qdot` against a weight pack ALREADY IN REGISTERS.
///
/// `quant_qmv.metal`'s `qdot` reads its pack from device memory inside the
/// term loop, which is right for a kernel that visits each pack once. This
/// one is handed the pack, because the whole file exists to visit it once and
/// spend it R times.
///
/// The accumulation is term for term the one `qdot` performs — same order,
/// same `float` accumulator, same `scale * SUM + bias * sum` factoring — over
/// THIS point's `values_per_thread`, which is `pack_factor * packs_per_thread`
/// and is the one thing the two files can disagree about. A row folded here
/// lands the bits it lands alone when `packs_per_thread` is the one-row
/// point's 2; see the header.
template <typename U, int values_per_thread, int bits, int packs>
inline U qdot_staged(
    const thread uint16_t* p,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(bits == 2 || bits == 4 || bits == 8,
                "port covers the widths mlx affine ships this box");
  U accum = 0;
  if (bits == 2) {
    // The staged twin of `quant_qmv.metal`'s two-bit `qdot`: the pack is in
    // registers rather than device memory, but the eight masks and the term
    // order are the one arm's, so a row folded here lands the bits the one-row
    // point lands AT THE SAME PACK WIDTH (header).
    // `words_per_thread` is `values_per_thread / 8` here, which is
    // exactly the count this loop walks.
    for (int i = 0; i < (values_per_thread / 8); i++) {
      const uint16_t q = p[i];
      accum +=
          (x_thread[8 * i] * (q & 0x0003) + x_thread[8 * i + 1] * (q & 0x000c) +
           x_thread[8 * i + 2] * (q & 0x0030) + x_thread[8 * i + 3] * (q & 0x00c0) +
           x_thread[8 * i + 4] * (q & 0x0300) + x_thread[8 * i + 5] * (q & 0x0c00) +
           x_thread[8 * i + 6] * (q & 0x3000) + x_thread[8 * i + 7] * (q & 0xc000));
    }
  } else if (bits == 4) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      const uint16_t q = p[i];
      accum +=
          (x_thread[4 * i] * (q & 0x000f) + x_thread[4 * i + 1] * (q & 0x00f0) +
           x_thread[4 * i + 2] * (q & 0x0f00) + x_thread[4 * i + 3] * (q & 0xf000));
    }
  } else {
    const thread uint8_t* w = (const thread uint8_t*)p;
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }
  return scale * accum + sum * bias;
}

/// R rows of `y = x W^T` against an affine bank, one weight fetch.
///
/// `rows_per_group` is the fold and `packs_per_thread_` is the pack width.
/// Both reach the template from `linear::quant`, which is where the sweep
/// that chose them is written down; the file header above is where the
/// argument for the shape is. The one R-wide thing held across the row loop
/// is `result[4][R]`, four floats a row, which is what the answer IS and
/// cannot be smaller.
template <typename T, int group_size, int bits, int rows_per_group, int packs_per_thread_>
METAL_FUNC void qmv_rows_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& row_count,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int R = rows_per_group;
  constexpr int packs_per_thread = packs_per_thread_;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  // Sixteen-bit words in one thread's slice of one output row: the pack the
  // term loop walks, and the unit the staging array counts in.
  constexpr int words_per_thread = values_per_thread * bits / 16;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup][R];
  for (int row = 0; row < results_per_simdgroup; row++) {
    #pragma clang loop unroll(full)
    for (int r = 0; r < R; r++) {
      result[row][r] = 0;
    }
  }

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int row0 = int(tid.x) * R;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

  // One activation base plus R row offsets, rather than R pointers: an
  // offset is an int where a pointer is two registers.
  const device T* xb = x + simd_lid * values_per_thread;
  int roff[R];
  #pragma clang loop unroll(full)
  for (int r = 0; r < R; r++) {
    roff[r] = min(row0 + r, row_count - 1) * in_vec_size;
  }
  y += row0 * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    if (k + int(simd_lid) * values_per_thread < in_vec_size) {
      // THE ONE READ. Four output rows' packs and their two factors, into
      // registers, before any row of the batch is looked at.
      thread uint16_t pack[results_per_simdgroup][words_per_thread];
      thread U s[results_per_simdgroup];
      thread U b[results_per_simdgroup];
      for (int row = 0; row < results_per_simdgroup; row++) {
        const device uint16_t* wl =
            (const device uint16_t*)(ws + row * in_vec_size_w);
        for (int i = 0; i < words_per_thread; i++) {
          pack[row][i] = wl[i];
        }
        s[row] = scales[row * in_vec_size_g];
        b[row] = biases[row * in_vec_size_g];
      }
            // Unrolled, and this loop alone — the header says what that is worth
      // and what unrolling the others does.
#pragma clang loop unroll(full)
      for (int r = 0; r < R; r++) {
        U sum = load_vector<T, U, values_per_thread, bits>(xb + roff[r], x_thread);
        for (int row = 0; row < results_per_simdgroup; row++) {
          result[row][r] += qdot_staged<U, values_per_thread, bits, packs_per_thread>(
              pack[row], x_thread, s[row], b[row], sum);
        }
      }
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    xb += block_size;
  }

  // `simd_sum` is a whole-simdgroup fold, so it stands OUTSIDE every
  // predicate; only the store is guarded, and it is guarded on both axes —
  // the output width (as the one-row point guards it) and the batch, which
  // is where a group's padded rows stop.
  for (int row = 0; row < results_per_simdgroup; row++) {
    for (int r = 0; r < R; r++) {
      U v = simd_sum(result[row][r]);
      if (simd_lid == 0 && out_row + row < out_vec_size && row0 + r < row_count) {
        y[r * out_vec_size + row] = static_cast<T>(v);
      }
    }
  }
}

template <typename T, int group_size, int bits, int rows_per_group, int packs_per_thread>
[[kernel]] void affine_qmv_rows(
    const device uint32_t* w   [[buffer(0)]],
    const device T* scales     [[buffer(1)]],
    const device T* biases     [[buffer(2)]],
    const device T* x          [[buffer(3)]],
    device T* y                [[buffer(4)]],
    const constant int& in_vec_size  [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& row_count    [[buffer(7)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  qmv_rows_impl<T, group_size, bits, rows_per_group, packs_per_thread>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, row_count, tid,
      simd_gid, simd_lid);
}

// The jit stamp, for `quant_qmm_t.metal`'s reason: the axis here is
// (group, bits) x fold x pack, and a source that spelled every product out
// would compile thirty-six points into a library that fires one.
#define PIE_STAMP_qmv_rows(entry, gs, b, r, p)                                 \
  template [[host_name(entry)]]                                                \
  [[kernel]] void affine_qmv_rows<bfloat, gs, b, r, p>(                        \
      const device uint32_t*, const device bfloat*, const device bfloat*,      \
      const device bfloat*, device bfloat*, const constant int&,               \
      const constant int&, const constant int&, uint3, uint, uint);
