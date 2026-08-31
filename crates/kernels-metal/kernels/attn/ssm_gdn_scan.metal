// ssm_gdn_scan.metal — the gated-delta prefill scan, in REGISTERS.
//
// # What this is, and what it is a port of
//
// `ssm_gated_delta.metal`'s `gated_delta_chunked` and this file compute the
// same recurrence over the same operands, in the same order down the tokens.
// They differ in where the state lives and in how the two per-token
// reductions are folded, and that difference is the whole file:
//
//                            gated_delta_chunked        this
//   threadgroup            one per (v_head, request)   one per four dv-row
//                                                      groups of one
//   threadgroups, 27B      48                          384
//   recurrent state        DEVICE memory, read and     REGISTERS, loaded once
//                          written twice per token     and stored once
//   per-token reduction    a threadgroup tree,         `simd_shuffle_xor`,
//                          ten barriers                no barrier at all
//
// The shape is `origin/dev`'s `gdn_core_recurrent_prefill`, whose
// `(lanes, rows)` axes that branch swept and whose answer this tree already
// records as `DeviceTuning::gdn_scan_lanes` and `gdn_scan_rows` — two
// constants that, until this file, no launch geometry read. What is NOT
// dev's is the interface: dev's core reads a pre-normalized q, k and gate
// plane that its own prep pass writes, and this tree's prep does not write
// one, so the normalization stays here where `gated_delta_chunked` does it.
//
// # THE MEASUREMENT THIS FILE EXISTS FOR
//
// `what_the_gated_delta_scan_costs` fires the scan alone at qwen3.6-27B's
// shape (16 key heads, 48 value heads, 128 wide apiece) and multiplies by the
// stack's 48 gated-delta layers:
//
//   T          128       256       512
//   chunked   33.7 ms   67.3 ms  134.3 ms   per layer
//   x48       1618 ms   3228 ms   6448 ms   per prefill
//
// against a whole 512-token prefill of 10663 ms measured by
// `throughput_probe`. **The scan is 60% of the fire.** It is exactly linear
// in the token count, which is what a serial walk with a fixed thread count
// looks like, and the thread count is the defect: 48 threadgroups of 128
// threads is two threadgroups a core on a 24-core M1 Max.
//
// The state traffic is the other half of it. `gated_delta_chunked` gives each
// thread one dv row and walks that row's whole `Dk` cell in device memory,
// twice a token, read-modify-write — 512 f32 accesses per thread per token,
// which over 48 heads, 512 tokens and 48 layers is around 600 GB of traffic
// for one prefill. Here the cell is 16 registers and the traffic is one load
// and one store for the whole scan.
//
// # WHAT THIS IS NOT: it is not bit-identical, and that is a reassociation
//
// Both per-token reductions sum `Dk` f32 terms, and the two kernels sum them
// in different orders — `gated_delta_chunked` walks one thread serially down
// 128 terms, this folds 32 lanes of four. Same terms, same values, different
// association, so the answers part in the last bits and the parting COMPOUNDS
// down the sequence, because the state a token leaves is the state the next
// token reads.
//
// That is a real cost and it is taken deliberately, under the owner's ruling
// that a much faster path may drift. `what_the_gated_delta_scan_costs`'s
// agreement arm is what keeps the drift a drift, and it prices it: over 512
// tokens of pseudo-random operands the residual is 5.2e-9 rms against the
// answer's own 2.8e-2 — 1.8e-5%, worst element 7.5e-8 — while 2.58 of the
// 3.15 million elements differ in SOME bit, which is what a reassociation
// looks like when it is only a reassociation. What the CHECKPOINT gates say
// about it is `four_bit_first_light` and `session_c_first_light`, which pin
// tokens against mlx and hold.
//
// # The axes, and why they are compile-time
//
// `LANES` lanes own one dv row and each holds `PER = Dk / LANES` of its cell;
// `VROWS` dv rows share those lanes, so a lane's live state is
// `VROWS x PER` floats. **All three are template arguments and `PER` is not
// derived at run time**, which is the difference between an array in
// registers and an array on the stack: a loop bounded by a `constant int&`
// cannot be unrolled, the array cannot be promoted, and the kernel then
// fetches its "registers" out of device memory once per term. That failure
// is measured elsewhere in this tree — `quant_qmv_rows.metal`'s header
// prices it at 6x — and it is the one this file cannot afford, since holding
// the state IS the point.
//
// The cost of compile-time `PER` is that a head width the stamp does not
// name is not served here at all. `attn::ssm` checks the width and falls
// back to `gated_delta_chunked`, which takes any.

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

/// A sum across the `LANES` lanes that own one dv row.
///
/// The xor tree stays inside the aligned `LANES`-wide slice, so the rows
/// sharing a simdgroup reduce independently and no barrier is needed —
/// a simdgroup executes in lockstep and `simd_shuffle_xor` is its own
/// synchronization. `origin/dev`'s `gdn_row_sum`, unchanged.
template <int LANES>
inline float gdn_row_sum(float v) {
  v += simd_shuffle_xor(v, 1u);
  if (LANES >= 4) v += simd_shuffle_xor(v, 2u);
  if (LANES >= 8) v += simd_shuffle_xor(v, 4u);
  if (LANES >= 16) v += simd_shuffle_xor(v, 8u);
  if (LANES >= 32) v += simd_shuffle_xor(v, 16u);
  return v;
}

/// One request's gated-delta scan, `VROWS` value rows to a lane group.
///
/// `pos.x` is the lane inside the group, `pos.y` the group of value rows,
/// `pos.z` the `(request, value head)` pair — `attn::ssm::gdn_scan_launch` is
/// where that packing is composed, and it is the only place it is spelled.
template <typename T, int LANES, int VROWS, int PER>
[[kernel]] void gated_delta_scan(
    const device T* qkv         [[buffer(0)]],
    const device int* indptr    [[buffer(1)]],
    const device float* gates   [[buffer(2)]],
    device float* rstate        [[buffer(3)]],
    const device uint* slots    [[buffer(4)]],
    device float* y             [[buffer(5)]],
    const constant int& k_heads [[buffer(6)]],
    const constant int& v_heads [[buffer(7)]],
    const constant int& k_dim   [[buffer(8)]],
    const constant int& v_dim   [[buffer(9)]],
    uint3 pos [[thread_position_in_grid]]) {
  const int lane = int(pos.x) % LANES;
  const int dv_base = int(pos.y) * VROWS;
  const int hv = int(pos.z) % v_heads;
  const int n = int(pos.z) / v_heads;

  const int begin = indptr[n];
  const int end = indptr[n + 1];
  if (end <= begin) {
    return;
  }

  const int dk = k_dim;
  const int dv = v_dim;
  const int hk = hv / (v_heads / k_heads);
  const size_t keys = size_t(k_heads) * size_t(dk);
  const size_t pitch = 2 * keys + size_t(v_heads) * size_t(dv);
  const float scale = 1.0f / metal::sqrt(float(dk));

  // The `VROWS` cells this lane group owns, in registers for the whole scan.
  device float* i_state =
      rstate +
      ((size_t(slots[begin]) * size_t(v_heads) + size_t(hv)) * size_t(dv) +
       size_t(dv_base)) *
          size_t(dk);
  float st[VROWS][PER];
  for (int v = 0; v < VROWS; ++v) {
    for (int i = 0; i < PER; ++i) {
      st[v][i] = i_state[size_t(v) * size_t(dk) + size_t(PER * lane + i)];
    }
  }

  for (int t = begin; t < end; ++t) {
    const size_t row = size_t(t) * pitch;
    const size_t qbase = row + size_t(hk) * size_t(dk);
    const size_t kbase = qbase + keys;
    const size_t vbase = row + 2 * keys + size_t(hv) * size_t(dv);

    // This lane's slice of q and k, and the two L2 norms folded across the
    // group. `gated_delta_chunked` computes the same two sums over the same
    // `dk` terms through a threadgroup tree; this is where the two kernels'
    // arithmetic first parts.
    float q[PER];
    float k[PER];
    float qs = 0.0f;
    float ks = 0.0f;
    for (int i = 0; i < PER; ++i) {
      const int d = PER * lane + i;
      const float qv = float(qkv[qbase + size_t(d)]);
      const float kv = float(qkv[kbase + size_t(d)]);
      q[i] = qv;
      k[i] = kv;
      qs += qv * qv;
      ks += kv * kv;
    }
    const float qinv = metal::rsqrt(gdn_row_sum<LANES>(qs) + 1e-6f) * scale;
    const float kinv = metal::rsqrt(gdn_row_sum<LANES>(ks) + 1e-6f);
    for (int i = 0; i < PER; ++i) {
      q[i] *= qinv;
      k[i] *= kinv;
    }

    const size_t fused = size_t(t) * 2 * size_t(v_heads) + size_t(hv);
    const float decay = metal::exp(gates[fused]);
    const float beta = gates[fused + size_t(v_heads)];
    const size_t out = (size_t(t) * size_t(v_heads) + size_t(hv)) * size_t(dv);

    // The decay and the read of the carried memory, all `VROWS` rows before
    // any of them is folded: the reductions are independent, so issuing them
    // together is what gives the simdgroup a second chain to interleave
    // against the first. That is `VROWS`'s whole argument and it is dev's.
    float kv_mem[VROWS];
    for (int v = 0; v < VROWS; ++v) {
      float acc = 0.0f;
      for (int i = 0; i < PER; ++i) {
        st[v][i] *= decay;
        acc += st[v][i] * k[i];
      }
      kv_mem[v] = gdn_row_sum<LANES>(acc);
    }
    for (int v = 0; v < VROWS; ++v) {
      const float delta =
          (float(qkv[vbase + size_t(dv_base + v)]) - kv_mem[v]) * beta;
      float sum = 0.0f;
      for (int i = 0; i < PER; ++i) {
        st[v][i] += k[i] * delta;
        sum += st[v][i] * q[i];
      }
      sum = gdn_row_sum<LANES>(sum);
      if (lane == 0) {
        y[out + size_t(dv_base + v)] = sum;
      }
    }
  }

  for (int v = 0; v < VROWS; ++v) {
    for (int i = 0; i < PER; ++i) {
      i_state[size_t(v) * size_t(dk) + size_t(PER * lane + i)] = st[v][i];
    }
  }
}

// Stamped rather than spelled out: the axis is `LANES x VROWS x PER` and a
// checkpoint fires one point of it. `attn::ssm` composes the invocation.
#define PIE_STAMP_gdn_scan(entry, lanes, vrows, per)                           \
  template [[host_name(entry)]]                                                \
  [[kernel]] void gated_delta_scan<bfloat, lanes, vrows, per>(                 \
      const device bfloat*, const device int*, const device float*,            \
      device float*, const device uint*, device float*,                        \
      const constant int&, const constant int&, const constant int&,           \
      const constant int&, uint3);
