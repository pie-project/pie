#include <metal_stdlib>

using namespace metal;

constant constexpr uint kRouterMaxTopK = 16;
constant constexpr uint kRouterMaxSimdgroups = 32;

template <typename T, typename W, bool SCALED>
[[kernel]] void router_topk(
    const device T* logits     [[buffer(0)]],
    device int* expert_ids     [[buffer(1)]],
    device W* expert_weights   [[buffer(2)]],

    const device T* per_expert_scale [[buffer(3)]],

    const constant uint& n_experts         [[buffer(4)]],
    const constant uint& experts_per_token [[buffer(5)]],
    const constant uint& softmax_over_all  [[buffer(6)]],
    const constant uint& logits_pitch      [[buffer(7)]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint3 tgsize [[threads_per_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  const uint lid = lid3.x;
  const uint n = n_experts;
  const uint k = min(experts_per_token, kRouterMaxTopK);
  const uint n_simd = min((tgsize.x + 31u) / 32u, kRouterMaxSimdgroups);
  constexpr float NEG_INF = -3.0e38f;

  const uint row = tgid.y;
  logits += size_t(row) * size_t(logits_pitch != 0u ? logits_pitch : n);
  expert_ids += size_t(row) * size_t(k);
  expert_weights += size_t(row) * size_t(k);

  float v = lid < n ? float(logits[lid]) : NEG_INF;

  threadgroup float part_v[kRouterMaxSimdgroups];
  threadgroup float part_s[kRouterMaxSimdgroups];
  threadgroup float all_max;
  threadgroup float all_sum;
  threadgroup uint part_i[kRouterMaxSimdgroups];
  threadgroup float chosen[kRouterMaxTopK];
  threadgroup uint winner_of_round;

  if (softmax_over_all != 0u) {
    const float m0 = simd_max(v);
    if (simd_lid == 0) part_v[simd_gid] = m0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
      float best = NEG_INF;
      for (uint sg = 0; sg < n_simd; ++sg) best = max(best, part_v[sg]);
      all_max = best;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float e = lid < n ? fast::exp(v - all_max) : 0.0f;
    const float s0 = simd_sum(e);
    if (simd_lid == 0) part_s[simd_gid] = s0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
      float total = 0.0f;
      for (uint sg = 0; sg < n_simd; ++sg) total += part_s[sg];
      all_sum = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint r = 0; r < k; ++r) {
    const float m = simd_max(v);
    const uint w = simd_min(v == m ? lid : 0xFFFFFFFFu);
    if (simd_lid == 0) {
      part_v[simd_gid] = m;
      part_i[simd_gid] = w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
      float best = NEG_INF;
      uint best_i = 0xFFFFFFFFu;
      for (uint sg = 0; sg < n_simd; ++sg) {
        if (part_v[sg] > best) {
          best = part_v[sg];
          best_i = part_i[sg];
        }
      }
      expert_ids[r] = int(best_i);
      chosen[r] = best;
      winner_of_round = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == winner_of_round) v = NEG_INF;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0) {

    float mx = all_max;
    float sum = all_sum;
    if (softmax_over_all == 0u) {
      mx = NEG_INF;
      for (uint r = 0; r < k; ++r) mx = max(mx, chosen[r]);
      sum = 0.0f;
      for (uint r = 0; r < k; ++r) sum += fast::exp(chosen[r] - mx);
    }
    for (uint r = 0; r < k; ++r) {
      chosen[r] = fast::exp(chosen[r] - mx);
    }
    for (uint r = 0; r < k; ++r) {
      float weight = chosen[r] / sum;
      if (SCALED) weight *= float(per_expert_scale[uint(expert_ids[r])]);
      expert_weights[r] = static_cast<W>(weight);
    }
  }
}

#define instantiate_router_topk(name, itype, wtype)                \
  template [[host_name("router_topk_" #name)]]                     \
  [[kernel]] void router_topk<itype, wtype, false>(                \
      const device itype*, device int*, device wtype*,             \
      const device itype*,                                         \
      const constant uint&, const constant uint&,                  \
      const constant uint&, const constant uint&,                  \
      uint3, uint, uint, uint3, uint3);                            \
  template [[host_name("router_topk_scaled_" #name)]]              \
  [[kernel]] void router_topk<itype, wtype, true>(                 \
      const device itype*, device int*, device wtype*,             \
      const device itype*,                                         \
      const constant uint&, const constant uint&,                  \
      const constant uint&, const constant uint&,                  \
      uint3, uint, uint, uint3, uint3);

instantiate_router_topk(bfloat16, bfloat, bfloat)

instantiate_router_topk(f32w_bfloat16, bfloat, float)

inline float router_sigmoid(float x) {
  return 1.0f / (1.0f + metal::exp(-x));
}

[[kernel]] void router_topk_sigmoid(
    const device bfloat* logits    [[buffer(0)]],
    device int* expert_ids         [[buffer(1)]],
    device float* expert_weights   [[buffer(2)]],
    const constant uint& n_experts         [[buffer(3)]],
    const constant uint& experts_per_token [[buffer(4)]],
    const constant uint& renormalize       [[buffer(5)]],
    const constant float& scaling          [[buffer(6)]],
    uint3 lid3    [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint3 tgsize  [[threads_per_threadgroup]],
    uint3 tgid    [[threadgroup_position_in_grid]]) {
  const uint lid = lid3.x;
  const uint n = n_experts;
  const uint k = min(experts_per_token, kRouterMaxTopK);
  const uint picks = min(k, n);
  const uint n_simd = min((tgsize.x + 31u) / 32u, kRouterMaxSimdgroups);
  constexpr float NEG_INF = -3.0e38f;

  const uint row = tgid.y;
  logits += size_t(row) * size_t(n);
  expert_ids += size_t(row) * size_t(k);
  expert_weights += size_t(row) * size_t(k);

  float v = lid < n ? router_sigmoid(float(logits[lid])) : NEG_INF;

  threadgroup float part_v[kRouterMaxSimdgroups];
  threadgroup uint part_i[kRouterMaxSimdgroups];
  threadgroup uint winner_of_round;

  for (uint r = 0; r < picks; ++r) {
    const float m = simd_max(v);
    const uint w = simd_min(v == m ? lid : 0xFFFFFFFFu);
    if (simd_lid == 0) {
      part_v[simd_gid] = m;
      part_i[simd_gid] = w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
      float best = NEG_INF;
      uint best_i = 0u;
      for (uint sg = 0; sg < n_simd; ++sg) {
        if (part_v[sg] > best) {
          best = part_v[sg];
          best_i = part_i[sg];
        }
      }
      expert_ids[r] = int(best_i);
      winner_of_round = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == winner_of_round) v = NEG_INF;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0) {
    float sum = 0.0f;
    for (uint r = 0; r < picks; ++r) {
      const float w = router_sigmoid(float(logits[uint(expert_ids[r])]));
      expert_weights[r] = w;
      sum += w;
    }
    for (uint r = picks; r < k; ++r) {
      expert_ids[r] = 0;
      expert_weights[r] = 0.0f;
    }
    const float scale = (renormalize != 0u && sum > 0.0f) ? scaling / sum : scaling;
    for (uint r = 0; r < k; ++r) expert_weights[r] *= scale;
  }
}

inline float sqrt_softplus(float x) {
  const float sp = x > 20.0f ? x : metal::log(1.0f + metal::exp(x));
  return metal::sqrt(max(sp, 0.0f));
}

[[kernel]] void router_topk_sqrt_softplus(
    const device bfloat* logits    [[buffer(0)]],
    const device float* correction [[buffer(1)]],
    device int* expert_ids         [[buffer(2)]],
    device float* expert_weights   [[buffer(3)]],
    const constant uint& n_experts         [[buffer(4)]],
    const constant uint& experts_per_token [[buffer(5)]],
    const constant uint& renormalize       [[buffer(6)]],
    const constant float& scaling          [[buffer(7)]],
    uint3 lid3    [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint3 tgsize  [[threads_per_threadgroup]],
    uint3 tgid    [[threadgroup_position_in_grid]]) {
  const uint lid = lid3.x;
  const uint n = n_experts;
  const uint k = min(experts_per_token, kRouterMaxTopK);
  const uint picks = min(k, n);
  const uint n_simd = min((tgsize.x + 31u) / 32u, kRouterMaxSimdgroups);
  constexpr float NEG_INF = -3.0e38f;

  const uint row = tgid.y;
  logits += size_t(row) * size_t(n);
  expert_ids += size_t(row) * size_t(k);
  expert_weights += size_t(row) * size_t(k);

  float v = lid < n ? sqrt_softplus(float(logits[lid])) + correction[lid] : NEG_INF;

  threadgroup float part_v[kRouterMaxSimdgroups];
  threadgroup uint part_i[kRouterMaxSimdgroups];
  threadgroup uint winner_of_round;

  for (uint r = 0; r < picks; ++r) {
    const float m = simd_max(v);
    const uint w = simd_min(v == m ? lid : 0xFFFFFFFFu);
    if (simd_lid == 0) {
      part_v[simd_gid] = m;
      part_i[simd_gid] = w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
      float best = NEG_INF;
      uint best_i = 0u;
      for (uint sg = 0; sg < n_simd; ++sg) {
        if (part_v[sg] > best) {
          best = part_v[sg];
          best_i = part_i[sg];
        }
      }
      expert_ids[r] = int(best_i);
      winner_of_round = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == winner_of_round) v = NEG_INF;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0) {
    float sum = 0.0f;
    for (uint r = 0; r < picks; ++r) {

      const float w = sqrt_softplus(float(logits[uint(expert_ids[r])]));
      expert_weights[r] = w;
      sum += w;
    }
    for (uint r = picks; r < k; ++r) {
      expert_ids[r] = 0;
      expert_weights[r] = 0.0f;
    }
    const float scale = (renormalize != 0u && sum > 0.0f) ? scaling / sum : scaling;
    for (uint r = 0; r < k; ++r) expert_weights[r] *= scale;
  }
}

/* The hash router's gather: layers 0..num_hash_layers route by a per-token
 * LOOKUP, not a learned gate. `tid2eid [vocab, top_k]` (I64) names `top_k`
 * expert ids for every token id; this reads the row the token id selects and
 * lays down UNIFORM weights, so its (expert_ids, expert_weights) are the same
 * pair `router_topk` writes above -- `int` ids and `float` weights, row-major
 * at `top_k` per token -- and drop straight into the `route_sort` /
 * `expert_combine` path with nothing between.
 *
 * One thread per (token row, slot).
 *
 * THE TABLE IS I64 AND THE ROUTES ARE I32, which is not a narrowing this op
 * gets to refuse: `tid2eid` is a lookup, not a weight-representation dtype the
 * trace can intern, and everything downstream -- `route_sort`,
 * `expert_bias_combine` -- already reads an expert id as `int`. An expert
 * count never approaches 2^31, so the id is read at 64 bits where the table
 * spells it and written at 32 where the path consumes it, in the one place
 * the two planes meet.
 *
 * THE TOKEN ID IS `uint` AND OUT-OF-RANGE FALLS TO ROW 0, exactly as
 * `embed.metal` reads it: a non-negative i32 and a u32 are the same bits, so
 * the id stream a shell hands this and the one it hands the embed gather need
 * not disagree, and a token id at the vocab boundary reads the last table row
 * rather than off the end. A row that names the same expert twice is copied
 * as-is -- the hash may repeat, and the uniform fold weights every slot
 * alike.
 */
[[kernel]] void hash_route_gather(
    const device uint* token_ids   [[buffer(0)]],
    const device long* tid2eid     [[buffer(1)]],
    device int* expert_ids         [[buffer(2)]],
    device float* expert_weights   [[buffer(3)]],
    const constant uint& vocab             [[buffer(4)]],
    const constant uint& experts_per_token [[buffer(5)]],
    uint2 gid                      [[thread_position_in_grid]]) {
  const uint k = experts_per_token;
  const uint slot = gid.x;
  const uint row = gid.y;
  if (slot >= k) return;
  const uint raw = token_ids[row];
  const uint tid = (vocab > 0u && raw < vocab) ? raw : 0u;
  const size_t at = size_t(row) * size_t(k) + size_t(slot);
  expert_ids[at] = int(tid2eid[size_t(tid) * size_t(k) + size_t(slot)]);
  expert_weights[at] = 1.0f / float(k);
}

constant constexpr uint kMaxExperts = 1024;

[[kernel]] void route_sort(
    const device int* expert_ids [[buffer(0)]],
    device int* perm            [[buffer(1)]],
    device int* row_expert      [[buffer(2)]],
    device int* tile_expert     [[buffer(3)]],

    device int* inv             [[buffer(4)]],

    const constant uint& n                 [[buffer(5)]],
    const constant uint& n_experts         [[buffer(6)]],
    const constant uint& experts_per_token [[buffer(7)]],
    const constant uint& tile_rows         [[buffer(8)]],
    const constant uint& padded            [[buffer(9)]],
    const constant uint& width             [[buffer(10)]],
    const constant uint& x_pitch           [[buffer(11)]],
    uint lid                    [[thread_position_in_threadgroup]],
    uint nthreads               [[threads_per_threadgroup]]) {
    threadgroup atomic_uint counts[kMaxExperts];
    threadgroup uint base[kMaxExperts];
    threadgroup uint sg_sum[32];

    const uint E = min(n_experts, kMaxExperts);
    const uint tile = tile_rows < 1u ? 1u : tile_rows;
    const uint tiles = padded / tile;

    for (uint e = lid; e < E; e += nthreads) atomic_store_explicit(&counts[e], 0u, memory_order_relaxed);
    for (uint i = lid; i < padded; i += nthreads) {
        perm[i] = -1;
        row_expert[i] = 0;
    }
    for (uint t = lid; t < tiles; t += nthreads) tile_expert[t] = -1;
    for (uint i = lid; i < n; i += nthreads) inv[i] = -1;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (uint i = lid; i < n; i += nthreads) {
        const int e = expert_ids[i];
        if (e >= 0 && uint(e) < E) {
            atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint span = lid < E
        ? (atomic_load_explicit(&counts[lid], memory_order_relaxed) > 0u
               ? ((atomic_load_explicit(&counts[lid], memory_order_relaxed) + tile - 1u) / tile) * tile
               : 0u)
        : 0u;
    const uint within = simd_prefix_exclusive_sum(span);
    const uint sg = lid / 32u;
    const uint n_sg = (nthreads + 31u) / 32u;

    const uint sg_total = simd_sum(span);
    if (lid % 32u == 0u) sg_sum[sg] = sg_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        uint at = 0;
        for (uint i = 0; i < n_sg && i < 32u; ++i) {
            const uint t = sg_sum[i];
            sg_sum[i] = at;
            at += t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < E) {
        const uint at = sg_sum[sg] + within;
        base[lid] = at;
        for (uint t = at / tile; t < (at + span) / tile && t < tiles; ++t) {
            tile_expert[t] = int(lid);
        }

        atomic_store_explicit(&counts[lid], 0u, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < n; i += nthreads) {
        const int e = expert_ids[i];
        if (e < 0 || uint(e) >= E) continue;
        const uint at = base[e] + atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
        if (at < padded) {
            perm[at] = int(i);
            row_expert[at] = e;
            inv[i] = int(at);
        }
    }
}

[[kernel]] void route_gather(
    const device bfloat* x     [[buffer(0)]],
    device bfloat* out         [[buffer(1)]],
    const device int* perm     [[buffer(2)]],

    const constant uint& n                 [[buffer(3)]],
    const constant uint& n_experts         [[buffer(4)]],
    const constant uint& experts_per_token [[buffer(5)]],
    const constant uint& tile_rows         [[buffer(6)]],
    const constant uint& padded            [[buffer(7)]],
    const constant uint& width             [[buffer(8)]],
    const constant uint& x_pitch           [[buffer(9)]],
    uint2 gid                  [[thread_position_in_grid]]) {
    if (gid.x >= width || gid.y >= padded) return;
    const int sel = perm[gid.y];
    const uint k = experts_per_token < 1u ? 1u : experts_per_token;

    const uint pitch = x_pitch != 0u ? x_pitch : width;
    out[uint(gid.y) * width + gid.x] =
        sel < 0 ? bfloat(0) : x[(uint(sel) / k) * pitch + gid.x];
}

/* The inverse of `route_gather`: sorted rows back into ROUTE order.
 *
 * `combine_sorted` beside this one weights and folds in the same pass, which
 * is what the reference driver's mixture does because its dataflow owns both
 * halves. This plane's IR does not: `linear.moe_matmul_select_*` lands a
 * result of `tokens * top_k` rows and `linear.moe_weighted_sum` folds it,
 * two statements a dispatch arm cannot merge. So the batched arm undoes its
 * own permutation and hands the fold the rectangle it was promised.
 *
 * The inverse costs nothing to compute -- `route_sort` writes `inv` as it
 * places each pair -- and one elementwise pass over `n_pairs x width` against
 * a GEMM that reads every expert's slice once is not the term that decides
 * anything.
 */
[[kernel]] void route_scatter(
    const device bfloat* sorted [[buffer(0)]],
    device bfloat* out         [[buffer(1)]],
    const device int* inv      [[buffer(2)]],

    const constant uint& rows      [[buffer(3)]],
    const constant uint& width     [[buffer(4)]],
    const constant uint& out_pitch [[buffer(5)]],
    uint2 gid                  [[thread_position_in_grid]]) {
    if (gid.x >= width || gid.y >= rows) return;
    const int at = inv[gid.y];
    const uint pitch = out_pitch != 0u ? out_pitch : width;
    out[uint(gid.y) * pitch + gid.x] =
        at < 0 ? bfloat(0) : sorted[uint(at) * width + gid.x];
}

[[kernel]] void expert_combine(
    const device bfloat* y             [[buffer(0)]],
    const device float* expert_weights [[buffer(1)]],
    device bfloat* out                 [[buffer(2)]],
    const constant uint& width             [[buffer(3)]],
    const constant uint& experts_per_token [[buffer(4)]],
    uint2 gid                          [[thread_position_in_grid]]) {
  const uint c = gid.x;
  const uint row = gid.y;
  const uint k = experts_per_token;
  const size_t base = size_t(row) * size_t(k);
  float acc = 0.0f;
  for (uint e = 0; e < k; ++e) {
    const size_t at = base + size_t(e);
    acc += expert_weights[at] * float(y[at * size_t(width) + size_t(c)]);
  }
  out[size_t(row) * size_t(width) + size_t(c)] = static_cast<bfloat>(acc);
}

[[kernel]] void expert_bias_combine(
    const device bfloat* x             [[buffer(0)]],
    const device bfloat* bias          [[buffer(1)]],
    const device int* expert_ids       [[buffer(2)]],
    const device float* expert_weights [[buffer(3)]],
    device bfloat* out                 [[buffer(4)]],
    const constant uint& width             [[buffer(5)]],
    const constant uint& experts_per_token [[buffer(6)]],
    uint2 gid                          [[thread_position_in_grid]]) {
  const uint c = gid.x;
  if (c >= width) return;
  const uint row = gid.y;
  const uint k = experts_per_token;
  const size_t at = size_t(row) * size_t(width) + size_t(c);
  const size_t base = size_t(row) * size_t(k);
  float acc = float(x[at]);
  for (uint e = 0; e < k; ++e) {
    const int expert = expert_ids[base + size_t(e)];
    if (expert < 0) continue;
    acc += expert_weights[base + size_t(e)] *
           float(bias[size_t(uint(expert)) * size_t(width) + size_t(c)]);
  }
  out[at] = static_cast<bfloat>(acc);
}

[[kernel]] void combine_sorted(
    const device bfloat* y              [[buffer(0)]],
    const device bfloat* expert_weights [[buffer(1)]],
    device bfloat* out                  [[buffer(2)]],

    const device int* inv               [[buffer(3)]],

    const constant uint& width             [[buffer(4)]],
    const constant uint& experts_per_token [[buffer(5)]],
    const constant uint& out_pitch         [[buffer(6)]],
    uint2 gid                           [[thread_position_in_grid]]) {
  const uint c = gid.x;
  if (c >= width) return;
  const uint row = gid.y;
  const uint k = experts_per_token;
  float acc = 0;
  for (uint e = 0; e < k; ++e) {
    const int at = inv[row * k + e];
    if (at < 0) continue;
    acc += float(expert_weights[row * k + e]) * float(y[uint(at) * width + c]);
  }
  out[row * (out_pitch != 0u ? out_pitch : width) + c] = static_cast<bfloat>(acc);
}

[[kernel]] void shared_expert_combine_strided(
    const device bfloat* routed [[buffer(0)]],
    const device bfloat* shared [[buffer(1)]],
    const device bfloat* gate   [[buffer(2)]],
    device bfloat* out          [[buffer(3)]],
    constant uint& width        [[buffer(4)]],
    const constant int& row_pitch [[buffer(5)]],
    uint2 gid                   [[thread_position_in_grid]]) {
  const uint c = gid.x;
  if (c >= width) return;
  const size_t row = size_t(gid.y) * size_t(row_pitch);
  const float g = 1.0f / (1.0f + metal::exp(-float(gate[row])));
  const size_t at = row + size_t(c);
  out[at] = static_cast<bfloat>(float(routed[at]) + g * float(shared[at]));
}

[[kernel]] void shared_expert_combine(
    const device bfloat* routed [[buffer(0)]],
    const device bfloat* shared [[buffer(1)]],
    const device bfloat* gate   [[buffer(2)]],
    device bfloat* out          [[buffer(3)]],
    constant uint& width        [[buffer(4)]],
    uint2 gid                   [[thread_position_in_grid]]) {
  const uint c = gid.x;
  if (c >= width) return;
  const uint row = gid.y;
  const float g = 1.0f / (1.0f + metal::exp(-float(gate[row])));
  const uint at = row * width + c;
  out[at] = static_cast<bfloat>(float(routed[at]) + g * float(shared[at]));
}
