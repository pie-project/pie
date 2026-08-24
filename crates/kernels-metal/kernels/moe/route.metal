// Generic MoE routing: select experts, group rows by expert, and combine them.
//
// A mixture's problem at prefill is not arithmetic, it is that every (token,
// slot) pair reads a DIFFERENT expert's weight matrix. The matvec form pays
// that literally: `rows * experts_per_token` matvecs, each re-reading a whole
// [N, K] stack slice, so a 512-token prefill reads the same 128 experts four
// thousand times. Nothing about the shape improves with the batch, which is
// exactly what the measurements said -- our routed prefill was flat in length
// where mlx-lm's climbed.
//
// The fix is not a wider kernel. It is to put the rows that share an expert
// NEXT TO EACH OTHER, at which point the projection is an ordinary batched
// matmul over a contiguous block of rows against one weight slice -- the
// `affine_qmm_t` this driver already has. The reordering is:
//
//   sort    -- a counting sort of the (row, slot) pairs by expert id, laid out
//              so each expert's run starts on a tile boundary.
//   gather  -- copy each sorted position's source row into that order.
//
// There is no third kernel putting the results back. The sort emits the INVERSE
// permutation alongside the forward one -- it knows both, at no cost, at the
// moment it places a pair -- and the combine step reads its k slots through it.
// Undoing a permutation to feed a kernel that is about to gather anyway is a
// dispatch and a full-width buffer spent to make an index arithmetic look
// simpler.
//
// Sort and gather also run at M=1. A decode handles eight pairs, which is
// microseconds -- and it means the routed
// dataflow has ONE shape rather than a decode shape and a prefill shape that
// have to be kept agreeing. The batched and unbatched paths differ in exactly
// one number, `tile_rows`: 1 leaves the sort a pure grouping with no padding
// and the projections stay matvecs; 16 rounds every expert's run up to a tile
// and they become matmuls.

#include <metal_stdlib>

using namespace metal;

// THE FOUR PARAMETER BLOCKS ARE GONE, AND `moe/params.h` WITH THEM.
//
// `router_topk` took a `constant RouterParams&`, `route_sort` and
// `route_gather` a `constant MoeRouteParams&`, and `combine_sorted` a
// `constant ExpertCombineParams&` -- one buffer each, holding every field, at
// the address of the statement's staged scalar run. That is where the tree's
// packed-params convention came from and it is the convention being unwound:
// a struct pointer is one slot for many numbers, so nothing between the text
// and the shader could name any single one of them, and a text that stated
// FEWER words than the struct declares was undetectable. It happened here.
// `driver-metal/tests/packed_params_cover_the_struct.rs` exists because
// `RouterParams` is four `unsigned int` and a text stated two: this file read
// `p.softmax_over_all` at byte 8 and `p.logits_pitch` at byte 12 out of the
// NEXT dispatch's scalars, which is a routing that softmaxes over all experts
// because a neighbouring statement's first word happened to be nonzero, and a
// logits stride taken from its second. Both produce weights and neither
// faults.
//
// Every field is a `const constant uint&` of its own now, one `setBytes` each
// where the block was one address, at ascending buffer indices AFTER this
// kernel's real operands -- which is how `driver-metal`'s `lay_out` numbers a
// routine's arguments: an argument's slot is its position in the list the body
// fired. The routines state one `Const<u32>` mark per field, in the struct's
// order, because that order is the statement's: word 0 of `route_sort`'s run
// is `n` on every plane and always was.
//
// Deleting a block from the MIDDLE of an operand list renumbers what follows
// it, and three of these four sat in the middle. `per_expert_scale`, and both
// `inv`s, each moved down one; each says so where it is declared.

// Top-k over each router-logit row, then a softmax over only the selected
// values. One threadgroup owns one row and one lane owns one expert.
//
// The reduction is threadgroup-wide. Qwen MoE has 128 experts, so reducing
// independently inside each simdgroup would select four quarter-local maxima;
// the explicit second level keeps the result correct beyond GPT-OSS's 32.
constant constexpr uint kRouterMaxTopK = 16;
constant constexpr uint kRouterMaxSimdgroups = 32;

// SCALED applies Gemma 4's learned per-expert gain after the top-k softmax.
// Separate instantiations avoid requiring an otherwise-unused scale binding.
// ── THE WEIGHT PLANE HAS ITS OWN ELEMENT ────────────────────────────────────
//
// `kernels::points::Moe::topk_softmax` declares `weights: Out<Tensor<f32>>`:
// a router weight is a probability, the fold that reads it multiplies in
// float on every plane, and rounding it to the activation element throws
// mantissa away for nothing. This kernel was one template over ONE `T` for
// the logits AND the weights, so the point could not be claimed -- an f32
// slot written through a `bfloat*` is half a plane.
//
// `W` is the weight element. `<bfloat, bfloat, ..>` keeps the two names and
// the ABI the legacy driver has always fired; `<bfloat, float, false>` is the
// arm `Moe::topk_softmax` claims. `per_expert_scale` stays at `T`: it is a
// checkpoint bank, read at the width the loader staged.
template <typename T, typename W, bool SCALED>
[[kernel]] void router_topk(
    const device T* logits     [[buffer(0)]],
    device int* expert_ids     [[buffer(1)]],
    device W* expert_weights   [[buffer(2)]],
    // MOVED DOWN A SLOT, because `RouterParams` was buffer 3 and this was 4.
    // Still bound by the unscaled instantiation: the slot is positional, so it
    // has to hold an address whether or not `SCALED` dereferences it.
    const device T* per_expert_scale [[buffer(3)]],
    // `RouterParams`, field for field and in its order.
    //
    // `logits_pitch` of zero means the pitch IS `n_experts`, which is
    // load-bearing: a router reading a slice of a wider activation has a pitch
    // that is not its expert count, and a host with no slice writes 0 rather
    // than restating the count. `softmax_over_all` is the word this file's
    // header names: 0 softmaxes the SELECTED logits so the k weights sum to
    // one -- `norm_topk_prob: true`, and what every family here shipped with --
    // while 1 softmaxes over ALL experts and then selects, so they sum to less
    // and scale the routed FFN's contribution down with them. Zero is the old
    // behaviour, so a site that does not state it keeps it.
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

  // `norm_topk_prob: false` wants the softmax taken over EVERY expert and the
  // top-k read out of it, so the k weights sum to less than one. Take that
  // denominator here, before the selection loop consumes `v` -- each lane
  // still holds its own logit and nothing has been knocked out yet.
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
    // Either denominator, one shape: exponentiate against a max and divide by
    // a sum. Which max and which sum is the whole of `norm_topk_prob`.
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

// The claimed arm: the weights at the element the declaration states. Both
// `router_topk_f32w_bfloat16` and its scaled twin are stamped, because the
// macro stamps a pair and the gemma gain is the same decision on the same
// plane -- `Moe::topk_softmax` fires the unscaled one.
instantiate_router_topk(f32w_bfloat16, bfloat, float)

// ── The two RANKED routers ───────────────────────────────────────────────────
//
// `router_topk` above ranks the logits and publishes a SOFTMAX over the k it
// picked. These two rank a TRANSFORM of the logits and publish that transform
// itself: there is no exponential and no shared denominator, so a row's k
// weights sum to whatever they sum to and `renormalize` is what decides
// whether that sum is divided out.
//
//   router_topk_sigmoid          w = sigmoid(x)
//   router_topk_sqrt_softplus    w = sqrt(log(1 + exp(x))), ranked with a bias
//
// THREE KERNELS AND NOT ONE TEMPLATE, and the duplication is deliberate. The
// selection scan below is `router_topk`'s, lane for lane, because it is the
// same scan; what differs is a transform, a denominator and -- for the third
// -- an operand. Folding them would put a `WEIGHT_FORM` enum in front of the
// one loop in this file whose correctness is a matter of which lane wins, and
// a router that picks the wrong expert is a wrong answer that reads as text.
//
// THE BIAS SHIFTS THE RANKING AND NOT THE WEIGHT. `sqrt_softplus`'s
// correction bias is DeepSeek's, and it is added to the value the scan
// compares and NOT to the value published -- which is why the published
// weight is recomputed from the logit at the chosen expert rather than
// carried out of the scan. Swapping the two reweights every expert by its own
// bias, and the model still produces text.
//
// A ROW WHOSE FAN-OUT EXCEEDS ITS EXPERT COUNT parks its spare slots on
// expert 0 with weight zero. Repeating the last winner would double-count it
// in the fold, and leaving the slot unwritten hands the combine whatever the
// arena held -- an id that indexes a bank out of bounds.

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

  // Sigmoid is monotone, so ranking the gate and ranking the logit choose the
  // same experts. It is taken here anyway rather than after the scan: this is
  // the shape the biased router beside it needs, and one scan written twice
  // over two different `v` is worse than one scan written twice over the same.
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

/// `sqrt(log(1 + exp(x)))`, saturated at zero.
///
/// The `x > 20` branch is not an optimisation. `exp(x)` overflows to inf a
/// little past 88 and `log(1 + inf)` is inf, so one large logit would route
/// with an infinite weight and renormalize every other expert to zero; past
/// 20 the two expressions agree to far inside bf16 anyway.
///
/// `log(1 + exp(x))` and not cuda's `log1pf(expf(x))`: MSL's math library has
/// no `log1p`. The two differ only where `exp(x)` falls under the float
/// epsilon -- below about x = -17 -- and there this returns exactly zero
/// where cuda returns `sqrt(exp(x))`. That is an expert the ranking has
/// already put last; the correction bias is the one thing that could promote
/// it, and a weight of 0 against one of 2e-4 is the same contribution to the
/// fold.
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

  // The RANKED value. `correction` is one float per expert and shared by every
  // row, so it is indexed by the lane and not by the row.
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
      // THE PUBLISHED WEIGHT, WITHOUT THE BIAS. Recomputed from the logit at
      // the chosen expert, which is the same expression the scan evaluated
      // minus the term that only ever belonged to the ranking.
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

// One lane per expert during the prefix scan, so this is the widest expert
// count this shape serves. `shared_kernels::kRouterMaxExperts` is the same
// number and the geometry refuses anything above it.
constant constexpr uint kMaxExperts = 1024;

/// Group the (row, slot) pairs by expert.
///
/// A single threadgroup: the scan is over `n_experts` (tens to hundreds), and
/// the scatter over `n` (thousands at most), so the parallelism that matters is
/// in the matmul this feeds, not here. Splitting it would need a global
/// histogram and a second dispatch to scan it, which is more synchronisation
/// than the work is worth.
///
/// One threadgroup is not the same as one LANE, though, and the difference was
/// measured: this scan was serial in lane 0 and cost 20 microseconds a layer,
/// which at 48 layers was most of what the reordering took off decode. There is
/// one thread per expert, so the prefix over the experts is a two-level simd
/// scan and each expert writes its own tiles.
///
/// Outputs, all indexed by SORTED position:
///   perm[p]        the (row, slot) pair at p, or -1 for a padding row
///   row_expert[p]  the expert p reads, for the matvec path
///   tile_expert[t] the expert tile t reads, or -1 for a tile past the end
///
/// and, indexed by PAIR rather than by position, the inverse:
///
///   inv[i]         the sorted position of pair i, or -1 if it has no expert
///
/// `perm` is a permutation of `[0, n)` followed by padding, never a truncation:
/// every pair the router chose gets a position, because a pair silently dropped
/// here is an expert contribution silently zeroed later.
[[kernel]] void route_sort(
    const device int* expert_ids [[buffer(0)]],
    device int* perm            [[buffer(1)]],
    device int* row_expert      [[buffer(2)]],
    device int* tile_expert     [[buffer(3)]],
    // MOVED DOWN A SLOT, because `MoeRouteParams` was buffer 4 and this was 5.
    device int* inv             [[buffer(4)]],
    // `MoeRouteParams`, field for field, and SHARED WITH `route_gather` below.
    //
    // One layout for the sort and the gather so that the padding one writes
    // and the bounds the other reads cannot disagree. That is not two structs
    // that happen to look alike: `model-dsl` states the same seven words for
    // both statements, and both routines take the same seven marks in the same
    // order -- three of which the gather never reads and carries anyway, so
    // that `width` is word 5 and `x_pitch` word 6 in both.
    //
    // `n` is the number of (row, slot) PAIRS -- one per expert choice -- while
    // `padded` is the length of the permutation, `n` rounded up so every
    // expert's span is a whole number of `tile_rows` tiles. The two are
    // different numbers and this kernel reads both; a body that used one for
    // the other would clear a permutation shorter than it fills.
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

    // Clear first, and clear EVERYTHING: the padding rows are read by the
    // gather and the spare tiles by the matmul, so a stale -1 that was never
    // written is a row of some previous layer's routing.
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

    // Each expert's run, rounded up to a whole tile. An expert nothing routed
    // to takes no space at all -- the padding is per TOUCHED expert, which is
    // what keeps the waste bounded when 128 experts see 8 pairs.
    //
    // `simd_prefix_exclusive_sum` has to be reached by every lane of the
    // simdgroup, so the span is computed as zero past `E` rather than branched
    // around.
    const uint span = lid < E
        ? (atomic_load_explicit(&counts[lid], memory_order_relaxed) > 0u
               ? ((atomic_load_explicit(&counts[lid], memory_order_relaxed) + tile - 1u) / tile) * tile
               : 0u)
        : 0u;
    const uint within = simd_prefix_exclusive_sum(span);
    const uint sg = lid / 32u;
    const uint n_sg = (nthreads + 31u) / 32u;
    // The group total from lane 0 rather than from its last lane. `simd_sum` is
    // uniform over the whole simdgroup either way, and lane 0 always exists --
    // where "the last lane" needs a second clause for a partial group that this
    // dispatch shape can never produce, and so could never be tested.
    const uint sg_total = simd_sum(span);  // uniform: every lane must reach it
    if (lid % 32u == 0u) sg_sum[sg] = sg_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // At most 32 simdgroups (1024 threads), so this residual scan is 32 adds in
    // one lane rather than 1024.
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
        // Reused as the per-expert write cursor by the scatter below.
        atomic_store_explicit(&counts[lid], 0u, memory_order_relaxed);
    }
    // `tile_expert` is device memory but nothing in THIS kernel reads it, and
    // the device clears above were already published by the first barrier, so
    // what has to be visible here is `base` and the reset cursors.
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

/// Copy each sorted position's source row into sorted order.
///
/// `perm[p]` is a (row, slot) pair, so the row it reads is `perm[p] /
/// experts_per_token` -- the k slots of one token share one activation, which
/// is why this is a broadcast rather than a permutation of equal sizes.
///
/// Padding rows are zeroed rather than left alone. They are multiplied by a
/// real weight and the result is discarded, so their VALUE never matters -- but
/// an unwritten row is whatever the pool held, and bf16 garbage can be inf,
/// which a later `simd_sum` would spread across a tile that does matter.
[[kernel]] void route_gather(
    const device bfloat* x     [[buffer(0)]],
    device bfloat* out         [[buffer(1)]],
    const device int* perm     [[buffer(2)]],
    // THE SORT'S BLOCK, WHOLE, and three of these seven words are never read
    // here. Deliberate, and `route_sort`'s note above says why: the sort's
    // `padded` is this kernel's row bound, so a gather carrying its own
    // shorter block would be a second place for the same number to be stated
    // and a second place for it to be wrong. `n`, `n_experts` and `tile_rows`
    // are the sort's alone and ride here so that the two kernels take the same
    // marks at the same slots.
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
    // Named `pitch` and not `x_pitch`: the resolved value is not the stated
    // one when the caller writes zero, and a local shadowing the argument it
    // is derived from would leave the two spellings meaning different things
    // in one body.
    const uint pitch = x_pitch != 0u ? x_pitch : width;
    out[uint(gid.y) * width + gid.x] =
        sel < 0 ? bfloat(0) : x[(uint(sel) / k) * pitch + gid.x];
}

/// Sum a token's k expert outputs, weighted by the router, reading them where
/// they were WRITTEN.
///
/// `y` is `[rows * k, width]` in (token, slot) order -- one row per route, in
/// the order the routes were chosen -- so slot `e` of token `n` is at row
/// `n * k + e` and there is no permutation to consult. That is the whole
/// difference between this and `combine_sorted` below, and it is why the two
/// are two kernels: every sorted family binds an inverse permutation, and a
/// caller whose rows were never moved should not carry a buffer it would only
/// fill with the identity.
///
/// The weights are FLOAT. `Moe::weighted_sum` declares them
/// `In<Tensor<f32>>`, because a router weight is a probability and the fold
/// multiplies in float on every plane; `combine_sorted`'s are bf16 because
/// that is the element the legacy driver has always staged them at.
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

/// Sum a token's k expert outputs, weighted by the router's softmax, reading
/// them where the SORT left them.
///
/// The same arithmetic as `expert_combine`, and deliberately a separate kernel
/// rather than that one taught an optional index: every sorted family binds the
/// inverse, while an unsorted caller should not carry a buffer it never reads.
///
/// A slot whose pair never got a position contributes zero. That cannot happen
/// for a routing the geometry accepted -- every id is in range and every pair
/// is placed -- but reading `y` at -1 if it ever did would be a wild load, and
/// the whole reason the sort is a permutation rather than a filter is that a
/// silently dropped expert is a silently wrong answer.
[[kernel]] void combine_sorted(
    const device bfloat* y              [[buffer(0)]],
    const device bfloat* expert_weights [[buffer(1)]],
    device bfloat* out                  [[buffer(2)]],
    // MOVED DOWN A SLOT, because `ExpertCombineParams` was buffer 3 and this
    // was 4.
    const device int* inv               [[buffer(3)]],
    // `ExpertCombineParams`, field for field. `out_pitch` of zero means
    // `width`, for the same reason `RouterParams::logits_pitch`'s zero means
    // `n_experts`: the mixture's output is token-major and lands in whatever
    // layout the caller's activations are in. A batched decode's are packed; a
    // prefill's are a uniform `scratch_widest_elems` apart, because a
    // per-token DAG binds one offset for every value it touches. Nonzero is
    // the second case.
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

// ── The shared expert ────────────────────────────────────────────────────────
//
// Every routed member of this family -- Qwen3-Next-80B, Qwen3.5-35B-A3B,
// Qwen3.5-122B-A10B -- runs one DENSE FFN beside the routed bank on every
// token, and adds it to the mixture's output under a learned gate:
//
//   y = routed + sigmoid(shared_expert_gate(x)) * shared_expert(x)
//
// The FFN half needs nothing new -- it is the same three projections and the
// same SwiGLU the dense members of this family already run, at
// `shared_expert_intermediate_size`. Only this last line is new, and it is new
// for one reason: the gate is ONE number per token, broadcast across the whole
// hidden row. `attn_gate` looks like it would serve and does not; its gate is
// full width, so it would read `hidden` gate values where there is one.
//
// Fused rather than a multiply and an add, because the alternative writes the
// scaled shared output to a full-width scratch buffer that the very next
// dispatch consumes and nothing else ever reads.
//
// The sigmoid is computed in float from a bf16 logit. That matters at the
// tails: bf16 has eight mantissa bits, so rounding the logit BEFORE the
// nonlinearity moves the gate by up to a few parts in a thousand, on a term
// that is added to every token's residual in every routed layer.
/// The prefill's shape of the same fused line: rows a uniform `row_pitch`
/// apart, so the whole prompt runs as one dispatch instead of one per token.
///
/// The GATE strides too, and that is the part worth stating. It is ONE number
/// per row, so reading it at `row` rather than `row * row_pitch` looks right
/// and gives every token row 0's gate -- a plausible answer and the wrong one.
/// It strides because `qmv_out_size` answers 1 rather than 0 for
/// `LlSharedGateProj`, which puts it on the projection arm of the prefill's
/// row-stride binding: its output is written a full pitch apart like every
/// other projection's.
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
    const device bfloat* routed [[buffer(0)]],   // [rows, width]
    const device bfloat* shared [[buffer(1)]],   // [rows, width]
    const device bfloat* gate   [[buffer(2)]],   // [rows, 1]
    device bfloat* out          [[buffer(3)]],   // [rows, width] (may alias routed)
    constant uint& width        [[buffer(4)]],
    uint2 gid                   [[thread_position_in_grid]]) {
  const uint c = gid.x;
  if (c >= width) return;
  const uint row = gid.y;
  const float g = 1.0f / (1.0f + metal::exp(-float(gate[row])));
  const uint at = row * width + c;
  out[at] = static_cast<bfloat>(float(routed[at]) + g * float(shared[at]));
}
