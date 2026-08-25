// Raw-Metal batched PAGED attention (read side, M>1) — beta's lane.
//
// Generalizes sdpa_vector_decode from the M=1 CONTIGUOUS-ring K/V to the M>1 PAGED KV
// cache that the Pie runtime owns. The flash online-softmax math is IDENTICAL to
// sdpa_vector.metal; ONLY the per-key/value fetch changes: instead of striding a
// contiguous [n_kv_heads, max_ctx, head_dim] ring, each kv position is gathered through
// the page table (the byte-for-byte seam with delta's kv_append — wiki mac-paged-kv-bridge).
//
// Page layout (LOCKED, token-major NHD): k/v pages = [num_pages, page_size, n_kv_heads, head_dim].
//   phys_slot(kp) = kv_page_indices[ kv_page_indptr[r] + kp/page_size ] * page_size + kp%page_size
//   element       = (phys_slot * n_kv_heads + kv_head) * head_dim + d        (== delta's append dst)
//
// Causal bound: query token `row` at absolute position q_pos = position_ids[row] attends
// exactly kv positions [0, q_pos] (its own K was appended before attention). So the read
// walks kp = 0..q_pos — no kv_last_page_lens needed on the read side (the position IS the
// bound). Decode (1 tok/req) and prefill (qo-span tokens, each its own q_pos) use the SAME
// loop; ragged lengths fall out of per-row q_pos. req_of_token[row] (host-precomputed from
// qo_indptr, batch_schedule::tok_req) gives the owning request → its page list.
//
// At N=1, R=1 this reduces to a single query row walking one request's pages == the M=1
// decode path (a 1-page table over the old ring), so the shipped fast path is preserved.
//
// Launch: one threadgroup per (q_batch_head, query_row): grid threads
//   = (n_q_heads*1024, N, 1), tg = (1024, 1, 1)   (1024 = BN*BD, matches sdpa_vector).
// D = head_dim (template; qwen 256, gemma4 256/512 — host picks the instantiation +
// per-layer pages/kv_source redirect). gqa_factor = n_q_heads/n_kv_heads.
//
// Buffer ORDER here is the binding contract; alpha assigns the matching bind::SdpaPaged
// ordinals in decode_abi.hpp (I bind against his published layout). u32 page tables per schema.

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

#include "sdpa_online.h"

// The online softmax strides keys by BN simdgroups. The generic kernel uses
// BN=32; the d64/page32 Llama specialization uses eight and gives simdgroup 0
// the final cross-group reduction. The launch must match the instantiation:
// missing simdgroups mean missing keys, not a smaller version of the same work.
//
// The attribute is the ceiling needed by the generic form; a short
// specialization may launch fewer threads.
//
// This carried a second sentence -- "at D=512 the compiler's default register
// allocation only admits 896, and a 1024-thread dispatch does not run AT ALL"
// -- and it is not true of this toolchain.
// `device_kernels::every_declared_entrypoint_builds_a_pipeline_on_this_device`
// reads `maxTotalThreadsPerThreadgroup` off every built pipeline and
// `sdpa_paged_decode_bfloat16_d_512` admits 1024; `device_attention` then
// dispatches it at 1024 and gets the reference answer. 180 of the 481
// pipelines DO admit fewer than 1024 -- the quantised GEMMs mostly, down to
// 576 -- so the effect the sentence described is real, just not here. That
// test now asserts the thing that matters instead of the anecdote: a pipeline
// must admit at least what its own launch rule dispatches.
//
// It is worth saying why a stale ceiling is dangerous rather than merely
// wrong. Metal does not refuse an over-wide threadgroup: it skips the
// dispatch, leaves the output holding whatever it held, and reports success.
// `device::encoder::dispatch` is what turns that into an error, and it is a
// guard, not a licence to guess the number.
// WITH_SINK adds gpt-oss's learned per-head scalar to the softmax denominator,
// as though it were one more key with no value to contribute. It is what lets a
// head attend to nothing. A template parameter rather than a second kernel: the
// paging, the CSR walk and the online softmax are identical, and two copies of
// them would be two things to keep true.
//
// WITH_LSE publishes the softmax's log-sum-exp beside the output, in base two
// -- see `sdpa_lse_base2`. It is a DIFFERENT decomposition of the same
// arithmetic from WITH_SINK, not a companion to it: a kernel that folded the
// sink into its denominator and then published an lse would be publishing a
// normaliser the sink is already inside, and `attention.sink` downstream would
// apply it twice. The two are mutually exclusive and the static_assert below
// is what says so at the instantiation rather than at the fire.
//
// # The body, and why the decode grew one
//
// `sdpa_paged_tiled_body` was already split this way, for the reason repeated
// here: an entry point's buffer list IS its binding contract, so a body with
// one more operand has to be a NEW entry point rather than an argument the old
// one learns to ignore. The threadgroup arrays come down as pointers because
// Metal forbids a non-kernel function from declaring threadgroup memory.
template <
    typename T,
    int D,
    int V,
    bool WITH_SINK,
    bool WITH_LSE,
    int PAGE_SIZE,
    bool FAST_FULL,
    int BN>
inline void sdpa_paged_decode_body(
    const device T* queries,
    const device T* k_pages,
    const device T* v_pages,
    device T* out,
    const int gqa_factor,
    const device int* position_ids,
    const device int* req_of_token,
    const device uint* kv_page_indices,
    const device uint* kv_page_indptr,
    const int page_size,
    const int n_kv_heads,
    const float scale,
    const device uchar* attention_mask,
    const uint attention_mask_stride,
    const device uchar* attention_mask_enabled,
    const int window,
    const device T* sinks,
    device float* lse,
    threadgroup float* outputs,
    threadgroup float* max_scores,
    threadgroup float* sum_exp_scores,
    threadgroup float* factors,
    threadgroup float* final_sum,
    uint3 tid,
    uint3 tpg,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      !(WITH_SINK && WITH_LSE),
      "a folded sink and a published lse are two readings of one denominator");
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr float NEG_INF = -3.0e38f;

  typedef float U;
  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U v[v_per_thread];
  thread U o[v_per_thread];

  const int q_batch_head_idx = tid.x;         // 0..n_q_heads-1
  const int row              = tid.y;         // query token index in [0, N)
  const int kv_head_idx      = q_batch_head_idx / gqa_factor;
  const int n_q_heads        = int(tpg.x);

  // This query row's request + causal bound + its request's page-list base.
  const int r          = req_of_token[row];
  const int q_pos      = position_ids[row];   // attends kv positions [kv_start, q_pos]
  // Sliding window: the span starts `window-1` before this row's own position
  // rather than at 0. window<=0 is full attention, which is the same loop with
  // kv_start==0 -- so one kernel serves both, and a sliding layer cannot drift
  // from a full one.
  const int kv_start   =
      FAST_FULL ? 0 : ((window > 0 && q_pos >= window) ? (q_pos - window + 1) : 0);
  const int page_base  = int(kv_page_indptr[r]);

  // queries[row, q_head, :]; out[row, q_head, :]
  queries += (size_t(row) * n_q_heads + q_batch_head_idx) * D + simd_lid * qk_per_thread;
  out     += (size_t(row) * n_q_heads + q_batch_head_idx) * V;
  if constexpr (BN == 32) out += simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) q[i] = static_cast<U>(scale) * queries[i];
  for (int i = 0; i < v_per_thread; i++) o[i] = 0;

  U max_score = NEG_INF;
  U sum_exp_score = 0;

  // HOISTED, because it is the ROW's and the loop is over keys. Read inside
  // the loop this was one device load per kv position -- cached, but also a
  // dependency the scheduler has to satisfy before it can decide whether to
  // issue the key load at all, which is the one thing this loop cannot
  // afford. See the note on `v` below.
  const bool masked = !FAST_FULL && attention_mask_enabled[row] != 0;

  // ONE KV POSITION: the page table has already been walked for it, so this
  // is only the two loads, the dot product and the online update.
  //
  // A lambda so the two loops below share it. They differ in nothing but
  // which simdgroup takes which position, and two copies of an online softmax
  // would be two things to keep true.
  //
  // BOTH LOADS ISSUED BEFORE EITHER IS USED, and that is the whole structure
  // of it. The value read used to be `vptr[j]` in the accumulate -- after the
  // dot product, after `simd_sum`, and after the online update that depends on
  // it. So each position was two round trips to memory in series. Costs
  // `v_per_thread` registers, which at D=64 is two.
  auto absorb = [&](size_t slot) {
    const device T* kptr =
        k_pages + (slot * n_kv_heads + kv_head_idx) * D + simd_lid * qk_per_thread;
    const device T* vptr =
        v_pages + (slot * n_kv_heads + kv_head_idx) * D + simd_lid * v_per_thread;
    for (int j = 0; j < qk_per_thread; j++) k[j] = kptr[j];
    for (int j = 0; j < v_per_thread; j++) v[j] = static_cast<U>(vptr[j]);
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) score += q[j] * k[j];
    score = simd_sum(score);
    U factor, exp_score;
    sdpa_online_update(score, max_score, sum_exp_score, factor, exp_score);
    for (int j = 0; j < v_per_thread; j++) o[j] = o[j] * factor + exp_score * v[j];
  };

  // Whether a kv position is attended at all: the row's mask, when it has one.
  auto attends = [&](int kp) {
    if constexpr (FAST_FULL) {
      return true;
    } else {
      return !masked ||
          (uint(kp) < attention_mask_stride &&
           attention_mask[size_t(row) * attention_mask_stride + uint(kp)] != 0);
    }
  };

  const int stride = PAGE_SIZE == 0 ? page_size : PAGE_SIZE;
  const int first_page = kv_start / stride;
  const int last_page = q_pos / stride;

  // WHICH SIMDGROUP TAKES WHICH POSITION, and it is the whole reason this
  // kernel was 32 GB/s.
  //
  // Striding the POSITIONS by BN -- `kp = simd_gid, +32, ...` -- puts every
  // iteration in a different page at page_size 16, so every iteration begins
  // with `kv_page_indices[page_base + kp / page_size]`: a load the address of
  // the key load depends on. The key cannot even be issued until the page
  // index comes back, so each iteration is a dependent pair of round trips to
  // memory and the loop is a pointer chase 32 links long. Measured over
  // Llama-3.2-1B at 1024 of context: 1.033 ms for sixteen launches reading
  // 2 MB each, which is 32 GB/s on a 400 GB/s part.
  //
  // Striding the PAGES instead gives a simdgroup a whole page at a time: one
  // page-index load, then `page_size` key loads whose addresses are all known
  // the moment it lands, so they issue together and the latency is paid once
  // per page rather than once per position.
  //
  // It is only the better shape when there are pages enough to go round. At
  // 128 of context a request holds eight pages and twenty-four of the
  // thirty-two simdgroups would sit idle, so the position-strided loop stays
  // for that case -- it is not a fallback, it is the right answer when the
  // context is short, and `what_a_decode_costs_at_length` measures both.
  if (last_page - first_page + 1 >= BN) {
    for (int pix = first_page + simd_gid; pix <= last_page; pix += BN) {
      const size_t base = size_t(kv_page_indices[page_base + pix]) * stride;
      const int lo = max(kv_start, pix * stride);
      const int hi = min(q_pos, pix * stride + stride - 1);
      for (int kp = lo; kp <= hi; ++kp) {
        if (attends(kp)) absorb(base + size_t(kp - pix * stride));
      }
    }
  } else {
    int fast_page_ix = 0;
    int fast_page_off = simd_gid;
    for (int kp = kv_start + simd_gid; kp <= q_pos; kp += BN) {
      // The incremental page walk the `_p32` arms exist for: a page index and
      // an offset carried across iterations instead of a divide.
      size_t slot;
      if constexpr (PAGE_SIZE == 32 && FAST_FULL) {
        slot = size_t(kv_page_indices[page_base + fast_page_ix]) * 32 + fast_page_off;
        fast_page_off += BN;
        if (fast_page_off >= 32) {
          fast_page_off -= 32;
          ++fast_page_ix;
        }
      } else {
        const int page_ix = PAGE_SIZE == 32 ? (kp >> 5) : (kp / page_size);
        const int page_off = PAGE_SIZE == 32 ? (kp & 31) : (kp % page_size);
        slot = size_t(kv_page_indices[page_base + page_ix]) * stride + page_off;
      }
      if (attends(kp)) absorb(slot);
    }
  }

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if constexpr (BN == 32) {
    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

    // BEFORE the sink, always, because `sdpa_merge_sink` mutates the
    // denominator this reads and the static_assert above is not the only place
    // that ordering should be legible. Every simdgroup ran the reduction over
    // the same thirty-two entries, so `new_max` and `sum_exp_score` are
    // threadgroup-uniform here and one lane of one simdgroup owns the store.
    if constexpr (WITH_LSE) {
      if (simd_gid == 0 && simd_lid == 0) {
        lse[size_t(row) * size_t(n_q_heads) + size_t(q_batch_head_idx)] =
            sdpa_lse_base2(new_max, sum_exp_score);
      }
    }

    U orescale = 1;
    if (WITH_SINK) {
      const U sink = static_cast<U>(sinks[tid.x]);
      orescale = sdpa_merge_sink(sink, new_max, sum_exp_score);
    }

    for (int i = 0; i < v_per_thread; i++) {
      outputs[simd_lid * BD + simd_gid] = o[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) * orescale;
      o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0)
      for (int i = 0; i < v_per_thread; i++) out[i] = static_cast<T>(o[i]);
  } else {
    if (simd_gid == 0) {
      const U lane_max = simd_lid < BN ? max_scores[simd_lid] : NEG_INF;
      U new_max = simd_max(lane_max);
      U factor = simd_lid < BN ? fast::exp(lane_max - new_max) : 0;
      U total = simd_sum(
          simd_lid < BN ? sum_exp_scores[simd_lid] * factor : U(0));
      if (simd_lid < BN) factors[simd_lid] = factor;
      if (simd_lid == 0) {
        // Before the sink for the same reason as the wide shape above: `total`
        // is the denominator the lse names, and `sdpa_merge_sink` grows it.
        if constexpr (WITH_LSE) {
          lse[size_t(row) * size_t(n_q_heads) + size_t(q_batch_head_idx)] =
              sdpa_lse_base2(new_max, total);
        }
        if (WITH_SINK) {
          const U sink = static_cast<U>(sinks[tid.x]);
          const U rescale = sdpa_merge_sink(sink, new_max, total);
          for (int s = 0; s < BN; ++s) factors[s] *= rescale;
        }
        final_sum[0] = total;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < v_per_thread; ++i) {
      outputs[simd_gid * BD + simd_lid] = o[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd_gid == 0) {
        U acc = 0;
        for (int s = 0; s < BN; ++s)
          acc += outputs[s * BD + simd_lid] * factors[s];
        const U denom = final_sum[0];
        out[simd_lid * v_per_thread + i] =
            static_cast<T>(denom == 0 ? acc : acc / denom);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// The threadgroup memory the decode body is handed, declared where Metal
// requires it. `BD` is 32 in the body and the two entry points below have to
// spell the same product, so it is written once here rather than twice there.
template <int BN> constexpr int sdpa_decode_outputs() { return BN * 32; }

/// The decode as every family but gpt-oss's lse arms fires it: seventeen
/// buffers, no log-sum-exp plane.
template <
    typename T,
    int D,
    int V = D,
    bool WITH_SINK = false,
    int PAGE_SIZE = 0,
    bool FAST_FULL = false,
    int BN = 32>
[[kernel]] [[max_total_threads_per_threadgroup(1024)]] void sdpa_paged_decode(
    const device T* queries     [[buffer(0)]],   // [N, n_q_heads, D]
    const device T* k_pages     [[buffer(1)]],   // [num_pages, page_size, n_kv_heads, D]
    const device T* v_pages     [[buffer(2)]],
    device T* out               [[buffer(3)]],   // [N, n_q_heads, V]
    const constant int& gqa_factor          [[buffer(4)]],
    const device int* position_ids          [[buffer(5)]],   // [N] abs pos → causal bound
    const device int* req_of_token          [[buffer(6)]],   // [N] owning request r
    const device uint* kv_page_indices      [[buffer(7)]],   // [total_pages]
    const device uint* kv_page_indptr       [[buffer(8)]],   // [R+1]
    const constant int& page_size           [[buffer(9)]],
    const constant int& n_kv_heads          [[buffer(10)]],
    const constant float& scale             [[buffer(11)]],
    const device uchar* attention_mask      [[buffer(12)]],
    const device uint& attention_mask_stride[[buffer(13)]],
    const device uchar* attention_mask_enabled [[buffer(14)]],
    const constant int& window                 [[buffer(15)]],
    const device T* sinks                      [[buffer(16)]],  // [n_q_heads], WITH_SINK only
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  threadgroup float outputs[sdpa_decode_outputs<BN>()];
  threadgroup float max_scores[BN];
  threadgroup float sum_exp_scores[BN];
  threadgroup float factors[BN];
  threadgroup float final_sum[1];
  sdpa_paged_decode_body<T, D, V, WITH_SINK, false, PAGE_SIZE, FAST_FULL, BN>(
      queries, k_pages, v_pages, out, gqa_factor, position_ids, req_of_token,
      kv_page_indices, kv_page_indptr, page_size, n_kv_heads, scale,
      attention_mask, attention_mask_stride, attention_mask_enabled, window,
      sinks, /*lse=*/nullptr, outputs, max_scores, sum_exp_scores, factors,
      final_sum, tid, tpg, simd_gid, simd_lid);
}

/// `attention.decode_lse`: the same decode, publishing its denominator.
///
/// An eighteenth buffer and therefore an entry point of its own — see the body
/// for why a contract cannot grow an operand in place. `WITH_SINK` is not a
/// parameter here because a sunk decode has no lse to state: the point that
/// wants the sink states it as a SECOND statement (`attention.sink`) reading
/// the plane this one writes, and that is the decomposition the floor declares.
template <typename T, int D, int V = D>
[[kernel]] [[max_total_threads_per_threadgroup(1024)]] void sdpa_paged_decode_lse(
    const device T* queries     [[buffer(0)]],
    const device T* k_pages     [[buffer(1)]],
    const device T* v_pages     [[buffer(2)]],
    device T* out               [[buffer(3)]],
    const constant int& gqa_factor          [[buffer(4)]],
    const device int* position_ids          [[buffer(5)]],
    const device int* req_of_token          [[buffer(6)]],
    const device uint* kv_page_indices      [[buffer(7)]],
    const device uint* kv_page_indptr       [[buffer(8)]],
    const constant int& page_size           [[buffer(9)]],
    const constant int& n_kv_heads          [[buffer(10)]],
    const constant float& scale             [[buffer(11)]],
    const device uchar* attention_mask      [[buffer(12)]],
    const device uint& attention_mask_stride[[buffer(13)]],
    const device uchar* attention_mask_enabled [[buffer(14)]],
    const constant int& window                 [[buffer(15)]],
    const device T* sinks                      [[buffer(16)]],  // unread; the seat is the contract
    device float* lse                          [[buffer(17)]],  // [N, n_q_heads], base two
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  threadgroup float outputs[sdpa_decode_outputs<BN>()];
  threadgroup float max_scores[BN];
  threadgroup float sum_exp_scores[BN];
  threadgroup float factors[BN];
  threadgroup float final_sum[1];
  sdpa_paged_decode_body<T, D, V, false, true, 0, false, BN>(
      queries, k_pages, v_pages, out, gqa_factor, position_ids, req_of_token,
      kv_page_indices, kv_page_indptr, page_size, n_kv_heads, scale,
      attention_mask, attention_mask_stride, attention_mask_enabled, window,
      sinks, lse, outputs, max_scores, sum_exp_scores, factors, final_sum, tid,
      tpg, simd_gid, simd_lid);
}

// ── the same attention, with the query rows tiled ──
//
// `sdpa_paged_decode` above gives one query row a whole threadgroup, and its
// thirty-two simdgroups split that row's keys between them. That is the right
// shape for a decode, where there IS only one row and the only parallelism
// available is over the keys. It is the wrong shape for a prefill: every row
// walks the entire K/V prefix for itself, so a fire of N rows reads the prefix
// N times, and measurement says that is where prefill goes. Fitting
// prefill = a + b*n to the 30B checkpoint puts the quadratic term at 39% of the
// time at n=2048, and the traffic that term implies is ~527 GB/s against about
// 5% of the machine's fp16 peak: bandwidth, not arithmetic.
//
// So this kernel keeps the arithmetic EXACTLY as it is above -- same lane
// layout, same online softmax, same paged gather -- and changes only who owns a
// row. A row is a SIMDGROUP here, not a threadgroup, and the threadgroup stages
// a block of KT key/value positions into threadgroup memory that all QT=32 rows
// then read. The prefix is fetched from device memory once per thirty-two rows
// instead of once per row. Total ALU is unchanged; only the reads move.
//
// Two things fall out. The cross-simdgroup reduction epilogue disappears --
// each simdgroup now owns a complete row, so there is nothing to merge -- and
// the kernel needs to know N, because its grid rounds up to whole tiles where
// the per-row grid was exactly N threadgroups tall.
//
// Rows in a tile must share a request before they can share a staged page walk.
// They usually do: a prefill's rows are contiguous per request. Rather than
// require it, the tile walks the runs of equal `req_of_token` inside itself --
// one iteration in the common case, and correct when a tile straddles a
// request boundary. Simdgroups outside the current run sit out the scoring but
// still reach every barrier, because the run bounds are threadgroup-uniform.
// The tiled kernel's shape constants. At file scope because Metal forbids a
// non-kernel function from DECLARING threadgroup memory: the arrays have to be
// declared in the entry point and handed down, so both need these sizes.
constant constexpr int kSdpaQT = 32;                            // query rows per threadgroup
template <int D> constexpr int sdpa_kt() { return 4096 / D; }
template <int D, bool KPL> constexpr int sdpa_kq() { return KPL ? kSdpaQT * D : 1; }
template <int D, bool KPL> constexpr int sdpa_kp_tile() {
  return KPL ? kSdpaQT * sdpa_kt<D>() : 1;
}
// K's rows are read one per LANE in the lane-per-key shape, so a bare `D`
// stride puts all thirty-two lanes in one bank; two elements of padding walks
// them across banks instead.
template <int D, bool KPL> constexpr int sdpa_kstride() { return KPL ? D + 2 : D; }

/// The body, shared by the two entry points below.
///
/// `q_row_pitch` / `o_row_pitch` are the distance between one ROW of `queries`
/// / `out` and the next, in elements, or 0 for the packed layout the batched
/// decode uses. A prefill is the case that needs them: it lays every scratch
/// value's rows a uniform `scratch_widest_elems` apart, which is wider than
/// `n_q_heads * D`, so a kernel that assumed packed would walk into the next
/// row's tensor and read plausible garbage rather than fail. The non-paged
/// `Sdpa` kind has carried `QRowStride`/`ORowStride` for exactly this reason
/// since the prefill first needed it; this is the paged twin.
///
/// Two entry points rather than one kernel with an extra buffer, because
/// gemma4, gpt-oss and llama all run the packed pipeline: a buffer the kernel
/// DECLARES and their bind tables do not set is not a slot they waste, it is
/// their attention reading an unbound argument.
template <typename T, int D, int V, bool WITH_SINK, bool KEY_PER_LANE, bool WITH_LSE>
inline void sdpa_paged_tiled_body(
    const device T* queries,
    const device T* k_pages,
    const device T* v_pages,
    device T* out,
    const int gqa_factor,
    const device int* position_ids,
    const device int* req_of_token,
    const device uint* kv_page_indices,
    const device uint* kv_page_indptr,
    const int page_size,
    const int n_kv_heads,
    const float scale,
    const device uchar* attention_mask,
    const uint attention_mask_stride,
    const device uchar* attention_mask_enabled,
    const int window,
    const device T* sinks,
    device float* lse,
    const int n_rows,
    const int q_row_pitch,
    const int o_row_pitch,
    threadgroup T* ktile,
    threadgroup T* vtile,
    threadgroup T* qtile,
    threadgroup float* ptile,
    uint3 tid,
    uint3 tpg,
    uint simd_gid,
    uint simd_lid) {
  // One staged tile serves both sides, so V must equal D. Every instantiation
  // in this driver has square heads; the static_assert is here so that the day
  // one does not, it fails to compile rather than reading V's values out of a
  // buffer sized for K's.
  static_assert(V == D, "the tiled path stages one shape for K and V");
  static_assert(
      !(WITH_SINK && WITH_LSE),
      "a folded sink and a published lse are two readings of one denominator");

  constexpr int QT = 32;          // query rows per threadgroup == simdgroups
  constexpr int KT = 4096 / D;    // kv positions staged per pass (16 KB of T)
  constexpr int per_lane = D / 32;
  constexpr float NEG_INF = -3.0e38f;
  typedef float U;

  // Where a score comes from, and it is the whole difference between the two
  // shapes this kernel compiles to.
  //
  // The original gives a KEY to the simdgroup: thirty-two lanes each multiply
  // `per_lane = D/32` dimensions and a `simd_sum` folds them into one number.
  // At llama's d=64 that is TWO fused multiply-adds behind a five-step shuffle
  // reduction -- the reduction costs more than the arithmetic it reduces.
  //
  // `KEY_PER_LANE` gives a key to the LANE instead. Each lane walks the whole
  // D-long dot product for a key of its own, so thirty-two keys are scored with
  // no reduction at all, and the softmax then needs one `simd_max` and one
  // `simd_sum` per THIRTY-TWO keys rather than one of each per key. It costs a
  // threadgroup copy of q -- the lane no longer owns the slice it needs -- and
  // a per-pass probability tile to hand the result back to the accumulation,
  // which still wants the original layout because that one needs no reduction.
  //
  // Not free at every width, and the width is what decides it. q and the
  // probabilities are `QT*D` and `QT*KT` on top of the staged keys and values:
  // 28 KB at d=64 and d=128, 34 KB at d=256, and at d=512 q alone is the whole
  // 32 KB. Halving KT does make d=256 fit, at 25 KB, and it was measured: 1304
  // -> 1196 tok/s on a 448-row gemma-4 prefill, because a pass half as deep is
  // twice as many staging barriers and at d=256 the reduction this removes is
  // only 62% of the arithmetic it sits on, against 250% at d=64. So the wide
  // heads keep the per-row shape, and this is compiled for the two widths where
  // it both fits and pays.
  // K's rows are read one per LANE in that shape, so a bare `D` stride puts all
  // thirty-two lanes in one bank. Two elements of padding walks them across
  // banks instead.
  constexpr int kstride = KEY_PER_LANE ? D + 2 : D;
  constexpr int kcols = KEY_PER_LANE ? (KT + 31) / 32 : 1;

  const int q_head    = int(tid.x);
  const int n_q_heads = int(tpg.x);
  const int kv_head   = q_head / gqa_factor;
  const int row_lo    = int(tid.y) * QT;
  const int row       = row_lo + int(simd_gid);
  const bool live     = row < n_rows;
  const uint lid      = simd_gid * 32u + simd_lid;

  thread U q[per_lane];
  thread U o[per_lane];
  for (int i = 0; i < per_lane; i++) o[i] = 0;
  if (live) {
    const device T* qp =
        queries + (q_row_pitch > 0 ? size_t(row) * size_t(q_row_pitch)
                                  : size_t(row) * size_t(n_q_heads) * size_t(D)) +
        size_t(q_head) * size_t(D) + simd_lid * per_lane;
    for (int i = 0; i < per_lane; i++) q[i] = static_cast<U>(scale) * static_cast<U>(qp[i]);
    // The lane-per-key shape needs the WHOLE row where the lane owns a slice of
    // it, so the row goes to threadgroup memory once. Unscaled: these are the
    // checkpoint's own bits, and the scale rides the score instead, which is
    // one rounding fewer than staging a scaled copy in T.
    if (KEY_PER_LANE) {
      for (int i = 0; i < per_lane; i++)
        qtile[simd_gid * D + simd_lid * per_lane + i] = qp[i];
    }
  } else {
    for (int i = 0; i < per_lane; i++) q[i] = 0;
    if (KEY_PER_LANE) {
      for (int i = 0; i < per_lane; i++)
        qtile[simd_gid * D + simd_lid * per_lane + i] = T(0);
    }
  }

  const int q_pos     = live ? position_ids[row] : 0;
  const int my_start  = (window > 0 && q_pos >= window) ? (q_pos - window + 1) : 0;
  const bool masked   = live && attention_mask_enabled[row] != 0;

  U max_score = NEG_INF;
  U sum_exp_score = 0;

  // Runs of equal request within this tile. Uniform across the threadgroup:
  // every thread walks the same runs, which is what keeps the barriers aligned.
  int sub = 0;
  while (sub < QT && row_lo + sub < n_rows) {
    const int r = req_of_token[row_lo + sub];
    int sub_hi = sub + 1;
    while (sub_hi < QT && row_lo + sub_hi < n_rows && req_of_token[row_lo + sub_hi] == r) sub_hi++;

    // The run's key span: the widest causal bound and the earliest window start
    // among its rows. Each row still masks itself to its own bounds below; this
    // only says how far the staging has to go.
    int kp_hi = 0;
    int kp_lo = 0x7fffffff;
    for (int i = sub; i < sub_hi; i++) {
      const int p = position_ids[row_lo + i];
      kp_hi = max(kp_hi, p);
      kp_lo = min(kp_lo, (window > 0 && p >= window) ? (p - window + 1) : 0);
    }
    const int page_base = int(kv_page_indptr[r]);
    const bool mine = live && int(simd_gid) >= sub && int(simd_gid) < sub_hi;

    for (int base = kp_lo; base <= kp_hi; base += KT) {
      const int cnt = min(KT, kp_hi + 1 - base);
      // Before the writes, not only after: the previous pass's readers must be
      // done with the tile this one overwrites.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int e = int(lid); e < cnt * D; e += 1024) {
        const int kk = e / D;
        const int d  = e - kk * D;
        const int kp = base + kk;
        const int page = int(kv_page_indices[page_base + kp / page_size]);
        const size_t slot = size_t(page) * page_size + (kp % page_size);
        const size_t off = (slot * n_kv_heads + kv_head) * D + d;
        ktile[kk * kstride + d] = k_pages[off];
        vtile[e] = v_pages[off];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (!mine) continue;  // still reaches the next pass's barrier

      const threadgroup T* vbase = vtile + simd_lid * per_lane;
      // Whether a key survives this row's causal bound, window and mask. Both
      // shapes ask it; only where they ask it from differs.
      const auto keeps = [&](int kp) {
        if (kp > q_pos || kp < my_start) return false;
        return !(masked && (uint(kp) >= attention_mask_stride ||
                            attention_mask[size_t(row) * attention_mask_stride +
                                           uint(kp)] == 0));
      };

      if (KEY_PER_LANE) {
        // ── the pass's scores, one key per lane, no reduction ──
        const threadgroup T* qrow = qtile + simd_gid * D;
        U s[kcols];
        U block_max = NEG_INF;
        for (int c = 0; c < kcols; c++) {
          const int kk = c * 32 + int(simd_lid);
          U sc = NEG_INF;
          if (kk < cnt && keeps(base + kk)) {
            const threadgroup T* kptr = ktile + kk * kstride;
            U acc = 0;
            for (int d = 0; d < D; d++)
              acc += static_cast<U>(qrow[d]) * static_cast<U>(kptr[d]);
            sc = static_cast<U>(scale) * acc;
          }
          s[c] = sc;
          block_max = sc > block_max ? sc : block_max;
        }
        // Two reductions for the whole pass, where the other shape spends two
        // per key.
        block_max = simd_max(block_max);
        const U new_max = block_max > max_score ? block_max : max_score;
        // A pass whose every key was masked leaves `new_max` at -inf; rescaling
        // by exp(-inf - -inf) is a NaN, and the pass has nothing to add anyway.
        if (new_max > NEG_INF) {
          const U factor = max_score == NEG_INF ? U(0) : fast::exp(max_score - new_max);
          U block_sum = 0;
          for (int c = 0; c < kcols; c++) {
            const int kk = c * 32 + int(simd_lid);
            const U p = s[c] == NEG_INF ? U(0) : fast::exp(s[c] - new_max);
            // A pass narrower than the simdgroup leaves lanes with no key of
            // their own, and their slot is the NEXT row's. `KT >= 32` at every
            // width compiled here, so this guards a shape rather than a bug --
            // but it guards it where the shape is decided, not where it is read.
            if (kk < KT) ptile[simd_gid * KT + kk] = p;
            block_sum += p;
          }
          block_sum = simd_sum(block_sum);
          simdgroup_barrier(mem_flags::mem_threadgroup);
          max_score = new_max;
          sum_exp_score = sum_exp_score * factor + block_sum;
          // ── the pass's values, back in the layout that needs no reduction ──
          for (int j = 0; j < per_lane; j++) o[j] *= factor;
          for (int kk = 0; kk < cnt; kk++) {
            // Uniform across the simdgroup: the row is the simdgroup and the
            // key is the loop, so this reads one broadcast word.
            const U p = ptile[simd_gid * KT + kk];
            if (p == U(0)) continue;
            const threadgroup T* vptr = vbase + kk * D;
            for (int j = 0; j < per_lane; j++)
              o[j] += p * static_cast<U>(vptr[j]);
          }
        }
      } else {
        const threadgroup T* kbase = ktile + simd_lid * per_lane;
        for (int kk = 0; kk < cnt; kk++) {
          const int kp = base + kk;
          // Every condition here is a property of the ROW, and a row is a
          // simdgroup, so the branch is simd-uniform and `simd_sum` below is
          // reached by all thirty-two lanes or by none.
          if (!keeps(kp)) continue;
          const threadgroup T* kptr = kbase + kk * kstride;
          U score = 0;
          for (int j = 0; j < per_lane; j++) score += q[j] * static_cast<U>(kptr[j]);
          score = simd_sum(score);

          U factor, exp_score;
          sdpa_online_update(
              score, max_score, sum_exp_score, factor, exp_score);
          const threadgroup T* vptr = vbase + kk * D;
          for (int j = 0; j < per_lane; j++)
            o[j] = o[j] * factor + exp_score * static_cast<U>(vptr[j]);
        }
      }
    }
    sub = sub_hi;
  }

  if (!live) return;

  // A row is a simdgroup here, so the denominator this publishes needs no
  // merge to be complete -- and, as above, it is read before the sink can
  // grow it. `simd_lid == 0` because thirty-two lanes hold the same number.
  if constexpr (WITH_LSE) {
    if (simd_lid == 0) {
      lse[size_t(row) * size_t(n_q_heads) + size_t(q_head)] =
          sdpa_lse_base2(max_score, sum_exp_score);
    }
  }

  // The sink joins the denominator at a shared reference, exactly as above --
  // there is just no cross-simdgroup merge to carry it through any more.
  U orescale = 1;
  if (WITH_SINK) {
    const U sink = static_cast<U>(sinks[q_head]);
    orescale = sdpa_merge_sink(sink, max_score, sum_exp_score);
  }

  device T* op = out +
                 (o_row_pitch > 0 ? size_t(row) * size_t(o_row_pitch)
                                  : size_t(row) * size_t(n_q_heads) * size_t(V)) +
                 size_t(q_head) * size_t(V) + simd_lid * per_lane;
  for (int j = 0; j < per_lane; j++) {
    const U x = o[j] * orescale;
    op[j] = static_cast<T>(sum_exp_score == 0 ? x : x / sum_exp_score);
  }
}

template <typename T, int D, int V = D, bool WITH_SINK = false, bool KEY_PER_LANE = false>
[[kernel]] [[max_total_threads_per_threadgroup(1024)]] void sdpa_paged_tiled(
    const device T* queries     [[buffer(0)]],   // [N, n_q_heads, D]
    const device T* k_pages     [[buffer(1)]],   // [num_pages, page_size, n_kv_heads, D]
    const device T* v_pages     [[buffer(2)]],
    device T* out               [[buffer(3)]],   // [N, n_q_heads, V]
    const constant int& gqa_factor          [[buffer(4)]],
    const device int* position_ids          [[buffer(5)]],
    const device int* req_of_token          [[buffer(6)]],
    const device uint* kv_page_indices      [[buffer(7)]],
    const device uint* kv_page_indptr       [[buffer(8)]],
    const constant int& page_size           [[buffer(9)]],
    const constant int& n_kv_heads          [[buffer(10)]],
    const constant float& scale             [[buffer(11)]],
    const device uchar* attention_mask      [[buffer(12)]],
    const device uint& attention_mask_stride[[buffer(13)]],
    const device uchar* attention_mask_enabled [[buffer(14)]],
    const constant int& window                 [[buffer(15)]],
    const device T* sinks                      [[buffer(16)]],  // [n_q_heads], WITH_SINK only
    const constant int& n_rows                 [[buffer(17)]],  // N; the grid rounds up past it,
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  threadgroup T ktile[sdpa_kt<D>() * sdpa_kstride<D, KEY_PER_LANE>()];
  threadgroup T vtile[sdpa_kt<D>() * D];
  threadgroup T qtile[sdpa_kq<D, KEY_PER_LANE>()];
  threadgroup float ptile[sdpa_kp_tile<D, KEY_PER_LANE>()];
  sdpa_paged_tiled_body<T, D, V, WITH_SINK, KEY_PER_LANE, false>(
      queries, k_pages, v_pages, out, gqa_factor, position_ids, req_of_token,
      kv_page_indices, kv_page_indptr, page_size, n_kv_heads, scale,
      attention_mask, attention_mask_stride, attention_mask_enabled, window,
      sinks, /*lse=*/nullptr, n_rows,
      /*q_row_pitch=*/0, /*o_row_pitch=*/0,
      ktile, vtile, qtile, ptile, tid, tpg, simd_gid, simd_lid);
}

/// `attention.prefill_lse`: the packed tiling, publishing its denominator.
///
/// A nineteenth buffer, so an entry point of its own for the reason the decode
/// twin above gives. `WITH_SINK` is again absent, and again because the point
/// that wants a sink states it as the separate rescale that reads this plane.
template <typename T, int D, int V = D, bool KEY_PER_LANE = false>
[[kernel]] [[max_total_threads_per_threadgroup(1024)]] void sdpa_paged_tiled_lse(
    const device T* queries     [[buffer(0)]],
    const device T* k_pages     [[buffer(1)]],
    const device T* v_pages     [[buffer(2)]],
    device T* out               [[buffer(3)]],
    const constant int& gqa_factor          [[buffer(4)]],
    const device int* position_ids          [[buffer(5)]],
    const device int* req_of_token          [[buffer(6)]],
    const device uint* kv_page_indices      [[buffer(7)]],
    const device uint* kv_page_indptr       [[buffer(8)]],
    const constant int& page_size           [[buffer(9)]],
    const constant int& n_kv_heads          [[buffer(10)]],
    const constant float& scale             [[buffer(11)]],
    const device uchar* attention_mask      [[buffer(12)]],
    const device uint& attention_mask_stride[[buffer(13)]],
    const device uchar* attention_mask_enabled [[buffer(14)]],
    const constant int& window                 [[buffer(15)]],
    const device T* sinks                      [[buffer(16)]],  // unread; the seat is the contract
    const constant int& n_rows                 [[buffer(17)]],
    device float* lse                          [[buffer(18)]],  // [N, n_q_heads], base two
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  threadgroup T ktile[sdpa_kt<D>() * sdpa_kstride<D, KEY_PER_LANE>()];
  threadgroup T vtile[sdpa_kt<D>() * D];
  threadgroup T qtile[sdpa_kq<D, KEY_PER_LANE>()];
  threadgroup float ptile[sdpa_kp_tile<D, KEY_PER_LANE>()];
  sdpa_paged_tiled_body<T, D, V, false, KEY_PER_LANE, true>(
      queries, k_pages, v_pages, out, gqa_factor, position_ids, req_of_token,
      kv_page_indices, kv_page_indptr, page_size, n_kv_heads, scale,
      attention_mask, attention_mask_stride, attention_mask_enabled, window,
      sinks, lse, n_rows,
      /*q_row_pitch=*/0, /*o_row_pitch=*/0,
      ktile, vtile, qtile, ptile, tid, tpg, simd_gid, simd_lid);
}

/// The prefill's twin: q and out rows a uniform pitch apart. See the body.
template <typename T, int D, int V = D, bool WITH_SINK = false, bool KEY_PER_LANE = false>
[[kernel]] [[max_total_threads_per_threadgroup(1024)]] void sdpa_paged_tiled_strided(
    const device T* queries     [[buffer(0)]],   // [N, n_q_heads, D]
    const device T* k_pages     [[buffer(1)]],   // [num_pages, page_size, n_kv_heads, D]
    const device T* v_pages     [[buffer(2)]],
    device T* out               [[buffer(3)]],   // [N, n_q_heads, V]
    const constant int& gqa_factor          [[buffer(4)]],
    const device int* position_ids          [[buffer(5)]],
    const device int* req_of_token          [[buffer(6)]],
    const device uint* kv_page_indices      [[buffer(7)]],
    const device uint* kv_page_indptr       [[buffer(8)]],
    const constant int& page_size           [[buffer(9)]],
    const constant int& n_kv_heads          [[buffer(10)]],
    const constant float& scale             [[buffer(11)]],
    const device uchar* attention_mask      [[buffer(12)]],
    const device uint& attention_mask_stride[[buffer(13)]],
    const device uchar* attention_mask_enabled [[buffer(14)]],
    const constant int& window                 [[buffer(15)]],
    const device T* sinks                      [[buffer(16)]],  // [n_q_heads], WITH_SINK only
    const constant int& n_rows                 [[buffer(17)]],  // N; the grid rounds up past it,
    const constant int& q_row_pitch            [[buffer(18)]],
    const constant int& o_row_pitch            [[buffer(19)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  threadgroup T ktile[sdpa_kt<D>() * sdpa_kstride<D, KEY_PER_LANE>()];
  threadgroup T vtile[sdpa_kt<D>() * D];
  threadgroup T qtile[sdpa_kq<D, KEY_PER_LANE>()];
  threadgroup float ptile[sdpa_kp_tile<D, KEY_PER_LANE>()];
  sdpa_paged_tiled_body<T, D, V, WITH_SINK, KEY_PER_LANE, false>(
      queries, k_pages, v_pages, out, gqa_factor, position_ids, req_of_token,
      kv_page_indices, kv_page_indptr, page_size, n_kv_heads, scale,
      attention_mask, attention_mask_stride, attention_mask_enabled, window,
      sinks, /*lse=*/nullptr, n_rows,
      q_row_pitch, o_row_pitch,
      ktile, vtile, qtile, ptile, tid, tpg, simd_gid, simd_lid);
}

#define instantiate_sdpa_tiled_strided(fn, name, itype, d, v, sink, kpl)   \
  template [[host_name(fn "_" #name "_d_" #d)]]                            \
  [[kernel]] void sdpa_paged_tiled_strided<itype, d, v, sink, kpl>(        \
      const device itype*, const device itype*, const device itype*,       \
      device itype*, const constant int&, const device int*,               \
      const device int*, const device uint*, const device uint*,           \
      const constant int&, const constant int&, const constant float&,     \
      const device uchar*, const device uint&, const device uchar*,        \
      const constant int&, const device itype*, const constant int&,       \
      const constant int&, const constant int&,                            \
      uint3, uint3, uint, uint);

// qwen3.5's head width; the only family whose prefill batches this kernel.
instantiate_sdpa_tiled_strided("sdpa_paged_tiled_strided", bfloat16, bfloat, 256, 256, false, false)

#define instantiate_sdpa_tiled_impl(fn, name, itype, d, v, sink, kpl)      \
  template [[host_name(fn "_" #name "_d_" #d)]]                            \
  [[kernel]] void sdpa_paged_tiled<itype, d, v, sink, kpl>(                \
      const device itype*, const device itype*, const device itype*,       \
      device itype*, const constant int&, const device int*,               \
      const device int*, const device uint*, const device uint*,           \
      const constant int&, const constant int&, const constant float&,     \
      const device uchar*, const device uint&, const device uchar*,        \
      const constant int&, const device itype*, const constant int&,       \
      uint3, uint3, uint, uint);

instantiate_sdpa_tiled_impl("sdpa_paged_tiled", bfloat16, bfloat, 128, 128, false, true)
instantiate_sdpa_tiled_impl("sdpa_paged_tiled", bfloat16, bfloat, 64, 64, false, true)
// gemma4's two head widths. `KT = 4096 / D` keeps the staged tile at 16 KB of
// each of K and V no matter how wide the head is, so these cost the same
// threadgroup memory as the narrow ones -- only the pass count changes.
instantiate_sdpa_tiled_impl("sdpa_paged_tiled", bfloat16, bfloat, 256, 256, false, false)
instantiate_sdpa_tiled_impl("sdpa_paged_tiled", bfloat16, bfloat, 512, 512, false, false)
// gpt-oss's head width, with the learned per-head sink in the denominator. The
// tiled kernel already merges the sink; it had simply never been asked for with
// one, because the family that has sinks was not tiling.
instantiate_sdpa_tiled_impl("sdpa_paged_tiled_sink", bfloat16, bfloat, 64, 64, true, true)

#define instantiate_sdpa_tiled_lse(fn, name, itype, d, v, kpl)             \
  template [[host_name(fn "_" #name "_d_" #d)]]                            \
  [[kernel]] void sdpa_paged_tiled_lse<itype, d, v, kpl>(                  \
      const device itype*, const device itype*, const device itype*,       \
      device itype*, const constant int&, const device int*,               \
      const device int*, const device uint*, const device uint*,           \
      const constant int&, const constant int&, const constant float&,     \
      const device uchar*, const device uint&, const device uchar*,        \
      const constant int&, const device itype*, const constant int&,       \
      device float*, uint3, uint3, uint, uint);

// gpt-oss's head width, and ONLY gpt-oss's.
//
// `attention.prefill_lse` is stated by two texts in this tree and neither of
// the other two SKUs on this plane is one of them: gpt-oss states it at 64,
// and deepseek-v4 states it on the `Mla` family, which this plane does not
// claim at all. A width nothing states would be one more entry point compiled
// into every build, signed, pipeline-buildable and unreachable, which is what
// the `_p32` note further down is about. `head_point` refuses the widths that
// are not here, by name, at the fire.
instantiate_sdpa_tiled_lse("sdpa_paged_tiled_lse", bfloat16, bfloat, 64, 64, true)

#define instantiate_sdpa_paged_impl(fn, name, itype, d, v, sink)           \
  template [[host_name(fn "_" #name "_d_" #d)]]                            \
  [[kernel]] void sdpa_paged_decode<itype, d, v, sink, 0, false, 32>(      \
      const device itype*, const device itype*, const device itype*,       \
      device itype*, const constant int&, const device int*,               \
      const device int*, const device uint*, const device uint*,           \
      const constant int&, const constant int&, const constant float&,     \
      const device uchar*, const device uint&, const device uchar*,        \
      const constant int&, const device itype*,                            \
      uint3, uint3, uint, uint);

#define instantiate_sdpa_paged(name, itype, d, v)                          \
  instantiate_sdpa_paged_impl("sdpa_paged_decode", name, itype, d, v, false)

#define instantiate_sdpa_paged_sink(name, itype, d, v)                     \
  instantiate_sdpa_paged_impl("sdpa_paged_decode_sink", name, itype, d, v, true)

instantiate_sdpa_paged(bfloat16, bfloat, 256, 256)
instantiate_sdpa_paged(bfloat16, bfloat, 512, 512)  // gemma4 full-attn (head_dim 512)
instantiate_sdpa_paged(bfloat16, bfloat, 128, 128)  // llama / qwen (head_dim 128)
instantiate_sdpa_paged(bfloat16, bfloat, 64, 64)  // llama 3.2 1B/3B (head_dim 64)
instantiate_sdpa_paged_sink(bfloat16, bfloat, 64, 64)  // gpt-oss (head_dim 64)

#define instantiate_sdpa_paged_lse(name, itype, d, v)                      \
  template [[host_name("sdpa_paged_decode_lse_" #name "_d_" #d)]]          \
  [[kernel]] void sdpa_paged_decode_lse<itype, d, v>(                      \
      const device itype*, const device itype*, const device itype*,       \
      device itype*, const constant int&, const device int*,               \
      const device int*, const device uint*, const device uint*,           \
      const constant int&, const constant int&, const constant float&,     \
      const device uchar*, const device uint&, const device uchar*,        \
      const constant int&, const device itype*, device float*,             \
      uint3, uint3, uint, uint);

// gpt-oss (head_dim 64), for the reason the tiled twin's block gives.
instantiate_sdpa_paged_lse(bfloat16, bfloat, 64, 64)

#define instantiate_sdpa_paged_p32(name, itype, d, v)                       \
  template [[host_name("sdpa_paged_decode_" #name "_d_" #d "_p32")]]        \
  [[kernel]] void sdpa_paged_decode<itype, d, v, false, 32, true, 32>(      \
      const device itype*, const device itype*, const device itype*,        \
      device itype*, const constant int&, const device int*,                \
      const device int*, const device uint*, const device uint*,            \
      const constant int&, const constant int&, const constant float&,      \
      const device uchar*, const device uint&, const device uchar*,         \
      const constant int&, const device itype*,                             \
      uint3, uint3, uint, uint);

instantiate_sdpa_paged_p32(bfloat16, bfloat, 128, 128)
instantiate_sdpa_paged_p32(bfloat16, bfloat, 64, 64)

#define instantiate_sdpa_paged_short(name, itype, d, v, sg)                  \
  template [[host_name("sdpa_paged_decode_" #name "_d_" #d "_p32_sg" #sg)]] \
  [[kernel]] void sdpa_paged_decode<itype, d, v, false, 32, true, sg>(       \
      const device itype*, const device itype*, const device itype*,         \
      device itype*, const constant int&, const device int*,                 \
      const device int*, const device uint*, const device uint*,             \
      const constant int&, const constant int&, const constant float&,       \
      const device uchar*, const device uint&, const device uchar*,          \
      const constant int&, const device itype*,                              \
      uint3, uint3, uint, uint);

instantiate_sdpa_paged_short(bfloat16, bfloat, 64, 64, 8)
