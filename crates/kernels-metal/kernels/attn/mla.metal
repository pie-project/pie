#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

// Multi-head latent attention, the metal mirror of `kernels-cuda`'s
// `attn/mla.cuh`. The projection split/norm/rope, the paged latent appender,
// the q-absorb batched matmul, and the naive simd flash engine — the one CUDA
// path that is a plain scalar/warp kernel rather than a Hopper/Blackwell mma.
//
// A CUDA warp is 32 lanes; a metal simdgroup is 32 lanes; the two dot-product
// reductions map across unchanged (`simd_sum` for `__shfl_xor` fold). The one
// deliberate divergence from `mla.cuh`: the flash kernel bounds its key sweep
// by `position_ids[row]` (the fire's causal position), not by a re-derived
// `kv_last_page_lens` — the metal pool carries no last-page table, and the
// paged sdpa family already reads the sequence bound this way. `j_end = pos+1`
// is the causal prefill bound and, for a decode row whose position is the last
// cached slot, is exactly the full cached length the CUDA `causal=false` decode
// sweeps. One kernel therefore serves both decode and prefill.

// Largest per-lane strip the register arrays hold: CKV (latent rank) up to 512
// (16 elements of 32-lane strip) and KPE (rope) up to 128 (4). The dispatch
// refuses any wider geometry, so these ceilings are never exceeded.
constant constexpr int kMaxCkvPer = 16;
constant constexpr int kMaxKpePer = 4;

// ── split kv_a into the rmsnormed latent and the rope tail ──────────────────
//
// One threadgroup per row. The rope tail (`k_pe`) is a straight copy of the
// last `rope` lanes of the source row; the latent (`kv_c`) is the first
// `kv_lora` lanes, rms-normalized with the learned weight. Mirrors
// `pie::attn::mla_latents<T, 256>`.
[[kernel]] void mla_latents_bfloat16(
    const device bfloat* kv_a          [[buffer(0)]],
    const device bfloat* norm_weight   [[buffer(1)]],
    device bfloat* kv_c                [[buffer(2)]],
    device bfloat* k_pe                [[buffer(3)]],
    const constant int& kv_lora        [[buffer(4)]],
    const constant int& rope           [[buffer(5)]],
    const constant int& src_row_stride [[buffer(6)]],
    const constant float& eps          [[buffer(7)]],
    uint gid            [[threadgroup_position_in_grid]],
    uint lid            [[thread_position_in_threadgroup]],
    uint simd_lane      [[thread_index_in_simdgroup]],
    uint simd_group     [[simdgroup_index_in_threadgroup]],
    uint tg_size        [[threads_per_threadgroup]]) {
  const device bfloat* row = kv_a + size_t(gid) * size_t(src_row_stride);

  for (int d = int(lid); d < rope; d += int(tg_size)) {
    k_pe[size_t(gid) * size_t(rope) + d] = row[kv_lora + d];
  }

  float local = 0.0f;
  for (int d = int(lid); d < kv_lora; d += int(tg_size)) {
    const float v = float(row[d]);
    local += v * v;
  }
  threadgroup float partials[32];
  threadgroup float inv_rms[1];
  local = simd_sum(local);
  if (simd_group == 0) partials[simd_lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0) partials[simd_group] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    float acc = simd_sum(partials[simd_lane]);
    if (simd_lane == 0) inv_rms[0] = precise::rsqrt(acc / float(kv_lora) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = inv_rms[0];

  for (int d = int(lid); d < kv_lora; d += int(tg_size)) {
    const float v = float(row[d]);
    const float w = float(norm_weight[d]);
    kv_c[size_t(gid) * size_t(kv_lora) + d] = bfloat(v * inv * w);
  }
}

// ── split q_b into per-head nope and rope planes ────────────────────────────
//
// One thread per source element. Mirrors `pie::attn::mla_split_q_b<T>`: the
// row-major `[tokens, heads, nope+rope]` block cut into `[tokens, heads, nope]`
// and `[tokens, heads, rope]`.
[[kernel]] void mla_split_q_b_bfloat16(
    const device bfloat* q_b   [[buffer(0)]],
    device bfloat* q_nope      [[buffer(1)]],
    device bfloat* q_pe        [[buffer(2)]],
    const constant int& total  [[buffer(3)]],
    const constant int& heads  [[buffer(4)]],
    const constant int& nope   [[buffer(5)]],
    const constant int& rope   [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  const int i = int(gid);
  if (i >= total) return;
  const int per = nope + rope;
  const int d = i % per;
  const int h = (i / per) % heads;
  const int n = i / (heads * per);
  const bfloat v = q_b[i];
  if (d < nope) {
    q_nope[(size_t(n) * heads + h) * nope + d] = v;
  } else {
    q_pe[(size_t(n) * heads + h) * rope + (d - nope)] = v;
  }
}

// ── append one latent row (ckv beside kpe) into the paged pool ──────────────
//
// One thread per (lane, row). The metal appender addresses by the op-named
// `write_page`/`write_offset` tables — the write-geometry seam the paged
// family closes — rather than re-deriving the destination slot from the
// read-side CSR the way `mla.cuh`'s appender does. `ckv` lands in the keys
// pages (rank-wide, one kv head), `kpe` in the values pages (rope-wide).
[[kernel]] void mla_kv_append_bfloat16(
    const device bfloat* kv_c    [[buffer(0)]],
    const device bfloat* k_pe    [[buffer(1)]],
    device bfloat* ckv_pages     [[buffer(2)]],
    device bfloat* kpe_pages     [[buffer(3)]],
    const device uint* w_page    [[buffer(4)]],
    const device uint* w_off     [[buffer(5)]],
    const constant int& page_size [[buffer(6)]],
    const constant int& kv_lora   [[buffer(7)]],
    const constant int& rope      [[buffer(8)]],
    uint2 tid [[thread_position_in_grid]]) {
  const int d = int(tid.x);
  const int row = int(tid.y);
  const size_t slot = size_t(w_page[row]) * size_t(page_size) + size_t(w_off[row]);
  if (d < kv_lora) {
    ckv_pages[slot * size_t(kv_lora) + d] = kv_c[size_t(row) * size_t(kv_lora) + d];
  }
  if (d < rope) {
    kpe_pages[slot * size_t(rope) + d] = k_pe[size_t(row) * size_t(rope) + d];
  }
}

// ── absorb kv_b's up-projection into q ──────────────────────────────────────
//
// Per-head matmul: `q_latent[t,h,i] = sum_j q_nope[t,h,j] * kv_b[h][j][i]`,
// where `kv_b` is the checkpoint's `[heads*(nope+v_dim), rank]` row-major
// weight and head `h`'s nope block is rows `[h*(nope+v_dim) .. +nope)`. This is
// the strided-batched `CUBLAS_OP_N` GEMM of `mla.cuh`'s `absorb_q`, written as
// one thread per output element (the parity scale keeps the naive form well
// inside budget). One thread computes `q_latent[t,h,i]` for a fixed rank lane
// `i`, head `h`, token `t`.
[[kernel]] void mla_absorb_q_bfloat16(
    const device bfloat* q_nope [[buffer(0)]],
    const device bfloat* kv_b   [[buffer(1)]],
    device bfloat* q_latent     [[buffer(2)]],
    const constant int& heads   [[buffer(3)]],
    const constant int& rank    [[buffer(4)]],
    const constant int& nope    [[buffer(5)]],
    const constant int& v_dim   [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]]) {
  const int i = int(tid.x);   // rank lane
  const int h = int(tid.y);   // head
  const int t = int(tid.z);   // token
  if (i >= rank) return;
  const size_t qn_base = (size_t(t) * heads + h) * nope;
  const size_t kb_base = size_t(h) * size_t(nope + v_dim) * size_t(rank);
  float acc = 0.0f;
  for (int j = 0; j < nope; ++j) {
    acc += float(q_nope[qn_base + j]) * float(kv_b[kb_base + size_t(j) * rank + i]);
  }
  q_latent[(size_t(t) * heads + h) * rank + i] = bfloat(acc);
}

// ── map the latent reading back through kv_b's value planes ────────────────
//
// Per-head matmul: `o[t,h,j] = sum_i latent[t,h,i] * kv_b_v[h][j][i]`, the
// other half of the absorb. `mla.cuh`'s `absorb_out` fires this as the
// `CUBLAS_OP_T` strided-batched GEMM whose A operand starts at
// `kv_b.ptr.wrapping_add(2 * nope * rank)`.
//
// # WHERE THE V BLOCK BEGINS, AND WHY THE `2` IS NOT A FACTOR OF TWO
//
// `Tensor::ptr` is a raw device ADDRESS and `wrapping_add` on it is BYTE
// arithmetic — the same units `kernels-cuda`'s own `plane_bytes =
// rows * width * 2` guard is written in. The `2` is `sizeof(bf16)`, not a
// doubled stride. So the V block's base is `nope * rank` ELEMENTS past
// `kv_b`, and with the batch stride `(nope + v_dim) * rank` the value planes
// land exactly where the standard DeepSeek packing puts them:
//
//   kv_b is row-major `[heads * (nope + v_dim), rank]`; head `h` owns the
//   `(nope + v_dim)` rows starting at `h*(nope+v_dim)`; the FIRST `nope` of
//   those rows are the key-up block `W_UK` that `mla_absorb_q` reads, and the
//   NEXT `v_dim` rows are the value-up block `W_UV` this kernel reads. Heads
//   are OUTER, the two blocks are contiguous within a head, each row `rank`
//   wide.
//
// So head `h`, value row `j`, rank lane `i` is
// `kv_b[h*(nope+v_dim)*rank + (nope + j)*rank + i]` — read as `A^T` by the
// GEMM (`lda = rank`, `op_a = T`), which is the same thing as reading this
// row-major `[v_dim, rank]` block directly. `the_absorbed_pair_is_the_
// unabsorbed_attention` in `engine-metal/tests/mla_on_device.rs` is the
// measurement that settles it: any other base answers garbage, not epsilon.
//
// One thread per output element, as `mla_absorb_q`.
[[kernel]] void mla_absorb_out_bfloat16(
    const device bfloat* latent [[buffer(0)]],
    const device bfloat* kv_b   [[buffer(1)]],
    device bfloat* o            [[buffer(2)]],
    const constant int& heads   [[buffer(3)]],
    const constant int& rank    [[buffer(4)]],
    const constant int& v_dim   [[buffer(5)]],
    const constant int& nope    [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]]) {
  const int j = int(tid.x);   // value lane
  const int h = int(tid.y);   // head
  const int t = int(tid.z);   // token
  if (j >= v_dim) return;
  const size_t lat_base = (size_t(t) * heads + h) * size_t(rank);
  const size_t wv_base =
      size_t(h) * size_t(nope + v_dim) * size_t(rank) + size_t(nope) * size_t(rank);
  float acc = 0.0f;
  for (int i = 0; i < rank; ++i) {
    acc += float(latent[lat_base + i]) * float(kv_b[wv_base + size_t(j) * rank + i]);
  }
  o[(size_t(t) * heads + h) * size_t(v_dim) + j] = bfloat(acc);
}

// ── naive paged flash over the latent kv ────────────────────────────────────
//
// One simdgroup per (head, query row). The 32 lanes split the latent rank
// (`ckv`, up to 512 => 16 per lane) and the rope width (`kpe`, up to 128 => 4
// per lane); each key contributes `q_nope . ckv + q_pe . kpe`, folded across
// the simdgroup with `simd_sum`, then an online-softmax accumulation exactly as
// `mla_naive_paged_kernel`. The output is the latent-space reading
// `o[row,head,:ckv]` that `mla_absorb_out` maps back to value space.
//
// # Dense and selected are ONE body, as they are one kernel on CUDA
//
// `mla_naive_paged_kernel` takes a nullable `const i32* selection` beside an
// `int top_k`: null sweeps `[0, j_end)` in key order, non-null sweeps the row
// `selection + t*top_k` and attends the keys it names. Metal has no way to
// leave a bound buffer unbound at an index the shader declares, so the two
// modes are two entrypoints over the one inlined body below, and the body
// keeps the CUDA predicate verbatim — `srow != nullptr`. After inlining each
// entry the branch is a constant, so the dense point pays nothing for the
// sparse one existing.
//
// The selection rows `index_topk_paged` publishes are ascending key ids with
// a **-1 padded tail**, and the CUDA reader `continue`s on any entry outside
// `[0, j_end)` rather than stopping — a padded slot contributes no key, and a
// slot naming a position the causal bound does not reach is dropped, not
// clamped. This body does the same. The skip is simdgroup-uniform (`j`
// depends on the row, never the lane), so the `simd_sum` fold below stays
// fully populated exactly as `__shfl_xor_sync(0xffffffffu, ...)` does.
inline void mla_naive_paged_body(
    const device bfloat* q_nope,
    const device bfloat* q_pe,
    const device bfloat* ckv_pages,
    const device bfloat* kpe_pages,
    device bfloat* o,
    const device int* position_ids,
    const device int* req_of_token,
    const device uint* kv_page_indices,
    const device uint* kv_page_indptr,
    const device int* selection,
    int top_k,
    int page_size,
    int heads,
    int ckv,
    int kpe,
    float sm_scale,
    uint2 gid,
    uint lane) {
  const int h   = int(gid.x);
  const int row = int(gid.y);
  const int per  = ckv / 32;   // <= kMaxCkvPer
  const int pper = kpe / 32;   // <= kMaxKpePer

  const int r      = req_of_token[row];
  const int q_pos  = position_ids[row];
  const int j_end  = q_pos + 1;
  const int page_base = int(kv_page_indptr[r]);

  const device int* srow =
      (selection != nullptr) ? selection + size_t(row) * size_t(top_k) : nullptr;

  const device bfloat* qn = q_nope + (size_t(row) * heads + h) * size_t(ckv);
  const device bfloat* qp = q_pe   + (size_t(row) * heads + h) * size_t(kpe);
  float qn_r[kMaxCkvPer];
  float qp_r[kMaxKpePer];
  for (int i = 0; i < per; ++i)  qn_r[i] = float(qn[lane + i * 32]);
  for (int i = 0; i < pper; ++i) qp_r[i] = float(qp[lane + i * 32]);

  float acc[kMaxCkvPer];
  for (int i = 0; i < per; ++i) acc[i] = 0.0f;
  float m = -3.0e38f, lsum = 0.0f;

  const int steps = (srow != nullptr) ? top_k : j_end;
  for (int n = 0; n < steps; ++n) {
    int j = n;
    if (srow != nullptr) {
      j = srow[n];
      if (j < 0 || j >= j_end) continue;
    }
    const uint page = kv_page_indices[page_base + j / page_size];
    const size_t slot = size_t(page) * size_t(page_size) + size_t(j % page_size);
    const device bfloat* ckv_j = ckv_pages + slot * size_t(ckv);
    const device bfloat* kpe_j = kpe_pages + slot * size_t(kpe);

    float kv[kMaxCkvPer];
    float pd = 0.0f;
    for (int i = 0; i < per; ++i) {
      kv[i] = float(ckv_j[lane + i * 32]);
      pd += qn_r[i] * kv[i];
    }
    for (int i = 0; i < pper; ++i) {
      pd += qp_r[i] * float(kpe_j[lane + i * 32]);
    }
    pd = simd_sum(pd);
    const float score = pd * sm_scale;
    const float m_new = max(m, score);
    const float corr = fast::exp(m - m_new);
    const float p = fast::exp(score - m_new);
    lsum = lsum * corr + p;
    for (int i = 0; i < per; ++i) acc[i] = acc[i] * corr + p * kv[i];
    m = m_new;
  }

  const float inv = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;
  device bfloat* orow = o + (size_t(row) * heads + h) * size_t(ckv);
  for (int i = 0; i < per; ++i) orow[lane + i * 32] = bfloat(acc[i] * inv);
}

// The dense reader: `attention.mla_decode` and `attention.mla_prefill`.
[[kernel]] void mla_naive_paged_bfloat16(
    const device bfloat* q_nope     [[buffer(0)]],
    const device bfloat* q_pe       [[buffer(1)]],
    const device bfloat* ckv_pages  [[buffer(2)]],
    const device bfloat* kpe_pages  [[buffer(3)]],
    device bfloat* o                [[buffer(4)]],
    const device int* position_ids  [[buffer(5)]],
    const device int* req_of_token  [[buffer(6)]],
    const device uint* kv_page_indices [[buffer(7)]],
    const device uint* kv_page_indptr  [[buffer(8)]],
    const constant int& page_size   [[buffer(9)]],
    const constant int& heads       [[buffer(10)]],
    const constant int& ckv         [[buffer(11)]],
    const constant int& kpe         [[buffer(12)]],
    const constant float& sm_scale  [[buffer(13)]],
    uint2 gid   [[threadgroup_position_in_grid]],
    uint lane   [[thread_index_in_simdgroup]]) {
  mla_naive_paged_body(q_nope, q_pe, ckv_pages, kpe_pages, o, position_ids,
                       req_of_token, kv_page_indices, kv_page_indptr,
                       (const device int*)nullptr, 0, page_size, heads, ckv,
                       kpe, sm_scale, gid, lane);
}

// The sparse reader: `attention.mla_decode_selected` and
// `attention.mla_prefill_selected`. The first fourteen seats are the dense
// point's, unmoved; the selection plane and its budget ride behind them.
[[kernel]] void mla_naive_paged_selected_bfloat16(
    const device bfloat* q_nope     [[buffer(0)]],
    const device bfloat* q_pe       [[buffer(1)]],
    const device bfloat* ckv_pages  [[buffer(2)]],
    const device bfloat* kpe_pages  [[buffer(3)]],
    device bfloat* o                [[buffer(4)]],
    const device int* position_ids  [[buffer(5)]],
    const device int* req_of_token  [[buffer(6)]],
    const device uint* kv_page_indices [[buffer(7)]],
    const device uint* kv_page_indptr  [[buffer(8)]],
    const constant int& page_size   [[buffer(9)]],
    const constant int& heads       [[buffer(10)]],
    const constant int& ckv         [[buffer(11)]],
    const constant int& kpe         [[buffer(12)]],
    const constant float& sm_scale  [[buffer(13)]],
    const device int* selection     [[buffer(14)]],
    const constant int& top_k       [[buffer(15)]],
    uint2 gid   [[threadgroup_position_in_grid]],
    uint lane   [[thread_index_in_simdgroup]]) {
  mla_naive_paged_body(q_nope, q_pe, ckv_pages, kpe_pages, o, position_ids,
                       req_of_token, kv_page_indices, kv_page_indptr, selection,
                       top_k, page_size, heads, ckv, kpe, sm_scale, gid, lane);
}
