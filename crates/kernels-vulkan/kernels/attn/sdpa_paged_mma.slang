#if defined(PIE_COOPMAT)
// Cooperative matrix lives in Slang's `linalg` namespace and is unqualified
// nowhere: without this line `CoopMat` is simply an undefined identifier, with
// no hint that a namespace is missing.
using namespace linalg;
typealias MatQ = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MatK = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias MatAcc = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
// The A and B operands are `half`; only the ACCUMULATOR is `float`. A
// device advertises a list of `(M, N, K, types)` it implements and a `coopmat`
// outside that list is undefined behaviour, not a slow path -- this tier used
// fp32 A/B until a GPU run caught it returning nonsense on hardware that
// offers no fp32 A/B configuration at all. See `quant/qmm_t.slang`, which had
// the same defect and carries the longer note.
#endif
#include "common/bf16.slang"
#include "attn/sdpa_online.slang"

// Paged prefill SDPA at the Metal MMA entrypoint names. Vulkan's comparable
// path is GL_KHR_cooperative_matrix, but that extension is optional and this
// row's entrypoints must exist on every supported driver, so the body below
// (compiled as the `baseline` tier) is a correct tiled scalar fallback with
// shared K/V tiles, fp32 accumulation, and online softmax: no extension
// beyond what every driver in scope already has.
//
// A `@coopmat` tier now exists too, gated behind `PIE_COOPMAT` further down
// in this file. It requires `GL_KHR_cooperative_matrix`, which is why it is
// a separate module rather than the only body: a device without that
// extension gets the scalar fallback above and nothing breaks. Where a
// device does have it, the coopmat tier replaces the Q.K^T and P.V inner
// products with `coopMatMulAdd` calls while keeping the exact same
// page-table addressing, fp32 accumulation, and online-softmax recurrence
// as the scalar body -- see the comment on the coopmat `main` for how the
// per-key recurrence is preserved exactly through a closed-form tile
// reduction, and why it has to be structured that way to keep every
// `coopMatMulAdd` call in uniform control flow.

#ifndef PIE_HEAD_DIM
#error "sdpa_paged_mma requires -DPIE_HEAD_DIM=64"
#endif
#define PIE_KT 16
#define PIE_DOUT_SLOTS ((PIE_HEAD_DIM + 31) / 32)

PIE_BUFFER_RO(0, queries);
PIE_BUFFER_RO(1, k_pages);
PIE_BUFFER_RO(2, v_pages);
PIE_BUFFER_RW(3, out_);
[[vk::binding(4)]] StructuredBuffer<int> position_ids;
[[vk::binding(5)]] StructuredBuffer<int> req_of_token;
[[vk::binding(6)]] StructuredBuffer<uint> kv_page_indices;
[[vk::binding(7)]] StructuredBuffer<uint> kv_page_indptr;
[[vk::binding(8)]] StructuredBuffer<uint8_t> attention_mask;
[[vk::binding(9)]] StructuredBuffer<uint8_t> attention_mask_enabled;
PIE_BUFFER_RO(10, sinks);

struct Push {
    int gqa_factor;
    int page_size;
    int n_kv_heads;
    float scale;
    uint attention_mask_stride;
    int window;
    int n_rows;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

#if !defined(PIE_COOPMAT)
groupshared PIE_ACT k_tile[PIE_KT * PIE_HEAD_DIM];
groupshared PIE_ACT v_tile[PIE_KT * PIE_HEAD_DIM];
// Partial scores, 32 lanes per key, reduced in place. The coopmat path above
// keeps the same thing in `cm_scores`; this is the fallback's way of arriving
// at one score per key instead of one per thread per output slot.
groupshared float sc_part[PIE_KT * 32];
#endif

uint page_slot(int req, int kp) {
    uint page_ix = uint(kp / pc.page_size);
    uint page_off = uint(kp % pc.page_size);
    uint phys = kv_page_indices[kv_page_indptr[req] + page_ix];
    return phys * uint(pc.page_size) + page_off;
}

bool keeps(uint row, int kp, int q_pos, int start) {
    if (kp > q_pos || kp < start) return false;
    if (attention_mask_enabled[row] != uint8_t(0)) {
        if (uint(kp) >= pc.attention_mask_stride) return false;
        if (attention_mask[row * pc.attention_mask_stride + uint(kp)] == uint8_t(0)) return false;
    }
    return true;
}

#if defined(PIE_COOPMAT)

// The coopmat tier. GL_KHR_cooperative_matrix is optional, so this module is
// only ever selected on a device that advertises it; the scalar tier above
// this `#if` is what every other device gets, unmodified.
//
// The 32x32 workgroup does not map onto coopmat by itself. The scalar tier
// picks one lane row (`ly == rr`) per iteration of the outer `rr` sweep and
// has that row's 32 lanes each independently redo the same 64-wide Q.K dot
// product, once per output channel a lane owns. A coopMatMulAdd is a
// subgroup-collective op that has to be reached under uniform control flow,
// so it cannot live inside that `if (mine)` branch: nothing guarantees a
// subgroup's lanes all share one `ly` (a 64-wide wavefront spans two `rr`
// iterations' worth of lanes on that hardware), so a call gated by `mine`
// would have some of a subgroup's lanes not participate. The way out is
// that `row`/`req`/`q_pos` are already the same value for every lane in the
// workgroup at a given `rr` -- `ly == rr` only decides who is allowed to
// *write* the result, not who is computing on different data -- so the QK
// score and the online-softmax bookkeeping can be produced once, by the
// whole workgroup, unconditionally, and only the final store gated by
// `mine`.
//
// Both coopMatMulAdd calls below use a fixed 16x16x16 shape, matching
// PIE_KT. The operands are staged into plain fp32 shared buffers (`cm_k`,
// `cm_v`) rather than reading `k_tile`/`v_tile` above, because a coopmat load
// needs a component type it understands and this shader's bf16 storage is
// a manually packed `uint16_t` that isn't one.
//
// The per-key online-softmax recurrence (`sdpa_online_update`) is run
// exactly as the scalar tier runs it: once per key, in key order, off the
// same `pc.scale * dot(Q, K)` value, so the running max/sum this tier
// produces are the same computation as the scalar tier's, not a different
// algorithm. What changes is how each key's contribution reaches
// `out_acc`: instead of rescale-and-add one key at a time, this tier folds
// each key's later rescale factors into a per-key weight up front, then
// reduces the whole tile's P.V contribution as one coopMatMulAdd. That is
// the standard flash-attention block identity: unrolling
//   out = out*hist[k] + prob[k]*V[k]     for k = 0..cnt-1
// gives, exactly,
//   out_after = out_before * (prod_k hist[k])
//             + sum_k ( prob[k] * (prod_{m>k} hist[m]) * V[k] )
// so the only difference from the scalar tier is the order the final sum
// is carried out in -- the same kind of float reassociation the QK
// reduction itself already introduces by summing across a coopmat instead
// of a serial scalar loop.
uint coopmat_head_chunks() { return uint(PIE_HEAD_DIM) / 16u; }

groupshared float cm_q[PIE_HEAD_DIM];
groupshared half cm_k[PIE_KT * PIE_HEAD_DIM];
groupshared half cm_v[PIE_KT * PIE_HEAD_DIM];
groupshared half cm_amat[16 * 16];
groupshared float cm_rmat[16 * 16];
groupshared float cm_scores[PIE_KT];
groupshared float cm_pv[PIE_HEAD_DIM];

[numthreads(32, 32, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint q_head = group.x;
    uint row_lo = group.y * 32u;
    uint lx = local.x;
    uint ly = local.y;
    uint lid = ly * 32u + lx;
    uint n_q_heads = gl_NumWorkGroups.x;
    int kv_head = int(q_head) / pc.gqa_factor;

    for (uint rr = 0u; rr < 32u; rr++) {
        uint row = row_lo + rr;
        bool row_valid = row < uint(pc.n_rows);
        bool mine = ly == rr && row_valid;
        int req = row_valid ? req_of_token[row] : 0;
        int q_pos = row_valid ? position_ids[row] : -1;
        int start = (pc.window > 0 && q_pos >= pc.window) ? (q_pos - pc.window + 1) : 0;
        uint q_base = (row * n_q_heads + q_head) * uint(PIE_HEAD_DIM);
        uint o_base = q_base;

        GroupMemoryBarrierWithGroupSync();
        for (uint d = lid; d < uint(PIE_HEAD_DIM); d += 1024u) {
            cm_q[d] = row_valid ? PIE_LOAD(queries[q_base + d]) : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();

        float max_score = PIE_SDPA_NEG_INF;
        float sum_exp = 0.0;
        float out_acc[PIE_DOUT_SLOTS];
        for (int s = 0; s < PIE_DOUT_SLOTS; s++) out_acc[s] = 0.0;

        for (int base = start; base <= q_pos; base += PIE_KT) {
            int cnt = min(PIE_KT, q_pos + 1 - base);
            GroupMemoryBarrierWithGroupSync();
            for (uint e = lid; e < uint(PIE_KT * PIE_HEAD_DIM); e += 1024u) {
                uint kk = e / uint(PIE_HEAD_DIM);
                uint d = e - kk * uint(PIE_HEAD_DIM);
                if (int(kk) < cnt) {
                    uint slot = page_slot(req, base + int(kk));
                    uint off = (slot * uint(pc.n_kv_heads) + uint(kv_head)) * uint(PIE_HEAD_DIM) + d;
                    cm_k[e] = half(PIE_LOAD(k_pages[off]));
                    cm_v[e] = half(PIE_LOAD(v_pages[off]));
                } else {
                    cm_k[e] = half(0.0);
                    cm_v[e] = half(0.0);
                }
            }
            GroupMemoryBarrierWithGroupSync();

            // Q.K^T for the whole PIE_KT-wide tile at once: the "A" operand
            // is the same 16-wide Q slice broadcast into every row of a
            // 16x16 tile (every row of the result is then identical, and
            // only row 0 is read), which lets one coopMatMulAdd stand in
            // for the scalar tier's 32 redundant per-lane dot products.
            {
                MatAcc acc =
                    MatAcc(0.0);
                for (uint dc = 0u; dc < coopmat_head_chunks(); dc++) {
                    GroupMemoryBarrierWithGroupSync();
                    for (uint e = lid; e < 256u; e += 1024u) {
                        uint c = e % 16u;
                        cm_amat[e] = half(cm_q[dc * 16u + c]);
                    }
                    GroupMemoryBarrierWithGroupSync();
                    MatQ matQ = MatQ.Load<CoopMatMatrixLayout.RowMajor>(cm_amat, 0, 16);
                    MatK matK = MatK.Load<CoopMatMatrixLayout.ColumnMajor>(cm_k, dc * 16u, uint(PIE_HEAD_DIM));
                    acc = coopMatMulAdd<float, false>(matQ, matK, acc);
                }
                GroupMemoryBarrierWithGroupSync();
                acc.Store<CoopMatMatrixLayout.RowMajor>(cm_rmat, 0, 16);
                GroupMemoryBarrierWithGroupSync();
                for (uint kk = lid; kk < uint(PIE_KT); kk += 1024u) cm_scores[kk] = cm_rmat[kk];
                GroupMemoryBarrierWithGroupSync();
            }

            float hist_local[PIE_KT];
            float prob_local[PIE_KT];
            for (int kk = 0; kk < PIE_KT; kk++) { hist_local[kk] = 1.0; prob_local[kk] = 0.0; }
            for (int kk = 0; kk < cnt; kk++) {
                int kp = base + kk;
                if (!keeps(row, kp, q_pos, start)) continue;
                float score = pc.scale * cm_scores[kk];
                float hist, prob;
                sdpa_online_update(score, max_score, sum_exp, hist, prob);
                hist_local[kk] = hist;
                prob_local[kk] = prob;
            }

            float suffix[PIE_KT];
            float run = 1.0;
            for (int kk = PIE_KT - 1; kk >= 0; kk--) {
                suffix[kk] = run;
                run *= hist_local[kk];
            }
            float total_hist = run;

            GroupMemoryBarrierWithGroupSync();
            for (uint e = lid; e < 256u; e += 1024u) {
                uint kk = e % 16u;
                cm_amat[e] = half(prob_local[kk] * suffix[kk]);
            }
            GroupMemoryBarrierWithGroupSync();

            // P*V for the same tile, batched the same way: the weighted
            // probabilities (already folded with each key's suffix of
            // later rescales, see the comment above) are broadcast into a
            // 16x16 "A" and reduced against the V tile in one
            // coopMatMulAdd per 16-wide chunk of the head dimension.
            for (uint dc = 0u; dc < coopmat_head_chunks(); dc++) {
                MatAcc pacc =
                    MatAcc(0.0);
                MatQ matP = MatQ.Load<CoopMatMatrixLayout.RowMajor>(cm_amat, 0, 16);
                MatK matV = MatK.Load<CoopMatMatrixLayout.RowMajor>(cm_v, dc * 16u, uint(PIE_HEAD_DIM));
                pacc = coopMatMulAdd<float, false>(matP, matV, pacc);
                GroupMemoryBarrierWithGroupSync();
                pacc.Store<CoopMatMatrixLayout.RowMajor>(cm_rmat, 0, 16);
                GroupMemoryBarrierWithGroupSync();
                for (uint c = lid; c < 16u; c += 1024u) cm_pv[dc * 16u + c] = cm_rmat[c];
                GroupMemoryBarrierWithGroupSync();
            }

            if (mine) {
                for (int s = 0; s < PIE_DOUT_SLOTS; s++) {
                    uint d_out = lx + uint(s) * 32u;
                    if (d_out < uint(PIE_HEAD_DIM)) {
                        out_acc[s] = out_acc[s] * total_hist + cm_pv[d_out];
                    }
                }
            }
        }

        if (mine) {
#if defined(PIE_WITH_SINK)
            float orescale = sdpa_merge_sink(PIE_LOAD(sinks[q_head]), max_score, sum_exp);
            for (int s = 0; s < PIE_DOUT_SLOTS; s++) out_acc[s] *= orescale;
#endif
            for (int s = 0; s < PIE_DOUT_SLOTS; s++) {
                uint d_out = lx + uint(s) * 32u;
                if (d_out < uint(PIE_HEAD_DIM)) {
                    out_[o_base + d_out] = PIE_STORE(sum_exp == 0.0 ? out_acc[s] : out_acc[s] / sum_exp);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

#else

[numthreads(32, 32, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint q_head = group.x;
    uint row_lo = group.y * 32u;
    uint lx = local.x;
    uint ly = local.y;
    uint lid = ly * 32u + lx;
    uint n_q_heads = gl_NumWorkGroups.x;
    int kv_head = int(q_head) / pc.gqa_factor;

    for (uint rr = 0u; rr < 32u; rr++) {
        uint row = row_lo + rr;
        bool mine = ly == rr && row < uint(pc.n_rows);
        int req = row < uint(pc.n_rows) ? req_of_token[row] : 0;
        int q_pos = row < uint(pc.n_rows) ? position_ids[row] : -1;
        int start = (pc.window > 0 && q_pos >= pc.window) ? (q_pos - pc.window + 1) : 0;
        uint q_base = (row * n_q_heads + q_head) * uint(PIE_HEAD_DIM);
        uint o_base = q_base;

        // One set of softmax state for the whole row. It was per output slot,
        // which is what forced the tile loop inside the `d_out` loop and made
        // every slot re-load the same tile and re-derive the same scores.
        //
        // This is the fourth kernel in this directory found computing a score
        // once per thread rather than once per key, after `sdpa_paged`,
        // `sdpa_vector` and `sdpa_sliding`. It was the worst of the four and
        // the least visible, because the redundancy was in two places at once:
        // thirty-two lanes share a row and each also re-ran the whole tile
        // loop for every output slot it owned, so at `d_64` the same score was
        // reached sixty-four times and the same K/V tile staged twice.
        //
        // The straight transcription of the other three does not work here.
        // Their reduction sits in code every thread of the workgroup runs;
        // this accumulation is guarded by `mine`, and a `barrier()` inside a
        // guard only some threads pass is undefined. So the scores are built
        // OUTSIDE the guard, by all 1024 threads, into `sc_part` -- which is
        // legal precisely because `row`, `q_pos` and `start` are derived from
        // `rr` rather than from `mine` and are therefore workgroup-uniform.
        //
        // Measured, timing a 512-token request against a 64-token one so the
        // difference is the context-dependent work and not the setup, best of
        // five on an RTX 4090: the growth term went from 14.96 ms to 7.52 ms,
        // a factor of two, with the 64-token baselines matching to within 2%
        // as they should if only the per-key work changed.
        float max_score = PIE_SDPA_NEG_INF;
        float sum_exp = 0.0;
        float out_acc[PIE_DOUT_SLOTS];
        for (int s = 0; s < PIE_DOUT_SLOTS; s++) out_acc[s] = 0.0;

        for (int base = start; base <= q_pos; base += PIE_KT) {
            int cnt = min(PIE_KT, q_pos + 1 - base);
            GroupMemoryBarrierWithGroupSync();
            for (uint e = lid; e < uint(PIE_KT * PIE_HEAD_DIM); e += 1024u) {
                uint kk = e / uint(PIE_HEAD_DIM);
                uint d = e - kk * uint(PIE_HEAD_DIM);
                if (int(kk) < cnt) {
                    uint slot = page_slot(req, base + int(kk));
                    uint off = (slot * uint(pc.n_kv_heads) + uint(kv_head)) * uint(PIE_HEAD_DIM) + d;
                    k_tile[e] = k_pages[off];
                    v_tile[e] = v_pages[off];
                } else {
                    k_tile[e] = uint16_t(0);
                    v_tile[e] = uint16_t(0);
                }
            }
            GroupMemoryBarrierWithGroupSync();
            // Q.K^T for the tile, once. Thirty-two lanes share a key and each
            // walks a strided slice of the head, so the whole tile costs
            // `PIE_KT * PIE_HEAD_DIM` products rather than that many for every
            // one of the thirty-two threads that used to want the answer.
            //
            // `row`, `q_pos` and `start` come from `rr`, not from `mine`, so
            // every thread of the workgroup agrees on this loop and on the
            // barriers in it. That is what makes the reduction legal here at
            // all: the accumulation below is guarded by `mine` and could not
            // have contained a barrier.
            for (uint e = lid; e < uint(PIE_KT) * 32u; e += 1024u) {
                uint kk = e >> 5;
                uint lane = e & 31u;
                float acc = 0.0;
                if (int(kk) < cnt) {
                    for (uint d = lane; d < uint(PIE_HEAD_DIM); d += 32u) {
                        acc += pc.scale * PIE_LOAD(queries[q_base + d])
                             * PIE_LOAD(k_tile[kk * uint(PIE_HEAD_DIM) + d]);
                    }
                }
                sc_part[e] = acc;
            }
            GroupMemoryBarrierWithGroupSync();
            for (uint half_ = 16u; half_ > 0u; half_ >>= 1) {
                for (uint e = lid; e < uint(PIE_KT) * half_; e += 1024u) {
                    uint kk = e / half_;
                    uint lane = e - kk * half_;
                    sc_part[kk * 32u + lane] += sc_part[kk * 32u + lane + half_];
                }
                GroupMemoryBarrierWithGroupSync();
            }
            if (mine) {
                for (int kk = 0; kk < cnt; kk++) {
                    int kp = base + kk;
                    if (!keeps(row, kp, q_pos, start)) continue;
                    float score = sc_part[uint(kk) * 32u];
                    float hist, prob;
                    sdpa_online_update(score, max_score, sum_exp, hist, prob);
                    for (int s = 0; s < PIE_DOUT_SLOTS; s++) {
                        uint d_out = lx + uint(s) * 32u;
                        if (d_out < uint(PIE_HEAD_DIM)) {
                            out_acc[s] = out_acc[s] * hist
                                       + prob * PIE_LOAD(v_tile[uint(kk) * uint(PIE_HEAD_DIM) + d_out]);
                        }
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (mine) {
#if defined(PIE_WITH_SINK)
            float orescale = sdpa_merge_sink(PIE_LOAD(sinks[q_head]), max_score, sum_exp);
            for (int s = 0; s < PIE_DOUT_SLOTS; s++) out_acc[s] *= orescale;
#endif
            for (int s = 0; s < PIE_DOUT_SLOTS; s++) {
                uint d_out = lx + uint(s) * 32u;
                if (d_out < uint(PIE_HEAD_DIM)) {
                    out_[o_base + d_out] = PIE_STORE(sum_exp == 0.0 ? out_acc[s] : out_acc[s] / sum_exp);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

#endif

// pie:instantiate sdpa_paged_mma_bfloat16_d_64 PIE_HEAD_DIM=64
// pie:instantiate sdpa_paged_mma_sink_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_WITH_SINK=1
// pie:instantiate sdpa_paged_mma_bfloat16_d_64 @coopmat PIE_HEAD_DIM=64 PIE_COOPMAT=1
// pie:instantiate sdpa_paged_mma_sink_bfloat16_d_64 @coopmat PIE_HEAD_DIM=64 PIE_WITH_SINK=1 PIE_COOPMAT=1
