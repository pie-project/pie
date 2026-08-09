// Quest envelope kernels: correctness vs the CPU golden.
//
// The CPU references here MIRROR `pie_sampling_ir::eval::envelope_dot_reference`
// (the canonical golden, verified in `interface/sampling-ir` with the SAME
// hand-computed vector this file cross-checks). Two checks:
//   1. `envelope_recompute` == per-(page,kv_head,dim) min/max over live keys.
//   2. `envelope_dot` == Σ_group Σ_dim max(q·min, q·max), −inf beyond live.
//
// Requires a CUDA device.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "layout/envelope.hpp"

using pie_cuda_driver::kernels::layout::launch_envelope_dot_f32;
using pie_cuda_driver::kernels::layout::launch_envelope_merge_written_bf16;
using pie_cuda_driver::kernels::layout::launch_envelope_recompute_bf16;
using pie_cuda_driver::kernels::layout::launch_envelope_update_appended_bf16;

namespace {

int g_fail = 0;

void rt(cudaError_t e, const char* what, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FATAL %s:%d: %s -> %s\n", __FILE__, line, what,
                     cudaGetErrorString(e));
        std::exit(2);
    }
}
#define RT(e) rt((e), #e, __LINE__)

// bf16 round-trip (so the CPU reference sees the exact bytes the kernel reads).
std::uint16_t f2b(float f) {
    __nv_bfloat16 h = __float2bfloat16(f);
    std::uint16_t u;
    __builtin_memcpy(&u, &h, sizeof(u));
    return u;
}
float b2f(std::uint16_t u) {
    __nv_bfloat16 h;
    __builtin_memcpy(&h, &u, sizeof(h));
    return __bfloat162float(h);
}

// CPU golden — mirrors envelope_dot_reference (per-kv_head SUM over the GQA group).
std::vector<float> cpu_envelope_dot(
    const std::vector<float>& q, const std::vector<float>& emin,
    const std::vector<float>& emax, int nqh, int nkvh, int hd, int p_max,
    int live) {
    const int group = nqh / nkvh;
    std::vector<float> out(static_cast<std::size_t>(nkvh) * p_max,
                           -INFINITY);
    for (int kh = 0; kh < nkvh; ++kh)
        for (int p = 0; p < live && p < p_max; ++p) {
            const long eb = (static_cast<long>(p) * nkvh + kh) * hd;
            float acc = 0.f;
            for (int g = 0; g < group; ++g) {
                const long qb = static_cast<long>(kh * group + g) * hd;
                for (int d = 0; d < hd; ++d) {
                    const float qd = q[qb + d];
                    const float lo = qd * emin[eb + d];
                    const float hi = qd * emax[eb + d];
                    acc += (lo > hi) ? lo : hi;
                }
            }
            out[static_cast<long>(kh) * p_max + p] = acc;
        }
    return out;
}

void check_recompute(const char* name, int num_pages, int page_size, int nkvh,
                     int hd, const std::vector<int>& live) {
    const long tok = static_cast<long>(nkvh) * hd;
    const long n = static_cast<long>(num_pages) * page_size * tok;
    std::vector<std::uint16_t> k(n);
    for (long g = 0; g < n; ++g)
        k[g] = f2b(std::sin(0.01f * static_cast<float>(g)) * 5.0f);

    std::uint16_t* dk;
    std::int32_t* dlive;
    std::uint16_t *dmin, *dmax;
    const long envn = static_cast<long>(num_pages) * nkvh * hd;
    RT(cudaMalloc(&dk, n * 2));
    RT(cudaMalloc(&dlive, num_pages * sizeof(std::int32_t)));
    RT(cudaMalloc(&dmin, envn * sizeof(std::uint16_t)));
    RT(cudaMalloc(&dmax, envn * sizeof(std::uint16_t)));
    RT(cudaMemcpy(dk, k.data(), n * 2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(dlive, live.data(), num_pages * sizeof(std::int32_t),
                  cudaMemcpyHostToDevice));

    launch_envelope_recompute_bf16(dk, dlive, dmin, dmax, num_pages, page_size,
                                   nkvh, hd, nullptr);
    RT(cudaDeviceSynchronize());

    std::vector<std::uint16_t> gmin(envn), gmax(envn);
    RT(cudaMemcpy(gmin.data(), dmin, envn * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(gmax.data(), dmax, envn * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int p = 0; p < num_pages && ok; ++p)
        for (int kh = 0; kh < nkvh && ok; ++kh)
            for (int d = 0; d < hd && ok; ++d) {
                float mn = INFINITY, mx = -INFINITY;
                for (int t = 0; t < live[p]; ++t) {
                    const long idx =
                        ((static_cast<long>(p) * page_size + t) * nkvh + kh) *
                            hd + d;
                    const float v = b2f(k[idx]);
                    mn = std::fmin(mn, v);
                    mx = std::fmax(mx, v);
                }
                const long e = (static_cast<long>(p) * nkvh + kh) * hd + d;
                ok = (b2f(gmin[e]) == mn) && (b2f(gmax[e]) == mx);
            }
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", name);
    if (!ok) ++g_fail;
    cudaFree(dk); cudaFree(dlive); cudaFree(dmin); cudaFree(dmax);
}

// One request in a synthetic fire: how many tokens it appends, and the physical
// pages its KV list points at. `last_page_len` is POST-append (that is the
// invariant `write_kv_kernel` reads), so the pre-append length is
// `(pages.size()-1)*page_size + last_page_len - qo_len`.
struct FireRequest {
    int qo_len;
    int last_page_len;
    std::vector<std::uint32_t> pages;  // physical page ids, deliberately scattered
};

// The incremental maintenance path must land on exactly the bytes a full
// recompute would. A fire only appends, so it refreshes a subset of pages; the
// kernel derives that subset on-device from the CSR arrays, so this builds real
// CSR arrays (with scattered physical pages, so a mis-index is caught), asserts
// the touched pages match the full recompute bit for bit, and asserts every
// other page stays poisoned (so a bug that just rewrites everything fails).
void check_update_appended(const char* name, int num_pages, int page_size,
                           int nkvh, int hd,
                           const std::vector<FireRequest>& reqs) {
    const long tok = static_cast<long>(nkvh) * hd;
    const long n = static_cast<long>(num_pages) * page_size * tok;
    const long envn = static_cast<long>(num_pages) * nkvh * hd;
    std::vector<std::uint16_t> k(n);
    for (long g = 0; g < n; ++g)
        k[g] = f2b(std::cos(0.017f * static_cast<float>(g)) * 3.0f);

    // CSR mirrors, exactly as the KV write path holds them.
    const int R = static_cast<int>(reqs.size());
    std::vector<std::uint32_t> qo_indptr(R + 1, 0), kv_page_indptr(R + 1, 0);
    std::vector<std::uint32_t> kv_page_indices, kv_last_page_lens(R, 0);
    for (int r = 0; r < R; ++r) {
        qo_indptr[r + 1] = qo_indptr[r] + static_cast<std::uint32_t>(reqs[r].qo_len);
        kv_page_indptr[r + 1] =
            kv_page_indptr[r] + static_cast<std::uint32_t>(reqs[r].pages.size());
        kv_page_indices.insert(kv_page_indices.end(), reqs[r].pages.begin(),
                               reqs[r].pages.end());
        kv_last_page_lens[r] = static_cast<std::uint32_t>(reqs[r].last_page_len);
    }

    // Pool-wide live lengths for the full recompute, and the host's independent
    // derivation of the touched set (same arithmetic, computed separately so a
    // kernel that agrees with itself but not the spec still fails).
    std::vector<int> live(num_pages, 0);
    std::vector<bool> touched(num_pages, false);
    for (const FireRequest& req : reqs) {
        const int np = static_cast<int>(req.pages.size());
        for (int i = 0; i < np; ++i)
            live[req.pages[i]] = (i == np - 1) ? req.last_page_len : page_size;
        const int total_after = (np - 1) * page_size + req.last_page_len;
        const int first = (total_after - req.qo_len) / page_size;
        const int last = (total_after - 1) / page_size;
        for (int i = first; i <= last && i < np; ++i) touched[req.pages[i]] = true;
    }

    std::uint16_t* dk;
    std::int32_t* dlive;
    std::uint16_t *dmin_ref, *dmax_ref, *dmin_inc, *dmax_inc;
    std::uint32_t *d_qo, *d_idx, *d_indptr, *d_lastlen;
    RT(cudaMalloc(&dk, n * 2));
    RT(cudaMalloc(&dlive, num_pages * sizeof(std::int32_t)));
    RT(cudaMalloc(&dmin_ref, envn * sizeof(std::uint16_t)));
    RT(cudaMalloc(&dmax_ref, envn * sizeof(std::uint16_t)));
    RT(cudaMalloc(&dmin_inc, envn * sizeof(std::uint16_t)));
    RT(cudaMalloc(&dmax_inc, envn * sizeof(std::uint16_t)));
    RT(cudaMalloc(&d_qo, qo_indptr.size() * sizeof(std::uint32_t)));
    RT(cudaMalloc(&d_idx, kv_page_indices.size() * sizeof(std::uint32_t)));
    RT(cudaMalloc(&d_indptr, kv_page_indptr.size() * sizeof(std::uint32_t)));
    RT(cudaMalloc(&d_lastlen, kv_last_page_lens.size() * sizeof(std::uint32_t)));
    RT(cudaMemcpy(dk, k.data(), n * 2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(dlive, live.data(), num_pages * sizeof(std::int32_t),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_qo, qo_indptr.data(),
                  qo_indptr.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_idx, kv_page_indices.data(),
                  kv_page_indices.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_indptr, kv_page_indptr.data(),
                  kv_page_indptr.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_lastlen, kv_last_page_lens.data(),
                  kv_last_page_lens.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice));

    // Poison the incremental buffers so any page the fire does NOT touch stays
    // poisoned -- only the touched ones may be compared against the golden.
    RT(cudaMemset(dmin_inc, 0x7f, envn * sizeof(std::uint16_t)));
    RT(cudaMemset(dmax_inc, 0x7f, envn * sizeof(std::uint16_t)));

    // The host's worst-case bound, exactly as the KV write path will compute it.
    const int total_tokens = static_cast<int>(qo_indptr[R]);
    const int max_touched = (total_tokens + page_size - 1) / page_size + R;

    launch_envelope_recompute_bf16(dk, dlive, dmin_ref, dmax_ref, num_pages,
                                   page_size, nkvh, hd, nullptr);
    launch_envelope_update_appended_bf16(dk, d_qo, d_idx, d_indptr, d_lastlen,
                                         dmin_inc, dmax_inc, R, max_touched,
                                         page_size, nkvh, hd, nullptr);
    RT(cudaDeviceSynchronize());

    std::vector<std::uint16_t> rmin(envn), rmax(envn), imin(envn), imax(envn);
    RT(cudaMemcpy(rmin.data(), dmin_ref, envn * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(rmax.data(), dmax_ref, envn * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(imin.data(), dmin_inc, envn * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(imax.data(), dmax_inc, envn * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToHost));

    // `cudaMemset` writes 0x7f per BYTE, so a bf16 lane reads back 0x7f7f.
    const std::uint16_t poison = 0x7f7fu;

    // The test is only meaningful if some page is deliberately left untouched.
    int untouched_used = 0;
    for (int p = 0; p < num_pages; ++p)
        if (live[p] > 0 && !touched[p]) ++untouched_used;
    bool ok = untouched_used > 0;
    if (!ok) std::printf("       (degenerate case: every live page is touched)\n");

    for (int p = 0; p < num_pages && ok; ++p) {
        for (int kh = 0; kh < nkvh && ok; ++kh)
            for (int d = 0; d < hd && ok; ++d) {
                const long e = (static_cast<long>(p) * nkvh + kh) * hd + d;
                ok = touched[p] ? (imin[e] == rmin[e] && imax[e] == rmax[e])
                                : (imin[e] == poison && imax[e] == poison);
            }
        if (!ok)
            std::printf("       (page %d: expected %s)\n", p,
                        touched[p] ? "refreshed" : "untouched");
    }
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", name);
    if (!ok) ++g_fail;
    cudaFree(dk); cudaFree(dlive);
    cudaFree(dmin_ref); cudaFree(dmax_ref);
    cudaFree(dmin_inc); cudaFree(dmax_inc);
    cudaFree(d_qo); cudaFree(d_idx); cudaFree(d_indptr); cudaFree(d_lastlen);
}

// The explicit-descriptor maintenance path: `launch_envelope_merge_written_bf16`.
//
// The claim it has to earn is that merging fire by fire ends up where a full
// recompute of the finished pages would. So this simulates a whole sequence --
// a prefill fire followed by `decode_fires` one-token fires -- feeding each one
// through the merge exactly as `kernels::attn::write_kv_explicit_bf16` does, and then
// compares against `envelope_recompute` over the final live lengths.
//
// It also plants garbage in the envelopes of every page the sequence will use,
// standing in for a page recycled out of the pool. Merging alone would keep
// that garbage forever (min/max only widen); the reset pass is what removes it,
// and the comparison below is exact, so a missing reset fails here.
void check_merge_written(const char* name, int num_pages, int page_size,
                         int nkvh, int hd,
                         const std::vector<int>& pages,
                         int prefill_len, int decode_fires,
                         bool poison_invalid_row) {
    const long tok = static_cast<long>(nkvh) * hd;
    const long n = static_cast<long>(num_pages) * page_size * tok;
    const long envn = static_cast<long>(num_pages) * nkvh * hd;
    const int total = prefill_len + decode_fires;

    // The pool, and the sequence's keys laid into it at the cells the
    // descriptors will name. Everything else stays zero and is never live.
    std::vector<std::uint16_t> k(n, f2b(0.f));
    std::vector<std::uint16_t> k_curr(static_cast<long>(total) * tok);
    std::vector<std::uint32_t> w_page(total), w_off(total);
    for (int t = 0; t < total; ++t) {
        w_page[t] = static_cast<std::uint32_t>(pages[t / page_size]);
        w_off[t] = static_cast<std::uint32_t>(t % page_size);
        const long cell = (static_cast<long>(w_page[t]) * page_size +
                           static_cast<long>(w_off[t])) * tok;
        for (long e = 0; e < tok; ++e) {
            const std::uint16_t v = f2b(
                std::sin(0.013f * static_cast<float>(t * tok + e)) * 2.5f);
            k[cell + e] = v;
            k_curr[static_cast<long>(t) * tok + e] = v;
        }
    }

    std::vector<int> live(num_pages, 0);
    for (int t = 0; t < total; ++t) live[pages[t / page_size]] = (t % page_size) + 1;

    std::uint16_t *dk, *dcurr;
    std::int32_t* dlive;
    std::uint16_t *dmin_ref, *dmax_ref, *dmin_inc, *dmax_inc;
    std::uint32_t *d_page, *d_off;
    RT(cudaMalloc(&dk, n * 2));
    RT(cudaMalloc(&dcurr, static_cast<long>(total) * tok * 2));
    RT(cudaMalloc(&dlive, num_pages * sizeof(std::int32_t)));
    RT(cudaMalloc(&dmin_ref, envn * sizeof(std::uint16_t)));
    RT(cudaMalloc(&dmax_ref, envn * sizeof(std::uint16_t)));
    RT(cudaMalloc(&dmin_inc, envn * sizeof(std::uint16_t)));
    RT(cudaMalloc(&dmax_inc, envn * sizeof(std::uint16_t)));
    RT(cudaMalloc(&d_page, total * sizeof(std::uint32_t)));
    RT(cudaMalloc(&d_off, total * sizeof(std::uint32_t)));
    RT(cudaMemcpy(dk, k.data(), n * 2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(dcurr, k_curr.data(), static_cast<long>(total) * tok * 2,
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(dlive, live.data(), num_pages * sizeof(std::int32_t),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_page, w_page.data(), total * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_off, w_off.data(), total * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice));

    launch_envelope_recompute_bf16(dk, dlive, dmin_ref, dmax_ref, num_pages,
                                   page_size, nkvh, hd, nullptr);

    // The recycled-page stand-in: a previous tenant's envelope, wide enough
    // that a merge could never narrow it back.
    std::vector<std::uint16_t> dirty_min(envn, f2b(-1e30f)),
        dirty_max(envn, f2b(1e30f));
    RT(cudaMemcpy(dmin_inc, dirty_min.data(), envn * sizeof(std::uint16_t),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(dmax_inc, dirty_max.data(), envn * sizeof(std::uint16_t),
                  cudaMemcpyHostToDevice));

    launch_envelope_merge_written_bf16(dcurr, d_page, d_off, nullptr,
                                       dmin_inc, dmax_inc, prefill_len,
                                       nkvh, hd, nullptr);
    for (int f = 0; f < decode_fires; ++f) {
        const int t = prefill_len + f;
        launch_envelope_merge_written_bf16(
            dcurr + static_cast<long>(t) * tok, d_page + t, d_off + t, nullptr,
            dmin_inc, dmax_inc, 1, nkvh, hd, nullptr);
    }

    // A masked-off row: the descriptor still names a cell (page 0 of the
    // sequence, offset 0 -- the most destructive target available, since an
    // unhonoured `row_valid` would BOTH reset that page and pollute it with a
    // key the pool never received). The comparison below is exact, so honouring
    // `row_valid` in only one of the two passes fails too.
    if (poison_invalid_row) {
        std::vector<std::uint16_t> junk(tok, f2b(9.0f));
        const std::uint32_t jp = static_cast<std::uint32_t>(pages[0]), jo = 0u;
        const std::uint8_t invalid = 0;
        std::uint16_t* d_junk;
        std::uint32_t *d_jp, *d_jo;
        std::uint8_t* d_valid;
        RT(cudaMalloc(&d_junk, tok * 2));
        RT(cudaMalloc(&d_jp, sizeof(std::uint32_t)));
        RT(cudaMalloc(&d_jo, sizeof(std::uint32_t)));
        RT(cudaMalloc(&d_valid, 1));
        RT(cudaMemcpy(d_junk, junk.data(), tok * 2, cudaMemcpyHostToDevice));
        RT(cudaMemcpy(d_jp, &jp, sizeof(jp), cudaMemcpyHostToDevice));
        RT(cudaMemcpy(d_jo, &jo, sizeof(jo), cudaMemcpyHostToDevice));
        RT(cudaMemcpy(d_valid, &invalid, 1, cudaMemcpyHostToDevice));
        launch_envelope_merge_written_bf16(d_junk, d_jp, d_jo, d_valid,
                                           dmin_inc, dmax_inc, 1, nkvh, hd,
                                           nullptr);
        RT(cudaDeviceSynchronize());
        cudaFree(d_junk); cudaFree(d_jp); cudaFree(d_jo); cudaFree(d_valid);
    }
    RT(cudaDeviceSynchronize());

    std::vector<std::uint16_t> rmin(envn), rmax(envn), imin(envn), imax(envn);
    RT(cudaMemcpy(rmin.data(), dmin_ref, envn * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(rmax.data(), dmax_ref, envn * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(imin.data(), dmin_inc, envn * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(imax.data(), dmax_inc, envn * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < total && ok; ++i) {
        const int p = pages[i / page_size];
        for (int kh = 0; kh < nkvh && ok; ++kh)
            for (int d = 0; d < hd && ok; ++d) {
                const long e = (static_cast<long>(p) * nkvh + kh) * hd + d;
                ok = imin[e] == rmin[e] && imax[e] == rmax[e];
                if (!ok)
                    std::printf(
                        "       (page %d kh %d d %d: merged [%g,%g] vs "
                        "recompute [%g,%g])\n",
                        p, kh, d, b2f(imin[e]), b2f(imax[e]),
                        b2f(rmin[e]), b2f(rmax[e]));
            }
    }
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", name);
    if (!ok) ++g_fail;
    cudaFree(dk); cudaFree(dcurr); cudaFree(dlive);
    cudaFree(dmin_ref); cudaFree(dmax_ref);
    cudaFree(dmin_inc); cudaFree(dmax_inc);
    cudaFree(d_page); cudaFree(d_off);
}

void check_dot(const char* name, int nqh, int nkvh, int hd, int p_max,
               int live) {
    const long envn = static_cast<long>(p_max) * nkvh * hd;
    std::vector<float> q(static_cast<std::size_t>(nqh) * hd);
    std::vector<float> emin(envn), emax(envn);
    for (std::size_t i = 0; i < q.size(); ++i)
        q[i] = std::cos(0.017f * static_cast<float>(i)) * 2.0f;
    // The envelopes are stored bf16, so ROUND FIRST and let the CPU reference
    // see the same numbers the kernel will read. Rounding only one side would
    // turn a storage-format question into an apparent kernel error, and the
    // tolerance below exists for reduction ORDER, not for storage width.
    std::vector<std::uint16_t> bmin(envn), bmax(envn);
    for (long i = 0; i < envn; ++i) {
        const float a = std::sin(0.013f * static_cast<float>(i));
        bmin[i] = f2b(a - 1.5f);
        bmax[i] = f2b(a + 2.0f);
        emin[i] = b2f(bmin[i]);
        emax[i] = b2f(bmax[i]);
    }

    float *dq, *dscore;
    std::uint16_t *dmin, *dmax;
    const long sn = static_cast<long>(nkvh) * p_max;
    RT(cudaMalloc(&dq, q.size() * sizeof(float)));
    RT(cudaMalloc(&dmin, envn * sizeof(std::uint16_t)));
    RT(cudaMalloc(&dmax, envn * sizeof(std::uint16_t)));
    RT(cudaMalloc(&dscore, sn * sizeof(float)));
    RT(cudaMemcpy(dq, q.data(), q.size() * sizeof(float),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(dmin, bmin.data(), envn * sizeof(std::uint16_t),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(dmax, bmax.data(), envn * sizeof(std::uint16_t),
                  cudaMemcpyHostToDevice));

    launch_envelope_dot_f32(dq, dmin, dmax, dscore, nqh, nkvh, hd, p_max, live,
                            nullptr);
    RT(cudaDeviceSynchronize());

    std::vector<float> gs(sn);
    RT(cudaMemcpy(gs.data(), dscore, sn * sizeof(float),
                  cudaMemcpyDeviceToHost));
    const std::vector<float> ref =
        cpu_envelope_dot(q, emin, emax, nqh, nkvh, hd, p_max, live);

    bool ok = true;
    for (long i = 0; i < sn && ok; ++i) {
        if (std::isinf(ref[i])) {
            ok = std::isinf(gs[i]) && (gs[i] < 0);  // both −inf beyond live
        } else {
            // The reduction order differs (grid-stride tree vs sequential), so
            // allow a tiny f32 rounding tolerance; not a bit-exact contract.
            ok = std::fabs(gs[i] - ref[i]) <=
                 1e-3f * (1.0f + std::fabs(ref[i]));
        }
    }
    std::printf("[%s] %s (nqh=%d nkvh=%d hd=%d P_MAX=%d live=%d)\n",
                ok ? " ok " : "FAIL", name, nqh, nkvh, hd, p_max, live);
    if (!ok) ++g_fail;
    cudaFree(dq); cudaFree(dmin); cudaFree(dmax); cudaFree(dscore);
}

// Cross-check the exact hand-computed vector from the Rust golden test
// (envelope_dot_reference_quest_math): score[0,0]=11, [0,1]=3, [0,2]=−inf.
void check_dot_golden_vector() {
    const std::vector<float> q{1.0f, -1.0f, 2.0f, 0.5f};
    // Every value that reaches the result is a small dyadic rational and so is
    // bf16-exact; the 9.9s belong to the page past `live_pages`, which is
    // -inf'd before any arithmetic. The golden is therefore unchanged by the
    // envelopes being stored bf16.
    const std::vector<std::uint16_t> emin{f2b(0.0f),  f2b(0.0f), f2b(-2.0f),
                                          f2b(1.0f),  f2b(9.9f), f2b(9.9f)};
    const std::vector<std::uint16_t> emax{f2b(3.0f),  f2b(4.0f), f2b(1.0f),
                                          f2b(2.0f),  f2b(9.9f), f2b(9.9f)};
    float *dq, *ds;
    std::uint16_t *dmin, *dmax;
    RT(cudaMalloc(&dq, 4 * sizeof(float)));
    RT(cudaMalloc(&dmin, 6 * sizeof(std::uint16_t)));
    RT(cudaMalloc(&dmax, 6 * sizeof(std::uint16_t)));
    RT(cudaMalloc(&ds, 3 * sizeof(float)));
    RT(cudaMemcpy(dq, q.data(), 4 * sizeof(float), cudaMemcpyHostToDevice));
    RT(cudaMemcpy(dmin, emin.data(), 6 * sizeof(std::uint16_t),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(dmax, emax.data(), 6 * sizeof(std::uint16_t),
                  cudaMemcpyHostToDevice));
    launch_envelope_dot_f32(dq, dmin, dmax, ds, 2, 1, 2, 3, 2, nullptr);
    RT(cudaDeviceSynchronize());
    float s[3];
    RT(cudaMemcpy(s, ds, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    const bool ok = (s[0] == 11.0f) && (s[1] == 3.0f) &&
                    (std::isinf(s[2]) && s[2] < 0);
    std::printf("[%s] golden vector cross-check (11, 3, -inf) got (%.1f, %.1f, %.1f)\n",
                ok ? " ok " : "FAIL", s[0], s[1], s[2]);
    if (!ok) ++g_fail;
    cudaFree(dq); cudaFree(dmin); cudaFree(dmax); cudaFree(ds);
}

// What the explicit-path maintenance costs. Envelopes are a process-wide
// switch, so this runs on every KV append once Quest is enabled, per layer per
// fire -- and Quest's decode fires carry only R tokens, where the two kernels
// are pure launch overhead rather than work. That is the regime that decides
// whether the feature is affordable, so it is the one reported first.
void bench_merge_written(int num_tokens, int nkvh, int hd, int layers) {
    const long tok = static_cast<long>(nkvh) * hd;
    const int num_pages = num_tokens + 8;
    const long envn = static_cast<long>(num_pages) * tok;

    std::vector<std::uint16_t> k(static_cast<long>(num_tokens) * tok, f2b(0.5f));
    std::vector<std::uint32_t> wp(num_tokens), wo(num_tokens);
    for (int t = 0; t < num_tokens; ++t) {
        wp[t] = static_cast<std::uint32_t>(t % num_pages);
        wo[t] = static_cast<std::uint32_t>(t % 16);
    }

    std::uint16_t* dk;
    std::uint32_t *d_page, *d_off;
    std::uint16_t *dmin, *dmax;
    RT(cudaMalloc(&dk, static_cast<long>(num_tokens) * tok * 2));
    RT(cudaMalloc(&d_page, num_tokens * sizeof(std::uint32_t)));
    RT(cudaMalloc(&d_off, num_tokens * sizeof(std::uint32_t)));
    RT(cudaMalloc(&dmin, envn * sizeof(std::uint16_t)));
    RT(cudaMalloc(&dmax, envn * sizeof(std::uint16_t)));
    RT(cudaMemcpy(dk, k.data(), static_cast<long>(num_tokens) * tok * 2,
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_page, wp.data(), num_tokens * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_off, wo.data(), num_tokens * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice));

    cudaEvent_t beg, end;
    RT(cudaEventCreate(&beg));
    RT(cudaEventCreate(&end));

    // One iteration = one whole forward pass worth of maintenance, because the
    // per-launch overhead is the dominant term at decode widths and only shows
    // up honestly when all `layers` launches are in flight together.
    //
    // This host is shared and heavily loaded, so a mean is worthless (repeated
    // runs disagree by >2x). Take the MIN over repetitions: the fastest pass is
    // the one that got the least interference, and the quantity of interest is
    // the cost of the work, not the cost of the contention.
    const int warmup = 20, iters = 100, reps = 7;
    for (int i = 0; i < warmup; ++i)
        for (int l = 0; l < layers; ++l)
            launch_envelope_merge_written_bf16(dk, d_page, d_off, nullptr, dmin,
                                               dmax, num_tokens, nkvh, hd, nullptr);
    RT(cudaDeviceSynchronize());

    double best_us = 1e30;
    for (int r = 0; r < reps; ++r) {
        RT(cudaEventRecord(beg));
        for (int i = 0; i < iters; ++i)
            for (int l = 0; l < layers; ++l)
                launch_envelope_merge_written_bf16(dk, d_page, d_off, nullptr,
                                                   dmin, dmax, num_tokens, nkvh,
                                                   hd, nullptr);
        RT(cudaEventRecord(end));
        RT(cudaEventSynchronize(end));
        float ms = 0.f;
        RT(cudaEventElapsedTime(&ms, beg, end));
        const double us = 1000.0 * static_cast<double>(ms) / iters;
        if (us < best_us) best_us = us;
    }

    const double per_pass_us = best_us;
    std::printf("  N=%-5d  %7.1f us / %d-layer pass   (%5.2f us / layer)\n",
                num_tokens, per_pass_us, layers, per_pass_us / layers);

    RT(cudaEventDestroy(beg));
    RT(cudaEventDestroy(end));
    cudaFree(dk); cudaFree(d_page); cudaFree(d_off);
    cudaFree(dmin); cudaFree(dmax);
}

}  // namespace

int main() {
    RT(cudaSetDevice(0));

    check_recompute("recompute: full + partial pages", 6, 16, 8, 128,
                    {16, 1, 9, 16, 5, 0});
    check_recompute("recompute: head_dim 64", 4, 16, 4, 64, {16, 3, 16, 8});

    // Decode: every request appends 1 token, so each touches exactly its last
    // page. Physical pages are scattered and interleaved between requests.
    check_update_appended("update/decode: 3 requests, 1 token each",
                          12, 16, 4, 64,
                          {{1, 7, {9, 2, 5}}, {1, 16, {0, 11}}, {1, 1, {4, 8, 3, 6}}});
    // A prefill that straddles a page boundary plus a partial last page, next to
    // a decode step -- the two touched-span shapes in the same fire.
    check_update_appended("update/mixed: straddling prefill + decode",
                          10, 16, 8, 128,
                          {{20, 4, {6, 1, 9}}, {1, 13, {3, 0}}});
    // A request whose new tokens exactly fill the last page (pre_len on a page
    // boundary), so `first == last` must NOT be off by one.
    check_update_appended("update: append lands on a page boundary",
                          8, 16, 4, 64, {{16, 16, {5, 2, 7}}, {1, 9, {1, 4}}});

    // The explicit-descriptor write path (Quest's decode fires go through it).
    // Fire-by-fire merge must land where a full recompute of the finished pages
    // would, starting from envelopes left dirty by a previous tenant.
    check_merge_written("merge/explicit: prefill 20 + 20 decodes",
                        10, 16, 4, 64, {6, 1, 9}, 20, 20, false);
    check_merge_written("merge/explicit: page-aligned prefill + decodes",
                        8, 16, 8, 128, {3, 0, 5}, 32, 9, false);
    check_merge_written("merge/explicit: masked row must not touch its page",
                        10, 16, 4, 64, {2, 7, 4}, 17, 20, true);
    // Wider than `kEnvelopeFuseMaxTokens`, so the prefill fire takes the
    // two-launch reset/merge path while its decode fires take the fused one.
    // Both must land on the same envelopes.
    check_merge_written("merge/explicit: wide fire (two-launch path)",
                        20, 16, 4, 64,
                        {13, 2, 17, 5, 0, 11, 8, 1, 15, 6, 3, 19, 9, 4},
                        200, 10, false);

    check_dot_golden_vector();
    check_dot("dot: GQA 16q/8kv, P_MAX 32", 16, 8, 128, 32, 20);
    check_dot("dot: MHA 8/8", 8, 8, 128, 16, 16);
    check_dot("dot: live < P_MAX (−inf tail)", 16, 8, 128, 64, 7);

    std::printf(g_fail ? "\nENVELOPE FAILED (%d)\n" : "\nALL PASS (0 failures)\n",
                g_fail);

    if (std::getenv("PIE_ENVELOPE_MERGE_BENCH") != nullptr) {
        std::printf("\nenvelope maintenance on the explicit write path"
                    " (nkvh=8 hd=128, 28 layers):\n");
        for (int n : {1, 2, 4, 8, 16, 32, 64, 128, 272, 2048})
            bench_merge_written(n, 8, 128, 28);
    }

    return g_fail ? 1 : 0;
}
