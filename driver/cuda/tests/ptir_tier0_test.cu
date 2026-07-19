// PTIR tier-0 kernel parity test.
//
// Runs every tier-0 op kernel (tier0_kernels.cuh) on the live GPU and diffs the
// result against the host reference evaluator (host_eval.hpp). This is the
// self-check gate that must hold before diffing against the canonical golden
// vectors. Standalone: compiled directly by nvcc, no driver-lib dependency.
//
//   nvcc -std=c++17 -I../src tests/ptir_tier0_test.cu -o ptir_tier0_test && ./ptir_tier0_test

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include "support/host_eval.hpp"
#include "pipeline/tier0/tier0_kernels.cuh"
#include "pipeline/tier0/tier0_launch.hpp"

using namespace pie_cuda_driver::pipeline;

namespace {

int g_pass = 0, g_fail = 0;

#define CUDA_OK(call)                                                            \
    do {                                                                         \
        cudaError_t e = (call);                                                  \
        if (e != cudaSuccess) {                                                  \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e),       \
                        __FILE__, __LINE__);                                     \
            std::exit(2);                                                        \
        }                                                                        \
    } while (0)

template <class T>
T* dev_from(const std::vector<T>& h) {
    T* d = nullptr;
    CUDA_OK(cudaMalloc(&d, h.size() * sizeof(T)));
    CUDA_OK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T>
T* dev_alloc(std::size_t n) {
    T* d = nullptr;
    CUDA_OK(cudaMalloc(&d, n * sizeof(T)));
    return d;
}
template <class T>
std::vector<T> to_host(const T* d, std::size_t n) {
    std::vector<T> h(n);
    CUDA_OK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

bool feq(float a, float b) {
    if (std::isinf(a) && std::isinf(b)) return (a > 0) == (b > 0);
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    float d = std::fabs(a - b);
    return d <= 1e-4f + 1e-4f * std::fabs(b);
}

template <class T>
void check(const std::string& name, const std::vector<T>& got, const std::vector<T>& want) {
    bool ok = got.size() == want.size();
    for (std::size_t i = 0; ok && i < got.size(); ++i) {
        if constexpr (std::is_same_v<T, float>) ok = feq(got[i], want[i]);
        else ok = (got[i] == want[i]);
    }
    if (ok) {
        ++g_pass;
        std::printf("  PASS  %s (n=%zu)\n", name.c_str(), got.size());
    } else {
        ++g_fail;
        std::printf("  FAIL  %s (n=%zu)\n", name.c_str(), got.size());
        std::size_t shown = 0;
        for (std::size_t i = 0; i < got.size() && shown < 6; ++i) {
            bool eq;
            if constexpr (std::is_same_v<T, float>) eq = feq(got[i], want[i]);
            else eq = (got[i] == want[i]);
            if (!eq) {
                if constexpr (std::is_same_v<T, float>)
                    std::printf("        [%zu] got=%g want=%g\n", i, (double)got[i], (double)want[i]);
                else
                    std::printf("        [%zu] got=%lld want=%lld\n", i, (long long)got[i], (long long)want[i]);
                ++shown;
            }
        }
    }
}

constexpr int GS(std::uint64_t n, int b = kTier0Block) { return (int)((n + b - 1) / b); }

void test_map() {
    std::vector<float> a{1, 2, 3, 4, 5, 6, -7, 8};
    std::vector<float> b{2, 2, 2, 3, 5, 4,  2, 3};
    float *da = dev_from(a), *db = dev_from(b), *dout = dev_alloc<float>(a.size());
    for (auto [k, nm] : {std::pair{BinKind::Add, "add"}, {BinKind::Sub, "sub"},
                         {BinKind::Mul, "mul"}, {BinKind::Div, "div"}, {BinKind::Rem, "rem"}}) {
        k_binary<float><<<GS(a.size()), kTier0Block>>>(da, db, dout, a.size(), k);
        CUDA_OK(cudaDeviceSynchronize());
        check(nm, to_host(dout, a.size()), host_eval::binary(k, a, b));
    }
    for (auto [k, nm] : {std::pair{UnKind::Neg, "neg"}, {UnKind::Exp, "exp"}, {UnKind::Log, "log"}}) {
        std::vector<float> pos{0.5f, 1, 2, 3, 4, 5, 6, 7};
        float* dp = dev_from(pos);
        k_unary<float><<<GS(pos.size()), kTier0Block>>>((k == UnKind::Neg ? da : dp), dout, pos.size(), k);
        CUDA_OK(cudaDeviceSynchronize());
        check(nm, to_host(dout, pos.size()), host_eval::unary(k, (k == UnKind::Neg ? a : pos)));
        CUDA_OK(cudaFree(dp));
    }
    CUDA_OK(cudaFree(da)); CUDA_OK(cudaFree(db)); CUDA_OK(cudaFree(dout));
}

void test_compare_logic_select_cast() {
    std::vector<float> a{1, 2, 3, 4, 5, 3, 7, 8};
    std::vector<float> b{1, 3, 2, 4, 1, 3, 9, 0};
    float *da = dev_from(a), *db = dev_from(b);
    std::uint8_t* dbool = dev_alloc<std::uint8_t>(a.size());
    for (auto [k, nm] : {std::pair{CmpKind::Eq, "eq"}, {CmpKind::Ne, "ne"}, {CmpKind::Lt, "lt"},
                         {CmpKind::Le, "le"}, {CmpKind::Gt, "gt"}, {CmpKind::Ge, "ge"}}) {
        k_compare<float><<<GS(a.size()), kTier0Block>>>(da, db, dbool, a.size(), k);
        CUDA_OK(cudaDeviceSynchronize());
        check(nm, to_host(dbool, a.size()), host_eval::compare(k, a, b));
    }
    std::vector<std::uint8_t> p{1, 0, 1, 1, 0, 0, 1, 0}, q{1, 1, 0, 1, 0, 1, 1, 0};
    std::uint8_t *dp = dev_from(p), *dq = dev_from(q), *dr = dev_alloc<std::uint8_t>(p.size());
    for (auto [k, nm] : {std::pair{LogicKind::And, "and"}, {LogicKind::Or, "or"}}) {
        k_logic<<<GS(p.size()), kTier0Block>>>(dp, dq, dr, p.size(), k);
        CUDA_OK(cudaDeviceSynchronize());
        check(nm, to_host(dr, p.size()), host_eval::logic(k, p, q));
    }
    k_not<<<GS(p.size()), kTier0Block>>>(dp, dr, p.size());
    CUDA_OK(cudaDeviceSynchronize());
    check("not", to_host(dr, p.size()), host_eval::logic_not(p));

    // select
    float* dsel = dev_alloc<float>(a.size());
    k_select<float><<<GS(a.size()), kTier0Block>>>(dp, da, db, dsel, a.size());
    CUDA_OK(cudaDeviceSynchronize());
    check("select", to_host(dsel, a.size()), host_eval::select(p, a, b));

    // cast f32 -> i32 and u32 -> f32
    std::int32_t* dcast = dev_alloc<std::int32_t>(a.size());
    k_cast<float, std::int32_t><<<GS(a.size()), kTier0Block>>>(da, dcast, a.size());
    CUDA_OK(cudaDeviceSynchronize());
    check("cast_f32_i32", to_host(dcast, a.size()), host_eval::cast<float, std::int32_t>(a));

    CUDA_OK(cudaFree(da)); CUDA_OK(cudaFree(db)); CUDA_OK(cudaFree(dbool));
    CUDA_OK(cudaFree(dp)); CUDA_OK(cudaFree(dq)); CUDA_OK(cudaFree(dr));
    CUDA_OK(cudaFree(dsel)); CUDA_OK(cudaFree(dcast));
}

void test_index() {
    std::uint32_t n = 300;   // > one block, exercises grid-stride
    std::uint32_t* diota = dev_alloc<std::uint32_t>(n);
    k_iota<<<GS(n), kTier0Block>>>(diota, n);
    CUDA_OK(cudaDeviceSynchronize());
    check("iota", to_host(diota, n), host_eval::iota(n));

    std::vector<float> src{10, 11, 12, 13, 14, 15, 16, 17};
    std::vector<std::int32_t> idx{3, -1, 4, 1};
    float* dsrc = dev_from(src);
    std::int32_t* didx = dev_from(idx);
    float* dg = dev_alloc<float>(idx.size() * 2);
    k_gather_axis0<float, std::int32_t><<<
        GS(idx.size() * 2), kTier0Block>>>(
        dsrc, didx, dg, idx.size(), 4, 2);
    CUDA_OK(cudaDeviceSynchronize());
    check(
        "gather.axis0_i32_bounds",
        to_host(dg, idx.size() * 2),
        std::vector<float>{16, 17, 0, 0, 0, 0, 12, 13});

    // Multidimensional axis-0 scatter: signed invalid indexes skip and the
    // second valid write to row 2 wins.
    std::vector<float> base{0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<std::int32_t> sidx{2, -1, 2, 8};
    std::vector<float> vals{9, 10, 4, 5, 7, 8, 1, 2};
    float* dbase = dev_from(base);
    std::int32_t* dsidx = dev_from(sidx);
    float* dvals = dev_from(vals);
    k_scatter_axis0_serial<float, std::int32_t, false><<<1, 1>>>(
        dbase, dsidx, dvals, sidx.size(), 4, 2, false);
    CUDA_OK(cudaDeviceSynchronize());
    check(
        "scatter_set.axis0_i32_bounds",
        to_host(dbase, base.size()),
        std::vector<float>{0, 0, 0, 0, 7, 8, 0, 0});

    CUDA_OK(cudaFree(diota)); CUDA_OK(cudaFree(dsrc)); CUDA_OK(cudaFree(didx)); CUDA_OK(cudaFree(dg));
    CUDA_OK(cudaFree(dbase)); CUDA_OK(cudaFree(dsidx)); CUDA_OK(cudaFree(dvals));
}

void test_reduce_scan() {
    std::uint32_t rows = 3, len = 500;   // len > block, multi-tile
    std::vector<float> in(rows * len);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t j = 0; j < len; ++j)
            in[r * len + j] = std::sin(0.1f * (r * len + j)) * 3.0f + (float)(j % 7);
    float* din = dev_from(in);

    float* dred = dev_alloc<float>(rows);
    for (auto [k, nm] : {std::pair{RedKind::Sum, "reduce_sum"}, {RedKind::Max, "reduce_max"}}) {
        k_reduce<float><<<rows, kCanonicalReduceWidth>>>(din, dred, rows, len, k);
        CUDA_OK(cudaDeviceSynchronize());
        check(nm, to_host(dred, rows), host_eval::reduce(k, in, rows, len));
    }
    std::uint32_t* darg = dev_alloc<std::uint32_t>(rows);
    k_reduce_argmax<<<rows, kTier0Block>>>(din, darg, rows, len);
    CUDA_OK(cudaDeviceSynchronize());
    check("reduce_argmax", to_host(darg, rows), host_eval::reduce_argmax(in, rows, len));

    std::vector<float> contract{1.0e20f, 1.0f, -1.0e20f, 1.0f};
    float* dcontract = dev_from(contract);
    k_reduce<float><<<1, kCanonicalReduceWidth>>>(
        dcontract, dred, 1, contract.size(), RedKind::Sum);
    CUDA_OK(cudaDeviceSynchronize());
    check("reduce_sum canonical tree", to_host(dred, 1), std::vector<float>{2.0f});
    std::vector<float> nan_arg{std::nanf(""), -INFINITY};
    float* dnan_arg = dev_from(nan_arg);
    k_reduce_argmax<<<1, kTier0Block>>>(dnan_arg, darg, 1, nan_arg.size());
    CUDA_OK(cudaDeviceSynchronize());
    check("reduce_argmax NaN/-inf", to_host(darg, 1), std::vector<std::uint32_t>{1});
    std::vector<float> signed_zero{-0.0f, 0.0f};
    float* dsigned_zero = dev_from(signed_zero);
    for (auto [kind, negative, name] : {
             std::tuple{RedKind::Max, false, "reduce_max signed zero"},
             std::tuple{RedKind::Min, true, "reduce_min signed zero"}}) {
        k_reduce<float><<<1, kCanonicalReduceWidth>>>(
            dsigned_zero, dred, 1, signed_zero.size(), kind);
        CUDA_OK(cudaDeviceSynchronize());
        const float value = to_host(dred, 1)[0];
        const bool ok = value == 0.0f && std::signbit(value) == negative;
        if (ok) {
            ++g_pass;
            std::printf("  PASS  %s\n", name);
        } else {
            ++g_fail;
            std::printf("  FAIL  %s\n", name);
        }
    }

    float* dscan = dev_alloc<float>(in.size());
    for (auto [k, nm] : {std::pair{ScanKind::Sum, "cumsum"}, {ScanKind::Prod, "cumprod"}}) {
        // cumprod on small magnitudes to stay finite
        std::vector<float> src = in;
        if (k == ScanKind::Prod) for (auto& v : src) v = 0.99f + 0.001f * v;
        float* ds = dev_from(src);
        k_scan<float><<<rows, kTier0Block>>>(ds, dscan, rows, len, k);
        CUDA_OK(cudaDeviceSynchronize());
        check(nm, to_host(dscan, in.size()), host_eval::scan(k, src, rows, len));
        CUDA_OK(cudaFree(ds));
    }
    CUDA_OK(cudaFree(din)); CUDA_OK(cudaFree(dred)); CUDA_OK(cudaFree(darg));
    CUDA_OK(cudaFree(dcontract)); CUDA_OK(cudaFree(dnan_arg));
    CUDA_OK(cudaFree(dsigned_zero));
    CUDA_OK(cudaFree(dscan));
}

void test_order_library() {
    std::uint32_t rows = 2, len = 32;
    std::vector<float> logits(rows * len);
    for (std::uint32_t i = 0; i < logits.size(); ++i) logits[i] = std::cos(0.3f * i) * 4.0f;
    float* dlog = dev_from(logits);

    // top_k (library kernel), k=4 with dynamic shared for the `taken` mask
    std::uint32_t tk = 4;
    float* dtv = dev_alloc<float>((std::size_t)rows * tk);
    std::uint32_t* dti = dev_alloc<std::uint32_t>((std::size_t)rows * tk);
    k_topk_rows<<<rows, kTier0Block, len * sizeof(std::uint8_t)>>>(dlog, dtv, dti, rows, len, tk);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> hv; std::vector<std::uint32_t> hi;
    host_eval::topk(logits, rows, len, tk, hv, hi);
    check("top_k.values", to_host(dtv, (std::size_t)rows * tk), hv);
    check("top_k.indices", to_host(dti, (std::size_t)rows * tk), hi);

    CUDA_OK(cudaFree(dlog));
    CUDA_OK(cudaFree(dtv)); CUDA_OK(cudaFree(dti));
}

// pivot_threshold's three DYNAMIC predicates (tier0_launch.hpp PivotThreshold
// dispatch): unlike the standalone `rank_le` above (an immediate `k`), the
// predicate payload here is ALWAYS a resolved device value (scalar or
// per-row) — exercises k_pivot_rankle<I32/U32> (dynamic k + NaN contract),
// k_pivot_probge (dynamic threshold), and k_pivot_cummassle (the CTA
// selection loop, incl. a 151936-token peaked-vocab scale smoke test — the
// "practical for 151k vocab" requirement).
void test_pivot_predicates() {
    std::printf("[pivot_threshold: dynamic rank_le/prob_ge/cummass_le predicates]\n");

    // rank_le: 2 rows, per-row U32 k = {2, 3}; row 1 also carries a NaN
    // (never selected, never counted toward another element's rank).
    {
        std::uint32_t rows = 2, len = 5;
        std::vector<float> x{5, 1, 4, 2, 3,           // row 0: ranks desc 5>4>3>2>1
                             std::nanf(""), 9, 9, 1, 2};  // row 1: NaN + a tie at 9
        std::vector<std::uint32_t> k{2, 3};
        float* dx = dev_from(x);
        std::uint32_t* dk = dev_from(k);
        std::uint8_t* dout = dev_alloc<std::uint8_t>(x.size());
        k_pivot_rankle<std::uint32_t><<<rows, kTier0Block>>>(dx, dout, rows, len, dk, (std::uint32_t)k.size());
        CUDA_OK(cudaDeviceSynchronize());
        check("pivot.rank_le (u32, per-row, NaN-safe)", to_host(dout, x.size()),
              host_eval::pivot_rankle<std::uint32_t>(x, rows, len, k, (std::uint32_t)k.size()));
        CUDA_OK(cudaFree(dx)); CUDA_OK(cudaFree(dk)); CUDA_OK(cudaFree(dout));
    }
    // rank_le: I32 k, scalar broadcast (k_numel=1).
    {
        std::uint32_t rows = 2, len = 4;
        std::vector<float> x{4, 3, 2, 1, 1, 2, 3, 4};
        std::vector<std::int32_t> k{2};
        float* dx = dev_from(x);
        std::int32_t* dk = dev_from(k);
        std::uint8_t* dout = dev_alloc<std::uint8_t>(x.size());
        k_pivot_rankle<std::int32_t><<<rows, kTier0Block>>>(dx, dout, rows, len, dk, 1u);
        CUDA_OK(cudaDeviceSynchronize());
        check("pivot.rank_le (i32, scalar broadcast)", to_host(dout, x.size()),
              host_eval::pivot_rankle<std::int32_t>(x, rows, len, k, 1u));
        CUDA_OK(cudaFree(dx)); CUDA_OK(cudaFree(dk)); CUDA_OK(cudaFree(dout));
    }
    // prob_ge: per-row F32 threshold.
    {
        std::uint32_t rows = 2, len = 4;
        std::vector<float> x{0.1f, 0.4f, 0.2f, 0.3f, 0.05f, 0.05f, 0.8f, 0.1f};
        std::vector<float> thr{0.25f, 0.5f};
        float* dx = dev_from(x);
        float* dthr = dev_from(thr);
        std::uint8_t* dout = dev_alloc<std::uint8_t>(x.size());
        k_pivot_probge<<<GS((std::uint64_t)rows * len), kTier0Block>>>(dx, dout, rows, len, dthr, (std::uint32_t)thr.size());
        CUDA_OK(cudaDeviceSynchronize());
        check("pivot.prob_ge (per-row)", to_host(dout, x.size()),
              host_eval::pivot_probge(x, rows, len, thr, (std::uint32_t)thr.size()));
        CUDA_OK(cudaFree(dx)); CUDA_OK(cudaFree(dthr)); CUDA_OK(cudaFree(dout));
    }
    // cummass_le: 2 rows — row 0 exercises a tie + a NaN tail; row 1 a plain
    // descending nucleus — with a per-row F32 `p`.
    {
        std::uint32_t rows = 2, len = 5;
        std::vector<float> x{0.5f, 0.5f, 0.3f, 0.2f, std::nanf(""),   // row 0: tie at 0.5, NaN last
                             0.6f, 0.1f, 0.1f, 0.1f, 0.1f};            // row 1: peaked
        std::vector<float> p{0.9f, 0.65f};
        float* dx = dev_from(x);
        float* dp = dev_from(p);
        std::uint8_t* dout = dev_alloc<std::uint8_t>(x.size());
        k_pivot_cummassle<<<rows, kTier0Block>>>(dx, dout, rows, len, dp, (std::uint32_t)p.size());
        CUDA_OK(cudaDeviceSynchronize());
        check("pivot.cummass_le (ties + NaN, per-row p)", to_host(dout, x.size()),
              host_eval::pivot_cummassle(x, rows, len, p, (std::uint32_t)p.size()));
        CUDA_OK(cudaFree(dx)); CUDA_OK(cudaFree(dp)); CUDA_OK(cudaFree(dout));
    }
    // cummass_le at the 151936-token vocab scale (a real LM head width): a
    // single sharply peaked row (one dominant logit) — the CTA selection
    // loop must terminate in a handful of picks, not a `len`-way rank pass,
    // to stay practical at this width.
    {
        std::uint32_t rows = 1, len = 151936;
        std::vector<float> x(len, 1e-6f);
        x[12345] = 1.0f;                       // one massively dominant token
        std::vector<float> p{0.99f};
        float* dx = dev_from(x);
        float* dp = dev_from(p);
        std::uint8_t* dout = dev_alloc<std::uint8_t>(x.size());
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_OK(cudaEventCreate(&start));
        CUDA_OK(cudaEventCreate(&stop));
        CUDA_OK(cudaEventRecord(start));
        k_pivot_cummassle<<<rows, kTier0Block>>>(dx, dout, rows, len, dp, 1u);
        CUDA_OK(cudaEventRecord(stop));
        CUDA_OK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        CUDA_OK(cudaDeviceSynchronize());
        check("pivot.cummass_le (151936-vocab peaked)", to_host(dout, x.size()),
              host_eval::pivot_cummassle(x, rows, len, p, 1u));
        if (elapsed_ms < 1000.0f) {
            ++g_pass;
            std::printf(
                "  PASS  production-vocab nucleus fallback timing (%g ms)\n",
                static_cast<double>(elapsed_ms));
        } else {
            ++g_fail;
            std::printf(
                "  FAIL  production-vocab nucleus fallback timing (%g ms)\n",
                static_cast<double>(elapsed_ms));
        }
        CUDA_OK(cudaEventDestroy(stop));
        CUDA_OK(cudaEventDestroy(start));
        CUDA_OK(cudaFree(dx)); CUDA_OK(cudaFree(dp)); CUDA_OK(cudaFree(dout));
    }
}

void test_shape_linear() {
    // broadcast: scalar and per-row
    std::vector<float> scal{3.5f};
    std::vector<float> per_row{1, 2, 3};
    std::uint32_t rows = 3, len = 4;
    float* dscal = dev_from(scal);
    float* dpr = dev_from(per_row);
    float* dbc = dev_alloc<float>((std::size_t)rows * len);
    k_broadcast<float><<<GS((std::uint64_t)rows * len), kTier0Block>>>(dscal, dbc, rows, len, 0);
    CUDA_OK(cudaDeviceSynchronize());
    check("broadcast_scalar", to_host(dbc, (std::size_t)rows * len), host_eval::broadcast(scal, rows, len, 0));
    k_broadcast<float><<<GS((std::uint64_t)rows * len), kTier0Block>>>(dpr, dbc, rows, len, 1);
    CUDA_OK(cudaDeviceSynchronize());
    check("broadcast_row", to_host(dbc, (std::size_t)rows * len), host_eval::broadcast(per_row, rows, len, 1));

    // transpose [3,4] -> [4,3]
    std::vector<float> m(rows * len);
    for (std::uint32_t i = 0; i < m.size(); ++i) m[i] = (float)i;
    float* dm = dev_from(m);
    float* dt = dev_alloc<float>(m.size());
    dim3 blk(16, 16), grd((len + 15) / 16, (rows + 15) / 16);
    k_transpose<float><<<grd, blk>>>(dm, dt, rows, len);
    CUDA_OK(cudaDeviceSynchronize());
    check("transpose", to_host(dt, m.size()), host_eval::transpose(m, rows, len));

    // matmul [2,3]x[3,4] -> [2,4]
    std::uint32_t M = 2, K = 3, N = 4;
    std::vector<float> A(M * K), B(K * N);
    for (std::uint32_t i = 0; i < A.size(); ++i) A[i] = (float)(i + 1);
    for (std::uint32_t i = 0; i < B.size(); ++i) B[i] = (float)(2 * i - 3);
    float* dA = dev_from(A);
    float* dB = dev_from(B);
    float* dC = dev_alloc<float>((std::size_t)M * N);
    dim3 mblk(32, 1), mgrd((N + 31) / 32, M);
    k_matmul<<<mgrd, mblk>>>(dA, dB, dC, M, K, N);
    CUDA_OK(cudaDeviceSynchronize());
    check("matmul", to_host(dC, (std::size_t)M * N), host_eval::matmul(A, B, M, K, N));

    CUDA_OK(cudaFree(dscal)); CUDA_OK(cudaFree(dpr)); CUDA_OK(cudaFree(dbc));
    CUDA_OK(cudaFree(dm)); CUDA_OK(cudaFree(dt));
    CUDA_OK(cudaFree(dA)); CUDA_OK(cudaFree(dB)); CUDA_OK(cudaFree(dC));
}

void test_new_ops() {
    std::printf("[new ops: max/min_elem, recip/abs/sign, reduce_min, gather_row, scatter_add, sort_desc, mask_apply_packed, rng_keyed]\n");
    std::vector<float> a{1, 5, 3, 8, 2, 6, -7, 4};
    std::vector<float> b{2, 2, 9, 3, 5, 4, -2, 3};
    float *da = dev_from(a), *db = dev_from(b), *dout = dev_alloc<float>(a.size());
    for (auto [k, nm] : {std::pair{BinKind::MaxElem, "max_elem"}, {BinKind::MinElem, "min_elem"}}) {
        k_binary<float><<<GS(a.size()), kTier0Block>>>(da, db, dout, a.size(), k);
        CUDA_OK(cudaDeviceSynchronize());
        check(nm, to_host(dout, a.size()), host_eval::binary(k, a, b));
    }

    auto test_structured_masks = [] {
        std::printf("[structured attention masks]\n");
        const std::vector<std::uint32_t> positions{3, 5};
        auto* positions_device = dev_from(positions);
        auto* mask_device = dev_alloc<std::uint8_t>(12);
        const std::vector<std::uint8_t> causal{
            1, 1, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 1,
        };
        const std::vector<std::uint8_t> sliding{
            0, 1, 1, 1, 0, 0,
            0, 0, 0, 1, 1, 1,
        };
        const std::vector<std::uint8_t> sink{
            1, 1, 1, 1, 0, 0,
            1, 0, 0, 1, 1, 1,
        };
        LaunchOp mask_launch;
        mask_launch.code = OpCode::CausalMask;
        mask_launch.in = {positions_device};
        mask_launch.out = mask_device;
        mask_launch.rows = 2;
        mask_launch.len = 6;
        mask_launch.imm = 6;
        if (!launch_op(mask_launch)) {
            ++g_fail;
            std::printf("  FAIL  launch causal semantic mask\n");
        }
        CUDA_OK(cudaDeviceSynchronize());
        check("causal_mask", to_host(mask_device, 12), causal);
        mask_launch.code = OpCode::SlidingWindowMask;
        mask_launch.imm2 = 3;
        if (!launch_op(mask_launch)) {
            ++g_fail;
            std::printf("  FAIL  launch sliding semantic mask\n");
        }
        CUDA_OK(cudaDeviceSynchronize());
        check("sliding_window_mask", to_host(mask_device, 12), sliding);
        mask_launch.code = OpCode::SinkWindowMask;
        mask_launch.imm2 = 1;
        mask_launch.imm3 = 3;
        if (!launch_op(mask_launch)) {
            ++g_fail;
            std::printf("  FAIL  launch sink-window semantic mask\n");
        }
        CUDA_OK(cudaDeviceSynchronize());
        check("sink_window_mask", to_host(mask_device, 12), sink);
        mask_launch.code = OpCode::SlidingWindowMask;
        mask_launch.imm2 = 0;
        if (!launch_op(mask_launch)) {
            ++g_fail;
            std::printf("  FAIL  launch zero-width sliding mask\n");
        }
        CUDA_OK(cudaDeviceSynchronize());
        check(
            "sliding_window_mask.window_zero",
            to_host(mask_device, 12),
            std::vector<std::uint8_t>(12, 0));
        mask_launch.code = OpCode::SinkWindowMask;
        mask_launch.imm2 = 99;
        mask_launch.imm3 = 0;
        if (!launch_op(mask_launch)) {
            ++g_fail;
            std::printf("  FAIL  launch oversized sink mask\n");
        }
        CUDA_OK(cudaDeviceSynchronize());
        check(
            "sink_window_mask.sink_gt_key_len",
            to_host(mask_device, 12),
            causal);

        CUDA_OK(cudaFree(mask_device));
        CUDA_OK(cudaFree(positions_device));
    };
    test_structured_masks();

    auto test_integer_semantics = [] {
        std::printf("[integer wrapping unary/reductions]\n");
        const std::vector<std::int32_t> input{
            std::numeric_limits<std::int32_t>::min(),
            16777217,
            -16777217,
            -1,
            0,
            1,
            std::numeric_limits<std::int32_t>::max(),
            1,
        };
        auto* device_input = dev_from(input);
        auto* device_output = dev_alloc<std::int32_t>(input.size());
        const std::vector<std::int32_t> expected_neg{
            std::numeric_limits<std::int32_t>::min(),
            -16777217,
            16777217,
            1,
            0,
            -1,
            -std::numeric_limits<std::int32_t>::max(),
            -1,
        };
        const std::vector<std::int32_t> expected_abs{
            std::numeric_limits<std::int32_t>::min(),
            16777217,
            16777217,
            1,
            0,
            1,
            std::numeric_limits<std::int32_t>::max(),
            1,
        };
        const std::vector<std::int32_t> expected_sign{
            -1, 1, -1, -1, 0, 1, 1, 1,
        };
        for (const auto& [kind, name, expected] :
             std::vector<std::tuple<
                 UnKind,
                 const char*,
                 std::vector<std::int32_t>>>{
                 {UnKind::Neg, "i32.neg", expected_neg},
                 {UnKind::Abs, "i32.abs", expected_abs},
                 {UnKind::Sign, "i32.sign", expected_sign},
             }) {
            k_unary<std::int32_t><<<GS(input.size()), kTier0Block>>>(
                device_input, device_output, input.size(), kind);
            CUDA_OK(cudaDeviceSynchronize());
            check(name, to_host(device_output, input.size()), expected);
        }
        LaunchOp solo_unary;
        solo_unary.code = OpCode::Neg;
        solo_unary.elem_dtype = DType::I32;
        solo_unary.in = {device_input};
        solo_unary.out = device_output;
        solo_unary.numel = input.size();
        if (!launch_op(solo_unary)) {
            ++g_fail;
            std::printf("  FAIL  solo i32 unary dispatch\n");
        }
        CUDA_OK(cudaDeviceSynchronize());
        check(
            "solo.i32.neg.dispatch",
            to_host(device_output, input.size()),
            expected_neg);

        const std::vector<std::uint32_t> unsigned_input{
            0, 1, 16777217u, 0xffffffffu,
        };
        auto* unsigned_device = dev_from(unsigned_input);
        auto* unsigned_output =
            dev_alloc<std::uint32_t>(unsigned_input.size());
        k_unary<std::uint32_t><<<GS(unsigned_input.size()), kTier0Block>>>(
            unsigned_device,
            unsigned_output,
            unsigned_input.size(),
            UnKind::Neg);
        CUDA_OK(cudaDeviceSynchronize());
        check(
            "u32.neg",
            to_host(unsigned_output, unsigned_input.size()),
            std::vector<std::uint32_t>{
                0, 0xffffffffu, 0xfeffffffu, 1});
        k_unary<std::uint32_t><<<GS(unsigned_input.size()), kTier0Block>>>(
            unsigned_device,
            unsigned_output,
            unsigned_input.size(),
            UnKind::Sign);
        CUDA_OK(cudaDeviceSynchronize());
        check(
            "u32.sign",
            to_host(unsigned_output, unsigned_input.size()),
            std::vector<std::uint32_t>{0, 1, 1, 1});
        LaunchOp solo_unsigned;
        solo_unsigned.code = OpCode::Sign;
        solo_unsigned.elem_dtype = DType::U32;
        solo_unsigned.in = {unsigned_device};
        solo_unsigned.out = unsigned_output;
        solo_unsigned.numel = unsigned_input.size();
        if (!launch_op(solo_unsigned)) {
            ++g_fail;
            std::printf("  FAIL  solo u32 unary dispatch\n");
        }
        CUDA_OK(cudaDeviceSynchronize());
        check(
            "solo.u32.sign.dispatch",
            to_host(unsigned_output, unsigned_input.size()),
            std::vector<std::uint32_t>{0, 1, 1, 1});

        const std::vector<std::uint32_t> argmax_u32{
            16777216u, 16777217u};
        const std::vector<std::int32_t> argmax_i32{
            -16777217, -16777216};
        auto* argmax_u32_device = dev_from(argmax_u32);
        auto* argmax_i32_device = dev_from(argmax_i32);
        auto* argmax_output = dev_alloc<std::uint32_t>(1);
        LaunchOp argmax_launch;
        argmax_launch.code = OpCode::ReduceArgmax;
        argmax_launch.out = argmax_output;
        argmax_launch.rows = 1;
        argmax_launch.len = 2;
        argmax_launch.elem_dtype = DType::U32;
        argmax_launch.in = {argmax_u32_device};
        if (!launch_op(argmax_launch)) {
            ++g_fail;
            std::printf("  FAIL  solo u32 argmax dispatch\n");
        }
        CUDA_OK(cudaDeviceSynchronize());
        check(
            "solo.u32.argmax.exact",
            to_host(argmax_output, 1),
            std::vector<std::uint32_t>{1});
        argmax_launch.elem_dtype = DType::I32;
        argmax_launch.in = {argmax_i32_device};
        if (!launch_op(argmax_launch)) {
            ++g_fail;
            std::printf("  FAIL  solo i32 argmax dispatch\n");
        }
        CUDA_OK(cudaDeviceSynchronize());
        check(
            "solo.i32.argmax.exact",
            to_host(argmax_output, 1),
            std::vector<std::uint32_t>{1});
        CUDA_OK(cudaFree(argmax_output));
        CUDA_OK(cudaFree(argmax_i32_device));
        CUDA_OK(cudaFree(argmax_u32_device));

        constexpr std::uint32_t rows = 2;
        constexpr std::uint32_t len = 4;
        const std::vector<std::int32_t> reduction_input{
            std::numeric_limits<std::int32_t>::max(),
            1,
            16777217,
            -16777217,
            std::numeric_limits<std::int32_t>::min(),
            -1,
            1,
            0,
        };
        auto* reduction_device = dev_from(reduction_input);
        auto* reduction_output = dev_alloc<std::int32_t>(rows);
        k_reduce<std::int32_t><<<rows, kCanonicalReduceWidth>>>(
            reduction_device,
            reduction_output,
            rows,
            len,
            RedKind::Sum);
        CUDA_OK(cudaDeviceSynchronize());
        check(
            "i32.reduce_sum",
            to_host(reduction_output, rows),
            std::vector<std::int32_t>{
                std::numeric_limits<std::int32_t>::min(),
                std::numeric_limits<std::int32_t>::min()});
        LaunchOp solo_reduce;
        solo_reduce.code = OpCode::ReduceSum;
        solo_reduce.elem_dtype = DType::I32;
        solo_reduce.in = {reduction_device};
        solo_reduce.out = reduction_output;
        solo_reduce.rows = rows;
        solo_reduce.len = len;
        if (!launch_op(solo_reduce)) {
            ++g_fail;
            std::printf("  FAIL  solo i32 reduction dispatch\n");
        }
        CUDA_OK(cudaDeviceSynchronize());
        check(
            "solo.i32.reduce_sum.dispatch",
            to_host(reduction_output, rows),
            std::vector<std::int32_t>{
                std::numeric_limits<std::int32_t>::min(),
                std::numeric_limits<std::int32_t>::min()});
        k_reduce<std::int32_t><<<rows, kCanonicalReduceWidth>>>(
            reduction_device,
            reduction_output,
            rows,
            len,
            RedKind::Max);
        CUDA_OK(cudaDeviceSynchronize());
        check(
            "i32.reduce_max",
            to_host(reduction_output, rows),
            std::vector<std::int32_t>{
                std::numeric_limits<std::int32_t>::max(), 1});
        k_reduce<std::int32_t><<<rows, kCanonicalReduceWidth>>>(
            reduction_device,
            reduction_output,
            rows,
            len,
            RedKind::Min);
        CUDA_OK(cudaDeviceSynchronize());
        check(
            "i32.reduce_min",
            to_host(reduction_output, rows),
            std::vector<std::int32_t>{
                -16777217, std::numeric_limits<std::int32_t>::min()});

        CUDA_OK(cudaFree(reduction_output));
        CUDA_OK(cudaFree(reduction_device));
        CUDA_OK(cudaFree(unsigned_output));
        CUDA_OK(cudaFree(unsigned_device));
        CUDA_OK(cudaFree(device_output));
        CUDA_OK(cudaFree(device_input));
    };
    test_integer_semantics();
    for (auto [k, nm] : {std::pair{UnKind::Recip, "recip"}, {UnKind::Abs, "abs"}, {UnKind::Sign, "sign"}}) {
        k_unary<float><<<GS(a.size()), kTier0Block>>>(da, dout, a.size(), k);
        CUDA_OK(cudaDeviceSynchronize());
        check(nm, to_host(dout, a.size()), host_eval::unary(k, a));
    }
    // reduce_min over rows
    std::uint32_t rows = 2, len = 4;
    std::vector<float> m{3, 1, 4, 1, 9, 2, 6, 5};
    float* dm = dev_from(m);
    float* dr = dev_alloc<float>(rows);
    k_reduce<float><<<rows, kCanonicalReduceWidth>>>(dm, dr, rows, len, RedKind::Min);
    CUDA_OK(cudaDeviceSynchronize());
    check("reduce_min", to_host(dr, rows), host_eval::reduce(RedKind::Min, m, rows, len));

    // gather_row: one scalar column selection per source row.
    std::vector<float> src{0,1,2, 3,4,5, 6,7,8, 9,10,11};
    std::vector<std::int32_t> idx{2, -1, 3, 0};
    float* dsrc = dev_from(src);
    std::int32_t* didx = dev_from(idx);
    float* dgr = dev_alloc<float>(idx.size());
    k_gather_row<float, std::int32_t><<<
        GS(idx.size()), kTier0Block>>>(
        dsrc, didx, dgr, idx.size(), 3);
    CUDA_OK(cudaDeviceSynchronize());
    check(
        "gather_row.i32_bounds",
        to_host(dgr, idx.size()),
        std::vector<float>{2, 0, 0, 9});

    // ScatterAdd is distinct from ScatterSet and supports scalar value
    // broadcasting over the selected axis-0 rows.
    std::vector<float> base{0,0,0,0,0,0};
    std::vector<std::uint32_t> sidx{1, 1, 4};
    std::vector<float> vals{5};
    float* dbase = dev_from(base);
    std::uint32_t* dsidx = dev_from(sidx);
    float* dvals = dev_from(vals);
    CUDA_OK(cudaMemcpy(dbase, base.data(), base.size()*sizeof(float), cudaMemcpyHostToDevice));
    k_scatter_axis0_serial<float, std::uint32_t, true><<<1,1>>>(
        dbase, dsidx, dvals, sidx.size(), 3, 2, true);
    CUDA_OK(cudaDeviceSynchronize());
    check(
        "scatter_add.axis0_scalar_bounds",
        to_host(dbase, base.size()),
        std::vector<float>{0, 0, 10, 10, 0, 0});

    // sort_desc [2,5]
    std::uint32_t sr = 2, sl = 5;
    std::vector<float> sv{3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    float* dsv = dev_from(sv);
    float* dsval = dev_alloc<float>(sv.size());
    std::uint32_t* dsidx2 = dev_alloc<std::uint32_t>(sv.size());
    k_topk_rows<<<sr, kTier0Block, sl*sizeof(std::uint8_t)>>>(dsv, dsval, dsidx2, sr, sl, sl);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> ev; std::vector<std::uint32_t> ei;
    host_eval::sort_desc(sv, sr, sl, ev, ei);
    check("sort_desc.values", to_host(dsval, sv.size()), ev);
    check("sort_desc.indices", to_host(dsidx2, sv.size()), ei);
    const std::vector<float> adversarial{
        INFINITY,
        std::numeric_limits<float>::quiet_NaN(),
        INFINITY,
        1.0f,
        -0.0f,
        0.0f,
        -INFINITY,
        std::numeric_limits<float>::quiet_NaN(),
    };
    const std::vector<std::uint32_t> adversarial_order{
        0, 2, 3, 4, 5, 6, 1, 7};
    float* adversarial_device = dev_from(adversarial);
    float* adversarial_values = dev_alloc<float>(adversarial.size());
    std::uint32_t* adversarial_indices =
        dev_alloc<std::uint32_t>(adversarial.size());
    k_topk_rows<<<1, kTier0Block>>>(
        adversarial_device,
        adversarial_values,
        adversarial_indices,
        1,
        static_cast<std::uint32_t>(adversarial.size()),
        static_cast<std::uint32_t>(adversarial.size()));
    CUDA_OK(cudaDeviceSynchronize());
    check(
        "topk.total_order.indices",
        to_host(adversarial_indices, adversarial.size()),
        adversarial_order);
    const auto ordered_values =
        to_host(adversarial_values, adversarial.size());
    bool exact_bits = true;
    for (std::size_t index = 0; index < adversarial.size(); ++index) {
        exact_bits = exact_bits &&
            std::memcmp(
                &ordered_values[index],
                &adversarial[adversarial_order[index]],
                sizeof(float)) == 0;
    }
    if (exact_bits) {
        ++g_pass;
        std::printf("  PASS  topk.total_order.value_bits (n=8)\n");
    } else {
        ++g_fail;
        std::printf("  FAIL  topk.total_order.value_bits (n=8)\n");
    }
    CUDA_OK(cudaFree(adversarial_indices));
    CUDA_OK(cudaFree(adversarial_values));
    CUDA_OK(cudaFree(adversarial_device));

    // mask_apply_packed [2, 40] (2 mask words/row)
    std::uint32_t pr = 2, pl = 40, pw = (pl + 31) / 32;
    std::vector<float> plog(pr * pl);
    for (std::uint32_t i = 0; i < plog.size(); ++i) plog[i] = 0.5f * i - 3.0f;
    std::vector<std::uint32_t> pmask(pr * pw);
    for (std::uint32_t i = 0; i < pmask.size(); ++i) pmask[i] = 0xA5A5A5A5u ^ (i * 2654435761u);
    float* dplog = dev_from(plog);
    std::uint32_t* dpmask = dev_from(pmask);
    float* dpout = dev_alloc<float>(plog.size());
    k_mask_apply_packed<<<GS(plog.size()), kTier0Block>>>(dplog, dpmask, dpout, pr, pl, pw);
    CUDA_OK(cudaDeviceSynchronize());
    check("mask_apply_packed", to_host(dpout, plog.size()), host_eval::mask_apply_packed(plog, pmask, pr, pl, pw));

    // rng_keyed: state=[key,ctr], numel=32, uniform + gumbel
    std::vector<std::uint32_t> state{0xDEADBEEFu, 7u};
    std::uint32_t* dstate = dev_from(state);
    std::uint32_t rn = 32;
    float* drk = dev_alloc<float>(rn);
    k_rng_keyed<<<GS(rn), kTier0Block>>>(dstate, drk, rn, 0);
    CUDA_OK(cudaDeviceSynchronize());
    check("rng_keyed.uniform", to_host(drk, rn), host_eval::rng_keyed(state[0], state[1], rn, false));
    k_rng_keyed<<<GS(rn), kTier0Block>>>(dstate, drk, rn, 1);
    CUDA_OK(cudaDeviceSynchronize());
    check("rng_keyed.gumbel", to_host(drk, rn), host_eval::rng_keyed(state[0], state[1], rn, true));

    // general broadcast: [1,1,4]->[2,3,4] (tiling) and [2,3,1]->[2,3,4] (last-dim)
    {
        std::vector<std::uint32_t> s1{10, 11, 12, 13};   // [1,1,4]
        std::uint32_t meta1[8] = {2,3,4,1, 0,0,1,0};     // tdims, sstride (broadcast dims 0,1)
        std::uint32_t* dm1 = dev_from(std::vector<std::uint32_t>(meta1, meta1+8));
        std::uint32_t* ds1 = dev_from(s1);
        std::uint32_t* dbg = dev_alloc<std::uint32_t>(24);
        k_broadcast_general<std::uint32_t><<<GS(24), kTier0Block>>>(ds1, dbg, dm1, 3, 24);
        CUDA_OK(cudaDeviceSynchronize());
        check("broadcast_general[1,1,4]->[2,3,4]", to_host(dbg, 24),
              host_eval::broadcast_general<std::uint32_t>(s1, {1,1,4}, {2,3,4}));
        std::vector<std::uint32_t> s2{1,2,3, 4,5,6};      // [2,3,1]
        std::uint32_t meta2[8] = {2,3,4,1, 3,1,0,0};      // sstride: dim2 broadcast
        std::uint32_t* dm2 = dev_from(std::vector<std::uint32_t>(meta2, meta2+8));
        std::uint32_t* ds2 = dev_from(s2);
        k_broadcast_general<std::uint32_t><<<GS(24), kTier0Block>>>(ds2, dbg, dm2, 3, 24);
        CUDA_OK(cudaDeviceSynchronize());
        check("broadcast_general[2,3,1]->[2,3,4]", to_host(dbg, 24),
              host_eval::broadcast_general<std::uint32_t>(s2, {2,3,1}, {2,3,4}));
        CUDA_OK(cudaFree(dm1)); CUDA_OK(cudaFree(ds1)); CUDA_OK(cudaFree(dbg));
        CUDA_OK(cudaFree(dm2)); CUDA_OK(cudaFree(ds2));
    }

    CUDA_OK(cudaFree(da)); CUDA_OK(cudaFree(db)); CUDA_OK(cudaFree(dout)); CUDA_OK(cudaFree(dm));
    CUDA_OK(cudaFree(dr)); CUDA_OK(cudaFree(dsrc)); CUDA_OK(cudaFree(didx)); CUDA_OK(cudaFree(dgr));
    CUDA_OK(cudaFree(dbase)); CUDA_OK(cudaFree(dsidx)); CUDA_OK(cudaFree(dvals));
    CUDA_OK(cudaFree(dsv)); CUDA_OK(cudaFree(dsval)); CUDA_OK(cudaFree(dsidx2));
    CUDA_OK(cudaFree(dplog)); CUDA_OK(cudaFree(dpmask)); CUDA_OK(cudaFree(dpout));
    CUDA_OK(cudaFree(dstate)); CUDA_OK(cudaFree(drk));
}

}  // namespace

int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDevice(&dev));
    CUDA_OK(cudaGetDeviceProperties(&prop, dev));
    std::printf("PTIR tier-0 kernel parity — device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    std::printf("[map]\n");                   test_map();
    std::printf("[compare/logic/select/cast]\n"); test_compare_logic_select_cast();
    std::printf("[index]\n");                 test_index();
    std::printf("[reduce/scan]\n"); test_reduce_scan();
    std::printf("[order/library]\n"); test_order_library();
    std::printf("[shape/linear]\n");          test_shape_linear();
    test_new_ops();
    test_pivot_predicates();

    std::printf("\n==== tier-0 parity: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
