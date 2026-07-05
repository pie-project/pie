// PTIR tier-0 kernel parity test (charlie, thrust-3 P4.2 milestone M-A/M-B).
//
// Runs every tier-0 op kernel (tier0_kernels.cuh) on the live GPU and diffs the
// result against the host reference evaluator (host_eval.hpp). This is the
// self-check gate that must hold before diffing against echo's canonical golden
// vectors. Standalone: compiled directly by nvcc, no driver-lib dependency.
//
//   nvcc -std=c++17 -I../src tests/ptir_tier0_test.cu -o ptir_tier0_test && ./ptir_tier0_test

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "ptir/host_eval.hpp"
#include "ptir/tier0_kernels.cuh"

using namespace pie_cuda_driver::ptir;

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
    std::vector<std::uint32_t> idx{7, 0, 3, 3, 5, 1};
    float* dsrc = dev_from(src);
    std::uint32_t* didx = dev_from(idx);
    float* dg = dev_alloc<float>(idx.size());
    k_gather<float><<<GS(idx.size()), kTier0Block>>>(dsrc, didx, dg, idx.size());
    CUDA_OK(cudaDeviceSynchronize());
    check("gather", to_host(dg, idx.size()), host_eval::gather(src, idx));

    // scatter_set with a duplicate target (index 3 written twice, last wins)
    std::vector<float> base{0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<std::uint32_t> sidx{2, 3, 3, 6};
    std::vector<float> vals{9, 4, 7, 5};
    float* dbase = dev_from(base);
    std::uint32_t* dsidx = dev_from(sidx);
    float* dvals = dev_from(vals);
    k_scatter_set_serial<float><<<1, 1>>>(dbase, dsidx, dvals, (std::uint32_t)sidx.size());
    CUDA_OK(cudaDeviceSynchronize());
    check("scatter_set", to_host(dbase, base.size()), host_eval::scatter_set(base, sidx, vals));

    CUDA_OK(cudaFree(diota)); CUDA_OK(cudaFree(dsrc)); CUDA_OK(cudaFree(didx)); CUDA_OK(cudaFree(dg));
    CUDA_OK(cudaFree(dbase)); CUDA_OK(cudaFree(dsidx)); CUDA_OK(cudaFree(dvals));
}

void test_reduce_scan_normalize() {
    std::uint32_t rows = 3, len = 500;   // len > block, multi-tile
    std::vector<float> in(rows * len);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t j = 0; j < len; ++j)
            in[r * len + j] = std::sin(0.1f * (r * len + j)) * 3.0f + (float)(j % 7);
    float* din = dev_from(in);

    float* dred = dev_alloc<float>(rows);
    for (auto [k, nm] : {std::pair{RedKind::Sum, "reduce_sum"}, {RedKind::Max, "reduce_max"}}) {
        k_reduce<float><<<rows, kTier0Block>>>(din, dred, rows, len, k);
        CUDA_OK(cudaDeviceSynchronize());
        check(nm, to_host(dred, rows), host_eval::reduce(k, in, rows, len));
    }
    std::uint32_t* darg = dev_alloc<std::uint32_t>(rows);
    k_reduce_argmax<<<rows, kTier0Block>>>(din, darg, rows, len);
    CUDA_OK(cudaDeviceSynchronize());
    check("reduce_argmax", to_host(darg, rows), host_eval::reduce_argmax(in, rows, len));

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
    float* dnorm = dev_alloc<float>(in.size());
    for (auto [k, nm] : {std::pair{NormKind::Softmax, "softmax"}, {NormKind::LogSoftmax, "log_softmax"},
                         {NormKind::L2Norm, "l2norm"}}) {
        k_normalize<<<rows, kTier0Block>>>(din, dnorm, rows, len, k);
        CUDA_OK(cudaDeviceSynchronize());
        check(nm, to_host(dnorm, in.size()), host_eval::normalize(k, in, rows, len));
    }
    CUDA_OK(cudaFree(din)); CUDA_OK(cudaFree(dred)); CUDA_OK(cudaFree(darg));
    CUDA_OK(cudaFree(dscan)); CUDA_OK(cudaFree(dnorm));
}

void test_sampling_order_library() {
    std::uint32_t rows = 2, len = 32;
    std::vector<float> logits(rows * len);
    for (std::uint32_t i = 0; i < logits.size(); ++i) logits[i] = std::cos(0.3f * i) * 4.0f;
    std::vector<std::uint8_t> mask(rows * len);
    for (std::uint32_t i = 0; i < mask.size(); ++i) mask[i] = (i % 3 != 0) ? 1u : 0u;
    float* dlog = dev_from(logits);
    std::uint8_t* dmask = dev_from(mask);
    float* dout = dev_alloc<float>(logits.size());
    k_mask_apply<<<GS(logits.size()), kTier0Block>>>(dlog, dmask, dout, logits.size());
    CUDA_OK(cudaDeviceSynchronize());
    check("mask_apply", to_host(dout, logits.size()), host_eval::mask_apply(logits, mask));

    // gumbel — parity with host RNG
    std::vector<std::uint32_t> seeds{123456u, 987654u};
    std::uint32_t* dseed = dev_from(seeds);
    float* dg = dev_alloc<float>((std::size_t)rows * len);
    k_gumbel<<<GS((std::uint64_t)rows * len), kTier0Block>>>(dseed, /*stream=*/0u, dg, rows, len);
    CUDA_OK(cudaDeviceSynchronize());
    check("gumbel(stream0)", to_host(dg, (std::size_t)rows * len), host_eval::gumbel(seeds, 0u, rows, len));

    // rank_le / pivot_threshold, k=5
    std::uint32_t k = 5;
    std::uint8_t* drl = dev_alloc<std::uint8_t>(logits.size());
    k_rank_le<<<rows, kTier0Block>>>(dlog, drl, rows, len, k);
    CUDA_OK(cudaDeviceSynchronize());
    check("rank_le", to_host(drl, logits.size()), host_eval::rank_le(logits, rows, len, k));

    float* dpt = dev_alloc<float>(logits.size());
    k_pivot_threshold_rankle<<<rows, kTier0Block>>>(dlog, dpt, rows, len, k);
    CUDA_OK(cudaDeviceSynchronize());
    check("pivot_threshold", to_host(dpt, logits.size()), host_eval::pivot_threshold_rankle(logits, rows, len, k));

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

    CUDA_OK(cudaFree(dlog)); CUDA_OK(cudaFree(dmask)); CUDA_OK(cudaFree(dout));
    CUDA_OK(cudaFree(dseed)); CUDA_OK(cudaFree(dg)); CUDA_OK(cudaFree(drl));
    CUDA_OK(cudaFree(dpt)); CUDA_OK(cudaFree(dtv)); CUDA_OK(cudaFree(dti));
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
    std::printf("[reduce/scan/normalize]\n"); test_reduce_scan_normalize();
    std::printf("[sampling/order/library]\n"); test_sampling_order_library();
    std::printf("[shape/linear]\n");          test_shape_linear();

    std::printf("\n==== tier-0 parity: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
