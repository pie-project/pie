// Standalone unit test for the sampling-IR primitive kernel templates
// (Lane L2 / charlie). It compiles the *canonical NVRTC-safe device prelude*
// (driver/cuda/src/sampling_ir/primitives_src.hpp) through NVRTC for the
// detected GPU arch, loads the resulting PTX via the CUDA driver API, launches
// per-primitive micro-kernels, and validates each against a CPU reference.
//
// This is high-fidelity: the bytes compiled here are the exact bytes the W2
// codegen will prepend to every generated kernel, run through the same NVRTC
// path delta's JIT uses. Run with `ctest -R sampling_ir_primitives` or invoke
// the binary directly (exit code 0 = all pass).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "sampling_ir/primitives_src.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

#define CU_CHECK(expr)                                                        \
    do {                                                                      \
        CUresult _e = (expr);                                                 \
        if (_e != CUDA_SUCCESS) {                                             \
            const char* _s = nullptr;                                         \
            cuGetErrorString(_e, &_s);                                        \
            std::fprintf(stderr, "CU error %d (%s) at %s:%d: %s\n",           \
                         (int)_e, _s ? _s : "?", __FILE__, __LINE__, #expr);  \
            std::exit(2);                                                     \
        }                                                                     \
    } while (0)

#define RT_CHECK(expr)                                                        \
    do {                                                                      \
        cudaError_t _e = (expr);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA RT error %s at %s:%d: %s\n",           \
                         cudaGetErrorString(_e), __FILE__, __LINE__, #expr);  \
            std::exit(2);                                                     \
        }                                                                     \
    } while (0)

#define NVRTC_CHECK(expr)                                                     \
    do {                                                                      \
        nvrtcResult _e = (expr);                                             \
        if (_e != NVRTC_SUCCESS) {                                            \
            std::fprintf(stderr, "NVRTC error %s at %s:%d: %s\n",             \
                         nvrtcGetErrorString(_e), __FILE__, __LINE__, #expr); \
            std::exit(2);                                                     \
        }                                                                     \
    } while (0)

void expect(bool cond, const char* what) {
    ++g_checks;
    if (!cond) {
        ++g_failures;
        std::fprintf(stderr, "  FAIL: %s\n", what);
    }
}

// Compile `prelude + body` for the current device arch, return a loaded module.
CUmodule compile_module(const std::string& body, int cc_major, int cc_minor) {
    std::string src = std::string(pie_cuda_driver::sampling_ir::primitive_prelude()) +
                      "\n" + body;

    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), "sampling_ir_test.cu",
                                   0, nullptr, nullptr));
    char arch[64];
    std::snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d",
                  cc_major, cc_minor);
    const char* opts[] = {arch, "--std=c++17"};
    nvrtcResult cres = nvrtcCompileProgram(prog, 2, opts);
    if (cres != NVRTC_SUCCESS) {
        size_t log_size = 0;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        nvrtcGetProgramLog(prog, log.data());
        std::fprintf(stderr, "NVRTC compile failed:\n%s\n", log.c_str());
        std::exit(2);
    }
    size_t ptx_size = 0;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
    std::string ptx(ptx_size, '\0');
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));

    CUmodule mod;
    CU_CHECK(cuModuleLoadData(&mod, ptx.c_str()));
    return mod;
}

CUfunction get_fn(CUmodule mod, const char* name) {
    CUfunction fn;
    CU_CHECK(cuModuleGetFunction(&fn, mod, name));
    return fn;
}

void launch(CUfunction fn, int grid, int block, void** args) {
    CU_CHECK(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr));
    CU_CHECK(cuCtxSynchronize());
}

template <typename T>
T* dmalloc(size_t n) {
    void* p = nullptr;
    RT_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return static_cast<T*>(p);
}
template <typename T>
void up(T* d, const std::vector<T>& h) {
    RT_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T>
void down(std::vector<T>& h, const T* d) {
    RT_CHECK(cudaMemcpy(h.data(), d, h.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

bool approx(float a, float b, float rtol = 1e-4f, float atol = 1e-5f) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

// --- CPU reference of the parity RNG (mirrors the prelude / sample_temp.cu) --
uint64_t splitmix64_ref(uint64_t x) {
    x ^= x >> 27; x *= 0x3C79AC492BA7B653ULL;
    x ^= x >> 33; x *= 0x1C69B3F74AC4AE35ULL;
    x ^= x >> 27;
    return x;
}
float hash_uniform_ref(uint64_t seed_eff, int j) {
    uint64_t x = seed_eff + 0x9E3779B97F4A7C15ULL * (uint64_t)(j + 1);
    x = splitmix64_ref(x);
    uint32_t bits = (uint32_t)(x >> 40);
    return ((float)bits + 0.5f) * (1.0f / 16777216.0f);
}

// =========================== Test cases =====================================

void test_rng(CUmodule mod) {
    std::printf("[rng parity]\n");
    const int n = 4096;
    const uint32_t seed = 0xC0FFEEu;
    float* d_u = dmalloc<float>(n);
    float* d_g = dmalloc<float>(n);
    CUfunction fn = get_fn(mod, "k_rng");
    void* args[] = {(void*)&seed, &d_u, &d_g, (void*)&n};
    launch(fn, 1, 256, args);

    std::vector<float> u(n), g(n);
    down(u, d_u); down(g, d_g);

    const uint64_t seed_eff = (uint64_t)seed ^ 0xA5A5A5A5ULL;
    bool u_exact = true, g_ok = true, in_range = true;
    for (int j = 0; j < n; ++j) {
        float ur = hash_uniform_ref(seed_eff, j);
        // uniform is pure integer->float: must be bit-exact.
        if (std::memcmp(&u[j], &ur, sizeof(float)) != 0) u_exact = false;
        if (!(u[j] > 0.0f && u[j] < 1.0f)) in_range = false;
        float gr = -std::log(-std::log(ur));
        if (!approx(g[j], gr, 1e-4f, 1e-4f)) g_ok = false;
    }
    expect(u_exact, "hash_uniform bit-exact vs CPU reference");
    expect(in_range, "uniform values in (0,1)");
    expect(g_ok, "gumbel matches -log(-log(u))");

    cudaFree(d_u); cudaFree(d_g);
}

void test_reduce(CUmodule mod) {
    std::printf("[reduce sum/max/min/argmax]\n");
    const int n = 1000;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-5.f, 5.f);
    std::vector<float> in(n);
    for (auto& v : in) v = dist(rng);
    in[523] = 9.0f;  // unique max

    float* d_in = dmalloc<float>(n);
    float* d_out = dmalloc<float>(4);
    up(d_in, in);
    CUfunction fn = get_fn(mod, "k_reduce");
    void* args[] = {&d_in, (void*)&n, &d_out};
    launch(fn, 1, 256, args);
    std::vector<float> out(4); down(out, d_out);

    double sum = 0; for (float v : in) sum += v;
    float mx = *std::max_element(in.begin(), in.end());
    float mn = *std::min_element(in.begin(), in.end());
    int am = (int)(std::max_element(in.begin(), in.end()) - in.begin());

    expect(approx(out[0], (float)sum, 1e-3f, 1e-2f), "block_sum");
    expect(out[1] == mx, "block_max");
    expect(out[2] == mn, "block_min");
    expect((int)out[3] == am, "block_argmax (lowest-index tie-break)");
    cudaFree(d_in); cudaFree(d_out);
}

void test_argmax_tiebreak(CUmodule mod) {
    std::printf("[argmax tie-break]\n");
    const int n = 64;
    std::vector<float> in(n, 0.f);
    in[10] = 1.0f; in[40] = 1.0f;  // tie -> lowest index 10
    float* d_in = dmalloc<float>(n);
    float* d_out = dmalloc<float>(4);
    up(d_in, in);
    CUfunction fn = get_fn(mod, "k_reduce");
    void* args[] = {&d_in, (void*)&n, &d_out};
    launch(fn, 1, 256, args);
    std::vector<float> out(4); down(out, d_out);
    expect((int)out[3] == 10, "argmax picks lowest index on tie");
    cudaFree(d_in); cudaFree(d_out);
}

void test_scan(CUmodule mod) {
    std::printf("[scan cumsum/cumprod]\n");
    const int n = 600;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    std::uniform_real_distribution<float> distp(0.95f, 1.05f);
    std::vector<float> in(n), inp(n);
    for (int i = 0; i < n; ++i) { in[i] = dist(rng); inp[i] = distp(rng); }

    float* d_in = dmalloc<float>(n);
    float* d_inp = dmalloc<float>(n);
    float* d_csum = dmalloc<float>(n);
    float* d_cprod = dmalloc<float>(n);
    up(d_in, in);
    CUfunction fn = get_fn(mod, "k_scan");
    void* a1[] = {&d_in, (void*)&n, &d_csum, &d_cprod};
    launch(fn, 1, 256, a1);
    std::vector<float> csum(n); down(csum, d_csum);
    // cumsum check
    bool sum_ok = true; double acc = 0;
    for (int i = 0; i < n; ++i) { acc += in[i]; if (!approx(csum[i], (float)acc, 1e-3f, 1e-2f)) sum_ok = false; }
    expect(sum_ok, "cumsum inclusive");

    // cumprod check on a separate stable input
    up(d_inp, inp);
    void* a2[] = {&d_inp, (void*)&n, &d_csum, &d_cprod};
    launch(fn, 1, 256, a2);
    std::vector<float> cprod(n); down(cprod, d_cprod);
    bool prod_ok = true; double pac = 1.0;
    for (int i = 0; i < n; ++i) { pac *= inp[i]; if (!approx(cprod[i], (float)pac, 1e-3f, 1e-3f)) prod_ok = false; }
    expect(prod_ok, "cumprod inclusive");

    cudaFree(d_in); cudaFree(d_inp); cudaFree(d_csum); cudaFree(d_cprod);
}

void test_pivot(CUmodule mod) {
    std::printf("[pivot top-k / top-p / min-p]\n");
    const int n = 500;
    const int k = 17;
    const float p = 0.9f;
    const float min_p = 0.05f;
    std::mt19937 rng(23);
    std::normal_distribution<float> dist(0.f, 2.f);
    std::vector<float> logits(n);
    for (auto& v : logits) v = dist(rng);
    // Make values distinct so top-k boundary is unambiguous.
    for (int i = 0; i < n; ++i) logits[i] += 1e-4f * i;

    // probs = softmax(logits)
    float mx = *std::max_element(logits.begin(), logits.end());
    std::vector<float> prob(n); double Z = 0;
    for (int i = 0; i < n; ++i) { prob[i] = std::exp(logits[i] - mx); Z += prob[i]; }
    for (auto& v : prob) v = (float)(v / Z);

    float* d_val = dmalloc<float>(n);
    float* d_prob = dmalloc<float>(n);
    float* d_out = dmalloc<float>(3);
    unsigned char* d_mk = dmalloc<unsigned char>(n);
    unsigned char* d_mp = dmalloc<unsigned char>(n);
    up(d_val, logits); up(d_prob, prob);
    CUfunction fn = get_fn(mod, "k_pivot");
    void* args[] = {&d_val, &d_prob, (void*)&n, (void*)&k, (void*)&p, (void*)&min_p,
                    &d_out, &d_mk, &d_mp};
    launch(fn, 1, 256, args);
    std::vector<float> out(3); down(out, d_out);
    std::vector<unsigned char> mk(n), mp(n); down(mk, d_mk); down(mp, d_mp);

    // top-k reference: kth largest value; mask should = top-k set.
    std::vector<float> sorted = logits;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    float kth = sorted[k - 1];
    int topk_sel = 0, topk_match = 0;
    for (int i = 0; i < n; ++i) {
        bool ref = logits[i] >= kth;
        if (mk[i]) ++topk_sel;
        if ((bool)mk[i] == ref) ++topk_match;
    }
    expect(topk_sel == k, "top-k mask selects exactly k (distinct values)");
    expect(topk_match == n, "top-k mask == reference top-k set");

    // top-p reference: sort prob desc, accumulate until >= p, last included = tau.
    std::vector<float> ps = prob;
    std::sort(ps.begin(), ps.end(), std::greater<float>());
    double cum = 0; float tau_p = ps.back();
    for (int i = 0; i < n; ++i) { cum += ps[i]; if (cum >= p) { tau_p = ps[i]; break; } }
    int topp_match = 0;
    for (int i = 0; i < n; ++i) {
        bool ref = prob[i] >= tau_p;
        if ((bool)mp[i] == ref) ++topp_match;
    }
    expect(topp_match == n, "top-p mask == nucleus reference set");
    double mass = 0; for (int i = 0; i < n; ++i) if (mp[i]) mass += prob[i];
    expect(mass >= p - 1e-3, "top-p selected mass >= p");

    // min-p: tau = min_p * max(prob)
    float ref_minp = min_p * (*std::max_element(prob.begin(), prob.end()));
    expect(approx(out[2], ref_minp, 1e-4f, 1e-7f), "min-p threshold = min_p * max(prob)");

    cudaFree(d_val); cudaFree(d_prob); cudaFree(d_out); cudaFree(d_mk); cudaFree(d_mp);
}

void test_sort(CUmodule mod) {
    std::printf("[sort desc + indices]\n");
    const int n = 300;  // non-pow2 -> padded to 512
    std::mt19937 rng(31);
    std::uniform_real_distribution<float> dist(-100.f, 100.f);
    std::vector<float> in(n);
    for (auto& v : in) v = dist(rng);
    // distinct
    for (int i = 0; i < n; ++i) in[i] += 1e-3f * i;

    float* d_in = dmalloc<float>(n);
    float* d_ov = dmalloc<float>(n);
    int* d_oi = dmalloc<int>(n);
    up(d_in, in);
    CUfunction fn = get_fn(mod, "k_sort");
    void* args[] = {&d_in, (void*)&n, &d_ov, &d_oi};
    launch(fn, 1, 256, args);
    std::vector<float> ov(n); std::vector<int> oi(n);
    down(ov, d_ov); down(oi, d_oi);

    std::vector<float> ref = in;
    std::sort(ref.begin(), ref.end(), std::greater<float>());
    bool vals_ok = true, desc_ok = true, map_ok = true;
    std::vector<int> seen(n, 0);
    for (int i = 0; i < n; ++i) {
        if (ov[i] != ref[i]) vals_ok = false;
        if (i > 0 && ov[i] > ov[i - 1]) desc_ok = false;
        if (oi[i] < 0 || oi[i] >= n) { map_ok = false; }
        else { seen[oi[i]]++; if (in[oi[i]] != ov[i]) map_ok = false; }
    }
    bool perm_ok = true;
    for (int i = 0; i < n; ++i) if (seen[i] != 1) perm_ok = false;
    expect(vals_ok, "sorted values match std::sort(desc)");
    expect(desc_ok, "output strictly non-increasing");
    expect(map_ok, "out_idx maps in[out_idx[i]] == out_val[i]");
    expect(perm_ok, "out_idx is a permutation of [0,n)");

    cudaFree(d_in); cudaFree(d_ov); cudaFree(d_oi);
}

void test_gather_scatter(CUmodule mod) {
    std::printf("[gather/scatter + OOB safety]\n");
    const int src_len = 50;
    std::vector<float> src(src_len);
    for (int i = 0; i < src_len; ++i) src[i] = (float)(i * 10);
    const int n = 8;
    std::vector<int> idx = {0, 49, -1, 7, 50, 100, 3, -5};  // some invalid
    float* d_src = dmalloc<float>(src_len);
    int* d_idx = dmalloc<int>(n);
    float* d_dst = dmalloc<float>(n);
    up(d_src, src); up(d_idx, idx);
    CUfunction fg = get_fn(mod, "k_gather");
    void* ag[] = {&d_src, (void*)&src_len, &d_idx, &d_dst, (void*)&n};
    launch(fg, 4, 256, ag);
    std::vector<float> dst(n); down(dst, d_dst);
    bool gok = true;
    for (int i = 0; i < n; ++i) {
        float ref = (idx[i] >= 0 && idx[i] < src_len) ? src[idx[i]] : 0.0f;
        if (dst[i] != ref) gok = false;
    }
    expect(gok, "gather fills 0 on invalid index, else src[idx]");

    // gather_row valid + invalid
    const int nrows = 4, ncols = 6;
    std::vector<float> mat(nrows * ncols);
    for (int i = 0; i < nrows * ncols; ++i) mat[i] = (float)i;
    float* d_mat = dmalloc<float>(nrows * ncols);
    float* d_row = dmalloc<float>(ncols);
    up(d_mat, mat);
    CUfunction fr = get_fn(mod, "k_gather_row");
    int good_row = 2;
    void* ar[] = {&d_mat, (void*)&nrows, &good_row, (void*)&ncols, &d_row};
    launch(fr, 2, 128, ar);
    std::vector<float> rowv(ncols); down(rowv, d_row);
    bool rok = true;
    for (int c = 0; c < ncols; ++c) if (rowv[c] != mat[good_row * ncols + c]) rok = false;
    expect(rok, "gather_row copies the requested row");
    int bad_row = 99;
    void* ar2[] = {&d_mat, (void*)&nrows, &bad_row, (void*)&ncols, &d_row};
    launch(fr, 2, 128, ar2);
    down(rowv, d_row);
    bool rz = true; for (int c = 0; c < ncols; ++c) if (rowv[c] != 0.0f) rz = false;
    expect(rz, "gather_row fills 0 on invalid row");

    // scatter_add with dups + OOB; scatter_set
    const int blen = 10;
    std::vector<int> sidx = {0, 0, 3, -1, 10, 5};
    std::vector<float> svals = {1.f, 2.f, 4.f, 9.f, 9.f, 7.f};
    float* d_base = dmalloc<float>(blen);
    int* d_sidx = dmalloc<int>((int)sidx.size());
    float* d_svals = dmalloc<float>((int)svals.size());
    std::vector<float> zeros(blen, 0.f);
    up(d_base, zeros); up(d_sidx, sidx); up(d_svals, svals);
    int sn = (int)sidx.size();
    CUfunction fa = get_fn(mod, "k_scatter_add");
    void* aa[] = {&d_base, (void*)&blen, &d_sidx, &d_svals, (void*)&sn};
    launch(fa, 4, 256, aa);
    std::vector<float> base(blen); down(base, d_base);
    std::vector<float> refbase(blen, 0.f);
    for (int i = 0; i < sn; ++i) if (sidx[i] >= 0 && sidx[i] < blen) refbase[sidx[i]] += svals[i];
    bool aok = true; for (int i = 0; i < blen; ++i) if (!approx(base[i], refbase[i])) aok = false;
    expect(aok, "scatter_add accumulates dups, drops OOB/neg");

    up(d_base, zeros);
    CUfunction fs = get_fn(mod, "k_scatter_set");
    void* as[] = {&d_base, (void*)&blen, &d_sidx, &d_svals, (void*)&sn};
    launch(fs, 4, 256, as);
    down(base, d_base);
    // valid writes present; OOB/neg dropped (base stays 0 where never written)
    expect(base[3] == 4.f && base[5] == 7.f, "scatter_set writes valid lanes");

    cudaFree(d_src); cudaFree(d_idx); cudaFree(d_dst);
    cudaFree(d_mat); cudaFree(d_row);
    cudaFree(d_base); cudaFree(d_sidx); cudaFree(d_svals);
}

void test_map(CUmodule mod) {
    std::printf("[map exp/select]\n");
    const int n = 256;
    std::mt19937 rng(41);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    std::vector<float> a(n), b(n);
    for (int i = 0; i < n; ++i) { a[i] = dist(rng); b[i] = dist(rng); }
    float* d_a = dmalloc<float>(n);
    float* d_b = dmalloc<float>(n);
    float* d_e = dmalloc<float>(n);
    float* d_s = dmalloc<float>(n);
    up(d_a, a); up(d_b, b);
    CUfunction fn = get_fn(mod, "k_map");
    void* args[] = {&d_a, &d_b, (void*)&n, &d_e, &d_s};
    launch(fn, 1, 256, args);
    std::vector<float> e(n), s(n); down(e, d_e); down(s, d_s);
    bool eok = true, sok = true;
    for (int i = 0; i < n; ++i) {
        if (!approx(e[i], std::exp(a[i]), 1e-4f, 1e-5f)) eok = false;
        float ref = (a[i] > b[i]) ? a[i] : b[i];
        if (s[i] != ref) sok = false;
    }
    expect(eok, "map exp(a) matches expf");
    expect(sok, "map select(gt(a,b), a, b) == max(a,b)");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_e); cudaFree(d_s);
}

// Micro-kernels that drive each primitive. Concatenated after the prelude.
const char* kTestKernels = R"TK(
extern "C" __global__ void k_rng(unsigned int seed, float* u_out, float* g_out, int n) {
    unsigned long long se = pie_ir_seed_eff(seed);
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        u_out[j] = pie_ir_hash_uniform(se, j);
        g_out[j] = pie_ir_gumbel(se, j);
    }
}
extern "C" __global__ void k_reduce(const float* in, int n, float* out) {
    float s = pie_ir_block_sum(in, n);
    float mx = pie_ir_block_max(in, n);
    float mn = pie_ir_block_min(in, n);
    int am = pie_ir_block_argmax(in, n);
    if (threadIdx.x == 0) { out[0] = s; out[1] = mx; out[2] = mn; out[3] = (float)am; }
}
extern "C" __global__ void k_scan(const float* in, int n, float* csum, float* cprod) {
    pie_ir_block_inclusive_scan(in, csum, n, 0);
    pie_ir_block_inclusive_scan(in, cprod, n, 1);
}
extern "C" __global__ void k_pivot(const float* val, const float* prob, int n, int k,
                                   float p, float min_p, float* out,
                                   unsigned char* maskk, unsigned char* maskp) {
    float tk = pie_ir_pivot_topk(val, n, k);
    float tp = pie_ir_pivot_topp(prob, n, p);
    float tm = pie_ir_pivot_minp(prob, n, min_p);
    pie_ir_write_ge_mask(val, maskk, n, tk);
    pie_ir_write_ge_mask(prob, maskp, n, tp);
    if (threadIdx.x == 0) { out[0] = tk; out[1] = tp; out[2] = tm; }
}
extern "C" __global__ void k_sort(const float* in, int n, float* ov, int* oi) {
    pie_ir_block_sort_desc(in, ov, oi, n);
}
extern "C" __global__ void k_topk_radix(const float* val, int n, int k, float* out) {
    float t = pie_ir_pivot_topk_radix(val, n, k);
    if (threadIdx.x == 0) out[0] = t;
}
extern "C" __global__ void k_gather(const float* src, int src_len, const int* idx,
                                    float* dst, int n) {
    pie_ir_gather(src, src_len, idx, dst, n);
}
extern "C" __global__ void k_gather_row(const float* src, int nrows, int row, int ncols,
                                        float* dst) {
    pie_ir_gather_row(src, nrows, row, ncols, dst);
}
extern "C" __global__ void k_scatter_add(float* base, int blen, const int* idx,
                                         const float* vals, int n) {
    pie_ir_scatter_add(base, blen, idx, vals, n);
}
extern "C" __global__ void k_scatter_set(float* base, int blen, const int* idx,
                                         const float* vals, int n) {
    pie_ir_scatter_set(base, blen, idx, vals, n);
}
extern "C" __global__ void k_map(const float* a, const float* b, int n,
                                 float* e_out, float* s_out) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int j = g; j < n; j += stride) {
        e_out[j] = pie_ir_exp(a[j]);
        s_out[j] = pie_ir_select(pie_ir_gt(a[j], b[j]), a[j], b[j]);
    }
}
extern "C" __global__ void k_mask_apply(const float* logits, const unsigned int* mask,
                                        int n, float* masked_out, unsigned int* bit_out) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int j = g; j < n; j += stride) {
        masked_out[j] = pie_ir_mask_apply(mask, j, logits[j]);
        bit_out[j]    = pie_ir_mask_bit(mask, j);
    }
}
)TK";

void test_topk_radix(CUmodule mod) {
    std::printf("[top-k radix-select]\n");
    CUfunction fn = get_fn(mod, "k_topk_radix");
    // Large vocab (FlashInfer-scale) + small/medium k; distinct values so the
    // k-th largest is unambiguous.
    struct Case { int n; int k; };
    const Case cases[] = {{500, 17}, {151936, 50}, {151936, 1}, {4096, 4000}, {1000, 1000}};
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.f, 3.f);
    for (const Case& c : cases) {
        std::vector<float> val(c.n);
        for (auto& v : val) v = dist(rng);
        for (int i = 0; i < c.n; ++i) val[i] += 1e-3f * (i % 4096);  // keep distinct-ish
        // dedup exact ties to make k-th unambiguous
        std::sort(val.begin(), val.end());
        for (int i = 1; i < c.n; ++i) if (val[i] <= val[i-1]) val[i] = val[i-1] + 1e-4f;
        std::shuffle(val.begin(), val.end(), rng);

        float* d_val = dmalloc<float>(c.n);
        float* d_out = dmalloc<float>(1);
        up(d_val, val);
        void* args[] = {&d_val, (void*)&c.n, (void*)&c.k, &d_out};
        launch(fn, 1, 256, args);
        std::vector<float> out(1); down(out, d_out);

        std::vector<float> sorted = val;
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        float kth = sorted[c.k - 1];
        // radix returns the exact k-th largest value; count >= tau must be k.
        int cnt = 0;
        for (float v : val) if (v >= out[0]) ++cnt;
        char msg[96];
        std::snprintf(msg, sizeof(msg), "radix top-k n=%d k=%d: tau == k-th largest", c.n, c.k);
        expect(out[0] == kth, msg);
        std::snprintf(msg, sizeof(msg), "radix top-k n=%d k=%d: count(>=tau)==k", c.n, c.k);
        expect(cnt == c.k, msg);
        cudaFree(d_val); cudaFree(d_out);
    }
}

// Cut #2 grammar `mask-apply` (OpKind 0x65). Non-degenerate by construction
// (delta's verify-integrity bar): the unconstrained-argmax token is bit-0
// (DISALLOWED) in the mask, so a passing run MUST exercise the `−∞` path —
// an all-allowed (no-op) mask would fail assertion 3. n is NOT a multiple of
// 32, exercising the tail word + don't-care pad bits.
void test_mask_apply(CUmodule mod) {
    std::printf("[mask-apply grammar constraint]\n");
    const int n = 300;
    const int words = (n + 31) / 32;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-5.f, 5.f);
    std::vector<float> logits(n);
    for (int i = 0; i < n; ++i) logits[i] = dist(rng);
    const int T_MAX = 137;  // dominant unconstrained argmax (will be DISALLOWED)
    const int T_2ND = 200;  // highest ALLOWED → the constrained argmax
    logits[T_MAX] = 100.f;
    logits[T_2ND] = 50.f;

    // All-allowed (init all-1s, matches golf's compose) except a few bit-0s.
    std::vector<unsigned int> mask(words, 0xFFFFFFFFu);
    auto clear_bit = [&](int j) { mask[j >> 5] &= ~(1u << (j & 31)); };
    clear_bit(T_MAX); clear_bit(5); clear_bit(290);

    float* d_logits = dmalloc<float>(n);
    unsigned int* d_mask = dmalloc<unsigned int>(words);
    float* d_masked = dmalloc<float>(n);
    unsigned int* d_bit = dmalloc<unsigned int>(n);
    up(d_logits, logits); up(d_mask, mask);
    CUfunction fn = get_fn(mod, "k_mask_apply");
    void* args[] = {&d_logits, &d_mask, (void*)&n, &d_masked, &d_bit};
    launch(fn, 1, 256, args);
    std::vector<float> masked(n); down(masked, d_masked);
    std::vector<unsigned int> bits(n); down(bits, d_bit);

    const float NEG_INF = -std::numeric_limits<float>::infinity();
    bool apply_ok = true, bit_ok = true;
    for (int j = 0; j < n; ++j) {
        const unsigned int ref = (mask[j >> 5] >> (j & 31)) & 1u;
        if (bits[j] != ref) bit_ok = false;
        if (ref) { if (masked[j] != logits[j]) apply_ok = false; }
        else     { if (masked[j] != NEG_INF)   apply_ok = false; }
    }
    expect(bit_ok, "mask-bit: word-indexed bit matches packed mask");
    expect(apply_ok, "mask-apply: allowed passthrough, disallowed -inf");

    int unconstr = 0, constr = 0;
    for (int j = 1; j < n; ++j) if (logits[j] > logits[unconstr]) unconstr = j;
    for (int j = 1; j < n; ++j) if (masked[j] > masked[constr]) constr = j;
    expect(unconstr == T_MAX, "unconstrained argmax = dominant (disallowed) token");
    expect(masked[T_MAX] == NEG_INF, "the unconstrained-max is forced to -inf");
    expect(constr == T_2ND && constr != unconstr,
           "mask-apply changes argmax -> highest ALLOWED token (-inf actually bit)");
    cudaFree(d_logits); cudaFree(d_mask); cudaFree(d_masked); cudaFree(d_bit);
}

}  // namespace

int main() {
    CU_CHECK(cuInit(0));
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));
    int cc_major = 0, cc_minor = 0;
    CU_CHECK(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU_CHECK(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));

    // Share the primary context so runtime-API allocations and driver-API
    // launches see the same address space.
    CUcontext ctx;
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxSetCurrent(ctx));

    char name[256];
    CU_CHECK(cuDeviceGetName(name, sizeof(name), dev));
    std::printf("Device: %s (sm_%d%d)\n", name, cc_major, cc_minor);

    CUmodule mod = compile_module(kTestKernels, cc_major, cc_minor);

    test_rng(mod);
    test_reduce(mod);
    test_argmax_tiebreak(mod);
    test_scan(mod);
    test_pivot(mod);
    test_topk_radix(mod);
    test_sort(mod);
    test_gather_scatter(mod);
    test_map(mod);
    test_mask_apply(mod);

    CU_CHECK(cuModuleUnload(mod));
    CU_CHECK(cuDevicePrimaryCtxRelease(dev));

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    if (g_failures == 0) std::printf("ALL PASS\n");
    return g_failures == 0 ? 0 : 1;
}
