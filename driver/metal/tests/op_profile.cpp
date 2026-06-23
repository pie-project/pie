// Per-op decode-forward attribution profiler for qwen3.6 (delta-owned).
//
// Purpose (roadmap perf, measure-first): attribute our single-token decode
// FORWARD GPU cost across op categories, and isolate our serving-design PAGING
// overhead (paged-KV take-gather + per-key mask) vs the contiguous SDPA that
// mlx-lm runs single-stream. Times each op IN ISOLATION at the real qwen3.6
// 0.8B shapes (no model load needed; synthetic weights — quantized-matmul cost
// is shape/group-bound, not value-bound), then multiplies by layer counts.
//
// Honest-attribution buckets (per manager): paging | other-closeable | structural.
//
// Method: each op is built K times from distinct inputs into one eval so the
// per-eval commit+wait floor is amortized; report median-of-iters of total/K,
// plus the bare-eval floor for reference. Run on a COOL, GPU-EXCLUSIVE box.
//
// Build: target `op_profile` (PIE_METAL_BUILD_TESTS=ON). No args.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "ops/ops.hpp"

namespace mx = mlx::core;
using pie_metal_driver::Tensor;
namespace ops = pie_metal_driver::ops;

namespace {

using clk = std::chrono::high_resolution_clock;
constexpr int GROUP = 32;   // qwen3.6 g32 checkpoint
constexpr int BITS = 4;

// qwen3.6 0.8B geometry (config.json text_config).
constexpr int H = 1024;       // hidden
constexpr int I = 3584;       // intermediate (MLP)
constexpr int VOCAB = 248320;
constexpr int N_LAYERS = 24;
constexpr int N_FULL = 6;     // full_attention_interval=4 -> 6 full
constexpr int N_LIN = 18;     // 18 gated-deltanet layers
// full-attn
constexpr int NQ = 8, NKV = 2, HD = 256;
// linear-attn (GDN)
constexpr int KH = 16, KD = 128, VH = 16, VD = 128, CONVK = 4;
constexpr int CONV_DIM = 2 * KH * KD + VH * VD;  // 6144
constexpr int V_DIM = VH * VD;                    // 2048
// decode context length to attend over (KV cache length).
constexpr int CTX = 256;
constexpr int PAGE = 32;

mx::array randf(mx::Shape shape) {
    return mx::random::normal(shape, mx::bfloat16);
}

// Quantized weight triple for an [out,in] linear (MLX affine packing).
struct QW { Tensor w, s, b; };
QW make_qw(int out, int in) {
    auto q = mx::quantize(mx::random::normal(mx::Shape{out, in}, mx::float32), GROUP, BITS);
    return {mx::astype(q[0], q[0].dtype()), q[1], q[2]};
}

// Time `make()` (returns one output array) K-batched, ITERS times; return the
// median per-single-op ms (total/K). make(i) must build a DISTINCT graph so MLX
// can't CSE the K copies into one.
double time_op(int K, int iters, const std::function<Tensor(int)>& make) {
    // warmup
    { std::vector<Tensor> v; for (int k = 0; k < K; ++k) v.push_back(make(k)); mx::eval(v); }
    std::vector<double> samples;
    for (int it = 0; it < iters; ++it) {
        std::vector<Tensor> v; v.reserve(K);
        for (int k = 0; k < K; ++k) v.push_back(make(k * 1000 + it));
        auto t0 = clk::now();
        mx::eval(v);
        auto t1 = clk::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / K);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];  // median
}

// bare-eval floor (1 trivial op / eval)
double eval_floor() {
    std::vector<double> s;
    Tensor a = mx::array({0}, {1}, mx::int32); mx::eval(a);
    for (int i = 0; i < 64; ++i) {
        auto t0 = clk::now();
        Tensor x = mx::add(a, mx::array({i}, {1}, mx::int32)); mx::eval(x);
        auto t1 = clk::now();
        s.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(s.begin(), s.end());
    return s[s.size() / 2];
}

}  // namespace

int main() {
    std::printf("qwen3.6 0.8B per-op decode-forward attribution (N=1 token, ctx=%d)\n", CTX);
    std::printf("  group_size=%d bits=%d  | %d full-attn + %d GDN layers\n\n", GROUP, BITS, N_FULL, N_LIN);
    const double floor = eval_floor();
    std::printf("  [bare-eval floor: %.4f ms]\n\n", floor);

    const int K = 16, IT = 40;
    struct Row { std::string name; int count; double per; };
    std::vector<Row> rows;
    auto add = [&](const std::string& n, int count, double per) {
        rows.push_back({n, count, per});
    };

    // ---- quantized linear projections (the GEMV memory-bound core) ----
    auto bench_qlin = [&](const std::string& nm, int out, int in, int count) {
        QW q = make_qw(out, in);
        double per = time_op(K, IT, [&](int i) {
            Tensor x = randf({1, in});
            return ops::quantized_linear(q.w, q.s, q.b, x, GROUP, BITS);
        });
        add(nm, count, per);
    };
    // full-attn projections (x6 layers)
    bench_qlin("full.q_proj  [1024->4096]", NQ * 2 * HD, H, N_FULL);
    bench_qlin("full.k_proj  [1024->512]",  NKV * HD, H, N_FULL);
    bench_qlin("full.v_proj  [1024->512]",  NKV * HD, H, N_FULL);
    bench_qlin("full.o_proj  [2048->1024]", H, NQ * HD, N_FULL);
    // GDN projections (x18 layers)
    bench_qlin("gdn.in_qkv   [1024->6144]", CONV_DIM, H, N_LIN);
    bench_qlin("gdn.in_z     [1024->2048]", V_DIM, H, N_LIN);
    bench_qlin("gdn.out_proj [2048->1024]", H, V_DIM, N_LIN);
    // MLP (x24 layers)
    bench_qlin("mlp.gate     [1024->3584]", I, H, N_LAYERS);
    bench_qlin("mlp.up       [1024->3584]", I, H, N_LAYERS);
    bench_qlin("mlp.down     [3584->1024]", H, I, N_LAYERS);
    // lm_head (x1)
    bench_qlin("lm_head      [1024->248320]", VOCAB, H, 1);

    // ---- norms (rms_norm) ----
    {
        Tensor w = randf({H});
        double per = time_op(K, IT, [&](int i) {
            return ops::rms_norm(randf({1, H}), w, 1e-6f, true);
        });
        // per layer: attn_norm + ffn_norm = 2; +final = 1 -> count ~ 2*24+1
        add("rms_norm     [1,1024]", 2 * N_LAYERS + 1, per);
    }

    // ---- rope (Q+K) on full-attn layers ----
    {
        Tensor pos = mx::array({CTX}, {1}, mx::int32);
        ops::RopeParams rp; rp.theta = 1000000.0f;
        double perQ = time_op(K, IT, [&](int i) {
            return ops::rope(randf({1, NQ, HD}), pos, HD, rp);
        });
        add("rope Q+K     [full]", N_FULL * 2, perQ);  // approx (K is smaller; upper bound)
    }

    // ---- swiglu activation (x24) ----
    {
        double per = time_op(K, IT, [&](int i) {
            return ops::swiglu(randf({1, I}), randf({1, I}));
        });
        add("swiglu       [1,3584]", N_LAYERS, per);
    }

    // ---- PAGING: paged_attention_decode vs contiguous sdpa (the tax) ----
    const int n_pages = (CTX + PAGE - 1) / PAGE;
    auto make_pages = [&]() {
        return randf({n_pages + 2, PAGE, NKV, HD});  // physical page buffer
    };
    Tensor k_cache = make_pages(), v_cache = make_pages();
    Tensor page_table = mx::arange(0, n_pages, mx::int32);
    Tensor last_page_len = mx::array({CTX - (n_pages - 1) * PAGE}, {1}, mx::int32);
    ops::AttnParams ap; ap.scale = 0.0f; ap.n_heads = NQ; ap.n_kv_heads = NKV; ap.head_dim = HD;

    double per_paged = time_op(K, IT, [&](int i) {
        Tensor q = randf({1, NQ, HD});
        return ops::paged_attention_decode(q, k_cache, v_cache, page_table,
                                           last_page_len, PAGE, ap);
    });
    add("PAGED attn_decode [full]", N_FULL, per_paged);

    // contiguous SDPA baseline = what mlx-lm pays (no gather, no per-key mask).
    Tensor kc = randf({CTX, NKV, HD}), vc = randf({CTX, NKV, HD});
    double per_sdpa = time_op(K, IT, [&](int i) {
        Tensor q = randf({1, NQ, HD});
        return ops::sdpa(q, kc, vc, ap);
    });
    add("contig sdpa (mlx-lm peer) [full]", N_FULL, per_sdpa);

    // ---- GDN decode step (x18) ----
    {
        Tensor conv_w = randf({CONV_DIM, CONVK});
        Tensor A_log = randf({VH}), dt_bias = randf({VH}), gnorm = randf({VD});
        ops::GdnParams gp; gp.n_heads_k = KH; gp.n_heads_v = VH; gp.head_k = KD;
        gp.head_v = VD; gp.conv_kernel = CONVK; gp.norm_eps = 1e-6f;
        double per = time_op(K, IT, [&](int i) {
            ops::GdnState st{ randf({1, CONVK, CONV_DIM}),
                              mx::astype(randf({1, VH, KD, VD}), mx::float32) };
            ops::GdnResult r = ops::gated_delta_net_decode(
                randf({1, CONV_DIM}),
                randf({1, V_DIM}), randf({1, VH}), randf({1, VH}),
                conv_w, std::nullopt, A_log, dt_bias, gnorm, st, gp);
            return r.output;
        });
        add("GDN decode step [lin]", N_LIN, per);
    }

    // ---- embed gather (1 row, quantized table dequant) ----
    {
        QW emb = make_qw(VOCAB, H);
        Tensor full = mx::dequantize(emb.w, emb.s, emb.b, GROUP, BITS);  // not the path; cost upper bound via take
        (void)full;
        double per = time_op(K, IT, [&](int i) {
            Tensor ids = mx::array({i % VOCAB}, {1}, mx::int32);
            // dequant-gather: gather the row's packed nibbles + scales, dequant.
            Tensor wq = mx::take(emb.w, ids, 0);
            Tensor sc = mx::take(emb.s, ids, 0);
            Tensor bi = mx::take(emb.b, ids, 0);
            return mx::dequantize(wq, sc, bi, GROUP, BITS);
        });
        add("embed dequant-gather [1 row]", 1, per);
    }

    // ---- report ----
    std::printf("%-38s %6s %10s %12s\n", "op", "xN", "per-op ms", "total ms");
    std::printf("%s\n", std::string(70, '-').c_str());
    double grand = 0;
    for (auto& r : rows) {
        double tot = r.per * r.count;
        grand += tot;
        std::printf("%-38s %6d %10.4f %12.4f\n", r.name.c_str(), r.count, r.per, tot);
    }
    std::printf("%s\n", std::string(70, '-').c_str());
    std::printf("%-38s %6s %10s %12.4f\n", "SUM (per-op isolated)", "", "", grand);
    std::printf("\nNote: isolated per-op sums exclude kernel-launch interleave &\n"
                "      include a per-batch floor share; treat as relative attribution.\n");
    return 0;
}
