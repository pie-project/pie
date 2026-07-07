// gdn_checkpoint_test — validates the GDN frozen-checkpointed scan (the core of
// the Metal GDN spec-decode rewind, `ptir-metal-gdn-rewind-design`).
//
// Claim: `gated_delta_net_prefill_checkpointed` emits per-token state checkpoints
// S_0..S_T such that checkpoint m is BIT-IDENTICAL to the recurrent+conv state a
// clean `gated_delta_net_prefill` over just the first m window tokens produces.
// => commit-advance by selecting S[n_acc+1] is LOSSLESS (the rejected drafts'
// state is never committed) — the correctness foundation of GDN spec-decode.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <mlx/mlx.h>

#include "ops/gated_delta.hpp"

namespace mx = mlx::core;
using namespace pie_metal_driver;

namespace {
mx::array randn(const std::vector<int>& shape, float scale = 1.0f) {
    return mx::multiply(mx::random::normal(mx::Shape(shape.begin(), shape.end())),
                        mx::array(scale));
}
double max_abs_diff(const mx::array& a, const mx::array& b) {
    mx::array d = mx::max(mx::abs(mx::subtract(mx::astype(a, mx::float32),
                                               mx::astype(b, mx::float32))));
    mx::eval(d);
    return (double)d.item<float>();
}
}  // namespace

int main() {
    mx::random::seed(1234);

    ops::GdnParams p;
    p.n_heads_k = 2; p.n_heads_v = 2; p.head_k = 4; p.head_v = 4;
    p.conv_kernel = 4; p.norm_eps = 1e-6f;
    const int Kc = p.conv_kernel;
    const int conv_dim = 2 * p.n_heads_k * p.head_k + p.n_heads_v * p.head_v;  // 24
    const int Vh = p.n_heads_v, Kd = p.head_k, Vd = p.head_v, Vdim = Vh * Vd;
    const int T = 5;  // window of 5 tokens (K=4 drafts + seed)

    // Weights (layer-shared).
    mx::array conv_w = randn({conv_dim, Kc}, 0.3f);
    std::optional<mx::array> conv_b = randn({conv_dim}, 0.1f);
    mx::array A_log = randn({Vh}, 0.5f);
    mx::array dt_bias = randn({Vh}, 0.2f);
    mx::array gate_norm_w = mx::add(randn({Vd}, 0.1f), mx::array(1.0f));

    // Window inputs + an arbitrary non-trivial initial (committed) state.
    mx::array mixed_qkv = randn({T, conv_dim});
    mx::array z = randn({T, Vdim});
    mx::array a = randn({T, Vh});
    mx::array b = randn({T, Vh});
    ops::GdnState s0{
        randn({1, Kc, conv_dim}, 0.5f),                    // conv_state
        mx::astype(randn({1, Vh, Kd, Vd}, 0.5f), mx::float32)};  // recurrent_state

    auto rows = [&](const mx::array& x, int m) {
        return mx::slice(x, {0, 0}, {m, x.shape(1)});
    };

    // Checkpointed scan over the full window.
    ops::GdnCheckpointResult ck = ops::gated_delta_net_prefill_checkpointed(
        mixed_qkv, z, a, b, conv_w, conv_b, A_log, dt_bias, gate_norm_w, s0, p);
    mx::eval(ck.recur_ckpts);
    mx::eval(ck.conv_ckpts);

    printf("[gdn-ckpt] T=%d conv_dim=%d Vh=%d Kd=%d Vd=%d — checkpoint vs clean sub-forward:\n",
           T, conv_dim, Vh, Kd, Vd);
    bool ok = true;
    const double TOL = 1e-5;  // fp32 recurrence, same op order ⇒ ~exact
    for (int m = 0; m <= T; ++m) {
        // checkpoint m
        mx::array rc = mx::slice(ck.recur_ckpts, {m, 0, 0, 0}, {m + 1, Vh, Kd, Vd});
        mx::array cc = mx::slice(ck.conv_ckpts, {m, 0, 0}, {m + 1, Kc, conv_dim});
        // clean state after folding the first m window tokens
        mx::array ref_rec = mx::reshape(s0.recurrent_state, {1, Vh, Kd, Vd});
        mx::array ref_conv = mx::reshape(s0.conv_state, {1, Kc, conv_dim});
        if (m > 0) {
            ops::GdnResult r = ops::gated_delta_net_prefill(
                rows(mixed_qkv, m), rows(z, m), rows(a, m), rows(b, m),
                conv_w, conv_b, A_log, dt_bias, gate_norm_w, s0, p);
            ref_rec = r.state.recurrent_state;
            ref_conv = mx::astype(r.state.conv_state, mx::float32);
        }
        double dr = max_abs_diff(rc, ref_rec);
        double dc = max_abs_diff(cc, ref_conv);
        bool m_ok = (dr <= TOL && dc <= TOL);
        ok = ok && m_ok;
        printf("  m=%d: |Δrecur|=%.2e |Δconv|=%.2e  %s\n", m, dr, dc, m_ok ? "OK" : "FAIL");
    }
    printf("%s\n", ok ? "GDN_CHECKPOINT_OK" : "GDN_CHECKPOINT_FAIL");
    return ok ? 0 : 1;
}
