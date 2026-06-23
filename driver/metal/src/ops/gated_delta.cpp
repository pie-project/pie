#include "ops/gated_delta.hpp"

#include <mlx/mlx.h>

#include "linear_state_cache.hpp"

namespace mx = mlx::core;

namespace pie_metal_driver::ops {

namespace {

// L2-normalise along the last axis: x / sqrt(sum(x^2) + eps).
Tensor l2norm_last(const Tensor& x, float eps) {
    Tensor denom = mx::sqrt(
        mx::add(mx::sum(mx::square(x), /*axis=*/-1, /*keepdims=*/true),
                mx::array(eps)));
    return mx::divide(x, denom);
}

// softplus(x) = log(1 + exp(x)), numerically stable.
Tensor softplus(const Tensor& x) {
    return mx::add(mx::maximum(x, mx::array(0.0f)),
                   mx::log1p(mx::exp(mx::negative(mx::abs(x)))));
}

}  // namespace

GdnResult gated_delta_net_decode(const Tensor& mixed_qkv,
                                 const Tensor& z,
                                 const Tensor& a,
                                 const Tensor& b,
                                 const Tensor& conv_w,
                                 const std::optional<Tensor>& conv_b,
                                 const Tensor& A_log,
                                 const Tensor& dt_bias,
                                 const Tensor& gate_norm_w,
                                 const GdnState& state_in,
                                 const GdnParams& p) {
    const int R = mixed_qkv.shape(0);
    const int Kh = p.n_heads_k, Vh = p.n_heads_v;
    const int Kd = p.head_k, Vd = p.head_v;
    const int Kc = p.conv_kernel;
    const int conv_dim = 2 * Kh * Kd + Vh * Vd;
    const int rep = Vh / Kh;

    // Everything computes in fp32 to match the cuda reference (bf16 inputs are
    // widened; recurrent state is fp32).
    Tensor xqkv = mx::astype(mixed_qkv, mx::float32);      // [R, conv_dim]

    // ── Causal depthwise conv1d update (decode, T=1) + silu ────────────────
    // conv_state holds the last Kc input rows (oldest first). Shift in the new
    // token, convolve the full Kc-window ending at it.
    Tensor cstate = mx::astype(state_in.conv_state, mx::float32);  // [R,Kc,conv_dim]
    Tensor new_cstate = mx::concatenate(
        {mx::slice(cstate, {0, 1, 0}, {R, Kc, conv_dim}),
         mx::reshape(xqkv, {R, 1, conv_dim})},
        /*axis=*/1);                                       // [R, Kc, conv_dim]
    // y[r,c] = silu( sum_j new_cstate[r,j,c] * conv_w[c,j] + conv_b[c] )
    Tensor wct = mx::reshape(mx::swapaxes(mx::astype(conv_w, mx::float32), 0, 1),
                             {1, Kc, conv_dim});           // [1, Kc, conv_dim]
    Tensor y = mx::sum(mx::multiply(new_cstate, wct), /*axis=*/1,
                       /*keepdims=*/false);                // [R, conv_dim]
    if (conv_b.has_value()) {
        y = mx::add(y, mx::reshape(mx::astype(*conv_b, mx::float32),
                                   {1, conv_dim}));
    }
    y = mx::multiply(y, mx::sigmoid(y));                   // silu

    // ── Split q/k/v and prep ───────────────────────────────────────────────
    Tensor q = mx::slice(y, {0, 0}, {R, Kh * Kd});
    Tensor k = mx::slice(y, {0, Kh * Kd}, {R, 2 * Kh * Kd});
    Tensor v = mx::slice(y, {0, 2 * Kh * Kd}, {R, conv_dim});
    q = mx::reshape(q, {R, Kh, Kd});
    k = mx::reshape(k, {R, Kh, Kd});
    v = mx::reshape(v, {R, Vh, Vd});

    // L2-normalise q,k along the head dim; pre-scale q by 1/sqrt(K_d).
    q = mx::multiply(l2norm_last(q, p.norm_eps),
                     mx::array(1.0f / std::sqrt(static_cast<float>(Kd))));
    k = l2norm_last(k, p.norm_eps);
    // GQA: repeat q/k heads to V_h.
    if (rep > 1) {
        q = mx::repeat(q, rep, /*axis=*/1);                // [R, Vh, Kd]
        k = mx::repeat(k, rep, /*axis=*/1);
    }

    // ── g / beta gating ────────────────────────────────────────────────────
    // g = -exp(A_log) * softplus(a + dt_bias);  beta = sigmoid(b)
    Tensor Al = mx::reshape(mx::astype(A_log, mx::float32), {1, Vh});
    Tensor dtb = mx::reshape(mx::astype(dt_bias, mx::float32), {1, Vh});
    Tensor af = mx::astype(a, mx::float32);                // [R, Vh]
    Tensor g = mx::multiply(mx::negative(mx::exp(Al)),
                            softplus(mx::add(af, dtb)));   // [R, Vh]
    Tensor beta = mx::sigmoid(mx::astype(b, mx::float32)); // [R, Vh]

    // ── Recurrent gated-delta step (per request, per V-head) ───────────────
    Tensor state = mx::astype(state_in.recurrent_state, mx::float32);  // [R,Vh,Kd,Vd]
    state = mx::multiply(state, mx::reshape(mx::exp(g), {R, Vh, 1, 1}));
    Tensor k4 = mx::reshape(k, {R, Vh, Kd, 1});
    Tensor q4 = mx::reshape(q, {R, Vh, Kd, 1});
    Tensor kv_mem = mx::sum(mx::multiply(state, k4), /*axis=*/2,
                            /*keepdims=*/false);           // [R, Vh, Vd]
    Tensor delta = mx::multiply(mx::subtract(v, kv_mem),
                                mx::reshape(beta, {R, Vh, 1}));
    Tensor outer = mx::multiply(k4, mx::reshape(delta, {R, Vh, 1, Vd}));
    state = mx::add(state, outer);                         // [R, Vh, Kd, Vd]
    Tensor out = mx::sum(mx::multiply(state, q4), /*axis=*/2,
                         /*keepdims=*/false);              // [R, Vh, Vd]

    // ── RMSNormGated(out, z): weight * rmsnorm(out) * silu(z) ──────────────
    Tensor zr = mx::reshape(mx::astype(z, mx::float32), {R, Vh, Vd});
    Tensor ms = mx::mean(mx::square(out), /*axis=*/-1, /*keepdims=*/true);
    Tensor outhat = mx::multiply(out, mx::rsqrt(mx::add(ms, mx::array(p.norm_eps))));
    Tensor gnw = mx::reshape(mx::astype(gate_norm_w, mx::float32), {1, 1, Vd});
    Tensor normed = mx::multiply(mx::multiply(outhat, gnw),
                                 mx::multiply(zr, mx::sigmoid(zr)));  // [R,Vh,Vd]
    Tensor result = mx::reshape(normed, {R, Vh * Vd});

    GdnState state_out{mx::astype(new_cstate, state_in.conv_state.dtype()),
                       state};
    return GdnResult{result, state_out};
}

Tensor gated_delta_net(const Tensor& mixed_qkv,
                       const Tensor& z,
                       const Tensor& a,
                       const Tensor& b,
                       const Tensor& conv_w,
                       const std::optional<Tensor>& conv_b,
                       const Tensor& A_log,
                       const Tensor& dt_bias,
                       const Tensor& gate_norm_w,
                       LinearStateCache& cache,
                       int lin_layer,
                       const Tensor& slot_ids,
                       const GdnParams& params) {
    GdnState state_in{cache.gather_conv_state(lin_layer, slot_ids),
                      cache.gather_recurrent_state(lin_layer, slot_ids)};
    GdnResult r =
        gated_delta_net_decode(mixed_qkv, z, a, b, conv_w, conv_b, A_log,
                               dt_bias, gate_norm_w, state_in, params);
    cache.write_conv_state(lin_layer, slot_ids, r.state.conv_state);
    cache.write_recurrent_state(lin_layer, slot_ids, r.state.recurrent_state);
    return r.output;
}

}  // namespace pie_metal_driver::ops
