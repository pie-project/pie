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

int conv_dim_of(const GdnParams& p) {
    return 2 * p.n_heads_k * p.head_k + p.n_heads_v * p.head_v;
}

// Depthwise causal conv (+silu) over a [Kc-1+T, conv_dim] padded stack whose
// first Kc-1 rows are the left context. Returns [T, conv_dim].
Tensor conv_causal_silu(const Tensor& padded, const Tensor& conv_w,
                        const std::optional<Tensor>& conv_b, int Kc) {
    const int conv_dim = padded.shape(1);
    const int T = padded.shape(0) - (Kc - 1);
    Tensor wct = mx::reshape(mx::swapaxes(mx::astype(conv_w, mx::float32), 0, 1),
                             {1, Kc, conv_dim});  // [1, Kc, conv_dim]
    std::vector<Tensor> windows;
    windows.reserve(Kc);
    for (int j = 0; j < Kc; ++j) {
        windows.push_back(mx::reshape(
            mx::slice(padded, {j, 0}, {j + T, conv_dim}), {T, 1, conv_dim}));
    }
    Tensor win = mx::concatenate(windows, /*axis=*/1);  // [T, Kc, conv_dim]
    Tensor y = mx::sum(mx::multiply(win, wct), /*axis=*/1, /*keepdims=*/false);
    if (conv_b.has_value()) {
        y = mx::add(y, mx::reshape(mx::astype(*conv_b, mx::float32),
                                   {1, conv_dim}));
    }
    return mx::multiply(y, mx::sigmoid(y));  // silu
}

// From conv output `y` [B, conv_dim], produce normalised q/k [B,Vh,Kd], v
// [B,Vh,Vd], and gating g/beta [B,Vh] (q/k already GQA-repeated to V_h).
struct Prepped {
    Tensor q, k, v, g, beta;
};
Prepped prep_qkvg(const Tensor& y, const Tensor& a, const Tensor& b,
                  const Tensor& A_log, const Tensor& dt_bias,
                  const GdnParams& p) {
    const int B = y.shape(0);
    const int Kh = p.n_heads_k, Vh = p.n_heads_v, Kd = p.head_k, Vd = p.head_v;
    const int rep = Vh / Kh;
    Tensor q = mx::reshape(mx::slice(y, {0, 0}, {B, Kh * Kd}), {B, Kh, Kd});
    Tensor k = mx::reshape(mx::slice(y, {0, Kh * Kd}, {B, 2 * Kh * Kd}),
                           {B, Kh, Kd});
    Tensor v = mx::reshape(mx::slice(y, {0, 2 * Kh * Kd}, {B, y.shape(1)}),
                           {B, Vh, Vd});
    q = mx::multiply(l2norm_last(q, p.norm_eps),
                     mx::array(1.0f / std::sqrt(static_cast<float>(Kd))));
    k = l2norm_last(k, p.norm_eps);
    if (rep > 1) {
        q = mx::repeat(q, rep, /*axis=*/1);
        k = mx::repeat(k, rep, /*axis=*/1);
    }
    Tensor Al = mx::reshape(mx::astype(A_log, mx::float32), {1, Vh});
    Tensor dtb = mx::reshape(mx::astype(dt_bias, mx::float32), {1, Vh});
    Tensor g = mx::multiply(
        mx::negative(mx::exp(Al)),
        softplus(mx::add(mx::astype(a, mx::float32), dtb)));  // [B, Vh]
    Tensor beta = mx::sigmoid(mx::astype(b, mx::float32));     // [B, Vh]
    return Prepped{q, k, v, g, beta};
}

// One recurrent gated-delta step over a batch [B, Vh, ...]. Returns
// {new_state, out [B,Vh,Vd]}. Mirrors the inline decode recurrence.
std::pair<Tensor, Tensor> recurrent_step(Tensor state, const Tensor& q,
                                         const Tensor& k, const Tensor& v,
                                         const Tensor& g, const Tensor& beta) {
    const int B = state.shape(0), Vh = state.shape(1), Kd = state.shape(2),
              Vd = state.shape(3);
    state = mx::multiply(state, mx::reshape(mx::exp(g), {B, Vh, 1, 1}));
    Tensor k4 = mx::reshape(k, {B, Vh, Kd, 1});
    Tensor q4 = mx::reshape(q, {B, Vh, Kd, 1});
    Tensor kv_mem = mx::sum(mx::multiply(state, k4), /*axis=*/2, false);
    Tensor delta = mx::multiply(mx::subtract(v, kv_mem),
                                mx::reshape(beta, {B, Vh, 1}));
    Tensor outer = mx::multiply(k4, mx::reshape(delta, {B, Vh, 1, Vd}));
    state = mx::add(state, outer);
    Tensor out = mx::sum(mx::multiply(state, q4), /*axis=*/2, false);
    return {state, out};
}

// RMSNormGated(out, z): weight * rmsnorm(out) * silu(z) over V_d. Returns
// [B, Vh*Vd].
Tensor rmsnorm_gated(const Tensor& out, const Tensor& z,
                     const Tensor& gate_norm_w, float eps) {
    const int B = out.shape(0), Vh = out.shape(1), Vd = out.shape(2);
    Tensor zr = mx::reshape(mx::astype(z, mx::float32), {B, Vh, Vd});
    Tensor ms = mx::mean(mx::square(out), /*axis=*/-1, /*keepdims=*/true);
    Tensor outhat = mx::multiply(out, mx::rsqrt(mx::add(ms, mx::array(eps))));
    Tensor gnw = mx::reshape(mx::astype(gate_norm_w, mx::float32), {1, 1, Vd});
    Tensor normed = mx::multiply(mx::multiply(outhat, gnw),
                                 mx::multiply(zr, mx::sigmoid(zr)));
    return mx::reshape(normed, {B, Vh * Vd});
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

GdnResult gated_delta_net_prefill(const Tensor& mixed_qkv,
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
    // Single request, T tokens. Causal conv over the T tokens using the cached
    // conv_state as left context, then a sequential recurrent scan (the
    // recurrence has a strict per-token state dependency, so T steps).
    const int T = mixed_qkv.shape(0);
    const int Kc = p.conv_kernel;
    const int conv_dim = conv_dim_of(p);
    const int Vh = p.n_heads_v, Kd = p.head_k, Vd = p.head_v;

    Tensor xseq = mx::astype(mixed_qkv, mx::float32);            // [T, conv_dim]
    Tensor cstate = mx::reshape(mx::astype(state_in.conv_state, mx::float32),
                                {Kc, conv_dim});
    // Left context = the Kc-1 newest cached rows; convolve [ctx ++ xseq].
    Tensor ctx = mx::slice(cstate, {1, 0}, {Kc, conv_dim});     // [Kc-1, conv_dim]
    Tensor padded = mx::concatenate({ctx, xseq}, /*axis=*/0);   // [Kc-1+T, conv_dim]
    Tensor y = conv_causal_silu(padded, conv_w, conv_b, Kc);    // [T, conv_dim]
    // New conv_state = the last Kc input rows of (cstate ++ xseq).
    Tensor full = mx::concatenate({cstate, xseq}, /*axis=*/0);  // [Kc+T, conv_dim]
    Tensor new_cstate = mx::slice(full, {T, 0}, {T + Kc, conv_dim});  // [Kc, conv_dim]

    Prepped pr = prep_qkvg(y, a, b, A_log, dt_bias, p);

    // Sequential scan over the T tokens (batch dim 1).
    Tensor state = mx::reshape(mx::astype(state_in.recurrent_state, mx::float32),
                               {1, Vh, Kd, Vd});
    std::vector<Tensor> outs;
    outs.reserve(T);
    for (int t = 0; t < T; ++t) {
        Tensor qt = mx::reshape(mx::slice(pr.q, {t, 0, 0}, {t + 1, Vh, Kd}),
                                {1, Vh, Kd});
        Tensor kt = mx::reshape(mx::slice(pr.k, {t, 0, 0}, {t + 1, Vh, Kd}),
                                {1, Vh, Kd});
        Tensor vt = mx::reshape(mx::slice(pr.v, {t, 0, 0}, {t + 1, Vh, Vd}),
                                {1, Vh, Vd});
        Tensor gt = mx::reshape(mx::slice(pr.g, {t, 0}, {t + 1, Vh}), {1, Vh});
        Tensor bt = mx::reshape(mx::slice(pr.beta, {t, 0}, {t + 1, Vh}), {1, Vh});
        auto [ns, ot] = recurrent_step(state, qt, kt, vt, gt, bt);
        state = ns;
        outs.push_back(ot);  // [1, Vh, Vd]
    }
    Tensor out = mx::concatenate(outs, /*axis=*/0);             // [T, Vh, Vd]
    Tensor result = rmsnorm_gated(out, z, gate_norm_w, p.norm_eps);  // [T, V_dim]

    GdnState state_out{
        mx::astype(mx::reshape(new_cstate, {1, Kc, conv_dim}),
                   state_in.conv_state.dtype()),
        mx::reshape(state, {1, Vh, Kd, Vd})};
    return GdnResult{result, state_out};
}

Tensor gated_delta_net_varlen(const Tensor& mixed_qkv,
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
                              const std::vector<int>& qo_indptr,
                              const GdnParams& params) {
    // Per-request dispatch over a ragged batch. Each request's T tokens are
    // scanned sequentially with its own slot state; outputs are emitted in
    // token order to match qo_indptr -> [N_total, V_dim].
    const int R = static_cast<int>(qo_indptr.size()) - 1;

    Tensor sids = mx::astype(slot_ids, mx::int32);
    mx::eval(sids);
    std::vector<int> sid_host(sids.data<int>(), sids.data<int>() + sids.size());

    std::vector<Tensor> outputs;
    outputs.reserve(R);
    for (int r = 0; r < R; ++r) {
        const int t0 = qo_indptr[r], t1 = qo_indptr[r + 1];
        if (t1 <= t0) continue;
        Tensor slot = mx::reshape(
            mx::slice(sids, {r}, {r + 1}), {1});  // single slot id
        GdnState st{cache.gather_conv_state(lin_layer, slot),
                    cache.gather_recurrent_state(lin_layer, slot)};
        auto rows = [&](const Tensor& x) {
            return mx::slice(x, {t0, 0}, {t1, x.shape(1)});
        };
        GdnResult res = gated_delta_net_prefill(
            rows(mixed_qkv), rows(z), rows(a), rows(b), conv_w, conv_b, A_log,
            dt_bias, gate_norm_w, st, params);
        cache.write_conv_state(lin_layer, slot, res.state.conv_state);
        cache.write_recurrent_state(lin_layer, slot, res.state.recurrent_state);
        outputs.push_back(res.output);  // [T, V_dim]
    }
    return mx::concatenate(outputs, /*axis=*/0);  // [N_total, V_dim]
}

}  // namespace pie_metal_driver::ops
