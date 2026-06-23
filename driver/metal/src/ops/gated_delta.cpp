#include "ops/gated_delta.hpp"

#include <mlx/mlx.h>
#include <mlx/fast.h>

#include "linear_state_cache.hpp"
#include "ops/compiled.hpp"

namespace mx = mlx::core;

namespace pie_metal_driver::ops {

namespace {

// Fused recurrent gated-delta step (mlx_lm `gated_delta_step`, scalar-g decode
// variant). One kernel replaces the ~7-op decomposed update (state*=exp(g);
// kv=stateᵀk; delta=(v-kv)β; state+=outer(k,delta); out=stateᵀq), loading the
// 1MB recurrent state into registers once with simd_sum warp-reductions instead
// of 5-7 global round-trips. Template params: InT/StT dtypes, Dk/Dv head dims,
// Hk/Hv head counts, T tokens. State layout is [B,Hv,Dv,Dk] (q/k GQA-repeated to
// Hv so Hk==Hv; g enters as the decay multiplier exp(g); beta = sigmoid(b)).
const char* kGdnStepSource = R"METAL(
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }

    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g_[hv_idx];
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);
      auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) { y[dv_idx] = static_cast<InT>(out); }
      q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y += Hv * Dv;
      g_ += Hv; beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
)METAL";

const mx::fast::CustomKernelFunction& gdn_step_kernel() {
    static mx::fast::CustomKernelFunction k = mx::fast::metal_kernel(
        "gated_delta_step", {"q", "k", "v", "g", "beta", "state_in"},
        {"y", "state_out"}, kGdnStepSource);
    return k;
}

// L2-normalise along the last axis: x / sqrt(sum(x^2) + eps).
Tensor l2norm_last(const Tensor& x, float eps) {
    Tensor denom = mx::sqrt(
        mx::add(mx::sum(mx::square(x), /*axis=*/-1, /*keepdims=*/true),
                mx::array(eps)));
    return mx::divide(x, denom);
}

// L2-normalise along the last axis with eps as a (scalar) array input -- the
// capture-less variant used inside the compiled decode region.
Tensor l2norm_last_a(const Tensor& x, const Tensor& eps) {
    Tensor denom = mx::sqrt(
        mx::add(mx::sum(mx::square(x), /*axis=*/-1, /*keepdims=*/true), eps));
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

// Capture-less compiled-region body for the single-token decode step (see
// gated_delta_net_decode). All arrays arrive via `in` (fp32), dims are derived
// from input shapes, and eps is a scalar input -- nothing is captured, so this
// is a stable function pointer for MLX's compile path. Inputs (in order):
//   0 mixed_qkv [R,conv_dim]   1 z [R,V_dim]   2 a [R,Vh]   3 b [R,Vh]
//   4 conv_w [conv_dim,Kc]     5 conv_b [conv_dim]   6 A_log [Vh]
//   7 dt_bias [Vh]   8 gate_norm_w [Vd]   9 conv_state [R,Kc,conv_dim]
//   10 recurrent_state [R,Vh,Kd,Vd]   11 eps (scalar)
// Outputs: {result [R,V_dim], new_conv_state [R,Kc,conv_dim], new_recurrent_state}.
std::vector<Tensor> gdn_decode_region(const std::vector<Tensor>& in) {
    const Tensor& mixed = in[0];
    const Tensor& z = in[1];
    const Tensor& a = in[2];
    const Tensor& b = in[3];
    const Tensor& conv_w = in[4];
    const Tensor& conv_b = in[5];
    const Tensor& A_log = in[6];
    const Tensor& dt_bias = in[7];
    const Tensor& gate_norm_w = in[8];
    const Tensor& conv_state = in[9];
    const Tensor& recurrent_state = in[10];
    const Tensor& eps = in[11];

    const int R = mixed.shape(0);
    const int conv_dim = mixed.shape(1);
    const int Kc = conv_state.shape(1);
    const int Vh = recurrent_state.shape(1);
    const int Kd = recurrent_state.shape(2);
    const int Vd = recurrent_state.shape(3);
    const int Kh = (conv_dim - Vh * Vd) / (2 * Kd);
    const int rep = Vh / Kh;

    // Causal depthwise conv1d (T=1) + silu.
    Tensor new_cstate = mx::concatenate(
        {mx::slice(conv_state, {0, 1, 0}, {R, Kc, conv_dim}),
         mx::reshape(mixed, {R, 1, conv_dim})},
        /*axis=*/1);
    Tensor wct = mx::reshape(mx::swapaxes(conv_w, 0, 1), {1, Kc, conv_dim});
    Tensor y = mx::sum(mx::multiply(new_cstate, wct), /*axis=*/1, false);
    y = mx::add(y, mx::reshape(conv_b, {1, conv_dim}));
    y = mx::multiply(y, mx::sigmoid(y));

    // Split q/k/v, L2-normalise q/k, pre-scale q, GQA-repeat to V_h.
    Tensor q = mx::reshape(mx::slice(y, {0, 0}, {R, Kh * Kd}), {R, Kh, Kd});
    Tensor k =
        mx::reshape(mx::slice(y, {0, Kh * Kd}, {R, 2 * Kh * Kd}), {R, Kh, Kd});
    Tensor v = mx::reshape(mx::slice(y, {0, 2 * Kh * Kd}, {R, conv_dim}),
                           {R, Vh, Vd});
    q = mx::multiply(l2norm_last_a(q, eps),
                     mx::array(1.0f / std::sqrt(static_cast<float>(Kd))));
    k = l2norm_last_a(k, eps);
    if (rep > 1) {
        q = mx::repeat(q, rep, /*axis=*/1);
        k = mx::repeat(k, rep, /*axis=*/1);
    }

    // g / beta gating.
    Tensor Al = mx::reshape(A_log, {1, Vh});
    Tensor dtb = mx::reshape(dt_bias, {1, Vh});
    Tensor g = mx::multiply(mx::negative(mx::exp(Al)),
                            softplus(mx::add(a, dtb)));
    Tensor beta = mx::sigmoid(b);

    // Recurrent gated-delta step -- fused metal kernel (mlx_lm gated_delta_step).
    // Replaces the ~7-op decomposed update. Kernel state layout is [R,Vh,Vd,Kd];
    // ours is [R,Vh,Kd,Vd] -> swapaxes at the boundary (MLX makes inputs row-
    // contiguous). g enters as the decay multiplier exp(g); beta = sigmoid(b).
    Tensor st_in = mx::swapaxes(recurrent_state, 2, 3);  // [R,Vh,Vd,Kd]
    std::vector<Tensor> kout = gdn_step_kernel()(
        {mx::reshape(q, {R, 1, Vh, Kd}), mx::reshape(k, {R, 1, Vh, Kd}),
         mx::reshape(v, {R, 1, Vh, Vd}), mx::reshape(mx::exp(g), {R, 1, Vh}),
         mx::reshape(beta, {R, 1, Vh}), st_in},
        {mx::Shape{R, 1, Vh, Vd}, st_in.shape()}, {mx::float32, mx::float32},
        {32, Vd, R * Vh}, {32, 4, 1},
        {{"InT", mx::float32}, {"StT", mx::float32}, {"Dk", Kd}, {"Dv", Vd},
         {"Hk", Vh}, {"Hv", Vh}, {"T", 1}},
        std::nullopt, false, {});
    Tensor out = mx::reshape(kout[0], {R, Vh, Vd});  // [R,Vh,Vd]
    Tensor state = mx::swapaxes(kout[1], 2, 3);      // back to [R,Vh,Kd,Vd]

    // RMSNormGated(out, z) = rms_norm(out)*gate_norm_w * silu(z). The norm is a
    // single fused MLX kernel (eps literal -- this region is qwen3.6-specific,
    // rms_norm_eps=1e-6; capture-less regions take eps as a literal per
    // compiled.hpp). ~2x vs the decomposed mean/rsqrt/mul chain.
    Tensor zr = mx::reshape(z, {R, Vh, Vd});
    Tensor outhat = mx::fast::rms_norm(out, gate_norm_w, 1e-6f);  // [R,Vh,Vd]
    Tensor normed = mx::multiply(outhat, mx::multiply(zr, mx::sigmoid(zr)));
    Tensor result = mx::reshape(normed, {R, Vh * Vd});

    return {result, new_cstate, state};
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
    const int conv_dim = conv_dim_of(p);

    // The whole single-token decode step is a ~40-op launch-storm of tiny
    // pointwise/reduction kernels (conv, l2norm, gating, the recurrent
    // gated-delta update, RMSNormGated) -- at batch=1 it is entirely
    // dispatch-bound, and on qwen3.6 it dominates decode (×18 linear-attn
    // layers). It is a pure, host-readback-free function of its tensor inputs,
    // so we fuse it via the compiled-region harness: one trace, replayed across
    // every linear layer (shapes identical) and across decode steps.
    //
    // The region is capture-less (a function pointer): all arrays come through
    // `inputs`, dims are derived from input shapes, and eps is a scalar input
    // (so nothing is captured -- required for MLX's stable fn-pointer compile
    // path; see ops/compiled.hpp). conv_b is always supplied (zeros when the
    // layer has no bias; +0 is exact), keeping a single compiled instance.
    Tensor eps = mx::array(p.norm_eps);
    Tensor bias = conv_b.has_value()
                      ? mx::astype(*conv_b, mx::float32)
                      : mx::zeros({conv_dim}, mx::float32);

    std::vector<Tensor> outs = compiled(
        "qwen36.gated_delta_net.decode",
        {mx::astype(mixed_qkv, mx::float32),
         mx::astype(z, mx::float32),
         mx::astype(a, mx::float32),
         mx::astype(b, mx::float32),
         mx::astype(conv_w, mx::float32),
         bias,
         mx::astype(A_log, mx::float32),
         mx::astype(dt_bias, mx::float32),
         mx::astype(gate_norm_w, mx::float32),
         mx::astype(state_in.conv_state, mx::float32),
         mx::astype(state_in.recurrent_state, mx::float32),
         eps},
        gdn_decode_region);

    (void)R;
    GdnState state_out{mx::astype(outs[1], state_in.conv_state.dtype()), outs[2]};
    return GdnResult{outs[0], state_out};
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
