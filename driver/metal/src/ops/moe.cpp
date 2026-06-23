#include "ops/moe.hpp"

#include <mlx/mlx.h>

#include "ops/activation.hpp"

namespace mx = mlx::core;

namespace pie_metal_driver::ops {

namespace {

// Top-K largest indices along the last axis -> [.., K] int32 (order undefined,
// which is fine: the combine is a weighted sum).
Tensor topk_indices(const Tensor& scores, int k) {
    const int axis = scores.ndim() - 1;
    const int n = scores.shape(axis);
    if (k >= n) {
        // All experts selected; argsort descending for a stable full ordering.
        Tensor ord = mx::argsort(mx::negative(scores), axis);
        return mx::astype(ord, mx::int32);
    }
    // argpartition puts the (n-k)..(n-1) positions as the K largest.
    Tensor part = mx::argpartition(scores, n - k, axis);
    mx::Shape start(scores.ndim(), 0);
    mx::Shape stop(scores.shape().begin(), scores.shape().end());
    start[axis] = n - k;
    return mx::astype(mx::slice(part, start, stop), mx::int32);
}

}  // namespace

MoeRouting moe_route(const Tensor& router_logits,
                     const MoeParams& params,
                     const std::optional<Tensor>& correction_bias,
                     const std::optional<Tensor>& per_expert_scale) {
    const int E = params.num_experts;
    const int K = params.experts_per_token;
    const int N = router_logits.shape(0);

    Tensor logits = mx::astype(router_logits, mx::float32);  // [N, E]

    // Combine weights come from softmax / sigmoid of the raw logits.
    Tensor probs = (params.gate == MoeGate::Softmax)
                       ? mx::softmax(logits, /*axis=*/-1, /*precise=*/true)
                       : mx::sigmoid(logits);  // [N, E]

    // Selection score may add an aux-loss-free correction bias (sigmoid router).
    Tensor sel = probs;
    if (correction_bias.has_value()) {
        sel = mx::add(sel, mx::reshape(mx::astype(*correction_bias, mx::float32),
                                       {1, E}));
    }

    // Grouped routing (DeepSeek): keep only experts in the top `topk_group`
    // groups, scored by the sum of their top-2 experts.
    if (params.n_group > 1) {
        const int G = params.n_group;
        const int gsz = E / G;
        Tensor selg = mx::reshape(sel, {N, G, gsz});                 // [N,G,gsz]
        Tensor top2 = mx::topk(selg, 2, /*axis=*/-1);                // [N,G,2]
        Tensor gscore = mx::sum(top2, /*axis=*/-1, /*keepdims=*/false);  // [N,G]
        Tensor gidx = topk_indices(gscore, params.topk_group);      // [N,topk_group]
        Tensor gmask = mx::zeros({N, G}, mx::float32);
        gmask = mx::put_along_axis(
            gmask, mx::astype(gidx, mx::uint32),
            mx::ones({N, params.topk_group}, mx::float32), /*axis=*/-1);
        // Broadcast the per-group mask over experts and mask out dropped groups.
        Tensor emask = mx::reshape(
            mx::broadcast_to(mx::reshape(gmask, {N, G, 1}), {N, G, gsz}),
            {N, E});  // [N,E] in {0,1}
        sel = mx::where(mx::greater(emask, mx::array(0.0f)), sel,
                        mx::full({N, E}, -1e30f, mx::float32));
    }

    Tensor indices = topk_indices(sel, K);                  // [N, K] int32
    Tensor weights = mx::take_along_axis(
        probs, mx::astype(indices, mx::uint32), /*axis=*/-1);  // [N, K]

    if (params.norm_topk) {
        Tensor denom = mx::sum(weights, /*axis=*/-1, /*keepdims=*/true);
        weights = mx::divide(weights, mx::maximum(denom, mx::array(1e-20f)));
    }
    if (params.routed_scaling != 1.0f) {
        weights = mx::multiply(weights, mx::array(params.routed_scaling));
    }
    if (per_expert_scale.has_value()) {
        Tensor pes = mx::astype(*per_expert_scale, mx::float32);  // [E]
        weights = mx::multiply(
            weights, mx::take(pes, mx::astype(indices, mx::uint32), /*axis=*/0));
    }

    return MoeRouting{indices, mx::astype(weights, mx::float32)};
}

Tensor moe_experts(const Tensor& x,
                   const MoeRouting& routing,
                   const Tensor& gate_up_w,
                   const Tensor& down_w,
                   MoeAct act) {
    const int N = x.shape(0);
    const int hidden = x.shape(1);
    const int K = routing.indices.shape(1);
    const int inter = down_w.shape(2);  // down_w: [E, hidden, inter]

    Tensor idx = mx::astype(routing.indices, mx::uint32);  // [N, K]

    // [N, 1, 1, hidden] so gather_mm broadcasts the per-token row across the K
    // selected experts (batch dims [N,1] broadcast against rhs_indices [N,K]).
    Tensor xe = mx::reshape(x, {N, 1, 1, hidden});

    // Expert weights are HF row-major [E, out, in]; gather_mm wants [E, in, out].
    Tensor guT = mx::swapaxes(gate_up_w, -1, -2);  // [E, hidden, 2*inter]
    Tensor dwT = mx::swapaxes(down_w, -1, -2);     // [E, inter, hidden]

    Tensor gu = mx::gather_mm(xe, guT, std::nullopt, idx);  // [N,K,1,2*inter]
    // Split fused gate/up along the last axis.
    Tensor g = mx::slice(gu, {0, 0, 0, 0}, {N, K, 1, inter});
    Tensor u = mx::slice(gu, {0, 0, 0, inter}, {N, K, 1, 2 * inter});
    Tensor h = (act == MoeAct::Gelu) ? geglu(g, u, /*tanh_approx=*/true)
                                     : swiglu(g, u);          // [N,K,1,inter]
    Tensor o = mx::gather_mm(h, dwT, std::nullopt, idx);      // [N,K,1,hidden]
    o = mx::squeeze(o, 2);                                    // [N,K,hidden]

    Tensor w = mx::reshape(routing.weights, {N, K, 1});       // [N,K,1] f32
    o = mx::sum(mx::multiply(o, mx::astype(w, o.dtype())), /*axis=*/1,
                /*keepdims=*/false);                          // [N,hidden]
    return o;
}

Tensor moe_ffn(const Tensor& x,
               const Tensor& router_logits,
               const Tensor& gate_up_w,
               const Tensor& down_w,
               const MoeParams& params,
               const std::optional<Tensor>& correction_bias,
               const std::optional<Tensor>& per_expert_scale) {
    MoeRouting routing =
        moe_route(router_logits, params, correction_bias, per_expert_scale);
    return moe_experts(x, routing, gate_up_w, down_w, params.act);
}

Tensor shared_expert(const Tensor& x,
                     const Tensor& gate_w,
                     const Tensor& up_w,
                     const Tensor& down_w,
                     const Tensor& shared_gate) {
    // Dense SwiGLU FFN scaled by a per-token sigmoid gate.
    Tensor g = mx::matmul(x, mx::swapaxes(gate_w, -1, -2));   // [N, inter]
    Tensor u = mx::matmul(x, mx::swapaxes(up_w, -1, -2));     // [N, inter]
    Tensor h = swiglu(g, u);                                  // [N, inter]
    Tensor out = mx::matmul(h, mx::swapaxes(down_w, -1, -2)); // [N, hidden]
    Tensor gate = mx::sigmoid(
        mx::matmul(x, mx::swapaxes(shared_gate, -1, -2)));    // [N, 1]
    return mx::multiply(out, gate);
}

}  // namespace pie_metal_driver::ops
