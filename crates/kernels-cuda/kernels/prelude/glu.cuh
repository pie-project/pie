#pragma once

// The GLUs pie's MLPs use (1 swiglu, 2 clamped, 3 clamped with alpha, 4
// GeGLU's tanh form, 5 SiTU), as the tcgen05 epilogue and the
// register-dequant MoE kernels spell them. Fast exp and divide: a precise
// division would pace a fused epilogue (an overflowing exp divides to 0,
// its limit).

namespace pie::glu_fn {

template <int kGlu>
__device__ __forceinline__ float glu(float g, float u, float limit, float galpha) {
    if constexpr (kGlu == 1) {
        return __fdividef(g, 1.f + __expf(-g)) * u;
    } else if constexpr (kGlu == 4) {
        // GeGLU, gelu's tanh form: 0.5 g (1 + tanh(sqrt(2 / pi) (g + 0.044715 g^3))) u.
        float t;
        const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
        asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(inner));
        return 0.5f * g * (1.f + t) * u;
    } else if constexpr (kGlu == 5) {
        // SiTU (Kimi-K3's `mlp_situ`): beta tanh(g / beta) sigmoid(g), times
        // the up value, capped as cap tanh(u / cap) under a cap (`galpha`,
        // 0 for none). `limit` holds beta.
        float t;
        asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(__fdividef(g, limit)));
        if (galpha > 0.f) {
            float c;
            asm("tanh.approx.f32 %0, %1;" : "=f"(c) : "f"(__fdividef(u, galpha)));
            u = galpha * c;
        }
        return __fdividef(limit * t, 1.f + __expf(-g)) * u;
    } else {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
        if constexpr (kGlu == 2) {
            return __fdividef(g, 1.f + __expf(-g)) * u;
        } else {
            return (u + 1.f) * __fdividef(g, 1.f + __expf(-galpha * g));
        }
    }
}

}  // namespace pie::glu_fn
