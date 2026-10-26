#include "elemwise/skeleton.cuh"

namespace pie::elemwise {

// A residual norm as a row skeleton's body. Parameters: 0 the residual x,
// 1 the stream y, 2 the weight; results: 0 the normed row, 1 the stream
// y + x, stored in place. The squares are summed a pair at a time, as a
// generated body sums them.
template <bool WEIGHT_PLUS_ONE>
struct ResidualAddRmsnorm {
    float eps;

    template <class F>
    __device__ __forceinline__ void operator()(F& f) const {
        float h[F::N];
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < F::N; e += 2) {
            h[e] = f.st(1, e, f.ld(1, e) + f.ld(0, e));
            h[e + 1] = f.st(1, e + 1, f.ld(1, e + 1) + f.ld(0, e + 1));
            if (f.valid(e)) ss += h[e] * h[e] + h[e + 1] * h[e + 1];
        }
        f.template flush<1>();
        const float r = rsqrtf(f.reduce_sum(ss) / static_cast<float>(f.width) + eps);
#pragma unroll
        for (int e = 0; e < F::N; ++e) {
            float w = f.ld(2, e);
            if constexpr (WEIGHT_PLUS_ONE) w += 1.f;
            f.st(0, e, h[e] * r * w);
        }
        f.template flush<0>();
    }
};

}
