#include "linear/contract.cuh"

namespace pie::linear {

// The plain store as a contraction frame's body, written by hand: each
// product stored as it is. The frames' own reference, as `elemwise/frame.cu`
// is the row frame's; a generated epilogue (`linear::frame`) fires through
// the same frames.
struct Store {
    template <class F>
    __device__ __forceinline__ void operator()(F& f) const {
#pragma unroll
        for (int e = 0; e < F::N; ++e) f.st(0, e, f.ld(0, e));
        f.template flush<0>();
    }
};

}
