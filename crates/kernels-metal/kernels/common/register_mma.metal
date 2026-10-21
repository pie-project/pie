#include <metal_stdlib>
using namespace metal;

// The register tile's 8x8 simdgroup matrix MMA on operands held in plain
// registers (linear/quant_qmm_sg.metal, linear/moe_sg.metal): lane (fm, fn)
// holds M[fm][fn] and M[fm][fn + 1] of each operand. The simdgroup matrices
// live only inside the call, so the accumulator is a float2 the compiler
// keeps in registers.

namespace qmm_sg {

template <typename T>
__attribute__((always_inline)) inline thread vec<T, 2> &te(thread simdgroup_matrix<T, 8, 8> &m) {
  return reinterpret_cast<thread vec<T, 2> &>(m.thread_elements());
}

// c += a x b.
__attribute__((always_inline)) inline void mma(thread float2 &c, bfloat2 a, bfloat2 b) {
  simdgroup_matrix<bfloat, 8, 8> A, B;
  simdgroup_matrix<float, 8, 8> C, D;
  te(A) = a;
  te(B) = b;
  te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}

} // namespace qmm_sg
