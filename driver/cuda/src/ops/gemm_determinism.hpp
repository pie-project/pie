#pragma once

// The opt-in reproducibility switch for the dense bf16 GEMM path.
//
// `DenseGemmTuner` in ops/gemm.cpp picks each shape's kernel by racing the
// candidates on the wall clock, and keeps the winner in `dense_gemm.txt` for
// the life of the machine. Candidates within `kGemmTacticMargin` of one
// another are separated by whatever else the GPU was doing during the probe,
// and a different kernel accumulates K in a different order, so the last bit
// of a logit -- and with it a greedy decode's choice between two close tokens
// -- can depend on when the tuner happened to run. It is the same failure the
// in-place split-K ban a few hundred lines up in gemm.cpp is written against:
// that ban fixed the reduction order *inside* a kernel and left the choice
// *of* kernel racing.
//
// For serving that is the right trade; the fastest kernel is worth a last bit.
// For a parity run, a determinism check, or a bisect against a reference the
// answer has to be a function of the input alone, and there is no amount of
// care at the call site that recovers it. `PIE_GEMM_DETERMINISTIC=1` is how a
// run says which of the two it is.
//
// Everything that wants the answer asks `dense_gemm_deterministic()`, so there
// is exactly one line to look at to see whether the knob is still wired to the
// environment. A knob whose `getenv` is later replaced by a constant leaves
// its documentation true and its behaviour dead, which is why
// tests/gemm_determinism_test.cpp re-execs itself with the variable set and
// unset and asserts this function follows it.

#include <cstdlib>

namespace pie_cuda_driver::ops {

inline constexpr char kDenseGemmDeterministicEnv[] = "PIE_GEMM_DETERMINISTIC";

// Set, non-empty, and not "0". Split out from the lookup so the spelling rules
// can be checked without an environment to mutate.
constexpr bool dense_gemm_deterministic_value(const char* v) {
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

// Cached: a process's environment does not change under it, and this sits in
// front of every dense bf16 GEMM.
inline bool dense_gemm_deterministic() {
    static const bool on =
        dense_gemm_deterministic_value(std::getenv(kDenseGemmDeterministicEnv));
    return on;
}

}  // namespace pie_cuda_driver::ops
