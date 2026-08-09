// The launchers, and nothing else.
//
// Every `__global__` this file used to hold now lives in `mlp/swiglu.cuh`,
// which the JIT compiles at run time and which this file includes so the
// ahead-of-time archive keeps exactly ONE definition of each. What is left
// here is the half NVRTC cannot have: host functions that take a
// `cudaStream_t`, and the run-time choices -- `I > 10000`, the parity of
// `row_stride` -- that decide WHICH kernel to fire.
//
// Those choices are why this file did not disappear the way `norm`'s did.
// A `LaunchRule` states a grid; it does not state a predicate over an
// operand's value, and the vectorised kernels below are chosen by exactly
// such a predicate. Until a row can say that, both halves exist and this is
// the half that says it.
//
// The scalar layer and the fixed-width integer names come out of the
// prelude, through the device header: NVRTC has no CUDA device headers, and
// `mlp/swiglu.cuh` is meant to compile under both it and nvcc.
#include "pie_device.cuh"
#include "mlp/swiglu.cuh"
#include "mlp/swiglu.hpp"

namespace pie_cuda_driver::kernels::mlp {
// # Thirteen launchers went from here, and the file is the family's clearest
// # case of a live JOB with a dead LAUNCHER
//
// `situ_bf16`, `swiglu_clamp_bf16`, `swiglu_bf16`, `gpt_oss_glu_bf16`,
// `geglu_tanh_bf16`, `chunked_swiglu_clamp_bf16`, `chunked_geglu_tanh_bf16`,
// `relu2_bf16`, `sigmoid_dot_scalar_gate_add_bf16` and `chunked_situ_bf16`
// all still have rows in `table::mlp` AND in `families::mlp`, and every one
// of those rows is in `device.rs`'s `JIT_DISPATCHED`: NVRTC compiles them out
// of `mlp/swiglu.cuh` and the generated shim forwards to none of them. §41's
// audit measured exactly that -- of twelve `table::mlp` rows the shim carries
// two. What went is the ahead-of-time launcher, which is §10.10 step 5.
//
// Three had no row anywhere and no caller either:
//
//   * `gpt_oss_glu_strided_bf16` and `chunked_swiglu_strided_bf16` -- strided
//     forks nothing ever asked for. The second was also the ONLY C++ caller
//     of `chunked_swiglu_bf16`, so deleting it frees that row from a hold
//     that was itself unreachable: orphaned at one remove, §41's shape.
//   * `sigmoid_scalar_gate_add_bf16` and its `_strided_` callee -- their rows
//     were removed once already, for the reason `families::mlp`'s header
//     states: nothing writes those symbols into a plan.
//
// `chunked_swiglu_bf16` and `sigmoid_gate_inplace_bf16` stay: both are shim
// roots, and the first's `I > 10000` scalar/vector fork is a host comparison
// no `Source` can name.

void sigmoid_gate_inplace_bf16(
    void* x, const void* gate, int n, cudaStream_t stream)
{
    if (n <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = (n + BLOCK - 1) / BLOCK;
    device::sigmoid_gate_inplace<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<device::bf16*>(x),
        static_cast<const device::bf16*>(gate),
        n);
}

// The vectorised fork. `I > 10000` picks the scalar kernel because at that
// width the tail predicate costs less than the halved block count saves;
// below it the packed pair wins. That comparison is this file's, and it is
// why the `_vec2` kernels carry no row.
void chunked_swiglu_bf16(
    const void* packed, void* y, int N, int I, cudaStream_t stream,
    bool gate_second)
{
    if (N <= 0 || I <= 0) return;
    constexpr int BLOCK = 128;
    const auto* p = static_cast<const device::bf16*>(packed);
    auto* yp = static_cast<device::bf16*>(y);
    if (I > 10000) {
        dim3 grid(N, (I + BLOCK - 1) / BLOCK);
        if (gate_second) {
            device::chunked_swiglu_gate_second<device::bf16>
                <<<grid, BLOCK, 0, stream>>>(p, yp, I);
        } else {
            device::chunked_swiglu<device::bf16><<<grid, BLOCK, 0, stream>>>(p, yp, I);
        }
        return;
    }
    dim3 grid(N, ((I + 1) / 2 + BLOCK - 1) / BLOCK);
    if (gate_second) {
        device::chunked_swiglu_vec2_gate_second<device::bf16>
            <<<grid, BLOCK, 0, stream>>>(p, yp, N, I);
    } else {
        device::chunked_swiglu_vec2<device::bf16>
            <<<grid, BLOCK, 0, stream>>>(p, yp, N, I);
    }
}


}  // namespace pie_cuda_driver::kernels::mlp
