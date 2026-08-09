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

void situ_bf16(
    const void* gate, const void* up, void* y,
    int num_elements, float beta, float linear_beta, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const int grid = (num_elements + BLOCK - 1) / BLOCK;
    device::situ<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(gate),
        static_cast<const device::bf16*>(up),
        static_cast<device::bf16*>(y),
        num_elements, beta, linear_beta);
}

void swiglu_clamp_bf16(
    const void* gate, const void* up, void* y,
    int num_elements, float limit, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const int grid = (num_elements + BLOCK - 1) / BLOCK;
    device::swiglu_clamp<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(gate),
        static_cast<const device::bf16*>(up),
        static_cast<device::bf16*>(y),
        num_elements, limit);
}

void swiglu_bf16(
    const void* gate, const void* up, void* y,
    int num_elements, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const int grid = (num_elements + BLOCK - 1) / BLOCK;
    device::swiglu<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(gate),
        static_cast<const device::bf16*>(up),
        static_cast<device::bf16*>(y),
        num_elements);
}

// `dim3(rows, tiles)`, rows on x -- which is `LaunchRule::ElementwiseRows`
// and the transpose of what this launcher used to write. The kernel's index
// lines moved with it; see `mlp/swiglu.cuh`.
void gpt_oss_glu_strided_bf16(
    const void* gate, const void* up, void* y,
    int rows, int cols, int in_stride, int out_stride, cudaStream_t stream,
    float limit, float alpha)
{
    if (rows <= 0 || cols <= 0) return;
    constexpr int BLOCK = 256;
    dim3 grid(rows, (cols + BLOCK - 1) / BLOCK);
    device::gpt_oss_glu_strided<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(gate),
        static_cast<const device::bf16*>(up),
        static_cast<device::bf16*>(y),
        cols, in_stride, out_stride, limit, alpha);
}

void gpt_oss_glu_bf16(
    const void* gate, const void* up, void* y,
    int num_elements, cudaStream_t stream,
    float limit, float alpha, void* y_fp16)
{
    constexpr int BLOCK = 256;
    const int grid = (num_elements + BLOCK - 1) / BLOCK;
    device::gpt_oss_glu<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(gate),
        static_cast<const device::bf16*>(up),
        static_cast<device::bf16*>(y),
        static_cast<device::f16*>(y_fp16),
        num_elements, limit, alpha);
}

void geglu_tanh_bf16(
    const void* gate, const void* up, void* y,
    int num_elements, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const int grid = (num_elements + BLOCK - 1) / BLOCK;
    device::geglu_tanh<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(gate),
        static_cast<const device::bf16*>(up),
        static_cast<device::bf16*>(y),
        num_elements);
}

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

void chunked_swiglu_clamp_bf16(
    const void* packed, void* y, int N, int I, float limit,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const dim3 grid(N, (I + BLOCK - 1) / BLOCK);
    device::chunked_swiglu_clamp<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(packed),
        static_cast<device::bf16*>(y), I, limit);
}

void chunked_swiglu_strided_bf16(
    const void* packed, void* y, int N, int I, int row_stride, cudaStream_t stream)
{
    if (N <= 0 || I <= 0) return;
    if (row_stride == 2 * I) {
        chunked_swiglu_bf16(packed, y, N, I, stream);
        return;
    }
    constexpr int BLOCK = 128;
    if (row_stride & 1) {
        dim3 grid(N, (I + BLOCK - 1) / BLOCK);
        device::chunked_swiglu_strided<device::bf16><<<grid, BLOCK, 0, stream>>>(
            static_cast<const device::bf16*>(packed),
            static_cast<device::bf16*>(y),
            I, row_stride);
        return;
    }
    dim3 grid(N, ((I + 1) / 2 + BLOCK - 1) / BLOCK);
    device::chunked_swiglu_strided_vec2<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(packed),
        static_cast<device::bf16*>(y),
        N, I, row_stride);
}

void chunked_geglu_tanh_bf16(
    const void* packed, void* y, int N, int I, cudaStream_t stream,
    bool gate_second)
{
    if (N <= 0 || I <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(N, (I + BLOCK - 1) / BLOCK);
    const auto* p = static_cast<const device::bf16*>(packed);
    auto* yp = static_cast<device::bf16*>(y);
    if (gate_second) {
        device::chunked_geglu_tanh_gate_second<device::bf16>
            <<<grid, BLOCK, 0, stream>>>(p, yp, I);
    } else {
        device::chunked_geglu_tanh<device::bf16><<<grid, BLOCK, 0, stream>>>(p, yp, I);
    }
}

void relu2_bf16(
    const void* x, void* y, int num_elements, cudaStream_t stream)
{
    if (num_elements <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = (num_elements + BLOCK - 1) / BLOCK;
    device::relu2<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(x),
        static_cast<device::bf16*>(y),
        num_elements);
}

void sigmoid_scalar_gate_add_bf16(
    void* out, const void* x, const void* scalar_gate, int N, int H,
    cudaStream_t stream)
{
    sigmoid_scalar_gate_strided_add_bf16(
        out, x, scalar_gate, N, H, /*stride=*/1, stream);
}

void sigmoid_scalar_gate_strided_add_bf16(
    void* out, const void* x, const void* scalar_gate,
    int N, int H, int stride, cudaStream_t stream)
{
    if (N <= 0 || H <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(N, (H + BLOCK - 1) / BLOCK);
    device::sigmoid_scalar_gate_add<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<device::bf16*>(out),
        static_cast<const device::bf16*>(x),
        static_cast<const device::bf16*>(scalar_gate),
        H, stride);
}

void sigmoid_dot_scalar_gate_add_bf16(
    const void* x, const void* gate_w, void* out, const void* y,
    int N, int H, cudaStream_t stream)
{
    if (N <= 0 || H <= 0) return;
    constexpr int BLOCK = 256;
    device::sigmoid_dot_scalar_gate_add<device::bf16><<<
        N, BLOCK, (BLOCK / 32) * sizeof(float), stream>>>(
        static_cast<const device::bf16*>(x),
        static_cast<const device::bf16*>(gate_w),
        static_cast<device::bf16*>(out),
        static_cast<const device::bf16*>(y),
        H);
}

void chunked_situ_bf16(
    const void* packed, void* y, int N, int I, float beta, float linear_beta,
    bool gate_second, cudaStream_t stream)
{
    if (N <= 0 || I <= 0) return;
    constexpr int BLOCK = 128;
    const auto* p = static_cast<const device::bf16*>(packed);
    auto* yp = static_cast<device::bf16*>(y);
    dim3 grid(N, (I + BLOCK - 1) / BLOCK);
    if (gate_second) {
        device::chunked_situ_gate_second<device::bf16><<<grid, BLOCK, 0, stream>>>(
            p, yp, I, beta, linear_beta);
    } else {
        device::chunked_situ<device::bf16><<<grid, BLOCK, 0, stream>>>(
            p, yp, I, beta, linear_beta);
    }
}

}  // namespace pie_cuda_driver::kernels::mlp
