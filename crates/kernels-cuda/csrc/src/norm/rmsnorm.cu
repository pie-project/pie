// The kernels live in the header; this file is the entry points that still
// launch them ahead of time, plus the one host predicate -- `rmsnorm_vec8_ok`
// -- that decides between a scalar kernel and its vectorised twin.
//
// That predicate is why most of this file survives. It reads pointer
// ALIGNMENT, and no `LaunchRule` can see an address, so the launchers pick
// the vectorised path and the JIT rows in `kernels_cuda_new::families::norm`
// fire the scalar one. `rmsnorm.cuh` records which four kernels the rows name
// and why the other eight are unreachable from a row.
#include "pie_device.cuh"
#include "norm/rmsnorm.cuh"
#include "norm/rmsnorm.hpp"

#include <type_traits>
#include "quant/dequant_wna16.hpp"

#include <cstdint>
#include <cstdlib>

namespace pie_cuda_driver::kernels::norm {

namespace {

// True when every row of a [num_rows, hidden] bf16 view starts on a 16-byte
// boundary and is a whole number of 8-element vectors.
inline bool rmsnorm_vec8_ok(const void* x, const void* y, const void* weight,
                            int hidden, int x_row_stride, int y_row_stride)
{
    auto aligned = [](const void* p) {
        return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0;
    };
    return hidden % 8 == 0 && x_row_stride % 8 == 0 && y_row_stride % 8 == 0 &&
           aligned(x) && aligned(y) && aligned(weight);
}

}  // namespace

void rmsnorm_bf16(
    const void* x, const void* weight, void* y,
    int num_rows, int hidden, float eps, cudaStream_t stream)
{
    rmsnorm_strided_bf16(
        x, weight, y, num_rows, hidden, hidden, hidden, eps, stream);
}

// RMSNorm that also writes an fp16 copy of its output, for a consumer that
// wants fp16 -- the MXFP4 decode GEMV. Falls back to the plain launcher plus a
// cast when the vectorised path does not apply, so the caller never has to ask
// whether its shape qualifies.
//
// That fallback is a SECOND launch (`quant::bf16_to_fp16`), which is why no
// row names this entry point: a row is one kernel, and this one is two
// whenever the rows are unaligned.
void rmsnorm_bf16_with_fp16(
    const void* x, const void* weight, void* y, void* y_fp16,
    int num_rows, int hidden, float eps, cudaStream_t stream)
{
    if (y_fp16 == nullptr) {
        rmsnorm_bf16(x, weight, y, num_rows, hidden, eps, stream);
        return;
    }
    if (!rmsnorm_vec8_ok(x, y, weight, hidden, hidden, hidden)) {
        rmsnorm_bf16(x, weight, y, num_rows, hidden, eps, stream);
        kernels::quant::bf16_to_fp16(y, y_fp16,
                            static_cast<device::usize>(num_rows) * hidden,
                            stream);
        return;
    }
    constexpr int VBLOCK = 512;
    dim3 grid(num_rows);
    device::rmsnorm_vec8<VBLOCK, /*WEIGHT_PLUS_ONE=*/false, /*EMIT_FP16=*/true>
        <<<grid, VBLOCK, 0, stream>>>(
            static_cast<const device::bf16*>(x),
            static_cast<const device::bf16*>(weight),
            static_cast<device::bf16*>(y),
            static_cast<device::f16*>(y_fp16),
            hidden, hidden, hidden, eps);
}

void rmsnorm_strided_bf16(
    const void* x, const void* weight, void* y,
    int num_rows, int hidden, int x_row_stride, int y_row_stride,
    float eps, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    if (rmsnorm_vec8_ok(x, y, weight, hidden, x_row_stride, y_row_stride)) {
        constexpr int VBLOCK = 512;
        device::rmsnorm_vec8<VBLOCK, /*WEIGHT_PLUS_ONE=*/false>
            <<<grid, VBLOCK, 0, stream>>>(
                static_cast<const device::bf16*>(x),
                static_cast<const device::bf16*>(weight),
                static_cast<device::bf16*>(y), nullptr,
                hidden, x_row_stride, y_row_stride, eps);
        return;
    }
    dim3 block(BLOCK);
    device::rmsnorm<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const device::bf16*>(x),
        static_cast<const device::bf16*>(weight),
        static_cast<device::bf16*>(y),
        hidden, x_row_stride, y_row_stride, eps);
}

void residual_add_rmsnorm_bf16(
    void* hidden,
    const void* residual,
    const void* weight,
    void* norm_out,
    int num_rows,
    int hidden_size,
    float eps,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    if (rmsnorm_vec8_ok(hidden, norm_out, weight, hidden_size,
                        hidden_size, hidden_size) &&
        (reinterpret_cast<std::uintptr_t>(residual) % 16) == 0) {
        constexpr int VBLOCK = 512;
        device::residual_add_rmsnorm_vec8<VBLOCK><<<grid, VBLOCK, 0, stream>>>(
            static_cast<device::bf16*>(hidden),
            static_cast<const device::bf16*>(residual),
            static_cast<const device::bf16*>(weight),
            static_cast<device::bf16*>(norm_out),
            hidden_size, eps);
        return;
    }
    dim3 block(BLOCK);
    device::residual_add_rmsnorm<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
        static_cast<device::bf16*>(hidden),
        static_cast<const device::bf16*>(residual),
        static_cast<const device::bf16*>(weight),
        static_cast<device::bf16*>(norm_out),
        hidden_size, eps);
}

void residual_add_scale_rmsnorm_bf16(
    void* hidden,
    const void* residual,
    float scale,
    const void* next_weight,
    void* norm_out,
    int num_rows,
    int hidden_size,
    float eps,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    device::residual_add_scale_rmsnorm<device::bf16, BLOCK>
        <<<grid, block, 0, stream>>>(
            static_cast<device::bf16*>(hidden),
            static_cast<const device::bf16*>(residual),
            scale,
            static_cast<const device::bf16*>(next_weight),
            static_cast<device::bf16*>(norm_out),
            hidden_size, eps);
}

void rmsnorm_residual_add_bf16(
    const void* x,
    const void* weight,
    void* hidden,
    int num_rows,
    int hidden_size,
    float eps,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    device::rmsnorm_residual_add<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const device::bf16*>(x),
        static_cast<const device::bf16*>(weight),
        static_cast<device::bf16*>(hidden),
        hidden_size, eps);
}

void rmsnorm_residual_add_scale_rmsnorm_bf16(
    const void* x,
    const void* weight,
    void* hidden,
    float scale,
    const void* next_weight,
    void* norm_out,
    int num_rows,
    int hidden_size,
    float eps,
    cudaStream_t stream)
{
    // Vectorized when the rows allow it (float4 = 8 bf16), scalar otherwise.
    //
    // The scalar form walks the row three times, once per pass, and measured
    // 10.79 us/call in gemma-4-26B's decode -- 8% of the step -- against 2.51
    // for the vectorized plain norm. Swept under graph replay at the shapes
    // these models use, us:
    //
    //   hidden   scalar256  scalar512   vec256  vec512  vec1024
    //     2048        4.38       3.68     2.72    2.93     3.31
    //     2816        6.17       4.83     3.46    3.12     3.51
    //     5376        8.48       6.55     4.44    4.07     4.02
    //
    // Against the shipping scalar/256 that is -38%, -49% and -53%. The
    // vectorized twin is BIT-IDENTICAL to the scalar form at all three sizes
    // (0 of 2048/2816/5376 bf16 values differ) -- only the two sum reductions
    // reassociate, and at these lengths that rounds to the same bf16.
    //
    // vec512 is chosen above hidden 2560 and vec256 below: it is best at 2816,
    // within 1.5% of best at 5376, and the 2048 case prefers the narrower
    // block. Scalar keeps hidden < 2560's old width only when the rows are
    // unaligned.
    dim3 grid(num_rows);
    const bool vec_ok = (hidden_size % 8) == 0 &&
        (reinterpret_cast<std::uintptr_t>(x) % 16) == 0 &&
        (reinterpret_cast<std::uintptr_t>(hidden) % 16) == 0 &&
        (reinterpret_cast<std::uintptr_t>(norm_out) % 16) == 0 &&
        (reinterpret_cast<std::uintptr_t>(weight) % 16) == 0 &&
        (reinterpret_cast<std::uintptr_t>(next_weight) % 16) == 0;
    if (vec_ok) {
        if (hidden_size >= 2560) {
            constexpr int kB = 512;
            device::rmsnorm_rasr_vec8<kB><<<grid, kB, 0, stream>>>(
                static_cast<const device::bf16*>(x),
                static_cast<const device::bf16*>(weight),
                static_cast<device::bf16*>(hidden), scale,
                static_cast<const device::bf16*>(next_weight),
                static_cast<device::bf16*>(norm_out), hidden_size, eps);
            return;
        }
        constexpr int kB = 256;
        device::rmsnorm_rasr_vec8<kB><<<grid, kB, 0, stream>>>(
            static_cast<const device::bf16*>(x),
            static_cast<const device::bf16*>(weight),
            static_cast<device::bf16*>(hidden), scale,
            static_cast<const device::bf16*>(next_weight),
            static_cast<device::bf16*>(norm_out), hidden_size, eps);
        return;
    }
    constexpr int BLOCK = 512;
    dim3 block(BLOCK);
    device::rmsnorm_residual_add_scale_rmsnorm<device::bf16, BLOCK>
        <<<grid, block, 0, stream>>>(
            static_cast<const device::bf16*>(x),
            static_cast<const device::bf16*>(weight),
            static_cast<device::bf16*>(hidden),
            scale,
            static_cast<const device::bf16*>(next_weight),
            static_cast<device::bf16*>(norm_out),
            hidden_size, eps);
}

void rmsnorm_gemma_bf16(
    const void* x, const void* weight, void* y,
    int num_rows, int hidden, float eps, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    if (rmsnorm_vec8_ok(x, y, weight, hidden, hidden, hidden)) {
        constexpr int VBLOCK = 512;
        device::rmsnorm_vec8<VBLOCK, /*WEIGHT_PLUS_ONE=*/true>
            <<<grid, VBLOCK, 0, stream>>>(
                static_cast<const device::bf16*>(x),
                static_cast<const device::bf16*>(weight),
                static_cast<device::bf16*>(y), nullptr,
                hidden, hidden, hidden, eps);
        return;
    }
    dim3 block(BLOCK);
    device::rmsnorm_gemma<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const device::bf16*>(x),
        static_cast<const device::bf16*>(weight),
        static_cast<device::bf16*>(y),
        hidden, hidden, hidden, eps);
}

void rmsnorm_no_scale_bf16(
    const void* x, void* y,
    int num_rows, int hidden, float eps, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    device::rmsnorm_no_scale<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const device::bf16*>(x),
        static_cast<device::bf16*>(y),
        hidden, eps);
}

void rmsnorm_gated_fp32_in_bf16(
    const void* x, const void* gate, const void* weight, void* y,
    int num_rows, int hidden, float eps, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    device::rmsnorm_gated_f32_in<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const float*>(x),
        static_cast<const device::bf16*>(gate),
        static_cast<const float*>(weight),
        static_cast<device::bf16*>(y),
        hidden, eps);
}

void rmsnorm_gated_bf16(
    const void* x, const void* gate, const void* weight, void* y,
    int num_rows, int hidden, float eps, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    device::rmsnorm_gated<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const device::bf16*>(x),
        static_cast<const device::bf16*>(gate),
        static_cast<const float*>(weight),
        static_cast<device::bf16*>(y),
        hidden, eps);
}

// Sweep entry point: VBLOCK is fixed at 512, chosen on an H100. At decode
// there is ONE row, so the whole kernel is a single block and VBLOCK decides
// both how many threads sit idle (hidden 2816 is 352 vec8 vectors, so 160 of
// 512 threads do nothing) and how deep the block reduction is (9 rounds at
// 512 vs 7 at 128). Shapes for the sweep come from the models' configs.
//
// The block width is an ARGUMENT here, which is the other reason no row names
// this: a `LaunchRule` states one geometry, and a sweep is a caller asking for
// five.
bool rmsnorm_bf16_tuned(
    const void* x, const void* weight, void* y, int num_rows, int hidden,
    float eps, int vblock, cudaStream_t stream)
{
    if (num_rows <= 0 || hidden <= 0 || (hidden % 8) != 0) return false;
    const dim3 grid(static_cast<unsigned>(num_rows));
    auto go = [&](auto V) {
        constexpr int VB = decltype(V)::value;
        device::rmsnorm_vec8<VB, /*WEIGHT_PLUS_ONE=*/false>
            <<<grid, VB, 0, stream>>>(
                static_cast<const device::bf16*>(x),
                static_cast<const device::bf16*>(weight),
                static_cast<device::bf16*>(y), nullptr,
                hidden, /*x_row_stride=*/hidden, /*y_row_stride=*/hidden, eps);
    };
#define PIE_RMS_CASE(V) \
    if (vblock == (V)) { go(std::integral_constant<int, V>{}); return true; }
    PIE_RMS_CASE(64) PIE_RMS_CASE(128) PIE_RMS_CASE(256)
    PIE_RMS_CASE(512) PIE_RMS_CASE(1024)
#undef PIE_RMS_CASE
    return false;
}

// Sweep entry point for the fused residual-add + norm + scale + norm.
// BLOCK is fixed at 256 and this kernel is the SCALAR form, so at hidden 2816
// each thread walks 11 loads -- twice, once per norm. The file's own note on
// the plain kernel records that vectorizing the same walk cut it ~7x. It
// measures 10.79 us/call in-engine against 2.51 for the vectorized plain
// norm, and is 8% of gemma-4-26B's decode step.
bool rmsnorm_rasr_tuned(
    const void* x, const void* weight, void* hidden, float scale,
    const void* next_weight, void* norm_out, int num_rows, int hidden_size,
    float eps, int block, cudaStream_t stream)
{
    if (num_rows <= 0 || hidden_size <= 0) return false;
    dim3 grid(static_cast<unsigned>(num_rows));
    const bool vec_ok = (hidden_size % 8) == 0 &&
        (reinterpret_cast<std::uintptr_t>(x) % 16) == 0 &&
        (reinterpret_cast<std::uintptr_t>(hidden) % 16) == 0 &&
        (reinterpret_cast<std::uintptr_t>(norm_out) % 16) == 0 &&
        (reinterpret_cast<std::uintptr_t>(weight) % 16) == 0 &&
        (reinterpret_cast<std::uintptr_t>(next_weight) % 16) == 0;
    if (block < 0) {   // negative block = "use the vectorized twin"
        const int b = -block;
        auto gov = [&](auto B) {
            constexpr int kB = decltype(B)::value;
            device::rmsnorm_rasr_vec8<kB><<<grid, kB, 0, stream>>>(
                static_cast<const device::bf16*>(x),
                static_cast<const device::bf16*>(weight),
                static_cast<device::bf16*>(hidden), scale,
                static_cast<const device::bf16*>(next_weight),
                static_cast<device::bf16*>(norm_out), hidden_size, eps);
        };
        if (!vec_ok) return false;
        if (b == 128) { gov(std::integral_constant<int,128>{}); return true; }
        if (b == 256) { gov(std::integral_constant<int,256>{}); return true; }
        if (b == 512) { gov(std::integral_constant<int,512>{}); return true; }
        if (b == 1024){ gov(std::integral_constant<int,1024>{}); return true; }
        return false;
    }
    auto go = [&](auto B) {
        constexpr int kB = decltype(B)::value;
        device::rmsnorm_residual_add_scale_rmsnorm<device::bf16, kB>
            <<<grid, kB, 0, stream>>>(
                static_cast<const device::bf16*>(x),
                static_cast<const device::bf16*>(weight),
                static_cast<device::bf16*>(hidden), scale,
                static_cast<const device::bf16*>(next_weight),
                static_cast<device::bf16*>(norm_out), hidden_size, eps);
    };
#define PIE_RASR_CASE(B) \
    if (block == (B)) { go(std::integral_constant<int, B>{}); return true; }
    PIE_RASR_CASE(128) PIE_RASR_CASE(256) PIE_RASR_CASE(512) PIE_RASR_CASE(1024)
#undef PIE_RASR_CASE
    return false;
}

}  // namespace pie_cuda_driver::kernels::norm
