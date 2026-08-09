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

// # Three fused norms went, and one of them is the §41 shape exactly
//
// §43 deleted `residual_add_rmsnorm_bf16`, `residual_add_scale_rmsnorm_bf16`
// and `rmsnorm_residual_add_bf16`. The first and third are routed rows
// (`device.rs`'s `JIT_DISPATCHED`) whose kernels stay in `norm/rmsnorm.cuh`.
//
// The middle one is the interesting one: it had a row until §28.4 measured
// the row as a second name for a job a reached row already does, and
// `launch_abi.rs` then carried the LAUNCHER as `NormNoRow::Orphaned` with the
// reason written out -- "this one's last consumer is `sources.rs`'
// `EXPECTED` `<<<>>>` census". A census is not a consumer. gemma-4 fires
// `rmsnorm_residual_add_scale_rmsnorm_bf16`, four statements and 221 golden
// lines, which is directly below and stays.

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

// `rmsnorm_gemma_bf16` and `rmsnorm_no_scale_bf16` were deleted here by §43.
// Both rows are live and routed (`device.rs`'s `JIT_DISPATCHED`), and
// `jit_parity.rs` carries a recorded AOT fingerprint for each from the day
// they were routed, which is what survives the launcher. `gemma4_vision.cu`
// used to call `rmsnorm_no_scale_bf16`; it now launches
// `nd::rmsnorm_no_scale<bfd,256>` itself, so that hold is released too.

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

// `rmsnorm_gated_bf16` was deleted here by §43 -- routed to NVRTC out of
// `norm/rmsnorm.cuh` (`device.rs`'s `JIT_DISPATCHED`), so the shim forwards
// to nothing and the row fires the template instead. §10.10 step 5.

// # The two sweep entry points went, and with them the file's only `bool`
//
// §43 deleted `rmsnorm_bf16_tuned` and `rmsnorm_rasr_tuned`, with their
// `PIE_RMS_CASE` and `PIE_RASR_CASE` ladders. Both were microbenchmark sweep
// entry points -- deliberately unrowed, because a `LaunchRule` states one
// geometry and a sweep is a caller asking for five -- and the sweep that
// called them is not in the tree: no bench, no example, no test, no `.cu`,
// no `ffi::` fire. `launch_abi.rs` carried them as `NormNoRow::AutotunerProbe`
// on exactly that reading, "returns bool and has zero driver call sites";
// §41's audit turned "zero driver call sites" into "zero call sites".
//
// The measurements they were written for survive in the comments above their
// callees and in `new-horizon.md`; what does not survive is a way to re-take
// them without re-adding a sweep, which is the trade §43 made knowingly.

}  // namespace pie_cuda_driver::kernels::norm
