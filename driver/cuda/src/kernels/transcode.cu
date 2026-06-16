#include "kernels/transcode.hpp"

#include <stdexcept>

#include "kernels/transcode.cuh"

namespace pie_cuda_driver::kernels {

namespace {

constexpr int kBlock = 256;

template <int GROUP, typename Decode, typename Encode>
void launch_kernel(const Decode& dec, const Encode& enc,
                   int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) {
        return;
    }
    transcode::transcode_rowmajor_kernel<GROUP, Decode, Encode>
        <<<rows, kBlock, 0, stream>>>(dec, enc, cols);
}

// Given the target's Encode + group width, dispatch over the source decoders.
// One arm per source; the compiler instantiates Decode x this Encode.
template <typename Encode, int GROUP>
void dispatch_source(TranscodeSource src, const TranscodeParams& p,
                     const Encode& enc, cudaStream_t stream)
{
    switch (src) {
    case TranscodeSource::Fp8E4m3PerGroup: {
        const int scale_cols =
            (p.cols + p.src_group_size - 1) / p.src_group_size;
        const transcode::DecodeFp8E4m3PerGroup dec{
            reinterpret_cast<const __nv_fp8_storage_t*>(p.src),
            p.src_scale, p.cols, scale_cols, p.src_group_size};
        launch_kernel<GROUP>(dec, enc, p.rows, p.cols, stream);
        return;
    }
    case TranscodeSource::Bf16: {
        const transcode::DecodeBf16 dec{
            reinterpret_cast<const __nv_bfloat16*>(p.src), p.cols};
        launch_kernel<GROUP>(dec, enc, p.rows, p.cols, stream);
        return;
    }
    }
    throw std::runtime_error("launch_transcode: unsupported source decoder");
}

}  // namespace

bool transcode_supported(TranscodeSource src, TranscodeTarget tgt)
{
    switch (tgt) {
    case TranscodeTarget::Mxfp4E2m1E8m0:
        return src == TranscodeSource::Fp8E4m3PerGroup
            || src == TranscodeSource::Bf16;
    }
    return false;
}

void launch_transcode(
    TranscodeSource src, TranscodeTarget tgt,
    const TranscodeParams& p, cudaStream_t stream)
{
    // One arm per target. Each binds its Encode + group width; dispatch_source
    // then binds the source. Source x Target kernels are compiler-generated.
    switch (tgt) {
    case TranscodeTarget::Mxfp4E2m1E8m0: {
        const transcode::EncodeMxfp4 enc{p.dst_packed, p.dst_scale, p.cols};
        dispatch_source<transcode::EncodeMxfp4, transcode::EncodeMxfp4::kGroup>(
            src, p, enc, stream);
        return;
    }
    }
    throw std::runtime_error("launch_transcode: unsupported target encoder");
}

}  // namespace pie_cuda_driver::kernels
