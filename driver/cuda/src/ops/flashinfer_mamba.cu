#include "ops/flashinfer_mamba.hpp"

#include "cuda_check.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifndef FLASHINFER_ENABLE_BF16
#define FLASHINFER_ENABLE_BF16 1
#endif

using state_t = nv_bfloat16;
using input_t = nv_bfloat16;
using weight_t = nv_bfloat16;
using matrixA_t = float;
using stateIndex_t = int32_t;
using cuSeqlensIndex_t = int32_t;
using numAcceptedIndex_t = int32_t;
using state_scale_t = void;

constexpr int DIM = 64;
constexpr int DSTATE = 128;
constexpr int NTOKENS_MTP = 1;
constexpr int PHILOX_ROUNDS = 0;

#include <flashinfer/mamba/selective_state_update.cuh>

namespace pie_cuda_driver::ops {
namespace {

bool env_truthy(const char* value) {
    if (value == nullptr || value[0] == '\0') return false;
    return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
           value[0] == 't' || value[0] == 'T' || value[0] == 'o' ||
           value[0] == 'O';
}

// -1 = unset, 0 = explicitly off, 1 = explicitly on.
int env_tristate(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return -1;
    return env_truthy(v) ? 1 : 0;
}

int current_device_major() {
    thread_local int cached_device = -1;
    thread_local int cached_major = 0;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (dev != cached_device) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        cached_device = dev;
        cached_major = prop.major;
    }
    return cached_major;
}

flashinfer::mamba::SSUAlgorithm requested_algorithm() {
    const char* v = std::getenv("PIE_NEMOTRON_FLASHINFER_SSU_ALGO");
    if (v == nullptr || v[0] == '\0' || std::strcmp(v, "auto") == 0) {
        return flashinfer::mamba::SSUAlgorithm::kAuto;
    }
    if (std::strcmp(v, "simple") == 0) {
        return flashinfer::mamba::SSUAlgorithm::kSimple;
    }
    if (std::strcmp(v, "vertical") == 0) {
        return flashinfer::mamba::SSUAlgorithm::kVertical;
    }
    if (std::strcmp(v, "horizontal") == 0) {
        return flashinfer::mamba::SSUAlgorithm::kHorizontal;
    }
    return flashinfer::mamba::SSUAlgorithm::kAuto;
}

}  // namespace

bool flashinfer_mamba_ssu_enabled() {
    // Env override is read once and cached. On L40 (sm_89) FlashInfer SSU only
    // exposes the "simple" algorithm, which trails the legacy warp_kernel past
    // R~144 (microbenched 2026-05-25); the "vertical"/"horizontal" algorithms
    // that beat the legacy path are gated to sm_90+. So default by SM:
    // on sm_90+, prefer FlashInfer; on sm_89 and below, keep the legacy path.
    static const int forced =
        env_tristate("PIE_NEMOTRON_FLASHINFER_SSU");
    if (forced >= 0) return forced != 0;
    return current_device_major() >= 9;
}

bool flashinfer_mamba_ssu_bf16(
    const std::uint16_t* conv_out,
    const std::uint16_t* dt,
    const float* A,
    const std::uint16_t* D,
    const std::uint16_t* dt_bias,
    std::uint16_t* state_base,
    const std::int32_t* slot_ids,
    std::uint16_t* y,
    int batch,
    int num_heads,
    int head_dim,
    int state_size,
    int num_groups,
    int conv_dim,
    int intermediate,
    int state_cache_size,
    cudaStream_t stream) {
    if (!flashinfer_mamba_ssu_enabled()) return false;
    if (conv_out == nullptr || dt == nullptr || A == nullptr || D == nullptr ||
        dt_bias == nullptr || state_base == nullptr || y == nullptr) {
        return false;
    }
    if (batch <= 0 || num_heads <= 0 || num_groups <= 0 ||
        num_heads % num_groups != 0) {
        return false;
    }
    if (head_dim != DIM || state_size != DSTATE) return false;
    if (conv_dim < intermediate + 2 * num_groups * state_size) return false;

    flashinfer::mamba::SelectiveStateUpdateParams p;
    p.batch = static_cast<std::uint32_t>(batch);
    p.nheads = static_cast<std::uint32_t>(num_heads);
    p.dim = static_cast<std::uint32_t>(head_dim);
    p.dstate = static_cast<std::uint32_t>(state_size);
    p.ngroups = static_cast<std::uint32_t>(num_groups);
    p.state_cache_size = static_cast<std::uint32_t>(
        std::max(state_cache_size, batch));

    p.x_stride_batch = conv_dim;
    p.dt_stride_batch = num_heads;
    p.B_stride_batch = conv_dim;
    p.C_stride_batch = conv_dim;
    p.out_stride_batch = intermediate;
    p.state_stride_batch =
        static_cast<long long>(num_heads) * head_dim * state_size;

    p.state = state_base;
    p.x = const_cast<std::uint16_t*>(conv_out);
    p.dt = const_cast<std::uint16_t*>(dt);
    p.dt_bias = const_cast<std::uint16_t*>(dt_bias);
    p.A = const_cast<float*>(A);
    p.B = const_cast<std::uint16_t*>(conv_out + intermediate);
    p.C = const_cast<std::uint16_t*>(
        conv_out + intermediate + num_groups * state_size);
    p.D = const_cast<std::uint16_t*>(D);
    p.output = y;
    p.state_batch_indices = const_cast<std::int32_t*>(slot_ids);
    p.state_batch_indices_stride_batch = 1;
    p.dt_softplus = true;
    p.update_state = true;

    flashinfer::mamba::invokeSelectiveStateUpdate<
        input_t, weight_t, matrixA_t, state_t, stateIndex_t, state_scale_t>(
        p, requested_algorithm(), stream);
    return true;
}

}  // namespace pie_cuda_driver::ops
