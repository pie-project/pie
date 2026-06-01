// Host launcher + compact config dispatch for the fused quant GEMM.
//
// This is the de-branded port of the upstream W4A16 launcher with the 313 KB
// generated shape-dispatch table replaced by a small hand-written switch over
// the instantiated configs (see qgemm_kernel_inst.cu). Only the bf16 x u4b8 ->
// bf16 (symmetric int4, no act-order, no zero point, fp32 accumulation) path
// is wired.

#include "qgemm.h"
#include "qgemm_common.cuh"
#include "qgemm_kernel_params.h"
// Pull in the full kernel definition so the kernel instantiations this
// dispatcher references are emitted in *this* translation unit. Under CUDA's
// default whole-program mode (-rdc=false), explicit __global__ template
// instantiations get internal linkage, so the function pointers taken below
// cannot resolve to definitions in another TU. Co-locating definition +
// dispatch keeps the build robust regardless of how the TUs are compiled.
// The canonical instantiation list still lives in qgemm_kernel_inst.cu (for
// builds that prefer separate instantiation); duplicate static copies across
// TUs are harmless.
#include "qgemm_kernel.cuh"

#include <algorithm>
#include <cuda_bf16.h>

namespace pie_cuda_device::qgemm {

using KernelPtr = void (*)(QGEMM_KERNEL_PARAMS);

namespace {

// The bf16 x u4b8 path is fixed at compile time.
constexpr QTypeId kA = kBFloat16.id();
constexpr QTypeId kB = kU4B8.id();
constexpr QTypeId kC = kBFloat16.id();
constexpr QTypeId kS = kBFloat16.id();
constexpr int kStages = 4;
constexpr int kNumBits = 4;

// The bf16 x fe4m3fn (fp8) path. Same activation / scale / output dtypes; the
// weight is 8-bit, so pack_factor and the B shared-mem footprint differ.
constexpr QTypeId kA8 = kBFloat16.id();
constexpr QTypeId kB8 = kFE4M3fn.id();
constexpr QTypeId kC8 = kBFloat16.id();
constexpr QTypeId kS8t = kBFloat16.id();
constexpr int kNumBits8 = 8;

// The bf16 x fe2m1f (mxfp4, GPT-OSS) path. 4-bit weight (pack_factor 8 like
// u4b8) but an 8-bit e8m0 *block* scale at group_size == 32 (group_blocks == 2).
constexpr QTypeId kAmx = kBFloat16.id();
constexpr QTypeId kBmx = kFE2M1f.id();
constexpr QTypeId kCmx = kBFloat16.id();
constexpr QTypeId kSmx = kFE8M0fnu.id();
constexpr int kNumBitsMx = 4;

struct ThreadConfig {
  int thread_k;
  int thread_n;
  int num_threads;
};

// Priority-ordered config lists (mirrors upstream small/large-batch tables).
constexpr ThreadConfig kSmallBatchConfigs[] = {
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128},
};
constexpr ThreadConfig kLargeBatchConfigs[] = {
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128},
};

// --- Shared-memory footprint of a given (thread_m_blocks, config) -----------
// Specialized to: no act-order, no zero point, 16-bit activation. Parameterized
// on the weight bit-width (4 for u4b8, 8 for fe4m3fn) since that scales the B
// shared-mem footprint via pack_factor. Returns size in int32 elements
// (matching upstream units).
int kernel_cache_size(const ThreadConfig& tc, int thread_m_blocks,
                      int group_size, int stages, int num_bits) {
  const int pack_factor = 32 / num_bits;
  int tb_k = tc.thread_k;
  int tb_n = tc.thread_n;
  int tb_m = thread_m_blocks * 16;

  int sh_a_size = stages * (tb_m * tb_k) * 2;             // bf16 activations
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size =
      (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = std::max(std::max(sh_b_size, sh_red_size), tmp_size);

  // Scales cache (no act-order branch).
  int tb_groups = (group_size == -1) ? 1 : div_ceil(tb_k, group_size);
  int sh_s_size = tb_groups * tb_n * 2 * stages;

  return tmp_size + sh_a_size + sh_s_size;  // no zp / g_idx for this path
}

bool is_valid_config(const ThreadConfig& tc, int thread_m_blocks, int prob_n,
                     int prob_k, int group_size, int stages, int num_bits,
                     int max_shared_mem) {
  if (tc.thread_k == -1 || tc.thread_n == -1 || tc.num_threads == -1)
    return false;
  if (prob_k % tc.thread_k != 0 || prob_n % tc.thread_n != 0) return false;
  if (tc.thread_n < min_thread_n || tc.thread_k < min_thread_k) return false;
  if (tc.num_threads < 128) return false;
  return kernel_cache_size(tc, thread_m_blocks, group_size, stages, num_bits) <=
         max_shared_mem;
}

// --- Compact instantiated-kernel switch -------------------------------------
// Returns nullptr if the (config, group_blocks, m_block_size_8) tuple is not
// one we instantiated. group_blocks is -1 (per-channel) or 8 (group 128).
// Templated on the (A,B,C,S) scalar ids so the same dispatch table serves both
// the u4b8 and the fe4m3fn weight paths (their instantiated tuples coincide).
template <QTypeId A, QTypeId B, QTypeId C, QTypeId S, int GB>
KernelPtr pick_kernel_gb(int threads, int thread_m_blocks, int thread_n_blocks,
                         int thread_k_blocks, bool m_block_size_8) {
#define QG(THREADS, TM, TN, TK, M8)                                       \
  if (threads == (THREADS) && thread_m_blocks == (TM) &&                  \
      thread_n_blocks == (TN) && thread_k_blocks == (TK) &&               \
      m_block_size_8 == (M8))                                             \
    return &QGemm<A, B, C, S, THREADS, TM, TN, TK, M8, kStages, GB,       \
                  false>;

  QG(256, 1, 8, 8, true)
  QG(128, 1, 8, 4, true)
  QG(128, 1, 4, 8, true)
  QG(256, 1, 8, 8, false)
  QG(128, 1, 8, 4, false)
  QG(128, 1, 4, 8, false)
  QG(256, 2, 16, 4, false)
  QG(128, 2, 8, 4, false)
  QG(128, 2, 4, 8, false)
  QG(256, 3, 16, 4, false)
  QG(128, 3, 8, 4, false)
  QG(128, 3, 4, 8, false)
  QG(256, 4, 16, 4, false)
  QG(128, 4, 8, 4, false)
  QG(128, 4, 4, 8, false)
#undef QG
  return nullptr;
}

template <QTypeId A, QTypeId B, QTypeId C, QTypeId S>
KernelPtr pick_kernel(int group_blocks, int threads, int thread_m_blocks,
                      int thread_n_blocks, int thread_k_blocks,
                      bool m_block_size_8) {
  // The kernel's static_assert restricts which (scale-type, group_blocks)
  // tuples are legal. e8m0 microscaling (mxfp4) is ONLY valid at group_blocks
  // == 2; the bf16-scale paths (u4b8 / fe4m3fn) use group_blocks ∈ {-1, 8}.
  // Guard each branch with `if constexpr` so an unreachable branch never forces
  // an illegal QGemm instantiation.
  constexpr bool kE8m0 = (S == kFE8M0fnu.id());
  if constexpr (!kE8m0) {
    if (group_blocks == -1)
      return pick_kernel_gb<A, B, C, S, -1>(threads, thread_m_blocks,
                                            thread_n_blocks, thread_k_blocks,
                                            m_block_size_8);
    if (group_blocks == 8)
      return pick_kernel_gb<A, B, C, S, 8>(threads, thread_m_blocks,
                                           thread_n_blocks, thread_k_blocks,
                                           m_block_size_8);
  } else {
    if (group_blocks == 2)
      return pick_kernel_gb<A, B, C, S, 2>(threads, thread_m_blocks,
                                           thread_n_blocks, thread_k_blocks,
                                           m_block_size_8);
  }
  return nullptr;
}

// Pick the first valid config for a given thread_m_blocks, returning both the
// kernel and the config used.
struct PickResult {
  KernelPtr kernel = nullptr;
  ThreadConfig cfg{-1, -1, -1};
};

template <QTypeId A, QTypeId B, QTypeId C, QTypeId S, int NumBits>
PickResult choose_config(int prob_m_split, int prob_n, int prob_k,
                         int thread_m_blocks, bool m_block_size_8,
                         int group_size, int group_blocks, int stages,
                         int max_shared_mem) {
  const ThreadConfig* configs =
      thread_m_blocks > 1 ? kLargeBatchConfigs : kSmallBatchConfigs;
  int n = thread_m_blocks > 1
              ? (int)(sizeof(kLargeBatchConfigs) / sizeof(ThreadConfig))
              : (int)(sizeof(kSmallBatchConfigs) / sizeof(ThreadConfig));
  for (int i = 0; i < n; i++) {
    ThreadConfig tc = configs[i];
    if (!is_valid_config(tc, thread_m_blocks, prob_n, prob_k, group_size,
                         stages, NumBits, max_shared_mem - 512))
      continue;
    KernelPtr k = pick_kernel<A, B, C, S>(group_blocks, tc.num_threads,
                                          thread_m_blocks, tc.thread_n / 16,
                                          tc.thread_k / 16, m_block_size_8);
    if (k) return {k, tc};
  }
  return {};
}

// --- Marlin scale permutation -----------------------------------------------
// The kernel reads `scales_ptr` assuming the scales have already been permuted
// along the N dimension into the tensor-core fragment layout (this matches
// upstream Marlin, where `marlin_permute_scales` is applied on the host before
// launch). The selftest / public API takes the *logical* [num_groups, N]
// row-major scales, so we permute them here into a scratch buffer.
//
// Mirrors vLLM `get_scale_perms()` / `marlin_permute_scales`:
//   grouped     (group_size < K, != -1):  block of 64 along N,
//       scale_perm[8*i + j] = i + 8*j        (8x8 transpose)
//   per-channel (group_size == -1):        block of 32 along N,
//       scale_perm_single[8*i + t] = 2*i + {0,1,8,9,16,17,24,25}[t]
// applied independently to every block; layout otherwise [num_groups, N].

// `bias_mul` lets the fp8 weight path fold the fp8->bf16 exponent-bias
// correction (2^120) into the permuted scale. For int4 it is 1.0f and is a
// no-op (multiplying a bf16 by an exact power of two only shifts the exponent,
// so no precision is lost as long as the scaled value stays in bf16 range).
__global__ void scale_permute_kernel(const __nv_bfloat16* __restrict__ src,
                                     __nv_bfloat16* __restrict__ dst, int total,
                                     int block, bool grouped, float bias_mul) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int base = idx - (idx % block);
  int c = idx % block;  // output column within the block
  int src_col;
  if (grouped) {
    // block == 64: out[8*i+j] = in[i + 8*j]
    int i = c / 8;
    int j = c % 8;
    src_col = i + 8 * j;
  } else {
    // block == 32: out[8*i+t] = in[2*i + perm8[t]]
    static const int perm8[8] = {0, 1, 8, 9, 16, 17, 24, 25};
    int i = c / 8;
    int t = c % 8;
    src_col = 2 * i + perm8[t];
  }
  __nv_bfloat16 v = src[base + src_col];
  if (bias_mul != 1.0f) {
    v = __float2bfloat16(__bfloat162float(v) * bias_mul);
  }
  dst[idx] = v;
}

// --- e8m0 (mxfp4) scale permutation -----------------------------------------
// The mxfp4 (fe2m1f) path uses 1-byte e8m0 block scales. The kernel expects
// them pre-permuted into the same MMA fragment layout as the bf16 scales, with
// an additional 4-lane byte reorder because e8m0 bytes are packed 4-per-int32.
//
// Forward permute (logical [num_groups, N] uint8, permuting the N axis within
// each group; N is always a multiple of 64). Within each 64-wide N block, for
// output column c:
//   (1) 4-lane reorder: m = (c & ~3) + perm4[c & 3],  perm4 = {0,2,1,3}
//   (2) 8x8 transpose:  src_col = (m % 8) * 8 + (m / 8)
// This is exactly the forward of the inverse permute the old driver applies in
// mxfp4_scales_to_marlin_e8m0 (Marlin's 64-wide transpose + vLLM/SGLang's
// four-lane [0,2,1,3] post-permute). No bias fold: e8m0 decodes to an exact
// power of two in-register.
__global__ void e8m0_scale_permute_kernel(const uint8_t* __restrict__ src,
                                          uint8_t* __restrict__ dst, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  constexpr int block = 64;
  int base = idx - (idx % block);
  int c = idx % block;
  static const int perm4[4] = {0, 2, 1, 3};
  int m = (c & ~3) + perm4[c & 3];
  int src_col = (m % 8) * 8 + (m / 8);
  dst[idx] = src[base + src_col];
}

// --- Shared launcher core ----------------------------------------------------
// Both the u4b8 (4-bit) and fe4m3fn (8-bit) weight paths share identical host
// orchestration (validation, scale permute, M-split stripe loop). They differ
// only in (A,B,C,S) scalar ids, the weight bit-width (for the shared-mem
// footprint check), and the scale bias-fold factor. Templating on those lets
// us avoid duplicating ~100 lines while keeping the int4 path byte-for-byte
// equivalent (bias_mul == 1.0f is a no-op in the permute kernel).
template <QTypeId A, QTypeId B, QTypeId C, QTypeId S, int NumBits>
cudaError_t launch_gemm(cudaStream_t stream, const void* act_bf16,
                        const void* qweight_packed, const void* scales_bf16,
                        void* out_bf16, int M, int N, int K, int group_size,
                        int* workspace, int sms, float bias_mul) {
  if (M <= 0 || N <= 0 || K <= 0) return cudaErrorInvalidValue;
  // Tile alignment constraints.
  if (K % tile_size != 0 || N % tile_size != 0) return cudaErrorInvalidValue;
  if (N % min_thread_n != 0 || K % min_thread_k != 0)
    return cudaErrorInvalidValue;
  if (group_size != -1 && group_size <= 0) return cudaErrorInvalidValue;
  if (group_size > 0 && (K % group_size != 0 || group_size % 16 != 0))
    return cudaErrorInvalidValue;

  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) return err;
  if (sms <= 0) {
    err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;
  }

  int max_shared_mem = 0;
  err = cudaDeviceGetAttribute(&max_shared_mem,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (err != cudaSuccess) return err;

  const int group_blocks = (group_size == -1) ? -1 : (group_size / 16);
  const int num_groups = (group_size > 0) ? (K / group_size) : 1;
  const int lda = K;
  const int stages = kStages;

  const int4* A_ptr = reinterpret_cast<const int4*>(act_bf16);
  const int4* B_ptr = reinterpret_cast<const int4*>(qweight_packed);
  int4* C_ptr = reinterpret_cast<int4*>(out_bf16);
  int* locks = workspace;

  // Permute the logical [num_groups, N] scales into the kernel's expected
  // tensor-core fragment layout (upstream `marlin_permute_scales`). The kernel
  // reads scales pre-permuted along N; the public API takes them logical, so we
  // do the transform here into a scratch buffer. For the fp8 path bias_mul
  // folds the 2^120 fp8->bf16 exponent-bias correction into the scale.
  const bool grouped_scales = (group_size != -1) && (group_size < K);
  const int perm_block = grouped_scales ? 64 : 32;  // along N
  const int total_scales = num_groups * N;
  __nv_bfloat16* s_perm = nullptr;
  err = cudaMallocAsync(&s_perm, (size_t)total_scales * sizeof(__nv_bfloat16),
                        stream);
  if (err != cudaSuccess) return err;
  {
    int threads_p = 256;
    int blocks_p = div_ceil(total_scales, threads_p);
    scale_permute_kernel<<<blocks_p, threads_p, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(scales_bf16), s_perm,
        total_scales, perm_block, grouped_scales, bias_mul);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      cudaFreeAsync(s_perm, stream);
      return err;
    }
  }
  const int4* s_ptr = reinterpret_cast<const int4*>(s_perm);

  // M-split loop: process the batch in chunks of up to (max_thread_m_blocks*16)
  // rows, running several parallel "stripes" for large M, exactly as upstream.
  int local_max_par = 16;
  if (N <= 4096) local_max_par = 16 * 8;
  int rest_m = M;
  int max_thread_m_blocks = 4;

  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > local_max_par) par_count = local_max_par;
    int prob_m_split =
        par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

    int thread_m_blocks = std::min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    bool m_block_size_8 = prob_m_split <= 8;  // 16-bit activation

    PickResult pick = choose_config<A, B, C, S, NumBits>(
        prob_m_split, N, K, thread_m_blocks, m_block_size_8, group_size,
        group_blocks, stages, max_shared_mem);

    if (!pick.kernel) {
      // Could not place this many thread_m_blocks; try fewer rows per stripe.
      if (max_thread_m_blocks > 1) {
        max_thread_m_blocks--;
        continue;
      }
      cudaFreeAsync(s_perm, stream);
      return cudaErrorInvalidConfiguration;
    }

    int num_threads = pick.cfg.num_threads;
    int blocks = sms;

    err = cudaFuncSetAttribute(pick.kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               max_shared_mem);
    if (err != cudaSuccess) {
      cudaFreeAsync(s_perm, stream);
      return err;
    }

    pick.kernel<<<blocks, num_threads, max_shared_mem, stream>>>(
        A_ptr, B_ptr, C_ptr, /*C_tmp=*/nullptr, /*b_bias_ptr=*/nullptr,
        /*a_scales_ptr=*/nullptr, s_ptr, /*global_scale_ptr=*/nullptr,
        /*zp_ptr=*/nullptr, /*g_idx=*/nullptr, num_groups, prob_m_split, N, K,
        lda, locks, /*has_bias=*/false, /*use_atomic_add=*/false,
        /*use_fp32_reduce=*/false, max_shared_mem);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      cudaFreeAsync(s_perm, stream);
      return err;
    }

    // Advance pointers (bf16 activation: 8 elems per int4).
    A_ptr += prob_m_split * (lda / 8);
    C_ptr += prob_m_split * (N / 8);
    rest_m -= prob_m_split;
  }
  cudaFreeAsync(s_perm, stream);
  return cudaSuccess;
}

// --- mxfp4 (fe2m1f weight + e8m0 block scale) launcher ----------------------
// Mirrors launch_gemm's host orchestration, but the scale is a 1-byte e8m0
// block scale (group_size == 32, group_blocks == 2) rather than a bf16 scale,
// so it permutes through e8m0_scale_permute_kernel into a uint8 scratch buffer.
// The kernel reads scales_ptr as int4; the e8m0 fragment expansion happens
// in-kernel (dequant_fp8_scales<kFE8M0fnu>).
cudaError_t launch_mxfp4_gemm(cudaStream_t stream, const void* act_bf16,
                              const void* qweight_packed,
                              const void* scales_e8m0, void* out_bf16, int M,
                              int N, int K, int* workspace, int sms) {
  if (M <= 0 || N <= 0 || K <= 0) return cudaErrorInvalidValue;
  if (K % tile_size != 0 || N % tile_size != 0) return cudaErrorInvalidValue;
  if (N % min_thread_n != 0 || K % min_thread_k != 0)
    return cudaErrorInvalidValue;
  constexpr int group_size = 32;  // fixed for mxfp4 e8m0 microscaling
  if (K % group_size != 0) return cudaErrorInvalidValue;

  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) return err;
  if (sms <= 0) {
    err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;
  }
  int max_shared_mem = 0;
  err = cudaDeviceGetAttribute(&max_shared_mem,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (err != cudaSuccess) return err;

  const int group_blocks = group_size / 16;  // == 2
  const int num_groups = K / group_size;
  const int lda = K;
  const int stages = kStages;

  const int4* A_ptr = reinterpret_cast<const int4*>(act_bf16);
  const int4* B_ptr = reinterpret_cast<const int4*>(qweight_packed);
  int4* C_ptr = reinterpret_cast<int4*>(out_bf16);
  int* locks = workspace;

  // Permute the logical [num_groups, N] uint8 e8m0 scales into the kernel's
  // fragment layout (64-wide transpose + 4-lane [0,2,1,3] reorder).
  const int total_scales = num_groups * N;
  uint8_t* s_perm = nullptr;
  err = cudaMallocAsync(&s_perm, (size_t)total_scales * sizeof(uint8_t), stream);
  if (err != cudaSuccess) return err;
  {
    int threads_p = 256;
    int blocks_p = div_ceil(total_scales, threads_p);
    e8m0_scale_permute_kernel<<<blocks_p, threads_p, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(scales_e8m0), s_perm, total_scales);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      cudaFreeAsync(s_perm, stream);
      return err;
    }
  }
  const int4* s_ptr = reinterpret_cast<const int4*>(s_perm);

  int local_max_par = 16;
  if (N <= 4096) local_max_par = 16 * 8;
  int rest_m = M;
  int max_thread_m_blocks = 4;

  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > local_max_par) par_count = local_max_par;
    int prob_m_split =
        par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

    int thread_m_blocks =
        std::min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    bool m_block_size_8 = prob_m_split <= 8;

    PickResult pick = choose_config<kAmx, kBmx, kCmx, kSmx, kNumBitsMx>(
        prob_m_split, N, K, thread_m_blocks, m_block_size_8, group_size,
        group_blocks, stages, max_shared_mem);

    if (!pick.kernel) {
      if (max_thread_m_blocks > 1) {
        max_thread_m_blocks--;
        continue;
      }
      cudaFreeAsync(s_perm, stream);
      return cudaErrorInvalidConfiguration;
    }

    int num_threads = pick.cfg.num_threads;
    int blocks = sms;

    err = cudaFuncSetAttribute(pick.kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               max_shared_mem);
    if (err != cudaSuccess) {
      cudaFreeAsync(s_perm, stream);
      return err;
    }

    pick.kernel<<<blocks, num_threads, max_shared_mem, stream>>>(
        A_ptr, B_ptr, C_ptr, /*C_tmp=*/nullptr, /*b_bias_ptr=*/nullptr,
        /*a_scales_ptr=*/nullptr, s_ptr, /*global_scale_ptr=*/nullptr,
        /*zp_ptr=*/nullptr, /*g_idx=*/nullptr, num_groups, prob_m_split, N, K,
        lda, locks, /*has_bias=*/false, /*use_atomic_add=*/false,
        /*use_fp32_reduce=*/false, max_shared_mem);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      cudaFreeAsync(s_perm, stream);
      return err;
    }

    A_ptr += prob_m_split * (lda / 8);
    C_ptr += prob_m_split * (N / 8);
    rest_m -= prob_m_split;
  }
  cudaFreeAsync(s_perm, stream);
  return cudaSuccess;
}

}  // namespace

// ---------------------------------------------------------------------------

int w4a16_workspace_ints(int N, int max_M) {
  // The kernel uses one int32 reduction lock per concurrently scheduled
  // threadblock; the launch heuristic caps blocks at the SM count. A fixed,
  // generous bound covers any current GPU (way past ~32K SMs) so callers can
  // allocate once. = 64 KiB / sizeof(int) = 16384 ints.
  (void)N;
  (void)max_M;
  return 16384;
}

cudaError_t w4a16_bf16_gemm(cudaStream_t stream, const void* act_bf16,
                            const int32_t* qweight_packed,
                            const void* scales_bf16, void* out_bf16, int M,
                            int N, int K, int group_size, int* workspace,
                            int sms) {
  return launch_gemm<kA, kB, kC, kS, kNumBits>(
      stream, act_bf16, qweight_packed, scales_bf16, out_bf16, M, N, K,
      group_size, workspace, sms, /*bias_mul=*/1.0f);
}

int w8a16_fp8_workspace_ints(int N, int max_M) {
  return w4a16_workspace_ints(N, max_M);
}

cudaError_t w8a16_fp8_bf16_gemm(cudaStream_t stream, const void* act_bf16,
                                const void* qweight_fp8,
                                const void* scales_bf16, void* out_bf16, int M,
                                int N, int K, int group_size, int* workspace,
                                int sms) {
  // fp8 (fe4m3fn) -> bf16 dequant uses the `skip_flop` path: the in-register
  // conversion produces value * 2^-120, so we fold the 2^120 exponent-bias
  // correction into the (permuted) bf16 scale. BIAS_OFFSET = 2^7 - 2^3 = 120.
  // 2^120 is exactly representable, so the fold only shifts the scale exponent
  // (no mantissa loss) provided scale * 2^120 stays within bf16 range.
  constexpr float kFp8BiasMul = 1.329227995784916e+36f;  // 2^120
  return launch_gemm<kA8, kB8, kC8, kS8t, kNumBits8>(
      stream, act_bf16, qweight_fp8, scales_bf16, out_bf16, M, N, K, group_size,
      workspace, sms, kFp8BiasMul);
}

int w4a16_mxfp4_workspace_ints(int N, int max_M) {
  return w4a16_workspace_ints(N, max_M);
}

cudaError_t w4a16_mxfp4_bf16_gemm(cudaStream_t stream, const void* act_bf16,
                                  const void* qweight_mxfp4,
                                  const void* scales_e8m0, void* out_bf16, int M,
                                  int N, int K, int* workspace, int sms) {
  return launch_mxfp4_gemm(stream, act_bf16, qweight_mxfp4, scales_e8m0,
                           out_bf16, M, N, K, workspace, sms);
}

}  // namespace pie_cuda_device::qgemm
