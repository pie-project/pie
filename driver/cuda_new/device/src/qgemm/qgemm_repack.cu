// Weight prepack for the fused quant GEMM. (De-branded port of the GPTQ
// repack kernel.) Converts plain GPTQ-packed int4 weights into the tile /
// interleave layout the GEMM kernel expects.
//
// Expected input layout (qweight_rowmajor_packed), GPTQ convention:
//   * shape [size_k / 8, size_n] int32, row-major.
//   * Each int32 at (kp, n) packs 8 consecutive int4 weights along K:
//     nibble j (bits [4j, 4j+4)) holds the weight for k = kp*8 + j, column n.
//   * Weights are stored unsigned in [0, 15]; the GEMM applies the symmetric
//     -8 zero point during dequant (u4b8), so a stored nibble v decodes to the
//     signed value (v - 8) before scaling.
//
// Output layout (qweight_out):
//   * shape [size_k / tile_size, size_n * tile_size / 8] int32, where
//     tile_size = 16. The kernel reads this with its tensor-core friendly
//     interleave (handled internally; the loader only needs to allocate the
//     output buffer at this size and run this repack).
//
// Constraints: size_k % tile_k_size(=16) == 0, size_n % tile_n_size(=64) == 0.
// No activation reordering (act-order / g_idx) and no zero points: this is the
// symmetric W4A16 GPTQ path.

#include "qgemm_common.cuh"
#include "qgemm.h"

namespace pie_cuda_device::qgemm {

template <int const num_threads, int const num_bits, bool const has_perm>
__global__ void qgemm_repack_kernel(uint32_t const* __restrict__ b_q_weight_ptr,
                                    uint32_t const* __restrict__ perm_ptr,
                                    uint32_t* __restrict__ out_ptr, int size_k,
                                    int size_n) {
  constexpr int pack_factor = 32 / num_bits;

  constexpr int target_tile_n_size = tile_n_size;
  constexpr int target_tile_k_size = tile_k_size;
  int k_tiles = size_k / target_tile_k_size;
  int n_tiles = size_n / target_tile_n_size;
  int block_k_tiles = div_ceil(k_tiles, gridDim.x);

  auto start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) return;

  int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

  auto wait_for_stage = [&]() {
    cp_async_wait<repack_stages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 sh[];

  constexpr int perm_size = target_tile_k_size / 4;

  int4* sh_perm_ptr = sh;
  int4* sh_pipe_ptr = sh_perm_ptr;
  if constexpr (has_perm) {
    sh_pipe_ptr += perm_size;
  }

  constexpr int tile_ints = target_tile_k_size / pack_factor;

  constexpr int stage_n_threads = target_tile_n_size / 4;
  constexpr int stage_k_threads = has_perm ? target_tile_k_size : tile_ints;
  constexpr int stage_size = stage_k_threads * stage_n_threads;

  auto load_perm_to_shared = [&](int k_tile_id) {
    int first_k_int4 = (k_tile_id * target_tile_k_size) / 4;
    int4 const* perm_int4_ptr = reinterpret_cast<int4 const*>(perm_ptr);
    if (threadIdx.x < perm_size) {
      sh_perm_ptr[threadIdx.x] = perm_int4_ptr[first_k_int4 + threadIdx.x];
    }
    __syncthreads();
  };

  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      cp_async_fence();
      return;
    }

    int first_n = n_tile_id * target_tile_n_size;
    int4* sh_ptr = sh_pipe_ptr + stage_size * pipe;

    if constexpr (has_perm) {
      if (threadIdx.x < stage_size) {
        auto k_id = threadIdx.x / stage_n_threads;
        auto n_id = threadIdx.x % stage_n_threads;
        uint32_t const* sh_perm_int_ptr =
            reinterpret_cast<uint32_t const*>(sh_perm_ptr);
        int src_k = sh_perm_int_ptr[k_id];
        int src_k_packed = src_k / pack_factor;
        cp_async4(
            &sh_ptr[k_id * stage_n_threads + n_id],
            reinterpret_cast<int4 const*>(&(
                b_q_weight_ptr[src_k_packed * size_n + first_n + (n_id * 4)])));
      }
    } else {
      if (threadIdx.x < stage_size) {
        auto k_id = threadIdx.x / stage_n_threads;
        auto n_id = threadIdx.x % stage_n_threads;
        int first_k = k_tile_id * target_tile_k_size;
        int first_k_packed = first_k / pack_factor;
        cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                  reinterpret_cast<int4 const*>(
                      &(b_q_weight_ptr[(first_k_packed + k_id) * size_n +
                                       first_n + (n_id * 4)])));
      }
    }

    cp_async_fence();
  };

  auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) return;

    auto warp_id = threadIdx.x / 32;
    auto th_id = threadIdx.x % 32;
    if (warp_id >= 4) return;

    int tc_col = th_id / 4;
    int tc_row = (th_id % 4) * 2;

    constexpr int tc_offsets[4] = {0, 1, 8, 9};

    int cur_n = warp_id * 16 + tc_col;

    constexpr int sh_stride = target_tile_n_size;
    constexpr uint32_t mask = (1 << num_bits) - 1;

    int4* sh_stage_ptr = sh_pipe_ptr + stage_size * pipe;
    uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);
    uint32_t* sh_perm_int_ptr = reinterpret_cast<uint32_t*>(sh_perm_ptr);

    uint32_t vals[8];

    if constexpr (has_perm) {
      for (int i = 0; i < 4; i++) {
        int k_idx = tc_row + tc_offsets[i];
        uint32_t src_k = sh_perm_int_ptr[k_idx];
        uint32_t src_k_pos = src_k % pack_factor;
        uint32_t b1_val = sh_stage_int_ptr[k_idx * sh_stride + cur_n];
        uint32_t b1_cur_val = (b1_val >> (src_k_pos * num_bits)) & mask;
        uint32_t b2_val = sh_stage_int_ptr[k_idx * sh_stride + cur_n + 8];
        uint32_t b2_cur_val = (b2_val >> (src_k_pos * num_bits)) & mask;
        vals[i] = b1_cur_val;
        vals[4 + i] = b2_cur_val;
      }
    } else {
      uint32_t b1_vals[tile_ints];
      uint32_t b2_vals[tile_ints];

#pragma unroll
      for (int i = 0; i < tile_ints; i++) {
        b1_vals[i] = sh_stage_int_ptr[cur_n + sh_stride * i];
        b2_vals[i] = sh_stage_int_ptr[cur_n + 8 + sh_stride * i];
      }

#pragma unroll
      for (int i = 0; i < 4; i++) {
        int cur_elem = tc_row + tc_offsets[i];
        int cur_int = cur_elem / pack_factor;
        int cur_pos = cur_elem % pack_factor;
        vals[i] = (b1_vals[cur_int] >> (cur_pos * num_bits)) & mask;
        vals[4 + i] = (b2_vals[cur_int] >> (cur_pos * num_bits)) & mask;
      }
    }

    constexpr int out_tile_size =
        target_tile_k_size * target_tile_n_size / pack_factor;
    int out_offset = (k_tile_id * n_tiles + n_tile_id) * out_tile_size;

    if constexpr (num_bits == 4) {
      // num_bits == 4 interleave (FasterTransformer numeric-conversion order).
      int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }
      out_ptr[out_offset + th_id * 4 + warp_id] = res;
    } else {
      // num_bits == 8 interleave (two output ints per thread, byte-packed).
      // Matches upstream gptq_marlin_repack (is_a_8bit == false) num_bits==8.
      constexpr int pack_idx[4] = {0, 2, 1, 3};
      uint32_t res1 = 0;
      uint32_t res2 = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int ii = pack_idx[i];
        res1 |= vals[ii] << (i * 8);
        res2 |= vals[4 + ii] << (i * 8);
      }
      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
    }
  };

  auto start_pipes = [&](int k_tile_id, int n_tile_id) {
#pragma unroll
    for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    }
    wait_for_stage();
  };

#pragma unroll
  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
    int n_tile_id = 0;

    if constexpr (has_perm) {
      load_perm_to_shared(k_tile_id);
    }

    start_pipes(k_tile_id, n_tile_id);

    while (n_tile_id < n_tiles) {
#pragma unroll
      for (int pipe = 0; pipe < repack_stages; pipe++) {
        fetch_to_shared((pipe + repack_stages - 1) % repack_stages, k_tile_id,
                        n_tile_id + pipe + repack_stages - 1);
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
      }
      n_tile_id += repack_stages;
    }
  }
}

cudaError_t w4a16_repack(cudaStream_t stream,
                         const int32_t* qweight_rowmajor_packed,
                         int32_t* qweight_out, int N, int K) {
  if (N <= 0 || K <= 0) return cudaErrorInvalidValue;
  if (K % tile_k_size != 0 || N % tile_n_size != 0)
    return cudaErrorInvalidValue;

  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) return err;
  int blocks = 0;
  err = cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev);
  if (err != cudaSuccess) return err;
  int max_shared_mem = 0;
  err = cudaDeviceGetAttribute(&max_shared_mem,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (err != cudaSuccess) return err;

  constexpr int NUM_BITS = 4;
  constexpr bool HAS_PERM = false;
  auto kern = qgemm_repack_kernel<repack_threads, NUM_BITS, HAS_PERM>;
  err = cudaFuncSetAttribute(
      kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
  if (err != cudaSuccess) return err;

  kern<<<blocks, repack_threads, max_shared_mem, stream>>>(
      reinterpret_cast<const uint32_t*>(qweight_rowmajor_packed),
      /*perm=*/nullptr, reinterpret_cast<uint32_t*>(qweight_out), K, N);
  return cudaGetLastError();
}

cudaError_t w8a16_fp8_repack(cudaStream_t stream,
                             const int32_t* qweight_rowmajor_packed,
                             int32_t* qweight_out, int N, int K) {
  if (N <= 0 || K <= 0) return cudaErrorInvalidValue;
  if (K % tile_k_size != 0 || N % tile_n_size != 0)
    return cudaErrorInvalidValue;

  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) return err;
  int blocks = 0;
  err = cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev);
  if (err != cudaSuccess) return err;
  int max_shared_mem = 0;
  err = cudaDeviceGetAttribute(&max_shared_mem,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (err != cudaSuccess) return err;

  constexpr int NUM_BITS = 8;
  constexpr bool HAS_PERM = false;
  auto kern = qgemm_repack_kernel<repack_threads, NUM_BITS, HAS_PERM>;
  err = cudaFuncSetAttribute(
      kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
  if (err != cudaSuccess) return err;

  kern<<<blocks, repack_threads, max_shared_mem, stream>>>(
      reinterpret_cast<const uint32_t*>(qweight_rowmajor_packed),
      /*perm=*/nullptr, reinterpret_cast<uint32_t*>(qweight_out), K, N);
  return cudaGetLastError();
}

}  // namespace pie_cuda_device::qgemm
