/*
 * Copyright (c) 2023-2025 by FlashInfer team.
 *
 * Carried from FlashInfer v0.6.15, which adapts it from vLLM (Copyright contributors to the vLLM project).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <type_traits>


namespace pie::collective::p2p {

constexpr int kMaxBlocks = 36;


using FlagType = uint32_t;
struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][8];

  alignas(128) FlagType peer_counter[2][kMaxBlocks][8];
};

struct __align__(16) RankData {
  void* ptrs[8];
};

struct __align__(16) RankSignals {
  Signal* signals[8];
};

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {

  using P = array_t<T, 16 / sizeof(T)>;

  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

DINLINE float upcast_s(half val) { return __half2float(val); }

template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) {
  return __float2half(val);
}

DINLINE half& assign_add(half& a, half b) {
  a = __hadd(a, b);
  return a;
}
DINLINE float& assign_add(float& a, float b) { return a += b; }

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }
template <>
DINLINE nv_bfloat16 downcast_s(float val) {
  return __float2bfloat16(val);
}
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) {
  a = __hadd(a, b);
  return a;
}
#endif

template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) {
      out.data[i] = upcast_s(val.data[i]);
    }
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) {
      out.data[i] = downcast_s<typename O::type>(val.data[i]);
    }
    return out;
  }
}

static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#else
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#endif
}

static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
  FlagType flag;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#else
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(flag) : "l"(flag_addr));
#endif
  return flag;
}

static DINLINE void st_flag_volatile(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_volatile(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

// pie: a rank whose partner never arrives (the ranks diverged) traps after
// ten minutes, so the load or fire fails with a device fault instead of
// spinning for the life of the process.
template <bool acquire>
static DINLINE void wait_flag(FlagType* flag_addr, FlagType val) {
  unsigned long long start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  while ((acquire ? ld_flag_acquire(flag_addr) : ld_flag_volatile(flag_addr)) != val) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if (now - start > 600ull * 1000000000ull) __trap();
  }
}

template <int ngpus, bool is_start, bool need_fence = false>
DINLINE void multi_gpu_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  if constexpr (!is_start) __syncthreads();
  static_assert(!(is_start && need_fence));
  if (threadIdx.x < ngpus) {

    auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;

    auto peer_counter_ptr = &sg.signals[threadIdx.x]->peer_counter[val % 2][blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->peer_counter[val % 2][blockIdx.x][threadIdx.x];
    if constexpr (need_fence) {
      st_flag_release(peer_counter_ptr, val);
      wait_flag<true>(self_counter_ptr, val);
    } else {
      st_flag_volatile(peer_counter_ptr, val);
      wait_flag<false>(self_counter_ptr, val);
    }
  }
  if constexpr (is_start || need_fence) __syncthreads();
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

}
