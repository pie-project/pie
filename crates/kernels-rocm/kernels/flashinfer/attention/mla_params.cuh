














#ifndef FLASHINFER_MLA_PARAMS_CUH_
#define FLASHINFER_MLA_PARAMS_CUH_
#include <cuda.h>

#include "../fastdiv.cuh"
#include "../profiler.cuh"

namespace flashinfer {

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct MLAParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  DTypeQ* q_nope;
  DTypeQ* q_pe;
  DTypeKV* ckv;
  DTypeKV* kpe;
  DTypeO* partial_o;
  float* partial_lse;
  DTypeO* final_o;
  float* final_lse;

  IdType* q_indptr;
  IdType* kv_indptr;
  IdType* partial_indptr;
  IdType* merge_packed_offset_start;
  IdType* merge_packed_offset_end;
  IdType* merge_partial_packed_offset_start;
  IdType* merge_partial_packed_offset_end;
  IdType* merge_partial_stride;
  IdType* kv_indices;
  IdType* q_len;
  IdType* kv_len;
  IdType* q_start;
  IdType* kv_start;
  IdType* kv_end;
  IdType* work_indptr;

  PROFILER_PARAMS_DECL

  uint_fastdiv block_size;
  uint_fastdiv num_heads;

  uint32_t q_nope_stride_n;
  uint32_t q_nope_stride_h;
  uint32_t q_pe_stride_n;
  uint32_t q_pe_stride_h;
  uint32_t ckv_stride_page;
  uint32_t ckv_stride_n;
  uint32_t kpe_stride_page;
  uint32_t kpe_stride_n;
  uint32_t o_stride_n;
  uint32_t o_stride_h;

  float sm_scale;

  float ckv_scale = 1.f;
  float kpe_scale = 1.f;
  bool return_lse_base_on_e;
};

};

#endif
