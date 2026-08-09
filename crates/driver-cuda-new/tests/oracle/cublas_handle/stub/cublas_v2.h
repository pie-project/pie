// Stub <cublas_v2.h> for the gemm oracle.
//
// Declarations only; the oracle defines recorders for what it drives and
// --gc-sections discards the rest. `cublasHandle_t` is spelled exactly as
// cublas_api.h spells it (a pointer to an incomplete cublasContext) for the
// launch-ABI stub's reason.
#pragma once
#include <cstddef>
#include <cstdint>

#include "cuda_runtime.h"

struct cublasContext;
using cublasHandle_t = cublasContext*;

using cublasStatus_t = int;
constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
constexpr cublasStatus_t CUBLAS_STATUS_NOT_INITIALIZED = 1;
constexpr cublasStatus_t CUBLAS_STATUS_NOT_SUPPORTED = 15;

enum cublasMath_t {
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_TENSOR_OP_MATH = 1,
};

enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
};

enum cublasGemmAlgo_t {
    CUBLAS_GEMM_DEFAULT = -1,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
};

// cudaDataType lives in library_types.h in the real toolkit; cublas_v2.h
// drags it in.
enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_R_32F = 0,
    CUDA_R_16BF = 14,
    CUDA_R_8I = 3,
    CUDA_R_32I = 10,
    CUDA_R_8F_E4M3 = 28,
};
using cudaDataType = cudaDataType_t;

enum cublasComputeType_t {
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32I = 70,
};

cublasStatus_t cublasCreate(cublasHandle_t* handle);
cublasStatus_t cublasDestroy(cublasHandle_t handle);
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream);
cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* stream);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t cublasGetVersion(cublasHandle_t handle, int* version);
const char* cublasGetStatusString(cublasStatus_t status);

cublasStatus_t cublasGemmEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype,
    int lda, const void* B, cudaDataType Btype, int ldb, const void* beta,
    void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType,
    cublasGemmAlgo_t algo);

cublasStatus_t cublasGemmBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha, const void* const Aarray[],
    cudaDataType Atype, int lda, const void* const Barray[], cudaDataType Btype,
    int ldb, const void* beta, void* const Carray[], cudaDataType Ctype,
    int ldc, int batchCount, cublasComputeType_t computeType,
    cublasGemmAlgo_t algo);

cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype,
    int lda, long long strideA, const void* B, cudaDataType Btype, int ldb,
    long long strideB, const void* beta, void* C, cudaDataType Ctype, int ldc,
    long long strideC, int batchCount, cublasComputeType_t computeType,
    cublasGemmAlgo_t algo);

cublasStatus_t cublasGemmGroupedBatchedEx(
    cublasHandle_t handle, const cublasOperation_t transa_array[],
    const cublasOperation_t transb_array[], const int m_array[],
    const int n_array[], const int k_array[], const void* alpha_array,
    const void* const Aarray[], cudaDataType Atype, const int lda_array[],
    const void* const Barray[], cudaDataType Btype, const int ldb_array[],
    const void* beta_array, void* const Carray[], cudaDataType Ctype,
    const int ldc_array[], int group_count, const int group_size[],
    cublasComputeType_t computeType);
