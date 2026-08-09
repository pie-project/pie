// Stub <cublasLt.h> for the gemm oracle. Declarations only — see
// cublas_v2.h beside it.
#pragma once
#include "cublas_v2.h"
#include "cuda_bf16.h"

struct cublasLtContext;
using cublasLtHandle_t = cublasLtContext*;

struct cublasLtMatmulDescOpaque_t;
using cublasLtMatmulDesc_t = cublasLtMatmulDescOpaque_t*;

struct cublasLtMatrixLayoutOpaque_t;
using cublasLtMatrixLayout_t = cublasLtMatrixLayoutOpaque_t*;

struct cublasLtMatmulPreferenceOpaque_t;
using cublasLtMatmulPreference_t = cublasLtMatmulPreferenceOpaque_t*;

struct cublasLtMatmulAlgo_t {
    std::uint64_t data[8];
};

struct cublasLtMatmulHeuristicResult_t {
    cublasLtMatmulAlgo_t algo;
    std::size_t workspaceSize;
    cublasStatus_t state;
    float wavesCount;
    int reserved[4];
};

enum cublasLtMatmulDescAttributes_t {
    CUBLASLT_MATMUL_DESC_TRANSA = 3,
    CUBLASLT_MATMUL_DESC_TRANSB = 4,
    CUBLASLT_MATMUL_DESC_EPILOGUE = 27,
    CUBLASLT_MATMUL_DESC_BIAS_POINTER = 28,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17,
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18,
    CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = 20,
    CUBLASLT_MATMUL_DESC_A_SCALE_MODE = 31,
    CUBLASLT_MATMUL_DESC_B_SCALE_MODE = 32,
    CUBLASLT_MATMUL_DESC_FAST_ACCUM = 29,
};

enum cublasLtMatmulPreferenceAttributes_t {
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
    CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK = 3,
};

enum cublasLtReductionScheme_t {
    CUBLASLT_REDUCTION_SCHEME_NONE = 0,
    CUBLASLT_REDUCTION_SCHEME_INPLACE = 1,
    CUBLASLT_REDUCTION_SCHEME_MASK = 0x7,
};

enum cublasLtMatmulMatrixScale_t {
    CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F = 0,
    CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F = 1,
    CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F = 2,
};

enum cublasLtEpilogue_t {
    CUBLASLT_EPILOGUE_DEFAULT = 1,
    CUBLASLT_EPILOGUE_BIAS = 4,
};

cublasStatus_t cublasLtCreate(cublasLtHandle_t* handle);
cublasStatus_t cublasLtDestroy(cublasLtHandle_t handle);
cublasStatus_t cublasLtMatmulDescCreate(
    cublasLtMatmulDesc_t* desc, cublasComputeType_t computeType,
    cudaDataType_t scaleType);
cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t desc);
cublasStatus_t cublasLtMatmulDescSetAttribute(
    cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attr,
    const void* buf, std::size_t size);
cublasStatus_t cublasLtMatrixLayoutCreate(
    cublasLtMatrixLayout_t* layout, cudaDataType type, std::uint64_t rows,
    std::uint64_t cols, std::int64_t ld);
cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t layout);
cublasStatus_t cublasLtMatmulPreferenceCreate(
    cublasLtMatmulPreference_t* pref);
cublasStatus_t cublasLtMatmulPreferenceDestroy(
    cublasLtMatmulPreference_t pref);
cublasStatus_t cublasLtMatmulPreferenceSetAttribute(
    cublasLtMatmulPreference_t pref,
    cublasLtMatmulPreferenceAttributes_t attr, const void* buf,
    std::size_t size);
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t desc,
    cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulPreference_t pref, int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
    int* returnAlgoCount);
cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t desc, const void* alpha,
    const void* A, cublasLtMatrixLayout_t Adesc, const void* B,
    cublasLtMatrixLayout_t Bdesc, const void* beta, const void* C,
    cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t* algo, void* workspace,
    std::size_t workspaceSizeInBytes, cudaStream_t stream);
