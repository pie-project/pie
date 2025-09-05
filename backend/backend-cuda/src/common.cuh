#pragma once

#include <thrust/device_vector.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cmath>

// A simple CUDA error-checking macro for robust code.
#define CUDA_CHECK(call)                                                                               \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            throw std::runtime_error(cudaGetErrorString(err));                                         \
        }                                                                                              \
    } while (0)

// Simple macro for checking CUBLAS API calls
#define CUBLAS_CHECK(status)                                    \
    do                                                          \
    {                                                           \
        cublasStatus_t err = (status);                          \
        if (err != CUBLAS_STATUS_SUCCESS)                       \
        {                                                       \
            fprintf(stderr, "cuBLAS error at %s:%d, code=%d\n", \
                    __FILE__, __LINE__, err);                   \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

// Forward declare the datatypes from CUDA headers
struct __half;
struct __nv_bfloat16;

/**
 * @brief Host-side launch function for embedding lookup.
 *
 * This function provides a clean, library-style interface using thrust::device_vector
 * and an explicit CUDA stream for asynchronous execution. It assumes uint32_t indices.
 *
 * @tparam T The base data type (float, __half, __nv_bfloat16).
 * @param embedding The embedding matrix stored as a flat device vector.
 * @param indices A device vector of uint32_t indices to look up.
 * @param result A pointer to a device vector where the output will be stored.
 * @param embed_width The dimensionality of a single embedding vector.
 * @param stream The CUDA stream for asynchronous execution.
 */
template <typename T, typename I>
void embed(
    const T* embedding,
    size_t embedding_num_rows,
    const I* indices,
    size_t num_indices,
    T* result,
    int embed_width,
    cudaStream_t stream);


/**
 * @brief Performs top-k selection on specified rows of a logit tensor and scatters the results.
 * @tparam T The data type of the logits (e.g., __nv_bfloat16 or float).
 * @param logits The flattened device vector containing all batched logits.
 * @param logit_indices_dev A device vector of indices specifying which rows of `logits` to process.
 * @param dest_embed_ids_dev A device vector specifying the destination slot in storage for each row's result.
 * @param vocab_size The number of columns in the logit tensor (the vocabulary size).
 * @param k The number of top elements to select. Must be <= MAX_K.
 * @param topk_probs_storage Output device vector for storing top-k probabilities/scores.
 * @param topk_tokens_storage Output device vector for storing top-k token IDs.
 * @param stream The CUDA stream for the operation.
 */

// template <typename T>
// void gemm_cublasLt2(cublasLtHandle_t ltHandle, cudaStream_t stream, const T *A, const T *B, T *C,
//                     int m, int n, int k, bool transa, bool transb);

// template <typename T>
// void gemm_cublasLt(cublasLtHandle_t ltHandle,
//                    cudaStream_t stream,
//                    const thrust::device_vector<T> &A,
//                    const thrust::device_vector<T> &B,
//                    const thrust::device_vector<T> *bias,
//                    thrust::device_vector<T> &C,
//                    int m, int n, int k,
//                    thrust::device_vector<char> &workspace,
//                    bool transa,
//                    bool transb);

template <typename T>
void gemm_cublasLt(cublasLtHandle_t ltHandle,
                   cudaStream_t stream,
                   const T *d_A,
                   const T *d_B,
                   const T *d_bias,
                   T *d_C,
                   int m, int n, int k,
                   void *d_workspace,
                   size_t workspaceSize,
                   bool transa,
                   bool transb);

void multiply_bf16_cublas(cublasHandle_t handle,
                          const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C,
                          int m, int n, int k, bool transa, bool transb);


template <typename T>
void extract_k_values(const T* A,
                      T* V,
                      int* I,
                      int M,
                      int N,
                      int k,
                      cudaStream_t stream);



template <typename InType, typename OutType>
void cast_type(const InType* d_input, OutType* d_output, size_t n, cudaStream_t stream);