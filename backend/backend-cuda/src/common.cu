#include "common.cuh"
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <type_traits>
#include <limits> // Required for std::numeric_limits
#include <cooperative_groups.h> // For warp-level operations

/**
 * @brief High-performance CUDA kernel for embedding lookup.
 *
 * This version is specialized for uint32_t indices and uses 128-bit
 * vectorized memory operations for maximum bandwidth.
 *
 * @tparam T The base data type (float, __half, etc.).
 */
template <typename T, typename I>
__global__ void embedding_lookup_kernel_128bit(T *output,
                                               const T *embedding_matrix,
                                               const I *indices,
                                               int n,
                                               int hidden_dim_div_16)
{
    // Each block processes one lookup index.
    int idx_n = blockIdx.x;
    if (idx_n >= n)
    {
        return;
    }

    // Use shared memory to broadcast the source row index for the block.
    __shared__ I source_row_idx;
    if (threadIdx.x == 0)
    {
        source_row_idx = indices[idx_n];
    }
    __syncthreads();

    // Cast pointers to a 128-bit type (float4) to perform 16-byte memory transfers.
    // This is the core optimization, significantly increasing memory throughput.
    const float4 *source_row_ptr = reinterpret_cast<const float4 *>(embedding_matrix) + (long long)source_row_idx * hidden_dim_div_16;
    float4 *dest_row_ptr = reinterpret_cast<float4 *>(output) + (long long)idx_n * hidden_dim_div_16;

    // Use a grid-stride loop for threads to collectively copy the entire row.
    // This ensures that all data is copied regardless of the number of threads per block.
    for (int i = threadIdx.x; i < hidden_dim_div_16; i += blockDim.x)
    {
        dest_row_ptr[i] = source_row_ptr[i];
    }
}

/**
 * @brief Host-side launch function with a Thrust-based API.
 *
 * This function provides a clean, library-style interface using thrust::device_vector
 * and an explicit CUDA stream for asynchronous execution. It assumes uint32_t indices.
 *
 * @tparam T The base data type (float, __half, etc.).
 */
/**
 * @brief Host-side launch function for embedding lookup with a raw pointer API.
 */
template <typename T, typename I>
void embed(
    const T* embedding,
    size_t embedding_num_rows,
    const I* indices,
    size_t num_indices,
    T* result,
    int embed_width,
    cudaStream_t stream)
{
    // --- Input Validation ---
    if (embedding_num_rows == 0 || num_indices == 0) return;
    if ((embedding_num_rows * embed_width) == 0) return;
    if ((embedding_num_rows * embed_width) % embed_width != 0) {
        throw std::invalid_argument("Embedding vector size is not divisible by the embed_width.");
    }
    if ((embed_width * sizeof(T)) % 16 != 0) {
        throw std::invalid_argument("Total byte size of a slice (embed_width * sizeof(T)) must be a multiple of 16.");
    }

    // --- Prepare Parameters ---
    // The result buffer is assumed to be pre-allocated by the caller.
    const int threads_per_block = 256;
    const int hidden_dim_div_16 = (embed_width * sizeof(T)) / 16;

    dim3 blocks(num_indices);
    dim3 threads(threads_per_block);

    // --- Kernel Launch ---
    embedding_lookup_kernel_128bit<T, I><<<blocks, threads, 0, stream>>>(
        result,
        embedding,
        indices,
        num_indices,
        hidden_dim_div_16);
}


/**
 * @brief CUDA Kernel: Extracts k non-negative-infinity values and their indices from each row.
 *
 * This kernel assigns one thread per row. Each thread scans its assigned row of the input
 * matrix, finds the first k valid (not -inf) elements, and writes their values and
 * column indices to the corresponding output matrices.
 *
 * @tparam T The data type of the matrix (e.g., float, __nv_bfloat16).
 * @param A Input dense matrix of size M x N on the device.
 * @param V Output value matrix of size M x k on the device.
 * @param I Output index matrix of size M x k on the device.
 * @param M Number of rows in the input matrix.
 * @param N Number of columns in the input matrix.
 * @param k The exact number of non-negative-infinity elements to extract from each row.
 */
template <typename T>
__global__ void extract_k_values_kernel(const T* __restrict__ A,
                                        T* __restrict__ V,
                                        int32_t* __restrict__ I,
                                        int M,
                                        int N,
                                        int k)
{
    // --- Threading Model: One Block per Row ---
    // The block index corresponds to the row index.
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // Each block uses a shared memory counter to coordinate writes.
    __shared__ int output_count;

    // Thread 0 of each block initializes its counter.
    if (tid == 0)
    {
        output_count = 0;
    }
    __syncthreads();

    // Pointers for the current row, calculated once per block.
    const T* input_row = A + (long long)row_idx * N;
    T* value_output_row = V + (long long)row_idx * k;
    int32_t* index_output_row = I + (long long)row_idx * k;

    const T neg_inf = static_cast<T>(-INFINITY);

    // --- Parallel Scan through the Row ---
    // Threads in the block scan the row in parallel chunks.
    for (int col_base = 0; col_base < N; col_base += blockDim.x)
    {
        // Early exit for the entire block if k elements have been found.
        // Reading a shared variable is much faster than continuing the loop.
        if (output_count >= k)
        {
            break;
        }

        const int col_idx = col_base + tid;

        // Boundary check for the last chunk of the row.
        if (col_idx < N)
        {
            // This read is coalesced: adjacent threads read adjacent memory addresses.
            T val = input_row[col_idx];

            if (val != neg_inf)
            {
                // Atomically increment the shared counter to get a unique, sequential
                // index for this thread to write its result to.
                int write_idx = atomicAdd(&output_count, 1);

                // If this thread's result is within the top k, write it out.
                if (write_idx < k)
                {
                    value_output_row[write_idx] = val;
                    index_output_row[write_idx] = col_idx;
                }
            }
        }
    }
}

/**
 * @brief Extracts the first k valid (non -inf) values/indices per row from a sparse matrix.
 *
 * This utility function launches a kernel to convert a sparse matrix, where "empty"
 * values are represented by -infinity, into two dense matrices: one for the k values
 * and one for their corresponding column indices. It operates entirely on device memory.
 *
 * @tparam T The data type of the matrix (e.g., float, __nv_bfloat16).
 * @param A Device pointer to the input dense matrix (size M*N).
 * @param V Device pointer to the output value matrix (size M*k).
 * @param I Device pointer to the output index matrix (size M*k).
 * @param M The number of rows.
 * @param N The number of columns.
 * @param k The number of non-negative-infinity elements to find in each row.
 * @param stream The CUDA stream for asynchronous execution.
 */
template <typename T>
void extract_k_values(const T* A,
                      T* V,
                      int32_t* I,
                      int M,
                      int N,
                      int k,
                      cudaStream_t stream)
{
    // --- Input Validation ---
    if (M <= 0 || N <= 0 || k <= 0)
    {
        return; // Nothing to do
    }

    // --- Kernel Launch Configuration ---
    // A block size of 256 is a good, general-purpose choice.
    // It should be a multiple of the warp size (32).
    const int threads_per_block = 256;

    // Launch one block for each row.
    const int blocks_per_grid = M;

    dim3 grid(blocks_per_grid);
    dim3 threads(threads_per_block);

    // --- Launch Kernel ---
    extract_k_values_kernel<T><<<grid, threads, 0, stream>>>(A, V, I, M, N, k);
}


/**
 * @brief CUDA kernel to perform element-wise type conversion.
 *
 * This kernel is a template that can handle various input and output types.
 * Each thread processes one element of the input array and stores the
 * converted value in the output array.
 *
 * @tparam InType The data type of the input array.
 * @tparam OutType The data type of the output array.
 * @param input Pointer to the input array on the device.
 * @param output Pointer to the output array on the device.
 * @param n The total number of elements in the arrays.
 */
template <typename InType, typename OutType>
__global__ void cast_type_kernel(const InType* input, OutType* output, size_t n) {
    // Calculate the global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check to ensure we don't access out of bounds
    if (idx < n) {
        // Perform the type conversion using a static_cast
        output[idx] = static_cast<OutType>(input[idx]);
    }
}

/**
 * @brief Host-side launch function for generic type conversion on the GPU.
 *
 * This function handles the kernel launch for converting a device array
 * from one data type to another. It assumes input and output buffers are
 * already allocated on the device.
 *
 * @tparam InType The data type of the input device array.
 * @tparam OutType The data type for the output device array.
 * @param d_input Pointer to the input array on the device.
 * @param d_output Pointer to the output array on the device.
 * @param n The number of elements to convert.
 * @param stream The CUDA stream for asynchronous execution.
 */
template <typename InType, typename OutType>
void cast_type(const InType* d_input, OutType* d_output, size_t n, cudaStream_t stream) {
    if (n == 0) {
        return;
    }

    // Define kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // Launch the conversion kernel
    cast_type_kernel<InType, OutType><<<blocks_per_grid, threads_per_block, 0, stream>>>(d_input, d_output, n);
}


// --- Explicit Template Instantiations ---
// We explicitly instantiate the templates for the supported types. This forces
// the compiler to generate the code for each of these types, which will then
// be linked against when another file includes embedding.h.

// template void embed<float>(
//     const thrust::device_vector<float> &,
//     const thrust::device_vector<uint32_t> &,
//     thrust::device_vector<float> *,
//     int, cudaStream_t);

// template void embed<__half>(
//     const thrust::device_vector<__half> &,
//     const thrust::device_vector<uint32_t> &,
//     thrust::device_vector<__half> *,
//     int, cudaStream_t);

// template void embed<__nv_bfloat16>(
//     const thrust::device_vector<__nv_bfloat16> &,
//     const thrust::device_vector<uint32_t> &,
//     thrust::device_vector<__nv_bfloat16> *,
//     int, cudaStream_t);

template void embed<float, int32_t>(
    const float*,
    size_t,
    const int32_t*,
    size_t,
    float* result,
    int,
    cudaStream_t);

template void embed<__nv_bfloat16, int32_t>(
    const __nv_bfloat16*,
    size_t,
    const int32_t*,
    size_t,
    __nv_bfloat16* result,
    int,
    cudaStream_t);


template void extract_k_values<float>(
    const float*,
    float*,
    int32_t*,
    int, int, int,
    cudaStream_t);

template void extract_k_values<__nv_bfloat16>(
    const __nv_bfloat16*,
    __nv_bfloat16*,
    int32_t*,
    int, int, int,
    cudaStream_t);


// Explicit template instantiations for convert_type_on_device
template void cast_type<__nv_bfloat16, float>(const __nv_bfloat16*, float*, size_t, cudaStream_t);
template void cast_type<float, __nv_bfloat16>(const float*, __nv_bfloat16*, size_t, cudaStream_t);
template void cast_type<__half, float>(const __half*, float*, size_t, cudaStream_t);
template void cast_type<float, __half>(const float*, __half*, size_t, cudaStream_t);
template void cast_type<__nv_bfloat16, __half>(const __nv_bfloat16*, __half*, size_t, cudaStream_t);
template void cast_type<__half, __nv_bfloat16>(const __half*, __nv_bfloat16*, size_t, cudaStream_t);



template <typename T>
constexpr cudaDataType_t get_cuda_data_type()
{
    if constexpr (std::is_same_v<T, float>)
    {
        return CUDA_R_32F;
    }
    else if constexpr (std::is_same_v<T, __half>)
    {
        return CUDA_R_16F;
    }
#if __CUDACC_VER_MAJOR__ >= 11
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return CUDA_R_16BF;
    }
#endif
    else if constexpr (std::is_same_v<T, double>)
    {
        return CUDA_R_64F;
    }
    else if constexpr (std::is_same_v<T, int8_t>)
    {
        return CUDA_R_8I;
    }
    // Add other types here as needed...
    else
    {
        // This will produce a compile-time error if an unsupported type is used.
        static_assert(std::is_same_v<T, void>, "Unsupported data type for gemm_cublasLt_improved");
        return CUDA_R_32F; // Dummy return to satisfy compiler
    }
}

/**
 * @brief Performs General Matrix Multiplication (GEMM) using cuBLASLt with raw device pointers.
 * * This function computes C = alpha * op(A) * op(B) + beta * C.
 * It uses a strategy of swapping A and B to handle column-major layout requirements of cuBLAS
 * while allowing the caller to think in terms of row-major layouts.
 * * @tparam T The data type of the matrices (e.g., float, __nv_bfloat16).
 * @param ltHandle The cuBLASLt library handle.
 * @param stream The CUDA stream for the operation.
 * @param d_A Pointer to matrix A on the device.
 * @param d_B Pointer to matrix B on the device.
 * @param d_bias Pointer to the bias vector on the device (can be nullptr).
 * @param d_C Pointer to the output matrix C on the device.
 * @param m The number of rows of matrix op(A) and C.
 * @param n The number of columns of matrix op(B) and C.
 * @param k The number of columns of op(A) and rows of op(B).
 * @param d_workspace Pointer to the workspace buffer on the device.
 * @param workspaceSize The size of the workspace buffer in bytes.
 * @param transa Specifies if matrix A should be transposed.
 * @param transb Specifies if matrix B should be transposed.
 */
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
                   bool transb)
{
    if (m <= 0 || n <= 0 || k <= 0)
    {
        return;
    }

    // --- Scaling Factors ---
    float alpha = 1.0f;
    float beta = (d_bias != nullptr) ? 1.0f : 0.0f;

    // --- Descriptors for cuBLASLt ---
    cublasLtMatmulDesc_t matmulDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    // --- Data and Compute Type Configuration ---
    cudaDataType_t cuda_dtype = get_cuda_data_type<T>();
    cublasComputeType_t compute_type;
    cudaDataType_t scale_type = CUDA_R_32F; // Use FP32 for alpha/beta for precision

    if (std::is_same<T, float>::value)
    {
        compute_type = CUBLAS_COMPUTE_32F_FAST_TF32; // Use TF32 for float GEMM
    }
    else
    {
        compute_type = CUBLAS_COMPUTE_32F; // Accumulate in FP32 for mixed-precision
    }

    // --- Core Correction using (A*B)^T = op(B)^T * op(A)^T strategy ---
    // We ask cuBLAS to compute C_col(n,m) = op(B)^T_col(n,k) * op(A)^T_col(k,m).
    cublasOperation_t opA_swapped = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB_swapped = transb ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Create the Matmul Descriptor with the swapped & transformed operations.
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, compute_type, scale_type));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB_swapped, sizeof(opB_swapped)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA_swapped, sizeof(opA_swapped)));

    // Create matrix layouts. Note that the dimensions are for the *swapped* matrices.
    // A (now B) has dimensions (n, k)
    if (transb)
    {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, cuda_dtype, k, n, k));
    }
    else
    {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, cuda_dtype, n, k, n));
    }

    // B (now A) has dimensions (k, m)
    if (transa)
    {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, cuda_dtype, m, k, m));
    }
    else
    {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, cuda_dtype, k, m, k));
    }

    // C has dimensions (n, m)
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, cuda_dtype, n, m, n));

    // Configure Epilogue (Bias Addition)
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    if (d_bias != nullptr)
    {
        epilogue = CUBLASLT_EPILOGUE_BIAS;
        void *d_bias_nonconst = const_cast<void *>(static_cast<const void *>(d_bias));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias_nonconst, sizeof(d_bias_nonconst)));
    }
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    // --- Algorithm Heuristics ---
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // Note the order of descriptors: Adesc (for B), Bdesc (for A), Cdesc
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0)
    {
        fprintf(stderr, "Error: No suitable cuBLASLt algorithm found!\n");
    }
    else
    {
        // Execute the Matmul. Note the order of pointers: d_B, d_A, d_C
        CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc, &alpha,
                                    d_B, Adesc, // First matrix is B
                                    d_A, Bdesc, // Second matrix is A
                                    &beta,
                                    d_C, Cdesc,
                                    d_C, Cdesc, // D is the same as C for this operation
                                    &heuristicResult.algo, d_workspace, workspaceSize, stream));
    }

    // --- Cleanup ---
    if (preference) CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc) CUBLAS_CHECK(cublasLtMatmulDescDestroy(matmulDesc));
}

// --- Explicit Template Instantiations for raw pointer version ---

template void gemm_cublasLt<__nv_bfloat16>(cublasLtHandle_t, cudaStream_t,
                                           const __nv_bfloat16 *, const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *,
                                           int, int, int,
                                           void *, size_t,
                                           bool, bool);

template void gemm_cublasLt<float>(cublasLtHandle_t, cudaStream_t,
                                   const float *, const float *, const float *, float *,
                                   int, int, int,
                                   void *, size_t,
                                   bool, bool);

void multiply_bf16_cublas(cublasHandle_t handle,
                          const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C,
                          int m, int n, int k, bool transa, bool transb)
{

    // Use FP32 for accumulation to preserve precision.
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasOperation_t opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda = transa ? m : k; // Leading dimension
    int ldb = transb ? k : n; // Leading dimension
    int ldc = n;              // Leading dimension for C

    CUBLAS_CHECK(cublasGemmEx(handle,
                              opB,
                              opA,
                              n,
                              m,
                              k,
                              &alpha,
                              B, CUDA_R_16BF, ldb,
                              A, CUDA_R_16BF, lda,
                              &beta,
                              C, CUDA_R_16BF, ldc,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
