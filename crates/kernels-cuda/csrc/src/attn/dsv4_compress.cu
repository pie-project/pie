// The launchers for `attn/dsv4_compress.cuh`, and nothing else.
//
// Every `__global__` this file used to hold is in that header now, which this
// file includes: nvcc and NVRTC compile the same text, so there is exactly
// one definition of each kernel in the tree. `new-horizon.md` §10.10 fixes
// the order of that split and says what the second copy costs.
//
// What stays here is the host half -- the extent arithmetic, the empty
// guards, the `CompressedAttnParams` upload and the `<<<>>>`. Six of these
// eleven are `LaunchRule`s the JIT can already state (`Elementwise` and
// `RouteRows`); the other five are named in the header with the geometry that
// has no rule. None of them is deleted: the ahead-of-time path is what every
// caller still uses, and retiring a launcher is a later commit with an A/B
// behind it.
//
// The scalar layer comes out of the prelude: NVRTC has no CUDA device
// headers, and the header this includes has to compile under both it and
// nvcc.
#include "pie_device.cuh"
#include "attn/dsv4_compress.cuh"
#include "attn/dsv4_compress.hpp"

#include <cmath>
#include <vector>

#include "cuda_check.hpp"

namespace pie_cuda_driver::kernels::attn {

namespace {

// The block widths the launchers below spell. They live in the `.cu` and not
// in the header because they are the HOST's arithmetic: under the JIT the
// same numbers are `LaunchRule::Elementwise`'s and `RouteRows`'s, stated once
// in `runtime::launch` instead of once per launcher.
constexpr int BLOCK = 256;
constexpr int ATTN_BLOCK = 128;

}  // namespace

void average_pool_bf16(
    const void* input,
    void* output,
    int N,
    int dim,
    int ratio,
    cudaStream_t stream)
{
    const int out_tokens = N / ratio;
    if (out_tokens <= 0 || dim <= 0) return;
    const int total = out_tokens * dim;
    const int grid = (total + BLOCK - 1) / BLOCK;
    device::average_pool<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(input),
        static_cast<device::bf16*>(output),
        N, dim, ratio);
}

void add_ape_f32(
    void* data,
    const float* ape,
    int N_compressed,
    int dim,
    int ratio,
    cudaStream_t stream)
{
    if (N_compressed <= 0 || dim <= 0) return;
    const int total = N_compressed * dim;
    const int grid = (total + BLOCK - 1) / BLOCK;
    device::add_ape<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<device::bf16*>(data),
        ape,
        N_compressed, dim, ratio);
}

void gated_softmax_pool_bf16(
    const void* kv,
    const void* score,
    void* output,
    int N,
    int dim,
    int ratio,
    cudaStream_t stream)
{
    const int out_tokens = N / ratio;
    if (out_tokens <= 0 || dim <= 0) return;
    const int total = out_tokens * dim;
    const int grid = (total + BLOCK - 1) / BLOCK;
    device::gated_softmax_pool<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(kv),
        static_cast<const device::bf16*>(score),
        static_cast<device::bf16*>(output),
        N, dim, ratio);
}

void combine_attn_outputs_bf16(
    const void* o1, const float* lse1,
    const void* o2, const float* lse2,
    void* o_out, float* lse_out,
    int N, int num_heads, int head_dim,
    cudaStream_t stream)
{
    if (N <= 0) return;
    dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(num_heads));
    // THE BLOCK IS 256 HERE AND `LaunchRule::PerHeadElementwise` CAPS AT 128.
    //
    // The grid is that rule to the digit -- token on `grid.x`, head on
    // `grid.y` -- and the block is not: this clamps `head_dim` into
    // `[32, 256]` and the rule clamps into `[32, 128]`, so on a head wider
    // than 128 the rule answers with half these threads. The kernel strides
    // `d += blockDim.x` and reduces nothing, so the narrower block computes
    // the same bytes in two passes; `runtime::launch`'s own
    // `per_head_elementwise` doc names this launcher as the second one its
    // clamp was written to serve.
    //
    // The row is still not written, and this comment is why. A row cites the
    // launcher it was checked against, and "checked" means the rule returns
    // what the `<<<>>>` returns for the same extents -- which these two do
    // only while `head_dim <= 128`. Rowing it would put a launch in the table
    // that agrees with this line at deepseek_v4's 128-wide heads and stops
    // agreeing at the first config that widens one, and the disagreement
    // would be invisible: a slower kernel, never a wrong answer, so nothing
    // fails and nothing reports. Reconciling it is a decision about
    // `SINK_BLOCK_MAX` in `kernels-cuda-new/src/runtime/launch.rs`, which is
    // not this file's to make.
    const int block = (head_dim < 32) ? 32 : ((head_dim > 256) ? 256 : head_dim);
    device::combine_attn_outputs<device::bf16><<<grid, block, 0, stream>>>(
        static_cast<const device::bf16*>(o1), lse1,
        static_cast<const device::bf16*>(o2), lse2,
        static_cast<device::bf16*>(o_out), lse_out,
        num_heads, head_dim);
}

void attention_compressed_bf16(
    const void* q,
    const void* comp_kv,
    void* o,
    float* lse_out,
    const int* qo_indptr,
    const int* comp_offsets,
    const int* comp_lens,
    const int* comp_ratios,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    int head_dim,
    float sm_scale,
    cudaStream_t stream)
{
    if (num_requests <= 0 || total_tokens <= 0) return;

    // Build params on host, upload to device
    std::vector<device::CompressedAttnParams> params_h(static_cast<std::size_t>(num_requests));
    for (int r = 0; r < num_requests; ++r) {
        params_h[r].qo_lo = qo_indptr[r];
        params_h[r].qo_hi = qo_indptr[r + 1];
        params_h[r].comp_offset = comp_offsets[r];
        params_h[r].comp_len = comp_lens[r];
        params_h[r].comp_ratio = comp_ratios[r];
    }

    // Allocate device memory for params
    device::CompressedAttnParams* params_d = nullptr;
    CUDA_CHECK(cudaMallocAsync(&params_d,
        sizeof(device::CompressedAttnParams) * num_requests, stream));
    CUDA_CHECK(cudaMemcpyAsync(params_d, params_h.data(),
        sizeof(device::CompressedAttnParams) * num_requests,
        cudaMemcpyHostToDevice, stream));

    dim3 grid(num_requests, total_tokens, num_q_heads);
    dim3 block(ATTN_BLOCK);
    const std::size_t smem = (static_cast<std::size_t>(head_dim) + ATTN_BLOCK) * sizeof(float);

    device::compressed_attn<<<grid, block, smem, stream>>>(
        static_cast<const device::bf16*>(q),
        static_cast<const device::bf16*>(comp_kv),
        static_cast<device::bf16*>(o),
        lse_out,
        params_d,
        num_q_heads, head_dim, sm_scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(params_d, stream));
}


void dsv4_compress_gather_bf16(
    const void* kv_proj,
    const void* score_proj,
    const float* ape,
    const device::i32* boundary_tok,
    const device::i32* boundary_pos,
    const device::i32* window_lo,
    void* out,
    int num_entries,
    int head_dim,
    int ratio,
    int coff,
    cudaStream_t stream) {
    if (num_entries <= 0 || head_dim <= 0 || ratio <= 0 || coff <= 0) return;
    const int threads = head_dim < BLOCK ? ((head_dim + 31) / 32) * 32 : BLOCK;
    device::dsv4_compress_gather<device::bf16><<<num_entries, threads, 0, stream>>>(
        static_cast<const device::bf16*>(kv_proj),
        static_cast<const device::bf16*>(score_proj),
        ape,
        boundary_tok,
        boundary_pos,
        window_lo,
        static_cast<device::bf16*>(out),
        head_dim,
        ratio,
        coff);
    CUDA_CHECK(cudaGetLastError());
}


void dsv4_compress_gather_paged_bf16(
    const void* state_kv,
    const void* state_score,
    const float* ape,
    const device::i32* boundary_pos,
    const device::i32* boundary_req,
    const device::u32* kv_page_indices,
    const device::u32* kv_page_indptr,
    void* out,
    int num_entries,
    int head_dim,
    int ratio,
    int coff,
    int page_size,
    cudaStream_t stream) {
    if (num_entries <= 0 || head_dim <= 0 || ratio <= 0 || coff <= 0) return;
    const int threads = head_dim < BLOCK ? ((head_dim + 31) / 32) * 32 : BLOCK;
    device::dsv4_compress_gather_paged<device::bf16><<<num_entries, threads, 0, stream>>>(
        static_cast<const device::bf16*>(state_kv),
        static_cast<const device::bf16*>(state_score),
        ape, boundary_pos, boundary_req, kv_page_indices, kv_page_indptr,
        static_cast<device::bf16*>(out),
        head_dim, ratio, coff, page_size);
    CUDA_CHECK(cudaGetLastError());
}

void dsv4_boundary_meta_decode(
    const device::i32* positions,
    device::i32* out_pos,
    device::i32* out_req,
    device::i32* out_rope,
    int n,
    int ratio,
    cudaStream_t stream,
    const device::u8* row_valid) {
    if (n <= 0 || ratio <= 0) return;
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    device::dsv4_boundary_meta_decode<<<blocks, threads, 0, stream>>>(
        positions, out_pos, out_req, out_rope, n, ratio, row_valid);
    CUDA_CHECK(cudaGetLastError());
}

void dsv4_boundary_meta_paged(
    const device::i32* positions,
    const device::u32* qo_indptr,
    device::i32* out_pos,
    device::i32* out_req,
    device::i32* out_rope,
    int n,
    int num_requests,
    int ratio,
    cudaStream_t stream,
    const device::u8* row_valid) {
    if (n <= 0 || ratio <= 0) return;
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    device::dsv4_boundary_meta_paged<<<blocks, threads, 0, stream>>>(
        positions, qo_indptr, out_pos, out_req, out_rope, n, num_requests, ratio,
        row_valid);
    CUDA_CHECK(cudaGetLastError());
}

void dsv4_store_comp_entries_bf16(
    const void* entries,
    void* comp_kv_pages,
    const device::i32* boundary_pos,
    const device::i32* boundary_req,
    const device::u32* kv_page_indices,
    const device::u32* kv_page_indptr,
    int num_entries,
    int head_dim,
    int page_size,
    cudaStream_t stream) {
    if (num_entries <= 0 || head_dim <= 0) return;
    const int threads = head_dim < BLOCK ? ((head_dim + 31) / 32) * 32 : BLOCK;
    device::dsv4_store_comp_entries<device::bf16><<<num_entries, threads, 0, stream>>>(
        static_cast<const device::bf16*>(entries),
        static_cast<device::bf16*>(comp_kv_pages),
        boundary_pos, boundary_req, kv_page_indices, kv_page_indptr,
        head_dim, page_size);
    CUDA_CHECK(cudaGetLastError());
}

void attention_compressed_paged_bf16(
    const void* q,
    const void* comp_kv_pages,
    void* o,
    float* lse_out,
    const device::i32* positions,
    const device::u32* /*qo_indptr*/,
    const device::u32* kv_page_indices,
    const device::u32* kv_page_indptr,
    const device::i32* req_of_token,
    int total_tokens,
    int num_q_heads,
    int head_dim,
    int ratio,
    int page_size,
    float sm_scale,
    cudaStream_t stream) {
    if (total_tokens <= 0 || num_q_heads <= 0) return;
    dim3 grid(static_cast<unsigned>(total_tokens),
              static_cast<unsigned>(num_q_heads));
    const std::size_t smem =
        (static_cast<std::size_t>(head_dim) + ATTN_BLOCK) * sizeof(float);
    device::compressed_attn_paged<<<grid, ATTN_BLOCK, smem, stream>>>(
        static_cast<const device::bf16*>(q),
        static_cast<const device::bf16*>(comp_kv_pages),
        static_cast<device::bf16*>(o), lse_out,
        positions, kv_page_indices, kv_page_indptr, req_of_token,
        num_q_heads, head_dim, ratio, page_size, sm_scale);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace pie_cuda_driver::kernels::attn
