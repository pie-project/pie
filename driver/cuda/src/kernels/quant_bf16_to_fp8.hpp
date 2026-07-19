#pragma once

// Runtime quantization helpers — convert bf16 weight tensors to FP8
// (E4M3) or INT8 with per-tensor / per-channel symmetric scaling.
// Used by the Rust LoadPlan runtime quantization path
// (`--runtime-quant {fp8|int8}`): the loader emits an Encode TileMap that
// reads the source weight, computes absmax, and stores the quantized weight
// plus scale tensor directly as runtime outputs.
//
// The scale convention matches what `ops::gemm_act_x_w` expects from a
// `QuantMeta` — cuBLASLt multiplies operand A by the scale pointer
// before the matmul, so storing `weight_scale_inv` (not its reciprocal)
// lets the dispatcher hand the same float to cuBLASLt unchanged. INT8
// follows the same convention.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

/// Per-channel symmetric INT8 quantization along axis 0 of a row-major
/// `[rows, cols]` weight. Mirrors the FP8 per-channel kernel: one block
/// per row, threads cooperatively absmax → scale → cast.
///   * `W_int8`        — `[rows, cols]` int8 (signed -128..127)
///   * `scale_inv_dev` — `[rows]` fp32 such that
///                       `bf16[i,j] ≈ int8[i,j] * scale_inv_dev[i]`
///
/// Symmetric: `scale_inv_row = absmax / 127`. Range outliers are
/// clamped to ±127.
void quantize_bf16_to_int8_per_channel(
    const void*    W_bf16,
    std::int8_t*   W_int8,
    float*         scale_inv_dev,    // [rows]
    int            rows,
    int            cols,
    cudaStream_t   stream);

/// Per-token symmetric INT8 activation quantization. Acts on a row-
/// major `[N, K]` activation buffer; produces one `act_scale_inv` per
/// row (per token). Used by the W8A8 GEMM dispatch:
///   bf16[i,j] ≈ int8[i,j] * act_scale_inv[i]
/// Symmetric: `act_scale_inv_row = absmax_row / 127`. Outliers clamped.
/// Done in one kernel pass per row: warp-level absmax reduction →
/// scale → cast.
void quantize_bf16_to_int8_per_token(
    const void*    act_bf16,      // [N, K] bf16
    std::int8_t*   act_int8,      // [N, K] int8
    float*         act_scale_inv, // [N] fp32
    int            n_tokens,
    int            k,
    cudaStream_t   stream);

/// In-place INT8 variant of `launch_absmax_to_scale_inv`: divides each
/// per-row absmax by INT8_MAX (127) instead of FP8_MAX. Used to split
/// `quantize_bf16_to_int8_per_channel` into stages so a TP all-reduce
/// can run between absmax and cast for row-parallel weights.
void launch_absmax_to_scale_inv_int8(
    float*        absmax_inout,   // [rows]
    int           rows,
    cudaStream_t  stream);

/// Stage 2 of split INT8 weight quant: cast `[rows, cols]` bf16 → int8
/// using a precomputed per-row scale_inv (one scale per output channel).
/// Pairs with `launch_absmax_per_row_bf16` + `launch_absmax_to_scale_inv_int8`.
void launch_cast_bf16_to_int8_per_channel(
    const void*   W_bf16,
    std::int8_t*  W_int8,
    const float*  scale_inv_dev, // [rows]
    int           rows,
    int           cols,
    cudaStream_t  stream);

/// Correctness fallback for runtime INT8 weights when cuBLAS cannot run
/// W8A8 for a shape (notably tiny decode/parity M). Dequantizes a row-major
/// `[rows, cols]` INT8 weight with per-row scales into BF16 scratch.
void launch_dequant_int8_to_bf16_per_channel(
    const std::int8_t* W_int8,
    void*             W_bf16,
    const float*      scale_inv_dev, // [rows]
    int               rows,
    int               cols,
    cudaStream_t      stream);

/// Post-GEMM W8A8 dequant: convert `[M, N] int32` accumulator to bf16
/// using per-row activation scales (`[M]`) and per-row weight scales
/// (`[N]`). Output:
///   bf16[m,n] = int32[m,n] * act_scale_inv[m] * w_scale_inv[n]
/// (Plus optional bias add — not yet supported here; the bias path
/// can layer on top via the existing residual-add kernel.)
void dequant_int32_w8a8_to_bf16(
    const std::int32_t*  acc_int32,        // [M, N]
    const float*         act_scale_inv,    // [M]
    const float*         w_scale_inv,      // [N]
    void*                out_bf16,         // [M, N]
    int                  M,
    int                  N,
    cudaStream_t         stream);

/// Compute `absmax(W)` over `n` bf16 elements into a single fp32 device
/// scalar. Caller allocates the 4-byte output buffer.
void launch_absmax_bf16(
    const void* W_bf16,    // [n] bf16
    float*      absmax_dev,// device scalar
    std::size_t n,
    cudaStream_t stream);

/// Cast `W_bf16` to `W_fp8 = clamp(round(W / scale), [-fp8_max, fp8_max])`.
/// `scale_inv = 1 / scale` is precomputed on the host (so the per-element
/// kernel does a single multiply, not a divide). The forward GEMM then
/// reconstructs bf16 via `W_fp8 * scale` using cuBLASLt's A_SCALE_POINTER.
void launch_quant_bf16_to_fp8_e4m3(
    const void*         W_bf16,    // [n] bf16
    std::uint8_t*       W_fp8,     // [n] fp8 raw bytes
    float               scale_inv, // host scalar (1 / weight_scale_inv)
    std::size_t         n,
    cudaStream_t        stream);

/// Convenience: runs the absmax pass + cast pass with one device sync,
/// returning the chosen `weight_scale_inv` (the multiplicative factor
/// such that bf16 ≈ fp8 * weight_scale_inv) on host.
float quantize_bf16_to_fp8_e4m3_per_tensor(
    const void*    W_bf16,
    std::uint8_t*  W_fp8,
    std::size_t    n,
    cudaStream_t   stream);

/// Per-channel symmetric FP8 quantization along axis 0 of a row-major
/// `[rows, cols]` weight. Computes one `weight_scale_inv` per row
/// (i.e. per output channel for the standard `[N, K]` projection
/// layout). Output:
///   * `W_fp8`           — `[rows, cols]` raw FP8 bytes
///   * `scale_inv_dev`   — `[rows]` fp32 scales such that
///                         `bf16[i,j] ≈ fp8[i,j] * scale_inv_dev[i]`
///
/// The fp8 GEMM dispatcher reads `scale_inv_dev` directly via the
/// `QuantMeta::PerChannel` path: cuBLASLt's vector-scale attr on H100,
/// or the dequant fallback's broadcasted multiplication on older arch.
void quantize_bf16_to_fp8_e4m3_per_channel(
    const void*    W_bf16,
    std::uint8_t*  W_fp8,
    float*         scale_inv_dev,    // [rows]
    int            rows,
    int            cols,
    cudaStream_t   stream);

/// Stage 1 of TP-aware per-channel FP8 quantization: compute `[rows]`
/// per-row absmax of a `[rows, cols]` bf16 weight into `absmax_dev`.
/// The caller is expected to follow this with a cross-rank all-reduce
/// (MAX op) for row-parallel weights, then call `cast_bf16_to_fp8_e4m3
/// _per_channel` to finish.
void launch_absmax_per_row_bf16(
    const void*    W_bf16,
    float*         absmax_dev,       // [rows]
    int            rows,
    int            cols,
    cudaStream_t   stream);

/// In-place: convert `[rows]` per-row absmax values to `weight_scale_inv
/// = absmax / fp8_max`. Zero-rows produce scale_inv = 1 (no-op cast).
void launch_absmax_to_scale_inv(
    float*         absmax_inout,     // [rows]
    int            rows,
    cudaStream_t   stream);

/// Stage 2: cast `[rows, cols]` bf16 → fp8 using `[rows]` weight_scale_inv
/// (one scale per row). Equivalent to dividing each element by its row's
/// scale before casting. Pairs with `launch_absmax_per_row_bf16` +
/// `launch_absmax_to_scale_inv`.
void launch_cast_bf16_to_fp8_e4m3_per_channel(
    const void*    W_bf16,
    std::uint8_t*  W_fp8,
    const float*   scale_inv_dev,    // [rows]
    int            rows,
    int            cols,
    cudaStream_t   stream);

}  // namespace pie_cuda_driver::kernels
