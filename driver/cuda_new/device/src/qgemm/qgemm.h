#pragma once
//
// Fused quantized-weight GEMM (qgemm) — public host entry points.
//
// This is a de-branded, internalized port of the in-register
// dequant + tensor-core MMA fused GEMM pipeline used for weight-only
// quantized linear layers. It computes
//
//     out[M, N] = act[M, K] @ dequant(qweight)^T
//
// where the weight is stored as packed signed-symmetric int4 (GPTQ-style,
// "u4b8" — unsigned 4-bit with an implicit zero point of 8, i.e. the stored
// nibble v maps to the signed value (v - 8)) and per-group bf16 scales.
//
// THIS SLICE instantiates ONLY the bf16-activation x u4b8-weight -> bf16-output
// path. The internal template machinery (qtype tag, dtype map, dequant
// specializations, kernel template) is kept general so that fp8 / mxfp4 / int8
// weight paths can later be added as new dequant specializations + new
// instantiations without restructuring.
//
// Accumulation is fp32; the bf16 output is produced by rounding the fp32
// accumulator.
//
// Namespace: pie_cuda_device::qgemm. No torch types appear in this API.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::qgemm {

// ---------------------------------------------------------------------------
// w4a16 (bf16 activations x u4b8 weights -> bf16 output)
// ---------------------------------------------------------------------------
//
// out[M, N] = act[M, K] @ dequant(qweight_packed)^T
//
// Layout / alignment contract (see qgemm_repack.cu for the prepack layout):
//   * act_bf16       : [M, K] row-major bf16 (nv_bfloat16). Must be 16-byte
//                      aligned; K must be a multiple of 8 (so a row is an
//                      integral number of int4 loads).
//   * qweight_packed : prepacked int4 weights produced by w4a16_repack().
//                      This is NOT the raw checkpoint layout — call
//                      w4a16_repack() first.
//   * scales_bf16    : [num_groups, N] row-major bf16, where
//                          num_groups = (group_size > 0) ? K / group_size : 1.
//                      For group_size == -1 (per-channel) there is a single
//                      group of length K.
//   * out_bf16       : [M, N] row-major bf16.
//   * group_size     : 128 or -1 (per-channel). Other positive group sizes are
//                      accepted if K % group_size == 0 and (group_size % 16)==0,
//                      but only {-1, 128} are exercised by the self-test.
//   * workspace      : device int32 scratch of at least w4a16_workspace_ints().
//                      MUST be zeroed before each call (it holds cross-block
//                      reduction locks; stale values deadlock the kernel).
//   * sms            : number of SMs to launch across; pass 0 to auto-detect.
//
// Alignment constraints (validated against the tile geometry):
//   * K % 16 == 0  and  N % 16 == 0      (tile_size == 16)
//   * N % 64 == 0                         (min_thread_n)
//   * K % 64 == 0                         (min_thread_k)
//   * if group_size > 0: K % group_size == 0
//
// Returns cudaErrorInvalidValue if the shapes violate the contract, otherwise
// the status of the kernel launch.
cudaError_t w4a16_bf16_gemm(cudaStream_t stream,
                            const void*    act_bf16,
                            const int32_t* qweight_packed,
                            const void*    scales_bf16,
                            void*          out_bf16,
                            int M, int N, int K,
                            int group_size,
                            int* workspace,
                            int  sms /* 0 = auto */);

// Required device workspace size, in int32 elements. The workspace holds one
// reduction lock per concurrently-scheduled threadblock; a safe upper bound is
// the number of SMs (one block per SM after the kernel's launch heuristic).
// We return a fixed, generous bound that covers any current GPU so the caller
// can allocate once at init. See definition for the exact formula.
int w4a16_workspace_ints(int N, int max_M);

// Repack plain GPTQ-packed int4 weights into the prepacked tile/interleave
// layout the GEMM kernel expects. See qgemm_repack.cu for the exact expected
// input layout.
//
//   qweight_rowmajor_packed : [K/8, N] int32, GPTQ packed (8 nibbles per int32,
//                             packed along K; nibble j of int32 at (kp, n)
//                             holds the int4 weight for k = kp*8 + j, column n).
//   qweight_out             : [K/16, N*16/8] int32 prepacked output.
//   N, K                    : logical output / reduction dims.
//                             Requires K % 16 == 0 and N % 64 == 0.
cudaError_t w4a16_repack(cudaStream_t stream,
                         const int32_t* qweight_rowmajor_packed,
                         int32_t*       qweight_out,
                         int N, int K);

// ---------------------------------------------------------------------------
// w8a16 (bf16 activations x fe4m3fn fp8 weights -> bf16 output)
// ---------------------------------------------------------------------------
//
// out[M, N] = act[M, K] @ dequant(qweight_fp8_packed)^T
//
// Same fused dequant + tensor-core MMA pipeline as w4a16, but the weight is
// stored as 8-bit fe4m3fn (E4M3) and dequantized to bf16 in-register. The
// per-group bf16 scales are applied with the fp8->bf16 exponent-bias
// correction folded in (handled internally during the scale permute), so the
// caller passes plain logical [num_groups, N] bf16 scales just like w4a16.
//
// Layout / alignment contract:
//   * act_bf16       : [M, K] row-major bf16 (nv_bfloat16), 16-byte aligned;
//                      K a multiple of 8.
//   * qweight_fp8    : prepacked fp8 weights produced by w8a16_fp8_repack().
//                      NOT the raw checkpoint layout — call the repack first.
//   * scales_bf16    : [num_groups, N] row-major bf16, where
//                          num_groups = (group_size > 0) ? K / group_size : 1.
//                      group_size == -1 => single per-channel group of len K.
//   * out_bf16       : [M, N] row-major bf16.
//   * group_size     : 128 or -1 (per-channel). Other positive sizes accepted
//                      if K % group_size == 0 and group_size % 16 == 0.
//   * workspace      : device int32 scratch >= w8a16_fp8_workspace_ints();
//                      MUST be zeroed before each call.
//   * sms            : SMs to launch across; pass 0 to auto-detect.
//
// Alignment constraints (validated): K % 16 == 0, N % 16 == 0, N % 64 == 0,
// K % 64 == 0, and (group_size > 0) ? K % group_size == 0.
cudaError_t w8a16_fp8_bf16_gemm(cudaStream_t stream,
                                const void*    act_bf16,
                                const void*    qweight_fp8,
                                const void*    scales_bf16,
                                void*          out_bf16,
                                int M, int N, int K,
                                int group_size,
                                int* workspace,
                                int  sms /* 0 = auto */);

// Required device workspace size, in int32 elements, for w8a16_fp8_bf16_gemm.
// Same fixed, generous bound as the w4a16 path (one reduction lock per
// concurrently scheduled threadblock; capped at the SM count).
int w8a16_fp8_workspace_ints(int N, int max_M);

// Repack plain row-major fe4m3fn fp8 weights into the prepacked tile /
// interleave layout the GEMM kernel expects (8-bit / pack_factor 4 variant of
// w4a16_repack).
//
//   qweight_rowmajor_packed : [K/4, N] int32, row-major. Each int32 at (kp, n)
//                             packs 4 consecutive fe4m3fn bytes along K:
//                             byte j (bits [8j, 8j+8)) holds the fp8 weight for
//                             k = kp*4 + j, column n.
//   qweight_out             : [K/16, N*16/4] int32 prepacked output.
//   N, K                    : logical output / reduction dims.
//                             Requires K % 16 == 0 and N % 64 == 0.
cudaError_t w8a16_fp8_repack(cudaStream_t stream,
                             const int32_t* qweight_rowmajor_packed,
                             int32_t*       qweight_out,
                             int N, int K);

// ---------------------------------------------------------------------------
// w4a16 mxfp4 (bf16 activations x fe2m1f 4-bit weights, e8m0 block scales)
// ---------------------------------------------------------------------------
//
// out[M, N] = act[M, K] @ dequant(qweight_mxfp4_packed)^T
//
// GPT-OSS microscaling format: the weight is 4-bit fe2m1f (E2M1, sign + 2 exp +
// 1 mantissa), packed two nibbles per byte; each contiguous block of
// MXFP4_BLOCK_SIZE (== 32) weights along K shares a single 1-byte e8m0 (E8M0)
// power-of-two block scale (the stored byte E decodes to the multiplier
// 2^(E - 127)). Because group_size == 32 == 2 * tile_size, the kernel runs with
// group_blocks == 2.
//
// Both the fe2m1f weight decode and the e8m0 scale decode are self-correcting
// in-register (the fe2m1f nibble is splatted to bf16 then re-biased by 2^126;
// the e8m0 byte is placed directly into the bf16 exponent field), so — unlike
// the fp8 path — there is NO host-side exponent-bias fold; the scale permute
// uses bias_mul == 1.
//
// Layout / alignment contract:
//   * act_bf16       : [M, K] row-major bf16 (nv_bfloat16), 16-byte aligned;
//                      K a multiple of 8.
//   * qweight_mxfp4  : prepacked fe2m1f weights produced by mxfp4_repack().
//                      NOT the raw checkpoint layout — call the repack first.
//                      (The repack is bit-identical to w4a16_repack: 4-bit
//                      nibble interleave. fe2m1f and u4b8 differ only in how the
//                      nibble is *decoded*, which is a kernel-side concern.)
//   * scales_e8m0    : [num_groups, N] row-major uint8 e8m0 block scales, where
//                          num_groups = K / MXFP4_BLOCK_SIZE.
//                      The launcher permutes these into the kernel's MMA
//                      fragment layout internally (Marlin 64-wide transpose +
//                      the mxfp4 four-lane [0,2,1,3] byte reorder).
//   * out_bf16       : [M, N] row-major bf16.
//   * workspace      : device int32 scratch >= w4a16_mxfp4_workspace_ints();
//                      MUST be zeroed before each call.
//   * sms            : SMs to launch across; pass 0 to auto-detect.
//
// Alignment constraints (validated): K % 16 == 0, N % 16 == 0, N % 64 == 0,
// K % 64 == 0, and K % MXFP4_BLOCK_SIZE == 0.
static constexpr int MXFP4_BLOCK_SIZE = 32;

cudaError_t w4a16_mxfp4_bf16_gemm(cudaStream_t stream,
                                  const void*    act_bf16,
                                  const void*    qweight_mxfp4,
                                  const void*    scales_e8m0,
                                  void*          out_bf16,
                                  int M, int N, int K,
                                  int* workspace,
                                  int  sms /* 0 = auto */);

// Required device workspace size, in int32 elements, for the mxfp4 GEMM. Same
// fixed, generous bound as the other qgemm paths (one reduction lock per
// concurrently scheduled threadblock; capped at the SM count).
int w4a16_mxfp4_workspace_ints(int N, int max_M);

// Repack plain GPTQ-style packed fe2m1f 4-bit weights into the prepacked
// tile/interleave layout the GEMM kernel expects. Bit-identical to w4a16_repack
// (the nibble interleave does not depend on the nibble's numeric meaning); a
// named entry is provided so callers/ABI are explicit about the format.
//
//   qweight_rowmajor_packed : [K/8, N] int32, 8 fe2m1f nibbles per int32 along
//                             K; nibble j of int32 at (kp, n) holds the fe2m1f
//                             weight for k = kp*8 + j, column n.
//   qweight_out             : [K/16, N*16/8] int32 prepacked output.
//   N, K                    : Requires K % 16 == 0 and N % 64 == 0.
cudaError_t mxfp4_repack(cudaStream_t stream,
                         const int32_t* qweight_rowmajor_packed,
                         int32_t*       qweight_out,
                         int N, int K);

}  // namespace pie_cuda_device::qgemm
