//! Quantization, dequantization and dtype casts.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::operands;
use kernels::Source;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    // 4-bit weights with a bf16 scale per group along K. Distinct from MXFP4
    // (E8M0 byte per 32) and from fp8 -- three quantizations, three
    // statements, because which one a checkpoint ships is a fact the
    // declaration reads.
    kernel!(dequant_wna16_int4b8 "quant::dequant_wna16_int4b8_to_bf16",
        operands = operands![
            packed: I32s,
            scale_bf16: Buf,
            out_bf16: BufMut,
            out_dim: I32,
            in_dim: I32,
            group_size: I32,
            stream: Stream,
        ]),
    kernel!(cast_f32_to_bf16 "quant::cast_fp32_to_bf16",
        operands = operands![
            src_fp32: Buf,
            dst_bf16: BufMut,
            n: Usize,
            stream: Stream,
        ]),
    kernel!(mxfp4_scales_to_marlin "quant::mxfp4_scales_to_marlin_e8m0",
        operands = operands![
            raw_e8m0: Buf,
            marlin_e8m0: BufMut,
            source_rows: I32,
            source_row_offset: I32,
            selected_rows: I32,
            valid_rows: I32,
            source_stride_groups: I32,
            source_group_offset: I32,
            source_groups: I32,
            target_groups: I32,
            row_select: Mxfp4RowSelect,
            stream: Stream,
        ]),
    // Three fp8 forms because the SCALE's shape differs -- per tensor, per
    // output channel, per group along K. A property of the checkpoint, so the
    // declaration states which; a driver that guessed would dequantize
    // correctly on one checkpoint and silently wrongly on another.
    kernel!(dequant_fp8_e4m3 "quant::dequant_fp8_e4m3_to_bf16",
        operands = operands![
            fp8_in: U8s,
            bf16_out: BufMut,
            scale: F32,
            n: Usize,
            stream: Stream,
        ]),
    kernel!(dequant_fp8_e4m3_per_channel "quant::dequant_fp8_e4m3_to_bf16_per_channel",
        operands = operands![
            fp8_in: U8s,
            bf16_out: BufMut,
            scale_inv_dev: F32s,
            rows: I32,
            cols: I32,
            stream: Stream,
        ]),
    kernel!(dequant_fp8_e4m3_per_group "quant::dequant_fp8_e4m3_to_bf16_per_group",
        operands = operands![
            fp8_in: U8s,
            bf16_out: BufMut,
            scale_dev: F32s,
            rows: I32,
            cols: I32,
            group_size: I32,
            stream: Stream,
        ]),
    kernel!(dequant_mxfp4 "quant::dequant_mxfp4_to_bf16",
        operands = operands![
            packed: U8s,
            block_scale: U8s,
            out: BufMut,
            out_dim: I32,
            in_dim: I32,
            stream: Stream,
        ]),
    kernel!(bf16_to_fp16 "quant::bf16_to_fp16",
        operands = operands![
            in_bf16: Buf <- Source::In(0),
            out_fp16: BufMut <- Source::Out(0),
            count: Usize <- Source::OutElements(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(scale_rows "quant::scale_rows_bf16",
        operands = operands![
            buf_bf16: BufMut,
            l_bf16: Buf,
            rows: I32,
            width: I32,
            stream: Stream,
        ]),
    // ── The ENCODING half, which the table was missing while the .cu files
    //    had it all along.
    //
    // `quant/quant_bf16_to_{fp8,mxfp4}.cu` implement runtime quantization and
    // say what for -- `quant_bf16_to_fp8.hpp:6` names the caller outright:
    // "Used by the Rust LoadPlan runtime quantization path: the loader emits
    // an Encode TileMap that reads the source weight, computes absmax, and
    // stores the quantized weight plus scale tensor directly as runtime
    // outputs." No row said so, so nothing could call them, and the loader's
    // Encode ran on the host against kernels sitting unused beside it.
    //
    // Both write TWO outputs -- the payload and the scales it cannot be read
    // without -- which is why `TileMapOp` carries a second destination.
    kernel!(quantize_bf16_to_mxfp4 "quant::quantize_bf16_to_mxfp4_e2m1_per_block",
        operands = operands![
            w_bf16: Buf,
            w_packed: U8sMut,
            w_scale_e8m0: U8sMut,
            rows: I32,
            cols: I32,
            stream: Stream,
        ]),
    kernel!(quantize_bf16_to_fp8_per_channel "quant::quantize_bf16_to_fp8_e4m3_per_channel",
        operands = operands![
            w_bf16: Buf,
            w_fp8: U8sMut,
            scale_inv_dev: F32sMut,
            rows: I32,
            cols: I32,
            stream: Stream,
        ]),
];
