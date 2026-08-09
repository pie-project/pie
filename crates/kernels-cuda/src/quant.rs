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
];
