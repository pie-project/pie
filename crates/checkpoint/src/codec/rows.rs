use half::{bf16, f16};

use crate::types::DType;

use super::fp8::fp8_e4m3_to_f32;
#[cfg(target_arch = "x86_64")]
use super::mxfp4::avx2;

pub enum EncodeOperand<'a> {
    Widened {
        bytes: &'a [u8],
        dtype: DType,
    },
    BlockScaledFp8 {
        bytes: &'a [u8],
        factors: Vec<f32>,
        scale_cols: usize,
        group: usize,
        scale_row_offset: usize,
        scale_col_offset: usize,
    },
}

impl EncodeOperand<'_> {
    pub fn row_bf16(&self, row: usize, cols: usize, buf: &mut [f32]) {
        match self {
            EncodeOperand::Widened { bytes, dtype } => {
                let width = dtype.bytes_ceil() as usize;
                let row_bytes = &bytes[row * cols * width..(row + 1) * cols * width];
                match dtype {
                    DType::Bf16 => {
                        #[cfg(target_arch = "x86_64")]
                        if std::arch::is_x86_feature_detected!("avx2") {
                            unsafe { avx2::decode_bf16_row(row_bytes, buf) };
                            return;
                        }
                        for (le, out) in row_bytes.as_chunks::<2>().0.iter().zip(buf.iter_mut()) {
                            let bits = u16::from_le_bytes(*le);
                            *out = f32::from_bits(u32::from(bits) << 16);
                        }
                    }
                    DType::F16 => {
                        for (le, out) in row_bytes.as_chunks::<2>().0.iter().zip(buf.iter_mut()) {
                            let wide = f16::from_bits(u16::from_le_bytes(*le)).to_f32();
                            *out = bf16::from_f32(wide).to_f32();
                        }
                    }
                    DType::F32 => {
                        for (le, out) in row_bytes.as_chunks::<4>().0.iter().zip(buf.iter_mut()) {
                            *out = bf16::from_f32(f32::from_le_bytes(*le)).to_f32();
                        }
                    }
                    _ => unreachable!("EncodeOperand::Widened holds a vetted dtype"),
                }
            }
            EncodeOperand::BlockScaledFp8 {
                bytes,
                factors,
                scale_cols,
                group,
                scale_row_offset,
                scale_col_offset,
            } => {
                let row_bytes = &bytes[row * cols..(row + 1) * cols];
                let scale_row = (scale_row_offset + row / group) * scale_cols;
                for (col, (&code, out)) in row_bytes.iter().zip(buf.iter_mut()).enumerate() {
                    let factor = factors[scale_row + scale_col_offset + col / group];
                    *out = bf16::from_f32(fp8_e4m3_to_f32(code) * factor).to_f32();
                }
            }
        }
    }
}

pub type EncodeRowJob<'a> = dyn Fn(usize, &mut [f32], &mut [u8], &mut [u8]) + Sync + 'a;

pub fn encode_rows(
    rows: usize,
    cols: usize,
    out_row_bytes: usize,
    scale_row_bytes: usize,
    out: &mut [u8],
    scales: &mut [u8],
    job: &EncodeRowJob<'_>,
) {
    let workers = if rows * cols < (1 << 20) {
        1
    } else {
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZero::get)
            .min(rows.max(1))
    };
    if workers <= 1 {
        let mut buf = vec![0.0f32; cols];
        for row in 0..rows {
            let out = &mut out[row * out_row_bytes..(row + 1) * out_row_bytes];
            let scale = &mut scales[row * scale_row_bytes..(row + 1) * scale_row_bytes];
            job(row, &mut buf, out, scale);
        }
        return;
    }
    let rows_per = rows.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut out_rest = out;
        let mut scale_rest = scales;
        let mut start = 0usize;
        while start < rows {
            let count = rows_per.min(rows - start);
            let (out_chunk, next) =
                std::mem::take(&mut out_rest).split_at_mut(count * out_row_bytes);
            out_rest = next;
            let (scale_chunk, next) =
                std::mem::take(&mut scale_rest).split_at_mut(count * scale_row_bytes);
            scale_rest = next;
            let first = start;
            scope.spawn(move || {
                let mut buf = vec![0.0f32; cols];
                for i in 0..count {
                    let out = &mut out_chunk[i * out_row_bytes..(i + 1) * out_row_bytes];
                    let scale = &mut scale_chunk[i * scale_row_bytes..(i + 1) * scale_row_bytes];
                    job(first + i, &mut buf, out, scale);
                }
            });
            start += count;
        }
    });
}
