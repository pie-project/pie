//! Encoding a 2-D tensor row by row, in parallel.
//!
//! The engine over the per-format group encoders. It is here rather than in
//! the walker because it knows nothing about a plan: it takes counts, two
//! flat output buffers and a closure, and the only reason it exists is that
//! a row's bytes depend on that row alone.

use half::{bf16, f16};

use crate::types::DType;

use super::fp8::fp8_e4m3_to_f32;
// Gated the same way the module is: `mxfp4::avx2` is `#[cfg(x86_64)]`, and
// its one caller below already sits behind that cfg. Without the gate this
// import is an unresolved name on every other target — aarch64 included,
// which is every Metal host.
#[cfg(target_arch = "x86_64")]
use super::mxfp4::avx2;

/// How an Encode reads its operand as `BF16` rows.
///
/// Resolved once, before the row loop, so the per-row work is indexing and
/// arithmetic only — which is also what lets the rows run on any thread.
pub enum EncodeOperand<'a> {
    /// A raw operand, narrowed to `BF16` element by element the way the
    /// device's cast does.
    Widened { bytes: &'a [u8], dtype: DType },
    /// An FP8 payload and the `F32` block factors that make it numbers,
    /// multiplied out per element: `bf16(f32(fp8) · factor)`, the blocked
    /// dequant kernel's expression.
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
    /// Fill `buf` with row `row` as the `f32` widening of its `BF16` reading.
    pub fn row_bf16(&self, row: usize, cols: usize, buf: &mut [f32]) {
        match self {
            EncodeOperand::Widened { bytes, dtype } => {
                let width = dtype.bytes_ceil() as usize;
                let row_bytes = &bytes[row * cols * width..(row + 1) * cols * width];
                match dtype {
                    DType::Bf16 => {
                        #[cfg(target_arch = "x86_64")]
                        if std::arch::is_x86_feature_detected!("avx2") {
                            // Sound: the feature was just detected, and the
                            // slices agree on length by the arm's slicing.
                            unsafe { avx2::decode_bf16_row(row_bytes, buf) };
                            return;
                        }
                        for (le, out) in row_bytes.chunks_exact(2).zip(buf.iter_mut()) {
                            // BF16 widens by a shift; spelled directly rather
                            // than through `half` so the decode vectorizes.
                            let bits = u16::from_le_bytes(le.try_into().unwrap());
                            *out = f32::from_bits(u32::from(bits) << 16);
                        }
                    }
                    DType::F16 => {
                        for (le, out) in row_bytes.chunks_exact(2).zip(buf.iter_mut()) {
                            let wide =
                                f16::from_bits(u16::from_le_bytes(le.try_into().unwrap())).to_f32();
                            *out = bf16::from_f32(wide).to_f32();
                        }
                    }
                    DType::F32 => {
                        for (le, out) in row_bytes.chunks_exact(4).zip(buf.iter_mut()) {
                            *out =
                                bf16::from_f32(f32::from_le_bytes(le.try_into().unwrap())).to_f32();
                        }
                    }
                    // `encode_bytes` admitted only the three above.
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

/// One row of an encode: `(row, f32 scratch, payload-row out, scale-row out)`.
pub type EncodeRowJob<'a> = dyn Fn(usize, &mut [f32], &mut [u8], &mut [u8]) + Sync + 'a;

/// Run `job` over every row of an encode, in parallel when the tensor pays
/// for it. Outputs are handed to each worker as disjoint `split_at_mut`
/// slices, so the parallelism needs no synchronisation.
pub fn encode_rows(
    rows: usize,
    cols: usize,
    out_row_bytes: usize,
    scale_row_bytes: usize,
    out: &mut [u8],
    scales: &mut [u8],
    job: &EncodeRowJob<'_>,
) {
    // Below about a megabyte of input the threads cost more than they carry.
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

