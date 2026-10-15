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
    /// FP8 codes under square `group`-wide block factors. The factor grid
    /// restarts every `chunk_rows` rows at `chunk_blocks` rows of its own (a
    /// checkpoint stored pre-sharded, each shard scaled apart); one chunk of
    /// every row is the plain grid. The operand's rows start at the weight's
    /// `first_row` and its columns at `first_col`.
    BlockScaledFp8 {
        bytes: &'a [u8],
        factors: Vec<f32>,
        scale_cols: usize,
        group: usize,
        chunk_rows: usize,
        chunk_blocks: usize,
        first_row: usize,
        first_col: usize,
    },
}

/// How an FP8 weight's block factors cover it: `group`-wide square blocks,
/// the grid restarting every `chunk_rows` rows at `chunk_blocks` rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8BlockGrid {
    pub group: usize,
    pub chunk_rows: usize,
    pub chunk_blocks: usize,
}

/// The grid `scales` factors lay over a `weight` whose rows are stored cut
/// into `chunks` shards, each scaled apart (MiMo-V2's fused qkv, cut at its
/// `num_key_value_heads`), or, with none stated (0), the one grid over every
/// row. Each shard's last block may be short, as 576 rows are under five
/// factors of 128.
///
/// Unstated, a weight one grid does not account for is read under the one
/// cut its factors do account for, and refused if two cuts do. That reading
/// cannot see a cut whose factor rows one grid also accounts for (rows
/// `128k + r` a shard, `r > 128 (1 - 1/n)`), so a reader that knows its
/// checkpoint is stored cut states the cut.
pub fn fp8_block_grid(
    weight: [usize; 2],
    scales: [usize; 2],
    chunks: usize,
) -> Result<Fp8BlockGrid, String> {
    let ([rows, cols], [scale_rows, scale_cols]) = (weight, scales);
    let refused = |cut: &str| {
        format!(
            "has unsupported scale shape [{scale_rows}, {scale_cols}] for weight \
             [{rows}, {cols}]{cut}"
        )
    };
    let cut = |n: usize| -> Option<Fp8BlockGrid> {
        if n == 0 || scale_cols == 0 || rows % n != 0 || scale_rows % n != 0 {
            return None;
        }
        let (chunk_rows, chunk_blocks) = (rows / n, scale_rows / n);
        if chunk_rows == 0 || chunk_blocks == 0 {
            return None;
        }
        // the block is the widest a factor covers, and each axis counts
        // exactly its factors in blocks of it
        let group = chunk_rows
            .div_ceil(chunk_blocks)
            .max(cols.div_ceil(scale_cols));
        (chunk_rows.div_ceil(group) == chunk_blocks && cols.div_ceil(group) == scale_cols)
            .then_some(Fp8BlockGrid {
                group,
                chunk_rows,
                chunk_blocks,
            })
    };
    if chunks > 0 {
        return cut(chunks).ok_or_else(|| refused(&format!(" cut into {chunks} shards")));
    }
    if let Some(grid) = cut(1) {
        return Ok(grid);
    }
    let fits: Vec<(usize, Fp8BlockGrid)> = (2..=scale_rows)
        .filter_map(|n| cut(n).map(|grid| (n, grid)))
        .collect();
    match fits.as_slice() {
        [(_, grid)] => Ok(*grid),
        [] => Err(refused("")),
        many => Err(format!(
            "{}: cuts into {:?} shards each account for its factors, and the read \
             states none",
            refused(""),
            many.iter().map(|(n, _)| *n).collect::<Vec<_>>()
        )),
    }
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
                chunk_rows,
                chunk_blocks,
                first_row,
                first_col,
            } => {
                let row_bytes = &bytes[row * cols..(row + 1) * cols];
                let at = first_row + row;
                let block = (at / chunk_rows) * chunk_blocks + (at % chunk_rows) / group;
                let scale_row = block * scale_cols;
                for (col, (&code, out)) in row_bytes.iter().zip(buf.iter_mut()).enumerate() {
                    let factor = factors[scale_row + (first_col + col) / group];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_every_case() {
        one_grid_covers_an_aligned_weight();
        one_grid_covers_a_weight_whose_last_block_is_short();
        a_pre_sharded_weight_restarts_its_grid_every_chunk();
        a_chunked_grid_reads_each_row_under_its_own_chunk();
        factors_no_cut_accounts_for_are_refused();
        a_cut_one_grid_also_accounts_for_is_read_as_stated();
    }

    fn one_grid_covers_an_aligned_weight() {
        let grid = fp8_block_grid([14848, 4096], [116, 32], 0).unwrap();
        assert_eq!(
            (grid.group, grid.chunk_rows, grid.chunk_blocks),
            (128, 14848, 116)
        );
    }

    fn one_grid_covers_a_weight_whose_last_block_is_short() {
        let grid = fp8_block_grid([200, 256], [2, 2], 0).unwrap();
        assert_eq!((grid.group, grid.chunk_rows), (128, 200));
        let grid = fp8_block_grid([576, 512], [5, 4], 0).unwrap();
        assert_eq!(
            (grid.group, grid.chunk_rows, grid.chunk_blocks),
            (128, 576, 5)
        );
    }

    /// MiMo-V2.6-Flash's full-attention qkv: four chunks of 3392 rows, 26.5
    /// blocks each, 27 factor rows each; Pro's: eight of them.
    fn a_pre_sharded_weight_restarts_its_grid_every_chunk() {
        for chunks in [0, 4] {
            let flash = fp8_block_grid([13568, 4096], [108, 32], chunks).unwrap();
            assert_eq!(
                (flash.group, flash.chunk_rows, flash.chunk_blocks),
                (128, 3392, 27)
            );
        }
        let pro = fp8_block_grid([27136, 6144], [216, 48], 8).unwrap();
        assert_eq!(
            (pro.group, pro.chunk_rows, pro.chunk_blocks),
            (128, 3392, 27)
        );
        // aligned shards: the stated cut and the one grid are the same grid
        let swa = fp8_block_grid([14848, 4096], [116, 32], 4).unwrap();
        assert_eq!(
            (swa.group, swa.chunk_rows, swa.chunk_blocks),
            (128, 3712, 29)
        );
    }

    fn a_chunked_grid_reads_each_row_under_its_own_chunk() {
        // two chunks of 320 rows, three factor rows each, one factor column
        let grid = fp8_block_grid([640, 128], [6, 1], 2).unwrap();
        assert_eq!((grid.chunk_rows, grid.chunk_blocks), (320, 3));
        // 0x38 is 1.0 in e4m3
        let codes = vec![0x38u8; 66 * 128];
        let operand = EncodeOperand::BlockScaledFp8 {
            bytes: &codes,
            factors: vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            scale_cols: 1,
            group: grid.group,
            chunk_rows: grid.chunk_rows,
            chunk_blocks: grid.chunk_blocks,
            first_row: 255,
            first_col: 0,
        };
        let mut buf = vec![0.0f32; 128];
        // rows 255, 256, 319 and 320 of the weight: the first chunk's second
        // block, its short tail, then the second chunk's first block
        for (row, want) in [(0, 2.0), (1, 4.0), (64, 4.0), (65, 8.0)] {
            operand.row_bf16(row, 128, &mut buf);
            assert_eq!(buf[0], want, "operand row {row}");
        }
    }

    fn factors_no_cut_accounts_for_are_refused() {
        assert!(fp8_block_grid([13568, 4096], [109, 32], 0).is_err());
        assert!(fp8_block_grid([64, 0], [1, 1], 0).is_err());
        // a stated cut its factors do not account for
        assert!(fp8_block_grid([13568, 4096], [108, 32], 8).is_err());
        // two shards of 192 rows or four of 96 both give four factor rows
        assert!(fp8_block_grid([384, 128], [4, 1], 0).is_err());
        assert_eq!(
            fp8_block_grid([384, 128], [4, 1], 2).unwrap().chunk_rows,
            192
        );
    }

    /// Two shards of 228 rows (128 + 100) hold four factor rows, and so does
    /// one grid over 456: unstated, it is read as the one grid, so the
    /// stated cut is what reads it right.
    fn a_cut_one_grid_also_accounts_for_is_read_as_stated() {
        let one = fp8_block_grid([456, 128], [4, 1], 0).unwrap();
        assert_eq!(one.chunk_rows, 456);
        let cut = fp8_block_grid([456, 128], [4, 1], 2).unwrap();
        assert_eq!((cut.chunk_rows, cut.chunk_blocks), (228, 2));
    }
}
