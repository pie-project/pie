//! A bf16 view of the u32 pool reads its elements in the packed rows order
//! (`kernels_xla::hlo::view`): element `r·256 + h·128 + l` of the flat view
//! is half `h` of word `l` of pool row `r`, as a Mosaic kernel bitcasts a
//! block in VMEM, not the bytes' natural little-endian order. Bytes laid in
//! that order (`store::streamed`'s `packed_rows`, what every streamed weight
//! lands as) read back their values through the view at every width, by
//! block and by row; bytes laid naturally read back through the u32 rows
//! bitcast, and through the view only permuted. Skipped without a device.

use dtype::Dtype;
use engine_xla::bench::Bench;
use kernels_xla::hlo::{Elem, GatherDims};

const ROWS: usize = 64;
const PER_ROW: usize = 256;

/// The value of flat element `e`: distinct finite bf16 bit patterns (a
/// NaN would come back canonical).
fn value(e: usize) -> u16 {
    (0x3000 + (e as u16 & 0x1fff)) | (((e >> 13) as u16 & 1) << 15)
}

/// Every element's bits, natural order.
fn natural() -> Vec<u16> {
    (0..ROWS * PER_ROW).map(value).collect()
}

/// Pool words holding `bits` laid naturally: element `2w` low half of word
/// `w`, `2w + 1` its high half.
fn words_natural(bits: &[u16]) -> Vec<u32> {
    bits.chunks(2)
        .map(|p| u32::from(p[0]) | u32::from(p[1]) << 16)
        .collect()
}

/// Pool words holding `bits` in the packed rows order: word `l` of a row
/// holds its elements `l` (low half) and `l + 128`.
fn words_packed(bits: &[u16]) -> Vec<u32> {
    bits.chunks(PER_ROW)
        .flat_map(|row| (0..128).map(move |l| u32::from(row[l]) | u32::from(row[l + 128]) << 16))
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum Read {
    /// `[n, width]` blocks of whole view rows at each start (`gather`).
    Block,
    /// One view row an index (`take_rows`).
    Rows,
    /// The u32 rows themselves, bitcast (no view).
    Bitcast,
}

/// Every view row of a `width`-wide bf16 view of `pool`, read by `how`, as
/// bf16 bits, in view order.
fn read(words: &[u32], width: usize, how: Read) -> Option<Vec<u16>> {
    let view_rows = ROWS * PER_ROW / width;
    let mut b = Bench::new();
    let pool = b.u32(ROWS as u32, 128, words);
    let out = b.zeros(Dtype::Bf16, view_rows as u32, width as u32);
    // Two blocks: the first half of the view rows and the second.
    let half = view_rows / 2;
    let starts = b.i32(2, 1, &[0, half as i32]);
    let ids: Vec<i32> = (0..view_rows as i32).collect();
    let rows = b.i32(view_rows as u32, 1, &ids);
    let ran = b
        .run(|ctx| {
            engine_xla::lower::lower(ctx, |l| {
                let pool_v = l.read(pool)?.val;
                let f = l.f();
                let v = match how {
                    Read::Bitcast => {
                        let words = f.bitcast(pool_v, Elem::Bf16)?;
                        f.reshape(words, &[view_rows as i64, width as i64])?
                    }
                    Read::Block | Read::Rows => {
                        let view =
                            f.pool_view(pool_v, Elem::Bf16, &[view_rows as i64, width as i64])?;
                        match how {
                            Read::Block => {
                                let at = l.read(starts)?.val;
                                let f = l.f();
                                let block = f.gather(
                                    view,
                                    at,
                                    &GatherDims {
                                        offset_dims: vec![1, 2],
                                        start_index_map: vec![0],
                                        index_vector_dim: 1,
                                        ..GatherDims::default()
                                    },
                                    &[half as i64, width as i64],
                                )?;
                                f.reshape(block, &[view_rows as i64, width as i64])?
                            }
                            _ => {
                                let at = l.read(rows)?.val;
                                let f = l.f();
                                let at = f.reshape(at, &[view_rows as i64])?;
                                f.take_rows(view, at)?
                            }
                        }
                    }
                };
                l.land(out, v)
            })
        })
        .unwrap_or_else(|err| panic!("{how:?} at {width}: {err}\n{}", b.last_module));
    if !ran {
        return None;
    }
    Some(
        b.bytes(out)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect(),
    )
}

#[test]
fn a_pool_view_reads_the_packed_rows_order() {
    let want = natural();
    let packed = words_packed(&want);
    let linear = words_natural(&want);
    // A row's width and the widths a streamed weight's view takes (its
    // plane's row: 128 to 2048 here, every one dividing the pool).
    for width in [128usize, 256, 512, 1024, 2048] {
        for how in [Read::Block, Read::Rows] {
            let Some(got) = read(&packed, width, how) else {
                return;
            };
            assert_eq!(
                got, want,
                "{how:?} at {width}: packed bytes read their values"
            );
            let permuted = read(&linear, width, how).expect("ran above");
            assert_ne!(
                permuted, want,
                "{how:?} at {width}: natural bytes read permuted"
            );
        }
        let got = read(&linear, width, Read::Bitcast).expect("ran above");
        assert_eq!(
            got, want,
            "bitcast at {width}: natural bytes read their values"
        );
    }
    eprintln!("a bf16 pool view reads the packed rows order at every width, by block and row");
}
