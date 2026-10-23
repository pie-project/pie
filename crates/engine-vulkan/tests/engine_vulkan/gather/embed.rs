//! The embedding lookups on the first Vulkan device against an f64 host
//! reference: a bf16 table by id (out-of-range ids read row 0), rows side by
//! side (out of range lands zeros), weighted taps, and affine-quantized
//! banks.

use crate::gpu::{Gpu, close, noise};
use kernels_slang::{Bank, Tensor};
use poem_ir::Dtype;

const VOCAB: usize = 10;

/// A small deterministic stream of 32-bit words.
fn words(seed: u64, n: usize) -> Vec<u32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 16) as u32
        })
        .collect()
}

#[test]
fn the_embeddings_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let sb = |b: Bank| kernels_slang::Bank {
        codes: b.codes,
        scales: b.scales,
        biases: b.biases,
        group: b.group,
        bits: b.bits,
    };
    let hidden = 128usize;
    let (tbuf, tr) = gpu.up(&noise(5, VOCAB * hidden));
    let tr = tr.as_slice();
    let table = Tensor::new(tbuf, VOCAB as u32, hidden as u32, Dtype::Bf16);
    let row = |id: i32| -> &[f32] { &tr[id as usize * hidden..][..hidden] };

    // By id; 12 and -1 are out of range.
    let idv = [3i32, 0, 9, 12, -1, 7];
    let ids = gpu.up_i32(&idv);
    let n = idv.len();
    let it = Tensor::new(ids, n as u32, 1, Dtype::I32);
    let yt = |buf: u32, rows: usize, width: usize| {
        Tensor::new(buf, rows as u32, width as u32, Dtype::Bf16)
    };
    let want: Vec<f64> = idv
        .iter()
        .flat_map(|&i| {
            row(if (0..VOCAB as i32).contains(&i) { i } else { 0 })
                .iter()
                .map(|v| f64::from(*v))
        })
        .collect();
    let y = gpu.zeros(n * hidden);
    gpu.run(|c| kernels_slang::gather::embed::embed(c, it, table, VOCAB as u32, yt(y, n, hidden)));
    close(&gpu.down(y, n * hidden), &want, "embed");

    // Side by side: three ids a row.
    let cat = Tensor::new(ids, 2, 3, Dtype::I32);
    let want: Vec<f64> = idv
        .iter()
        .flat_map(|&i| {
            let inside = (0..VOCAB as i32).contains(&i);
            (0..hidden).map(move |c| {
                if inside {
                    f64::from(tr[i as usize * hidden + c])
                } else {
                    0.0
                }
            })
        })
        .collect();
    let y = gpu.zeros(n * hidden);
    gpu.run(|c| {
        kernels_slang::gather::embed::embed_concat(
            c,
            cat,
            table,
            VOCAB as u32,
            yt(y, 2, 3 * hidden),
        )
    });
    close(&gpu.down(y, n * hidden), &want, "embed concat");

    // Weighted: two rows of three taps.
    let wv = [0.5f32, 0.25, 0.25, 1.0, -0.5, 2.0];
    let weights = gpu.up_f32(&wv);
    let wt = Tensor::new(weights, 2, 3, Dtype::F32);
    let want: Vec<f64> = (0..2)
        .flat_map(|r| {
            let (idv, wv, tr) = (&idv, &wv, &tr);
            (0..hidden).map(move |c| {
                (0..3)
                    .map(|t| {
                        let i = idv[r * 3 + t];
                        let at = if (0..VOCAB as i32).contains(&i) {
                            i as usize
                        } else {
                            0
                        };
                        f64::from(wv[r * 3 + t]) * f64::from(tr[at * hidden + c])
                    })
                    .sum::<f64>()
            })
        })
        .collect();
    let y = gpu.zeros(2 * hidden);
    gpu.run(|c| {
        kernels_slang::gather::embed::embed_weighted(
            c,
            cat,
            wt,
            table,
            VOCAB as u32,
            yt(y, 2, hidden),
        )
    });
    close(&gpu.down(y, 2 * hidden), &want, "embed weighted");

    // Affine banks, by id and side by side.
    for (group, bits) in [(32usize, 4usize), (64, 8), (128, 2)] {
        let per_word = 32 / bits;
        let codes = gpu.up_u32(&words(
            group as u64 + bits as u64,
            VOCAB * hidden / per_word,
        ));
        let cv = words(group as u64 + bits as u64, VOCAB * hidden / per_word);
        let groups = VOCAB * hidden / group;
        let (scales, sr) = gpu.up(&noise(group as u64, groups)
            .iter()
            .map(|v| v * 0.1)
            .collect::<Vec<_>>());
        let (biases, br) = gpu.up(&noise(group as u64 + 1, groups));
        let bank = Bank {
            codes: Tensor::new(codes, VOCAB as u32, (hidden / per_word) as u32, Dtype::U32),
            scales: Tensor::new(scales, VOCAB as u32, (hidden / group) as u32, Dtype::Bf16),
            biases: Some(Tensor::new(
                biases,
                VOCAB as u32,
                (hidden / group) as u32,
                Dtype::Bf16,
            )),
            group: group as u32,
            bits: bits as u32,
        };
        let value = |id: i32, k: usize| -> f64 {
            if !(0..VOCAB as i32).contains(&id) {
                return 0.0;
            }
            let r = id as usize;
            let word = cv[r * (hidden / per_word) + k / per_word];
            let code = (word >> ((k % per_word) * bits)) & ((1u32 << bits) - 1);
            let g = r * (hidden / group) + k / group;
            f64::from(sr[g]) * f64::from(code) + f64::from(br[g])
        };
        let want: Vec<f64> = idv
            .iter()
            .flat_map(|&i| (0..hidden).map(move |k| value(i, k)))
            .collect();
        let shape = format!("group {group}, {bits} bits");
        let y = gpu.zeros(n * hidden);
        gpu.run(|c| {
            kernels_slang::gather::embed::embed_banked(
                c,
                it,
                sb(bank),
                VOCAB as u32,
                yt(y, n, hidden),
            )
        });
        close(
            &gpu.down(y, n * hidden),
            &want,
            &format!("embed bank {shape}"),
        );
        let y = gpu.zeros(n * hidden);
        gpu.run(|c| {
            kernels_slang::gather::embed::embed_concat_banked(
                c,
                cat,
                sb(bank),
                VOCAB as u32,
                yt(y, 2, 3 * hidden),
            )
        });
        close(
            &gpu.down(y, n * hidden),
            &want,
            &format!("embed concat bank {shape}"),
        );
    }
}
