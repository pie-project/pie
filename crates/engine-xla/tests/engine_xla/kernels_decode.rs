//! The kernels the catalog coverage pass added or widened, against host
//! references: a split-plane bank decoded to a dense bf16 plane (MLA's
//! absorbed `kv_b`, kernels-cuda `decoded_plane`), `mul_scalar` over f32 and
//! f16 rows (the DiT timestep scalings), and a pooled reader whose page
//! bound closes no block.

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use kernels_xla::attn::pool;
use kernels_xla::elemwise::norm;
use kernels_xla::linear::decode;
use kernels_xla::{Bank, KvPool, Tensor};

fn hash(i: usize, seed: u32) -> u32 {
    (i as u32)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(seed.wrapping_mul(40503))
        .rotate_left(13)
        .wrapping_mul(2_246_822_519)
        .rotate_left(5)
}

fn data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| round_bf16(((hash(i, seed) >> 8) % 2000) as f32 / 1000.0 - 1.0))
        .collect()
}

/// Codes `[n, k]` packed LSB-first, `bits` each, into bytes `[n, k·bits/8]`.
fn pack(codes: &[u32], bits: u32) -> Vec<u8> {
    let per = (8 / bits) as usize;
    codes
        .chunks(per)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u8, |acc, (i, &v)| acc | ((v as u8) << (i as u32 * bits)))
        })
        .collect()
}

fn f16_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f32::from(h & 0x3ff);
    sign * match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::NAN,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    }
}

fn retype(t: Tensor, rows: u32, width: u32, dtype: Dtype) -> Tensor {
    Tensor::new(t.buf, rows, width, dtype)
}

#[test]
fn a_bank_decodes_to_the_plane_its_codes_and_factors_state() {
    let (n, k) = (24usize, 256usize);
    let mut b = Bench::new();
    let mut cases: Vec<(Bank, Tensor, Vec<f32>, String)> = Vec::new();
    for (at, &(group, bits, dtype)) in [
        (64u32, 4u32, Dtype::U4g64),
        (32, 4, Dtype::U4g32),
        (32, 2, Dtype::U2g32),
        (128, 2, Dtype::U2g128),
        (64, 8, Dtype::U8g64),
    ]
    .iter()
    .enumerate()
    {
        let seed = 10 + at as u32 * 5;
        let groups = k / group as usize;
        let codes: Vec<u32> = (0..n * k)
            .map(|i| hash(i, seed) >> 7 & ((1 << bits) - 1))
            .collect();
        let scales: Vec<f32> = data(n * groups, seed + 1)
            .iter()
            .map(|v| round_bf16((v.abs() + 0.1) / (1 << bits) as f32))
            .collect();
        let biases: Vec<f32> = data(n * groups, seed + 2)
            .iter()
            .map(|v| round_bf16(v * 0.3))
            .collect();
        let want: Vec<f32> = (0..n * k)
            .map(|i| {
                let g = (i / k) * groups + (i % k) / group as usize;
                round_bf16(scales[g] * codes[i] as f32 + biases[g])
            })
            .collect();
        let bytes = b.u8(n as u32, (k as u32) * bits / 8, &pack(&codes, bits));
        // Both code views: the bank's packed dtype and the raw bytes.
        let codes_t = if at % 2 == 0 {
            retype(bytes, n as u32, k as u32, dtype)
        } else {
            bytes
        };
        let bank = Bank {
            codes: codes_t,
            scales: b.bf16(n as u32, groups as u32, &scales),
            biases: Some(b.bf16(n as u32, groups as u32, &biases)),
            group,
            bits,
        };
        let out = b.zeros(Dtype::Bf16, n as u32, k as u32);
        cases.push((bank, out, want, format!("affine g{group} b{bits}")));
    }
    // mxfp4: symmetric, e8m0 scales per 32 codes.
    let lut = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let codes: Vec<u32> = (0..n * k).map(|i| hash(i, 90) >> 9 & 15).collect();
    let exps: Vec<u8> = (0..n * k / 32)
        .map(|i| 122 + (hash(i, 91) % 8) as u8)
        .collect();
    let want: Vec<f32> = (0..n * k)
        .map(|i| {
            let c = codes[i];
            let v = lut[(c & 7) as usize] * if c & 8 != 0 { -1.0 } else { 1.0 };
            round_bf16(v * 2f32.powi(i32::from(exps[(i / k) * (k / 32) + (i % k) / 32]) - 127))
        })
        .collect();
    let mc = b.u8(n as u32, (k / 2) as u32, &pack(&codes, 4));
    let ms = b.u8(n as u32, (k / 32) as u32, &exps);
    let bank = Bank {
        codes: retype(mc, n as u32, k as u32, Dtype::Mxfp4),
        scales: retype(ms, n as u32, (k / 32) as u32, Dtype::E8m0),
        biases: None,
        group: 32,
        bits: 4,
    };
    let out = b.zeros(Dtype::Bf16, n as u32, k as u32);
    cases.push((bank, out, want, "mxfp4".into()));

    let ran = b
        .run(|ctx| {
            for (bank, out, _, _) in &cases {
                decode::decoded_plane(ctx, "attention.mla_absorb_q", *bank, *out)?;
            }
            Ok(())
        })
        .unwrap();
    if !ran {
        return;
    }
    for (_, out, want, label) in &cases {
        eprintln!("{label}");
        assert_close(&b.read_f32(*out), want, 0.0, 0.0);
    }
}

#[test]
fn mul_scalar_rounds_its_scalar_to_the_row_element() {
    let (rows, width) = (5usize, 37usize);
    let xs = data(rows * width, 3);
    let s = 1.0f32 / 3.0;
    let mut b = Bench::new();
    let xf = b.f32(rows as u32, width as u32, &xs);
    let xb = b.bf16(rows as u32, width as u32, &xs);
    let f16_bytes: Vec<u8> = xs
        .iter()
        .flat_map(|&v| kernels_xla::hlo::f16_bits(v).to_le_bytes())
        .collect();
    let xh = b.raw(Dtype::F16, rows as u32, width as u32, f16_bytes);
    let ran = b
        .run(|ctx| {
            norm::mul_scalar(ctx, s, xf)?;
            norm::mul_scalar(ctx, s, xb)?;
            norm::mul_scalar(ctx, s, xh)
        })
        .unwrap();
    if !ran {
        return;
    }
    // f32 rows scale by the scalar as stated; bf16 rows by its bf16.
    let want_f: Vec<f32> = xs.iter().map(|v| v * s).collect();
    assert_close(&b.read_f32(xf), &want_f, 0.0, 1e-7);
    let sb = round_bf16(s);
    let want_b: Vec<f32> = xs.iter().map(|v| round_bf16(v * sb)).collect();
    assert_close(&b.read_f32(xb), &want_b, 0.0, 0.0);
    // f16 rows by its f16 (0.333251953125), rounded to f16 once.
    let sh = 0.333_251_95_f32;
    let want_h: Vec<f32> = xs.iter().map(|v| v * sh).collect();
    let got_h: Vec<f32> = b
        .bytes(xh)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f16_value(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    assert_close(&got_h, &want_h, 0.0, 1e-3);
}

#[test]
fn a_pooled_reader_whose_bound_closes_no_block_reads_nothing() {
    // Three pages of two cells at a pooling ratio of eight: no block closes
    // under the bound, so every row reads nothing (o = 0, lse = -inf).
    let (rows, heads, hd, ps, ratio) = (3usize, 2usize, 8usize, 2usize, 8u32);
    let mut b = Bench::new();
    let q = b.bf16(
        rows as u32,
        (heads * hd) as u32,
        &data(rows * heads * hd, 7),
    );
    let pos = b.i32(rows as u32, 1, &[0, 3, 5]);
    let req = b.i32(rows as u32, 1, &[0, 0, 0]);
    let keys = b.bf16(8, hd as u32, &data(8 * hd, 8));
    let pages = KvPool {
        keys,
        values: keys,
        page_indices: b.u32(3, 1, &[0, 1, 2]),
        page_indptr: b.u32(2, 1, &[0, 3]),
        page_size: ps as i32,
        max_pages: 3,
        seq_stride: hd as u64,
        head_stride: hd as u64,
    };
    let o = b.bf16(
        rows as u32,
        (heads * hd) as u32,
        &vec![1.0; rows * heads * hd],
    );
    let lse = b.f32(rows as u32, heads as u32, &vec![1.0; rows * heads]);
    let ran = b
        .run(|ctx| {
            pool::attention_lse(
                ctx,
                q,
                pos,
                req,
                &pages,
                ratio,
                heads as u32,
                hd as u32,
                0.5,
                o,
                lse,
            )
        })
        .unwrap();
    if !ran {
        return;
    }
    assert_close(&b.read_f32(o), &vec![0.0; rows * heads * hd], 0.0, 0.0);
    assert!(b.read_f32(lse).iter().all(|v| *v == f32::NEG_INFINITY));
}
