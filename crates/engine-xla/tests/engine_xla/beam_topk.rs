//! `PIE_XLA_TOPK_BENCH=1`: a beam step's top-k over the vocabulary
//! (default one row of 4 x 151936 f32, k beams), device time a run: the
//! epilogue's stable sort as it laid it (`Sort3`: a `[1, rows, n]` sort, a
//! reshape, then the slice), the same sort sliced straight off (`Sort`: XLA
//! rewrites a sort sliced to its first k into its top-k), `chlo.top_k` over
//! the order key, and a two-stage top-k (each tile's k, then the k of
//! those); each answering the stable sort's ids, ties and signed zeros at
//! the top. v6e: Sort3 3.95 ms, Sort and TopK 0.20 ms, Tiled(2048) 0.13 ms.

use dtype::Dtype;
use engine_xla::bench::Bench;
use engine_xla::handle::Tensor;
use kernels_xla::hlo::{Cmp, Elem, Func, Val};

/// The epilogue's order key (`guest::lower::order_key`): monotone in the
/// value, -0 and +0 one key, NaN below -inf.
fn order_key(f: &mut Func, x: Val) -> Val {
    let zero = f.like_f(x, 0.0);
    let is_zero = f.compare(Cmp::Eq, x, zero).unwrap();
    let x = f.select(is_zero, zero, x).unwrap();
    let bits = f.bitcast(x, Elem::I32).unwrap();
    let flip = f.like_i(bits, 0x7FFF_FFFF);
    let flipped = f.xor(bits, flip).unwrap();
    let nought = f.like_i(bits, 0);
    let negative = f.compare(Cmp::Lt, bits, nought).unwrap();
    let key = f.select(negative, flipped, bits).unwrap();
    let nan = f.compare(Cmp::Ne, x, x).unwrap();
    let bottom = f.like_i(bits, i64::from(i32::MIN));
    f.select(nan, bottom, key).unwrap()
}

#[derive(Clone, Copy, Debug)]
enum Way {
    /// The epilogue's form: the rows as `[1, rows, n]` (`rows_of`'s `[B, R, n]`).
    Sort3,
    Sort,
    TopK,
    Tiled(i64),
}

fn ids(f: &mut Func, x: Val, k: i64, way: Way) -> Val {
    let dims = f.dims(x).to_vec();
    let (rows, n) = (dims[0], dims[1]);
    let key = order_key(f, x);
    match way {
        Way::Sort3 => {
            let k3 = f.reshape(key, &[1, rows, n]).unwrap();
            let x3 = f.reshape(x, &[1, rows, n]).unwrap();
            let index = f.iota(Elem::I32, &[1, rows, n], 2);
            let sorted = f
                .sort(&[k3, index, x3], 2, true, |f, a, b| {
                    f.compare(Cmp::Gt, a[0], b[0])
                })
                .unwrap();
            let ids = f
                .slice(sorted[1], &[0, 0, 0], &[1, rows, k], &[1, 1, 1])
                .unwrap();
            f.reshape(ids, &[rows, k]).unwrap()
        }
        Way::Sort => {
            let index = f.iota(Elem::I32, &dims, 1);
            let sorted = f
                .sort(&[key, index, x], 1, true, |f, a, b| {
                    f.compare(Cmp::Gt, a[0], b[0])
                })
                .unwrap();
            f.slice(sorted[1], &[0, 0], &[rows, k], &[1, 1]).unwrap()
        }
        Way::TopK => f.top_k(key, k).unwrap().1,
        Way::Tiled(t) => {
            let tiles = (n + t - 1) / t;
            let padded = tiles * t;
            let key = if padded > n {
                let low = f.const_i(Elem::I32, i64::from(i32::MIN), &[rows, padded - n]);
                f.concat(&[key, low], 1).unwrap()
            } else {
                key
            };
            let key3 = f.reshape(key, &[rows, tiles, t]).unwrap();
            let (v, i) = f.top_k(key3, k).unwrap();
            // a tile's ids over the row
            let base = f.iota(Elem::I32, &[rows, tiles, k], 1);
            let tv = f.const_i(Elem::I32, t, &[rows, tiles, k]);
            let base = f.mul(base, tv).unwrap();
            let i = f.add(i, base).unwrap();
            let v = f.reshape(v, &[rows, tiles * k]).unwrap();
            let i = f.reshape(i, &[rows, tiles * k]).unwrap();
            // the k of the tiles' k: ties by the lower id, as the sort
            let sorted = f
                .sort(&[v, i], 1, true, |f, a, b| {
                    let gt = f.compare(Cmp::Gt, a[0], b[0])?;
                    let eq = f.compare(Cmp::Eq, a[0], b[0])?;
                    let lo = f.compare(Cmp::Lt, a[1], b[1])?;
                    let tie = f.and(eq, lo)?;
                    f.or(gt, tie)
                })
                .unwrap();
            f.slice(sorted[1], &[0, 0], &[rows, k], &[1, 1]).unwrap()
        }
    }
}

fn run(rows: usize, n: usize, k: i64, way: Way, xs: &[f32]) -> (f64, Vec<i32>) {
    let mut b = Bench::new();
    let x = b.f32(rows as u32, n as u32, xs);
    let out: Tensor = b.zeros(Dtype::I32, rows as u32, k as u32);
    let body = |ctx: &engine_xla::emit::Ctx<'_>| {
        engine_xla::lower::lower(ctx, |l| {
            let xv = l.read(x)?.val;
            let v = ids(l.f(), xv, k, way);
            l.land(out, v)
        })
    };
    let secs = b.time_pipelined(50, body).unwrap().unwrap_or(f64::NAN);
    assert!(b.run(body).unwrap());
    (secs * 1e3, b.read_i32(out))
}

#[test]
fn a_beam_steps_top_k_is_timed() {
    if std::env::var_os("PIE_XLA_TOPK_BENCH").is_none() {
        return;
    }
    // `PIE_XLA_TOPK_SHAPE=rows,n` (default: a beam-4 step's flat 4 x 151936)
    let (rows, n) = std::env::var("PIE_XLA_TOPK_SHAPE")
        .ok()
        .and_then(|v| {
            v.split_once(',')
                .map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap()))
        })
        .unwrap_or((1usize, 4 * 151_936usize));
    // log-softmax-like scores with ties at the top (a beam's ids must
    // break them as the sort does)
    let mut seed = 7u64;
    let xs: Vec<f32> = (0..rows * n)
        .map(|i| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = ((seed >> 40) as f32) / (1u64 << 24) as f32;
            // six equal maxima across the row (ties the top-k must break by
            // the lower index), a -0 and a +0 among them
            if i % 100_000 == 17 {
                -0.25
            } else if i == 250_001 {
                -0.0
            } else if i == 350_003 {
                0.0
            } else {
                -1.0 - 20.0 * u
            }
        })
        .collect();
    for k in [4i64, 8] {
        let (t0, want) = run(rows, n, k, Way::Sort, &xs);
        eprintln!("TOPK k={k} sort {t0:.4} ms");
        for way in [
            Way::Sort3,
            Way::TopK,
            Way::Tiled(1024),
            Way::Tiled(2048),
            Way::Tiled(4096),
            Way::Tiled(8192),
        ] {
            let (t, got) = run(rows, n, k, way, &xs);
            eprintln!(
                "TOPK k={k} {way:?} {t:.4} ms ids {}",
                if got == want { "equal" } else { "DIFFER" }
            );
        }
    }
}
