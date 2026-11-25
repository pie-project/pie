//! The LoRA correction (`linear::lora`) against a host reference: a row's
//! `y + B (A x)` in f64, its down projection rounded to the activation's dtype
//! as the kernels hold it, in each dtype it takes; a row routed to no adapter, or in no segment, keeps its bits.

use crate::common::linear_ref::{sum, t, tab};
use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::lora::{self, Segments};

const ROWS: usize = 7;
/// Widths that fill no whole block (the input whole `float4` loads).
const INPUT: usize = 296;
const OUT: usize = 520;
const ADAPTERS: usize = 3;
const ROUTES: [i32; ROWS] = [2, -1, 0, 1, 1, -1, 2];

struct Form {
    dtype: Dtype,
    encode: fn(f32) -> u16,
    decode: fn(u16) -> f32,
    /// Two units in the last place, of a value.
    tol: f64,
}

/// An f32 as f16, rounded to nearest, its normal range alone (smaller: zero).
fn to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 112;
    if exp <= 0 {
        return sign;
    }
    assert!(exp < 31, "{v} is past f16");
    let half = ((exp as u32) << 10) | ((bits >> 13) & 0x3ff);
    sign | (half + ((bits >> 12) & 1)) as u16
}

fn from_f16(v: u16) -> f32 {
    let sign = if v & 0x8000 == 0 { 1.0 } else { -1.0 };
    let (exp, frac) = (i32::from((v >> 10) & 31), f32::from(v & 0x3ff));
    if exp == 0 {
        sign * frac * 2f32.powi(-24)
    } else {
        sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15)
    }
}

/// `count` values of `scale`, in `form`, raw and as the kernel reads them.
fn draw(lcg: &mut Lcg, form: &Form, count: usize, scale: f32) -> (Vec<u16>, Vec<f64>) {
    let (_, exact) = lcg.row(count);
    let raw = tab(count, |i| (form.encode)(exact[i] * scale));
    let read = tab(count, |i| f64::from((form.decode)(raw[i])));
    (raw, read)
}

fn answers(form: &Form, rank: usize) {
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(0x10a + rank as u64);
    let (x_raw, x) = draw(&mut lcg, form, ROWS * INPUT, 1.0);
    let (a_raw, a) = draw(&mut lcg, form, ADAPTERS * rank * INPUT, 0.25);
    let (b_raw, b) = draw(&mut lcg, form, ADAPTERS * OUT * rank, 0.25);
    let (y_raw, y) = draw(&mut lcg, form, ROWS * OUT, 1.0);

    let what = format!("{:?} rank {rank}", form.dtype);
    let tensor = |at: u64, rows: usize, width: usize| t(at, rows, width, form.dtype);
    let xt = tensor(gpu.up(&x_raw), ROWS, INPUT);
    let at = tensor(gpu.up(&a_raw), ADAPTERS, rank * INPUT);
    let bt = tensor(gpu.up(&b_raw), ADAPTERS, OUT * rank);
    let routes = t(gpu.up(&ROUTES), ROWS, 1, Dtype::I32);
    let list = t(gpu.up(&[0i32, 2, 4, 2]), 4, 1, Dtype::I32);

    let segmented = Segments { list, count: 2, cap: 2, max_rows: 2 };
    for segments in [None, Some(segmented)] {
        let mut yt = tensor(gpu.up(&y_raw), ROWS, OUT);
        let ctx = gpu.ctx();
        gpu.laid(&ctx, |ctx| lora::correct(ctx, xt, at, bt, routes, &mut yt, segments))
            .expect("the correction fires");
        gpu.sync();
        let got: Vec<u16> = gpu.down(yt.ptr, ROWS * OUT);
        let mut moved = 0f64;
        for row in 0..ROWS {
            let fired = segments.is_none() || [0, 1, 4, 5].contains(&row);
            let Some(ad) = usize::try_from(ROUTES[row]).ok().filter(|_| fired) else {
                let span = row * OUT..(row + 1) * OUT;
                assert_eq!(got[span.clone()], y_raw[span], "{what}: row {row} is not corrected");
                continue;
            };
            let down = tab(rank, |r| {
                let exact = sum(INPUT, |i| a[(ad * rank + r) * INPUT + i] * x[row * INPUT + i]);
                f64::from((form.decode)((form.encode)(exact as f32)))
            });
            for n in 0..OUT {
                let delta = sum(rank, |r| b[(ad * OUT + n) * rank + r] * down[r]);
                let want = y[row * OUT + n] + delta;
                let have = f64::from((form.decode)(got[row * OUT + n]));
                // Two last places of the answer, and of each down-projected
                // term, which an f32 sum may round to its other neighbour.
                let terms = sum(rank, |r| (b[(ad * OUT + n) * rank + r] * down[r]).abs());
                let room = form.tol * (want.abs() + terms / (rank as f64).sqrt());
                assert!(
                    (have - want).abs() <= room,
                    "{what}: row {row} column {n}: {have} where the host answers {want}"
                );
                moved = moved.max(delta.abs());
            }
        }
        assert!(moved > 0.1, "{what}: the correction moves its rows ({moved})");
    }
}

#[test]
fn a_correction_answers_the_host_reference() {
    let bf16 = Form {
        dtype: Dtype::Bf16,
        encode: to_bf16,
        decode: from_bf16,
        tol: 1.0 / 128.0,
    };
    let f16 = Form {
        dtype: Dtype::F16,
        encode: to_f16,
        decode: from_f16,
        tol: 1.0 / 1024.0,
    };
    for form in [bf16, f16] {
        for rank in [8, 64] {
            answers(&form, rank);
        }
    }
}
