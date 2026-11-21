//! The conv half of the state + buffer step (`kernels_xla::ssm::carried_conv`)
//! is the sequential one: the own row's output reads the committed window,
//! the carried rows and the own row at its taps, and the window stored is
//! the one at the lane's commit. Lanes carrying none, some and the most
//! rows; commits of nothing, part, all and the own row; a dilation past one.
//! Skipped without a PJRT plugin.

use engine_xla::bench::Bench;
use kernels_xla::hlo::Func;
use kernels_xla::ssm::carried_conv::{CarriedConv, conv_step_carried};

const L: usize = 4;
const C: usize = 8;
const K: usize = 3;
const DIL: usize = 2;
const SPAN: usize = (K - 1) * DIL;
const R: usize = 5;

#[test]
fn a_carried_conv_step_matches_the_sequential_window() {
    let val = |i: usize| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0;
    let window: Vec<f32> = (0..L * SPAN * C).map(|i| val(i + 1)).collect();
    let carried: Vec<f32> = (0..L * R * C).map(|i| val(i + 7000)).collect();
    let own: Vec<f32> = (0..L * C).map(|i| val(i + 9000)).collect();
    let weight: Vec<f32> = (0..C * K).map(|i| val(i + 11000)).collect();
    let bias: Vec<f32> = (0..C).map(|i| val(i + 12000) * 0.1).collect();
    let count: Vec<i32> = vec![0, 2, 5, 4];
    let commit: Vec<i32> = vec![0, 1, 5, 0];
    // reference
    let mut want_y = vec![0f64; L * C];
    let mut want_w = vec![0f64; L * SPAN * C];
    for l in 0..L {
        let n = count[l] as i64;
        let at = |t: i64, c: usize| -> f64 {
            if t < 0 {
                f64::from(window[(l * SPAN + (SPAN as i64 + t) as usize) * C + c])
            } else if t < n {
                f64::from(carried[(l * R + t as usize) * C + c])
            } else {
                f64::from(own[l * C + c])
            }
        };
        for c in 0..C {
            let mut s = f64::from(bias[c]);
            for k in 0..K {
                s += f64::from(weight[c * K + k]) * at(n - ((K - 1 - k) * DIL) as i64, c);
            }
            want_y[l * C + c] = s / (1.0 + (-s).exp());
            for p in 0..SPAN {
                want_w[(l * SPAN + p) * C + c] =
                    at(i64::from(commit[l]) - SPAN as i64 + p as i64, c);
            }
        }
    }
    let mut b = Bench::new();
    let w_t = b.f32(L as u32, (SPAN * C) as u32, &window);
    let c_t = b.f32(L as u32, (R * C) as u32, &carried);
    let o_t = b.f32(L as u32, C as u32, &own);
    let k_t = b.f32(C as u32, K as u32, &weight);
    let bias_t = b.f32(1, C as u32, &bias);
    let n_t = b.i32(1, L as u32, &count);
    let m_t = b.i32(1, L as u32, &commit);
    let y_o = b.zeros(dtype::Dtype::F32, L as u32, C as u32);
    let win_o = b.zeros(dtype::Dtype::F32, L as u32, (SPAN * C) as u32);
    let ran = b
        .run(|ctx| {
            engine_xla::lower::lower(ctx, |lw| {
                let read = |lw: &mut engine_xla::lower::Lower<'_, '_>,
                            t,
                            dims: &[i64]|
                 -> Result<kernels_xla::hlo::Val, kernels_xla::Error> {
                    let v = lw.read(t)?.val;
                    Ok(lw.f().reshape(v, dims)?)
                };
                let (l, c, k, sp, r) = (L as i64, C as i64, K as i64, SPAN as i64, R as i64);
                let cc = CarriedConv {
                    window: read(lw, w_t, &[l, sp, c])?,
                    carried: read(lw, c_t, &[l, r, c])?,
                    own: read(lw, o_t, &[l, c])?,
                    count: read(lw, n_t, &[l])?,
                    commit: read(lw, m_t, &[l])?,
                    weight: read(lw, k_t, &[c, k])?,
                    bias: Some(read(lw, bias_t, &[c])?),
                };
                let f: &mut Func = lw.f();
                let out = conv_step_carried(f, &cc, DIL as i64)?;
                let win = f.reshape(out.window, &[l, sp * c])?;
                lw.land(y_o, out.y)?;
                lw.land(win_o, win)
            })
        })
        .expect("the step runs");
    if !ran {
        return;
    }
    let close = |got: &[f32], want: &[f64], what: &str| {
        let worst = got
            .iter()
            .zip(want)
            .map(|(g, w)| (f64::from(*g) - w).abs())
            .fold(0f64, f64::max);
        assert!(worst <= 1e-5, "{what}: worst {worst:e}");
    };
    close(&b.read_f32(y_o), &want_y, "the own output");
    close(&b.read_f32(win_o), &want_w, "the window at the commit");
}
