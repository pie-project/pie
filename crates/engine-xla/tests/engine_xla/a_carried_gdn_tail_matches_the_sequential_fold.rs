//! The tail of a buffering fire (`kernels_xla::ssm::carried::gdn_tail_corrected`)
//! is the sequential fold from the commit: each tail row's corrected value
//! is `beta (v - exp(g) S^T k)` with `S` folded through the tail rows before
//! it, zero past a lane's count. Lanes with no tail, part of the rows and
//! all of them; key heads shared by value heads. Within f32 rounding of an
//! f64 reference. Skipped without a PJRT plugin.

use engine_xla::bench::Bench;
use kernels_xla::hlo::Func;
use kernels_xla::ssm::carried::{Tail, gdn_tail_corrected};

const L: usize = 3;
const H: usize = 4;
const HK: usize = 2;
const K: usize = 16;
const V: usize = 16;
const R: usize = 4;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

fn norm(x: &[f64]) -> Vec<f64> {
    let s: f64 = x.iter().map(|v| v * v).sum::<f64>() + 1e-6;
    x.iter().map(|v| v / s.sqrt()).collect()
}

#[test]
fn a_carried_gdn_tail_matches_the_sequential_fold() {
    let mut rng = Rng(23);
    let state = rng.vec(L * H * K * V, 0.3);
    let keys = rng.vec(L * R * HK * K, 1.0);
    let values = rng.vec(L * R * H * V, 1.0);
    let decays: Vec<f32> = rng.vec(L * R * H, 1.0).iter().map(|x| -0.05 - 0.4 * x.abs()).collect();
    let betas: Vec<f32> = rng.vec(L * R * H, 1.0).iter().map(|x| 0.3 + 0.5 * x.abs()).collect();
    let count: Vec<i32> = vec![0, 2, 4];

    let group = H / HK;
    let mut want = vec![0f64; L * R * H * V];
    for l in 0..L {
        for h in 0..H {
            let kh = h / group;
            let mut s: Vec<f64> = (0..K * V)
                .map(|i| f64::from(state[((l * H + h) * K) * V + i]))
                .collect();
            for r in 0..count[l] as usize {
                let k = norm(
                    &(0..K)
                        .map(|d| f64::from(keys[((l * R + r) * HK + kh) * K + d]))
                        .collect::<Vec<_>>(),
                );
                let g = f64::from(decays[(l * R + r) * H + h]).exp();
                let b = f64::from(betas[(l * R + r) * H + h]);
                let u: Vec<f64> = (0..V)
                    .map(|bb| {
                        let a = g * (0..K).map(|a| s[a * V + bb] * k[a]).sum::<f64>();
                        b * (f64::from(values[((l * R + r) * H + h) * V + bb]) - a)
                    })
                    .collect();
                for a in 0..K {
                    for bb in 0..V {
                        s[a * V + bb] = g * s[a * V + bb] + k[a] * u[bb];
                    }
                }
                for bb in 0..V {
                    want[((l * R + r) * H + h) * V + bb] = u[bb];
                }
            }
        }
    }

    let mut bench = Bench::new();
    let t = |b: &mut Bench, x: &[f32]| b.f32(L as u32, (x.len() / L) as u32, x);
    let state_t = t(&mut bench, &state);
    let keys_t = t(&mut bench, &keys);
    let values_t = t(&mut bench, &values);
    let decays_t = t(&mut bench, &decays);
    let betas_t = t(&mut bench, &betas);
    let count_t = bench.i32(1, L as u32, &count);
    let u_o = bench.zeros(dtype::Dtype::F32, L as u32, (R * H * V) as u32);
    let ran = bench
        .run(|ctx| {
            engine_xla::lower::lower(ctx, |lw| {
                let read = |lw: &mut engine_xla::lower::Lower<'_, '_>,
                            t,
                            dims: &[i64]|
                 -> Result<kernels_xla::hlo::Val, kernels_xla::Error> {
                    let v = lw.read(t)?.val;
                    Ok(lw.f().reshape(v, dims)?)
                };
                let (l, h, hk, kk, vv, r) =
                    (L as i64, H as i64, HK as i64, K as i64, V as i64, R as i64);
                let tail = Tail {
                    state: read(lw, state_t, &[l, h, kk, vv])?,
                    keys: read(lw, keys_t, &[l, r, hk, kk])?,
                    values: read(lw, values_t, &[l, r, h, vv])?,
                    decays: read(lw, decays_t, &[l, r, h])?,
                    betas: read(lw, betas_t, &[l, r, h])?,
                    count: read(lw, count_t, &[l])?,
                };
                let f: &mut Func = lw.f();
                let u = gdn_tail_corrected(f, &tail)?;
                let u = f.reshape(u, &[l, r * h * vv])?;
                lw.land(u_o, u)
            })
        })
        .expect("the tail runs");
    if !ran {
        return;
    }
    let got = bench.read_f32(u_o);
    let scale = want.iter().fold(0f64, |m, x| m.max(x.abs())).max(1.0);
    let worst = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (f64::from(*g) - w).abs())
        .fold(0f64, f64::max);
    assert!(worst <= 1e-4 * scale, "u: worst {worst:e} of scale {scale:e}");
}
