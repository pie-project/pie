//! The state + buffer gated-delta step (`kernels_xla::ssm::carried`) is the
//! sequential one: its own row's output and corrected value are what a step
//! from the state folded through every carried row gives, and its state is
//! the committed state folded through the first `min(commit, count)` carried
//! rows, and the own row where the commit takes it. Lanes carrying none,
//! some and the most rows; commits of nothing, part of the carried rows,
//! all of them, and the own row; key heads shared by value heads. Within
//! f32 rounding of an f64 reference. Skipped without a PJRT plugin.

use engine_xla::bench::Bench;
use kernels_xla::hlo::Func;
use kernels_xla::ssm::carried::{Carried, gdn_step_carried};

const L: usize = 4;
const H: usize = 4;
const HK: usize = 2;
const K: usize = 16;
const V: usize = 16;
const R: usize = 5;
const Q_SCALE: f32 = 0.25;

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
fn a_carried_gdn_step_matches_the_sequential_fold() {
    let mut rng = Rng(11);
    let state = rng.vec(L * H * K * V, 0.3);
    let keys = rng.vec(L * R * HK * K, 1.0);
    let values = rng.vec(L * R * H * V, 0.5);
    let decays: Vec<f32> = rng
        .vec(L * R * H, 1.0)
        .iter()
        .map(|x| -0.05 - 0.4 * x.abs())
        .collect();
    let count: Vec<i32> = vec![0, 2, 5, 3];
    let commit: Vec<i32> = vec![0, 3, 2, 4];
    let q = rng.vec(L * HK * K, 1.0);
    let k = rng.vec(L * HK * K, 1.0);
    let v = rng.vec(L * H * V, 1.0);
    let decay: Vec<f32> = rng
        .vec(L * H, 1.0)
        .iter()
        .map(|x| -0.05 - 0.3 * x.abs())
        .collect();
    let beta: Vec<f32> = rng
        .vec(L * H, 1.0)
        .iter()
        .map(|x| 0.3 + 0.5 * x.abs())
        .collect();

    // the f64 reference, step by step
    let group = H / HK;
    let mut want_y = vec![0f64; L * H * V];
    let mut want_u = vec![0f64; L * H * V];
    let mut want_s = vec![0f64; L * H * K * V];
    for l in 0..L {
        for h in 0..H {
            let kh = h / group;
            let s0: Vec<f64> = (0..K * V)
                .map(|i| f64::from(state[((l * H + h) * K) * V + i]))
                .collect();
            let key = |r: usize| -> Vec<f64> {
                norm(
                    &(0..K)
                        .map(|d| f64::from(keys[((l * R + r) * HK + kh) * K + d]))
                        .collect::<Vec<_>>(),
                )
            };
            let uval = |r: usize| -> Vec<f64> {
                (0..V)
                    .map(|d| f64::from(values[((l * R + r) * H + h) * V + d]))
                    .collect()
            };
            let g = |r: usize| f64::from(decays[(l * R + r) * H + h]);
            let fold = |s: &mut Vec<f64>, gl: f64, kk: &[f64], uu: &[f64]| {
                for a in 0..K {
                    for b in 0..V {
                        s[a * V + b] = gl.exp() * s[a * V + b] + kk[a] * uu[b];
                    }
                }
            };
            let n = count[l] as usize;
            let mut sn = s0.clone();
            for r in 0..n {
                fold(&mut sn, g(r), &key(r), &uval(r));
            }
            let qn: Vec<f64> = norm(
                &(0..K)
                    .map(|d| f64::from(q[(l * HK + kh) * K + d]))
                    .collect::<Vec<_>>(),
            )
            .iter()
            .map(|x| x * f64::from(Q_SCALE))
            .collect();
            let kn = norm(
                &(0..K)
                    .map(|d| f64::from(k[(l * HK + kh) * K + d]))
                    .collect::<Vec<_>>(),
            );
            let gn = f64::from(decay[l * H + h]);
            let b = f64::from(beta[l * H + h]);
            let st = |x: &[f64]| -> Vec<f64> {
                (0..V)
                    .map(|bb| gn.exp() * (0..K).map(|a| sn[a * V + bb] * x[a]).sum::<f64>())
                    .collect()
            };
            let a = st(&kn);
            let o = st(&qn);
            let u: Vec<f64> = (0..V)
                .map(|d| b * (f64::from(v[(l * H + h) * V + d]) - a[d]))
                .collect();
            let kq: f64 = (0..K).map(|d| kn[d] * qn[d]).sum();
            for d in 0..V {
                want_u[(l * H + h) * V + d] = u[d];
                want_y[(l * H + h) * V + d] = o[d] + kq * u[d];
            }
            let c = commit[l] as usize;
            let mut sc = s0.clone();
            for r in 0..c.min(n) {
                fold(&mut sc, g(r), &key(r), &uval(r));
            }
            if c == n + 1 {
                fold(&mut sc, gn, &kn, &u);
            }
            for i in 0..K * V {
                want_s[((l * H + h) * K) * V + i] = sc[i];
            }
        }
    }

    let mut bench = Bench::new();
    let t = |b: &mut Bench, x: &[f32], rows: usize| b.f32(rows as u32, (x.len() / rows) as u32, x);
    let state_t = t(&mut bench, &state, L);
    let keys_t = t(&mut bench, &keys, L);
    let values_t = t(&mut bench, &values, L);
    let decays_t = t(&mut bench, &decays, L);
    let count_t = bench.i32(1, L as u32, &count);
    let commit_t = bench.i32(1, L as u32, &commit);
    let q_t = t(&mut bench, &q, L);
    let k_t = t(&mut bench, &k, L);
    let v_t = t(&mut bench, &v, L);
    let decay_t = t(&mut bench, &decay, L);
    let beta_t = t(&mut bench, &beta, L);
    let y_o = bench.zeros(dtype::Dtype::F32, L as u32, (H * V) as u32);
    let u_o = bench.zeros(dtype::Dtype::F32, L as u32, (H * V) as u32);
    let s_o = bench.zeros(dtype::Dtype::F32, L as u32, (H * K * V) as u32);
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
                let c = Carried {
                    state: read(lw, state_t, &[l, h, kk, vv])?,
                    keys: read(lw, keys_t, &[l, r, hk, kk])?,
                    values: read(lw, values_t, &[l, r, h, vv])?,
                    decays: read(lw, decays_t, &[l, r, h])?,
                    count: read(lw, count_t, &[l])?,
                    commit: read(lw, commit_t, &[l])?,
                    q: read(lw, q_t, &[l, hk, kk])?,
                    k: read(lw, k_t, &[l, hk, kk])?,
                    v: read(lw, v_t, &[l, h, vv])?,
                    decay: read(lw, decay_t, &[l, h])?,
                    beta: read(lw, beta_t, &[l, h])?,
                };
                let f: &mut Func = lw.f();
                let out = gdn_step_carried(f, &c, Q_SCALE)?;
                let y = f.reshape(out.y, &[l, h * vv])?;
                let u = f.reshape(out.u, &[l, h * vv])?;
                let s = f.reshape(out.state, &[l, h * kk * vv])?;
                lw.land(y_o, y)?;
                lw.land(u_o, u)?;
                lw.land(s_o, s)
            })
        })
        .expect("the step runs");
    if !ran {
        return;
    }
    let close = |got: &[f32], want: &[f64], what: &str| {
        let scale = want.iter().fold(0f64, |m, x| m.max(x.abs())).max(1.0);
        let worst = got
            .iter()
            .zip(want)
            .map(|(g, w)| (f64::from(*g) - w).abs())
            .fold(0f64, f64::max);
        assert!(
            worst <= 1e-4 * scale,
            "{what}: worst {worst:e} of scale {scale:e}"
        );
    };
    close(&bench.read_f32(y_o), &want_y, "y");
    close(&bench.read_f32(u_o), &want_u, "u");
    close(&bench.read_f32(s_o), &want_s, "the state at the commit");
}
