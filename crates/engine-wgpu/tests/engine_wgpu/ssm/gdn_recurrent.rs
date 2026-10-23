//! The gated delta recurrence on the first wgpu adapter, against an f64 host
//! reference: the decode step, the chunked walk (an empty request among
//! them) and the committed window (replayed rows, a partial commit, a commit
//! of none, a lane with no slot), with 128-wide keys (the register-state
//! kernels) and 32-wide ones (the state walked in memory), at the baseline
//! tier. The f32 outputs and every slot's state are compared.

use kernels_slang::Tier;
use kernels_slang::ssm::gdn_recurrent::Tactic;
use kernels_slang::{RaggedTensor, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, Out, landed, noise};

const SLOTS: usize = 3;

#[derive(Clone, Copy)]
struct Shape {
    k_heads: usize,
    v_heads: usize,
    dk: usize,
    dv: usize,
}

impl Shape {
    fn pitch(self) -> usize {
        2 * self.k_heads * self.dk + self.v_heads * self.dv
    }

    fn head(self) -> usize {
        self.dk * self.dv
    }

    fn slab(self) -> usize {
        self.v_heads * self.head()
    }

    /// Row `t` of `qkv`/`gates` stepped through the slot state `s`, its
    /// output written to `y`.
    fn token(self, qkv: &[f64], gates: &[f64], t: usize, s: &mut [f64], y: &mut [f64]) {
        let keys = self.k_heads * self.dk;
        let row = &qkv[t * self.pitch()..][..self.pitch()];
        for hv in 0..self.v_heads {
            let hk = hv / (self.v_heads / self.k_heads);
            let q = &row[hk * self.dk..][..self.dk];
            let k = &row[keys + hk * self.dk..][..self.dk];
            let v = &row[2 * keys + hv * self.dv..][..self.dv];
            let qinv = 1.0
                / (q.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt()
                / (self.dk as f64).sqrt();
            let kinv = 1.0 / (k.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
            let decay = gates[t * 2 * self.v_heads + hv].exp();
            let beta = gates[t * 2 * self.v_heads + self.v_heads + hv];
            let state = &mut s[hv * self.head()..][..self.head()];
            for c in 0..self.dv {
                let cell = &mut state[c * self.dk..][..self.dk];
                let mut mem = 0.0;
                for i in 0..self.dk {
                    cell[i] *= decay;
                    mem += cell[i] * k[i] * kinv;
                }
                let delta = (v[c] - mem) * beta;
                let mut acc = 0.0;
                for i in 0..self.dk {
                    cell[i] += k[i] * kinv * delta;
                    acc += cell[i] * q[i] * qinv;
                }
                y[(t * self.v_heads + hv) * self.dv + c] = acc;
            }
        }
    }
}

fn gates_of(seed: u64, rows: usize, v_heads: usize) -> Vec<f32> {
    let raw = noise(seed, rows * 2 * v_heads);
    let mut g = vec![0.0; raw.len()];
    for t in 0..rows {
        for h in 0..v_heads {
            let at = t * 2 * v_heads;
            g[at + h] = -0.25 * raw[at + h].abs();
            g[at + v_heads + h] = 1.0 / (1.0 + (-raw[at + v_heads + h]).exp());
        }
    }
    g
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

#[test]
fn the_gdn_recurrence_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |buf: u32, r: usize, w: usize, d: Dtype| Tensor::new(buf, r as u32, w as u32, d);
    let shapes = [
        Shape {
            k_heads: 1,
            v_heads: 2,
            dk: 128,
            dv: 64,
        },
        Shape {
            k_heads: 2,
            v_heads: 4,
            dk: 32,
            dv: 48,
        },
    ];
    for (n, sh) in shapes.into_iter().enumerate() {
        for tier in [Tier::Baseline] {
            let tactic = Tactic {
                tier,
                register_min_rows: 0,
            };
            let what = |form: &str| format!("gdn {form}, k {} at {tier:?}", sh.dk);
            let slab = sh.slab();
            let bank0: Vec<f32> = noise(5 + n as u64, SLOTS * slab)
                .iter()
                .map(|v| v * 0.25)
                .collect();
            let bank64 = f64s(&bank0);
            let bank0 = &bank0;
            let pool = |bank: u32, slots: u32, rows: usize| kernels_slang::ssm::StatePool {
                bank: t(bank, SLOTS, slab, Dtype::F32),
                slots: t(slots, rows, 1, Dtype::U32),
            };
            let y_n = |rows: usize| rows * sh.v_heads * sh.dv;
            let fresh = |rows: usize, work: usize| {
                move |g: &mut Gpu| -> Vec<Out> {
                    vec![
                        (g.up_f32(&vec![0.0; y_n(rows)]), y_n(rows), true),
                        (g.up_f32(bank0), SLOTS * slab, true),
                        (g.up_f32(&vec![0.0; work.max(1)]), work, true),
                    ]
                }
            };
            let inputs = |g: &mut Gpu, seed: u64, rows: usize| {
                let (qb, qr) = g.up(&noise(seed, rows * sh.pitch()));
                let gr = gates_of(seed + 1, rows, sh.v_heads);
                let gb = g.up_f32(&gr);
                (
                    t(qb, rows, sh.pitch(), Dtype::Bf16),
                    f64s(&qr),
                    t(gb, rows, 2 * sh.v_heads, Dtype::F32),
                    f64s(&gr),
                )
            };
            let y = |buf: u32, rows: usize| t(buf, rows, sh.v_heads * sh.dv, Dtype::F32);
            let (kh, vh, dk, dv) = (
                sh.k_heads as u32,
                sh.v_heads as u32,
                sh.dk as u32,
                sh.dv as u32,
            );

            // The decode step: two rows on slots 2 and 0.
            let slot_of = [2u32, 0];
            let (qkv, qr, gates, gr) = inputs(&mut gpu, 10, 2);
            let sb = gpu.up_u32(&slot_of);
            let mut want_y = vec![0.0; y_n(2)];
            let mut want_s = bank64.clone();
            for (r, &s) in slot_of.iter().enumerate() {
                let s = s as usize;
                sh.token(&qr, &gr, r, &mut want_s[s * slab..][..slab], &mut want_y);
            }
            landed(
                &mut gpu,
                &what("step"),
                &fresh(2, 0),
                &[want_y, want_s],
                &|g, o| {
                    g.run(|c| {
                        kernels_slang::ssm::gdn_recurrent::gated_delta(
                            c,
                            tactic,
                            qkv,
                            gates,
                            &pool(o[1].0, sb, 2),
                            kh,
                            vh,
                            dk,
                            dv,
                            y(o[0].0, 2),
                        )
                    })
                },
            );

            // The chunked walk: requests of 3, 0 and 2 rows on slots 1 and 2.
            let indptr = [0i32, 3, 3, 5];
            let rows = 5;
            let slot_rows = [1u32, 1, 1, 2, 2];
            let (qkv, qr, gates, gr) = inputs(&mut gpu, 20, rows);
            let (ib, sb) = (gpu.up_i32(&indptr), gpu.up_u32(&slot_rows));
            let mut want_y = vec![0.0; y_n(rows)];
            let mut want_s = bank64.clone();
            for r in 0..3 {
                for row in indptr[r] as usize..indptr[r + 1] as usize {
                    let s = slot_rows[indptr[r] as usize] as usize;
                    sh.token(&qr, &gr, row, &mut want_s[s * slab..][..slab], &mut want_y);
                }
            }
            let ragged = RaggedTensor {
                data: qkv,
                indptr: t(ib, 4, 1, Dtype::I32),
            };
            landed(
                &mut gpu,
                &what("chunked"),
                &fresh(rows, 0),
                &[want_y, want_s],
                &|g, o| {
                    g.run(|c| {
                        kernels_slang::ssm::gdn_recurrent::gated_delta_chunked(
                            c,
                            tactic,
                            kernels_slang::RaggedTensor {
                                data: ragged.data,
                                indptr: ragged.indptr,
                            },
                            gates,
                            &pool(o[1].0, sb, rows),
                            kh,
                            vh,
                            dk,
                            dv,
                            y(o[0].0, rows),
                        )
                    })
                },
            );

            // The committed window, lanes 1..5: 2 own rows after 1 replayed,
            // kept as of row 2 on slot 2; 1 after 1, kept whole on slot 0; 1
            // after 0 with no slot; 1 after 0 kept as of none on slot 1.
            let lane0 = 1usize;
            let indptr = [0i32, 2, 3, 4, 5];
            let replay = [7i32, 1, 1, 0, 0];
            let commit = [7i32, 2, 2, 1, 0];
            let slots = [7i32, 2, 0, -1, 1];
            let rows = 7;
            let (qkv, qr, gates, gr) = inputs(&mut gpu, 30, rows);
            let ib = gpu.up_i32(&indptr);
            let (rb, cb, sb) = (gpu.up_i32(&replay), gpu.up_i32(&commit), gpu.up_i32(&slots));
            let mut want_y = vec![0.0; y_n(rows)];
            let mut want_s = bank64.clone();
            let mut begin = 0usize;
            for r in 0..4 {
                let lane = lane0 + r;
                let span = (indptr[r + 1] - indptr[r] + replay[lane]) as usize;
                if slots[lane] >= 0 {
                    let s = slots[lane] as usize;
                    let keep = (commit[lane] as usize).min(span);
                    let mut walked = bank64[s * slab..][..slab].to_vec();
                    for i in 0..span {
                        sh.token(&qr, &gr, begin + i, &mut walked, &mut want_y);
                        if i + 1 == keep {
                            want_s[s * slab..][..slab].copy_from_slice(&walked);
                        }
                    }
                }
                begin += span;
            }
            let i32s = |buf: u32, n: usize| t(buf, n, 1, Dtype::I32);
            let committed = kernels_slang::ssm::Committed {
                replay: i32s(rb, 5),
                commit: i32s(cb, 5),
                slots: i32s(sb, 5),
                lane0: lane0 as u32,
            };
            let window = i32s(ib, 5);
            let work = 5 * slab;
            let work_t = |buf: u32| t(buf, 5, slab, Dtype::F32);
            landed(
                &mut gpu,
                &what("committed"),
                &fresh(rows, work),
                &[want_y, want_s],
                &|g, o| {
                    g.run(|c| {
                        kernels_slang::ssm::gdn_recurrent::gated_delta_committed(
                            c,
                            tactic,
                            qkv,
                            window,
                            &committed,
                            gates,
                            &pool(o[1].0, sb, 5),
                            work_t(o[2].0),
                            kh,
                            vh,
                            dk,
                            dv,
                            y(o[0].0, rows),
                        )
                    })
                },
            );
        }
    }
}
