//! Kimi delta attention on the first Vulkan device, against an f64 host
//! reference: the decode step, the chunked walk (an empty request among
//! them) and the committed window (replayed rows, a partial commit, a lane
//! with no slot), with the softplus gate and the floored sigmoid gate, at
//! landed tiers. The f32 outputs and every slot's state are compared.

use kernels_slang::Tier;
use kernels_slang::{RaggedTensor, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, Out, landed, noise};

const SLOTS: usize = 3;
const HEADS: usize = 2;
const D: usize = 32;
const EPS: f32 = 1e-6;

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

/// The host side of one fire's inputs.
struct Host {
    mixed: Vec<f64>,
    f: Vec<f64>,
    b: Vec<f64>,
    dt_bias: Vec<f64>,
    a_log: Vec<f64>,
    floor: f64,
}

impl Host {
    /// Row `t` stepped through the slot state `s`, its output written to `y`.
    fn token(&self, t: usize, s: &mut [f64], y: &mut [f64]) {
        let wide = HEADS * D;
        let row = &self.mixed[t * 3 * wide..][..3 * wide];
        let eps = f64::from(EPS);
        for h in 0..HEADS {
            let q = &row[h * D..][..D];
            let k = &row[wide + h * D..][..D];
            let v = &row[2 * wide + h * D..][..D];
            let qinv =
                1.0 / (q.iter().map(|x| x * x).sum::<f64>() + eps).sqrt() / (D as f64).sqrt();
            let kinv = 1.0 / (k.iter().map(|x| x * x).sum::<f64>() + eps).sqrt();
            let alpha = self.a_log[h].exp();
            let g: Vec<f64> = (0..D)
                .map(|i| {
                    let z = self.f[t * wide + h * D + i] + self.dt_bias[h * D + i];
                    if self.floor != 0.0 {
                        (self.floor / (1.0 + (-alpha * z).exp())).exp()
                    } else {
                        let sp = if z > 20.0 { z } else { z.exp().ln_1p() };
                        (-alpha * sp).exp()
                    }
                })
                .collect();
            let beta = 1.0 / (1.0 + (-self.b[t * HEADS + h]).exp());
            let state = &mut s[h * D * D..][..D * D];
            for vi in 0..D {
                let cell = &mut state[vi * D..][..D];
                let mut mem = 0.0;
                for i in 0..D {
                    cell[i] *= g[i];
                    mem += cell[i] * k[i] * kinv;
                }
                let delta = (v[vi] - mem) * beta;
                let mut acc = 0.0;
                for i in 0..D {
                    cell[i] += k[i] * kinv * delta;
                    acc += cell[i] * q[i] * qinv;
                }
                y[t * wide + h * D + vi] = acc;
            }
        }
    }
}

/// The device planes of one fire.
#[derive(Clone, Copy)]
struct Planes {
    mixed: Tensor,
    f: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
}

impl Planes {
    fn gates(&self, floor: f32, f: Tensor, b: Tensor) -> kernels_slang::ssm::kda::Gates {
        kernels_slang::ssm::kda::Gates {
            f,
            b,
            dt_bias: self.dt_bias,
            a_log: self.a_log,
            heads: HEADS as u32,
            head_dim: D as u32,
            norm_eps: EPS,
            gate_floor: floor,
        }
    }
}

fn upload(gpu: &mut Gpu, seed: u64, rows: usize, floor: f32) -> (Planes, Host) {
    let t = |buf: u32, r: usize, w: usize, d: Dtype| Tensor::new(buf, r as u32, w as u32, d);
    let wide = HEADS * D;
    let (mb, mr) = gpu.up(&noise(seed, rows * 3 * wide));
    let (fb, fr) = gpu.up(&noise(seed + 1, rows * wide));
    let (bb, br) = gpu.up(&noise(seed + 2, rows * HEADS));
    let dt = noise(seed + 3, wide);
    let al: Vec<f32> = noise(seed + 4, HEADS).iter().map(|v| v * 0.5).collect();
    let (dtb, alb) = (gpu.up_f32(&dt), gpu.up_f32(&al));
    (
        Planes {
            mixed: t(mb, rows, 3 * wide, Dtype::Bf16),
            f: t(fb, rows, wide, Dtype::Bf16),
            b: t(bb, rows, HEADS, Dtype::Bf16),
            dt_bias: t(dtb, 1, wide, Dtype::F32),
            a_log: t(alb, 1, HEADS, Dtype::F32),
        },
        Host {
            mixed: f64s(&mr),
            f: f64s(&fr),
            b: f64s(&br),
            dt_bias: f64s(&dt),
            a_log: f64s(&al),
            floor: f64::from(floor),
        },
    )
}

#[test]
fn the_kda_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |buf: u32, r: usize, w: usize, d: Dtype| Tensor::new(buf, r as u32, w as u32, d);
    let slab = HEADS * D * D;
    let wide = HEADS * D;
    let bank0: Vec<f32> = noise(5, SLOTS * slab).iter().map(|v| v * 0.25).collect();
    let bank64 = f64s(&bank0);
    let bank0 = &bank0;
    let pool = |bank: u32, slots: u32, rows: usize| kernels_slang::ssm::StatePool {
        bank: t(bank, SLOTS, slab, Dtype::F32),
        slots: t(slots, rows, 1, Dtype::U32),
    };
    let fresh = |rows: usize, work: usize| {
        move |g: &mut Gpu| -> Vec<Out> {
            vec![
                (g.up_f32(&vec![0.0; rows * wide]), rows * wide, true),
                (g.up_f32(bank0), SLOTS * slab, true),
                (g.up_f32(&vec![0.0; work.max(1)]), work, true),
            ]
        }
    };
    let y = |buf: u32, rows: usize| t(buf, rows, wide, Dtype::F32);
    for floor in [0.0f32, -5.0] {
        for tier in [Tier::Baseline, Tier::Subgroup] {
            let what = |form: &str| format!("kda {form}, floor {floor} at {tier:?}");
            let fl = floor;

            // The decode step: two rows on slots 2 and 0.
            let slot_of = [2u32, 0];
            let (p, host) = upload(&mut gpu, 10, 2, floor);
            let sb = gpu.up_u32(&slot_of);
            let mut want_y = vec![0.0; 2 * wide];
            let mut want_s = bank64.clone();
            for (r, &s) in slot_of.iter().enumerate() {
                let s = s as usize;
                host.token(r, &mut want_s[s * slab..][..slab], &mut want_y);
            }
            landed(
                &mut gpu,
                &what("step"),
                &fresh(2, 0),
                &[want_y, want_s],
                &|g, o| {
                    g.run(|c| {
                        kernels_slang::ssm::kda::kda(
                            c,
                            tier,
                            p.mixed,
                            &p.gates(fl, p.f, p.b),
                            &pool(o[1].0, sb, 2),
                            y(o[0].0, 2),
                        )
                    })
                },
            );

            // The chunked walk: requests of 3, 0 and 2 rows on slots 1 and 2.
            let indptr = [0i32, 3, 3, 5];
            let rows = 5;
            let slot_rows = [1u32, 1, 1, 2, 2];
            let (p, host) = upload(&mut gpu, 20, rows, floor);
            let (ib, sb) = (gpu.up_i32(&indptr), gpu.up_u32(&slot_rows));
            let mut want_y = vec![0.0; rows * wide];
            let mut want_s = bank64.clone();
            for r in 0..3 {
                for row in indptr[r] as usize..indptr[r + 1] as usize {
                    let s = slot_rows[indptr[r] as usize] as usize;
                    host.token(row, &mut want_s[s * slab..][..slab], &mut want_y);
                }
            }
            let ragged = RaggedTensor {
                data: p.mixed,
                indptr: t(ib, 4, 1, Dtype::I32),
            };
            landed(
                &mut gpu,
                &what("chunked"),
                &fresh(rows, 0),
                &[want_y, want_s],
                &|g, o| {
                    g.run(|c| {
                        kernels_slang::ssm::kda::kda_chunked(
                            c,
                            tier,
                            kernels_slang::RaggedTensor {
                                data: ragged.data,
                                indptr: ragged.indptr,
                            },
                            &p.gates(fl, p.f, p.b),
                            &pool(o[1].0, sb, rows),
                            y(o[0].0, rows),
                        )
                    })
                },
            );

            // The committed window, lanes 1..4: 2 own rows after 1 replayed,
            // kept as of row 2 on slot 2; 1 after 1, kept whole on slot 0; 1
            // after 0 with no slot.
            let lane0 = 1usize;
            let indptr = [0i32, 2, 3, 4];
            let replay = [7i32, 1, 1, 0];
            let commit = [7i32, 2, 2, 1];
            let slots = [7i32, 2, 0, -1];
            let rows = 6;
            let (p, host) = upload(&mut gpu, 30, rows, floor);
            let ib = gpu.up_i32(&indptr);
            let (rb, cb, sb) = (gpu.up_i32(&replay), gpu.up_i32(&commit), gpu.up_i32(&slots));
            let mut want_y = vec![0.0; rows * wide];
            let mut want_s = bank64.clone();
            let mut begin = 0usize;
            for r in 0..3 {
                let lane = lane0 + r;
                let span = (indptr[r + 1] - indptr[r] + replay[lane]) as usize;
                if slots[lane] >= 0 {
                    let s = slots[lane] as usize;
                    let keep = (commit[lane] as usize).min(span);
                    let mut walked = bank64[s * slab..][..slab].to_vec();
                    for i in 0..span {
                        host.token(begin + i, &mut walked, &mut want_y);
                        if i + 1 == keep {
                            want_s[s * slab..][..slab].copy_from_slice(&walked);
                        }
                    }
                }
                begin += span;
            }
            let i32s = |buf: u32, n: usize| t(buf, n, 1, Dtype::I32);
            let committed = kernels_slang::ssm::Committed {
                replay: i32s(rb, 4),
                commit: i32s(cb, 4),
                slots: i32s(sb, 4),
                lane0: lane0 as u32,
            };
            let window = i32s(ib, 4);
            let work = 4 * slab;
            let work_t = |buf: u32| t(buf, 4, slab, Dtype::F32);
            landed(
                &mut gpu,
                &what("committed"),
                &fresh(rows, work),
                &[want_y, want_s],
                &|g, o| {
                    g.run(|c| {
                        kernels_slang::ssm::kda::kda_committed(
                            c,
                            tier,
                            p.mixed,
                            window,
                            &committed,
                            &p.gates(fl, p.f, p.b),
                            &pool(o[1].0, sb, 4),
                            work_t(o[2].0),
                            y(o[0].0, rows),
                        )
                    })
                },
            );
        }
    }
}
