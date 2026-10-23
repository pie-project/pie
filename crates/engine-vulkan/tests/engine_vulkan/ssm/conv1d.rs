//! The causal conv1d on the first Vulkan device, beside the hand-written kernels,
//! against an f64 host reference: the decode step, the chunked walk (an
//! empty request among them) and the committed window (replayed rows, a
//! partial commit, a lane with no slot), at dilations 1 and 2. Both the
//! SiLU'd rows and the rolled history are compared.

use kernels_vulkan::attn::ssm;
use kernels_vulkan::tensor::RecurrentPool;
use kernels_vulkan::{RaggedTensor, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, Out, both, noise};

const C: usize = 6;
const W: usize = 4;
const SLOTS: usize = 3;

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

fn silu(z: f64) -> f64 {
    z / (1.0 + (-z).exp())
}

/// Rows `xs` convolved over the history `past` (`hist` rows of `C`): the
/// outputs, and the history as of row `keep` (unchanged when `keep == 0`).
fn walk(past: &[f64], xs: &[f64], w: &[f64], dil: usize, keep: usize) -> (Vec<f64>, Vec<f64>) {
    let hist = (W - 1) * dil + 1;
    let span = xs.len() / C;
    let tap = |src: isize, c: usize| {
        if src < 0 {
            past[(hist as isize + src) as usize * C + c]
        } else {
            xs[src as usize * C + c]
        }
    };
    let mut y = vec![0.0; span * C];
    for t in 0..span {
        for c in 0..C {
            let acc: f64 = (0..W)
                .map(|k| tap(t as isize - ((W - 1 - k) * dil) as isize, c) * w[c * W + k])
                .sum();
            y[t * C + c] = silu(acc);
        }
    }
    let mut kept = past.to_vec();
    if keep > 0 {
        for s in 0..hist {
            for c in 0..C {
                kept[s * C + c] = tap(keep as isize - hist as isize + s as isize, c);
            }
        }
    }
    (y, kept)
}

#[test]
fn the_conv1d_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |buf: u32, r: usize, w: usize, d: Dtype| Tensor::new(buf, r as u32, w as u32, d);
    let (wb, wr) = gpu.up(&noise(1, C * W));
    let wr: Vec<f64> = wr.iter().map(|v| f64::from(*v)).collect();
    let wt = t(wb, C, W, Dtype::Bf16);
    for dil in [1usize, 2] {
        let hist = (W - 1) * dil + 1;
        let slab = hist * C;
        let bank0: Vec<f32> = noise(2 + dil as u64, SLOTS * slab);
        let bank64: Vec<f64> = bank0.iter().map(|v| f64::from(*v)).collect();
        let pool = |bank: u32, slots: u32, rows: usize| RecurrentPool {
            state: t(bank, SLOTS, slab, Dtype::F32),
            slots: t(slots, rows, 1, Dtype::U32),
            conv_state: t(bank, SLOTS, slab, Dtype::F32),
            new_conv_state: t(bank, SLOTS, slab, Dtype::F32),
        };
        let sp = |p: &RecurrentPool| kernels_slang::ssm::StatePool {
            bank: st(p.state),
            slots: st(p.slots),
        };
        let bank0 = &bank0;
        let fresh = |rows: usize| {
            move |g: &mut Gpu| -> Vec<Out> {
                vec![
                    (g.zeros(rows * C), rows * C, false),
                    (g.up_f32(bank0), SLOTS * slab, true),
                ]
            }
        };

        // The decode step: two rows on slots 2 and 0.
        let slot_of = [2u32, 0];
        let (xb, xr) = gpu.up(&noise(10 + dil as u64, 2 * C));
        let xr: Vec<f64> = xr.iter().map(|v| f64::from(*v)).collect();
        let sb = gpu.up_u32(&slot_of);
        let mut want_y = vec![0.0; 2 * C];
        let mut want_s = bank64.clone();
        for (r, &s) in slot_of.iter().enumerate() {
            let s = s as usize;
            let (y, kept) = walk(&bank64[s * slab..][..slab], &xr[r * C..][..C], &wr, dil, 1);
            want_y[r * C..][..C].copy_from_slice(&y);
            want_s[s * slab..][..slab].copy_from_slice(&kept);
        }
        let x = t(xb, 2, C, Dtype::Bf16);
        both(
            &mut gpu,
            &format!("conv1d step, dilation {dil}"),
            &fresh(2),
            &[want_y, want_s],
            &|g, o| {
                g.run(|c| {
                    ssm::causal_conv1d(
                        c,
                        x,
                        wt,
                        &pool(o[1].0, sb, 2),
                        W as u32,
                        dil as u32,
                        t(o[0].0, 2, C, Dtype::Bf16),
                    )
                })
            },
            &|g, o| {
                g.run_slang(|c| {
                    kernels_slang::ssm::conv1d::conv1d(
                        c,
                        st(x),
                        st(wt),
                        &sp(&pool(o[1].0, sb, 2)),
                        W as u32,
                        dil as u32,
                        st(t(o[0].0, 2, C, Dtype::Bf16)),
                    )
                })
            },
        );

        // The chunked walk: requests of 3, 0 and 4 rows on slots 1 and 2.
        let indptr = [0i32, 3, 3, 7];
        let rows = 7;
        let slot_rows = [1u32, 1, 1, 2, 2, 2, 2];
        let (xb, xr) = gpu.up(&noise(20 + dil as u64, rows * C));
        let xr: Vec<f64> = xr.iter().map(|v| f64::from(*v)).collect();
        let (ib, sb) = (gpu.up_i32(&indptr), gpu.up_u32(&slot_rows));
        let mut want_y = vec![0.0; rows * C];
        let mut want_s = bank64.clone();
        for r in 0..3 {
            let (b, e) = (indptr[r] as usize, indptr[r + 1] as usize);
            if e == b {
                continue;
            }
            let s = slot_rows[b] as usize;
            let (y, kept) = walk(
                &bank64[s * slab..][..slab],
                &xr[b * C..e * C],
                &wr,
                dil,
                e - b,
            );
            want_y[b * C..e * C].copy_from_slice(&y);
            want_s[s * slab..][..slab].copy_from_slice(&kept);
        }
        let x = RaggedTensor {
            data: t(xb, rows, C, Dtype::Bf16),
            indptr: t(ib, 4, 1, Dtype::I32),
        };
        both(
            &mut gpu,
            &format!("conv1d chunked, dilation {dil}"),
            &fresh(rows),
            &[want_y, want_s],
            &|g, o| {
                g.run(|c| {
                    ssm::causal_conv1d_chunked(
                        c,
                        x,
                        wt,
                        &pool(o[1].0, sb, rows),
                        W as u32,
                        dil as u32,
                        t(o[0].0, rows, C, Dtype::Bf16),
                    )
                })
            },
            &|g, o| {
                g.run_slang(|c| {
                    kernels_slang::ssm::conv1d::conv1d_chunked(
                        c,
                        kernels_slang::RaggedTensor {
                            data: st(x.data),
                            indptr: st(x.indptr),
                        },
                        st(wt),
                        &sp(&pool(o[1].0, sb, rows)),
                        W as u32,
                        dil as u32,
                        st(t(o[0].0, rows, C, Dtype::Bf16)),
                    )
                })
            },
        );

        // The committed window, lanes 1..4: 2 own rows after 1 replayed, kept
        // as of row 2 on slot 2; 1 after 2, kept whole on slot 0; 2 after 0
        // with no slot.
        let lane0 = 1usize;
        let indptr = [0i32, 2, 3, 5];
        let replay = [7i32, 1, 2, 0];
        let commit = [7i32, 2, 3, 2];
        let slots = [7i32, 2, 0, -1];
        let rows = 2 + 1 + 1 + 2 + 2;
        let (xb, xr) = gpu.up(&noise(30 + dil as u64, rows * C));
        let xr: Vec<f64> = xr.iter().map(|v| f64::from(*v)).collect();
        let ib = gpu.up_i32(&indptr);
        let (rb, cb, sb) = (gpu.up_i32(&replay), gpu.up_i32(&commit), gpu.up_i32(&slots));
        let mut want_y = vec![0.0; rows * C];
        let mut want_s = bank64.clone();
        let mut begin = indptr[0] as usize;
        for r in 0..3 {
            let lane = lane0 + r;
            let span = (indptr[r + 1] - indptr[r] + replay[lane]) as usize;
            if slots[lane] >= 0 {
                let s = slots[lane] as usize;
                let keep = (commit[lane] as usize).min(span);
                let (y, kept) = walk(
                    &bank64[s * slab..][..slab],
                    &xr[begin * C..(begin + span) * C],
                    &wr,
                    dil,
                    keep,
                );
                want_y[begin * C..(begin + span) * C].copy_from_slice(&y);
                want_s[s * slab..][..slab].copy_from_slice(&kept);
            }
            begin += span;
        }
        let i32s = |buf: u32, n: usize| t(buf, n, 1, Dtype::I32);
        let committed = ssm::Committed {
            replay: i32s(rb, 4),
            commit: i32s(cb, 4),
            slots: i32s(sb, 4),
            lane0: lane0 as u32,
        };
        let slang_committed = kernels_slang::ssm::Committed {
            replay: st(committed.replay),
            commit: st(committed.commit),
            slots: st(committed.slots),
            lane0: committed.lane0,
        };
        let x = t(xb, rows, C, Dtype::Bf16);
        let window = i32s(ib, 4);
        both(
            &mut gpu,
            &format!("conv1d committed, dilation {dil}"),
            &fresh(rows),
            &[want_y, want_s],
            &|g, o| {
                g.run(|c| {
                    ssm::causal_conv1d_committed(
                        c,
                        x,
                        window,
                        &committed,
                        wt,
                        &pool(o[1].0, sb, 4),
                        W as u32,
                        dil as u32,
                        t(o[0].0, rows, C, Dtype::Bf16),
                    )
                })
            },
            &|g, o| {
                g.run_slang(|c| {
                    kernels_slang::ssm::conv1d::conv1d_committed(
                        c,
                        st(x),
                        st(window),
                        &slang_committed,
                        st(wt),
                        &sp(&pool(o[1].0, sb, 4)),
                        W as u32,
                        dil as u32,
                        st(t(o[0].0, rows, C, Dtype::Bf16)),
                    )
                })
            },
        );
    }
}
