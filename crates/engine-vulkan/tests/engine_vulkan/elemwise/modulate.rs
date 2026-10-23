//! The modulations (`elemwise.modulate` in its three forms,
//! `gated_residual_add`) on the first Vulkan device against
//! an f64 host reference, beside the hand-written kernels, bf16 and f32, with and without a lane map.

use crate::gpu::{Gpu, close, noise};
use kernels_vulkan::elemwise::modulate;
use poem_ir::Dtype;

fn up(gpu: &mut Gpu, v: &[f32], dtype: Dtype) -> u32 {
    if dtype == Dtype::F32 {
        gpu.up_f32(v)
    } else {
        gpu.up(v).0
    }
}

fn down(gpu: &Gpu, buf: u32, n: usize, dtype: Dtype) -> Vec<f32> {
    if dtype == Dtype::F32 {
        gpu.down_f32(buf, n)
    } else {
        gpu.down(buf, n)
    }
}

/// `v` as a `dtype` buffer holds it.
fn held(v: &[f32], dtype: Dtype) -> Vec<f64> {
    v.iter()
        .map(|x| {
            if dtype == Dtype::F32 {
                f64::from(*x)
            } else {
                f64::from(f32::from_bits(
                    ((x.to_bits() + 0x7fff + ((x.to_bits() >> 16) & 1)) >> 16) << 16,
                ))
            }
        })
        .collect()
}

#[test]
fn the_modulations_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let st =
        |t: kernels_vulkan::Tensor| kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype);
    let t = |buf: u32, rows: usize, width: usize, dtype: Dtype| {
        kernels_vulkan::Tensor::new(buf, rows as u32, width as u32, dtype)
    };
    let (rows, width) = (4usize, 96usize);
    let n = rows * width;
    let lanes = [2i32, 0, 1, 2];
    let lane_buf = gpu.up_i32(&lanes);
    for (act, md) in [
        (Dtype::Bf16, Dtype::Bf16),
        (Dtype::Bf16, Dtype::F32),
        (Dtype::F32, Dtype::F32),
    ] {
        let xv = noise(11, n);
        let x = up(&mut gpu, &xv, act);
        let xr = held(&xv, act);
        for (form, vectors) in [(0usize, 2usize), (1, 1), (2, 1)] {
            for mapped in [false, true] {
                let m_rows = if mapped { 3 } else { rows };
                let mv = noise(20 + form as u64, m_rows * vectors * width);
                let mb = up(&mut gpu, &mv, md);
                let mr = held(&mv, md);
                let mt = t(mb, m_rows, vectors * width, md);
                let lt = mapped.then(|| t(lane_buf, rows, 1, Dtype::I32));
                let want: Vec<f64> = (0..n)
                    .map(|i| {
                        let (r, c) = (i / width, i % width);
                        let lane = if mapped { lanes[r] as usize } else { r };
                        let at = lane * vectors * width + c;
                        match form {
                            0 => xr[i] * (1.0 + mr[at]) + mr[at + width],
                            1 => xr[i] * (1.0 + mr[at]),
                            _ => mr[at].tanh() * xr[i],
                        }
                    })
                    .collect();
                let shape =
                    format!("form {form}, {act:?} rows, {md:?} modulation, mapped {mapped}");
                let sf = [
                    kernels_slang::elemwise::modulate::Form::ScaleShift,
                    kernels_slang::elemwise::modulate::Form::Scale,
                    kernels_slang::elemwise::modulate::Form::TanhGate,
                ][form];
                let hand = Some(
                    [
                        modulate::Form::ScaleShift,
                        modulate::Form::Scale,
                        modulate::Form::TanhGate,
                    ][form],
                );

                if let Some(hf) = hand {
                    let y = gpu.zeros(2 * n);
                    gpu.run(|c| {
                        modulate::modulate(
                            c,
                            hf,
                            t(x, rows, width, act),
                            mt,
                            lt,
                            t(y, rows, width, act),
                        )
                    });
                    close(&down(&gpu, y, n, act), &want, &format!("modulate {shape}"));
                }

                let y = gpu.zeros(2 * n);
                gpu.run_slang(|c| {
                    kernels_slang::elemwise::modulate::modulate(
                        c,
                        sf,
                        st(t(x, rows, width, act)),
                        st(mt),
                        lt.map(st),
                        st(t(y, rows, width, act)),
                    )
                });
                close(
                    &down(&gpu, y, n, act),
                    &want,
                    &format!("slang modulate {shape}"),
                );
            }
        }
        for mapped in [false, true] {
            let m_rows = if mapped { 3 } else { rows };
            let rv = noise(31, n);
            let yv = noise(32, n);
            let gv = noise(33, m_rows * width);
            let (rr, yr) = (held(&rv, act), held(&yv, act));
            let gb = up(&mut gpu, &gv, md);
            let gr = held(&gv, md);
            let yb = up(&mut gpu, &yv, act);
            let gt = t(gb, m_rows, width, md);
            let lt = mapped.then(|| t(lane_buf, rows, 1, Dtype::I32));
            let want: Vec<f64> = (0..n)
                .map(|i| {
                    let (r, c) = (i / width, i % width);
                    let lane = if mapped { lanes[r] as usize } else { r };
                    gr[lane * width + c] * yr[i] + rr[i]
                })
                .collect();
            let shape = format!("{act:?} rows, {md:?} gate, mapped {mapped}");

            let r = up(&mut gpu, &rv, act);
            gpu.run(|c| {
                modulate::gated_residual_add(
                    c,
                    t(r, rows, width, act),
                    gt,
                    t(yb, rows, width, act),
                    lt,
                    t(r, rows, width, act),
                )
            });
            close(
                &down(&gpu, r, n, act),
                &want,
                &format!("gated residual add {shape}"),
            );

            let r = up(&mut gpu, &rv, act);
            gpu.run_slang(|c| {
                kernels_slang::elemwise::modulate::gated_residual_add(
                    c,
                    st(t(r, rows, width, act)),
                    st(gt),
                    st(t(yb, rows, width, act)),
                    lt.map(st),
                )
            });
            close(
                &down(&gpu, r, n, act),
                &want,
                &format!("slang gated residual add {shape}"),
            );
        }
    }
}
