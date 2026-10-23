//! The pointwise kernels (`elemwise.add`, `silu`, `tanh`, `cast`) on the
//! first wgpu adapter against an f64 host reference, over even and odd
//! counts.

use crate::gpu::{Gpu, bf16, close, noise, widen};
use kernels_slang::Tensor;
use poem_ir::Dtype;

fn silu(v: f64) -> f64 {
    v / (1.0 + (-v).exp())
}

#[test]
fn the_pointwise_kernels_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    for (rows, width) in [(3usize, 64usize), (5, 1000), (3, 7)] {
        let n = rows * width;
        let bt = |buf: u32| Tensor::new(buf, rows as u32, width as u32, Dtype::Bf16);
        let ft = |buf: u32| Tensor::new(buf, rows as u32, width as u32, Dtype::F32);
        let shape = format!("{rows} x {width}");
        // Wide enough that tanh's exponential overflows; one sentinel past
        // the tensor, which the in-place kernels leave alone.
        let mut xv: Vec<f32> = noise(width as u64 + 30, n)
            .iter()
            .map(|v| v * 30.0)
            .collect();
        xv[0] = 60.0;
        xv[1] = -60.0;
        xv.push(7.0);
        let yv = noise(width as u64 + 31, n + 1);
        let (x0, xr) = gpu.up(&xv);
        let (y0, yr) = gpu.up(&yv);
        let keep = |mut v: Vec<f64>, last: f64| {
            v[n] = last;
            v
        };
        let silued = keep(xr.iter().map(|v| silu(f64::from(*v))).collect(), 7.0);
        let tanhed = keep(xr.iter().map(|v| f64::from(*v).tanh()).collect(), 7.0);
        let summed: Vec<f64> = xr
            .iter()
            .zip(&yr)
            .map(|(a, b)| f64::from(*a) + f64::from(*b))
            .collect();

        let (x, _) = gpu.up(&xv);
        gpu.run(|k| kernels_slang::elemwise::pointwise::silu(k, bt(x)));
        close(&gpu.down(x, n + 1), &silued, &format!("silu bf16 {shape}"));
        let (x, _) = gpu.up(&xv);
        gpu.run(|k| kernels_slang::elemwise::pointwise::tanh(k, bt(x)));
        close(&gpu.down(x, n + 1), &tanhed, &format!("tanh bf16 {shape}"));
        let z = gpu.zeros(n + 1);
        gpu.run(|k| kernels_slang::elemwise::pointwise::add(k, bt(x0), bt(y0), bt(z)));
        close(&gpu.down(z, n), &summed[..n], &format!("add bf16 {shape}"));

        let x = gpu.up_f32(&xv);
        gpu.run(|k| kernels_slang::elemwise::pointwise::tanh(k, ft(x)));
        let want = keep(xv.iter().map(|v| f64::from(*v).tanh()).collect(), 7.0);
        close(&gpu.down_f32(x, n + 1), &want, &format!("tanh f32 {shape}"));
        let x = gpu.up_f32(&xv);
        gpu.run(|k| kernels_slang::elemwise::pointwise::silu(k, ft(x)));
        let want = keep(xv.iter().map(|v| silu(f64::from(*v))).collect(), 7.0);
        close(&gpu.down_f32(x, n + 1), &want, &format!("silu f32 {shape}"));
        let (x, y) = (gpu.up_f32(&xv), gpu.up_f32(&yv));
        let z = gpu.up_f32(&vec![0.0; n + 1]);
        gpu.run(|k| kernels_slang::elemwise::pointwise::add(k, ft(x), ft(y), ft(z)));
        let want: Vec<f64> = xv
            .iter()
            .zip(&yv)
            .map(|(a, b)| f64::from(*a) + f64::from(*b))
            .collect();
        close(&gpu.down_f32(z, n), &want[..n], &format!("add f32 {shape}"));

        // The cast rounds each f32 to the nearest bf16, ties to even.
        let x = gpu.up_f32(&xv);
        let (o, _) = gpu.up(&vec![7.0; n + 1]);
        gpu.run(|k| kernels_slang::elemwise::pointwise::cast_f32_to_bf16(k, ft(x), bt(o)));
        let want = keep(xr.iter().map(|v| f64::from(*v)).collect(), 7.0);
        close(
            &gpu.down(o, n + 1),
            &want,
            &format!("cast f32 to bf16 {shape}"),
        );
        assert!(
            gpu.down(o, n)
                .iter()
                .zip(&xr)
                .all(|(g, w)| g.to_bits() == w.to_bits()),
            "the cast lands on the host's bf16 rounding"
        );
    }
}

/// The in-place bounds and gate: `clamp` (stated and learned bounds) and
/// `x *= sigmoid(gate)`.
#[test]
fn the_clamps_and_the_gate_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let (lo, hi) = (-0.7f32, 1.3f32);
    let round = |v: f32| f64::from(widen(bf16(v)));
    for (rows, width) in [(3usize, 64usize), (5, 1000), (3, 7)] {
        let n = rows * width;
        let bt = |buf: u32| Tensor::new(buf, rows as u32, width as u32, Dtype::Bf16);
        let one = |buf: u32| Tensor::new(buf, 1, 1, Dtype::Bf16);
        let shape = format!("{rows} x {width}");
        let mut xv = noise(width as u64 + 40, n);
        xv.push(7.0);
        let mut gv: Vec<f32> = noise(width as u64 + 41, n)
            .iter()
            .map(|v| v * 20.0)
            .collect();
        gv.push(0.0);
        let (_, xr) = gpu.up(&xv);
        let (g, gr) = gpu.up(&gv);
        let (lo_t, _) = gpu.up(&[-0.4]);
        let (hi_t, _) = gpu.up(&[0.9]);
        let keep = |mut v: Vec<f64>| {
            v[n] = 7.0;
            v
        };
        let clamped = keep(
            xr.iter()
                .map(|v| f64::from(*v).clamp(round(lo), round(hi)))
                .collect(),
        );
        let learned = keep(
            xr.iter()
                .map(|v| f64::from(*v).clamp(round(-0.4), round(0.9)))
                .collect(),
        );
        let gated = keep(
            xr.iter()
                .zip(&gr)
                .map(|(v, g)| f64::from(*v) / (1.0 + (-f64::from(*g)).exp()))
                .collect(),
        );
        let (x, _) = gpu.up(&xv);
        gpu.run(|k| kernels_slang::elemwise::pointwise::clamp(k, lo, hi, bt(x)));

        close(
            &gpu.down(x, n + 1),
            &clamped,
            &format!("clamp bf16 {shape}"),
        );
        let (x, _) = gpu.up(&xv);
        gpu.run(|k| {
            kernels_slang::elemwise::pointwise::clamp_learned(k, one(lo_t), one(hi_t), bt(x))
        });

        close(
            &gpu.down(x, n + 1),
            &learned,
            &format!("clamp learned bf16 {shape}"),
        );
        let (x, _) = gpu.up(&xv);
        gpu.run(|k| kernels_slang::elemwise::pointwise::sigmoid_mul(k, bt(g), bt(x)));

        close(
            &gpu.down(x, n + 1),
            &gated,
            &format!("sigmoid mul bf16 {shape}"),
        );
    }
}
