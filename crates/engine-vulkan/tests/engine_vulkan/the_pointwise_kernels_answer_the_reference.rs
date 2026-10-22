//! The pointwise kernels (`elemwise.add`, `silu`, `tanh`, `cast`) on the
//! first Vulkan device against an f64 host reference, beside the hand-written
//! kernels, over even and odd counts.

use crate::gpu::{Gpu, close, noise};
use kernels_vulkan::Tensor;
use kernels_vulkan::elemwise::pointwise;
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
    let st = |t: Tensor| kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype);
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
        gpu.run(|k| pointwise::silu(k, bt(x), bt(x)));
        close(&gpu.down(x, n + 1), &silued, &format!("silu bf16 {shape}"));
        let (x, _) = gpu.up(&xv);
        gpu.run(|k| pointwise::tanh(k, bt(x), bt(x)));
        close(&gpu.down(x, n + 1), &tanhed, &format!("tanh bf16 {shape}"));
        let z = gpu.zeros(n + 1);
        gpu.run(|k| pointwise::add(k, bt(x0), bt(y0), bt(z)));
        close(&gpu.down(z, n), &summed[..n], &format!("add bf16 {shape}"));

        let (x, _) = gpu.up(&xv);
        gpu.run_slang(|k| kernels_slang::elemwise::silu_scaled::silu(k, st(bt(x))));
        close(
            &gpu.down(x, n + 1),
            &silued,
            &format!("slang silu bf16 {shape}"),
        );
        let (x, _) = gpu.up(&xv);
        gpu.run_slang(|k| kernels_slang::elemwise::tanh::tanh(k, st(bt(x))));
        close(
            &gpu.down(x, n + 1),
            &tanhed,
            &format!("slang tanh bf16 {shape}"),
        );
        let z = gpu.zeros(n + 1);
        gpu.run_slang(|k| kernels_slang::elemwise::add::add(k, st(bt(x0)), st(bt(y0)), st(bt(z))));
        close(
            &gpu.down(z, n),
            &summed[..n],
            &format!("slang add bf16 {shape}"),
        );

        let x = gpu.up_f32(&xv);
        gpu.run_slang(|k| kernels_slang::elemwise::tanh::tanh(k, st(ft(x))));
        let want = keep(xv.iter().map(|v| f64::from(*v).tanh()).collect(), 7.0);
        close(
            &gpu.down_f32(x, n + 1),
            &want,
            &format!("slang tanh f32 {shape}"),
        );
        let x = gpu.up_f32(&xv);
        gpu.run_slang(|k| kernels_slang::elemwise::silu_scaled::silu(k, st(ft(x))));
        let want = keep(xv.iter().map(|v| silu(f64::from(*v))).collect(), 7.0);
        close(
            &gpu.down_f32(x, n + 1),
            &want,
            &format!("slang silu f32 {shape}"),
        );
        let (x, y) = (gpu.up_f32(&xv), gpu.up_f32(&yv));
        let z = gpu.up_f32(&vec![0.0; n + 1]);
        gpu.run_slang(|k| kernels_slang::elemwise::add::add(k, st(ft(x)), st(ft(y)), st(ft(z))));
        let want: Vec<f64> = xv
            .iter()
            .zip(&yv)
            .map(|(a, b)| f64::from(*a) + f64::from(*b))
            .collect();
        close(
            &gpu.down_f32(z, n),
            &want[..n],
            &format!("slang add f32 {shape}"),
        );

        // The cast rounds each f32 to the nearest bf16, ties to even.
        let x = gpu.up_f32(&xv);
        let (o, _) = gpu.up(&vec![7.0; n + 1]);
        gpu.run_slang(|k| kernels_slang::elemwise::cast::cast_f32_to_bf16(k, st(ft(x)), st(bt(o))));
        let want = keep(xr.iter().map(|v| f64::from(*v)).collect(), 7.0);
        close(
            &gpu.down(o, n + 1),
            &want,
            &format!("slang cast f32 to bf16 {shape}"),
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
