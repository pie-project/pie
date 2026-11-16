#![cfg(target_vendor = "apple")]

//! The device's rates as pie's Metal kernels reach them, for a profile's
//! `[rates]`: a 4096³ bf16 GEMM through the dense kernel serving dispatches
//! (`linear::gemm::matmul`, under the device's own tuning), a 256 MiB blit
//! between two buffers (read and write counted), and a dispatch's launch
//! cost (512 of the smallest GEMM in one frame, apart). Each the median of 5
//! timed by the GPU after a warm one. Ignored: run by hand on the device
//! whose file it fills.
//!
//!     cargo test --release -p engine-metal --test engine_metal \
//!         the_device_rates -- --ignored --nocapture

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::linear::gemm;
use poem_ir::Dtype;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "a measurement for a device file, run by hand"]
fn the_device_rates() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let tuning = engine_metal::chip::tuning("", false).with_family(device.family());
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let zeroed = |bytes: u64| Buffer::zeroed(&device, bytes).expect("a buffer");
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let timed = |encode: &dyn Fn(&Sink<'_>)| -> Vec<f64> {
        (0..6)
            .map(|_| {
                let frame = device.frame().expect("a frame");
                encode(&Sink::new(&device, &frame, &pipelines, &handles));
                frame.commit_timed().expect("it lands")
            })
            .skip(1)
            .collect()
    };

    let n = 4096u32;
    let (a, w, y) = (
        zeroed(u64::from(n) * u64::from(n) * 2),
        zeroed(u64::from(n) * u64::from(n) * 2),
        zeroed(u64::from(n) * u64::from(n) * 2),
    );
    let (at, wt, yt) = (
        Tensor::new(bind(&a), n, n, Dtype::Bf16),
        Tensor::new(bind(&w), n, n, Dtype::Bf16),
        Tensor::new(bind(&y), n, n, Dtype::Bf16),
    );
    let gemm_s = median(timed(&|sink| {
        gemm::matmul(sink, &tuning, at, wt, yt).expect("the gemm")
    }));
    let flops_per_ns = 2.0 * f64::from(n).powi(3) / (gemm_s * 1e9);

    let bytes = 256u64 << 20;
    let (from, into) = (zeroed(bytes), zeroed(bytes));
    let copy_s = median(
        (0..6)
            .map(|_| {
                let mut frame = device.frame().expect("a frame");
                frame
                    .copy_span(&from, 0, &into, 0, bytes)
                    .expect("the copy");
                frame.commit_timed().expect("it lands")
            })
            .skip(1)
            .collect(),
    );
    let bytes_per_ns = 2.0 * bytes as f64 / (copy_s * 1e9);

    let (sa, sw, sy) = (zeroed(8 * 32 * 2), zeroed(32 * 32 * 2), zeroed(8 * 32 * 2));
    let (st, swt, syt) = (
        Tensor::new(bind(&sa), 8, 32, Dtype::Bf16),
        Tensor::new(bind(&sw), 32, 32, Dtype::Bf16),
        Tensor::new(bind(&sy), 8, 32, Dtype::Bf16),
    );
    const LAUNCHES: u32 = 512;
    let launch_s = median(timed(&|sink| {
        for _ in 0..LAUNCHES {
            gemm::matmul(sink, &tuning, st, swt, syt).expect("a small gemm");
        }
    }));
    let launch_ns = (launch_s * 1e9 / f64::from(LAUNCHES)).round() as u64;

    println!("device: {}", device.name());
    println!(
        "4096^3 bf16 gemm: {:.2} ms, {:.2} TFLOPS",
        gemm_s * 1e3,
        flops_per_ns / 1e3
    );
    println!(
        "256 MiB blit: {:.2} ms, {:.1} GB/s read+write",
        copy_s * 1e3,
        bytes_per_ns
    );
    println!("launch: {launch_ns} ns a dispatch ({LAUNCHES} small gemms in one frame)");
    println!(
        "[rates]\nbf16_flops_per_ns_by_rows = [[4096, {flops_per_ns:.1}]]\nbytes_per_ns = {bytes_per_ns:.1}\nlaunch_ns = {launch_ns}"
    );
}
