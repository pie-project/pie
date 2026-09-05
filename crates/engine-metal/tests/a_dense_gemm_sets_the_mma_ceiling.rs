//! **WHAT THIS GPU'S MATRIX UNITS DELIVER ON A PLAIN bf16 GEMM** — the
//! reference the quantized tile's TFLOP/s is read against.
//!
//! `a_quantized_matmul_is_priced_by_its_rows` shows the 4-bit tile at
//! 5.7 TFLOP/s from sixteen rows up and flat to 256, and an 8-bit bank of the
//! same shape (twice the bytes, the same FLOPs) costing the same at 64 rows —
//! which reads as a COMPUTE ceiling, not a loader defect. This times
//! `gemm_dense` (no dequantization at all) on the same shape so the ceiling
//! has a number of its own: if the dense tile lands near the quantized one,
//! the quantized tile is at the machine's MMA rate and the remaining lever on
//! it is the arithmetic itself, not its loads.
//!
//! ```text
//! [PIE_GEMM_ROWS=16,64,256] [PIE_GEMM_STEPS=20] \
//!   cargo test -p engine-metal --release --test a_dense_gemm_sets_the_mma_ceiling -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::linear::gemm;
use model_ir::Dtype;

const K: u32 = 5120;
const N: u32 = 17408;

fn env<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(fallback)
}

#[test]
fn the_dense_tile_is_timed() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    eprintln!("device: {}", device.name());
    let rows: Vec<u32> = std::env::var("PIE_GEMM_ROWS")
        .unwrap_or_else(|_| "16,64,256".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let steps: usize = env("PIE_GEMM_STEPS", 20usize);
    let batch: usize = 8;
    let widest = *rows.iter().max().expect("a row count");

    // Small deterministic values so the products stay in bf16's range.
    let mut w_b = Buffer::zeroed(&device, u64::from(N) * u64::from(K) * 2).expect("w");
    {
        let bytes: Vec<u8> = (0..u64::from(N) * u64::from(K))
            .map(|at| {
                let v = 0.01 * ((at % 17) as f32 - 8.0);
                ((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) >> 16) as u16
            })
            .flat_map(u16::to_le_bytes)
            .collect();
        w_b.write(0, &bytes).expect("write w");
    }
    let mut act_b = Buffer::zeroed(&device, u64::from(widest) * u64::from(K) * 2).expect("act");
    {
        let bytes: Vec<u8> = (0..u64::from(widest) * u64::from(K))
            .map(|at| {
                let v = 0.02 * ((at % 13) as f32 - 6.0);
                ((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) >> 16) as u16
            })
            .flat_map(u16::to_le_bytes)
            .collect();
        act_b.write(0, &bytes).expect("write act");
    }
    let y_b = Buffer::zeroed(&device, u64::from(widest) * u64::from(N) * 2).expect("y");
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let (hw, ha, hy) = (bind(&w_b), bind(&act_b), bind(&y_b));
    let w = Tensor::new(hw, N, K, Dtype::Bf16);

    for &m in &rows {
        let act = Tensor::new(ha, m, K, Dtype::Bf16);
        let y = Tensor::new(hy, m, N, Dtype::Bf16);
        // Warm, as the quantized bench does: the governor needs ~300 ms of work.
        let began = std::time::Instant::now();
        while began.elapsed().as_millis() < 300 {
            let frame = device.frame().expect("a frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            for _ in 0..batch {
                gemm::matmul(&sink, act, w, y).expect("the launch");
            }
            frame.commit().expect("the warm commit");
        }
        let mut device_s = 0.0f64;
        for _ in 0..steps {
            let frame = device.frame().expect("a frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            for _ in 0..batch {
                gemm::matmul(&sink, act, w, y).expect("the launch");
            }
            device_s += frame.commit_timed().expect("the commit");
        }
        let us = device_s * 1e6 / (steps * batch) as f64;
        let gflop = 2.0 * f64::from(K) * f64::from(N) * f64::from(m) / 1e9;
        let bytes_gb = f64::from(N) * f64::from(K) * 2.0 / 1e9;
        eprintln!(
            "    rows {m:>4}: {us:>9.1} us  {:.2} TFLOP/s  ({:.0} GB/s of bf16 weight)",
            gflop / (us / 1e6) / 1e3,
            bytes_gb / (us / 1e6)
        );
    }
}
