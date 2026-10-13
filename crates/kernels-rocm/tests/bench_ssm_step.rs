//! The recurrent decode steps at the minis' shapes: KDA (GLM-5.3-Flash, 64
//! heads of 128, an f32 state) and gated delta (Qwen3.8-Flash-Next, 16 key
//! heads and 48 value heads of 128, a bf16 state). Each call walks its own
//! slot set of a slab past the 256 MiB last-level cache, as a model's layers
//! do, and the bandwidth counts the state read and written once.
//!
//! `flock /tmp/gpu.lock cargo test --release --features rocm --test bench_ssm_step -- --ignored --nocapture`

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::attn::ssm;
use kernels_rocm::hip;
use kernels_rocm::{RecurrentPool, Tensor};

const D: u32 = 128;
const ROWS: [u32; 4] = [1, 32, 128, 512];

fn pool(slab: Tensor, slot_ids: Tensor, stride: usize) -> RecurrentPool {
    RecurrentPool {
        slab,
        slot_ids,
        slot_stride_elems: stride as i64,
        conv_slab: Tensor::ABSENT,
        conv_stride: 0,
        write_state: true,
        write_state_mask: Tensor::ABSENT,
        commit_len: Tensor::ABSENT,
        begin_at: Tensor::ABSENT,
        fused_decay: false,
    }
}

/// Slot sets enough that consecutive calls miss the cache, and the slab.
fn slots(
    gpu: &Gpu,
    rows: u32,
    head_bytes: usize,
    heads: usize,
    dtype: Dtype,
) -> (Tensor, Vec<Tensor>, usize) {
    let footprint = rows as usize * heads * head_bytes;
    let sets = (1usize << 30).div_ceil(footprint).max(2);
    let stride = heads * (D * D) as usize;
    let slab = gpu.zeros(sets as u32 * rows, stride as u32, dtype);
    let ids = (0..sets)
        .map(|s| {
            let v: Vec<i32> = (0..rows as i32)
                .map(|r| s as i32 * rows as i32 + r)
                .collect();
            gpu.column(&v)
        })
        .collect();
    (slab, ids, stride)
}

/// The median of five timings (each a graph of many calls).
fn median(mut f: impl FnMut() -> f64) -> f64 {
    let mut t: Vec<f64> = (0..5).map(|_| f()).collect();
    t.sort_by(f64::total_cmp);
    t[2]
}

fn report(what: &str, rows: u32, us: f64, bytes: f64) {
    eprintln!(
        "{what} rows {rows:4}: {us:8.1} us  {:5.2} TB/s",
        bytes / (us * 1e-6) / 1e12
    );
}

#[test]
#[ignore = "benchmark"]
fn bench_ssm_step() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();

    // The ceiling: a device-to-device copy of 512 MiB, read and write counted.
    let bytes = 512usize << 20;
    let (src, dst) = (
        hip::Buffer::alloc(bytes).unwrap(),
        hip::Buffer::alloc(bytes).unwrap(),
    );
    let us = gpu.time(20, || unsafe {
        hip::memcpy_async(
            dst.device_ptr() as *mut _,
            src.device_ptr() as *const _,
            bytes,
            hip::Direction::DeviceToDevice,
            gpu.stream.raw(),
        )
        .unwrap();
    });
    report("copy 512 MiB      ", 1, us, 2.0 * bytes as f64);

    let heads = 64u32;
    let w = heads * D;
    for rows in ROWS {
        let bf = |v: Vec<f32>| v.into_iter().map(round_bf16).collect::<Vec<_>>();
        let mixed = gpu.tensor(
            &bf(values((rows * 3 * w) as usize, 1, 1.0)),
            rows,
            3 * w,
            Dtype::Bf16,
        );
        let f = gpu.tensor(
            &bf(values((rows * w) as usize, 2, 2.0)),
            rows,
            w,
            Dtype::Bf16,
        );
        let b = gpu.tensor(
            &bf(values((rows * heads) as usize, 3, 3.0)),
            rows,
            heads,
            Dtype::Bf16,
        );
        let dt = gpu.tensor(&values(w as usize, 4, 1.0), 1, w, Dtype::F32);
        let a_log = gpu.tensor(&values(heads as usize, 5, 0.3), 1, heads, Dtype::F32);
        let mut y = gpu.zeros(rows, w, Dtype::F32);
        let (slab, ids, stride) =
            slots(&gpu, rows, (D * D * 4) as usize, heads as usize, Dtype::F32);
        let mut i = 0usize;
        let reps = (ids.len() as u32).max(64);
        let us = median(|| {
            gpu.time_graph(reps, || {
                let st = pool(slab, ids[i % ids.len()], stride);
                i += 1;
                ssm::kda_step(
                    &ctx, mixed, f, b, dt, a_log, &st, heads, D, 1e-6, -5.0, &mut y,
                )
                .unwrap();
            })
        });
        report(
            "kda  64x128 f32 ",
            rows,
            us,
            2.0 * f64::from(rows * heads) * f64::from(D * D * 4),
        );
    }

    let (kh, vh) = (16u32, 48u32);
    let c = 2 * kh * D + vh * D;
    for rows in ROWS {
        let qkv = gpu.tensor(
            &values((rows * c) as usize, 6, 1.0)
                .into_iter()
                .map(round_bf16)
                .collect::<Vec<_>>(),
            rows,
            c,
            Dtype::Bf16,
        );
        let gates: Vec<f32> = values((rows * 2 * vh) as usize, 7, 1.0)
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                if i % (2 * vh as usize) < vh as usize {
                    -0.05 - 0.3 * v.abs()
                } else {
                    0.5 + 0.45 * v
                }
            })
            .collect();
        let gates = gpu.tensor(&gates, rows, 2 * vh, Dtype::F32);
        let mut y = gpu.zeros(rows, vh * D, Dtype::F32);
        let (slab, ids, stride) = slots(&gpu, rows, (D * D * 2) as usize, vh as usize, Dtype::Bf16);
        let mut i = 0usize;
        let reps = (ids.len() as u32).max(64);
        let us = median(|| {
            gpu.time_graph(reps, || {
                let st = pool(slab, ids[i % ids.len()], stride);
                i += 1;
                ssm::gated_delta(&ctx, qkv, Tensor::ABSENT, gates, &st, kh, vh, D, D, &mut y)
                    .unwrap();
            })
        });
        report(
            "gdn 48x128 bf16 ",
            rows,
            us,
            2.0 * f64::from(rows * vh) * f64::from(D * D * 2),
        );
    }
}
