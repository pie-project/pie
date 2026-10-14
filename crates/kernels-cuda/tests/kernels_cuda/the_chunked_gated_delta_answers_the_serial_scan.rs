#![cfg(feature = "cuda")]

use crate::common;

use common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::attn::ssm;
use kernels_cuda::tensor::{RaggedTensor, RecurrentPool, Tensor};
use kernels_cuda::tuning::Tuning;

/// One chunked fire: its requests, its heads, and the per-lane seats a
/// speculative or staged fire carries. Per-lane arrays are indexed by the
/// lane's seat, `lane + lane_base`.
struct Case {
    lens: &'static [usize],
    k_heads: u32,
    v_heads: u32,
    commit_len: Option<&'static [i32]>,
    begin_at: Option<&'static [i32]>,
    folds: Option<&'static [u8]>,
    /// A staged window: the fire reads and lands its rows this many rows
    /// into the planes, and seats its lanes this many seats in.
    window: Option<(u32, u32)>,
}

const PLAIN: Case = Case {
    lens: &[70, 5, 130],
    k_heads: 2,
    v_heads: 4,
    commit_len: None,
    begin_at: None,
    folds: None,
    window: None,
};

#[test]
fn the_chunked_gated_delta_answers_the_serial_scan() {
    check(&PLAIN);
}

/// A request several chunks long carries its state across them, at a
/// three-way head fan.
#[test]
fn the_chunked_gated_delta_carries_the_state_across_chunks() {
    check(&Case {
        lens: &[300, 64, 129],
        k_heads: 2,
        v_heads: 6,
        ..PLAIN
    });
}

/// A replay that starts inside a row, commits a prefix of it, and folds only
/// the lanes its predicate names answers the serial scan on the same seats.
#[test]
fn the_chunked_gated_delta_honours_the_begin_the_commit_and_the_fold_predicate() {
    check(&Case {
        commit_len: Some(&[50, 9, 100]),
        begin_at: Some(&[3, 0, 64]),
        folds: Some(&[1, 0, 1]),
        ..PLAIN
    });
}

/// A staged window reads and lands its rows past the window's base and
/// seats its lanes past the window's first seat.
#[test]
fn the_chunked_gated_delta_reads_and_lands_where_the_window_says() {
    check(&Case {
        commit_len: Some(&[0, 70, 4, 120]),
        window: Some((7, 1)),
        ..PLAIN
    });
}

/// Each case runs both prefill paths: the three wmma launches every card
/// runs, and the two-launch scan the sm_100 profile turns on (a card whose
/// blocks cannot hold it falls back to the three).
fn check(case: &Case) {
    for fused in [false, true] {
        check_on(case, fused);
    }
}

fn tuned(fused: bool) -> Tuning {
    Tuning {
        ssm_gdn_fused_scan: fused,
        ..Tuning::DEFAULT
    }
}

#[allow(clippy::too_many_lines)]
fn check_on(case: &Case, fused: bool) {
    let lens = case.lens;
    let (k_heads, v_heads, k_dim, v_dim) = (case.k_heads, case.v_heads, 128u32, 128u32);
    let conv_dim = (2 * k_heads * k_dim + v_heads * v_dim) as usize;
    let (base, lane_base) = case.window.unwrap_or((0, 0));
    let (base, lane_base) = (base as usize, lane_base as usize);
    let live: usize = lens.iter().sum();
    let rows = base + live;
    let seats = lane_base + lens.len();
    let slots = 3usize;
    let slot_of: Vec<i32> = (0..seats).map(|s| ((s + 2) % slots) as i32).collect();
    let stride = (v_heads * k_dim * v_dim) as usize;
    let mut indptr: Vec<i32> = vec![0];
    for &len in lens {
        indptr.push(indptr.last().unwrap() + len as i32);
    }
    let mut lcg = Lcg::seeded(0x9d);
    let (qkv_raw, _) = lcg.row(rows * conv_dim);
    let mut gates: Vec<f32> = Vec::with_capacity(rows * 2 * v_heads as usize);
    for _ in 0..rows {
        for _ in 0..v_heads {
            gates.push(-0.15 * (lcg.unit() * 0.5 + 0.5));
        }
        for _ in 0..v_heads {
            gates.push(0.2 + 0.7 * (lcg.unit() * 0.5 + 0.5));
        }
    }
    let (slab_raw, _) = lcg.row(slots * stride);
    let slab_raw: Vec<u16> = slab_raw
        .iter()
        .map(|&v| common::to_bf16(common::from_bf16(v) * 0.05))
        .collect();
    let mut gpu = Gpu::open();
    let qkv_at = gpu.up(&qkv_raw);
    let gates_at = gpu.up(&gates);
    let indptr_at = gpu.up(&indptr);
    let slots_at = gpu.up(&slot_of);
    let seat = |gpu: &mut Gpu, values: Option<&'static [i32]>| {
        values.map_or(Tensor::ABSENT, |v| {
            Tensor::new(gpu.up(v), v.len() as u32, 1, Dtype::I32)
        })
    };
    let commit_len = seat(&mut gpu, case.commit_len);
    let begin_at = seat(&mut gpu, case.begin_at);
    let mask = case.folds.map_or(Tensor::ABSENT, |v| {
        Tensor::new(gpu.up(v), v.len() as u32, 1, Dtype::U8)
    });
    let win_at = case.window.map(|_| {
        gpu.up(&[
            live as u32,
            base as u32,
            lens.len() as u32,
            lane_base as u32,
        ])
    });
    let mut run = |serial: bool| -> (Vec<f32>, Vec<u16>) {
        let slab_at = gpu.up(&slab_raw);
        let y_at = gpu.zeros(rows * (v_heads * v_dim) as usize * 4);
        let pool = RecurrentPool {
            slab: Tensor::new(slab_at, slots as u32, stride as u32, Dtype::Bf16),
            slot_ids: Tensor::new(slots_at, seats as u32, 1, Dtype::I32),
            slot_stride_elems: stride as i64,
            conv_slab: Tensor::ABSENT,
            conv_stride: 0,
            write_state: true,
            write_state_mask: mask,
            commit_len,
            begin_at,
            fused_decay: false,
        };
        let mut y = Tensor::new(y_at, live as u32, v_heads * v_dim, Dtype::F32);
        let qkv = RaggedTensor {
            data: Tensor::new(qkv_at, live as u32, conv_dim as u32, Dtype::Bf16),
            indptr: Tensor::new(indptr_at, lens.len() as u32 + 1, 1, Dtype::I32),
        };
        let gates_t = Tensor::new(gates_at, live as u32, 2 * v_heads, Dtype::F32);
        let ctx = gpu.ctx().with_tuning(tuned(fused));
        if let Some(win) = win_at {
            ctx.arm_stage(win);
        }
        let fired = if serial {
            ssm::gated_delta_chunked_serial(
                &ctx,
                qkv,
                Tensor::ABSENT,
                gates_t,
                &pool,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                &mut y,
            )
        } else {
            ssm::gated_delta_chunked(
                &ctx,
                qkv,
                Tensor::ABSENT,
                gates_t,
                &pool,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                &mut y,
            )
        };
        fired.expect("the gated delta fires");
        gpu.sync();
        (
            gpu.down(y_at, rows * (v_heads * v_dim) as usize),
            gpu.down(slab_at, slots * stride),
        )
    };
    let (y_serial, state_serial) = run(true);
    let (y_chunked, state_chunked) = run(false);
    let (y_again, state_again) = run(false);
    let unstable = y_chunked
        .iter()
        .zip(&y_again)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let unstable_state = state_chunked
        .iter()
        .zip(&state_again)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        unstable == 0 && unstable_state == 0,
        "two chunked fires (fused scan {fused}) disagree with each other: {unstable} output cells, {unstable_state} state cells"
    );
    let landed = y_serial.iter().filter(|v| **v != 0.0).count();
    assert!(landed > 0, "the serial scan landed no output");
    let mut worst = 0f32;
    let mut worst_at = 0usize;
    for (at, (&a, &b)) in y_serial.iter().zip(&y_chunked).enumerate() {
        let err = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
        if err > worst {
            worst = err;
            worst_at = at;
        }
    }
    let width = (v_heads * v_dim) as usize;
    assert!(
        worst < 3e-2,
        "fused scan {fused}: output row {} col {}: serial {} against chunked {} (relative error {worst})",
        worst_at / width,
        worst_at % width,
        y_serial[worst_at],
        y_chunked[worst_at]
    );
    let mut worst_state = 0f32;
    for (&a, &b) in state_serial.iter().zip(&state_chunked) {
        let (a, b) = (common::from_bf16(a), common::from_bf16(b));
        let err = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
        worst_state = worst_state.max(err);
    }
    assert!(
        worst_state < 3e-2,
        "fused scan {fused}: the final states differ by {worst_state}"
    );
    eprintln!(
        "chunked (fused scan {fused}) vs serial: worst output error {worst}, worst state error {worst_state}"
    );
}

/// vLLM 0.31 prefills 32 x 4096 tokens of Qwen3.8-27B through FlashInfer's
/// CUTLASS chunked kernel in 150.6 ms over 48 layers (nsys), about 3.1 ms a
/// layer against this table's last row.
///
/// The chunked scan's time per layer at Qwen3.8-27B's linear-attention head
/// (sixteen key heads fanned to 48 value heads, 128 wide) over prefills of
/// 4096-token requests. Run it alone on an idle GPU:
///
///     cargo test -p kernels-cuda --features cuda --release --test kernels_cuda \
///         the_chunked_gated_delta_reports -- --ignored --nocapture
#[test]
#[ignore = "a measurement: run alone, on an idle GPU"]
fn the_chunked_gated_delta_reports_its_rate_at_a_qwen38_27b_head() {
    use crate::the_tcgen05_gemm_lands_and_paces_what_cublaslt_does::{FLUSH, Timer};
    use kernels_cuda::cudarc::runtime::sys as rt;

    let (k_heads, v_heads, k_dim, v_dim) = (16u32, 48u32, 128u32, 128u32);
    let conv_dim = (2 * k_heads * k_dim + v_heads * v_dim) as usize;
    let stride = (v_heads * k_dim * v_dim) as usize;
    let len = 4096usize;
    let mut gpu = Gpu::open();
    let stream: rt::cudaStream_t = gpu.ctx().stream().cast();
    let timer = Timer {
        flush: gpu.zeros(FLUSH),
        stream,
    };
    println!("| scan | requests x tokens | ms / layer | tokens/s per layer |");
    println!("|---|---|---:|---:|");
    for (fused, requests) in [false, true]
        .into_iter()
        .flat_map(|fused| [1usize, 8, 32].map(|requests| (fused, requests)))
    {
        let rows = requests * len;
        let mut lcg = Lcg::seeded(0x27);
        let (qkv_raw, _) = lcg.row(rows * conv_dim);
        let mut gates: Vec<f32> = Vec::with_capacity(rows * 2 * v_heads as usize);
        for _ in 0..rows {
            for _ in 0..v_heads {
                gates.push(-0.15 * (lcg.unit() * 0.5 + 0.5));
            }
            for _ in 0..v_heads {
                gates.push(0.2 + 0.7 * (lcg.unit() * 0.5 + 0.5));
            }
        }
        let indptr: Vec<i32> = (0..=requests).map(|r| (r * len) as i32).collect();
        let slot_of: Vec<i32> = (0..requests as i32).collect();
        let qkv_at = gpu.up(&qkv_raw);
        let gates_at = gpu.up(&gates);
        let indptr_at = gpu.up(&indptr);
        let slots_at = gpu.up(&slot_of);
        let slab_at = gpu.zeros(requests * stride * 2);
        let y_at = gpu.zeros(rows * (v_heads * v_dim) as usize * 4);
        let ctx = gpu.ctx().with_tuning(tuned(fused));
        let fire = |_: usize| {
            let pool = RecurrentPool {
                slab: Tensor::new(slab_at, requests as u32, stride as u32, Dtype::Bf16),
                slot_ids: Tensor::new(slots_at, requests as u32, 1, Dtype::I32),
                slot_stride_elems: stride as i64,
                conv_slab: Tensor::ABSENT,
                conv_stride: 0,
                write_state: true,
                write_state_mask: Tensor::ABSENT,
                commit_len: Tensor::ABSENT,
                begin_at: Tensor::ABSENT,
                fused_decay: false,
            };
            let mut y = Tensor::new(y_at, rows as u32, v_heads * v_dim, Dtype::F32);
            ssm::gated_delta_chunked(
                &ctx,
                RaggedTensor {
                    data: Tensor::new(qkv_at, rows as u32, conv_dim as u32, Dtype::Bf16),
                    indptr: Tensor::new(indptr_at, requests as u32 + 1, 1, Dtype::I32),
                },
                Tensor::ABSENT,
                Tensor::new(gates_at, rows as u32, 2 * v_heads, Dtype::F32),
                &pool,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                &mut y,
            )
            .expect("the chunked scan fires");
        };
        let us = timer.median(fire);
        println!(
            "| {} | {requests} x {len} | {:.3} | {:.0} |",
            if fused { "two-launch" } else { "three wmma" },
            us * 1e-3,
            rows as f64 / us * 1e6
        );
    }
}
