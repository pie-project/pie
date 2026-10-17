#![cfg(feature = "cuda")]

//! The chunked gated delta rule, on every prefill path, answers the serial
//! scan: its output and its final states.

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

/// A plain fire; a request several chunks long at a three-way head fan; a
/// replay that starts inside a row, commits a prefix and folds only the
/// lanes its predicate names; a staged window, read and landed past its
/// base and seated past its first seat.
#[test]
fn the_chunked_gated_delta_answers_the_serial_scan() {
    let cases = [
        PLAIN,
        Case {
            lens: &[300, 64, 129],
            k_heads: 2,
            v_heads: 6,
            ..PLAIN
        },
        Case {
            commit_len: Some(&[50, 9, 100]),
            begin_at: Some(&[3, 0, 64]),
            folds: Some(&[1, 0, 1]),
            ..PLAIN
        },
        Case {
            commit_len: Some(&[0, 70, 4, 120]),
            window: Some((7, 1)),
            ..PLAIN
        },
    ];
    for case in &cases {
        check(case);
    }
}

/// Each case runs every prefill path: the three wmma launches every card
/// runs, the two-launch scan, and the one-launch tcgen05 walk the sm_100
/// profile turns on (a card that cannot run one falls back to the next).
fn check(case: &Case) {
    for arm in [Arm::Split, Arm::Scan, Arm::Tc5, Arm::ScanBf16, Arm::Tc5Bf16] {
        check_on(case, arm);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Split,
    Scan,
    Tc5,
    /// The walks landing a bf16 output (a trace narrowed for its gated
    /// norm): the tcgen05 walk lands it itself, the scan through a cast.
    ScanBf16,
    Tc5Bf16,
}

impl Arm {
    fn narrow(self) -> bool {
        matches!(self, Arm::ScanBf16 | Arm::Tc5Bf16)
    }
}

fn tuned(arm: Arm) -> Tuning {
    Tuning {
        ssm_gdn_fused_scan: arm != Arm::Split,
        ssm_gdn_tc5: matches!(arm, Arm::Tc5 | Arm::Tc5Bf16),
        ..Tuning::DEFAULT
    }
}

#[allow(clippy::too_many_lines)]
fn check_on(case: &Case, fused: Arm) {
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
        let narrow = !serial && fused.narrow();
        let mut y = Tensor::new(
            y_at,
            live as u32,
            v_heads * v_dim,
            if narrow { Dtype::Bf16 } else { Dtype::F32 },
        );
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
        let cells = rows * (v_heads * v_dim) as usize;
        let y: Vec<f32> = if narrow {
            gpu.down::<u16>(y_at, cells)
                .into_iter()
                .map(common::from_bf16)
                .collect()
        } else {
            gpu.down(y_at, cells)
        };
        (y, gpu.down(slab_at, slots * stride))
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
        "two chunked fires ({fused:?}) disagree with each other: {unstable} output cells, {unstable_state} state cells"
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
        "{fused:?}: output row {} col {}: serial {} against chunked {} (relative error {worst})",
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
        "{fused:?}: the final states differ by {worst_state}"
    );
}
