#![cfg(feature = "cuda")]

//! A snapshot inside a lane lands the state a pass cut there folds, bit for
//! bit, and the lane's rows and its last state are the cut passes' too: the
//! gated delta rule's two-launch scan and its serial walk, and the causal
//! conv's window.

use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::Snapshots;
use kernels_cuda::attn::ssm;
use kernels_cuda::tensor::{RaggedTensor, RecurrentPool, Tensor};
use kernels_cuda::tuning::Tuning;

const D: u32 = 128;
const K_HEADS: u32 = 2;
const V_HEADS: u32 = 4;
const SLOTS: usize = 4;

/// A body's snapshot capacity: every fire of a snapshotting load seats this
/// many, used or not.
const CAPACITY: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Scan,
    Serial,
    Conv,
}

struct Fire<'a> {
    rows: std::ops::Range<u32>,
    lens: &'a [u32],
    slots: &'a [i32],
    snapshots: &'a [(u32, i32)],
}

struct Inputs {
    qkv: Vec<u16>,
    gates: Vec<f32>,
    weight: Vec<u16>,
}

const CONV_K: u32 = 4;

fn width(arm: Arm) -> usize {
    match arm {
        Arm::Conv => 200,
        _ => (2 * K_HEADS * D + V_HEADS * D) as usize,
    }
}

fn stride(arm: Arm) -> usize {
    match arm {
        Arm::Conv => CONV_K as usize * width(arm),
        _ => (V_HEADS * D * D) as usize,
    }
}

/// One fire of `arm` over the slab at `slab_at` (its first lane snapshotting
/// at `snapshots`); its output rows, as bits.
fn fire(gpu: &mut Gpu, arm: Arm, inputs: &Inputs, slab_at: u64, f: &Fire) -> Vec<u32> {
    let rows = f.rows.len();
    let w = width(arm);
    let mut indptr = vec![0i32];
    for &len in f.lens {
        indptr.push(indptr.last().unwrap() + len as i32);
    }
    let count = f.snapshots.len() as i32;
    let mut snap = vec![0, count];
    snap.resize(f.lens.len() + 1, count);
    let snapshots = if f.snapshots.is_empty() {
        Snapshots::NONE
    } else {
        let snap_rows: Vec<i32> = f.snapshots.iter().map(|s| s.0 as i32).collect();
        let snap_slots: Vec<i32> = f.snapshots.iter().map(|s| s.1).collect();
        Snapshots {
            indptr: Tensor::new(gpu.up(&snap), snap.len() as u32, 1, Dtype::I32),
            rows: Tensor::new(gpu.up(&snap_rows), snap_rows.len() as u32, 1, Dtype::I32),
            slots: Tensor::new(gpu.up(&snap_slots), snap_slots.len() as u32, 1, Dtype::I32),
            count: CAPACITY,
        }
    };
    let slots_at = gpu.up(f.slots);
    let indptr_t = Tensor::new(gpu.up(&indptr), indptr.len() as u32, 1, Dtype::I32);
    let slab = Tensor::new(slab_at, SLOTS as u32, stride(arm) as u32, Dtype::Bf16);
    let pool = RecurrentPool {
        slab: if arm == Arm::Conv {
            Tensor::ABSENT
        } else {
            slab
        },
        slot_ids: Tensor::new(slots_at, f.slots.len() as u32, 1, Dtype::I32),
        slot_stride_elems: if arm == Arm::Conv {
            0
        } else {
            stride(arm) as i64
        },
        conv_slab: if arm == Arm::Conv {
            slab
        } else {
            Tensor::ABSENT
        },
        conv_stride: if arm == Arm::Conv {
            stride(arm) as i64
        } else {
            0
        },
        write_state: true,
        write_state_mask: Tensor::ABSENT,
        commit_len: Tensor::ABSENT,
        begin_at: Tensor::ABSENT,
        fused_decay: false,
        snapshots,
    };
    let (r0, r1) = (f.rows.start as usize, f.rows.end as usize);
    let x_at = gpu.up(&inputs.qkv[r0 * w..r1 * w]);
    let x = RaggedTensor {
        data: Tensor::new(x_at, rows as u32, w as u32, Dtype::Bf16),
        indptr: indptr_t,
    };
    let ctx = gpu.ctx().with_tuning(Tuning {
        ssm_gdn_fused_scan: true,
        ..Tuning::DEFAULT
    });
    match arm {
        Arm::Conv => {
            let y_at = gpu.zeros(rows * w * 2);
            let mut y = Tensor::new(y_at, rows as u32, w as u32, Dtype::Bf16);
            let weight = Tensor::new(gpu.up(&inputs.weight), w as u32, CONV_K, Dtype::Bf16);
            ssm::causal_conv1d_chunked(&ctx, x, weight, None, &pool, CONV_K, 1, &mut y)
                .expect("the chunked conv fires");
            gpu.sync();
            gpu.down::<u16>(y_at, rows * w)
                .into_iter()
                .map(u32::from)
                .collect()
        }
        Arm::Scan | Arm::Serial => {
            let g = 2 * V_HEADS as usize;
            let gates_at = gpu.up(&inputs.gates[r0 * g..r1 * g]);
            let gates = Tensor::new(gates_at, rows as u32, 2 * V_HEADS, Dtype::F32);
            let cells = rows * (V_HEADS * D) as usize;
            let y_at = gpu.zeros(cells * 4);
            let mut y = Tensor::new(y_at, rows as u32, V_HEADS * D, Dtype::F32);
            let scan = if arm == Arm::Serial {
                ssm::gated_delta_chunked_serial
            } else {
                ssm::gated_delta_chunked
            };
            scan(
                &ctx,
                x,
                Tensor::ABSENT,
                gates,
                &pool,
                K_HEADS,
                V_HEADS,
                D,
                D,
                &mut y,
            )
            .expect("the gated delta fires");
            gpu.sync();
            gpu.down::<f32>(y_at, cells)
                .into_iter()
                .map(f32::to_bits)
                .collect()
        }
    }
}

#[test]
fn a_snapshot_lands_what_a_pass_cut_there_folds() {
    // Lane 0 snapshots mid-chunk and a whole chunk later; lane 1 rides along.
    let (a, b) = (200u32, 70u32);
    let cuts = [0u32, 100, 164, a];
    for arm in [Arm::Scan, Arm::Serial, Arm::Conv] {
        let mut gpu = Gpu::open();
        let mut lcg = Lcg::seeded(0x5a);
        let w = width(arm);
        let rows = (a + b) as usize;
        let (qkv, _) = lcg.row(rows * w);
        let mut gates: Vec<f32> = Vec::with_capacity(rows * 2 * V_HEADS as usize);
        for _ in 0..rows {
            for _ in 0..V_HEADS {
                gates.push(-0.02 - 0.2 * (lcg.unit() * 0.5 + 0.5));
            }
            for _ in 0..V_HEADS {
                gates.push(0.5 + 0.45 * lcg.unit());
            }
        }
        let (weight, _) = lcg.row(w * CONV_K as usize);
        let inputs = Inputs { qkv, gates, weight };
        let stride = stride(arm);
        let (init, _) = lcg.row(SLOTS * stride);
        let init: Vec<u16> = init.iter().map(|&v| to_bf16(from_bf16(v) * 0.05)).collect();
        let slot = |slab: &[u16], s: usize| slab[s * stride..(s + 1) * stride].to_vec();

        let whole_at = gpu.up(&init);
        let y = fire(
            &mut gpu,
            arm,
            &inputs,
            whole_at,
            &Fire {
                rows: 0..a + b,
                lens: &[a, b],
                slots: &[0, 1],
                snapshots: &[(cuts[1], 2), (cuts[2], 3)],
            },
        );
        let whole: Vec<u16> = gpu.down(whole_at, SLOTS * stride);

        let cut_at = gpu.up(&init);
        let mut y_cut = Vec::new();
        for (piece, pair) in cuts.windows(2).enumerate() {
            y_cut.extend(fire(
                &mut gpu,
                arm,
                &inputs,
                cut_at,
                &Fire {
                    rows: pair[0]..pair[1],
                    lens: &[pair[1] - pair[0]],
                    slots: &[0],
                    snapshots: &[],
                },
            ));
            if piece < 2 {
                let cut: Vec<u16> = gpu.down(cut_at, SLOTS * stride);
                assert_eq!(
                    slot(&whole, 2 + piece),
                    slot(&cut, 0),
                    "{arm:?}: the snapshot at row {} is not the state a pass cut there folds",
                    pair[1]
                );
            }
        }
        y_cut.extend(fire(
            &mut gpu,
            arm,
            &inputs,
            cut_at,
            &Fire {
                rows: a..a + b,
                lens: &[b],
                slots: &[1],
                snapshots: &[],
            },
        ));
        let cut: Vec<u16> = gpu.down(cut_at, SLOTS * stride);
        assert_eq!(y, y_cut, "{arm:?}: the rows differ from the cut passes'");
        for s in 0..2 {
            assert_eq!(
                slot(&whole, s),
                slot(&cut, s),
                "{arm:?}: lane {s}'s last state differs"
            );
        }
    }
}
