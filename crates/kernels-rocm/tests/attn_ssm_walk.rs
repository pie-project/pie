//! A short recurrent run walked in one launch lands the bits of its one-row
//! decode steps: the gated delta rule's outputs, the state at each lane's
//! commit length and at a snapshot, and the causal conv's outputs and
//! windows. kernels-cuda's `ssm::walk` checks, on HIP.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, STATE, values};
use dtype::Dtype;
use kernels_rocm::ssm;
use kernels_rocm::{RaggedTensor, RecurrentPool, Snapshots, Tensor};

const K_HEADS: usize = 2;
const V_HEADS: usize = 4;
const D: usize = 128;
const CONV_DIM: usize = 2 * K_HEADS * D + V_HEADS * D;
const BANK: usize = V_HEADS * D * D;
const SLOTS: usize = 6;
const RUNS: [usize; 3] = [3, 1, 4];
const COMMITS: [i32; 3] = [2, 1, 4];
const SLOT_OF: [i32; 3] = [2, 0, 3];
const SNAP_LANE: usize = 2;
const SNAP_ROW: i32 = 2;
const SNAP_SLOT: i32 = 5;

fn rows() -> usize {
    RUNS.iter().sum()
}

fn indptr() -> Vec<i32> {
    let mut at = vec![0i32];
    for rows in RUNS {
        at.push(at.last().unwrap() + rows as i32);
    }
    at
}

fn snapshots(gpu: &Gpu) -> Snapshots {
    let mut indptr = vec![0i32; RUNS.len() + 1];
    for lane in 0..RUNS.len() {
        indptr[lane + 1] = indptr[lane] + i32::from(lane == SNAP_LANE);
    }
    Snapshots {
        indptr: gpu.ints(&indptr),
        rows: gpu.ints(&[SNAP_ROW]),
        slots: gpu.ints(&[SNAP_SLOT]),
        count: 1,
    }
}

fn gates_of(rows: usize) -> Vec<f32> {
    let raw = values(rows * 2 * V_HEADS, 9, 1.0);
    (0..rows * 2 * V_HEADS)
        .map(|i| {
            if i % (2 * V_HEADS) < V_HEADS {
                -0.05 - 0.5 * raw[i].abs()
            } else {
                0.5 + 0.4 * raw[i]
            }
        })
        .collect()
}

#[test]
fn a_short_gated_delta_run_lands_the_bits_of_its_one_row_steps() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let qkv = values(rows() * CONV_DIM, 1, 1.0);
    let slab = values(SLOTS * BANK, 2, 0.2);
    let gates = gates_of(rows());

    // The one-row steps, each lane stepping its slot of a working slab.
    let work = gpu.tensor(&slab, SLOTS as u32, BANK as u32, Dtype::Bf16);
    let mut want_slab: Vec<u16> = gpu.read_as(&work);
    let mut want_y: Vec<f32> = Vec::new();
    let starts = indptr();
    for lane in 0..RUNS.len() {
        let slot = SLOT_OF[lane] as usize;
        for t in 0..RUNS[lane] {
            let row = starts[lane] as usize + t;
            let pool = RecurrentPool {
                slab: work,
                slot_ids: gpu.column(&[SLOT_OF[lane]]),
                slot_stride_elems: BANK as i64,
                ..STATE
            };
            let mut y = gpu.zeros(1, (V_HEADS * D) as u32, Dtype::F32);
            ssm::gated_delta(
                &ctx,
                gpu.tensor(
                    &qkv[row * CONV_DIM..(row + 1) * CONV_DIM],
                    1,
                    CONV_DIM as u32,
                    Dtype::Bf16,
                ),
                Tensor::ABSENT,
                gpu.tensor(
                    &gates[row * 2 * V_HEADS..(row + 1) * 2 * V_HEADS],
                    1,
                    (2 * V_HEADS) as u32,
                    Dtype::F32,
                ),
                &pool,
                K_HEADS as u32,
                V_HEADS as u32,
                D as u32,
                D as u32,
                &mut y,
            )
            .unwrap();
            want_y.extend(gpu.read_as::<f32>(&y));
            let state: Vec<u16> = gpu.read_as(&work);
            let done = t as i32 + 1;
            if done == COMMITS[lane] {
                want_slab[slot * BANK..(slot + 1) * BANK]
                    .copy_from_slice(&state[slot * BANK..(slot + 1) * BANK]);
            }
            if lane == SNAP_LANE && done == SNAP_ROW {
                let at = SNAP_SLOT as usize * BANK;
                want_slab[at..at + BANK].copy_from_slice(&state[slot * BANK..(slot + 1) * BANK]);
            }
        }
    }

    let slab_t = gpu.tensor(&slab, SLOTS as u32, BANK as u32, Dtype::Bf16);
    let pool = RecurrentPool {
        slab: slab_t,
        slot_ids: gpu.column(&SLOT_OF),
        slot_stride_elems: BANK as i64,
        commit_len: gpu.column(&COMMITS),
        snapshots: snapshots(&gpu),
        ..STATE
    };
    let mut y = gpu.zeros(rows() as u32, (V_HEADS * D) as u32, Dtype::F32);
    let walked = ssm::gated_delta_walk(
        &ctx,
        RaggedTensor {
            data: gpu.tensor(&qkv, rows() as u32, CONV_DIM as u32, Dtype::Bf16),
            indptr: gpu.column(&indptr()),
        },
        gpu.tensor(&gates, rows() as u32, (2 * V_HEADS) as u32, Dtype::F32),
        &pool,
        K_HEADS as u32,
        V_HEADS as u32,
        D as u32,
        D as u32,
        &mut y,
        None,
    )
    .unwrap();
    assert!(walked, "a run of {} rows walks", rows());
    let got_y: Vec<f32> = gpu.read_as(&y);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert!(
        bits(&got_y) == bits(&want_y),
        "the walk's rows are not the steps'"
    );
    assert!(
        gpu.read_as::<u16>(&slab_t) == want_slab,
        "the walk's slab is not the steps'"
    );
}

#[test]
fn a_short_conv_run_lands_the_bits_of_its_one_row_updates() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    const C: usize = 64;
    const K: usize = 4;
    let stride = K * C;
    let x = values(rows() * C, 4, 2.0);
    let w = values(C * K, 5, 0.5);
    let slab = values(SLOTS * stride, 6, 1.0);
    let wt = gpu.tensor(&w, C as u32, K as u32, Dtype::Bf16);

    let work = gpu.tensor(&slab, SLOTS as u32, stride as u32, Dtype::Bf16);
    let mut want_slab: Vec<u16> = gpu.read_as(&work);
    let mut want_y: Vec<u16> = Vec::new();
    let starts = indptr();
    for lane in 0..RUNS.len() {
        let slot = SLOT_OF[lane] as usize;
        for t in 0..RUNS[lane] {
            let row = starts[lane] as usize + t;
            let pool = RecurrentPool {
                conv_slab: work,
                conv_stride: stride as i64,
                slot_ids: gpu.column(&[SLOT_OF[lane]]),
                ..STATE
            };
            let mut y = gpu.zeros(1, C as u32, Dtype::Bf16);
            ssm::causal_conv1d(
                &ctx,
                gpu.tensor(&x[row * C..(row + 1) * C], 1, C as u32, Dtype::Bf16),
                wt,
                None,
                &pool,
                K as u32,
                1,
                &mut y,
            )
            .unwrap();
            want_y.extend(gpu.read_as::<u16>(&y));
            let state: Vec<u16> = gpu.read_as(&work);
            let done = t as i32 + 1;
            if done == COMMITS[lane] {
                want_slab[slot * stride..(slot + 1) * stride]
                    .copy_from_slice(&state[slot * stride..(slot + 1) * stride]);
            }
            if lane == SNAP_LANE && done == SNAP_ROW {
                let at = SNAP_SLOT as usize * stride;
                want_slab[at..at + stride]
                    .copy_from_slice(&state[slot * stride..(slot + 1) * stride]);
            }
        }
    }

    let slab_t = gpu.tensor(&slab, SLOTS as u32, stride as u32, Dtype::Bf16);
    let pool = RecurrentPool {
        conv_slab: slab_t,
        conv_stride: stride as i64,
        slot_ids: gpu.column(&SLOT_OF),
        commit_len: gpu.column(&COMMITS),
        snapshots: snapshots(&gpu),
        ..STATE
    };
    let mut y = gpu.zeros(rows() as u32, C as u32, Dtype::Bf16);
    let walked = ssm::causal_conv1d_walk(
        &ctx,
        RaggedTensor {
            data: gpu.tensor(&x, rows() as u32, C as u32, Dtype::Bf16),
            indptr: gpu.column(&indptr()),
        },
        None,
        wt,
        None,
        &pool,
        K as u32,
        1,
        &mut y,
        None,
    )
    .unwrap();
    assert!(walked, "a run of {} rows walks", rows());
    assert!(
        gpu.read_as::<u16>(&y) == want_y,
        "the walk's rows are not the updates'"
    );
    assert!(
        gpu.read_as::<u16>(&slab_t) == want_slab,
        "the walk's windows are not the updates'"
    );
}
