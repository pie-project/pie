#![cfg(feature = "cuda")]

//! A short recurrent run walked in one launch lands the bits of its one-row
//! decode steps: the gated delta rule's outputs, the state it lands at each
//! lane's commit length and at its snapshots, the causal conv's outputs and
//! windows, and the gate prep over an extended run read in place.

use crate::common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::gather::PagedPlane;
use kernels_cuda::ssm::{self, RunRows};
use kernels_cuda::tensor::{RaggedTensor, RecurrentPool, Snapshots, Tensor};

const K_HEADS: usize = 2;
const V_HEADS: usize = 4;
const K_DIM: usize = 128;
const V_DIM: usize = 128;
const CONV_DIM: usize = 2 * K_HEADS * K_DIM + V_HEADS * V_DIM;
const BANK: usize = V_HEADS * K_DIM * V_DIM;
const SLOTS: usize = 6;

/// Three lanes' runs: their rows, the rows each commits, and the slot each
/// reads. Lane 0 walks a row past its commit; lane 2 lands a snapshot.
const RUNS: [u32; 3] = [3, 1, 4];
const COMMITS: [i32; 3] = [2, 1, 4];
const SLOT_OF: [i32; 3] = [2, 0, 3];
const SNAP_LANE: usize = 2;
const SNAP_ROW: i32 = 2;
const SNAP_SLOT: i32 = 5;

fn indptr() -> Vec<i32> {
    let mut at = vec![0i32];
    for rows in RUNS {
        at.push(at.last().unwrap() + rows as i32);
    }
    at
}

fn rows() -> usize {
    RUNS.iter().sum::<u32>() as usize
}

struct Delta {
    qkv: Vec<u16>,
    gates: Vec<f32>,
    slab: Vec<u16>,
}

fn delta_inputs() -> Delta {
    let mut lcg = Lcg::seeded(0x3a1c);
    let (qkv, _) = lcg.row(rows() * CONV_DIM);
    let (slab, _) = lcg.row(SLOTS * BANK);
    let mut gates = vec![0f32; rows() * 2 * V_HEADS];
    for r in 0..rows() {
        for h in 0..V_HEADS {
            gates[r * 2 * V_HEADS + h] = -0.05 - 0.5 * lcg.unit().abs();
            gates[r * 2 * V_HEADS + V_HEADS + h] = 0.5 + 0.4 * lcg.unit();
        }
    }
    Delta { qkv, gates, slab }
}

fn pool(slab: u64, slots: u64, lanes: u32) -> RecurrentPool {
    RecurrentPool {
        slab: Tensor::new(slab, SLOTS as u32, BANK as u32, Dtype::Bf16),
        slot_ids: Tensor::new(slots, lanes, 1, Dtype::I32),
        slot_stride_elems: BANK as i64,
        conv_slab: Tensor::ABSENT,
        conv_stride: 0,
        write_state: true,
        write_state_mask: Tensor::ABSENT,
        commit_len: Tensor::ABSENT,
        begin_at: Tensor::ABSENT,
        fused_decay: false,
        snapshots: Snapshots::NONE,
    }
}

/// What the one-row steps land: each row's f32 output, and the slab after
/// each lane's commit (and the snapshot), each lane stepping a copy of its
/// slot row by row.
fn delta_steps(gpu: &mut Gpu, d: &Delta) -> (Vec<f32>, Vec<u16>) {
    let ctx = gpu.ctx();
    let qkv_at = gpu.up(&d.qkv);
    let gates_at = gpu.up(&d.gates);
    let work_at = gpu.up(&d.slab);
    let y_at = gpu.zeros(rows() * V_HEADS * V_DIM * 4);
    let mut want_slab = d.slab.clone();
    let starts = indptr();
    for lane in 0..RUNS.len() {
        let slot = SLOT_OF[lane] as usize;
        let slot_at = gpu.up(&[SLOT_OF[lane]]);
        for t in 0..RUNS[lane] as usize {
            let row = starts[lane] as usize + t;
            let mut y = Tensor::new(
                y_at + (row * V_HEADS * V_DIM * 4) as u64,
                1,
                (V_HEADS * V_DIM) as u32,
                Dtype::F32,
            );
            ssm::gated_delta(
                &ctx,
                Tensor::new(
                    qkv_at + (row * CONV_DIM * 2) as u64,
                    1,
                    CONV_DIM as u32,
                    Dtype::Bf16,
                ),
                Tensor::ABSENT,
                Tensor::new(
                    gates_at + (row * 2 * V_HEADS * 4) as u64,
                    1,
                    (2 * V_HEADS) as u32,
                    Dtype::F32,
                ),
                &pool(work_at, slot_at, 1),
                K_HEADS as u32,
                V_HEADS as u32,
                K_DIM as u32,
                V_DIM as u32,
                &mut y,
            )
            .expect("the one-row step fires");
            gpu.sync();
            let done = t as i32 + 1;
            let state: Vec<u16> = gpu.down(work_at + (slot * BANK * 2) as u64, BANK);
            if done == COMMITS[lane] {
                want_slab[slot * BANK..(slot + 1) * BANK].copy_from_slice(&state);
            }
            if lane == SNAP_LANE && done == SNAP_ROW {
                let at = SNAP_SLOT as usize * BANK;
                want_slab[at..at + BANK].copy_from_slice(&state);
            }
        }
    }
    (gpu.down(y_at, rows() * V_HEADS * V_DIM), want_slab)
}

fn snapshots(gpu: &mut Gpu) -> Snapshots {
    let mut indptr = vec![0i32; RUNS.len() + 1];
    for lane in 0..RUNS.len() {
        indptr[lane + 1] = indptr[lane] + i32::from(lane == SNAP_LANE);
    }
    Snapshots {
        indptr: Tensor::new(gpu.up(&indptr), 1, 1, Dtype::I32),
        rows: Tensor::new(gpu.up(&[SNAP_ROW]), 1, 1, Dtype::I32),
        slots: Tensor::new(gpu.up(&[SNAP_SLOT]), 1, 1, Dtype::I32),
        count: 1,
    }
}

/// The walk over three lanes' runs answers their one-row steps bit for bit:
/// every row's output, each slot's state at its lane's commit (lane 0's
/// row past it walking on without landing), the snapshot's state, and the
/// slots no lane names left alone. Once landing f32 rows, once bf16 rows
/// (the step's f32 narrowed to nearest).
#[test]
fn a_short_gated_delta_run_lands_the_bits_of_its_one_row_steps() {
    let d = delta_inputs();
    let mut gpu = Gpu::open();
    let (want_y, want_slab) = delta_steps(&mut gpu, &d);
    for narrow in [false, true] {
        let ctx = gpu.ctx();
        let qkv_at = gpu.up(&d.qkv);
        let gates_at = gpu.up(&d.gates);
        let slab_at = gpu.up(&d.slab);
        let slots_at = gpu.up(&SLOT_OF);
        let commits_at = gpu.up(&COMMITS);
        let indptr_at = gpu.up(&indptr());
        let elem = if narrow { 2 } else { 4 };
        let y_at = gpu.zeros(rows() * V_HEADS * V_DIM * elem);
        let mut walk_pool = pool(slab_at, slots_at, RUNS.len() as u32);
        walk_pool.commit_len = Tensor::new(commits_at, RUNS.len() as u32, 1, Dtype::I32);
        walk_pool.snapshots = snapshots(&mut gpu);
        let mut y = Tensor::new(
            y_at,
            rows() as u32,
            (V_HEADS * V_DIM) as u32,
            if narrow { Dtype::Bf16 } else { Dtype::F32 },
        );
        let walked = ssm::gated_delta_walk(
            &ctx,
            RaggedTensor {
                data: Tensor::new(qkv_at, rows() as u32, CONV_DIM as u32, Dtype::Bf16),
                indptr: Tensor::new(indptr_at, RUNS.len() as u32 + 1, 1, Dtype::I32),
            },
            Tensor::new(gates_at, rows() as u32, (2 * V_HEADS) as u32, Dtype::F32),
            &walk_pool,
            K_HEADS as u32,
            V_HEADS as u32,
            K_DIM as u32,
            V_DIM as u32,
            &mut y,
            None,
        )
        .expect("the walk fires");
        assert!(walked, "a run of {} rows walks", rows());
        gpu.sync();
        let got_slab: Vec<u16> = gpu.down(slab_at, SLOTS * BANK);
        assert!(
            got_slab == want_slab,
            "narrow {narrow}: the walk's slab is not the steps'"
        );
        if narrow {
            let got: Vec<u16> = gpu.down(y_at, rows() * V_HEADS * V_DIM);
            let want: Vec<u16> = want_y.iter().map(|&v| crate::common::to_bf16(v)).collect();
            assert!(
                got == want,
                "the walk's bf16 rows are not the steps' narrowed"
            );
        } else {
            let got: Vec<f32> = gpu.down(y_at, rows() * V_HEADS * V_DIM);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert!(
                bits(&got) == bits(&want_y),
                "the walk's rows are not the steps'"
            );
        }
    }
}

/// Over an extended run the walk lands only the rows the fire owns, at the
/// fire rows `back` names less the plane's first row; a replayed row's
/// output lands nowhere.
#[test]
fn an_extended_gated_delta_walk_lands_only_the_fires_own_rows() {
    let d = delta_inputs();
    let mut gpu = Gpu::open();
    let (want_y, _) = delta_steps(&mut gpu, &d);
    let ctx = gpu.ctx();
    // Lane 0 replays two rows, lane 2 one; the fire's own rows stand from
    // fire row 3, the handed plane's first.
    let back: Vec<i32> = vec![-1, -1, 3, 4, -1, 5, 6, 7];
    const SKIP: u32 = 3;
    let fire_rows = 5usize;
    let qkv_at = gpu.up(&d.qkv);
    let gates_at = gpu.up(&d.gates);
    let slab_at = gpu.up(&d.slab);
    let slots_at = gpu.up(&SLOT_OF);
    let commits_at = gpu.up(&COMMITS);
    let indptr_at = gpu.up(&indptr());
    let back_at = gpu.up(&back);
    let width = V_HEADS * V_DIM;
    let y_at = gpu.zeros(fire_rows * width * 4);
    let mut walk_pool = pool(slab_at, slots_at, RUNS.len() as u32);
    walk_pool.commit_len = Tensor::new(commits_at, RUNS.len() as u32, 1, Dtype::I32);
    let mut y = Tensor::new(y_at, fire_rows as u32, width as u32, Dtype::F32);
    assert!(
        ssm::gated_delta_walk(
            &ctx,
            RaggedTensor {
                data: Tensor::new(qkv_at, rows() as u32, CONV_DIM as u32, Dtype::Bf16),
                indptr: Tensor::new(indptr_at, RUNS.len() as u32 + 1, 1, Dtype::I32),
            },
            Tensor::new(gates_at, rows() as u32, (2 * V_HEADS) as u32, Dtype::F32),
            &walk_pool,
            K_HEADS as u32,
            V_HEADS as u32,
            K_DIM as u32,
            V_DIM as u32,
            &mut y,
            Some((Tensor::new(back_at, rows() as u32, 1, Dtype::I32), SKIP)),
        )
        .expect("the walk fires")
    );
    gpu.sync();
    let got: Vec<f32> = gpu.down(y_at, fire_rows * width);
    for (e, &fire) in back.iter().enumerate() {
        if fire < 0 {
            continue;
        }
        let at = (fire as usize - SKIP as usize) * width;
        let want = &want_y[e * width..(e + 1) * width];
        assert!(
            got[at..at + width]
                .iter()
                .zip(want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "extended row {e} did not land at fire row {fire}"
        );
    }
}

const CHANNELS: usize = 64;
const TAPS: usize = 4;
const PAGE_TOKENS: u32 = 4;
const PAGES: usize = 3;

/// The conv's one-row updates over the rows `x` holds in run order: each
/// row's output and the window slab after each lane's commit (and the
/// snapshot).
fn conv_steps(gpu: &mut Gpu, x: &[u16], w: &[u16], slab: &[u16]) -> (Vec<u16>, Vec<u16>) {
    let ctx = gpu.ctx();
    let stride = TAPS * CHANNELS;
    let x_at = gpu.up(x);
    let w_at = gpu.up(w);
    let work_at = gpu.up(slab);
    let y_at = gpu.zeros(rows() * CHANNELS * 2);
    let mut want = slab.to_vec();
    let starts = indptr();
    for lane in 0..RUNS.len() {
        let slot = SLOT_OF[lane] as usize;
        let slot_at = gpu.up(&[SLOT_OF[lane]]);
        for t in 0..RUNS[lane] as usize {
            let row = starts[lane] as usize + t;
            let step_pool = RecurrentPool {
                slab: Tensor::ABSENT,
                slot_ids: Tensor::new(slot_at, 1, 1, Dtype::I32),
                slot_stride_elems: 0,
                conv_slab: Tensor::new(work_at, SLOTS as u32, stride as u32, Dtype::Bf16),
                conv_stride: stride as i64,
                write_state: true,
                write_state_mask: Tensor::ABSENT,
                commit_len: Tensor::ABSENT,
                begin_at: Tensor::ABSENT,
                fused_decay: false,
                snapshots: Snapshots::NONE,
            };
            let mut y = Tensor::new(
                y_at + (row * CHANNELS * 2) as u64,
                1,
                CHANNELS as u32,
                Dtype::Bf16,
            );
            ssm::causal_conv1d(
                &ctx,
                Tensor::new(
                    x_at + (row * CHANNELS * 2) as u64,
                    1,
                    CHANNELS as u32,
                    Dtype::Bf16,
                ),
                Tensor::new(w_at, CHANNELS as u32, TAPS as u32, Dtype::Bf16),
                None,
                &step_pool,
                TAPS as u32,
                1,
                &mut y,
            )
            .expect("the one-row update fires");
            gpu.sync();
            let done = t as i32 + 1;
            let window: Vec<u16> = gpu.down(work_at + (slot * stride * 2) as u64, stride);
            if done == COMMITS[lane] {
                want[slot * stride..(slot + 1) * stride].copy_from_slice(&window);
            }
            if lane == SNAP_LANE && done == SNAP_ROW {
                let at = SNAP_SLOT as usize * stride;
                want[at..at + stride].copy_from_slice(&window);
            }
        }
    }
    (gpu.down(y_at, rows() * CHANNELS), want)
}

/// The conv walk answers its one-row updates bit for bit, over the lanes'
/// own rows and over an extended run whose replayed rows it reads out of
/// their pages: every row's output (the fire's own rows also in the fire's
/// plane), each lane's window at its commit, and the snapshot's.
#[test]
fn a_short_conv_run_lands_the_bits_of_its_one_row_updates() {
    let mut lcg = Lcg::seeded(0xc0f);
    let (x, _) = lcg.row(rows() * CHANNELS);
    let (w, _) = lcg.row(CHANNELS * TAPS);
    let (slab, _) = lcg.row(SLOTS * TAPS * CHANNELS);
    let mut gpu = Gpu::open();
    let (want_y, want_slab) = conv_steps(&mut gpu, &x, &w, &slab);
    let stride = TAPS * CHANNELS;

    // Extended rows 0, 1 and 4 replay page tokens; the rest are the fire's
    // own, which stand in the fire's plane from fire row 2.
    let replay: Vec<i32> = vec![5, 9, -1, -1, 2, -1, -1, -1];
    let back: Vec<i32> = vec![-1, -1, 2, 3, -1, 4, 5, 6];
    const SKIP: u32 = 2;
    let mut pages = vec![0u16; PAGES * PAGE_TOKENS as usize * CHANNELS];
    let mut own = vec![0u16; 5 * CHANNELS];
    for e in 0..rows() {
        let row = &x[e * CHANNELS..(e + 1) * CHANNELS];
        if replay[e] >= 0 {
            let at = replay[e] as usize * CHANNELS;
            pages[at..at + CHANNELS].copy_from_slice(row);
        } else {
            let at = (back[e] as usize - SKIP as usize) * CHANNELS;
            own[at..at + CHANNELS].copy_from_slice(row);
        }
    }
    for extended in [false, true] {
        let ctx = gpu.ctx();
        let slab_at = gpu.up(&slab);
        let slots_at = gpu.up(&SLOT_OF);
        let commits_at = gpu.up(&COMMITS);
        let indptr_at = gpu.up(&indptr());
        let w_at = gpu.up(&w);
        let walk_pool = RecurrentPool {
            slab: Tensor::ABSENT,
            slot_ids: Tensor::new(slots_at, RUNS.len() as u32, 1, Dtype::I32),
            slot_stride_elems: 0,
            conv_slab: Tensor::new(slab_at, SLOTS as u32, stride as u32, Dtype::Bf16),
            conv_stride: stride as i64,
            write_state: true,
            write_state_mask: Tensor::ABSENT,
            commit_len: Tensor::new(commits_at, RUNS.len() as u32, 1, Dtype::I32),
            begin_at: Tensor::ABSENT,
            fused_decay: false,
            snapshots: snapshots(&mut gpu),
        };
        let csr = Tensor::new(indptr_at, RUNS.len() as u32 + 1, 1, Dtype::I32);
        let (x_at, run, y_rows) = if extended {
            let pages_at = gpu.up(&pages);
            let run = RunRows {
                pages: PagedPlane {
                    base: pages_at,
                    per_slot: u64::from(PAGE_TOKENS) * CHANNELS as u64,
                    at: 0,
                    width: CHANNELS as u32,
                    page_tokens: PAGE_TOKENS,
                    dtype: Dtype::Bf16,
                },
                replay: Tensor::new(gpu.up(&replay), rows() as u32, 1, Dtype::I32),
                back: Tensor::new(gpu.up(&back), rows() as u32, 1, Dtype::I32),
                skip: SKIP,
            };
            (gpu.up(&own), Some(run), 5)
        } else {
            (gpu.up(&x), None, rows())
        };
        let y_at = gpu.zeros(y_rows * CHANNELS * 2);
        let ext_at = gpu.zeros(rows() * CHANNELS * 2);
        let mut y = Tensor::new(y_at, y_rows as u32, CHANNELS as u32, Dtype::Bf16);
        let mut y_ext = Tensor::new(ext_at, rows() as u32, CHANNELS as u32, Dtype::Bf16);
        let walked = ssm::causal_conv1d_walk(
            &ctx,
            RaggedTensor {
                data: Tensor::new(x_at, y_rows as u32, CHANNELS as u32, Dtype::Bf16),
                indptr: csr,
            },
            run,
            Tensor::new(w_at, CHANNELS as u32, TAPS as u32, Dtype::Bf16),
            None,
            &walk_pool,
            TAPS as u32,
            1,
            &mut y,
            extended.then_some(&mut y_ext),
        )
        .expect("the conv walk fires");
        assert!(walked, "a run of {} rows walks", rows());
        gpu.sync();
        let got_slab: Vec<u16> = gpu.down(slab_at, SLOTS * stride);
        assert!(
            got_slab == want_slab,
            "extended {extended}: the windows are not the updates'"
        );
        let got_y: Vec<u16> = gpu.down(y_at, y_rows * CHANNELS);
        if extended {
            let got_ext: Vec<u16> = gpu.down(ext_at, rows() * CHANNELS);
            assert!(got_ext == want_y, "the extended rows are not the updates'");
            for e in 0..rows() {
                if back[e] < 0 {
                    continue;
                }
                let at = (back[e] as usize - SKIP as usize) * CHANNELS;
                assert_eq!(
                    got_y[at..at + CHANNELS],
                    want_y[e * CHANNELS..(e + 1) * CHANNELS],
                    "the fire's own row of extended row {e}"
                );
            }
        } else {
            assert!(got_y == want_y, "the rows are not the updates'");
        }
    }
}

/// The gate prep over an extended run reads each row's `[b | a]` where it
/// stands (a replayed row in its page, an own row in the fire's plane) and
/// answers what the gate prep answers over the rows gathered.
#[test]
fn the_gate_prep_over_a_run_answers_the_gathered_rows() {
    const HEADS: usize = 16;
    let width = 2 * HEADS;
    let mut lcg = Lcg::seeded(0x9a7);
    let ext = 6usize;
    let (rows_raw, _) = lcg.row(ext * width);
    let (dt_raw, _) = lcg.row(HEADS);
    let a_log: Vec<f32> = (0..HEADS).map(|_| lcg.unit()).collect();
    let replay: Vec<i32> = vec![7, -1, 1, -1, -1, -1];
    let back: Vec<i32> = vec![-1, 4, -1, 5, 6, -1];
    const SKIP: u32 = 4;
    let mut pages = vec![0u16; 2 * PAGE_TOKENS as usize * width];
    let mut own = vec![0u16; 3 * width];
    for e in 0..ext {
        let row = &rows_raw[e * width..(e + 1) * width];
        if replay[e] >= 0 {
            pages[replay[e] as usize * width..][..width].copy_from_slice(row);
        } else if back[e] >= 0 {
            own[(back[e] as usize - SKIP as usize) * width..][..width].copy_from_slice(row);
        }
    }
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let gathered_at = gpu.up(&rows_raw);
    let dt_at = gpu.up(&dt_raw);
    let a_at = gpu.up(&a_log);
    let want_at = gpu.zeros(ext * width * 4);
    let mut want = Tensor::new(want_at, ext as u32, width as u32, Dtype::F32);
    ssm::gdn_gates(
        &ctx,
        Tensor::new(gathered_at, ext as u32, width as u32, Dtype::Bf16),
        Tensor::new(dt_at, 1, HEADS as u32, Dtype::Bf16),
        Tensor::new(a_at, 1, HEADS as u32, Dtype::F32),
        &mut want,
    )
    .expect("the gate prep fires");
    let pages_at = gpu.up(&pages);
    let own_at = gpu.up(&own);
    let ext_at = gpu.zeros(ext * width * 4);
    let mine_at = gpu.zeros(3 * width * 4);
    let run = RunRows {
        pages: PagedPlane {
            base: pages_at,
            per_slot: u64::from(PAGE_TOKENS) * width as u64,
            at: 0,
            width: width as u32,
            page_tokens: PAGE_TOKENS,
            dtype: Dtype::Bf16,
        },
        replay: Tensor::new(gpu.up(&replay), ext as u32, 1, Dtype::I32),
        back: Tensor::new(gpu.up(&back), ext as u32, 1, Dtype::I32),
        skip: SKIP,
    };
    ssm::gdn_gates_run(
        &ctx,
        Tensor::new(own_at, 3, width as u32, Dtype::Bf16),
        run,
        Tensor::new(dt_at, 1, HEADS as u32, Dtype::Bf16),
        Tensor::new(a_at, 1, HEADS as u32, Dtype::F32),
        &mut Tensor::new(ext_at, ext as u32, width as u32, Dtype::F32),
        &mut Tensor::new(mine_at, 3, width as u32, Dtype::F32),
    )
    .expect("the run's gate prep fires");
    gpu.sync();
    let want: Vec<f32> = gpu.down(want_at, ext * width);
    let got: Vec<f32> = gpu.down(ext_at, ext * width);
    let mine: Vec<f32> = gpu.down(mine_at, 3 * width);
    for e in 0..ext {
        if replay[e] < 0 && back[e] < 0 {
            continue;
        }
        let w = &want[e * width..(e + 1) * width];
        assert!(
            got[e * width..(e + 1) * width]
                .iter()
                .zip(w)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "extended row {e}"
        );
        if back[e] >= 0 {
            let at = (back[e] as usize - SKIP as usize) * width;
            assert!(
                mine[at..at + width]
                    .iter()
                    .zip(w)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "the fire's own row of extended row {e}"
            );
        }
    }
}

/// The crossover `kernel.ssm.gdn_scan_until` stands at: a gated-delta
/// layer's recurrence (Qwen3.5-0.8B's, 16 x 16 heads of 128) over a run of
/// `rows` rows, walked in order against tiled into chunks, at one lane and
/// at four, and the causal conv in front of it, walked against chunked.
/// Prints microseconds a launch set; run by hand (`--ignored --nocapture`).
#[test]
#[ignore = "a measurement, not a check"]
fn the_walk_crossover_is_where_a_run_costs_a_tile() {
    const KH: usize = 16;
    const VH: usize = 16;
    const D: usize = 128;
    const CONV: usize = 2 * KH * D + VH * D;
    const MAX_ROWS: usize = 256;
    const BANK: usize = VH * D * D;
    let mut lcg = Lcg::seeded(0xbe7c);
    let (qkv, _) = lcg.row(MAX_ROWS * CONV);
    let (slab, _) = lcg.row(8 * BANK);
    let (w, _) = lcg.row(CONV * TAPS);
    let (conv_slab, _) = lcg.row(8 * TAPS * CONV);
    let mut gates = vec![0f32; MAX_ROWS * 2 * VH];
    for r in 0..MAX_ROWS {
        for h in 0..VH {
            gates[r * 2 * VH + h] = -0.05 - 0.5 * lcg.unit().abs();
            gates[r * 2 * VH + VH + h] = 0.5 + 0.4 * lcg.unit();
        }
    }
    let mut gpu = Gpu::open();
    let qkv_at = gpu.up(&qkv);
    let gates_at = gpu.up(&gates);
    let slab_at = gpu.up(&slab);
    let w_at = gpu.up(&w);
    let conv_at = gpu.up(&conv_slab);
    let slots_at = gpu.up(&[0i32, 1, 2, 3]);
    let y_at = gpu.zeros(MAX_ROWS * VH * D * 4);
    let x_out_at = gpu.zeros(MAX_ROWS * CONV * 2);
    let base = gpu.ctx().tuning();
    let at_until = |until: u32, fused: bool| {
        gpu.ctx().with_tuning(kernels_cuda::tuning::Tuning {
            ssm_gdn_scan_until: until,
            ssm_gdn_fused_scan: fused,
            ..base
        })
    };
    // The tiled side both ways a profile may take it: the three-launch
    // split, and the two-launch scan (`kernel.ssm.gdn_fused_scan`, on where
    // a block holds a quarter head, as sm_89's does).
    let (walked, tiled, scanned) = (at_until(4096, false), at_until(0, false), at_until(0, true));
    println!("rows lanes | gdn walk us  tiled us  2-launch us | conv walk us  chunked us");
    for lanes in [1usize, 4] {
        for rows in [1usize, 2, 4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 256] {
            if rows < lanes {
                continue;
            }
            let per = rows / lanes;
            let indptr: Vec<i32> = (0..=lanes).map(|l| (l * per) as i32).collect();
            let rows = per * lanes;
            let indptr_at = gpu.up(&indptr);
            let csr = Tensor::new(indptr_at, lanes as u32 + 1, 1, Dtype::I32);
            let qkv_t = Tensor::new(qkv_at, rows as u32, CONV as u32, Dtype::Bf16);
            let gates_t = Tensor::new(gates_at, rows as u32, (2 * VH) as u32, Dtype::F32);
            let mut gdn_pool = RecurrentPool {
                slab: Tensor::new(slab_at, 8, BANK as u32, Dtype::Bf16),
                slot_ids: Tensor::new(slots_at, lanes as u32, 1, Dtype::I32),
                slot_stride_elems: BANK as i64,
                conv_slab: Tensor::ABSENT,
                conv_stride: 0,
                write_state: true,
                write_state_mask: Tensor::ABSENT,
                commit_len: Tensor::ABSENT,
                begin_at: Tensor::ABSENT,
                fused_decay: false,
                snapshots: Snapshots::NONE,
            };
            let gdn = |ctx: &kernels_cuda::Ctx, pool: &RecurrentPool| {
                let mut y = Tensor::new(y_at, rows as u32, (VH * D) as u32, Dtype::F32);
                let ragged = RaggedTensor {
                    data: qkv_t,
                    indptr: csr,
                };
                if !ssm::gated_delta_walk(
                    ctx, ragged, gates_t, pool, KH as u32, VH as u32, D as u32, D as u32, &mut y,
                    None,
                )
                .expect("the walk fires")
                {
                    ssm::gated_delta_chunked(
                        ctx,
                        ragged,
                        Tensor::ABSENT,
                        gates_t,
                        pool,
                        KH as u32,
                        VH as u32,
                        D as u32,
                        D as u32,
                        &mut y,
                    )
                    .expect("the chunked scan fires");
                }
            };
            gdn_pool.write_state = true;
            let walk_us = gpu.time(20, || gdn(&walked, &gdn_pool));
            let tiled_us = gpu.time(20, || gdn(&tiled, &gdn_pool));
            let scanned_us = gpu.time(20, || gdn(&scanned, &gdn_pool));
            let conv_pool = RecurrentPool {
                slab: Tensor::ABSENT,
                slot_ids: Tensor::new(slots_at, lanes as u32, 1, Dtype::I32),
                slot_stride_elems: 0,
                conv_slab: Tensor::new(conv_at, 8, (TAPS * CONV) as u32, Dtype::Bf16),
                conv_stride: (TAPS * CONV) as i64,
                ..gdn_pool
            };
            let conv = |ctx: &kernels_cuda::Ctx| {
                let mut y = Tensor::new(x_out_at, rows as u32, CONV as u32, Dtype::Bf16);
                let ragged = RaggedTensor {
                    data: qkv_t,
                    indptr: csr,
                };
                let weight = Tensor::new(w_at, CONV as u32, TAPS as u32, Dtype::Bf16);
                if !ssm::causal_conv1d_walk(
                    ctx,
                    ragged,
                    None,
                    weight,
                    None,
                    &conv_pool,
                    TAPS as u32,
                    1,
                    &mut y,
                    None,
                )
                .expect("the conv walk fires")
                {
                    ssm::causal_conv1d_chunked(
                        ctx,
                        ragged,
                        weight,
                        None,
                        &conv_pool,
                        TAPS as u32,
                        1,
                        &mut y,
                    )
                    .expect("the chunked conv fires");
                }
            };
            let conv_walk_us = gpu.time(20, || conv(&walked));
            let conv_chunk_us = gpu.time(20, || conv(&tiled));
            println!(
                "{rows:4} {lanes:5} | {walk_us:11.1} {tiled_us:9.1} {scanned_us:12.1} | {conv_walk_us:12.1} {conv_chunk_us:11.1}"
            );
        }
    }
}
