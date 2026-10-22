#![cfg(feature = "cuda")]

use crate::common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::ssm;
use kernels_cuda::tensor::{RaggedTensor, RecurrentPool, Tensor};

const HEADS: u32 = 2;
const HEAD_DIM: u32 = 32;
const WIDE: usize = (HEADS * HEAD_DIM) as usize;
const LENS: [usize; 2] = [9, 6];
const SLOT_OF: [i32; 2] = [1, 0];
const STRIDE: usize = (HEADS * HEAD_DIM * HEAD_DIM) as usize;

struct Rows {
    mixed: u64,
    f: u64,
    b: u64,
    dt_bias: u64,
    a_log: u64,
    slab: Vec<f32>,
}

fn rows(gpu: &mut Gpu) -> Rows {
    let rows: usize = LENS.iter().sum();
    let mut lcg = Lcg::seeded(0x4b);
    let (mixed, _) = lcg.row(rows * 3 * WIDE);
    let (f, _) = lcg.row(rows * WIDE);
    let (b, _) = lcg.row(rows * HEADS as usize);
    let dt_bias: Vec<f32> = (0..WIDE).map(|_| 0.1 * lcg.unit()).collect();
    let a_log: Vec<f32> = (0..HEADS).map(|_| -0.5 + 0.2 * lcg.unit()).collect();
    let slab: Vec<f32> = (0..2 * STRIDE).map(|_| 0.05 * lcg.unit()).collect();
    Rows {
        mixed: gpu.up(&mixed),
        f: gpu.up(&f),
        b: gpu.up(&b),
        dt_bias: gpu.up(&dt_bias),
        a_log: gpu.up(&a_log),
        slab,
    }
}

struct Ask {
    lens: Vec<usize>,
    write_state: bool,
    mask: Option<Vec<u8>>,
    commit_len: Option<Vec<i32>>,
    begin_at: Option<Vec<i32>>,
}

/// One fire over `ask.lens` lanes of the same rows, from `slab`; answers
/// the outputs and the slab after it.
fn fire(gpu: &mut Gpu, rows: &Rows, slab: &[f32], asks: &[Ask]) -> (Vec<f32>, Vec<f32>) {
    let total: usize = LENS.iter().sum();
    let slab_at = gpu.up(slab);
    let y_at = gpu.zeros(total * WIDE * 4);
    let slots_at = gpu.up(&SLOT_OF);
    for ask in asks {
        let n: usize = ask.lens.iter().sum();
        let mut indptr: Vec<i32> = vec![0];
        for len in &ask.lens {
            indptr.push(indptr.last().unwrap() + *len as i32);
        }
        let indptr_at = gpu.up(&indptr);
        let mut table = |values: &Option<Vec<i32>>| match values {
            Some(values) => Tensor::new(gpu.up(values), values.len() as u32, 1, Dtype::I32),
            None => Tensor::ABSENT,
        };
        let commit_len = table(&ask.commit_len);
        let begin_at = table(&ask.begin_at);
        let mask = match &ask.mask {
            Some(mask) => Tensor::new(gpu.up(mask), mask.len() as u32, 1, Dtype::U8),
            None => Tensor::ABSENT,
        };
        let pool = RecurrentPool {
            slab: Tensor::new(slab_at, 2, STRIDE as u32, Dtype::F32),
            slot_ids: Tensor::new(slots_at, ask.lens.len() as u32, 1, Dtype::I32),
            slot_stride_elems: STRIDE as i64,
            conv_slab: Tensor::ABSENT,
            conv_stride: 0,
            write_state: ask.write_state,
            write_state_mask: mask,
            commit_len,
            begin_at,
            fused_decay: false,
            snapshots: kernels_cuda::Snapshots::NONE,
        };
        let mut y = Tensor::new(y_at, n as u32, WIDE as u32, Dtype::F32);
        ssm::kda_chunked(
            &gpu.ctx(),
            RaggedTensor {
                data: Tensor::new(rows.mixed, n as u32, 3 * WIDE as u32, Dtype::Bf16),
                indptr: Tensor::new(indptr_at, indptr.len() as u32, 1, Dtype::I32),
            },
            Tensor::new(rows.f, n as u32, WIDE as u32, Dtype::Bf16),
            Tensor::new(rows.b, n as u32, HEADS, Dtype::Bf16),
            Tensor::new(rows.dt_bias, 1, WIDE as u32, Dtype::F32),
            Tensor::new(rows.a_log, 1, HEADS, Dtype::F32),
            &pool,
            HEADS,
            HEAD_DIM,
            1e-6,
            0.0,
            &mut y,
        )
        .expect("the KDA scan fires");
    }
    gpu.sync();
    (gpu.down(y_at, total * WIDE), gpu.down(slab_at, 2 * STRIDE))
}

fn whole(write_state: bool) -> Ask {
    Ask {
        lens: LENS.to_vec(),
        write_state,
        mask: None,
        commit_len: None,
        begin_at: None,
    }
}

fn slot(slab: &[f32], at: i32) -> &[f32] {
    &slab[at as usize * STRIDE..(at as usize + 1) * STRIDE]
}

/// The KDA scan walks its state in place, so a fire that folds the first
/// `c` tokens of a lane and replays the rest from that fold — the buffered
/// window's head and tail — lands what one unbroken fold lands, and leaves
/// each slot at the fold of exactly its committed prefix; a fire asked not to
/// fold, or a lane its predicate names, leaves its slot untouched.
#[test]
fn the_chunked_kda_folds_its_commit_and_replays_its_tail() {
    let mut gpu = Gpu::open();
    let rows = rows(&mut gpu);
    let seed = rows.slab.clone();
    let (y_whole, slab_whole) = fire(&mut gpu, &rows, &seed, &[whole(true)]);

    let commit = 4usize;
    let (y_split, slab_split) = fire(
        &mut gpu,
        &rows,
        &seed,
        &[
            Ask {
                lens: LENS.to_vec(),
                write_state: true,
                mask: None,
                commit_len: Some(vec![commit as i32, LENS[1] as i32]),
                begin_at: None,
            },
            Ask {
                lens: LENS.to_vec(),
                write_state: false,
                mask: None,
                commit_len: None,
                begin_at: Some(vec![commit as i32, LENS[1] as i32]),
            },
        ],
    );
    assert_eq!(
        y_whole, y_split,
        "the head's fold and the tail's replay land what the unbroken fold lands"
    );
    let (_, slab_prefix) = fire(
        &mut gpu,
        &rows,
        &seed,
        &[Ask {
            lens: vec![commit],
            write_state: true,
            mask: None,
            commit_len: None,
            begin_at: None,
        }],
    );
    assert_eq!(
        slot(&slab_split, SLOT_OF[0]),
        slot(&slab_prefix, SLOT_OF[0]),
        "lane 0's slot is the fold of its committed {commit} tokens alone"
    );
    assert_eq!(
        slot(&slab_split, SLOT_OF[1]),
        slot(&slab_whole, SLOT_OF[1]),
        "lane 1 commits its whole window"
    );

    let (y_unfolded, slab_unfolded) = fire(&mut gpu, &rows, &seed, &[whole(false)]);
    assert_eq!(y_unfolded, y_whole, "an unfolded fire lands the same rows");
    assert_eq!(
        slab_unfolded, seed,
        "an unfolded fire leaves every slot as it was"
    );

    let (y_masked, slab_masked) = fire(
        &mut gpu,
        &rows,
        &seed,
        &[Ask {
            lens: LENS.to_vec(),
            write_state: true,
            mask: Some(vec![0, 1]),
            commit_len: None,
            begin_at: None,
        }],
    );
    assert_eq!(y_masked, y_whole, "a predicated fire lands the same rows");
    assert_eq!(
        slot(&slab_masked, SLOT_OF[0]),
        slot(&seed, SLOT_OF[0]),
        "the lane the predicate refuses keeps its slot"
    );
    assert_eq!(
        slot(&slab_masked, SLOT_OF[1]),
        slot(&slab_whole, SLOT_OF[1]),
        "the lane the predicate names folds"
    );
}

/// B200's tensor-core KDA (128-wide heads, sixteen-token chunks) lands the
/// serial walk's rows and slots: lanes crossing and falling short of a
/// chunk, a lane that commits a prefix, one that replays from a begun row.
#[test]
fn the_tensor_core_kda_lands_the_serial_walk() {
    const H: u32 = 2;
    const D: u32 = 128;
    let wide = (H * D) as usize;
    let stride = (H * D * D) as usize;
    let lens = [37usize, 21, 16];
    let n: usize = lens.iter().sum();
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(0x4c);
    let mixed = gpu.up(&lcg.row(n * 3 * wide).0);
    let f = gpu.up(&lcg.row(n * wide).0);
    let b = gpu.up(&lcg.row(n * H as usize).0);
    let dt_bias: Vec<f32> = (0..wide).map(|_| 0.1 * lcg.unit()).collect();
    let a_log: Vec<f32> = (0..H).map(|_| -0.5 + 0.2 * lcg.unit()).collect();
    let (dt_bias, a_log) = (gpu.up(&dt_bias), gpu.up(&a_log));
    let seed: Vec<f32> = (0..lens.len() * stride)
        .map(|_| 0.05 * lcg.unit())
        .collect();
    let mut indptr = vec![0i32];
    for len in lens {
        indptr.push(indptr.last().unwrap() + len as i32);
    }
    let indptr_at = gpu.up(&indptr);
    let slots_at = gpu.up(&[2i32, 0, 1]);
    let commit_at = gpu.up(&[37i32, 9, 16]);
    let begin_at = gpu.up(&[0i32, 0, 5]);
    let mut run = |chunked: bool, floor: f32| {
        let slab_at = gpu.up(&seed);
        let y_at = gpu.zeros(n * wide * 4);
        let pool = RecurrentPool {
            slab: Tensor::new(slab_at, lens.len() as u32, stride as u32, Dtype::F32),
            slot_ids: Tensor::new(slots_at, lens.len() as u32, 1, Dtype::I32),
            slot_stride_elems: stride as i64,
            conv_slab: Tensor::ABSENT,
            conv_stride: 0,
            write_state: true,
            write_state_mask: Tensor::ABSENT,
            commit_len: Tensor::new(commit_at, lens.len() as u32, 1, Dtype::I32),
            begin_at: Tensor::new(begin_at, lens.len() as u32, 1, Dtype::I32),
            fused_decay: false,
            snapshots: kernels_cuda::Snapshots::NONE,
        };
        let tuning = crate::common::attn_ref::tuned(|t| t.ssm_kda_chunked = chunked);
        let mut y = Tensor::new(y_at, n as u32, wide as u32, Dtype::F32);
        ssm::kda_chunked(
            &gpu.ctx().with_tuning(tuning),
            RaggedTensor {
                data: Tensor::new(mixed, n as u32, 3 * wide as u32, Dtype::Bf16),
                indptr: Tensor::new(indptr_at, indptr.len() as u32, 1, Dtype::I32),
            },
            Tensor::new(f, n as u32, wide as u32, Dtype::Bf16),
            Tensor::new(b, n as u32, H, Dtype::Bf16),
            Tensor::new(dt_bias, 1, wide as u32, Dtype::F32),
            Tensor::new(a_log, 1, H, Dtype::F32),
            &pool,
            H,
            D,
            1e-6,
            floor,
            &mut y,
        )
        .expect("the KDA scan fires");
        gpu.sync();
        let y: Vec<f32> = gpu.down(y_at, n * wide);
        let slab: Vec<f32> = gpu.down(slab_at, lens.len() * stride);
        (y, slab)
    };
    let near = |got: &[f32], want: &[f32], what: &str| {
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(want)
            .fold(0f32, |m, (g, w)| m.max((g - w).abs()));
        assert!(worst <= 2e-2 * scale, "{what}: off by {worst} of {scale}");
    };
    // Unbounded gates, and gates floored by a sigmoid.
    for floor in [0.0, -5.0] {
        let (y_walk, slab_walk) = run(false, floor);
        let (y, slab) = run(true, floor);
        near(&y, &y_walk, "the rows");
        near(&slab, &slab_walk, "the slots");
    }
}

/// A decode's step (a token a lane, its heads' value rows split over blocks)
/// lands the serial walk's rows and slots, folding and not.
#[test]
fn the_kda_decode_step_lands_the_serial_walk() {
    const H: u32 = 2;
    const D: u32 = 128;
    let wide = (H * D) as usize;
    let stride = (H * D * D) as usize;
    let n = 3usize;
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(0x4d);
    let mixed = gpu.up(&lcg.row(n * 3 * wide).0);
    let f = gpu.up(&lcg.row(n * wide).0);
    let b = gpu.up(&lcg.row(n * H as usize).0);
    let dt_bias: Vec<f32> = (0..wide).map(|_| 0.1 * lcg.unit()).collect();
    let a_log: Vec<f32> = (0..H).map(|_| -0.5 + 0.2 * lcg.unit()).collect();
    let (dt_bias, a_log) = (gpu.up(&dt_bias), gpu.up(&a_log));
    let seed: Vec<f32> = (0..n * stride).map(|_| 0.05 * lcg.unit()).collect();
    let indptr_at = gpu.up(&[0i32, 1, 2, 3]);
    let slots_at = gpu.up(&[2i32, 0, 1]);
    let mut run = |step: bool, write_state: bool| {
        let slab_at = gpu.up(&seed);
        let y_at = gpu.zeros(n * wide * 4);
        let pool = RecurrentPool {
            slab: Tensor::new(slab_at, n as u32, stride as u32, Dtype::F32),
            slot_ids: Tensor::new(slots_at, n as u32, 1, Dtype::I32),
            slot_stride_elems: stride as i64,
            conv_slab: Tensor::ABSENT,
            conv_stride: 0,
            write_state,
            write_state_mask: Tensor::ABSENT,
            commit_len: Tensor::ABSENT,
            begin_at: Tensor::ABSENT,
            fused_decay: false,
            snapshots: kernels_cuda::Snapshots::NONE,
        };
        let mixed_t = Tensor::new(mixed, n as u32, 3 * wide as u32, Dtype::Bf16);
        let (f_t, b_t) = (
            Tensor::new(f, n as u32, wide as u32, Dtype::Bf16),
            Tensor::new(b, n as u32, H, Dtype::Bf16),
        );
        let (dt_t, a_t) = (
            Tensor::new(dt_bias, 1, wide as u32, Dtype::F32),
            Tensor::new(a_log, 1, H, Dtype::F32),
        );
        let mut y = Tensor::new(y_at, n as u32, wide as u32, Dtype::F32);
        let tuning = crate::common::attn_ref::tuned(|t| t.ssm_kda_chunked = false);
        let ctx = gpu.ctx().with_tuning(tuning);
        if step {
            ssm::kda_step(
                &ctx, mixed_t, f_t, b_t, dt_t, a_t, &pool, H, D, 1e-6, 0.0, &mut y,
            )
            .expect("the KDA step fires");
        } else {
            let data = RaggedTensor {
                data: mixed_t,
                indptr: Tensor::new(indptr_at, 4, 1, Dtype::I32),
            };
            ssm::kda_chunked(
                &ctx, data, f_t, b_t, dt_t, a_t, &pool, H, D, 1e-6, 0.0, &mut y,
            )
            .expect("the KDA scan fires");
        }
        gpu.sync();
        let y: Vec<f32> = gpu.down(y_at, n * wide);
        let slab: Vec<f32> = gpu.down(slab_at, n * stride);
        (y, slab)
    };
    let near = |got: &[f32], want: &[f32], what: &str| {
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(want)
            .fold(0f32, |m, (g, w)| m.max((g - w).abs()));
        assert!(
            worst <= 1e-4 * scale.max(1e-6),
            "{what}: off by {worst} of {scale}"
        );
    };
    for write_state in [true, false] {
        let (y_walk, slab_walk) = run(false, write_state);
        let (y, slab) = run(true, write_state);
        near(&y, &y_walk, "the rows");
        near(&slab, &slab_walk, "the slots");
    }
}
