#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::attn::ssm;
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
