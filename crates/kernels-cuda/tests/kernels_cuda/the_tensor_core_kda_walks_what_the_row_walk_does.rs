#![cfg(feature = "cuda")]

//! The KDA prefill on the tensor cores (`kernel.ssm.kda_chunked`: a
//! chunk's update directions in parallel, then the state chunk by chunk)
//! lands what the row-at-a-time walk lands — every output row and the
//! folded state — across chunks, a chunk's partial tail, a lane shorter
//! than a chunk, a commit that folds a prefix and a predicate that folds
//! nothing, to its bf16 operands' rounding.

use crate::common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::attn::ssm;
use kernels_cuda::tensor::{RaggedTensor, RecurrentPool, Tensor};

const HEADS: u32 = 2;
const HEAD_DIM: u32 = 128;
const WIDE: usize = (HEADS * HEAD_DIM) as usize;
const STRIDE: usize = (HEADS * HEAD_DIM * HEAD_DIM) as usize;

struct Case {
    lens: Vec<usize>,
    commit_len: Option<Vec<i32>>,
    mask: Option<Vec<u8>>,
}

fn walk(gpu: &mut Gpu, case: &Case, chunked: bool) -> (Vec<f32>, Vec<f32>) {
    let rows: usize = case.lens.iter().sum();
    let lanes = case.lens.len();
    let mut lcg = Lcg::seeded(0x4da);
    let (mixed, _) = lcg.row(rows * 3 * WIDE);
    let (f, _) = lcg.row(rows * WIDE);
    let (b, _) = lcg.row(rows * HEADS as usize);
    let dt_bias: Vec<f32> = (0..WIDE).map(|_| 0.1 * lcg.unit()).collect();
    let a_log: Vec<f32> = (0..HEADS).map(|_| -0.5 + 0.2 * lcg.unit()).collect();
    let slab: Vec<f32> = (0..lanes * STRIDE).map(|_| 0.05 * lcg.unit()).collect();
    let slots: Vec<i32> = (0..lanes as i32).rev().collect();
    let mut indptr: Vec<i32> = vec![0];
    for len in &case.lens {
        indptr.push(indptr.last().unwrap() + *len as i32);
    }
    let slab_at = gpu.up(&slab);
    let y_at = gpu.zeros(rows * WIDE * 4);
    let commit_len = match &case.commit_len {
        Some(v) => Tensor::new(gpu.up(v), v.len() as u32, 1, Dtype::I32),
        None => Tensor::ABSENT,
    };
    let mask = match &case.mask {
        Some(v) => Tensor::new(gpu.up(v), v.len() as u32, 1, Dtype::U8),
        None => Tensor::ABSENT,
    };
    let pool = RecurrentPool {
        slab: Tensor::new(slab_at, lanes as u32, STRIDE as u32, Dtype::F32),
        slot_ids: Tensor::new(gpu.up(&slots), lanes as u32, 1, Dtype::I32),
        slot_stride_elems: STRIDE as i64,
        conv_slab: Tensor::ABSENT,
        conv_stride: 0,
        write_state: true,
        write_state_mask: mask,
        commit_len,
        begin_at: Tensor::ABSENT,
        fused_decay: false,
    };
    let mut y = Tensor::new(y_at, rows as u32, WIDE as u32, Dtype::F32);
    let ctx = gpu.ctx().with_tuning(kernels_cuda::tuning::Tuning {
        ssm_kda_chunked: chunked,
        ..kernels_cuda::tuning::Tuning::DEFAULT
    });
    ssm::kda_chunked(
        &ctx,
        RaggedTensor {
            data: Tensor::new(gpu.up(&mixed), rows as u32, 3 * WIDE as u32, Dtype::Bf16),
            indptr: Tensor::new(gpu.up(&indptr), indptr.len() as u32, 1, Dtype::I32),
        },
        Tensor::new(gpu.up(&f), rows as u32, WIDE as u32, Dtype::Bf16),
        Tensor::new(gpu.up(&b), rows as u32, HEADS, Dtype::Bf16),
        Tensor::new(gpu.up(&dt_bias), 1, WIDE as u32, Dtype::F32),
        Tensor::new(gpu.up(&a_log), 1, HEADS, Dtype::F32),
        &pool,
        HEADS,
        HEAD_DIM,
        1e-6,
        -5.0,
        &mut y,
    )
    .expect("the KDA scan fires");
    gpu.sync();
    (
        gpu.down(y_at, rows * WIDE),
        gpu.down(slab_at, lanes * STRIDE),
    )
}

fn near(what: &str, got: &[f32], want: &[f32]) {
    let scale = want.iter().map(|w| w.abs()).fold(0f32, f32::max).max(1e-3);
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    assert!(
        worst <= 2e-2 * scale,
        "{what}: the tensor-core walk is {worst} from the row walk (scale {scale})"
    );
}

#[test]
fn the_tensor_core_kda_walks_what_the_row_walk_does() {
    let mut gpu = Gpu::open();
    for case in [
        Case {
            lens: vec![150, 70, 5],
            commit_len: None,
            mask: None,
        },
        Case {
            lens: vec![64, 129],
            commit_len: Some(vec![40, 129]),
            mask: None,
        },
        Case {
            lens: vec![200],
            commit_len: None,
            mask: Some(vec![0]),
        },
    ] {
        let (y_row, s_row) = walk(&mut gpu, &case, false);
        let (y_mma, s_mma) = walk(&mut gpu, &case, true);
        near(&format!("{:?} outputs", case.lens), &y_mma, &y_row);
        near(&format!("{:?} states", case.lens), &s_mma, &s_row);
    }
}
