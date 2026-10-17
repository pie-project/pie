#![cfg(feature = "cuda")]

//! KDA's tuned routes land what its plain walk lands, every output row and
//! every slot's state: the decode step reading its packed rows itself
//! (`ssm_kda_step_fused`) and the prefill on the tensor cores
//! (`ssm_kda_chunked`), across chunks, a lane shorter than one, a commit
//! folding a prefix and a predicate folding nothing, a gate floor or none.

use crate::common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::attn::ssm;
use kernels_cuda::tensor::{RaggedTensor, RecurrentPool, Tensor};
use kernels_cuda::tuning::Tuning;

const D: usize = 128;

/// One fire over lanes `lens` (`step`: one row each, through `kda_step`)
/// at `heads`; answers the outputs and the slab.
fn fire(
    lens: &[usize],
    heads: usize,
    commit: &[i32],
    mask: &[u8],
    floor: f32,
    step: bool,
    tuned: bool,
) -> (Vec<f32>, Vec<f32>) {
    let (rows, lanes) = (lens.iter().sum::<usize>(), lens.len());
    let (wide, stride) = (heads * D, heads * D * D);
    let mut lcg = Lcg::seeded(0x4da ^ rows as u64);
    let (mixed, _) = lcg.row(rows * 3 * wide);
    let (f, _) = lcg.row(rows * wide);
    let (b, _) = lcg.row(rows * heads);
    let dt_bias: Vec<f32> = (0..wide).map(|_| 0.1 * lcg.unit()).collect();
    let a_log: Vec<f32> = (0..heads).map(|_| -0.5 + 0.2 * lcg.unit()).collect();
    let slab: Vec<f32> = (0..lanes * stride).map(|_| 0.05 * lcg.unit()).collect();
    let mut gpu = Gpu::open();
    let slots: Vec<i32> = (0..lanes as i32).rev().collect();
    let opt = |gpu: &mut Gpu, v: &[u8], dt| {
        if v.is_empty() {
            Tensor::ABSENT
        } else {
            Tensor::new(gpu.up(v), v.len() as u32, 1, dt)
        }
    };
    let commit_bytes: Vec<u8> = commit.iter().flat_map(|c| c.to_le_bytes()).collect();
    let (commit_len, write_state_mask) = (
        opt(&mut gpu, &commit_bytes, Dtype::I32),
        opt(&mut gpu, mask, Dtype::U8),
    );
    let commit_len = Tensor {
        rows: commit.len() as u32,
        ..commit_len
    };
    let slab_at = gpu.up(&slab);
    let pool = RecurrentPool {
        slab: Tensor::new(slab_at, lanes as u32, stride as u32, Dtype::F32),
        slot_ids: Tensor::new(gpu.up(&slots), lanes as u32, 1, Dtype::I32),
        slot_stride_elems: stride as i64,
        conv_slab: Tensor::ABSENT,
        conv_stride: 0,
        write_state: true,
        write_state_mask,
        commit_len,
        begin_at: Tensor::ABSENT,
        fused_decay: false,
    };
    let (r, w, h) = (rows as u32, wide as u32, heads as u32);
    let y_at = gpu.zeros(rows * wide * 4);
    let mut y = Tensor::new(y_at, r, w, Dtype::F32);
    let mixed = Tensor::new(gpu.up(&mixed), r, 3 * w, Dtype::Bf16);
    let (f, b) = (
        Tensor::new(gpu.up(&f), r, w, Dtype::Bf16),
        Tensor::new(gpu.up(&b), r, h, Dtype::Bf16),
    );
    let (dt, a) = (
        Tensor::new(gpu.up(&dt_bias), 1, w, Dtype::F32),
        Tensor::new(gpu.up(&a_log), 1, h, Dtype::F32),
    );
    let ctx = gpu.ctx().with_tuning(Tuning {
        ssm_kda_step_fused: tuned,
        ssm_kda_chunked: tuned,
        ..Tuning::DEFAULT
    });
    if step {
        ssm::kda_step(
            &ctx, mixed, f, b, dt, a, &pool, h, D as u32, 1e-6, floor, &mut y,
        )
    } else {
        let mut indptr = vec![0i32];
        lens.iter()
            .for_each(|l| indptr.push(indptr[indptr.len() - 1] + *l as i32));
        let indptr = Tensor::new(gpu.up(&indptr), lanes as u32 + 1, 1, Dtype::I32);
        ssm::kda_chunked(
            &ctx,
            RaggedTensor {
                data: mixed,
                indptr,
            },
            f,
            b,
            dt,
            a,
            &pool,
            h,
            D as u32,
            1e-6,
            floor,
            &mut y,
        )
    }
    .expect("the KDA fire");
    gpu.sync();
    (
        gpu.down(y_at, rows * wide),
        gpu.down(slab_at, lanes * stride),
    )
}

#[test]
fn the_kda_routes_land_what_the_row_walk_does() {
    let ones = |n| vec![1usize; n];
    let cases: [(Vec<usize>, usize, &[i32], &[u8], f32, bool, f32); 6] = [
        (ones(1), 96, &[], &[], -5.0, true, 1e-5),
        (ones(8), 4, &[], &[1, 0, 1, 1, 0, 1, 1, 1], -5.0, true, 1e-5),
        (ones(3), 2, &[], &[], 0.0, true, 1e-5),
        (vec![150, 70, 5], 2, &[], &[], -5.0, false, 2e-2),
        (vec![64, 129], 2, &[40, 129], &[], -5.0, false, 2e-2),
        (vec![200], 2, &[], &[0], -5.0, false, 2e-2),
    ];
    for (lens, heads, commit, mask, floor, step, tol) in cases {
        let plain = fire(&lens, heads, commit, mask, floor, step, false);
        let tuned = fire(&lens, heads, commit, mask, floor, step, true);
        for (what, got, want) in [
            ("outputs", &tuned.0, &plain.0),
            ("states", &tuned.1, &plain.1),
        ] {
            let scale = want.iter().fold(1e-3f32, |m, w| m.max(w.abs()));
            let worst = got
                .iter()
                .zip(want.iter())
                .fold(0f32, |m, (g, w)| m.max((g - w).abs()));
            assert!(
                worst <= tol * scale,
                "{lens:?} x {heads} heads, step {step}: the {what} are {worst} apart"
            );
        }
    }
}
