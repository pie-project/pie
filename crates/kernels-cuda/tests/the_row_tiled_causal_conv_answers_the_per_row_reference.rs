#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::attn::ssm;
use kernels_cuda::tensor::{RaggedTensor, RecurrentPool, Tensor};

#[test]
fn the_row_tiled_causal_conv_answers_the_per_row_reference() {
    let lens: [usize; 4] = [31, 2, 131, 45];
    let (channels, k) = (200usize, 4u32);
    let rows: usize = lens.iter().sum();
    let slot_of: [i32; 4] = [3, 2, 0, 1];
    let stride = (k as usize - 1 + 1) * channels;
    let mut indptr: Vec<i32> = vec![0];
    for len in lens {
        indptr.push(indptr.last().unwrap() + len as i32);
    }
    let mut lcg = Lcg::seeded(0x7c);
    let (x_raw, x) = lcg.row(rows * channels);
    let (w_raw, w) = lcg.row(channels * k as usize);
    let (slab_raw, slab) = lcg.row(4 * stride);
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let w_at = gpu.up(&w_raw);
    let slab_at = gpu.up(&slab_raw);
    let slots_at = gpu.up(&slot_of);
    let indptr_at = gpu.up(&indptr);
    let y_at = gpu.zeros(rows * channels * 2);
    let pool = RecurrentPool {
        slab: Tensor::ABSENT,
        slot_ids: Tensor::new(slots_at, 4, 1, Dtype::I32),
        slot_stride_elems: 0,
        conv_slab: Tensor::new(slab_at, 4, stride as u32, Dtype::Bf16),
        conv_stride: stride as i64,
        write_state: true,
        write_state_mask: Tensor::ABSENT,
        commit_len: Tensor::ABSENT,
        begin_at: Tensor::ABSENT,
        fused_decay: false,
    };
    let mut y = Tensor::new(y_at, rows as u32, channels as u32, Dtype::Bf16);
    ssm::causal_conv1d_chunked(
        &gpu.ctx(),
        RaggedTensor {
            data: Tensor::new(x_at, rows as u32, channels as u32, Dtype::Bf16),
            indptr: Tensor::new(indptr_at, lens.len() as u32 + 1, 1, Dtype::I32),
        },
        Tensor::new(w_at, channels as u32, k, Dtype::Bf16),
        &pool,
        k,
        1,
        &mut y,
    )
    .expect("the chunked conv fires");
    gpu.sync();
    let got_y: Vec<u16> = gpu.down(y_at, rows * channels);
    let got_slab: Vec<u16> = gpu.down(slab_at, 4 * stride);
    let span = k as usize;
    let mut want_slab = slab.clone();
    for (lane, &len) in lens.iter().enumerate() {
        let t0 = indptr[lane] as usize;
        let slot = slot_of[lane] as usize;
        let state = &slab[slot * stride..(slot + 1) * stride];
        let x_at = |t: isize, c: usize| -> f32 {
            if t < 0 {
                state[(span as isize + t) as usize * channels + c]
            } else {
                x[(t0 + t as usize) * channels + c]
            }
        };
        for t in 0..len {
            for c in 0..channels {
                let mut acc = 0f32;
                for tap in 0..k as usize {
                    let src = t as isize - (k as isize - 1 - tap as isize);
                    acc += w[c * k as usize + tap] * x_at(src, c);
                }
                let silu = acc / (1.0 + (-acc).exp());
                let got = from_bf16(got_y[(t0 + t) * channels + c]);
                assert!(
                    close(got, silu),
                    "lane {lane} row {t} channel {c}: {got} against {silu}"
                );
            }
        }
        let new_state = &mut want_slab[slot * stride..(slot + 1) * stride];
        for s in 0..span {
            let src = len as isize - span as isize + s as isize;
            for c in 0..channels {
                new_state[s * channels + c] = from_bf16(to_bf16(x_at(src, c)));
            }
        }
    }
    for (at, (&got, &want)) in got_slab.iter().zip(&want_slab).enumerate() {
        assert!(
            close(from_bf16(got), want),
            "state element {at}: {} against {want}",
            from_bf16(got)
        );
    }
}
