//! The eight-wide forms of the decode conv update and the row cut land what
//! their scalar forms land: the convolved row with the window shifted one
//! step, and the two halves of a packed row.

#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::attn::ssm;
use kernels_cuda::layout;
use kernels_cuda::tensor::{RecurrentPool, Tensor};

#[test]
fn the_conv_update_convolves_the_window_and_shifts_it() {
    let (rows, channels, k) = (3u32, 64usize, 4u32);
    let slot_of: [i32; 3] = [1, 3, 0];
    let stride = (k as usize) * channels;
    let mut lcg = Lcg::seeded(0xc0);
    let (x_raw, x) = lcg.row(rows as usize * channels);
    let (w_raw, w) = lcg.row(channels * k as usize);
    let (slab_raw, slab) = lcg.row(4 * stride);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let w_at = gpu.up(&w_raw);
    let slab_at = gpu.up(&slab_raw);
    let slots_at = gpu.up(&slot_of);
    let y_at = gpu.zeros(rows as usize * channels * 2);
    let pool = RecurrentPool {
        slab: Tensor::ABSENT,
        slot_ids: Tensor::new(slots_at, rows, 1, Dtype::I32),
        slot_stride_elems: 0,
        conv_slab: Tensor::new(slab_at, 4, stride as u32, Dtype::Bf16),
        conv_stride: stride as i64,
        write_state: true,
        write_state_mask: Tensor::ABSENT,
        commit_len: Tensor::ABSENT,
        begin_at: Tensor::ABSENT,
        fused_decay: false,
    };
    let mut y = Tensor::new(y_at, rows, channels as u32, Dtype::Bf16);
    ssm::causal_conv1d(
        &gpu.ctx(),
        Tensor::new(x_at, rows, channels as u32, Dtype::Bf16),
        Tensor::new(w_at, channels as u32, k, Dtype::Bf16),
        &pool,
        k,
        1,
        &mut y,
    )
    .expect("the conv fires");
    gpu.sync();
    let got_y: Vec<u16> = gpu.down(y_at, rows as usize * channels);
    let got_slab: Vec<u16> = gpu.down(slab_at, 4 * stride);

    let mut want = slab.clone();
    for r in 0..rows as usize {
        let state = &mut want[slot_of[r] as usize * stride..(slot_of[r] as usize + 1) * stride];
        for c in 0..channels {
            let mut acc = 0f32;
            for t in 0..k as usize {
                let xv = if t + 1 < k as usize {
                    state[(t + 1) * channels + c]
                } else {
                    x[r * channels + c]
                };
                acc += w[c * k as usize + t] * xv;
            }
            let silu = acc / (1.0 + (-acc).exp());
            let got = from_bf16(got_y[r * channels + c]);
            assert!(close(got, silu), "row {r} channel {c}: {got} against {silu}");
            for t in 0..k as usize - 1 {
                state[t * channels + c] = state[(t + 1) * channels + c];
            }
            state[(k as usize - 1) * channels + c] = from_bf16(to_bf16(x[r * channels + c]));
        }
    }
    for (at, (&got, &want)) in got_slab.iter().zip(&want).enumerate() {
        assert!(
            close(from_bf16(got), want),
            "window element {at}: {} against {want}",
            from_bf16(got)
        );
    }
}

#[test]
fn the_row_cut_lands_both_halves() {
    let (rows, left_w, right_w) = (5u32, 48usize, 80usize);
    let total = left_w + right_w;
    let mut lcg = Lcg::seeded(0x5e);
    let (x_raw, _) = lcg.row(rows as usize * total);
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let l_at = gpu.zeros(rows as usize * left_w * 2);
    let r_at = gpu.zeros(rows as usize * right_w * 2);
    let mut left = Tensor::new(l_at, rows, left_w as u32, Dtype::Bf16);
    let mut right = Tensor::new(r_at, rows, right_w as u32, Dtype::Bf16);
    layout::split_rows(
        &gpu.ctx(),
        Tensor::new(x_at, rows, total as u32, Dtype::Bf16),
        left_w as u32,
        &mut left,
        &mut right,
    )
    .expect("the cut fires");
    gpu.sync();
    let got_l: Vec<u16> = gpu.down(l_at, rows as usize * left_w);
    let got_r: Vec<u16> = gpu.down(r_at, rows as usize * right_w);
    for r in 0..rows as usize {
        assert_eq!(&got_l[r * left_w..(r + 1) * left_w], &x_raw[r * total..r * total + left_w]);
        assert_eq!(&got_r[r * right_w..(r + 1) * right_w], &x_raw[r * total + left_w..(r + 1) * total]);
    }
}
