#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::attn::ssm;
use kernels_cuda::linear::{moe, rel_bias};
use kernels_cuda::tensor::{RecurrentPool, Tensor};

#[test]
fn the_inkling_points_answer_their_host_reference_every_case() {
    the_short_conv_adds_the_input_back_and_shifts_the_window();
    the_relative_profile_is_the_features_through_the_bank();
    the_sink_router_normalizes_the_picks_with_the_sinks();
}

fn the_short_conv_adds_the_input_back_and_shifts_the_window() {
    let (rows, channels, k) = (3u32, 64usize, 4u32);
    let slot_of: [i32; 3] = [1, 3, 0];
    let stride = (k as usize) * channels;
    let mut lcg = Lcg::seeded(0x1a);
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
    ssm::short_conv(
        &gpu.ctx(),
        Tensor::new(x_at, rows, channels as u32, Dtype::Bf16),
        Tensor::new(w_at, channels as u32, k, Dtype::Bf16),
        &pool,
        k,
        &mut y,
    )
    .expect("the short conv fires");
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
            let expect = acc + x[r * channels + c];
            let got = from_bf16(got_y[r * channels + c]);
            assert!(close(got, expect), "row {r} channel {c}: {got} against {expect}");
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

fn the_relative_profile_is_the_features_through_the_bank() {
    let (rows, heads, d_rel, extent) = (5u32, 4u32, 16u32, 96u32);
    let mut lcg = Lcg::seeded(0x2b);
    let (r_raw, r) = lcg.row((rows * heads * d_rel) as usize);
    let (p_raw, p) = lcg.row((d_rel * extent) as usize);

    let mut gpu = Gpu::open();
    let r_at = gpu.up(&r_raw);
    let p_at = gpu.up(&p_raw);
    let b_at = gpu.zeros((rows * heads * extent) as usize * 4);
    let mut bias = Tensor::new(b_at, rows, heads * extent, Dtype::F32);
    rel_bias::rel_bias(
        &gpu.ctx(),
        Tensor::new(r_at, rows, heads * d_rel, Dtype::Bf16),
        Tensor::new(p_at, d_rel, extent, Dtype::Bf16),
        heads,
        d_rel,
        extent,
        &mut bias,
    )
    .expect("the profile fires");
    gpu.sync();
    let got: Vec<f32> = gpu.down(b_at, (rows * heads * extent) as usize);
    for row in 0..rows as usize {
        for h in 0..heads as usize {
            for d in 0..extent as usize {
                let mut want = 0f32;
                for j in 0..d_rel as usize {
                    want += r[(row * heads as usize + h) * d_rel as usize + j]
                        * p[j * extent as usize + d];
                }
                let at = (row * heads as usize + h) * extent as usize + d;
                assert!(
                    close(got[at], want),
                    "row {row} head {h} distance {d}: {} against {want}",
                    got[at]
                );
            }
        }
    }
}

fn the_sink_router_normalizes_the_picks_with_the_sinks() {
    let (rows, experts, sink, top_k) = (7u32, 16u32, 2u32, 6u32);
    let width = (experts + sink) as usize;
    let fan = (top_k + sink) as usize;
    let mut lcg = Lcg::seeded(0x3c);
    let (l_raw, l) = lcg.row(rows as usize * width);
    let bias: Vec<f32> = (0..experts).map(|e| 0.25 * f32::from(u8::try_from(e % 4).unwrap()) - 0.3).collect();
    let scale: [f32; 1] = [1.375];
    let route_scale = 8.0f32;

    let mut gpu = Gpu::open();
    let l_at = gpu.up(&l_raw);
    let bias_at = gpu.up(&bias);
    let scale_at = gpu.up(&scale);
    let routes_at = gpu.zeros(rows as usize * fan * 4);
    let weights_at = gpu.zeros(rows as usize * fan * 4);
    let mut routes = Tensor::new(routes_at, rows, fan as u32, Dtype::I32);
    let mut weights = Tensor::new(weights_at, rows, fan as u32, Dtype::F32);
    moe::topk_sigmoid_sink(
        &gpu.ctx(),
        Tensor::new(l_at, rows, width as u32, Dtype::Bf16),
        Some(Tensor::new(bias_at, experts, 1, Dtype::F32)),
        Some(Tensor::new(scale_at, 1, 1, Dtype::F32)),
        experts,
        top_k,
        sink,
        route_scale,
        &mut routes,
        &mut weights,
    )
    .expect("the router fires");
    gpu.sync();
    let got_routes: Vec<i32> = gpu.down(routes_at, rows as usize * fan);
    let got_weights: Vec<f32> = gpu.down(weights_at, rows as usize * fan);

    for row in 0..rows as usize {
        let logits = &l[row * width..(row + 1) * width];
        let sigma: Vec<f32> = logits.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
        let mut order: Vec<usize> = (0..experts as usize).collect();
        order.sort_by(|&a, &b| {
            (sigma[b] + bias[b]).partial_cmp(&(sigma[a] + bias[a])).expect("finite scores")
        });
        let picks = &order[..top_k as usize];
        let mut chosen: Vec<usize> = picks.to_vec();
        chosen.extend((experts as usize)..width);
        let sum: f32 = chosen.iter().map(|&e| sigma[e]).sum();
        let got_r = &got_routes[row * fan..(row + 1) * fan];
        let got_w = &got_weights[row * fan..(row + 1) * fan];
        let mut got_picks: Vec<i32> = got_r[..top_k as usize].to_vec();
        let mut want_picks: Vec<i32> = picks.iter().map(|&e| e as i32).collect();
        got_picks.sort_unstable();
        want_picks.sort_unstable();
        assert_eq!(got_picks, want_picks, "row {row} picks");
        assert_eq!(
            &got_r[top_k as usize..],
            &[experts as i32, experts as i32 + 1],
            "row {row} sinks"
        );
        for (slot, &e) in got_r.iter().enumerate() {
            let want = sigma[e as usize] / sum * route_scale * scale[0];
            assert!(
                close(got_w[slot], want),
                "row {row} slot {slot} (expert {e}): {} against {want}",
                got_w[slot]
            );
        }
    }
}
