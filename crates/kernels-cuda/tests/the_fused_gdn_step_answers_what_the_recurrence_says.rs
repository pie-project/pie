//! The one-launch gated-delta decode step lands the same output row and the
//! same folded state as the recurrence written out on the host: q/k L2-normed
//! per key head, the decayed state rounded to bf16 before the update, the
//! value heads fanned over the key heads, and each lane's slot its own.

#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::attn::ssm;
use kernels_cuda::tensor::{RecurrentPool, Tensor};

struct Geometry {
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
}

/// A staged window: the grid stands `bucket` rows tall, `live` of them
/// count, and the row planes start `base` rows in.
struct Window {
    bucket: u32,
    live: u32,
    base: u32,
}

#[allow(clippy::too_many_lines)]
fn check(geo: Geometry, window: Option<Window>) {
    let Geometry {
        k_heads,
        v_heads,
        k_dim,
        v_dim,
    } = geo;
    let (rows, live, base) = window.as_ref().map_or((3, 3, 0), |w| (w.bucket, w.live, w.base));
    let planes = base + rows;
    let slots: u32 = 4;
    let slot_of: [i32; 3] = [2, 0, 3];
    let (kh, vh, kd, vd) = (
        k_heads as usize,
        v_heads as usize,
        k_dim as usize,
        v_dim as usize,
    );
    let conv_dim = 2 * kh * kd + vh * vd;
    let bank = vh * kd * vd;
    let stride = bank + 8;

    let mut lcg = Lcg::seeded(0x6d6e);
    let (qkv_raw, qkv) = lcg.row(planes as usize * conv_dim);
    let (slab_raw, slab) = lcg.row(slots as usize * stride);
    let mut gates = vec![0f32; planes as usize * 2 * vh];
    for r in 0..planes as usize {
        for h in 0..vh {
            gates[r * 2 * vh + h] = -0.05 - 0.5 * lcg.unit().abs();
            gates[r * 2 * vh + vh + h] = 0.5 + 0.4 * lcg.unit();
        }
    }

    let mut gpu = Gpu::open();
    let qkv_at = gpu.up(&qkv_raw);
    let slab_at = gpu.up(&slab_raw);
    let gates_at = gpu.up(&gates);
    let slots_at = gpu.up(&slot_of);
    let y_at = gpu.zeros(planes as usize * vh * vd * 4);
    let ctx = gpu.ctx();
    if window.is_some() {
        let win_at = gpu.up(&[live, base, 0u32, 0u32]);
        ctx.arm_stage(win_at);
    }

    let pool = RecurrentPool {
        slab: Tensor::new(slab_at, slots, stride as u32, Dtype::Bf16),
        slot_ids: Tensor::new(slots_at, rows, 1, Dtype::I32),
        slot_stride_elems: stride as i64,
        conv_slab: Tensor::ABSENT,
        conv_stride: 0,
        write_state: true,
        write_state_mask: Tensor::ABSENT,
        commit_len: Tensor::ABSENT,
        begin_at: Tensor::ABSENT,
        fused_decay: false,
    };
    let mut y = Tensor::new(y_at, rows, (vh * vd) as u32, Dtype::F32);
    ssm::gated_delta(
        &ctx,
        Tensor::new(qkv_at, rows, conv_dim as u32, Dtype::Bf16),
        Tensor::ABSENT,
        Tensor::new(gates_at, rows, (2 * vh) as u32, Dtype::F32),
        &pool,
        k_heads,
        v_heads,
        k_dim,
        v_dim,
        &mut y,
    )
    .expect("the step fires");
    gpu.sync();
    let got_y: Vec<f32> = gpu.down(y_at, planes as usize * vh * vd);
    let got_slab: Vec<u16> = gpu.down(slab_at, slots as usize * stride);

    let round = |x: f32| from_bf16(to_bf16(x));
    let q_scale = (k_dim as f32).sqrt().recip();
    let mut want_slab = slab.clone();
    let mut want_y = vec![0f32; planes as usize * vh * vd];
    for r in 0..live as usize {
        let plane = base as usize + r;
        let row = &qkv[plane * conv_dim..(plane + 1) * conv_dim];
        let slot = slot_of[r] as usize;
        for h in 0..vh {
            let h_k = h / (vh / kh);
            let q = &row[h_k * kd..(h_k + 1) * kd];
            let k = &row[kh * kd + h_k * kd..kh * kd + (h_k + 1) * kd];
            let v = &row[2 * kh * kd + h * vd..2 * kh * kd + (h + 1) * vd];
            let q_inv = (q.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip() * q_scale;
            let k_inv = (k.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip();
            let g = gates[plane * 2 * vh + h].exp();
            let beta = gates[plane * 2 * vh + vh + h];
            let state = &mut want_slab[slot * stride + h * kd * vd..slot * stride + (h + 1) * kd * vd];
            for j in 0..vd {
                let kv_mem: f32 = (0..kd).map(|i| state[i * vd + j] * g * k[i] * k_inv).sum();
                let delta = (v[j] - kv_mem) * beta;
                let mut out = 0f32;
                for i in 0..kd {
                    let sn = round(state[i * vd + j] * g) + k[i] * k_inv * delta;
                    out += sn * q[i] * q_inv;
                    state[i * vd + j] = round(sn);
                }
                want_y[(plane * vh + h) * vd + j] = out;
            }
        }
    }
    for (at, (&got, &want)) in got_y.iter().zip(&want_y).enumerate() {
        assert!(
            close(got, want),
            "output element {at}: the step answered {got} and the recurrence says {want}"
        );
    }
    for (at, (&got, &want)) in got_slab.iter().zip(&want_slab).enumerate() {
        let got = from_bf16(got);
        assert!(
            close(got, want),
            "state element {at}: the step left {got} and the recurrence says {want}"
        );
    }
}

#[test]
fn the_fused_step_answers_at_a_128_wide_head_with_a_gqa_fan() {
    check(
        Geometry {
            k_heads: 2,
            v_heads: 4,
            k_dim: 128,
            v_dim: 128,
        },
        None,
    );
}

#[test]
fn the_fused_step_retires_a_buckets_padded_rows_and_reads_the_planes_where_the_window_says() {
    check(
        Geometry {
            k_heads: 2,
            v_heads: 2,
            k_dim: 128,
            v_dim: 128,
        },
        Some(Window {
            bucket: 3,
            live: 2,
            base: 1,
        }),
    );
}

#[test]
fn the_fused_step_answers_at_a_64_wide_head() {
    check(
        Geometry {
            k_heads: 1,
            v_heads: 1,
            k_dim: 64,
            v_dim: 64,
        },
        None,
    );
}

#[test]
fn the_fused_step_answers_at_an_uneven_head() {
    check(
        Geometry {
            k_heads: 2,
            v_heads: 2,
            k_dim: 96,
            v_dim: 48,
        },
        None,
    );
}
