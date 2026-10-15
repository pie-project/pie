#![cfg(feature = "cuda")]

use crate::common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::attn::ssm;
use kernels_cuda::tensor::{RecurrentPool, Tensor};

struct Geometry {
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
}

struct Window {
    bucket: u32,
    live: u32,
    base: u32,
}

/// How the fire treats the recurrent boundary: folded into the state, left
/// buffered for a later commit, or folded on the lanes a predicate names.
#[derive(Clone, Copy)]
enum Boundary {
    Folded,
    Buffered,
    Predicated([u8; 3]),
}

/// Elements past each slot's state bank; eight keeps the fused arm's
/// vector stride, anything else sends a 128-wide head to the staged arm.
const PAD: usize = 8;

#[allow(clippy::too_many_lines)]
fn check(geo: Geometry, window: Option<Window>) {
    check_at(geo, window, Boundary::Folded, PAD);
}

#[allow(clippy::too_many_lines)]
fn check_at(geo: Geometry, window: Option<Window>, boundary: Boundary, pad: usize) {
    let Geometry {
        k_heads,
        v_heads,
        k_dim,
        v_dim,
    } = geo;
    let (rows, live, base) = window
        .as_ref()
        .map_or((3, 3, 0), |w| (w.bucket, w.live, w.base));
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
    let stride = bank + pad;

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
    let mask_at = match boundary {
        Boundary::Predicated(lanes) => gpu.up(&lanes),
        Boundary::Folded | Boundary::Buffered => 0,
    };
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
        write_state: !matches!(boundary, Boundary::Buffered),
        write_state_mask: match boundary {
            Boundary::Predicated(_) => Tensor::new(mask_at, rows, 1, Dtype::U8),
            Boundary::Folded | Boundary::Buffered => Tensor::ABSENT,
        },
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
        let folds = match boundary {
            Boundary::Folded => true,
            Boundary::Buffered => false,
            Boundary::Predicated(lanes) => lanes[r] != 0,
        };
        for h in 0..vh {
            let h_k = h / (vh / kh);
            let q = &row[h_k * kd..(h_k + 1) * kd];
            let k = &row[kh * kd + h_k * kd..kh * kd + (h_k + 1) * kd];
            let v = &row[2 * kh * kd + h * vd..2 * kh * kd + (h + 1) * vd];
            let q_inv = (q.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip() * q_scale;
            let k_inv = (k.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip();
            let g = gates[plane * 2 * vh + h].exp();
            let beta = gates[plane * 2 * vh + vh + h];
            let state =
                &mut want_slab[slot * stride + h * kd * vd..slot * stride + (h + 1) * kd * vd];
            for j in 0..vd {
                let kv_mem: f32 = (0..kd).map(|i| state[i * vd + j] * g * k[i] * k_inv).sum();
                let delta = (v[j] - kv_mem) * beta;
                let mut out = 0f32;
                for i in 0..kd {
                    let sn = round(state[i * vd + j] * g) + k[i] * k_inv * delta;
                    out += sn * q[i] * q_inv;
                    if folds {
                        state[i * vd + j] = round(sn);
                    }
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
fn the_fused_gdn_step_answers_what_the_recurrence_says_every_case() {
    the_fused_step_answers_at_a_128_wide_head_with_a_gqa_fan();
    the_fused_step_retires_a_buckets_padded_rows_and_reads_the_planes_where_the_window_says();
    the_fused_step_answers_at_a_64_wide_head();
    the_fused_step_answers_at_an_uneven_head();
    every_step_arm_leaves_the_state_when_the_fire_buffers_its_rows();
    every_step_arm_folds_the_lanes_the_predicate_names();
}

/// A speculative window buffers its rows unfolded: each arm answers what the
/// recurrence says and leaves the state where it found it. The fused arm at
/// a vector stride, the staged arm at a 128-wide head off it, and the
/// per-value arm at a head the fused tile does not fit.
fn every_step_arm_leaves_the_state_when_the_fire_buffers_its_rows() {
    for (geo, pad) in step_arms() {
        check_at(geo, None, Boundary::Buffered, pad);
    }
}

fn every_step_arm_folds_the_lanes_the_predicate_names() {
    for (geo, pad) in step_arms() {
        check_at(geo, None, Boundary::Predicated([1, 0, 1]), pad);
    }
}

fn step_arms() -> [(Geometry, usize); 3] {
    [
        (
            Geometry {
                k_heads: 2,
                v_heads: 4,
                k_dim: 128,
                v_dim: 128,
            },
            PAD,
        ),
        (
            Geometry {
                k_heads: 1,
                v_heads: 2,
                k_dim: 128,
                v_dim: 128,
            },
            4,
        ),
        (
            Geometry {
                k_heads: 1,
                v_heads: 2,
                k_dim: 40,
                v_dim: 40,
            },
            PAD,
        ),
    ]
}

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

/// Qwen3.8-27B's linear-attention head: sixteen key heads fanned to 48 value
/// heads, 128 wide each.
const QWEN38_27B: Geometry = Geometry {
    k_heads: 16,
    v_heads: 48,
    k_dim: 128,
    v_dim: 128,
};

/// vLLM 0.31's packed decode kernel (FLA's Triton `fused_recurrent`, on the
/// f32 state Qwen3.8-27B's config asks for) at the same head, rows and
/// timing, in microseconds: what the table's last column reads.
const VLLM_US: [(usize, f64); 4] = [(1, 11.9), (8, 15.5), (64, 70.7), (256, 253.2)];

/// The fused step's time per layer at Qwen3.8-27B's head, beside the state
/// bytes it moves. Each launch of a timed batch walks its own slots, so a
/// batch's state is not in L2 when a launch reads it, as a decode step's
/// layers' states are not. Run it alone on an idle GPU:
///
///     cargo test -p kernels-cuda --features cuda --release --test kernels_cuda \
///         the_fused_gdn_step_reports -- --ignored --nocapture
#[test]
#[ignore = "a measurement: run alone, on an idle GPU"]
fn the_fused_gdn_step_reports_its_rate_at_a_qwen38_27b_head() {
    use crate::the_tcgen05_gemm_lands_and_paces_what_cublaslt_does::{FLUSH, Timer};
    use kernels_cuda::cudarc::runtime::sys as rt;

    let Geometry {
        k_heads,
        v_heads,
        k_dim,
        v_dim,
    } = QWEN38_27B;
    let (kh, vh, kd, vd) = (
        k_heads as usize,
        v_heads as usize,
        k_dim as usize,
        v_dim as usize,
    );
    let conv_dim = 2 * kh * kd + vh * vd;
    let bank = vh * kd * vd;
    const COLD: usize = 768 << 20;
    let mut gpu = Gpu::open();
    let stream: rt::cudaStream_t = gpu.ctx().stream().cast();
    let timer = Timer {
        flush: gpu.zeros(FLUSH),
        stream,
    };
    println!("| rows | us / layer | state GB/s | vLLM us |");
    println!("|---:|---:|---:|---:|");
    for rows in [1usize, 8, 64, 256] {
        let sets = (COLD / (rows * bank * 2)).clamp(1, 16);
        let slots = rows * sets;
        let mut lcg = Lcg::seeded(0x51);
        let (qkv_raw, _) = lcg.row(rows * conv_dim);
        let slab_raw: Vec<u16> = (0..slots * bank)
            .map(|i| to_bf16(((i % 977) as f32 - 488.0) * 1e-4))
            .collect();
        let mut gates = vec![0f32; rows * 2 * vh];
        for r in 0..rows {
            for h in 0..vh {
                gates[r * 2 * vh + h] = -0.05 - 0.5 * lcg.unit().abs();
                gates[r * 2 * vh + vh + h] = 0.5 + 0.4 * lcg.unit();
            }
        }
        let qkv_at = gpu.up(&qkv_raw);
        let slab_at = gpu.up(&slab_raw);
        let gates_at = gpu.up(&gates);
        let y_at = gpu.zeros(rows * vh * vd * 4);
        let id_sets: Vec<u64> = (0..sets)
            .map(|set| {
                let ids: Vec<i32> = (0..rows).map(|r| (set * rows + r) as i32).collect();
                gpu.up(&ids)
            })
            .collect();
        let ctx = gpu.ctx();
        let fire = |set: usize| {
            let pool = RecurrentPool {
                slab: Tensor::new(slab_at, slots as u32, bank as u32, Dtype::Bf16),
                slot_ids: Tensor::new(id_sets[set % sets], rows as u32, 1, Dtype::I32),
                slot_stride_elems: bank as i64,
                conv_slab: Tensor::ABSENT,
                conv_stride: 0,
                write_state: true,
                write_state_mask: Tensor::ABSENT,
                commit_len: Tensor::ABSENT,
                begin_at: Tensor::ABSENT,
                fused_decay: false,
            };
            let mut y = Tensor::new(y_at, rows as u32, (vh * vd) as u32, Dtype::F32);
            ssm::gated_delta(
                &ctx,
                Tensor::new(qkv_at, rows as u32, conv_dim as u32, Dtype::Bf16),
                Tensor::ABSENT,
                Tensor::new(gates_at, rows as u32, (2 * vh) as u32, Dtype::F32),
                &pool,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                &mut y,
            )
            .expect("the step fires");
        };
        let us = timer.median(fire);
        let bytes = (2 * rows * bank * 2) as f64;
        let vllm = VLLM_US
            .iter()
            .find(|(r, _)| *r == rows)
            .map_or(0.0, |(_, us)| *us);
        println!(
            "| {rows} | {us:.1} | {:.0} | {vllm:.1} |",
            bytes / us * 1e-3
        );
    }
}
