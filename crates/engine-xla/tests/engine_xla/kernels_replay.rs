//! DeepSeek-V4.1's bounded-replay floors, a lane's last key and the
//! DSpark window with its sinks, against host references (engine-cuda's
//! `attn/kv_window.cu` and `attn/compressed/decode.cu`).

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use engine_xla::handle::{KvPool, RaggedTensor, RecurrentPool, Tensor};
use engine_xla::lower::attn;

const PS: usize = 4;
/// Request 0 holds pages 3, 1, 4; request 1 pages 0, 2.
const PAGES: [&[u32]; 2] = [&[3, 1, 4], &[0, 2]];
const CELLS: usize = 5 * PS;

fn kv_pool(b: &mut Bench, keys: Tensor) -> KvPool {
    let indices: Vec<i32> = PAGES
        .iter()
        .flat_map(|p| p.iter().map(|&x| x as i32))
        .collect();
    KvPool {
        keys,
        values: keys,
        page_indices: b.i32(indices.len() as u32, 1, &indices),
        page_indptr: b.i32(3, 1, &[0, 3, 5]),
        page_size: PS as i32,
        max_pages: 3,
        seq_stride: u64::from(keys.width),
        head_stride: u64::from(keys.width),
    }
}

#[test]
fn a_replay_floor_marks_its_lanes_or_reads_their_state() {
    // Lane 0 rows 0..3 (slot 2), lane 1 empty, lane 2 rows 3..7 (slot 0),
    // row 7 past every lane.
    let positions = [10i32, 11, 12, 4, 5, 6, 7, 99];
    let slots = [2i32, 2, 2, 0, 0, 0, 0, 4];
    let state0 = [7i32, 0, 9, 0, 0];
    for mark in [true, false] {
        let mut b = Bench::new();
        let data = b.i32(8, 1, &positions);
        let indptr = b.i32(4, 1, &[0, 3, 3, 7]);
        let state = b.i32(5, 1, &state0);
        let slot_t = b.i32(8, 1, &slots);
        let floor = b.zeros(Dtype::I32, 8, 1);
        let pool = RecurrentPool {
            state,
            slots: slot_t,
            conv_state: state,
            new_conv_state: state,
        };
        if !b
            .run(|ctx| attn::replay_floor(ctx, RaggedTensor { data, indptr }, pool, mark, floor))
            .unwrap()
        {
            return;
        }
        if mark {
            assert_eq!(b.read_i32(floor), vec![13, 13, 13, 8, 8, 8, 8, 0]);
            assert_eq!(b.read_i32(state), vec![8, 0, 13, 0, 0]);
        } else {
            assert_eq!(b.read_i32(floor), vec![9, 9, 9, 7, 7, 7, 7, 0]);
            assert_eq!(b.read_i32(state), state0.to_vec());
        }
    }
}

#[test]
fn a_rows_kv_end_is_its_lanes_last_key() {
    let mut b = Bench::new();
    let keys = b.zeros(Dtype::Bf16, CELLS as u32, 8);
    let mut pool = kv_pool(&mut b, keys);
    pool.page_indptr = b.i32(4, 1, &[0, 3, 5, 5]);
    let last = b.i32(3, 1, &[2, 4, 0]);
    let req = b.i32(6, 1, &[0, 1, -1, 2, 1, 0]);
    let ends = b.zeros(Dtype::I32, 6, 1);
    if !b
        .run(|ctx| attn::kv_ends(ctx, req, &pool, last, ends))
        .unwrap()
    {
        return;
    }
    assert_eq!(b.read_i32(ends), vec![9, 7, -1, -1, 7, 9]);
}

#[test]
fn a_window_with_sinks_reads_the_keys_that_end_at_its_position() {
    let (heads, d, window) = (4usize, 8usize, 3usize);
    let ends = [9i32, 9, 6, 1];
    let reqs = [0i32, 0, 1, 1];
    let rows = ends.len();
    let mut seed = 11u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        round_bf16(((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
    };
    let q: Vec<f32> = (0..rows * heads * d).map(|_| next()).collect();
    let kv: Vec<f32> = (0..CELLS * d).map(|_| next()).collect();
    let sinks: Vec<f32> = (0..heads).map(|_| next()).collect();
    let scale = 0.35f32;

    let mut b = Bench::new();
    let qt = b.bf16(rows as u32, (heads * d) as u32, &q);
    let keys = b.bf16(CELLS as u32, d as u32, &kv);
    let pool = kv_pool(&mut b, keys);
    let pos = b.i32(rows as u32, 1, &ends);
    let req = b.i32(rows as u32, 1, &reqs);
    let sink = b.f32(heads as u32, 1, &sinks);
    let o = b.zeros(Dtype::Bf16, rows as u32, (heads * d) as u32);
    let lse = b.zeros(Dtype::F32, rows as u32, heads as u32);
    if !b
        .run(|ctx| {
            attn::window_sink(
                ctx,
                qt,
                pos,
                req,
                pool,
                sink,
                window as u32,
                d as u32,
                scale,
                o,
                lse,
            )
        })
        .unwrap()
    {
        return;
    }

    let mut want = vec![0f32; rows * heads * d];
    for r in 0..rows {
        let (p, lane) = (ends[r] as usize, reqs[r] as usize);
        let keys: Vec<usize> = (p + 1 - window.min(p + 1)..=p)
            .map(|at| PAGES[lane][at / PS] as usize * PS + at % PS)
            .collect();
        for h in 0..heads {
            let qh = &q[(r * heads + h) * d..(r * heads + h + 1) * d];
            let s: Vec<f32> = keys
                .iter()
                .map(|&c| (0..d).map(|i| qh[i] * kv[c * d + i]).sum::<f32>() * scale)
                .collect();
            let m = s.iter().copied().fold(sinks[h], f32::max);
            let den = s.iter().map(|x| (x - m).exp()).sum::<f32>() + (sinks[h] - m).exp();
            for i in 0..d {
                let num: f32 = keys
                    .iter()
                    .zip(&s)
                    .map(|(&c, x)| (x - m).exp() * kv[c * d + i])
                    .sum();
                want[(r * heads + h) * d + i] = num / den;
            }
        }
    }
    assert_close(&b.read_f32(o), &want, 2e-2, 2e-2);
}
