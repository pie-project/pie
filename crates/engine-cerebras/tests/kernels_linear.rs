mod common;

use common::data;
use engine_cerebras::bench::{Bench, assert_close, round_bf16};
use kernels_cerebras::linear::{gemm, mlp};

fn matmul_ref(act: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    (0..m)
        .flat_map(|i| {
            (0..n).map(move |j| {
                round_bf16((0..k).map(|t| act[i * k + t] * w[j * k + t]).sum::<f32>())
            })
        })
        .collect()
}

#[test]
fn a_matmul_projects_the_rows_it_is_asked_for() {
    let (m, k, n) = (4u32, 24u32, 10u32);
    let acts = data((6 * k) as usize, 5); // two padded rows the op must ignore
    let ws = data((n * k) as usize, 6);
    let mut b = Bench::new();
    let act = b.bf16(6, k, &acts);
    let w = b.bf16(n, k, &ws);
    let y = b.zeros(dtype::Dtype::Bf16, m, n);
    let z = b.zeros(dtype::Dtype::Bf16, m, n);
    let ran = b
        .run(|ctx| {
            gemm::matmul(ctx, act, w, y)?;
            gemm::lm_head(ctx, act, w, z)
        })
        .unwrap();
    if !ran {
        return;
    }
    let want = matmul_ref(&acts, &ws, m as usize, k as usize, n as usize);
    assert_close(&b.read_f32(y), &want, 1e-2, 1e-2);
    assert_close(&b.read_f32(z), &want, 1e-2, 1e-2);
}

#[test]
fn swiglu_gates_the_second_half_by_the_first() {
    swiglu_case(3, 14, 7);
}

/// Rows past a PE (2 x 12288 packed) split their columns, the same block
/// of the gate and the up halves on one PE.
#[test]
fn a_wide_swiglu_splits_its_columns() {
    swiglu_case(2, 6144, 9);
}

fn swiglu_case(rows: u32, inter: u32, seed: u32) {
    let ps: Vec<f32> = data((rows * 2 * inter) as usize, seed)
        .iter()
        .map(|v| v * 3.0)
        .map(round_bf16)
        .collect();
    let mut b = Bench::new();
    let packed = b.bf16(rows, 2 * inter, &ps);
    let y = b.zeros(dtype::Dtype::Bf16, rows, inter);
    if !b.run(|ctx| mlp::swiglu(ctx, packed, inter, y)).unwrap() {
        return;
    }
    let want: Vec<f32> = ps
        .chunks(2 * inter as usize)
        .flat_map(|row| {
            let (g, u) = row.split_at(inter as usize);
            g.iter()
                .zip(u)
                .map(|(g, u)| round_bf16(g / (1.0 + (-g).exp()) * u))
                .collect::<Vec<_>>()
        })
        .collect();
    assert_close(&b.read_f32(y), &want, 2e-2, 2e-2);
}

/// Activations past a PE (8 x 1536) spread over a grid of PEs by depth
/// (and weight rows by column group), the host adding the partial outputs.
#[test]
fn a_tall_matmul_spreads_its_rows_and_weight_over_a_grid() {
    let (m, k, n) = (8u32, 1536u32, 16u32);
    let acts = data((m * k) as usize, 50);
    let ws = data((n * k) as usize, 51);
    let mut b = Bench::new();
    let act = b.bf16(m, k, &acts);
    let w = b.bf16(n, k, &ws);
    let y = b.zeros(dtype::Dtype::Bf16, m, n);
    if !b.run(|ctx| gemm::matmul(ctx, act, w, y)).unwrap() {
        return;
    }
    // The call's local sizes: half the rows, one weight row per PE.
    let rendered = b.last.as_ref().expect("rendered");
    let call = rendered
        .pe
        .lines()
        .find(|l| l.contains("k_matmul(@ptrcast") || l.contains("k_matmul_packed(@ptrcast"))
        .expect("the matmul call");
    assert!(
        !call.contains(", 1536, "),
        "the depth should be split: {call}"
    );
    let want = matmul_ref(&acts, &ws, m as usize, k as usize, n as usize);
    assert_close(&b.read_f32(y), &want, 1e-2, 1e-2);
}

/// Many rows against a wide weight (128 x 128 by 128 x 128, where no
/// two-way split fits a PE): the grid splits rows, columns and depth, the
/// host adding the partials.
#[test]
fn a_tall_and_wide_matmul_spreads_over_a_three_way_grid() {
    let (m, k, n) = (128u32, 128u32, 128u32);
    let acts = data((m * k) as usize, 52);
    let ws = data((n * k) as usize, 53);
    let mut b = Bench::new();
    let act = b.bf16(m, k, &acts);
    let w = b.bf16(n, k, &ws);
    let y = b.zeros(dtype::Dtype::Bf16, m, n);
    if !b.run(|ctx| gemm::matmul(ctx, act, w, y)).unwrap() {
        return;
    }
    let want = matmul_ref(&acts, &ws, m as usize, k as usize, n as usize);
    assert_close(&b.read_f32(y), &want, 2e-2, 2e-2);
}

/// A weight no row of PEs holds (the vocabulary projection) is multiplied
/// on the host, as a host phase of the program.
#[test]
fn a_vocabulary_sized_matmul_runs_on_the_host() {
    let (m, k, n) = (2u32, 64u32, 135_168u32); // over MAX_PES x shard_words() elements
    let acts = data((m * k) as usize, 40);
    let ws = data((n * k) as usize, 41);
    let mut b = Bench::new();
    let act = b.bf16(m, k, &acts);
    let w = b.bf16(n, k, &ws);
    let y = b.zeros(dtype::Dtype::Bf16, m, n);
    if !b.run(|ctx| gemm::lm_head(ctx, act, w, y)).unwrap() {
        return;
    }
    let rendered = b.last.as_ref().expect("rendered");
    assert!(
        !rendered.pe.contains("k_matmul("),
        "the projection should not be a device phase"
    );
    let want: Vec<f32> = matmul_ref(&acts, &ws, m as usize, k as usize, n as usize)
        .iter()
        .map(|v| round_bf16(*v))
        .collect();
    assert_close(&b.read_f32(y), &want, 1e-2, 1e-2);
}

#[test]
fn a_lora_correction_adds_each_rows_own_adapter() {
    lora_case(6, 24, 10, 4, 3, 20);
}

/// Adapters past a PE (4096 + 4096 words each) split their up bank by
/// output rows over PEs, y travelling by column block.
#[test]
fn a_wide_lora_adapter_splits_its_outputs_over_pes() {
    lora_case(6, 128, 128, 32, 4, 30);
}

/// An adapter whose down bank alone (128 x 128) exceeds a PE splits its
/// rank over PEs: each PE holds rows of the down bank and columns of the
/// up bank, and the host adds the partial outputs.
#[test]
fn a_wide_lora_adapter_splits_its_rank_over_pes() {
    lora_case(6, 128, 128, 128, 3, 50);
}

/// An adapter whose one rank row (6144 wide) plus one row of x exceed a PE
/// splits the input too: the waist as partials over input slices.
#[test]
fn a_wide_lora_adapter_splits_its_input_over_pes() {
    lora_case(2, 6144, 64, 8, 3, 60);
}

fn lora_case(rows: usize, n_in: usize, n_out: usize, rank: usize, adapters: usize, seed: u32) {
    let xs = data(rows * n_in, seed);
    let a = data(adapters * rank * n_in, seed + 1);
    let bb = data(adapters * n_out * rank, seed + 2);
    let ys = data(rows * n_out, seed + 3);
    let routes: Vec<i32> = [0i32, 2, -1, 1, 2, 3][..rows].to_vec();
    let mut b = Bench::new();
    let x = b.bf16(rows as u32, n_in as u32, &xs);
    let at = b.bf16(adapters as u32, (rank * n_in) as u32, &a);
    let bt = b.bf16(adapters as u32, (n_out * rank) as u32, &bb);
    let rt = b.i32(rows as u32, 1, &routes);
    let y = b.bf16(rows as u32, n_out as u32, &ys);
    if !b
        .run(|ctx| kernels_cerebras::linear::lora::correct(ctx, x, at, bt, rt, y))
        .unwrap()
    {
        return;
    }
    let mut want = ys.clone();
    for r in 0..rows {
        let ad = routes[r];
        if ad < 0 || ad as usize >= adapters {
            continue;
        }
        let ad = ad as usize;
        let mut waist = vec![0.0f64; rank];
        for (i, t) in waist.iter_mut().enumerate() {
            for c in 0..n_in {
                *t += f64::from(a[(ad * rank + i) * n_in + c]) * f64::from(xs[r * n_in + c]);
            }
        }
        for n in 0..n_out {
            let mut acc = 0.0f64;
            for (i, t) in waist.iter().enumerate() {
                acc += f64::from(bb[(ad * n_out + n) * rank + i]) * t;
            }
            want[r * n_out + n] = round_bf16(ys[r * n_out + n] + acc as f32);
        }
    }
    assert_close(&b.read_f32(y), &want, 1e-2, 1e-2);
}

#[test]
fn a_wide_matmul_spreads_its_weight_over_a_row_of_pes() {
    // 12 x 1024 weights: over shard_words() (4096) elements, split 3 ways.
    let (m, k, n) = (3u32, 1024u32, 12u32);
    let acts = data((m * k) as usize, 30);
    let ws = data((n * k) as usize, 31);
    let mut b = Bench::new();
    let act = b.bf16(m, k, &acts);
    let w = b.bf16(n, k, &ws);
    let y = b.zeros(dtype::Dtype::Bf16, m, n);
    if !b.run(|ctx| gemm::matmul(ctx, act, w, y)).unwrap() {
        return;
    }
    let want = matmul_ref(&acts, &ws, m as usize, k as usize, n as usize);
    assert_close(&b.read_f32(y), &want, 1e-2, 1e-2);
    assert!(b.last.as_ref().is_some_and(|r| r.pe.contains("k_matmul")));
}
