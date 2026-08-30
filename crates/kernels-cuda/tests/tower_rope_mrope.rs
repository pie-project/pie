//! **THE MULTIMODAL ROTARY, AGAINST A CPU REFERENCE AND AGAINST THE SCALAR
//! ROTATION IT IS A VARIANT OF.**
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test tower_rope_mrope -- --nocapture
//! ```
//!
//! `elemwise::rope_mrope::interleaved` turns each head by one of THREE
//! positions, and which one is a function of the frequency pair alone
//! (`.wiki/alto/multimodal.md` §2; qwen36 states the sections `[11, 11, 10]`).
//! The ways that can be silently wrong are all about which pair went to which
//! axis, so that is what the gates pin:
//!
//! ```text
//! (a) the rotation is the rotation: every pair of every head matches an f32
//!     reference built from the same bf16 inputs and the same section split
//! (b) THE DEGENERATE TRIPLE: a row whose (t, h, w) are one number comes out
//!     where `rope::partial` at that number puts it — the mrope arm is the
//!     scalar arm plus a choice of position, and nothing else
//! (c) THE SPLIT IS THE SECTIONS: moving only `w` moves exactly the `w` pairs
//!     — every other pair of every head is bit-identical — and it does move
//!     some, so the claim is not vacuous. Likewise for `h`
//! (d) a partial rotation leaves the tail of the head alone, bit for bit
//! (e) the refusals fire by name: sections wider than the head's pairs, a
//!     position stream that is not [rows, 3] i32, an odd head, a rotated
//!     prefix wider than the head it sits in
//! ```

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16};

use dtype::Dtype;
use kernels_cuda::elemwise::{rope, rope_mrope};
use kernels_cuda::tensor::Tensor;

const HEAD_DIM: u32 = 64;

const Q_HEADS: u32 = 3;

const KV_HEADS: u32 = 2;

const ROWS: u32 = 6;

const THETA: f32 = 10_000.0;

/// qwen36's own, and the only section triple this campaign has to serve.
const SECTIONS: [u32; 3] = [11, 11, 10];

/// Which of `(t, h, w)` turns pair `p` — the interleaved split, transcribed
/// from the unit so the reference and the kernel can be compared and not just
/// agreed with.
fn axis_of(pair: u32, sections: [u32; 3]) -> usize {
    match pair % 3 {
        1 if pair < 3 * sections[1] => 1,
        2 if pair < 3 * sections[2] => 2,
        _ => 0,
    }
}

struct Rotated {
    q: Vec<u16>,
    k: Vec<u16>,
}

/// One fire of the mrope arm, in place, over fresh copies of `q`/`k`.
fn fire_mrope(
    q: &[u16],
    k: &[u16],
    positions: &[i32],
    sections: [u32; 3],
    rotary_dim: u32,
) -> Rotated {
    let mut gpu = Gpu::open();
    let q_at = gpu.up(q);
    let k_at = gpu.up(k);
    let pos_at = gpu.up(positions);

    let mut q_handle = Tensor::new(q_at, ROWS, Q_HEADS * HEAD_DIM, Dtype::Bf16);
    let mut k_handle = Tensor::new(k_at, ROWS, KV_HEADS * HEAD_DIM, Dtype::Bf16);
    rope_mrope::interleaved(
        &gpu.ctx(),
        &mut q_handle,
        &mut k_handle,
        Tensor::new(pos_at, ROWS, 3, Dtype::I32),
        sections,
        rotary_dim,
        HEAD_DIM,
        THETA,
    )
    .expect("the multimodal rotary enqueues");
    gpu.sync();
    Rotated {
        q: gpu.down(q_at, q.len()),
        k: gpu.down(k_at, k.len()),
    }
}

/// The scalar arm this one is a variant of, fired over the same bytes.
fn fire_scalar(q: &[u16], k: &[u16], positions: &[i32]) -> Rotated {
    let mut gpu = Gpu::open();
    let q_at = gpu.up(q);
    let k_at = gpu.up(k);
    let pos_at = gpu.up(positions);

    let mut q_handle = Tensor::new(q_at, ROWS, Q_HEADS * HEAD_DIM, Dtype::Bf16);
    let mut k_handle = Tensor::new(k_at, ROWS, KV_HEADS * HEAD_DIM, Dtype::Bf16);
    rope::partial(
        &gpu.ctx(),
        &mut q_handle,
        &mut k_handle,
        Tensor::new(pos_at, ROWS, 1, Dtype::I32),
        HEAD_DIM,
        HEAD_DIM,
        THETA,
    )
    .expect("the scalar rotary enqueues");
    gpu.sync();
    Rotated {
        q: gpu.down(q_at, q.len()),
        k: gpu.down(k_at, k.len()),
    }
}

/// The rotation, in f32, from the same bf16 inputs and the same split.
fn reference(
    values: &[f32],
    heads: u32,
    positions: &[i32],
    sections: [u32; 3],
    rotary_dim: u32,
) -> Vec<f32> {
    let head_dim = HEAD_DIM as usize;
    let half = head_dim / 2;
    let angles = (rotary_dim / 2) as usize;
    let mut out = values.to_vec();
    for row in 0..ROWS as usize {
        for head in 0..heads as usize {
            let base = (row * heads as usize + head) * head_dim;
            for pair in 0..half {
                if pair >= angles {
                    continue;
                }
                let axis = axis_of(u32::try_from(pair).expect("a pair fits"), sections);
                #[allow(clippy::cast_precision_loss)]
                let position = positions[row * 3 + axis] as f32;
                #[allow(clippy::cast_precision_loss)]
                let freq = THETA.powf(-2.0 * pair as f32 / head_dim as f32);
                let (sin, cos) = (position * freq).sin_cos();
                let a = values[base + pair];
                let b = values[base + pair + half];
                out[base + pair] = a * cos - b * sin;
                out[base + pair + half] = b * cos + a * sin;
            }
        }
    }
    out
}

fn inputs(seed: u64) -> (Vec<u16>, Vec<f32>, Vec<u16>, Vec<f32>) {
    let mut rng = Lcg::seeded(seed);
    let (q_raw, q_exact) = rng.row((ROWS * Q_HEADS * HEAD_DIM) as usize);
    let (k_raw, k_exact) = rng.row((ROWS * KV_HEADS * HEAD_DIM) as usize);
    (q_raw, q_exact, k_raw, k_exact)
}

/// Positions that are three different small numbers per row. Small on
/// purpose: the kernel's `__sincosf` is a fast approximation whose error
/// grows with the angle, and a golden should measure the split rather than
/// the range reduction.
fn triples() -> Vec<i32> {
    let mut out = Vec::with_capacity(ROWS as usize * 3);
    for row in 0..ROWS as i32 {
        out.extend([row + 1, 2 * row + 3, 3 * row + 7]);
    }
    out
}

/// (a): the rotation is the rotation.
#[test]
fn every_pair_turns_by_the_axis_its_section_names() {
    let (q_raw, q, k_raw, k) = inputs(21);
    let positions = triples();
    let landed = fire_mrope(&q_raw, &k_raw, &positions, SECTIONS, HEAD_DIM);

    for (what, got, want) in [
        (
            "q",
            &landed.q,
            reference(&q, Q_HEADS, &positions, SECTIONS, HEAD_DIM),
        ),
        (
            "k",
            &landed.k,
            reference(&k, KV_HEADS, &positions, SECTIONS, HEAD_DIM),
        ),
    ] {
        for (at, expected) in want.iter().enumerate() {
            let landed = from_bf16(got[at]);
            assert!(
                close(landed, *expected),
                "{what}[{at}] landed {landed} and the reference says {expected}"
            );
        }
    }
}

/// (b) THE DEGENERATE TRIPLE: three copies of one position is the scalar
/// rotation. Compared on the raw bf16 words, one unit of the last place
/// apart — the two kernels compute the same expression, and anything wider
/// than a rounding is a different formula.
#[test]
fn a_row_whose_three_positions_agree_is_the_scalar_rotation() {
    let (q_raw, _, k_raw, _) = inputs(4);
    let scalars: Vec<i32> = (0..ROWS as i32).map(|row| row * 3 + 2).collect();
    let mut positions = Vec::with_capacity(scalars.len() * 3);
    for position in &scalars {
        positions.extend([*position, *position, *position]);
    }

    let mrope = fire_mrope(&q_raw, &k_raw, &positions, SECTIONS, HEAD_DIM);
    let scalar = fire_scalar(&q_raw, &k_raw, &scalars);

    for (what, got, want) in [("q", &mrope.q, &scalar.q), ("k", &mrope.k, &scalar.k)] {
        for (at, expected) in want.iter().enumerate() {
            let apart = i32::from(got[at]) - i32::from(*expected);
            assert!(
                apart.abs() <= 1,
                "{what}[{at}]: the mrope arm answered {:#06x} where the scalar arm answered \
                 {expected:#06x}",
                got[at]
            );
        }
    }
}

/// (c) THE SPLIT IS THE SECTIONS: one axis moves, and exactly its own pairs
/// move with it.
#[test]
fn moving_one_axis_moves_exactly_that_axiss_pairs() {
    let (q_raw, _, k_raw, _) = inputs(8);
    let base = triples();

    for axis in [1usize, 2usize] {
        let mut moved = base.clone();
        for row in 0..ROWS as usize {
            moved[row * 3 + axis] += 5;
        }
        let before = fire_mrope(&q_raw, &k_raw, &base, SECTIONS, HEAD_DIM);
        let after = fire_mrope(&q_raw, &k_raw, &moved, SECTIONS, HEAD_DIM);

        let half = (HEAD_DIM / 2) as usize;
        let mut moved_pairs = 0;
        for (what, got, want, heads) in [
            ("q", &before.q, &after.q, Q_HEADS),
            ("k", &before.k, &after.k, KV_HEADS),
        ] {
            for row in 0..ROWS as usize {
                for head in 0..heads as usize {
                    let at = (row * heads as usize + head) * HEAD_DIM as usize;
                    for pair in 0..half {
                        let mine =
                            axis_of(u32::try_from(pair).expect("a pair fits"), SECTIONS) == axis;
                        let same = got[at + pair] == want[at + pair]
                            && got[at + pair + half] == want[at + pair + half];
                        if mine {
                            moved_pairs += usize::from(!same);
                        } else {
                            assert!(
                                same,
                                "{what}: pair {pair} of row {row} head {head} moved when only \
                                 axis {axis} did, and pair {pair} is not that axis's"
                            );
                        }
                    }
                }
            }
        }
        assert!(
            moved_pairs > 0,
            "axis {axis} owns pairs and moving it moved none of them"
        );
    }
}

/// (d): a partial rotation is partial. The pairs above `rotary_dim / 2` are
/// the head's untouched tail, bit for bit.
#[test]
fn a_partial_rotation_leaves_the_heads_tail_alone() {
    let (q_raw, _, k_raw, _) = inputs(13);
    let positions = triples();
    let rotary_dim = HEAD_DIM / 2;
    let landed = fire_mrope(&q_raw, &k_raw, &positions, SECTIONS, rotary_dim);

    let half = (HEAD_DIM / 2) as usize;
    let angles = (rotary_dim / 2) as usize;
    for (what, before, after, heads) in [
        ("q", &q_raw, &landed.q, Q_HEADS),
        ("k", &k_raw, &landed.k, KV_HEADS),
    ] {
        for row in 0..ROWS as usize {
            for head in 0..heads as usize {
                let at = (row * heads as usize + head) * HEAD_DIM as usize;
                for pair in angles..half {
                    assert_eq!(
                        before[at + pair],
                        after[at + pair],
                        "{what}: pair {pair} is past the rotated prefix and moved"
                    );
                    assert_eq!(
                        before[at + pair + half],
                        after[at + pair + half],
                        "{what}: pair {pair}'s partner is past the rotated prefix and moved"
                    );
                }
            }
        }
    }
}

/// (e): the refusals, by name.
#[test]
fn the_refusals_fire_by_name() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let q_at = gpu.zeros((ROWS * Q_HEADS * HEAD_DIM) as usize * 2);
    let k_at = gpu.zeros((ROWS * KV_HEADS * HEAD_DIM) as usize * 2);
    let pos_at = gpu.up(&triples());

    let mut q = Tensor::new(q_at, ROWS, Q_HEADS * HEAD_DIM, Dtype::Bf16);
    let mut k = Tensor::new(k_at, ROWS, KV_HEADS * HEAD_DIM, Dtype::Bf16);
    let triple = Tensor::new(pos_at, ROWS, 3, Dtype::I32);

    let overrun = rope_mrope::interleaved(
        &ctx,
        &mut q,
        &mut k,
        triple,
        [16, 16, 16],
        HEAD_DIM,
        HEAD_DIM,
        THETA,
    );
    assert!(
        format!(
            "{:?}",
            overrun.expect_err("sections wider than the head are refused")
        )
        .contains("frequency pairs"),
        "sections that do not fit the head's pairs are refused"
    );

    let scalar_stream = rope_mrope::interleaved(
        &ctx,
        &mut q,
        &mut k,
        Tensor::new(pos_at, ROWS, 1, Dtype::I32),
        SECTIONS,
        HEAD_DIM,
        HEAD_DIM,
        THETA,
    );
    assert!(
        format!(
            "{:?}",
            scalar_stream.expect_err("a scalar position stream is refused")
        )
        .contains("(t, h, w)"),
        "a position stream that is not one triple per row is refused"
    );

    let odd_head = rope_mrope::interleaved(
        &ctx, &mut q, &mut k, triple, SECTIONS, 63, 63, THETA,
    );
    assert!(
        odd_head.is_err(),
        "a head with no whole number of rotation pairs is refused"
    );

    let long_prefix = rope_mrope::interleaved(
        &ctx,
        &mut q,
        &mut k,
        triple,
        SECTIONS,
        HEAD_DIM + 2,
        HEAD_DIM,
        THETA,
    );
    assert!(
        long_prefix.is_err(),
        "a rotated prefix wider than its head is refused"
    );
}
