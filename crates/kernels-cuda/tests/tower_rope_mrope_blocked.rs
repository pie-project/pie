//! **THE TOWER'S ROTATION, AGAINST `apply_rotary_pos_emb_vision` ITSELF.**
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test tower_rope_mrope_blocked -- --nocapture
//! ```
//!
//! `elemwise::rope_mrope::blocked` is the arm `.wiki/alto/multimodal.md` §6.3
//! calls "the half-split one": the vision blocks rotate, and they hand their
//! sections out in CONTIGUOUS BLOCKS where the trunk interleaves them. Two
//! things about it would look plausible while being wrong, so both are pinned
//! against a reference transcribed from `transformers` rather than from the
//! kernel:
//!
//! * WHICH pairs take which axis — contiguous, not `t, h, w, t, h, w, …`;
//! * and at WHAT FREQUENCY. Each block RESTARTS the ladder:
//!   `Qwen3_5VisionRotaryEmbedding(head_dim / 2)` builds `head_dim/4`
//!   frequencies over a `head_dim/2`-wide ladder and `freqs[pos_ids]` indexes
//!   that one ladder once per axis before flattening, so the exponent counts
//!   WITHIN the block. A kernel that kept the head's global pair index would
//!   answer smooth, sensible, wrong numbers.
//!
//! ```text
//! (a) THE VISION FORMULA: at the tower's own sections, every pair of every
//!     head matches `apply_rotary_pos_emb_vision` written out in f32 —
//!     VisionRotaryEmbedding(head_dim/2), freqs[(h, w)].flatten, cat(e, e),
//!     rotate_half
//! (b) the general blocked reference agrees with it at those sections and
//!     with the kernel at others, including a three-section split
//! (c) THE BLOCKS ARE THE SECTIONS: moving `h` moves exactly the h block's
//!     pairs, moving `w` exactly the w block's, and moving `t` — which the
//!     tower gives zero pairs — moves nothing at all
//! (d) THE TWO ARMS ARE TWO: blocked and interleaved disagree on the same
//!     bytes, so the arm is not a second name for the first
//! (e) pairs past `Σsections` and past `rotary_dim / 2` are left alone, bit
//!     for bit
//! (f) the refusals are the interleaved arm's, because the body is
//! ```

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16};

use dtype::Dtype;
use kernels_cuda::elemwise::rope_mrope;
use kernels_cuda::tensor::Tensor;

const HEAD_DIM: u32 = 64;

const Q_HEADS: u32 = 3;

const KV_HEADS: u32 = 2;

const ROWS: u32 = 6;

const THETA: f32 = 10_000.0;

/// **THE TOWER'S OWN SECTIONS**: no time axis, then `head_dim/4` pairs of `h`
/// and `head_dim/4` of `w`. `sections[0] == 0` is how a two-axis rotation is
/// spelled here — the position stream stays `[rows, 3]` on both axes and the
/// `t` column is read by nothing.
const TOWER: [u32; 3] = [0, HEAD_DIM / 4, HEAD_DIM / 4];

/// A three-section split, so (b) and (c) are not statements about zero.
const THREE: [u32; 3] = [8, 12, 12];

struct Rotated {
    q: Vec<u16>,
    k: Vec<u16>,
}

fn fire_blocked(
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
    rope_mrope::blocked(
        &gpu.ctx(),
        &mut q_handle,
        &mut k_handle,
        Tensor::new(pos_at, ROWS, 3, Dtype::I32),
        sections,
        rotary_dim,
        HEAD_DIM,
        THETA,
    )
    .expect("the tower's rotary enqueues");
    gpu.sync();
    Rotated {
        q: gpu.down(q_at, q.len()),
        k: gpu.down(k_at, k.len()),
    }
}

fn fire_interleaved(q: &[u16], k: &[u16], positions: &[i32], sections: [u32; 3]) -> Rotated {
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
        HEAD_DIM,
        HEAD_DIM,
        THETA,
    )
    .expect("the trunk's rotary enqueues");
    gpu.sync();
    Rotated {
        q: gpu.down(q_at, q.len()),
        k: gpu.down(k_at, k.len()),
    }
}

/// **`apply_rotary_pos_emb_vision`, WRITTEN OUT.** Not the kernel's formula
/// restated: `transformers`' own, in the order it computes it, so the two can
/// disagree.
///
/// `VisionRotaryEmbedding(dim = head_dim / 2)` holds `dim / 2` inverse
/// frequencies, `inv[i] = theta^(-2i/dim)`. `freqs[pos_ids]` indexes it once
/// per axis — `(h, w)` — and `.flatten(1)` concatenates the two into a
/// `head_dim/2`-wide angle row; `cat(emb, emb)` doubles it, and `rotate_half`
/// pairs `d` with `d + head_dim/2`.
fn vision_reference(values: &[f32], heads: u32, positions: &[i32]) -> Vec<f32> {
    let head_dim = HEAD_DIM as usize;
    let half = head_dim / 2;
    let ladder = half;
    let per_axis = ladder / 2;

    let mut out = values.to_vec();
    for row in 0..ROWS as usize {
        // The angle row: `per_axis` entries turned by `h`, then `per_axis`
        // turned by `w`.
        let mut angle = vec![0.0f32; half];
        for (axis, column) in [(1usize, 0usize), (2usize, 1usize)] {
            #[allow(clippy::cast_precision_loss)]
            let position = positions[row * 3 + axis] as f32;
            for i in 0..per_axis {
                #[allow(clippy::cast_precision_loss)]
                let inv = THETA.powf(-2.0 * i as f32 / ladder as f32);
                angle[column * per_axis + i] = position * inv;
            }
        }
        for head in 0..heads as usize {
            let base = (row * heads as usize + head) * head_dim;
            for d in 0..half {
                let (sin, cos) = angle[d].sin_cos();
                let a = values[base + d];
                let b = values[base + d + half];
                out[base + d] = a * cos - b * sin;
                out[base + d + half] = b * cos + a * sin;
            }
        }
    }
    out
}

/// The general blocked split, in f32 — contiguous blocks, each restarting the
/// ladder over `Σsections`.
fn blocked_reference(
    values: &[f32],
    heads: u32,
    positions: &[i32],
    sections: [u32; 3],
    rotary_dim: u32,
) -> Vec<f32> {
    let head_dim = HEAD_DIM as usize;
    let half = head_dim / 2;
    let angles = (rotary_dim / 2) as usize;
    let total: usize = sections.iter().map(|&s| s as usize).sum();
    let bounds = [
        sections[0] as usize,
        sections[0] as usize + sections[1] as usize,
    ];

    let mut out = values.to_vec();
    for row in 0..ROWS as usize {
        for head in 0..heads as usize {
            let base = (row * heads as usize + head) * head_dim;
            for pair in 0..half {
                if pair >= angles || pair >= total {
                    continue;
                }
                let (axis, within) = if pair < bounds[0] {
                    (0usize, pair)
                } else if pair < bounds[1] {
                    (1usize, pair - bounds[0])
                } else {
                    (2usize, pair - bounds[1])
                };
                #[allow(clippy::cast_precision_loss)]
                let position = positions[row * 3 + axis] as f32;
                #[allow(clippy::cast_precision_loss)]
                let freq = THETA.powf(-2.0 * within as f32 / total as f32);
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

/// A grid: each row is one patch's `(t, h, w)`, `t` always zero as a still
/// image's is. Small, because `__sincosf` is a fast approximation whose error
/// grows with the angle and a golden should measure the split.
fn grid() -> Vec<i32> {
    let mut out = Vec::with_capacity(ROWS as usize * 3);
    for row in 0..ROWS as i32 {
        out.extend([0, row / 3 + 1, row % 3 + 2]);
    }
    out
}

/// (a) THE VISION FORMULA.
#[test]
fn the_towers_sections_reproduce_apply_rotary_pos_emb_vision() {
    let (q_raw, q, k_raw, k) = inputs(31);
    let positions = grid();
    let landed = fire_blocked(&q_raw, &k_raw, &positions, TOWER, HEAD_DIM);

    for (what, got, want) in [
        ("q", &landed.q, vision_reference(&q, Q_HEADS, &positions)),
        ("k", &landed.k, vision_reference(&k, KV_HEADS, &positions)),
    ] {
        for (at, expected) in want.iter().enumerate() {
            let landed = from_bf16(got[at]);
            assert!(
                close(landed, *expected),
                "{what}[{at}] landed {landed} and apply_rotary_pos_emb_vision says {expected}"
            );
        }
    }
}

/// (b) the general reference agrees at the tower's sections and at a
/// three-section split, which is what says the arm is not hard-wired to the
/// one shape.
#[test]
fn the_blocked_split_is_the_blocked_split_at_every_section_triple() {
    let (q_raw, q, k_raw, k) = inputs(97);
    let positions = grid();

    for sections in [TOWER, THREE] {
        let landed = fire_blocked(&q_raw, &k_raw, &positions, sections, HEAD_DIM);
        for (what, got, want) in [
            (
                "q",
                &landed.q,
                blocked_reference(&q, Q_HEADS, &positions, sections, HEAD_DIM),
            ),
            (
                "k",
                &landed.k,
                blocked_reference(&k, KV_HEADS, &positions, sections, HEAD_DIM),
            ),
        ] {
            for (at, expected) in want.iter().enumerate() {
                let landed = from_bf16(got[at]);
                assert!(
                    close(landed, *expected),
                    "{what}[{at}] at {sections:?} landed {landed}, the reference says {expected}"
                );
            }
        }
    }
}

/// (c) THE BLOCKS ARE THE SECTIONS — and the tower's zero-wide `t` block is a
/// column nothing reads.
#[test]
fn moving_one_axis_moves_exactly_its_own_block() {
    let (q_raw, _, k_raw, _) = inputs(55);
    let base = grid();
    let half = (HEAD_DIM / 2) as usize;
    let bounds = [
        TOWER[0] as usize,
        TOWER[0] as usize + TOWER[1] as usize,
        TOWER[0] as usize + TOWER[1] as usize + TOWER[2] as usize,
    ];

    for axis in 0usize..3 {
        let mut moved = base.clone();
        for row in 0..ROWS as usize {
            moved[row * 3 + axis] += 4;
        }
        let before = fire_blocked(&q_raw, &k_raw, &base, TOWER, HEAD_DIM);
        let after = fire_blocked(&q_raw, &k_raw, &moved, TOWER, HEAD_DIM);

        let owned = match axis {
            0 => 0..bounds[0],
            1 => bounds[0]..bounds[1],
            _ => bounds[1]..bounds[2],
        };
        let mut moved_pairs = 0;
        for (what, got, want, heads) in [
            ("q", &before.q, &after.q, Q_HEADS),
            ("k", &before.k, &after.k, KV_HEADS),
        ] {
            for row in 0..ROWS as usize {
                for head in 0..heads as usize {
                    let at = (row * heads as usize + head) * HEAD_DIM as usize;
                    for pair in 0..half {
                        let same = got[at + pair] == want[at + pair]
                            && got[at + pair + half] == want[at + pair + half];
                        if owned.contains(&pair) {
                            moved_pairs += usize::from(!same);
                        } else {
                            assert!(
                                same,
                                "{what}: pair {pair} of row {row} head {head} moved when only \
                                 axis {axis} did, and that axis owns {owned:?}"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(
            moved_pairs > 0,
            !owned.is_empty(),
            "axis {axis} owns {owned:?} and moving it moved {moved_pairs} pairs"
        );
    }
}

/// (d) THE TWO ARMS ARE TWO. If blocked and interleaved agreed on a section
/// triple that is not degenerate, this whole file would be pinning the arm
/// that already shipped.
#[test]
fn the_blocked_arm_and_the_interleaved_arm_disagree() {
    let (q_raw, _, k_raw, _) = inputs(64);
    let positions = grid();

    let blocked = fire_blocked(&q_raw, &k_raw, &positions, THREE, HEAD_DIM);
    let interleaved = fire_interleaved(&q_raw, &k_raw, &positions, THREE);

    let apart = blocked
        .q
        .iter()
        .zip(&interleaved.q)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        apart > 0,
        "the two section layouts answered the same {} words, so one of them is not \
         computing what it says",
        blocked.q.len()
    );
}

/// (e) the two tails: past `Σsections`, and past `rotary_dim / 2`.
#[test]
fn the_pairs_no_section_claims_are_left_alone() {
    let (q_raw, _, k_raw, _) = inputs(11);
    let positions = grid();
    let half = (HEAD_DIM / 2) as usize;

    // `[4, 4, 4]` claims twelve pairs of thirty-two; the other twenty are the
    // head's untouched tail.
    let sections = [4u32, 4, 4];
    let total: usize = sections.iter().map(|&s| s as usize).sum();
    let landed = fire_blocked(&q_raw, &k_raw, &positions, sections, HEAD_DIM);
    for (what, before, after, heads) in [
        ("q", &q_raw, &landed.q, Q_HEADS),
        ("k", &k_raw, &landed.k, KV_HEADS),
    ] {
        for row in 0..ROWS as usize {
            for head in 0..heads as usize {
                let at = (row * heads as usize + head) * HEAD_DIM as usize;
                for pair in total..half {
                    assert_eq!(
                        before[at + pair], after[at + pair],
                        "{what}: pair {pair} is past the sections and moved"
                    );
                    assert_eq!(
                        before[at + pair + half],
                        after[at + pair + half],
                        "{what}: pair {pair}'s partner is past the sections and moved"
                    );
                }
            }
        }
    }

    // And the rotated prefix, which bounds the sections from the other side.
    let rotary_dim = HEAD_DIM / 2;
    let angles = (rotary_dim / 2) as usize;
    let short = fire_blocked(&q_raw, &k_raw, &positions, TOWER, rotary_dim);
    for (what, before, after, heads) in [
        ("q", &q_raw, &short.q, Q_HEADS),
        ("k", &k_raw, &short.k, KV_HEADS),
    ] {
        for row in 0..ROWS as usize {
            for head in 0..heads as usize {
                let at = (row * heads as usize + head) * HEAD_DIM as usize;
                for pair in angles..half {
                    assert_eq!(
                        before[at + pair], after[at + pair],
                        "{what}: pair {pair} is past the rotated prefix and moved"
                    );
                }
            }
        }
    }
}

/// (f) the refusals are the same refusals — the two arms share one body, and
/// this is what says so from the outside.
#[test]
fn the_blocked_arm_refuses_what_the_interleaved_one_does() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let q_at = gpu.zeros((ROWS * Q_HEADS * HEAD_DIM) as usize * 2);
    let k_at = gpu.zeros((ROWS * KV_HEADS * HEAD_DIM) as usize * 2);
    let pos_at = gpu.up(&grid());

    let mut q = Tensor::new(q_at, ROWS, Q_HEADS * HEAD_DIM, Dtype::Bf16);
    let mut k = Tensor::new(k_at, ROWS, KV_HEADS * HEAD_DIM, Dtype::Bf16);

    let overrun = rope_mrope::blocked(
        &ctx,
        &mut q,
        &mut k,
        Tensor::new(pos_at, ROWS, 3, Dtype::I32),
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

    let scalar_stream = rope_mrope::blocked(
        &ctx,
        &mut q,
        &mut k,
        Tensor::new(pos_at, ROWS, 1, Dtype::I32),
        TOWER,
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
}
