//! **THE VISION TOWERS' DENSE ATTENTION, AGAINST A CPU REFERENCE.**
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test tower_dense_attention -- --nocapture
//! ```
//!
//! `attn_dense::bidirectional` is the one real kernel the multimodal design
//! adds (`.wiki/alto/multimodal.md` §2), and every property below is a way it
//! can be silently wrong on a machine that never faults:
//!
//! ```text
//! (a) it is the softmax: over several head counts, head widths and image
//!     sizes — odd ones included — every output row matches an f32 reference
//!     computed from the same bf16 inputs
//! (b) it is BLOCK-DIAGONAL: an image's rows are a function of that image's
//!     rows alone. Perturbing a neighbouring image's keys moves nothing
//! (c) it is BIDIRECTIONAL: row 0 of an image sees the image's last row, so
//!     the answer is not a causal one (a causal kernel would pass (a) for
//!     row 0 of a one-row image and fail it here)
//! (d) a one-row image answers its own value row BIT FOR BIT — softmax over
//!     one key is one, and the kernel's rescale must leave it untouched
//! (e) a rung's padding row — past the last segment — lands zeros rather
//!     than reading a neighbour's keys
//! (f) grouped heads read the kv head they share; they do not read head 0
//! (g) the refusals fire by name: a head wider than the widest stamp, a row
//!     width that is not a whole number of heads, q heads that do not group
//!     over kv heads, and a segment list that is not an i32 indptr
//! ```

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16, to_bf16};

use dtype::Dtype;
use kernels_cuda::attn_dense;
use kernels_cuda::tensor::Tensor;

/// One shape the towers actually ask for, or an awkward neighbour of one.
struct Shape {
    what: &'static str,
    /// Patch rows per image — the segment list, spelled as sizes.
    images: &'static [u32],
    /// Rows past the last image: a patch rung's padding.
    padding: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
}

const SHAPES: &[Shape] = &[
    Shape {
        what: "one image, one head, qwen35's tower width",
        images: &[5],
        padding: 0,
        q_heads: 1,
        kv_heads: 1,
        head_dim: 64,
    },
    Shape {
        what: "three images of odd sizes, twelve heads, with a padded tail",
        images: &[3, 7, 1],
        padding: 2,
        q_heads: 12,
        kv_heads: 12,
        head_dim: 64,
    },
    Shape {
        what: "a head width that divides by neither 32 nor 64",
        images: &[4, 4],
        padding: 0,
        q_heads: 3,
        kv_heads: 3,
        head_dim: 40,
    },
    Shape {
        what: "a SigLIP-shaped head, grouped over half as many kv heads",
        images: &[6],
        padding: 1,
        q_heads: 4,
        kv_heads: 2,
        head_dim: 72,
    },
    Shape {
        what: "a 128-wide head over more rows than the block has warps",
        images: &[17],
        padding: 0,
        q_heads: 2,
        kv_heads: 2,
        head_dim: 128,
    },
];

impl Shape {
    fn rows(&self) -> u32 {
        self.images.iter().sum::<u32>() + self.padding
    }

    /// The indptr the kernel reads: `[0, n0, n0 + n1, ...]`.
    fn segments(&self) -> Vec<i32> {
        let mut out = vec![0i32];
        for size in self.images {
            let last = *out.last().expect("seeded with zero");
            out.push(last + i32::try_from(*size).expect("a test's image fits an i32"));
        }
        out
    }

    fn scale(&self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let width = self.head_dim as f32;
        1.0 / width.sqrt()
    }
}

/// The fire, and what came back: one output rectangle of bf16 raw words.
fn fire(shape: &Shape, q: &[u16], k: &[u16], v: &[u16]) -> Vec<u16> {
    let mut gpu = Gpu::open();
    let rows = shape.rows();
    let q_width = shape.q_heads * shape.head_dim;
    let kv_width = shape.kv_heads * shape.head_dim;
    let segments = shape.segments();

    let q_at = gpu.up(q);
    let k_at = gpu.up(k);
    let v_at = gpu.up(v);
    let seg_at = gpu.up(&segments);
    let o_at = gpu.zeros(q.len() * core::mem::size_of::<u16>());

    let mut o = Tensor::new(o_at, rows, q_width, Dtype::Bf16);
    attn_dense::bidirectional(
        &gpu.ctx(),
        Tensor::new(q_at, rows, q_width, Dtype::Bf16),
        Tensor::new(k_at, rows, kv_width, Dtype::Bf16),
        Tensor::new(v_at, rows, kv_width, Dtype::Bf16),
        Tensor::new(
            seg_at,
            u32::try_from(segments.len()).expect("a test's segment list fits"),
            1,
            Dtype::I32,
        ),
        shape.head_dim,
        shape.scale(),
        &mut o,
    )
    .expect("the dense attention enqueues");
    gpu.sync();
    gpu.down(o_at, q.len())
}

/// The whole of what the kernel claims, in f32, from the same bf16 inputs.
fn reference(shape: &Shape, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let head_dim = shape.head_dim as usize;
    let q_heads = shape.q_heads as usize;
    let kv_heads = shape.kv_heads as usize;
    let group = q_heads / kv_heads;
    let rows = shape.rows() as usize;
    let segments = shape.segments();

    let mut out = vec![0.0f32; rows * q_heads * head_dim];
    for span in segments.windows(2) {
        let (begin, end) = (span[0] as usize, span[1] as usize);
        for row in begin..end {
            for head in 0..q_heads {
                let kv_head = head / group;
                let q_row = &q[(row * q_heads + head) * head_dim..][..head_dim];
                let mut scores = Vec::with_capacity(end - begin);
                for key in begin..end {
                    let k_row = &k[(key * kv_heads + kv_head) * head_dim..][..head_dim];
                    let dot: f32 = q_row.iter().zip(k_row).map(|(a, b)| a * b).sum();
                    scores.push(dot * shape.scale());
                }
                let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
                let total: f32 = weights.iter().sum();
                let landed = &mut out[(row * q_heads + head) * head_dim..][..head_dim];
                for (key, weight) in (begin..end).zip(&weights) {
                    let v_row = &v[(key * kv_heads + kv_head) * head_dim..][..head_dim];
                    for (dst, src) in landed.iter_mut().zip(v_row) {
                        *dst += weight * src / total;
                    }
                }
            }
        }
    }
    out
}

fn inputs(shape: &Shape, seed: u64) -> (Vec<u16>, Vec<f32>, Vec<u16>, Vec<f32>, Vec<u16>, Vec<f32>) {
    let rows = shape.rows() as usize;
    let mut rng = Lcg::seeded(seed);
    let (q_raw, q_exact) = rng.row(rows * (shape.q_heads * shape.head_dim) as usize);
    let (k_raw, k_exact) = rng.row(rows * (shape.kv_heads * shape.head_dim) as usize);
    let (v_raw, v_exact) = rng.row(rows * (shape.kv_heads * shape.head_dim) as usize);
    (q_raw, q_exact, k_raw, k_exact, v_raw, v_exact)
}

/// (a), (c) and (f): the softmax, over every shape, bidirectional and
/// grouped. (c) rides on (a) rather than standing apart — a causal kernel
/// answers a different number for every row but the last of each image, and
/// the reference below has no mask in it at all.
#[test]
fn a_dense_row_is_the_bidirectional_softmax_of_its_own_image() {
    for shape in SHAPES {
        let (q_raw, q, k_raw, k, v_raw, v) = inputs(shape, 5);
        let got = fire(shape, &q_raw, &k_raw, &v_raw);
        let want = reference(shape, &q, &k, &v);
        let head_dim = shape.head_dim as usize;
        let q_heads = shape.q_heads as usize;
        let live = shape.images.iter().sum::<u32>() as usize;
        for row in 0..live {
            for head in 0..q_heads {
                for lane in 0..head_dim {
                    let at = (row * q_heads + head) * head_dim + lane;
                    let landed = from_bf16(got[at]);
                    assert!(
                        close(landed, want[at]),
                        "{}: row {row} head {head} lane {lane} landed {landed} and the \
                         reference says {}",
                        shape.what,
                        want[at]
                    );
                }
            }
        }
    }
}

/// (b) THE BLOCK DIAGONAL: an image's answer is a function of its own rows.
/// Rewriting a neighbouring image's keys and values moves nothing — which is
/// the property a mask plane would have had to carry and the segment list
/// carries instead.
#[test]
fn one_images_rows_never_read_another_images_keys() {
    let shape = &SHAPES[1];
    let (q_raw, _, k_raw, _, v_raw, _) = inputs(shape, 7);
    let before = fire(shape, &q_raw, &k_raw, &v_raw);

    // The second image's span, rewritten.
    let segments = shape.segments();
    let (begin, end) = (segments[1] as usize, segments[2] as usize);
    let kv_width = (shape.kv_heads * shape.head_dim) as usize;
    let mut k_moved = k_raw.clone();
    let mut v_moved = v_raw.clone();
    let mut rng = Lcg::seeded(99);
    for row in begin..end {
        for lane in 0..kv_width {
            k_moved[row * kv_width + lane] = to_bf16(rng.unit());
            v_moved[row * kv_width + lane] = to_bf16(rng.unit());
        }
    }
    let after = fire(shape, &q_raw, &k_moved, &v_moved);

    let q_width = (shape.q_heads * shape.head_dim) as usize;
    for row in 0..shape.rows() as usize {
        let touched = row >= begin && row < end;
        let same = before[row * q_width..][..q_width] == after[row * q_width..][..q_width];
        assert_eq!(
            same, !touched,
            "row {row} {} when the second image's keys moved",
            if touched { "did not move" } else { "moved" }
        );
    }
}

/// (d): softmax over one key is one, so a one-row image answers its own value
/// row and the rescale must not have touched it. Bit for bit — this is the
/// gate that catches an accumulator that divides by a denominator it built
/// with `__expf` twice.
#[test]
fn a_one_row_image_answers_its_own_value_row_bit_for_bit() {
    let shape = &SHAPES[1];
    let (q_raw, _, k_raw, _, v_raw, _) = inputs(shape, 3);
    let got = fire(shape, &q_raw, &k_raw, &v_raw);

    let segments = shape.segments();
    let row = segments[2] as usize; // the third image is one row wide
    assert_eq!(segments[3] - segments[2], 1, "the shape states a lone row");
    let width = (shape.q_heads * shape.head_dim) as usize;
    assert_eq!(
        &got[row * width..][..width],
        &v_raw[row * width..][..width],
        "a lone patch row is its own value row"
    );
}

/// (e): the rows a patch rung padded to are nobody's image, and they land
/// zeros — never a neighbour's keys, which is the second-axis form of the
/// zero-row contract the token axis already keeps.
#[test]
fn a_padding_row_past_every_image_lands_zeros() {
    for shape in SHAPES.iter().filter(|shape| shape.padding > 0) {
        let (q_raw, _, k_raw, _, v_raw, _) = inputs(shape, 11);
        let got = fire(shape, &q_raw, &k_raw, &v_raw);
        let width = (shape.q_heads * shape.head_dim) as usize;
        let live = shape.images.iter().sum::<u32>() as usize;
        for row in live..shape.rows() as usize {
            for lane in 0..width {
                assert_eq!(
                    got[row * width + lane],
                    0,
                    "{}: padding row {row} lane {lane} is not zero",
                    shape.what
                );
            }
        }
    }
}

/// (g): the four refusals, by name. Every one of them is a text or a fold
/// that disagrees with itself, and none of them may reach a launch.
#[test]
fn the_refusals_fire_by_name() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let rows = 4u32;
    let head_dim = 64u32;
    let buffer = gpu.zeros(rows as usize * 8 * head_dim as usize * 2);
    let segments = gpu.up(&[0i32, 4]);

    let bf16 = |width: u32| Tensor::new(buffer, rows, width, Dtype::Bf16);
    let indptr = Tensor::new(segments, 2, 1, Dtype::I32);
    let mut o = bf16(2 * head_dim);

    let too_wide = attn_dense::bidirectional(
        &ctx,
        bf16(2 * 512),
        bf16(2 * 512),
        bf16(2 * 512),
        indptr,
        512,
        0.1,
        &mut Tensor::new(buffer, rows, 2 * 512, Dtype::Bf16),
    );
    assert!(
        format!("{:?}", too_wide.expect_err("a 512-wide head is refused"))
            .contains("wider than"),
        "the wide head is refused for being wider than the stamp"
    );

    let ragged = attn_dense::bidirectional(
        &ctx,
        bf16(2 * head_dim + 1),
        bf16(2 * head_dim),
        bf16(2 * head_dim),
        indptr,
        head_dim,
        0.1,
        &mut o,
    );
    assert!(
        format!("{:?}", ragged.expect_err("a ragged row is refused")).contains("does not divide"),
        "a row that is not a whole number of heads is refused"
    );

    let ungrouped = attn_dense::bidirectional(
        &ctx,
        bf16(3 * head_dim),
        bf16(2 * head_dim),
        bf16(2 * head_dim),
        indptr,
        head_dim,
        0.1,
        &mut Tensor::new(buffer, rows, 3 * head_dim, Dtype::Bf16),
    );
    assert!(
        format!("{:?}", ungrouped.expect_err("3 over 2 heads is refused")).contains("group over"),
        "query heads that do not group over kv heads are refused"
    );

    let unlisted = attn_dense::bidirectional(
        &ctx,
        bf16(2 * head_dim),
        bf16(2 * head_dim),
        bf16(2 * head_dim),
        Tensor::new(segments, 2, 1, Dtype::F32),
        head_dim,
        0.1,
        &mut o,
    );
    assert!(
        unlisted.is_err(),
        "a segment list that is not an i32 indptr is refused"
    );
}
