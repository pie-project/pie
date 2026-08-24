//! The rank cut, run: `model::produce` handed a demand column and a rank,
//! against a checkpoint that holds the whole tensor.
//!
//! # What has to be true, and what only looks like it
//!
//! A `-tp2` plan states per-rank shapes, so a join that closes proves the
//! slice was the RIGHT SIZE and nothing more. Two ranks reading the same half
//! would satisfy it; so would a rank taking the first half of a `[gate | up]`
//! bank's concatenated axis, which is the whole gate and none of the up —
//! same shape, same byte count, a model that computes a plausible wrong
//! answer. So the checks here are about WHICH bytes:
//!
//! * the ranks PARTITION the checkpoint — every byte lands on exactly one
//!   rank, and in checkpoint order within it;
//! * each segment of a packed axis is cut, not the axis as a whole;
//! * an axis below the cut moves as a unit and an axis above it repeats,
//!   which is what makes the routed `[experts, out, in]` banks and the mxfp4
//!   `[.., K/32, 16]` code planes one arm and not three;
//! * at world 1 the bytes are the checkpoint's, unmoved — the property every
//!   single-GPU row in the catalog rests on.
//!
//! The synthetic checkpoints here carry byte-per-index payloads so a slice is
//! readable as a list of positions rather than as a digest. The real-weights
//! half of this is `bin/baker_load.rs --digest` against an independent slicer.

use model::produce::{Dtype, HostTensor, ProduceError, produce};
use model_dsl::load::{Import, SfBf16, copy};
use model_ir::plan::{Param, Shard};

/// A checkpoint tensor whose bytes are their own positions, so that a slice
/// reads as the offsets it took.
fn whole(shape: &[u64]) -> HostTensor {
    let n: u64 = shape.iter().product();
    assert!(n <= 256, "the payload is one byte per position");
    HostTensor::new(shape.iter().copied(), Dtype::U8, (0..n as u8).collect())
}

fn param(name: &str, shape: &[u64], shard: Shard) -> Param {
    Param {
        name: name.to_string(),
        shape: shape.to_vec(),
        shard,
        repr: "u8".to_string(),
    }
}

fn cut(axis: u32, segments: &[u64]) -> Shard {
    Shard::Cut {
        axis,
        segments: segments.to_vec(),
    }
}

/// One production row, `w` from `src`, produced for `rank` against `t`.
fn slice_of(t: &HostTensor, p: &Param, rank: u32) -> Result<HostTensor, ProduceError> {
    let mut import = Import::new::<SfBf16>();
    import.write(p.name.clone(), copy("src"));
    let read = |name: &str| (name == "src").then(|| t.clone());
    let mut out = produce(&import, std::slice::from_ref(p), rank, &read)?;
    Ok(out.remove(0).1)
}

/// A plain column cut: the leading axis, one segment, two ranks.
#[test]
fn a_column_cut_hands_each_rank_its_own_rows() {
    // `[4, 3]` in the checkpoint, `[2, 3]` per rank -- rows 0,1 and rows 2,3.
    let t = whole(&[4, 3]);
    let p = param("w", &[2, 3], cut(0, &[2]));

    assert_eq!(
        slice_of(&t, &p, 0).expect("rank 0").bytes,
        vec![0, 1, 2, 3, 4, 5],
    );
    assert_eq!(
        slice_of(&t, &p, 1).expect("rank 1").bytes,
        vec![6, 7, 8, 9, 10, 11],
    );
    assert_eq!(slice_of(&t, &p, 0).expect("rank 0").shape, vec![2, 3]);
}

/// A row cut: the TRAILING axis, where the bytes a rank wants are not
/// contiguous in the checkpoint at all.
#[test]
fn a_row_cut_takes_a_stripe_out_of_every_row() {
    // `[2, 6]` in the checkpoint, `[2, 3]` per rank.
    let t = whole(&[2, 6]);
    let p = param("w", &[2, 3], cut(1, &[3]));

    assert_eq!(
        slice_of(&t, &p, 0).expect("rank 0").bytes,
        vec![0, 1, 2, 6, 7, 8]
    );
    assert_eq!(
        slice_of(&t, &p, 1).expect("rank 1").bytes,
        vec![3, 4, 5, 9, 10, 11],
    );
}

/// THE ONE THAT A SHAPE CHECK CANNOT SEE. A `[gate | up]` bank cut in halves
/// is `[gate/2 | up/2]`, and taking the first half of the concatenated axis
/// instead gives a rank the same number of bytes and the whole of the wrong
/// thing.
#[test]
fn a_packed_cut_cuts_every_segment_and_not_the_axis() {
    // `[8, 1]` = gate rows 0..4 then up rows 4..8; `[4, 1]` per rank.
    let t = whole(&[8, 1]);
    let p = param("gate_up", &[4, 1], cut(0, &[2, 2]));

    // gate's first half, then up's first half.
    assert_eq!(slice_of(&t, &p, 0).expect("rank 0").bytes, vec![0, 1, 4, 5]);
    // gate's second half, then up's second half -- NOT rows 4..8, which is
    // what a cut of the axis rather than of its segments would give.
    assert_eq!(slice_of(&t, &p, 1).expect("rank 1").bytes, vec![2, 3, 6, 7]);
}

/// An expert bank: the fan is axis 0 and is NOT cut, the out axis is axis 1
/// and is, and axis 2 rides below it. One arm covers all three because the
/// slice walks the shape rather than the repr.
#[test]
fn a_bank_cut_repeats_under_the_expert_fan() {
    // `[2 experts, 4 out, 2 in]`, `[2, 2, 2]` per rank, cut on axis 1 as a
    // `[gate | up]` pair of one row each.
    let t = whole(&[2, 4, 2]);
    let p = param("experts_gate_up", &[2, 2, 2], cut(1, &[1, 1]));

    // Expert 0 holds positions 0..8, expert 1 holds 8..16. Within an expert,
    // gate is rows 0,1 and up is rows 2,3, each row two bytes wide.
    assert_eq!(
        slice_of(&t, &p, 0).expect("rank 0").bytes,
        vec![0, 1, 4, 5, /* expert 1 */ 8, 9, 12, 13],
    );
    assert_eq!(
        slice_of(&t, &p, 1).expect("rank 1").bytes,
        vec![2, 3, 6, 7, /* expert 1 */ 10, 11, 14, 15],
    );
}

/// The element width is a byte count and the slice respects it.
///
/// A slicer written in elements would pass every test above (they are all
/// `U8`) and halve every bf16 weight in the catalog.
#[test]
fn a_cut_moves_elements_and_not_bytes() {
    // `[4]` of bf16 = 8 bytes; `[2]` per rank = 4 bytes.
    let t = HostTensor::new([4], Dtype::Bf16, (0..8u8).collect());
    let p = param("w", &[2], cut(0, &[2]));

    assert_eq!(slice_of(&t, &p, 0).expect("rank 0").bytes, vec![0, 1, 2, 3]);
    assert_eq!(slice_of(&t, &p, 1).expect("rank 1").bytes, vec![4, 5, 6, 7]);
}

/// Every byte of the checkpoint lands on exactly one rank, in order.
///
/// The property the four shape-specific tests above are instances of, stated
/// once over every shape and partition in them: a cut LOSES nothing and
/// DUPLICATES nothing.
#[test]
fn the_ranks_partition_the_checkpoint() {
    struct Case {
        checkpoint: &'static [u64],
        mine: &'static [u64],
        axis: u32,
        segments: &'static [u64],
    }
    let cases = [
        Case {
            checkpoint: &[4, 3],
            mine: &[2, 3],
            axis: 0,
            segments: &[2],
        },
        Case {
            checkpoint: &[2, 6],
            mine: &[2, 3],
            axis: 1,
            segments: &[3],
        },
        Case {
            checkpoint: &[8, 1],
            mine: &[4, 1],
            axis: 0,
            segments: &[2, 2],
        },
        Case {
            checkpoint: &[2, 4, 2],
            mine: &[2, 2, 2],
            axis: 1,
            segments: &[1, 1],
        },
        Case {
            checkpoint: &[4, 4],
            mine: &[1, 4],
            axis: 0,
            segments: &[1],
        },
    ];
    for Case {
        checkpoint,
        mine,
        axis,
        segments,
    } in cases
    {
        let elems: u64 = checkpoint.iter().product();
        let t = whole(checkpoint);
        let p = param("w", mine, cut(axis, segments));
        let world = checkpoint[axis as usize] / mine[axis as usize];
        let mut seen: Vec<u8> = Vec::new();
        for rank in 0..world as u32 {
            let s = slice_of(&t, &p, rank).expect("a rank of a cut this shape admits");
            assert_eq!(
                s.shape, mine,
                "{checkpoint:?} cut on {axis} for rank {rank}"
            );
            seen.extend_from_slice(&s.bytes);
        }
        seen.sort_unstable();
        assert_eq!(
            seen,
            (0..elems as u8).collect::<Vec<u8>>(),
            "{checkpoint:?} cut {world} ways on axis {axis} into {segments:?} \
             does not partition the checkpoint",
        );
    }
}

/// A replicated weight is the checkpoint's, whatever the rank.
#[test]
fn a_replicated_weight_reaches_every_rank_whole() {
    let t = whole(&[4, 3]);
    let p = param("embed", &[4, 3], Shard::Replicated);
    // Rank 1 of what? Nothing here is cut, so nothing says -- and a plan that
    // cuts nothing serves the same bytes to every rank it is deployed on.
    for rank in [0u32, 1, 7] {
        let s = slice_of(&t, &p, rank).expect("replicated");
        assert_eq!(s.bytes, t.bytes, "rank {rank}");
        assert_eq!(s.shape, t.shape, "rank {rank}");
    }
}

/// At world 1 the cut is the identity, and a whole-model SKU has no rank 1.
#[test]
fn a_world_of_one_is_the_identity_and_holds_one_rank() {
    let t = whole(&[4, 3]);
    let p = param("w", &[4, 3], cut(0, &[4]));

    assert_eq!(slice_of(&t, &p, 0).expect("rank 0").bytes, t.bytes);

    // The check that catches a `tp_size = 2` deployment pointed at a row
    // nothing cuts: rank 1 of a world of one does not exist, and serving it
    // the whole model would have both ranks reduce a leg neither cut.
    let why = slice_of(&t, &p, 1)
        .expect_err("rank 1 of a world of one")
        .to_string();
    assert!(why.contains("rank 1"), "{why}");
}

/// A checkpoint whose extent this rank's does not divide is refused, naming
/// both rectangles.
#[test]
fn an_indivisible_extent_is_refused_by_name() {
    let t = whole(&[5, 3]);
    let p = param("w", &[2, 3], cut(0, &[2]));
    let why = slice_of(&t, &p, 0)
        .expect_err("5 is not a whole number of 2s")
        .to_string();
    assert!(why.contains("[5, 3]") && why.contains("[2, 3]"), "{why}");

    // And a disagreement OFF the cut axis, which is a different checkpoint
    // rather than a different degree.
    let t = whole(&[4, 5]);
    let why = slice_of(&t, &p, 0)
        .expect_err("axis 1 disagrees")
        .to_string();
    assert!(why.contains("[4, 5]"), "{why}");
}

/// Two weights, two worlds: a checkpoint is ONE cut of one plan.
///
/// The fault a `-tp2` text would produce if it divided one dim and not
/// another -- half its weights cut two ways and half cut four -- and the only
/// thing that can catch it, because each row on its own is arithmetically
/// fine.
#[test]
fn one_checkpoint_states_one_world() {
    let mut import = Import::new::<SfBf16>();
    import.write("two", copy("a"));
    import.write("four", copy("b"));
    let params = vec![
        param("two", &[2], cut(0, &[2])),
        param("four", &[2], cut(0, &[2])),
    ];
    let read = |name: &str| match name {
        // `a` is cut two ways, `b` four.
        "a" => Some(whole(&[4])),
        "b" => Some(whole(&[8])),
        _ => None,
    };
    let why = produce(&import, &params, 0, &read)
        .expect_err("two worlds")
        .to_string();
    assert!(why.contains("`four`") && why.contains("ONE cut"), "{why}");
}

/// A produced row the plan names no param for is passed through whole.
///
/// There is nothing to cut it by; the join already reports it as bytes moved
/// for a weight nobody reads, and guessing an axis would be worse than the
/// bytes.
#[test]
fn a_row_no_param_names_is_not_cut() {
    let mut import = Import::new::<SfBf16>();
    import.write("stray", copy("src"));
    let t = whole(&[4]);
    let read = |name: &str| (name == "src").then(|| t.clone());
    let out = produce(&import, &[], 0, &read).expect("no demand column");
    assert_eq!(out[0].1.bytes, t.bytes);
}
