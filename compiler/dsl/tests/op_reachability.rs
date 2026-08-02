//! Every op in the IR's table can be authored from this crate, or says why not.
//!
//! `declare_ops!` gives an op a tag, a wire layout and a row in `OP_TABLE`, and
//! the exhaustive matches in `pie-ir` and `pie-eval` make the compiler demand
//! inference and evaluation for it. Nothing plays that role for the authoring
//! surface: `value.rs` is hand-written constructors, so an op can be added to
//! the IR, type-checked, interpreted, and emitted by both backends while
//! remaining impossible to write. That gap is not hypothetical — `abs`, `sign`,
//! `recip` and `sort_desc` were each reachable everywhere except from here.
//!
//! So this traces programs that exercise the authoring surface, collects the op
//! tags that actually reach a container, and requires every row of `OP_TABLE`
//! to be either in that set or in [`NOT_AUTHORABLE`] with a reason. Both
//! directions are checked, so the exemption list cannot outlive the exemption.
//!
//! Reachability is established by *tracing*, not by naming functions: a
//! constructor that exists but emits the wrong op leaves its tag missing here.

use std::collections::BTreeSet;

use pie_dsl::builder::Builder;
use pie_dsl::prelude::*;
use pie_dsl::ptir::op::{OP_TABLE, Op};
use pie_dsl::{Channel, Traced};

/// Ops with no authoring surface, and why each one has none.
///
/// Both entries are v4-exact wire forms: the compiler must keep *reading* them
/// because shipped programs contain them, but each has a better spelling that
/// the DSL offers instead. An entry here is a claim that the op is unreachable
/// *on purpose* — adding one to silence this test re-opens the hole it closes.
const NOT_AUTHORABLE: &[(u8, &str)] = &[
    (
        0x65,
        "`Op::MaskApply` takes a packed `[ceil(n/32)]` U32 bitmask, so authoring \
         one would mean asking the caller to pack bits by hand. `mask_apply` \
         takes a bool mask and expands to `Select`, which infers and evaluates \
         identically; the packed form stays core because it is cheaper on the \
         wire, not because anything should write it.",
    ),
    (
        0x70,
        "`Op::Rng` draws from the per-fire ambient seed, so two runs of one \
         trace disagree. `rng`/`gumbel` emit `Op::RngKeyed` instead, whose \
         noise is a pure function of an explicit `[key, ctr]` state channel and \
         therefore replays bit-identically. Nothing should author a new one.",
    ),
];

/// The op tags a trace put in its container.
fn tags_of(traced: &Traced) -> BTreeSet<u8> {
    traced
        .container()
        .stages
        .iter()
        .flat_map(|stage| stage.ops.iter())
        .map(Op::tag)
        .collect()
}

/// Elementwise, comparison, reduction, shape and scan ops.
fn arithmetic() -> Traced {
    let logits = Channel::seeded([4, 8], dtype::f32);
    let idx = Channel::seeded([4], dtype::u32);
    let out = Channel::new([4], dtype::f32);
    let flags = Channel::new([8], dtype::bool);

    let mut builder = Builder::new(32_000, 16);
    builder.stage(Stage::Epilogue, || {
        let x = logits.take();
        let row = gather_row(&x, idx.take());

        // unary
        let u = add(
            add(exp(&row), log(abs(&row))),
            add(neg(recip(&row)), sign(&row)),
        );
        // binary + rem, and a cast to reach the integer ops
        let v = div(sub(mul(&u, 2.0f32), 1.0f32), 3.0f32);
        let w = max_elem(min_elem(&v, 9.0f32), -9.0f32);
        let counts = rem(cast(&w, dtype::u32), 5u32);

        // comparison and logic
        let hot = and(gt(&w, 0.0f32), lt(&w, 8.0f32));
        let cold = or(le(&w, 0.0f32), ge(&w, 8.0f32));
        let neither = not(and(&hot, &cold));
        let picked = select(ne(eq(&hot, &cold), &neither), &w, &v);

        // reduce and scan
        let total = reduce_sum(&picked);
        let span = sub(reduce_max(&picked), reduce_min(&picked));
        let best = reduce_argmax(&picked);
        let running = sub(cumsum(&picked), cumprod(add(&picked, 1.0f32)));

        // shape
        let square = reshape(&x, [8, 4]);
        let moved = transpose(&square);
        let spread = broadcast(&total, [8]);
        let ramp = iota(8);

        // gather/scatter
        let taken = gather(&picked, cast(&ramp, dtype::u32));
        let placed = scatter_set(&taken, cast(&ramp, dtype::u32), &picked);
        let summed = scatter_add(&placed, cast(&ramp, dtype::u32), &picked);

        // order and linear
        let (top_v, _top_i) = top_k(&summed, 4);
        let (sorted, _order) = sort_desc(&summed);
        let product = matmul(&moved, reshape(&x, [4, 8]));
        let cut = pivot_threshold(&sorted, rank_le(2));

        out.put(reshape(
            add(
                add(reduce_sum(&product), reduce_sum(top_v)),
                add(
                    add(span, cut),
                    add(cast(best, dtype::f32), cast(counts, dtype::f32)),
                ),
            ),
            [1],
        ));
        flags.put(gt(add(&spread, running), 0.0f32));
    });
    builder.build().unwrap()
}

/// Masking, sampling and the predicate forms `arithmetic` does not use.
fn masking() -> Traced {
    let state = Channel::from([7u32, 0]).named("rng");
    let positions = Channel::seeded([4], dtype::u32);
    let out = Channel::new([4, 8], dtype::f32);

    let mut builder = Builder::new(32_000, 16);
    builder.stage(Stage::Epilogue, || {
        let noise = rng(state.read(), [4, 8]);
        let jitter = gumbel(state.read(), [4, 8]);
        let scores = add(noise, jitter);

        let pos = positions.take();
        let causal = causal_mask(&pos, 8);
        let sliding = sliding_window_mask(&pos, 8, 4);
        let sunk = sink_window_mask(&pos, 8, 1, 4);
        let keep = and(causal, or(sliding, sunk));

        let masked = mask_apply(scores, keep);
        let softened = softmax(&masked);
        let by_mass = pivot_threshold(&softened, cummass_le(0.9));
        let by_prob = pivot_threshold(&softened, prob_ge(0.1));
        out.put(add(masked, broadcast(add(by_mass, by_prob), [4, 8])));
    });
    builder.build().unwrap()
}

/// Channel endpoints, intrinsics, and the two second-party call ops.
fn calls() -> Traced {
    let page_mask = Channel::seeded([4], dtype::u32);
    let out = Channel::new([1], dtype::i32);

    let mut builder = Builder::new(32_000, 16);
    builder.stage(Stage::OnAttnProj, || {
        intrinsics::kernel::attn_page_mask(page_mask.read());
    });
    builder.stage(Stage::Epilogue, || {
        let scores = intrinsics::kernel::envelope_dot(4);
        let logits = intrinsics::logits();
        out.put(cast(
            reduce_argmax(add(logits, broadcast(reduce_sum(scores), [32_000]))),
            dtype::i32,
        ));
    });
    builder.build().unwrap()
}

#[test]
fn every_op_is_authorable_or_says_why_not() {
    let mut reached = BTreeSet::new();
    for traced in [arithmetic(), masking(), calls()] {
        reached.extend(tags_of(&traced));
    }

    let exempt: BTreeSet<u8> = NOT_AUTHORABLE.iter().map(|(tag, _)| *tag).collect();

    let unreachable: Vec<&str> = OP_TABLE
        .iter()
        .filter(|spec| !reached.contains(&spec.tag) && !exempt.contains(&spec.tag))
        .map(|spec| spec.name)
        .collect();
    assert!(
        unreachable.is_empty(),
        "no DSL constructor emits these ops, and they are not in NOT_AUTHORABLE: {unreachable:?}.\n\
         Add a constructor to `value.rs`, or record here why the op must not be authored."
    );

    let stale: Vec<&str> = OP_TABLE
        .iter()
        .filter(|spec| reached.contains(&spec.tag) && exempt.contains(&spec.tag))
        .map(|spec| spec.name)
        .collect();
    assert!(
        stale.is_empty(),
        "NOT_AUTHORABLE claims these are unreachable, but a trace emitted them: {stale:?}"
    );
}

#[test]
fn the_exemption_list_describes_real_ops() {
    for (tag, reason) in NOT_AUTHORABLE {
        assert!(
            OP_TABLE.iter().any(|spec| spec.tag == *tag),
            "NOT_AUTHORABLE exempts tag {tag:#x}, which no longer has a row in OP_TABLE"
        );
        assert!(
            reason.len() > 40,
            "the exemption for tag {tag:#x} needs a reason, not a note"
        );
    }
}
