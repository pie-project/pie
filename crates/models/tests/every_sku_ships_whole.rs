//! Checks every catalog SKU names exactly one import/template/tokenizer
//! contract, its tp matches its name, and its trace is platform-consistent.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use checkpoint_dsl::Error;

#[test]
fn a_sku_name_states_the_world_its_row_ships() {
    let mut faults = Vec::new();

    for row in models::skus() {
        let (sku, tp) = (row.name.as_str(), row.recipe.tp);
        let named = match sku.rsplit_once("-tp") {
            Some((_, ranks)) => ranks.parse::<u32>().unwrap_or_else(|why| {
                panic!("`{sku}` ends in a world of `{ranks}` ranks, which is no number: {why}")
            }),
            None => 1,
        };
        if named != tp {
            faults.push(format!(
                "`{sku}` names a world of {named} rank(s) and its catalog row \
                 ships tp {tp}; the name a runtime selects by and the world it \
                 gets are the same fact"
            ));
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

#[test]
fn every_import_row_reads_the_checkpoint_it_is_handed() {
    let dir = scratch();
    let path = dir.join("holds-nothing.zt");
    write_a_checkpoint_of_one_stranger(&path);
    let src = ztensor::Source::open(&path).unwrap_or_else(|why| {
        panic!(
            "{}: the checkpoint just written does not open: {why}",
            path.display()
        )
    });

    let mut faults = Vec::new();
    let mut sharded = 0usize;
    for row in models::skus() {
        let (sku, tp) = (row.name.as_str(), row.recipe.tp);
        let refusal = match row.contract(&src, model_dsl::Platform::Cuda) {
            Ok(_) => {
                faults.push(format!(
                    "`{sku}` states a whole contract over a checkpoint holding \
                     one tensor no model reads, so its import table never asked \
                     the file what it holds"
                ));
                continue;
            }
            Err(Error::Missing(why)) => why.to_string(),
            Err(Error::Illegible { detail, .. }) => detail,
            Err(why @ Error::Incompatible { .. }) => {
                faults.push(format!(
                    "`{sku}` refuses a checkpoint that holds nothing it reads \
                     with `{why}`, and a file that states none of its planes \
                     is missing them, not storing them in another \
                     representation"
                ));
                continue;
            }
        };
        // A sharded row (tp > 1) refuses at the rank width before it reads
        // the checkpoint at all, which is a different refusal than this test checks.
        if tp > 1 {
            assert!(
                refusal.contains("WHOLE checkpoint"),
                "`{sku}` is built for {tp} ranks, so it should refuse at the \
                 width before it reads anything, and it refused with: {refusal}"
            );
            sharded += 1;
        } else if refusal.contains("WHOLE checkpoint") {
            faults.push(format!(
                "`{sku}` is a one-rank row and refused at the width: {refusal}"
            ));
        }
    }
    assert!(
        sharded > 0,
        "no import row is built for more than one rank, so the arm above is \
         dead — if the catalog lost its sharded rows, delete it"
    );

    drop(src);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

fn scratch() -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);

    let dir = std::env::temp_dir().join(format!(
        "model_import_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    dir
}

fn write_a_checkpoint_of_one_stranger(path: &Path) {
    let mut writer =
        ztensor::Writer::create(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    writer
        .add("a.tensor.no.model.in.this.catalog.reads", vec![1u64], ztensor::Leaf::U8, &[0u8])
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

/// **THE BLOCK DRAFTER'S TEXT TRACES** — the plan `qwen36-27b-dflash` builds
/// runs the validator in `trace_hybrid`'s `finish`, which is what says the
/// two-armed shape (a trunk guarded away from the draft rows, a drafter
/// reading the target's own head) is a plan the compiler will take at all.
///
/// It is also the regression for `Guard::common`: the trunk here runs inside
/// a split arm, and every attention layer inside it MERGES. Without a merge
/// coming back on the arm its siblings are on, the first `lora_correct` past
/// it panics as mixed arms, which is exactly how this plan failed before.
#[test]
fn the_block_drafters_plan_is_whole() {
    use model_dsl::Platform;
    let row = models::skus()
        .find(|row| row.recipe.text == "qwen36-27b-dflash")
        .expect("this build ships the block-drafter row");
    for platform in [Platform::Metal, Platform::Cuda] {
        let trace = (row.trace)(platform);
        assert!(
            !trace.nodes.is_empty(),
            "{platform:?}: the drafter's plan is empty"
        );
        let seams: Vec<&str> = trace.seams.iter().map(|s| s.seam.as_str()).collect();
        assert!(
            seams.iter().any(|s| s.contains("mtp")),
            "{platform:?}: the drafter plants no draft seam; seams are {seams:?}"
        );
    }
}

/// **THE DFLASH2 TEXT TRACES, AND CONVOLVES.** The v2 row builds the same
/// two-armed plan with the dynamic convolution around every sublayer of the
/// drafter — four `attention.block_dyn_conv` nodes a block, twenty in all —
/// and no masked read: every v2 layer is sliding and causal inside the block.
#[test]
fn the_dflash2_plan_is_whole_and_convolves() {
    use model_dsl::Platform;
    let row = models::skus()
        .find(|row| row.recipe.text == "qwen38-27b-dflash2")
        .expect("this build ships the DFlash2 row");
    for platform in [Platform::Metal, Platform::Cuda] {
        let trace = (row.trace)(platform);
        let convs = trace
            .nodes
            .iter()
            .filter(|n| matches!(&n.op, model_dsl::Operation::Attention(model_dsl::Attention::BlockDynConv { .. })))
            .count();
        assert_eq!(convs, 20, "{platform:?}: five blocks x two sublayers x two sides");
        let walks = trace
            .nodes
            .iter()
            .filter(|n| matches!(&n.op, model_dsl::Operation::Attention(model_dsl::Attention::SelectorWalk { .. })))
            .count();
        let topks = trace
            .nodes
            .iter()
            .filter(|n| matches!(&n.op, model_dsl::Operation::Layout(model_dsl::Layout::TopK { .. })))
            .count();
        assert_eq!((topks, walks), (1, 1), "{platform:?}: the selector reads the block out once");
        let seams: Vec<&str> = trace.seams.iter().map(|s| s.seam.as_str()).collect();
        assert!(seams.iter().any(|s| s.contains("mtp")), "{platform:?}: no draft seam; {seams:?}");
        // The facts a guest seeds the block from ride on the trace.
        let facts = trace.drafter.expect("the v2 text states its block drafter");
        assert_eq!((facts.rows, facts.mask_token, facts.bidirectional), (8, 248_070, false));
    }
}

/// **THE V1 TEXT STATES ITS BLOCK TOO**, and says it is bidirectional — its
/// last layer is full attention over the block, so a guest must bind a mask.
#[test]
fn the_v1_text_states_a_bidirectional_block_of_sixteen() {
    use model_dsl::Platform;
    let row = models::skus()
        .find(|row| row.recipe.text == "qwen36-27b-dflash")
        .expect("this build ships the block-drafter row");
    let trace = (row.trace)(Platform::Metal);
    let facts = trace.drafter.expect("the v1 text states its block drafter");
    assert_eq!((facts.rows, facts.mask_token, facts.bidirectional), (16, 248_070, true));
    // The A3B mixture carries the same shape with eight taps and its own mask id.
    let a3b = models::skus()
        .find(|row| row.recipe.text == "qwen36-35b-a3b-dflash")
        .expect("this build ships the A3B block-drafter row");
    let facts = (a3b.trace)(Platform::Metal).drafter.expect("the A3B text states its block drafter");
    assert_eq!((facts.rows, facts.mask_token, facts.bidirectional, facts.proposals_from), (16, 248_077, true, 1));
    // And an undrafted text states none.
    let plain = models::skus()
        .find(|row| row.recipe.text == "qwen38-27b" && row.recipe.weights.contains(&model_dsl::Dtype::U4g64))
        .expect("the plain row");
    assert!((plain.trace)(Platform::Metal).drafter.is_none());
}

/// **THE DSPARK TEXT**: v1's backbone with no convolution, a top-k and a
/// bigram walk for its readout, and a block of fifteen whose every row
/// proposes — the anchor row included.
#[test]
fn the_dspark_plan_is_whole_and_walks_a_bigram() {
    use model_dsl::Platform;
    let row = models::skus()
        .find(|row| row.recipe.text == "qwen38-27b-dspark")
        .expect("this build ships the DSpark row");
    let trace = (row.trace)(Platform::Metal);
    let count = |pred: &dyn Fn(&model_dsl::Operation) -> bool| trace.nodes.iter().filter(|n| pred(&n.op)).count();
    assert_eq!(count(&|op| matches!(op, model_dsl::Operation::Attention(model_dsl::Attention::BlockDynConv { .. }))), 0);
    assert_eq!(count(&|op| matches!(op, model_dsl::Operation::Layout(model_dsl::Layout::TopK { .. }))), 1);
    let walks: Vec<_> = trace
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            model_dsl::Operation::Attention(model_dsl::Attention::SelectorWalk { hp, first, .. }) => Some((*hp, *first)),
            _ => None,
        })
        .collect();
    assert_eq!(walks.len(), 1);
    assert_eq!(walks[0], (None, 0), "a bigram lattice walked from the anchor row");
    let facts = trace.drafter.expect("the DSpark text states its block drafter");
    assert_eq!(
        (facts.rows, facts.mask_token, facts.bidirectional, facts.proposals_from),
        (15, 248_200, true, 0)
    );
}


/// **THE SAME FOUR HOOKS IN A SECOND FAMILY.** gemma's text carries z-lab's
/// head for its mixture: six taps (six `fc` slices into the fusion), the v1
/// shape's one bidirectional layer (one non-causal masked read), a block of
/// sixteen whose mask id is 4, and no trunk plan guarded on anything but the
/// trunk's own rows.
#[test]
fn gemma_carries_the_block_drafter_too() {
    use model_dsl::Platform;
    let row = models::skus()
        .find(|row| row.recipe.text == "gemma4-26b-a4b-dflash")
        .expect("this build ships gemma's DFlash row");
    let trace = (row.trace)(Platform::Metal);
    let facts = trace.drafter.expect("gemma's text states its block drafter");
    assert_eq!(
        (facts.rows, facts.mask_token, facts.bidirectional, facts.proposals_from),
        (16, 4, true, 1)
    );
    let bidirectional = trace
        .nodes
        .iter()
        .filter(|n| {
            matches!(
                &n.op,
                model_dsl::Operation::Attention(model_dsl::Attention::Masked { causal: false, .. })
            )
        })
        .count();
    assert_eq!(bidirectional, 1, "the head's full layer is the one non-causal read");
    let plain = models::skus()
        .find(|row| row.recipe.text == "gemma4-26b-a4b")
        .expect("the plain row");
    assert!((plain.trace)(Platform::Metal).drafter.is_none());
}

/// **A THIRD FAMILY, AND THE FIRST HEAD OF ANOTHER GEOMETRY.** gpt-oss carries
/// z-lab's head: eight layers all full attention (eight non-causal reads over
/// a block of eight), 64 query heads at head dim 64 with biased projections.
#[test]
fn gpt_oss_carries_the_block_drafter_too() {
    use model_dsl::Platform;
    let row = models::skus()
        .find(|row| row.recipe.text == "gptoss-20b-dflash")
        .expect("this build ships gpt-oss's DFlash row");
    let trace = (row.trace)(Platform::Metal);
    let facts = trace.drafter.expect("gpt-oss's text states its block drafter");
    assert_eq!(
        (facts.rows, facts.mask_token, facts.bidirectional, facts.proposals_from),
        (8, 200_000, true, 1)
    );
    let bidirectional = trace
        .nodes
        .iter()
        .filter(|n| {
            matches!(
                &n.op,
                model_dsl::Operation::Attention(model_dsl::Attention::Masked { causal: false, .. })
            )
        })
        .count();
    assert_eq!(bidirectional, 8, "every layer of this head is full attention over the block");
}
