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
        let seams: Vec<&str> = trace.seams.iter().map(|s| s.seam.as_str()).collect();
        assert!(seams.iter().any(|s| s.contains("mtp")), "{platform:?}: no draft seam; {seams:?}");
    }
}
