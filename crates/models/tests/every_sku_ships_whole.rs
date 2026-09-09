use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use checkpoint_dsl::Error;

#[test]
fn every_sku_ships_whole_every_case() {
    a_sku_name_states_the_world_its_row_ships();
    every_import_row_reads_the_checkpoint_it_is_handed();
    the_block_drafters_plan_is_whole();
    the_dflash2_plan_is_whole_and_convolves();
    the_v1_text_states_a_bidirectional_block_of_sixteen();
    the_dspark_plan_is_whole_and_walks_a_bigram();
    gemma_carries_the_block_drafter_too();
    gpt_oss_carries_the_block_drafter_too();
}

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
        .add(
            "a.tensor.no.model.in.this.catalog.reads",
            vec![1u64],
            ztensor::Leaf::U8,
            &[0u8],
        )
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

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
            .filter(|n| {
                matches!(
                    &n.op,
                    model_dsl::Operation::Attention(model_dsl::Attention::BlockDynConv { .. })
                )
            })
            .count();
        assert_eq!(
            convs, 20,
            "{platform:?}: five blocks x two sublayers x two sides"
        );
        let walks = trace
            .nodes
            .iter()
            .filter(|n| {
                matches!(
                    &n.op,
                    model_dsl::Operation::Attention(model_dsl::Attention::SelectorWalk { .. })
                )
            })
            .count();
        let topks = trace
            .nodes
            .iter()
            .filter(|n| {
                matches!(
                    &n.op,
                    model_dsl::Operation::Layout(model_dsl::Layout::TopK { .. })
                )
            })
            .count();
        assert_eq!(
            (topks, walks),
            (1, 1),
            "{platform:?}: the selector reads the block out once"
        );
        let seams: Vec<&str> = trace.seams.iter().map(|s| s.seam.as_str()).collect();
        assert!(
            seams.iter().any(|s| s.contains("mtp")),
            "{platform:?}: no draft seam; {seams:?}"
        );
        let facts = trace.drafter.expect("the v2 text states its block drafter");
        assert_eq!(
            (facts.rows, facts.mask_token, facts.bidirectional),
            (8, 248_070, false)
        );
    }
}

fn the_v1_text_states_a_bidirectional_block_of_sixteen() {
    use model_dsl::Platform;
    let row = models::skus()
        .find(|row| row.recipe.text == "qwen36-27b-dflash")
        .expect("this build ships the block-drafter row");
    let trace = (row.trace)(Platform::Metal);
    let facts = trace.drafter.expect("the v1 text states its block drafter");
    assert_eq!(
        (facts.rows, facts.mask_token, facts.bidirectional),
        (16, 248_070, true)
    );
    let a3b = models::skus()
        .find(|row| row.recipe.text == "qwen36-35b-a3b-dflash")
        .expect("this build ships the A3B block-drafter row");
    let facts = (a3b.trace)(Platform::Metal)
        .drafter
        .expect("the A3B text states its block drafter");
    assert_eq!(
        (
            facts.rows,
            facts.mask_token,
            facts.bidirectional,
            facts.proposals_from
        ),
        (16, 248_077, true, 1)
    );
    let plain = models::skus()
        .find(|row| {
            row.recipe.text == "qwen38-27b" && row.recipe.weights.contains(&model_dsl::Dtype::U4g64)
        })
        .expect("the plain row");
    assert!((plain.trace)(Platform::Metal).drafter.is_none());
}

fn the_dspark_plan_is_whole_and_walks_a_bigram() {
    use model_dsl::Platform;
    let row = models::skus()
        .find(|row| row.recipe.text == "qwen38-27b-dspark")
        .expect("this build ships the DSpark row");
    let trace = (row.trace)(Platform::Metal);
    let count = |pred: &dyn Fn(&model_dsl::Operation) -> bool| {
        trace.nodes.iter().filter(|n| pred(&n.op)).count()
    };
    assert_eq!(
        count(&|op| matches!(
            op,
            model_dsl::Operation::Attention(model_dsl::Attention::BlockDynConv { .. })
        )),
        0
    );
    assert_eq!(
        count(&|op| matches!(
            op,
            model_dsl::Operation::Layout(model_dsl::Layout::TopK { .. })
        )),
        1
    );
    let walks: Vec<_> = trace
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            model_dsl::Operation::Attention(model_dsl::Attention::SelectorWalk {
                hp,
                first,
                ..
            }) => Some((*hp, *first)),
            _ => None,
        })
        .collect();
    assert_eq!(walks.len(), 1);
    assert_eq!(
        walks[0],
        (None, 0),
        "a bigram lattice walked from the anchor row"
    );
    let facts = trace
        .drafter
        .expect("the DSpark text states its block drafter");
    assert_eq!(
        (
            facts.rows,
            facts.mask_token,
            facts.bidirectional,
            facts.proposals_from
        ),
        (15, 248_200, true, 0)
    );
}

fn gemma_carries_the_block_drafter_too() {
    use model_dsl::Platform;
    let row = models::skus()
        .find(|row| row.recipe.text == "gemma4-26b-a4b-dflash")
        .expect("this build ships gemma's DFlash row");
    let trace = (row.trace)(Platform::Metal);
    let facts = trace
        .drafter
        .expect("gemma's text states its block drafter");
    assert_eq!(
        (
            facts.rows,
            facts.mask_token,
            facts.bidirectional,
            facts.proposals_from
        ),
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
    assert_eq!(
        bidirectional, 1,
        "the head's full layer is the one non-causal read"
    );
    let plain = models::skus()
        .find(|row| row.recipe.text == "gemma4-26b-a4b")
        .expect("the plain row");
    assert!((plain.trace)(Platform::Metal).drafter.is_none());
}

fn gpt_oss_carries_the_block_drafter_too() {
    use model_dsl::Platform;
    let row = models::skus()
        .find(|row| row.recipe.text == "gptoss-20b-dflash")
        .expect("this build ships gpt-oss's DFlash row");
    let trace = (row.trace)(Platform::Metal);
    let facts = trace
        .drafter
        .expect("gpt-oss's text states its block drafter");
    assert_eq!(
        (
            facts.rows,
            facts.mask_token,
            facts.bidirectional,
            facts.proposals_from
        ),
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
    assert_eq!(
        bidirectional, 8,
        "every layer of this head is full attention over the block"
    );
}
