//! The import table and the plan are two halves of one statement, and this is
//! where they are made to agree.
//!
//! A plan's `params` column is the DEMAND: every weight the forward pass will
//! bind, at the shape and repr it will bind it. An `Import` is the SUPPLY:
//! where each of those comes from in a checkpoint. Nothing in the type system
//! joins them -- one is keyed by `Param::name`, the other by `Row::target`,
//! and both are `String`. So the join is checked here instead, and it is
//! checked as a BIJECTION rather than as coverage: a param with no producer is
//! a weight nothing can fill, and a producer with no param is bytes moved for
//! a weight nobody reads. Both are faults, and the second is the one a
//! coverage-only test would let through.
//!
//! This costs no I/O and reads no checkpoint. That is the point of doing it
//! here: the six families' tables are proved before a byte is moved, where
//! today the same mistake surfaces as a load-time refusal on a machine that
//! already spent an hour downloading.

use std::collections::BTreeSet;

use model_dsl::Plane;
use model_dsl::load::Source;

/// Every name a `Source` reads out of the checkpoint.
fn cites(source: &Source) -> Vec<&str> {
    match source {
        Source::Copy(n) | Source::Deinterleave(n, _, _) | Source::Squeeze(n, _) => vec![n.as_str()],
        Source::Pack(parts) | Source::Stack(parts) => parts.iter().flat_map(cites).collect(),
    }
}

#[test]
fn every_param_has_one_producer() {
    let catalog = model::catalog();
    let mut faults = Vec::new();

    for row in model::imports() {
        let Some((_, trace)) = catalog.iter().find(|(n, _)| *n == row.sku) else {
            faults.push(format!(
                "`{}` [{}] imports a SKU the catalog does not ship",
                row.sku, row.base
            ));
            continue;
        };

        // The plane cannot matter here -- a param column is the model's
        // weights, and a weight is not a property of the device that reads it.
        // Tracing on one plane is therefore enough, and `planes_do_not_move_a_param`
        // below is what holds that true.
        let plan = trace(Plane::Cuda);
        let demand: BTreeSet<&str> = plan.params.iter().map(|p| p.name.as_str()).collect();

        let table = (row.make)();
        let supply: BTreeSet<&str> = table.rows.iter().map(|r| r.target.as_str()).collect();

        for name in demand.difference(&supply) {
            faults.push(format!(
                "`{}` [{}]: the plan binds `{name}` and the import table produces nothing for it",
                row.sku, row.base
            ));
        }
        for name in supply.difference(&demand) {
            faults.push(format!(
                "`{}` [{}]: the import table produces `{name}` and the plan binds no such weight",
                row.sku, row.base
            ));
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

#[test]
fn planes_do_not_move_a_param() {
    // The claim `every_param_has_one_producer` rests on, stated on its own so
    // that the day a family's declaration branches on its plane, the failure
    // names that fact rather than showing up as a phantom import gap.
    for (sku, trace) in model::catalog() {
        let cuda = trace(Plane::Cuda);
        let metal = trace(Plane::Metal);
        assert_eq!(
            cuda.params, metal.params,
            "`{sku}` declares different weights per plane; an artifact can no longer be one file"
        );
        assert_eq!(
            cuda.caches, metal.caches,
            "`{sku}` declares different caches per plane"
        );
    }
}

#[test]
fn a_producer_reads_each_source_name_once() {
    // Two rows citing one checkpoint tensor is legal and load-bearing --
    // Gemma's GGUF leg reads `post_per_layer_norm.weight` as both a norm and
    // a scalar. Two rows citing it the SAME way is not: it is the same bytes
    // written twice under different names, which is a copy-paste fault every
    // time.
    let mut faults = Vec::new();
    for row in model::imports() {
        let table = (row.make)();
        let mut seen: Vec<(&str, String)> = Vec::new();
        for r in &table.rows {
            let shape = format!("{:?}", std::mem::discriminant(&r.source));
            for name in cites(&r.source) {
                if let Some((first, _)) = seen
                    .iter()
                    .find(|(n, k)| *n == name && *k == shape)
                    .map(|(n, k)| (*n, k))
                {
                    faults.push(format!(
                        "`{}` [{}]: `{first}` is read the same way by two rows",
                        row.sku, row.base
                    ));
                }
                seen.push((name, shape.clone()));
            }
        }
    }
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

#[test]
fn every_shipped_sku_is_importable() {
    // THIS TEST HAD AN EXEMPTION AND THE EXEMPTION WAS THE BUG. It read "a
    // shipped SKU is importable OR is a rank cut", on the argument that a
    // `-tp2` plan is the same weights read by two ranks and so needs no import
    // of its own. The bytes half was right; the conclusion was not. `import_of`
    // is how a load reaches a production table at all, so a row that named no
    // import was a row no load could produce — a SKU a user can select and
    // never obtain, which is precisely what the exemption was written to rule
    // out. A `-tp2` row now names its sibling's table and the cut is applied by
    // `model::produce` from the plan's own shard column, so there is nothing
    // left to exempt.
    let importable: BTreeSet<&str> = model::imports().into_iter().map(|r| r.sku).collect();
    let unobtainable: Vec<&str> = model::catalog()
        .into_iter()
        .map(|(n, _)| n)
        .filter(|sku| !importable.contains(sku))
        .collect();
    assert!(
        unobtainable.is_empty(),
        "these SKUs ship and import nothing, so they can be selected and never \
         obtained: {unobtainable:?}",
    );
}

/// A rank cut row's table IS its sibling's, and this is what makes that a
/// measurement rather than a promise.
///
/// The two rows are traced at different degrees, so their PLANS differ in
/// every sharded shape — but a checkpoint holds the same bytes however a
/// deployment cuts them, so the production tables must not differ at all. If
/// one ever did, the cut would have been stated twice: once in the shard
/// column `produce` reads and once in a source expression, with nothing
/// holding the two together.
#[test]
fn a_rank_cut_row_imports_exactly_its_siblings_table() {
    let rows = model::imports();
    let mut pairs = 0usize;
    for row in &rows {
        let Some((base, _)) = row.sku.rsplit_once("-tp") else {
            continue;
        };
        let sibling = rows
            .iter()
            .find(|r| r.sku == base && r.base == row.base)
            .unwrap_or_else(|| {
                panic!(
                    "`{}` [{}] is a rank cut of `{base}`, which ships no `{}` import",
                    row.sku, row.base, row.base
                )
            });
        assert_eq!(
            (row.make)(),
            (sibling.make)(),
            "`{}` [{}] and `{base}` read the checkpoint differently; a rank cut \
             is a way of READING the same bytes",
            row.sku,
            row.base,
        );
        pairs += 1;
    }
    assert!(pairs > 0, "no rank cut row ships an import to compare");
}

/// A checkpoint is never identified as one rank of a world.
///
/// THE HAZARD THE LINE ABOVE CREATES, held shut. Now that a `-tp2` row reads
/// its sibling's table verbatim, the two rows are indistinguishable to
/// anything that asks the tensors — which is exactly right about the bytes
/// and would make [`model::identify`] answer `Ambiguous` for every family
/// that ships one, turning every load of every MoE checkpoint into a refusal.
/// It does not, because a plan that states `dist.all_reduce` is one rank of a
/// world and a checkpoint has no peer.
///
/// The reader is built out of the sibling's own production table — every
/// source name it cites, with the `embed` source at the serving row's
/// vocabulary — so this asks the question with the checkpoint the pair was
/// written for and no other.
///
/// # Why the answer is checked by MEMBERSHIP and not by equality
///
/// Because one family already answers with two, for a reason that has
/// nothing to do with ranks: gpt-oss's 20b and 120b rows are the same tensor
/// names at the same vocabulary and differ only in depth, so every source the
/// 24-layer row reads is one the 36-layer row also holds and a 120b
/// checkpoint satisfies both. That is a real catalog defect and it is a
/// PRE-EXISTING one — both rows were importable before any of this — so
/// asserting `Ok(sibling)` here would be this test failing for someone else's
/// bug. What it must say is the thing the rank cut put at risk: whatever
/// `identify` answers, the answer is a whole model and never a rank.
#[test]
fn a_rank_cut_is_never_what_a_checkpoint_is() {
    let mut asked = 0usize;
    for row in model::imports() {
        let Some((sibling, _)) = row.sku.rsplit_once("-tp") else {
            continue;
        };
        let table = model::import_of(sibling, row.base)
            .unwrap_or_else(|| panic!("`{sibling}` ships no `{}` import", row.base));
        let vocab = u64::from(
            model::serve::row(sibling)
                .unwrap_or_else(|| panic!("`{sibling}` has no serving row"))
                .vocab,
        );
        let held: BTreeSet<String> = table
            .rows
            .iter()
            .flat_map(|r| cites(&r.source))
            .map(str::to_string)
            .collect();
        let embed: BTreeSet<String> = table
            .rows
            .iter()
            .filter(|r| r.target == "embed")
            .flat_map(|r| cites(&r.source))
            .map(str::to_string)
            .collect();
        let shape_of = |name: &str| -> Option<Vec<u64>> {
            if !held.contains(name) {
                return None;
            }
            Some(if embed.contains(name) {
                vec![vocab, 1]
            } else {
                vec![1]
            })
        };
        let answered: Vec<&str> = match model::identify(&shape_of) {
            Ok(sku) => vec![sku],
            Err(model::Unmatched::Ambiguous { skus }) => skus,
            Err(why) => panic!("a `{}` checkpoint of `{sibling}`: {why}", row.base),
        };
        assert!(
            !answered.contains(&row.sku),
            "a `{}` checkpoint of `{sibling}` identifies as `{}`, which is ONE \
             RANK of a world — the tensors cannot say which cut to deploy",
            row.base,
            row.sku,
        );
        assert!(
            answered.contains(&sibling),
            "a `{}` checkpoint of `{sibling}` identifies as {answered:?}",
            row.base,
        );
        asked += 1;
    }
    assert!(asked > 0, "no rank cut row to ask about");
}

/// EVERY VERB A REGISTERED TABLE SPELLS IS ONE `model::produce` PERFORMS.
///
/// A production table is only ever checked by being RUN, and a table nothing
/// selects is a table nobody has checked. Gemma's GGUF leg was that: 90 rows
/// of `blk.{l}.*` spellings behind three independent walls — every driver
/// picks a `safetensors*` base by name, `model::snapshot` reads safetensors
/// and nothing else, and its E4B arm said `scalar_of`, a verb the
/// interpreter answered with an unconditional refusal. The leg and the verb
/// are both gone; this is what stops the pair coming back.
///
/// THE READER ANSWERS NOTHING, and that is the discrimination. Every verb
/// this interpreter performs fetches its operands first, so an empty
/// checkpoint takes each of them to `Fault::Absent` — the checkpoint has no
/// such tensor, which is true and is not this test's business. A verb the
/// interpreter does NOT perform refuses BEFORE it fetches, so it is the only
/// thing that can answer `Fault::Refused` here, whatever its operands.
///
/// ONE ROW AT A TIME, because `produce` stops at the first fault and every
/// row of an empty run has one. Each source is handed over as an `Import` of
/// its own, with no `params`, so `cut` never runs and the answer is the
/// interpreter's alone.
#[test]
fn every_registered_verb_is_one_the_interpreter_runs() {
    let nothing = |_: &str| None;
    let mut asked = 0;
    for row in model::imports() {
        let table = (row.make)();
        for one in &table.rows {
            let alone = model_dsl::load::Import {
                base: table.base,
                rows: vec![one.clone()],
            };
            let why = model::produce::produce(&alone, &[], 0, &nothing)
                .expect_err("an empty checkpoint satisfies no row");
            assert!(
                !matches!(why.fault, model::produce::Fault::Refused { .. }),
                "`{}` of `{}` ({}) states a verb this interpreter refuses \
                 whatever it is handed: {why}. Either the verb gets a reading \
                 or the row goes — a table that cannot run has never been \
                 checked against a file.",
                one.target,
                row.sku,
                row.base,
            );
            asked += 1;
        }
    }
    assert!(asked > 0, "no import row to ask about");
}
