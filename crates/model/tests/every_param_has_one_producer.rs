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

use model_dsl::load::Source;
use model_dsl::Plane;

/// Every name a `Source` reads out of the checkpoint.
fn cites(source: &Source) -> Vec<&str> {
    match source {
        Source::Copy(n)
        | Source::PlusOne(n)
        | Source::ScalarOf(n)
        | Source::Deinterleave(n, _)
        | Source::Squeeze(n, _) => vec![n.as_str()],
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
fn a_shipped_sku_is_importable_or_is_a_rank_cut() {
    // Import keys on the bytes, serving keys on the layout, and the two differ
    // by exactly the tensor-parallel rows: a `-tp2` plan is the same weights
    // read by two ranks, so it ships no import of its own and must resolve to
    // the sibling it shares an artifact with. Anything else with no import row
    // is a SKU a user can select and never obtain.
    let importable: BTreeSet<&str> = model::imports().into_iter().map(|r| r.sku).collect();
    let shipped: BTreeSet<&str> = model::catalog().into_iter().map(|(n, _)| n).collect();

    let mut faults = Vec::new();
    for sku in &shipped {
        if importable.contains(sku) {
            continue;
        }
        let Some(base) = sku.rsplit_once("-tp").map(|(head, _)| head) else {
            faults.push(format!("`{sku}` ships, imports nothing, and is not a rank cut"));
            continue;
        };
        if !importable.contains(base) {
            faults.push(format!(
                "`{sku}` is a rank cut of `{base}`, which is itself not importable"
            ));
        }
    }
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
