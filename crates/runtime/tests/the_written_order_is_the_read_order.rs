//! **The order `pie model import` lays a serving artifact's planes in is the
//! order a boot reads them.**
//!
//! `runtime::engine::load::sequence` derives it from the trace, because the
//! import has no compiled load plan to read the pairings off — the plan would
//! be compiled against the artifact it is about to write. What this holds is
//! the two things that derivation can silently get wrong.
//!
//! **It can DROP a plane.** The ranking answers param INDICES and the lookup
//! back to names is a `filter_map`, so an index the trace does not hold
//! disappears without a word. An artifact missing a plane from its sequence is
//! still readable — `serving::sequence` is derived from offsets and the object
//! is still in the file — so nothing downstream would say anything; the plane
//! would simply sit outside the hot run forever.
//!
//! **It can NAME one twice.** A duplicate makes the import's sort key
//! ambiguous and puts one object at two ranks, which the writer resolves by
//! writing it once, at whichever came first.
//!
//! Both platforms, because a placement is resolved per shell and the ORDER
//! must not be: a repack moves no value, so the two shells' rankings are
//! byte-identical spans — `engine_cuda`'s own
//! `a_tiled_row_and_a_row_major_one_rank_to_the_same_spans` measures it. A row
//! whose sequence differed between them would be one whose artifact could not
//! be written for one shell and read by the other.
#![cfg(feature = "_engine-cuda")]

use std::collections::BTreeSet;

use model_ir::Platform;

#[test]
fn every_sku_names_each_of_its_ranked_planes_exactly_once() {
    let mut walked = 0usize;
    let mut faults: Vec<String> = Vec::new();
    for (sku, _, _, _) in models::catalog() {
        for platform in [Platform::Cuda, Platform::Metal] {
            let Ok(trace) = runtime::engine::load::trace(sku, platform) else {
                continue;
            };
            let Some(order) = runtime::engine::load::sequence(&trace) else {
                faults.push(format!("{sku} states no sequence at {platform:?}"));
                continue;
            };
            walked += 1;
            let held: BTreeSet<&str> = trace.params.iter().map(|p| p.name.as_str()).collect();
            let mut seen: BTreeSet<&str> = BTreeSet::new();
            for name in &order {
                if !held.contains(name.as_str()) {
                    faults.push(format!(
                        "{sku} at {platform:?} sequences {name}, which is not a param of \
                         its trace"
                    ));
                }
                if !seen.insert(name.as_str()) {
                    faults.push(format!("{sku} at {platform:?} sequences {name} twice"));
                }
            }
            // Nothing is dropped: a `filter_map` losing an image would show
            // here and nowhere else.
            if order.len() != seen.len() {
                faults.push(format!(
                    "{sku} at {platform:?} sequences {} names of which {} are distinct",
                    order.len(),
                    seen.len(),
                ));
            }
        }
    }
    assert!(
        walked >= 40,
        "the sequence was derived for only {walked} (sku, platform) pairs"
    );
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
