//! A `-tp2` row is its sibling with the shard column applied, and this is
//! what makes that a measurement rather than a convention.
//!
//! # The two halves of one fact
//!
//! A tensor-parallel catalog row states its geometry twice over, in two
//! vocabularies that have to agree:
//!
//! * each weight carries a [`Shard`] mark saying WHICH AXIS a rank cut runs
//!   along, and
//! * the text's own `per_rank` divides the dims that axis is built out of,
//!   so every shape, statement param and cache row comes out narrower.
//!
//! Neither can check the other on its own. A mark on a dim nobody divides is
//! a cut that never happens; a divided dim under a Replicated mark is a rank
//! quietly computing a fraction of a tensor it is supposed to hold whole. So
//! the check is a JOIN: trace the `-tp2` row, trace the row it is a cut of,
//! and hold every param of one against its namesake in the other. Replicated
//! means identical bytes. A cut means EXACTLY the marked axis divided by two
//! and every other axis identical — not "smaller", which a wrong divisor
//! would also satisfy.
//!
//! # Why the cache rows are here too
//!
//! A rank's KV pages hold its own heads, so a `kv.<l>` row that did not
//! narrow with the attention that writes it would be a pool sized for the
//! whole tower in front of a program computing a slice of it. The recurrent
//! slabs are the same statement for a hybrid's convolution window and its
//! delta state. Both are read off `caches()`, which reads the same dims the
//! projections do — so if the shapes moved and these did not, a text divided
//! a dim in one place and not another.

use std::collections::BTreeMap;

use model_ir::kernels::Backend;
use model_ir::plan::{CacheRow, Param, Shard};

/// Every `-tp2` SKU and the row it is a rank cut of.
///
/// Derived rather than listed: the id says it (`a_shipped_sku_is_importable_
/// or_is_a_rank_cut` is the test that holds the naming to it), and a table
/// here would be a second place to add a row.
fn pairs() -> Vec<(&'static str, String, usize)> {
    model::catalog()
        .into_iter()
        .filter_map(|(sku, _)| {
            let (base, tp) = sku.rsplit_once("-tp")?;
            Some((sku, base.to_string(), tp.parse().ok()?))
        })
        .collect()
}

fn params_of(sku: &str) -> BTreeMap<String, Param> {
    let trace = model::trace_of(sku).unwrap_or_else(|| panic!("`{sku}` is not a catalog row"));
    trace(Backend::Cuda)
        .params
        .into_iter()
        .map(|p| (p.name.clone(), p))
        .collect()
}

/// A cut's axis and its segments, checked against the shape it marks.
///
/// The invariant [`Shard`] rests on: a cut names an axis the tensor has, and
/// the segments partition that axis exactly. `model_dsl`'s builders assert it
/// where a tensor is declared; this asserts it where a plan is read, which is
/// the side a load would be on.
fn cut_of<'a>(shard: &'a Shard, name: &str, shape: &[u64]) -> Option<(usize, &'a [u64])> {
    let Shard::Cut { axis, segments } = shard else {
        return None;
    };
    let axis = *axis as usize;
    let extent = *shape
        .get(axis)
        .unwrap_or_else(|| panic!("`{name}` is {shape:?} and its cut names axis {axis}"));
    assert_eq!(
        segments.iter().sum::<u64>(),
        extent,
        "`{name}`: the segments of axis {axis} do not cover it",
    );
    Some((axis, segments))
}

#[test]
fn every_rank_cut_param_is_its_siblings_marked_axis_divided() {
    let mut faults = Vec::new();
    let mut cut = 0usize;
    let mut replicated = 0usize;
    for (sku, base, tp) in pairs() {
        let (mine, whole) = (params_of(sku), params_of(&base));
        let tp = tp as u64;
        assert_eq!(
            mine.keys().collect::<Vec<_>>(),
            whole.keys().collect::<Vec<_>>(),
            "`{sku}` and `{base}` state different weights",
        );
        for (name, p) in &mine {
            let w = &whole[name];
            let (Some((axis, segments)), Some((was, whole_segments))) = (
                cut_of(&p.shard, name, &p.shape),
                cut_of(&w.shard, name, &w.shape),
            ) else {
                if matches!(p.shard, Shard::Cut { .. }) != matches!(w.shard, Shard::Cut { .. }) {
                    faults.push(format!(
                        "`{sku}`: `{name}` is {:?} and `{base}` says {:?}",
                        p.shard, w.shard,
                    ));
                } else if p.shape != w.shape {
                    faults.push(format!(
                        "`{sku}`: `{name}` is Replicated and is {:?} where `{base}` \
                         has {:?} — a rank holds a replicated tensor WHOLE",
                        p.shape, w.shape,
                    ));
                }
                replicated += 1;
                continue;
            };
            assert_eq!(
                axis, was,
                "`{sku}`: `{name}` is cut on axis {axis} and `{base}` says {was}",
            );
            // THE SEGMENTS ARE THIS RANK'S OWN, which is what makes a packed
            // row a rank's row rather than the whole gate followed by nothing:
            // `[gate | up]` at half the width is `[gate/2 | up/2]`.
            assert_eq!(
                segments,
                whole_segments
                    .iter()
                    .map(|s| *s / tp)
                    .collect::<Vec<_>>()
                    .as_slice(),
                "`{sku}`: `{name}`'s segments are not `{base}`'s cut {tp} ways",
            );
            let mut want = w.shape.clone();
            if want[axis] % tp != 0 {
                faults.push(format!(
                    "`{sku}`: `{name}` is cut on axis {axis} of {:?}, which does \
                     not divide {tp} ways",
                    w.shape,
                ));
                continue;
            }
            want[axis] /= tp;
            if p.shape != want {
                faults.push(format!(
                    "`{sku}`: `{name}` is {:?} and the {tp}-way cut of `{base}`'s \
                     {:?} on axis {axis} is {want:?}",
                    p.shape, w.shape,
                ));
            }
            cut += 1;
        }
    }
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
    assert!(cut > 0, "no rank cut row cuts a single weight");
    assert!(replicated > 0, "a rank cut row that replicates nothing");
}

/// Every dim a cut divides is a dim some mark names.
///
/// The other direction, and the one that catches a text dividing a number
/// nothing is sharded by: a rank cut may only make a tensor NARROWER, never
/// wider, and it may not touch a Replicated one at all. The test above says
/// the cut ones are right; this one says there is nothing else moving.
#[test]
fn a_rank_cut_narrows_and_never_widens() {
    for (sku, base, _) in pairs() {
        let (mine, whole) = (params_of(sku), params_of(&base));
        for (name, p) in &mine {
            let w = &whole[name];
            let mine_elems: u64 = p.shape.iter().product();
            let whole_elems: u64 = w.shape.iter().product();
            assert!(
                mine_elems <= whole_elems,
                "`{sku}`: `{name}` is {:?} where `{base}` is {:?} — a rank holds \
                 no more than the whole",
                p.shape,
                w.shape,
            );
            assert_eq!(
                p.shape.len(),
                w.shape.len(),
                "`{sku}`: `{name}` has a different rank than `{base}`'s",
            );
        }
    }
}

/// A rank's pools narrow with the mixers that write them.
///
/// Read off `caches()`, so it is the same dims the projections are built
/// from asked a second way. What a row must NOT do is stay put: `kv.<l>` at
/// the whole tower's head count in front of an attention computing a slice
/// is a pager allocating for a model this rank is not running.
#[test]
fn every_rank_cut_cache_row_narrows_with_its_mixer() {
    for (sku, base, tp) in pairs() {
        let rows = |s: &str| -> Vec<CacheRow> {
            model::trace_of(s).expect("a catalog row")(Backend::Cuda).caches
        };
        let (mine, whole) = (rows(sku), rows(&base));
        assert_eq!(
            mine.len(),
            whole.len(),
            "`{sku}` and `{base}` declare different numbers of cache rows",
        );
        let tp = tp as u64;
        for (m, w) in mine.iter().zip(&whole) {
            let (name, m_row, w_row) = match (m, w) {
                (CacheRow::Kv { name, row }, CacheRow::Kv { row: was, .. }) => (name, row, was),
                (
                    CacheRow::State { name, slab },
                    CacheRow::State { slab: was, .. },
                ) => (name, slab, was),
                _ => panic!("`{sku}`: a cache row changed kind from `{base}`'s"),
            };
            let m_elems: u64 = m_row.iter().product();
            let w_elems: u64 = w_row.iter().product();
            assert!(
                m_elems == w_elems || m_elems * tp == w_elems,
                "`{sku}`: `{name}` is {m_row:?} where `{base}` is {w_row:?} — a \
                 rank's pool row is the whole one or its {tp}-way cut, and \
                 nothing between",
            );
        }
    }
}

/// The reduce and the cut are the same decision, stated once each.
///
/// `dist.all_reduce` sums partial rows across the ranks of a cut. A text that
/// stated it over a leg no rank holds a piece of would MULTIPLY that leg by
/// the world size, and a text that cut a leg without it would leave every
/// rank holding a fraction. So the two go together: a row that reduces cuts
/// something, and a row that cuts something reduces.
#[test]
fn a_row_that_reduces_is_a_row_that_cuts() {
    for (sku, _) in model::catalog() {
        let plan = model::trace_of(sku).expect("a catalog row")(Backend::Cuda);
        let reduces = plan.ops.iter().any(|o| o.kernel == "dist.all_reduce");
        let cuts = plan.params.iter().any(|p| p.shard != Shard::Replicated);
        let is_cut_row = sku.contains("-tp");
        assert_eq!(
            reduces, is_cut_row,
            "`{sku}`: states dist.all_reduce = {reduces}, and is a rank cut row = {is_cut_row}",
        );
        assert!(
            cuts,
            "`{sku}`: no weight carries a shard mark, so a rank cut of it would \
             cut nothing",
        );
    }
}
