//! Lowerings, kept by the fire shape that produced them.
//!
//! # Why
//!
//! `lower_step` is a **pure function of the plan and the rows**, and a decode
//! has exactly one row shape. Serving re-derived it anyway, once per token:
//! `binding::text` rebuilt the `ForwardPlan` and `lower` walked the whole
//! trace to produce the same 228 launches with the same grids in the same
//! order. Nothing about a decode at position 129 differs from one at position
//! 128 — the position lives in the fire TABLES, which are device data and no
//! part of a lowering.
//!
//! Measured on Llama-3.2-1B-Instruct-4bit over an M1 Max, per decode:
//! `text` 0.21 ms and `lower` 0.60 ms, against a 4.9 ms step. That is 17 % of
//! a token spent recomputing a constant, and it is most of the fixed cost
//! that separates this driver's decode from mlx-lm's — the GPU work either
//! side of it is within noise of each other.
//!
//! # What the key has to be
//!
//! Everything `lower` reads: the rows, and the plan. The rows are hashed
//! directly. The plan is *not* — it is derived from `(row, class, binding)`,
//! and `row`/`binding` are fixed for as long as a model is loaded, so the
//! class stands in for the plan and [`Lowerings::clear`] is what a load
//! calls. A cache that outlived a model reload would serve the old
//! architecture's graph over the new one's weights, which is why the clear is
//! a method and not a comment.
//!
//! # What it does not do
//!
//! Grow without bound. Every distinct prefill length is a distinct row
//! vector, and a deployment sees as many of those as it sees prompt lengths.
//! Past [`CAP`] shapes the map is emptied wholesale rather than evicted one
//! at a time: the entry that matters is the decode's, it is re-derived in
//! 0.8 ms, and an LRU would be machinery to save that once per hundreds of
//! prefills.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use model_compiler::lower::{Lowered, Row};
use model_ir::trace::ForwardPlan;

use super::frame::{Step, Unbridged, rows_of};
use model_ir::trace::FireClass;

/// How many distinct fire shapes are kept before the map is emptied.
pub const CAP: usize = 64;

/// One fire shape's plan-derived pair: the lowering, and the runtime-stream
/// map the resolver answers `Arg::Named` through.
///
/// Kept together because both are pure functions of the `ForwardPlan`, and
/// this driver DROPS the plan once the lowering exists — the stream map is
/// the one other thing a fire needs from it (`super::runtime`), so it is
/// derived on the same miss and cached beside the thing it serves.
#[derive(Debug)]
pub struct Planned {
    /// The flat launch list `lower` produced.
    pub lowered: Lowered,
    /// Value id → fire table, from the plan's runtime table.
    pub streams: super::runtime::Streams,
}

/// Lowerings, kept by the fire shape that produced them.
#[derive(Debug, Default)]
pub struct Lowerings {
    by_shape: HashMap<u64, Planned>,
    /// How many lowerings have been derived, for the test that asks whether
    /// the cache is a cache.
    lowered: usize,
}

impl Lowerings {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// This step's lowering, derived if there is not one.
    ///
    /// `plan` is called **only on a miss** — building the `ForwardPlan` is
    /// a quarter of what this exists to skip.
    ///
    /// # Errors
    ///
    /// As [`rows_of`] and [`model_compiler::lower::lower`].
    pub fn for_step<E>(
        &mut self,
        class: FireClass,
        step: &Step<'_>,
        plan: impl FnOnce() -> Result<ForwardPlan, E>,
    ) -> Result<&Planned, Miss<E>> {
        let rows = rows_of(step).map_err(|why| Miss::Lower(Unbridged::Step(why)))?;
        let key = fingerprint(class, &rows);
        if let std::collections::hash_map::Entry::Vacant(slot) = self.by_shape.entry(key) {
            let plan = plan().map_err(Miss::Plan)?;
            let lowered = model_compiler::lower::lower(
                &plan,
                &rows,
                model_compiler::lower::Fire {
                    // As `lower_step`: this driver re-encodes every step, so
                    // no fire is replayed across a different row split.
                    captures_across_splits: false,
                },
            )
            .map_err(|why| Miss::Lower(Unbridged::Uncovered(format!("{why:?}"))))?;
            // Beside the lowering, because the plan is about to be dropped
            // and the resolver will still need its runtime table.
            let streams = super::runtime::Streams::of(&plan);
            slot.insert(Planned { lowered, streams });
            self.lowered += 1;
        }
        // Emptied AFTER the insert, and never before returning: the caller is
        // about to borrow what was just put in.
        if self.by_shape.len() > CAP {
            let keep = self.by_shape.remove(&key);
            self.by_shape.clear();
            if let Some(keep) = keep {
                self.by_shape.insert(key, keep);
            }
        }
        Ok(&self.by_shape[&key])
    }

    /// How many lowerings this cache has derived.
    ///
    /// The number that says whether reuse is happening: a hundred decodes
    /// over one deployment should move it by one.
    #[must_use]
    pub fn lowered(&self) -> usize {
        self.lowered
    }

    /// Forget every lowering.
    ///
    /// **For a model load.** A lowering is the graph of the text the loaded
    /// row named, and a new row names a different one.
    pub fn clear(&mut self) {
        self.by_shape.clear();
    }
}

/// Why a step did not reach a lowering.
///
/// Two sources, kept apart: the caller's own refusal to build a plan, and
/// this crate's refusal to lower one. Merging them would make a checkpoint
/// the row declines to serve indistinguishable from a text the lowering
/// cannot flatten.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Miss<E> {
    /// The caller could not build a plan for this fire.
    Plan(E),
    /// The step did not lower.
    Lower(Unbridged),
}

/// What a lowering is only valid for.
#[must_use]
fn fingerprint(class: FireClass, rows: &[Row]) -> u64 {
    let mut hasher = DefaultHasher::new();
    // The class is a stand-in for the PLAN, which is a pure function of
    // `(row, class, binding)` and whose other two are fixed for a load.
    (class as u8).hash(&mut hasher);
    rows.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trace with no ops, which lowers to no launches. What is under test
    /// is the CACHE, and a real plan would only make the misses slower.
    fn empty_plan() -> ForwardPlan {
        ForwardPlan {
            family: "test".to_string(),
            values: Vec::new(),
            ops: Vec::new(),
            depth_window: false,
            seams: Vec::new(),
            runtime: Vec::new(),
        }
    }

    fn step<'a>(token_ids: &'a [u32], qo_indptr: &'a [u32]) -> Step<'a> {
        Step {
            token_ids,
            qo_indptr,
            ..Step::default()
        }
    }

    /// Two rows that differ in one bit are two lowerings.
    #[test]
    fn a_different_row_shape_is_a_different_key() {
        let a = Row {
            samples: true,
            ..Row::default()
        };
        let b = Row::default();
        assert_ne!(
            fingerprint(FireClass::Decode, &[a]),
            fingerprint(FireClass::Decode, &[b])
        );
        assert_ne!(
            fingerprint(FireClass::Decode, &[a]),
            fingerprint(FireClass::Prefill, &[a])
        );
        assert_eq!(
            fingerprint(FireClass::Decode, &[a]),
            fingerprint(FireClass::Decode, &[a])
        );
    }

    /// A hundred decodes derive ONE lowering, and it is the same one. This is
    /// the whole point: the position a decode is at lives in the fire tables,
    /// so a thousand tokens of generation are a thousand fires over one graph.
    #[test]
    fn a_hundred_decodes_lower_once() {
        let mut cache = Lowerings::new();
        let mut plans = 0;
        for token in 0..100u32 {
            let ids = [token];
            let step = step(&ids, &[0, 1]);
            let lowered = cache
                .for_step(FireClass::Decode, &step, || {
                    plans += 1;
                    Ok::<_, ()>(empty_plan())
                })
                .expect("an empty plan lowers");
            // THROUGH `lowered`. The cache hands back a `Planned`, which
            // holds the flat launch list beside the runtime streams the
            // plan named; the launches were never its own field.
            assert_eq!(lowered.lowered.launches.len(), 0);
        }
        assert_eq!(plans, 1, "the plan is built on a miss and only on a miss");
        assert_eq!(cache.lowered(), 1);
    }

    /// A reload forgets, because the graph is the old model's.
    #[test]
    fn a_clear_makes_the_next_step_a_miss() {
        let mut cache = Lowerings::new();
        let ids = [7u32];
        let step = step(&ids, &[0, 1]);
        let lower = |cache: &mut Lowerings| {
            cache
                .for_step(FireClass::Decode, &step, || Ok::<_, ()>(empty_plan()))
                .map(|_| ())
                .expect("an empty plan lowers");
        };
        lower(&mut cache);
        lower(&mut cache);
        assert_eq!(cache.lowered(), 1);
        cache.clear();
        lower(&mut cache);
        assert_eq!(cache.lowered(), 2);
    }
}
