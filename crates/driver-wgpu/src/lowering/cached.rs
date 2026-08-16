//! Lowerings, kept by the fire shape that produced them.
//!
//! Beside [`super`] because it is the same subject at a different timescale:
//! that module turns one fire into launches, and this one keeps the answer
//! for the next thousand fires that ask the same question. The path matches
//! `driver-metal::lowering::cached` so the two are comparable.
//!
//! # Why
//!
//! [`model_compiler::lower::lower`] is a **pure function of its three
//! arguments**, and one of them is a constant here: `Serving` holds the two
//! `ForwardPlan`s by reference for as long as a shell is open, and a shell's
//! text is set once at [`crate::shell::Shell::on`] and never replaced. The
//! other two are the rows and a fire flag. So a decode's lowering depends on
//! nothing that changes between the steps of a generation — the position a
//! decode is at lives in the fire TABLES, which are device data and no part
//! of a lowering.
//!
//! This driver derived it anyway, once per token. Measured on this crate's
//! own qwen3 decode text, release build:
//!
//! | | ms |
//! | --- | --- |
//! | `lower`, 1 row, 452 launches | 0.765 |
//! | `lower`, 32 rows | 1.083 |
//!
//! End to end on an RTX 4090, qwen3-0.6B, median of 30 decodes each:
//!
//! | | ms | tok/s |
//! | --- | --- | --- |
//! | cached | 38.9 | 25.7 |
//! | cleared before every step | 40.5 | 24.7 |
//!
//! **1.6 ms of a 39 ms decode, 4 %** — the `lower` call plus the allocation
//! and drop of a 452-launch `Lowered` either side of it.
//!
//! The first version of that measurement ran thirty cached decodes and then
//! thirty uncached ones and reported 8.4 ms, 20 %. It was measuring the
//! CONTEXT: a decode's attention reads every page written so far, so the
//! condition that runs second is slower for a reason that has nothing to do
//! with the cache. The table above interleaves them. Recorded because the
//! wrong number was five times the right one and looked entirely plausible.
//!
//! `driver-metal` found the same defect from the other side and measured
//! 0.60 ms of a 4.9 ms step — 17 %, four times this share, because the same
//! constant host cost sits against a decode eight times faster.
//!
//! # The key is the whole input, not a digest of it
//!
//! `driver-metal`'s version keys its map on a `u64` `DefaultHasher`
//! fingerprint, so two distinct fire shapes that collide would serve each
//! other's graph — 452 launches over the wrong rectangles, silently, with
//! nothing to look at afterwards. The odds are tiny and the consequence has
//! no floor.
//!
//! Nothing is bought by the digest: a lookup hashes the rows either way, and
//! `Row` is eight bytes of flags. So the key here is the rows THEMSELVES,
//! `HashMap` compares them on a hit, and the collision question does not
//! arise. [`Shape`] is that key.
//!
//! # Which plan is part of the key
//!
//! Two texts are held — decode and prefill — and a lowering of one is not a
//! lowering of the other. They are not distinguished by anything inside the
//! rows: today the caller picks by `rows.len() > 1`, but that is the
//! CALLER's rule and not this module's, so [`Shape`] carries the choice
//! explicitly. See `a_decode_and_a_prefill_of_one_row_are_not_one_shape` for
//! what that buys.
//!
//! # What it does not do
//!
//! Grow without bound. Every distinct prefill length is a distinct row
//! vector, and a deployment sees as many of those as it sees prompt lengths.
//! Past [`CAP`] shapes the map is emptied wholesale rather than evicted one
//! at a time: the entry that matters is the decode's, it costs 0.8 ms to
//! re-derive, and an LRU would be machinery to save that once per hundreds
//! of prefills.

use std::collections::HashMap;

use model_compiler::lower::{Fire, Lowered, Row, Uncovered, lower};
use model_ir::trace::ForwardPlan;

/// How many distinct fire shapes are kept before the map is emptied.
pub const CAP: usize = 64;

/// What a lowering is only valid for.
///
/// Both fields, because both are inputs to `lower` that a step can vary. The
/// third input — the plan — is a *reference* the shell holds for its whole
/// life, which is why `prefill` (which of the two) is enough to stand for it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    /// The wider text was used, rather than the decode one.
    pub prefill: bool,
    /// The rows, as the seriation ordered them and as the caller amended
    /// them. `Row` is flags only — no position, no length — which is the
    /// whole reason a thousand decodes are one shape.
    pub rows: Vec<Row>,
}

/// Lowerings, kept by the fire shape that produced them.
#[derive(Debug, Default)]
pub struct Lowerings {
    by_shape: HashMap<Shape, Lowered>,
    /// How many lowerings have been derived, for the test that asks whether
    /// the cache is a cache.
    derived: usize,
}

impl Lowerings {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// This shape's lowering, derived if there is not one.
    ///
    /// `plan` must be the text `shape.prefill` names. Passing the other one
    /// is the single way to misuse this, and it is why the caller computes
    /// the flag ONCE and uses it for both — see [`crate::turns::Serving`].
    ///
    /// # Errors
    ///
    /// [`Uncovered`], from `lower`, on a miss only. A shape that lowered
    /// once lowers forever, which is the point.
    pub fn get(
        &mut self,
        plan: &ForwardPlan,
        shape: Shape,
        fire: Fire,
    ) -> Result<&Lowered, Uncovered> {
        if !self.by_shape.contains_key(&shape) {
            let lowered = lower(plan, &shape.rows, fire)?;
            self.by_shape.insert(shape.clone(), lowered);
            self.derived += 1;
        }
        // Emptied AFTER the insert and never before the return: the caller
        // is about to borrow what was just put in.
        if self.by_shape.len() > CAP {
            let keep = self.by_shape.remove(&shape);
            self.by_shape.clear();
            if let Some(keep) = keep {
                self.by_shape.insert(shape.clone(), keep);
            }
        }
        Ok(&self.by_shape[&shape])
    }

    /// How many lowerings this cache has derived.
    ///
    /// The number that says whether reuse is happening: a hundred decodes
    /// over one deployment should move it by one.
    #[must_use]
    pub fn derived(&self) -> usize {
        self.derived
    }

    /// How many shapes are held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_shape.len()
    }

    /// Whether no shape is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_shape.is_empty()
    }

    /// Forget every lowering.
    ///
    /// **For a text change.** A lowering is the graph of one text, and no
    /// path in this crate replaces a shell's text today — see
    /// `a_shell_never_replaces_its_text`, which is what makes that a checked
    /// claim rather than a hope. This exists so that a path which one day
    /// does has something to call other than a new shell.
    pub fn clear(&mut self) {
        self.by_shape.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_ir::trace::ForwardPlan;

    /// An `n`-op plan, whose lowering is DISTINGUISHABLE by its launch count.
    ///
    /// Not an empty plan: an empty one lowers to an empty `Lowered` whatever
    /// its family is, so a test asking "was the wrong graph served?" could
    /// not tell — the first version of this fixture varied the embedding's
    /// WEIGHT NAME, which does not reach `Lowered` at all for an input-less
    /// embed, and the test failed for that reason rather than for a defect.
    /// Launch count reaches it.
    ///
    /// `Backend::of_family` reads the segment after the first `.`, so the
    /// family ends `.metal`.
    fn plan_embeds(n: usize) -> ForwardPlan {
        use model_ir::trace::{DType, Op, OpKind, Shape as VShape, ValueInfo};
        ForwardPlan {
            family: "cache-fixture.metal".to_string(),
            values: (0..n)
                .map(|_| ValueInfo {
                    shape: VShape(Vec::new()),
                    dtype: DType::BF16,
                    dyn_axis: None,
                })
                .collect(),
            ops: (0..n)
                .map(|i| Op {
                    kind: OpKind::Embed {
                        weight: format!("embed{i}"),
                    },
                    inputs: Vec::new(),
                    outputs: vec![u32::try_from(i).expect("a small fixture")],
                    layer: None,
                })
                .collect(),
            depth_window: false,
            seams: Vec::new(),
        }
    }

    /// The plan the cache tests that do not care about the graph use.
    fn empty_plan(family: &str) -> ForwardPlan {
        ForwardPlan {
            family: family.to_string(),
            values: Vec::new(),
            ops: Vec::new(),
            depth_window: false,
            seams: Vec::new(),
        }
    }

    fn fire() -> Fire {
        Fire {
            captures_across_splits: false,
        }
    }

    fn shape(prefill: bool, n: usize) -> Shape {
        Shape {
            prefill,
            rows: vec![Row::default(); n],
        }
    }

    /// A plan that will not lower: one op, and a family no backend claims.
    fn refusing_plan() -> ForwardPlan {
        use model_ir::trace::{DType, Op, OpKind, Shape as VShape, ValueInfo};
        ForwardPlan {
            family: "no-such-family".to_string(),
            values: vec![ValueInfo {
                shape: VShape(Vec::new()),
                dtype: DType::BF16,
                dyn_axis: None,
            }],
            ops: vec![Op {
                kind: OpKind::Embed {
                    weight: "embed".to_string(),
                },
                inputs: Vec::new(),
                outputs: vec![0],
                layer: None,
            }],
            depth_window: false,
            seams: Vec::new(),
        }
    }

    /// A thousand decodes derive ONE lowering.
    ///
    /// The whole claim, as a number. Falsified by returning `lower(..)`
    /// directly from `get`: 1000.
    #[test]
    fn a_thousand_decodes_derive_one_lowering() {
        let plan = empty_plan("test");
        let mut cache = Lowerings::new();
        for _ in 0..1000 {
            cache
                .get(&plan, shape(false, 1), fire())
                .expect("an empty plan lowers");
        }
        assert_eq!(cache.derived(), 1, "a decode's shape is a constant");
        assert_eq!(cache.len(), 1);
    }

    /// A decode and a one-row prefill are not the same shape.
    ///
    /// They have IDENTICAL rows — one `Row::default()` — and different
    /// texts. A key made only of the rows would serve the decode's graph to
    /// the prefill, which is the defect this field exists to prevent, and
    /// which no amount of row-hashing could catch.
    ///
    /// Falsified by dropping `prefill` from `Shape`: `derived()` is 1 and
    /// the second ask comes back holding `decode`'s weight name.
    #[test]
    fn a_decode_and_a_prefill_of_one_row_are_not_one_shape() {
        // One launch for the decode text, two for the prefill one.
        let decode = plan_embeds(1);
        let prefill = plan_embeds(2);
        let mut cache = Lowerings::new();
        let first = cache
            .get(&decode, shape(false, 1), fire())
            .expect("a one-op plan lowers")
            .clone();
        let second = cache
            .get(&prefill, shape(true, 1), fire())
            .expect("a two-op plan lowers")
            .clone();
        assert_eq!(cache.derived(), 2, "two texts are two lowerings");
        // The two lowerings must actually DIFFER, or everything below is
        // vacuous. The first version of this test used two empty plans, which
        // lowered identically, and it failed for THAT rather than for a
        // defect -- which is the only reason the check is here.
        assert_eq!(first.launches.len(), 1);
        assert_eq!(
            second.launches.len(),
            2,
            "the prefill was served the decode's graph"
        );
        // And back the other way: the decode's entry survived the prefill.
        let again = cache.get(&decode, shape(false, 1), fire()).expect("held");
        assert_eq!(
            again.launches.len(),
            1,
            "the decode was served the prefill's graph"
        );
        assert_eq!(cache.derived(), 2, "the third ask was a hit");
    }

    /// Row COUNT is part of the shape.
    #[test]
    fn rows_of_different_length_are_different_shapes() {
        let plan = empty_plan("test");
        let mut cache = Lowerings::new();
        for n in 1..=8 {
            cache.get(&plan, shape(true, n), fire()).expect("lowers");
        }
        assert_eq!(cache.derived(), 8);
    }

    /// Row CONTENT is part of the shape, not just the count.
    ///
    /// `Row`'s flags change the graph — a masked row lowers to a different
    /// attention than an unmasked one — so two one-row fires that differ in
    /// a flag must not share an entry.
    #[test]
    fn rows_differing_only_in_a_flag_are_different_shapes() {
        let plan = empty_plan("test");
        let mut cache = Lowerings::new();
        let plain = Shape {
            prefill: false,
            rows: vec![Row::default()],
        };
        let masked = Shape {
            prefill: false,
            rows: vec![Row {
                custom_mask: true,
                ..Row::default()
            }],
        };
        cache.get(&plan, plain, fire()).expect("lowers");
        cache.get(&plan, masked, fire()).expect("lowers");
        assert_eq!(cache.derived(), 2);
    }

    /// Past the cap the map is emptied, and the shape just asked for
    /// SURVIVES.
    ///
    /// The caller is about to borrow it. A clear that dropped it would
    /// return a reference into a map that no longer holds the key, which is
    /// a panic at the indexing — so this is a liveness check as much as a
    /// bookkeeping one.
    #[test]
    fn past_the_cap_the_map_empties_but_keeps_what_was_just_asked_for() {
        let plan = empty_plan("test");
        let mut cache = Lowerings::new();
        for n in 1..=CAP + 1 {
            cache.get(&plan, shape(true, n), fire()).expect("lowers");
        }
        assert_eq!(cache.len(), 1, "the map was emptied at the cap");
        assert_eq!(cache.derived(), CAP + 1);
        // And what survived is the last one asked for.
        cache
            .get(&plan, shape(true, CAP + 1), fire())
            .expect("lowers");
        assert_eq!(cache.derived(), CAP + 1, "the survivor was a hit");
    }

    /// A cleared cache derives again.
    #[test]
    fn clearing_forgets() {
        let plan = empty_plan("test");
        let mut cache = Lowerings::new();
        cache.get(&plan, shape(false, 1), fire()).expect("lowers");
        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
        cache.get(&plan, shape(false, 1), fire()).expect("lowers");
        assert_eq!(cache.derived(), 2);
    }

    /// Lines of `src` that assign a `text` field.
    ///
    /// Separated from the test so the DETECTOR can be checked against a
    /// source it is supposed to catch — see
    /// `the_text_check_catches_a_reassignment`. Breaking the real `shell.rs`
    /// to falsify the test does not work: the edit has to compile, and the
    /// obvious one (`self.text = self.text.clone()`) does not, so the test
    /// would silently not run at all.
    fn text_assignments(src: &str) -> Vec<&str> {
        src.lines()
            .map(str::trim)
            .filter(|l| !l.starts_with("//"))
            .filter(|l| l.starts_with("self.text =") || l.contains(".text = "))
            .collect()
    }

    /// The detector catches what it is for.
    #[test]
    fn the_text_check_catches_a_reassignment() {
        assert_eq!(
            text_assignments("fn reload(&mut self, t: Text) {\n    self.text = t;\n}"),
            ["self.text = t;"]
        );
        assert_eq!(
            text_assignments("    shell.text = other;"),
            ["shell.text = other;"]
        );
        // And does not fire on a comment about one, or on a struct literal
        // field, which is how a shell is BUILT rather than changed.
        assert!(text_assignments("// self.text = t;").is_empty());
        assert!(text_assignments("            text,").is_empty());
        assert!(text_assignments("    text: Text,").is_empty());
    }

    /// A shell never replaces its text.
    ///
    /// [`Lowerings::clear`] exists and NOTHING CALLS IT, which is either
    /// correct or a leak of the previous model's graph, and the difference is
    /// a fact about `shell.rs` rather than about this module. So it is read.
    ///
    /// `Shell::text` is private and assigned once, in the struct literal that
    /// builds a shell. If a path ever assigns it again — a reload, a
    /// hot-swap — this fails, and whoever wrote that path is standing next to
    /// the `clear` they need to call.
    #[test]
    fn a_shell_never_replaces_its_text() {
        let src = include_str!("../shell.rs");
        let assignments = text_assignments(src);
        assert!(
            assignments.is_empty(),
            "`shell.rs` reassigns its text, so `lowering::cached` is now \
             serving the OLD text's graph over the new one's weights. Call \
             `Lowerings::clear` there. Lines: {assignments:?}"
        );
        // And the field is private, so no caller can do it either.
        assert!(
            src.contains("\n    text: Text,"),
            "`Shell::text` is no longer a private field declared as `text: \
             Text`; if it became `pub`, a caller can replace it and the \
             check above sees nothing"
        );
    }

    /// A refusal is not cached as a success.
    ///
    /// `lower` can refuse, and a cache that stores nothing on the error path
    /// must return the error EVERY time rather than once — a `derived()` that
    /// counted a failure, or an entry inserted before the `?`, would turn one
    /// bad shape into a permanently poisoned key.
    ///
    /// The fixture refuses through `Uncovered::UnknownBackend`: one op, and a
    /// family no `Backend::of_family` knows. (`Uncovered::Rows`, which the
    /// enum documents as the partition case, is constructed NOWHERE in
    /// `model-compiler` — so it could not have been used here.)
    #[test]
    fn a_shape_that_does_not_lower_refuses_every_time() {
        let mut cache = Lowerings::new();
        let first = cache.get(&refusing_plan(), shape(false, 1), fire());
        assert!(
            first.is_err(),
            "this fixture was meant to REFUSE; `lower` accepted it, so every \
             assertion below would pass vacuously"
        );
        assert!(cache.is_empty(), "a refusal must not occupy the map");
        assert!(
            cache
                .get(&refusing_plan(), shape(false, 1), fire())
                .is_err(),
            "the second ask was served something"
        );
        assert_eq!(cache.derived(), 0, "a refusal is not a derivation");
    }
}
