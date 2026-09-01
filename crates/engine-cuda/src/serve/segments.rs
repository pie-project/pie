//! **WHICH REGIONS OF ONE KEY A GRAPH MAY HOLD** — the admissibility table,
//! widened, memoized, and the one question asked of it.
//!
//! A child module of [`serve`](super) for `stats.rs`'s reason: these are
//! `Shell`'s own methods on `Shell`'s own private fields, and what moved is
//! the TEXT. What makes them a module rather than three hundred lines in the
//! middle of the call order is that they are a DERIVATION and not a step —
//! nothing here stages, launches or orders anything, and the fire path
//! touches it exactly twice, both times from `prepare`'s bodies gate.
//!
//! The whole of it is one memo and its two readers:
//!
//! ```text
//! Segmented              what the derivation answered for one key
//! Shell::segmentation    derive it once, hand out the same Arc after
//! Shell::cuttable        and is there anything left for a graph to hold?
//! ```
//!
//! The soundness argument — why one key has one table for the life of the
//! load, and why the copy world is what the entry STANDS FOR rather than a
//! caveat beside it — is on [`Shell::segmentation`], where it always was.

use crate::record;
use crate::window::Windows;

use super::Shell;

/// One key's segmentation, memoized — [`Shell::segments`]'s value.
///
/// Three fields and no machinery: what the admissibility rule said, whether a
/// legal cut exists over it, and the one input to the first that the
/// `record::BodyKey` does not carry.
pub(super) struct Segmented {
    /// `window::Copies::enabled` as the deriving fire answered it. Not a key
    /// coordinate, so it is stored — and since the capacity wave it is what
    /// the entry STANDS FOR rather than a caveat beside it: a fire whose
    /// answer differs is refused a body ([`Shell::segmentation`]) instead of
    /// being handed a re-derivation the resident exec was not cut to.
    copies: bool,
    /// `Windows::admits` WIDENED (`record::widen`), one entry per template
    /// region. Shared rather than cloned: `Prepared` holds a handle for the
    /// length of the fire and the table is read, never written — and shared is
    /// also what makes the widening one answer, since the `Run`, the capture
    /// loop and `record::cuts` all read this same slice.
    admits: std::sync::Arc<[crate::window::Admit]>,
    /// Did `record::cuts` accept it, or has nobody asked yet?
    ///
    /// `None` is a table nobody has put the cutting question to yet — the
    /// entry is minted where the segmentation is derived and the verdict is
    /// asked one clause later, so the two are never one write. A load that
    /// records no bodies at all derives no table and mints no entry:
    /// `Shell::prepare` asks its four load-wide clauses in front of the
    /// segmentation, so the map holds keys only for deployments that were
    /// actually going to spend one. `Some(false)` is a key
    /// `record::Graphs::body_refuse` has been told about and the operator has
    /// been shown once.
    cuttable: Option<bool>,
}

impl Shell {
    /// **THIS KEY'S ADMISSIBILITY TABLE, DERIVED ONCE AND READ EVERY FIRE
    /// AFTER** (the tier-2 campaign) — `Windows::admits` WIDENED
    /// (`record::widen`), memoized in [`Shell::segments`].
    ///
    /// **AND THE WIDENING IS INSIDE THE MEMO, WHICH IS THE WHOLE OF HOW THE
    /// THREE READERS STAY ONE ANSWER.** `Windows::admits` says which regions
    /// a graph MAY hold; some of those answers cannot be cut at — a boundary
    /// inside a fork group, one between two arms of a conditional, a schedule
    /// on the far side of one from its readers — and `record::widen` grows the
    /// islands until every boundary is legal. That widened table is what this
    /// hands out: to the `Run` (`run::Ceilings::admits`, which stands every ceiling
    /// down inside an island), to `record::Fire::admits` (the capture loop and
    /// the ledger) and to `record::cuts` (the gate's verdict and the capture
    /// script). A caller that widened for itself would be a region a graph
    /// holds and a walk re-issues, which is the one failure this campaign can
    /// produce and the one nothing downstream would notice.
    ///
    /// **THE MEMO IS SOUND BECAUSE THE DERIVATION IS A FUNCTION OF THE KEY**,
    /// and that is not this method's claim to make: `Windows::admits` argues
    /// it clause by clause — gathered is `fallback::copies`' bucket-keyed
    /// answer, a segment list is the artifact's, the interval clauses are the
    /// present set's, `shifted` is read once at load. A key therefore has ONE
    /// table for the life of the load, which is exactly what a body captured
    /// at that key replays: `record::Graphs::fire_body` still `debug_assert`s
    /// its island list on every hit, and what that now compares is the
    /// resident body's script against the table this memo served — which is
    /// the comparison that catches a body captured before something moved.
    ///
    /// **AND THE ONE INPUT THAT IS NOT A KEY COORDINATE IS WHAT THE ENTRY
    /// STANDS FOR, WHICH IS HOW THE HOLE IS CLOSED** (the capacity wave).
    /// `window::Copies::enabled` is `[engine] fallback_copy` — a load constant
    /// — AND "did this fire stage mask bits", which is not. A masked fire
    /// takes the split, so on a SKU with a masked axis and a P4 copy row two
    /// fires of one key can derive different tables, and a resident body cut
    /// for one of them is wrong for the other.
    ///
    /// **SO THE ENTRY IS THE KEY'S WORLD AND THIS FUNCTION DOES NOT LEAVE
    /// IT.** A fire whose copy answer disagrees with the entry's is handed the
    /// ENTRY's table and told so, and the bodies gate turns it away — the fire
    /// walks eagerly and is counted (`record::BodyTally::eager_copy_world`).
    /// It used to re-derive and overwrite, which kept the MEMO honest and left
    /// the BODY dangling: the resident exec was cut in the first world, the
    /// second world's fire would have replayed it against a different island
    /// list, and only a `debug_assert` stood between that and a silent wrong
    /// answer in release.
    ///
    /// **AND WHICH WORLD A KEY IS IN IS FIXED BY THE ARMING SYNTHETIC, NOT BY
    /// WHOEVER FIRES FIRST.** `Shell::arm_bodies` composes a synthetic per key
    /// and `Shell::synthetic_lanes` stages a mask on exactly the lanes whose
    /// class runs the masked arm — which is what every real fire of that key
    /// does too, because `Fault::MaskWord` refuses any other pairing. So the
    /// arming fire's copy answer is the copy answer of every fire of its key,
    /// and on a sealed load the whole population's worlds were written by the
    /// boot.
    ///
    /// **WHICH MAKES THE PRICE ZERO ON EVERY DEPLOYMENT THAT DOES NOT FLIP THE
    /// KNOB.** `Windows::admits` carries the derivation: the mask half of
    /// `copies_here` is a function of the present SET, so the only way two
    /// fires of one key can disagree is `Shell::set_copies` between them — a
    /// diagnostic A/B. This clause costs that session its replays and costs a
    /// serving deployment nothing at all. What it buys is that a body is only
    /// ever replayed by the world it was captured in, which is the sealed
    /// lattice's own doctrine applied to the one axis outside the key.
    ///
    /// Answers the table and whether this fire is IN the key's world: `false`
    /// is the refusal above, and the table beside it is the entry's rather
    /// than this fire's — inert, because a fire the gate turns away reads no
    /// admission at all (`run::Ceilings::admit` is gated on `bodied` first).
    pub(super) fn segmentation(
        &mut self,
        key: &record::BodyKey,
        windows: &Windows,
        totals: model_ir::PerAxis<u32>,
        copies: bool,
    ) -> (std::sync::Arc<[crate::window::Admit]>, bool) {
        // **THE ENTRY IS READ ONCE AND BOTH QUESTIONS ARE ANSWERED OFF IT.**
        // The table and the world it was derived in come out of the same
        // `get`, so "is this key's world mine" and "what is this key's table"
        // can never be answered off different entries — and the borrow ends
        // here, which is what lets the mismatch arm reach the counter.
        let held = self
            .segments
            .get(key)
            .map(|held| (std::sync::Arc::clone(&held.admits), held.copies));
        if let Some((admits, world)) = held {
            if world != copies {
                // **THE KEY IS IN ANOTHER WORLD AND THIS FIRE IS NOT SERVED
                // FROM IT.** Counted here rather than at the gate because this
                // is the only line that can see both words; the gate acts on
                // the answer.
                self.cache.eager_copy_world();
                return (admits, false);
            }
            // **AND THE MEMO IS CHECKED AT ITS OWN DOOR, IN DEBUG.** The claim
            // above is that this table is a function of the key; a memo that
            // merely believed it would be the thing that hid the day it stops
            // being true. Re-deriving here and diffing the WHOLE table is
            // strictly stronger than what `Graphs::fire_body` asserts — that
            // one sees only the island projection, and only for a key that
            // holds a body — and it costs a `Vec` per fire in a debug build
            // and exactly nothing in a release one.
            //
            // **AND IT IS ASKED PAST THE WORLD CHECK, WHICH IS WHAT MAKES IT
            // A REAL ASSERT AGAIN** (the capacity wave). The copy answer is
            // the one input to `Windows::admits` the key does not carry, so
            // while it could differ this comparison had a legal way to fail
            // and could not be trusted. Past the arm above the two fires agree
            // about copies by construction, and anything this catches is a
            // genuinely new input.
            debug_assert!(
                admits.as_ref()
                    == record::widen(
                        &self.compiled,
                        &windows.admits_axes(totals, &self.shifted)
                    ),
                "the admissibility table for {key} is not what this key derived \
                 before, so `Windows::admits` has grown an input the key does \
                 not carry",
            );
            return (admits, true);
        }
        // **WIDENED HERE AND NOWHERE ELSE.** One call, one table, three
        // readers — see this method's header for why that is not a
        // convenience.
        let admits: std::sync::Arc<[crate::window::Admit]> =
            record::widen(
                &self.compiled,
                &windows.admits_axes(totals, &self.shifted),
            )
            .into();
        // **AND THE MAP IS BOUNDED, ON `record::Graphs::body_warm`'S OWN
        // DISCIPLINE.** Nothing ever evicted from here: an entry per distinct
        // (bucket x present set) went in on the key's first fire and stayed
        // for the life of the load, which is a table with an unbounded number
        // of realizable keys behind it and a `Vec<Admit>` per template on each
        // one. What is kept when the map grows past the seat count is what
        // the cache can still SPEND — the keys holding a body, and the keys
        // that were refused one (whose refusal is the reason nobody will ask
        // again) — and everything else is a memo of a shape that came once.
        // Forgetting one costs a re-derive, which is the honest price of not
        // remembering.
        if self.segments.len() > record::MAX_BODIES * 4 {
            let cache = &self.cache;
            self.segments
                .retain(|key, _| cache.holds_body(key) || cache.body_refused(key));
        }
        self.segments.insert(key.clone(), Segmented {
            copies,
            admits: std::sync::Arc::clone(&admits),
            cuttable: None,
        });
        // The first fire of a key WRITES the world rather than being checked
        // against it — on an armed load that fire is the arming synthetic, and
        // on an unarmed one it is whoever arrives first. Either way the entry
        // is the key's world from here on and every later fire is measured
        // against it.
        (admits, true)
    }

    /// **IS THERE ANYTHING LEFT FOR A GRAPH TO HOLD?** — `record::cuts`
    /// asked as the predicate `prepare`'s gate wants, once per key.
    ///
    /// `prepare` throws the script away — the capture loop derives its own,
    /// off the same table, at the one instant that is going to record — so
    /// what the gate needs is the verdict alone, and the verdict is a
    /// function of the key for [`segmentation`](Shell::segmentation)'s
    /// reason: `cuts` reads that table and the template and nothing else.
    /// Memoized in the same entry, so a steady stream allocates no `Vec<Cut>`
    /// per fire.
    ///
    /// **AND THE DECLINE IS TAKEN HERE**, which is why this is a second
    /// method and not a field of the first. It is `prepare`'s gate that
    /// decides whether a composition is being ASKED to record — a load
    /// serving `graphs = off`, or `bodies = off`, or one whose weights rotate
    /// is not — and a shell that printed "this body declines to be
    /// segmented" at a deployment that never wanted a body would be counting
    /// traffic against a path it does not serve. So the table above is
    /// derived for every fire and this question is asked only past the outer
    /// clauses, exactly where the old inline `cuts` call stood.
    pub(super) fn cuttable(
        &mut self,
        key: &record::BodyKey,
        admits: &[crate::window::Admit],
    ) -> bool {
        if let Some(Some(held)) = self.segments.get(key).map(|seg| seg.cuttable) {
            return held;
        }
        // Bound first: the script is dropped here and the borrow of
        // `self.compiled` with it, so the decline arm below is free to write
        // the refusal memo.
        let script = record::cuts(&self.compiled, admits);
        let verdict = match script {
            Ok(_) => true,
            Err(uncut) => {
                // **THE ONE REFUSAL LEFT ON THIS AXIS, AND IT IS NO LONGER
                // ABOUT A BOUNDARY** (the tier-2 campaign, then the
                // widening). A boundary a graph cannot be cut at — inside a
                // fork group, between two arms of a conditional, across a
                // schedule from its readers — used to decline the whole
                // composition and throw away every capturable region of it.
                // `record::widen` GROWS the island to the nearest legal
                // boundary instead, because a region served eagerly is the
                // eager walk and is always right. So what reaches this arm is
                // the terminal case: a composition the growing consumed
                // entirely, whose body would be a script of islands with no
                // exec in it. It is declined BY NAME, before a stream is
                // touched, and the sentence is printed once per key because
                // `body_refuse` is the memo that deduplicates it and counts
                // the composition.
                //
                // **AND IT IS A SENTENCE ABOUT THE ARTIFACT, WHICH IS WHY IT
                // IS WORTH A LINE.** Every window of this composition is one
                // this shell has to re-issue every fire, so the answer to it
                // is a `crate::SHIFTED` look or a seat — not a capture.
                eprintln!(
                    "engine-cuda: body {key} holds nothing a graph can keep — {uncut}. \
                     This composition walks eagerly for the life of the load; \
                     `record::widen` grew its islands to the nearest legal boundary \
                     first, and `record::Uncut` names what was left."
                );
                self.cache.body_refuse(key.clone());
                false
            }
        };
        if let Some(seg) = self.segments.get_mut(key) {
            seg.cuttable = Some(verdict);
        }
        verdict
    }
}
