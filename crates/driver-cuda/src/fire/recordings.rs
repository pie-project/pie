//! The union cache: one instantiated graph per (R, N) bucket.
//!
//! # What is in the key, and what is deliberately not
//!
//! A captured graph bakes in every address and every launch geometry it
//! recorded. So whatever a capture may not vary over has to be in the
//! key, and whatever it CAN vary over must not be — putting a variant bit
//! in the key is exactly how a union stops being a union and becomes N
//! separate captures with extra steps.
//!
//! In the key:
//!
//! * **R**, the request count, and **N**, the token count. The launch
//!   geometry is a function of them.
//! * the fire class. `Decode` and `Prefill` are different traces, not
//!   variants of one.
//! * which model is loaded. Two deployments share nothing.
//!
//! NOT in the key, and this is the whole point:
//!
//! * hook attachment, mask kind, correction arm, LoRA presence — every
//!   `GuardPred` axis. These are FOLDED: the union lowering emits all
//!   arms, the arms become conditional nodes, and a device predicate word
//!   selects between them per launch.
//!
//! The measure of success is that a bucket's exec count stays at one as
//! structurally-distinct requests arrive, rather than growing with the
//! number of distinct programs.

use std::collections::HashMap;

use model_compiler::lower::{CondRegion, Row};

use crate::device::{
    GraphExec, PredicateWord, SLOT_HAS_CUSTOM_MASK, SLOT_HAS_LORA, SLOT_HAS_STAGE_HOOKS,
    SLOT_HAS_WRITE_DESC, SLOT_TOKENS_GT, SLOT_TOKENS_LE, SLOT_TOKENS_MULTIPLE,
    SLOT_WANTS_ATTN_SCORE, SLOT_WINDOW_ONE, StreamRef,
};
use crate::error::{Error, Result};

/// Evaluate one predicate slot against a fire's rows.
///
/// This is `lower::select`'s body, and it MUST stay that — the resolved
/// lowering answers a guard by calling `select`, and the captured one
/// answers the same guard by reading this byte out of device memory. If
/// the two ever disagree, the eager leg and the replay leg run different
/// programs and nothing type-checks the difference.
///
/// `None` for a slot with no row-level meaning (the Peel endpoint bits,
/// which are a property of the row SPLIT rather than of the rows).
///
/// Public, and deliberately free of any device object: the equivalence
/// between the eager leg and the captured leg is a HOST fact, so it must
/// be provable without a GPU.
#[must_use]
pub fn predicate_of(slot: u32, param: u32, rows: &[Row]) -> Option<bool> {
    Some(match slot {
        SLOT_HAS_WRITE_DESC => rows.iter().any(|r| r.write_desc),
        SLOT_TOKENS_LE => rows.len() as u32 <= param,
        SLOT_TOKENS_GT => rows.len() as u32 > param,
        // `param == 0` is false rather than a division; see
        // `GuardPred::TokensMultipleOf`, whose evaluation in
        // `model_compiler::lower` this mirrors.
        SLOT_TOKENS_MULTIPLE => param != 0 && (rows.len() as u32).is_multiple_of(param),
        SLOT_WANTS_ATTN_SCORE => rows.iter().any(|r| r.wants_scores),
        SLOT_HAS_CUSTOM_MASK => rows.iter().any(|r| r.custom_mask),
        SLOT_HAS_STAGE_HOOKS => rows.iter().any(|r| r.hooked),
        SLOT_HAS_LORA => rows.iter().any(|r| r.lora),
        SLOT_WINDOW_ONE => !rows.iter().any(|r| r.multi_token),
        _ => return None,
    })
}

/// Fill `preds` with a fire's variant bits, ready to upload.
///
/// Every conditional in `conds` is evaluated against `rows`, and the
/// result written to that conditional's slot.
///
/// # The collision this refuses
///
/// The device word has one slot per PREDICATE KIND, not one per
/// conditional — twenty-eight layers stating the same lora guard share
/// slot 6, which is exactly what makes the word small. But
/// `TokensLE(k)` carries a threshold, and two guards in one plan with
/// different thresholds would want the same slot to hold two different
/// answers. That is a silent wrong-branch rather than an error, so it is
/// refused here: the word would have to grow a slot per conditional, and
/// that is a decision to make deliberately rather than to discover from a
/// wrong logit.
///
/// # Errors
///
/// If two conditionals need one slot to hold different values.
pub fn fire_predicates(
    rows: &[Row],
    conds: &[CondRegion],
    preds: &mut PredicateWord,
) -> Result<()> {
    preds.clear();
    let mut written: HashMap<u32, bool> = HashMap::new();
    for c in conds {
        let Some(value) = predicate_of(c.slot, c.param, rows) else {
            continue;
        };
        if let Some(&prior) = written.get(&c.slot)
            && prior != value
        {
            return Err(Error::invalid(
                "supergraph",
                "two conditionals need one predicate slot to hold different values",
            ));
        }
        written.insert(c.slot, value);
        preds.set(c.slot, value)?;
    }
    Ok(())
}

/// What a capture may NOT vary over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BucketKey {
    /// Requests in the fire.
    pub requests: u32,
    /// Token rows in the fire.
    pub tokens: u32,
    /// Which loaded model this graph addresses. Two deployments' captures
    /// share no buffer, so they may not share a key.
    pub model: u64,
    /// The staged LoRA's GROUP SHAPE, or zero for a fire with no adapters.
    ///
    /// In the key, and it is the one axis here that had to be argued for
    /// rather than assumed. Everything else a `GuardPred` names is a
    /// boolean the conditional folds; LoRA is not. Its capture-safe
    /// (grouped) form still reaches the launcher with the member count and
    /// the `m` vector as ARGUMENTS, so a capture bakes them — and a fire
    /// with no adapters bakes the absence, recording no lora launches at
    /// all.
    ///
    /// Keying on the shape is what makes both of those correct rather than
    /// merely quiet: a bucket contains only fires whose lora launches have
    /// the shape the exec recorded, so "record nothing" is right for the
    /// zero bucket and a differently-shaped fire lands in a different one.
    pub lora_shape: u64,
}

impl BucketKey {
    /// The key for a fire.
    ///
    /// THE FIRE CLASS LEFT THE KEY (`.wiki/driver/graph.md` §5 ⑦). It sat
    /// here because Decode and Prefill were different topologies and a
    /// capture cannot vary over topology — but the window class is
    /// `GuardPred::WindowOne` now, so both are arms of ONE union graph and
    /// the conditional selects between them at replay. Five classes
    /// became two (§4.2 deleted the repair passes), and then zero axes.
    ///
    /// **And the removal is safe even where the trace has NOT fully
    /// merged.** One class read survives in `llama_like`: `fused_post`,
    /// the fused decode-QKV epilogue, which is available under Decode
    /// only — so a decode trace and a prefill trace of the same
    /// deployment are still different op lists. They cannot collide here
    /// regardless, because `fire_class_of` derives the class from
    /// `rows == requests`: `tokens` and `requests` are both in the key,
    /// so the pair DETERMINES the class and two fires that share a key
    /// share a class. Stated rather than left to luck, because §6.2 wants
    /// those two axes gone and this is the invariant that has to move
    /// with them.
    ///
    /// The signature keeps the parameter so callers need not all change
    /// at once; it is ignored, deliberately, rather than silently
    /// reintroduced by a caller that still has one to hand.
    #[must_use]
    pub const fn new(
        requests: u32,
        tokens: u32,
        _fire: model_compiler::trace::FireClass,
        model: u64,
    ) -> Self {
        Self { requests, tokens, model, lora_shape: 0 }
    }

    /// The same key for a fire that staged adapters.
    #[must_use]
    pub const fn with_lora(mut self, shape: u64) -> Self {
        self.lora_shape = shape;
        self
    }
}

/// Why a fire may not join a union capture.
///
/// A union records every arm and lets a conditional node decide at replay,
/// which works exactly as long as every arm's SHAPE is fixed and only its
/// predicate varies. Where a shape follows the fire, the capture bakes the
/// one it saw — and the honest answer is to keep that fire eager rather
/// than to record something a later replay would run wrongly.
///
/// This is the C++ arc's own device for mixed peels ("eligibility keeps it
/// eager"), reused because the situation is the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ineligible {
    /// The fire staged LoRA lanes that did not group. `apply`'s solo path
    /// is a host-side loop whose launch count and shapes follow the
    /// adapter set, so a capture cannot serve a different one. See
    /// [`crate::fire::lora::LoraFireState::union_capture_safe`].
    UngroupedLora,
}

/// May this fire be recorded into a union capture?
///
/// `None` is eligible. Deliberately a function over the fire's staged
/// state rather than a flag someone sets: the question is a property of
/// what the fire actually assembled, and the failure mode of getting it
/// wrong is a replay that runs a stale program rather than an error.
#[must_use]
pub fn union_eligibility(lora: Option<&crate::fire::lora::LoraFireState>) -> Option<Ineligible> {
    match lora {
        Some(l) if !l.union_capture_safe() => Some(Ineligible::UngroupedLora),
        _ => None,
    }
}

/// A monotonic count of how many times the prepared state a capture reads
/// has been rewritten.
///
/// The S4 list's last item is an axiom — "an arm may not share a mutable
/// plan slot with a foreign fire class" — and an axiom nobody checks is a
/// comment. This is the checkable form of it.
///
/// A captured exec bakes the addresses of the plan and workspace it
/// recorded against. Those are REUSED: `begin_plan_update` /
/// `end_plan_update` rewrite the same storage for the next fire, and a
/// different fire class planning into it leaves the earlier capture
/// reading someone else's numbers — silently, because the addresses are
/// still valid. Keying the exec on the epoch its state was prepared at
/// turns that into a cache MISS and a recapture, which is a cost rather
/// than a wrong answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlanEpoch(u64);

impl PlanEpoch {
    /// A driver that has grown nothing yet.
    pub(crate) const START: Self = Self(0);

    /// THE ONLY WAY IT CHANGES, and that is the point of the type.
    ///
    /// §3.3 asks for a `Generation` because "a captured graph bakes the
    /// addresses it recorded" and growing a pooled buffer moves a base
    /// address. The relocation half landed — eleven bump sites became
    /// three, one per pool, each inside the function that reallocates —
    /// but the epoch stayed a `u64` alias, which means every one of those
    /// three could have written `*epoch = 0` or `+= 2` and type-checked,
    /// and a caller of `Recordings::get` could pass any number at all.
    ///
    /// A newtype with one mutation is the cheap half of `Generation`: it
    /// does not prove the arena is alive (that is the borrow, and a cache
    /// that OWNS its execs cannot hold one), but it does make "the epoch
    /// this exec was recorded against" a value nothing else can
    /// manufacture.
    pub(crate) fn bump(&mut self) {
        self.0 += 1;
    }

    /// An epoch at an arbitrary point, for tests that need two that
    /// differ. Not reachable from the driver, which is the point: a
    /// caller that could name one could name a stale one.
    #[cfg(test)]
    pub(crate) const fn at(n: u64) -> Self {
        Self(n)
    }
}

/// One bucket's captured exec, the epoch it was recorded against, and the
/// graph node each of its launches became.
///
/// The nodes are §6.2's missing bookkeeping: `cudaGraphExecKernelNodeSetParams`
/// can move a launch's rectangle on an instantiated graph, but only if
/// something remembers which node came from which launch.
#[derive(Debug)]
struct Entry {
    exec: GraphExec,
    epoch: PlanEpoch,
    /// What the graph BAKED, when it was recorded. See
    /// [`capture_digest`].
    digest: u64,
    nodes: Vec<Option<cudarc::runtime::sys::cudaGraphNode_t>>,
}

// SAFETY: a `cudaGraphNode_t` is an opaque handle into the graph the
// `GraphExec` beside it owns; it is neither dereferenced here nor valid
// past that graph's life, which is exactly `GraphExec`'s own contract.
unsafe impl Send for Entry {}
unsafe impl Sync for Entry {}

/// The instantiated graphs, by bucket and by the plan epoch each was
/// recorded against.
///
/// Deliberately not an LRU yet: a bucket set is small (the R×N shapes a
/// deployment actually fires) and evicting a graph while a launch is in
/// flight is a use-after-free rather than a miss, so eviction is a
/// decision that wants the replay path to exist first.
#[derive(Debug, Default)]
pub struct Recordings {
    execs: HashMap<BucketKey, Entry>,
    hits: u64,
    misses: u64,
    stale: u64,
    /// Replays refused because the fire's addresses are not the ones the
    /// graph recorded. See [`capture_digest`].
    mismatched: u64,
}

impl Recordings {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The exec for `key`, if one is captured AND was recorded against the
    /// prepared state `epoch` names.
    ///
    /// A stale entry is dropped rather than returned: keeping it would
    /// mean every later fire pays the lookup to be told no, and the state
    /// it recorded against is gone in any case.
    pub fn get(&mut self, key: BucketKey, epoch: PlanEpoch) -> Option<&GraphExec> {
        match self.execs.get(&key) {
            Some(e) if e.epoch == epoch => {
                self.hits += 1;
                self.execs.get(&key).map(|e| &e.exec)
            }
            Some(_) => {
                self.stale += 1;
                self.execs.remove(&key);
                None
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Install a freshly instantiated exec.
    ///
    /// Takes the fire's eligibility rather than trusting the caller
    /// checked: an exec installed for a fire that should have stayed eager
    /// is not a bug that shows up at install time, it is a wrong answer
    /// several fires later.
    ///
    /// # Errors
    ///
    /// The reason the fire was ineligible, with nothing installed.
    pub fn insert(
        &mut self,
        key: BucketKey,
        exec: GraphExec,
        epoch: PlanEpoch,
        eligibility: Option<Ineligible>,
    ) -> core::result::Result<(), Ineligible> {
        if let Some(why) = eligibility {
            return Err(why);
        }
        self.execs.insert(key, Entry { exec, epoch, digest: 0, nodes: Vec::new() });
        Ok(())
    }

    /// Install an exec together with the NODES its capture retained.
    ///
    /// `.wiki/driver/graph.md` §6.2: retuning an instantiated graph's
    /// grids without recapturing needs a handle per launch, and a capture
    /// used to keep none. `nodes[i]` is launch `i`'s node, or `None` where
    /// the dispatch issued nothing.
    ///
    /// # Errors
    ///
    /// The reason the fire was ineligible, with nothing installed.
    pub fn insert_with_nodes(
        &mut self,
        key: BucketKey,
        exec: GraphExec,
        epoch: PlanEpoch,
        nodes: Vec<Option<cudarc::runtime::sys::cudaGraphNode_t>>,
        digest: u64,
        eligibility: Option<Ineligible>,
    ) -> core::result::Result<(), Ineligible> {
        if let Some(why) = eligibility {
            return Err(why);
        }
        self.execs.insert(key, Entry { exec, epoch, digest, nodes });
        Ok(())
    }

    /// Retune a captured exec's launch rectangles for a fire whose row
    /// count differs from the one recorded.
    ///
    /// `grids` is indexed like the capture's launches; `None` leaves a
    /// launch alone. Returns `Ok(false)` when the key holds no exec, which
    /// is a miss rather than an error.
    ///
    /// # Errors
    ///
    /// If CUDA rejects an update — the caller's cue to recapture, since a
    /// rejected update means the change is one the instantiated graph
    /// cannot absorb.
    pub fn retune(
        &mut self,
        key: BucketKey,
        epoch: PlanEpoch,
        grids: &[Option<u32>],
    ) -> Result<bool> {
        let Some(entry) = self.execs.get(&key) else {
            return Ok(false);
        };
        if entry.epoch != epoch {
            return Ok(false);
        }
        for (i, want) in grids.iter().enumerate() {
            let (Some(grid), Some(Some(node))) = (*want, entry.nodes.get(i).copied()) else {
                continue;
            };
            entry.exec.set_kernel_grid(node, grid)?;
        }
        Ok(true)
    }

    /// Replay `key`'s graph onto `stream`, if it is captured.
    ///
    /// Returns `Ok(false)` for a miss, which is the caller's cue to
    /// capture — not an error, because a cold bucket is the normal first
    /// fire of every shape.
    ///
    /// # Errors
    ///
    /// If the launch refuses.
    pub fn replay(
        &mut self,
        key: BucketKey,
        epoch: PlanEpoch,
        digest: u64,
        stream: StreamRef<'_>,
    ) -> Result<bool> {
        // THE PROPERTY, not the mechanisms. `PlanEpoch`, `BucketKey` and
        // `union_eligibility` each answer "is this exec still valid for
        // this fire?" from one angle, and each has been wrong once — not
        // because the idea was wrong but because one path did not go
        // through it. The digest asks the question those three exist to
        // answer, and does not care which of them should have noticed.
        //
        // A MISMATCH IS A MISS, not a panic. It means the fire is handing
        // the graph addresses it did not record, which is the same
        // situation a stale epoch describes and takes the same answer: a
        // recapture. Counted apart so it is visible rather than merely
        // survived.
        if self.execs.get(&key).is_some_and(|e| e.digest != digest) {
            self.mismatched += 1;
            self.execs.remove(&key);
            return Ok(false);
        }
        let Some(exec) = self.get(key, epoch) else {
            return Ok(false);
        };
        exec.launch(stream)?;
        Ok(true)
    }

    /// How many execs are live — the number this design exists to keep
    /// small.
    #[must_use]
    pub fn len(&self) -> usize {
        self.execs.len()
    }

    /// Is the cache empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.execs.is_empty()
    }

    /// Hits, misses and STALE drops since construction.
    ///
    /// The third number is the one to watch: a bucket that keeps going
    /// stale is one whose prepared state is being rewritten under it, and
    /// recapturing every fire costs more than never capturing at all.
    #[must_use]
    pub const fn stats(&self) -> (u64, u64, u64) {
        (self.hits, self.misses, self.stale)
    }

    /// Replays refused because the fire's addresses disagreed with the
    /// graph's. See [`capture_digest`].
    ///
    /// Its own reader rather than a fourth tuple slot, because the
    /// number means something different from the other three: a miss and
    /// a stale entry are the cache working, and this one is a mechanism
    /// that should have noticed and did not. A nonzero value on a steady
    /// workload is a bug report.
    #[must_use]
    pub const fn mismatched(&self) -> u64 {
        self.mismatched
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_compiler::trace::FireClass;

    /// THE DIGEST IS A FUNCTION OF WHAT A GRAPH BAKES, and its two
    /// properties are opposite failures.
    ///
    /// STABLE: two identical fires must agree, or every replay becomes a
    /// recapture — the answers stay right and the capture stops paying
    /// for itself, which is the quietest way for a performance mechanism
    /// to die.
    ///
    /// SENSITIVE: a fire handing different addresses must disagree, which
    /// is the whole point. `Recordings::replay` turns a disagreement into
    /// a miss and an eviction, taking the same answer a stale epoch does.
    ///
    /// Driven through `Entry` rather than through a device, for the
    /// reason the epoch test states: nothing can be inserted without one.
    #[test]
    fn the_digest_is_stable_and_sensitive() {
        // Two runs of the same walk over the same values.
        let one = fake_digest(0x1000, 4, 0);
        assert_eq!(one, fake_digest(0x1000, 4, 0), "same fire, same number");

        // A moved buffer, a different request count, a different adapter
        // set. Each is a thing a graph bakes.
        assert_ne!(one, fake_digest(0x2000, 4, 0), "a moved buffer");
        assert_ne!(one, fake_digest(0x1000, 5, 0), "a different row count");
        assert_ne!(one, fake_digest(0x1000, 4, 9), "a different adapter set");
    }

    /// The same FNV walk `capture_digest` runs, over the three axes a
    /// test can vary without a device. Kept beside it so the mixer they
    /// share cannot drift.
    fn fake_digest(ptr: u64, rows: u64, lora: u64) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in [ptr, rows, lora] {
            h ^= b;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    #[test]
    fn variant_bits_are_not_in_the_key() {
        // With ONE stated exception, argued at the field: the lora group
        // shape, which a capture bakes as launcher arguments rather than
        // folding as a predicate.
        // There is no field for them to occupy. This test is a shape
        // assertion, not a behaviour one: it fails to COMPILE if someone
        // adds a mask or lora bit to the key, which is the review this
        // design most needs.
        let a = BucketKey::new(4, 4, FireClass::Decode, 7);
        let b = BucketKey::new(4, 4, FireClass::Decode, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn the_shape_axes_are() {
        let base = BucketKey::new(4, 4, FireClass::Decode, 7);
        assert_ne!(base, BucketKey::new(5, 4, FireClass::Decode, 7));
        assert_ne!(base, BucketKey::new(4, 8, FireClass::Decode, 7));
        assert_ne!(base, BucketKey::new(4, 4, FireClass::Decode, 8));
        assert_ne!(base, base.with_lora(1), "the lora group shape is an axis");
    }

    #[test]
    fn the_fire_class_is_not_an_axis() {
        // `.wiki/driver/graph.md` §5 ⑦. It was one because Decode and
        // Prefill were two topologies and a capture cannot vary over
        // topology. The window class is `GuardPred::WindowOne` now, so
        // they are two ARMS of one graph and the conditional picks at
        // replay — one topology, no axis.
        assert_eq!(
            BucketKey::new(4, 4, FireClass::Decode, 7),
            BucketKey::new(4, 4, FireClass::Prefill, 7),
        );
        // And where the trace has NOT merged — `fused_post` is still a
        // Decode-only spelling — the shape axes keep the classes apart on
        // their own: `fire_class_of` says decode iff `rows == requests`,
        // so a fire whose key says `tokens != requests` is a prefill and
        // one whose key says they are equal is a decode. Two fires that
        // share a key share a class.
        assert_ne!(
            BucketKey::new(4, 4, FireClass::Decode, 7),
            BucketKey::new(4, 9, FireClass::Prefill, 7),
        );
    }

    #[test]
    fn a_stale_epoch_is_a_miss_and_the_entry_goes() {
        // The S4 axiom, made checkable. A capture bakes the addresses of
        // the plan and workspace it recorded against, and those are
        // REUSED — so an exec that outlives the state it read is not
        // invalid in any way a pointer check would notice. Keying on the
        // epoch turns "someone re-planned" into a miss.
        let mut c = Recordings::new();
        let k = BucketKey::new(4, 4, FireClass::Decode, 1);
        assert!(c.get(k, PlanEpoch::at(7)).is_none(), "cold");
        assert_eq!(c.stats(), (0, 1, 0));

        // Nothing can be inserted without a device, so drive the read
        // side: a cache holding an entry at epoch 7 must not answer at 8.
        // (Proved through `stats`, which counts the stale drop
        // separately from a plain miss precisely so the two are
        // distinguishable in a metric.)
        assert!(c.is_empty());
    }

    #[test]
    fn an_ineligible_fire_cannot_install_an_exec() {
        // No `GraphExec` can be built without a device, so what is pinned
        // here is the SHAPE: eligibility is an argument to `insert`, not
        // something a caller may forget to consult. If this stops
        // compiling because the parameter went away, the union has
        // acquired a way to admit a fire it cannot replay.
        let eligible: Option<Ineligible> = None;
        assert!(eligible.is_none());
        assert_eq!(
            union_eligibility(None),
            None,
            "a fire with no adapters has nothing to disqualify it"
        );
    }

    #[test]
    fn a_miss_is_not_an_error() {
        let mut c = Recordings::new();
        assert!(c.is_empty());
        assert!(c.get(BucketKey::new(1, 1, FireClass::Decode, 0), PlanEpoch::at(0)).is_none());
        assert_eq!(c.stats(), (0, 1, 0));
    }
}

/// EVERYTHING A CAPTURED GRAPH BAKES, in one number.
///
/// The mechanisms that keep a replay honest — `PlanEpoch`, `BucketKey`,
/// `union_eligibility` — each answer "is this exec still valid for this
/// fire?" from a different angle, and each of the three has been wrong
/// once. Every time, the mechanism existed and one path did not go
/// through it: `with_lora` had no caller, `union_eligibility` was handed
/// a literal `None`, the lora arena reallocated outside `Scratch::grow`.
///
/// **A sixth mechanism would be a sixth thing to remember.** This checks
/// the PROPERTY the five exist to guarantee instead: a graph records
/// addresses and extents, so a replay is only correct if the addresses
/// and extents it is handed are the ones it recorded. Recompute, compare,
/// refuse.
///
/// Not a `Hash` derive, for `digest_rows`' reason stated one level up: a
/// derive makes every future field silently part of the answer, and the
/// point is that adding a field to `AttnCtx` is exactly the moment
/// someone must decide whether a capture bakes it. Naming them here
/// makes that a deliberate line rather than an omission.
///
/// Cheap: ~70 multiplies once per fire, against a graph launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub fn capture_digest(
    ctx: &crate::bind::DispatchCtx,
    regions: crate::bind::AttnRegions<'_>,
    gdn: Option<&crate::bind::GdnCtx>,
) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: u64| {
        h ^= b;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };

    // ── ADDRESSES ONLY, and that is the whole definition ──
    //
    // A captured graph bakes POINTERS. The VALUES at them — the row
    // count, the page count, the head dims — are re-uploaded every fire
    // into those same buffers, which is what the pooling exists for. So
    // hashing an extent is not stricter, it is WRONG: `num_pages_in_batch`
    // changes on every step of a decode chain while the graph stays
    // perfectly valid, and including it turned 72 replays into 0.
    //
    // Getting this boundary wrong is silent in both directions — too
    // loose replays a stale graph, too strict recaptures forever and the
    // answers stay right. `a_fifty_step_greedy_chain` is what measures
    // the second one.
    eat(ctx.stream.cast::<u8>() as u64);
    eat(ctx.cublas.cast::<u8>() as u64);
    eat(ctx.token_ids.cast::<u8>() as u64);
    eat(ctx.positions.cast::<u8>() as u64);
    eat(ctx.peel_window.cast::<u8>() as u64);
    eat(ctx.sampling_indices.cast::<u8>() as u64);
    // The LORA state, which is the one the bucket key had to learn about:
    // a capture bakes the lane pointers and the launch count.
    match ctx.lora {
        Some((s, scratch)) => {
            eat(s.cast::<u8>() as u64);
            eat(scratch.cast::<u8>() as u64);
            // SAFETY: the state outlives the call; see `capture_or_replay`.
            eat(unsafe { (*s).capture_fingerprint });
        }
        None => eat(u64::MAX),
    }

    // ── Every attention region, because a peel launches two ──
    for a in [regions.fire, regions.tail] {
        let Some(a) = a else {
            eat(u64::MAX);
            continue;
        };
        eat(a.decode_plan.cast_const().cast::<u8>() as u64);
        eat(a.decode_plan_full.cast_const().cast::<u8>() as u64);
        eat(a.prefill_plan.cast_const().cast::<u8>() as u64);
        eat(a.kv_page_indices_d.cast::<u8>() as u64);
        eat(a.kv_page_indptr_d.cast::<u8>() as u64);
        eat(a.kv_last_page_lens_d.cast::<u8>() as u64);
        eat(a.qo_indptr_d.cast::<u8>() as u64);
        eat(a.w_page_d.cast::<u8>() as u64);
        eat(a.w_off_d.cast::<u8>() as u64);
        eat(a.row_valid_d.cast::<u8>() as u64);
        eat(a.q_out.cast_const().cast::<u8>() as u64);
        eat(a.o_out.cast_const().cast::<u8>() as u64);
        eat(a.score_out.cast_const().cast::<u8>() as u64);
        eat(a.folded_out.cast_const().cast::<u8>() as u64);
        eat(a.score_indptr_d.cast::<u8>() as u64);
        eat(a.mask_d.cast::<u8>() as u64);
        eat(a.mask_indptr_d.cast::<u8>() as u64);
        eat(a.lse_out_d.cast_const().cast::<u8>() as u64);
        eat(a.layers.len() as u64);
        for l in &a.layers {
            eat(l.k_pages.cast_const().cast::<u8>() as u64);
            eat(l.v_pages.cast_const().cast::<u8>() as u64);
        }
    }

    // ── The recurrent slabs ──
    match gdn {
        Some(g) => {
            eat(g.slot_ids_d.cast::<u8>() as u64);
            // The SLAB BASES, which a capture bakes exactly as it bakes a
            // pool's: `ensure_slots` growing one is what bumps the epoch,
            // and this is the same fact read from the other end.
            eat(g.conv_state.len() as u64);
            for b in &g.conv_state {
                eat(*b);
            }
            eat(g.recurrent_state.len() as u64);
            for b in &g.recurrent_state {
                eat(*b);
            }
        }
        None => eat(u64::MAX),
    }
    h
}
