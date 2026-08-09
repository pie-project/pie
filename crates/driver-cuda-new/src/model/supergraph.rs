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

use crate::cuda::{
    GraphExec, PredicateWord, SLOT_HAS_CUSTOM_MASK, SLOT_HAS_LORA, SLOT_HAS_STAGE_HOOKS,
    SLOT_HAS_WRITE_DESC, SLOT_TOKENS_GT, SLOT_TOKENS_LE, SLOT_WANTS_ATTN_SCORE, StreamRef,
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
        SLOT_WANTS_ATTN_SCORE => rows.iter().any(|r| r.wants_scores),
        SLOT_HAS_CUSTOM_MASK => rows.iter().any(|r| r.custom_mask),
        SLOT_HAS_STAGE_HOOKS => rows.iter().any(|r| r.hooked),
        SLOT_HAS_LORA => rows.iter().any(|r| r.lora),
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
pub fn fire_predicates(rows: &[Row], conds: &[CondRegion], preds: &mut PredicateWord) -> Result<()> {
    preds.clear();
    let mut written: HashMap<u32, bool> = HashMap::new();
    for c in conds {
        let Some(value) = predicate_of(c.slot, c.param, rows) else { continue };
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
    /// The fire class, as `FireClass as u8` — a plain integer because the
    /// key is a hash key and the trace type carries no `Hash`.
    pub fire: u8,
    /// Which loaded model this graph addresses. Two deployments' captures
    /// share no buffer, so they may not share a key.
    pub model: u64,
}

impl BucketKey {
    /// The key for a fire.
    #[must_use]
    pub const fn new(
        requests: u32,
        tokens: u32,
        fire: model_compiler::trace::FireClass,
        model: u64,
    ) -> Self {
        Self { requests, tokens, fire: fire as u8, model }
    }
}

/// The instantiated graphs, by bucket.
///
/// Deliberately not an LRU yet: a bucket set is small (the R×N shapes a
/// deployment actually fires) and evicting a graph while a launch is in
/// flight is a use-after-free rather than a miss, so eviction is a
/// decision that wants the replay path to exist first.
#[derive(Debug, Default)]
pub struct SupergraphCache {
    execs: HashMap<BucketKey, GraphExec>,
    hits: u64,
    misses: u64,
}

impl SupergraphCache {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The exec for `key`, if one is captured.
    pub fn get(&mut self, key: BucketKey) -> Option<&GraphExec> {
        if self.execs.contains_key(&key) {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
        self.execs.get(&key)
    }

    /// Install a freshly instantiated exec.
    pub fn insert(&mut self, key: BucketKey, exec: GraphExec) {
        self.execs.insert(key, exec);
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
    pub fn replay(&mut self, key: BucketKey, stream: StreamRef<'_>) -> Result<bool> {
        let Some(exec) = self.get(key) else { return Ok(false) };
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

    /// Hits and misses since construction, for the metric that says
    /// whether the union is actually folding anything.
    #[must_use]
    pub const fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_compiler::trace::FireClass;
    #[test]
    fn variant_bits_are_not_in_the_key() {
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
        assert_ne!(base, BucketKey::new(4, 4, FireClass::Prefill, 7));
        assert_ne!(base, BucketKey::new(4, 4, FireClass::Decode, 8));
    }

    #[test]
    fn a_miss_is_not_an_error() {
        let mut c = SupergraphCache::new();
        assert!(c.is_empty());
        assert!(c.get(BucketKey::new(1, 1, FireClass::Decode, 0)).is_none());
        assert_eq!(c.stats(), (0, 1));
    }
}
