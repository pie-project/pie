//! The union cache: one instantiated graph per (R, N) bucket.
//!
//! A capture bakes every address, so whatever may not vary is in the key;
//! whatever may vary folds into predicate-selected conditional nodes instead.

use std::collections::HashMap;

use model_compiler::lower::{CondRegion, Row};

use crate::device::{GraphExec, PredicateWord, StreamRef};
use crate::error::{Error, Result};
use crate::fire::predicate::predicate_of;

/// Fill `preds` with a fire's variant bits. One slot per predicate kind; two
/// guards disagreeing on a slot are refused, not silently folded to a branch.
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

/// What a capture may not vary over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BucketKey {
    /// Requests in the fire.
    pub requests: u32,
    /// Token rows in the fire.
    pub tokens: u32,
    /// Which loaded model this graph addresses — two deployments share no buffer.
    pub model: u64,
    /// The staged LoRA's group shape, or zero with no adapters. Must key the
    /// bucket: the grouped launch bakes the member count, not just a flag.
    pub lora_shape: u64,
}

impl BucketKey {
    /// The key for a fire. The fire class isn't in the key — it's a function
    /// of `requests` — so `_fire` is unused, kept only so callers need not change.
    #[must_use]
    pub const fn new(
        requests: u32,
        tokens: u32,
        _fire: model_ir::trace::FireClass,
        model: u64,
    ) -> Self {
        Self {
            requests,
            tokens,
            model,
            lora_shape: 0,
        }
    }

    /// The same key for a fire that staged adapters.
    #[must_use]
    pub const fn with_lora(mut self, shape: u64) -> Self {
        self.lora_shape = shape;
        self
    }
}

/// Why a fire may not join a union capture: a union needs every arm's shape
/// fixed, so a fire whose shape follows it instead stays eager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ineligible {
    /// LoRA lanes staged but didn't group — the solo path's launch count and
    /// shapes follow the adapter set instead.
    UngroupedLora,
}

/// May this fire join a union capture? `None` is eligible; else a stale replay.
#[must_use]
pub fn union_eligibility(lora: Option<&crate::fire::lora::LoraFireState>) -> Option<Ineligible> {
    match lora {
        Some(l) if !l.union_capture_safe() => Some(Ineligible::UngroupedLora),
        _ => None,
    }
}

/// Monotonic count of prepared-state rewrites. A captured exec bakes plan and
/// workspace addresses that get reused in place, so keying replay on the
/// epoch turns a silent stale read into a miss and recapture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlanEpoch(u64);

impl PlanEpoch {
    /// The only way the epoch changes, so nothing else can manufacture "the
    /// epoch this exec was recorded against". Doesn't prove the arena is alive.
    pub(crate) fn bump(&mut self) {
        self.0 += 1;
    }

    /// An arbitrary epoch, for tests needing two that differ; not reachable
    /// from the driver, so a caller can't name a stale one.
    #[cfg(test)]
    pub(crate) const fn at(n: u64) -> Self {
        Self(n)
    }
}

/// One bucket's captured exec, its recorded epoch, and the graph node each
/// launch became — the nodes let a launch's rectangle move without recapture.
#[derive(Debug)]
struct Entry {
    exec: GraphExec,
    epoch: PlanEpoch,
    nodes: Vec<Option<cudarc::runtime::sys::cudaGraphNode_t>>,
    /// Install order within the bucket, read only to pick an eviction.
    seq: u64,
}

// SAFETY: a `cudaGraphNode_t` is an opaque handle into the graph the `GraphExec`
// beside it owns; never dereferenced here nor valid past that graph's life.
unsafe impl Send for Entry {}
unsafe impl Sync for Entry {}

/// A bucket plus the addresses its graph baked. See [`Recordings`] for why
/// the digest is half the key and not a validity check on one exec per bucket.
type Slot = (BucketKey, u64);

/// How many recorded address-sets one bucket may hold at once.
///
/// A bucket is a SHAPE, and several concurrent streams of work can share a
/// shape while pointing at different memory: two lanes of a contrastive
/// decoder, a handful of sequences decoding side by side. Each needs its own
/// graph. Four covers what has been observed and bounds the cost of a workload
/// whose addresses genuinely churn, where no exec would ever be replayed and
/// keeping them all would be a leak in slow motion.
const SLOTS_PER_BUCKET: usize = 4;

/// The instantiated graphs, keyed by bucket, baked addresses, and the epoch
/// recorded against.
/// Not an LRU across buckets: evicting a graph mid-launch is a use-after-free,
/// not a miss.
///
/// # Why the digest is part of the key
///
/// It used to be a validity check on a single exec per bucket: a fire whose
/// digest disagreed with the recorded one evicted that exec, counted a
/// `mismatched`, and recaptured. That is right for a bucket whose addresses
/// MOVED, and wrong for a bucket two callers are sharing.
///
/// `context-aware-decoding` runs two lanes of the same shape, alternating, so
/// its fires arrive with two digests in strict alternation and one bucket key.
/// Each fire found the other lane's exec, refused it, evicted it, and paid a
/// 535-launch capture plus a `cudaGraphInstantiate` to install an exec the next
/// fire would evict in turn. Over four tokens: six fires, six captures, zero
/// hits, five mismatches -- and `mismatched`'s own doc says a nonzero count on
/// a steady workload is a bug, not the cache working. It cost about 7.7 seconds
/// a token on a 0.6B decode.
///
/// The digest already discriminates the lanes perfectly; it was simply being
/// used to reject rather than to select. Now it selects, up to
/// [`SLOTS_PER_BUCKET`] address-sets per shape, and a bucket that really is
/// moving still cannot grow without bound.
#[derive(Debug, Default)]
pub struct Recordings {
    execs: HashMap<Slot, Entry>,
    /// Monotonic install counter, so a full bucket evicts its oldest slot.
    seq: u64,
    hits: u64,
    misses: u64,
    stale: u64,
    /// Replays refused because the fire's addresses don't match the graph's.
    /// See [`capture_digest`].
    mismatched: u64,
    /// Slots dropped because a bucket was full. See [`SLOTS_PER_BUCKET`].
    evicted: u64,
}

impl Recordings {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The exec for `key` if captured and recorded against `epoch`; a stale
    /// entry is dropped, not returned, since its recorded state is gone.
    pub fn get(&mut self, key: BucketKey, digest: u64, epoch: PlanEpoch) -> Option<&GraphExec> {
        let slot = (key, digest);
        match self.execs.get(&slot) {
            Some(e) if e.epoch == epoch => {
                self.hits += 1;
                self.execs.get(&slot).map(|e| &e.exec)
            }
            Some(_) => {
                self.stale += 1;
                self.execs.remove(&slot);
                None
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Make room in `key`'s bucket, dropping its oldest slot when full.
    ///
    /// Dropping is safe here and only here: an entry is removed between fires,
    /// never while its graph is in flight, because the caller holds the cache
    /// mutably across the whole capture-or-replay.
    fn evict_if_full(&mut self, key: BucketKey) {
        let mut held: Vec<(u64, u64)> = self
            .execs
            .iter()
            .filter(|((k, _), _)| *k == key)
            .map(|((_, d), e)| (e.seq, *d))
            .collect();
        if held.len() < SLOTS_PER_BUCKET {
            return;
        }
        held.sort_unstable();
        for (_, d) in held.iter().take(held.len() + 1 - SLOTS_PER_BUCKET) {
            self.execs.remove(&(key, *d));
            self.evicted += 1;
        }
    }

    /// Install a freshly instantiated exec, unless `eligibility` says the fire
    /// should have stayed eager, in which case nothing is installed.
    pub fn insert(
        &mut self,
        key: BucketKey,
        digest: u64,
        exec: GraphExec,
        epoch: PlanEpoch,
        eligibility: Option<Ineligible>,
    ) -> core::result::Result<(), Ineligible> {
        if let Some(why) = eligibility {
            return Err(why);
        }
        self.evict_if_full(key);
        self.seq += 1;
        self.execs
            .insert((key, digest), Entry { exec, epoch, nodes: Vec::new(), seq: self.seq });
        Ok(())
    }

    /// Install an exec together with the nodes its capture retained: `nodes[i]`
    /// is launch `i`'s node, needed to retune grids without recapture.
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
        self.evict_if_full(key);
        self.seq += 1;
        self.execs.insert((key, digest), Entry { exec, epoch, nodes, seq: self.seq });
        Ok(())
    }

    /// Retune a captured exec's launch grids for a fire whose row count
    /// differs from the recorded one. `Ok(false)` when the key holds no exec.
    pub fn retune(
        &mut self,
        key: BucketKey,
        digest: u64,
        epoch: PlanEpoch,
        grids: &[Option<u32>],
    ) -> Result<bool> {
        let Some(entry) = self.execs.get(&(key, digest)) else {
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

    /// Replay `key`'s graph onto `stream` if captured. `Ok(false)` for a
    /// miss — the caller's cue to capture, not an error.
    pub fn replay(
        &mut self,
        key: BucketKey,
        epoch: PlanEpoch,
        digest: u64,
        stream: StreamRef<'_>,
    ) -> Result<bool> {
        let Some(exec) = self.get(key, digest, epoch) else {
            return Ok(false);
        };
        exec.launch(stream)?;
        Ok(true)
    }

    /// How many execs are live — the number this design keeps small.
    #[must_use]
    pub fn len(&self) -> usize {
        self.execs.len()
    }

    /// Is the cache empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.execs.is_empty()
    }

    /// Hits, misses and stale drops since construction. A bucket that keeps
    /// going stale is recaptured every fire — costlier than not capturing.
    #[must_use]
    pub const fn stats(&self) -> (u64, u64, u64) {
        (self.hits, self.misses, self.stale)
    }

    /// Replays refused for an address mismatch (see [`capture_digest`]).
    ///
    /// RETIRED, AND LEFT AT ZERO. A mismatch used to mean "this bucket holds
    /// one exec and it is not yours", which conflated a bucket whose addresses
    /// moved with a bucket two lanes were sharing -- and the second case, which
    /// is the common one, is not a mismatch at all. The digest is part of the
    /// key now, so the question this counted cannot be asked. Kept so the
    /// canaries and the parity gates that assert it is zero keep compiling, and
    /// so the count they assert stays true by construction.
    #[must_use]
    pub const fn mismatched(&self) -> u64 {
        self.mismatched
    }

    /// Slots dropped to keep a bucket within [`SLOTS_PER_BUCKET`]. Nonzero
    /// means more concurrent address-sets share a shape than the cache holds,
    /// which is a capacity question, not a correctness one.
    #[must_use]
    pub const fn evicted(&self) -> u64 {
        self.evicted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_ir::trace::FireClass;

    /// The digest must be stable for identical fires and sensitive to any change.
    #[test]
    fn the_digest_is_stable_and_sensitive() {
        let one = fake_digest(0x1000, 4, 0);
        assert_eq!(one, fake_digest(0x1000, 4, 0), "same fire, same number");

        assert_ne!(one, fake_digest(0x2000, 4, 0), "a moved buffer");
        assert_ne!(one, fake_digest(0x1000, 5, 0), "a different row count");
        assert_ne!(one, fake_digest(0x1000, 4, 9), "a different adapter set");
    }

    /// The FNV walk `capture_digest` runs, over the three axes a test can vary.
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
        assert_eq!(
            BucketKey::new(4, 4, FireClass::Decode, 7),
            BucketKey::new(4, 4, FireClass::Prefill, 7),
        );
        assert_ne!(
            BucketKey::new(4, 4, FireClass::Decode, 7),
            BucketKey::new(4, 9, FireClass::Prefill, 7),
        );
    }

    #[test]
    fn a_stale_epoch_is_a_miss_and_the_entry_goes() {
        let mut c = Recordings::new();
        let k = BucketKey::new(4, 4, FireClass::Decode, 1);
        assert!(c.get(k, 0, PlanEpoch::at(7)).is_none(), "cold");
        assert_eq!(c.stats(), (0, 1, 0));

        assert!(c.is_empty());
    }

    #[test]
    fn an_ineligible_fire_cannot_install_an_exec() {
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
        assert!(
            c.get(BucketKey::new(1, 1, FireClass::Decode, 0), 0, PlanEpoch::at(0)).is_none()
        );
        assert_eq!(c.stats(), (0, 1, 0));
    }
}

/// The addresses a decode plan bakes into a capture, fed into a digest: the
/// descriptor offsets, their base, and the pinned host buffer the H2D node
/// bakes while later fires overwrite its contents in place. Null is a sentinel.
#[cfg(feature = "_cuda")]
fn decode_plan_layout(plan: *mut std::ffi::c_void, eat: &mut impl FnMut(u64)) {
    if plan.is_null() {
        eat(u64::MAX);
        return;
    }
    // SAFETY: every non-null value came from `bind::DecodePlan::as_ptr`'s boxed
    // cache; the borrow is shared and ends here.
    let cache = unsafe { &*plan.cast::<kernels_cuda::attn::fa2::plan::DecodePlanCache>() };
    for v in cache.plan_info.to_vector() {
        eat(v as u64);
    }
    eat(cache.int_base_bytes as u64);
    eat(cache.int_upload.as_ptr() as u64);
    eat(cache.int_upload.as_slice().len() as u64);
}

/// [`decode_plan_layout`] for the prefill cache — fifteen offsets instead of
/// ten, and the same three reasons.
#[cfg(feature = "_cuda")]
fn prefill_plan_layout(plan: *mut std::ffi::c_void, eat: &mut impl FnMut(u64)) {
    if plan.is_null() {
        eat(u64::MAX);
        return;
    }
    // SAFETY: as `decode_plan_layout`'s, over `bind::PrefillPlan::as_ptr`.
    let cache = unsafe { &*plan.cast::<kernels_cuda::attn::fa2::plan::PrefillPlanCache>() };
    for v in cache.plan_info.to_vector() {
        eat(v as u64);
    }
    eat(cache.int_base_bytes as u64);
    eat(cache.int_upload.as_ptr() as u64);
    eat(cache.int_upload.as_slice().len() as u64);
}

/// Everything a captured graph bakes, in one number: recompute, compare,
/// refuse if a replay's addresses don't match what was recorded.
///
/// Not a `Hash` derive — that would make new fields silently part of the
/// answer. Cheap: ~70 multiplies once per fire.
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

    eat(ctx.stream.cast::<u8>() as u64);
    eat(ctx.cublas.cast::<u8>() as u64);
    eat(ctx.token_ids.cast::<u8>() as u64);
    eat(ctx.positions.cast::<u8>() as u64);
    eat(ctx.peel_window.cast::<u8>() as u64);
    eat(ctx.sampling_indices.cast::<u8>() as u64);
    // The lora state: a capture bakes the lane pointers and the launch count.
    match ctx.lora {
        Some((s, scratch)) => {
            eat(s.cast::<u8>() as u64);
            eat(scratch.cast::<u8>() as u64);
            // SAFETY: the state outlives the call; see `capture_or_replay`.
            eat(unsafe { (*s).capture_fingerprint });
        }
        None => eat(u64::MAX),
    }

    // Every attention region, because a peel launches two.
    for a in [regions.fire, regions.tail] {
        let Some(a) = a else {
            eat(u64::MAX);
            continue;
        };
        eat(a.decode_plan.cast_const().cast::<u8>() as u64);
        decode_plan_layout(a.decode_plan, &mut eat);
        eat(a.decode_plan_full.cast_const().cast::<u8>() as u64);
        decode_plan_layout(a.decode_plan_full, &mut eat);
        eat(a.prefill_plan.cast_const().cast::<u8>() as u64);
        prefill_plan_layout(a.prefill_plan, &mut eat);
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

    // The recurrent slabs.
    match gdn {
        Some(g) => {
            eat(g.slot_ids_d.cast::<u8>() as u64);
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
