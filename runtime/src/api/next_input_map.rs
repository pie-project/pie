//! #6/#21 run-ahead next-input carrier — host-side link-id (L) map.
//!
//! The guest declares only `next-inputs(positions)` on a *producer* pass: "carry
//! this pass's sampled token into the NEXT pass's input at these `positions`."
//! The guest threads no link-ids — the **host** owns the global monotonic link id
//! `L` and computes the producer's source row. This module populates the driver
//! carrier (`pipeline_source_link` / `next_input_{producer_links,src_rows,
//! dest_slots,free_links}`) on the `ForwardRequest`; the CUDA executor consumes
//! it UNCHANGED (retain → inject → free). See `run-ahead-scheduler-spec` §2.1.
//!
//! A pass is BOTH the implicit *consumer* of the prior producer's pending carry
//! AND (if it declared `next-inputs`) a *producer* for the next pass. The
//! producer→consumer association is "consecutive passes on the same context"; the
//! cross-pass state ([`PendingNextInput`]) is held by the caller (per context).

use pie_driver_abi::ForwardRequest;
use std::collections::HashMap;

/// The link dependencies a pass established at submit (the #23 overlap write-log
/// inputs): the link it PRODUCED (if it declared `next-inputs`) and the prior
/// producer link it CONSUMED (injected from — same-context only). Threaded onto
/// the pass's in-flight handle so its finalize can resolve the overlap cascade
/// (a consumer aborts if the producer it injected from aborted) and publish its
/// own outcome for its consumer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NextInputDeps {
    /// The link this pass produced for the next pass (`None` = not a producer).
    pub produced: Option<u32>,
    /// The producer link this pass injected from (`None` = not a real consumer;
    /// a context-mismatch drop is NOT a dependency — it injects nothing).
    pub consumed: Option<u32>,
}

/// Resolved outcome of a producer pass, published at its finalize for its
/// consumer's #23 cascade check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkOutcome {
    Committed,
    Aborted,
}

/// #23 abort-isolation write-log: tracks each in-flight producer link's resolved
/// outcome so a consumer's finalize can cascade-abort if the producer it injected
/// from aborted — preventing a poisoned generation (the consumer ran on the
/// producer's invalid sampled token) from committing.
///
/// **FAIL-CLOSED (LOCK 1):** an unresolved consumed link (absent from the map) is
/// treated as `Aborted` — only an explicit `Committed` clears the cascade, so a
/// finalize-ordering violation can never silently commit a poisoned generation.
///
/// Bounded: a consumer removes the link it reads; the only lingering entry is a
/// terminal producer's (no consumer), cleared at the next `generate()` boundary
/// via [`clear`](Self::clear).
#[derive(Debug, Default)]
pub struct OverlapLinkLog {
    status: std::collections::HashMap<u32, LinkOutcome>,
}

impl OverlapLinkLog {
    /// Resolve a pass's finalize against the overlap links and return the
    /// **effective** success the caller commits/aborts on.
    ///
    /// 1. Consumer cascade (fail-closed): if this pass injected from a producer
    ///    link that is not explicitly `Committed` (aborted OR unresolved), force
    ///    `effective = false` — the consumer's txn + KV must roll back.
    /// 2. Producer record: publish this pass's *effective* outcome under its
    ///    produced link, so its consumer sees it — chaining the cascade downstream
    ///    (a poisoned pass poisons the whole dependent run until a fresh prime).
    pub fn finalize(&mut self, driver_success: bool, deps: NextInputDeps) -> bool {
        let producer_poisoned = deps
            .consumed
            .is_some_and(|l| self.status.remove(&l) != Some(LinkOutcome::Committed));
        let effective = driver_success && !producer_poisoned;
        if let Some(prod) = deps.produced {
            self.status.insert(
                prod,
                if effective {
                    LinkOutcome::Committed
                } else {
                    LinkOutcome::Aborted
                },
            );
        }
        effective
    }

    /// Drop all tracked links — a fresh `generate()` starts clean (also clears
    /// any lingering terminal-producer entry so the map stays bounded).
    pub fn clear(&mut self) {
        self.status.clear();
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.status.is_empty()
    }
}

/// The pending carry from the prior producer pass, applied to the next pass (the
/// implicit consumer) on the same context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingNextInput {
    /// Global monotonic link id assigned to the producer (`> 0`).
    pub link: u32,
    /// Dest slots in the consumer's input where the producer's sampled token is
    /// injected (the guest's `next-inputs(positions)`, in the next pass's coords).
    pub positions: Vec<u32>,
    /// The producer's forward row count: its sampled output is at row
    /// `n_rows - 1` (prefill prime → last row; single-row decode → 0).
    pub n_rows: u32,
    /// The producer's context identity (the KV working-set rep). The carryover is
    /// strictly between consecutive passes on the SAME context, so a pending is
    /// only consumed by a pass with a matching `context_id`; a pass on a different
    /// context drops it (freeing the retained buffer) rather than injecting it —
    /// otherwise a terminal producer's dangling carry leaks into the next context.
    pub context_id: u32,
    /// Drafts-channel routing (§9): `0 = PrevSample` (retain `pi.sampled`; the
    /// consumer injects from the producer's last forward row `n_rows-1`) / `1 =
    /// PrevDrafts` (retain the composed `[k+1]` `[seed, drafts]` window; the consumer
    /// injects retained-buffer row `i` → the `i`-th declared window slot — static
    /// `src_rows=[0..=k]`, §8.3). Recorded from the producer's `pipeline_source_kind`.
    pub source_kind: u8,
}

/// `pipeline_source_kind` values (§9): the single-token sample carrier (default).
pub const PREV_SAMPLE: u8 = 0;
/// `pipeline_source_kind` = the `[k+1]` `[seed, drafts]` window (MTP drafts channel).
pub const PREV_DRAFTS: u8 = 1;

/// Process-global monotonic carrier link-id source. The link is the key into the
/// DRIVER's ONE GLOBAL `retained_next_input` map, so it MUST be unique across ALL
/// instances: a per-instance counter (each init 0) makes concurrent instances
/// allocate the SAME ids (1,2,3,…) → they collide in the global map → cross-inject
/// / retain-evict → the R>1 concurrent-decode corruption (thrust-2 Bug#2). Sparse
/// global ids are fine (the map is keyed, not indexed); the map frees drained
/// entries so a 2^32 wrap can't alias a live id.
static CARRIER_LINK_COUNTER: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(1);

/// Allocate the next globally-unique carrier link (never `0`, the not-a-source
/// sentinel — skipped on the rare full wrap).
fn next_carrier_link() -> u32 {
    let l = CARRIER_LINK_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if l == 0 {
        CARRIER_LINK_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    } else {
        l
    }
}

/// Populate `req`'s next-input carrier for one forward pass.
///
/// Call once per pass at execute-time eager-submit, **after** `input-tokens` is
/// staged (it reads `req.token_ids.len()` for the producer row count) and
/// **before** submit. `pending` is the per-context carry state owned by the caller
/// (e.g. `InstanceState`), keyed per `context_id`. Link ids are drawn from a
/// PROCESS-GLOBAL counter ([`next_carrier_link`]) so they never collide across
/// concurrent instances in the driver's global retained map. `positions` is this
/// pass's declared `next-inputs(positions)` (empty ⇒ not a producer); `context_id`
/// identifies this pass's context (the KV working-set rep) so the carryover stays
/// scoped to consecutive passes on the same context.
pub fn apply_next_input_carrier(
    pending: &mut HashMap<u32, PendingNextInput>,
    req: &mut ForwardRequest,
    positions: &[u32],
    context_id: u32,
) -> NextInputDeps {
    let mut deps = NextInputDeps::default();
    // ── CONSUMER role ───────────────────────────────────────────────
    // Inject THIS context's prior producer sample. The pending is keyed per
    // context, so a concurrently-decoding sibling pipeline's producer can never
    // clobber this context's carry (thrust-2 Bug#2) — the lookup is by
    // `context_id`, so there is no cross-context mismatch to drop. `src_row =
    // n_rows-1` is the producer's last forward row (the decode position). One
    // consumer per link ⇒ free it here (host refcount = 1).
    if let Some(p) = pending.remove(&context_id) {
        if p.source_kind == PREV_DRAFTS {
            // Drafts-channel `[k+1]` window (§8.3): the retain pre-composed the
            // `[seed, drafts]` buffer, so retained-buffer row `i` → the `i`-th
            // declared window slot (static `src_rows=[0..=k]`), NOT the single-token
            // `n_rows-1`. `dselect`: the driver reads row `i` of the retained window.
            for (i, &pos) in p.positions.iter().enumerate() {
                req.push_next_input_link(p.link, i as u32, pos);
            }
        } else {
            // PrevSample single-token carrier: `src_row = n_rows-1` (the producer's
            // last forward row, the decode position) for every declared slot.
            let src_row = p.n_rows.saturating_sub(1);
            // TODO(multi-seq): single `src_row` for all positions is correct for the
            // single-sequence one-ahead path (one position, the producer's last row).
            for &pos in &p.positions {
                req.push_next_input_link(p.link, src_row, pos);
            }
        }
        req.push_next_input_free_link(p.link);
        // #23: a real cross-pass dependency — record the producer link this pass
        // injected from so its finalize can cascade-abort if the producer aborted.
        deps.consumed = Some(p.link);
    }

    // ── PRODUCER role ───────────────────────────────────────────────
    // If this pass declared `next-inputs`, assign a fresh monotonic link
    // (`0` = the not-a-source sentinel), mark this pass as the retain source, and
    // stash the carry under THIS context for its next pass to consume.
    if !positions.is_empty() {
        let link = next_carrier_link();
        // Drafts-channel routing (§9): the guest set `pipeline_source_kind` (echo's
        // `set_pipeline_source_kind(1)`) BEFORE this call; carry it on the pending so
        // the CONSUMER picks the window vs single-token src_row mapping. `0` default.
        let source_kind = req.pipeline_source_kind;
        req.set_pipeline_source_link(link);
        let n_rows = req.token_ids.len() as u32;
        deps.produced = Some(link);
        pending.insert(
            context_id,
            PendingNextInput {
                link,
                positions: positions.to_vec(),
                n_rows,
                context_id,
                source_kind,
            },
        );
    }
    deps
}

/// Drop this context's dangling carrier pending at a fresh-`generate()` boundary
/// (#26, `forward-pass.fresh-generate`).
///
/// When a generator starts a fresh `generate()` on a context that previously
/// *stop*-terminated with an un-consumed carry, the terminal producer left a
/// dangling `pending` whose intended same-context consumer never fired. The new
/// generate's first pass must drop it so it isn't injected into the new prefill.
/// (The count-predictable *max*-boundary terminal is handled loop-side by not
/// emitting `next-inputs`; *cross-context* dangling is handled by
/// [`apply_next_input_carrier`]'s mismatch branch — this closes the remaining
/// *same-context* `generate()`-restart path.)
///
/// Clears only a pending that belongs to THIS `context_id` (a different context's
/// pending is left for the cross-context drop). Returns the dropped producer's
/// link, if any, so the caller frees its retained device buffer via
/// `req.push_next_input_free_link(link)` (no leak) on the fresh pass — call this
/// BEFORE [`apply_next_input_carrier`] so the fresh pass neither injects nor
/// re-frees the stale carry.
pub fn clear_pending_for_context(
    pending: &mut HashMap<u32, PendingNextInput>,
    context_id: u32,
) -> Option<u32> {
    pending.remove(&context_id).map(|p| p.link)
}

/// Depth-k run-ahead rollback (spec §4): record a carrier producer link created on
/// a context, so a later fresh-`generate()` can free-ALL of them. No-op for the
/// no-KV pass (context `0` is never a carrier producer/consumer).
pub fn record_produced_link(
    produced: &mut HashMap<u32, Vec<u32>>,
    context_id: u32,
    link: u32,
) {
    if context_id != 0 {
        produced.entry(context_id).or_default().push(link);
    }
}

/// Depth-k run-ahead rollback (spec §4): take (removing the entry) ALL carrier
/// producer links created on a context, for the fresh-`generate()` free-all.
///
/// A run-ahead STOP over-shoot drops ≤`depth`−1 speculative fires; the last
/// COMMITTED fire's retained carry is then orphaned — its drain-gated free rode the
/// FIRST dropped fire's request, which never ran — while `pending` points at a
/// never-retained over-shot link. Freeing every produced link on the context
/// reclaims it. Safe: the driver free is idempotent (find-or-skip) + drain-gated,
/// so re-freeing an already-freed or still-in-flight link is a no-op.
pub fn take_produced_links_for_context(
    produced: &mut HashMap<u32, Vec<u32>>,
    context_id: u32,
) -> Vec<u32> {
    produced.remove(&context_id).unwrap_or_default()
}

/// Depth-k run-ahead rollback (spec §4): a carrier producer link's SINGLE consumer
/// just freed it (host refcount = 1 — `apply_next_input_carrier` emitted its
/// `next_input_free_link` on consume) — so drop it from the produced set here. This
/// keeps `produced` == exactly the ORPHAN set (a terminal producer with no consumer,
/// or an over-shot's un-run consumer), so the fresh-`generate()` free-all reclaims
/// ONLY genuine orphans. Without it the free-all re-emits the whole chain's N−1
/// already-freed links every generation (the "too broad" #17 shape — wrong per se:
/// stale frees on links whose one legitimate free already fired). A link id is
/// globally unique (`next_carrier_link`), so at most one entry matches. Empty entry
/// removed (mirrors `take`'s cleanup). No-op if the link isn't tracked.
pub fn remove_produced_link(
    produced: &mut HashMap<u32, Vec<u32>>,
    context_id: u32,
    link: u32,
) {
    if let Some(links) = produced.get_mut(&context_id) {
        if let Some(pos) = links.iter().position(|&l| l == link) {
            links.remove(pos);
        }
        if links.is_empty() {
            produced.remove(&context_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prime (2-token prefill) producer → decode consumer: the one-ahead pattern.
    /// Verifies host L-assignment, the `src_row = n_rows-1` prime fix, the implicit
    /// consumer's inject+free, and the carry-forward.
    #[test]
    fn prime_then_consumer_carryover() {
        let mut pending: HashMap<u32, PendingNextInput> = HashMap::new();
        const CTX: u32 = 7;

        // Pass t (prime): prefill 2 prompt tokens, sample at the last row, carry to
        // the next pass's input slot 5.
        let mut req_p = ForwardRequest::default();
        req_p.token_ids = vec![100, 200];
        apply_next_input_carrier(&mut pending, &mut req_p, &[5], CTX);

        // Producer marked as a source (a fresh global link); no consumer links
        // (pending was empty). Link ids come from a PROCESS-GLOBAL counter, so we
        // capture the actual value rather than assert an absolute id.
        let link_p = req_p.pipeline_source_link;
        assert_ne!(link_p, 0, "a producer pass gets a non-zero carrier link");
        assert!(req_p.next_input_producer_links.is_empty());
        assert_eq!(
            pending.get(&CTX),
            Some(&PendingNextInput { link: link_p, positions: vec![5], n_rows: 2, context_id: CTX, source_kind: 0 })
        );

        // Pass t+1 (consumer + next producer): 1 placeholder token, carry to slot 6.
        let mut req_c = ForwardRequest::default();
        req_c.token_ids = vec![0];
        apply_next_input_carrier(&mut pending, &mut req_c, &[6], CTX);

        // Consumer of the prime's link: src_row = n_rows-1 = 1 (the prime's last
        // row), dest 5, freed (single consumer).
        assert_eq!(req_c.next_input_producer_links, vec![link_p]);
        assert_eq!(req_c.next_input_src_rows, vec![1]);
        assert_eq!(req_c.next_input_dest_slots, vec![5]);
        assert_eq!(req_c.next_input_free_links, vec![link_p]);
        // And itself a producer: a new (distinct) global link, single-row ⇒ next
        // consumer's src_row=0.
        let link_c = req_c.pipeline_source_link;
        assert_ne!(link_c, 0);
        assert_ne!(link_c, link_p, "each producer gets a distinct global link");
        assert_eq!(
            pending.get(&CTX),
            Some(&PendingNextInput { link: link_c, positions: vec![6], n_rows: 1, context_id: CTX, source_kind: 0 })
        );
    }

    /// A terminal producer's dangling carry must NOT leak into another context.
    /// With the PER-CONTEXT pending map, a pass on a different `context_id` simply
    /// reads its OWN (absent) carry — it neither injects NOR touches context A's
    /// pending (unlike the old shared-slot design, which let any next pass drop
    /// A's carry). A's dangling carry is cleared by A's OWN `fresh-generate`
    /// (`clear_pending_for_context`, #26) or its context teardown. Regression for
    /// the cross-context `pi.tokens[0]` corruption AND the Bug#2 concurrent clash.
    #[test]
    fn dangling_carry_does_not_leak_across_contexts() {
        let mut pending: HashMap<u32, PendingNextInput> = HashMap::new();

        // Context A's terminal producer declares a carry that never gets consumed
        // on A (the loop ended).
        let mut req_a = ForwardRequest::default();
        req_a.token_ids = vec![0];
        apply_next_input_carrier(&mut pending, &mut req_a, &[0], 10);
        let link_a = req_a.pipeline_source_link;
        assert_ne!(link_a, 0);
        assert!(pending.contains_key(&10));

        // Context B's first (prefill) pass: a DIFFERENT context_id. It reads its
        // OWN (empty) carry → NO inject, and it does NOT touch A's pending
        // (per-context isolation — the fix for the concurrent-carrier clash).
        let mut req_b = ForwardRequest::default();
        req_b.token_ids = vec![100, 200];
        apply_next_input_carrier(&mut pending, &mut req_b, &[], 20);
        assert!(req_b.next_input_producer_links.is_empty());
        assert!(req_b.next_input_dest_slots.is_empty());
        assert!(req_b.next_input_free_links.is_empty()); // B does NOT free A's carry
        assert!(pending.contains_key(&10)); // A's carry survives B (isolated)

        // A's own fresh-generate clears A's dangling carry (returns its link to free).
        assert_eq!(clear_pending_for_context(&mut pending, 10), Some(link_a));
        assert!(pending.is_empty());
    }

    /// A pass with no `next-inputs` is neither source nor (absent a pending carry)
    /// consumer — the carrier stays empty.
    #[test]
    fn non_producer_pass_is_inert() {
        let mut pending: HashMap<u32, PendingNextInput> = HashMap::new();
        let mut req = ForwardRequest::default();
        req.token_ids = vec![42];
        apply_next_input_carrier(&mut pending, &mut req, &[], 1);
        assert_eq!(req.pipeline_source_link, 0);
        assert!(req.next_input_producer_links.is_empty());
        assert!(pending.is_empty());
    }

    /// **Bug#2 concurrent-carrier RED repro (thrust-2, bravo).** Two INDEPENDENT
    /// run-ahead pipelines (contexts A + B) co-scheduled: under concurrent submit,
    /// BOTH producers fire before BOTH consumers (`P_a, P_b, C_a, C_b`). With a
    /// SINGLE shared `pending` slot, `P_b` overwrites `P_a`'s carry, then each
    /// consumer reads the OTHER context's pending → context-mismatch → drop → NO
    /// inject: EVERY co-batched consumer fire arrives with `next_input_producer_
    /// links` empty (charlie's `ef69dd70` trace: producer_links=0 on every R>1
    /// fire). Each consumer MUST inject its OWN producer's token. This fails on the
    /// single-`Option` pending; the fix is a per-context pending map.
    #[test]
    fn concurrent_pipelines_carry_survives_co_submit() {
        const CTX_A: u32 = 10;
        const CTX_B: u32 = 20;
        let mut pending: HashMap<u32, PendingNextInput> = HashMap::new();

        // Concurrent co-submit order: both producers, THEN both consumers.
        let mut p_a = ForwardRequest::default();
        p_a.token_ids = vec![0];
        apply_next_input_carrier(&mut pending, &mut p_a, &[0], CTX_A);
        let link_a = p_a.pipeline_source_link;

        let mut p_b = ForwardRequest::default();
        p_b.token_ids = vec![0];
        apply_next_input_carrier(&mut pending, &mut p_b, &[0], CTX_B);
        let link_b = p_b.pipeline_source_link;

        let mut c_a = ForwardRequest::default();
        c_a.token_ids = vec![0];
        apply_next_input_carrier(&mut pending, &mut c_a, &[0], CTX_A);

        let mut c_b = ForwardRequest::default();
        c_b.token_ids = vec![0];
        apply_next_input_carrier(&mut pending, &mut c_b, &[0], CTX_B);

        assert_eq!(
            c_a.next_input_producer_links, vec![link_a],
            "pipeline A's consumer must inject A's OWN producer token (not drop under co-submit)"
        );
        assert_eq!(
            c_b.next_input_producer_links, vec![link_b],
            "pipeline B's consumer must inject B's OWN producer token (not drop under co-submit)"
        );
    }

    /// **Bug#2 link-collision RED repro (thrust-2, bravo — charlie's exact pin).**
    /// Two INDEPENDENT instances (separate `InstanceState` ⇒ separate link
    /// counters, both init 0) each run a run-ahead producer. The link id is the key
    /// into the DRIVER's ONE GLOBAL `retained_next_input` map, so it MUST be
    /// globally unique across instances. With per-instance counters both start at 0
    /// → BOTH allocate link 1 → COLLIDE in the global map: instance B's consumer
    /// injects A's retained sample (9707) / A's free-link evicts B's (0). 1 pipeline
    /// = no collision = 8/8; 8 concurrent = 0/8. The fix is a process-global counter.
    #[test]
    fn concurrent_instances_get_globally_unique_carrier_links() {
        // Instance A + instance B: separate per-instance link counters, both 0.
        let mut pending_a: HashMap<u32, PendingNextInput> = HashMap::new();
        let mut pending_b: HashMap<u32, PendingNextInput> = HashMap::new();

        let mut p_a = ForwardRequest::default();
        p_a.token_ids = vec![0];
        apply_next_input_carrier(&mut pending_a, &mut p_a, &[0], 1);

        let mut p_b = ForwardRequest::default();
        p_b.token_ids = vec![0];
        apply_next_input_carrier(&mut pending_b, &mut p_b, &[0], 1);

        assert_ne!(
            p_a.pipeline_source_link, p_b.pipeline_source_link,
            "two instances' carrier link ids must be globally unique — they key the \
             driver's ONE global retained_next_input map"
        );
    }

    /// link to free; a different context's pending is left for the cross-context
    /// drop; no pending ⇒ nothing to free.
    #[test]
    fn clear_pending_for_context_drops_same_context_only() {
        // Same-context dangling carry → cleared, link returned for the free.
        let mut pending: HashMap<u32, PendingNextInput> = HashMap::new();
        pending.insert(
            42,
            PendingNextInput { link: 3, positions: vec![0], n_rows: 1, context_id: 42, source_kind: 0 },
        );
        assert_eq!(clear_pending_for_context(&mut pending, 42), Some(3));
        assert!(pending.is_empty());

        // Different-context pending → untouched (keyed removal), nothing returned.
        let mut pending: HashMap<u32, PendingNextInput> = HashMap::new();
        pending.insert(
            10,
            PendingNextInput { link: 5, positions: vec![0], n_rows: 1, context_id: 10, source_kind: 0 },
        );
        assert_eq!(clear_pending_for_context(&mut pending, 99), None);
        assert_eq!(
            pending.get(&10),
            Some(&PendingNextInput { link: 5, positions: vec![0], n_rows: 1, context_id: 10, source_kind: 0 })
        );

        // No pending → None.
        let mut pending: HashMap<u32, PendingNextInput> = HashMap::new();
        assert_eq!(clear_pending_for_context(&mut pending, 1), None);
    }

    /// Drafts-channel §8.3: a `PrevDrafts` producer (`pipeline_source_kind=1`, `[k+1]`
    /// window positions `0..=k`) → the consumer injects retained-buffer row `i` → the
    /// `i`-th window slot — STATIC `src_rows=[0..=k]` (the pre-composed window),
    /// NOT the single-token `n_rows-1`.
    #[test]
    fn drafts_channel_consumer_injects_window_src_rows() {
        let mut pending: HashMap<u32, PendingNextInput> = HashMap::new();
        const CTX: u32 = 3;
        let k = 3u32;
        let k1 = (k + 1) as usize; // the [k+1] window

        // Producer: a [k+1]-row window forward declares the drafts carrier
        // (positions 0..=k) with pipeline_source_kind = PrevDrafts(1).
        let mut req_p = ForwardRequest::default();
        req_p.token_ids = vec![0; k1];
        req_p.set_pipeline_source_kind(PREV_DRAFTS);
        let positions: Vec<u32> = (0..=k).collect();
        let deps_p = apply_next_input_carrier(&mut pending, &mut req_p, &positions, CTX);
        let link_p = deps_p.produced.expect("producer link");
        assert_eq!(pending.get(&CTX).map(|p| p.source_kind), Some(PREV_DRAFTS));

        // Consumer: [k+1] placeholders; the carrier injects the retained [seed,drafts]
        // window at STATIC src_rows [0..=k] → dest_slots [0..=k].
        let mut req_c = ForwardRequest::default();
        req_c.token_ids = vec![0; k1];
        apply_next_input_carrier(&mut pending, &mut req_c, &positions, CTX);
        assert_eq!(req_c.next_input_producer_links, vec![link_p; k1]);
        assert_eq!(req_c.next_input_src_rows, (0..=k).collect::<Vec<u32>>()); // window rows, NOT n_rows-1
        assert_eq!(req_c.next_input_dest_slots, (0..=k).collect::<Vec<u32>>());
        assert_eq!(req_c.next_input_free_links, vec![link_p]);
    }

    /// Depth-k rollback free-all (spec §4): record a run-ahead producer chain, then
    /// the fresh-generate free-all reclaims EVERY link on the context (incl. an
    /// over-shoot-orphaned last-committed carry a dropped fire left), context-isolated.
    #[test]
    fn produced_links_free_all_reclaims_context_chain() {
        let mut produced: HashMap<u32, Vec<u32>> = HashMap::new();
        const CTX: u32 = 7;
        const OTHER: u32 = 9;

        // A run-ahead producer chain on CTX (prime + over-shot fires) + one on OTHER.
        record_produced_link(&mut produced, CTX, 101);
        record_produced_link(&mut produced, CTX, 102);
        record_produced_link(&mut produced, CTX, 103);
        record_produced_link(&mut produced, OTHER, 201);
        // The no-KV pass (context 0) is never a carrier producer → not tracked.
        record_produced_link(&mut produced, 0, 999);
        assert!(!produced.contains_key(&0));

        // Free-all for CTX takes EVERY link (order preserved) — the fresh-generate
        // emits these frees, reclaiming any over-shoot-orphaned carry.
        assert_eq!(take_produced_links_for_context(&mut produced, CTX), vec![101, 102, 103]);
        // Idempotent: a second free-all is empty (entry removed) — re-free is a no-op.
        assert!(take_produced_links_for_context(&mut produced, CTX).is_empty());
        // Context isolation (Bug#2): OTHER's links are untouched by CTX's free-all.
        assert_eq!(take_produced_links_for_context(&mut produced, OTHER), vec![201]);
        // A never-recorded context yields nothing.
        assert!(take_produced_links_for_context(&mut produced, 123).is_empty());
    }

    /// #17 orphans-only: as each link's single consumer frees it, `remove_produced_link`
    /// drops it, so `produced` converges to exactly the ORPHAN set — the fresh-generate
    /// free-all then emits ONLY genuine orphans, never the already-freed chain history.
    #[test]
    fn remove_produced_link_leaves_only_orphans() {
        let mut produced: HashMap<u32, Vec<u32>> = HashMap::new();
        const CTX: u32 = 7;

        // A 4-link run-ahead chain on CTX: links 101,102,103 each get a consumer that
        // frees them; 104 is the terminal producer whose consumer never runs (orphan).
        for l in [101, 102, 103, 104] {
            record_produced_link(&mut produced, CTX, l);
        }
        // Consumers of 101,102,103 run (each frees its injected link → removed here).
        remove_produced_link(&mut produced, CTX, 101);
        remove_produced_link(&mut produced, CTX, 102);
        remove_produced_link(&mut produced, CTX, 103);
        // Only the orphaned terminal carry (104) remains → the free-all frees just it,
        // NOT the already-freed 101/102/103 (the "too broad" re-free is eliminated).
        assert_eq!(take_produced_links_for_context(&mut produced, CTX), vec![104]);

        // Order-independence + empty-cleanup: remove the middle, then the rest.
        record_produced_link(&mut produced, CTX, 201);
        record_produced_link(&mut produced, CTX, 202);
        record_produced_link(&mut produced, CTX, 203);
        remove_produced_link(&mut produced, CTX, 202);
        assert_eq!(take_produced_links_for_context(&mut produced, CTX), vec![201, 203]);
        // Removing the last tracked link drops the context entry (mirrors `take`).
        record_produced_link(&mut produced, CTX, 301);
        remove_produced_link(&mut produced, CTX, 301);
        assert!(!produced.contains_key(&CTX));
        // Removing an untracked link (already consumed, or wrong ctx) is a no-op.
        remove_produced_link(&mut produced, CTX, 999);
        remove_produced_link(&mut produced, 42, 999);
    }

    // ── #23 overlap abort-isolation write-log ───────────────────────────────

    fn dep(produced: Option<u32>, consumed: Option<u32>) -> NextInputDeps {
        NextInputDeps { produced, consumed }
    }

    #[test]
    fn overlap_log_happy_path_commits() {
        // Producer t (link 1) commits, then consumer t+1 (consumed 1) commits.
        let mut log = OverlapLinkLog::default();
        assert!(log.finalize(true, dep(Some(1), None))); // producer succeeds → link 1 Committed
        assert!(log.finalize(true, dep(None, Some(1)))); // consumer: producer committed → success
        assert!(log.is_empty()); // consumer removed link 1; nothing lingers
    }

    #[test]
    fn overlap_log_cascade_aborts_consumer() {
        // Producer t aborts → consumer t+1 (driver-success!) cascade-aborts.
        let mut log = OverlapLinkLog::default();
        assert!(!log.finalize(false, dep(Some(1), None))); // producer aborts → link 1 Aborted
        assert!(
            !log.finalize(true, dep(None, Some(1))),
            "consumer must cascade-abort despite driver success — poisoned input"
        );
    }

    #[test]
    fn overlap_log_fail_closed_on_unresolved_link() {
        // LOCK 1: a consumed link never recorded (ordering violation) ⇒ treated as
        // Aborted (fail-closed), NOT fail-open.
        let mut log = OverlapLinkLog::default();
        assert!(
            !log.finalize(true, dep(None, Some(42))),
            "unresolved producer link must fail CLOSED (cascade-abort), never commit"
        );
    }

    #[test]
    fn overlap_log_chained_cascade_propagates_whole_run() {
        // The realistic multi-step overlap: t aborts → t+1 cascade-aborts → its
        // produced link is Aborted → t+2 cascade-aborts → … the poison propagates
        // the WHOLE dependent chain (delta's ≥3-in-flight gate), no hop stops short.
        let mut log = OverlapLinkLog::default();
        assert!(!log.finalize(false, dep(Some(1), None))); // t aborts (link 1)
        // t+1: consumes 1 (poisoned) AND produces 2 → cascades, publishes 2=Aborted.
        assert!(!log.finalize(true, dep(Some(2), Some(1))));
        // t+2: consumes 2 (poisoned) AND produces 3 → cascades, publishes 3=Aborted.
        assert!(!log.finalize(true, dep(Some(3), Some(2))));
        // t+3: consumes 3 (poisoned) → still cascades. The whole chain is poisoned.
        assert!(!log.finalize(true, dep(None, Some(3))));
        assert!(log.is_empty());
    }

    #[test]
    fn overlap_log_independent_passes_commit() {
        // A pass that is neither producer nor consumer is unaffected.
        let mut log = OverlapLinkLog::default();
        assert!(log.finalize(true, dep(None, None)));
        // A driver-failed pass with no overlap deps still reports its own failure.
        assert!(!log.finalize(false, dep(None, None)));
    }

    #[test]
    fn overlap_log_clear_drops_terminal_entry() {
        // A terminal producer (no consumer) lingers until `clear()` (fresh generate).
        let mut log = OverlapLinkLog::default();
        assert!(log.finalize(true, dep(Some(9), None))); // terminal producer, link 9
        assert!(!log.is_empty());
        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn apply_carrier_reports_produced_and_consumed_links() {
        // The deps the write-log consumes: producer role → `produced`, matching
        // consumer → `consumed`, context-mismatch → NOT consumed (no dependency).
        let mut pending: HashMap<u32, PendingNextInput> = HashMap::new();
        const CTX: u32 = 7;

        // Pass t: pure producer (no prior carry) → produced=Some(link_t), consumed=None.
        let mut req_t = ForwardRequest::default();
        req_t.token_ids = vec![1, 2];
        let d_t = apply_next_input_carrier(&mut pending, &mut req_t, &[5], CTX);
        let link_t = req_t.pipeline_source_link;
        assert_eq!(d_t, dep(Some(link_t), None));

        // Pass t+1: consumes t's carry (same ctx) AND produces → consumed=Some(link_t),
        // produced=Some(link_t1).
        let mut req_t1 = ForwardRequest::default();
        req_t1.token_ids = vec![3];
        let d_t1 = apply_next_input_carrier(&mut pending, &mut req_t1, &[0], CTX);
        let link_t1 = req_t1.pipeline_source_link;
        assert_eq!(d_t1, dep(Some(link_t1), Some(link_t)));

        // Pass on a DIFFERENT context: reads its OWN (empty) carry → no inject, no
        // produce → NOT a dependency (consumed stays None). CTX's carry is untouched
        // (per-context isolation).
        let mut req_x = ForwardRequest::default();
        req_x.token_ids = vec![9];
        let d_x = apply_next_input_carrier(&mut pending, &mut req_x, &[], 999);
        assert_eq!(d_x, dep(None, None));
    }
}
