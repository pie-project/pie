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

