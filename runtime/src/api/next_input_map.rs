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
}

/// Populate `req`'s next-input carrier for one forward pass.
///
/// Call once per pass at execute-time eager-submit, **after** `input-tokens` is
/// staged (it reads `req.token_ids.len()` for the producer row count) and
/// **before** submit. `pending`/`counter` are the per-context carry state owned by
/// the caller (e.g. `InstanceState`). `positions` is this pass's declared
/// `next-inputs(positions)` (empty ⇒ not a producer). `context_id` identifies this
/// pass's context (the KV working-set rep) so the carryover stays scoped to
/// consecutive passes on the same context.
pub fn apply_next_input_carrier(
    pending: &mut Option<PendingNextInput>,
    counter: &mut u32,
    req: &mut ForwardRequest,
    positions: &[u32],
    context_id: u32,
) -> NextInputDeps {
    let mut deps = NextInputDeps::default();
    // ── CONSUMER role ───────────────────────────────────────────────
    // Inject the prior producer's retained sample into this pass's declared dest
    // slots. `src_row = n_rows - 1` is the producer's last forward row (the decode
    // position). One consumer per link ⇒ free it here (host refcount = 1).
    if let Some(p) = pending.take() {
        if p.context_id == context_id {
            let src_row = p.n_rows.saturating_sub(1);
            // TODO(multi-seq): single `src_row` for all positions is correct for the
            // single-sequence one-ahead path (one position, the producer's last row).
            // Batched multi-sequence run-ahead needs per-position src_rows (each
            // sequence's own last sampled row) — a follow-up generalization.
            for &pos in &p.positions {
                req.push_next_input_link(p.link, src_row, pos);
            }
            req.push_next_input_free_link(p.link);
            // #23: a real cross-pass dependency — record the producer link this
            // pass injected from so its finalize can cascade-abort if the producer
            // aborted (and so the free above rides the drain-gated deferred-free).
            deps.consumed = Some(p.link);
        } else {
            // Stale carry from a DIFFERENT context: the producer's intended
            // (immediate-next, same-context) consumer never fired — this terminal
            // producer's carry would otherwise leak into a new context's prefill.
            // Don't inject; free the retained buffer (global, keyed by link) so it
            // doesn't leak. `pending.take()` already cleared the carry. NOT a real
            // dependency (no inject) ⇒ `deps.consumed` stays `None`.
            req.push_next_input_free_link(p.link);
        }
    }

    // ── PRODUCER role ───────────────────────────────────────────────
    // If this pass declared `next-inputs`, assign a fresh monotonic link
    // (`0` = the not-a-source sentinel), mark this pass as the retain source, and
    // stash the carry for the next pass to consume.
    if !positions.is_empty() {
        *counter += 1;
        let link = *counter;
        req.set_pipeline_source_link(link);
        let n_rows = req.token_ids.len() as u32;
        deps.produced = Some(link);
        *pending = Some(PendingNextInput {
            link,
            positions: positions.to_vec(),
            n_rows,
            context_id,
        });
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
    pending: &mut Option<PendingNextInput>,
    context_id: u32,
) -> Option<u32> {
    if pending.as_ref().is_some_and(|p| p.context_id == context_id) {
        pending.take().map(|p| p.link)
    } else {
        None
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
        let mut pending: Option<PendingNextInput> = None;
        let mut counter: u32 = 0;
        const CTX: u32 = 7;

        // Pass t (prime): prefill 2 prompt tokens, sample at the last row, carry to
        // the next pass's input slot 5.
        let mut req_p = ForwardRequest::default();
        req_p.token_ids = vec![100, 200];
        apply_next_input_carrier(&mut pending, &mut counter, &mut req_p, &[5], CTX);

        // Producer marked as source L=1; no consumer links (pending was empty).
        assert_eq!(req_p.pipeline_source_link, 1);
        assert!(req_p.next_input_producer_links.is_empty());
        assert_eq!(
            pending,
            Some(PendingNextInput { link: 1, positions: vec![5], n_rows: 2, context_id: CTX })
        );

        // Pass t+1 (consumer + next producer): 1 placeholder token, carry to slot 6.
        let mut req_c = ForwardRequest::default();
        req_c.token_ids = vec![0];
        apply_next_input_carrier(&mut pending, &mut counter, &mut req_c, &[6], CTX);

        // Consumer of link 1: src_row = n_rows-1 = 1 (the prime's last row),
        // dest 5, freed (single consumer).
        assert_eq!(req_c.next_input_producer_links, vec![1]);
        assert_eq!(req_c.next_input_src_rows, vec![1]);
        assert_eq!(req_c.next_input_dest_slots, vec![5]);
        assert_eq!(req_c.next_input_free_links, vec![1]);
        // And itself a producer: L=2, single-row ⇒ next consumer's src_row=0.
        assert_eq!(req_c.pipeline_source_link, 2);
        assert_eq!(
            pending,
            Some(PendingNextInput { link: 2, positions: vec![6], n_rows: 1, context_id: CTX })
        );
    }

    /// A terminal producer's dangling carry must NOT leak into the next context: a
    /// pass on a different `context_id` drops it (freeing the link) instead of
    /// injecting it. Regression for the cross-context `pi.tokens[0]` corruption.
    #[test]
    fn dangling_carry_does_not_leak_across_contexts() {
        let mut pending: Option<PendingNextInput> = None;
        let mut counter: u32 = 0;

        // Context A's terminal producer declares a carry that never gets consumed
        // on A (the loop ended).
        let mut req_a = ForwardRequest::default();
        req_a.token_ids = vec![0];
        apply_next_input_carrier(&mut pending, &mut counter, &mut req_a, &[0], 10);
        assert_eq!(req_a.pipeline_source_link, 1);
        assert!(pending.is_some());

        // Context B's first (prefill) pass: a DIFFERENT context_id. It must NOT
        // inject A's carry (no producer links / dest slots), only free the stale
        // link, and the pending must be cleared.
        let mut req_b = ForwardRequest::default();
        req_b.token_ids = vec![100, 200];
        apply_next_input_carrier(&mut pending, &mut counter, &mut req_b, &[], 20);
        assert!(req_b.next_input_producer_links.is_empty());
        assert!(req_b.next_input_dest_slots.is_empty());
        assert_eq!(req_b.next_input_free_links, vec![1]);
        assert_eq!(pending, None);
    }

    /// A pass with no `next-inputs` is neither source nor (absent a pending carry)
    /// consumer — the carrier stays empty.
    #[test]
    fn non_producer_pass_is_inert() {
        let mut pending: Option<PendingNextInput> = None;
        let mut counter: u32 = 0;
        let mut req = ForwardRequest::default();
        req.token_ids = vec![42];
        apply_next_input_carrier(&mut pending, &mut counter, &mut req, &[], 1);
        assert_eq!(req.pipeline_source_link, 0);
        assert!(req.next_input_producer_links.is_empty());
        assert_eq!(pending, None);
        assert_eq!(counter, 0);
    }

    /// `fresh-generate` (#26) clears THIS context's dangling carry and returns its
    /// link to free; a different context's pending is left for the cross-context
    /// drop; no pending ⇒ nothing to free.
    #[test]
    fn clear_pending_for_context_drops_same_context_only() {
        // Same-context dangling carry → cleared, link returned for the free.
        let mut pending = Some(PendingNextInput {
            link: 3,
            positions: vec![0],
            n_rows: 1,
            context_id: 42,
        });
        assert_eq!(clear_pending_for_context(&mut pending, 42), Some(3));
        assert_eq!(pending, None);

        // Different-context pending → untouched (apply_next_input_carrier's
        // mismatch branch handles it), nothing returned.
        let mut pending = Some(PendingNextInput {
            link: 5,
            positions: vec![0],
            n_rows: 1,
            context_id: 10,
        });
        assert_eq!(clear_pending_for_context(&mut pending, 99), None);
        assert_eq!(
            pending,
            Some(PendingNextInput { link: 5, positions: vec![0], n_rows: 1, context_id: 10 })
        );

        // No pending → None.
        let mut pending: Option<PendingNextInput> = None;
        assert_eq!(clear_pending_for_context(&mut pending, 1), None);
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
        let mut pending: Option<PendingNextInput> = None;
        let mut counter: u32 = 0;
        const CTX: u32 = 7;

        // Pass t: pure producer (no prior carry) → produced=Some(1), consumed=None.
        let mut req_t = ForwardRequest::default();
        req_t.token_ids = vec![1, 2];
        let d_t = apply_next_input_carrier(&mut pending, &mut counter, &mut req_t, &[5], CTX);
        assert_eq!(d_t, dep(Some(1), None));

        // Pass t+1: consumes t's carry (same ctx) AND produces → consumed=Some(1),
        // produced=Some(2).
        let mut req_t1 = ForwardRequest::default();
        req_t1.token_ids = vec![3];
        let d_t1 = apply_next_input_carrier(&mut pending, &mut counter, &mut req_t1, &[0], CTX);
        assert_eq!(d_t1, dep(Some(2), Some(1)));

        // Pass on a DIFFERENT context: the carry is dropped (freed, not injected) →
        // NOT a dependency (consumed stays None).
        let mut req_x = ForwardRequest::default();
        req_x.token_ids = vec![9];
        let d_x = apply_next_input_carrier(&mut pending, &mut counter, &mut req_x, &[], 999);
        assert_eq!(d_x, dep(None, None));
    }
}
