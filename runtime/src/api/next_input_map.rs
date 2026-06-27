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
) {
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
        } else {
            // Stale carry from a DIFFERENT context: the producer's intended
            // (immediate-next, same-context) consumer never fired — this terminal
            // producer's carry would otherwise leak into a new context's prefill.
            // Don't inject; free the retained buffer (global, keyed by link) so it
            // doesn't leak. `pending.take()` already cleared the carry.
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
        *pending = Some(PendingNextInput {
            link,
            positions: positions.to_vec(),
            n_rows,
            context_id,
        });
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
}
