//! Fire RS (recurrent-state) preparation over the typed `RsStore`
//! (kv_refact.md). The GDN / linear-attention forward writes the advanced
//! folded state INTO the folded slot directly (the driver's in-forward
//! `commit_len` path): [`prepare_many`] classifies every folded-slot target
//! (fresh+reset / CoW-after-fork / in-place), PUBLISHES the resulting mapping
//! under the same store lock — in guest submission order, before the launch
//! reaches the scheduler, exactly as `fire::kv::prepare` does — and returns
//! the driver lowering (`rs_slot_ids`, `rs_slot_flags`, pre-launch d2d copy)
//! plus an [`RsTxn`] receipt the fire holds until [`settle`].
//!
//! Publishing at prepare is what lets RS fires run ahead: a successor
//! prepared before its predecessor completes classifies against a mapping
//! that already carries the predecessor's decision, so it can neither RESET
//! a slot twice nor re-CoW an already-privatized one. The advanced state's
//! CONTENTS are ordered by the pipeline stream, which the pre-launch CoW copy
//! already relies on.
//!
//! The allow below covers exactly the singular test-exercised convenience
//! wrapper ([`prepare`]) around the `_many` production entries — kept rather
//! than deleted, allowed rather than silently masked.
#![allow(dead_code)]

use crate::store::rs::write::RsBufferIntent;
use crate::store::rs::write::{RsPreparedWrite, RsPublished};
use crate::store::rs::{RsStore, RsWorkingSetId};

/// The published RS write for one in-flight PTIR fire, held across
/// `submit_async` until [`settle`].
#[derive(Debug)]
pub struct RsTxn {
    published: RsPublished,
}

impl RsTxn {
    /// Submission sequence of the newest row this fire published.
    pub fn seq(&self) -> u64 {
        self.published.seq()
    }
}

/// Validate the recurrent-state arity against the resolved forward rows.
pub fn validate_count(
    rs_count: usize,
    qo_indptr: &[u32],
    has_recurrent_state: bool,
) -> Result<usize, String> {
    if !has_recurrent_state {
        if rs_count == 0 {
            return Ok(qo_indptr.len().saturating_sub(1));
        }
        return Err(format!(
            "pure-attention model bound {rs_count} rs-working-set(s); expected 0"
        ));
    }
    let request_count = qo_indptr
        .len()
        .checked_sub(1)
        .ok_or_else(|| "resolved qo_indptr is empty".to_string())?;
    if rs_count != request_count {
        return Err(format!(
            "resolved forward has {request_count} request row(s), but recurrent-state model bound \
             {rs_count} rs-working-set(s); expected {request_count}",
        ));
    }
    Ok(request_count)
}

/// What a pass does with the recurrent state of its bound working sets, with
/// the per-row token counts the lowering needs resolved from the fire's
/// geometry. Derived from WIT `rs-geometry.fold-len` at prepare time by
/// `super::rs_plan_for` — a position for the folded boundary, classified into
/// the three shapes the driver implements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RsPlan {
    /// Advance the folded state in-forward over every row.
    Fold,
    /// Scatter each row's pre-recurrence activations into buffered slots
    /// covering `[start_tokens[r], start_tokens[r] + row_tokens[r])`, and
    /// advance the folded boundary through `fold_tokens[r]` of the resulting
    /// buffer.
    ///
    /// `start_tokens[r]` is row `r`'s existing buffer occupancy, so a non-zero
    /// entry means the fire appends onto a NON-EMPTY buffer and its recurrence
    /// must read what is already there. That is the buffer read path. It is
    /// per-row rather than per-pass because occupancy is a property of the
    /// working set, not of the fire: one request commits while another
    /// speculates, and they arrive at different offsets in the same batch.
    ///
    /// `fold_tokens[r]` is where the folded boundary lands, counted in the
    /// row's EXTENDED layout `[b | t]` — which is exactly the row's buffer
    /// token space, so it is also just "fold this many buffered tokens". Zero
    /// leaves the folded state untouched (a pure append). Any other value
    /// makes the pass a fold as well as a write: the driver runs the whole
    /// extended row but snapshots the recurrent state at token
    /// `fold_tokens[r]` via `commit_len`, so the outputs cover every new token
    /// while the boundary lands wherever the guest asked — including strictly
    /// INSIDE this fire's own new tokens.
    ///
    /// `in_forward[r]` marks a row that owns NO buffer in this pass: it folds
    /// its own new tokens straight into the folded state, exactly as
    /// `RsPlan::Fold` does, while riding along in a fire whose other rows
    /// buffer. Such a row carries `start_tokens = row_tokens = fold_tokens =
    /// 0` — it has no buffered span to measure a fold against — and the driver
    /// recognises it by its empty buffer CSR span.
    Buffer {
        start_tokens: Vec<u32>,
        row_tokens: Vec<u32>,
        fold_tokens: Vec<u32>,
        in_forward: Vec<bool>,
    },
    /// Replay `tokens[r]` buffered tokens of row `r` into its folded state.
    ///
    /// When `fold_len_is_device` the host never learned the count: `tokens[r]`
    /// is its own UPPER BOUND (the whole live buffer), the real value comes
    /// from the `rs_fold_len` descriptor port, and the driver clamps one to
    /// the other. The host's boundary then becomes indeterminate until the
    /// guest frees the buffer.
    FoldBuffered {
        tokens: Vec<u32>,
        fold_len_is_device: bool,
    },
}

impl RsPlan {
    /// `(write_state, fold_tokens, buffer_tokens, buffer_intent)` for row
    /// `index` — exactly the arguments `RsStore::prepare` classifies against.
    fn row(&self, index: usize) -> (bool, Option<u32>, Option<(u32, u32)>, RsBufferIntent) {
        match self {
            RsPlan::Fold => (true, None, None, RsBufferIntent::Write),
            RsPlan::Buffer {
                start_tokens,
                row_tokens,
                fold_tokens,
                in_forward,
            } => {
                if in_forward.get(index).copied().unwrap_or(false) {
                    return (true, None, None, RsBufferIntent::Write);
                }
                let n = fold_tokens.get(index).copied().unwrap_or(0);
                (
                    n > 0,
                    (n > 0).then_some(n),
                    Some((
                        start_tokens.get(index).copied().unwrap_or(0),
                        row_tokens.get(index).copied().unwrap_or(0),
                    )),
                    RsBufferIntent::Write,
                )
            }
            // A fold gathers the buffered PREFIX: the driver walks the CSR
            // from slab zero, so the span always starts at buffer token 0.
            RsPlan::FoldBuffered { tokens, .. } => {
                let n = tokens.get(index).copied().unwrap_or(0);
                (true, Some(n), Some((0, n)), RsBufferIntent::Replay)
            }
        }
    }
}

/// Phase-A demand for [`prepare_many`] over these working sets: how many
/// slots the prepare would allocate — the folded target plus any buffered
/// page it must materialize or copy-on-write — with no allocation or open
/// transaction. The acquisition seam sizes its RS ask from this.
pub fn demand(
    store: &RsStore,
    working_sets: &[RsWorkingSetId],
    plan: &RsPlan,
) -> Result<usize, String> {
    let mut total = 0;
    for (index, &ws) in working_sets.iter().enumerate() {
        let (write_state, _, buffer_tokens, _) = plan.row(index);
        total += store
            .write_demand(ws, write_state, buffer_tokens)
            .map_err(|e| e.to_string())?;
    }
    Ok(total)
}

/// Driver lowering for one fire's published recurrent-state work.
#[derive(Debug)]
pub struct PreparedRs {
    /// Folded slot per request row. Always present: buffered execution needs
    /// it too, to validate the row and to read the state it does not write.
    pub slot_ids: Vec<u32>,
    pub slot_flags: Vec<u8>,
    /// Pre-launch device copies `(src, dst)` for copy-on-write targets.
    pub copies: (Vec<u32>, Vec<u32>),
    /// Tokens to replay per row; empty unless this is a fold.
    pub fold_lens: Vec<u32>,
    /// Buffered slab ids, flattened, with the per-row CSR boundaries the
    /// driver walks page-major. Empty unless the pass touches the buffer.
    ///
    /// These are the slabs the fire WRITES: `prepare` may materialize or
    /// privatize them, so they are targets, not sources.
    pub buffer_slot_ids: Vec<u32>,
    pub buffer_slot_indptr: Vec<u32>,
    /// The slabs the fire READS, and how many tokens of them, per row.
    ///
    /// Separate from the write CSR because reading and writing the buffer are
    /// different intents on the same pages. A write may allocate; a read must
    /// not, and a read of a page that was merely reserved would gather
    /// uninitialized activations straight into the recurrence. The WIT draws
    /// the same line, with `readable-buffer` and `writable-buffer` as distinct
    /// page spans.
    ///
    /// Non-empty only when a row appends onto a NON-EMPTY buffer: its
    /// recurrence has to start from `folded ⊕ replay(buffer)`, so the driver
    /// gathers `buffer_read_lens[r]` tokens ahead of the row's own tokens.
    pub buffer_read_slot_ids: Vec<u32>,
    pub buffer_read_indptr: Vec<u32>,
    pub buffer_read_lens: Vec<u32>,
    /// Where each row's logical buffer token 0 physically sits, in tokens from
    /// the start of its first page.
    ///
    /// A fold absorbs tokens off the front of the buffer but can only release
    /// WHOLE covered pages, and `fold_granularity` is 1 in production while a
    /// page is the KV page size — so a fold routinely lands mid-page and the
    /// survivors keep their offsets. Every buffer span the driver walks is
    /// therefore `head + logical`. Emitted whenever the pass touches the
    /// buffer, including a pure write, because a buffer can be logically empty
    /// and still be physically offset.
    pub buffer_heads: Vec<u32>,
    /// WorkingSet-relative buffer page -> physical slot, concatenated over
    /// request rows with `translation_indptr` as the per-row CSR.
    ///
    /// Per REQUEST, not per pass: a pass binds one RS working set per request,
    /// so unlike `kv_translation` there is no single table for the fire. Built
    /// after the prepare, because materializing or copying-on-write a buffer
    /// page changes the mapping it publishes.
    pub translation: Vec<u32>,
    pub translation_indptr: Vec<u32>,
    pub txn: Option<RsTxn>,
}

impl PreparedRs {
    /// The no-recurrent-state lowering: every field empty.
    pub fn empty() -> Self {
        empty_prepared()
    }

    /// Thread this lowering into a launch plan.
    pub fn apply_to(&self, request: &mut crate::driver::LaunchPlan) {
        request.rs_slot_ids = self.slot_ids.clone();
        request.rs_slot_flags = self.slot_flags.clone();
        request.rs_fold_lens = self.fold_lens.clone();
        request.rs_buffer_slot_ids = self.buffer_slot_ids.clone();
        request.rs_buffer_slot_indptr = self.buffer_slot_indptr.clone();
        request.rs_buffer_read_slot_ids = self.buffer_read_slot_ids.clone();
        request.rs_buffer_read_indptr = self.buffer_read_indptr.clone();
        request.rs_buffer_read_lens = self.buffer_read_lens.clone();
        request.rs_buffer_heads = self.buffer_heads.clone();
        request.rs_translation = self.translation.clone();
        request.rs_translation_indptr = self.translation_indptr.clone();
    }
}

/// Prepare and publish this fire's recurrent-state work. Returns the driver
/// lowering: thread the ids and flags into the launch in request order, issue
/// one aggregated state-copy command before the launch when non-empty, hold
/// `txn` across the fire, then [`settle`].
pub fn prepare_many(
    store: &mut RsStore,
    working_sets: &[RsWorkingSetId],
    plan: &RsPlan,
) -> Result<PreparedRs, String> {
    prepare_many_impl(store, working_sets, plan, None)
}

/// [`prepare_many`] from caller-owned reserved slots (the acquisition
/// grant), consuming exactly the required prefix of `granted`.
pub fn prepare_many_reserved(
    store: &mut RsStore,
    working_sets: &[RsWorkingSetId],
    plan: &RsPlan,
    granted: &mut Vec<crate::store::rs::RsSlotId>,
) -> Result<PreparedRs, String> {
    prepare_many_impl(store, working_sets, plan, Some(granted))
}

fn empty_prepared() -> PreparedRs {
    PreparedRs {
        slot_ids: Vec::new(),
        slot_flags: Vec::new(),
        copies: (Vec::new(), Vec::new()),
        fold_lens: Vec::new(),
        buffer_slot_ids: Vec::new(),
        buffer_read_slot_ids: Vec::new(),
        buffer_read_indptr: Vec::new(),
        buffer_read_lens: Vec::new(),
        buffer_heads: Vec::new(),
        buffer_slot_indptr: Vec::new(),
        translation: Vec::new(),
        translation_indptr: Vec::new(),
        txn: None,
    }
}

fn prepare_many_impl(
    store: &mut RsStore,
    working_sets: &[RsWorkingSetId],
    plan: &RsPlan,
    mut granted: Option<&mut Vec<crate::store::rs::RsSlotId>>,
) -> Result<PreparedRs, String> {
    for (index, ws) in working_sets.iter().enumerate() {
        if working_sets[..index].contains(ws) {
            return Err(format!(
                "rs-working-set at request row {index} aliases an earlier row"
            ));
        }
    }
    if working_sets.is_empty() {
        return Ok(empty_prepared());
    }

    let buffered = !matches!(plan, RsPlan::Fold);
    let mut out = empty_prepared();
    if buffered {
        out.buffer_slot_indptr.push(0);
    }
    let mut prepared_rows: Vec<RsPreparedWrite> = Vec::with_capacity(working_sets.len());

    for (index, &ws) in working_sets.iter().enumerate() {
        let (write_state, fold_tokens, buffer_tokens, buffer_intent) = plan.row(index);

        // Buffered execution reads the folded state it does not write, and
        // a fold advances it — both need one already to exist. The driver
        // skips its per-slot reset in either mode, so a cold working set
        // would consume undefined state. Reject it here, where the guest
        // still gets a message it can act on.
        if buffered
            && store
                .folded_slot(ws)
                .map_err(|error| error.to_string())?
                .is_none()
        {
            store.cancel_batch(prepared_rows);
            return Err(format!(
                "rs-working-set at request row {index} has no folded state yet: run a folding \
                 pass (the prefill) before buffering or folding buffered tokens"
            ));
        }

        let prepared = match granted.as_deref_mut() {
            Some(granted) => store.prepare_reserved(
                ws,
                write_state,
                fold_tokens,
                buffer_tokens,
                buffer_intent,
                granted,
            ),
            None => {
                store.prepare_general(ws, write_state, fold_tokens, buffer_tokens, buffer_intent)
            }
        };
        let prepared = match prepared {
            Ok(mut prepared) => {
                if matches!(
                    plan,
                    RsPlan::FoldBuffered {
                        fold_len_is_device: true,
                        ..
                    }
                ) {
                    // The fold length that reached `prepare` is the host's own
                    // upper bound, good enough to size the allocation and to
                    // shape the replay CSR. Publishing it as if it were the
                    // truth would move the boundary too far, so say so here
                    // and let `publish_batch` hold the boundary instead.
                    prepared.mark_fold_len_device();
                }
                prepared
            }
            Err(error) => {
                store.cancel_batch(prepared_rows);
                return Err(error.to_string());
            }
        };

        // Folded slot: the write target when this pass folds, otherwise the
        // committed slot the buffered forward reads. `RS_FLAG_FOLD` is what
        // selects the driver's replay mode, and it must be uniform across
        // the batch — `requires_solo_submission` keeps such a fire alone.
        match prepared.state() {
            Some(state) => {
                out.slot_ids.push(state.slot.0);
                let mut flags = if state.reset {
                    crate::driver::RS_FLAG_RESET
                } else {
                    0
                };
                if state.fold_tokens.is_some() {
                    flags |= crate::driver::RS_FLAG_FOLD;
                }
                // The wire `fold_lens[r]` this row is about to carry is the
                // host's upper bound, not the fold length. The driver replaces
                // it with the resolved `rs_fold_len` port CLAMPED to the bound,
                // which is what keeps a device-named count inside the buffer
                // that can supply it.
                if prepared.fold_len_is_bound() {
                    flags |= crate::driver::RS_FLAG_FOLD_LEN_DEVICE;
                }
                // Orthogonal to FOLD. A pass that both writes the buffer and
                // folds a prefix of the result runs the extended layout and
                // snapshots the state at `fold_lens[r]`; a pure commit's rows
                // ARE the replay. The driver cannot tell them apart from the
                // fold flag alone.
                if buffer_intent == RsBufferIntent::Write
                    && buffer_tokens.is_some_and(|(_, len)| len > 0)
                {
                    flags |= crate::driver::RS_FLAG_BUFFER_WRITE;
                }
                out.slot_flags.push(flags);
                if let Some(src) = state.copy_from {
                    out.copies.0.push(src.0);
                    out.copies.1.push(state.slot.0);
                }
                // ONE length per row, always: `wire.rs` concatenates these
                // across a composed batch, so a row that contributes a slot
                // id but no length would shift every later row's.
                out.fold_lens.push(state.fold_tokens.unwrap_or(0));
            }
            None => {
                let slot = store
                    .folded_slot(ws)
                    .map_err(|error| error.to_string())?
                    .expect("buffered rows are rejected without a folded state");
                out.slot_ids.push(slot.0);
                out.slot_flags.push(0);
                out.fold_lens.push(0);
            }
        }

        // Buffer CSR. The driver walks a row's slabs page-major from the
        // first listed one, so the order here IS the token order.
        for target in prepared.buffer_targets() {
            out.buffer_slot_ids.push(target.dst().0);
            if let crate::store::rs::write::RsBufferTarget::Cow { src, dst, .. } = *target {
                out.copies.0.push(src.0);
                out.copies.1.push(dst.0);
            }
        }
        if buffered {
            out.buffer_slot_indptr
                .push(out.buffer_slot_ids.len() as u32);
        }

        prepared_rows.push(prepared);
    }

    // Publish before returning: the successor fire's classification must see
    // this fire's decision without waiting for the device.
    //
    // The FOLD advance is deferred (see `RsStore::commit_folds`). Everything
    // below describes the buffer as THIS fire's rows are laid out — extended
    // row `j` is physical `buffer_head + j` — and that frame is the pre-fold
    // one. Advancing first would hand a fire whose rows straddle the boundary
    // a head, and a page list, from the other side of it.
    let (published, pending_folds) = store
        .publish_batch(prepared_rows)
        .map_err(|error| error.to_string())?;
    // Everything from here to `commit_folds` runs inside a closure so that a
    // failure cannot skip it. Once `publish_batch` returns, the store has
    // ALREADY committed this fire's mapping and taken an in-flight hold, and
    // two obligations outlive any error below: the deferred fold must be
    // applied (dropping it would leave the boundary where no one expects it)
    // and the hold must be settled (`retire_idle` gates ALL pool retirement
    // on `in_flight == 0`, so leaking one wedges slot recycling for the rest
    // of the process). `#[must_use]` does not help here -- the value is bound
    // to a local, and the lint only catches a discarded temporary.
    let read_side = (|out: &mut PreparedRs| -> Result<(), String> {
        // Built AFTER the publish, so it reflects the pages this fire just
        // materialized or privatized rather than the mapping the guest saw.
        out.translation_indptr.push(0);
        let page_tokens_of = |ws| -> Result<u32, String> {
            store
                .geometry(ws)
                .map(|g| g.buffer_page_tokens.max(1))
                .map_err(|error| error.to_string())
        };
        let mut any_read = false;
        out.buffer_read_indptr.push(0);
        for (index, &ws) in working_sets.iter().enumerate() {
            let row = store
                .buffer_translation(ws)
                .map_err(|error| error.to_string())?;

            let head = if buffered {
                store.buffer_head(ws).map_err(|error| error.to_string())?
            } else {
                0
            };
            if buffered {
                out.buffer_heads.push(head);
            }

            // The read prefix is the row's PRE-EXISTING occupancy, which is
            // exactly where its own tokens begin.
            let read_tokens = match plan.row(index).2 {
                Some((start, _)) => start,
                None => 0,
            };
            out.buffer_read_lens.push(read_tokens);
            if read_tokens > 0 {
                any_read = true;
                let page = page_tokens_of(ws)?;
                // Physical, not logical: the replay starts at `head`, which a
                // mid-page fold leaves non-zero.
                let first = (head / page) as usize;
                let last = ((head + read_tokens - 1) / page) as usize;
                for p in first..=last {
                    match row.get(p) {
                        Some(&slot) if slot != crate::store::rs::RS_TRANSLATION_UNMAPPED => {
                            out.buffer_read_slot_ids.push(slot);
                        }
                        // Reserved-but-unmaterialized, or off the end. Either way
                        // the recurrence would replay activations that were never
                        // written, which is silent corruption of the state rather
                        // than a visible failure -- so refuse.
                        _ => {
                            return Err(format!(
                                "request row {index} must replay {read_tokens} buffered \
                                 token(s), but buffer page {p} of its working set \
                                 is not materialized"
                            ));
                        }
                    }
                }
            }
            out.buffer_read_indptr
                .push(out.buffer_read_slot_ids.len() as u32);

            out.translation.extend_from_slice(&row);
            out.translation_indptr.push(out.translation.len() as u32);
        }
        // Keep the wire quiet for the overwhelmingly common empty-buffer fire: an
        // all-zero read side is the same statement as an absent one, and the
        // driver's shape checks are simpler when "no read" is literally no data.
        if !any_read {
            out.buffer_read_slot_ids.clear();
            out.buffer_read_indptr.clear();
            out.buffer_read_lens.clear();
        }
        Ok(())
    })(&mut out);

    // Every wire array is now built against the pre-fold buffer, so the
    // boundary can finally move. A row that folds through its own new tokens
    // sees its head advance here and NOT one array earlier.
    store.commit_folds(pending_folds);
    if let Err(error) = read_side {
        // The mapping is committed and cannot be taken back, but the hold on
        // pool retirement can and must be.
        store.settle(published);
        return Err(error);
    }
    out.txn = Some(RsTxn { published });
    Ok(out)
}

pub fn prepare(store: &mut RsStore, ws: RsWorkingSetId) -> Result<PreparedRs, String> {
    prepare_many(store, &[ws], &RsPlan::Fold)
}

/// Settle a fire's published RS write once it resolves, successfully or not.
///
/// There is no mapping rollback: a published RS mapping is fail-stop
/// pipeline-local state (`fire::kv::abandon` says the same for KV). A failed
/// fire leaves the recurrent state undefined, the pass poisons its readers,
/// and the pre-failure state survives only through a fork taken before it —
/// the rule `RsStore` already documents for a committed fold. Settling
/// releases the store's in-flight hold so recycled slots become allocatable.
pub fn settle(store: &mut RsStore, txn: Option<RsTxn>) {
    let Some(RsTxn { published }) = txn else {
        return;
    };
    store.settle(published);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::rs::RsGeometry;

    fn geom() -> RsGeometry {
        RsGeometry {
            state_size: 1024,
            buffer_page_tokens: 4,
            fold_granularity: 1,
        }
    }

    /// A working set with a materialized folded state, which every buffered
    /// or folding pass requires.
    fn warm(store: &mut RsStore) -> RsWorkingSetId {
        let ws = store.create_working_set(geom());
        let prepared = prepare(store, ws).unwrap();
        settle(store, prepared.txn);
        ws
    }

    #[test]
    fn first_fire_resets_then_continues_in_place() {
        let mut store = RsStore::new(4);
        let ws = store.create_working_set(geom());

        let out = prepare(&mut store, ws).unwrap();
        assert_eq!(out.slot_ids.len(), 1);
        assert_eq!(out.slot_flags, vec![crate::driver::RS_FLAG_RESET]);
        assert!(out.copies.0.is_empty());
        assert_eq!(out.fold_lens, vec![0], "one length per row, zero = no fold");
        assert!(out.buffer_slot_ids.is_empty() && out.buffer_slot_indptr.is_empty());
        settle(&mut store, out.txn);
        let slot = store.folded_slot(ws).unwrap().unwrap();

        let out = prepare(&mut store, ws).unwrap();
        assert_eq!(out.slot_ids, vec![slot.0]);
        assert_eq!(out.slot_flags, vec![0]);
        settle(&mut store, out.txn);
    }

    #[test]
    fn forked_fire_copies_the_folded_state() {
        let mut store = RsStore::new(4);
        let ws = warm(&mut store);
        let shared = store.folded_slot(ws).unwrap().unwrap();

        let forked = store.fork(ws).unwrap();
        let out = prepare(&mut store, forked).unwrap();
        assert_eq!(out.copies.0, vec![shared.0]);
        assert_eq!(out.copies.1, out.slot_ids);
        assert_eq!(out.slot_flags, vec![0]); // copied, not reset
        settle(&mut store, out.txn);
        assert_eq!(store.folded_slot(ws).unwrap(), Some(shared));
        assert_ne!(store.folded_slot(forked).unwrap(), Some(shared));
    }

    // ------------------------------------------------------------------
    // Run-ahead. Each successor prepares while its predecessor is still in
    // flight — the shape the deleted `drain_rs_predecessors` forbade.
    // ------------------------------------------------------------------

    #[test]
    fn unsettled_fresh_runahead_never_lowers_a_second_reset() {
        let mut store = RsStore::new(3);
        let ws = store.create_working_set(geom());

        let first = prepare(&mut store, ws).unwrap();
        assert_eq!(first.slot_flags, vec![crate::driver::RS_FLAG_RESET]);

        // No settle: the first fire is still on the device.
        let second = prepare(&mut store, ws).unwrap();
        assert_eq!(
            second.slot_ids, first.slot_ids,
            "second fire continues the published slot"
        );
        assert_eq!(second.slot_flags, vec![0], "must not RESET again");
        assert!(second.copies.0.is_empty());
        assert_eq!(store.available_slots(), 2, "one slot serves both fires");

        settle(&mut store, first.txn);
        settle(&mut store, second.txn);
    }

    #[test]
    fn unsettled_forked_runahead_cows_once_then_continues_the_child() {
        let mut store = RsStore::new(4);
        let parent = warm(&mut store);
        let shared = store.folded_slot(parent).unwrap().unwrap().0;
        let child = store.fork(parent).unwrap();

        let first = prepare(&mut store, child).unwrap();
        assert_eq!(first.copies.0, vec![shared]);

        let second = prepare(&mut store, child).unwrap();
        assert_eq!(
            second.slot_ids, first.slot_ids,
            "child continues its published CoW slot"
        );
        assert!(
            second.copies.0.is_empty(),
            "a run-ahead successor must not CoW from the stale parent again"
        );
        settle(&mut store, first.txn);
        settle(&mut store, second.txn);
    }

    #[test]
    fn a_three_slot_frame_lowers_one_slot_and_one_reset() {
        let mut store = RsStore::new(2);
        let ws = store.create_working_set(geom());

        let mut txns = Vec::new();
        let mut lowered = Vec::new();
        for slot in 0..3 {
            let out = prepare(&mut store, ws).unwrap();
            assert!(
                out.copies.0.is_empty() && out.copies.1.is_empty(),
                "slot {slot} must not need a pre-launch copy"
            );
            lowered.push((out.slot_ids.clone(), out.slot_flags.clone()));
            txns.push(out.txn);
        }
        assert_eq!(lowered[0].1, vec![crate::driver::RS_FLAG_RESET]);
        assert_eq!(lowered[1].1, vec![0]);
        assert_eq!(lowered[2].1, vec![0]);
        assert_eq!(lowered[0].0, lowered[1].0);
        assert_eq!(lowered[1].0, lowered[2].0);
        assert_eq!(store.available_slots(), 1);
        for txn in txns {
            settle(&mut store, txn);
        }
    }

    // ------------------------------------------------------------------
    // Buffered write and fold — the FOLD-COMMIT path.
    // ------------------------------------------------------------------

    fn buffer_plan(start_token: u32, row_tokens: Vec<u32>) -> RsPlan {
        RsPlan::Buffer {
            start_tokens: vec![start_token; row_tokens.len()],
            fold_tokens: vec![0; row_tokens.len()],
            in_forward: vec![false; row_tokens.len()],
            row_tokens,
        }
    }

    #[test]
    fn buffered_write_materializes_slabs_and_leaves_the_fold_alone() {
        let mut store = RsStore::new(8);
        let ws = warm(&mut store);
        let folded = store.folded_slot(ws).unwrap().unwrap();
        store.alloc_buffer(ws, 3).unwrap();

        // 6 tokens at page 4 spans slabs 0 and 1.
        let out = prepare_many(&mut store, &[ws], &buffer_plan(0, vec![6])).unwrap();
        assert_eq!(
            out.slot_ids,
            vec![folded.0],
            "a buffered fire still names the folded slot it reads"
        );
        assert_eq!(out.slot_flags, vec![0], "a buffered fire never resets");
        assert_eq!(out.fold_lens, vec![0], "nothing is folded here");
        assert_eq!(out.buffer_slot_ids.len(), 2);
        assert_eq!(out.buffer_slot_indptr, vec![0, 2], "CSR: one row, 2 slabs");
        settle(&mut store, out.txn);

        assert_eq!(
            store.folded_slot(ws).unwrap(),
            Some(folded),
            "the folded state is untouched, so the tokens stay abandonable"
        );
        assert_eq!(store.resolve_buffer(ws, 0, 6).unwrap().len(), 2);
    }

    #[test]
    fn a_second_buffered_chunk_appends_to_later_slabs() {
        let mut store = RsStore::new(8);
        let ws = warm(&mut store);
        store.alloc_buffer(ws, 3).unwrap();

        let first = prepare_many(&mut store, &[ws], &buffer_plan(0, vec![4])).unwrap();
        let first_slab = first.buffer_slot_ids.clone();
        settle(&mut store, first.txn);

        let second = prepare_many(&mut store, &[ws], &buffer_plan(4, vec![4])).unwrap();
        assert_eq!(second.buffer_slot_indptr, vec![0, 1]);
        assert_ne!(
            second.buffer_slot_ids, first_slab,
            "a page-aligned successor chunk takes the NEXT slab"
        );
        settle(&mut store, second.txn);
    }

    #[test]
    fn fold_buffered_lowers_the_prefix_csr_and_advances_the_boundary() {
        let mut store = RsStore::new(8);
        let ws = warm(&mut store);
        let folded = store.folded_slot(ws).unwrap().unwrap();
        store.alloc_buffer(ws, 3).unwrap();
        let write = prepare_many(&mut store, &[ws], &buffer_plan(0, vec![12])).unwrap();
        let slabs = write.buffer_slot_ids.clone();
        settle(&mut store, write.txn);
        assert_eq!(slabs.len(), 3);

        // Fold the first 8 buffered tokens = slabs 0 and 1.
        let out = prepare_many(
            &mut store,
            &[ws],
            &RsPlan::FoldBuffered {
                tokens: vec![8],
                fold_len_is_device: false,
            },
        )
        .unwrap();
        assert_eq!(out.fold_lens, vec![8]);
        assert_eq!(out.buffer_slot_indptr, vec![0, 2]);
        assert_eq!(
            out.buffer_slot_ids,
            slabs[..2].to_vec(),
            "the fold gathers the buffered PREFIX, from slab zero"
        );
        assert_eq!(
            out.slot_ids,
            vec![folded.0],
            "the fold writes the working set's own folded slot in place"
        );
        settle(&mut store, out.txn);

        assert_eq!(
            store.buffer_size(ws).unwrap(),
            1,
            "the two fully covered head slabs are dropped"
        );
    }

    #[test]
    fn a_cold_working_set_cannot_buffer_or_fold() {
        let mut store = RsStore::new(4);
        let ws = store.create_working_set(geom());
        store.alloc_buffer(ws, 2).unwrap();

        let error = prepare_many(&mut store, &[ws], &buffer_plan(0, vec![4])).unwrap_err();
        assert!(error.contains("no folded state"), "{error}");
        let error = prepare_many(
            &mut store,
            &[ws],
            &RsPlan::FoldBuffered {
                tokens: vec![4],
                fold_len_is_device: false,
            },
        )
        .unwrap_err();
        assert!(error.contains("no folded state"), "{error}");
        assert_eq!(store.available_slots(), 4, "nothing leaked");
    }

    #[test]
    fn fold_beyond_the_buffer_is_rejected_before_any_allocation() {
        let mut store = RsStore::new(8);
        let ws = warm(&mut store);
        store.alloc_buffer(ws, 1).unwrap(); // capacity 4 tokens
        let free = store.available_slots();
        assert!(
            prepare_many(
                &mut store,
                &[ws],
                &RsPlan::FoldBuffered {
                    tokens: vec![8],
                    fold_len_is_device: false
                }
            )
            .is_err()
        );
        assert_eq!(store.available_slots(), free, "nothing leaked");
    }

    #[test]
    fn demand_counts_buffered_materialization() {
        let mut store = RsStore::new(8);
        let ws = warm(&mut store);
        store.alloc_buffer(ws, 3).unwrap();

        assert_eq!(
            demand(&store, &[ws], &RsPlan::Fold).unwrap(),
            0,
            "an in-place fold allocates nothing"
        );
        assert_eq!(
            demand(&store, &[ws], &buffer_plan(0, vec![12])).unwrap(),
            3,
            "three reserved slabs must be materialized"
        );

        let write = prepare_many(&mut store, &[ws], &buffer_plan(0, vec![12])).unwrap();
        settle(&mut store, write.txn);
        assert_eq!(
            demand(&store, &[ws], &buffer_plan(0, vec![12])).unwrap(),
            0,
            "rewriting materialized, uniquely owned slabs is in place"
        );

        let forked = store.fork(ws).unwrap();
        assert_eq!(
            demand(&store, &[forked], &buffer_plan(0, vec![12])).unwrap(),
            3,
            "after a fork every touched slab copies on write"
        );
    }

    #[test]
    fn buffered_rows_lower_one_csr_span_each() {
        let mut store = RsStore::new(12);
        let first = warm(&mut store);
        let second = warm(&mut store);
        store.alloc_buffer(first, 2).unwrap();
        store.alloc_buffer(second, 2).unwrap();

        let out = prepare_many(&mut store, &[first, second], &buffer_plan(0, vec![8, 4])).unwrap();
        assert_eq!(out.slot_ids.len(), 2);
        assert_eq!(
            out.buffer_slot_indptr,
            vec![0, 2, 3],
            "R+1 boundaries, leading zero, monotonic"
        );
        assert_eq!(out.buffer_slot_ids.len(), 3);
        settle(&mut store, out.txn);
    }

    #[test]
    fn no_working_sets_prepares_nothing() {
        let mut store = RsStore::new(2);
        let out = prepare_many(&mut store, &[], &RsPlan::Fold).unwrap();
        assert!(out.slot_ids.is_empty() && out.slot_flags.is_empty());
        assert!(out.buffer_slot_indptr.is_empty(), "no CSR without rows");
        assert!(out.txn.is_none());
        assert_eq!(store.available_slots(), 2);
    }

    #[test]
    fn aliased_rows_are_rejected_before_any_allocation() {
        let mut store = RsStore::new(4);
        let ws = store.create_working_set(geom());
        assert!(prepare_many(&mut store, &[ws, ws], &RsPlan::Fold).is_err());
        assert_eq!(store.available_slots(), 4);
        assert_eq!(store.folded_slot(ws).unwrap(), None);
    }

    #[test]
    fn preparation_failure_rolls_back_earlier_rows() {
        let mut store = RsStore::new(1);
        let first = store.create_working_set(geom());
        let second = store.create_working_set(geom());
        assert!(prepare_many(&mut store, &[first, second], &RsPlan::Fold).is_err());
        assert_eq!(store.folded_slot(first).unwrap(), None);
        assert_eq!(store.folded_slot(second).unwrap(), None);
        assert_eq!(store.available_slots(), 1);
    }

    #[test]
    fn failed_fire_keeps_the_published_state() {
        let mut store = RsStore::new(2);
        let ws = store.create_working_set(geom());
        let out = prepare(&mut store, ws).unwrap();
        let ids = out.slot_ids.clone();
        settle(&mut store, out.txn); // the fire failed on the device
        assert_eq!(
            store.folded_slot(ws).unwrap().map(|s| s.0),
            Some(ids[0]),
            "fail-stop: the published mapping stays authoritative"
        );
    }

    #[test]
    fn validates_pure_single_and_multi_request_arity() {
        assert_eq!(validate_count(0, &[], false).unwrap(), 0);
        assert_eq!(validate_count(0, &[0, 1, 2], false).unwrap(), 2);
        assert_eq!(validate_count(1, &[0, 1], true).unwrap(), 1);
        assert_eq!(validate_count(2, &[0, 1, 2], true).unwrap(), 2);
        assert!(validate_count(1, &[0, 1, 2], true).is_err());
        assert!(validate_count(1, &[0, 1], false).is_err());
    }
}
