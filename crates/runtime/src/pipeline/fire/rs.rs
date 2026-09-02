//! Fire RS (recurrent-state) preparation over the typed `RsStore`: classify,
//! publish, and lower each request row's recurrent-state work for the engine.
#![allow(dead_code)]

use crate::store::rs::write::RsBufferIntent;
use crate::store::rs::write::{RsPreparedWrite, RsPublished};
use crate::store::rs::{RsStore, RsWorkingSetId};

/// The published RS write for one in-flight ETA fire, held across
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
/// geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RsPlan {
    /// Advance the folded state in-forward over every row.
    Fold,
    /// Scatter each row's pre-recurrence activations into buffered slots
    /// covering `[start_tokens[r], start_tokens[r] + row_tokens[r])`, and
    /// advance the folded boundary through `fold_tokens[r]` of the resulting
    /// buffer.
    ///
    /// `start_tokens[r]` is row `r`'s existing buffer occupancy; non-zero
    /// means the fire appends onto a non-empty buffer and must read what is
    /// already there (the buffer read path). Per-row since occupancy is a
    /// property of the working set, not the fire.
    ///
    /// `fold_tokens[r]` is where the folded boundary lands, counted in the
    /// row's extended layout `[b | t]`. Zero leaves the folded state
    /// untouched (a pure append); any other value makes the pass a fold as
    /// well as a write, snapshotting recurrent state at that token via
    /// `commit_len`.
    ///
    /// `in_forward[r]` marks a row that owns no buffer in this pass: it
    /// folds its own new tokens straight into the folded state, riding
    /// along in a fire whose other rows buffer.
    Buffer {
        start_tokens: Vec<u32>,
        row_tokens: Vec<u32>,
        fold_tokens: Vec<u32>,
        in_forward: Vec<bool>,
    },
    /// Replay `tokens[r]` buffered tokens of row `r` into its folded state.
    ///
    /// When `fold_len_is_device`, `tokens[r]` is only an upper bound (the
    /// whole live buffer); the real value comes from the `rs_fold_len`
    /// descriptor port, and the engine clamps one to the other.
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
            // A fold gathers the buffered prefix from slab zero.
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

/// Engine lowering for one fire's published recurrent-state work.
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
    /// engine walks page-major. Empty unless the pass touches the buffer.
    /// These are the slabs the fire writes: `prepare` may materialize or
    /// privatize them, so they are targets, not sources.
    pub buffer_slot_ids: Vec<u32>,
    pub buffer_slot_indptr: Vec<u32>,
    /// The slabs the fire reads, and how many tokens of them, per row.
    /// Separate from the write CSR: reading a page that was merely reserved
    /// would gather uninitialized activations into the recurrence, so a
    /// write may allocate but a read must not.
    ///
    /// Non-empty only when a row appends onto a non-empty buffer: its
    /// recurrence starts from `folded (+) replay(buffer)`, so the engine
    /// gathers `buffer_read_lens[r]` tokens ahead of the row's own tokens.
    pub buffer_read_slot_ids: Vec<u32>,
    pub buffer_read_indptr: Vec<u32>,
    pub buffer_read_lens: Vec<u32>,
    /// Where each row's logical buffer token 0 physically sits, in tokens
    /// from the start of its first page.
    ///
    /// A fold absorbs tokens off the front of the buffer but can only
    /// release whole covered pages, so a fold routinely lands mid-page and
    /// the survivors keep their offsets; every buffer span the engine walks
    /// is `head + logical`.
    pub buffer_heads: Vec<u32>,
    /// The verb each request row asks the engine for, one per row, in
    /// resolved request order — what [`PreparedRs::apply_to`] stamps onto
    /// the lane that carries the row (`engine::Lane::rs`).
    ///
    /// `RsVerb::Buffer::pages` doubles as the working set's buffer page ->
    /// physical slot translation: a list of physical slot ids in buffer
    /// order, which the engine indexes to find the page a buffer token
    /// lives in.
    pub verbs: Vec<engine::fire::RsVerb>,
    pub txn: Option<RsTxn>,
}

impl PreparedRs {
    /// The no-recurrent-state lowering: every field empty.
    pub fn empty() -> Self {
        empty_prepared()
    }

    /// Stamp this lowering onto the lanes that carry it: one `RsVerb` and
    /// one `RsReset` per lane, addressed to the lane's own engine.
    ///
    /// Row `r` is lane `r`: `validate_count` refuses a fire whose bound
    /// working sets and resolved `qo_indptr` rows disagree, and a
    /// `FireRequest`'s `qo_indptr` is cut from its lanes.
    pub fn apply_to(&self, request: &mut crate::engine::FireRequest) {
        for (row, lane) in request.lanes.iter_mut().enumerate() {
            if let Some(verb) = self.verbs.get(row) {
                lane.rs = verb.clone();
            }
            // `RS_FLAG_RESET` is the RS store's own classification, not
            // inferred from `kv.held == 0` (a coincidence, not an identity:
            // this store forks, restores and recycles seats independently
            // of the KV side).
            if let Some(&flags) = self.slot_flags.get(row) {
                lane.rs_reset = if flags & crate::engine::RS_FLAG_RESET != 0 {
                    engine::fire::RsReset::Fresh
                } else {
                    engine::fire::RsReset::Held
                };
            }
        }
    }
}

/// Prepare and publish this fire's recurrent-state work. Returns the engine
/// lowering: thread the ids and flags into the launch in request order, issue
/// one aggregated state-copy command before the launch when non-empty, hold
/// `txn` across the fire, then [`settle`].
///
/// Publishing here (not at settle) is what lets RS fires run ahead: a
/// successor prepared before its predecessor completes classifies against a
/// mapping that already carries the predecessor's decision, so it cannot
/// reset a slot twice or re-CoW an already-privatized one.
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
        verbs: Vec::new(),
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
        // a fold advances it — both need one already to exist. Reject it
        // here, where the guest gets a message it can act on.
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
                    // The fold length here is only the host's upper bound;
                    // mark it so `publish_batch` holds the boundary instead
                    // of moving it as if this were the truth.
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
        // committed slot the buffered forward reads.
        match prepared.state() {
            Some(state) => {
                out.slot_ids.push(state.slot.0);
                let mut flags = if state.reset {
                    crate::engine::RS_FLAG_RESET
                } else {
                    0
                };
                if state.fold_tokens.is_some() {
                    flags |= crate::engine::RS_FLAG_FOLD;
                }
                // `fold_lens[r]` here is the host's upper bound, not the
                // fold length; the engine clamps the resolved `rs_fold_len`
                // port to it.
                if prepared.fold_len_is_bound() {
                    flags |= crate::engine::RS_FLAG_FOLD_LEN_DEVICE;
                }
                // Orthogonal to FOLD: a pass that both writes the buffer and
                // folds a prefix runs the extended layout and snapshots the
                // state at `fold_lens[r]`, which the fold flag alone cannot
                // tell apart from a pure replay.
                if buffer_intent == RsBufferIntent::Write
                    && buffer_tokens.is_some_and(|(_, len)| len > 0)
                {
                    flags |= crate::engine::RS_FLAG_BUFFER_WRITE;
                }
                out.slot_flags.push(flags);
                if let Some(src) = state.copy_from {
                    out.copies.0.push(src.0);
                    out.copies.1.push(state.slot.0);
                }
                // One length per row, always: a row that skipped one would
                // shift every later row's.
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

        // Buffer CSR. The engine walks a row's slabs page-major from the
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
    // this fire's decision without waiting for the device. The fold advance
    // is deferred (`RsStore::commit_folds`) so the buffer below is described
    // in the pre-fold frame this fire's rows were laid out in.
    let (published, pending_folds) = store
        .publish_batch(prepared_rows)
        .map_err(|error| error.to_string())?;
    // Runs inside a closure so a failure cannot skip `commit_folds` or the
    // settle below: once `publish_batch` returns, the store has already
    // committed the mapping and taken an in-flight hold that must be
    // released or slot recycling wedges for the rest of the process.
    let read_side = (|out: &mut PreparedRs| -> Result<(), String> {
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

            // The read prefix is the row's pre-existing occupancy, which is
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

            // `row` is the working set's whole buffer page -> physical slot
            // map, read after the publish so it names the pages this fire
            // just materialized or privatized. The head is folded into `at`
            // rather than stated beside it, since physical buffer token
            // `head + logical` is what the store's own page arithmetic uses.
            //
            // Cut at the last page this fire can address: a reserved (not
            // yet materialized) tail has no physical slot to name, so
            // stating the addressed prefix keeps the engine's capacity check
            // measuring real pages.
            let page_tokens = page_tokens_of(ws)?;
            let run_through = |end: u32| -> Vec<u32> {
                if row.is_empty() || end == 0 {
                    return Vec::new();
                }
                let last = ((end - 1) / page_tokens) as usize;
                row[..=last.min(row.len() - 1)].to_vec()
            };
            out.verbs.push(match plan {
                RsPlan::Fold => engine::fire::RsVerb::Fold,
                RsPlan::Buffer {
                    start_tokens,
                    row_tokens,
                    fold_tokens,
                    in_forward,
                } => {
                    // A row that owns no buffer in this pass folds its own
                    // new tokens in the forward, exactly as `RsPlan::Fold`
                    // does, while riding in a fire whose peers buffer.
                    if in_forward.get(index).copied().unwrap_or(false) {
                        engine::fire::RsVerb::Fold
                    } else {
                        let at =
                            head.saturating_add(start_tokens.get(index).copied().unwrap_or(0));
                        engine::fire::RsVerb::Buffer {
                            pages: run_through(
                                at.saturating_add(row_tokens.get(index).copied().unwrap_or(0)),
                            ),
                            at,
                            // A non-zero fold is stated, not smoothed: the
                            // engine cuts the row into the segment that
                            // folds and the segment that does not. Counted
                            // in the row's extended layout `[b | t]`; a
                            // count past the lane's rows is refused by
                            // `Lane::validate`.
                            fold: engine::fire::FoldLen::Host(
                                fold_tokens.get(index).copied().unwrap_or(0),
                            ),
                            // The read path: the row's pre-existing occupancy
                            // is what the recurrence must replay ahead of the
                            // row's own tokens, and it sits right below `at`.
                            replay: start_tokens.get(index).copied().unwrap_or(0),
                        }
                    }
                }
                RsPlan::FoldBuffered {
                    tokens,
                    fold_len_is_device,
                } => {
                    let bound = tokens.get(index).copied().unwrap_or(0);
                    // A working set whose last fold landed mid-page needs
                    // its head stated, or a replay would re-fold the
                    // absorbed tokens ahead of the live ones.
                    // `RsVerb::FoldBuffered::at` is that number, from the
                    // same origin `RsVerb::Buffer::at` counts in.
                    engine::fire::RsVerb::FoldBuffered {
                        pages: run_through(head.saturating_add(bound)),
                        at: head,
                        bound,
                        // A device-resident count is read off the
                        // `rs_fold_len` port at compose and clamped to the
                        // bound the host published here.
                        len: if *fold_len_is_device {
                            engine::fire::FoldLen::Device(
                                eta_ir::registry::Port::RsFoldLen,
                            )
                        } else {
                            engine::fire::FoldLen::Host(bound)
                        },
                    }
                }
            });
        }
        // Keep the wire quiet for the common empty-buffer fire: an all-zero
        // read side is the same statement as an absent one.
        if !any_read {
            out.buffer_read_slot_ids.clear();
            out.buffer_read_indptr.clear();
            out.buffer_read_lens.clear();
        }
        Ok(())
    })(&mut out);

    // Every wire array is now built against the pre-fold buffer, so the
    // boundary can finally move.
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
/// pipeline-local state. A failed fire leaves the recurrent state undefined
/// and the pass poisons its readers; settling releases the store's
/// in-flight hold so recycled slots become allocatable.
pub fn settle(store: &mut RsStore, txn: Option<RsTxn>) {
    let Some(RsTxn { published }) = txn else {
        return;
    };
    store.settle(published);
}

#[cfg(test)]
mod tests {
    use engine::fire::{FoldLen, RsReset, RsVerb};

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
        assert_eq!(out.slot_flags, vec![crate::engine::RS_FLAG_RESET]);
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

    // Run-ahead: each successor prepares while its predecessor is still in
    // flight.

    // Buffered write and fold: the fold-commit path.

    fn buffer_plan(start_token: u32, row_tokens: Vec<u32>) -> RsPlan {
        RsPlan::Buffer {
            start_tokens: vec![start_token; row_tokens.len()],
            fold_tokens: vec![0; row_tokens.len()],
            in_forward: vec![false; row_tokens.len()],
            row_tokens,
        }
    }

    /// One request row, one lane, and the lane carries whatever the plan
    /// resolved to.
    fn request(rows: usize) -> crate::engine::FireRequest {
        crate::engine::FireRequest {
            lanes: (0..rows)
                .map(|row| {
                    crate::engine::fire::lane_of(row as u32, vec![7], 0, vec![row as u32])
                })
                .collect(),
            ..crate::engine::FireRequest::default()
        }
    }

    fn lowered(prepared: &PreparedRs, rows: usize) -> Vec<(RsVerb, RsReset)> {
        let mut req = request(rows);
        prepared.apply_to(&mut req);
        req.lanes
            .into_iter()
            .map(|lane| (lane.rs, lane.rs_reset))
            .collect()
    }

    /// The verb each plan shape asks the engine for. A plan shape that
    /// lowered to the wrong verb is a fire that folds state it was asked to
    /// buffer (unrecoverable, invisible until a speculation's rejection
    /// reads a state that already advanced), or a page list that disagrees
    /// with the store's own translation and scatters into someone else's
    /// activations.
    #[test]
    fn every_plan_shape_lowers_to_its_lane_verb() {
        let mut store = RsStore::new(16);
        let ws = store.create_working_set(geom());

        // FOLD, and the reset fact that is the RS store's own.
        let first = prepare(&mut store, ws).unwrap();
        assert_eq!(
            lowered(&first, 1),
            vec![(RsVerb::Fold, RsReset::Fresh)],
            "a first fire folds in-forward into a bank that must be zeroed"
        );
        settle(&mut store, first.txn);
        let second = prepare(&mut store, ws).unwrap();
        assert_eq!(
            lowered(&second, 1),
            vec![(RsVerb::Fold, RsReset::Held)],
            "and the successor continues the same bank, whatever its KV says"
        );
        settle(&mut store, second.txn);

        // BUFFER: a pure scatter over 6 tokens of a 4-token page, so the run
        // is the two pages the write materialized, not the reserved third.
        store.alloc_buffer(ws, 3).unwrap();
        let write = prepare_many(&mut store, &[ws], &buffer_plan(0, vec![6])).unwrap();
        let slabs = write.buffer_slot_ids.clone();
        assert_eq!(slabs.len(), 2);
        assert_eq!(
            lowered(&write, 1),
            vec![(
                RsVerb::Buffer {
                    pages: slabs.clone(),
                    at: 0,
                    fold: FoldLen::Host(0),
                    replay: 0,
                },
                RsReset::Held,
            )],
            "the list IS the translation: physical slots, in buffer order"
        );
        settle(&mut store, write.txn);

        // BUFFER, appending: `at` is the row's occupancy; the run still
        // starts at buffer token zero.
        let append = prepare_many(&mut store, &[ws], &buffer_plan(6, vec![2])).unwrap();
        let RsVerb::Buffer {
            pages,
            at,
            fold,
            replay,
        } = lowered(&append, 1)[0].0.clone()
        else {
            panic!("an append is a scatter");
        };
        assert_eq!(at, 6, "the fire's first row lands on the row's occupancy");
        assert_eq!(replay, 6, "the six tokens below `at` are replayed ahead of the new rows");
        assert_eq!(fold, FoldLen::Host(0), "a pure append folds nothing");
        assert_eq!(pages, slabs, "the same two pages, from the same origin");
        settle(&mut store, append.txn);

        // FOLD-BUFFERED: the bound sizes the launch, and the run covers the
        // pages it replays.
        let fold = prepare_many(
            &mut store,
            &[ws],
            &RsPlan::FoldBuffered {
                tokens: vec![8],
                fold_len_is_device: false,
            },
        )
        .unwrap();
        assert_eq!(
            lowered(&fold, 1),
            vec![(
                RsVerb::FoldBuffered {
                    pages: slabs.clone(),
                    at: 0,
                    bound: 8,
                    len: FoldLen::Host(8),
                },
                RsReset::Held,
            )]
        );
        settle(&mut store, fold.txn);
    }

    /// A device-resident fold length is the port, not a number: the count is
    /// computed by the verifier on the stream. The host still states the
    /// bound (the whole live buffer), since that sizes the launch, and the
    /// engine clamps.
    #[test]
    fn a_device_resident_fold_length_lowers_to_its_port() {
        let mut store = RsStore::new(16);
        let ws = warm(&mut store);
        store.alloc_buffer(ws, 3).unwrap();
        let write = prepare_many(&mut store, &[ws], &buffer_plan(0, vec![12])).unwrap();
        let slabs = write.buffer_slot_ids.clone();
        settle(&mut store, write.txn);

        let fold = prepare_many(
            &mut store,
            &[ws],
            &RsPlan::FoldBuffered {
                tokens: vec![12],
                fold_len_is_device: true,
            },
        )
        .unwrap();
        assert_eq!(
            lowered(&fold, 1),
            vec![(
                RsVerb::FoldBuffered {
                    pages: slabs,
                    at: 0,
                    bound: 12,
                    len: FoldLen::Device(eta_ir::registry::Port::RsFoldLen),
                },
                RsReset::Held,
            )],
            "the bound is the host's; the length is the port's"
        );
        settle(&mut store, fold.txn);
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

    /// A replay after a mid-page fold starts at the buffer's head: a fold
    /// absorbs tokens off the front of the buffer but can only release whole
    /// covered pages, so survivors sit physically offset inside a page they
    /// share with tokens that are gone. Without an origin, the replay would
    /// re-fold the absorbed tokens before reaching a live one.
    #[test]
    fn a_replay_after_a_mid_page_fold_starts_at_the_buffer_head() {
        let mut store = RsStore::new(8);
        let ws = warm(&mut store);
        store.alloc_buffer(ws, 2).unwrap();
        let write = prepare_many(&mut store, &[ws], &buffer_plan(0, vec![6])).unwrap();
        let slabs = write.buffer_slot_ids.clone();
        settle(&mut store, write.txn);
        assert_eq!(slabs.len(), 2, "six tokens over four-token pages span two");

        // Fold THREE of the six: a page holds four, so the boundary lands
        // inside page zero and neither page can be released.
        let fold = prepare_many(
            &mut store,
            &[ws],
            &RsPlan::FoldBuffered {
                tokens: vec![3],
                fold_len_is_device: false,
            },
        )
        .unwrap();
        settle(&mut store, fold.txn);
        assert_eq!(
            store.buffer_head(ws).unwrap(),
            3,
            "a mid-page fold leaves the survivors offset inside their page"
        );
        assert_eq!(
            store.buffer_size(ws).unwrap(),
            2,
            "and releases neither page, because neither is wholly covered"
        );

        // The three survivors replay from the head, not from buffer token
        // zero, and the run still names the pages from the same origin.
        let replay = prepare_many(
            &mut store,
            &[ws],
            &RsPlan::FoldBuffered {
                tokens: vec![3],
                fold_len_is_device: false,
            },
        )
        .unwrap();
        assert_eq!(
            lowered(&replay, 1)[0].0,
            RsVerb::FoldBuffered {
                pages: slabs,
                at: 3,
                bound: 3,
                len: FoldLen::Host(3),
            },
            "the replay states the head the last fold left behind"
        );
        settle(&mut store, replay.txn);
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

}
