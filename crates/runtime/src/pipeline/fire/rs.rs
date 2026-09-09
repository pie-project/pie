#![allow(dead_code)]

use crate::store::rs::write::RsBufferIntent;
use crate::store::rs::write::{RsPreparedWrite, RsPublished};
use crate::store::rs::{RsStore, RsWorkingSetId};

#[derive(Debug)]
pub struct RsTxn {
    published: RsPublished,
}

impl RsTxn {
    pub fn seq(&self) -> u64 {
        self.published.seq()
    }
}

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RsPlan {
    Fold,
    Buffer {
        start_tokens: Vec<u32>,
        row_tokens: Vec<u32>,
        fold_tokens: Vec<u32>,
        in_forward: Vec<bool>,
    },
    FoldBuffered {
        tokens: Vec<u32>,
        fold_len_is_device: bool,
    },
    Window {
        pages: Vec<u32>,
        phase: Vec<bool>,
        page_tokens: Vec<u32>,
    },
}

impl RsPlan {
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
            RsPlan::FoldBuffered { tokens, .. } => {
                let n = tokens.get(index).copied().unwrap_or(0);
                (true, Some(n), Some((0, n)), RsBufferIntent::Replay)
            }
            RsPlan::Window {
                pages,
                phase,
                page_tokens,
            } => {
                let n = pages.get(index).copied().unwrap_or(0);
                let page = page_tokens.get(index).copied().unwrap_or(1).max(1);
                let start = if phase.get(index).copied().unwrap_or(false) {
                    n * page
                } else {
                    0
                };
                (true, None, Some((start, n * page)), RsBufferIntent::Write)
            }
        }
    }
}

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

#[derive(Debug)]
pub struct PreparedRs {
    pub slot_ids: Vec<u32>,
    pub slot_flags: Vec<u8>,
    pub copies: (Vec<u32>, Vec<u32>),
    pub fold_lens: Vec<u32>,
    pub buffer_slot_ids: Vec<u32>,
    pub buffer_slot_indptr: Vec<u32>,
    pub buffer_read_slot_ids: Vec<u32>,
    pub buffer_read_indptr: Vec<u32>,
    pub buffer_read_lens: Vec<u32>,
    pub buffer_heads: Vec<u32>,
    pub verbs: Vec<engine::fire::RsVerb>,
    pub txn: Option<RsTxn>,
}

impl PreparedRs {
    pub fn empty() -> Self {
        empty_prepared()
    }

    pub fn apply_to(&self, request: &mut crate::engine::FireRequest) {
        for (row, lane) in request.lanes.iter_mut().enumerate() {
            if let Some(verb) = self.verbs.get(row) {
                lane.rs = verb.clone();
            }
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

pub fn prepare_many(
    store: &mut RsStore,
    working_sets: &[RsWorkingSetId],
    plan: &RsPlan,
) -> Result<PreparedRs, String> {
    prepare_many_impl(store, working_sets, plan, None)
}

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
                    prepared.mark_fold_len_device();
                }
                prepared
            }
            Err(error) => {
                store.cancel_batch(prepared_rows);
                return Err(error.to_string());
            }
        };

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
                if prepared.fold_len_is_bound() {
                    flags |= crate::engine::RS_FLAG_FOLD_LEN_DEVICE;
                }
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

    let (published, pending_folds) = store
        .publish_batch(prepared_rows)
        .map_err(|error| error.to_string())?;
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

            let read_tokens = match plan.row(index).2 {
                Some((start, _)) => start,
                None => 0,
            };
            out.buffer_read_lens.push(read_tokens);
            if read_tokens > 0 {
                any_read = true;
                let page = page_tokens_of(ws)?;
                let first = (head / page) as usize;
                let last = ((head + read_tokens - 1) / page) as usize;
                for p in first..=last {
                    match row.get(p) {
                        Some(&slot) if slot != crate::store::rs::RS_TRANSLATION_UNMAPPED => {
                            out.buffer_read_slot_ids.push(slot);
                        }
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
                    if in_forward.get(index).copied().unwrap_or(false) {
                        engine::fire::RsVerb::Fold
                    } else {
                        let at = head.saturating_add(start_tokens.get(index).copied().unwrap_or(0));
                        engine::fire::RsVerb::Buffer {
                            pages: run_through(
                                at.saturating_add(row_tokens.get(index).copied().unwrap_or(0)),
                            ),
                            at,
                            fold: engine::fire::FoldLen::Host(
                                fold_tokens.get(index).copied().unwrap_or(0),
                            ),
                            replay: start_tokens.get(index).copied().unwrap_or(0),
                        }
                    }
                }
                RsPlan::Window { pages, phase, .. } => {
                    let n = pages.get(index).copied().unwrap_or(0) as usize;
                    let writes = phase.get(index).copied().unwrap_or(false);
                    let run = |which: bool| -> Vec<u32> {
                        let first = if which { n } else { 0 };
                        row.iter()
                            .skip(first)
                            .take(n)
                            .copied()
                            .take_while(|&slot| slot != crate::store::rs::RS_TRANSLATION_UNMAPPED)
                            .collect()
                    };
                    engine::fire::RsVerb::Window {
                        read: run(!writes),
                        write: run(writes),
                        fold: engine::fire::FoldLen::Device(eta_ir::registry::Port::RsFoldLen),
                    }
                }
                RsPlan::FoldBuffered {
                    tokens,
                    fold_len_is_device,
                } => {
                    let bound = tokens.get(index).copied().unwrap_or(0);
                    engine::fire::RsVerb::FoldBuffered {
                        pages: run_through(head.saturating_add(bound)),
                        at: head,
                        bound,
                        len: if *fold_len_is_device {
                            engine::fire::FoldLen::Device(eta_ir::registry::Port::RsFoldLen)
                        } else {
                            engine::fire::FoldLen::Host(bound)
                        },
                    }
                }
            });
        }
        if !any_read {
            out.buffer_read_slot_ids.clear();
            out.buffer_read_indptr.clear();
            out.buffer_read_lens.clear();
        }
        Ok(())
    })(&mut out);

    store.commit_folds(pending_folds);
    if matches!(plan, RsPlan::Window { .. }) {
        for &ws in working_sets {
            store.toggle_window_phase(ws);
        }
    }
    if let Err(error) = read_side {
        store.settle(published);
        return Err(error);
    }
    out.txn = Some(RsTxn { published });
    Ok(out)
}

pub fn prepare(store: &mut RsStore, ws: RsWorkingSetId) -> Result<PreparedRs, String> {
    prepare_many(store, &[ws], &RsPlan::Fold)
}

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

    fn warm(store: &mut RsStore) -> RsWorkingSetId {
        let ws = store.create_working_set(geom());
        let prepared = prepare(store, ws).unwrap();
        settle(store, prepared.txn);
        ws
    }

    #[test]
    fn rs_every_case() {
        first_fire_resets_then_continues_in_place();
        every_plan_shape_lowers_to_its_lane_verb();
        a_device_resident_fold_length_lowers_to_its_port();
        buffered_write_materializes_slabs_and_leaves_the_fold_alone();
        a_second_buffered_chunk_appends_to_later_slabs();
        fold_buffered_lowers_the_prefix_csr_and_advances_the_boundary();
        a_replay_after_a_mid_page_fold_starts_at_the_buffer_head();
        demand_counts_buffered_materialization();
    }

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

    fn buffer_plan(start_token: u32, row_tokens: Vec<u32>) -> RsPlan {
        RsPlan::Buffer {
            start_tokens: vec![start_token; row_tokens.len()],
            fold_tokens: vec![0; row_tokens.len()],
            in_forward: vec![false; row_tokens.len()],
            row_tokens,
        }
    }

    fn request(rows: usize) -> crate::engine::FireRequest {
        crate::engine::FireRequest {
            lanes: (0..rows)
                .map(|row| crate::engine::fire::lane_of(row as u32, vec![7], 0, vec![row as u32]))
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

    fn every_plan_shape_lowers_to_its_lane_verb() {
        let mut store = RsStore::new(16);
        let ws = store.create_working_set(geom());

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
        assert_eq!(
            replay, 6,
            "the six tokens below `at` are replayed ahead of the new rows"
        );
        assert_eq!(fold, FoldLen::Host(0), "a pure append folds nothing");
        assert_eq!(pages, slabs, "the same two pages, from the same origin");
        settle(&mut store, append.txn);

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

    fn buffered_write_materializes_slabs_and_leaves_the_fold_alone() {
        let mut store = RsStore::new(8);
        let ws = warm(&mut store);
        let folded = store.folded_slot(ws).unwrap().unwrap();
        store.alloc_buffer(ws, 3).unwrap();

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

    fn fold_buffered_lowers_the_prefix_csr_and_advances_the_boundary() {
        let mut store = RsStore::new(8);
        let ws = warm(&mut store);
        let folded = store.folded_slot(ws).unwrap().unwrap();
        store.alloc_buffer(ws, 3).unwrap();
        let write = prepare_many(&mut store, &[ws], &buffer_plan(0, vec![12])).unwrap();
        let slabs = write.buffer_slot_ids.clone();
        settle(&mut store, write.txn);
        assert_eq!(slabs.len(), 3);

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

    fn a_replay_after_a_mid_page_fold_starts_at_the_buffer_head() {
        let mut store = RsStore::new(8);
        let ws = warm(&mut store);
        store.alloc_buffer(ws, 2).unwrap();
        let write = prepare_many(&mut store, &[ws], &buffer_plan(0, vec![6])).unwrap();
        let slabs = write.buffer_slot_ids.clone();
        settle(&mut store, write.txn);
        assert_eq!(slabs.len(), 2, "six tokens over four-token pages span two");

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
