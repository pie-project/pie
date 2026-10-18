use crate::store::rs::write::{RsPreparedWrite, RsPublished, RsSnapshot};
use crate::store::rs::{RsGrant, RsSlotId, RsStore, RsUnits, RsWorkingSetId};

#[derive(Debug)]
pub struct RsTxn {
    published: RsPublished,
}

impl RsTxn {
    /// The snapshots the fire was to land, which no checkpoint will take.
    pub fn abandon(&self, store: &mut RsStore) {
        for snap in self.published.snapshots() {
            store.release_working_set(snap.snapshot);
        }
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
            "a model without recurrent state bound {rs_count} recurrent state(s)"
        ));
    }
    let request_count = qo_indptr
        .len()
        .checked_sub(1)
        .ok_or_else(|| "resolved qo_indptr is empty".to_string())?;
    if rs_count != request_count {
        return Err(format!(
            "the pass has {request_count} lane(s) and binds {rs_count} recurrent state(s); a \
             model with recurrent state binds one a lane",
        ));
    }
    Ok(request_count)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RsPlan {
    Groups(Vec<GroupPlan>),
    /// The model's state is a window: each row writes the ring pages of
    /// the kv page indexes it lands on, then moves its ring.
    Ring(Vec<RingRow>),
}

/// The first lane carries its fold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupPlan {
    pub lanes: Vec<usize>,
    /// The slots its lanes' rows land on in its buffer. None when its one
    /// lane's rows follow what the state holds and all fold in the forward,
    /// so nothing is buffered.
    pub writes: Option<std::ops::Range<u32>>,
    pub fold: GroupFold,
    /// Rows of its one lane the fire lands the state as of, each in a slot
    /// of its own (`Lane::rs_snapshots`).
    pub snapshots: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupFold {
    None,
    Host(u32),
    Device,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingRow {
    /// The kv page indexes the row writes.
    pub pages: Vec<u32>,
    /// The kv page indexes the lane reads, in the order its table states.
    pub translate: Vec<u32>,
    pub advance: crate::store::rs::RingAdvance,
}

/// What a fire writes of a working set's state: whether the state, to a
/// fresh slot (other lanes read the one it holds), the count it freezes on
/// publish, and the buffered slots.
type GroupWrite = (bool, bool, Option<u32>, Option<std::ops::Range<u32>>);

impl GroupPlan {
    fn write(&self, store: &RsStore, ws: RsWorkingSetId) -> Result<GroupWrite, String> {
        let held = store.folded_slot(ws).map_err(|error| error.to_string())?;
        let fold = match self.fold {
            GroupFold::Host(count) if count > 0 => Some(count),
            _ => None,
        };
        let folds = fold.is_some() || self.fold == GroupFold::Device;
        // A state that holds nothing yet starts from zeros, which every lane
        // reads: only when none of them folds into it in the same fire.
        if held.is_none() && folds && self.lanes.len() > 1 {
            return Err(format!(
                "lanes {:?} share a working set whose recurrent state holds nothing yet and \
                 fold it in the same fire; fold a pass of one lane into it first",
                self.lanes
            ));
        }
        let write_state = folds || held.is_none();
        let fresh = folds && held.is_some() && self.lanes.len() > 1;
        Ok((write_state, fresh, fold, self.writes.clone()))
    }
}

pub fn demand(
    store: &RsStore,
    working_sets: &[RsWorkingSetId],
    plan: &RsPlan,
) -> Result<RsUnits, String> {
    let mut total = RsUnits::default();
    match plan {
        RsPlan::Ring(rows) => {
            for (&ws, row) in working_sets.iter().zip(rows) {
                total = total.add(
                    store
                        .ring_demand(ws, &row.pages)
                        .map_err(|e| e.to_string())?,
                );
            }
        }
        RsPlan::Groups(groups) => {
            for group in groups {
                let ws = working_sets[group.lanes[0]];
                let (write_state, fresh, _, writes) = group.write(store, ws)?;
                total = total.add(
                    store
                        .group_demand(ws, write_state, fresh, writes)
                        .map_err(|e| e.to_string())?,
                );
                total.slots += group.snapshots.len() as u32;
            }
        }
    }
    Ok(total)
}

#[derive(Debug, Default)]
pub struct PreparedRs {
    pub slot_ids: Vec<u32>,
    pub slot_flags: Vec<u8>,
    pub copies: (Vec<u32>, Vec<u32>),
    /// In the engine's page numbering.
    pub page_copies: (Vec<u32>, Vec<u32>),
    pub buffer_slot_ids: Vec<u32>,
    pub verbs: Vec<engine::fire::RsVerb>,
    /// A ring model's rows: each lane's windowed page ids, aligned with
    /// the page indexes its row translates, and the windowed pages copied
    /// `(src, dst)` before the fire runs.
    pub window: Vec<Vec<u32>>,
    pub window_copies: Vec<(u32, u32)>,
    pub snapshots: Vec<RsSnapshot>,
    pub txn: Option<RsTxn>,
}

impl PreparedRs {
    pub fn state_copy(&self) -> Option<engine::StateCopy> {
        let moves = |(src, dst): &(Vec<u32>, Vec<u32>)| -> Vec<engine::transfer::StateMove> {
            src.iter()
                .zip(dst)
                .map(|(&src_slot_id, &dst_slot_id)| engine::transfer::StateMove {
                    src_slot_id,
                    dst_slot_id,
                    ..Default::default()
                })
                .collect()
        };
        let copy = engine::StateCopy {
            moves: moves(&self.copies),
            pages: moves(&self.page_copies),
            ..Default::default()
        };
        (!copy.moves.is_empty() || !copy.pages.is_empty()).then_some(copy)
    }

    pub fn apply_to(&self, request: &mut crate::engine::FireRequest) {
        for (row, lane) in request.lanes.iter_mut().enumerate() {
            if let Some(verb) = self.verbs.get(row) {
                lane.rs = verb.clone();
            }
            lane.rs_slot = self.slot_ids.get(row).copied();
            if let Some(&flags) = self.slot_flags.get(row) {
                lane.rs_reset = if flags & crate::engine::RS_FLAG_RESET != 0 {
                    engine::fire::RsReset::Fresh
                } else {
                    engine::fire::RsReset::Held
                };
            }
            if let Some(window) = self.window.get(row) {
                lane.kv.window.clone_from(window);
            }
        }
        if let [lane] = request.lanes.as_mut_slice() {
            lane.rs_snapshots = self
                .snapshots
                .iter()
                .map(|snap| engine::fire::RsSnapshot {
                    row: snap.row,
                    slot: snap.slot.0,
                })
                .collect();
        }
        if let Some(lane) = request.lanes.first_mut()
            && !self.window_copies.is_empty()
        {
            lane.kv.window_copies.clone_from(&self.window_copies);
        }
    }
}

/// Records every slot the batch hands the engine as read by it, so a
/// slot one of its fires reads outlives that fire however it is dropped.
fn note_reads(store: &mut RsStore, published: &RsPublished, out: &PreparedRs) {
    let Some(&seq) = published.seqs().last() else {
        return;
    };
    let verb_pages = out.verbs.iter().flat_map(|verb| match verb {
        engine::fire::RsVerb::Buffer { pages, .. } => pages.clone(),
        engine::fire::RsVerb::Fold => Vec::new(),
    });
    let live = |&id: &u32| id != 0 && id != crate::store::rs::RS_TRANSLATION_UNMAPPED;
    let slots: Vec<RsSlotId> = out
        .slot_ids
        .iter()
        .chain(&out.buffer_slot_ids)
        .chain(&out.copies.0)
        .copied()
        .filter(live)
        .map(RsSlotId)
        .collect();
    let pages: Vec<RsSlotId> = out
        .window
        .iter()
        .flatten()
        .chain(&out.page_copies.0)
        .copied()
        .chain(verb_pages)
        .filter(live)
        .map(|page| store.page_id(page))
        .collect();
    store.note_reads(seq, slots.into_iter().chain(pages));
}

fn prepare_ring_rows(
    store: &mut RsStore,
    working_sets: &[RsWorkingSetId],
    rows: &[RingRow],
    mut granted: Option<&mut RsGrant>,
) -> Result<PreparedRs, String> {
    if rows.len() != working_sets.len() {
        return Err(format!(
            "a ring plan of {} row(s) for {} recurrent state(s)",
            rows.len(),
            working_sets.len()
        ));
    }
    let mut out = PreparedRs::default();
    let mut prepared_rows: Vec<RsPreparedWrite> = Vec::with_capacity(rows.len());
    for (index, (&ws, row)) in working_sets.iter().zip(rows).enumerate() {
        match store.prepare_ring(ws, &row.pages, granted.as_deref_mut()) {
            Ok(prepared) => prepared_rows.push(prepared),
            Err(error) => {
                store.cancel_batch(prepared_rows);
                return Err(format!("request row {index}: {error}"));
            }
        }
    }
    for prepared in &prepared_rows {
        out.window_copies.extend(
            prepared
                .buffer_copy_plan()
                .map(|(src, dst)| (store.engine_id(src), store.engine_id(dst))),
        );
    }
    let (published, folds) = store
        .publish_batch(prepared_rows)
        .map_err(|error| error.to_string())?;
    store.commit_folds(folds);
    // The fire reads through the ring as published; only then does the
    // commit retire what falls behind it.
    for (&ws, row) in working_sets.iter().zip(rows) {
        out.window.push(
            store
                .ring_translation(ws, row.translate.iter().copied())
                .map_err(|error| error.to_string())?,
        );
    }
    note_reads(store, &published, &out);
    for (&ws, row) in working_sets.iter().zip(rows) {
        store.ring_advance(ws, row.advance, &row.translate);
    }
    out.txn = Some(RsTxn { published });
    Ok(out)
}

pub fn prepare_many(
    store: &mut RsStore,
    working_sets: &[RsWorkingSetId],
    plan: &RsPlan,
    mut granted: Option<&mut RsGrant>,
) -> Result<PreparedRs, String> {
    if working_sets.is_empty() {
        return Ok(PreparedRs::default());
    }
    let groups = match plan {
        RsPlan::Ring(rows) => {
            for (index, ws) in working_sets.iter().enumerate() {
                if working_sets[..index].contains(ws) {
                    return Err(format!(
                        "lane {index} names an earlier lane's working set; on a model whose state is a \
                         window each lane needs a working set of its own"
                    ));
                }
            }
            return prepare_ring_rows(store, working_sets, rows, granted);
        }
        RsPlan::Groups(groups) => groups,
    };
    let lanes = working_sets.len();
    let mut out = PreparedRs {
        verbs: vec![engine::fire::RsVerb::Fold; lanes],
        slot_ids: vec![0; lanes],
        slot_flags: vec![0; lanes],
        ..PreparedRs::default()
    };
    let mut prepared: Vec<RsPreparedWrite> = Vec::with_capacity(groups.len());
    for group in groups {
        let ws = working_sets[group.lanes[0]];
        let held = store.folded_slot(ws).map_err(|error| error.to_string())?;
        let write = group
            .write(store, ws)
            .and_then(|(write_state, fresh, fold, writes)| {
                let mut write = store
                    .prepare_group(ws, write_state, fresh, fold, writes, granted.as_deref_mut())
                    .map_err(|error| error.to_string())?;
                if let Err(error) =
                    store.prepare_snapshots(&mut write, &group.snapshots, granted.as_deref_mut())
                {
                    store.cancel_prepared(write);
                    return Err(error.to_string());
                }
                Ok(write)
            });
        let write = match write {
            Ok(write) => write,
            Err(error) => {
                store.cancel_batch(prepared);
                return Err(format!("lanes {:?}: {error}", group.lanes));
            }
        };
        // The first lane writes the state; the others read what it held, or
        // the zeros it starts from.
        for (at, &lane) in group.lanes.iter().enumerate() {
            match (at, write.state(), held) {
                (_, Some(state), None) if at > 0 => {
                    out.slot_ids[lane] = state.slot.0;
                    out.slot_flags[lane] = crate::engine::RS_FLAG_RESET;
                }
                (0, Some(state), _) => {
                    out.slot_ids[lane] = state.slot.0;
                    if state.reset {
                        out.slot_flags[lane] = crate::engine::RS_FLAG_RESET;
                    }
                    if let Some(src) = state.copy_from {
                        out.copies.0.push(src.0);
                        out.copies.1.push(state.slot.0);
                    }
                }
                (_, _, Some(slot)) => out.slot_ids[lane] = slot.0,
                (_, _, None) => {}
            }
        }
        for target in write.buffer_targets() {
            out.buffer_slot_ids.push(target.dst().0);
            if let crate::store::rs::write::RsBufferTarget::Cow { src, dst, .. } = *target {
                out.page_copies.0.push(store.engine_id(src));
                out.page_copies.1.push(store.engine_id(dst));
            }
        }
        prepared.push(write);
    }
    let (published, folds) = store
        .publish_batch(prepared)
        .map_err(|error| error.to_string())?;
    // The lanes read the buffer as the fire found it, before what it
    // folds on publish drops the pages behind it.
    for group in groups {
        if group.writes.is_none() {
            continue;
        }
        let ws = working_sets[group.lanes[0]];
        let read = (|| {
            Ok::<_, crate::store::rs::RsError>((
                store.buffer_translation(ws)?,
                store.base(ws)?,
                store.frozen(ws)?,
            ))
        })();
        let (pages, base, folded) = match read {
            Ok(read) => read,
            Err(error) => {
                store.commit_folds(folds);
                store.settle(published);
                return Err(error.to_string());
            }
        };
        for (at, &lane) in group.lanes.iter().enumerate() {
            let fold = match (at, group.fold) {
                (0, GroupFold::Host(count)) => engine::fire::FoldLen::Host(count),
                (0, GroupFold::Device) => engine::fire::FoldLen::Device,
                _ => engine::fire::FoldLen::Host(0),
            };
            out.verbs[lane] = engine::fire::RsVerb::Buffer {
                pages: pages.clone(),
                base,
                folded,
                fold,
            };
        }
    }
    note_reads(store, &published, &out);
    store.commit_folds(folds);
    out.snapshots = published.snapshots().to_vec();
    out.txn = Some(RsTxn { published });
    Ok(out)
}

#[cfg(test)]
pub fn prepare(store: &mut RsStore, ws: RsWorkingSetId, rows: u32) -> Result<PreparedRs, String> {
    let plan = RsPlan::Groups(vec![GroupPlan {
        lanes: vec![0],
        writes: None,
        fold: GroupFold::Host(rows),
        snapshots: Vec::new(),
    }]);
    prepare_many(store, &[ws], &plan, None)
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
        }
    }

    fn warm(store: &mut RsStore, rows: u32) -> RsWorkingSetId {
        let ws = store.create_working_set(geom());
        let prepared = prepare(store, ws, rows).unwrap();
        settle(store, prepared.txn);
        ws
    }

    fn group(lanes: Vec<usize>, writes: std::ops::Range<u32>, fold: GroupFold) -> RsPlan {
        RsPlan::Groups(vec![GroupPlan {
            lanes,
            writes: Some(writes),
            fold,
            snapshots: Vec::new(),
        }])
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

    #[test]
    fn rs_every_case() {
        first_fire_resets_then_continues_in_place();
        rows_past_the_state_are_buffered_at_their_slots();
        a_host_fold_freezes_when_the_fire_publishes();
        lanes_sharing_a_working_set_read_the_state_one_of_them_folds();
        a_device_fold_lowers_to_its_port_and_waits_for_its_report();
        demand_counts_the_state_copy_and_the_pages();
        a_forked_state_lowers_its_copy_as_the_lane_row();
    }

    fn first_fire_resets_then_continues_in_place() {
        let mut store = RsStore::new(4, 4, 0, 0);
        let ws = store.create_working_set(geom());

        let out = prepare(&mut store, ws, 6).unwrap();
        assert_eq!(out.slot_ids.len(), 1);
        assert_eq!(out.slot_flags, vec![crate::engine::RS_FLAG_RESET]);
        assert!(out.copies.0.is_empty() && out.buffer_slot_ids.is_empty());
        assert_eq!(lowered(&out, 1)[0].0, RsVerb::Fold);
        settle(&mut store, out.txn);
        let slot = store.folded_slot(ws).unwrap().unwrap();
        assert_eq!(store.frozen(ws), Ok(6));

        let out = prepare(&mut store, ws, 1).unwrap();
        assert_eq!(out.slot_ids, vec![slot.0]);
        assert_eq!(out.slot_flags, vec![0]);
        settle(&mut store, out.txn);
        assert_eq!(store.frozen(ws), Ok(7));
    }

    fn rows_past_the_state_are_buffered_at_their_slots() {
        let mut store = RsStore::new(8, 8, 0, 0);
        let ws = warm(&mut store, 5);
        let out = prepare_many(
            &mut store,
            &[ws],
            &group(vec![0], 7..10, GroupFold::None),
            None,
        )
        .unwrap();
        assert_eq!(
            out.buffer_slot_ids.len(),
            2,
            "slots 7..10 fall in pages 5..9 and 9..13"
        );
        let RsVerb::Buffer {
            pages,
            base,
            folded,
            fold,
        } = &lowered(&out, 1)[0].0
        else {
            panic!("a buffering lane");
        };
        assert_eq!((*base, *folded, *fold), (5, 5, FoldLen::Host(0)));
        assert_eq!(pages.len(), 2);
        settle(&mut store, out.txn);
        assert_eq!(store.frozen(ws), Ok(5), "nothing folded");
    }

    fn a_host_fold_freezes_when_the_fire_publishes() {
        let mut store = RsStore::new(8, 8, 0, 0);
        let ws = warm(&mut store, 4);
        let out = prepare_many(
            &mut store,
            &[ws],
            &group(vec![0], 4..12, GroupFold::Host(5)),
            None,
        )
        .unwrap();
        let RsVerb::Buffer {
            base, folded, fold, ..
        } = &lowered(&out, 1)[0].0
        else {
            panic!("a buffering lane");
        };
        assert_eq!(
            (*base, *folded, *fold),
            (4, 4, FoldLen::Host(5)),
            "the lane reads the buffer as the fire found it"
        );
        assert_eq!((store.frozen(ws), store.base(ws)), (Ok(9), Ok(8)));
        assert_eq!(store.buffer_size(ws), Ok(1), "the page of slots 4..8 went");
        settle(&mut store, out.txn);
    }

    fn lanes_sharing_a_working_set_read_the_state_one_of_them_folds() {
        let mut store = RsStore::new(8, 8, 0, 0);
        let ws = warm(&mut store, 4);
        let held = store.folded_slot(ws).unwrap().unwrap();
        let out = prepare_many(
            &mut store,
            &[ws, ws],
            &group(vec![0, 1], 4..6, GroupFold::Host(1)),
            None,
        )
        .unwrap();
        assert_ne!(out.slot_ids[0], held.0, "the folding lane writes a copy");
        assert_eq!(out.copies.0.first(), Some(&held.0));
        assert_eq!(
            out.slot_ids[1], held.0,
            "the other lane reads the state it held"
        );
        let verbs = lowered(&out, 2);
        assert!(matches!(
            verbs[0].0,
            RsVerb::Buffer {
                fold: FoldLen::Host(1),
                ..
            }
        ));
        assert!(matches!(
            verbs[1].0,
            RsVerb::Buffer {
                fold: FoldLen::Host(0),
                ..
            }
        ));
        settle(&mut store, out.txn);
        assert_eq!(
            store.folded_slot(ws).unwrap().map(|slot| slot.0),
            Some(out.slot_ids[0])
        );
    }

    fn a_device_fold_lowers_to_its_port_and_waits_for_its_report() {
        let mut store = RsStore::new(8, 8, 0, 0);
        let ws = warm(&mut store, 4);
        let out = prepare_many(
            &mut store,
            &[ws],
            &group(vec![0], 4..8, GroupFold::Device),
            None,
        )
        .unwrap();
        assert!(matches!(
            lowered(&out, 1)[0].0,
            RsVerb::Buffer {
                fold: FoldLen::Device,
                ..
            }
        ));
        assert_eq!(
            store.frozen(ws),
            Ok(4),
            "frozen once the engine reports the count"
        );
        settle(&mut store, out.txn);
        store.fold(ws, 3);
        assert_eq!(store.frozen(ws), Ok(7));
    }

    fn demand_counts_the_state_copy_and_the_pages() {
        let mut store = RsStore::new(8, 8, 0, 0);
        let ws = warm(&mut store, 4);
        let shared = group(vec![0, 1], 4..12, GroupFold::Host(2));
        assert_eq!(
            demand(&store, &[ws, ws], &shared),
            Ok(RsUnits { slots: 1, pages: 2 }),
            "a state copy and two pages"
        );
        let alone = group(vec![0], 4..6, GroupFold::None);
        assert_eq!(
            demand(&store, &[ws], &alone),
            Ok(RsUnits { slots: 0, pages: 1 }),
            "one page, the state untouched"
        );
    }

    fn a_forked_state_lowers_its_copy_as_the_lane_row() {
        let mut store = RsStore::new(16, 16, 0, 0);
        let parent = warm(&mut store, 2);
        let child = store.fork(parent).unwrap();

        let write = prepare(&mut store, child, 1).unwrap();
        let copy = write.copies.1[0];
        let mut req = request(1);
        req.lanes[0].slot = 9;
        write.apply_to(&mut req);
        assert_eq!(req.lanes[0].rs_slot, Some(copy), "not the kv seat");
        settle(&mut store, write.txn);
    }
}
