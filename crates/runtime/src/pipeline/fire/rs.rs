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
    /// Recurrent state, and windowed layers an engine serves as ring +
    /// buffer (CONTRACT-BC 19, 21): lanes sharing a set (a beam group)
    /// share its folded window and fold it once.
    Groups(Vec<GroupPlan>),
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
    /// Whether its lanes are a beam group's (`parents`): each beam replays
    /// every buffered slot through its own page list, which the engine's
    /// head step resolves (`BEAM-PAGE-LISTS.md` §11), not the slots a mask
    /// admits of one shared list.
    pub beams: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupFold {
    None,
    Host(u32),
    Device,
}

/// What a fire writes of a working set's state: whether the state, to a
/// fresh slot (other lanes read the one it holds), the count it freezes on
/// publish, and the buffered slots: the rs list's (or a store's one
/// buffer) and a hybrid's window buffer's ([`RsStore::split_writes`]).
type GroupWrite = (
    bool,
    bool,
    Option<u32>,
    Option<std::ops::Range<u32>>,
    Option<std::ops::Range<u32>>,
);

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
        // A window buffer holds no row a lone lane folds itself.
        let (writes, wwrites) = store
            .split_writes(ws, self.lanes.len() == 1, fold, self.writes.clone())
            .map_err(|error| error.to_string())?;
        Ok((write_state, fresh, fold, writes, wwrites))
    }
}

pub fn demand(
    store: &RsStore,
    working_sets: &[RsWorkingSetId],
    plan: &RsPlan,
) -> Result<RsUnits, String> {
    let mut total = RsUnits::default();
    match plan {
        RsPlan::Groups(groups) => {
            for group in groups {
                let ws = working_sets[group.lanes[0]];
                let (write_state, fresh, _, writes, wwrites) = group.write(store, ws)?;
                total = total.add(
                    store
                        .group_demand_split(ws, write_state, fresh, writes, wwrites)
                        .map_err(|e| e.to_string())?,
                );
                total.slots += group.snapshots.len() as u32;
            }
        }
    }
    Ok(total)
}

/// What `plan` asks of the state of the lanes `lanes` alone: the groups it
/// leads with one of them ([`demand`] of the lanes one KV working set
/// carries).
pub fn demand_of_lanes(
    store: &RsStore,
    working_sets: &[RsWorkingSetId],
    plan: &RsPlan,
    lanes: &[usize],
) -> Result<RsUnits, String> {
    let mut total = RsUnits::default();
    match plan {
        RsPlan::Groups(groups) => {
            for group in groups.iter().filter(|group| lanes.contains(&group.lanes[0])) {
                let ws = working_sets[group.lanes[0]];
                let (write_state, fresh, _, writes, wwrites) = group.write(store, ws)?;
                total = total.add(
                    store
                        .group_demand_split(ws, write_state, fresh, writes, wwrites)
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
    /// The lane each of `page_copies` is stated on in a shared pool: its
    /// group's first ([`apply_shared`] lowers them to
    /// `Lane::slot_copies` of the rs list).
    pub page_copy_lanes: Vec<usize>,
    pub buffer_slot_ids: Vec<u32>,
    pub verbs: Vec<engine::fire::RsVerb>,
    /// Each lane's window-buffer table as the fire found it and the row
    /// its entry 0 holds, where the store buffers windowed layers' rows
    /// (CONTRACT-BC 19, 21); `None` for a lane that buffers nothing.
    pub window_tables: Vec<Option<(u32, Vec<u32>)>>,
    /// The window buffer is a pool beside the rs list (a hybrid): its tail
    /// copies are `wpage_copies`, `(lane, src, dst)` slot refs, apart from
    /// the rs list's `page_copies`.
    pub window_buffer_apart: bool,
    pub wpage_copies: Vec<(usize, u32, u32)>,
    pub snapshots: Vec<RsSnapshot>,
    pub txn: Option<RsTxn>,
    /// The state slots are a shared pool's records: a lane names them by
    /// `Lane::fixed`, and a copy is the engine's `[run, cow]`, not a
    /// `StateCopy`.
    pub shared: bool,
    /// Lanes whose `slot_ids` name a record their working set writes or
    /// holds.
    pub bound: Vec<bool>,
}

impl PreparedRs {
    /// Lane `row`'s `[run, cow]` records: the one its working set held and
    /// the one the fire writes, the same where it writes in place.
    #[must_use]
    pub fn run_cow(&self, row: usize) -> Option<(u32, u32)> {
        if !self.bound.get(row).copied().unwrap_or(false) {
            return None;
        }
        let slot = *self.slot_ids.get(row)?;
        let run = self
            .copies
            .1
            .iter()
            .position(|&dst| dst == slot)
            .map_or(slot, |at| self.copies.0[at]);
        Some((run, slot))
    }

    pub fn state_copy(&self) -> Option<engine::StateCopy> {
        if self.shared {
            // The engine copies each lane's run onto its cow itself, and
            // the rs list slots a lane's `slot_copies` state.
            return None;
        }
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
        };
        (!copy.moves.is_empty() || !copy.pages.is_empty()).then_some(copy)
    }

    pub fn apply_to(&self, request: &mut crate::engine::FireRequest) {
        for (row, lane) in request.lanes.iter_mut().enumerate() {
            if let Some(verb) = self.verbs.get(row) {
                lane.rs = verb.clone();
                // Its replay as the runtime counts it, from the verb it
                // sends (what the engine's submit cross-check compares): a
                // lane sharing its set whose mask the device resolves is no
                // one run as far as the host can say.
                lane.lane.replay_stated =
                    engine::fire::LaneReplay::of(verb, lane.lane.mask.as_ref(), lane.lane.kv.held)
                        .map(|mut stated| {
                            if lane.lane.mask.is_none()
                                && matches!(
                                    verb,
                                    engine::fire::RsVerb::Buffer {
                                        lineage: engine::fire::Lineage::Admitted,
                                        ..
                                    }
                                )
                            {
                                stated.one_run = false;
                            }
                            stated
                        });
            }
            if let Some(&flags) = self.slot_flags.get(row) {
                lane.rs_reset = if flags & crate::engine::RS_FLAG_RESET != 0 {
                    engine::fire::RsReset::Fresh
                } else {
                    engine::fire::RsReset::Held
                };
            }
        }
        if let [lane] = request.lanes.as_mut_slice() {
            lane.rs_snapshots = self
                .snapshots
                .iter()
                .map(|snap| engine::fire::RsSnapshot {
                    row: snap.row,
                    slot: snap.slot.0,
                    pieces: Vec::new(),
                    fixed: Vec::new(),
                })
                .collect();
        }
    }
}

/// Names each lane's slots by the shared pool's shapes, for an engine that
/// states one (`Lane::tables`, `Lane::fixed`, and the rs list slots a
/// group's first lane copies before the fire, `Lane::slot_copies`, where it
/// writes buffered rows into a slot it shares); nothing where `shared` is
/// `None`. The lane's other fields stay as they are. A lane's list table is its
/// translation where one is stated, else its pages (the list shape's slot
/// refs either way); its fixed records the
/// `[run, cow]` [`PreparedRs::run_cow`] names, a snapshot the slot ref of its
/// record's first piece and, where the record takes more than one, every
/// piece.
pub fn apply_shared(
    req: &mut crate::engine::FireRequest,
    shared: Option<&crate::store::shared::Shared>,
    rs: Option<&PreparedRs>,
) {
    let Some(shared) = shared else {
        return;
    };
    let pool = shared.lock().unwrap();
    for (row, lane) in req.lanes.iter_mut().enumerate() {
        lane.lane.list = if lane.translation.is_empty() {
            ::engine::ListKind::Pages
        } else {
            ::engine::ListKind::Translation
        };
        let list = if lane.translation.is_empty() {
            &lane.pages
        } else {
            &lane.translation
        };
        lane.tables = pool.lane_tables(list);
        let records = rs.filter(|rs| rs.shared).and_then(|rs| rs.run_cow(row));
        // A load that holds state states every lane's record: its pieces,
        // or none.
        lane.record = match records {
            Some((run, cow)) => ::engine::LaneRecord::Pieces(pool.lane_fixed(run, cow)),
            None if pool.holds_records() => ::engine::LaneRecord::Null,
            None => ::engine::LaneRecord::Unstated,
        };
        lane.slot_copies = rs
            .filter(|rs| rs.shared)
            .zip(pool.buffer_list())
            .map(|(rs, list)| {
                let pairs: Vec<[u32; 2]> = rs
                    .page_copy_lanes
                    .iter()
                    .zip(rs.page_copies.0.iter().zip(&rs.page_copies.1))
                    .filter(|&(&at, _)| at == row)
                    .map(|(_, (&src, &dst))| [src, dst])
                    .collect();
                (list.id, pairs)
            })
            .filter(|(_, pairs)| !pairs.is_empty())
            .into_iter()
            .chain(
                // A hybrid's window-buffer tail copies, under its shape.
                rs.filter(|rs| rs.shared && rs.window_buffer_apart)
                    .zip(pool.window_buffer())
                    .map(|(rs, list)| {
                        let pairs: Vec<[u32; 2]> = rs
                            .wpage_copies
                            .iter()
                            .filter(|&&(at, ..)| at == row)
                            .map(|&(_, src, dst)| [src, dst])
                            .collect();
                        (list.id, pairs)
                    })
                    .filter(|(_, pairs)| !pairs.is_empty()),
            )
            .collect();
        if records.is_some() {
            for snap in &mut lane.rs_snapshots {
                snap.pieces = pool.snapshot_pieces(snap.slot);
                snap.fixed = pool.record_refs(snap.slot);
                snap.slot = pool.snapshot_ref(snap.slot);
            }
        }
        if let Some(buffer) = pool.window_buffer() {
            let table = rs.and_then(|rs| rs.window_tables.get(row).cloned().flatten());
            lower_window(&mut lane.lane, buffer, table, pool.rs_list().is_some());
        }
    }
}

/// A lane of a load whose engine serves windowed layers as ring + buffer
/// (CONTRACT-BC 19, 21): the rs store keeps the set's folded window in its
/// record and its unfolded window rows in the window buffer's slots (the
/// pages of a store of windowed layers alone, a pool beside the rs list on
/// a hybrid, [`crate::store::rs::RsStore::shared`]). `table` is the lane's
/// window-buffer table as the fire found it and the row its entry 0 holds
/// ([`PreparedRs::window_tables`]): stated in `Lane::tables`, and with the
/// lane's `fold` and `lineage` (those of its recurrent verb, one fold for
/// both) its `Lane::window`. A lane whose table is empty buffers nothing:
/// its own folded rows' K/V live in the fire, and its base is the folded
/// index after the fire rounded down. A lane of no recurrent layer
/// (`recurrent` false) carries no recurrent verb; a hybrid's keeps its
/// own. A kv-less lane states no window.
fn lower_window(
    lane: &mut engine::Lane,
    buffer: crate::store::shared::TokenShape,
    table: Option<(u32, Vec<u32>)>,
    recurrent: bool,
) {
    if lane.kv_less {
        lane.window = None;
        return;
    }
    let tokens = buffer.tokens.max(1);
    let (fold, lineage) = match &lane.rs {
        engine::fire::RsVerb::Buffer { fold, lineage, .. } => (*fold, *lineage),
        // Folds every row it writes, from where its set is folded.
        engine::fire::RsVerb::Fold => (
            engine::fire::FoldLen::Host(u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX)),
            engine::fire::Lineage::Set,
        ),
    };
    let after = match fold {
        engine::fire::FoldLen::Host(rows) => lane.kv.folded.saturating_add(rows),
        engine::fire::FoldLen::Device => lane.kv.folded,
    };
    let base = match table {
        Some((base, pages)) if !pages.is_empty() => {
            lane.tables.push((buffer.id, pages));
            base
        }
        // A lane carrying rows earlier fires buffered starts no later than
        // them; one folding all it computes, at the folded index after it.
        _ if lane.kv.held > lane.kv.folded => lane.kv.folded - lane.kv.folded % tokens,
        _ => after - after % tokens,
    };
    lane.window = Some(engine::fire::WindowFold {
        base,
        fold,
        lineage,
    });
    if !recurrent {
        lane.rs = engine::fire::RsVerb::Fold;
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
    let window_buffer: Vec<RsSlotId> = if out.window_buffer_apart {
        out.window_tables
            .iter()
            .flatten()
            .flat_map(|(_, table)| table.iter().copied())
            .chain(out.wpage_copies.iter().map(|&(_, src, _)| src))
            .filter(live)
            .map(|slot| store.wpage_id(slot))
            .collect()
    } else {
        Vec::new()
    };
    let pages: Vec<RsSlotId> = out
        .page_copies
        .0
        .iter()
        .copied()
        .chain(verb_pages)
        .filter(live)
        .map(|page| store.page_id(page))
        .chain(window_buffer)
        .collect();
    store.note_reads(seq, slots.into_iter().chain(pages));
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
    let RsPlan::Groups(groups) = plan;
    // The beam groups' sets: their folds keep their buffers (§11.3).
    let beam_sets: Vec<RsWorkingSetId> = groups
        .iter()
        .filter(|group| group.beams)
        .map(|group| working_sets[group.lanes[0]])
        .collect();
    let lanes = working_sets.len();
    let mut out = PreparedRs {
        verbs: vec![engine::fire::RsVerb::Fold; lanes],
        slot_ids: vec![0; lanes],
        slot_flags: vec![0; lanes],
        shared: store.is_shared(),
        bound: vec![false; lanes],
        window_tables: vec![None; lanes],
        window_buffer_apart: store.window_buffer_apart(),
        ..PreparedRs::default()
    };
    let mut prepared: Vec<RsPreparedWrite> = Vec::with_capacity(groups.len());
    for group in groups {
        let ws = working_sets[group.lanes[0]];
        let held = store.folded_slot(ws).map_err(|error| error.to_string())?;
        let write = group
            .write(store, ws)
            .and_then(|(write_state, fresh, fold, writes, wwrites)| {
                let mut write = store
                    .prepare_group_split(
                        ws,
                        write_state,
                        fresh,
                        fold,
                        writes,
                        wwrites,
                        granted.as_deref_mut(),
                    )
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
        // the zeros it starts from. In a shared pool a group whose first
        // lane folds writes a fresh record, and every lane names it,
        // `[run, cow]` as the first does (CONTRACT-BC 15), as does a beam
        // group's; the buffered rows are rs list slots apart from the record.
        for (at, &lane) in group.lanes.iter().enumerate() {
            out.bound[lane] = true;
            match (at, write.state(), held) {
                (_, Some(state), None) if at > 0 => {
                    out.slot_ids[lane] = state.slot.0;
                    out.slot_flags[lane] = crate::engine::RS_FLAG_RESET;
                }
                // A beam group's lanes all read the state its fold wrote
                // (`BEAM-PAGE-LISTS.md` §11.3), past which each replays.
                (_, Some(state), Some(_)) if at > 0 && (out.shared || group.beams) => {
                    out.slot_ids[lane] = state.slot.0;
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
                (_, _, None) => out.bound[lane] = false,
            }
        }
        for target in write.buffer_targets() {
            out.buffer_slot_ids.push(target.dst().0);
            if let crate::store::rs::write::RsBufferTarget::Cow { src, dst, .. } = *target {
                if store.is_window_page(dst) && out.window_buffer_apart {
                    out.wpage_copies
                        .push((group.lanes[0], store.engine_id(src), store.engine_id(dst)));
                    continue;
                }
                out.page_copies.0.push(store.engine_id(src));
                out.page_copies.1.push(store.engine_id(dst));
                out.page_copy_lanes.push(group.lanes[0]);
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
                store.commit_folds_keeping(folds, &beam_sets);
                store.settle(published);
                return Err(error.to_string());
            }
        };
        // A lane alone on its set replays every buffered slot of it; lanes
        // sharing a set replay the slots their masks admit, the ancestry
        // they state.
        // A beam group's lanes each replay every buffered slot of their own
        // list (the head step resolves it per beam, §11).
        let lineage = if group.lanes.len() > 1 && !group.beams {
            engine::fire::Lineage::Admitted
        } else {
            engine::fire::Lineage::Set
        };
        // The window buffer's table as the fire found it, where the store
        // buffers windowed layers' rows (CONTRACT-BC 19, 21).
        let window = match store.window_table(ws) {
            Ok(window) => window,
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
            out.window_tables[lane] = window.clone();
            out.verbs[lane] = engine::fire::RsVerb::Buffer {
                pages: pages.clone(),
                base,
                folded,
                fold,
                lineage,
            };
        }
    }
    note_reads(store, &published, &out);
    store.commit_folds_keeping(folds, &beam_sets);
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
        beams: false,
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
            beams: false,
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
            .map(|lane| (lane.lane.rs, lane.lane.rs_reset))
            .collect()
    }

    /// A lane as the runtime states it from the verb it sends: a decode lane
    /// of a group sharing its set, folded through 34, its row at 38, its mask
    /// admitting the one buffered slot 34 (a parent's): one row replayed, not
    /// one stretch ending at its row (Metal lays such a fire wide). Alone on
    /// its set it replays all four, one stretch.
    #[test]
    fn a_lane_states_its_replay_from_its_verb() {
        let verb = |lineage| RsVerb::Buffer {
            pages: vec![1],
            base: 32,
            folded: 34,
            fold: FoldLen::Host(0),
            lineage,
        };
        let mask = engine::Masking::Extent(engine::Mask::new(vec![0, 1, 3, 1], 5));
        let beam = engine::fire::LaneReplay::of(&verb(engine::fire::Lineage::Admitted), Some(&mask), 38);
        assert_eq!(
            beam,
            Some(engine::fire::LaneReplay {
                replayed: 1,
                one_run: false
            })
        );
        let alone = engine::fire::LaneReplay::of(&verb(engine::fire::Lineage::Set), None, 38);
        assert_eq!(
            alone,
            Some(engine::fire::LaneReplay {
                replayed: 4,
                one_run: true
            })
        );
        assert_eq!(engine::fire::LaneReplay::of(&RsVerb::Fold, None, 38), None);
        // The runtime lowers it onto the lane it sends, and a lane sharing
        // its set whose mask the device resolves states no one run.
        let mut store = RsStore::new(8, 8);
        let ws = warm(&mut store, 4);
        let out = prepare_many(
            &mut store,
            &[ws, ws],
            &group(vec![0, 1], 4..6, GroupFold::None),
            None,
        )
        .unwrap();
        let mut req = request(2);
        req.lanes[0].lane.kv.held = 4;
        req.lanes[1].lane.kv.held = 5;
        out.apply_to(&mut req);
        assert!(req.lanes.iter().all(|lane| lane.replay_stated.is_some_and(|s| !s.one_run)));
        settle(&mut store, out.txn);
    }

    /// The engine's door: the runtime's stated count against the engine's,
    /// a disagreement where the runtime counted less (priced compact what
    /// the engine lays wide, or found held what it refuses) named with both;
    /// the runtime counting more is no bug.
    #[test]
    fn the_engine_s_door_names_a_count_the_runtime_got_wrong() {
        let rung = engine::RungReplay {
            own_max: Some(256),
            carried_max: Some(256),
            scratch: engine::ScratchBytes {
                slot_bytes: vec![1],
                lane_bytes: 0,
                room_bytes: 300,
                pad: 1,
            },
        };
        let stated = |one_run| {
            (0..10).map(move |_| (1, Some(engine::fire::LaneReplay { replayed: 37, one_run })))
        };
        let wide = engine::ReplayRows::of_lanes((0..10).map(|_| (1, 37)), false);
        // The runtime priced it compact, the engine lays it wide: a bug.
        let why = engine::fire::replay_agrees(stated(true), &rung, wide, false, true, true, false)
            .unwrap_err();
        assert!(why.contains("disagree"), "{why}");
        // Both wide: they agree (and both refuse it).
        assert!(engine::fire::replay_agrees(stated(false), &rung, wide, false, true, true, false).is_ok());
        // The runtime wider than the engine: no bug.
        let narrow = engine::ReplayRows::of_lanes((0..10).map(|_| (1, 37)), true);
        assert!(engine::fire::replay_agrees(stated(false), &rung, narrow, true, true, true, false).is_ok());
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
        let mut store = RsStore::new(4, 4);
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
        let mut store = RsStore::new(8, 8);
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
            ..
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
        let mut store = RsStore::new(8, 8);
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
        let mut store = RsStore::new(8, 8);
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
        let mut store = RsStore::new(8, 8);
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
        let mut store = RsStore::new(8, 8);
        let ws = warm(&mut store, 4);
        let shared = group(vec![0, 1], 4..12, GroupFold::Host(2));
        assert_eq!(
            demand(&store, &[ws, ws], &shared),
            Ok(RsUnits {
                slots: 1,
                pages: 2,
                ..RsUnits::default()
            }),
            "a state copy and two pages"
        );
        let alone = group(vec![0], 4..6, GroupFold::None);
        assert_eq!(
            demand(&store, &[ws], &alone),
            Ok(RsUnits {
                slots: 0,
                pages: 1,
                ..RsUnits::default()
            }),
            "one page, the state untouched"
        );
    }

    fn a_forked_state_lowers_its_copy_as_the_lane_row() {
        let mut store = RsStore::new(16, 16);
        let parent = warm(&mut store, 2);
        let child = store.fork(parent).unwrap();

        let write = prepare(&mut store, child, 1).unwrap();
        let copy = write.copies.1[0];
        let mut req = request(1);
        req.lanes[0].slot = 9;
        write.apply_to(&mut req);
        assert_eq!(write.slot_ids[0], copy, "not the kv seat");
        settle(&mut store, write.txn);
    }
}
