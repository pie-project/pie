//! Submitting a pass or a frame: its lanes, folds and pieces, and closing the pipeline.

use super::*;

pub(super) type Anyhow<T> = anyhow::Result<T>;

pub(super) struct PassLanes {
    pub(super) places: Vec<geometry::Place>,
    pub(super) cursors: Vec<std::sync::Arc<std::sync::Mutex<crate::inferlet::host::working_set::Cursor>>>,
    pub(super) kv: Vec<KvWorkingSet>,
}

pub(super) fn pass_lanes<C: FireContext>(ctx: &mut C, fwd: &Resource<ForwardPass>) -> Anyhow<PassLanes> {
    use crate::inferlet::host::working_set::WorkingSet;
    let (lanes, kv_lanes) = {
        let pass = ctx.resources().get(fwd)?;
        (pass.lanes.clone(), pass.kv_lanes.clone())
    };
    let mut kv = Vec::with_capacity(kv_lanes.len());
    for &rep in &kv_lanes {
        let res: Resource<KvWorkingSet> = Resource::new_borrow(rep);
        kv.push(ctx.resources().get(&res)?.clone());
    }
    // What each lane's working set has folded, and whether that is all of
    // it, are its KV working set's.
    let folded: Vec<(u32, bool)> = match kv.first() {
        Some(ws) => {
            let stores = crate::store::registry::get(ws.model, ws.engine);
            crate::store::registry::with_kv_lock(&stores.kv, "host-other", |store| {
                kv.iter()
                    .map(|ws| {
                        let folded = u32::try_from(store.folded(ws.id)?).unwrap_or(u32::MAX);
                        Ok((folded, store.folded_exact(ws.id)?))
                    })
                    .collect::<Result<_, crate::store::kv::KvStoreError>>()
            })?
        }
        None => Vec::new(),
    };
    let mut places = Vec::with_capacity(lanes.len());
    let mut cursors = Vec::with_capacity(lanes.len());
    for (lane, &rep) in lanes.iter().enumerate() {
        let res: Resource<WorkingSet> = Resource::new_borrow(rep);
        let cursor = ctx.resources().get(&res)?.cursor.clone();
        let now = cursor.lock().unwrap().clone();
        let (folded, folded_exact) = folded.get(lane).copied().unwrap_or((0, true));
        places.push(geometry::Place {
            folded,
            folded_exact,
            tail: now.tail,
            shared: lanes.iter().filter(|&&other| other == rep).count() > 1,
            readable: now.readable(),
            readable_known: now.readable_known(),
            set: kv.get(lane).map_or(0, |ws| set_token(ws.id)),
            spans: now.spans,
        });
        cursors.push(cursor);
    }
    Ok(PassLanes {
        places,
        cursors,
        kv,
    })
}

pub(super) type CursorCell = std::sync::Arc<std::sync::Mutex<crate::inferlet::host::working_set::Cursor>>;

/// The token a working set's lanes name it by to the engine
/// ([`::engine::KvDelta::set`]): never 0, which names no set.
pub(super) fn set_token(ws: crate::store::kv::page_table::WorkingSetId) -> u64 {
    ws.to_bits().wrapping_add(1)
}

/// A fire on its way moves its working set's cursor, as soon as it is
/// submitted: a pass that stops short leaves the cursor at the fires that
/// went. `tail` is where its rows end, `None` where the device places them
/// past rows the host cannot place; `spans` the first slot it writes and
/// the lags of the media runs it carries. What it folds is not the
/// cursor's: that is its KV working set's.
pub(super) fn advance_cursor(
    cell: &CursorCell,
    tail: Option<u32>,
    spans: Option<(u32, Vec<(u32, u32, u32)>)>,
) {
    let mut cursor = cell.lock().unwrap();
    cursor.tail = tail;
    if let Some(end) = tail {
        cursor.reserved = cursor.reserved.max(end);
        cursor.written = cursor.written.max(end);
    }
    if let Some((start, spans)) = spans {
        cursor.rewrite(start, spans);
    }
}

/// A fire just submitted lands host-placed rows `start..end` on the set:
/// a fire staged after it reads them. Should it fail, they may never have
/// landed, and nothing from `start` on counts as written.
pub(super) fn host_wrote(
    cell: &CursorCell,
    fire: &crate::engine::completion::WorkItemCompletion,
    start: u32,
    end: u32,
) {
    cell.lock().unwrap().wrote(end);
    let cell = std::sync::Arc::downgrade(cell);
    fire.on_settled_landed(move |ok| {
        if let (false, Some(cell)) = (ok, cell.upgrade()) {
            cell.lock().unwrap().unwrote(start);
        }
    });
}

/// The lags of the media runs `lanes` carry that start in `slots`.
pub(super) fn lagging_spans(
    landings: &[geometry::Landing],
    matched: &[crate::pipeline::media::MatchedRun],
    lanes: &[usize],
    slots: &std::ops::Range<u64>,
) -> Vec<(u32, u32, u32)> {
    matched
        .iter()
        .filter(|run| lanes.contains(&(run.lane as usize)))
        .map(|run| {
            let at = landings[run.lane as usize].start + run.anchor;
            let lag = run.span.token_count.saturating_sub(run.span.position_span);
            (at, at + run.rows, lag)
        })
        .filter(|&(at, _, lag)| lag > 0 && slots.contains(&u64::from(at)))
        .collect()
}

/// The fire is submitted: each working set it folds by a count placed when
/// it settles says so, its folded length a lower bound until the fire
/// settles ([`settle_fold`]) or, dropped, the engine is done with it. It is
/// not forked or exported until then.
pub(super) fn unsettle(
    stores: &crate::store::registry::Stores,
    completion: &crate::engine::WorkItemCompletion,
    parts: &mut [FireWs],
) {
    let mut settling = parts
        .iter_mut()
        .filter_map(|part| match &mut part.fold {
            Some(FoldAt {
                by: FoldBy::Settled { unplaced, .. },
                ..
            }) => Some((part.ws, unplaced)),
            _ => None,
        })
        .peekable();
    if settling.peek().is_none() {
        return;
    }
    crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        for (ws, unplaced) in settling {
            match kv.unsettle(ws) {
                Ok(()) => {
                    *unplaced = Some(Unplaced {
                        ws,
                        submitted: Some((stores.kv.clone(), completion.clone())),
                    });
                }
                Err(error) => {
                    tracing::debug!(%error, "pipeline: a fold placed at settle on a set gone");
                }
            }
        }
    });
}

/// The fold of a run-ahead fire (its tokens the device's) on sets with a
/// recurrent state or windows: each group's no further than where the rows
/// its program has accepted end ([`Cursor::accepted`], bounded by the
/// run-ahead window), so the rows past them stay buffered and a stop takes
/// them back. (A decode's buffered row costs it about what one fold's
/// state write does, CUDA's gated delta on Qwen3.5-0.8B at 64 lanes: a fold
/// takes every accepted row at once.) A fold the device counts, or over a
/// folded length the host does not know, stays as stated.
///
/// [`Cursor::accepted`]: crate::inferlet::host::working_set::Cursor::accepted
pub(super) fn lagged_fold(
    fold: &geometry::Fold,
    lanes: &PassLanes,
    groups: &LaneGroups,
    landings: &[geometry::Landing],
) -> geometry::Fold {
    let count = landings.len();
    if (0..count).any(|lane| fold.of(lane).is_none()) {
        return fold.clone();
    }
    let bound = crate::scheduler::run_ahead_window();
    let mut counts: Vec<u32> = (0..count).map(|lane| fold.of(lane).unwrap_or(0)).collect();
    for (_, group) in groups {
        let first = group[0];
        let place = &lanes.places[first];
        if !place.folded_exact {
            continue;
        }
        let accepted = lanes.cursors[first]
            .lock()
            .unwrap()
            .accepted(landings[first].start, bound);
        let room = accepted.saturating_sub(place.folded);
        for &lane in group {
            counts[lane] = counts[lane].min(room);
        }
    }
    geometry::Fold::Host(counts)
}

/// What a host-placed fire folds, `fold` as its pass states it, on sets
/// that hold unfolded rows (`lags`: a recurrent state or windows). A
/// run-ahead fire (`run_ahead`, its tokens the device's) folds no further
/// than its program has accepted ([`lagged_fold`]), unless its pass stated
/// its rows final (CONTRACT-BC 25): then, as a fire whose tokens the host
/// places, it folds what it states at fire and accepts every row its sets
/// hold, so its next fire carries nothing in.
pub(super) fn pass_fold(
    fold: geometry::Fold,
    lags: bool,
    run_ahead: bool,
    final_rows: bool,
    lanes: &PassLanes,
    groups: &LaneGroups,
    landings: &[geometry::Landing],
) -> geometry::Fold {
    if !lags {
        return fold;
    }
    if run_ahead && !final_rows {
        return lagged_fold(&fold, lanes, groups, landings);
    }
    for cursor in &lanes.cursors {
        cursor.lock().unwrap().accept_all();
    }
    fold
}

/// The outputs a fire just reserved its moves for publishes to the host:
/// the program accepts the fire's rows as it takes them.
pub(super) fn host_reads(
    cells: &BoundCells,
    accesses: &[(bool, bool)],
) -> Vec<crate::inferlet::host::working_set::HostRead> {
    cells
        .iter()
        .zip(accesses)
        .filter(|(_, (_, publish))| *publish)
        .filter_map(|(cell, _)| {
            let c = cell.lock().unwrap();
            if c.role != Some(HostRole::Reader) {
                return None;
            }
            Some(crate::inferlet::host::working_set::HostRead {
                cell: Arc::downgrade(cell),
                index: c.last_published()?,
            })
        })
        .collect()
}

/// What a fire whose rows the host places folds on `group`'s working set.
/// A count the host knows over a folded length it knows is a slot, frozen
/// as the fire is prepared; anything else is placed when the pass's last
/// fire settles. A fold reaches no further than where the group's first
/// lane lands its rows (`landings`), as its recurrent state's does
/// ([`host_rs_plan`]), nor than `end`, where this fire's rows end.
#[allow(
    clippy::too_many_arguments,
    reason = "the fold, where it ends, and the request whose lanes report it"
)]
pub(super) fn fold_at(
    lanes: &PassLanes,
    group: &[usize],
    landings: &[geometry::Landing],
    fold: &geometry::Fold,
    end: u64,
    last: bool,
    rs: Option<crate::store::rs::RsWorkingSetId>,
    req: &mut crate::engine::FireRequest,
    tags: &mut Vec<ReportTag>,
) -> Option<FoldBy> {
    if matches!(fold, geometry::Fold::None) {
        return None;
    }
    let first = *group.first()?;
    let place = &lanes.places[first];
    let reach = u64::from(landings[first].end());
    let count = fold.of(first);
    if count == Some(0) {
        return None;
    }
    if place.folded_exact
        && let Some(count) = count
    {
        return Some(FoldBy::To(
            u64::from(place.folded)
                .saturating_add(u64::from(count))
                .min(reach)
                .min(end),
        ));
    }
    if !last {
        return None;
    }
    let (tags, rs) = match count {
        Some(_) => (Vec::new(), None),
        None => (
            vec![report_on(req, first, tags)],
            rs.map(RsDeviceFold::planned),
        ),
    };
    Some(FoldBy::Settled {
        count,
        tags,
        reach: reach.min(end),
        written: false,
        rs,
        unplaced: None,
    })
}

/// A piece states the boundary its prepare found, which a cut pass that
/// folds as it goes moves on from the one the pass was staged on: the
/// columns of the lane's mask between the two are now committed slots, read
/// as every committed slot is, and the mask starts past them.
pub(super) fn restate_boundary(lane: &mut ::engine::Lane, folded: u64) {
    let folded = u32::try_from(folded).unwrap_or(u32::MAX).min(lane.kv.held);
    let moved = u64::from(folded.saturating_sub(lane.kv.folded));
    if moved > 0 {
        match &mut lane.mask {
            Some(::engine::Masking::Extent(mask)) => *mask = mask.after(moved),
            Some(::engine::Masking::Rows(masks)) => {
                for mask in masks.iter_mut() {
                    *mask = mask.after(moved);
                }
            }
            None => {}
        }
    }
    lane.kv.folded = lane.kv.folded.max(folded);
}

/// The last step of a piece's prepare on a working set it writes from
/// `start`: its log begins there, and what it folds by a count the host
/// knows is frozen through `freeze`. A piece that starts inside what the
/// set has frozen begins no log, and freezes nothing.
pub(super) fn begin_piece(
    kv: &mut crate::store::kv::KvStore,
    ws: crate::store::kv::page_table::WorkingSetId,
    start: u64,
    freeze: Option<u64>,
) -> Result<Option<u64>, crate::store::kv::KvStoreError> {
    if start < kv.folded(ws)? {
        return Ok(None);
    }
    let seq = kv.log_begin(ws, start)?;
    if let Some(to) = freeze {
        kv.freeze(ws, to)?;
    }
    Ok(Some(seq))
}

/// What a pass whose rows the device places folds on a working set, as far
/// as the host can say before the fire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DeviceFold {
    None,
    /// By a count the device computes.
    Device,
    /// This many slots past the folded length.
    Count(u32),
    /// By a count that runs past the lane's rows, from rows the host cannot
    /// place: how far it reaches is where the engine says the rows ended.
    PastRows(u32),
}

/// Whether the rows the device places on a working set start at its folded
/// length: one lane writes it, it holds no unfolded row, and the host knows
/// it.
pub(super) fn appends_folded(lanes: &PassLanes, group: &[usize]) -> bool {
    match group {
        [lane] => {
            let place = &lanes.places[*lane];
            place.folded_exact && place.tail == Some(place.folded)
        }
        _ => false,
    }
}

/// The engine folds through `min(folded + count, first + rows)`, `first`
/// being where the device lands the lane's `rows` rows, at or past the
/// folded length. A count within the rows stops short of them wherever
/// they land; one past them stops at their end, which the host knows only
/// when they start at the folded length. `room` is what the set reserved
/// past it.
pub(super) fn device_fold(count: Option<u32>, rows: u32, room: u32, appends: bool) -> DeviceFold {
    match count {
        None => DeviceFold::Device,
        Some(0) => DeviceFold::None,
        Some(count) if count <= rows || appends => match count.min(rows).min(room) {
            0 => DeviceFold::None,
            count => DeviceFold::Count(count),
        },
        Some(count) => DeviceFold::PastRows(count),
    }
}

/// The refusal of [`DeviceFold::PastRows`] where nothing tells the host
/// where the rows ended: a fire that reports no rows, or a recurrent state,
/// whose fold the host states is final as stated.
pub(super) fn past_rows_refusal(lane: usize, rows: u32) -> String {
    format!(
        "pipeline: lane {lane} folds past the {rows} row(s) it writes, which land where the \
         device says on a working set that may hold unfolded rows before them; that fold \
         ends where the rows do, which the device alone knows here. Fold no more than the \
         lane writes, or fold what the set holds before this pass"
    )
}

/// [`fold_at`] for rows the device places: `stated` is [`device_fold`] for
/// the working set's first lane, `reported` whether the fire reports the
/// rows it wrote.
pub(super) fn device_fold_at(
    lanes: &PassLanes,
    first: usize,
    stated: DeviceFold,
    reported: bool,
    rs: Option<crate::store::rs::RsWorkingSetId>,
    req: &mut crate::engine::FireRequest,
    tags: &mut Vec<ReportTag>,
) -> Option<FoldBy> {
    let place = &lanes.places[first];
    let reach = u64::from(lanes.cursors[first].lock().unwrap().reserved);
    let settled = |count, tags, written, rs| FoldBy::Settled {
        count,
        tags,
        reach,
        written,
        rs,
        unplaced: None,
    };
    match stated {
        DeviceFold::None => None,
        DeviceFold::Device => {
            let tag = report_on(req, first, tags);
            Some(settled(
                None,
                vec![tag],
                false,
                rs.map(RsDeviceFold::planned),
            ))
        }
        DeviceFold::Count(count) if place.folded_exact => Some(FoldBy::To(
            u64::from(place.folded).saturating_add(u64::from(count)),
        )),
        DeviceFold::Count(count) => Some(settled(Some(count), Vec::new(), false, None)),
        DeviceFold::PastRows(count) => {
            reported.then(|| settled(Some(count), Vec::new(), true, None))
        }
    }
}

/// The working sets a piece writes: those with a lane that has rows in it. A set whose lanes
/// all ride only in a later piece (a decode lane's one row beside a prompt cut into pieces) is
/// not written here; its span would be empty, which reads as a write at slot 0, inside what the
/// set already froze, so it takes no lease, transaction or log in this piece.
pub(super) fn written_groups<W>(piece_groups: &[(W, Vec<usize>)]) -> Vec<usize> {
    piece_groups
        .iter()
        .enumerate()
        .filter(|(_, (_, lanes))| !lanes.is_empty())
        .map(|(group, _)| group)
        .collect()
}

pub(super) fn written_span_of(
    req: &crate::engine::FireRequest,
    lanes: &[usize],
    page: u32,
) -> (std::ops::Range<u64>, std::ops::Range<u64>) {
    let page = u64::from(page.max(1));
    let rows = lanes
        .iter()
        .filter_map(|&lane| req.lanes.get(lane))
        .filter(|lane| !lane.tokens.is_empty());
    let start = rows
        .clone()
        .map(|lane| u64::from(lane.kv.held))
        .min()
        .unwrap_or(0);
    let end = rows
        .map(|lane| u64::from(lane.kv.held) + lane.tokens.len() as u64)
        .max()
        .unwrap_or(start);
    (start..end, start / page..end.div_ceil(page))
}

pub(super) async fn open_pipeline<C: FireContext>(
    ctx: &mut C,
    this: &Resource<Pipeline>,
    fwd: &Resource<ForwardPass>,
) -> Anyhow<Result<(PendingFires, PipelineFailure, crate::store::PipelineScope), String>> {
    let (fires, failure, scope) = {
        let pipeline = ctx.resources().get(this)?;
        if pipeline.scope.is_closed() {
            return Ok(Err("pipeline: pipeline is closed".to_string()));
        }
        (
            pipeline.fires.clone(),
            pipeline.failure.clone(),
            pipeline.scope.clone(),
        )
    };
    drain_settled(ctx, Some(&fires)).await?;
    if let Some(error) = pipeline_failed(&failure) {
        return Ok(Err(error));
    }
    if let Err(error) = wire_channels_to_pipeline(ctx, fwd, &fires)? {
        return Ok(Err(error));
    }
    Ok(Ok((fires, failure, scope)))
}

pub(super) fn stamp_lane_facts<C: FireContext>(
    ctx: &mut C,
    fwd: &Resource<ForwardPass>,
    req: &mut crate::engine::FireRequest,
) -> Anyhow<()> {
    let pass = ctx.resources().get(fwd)?;
    crate::pipeline::instance::LaneFacts::stamp(&pass.facts, req);
    Ok(())
}

/// A lane binding no span carries no run of the model's placeholder.
pub(super) fn scan_lane_media(
    lane_tokens: &[&[u32]],
    spans: &[Vec<Arc<media::front::EncodedSpan>>],
) -> Result<Vec<crate::pipeline::media::MatchedRun>, crate::pipeline::media::Refusal> {
    let mut matched = Vec::new();
    for (lane, tokens) in lane_tokens.iter().enumerate() {
        match spans.get(lane).filter(|spans| !spans.is_empty()) {
            Some(spans) => {
                for mut run in crate::pipeline::media::scan(&[tokens], spans)? {
                    run.lane = lane as u32;
                    matched.push(run);
                }
            }
            None => {
                crate::pipeline::media::refuse_orphan_runs(&[tokens], crate::model::media_pad())?
            }
        }
    }
    Ok(matched)
}

pub(super) fn advance_shadow<C: FireContext>(ctx: &mut C, fwd: &Resource<ForwardPass>) -> Anyhow<()> {
    let pass = ctx
        .resources()
        .get_mut(fwd)?
        .bound_mut()
        .map_err(anyhow::Error::msg)?;
    pass.host_shadow
        .advance(&pass.instance.program.bound, &pass.cells);
    Ok(())
}

/// A pass whose rows the device places (a device-geometry or a float
/// lane's): fired whole, its writes recorded once it is submitted.
async fn submit_placed_by_device<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    fwd: Resource<ForwardPass>,
    frame: Option<crate::scheduler::FrameStamp>,
) -> Anyhow<Result<(), String>> {
    let pipeline = this.rep();
    let final_rows = ctx.resources().get(&fwd)?.bindings.final_rows;
    let submitted = if ctx.resources().get(&fwd)?.devgeo.is_some() {
        fire_device_geometry(ctx, this, fwd, frame).await?
    } else {
        float::fire_float_lane(ctx, this, fwd, frame).await?
    };
    if submitted.is_ok() {
        record_writes(ctx, pipeline, None, final_rows)?;
    }
    Ok(submitted)
}

/// The host-placed pass's geometry and attention mask, evaluated from what
/// its shadow knows; the ports in `device_resolved` are the engine's.
/// `Err` refuses the fire by name.
fn host_geometry_of(
    p: &ForwardPass,
    device_resolved: PortMask,
) -> Result<(geometry::ReqGeometry, geometry::FireAttnMask), String> {
    let bound = &p.instance.program.bound;
    let (shadow, shadow_cells) = (&p.host_shadow, &p.cells);
    let mut known = |chan: u32| shadow.fire_value(bound, shadow_cells, chan);
    let (geometry, evaluated) =
        geometry::map_geometry_evaluated_with(bound, &mut known, device_resolved)
            .map_err(|error| format!("pipeline: fire geometry: {error}"))?;
    if device_resolved.is_empty() && geometry.token_ids.contains(&u32::MAX) {
        return Err("pipeline: fire geometry: in-band -1 skip tokens require a \
                    device-resolved geometry class; this fire resolved on the \
                    host wire"
            .to_string());
    }
    // A mask the program computes on the device is the engine's to resolve,
    // even on a fire whose seeds let the host evaluate it: one source a port.
    let attn_mask = if device_resolved.contains(Port::AttnMask) {
        geometry::FireAttnMask::Device
    } else {
        geometry::lower_attn_mask_evaluated(&bound.container, &geometry.qo_indptr, &evaluated)
            .map_err(|error| format!("pipeline: fire attention mask: {error}"))?
    };
    Ok((geometry, attn_mask))
}

/// What a beam pass reserves of its pool before its fire.
struct BeamReserve {
    /// Each group's pages from its folded pages to its column's end;
    /// `None` where the pass's lanes do not state their own page lists.
    paged: Option<Vec<Vec<u64>>>,
    /// Each beam group's pool end (a slot), which its rs buffer is backed
    /// to as well (`BEAM-PAGE-LISTS.md` §11).
    ends: Vec<u32>,
    /// Each beam group's common slot, as its last harvested steps left it
    /// (§11.3): one pass, one group, one instance.
    commons: Vec<u32>,
}

/// Reserves each beam group's pool up to the column this fire opens and
/// backs the pages the column adds; `Err` refuses the pass by name.
#[allow(clippy::too_many_arguments)]
fn reserve_beam_columns(
    binds_pages: bool,
    beam_common: u32,
    page_size: u32,
    groups: &LaneGroups,
    pass_lanes: &PassLanes,
    landings: &[geometry::Landing],
    stores: &crate::store::registry::Stores,
) -> Anyhow<Result<BeamReserve, String>> {
    let mut beam_ends: Vec<u32> = Vec::new();
    let mut beam_commons: Vec<u32> = Vec::new();
    if !binds_pages {
        return Ok(Ok(BeamReserve {
            paged: None,
            ends: beam_ends,
            commons: beam_commons,
        }));
    }
    let page = u64::from(page_size.max(1));
    let mut reserves = Vec::with_capacity(groups.len());
    for (_, lanes) in groups {
        let lane = lanes[0];
        let folded = pass_lanes.places[lane].folded;
        // The beams' pool grows a column at a time (BEAM-PAGE-LISTS.md
        // §3): before the fire that opens a column the set reserves
        // what its column needs, waiting like any reservation. The
        // step hands out the lowest free pages, so every page named or
        // pending lies below `B (c + 2) + B F + 1`, `c` this fire's
        // column and `F` the engine's frames in flight.
        let f0 = u64::from(folded) / page;
        let last = lanes
            .iter()
            .map(|&lane| u64::from(landings[lane].start) + u64::from(landings[lane].rows.max(1)) - 1)
            .max()
            .unwrap_or(u64::from(folded));
        let beams = lanes.len() as u64;
        let frames = crate::scheduler::configured_dispatch_depth().max(1) as u64;
        let column = (last / page).saturating_sub(f0);
        let named = beams * (column + 2) + beams * frames + 1;
        let end = f0 + named;
        let end_slot = u32::try_from(end * page).unwrap_or(u32::MAX);
        let capacity = crate::store::registry::with_kv_lock(
            &stores.kv,
            "host-other",
            |kv| kv.capacity_pages(),
        );
        if let Err(refusal) = beam_pool_fits(lane, end, named, capacity) {
            return Ok(Err(refusal));
        }
        if let Err(refusal) = crate::inferlet::host::working_set::grow_pages(
            &pass_lanes.kv[lane],
            end_slot,
        ) {
            return Ok(Err(format!(
                "pipeline: lane {lane}'s beams' pool cannot grow to its column's {named} \
                 page(s): {refusal}"
            )));
        }
        {
            let mut cursor = pass_lanes.cursors[lane].lock().unwrap();
            cursor.reserved = cursor.reserved.max(end_slot);
        }
        beam_ends.push(end_slot);
        beam_commons.push(beam_common);
        // Only the pages the column adds are backed here: the pool's
        // pages are the group's alone and the device keeps what they
        // hold, so a page an earlier fire still in flight writes is
        // not copied before this one writes it.
        let mapped = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
            kv.mapped_len(pass_lanes.kv[lane].id)
        })
        .map_err(|error| anyhow::anyhow!("pipeline: KV rows: {error}"))?;
        reserves.push((mapped.max(f0)..end).collect());
    }
    Ok(Ok(BeamReserve {
        paged: Some(reserves),
        ends: beam_ends,
        commons: beam_commons,
    }))
}

pub async fn submit_pass_stamped<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    fwd: Resource<ForwardPass>,
    frame: Option<crate::scheduler::FrameStamp>,
    ahead: u32,
) -> Anyhow<Result<(), String>> {
    let submit_probe = crate::scheduler::probe::host_submit();
    let submit_clock = crate::scheduler::probe::ProbeClock::start();
    if let Some(refusal) = crate::policy::stopped(ctx.process_id()) {
        return Ok(Err(refusal));
    }
    {
        let placed_by_device = {
            let pass = ctx.resources().get(&fwd)?;
            pass.devgeo.is_some() || pass.float.is_some()
        };
        if placed_by_device {
            return submit_placed_by_device(ctx, this, fwd, frame).await;
        }
        let began = crate::scheduler::probe::ProbeClock::start();
        let opened = open_pipeline(ctx, &this, &fwd).await?;
        crate::probe_fire_record!(submit_probe.drain_settled_us, began.elapsed());
        let (pipe_fires, pipeline_failure, pipeline_scope) = match opened {
            Ok(opened) => opened,
            Err(error) => return Ok(Err(error)),
        };
        let (
            geometry,
            cells,
            rs_reps,
            fwd_rep,
            instance_id,
            scheduler,
            attn_mask,
            accesses,
            decode_envelope,
        ) = {
            let p = ctx.resources().get_mut(&fwd)?;
            if let Some(e) = &p.failed {
                return Ok(Err(format!(
                    "pipeline: forward-pass failed by an earlier fire: {e}"
                )));
            }
            let device_resolved = p.decode_envelope.unwrap_or(PortMask::NONE);
            let geometry_clock = crate::scheduler::probe::ProbeClock::start();
            let (geometry, attn_mask) = match host_geometry_of(p, device_resolved) {
                Ok(mapped) => mapped,
                Err(refusal) => return Ok(Err(refusal)),
            };
            crate::probe_fire_record!(submit_probe.geometry_us, geometry_clock.elapsed());
            if let Some(fires) = &p.host_geometry {
                fires.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let accesses = p.instance.program.channel_accesses.clone();
            (
                geometry,
                p.cells.clone(),
                p.rs_ws.clone(),
                fwd.rep(),
                p.bound_instance.instance_id,
                p.scheduler.clone(),
                attn_mask,
                accesses,
                p.decode_envelope,
            )
        };
        // A recurrent state a device-counted fold left unsettled is staged
        // on the frozen length as a lower bound, and the engine replays
        // from where its record stands: the fire goes ahead of the settle.
        // It could not wait for it: a frame's fires are dispatched once all
        // of them have arrived, this one among them.
        let pass_lanes = pass_lanes(ctx, &fwd)?;
        let ws = pass_lanes.kv[0].clone();
        let stores = crate::store::registry::get(ws.model, ws.engine);
        let page_size = stores.kv_page_size;
        let (lanes, landings) = match geometry.lanes(&pass_lanes.places, page_size) {
            Ok(placed) => placed,
            Err(error) => return Ok(Err(format!("pipeline: {error}"))),
        };
        let mut req = crate::engine::FireRequest {
            lanes,
            ..crate::engine::FireRequest::default()
        };
        let fold = geometry.fold.clone();
        stamp_lane_facts(ctx, &fwd, &mut req)?;
        req.boundary_program = true;
        let switched = Switched::of(ctx.resources().get(&fwd)?);
        let lanes_of = every_lane(&req);
        req.geometry = if decode_envelope.is_some() {
            GeometryClass::DecodeEnvelope
        } else {
            GeometryClass::Host
        };
        tracing::debug!(
            geometry = ?req.geometry,
            lanes = req.lanes.len(),
            qo_indptr = ?geometry.qo_indptr,
            "fire geometry resolved on host path"
        );
        req.single_token_mode = req.lanes.iter().all(|lane| lane.tokens.len() == 1);
        let media_spans: Vec<Vec<Arc<media::front::EncodedSpan>>> = ctx
            .resources()
            .get(&fwd)?
            .bindings
            .lanes
            .iter()
            .map(|lane| lane.media.clone())
            .collect();
        let carries_media = media_spans.iter().any(|spans| !spans.is_empty());
        let matched = {
            let lane_tokens: Vec<&[u32]> = req
                .lanes
                .iter()
                .map(|lane| lane.tokens.as_slice())
                .collect();
            match scan_lane_media(&lane_tokens, &media_spans) {
                Ok(matched) => matched,
                Err(refusal) => return Ok(Err(format!("pipeline: {refusal}"))),
            }
        };
        let fire_wide_mask = matches!(attn_mask, geometry::FireAttnMask::Device);
        if let Err(error) = attn_mask.apply_to(&mut req) {
            return Ok(Err(format!("pipeline: fire attention mask: {error}")));
        }
        if !matched.is_empty() {
            let lane_rows: Vec<u32> = req
                .lanes
                .iter()
                .map(|lane| u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX))
                .collect();
            let lane_base: Vec<u32> = req
                .lanes
                .iter()
                .map(|lane| lane.positions.first().copied().unwrap_or(lane.kv.held))
                .collect();
            req.media = crate::pipeline::media::lane_media(&matched, &lane_rows, &lane_base);
        }
        stamp_lane_words(&mut req, &switched, &lanes_of, fire_wide_mask, None);
        let wide = reads_each_other(&req);
        let kv_clock = crate::scheduler::probe::ProbeClock::start();
        let groups = lane_groups(&pass_lanes.kv.iter().map(|ws| ws.id).collect::<Vec<_>>());
        let group_sets: Vec<KvWorkingSet> = groups
            .iter()
            .map(|(_, lanes)| pass_lanes.kv[lanes[0]].clone())
            .collect();
        // Lanes that state their own page lists (beam search) write pages of
        // their sets' reserve the device names: every page from the frozen
        // full pages to what `reserve` granted is backed, and the engine is
        // handed each set's whole table to resolve them in.
        let BeamReserve {
            paged: paged_reserve,
            ends: beam_ends,
            commons: beam_commons,
        } = match reserve_beam_columns(
            binds_pages(ctx, &fwd)?,
            beam_common_of(instance_id),
            page_size,
            &groups,
            &pass_lanes,
            &landings,
            &stores,
        )? {
            Ok(reserve) => reserve,
            Err(refusal) => return Ok(Err(refusal)),
        };
        if let Err(refusal) = crate::inferlet::process::said(
            "seat the lanes",
            ctx.process_id(),
            seat_lane_slots(&mut req, &stores, &groups),
        )
        .await
        {
            return Ok(Err(refusal));
        }
        for (group, (_, lanes)) in groups.iter().enumerate() {
            let end = lanes
                .iter()
                .map(|&lane| landings[lane].end())
                .max()
                .unwrap_or(0);
            if let Err(error) =
                crate::inferlet::host::working_set::grow_pages(&group_sets[group], end)
            {
                return Ok(Err(format!("pipeline: KV working set: {error}")));
            }
        }
        let model = ws.model;
        let engine = ws.engine;
        let pid = ctx.process_id();
        let quorum_pipeline_id = pipeline_scope.scheduler_id();
        for set in &group_sets {
            if let Err(refusal) = set.claim_pipeline_scope(&pipeline_scope) {
                return Ok(Err(format!("pipeline: KV working set {refusal}")));
            }
        }
        let rs_ws_ids =
            match bound_rs_working_sets(ctx, &ws, &rs_reps, &req.qo_indptr(), &pipeline_scope)? {
                Ok(ids) => ids,
                Err(error) => return Ok(Err(error)),
            };
        // On a set with a recurrent state a run-ahead fire folds
        // no further than the rows the program has accepted, so a stop can
        // take back the rest; a fire whose tokens the host places is
        // accepted as it is submitted, and everything before it with it.
        // A state that buffers no row folds every row as before.
        // A pass whose program stated its rows final (CONTRACT-BC 25) takes
        // no lag: its rows are never taken back, so it folds at fire what
        // it states, as a host-placed fire does, and its next fire carries
        // nothing in.
        let lags = !rs_ws_ids.is_empty() && stores.rs.lock().unwrap().holds_unfolded();
        let final_rows = ctx.resources().get(&fwd)?.bindings.final_rows;
        let fold = pass_fold(
            fold,
            lags,
            decode_envelope.is_some(),
            final_rows,
            &pass_lanes,
            &groups,
            &landings,
        );
        let window_state = crate::model::model().rs_caps().window_state;
        let window_only = window_only(&stores, window_state);
        // A long host pass runs as several fires; all but the last run the
        // model's forward alone. A recurrent state folding all its rows, or
        // none, folds each fire's rows in turn; a windowed-only load's
        // other folds (a beam group's lagged common-ancestor fold) land with
        // the pass's last piece, its earlier pieces' rows buffered
        // (`piece_fold`).
        let folds_all = rs_ws_ids.is_empty()
            || [0, u32::MAX]
                .into_iter()
                .any(|count| (0..req.lanes.len()).all(|lane| fold.of(lane) == Some(count)));
        let adapted = lora_program_of(ctx, &fwd)?;
        // The program runs on a cut pass's last fire, after the first fires' forwards.
        let prologue_moves =
            container_prologue_moves(&ctx.resources().get(&fwd)?.instance.program.bound.container);
        let plain = switched.plain();
        let keyable = decode_envelope.is_none()
            && !wide
            && plain
            && groups.len() == 1
            && !adapted
            && !fire_wide_mask;
        // A pass that may start from a prefix the cache holds may be cut.
        let cuttable = decode_envelope.is_none()
            && !wide
            && plain
            && !prologue_moves
            && (!adapted || crate::model::model().eta_caps().adapter_attachments)
            && !fire_wide_mask
            && (folds_all || window_only);
        let mut full = req;
        let keyed = if keyable { keyed_rows(&full) } else { 0 };
        // Keyed under the set's domain: its scope's, resolved again now,
        // where it asked to share (`working-set.share`); else its own.
        let mut shares = false;
        let mut key_domain = None;
        let mut pass_slots = (keyed > 0)
            .then(|| {
                let lane = &full.lanes[0];
                let start = u64::from(lane.kv.held);
                let root = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
                    let (domain, shared) = kv.pass_domain(ws.id)?;
                    kv.chain_root(ws.id, start, page_size)
                        .map(|root| root.map(|prev| (domain, shared, prev)))
                });
                let (domain, shared, prev) = root.ok().flatten()?;
                shares = shared;
                key_domain = Some(domain);
                Some(crate::store::kv::hash::chain_slots(
                    &domain,
                    prev,
                    start,
                    &lane.tokens[..keyed as usize],
                ))
            })
            .flatten();
        // A window's pages are checkpointed like a folded state: a fork
        // the prefix cache keeps, sharing them by refcount (CONTRACT-BC 19).
        let state = match rs_ws_ids.as_slice() {
            [rs] => Some(*rs),
            _ => None,
        };
        // Only a set that shares maps, waits on, fetches or resumes from
        // what others computed.
        // A pass that ends no page past its held rows, and starts inside
        // one, can map, claim or bring up nothing: no whole page of it is
        // keyed beyond what the set holds (a decode step mid-page).
        let page = u64::from(page_size.max(1));
        let held_rows = u64::from(full.lanes[0].kv.held);
        let ends_no_page = !held_rows.is_multiple_of(page)
            && (held_rows + u64::from(keyed)) / page == held_rows / page;
        let attach = if !shares || !cuttable || !keyable || !plain_rows(&full) || ends_no_page {
            Attach::No
        } else if rs_ws_ids.is_empty() {
            Attach::Kv
        } else if let Some(rs) = state {
            Attach::Checkpoint(rs)
        } else {
            Attach::No
        };
        // What this pass claims to compute for others to wait on, held until
        // its last piece settles.
        let mut coalesced: Option<crate::pipeline::coalesce::ClaimGuard> = None;
        // Rows no worker here holds may be prefilled on a peer, whose pages
        // land in the host tier for the promote below to find. Media stays
        // here.
        if matches!(attach, Attach::Kv)
            && !carries_media
            && let Some(slots) = &pass_slots
            && let Some(domain) = key_domain
        {
            let lane = &full.lanes[0];
            // Pages another pass is computing now are waited for, not
            // computed again. Waiting on that pass's fire, the process
            // submits nothing: it is parked while it waits, unblocked as the
            // wait ends (greedy decodes of one prompt arriving together held
            // the claimant's first frame for the waiters' lanes for good).
            coalesced = crate::inferlet::process::said(
                "plan: pages another pass computes",
                pid,
                crate::pipeline::coalesce::before_pass(
                    &stores,
                    ws.id,
                    u64::from(lane.kv.held),
                    slots,
                    crate::inferlet::process::parks_here,
                ),
            )
            .await;
            crate::offload::prefill::before_pass(
                pid,
                &stores,
                engine,
                domain,
                ws.id,
                u64::from(lane.kv.held),
                &lane.tokens,
                slots,
            )
            .await;
            // A pass that computes a run a peer streams taps its fires.
            crate::offload::stream::bind(pid, &stores, ws.id, u64::from(lane.kv.held), slots);
        }
        // Pages below the device come back up first; the run stays held
        // until it is mapped in just below.
        let promoted = match &pass_slots {
            Some(slots) if !matches!(attach, Attach::No) => promote(
                ctx,
                &stores,
                &ws,
                &full,
                slots,
                quorum_pipeline_id,
            )
            .await
            .unwrap_or_else(|error| {
                tracing::debug!(%error, "kv supply: bringing sealed pages back to the device");
                None
            }),
            _ => None,
        };
        let Supplied {
            rows: supplied,
            keys,
        } = match &pass_slots {
            // Mapping nothing and keeping no state, a pass needs none of its
            // keys: the walk over the set's pages under the kv lock is not
            // taken for it.
            Some(_) if matches!(attach, Attach::No) && state.is_none() => Supplied::default(),
            Some(slots) => {
                let folds = fold.of(0).map_or(0, |count| {
                    u64::from(count).min(full.lanes[0].tokens.len() as u64)
                });
                supply(&stores, ws.id, &full, slots, attach, folds, page_size)
            }
            None => Supplied::default(),
        };
        // A state resumed is folded through what was mapped in, rows the
        // set holds from here on whatever becomes of the rest of the pass.
        if supplied > 0 && matches!(attach, Attach::Checkpoint(_)) {
            let mut cursor = pass_lanes.cursors[0].lock().unwrap();
            let end = full.lanes[0].kv.held.saturating_add(supplied);
            cursor.tail = Some(end);
            cursor.reserved = cursor.reserved.max(end);
        }
        // Of the rows mapped in, those on pages that came up from below.
        if let Some(promoted) = &promoted {
            let below = (promoted.pages * u64::from(page_size)).min(u64::from(supplied));
            crate::store::CACHED_FROM_BELOW_TOKENS
                .fetch_add(below, std::sync::atomic::Ordering::Relaxed);
        }
        drop(promoted);
        // What the fire writes and reads out, for the policies' prompt and
        // generated counts: rows mapped in from the cache are rows of the
        // prompt too.
        crate::policy::report(crate::policy::Event::Fired(
            pid,
            full.lanes.iter().map(|lane| lane.tokens.len() as u64).sum(),
            full.lanes
                .iter()
                .map(|lane| match &lane.readout {
                    ::engine::Readout::None => 0,
                    ::engine::Readout::Last => 1,
                    ::engine::Readout::Rows(rows) => rows.len() as u64,
                })
                .sum(),
        ));
        if pass_slots.is_some() && shares {
            // It looked, so what it found is measured, nothing included.
            ctx.note_cached(u64::from(supplied));
            crate::policy::report(crate::policy::Event::Cached(pid, u64::from(supplied)));
        }
        if supplied > 0 {
            // What was mapped in is skipped: the pass runs from there. The
            // rows mapped in are the set's to read: a prefix it shares.
            let mapped = full.lanes[0].kv.held.saturating_add(supplied);
            pass_lanes.cursors[0].lock().unwrap().wrote(mapped);
            let rows = u32::try_from(full.lanes[0].tokens.len()).unwrap_or(u32::MAX);
            full = rows_of_lanes(&full, &[Some((supplied, rows))], true, page_size);
            for lane in &mut full.lanes {
                lane.kv.readable = lane.kv.readable.max(mapped);
            }
            stamp_lane_words(&mut full, &switched, &lanes_of, false, None);
            pass_slots = pass_slots.map(|slots| slots[supplied as usize..].to_vec());
        }
        // Where the pass keeps its state: the checkpoint rows it reaches,
        // floored to whole pages it keys, each the end of a fire of its own.
        let held = u64::from(full.lanes[0].kv.held);
        // The rows it computes from here, sealed for its claims' waiters
        // once its fires land.
        let mut sealing = coalesced.take().map(|claims| SealOnLanding {
            claims,
            stores: stores.clone(),
            ws: ws.id,
            start: held,
            slots: pass_slots.clone().unwrap_or_default(),
            page_size,
            failure: pipeline_failure.clone(),
        });
        let reach = held + full.lanes[0].tokens.len() as u64;
        // Only on a set that shares: a later pass resumes from one only
        // through its scope. Asking on one that never did is refused at the
        // ask; one whose scope moved past its rows keeps none.
        let checkpoints: Vec<u64> = match state {
            Some(_) if shares => checkpoint_rows(
                &ctx.resources().get(&fwd)?.bindings.checkpoints,
                held..reach,
                page_size,
                keys.len(),
            ),
            _ => Vec::new(),
        };
        // An engine that lands a lane's state at rows inside a fire is cut only
        // past as many as it lands at once.
        let snapshots = crate::model::model().eta_caps().rs_snapshots as usize;
        let forced: Vec<u32> = checkpoints
            .iter()
            .skip(snapshots)
            .step_by(snapshots + 1)
            .map(|&row| (row - held) as u32)
            .collect();
        // Several lanes run in pieces only where none carries recurrent
        // state, which is bound lane by lane.
        let rows_total = full
            .lanes
            .iter()
            .map(|lane| lane.tokens.len())
            .sum::<usize>();
        // A prompt asks the activation rung for the rows a fire takes to
        // carry it whole beside the other live lanes' decodes (SPEC decision
        // 6, by demand), from before it is cut until its last piece settles:
        // the pool raises toward it where room allows, and lowers once no
        // pass asks past its smallest rung, so no piece cut wider waits on a
        // lowered rung. A few rows ask as a prompt does, whatever the cap
        // is now (the smallest rung that covers them, held while they are
        // queued); only a lone row the rung carries beside the live lanes'
        // asks nothing, no cap being narrower than it.
        // A fire asks for the recurrent rows it carries in unfolded too,
        // where they are more than its own: a rung's replay scratch holds
        // its tokens of them. The pass carries these into its first fire.
        let frozen = match lanes_frozen(
            &stores,
            &rs_ws_ids,
            &groups,
            full.lanes.len(),
            window_state,
        ) {
            Ok(frozen) => frozen,
            Err(error) => return Ok(Err(error)),
        };
        // A pass folding all it writes folds piece by piece, so only its
        // first piece carries rows in; one folding nothing carries its
        // earlier pieces' rows into each later one.
        let folds_each = (0..full.lanes.len()).all(|lane| fold.of(lane) == Some(u32::MAX));
        let whole: Vec<Option<(u32, u32)>> = full
            .lanes
            .iter()
            .map(|lane| Some((0, u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX))))
            .collect();
        // What no cut removes: the rows the pass carries in, checked with no
        // own row (a piece as small as the rules allow); its own rows and
        // carried rows together are each piece's, checked as it is cut.
        let carried = piece_rows(&full, &whole, &frozen, false).carried;
        let replayed = piece_replayed(&full, &whole, &frozen, false);
        let scratch = PassScratch {
            shared: (0..full.lanes.len())
                .map(|lane| {
                    groups
                        .iter()
                        .any(|(_, lanes)| lanes.len() > 1 && lanes.contains(&lane))
                })
                .collect(),
            checkpoints: &checkpoints,
            snapshots,
        };
        let whole_compact = scratch.compact(&stores, &full, &whole, &frozen, false);
        if let Err(refusal) = replay_rung(
            &stores,
            engine,
            FireRows { own: 0, carried },
            replayed,
            piece_lanes(&whole),
            whole_compact,
        ) {
            return Ok(Err(refusal));
        }
        let wanted = prefill_want(rows_total, engine).max(carried);
        // What its fires take of the rung: a lone row carrying more than
        // itself asks as a wider pass does.
        let taken = rows_total.max(carried as usize);
        let rung_ask = ask_rung(&stores, engine, pid, quorum_pipeline_id, taken, wanted);
        let mut cut_at = crate::store::shared::fire_cap(engine);
        let mut cuts = pass_pieces(
            &full,
            cuttable,
            rs_ws_ids.is_empty() || window_only,
            mask_share(&full, prefill_share(page_size, engine), &stores, engine),
            page_size,
            &forced,
            &switched.tails,
        );
        if let Err(refusal) = every_lane_in_last_piece(&cuts, full.lanes.len()) {
            return Ok(Err(refusal));
        }
        // Only a lone lane's rest is cut again once a raise lands.
        let recuttable = full.lanes.len() == 1 && cuts.len() > 1;
        // A piece that would not cut to the rung asks for its own width,
        // and one that carries more rows than its own for those.
        let pieces: Vec<FireRows> = cuts
            .iter()
            .enumerate()
            .map(|(at, piece)| piece_rows(&full, piece, &frozen, folds_each && at > 0))
            .collect();
        let widest = match pieces_rung(&stores, engine, &full, &cuts, &frozen, folds_each, &scratch)
        {
            Ok(widest) => widest,
            Err(refusal) => return Ok(Err(refusal)),
        };
        let (wanted, rung_ask) = if widest > wanted {
            // The wider ask stands before the narrower goes.
            let taken = taken.max(widest as usize);
            let wider = ask_rung(&stores, engine, pid, quorum_pipeline_id, taken, widest);
            drop(rung_ask);
            (widest, wider)
        } else {
            (wanted, rung_ask)
        };
        // The process holds the ask past this pass (an inferlet feeding a
        // prompt in several passes lowers nothing between them); a pass of a
        // few rows says its process asks nothing, unless it carries more.
        let pass_rows = pieces
            .iter()
            .map(|rows| rows.carried as usize)
            .max()
            .unwrap_or(0)
            .max(rows_total);
        crate::store::shared::hold_for_process(
            pid,
            engine,
            (pass_rows > RUNG_WANT_FROM).then_some(wanted),
        );
        // What one fire would run past what a fire holds is refused here, by
        // name, rather than by the scheduler's row count.
        let capacity = crate::engine::get_spec(engine)
            .map_or(usize::MAX, |spec| spec.limits.max_forward_tokens);
        let last_rows: usize = cuts.last().map_or(0, |piece| {
            piece
                .iter()
                .flatten()
                .map(|&(from, to)| (to - from) as usize)
                .sum()
        });
        if last_rows > capacity {
            let mut why: Vec<&str> = Vec::new();
            if cuts.len() > 1 || (cuttable && separable_rows(&full)) {
                why.push("the rows it reads out, or a media span and what follows it, ride its last fire");
            }
            if decode_envelope.is_some() || fire_wide_mask {
                why.push("its geometry or mask resolves on the device");
            }
            if prologue_moves {
                why.push("its prologue takes or puts a channel before the forward");
            }
            if adapted && !crate::model::model().eta_caps().adapter_attachments {
                why.push("this engine cannot carry its adapter on a fire of its own");
            }
            if switched.rows_read {
                why.push("it reads out an output of rows a cut fire would not hold");
            }
            if !folds_all && !window_only {
                why.push("it states where its recurrent state folds");
            }
            if full.lanes.len() > 1 && !rs_ws_ids.is_empty() && !window_only {
                why.push(
                    "its lanes carry rows its recurrent layers replay, which several lanes do \
                     not take in pieces",
                );
            }
            if why.is_empty() {
                why.push("its rows read one another (attention both ways, a block draft, an attention group, inputs or voxels)");
            }
            return Ok(Err(format!(
                "pipeline: a pass of {rows_total} rows would run {last_rows} in one fire, \
                 past the {capacity} a fire holds, and cannot be cut further: {}",
                why.join("; ")
            )));
        }
        // The pieces in frames of their own, in the order they are cut, so
        // the pieces of concurrent prompts seal together.
        let frame_size = crate::scheduler::configured_frame_size().max(1);
        // A cut pass behind fires of its own frame would run its leading
        // pieces (unframed) between them and its last piece (framed with
        // them): no order runs both before it. The frame goes with the fires
        // ahead of it, and the pass and the frame's later slots in frames of
        // their own, in order.
        let frame = match frame {
            Some(stamp) if reframes(stamp, ahead, cuts.len()) => {
                let _ = scheduler.frame_truncate(stamp.lane, stamp.seq, ahead);
                let pipeline = ctx.resources().get(&this)?;
                pipeline.mark_reframed(stamp.seq);
                Some(crate::scheduler::FrameStamp {
                    seq: pipeline.next_frame_seq(),
                    slot: 0,
                    fires: 1,
                    ..stamp
                })
            }
            frame => frame,
        };
        let mut frames = {
            let pipeline = ctx.resources().get(&this)?;
            piece_frames(frame, cuts.len(), frame_size, || pipeline.next_frame_seq())
        };
        let framed = matches!(frame, Some(stamp) if stamp.fires == 1);
        let mut submitted: Vec<crate::engine::completion::WorkItemCompletion> = Vec::new();
        // The working sets a piece of this pass has moved the cursor of.
        let mut advanced = vec![false; groups.len()];
        let mut writes = Vec::with_capacity(groups.len());
        let set_start = |group: usize, start: u64| {
            let (set, lanes) = (&group_sets[group], &groups[group].1);
            SetStart {
                model: set.model,
                engine: set.engine,
                kv: set.id,
                cursor: pass_lanes.cursors[lanes[0]].clone(),
                start,
                rs: rs_ws_ids.get(lanes[0]).copied(),
            }
        };
        let mut at = 0;
        while at < cuts.len() {
            if recuttable && at > 0 && at + 1 < cuts.len() {
                // While the raise is pending the pass runs no further ahead
                // of the device than a few pieces, so what is left is cut at
                // the raised rung once it lands.
                // A framed piece waits only on a frame already whole.
                let ahead = PIECES_AHEAD.max(frame_size);
                // A rise short of free room lands only once kv gives room
                // back: the pass fires at the rung held, and holds back no
                // piece for it.
                let held = rung_ask.as_ref().is_some_and(RungAsk::fires_at_held_rung);
                if at >= ahead && wanted > cut_at && !held {
                    let behind = submitted[at - ahead].clone();
                    // The piece behind settling or the cap rising, each its
                    // own event.
                    loop {
                        let watch = crate::store::shared::rung_watch(engine);
                        tokio::pin!(watch);
                        watch.as_mut().enable();
                        if behind.is_settled() || crate::store::shared::fire_cap(engine) > cut_at {
                            break;
                        }
                        tokio::select! {
                            () = &mut watch => {}
                            _ = behind.clone() => {}
                        }
                    }
                }
                let cap = crate::store::shared::fire_cap(engine);
                if cap > cut_at && (!framed || at % frame_size == 0) {
                    cut_at = cap;
                    let start = cuts[at - 1][0].map_or(0, |(_, to)| to);
                    let rest = host_cuts_from(
                        &full,
                        cuttable,
                        mask_share(&full, prefill_share(page_size, engine), &stores, engine),
                        page_size,
                        &forced,
                        switched.tails.first().copied().unwrap_or(0),
                        start,
                    );
                    cuts.truncate(at);
                    cuts.extend(rest.into_iter().map(|cut| vec![Some(cut)]));
                    let stamp = if framed {
                        let pipeline = ctx.resources().get(&this)?;
                        frame.map(|stamp| crate::scheduler::FrameStamp {
                            seq: pipeline.next_frame_seq(),
                            ..stamp
                        })
                    } else {
                        frame
                    };
                    let pipeline = ctx.resources().get(&this)?;
                    frames.truncate(at);
                    frames.extend(piece_frames(stamp, cuts.len() - at, frame_size, || {
                        pipeline.next_frame_seq()
                    }));
                }
            }
            let piece = cuts[at].clone();
            let piece = &piece;
            // A piece wider than the rung (one no cut brings under it) waits
            // for the raise its pass funds, or fails with what the planner
            // said: it never waits on a rung that does not come. A piece
            // cut again once a raise landed is counted as it is cut now.
            let counted = piece_rows(&full, piece, &frozen, folds_each && at > 0);
            let width = match replay_rung(
                &stores,
                engine,
                counted,
                piece_replayed(&full, piece, &frozen, folds_each && at > 0),
                piece_lanes(piece),
                scratch.compact(&stores, &full, piece, &frozen, folds_each && at > 0),
            ) {
                Ok(width) => width,
                Err(refusal) => return Ok(Err(refusal)),
            };
            let piece_compact =
                scratch.compact(&stores, &full, piece, &frozen, folds_each && at > 0);
            if let Some(ask) = rung_ask.as_ref()
                && let Err(refusal) = ask.wait(width).await
            {
                return Ok(Err(refusal));
            }
            let last = at + 1 == cuts.len();
            let mut req = if cuts.len() == 1 {
                full.clone()
            } else {
                let mut rows = rows_of_lanes(&full, piece, last, page_size);
                let kept: Vec<usize> = piece
                    .iter()
                    .enumerate()
                    .filter(|(_, cut)| cut.is_some())
                    .map(|(lane, _)| lane)
                    .collect();
                stamp_lane_words(&mut rows, &switched, &kept, false, None);
                rows
            };
            // What the scheduler steps and holds the fire by.
            req.carried = counted.carried;
            req.replayed = piece_replayed(&full, piece, &frozen, folds_each && at > 0);
            req.scratch_compact = piece_compact;
            req.replay_rungs = fire_rungs(&stores, !rs_ws_ids.is_empty());
            // A piece's request keeps only the lanes with rows in it,
            // renumbered; `groups` holds the pass's lane indices. What reads
            // the piece's request (its written spans, its page tables) needs
            // the piece's.
            let kept_lanes: Vec<usize> = if cuts.len() == 1 {
                (0..full.lanes.len()).collect()
            } else {
                piece
                    .iter()
                    .enumerate()
                    .filter(|(_, cut)| cut.is_some())
                    .map(|(lane, _)| lane)
                    .collect()
            };
            let piece_groups: LaneGroups = groups
                .iter()
                .map(|(ws, lanes)| {
                    (
                        *ws,
                        lanes
                            .iter()
                            .filter_map(|lane| kept_lanes.iter().position(|kept| kept == lane))
                            .collect(),
                    )
                })
                .collect();
            let spans: Vec<(std::ops::Range<u64>, std::ops::Range<u64>)> = piece_groups
                .iter()
                .map(|(_, lanes)| written_span_of(&req, lanes, page_size))
                .collect();
            let (slots, _) = spans[0].clone();
            let at_row = |rs, row: u64, snapshot| {
                let pages = row / u64::from(page_size.max(1));
                TakeCheckpoint {
                    kv: ws.id,
                    rs,
                    snapshot,
                    pages,
                    key: keys[pages as usize - 1],
                }
            };
            let mut checkpoints_here: Vec<TakeCheckpoint> = match state {
                Some(rs) if checkpoints.contains(&slots.end) => {
                    vec![at_row(rs, slots.end, false)]
                }
                _ => Vec::new(),
            };
            // The furthest the engine lands of those inside the fire.
            let mut inside: Vec<u32> = checkpoints
                .iter()
                .filter(|&&row| slots.start < row && row < slots.end)
                .map(|&row| (row - slots.start) as u32)
                .collect();
            inside.drain(..inside.len().saturating_sub(snapshots));
            let present = written_groups(&piece_groups);
            let sets: Vec<(&KvWorkingSet, Written)> = present
                .iter()
                .map(|&group| {
                    let written = match &paged_reserve {
                        Some(reserves) => Written::Pages(reserves[group].clone()),
                        None => Written::Span(spans[group].1.clone()),
                    };
                    (&group_sets[group], written)
                })
                .collect();
            // The piece's lanes' recurrent states, in the piece's lane
            // order: a cut of several lanes (a windowed-only load's) keeps
            // only those with rows in it.
            // What the piece folds: the pass's fold where it folds all it
            // writes or nothing, else (a windowed-only load's cut of other
            // folds) nothing before its last piece, which folds as the pass
            // states, every lane in it.
            let piece_fold = if folds_all || last {
                fold.clone()
            } else {
                geometry::Fold::None
            };
            let piece_rs_ids: Vec<crate::store::rs::RsWorkingSetId> = if rs_ws_ids.is_empty() {
                Vec::new()
            } else {
                kept_lanes.iter().map(|&lane| rs_ws_ids[lane]).collect()
            };
            let piece_rs_groups: LaneGroups = piece_groups
                .iter()
                .filter(|(_, lanes)| !lanes.is_empty())
                .cloned()
                .collect();
            // Each beam group's pool end and common slot, for the groups the
            // piece keeps (in `piece_rs_groups`' order).
            let (piece_beam_ends, piece_beam_commons): (Vec<u32>, Vec<u32>) = piece_groups
                .iter()
                .zip(beam_ends.iter().zip(&beam_commons))
                .filter(|((_, lanes), _)| !lanes.is_empty())
                .map(|(_, (&end, &common))| (end, common))
                .unzip();
            // The lanes each set carries, for what their state asks alone.
            let set_lanes: Vec<Vec<usize>> = if rs_ws_ids.is_empty() {
                Vec::new()
            } else {
                present.iter().map(|&group| piece_groups[group].1.clone()).collect()
            };
            let rs_plan = || {
                host_rs_plan(
                    &stores,
                    &piece_rs_ids,
                    &piece_rs_groups,
                    &req,
                    &piece_fold,
                    &inside,
                    (!piece_beam_ends.is_empty()).then_some(piece_beam_ends.as_slice()),
                    (!piece_beam_ends.is_empty()).then_some(piece_beam_commons.as_slice()),
                )
            };
            let reserved = reserve(
                ctx,
                (model, engine),
                &stores,
                quorum_pipeline_id,
                &pipeline_failure,
                &sets,
                &set_lanes,
                &piece_rs_ids,
                rs_plan,
            )
            .await;
            let Reserved {
                leases,
                copies: (copy_src, copy_dst),
                kvtxns,
                rs: rs_prepared,
                epoch,
                unfired,
            } = match reserved {
                Ok(reserved) => reserved,
                Err(error) => return Ok(Err(error)),
            };
            crate::probe_fire_record!(submit_probe.kv_prepare_us, kv_clock.elapsed());
            // Every fire logs what it writes where its tokens are known; what
            // it folds seals once it settles.
            let reports = pass_slots.is_none()
                && paged_reserve.is_none()
                && decode_envelope.is_some()
                && !wide
                && plain
                && plain_decode(&req, adapted, fire_wide_mask);
            let mut parts = Vec::with_capacity(group_sets.len());
            for (&group, (lease, kvtxn)) in present.iter().zip(leases.into_iter().zip(kvtxns)) {
                let (set, (slots, written)) = (&group_sets[group], &spans[group]);
                let lanes = &groups[group].1;
                let mut tags = Vec::new();
                let rs = rs_ws_ids.get(lanes[0]).copied();
                let by = fold_at(
                    &pass_lanes,
                    lanes,
                    &landings,
                    &piece_fold,
                    slots.end,
                    last,
                    rs,
                    &mut req,
                    &mut tags,
                );
                // What the piece folds is frozen with its log begun, in the
                // step `reserve` ended by committing the recurrent state's
                // fold: nothing is awaited between them. A fire that does
                // not follow fails the pipeline, and its sets with it,
                // however this call ends (`unfired`).
                let freeze = match &by {
                    Some(FoldBy::To(to)) => Some(*to),
                    _ => None,
                };
                let logged = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
                    let folded = kv.folded(set.id)?;
                    let pages = if reports && kv.keeps() {
                        let flat = kv.flat_table(set.id)?;
                        Some(
                            written
                                .clone()
                                .filter_map(|index| {
                                    flat.get(index as usize).map(|page| (page.0, index))
                                })
                                .collect::<Vec<_>>(),
                        )
                    } else {
                        None
                    };
                    let seq = begin_piece(kv, set.id, slots.start, freeze)?;
                    Ok::<_, crate::store::kv::KvStoreError>((folded, seq, pages))
                });
                let (folded, seq, pages) = match logged {
                    Ok(logged) => logged,
                    Err(error) => {
                        let reason = format!("pipeline: KV rows: {error}");
                        record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
                        return Ok(Err(reason));
                    }
                };
                for &lane in lanes {
                    if let Some(lane) = req.lanes.get_mut(lane) {
                        restate_boundary(lane, folded);
                    }
                }
                let Some(seq) = seq else {
                    let reason = format!(
                        "pipeline: a fire writes slot {}, inside its working set's frozen \
                         prefix (0..{folded})",
                        slots.start
                    );
                    record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
                    return Ok(Err(reason));
                };
                // A piece carrying media is not its tokens' alone.
                let rows = match (&pass_slots, pages) {
                    // A lane's own page list holds its slots apart from the
                    // set's: the set's rows there are no one token's.
                    _ if paged_reserve.is_some() => LogRows::Unknown,
                    (Some(_), _) if !req.media.is_empty() => LogRows::Unknown,
                    (Some(_), _) => LogRows::Known {
                        start: slots.start,
                        tokens: req.lanes[0].tokens.clone(),
                    },
                    (None, Some(pages)) => LogRows::Reported {
                        tag: report_on(&mut req, 0, &mut tags),
                        pages,
                        page_size,
                    },
                    (None, None) => LogRows::Unknown,
                };
                parts.push(FireWs {
                    ws: set.id,
                    kvtxn,
                    log: Some(LogFire { seq, rows }),
                    lease,
                    fold: by.map(|by| FoldAt {
                        by,
                        decoded: reports,
                    }),
                    reports: tags,
                    staged: None,
                });
            }
            if let Err(refusal) = map_lane_pages(&mut req, &stores, &piece_groups) {
                record_submit_failure(ctx, &fwd, &pipeline_failure, &refusal);
                return Ok(Err(refusal));
            }
            if paged_reserve.is_some()
                && let Err(refusal) = stamp_lane_reserve(&mut req, &stores, &piece_groups)
            {
                record_submit_failure(ctx, &fwd, &pipeline_failure, &refusal);
                return Ok(Err(refusal));
            }
            crate::offload::stream::tap(pid, &stores, &mut req);
            rs_prepared.apply_to(&mut req);
            debug_assert_eq!(
                untagged_device_fold(&req),
                None,
                "a lane whose fold the device counts asks for its report"
            );
            rs::apply_shared(&mut req, stores.shared.as_ref(), Some(&rs_prepared));
            checkpoints_here.extend(
                rs_prepared
                    .snapshots
                    .iter()
                    .map(|snap| at_row(snap.snapshot, slots.start + u64::from(snap.row), true)),
            );
            let rs_copy = rs_prepared.state_copy();
            let rstxns = RsTxnsGuard::new(model, engine, rs_prepared.txn);
            let completion = ctx
                .resources()
                .get_mut(&fwd)?
                .bound_instance
                .reserve_completion();

            // Only the last fire runs the program, so only it takes channel values.
            let ticket_reservation = last.then(|| TicketReservation::new(&cells, &accesses));
            if let Some(reservation) = &ticket_reservation {
                reservation.apply_to(&mut req);
            }
            // A run-ahead fire's rows on a set with a recurrent state or
            // windows wait for the program to take what it publishes.
            let run_ahead_reads = (last && decode_envelope.is_some() && lags && !final_rows)
                .then(|| host_reads(&cells, &accesses));

            let scheduler_clock = crate::scheduler::probe::ProbeClock::start();
            let submit_error =
                crate::scheduler::submit_prebuilt_tracked_async_with_kv_and_rs_copy_on(
                    &scheduler,
                    req,
                    instance_id,
                    pid,
                    quorum_pipeline_id,
                    completion.clone(),
                    copy_src,
                    copy_dst,
                    rs_copy,
                    frames[at],
                    adapted,
                )
                .err()
                .map(|error| format!("{error:#}"));
            crate::probe_fire_record!(submit_probe.scheduler_submit_us, scheduler_clock.elapsed());
            if let Some(error) = submit_error {
                let reason = format!("pipeline: submit failed: {error}");
                record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
                return Ok(Err(reason));
            }
            let unqueued = unfired.submitted();
            epoch.retire_on(&completion);
            for part in &mut parts {
                part.kvtxn.retire_on(&completion);
            }
            // Before the next piece is laid, which would write the record
            // the checkpoint keeps.
            checkpoints_ahead(&stores, &mut checkpoints_here, &mut parts);
            mark_device_folds(&stores, &completion, &mut parts);
            unsettle(&stores, &completion, &mut parts);
            submitted.push(completion.clone());
            if let Some(reservation) = ticket_reservation {
                reservation.commit();
            }

            if last {
                let began = crate::scheduler::probe::ProbeClock::start();
                advance_shadow(ctx, &fwd)?;
                crate::probe_fire_record!(submit_probe.shadow_advance_us, began.elapsed());
            }

            // Each set the piece writes ends where the piece does: a pass
            // that stops short of its last piece leaves the cursors at the
            // pieces that went.
            for &group in &present {
                let (lanes, slots) = (&groups[group].1, &spans[group].0);
                let cell = &pass_lanes.cursors[lanes[0]];
                let start = if advanced[group] {
                    slots.start
                } else {
                    let start = lanes
                        .iter()
                        .map(|&lane| u64::from(landings[lane].start))
                        .min()
                        .unwrap_or(slots.start);
                    writes.push(set_start(group, start));
                    start
                };
                advanced[group] = true;
                advance_cursor(
                    cell,
                    Some(slots.end as u32),
                    Some((
                        start as u32,
                        lagging_spans(&landings, &matched, lanes, &(start..slots.end)),
                    )),
                );
                host_wrote(cell, &completion, start as u32, slots.end as u32);
                if let Some(reads) = &run_ahead_reads {
                    cell.lock()
                        .unwrap()
                        .run_ahead(start as u32, slots.end as u32, reads.clone());
                }
            }
            unqueued.queue(
                &pipe_fires,
                PendingFire {
                    completion,
                    checkpoints: checkpoints_here,
                    kv: parts,
                    device_geometry: false,
                    rstxn: rstxns,
                    model,
                    engine,
                    fwd_rep,
                    cells: cells.clone(),
                    failure: pipeline_failure.clone(),
                    retired: None,
                    lane: Some(quorum_pipeline_id),
                },
            );
            // The next piece reads the KV this one writes: they run in
            // order on the device, so it is submitted now, as a decode's
            // next pass is, rather than once this one retires.
            at += 1;
        }
        if let (Some(ask), Some(last)) = (rung_ask, submitted.last()) {
            ask.until_settled(last);
        }
        // The claims on what this pass computes stand until its last piece
        // settles, its pages sealed if it landed; a pass that ends early
        // drops them.
        if let Some(sealing) = sealing.take()
            && !submitted.is_empty()
        {
            sealing.on_landing(&submitted);
        }
        // A set no piece wrote (its lanes carry no rows) ends where they land.
        for (group, (_, lanes)) in groups.iter().enumerate() {
            if advanced[group] {
                continue;
            }
            let end = lanes.iter().map(|&lane| landings[lane].end()).max();
            let start = lanes.iter().map(|&lane| landings[lane].start).min();
            if let (Some(end), Some(start)) = (end, start) {
                let cell = &pass_lanes.cursors[lanes[0]];
                advance_cursor(cell, Some(end), Some((start, Vec::new())));
                writes.push(set_start(group, u64::from(start)));
                if let Some(last) = submitted.last() {
                    host_wrote(cell, last, start, end);
                }
            }
        }
        record_writes(ctx, this.rep(), Some(writes), final_rows)?;
        crate::probe_fire_record!(submit_probe.total_us, submit_clock.elapsed());
        crate::probe_fire_count!(submit_probe.submits);
        Ok(Ok(()))
    }
}

/// The frame each of a pass's `pieces` fires carries, the pass's frame
/// `frame`, in frames of `k` steps. A piece reads the KV and state the one
/// before it writes, so a pass alone in its frame runs its pieces in order,
/// one a step, `k` to a frame: the first frame takes `frame`'s sequence and
/// each later one the next the pipeline hands out (`next_seq`). A lane's
/// pieces queue in their order and the pieces of concurrent lanes seal
/// together rather than one launch at a time. A pass sharing its frame keeps
/// that frame for its last piece, its earlier pieces unframed.
/// Whether a pass cut into `pieces`, stamped `stamp` behind `ahead` fires
/// of its own frame, goes in frames of its own: its leading pieces go
/// unframed and its last with the frame, so with fires of its frame ahead
/// no order would run them all first.
pub(super) fn reframes(stamp: crate::scheduler::FrameStamp, ahead: u32, pieces: usize) -> bool {
    stamp.fires > 1 && ahead > 0 && pieces > 1
}

/// Every lane of a pass rides its last fire, the one that runs the program:
/// a pass whose last fire left a lane out would read it nowhere.
pub(super) fn every_lane_in_last_piece(cuts: &[Vec<Option<(u32, u32)>>], lanes: usize) -> Result<(), String> {
    let held = cuts
        .last()
        .map_or(0, |piece| piece.iter().filter(|cut| cut.is_some()).count());
    if held == lanes {
        Ok(())
    } else {
        Err(format!(
            "pipeline: the last fire of a pass of {lanes} lanes carries {held}: every lane of \
             a pass rides its last fire, the one that runs its program"
        ))
    }
}

pub(crate) fn piece_frames(
    frame: Option<crate::scheduler::FrameStamp>,
    pieces: usize,
    k: usize,
    mut next_seq: impl FnMut() -> u64,
) -> Vec<Option<crate::scheduler::FrameStamp>> {
    let k = k.max(1);
    match frame {
        Some(stamp) if stamp.fires == 1 => {
            let mut seq = stamp.seq;
            (0..pieces)
                .map(|at| {
                    if at > 0 && at % k == 0 {
                        seq = next_seq();
                    }
                    let first = at - at % k;
                    Some(crate::scheduler::FrameStamp {
                        seq,
                        slot: (at % k) as u32,
                        fires: (pieces - first).min(k) as u32,
                        ..stamp
                    })
                })
                .collect()
        }
        frame => (0..pieces)
            .map(|at| if at + 1 == pieces { frame } else { None })
            .collect(),
    }
}

pub async fn submit_frame<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    slot_reps: Vec<Option<u32>>,
) -> Anyhow<Result<(), String>> {
    tracing::debug!("guest submits a frame");
    let k = crate::scheduler::configured_frame_size();
    if slot_reps.len() != k {
        return Ok(Err(format!(
            "pipeline: frame holds {} slot(s); model.frame-size() is {k} — \
             supply exactly k ordered slots (none = no-op)",
            slot_reps.len()
        )));
    }
    let fired: Vec<(u32, u32)> = slot_reps
        .iter()
        .enumerate()
        .filter_map(|(slot, rep)| rep.map(|rep| (slot as u32, rep)))
        .collect();
    if fired.is_empty() {
        return Ok(Err(
            "pipeline: a frame needs at least one non-no-op slot".to_string()
        ));
    }
    for &(slot, rep) in &fired {
        let fwd: Resource<ForwardPass> = Resource::new_borrow(rep);
        if !ctx.resources().get(&fwd)?.is_bound() {
            return Ok(Err(format!(
                "pipeline: frame slot {slot}: forward pass program is not attached"
            )));
        }
    }
    if k == 1 {
        let (_, rep) = fired[0];
        return submit_pass_stamped(ctx, this, Resource::new_borrow(rep), None, 0).await;
    }
    {
        let probe = crate::scheduler::probe::host_submit();
        let began = crate::scheduler::probe::ProbeClock::start();
        let verdict = validate_frame(ctx, k, &fired)?;
        crate::probe_fire_record!(probe.validate_frame_us, began.elapsed());
        crate::probe_fire_count!(probe.validate_frame_calls);
        if let Err(error) = verdict {
            return Ok(Err(error));
        }
    }
    let (lane, seq) = {
        let pipeline = ctx.resources().get(&this)?;
        if pipeline.scope.is_closed() {
            return Ok(Err("pipeline: pipeline is closed".to_string()));
        }
        (pipeline.scope.scheduler_id(), pipeline.next_frame_seq())
    };
    let fires = fired.len() as u32;
    for (index, &(slot, rep)) in fired.iter().enumerate() {
        // Once a cut pass truncated the frame, each slot after it goes in a
        // frame of its own.
        let reframed = ctx.resources().get(&this)?.reframed(seq);
        let (stamp, ahead) = if reframed {
            let seq = ctx.resources().get(&this)?.next_frame_seq();
            let own = crate::scheduler::FrameStamp {
                lane,
                seq,
                slot: 0,
                fires: 1,
            };
            (own, 0)
        } else {
            let stamp = crate::scheduler::FrameStamp {
                lane,
                seq,
                slot,
                fires,
            };
            (stamp, index as u32)
        };
        let pipeline: Resource<Pipeline> = Resource::new_borrow(this.rep());
        let fwd: Resource<ForwardPass> = Resource::new_borrow(rep);
        let outcome = submit_pass_stamped(ctx, pipeline, fwd, Some(stamp), ahead).await;
        if !matches!(outcome, Ok(Ok(()))) && index > 0 && !reframed {
            let first: Resource<ForwardPass> = Resource::new_borrow(fired[0].1);
            if let Ok(pass) = ctx.resources().get(&first)
                && let Ok(bound) = pass.bound()
            {
                let _ = bound.scheduler.frame_truncate(lane, seq, index as u32);
            }
        }
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                return Ok(Err(format!("pipeline: frame slot {slot}: {error}")));
            }
            Err(error) => return Err(error),
        }
    }
    Ok(Ok(()))
}

pub fn park_frame<C: FireContext>(ctx: &mut C, this: Resource<Pipeline>) -> Anyhow<()> {
    let pipeline = ctx.resources().get(&this)?;
    // Parked, it holds no batch, whatever the frame size.
    pipeline.set_active(false);
    if crate::scheduler::configured_frame_size() == 1 {
        return Ok(());
    }
    if pipeline.scope.is_closed() {
        return Ok(());
    }
    let (lane, seq) = (pipeline.scope.scheduler_id(), pipeline.next_frame_seq());
    crate::scheduler::worker::notify_lane_park(lane, seq);
    Ok(())
}

pub(super) fn validate_frame<C: FireContext>(
    ctx: &mut C,
    k: usize,
    fired: &[(u32, u32)],
) -> Anyhow<Result<(), String>> {
    let mut slots: Vec<SlotAccess> = Vec::with_capacity(fired.len());
    for &(_, rep) in fired {
        let fwd: Resource<ForwardPass> = Resource::new_borrow(rep);
        let pass = ctx.resources().get(&fwd)?;
        let bound = match pass.bound() {
            Ok(bound) => bound,
            Err(error) => return Ok(Err(format!("pipeline: {error}"))),
        };
        slots.push(SlotAccess {
            cells: bound.cells.clone(),
            accesses: bound.instance.program.channel_accesses.clone(),
        });
    }
    Ok(prove_frame_admissible(k, &slots))
}

pub(super) struct SlotAccess {
    pub(super) cells: BoundCells,
    pub(super) accesses: Vec<(bool, bool)>,
}

pub(super) fn prove_frame_admissible(k: usize, slots: &[SlotAccess]) -> Result<(), String> {
    struct ChannelUse {
        cell: Arc<Mutex<crate::pipeline::channel::ChannelCell>>,
        consumes: usize,
        publishes: usize,
    }
    struct DeviceRingUse {
        global_id: ::engine::ChannelId,
        capacity: u64,
        pressure: u64,
    }
    let mut uses: std::collections::HashMap<usize, ChannelUse> = std::collections::HashMap::new();
    let mut device_rings: std::collections::HashMap<usize, DeviceRingUse> =
        std::collections::HashMap::new();

    for (slot, step) in slots.iter().enumerate() {
        for (cell, &(consume, publish)) in step.cells.iter().zip(&step.accesses) {
            let key = Arc::as_ptr(cell) as usize;
            let entry = uses.entry(key).or_insert_with(|| ChannelUse {
                cell: cell.clone(),
                consumes: 0,
                publishes: 0,
            });
            entry.consumes += usize::from(consume);
            entry.publishes += usize::from(publish);

            if consume || publish {
                let guard = cell.lock().unwrap();
                if guard.role == Some(HostRole::None) && !guard.seeded {
                    let ring = device_rings.entry(key).or_insert_with(|| DeviceRingUse {
                        global_id: guard.global_id,
                        capacity: u64::from(guard.capacity),
                        pressure: guard.device_ring_backlog(),
                    });
                    if publish && ring.pressure >= ring.capacity + u64::from(consume) {
                        return Err(format!(
                            "pipeline: channel {}: frame slot {slot} would raise \
                             device-ring occupancy past capacity {} (reserved backlog \
                             plus in-frame publishes) — size device-only rings so \
                             every frame's publish backlog fits",
                            ring.global_id, ring.capacity,
                        ));
                    }
                    ring.pressure += u64::from(publish);
                    ring.pressure = ring.pressure.saturating_sub(u64::from(consume));
                }
            }
        }
    }

    for entry in uses.values() {
        let cell = entry.cell.lock().unwrap();
        match cell.role {
            Some(HostRole::Writer) => {
                if entry.publishes == 0 && entry.consumes > 0 {
                    let available = cell.writer_available_cells();
                    if available < entry.consumes as u64 {
                        return Err(format!(
                            "pipeline: channel {}: frame consumes {} host-writer \
                             cell(s) but only {available} are staged — stage every \
                             per-fire input before submitting the frame",
                            cell.global_id, entry.consumes
                        ));
                    }
                } else if entry.publishes == 0 && entry.consumes == 0 && !cell.has_committed_front()
                {
                    return Err(format!(
                        "pipeline: channel {}: latest-value control word has \
                         no committed cell at frame submit",
                        cell.global_id
                    ));
                }
            }
            Some(HostRole::Reader) if entry.publishes > 0 => {
                let (reserved_tail, consumed) = cell.reader_ring_pressure();
                let needed = reserved_tail
                    .saturating_sub(consumed)
                    .saturating_add(entry.publishes as u64);
                if needed > u64::from(cell.capacity) {
                    return Err(format!(
                        "pipeline: channel {}: frame would need {needed} reader \
                             cell(s) (capacity {}) — size take-side channels to at \
                             least 2k-1 = {} for frame-size k = {k}",
                        cell.global_id,
                        cell.capacity,
                        2 * k - 1,
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

pub(super) async fn pipeline_close_inner<C: FireContext>(
    ctx: &mut C,
    this: &Resource<Pipeline>,
) -> Anyhow<()> {
    let state = ctx.resources().get(this).ok().map(|pipeline| {
        pipeline.set_active(false);
        let first_close = pipeline.scope.close();
        (
            first_close,
            pipeline.scope.scheduler_id(),
            pipeline.fires.clone(),
        )
    });
    if let Some((first_close, pipeline_id, fires)) = state {
        if first_close {
            crate::scheduler::worker::notify_lane_close(pipeline_id, None);
        }
        drain_settled(ctx, Some(&fires)).await?;
    }
    Ok(())
}

pub async fn pipeline_close<C: FireContext>(ctx: &mut C, this: Resource<Pipeline>) -> Anyhow<()> {
    pipeline_close_inner(ctx, &this).await
}
