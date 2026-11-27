//! Stamping a fire's lanes: words, slots, pages, translation and channel tickets.

use super::*;

/// Stamps each lane's word: lane `at` of `req` is the pass's lane `of[at]`.
/// `device_rows`, when given, are the rows each lane carries on the device:
/// a pooled pass stages no host tokens, and its words still state its rows,
/// so a single-token lane takes the decode branch.
pub(super) fn stamp_lane_words(
    req: &mut crate::engine::FireRequest,
    switched: &Switched,
    of: &[usize],
    fire_wide_mask: bool,
    device_rows: Option<&[u32]>,
) {
    let model = crate::model::model();
    let requests = lane_requests(req, switched, of, fire_wide_mask, device_rows);
    for (lane, request) in req.lanes.iter_mut().zip(&requests) {
        lane.word = model.word(request);
    }
}

/// What each lane of `req` asks of the model ([`stamp_lane_words`]). A lane
/// captures its attention scores only on the fire that runs the pass's
/// program, the one reading them: an earlier cut of a cut pass carries the
/// adapter alone, and scores it captured would have no reader.
pub(super) fn lane_requests(
    req: &crate::engine::FireRequest,
    switched: &Switched,
    of: &[usize],
    fire_wide_mask: bool,
    device_rows: Option<&[u32]>,
) -> Vec<poem::Request> {
    let media: Vec<u32> = req.media.iter().map(|media| media.lane).collect();
    req.lanes
        .iter()
        .enumerate()
        .map(|(at, lane)| {
            let rows = device_rows
                .and_then(|rows| rows.get(at).copied())
                .unwrap_or_else(|| u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX));
            let on = of
                .get(at)
                .and_then(|&pass_lane| switched.on.get(pass_lane))
                .map_or(&[][..], Vec::as_slice);
            let request = poem::Request::new(rows, lane.mask.is_some() || fire_wide_mask)
                .adapted(lane.adapter.is_some())
                .drafting(on.contains(&poem_ir::Builtin::Drafts))
                .capturing_scores(req.boundary_program && on.contains(&poem_ir::Builtin::Scores))
                .with_media(media.contains(&(at as u32)))
                .starting(starts(lane));
            let switches = of
                .get(at)
                .and_then(|&pass_lane| switched.lanes.get(pass_lane))
                .map_or(&[][..], Vec::as_slice);
            switches.iter().fold(request, |request, (name, value)| {
                request.switched(name, value)
            })
        })
        .collect()
}

/// Whether the lane's recurrent state is empty, as the engine reads its reset.
pub(super) fn starts(lane: &::engine::Lane) -> bool {
    match lane.rs_reset {
        ::engine::RsReset::Fresh => true,
        ::engine::RsReset::Held => false,
        ::engine::RsReset::Inferred => !lane.kv_less && lane.kv.held == 0,
    }
}

pub(super) fn every_lane(req: &crate::engine::FireRequest) -> Vec<usize> {
    (0..req.lanes.len()).collect()
}

/// A lane in a both-ways class is never cut, keyed or logged by its tokens.
pub(super) fn reads_each_other(req: &crate::engine::FireRequest) -> bool {
    let model = crate::model::model();
    req.lanes.iter().any(|lane| model.wide(lane.word))
}

pub(crate) type LaneGroups = Vec<(crate::store::kv::page_table::WorkingSetId, Vec<usize>)>;

pub(crate) fn lane_groups(ids: &[crate::store::kv::page_table::WorkingSetId]) -> LaneGroups {
    let mut groups: LaneGroups = Vec::new();
    for (lane, &id) in ids.iter().enumerate() {
        match groups.iter_mut().find(|(ws, _)| *ws == id) {
            Some((_, lanes)) => lanes.push(lane),
            None => groups.push((id, vec![lane])),
        }
    }
    groups
}

/// Each lane rides a seat its working set holds: lane `i` of a group its
/// working set's `i`th.
pub(crate) fn stamp_lane_slots(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    groups: &LaneGroups,
) -> Result<(), crate::store::chunks::SeatsShort> {
    for (ws, lanes) in groups {
        let seats = stores.seats.lock().unwrap().seats(*ws, lanes.len())?;
        for (&lane, &seat) in lanes.iter().zip(&seats) {
            if let Some(lane) = req.lanes.get_mut(lane) {
                lane.slot = seat;
            }
        }
    }
    Ok(())
}

/// How long a process's fire may go without progress before the planner
/// counts it lapsed and plans again (its `FIRE_PROGRESS`); no waiter fails
/// on it.
pub(crate) const SEAT_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

pub(crate) async fn seat_lane_slots(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    groups: &LaneGroups,
) -> Result<(), String> {
    let freed = stores.seats.lock().unwrap().freed();
    // A working set letting its seats go is the event: it waits for it,
    // however long, and is refused at once only where no release could
    // ever seat it.
    loop {
        let mut freed = Box::pin(freed.notified());
        freed.as_mut().enable();
        let refusal = match stamp_lane_slots(req, stores, groups) {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        if !stores.seats.lock().unwrap().could_seat(refusal.need) {
            return Err(format!(
                "pipeline: seating this fire's lanes: {refusal}; no release can seat it"
            ));
        }
        // Waiting for seats let go: its owner is parked here.
        let _parked = crate::inferlet::process::parks_here();
        freed.await;
    }
}

pub(crate) fn map_lane_pages(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    groups: &LaneGroups,
) -> Result<(), String> {
    for (ws, lanes) in groups {
        if lanes
            .iter()
            .all(|&lane| req.lanes.get(lane).is_none_or(|lane| lane.pages.is_empty()))
        {
            continue;
        }
        let table = working_set_flat_table(stores, *ws)?;
        for &lane in lanes {
            let Some(lane) = req.lanes.get_mut(lane) else {
                continue;
            };
            map_pages(&mut lane.pages, &table)?;
            lane.ws = Some(*ws);
        }
    }
    Ok(())
}

pub(super) fn map_pages(pages: &mut [u32], table: &[u32]) -> Result<(), String> {
    for page in pages.iter_mut() {
        let Some(&pool) = table.get(*page as usize) else {
            return Err(format!(
                "pipeline: KV page {page} escapes the working set's {} mapped page(s)",
                table.len()
            ));
        };
        *page = pool;
    }
    Ok(())
}

pub(super) fn working_set_flat_table(
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
) -> Result<Vec<u32>, String> {
    crate::store::registry::with_kv_lock(&stores.kv, "host-pages", |kv| {
        Ok::<Vec<u32>, crate::store::kv::KvStoreError>(
            kv.flat_table(ws)?.iter().map(|page| page.0).collect(),
        )
    })
    .map_err(|error| format!("pipeline: KV page translation: {error}"))
}

/// Hands each lane that states its own page list its working set's whole
/// table: the pages its list names lie in it (`engine::fire::Lane::reserve`).
/// Its own list stays its rows' pages, which the engine stages and the
/// device then resolves over.
pub(super) fn stamp_lane_reserve(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    groups: &LaneGroups,
) -> Result<(), String> {
    for (ws, lanes) in groups {
        let table = working_set_flat_table(stores, *ws)?;
        for &lane in lanes {
            if let Some(lane) = req.lanes.get_mut(lane) {
                lane.lane.reserve.clone_from(&table);
                lane.ws = Some(*ws);
            }
        }
    }
    Ok(())
}

/// The named refusal of a beam group's growth that could never fit alone:
/// its set would need `end` pages (its frozen ones and its pool's `named`
/// for this fire's column) and the kv pool holds `capacity` with nothing
/// else resident. Anything smaller waits like any reservation. The search's
/// earlier columns stay valid output (`BEAM-PAGE-LISTS.md` §3).
pub(crate) const BEAM_POOL_OUTGROWN: &str = "outgrows the kv pool";

pub(super) fn beam_pool_fits(
    lane: usize,
    end: u64,
    named: u64,
    capacity: u32,
) -> Result<(), String> {
    if end > u64::from(capacity) {
        return Err(format!(
            "pipeline: lane {lane}'s beams' pool {BEAM_POOL_OUTGROWN}: its next column needs \
             {named} pool page(s), {end} with its frozen ones, and the kv pool holds \
             {capacity} with nothing else resident; the search's earlier columns are its output"
        ));
    }
    Ok(())
}

/// Whether `fwd`'s program states each beam's parent (its lanes hold their
/// own page lists of their set's reserve): decided when it registered.
pub(super) fn binds_pages<C: FireContext>(
    ctx: &mut C,
    fwd: &Resource<ForwardPass>,
) -> Anyhow<bool> {
    Ok(ctx.resources().get(fwd)?.instance.program.binds_pages)
}

/// The slots lane `lane`'s rows may land at where the device places them,
/// on a set frozen through `folded` that reserved through `reserved`: past
/// its frozen prefix, inside what `reserve` granted. The engine is handed
/// the set's whole page table to read through, and refuses a row its
/// program places outside these ([`::engine::KvDelta::writable`]).
pub(super) fn device_writable(
    lane: usize,
    folded: u32,
    reserved: u32,
) -> Result<std::ops::Range<u32>, String> {
    if reserved <= folded {
        return Err(format!(
            "pipeline: lane {lane}'s rows land where the device says, inside what \
             `reserve` granted; its working set reserved {reserved} slot(s) and froze \
             {folded}. Reserve the slots it may write"
        ));
    }
    Ok(folded..reserved)
}

/// The pages under `writable`, which the fire makes its set's own (copied
/// where shared) before it runs.
pub(super) fn writable_pages(writable: &std::ops::Range<u32>, page: u64) -> std::ops::Range<u64> {
    u64::from(writable.start) / page..u64::from(writable.end).div_ceil(page)
}

pub(super) fn stamp_lane_translation(
    req: &mut crate::engine::FireRequest,
    stores: &crate::store::registry::Stores,
    groups: &LaneGroups,
) -> Result<(), String> {
    for (ws, lanes) in groups {
        let table = working_set_flat_table(stores, *ws)?;
        for &lane in lanes {
            if let Some(lane) = req.lanes.get_mut(lane) {
                lane.translation.clone_from(&table);
                lane.ws = Some(*ws);
            }
        }
    }
    Ok(())
}

pub(super) fn poison_readers(cells: &BoundCells, reason: &str) {
    for cell in cells {
        let mut c = cell.lock().unwrap();
        if c.role == Some(HostRole::Reader) {
            c.poison(reason);
            if let Some(endpoint) = c.endpoint() {
                waker::WakerTable::global().wake(endpoint.registered().reader_wait_id);
            }
        }
    }
}

/// The moves a fire counted on its rings, given back if the fire never
/// reaches the engine.
pub(super) struct TicketReservation {
    pub(super) cells: BoundCells,
    pub(super) accesses: Vec<(bool, bool)>,
    pub(super) committed: bool,
}

impl TicketReservation {
    pub(super) fn new(cells: &BoundCells, accesses: &[(bool, bool)]) -> Self {
        for (cell, &(consume, publish)) in cells.iter().zip(accesses) {
            cell.lock().unwrap().reserve_device_moves(consume, publish);
        }
        Self {
            cells: cells.clone(),
            accesses: accesses[..cells.len().min(accesses.len())].to_vec(),
            committed: false,
        }
    }

    /// The fire carries which ends of which rings it moves, so the step
    /// builder keeps two movers of one end in different steps.
    pub(super) fn apply_to(&self, request: &mut crate::engine::FireRequest) {
        request.ring_moves = self
            .cells
            .iter()
            .zip(&self.accesses)
            .filter(|(_, (consume, publish))| *consume || *publish)
            .map(|(cell, &(takes, puts))| crate::engine::fire::RingMove {
                channel: cell.lock().unwrap().global_id,
                takes,
                puts,
            })
            .collect();
    }

    pub(super) fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for TicketReservation {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for (cell, &(consume, publish)) in self.cells.iter().zip(&self.accesses) {
            cell.lock().unwrap().release_device_moves(consume, publish);
        }
    }
}

pub(super) fn trace_channel_wait(waiting_on_reader: bool, waiting_on_fire: bool) {
    tracing::debug!(
        waiting_on_reader,
        waiting_on_fire,
        "guest parks on a channel"
    );
}

/// The process is about to wait on its pipeline's oldest fire: its lane
/// tells the scheduler so ([`PendingOp::tell_awaited`]).
pub(crate) fn tell_front_awaited(fires: Option<&PendingFires>) {
    if let Some(fires) = fires
        && let Some(front) = fires.lock().unwrap().front()
    {
        front.tell_awaited();
    }
}

pub(crate) async fn await_channel_progress(
    cell: &Arc<Mutex<crate::pipeline::channel::ChannelCell>>,
    fires: Option<&PendingFires>,
) -> Result<(), String> {
    let wait = cell.lock().unwrap().reader_wait_state();
    let oldest = fires.and_then(|f| f.lock().unwrap().front().map(|op| op.completion_signal()));
    trace_channel_wait(wait.is_some(), oldest.is_some());
    match (wait, oldest) {
        (Some((endpoint, observed_tail)), Some(signal)) => {
            tokio::select! {
                _ = endpoint.wait_for_reader_change(observed_tail) => {}
                _ = signal => {}
            }
            Ok(())
        }
        (Some((endpoint, observed_tail)), None) => endpoint
            .wait_for_reader_change(observed_tail)
            .await
            .map_err(|error| error.to_string()),
        (None, Some(signal)) => {
            signal.await;
            Ok(())
        }
        (None, None) => Err(ChannelError::Empty.to_string()),
    }
}
