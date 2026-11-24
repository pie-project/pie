//! The rows a pass carries: lane rows, pieces and their rungs, and the prefill and mask shares.

use super::*;

/// A lane's rows in a fire, or a fire's (its lanes' summed): its own, and
/// the recurrent rows it carries in unfolded, which earlier fires wrote and
/// nothing has folded, and which it replays ahead of its own
/// ([`::engine::fire::rung_holds`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct FireRows {
    pub(super) own: u32,
    pub(super) carried: u32,
}

impl FireRows {
    /// The rows a rung must hold for these: its own and its carried rows
    /// each.
    #[cfg(test)]
    pub(super) fn ask(self) -> u32 {
        self.own.max(self.carried)
    }

    pub(super) fn of(lanes: impl IntoIterator<Item = FireRows>) -> FireRows {
        lanes
            .into_iter()
            .fold(FireRows::default(), |sum, lane| FireRows {
                own: sum.own.saturating_add(lane.own),
                carried: sum.carried.saturating_add(lane.carried),
            })
    }
}

/// The own and carried rows of a lane the host places at `start..end`, its
/// recurrent state folded through `frozen` (`None`: it has none, and
/// carries nothing): it carries the rows between its state and its first
/// row. A lane of one row asks the rung for none of them: the engines'
/// decode step replays them as data, inside its own rows' body.
pub(super) fn lane_rows(start: u32, end: u32, frozen: Option<u32>) -> FireRows {
    let own = end.saturating_sub(start);
    FireRows {
        own,
        carried: ::engine::fire::lane_carried(
            own,
            frozen.map_or(0, |frozen| start.saturating_sub(frozen)),
        ),
    }
}

/// The `own` rows of a lane the device places, and a bound on what it
/// carries, its recurrent state folded through `frozen` (`None`: it has
/// none, and carries nothing). Its first row is its program's, resolved on
/// the device after its set's earlier fires have run, so no count exists
/// here; it replays no more than the rows standing unfolded in its set
/// then: those written past `frozen`, up to the set's written end, by
/// fires that settled, and every row of its fires staged and still to
/// settle ([`::engine::fire::rung_holds`], device geometry).
pub(super) fn device_lane_rows(
    own: u32,
    cursor: &crate::inferlet::host::working_set::Cursor,
    frozen: Option<u32>,
) -> FireRows {
    FireRows {
        own,
        carried: ::engine::fire::lane_carried(
            own,
            frozen.map_or(0, |frozen| cursor.unfolded(frozen)),
        ),
    }
}

/// The frozen length of each lane's recurrent state, by the lane's group
/// (`None` a lane where the pass binds none, or on a model with window
/// lists, whose rows trim behind their windows and are never replayed, or
/// whose windowed layers are ring + buffer, `replays_none`: their buffered
/// rows' K/V are read from the window buffer, never replayed).
pub(super) fn lanes_frozen(
    stores: &crate::store::registry::Stores,
    ids: &[crate::store::rs::RsWorkingSetId],
    groups: &LaneGroups,
    lanes: usize,
    replays_none: bool,
) -> Result<Vec<Option<u32>>, String> {
    let mut frozen = vec![None; lanes];
    if ids.is_empty() || replays_none {
        return Ok(frozen);
    }
    let store = stores.rs.lock().unwrap();
    for (_, group) in groups {
        let at = store
            .frozen(ids[group[0]])
            .map_err(|error| format!("pipeline: recurrent-state mode: {error}"))?;
        for &lane in group {
            frozen[lane] = Some(at);
        }
    }
    Ok(frozen)
}

/// Whether the load's only state is windowed layers served as ring +
/// buffer (`window_state`, its record the rings alone: no recurrent layer,
/// whatever buffer lists the engine states): its lanes, a beam group's
/// sharing one record among them, replay nothing, their unfolded rows going
/// to window-buffer slots on demand, so a pass of several lanes is cut as
/// one of kv lanes is (CONTRACT-BC 19: a fire too wide is cut like chunked
/// prefill, never refused). A hybrid's recurrent layers replay the rows its
/// lanes carry: its lanes are cut as an rs load's.
pub(crate) fn window_only(stores: &crate::store::registry::Stores, window_state: bool) -> bool {
    window_state
        && stores
            .shared
            .as_ref()
            .is_some_and(|shared| !shared.lock().unwrap().recurrent())
}

/// The rows of one piece of a host pass: each lane's cut of `full`, over a
/// state folded through `frozen` when the pass began. A pass that folds
/// all it writes folds piece by piece (a fold lands as its fire is
/// prepared), so its pieces past the first carry nothing (`folded_behind`);
/// one that folds nothing carries every row its earlier pieces wrote.
pub(super) fn piece_rows(
    full: &crate::engine::FireRequest,
    piece: &[Option<(u32, u32)>],
    frozen: &[Option<u32>],
    folded_behind: bool,
) -> FireRows {
    FireRows::of(
        full.lanes
            .iter()
            .zip(piece)
            .enumerate()
            .filter_map(|(lane, (rows, cut))| {
                let (from, to) = (*cut)?;
                let start = rows.kv.held.saturating_add(from);
                let end = rows.kv.held.saturating_add(to);
                let frozen = frozen
                    .get(lane)
                    .copied()
                    .flatten()
                    .map(|frozen| if folded_behind { start } else { frozen });
                Some(lane_rows(start, end, frozen))
            }),
    )
}

/// The rung a fire of `rows` asks, `replayed` every lane's carried rows (a
/// decode lane's included): the rows of the smallest activation rung whose
/// stated replay scratch holds it ([`::engine::fire::rung_holds`] against
/// each rung's [`::engine::RungReplay`], its replay scratch rows counted as
/// the load lays them, [`::engine::ReplayRows::of_sums`]), or its own and
/// carried rows where the first rung that wide holds it already. Where the
/// pool states no rung, one fire's tokens at the contract's floor. Refused
/// by name, naming the bound that binds, where no rung holds it: before any
/// step. Own rows past every rung are not this check's (a fire's capacity,
/// refused where the pass is cut).
pub(super) fn replay_rung(
    stores: &crate::store::registry::Stores,
    engine: usize,
    rows: FireRows,
    replayed: u32,
    lanes: u32,
    compact: bool,
) -> Result<u32, String> {
    let holds = crate::engine::get_spec(engine).map_or(u32::MAX, |spec| {
        u32::try_from(spec.limits.max_forward_tokens).unwrap_or(u32::MAX)
    });
    let (stated, widest) = stores.shared.as_ref().map_or_else(
        || (std::sync::Arc::new(Vec::new()), u32::MAX),
        |shared| shared.lock().unwrap().rung_replays(),
    );
    let reach = holds.min(widest);
    let end = stated
        .iter()
        .position(|&(tokens, _)| tokens > reach)
        .unwrap_or(stated.len());
    let pinned;
    let rungs: &[(u32, ::engine::RungReplay)] = if end == 0 {
        // No rung stated (pinned pools) or none a fire may take: one fire's
        // tokens, its own and carried rows each, the engine's scratch its own.
        pinned = [(holds, crate::store::shared::rows_only(holds))];
        &pinned
    } else {
        &stated[..end]
    };
    let count = ::engine::ReplayRows::of_sums(rows.own, rows.carried, replayed, lanes, compact);
    let floor = rows.own.max(rows.carried);
    let mut first = true;
    let mut past = None;
    for (tokens, bound) in rungs {
        let tokens = *tokens;
        if tokens < floor {
            continue;
        }
        match ::engine::fire::rung_holds(bound, count) {
            Ok(()) => return Ok(if first { floor } else { tokens }),
            Err(why) => past = Some((tokens, why)),
        }
        first = false;
    }
    let (tokens, why) = match past {
        Some(past) => past,
        None => {
            let (tokens, bound) = rungs.last().expect("one rung at least");
            let tokens = *tokens;
            match ::engine::fire::rung_holds(bound, count) {
                Ok(()) => return Ok(tokens),
                Err(why) => (tokens, why),
            }
        }
    };
    if why.bound == ::engine::ReplayBound::Own {
        return Ok(tokens);
    }
    Err(format!(
        "pipeline: no activation rung holds this fire's replay: at the largest, of {tokens} \
         tokens, it has {why}; fold what its lanes have written before they write more"
    ))
}

/// The load's stated rungs a fire carrying recurrent state hands the
/// scheduler, which holds a step of several such fires to the rung its
/// width picks ([`crate::engine::FireRequest::replay_rungs`]); `None` for a
/// fire carrying none.
pub(super) fn fire_rungs(
    stores: &crate::store::registry::Stores,
    carries: bool,
) -> Option<crate::store::shared::Replays> {
    let shared = stores.shared.as_ref().filter(|_| carries)?;
    Some(shared.lock().unwrap().rung_replays().0)
}

/// The widest rung a host pass's pieces ask, each piece as it is cut
/// ([`replay_rung`]: its own rows and the rows it carries in, against every
/// rung's stated bounds); refused by name where some piece no rung holds.
/// The uncut pass's own rows are no piece's.
pub(super) fn pieces_rung(
    stores: &crate::store::registry::Stores,
    engine: usize,
    full: &crate::engine::FireRequest,
    cuts: &[Vec<Option<(u32, u32)>>],
    frozen: &[Option<u32>],
    folds_each: bool,
    scratch: &PassScratch<'_>,
) -> Result<u32, String> {
    let mut widest = 0u32;
    for (at, piece) in cuts.iter().enumerate() {
        let behind = folds_each && at > 0;
        let rows = replay_rung(
            stores,
            engine,
            piece_rows(full, piece, frozen, behind),
            piece_replayed(full, piece, frozen, behind),
            piece_lanes(piece),
            scratch.compact(stores, full, piece, frozen, behind),
        )?;
        widest = widest.max(rows);
    }
    Ok(widest)
}

/// The lanes with rows in `piece`.
pub(super) fn piece_lanes(piece: &[Option<(u32, u32)>]) -> u32 {
    u32::try_from(piece.iter().filter(|cut| cut.is_some()).count()).unwrap_or(u32::MAX)
}

/// What a host pass's pieces' rs scratch is laid by
/// ([`::engine::fire::scratch_compact`]), from what the runtime puts in
/// each piece's fire: which lanes share their set with another (their
/// buffered reads are their masks' lineage, `Lineage::Admitted`, else
/// every buffered slot, `Lineage::Set`, as `RsVerb::Buffer` states them),
/// and the rows a piece lands its state at (`checkpoints`, as many of them
/// inside a fire as the engine lands, `snapshots`; a piece of several
/// lanes lands none), each an rs snapshot verb.
pub(crate) struct PassScratch<'a> {
    pub(crate) shared: Vec<bool>,
    pub(crate) checkpoints: &'a [u64],
    pub(crate) snapshots: usize,
}

impl PassScratch<'_> {
    /// Whether `piece`'s fire's rs scratch is compact.
    pub(super) fn compact(
        &self,
        stores: &crate::store::registry::Stores,
        full: &crate::engine::FireRequest,
        piece: &[Option<(u32, u32)>],
        frozen: &[Option<u32>],
        folded_behind: bool,
    ) -> bool {
        let mut kept: Vec<(u64, u64)> = Vec::new();
        let mut one_run = true;
        for (lane, (rows, cut)) in full.lanes.iter().zip(piece).enumerate() {
            let Some((from, to)) = *cut else {
                continue;
            };
            let start = rows.kv.held.saturating_add(from);
            let end = rows.kv.held.saturating_add(to);
            kept.push((u64::from(start), u64::from(end)));
            let Some(state) = frozen.get(lane).copied().flatten() else {
                continue;
            };
            let folded = if folded_behind { start } else { state };
            let shared = self.shared.get(lane).copied().unwrap_or(false);
            let lineage = if shared {
                ::engine::fire::Lineage::Admitted
            } else {
                ::engine::fire::Lineage::Set
            };
            // A lane sharing its set whose mask the host does not state (the
            // device resolves it) reads slots the host cannot name: not one
            // run as far as it can say.
            one_run &= !(shared && rows.mask.is_none())
                && ::engine::fire::lane_one_run(rows.mask.as_ref(), folded, start, lineage);
        }
        let snapshots = match kept.as_slice() {
            [(start, end)] => {
                self.snapshots > 0
                    && self.checkpoints.iter().any(|&row| *start < row && row < *end)
            }
            _ => false,
        };
        fire_compact(stores, one_run, snapshots)
    }
}

/// Whether a fire whose lanes' buffered rows are each one stretch (or not)
/// and that lands rs snapshots (or not) has a compact rs scratch on these
/// stores' load ([`::engine::fire::scratch_compact`]).
pub(super) fn fire_compact(
    stores: &crate::store::registry::Stores,
    every_lane_one_run: bool,
    any_lane_snapshots: bool,
) -> bool {
    stores.shared.as_ref().is_some_and(|shared| {
        let pool = shared.lock().unwrap();
        ::engine::fire::scratch_compact(
            pool.rs_compact(),
            pool.rs_scratch_per_fire(),
            every_lane_one_run,
            any_lane_snapshots,
        )
    })
}

/// The buffered recurrent rows one piece of a host pass replays, summed
/// over every lane, a decode lane's included (what
/// [`::engine::FireLimits::replayed_rows`] bounds a step by): each lane's
/// rows between its state and its first, as [`piece_rows`] counts them
/// before [`::engine::fire::lane_carried`] drops a decode lane's.
pub(super) fn piece_replayed(
    full: &crate::engine::FireRequest,
    piece: &[Option<(u32, u32)>],
    frozen: &[Option<u32>],
    folded_behind: bool,
) -> u32 {
    full.lanes
        .iter()
        .zip(piece)
        .enumerate()
        .filter_map(|(lane, (rows, cut))| {
            let (from, _) = (*cut)?;
            let start = rows.kv.held.saturating_add(from);
            let frozen = frozen.get(lane).copied().flatten()?;
            Some(if folded_behind { 0 } else { start.saturating_sub(frozen) })
        })
        .fold(0u32, u32::saturating_add)
}


pub(super) fn group_fold(fold: &geometry::Fold, first: usize, frozen: u32, end: u32) -> rs::GroupFold {
    match fold.of(first) {
        None => rs::GroupFold::Device,
        Some(count) => match count.min(end.saturating_sub(frozen)) {
            0 => rs::GroupFold::None,
            count => rs::GroupFold::Host(count),
        },
    }
}

/// Whether the pass's program adapts the KV it writes beyond its tokens.
pub(super) fn lora_program_of<C: FireContext>(ctx: &mut C, fwd: &Resource<ForwardPass>) -> Anyhow<bool> {
    let p = ctx.resources().get(fwd)?;
    Ok(container_has_lora_sink(&p.instance.program.bound.container))
}

/// The rows a piece of a lone masked lane takes so its mask plane fits the
/// room of the rung it runs at: a rung of `t` tokens holds `max(t * t,
/// max_context)` mask bits ([`::engine::Lane::mask`]), and a piece of `r`
/// rows over a buffer of `b` stages `r * (b + r)`. Bounded by the buffer the
/// pass's last piece finds (`b0 + rows - r`), every piece fits; nothing is
/// refused for a long buffer, the pass takes more fires.
pub(super) fn mask_share(
    req: &crate::engine::FireRequest,
    share: u32,
    stores: &crate::store::registry::Stores,
    engine: usize,
) -> u32 {
    let [lane] = req.lanes.as_slice() else {
        return share;
    };
    if lane.mask.is_none() || lane.kv_less {
        return share;
    }
    let context = crate::engine::get_spec(engine).map_or(0, |spec| spec.limits.max_context) as u64;
    let rows = lane.tokens.len() as u64;
    let span = u64::from(lane.kv.held.saturating_sub(lane.kv.folded)) + rows;
    let rung = |r: u32| -> u64 {
        stores
            .shared
            .as_ref()
            .map_or(r, |shared| shared.lock().unwrap().rung_for(r).max(r))
            .into()
    };
    mask_rows(share, span, context, rung)
}

/// The largest `r <= share` with `r * span <= max(rung(r)^2, context)`, at
/// least one row: [`mask_share`] apart from the stores.
pub(super) fn mask_rows(share: u32, span: u64, context: u64, rung: impl Fn(u32) -> u64) -> u32 {
    let room = |r: u32| {
        let t = rung(r);
        t.saturating_mul(t).max(context)
    };
    let mut r = share.max(1);
    while r > 1 && u64::from(r).saturating_mul(span) > room(r) {
        let fits = u32::try_from(room(r) / span.max(1)).unwrap_or(u32::MAX);
        r = fits.min(r - 1).max(1);
    }
    r
}

/// The prompt rows one fire on `engine` carries: the prefill budget shared
/// among the live lanes.
pub(super) fn prefill_share(page: u32, engine: usize) -> u32 {
    crate::engine::get_spec(engine).map_or(u32::MAX, |spec| {
        crate::inferlet::host::model::prefill_share(
            // Within the pool's activation rung (SPEC decision 6).
            crate::inferlet::host::model::prefill_budget(spec.limits)
                .min(crate::store::shared::fire_cap(engine) as usize)
                .max(1),
            crate::inferlet::process::live_count(),
            page as usize,
        )
    })
}

/// The rows a fire on `engine` takes to carry `rows` of a prompt whole
/// beside the decode row each other live lane spends in it
/// ([`crate::inferlet::host::model::prefill_share`]), up to what the
/// largest activation rung holds: what a pass asks the rung for
/// ([`crate::store::shared::want_rows`]).
pub(super) fn prefill_want(rows: usize, engine: usize) -> u32 {
    let live = crate::inferlet::process::live_count();
    let wide = rows.saturating_add(live.saturating_sub(1));
    let wide = u32::try_from(wide).unwrap_or(u32::MAX);
    crate::engine::get_spec(engine).map_or(wide, |spec| {
        let budget = crate::inferlet::host::model::prefill_budget(spec.limits).max(1);
        wide.min(u32::try_from(budget).unwrap_or(u32::MAX))
    })
}
