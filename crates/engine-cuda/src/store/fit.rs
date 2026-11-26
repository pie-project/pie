//! How the pool fits the card: the room a load has, the working set it
//! seats, the declared context it serves, and the activation's span.

#[allow(clippy::wildcard_imports)]
use super::*;

/// What `Pools::fit_the_card` takes of the card's room for one working set:
/// its chunks and their bytes backed, what is held for the bodies and the
/// guests' programs beside it, the room past those holds, and the whole map
/// units of it the chunks are fitted in.
pub(super) struct WorkingSet {
    pub(super) least: u32,
    pub(super) need: u64,
    pub(super) bodies: u64,
    pub(super) programs: u64,
    pub(super) room: u64,
    pub(super) limit: u64,
}

/// The bodies' hold on a load that has not yet armed them: everything the
/// card has past the claim (one working set and the weights the pool seats
/// whole), so arming takes what the bodies take and cannot come short of
/// spare memory while the card holds them; the kv pool fills what they
/// leave, once they are captured ([`Pools::fit_and_state`]).
pub const MEASURING: u64 = u64::MAX;

/// What the fit holds beside its claim, ahead of the kv pool's fill: the
/// bodies first (what they take where it is measured, every byte past the
/// claim where they are still to arm, [`MEASURING`]), then the guests'
/// programs. `live` is the card's room for the claim and the holds; the kv
/// pool fills only what is past the holds. Returns `(bodies, programs)`.
///
/// # Errors
/// What the card has past the claim, where the bodies' measured figure is
/// more than that.
pub fn holds(
    claim: u64,
    live: u64,
    bodies: u64,
    programs: u64,
) -> core::result::Result<(u64, u64), u64> {
    let spare = live.saturating_sub(claim);
    let bodies = match bodies {
        MEASURING => spare.saturating_sub(programs),
        bodies if bodies > spare => return Err(spare),
        bodies => bodies,
    };
    Ok((bodies, programs.min(spare - bodies)))
}

/// What the card has left to grow the pool's range by, in whole pages of
/// `page`: its `free` bytes past the activation's span still to back
/// (`unbacked`) and what is `held` beside the pool, counted once.
pub(super) fn growth_room(free: u64, unbacked: u64, held: u64, page: u64) -> u64 {
    free.saturating_sub(unbacked.saturating_add(held)) / page.max(1)
}

/// What a pool takes of the card's room (`live`: free bytes and what the
/// range backs, past the fires' `activation`) where backing it grows the
/// range by `growth` pages of `page` beside the `backed` bytes it holds:
/// the claim [`growth_room`] leaves exactly that growth for, once the card
/// holds its claim and everything past it is held for others.
pub(super) fn claim_of(growth: u64, page: u64, backed: u64, activation: u64) -> u64 {
    growth
        .saturating_mul(page.max(1))
        .saturating_add(backed.saturating_sub(activation))
}

/// The largest token budget at or under `max_tokens`, halving toward
/// `floor`, whose working set (`working`, the activation arena and the
/// attention workspaces at that budget) fits `room`. Halving keeps the
/// bucket lattice a prefix of what the operator asked for.
#[must_use]
pub fn tokens_within(max_tokens: u32, floor: u32, room: u64, working: impl Fn(u32) -> u64) -> u32 {
    let mut tokens = max_tokens;
    while tokens > floor && working(tokens) > room {
        tokens = (tokens / 2).max(floor);
    }
    tokens
}

/// What the decoded-weight tiles take on this load: a plane stored in codes
/// and scales is decoded to bf16 before a prefill's dense gemm reads it,
/// into one tile a stream the size of the widest plane fired on it, rounded
/// up to 8 MiB: a plane up to the tile decodes into it, one past it takes
/// the fused-dequant arm. The tile is a slab laid with the kernels'
/// workspaces.
#[must_use]
pub fn decoded_weight_reserve(widest_plane_bytes: u64, streams: u32) -> u64 {
    const GRAIN: u64 = 8 << 20;
    widest_plane_bytes
        .checked_next_multiple_of(GRAIN)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::from(streams.max(1)))
}

/// The most pages `bytes` of cache rows could hold, the ceiling the pool's
/// address space is reserved for: `bytes` over the least a page adds to the
/// rows. A full row adds the page; a windowed one at most half of it, since
/// its window grows by a page for every two past the first sequence.
pub fn page_ceiling(trace: &Trace, page_size: u32, context: u32, bytes: u64) -> Result<u64> {
    let mut page: u64 = 0;
    for row in &trace.caches {
        if let CacheRow::Kv {
            name,
            planes,
            dtype,
            window,
            ..
        } = row
        {
            let cells = u64::from(page_size) / if window.is_some() { 2 } else { 1 };
            let element = elem_bytes(name, *dtype)?;
            for width in planes {
                page = page.saturating_add(cells * width * element);
            }
        }
    }
    let one_sequence = u64::from(context.div_ceil(page_size.max(1)));
    Ok(match page {
        0 => one_sequence,
        page => (bytes / page).clamp(one_sequence, u64::from(u32::MAX)),
    })
}

/// A model's declared context shortened to what the load's room holds: the
/// longest whole number of pages, at most `declared`, at which one sequence
/// takes no more than half of `live`, the room the load's own room line
/// leaves past the weight tier (read after this process's context and
/// modules took theirs): the activations and a second sequence have the
/// other half. The most a declared context is served at: the fit shortens it
/// to what it seats (`Pools::declare_context`). Never under one page.
pub fn declared_context_within(trace: &Trace, page_size: u32, declared: u32, live: u64) -> Result<u32> {
    let page_size = page_size.max(1);
    let window = window_of(trace)?;
    // A sequence as it is held between fires: a windowed row its window
    // alone; a fire's rows past it are the activations' half, weighed per
    // rung once the rung is ruled (`window_fire_bytes`).
    let one = |pages: u32| -> Result<u64> {
        let paging =
            Paging::of(page_size, pages.saturating_mul(page_size), 1, 1)?.held(window);
        one_slot_bytes(trace, paging)
    };
    let room = live / 2;
    let most = declared.div_ceil(page_size).max(1);
    if one(most)? <= room {
        return Ok(declared);
    }
    let mut refused = None;
    let fits = longest(1, u64::from(most), |pages| match one(pages as u32) {
        Ok(bytes) => bytes <= room,
        Err(why) => {
            refused.get_or_insert(why);
            false
        }
    });
    if let Some(why) = refused {
        return Err(why);
    }
    Ok((fits as u32).saturating_mul(page_size))
}

/// The most of `fits ..= over - 1` that `seats` holds, by bisection: `seats`
/// holds `fits` and no more than it holds below `over`.
pub(super) fn longest(mut fits: u64, mut over: u64, seats: impl FnMut(u64) -> bool) -> u64 {
    let mut seats = seats;
    while over - fits > 1 {
        let mid = fits + (over - fits) / 2;
        if seats(mid) {
            fits = mid;
        } else {
            over = mid;
        }
    }
    fits
}

/// The line a load states when it serves a model's declared context cut
/// to `serving` tokens, `why` the room that cut it.
pub(crate) fn say_context_cut(declared: u32, serving: u32, why: &str) {
    tracing::info!(
        "the model declares a {declared}-token context and {why}; serving \
         {serving} tokens. State `[engine] max_context` to choose."
    );
}

/// The staging ring for a host tier of this rank, known before the load
/// measures the device: a piece for each stage a move's piece passes for
/// each of the `frames` the engine keeps in flight ([`host::staging_ring`]),
/// a piece the largest chunk the pool's recipe may lay, so any shape's slot
/// moves through one whole ([`poem_compiler::chunk::ChunkProfile::range`]);
/// within what the tier may take (the operator's `host` bytes shared by
/// `ranks`, else half of what the host has free, as
/// [`crate::Shell::seat_host_kv`] bounds its default). The ring holds the
/// copies in flight, not a sequence: a move of any size goes through it a
/// piece at a time. `None` where no tier is stated.
pub fn staging_ring_for(host: Option<u64>, ranks: u32, frames: u8) -> Result<Option<RingSize>> {
    let ranks = u64::from(ranks.max(1));
    let free = crate::weights::available_memory() / ranks;
    let budget = match host {
        Some(0) => return Ok(None),
        Some(bytes) => (bytes / ranks).min(free),
        None => free / 2,
    };
    let (_, piece) = poem_compiler::chunk::ChunkProfile::SERVER_CUDA.range();
    Ok(Some(host::staging_ring(piece, frames, budget)))
}

pub fn one_slot_bytes(trace: &Trace, paging: Paging) -> Result<u64> {
    let mut bytes: u64 = 0;
    for row in &trace.caches {
        match row {
            CacheRow::Kv {
                name,
                planes,
                dtype,
                window,
                ..
            } => {
                let element = elem_bytes(name, *dtype)?;
                let pages = match (window, paging.window) {
                    (Some(_), Some(_)) => paging.window_ring(),
                    _ => paging.pages_per_slot,
                };
                let cells = u64::from(pages) * u64::from(paging.page_size);
                for width in planes {
                    bytes = bytes.saturating_add(cells * width * element);
                }
            }
            CacheRow::State {
                name, slab, dtype, ..
            } => {
                let stride: u64 = slab.iter().product();
                bytes = bytes.saturating_add(stride * elem_bytes(name, *dtype)?);
            }
        }
    }
    Ok(bytes)
}

/// What one fire's rows take in the windowed kv rows past the windows they
/// land in: every row a fire writes is resident there at once, so this
/// grows with `max_forward_tokens` as the activation arena does.
pub fn window_fire_bytes(trace: &Trace, paging: Paging) -> Result<u64> {
    let Some(window) = paging.window else {
        return Ok(0);
    };
    let cells = u64::from(window.fire_pages) * u64::from(paging.page_size);
    let mut bytes: u64 = 0;
    for row in &trace.caches {
        if let CacheRow::Kv {
            name,
            planes,
            dtype,
            window: Some(_),
            ..
        } = row
        {
            let element = elem_bytes(name, *dtype)?;
            for width in planes {
                bytes = bytes.saturating_add(cells * width * element);
            }
        }
    }
    Ok(bytes)
}

/// One working set's cache rows: its kv pages at the declared context and
/// its recurrent record, the least `Pools::fit_the_card` seats before
/// refusing.
pub fn least_sequence_bytes(trace: &Trace, paging: Paging) -> Result<u64> {
    one_slot_bytes(trace, paging)
}

pub fn admit_the_card(
    allowance: engine::Allowance,
    weights: u64,
    extra_resident: u64,
    trace: &Trace,
    paging: Paging,
    kv_bytes: Option<u64>,
) -> Result<Accounting> {
    let accounting = Accounting::of(
        device_memory()?.1,
        allowance,
        weight_tier_bytes(trace, weights, extra_resident)?,
        match kv_bytes {
            Some(budget) => budget,
            None => one_slot_bytes(trace, paging)?,
        },
    );
    accounting.admit()?;
    Ok(accounting)
}

/// The bytes this load's weight tier puts on the device: every plane the
/// trace declares, or (`stated`, nonzero) what the load's plan keeps there
/// once the weight cache's planes leave, plus what else rides along
/// resident.
pub fn weight_tier_bytes(trace: &Trace, stated: u64, extra_resident: u64) -> Result<u64> {
    let full: u64 = crate::weights::plane_bytes(trace)?
        .iter()
        .map(|plane| plane.next_multiple_of(crate::weights::ALIGN))
        .sum();
    Ok(match stated {
        0 => full,
        stated => stated.min(full),
    }
    .saturating_add(extra_resident))
}

/// What the bound card may hold in all while a load under `allowance` runs
/// ([`elastic::ceiling_of`]), resolved on the card as it is now, before the
/// load takes anything. What this process took before its loads
/// ([`crate::device::own::before_loads`]) is not held before: a stated
/// budget counts it, and only what other processes and this one's earlier
/// loads hold lies outside it. `latch` says a load is resolving its one
/// ceiling, not a reading ahead of it.
pub fn load_ceiling(allowance: engine::Allowance, latch: bool) -> Result<u64> {
    crate::device::card::stated(allowance.budget);
    if latch
        && allowance.budget.is_none()
        && let Some(terms) = crate::device::card::integrated()?
    {
        let mib = |bytes: u64| bytes >> 20;
        tracing::info!(
            "integrated device: capacity {} MiB = min(dedicated {} MiB - held {} MiB, Windows commit available within physical memory {} MiB), less the driver's {} MiB floor; the system memory the driver lends it is not counted (state [engine.memory] budget_device to size by it)",
            mib(terms.free()),
            mib(terms.dedicated),
            mib(terms.held),
            mib(terms.commit_available),
            mib(elastic::safety_floor_bytes()),
        );
    }
    let (free, total) = device_memory()?;
    let held = (total - free).saturating_sub(crate::device::own::before_loads(latch)?);
    Ok(elastic::ceiling_of(total, held, allowance))
}

/// The device's free and total bytes, as the driver reports them now.
pub fn device_memory() -> Result<(u64, u64)> {
    crate::device::card::memory()
}

impl Pools {
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.chunks.bytes()
    }

    /// The watermark the pool reaches: the chunks the fit left it, as the
    /// map units the range backs them in.
    #[must_use]
    pub fn declared_bytes(&self) -> u64 {
        self.chunk_bytes(self.fitted)
    }

    /// The bytes `chunks` chunks take backed: the last map unit whole.
    pub(crate) fn chunk_bytes(&self, chunks: u32) -> u64 {
        let unit = self.chunks.map_unit().max(1);
        u64::from(chunks)
            .saturating_mul(self.chunks.layout().chunk_bytes)
            .div_ceil(unit)
            .saturating_mul(unit)
    }

    #[must_use]
    pub fn map_unit_bytes(&self) -> u64 {
        self.pool.map_unit_bytes()
    }

    /// The map units the pool may still take beside the bodies while they
    /// arm: none once the load backed it whole.
    #[must_use]
    pub fn arming_margin_bytes(&self) -> u64 {
        if self.whole {
            0
        } else {
            self.pool.map_unit_bytes().saturating_mul(16)
        }
    }

    /// What the cache rows may still take of the device: the bytes the fit
    /// declared them, or the most their segments ever backed, past what
    /// they back now. The activation's held spans are no part of it: they
    /// are backed already, out of what the device has free.
    pub(super) fn segment_reserve(&self) -> u64 {
        self.declared_bytes()
            .max(self.chunks.segment_high_water_bytes())
            .saturating_sub(self.committed_bytes())
    }

    pub fn spare_bytes(&self) -> Result<u64> {
        self.pool.spare_bytes(self.segment_reserve())
    }

    /// What the bodies may still take while the lattice arms: the device's
    /// free bytes past the cache rows' reserve, or `room` where the device
    /// has that free.
    pub fn spare_bytes_for_bodies(&self, room: u64) -> Result<u64> {
        let live = self.pool.spare_bytes(0)?;
        Ok(live
            .saturating_sub(self.segment_reserve())
            .max(live.min(room)))
    }

    /// The kv pages `chunks` chunks hold beside the weights they seat
    /// whole, as the fit counts them.
    pub(super) fn pages_within(&self, chunks: u32) -> u64 {
        let layout = self.chunks.layout();
        let weights =
            u32::try_from(self.weight_pool_bytes() / layout.chunk_bytes.max(1)).unwrap_or(u32::MAX);
        let kv_chunks = chunks.saturating_sub(if weights <= chunks { weights } else { 0 });
        pages_in(layout, &self.paging, kv_chunks, 0, 1, self.ceiling)
    }

    /// What backing the pool's first `chunks` chunks takes of the card, as
    /// the fit counts the room it grows by: the pages the range would back
    /// for them anew (whole map units of its segments as `placed` lays
    /// them, [`chunks::Chunks::growth_for`]), beside what it backs already
    /// but for the `activation` bytes of it that are the fires'. A flat
    /// range is one allocation in chunk-sized pieces: exactly their bytes.
    pub(super) fn backing_bytes(&self, chunks: u32, placed: bool, activation: u64) -> Result<u64> {
        if !elastic::mapped_on_demand() {
            return Ok(self.chunk_bytes(chunks));
        }
        let growth = self
            .chunks
            .growth_for(chunks, placed.then_some(self.contiguous))?;
        Ok(claim_of(
            growth,
            self.pool.page_bytes(),
            self.chunks.backed_bytes(),
            activation,
        ))
    }

    /// Pledges the pool the chunks the runtime is stated, beside what its
    /// other arenas hold now.
    pub(super) fn pledge(&mut self) {
        let chunks = self
            .stated
            .map_or(self.fitted, |stated| stated.max(self.fitted));
        let others = self
            .pool
            .committed_pages()
            .saturating_sub(elastic::pages_for_bytes(self.committed_bytes()));
        self.pool
            .pledge(elastic::pages_for_bytes(self.chunk_bytes(chunks)).saturating_add(others));
    }

    /// The bytes of the chunks stated to the runtime that the fit does not
    /// hold for the cache rows now. None but on a load that arms its bodies
    /// at the starting weight residency, after it stated its pool
    /// ([`Pools::state_whole`]): every other load states the chunks its
    /// last fit backed ([`Pools::fit_and_state`]).
    #[must_use]
    pub fn kept_back_bytes(&self) -> u64 {
        self.stated.map_or(0, |stated| {
            self.chunk_bytes(stated)
                .saturating_sub(self.chunk_bytes(self.fitted_chunks()))
        })
    }

    /// The activation's span no fire has backed.
    #[must_use]
    pub fn activation_unheld_bytes(&self) -> u64 {
        self.chunks.activation_unheld_bytes()
    }

    /// Whether the range is stated whole, past what the fit holds
    /// ([`Pools::state_whole`]).
    #[must_use]
    pub fn stated_whole(&self) -> bool {
        self.stated.is_some()
    }

    /// The range stays where it lies from here on: a body captured over it
    /// names its addresses.
    pub(crate) fn pin(&mut self) {
        self.pinned = true;
    }

    /// What a load's synthetic fires may name, inside the chunks the fit
    /// left the pool: of `records` recurrent records and `pages` kv pages
    /// wanted, the most records beside every page, or one record and the
    /// most pages; and the windowed pages beside those pages, the null page
    /// among them. A synthetic lane's rows are never read back, so lanes
    /// may share what they name; the fit holds one working set, so one
    /// record and its pages always fit.
    #[must_use]
    pub fn synthetic_room(&self, records: u32, pages: u64) -> (u32, u64, u64) {
        let fitted = self.fitted_chunks();
        let room = fitted.saturating_sub(self.chunks.weight_chunks_under(fitted));
        let layout = self.chunks.layout();
        let state = self.has_record();
        let fits = |records: u32, pages: u64| {
            chunks_at(layout, &self.paging, pages, if state { records } else { 0 }) <= room
        };
        let largest = |lo: u64, hi: u64, ok: &dyn Fn(u64) -> bool| {
            let (mut lo, mut hi) = (lo, hi);
            while lo < hi {
                let mid = lo + (hi - lo).div_ceil(2);
                if ok(mid) { lo = mid } else { hi = mid - 1 }
            }
            lo
        };
        let records = records.max(1);
        let pages = pages.max(1);
        let (records, pages) = if fits(records, pages) {
            (records, pages)
        } else if fits(1, pages) {
            let most = largest(1, u64::from(records), &|r| fits(r as u32, pages));
            (u32::try_from(most).unwrap_or(1), pages)
        } else {
            (1, largest(1, pages, &|p| fits(1, p)))
        };
        (records, pages, self.paging.window_pages_at(pages))
    }

    /// The model's declared context stands unstated by the operator: the fit
    /// before arming shortens it to what it seats.
    pub fn declare_context(&mut self) {
        self.declared = true;
    }

    /// What the fit takes of `live` for one working set of `pages` kv pages
    /// beside the null chunk (its kv pages, its windowed pages and its
    /// recurrent record), and what it holds for the bodies and the guests'
    /// programs beside it.
    /// Errs with what the card has past the claim where that is under the
    /// bodies' measured figure ([`holds`]).
    /// `stated` is what backing a stated pool's chunks takes of the card
    /// ([`Pools::backing_bytes`]): its claim, in place of their bytes.
    pub(super) fn working_set(
        &self,
        pages: u64,
        bodies: u64,
        programs: u64,
        live: u64,
        stated: Option<u64>,
    ) -> core::result::Result<WorkingSet, u64> {
        let slots = u32::from(self.has_record());
        let least = chunks_at(self.chunks.layout(), &self.paging, pages, slots);
        let need = self.chunk_bytes(least);
        // A stated budget is the pool's before the bodies and programs are
        // held anything; with none, one working set is, and beside it every
        // weight the pool would hold wherever the card holds them with the
        // programs (the activation's span already out of `live`): the
        // bodies are held out of what is left after the weights, so their
        // allowance never turns weights that fit into weights that stream.
        // A card short of that streams them (the runtime's call).
        let whole = need.saturating_add(self.weight_pool_bytes());
        let claim = stated
            .or(self.budget)
            .unwrap_or(if live >= whole.saturating_add(programs) {
                whole
            } else {
                need
            });
        let (bodies, programs) = holds(claim, live, bodies, programs)?;
        let room = live.saturating_sub(programs.saturating_add(bodies));
        // Whole map units of the room, in whole chunks, of the range's.
        let unit = self.chunks.map_unit().max(1);
        Ok(WorkingSet {
            least,
            need,
            bodies,
            programs,
            room,
            // Stated tokens are exactly their chunks, never a map unit's
            // round number of them.
            limit: match (self.stated_kv, self.budget) {
                (Some(_), Some(budget)) => budget,
                (_, budget) => budget.unwrap_or(room) / unit * unit,
            },
        })
    }

    /// Sizes the pool to the card: every whole chunk the card has room for,
    /// each any shape's to take, provided they seat one working set at the
    /// declared context ([`engine::chunk::fit`]). The room is the elastic
    /// budget as it stands now, so the fit is called once everything else
    /// this load puts on the device is resident, and again after arming,
    /// when the bodies have taken what they take and the rest is the pool's.
    /// `programs` is held for the guests' programs throughout; `bodies` is
    /// held ahead of the kv pool's fill ([`holds`]): what they took where
    /// that is measured, refused by name where the card does not hold it
    /// beside one working set, and everything past that working set where
    /// they are still to arm ([`MEASURING`]). `resident` spells what is on
    /// the device already, for the refusal when not even that fits.
    /// `settled` says the activation has reached its high water: every
    /// composition fired while the bodies armed.
    ///
    /// This fit claims the pool's room and states nothing: before the
    /// bodies arm it is what arming's fires seat their rows in and what
    /// the bodies are kept out of, and the load's last fit is
    /// [`Pools::fit_and_state`]. Only a load that arms after it stated its
    /// pool keeps this one's chunks, and settles them with another.
    pub fn fit_the_card(
        &mut self,
        bodies: u64,
        programs: u64,
        settled: bool,
        resident: &dyn core::fmt::Display,
    ) -> Result<Fitted> {
        let whole = self.stated_whole();
        self.fit(bodies, programs, settled, false, whole, resident)
    }

    /// [`Pools::fit_the_card`] for a pool then stated whole
    /// ([`Pools::state_whole`]): the fit leaves the card room to back the
    /// state's chunks in the segment the activation starts in beside the
    /// chunks it fits, since the runtime counts that segment committed from
    /// the load ([`chunks::Chunks::hold_boundary`]).
    pub fn fit_to_state_whole(
        &mut self,
        bodies: u64,
        programs: u64,
        resident: &dyn core::fmt::Display,
    ) -> Result<Fitted> {
        let fitted = self.fit(bodies, programs, false, false, true, resident)?;
        self.state_whole();
        Ok(fitted)
    }

    /// The load's last fit, once everything it puts on the device is there
    /// (its bodies captured, where it records any): the pool is the chunks
    /// the card backs now, and the range is stated as exactly those
    /// ([`chunks::Chunks::state_fitted`]). The range was reserved for every chunk
    /// the card could have backed, since a body names its base and the
    /// activation above it and neither may move; the chunks it holds past
    /// this fit are skipped, named by no chunk id and never backed. So the
    /// chunks stated are the chunks [`Pools::back_whole`] backs, whatever
    /// the bodies took: nothing is held back, and nothing was held for a
    /// figure the bodies did not use.
    ///
    /// `bodies` is held out of it only for a load whose bodies arm after
    /// it has stated its pool and that states no more than it backs.
    pub fn fit_and_state(
        &mut self,
        bodies: u64,
        programs: u64,
        settled: bool,
        resident: &dyn core::fmt::Display,
    ) -> Result<Fitted> {
        // What the load's own fires seated past where the pool parts goes
        // back first: the fit reads the card without it.
        self.chunks
            .clear_past_parting(&mut self.pool, self.contiguous)?;
        let fitted = self.fit(bodies, programs, settled, true, false, resident)?;
        self.chunks
            .state_fitted(self.fitted_chunks(), self.contiguous)?;
        Ok(fitted)
    }

    /// The pool's chunks stay in one run from the range's base, where
    /// another worker reads its pages in place ([`Pools::export_kv`]): the
    /// chunks skipped past the fit are those right under the activation,
    /// and no rung lends its span to state.
    pub fn keep_contiguous(&mut self) {
        self.contiguous = true;
    }

    /// `whole`: the pool is stated whole past this fit, so the state's
    /// chunks of the segment the activation starts in are backed whatever
    /// it fits ([`chunks::Chunks::boundary_unbacked_bytes`]).
    pub(super) fn fit(
        &mut self,
        bodies: u64,
        programs: u64,
        settled: bool,
        placed: bool,
        whole: bool,
        resident: &dyn core::fmt::Display,
    ) -> Result<Fitted> {
        // What the load dropped and the process keeps for reuse is free
        // room, not the card's use.
        crate::device::held::release_idle();
        // The room is what the card has free plus every byte the range has
        // backed (its committed segments and held spans; a flat range, once
        // allocated, the whole of it: the chunks the kv rows have not yet
        // taken are backed too, and so not free), less the activation the
        // cache rows never take. Until the activation is settled that is its
        // whole span, held or still to be (the first fire's activation, and
        // the room held for the bodies, must find it); once settled (every
        // composition armed, the activation at its high water), only the
        // spans held backed in it (the arena's prefix a fire backed, the
        // carves laid at its end, the workspaces, the staged runs): what
        // nothing holds is no fire's, and the cache rows take it. A flat
        // range's one allocation backs the whole span.
        let activation = if settled && elastic::mapped_on_demand() {
            self.chunks.activation_held_bytes()
        } else {
            self.chunks.activation_bytes()
        };
        // The card's free bytes, read once: the room and what backing the
        // chunks fitted may grow by are counted from the same figure.
        let free = self.pool.spare_bytes(0)?;
        let live = free
            .saturating_add(self.chunks.backed_bytes())
            .saturating_sub(activation);
        // The activation's span no fire has backed yet, still to come out
        // of what the card has free while the bodies arm.
        let unbacked = if settled {
            0
        } else {
            self.chunks.activation_unheld_bytes()
        };
        let layout = self.chunks.layout();
        let chunk = layout.chunk_bytes;
        // A stated pool claims what backing its chunks takes of the card,
        // counted as the growth below counts it, not their bytes alone.
        let stated = match self.stated_kv {
            Some(kv) => Some(self.backing_bytes(
                kv.chunks(),
                placed,
                activation.saturating_sub(unbacked),
            )?),
            None => None,
        };
        // One working set at the declared context beside the null chunk:
        // its kv pages, its windowed pages and its recurrent record.
        let mut one_slot = u64::from(self.paging.pages_per_slot).min(self.ceiling);
        // A declared context is the model's, not the operator's: where this
        // fit does not seat one working set at it beside the programs (and a
        // recorded figure for the bodies), it is served at the longest the
        // fit does.
        if self.declared && !settled {
            let seats = |pages: u64| {
                let Ok(set) = self.working_set(pages, bodies, programs, live, stated) else {
                    return false;
                };
                set.programs >= programs
                    && engine::chunk::fit(set.limit, chunk, 0, set.least).is_ok()
            };
            // Never down to a window the rows are read through.
            let shortest = self.paging.window.map_or(1, |window| {
                u64::from(window.tokens / self.paging.page_size.max(1)) + 1
            });
            if shortest < one_slot && !seats(one_slot) && seats(shortest) {
                let fits = longest(shortest, one_slot, seats);
                let declared = self.paging.context();
                one_slot = fits;
                self.paging.pages_per_slot = u32::try_from(fits).unwrap_or(u32::MAX);
                say_context_cut(
                    declared,
                    self.paging.context(),
                    "with everything else this load puts on the device resident, the card does \
                     not seat one sequence at it beside the guests' programs and the graph \
                     bodies' record",
                );
            }
        }
        // A declared context the fit cannot seat even shortened is the
        // card's refusal by name, not a residency fault: the load asked the
        // model's context, and the card holds no sequence of it.
        let declared = self.declared;
        let short = |why: String| if declared { Fault::Card(why) } else { Fault::Residency(why) };
        let WorkingSet {
            least,
            need,
            bodies,
            programs,
            room,
            limit,
        } = self
            .working_set(one_slot, bodies, programs, live, stated)
            .map_err(|spare| {
                short(format!(
                    "the card does not hold this deployment's graph bodies: {resident}; past \
                     one working set at the declared context and the weights the pool seats it \
                     has {spare} bytes, and the bodies took {bodies} when this deployment last \
                     armed whole. Lower `[engine] max_context`, raise `[engine.memory] \
                     budget_device`, or free what else holds the device."
                ))
            })?;
        let held = programs.saturating_add(bodies);
        if let (Some(kv), Some(backing)) = (self.stated_kv, stated) {
            kv.admit(room, backing, self.chunks.map_unit())?;
        } else if let Some(budget) = self.budget.filter(|&budget| budget > room) {
            return Err(Fault::Residency(
                engine::fit::Short::Budget { budget, room }.to_string(),
            ));
        }
        // Before arming a fresh range may be reserved anew up to the asked
        // count. Once armed it never moves, fresh or not: the bodies name
        // addresses in it.
        let most = if !settled && !self.pinned && self.chunks.fresh() {
            self.asked_chunks.max(self.chunks.total())
        } else {
            self.chunks.total()
        };
        let fitted = match engine::chunk::fit(limit, chunk, 0, least) {
            Ok(chunks) => chunks.min(most),
            Err(_) if self.budget.is_some() => {
                return Err(Fault::Residency(
                    engine::fit::Short::Sequence {
                        budget: self.budget.unwrap_or(0),
                        least: need,
                    }
                    .to_string(),
                ));
            }
            Err(_) => {
                return Err(short(format!(
                    "the card does not hold this deployment: {resident}; under the load's \
                     allowance that leaves {room} bytes for the state pool, and one working set \
                     at the declared context needs {need}. Lower `[engine] max_context`, raise \
                     `[engine.memory] budget_device`, serve a quantized deployment of the \
                     model, or free what else holds the device."
                )));
            }
        };
        // The room counts every backed byte as the pool's to reuse, and a
        // span held elsewhere in the range (the boundary of an earlier fit,
        // a workspace) is not under the chunks fitted. What backing these
        // chunks really asks of the card is counted from the range itself,
        // and the fit is the most chunks whose growth the card holds beside
        // the activation still to back and what is held for the bodies.
        let fitted = if elastic::mapped_on_demand() {
            let page = self.pool.page_bytes().max(1);
            // In whole pages of what is left, counted once: pages of each
            // part rounded apart would come a page short of a claim that
            // is exactly its growth.
            let have = growth_room(free, unbacked, held, page);
            let mut fitted = fitted;
            while fitted > least {
                let boundary = if whole {
                    self.chunks.boundary_unbacked_bytes(fitted).div_ceil(page)
                } else {
                    0
                };
                let growth = self
                    .chunks
                    .growth_for(fitted, placed.then_some(self.contiguous))?
                    .saturating_add(boundary);
                if growth <= have {
                    break;
                }
                let over = (growth - have).saturating_mul(page).div_ceil(chunk);
                fitted = fitted.saturating_sub(u32::try_from(over).unwrap_or(u32::MAX).max(1));
            }
            fitted.max(least)
        } else {
            fitted
        };
        self.fitted = fitted;
        // The kv pages the shell's own seats may name: what the chunks hold
        // were every one the kv rows', but those the weights take. Weights
        // the fitted chunks seat whole the runtime holds for the load's life
        // (`store::weights::boot_cache`), whatever the claim was, a stated
        // budget's too, so no kv row ever takes their chunks; counted as the
        // rows', they are pages admission hands out that no chunk backs, and
        // a load filling them starves with nothing to discard. Where they
        // leave less than one working set at the declared context, the
        // pages are what they leave.
        let weight_chunks = u32::try_from(self.weight_pool_bytes() / chunk).unwrap_or(u32::MAX);
        let weights_held = if weight_chunks <= fitted {
            weight_chunks
        } else {
            0
        };
        let kv_chunks = fitted.saturating_sub(weights_held);
        let layout = self.chunks.layout();
        let floor = if kv_chunks >= least {
            one_slot
        } else {
            if settled {
                tracing::info!(
                    "the weights this card holds whole leave the cache rows {} MiB, \
                     under one sequence at the declared context ({} MiB): the kv pool states \
                     what they leave, and live demand takes chunks from the weight cache as it \
                     needs them. Lower `[engine] max_context` or raise `[engine.memory] \
                     budget_device` to hold more.",
                    (u64::from(kv_chunks) * chunk) >> 20,
                    need >> 20,
                );
            }
            1
        };
        // A stated pool's pages are counted beside the records it was
        // sized with, as [`StatedKv::of`] counted them.
        let records = self.stated_kv.map_or(0, |kv| kv.records);
        let fits = pages_in(
            layout,
            &self.paging,
            kv_chunks,
            records,
            floor,
            self.ceiling,
        );
        // A pool stated for more than this fit holds keeps the pages of
        // what it stated: the runtime names them, and a count that moved
        // under live tables would not be the one they were laid by.
        if let Some(kv) = self.stated_kv.filter(|kv| kv.pages != fits) {
            let backing = stated.unwrap_or(kv.bytes());
            return Err(kv.short(room, backing, self.chunks.map_unit(), Some(fits)));
        }
        if self.stated.is_none() {
            self.paging.pages = fits;
        }
        // A flat range is one allocation made on its first commit: of the
        // fitted chunks, not of every chunk the asked count reserved.
        // A range mapped on demand only reserves addresses, and a body
        // recorded over it names its base and the activation above it, so
        // it is reserved before the bodies arm for every chunk the card
        // could back were nothing more to arrive: what it has free and
        // what the range has backed, less the activation's prefix a fire
        // backed. The load's last fit backs what the card then has and
        // skips the rest ([`Pools::fit_and_state`]); reserved to this fit
        // alone, that one could never grow past it.
        let flat = !elastic::mapped_on_demand();
        let reserved = if flat || settled {
            fitted
        } else {
            let backable = self
                .pool
                .spare_bytes(0)?
                .saturating_add(self.chunks.backed_bytes())
                .saturating_sub(
                    self.chunks
                        .activation_bytes()
                        .saturating_sub(self.chunks.activation_unheld_bytes()),
                )
                .div_ceil(chunk);
            u32::try_from(backable)
                .unwrap_or(u32::MAX)
                .max(fitted)
                .min(most)
        };
        let reshape = flat || (self.whole_fit && reserved != self.chunks.total());
        // Never once a body names the range ([`Pools::pin`]): reshaped, its
        // planes would lie where the bodies captured over it no longer look.
        if reshape && !self.pinned && self.chunks.fresh() {
            self.chunks.reshape(&mut self.pool, reserved)?;
        }
        self.chunks
            .limit(self.stated.map_or(fitted, |stated| stated.max(fitted)));
        // The runtime admits frames against this watermark, so the pool is
        // pledged it beside what its other arenas hold now: a frame inside
        // it is a frame the fit already sized the card for, and what a fire
        // allocates beside the pool later is not its refusal to answer.
        self.pledge();
        Ok(Fitted {
            asked: self.asked,
            fit: fits,
            room,
            held,
            bodies,
            spare: live.saturating_sub(need).saturating_sub(programs),
            chunks: fitted,
        })
    }

    /// Whether no segment is backed yet: a fit before arming may still
    /// reserve the range anew (its held activation laid again).
    #[must_use]
    pub fn fresh(&self) -> bool {
        self.chunks.fresh()
    }

    /// Where a fire's activation starts: under the state pool's end.
    #[must_use]
    pub fn activation_base(&self) -> u64 {
        self.chunks.activation_base()
    }

    /// Backs the first `bytes` of the activation's span, beside the pool's
    /// budget where `beside` and the budget refuses them.
    pub(crate) fn hold_activation(&mut self, bytes: u64, beside: bool) -> Result<()> {
        match self.chunks.hold_activation(&mut self.pool, bytes, beside)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    /// Reserves the range anew to the fit even where it maps on demand. The
    /// chunks it keeps past the settled fit, under the activation, are
    /// stated to no runtime ([`chunks::Chunks::skip_past`]).
    pub fn fit_range_whole(&mut self) {
        self.whole_fit = true;
    }

    /// The bytes of the activation's span under the pool's end.
    #[must_use]
    pub fn activation_bytes(&self) -> u64 {
        self.chunks.activation_bytes()
    }

    /// Reserves the range anew with `bytes` more of activation
    /// ([`Chunks::widen_activation`]).
    pub(crate) fn widen_activation(&mut self, bytes: u64) -> Result<u64> {
        self.chunks.widen_activation(&mut self.pool, bytes)
    }

    /// Backs `bytes` of the activation's span from `offset` into it.
    pub(crate) fn hold_activation_at(
        &mut self,
        offset: u64,
        bytes: u64,
        beside: bool,
    ) -> Result<()> {
        match self
            .chunks
            .hold_activation_at(&mut self.pool, offset, bytes, beside)?
        {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.chunks.committed_bytes()
    }

    /// Every byte the range has backed: on a flat range, all of it.
    #[must_use]
    pub fn backed_bytes(&self) -> u64 {
        self.chunks.backed_bytes()
    }

    /// The bytes of the activation's span, under the range's end.
    #[must_use]
    pub fn activation_span_bytes(&self) -> u64 {
        u64::from(self.chunks.act_chunks()) * self.chunks.layout().chunk_bytes
    }

    #[must_use]
    pub fn high_water_bytes(&self) -> u64 {
        self.chunks.high_water_bytes()
    }

    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.shapes
            .iter()
            .map(|shape| match *shape {
                Shape::State { stride, dtype, .. } => stride * u64::from(elem_size(dtype)),
                Shape::Kv { .. } => 0,
            })
            .sum::<u64>()
    }
}
