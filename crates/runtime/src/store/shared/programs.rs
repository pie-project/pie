//! What a bound guest program asks of the activation span beside the
//! model's carve (REDESIGN §6): nothing of its own. A program's scratch is
//! state ([`::engine::GuestState`]): each instance's lanes' scratch and its
//! (program, extents) pair's, whole `Instance` chunks taken at the bind and
//! held while bound ([`super::super::roles`]), refused or waited on at the
//! bind like any state and counted by admission. A rung's activation is the
//! model's carve, the rows a fire reads out where they land beside it, and
//! the engine's standing guest bytes ([`super::super::slots::SlotPool::act_chunks`]).
//!
//! The engine keeps standing guest bytes in the span for the live
//! instances of registered programs, whatever the fire (Metal: the batch
//! tables it lays once a set, every set together;
//! [`::engine::Engine::standing_after`]). They change only at a bind and a
//! close, a set shared by instances counted once. A bind asks the engine,
//! on the lane, what the total would be once it is bound, and holds what
//! that adds to the total booked ([`ProgramOpen`]) while it is under way:
//! a total that does not fit the span at the smallest rung refuses the bind
//! by name before the engine is asked to bind, and one that does is backed
//! below the carve, at every rung, before the bind is sent ([`fund_span`]),
//! so the engine lays it in room set aside. The engine's total after each
//! bind and each close is booked on the lane
//! ([`SharedPool::standing_stated`]). Programs are never unregistered, but
//! a set that no live instance shares any more is given back at the close
//! that leaves it so.

use ::engine::{BindExtents, ProgramId};

use super::{RungRaise, Shared, SharedPool};

/// A program bound at given extents: the pair whose state
/// ([`::engine::GuestState::per_pair`]) its instances share.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FireKey {
    pub program: ProgramId,
    pub extents: BindExtents,
}

impl std::fmt::Display for FireKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "program {} at {:?}", self.program, self.extents)
    }
}

/// The engine's standing guest bytes and its readout store, as the binds
/// read them.
#[derive(Debug, Default)]
pub struct ProgramFires {
    /// The engine's standing guest bytes as it last stated them, after a
    /// bind or a close; and what the binds under way may add to them, each
    /// held by its [`ProgramOpen`].
    standing_booked: u64,
    standing_pending: u64,
    /// The rows the engine's fixed readout store holds
    /// ([`::engine::ReadoutFacts::store_rows`]): a bind whose lane alone
    /// reads out more is refused ([`SharedPool::readouts_refused`]).
    readout_store: Option<u32>,
}

impl std::fmt::Display for ProgramFires {
    /// The standing bytes booked and pending: what a refused submission is
    /// read against ([`impossible_submission`]).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "standing {} booked, {} pending",
            self.standing_booked, self.standing_pending
        )
    }
}

/// The engine called a submission impossible
/// ([`::engine::Error::Impossible`]): a fire asked more of the span than
/// the engine backs, which the rung arithmetic is there to rule out. One
/// line with what that arithmetic held when it came back: the pool's rung
/// and the one the engine holds, the cap fires are cut to, the activation
/// reserved, the rung's own, its readout and standing bytes, the span the
/// engine states and the most a rung may take under it.
pub fn impossible_submission(engine: usize, error: &::engine::Error) {
    let Some(shared) = crate::store::registry::try_get(0, engine).and_then(|stores| stores.shared)
    else {
        return;
    };
    let pool = shared.lock().unwrap();
    let rung = pool.pool().rung();
    let span = pool
        .pool()
        .act_span()
        .map_or_else(|| "none stated".to_owned(), |chunks| chunks.to_string());
    tracing::error!(
        engine,
        %error,
        rung,
        told = pool.told(),
        cap = super::fire_cap(engine),
        act_reserved = pool.pool().act_reserved(),
        act_chunks = pool.pool().act_chunks(rung),
        readout_bytes = pool.pool().readout_bytes(pool.pool().rung_of(rung)),
        standing_bytes = pool.pool().standing(),
        act_span_chunks = %span,
        act_most = pool.act_most(),
        guests = %pool.programs,
        "the engine called a submission impossible"
    );
}

impl SharedPool {
    /// The engine's readout facts ([`::engine::Engine::readout_facts`]),
    /// with its own ceiling on the rows a fire reads out
    /// (`FireLimits::max_readouts`, 0: none): every rung counts the rows a
    /// fire of it reads out where they land beside the activation, so fires are cut to the widest rung that holds
    /// them; and a fixed store's rows bound every fire's sequence readouts
    /// and refuse a sequence lane that alone reads more
    /// ([`Self::readouts_refused`]). With `float_lanes` (the model has a
    /// class that reads no kv: an encoder's, an embedding's), a fire may
    /// read out every token it holds through the landing, past the store
    /// and the ceiling, so the term counts up to the rung's tokens.
    pub fn set_readouts(
        &mut self,
        facts: &::engine::ReadoutFacts,
        ceiling: u32,
        float_lanes: bool,
    ) {
        let rows = if float_lanes {
            0
        } else {
            facts.fire_rows(ceiling)
        };
        self.pool.set_readouts(facts.row_bytes(), rows);
        self.programs.readout_store = facts.store_rows;
    }

    /// Why a bind of `key` is refused for its readouts: one lane of it
    /// reads out `extents.sampled_rows` rows, more than the engine's fixed
    /// readout store holds, so no fire, however narrow, seats it. `None`
    /// where it fits, or no store bounds them.
    ///
    /// What the bind guards, and what it does not: every fire is of a bound
    /// instance (`pipeline::fire`), and a sequence pass reads out the rows
    /// of its program's `Readout` port (`pipeline::fire::geometry`, its
    /// last row a lane where the port is absent), no more than the port's
    /// elements, which are `sampled_rows` (`pipeline::program::price`). A
    /// float pass (an encoder's, an embedding's) reads out every row it
    /// embeds (`pipeline::fire::float`), which its extents do not state
    /// (`sampled_rows` is then its lanes, or its outputs' rows); those rows
    /// are no sequence readouts and land through the engine's readout
    /// landing whatever the store holds, so its bind is not asked
    /// ([`ProgramOpen::float`]) and its fires are neither cut nor refused
    /// against the store (`FireRequest::sequence_readouts`). A sequence
    /// request that came past its bind is named at admission
    /// (`AdmissionLimits`).
    #[must_use]
    pub fn readouts_refused(&self, key: &FireKey) -> Option<String> {
        let rows = self.programs.readout_store?.max(1);
        (key.extents.sampled_rows > rows).then(|| {
            format!(
                "{key}'s lanes read out {} rows each, more than the {rows} rows the \
                 engine's readout store holds for a fire",
                key.extents.sampled_rows
            )
        })
    }

    /// The chunks a rung's activation may take at most, its readout and
    /// standing bytes included: the top rung's span (`act_chunks_max`, below which live
    /// state lies at every rung, so no raise ever moves it), and no more
    /// than the engine states its span covers as it stands
    /// (`ChunkFacts::act_span_chunks`), read here at each fit: a span the
    /// engine states anew ([`SlotPool::set_act_span`]) counts from the
    /// next. Fires are cut to the widest rung whose activation fits it
    /// ([`Self::widest_fit`]).
    #[must_use]
    pub fn act_most(&self) -> u32 {
        let span = self.pool.act_max();
        self.pool.act_span().map_or(span, |stated| stated.min(span))
    }

    /// Whether every rung fits unasked: nothing stands or reads out beside
    /// the carve and the engine states no span of its own, so a rung's activation is the
    /// model's, which the pool was built to hold.
    fn fits_unasked(&self) -> bool {
        self.pool.guests_zero() && self.pool.act_span().is_none()
    }

    /// The widest rung whose activation, its readout rows and the
    /// standing bytes included, is no more than a rung may take
    /// ([`Self::act_most`]); `None` where even the smallest is. Without
    /// either or a stated span, the largest. No rung past it is set aside:
    /// every goal a raise is planned for is cut to it
    /// ([`Self::rung_for`], [`super::RungWalk::step`]).
    #[must_use]
    pub fn widest_fit(&self) -> Option<u32> {
        let rungs = self.pool.rungs();
        if self.fits_unasked() {
            return rungs.last().copied();
        }
        let most = self.act_most();
        rungs
            .into_iter()
            .filter(|&rung| self.pool.act_chunks(rung) <= most)
            .max()
    }

    /// Whether the engine's standing guest bytes leave no rung room in the
    /// top rung's span: never where nothing stands. A pool stating no rungs
    /// is one rung, its span the room its load backs
    /// ([`super::super::slots::SlotPool::new`]), and is refused the same way.
    fn standing_refused(&self) -> bool {
        self.pool.standing() > 0 && self.widest_fit().is_none()
    }

    /// The widest rung fires may take: [`Self::widest_fit`], the smallest
    /// where none fits (a bind is refused then), `u32::MAX` without rungs.
    pub(super) fn fit_rung(&self) -> u32 {
        if self.fits_unasked() {
            return u32::MAX;
        }
        self.widest_fit()
            .or_else(|| self.pool.rungs().first().copied())
            .unwrap_or(u32::MAX)
    }

    /// The cap fires may take at a rung of `rung` tokens: `u32::MAX` at the
    /// largest.
    #[must_use]
    pub fn lifted(&self, rung: u32) -> u32 {
        if self
            .pool
            .rungs()
            .last()
            .is_none_or(|&largest| rung >= largest)
        {
            u32::MAX
        } else {
            rung
        }
    }

    /// The widest cap fires may take: the widest rung that fits, lifted;
    /// `u32::MAX` where nothing stands or reads out beside the carve.
    #[must_use]
    pub fn fit_cap(&self) -> u32 {
        self.lifted(self.fit_rung())
    }

    /// What a bind that would leave the engine's standing guest bytes at
    /// `total` ([`::engine::Engine::standing_after`], asked on the lane)
    /// adds to the total booked: what its [`ProgramOpen`] holds. Read on
    /// the lane, where the booked total is the engine's as it stands.
    #[must_use]
    pub fn standing_added(&self, total: u64) -> u64 {
        total.saturating_sub(self.programs.standing_booked)
    }

    /// The engine stated its standing guest bytes after a bind or a close
    /// (`BoundInstance::standing_guest_bytes`, `Engine::close_instance`):
    /// booked as the engine's, beside what the binds under way hold.
    pub fn standing_stated(&mut self, total: u64) {
        self.programs.standing_booked = total;
        self.sync_standing();
    }

    /// The engine's standing guest bytes booked, as it last stated them.
    #[must_use]
    pub fn standing_booked(&self) -> u64 {
        self.programs.standing_booked
    }

    /// The span sets aside the booked total and what every bind under way
    /// may add to it.
    fn sync_standing(&mut self) {
        let standing = self
            .programs
            .standing_booked
            .saturating_add(self.programs.standing_pending);
        self.pool.set_standing(standing);
    }

    /// Why a bind of `key` is refused: the smallest rung's activation with
    /// the standing bytes the bind would leave, read while the pool holds
    /// them.
    fn refusal(&self, key: FireKey) -> String {
        let most = self.act_most();
        let smallest = self.pool.act_chunks(0);
        let of = match self.pool.act_span() {
            Some(span) if span <= most => {
                format!("{span} chunks the engine's activation span covers")
            }
            _ => format!("{most} of the top rung's span"),
        };
        format!(
            "{key}'s bind would leave the engine's standing guest bytes at {} beside \
             every fire, and the smallest rung would then take {smallest} chunks of the {of}",
            self.pool.standing()
        )
    }

    /// The chunks the activation lacks for fires of `rows` (the rung that
    /// holds them, its readout and standing bytes included); 0 where it
    /// holds them.
    #[must_use]
    pub fn span_short(&self, rows: u32) -> u32 {
        let rung = self.rung_for(rows);
        self.pool
            .act_chunks(rung)
            .saturating_sub(self.pool.act_reserved())
    }
}

/// A bind of a program at given extents under way: the standing bytes it
/// may add are held beside the booked total until it is through, either
/// way.
pub struct ProgramOpen {
    shared: Shared,
    /// What the bind may add to the engine's standing guest bytes, held
    /// beside the booked total until the bind is through.
    standing: u64,
}

impl ProgramOpen {
    /// The `standing` bytes a bind of `key` may add to the engine's
    /// standing guest bytes ([`SharedPool::standing_added`]) are held
    /// beside the booked total. Refused, by the pair's name and the pool as
    /// it was, where a lane of it reads out more than the engine's readout
    /// store holds ([`SharedPool::readouts_refused`]), or where even the
    /// smallest rung's activation with the standing bytes is more than a
    /// rung may take ([`SharedPool::act_most`]); a narrower fit cuts fires
    /// to the widest rung that does ([`SharedPool::widest_fit`]). The
    /// program's own scratch is state, taken at the bind
    /// ([`super::super::roles::bind_instance`]).
    pub fn new(shared: &Shared, key: FireKey, standing: u64) -> Result<Self, String> {
        if let Some(why) = shared.lock().unwrap().readouts_refused(&key) {
            return Err(why);
        }
        Self::open(shared, key, standing)
    }

    /// [`Self::new`] for a float pass's bind (a kv-less encoder's or
    /// embedding's): its lanes read out every row they embed through the
    /// engine's readout landing, not its fixed store, so the store refuses
    /// nothing of it; the standing bytes are held as for any.
    pub fn float(shared: &Shared, key: FireKey, standing: u64) -> Result<Self, String> {
        Self::open(shared, key, standing)
    }

    fn open(shared: &Shared, key: FireKey, standing: u64) -> Result<Self, String> {
        let mut pool = shared.lock().unwrap();
        pool.programs.standing_pending += standing;
        pool.sync_standing();
        if pool.standing_refused() {
            let why = pool.refusal(key);
            pool.programs.standing_pending -= standing;
            pool.sync_standing();
            return Err(why);
        }
        Ok(Self {
            shared: Shared::clone(shared),
            standing,
        })
    }
}

impl Drop for ProgramOpen {
    fn drop(&mut self) {
        let mut pool = self.shared.lock().unwrap();
        // What the bind added is the engine's total now, booked on the
        // lane; a bind refused added nothing.
        pool.programs.standing_pending -= self.standing;
        pool.sync_standing();
    }
}

/// The engine's side of a raise: the notice of a rung it does not hold
/// (`Engine::set_rung` through the engine's lane), where it backs its carve
/// for the wider rung. Nothing moves in a raise.
pub trait Mover {
    /// `Engine::set_rung(from, to)` through the engine's lane.
    fn set_rung(
        &self,
        from: u32,
        to: u32,
    ) -> impl std::future::Future<Output = Result<(), String>> + Send;

    /// The engine whose rung events ([`super::rung_changed`]) a wait on
    /// this raise waits on.
    fn engine(&self) -> usize;
}

/// Raises `shared`'s activation to `goal`'s (a rung's tokens, given the
/// pool; the rung the engine holds where that is wider,
/// [`SharedPool::raise_at`]) where it falls short and nothing else is
/// raising it: planned and taken in one hold of the pool, so no slot is
/// handed out between, and the engine's notice follows before fires pass
/// the rung held before. The raise taken, if any.
pub async fn take_raise(
    shared: &Shared,
    goal: impl FnOnce(&SharedPool) -> u32,
    mover: &impl Mover,
) -> Result<Option<RungRaise>, String> {
    let taken = {
        let mut pool = shared.lock().unwrap();
        let goal = goal(&pool);
        // At the rung the engine holds where that is wider: a raise tells
        // it no narrower one, and its activation is what falls short.
        match pool.raise_at(goal) {
            None => {
                // The engine holds a rung that no longer fits: the walk
                // brings it down first, behind the retire fence.
                super::ask_walk(mover.engine());
                None
            }
            Some(at)
                if pool.pool().act_chunks(at) <= pool.pool().act_reserved()
                    || pool.awaiting_notice().is_some() =>
            {
                None
            }
            Some(at) => {
                let stood = pool.stands_at(at);
                match pool.plan_raise(at) {
                    Ok(raise) => pool
                        .raise_applied(&raise)
                        .ok()
                        .map(|notice| (raise, notice, stood)),
                    Err(error) => {
                        // Idle cache lies in the span, or the span is
                        // yield's to grow back: the walk has it cleared or
                        // waits on the grow, and raises then.
                        if error.span_cached() || error.span_shed() {
                            super::ask_walk(mover.engine());
                        }
                        None
                    }
                }
            }
        }
    };
    let Some((raise, notice, stood)) = taken else {
        return Ok(None);
    };
    if let Err(refused) = carry_raise(shared, notice, mover).await {
        // Refused and taken back: the rung's waiters look again, and the
        // caller is told; nothing tries it again on its own wake.
        super::rung_changed(mover.engine());
        return Err(refused);
    }
    super::rung_changed(mover.engine());
    super::raise_said(mover.engine(), raise.rung, stood);
    Ok(Some(raise))
}

/// Carries a raise `shared` applied to the engine, with the notice the
/// apply said it owes ([`SharedPool::raise_applied`]): `set_rung` where the
/// raise leaves the engine another rung than it holds (it backs its carve
/// for the wider rung). A raise at the rung the engine holds sends none:
/// the engine covers standing bytes that grew under its rung inside the
/// bind that grew them. Refused, the raise is taken back
/// ([`SharedPool::raise_refused`]) and fires stay at the rung in hand; the
/// refusal answered. Nothing faults.
pub async fn carry_raise(
    shared: &Shared,
    notice: Option<(u32, u32)>,
    mover: &impl Mover,
) -> Result<(), String> {
    let Some((from, to)) = notice else {
        return Ok(());
    };
    if let Err(error) = mover.set_rung(from, to).await {
        tracing::warn!(%error, from, to, "a raise's notice was refused; fires stay at the rung in hand");
        shared.lock().unwrap().raise_refused();
        return Err(format!("the engine refused rung {to}'s notice: {error}"));
    }
    shared.lock().unwrap().rung_told();
    Ok(())
}

/// Waits until `shared`'s activation holds what fires may take now, the
/// engine's standing guest bytes included: fires cut to `cap()` rows, and
/// no wider than the widest rung that fits ([`SharedPool::widest_fit`]). Where the span falls short the raise is
/// taken: nothing moves and no room is asked of anyone (idle cache in the
/// widened span is cleared first, live state never lies there, and the
/// span is reserved at load: given back only by yield, which the
/// governor's grow backs again), and the engine's notice is taken before
/// this returns. Where fires are cut wider than the widest
/// rung that fits, or the engine holds a wider rung than that
/// ([`SharedPool::told_unfit`]), it waits on the walk to cut them and
/// bring the engine down ([`super::SharedGrow`], asked to look at once:
/// [`super::ask_walk`]). The engine covers the bind's standing bytes at
/// the rung it holds, so the room is set aside at that rung where it is
/// wider than the fires' ([`SharedPool::raise_at`]), and the bind is sent
/// only once that rung fits with them. What a bind whose standing bytes
/// raised the span waits on before it is sent; an `Err` refuses the bind.
///
/// At a rung the engine holds, the raise sends the engine no notice
/// ([`carry_raise`]).
pub async fn fund_span(
    shared: &Shared,
    cap: impl Fn() -> u32,
    mover: &impl Mover,
) -> Result<(), String> {
    let engine = mover.engine();
    loop {
        // Armed before the look: a change between the two is not lost.
        let watch = super::rung_watch(engine);
        tokio::pin!(watch);
        watch.as_mut().enable();
        let (short, awaiting, goal, cut) = {
            let pool = shared.lock().unwrap();
            let rows = cap();
            let fires = pool.rung_for(rows);
            // Set aside at the rung the engine holds where that is wider:
            // it covers the bind's standing bytes there.
            let goal = pool.raise_at(fires);
            let short = goal.map_or(0, |goal| {
                pool.pool()
                    .act_chunks(goal)
                    .saturating_sub(pool.pool().act_reserved())
            });
            (
                short,
                pool.awaiting_notice().is_some(),
                goal,
                rows <= pool.lifted(fires),
            )
        };
        if goal.is_none() {
            // The engine holds a rung that no longer fits: the walk
            // brings it down, and tells the rung.
            super::ask_walk(engine);
            watch.await;
            continue;
        }
        if short == 0 && !awaiting && cut {
            return Ok(());
        }
        if awaiting || short == 0 {
            // A raise's notice is the engine's to take, or the walk is to cut
            // fires to the widest rung that fits: either tells the rung.
            if !cut {
                super::ask_walk(engine);
            }
            watch.await;
            continue;
        }
        if take_raise(shared, |pool| pool.rung_for(cap()), mover)
            .await?
            .is_none()
        {
            // The walk is raising it (idle cache in the span cleared first,
            // or the span yield gave back backed again): it tells the rung.
            watch.await;
        }
    }
}

/// The runtime's [`Mover`]: engine `engine`'s lane.
pub struct LaneMover {
    pub engine: usize,
}

impl Mover for LaneMover {
    fn engine(&self) -> usize {
        self.engine
    }

    async fn set_rung(&self, from: u32, to: u32) -> Result<(), String> {
        crate::scheduler::pool(
            self.engine,
            crate::scheduler::worker::PoolOp::Rung { from, to },
        )
        .await
        .map_err(|error| format!("{error:#}"))
    }
}
