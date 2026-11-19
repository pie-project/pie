//! The bound programs' guest term (REDESIGN §6): what a fire holding guest
//! programs' lanes needs beside the model's activation, each program's
//! per-batch tables and value scratch, sized by the bind's extents
//! ([`::engine::Engine::fire_facts`]). A program registers and binds after
//! load, so the engine cannot state it in [`::engine::chunk::ChunkFacts`];
//! the pool folds it into every rung's activation instead
//! ([`super::super::slots::SlotPool::act_chunks`]).
//!
//! The facts are a pair's, a live (program, extents) ([`FireKey`]); the
//! term is a fire's ([`::engine::GuestTerm`]): the worst a fire of a rung's
//! lanes takes whatever pairs its members are of, each pair paying only
//! for the members a fire gives it. A pair counts from the first bind of
//! those extents (under way, [`ProgramOpen`], or bound) to the last one's
//! close. A pair that raises the activation the fires may take now is
//! taken before its first bind is sent, through the raise a pass's wider
//! rung takes, its room funded as a planner waiter ([`fund_span`]). Once
//! the pair's last bind closes its part of the term drops, and the walk's
//! next lower gives the room back, once every fire cut to the old rung has
//! retired.
//!
//! Beside the term, the engine keeps standing guest bytes in the span for
//! the live instances of registered programs, whatever the fire (Metal:
//! the batch tables it lays once a set, every set together;
//! [`::engine::Engine::standing_after`]). They change only at a bind and a
//! close, a set shared by instances counted once. A bind asks the engine,
//! on the lane, what the total would be once it is bound, and holds what
//! that adds to the total booked ([`ProgramOpen`]) while it is under way:
//! a total that does not fit the span with the term at the smallest rung
//! refuses the bind by name before the engine is asked to bind, and one
//! that does is backed below the carve, at every rung, before the bind is
//! sent ([`fund_span`]), so the engine lays it in room set aside. The
//! engine's total after each bind and each close is booked on the lane
//! ([`SharedPool::standing_stated`]). Programs are never unregistered, but
//! a set that no live instance shares any more is given back at the close
//! that leaves it so.
//!
//! The term is a table, built when the pairs change and never in a fire.
//! No build runs under the pool's lock on a task that waits on nothing
//! else there, and none on the engine's lane: a bind builds the table it
//! would leave on its own task, beside the span it funds ([`ProgramOpen`]);
//! the lane's bind takes over the count that bind held and changes
//! nothing; a close drops the count on the lane and the scheduler builds
//! the table at the close's commit ([`fit_fire`]). Until then the pool's
//! term is the one of before the close, which is never under what a fire
//! takes.

use std::collections::HashMap;

use ::engine::{BindExtents, FireFacts, FirePair, InstanceId, ProgramId};

use super::{RungEvacuation, Shared, SharedPool};

/// A fire term's key: a program bound at given extents.
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

/// Each pair's fire facts, and who keeps it counted.
#[derive(Debug, Default)]
pub struct ProgramFires {
    facts: HashMap<FireKey, FireFacts>,
    /// The instances of each pair that can be in a fire: bound, and
    /// binding (a guard out whose bind the lane has not counted).
    open: HashMap<FireKey, u32>,
    /// The guards out for each pair ([`ProgramOpen`]) and, of them, those
    /// whose count an instance took over ([`SharedPool::instance_bound`]).
    guards: HashMap<FireKey, (u32, u32)>,
    instances: HashMap<InstanceId, FireKey>,
    /// Moves with every change to what the term is built from (a count,
    /// a pair's facts, the fire's lanes); the pool's term is the one of
    /// `fitted`.
    generation: u64,
    fitted: u64,
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

impl ProgramFires {
    /// The pairs that can be in a fire now, each with its live instances
    /// (bound or binding): what the term is built from.
    fn pairs(&self) -> Vec<FirePair> {
        self.open
            .iter()
            .filter_map(|(key, &live)| {
                let facts = *self.facts.get(key)?;
                Some(FirePair { facts, live })
            })
            .collect()
    }

    /// [`Self::pairs`] with one more of `key` counted: what a bind of it
    /// would leave.
    fn pairs_with(&self, key: &FireKey) -> Vec<FirePair> {
        let mut pairs: Vec<FirePair> = self
            .open
            .iter()
            .filter(|(open, _)| *open != key)
            .filter_map(|(open, &live)| {
                let facts = *self.facts.get(open)?;
                Some(FirePair { facts, live })
            })
            .collect();
        if let Some(&facts) = self.facts.get(key) {
            let live = self.open.get(key).copied().unwrap_or(0) + 1;
            pairs.push(FirePair { facts, live });
        }
        pairs
    }

    /// One more of `key` can be in a fire.
    fn count(&mut self, key: FireKey) {
        *self.open.entry(key).or_default() += 1;
        self.generation += 1;
    }

    /// One of `key` less can be in a fire; the pair drops with its last.
    fn uncount(&mut self, key: FireKey) {
        let Some(open) = self.open.get_mut(&key) else {
            return;
        };
        *open -= 1;
        if *open == 0 {
            self.open.remove(&key);
        }
        self.generation += 1;
    }
}

/// A pair's own cap on the lanes a fire gives it, as a line shows it:
/// `FireFacts::max_lanes` of 0 states none, not a cap of no lanes.
fn own_cap(max_lanes: u32) -> String {
    if max_lanes == 0 {
        "no cap of its own".to_owned()
    } else {
        max_lanes.to_string()
    }
}

impl std::fmt::Display for ProgramFires {
    /// Every open pair with the facts it counts, pair after pair in the
    /// keys' order: what a refused submission is read against
    /// ([`impossible_submission`]).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut open: Vec<_> = self.open.iter().collect();
        open.sort_unstable_by_key(|(key, _)| (key.program.0, format!("{:?}", key.extents)));
        for (at, (key, live)) in open.into_iter().enumerate() {
            if at > 0 {
                f.write_str("; ")?;
            }
            match self.facts.get(key) {
                Some(facts) => write!(
                    f,
                    "{key}: live {live}, a_lane {}, fixed {}, max_lanes {}, laid {:?}",
                    facts.a_lane,
                    facts.fixed,
                    own_cap(facts.max_lanes),
                    facts.laid
                )?,
                None => write!(f, "{key}: live {live}, facts not learned")?,
            }
        }
        Ok(())
    }
}

/// The engine called a submission impossible
/// ([`::engine::Error::Impossible`]): a fire asked more of the span than
/// the engine backs, which the rung arithmetic is there to rule out. One
/// line with what that arithmetic held when it came back: the pool's rung
/// and the one the engine holds, the cap fires are cut to, the activation
/// reserved, the rung's own with the term, the term's bytes at that rung,
/// the span the engine states and the most a rung may take under it, and
/// every open pair's facts.
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
        term_bytes = pool.pool().fire_bytes(rung),
        readout_bytes = pool.pool().readout_bytes(pool.pool().rung_of(rung)),
        standing_bytes = pool.pool().standing(),
        act_span_chunks = %span,
        act_most = pool.act_most(),
        max_lanes = pool.pool().max_lanes(),
        terms = %pool.programs,
        "the engine called a submission impossible"
    );
}

impl SharedPool {
    /// The most lanes a fire on the engine carries (`FireLimits::max_lanes`):
    /// a rung of `r` tokens holds `min(r, lanes)`, and the term is built
    /// for fires of up to that many.
    pub fn set_max_lanes(&mut self, lanes: u32) {
        self.pool.set_max_lanes(lanes);
        self.programs.generation += 1;
        self.refit();
    }

    /// The engine's readout facts ([`::engine::Engine::readout_facts`]),
    /// with its own ceiling on the rows a fire reads out
    /// (`FireLimits::max_readouts`, 0: none): every rung counts the rows a
    /// fire of it reads out beside the guest term where they land beside
    /// the activation, so fires are cut to the widest rung that holds
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

    /// The engine said fires of `key` need `facts`
    /// ([`::engine::Engine::fire_facts`]), asked once a pair, before its
    /// first bind counts: nothing is built here unless the pair is open
    /// already and its facts differ.
    pub fn fire_facts_learned(&mut self, key: FireKey, facts: &FireFacts) {
        let known = self.programs.facts.insert(key, *facts);
        if known.as_ref() != Some(facts) && self.programs.open.contains_key(&key) {
            self.programs.generation += 1;
            self.refit();
        }
    }

    /// Whether `key`'s fire facts were asked already.
    #[must_use]
    pub fn knows_fire(&self, key: &FireKey) -> bool {
        self.programs.facts.contains_key(key)
    }

    /// The chunks an activation may take at most: every chunk but the
    /// reserved ones and the weights' pinned slots, which no move frees.
    fn act_room(&self) -> u32 {
        let room = self.room();
        room.total_chunks
            .saturating_sub(room.reserved_chunks)
            .saturating_sub(room.weight_chunks)
    }

    /// The chunks a rung's activation may take at most, its guest term
    /// included: what the device has for it ([`Self::act_room`]), and no
    /// more than the engine states its span covers as it stands
    /// (`ChunkFacts::act_span_chunks`). Both read here, at each fit:
    /// nothing is kept of either, so pinned weights that come or go and a
    /// span the engine states anew ([`SlotPool::set_act_span`]) count from
    /// the next.
    #[must_use]
    pub fn act_most(&self) -> u32 {
        let room = self.act_room();
        self.pool.act_span().map_or(room, |span| span.min(room))
    }

    /// Whether every rung fits unasked: no pair takes anything and the
    /// engine states no span of its own, so a rung's activation is the
    /// model's, which the pool was built to hold.
    fn fits_unasked(&self) -> bool {
        self.pool.guests_zero() && self.pool.act_span().is_none()
    }

    /// The widest rung whose activation, the guest term of the programs
    /// that can be in a fire included, is no more than a rung may take
    /// ([`Self::act_most`]); `None` where even the smallest is. Without a
    /// term or a stated span, the largest. No rung past it is set aside:
    /// every goal a raise is planned for is cut to it
    /// ([`Self::rung_for`], [`RungWalk::step`]), so no engine is asked for
    /// a rung whose term was not held against its span here.
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

    /// Whether the guest term leaves no rung room on the device: never
    /// where no pair takes anything (a zero term is no term), and never on
    /// a pool that states no rungs, whose activation is no part of it, so
    /// the term is ignored there.
    fn term_refused(&self) -> bool {
        !self.pool.guests_zero() && !self.pool.rungs().is_empty() && self.widest_fit().is_none()
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

    /// The widest cap fires may take with the programs' guest term: the
    /// widest rung that fits, lifted; `u32::MAX` without a term.
    #[must_use]
    pub fn fit_cap(&self) -> u32 {
        self.lifted(self.fit_rung())
    }

    /// The engine bound `instance` as `key`: it can be in a fire until it
    /// closes. The bind that sent it counted the pair already and holds
    /// that count ([`ProgramOpen`]); the instance takes it over here, so
    /// the term stands as it is and nothing is built on the lane. Where no
    /// such bind is left (its task was dropped before the lane reached the
    /// bind), the instance is counted here and the term built under the
    /// lock.
    pub fn instance_bound(&mut self, instance: InstanceId, key: FireKey) {
        if self.programs.instances.insert(instance, key).is_some() {
            return;
        }
        if let Some((out, taken)) = self.programs.guards.get_mut(&key)
            && *taken < *out
        {
            *taken += 1;
            return;
        }
        self.programs.count(key);
        self.refit();
    }

    /// The engine closed `instance`: one of its pair less can be in a
    /// fire. The count drops here, on the lane; the term is built where
    /// the close commits ([`fit_fire`]), and until then counts the
    /// instance still.
    pub fn instance_closed(&mut self, instance: InstanceId) {
        if let Some(key) = self.programs.instances.remove(&instance) {
            self.programs.uncount(key);
        }
    }

    /// A bind of `key` is through, either way: the count it held is its
    /// instance's where the lane bound one, and drops otherwise. Whether
    /// it dropped (the term is then to be built, [`fit_fire`]).
    fn guard_back(&mut self, key: FireKey) -> bool {
        let Some((out, taken)) = self.programs.guards.get_mut(&key) else {
            return false;
        };
        *out -= 1;
        let kept = *taken > 0;
        if kept {
            *taken -= 1;
        }
        if *out == 0 {
            self.programs.guards.remove(&key);
        }
        if !kept {
            self.programs.uncount(key);
        }
        !kept
    }

    /// The term of the pairs open now, built under the lock: for the
    /// changes no task builds it for ([`Self::set_max_lanes`], facts that
    /// change under an open pair, an instance no bind holds a count for).
    pub fn refit(&mut self) {
        let pairs = self.programs.pairs();
        self.pool.fit_fire(&pairs);
        self.programs.fitted = self.programs.generation;
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
    /// the term the bind would leave and the standing bytes it would hold,
    /// read while the pool holds both.
    fn refusal(&self, key: FireKey) -> String {
        let most = self.act_most();
        let smallest = self.pool.act_chunks(0);
        let facts = self.programs.facts.get(&key).copied().unwrap_or_default();
        let of = match self.pool.act_span() {
            Some(span) if span <= most => {
                format!("{span} chunks the engine's activation span covers")
            }
            _ => format!("{most} the device has for it"),
        };
        format!(
            "{key}'s fires need {} bytes and {} bytes a lane beside the model's \
             activation, the engine's standing guest bytes would be {} beside every \
             fire, and the smallest rung would then take {smallest} chunks of the {of}",
            facts.fixed,
            facts.a_lane,
            self.pool.standing()
        )
    }

    /// The chunks the activation lacks for fires of `rows` (the rung that
    /// holds them, the guest term's included); 0 where it holds them.
    #[must_use]
    pub fn span_short(&self, rows: u32) -> u32 {
        let rung = self.rung_for(rows);
        self.pool
            .act_chunks(rung)
            .saturating_sub(self.pool.act_reserved())
    }
}

/// Makes `shared`'s term the one of the pairs open now: the table is built
/// outside the pool's lock from the pairs read under it, and installed
/// only where nothing it was built from changed since (else it is built
/// again from what stands). What a close's commit and a dropped bind call,
/// on their own threads; nothing where the term is current.
pub fn fit_fire(shared: &Shared) {
    loop {
        let (pairs, most, generation, mut term) = {
            let pool = shared.lock().unwrap();
            if pool.programs.fitted == pool.programs.generation {
                return;
            }
            (
                pool.programs.pairs(),
                pool.pool.max_lanes(),
                pool.programs.generation,
                pool.pool.fire().clone(),
            )
        };
        term.fit(&pairs, most);
        let mut pool = shared.lock().unwrap();
        if pool.programs.generation == generation {
            pool.pool.set_fire(term);
            pool.programs.fitted = generation;
            return;
        }
    }
}

/// A bind of a program at given extents under way: the pair counts one
/// more while it is held. The lane's bind hands that count to the instance
/// ([`SharedPool::instance_bound`]), so the guard is dropped once the bind
/// is through, either way, and gives back only a count no instance took.
pub struct ProgramOpen {
    shared: Shared,
    key: FireKey,
    /// What the bind may add to the engine's standing guest bytes, held
    /// beside the booked total until the bind is through.
    standing: u64,
}

impl ProgramOpen {
    /// One more bind of `key` can be in a fire: the pair counts from now,
    /// and the `standing` bytes the bind may add to the engine's standing
    /// guest bytes ([`SharedPool::standing_added`]) are held beside the
    /// booked total. Refused where even the smallest rung's activation, the
    /// term and the standing bytes included, is more than a rung may take ([`SharedPool::act_most`]:
    /// every chunk but the reserved ones and the weights' pinned slots,
    /// and no more than the engine states its span covers); a narrower fit
    /// cuts fires to the widest rung that does
    /// ([`SharedPool::widest_fit`]).
    ///
    /// The term grows with each bind, so each builds the table it would
    /// leave: here, on the binding task, outside the pool's lock, from the
    /// pairs read under it with this one counted. It is installed, with
    /// the count, in one hold of the pool and only where nothing it was
    /// built from changed since; a bind refused leaves the pool as it
    /// found it, its term never seen by a fire.
    pub fn new(shared: &Shared, key: FireKey, standing: u64) -> Result<Self, String> {
        // A lane that alone reads out more than the readout store holds:
        // no fire seats it, whatever the rung.
        if let Some(why) = shared.lock().unwrap().readouts_refused(&key) {
            return Err(why);
        }
        Self::open(shared, key, standing)
    }

    /// [`Self::new`] for a float pass's bind (a kv-less encoder's or
    /// embedding's): its lanes read out every row they embed through the
    /// engine's readout landing, not its fixed store, so the store refuses
    /// nothing of it; the term and the standing bytes are held as for any.
    pub fn float(shared: &Shared, key: FireKey, standing: u64) -> Result<Self, String> {
        Self::open(shared, key, standing)
    }

    fn open(shared: &Shared, key: FireKey, standing: u64) -> Result<Self, String> {
        loop {
            let (pairs, most, generation, mut term) = {
                let pool = shared.lock().unwrap();
                (
                    pool.programs.pairs_with(&key),
                    pool.pool.max_lanes(),
                    pool.programs.generation,
                    pool.pool.fire().clone(),
                )
            };
            term.fit(&pairs, most);
            let mut pool = shared.lock().unwrap();
            if pool.programs.generation != generation {
                continue;
            }
            let before = pool.pool.set_fire(term);
            pool.programs.standing_pending += standing;
            pool.sync_standing();
            if pool.term_refused() {
                let why = pool.refusal(key);
                pool.pool.set_fire(before);
                pool.programs.standing_pending -= standing;
                pool.sync_standing();
                return Err(why);
            }
            let first = !pool.programs.open.contains_key(&key);
            pool.programs.count(key);
            pool.programs.fitted = pool.programs.generation;
            let (out, _) = pool.programs.guards.entry(key).or_default();
            *out += 1;
            if first && let Some(facts) = pool.programs.facts.get(&key) {
                tracing::debug!(
                    %key,
                    a_lane = facts.a_lane,
                    fixed = facts.fixed,
                    max_lanes = %own_cap(facts.max_lanes),
                    laid = ?facts.laid,
                    "a pair's first bind counts its fire facts"
                );
            }
            return Ok(Self {
                shared: Shared::clone(shared),
                key,
                standing,
            });
        }
    }
}

impl Drop for ProgramOpen {
    fn drop(&mut self) {
        let dropped = {
            let mut pool = self.shared.lock().unwrap();
            // What the bind added is the engine's total now, booked on the
            // lane; a bind refused added nothing.
            pool.programs.standing_pending -= self.standing;
            pool.sync_standing();
            pool.guard_back(self.key)
        };
        if dropped {
            fit_fire(&self.shared);
        }
    }
}

/// How a span short of room has it freed: a planner waiter's ask of `pages`
/// kv pages, granted (`Ok(true)`), to be asked again (`Ok(false)`), or
/// refused.
pub trait Funder {
    fn fund(&self, pages: u32) -> impl std::future::Future<Output = Result<bool, String>> + Send;
}

/// The engine's side of a raise: the copy of what it moves out of the span
/// (`Engine::evacuate` through the engine's lane), and the fire epoch handed
/// out last.
pub trait Mover {
    fn evacuate(
        &self,
        evacuation: ::engine::chunk::Evacuation,
    ) -> impl std::future::Future<Output = Result<(), String>> + Send;

    /// `Engine::set_rung(from, to)` through the engine's lane.
    fn set_rung(
        &self,
        from: u32,
        to: u32,
    ) -> impl std::future::Future<Output = Result<(), String>> + Send;

    fn epoch_now(&self) -> u64;

    /// The engine whose rung events ([`super::rung_changed`]) a wait on
    /// this raise waits on.
    fn engine(&self) -> usize;
}

/// Raises `shared`'s activation to `goal`'s (a rung's tokens, given the
/// pool; the rung the engine holds where that is wider,
/// [`SharedPool::raise_at`]) where it falls short and nothing else is
/// raising it: planned and
/// taken in one hold of the pool, so no slot is handed out between, and the
/// engine's copy of what it moves follows before fires pass the rung held
/// before. The raise taken, if any.
pub async fn take_raise(
    shared: &Shared,
    goal: impl FnOnce(&SharedPool) -> u32,
    mover: &impl Mover,
) -> Result<Option<RungEvacuation>, String> {
    let taken = {
        let mut pool = shared.lock().unwrap();
        let goal = goal(&pool);
        // At the rung the engine holds where that is wider: a raise tells
        // it no narrower one, and its activation is what falls short.
        match pool.raise_at(goal) {
            None => {
                // The engine holds a rung the terms no longer fit: the
                // walk brings it down first, behind the retire fence.
                super::ask_walk(mover.engine());
                None
            }
            Some(at) if !pool.raise_owed(at) || pool.copying().is_some() => None,
            Some(at) => {
                let stood = pool.stands_at(at);
                pool.plan_raise(at).ok().and_then(|evacuation| {
                    let notice = pool.raise_applied(&evacuation).ok()?;
                    Some((evacuation, notice, stood))
                })
            }
        }
    };
    let Some((evacuation, notice, stood)) = taken else {
        return Ok(None);
    };
    if let Err(refused) = carry_raise(shared, &evacuation, notice, mover).await {
        // Refused and taken back: the rung's waiters look again, and the
        // caller is told; nothing tries it again on its own wake.
        super::rung_changed(mover.engine());
        return Err(refused);
    }
    super::rung_changed(mover.engine());
    super::raise_said(
        mover.engine(),
        evacuation.rung,
        evacuation.moves.len(),
        stood,
    );
    Ok(Some(evacuation))
}

/// Carries a raise `shared` applied to the engine, with the notice the
/// apply said it owes ([`SharedPool::raise_applied`]): `set_rung` first
/// where the raise leaves the engine another rung than it holds (it backs
/// its carve for the wider rung), then the copy of what the raise moves, a
/// message of its own. A raise at the rung the engine holds sends no
/// notice, and none at all where it moves nothing: the engine covers a
/// guest term that grew under its rung inside the bind that grew it.
/// Either refused, the raise is taken back ([`SharedPool::raise_refused`]),
/// fires stay at the rung in hand, and an engine that backed the carve is
/// told the rung went back; the refusal answered. Nothing faults.
pub async fn carry_raise(
    shared: &Shared,
    evacuation: &RungEvacuation,
    notice: Option<(u32, u32)>,
    mover: &impl Mover,
) -> Result<(), String> {
    if notice.is_none() && evacuation.moves.is_empty() {
        return Ok(());
    }
    if let Some((from, to)) = notice
        && let Err(error) = mover.set_rung(from, to).await
    {
        tracing::warn!(%error, from, to, "a raise's carve was refused; fires stay at the rung in hand");
        shared.lock().unwrap().raise_refused(evacuation);
        return Err(format!(
            "the engine refused the carve for rung {to}: {error}"
        ));
    }
    if !evacuation.moves.is_empty()
        && let Err(error) = mover.evacuate(SharedPool::rung_moves(evacuation)).await
    {
        tracing::warn!(%error, "a raise's copy was refused; fires stay at the rung in hand");
        shared.lock().unwrap().raise_refused(evacuation);
        if let Some((from, to)) = notice
            && let Err(error) = mover.set_rung(to, from).await
        {
            tracing::warn!(%error, from = to, to = from, "a rung notice was refused");
        }
        return Err(format!("the engine refused a raise's copy: {error}"));
    }
    let next = mover.epoch_now().saturating_add(1);
    shared.lock().unwrap().raise_copied(next);
    Ok(())
}

/// Waits until `shared`'s activation holds what fires may take now, the
/// fire terms of the programs that can be in a fire included: fires cut to
/// `cap()` rows, and no wider than the widest rung that fits with the terms
/// ([`SharedPool::widest_fit`]). Where the span falls short, the room is
/// asked of `funder` as a rung raise's is, then the raise taken: every
/// state chunk in the widened span moved below it and the engine's copy
/// done before this returns. Where fires are cut wider than the widest
/// rung that fits, or the engine holds a wider rung than that
/// ([`SharedPool::told_unfit`]), it waits on the walk to cut them and
/// bring the engine down ([`super::SharedGrow`], asked to look at once:
/// [`super::ask_walk`]). The engine covers the bind's term at the rung it
/// holds, so the room is set aside at that rung where it is wider than
/// the fires' ([`SharedPool::raise_at`]), and the bind is sent only once
/// that rung fits with the term. What a bind of a program whose term
/// raised the span waits on before it is sent; an `Err` refuses the bind.
///
/// At a rung the engine holds, the raise the term asks sends the engine
/// no notice ([`carry_raise`]); state it moves out of the span goes as its
/// own message.
pub async fn fund_span(
    shared: &Shared,
    cap: impl Fn() -> u32,
    funder: &impl Funder,
    mover: &impl Mover,
) -> Result<(), String> {
    let engine = mover.engine();
    loop {
        // Armed before the look: a change between the two is not lost.
        let watch = super::rung_watch(engine);
        tokio::pin!(watch);
        watch.as_mut().enable();
        let (short, copying, goal, cut) = {
            let pool = shared.lock().unwrap();
            let rows = cap();
            let fires = pool.rung_for(rows);
            // Set aside at the rung the engine holds where that is wider:
            // it covers the bind's term there.
            let goal = pool.raise_at(fires);
            let short = goal.map_or(0, |goal| {
                pool.pool()
                    .act_chunks(goal)
                    .saturating_sub(pool.pool().act_reserved())
            });
            (
                short,
                pool.copying().is_some(),
                goal,
                rows <= pool.lifted(fires),
            )
        };
        let Some(goal) = goal else {
            // The engine holds a rung the terms no longer fit: the walk
            // brings it down, and tells the rung.
            super::ask_walk(engine);
            watch.await;
            continue;
        };
        if short == 0 && !copying && cut {
            return Ok(());
        }
        if copying || short == 0 {
            // A raise's copy is the engine's to take, or the walk is to cut
            // fires to the widest rung that fits: either tells the rung.
            if !cut {
                super::ask_walk(engine);
            }
            watch.await;
            continue;
        }
        let pages = shared.lock().unwrap().raise_short(goal).1;
        if pages > 0 {
            tracing::debug!(
                pages,
                goal,
                "a program's fire term asks the planner for room"
            );
            if !funder.fund(pages).await? {
                // Quoted again: the planner's answer is the event.
                continue;
            }
        }
        if take_raise(shared, |pool| pool.rung_for(cap()), mover)
            .await?
            .is_none()
        {
            // The walk is raising it, or the room went to another since:
            // either moves the rung or frees room.
            watch.await;
        }
    }
}

/// The runtime's [`Funder`]: process `pid`'s planner, as a waiter of
/// `quorum`.
pub struct Planner {
    pub pid: uuid::Uuid,
    pub quorum: uuid::Uuid,
}

impl Funder for Planner {
    async fn fund(&self, pages: u32) -> Result<bool, String> {
        let Some(planner) = crate::planner::planner_of(self.pid) else {
            return Err(format!("process {} has no planner to fund it", self.pid));
        };
        fund_with(&planner, self.pid, self.quorum, pages).await
    }
}

/// `pid`'s ask of `pages` kv pages of `planner` for a span short of room,
/// as a waiter of `quorum`: a rise that cannot fire narrower (a program's
/// fire term, a piece no cut brings under the rung held), keyed by its
/// process ([`crate::planner::ResidencyPlanner::acquire_rise`]).
///
/// Suspended meanwhile, the process has nothing to ask until it is
/// resident again: the planner answers its ask at once (`Yield`), and asked
/// again at once, the span walk spun on its worker with no await between
/// asks, holding the planner's own task (woken there, in the worker's
/// next-task slot, which no other worker takes) off every worker: no plan
/// ran again, and every waiter stood with pages free. It asks again once
/// its residency comes back, the planner's event.
pub async fn fund_with(
    planner: &std::sync::Arc<crate::planner::ResidencyPlanner>,
    pid: uuid::Uuid,
    quorum: uuid::Uuid,
    pages: u32,
) -> Result<bool, String> {
    match planner.acquire_rise(pid, quorum, pages).await {
        Ok(crate::planner::Acquired::Granted(grant)) => {
            // Given back at once: the raise takes the room it freed.
            drop(grant);
            Ok(true)
        }
        Ok(crate::planner::Acquired::Yield) => {
            planner
                .wait_resident(pid)
                .await
                .map_err(|error| error.to_string())?;
            Ok(false)
        }
        Ok(crate::planner::Acquired::Requote) => Ok(false),
        Err(error) => Err(error.to_string()),
    }
}

/// The runtime's [`Mover`]: engine `engine`'s lane, and its stores' fire
/// epoch.
pub struct LaneMover<F> {
    pub engine: usize,
    pub epoch_now: F,
}

impl<F: Fn() -> u64 + Sync> Mover for LaneMover<F> {
    fn engine(&self) -> usize {
        self.engine
    }

    async fn evacuate(&self, evacuation: ::engine::chunk::Evacuation) -> Result<(), String> {
        crate::scheduler::pool(
            self.engine,
            crate::scheduler::worker::PoolOp::Evacuate(evacuation),
        )
        .await
        .map_err(|error| format!("{error:#}"))
    }

    async fn set_rung(&self, from: u32, to: u32) -> Result<(), String> {
        crate::scheduler::pool(
            self.engine,
            crate::scheduler::worker::PoolOp::Rung { from, to },
        )
        .await
        .map_err(|error| format!("{error:#}"))
    }

    fn epoch_now(&self) -> u64 {
        (self.epoch_now)()
    }
}

/// Engine `engine`'s stores' fire epoch handed out last (model 0's).
#[must_use]
pub fn epoch_of(engine: usize) -> u64 {
    crate::store::registry::try_get(0, engine).map_or(0, |stores| {
        crate::store::registry::with_kv_lock(&stores.kv, "program-fire", |kv| kv.current_epoch())
    })
}
