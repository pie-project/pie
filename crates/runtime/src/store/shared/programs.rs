//! The registered programs' fire term (REDESIGN §6): what a fire holding a
//! guest program's lanes needs beside the model's activation, the
//! program's per-batch tables and value scratch
//! ([`::engine::ProgramFacts::fire_bytes`]). A program registers after
//! load, so the engine cannot state it in [`::engine::chunk::ChunkFacts`];
//! the pool folds it into every rung's activation instead
//! ([`super::super::slots::SlotPool::act_chunks`]).
//!
//! A program can be in a fire while an instance of it is bound, or about to
//! be ([`ProgramOpen`]): its term counts from the first of those to the
//! last. A term that raises the activation the fires may take now is taken
//! before the bind proceeds, through the raise a pass's wider rung takes,
//! its room funded as a planner waiter ([`fund_span`]). Once the last
//! instance closes the term drops, and the walk's next lower gives the room
//! back.

use std::collections::HashMap;

use ::engine::{InstanceId, ProgramFacts, ProgramId};

use super::{RungEvacuation, Shared, SharedPool};
use crate::store::slots::FireTerm;

/// Each program's fire term, and who keeps it counted.
#[derive(Debug, Default)]
pub struct ProgramFires {
    terms: HashMap<ProgramId, FireTerm>,
    /// Bound instances and binds under way, by program.
    open: HashMap<ProgramId, u32>,
    instances: HashMap<InstanceId, ProgramId>,
}

impl ProgramFires {
    /// The terms of the programs that can be in a fire now.
    fn open_terms(&self) -> Vec<FireTerm> {
        self.open
            .keys()
            .filter_map(|program| self.terms.get(program).copied())
            .collect()
    }
}

/// A program's fire term from its facts.
#[must_use]
pub fn term_of(facts: &ProgramFacts) -> FireTerm {
    FireTerm {
        fixed: facts.fire_bytes_fixed,
        a_lane: facts.fire_bytes_a_lane,
        max_lanes: facts.max_lanes,
    }
}

impl SharedPool {
    /// The most lanes a fire on the engine carries (`FireLimits::max_lanes`):
    /// a rung of `r` tokens holds `min(r, lanes)`, each needing the fire
    /// term's bytes a lane.
    pub fn set_max_lanes(&mut self, lanes: u32) {
        self.pool.set_max_lanes(lanes);
    }

    /// The engine registered `program`, whose fires need `facts`' term.
    pub fn program_registered(&mut self, program: ProgramId, facts: &ProgramFacts) {
        self.programs.terms.insert(program, term_of(facts));
        self.refold();
    }

    /// The chunks an activation may take at most: every chunk but the
    /// reserved ones and the weights' pinned slots, which no move frees.
    fn act_room(&self) -> u32 {
        let room = self.room();
        room.total_chunks
            .saturating_sub(room.reserved_chunks)
            .saturating_sub(room.weight_chunks)
    }

    /// The widest rung whose activation, the fire terms of the programs
    /// that can be in a fire included, fits on the device; `None` where
    /// even the smallest does not. Without a term, the largest.
    #[must_use]
    pub fn widest_fit(&self) -> Option<u32> {
        let rungs = self.pool.rungs();
        if self.pool.fire().is_empty() {
            return rungs.last().copied();
        }
        let most = self.act_room();
        rungs
            .into_iter()
            .filter(|&rung| self.pool.act_chunks(rung) <= most)
            .max()
    }

    /// The widest rung fires may take: [`Self::widest_fit`], the smallest
    /// where none fits (a bind is refused then), `u32::MAX` without rungs.
    pub(super) fn fit_rung(&self) -> u32 {
        if self.pool.fire().is_empty() {
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

    /// The widest cap fires may take with the programs' fire terms: the
    /// widest rung that fits, lifted; `u32::MAX` without a term.
    #[must_use]
    pub fn fit_cap(&self) -> u32 {
        self.lifted(self.fit_rung())
    }

    /// One more instance of `program` can be in a fire: its term counts
    /// from now. Refused where even the smallest rung's activation, the
    /// term included, cannot fit on the device (past every chunk but the
    /// reserved ones and the weights' pinned slots); a narrower fit cuts
    /// fires to the widest rung that does ([`Self::widest_fit`]).
    pub fn open_program(&mut self, program: ProgramId) -> Result<(), String> {
        let first = !self.programs.open.contains_key(&program);
        *self.programs.open.entry(program).or_default() += 1;
        if first {
            self.refold();
            if self.widest_fit().is_none() {
                let most = self.act_room();
                let smallest = self.pool.act_chunks(0);
                let term = self
                    .programs
                    .terms
                    .get(&program)
                    .copied()
                    .unwrap_or_default();
                self.close_program(program);
                return Err(format!(
                    "program {program}'s fires need {} bytes and {} bytes a lane beside the \
                     model's activation, and the smallest rung would then take {smallest} chunks \
                     of the {most} the device has for it",
                    term.fixed, term.a_lane
                ));
            }
        }
        Ok(())
    }

    /// One instance of `program` less can be in a fire; its term drops with
    /// the last.
    pub fn close_program(&mut self, program: ProgramId) {
        let Some(open) = self.programs.open.get_mut(&program) else {
            return;
        };
        *open -= 1;
        if *open == 0 {
            self.programs.open.remove(&program);
            self.refold();
        }
    }

    /// The engine bound `instance` of `program`: the program's term counts
    /// until it closes.
    pub fn instance_bound(&mut self, instance: InstanceId, program: ProgramId) {
        if self.programs.instances.insert(instance, program).is_none() {
            self.open_or_count(program);
        }
    }

    /// The engine closed `instance`.
    pub fn instance_closed(&mut self, instance: InstanceId) {
        if let Some(program) = self.programs.instances.remove(&instance) {
            self.close_program(program);
        }
    }

    /// Counts one more of `program` without the room check: the engine
    /// already bound it.
    fn open_or_count(&mut self, program: ProgramId) {
        let first = !self.programs.open.contains_key(&program);
        *self.programs.open.entry(program).or_default() += 1;
        if first {
            self.refold();
        }
    }

    fn refold(&mut self) {
        let mut terms = self.programs.open_terms();
        terms.retain(|term| !term.is_zero());
        terms.sort_unstable();
        if terms != self.pool.fire() {
            self.pool.set_fire(terms);
        }
    }

    /// The chunks the activation lacks for fires of `rows` (the rung that
    /// holds them, the fire term's included); 0 where it holds them.
    #[must_use]
    pub fn span_short(&self, rows: u32) -> u32 {
        let rung = self.rung_for(rows);
        self.pool
            .act_chunks(rung)
            .saturating_sub(self.pool.act_reserved())
    }
}

/// A bind of an instance of a program under way: the program's term counts
/// while it is held. The engine's bind counts the instance itself
/// ([`SharedPool::instance_bound`]), so the guard is dropped once the bind
/// is through, either way.
pub struct ProgramOpen {
    shared: Shared,
    program: ProgramId,
}

impl ProgramOpen {
    pub fn new(shared: &Shared, program: ProgramId) -> Result<Self, String> {
        shared.lock().unwrap().open_program(program)?;
        Ok(Self {
            shared: Shared::clone(shared),
            program,
        })
    }
}

impl Drop for ProgramOpen {
    fn drop(&mut self) {
        self.shared.lock().unwrap().close_program(self.program);
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
}

/// Raises `shared`'s activation to `goal`'s (a rung's tokens, given the
/// pool) where it falls short and nothing else is raising it: planned and
/// taken in one hold of the pool, so no slot is handed out between, and the
/// engine's copy of what it moves follows before fires pass the rung held
/// before. The raise taken, if any.
pub async fn take_raise(
    shared: &Shared,
    goal: impl FnOnce(&SharedPool) -> u32,
    mover: &impl Mover,
) -> Result<Option<RungEvacuation>, String> {
    let evacuation = {
        let mut pool = shared.lock().unwrap();
        let goal = goal(&pool);
        (pool.pool().act_chunks(goal) > pool.pool().act_reserved() && pool.copying().is_none())
            .then(|| pool.plan_raise(goal))
            .and_then(Result::ok)
            .filter(|evacuation| pool.raise_applied(evacuation).is_ok())
    };
    let Some(evacuation) = evacuation else {
        return Ok(None);
    };
    if !carry_raise(shared, &evacuation, mover).await {
        return Ok(None);
    }
    tracing::info!(
        rung = evacuation.rung,
        moved = evacuation.moves.len(),
        "the activation rung is raised"
    );
    Ok(Some(evacuation))
}

/// Carries a raise `shared` applied ([`SharedPool::raise_applied`]) to the
/// engine: where the span is backed on demand, `set_rung` first (the engine
/// backs its carve for the wider rung), then the copy of what it moves.
/// Either refused, the raise is taken back ([`SharedPool::raise_refused`]),
/// fires stay at the rung in hand, and an engine that backed the carve is
/// told the rung went back; `false` then. Nothing faults.
pub async fn carry_raise(shared: &Shared, evacuation: &RungEvacuation, mover: &impl Mover) -> bool {
    let (waits, tells) = {
        let pool = shared.lock().unwrap();
        (pool.raise_waits(evacuation), pool.tells_rungs())
    };
    if !waits {
        return true;
    }
    let (from, to) = (evacuation.from, evacuation.rung);
    if tells && let Err(error) = mover.set_rung(from, to).await {
        tracing::warn!(%error, from, to, "a raise's carve was refused; fires stay at the rung in hand");
        shared.lock().unwrap().raise_refused(evacuation);
        return false;
    }
    if !evacuation.moves.is_empty()
        && let Err(error) = mover.evacuate(SharedPool::rung_moves(evacuation)).await
    {
        tracing::warn!(%error, "a raise's copy was refused; fires stay at the rung in hand");
        shared.lock().unwrap().raise_refused(evacuation);
        if tells && let Err(error) = mover.set_rung(to, from).await {
            tracing::warn!(%error, from = to, to = from, "a rung notice was refused");
        }
        return false;
    }
    let next = mover.epoch_now().saturating_add(1);
    shared.lock().unwrap().raise_copied(next);
    true
}

/// Waits until `shared`'s activation holds what fires may take now, the
/// fire terms of the programs that can be in a fire included: fires cut to
/// `cap()` rows, and no wider than the widest rung that fits with the terms
/// ([`SharedPool::widest_fit`]). Where the span falls short, the room is
/// asked of `funder` as a rung raise's is, then the raise taken: every
/// state chunk in the widened span moved below it and the engine's copy
/// done before this returns. Where fires are cut wider than the widest
/// rung that fits, it waits on the walk to cut them
/// ([`super::SharedGrow`]). What a bind of a program whose term raised the
/// span waits on before it is sent; an `Err` refuses the bind.
pub async fn fund_span(
    shared: &Shared,
    cap: impl Fn() -> u32,
    funder: &impl Funder,
    mover: &impl Mover,
) -> Result<(), String> {
    loop {
        let (short, copying, goal, capped) = {
            let pool = shared.lock().unwrap();
            let rows = cap();
            let goal = pool.rung_for(rows);
            (
                pool.span_short(rows),
                pool.copying().is_some(),
                goal,
                rows <= pool.lifted(goal),
            )
        };
        if short == 0 && !copying && capped {
            return Ok(());
        }
        if copying || short == 0 {
            // A raise's copy is the engine's to take, or the walk is to cut
            // fires to the widest rung that fits.
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
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
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                continue;
            }
        }
        if take_raise(shared, |pool| pool.rung_for(cap()), mover)
            .await?
            .is_none()
        {
            // The walk is raising it, or the room went to another since.
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
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
        let demand = crate::planner::Demand {
            kv_pages: pages,
            rs: crate::store::rs::RsUnits::default(),
        };
        match planner.acquire(self.pid, self.quorum, demand).await {
            Ok(crate::planner::Acquired::Granted(grant)) => {
                // Given back at once: the raise takes the room it freed.
                drop(grant);
                Ok(true)
            }
            Ok(crate::planner::Acquired::Requote | crate::planner::Acquired::Yield) => Ok(false),
            Err(error) => Err(error.to_string()),
        }
    }
}

/// The runtime's [`Mover`]: engine `engine`'s lane, and its stores' fire
/// epoch.
pub struct LaneMover<F> {
    pub engine: usize,
    pub epoch_now: F,
}

impl<F: Fn() -> u64 + Sync> Mover for LaneMover<F> {
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
