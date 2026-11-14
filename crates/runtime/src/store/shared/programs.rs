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
    /// The sum over the programs that can be in a fire now.
    fn sum(&self) -> FireTerm {
        self.open
            .keys()
            .filter_map(|program| self.terms.get(program))
            .fold(FireTerm::default(), |sum, &term| sum.plus(term))
    }
}

/// A program's fire term from its facts.
#[must_use]
pub fn term_of(facts: &ProgramFacts) -> FireTerm {
    FireTerm {
        fixed: facts.fire_bytes_fixed,
        a_lane: facts.fire_bytes_a_lane,
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

    /// One more instance of `program` can be in a fire: its term counts
    /// from now. Refused where the term leaves no rung room on the device
    /// at all (the smallest rung's activation past every chunk but the
    /// reserved ones and the weights' pinned slots).
    pub fn open_program(&mut self, program: ProgramId) -> Result<(), String> {
        let first = !self.programs.open.contains_key(&program);
        *self.programs.open.entry(program).or_default() += 1;
        if first {
            self.refold();
            let room = self.room();
            let most = room
                .total_chunks
                .saturating_sub(room.reserved_chunks)
                .saturating_sub(room.weight_chunks);
            if room.act_chunks > most {
                let term = self
                    .programs
                    .terms
                    .get(&program)
                    .copied()
                    .unwrap_or_default();
                self.close_program(program);
                return Err(format!(
                    "program {program}'s fires need {} bytes and {} bytes a lane beside the \
                     model's activation, and the smallest rung would then take {} chunks of the \
                     {most} the device has for it",
                    term.fixed, term.a_lane, room.act_chunks
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
        let sum = self.programs.sum();
        if sum != self.pool.fire() {
            self.pool.set_fire(sum);
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
    if !evacuation.moves.is_empty() {
        mover
            .evacuate(SharedPool::rung_moves(&evacuation))
            .await
            .map_err(|error| format!("a raise's copy was refused; the rung stays held: {error}"))?;
        let next = mover.epoch_now().saturating_add(1);
        shared.lock().unwrap().raise_copied(next);
    }
    tracing::info!(
        rung = evacuation.rung,
        moved = evacuation.moves.len(),
        "the activation rung is raised"
    );
    Ok(Some(evacuation))
}

/// Waits until `shared`'s activation holds what fires may take now (fires
/// cut to `cap()` rows), the registered programs' fire term included:
/// where it falls short, the room is asked of `funder` as a rung raise's is,
/// then the raise taken. What a bind of a program whose term raised the
/// span waits on before it proceeds.
pub async fn fund_span(
    shared: &Shared,
    cap: impl Fn() -> u32,
    funder: &impl Funder,
    mover: &impl Mover,
) -> Result<(), String> {
    loop {
        let (short, copying, goal) = {
            let pool = shared.lock().unwrap();
            let rows = cap();
            (
                pool.span_short(rows),
                pool.copying().is_some(),
                pool.rung_for(rows),
            )
        };
        if short == 0 && !copying {
            return Ok(());
        }
        if copying {
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
