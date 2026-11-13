//! One engine's pool as its governor steers it: its shared pool's segments
//! (nothing to move where the pools are pinned), with the weight cache in
//! front ([`WeightFirst`]), shared with the engine's
//! planner, which sheds the cache before it raises tier pressure.

use std::sync::Arc;

use crate::policy::pressure::{self, Tier};
use crate::store::registry;
use crate::store::shared::SharedGrow;
use crate::store::slots::SlotPool;
use crate::store::weights::port::EnginePort;
use crate::store::weights::{SlotAccess, WeightFirst};

use super::{Demand, Growth, PoolTarget, Retarget, Signal};

/// Today's pools of one engine, and the slots its weight shapes take; or
/// the shared pool an engine that states one runs on, the weight shapes'
/// slots among the state's.
pub struct EnginePool {
    model: usize,
    engine: usize,
    weights: SlotPool,
    shared: Option<SharedGrow>,
}

impl EnginePool {
    #[must_use]
    pub const fn new(model: usize, engine: usize, weights: SlotPool) -> Self {
        Self {
            model,
            engine,
            weights,
            shared: None,
        }
    }

    /// An engine's shared pool, which `grow` moves.
    #[must_use]
    pub fn shared(model: usize, engine: usize, grow: SharedGrow) -> Self {
        Self {
            model,
            engine,
            weights: crate::store::weights::slot_pool(&::engine::chunk::ChunkFacts::default()),
            shared: Some(grow),
        }
    }

    #[must_use]
    pub const fn engine(&self) -> usize {
        self.engine
    }

    /// Whether the engine runs on a shared pool.
    #[must_use]
    pub const fn is_shared(&self) -> bool {
        self.shared.is_some()
    }

    #[must_use]
    pub const fn locus(&self) -> (usize, usize) {
        (self.model, self.engine)
    }
}

impl SlotAccess for EnginePool {
    /// The weight slots, at the epoch of the engine's last fire; what waited
    /// on fires that have retired is free first. With no stores, no fire
    /// reads them.
    fn with_slots(&mut self, f: &mut dyn FnMut(&mut SlotPool, u64)) {
        let (epoch, retired) =
            registry::try_get(self.model, self.engine).map_or((0, u64::MAX), |stores| {
                registry::with_kv_lock(&stores.kv, "weights", |kv| {
                    (kv.current_epoch(), kv.retired_through())
                })
            });
        if let Some(shared) = &self.shared {
            let mut pool = shared.shared().lock().unwrap();
            pool.pool_mut().retire_through(retired);
            f(pool.pool_mut(), epoch);
            return;
        }
        self.weights.retire_through(retired);
        f(&mut self.weights, epoch);
    }
}

impl EnginePool {
    /// The target that moves: the shared pool's; pinned pools never move.
    fn target(&self) -> Option<&dyn PoolTarget> {
        self.shared.as_ref().map(|shared| shared as &dyn PoolTarget)
    }

    fn target_mut(&mut self) -> Option<&mut dyn PoolTarget> {
        self.shared
            .as_mut()
            .map(|shared| shared as &mut dyn PoolTarget)
    }
}

impl PoolTarget for EnginePool {
    fn demand(&mut self) -> Demand {
        self.target_mut()
            .map_or(Demand { wanted: 0, live: 0 }, PoolTarget::demand)
    }

    fn segment_bytes(&self) -> u64 {
        self.target().map_or(1, PoolTarget::segment_bytes)
    }

    fn floor(&self) -> u64 {
        self.target().map_or(0, PoolTarget::floor)
    }

    /// The pool's range, less what the engine holds beside it for fires and
    /// bodies now ([`::engine::live::Live::held`]), never under the floor.
    /// The range stated at load already leaves out what the driver holds
    /// (`outside`, a ledger term the policy reads, not subtracted here).
    fn hard_limit(&self) -> u64 {
        self.target().map_or(0, |target| {
            let held = crate::engine::live(self.engine).held();
            target
                .hard_limit()
                .saturating_sub(held)
                .max(target.floor())
        })
    }

    fn held(&self) -> u64 {
        self.target().map_or(0, PoolTarget::held)
    }

    fn growth(&self) -> Growth {
        self.target().map_or(Growth::Fixed, PoolTarget::growth)
    }

    fn retarget(&mut self, segments: u64) -> Retarget {
        self.target_mut()
            .map_or(Retarget::Reached, |grow| grow.retarget(segments))
    }

    fn take_refused(&mut self) -> bool {
        self.target_mut().is_some_and(PoolTarget::take_refused)
    }

    fn ticked(&mut self) {
        if let Some(grow) = self.target_mut() {
            grow.ticked();
        }
    }
}

/// An engine's governor target, the weight cache first.
pub type Steered = WeightFirst<EnginePool, EnginePort>;

impl Shared<Steered> {
    /// Tells the engine the resident set it starts with and waits until it
    /// took it: what the load does before a fire can be submitted, so none
    /// reads from the host a row the cache holds.
    ///
    /// # Errors
    ///
    /// The engine refused it, or its lane closed first. An engine that does
    /// not stream (it answers unsupported) is no error.
    pub async fn boot(&self) -> anyhow::Result<()> {
        // The segments the weights were committed into past the load's
        // own: the engine backs them before it is told where the weights
        // lie, so it fills backed memory.
        let (engine, backing) = self.with(|pool| (pool.inner().engine(), pool.take_backing()));
        if let Some(prefix) = backing {
            crate::scheduler::pool(
                engine,
                crate::scheduler::worker::PoolOp::CommitSegments(prefix),
            )
            .await
            .map_err(|error| {
                anyhow::anyhow!(
                    "the engine did not back the {prefix} segments its weights take: {error}"
                )
            })?;
        }
        let queued = self.with(|pool| -> anyhow::Result<_> {
            pool.boot()?;
            Ok(pool.port_mut().take_queued())
        })?;
        for answer in queued {
            let answer = answer
                .await
                .unwrap_or_else(|_| Err(anyhow::anyhow!("the engine's lane closed")));
            self.with(|pool| pool.port_mut().answered(answer))?;
        }
        Ok(())
    }
}

/// A pool target the governor's loop and the planner both reach: each
/// call takes the lock for itself.
pub struct Shared<P>(Arc<parking_lot::Mutex<P>>);

impl<P> Clone for Shared<P> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<P> Shared<P> {
    pub fn new(pool: P) -> Self {
        Self(Arc::new(parking_lot::Mutex::new(pool)))
    }

    pub fn with<T>(&self, f: impl FnOnce(&mut P) -> T) -> T {
        f(&mut self.0.lock())
    }
}

impl<P: PoolTarget + SlotAccess, W: crate::store::weights::WeightPort> Shared<WeightFirst<P, W>> {
    /// The planner's hook: sheds the weight cache to its floor (held, it
    /// offers the engine the streaming set first, and gives nothing back
    /// until the engine took it); the slots given back.
    pub fn shed(&self) -> u64 {
        self.with(WeightFirst::shed)
    }

    /// [`Self::shed`] as the planner takes it.
    pub fn shed_hook(&self) -> crate::planner::WeightShed
    where
        P: Send + 'static,
        W: Send + 'static,
    {
        let shared = self.clone();
        Arc::new(move || shared.shed())
    }
}

impl<P: PoolTarget> PoolTarget for Shared<P> {
    fn demand(&mut self) -> Demand {
        self.0.lock().demand()
    }

    fn segment_bytes(&self) -> u64 {
        self.0.lock().segment_bytes()
    }

    fn floor(&self) -> u64 {
        self.0.lock().floor()
    }

    fn hard_limit(&self) -> u64 {
        self.0.lock().hard_limit()
    }

    fn held(&self) -> u64 {
        self.0.lock().held()
    }

    fn growth(&self) -> Growth {
        self.0.lock().growth()
    }

    fn retarget(&mut self, segments: u64) -> Retarget {
        self.0.lock().retarget(segments)
    }

    fn take_refused(&mut self) -> bool {
        self.0.lock().take_refused()
    }

    fn ticked(&mut self) {
        self.0.lock().ticked();
    }
}

/// The gauge entry the governor of `engine` speaks under: apart from its
/// planner's (which is `engine`), so neither clears what the other said.
#[must_use]
pub const fn source(engine: usize) -> usize {
    usize::MAX - engine
}

/// What `signal` tells the device tier: the bytes the pool holds past its
/// target with no room to give back, or 0 once it reached it.
#[must_use]
pub const fn short_bytes(signal: Signal) -> u64 {
    match signal {
        Signal::TierPressure {
            held_bytes,
            target_bytes,
        } => held_bytes.saturating_sub(target_bytes),
        Signal::Eased => 0,
    }
}

/// Tells the policy of `signal` through `raise` ([`pressure::raise`] in
/// the runtime): `tier-pressure` on the device, or its relief.
pub fn tell(engine: usize, signal: Signal, raise: &mut impl FnMut(Tier, usize, u64)) {
    let bytes = short_bytes(signal);
    if bytes == 0 && signal != Signal::Eased {
        // A pool short of nothing it could say in bytes is still short.
        raise(Tier::Device, source(engine), 1);
    } else {
        raise(Tier::Device, source(engine), bytes);
    }
}

/// [`tell`], to the policy's gauge.
pub fn signal(engine: usize, signal: Signal) {
    tracing::debug!(engine, ?signal, "governor pool short of its target");
    tell(engine, signal, &mut |tier, from, bytes| {
        if bytes == 0 {
            pressure::relieve(tier, from);
        } else {
            pressure::raise(tier, from, bytes);
        }
    });
}
