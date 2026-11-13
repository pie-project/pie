//! The governor's [`PoolTarget`] for an engine whose stores draw from a
//! [`SharedPool`]: the target in segments, moved toward off the fire path
//! by a task that speaks to the engine through its scheduler lane, so each
//! commit, evacuation and release is ordered with the fires around it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

use tokio::sync::Notify;

use super::{RungStep, RungWalk, SegmentMove, Shared, SharedPool};
use crate::governor::{Demand, Growth, PoolTarget, Retarget};
use crate::scheduler::worker::PoolOp;
use crate::store::registry;

/// How often the task looks again while a segment drains or a waiter is
/// short, beside the wakes a retarget gives it.
const POLL: Duration = Duration::from_millis(50);

#[derive(Clone)]
pub struct SharedGrow {
    model: usize,
    engine: usize,
    shared: Shared,
    /// The governor's target, in mapped segments.
    target: Arc<AtomicU32>,
    wake: Arc<Notify>,
    refused: Arc<AtomicBool>,
    /// The last shrink found no room.
    no_room: Arc<AtomicBool>,
    /// The walk of the activation rung toward what the passes still to run
    /// ask of a fire (SPEC decision 6, by demand): the smallest rung while
    /// none asks past it.
    rung_walk: Arc<std::sync::Mutex<RungWalk>>,
}

impl SharedGrow {
    /// Steers `shared`, engine `engine`'s, from the segments mapped at load.
    #[must_use]
    pub fn new(model: usize, engine: usize, shared: Shared) -> Self {
        let mapped = shared.lock().unwrap().mapped_segments();
        Self {
            model,
            engine,
            shared,
            target: Arc::new(AtomicU32::new(mapped)),
            wake: Arc::new(Notify::new()),
            refused: Arc::new(AtomicBool::new(false)),
            no_room: Arc::new(AtomicBool::new(false)),
            rung_walk: Arc::new(std::sync::Mutex::new(RungWalk::default())),
        }
    }

    /// Steps the activation rung toward the rung the passes still to run
    /// ask of a fire ([`super::want_rows`]), the smallest while none asks
    /// past it, until there or until a step has to wait: caps fires, lowers
    /// the pool's rung once no wider fire is in flight, or raises it, the
    /// engine first copying the state out of the larger span. A raise short
    /// of room waits on the room the asking pass has the planner free.
    async fn walk_rung(&self) {
        loop {
            let (current, retired) = self.epochs();
            let (step, goal) = {
                let mut pool = self.shared.lock().unwrap();
                pool.pool_mut().retire_through(retired);
                let goal = pool.rung_for(super::wanted_rows(self.engine));
                let step = self.rung_walk.lock().unwrap().step(
                    &pool,
                    goal,
                    super::fire_cap(self.engine),
                    super::rung_granted(self.engine),
                    current,
                    retired,
                );
                (step, goal)
            };
            let step = match step {
                Ok(step) => step,
                Err(error) => {
                    // The pass asking funds the room as a planner waiter
                    // (`pipeline::fire`'s raise funding).
                    tracing::debug!(engine = self.engine, %error, goal, "the activation rung waits");
                    return;
                }
            };

            match step {
                RungStep::Wait => {
                    // A lower waits on the scheduler's grant, which it
                    // publishes each turn: an idle scheduler takes none
                    // until woken.
                    if super::rung_granted(self.engine) > super::fire_cap(self.engine) {
                        crate::scheduler::nudge(self.engine);
                    }
                    return;
                }
                RungStep::Cap(tokens) => {
                    {
                        // Under the pool's lock, as a pass's `raise_now`
                        // lifts it: a cap down past what a pass asks since
                        // is not taken.
                        let _pool = self.shared.lock().unwrap();
                        let now = super::fire_cap(self.engine);
                        if tokens < now && super::wanted_rows(self.engine) > tokens {
                            continue;
                        }
                        super::set_fire_cap(self.engine, tokens);
                    }
                    tracing::info!(
                        engine = self.engine,
                        tokens,
                        "fires cut to the activation rung"
                    );
                    crate::scheduler::nudge(self.engine);
                }
                RungStep::Lower(rung) => {
                    let lowered = {
                        let mut pool = self.shared.lock().unwrap();
                        // A pass lifted the cap since the walk capped it.
                        if super::fire_cap(self.engine) > rung {
                            continue;
                        }
                        pool.lower_rung(rung)
                    };
                    if let Err(error) = lowered {
                        tracing::warn!(engine = self.engine, %error, "the activation rung stays");
                        return;
                    }
                    tracing::info!(engine = self.engine, rung, "the activation rung is lowered");
                    if let Some(planner) = crate::planner::planner_for(self.model, self.engine) {
                        planner.pages_freed();
                    }
                }
                RungStep::Raise { rung, evacuation } => {
                    if !evacuation.moves.is_empty() {
                        let moves = SharedPool::rung_moves(&evacuation);
                        let copied =
                            crate::scheduler::pool(self.engine, PoolOp::Evacuate(moves)).await;
                        if let Err(error) = copied {
                            tracing::warn!(engine = self.engine, %error, "a rung's evacuation was refused");
                            return;
                        }
                    }
                    let (current, _) = self.epochs();
                    let applied = self
                        .shared
                        .lock()
                        .unwrap()
                        .apply_rung(&evacuation, current.saturating_add(1));
                    if let Err(error) = applied {
                        tracing::warn!(engine = self.engine, %error, "the activation rung stays");
                        return;
                    }
                    tracing::info!(
                        engine = self.engine,
                        rung,
                        moved = evacuation.moves.len(),
                        "the activation rung is raised"
                    );
                }
            }
        }
    }

    /// Holds the segments mapped now as the target: what the load
    /// committed for the weights is not given back before the governor
    /// first says otherwise.
    pub fn hold_mapped(&self) {
        let mapped = self.shared.lock().unwrap().mapped_segments();
        self.target.fetch_max(mapped, Ordering::AcqRel);
    }

    #[must_use]
    pub fn shared(&self) -> &Shared {
        &self.shared
    }

    /// Looks again now: a waiter is short.
    pub fn wake(&self) {
        self.wake.notify_one();
    }

    /// Whether a segment below the top span is still uncommitted, under the
    /// governor's target.
    #[must_use]
    pub fn can_grow(&self) -> bool {
        let pool = self.shared.lock().unwrap();
        let top = pool.pool().top_segment();
        pool.pool().prefix_segments() < top
            && pool.mapped_segments() < self.target.load(Ordering::Acquire)
    }

    /// Starts moving; the scheduler that takes the moves is up.
    pub fn start(&self) {
        crate::rt::spawn(self.clone().run());
        self.wake.notify_one();
    }

    /// The fire epochs of the engine's stores: the last handed out, and
    /// the last retired.
    fn epochs(&self) -> (u64, u64) {
        registry::try_get(self.model, self.engine).map_or((0, u64::MAX), |stores| {
            registry::with_kv_lock(&stores.kv, "shared-grow", |kv| {
                (kv.current_epoch(), kv.retired_through())
            })
        })
    }

    /// The segments to be at: the governor's target, and on the way up no
    /// further than what stands ready ahead of the waiters asks.
    fn goal(&self) -> u32 {
        let pinned = pinned_segments();
        let target = pinned.unwrap_or_else(|| self.target.load(Ordering::Acquire));
        let pool = self.shared.lock().unwrap();
        let mapped = pool.mapped_segments();
        if pinned.is_some() {
            return target;
        }
        if target <= mapped {
            return target;
        }
        let short = pool
            .pool()
            .demand()
            .wanted
            .saturating_sub(pool.pool().demand().live);
        let ahead = pool.ahead_target(short);
        tracing::trace!(
            target,
            mapped,
            short,
            ahead,
            headroom = pool.pool().headroom(),
            "segment goal"
        );
        target.min(ahead)
    }

    async fn run(self) {
        loop {
            let _ = tokio::time::timeout(POLL, self.wake.notified()).await;
            self.walk().await;
        }
    }

    /// Steps toward the goal until there, or until a move has to wait.
    async fn walk(&self) {
        self.walk_rung().await;
        loop {
            let goal = self.goal();
            let (current, retired) = self.epochs();
            let step = {
                let mut pool = self.shared.lock().unwrap();
                pool.pool_mut().retire_through(retired);
                // What an evacuation moves is read last by the next fire.
                pool.next_move(goal, current.saturating_add(1))
            };
            let op = match step {
                None | Some(SegmentMove::Wait) => return,
                Some(SegmentMove::NoRoom) => {
                    self.no_room.store(true, Ordering::Release);
                    return;
                }
                Some(SegmentMove::Commit { prefix }) => PoolOp::CommitSegments(prefix),
                Some(SegmentMove::Evacuate { evacuation }) => PoolOp::Evacuate(evacuation),
                Some(SegmentMove::Release { prefix }) => PoolOp::ReleaseSegments(prefix),
            };
            self.no_room.store(false, Ordering::Release);
            tracing::debug!(engine = self.engine, goal, ?op, "a segment move");
            let commit = match &op {
                PoolOp::CommitSegments(prefix) => Some(*prefix),
                _ => None,
            };
            let evacuating = matches!(op, PoolOp::Evacuate(_));
            match crate::scheduler::pool(self.engine, op).await {
                Ok(()) => {
                    if let Some(prefix) = commit {
                        self.shared.lock().unwrap().committed(prefix);
                        if let Some(planner) = crate::planner::planner_for(self.model, self.engine)
                        {
                            planner.pages_freed();
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(engine = self.engine, %error, "a segment move was refused");
                    if commit.is_some() {
                        self.refused.store(true, Ordering::Release);
                    }
                    return;
                }
            }
            if evacuating {
                // The segment drains until the first fire handed out after
                // the engine took the copy retires.
                let (current, _) = self.epochs();
                self.shared.lock().unwrap().evacuated(current);
                return;
            }
        }
    }
}

/// A test hook: while the file `PIE_SHARED_SEGMENTS_FILE` names holds a
/// number, that many mapped segments stand in for the governor's target, so
/// a test can force a grow or a shrink (and its evacuation) mid-run.
fn pinned_segments() -> Option<u32> {
    static PATH: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    let path = PATH
        .get_or_init(|| std::env::var_os("PIE_SHARED_SEGMENTS_FILE").map(Into::into))
        .as_ref()?;
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

impl PoolTarget for SharedGrow {
    fn demand(&mut self) -> Demand {
        self.shared.lock().unwrap().demand()
    }

    fn segment_bytes(&self) -> u64 {
        self.shared.lock().unwrap().segment_bytes()
    }

    fn floor(&self) -> u64 {
        self.shared.lock().unwrap().floor()
    }

    fn hard_limit(&self) -> u64 {
        self.shared.lock().unwrap().hard_limit()
    }

    fn held(&self) -> u64 {
        self.shared.lock().unwrap().held()
    }

    fn growth(&self) -> Growth {
        self.shared.lock().unwrap().growth()
    }

    fn retarget(&mut self, segments: u64) -> Retarget {
        let segments = u32::try_from(segments).unwrap_or(u32::MAX);
        self.target.store(segments, Ordering::Release);
        self.wake.notify_one();
        let mapped = self.shared.lock().unwrap().mapped_segments();
        if self.no_room.load(Ordering::Acquire) && segments < mapped {
            Retarget::NoRoom {
                held: u64::from(mapped),
            }
        } else {
            Retarget::Reached
        }
    }

    fn take_refused(&mut self) -> bool {
        self.refused.swap(false, Ordering::AcqRel)
    }

}
