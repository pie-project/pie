//! The governor's [`PoolTarget`] for an engine whose stores draw from a
//! [`SharedPool`]: the target in segments, moved toward off the fire path
//! by a task that speaks to the engine through its scheduler lane, so each
//! commit, evacuation and release is ordered with the fires around it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

use tokio::sync::Notify;

use super::{SegmentMove, Shared};
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
        }
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

    /// Fires keep the largest rung: the scheduler does not cap a fire's
    /// tokens by rung yet, so state never takes the top span and no rung
    /// moves. [`super::SharedPool::lower_rung`] and
    /// [`super::SharedPool::raise_rung`] are ready for when it does.
    fn shrink_rung(&mut self) {}

    fn restore_rung(&mut self) {}
}
