pub mod chunks;
pub(crate) mod genmap;
pub(crate) mod kv;
pub(crate) mod pool;
pub(crate) mod registry;
pub(crate) mod roles;
pub(crate) mod rs;
pub mod segments;
pub mod shared;
pub mod slots;
pub mod waiting;
pub mod weights;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PipelineScopeId(u128);

/// Why a pipeline did not take a working set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ScopeRefusal {
    /// Another pipeline holds the set, or the one asking is closed.
    Held(PipelineScopeId),
    /// The pipeline that owns the set failed, for `reason`, its first
    /// failure. A fire that fails, refused or faulted, ends its pipeline
    /// and the sets it owns.
    Failed {
        owner: PipelineScopeId,
        reason: String,
    },
}

impl std::fmt::Display for ScopeRefusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScopeRefusal::Held(owner) => {
                write!(formatter, "is already scoped to pipeline {owner:032x}")
            }
            ScopeRefusal::Failed { owner, reason } => write!(
                formatter,
                "ended with its pipeline {owner:032x}, which failed ({reason}): what \
                 that pipeline's fires folded is frozen and did not run, so no pipeline \
                 takes it"
            ),
        }
    }
}

#[derive(Clone)]
pub(crate) struct PipelineScope {
    state: std::sync::Arc<PipelineScopeState>,
}

struct PipelineScopeState {
    id: PipelineScopeId,
    closed: std::sync::atomic::AtomicBool,
    drained: Box<dyn Fn() -> bool + Send + Sync>,
    /// Why its pipeline failed, once it has.
    failure: crate::pipeline::fire::PipelineFailure,
}

impl std::fmt::Debug for PipelineScope {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PipelineScope")
            .field("id", &self.id())
            .field("closed", &self.is_closed())
            .finish_non_exhaustive()
    }
}

impl PipelineScope {
    #[cfg(test)]
    pub(crate) fn new(drained: impl Fn() -> bool + Send + Sync + 'static) -> Self {
        Self::failing_as(Default::default(), drained)
    }

    /// A scope whose pipeline records its failure in `failure`.
    pub(crate) fn failing_as(
        failure: crate::pipeline::fire::PipelineFailure,
        drained: impl Fn() -> bool + Send + Sync + 'static,
    ) -> Self {
        static NEXT_SCOPE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            state: std::sync::Arc::new(PipelineScopeState {
                id: PipelineScopeId(u128::from(
                    NEXT_SCOPE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                )),
                closed: std::sync::atomic::AtomicBool::new(false),
                drained: Box::new(drained),
                failure,
            }),
        }
    }

    pub(crate) fn failure(&self) -> Option<String> {
        self.state.failure.lock().unwrap().clone()
    }

    pub(crate) fn id(&self) -> PipelineScopeId {
        self.state.id
    }

    pub(crate) fn scheduler_id(&self) -> uuid::Uuid {
        uuid::Uuid::from_u128(self.state.id.0)
    }

    pub(crate) fn close(&self) -> bool {
        !self
            .state
            .closed
            .swap(true, std::sync::atomic::Ordering::AcqRel)
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.state.closed.load(std::sync::atomic::Ordering::Acquire)
    }

    fn is_releasable(&self) -> bool {
        self.is_closed() && (self.state.drained)()
    }

    /// Takes the set `owner` scopes: one no pipeline holds, or one whose
    /// pipeline closed with every fire of it settled. Not one whose
    /// pipeline failed. Its failed fire left the set's cursor past rows
    /// that were never written, and its records wherever the step stopped,
    /// whether the engine refused the step or the device faulted under it:
    /// the set ends with its pipeline, and fires on no other.
    pub(crate) fn claim(
        &self,
        owner: &std::sync::Mutex<Option<PipelineScope>>,
    ) -> Result<(), ScopeRefusal> {
        let mut owner = owner.lock().unwrap();
        match owner.as_ref() {
            Some(existing) if existing.id() == self.id() => Ok(()),
            // What a fire folds is frozen from its prepare: a set whose
            // pipeline failed holds a fold whose fire never ran.
            Some(existing) => {
                // Read ahead of the failure: a pipeline that is drained has
                // written every failure of its fires.
                let released = existing.is_releasable();
                match existing.failure() {
                    Some(reason) => Err(ScopeRefusal::Failed {
                        owner: existing.id(),
                        reason,
                    }),
                    None if !self.is_closed() && released => {
                        *owner = Some(self.clone());
                        Ok(())
                    }
                    None => Err(ScopeRefusal::Held(existing.id())),
                }
            }
            None if self.is_closed() => Err(ScopeRefusal::Held(self.id())),
            None => {
                *owner = Some(self.clone());
                Ok(())
            }
        }
    }
}

impl std::fmt::LowerHex for PipelineScopeId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::LowerHex::fmt(&self.0, formatter)
    }
}

/// Cached pages are reclaimable, so they count as free.
pub fn kv_tokens() -> (u64, u64) {
    // The report asks every 100 ms, and counting the cached pages walks every
    // cache entry under the kv lock every guest's fire waits on: the count
    // stands for a second.
    let mut last = KV_TOKENS_LAST.lock().unwrap();
    if let Some((at, tokens)) = *last
        && at.elapsed() < std::time::Duration::from_secs(1)
    {
        return tokens;
    }
    let mut capacity = 0u64;
    let mut used = 0u64;
    for stores in registry::all_for_model(0) {
        let page = u64::from(stores.kv_page_size);
        let (total, available, cached) = registry::with_kv_lock(&stores.kv, "other", |kv| {
            let cached: u64 = kv.cached_working_sets().iter().map(|set| set.pages).sum();
            (
                u64::from(kv.capacity_pages()),
                u64::from(kv.capacity_pages().saturating_sub(kv.pages_out())),
                cached,
            )
        });
        capacity += total * page;
        used += total.saturating_sub(available).saturating_sub(cached) * page;
    }
    *last = Some((crate::rt::Instant::now(), (capacity, used)));
    (capacity, used)
}

/// Device kv pages of model 0's first engine, by what holds them: free,
/// idle cache a reclaim would free now, the rest (working sets and pages
/// pinned by readers or moves), and the total. For the contention trace.
pub fn kv_census() -> (usize, usize, usize, usize) {
    let Some(stores) = registry::try_get(0, 0) else {
        return (0, 0, 0, 0);
    };
    registry::with_kv_lock(&stores.kv, "other", |kv| {
        let total = kv.capacity_pages() as usize;
        let free = (kv.capacity_pages().saturating_sub(kv.pages_out())) as usize;
        let idle = kv.reclaimable_cache_pages() as usize;
        (
            free,
            idle,
            total.saturating_sub(free).saturating_sub(idle),
            total,
        )
    })
}

pub fn kv_pressure_bucket() -> u8 {
    let planners = crate::planner::planners();
    if !planners.is_empty() {
        return planners
            .iter()
            .map(|planner| planner.kv_pressure_bucket())
            .max()
            .unwrap_or(0);
    }
    let Some(stores) = registry::try_get(0, 0) else {
        return 0;
    };
    let (total, used) = registry::with_kv_lock(&stores.kv, "other", |kv| {
        (kv.capacity_pages(), kv.pages_out())
    });
    if total == 0 {
        return 0;
    }
    (f64::from(used) / f64::from(total) * 255.0).round() as u8
}

/// Prompt tokens served from cached KV that a fire brought back up from
/// below the device (the host or disk tier), since the process started:
/// `/metrics`' `pie_kv_cached_from_below_tokens_total`.
pub static CACHED_FROM_BELOW_TOKENS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Forgets the boot's stores and what their engines were set to, for
/// [`crate::bootstrap::reset`].
pub(crate) fn reset() {
    registry::reset();
    shared::reset();
    waiting::reset();
    *KV_TOKENS_LAST.lock().unwrap() = None;
    CACHED_FROM_BELOW_TOKENS.store(0, std::sync::atomic::Ordering::Relaxed);
    kv::RELIST_COPIES.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// The last [`kv_tokens`] count and when it was taken.
static KV_TOKENS_LAST: std::sync::Mutex<Option<(crate::rt::Instant, (u64, u64))>> =
    std::sync::Mutex::new(None);
