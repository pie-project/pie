pub(crate) mod genmap;
pub(crate) mod kv;
pub(crate) mod pool;
pub(crate) mod registry;
pub(crate) mod rs;
pub(crate) mod seat;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PipelineScopeId(u128);

#[derive(Clone)]
pub(crate) struct PipelineScope {
    state: std::sync::Arc<PipelineScopeState>,
}

struct PipelineScopeState {
    id: PipelineScopeId,
    closed: std::sync::atomic::AtomicBool,
    drained: Box<dyn Fn() -> bool + Send + Sync>,
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
    pub(crate) fn new(drained: impl Fn() -> bool + Send + Sync + 'static) -> Self {
        static NEXT_SCOPE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            state: std::sync::Arc::new(PipelineScopeState {
                id: PipelineScopeId(u128::from(
                    NEXT_SCOPE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                )),
                closed: std::sync::atomic::AtomicBool::new(false),
                drained: Box::new(drained),
            }),
        }
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

    pub(crate) fn claim(
        &self,
        owner: &std::sync::Mutex<Option<PipelineScope>>,
    ) -> Result<(), PipelineScopeId> {
        let mut owner = owner.lock().unwrap();
        match owner.as_ref() {
            Some(existing) if existing.id() == self.id() => Ok(()),
            Some(existing) if !self.is_closed() && existing.is_releasable() => {
                *owner = Some(self.clone());
                Ok(())
            }
            Some(existing) => Err(existing.id()),
            None if self.is_closed() => Err(self.id()),
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
    let mut capacity = 0u64;
    let mut used = 0u64;
    for stores in registry::all_for_model(0) {
        let page = u64::from(stores.kv_page_size);
        let (total, available, cached) = registry::with_kv_lock(&stores.kv, "other", |kv| {
            let cached: u64 = kv.cached_working_sets().iter().map(|set| set.pages).sum();
            (
                u64::from(kv.capacity_pages()),
                kv.available_pages() as u64,
                cached,
            )
        });
        capacity += total * page;
        used += total.saturating_sub(available).saturating_sub(cached) * page;
    }
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
        let free = kv.available_pages();
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
    let (total, available) = registry::with_kv_lock(&stores.kv, "other", |kv| {
        (kv.capacity_pages(), kv.available_pages())
    });
    if total == 0 {
        return 0;
    }
    let used = total.saturating_sub(available as u32);
    (f64::from(used) / f64::from(total) * 255.0).round() as u8
}
