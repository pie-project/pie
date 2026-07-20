//! Typed resource stores (kv_refact.md): device memory as pages, slots, and
//! mappings. Replaces the generic KV-page-sized `Arena` with resource-specific
//! stores over typed static backing pools.
//!
//! - [`kv`]: `KvStore` — the mapping trie, hash lifecycle, and prepare/commit/
//!   abort protocol over the physical KV page pool.
//! - `rs`: `RsStore` — the recurrent-state slot store (GDN/Mamba2 folded
//!   state) with the same prepare/commit/abort protocol.
//! - `pool`/`genmap`: the physical-id free list and generational key map the
//!   typed stores are built on.
//! - `registry`: per-(model, driver) lookup of the owning `KvStore`/`RsStore`.
//! - [`reclaim`]: the pressure ladder (idle-lease drop, preempt-youngest,
//!   wait queue, restore-on-free) — the only submodule external tests reach
//!   directly (`store::reclaim::contention()`), since it is this crate's
//!   sole KV-contention diagnostic surface.

pub(crate) mod genmap;
pub(crate) mod kv;
pub(crate) mod pool;
pub mod reclaim;
pub(crate) mod registry;
pub(crate) mod rs;

/// Stable identity for one pipeline ownership scope.
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

    pub(crate) fn close(&self) {
        self.state
            .closed
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.state.closed.load(std::sync::atomic::Ordering::Acquire)
    }

    fn is_releasable(&self) -> bool {
        self.is_closed() && (self.state.drained)()
    }
}

impl std::fmt::LowerHex for PipelineScopeId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::LowerHex::fmt(&self.0, formatter)
    }
}

/// Coarse worker-routing signal derived from real KV residency and contention.
pub fn kv_pressure_bucket() -> u8 {
    if let Some(orchestrator) = reclaim::contention() {
        return orchestrator.kv_pressure_bucket();
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
