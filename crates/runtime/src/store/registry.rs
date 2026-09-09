use std::sync::atomic::Ordering;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use super::kv::KvStore;
use super::rs::RsStore;
use super::seat::SeatBook;

static KV_TAINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

struct KvTaintOnPanic;

impl Drop for KvTaintOnPanic {
    fn drop(&mut self) {
        if std::thread::panicking() {
            KV_TAINTED.store(true, Ordering::SeqCst);
        }
    }
}

#[inline(always)]
pub fn with_kv_lock<T>(
    store: &parking_lot::Mutex<KvStore>,
    tag: &'static str,
    operation: impl FnOnce(&mut KvStore) -> T,
) -> T {
    assert!(
        !KV_TAINTED.load(Ordering::Relaxed),
        "KV store tainted by an earlier panic mid-mutation ({tag})"
    );
    let taint = KvTaintOnPanic;
    let mut guard = store.lock();
    let result = operation(&mut guard);
    drop(guard);
    drop(taint);
    result
}

#[derive(Clone)]
pub struct Stores {
    pub kv: Arc<parking_lot::Mutex<KvStore>>,
    pub rs: Arc<Mutex<RsStore>>,
    pub seats: Arc<Mutex<SeatBook>>,
    pub seats_freed: Arc<tokio::sync::Notify>,
    pub kv_page_size: u32,
    pub context_pages: u64,
}

fn context_pages(max_context: usize, kv_page_size: u32) -> u64 {
    (max_context as u64).div_ceil(u64::from(kv_page_size.max(1)))
}

static REGISTRY: LazyLock<boxcar::Vec<RwLock<Vec<Option<Stores>>>>> =
    LazyLock::new(boxcar::Vec::new);

#[cfg(test)]
pub fn register_model(kv_page_size: u32, num_kv_pages: &[usize], num_slots: &[usize]) -> usize {
    register_model_with_swap(
        kv_page_size,
        num_kv_pages,
        &vec![0; num_kv_pages.len()],
        num_slots,
        &vec![0; num_kv_pages.len()],
    )
}

pub fn register_model_with_swap(
    kv_page_size: u32,
    num_kv_pages: &[usize],
    num_host_pages: &[usize],
    num_slots: &[usize],
    max_context: &[usize],
) -> usize {
    let stores: Vec<Option<Stores>> = (0..num_kv_pages.len())
        .map(|d| {
            let kv = Arc::new(parking_lot::Mutex::new(KvStore::new_with_swap(
                num_kv_pages[d] as u32,
                num_host_pages.get(d).copied().unwrap_or(0) as u32,
                rand::random::<[u8; 32]>(),
            )));
            let slots = num_slots.get(d).copied().unwrap_or(0) as u32;
            let max_context = max_context.get(d).copied().unwrap_or(0);
            Some(Stores {
                kv,
                rs: Arc::new(Mutex::new(RsStore::new(slots))),
                seats: Arc::new(Mutex::new(SeatBook::new(slots))),
                seats_freed: Arc::new(tokio::sync::Notify::new()),
                kv_page_size,
                context_pages: context_pages(max_context, kv_page_size),
            })
        })
        .collect();
    REGISTRY.push(RwLock::new(stores))
}

#[allow(clippy::too_many_arguments)]
pub fn register_engine_with_swap(
    model_idx: usize,
    engine_idx: usize,
    kv_page_size: u32,
    base_page: u32,
    num_kv_pages: usize,
    num_host_pages: usize,
    num_slots: usize,
    max_context: usize,
) -> anyhow::Result<()> {
    let model = REGISTRY
        .get(model_idx)
        .ok_or_else(|| anyhow::anyhow!("store registry: unknown model {model_idx}"))?;
    let mut stores = model.write().unwrap();
    if stores.len() <= engine_idx {
        stores.resize_with(engine_idx + 1, || None);
    }
    anyhow::ensure!(
        stores[engine_idx].is_none(),
        "store registry: engine {engine_idx} is already registered for model {model_idx}"
    );
    if let Some(existing) = stores.iter().flatten().next() {
        anyhow::ensure!(
            existing.kv_page_size == kv_page_size,
            "store registry: KV page size {kv_page_size} does not match model {model_idx} page size {}",
            existing.kv_page_size
        );
    }
    let kv = Arc::new(parking_lot::Mutex::new(KvStore::new_with_swap_range(
        base_page,
        num_kv_pages as u32,
        num_host_pages as u32,
        rand::random::<[u8; 32]>(),
    )));
    stores[engine_idx] = Some(Stores {
        kv,
        rs: Arc::new(Mutex::new(RsStore::new(num_slots as u32))),
        seats: Arc::new(Mutex::new(SeatBook::new(num_slots as u32))),
        seats_freed: Arc::new(tokio::sync::Notify::new()),
        kv_page_size,
        context_pages: context_pages(max_context, kv_page_size),
    });
    Ok(())
}

pub fn unregister_engine(model_idx: usize, engine_idx: usize) -> anyhow::Result<()> {
    let model = REGISTRY
        .get(model_idx)
        .ok_or_else(|| anyhow::anyhow!("store registry: unknown model {model_idx}"))?;
    let mut stores = model.write().unwrap();
    let slot = stores.get_mut(engine_idx).ok_or_else(|| {
        anyhow::anyhow!("store registry: unknown engine {engine_idx} for model {model_idx}")
    })?;
    anyhow::ensure!(
        slot.take().is_some(),
        "store registry: engine {engine_idx} for model {model_idx} is already unregistered"
    );
    Ok(())
}

pub fn get(model_idx: usize, engine_idx: usize) -> Stores {
    try_get(model_idx, engine_idx).unwrap_or_else(|| {
        panic!("store registry: no stores for model {model_idx} engine {engine_idx}")
    })
}

pub fn try_get(model_idx: usize, engine_idx: usize) -> Option<Stores> {
    REGISTRY
        .get(model_idx)?
        .read()
        .unwrap()
        .get(engine_idx)
        .cloned()
        .flatten()
}

pub fn all_for_model(model_idx: usize) -> Vec<Stores> {
    REGISTRY
        .get(model_idx)
        .map(|stores| stores.read().unwrap().iter().flatten().cloned().collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_every_case() {
        dynamic_store_slots_unregister_without_reusing_engine_ids();
        dynamic_store_slots_allow_global_engine_id_gaps();
    }

    fn dynamic_store_slots_unregister_without_reusing_engine_ids() {
        let model = register_model(16, &[8], &[0]);
        register_engine_with_swap(model, 1, 16, 10, 4, 0, 0, 0).unwrap();
        assert!(try_get(model, 1).is_some());
        unregister_engine(model, 1).unwrap();
        assert!(try_get(model, 1).is_none());
        register_engine_with_swap(model, 2, 16, 20, 4, 0, 0, 0).unwrap();
        assert!(try_get(model, 2).is_some());
    }

    fn dynamic_store_slots_allow_global_engine_id_gaps() {
        let model = register_model(16, &[8], &[0]);
        register_engine_with_swap(model, 4, 16, 40, 4, 0, 0, 0).unwrap();
        assert!(try_get(model, 1).is_none());
        assert!(try_get(model, 3).is_none());
        assert!(try_get(model, 4).is_some());
        assert_eq!(all_for_model(model).len(), 2);
    }
}
