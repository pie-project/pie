use std::sync::atomic::Ordering;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use super::chunks::{Seats, ShapePool, StorePools};
use super::kv::KvStore;
use super::rs::RsStore;
use super::shared::{Shared, Tiers};
use super::slots::ShapeKind;

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
    pub seats: Arc<Mutex<Seats>>,
    pub kv_page_size: u32,
    pub context_pages: u64,
    /// The engine's shared chunk pool, where it states one: the stores
    /// draw their ids from it, and lanes name their slots by its shapes.
    pub shared: Option<Shared>,
    /// The shared pool's host tier, where the engine states one
    /// (CONTRACT-BC 22): what suspended and demoted state parks in.
    pub host: Option<Shared>,
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
        &vec![0; num_kv_pages.len()],
        &num_slots
            .iter()
            .map(|&slots| StatePool {
                slots: slots as u32,
                ..StatePool::default()
            })
            .collect::<Vec<_>>(),
        &vec![0; num_kv_pages.len()],
    )
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StatePool {
    pub slots: u32,
    pub pages: u32,
}

/// Today's pool counts as the pinned shapes the degenerate mode states.
#[cfg(test)]
pub fn register_model_with_swap(
    kv_page_size: u32,
    num_kv_pages: &[usize],
    num_host_pages: &[usize],
    num_disk_pages: &[u32],
    state: &[StatePool],
    max_context: &[usize],
) -> usize {
    use super::chunks::ShapeRun;
    let engines: Vec<EngineStores> = (0..num_kv_pages.len())
        .map(|d| {
            let pool = state.get(d).copied().unwrap_or_default();
            let run = |device: u32, host: u32| ShapeRun {
                device,
                host,
                ..ShapeRun::default()
            };
            EngineStores {
                pools: StorePools {
                    shared: false,
                    page_size: kv_page_size,
                    kv: ShapeRun {
                        disk: num_disk_pages.get(d).copied().unwrap_or(0),
                        ..run(
                            num_kv_pages[d] as u32,
                            num_host_pages.get(d).copied().unwrap_or(0) as u32,
                        )
                    },
                    window_tokens: 0,
                    state: run(pool.slots, 0),
                    buffer: run(pool.pages, 0),
                },
                max_context: max_context.get(d).copied().unwrap_or(0),
                shared: None,
                tiers: Tiers::default(),
            }
        })
        .collect();
    register_model_with_chunks(kv_page_size, &engines)
}

/// One engine's stores: its pools as chunks, its longest context, and the
/// shared pool the stores draw from where it states one.
#[derive(Clone, Debug, Default)]
pub struct EngineStores {
    pub pools: StorePools,
    pub max_context: usize,
    pub shared: Option<Shared>,
    /// The tiers below the shared pool.
    pub tiers: Tiers,
}

/// Registers a model whose engines state their pools as `engines` say.
pub fn register_model_with_chunks(kv_page_size: u32, engines: &[EngineStores]) -> usize {
    let stores: Vec<Option<Stores>> = engines
        .iter()
        .map(|engine| {
            let pools = &engine.pools;
            if let Some(shared) = &engine.shared {
                return Some(shared_stores(
                    shared,
                    &engine.tiers,
                    kv_page_size,
                    engine.max_context,
                ));
            }
            let mut kv = KvStore::with_pages(
                pools.kv,
                pools.kv.host,
                pools.kv.disk,
                rand::random::<[u8; 32]>(),
            );
            if let Some(identity) = crate::model::identity() {
                kv.set_domain(identity.domain());
            }
            // KV alone resumes a model only when it has no recurrent state.
            kv.set_keeps(kv_sharing() && pools.state.device == 0 && pools.window_tokens == 0);
            kv.set_page_size(kv_page_size);
            let kv = Arc::new(parking_lot::Mutex::new(kv));
            let slots = pools.state.device;
            let max_context = engine.max_context;
            let rs = RsStore::of(&StorePools {
                page_size: kv_page_size,
                ..*pools
            });
            Some(Stores {
                kv,
                rs: Arc::new(Mutex::new(rs)),
                seats: Arc::new(Mutex::new(Seats::new(slots))),
                kv_page_size,
                context_pages: context_pages(max_context, kv_page_size),
                shared: None,
                host: None,
            })
        })
        .collect();
    REGISTRY.push(RwLock::new(stores))
}

/// The stores of an engine whose pool is `shared`: kv pages are the list
/// shape's slot refs, the rs store's state slots its records (or its
/// window lists' pages each list's refs), as many seats as records. Below
/// it `tiers`: the host tier's slots of the same shapes, and kv list slots
/// on disk, every move between them the engine's `copy_slots`
/// (CONTRACT-BC 22).
pub fn shared_stores(
    shared: &Shared,
    tiers: &Tiers,
    kv_page_size: u32,
    max_context: usize,
) -> Stores {
    let (list, records) = {
        let pool = shared.lock().unwrap();
        (pool.list(), pool.records_most())
    };
    let pages = list.map_or_else(
        || ShapePool::new(ShapeKind::List, 0, 0, 0),
        |list| ShapePool::shared(shared, list),
    );
    let mut kv = KvStore::with_pool(pages, 0, 0, rand::random::<[u8; 32]>());
    if let Some(list) = list {
        let host = tiers
            .host
            .as_ref()
            .map(|host| ShapePool::shared(host, list));
        kv.set_tiers(list.id, host, tiers.disk_slots);
    }
    if let Some(identity) = crate::model::identity() {
        kv.set_domain(identity.domain());
    }
    let rs = RsStore::shared(shared, tiers.host.as_ref());
    kv.set_keeps(kv_sharing() && records == 0 && !rs.is_window());
    kv.set_page_size(kv_page_size);
    Stores {
        kv: Arc::new(parking_lot::Mutex::new(kv)),
        rs: Arc::new(Mutex::new(rs)),
        seats: Arc::new(Mutex::new(Seats::new(records))),
        kv_page_size,
        context_pages: context_pages(max_context, kv_page_size),
        shared: Some(Arc::clone(shared)),
        host: tiers.host.clone(),
    }
}

/// `PIE_KV_SHARE`, on unless `0`.
pub fn kv_sharing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PIE_KV_SHARE").is_none_or(|value| value != "0"))
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

    fn register_engine_with_swap(
        model_idx: usize,
        engine_idx: usize,
        kv_page_size: u32,
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
        let kv = Arc::new(parking_lot::Mutex::new(KvStore::new_with_swap(
            num_kv_pages as u32,
            num_host_pages as u32,
            0,
            rand::random::<[u8; 32]>(),
        )));
        stores[engine_idx] = Some(Stores {
            kv,
            rs: Arc::new(Mutex::new(RsStore::new(num_slots as u32, num_slots as u32))),
            seats: Arc::new(Mutex::new(Seats::new(num_slots as u32))),
            kv_page_size,
            context_pages: context_pages(max_context, kv_page_size),
            shared: None,
            host: None,
        });
        Ok(())
    }

    #[test]
    fn registry_every_case() {
        dynamic_store_slots_allow_global_engine_id_gaps();
    }

    fn dynamic_store_slots_allow_global_engine_id_gaps() {
        let model = register_model(16, &[8], &[0]);
        register_engine_with_swap(model, 4, 16, 4, 0, 0, 0).unwrap();
        assert!(try_get(model, 1).is_none());
        assert!(try_get(model, 3).is_none());
        assert!(try_get(model, 4).is_some());
        assert_eq!(all_for_model(model).len(), 2);
    }
}
