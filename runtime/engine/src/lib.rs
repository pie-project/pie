//! Pie - Programmable Inference Engine

pub mod bootstrap;
pub mod driver;
pub mod inferlet;
/// The served model: the global cache the runtime binds once at bootstrap.
///
/// Moved here from `pie-model`, which defines itself as backend-blind family
/// knowledge — and a process-global `OnceLock` holding whatever this engine
/// booted is neither a family fact nor knowledge. `pie_model::ModelMetadata`
/// stayed behind: what an artifact carries is a fact about models.
pub mod model;
pub(crate) mod pipeline;
pub mod offload {
    pub use crate::pipeline::offload::{
        OffloadCounterSnapshot, Partner, PartnerGuard, PartnerRole, close_driver_surrogates,
        configure, configure_encode_injection, counters, register_partner, remove_partner,
        select_partner, set_home_kv_handle,
    };

    pub fn register_remote_store(
        model_idx: usize,
        driver_idx: usize,
        kv_page_size: u32,
        base_page: u32,
        num_kv_pages: usize,
    ) -> anyhow::Result<()> {
        crate::store::registry::register_driver_with_swap(
            model_idx,
            driver_idx,
            kv_page_size,
            base_page,
            num_kv_pages,
            0,
            0,
        )
    }

    pub fn unregister_remote_store(model_idx: usize, driver_idx: usize) -> anyhow::Result<()> {
        crate::store::registry::unregister_driver(model_idx, driver_idx)
    }
}
pub mod planner;
pub mod scheduler;
pub mod server;
pub(crate) mod service;
pub mod store;
pub(crate) mod telemetry;
