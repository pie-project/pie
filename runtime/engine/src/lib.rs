//! Pie - Programmable Inference Engine

pub mod bootstrap;
pub mod driver;
pub mod inferlet;
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
pub mod scheduler;
pub mod server;
pub(crate) mod service;
pub mod store;
pub(crate) mod telemetry;
