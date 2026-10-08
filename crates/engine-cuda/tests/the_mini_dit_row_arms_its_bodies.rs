#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine::Engine;
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use poem::{Platform, Trace};

const SKU: &str = "mini-dit-bf16-kv-bf16";

fn artifact() -> Option<PathBuf> {
    let path = std::env::var("PIE_MINI_DIT_ARTIFACT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/root/.cache/pie-imagegen/mini-dit.zt"));
    path.is_file().then_some(path)
}

fn contract_for(trace: &Trace, path: &Path) -> Result<ModelContract, String> {
    let source = ztensor_compat::index(path).map_err(|why| why.to_string())?;
    poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
        .map_err(|why| why.to_string())
}

#[test]
fn every_armed_body_answers_its_eager_walk() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let Some(path) = artifact() else {
        eprintln!("not asked: no mini-dit artifact (PIE_MINI_DIT_ARTIFACT)");
        return;
    };
    let sku = models::deployment(SKU).expect("the catalog ships the mini-dit row");
    let trace = sku.trace(Platform::Cuda);
    let mut engine = engine_cuda::open(engine_cuda::DeviceBoot::default(), contract_for)
        .expect("the engine opens");
    let loaded = engine
        .load(LoadRequest {
            trace,
            checkpoint: Checkpoint::Path(path),
            budgets: Budgets::default(),
            residency: Residency::default(),
            ordinal: 0,
            frames_in_flight: 2,
        })
        .expect("the row loads: every armed body agrees with its eager walk");
    assert!(loaded.caps.profile.has_velocity);
}
