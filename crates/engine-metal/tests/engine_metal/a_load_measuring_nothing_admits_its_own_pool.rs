#![cfg(target_vendor = "apple")]

//! A load sizes its kv pool from one reading of the device's working set
//! and admits the pool against that same reading: what the load itself
//! allocates on the way (the pool layout's host tables, sized by the pool)
//! is not a shortfall. A load that times nothing (`measure = false`, or
//! every curve stored or committed: a clean home's boot under a committed
//! profile) is admitted every time.

use engine_metal::{Boot, Shell};
use poem_compiler::Budget;

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of};

const LOADS: u32 = 200;

fn load(fixture: &Fixture) -> Result<Shell, String> {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(4, 256),
        patches: None,
        voxels: None,
        profile: None,
        page_size: 16,
        context: 256,
        slots: 4,
        kv_budget: None,
        allowance: engine::Allowance::WHOLE,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
        profiles: None,
        top: engine_metal::Top::EveryRung,
        measure: engine::rates::Measuring::Skip,
    })
    .map_err(|why| why.to_string())
}

#[test]
fn a_load_measuring_nothing_admits_its_own_pool() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let refused: Vec<(u32, String)> = (0..LOADS)
        .filter_map(|at| load(&fixture).err().map(|why| (at, why)))
        .collect();
    assert!(
        refused.is_empty(),
        "{} of {LOADS} loads measuring nothing were refused; the first, load {}: {}",
        refused.len(),
        refused.first().map_or(0, |(at, _)| *at),
        refused.first().map_or("", |(_, why)| why.as_str())
    );
}
