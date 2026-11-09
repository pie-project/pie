#![cfg(target_vendor = "apple")]

//! A loaded shell states its pools as degenerate chunks: every pool count
//! reads back, the facts hang together, a chunk is the largest page or slot,
//! and a pool on a sparse heap grows a chunk at a time from what it maps.

use std::path::PathBuf;

use engine::chunk::{DEGENERATE_BUFFER, DEGENERATE_KV, DEGENERATE_STATE, DegenerateCounts, Growth};
use engine_metal::{Boot, Shell};
use poem::{Dtype, Platform};
use poem_compiler::Budget;

const PAGE: u32 = 16;
const CONTEXT: u32 = 128;
const SLOTS: u32 = 8;

fn fixture() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_QWEN4_FIXTURE") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-micro");
    path.is_dir().then_some(path)
}

/// The micro model of the qwen_4 package: a GDN hybrid, so kv pages, state
/// slots and rs buffer pages all stand.
const MICRO: &str = "qwen38-flash-next-micro";

fn package() -> &'static poem::star::Package {
    poem_compiler::catalog::package_of(MICRO).expect("the qwen_4 package holds its micro model")
}

fn deploy() -> poem::star::Deploy {
    poem::star::Deploy {
        weights: vec![Dtype::Bf16.into()],
        kv: Dtype::Bf16,
        tp: 1,
        ep: 1,
        parts: vec![],
        drafter: None,
        yarn: None,
        tap: None,
        layers: None,
        experts: None,
    }
}

fn loaded() -> Option<Shell> {
    if !engine_metal::device::present() {
        eprintln!("skipping the chunk facts gate: no Metal device");
        return None;
    }
    let Some(fixture) = fixture() else {
        eprintln!("skipping the chunk facts gate: no qwen4 fixture");
        return None;
    };
    let trace = package()
        .trace(MICRO, &deploy(), "qwen4-micro", Platform::Metal)
        .unwrap_or_else(|why| panic!("the micro model traces: {why:#}"));
    let source =
        ztensor_compat::index(fixture.join("model.safetensors")).expect("the fixture opens");
    let contract = package()
        .import(MICRO, &deploy(), &source, Platform::Metal)
        .expect("the micro text's import fits its fixture");
    drop(source);
    Some(
        Shell::load(Boot {
            kv_budget: crate::full_pool(&trace, PAGE, CONTEXT, SLOTS),
            voxels: None,
            trace,
            contract: &contract,
            checkpoint: &fixture,
            budget: Budget::new(4, 64),
            patches: None,
            profile: None,
            page_size: PAGE,
            context: CONTEXT,
            slots: SLOTS,
            runahead: engine::runahead::Runahead::F1,
            memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
            profiles: None,
            residency: engine_metal::ResidencyPlan::default(),
        })
        .expect("the micro shell loads"),
    )
}

#[test]
fn a_hybrid_pool_reads_back_as_degenerate_chunks() {
    let Some(mut shell) = loaded() else { return };
    let pools = shell.pool_facts(0);
    let facts = shell
        .chunk_facts(&pools)
        .expect("the pools state as chunks");
    eprintln!(
        "{MICRO}: {} chunks of {} bytes, segments {:?} of {}, shapes {:?}",
        facts.total_chunks,
        facts.chunk_bytes,
        facts.segments,
        facts.segment_chunks,
        facts
            .shapes
            .iter()
            .map(|s| (
                s.id.0,
                s.kind,
                s.slot_bytes,
                s.pinned.map(|p| p.device.chunks)
            ))
            .collect::<Vec<_>>()
    );
    facts.validate().expect("the facts hang together");
    assert_eq!(facts.degenerate_counts(), DegenerateCounts::of(&pools));
    assert_eq!(
        facts.total_chunks,
        1 + pools.kv_pages + pools.state_slots + pools.buffer_pages
    );
    for id in [DEGENERATE_KV, DEGENERATE_STATE, DEGENERATE_BUFFER] {
        let shape = facts.shape(id).expect("a GDN hybrid holds every pool");
        assert!(shape.slot_bytes > 0 && shape.slot_bytes <= facts.chunk_bytes);
    }
    assert_eq!(
        facts.shape(DEGENERATE_STATE).unwrap().slot_bytes,
        shell.state_slot_bytes()
    );
    if pools.elastic_page_bytes == 0 {
        assert_eq!(facts.segments.growth, Growth::Fixed);
        return;
    }
    assert_eq!(facts.segments.growth, Growth::Bottom);
    assert_eq!(facts.segment_chunks, 1);
    assert_eq!(facts.segments.max, facts.total_chunks);
    let before = facts.segments.committed;

    // Committing pages and slots maps chunks; the facts count them.
    shell.commit_pool(4).expect("four kv pages commit");
    shell.commit_state(2).expect("two state slots commit");
    let grown = shell
        .chunk_facts(&pools)
        .expect("the pools state as chunks");
    grown.validate().expect("the grown facts hang together");
    assert!(
        grown.segments.committed >= before + 4 + 2,
        "{before} -> {}",
        grown.segments.committed
    );
}
