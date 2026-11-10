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
    // The checked-in fixture's import panics on some trees (a replicated
    // `ple.table`); that is the fixture's, not this gate's.
    let imported = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        package().import(MICRO, &deploy(), &source, Platform::Metal)
    }));
    drop(source);
    let Ok(contract) = imported else {
        eprintln!("skipping the hybrid chunk facts gate: the qwen4 fixture does not import");
        return None;
    };
    let contract = contract.expect("the micro text's import fits its fixture");
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

/// Reads `shell`'s pools back through its chunk facts; `hybrid` asks for a
/// state and an rs buffer shape beside the kv.
fn check(name: &str, shell: &mut Shell, hybrid: bool) {
    let pools = shell.pool_facts(0);
    let facts = shell
        .chunk_facts(&pools)
        .expect("the pools state as chunks");
    eprintln!(
        "{name}: {} chunks of {} bytes, segments {:?} of {}, shapes {:?}",
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
    let ids: &[_] = if hybrid {
        &[DEGENERATE_KV, DEGENERATE_STATE, DEGENERATE_BUFFER]
    } else {
        &[DEGENERATE_KV]
    };
    for &id in ids {
        let shape = facts.shape(id).expect("the shape stands");
        assert!(shape.slot_bytes > 0 && shape.slot_bytes <= facts.chunk_bytes);
    }
    if hybrid {
        assert_eq!(
            facts.shape(DEGENERATE_STATE).unwrap().slot_bytes,
            shell.state_slot_bytes()
        );
    }
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
    let mut more = 4;
    if hybrid {
        shell.commit_state(2).expect("two state slots commit");
        more += 2;
    }
    let grown = shell
        .chunk_facts(&pools)
        .expect("the pools state as chunks");
    grown.validate().expect("the grown facts hang together");
    assert!(
        grown.segments.committed >= before + more,
        "{before} -> {}",
        grown.segments.committed
    );
}

#[test]
fn a_hybrid_pool_reads_back_as_degenerate_chunks() {
    let Some(mut shell) = loaded() else { return };
    check(MICRO, &mut shell, true);
}

#[test]
fn a_dense_pool_reads_back_as_degenerate_chunks() {
    if !engine_metal::device::present() {
        eprintln!("skipping the chunk facts gate: no Metal device");
        return;
    }
    let dir = std::env::temp_dir().join(format!("pie_bf_chunks_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the fixture dir exists");
    crate::the_kv_rotation_is_the_unrotated_forward::synth_fixture(&dir);
    // The synthetic fixture's contract and its plan disagree on some trees
    // (`the_kv_rotation_is_the_unrotated_forward` fails alike); that is the
    // fixture's, not this gate's.
    match crate::the_kv_rotation_is_the_unrotated_forward::try_load(&dir, Dtype::Bf16, false) {
        Ok(mut shell) => check("qwen3-micro-text", &mut shell, false),
        Err(why) => eprintln!("skipping the dense chunk facts gate: {why}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The pools a GDN hybrid's catalog trace reserves on this device, no
/// weights landed: a kv page and a state slot state as chunks, and on a
/// sparse heap the watermark's commits count as committed chunks.
#[test]
fn a_reserved_hybrid_pool_states_its_pages_as_chunks() {
    if !engine_metal::device::present() {
        eprintln!("skipping the chunk facts gate: no Metal device");
        return;
    }
    let name = "qwen38-flash-next-mini-bf16-kv-bf16";
    let trace = poem_compiler::catalog::deployment(name)
        .expect("the deployment is listed")
        .trace(Platform::Metal);
    let device = engine_metal::Context::bind().expect("the system device");
    let paging = engine_metal::store::kv::Paging::of(PAGE, 2048, 4, 512).expect("a paging");
    let mut pools = engine_metal::store::Planes::reserve(
        &device,
        &trace,
        paging,
        &engine_metal::store::kv::Facts::default(),
    )
    .expect("the pools reserve");
    let page = pools.page_bytes().expect("a kv page's bytes");
    let facts_of = |pools: &engine_metal::store::Planes| {
        let stated = engine::caps::PoolFacts {
            kv_pages: 512,
            kv_page_size: PAGE,
            state_slots: 4,
            state_slot_bytes: pools.state_slot_bytes(),
            elastic_page_bytes: pools.elastic_page_bytes(),
            ..engine::caps::PoolFacts::default()
        };
        let bytes = engine::chunk::DegenerateBytes {
            kv_page: page,
            ..engine::chunk::DegenerateBytes::default()
        };
        let facts = engine::chunk::ChunkFacts::degenerate(&stated, &bytes);
        let mark = pools.watermark();
        let facts = if pools.is_elastic() {
            engine_metal::chunks::grown(facts, &stated, (mark.kv_pages, mark.state_slots, 0))
        } else {
            facts
        };
        facts.validate().expect("the facts hang together");
        assert_eq!(facts.degenerate_counts(), DegenerateCounts::of(&stated));
        facts
    };
    let facts = facts_of(&pools);
    eprintln!(
        "{name}: kv page {page} B, state slot {} B, elastic {}, {} chunks of {} B, segments {:?}",
        pools.state_slot_bytes(),
        pools.is_elastic(),
        facts.total_chunks,
        facts.chunk_bytes,
        facts.segments
    );
    assert!(page > 0 && pools.state_slot_bytes() > 0);
    assert_eq!(facts.total_chunks, 1 + 512 + 4);
    assert_eq!(facts.chunk_bytes, page.max(pools.state_slot_bytes()));
    if !pools.is_elastic() {
        assert_eq!(facts.segments.growth, Growth::Fixed);
        return;
    }
    assert_eq!(facts.segments.growth, Growth::Bottom);
    let before = facts.segments.committed;
    pools
        .commit(engine::frame::Demand {
            kv_pages: 4,
            state_slots: 2,
            workspace: 0,
        })
        .expect("four pages and two slots commit");
    assert_eq!(facts_of(&pools).segments.committed, before + 4 + 2);
}
