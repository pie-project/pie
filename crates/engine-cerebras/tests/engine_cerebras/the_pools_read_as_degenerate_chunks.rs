//! A dry load of the host-state shell states its pools as degenerate chunks: every pool count reads
//! back, the facts hang together, the sink page and slot padding lanes
//! write are reserved past the pools, and the bytes of a chunk are the
//! largest page or slot.

use std::path::Path;

use engine::chunk::{
    ChunkFacts, DEGENERATE_BUFFER, DEGENERATE_KV, DEGENERATE_STATE, DEGENERATE_WINDOW,
    DegenerateCounts, Growth, Reserved,
};
use engine_cerebras::{Boot, DeviceBoot, Shell};
use poem::Platform;
use poem_compiler::Budget;

const PAGE: u32 = 16;
const CONTEXT: u32 = 2048;
const SLOTS: u32 = 4;

fn shell(name: &str) -> Shell {
    let deployment = poem_compiler::catalog::deployment(name).expect("the deployment is listed");
    let trace = deployment.trace(Platform::Cerebras);
    let paging = engine_cerebras::store::kv::Paging::of(
        PAGE,
        CONTEXT,
        SLOTS,
        u64::from(SLOTS * CONTEXT / PAGE),
    )
    .expect("the paging lays out");
    let kv_budget = engine_cerebras::store::pool_demand(&trace, paging).expect("the pools size");
    let contract = checkpoint::contract::ModelContract {
        alignment: 1,
        tensors: Vec::new(),
        groups: Vec::new(),
    };
    Shell::load_dry(
        Boot {
            trace,
            contract: &contract,
            checkpoint: Path::new("/dev/null"),
            budget: Budget {
                max_lanes: 64,
                max_tokens: 2048,
                buckets: Vec::new(),
                max_adapters: 0,
                readouts_per_lane: poem_compiler::arena::READOUTS_PER_LANE,
            },
            patches: None,
            page_size: PAGE,
            context: CONTEXT,
            slots: SLOTS,
            kv_budget: Some(kv_budget),
            memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
            device: &DeviceBoot::default(),
        },
        None,
        false,
    )
    .expect("the dry shell loads")
}

fn reads_back(name: &str) -> ChunkFacts {
    let shell = shell(name);
    let pools = shell.pool_facts();
    let facts = shell
        .chunk_facts(&pools)
        .expect("the pools state as chunks");
    let shapes: Vec<_> = facts
        .shapes
        .iter()
        .map(|s| {
            (
                s.id.0,
                s.kind,
                s.slot_bytes,
                s.pinned.map(|p| p.device.chunks),
            )
        })
        .collect();
    eprintln!(
        "{name}: {} chunks of {} bytes, reserved {:?}, shapes {shapes:?}",
        facts.total_chunks, facts.chunk_bytes, facts.reserved
    );
    facts.validate().expect("the facts hang together");
    let profile = poem_compiler::DeviceProfile {
        chunk: poem_compiler::chunk::ChunkProfile::SERVER_CUDA,
        ..poem_compiler::DeviceProfile::default()
    };
    let budgets = poem_compiler::Budgets::of(shell.budget().clone());
    match poem_compiler::chunk::plan(shell.trace(), &budgets, &profile) {
        Ok(recipe) => eprintln!(
            "{name}: under the server preset the recipe would be {} chunks: {}",
            poem_compiler::chunk::human(recipe.chunk_bytes),
            recipe.shapes()
        ),
        Err(error) => eprintln!("{name}: no recipe under the server preset: {error}"),
    }
    assert_eq!(
        facts.degenerate_counts(),
        DegenerateCounts::of(&pools),
        "{name}"
    );
    assert_eq!(facts.segments.growth, Growth::Fixed);
    assert_eq!(facts.segments.committed, facts.segments.max);
    let sinks = u32::from(pools.kv_pages > 0 && facts.shape(DEGENERATE_KV).is_some())
        + u32::from(facts.shape(DEGENERATE_WINDOW).is_some())
        + u32::from(pools.state_slots > 0);
    let sink = facts
        .reserved
        .iter()
        .find(|run| run.what == Reserved::Sink)
        .map_or(0, |run| run.chunks);
    assert_eq!(sink, sinks, "{name}: a sink for each pool padding writes");
    assert_eq!(
        facts.total_chunks,
        1 + pools.kv_pages
            + pools.window_pages.saturating_sub(1)
            + pools.state_slots
            + pools.buffer_pages
            + sinks,
    );
    assert_eq!(
        facts.chunk_bytes,
        facts.shapes.iter().map(|s| s.slot_bytes).max().unwrap_or(0)
    );
    assert!(facts.shapes.iter().all(|s| s.slot_bytes > 0), "{name}");
    facts
}

#[test]
fn a_dense_pool_is_one_list_shape_and_its_sink() {
    let facts = reads_back("qwen3-30b-a3b-mini-bf16-kv-bf16");
    assert!(facts.shape(DEGENERATE_KV).is_some());
    assert!(facts.shape(DEGENERATE_STATE).is_none());
}

#[test]
fn a_windowed_pool_is_a_ring_shape_past_the_null_chunk() {
    let facts = reads_back("gemma4-e4b-mini-l6-bf16-kv-bf16");
    if let Some(ring) = facts.shape(DEGENERATE_WINDOW) {
        assert_eq!(
            ring.pinned.expect("degenerate shapes are pinned").ids_from,
            1
        );
    }
}

#[test]
fn recurrent_state_is_a_fixed_shape_held_twice() {
    for name in [
        "nemotron3-nano-mini-bf16-kv-bf16",
        "qwen38-flash-next-mini-bf16-kv-bf16",
    ] {
        let facts = reads_back(name);
        let state = facts.shape(DEGENERATE_STATE).expect("a state shape");
        assert_eq!(state.sets_per_working_set, 2);
        if let Some(buffer) = facts.shape(DEGENERATE_BUFFER) {
            assert_eq!(buffer.sets_per_working_set, 2);
        }
    }
}
