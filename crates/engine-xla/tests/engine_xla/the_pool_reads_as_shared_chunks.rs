//! A dry load states its pool as one fixed set of shared chunks: the facts
//! hang together, every shape draws on the whole pool (none pinned), the
//! null and sink chunks are reserved, the pool is one segment fixed at load,
//! and the shapes are the model's: a kv list, a window list, a fixed record
//! shape a state block size, an rs list.

use std::path::Path;

use engine::chunk::{ChunkFacts, Growth, Reserved, SlotKind, SlotRole};
use engine_xla::{Boot, DeviceBoot, Shell};
use poem::Platform;
use poem_compiler::Budget;

const PAGE: u32 = 16;
const CONTEXT: u32 = 2048;
const SLOTS: u32 = 4;

fn shell(name: &str) -> Shell {
    let deployment = poem_compiler::catalog::deployment(name).expect("the deployment is listed");
    let trace = deployment.trace(Platform::Xla);
    let paging =
        engine_xla::store::kv::Paging::of(PAGE, CONTEXT, SLOTS, u64::from(SLOTS * CONTEXT / PAGE))
            .expect("the paging lays out");
    let kv_budget = engine_xla::store::pool_demand(&trace, paging).expect("the pools size");
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
    let facts = shell.chunk_facts();
    let shapes: Vec<_> = facts
        .shapes
        .iter()
        .map(|s| (s.id.0, s.kind, s.slot_bytes, s.slots_per_chunk))
        .collect();
    eprintln!(
        "{name}: {} chunks of {} bytes, shapes {shapes:?}",
        facts.total_chunks, facts.chunk_bytes
    );
    facts.validate().expect("the facts hang together");
    assert_eq!(facts.segments.growth, Growth::Fixed);
    assert_eq!((facts.segments.committed, facts.segments.max), (1, 1));
    assert_eq!(facts.segment_chunks, facts.total_chunks);
    assert!(facts.act_chunks.is_empty(), "activation lives apart");
    let reserved: Vec<_> = facts
        .reserved
        .iter()
        .map(|r| (r.what, r.first, r.chunks))
        .collect();
    assert_eq!(
        reserved,
        vec![
            (Reserved::Null, 0, 1),
            (Reserved::Sink, facts.total_chunks - 1, 1)
        ],
        "{name}"
    );
    for shape in &facts.shapes {
        assert!(
            shape.pinned.is_none(),
            "{name}: shape {} shares the pool",
            shape.id
        );
        assert!(shape.slot_bytes > 0 && shape.slot_bytes <= facts.chunk_bytes);
        assert_eq!(
            shape.slots_per_chunk as u64,
            facts.chunk_bytes / shape.slot_bytes,
            "{name}"
        );
        assert_eq!(shape.sets_per_working_set, 1);
    }
    let pool = shell.pools().bytes();
    assert_eq!(pool, u64::from(facts.total_chunks) * facts.chunk_bytes);
    facts
}

fn kinds(facts: &ChunkFacts) -> Vec<(SlotKind, SlotRole)> {
    facts.shapes.iter().map(|s| (s.kind, s.role)).collect()
}

#[test]
fn a_dense_pool_is_one_kv_list() {
    let facts = reads_back("qwen3-30b-a3b-mini-bf16-kv-bf16");
    assert_eq!(
        kinds(&facts),
        vec![(
            SlotKind::List {
                tokens: PAGE,
                rate: 1,
                window: None
            },
            SlotRole::State
        )]
    );
}

#[test]
fn a_windowed_pool_states_its_window_list() {
    let facts = reads_back("gemma4-e4b-mini-l6-bf16-kv-bf16");
    let windows: Vec<_> = facts
        .shapes
        .iter()
        .filter_map(|s| s.kind.window())
        .collect();
    assert_eq!(windows.len(), 1, "one window list, the widest window's");
    assert!(windows[0] < CONTEXT);
}

#[test]
fn recurrent_state_is_a_record_of_fixed_shapes() {
    for name in [
        "nemotron3-nano-mini-bf16-kv-bf16",
        "qwen38-flash-next-mini-bf16-kv-bf16",
    ] {
        let facts = reads_back(name);
        let fixed = facts
            .shapes
            .iter()
            .filter(|s| matches!(s.kind, SlotKind::Fixed { .. }))
            .count();
        assert!(fixed >= 1, "{name}: a record");
        let rs = facts
            .shapes
            .iter()
            .filter(|s| s.role == SlotRole::RsBuffer)
            .count();
        assert!(rs <= 1, "{name}: at most one rs list");
    }
}
