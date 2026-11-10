//! A dry load of the host-state shell states its pool as one shared chunk
//! pool: the facts hang together, no shape is pinned, chunk 0 is the null
//! chunk and no sink is reserved (padding writes the fire's own planes),
//! each pool the budget sized fits beside a copy-on-write record a lane,
//! and a fire whose lanes name slot refs stages.

use std::path::Path;

use engine::chunk::{ChunkFacts, Growth, Reserved, SlotKind, SlotRole};
use engine_cerebras::serve::{Lane, Seated};
use engine_cerebras::store::pool::{FIXED, LIST, RS, WINDOW};
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

fn reads_back(shell: &Shell, name: &str) -> ChunkFacts {
    let pools = shell.pool_facts();
    let facts = shell.chunk_facts().expect("the pool states as chunks");
    let shapes: Vec<_> = facts
        .shapes
        .iter()
        .map(|s| (s.id.0, s.kind, s.slot_bytes, s.slots_per_chunk))
        .collect();
    eprintln!(
        "{name}: {} chunks of {} bytes in segments of {}, shapes {shapes:?}",
        facts.total_chunks, facts.chunk_bytes, facts.segment_chunks
    );
    facts.validate().expect("the facts hang together");
    assert!(facts.shapes.iter().all(|s| s.pinned.is_none()), "{name}");
    assert_eq!(facts.reserved.len(), 1, "{name}: the null chunk alone");
    assert_eq!(facts.reserved[0].what, Reserved::Null);
    assert_eq!(facts.segments.growth, Growth::Bottom);
    assert_eq!(facts.segments.committed, facts.segments.max);
    assert!(facts.shapes.iter().all(|s| s.slot_bytes > 0), "{name}");
    // Each pool the budget sized fits in the shared pool.
    let slots = |id| {
        facts.shape(id).map_or(0, |s| {
            u64::from(s.slots_per_chunk) * u64::from(facts.total_chunks - 1)
        })
    };
    if facts.shape(LIST).is_some() {
        assert!(slots(LIST) >= u64::from(pools.kv_pages), "{name}");
    }
    if let Some(state) = facts.shape(FIXED) {
        let SlotKind::Fixed { pieces } = state.kind else {
            panic!("{name}: the record is a fixed shape");
        };
        assert!(
            slots(FIXED) >= u64::from(pools.state_slots * pieces),
            "{name}"
        );
    }
    facts
}

/// One slot ref of shape `id` a lane names, `n` of them from `chunk` on.
fn refs(facts: &ChunkFacts, id: engine::chunk::ShapeId, chunk: u32, n: u32) -> Vec<u32> {
    let per = facts.shape(id).expect("the shape").slots_per_chunk;
    (0..n).map(|i| chunk * per + i).collect()
}

/// Two lanes sharing a prefix, each with a record, fire dry: the staging
/// takes their slot refs.
fn fires(shell: &mut Shell, facts: &ChunkFacts) {
    let list = refs(facts, LIST, 1, 4);
    let tokens = [vec![1u32; 7], vec![2u32]];
    let tables = [vec![list[0], list[1]], vec![list[0], list[2], list[3]]];
    let held = [20u32, 40];
    let windows: Vec<Vec<u32>> = if facts.shape(WINDOW).is_some() {
        let w = refs(facts, WINDOW, 1, 4);
        vec![vec![w[0], w[1]], vec![w[0], w[2], w[3]]]
    } else {
        vec![Vec::new(), Vec::new()]
    };
    let records: Vec<Vec<u32>> = match facts.shape(FIXED).map(|s| s.kind) {
        Some(SlotKind::Fixed { pieces }) => {
            let all = refs(facts, FIXED, 1, 2 * pieces);
            all.chunks(pieces as usize).map(<[u32]>::to_vec).collect()
        }
        _ => vec![Vec::new(), Vec::new()],
    };
    let seated: Vec<Seated<'_>> = (0..2)
        .map(|at| {
            let mut seated = Seated::of(Lane {
                slot: at as u32,
                word: 0,
                tokens: &tokens[at],
            });
            seated.pages = &tables[at];
            seated.window = &windows[at];
            seated.held = Some(held[at]);
            seated.record = &records[at];
            seated
        })
        .collect();
    // The staging comes first; a dry walk of these minis may then stop at
    // a kernel the host walk does not lower, which is no staging fault.
    match shell.fire_full(&seated) {
        Ok(_) => {}
        Err(fault) => {
            let fault = format!("{fault:?}");
            assert!(
                fault.starts_with("Fire(Kernel("),
                "the fire stages: {fault}"
            );
        }
    }
}

#[test]
fn a_dense_pool_is_one_kv_list() {
    let name = "qwen3-30b-a3b-mini-bf16-kv-bf16";
    let mut shell = shell(name);
    let facts = reads_back(&shell, name);
    assert!(facts.shape(LIST).is_some());
    assert!(facts.shape(FIXED).is_none() && facts.shape(RS).is_none());
    fires(&mut shell, &facts);
}

#[test]
fn a_windowed_pool_states_its_window_list() {
    let name = "gemma4-e4b-mini-l6-bf16-kv-bf16";
    let mut shell = shell(name);
    let facts = reads_back(&shell, name);
    let window = facts.shape(WINDOW).expect("a window list");
    assert_eq!(
        window.kind,
        SlotKind::List {
            tokens: PAGE,
            rate: 1,
            window: shell.paging().window.map(|w| w.tokens),
        }
    );
    fires(&mut shell, &facts);
}

#[test]
fn recurrent_state_is_a_record_and_an_rs_list() {
    for name in [
        "nemotron3-nano-mini-bf16-kv-bf16",
        "qwen38-flash-next-mini-bf16-kv-bf16",
    ] {
        let mut shell = shell(name);
        let facts = reads_back(&shell, name);
        let state = facts.shape(FIXED).expect("a record");
        assert_eq!(state.sets_per_working_set, 1);
        if let Some(rs) = facts.shape(RS) {
            assert_eq!(rs.role, SlotRole::RsBuffer);
            assert_eq!(
                rs.kind,
                SlotKind::List {
                    tokens: PAGE,
                    rate: 1,
                    window: None
                }
            );
        }
        fires(&mut shell, &facts);
    }
}
