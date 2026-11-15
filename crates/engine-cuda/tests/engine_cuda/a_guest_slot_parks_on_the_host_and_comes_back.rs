#![cfg(feature = "device")]

//! A guest's `Instance` and `Channel` slots move between the device and the
//! host tier as every state shape's do (CONTRACT-BC 22): what suspends a
//! process that has bound a guest program.

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::chunk::SlotRole;
use engine_cuda::store::Tier;
use engine_cuda::{Boot, Graphs, Recording, Shell};
use poem::{Dtype, ForwardHybrid, Input, Platform, StateRows, Weight, ops};
use poem_compiler::Budget;

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const KV_ROW: &str = "kv";
const PAGE: u32 = 16;
const LANES: u32 = 4;
/// The slots of an item: a run, as a guest item past one slot takes.
const RUN: u32 = 3;

/// One attention layer.
struct Micro;

impl ForwardHybrid for Micro {
    fn states(&self) -> StateRows {
        let mut spec = StateRows::new();
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        spec.kv(KV_ROW, Dtype::Bf16, [plane, plane], HEAD_DIM);
        spec
    }

    fn forward(&self, inputs: Input) {
        let w = |name: &str, out: u64, inner: u64| Weight::sym(name, [out, inner], Dtype::Bf16);
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        let table = w("embed", u64::from(VOCAB), WIDTH);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let q = ops::linear::matmul(&x, &w("q", plane, WIDTH));
        let k = ops::linear::matmul(&x, &w("k", plane, WIDTH));
        let v = ops::linear::matmul(&x, &w("v", plane, WIDTH));
        ops::attn::kv_append(
            &k,
            Some(&v),
            inputs.kv(KV_ROW),
            &inputs.write_page(KV_ROW),
            &inputs.write_offset(KV_ROW),
        );
        let a = ops::attn::attend(
            &q, &inputs, KV_ROW, HEADS, HEADS, HEAD_DIM, 0.125, None, None, None, false, None,
            None, false, false,
        );
        let h = ops::linear::matmul(&a, &w("o", WIDTH, plane));
        let logits = ops::linear::lm_head(&h, &table);
        poem::output("logits", &logits, None);
    }
}

fn shell(trace: poem::Trace, path: &Path) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    Shell::load(Boot {
        voxels: None,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(LANES, 64),
        patches: None,
        page_size: PAGE,
        context: 256,
        slots: LANES,
        kv_budget: Some(crate::POOL),
        allowance: engine::Allowance::WHOLE,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            chip: Some(engine_cuda::chip::pinned(
                Graphs::Off,
                Some(Recording::Shaped),
            )),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads")
}

/// `RUN` adjacent slot refs from slot `index` of chunk `chunk`.
fn run(per_chunk: u32, chunk: u32, index: u32) -> Vec<u32> {
    (0..RUN).map(|at| chunk * per_chunk + index + at).collect()
}

fn pairs(src: &[u32], dst: &[u32]) -> Vec<(u32, u32)> {
    src.iter().copied().zip(dst.iter().copied()).collect()
}

/// For each guest shape: known bytes written through the host tier into a
/// run of device slots lie where a guest's item of that run is addressed;
/// they park on the host (a suspend), the device slots are overwritten, and
/// they come back into fresh slots of another chunk (a restore), byte for
/// byte.
#[test]
fn a_guest_slot_parks_on_the_host_and_comes_back() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("guest-tier-micro", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 7717).write(dir.path());
    let mut shell = shell(trace, &path);
    shell
        .seat_host_kv(Some(crate::POOL), 1, 256, 1)
        .expect("the host tier seats");
    let facts = shell.chunk_facts().expect("the pool states its chunks");
    assert!(facts.host_chunks >= 6, "three host chunks a guest shape");
    shell.take_refs();
    let segments = facts.segments.max.min(4);
    shell
        .commit_segments(segments)
        .expect("the segments commit");
    assert!(
        segments * facts.segment_chunks > 4,
        "two committed chunks a guest shape past the null chunk"
    );
    let stream = shell.stream();
    for (at, role) in [SlotRole::Instance, SlotRole::Channel]
        .into_iter()
        .enumerate()
    {
        let at = at as u32;
        let shape = facts
            .shapes
            .iter()
            .find(|shape| shape.role == role)
            .expect("the load states the guest shapes");
        let (id, per_chunk) = (shape.id, shape.slots_per_chunk);
        let bytes = u64::from(RUN) * shape.slot_bytes;
        // A shape's chunks are its own on either tier, as the runtime's
        // pools hand them out.
        let device = run(per_chunk, 1 + 2 * at, 5);
        let fresh = run(per_chunk, 2 + 2 * at, 11);
        let seed = run(per_chunk, 3 * at, 7);
        let parked = run(per_chunk, 3 * at + 1, 2);
        let pattern: Vec<u8> = (0..bytes)
            .map(|byte| (byte * 31 + 17 + u64::from(at) * 101) as u8 ^ (byte >> 8) as u8)
            .collect();
        let other = vec![0xa5u8; pattern.len()];
        let on_device = |shell: &mut Shell, refs: &[u32]| {
            let item = shell
                .guest_span(role, refs, bytes)
                .expect("the run is a guest item's span");
            let mut held = vec![0u8; pattern.len()];
            engine_cuda::device::alloc::copy_d2h_on(stream, item, &mut held)
                .expect("the device slots read");
            held
        };

        shell
            .write_tier_slots(&[(id, seed.clone())], &pattern)
            .expect("the host slots are written");
        shell
            .copy_tier_slots(id, Tier::Host, Tier::Device, &pairs(&seed, &device))
            .expect("the slots land on the device");
        shell
            .copy_tier_slots(id, Tier::Device, Tier::Host, &pairs(&device, &parked))
            .expect("the slots park");
        let held = shell
            .read_tier_slots(&[(id, parked.clone())])
            .expect("the parked slots read");
        assert!(held == pattern, "{role:?}: the host tier holds the slots");
        assert!(
            on_device(&mut shell, &device) == pattern,
            "{role:?}: the copy wrote where a guest's item lies"
        );

        shell
            .write_tier_slots(&[(id, seed.clone())], &other)
            .expect("the host slots are written");
        shell
            .copy_tier_slots(id, Tier::Host, Tier::Device, &pairs(&seed, &device))
            .expect("the device slots are overwritten");
        shell
            .copy_tier_slots(id, Tier::Host, Tier::Device, &pairs(&parked, &fresh))
            .expect("the slots come back");
        // Every move landed: a byte read of host slots waits for them.
        shell
            .read_tier_slots(&[(id, parked.clone())])
            .expect("the parked slots read");
        assert!(
            on_device(&mut shell, &device) == other,
            "{role:?}: the slots the item left were overwritten"
        );
        assert!(
            on_device(&mut shell, &fresh) == pattern,
            "{role:?}: the item came back to its fresh slots"
        );
    }
}
