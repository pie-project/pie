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
        kv_budget: Some(engine_cuda::store::KvSize::Bytes(crate::POOL)),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            pin: Some((Graphs::Off, Some(Recording::Shaped))),
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

/// Host refs handed out as the runtime's host tier hands them
/// (`SharedPool::tier`): one chunk id space from 0, a chunk opened for one
/// shape, a run its lowest free adjacent slots.
#[derive(Default)]
struct HostRefs {
    next: u32,
    open: Vec<(engine::chunk::ShapeId, u32, u32)>,
}

impl HostRefs {
    fn take_run(&mut self, shape: &engine::chunk::SlotShape) -> Vec<u32> {
        let at = match self.open.iter().position(|&(id, ..)| id == shape.id) {
            Some(at) => at,
            None => {
                self.open.push((shape.id, self.next, 0));
                self.next += 1;
                self.open.len() - 1
            }
        };
        let (_, chunk, used) = self.open[at];
        assert!(
            used + RUN <= shape.slots_per_chunk,
            "a test's runs fit a chunk"
        );
        self.open[at].2 += RUN;
        run(shape.slots_per_chunk, chunk, used)
    }
}

/// One guest shape's item through the test.
struct Item {
    role: SlotRole,
    id: engine::chunk::ShapeId,
    bytes: u64,
    /// The device slots it is written in, and those it comes back to.
    device: Vec<u32>,
    fresh: Vec<u32>,
    /// The host slots its bytes are written through, and those it parks in.
    seed: Vec<u32>,
    parked: Vec<u32>,
    pattern: Vec<u8>,
}

fn on_device(shell: &mut Shell, item: &Item, refs: &[u32]) -> Vec<u8> {
    let at = shell
        .guest_span(item.role, refs, item.bytes)
        .expect("the run is a guest item's span");
    let mut held = vec![0u8; item.pattern.len()];
    engine_cuda::device::alloc::copy_d2h_on(shell.stream(), at, &mut held)
        .expect("the device slots read");
    held
}

/// An `Instance` item and a `Channel` item, both parked at once in host
/// refs the runtime's host tier would hand out: known bytes written through
/// the host tier into a run of device slots lie where a guest's item of
/// that run is addressed; they park on the host (a suspend), the device
/// slots are overwritten, and they come back into fresh slots of another
/// chunk (a restore), byte for byte, neither shape's host slots touched by
/// the other's.
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
    shell.take_refs();
    let facts = shell.chunk_facts().expect("the pool states its chunks");
    let segments = facts.segments.max.min(4);
    shell
        .commit_segments(segments)
        .expect("the segments commit");
    assert!(
        segments * facts.segment_chunks > 4,
        "two committed chunks a guest shape past the null chunk"
    );
    let guest = |role| {
        facts
            .shapes
            .iter()
            .find(|shape| shape.role == role)
            .expect("the load states the guest shapes")
    };

    // No host tier yet: the runtime parks nothing (its roles have no host
    // pool), and a copy asked all the same is refused by name.
    assert_eq!(facts.host_chunks, 0, "no host tier is seated at the load");
    let instance = guest(SlotRole::Instance);
    let refused = shell
        .copy_tier_slots(
            instance.id,
            Tier::Device,
            Tier::Host,
            &[(instance.slots_per_chunk, 0)],
        )
        .expect_err("a copy to a host tier of no chunks");
    println!("no host tier: {refused}");
    assert!(
        refused.to_string().contains("chunks of the host tier"),
        "{refused}"
    );

    let ring = engine_cuda::store::staging_ring_for(
        Some(crate::POOL),
        1,
        engine::runahead::Runahead::DEFAULT_FRAMES_IN_FLIGHT,
    )
    .and_then(|bytes| bytes.map(engine_cuda::store::Ring::pin).transpose())
    .expect("the staging ring pins");
    shell
        .seat_host_kv(Some(crate::POOL), 1, 256, 1, ring)
        .expect("the host tier seats");
    let facts = shell.chunk_facts().expect("the pool states its chunks");
    assert!(facts.host_chunks >= 2, "a host chunk a guest shape");

    let mut host = HostRefs::default();
    let mut items: Vec<Item> = [SlotRole::Instance, SlotRole::Channel]
        .into_iter()
        .enumerate()
        .map(|(at, role)| {
            let at = at as u32;
            let shape = guest(role);
            let bytes = u64::from(RUN) * shape.slot_bytes;
            Item {
                role,
                id: shape.id,
                bytes,
                // A shape's chunks are its own on the device too.
                device: run(shape.slots_per_chunk, 1 + 2 * at, 5),
                fresh: run(shape.slots_per_chunk, 2 + 2 * at, 11),
                seed: host.take_run(shape),
                parked: Vec::new(),
                pattern: (0..bytes)
                    .map(|byte| (byte * 31 + 17 + u64::from(at) * 101) as u8 ^ (byte >> 8) as u8)
                    .collect(),
            }
        })
        .collect();
    for item in &mut items {
        item.parked = host.take_run(guest(item.role));
    }
    // Both shapes' host refs start at their chunk's first slots: the same
    // ref numbers a shape apart would be the same bytes.
    println!(
        "host refs: instance seed {:?} parked {:?}, channel seed {:?} parked {:?}",
        items[0].seed, items[0].parked, items[1].seed, items[1].parked
    );

    // Onto the device, then both parked.
    for item in &items {
        shell
            .write_tier_slots(&[(item.id, item.seed.clone())], &item.pattern)
            .expect("the host slots are written");
        shell
            .copy_tier_slots(
                item.id,
                Tier::Host,
                Tier::Device,
                &pairs(&item.seed, &item.device),
            )
            .expect("the slots land on the device");
    }
    for item in &items {
        shell
            .copy_tier_slots(
                item.id,
                Tier::Device,
                Tier::Host,
                &pairs(&item.device, &item.parked),
            )
            .expect("the slots park");
    }
    for item in &items {
        let held = shell
            .read_tier_slots(&[(item.id, item.parked.clone())])
            .expect("the parked slots read");
        assert!(
            held == item.pattern,
            "{:?}: the host tier holds the slots",
            item.role
        );
        assert!(
            on_device(&mut shell, item, &item.device) == item.pattern,
            "{:?}: the copy wrote where a guest's item lies",
            item.role
        );
    }

    // The device slots overwritten, then both restored to fresh slots.
    for item in &items {
        let other = vec![0xa5u8; item.pattern.len()];
        shell
            .write_tier_slots(&[(item.id, item.seed.clone())], &other)
            .expect("the host slots are written");
        shell
            .copy_tier_slots(
                item.id,
                Tier::Host,
                Tier::Device,
                &pairs(&item.seed, &item.device),
            )
            .expect("the device slots are overwritten");
    }
    for item in &items {
        shell
            .copy_tier_slots(
                item.id,
                Tier::Host,
                Tier::Device,
                &pairs(&item.parked, &item.fresh),
            )
            .expect("the slots come back");
    }
    for item in &items {
        // Every move landed: a byte read of host slots waits for them.
        let held = shell
            .read_tier_slots(&[(item.id, item.parked.clone())])
            .expect("the parked slots read");
        assert!(
            held == item.pattern,
            "{:?}: its parked slots stand through the other shape's moves",
            item.role
        );
        assert!(
            on_device(&mut shell, item, &item.device) == vec![0xa5u8; item.pattern.len()],
            "{:?}: the slots the item left were overwritten",
            item.role
        );
        assert!(
            on_device(&mut shell, item, &item.fresh) == item.pattern,
            "{:?}: the item came back to its fresh slots",
            item.role
        );
    }
}

/// A park rides the swap stream past the call: the shell tells it landed
/// only once the copy has read its device slots, so what the runtime gives
/// those slots to next (any shape, written by anything, not a fire alone)
/// cannot reach the parked bytes. The slots are overwritten the moment the
/// park is told landed, from the host and on no stream ordered behind it,
/// and the host tier still holds what they held.
#[test]
fn a_park_is_told_landed_once_it_has_read_its_slots() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("guest-tier-landing", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 7718).write(dir.path());
    let mut shell = shell(trace, &path);
    shell.take_refs();
    let facts = shell.chunk_facts().expect("the pool states its chunks");
    shell
        .commit_segments(facts.segments.max.min(4))
        .expect("the segments commit");
    let ring = engine_cuda::store::staging_ring_for(
        Some(crate::POOL),
        1,
        engine::runahead::Runahead::DEFAULT_FRAMES_IN_FLIGHT,
    )
    .and_then(|bytes| bytes.map(engine_cuda::store::Ring::pin).transpose())
    .expect("the staging ring pins");
    shell
        .seat_host_kv(Some(crate::POOL), 1, 256, 1, ring)
        .expect("the host tier seats");
    let facts = shell.chunk_facts().expect("the pool states its chunks");
    let shape = facts
        .shapes
        .iter()
        .find(|shape| shape.role == SlotRole::Channel)
        .expect("the load states the guest shapes")
        .clone();
    let mut host = HostRefs::default();
    let bytes = u64::from(RUN) * shape.slot_bytes;
    let item = Item {
        role: SlotRole::Channel,
        id: shape.id,
        bytes,
        device: run(shape.slots_per_chunk, 1, 5),
        fresh: Vec::new(),
        seed: host.take_run(&shape),
        parked: host.take_run(&shape),
        pattern: (0..bytes).map(|byte| (byte * 13 + 7) as u8 ^ (byte >> 9) as u8).collect(),
    };
    shell
        .write_tier_slots(&[(item.id, item.seed.clone())], &item.pattern)
        .expect("the host slots are written");
    shell
        .copy_tier_slots(item.id, Tier::Host, Tier::Device, &pairs(&item.seed, &item.device))
        .expect("the slots land on the device");

    for round in 0..8 {
        shell
            .copy_tier_slots(item.id, Tier::Device, Tier::Host, &pairs(&item.device, &item.parked))
            .expect("the slots park");
        let (told, landed) = std::sync::mpsc::channel();
        let telling = shell
            .tell_swaps_landed(Box::new(move || {
                let _ = told.send(());
            }))
            .expect("the shell takes the landing");
        assert!(telling, "a park on the swap stream is told landed later");
        landed
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("the park is told landed");
        // The slots go to their next owner, which writes them at once.
        let at = shell
            .guest_span(item.role, &item.device, item.bytes)
            .expect("the run is a guest item's span");
        engine_cuda::device::alloc::stage_raw(
            shell.stream(),
            at,
            &vec![0x5au8 ^ round as u8; item.pattern.len()],
        )
        .expect("the next owner writes the slots");
        let held = shell
            .read_tier_slots(&[(item.id, item.parked.clone())])
            .expect("the parked slots read");
        assert!(
            held == item.pattern,
            "round {round}: the host tier holds what the slots held when the park was told landed"
        );
        // Back on the device for the next round's park.
        shell
            .copy_tier_slots(item.id, Tier::Host, Tier::Device, &pairs(&item.seed, &item.device))
            .expect("the slots land on the device");
    }
}
