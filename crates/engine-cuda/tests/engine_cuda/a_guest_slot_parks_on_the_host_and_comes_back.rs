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
        context_declared: false,
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

    shell
        .seat_host_kv(Some(crate::POOL), 1, 256, 1)
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

/// What a scored session gave: each decode's summed scores and logits, and
/// the bodies' tally after the load armed and after the session.
struct Scored {
    decodes: Vec<(f32, Vec<f32>)>,
    armed: engine_cuda::record::BodyTally,
    after: engine_cuda::record::BodyTally,
}

/// An eviction guest's session on a shell recording bodies as `graphs`
/// says: a score reader registered and bound after the load (after ready),
/// two scored decodes, a park and an unpark of its rows to fresh room
/// elsewhere, two more. Every decode at the same depth: the same scores
/// and logits each time.
fn scored_session(graphs: Graphs) -> Scored {
    use crate::a_score_reader_states_its_planes_and_nothing_backs_them as scored;
    use engine::fire::Readout;
    use engine_cuda::serve::Attached;
    use engine_cuda::{Lane, Seated};
    use eta_compiler::codegen::program::{Backend, emit_program};
    use poem::Request;

    let trace = poem::trace_hybrid("scored-micro", &scored::Scored, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 4409).write(dir.path());
    let mut shell = scored::shell(trace.clone(), &path, graphs);
    let armed = shell.body_stats().tally;
    let pages: Vec<u32> = (0..64 / scored::PAGE).collect();
    let prompt: Vec<u32> = (0..scored::PROMPT)
        .map(|t| (t * 7 + 11) % scored::VOCAB)
        .collect();
    shell
        .fire_rows(
            &[Seated {
                pages: &pages,
                held: Some(0),
                ..Seated::of(Lane {
                    slot: 0,
                    word: trace.facts.word(&Request::new(scored::PROMPT, false)),
                    tokens: &prompt,
                })
            }],
            &[Readout::Last],
        )
        .expect("the prompt prefills");

    let facts = shell.chunk_facts().expect("the chunk facts");
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let decls = crate::bound::loaded_decls_of(&shell.classes());
    let bound = eta_ir::validate::bind(scored::reader(), vec![decls], crate::bound::caps())
        .expect("the reader binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let program = shell
        .register_program(&engine::ProgramRegistration {
            program_hash: 0x5c0f,
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
            launch: eta_compiler::codegen::launch::build(&bound, &stages),
            ..Default::default()
        })
        .expect("the reader registers");
    let extents = eta_exec::Extents {
        row_count: 1,
        token_count: 1,
        sampled_rows: 1,
        query_len: 1,
        ..eta_exec::Extents::default()
    };
    let state = shell
        .guest_state(program, extents)
        .expect("the reader's guest state");
    let mut guest = crate::guest::Guest::of(&facts, segments * facts.segment_chunks);
    let program_facts = shell.program_facts(program).expect("the program's facts");
    let slots = guest.shell_slots(program, &program_facts, extents, state);
    let instance = shell
        .bind_program(
            program,
            &[],
            extents,
            eta_ir::registry::GeometryClass::Host,
            eta_ir::registry::PortMask::NONE,
            &[None],
            &[engine::ChannelId(1)],
            slots.clone(),
        )
        .expect("the reader binds");

    let token = [prompt[0]];
    let depth = [scored::PROMPT];
    let decode = |shell: &mut Shell| -> (f32, Vec<f32>) {
        let logits = shell
            .fire_media(
                &[Seated {
                    pages: &pages,
                    held: Some(scored::PROMPT),
                    positions: &depth,
                    ..Seated::of(Lane {
                        slot: 0,
                        word: trace
                            .facts
                            .word(&scored::decoding(&trace).capturing_scores(true)),
                        tokens: &token,
                    })
                }],
                &[Attached {
                    lane: 0,
                    lanes: 1,
                    instance,
                    adapter_only: false,
                }],
                &[],
            )
            .expect("the scored decode fires");
        shell.drain().expect("the decode lands");
        let cell = shell
            .program_instance(instance)
            .expect("the reader's instance")
            .expect("bound")
            .take(0)
            .expect("the sum reads")
            .expect("a sum");
        (
            f32::from_le_bytes(cell[..4].try_into().expect("an f32")),
            logits.into_iter().flatten().collect(),
        )
    };
    let mut decodes = vec![decode(&mut shell), decode(&mut shell)];

    // Parked and unparked: its scratch and its pair's state, the score
    // rows among them, taken afresh elsewhere; nothing copied.
    let (scratch, pair) = guest.moved(program, extents, state);
    assert_ne!(scratch, slots.scratch, "the instance's scratch moved");
    shell
        .rebind_program(
            instance,
            engine_cuda::program::Slots {
                scratch,
                pair,
                ..slots
            },
        )
        .expect("the reader rebinds");
    decodes.push(decode(&mut shell));
    decodes.push(decode(&mut shell));
    Scored {
        decodes,
        armed,
        after: shell.body_stats().tally,
    }
}

/// A score reader's rows are its instance's state, and a park gives them
/// back: unparked to fresh room elsewhere, the reader's scored decode
/// captures where its rows lie now (through the fire's staged table) and
/// sums the same scores.
#[test]
fn a_parked_score_reader_sums_its_scores_where_its_planes_moved() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let walked = scored_session(Graphs::Off);
    let (first, _) = &walked.decodes[0];
    assert!(*first != 0.0, "the reader summed no scores");
    for (sum, _) in &walked.decodes {
        assert_eq!(sum, first, "the capture writes the rows where they lie now");
    }
}

/// A scored fire is armed like any other (every body is captured before
/// ready): a score reader bound after ready replays its decode's body,
/// parked and unparked too, and nothing is captured or walked after ready.
/// Its scores and its tokens are the walked session's.
#[test]
fn a_scored_fire_after_ready_replays_its_armed_body() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let walked = scored_session(Graphs::Off);
    let replayed = scored_session(Graphs::On);
    eprintln!("armed {:?}\nafter {:?}", replayed.armed, replayed.after);
    assert!(replayed.armed.armed_at_load > 0, "the load armed bodies");
    assert_eq!(
        replayed.after.captures, replayed.armed.captures,
        "no body is captured after ready"
    );
    assert_eq!(
        replayed.after.eager_unbodied, replayed.armed.eager_unbodied,
        "no fire walks unbodied after ready"
    );
    assert!(
        replayed.after.hits >= replayed.armed.hits + replayed.decodes.len() as u64,
        "every scored decode replays a body"
    );
    // The token a decode draws greedily: a body's kernels may round its
    // logits other than the walk's, never past the top token.
    let token = |logits: &[f32]| {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(at, _)| at)
    };
    for ((sum, logits), (want_sum, want_logits)) in replayed.decodes.iter().zip(&walked.decodes) {
        assert_eq!(sum, want_sum, "a replay sums the walked scores");
        assert_eq!(
            token(logits),
            token(want_logits),
            "a replay draws the walked token"
        );
    }
}
