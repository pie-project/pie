#![cfg(target_vendor = "apple")]

//! The weight cache's streamed weights (`engine_metal::store::streamed`:
//! token-id-gathered table rows, routed experts) land the same logits, bit
//! for bit, wherever their atoms are read from: the weight in place (nothing
//! named), every atom in a pool slot, every other atom in one, the whole set
//! held (the weights' own copies given back), and every atom in a slot whose
//! chunks are evacuated mid-decode; each against the load that states no
//! weight shape at all (the contiguous weights, as before the cache). The
//! use record counts one row read a token and whole routes of experts, and
//! a seat-addressed fire refuses to run over a weight slot in the seats' own
//! chunks.

use engine::chunk::{Evacuation, ResidentWeight, WeightAtom, WeightKind, WeightResidency};
use engine_metal::serve::Seated;
use engine_metal::{Boot, Lane, Shell};
use poem_compiler::Budget;

use super::a_scattered_pool_serves_the_same_logits::{Fixture, argmax, fixture_of, same, word};

const PAGE: u32 = 16;
const CONTEXT: u32 = 256;
const SEATS: u32 = 4;
const STEPS: usize = 12;

fn load(fixture: &Fixture, whole: bool) -> Shell {
    load_as(fixture, whole, false, &fixture.container)
}

/// The shell over `checkpoint`, told the runtime holds its weights where
/// `held`.
fn load_as(fixture: &Fixture, whole: bool, held: bool, checkpoint: &std::path::Path) -> Shell {
    let residency = engine_metal::ResidencyPlan::default().with_weights_held(held);
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint,
        budget: Budget::new(SEATS, 256),
        patches: None,
        voxels: None,
        profile: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: SEATS,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        runahead: engine::runahead::Runahead::F1,
        residency: if whole {
            residency.with_whole_weights()
        } else {
            residency
        },
        profiles: None,
    })
    .unwrap_or_else(|why| panic!("the shell loads: {why}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// No weight shape stated: the contiguous table.
    Whole,
    /// Shapes stated, nothing named: every block read in place.
    Host,
    /// Every block in a slot.
    Resident,
    /// Every other block in a slot.
    Mixed,
    /// Every block in a slot, held: the tables' own copies go.
    Held,
    /// Every block in a slot, the slots' chunks evacuated mid-decode.
    Evacuated,
    /// Told held: the load seats every atom itself (out of a serving
    /// artifact's file where `warm`) and the runtime adopts the seating.
    Seated { warm: bool },
    /// Told held, warm: the runtime names every atom another slot.
    Reseated,
}

/// Whether the fixture's container is a serving artifact already (a
/// quantized deployment's: it loads from nothing else).
fn is_serving(fixture: &Fixture) -> bool {
    fixture.container.to_string_lossy().ends_with(".serving.zt")
}

/// The fixture's planes as a serving artifact beside its checkpoint.
fn serving(fixture: &Fixture) -> std::path::PathBuf {
    if is_serving(fixture) {
        return fixture.container.clone();
    }
    let path = fixture.container.with_extension("serving.zt");
    if !path.exists() {
        super::a_scattered_pool_serves_the_same_logits::write_serving(&path, &fixture.trace);
    }
    path
}

/// The slot refs every block of every stated shape takes from chunk
/// `first` on, shape after shape; and the chunks they span.
fn lay(shell: &Shell, first: u32, keep: impl Fn(u32) -> bool) -> (Vec<ResidentWeight>, u32) {
    let facts = shell.pool_chunk_facts();
    let mut chunk = first;
    let mut named = Vec::new();
    for shape in &facts.weights {
        named.extend(
            (0..shape.count)
                .filter(|&atom| keep(atom))
                .map(|atom| ResidentWeight {
                    atom: WeightAtom {
                        shape: shape.id,
                        atom,
                    },
                    slot: chunk * shape.slots_per_chunk + atom,
                }),
        );
        chunk += shape.count.div_ceil(shape.slots_per_chunk);
    }
    (named, chunk - first)
}

/// A 150-token prefill and `STEPS` greedy decodes on seat 0: every logit
/// row, and the tokens fed.
fn serve(fixture: &Fixture, mode: Mode) -> (Vec<Vec<f32>>, u64) {
    let told = matches!(mode, Mode::Seated { .. } | Mode::Reseated);
    let warm = matches!(mode, Mode::Seated { warm: true } | Mode::Reseated);
    let started = std::time::Instant::now();
    let mut shell = if told {
        let checkpoint = if warm {
            serving(fixture)
        } else {
            fixture.container.clone()
        };
        load_as(fixture, false, true, &checkpoint)
    } else {
        load(fixture, mode == Mode::Whole)
    };
    eprintln!(
        "{mode:?}: loaded in {} ms (weights warm: {})",
        started.elapsed().as_millis(),
        shell.weights_warm()
    );
    assert_eq!(
        shell.weights_warm(),
        warm || is_serving(fixture),
        "{mode:?}: the road the load took"
    );
    let facts = shell.pool_chunk_facts();
    for shape in &facts.weights {
        assert_eq!(
            shape.seated.is_some(),
            told,
            "{mode:?}: shape {} states the load's seating only where told held",
            shape.id
        );
    }
    facts.validate().expect("the facts hang together");
    // A load told held reads its token-id tables in place (unified memory:
    // no pool copy); a miniature that streams only such tables then states
    // no shape and serves as the whole load does.
    let in_place = told && facts.weights.is_empty();
    if mode == Mode::Whole || in_place {
        assert!(facts.weights.is_empty(), "a whole-weights load states none");
    } else {
        assert!(
            !facts.weights.is_empty(),
            "this miniature streams a table by its rows"
        );
        if mode == Mode::Host {
            for shape in &facts.weights {
                eprintln!(
                    "weight shape {}: {:?} {} atoms, {} B slots, {} a chunk of {} B, floor {} ({})",
                    shape.id,
                    shape.kind,
                    shape.count,
                    shape.slot_bytes,
                    shape.slots_per_chunk,
                    facts.chunk_bytes,
                    shape.floor_slots,
                    shape.name.split(',').count(),
                );
            }
        }
    }
    // The weight slots from the first segment above the seats' runs, so a
    // seat-addressed fire never writes them.
    let segment = facts.segment_chunks;
    let first = (1 + shell.pool_layout().seat_chunks()).div_ceil(segment) * segment;
    let (all, span) = lay(&shell, first, |_| true);
    // Room for the slots twice over: the evacuation's targets above them.
    let commit = (first + 2 * span).div_ceil(segment);
    shell.commit_segments(commit).expect("the segments commit");
    let named = match mode {
        _ if in_place => None,
        Mode::Seated { .. } => Some((
            facts
                .weights
                .iter()
                .flat_map(|shape| {
                    let seated = shape.seated.expect("seated");
                    (0..shape.count).map(move |atom| ResidentWeight {
                        atom: WeightAtom {
                            shape: shape.id,
                            atom,
                        },
                        slot: seated + atom,
                    })
                })
                .collect(),
            true,
        )),
        // Past the load's seating: every atom moves.
        Mode::Reseated => {
            let last = facts
                .weights
                .iter()
                .map(|shape| {
                    (shape.seated.expect("seated") + shape.count - 1) / shape.slots_per_chunk
                })
                .max()
                .unwrap_or(0);
            Some((lay(&shell, last + 1, |_| true).0, true))
        }
        Mode::Whole | Mode::Host => None,
        Mode::Resident | Mode::Evacuated => Some((all.clone(), false)),
        Mode::Held => Some((all.clone(), true)),
        Mode::Mixed => Some((lay(&shell, first, |atom| atom % 2 == 0).0, false)),
    };
    if let Some((resident, held)) = named {
        let count = resident.len() as u64;
        shell
            .weight_residency(&WeightResidency {
                staging: Vec::new(),
                resident,
                held,
            })
            .expect("the residency lands");
        assert_eq!(shell.weight_atoms().0, count, "{mode:?}: the blocks named");
        assert_eq!(
            shell.weights_given_back(),
            mode == Mode::Held || (told && !in_place),
            "{mode:?}: the tables' own copies go only under a held set naming every block"
        );
    }
    let prompt: Vec<u32> = (0..150u32).map(|at| (at * 37 + 11) % 256).collect();
    let mut held = 0u32;
    let mut fed = prompt;
    let mut tokens = 0u64;
    let mut rows = Vec::new();
    for step in 0..=STEPS {
        if mode == Mode::Evacuated && step == STEPS / 2 {
            let moves = (0..span)
                .map(|at| (first + at, first + span + at))
                .collect();
            shell
                .evacuate(&Evacuation { moves })
                .expect("the evacuation lands");
        }
        let seated = Seated {
            held: Some(held),
            ..Seated::of(Lane {
                slot: 0,
                word: word(&fixture.trace, fed.len() as u32),
                tokens: &fed,
            })
        };
        let got = shell.fire_seated(&[seated]).expect("the fire lands");
        held += fed.len() as u32;
        tokens += fed.len() as u64;
        fed = vec![argmax(&got[0])];
        rows.extend(got);
    }
    if mode != Mode::Whole {
        let uses = shell.weight_uses().expect("the use record reads");
        for shape in &facts.weights {
            let reads: u64 = uses
                .uses
                .iter()
                .filter(|(atom, _)| atom.shape == shape.id)
                .map(|(_, n)| u64::from(*n))
                .sum();
            match shape.kind {
                // One read a token, at the shape's first table.
                WeightKind::Rows => assert_eq!(reads, tokens, "{mode:?}: the row reads counted"),
                // Each route a token takes reads an expert of each layer.
                WeightKind::Expert => {
                    assert!(
                        reads >= tokens && reads.is_multiple_of(tokens),
                        "{mode:?}: {reads} expert reads over {tokens} tokens"
                    );
                    if mode == Mode::Host {
                        eprintln!("weight shape {}: {reads} expert reads", shape.id);
                    }
                }
            }
        }
        let resident: u32 = uses.held.iter().map(|h| h.resident).sum();
        assert_eq!(u64::from(resident), shell.weight_atoms().0);
        let again = shell.weight_uses().expect("the use record reads");
        assert!(again.uses.is_empty(), "a read record is zeroed");
    }
    if mode == Mode::Held {
        // A held set is the load's for life.
        let moved = lay(&shell, first + span, |_| true).0;
        assert!(
            shell
                .weight_residency(&WeightResidency {
                    staging: Vec::new(),
                    resident: moved,
                    held: true,
                })
                .is_err(),
            "a held block named a new slot after its table went"
        );
        assert!(
            shell
                .weight_residency(&WeightResidency {
                    staging: Vec::new(),
                    resident: all,
                    held: false,
                })
                .is_err(),
            "a set that is not held after a held one"
        );
    }
    if mode == Mode::Host {
        // A weight slot in the seats' runs: a seat-addressed fire refuses.
        let (seat_named, _) = lay(&shell, 1, |atom| atom == 0);
        shell
            .weight_residency(&WeightResidency {
                staging: Vec::new(),
                resident: seat_named,
                held: false,
            })
            .expect("the residency lands");
        let seated = Seated {
            held: Some(held),
            ..Seated::of(Lane {
                slot: 0,
                word: word(&fixture.trace, 1),
                tokens: &fed,
            })
        };
        assert!(
            shell.fire_seated(&[seated]).is_err(),
            "a seat-addressed fire over a weight slot in a seat run"
        );
    }
    (rows, tokens)
}

fn check(model: &str) {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    check_fixture(model, &fixture_of(model, None));
}

fn check_fixture(model: &str, fixture: &Fixture) {
    let fixture = fixture;
    let (whole, _) = serve(fixture, Mode::Whole);
    for mode in [
        Mode::Host,
        Mode::Resident,
        Mode::Mixed,
        Mode::Held,
        Mode::Evacuated,
        Mode::Seated { warm: false },
        Mode::Seated { warm: true },
        Mode::Reseated,
    ] {
        let (got, _) = serve(fixture, mode);
        // A dropped shell gives the device back what it held, its
        // weight-cache tables in the residency set too: the next load
        // measures what the device holds before it as held outside.
        let left = engine_metal::Context::bind().expect("a device").allocated();
        assert!(
            left <= 16 << 20,
            "{model}: {mode:?}: the dropped shell leaves {} MiB allocated",
            left >> 20
        );
        same(
            &format!("{model}: rows {mode:?} against whole"),
            &got,
            &whole,
        );
        eprintln!("{model}: {mode:?} = whole over {} rows", got.len());
    }
}

#[test]
fn a_row_streamed_per_layer_table_serves_the_dense_logits() {
    check("gemma4-e4b-mini-l1");
}

#[test]
fn a_row_streamed_embedding_serves_the_dense_logits() {
    check("dsv4-flash-mini");
}

/// Six layers' per-layer tables gathered alike: one rows shape whose slot
/// holds a block of each, table after table.
#[test]
fn row_streamed_tables_gathered_alike_serve_the_dense_logits() {
    check("gemma4-e4b-mini-l6");
}

/// A mixture's dense routed banks, an expert at a time.
fn experts(model: &str) {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of(model, None);
    let shell = load(&fixture, false);
    let facts = shell.pool_chunk_facts();
    drop(shell);
    assert!(
        facts
            .weights
            .iter()
            .any(|shape| shape.kind == WeightKind::Expert),
        "{model} streams no routed bank an expert at a time: {:?}",
        facts
            .weights
            .iter()
            .map(|shape| (shape.id, shape.kind, &shape.name))
            .collect::<Vec<_>>()
    );
    check(model);
}

// The other miniatures with dense routed banks do not serve on this engine
// today, for reasons apart from the cache: dots3-note-mini needs MLA
// prefill, and mimo26-flash-mini's layout reads an experts dtype the bf16
// fixture does not state.

/// A Mamba2 hybrid: its convs feed no deferred recurrence, so the shell
/// buffers none in an rs list (as the recipe keeps none).
#[test]
fn a_mixtures_dense_experts_serve_the_dense_logits_nemotron() {
    experts("nemotron3-nano-mini");
}

/// A gated-delta hybrid with a conv no deferred recurrence reads.
#[test]
fn a_mixtures_dense_experts_serve_the_dense_logits_qwen38_flash_next() {
    experts("qwen38-flash-next-mini");
}

#[test]
fn a_mixtures_dense_experts_serve_the_dense_logits_minimax() {
    experts("minimax-m3-mini");
}

#[test]
fn a_mixtures_dense_experts_serve_the_dense_logits_qwen3() {
    experts("qwen3-30b-a3b-mini");
}

/// Experts the recipe cuts into 27 pieces on their output axis.
#[test]
fn a_mixtures_dense_experts_serve_the_dense_logits_dsv4() {
    experts("dsv4-flash-mini");
}

/// The process's resident set and physical footprint (what the system
/// charges it, Metal heaps and wired pages included), in MiB.
#[allow(unsafe_code)]
fn memory_mib() -> (u64, u64) {
    let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
    // SAFETY: a v2 record the call fills for this process.
    let ok = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as i32,
            libc::RUSAGE_INFO_V2,
            (&raw mut info).cast(),
        )
    };
    assert_eq!(ok, 0, "proc_pid_rusage");
    (info.ri_resident_size >> 20, info.ri_phys_footprint >> 20)
}

/// Probe (ignored): a warm load out of a serving artifact, told held
/// (`PIE_PROBE_HELD=1`) or not, the runtime then naming every atom held,
/// then a prefill and decodes: load time and the resident set before the
/// load, after it, and after the fires. `PIE_PROBE_MODEL` names the
/// miniature (qwen3-30b-a3b-mini), `PIE_PROBE_LAYERS` cuts it.
#[test]
#[ignore = "a probe: run one mode a process"]
fn a_held_warm_load_probe() {
    if !engine_metal::device::present() {
        return;
    }
    let model = std::env::var("PIE_PROBE_MODEL").unwrap_or("qwen3-30b-a3b-mini".into());
    let layers = std::env::var("PIE_PROBE_LAYERS")
        .ok()
        .and_then(|n| n.parse().ok());
    let held = std::env::var("PIE_PROBE_HELD").is_ok_and(|v| v == "1");
    let fixture = fixture_of(&model, layers);
    let path = serving(&fixture);
    // Written by an earlier process (`PIE_PROBE_WRITE=1` writes and stops),
    // so this one's memory holds no fixture writing.
    if std::env::var("PIE_PROBE_WRITE").is_ok_and(|v| v == "1") {
        eprintln!("probe fixture written: {}", path.display());
        return;
    }
    let before = memory_mib();
    let started = std::time::Instant::now();
    let mut shell = load_as(&fixture, false, held, &path);
    let loaded = started.elapsed();
    let at_load = memory_mib();
    let facts = shell.pool_chunk_facts();
    let segment = facts.segment_chunks;
    let first = (1 + shell.pool_layout().seat_chunks()).div_ceil(segment) * segment;
    let resident: Vec<ResidentWeight> = if held {
        facts
            .weights
            .iter()
            .flat_map(|shape| {
                let seated = shape.seated.expect("seated");
                (0..shape.count).map(move |atom| ResidentWeight {
                    atom: WeightAtom {
                        shape: shape.id,
                        atom,
                    },
                    slot: seated + atom,
                })
            })
            .collect()
    } else {
        let (all, span) = lay(&shell, first, |_| true);
        shell
            .commit_segments((first + span).div_ceil(segment))
            .expect("the segments commit");
        all
    };
    let started = std::time::Instant::now();
    shell
        .weight_residency(&WeightResidency {
            staging: Vec::new(),
            resident,
            held: true,
        })
        .expect("the residency lands");
    let named = started.elapsed();
    let mut fed: Vec<u32> = (0..150u32).map(|at| (at * 37 + 11) % 256).collect();
    let mut seen = 0u32;
    let mut last = Vec::new();
    for _ in 0..8 {
        let seated = Seated {
            held: Some(seen),
            ..Seated::of(Lane {
                slot: 0,
                word: word(&fixture.trace, fed.len() as u32),
                tokens: &fed,
            })
        };
        let got = shell.fire_seated(&[seated]).expect("the fire lands");
        seen += fed.len() as u32;
        fed = vec![argmax(&got[0])];
        last = got;
    }
    let weights: u64 = facts.weights.iter().map(|s| s.all_bytes()).sum();
    eprintln!(
        "probe {model} held={held}: load {} ms, first residency {} ms, weight shapes {} MiB, \
         (resident, footprint) MiB before {before:?} at load {at_load:?} after fires {:?} \
         (token {})",
        loaded.as_millis(),
        named.as_millis(),
        weights >> 20,
        memory_mib(),
        argmax(&last[0]),
    );
}

/// A quantized deployment's routed banks (their codes, scales, zero points
/// and expert biases, an expert at a time) or tables (a block of rows at a
/// time), out of the test kit's serving artifact.
fn quantized(name: &str, kind: WeightKind) {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = super::a_scattered_pool_serves_the_same_logits::deployed_fixture(name);
    let shell = load(&fixture, false);
    let facts = shell.pool_chunk_facts();
    drop(shell);
    assert!(
        facts.weights.iter().any(|shape| shape.kind == kind),
        "{name} streams no {kind:?} shape: {:?}",
        facts
            .weights
            .iter()
            .map(|shape| (shape.id, shape.kind, &shape.name))
            .collect::<Vec<_>>()
    );
    check_fixture(name, &fixture);
}

/// Affine u4/64 experts (and dense planes).
#[test]
fn a_mixtures_quantized_experts_serve_the_whole_logits_affine() {
    quantized("qwen3-30b-a3b-mini-u4g64-kv-bf16", WeightKind::Expert);
}

/// MXFP4 experts (bf16 dense planes).
#[test]
fn a_mixtures_quantized_experts_serve_the_whole_logits_mxfp4() {
    quantized("dsv41-flash-mini-bf16-mxfp4-kv-bf16", WeightKind::Expert);
}

/// Affine 4-bit per-layer tables, their rows a block at a time.
#[test]
fn quantized_per_layer_tables_serve_the_whole_logits() {
    quantized("gemma4-e4b-mini-l6-u4g64-kv-bf16", WeightKind::Rows);
}

/// Affine u2/128 experts (and a u4/32 n-gram table, u4/64 and u8/64 dense
/// planes).
#[test]
fn a_mixtures_quantized_experts_serve_the_whole_logits_affine_u2() {
    quantized(
        "qwen38-flash-next-mini-u4g64-u2g128-kv-bf16",
        WeightKind::Expert,
    );
}
