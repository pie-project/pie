#![cfg(target_vendor = "apple")]

//! The weight cache's row-streamed tables (`engine_metal::store::rows`) land
//! the same logits, bit for bit, wherever their blocks are read from: the
//! table in place (nothing named), every block in a pool slot, every other
//! block in one, the whole set held (the tables' own copies given back
//! mid-script, the decode running on), and every block in a slot whose
//! chunks are evacuated mid-decode; each against the load that states no
//! weight shape at all (the contiguous table, as before the cache). The
//! use record counts one read a token, and a seat-addressed fire refuses to
//! run over a weight slot in the seats' own chunks.

use engine::chunk::{Evacuation, ResidentWeight, WeightAtom, WeightResidency};
use engine_metal::serve::Seated;
use engine_metal::{Boot, Lane, Shell};
use poem_compiler::Budget;

use super::a_scattered_pool_serves_the_same_logits::{Fixture, argmax, fixture_of, same, word};

const PAGE: u32 = 16;
const CONTEXT: u32 = 256;
const SEATS: u32 = 4;
const STEPS: usize = 12;

fn load(fixture: &Fixture, whole: bool) -> Shell {
    let residency = engine_metal::ResidencyPlan::default();
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
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
    let mut shell = load(fixture, mode == Mode::Whole);
    let facts = shell.pool_chunk_facts();
    facts.validate().expect("the facts hang together");
    if mode == Mode::Whole {
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
        assert_eq!(shell.row_blocks().0, count, "{mode:?}: the blocks named");
        assert_eq!(
            shell.row_tables_given_back(),
            mode == Mode::Held,
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
        let reads: u64 = uses.uses.iter().map(|(_, n)| u64::from(*n)).sum();
        // One read a token at each shape's first table.
        assert_eq!(
            reads,
            tokens * facts.weights.len() as u64,
            "{mode:?}: the reads counted"
        );
        let resident: u32 = uses.held.iter().map(|h| h.resident).sum();
        assert_eq!(u64::from(resident), shell.row_blocks().0);
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
    let fixture = fixture_of(model, None);
    let (whole, _) = serve(&fixture, Mode::Whole);
    for mode in [
        Mode::Host,
        Mode::Resident,
        Mode::Mixed,
        Mode::Held,
        Mode::Evacuated,
    ] {
        let (got, _) = serve(&fixture, mode);
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
