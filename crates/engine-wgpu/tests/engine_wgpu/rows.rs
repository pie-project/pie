//! The weight cache's row tables (per-layer embeddings): the tables stream
//! by blocks of rows through weight slots of the pool the runtime names
//! (`Engine::weight_residency`). A gather copies rows, so every residency
//! lands the same bits: every block resident and held (the deployment that
//! does not stream), none (each row staged from the host), every other one.
//! Where `$PIE_WGPU_BASE/seats-<model>.bin` holds the logits a build before
//! the cache landed (`$PIE_WGPU_LOGITS` of `a_windowed_miniature_serves_two_lanes`),
//! each run is held against them too.

use engine::chunk::{ResidentWeight, WeightAtom, WeightResidency, WeightUses};

use crate::model_logits::{CONTEXT, fixture, load, same, serve_seats};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    /// No block resident: every row staged from the host.
    None,
    /// Every other block resident.
    Half,
    /// Every block resident, the set held for the load's life.
    All,
}

/// Serves the seats' script under `held`; the logits, and the uses the
/// engine reports after.
fn served(model: &str, layers: Option<u32>, held: Held) -> Option<(Vec<Vec<f32>>, WeightUses)> {
    let fixture = fixture(model, layers);
    let mut shell = load(&fixture, CONTEXT)?;
    let facts = shell.pool_chunk_facts();
    facts.validate().expect("the facts hang together");
    assert!(!facts.weights.is_empty(), "{model}: a row table is stated");
    eprintln!(
        "{model}: weight shapes {:?}",
        facts
            .weights
            .iter()
            .map(|w| (
                w.id,
                w.kind,
                w.slot_bytes,
                w.slots_per_chunk,
                w.count,
                w.floor_slots
            ))
            .collect::<Vec<_>>()
    );
    // Weight slots in chunks of their own past the seats' runs.
    let segment = facts.segment_chunks;
    let first = (1 + shell.pool_layout().seat_chunks()).div_ceil(segment) * segment;
    let mut next = first;
    let mut resident = Vec::new();
    for shape in &facts.weights {
        let per = shape.slots_per_chunk;
        for atom in 0..shape.count {
            let keep = match held {
                Held::None => false,
                Held::Half => atom % 2 == 0,
                Held::All => true,
            };
            if keep {
                resident.push(ResidentWeight {
                    atom: WeightAtom {
                        shape: shape.id,
                        atom,
                    },
                    slot: (next + atom / per) * per + atom % per,
                });
            }
        }
        next += shape.count.div_ceil(per);
    }
    let span = next - first;
    shell
        .commit_segments((next + span).div_ceil(segment))
        .expect("the segments commit");
    shell
        .weight_residency(&WeightResidency {
            staging: Vec::new(),
            resident,
            held: held == Held::All,
        })
        .expect("the residency lands");
    // Every weight chunk moved by an evacuation: the entries follow.
    if held == Held::All {
        let moves = (first..next).map(|c| (c, c + span)).collect();
        shell
            .evacuate(&engine::chunk::Evacuation { moves })
            .expect("the evacuation lands");
    }
    let started = std::time::Instant::now();
    let rows = serve_seats(&mut shell, &fixture.trace, 12);
    eprintln!(
        "{model}: {held:?} resident: the script took {:?}",
        started.elapsed()
    );
    let uses = shell.weight_uses().expect("the uses read");
    Some((rows, uses))
}

fn every_residency(model: &str, layers: Option<u32>) {
    let Some((all, uses)) = served(model, layers, Held::All) else {
        return;
    };
    let shapes = uses.held.len();
    assert!(
        uses.held.iter().all(|h| h.host == 0 && h.resident > 0),
        "{model}: every block resident: {:?}",
        uses.held
    );
    assert!(!uses.uses.is_empty(), "{model}: the fires read rows");
    let (none, uses) = served(model, layers, Held::None).expect("loaded before");
    assert!(
        uses.held.len() == shapes && uses.held.iter().all(|h| h.resident == 0),
        "{model}: no block resident: {:?}",
        uses.held
    );
    let (half, uses) = served(model, layers, Held::Half).expect("loaded before");
    assert!(
        uses.held.iter().all(|h| h.resident > 0 && h.host > 0),
        "{model}: half resident: {:?}",
        uses.held
    );
    same(&format!("{model}: none resident against all"), &none, &all);
    same(&format!("{model}: half resident against all"), &half, &all);
    if let Some(dir) = std::env::var_os("PIE_WGPU_BASE") {
        let path = std::path::Path::new(&dir).join(format!("seats-{model}.bin"));
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("{model}: no base logits at {}", path.display());
            return;
        };
        let width = all[0].len();
        let base: Vec<Vec<f32>> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_bits(u32::from_le_bytes(*w)))
            .collect::<Vec<_>>()
            .chunks(width)
            .map(<[f32]>::to_vec)
            .collect();
        same(&format!("{model}: all resident against base"), &all, &base);
        eprintln!("{model}: all resident = base, bit for bit");
    }
}

/// One per-layer table, a slot a token id's row.
#[test]
fn a_row_table_serves_the_same_logits_whatever_is_resident() {
    every_residency("gemma4-e4b-mini-l1", None);
}

/// Six per-layer tables gathered alike: an atom a token id's row of each.
#[test]
fn row_tables_sharing_atoms_serve_the_same_logits_whatever_is_resident() {
    every_residency("gemma4-e4b-mini-l6", None);
}
