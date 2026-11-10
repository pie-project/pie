//! The weight cache's routed experts: every routed bank streams an expert at
//! a time through weight slots of the pool the runtime names
//! (`Engine::weight_residency`). Every routed kernel reads an expert through
//! the expert table (a slot, or the stage an expert no slot holds is copied
//! to for its fire) and reads the same elements in the same order, so every
//! residency lands the same bits: every expert resident and held (the
//! deployment that does not stream, its slots then moved by an evacuation),
//! none (every routed expert staged from the host), every other one. Where
//! `$PIE_WGPU_BASE/seats-<model>.bin` holds the logits a build before the cache
//! landed (`$PIE_WGPU_LOGITS` of `serve_seats`, the engine's own expert tier),
//! each run is held against them too.

use engine::chunk::{ResidentWeight, WeightAtom, WeightKind, WeightResidency, WeightUses};

use crate::model_logits::{CONTEXT, fixture, load, same, serve_seats};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    /// No expert resident: every routed expert staged from the host.
    None,
    /// Every other expert resident.
    Half,
    /// Every expert resident, the set held for the load's life.
    All,
}

/// Serves the seats' script under `held`; the logits, and the uses the
/// engine reports after.
fn served(model: &str, held: Held) -> Option<(Vec<Vec<f32>>, WeightUses)> {
    let fixture = fixture(model, None);
    let mut shell = load(&fixture, CONTEXT)?;
    let facts = shell.pool_chunk_facts();
    facts.validate().expect("the facts hang together");
    assert!(
        facts.weights.iter().any(|w| w.kind == WeightKind::Expert),
        "{model}: an expert shape is stated"
    );
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

/// The most any logit of `got` lies from `want`'s, and the steps whose
/// argmax differs.
fn apart(got: &[Vec<f32>], want: &[Vec<f32>]) -> (f32, usize) {
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(at, _)| at)
    };
    let mut most = 0f32;
    let mut flips = 0;
    for (g, w) in got.iter().zip(want) {
        for (a, b) in g.iter().zip(w) {
            most = most.max((a - b).abs());
        }
        flips += usize::from(argmax(g) != argmax(w));
    }
    (most, flips)
}

fn base(model: &str, width: usize) -> Option<Vec<Vec<f32>>> {
    let dir = std::env::var_os("PIE_WGPU_BASE")?;
    let path = std::path::Path::new(&dir).join(format!("seats-{model}.bin"));
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("{model}: no base logits at {}", path.display());
        return None;
    };
    Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_bits(u32::from_le_bytes(*w)))
            .collect::<Vec<_>>()
            .chunks(width)
            .map(<[f32]>::to_vec)
            .collect(),
    )
}

fn every_residency(model: &str) {
    let Some((all, uses)) = served(model, Held::All) else {
        return;
    };
    assert!(
        uses.held.iter().all(|h| h.host == 0 && h.resident > 0),
        "{model}: every expert resident: {:?}",
        uses.held
    );
    assert!(!uses.uses.is_empty(), "{model}: the fires read experts");
    let base = base(model, all[0].len());
    if let Some(base) = &base {
        same(&format!("{model}: all resident against base"), &all, base);
        eprintln!("{model}: all resident and held = base, bit for bit");
    }
    let (none, uses) = served(model, Held::None).expect("loaded before");
    assert!(
        uses.held.iter().all(|h| h.resident == 0),
        "{model}: no expert resident: {:?}",
        uses.held
    );
    let (half, uses) = served(model, Held::Half).expect("loaded before");
    assert!(
        uses.held.iter().all(|h| h.resident > 0 && h.host > 0),
        "{model}: half resident: {:?}",
        uses.held
    );
    for (what, got) in [("none", &none), ("half", &half)] {
        let (most, flips) = apart(got, &all);
        eprintln!("{model}: {what} resident against all: most |d| {most}, argmax flips {flips}");
        same(&format!("{model}: {what} resident against all"), got, &all);
    }
}

/// Qwen3's MoE: dense bf16 routed banks, gate/up and down an atom.
#[test]
fn routed_experts_serve_the_same_logits_whatever_is_resident() {
    every_residency("qwen3-30b-a3b-mini");
}

/// GLM-4.5-Air: dense layers before the routed ones, a shared expert.
#[test]
fn routed_experts_beside_dense_layers_serve_the_same_logits_whatever_is_resident() {
    every_residency("glm45-air-mini");
}

/// Qwen3.6: routed experts in a hybrid (linear attention) model. Ignored:
/// its one-column projection does not serve on WGSL, before the cache as
/// after.
#[test]
#[ignore = "qwen36-35b-a3b-mini does not serve on this backend"]
fn routed_experts_of_a_hybrid_serve_the_same_logits_whatever_is_resident() {
    every_residency("qwen36-35b-a3b-mini");
}
