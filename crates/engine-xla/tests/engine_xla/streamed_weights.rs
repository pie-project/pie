//! Weight streaming on XLA (`engine_xla::store::streamed`): a model's
//! streamable weights (PLE row tables, routed experts) live in the pool,
//! and whatever lies resident, every fire answers what the base build
//! answered at the same pool size. Each case fires the bitwise gate's
//! scenario (`common_llm::scenario`) after a residency and is compared bit
//! for bit with a load that streams nothing (the base build's programs)
//! whose pool is grown by the chunks the weights took; with
//! `PIE_XLA_GOLDEN=<dir>` (the base build's dumps,
//! `a_pool_answers_what_the_base_build_answered`) it says which cases also
//! equal the base build at its own pool size. Skipped without a PJRT plugin.

use engine::Engine;
use engine::chunk::{ResidentWeight, WeightAtom, WeightKind, WeightResidency, WeightShape};

use crate::common_llm::{Model, scenario};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    /// As the load seated them, no residency told.
    Seated,
    /// Every atom named elsewhere in the shape's chunks: every slot filled
    /// again from the host copy.
    Moved,
    /// None resident: every read staged.
    None,
    /// Every other atom resident, the rest staged.
    Half,
    /// Every atom where the load seated it, the set held: the host copy goes.
    Whole,
}

/// The residency `held` names over `weights` (each seated whole).
fn residency(weights: &[WeightShape], held: Held) -> WeightResidency {
    let mut out = WeightResidency {
        held: held == Held::Whole,
        ..WeightResidency::default()
    };
    for shape in weights {
        let first = shape.seated.expect("a streaming load seats its weights");
        let atom = |a: u32| WeightAtom {
            shape: shape.id,
            atom: a,
        };
        let keep: Vec<u32> = match held {
            Held::Seated | Held::Whole | Held::Moved => (0..shape.count).collect(),
            Held::None => Vec::new(),
            Held::Half => (0..shape.count).step_by(2).collect(),
        };
        for &a in &keep {
            let slot = match held {
                Held::Moved => first + (shape.count - 1 - a),
                _ => first + a,
            };
            out.resident.push(ResidentWeight {
                atom: atom(a),
                slot,
            });
        }
        // Staging: the seated slots no kept atom is named in.
        let free: Vec<u32> = (0..shape.count)
            .filter(|a| held == Held::None || (held == Held::Half && a % 2 == 1))
            .map(|a| first + a)
            .take(shape.floor_slots as usize)
            .collect();
        out.staging.push((shape.id, free));
    }
    out
}

fn bits_file(name: &str) -> Option<Vec<u8>> {
    let dir = std::env::var("PIE_XLA_GOLDEN").ok()?;
    std::fs::read(std::path::Path::new(&dir).join(format!("{name}.bits"))).ok()
}

fn encode(readouts: &[Vec<u32>]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in readouts {
        out.extend_from_slice(&(r.len() as u32).to_le_bytes());
        for v in r {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Fires the scenario under `held`; the readouts' bits, the uses read and
/// the chunks the load seated its weights in.
fn served(
    name: &str,
    context: u32,
    long: Option<u32>,
    kind: WeightKind,
    held: Held,
) -> (Vec<u8>, engine::chunk::WeightUses, u32) {
    let mut model = Model::load(name, None, context, 7);
    let facts = model.loaded.caps.chunks.clone().expect("chunk facts");
    let chunks: u32 = facts
        .weights
        .iter()
        .map(|w| w.count.div_ceil(w.slots_per_chunk))
        .sum();
    facts.validate().expect("the facts hang together");
    assert!(
        facts.weights.iter().any(|w| w.kind == kind),
        "{name}: a {kind:?} weight shape is stated: {:?}",
        facts.weights
    );
    if held != Held::Seated {
        model
            .engine
            .weight_residency(&residency(&facts.weights, held))
            .expect("the residency lands");
    }
    let started = std::time::Instant::now();
    let readouts = scenario(&mut model, long);
    eprintln!(
        "{name}: {held:?}: the scenario took {:?}",
        started.elapsed()
    );
    let uses = model.engine.weight_uses().expect("the uses read");
    if held == Held::Whole {
        // The host copy is gone: a residency that moves an atom is refused.
        let moved = residency(&facts.weights, Held::Moved);
        let refused = model.engine.weight_residency(&WeightResidency {
            held: true,
            ..moved
        });
        assert!(
            refused.is_err(),
            "{name}: a held set keeps no host copy to move an atom with"
        );
    }
    (encode(&readouts), uses, chunks)
}

fn every_residency(name: &str, context: u32, long: Option<u32>, kind: WeightKind) {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let golden = bits_file(name);
    let mut seated: Option<Vec<u8>> = None;
    let mut grown = None;
    for held in [
        Held::Seated,
        Held::Moved,
        Held::None,
        Held::Half,
        Held::Whole,
    ] {
        let (bits, uses, chunks) = served(name, context, long, kind, held);
        // The base build's programs at this pool size: a load that streams
        // nothing (its programs are the base build's, `PIE_XLA_DUMP`
        // shows), its pool grown by the chunks the weights were seated in.
        let control = grown.get_or_insert_with(|| {
            let mut model = Model::load_held_grown(name, None, context, 7, chunks);
            let bits = encode(&scenario(&mut model, long));
            if let Some(want) = &golden {
                if *want == bits {
                    eprintln!("{name}: the base build grown by {chunks} chunks = the base build");
                } else {
                    eprintln!(
                        "{name}: the base build grown by {chunks} chunks answers otherwise than \
                         at its own size (the pool's size reaches its programs)"
                    );
                }
            }
            bits
        });
        assert!(
            *control == bits,
            "{name}: {held:?} answers otherwise than the base build's programs at its pool size"
        );
        eprintln!("{name}: {held:?} = the base build's programs at its pool size, bit for bit");
        assert!(
            !uses.uses.is_empty(),
            "{name}: {held:?}: the fires read weights"
        );
        let resident: u32 = uses.held.iter().map(|h| h.resident).sum();
        let host: u32 = uses.held.iter().map(|h| h.host).sum();
        match held {
            Held::None => assert_eq!(resident, 0, "{name}: none resident"),
            Held::Half => assert!(resident > 0 && host > 0, "{name}: half resident"),
            _ => assert_eq!(host, 0, "{name}: {held:?}: every atom resident"),
        }
        match &seated {
            None => seated = Some(bits.clone()),
            Some(want) => assert!(
                *want == bits,
                "{name}: {held:?} answers otherwise than the seated load"
            ),
        }
        if golden.as_ref() == Some(&bits) {
            eprintln!("{name}: {held:?} = the base build's dump, bit for bit");
        }
    }
}

#[test]
fn a_streamed_row_table_answers_the_base_build_whatever_lies_resident() {
    every_residency("gemma4-e4b-mini-l1", 1024, Some(700), WeightKind::Rows);
}

#[test]
fn streamed_experts_answer_the_base_build_whatever_lies_resident() {
    every_residency("qwen3-30b-a3b-mini", 256, None, WeightKind::Expert);
}

/// A load told its weights are held streams nothing: no weight shape.
#[test]
fn a_held_load_states_no_weight_shape() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    for name in ["gemma4-e4b-mini-l1", "qwen3-30b-a3b-mini"] {
        let model = Model::load_held(name, None, 256, 7);
        let facts = model.loaded.caps.chunks.clone().expect("chunk facts");
        assert!(facts.weights.is_empty(), "{name}: {:?}", facts.weights);
    }
}
