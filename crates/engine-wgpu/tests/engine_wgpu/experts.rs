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

use poem::Dtype;

use crate::model_logits::{
    CONTEXT, Fixture, PAGE, fixture, fixture_weighed, load, load_with, same, serve_seats,
};

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
fn served(model: &str, fixture: &Fixture, held: Held) -> Option<(Vec<Vec<f32>>, WeightUses)> {
    let mut shell = load(fixture, CONTEXT)?;
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
    every_residency_of(model, &fixture(model, None));
}

/// `model` with its routed experts (and every other component the
/// deployment states) quantized: `weights` a dtype a component, the bank's
/// codes and scales imported from seeded floats.
fn every_residency_quantized(model: &str, weights: &[Dtype]) {
    let fixture = fixture_weighed(model, weights);
    let label: String = std::iter::once(model.to_string())
        .chain(weights.iter().map(|dtype| format!("{dtype:?}").to_lowercase()))
        .collect::<Vec<_>>()
        .join("-");
    every_residency_of(&label, &fixture);
}

fn every_residency_of(model: &str, fixture: &Fixture) {
    let Some((all, uses)) = served(model, fixture, Held::All) else {
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
    let (none, uses) = served(model, fixture, Held::None).expect("loaded before");
    assert!(
        uses.held.iter().all(|h| h.resident == 0),
        "{model}: no expert resident: {:?}",
        uses.held
    );
    let (half, uses) = served(model, fixture, Held::Half).expect("loaded before");
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
    a_held_load_seats_its_weights(model, fixture, &all);
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

/// The quantized deployments below, each with its routed banks packed as
/// it states.
const QUANTIZED: &[(&str, &[Dtype])] = &[
    ("qwen3-30b-a3b-mini", &[Dtype::Bf16, Dtype::U4g64]),
    ("qwen3-30b-a3b-mini", &[Dtype::Bf16, Dtype::U2g32]),
    ("qwen3-30b-a3b-mini", &[Dtype::Bf16, Dtype::U2g64]),
    ("qwen3-30b-a3b-mini", &[Dtype::Bf16, Dtype::U2g128]),
    ("qwen3-30b-a3b-mini", &[Dtype::Bf16, Dtype::Mxfp4]),
    ("gptoss-20b-mini", &[Dtype::Bf16, Dtype::Mxfp4]),
    ("glm45-air-mini", &[Dtype::U4g64]),
    ("qwen36-35b-a3b-mini", &[Dtype::U4g64]),
];

/// Writes (imports) the quantized fixtures without a device, so the runs
/// that serve them find them written (`$PIE_VK_FIXTURES` /
/// `$PIE_WGPU_FIXTURES` keep them).
#[test]
#[ignore = "fixture preparation: the serving tests write what they lack"]
fn quantized_fixtures_import() {
    let mut failed = Vec::new();
    for &(model, weights) in QUANTIZED {
        let started = std::time::Instant::now();
        match std::panic::catch_unwind(|| fixture_weighed(model, weights)) {
            Ok(fixture) => eprintln!(
                "{model} {weights:?}: {} in {:?}",
                fixture.container.display(),
                started.elapsed()
            ),
            Err(_) => failed.push(format!("{model} {weights:?}")),
        }
    }
    assert!(failed.is_empty(), "these do not import: {failed:?}");
}

/// Affine 4-bit routed banks, a scale and a zero point a group of 64.
#[test]
fn affine_u4_routed_experts_serve_the_same_logits_whatever_is_resident() {
    every_residency_quantized("qwen3-30b-a3b-mini", &[Dtype::Bf16, Dtype::U4g64]);
}

/// Affine 2-bit routed banks, groups of 32, 64 and 128.
#[test]
fn affine_u2_routed_experts_serve_the_same_logits_whatever_is_resident() {
    for bank in [Dtype::U2g32, Dtype::U2g64, Dtype::U2g128] {
        every_residency_quantized("qwen3-30b-a3b-mini", &[Dtype::Bf16, bank]);
    }
}

/// MXFP4 routed banks (e2m1 codes, an e8m0 exponent a block of 32).
#[test]
fn mxfp4_routed_experts_serve_the_same_logits_whatever_is_resident() {
    every_residency_quantized("qwen3-30b-a3b-mini", &[Dtype::Bf16, Dtype::Mxfp4]);
}

/// gpt-oss as deployed: MXFP4 routed banks with an expert bias each, a
/// routed bias sum, attention sinks.
#[test]
fn gpt_oss_mxfp4_routed_experts_serve_the_same_logits_whatever_is_resident() {
    every_residency_quantized("gptoss-20b-mini", &[Dtype::Bf16, Dtype::Mxfp4]);
}

/// GLM-4.5-Air's affine 4-bit routed banks beside dense layers and a shared
/// expert, every dense plane 4-bit too.
#[test]
fn affine_u4_routed_experts_beside_dense_layers_serve_the_same_logits_whatever_is_resident() {
    every_residency_quantized("glm45-air-mini", &[Dtype::U4g64]);
}

/// Qwen3.6 as deployed (4-bit throughout, 8-bit where the deployment
/// states): routed experts of a hybrid.
#[test]
#[ignore = "qwen36-35b-a3b-mini does not serve on this backend"]
fn affine_u4_routed_experts_of_a_hybrid_serve_the_same_logits_whatever_is_resident() {
    every_residency_quantized("qwen36-35b-a3b-mini", &[Dtype::U4g64]);
}

/// The pages of `file` this process's mappings of it hold resident, in
/// KiB (`/proc/self/smaps`).
pub fn mapped_kib(file: &std::path::Path) -> u64 {
    let Ok(smaps) = std::fs::read_to_string("/proc/self/smaps") else {
        return 0;
    };
    let name = file.to_string_lossy();
    let mut ours = false;
    let mut kib = 0u64;
    for line in smaps.lines() {
        let mut words = line.split_whitespace();
        let first = words.next().unwrap_or("");
        if first.contains('-') && first.chars().all(|c| c.is_ascii_hexdigit() || c == '-') {
            ours = line.ends_with(name.as_ref());
        } else if ours && first == "Rss:" {
            kib += words.next().and_then(|n| n.parse::<u64>().ok()).unwrap_or(0);
        }
    }
    kib
}

/// A load told its weights are held (`Residency::weights_held`) seats every
/// atom of every weight shape in pool slots itself, filled from the serving
/// file's mapping, states where (`WeightShape::seated`), and serves from
/// them before any residency: the logits of a residency naming every atom
/// (`all`). A runtime adopting the seating (a held residency naming exactly
/// those slots) fills nothing and keeps what was recorded.
pub fn a_held_load_seats_its_weights(model: &str, fixture: &Fixture, all: &[Vec<f32>]) {
    let Some(mut shell) = load_with(
        fixture,
        PAGE,
        CONTEXT,
        None,
        engine::load::Residency {
            weights_held: true,
            ..engine::load::Residency::default()
        },
    ) else {
        return;
    };
    let loaded = mapped_kib(&fixture.container);
    let facts = shell.pool_chunk_facts();
    facts.validate().expect("the facts hang together");
    let mut resident = Vec::new();
    let mut bytes = 0u64;
    for shape in &facts.weights {
        let first = shape
            .seated
            .unwrap_or_else(|| panic!("{model}: weight shape {} is seated", shape.id));
        let per = shape.slots_per_chunk.max(1);
        let last = (first + shape.count - 1) / per;
        assert!(
            last < facts.segments.committed * facts.segment_chunks,
            "{model}: shape {}'s chunks {}..={last} lie in committed segments",
            shape.id,
            first / per
        );
        bytes += shape.all_bytes();
        resident.extend((0..shape.count).map(|atom| ResidentWeight {
            atom: WeightAtom {
                shape: shape.id,
                atom,
            },
            slot: first + atom,
        }));
    }
    let file = std::fs::metadata(&fixture.container).map_or(0, |m| m.len());
    eprintln!(
        "{model}: a held load seated {} weight shapes, {} MiB; of the {} MiB serving file, {} \
         MiB of its mappings resident after the load",
        facts.weights.len(),
        bytes >> 20,
        file >> 20,
        loaded >> 10
    );
    let early = serve_seats(&mut shell, &fixture.trace, 12);
    same(&format!("{model}: seated by the load against all"), &early, all);
    let uses = shell.weight_uses().expect("the uses read");
    assert!(
        uses.held.iter().all(|h| h.host == 0),
        "{model}: every atom resident: {:?}",
        uses.held
    );
    // The runtime adopts the seating: the first residency names those
    // slots, held.
    shell
        .weight_residency(&WeightResidency {
            staging: Vec::new(),
            resident,
            held: true,
        })
        .expect("the residency lands");
    let adopted = serve_seats(&mut shell, &fixture.trace, 12);
    same(&format!("{model}: adopted against all"), &adopted, all);
    // The seating stands: the facts no longer state it, the runtime's own.
    assert!(
        shell.pool_chunk_facts().weights.iter().all(|w| w.seated.is_none()),
        "{model}: the runtime's residency stands in the load's seating's place"
    );
    eprintln!(
        "{model}: {} MiB of the serving file's mappings resident after serving",
        mapped_kib(&fixture.container) >> 10
    );
}
