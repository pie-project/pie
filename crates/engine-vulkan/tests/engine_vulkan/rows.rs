//! The weight cache's row tables (per-layer embeddings): the tables stream
//! by blocks of rows through weight slots of the pool the runtime names
//! (`Engine::weight_residency`). A gather copies rows, so every residency
//! lands the same bits: every block resident and held (the deployment that
//! does not stream), none (each row staged from the host), every other one.
//! Where `$PIE_VK_BASE/seats-<model>.bin` holds the logits a build before
//! the cache landed (`$PIE_VK_LOGITS` of `a_windowed_miniature_serves_two_lanes`),
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
    if let Some(dir) = std::env::var_os("PIE_VK_BASE") {
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

/// The miss path's cost (not a check; `--ignored`, a release build): decode
/// throughput of `lanes` lanes over `steps` fires with every block resident
/// (no fire flushes or stages) against every other block and none
/// (`$PIE_PLE_MODEL`, default gemma4-e4b-mini-l6; `$PIE_PLE_LANES`, 2).
#[test]
#[ignore = "a measurement: run it alone, in release"]
fn the_miss_path_costs_this_much_decode_throughput() {
    use crate::model_logits::{PAGE, argmax, load_with, word};
    use engine_vulkan::Lane;
    use engine_vulkan::serve::Seated;
    let model = std::env::var("PIE_PLE_MODEL").unwrap_or_else(|_| "gemma4-e4b-mini-l6".into());
    let lanes: usize = std::env::var("PIE_PLE_LANES").map_or(2, |n| n.parse().expect("a count"));
    const STEPS: usize = 192;
    let fixture = fixture(&model, None);
    let rounds: usize = std::env::var("PIE_PLE_ROUNDS").map_or(3, |n| n.parse().expect("a count"));
    let order = [Held::All, Held::Half, Held::None];
    let mut best = [f64::MAX; 3];
    for (at, held) in order.iter().copied().cycle().take(3 * rounds).enumerate() {
        let Some(mut shell) = load_with(
            &fixture,
            PAGE,
            512,
            None,
            engine::load::Residency::default(),
        ) else {
            return;
        };
        let facts = shell.pool_chunk_facts();
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
        shell
            .commit_segments((next + 1).div_ceil(segment))
            .expect("the segments commit");
        shell
            .weight_residency(&WeightResidency {
                staging: Vec::new(),
                resident,
                held: held == Held::All,
            })
            .expect("the residency lands");
        let trace = &fixture.trace;
        let mut fed: Vec<Vec<u32>> = (0..lanes)
            .map(|l| {
                (0..64u32)
                    .map(|at| (at * 37 + 11 + l as u32 * 7) % 256)
                    .collect()
            })
            .collect();
        let mut have = vec![0u32; lanes];
        for lane in 0..lanes {
            let got = shell
                .fire_seated(&[Seated {
                    held: Some(have[lane]),
                    ..Seated::of(Lane {
                        slot: lane as u32,
                        word: word(trace, fed[lane].len() as u32),
                        tokens: &fed[lane],
                    })
                }])
                .expect("the prefill lands");
            have[lane] += fed[lane].len() as u32;
            fed[lane] = vec![argmax(&got[0])];
        }
        let fire =
            |shell: &mut engine_vulkan::Shell, fed: &mut Vec<Vec<u32>>, have: &mut Vec<u32>| {
                let seated: Vec<Seated<'_>> = (0..lanes)
                    .map(|lane| Seated {
                        held: Some(have[lane]),
                        ..Seated::of(Lane {
                            slot: lane as u32,
                            word: word(trace, 1),
                            tokens: &fed[lane],
                        })
                    })
                    .collect();
                let got = shell.fire_seated(&seated).expect("the decode lands");
                for lane in 0..lanes {
                    have[lane] += 1;
                    fed[lane] = vec![argmax(&got[lane])];
                }
            };
        for _ in 0..8 {
            fire(&mut shell, &mut fed, &mut have);
        }
        let started = std::time::Instant::now();
        for _ in 0..STEPS {
            fire(&mut shell, &mut fed, &mut have);
        }
        let took = started.elapsed();
        best[at % 3] = best[at % 3].min(took.as_secs_f64());
        eprintln!(
            "{model}: {held:?} resident, {lanes} lanes: {STEPS} decode fires in {took:?}: \
             {:.1} tok/s, {:.2} ms a fire",
            (STEPS * lanes) as f64 / took.as_secs_f64(),
            took.as_secs_f64() * 1e3 / STEPS as f64
        );
    }
    for (held, took) in order.iter().zip(best) {
        eprintln!(
            "{model}: {held:?} resident, {lanes} lanes, best of {rounds}: {:.1} tok/s, {:.3} ms a fire",
            (STEPS * lanes) as f64 / took,
            took * 1e3 / STEPS as f64
        );
    }
}
