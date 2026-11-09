use std::collections::BTreeMap;

use poem_compiler::catalog::{self, Deployment};
use poem_compiler::chunk::{
    self, ChunkProfile, ChunkRecipe, FIXED_SETS, MIN_PIECE, MIN_RUN, RowId, SlotLayout,
    TOKEN_GRAIN, human,
};
use poem_compiler::{Budget, Budgets, DeviceProfile};
use poem_ir::{CacheRow, Platform, Trace};

const PRESETS: [(&str, ChunkProfile); 2] = [
    ("server-cuda", ChunkProfile::SERVER_CUDA),
    ("metal", ChunkProfile::ON_DEVICE_METAL),
];

fn on(chunk: ChunkProfile) -> DeviceProfile {
    DeviceProfile {
        chunk,
        ..DeviceProfile::default()
    }
}

/// Every listed deployment's trace, traced across the host's cores.
fn traces() -> Vec<(&'static Deployment, Trace)> {
    let rows: Vec<&'static Deployment> = catalog::deployments().collect();
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let per = rows.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(per)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|row| (*row, row.trace(Platform::Cuda)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a tracing thread"))
            .collect()
    })
}

/// What is wrong with `recipe`, by the invariants every recipe keeps.
fn wrongs(trace: &Trace, recipe: &ChunkRecipe) -> Vec<String> {
    let mut wrong = Vec::new();
    let c = recipe.chunk_bytes;
    let device = recipe.profile;
    if c > device.max_binding_bytes || c > device.segment_bytes {
        wrong.push(format!("C {c} past a binding or a segment"));
    }
    if !recipe.short_runs.is_empty() {
        wrong.push(format!("layers run short: {:?}", recipe.short_runs));
    }
    if !recipe.waste.percent().is_finite() {
        wrong.push("waste unreported".into());
    }
    for shape in &recipe.slots {
        if shape.bytes() > c {
            wrong.push(format!("a slot of {} past C {c}", shape.bytes()));
        }
        match shape {
            SlotLayout::List { tokens, rows, .. } | SlotLayout::Ring { tokens, rows, .. } => {
                if tokens % TOKEN_GRAIN != 0 {
                    wrong.push(format!("{tokens} tokens a slot"));
                }
                let mut layers: BTreeMap<(Option<u32>, Option<RowId>), u64> = BTreeMap::new();
                for run in rows {
                    let key = (run.layer, run.layer.is_none().then_some(run.row));
                    *layers.entry(key).or_default() += run.per_token;
                }
                for (layer, per_token) in layers {
                    if u64::from(*tokens) * per_token < MIN_RUN {
                        wrong.push(format!(
                            "layer {layer:?} runs {}",
                            u64::from(*tokens) * per_token
                        ));
                    }
                }
            }
            SlotLayout::Fixed {
                bytes,
                record,
                pieces_per_ws,
                pieces,
                ..
            } => {
                if *record > c && *bytes < MIN_PIECE {
                    wrong.push(format!("a piece of {bytes} under the least piece"));
                }
                let mut covered: BTreeMap<RowId, u64> = BTreeMap::new();
                let mut ends: BTreeMap<u32, u64> = BTreeMap::new();
                for piece in pieces {
                    let atom = recipe.atoms[&piece.row];
                    let at = covered.entry(piece.row).or_default();
                    if piece.span.start != *at || piece.span.end > atom.extent {
                        wrong.push(format!("{:?} spans {:?} after {at}", piece.row, piece.span));
                    }
                    *at = piece.span.end;
                    let cut = piece.span != (0..atom.extent);
                    if cut
                        && (piece.span.start % atom.grain != 0
                            || (piece.span.end != atom.extent && piece.span.end % atom.grain != 0))
                    {
                        wrong.push(format!(
                            "{:?} cut off its grain: {:?}",
                            piece.row, piece.span
                        ));
                    }
                    if cut && atom.extent == 1 {
                        wrong.push(format!("{:?} is uncuttable but cut", piece.row));
                    }
                    let end = piece.offset + (piece.span.end - piece.span.start) * atom.bytes;
                    let last = ends.entry(piece.piece).or_default();
                    if piece.offset < *last || end > *bytes || piece.piece >= *pieces_per_ws {
                        wrong.push(format!(
                            "{:?} lands at {} past its room",
                            piece.row, piece.offset
                        ));
                    }
                    *last = end;
                }
                for (row, at) in covered {
                    if at != recipe.atoms[&row].extent {
                        wrong.push(format!("{row:?} covered to {at}"));
                    }
                }
            }
        }
    }
    // Every row the trace remembers lands in exactly one shape.
    let mut held: BTreeMap<RowId, usize> = BTreeMap::new();
    for shape in &recipe.slots {
        match shape {
            SlotLayout::List { rows, .. } | SlotLayout::Ring { rows, .. } => {
                for run in rows {
                    *held.entry(run.row).or_default() += 1;
                }
            }
            SlotLayout::Fixed { pieces, .. } => {
                let rows: std::collections::BTreeSet<RowId> =
                    pieces.iter().map(|p| p.row).collect();
                for row in rows {
                    *held.entry(row).or_default() += 1;
                }
            }
        }
    }
    for at in 0..trace.caches.len() {
        let id = RowId::Cache(u32::try_from(at).unwrap());
        if held.get(&id) != Some(&1) {
            wrong.push(format!("{id:?} held {:?} times", held.get(&id)));
        }
    }
    for tp in catalog::RANKS {
        let rank = recipe.per_rank(tp);
        for ((bytes, per), shape) in rank.slots.iter().zip(&recipe.slots) {
            if bytes * per > rank.chunk_bytes || *per != shape.per_chunk(c) {
                wrong.push(format!(
                    "rank of {tp}: {per} slots of {bytes} in {}",
                    rank.chunk_bytes
                ));
            }
        }
    }
    wrong
}

fn row(name: &str, recipe: &ChunkRecipe) -> String {
    let per: Vec<String> = recipe
        .slots
        .iter()
        .map(|s| s.per_chunk(recipe.chunk_bytes).to_string())
        .collect();
    let fixed = recipe.fixed().map_or(0, |f| match f {
        SlotLayout::Fixed {
            bytes,
            pieces_per_ws,
            ..
        } => u64::from(FIXED_SETS) * u64::from(*pieces_per_ws) * bytes,
        _ => 0,
    });
    #[allow(clippy::cast_precision_loss)]
    let fixed = fixed as f64 / f64::from(1u32 << 20);
    format!(
        "| {name} | {} | {} | {} | {fixed:.1} | {:.1} | {:.2} |",
        human(recipe.chunk_bytes),
        recipe.shapes(),
        per.join("/"),
        recipe.chunks_per_ws(),
        recipe.waste.percent()
    )
}

/// Every listed deployment packs under both presets: no slot past C, a
/// layer's run no shorter than 32 KiB, pieces cut only on their atoms and
/// covering each row once, every cache row in one shape, every rank of a
/// cut holding as many chunks; the table and the worst waste are printed.
#[test]
fn every_deployment_packs_into_chunks_on_both_presets() {
    let budgets = Budgets::of(Budget::default());
    let traces = traces();
    let mut wrong = Vec::new();
    for (preset, chunk) in PRESETS {
        let profile = on(chunk);
        let mut worst: Option<(f64, String)> = None;
        println!("\n## {preset}\n");
        println!("| model | C | shapes | slots/chunk | fixed MiB/ws | chunks/ws | waste % |");
        println!("|---|---|---|---|---|---|---|");
        for (deployment, trace) in &traces {
            let recipe = match chunk::plan(trace, &budgets, &profile) {
                Ok(recipe) => recipe,
                Err(why) => {
                    wrong.push(format!("{preset} {}: {why}", deployment.name));
                    continue;
                }
            };
            for why in wrongs(trace, &recipe) {
                wrong.push(format!("{preset} {}: {why}", deployment.name));
            }
            println!("{}", row(&deployment.name, &recipe));
            let waste = recipe.waste.percent();
            if worst.as_ref().is_none_or(|(w, _)| waste > *w) {
                worst = Some((waste, deployment.name.clone()));
            }
        }
        if let Some((waste, name)) = worst {
            println!("\n{preset} worst waste: {waste:.2}% ({name})");
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

fn planned(name: &str) -> (Trace, ChunkRecipe) {
    let deployment = catalog::deployment(name).unwrap_or_else(|| panic!("`{name}` is listed"));
    let trace = deployment.trace(Platform::Cuda);
    let recipe = chunk::plan(
        &trace,
        &Budgets::of(Budget::default()),
        &on(ChunkProfile::SERVER_CUDA),
    )
    .unwrap();
    (trace, recipe)
}

fn name_of(trace: &Trace, row: RowId) -> &str {
    match row {
        RowId::Cache(at) => match &trace.caches[at as usize] {
            CacheRow::Kv { name, .. } | CacheRow::State { name, .. } => name,
        },
        RowId::Rs(_) => "rs",
    }
}

/// The fixed rows of a record by name prefix: how many, of how many bytes.
fn fixed_rows(trace: &Trace, recipe: &ChunkRecipe, prefix: &str) -> (usize, Vec<u64>) {
    let mut bytes: Vec<u64> = recipe
        .atoms
        .iter()
        .filter(|(row, _)| name_of(trace, **row).starts_with(prefix))
        .map(|(_, atom)| atom.bytes * atom.extent)
        .collect();
    bytes.dedup();
    (
        recipe
            .atoms
            .keys()
            .filter(|row| name_of(trace, **row).starts_with(prefix))
            .count(),
        bytes,
    )
}

fn per_token(shape: &SlotLayout) -> u64 {
    match shape {
        SlotLayout::List { rows, .. } | SlotLayout::Ring { rows, .. } => {
            rows.iter().map(|r| r.per_token).sum()
        }
        SlotLayout::Fixed { .. } => 0,
    }
}

#[test]
fn qwen3_next_packs_36_delta_and_conv_states_with_next_to_no_waste() {
    let (trace, recipe) = planned("qwen3-next-80b-a3b-bf16-kv-bf16");
    assert_eq!(fixed_rows(&trace, &recipe, "delta."), (36, vec![1 << 20]));
    assert_eq!(fixed_rows(&trace, &recipe, "conv."), (36, vec![64 << 10]));
    let rs = recipe
        .atoms
        .keys()
        .filter(|r| matches!(r, RowId::Rs(_)))
        .count();
    assert_eq!(rs, 36, "a rs buffer a gated-delta layer");
    // What the fixed records take past their bytes, in their pieces and in
    // the chunks those fill: the rs rows (258 KiB a layer) keep a slot from
    // dividing a chunk of list slots exactly, so a little stays.
    let overhead = fixed_overhead(&recipe);
    println!("qwen3-next fixed overhead {:.2}%", 100.0 * overhead);
    assert!(overhead < 0.02, "{overhead}");
}

/// The fixed shape's bytes past both records, in its pieces and its share of
/// the chunks they fill, over the records.
#[allow(clippy::cast_precision_loss)]
fn fixed_overhead(recipe: &ChunkRecipe) -> f64 {
    let Some(SlotLayout::Fixed {
        bytes,
        record,
        pieces_per_ws,
        ..
    }) = recipe.fixed()
    else {
        panic!("no fixed record")
    };
    let sets = u64::from(FIXED_SETS);
    let per = recipe.chunk_bytes / bytes;
    let slots = sets * u64::from(*pieces_per_ws);
    let held = slots as f64 * recipe.chunk_bytes as f64 / per as f64;
    held / (sets * record) as f64 - 1.0
}

#[test]
fn nemotron_super_holds_40_four_mib_states_and_their_convs() {
    let (trace, recipe) = planned("nemotron3-super-bf16-nvfp4a4-e4m3tile128a8-kv-e4m3");
    assert_eq!(fixed_rows(&trace, &recipe, "ssm."), (40, vec![4 << 20]));
    assert_eq!(fixed_rows(&trace, &recipe, "conv."), (40, vec![80 << 10]));
    let overhead = fixed_overhead(&recipe);
    println!("nemotron-super fixed overhead {:.2}%", 100.0 * overhead);
    assert!(overhead < 0.02, "{overhead}");
}

#[test]
fn gemma4_31b_lists_80_kib_a_token_and_rings_800_kib_through_its_window() {
    let (_, recipe) = planned("gemma4-31b-bf16-kv-bf16");
    assert_eq!(per_token(recipe.list().unwrap()), 80 << 10);
    let rings: Vec<&SlotLayout> = recipe.rings().collect();
    assert_eq!(rings.len(), 1);
    let SlotLayout::Ring { cap, .. } = rings[0] else {
        unreachable!()
    };
    assert_eq!(cap.window, 1024);
    assert_eq!(per_token(rings[0]), 800 << 10);
}

#[test]
fn dsv4_pro_lists_its_pools_as_one_shape_and_rings_its_window() {
    let (_, recipe) = planned("dsv4-pro-bf16-mxfp4-kv-bf16");
    let list = recipe.list().unwrap();
    // 30 pools of 256 B, 30 index keys of 64 B and 31 pools of 1 KiB a token:
    // every compressed space in one list, its slot a multiple of each ratio.
    assert_eq!(per_token(list), 41_344);
    let SlotLayout::List { tokens, .. } = list else {
        unreachable!()
    };
    assert_eq!(tokens % 128, 0);
    let rings: Vec<&SlotLayout> = recipe.rings().collect();
    assert_eq!(rings.len(), 1);
    assert_eq!(per_token(rings[0]), 61 << 10);
}

#[test]
fn dsv41_flash_keeps_its_replay_floor_and_ngram_ids_in_a_slot_a_thousand_share() {
    let (trace, recipe) = planned("dsv41-flash-fp8b32a8g32-mxfp4a8g32-kv-nvfp4");
    let fixed = recipe.fixed().unwrap();
    let SlotLayout::Fixed { record, .. } = fixed else {
        unreachable!()
    };
    assert_eq!(fixed_rows(&trace, &recipe, "replay").0, 1);
    assert_eq!(fixed_rows(&trace, &recipe, "engram").0, 2);
    assert!(*record < 1 << 10, "{record}");
    assert!(fixed.per_chunk(recipe.chunk_bytes) >= 1000);
}
