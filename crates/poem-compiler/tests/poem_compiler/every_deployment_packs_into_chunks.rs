use std::collections::BTreeMap;

use poem_compiler::catalog::{self, Deployment};
use poem_compiler::chunk::{
    self, ChunkProfile, ChunkRecipe, FIXED_SETS_PER_WS, MIN_PIECE, MIN_RUN, MOST_RS_TOKENS, RowId,
    SlotLayout, TOKEN_GRAIN, human,
};
use poem_compiler::{Budget, Budgets, DeviceProfile};
use poem_ir::{CacheRow, Dtype, Platform, Trace};

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
            SlotLayout::List { tokens, rows, .. } => {
                if tokens % TOKEN_GRAIN != 0 {
                    wrong.push(format!("{tokens} tokens a slot"));
                }
                // Every list, kv or window, holds the kv page.
                if let Some(SlotLayout::List { tokens: page, .. }) = recipe.list()
                    && page != tokens
                {
                    wrong.push(format!(
                        "a list of {tokens} tokens beside the kv page {page}"
                    ));
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
            // The rs list: a power of two of tokens, rows apart and inside
            // the slot; no run need reach MIN_RUN.
            SlotLayout::RsList {
                tokens,
                bytes,
                rows,
            } => {
                if !tokens.is_power_of_two() || *tokens > MOST_RS_TOKENS {
                    wrong.push(format!("{tokens} rs tokens a slot"));
                }
                let mut end = 0;
                for run in rows {
                    if run.offset < end || run.offset % device.align() != 0 {
                        wrong.push(format!("rs run {run:?} overlaps or is misaligned"));
                    }
                    end = run.offset + u64::from(*tokens) * run.per_token;
                }
                if end > *bytes {
                    wrong.push(format!("rs runs end at {end} past the slot of {bytes}"));
                }
                if rows.windows(2).any(|w| w[0].layer > w[1].layer) {
                    wrong.push("rs runs not layer-major".into());
                }
            }
            SlotLayout::Fixed {
                bytes,
                record,
                pieces_per_ws,
                rows,
                ..
            } => {
                if *record > c && *bytes < MIN_PIECE {
                    wrong.push(format!("a piece of {bytes} under the least piece"));
                }
                let mut held: BTreeMap<u32, Vec<(u64, u64)>> = BTreeMap::new();
                for run in rows {
                    let atom = recipe.atoms[&run.row];
                    let pieces = u64::from(run.pieces);
                    // Equal pieces of whole grains, covering the row, none
                    // empty, one a slot.
                    if run.pieces == 0
                        || pieces * run.atoms_per_piece < atom.extent
                        || (pieces - 1) * run.atoms_per_piece >= atom.extent
                        || run.slots.len() != run.pieces as usize
                    {
                        wrong.push(format!("{:?} pieces {run:?} of {atom:?}", run.row));
                    }
                    if run.is_cut() && run.atoms_per_piece % atom.grain != 0 {
                        wrong.push(format!("{:?} cut off its grain: {run:?}", run.row));
                    }
                    if run.is_cut() && atom.extent == 1 {
                        wrong.push(format!("{:?} is uncuttable but cut", run.row));
                    }
                    let end = run.offset + run.piece_bytes(&atom);
                    if end > *bytes {
                        wrong.push(format!(
                            "{:?} lands at {} past its room",
                            run.row, run.offset
                        ));
                    }
                    for &slot in &run.slots {
                        if slot >= *pieces_per_ws {
                            wrong.push(format!("{:?} in slot {slot} past the record", run.row));
                        }
                        held.entry(slot).or_default().push((run.offset, end));
                    }
                }
                for (slot, mut spans) in held {
                    spans.sort_unstable();
                    if spans.windows(2).any(|pair| pair[1].0 < pair[0].1) {
                        wrong.push(format!("slot {slot} holds overlapping rows: {spans:?}"));
                    }
                }
            }
        }
    }
    // Every row the trace remembers lands in exactly one shape.
    let mut held: BTreeMap<RowId, usize> = BTreeMap::new();
    for shape in &recipe.slots {
        match shape {
            SlotLayout::List { rows, .. } | SlotLayout::RsList { rows, .. } => {
                for run in rows {
                    *held.entry(run.row).or_default() += 1;
                }
            }
            SlotLayout::Fixed { rows, .. } => {
                for run in rows {
                    *held.entry(run.row).or_default() += 1;
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
    let fixed = recipe.fixed().map_or(&[][..], |f| match f {
        SlotLayout::Fixed { rows, .. } => rows.as_slice(),
        _ => &[],
    });
    // The fixed record holds state only; every rs row lies once in the rs
    // list, as the recipe's rs layout says, and the recipe says what each
    // buffers, its planes filling its bytes a token.
    if fixed.iter().any(|r| matches!(r.row, RowId::Rs(_))) {
        wrong.push("an rs row in the fixed record".into());
    }
    let layout = &recipe.rs;
    match recipe.rs_list() {
        Some(SlotLayout::RsList {
            tokens,
            bytes,
            rows,
        }) => {
            if (*tokens, *bytes, rows) != (layout.tokens, layout.bytes, &layout.runs) {
                wrong.push(format!("rs list unlike the rs layout {layout:?}"));
            }
        }
        _ if !layout.rows.is_empty() || !layout.is_empty() => {
            wrong.push(format!("rs rows without an rs list: {layout:?}"));
        }
        _ => {}
    }
    if layout.runs.len() != layout.rows.len() {
        wrong.push(format!(
            "{} rs runs, {} described",
            layout.runs.len(),
            layout.rows.len()
        ));
    }
    for run in &layout.runs {
        let what = layout.rows.get(&run.row).map(chunk::RsRow::per_token);
        if what != Some(run.per_token) {
            wrong.push(format!("{:?} buffers {what:?} bytes a token", run.row));
        }
        if !matches!(run.row, RowId::Rs(_)) {
            wrong.push(format!("{:?} in the rs list", run.row));
        }
    }
    for tp in catalog::RANKS {
        let rank = recipe.per_rank(tp);
        if rank.fixed.len() != fixed.len()
            || rank.fixed.iter().zip(fixed).any(|(on, run)| {
                on.row != run.row || on.piece_bytes > run.piece_bytes(&recipe.atoms[&run.row])
            })
        {
            wrong.push(format!("rank of {tp}: fixed runs unlike the recipe's"));
        }
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
        } => u64::from(FIXED_SETS_PER_WS) * u64::from(*pieces_per_ws) * bytes,
        _ => 0,
    });
    #[allow(clippy::cast_precision_loss)]
    let fixed = fixed as f64 / f64::from(1u32 << 20);
    format!(
        "| {name} | {} | {} | {} | {fixed:.1} | {:.1} | {:.1} | {:.2} |",
        human(recipe.chunk_bytes),
        recipe.shapes(),
        per.join("/"),
        recipe.chunks_per_ws(),
        recipe.cow_chunks(),
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
        println!(
            "| model | C | shapes | slots/chunk | fixed MiB/ws | chunks/ws | CoW chunks/pool | waste % |"
        );
        println!("|---|---|---|---|---|---|---|---|");
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

/// The fixed rows of a record whose name holds `part`: how many, of how many
/// bytes.
fn fixed_rows(trace: &Trace, recipe: &ChunkRecipe, part: &str) -> (usize, Vec<u64>) {
    let mut bytes: Vec<u64> = recipe
        .atoms
        .iter()
        .filter(|(row, _)| name_of(trace, **row).contains(part))
        .map(|(_, atom)| atom.bytes * atom.extent)
        .collect();
    bytes.dedup();
    (
        recipe
            .atoms
            .keys()
            .filter(|row| name_of(trace, **row).contains(part))
            .count(),
        bytes,
    )
}

fn per_token(shape: &SlotLayout) -> u64 {
    match shape {
        SlotLayout::List { rows, .. } | SlotLayout::RsList { rows, .. } => {
            rows.iter().map(|r| r.per_token).sum()
        }
        SlotLayout::Fixed { .. } => 0,
    }
}

#[test]
fn qwen3_next_packs_36_delta_and_conv_states_with_next_to_no_waste() {
    let (trace, recipe) = planned("qwen3-next-80b-a3b-bf16-kv-bf16");
    assert_eq!(
        fixed_rows(&trace, &recipe, "delta_state"),
        (36, vec![1 << 20])
    );
    assert_eq!(
        fixed_rows(&trace, &recipe, "conv_state"),
        (36, vec![64 << 10])
    );
    assert_eq!(recipe.rs.rows.len(), 36, "a rs row a gated-delta layer");
    assert!(recipe.atoms.keys().all(|r| !matches!(r, RowId::Rs(_))));
    // What the fixed records take past their bytes, in their pieces and in
    // the chunks those fill.
    let overhead = fixed_overhead(&recipe);
    println!("qwen3-next fixed overhead {:.2}%", 100.0 * overhead);
    assert!(overhead < 0.02, "{overhead}");
}

/// The fixed shape's bytes past the pool's records (a set a working set and
/// the copy-on-write reserve), in its pieces and the chunks they fill, over
/// the records.
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
    let sets = u64::from(recipe.profile.concurrency) * u64::from(FIXED_SETS_PER_WS)
        + u64::from(recipe.cow_sets());
    let per = recipe.chunk_bytes / bytes;
    let slots = sets * u64::from(*pieces_per_ws);
    let held = slots as f64 * recipe.chunk_bytes as f64 / per as f64;
    held / (sets * record) as f64 - 1.0
}

#[test]
fn nemotron_super_holds_40_four_mib_states_and_their_convs() {
    let (trace, recipe) = planned("nemotron3-super-bf16-nvfp4a4-e4m3tile128a8-kv-e4m3");
    assert_eq!(
        fixed_rows(&trace, &recipe, "mamba.ssm_state"),
        (40, vec![4 << 20])
    );
    assert_eq!(
        fixed_rows(&trace, &recipe, "mamba.conv_state"),
        (40, vec![80 << 10])
    );
    let overhead = fixed_overhead(&recipe);
    println!("nemotron-super fixed overhead {:.2}%", 100.0 * overhead);
    assert!(overhead < 0.02, "{overhead}");
}

#[test]
fn gemma4_31b_lists_80_kib_a_token_and_800_kib_through_its_window() {
    let (_, recipe) = planned("gemma4-31b-bf16-kv-bf16");
    let list = recipe.list().unwrap();
    assert_eq!(per_token(list), 80 << 10);
    let windows: Vec<&SlotLayout> = recipe.windows().collect();
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0].window(), Some(1024));
    assert_eq!(per_token(windows[0]), 800 << 10);
    let (SlotLayout::List { tokens: page, .. }, SlotLayout::List { tokens, .. }) =
        (list, windows[0])
    else {
        unreachable!()
    };
    assert_eq!(tokens, page, "the window list's slot is the kv page");
}

#[test]
fn dsv4_pro_lists_its_pools_as_one_shape_and_its_window_as_another() {
    let (_, recipe) = planned("dsv4-pro-bf16-mxfp4-kv-bf16");
    let list = recipe.list().unwrap();
    // 30 pools of 256 B, 30 index keys of 64 B and 31 pools of 1 KiB a token:
    // every compressed space in one list, its slot a multiple of each ratio.
    assert_eq!(per_token(list), 41_344);
    let SlotLayout::List { tokens, .. } = list else {
        unreachable!()
    };
    assert_eq!(tokens % 128, 0);
    let windows: Vec<&SlotLayout> = recipe.windows().collect();
    assert_eq!(windows.len(), 1);
    assert_eq!(per_token(windows[0]), 61 << 10);
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
    // Each window's unfolded ids buffer as an rs row of 16 i32 ids.
    let windows: Vec<_> = recipe
        .rs
        .rows
        .iter()
        .filter(|(_, rs)| {
            rs.state
                .is_some_and(|s| name_of(&trace, RowId::Cache(s)).contains("engram"))
        })
        .collect();
    assert_eq!(windows.len(), 2);
    assert!(windows.iter().all(|(_, rs)| rs.per_token() == 4));
    assert!(*record < 2 << 10, "{record}");
    assert!(fixed.per_chunk(recipe.chunk_bytes) >= 1000);
}

fn planned_on(name: &str, chunk: ChunkProfile) -> (Trace, ChunkRecipe) {
    let deployment = catalog::deployment(name).unwrap_or_else(|| panic!("`{name}` is listed"));
    let trace = deployment.trace(Platform::Cuda);
    let recipe = chunk::plan(&trace, &Budgets::of(Budget::default()), &on(chunk)).unwrap();
    (trace, recipe)
}

/// The rs list's runs.
fn rs_runs(recipe: &ChunkRecipe) -> &[chunk::TokenRun] {
    let Some(SlotLayout::RsList { rows, .. }) = recipe.rs_list() else {
        panic!("no rs list")
    };
    rows
}

#[test]
fn qwen35_vision_mtp_lays_its_rs_rows_in_an_rs_list_layer_major() {
    for (preset, chunk) in PRESETS {
        let (trace, recipe) = planned_on("qwen35-d0.8b-vision-mtp-bf16-kv-bf16", chunk);
        let runs = rs_runs(&recipe);
        // A row a gated-delta layer: 18 of the 24 (every fourth is attention).
        assert_eq!(runs.len(), 18, "{preset}");
        let rs = &recipe.rs;
        println!(
            "{preset}: rs list of {} tokens a slot, {} a slot, {} a token",
            rs.tokens,
            human(rs.bytes),
            human(runs.iter().map(|r| r.per_token).sum())
        );
        assert!(rs.tokens >= 1 && rs.bytes <= recipe.chunk_bytes);
        // The rows lie one after another, layer by layer.
        assert!(runs.windows(2).all(|w| w[0].offset < w[1].offset));
        assert!(runs.windows(2).all(|w| w[0].layer < w[1].layer));
        // The fixed record holds the delta and conv states only.
        let Some(SlotLayout::Fixed { rows, .. }) = recipe.fixed() else {
            panic!("no fixed record")
        };
        assert!(rows.iter().all(|r| matches!(r.row, RowId::Cache(_))));
        // Each row buffers the conv's input and `[b | a]`, committing into
        // its layer's delta state.
        for rs in recipe.rs.rows.values() {
            let state = name_of(&trace, RowId::Cache(rs.state.unwrap()));
            assert!(state.ends_with("gdn.delta_state"), "{state}");
            assert_eq!(rs.planes.len(), 2);
        }
    }
}

#[test]
fn qwen38_flash_next_mini_buffers_its_ngram_ids_as_an_rs_row_beside_the_others() {
    for (preset, chunk) in PRESETS {
        let (trace, recipe) = planned_on("qwen38-flash-next-mini-bf16-kv-bf16", chunk);
        let ids: Vec<_> = recipe
            .rs
            .rows
            .iter()
            .filter(|(_, rs)| {
                rs.state
                    .is_some_and(|s| name_of(&trace, RowId::Cache(s)) == "ple.ids_state")
            })
            .collect();
        assert_eq!(ids.len(), 1, "{preset}");
        let (id, rs) = ids[0];
        // One plane of the unfolded token ids, an i32 each.
        assert_eq!(rs.planes.len(), 1);
        assert_eq!(rs.per_token(), 4);
        let runs = rs_runs(&recipe);
        assert_eq!(runs.len(), 4, "{preset}: three delta layers and the ids");
        assert_eq!(recipe.rs.run(*id).unwrap().per_token, 4);
        println!("{preset}: rs list {}", recipe.shapes());
    }
}

#[test]
fn kimik3_2_8t_buffers_a_token_a_slot_of_its_rs_list() {
    for (preset, chunk) in PRESETS {
        let (_, recipe) = planned_on("kimik3-2.8t-bf16-mxfp4-kv-bf16", chunk);
        let per_token: u64 = rs_runs(&recipe).iter().map(|r| r.per_token).sum();
        // Every KDA layer's conv input, forget and beta: megabytes a token.
        assert!(per_token > 6 << 20, "{preset}: {per_token}");
        println!(
            "{preset}: C {} rs list {} tokens x {}",
            human(recipe.chunk_bytes),
            recipe.rs.tokens,
            human(recipe.rs.bytes)
        );
        assert_eq!(recipe.rs.tokens, 1, "{preset}");
        assert!(recipe.rs.bytes <= recipe.chunk_bytes);
    }
}

#[test]
fn a_kv_u4_deployment_lists_its_heads_packed() {
    let deployment = catalog::deployment("qwen35-d0.8b-bf16-kv-bf16").unwrap();
    let mut trace = deployment.trace(Platform::Cuda);
    let (mut want, mut rows) = (0, 0);
    for row in &mut trace.caches {
        if let CacheRow::Kv {
            dtype,
            planes,
            head_dim,
            ..
        } = row
        {
            *dtype = Dtype::KvU4;
            rows += 1;
            let block = u64::from(*head_dim);
            assert!(block > 0 && planes.iter().all(|w| w % block == 0));
            want += planes
                .iter()
                .map(|w| w / block * (block / 2 + 2))
                .sum::<u64>();
        }
    }
    for (preset, chunk) in PRESETS {
        let recipe = chunk::plan(&trace, &Budgets::of(Budget::default()), &on(chunk))
            .unwrap_or_else(|why| panic!("{preset}: {why}"));
        // A kv row of 2 heads of 256, keys and values: 130 bytes a head
        // where bf16 takes 512.
        assert_eq!(per_token(recipe.list().unwrap()), want, "{preset}");
        assert_eq!(want, rows * 2 * 2 * 130);
        assert!(wrongs(&trace, &recipe).is_empty(), "{preset}");
    }
}
