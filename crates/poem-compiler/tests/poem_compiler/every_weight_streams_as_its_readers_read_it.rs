//! The weights a pool may stream, inferred from the ops that read them, and
//! the weight slots a chunk recipe lays them into.

use std::collections::BTreeSet;

use poem_compiler::catalog::{self, Deployment};
use poem_compiler::chunk::{
    self, ChunkProfile, ChunkRecipe, MIN_PIECE, WeightKind, WeightSlot, human, weights::plane_bytes,
};
use poem_compiler::{Budget, Budgets, DeviceProfile};
use poem_ir::residency::{Residency, leads, residency};
use poem_ir::{ParamPlacement, Platform, Trace};

const PRESETS: [(&str, ChunkProfile); 2] = [
    ("server-cuda", ChunkProfile::SERVER_CUDA),
    ("metal", ChunkProfile::ON_DEVICE_METAL),
];

/// The plane packages marked `host` before the compiler inferred it: Qwen
/// 3.8 Flash Next's PLE n-gram table, in every deployment of its models.
const HAND_MARKED: (&[&str], &str) = (
    &["qwen38-flash-next", "qwen38-flash-next-mini"],
    "ple.table",
);

/// The listed deployments that held a `host` plane when it was declared,
/// on CUDA and on Metal alike.
const MARKED_DEPLOYMENTS: &[&str] = &[
    "qwen38-flash-next-bf16-kv-bf16",
    "qwen38-flash-next-bf16-nvfp4-kv-bf16",
    "qwen38-flash-next-bf16-nvfp4a4-kv-bf16",
    "qwen38-flash-next-mini-bf16-kv-bf16",
    "qwen38-flash-next-mini-bf16-nvfp4-kv-bf16",
    "qwen38-flash-next-mini-bf16-nvfp4a4-kv-bf16",
    "qwen38-flash-next-mini-u4g64-u2g128-kv-bf16",
    "qwen38-flash-next-mtp-u4g64-u2g128-kv-bf16",
    "qwen38-flash-next-u4g64-kv-bf16",
    "qwen38-flash-next-u4g64-u2g128-kv-bf16",
    "qwen38-flash-next-vision-mtp-u4g64-u2g128-kv-bf16",
    "qwen38-flash-next-vision-u4g64-u2g128-kv-bf16",
];

fn on(chunk: ChunkProfile) -> DeviceProfile {
    DeviceProfile {
        chunk,
        ..DeviceProfile::default()
    }
}

/// `rows`' traces on `platform`, as each package records it (before the
/// compiler lowers it), across the host's cores.
fn traces(rows: Vec<&'static Deployment>, platform: Platform) -> Vec<(&'static Deployment, Trace)> {
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let per = rows.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(per)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|row| {
                            let trace = row
                                .package
                                .trace(&row.model.id, &row.deploy, &row.name, platform)
                                .unwrap_or_else(|why| panic!("`{}`: {why:#}", row.name));
                            (*row, trace)
                        })
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

/// Placement is inferred, never declared: every listed deployment,
/// on CUDA and on Metal, places on the host exactly the planes its
/// package once marked, and nothing else. The tables newly found
/// row-streamable keep their device placement; they are printed.
#[test]
fn the_inferred_placement_is_the_one_the_packages_marked() {
    let (models, plane) = HAND_MARKED;
    let rows: Vec<&'static Deployment> = catalog::deployments().collect();
    let listed: BTreeSet<&str> = catalog::deployments().map(|d| d.name.as_str()).collect();
    let mut wrong = Vec::new();
    let mut fresh: BTreeSet<String> = BTreeSet::new();
    for platform in [Platform::Cuda, Platform::Metal] {
        let mut hosting: BTreeSet<&str> = BTreeSet::new();
        for (deployment, trace) in traces(rows.clone(), platform) {
            let classes = residency(&trace);
            for (param, class) in trace.params.iter().zip(&classes) {
                let host = param.placement == ParamPlacement::Host;
                let marked = models.contains(&deployment.model.id.as_str()) && param.name == plane;
                if host && listed.contains(deployment.name.as_str()) {
                    hosting.insert(deployment.name.as_str());
                }
                if host != marked {
                    wrong.push(format!(
                        "{platform:?} {}: `{}` is {:?}",
                        deployment.name, param.name, param.placement
                    ));
                }
                if let Residency::Rows(atom) = class
                    && !host
                    && !param.name.ends_with(poem_ir::SCALES)
                    && !param.name.ends_with(poem_ir::BIASES)
                {
                    // One line a model and table kind: layers' tables as `N`.
                    let name: String = param
                        .name
                        .split('.')
                        .map(|part| {
                            if part.parse::<u32>().is_ok() {
                                "N"
                            } else {
                                part
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(".");
                    fresh.insert(format!(
                        "{} `{name}`{}",
                        deployment.model.id,
                        if atom.hashed { " (hashed)" } else { "" }
                    ));
                }
            }
        }
        let expected: BTreeSet<&str> = MARKED_DEPLOYMENTS.iter().copied().collect();
        if hosting != expected {
            wrong.push(format!("{platform:?}: host planes in {hosting:?}"));
        }
    }
    println!("row-streamable, placed on the device as before:");
    for line in &fresh {
        println!("  {line}");
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A model's weights by class: params and bytes.
#[derive(Default)]
struct Census {
    expert: (usize, u64),
    rows: (usize, u64),
    resident: (usize, u64),
}

fn census(trace: &Trace) -> Census {
    let mut census = Census::default();
    for (param, class) in trace.params.iter().zip(residency(trace)) {
        let bytes = plane_bytes(param);
        let at = match class {
            Residency::Expert(_) => &mut census.expert,
            Residency::Rows(_) => &mut census.rows,
            Residency::Resident => &mut census.resident,
        };
        at.0 += 1;
        at.1 += bytes;
    }
    census
}

/// What is wrong with `trace`'s classes: a streamable plane not cut along
/// its atom axis, a companion streaming apart from its lead, an expert
/// bank of no expert.
fn class_wrongs(trace: &Trace) -> Vec<String> {
    let mut wrong = Vec::new();
    let classes = residency(trace);
    let lead = leads(trace);
    for (q, param) in trace.params.iter().enumerate() {
        let first = param.shape.first().copied();
        match classes[q] {
            Residency::Resident => {}
            Residency::Expert(atom) => {
                if atom.experts == 0 || atom.top_k == 0 || atom.top_k > atom.experts {
                    wrong.push(format!("`{}`: {atom:?}", param.name));
                }
                if first != Some(u64::from(atom.experts)) {
                    wrong.push(format!(
                        "`{}` {:?} is no bank of {}",
                        param.name, param.shape, atom.experts
                    ));
                }
            }
            Residency::Rows(atom) => {
                if first != Some(atom.rows) || atom.ids_per_token == 0 {
                    wrong.push(format!("`{}` {:?}: {atom:?}", param.name, param.shape));
                }
            }
        }
        if lead[q] != q
            && !classes[q].is_resident()
            && std::mem::discriminant(&classes[q]) != std::mem::discriminant(&classes[lead[q]])
        {
            wrong.push(format!("`{}` streams apart from its lead", param.name));
        }
    }
    wrong
}

/// What is wrong with `recipe`'s weight slots.
fn slot_wrongs(trace: &Trace, recipe: &ChunkRecipe) -> Vec<String> {
    let mut wrong = Vec::new();
    let c = recipe.chunk_bytes;
    let align = recipe.profile.align();
    let mut covered = BTreeSet::new();
    for slot in &recipe.weights {
        if slot.bytes > c || slot.bytes == 0 || slot.bytes % align != 0 {
            wrong.push(format!("{} in a chunk of {c}", slot.shape()));
        }
        if slot.floor > slot.slots || slot.floor == 0 {
            wrong.push(format!(
                "{}: floor {} of {}",
                slot.shape(),
                slot.floor,
                slot.slots
            ));
        }
        match slot.kind {
            WeightKind::Expert if slot.pieces > 1 => {
                if !recipe.profile.cut_experts {
                    wrong.push(format!("{}: cut where experts are whole", slot.shape()));
                }
                if slot.bytes < MIN_PIECE {
                    wrong.push(format!("{}: a piece under {MIN_PIECE}", slot.shape()));
                }
                if u64::from(slot.pieces) * slot.bytes < slot.atom_bytes {
                    wrong.push(format!("{}: pieces hold less than an expert", slot.shape()));
                }
            }
            WeightKind::Expert => {
                if slot.bytes < slot.atom_bytes || slot.atoms_per_slot != 1 {
                    wrong.push(format!(
                        "{}: an expert of {}",
                        slot.shape(),
                        slot.atom_bytes
                    ));
                }
            }
            WeightKind::Rows => {
                if slot.bytes < slot.atoms_per_slot * slot.atom_bytes {
                    wrong.push(format!(
                        "{}: {} rows of {}",
                        slot.shape(),
                        slot.atoms_per_slot,
                        slot.atom_bytes
                    ));
                }
            }
        }
        // The slots hold every plane they name, an atom's share each.
        let held: u64 = slot
            .banks
            .iter()
            .flat_map(|b| &b.params)
            .map(|p| plane_bytes(&trace.params[*p as usize]))
            .sum();
        let atoms: u64 = slot.banks.iter().map(|b| b.atoms).sum();
        if slot.total_bytes() < held || atoms * slot.atom_bytes > held + atoms {
            wrong.push(format!(
                "{}: {} held of {held}",
                slot.shape(),
                slot.total_bytes()
            ));
        }
        for p in slot.banks.iter().flat_map(|b| &b.params) {
            if !covered.insert(*p) {
                wrong.push(format!("`{}` in two slots", trace.params[*p as usize].name));
            }
        }
    }
    // Every streamable plane is in a slot.
    for (q, class) in residency(trace).iter().enumerate() {
        if !class.is_resident() && !covered.contains(&(q as u32)) {
            wrong.push(format!("`{}` streams in no slot", trace.params[q].name));
        }
    }
    wrong
}

fn floor(recipe: &ChunkRecipe) -> String {
    format!(
        "{} ({} chunks)",
        human(recipe.weight_floor_bytes()),
        recipe.weight_floor_chunks()
    )
}

/// Every listed deployment's weights classify by the invariants, and its
/// weight slots lay into chunks under both presets: the per-model table of
/// classes and slots is printed.
#[test]
fn every_deployment_classifies_its_weights_and_lays_them_into_chunks() {
    let budgets = Budgets::of(Budget::default());
    let rows: Vec<&'static Deployment> = catalog::deployments().collect();
    // The traces each rank runs, as engines read them: joined, tiled, cut.
    let traced: Vec<(&'static Deployment, Trace)> = traces(rows, Platform::Cuda)
        .into_iter()
        .map(|(d, trace)| {
            let lowered = catalog::lowered(trace, &d.deploy).unwrap_or_else(|why| panic!("{why}"));
            (d, lowered)
        })
        .collect();
    let mut wrong = Vec::new();
    println!(
        "\n| model | expert params | expert bytes | row params | row bytes | resident bytes |"
    );
    println!("|---|---|---|---|---|---|");
    for (deployment, trace) in &traced {
        for why in class_wrongs(trace) {
            wrong.push(format!("{}: {why}", deployment.name));
        }
        let c = census(trace);
        println!(
            "| {} | {} | {} | {} | {} | {} |",
            deployment.name,
            c.expert.0,
            human(c.expert.1),
            c.rows.0,
            human(c.rows.1),
            human(c.resident.1)
        );
    }
    for (preset, profile) in PRESETS {
        let profile = on(profile);
        println!("\n## {preset}\n");
        println!("| model | C | weight shapes | floor | weight chunks | weight tail |");
        println!("|---|---|---|---|---|---|");
        for (deployment, trace) in &traced {
            let recipe = match chunk::plan(trace, &budgets, &profile) {
                Ok(recipe) => recipe,
                Err(why) => {
                    wrong.push(format!("{preset} {}: {why}", deployment.name));
                    continue;
                }
            };
            for why in slot_wrongs(trace, &recipe) {
                wrong.push(format!("{preset} {}: {why}", deployment.name));
            }
            if !recipe.weights.is_empty() {
                println!(
                    "| {} | {} | {} | {} | {} | {} |",
                    deployment.name,
                    human(recipe.chunk_bytes),
                    recipe.weight_shapes(),
                    floor(&recipe),
                    recipe.weight_chunks(),
                    human(recipe.waste.weight_tail)
                );
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

fn planned(name: &str, preset: ChunkProfile) -> (Trace, ChunkRecipe) {
    let deployment = catalog::deployment(name).unwrap_or_else(|| panic!("`{name}` is listed"));
    let trace = deployment.trace(Platform::Cuda);
    let recipe = chunk::plan(&trace, &Budgets::of(Budget::default()), &on(preset)).unwrap();
    (trace, recipe)
}

fn only(recipe: &ChunkRecipe, kind: WeightKind) -> Vec<&WeightSlot> {
    recipe.weights.iter().filter(|w| w.kind == kind).collect()
}

/// Qwen3-30B-A3B: every layer's 128 experts (gate, up and down, 3 x 768 x
/// 2048 bf16: 9 MiB) one shape; the floor is 8 experts of two layers; a
/// server chunk holds whole experts, Metal cuts each on its output rows.
#[test]
fn qwen3_30b_streams_its_9_mib_experts_whole_on_a_server_and_cut_on_metal() {
    let (_, server) = planned("qwen3-30b-a3b-bf16-kv-bf16", ChunkProfile::SERVER_CUDA);
    let experts = only(&server, WeightKind::Expert);
    assert_eq!(experts.len(), 1, "{}", server.weight_shapes());
    let e = experts[0];
    assert_eq!(e.atom_bytes, 3 * 768 * 2048 * 2);
    assert_eq!(e.banks.len(), 48);
    assert!(e.banks.iter().all(|b| b.atoms == 128));
    assert_eq!((e.pieces, e.floor), (1, 16));
    assert_eq!(e.slots, 48 * 128);
    assert!(server.chunk_bytes >= e.bytes);
    println!(
        "server: C {} {} tail {}",
        human(server.chunk_bytes),
        server.weight_shapes(),
        human(server.waste.weight_tail)
    );

    let (_, metal) = planned("qwen3-30b-a3b-bf16-kv-bf16", ChunkProfile::ON_DEVICE_METAL);
    let e = only(&metal, WeightKind::Expert)[0];
    assert!(e.pieces > 1 && e.bytes <= metal.chunk_bytes && e.bytes >= MIN_PIECE);
    assert!(u64::from(e.pieces) * e.bytes >= e.atom_bytes);
    assert_eq!(e.floor, 16 * u64::from(e.pieces));
    println!(
        "metal: C {} {}",
        human(metal.chunk_bytes),
        metal.weight_shapes()
    );
}

/// GPT-OSS-20B: its MXFP4 experts stream with their scales and biases, one
/// shape, every plane of a layer's banks in an expert's slot; the router
/// and attention stay resident.
#[test]
fn gpt_oss_streams_its_mxfp4_experts_with_their_scales_and_biases() {
    let (trace, recipe) = planned("gptoss-20b-bf16-mxfp4-kv-bf16", ChunkProfile::SERVER_CUDA);
    let experts = only(&recipe, WeightKind::Expert);
    assert_eq!(experts.len(), 1, "{}", recipe.weight_shapes());
    let e = experts[0];
    assert_eq!(e.banks.len(), 24);
    assert!(e.banks.iter().all(|b| b.atoms == 32));
    assert_eq!(e.floor, 2 * 4);
    let names: BTreeSet<&str> = e.banks[0]
        .params
        .iter()
        .map(|p| trace.params[*p as usize].name.as_str())
        .collect();
    assert!(
        names.iter().any(|n| n.ends_with(poem_ir::SCALES)),
        "{names:?}"
    );
    let held: u64 = e.banks[0]
        .params
        .iter()
        .map(|p| plane_bytes(&trace.params[*p as usize]))
        .sum();
    assert_eq!(e.atom_bytes * 32, held);
    for (param, class) in trace.params.iter().zip(residency(&trace)) {
        if param.name.contains("router") || param.name.contains("attn") {
            assert!(class.is_resident(), "`{}` is {class:?}", param.name);
        }
    }
    println!("gpt-oss: {}", recipe.weight_shapes());
}

/// Gemma 4 E4B: its per-layer embeddings (one table a layer, read by the
/// token's id) gather alike, so a token id's rows of every one are one
/// atom; the floor is a fire's ids.
#[test]
fn gemma4_e4b_gathers_a_token_ids_rows_of_every_per_layer_table_as_one_atom() {
    let (trace, recipe) = planned("gemma4-e4b-bf16-kv-bf16", ChunkProfile::SERVER_CUDA);
    assert!(only(&recipe, WeightKind::Expert).is_empty());
    let per_layer = |p: &u32| trace.params[*p as usize].name.starts_with("ple.per_layer.");
    let ple = only(&recipe, WeightKind::Rows)
        .into_iter()
        .find(|w| w.banks[0].params.iter().any(per_layer))
        .expect("the per-layer tables stream");
    let tables = ple.banks[0].params.iter().filter(|p| per_layer(p)).count();
    assert_eq!(tables, 42, "{}", recipe.weight_shapes());
    assert_eq!(ple.atoms_per_slot, 1);
    assert_eq!(ple.floor, u64::from(Budget::default().max_tokens));
    println!(
        "gemma4-e4b: {} atom {}",
        recipe.weight_shapes(),
        human(ple.atom_bytes)
    );
}

/// Qwen 3.8 Flash Next: the PLE n-gram table is hashed, 16 ids a token, the
/// one plane placed on the host; it streams as row blocks whose floor is a
/// fire's 16 ids a token.
#[test]
fn qwen38_streams_its_hashed_ple_table_in_row_blocks() {
    let (trace, recipe) = planned("qwen38-flash-next-u4g64-kv-bf16", ChunkProfile::SERVER_CUDA);
    let at = trace
        .params
        .iter()
        .position(|p| p.name == "ple.table")
        .expect("a PLE table");
    assert_eq!(trace.params[at].placement, ParamPlacement::Host);
    let Residency::Rows(atom) = residency(&trace)[at] else {
        panic!("the table is row-streamable")
    };
    assert!(atom.hashed && atom.layer.is_none());
    assert_eq!(atom.ids_per_token, 16);
    let table = only(&recipe, WeightKind::Rows)
        .into_iter()
        .find(|w| w.banks[0].params.contains(&(at as u32)))
        .expect("the table has a slot");
    assert_eq!(table.floor, 16 * u64::from(Budget::default().max_tokens));
    assert!(table.bytes <= 4 << 10);
    println!("qwen38: {}", recipe.weight_shapes());
}
