//! **THE WIDTH-INVARIANCE FIXTURES, HELD AGAINST THE TEXT THAT READS THEM.**
//!
//! `benches/shrink_checkpoint.py --family qwen3_5_moe_mlx` carves
//! `mini-l5-e16-k8` and `mini-l5-e64-k8` out of
//! `mlx-community/Qwen3.6-35B-A3B-4bit`, and the CUDA node gates its tiled
//! `Linear::Matmul` and `LmHead` on the result. This file is the local half of
//! that agreement: it says what the artifacts must BE, so that a carve which
//! drifts is caught here rather than as a gate that quietly stops testing
//! anything.
//!
//! Four claims, held over EVERY carve in [`CARVES`], and the third is the one
//! the fixtures exist for:
//!
//! 1. the artifact lands on its own row — `qwen36-35b-a3b-mini-mlxu4-kv-bf16`
//!    or `qwen36-35b-a3b-mini64-mlxu4-kv-bf16` — and that row's plan and its
//!    import contract name the same planes;
//! 2. every stored quantised triplet's codes, scales and biases agree with the
//!    `(bits, group)` its own `config.json` declares for it — the file is
//!    self-consistent, and the ten 8-bit router overrides survived the
//!    renumbering that dropped thirty-five layers;
//! 3. **THE CARVE MOVED DEPTH AND THE EXPERT BANK AND NOTHING ELSE.** The
//!    fault under test is accumulation ORDER over K. `hidden`, the expert
//!    `moe_intermediate_size`, the whole attention block that computes the
//!    router's input, and `vocab_size` — the other tiled entry, and where the
//!    delta is finally read — all keep production width, because halving any
//!    of them halves the partial sums and can hide the thing the gate is for;
//! 4. the top-k tail is CONTESTED: 8 of 16 or 8 of 64, not 8 of 8 (not a
//!    choice) and not 8 of 256 (gaps wider than a ulp can cross).
//!
//! **AND THE SECOND CARVE IS WHY THIS FILE IS A LOOP RATHER THAN A COPY.** The
//! two artifacts are ONE geometry apart from the routed bank's leading
//! dimension, and the CUDA gate reads them against each other: the 16-expert
//! run reproduced the tiled path's fault structure at a routed delta of 0.078
//! and yet crossed no expert boundary, so 64 exists to contract the
//! eighth-to-ninth logit gap until a ulp has somewhere to cross. A difference
//! between the two runs is only readable as the bank's crowding if nothing else
//! differs, so every claim below is asserted identically over both — including
//! the plane-count histograms, which do NOT move with the bank, because the
//! carve changes shapes and never names.
//!
//! A box without a snapshot skips that carve loudly and never fails — each
//! artifact is reproduced by one command, printed in `shrink_checkpoint.py`'s
//! help.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use model_dsl::{ParamSource, Platform};

const REPO: &str = "models--mlx-community--Qwen3.6-35B-A3B-4bit";

/// One carve of the shipped artifact, and the row that declares it.
struct Carve {
    /// The snapshot directory's name, which states the geometry it holds:
    /// `mini-l{layers}-e{experts}-k{top_k}`.
    snapshot: &'static str,
    /// The catalog row this carve must land on, and only this one.
    sku: &'static str,
    /// The routed bank's leading dimension — the ONE number the carves
    /// disagree on, and therefore the only thing parametrised below.
    experts: u64,
}

/// **BOTH CARVES, HELD TO ONE STANDARD.** Every claim in this file runs over
/// this list, so a fixture cannot be added to the fleet with less rigour than
/// the one it was added to sharpen. `--experts` is the only flag that differs
/// in the two commands that build them.
const CARVES: &[Carve] = &[
    Carve {
        snapshot: "mini-l5-e16-k8",
        sku: "qwen36-35b-a3b-mini-mlxu4-kv-bf16",
        experts: 16,
    },
    Carve {
        snapshot: "mini-l5-e64-k8",
        sku: "qwen36-35b-a3b-mini64-mlxu4-kv-bf16",
        experts: 64,
    },
];

/// The row the FULL artifact still lands on, which no miniature may steal.
const FULL_SKU: &str = "qwen36-35b-a3b-mlxu4-kv-bf16";

fn hub() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let hub = Path::new(&home).join(".cache/huggingface/hub");
    hub.is_dir().then_some(hub)
}

fn carve_dir(carve: &Carve) -> Option<PathBuf> {
    let dir = hub()?.join(REPO).join("snapshots").join(carve.snapshot);
    dir.join("model.safetensors.index.json")
        .is_file()
        .then_some(dir)
}

/// The full snapshot: the one that is none of the carves. Matched by the
/// `mini-` prefix rather than against [`CARVES`] by name, so that a carve added
/// to the fleet and not yet listed still cannot be mistaken for the shipped
/// artifact and quietly turn the no-theft claim into a tautology.
fn full_dir() -> Option<PathBuf> {
    let snapshots = hub()?.join(REPO).join("snapshots");
    std::fs::read_dir(snapshots).ok()?.find_map(|entry| {
        let dir = entry.ok()?.path();
        let name = dir.file_name()?.to_str()?.to_string();
        (!name.starts_with("mini-") && dir.join("config.json").is_file()).then_some(dir)
    })
}

fn shards(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("the snapshot lists")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            name.ends_with(".safetensors").then_some(path)
        })
        .collect();
    files.sort();
    files
}

fn config(dir: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(dir.join("config.json")).expect("the config reads");
    serde_json::from_str(&text).expect("the config parses")
}

/// One tensor's stored `(safetensors dtype, shape)`, over every shard.
type Header = BTreeMap<String, (String, Vec<u64>)>;

fn headers(dir: &Path) -> Header {
    let mut all = Header::new();
    for shard in shards(dir) {
        let mut file = std::fs::File::open(&shard).expect("the shard opens");
        let mut len = [0u8; 8];
        std::io::Read::read_exact(&mut file, &mut len).expect("the length prefix reads");
        let mut buf = vec![0u8; u64::from_le_bytes(len) as usize];
        std::io::Read::read_exact(&mut file, &mut buf).expect("the header reads");
        let parsed: serde_json::Value = serde_json::from_slice(&buf).expect("the header parses");
        for (name, value) in parsed.as_object().expect("the header is an object") {
            let Some(dtype) = value.get("dtype").and_then(serde_json::Value::as_str) else {
                continue; // `__metadata__`
            };
            let shape = value
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .expect("a stored tensor states a shape")
                .iter()
                .filter_map(serde_json::Value::as_u64)
                .collect();
            all.insert(name.clone(), (dtype.to_string(), shape));
        }
    }
    all
}

/// **EACH MINIATURE LANDS ON ITS OWN ROW, AND THAT ROW'S PLAN AND CONTRACT
/// NAME THE SAME PLANES.**
///
/// The bijection is the qwen36 census's, one artifact over: a plan demands
/// checkpoint params, an import contract supplies them, and a name in one and
/// not the other is a plane that is either computed from nothing or read for
/// nobody.
///
/// **AND `identify` LANDING PER CARVE IS ALSO THE NO-THEFT CLAIM BETWEEN
/// THEM.** The two rows are one number apart, so the risk is not that a carve
/// matches nothing but that it matches its sibling — silently, since a bank of
/// sixty-four read as sixteen just never routes to forty-eight of its experts.
/// Asserting the landing for both carves closes that in both directions at
/// once, which is why this is a loop and not a per-row test.
#[test]
fn the_a3b_mini_rows_cover_the_carves_they_were_written_for() {
    let mut faults = Vec::new();
    let mut asked = 0usize;

    for carve in CARVES {
        let (snapshot, sku) = (carve.snapshot, carve.sku);
        let Some(dir) = carve_dir(carve) else {
            eprintln!(
                "not asked: no {REPO}/snapshots/{snapshot} on this machine \
                 (build it with benches/shrink_checkpoint.py)"
            );
            continue;
        };
        asked += 1;
        let source =
            ztensor_compat::index_all(&shards(&dir)).expect("the shards open as one source");

        let import = models::import_of(sku).expect("this build ships the row");
        match import(&source) {
            Ok(contract) => {
                let trace = models::trace_of(sku).expect("and its trace");
                let trace = trace(Platform::Cuda);
                let demand: BTreeSet<&str> = trace
                    .params
                    .iter()
                    .filter(|p| p.source == ParamSource::Checkpoint)
                    .map(|p| p.name.as_str())
                    .collect();
                let supply: BTreeSet<&str> = contract
                    .tensors
                    .iter()
                    .filter(|t| t.visibility == checkpoint::contract::Visibility::Public)
                    .map(|t| t.name.as_str())
                    .collect();
                for name in demand.symmetric_difference(&supply) {
                    faults.push(format!(
                        "`{sku}`: `{name}` is in one of the plan and its import \
                         contract and not the other"
                    ));
                }
            }
            Err(why) => faults.push(format!("`{sku}` refuses the carve it names: {why}")),
        }

        match models::identify(&source) {
            Ok(row) if row == sku => {}
            Ok(row) => faults.push(format!("`{snapshot}` identifies as `{row}`, not `{sku}`")),
            Err(why) => faults.push(format!("`{snapshot}` matches no SKU: {why}")),
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
    if asked == 0 {
        eprintln!("not asked: no A3B miniature on this machine");
    }
}

/// **AND NO MINIATURE STEALS THE SHIPPED ARTIFACT'S ROW.** The three rows read
/// one architecture and separate on shape: the full file has forty layers where
/// both mini rows declare five, and its routed banks lead with 256 experts
/// where they declare sixteen and sixty-four. If that ever stopped being true
/// the catalog order would be the only thing holding them apart, which is
/// exactly the ambiguity `qwen_3::IMPORTS`' ordering note refuses to rely on.
#[test]
fn the_full_artifact_keeps_its_own_row() {
    let Some(dir) = full_dir() else {
        eprintln!("not asked: no full {REPO} snapshot on this machine");
        return;
    };
    let source = ztensor_compat::index_all(&shards(&dir)).expect("the shards open as one source");
    match models::identify(&source) {
        Ok(row) if row == FULL_SKU => {}
        Ok(row) => panic!("the full A3B artifact identifies as `{row}`, not `{FULL_SKU}`"),
        Err(why) => panic!("the full A3B artifact matches no SKU: {why}"),
    }
}

/// **EVERY TRIPLET IS THE WIDTH ITS OWN CONFIG CLAIMS FOR IT.**
///
/// A 4-bit affine plane stores its codes packed 8-to-a-`u32` and one scale per
/// group, so `last(weight) * 32 / bits == last(scales) * group` is the file
/// saying its own recipe back. This is what catches a mis-sliced expert bank:
/// a dim-0 prefix that cut codes and scales at different rows lands here.
///
/// It is also where the ten 8-bit router overrides are held. `mlp.gate` and
/// `mlp.shared_expert_gate` depart from the file's 4-bit default, their keys
/// name SOURCE layers, and the carve dropped thirty-five layers and renumbered
/// the rest — so the overrides had to be rebuilt onto the surviving names or
/// every router in the artifact would silently fall back to a width it is not
/// stored at.
#[test]
fn every_stored_plane_spends_the_point_its_config_declares() {
    let mut asked = 0usize;
    for carve in CARVES {
        let Some(dir) = carve_dir(carve) else {
            eprintln!(
                "not asked: no {REPO}/snapshots/{} on this machine",
                carve.snapshot
            );
            continue;
        };
        asked += 1;
        every_stored_plane_of(carve, &dir);
    }
    if asked == 0 {
        eprintln!("not asked: no A3B miniature on this machine");
    }
}

/// One carve's per-plane quantisation, checked. **THE PINNED HISTOGRAM DOES NOT
/// TAKE THE BANK AS A PARAMETER, AND THAT IS THE POINT**: `--experts` slices
/// leading dimensions and never adds or drops a name, so both carves write the
/// same 227 tensors and land the same 56/10 split. A count that moved with the
/// bank would mean the carve changed the artifact's SHAPE of file, not its
/// widths, and the two gates would no longer be comparable.
fn every_stored_plane_of(carve: &Carve, dir: &Path) {
    let snapshot = carve.snapshot;
    let cfg = config(dir);
    let quant = cfg
        .get("quantization")
        .expect("the carve states a quantization");
    let default_bits = quant
        .get("bits")
        .and_then(serde_json::Value::as_u64)
        .expect("bits");
    let default_group = quant
        .get("group_size")
        .and_then(serde_json::Value::as_u64)
        .expect("group_size");
    assert_eq!(
        (default_bits, default_group),
        (4, 64),
        "the A3B conversion is 4-bit affine at group 64"
    );

    // The per-module overrides, by the module name they key on.
    let overrides: BTreeMap<&str, (u64, u64)> = quant
        .as_object()
        .expect("the quantization is an object")
        .iter()
        .filter_map(|(k, v)| {
            let spec = v.as_object()?;
            Some((
                k.as_str(),
                (
                    spec.get("bits")?.as_u64()?,
                    spec.get("group_size")?.as_u64()?,
                ),
            ))
        })
        .collect();

    let head = headers(dir);
    let mut faults = Vec::new();
    let mut points: BTreeMap<(u64, u64), usize> = BTreeMap::new();
    let mut dense = 0usize;

    for (name, (dtype, shape)) in &head {
        let Some(stem) = name.strip_suffix(".weight") else {
            continue; // `.scales`/`.biases` are checked beside their codes
        };
        let Some((scales_dtype, scales_shape)) = head.get(&format!("{stem}.scales")) else {
            dense += 1;
            continue;
        };
        let (bits, group) = overrides
            .get(stem)
            .copied()
            .unwrap_or((default_bits, default_group));
        *points.entry((bits, group)).or_default() += 1;

        if dtype != "U32" {
            faults.push(format!(
                "`{name}` is a quantised plane stored {dtype}, not U32"
            ));
            continue;
        }
        if scales_dtype != "BF16" {
            faults.push(format!(
                "`{stem}.scales` is stored {scales_dtype}, not BF16"
            ));
        }
        match head.get(&format!("{stem}.biases")) {
            Some((_, biases_shape)) if biases_shape == scales_shape => {}
            Some((_, biases_shape)) => faults.push(format!(
                "`{stem}`: biases {biases_shape:?} and scales {scales_shape:?} \
                 do not agree, so a group's zero point and its scale describe \
                 different rows"
            )),
            None => faults.push(format!("`{stem}` is affine-quantised and ships no biases")),
        }
        // The leading axes must agree exactly: for a routed bank that is the
        // expert axis, and it is the axis the carve sliced.
        if shape[..shape.len() - 1] != scales_shape[..scales_shape.len() - 1] {
            faults.push(format!(
                "`{stem}`: codes {shape:?} and scales {scales_shape:?} disagree \
                 above the last axis, so the slice cut them at different rows"
            ));
            continue;
        }
        let codes = shape.last().expect("a rank") * (32 / bits);
        let covered = scales_shape.last().expect("a rank") * group;
        if codes != covered {
            faults.push(format!(
                "`{stem}`: {codes} codes at {bits} bits and {covered} columns \
                 covered by {} groups of {group}",
                scales_shape.last().expect("a rank")
            ));
        }
    }

    assert!(faults.is_empty(), "\n{snapshot}:\n{}\n", faults.join("\n"));

    // **THE HISTOGRAM, PINNED.** Ten 8-bit modules — one `mlp.gate` and one
    // `mlp.shared_expert_gate` per layer, five layers — and fifty-six planes
    // at the file's 4-bit default:
    //
    // * four `linear_attention` layers x 11 (five `linear_attn` projections,
    //   three `switch_mlp` banks, three `shared_expert` projections) = 44;
    // * the one `full_attention` layer x 10 (four `self_attn` projections and
    //   the same six MLP planes) = 10;
    // * `embed_tokens` and `lm_head`, both affine triplets and untied = 2.
    //
    // The vocabulary pair is the load-bearing entry: `lm_head` is the second
    // tiled entry the width-invariance gate rides on, and it is here at full
    // width.
    assert_eq!(
        points,
        BTreeMap::from([((4, 64), 56), ((8, 64), 10)]),
        "`{snapshot}`'s per-plane quantisation points"
    );
    assert_eq!(
        overrides.len(),
        10,
        "`{snapshot}`: five layers, each with a router and a shared-expert gate, \
         carried onto the renumbered names"
    );
    for layer in 0..5 {
        for leaf in ["mlp.gate", "mlp.shared_expert_gate"] {
            let key = format!("language_model.model.layers.{layer}.{leaf}");
            assert_eq!(
                overrides.get(key.as_str()),
                Some(&(8, 64)),
                "`{snapshot}`: `{key}` is an 8-bit module and the carve must have \
                 said so"
            );
        }
    }
    assert!(dense > 0, "`{snapshot}` also ships dense planes");
}

/// **THE CARVE MOVED DEPTH AND THE BANK AND NOTHING ELSE.**
///
/// This is the fixture's whole substance, so it is asserted as a DIFFERENCE
/// against the shipped artifact's own config rather than as a list of numbers
/// that could drift into agreement with a wrong carve. Exactly three keys may
/// move, and each one is depth or the bank saying it moved.
///
/// **AND THE SAME THREE KEYS MOVE FOR BOTH CARVES**, which is the claim that
/// makes the pair a controlled comparison: they differ from the shipped
/// artifact along one axis each, and from each other along only `num_experts`.
/// If the 64 carve moved a fourth key the CUDA node's two runs would differ by
/// that key as much as by the bank, and the gate would be reading a confound.
#[test]
fn the_carves_shrank_only_the_depth_and_the_expert_bank() {
    let Some(full) = full_dir() else {
        eprintln!("not asked: the full {REPO} snapshot is needed");
        return;
    };
    let mut asked = 0usize;
    for carve in CARVES {
        let Some(dir) = carve_dir(carve) else {
            eprintln!(
                "not asked: no {REPO}/snapshots/{} on this machine",
                carve.snapshot
            );
            continue;
        };
        asked += 1;
        the_carve_shrank_only(carve, &dir, &full);
    }
    if asked == 0 {
        eprintln!("not asked: no A3B miniature on this machine");
    }
}

fn the_carve_shrank_only(carve: &Carve, dir: &Path, full: &Path) {
    let snapshot = carve.snapshot;
    let mini = config(dir);
    let full = config(full);
    let mini_tc = mini
        .get("text_config")
        .expect("the carve wraps a text_config");
    let full_tc = full
        .get("text_config")
        .expect("the source wraps a text_config");

    let moved: BTreeSet<&str> = mini_tc
        .as_object()
        .expect("an object")
        .iter()
        .chain(full_tc.as_object().expect("an object").iter())
        .filter(|(k, _)| mini_tc.get(k.as_str()) != full_tc.get(k.as_str()))
        .map(|(k, _)| k.as_str())
        .collect();

    assert_eq!(
        moved,
        BTreeSet::from(["num_hidden_layers", "num_experts", "layer_types"]),
        "`{snapshot}`: only depth, the routed bank, and the per-layer attention \
         list the depth truncates may differ from the shipped artifact's \
         text_config"
    );

    // And what they moved TO. Depth is the same five for every carve; the bank
    // is the one number the fleet is parametrised on, and the snapshot's own
    // name is where it is read from.
    let n = |v: &serde_json::Value, k: &str| v.get(k).and_then(serde_json::Value::as_u64);
    assert_eq!(n(mini_tc, "num_hidden_layers"), Some(5));
    assert_eq!(
        n(mini_tc, "num_experts"),
        Some(carve.experts),
        "`{snapshot}` must hold the bank its name and its row both state"
    );

    // **THE CONTESTED TAIL.** `num_experts_per_tok` is in neither the moved
    // set nor a number either carve chose: it is the SOURCE's own 8, kept, and
    // it is what makes the fleet a fleet — 8 held fixed while the bank moves is
    // the only way the tail's crowding is the independent variable. Eight of
    // sixteen crowds the routing decision; eight of sixty-four crowds it
    // further, contracting the eighth-to-ninth logit gap until a ulp moved by a
    // tiled projection can actually change which expert a token takes. Eight of
    // eight is not a choice at all.
    assert_eq!(n(mini_tc, "num_experts_per_tok"), Some(8));
    assert_eq!(n(full_tc, "num_experts_per_tok"), Some(8));
    assert!(
        n(mini_tc, "num_experts") > n(mini_tc, "num_experts_per_tok"),
        "`{snapshot}`: a top-k equal to the bank routes to every expert and \
         contests nothing"
    );

    // **THE CONTRACTION IS UNTOUCHED**, which the difference above already
    // proves; these restate the three widths the gate actually rides on, so a
    // reader of this file does not have to go and look them up.
    assert_eq!(n(mini_tc, "hidden_size"), Some(2048));
    assert_eq!(n(mini_tc, "moe_intermediate_size"), Some(512));
    assert_eq!(n(mini_tc, "vocab_size"), Some(248_320));

    // `layer_types` moved only by truncation, and it still says what
    // `full_attention_interval` derives — five layers is 1.25 periods of a
    // period-4 pattern, and the two declarations agree anyway.
    let interval = n(mini_tc, "full_attention_interval").expect("an interval");
    let types: Vec<&str> = mini_tc
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .expect("a layer_types list")
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    assert_eq!(types.len(), 5);
    for (i, kind) in types.iter().enumerate() {
        let want = if (i as u64 + 1) % interval == 0 {
            "full_attention"
        } else {
            "linear_attention"
        };
        assert_eq!(
            *kind, want,
            "`{snapshot}`: layer {i}'s type and full_attention_interval \
             {interval} disagree"
        );
    }
    assert!(
        types.contains(&"full_attention") && types.contains(&"linear_attention"),
        "`{snapshot}` must keep both attention kinds, and both cache kinds with \
         them"
    );
}

/// **THE EXPERT AXIS IS WHAT THE CONFIG SAYS ON EVERY ROUTED PLANE, AND THE
/// ROUTER AGREES WITH THE BANK IT SELECTS FROM.**
///
/// The config states a bank; this asks the bytes. A router still leading with
/// 256 rows over a bank of 16 — or a bank sliced to 16 under a config that says
/// 64 — is the failure the slice exists to avoid, and it is silent: the extra
/// rows just never win.
#[test]
fn every_routed_plane_leads_with_the_experts_its_config_states() {
    let mut asked = 0usize;
    for carve in CARVES {
        let Some(dir) = carve_dir(carve) else {
            eprintln!(
                "not asked: no {REPO}/snapshots/{} on this machine",
                carve.snapshot
            );
            continue;
        };
        asked += 1;
        every_routed_plane_of(carve, &dir);
    }
    if asked == 0 {
        eprintln!("not asked: no A3B miniature on this machine");
    }
}

fn every_routed_plane_of(carve: &Carve, dir: &Path) {
    let (snapshot, experts) = (carve.snapshot, carve.experts);
    let head = headers(dir);
    let mut faults = Vec::new();
    let mut seen = 0usize;

    for layer in 0..5 {
        let n = |leaf: &str| format!("language_model.model.layers.{layer}.{leaf}");
        for leaf in [
            "mlp.gate",
            "mlp.switch_mlp.gate_proj",
            "mlp.switch_mlp.up_proj",
            "mlp.switch_mlp.down_proj",
        ] {
            for plane in ["weight", "scales", "biases"] {
                let name = n(&format!("{leaf}.{plane}"));
                let Some((_, shape)) = head.get(&name) else {
                    faults.push(format!("`{name}` is missing from the carve"));
                    continue;
                };
                seen += 1;
                if shape.first() != Some(&experts) {
                    faults.push(format!(
                        "`{name}` leads with {:?}, not the {experts} experts the \
                         config declares",
                        shape.first()
                    ));
                }
            }
        }
        // The fused `experts_gate_up` bank the text declares is only stateable
        // if the two stored halves are the same rectangle.
        let gate = head.get(&n("mlp.switch_mlp.gate_proj.weight"));
        let up = head.get(&n("mlp.switch_mlp.up_proj.weight"));
        if gate.map(|(_, s)| s) != up.map(|(_, s)| s) {
            faults.push(format!(
                "layer {layer}: gate_proj and up_proj are different rectangles, \
                 so they cannot join into one bank"
            ));
        }
        // The shared expert runs for every token and is ONE expert: its
        // leading axis is a row space, and slicing it would have halved the
        // tensor instead of selecting experts.
        for leaf in ["mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj"] {
            let name = n(&format!("{leaf}.weight"));
            if let Some((_, shape)) = head.get(&name)
                && shape.first() == Some(&experts)
            {
                faults.push(format!(
                    "`{name}` leads with {experts} — the shared expert is not a \
                     bank and must not have been sliced with one"
                ));
            }
        }
    }

    assert!(faults.is_empty(), "\n{snapshot}:\n{}\n", faults.join("\n"));
    assert_eq!(
        seen, 60,
        "`{snapshot}`: five layers x four routed modules x three planes"
    );
}
