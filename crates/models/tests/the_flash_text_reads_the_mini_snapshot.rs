//! **THE FLASH TEXT'S NAMES, HELD AGAINST THE MINI SNAPSHOT** (2-bit campaign
//! wave 4b).
//!
//! The `mlx-community/DeepSeek-V4-Flash-2bit-DQ` mini snapshot (`mini-l5-e16`:
//! the five renumbered layers 0,1,2,3,42, sixteen experts) is the name/shape
//! oracle this family's flash text was carved against. A bf16 structure text
//! cannot READ its 2-bit/4-bit planes — the weights are packed `U32` triplets —
//! so this is not an import bijection but a NAME one: every tensor the snapshot
//! survives with maps to a plane the flash arm states and reads, and every
//! plane the flash arm reads is one the snapshot holds.
//!
//! A quantized plane is stored as an `X.weight`/`X.scales`/`X.biases` triplet;
//! the bf16 arm reads the `X.weight` view, so the snapshot's logical census is
//! its `weight_map` keys with the `.scales`/`.biases` companions dropped. Two
//! stored planes fuse into one bank (`switch_mlp.{gate,up}_proj`, the shared
//! pair), so the arm READS both names — the census is over names read, not plan
//! params, which is exactly "every surviving name maps to a plan tensor".
//!
//! **AND WHY IT SKIPS RATHER THAN FAILS.** The snapshot is a fact about a
//! machine that pulled it; a build box that has not is "not asked", said out
//! loud, not a red run.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use models::deepseek_v4::flash_mlxu2;
use models::deepseek_v4::model::{GateUp, Mlp};
use model_dsl::Dtype;

const REPO: &str = "models--mlx-community--DeepSeek-V4-Flash-2bit-DQ";

/// The snapshot directory itself, for the readers that want more than the
/// index's key set.
fn snapshot_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let snapshots = Path::new(&home)
        .join(".cache/huggingface/hub")
        .join(REPO)
        .join("snapshots");
    let mut found = None;
    for entry in std::fs::read_dir(snapshots).ok()? {
        let dir: PathBuf = entry.ok()?.path();
        if dir.join("model.safetensors.index.json").is_file() {
            found = Some(dir);
        }
    }
    found
}

fn index_names() -> Option<BTreeSet<String>> {
    let home = std::env::var_os("HOME")?;
    let snapshots = Path::new(&home)
        .join(".cache/huggingface/hub")
        .join(REPO)
        .join("snapshots");
    let mut found: Option<BTreeSet<String>> = None;
    for entry in std::fs::read_dir(snapshots).ok()? {
        let dir: PathBuf = entry.ok()?.path();
        let index = dir.join("model.safetensors.index.json");
        let Ok(text) = std::fs::read_to_string(&index) else {
            continue;
        };
        let parsed: serde_json::Value = serde_json::from_str(&text).ok()?;
        let map = parsed.get("weight_map")?.as_object()?;
        // The logical census: the `.weight`/bare views, the `.scales`/`.biases`
        // companions of a quantized triplet dropped.
        let names = map
            .keys()
            .filter(|name| !name.ends_with(".scales") && !name.ends_with(".biases"))
            .cloned()
            .collect();
        found = Some(names);
    }
    found
}

#[test]
fn the_flash_arm_reads_exactly_the_mini_snapshot() {
    let Some(census) = index_names() else {
        eprintln!(
            "not asked: no {REPO} snapshot index under \
             $HOME/.cache/huggingface/hub"
        );
        return;
    };

    let model =
        models::deepseek_v4::model::Model::flash_micro(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1);
    let read: BTreeSet<String> = model.mlx_source_names().into_iter().collect();

    let mut faults = Vec::new();
    for name in census.symmetric_difference(&read) {
        let side = if census.contains(name) {
            "the mini snapshot names it and the flash arm never reads it"
        } else {
            "the flash arm reads it and the mini snapshot does not name it"
        };
        faults.push(format!("`{name}`: {side}"));
    }
    assert!(
        faults.is_empty(),
        "the flash text and the mini-l5-e16 census disagree on {} name(s):\n{}\n",
        faults.len(),
        faults.join("\n"),
    );
}

/// The flash forward RECORDS — every organ (MLA, the compressor pool, the NSA
/// indexer, the two gate kinds, the shared expert, the o-group reduce, the
/// hyper mix) threads a shape the recorder accepts, and every plane it declares
/// is interned as a checkpoint param. Traced on both lowering targets so
/// neither shell's trace panics; the fire each still owes is a fire, not a
/// trace.
#[test]
fn the_flash_forward_traces_on_both_shells() {
    use model_dsl::{ParamSource, Platform};

    let trace = models::trace_of("dsv4-flash-bf16-kv-bf16").expect("the catalog ships the row");
    for platform in [Platform::Cuda, Platform::Metal] {
        let t = trace(platform);
        let checkpoint = t
            .params
            .iter()
            .filter(|p| p.source == ParamSource::Checkpoint)
            .count();
        assert!(
            checkpoint > 0,
            "the flash trace on {platform:?} interns no checkpoint params"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The AFFINE census (2-bit campaign wave 5).
// ─────────────────────────────────────────────────────────────────────────

/// The `(bits, group)` a `Dtype` row reports through the load contract — the
/// one place the dtype's name becomes the two integers a plane is read by.
fn point_of(dtype: Dtype) -> (u32, u32) {
    match checkpoint_dsl::encoding(dtype) {
        checkpoint::types::Encoding::Quant(spec) => (u32::from(spec.bits_per_element), spec.group_size),
        other => panic!("`{dtype:?}` is not a quantized plane: {other:?}"),
    }
}

/// The `Dtype` row that names one MLX affine `(bits, group)`, or `None` for a
/// pair this tree cannot spell.
fn row_for(bits: u64, group: u64) -> Option<Dtype> {
    Some(match (bits, group) {
        (4, 64) => Dtype::U4g64,
        (8, 64) => Dtype::U8g64,
        (4, 32) => Dtype::U4g32,
        (2, 32) => Dtype::U2g32,
        (2, 64) => Dtype::U2g64,
        (2, 128) => Dtype::U2g128,
        _ => return None,
    })
}

/// One safetensors header, as `name -> (dtype, shape)`.
fn header(path: &Path) -> Option<serde_json::Map<String, serde_json::Value>> {
    let bytes = std::fs::read(path).ok()?;
    let len = u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?) as usize;
    let parsed: serde_json::Value = serde_json::from_slice(bytes.get(8..8 + len)?).ok()?;
    parsed.as_object().cloned()
}

/// **EVERY `(bits, group)` THE 2-BIT SNAPSHOT SPENDS IS A ROW THIS TREE CAN
/// SPELL, AND THE STORED RECTANGLES AGREE WITH IT.**
///
/// A DQ conversion states its quantization PER TENSOR: `config.json` carries a
/// default pair and then seventy-one overrides, and the whole reason the
/// `MlxU2G*` rows exist is that a text which could only say "two bits" would
/// read half of them at the wrong scales length. So this holds the census three
/// ways — the config's own pairs, the `Dtype` row each maps to, and the
/// `.scales` rectangle the file actually ships — and a disagreement anywhere is
/// a plane that would decode around the wrong centre with no NaN to notice it
/// by.
///
/// **AND IT PINS THE LANDMINE.** The exception is not a family rule and not a
/// tidy prefix: the routed `gate_proj` is grouped by 32 on the first four
/// layers and by 64 on the LAST — original layer 42, renumbered to 4 — while
/// `up_proj` and `down_proj` are grouped by 64 throughout. That one layer is
/// the difference between a text that reads this artifact and a text that reads
/// four fifths of it.
#[test]
fn the_two_bit_snapshot_spends_only_pairs_this_tree_can_spell() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("not asked: no {REPO} snapshot under $HOME/.cache/huggingface/hub");
        return;
    };
    let text = std::fs::read_to_string(dir.join("config.json")).expect("the snapshot's config");
    let config: serde_json::Value = serde_json::from_str(&text).expect("config parses");
    let quant = config
        .get("quantization")
        .and_then(|q| q.as_object())
        .expect("a converted MLX snapshot states its quantization");

    let pair = |v: &serde_json::Value| -> (u64, u64) {
        (
            v.get("bits").and_then(serde_json::Value::as_u64).expect("bits"),
            v.get("group_size")
                .and_then(serde_json::Value::as_u64)
                .expect("group_size"),
        )
    };
    let default = pair(&serde_json::Value::Object(quant.clone()));
    assert_eq!(default, (4, 64), "the DQ default is MLX's own");

    let head = header(&dir.join("model-00001-of-00001.safetensors"));
    let mut seen: BTreeSet<(u64, u64)> = BTreeSet::new();
    seen.insert(default);
    let mut overrides = 0usize;
    let mut gate_groups: Vec<(usize, u64)> = Vec::new();

    for (name, value) in quant {
        let Some(spec) = value.as_object() else {
            continue; // the three scalar keys
        };
        overrides += 1;
        let (bits, group) = pair(&serde_json::Value::Object(spec.clone()));
        seen.insert((bits, group));

        let row = row_for(bits, group).unwrap_or_else(|| {
            panic!("`{name}` is stored ({bits}, {group}), which no `Dtype` row spells")
        });
        assert_eq!(
            point_of(row),
            (bits as u32, group as u32),
            "`{name}`: the row `{row:?}` reports a different point than the file states",
        );

        // The stored `.scales` rectangle is the group count, so its last extent
        // is the row width over the group — the arithmetic `affine_planes`
        // will do, checked here against the bytes rather than against itself.
        if let Some(head) = &head {
            let codes = head
                .get(&format!("{name}.weight"))
                .and_then(|t| t.get("shape"))
                .and_then(serde_json::Value::as_array)
                .unwrap_or_else(|| panic!("`{name}.weight` is in the header"));
            let scales = head
                .get(&format!("{name}.scales"))
                .and_then(|t| t.get("shape"))
                .and_then(serde_json::Value::as_array)
                .unwrap_or_else(|| panic!("`{name}.scales` is in the header"));
            // The codes are `u32` words of `32 / bits` codes each.
            let words = codes.last().and_then(serde_json::Value::as_u64).expect("width");
            let groups = scales.last().and_then(serde_json::Value::as_u64).expect("groups");
            assert_eq!(
                words * (32 / bits),
                groups * group,
                "`{name}`: {words} words of {} codes is not {groups} groups of {group}",
                32 / bits,
            );
        }

        if let Some(rest) = name.strip_prefix("model.layers.") {
            if let Some((l, tail)) = rest.split_once('.') {
                if tail == "ffn.switch_mlp.gate_proj" {
                    gate_groups.push((l.parse().expect("a layer number"), group));
                }
            }
        }
    }

    assert_eq!(overrides, 71, "the mini snapshot states seventy-one overrides");
    assert_eq!(
        seen,
        BTreeSet::from([(4, 64), (2, 64), (2, 32)]),
        "the DQ mix is three pairs and no more",
    );

    // The landmine, stated as the census sees it.
    gate_groups.sort_unstable();
    assert_eq!(
        gate_groups,
        vec![(0, 32), (1, 32), (2, 32), (3, 32), (4, 64)],
        "the routed gate_proj groups by 32 on layers 0-3 and by 64 on the LAST \
         (original layer 42, renumbered to 4) — a PER-LAYER exception",
    );
}

// ─────────────────────────────────────────────────────────────────────────
// The UNFUSED expert pair (2-bit campaign wave 6).
// ─────────────────────────────────────────────────────────────────────────

/// **WHAT THE FUSED BANK COULD NOT STATE, AND WHAT THE SPLIT PAIR DOES.**
///
/// This was a red assert. `Mlp::MoeFlash` declared ONE `experts_gate_up` bank
/// and the import read it as a concat of the stored `gate_proj` and `up_proj`
/// (`import.rs`'s `Read::Concat`); a `Weight` carries ONE `dtype`, and `Dtype`
/// is where an MLX affine group is written down, so a fused bank had ONE group
/// for both halves and `checkpoint_dsl::affine_planes` sized one `.scales`
/// rectangle from it. On this artifact the halves DISAGREE on four of five
/// layers — 2048 rows of 128 groups beside 2048 rows of 64 — and those join
/// into no rectangle at any axis. The fused declaration was not awkward here;
/// it was unstateable.
///
/// It is stateable now, and this test is both halves of that sentence: the
/// file's disagreement (unchanged, and still the reason) and the text's answer
/// to it — [`Model::flash_mini`] under [`Routed::DQ_2BIT`] declares
/// `experts_gate` and `experts_up` as SEPARATE banks, each carrying the point
/// its own stored triplet was written at, layer by layer.
#[test]
fn the_split_expert_pair_states_the_two_groups_a_fused_bank_could_not() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("not asked: no {REPO} snapshot under $HOME/.cache/huggingface/hub");
        return;
    };
    let Some(head) = header(&dir.join("model-00001-of-00001.safetensors")) else {
        eprintln!("not asked: the snapshot's shard is not readable here");
        return;
    };
    let groups = |name: &str| -> u64 {
        head.get(name)
            .and_then(|t| t.get("shape"))
            .and_then(serde_json::Value::as_array)
            .and_then(|s| s.last())
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_else(|| panic!("`{name}` is in the header"))
    };

    // Half one: the file. The `.scales` rectangles of the two halves still
    // disagree, which is still why the pair is unfused.
    let mut disagreeing = Vec::new();
    for l in 0..5 {
        let gate = groups(&format!("model.layers.{l}.ffn.switch_mlp.gate_proj.scales"));
        let up = groups(&format!("model.layers.{l}.ffn.switch_mlp.up_proj.scales"));
        if gate != up {
            disagreeing.push((l, gate, up));
        }
    }
    assert_eq!(
        disagreeing,
        vec![(0, 128, 64), (1, 128, 64), (2, 128, 64), (3, 128, 64)],
        "the gate and up halves disagree on their scales rectangle for layers \
         0-3, which is what a fused `experts_gate_up` bank cannot state",
    );

    // Half two: the text. Each half is its own bank, and each bank's declared
    // point is the one the file wrote its half at — the landmine layer
    // included, where gate and up finally agree and are STILL two banks,
    // because the form is the SKU's and not the layer's.
    let model = flash_mlxu2(1);
    let mut declared = Vec::new();
    for (l, layer) in model.layers.iter().enumerate() {
        let Mlp::MoeFlash { gate_up, down, .. } = &layer.mlp else {
            panic!("layer {l} of a flash text is a MoeFlash");
        };
        let GateUp::Split { gate, up } = gate_up else {
            panic!(
                "layer {l} of the 2-bit row declares a FUSED expert bank, which is \
                 the declaration this artifact has no rectangle for"
            );
        };
        declared.push((l, point_of(gate.dtype), point_of(up.dtype), point_of(down.dtype)));
    }
    assert_eq!(
        declared,
        vec![
            (0, (2, 32), (2, 64), (2, 64)),
            (1, (2, 32), (2, 64), (2, 64)),
            (2, (2, 32), (2, 64), (2, 64)),
            (3, (2, 32), (2, 64), (2, 64)),
            // The landmine: original layer 42, renumbered to 4.
            (4, (2, 64), (2, 64), (2, 64)),
        ],
        "the split pair's declared `(bits, group)` per layer, gate then up then down",
    );
}

/// **AND THE DECLARED POINTS ARE THE FILE'S, TENSOR BY TENSOR.**
///
/// The test above pins the routed block against a table typed in this file;
/// this one pins the WHOLE text against the artifact's own bytes. For every
/// weight the flash arm reads out of a stored triplet, the `(bits, group)` its
/// declared `Dtype` reports must equal the `(bits, group)` the file's codes and
/// `.scales` rectangles actually spend — and every plane the text declares
/// dense must be one the file ships dense.
///
/// This is the census that would have caught a text reading four fifths of the
/// artifact: a 2-bit plane read at group 64 where the file wrote 32
/// dequantizes around the wrong centre at the right spread, finite and
/// deterministic and wrong, with no NaN to notice it by.
#[test]
fn every_plane_the_two_bit_row_declares_is_the_point_the_file_wrote() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("not asked: no {REPO} snapshot under $HOME/.cache/huggingface/hub");
        return;
    };
    let Some(head) = header(&dir.join("model-00001-of-00001.safetensors")) else {
        eprintln!("not asked: the snapshot's shard is not readable here");
        return;
    };

    let model = flash_mlxu2(1);
    let mut faults = Vec::new();
    let mut quantized = 0usize;
    let mut dense = 0usize;

    for (weight, name) in model.mlx_planes() {
        let declared = matches!(
            weight.dtype,
            Dtype::U4g64
                | Dtype::U8g64
                | Dtype::U4g32
                | Dtype::U2g32
                | Dtype::U2g64
                | Dtype::U2g128
        );
        // A quantized plane is a `.weight`/`.scales`/`.biases` triplet; a name
        // the arm reads bare (the hash table, `attn_sink`, the hyper planes,
        // `compressor.ape`) has neither companion beside it.
        let stem = name.strip_suffix(".weight").unwrap_or(name.as_str());
        let stored = head.contains_key(&format!("{stem}.scales"));
        if declared != stored {
            faults.push(format!(
                "`{stem}`: the text declares it {:?} and the file ships it {}",
                weight.dtype,
                if stored { "as a triplet" } else { "dense" },
            ));
            continue;
        }
        if !declared {
            dense += 1;
            continue;
        }
        quantized += 1;
        // The stored point, read off the bytes: the codes are u32 words of
        // `32 / bits` codes, and the `.scales` last extent is the group count.
        let last = |suffix: &str| -> u64 {
            head.get(&format!("{stem}{suffix}"))
                .and_then(|t| t.get("shape"))
                .and_then(serde_json::Value::as_array)
                .and_then(|s| s.last())
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_else(|| panic!("`{stem}{suffix}` is in the header"))
        };
        let (bits, group) = point_of(weight.dtype);
        let words = last(".weight");
        let groups = last(".scales");
        if words * u64::from(32 / bits) != groups * u64::from(group) {
            faults.push(format!(
                "`{stem}`: declared ({bits}, {group}) reads {} codes as {} groups, and \
                 the file ships {groups}",
                words * u64::from(32 / bits),
                words * u64::from(32 / bits) / u64::from(group),
            ));
        }
    }

    assert!(
        faults.is_empty(),
        "the 2-bit text and the artifact disagree on {} plane(s):\n{}\n",
        faults.len(),
        faults.join("\n"),
    );
    assert!(
        quantized > 0 && dense > 0,
        "the census read {quantized} quantized and {dense} dense planes, and a \
         DQ artifact is both"
    );
    eprintln!("the 2-bit census: {quantized} triplets and {dense} dense planes agree");
}
