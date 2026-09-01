//! **THE QWEN4 TEXT'S NAMES AND WIDTHS, HELD AGAINST THE 2-BIT MINIATURE**
//! (2-bit campaign, the qwen half).
//!
//! `Sawfwair/Qwen3.8-Flash-Next-MLX-Mixed-2bit`'s `mini-l4-e16-p8` snapshot is
//! the real `Qwen3.8-Flash-Next` geometry — hidden 2560, heads 24/2 at 256, the
//! GatedDeltaNet at 16/48 of 128, `moe_intermediate_size` 640, 248 320 tokens —
//! over FOUR layers (`linear, linear, linear, full`), SIXTEEN routed experts
//! and EIGHT n-gram shards. [`Model::flash_mini`] is that geometry and
//! [`Mix::MIXED_2BIT`] is what the file stores each role at.
//!
//! **AND THE MIX IS THE POINT.** The derivation this text used before —
//! `proj = U8g64`, `narrow_group = U4g32`, both off one `w` — was written
//! against the mixed-4/8 conversion, whose exceptions all point UP from a
//! four-bit default. This conversion's point DOWN: a default of `(4, 64)`, the
//! routed experts at `(2, 128)`, the n-gram shards at `(4, 32)`, and no 8-bit
//! entry anywhere — its embedding and head are bare bf16 planes. One `w` can
//! express one of those artifacts. `Mix` expresses both, and this file is
//! where the claim meets the bytes.
//!
//! Every assert here reads the snapshot's own header rather than a table typed
//! beside it, because a table typed here would be the same claim written
//! twice.
//!
//! **AND ONE OF THEM SAYS THE ARTIFACT IS WRONG ABOUT ITSELF.** The miniature's
//! n-gram table is eight shards of 2 500 012 rows — the SHIPPED table's padded
//! row space over sixteen — while the buffers beside it publish the shipped
//! model's sixteen primes past 20 000 000 and its config's
//! `ngram_vocab_size_base: 1250000` derives a third number again. The metadata
//! was not carved with the table, so the row imports and bakes and refuses at
//! LOAD, at that one rectangle. That refusal is
//! [`the_miniatures_ngram_table_is_a_carve_its_own_metadata_does_not_describe`]
//! here and `engine-metal/tests/qwen4_two_bit_first_light.rs` there; everything
//! else in this file is green against the bytes.
//!
//! **AND WHY IT SKIPS RATHER THAN FAILS.** The snapshot is a fact about a
//! machine that pulled it; a build box that has not is "not asked", said out
//! loud, not a red run.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use model_dsl::Dtype;
use models::qwen_4::model::{Mix, Model};

const REPO: &str = "models--Sawfwair--Qwen3.8-Flash-Next-MLX-Mixed-2bit";

/// The catalog row this miniature is declared by.
const SKU: &str = "qwen38-flash-mlxu2-kv-bf16";

/// The text under test: the miniature's own geometry at the miniature's own
/// mix, which is exactly what the catalog row is.
fn text() -> Model {
    Model::flash_mini(Mix::MIXED_2BIT, Dtype::Bf16, 1)
}

/// The snapshot directory — the one with an index in it.
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

/// The snapshot's `weight_map`, as `tensor -> shard file`.
fn weight_map(dir: &Path) -> Option<BTreeMap<String, String>> {
    let text = std::fs::read_to_string(dir.join("model.safetensors.index.json")).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&text).ok()?;
    let map = parsed.get("weight_map")?.as_object()?;
    Some(
        map.iter()
            .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string())))
            .collect(),
    )
}

/// One tensor's stored `(safetensors dtype, shape)`, over every shard the
/// snapshot is split into.
type Header = BTreeMap<String, (String, Vec<u64>)>;

fn headers(dir: &Path) -> Option<Header> {
    let mut all = Header::new();
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            name.ends_with(".safetensors").then_some(path)
        })
        .collect();
    shards.sort();
    for shard in shards {
        for (name, value) in shard_header(&shard)? {
            let Some(dtype) = value.get("dtype").and_then(serde_json::Value::as_str) else {
                continue; // `__metadata__`
            };
            let shape = value
                .get("shape")?
                .as_array()?
                .iter()
                .filter_map(serde_json::Value::as_u64)
                .collect();
            all.insert(name, (dtype.to_string(), shape));
        }
    }
    Some(all)
}

fn shard_header(path: &Path) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut len = [0u8; 8];
    std::io::Read::read_exact(&mut file, &mut len).ok()?;
    let len = u64::from_le_bytes(len) as usize;
    let mut buf = vec![0u8; len];
    std::io::Read::read_exact(&mut file, &mut buf).ok()?;
    let parsed: serde_json::Value = serde_json::from_slice(&buf).ok()?;
    parsed.as_object().cloned()
}

/// One stored `I64` buffer, read out of the bytes — the three hash tables the
/// checkpoint publishes and this text derives instead.
fn read_i64(dir: &Path, map: &BTreeMap<String, String>, name: &str) -> Option<Vec<i64>> {
    let shard = dir.join(map.get(name)?);
    let header = shard_header(&shard)?;
    let entry = header.get(name)?;
    assert_eq!(
        entry.get("dtype").and_then(serde_json::Value::as_str),
        Some("I64"),
        "`{name}` is a published hash buffer and those are I64"
    );
    let offsets = entry.get("data_offsets")?.as_array()?;
    let start = offsets.first()?.as_u64()? as usize;
    let end = offsets.get(1)?.as_u64()? as usize;
    let bytes = std::fs::read(&shard).ok()?;
    let head = u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?) as usize;
    let base = 8 + head;
    let slab = bytes.get(base + start..base + end)?;
    Some(
        slab.chunks_exact(8)
            .map(|w| i64::from_le_bytes(w.try_into().expect("eight bytes")))
            .collect(),
    )
}

/// **THE NAMES THIS ARTIFACT WOULD BE READ WRONG WITHOUT, AND THE SIX IT IS
/// DELIBERATELY READ WITHOUT.**
///
/// Six tensors the snapshot ships that no read of this text names, each for a
/// reason already written down somewhere else in the tree:
///
/// * the full-attention layer's three `self_attn.indexer.*` planes — the QSA
///   cut, which `models::Mixer::Attn`'s doc states outright;
/// * the PLE's three published hash buffers, which are pure functions of the
///   config and are DERIVED here rather than read (`models::Ple`'s doc), and
///   which the gate below holds the derivation against.
///
/// A seventh would be a hole. This asserts the difference is exactly those six
/// and that nothing else is missing in either direction.
#[test]
fn the_two_bit_arm_reads_the_miniature_and_names_the_six_it_does_not() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("not asked: no {REPO} snapshot under $HOME/.cache/huggingface/hub");
        return;
    };
    let Some(map) = weight_map(&dir) else {
        eprintln!("not asked: {dir:?} has no readable index");
        return;
    };

    // The logical census: the `.weight`/bare views, a quantized triplet's
    // `.scales`/`.biases` companions dropped — they join at the seams their
    // codes join at and are not separately named by any declaration.
    let census: BTreeSet<String> = map
        .keys()
        .filter(|name| !name.ends_with(".scales") && !name.ends_with(".biases"))
        .cloned()
        .collect();

    let model = text();
    let read: BTreeSet<String> = model.mlx_source_names().into_iter().collect();

    let unread: BTreeSet<String> = census.difference(&read).cloned().collect();
    let invented: BTreeSet<String> = read.difference(&census).cloned().collect();

    assert!(
        invented.is_empty(),
        "the 2-bit arm reads {} name(s) the miniature does not hold:\n{}\n",
        invented.len(),
        invented.iter().cloned().collect::<Vec<_>>().join("\n"),
    );
    assert_eq!(
        unread.iter().map(String::as_str).collect::<Vec<_>>(),
        vec![
            "language_model.model.layers.1.ple.ple_embedding.layer_multipliers",
            "language_model.model.layers.1.ple.ple_embedding.ngram_heads_offsets",
            "language_model.model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
            "language_model.model.layers.3.self_attn.indexer.index_qk_proj.weight",
            "language_model.model.layers.3.self_attn.indexer.k_layernorm.weight",
            "language_model.model.layers.3.self_attn.indexer.q_layernorm.weight",
        ],
        "exactly six of the miniature's names go deliberately unread — the QSA \
         indexer's three and the PLE's three derived hash buffers",
    );
    assert_eq!(
        read.len(),
        116,
        "the 2-bit arm reads a hundred and sixteen of the miniature's \
         hundred and twenty-two names"
    );
}

/// **EVERY PLANE THE 2-BIT ARM DECLARES IS AT THE POINT THE FILE WROTE IT.**
///
/// This is the whole of the mix, checked one tensor at a time, and it is the
/// gate the single-`w` derivation fails. Under that derivation the miniature's
/// `q_proj` would be declared `U8g64` — eight bits over sixty-four codes — and
/// the file writes it at four; its `embed_tokens.weight` would be declared a
/// quantized triplet and the file ships one bare bf16 plane; its two
/// `block_inject_weight`/`in_proj_{a,b}` slivers would be declared dense and
/// the file quantizes them.
///
/// Read off the DECLARATION's own `(bits, group)` — `checkpoint_dsl::encoding`,
/// which is where a `Dtype` name becomes the two integers a plane is read by —
/// against the stored rectangles, which is the arithmetic `affine_planes` will
/// do at load. A quantized plane's `.scales` last extent is its group count, so
/// `words · 32/bits == groups · group` is the file agreeing with the text; an
/// unquantized one is asserted to have NO `.scales` beside it at all, because
/// "the text says bf16" and "the file ships bf16" are two different claims and
/// only the second one is about the artifact.
#[test]
fn every_plane_the_two_bit_arm_declares_is_the_point_the_file_wrote() {
    use checkpoint::types::Encoding;

    let Some(dir) = snapshot_dir() else {
        eprintln!("not asked: no {REPO} snapshot under $HOME/.cache/huggingface/hub");
        return;
    };
    let Some(head) = headers(&dir) else {
        eprintln!("not asked: {dir:?} holds no readable safetensors header");
        return;
    };

    let model = text();
    let mut faults = Vec::new();
    let mut quantized = 0usize;
    let mut raw = 0usize;
    let mut points: BTreeMap<(u32, u32), usize> = BTreeMap::new();

    for (weight, name) in model.mlx_planes() {
        let Some((stored, shape)) = head.get(&name) else {
            faults.push(format!("`{name}`: the miniature does not hold it"));
            continue;
        };
        let companion = name
            .strip_suffix(".weight")
            .map(|stem| format!("{stem}.scales"));
        let scales = companion.as_ref().and_then(|n| head.get(n));

        match checkpoint_dsl::encoding(weight.dtype) {
            Encoding::Quant(spec) => {
                quantized += 1;
                let bits = u32::from(spec.bits_per_element);
                let group = spec.group_size;
                *points.entry((bits, group)).or_default() += 1;
                let Some((_, scales)) = scales else {
                    faults.push(format!(
                        "`{name}`: declared ({bits}, {group}) and the file ships no \
                         `.scales` companion — it is not quantized at all"
                    ));
                    continue;
                };
                let (Some(words), Some(groups)) = (shape.last(), scales.last()) else {
                    faults.push(format!("`{name}`: a bank with no contracted axis"));
                    continue;
                };
                let codes = words * u64::from(32 / bits);
                if codes != groups * u64::from(group) {
                    faults.push(format!(
                        "`{name}`: `{:?}` is ({bits}, {group}), which reads {words} words \
                         as {} groups, and the file ships {groups}",
                        weight.dtype,
                        codes / u64::from(group),
                    ));
                }
            }
            Encoding::Raw(dtype) => {
                raw += 1;
                if scales.is_some() {
                    faults.push(format!(
                        "`{name}`: declared `{:?}`, unquantized, and the file ships a \
                         `.scales` companion beside it",
                        weight.dtype,
                    ));
                    continue;
                }
                // The one width the import bends: `A_log` and the GDN's gated
                // norm are declared f32 because that is what the forward
                // accumulates them in, and every MLX conversion of this family
                // stores them bf16. That is a cast the import states, not a
                // disagreement about the bytes.
                let bends = matches!(dtype, checkpoint::types::DType::F32);
                let want = match dtype {
                    checkpoint::types::DType::Bf16 => "BF16",
                    checkpoint::types::DType::F32 => "F32",
                    other => {
                        faults.push(format!("`{name}`: declared `{other:?}`, unexpected here"));
                        continue;
                    }
                };
                if stored != want && !(bends && stored == "BF16") {
                    faults.push(format!(
                        "`{name}`: declared `{:?}` and the file ships `{stored}`",
                        weight.dtype,
                    ));
                }
            }
        }
    }

    assert!(
        faults.is_empty(),
        "the 2-bit arm and the miniature disagree on {} plane(s):\n{}\n",
        faults.len(),
        faults.join("\n"),
    );
    assert_eq!(
        quantized, 79,
        "seventy-nine of the arm's reads land in a quantized triplet, which is \
         exactly how many `.scales` planes the miniature ships"
    );
    assert_eq!(raw, 37, "the other thirty-seven are stored plainly");
    assert_eq!(
        points,
        BTreeMap::from([((4, 64), 59), ((4, 32), 8), ((2, 128), 12)]),
        "the mix is three points and no more: the (4, 64) default, the eight \
         n-gram shards at (4, 32), and the twelve routed expert banks at \
         (2, 128) — and NO 8-bit entry anywhere, which is the reading a \
         single-`w` derivation could not have produced",
    );
}

/// **THE CONVERSION'S OWN `config.json`, AND THE ROW EACH PAIR MAPS TO.**
///
/// The pairing test above reads the FILE. This reads what the conversion SAYS
/// it did, holds the two against each other, and checks that every pair it
/// spends is one this tree has a `Dtype` row for. A pair with no row is a plane
/// that would be decoded around the wrong centre with no NaN to notice it by.
#[test]
fn the_miniature_spends_only_pairs_this_tree_can_spell() {
    use checkpoint::types::Encoding;

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
            v.get("bits")
                .and_then(serde_json::Value::as_u64)
                .expect("bits"),
            v.get("group_size")
                .and_then(serde_json::Value::as_u64)
                .expect("group_size"),
        )
    };
    let row_for = |bits: u64, group: u64| -> Option<Dtype> {
        Some(match (bits, group) {
            (4, 64) => Dtype::U4g64,
            (8, 64) => Dtype::U8g64,
            (4, 32) => Dtype::U4g32,
            (2, 32) => Dtype::U2g32,
            (2, 64) => Dtype::U2g64,
            (2, 128) => Dtype::U2g128,
            _ => return None,
        })
    };
    let point_of = |dtype: Dtype| -> (u64, u64) {
        match checkpoint_dsl::encoding(dtype) {
            Encoding::Quant(spec) => (u64::from(spec.bits_per_element), u64::from(spec.group_size)),
            other => panic!("`{dtype:?}` is not a quantized plane: {other:?}"),
        }
    };

    let default = pair(&serde_json::Value::Object(quant.clone()));
    assert_eq!(default, (4, 64), "the mixed-2bit default is MLX's own");

    let mut seen = BTreeSet::from([default]);
    let mut experts: Vec<(u32, String, u64, u64)> = Vec::new();
    let mut shards = 0usize;
    let mut overrides = 0usize;

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
            (bits, group),
            "`{name}`: the row `{row:?}` reports a different point than the file states",
        );

        if let Some(rest) = name.strip_prefix("language_model.model.layers.") {
            if let Some((l, tail)) = rest.split_once('.') {
                let l: u32 = l.parse().expect("a layer number");
                if let Some(half) = tail.strip_prefix("mlp.switch_mlp.") {
                    experts.push((l, half.to_string(), bits, group));
                }
                if tail.starts_with("ple.ple_embedding.ngram_embedding.shard_") {
                    shards += 1;
                    assert_eq!(
                        (bits, group),
                        (4, 32),
                        "`{name}`: the n-gram shards' 160-wide rows cannot group by \
                         sixty-four"
                    );
                }
            }
        }
    }

    assert_eq!(overrides, 20, "the miniature states twenty overrides");
    assert_eq!(
        seen,
        BTreeSet::from([(4, 64), (4, 32), (2, 128)]),
        "the mix is three pairs, and NONE of them is eight bits — the fact the \
         `proj = U8g64` derivation could not have read off this file",
    );
    assert_eq!(shards, 8, "eight n-gram shards, `split_ngram_parts: 8`");

    // **UNIFORM, WHICH IS WHY THE FUSED BANK SURVIVES HERE.** See the gate
    // below: the DeepSeek-V4 DQ artifact next door moves its gate half's group
    // and this one does not.
    experts.sort();
    assert_eq!(
        experts,
        vec![
            (0, "down_proj".to_string(), 2, 128),
            (0, "gate_proj".to_string(), 2, 128),
            (0, "up_proj".to_string(), 2, 128),
            (1, "down_proj".to_string(), 2, 128),
            (1, "gate_proj".to_string(), 2, 128),
            (1, "up_proj".to_string(), 2, 128),
            (2, "down_proj".to_string(), 2, 128),
            (2, "gate_proj".to_string(), 2, 128),
            (2, "up_proj".to_string(), 2, 128),
            (3, "down_proj".to_string(), 2, 128),
            (3, "gate_proj".to_string(), 2, 128),
            (3, "up_proj".to_string(), 2, 128),
        ],
        "all three routed projections are (2, 128) on all four layers",
    );
}

/// **WHY THIS FAMILY KEEPS THE FUSED EXPERT BANK.**
///
/// `deepseek_v4::model::GateUp::Split` exists because a `Weight` carries ONE
/// `Dtype` and a `Dtype` is where an MLX affine group is written down, so a
/// fused `[gate | up]` bank needs the two stored halves to agree on their
/// point — and on the DeepSeek-V4 DQ artifact they do not, on four of five
/// layers.
///
/// This artifact is the other answer, and it is a fact about the file and not
/// a preference: the halves' `.scales` rectangles are identical on every layer,
/// so they join into one rectangle at the intermediate axis, so
/// `experts_gate_up` states them and the routed matmul fires once. Asserted
/// against the bytes, because the day a qwen conversion moves one half is the
/// day this text needs the split form too, and this is the assert that would
/// go red.
#[test]
fn the_fused_expert_bank_is_stateable_because_the_halves_agree() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("not asked: no {REPO} snapshot under $HOME/.cache/huggingface/hub");
        return;
    };
    let Some(head) = headers(&dir) else {
        eprintln!("not asked: {dir:?} holds no readable safetensors header");
        return;
    };

    let mut rectangles = Vec::new();
    for l in 0..4 {
        let of = |half: &str, suffix: &str| -> Vec<u64> {
            let name = format!(
                "language_model.model.layers.{l}.mlp.switch_mlp.{half}_proj{suffix}"
            );
            head.get(&name)
                .unwrap_or_else(|| panic!("`{name}` is in the header"))
                .1
                .clone()
        };
        rectangles.push((
            l,
            of("gate", ".weight"),
            of("gate", ".scales"),
            of("up", ".weight"),
            of("up", ".scales"),
        ));
    }

    let mut disagreeing = Vec::new();
    for (l, gate_w, gate_s, up_w, up_s) in &rectangles {
        if gate_w != up_w || gate_s != up_s {
            disagreeing.push(format!(
                "layer {l}: gate {gate_w:?}/{gate_s:?} beside up {up_w:?}/{up_s:?}"
            ));
        }
    }
    assert!(
        disagreeing.is_empty(),
        "the gate and up halves disagree on {} layer(s), which is what a fused \
         `experts_gate_up` bank cannot state — this text would need \
         `GateUp::Split`:\n{}\n",
        disagreeing.len(),
        disagreeing.join("\n"),
    );

    // And the fused bank is the sum of them at the seam it bands, which is
    // what `read_concat` joins.
    let (_, gate_w, gate_s, ..) = &rectangles[0];
    assert_eq!(
        (gate_w.as_slice(), gate_s.as_slice()),
        ([16, 640, 160].as_slice(), [16, 640, 20].as_slice()),
        "sixteen experts of 640 rows, 2560 columns at two bits (160 u32 words) \
         over twenty groups of a hundred and twenty-eight",
    );
}

/// **THE HASH CONSTANTS ARE DERIVED, AND THE MINIATURE'S TABLE IS THE ONE THEY
/// DESCRIBE — WHICH IT WAS NOT.**
///
/// `models::Ple`'s doc says the three buffers the checkpoint ships —
/// `layer_multipliers`, `ngram_heads_vocab_sizes`, `ngram_heads_offsets` — are
/// pure functions of the config, so the text computes them and the census holds
/// the computation against the published buffers rather than trusting either
/// alone. The first time that census was run against a file of this family it
/// found a THIRD answer neither side predicted, and this doc recorded it:
///
/// * the miniature PUBLISHED the shipped model's sixteen primes past
///   20 000 000, summing to 320 001 446 and padding to 320 001 536, with
///   offsets running to 300 001 275;
/// * it STORED eight shards of 2 500 012 rows — 20 000 096, which is
///   `320 001 536 / 16`: eight of the shipped table's 128 stored shards, kept
///   verbatim;
/// * 20 000 096 is not a multiple of the config's own
///   `make_ngram_vocab_size_divisible_by: 128`, which a padded table always is,
///   so the stored table was a SLICE and not a table this hashing built, and NO
///   `ngram_vocab_size_base` derives it;
/// * so fifteen of the sixteen head offsets pointed past the end of the table
///   beside them, and the row's load refused at that one rectangle.
///
/// **THE ARTIFACT MOVED, NOT THE TEXT.** `benches/shrink_checkpoint.py` now
/// re-cuts the table BY HEAD instead of by stored shard: miniature head `h`
/// takes its own prime's worth of rows out of head `h`'s segment of the
/// original, the sixteen segments are concatenated and re-chopped into eight
/// equal shards of 2 500 192, and the two published head buffers are rewritten
/// to the miniature's own primes and offsets. `layer_multipliers` is carried
/// unchanged because it is a function of the vocabulary and the seed, and a
/// shrink touches neither.
///
/// So this is the agreement census the disagreement one turned into. Every
/// number `ngram_vocab_size_base: 1250000` derives is the number the file
/// holds: sixteen primes past 1 250 000, offsets to 18 751 345, 20 001 534 rows
/// padded to 20 001 536 = 8 × 2 500 192, and a last addressable row of
/// 20 001 533 that is INSIDE the table. The sweep that showed no base reaches
/// the old 20 000 096 is kept, pointed at the number the file holds now, so it
/// still says the row count is a derivation's output and not a slicer's.
#[test]
fn the_miniatures_ngram_table_is_the_carve_its_own_metadata_describes() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("not asked: no {REPO} snapshot under $HOME/.cache/huggingface/hub");
        return;
    };
    let Some(map) = weight_map(&dir) else {
        eprintln!("not asked: {dir:?} has no readable index");
        return;
    };
    let Some(head) = headers(&dir) else {
        eprintln!("not asked: {dir:?} holds no readable safetensors header");
        return;
    };
    let stem = "language_model.model.layers.1.ple.ple_embedding";
    let published = |tail: &str| -> Vec<i64> {
        read_i64(&dir, &map, &format!("{stem}.{tail}"))
            .unwrap_or_else(|| panic!("`{stem}.{tail}` reads out of the shard"))
    };

    let ple = text().ple.expect("the mini arm carries the PLE");
    let as_i64 = |v: &[u64]| -> Vec<i64> { v.iter().map(|x| *x as i64).collect() };

    // The one that agrees: splitmix64 over the seed and the vocabulary, neither
    // of which the carve touched.
    assert_eq!(
        as_i64(&ple.mults),
        published("layer_multipliers"),
        "the splitmix64 multipliers this text derives are the ones the \
         conversion wrote down"
    );

    // The two the re-carve rewrote. They are the miniature's own now, and the
    // text derives them from the config without reading them.
    let primes = published("ngram_heads_vocab_sizes");
    let offsets = published("ngram_heads_offsets");
    assert_eq!(primes.len(), 16, "sixteen hashed heads");
    assert_eq!(
        primes[0], 1_250_003,
        "the miniature publishes the first prime past its OWN \
         `ngram_vocab_size_base: 1250000` — it used to publish the shipped \
         model's 20 000 003, describing a table sixteen times taller"
    );
    assert_eq!(
        as_i64(&ple.primes),
        primes,
        "the sixteen primes this text derives are the ones the file publishes"
    );
    assert_eq!(
        as_i64(&ple.offsets),
        offsets,
        "and so are their prefix sums — the offsets ran to 300 001 275 before \
         the re-carve, into a row space that was not there"
    );
    let published_total: i64 = primes.iter().sum();
    assert_eq!(published_total, 20_001_534);
    assert_eq!(offsets.last().copied(), Some(18_751_345));

    // The table it actually ships.
    let rows: Vec<u64> = (0..8)
        .map(|i| {
            head.get(&format!("{stem}.ngram_embedding.shard_{i}.weight"))
                .expect("each shard is in the header")
                .1[0]
        })
        .collect();
    assert_eq!(rows, vec![2_500_192u64; 8], "eight shards, all one size");
    let stored: u64 = rows.iter().sum();
    assert_eq!(stored, 20_001_536);
    assert_eq!(
        stored % 128,
        0,
        "and they ARE a multiple of `make_ngram_vocab_size_divisible_by`, which \
         every padded table is and the verbatim slice this used to be was not"
    );

    // What the text derives from the config, held against the file — and the
    // sweep that says the row count is a derivation's output.
    assert_eq!(
        ple.padded_vocab, 20_001_536,
        "`ngram_vocab_size_base: 1250000` derives 20 001 536 rows"
    );
    assert_eq!(
        ple.padded_vocab, stored,
        "and that is what the eight shards hold: the load lands at this \
         rectangle, which is where it used to refuse"
    );

    // **THE GATHER IS INSIDE THE TABLE.** The refusal this file used to record
    // was worth taking because the alternative was a hasher indexing past the
    // end of a real plane. That is the claim, so it is asserted directly.
    let last = ple.offsets[15] + ple.primes[15] - 1;
    assert_eq!(last, 20_001_533);
    assert!(
        last < stored,
        "head 15 can name row {last} of a {stored}-row table, and a hashed \
         gather that leaves its plane is what wall two existed to prevent"
    );

    let mut reachable = Vec::new();
    for base in 1_240_000u64..1_260_000 {
        if padded_for(base) == stored {
            reachable.push(base);
        }
    }
    assert_eq!(
        reachable,
        vec![1_250_000, 1_250_001, 1_250_002, 1_250_003],
        "the bases in twenty thousand that derive the miniature's {stored} rows \
         are the four that round up to the same first prime, and the config's \
         own 1 250 000 is the least of them. The old 20 000 096 was reached by \
         NONE of them, which is how the census knew the metadata had not been \
         carved with the table."
    );
}

/// `hash_constants`' own arithmetic, restated over one base — sixteen primes at
/// or past it, summed, rounded up to 128. Written here rather than exported
/// because the claim it serves is that NO base reaches the miniature's row
/// count, and a sweep needs the function.
fn padded_for(base: u64) -> u64 {
    fn is_prime(v: u64) -> bool {
        if v < 2 {
            return false;
        }
        if v % 2 == 0 {
            return v == 2;
        }
        let mut d = 3;
        while d * d <= v {
            if v % d == 0 {
                return false;
            }
            d += 2;
        }
        true
    }
    let mut total = 0u64;
    let mut prime = base - 1;
    for _ in 0..16 {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
        total += prime;
    }
    total.div_ceil(128) * 128
}

/// **THE LADDER HANDS THIS FILE TO THIS ROW.**
///
/// `identify` walks the catalog's imports in order and takes the first that
/// fits. The two quantized qwen4 rows must miss on each other's artifacts —
/// the 4/8 row declares an eight-bit embedding this file ships bare, and this
/// row declares four-bit triplets a bf16 file has none of — and this is the
/// half of that sentence a box holding the miniature can actually check.
#[test]
fn the_miniature_identifies_as_the_two_bit_row() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("not asked: no {REPO} snapshot under $HOME/.cache/huggingface/hub");
        return;
    };
    let mut shards: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("the snapshot reads")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            name.ends_with(".safetensors").then_some(path)
        })
        .collect();
    shards.sort();
    if shards.is_empty() {
        eprintln!("not asked: {dir:?} holds no tensor container");
        return;
    }
    let source = ztensor_compat::index_all(&shards).expect("the miniature's shards open as one");
    match models::identify(&source) {
        Ok(named) if named == SKU => {}
        Ok(other) => panic!("the miniature identifies as `{other}`"),
        Err(why) => panic!("the miniature matches no SKU: {why:?}"),
    }
}

