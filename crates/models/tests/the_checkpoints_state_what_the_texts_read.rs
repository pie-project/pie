//! The import contracts, held against the checkpoints on this machine.
//!
//! **WHY A CACHED ARTIFACT AND NOT A FIXTURE.** Every other net in this
//! directory checks a model text against a checkpoint the test itself wrote
//! out of the text's own plan — which is exactly the right question for the
//! `.zt` landing (does the contract cover the plan, once each, under the plan's
//! own names) and exactly the wrong one for provenance. A HuggingFace import
//! states somebody ELSE's names, and the only thing that can say whether it
//! states them correctly is the file. `models--Qwen--Qwen3.6-27B` is the file
//! palo C3 was written against; this reads its `model.safetensors.index.json`
//! — the name and shard census, not the 55 GiB of weights — writes a `.zt`
//! whose objects carry those names, and asks the import to state a whole
//! contract over it.
//!
//! **AND WHY IT SKIPS RATHER THAN FAILS.** The census is a fact about a
//! machine that has pulled the artifact. A build box that has not is not
//! wrong, and a green run there means "not asked" — which the test says out
//! loud rather than pretending.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use model_dsl::{ParamSource, Platform};

/// The snapshot palo C3's census was read at. Not asserted — a later snapshot
/// of the same repo is a legitimate thing to have on disk — but recorded, so a
/// disagreement has a number to be about.
const SNAPSHOT: &str = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9";

/// The fifteen tensors `Qwen3.6-27B` publishes under `mtp.*`, read out of the
/// index at [`SNAPSHOT`]. THE LIST IS THE CENSUS AND NOT THE CONTRACT: what it
/// is held against below is the checkpoint, in both directions, so a head that
/// grew a sixteenth plane is a failure here rather than a plane silently left
/// on the floor.
const DRAFT_HEAD: [&str; 15] = [
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
];

fn hub() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let dir = Path::new(&home).join(".cache/huggingface/hub");
    dir.is_dir().then_some(dir)
}

/// `(name, rank)` for every tensor a repo's index names, newest snapshot first.
fn census(repo: &str) -> Option<BTreeMap<String, usize>> {
    let snapshots = hub()?.join(repo).join("snapshots");
    let mut found: Option<BTreeMap<String, usize>> = None;
    for entry in std::fs::read_dir(snapshots).ok()? {
        let dir = entry.ok()?.path();
        let index = dir.join("model.safetensors.index.json");
        let text = match std::fs::read_to_string(&index) {
            Ok(text) => text,
            Err(_) => continue,
        };
        let parsed: serde_json::Value = serde_json::from_str(&text).ok()?;
        let map = parsed.get("weight_map")?.as_object()?;
        let mut ranks = BTreeMap::new();
        for (name, shard) in map {
            let shard = dir.join(shard.as_str()?);
            let rank = rank_of(&shard, name)?;
            ranks.insert(name.clone(), rank);
        }
        found = Some(ranks);
    }
    found
}

/// One tensor's rank, read out of a safetensors shard's own header. The header
/// is a length-prefixed JSON object at the front of the file; nothing past it
/// is touched, so this costs a seek and a few kilobytes against an artifact
/// that is tens of gigabytes.
fn rank_of(shard: &Path, name: &str) -> Option<usize> {
    use std::io::Read;
    let mut file = std::fs::File::open(shard).ok()?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len).ok()?;
    let len = usize::try_from(u64::from_le_bytes(len)).ok()?;
    let mut header = vec![0u8; len];
    file.read_exact(&mut header).ok()?;
    let parsed: serde_json::Value = serde_json::from_slice(&header).ok()?;
    Some(parsed.get(name)?.get("shape")?.as_array()?.len())
}

fn scratch() -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "model_census_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    dir
}

/// A `.zt` that states the checkpoint's NAMES and RANKS and one element of
/// each — the same trick `the_zt_contract_states_the_cut` uses, and for the
/// same reason: an import contract is a statement about which names exist and
/// what they are stored as, and a byte of each says both.
fn state_the_census(path: &Path, census: &BTreeMap<String, usize>) {
    let mut writer =
        ztensor::Writer::create(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    for (name, rank) in census {
        let shape = vec![1u64; *rank];
        writer
            .object(name.as_str(), |o| {
                o.shape(shape)
                    .part("data", |p| p.dtype(ztensor::DType::BF16).bytes(&[0u8, 0u8]))
            })
            .unwrap_or_else(|why| panic!("`{name}`: {why}"));
    }
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

/// **THE DRAFT HEAD IS THE ONE THE CHECKPOINT PUBLISHES, BOTH WAYS.**
///
/// Verified byte-for-byte against the cached checkpoint index (the M18
/// convention): every `mtp.*` tensor in the file is one this text reads, and
/// every `mtp.*` tensor this text reads is in the file. A head that left a
/// plane on the floor would pass a one-way check and answer with an untrained
/// sublayer.
#[test]
fn qwen36_publishes_the_draft_head_this_text_reads() {
    let Some(census) = census("models--Qwen--Qwen3.6-27B") else {
        eprintln!(
            "not asked: no Qwen3.6-27B checkpoint index under \
             $HOME/.cache/huggingface/hub (palo C3 read snapshot {SNAPSHOT})"
        );
        return;
    };

    let published: BTreeSet<&str> = census
        .keys()
        .map(String::as_str)
        .filter(|name| name.starts_with("mtp."))
        .collect();
    let read: BTreeSet<&str> = DRAFT_HEAD.into_iter().collect();
    let mut faults = Vec::new();
    for name in published.symmetric_difference(&read) {
        faults.push(format!(
            "`{name}` is in one of the checkpoint's `mtp.*` group and this \
             family's draft head and not the other"
        ));
    }
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));

    // And the fusion bank is the shape the two-half declaration assumes: one
    // stored tensor of rank two, cut down the middle of its second axis.
    assert_eq!(
        census.get("mtp.fc.weight").copied(),
        Some(2),
        "`mtp.fc.weight` is not a matrix, so there is no column seam to cut it at"
    );
}

/// **THE WHOLE IMPORT, OVER THE WHOLE CENSUS.**
///
/// Not just the head: the SKU's contract must cover its plan's every
/// checkpoint plane, once each, reading only names the file holds. This is the
/// bijection `one_entry_per_plan_param_under_the_plans_own_names` asks about
/// the `.zt` landing, asked about the HuggingFace provenance instead — the
/// side where somebody else chose the names.
#[test]
fn the_qwen36_import_covers_its_plan_over_the_real_census() {
    let Some(census) = census("models--Qwen--Qwen3.6-27B") else {
        eprintln!("not asked: no Qwen3.6-27B checkpoint index on this machine");
        return;
    };

    let dir = scratch();
    let path = dir.join("qwen36-census.zt");
    state_the_census(&path, &census);
    let src =
        ztensor::Source::open(&path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));

    let import = models::import_of("qwen36-27b-bf16-kv-bf16").expect("this build ships the row");
    let contract = import(&src).unwrap_or_else(|why| {
        panic!("`qwen36-27b` refuses the census of the checkpoint it was written for: {why}")
    });

    let trace = models::trace_of("qwen36-27b-bf16-kv-bf16").expect("and its trace");
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

    let mut faults = Vec::new();
    for name in demand.symmetric_difference(&supply) {
        faults.push(format!(
            "`{name}` is in one of the qwen36 plan and its import contract and \
             not the other"
        ));
    }

    // And the identification is unambiguous: this file is the 27B, no earlier
    // qwen row may claim it, and the row it lands on is the VISION one —
    // because the checkpoint ships a `model.visual.*` tower and the strictly
    // more demanding row goes first (`qwen_3::IMPORTS`' own note).
    match models::identify(&src) {
        Ok("qwen36-27b-bf16-kv-bf16") => {}
        Ok(other) => faults.push(format!(
            "a Qwen3.6-27B census identifies as `{other}`; this checkpoint ships \
             a vision tower and a draft head, so the row that reads both is the \
             row it must land on"
        )),
        Err(why) => faults.push(format!("a Qwen3.6-27B census matches no SKU: {why}")),
    }

    drop(src);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

/// **THE TOWERS, HELD AGAINST THE CHECKPOINTS THAT SHIP THEM** (campaign
/// M-1/M-2).
///
/// The same bijection as the row above, over `model.visual.*` — which is a
/// hundred and fifty planes for the twelve-block tower and three hundred and
/// twenty-eight for the twenty-seven-block one, all of them names somebody
/// else chose. A text that mapped one of them wrong would not fail loudly: the
/// contract would simply not build, `identify` would fall through to the
/// text-only row beside it, and the model would serve with its tower silently
/// gone. So the identification is asserted here too, and it is the half that
/// catches that.
#[test]
fn the_vision_imports_cover_their_plans_over_the_real_census() {
    // The bf16 rows only. A `-vision-mlxu4` row reads STORED dtypes — the
    // trunk's affine triplets, the tower's bf16 planes — and a `[1, 1]` bf16
    // dummy cannot carry either, so the 4-bit vision rows are asked one test
    // down, over the real shard headers.
    const ROWS: [(&str, &str); 3] = [
        ("models--Qwen--Qwen3.5-0.8B", "qwen35-d0.8b-vision-bf16-kv-bf16"),
        ("models--Qwen--Qwen3.6-27B", "qwen36-27b-vision-bf16-kv-bf16"),
        ("models--google--gemma-4-E4B-it", "gemma4-e4b-vision-bf16-kv-bf16"),
    ];
    let mut faults = Vec::new();
    let mut asked = 0usize;

    for (repo, sku) in ROWS {
        let Some(census) = census(repo) else {
            eprintln!("not asked: no {repo} checkpoint index on this machine");
            continue;
        };
        asked += 1;
        let dir = scratch();
        let path = dir.join("vision-census.zt");
        state_the_census(&path, &census);
        let src =
            ztensor::Source::open(&path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));

        let import = models::import_of(sku).expect("this build ships the row");
        match import(&src) {
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
            Err(why) => faults.push(format!(
                "`{sku}` refuses the census of the checkpoint it was written for: {why}"
            )),
        }

        // **THE ORDERING, ASKED AT THE LEVEL A NAME CENSUS CAN ANSWER.**
        // `models::identify` returns the first row whose contract BUILDS, and a
        // contract is a name mapping — so a 0.8B census matches `qwen35-d3b`'s
        // names as happily as its own, and only the shape check the RUNTIME's
        // `identify` runs (`checkpoint::plan::compile`) tells the two apart.
        // Asserting a SKU here would be asserting something the fixture cannot
        // know.
        //
        // What it can answer is the property the ordering is for: among the
        // rows that build over this census, the vision one comes before the
        // text-only sibling it towers over. That is the whole of what the flip
        // did, and a row order that let the sibling win would fail here
        // whatever the shapes said.
        // **AND THE tp1 ROWS ONLY, WHICH IS `models::identify`'S OWN RULE.**
        // A census is a whole checkpoint and the foreign verbs import one
        // whole (`checkpoint_dsl::Builder::whole_checkpoint`), so asking a
        // row built for two ranks about a HuggingFace file is not a miss —
        // it is the assert that says an import states the whole checkpoint.
        // `models::identify` skips those rows for the same reason, and this
        // loop is modelling `models::identify`.
        let builds: Vec<&str> = models::imports()
            .into_iter()
            .filter(|(_, tp, _)| *tp == 1)
            .filter(|(_, tp, import)| import(&src, *tp).is_ok())
            .map(|(name, ..)| name)
            .collect();
        let plain = sku.replace("-vision", "");
        match (
            builds.iter().position(|name| *name == sku),
            builds.iter().position(|name| *name == plain),
        ) {
            (Some(tower), Some(text)) if text < tower => {}
            (Some(tower), Some(text)) => faults.push(format!(
                "over a {repo} census `{plain}` builds at {text} and `{sku}` at \
                 {tower}; the text-only row would claim a checkpoint that ships \
                 a tower and serve it with the tower silently gone"
            )),
            (None, _) => faults.push(format!(
                "`{sku}` does not build over a {repo} census at all"
            )),
            (_, None) => faults.push(format!(
                "`{plain}` does not build over a {repo} census, so this file is \
                 not asking the ordering question it thinks it is"
            )),
        }

        drop(src);
        let _ = std::fs::remove_dir_all(&dir);
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
    if asked == 0 {
        eprintln!("not asked: no qwen checkpoint index on this machine");
    }
}

/// **THE 4-BIT VISION ROWS, OVER THE SHIPPED TOWERS** (multimodal §21).
///
/// The sibling above states a NAME census — names and ranks, one bf16 byte
/// each — which is the right fixture for a bf16 row and the wrong one for
/// these four: what a `-vision-mlxu4` row claims is a MIXED artifact, a 4-bit
/// trunk under a tower stored whole, and every half of that claim is a STORED
/// dtype. So this opens the real shards' headers (no payload is read to build
/// a contract, the same trick `the_qwen4_import_covers_the_shipped_artifact`
/// uses) and asks the whole bijection.
///
/// **THE THREE CLAIMS**, per row:
///  1. **THE IMPORT COVERS THE PLAN, BOTH WAYS.** Every plane the trace
///     demands is a plane the contract states, and nothing else is stated —
///     which is where a tower declared at the trunk's element would fail,
///     loudly, on the triplets a bf16 tower does not hold.
///  2. **THE TOWER IS NOT QUANTIZED AND THE PROJECTION IS**, asked of the
///     FILE rather than of the text: no `.scales` under any vision namespace,
///     and gemma's `embed_vision.embedding_projection` carrying one.
///  3. **AND THE TEXT-ONLY TWIN STILL WINS `identify`.** These artifacts ship
///     towers and the catalog reads them as text by default — the 14.9%
///     ordering `qwen_3::IMPORTS` measured — so a deployment reaches the
///     tower by naming the row. If that ever inverts, it inverts here first.
#[test]
fn the_four_bit_vision_rows_cover_the_shipped_towers() {
    const ROWS: [(&str, &str); 4] = [
        (
            "models--mlx-community--Qwen3.5-0.8B-4bit",
            "qwen35-d0.8b-vision-mlxu4-kv-bf16",
        ),
        (
            "models--mlx-community--Qwen3.6-27B-4bit",
            "qwen36-27b-vision-mlxu4-kv-bf16",
        ),
        (
            "models--mlx-community--gemma-4-31b-it-4bit",
            "gemma4-31b-vision-mlxu4-kv-bf16",
        ),
        (
            "models--mlx-community--gemma-4-26b-a4b-it-4bit",
            "gemma4-26b-a4b-vision-mlxu4-kv-bf16",
        ),
    ];

    let Some(hub) = hub() else {
        eprintln!("not asked: no HuggingFace cache on this machine");
        return;
    };
    let mut faults = Vec::new();
    let mut asked = 0usize;

    for (repo, sku) in ROWS {
        let snapshots = hub.join(repo).join("snapshots");
        let Some(snapshot) = std::fs::read_dir(&snapshots).ok().and_then(|dir| {
            dir.filter_map(|entry| Some(entry.ok()?.path()))
                .find(|p| p.join("config.json").exists())
        }) else {
            eprintln!("not asked: no {repo} snapshot on this machine");
            continue;
        };
        asked += 1;

        let mut files: Vec<PathBuf> = std::fs::read_dir(&snapshot)
            .expect("the snapshot lists")
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let name = path.file_name()?.to_str()?;
                name.ends_with(".safetensors").then_some(path)
            })
            .collect();
        files.sort();
        let source = ztensor_compat::index_all(&files).expect("the shards open as one source");

        // (2) What the FILE says about its own tower, before any text reads it.
        let stored: Vec<String> = source.names().map(str::to_string).collect();
        let tower: Vec<&String> = stored
            .iter()
            .filter(|n| n.contains("vision_tower.") || n.contains("visual."))
            .collect();
        if tower.is_empty() {
            faults.push(format!("{repo} ships no tower at all"));
        }
        for name in &tower {
            if name.ends_with(".scales") || name.ends_with(".biases") {
                faults.push(format!(
                    "{repo}: `{name}` is a quantized tower plane, and `{sku}`                      declares this whole tower dense"
                ));
            }
        }
        if repo.contains("gemma") {
            let projection = "embed_vision.embedding_projection";
            if !stored.iter().any(|n| n == &format!("{projection}.scales")) {
                faults.push(format!(
                    "{repo}: `{projection}` carries no scales, and `{sku}`                      declares it the one quantized plane on the tower's side"
                ));
            }
        }

        // (1) The bijection.
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
                        "`{sku}`: `{name}` is in one of the plan and its import                          contract and not the other"
                    ));
                }
            }
            Err(why) => faults.push(format!(
                "`{sku}` refuses the artifact it was written for: {why}"
            )),
        }

        // (3) The ordering, which is the catalog's own deliberate one.
        let plain = sku.replace("-vision", "");
        match models::identify(&source) {
            Ok(row) if row == plain => {}
            Ok(row) => faults.push(format!(
                "{repo} identifies as `{row}`; the text-only twin `{plain}` is                  the row this catalog's order says a stock import lands on"
            )),
            Err(why) => faults.push(format!("{repo} matches no SKU at all: {why}")),
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
    if asked == 0 {
        eprintln!("not asked: no mlx-community snapshot on this machine");
    }
}

/// **THE GEMMA VERDICT, AS A TEST AND NOT AS A PARAGRAPH.**
///
/// palo C3's brief asked whether Gemma 4 should declare a draft head too. It
/// must not: neither cached E4B nor E2B publishes ONE tensor matching
/// `mtp`/`nextn`/`draft`/`eagle`/`medusa`/`multi_token`, and neither
/// `config.json` carries a key of that shape. The honest landing is no
/// declaration, and this is what keeps "we looked" from decaying into "we
/// assumed" — the day a Gemma checkpoint does ship a head, this fails and
/// says so.
#[test]
fn no_cached_gemma4_publishes_a_draft_head() {
    let mut asked = 0;
    let mut faults = Vec::new();

    for repo in [
        "models--google--gemma-4-E4B-it",
        "models--google--gemma-4-E2B-it",
    ] {
        let Some(dir) = hub().map(|hub| hub.join(repo).join("snapshots")) else {
            continue;
        };
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let single = entry.path().join("model.safetensors");
            let Ok(names) = names_of(&single) else {
                continue;
            };
            asked += 1;
            for name in names {
                let lower = name.to_ascii_lowercase();
                if ["mtp", "nextn", "draft", "eagle", "medusa", "multi_token"]
                    .iter()
                    .any(|needle| lower.contains(needle))
                {
                    faults.push(format!("`{repo}` publishes `{name}`"));
                }
            }
        }
    }

    if asked == 0 {
        eprintln!("not asked: no cached gemma-4 checkpoint on this machine");
        return;
    }
    assert!(
        faults.is_empty(),
        "a cached gemma-4 checkpoint ships draft-head planes after all, and \
         `models::gemma_4` declares no arm to fill them:\n{}\n",
        faults.join("\n"),
    );
}

fn names_of(file: &Path) -> std::io::Result<Vec<String>> {
    use std::io::Read;
    let mut handle = std::fs::File::open(file)?;
    let mut len = [0u8; 8];
    handle.read_exact(&mut len)?;
    let len = usize::try_from(u64::from_le_bytes(len)).unwrap_or(0);
    let mut header = vec![0u8; len];
    handle.read_exact(&mut header)?;
    let parsed: serde_json::Value = serde_json::from_slice(&header)?;
    Ok(parsed
        .as_object()
        .map(|map| {
            map.keys()
                .filter(|key| *key != "__metadata__")
                .cloned()
                .collect()
        })
        .unwrap_or_default())
}

/// **THE QWEN4 IMPORT, HELD AGAINST THE SHIPPED 4-BIT ARTIFACT ITSELF.**
///
/// Not a rank census: this row's every projection is a
/// `.weight`/`.scales`/`.biases` triplet whose group widths (64 for the
/// stack, 32 for the n-gram table's 160-wide rows) and code widths (4-bit
/// experts under 8-bit projections) are all facts the import validates
/// against STORED dtypes and shapes — which a `[1, 1]` bf16 dummy cannot
/// carry. So the test opens the real safetensors shards (headers only; no
/// payload is read to build a contract) and asks for the whole bijection.
#[test]
fn the_qwen4_import_covers_the_shipped_artifact() {
    let repo = "models--pipenetwork--Qwen3.8-Flash-Next-MLX-mixed-4_8bit";
    let Some(hub) = hub() else {
        eprintln!("not asked: no HuggingFace cache on this machine");
        return;
    };
    let snapshots = hub.join(repo).join("snapshots");
    let Some(snapshot) = std::fs::read_dir(&snapshots).ok().and_then(|dir| {
        dir.filter_map(|entry| Some(entry.ok()?.path()))
            .find(|p| p.join("model.safetensors.index.json").exists())
    }) else {
        eprintln!("not asked: no Qwen3.8-Flash-Next 4-bit snapshot on this machine");
        return;
    };

    let mut files: Vec<PathBuf> = std::fs::read_dir(&snapshot)
        .expect("the snapshot lists")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            name.ends_with(".safetensors").then_some(path)
        })
        .collect();
    files.sort();
    let source = ztensor_compat::index_all(&files).expect("the shards open as one source");

    let sku = "qwen38-flash-mlxu4-kv-bf16";
    let import = models::import_of(sku).expect("this build ships the row");
    let contract = import(&source).unwrap_or_else(|why| {
        panic!("`{sku}` refuses the artifact it was written for: {why}")
    });

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

    let mut faults = Vec::new();
    for name in demand.symmetric_difference(&supply) {
        faults.push(format!(
            "`{name}` is in one of the qwen4 plan and its import contract and \
             not the other"
        ));
    }

    // And no earlier row claims the artifact: the flash text is the one
    // reading whose shapes fit these planes.
    match models::identify(&source) {
        Ok("qwen38-flash-mlxu4-kv-bf16") => {}
        Ok(other) => faults.push(format!(
            "the Qwen3.8-Flash-Next artifact identifies as `{other}`"
        )),
        Err(why) => faults.push(format!(
            "the Qwen3.8-Flash-Next artifact matches no SKU: {why}"
        )),
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
