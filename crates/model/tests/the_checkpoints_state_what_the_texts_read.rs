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

    let import = model::import_of("qwen36-27b-bf16-kv-bf16").expect("this build ships the row");
    let contract = import(&src).unwrap_or_else(|why| {
        panic!("`qwen36-27b` refuses the census of the checkpoint it was written for: {why}")
    });

    let trace = model::trace_of("qwen36-27b-bf16-kv-bf16").expect("and its trace");
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
        .filter(|t| t.visibility == model_loader::contract::Visibility::Public)
        .map(|t| t.name.as_str())
        .collect();

    let mut faults = Vec::new();
    for name in demand.symmetric_difference(&supply) {
        faults.push(format!(
            "`{name}` is in one of the qwen36 plan and its import contract and \
             not the other"
        ));
    }

    // And the identification is unambiguous: this file is the 27B and no
    // earlier qwen row may claim it.
    match model::identify(&src) {
        Ok("qwen36-27b-bf16-kv-bf16") => {}
        Ok(other) => faults.push(format!(
            "a Qwen3.6-27B census identifies as `{other}`; the IMPORTS order \
             lets a smaller row claim a bigger artifact"
        )),
        Err(why) => faults.push(format!("a Qwen3.6-27B census matches no SKU: {why}")),
    }

    drop(src);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
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
         `model::gemma_4` declares no arm to fill them:\n{}\n",
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
