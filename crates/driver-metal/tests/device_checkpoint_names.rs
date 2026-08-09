//! A real checkpoint, and whether the text's names find it.
//!
//! `model_dispatch.rs` proves every name the text states has a *spelling*.
//! That is a claim about the map. This is the claim about the **checkpoint**:
//! that the spelling names a tensor the plan actually published.
//!
//! The two can disagree, and did. An earlier draft of `Names::mlx` assumed the
//! HuggingFace convention (`model.layers.3.…`, `model.embed_tokens`), was
//! self-consistent, and passed its own test — because both sides of that test
//! were the same file. Only a checkpoint can settle it, so this test exists
//! and is gated on one.
//!
//! Gated on `PIE_METAL_SMOKE_CHECKPOINT`, the same variable `device_smoke.rs`
//! takes. **It has been run.** Against
//! `mlx-community/Llama-3.2-1B-Instruct-4bit` (372 tensors) every name both
//! fire classes state binds, which is the first claim about this text that a
//! real checkpoint rather than another source file settles.
//!
//! Two defects were between here and that result, and neither was in the name
//! map:
//!
//!   * The gate stated `qwen3_0_6b()` facts against a llama snapshot and
//!     reported 308 missing names -- every one a `qkv` or a `q_norm`, which is
//!     to say the FIXTURE's bindings. It derives facts from the checkpoint
//!     now, through the chain the seam uses.
//!   * The projection into `DecodeGeometry` read only the `q35_*` block, so
//!     a llama config -- which the descriptor reader put in `ll_*` -- was
//!     refused as "carrying no decoder shape" while carrying it in the other
//!     block. And it demanded a linear-attention slab of a stack that has no
//!     linear layers.
//!
//! Both of those are structurally impossible now, and saying why is the
//! point of keeping the paragraph. There is no descriptor, no family-prefixed
//! block, and no reader that guesses which block was filled: the checkpoint's
//! TENSORS pick a `model::catalog` row, the row projects one
//! `model::deployment::Deployment`, and `batch::geometry_from_deployment` is
//! arithmetic over that one value. A shape cannot be "in the other block"
//! when there is one block, and a refusal cannot demand a slab of a stack
//! whose row states none.


use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

use driver_metal::device::Context;
use driver_metal::weights::load::load;
use driver_metal::lowering::resolve::{Names, Store};
use model::families::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::families::llama_like::forward::llama_like_metal;
use model_compiler::lower::{Arg, Fire, Row, lower};
use model_compiler::trace::FireClass;

fn snapshot() -> Option<PathBuf> {
    std::env::var_os("PIE_METAL_SMOKE_CHECKPOINT").map(PathBuf::from)
}

/// WHICH MODEL a snapshot is, and at what affine point it was published.
///
/// This is the same two-step `serve/load.rs` runs, in the same order, and the
/// order is the whole change. What was here before read `config.json`, handed
/// it to an 845-line normalizer, and got back a `pie.model/1` document that
/// everything downstream then re-parsed — so this gate proved the text's
/// names against whatever THE DOCUMENT said, which is not the same claim as
/// proving them against the checkpoint. It is now: `identify` matches the
/// row's manifest against the tensors that are actually in the file, and a
/// checkpoint that does not match any row fails here rather than binding
/// half its names.
///
/// The config is still read, for ONE field. A row cannot state a
/// quantization: `mlx-community` publishes the same weights at 4 bits group
/// 64 and at 8 bits group 32, and the two pack to shapes no tensor's extents
/// tell apart. Every `affine_qmv_fast_bfloat16_gs_N_b_M` symbol the text
/// names carries that pair in its name, so a gate that guessed it would
/// report missing kernels for a checkpoint that is fine.
fn served(dir: &std::path::Path) -> (&'static dyn model::catalog::Variant, model::encoding::Encoding) {
    let meta = model_loader::checkpoint::read::parse_checkpoint_metadata(dir)
        .unwrap_or_else(|e| panic!("{} did not read as a checkpoint: {e:?}", dir.display()));
    let row = model::catalog::identify(&meta, &model::catalog::Override::None)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    // The embedded copy when the artifact carries one, the snapshot's own
    // file when it does not. A converted `.pie` archive has the first; a raw
    // HuggingFace snapshot -- which is what `PIE_METAL_NAMES_SNAPSHOT`
    // usually points at -- has only the second.
    let config = match model_loader::checkpoint::read::read_meta(&meta, model::encoding::CONFIG_OBJECT) {
        Ok(Some(bytes)) => String::from_utf8(bytes).expect("the embedded config is utf8"),
        _ => std::fs::read_to_string(dir.join("config.json"))
            .unwrap_or_else(|e| panic!("{}/config.json: {e}", dir.display())),
    };
    let encoding = model::encoding::Encoding::from_config_json(&config)
        .unwrap_or_else(|e| panic!("{}: the config does not state its encoding: {e}", dir.display()));
    (row, encoding)
}

/// The affine point a snapshot's encoding names, as the kernels spell it.
fn affine(encoding: &model::encoding::Encoding) -> driver_metal::batch::AffineFormat {
    driver_metal::batch::AffineFormat {
        bits: encoding.bits,
        group: encoding.group_size,
    }
}

/// Every weight name the Metal text states, over both fire classes.
///
/// The facts come from the CHECKPOINT, through the same chain the seam uses:
/// tensors -> `catalog` row -> `Deployment` -> `DecodeGeometry` ->
/// `text::facts_from`. An
/// earlier draft named `qwen3_0_6b()` here and reported 308 missing names
/// against a llama-3.2 snapshot -- every one of them a `qkv` or a `q_norm`,
/// which is to say the fixture's bindings and not the checkpoint's. A gate
/// that states its own facts is not testing the checkpoint.
fn names_the_text_states(
    facts: &LlamaLikeFacts,
    metal: &LlamaLikeMetalFacts,
) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (class, rows) in [(FireClass::Decode, 1usize), (FireClass::Prefill, 16)] {
        let plan = llama_like_metal(facts, metal, class);
        let low = lower(
            &plan,
            &vec![
                Row {
                    samples: true,
                    ..Row::default()
                };
                rows
            ],
            Fire {
                captures_across_splits: false,
            },
        )
        .expect("the metal text lowers");
        for arg in &low.args {
            if let Arg::Weight(name) = arg {
                // A `scale.` marker is a constant riding the weight slot; the
                // binder never looks it up.
                if !name.starts_with("scale.") {
                    out.insert(name.clone());
                }
            }
        }
    }
    out
}

#[test]
fn the_checkpoint_answers_the_names_the_text_states() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let (row, encoding) = served(&snapshot);
    let loaded = load(&context, &snapshot, row, &encoding).expect("the checkpoint loads");
    assert!(
        !loaded.tensors.is_empty(),
        "the plan published no tensors at all"
    );

    // The checkpoint's own shape, projected the way the seam projects it:
    // once, off the row the tensors matched, at the affine point the config
    // declares.
    let deployment = row
        .deployment(model::catalog::Deployed::single())
        .expect("the matched row deploys");
    let geometry = driver_metal::batch::geometry_from_deployment(
        &deployment,
        row.load_shape(),
        affine(&encoding),
    )
    .expect("a decodable geometry");
    let (facts, metal) =
        driver_metal::model::text::facts_from(&geometry, |t| loaded.tensors.contains_key(t));

    let named = HashMap::new();
    let mut store = Store::new(Names::mlx(), &loaded.tensors, &named);
    let mut missing: BTreeSet<String> = BTreeSet::new();
    for name in names_the_text_states(&facts, &metal) {
        use driver_metal::lowering::executor::Resolver as _;
        if store.weight(&name).is_none() {
            missing.insert(name);
        }
    }

    assert!(
        missing.is_empty(),
        "the text states {} name(s) this checkpoint does not answer:\n  {}\n\n\
         Either `Names::mlx` spells one wrong, or the plan did not publish it \
         — and the two are told apart by looking. The checkpoint published:\n  {}",
        missing.len(),
        missing.iter().cloned().collect::<Vec<_>>().join("\n  "),
        loaded
            .names()
            .iter()
            .take(40)
            .copied()
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

/// Not an assertion — a report, so a run against a new checkpoint says what it
/// holds without anyone editing a test to find out.
#[test]
fn what_this_checkpoint_published() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let (row, encoding) = served(&snapshot);
    let loaded = load(&context, &snapshot, row, &encoding).expect("the checkpoint loads");
    let names = loaded.names();
    eprintln!("{} tensors published; layer 0 and the globals:", names.len());
    for name in &names {
        if name.starts_with("layers.0.") || !name.starts_with("layers.") {
            eprintln!("  {name}");
        }
    }
}

/// **Every name a text states, against a checkpoint's PLAN — no device.**
///
/// The same claim as `the_checkpoint_answers_the_names_the_text_states` and a
/// hundredth of the cost: `compile_load_plan_for` reads a snapshot's metadata
/// and a catalog row's `LoadShape` and publishes every tensor the load WILL
/// stage, without staging any of it. The 26B gemma4 on this machine is SIGKILLed by the
/// staging test — fifteen gigabytes — and answers this one in milliseconds.
///
/// The published names are what matters, and they are NOT the checkpoint's own
/// keys: the plan renames. Comparing `Names::mlx` against a
/// `safetensors.index.json` was the first draft of this test and it reported
/// fifty mismatches that were all the rename.
///
/// Gated on `PIE_METAL_NAMES_SNAPSHOT` alone. It used to also want
/// `PIE_METAL_NAMES_ARCH` saying which naming convention to read a snapshot
/// with -- an operator hand-typing the answer to a question the checkpoint
/// already contains. `catalog::identify` reads it off the tensors, so the
/// variable is gone and with it the run that named the wrong one.
#[test]
fn every_name_the_text_states_is_a_tensor_the_load_plan_publishes() {
    let dirs = snapshots_to_check();
    if dirs.is_empty() {
        eprintln!(
            "SKIP: no snapshots. Set PIE_METAL_NAMES_SNAPSHOT (one path, or \
             several separated by `:`), or populate the HuggingFace cache."
        );
        return;
    }
    for dir in dirs {
        eprintln!("--- {}", dir.display());
        check_one_snapshot(&dir);
    }
}

/// Which snapshots this run holds the text to.
///
/// `PIE_METAL_NAMES_SNAPSHOT` when set -- one path, or several separated by
/// `:`. Otherwise every MLX snapshot in the HuggingFace cache, because a
/// machine that HAS six checkpoints should be told about all six and not the
/// one somebody remembered to name. Discovery is what turned this from a gate
/// somebody ran once into the gate that found gpt-oss's 246 misses.
fn snapshots_to_check() -> Vec<PathBuf> {
    if let Some(v) = std::env::var_os("PIE_METAL_NAMES_SNAPSHOT") {
        return std::env::split_paths(&v).collect();
    }
    let Some(home) = std::env::var_os("HOME") else {
        return Vec::new();
    };
    let hub = PathBuf::from(home).join(".cache/huggingface/hub");
    let Ok(entries) = std::fs::read_dir(&hub) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for repo in entries.flatten() {
        // MLX snapshots only: a GGUF export is a different container and this
        // gate reads safetensors.
        if !repo.file_name().to_string_lossy().contains("mlx-community") {
            continue;
        }
        let Ok(snaps) = std::fs::read_dir(repo.path().join("snapshots")) else {
            continue;
        };
        for snap in snaps.flatten() {
            if snap.path().join("config.json").is_file() {
                found.push(snap.path());
            }
        }
    }
    found.sort();
    found
}

/// Hold the text's names to one snapshot's load plan.
///
/// An architecture no Metal text serves is SKIPPED rather than failed: the
/// llama-like text does not model `qwen3_5`'s linear-attention interleave, and
/// reporting its every tensor as missing says nothing except that.
fn check_one_snapshot(dir: &std::path::Path) {
    let dir = dir.to_path_buf();
    let (row, encoding) = served(&dir);
    let deployment = row
        .deployment(model::catalog::Deployed::single())
        .unwrap_or_else(|e| panic!("{}: the matched row does not deploy -- {e}", dir.display()));

    // Whether a text serves this checkpoint at all, asked of the DRIVER
    // rather than of a list kept here. `qwen3_5` interleaves linear
    // attention, which the metal text does not model -- and every one of its
    // tensors would then report as missing for that one reason, which is a
    // page of output saying nothing.
    //
    // THE NAME COMES OFF THE ROW NOW, not off `config.json`'s `model_type`.
    // That is not a tidy-up. This gate read `model_type` while the seam
    // reduced `architectures[0]` -- lowercase, drop the `ForCausalLM` tail --
    // so the two held DIFFERENT strings for the same checkpoint, and the two
    // that differ (`qwen3_moe` against `qwen3moe`, `gpt_oss` against
    // `gptoss`) are exactly the two the seam refused while this gate reported
    // them healthy over five checkpoints. One row states one architecture,
    // so there is nothing left to disagree.
    let arch = deployment.advertised.arch;
    if !driver_metal::model::text::serves(arch) {
        eprintln!("    SKIP: no metal text serves `{arch}` (row `{}`)", row.id());
        return;
    }

    let target = driver_metal::loader::metal_storage_target();
    let (plan, _) =
        driver_metal::loader::compile_load_plan_for(&dir, &target, row, &encoding)
            .expect("the plan compiles");
    let published: BTreeSet<&str> = plan.tensors.iter().map(|t| t.name.as_str()).collect();

    // ONE map. gemma4 used to need its own, and the arch variable said which
    // -- a driver picking a name map per checkpoint. Both conventions are
    // candidates of `Names::mlx` now, so the checkpoint picks and this does
    // not.
    let names = Names::mlx();
    // A refusal is a FINDING and not a skip. The message says which fact the
    // geometry could not express, and swallowing it is how a checkpoint stops
    // being covered without anyone noticing -- the gate keeps passing and one
    // fewer model is held to it. It is a skip here anyway, for one reason:
    // `catalog_coverage.rs` holds EVERY row to the rule that a refusal names
    // something its `Deployment` shows, so the finding has a gate of its own
    // that runs without a checkpoint on the machine.
    let geometry = match driver_metal::batch::geometry_from_deployment(
        &deployment,
        row.load_shape(),
        affine(&encoding),
    ) {
        Ok(g) => g,
        Err(why) => {
            eprintln!("    SKIP: the geometry refuses this checkpoint -- {}", why.0);
            return;
        }
    };
    // Which weights the plan leaves in MXFP4, asked the same way the seam
    // asks it. Without this the text states an AFFINE routed projection at
    // gpt-oss's real group (32) -- a symbol no `kernel!` row declares, which
    // the declaration checker refuses by name. That refusal is correct and
    // the fix is to state the format the bank actually has.
    let mxfp4: std::collections::HashSet<&str> = plan
        .tensors
        .iter()
        .filter(|t| {
            matches!(
                &t.encoding,
                model_loader::types::Encoding::Quant(spec)
                    if spec.scheme == model_loader::types::QuantScheme::Mxfp4E2M1E8M0
            )
        })
        .map(|t| t.name.as_str())
        .collect();
    let (facts, metal) = driver_metal::model::text::facts_from_with(
        &geometry,
        |t| published.contains(t),
        |t| mxfp4.contains(t),
    );

    let tensors = HashMap::new();
    let named = HashMap::new();
    let store = Store::new(names, &tensors, &named);
    let mut missing: BTreeSet<String> = BTreeSet::new();
    for traced in names_the_text_states(&facts, &metal) {
        // EVERY spelling, not the first: this store has no staged tensors to
        // choose with, and a role that has several candidates resolves at run
        // time against the ones the checkpoint published. Asking only the
        // first would fail a checkpoint that spells it the second way, which
        // is every checkpoint but one.
        let candidates = store.checkpoint_names(&traced);
        if candidates.is_empty() {
            missing.insert(format!("{traced} -> (no spelling)"));
            continue;
        }
        if !candidates.iter().any(|c| published.contains(c.as_str())) {
            missing.insert(format!("{traced} -> {}", candidates.join(" | ")));
        }
    }

    assert!(
        missing.is_empty(),
        "{}: {} name(s) the text states are not tensors this load plan publishes:\n  \
         {}\n\nEither the map spells one wrong or the deployment's facts claim \
         something it does not ship — and the two are told apart by looking. \
         The plan publishes:\n  {}",
        dir.display(),
        missing.len(),
        missing.iter().cloned().collect::<Vec<_>>().join("\n  "),
        published
            .iter()
            .take(30)
            .copied()
            .collect::<Vec<_>>()
            .join("\n  ")
    );

    check_projection_widths(dir.as_path(), &plan, &facts, &metal, &store);
}

/// Every attention projection is as wide as the facts say it is.
///
/// # Why names are not enough
///
/// The gate above asks whether a name RESOLVES. gemma-4 passed it while
/// running ten of its sixty layers at the wrong shape, because the tensors
/// were all present and correctly spelled -- they were simply twice the
/// width the text was reading. A name gate cannot see that; only a shape
/// can.
///
/// Measured on `gemma-4-31b-it-4bit`: layer 0 (sliding) publishes
/// `q_proj [8192, ...]` = 32x256 and `k_proj [4096, ...]` = 16x256, while
/// layer 5 (full) publishes `[16384, ...]` = 32x512 and `[2048, ...]` =
/// 4x512. A driver holding one head shape read half of each full layer's Q
/// and ran a quarter past the end of its K.
///
/// The first dimension is the OUTPUT width for every layout here, quantized
/// or not -- the packing is on the input axis.
fn check_projection_widths(
    dir: &std::path::Path,
    plan: &model_loader::plan::LoadPlan,
    facts: &LlamaLikeFacts,
    metal: &LlamaLikeMetalFacts,
    store: &Store,
) {
    let shapes: HashMap<&str, &[i64]> = plan
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t.shape.as_slice()))
        .collect();
    let width_of = |role: &str| -> Option<i64> {
        store
            .checkpoint_names(role)
            .iter()
            .find_map(|c| shapes.get(c.as_str()))
            .and_then(|s| s.first().copied())
    };

    let mut wrong = Vec::new();
    for l in 0..facts.layers {
        let head_dim = metal.head_dim_at(l, facts.head_dim);
        let kv_heads = metal.kv_heads_at(l, facts.kv_heads);
        for (role, heads) in [
            (format!("layer.{l}.q_proj"), facts.q_heads),
            (format!("layer.{l}.k_proj"), kv_heads),
        ] {
            let Some(got) = width_of(&role) else { continue };
            let want = i64::from(heads * head_dim);
            if got != want {
                wrong.push(format!(
                    "{role}: the checkpoint publishes {got} and the text reads \
                     {want} ({heads} heads x {head_dim})"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{}: {} projection(s) are not the width the text reads them at. This \
         is the failure a NAME gate cannot see -- every tensor is present and \
         correctly spelled, and the driver reads the wrong number of bytes out \
         of it:\n  {}",
        dir.display(),
        wrong.len(),
        wrong.join("\n  ")
    );
}
