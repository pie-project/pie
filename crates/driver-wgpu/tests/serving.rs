//! **A real model, on a real adapter, answering.**
//!
//! Every other test in this crate stops one step short of a model.
//! `tests/device.rs` fires kernels against host references, `tests/arena.rs`
//! lowers six real texts and binds the offsets they assign, and
//! `tests/checkpoint.rs` proves that all 704 weight names a qwen3 plan binds
//! resolve to tensors a real checkpoint publishes. None of them ever puts a
//! weight on the card. Until this file, `driver_wgpu::shell::Shell` — the type
//! the whole crate exists to assemble — had **no caller in this crate at all**:
//! no unit test, no integration test, only `crates/engine`'s seam.
//!
//! So this asks the question none of the others can: given the bytes of
//! `Qwen/Qwen3-0.6B`, does this shell produce a distribution the model would
//! recognise as its own?
//!
//! # It does
//!
//! Shown `[15339, 1723, 88204, 6100, 41777, 2930]` five times and then the
//! first two of a sixth repeat, the model answers `88204, 6100, 41777, 2930` —
//! it continues the pattern, four for four. Measured on an RTX 4090 through
//! Vulkan, from the unquantised bf16 release with the int4 encode run at load,
//! in 704 [`Shell::hold`] calls and 47 [`Shell::step`]s.
//!
//! And it is the same answer four ways: prefilled in one 32-row fire, fed one
//! token at a time through 32 decode fires, prefilled in a batch beside a
//! second conversation saying something else, and — since
//! [`a_frame_the_engine_built_answers_what_the_drivers_own_turns_do`] — through
//! a `FrameSubmission` on pages a scheduler chose rather than pages the
//! driver's own book handed out. See
//! [`a_conversation_is_answered_the_same_however_it_reaches_the_driver`] for why
//! that agreement is worth more than any single one of them.
//!
//! # Why induction and not a golden token
//!
//! A test that pins *the model says X* breaks on any numerical change and
//! teaches nothing when it does — the number moves, somebody updates it, and
//! the test has measured a constant rather than a model. What is pinned here is
//! that the model does something a broken forward pass cannot: it copies a
//! sequence it was shown. The six ids are arbitrary and no tokenizer is
//! involved, because induction is a copying circuit and what they SPELL does
//! not matter.
//!
//! # And it agrees with an implementation that shares no code with it
//!
//! [`the_distribution_this_shell_answers_with_is_the_one_an_independent_implementation_states`]
//! holds both rows against `driver-vulkan/tests/device.rs`'s recorded oracle: a
//! numpy forward that reads safetensors directly and dequantizes MLX's 4-bit
//! groups itself, over this same prompt. Different artifact — that file's
//! fixture is the pre-quantised `mlx-community/Qwen3-0.6B-4bit` where this one
//! encodes an unquantised release at load — different arithmetic, no crate in
//! common. Same eight winning ids in both rows, same top-1, and every one of
//! the 26 numbers compared within **0.27** on rows spanning 31 and 37.
//!
//! # What this file does NOT reach, and where each stops
//!
//! **The engine's seam serves a frame, and that is measured rather than read
//! off the source.** [`a_frame_the_engine_built_answers_what_the_drivers_own_turns_do`]
//! builds a `driver_api::FrameSubmission` by hand — the engine's own record,
//! page CSR and all — puts it through [`Shell::launch`], and holds the
//! distribution that comes back against the one [`Shell::step`] produced for
//! the same prompt. Nothing in that path touches the driver's own
//! [`Book`](driver_wgpu::pages::Book): the scheduler's pages are the pages the
//! fire reads, which is the whole reason `launch` exists beside `step`. What is
//! still NOT measured here is `crates/engine`'s side of the seam — the
//! completion broker, the program loop, `pie run` end to end — because this is
//! a driver test and that machinery lives a crate away.
//!
//! **For THIS model the CLI stops one wall earlier than that.** `pie serve`
//! wants an artifact in the model store, and neither route makes a servable one:
//! `pie model build --backend vulkan --quant int4 Qwen/Qwen3-0.6B` and a plain
//! `pie model import` of the same snapshot both end at *this checkpoint matches
//! no model this build serves — qwen3-0.6b: unexpected lm_head*. That is
//! `catalog::identify` on an export that ships a tied head as a real tensor.
//! This file steps around it by taking the row by id, the way
//! `tests/checkpoint.rs` does, which is why it can measure a model the CLI
//! cannot boot.
//!
//! **`~/.cache/pie/models/*.zt` is unreadable by this tree.** Handed the 1.5 GB
//! artifact pie ships for this very model, the loader answers *cannot read
//! .../qwen-3-0.6b.zt: unsupported: unsupported: cannot detect the format*. The
//! file begins `5a 54 45 4e 30 30 30 31` — `ZTEN0001`, ztensor v1 — and the
//! workspace pins `ztensor = "2.1.1"`, whose container is a different frame. The
//! model this repository ships in its own format cannot be read by its own
//! loader, so the HuggingFace snapshot cache is what this measures.
//!
//! **Qwen3.5-0.8B is out of reach, and qwen3-0.6B is the substitution** — stated
//! here rather than made quietly. Its `config.json` declares 24 layers of which
//! **18 are `linear_attention`**.
//!
//! THE REASON GIVEN HERE WAS WRONG, and it is worth saying which part. It read:
//! *"`kernels-wgpu`'s `ssm.rs` rows declare axes and no operands, and
//! `geometry.rs` refuses `Rule::RecurrentScan` as `Ungeometric::Unruled`. A GDN
//! model needs kernels this tree does not have."* The tree HAS those kernels —
//! `ssm` crossed, all eight have routines and arms — and no launch rule is
//! consulted any more, so nothing can be refused as unruled. `tests/hybrid_
//! probe.rs` lowers the hybrid and plans every rectangle: **twelve of fourteen
//! symbols plan, and none is unclaimed by an arm.**
//!
//! What actually blocks it is RECURRENT STATE. `driver-metal` reads
//! `conv_state` from `o.slab(layer, "conv_state")`; this crate's `Resolve` has
//! no slab seam at all, so its GDN arms improvised and read the wrong operands.
//! That is a feature this backend does not have, not a kernel it is missing,
//! and the probe names both blocked symbols so the next attempt starts from a
//! measurement instead of a sentence.
//!
//! # What it costs, and why that is paid rather than trimmed
//!
//! About **ten minutes** on this machine, of which 105 seconds is the load —
//! `model-loader`'s executor encoding 600M bf16 parameters to int4 in a debug
//! build — and the rest is 55 fires whose host-side lowering is also debug. The
//! weights are read once per process and leaked, so that cost is per RUN and
//! not per test; each shell still re-stages its 335 MiB onto the card, which is
//! where the frame test's own two minutes mostly go.
//!
//! The obvious trims are refused. A shorter prompt weakens the induction;
//! sharing one shell between the three feedings lets a stale cache hide; and a
//! test file cannot ask for `--release`. What IS avoided is firing one prompt
//! twice for two claims: the prefilled run is computed once and shared, and the
//! one place a second one is fired
//! ([`a_conversation_is_answered_the_same_however_it_reaches_the_driver`]) is
//! held against the first, so the repetition buys a determinism check.
//!
//! # Why it must skip, and why it must not skip quietly
//!
//! `required-features = ["native"]` keeps this off a build box's compile
//! entirely. On a machine WITH the feature two things can still be absent — an
//! adapter, and a checkpoint — and each prints a line naming which. A test that
//! returned green on a machine with neither would be reporting the absence of a
//! model as the presence of agreement.

#![allow(clippy::print_stdout)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, MutexGuard, OnceLock};

use driver_wgpu::device::Device;
use driver_wgpu::dispatch::Geometry;
use driver_wgpu::shell::{Deployment, Shell, Text};
use driver_wgpu::turns::Turn;
use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::shared::llama_like::forward::llama_like_metal;
use model_ir::trace::FireClass;

/// One device at a time, for the whole suite.
///
/// `tests/device.rs` states the measurement this stands on: with `cargo test`'s
/// default parallelism, ten `wgpu::Device`s open at once wedges roughly one run
/// in three on the NVIDIA proprietary driver. It matters more here than there —
/// each shell holds 335 MiB of weights, and three racing is a card that is out
/// of memory for a reason no message would explain.
///
/// It is taken by [`gpu`] and held by the CALLER, which means a test that
/// forgets the line takes no lock and nothing says so. Five did, and the plain
/// `cargo test -p driver-wgpu --features native` this suite documents died
/// with a SIGSEGV in llvmpipe about three tests in — while every one of the
/// sixteen passed under `--test-threads=1`, which is the shape of a failure
/// that gets called flaky and rerun. `every_test_that_opens_a_shell_holds_the
/// _suites_lock` is the check that would have said.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// A test that builds a [`Shell`] holds the lock while it does.
///
/// The rule cannot be expressed in the type system without threading the guard
/// through [`shelled`] and its sixteen call sites, so it is expressed here:
/// this file is read, and any `#[test]` whose body reaches a device — through
/// `shelled`, `shelled_with` or `opened` — must also call `gpu()`.
///
/// It reads the SOURCE rather than asking the runtime, because the thing being
/// checked is unobservable from inside a passing run: an unlocked test that
/// happens not to overlap a locked one behaves identically to a correct one.
#[test]
fn every_test_that_opens_a_shell_holds_the_suites_lock() {
    let source = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/serving.rs"),
    )
    .expect("this file is readable");
    let lines: Vec<&str> = source.lines().collect();

    let mut unlocked = Vec::new();
    for (n, line) in lines.iter().enumerate() {
        let Some(name) = line.strip_prefix("fn ").map(|r| r.trim_end_matches("() {")) else {
            continue;
        };
        if !lines[..n].iter().rev().take(6).any(|l| *l == "#[test]") {
            continue;
        }
        let end = lines[n..]
            .iter()
            .position(|l| *l == "}")
            .map_or(lines.len(), |o| n + o);
        let body = lines[n..end].join("\n");
        let opens =
            body.contains("shelled(") || body.contains("shelled_with(") || body.contains("opened(");
        if opens && !body.contains("gpu()") {
            unlocked.push(format!("  `{name}` at tests/serving.rs:{}", n + 1));
        }
    }

    assert!(
        unlocked.is_empty(),
        "{} test(s) build a shell without holding `ONE_AT_A_TIME`. Each opens \
         its own `wgpu::Device` and stages 335 MiB, so under `cargo test`'s \
         default parallelism it races whichever test does hold the lock — \
         which is a SIGSEGV on llvmpipe and an out-of-memory with no message \
         on a card. Add `let Some(_held) = gpu() else {{ return }};` as the \
         first line.\n{}",
        unlocked.len(),
        unlocked.join("\n"),
    );
}

/// Arbitrary ids, well inside the vocabulary and away from the special tokens
/// at either end.
///
/// What they SPELL does not matter — induction is a copying circuit — which is
/// the whole reason this needs no tokenizer. The same six
/// `driver-vulkan/tests/device.rs` shows its models, so that the two backends
/// answer one prompt and that file's CPU oracle is an oracle for this one.
const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];

/// The catalog row this file serves.
const MODEL: &str = "qwen3-0.6b";

/// `model.embed_tokens.weight`'s published shape, which is how a snapshot says
/// which model it is.
///
/// `[vocab, hidden]` and not a packed width: the release this reads is the
/// unquantised one, where a `*-4bit` conversion of the same model publishes
/// `[151936, 128]`. Guessing wrong is not an error — it is this file measuring
/// one model's text against another's tensors, which reads like a loader
/// defect.
const EMBED: &[i64] = &[151_936, 1024];

/// How many `model.layers.N.` indices that checkpoint carries.
///
/// The other half of the identification, and not redundant:
/// `tests/checkpoint.rs` needed it the moment discovery started walking thirty
/// cache directories instead of one, because `Qwen/Qwen3-1.7B` embeds the same
/// `[151936, 1024]` and would otherwise be measured against this text.
const LAYERS: usize = 28;

/// How many weights the decode text binds, sidecars included.
///
/// Pinned for the reason `tests/arena.rs` gives at length: a sweep that
/// iterated nothing passes exactly as loudly as one that iterated everything
/// and agreed. If `crates/model` changes the text this moves, and the assertion
/// prints the new number.
const BOUND: usize = 704;

// ---------------------------------------------------------------------------
// The device
// ---------------------------------------------------------------------------

/// An adapter, however this run asks for one.
///
/// `PIE_WGPU_FALLBACK=1` asks for the SOFTWARE adapter, the same knob
/// `tests/device.rs` has. Not a deployment knob and not fast — a 0.6B model
/// through lavapipe is minutes a fire — but it is the difference between "it
/// agrees on the card it was written on" and an answer from a second
/// implementation of the same WGSL.
fn opened() -> Result<Device, driver_wgpu::device::Unavailable> {
    if std::env::var("PIE_WGPU_FALLBACK").is_ok() {
        Device::software()
    } else {
        Device::open()
    }
}

/// The suite's lock, once an adapter has answered at all.
///
/// The probe device is opened and dropped rather than kept, because every shell
/// below opens its own: [`Device`] is deliberately not `Clone` (its error sink
/// is per-device state, and two over one `wgpu::Device` would drain each
/// other's refusals), and [`Shell::on`] takes one by value. Opening an adapter
/// twice in a process is what `shell.rs` calls "legal and slow", and the probe
/// costs milliseconds against a 105-second load.
fn gpu() -> Option<MutexGuard<'static, ()>> {
    let held = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match opened() {
        Ok(device) => {
            println!("adapter: {}", device.name());
            Some(held)
        }
        Err(why) => {
            println!("SKIP: no adapter answered ({why}), so nothing here is measured");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// The weights
// ---------------------------------------------------------------------------

/// The snapshot directories to look in, and whether a person named them.
///
/// `tests/checkpoint.rs`'s rule, kept for its reason: `PIE_CHECKPOINT` first,
/// colon-separated the way a `PATH` is, and the local HuggingFace cache when it
/// is unset — because a skip that could have been a measurement is the failure
/// mode both files are written against.
fn snapshots() -> (Vec<String>, bool) {
    if let Ok(v) = std::env::var("PIE_CHECKPOINT")
        && !v.trim().is_empty()
    {
        return (
            v.split(':')
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect(),
            true,
        );
    }
    (hugging_face_cache(), false)
}

/// Every `models--*/snapshots/*` directory under the local HuggingFace cache.
///
/// `HF_HOME`/`HF_HUB_CACHE` are honoured because a machine that moved its cache
/// did so to a disk with room on it, and a file that only knew `~/.cache` would
/// report "unmeasured" on exactly the machine with the most artifacts.
fn hugging_face_cache() -> Vec<String> {
    let hub = match (std::env::var("HF_HUB_CACHE"), std::env::var("HF_HOME")) {
        (Ok(dir), _) if !dir.is_empty() => std::path::PathBuf::from(dir),
        (_, Ok(home)) if !home.is_empty() => std::path::PathBuf::from(home).join("hub"),
        _ => match std::env::var("HOME") {
            Ok(home) => std::path::PathBuf::from(home).join(".cache/huggingface/hub"),
            Err(_) => return Vec::new(),
        },
    };
    let Ok(repos) = std::fs::read_dir(&hub) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for repo in repos.flatten() {
        if !repo.file_name().to_string_lossy().starts_with("models--") {
            continue;
        }
        let Ok(revisions) = std::fs::read_dir(repo.path().join("snapshots")) else {
            continue;
        };
        for revision in revisions.flatten() {
            out.push(revision.path().to_string_lossy().into_owned());
        }
    }
    // Sorted, so a machine holding two revisions of one repository measures the
    // same one on every run.
    out.sort();
    out
}

/// The snapshot that IS qwen3-0.6B, by what the artifact states rather than by
/// what a path claims.
fn checkpoint() -> Option<String> {
    let (dirs, named) = snapshots();
    for dir in &dirs {
        let meta = match model_loader::checkpoint::read::parse_checkpoint_metadata(
            std::path::Path::new(dir),
        ) {
            Ok(meta) => meta,
            Err(e) => {
                if named {
                    println!("{dir} is not readable as a checkpoint ({e})");
                }
                continue;
            }
        };
        let shape = meta
            .tensors
            .iter()
            .find(|t| t.name == "model.embed_tokens.weight")
            .map(|t| t.shape.clone())
            .unwrap_or_default();
        let layers = meta
            .tensors
            .iter()
            .filter_map(|t| t.name.strip_prefix("model.layers."))
            .filter_map(|rest| rest.split('.').next())
            .filter_map(|n| n.parse::<usize>().ok())
            .collect::<BTreeSet<_>>()
            .len();
        if shape == EMBED && layers == LAYERS {
            return Some(dir.clone());
        }
    }
    println!(
        "no {MODEL} among {} candidate{} in {}, so THE FORWARD PASS COULD NOT BE MEASURED",
        dirs.len(),
        if dirs.len() == 1 { "" } else { "s" },
        if named {
            "PIE_CHECKPOINT"
        } else {
            "the HuggingFace cache"
        },
    );
    None
}

/// The load plan for that snapshot.
///
/// `tests/checkpoint.rs`'s `compiled_plan_for`, trimmed to the one model this
/// file serves and carrying both of its findings:
///
/// * the row is taken **by id**, because `catalog::identify` refuses a stock
///   `Qwen/Qwen3-0.6B` with `unexpected lm_head` — that export ships a tied head
///   as a real tensor, and loosening another crate's manifest from a driver's
///   test is how a refusal that meant something becomes one nobody remembers;
/// * `Binding::MLX_IN_PLACE` is tried FIRST, because it is what a driver boot
///   asks for, and its refusal (`needs quantized weights: this checkpoint
///   carries no .scales tensors`) names the other way in. `RuntimeQuant::Int4`
///   is that way, and taking it is what turns the one qwen3 release on this
///   machine into a measurement instead of a skip.
///
/// The refusal is matched on its TEXT rather than assumed, so a checkpoint
/// refused for some other reason stops here instead of being quietly re-planned
/// under a policy that cannot answer it.
fn plan_for(dir: &str) -> model_loader::plan::LoadPlan {
    let path = std::path::Path::new(dir);
    let meta = model_loader::checkpoint::read::parse_checkpoint_metadata(path)
        .expect("it parsed once already");
    let row = model::catalog::find(MODEL).unwrap_or_else(|| panic!("this build has no `{MODEL}`"));
    let config =
        match model_loader::checkpoint::read::read_meta(&meta, model::encoding::CONFIG_OBJECT) {
            Ok(Some(bytes)) => String::from_utf8(bytes).expect("the embedded config is utf8"),
            _ => std::fs::read_to_string(path.join("config.json"))
                .unwrap_or_else(|e| panic!("{dir}/config.json: {e}")),
        };
    let encoding = model::encoding::Encoding::from_config_json(&config)
        .expect("the config states an encoding");
    // `BackendKind::Vulkan`, and this backend is not Vulkan: it is whichever of
    // Vulkan, Metal and D3D12 the adapter answered on, and there is no `Wgpu`
    // arm because a plan is compiled before an adapter is asked. The engine's
    // own seam compiles against this same arm and says why at length; what a
    // target decides is alignment and tile budget, not what the tensors are
    // called.
    let target = model_loader::plan::StorageTarget::for_backend(
        model_loader::types::BackendKind::Vulkan,
        0,
        1,
    );
    match model::boot::compile_load_plan_for(
        path,
        &meta,
        &target,
        row,
        &encoding,
        model::boot::Binding::MLX_IN_PLACE,
    ) {
        Ok((plan, _)) => {
            println!(
                "{MODEL}: plan through `MLX_IN_PLACE`, {} tensors",
                plan.tensors.len()
            );
            return plan;
        }
        Err(e) => {
            let refusal = e.to_string();
            assert!(
                refusal.contains("needs quantized weights"),
                "{MODEL} was refused for a reason this file does not know how to answer: {refusal}"
            );
            println!(
                "{MODEL}: `MLX_IN_PLACE` refused an unquantised release, so the weights are \
                 encoded at load — the remedy the refusal names"
            );
        }
    }
    let policy = model::shared::policy::Policy {
        projections: model::shared::policy::Projections::InPlace,
        naming: model::shared::policy::Naming::Mlx,
        runtime_quant: model::shared::policy::RuntimeQuant::Int4,
        moe_request: model::shared::policy::Mxfp4MoeRequest::Auto,
        component: model::shared::policy::Component::Full,
        stream_routed_experts: false,
        knobs: model::shared::policy::FamilyKnobs::default(),
    };
    let (contract, _) =
        model::contract::author_with_policy(row, &encoding, &meta, &target, &policy)
            .unwrap_or_else(|e| panic!("the loader would not author `{MODEL}`: {e}"));
    let plan = model_loader::plan::compile(&meta, &contract, target)
        .unwrap_or_else(|e| panic!("the loader would not compile a plan for `{MODEL}`: {e}"));
    // `compile_load_plan_for` runs this and the policy path does not, so it is
    // run here: a snapshot that moved under a plan compiled against it is a
    // refusal, and dropping it would mean the two roads through this function
    // checked different things.
    model_loader::checkpoint::read::verify_declared_files(&plan, path)
        .unwrap_or_else(|e| panic!("the plan for `{MODEL}` names a file that is not there: {e}"));
    println!("{MODEL}: plan compiled, {} tensors", plan.tensors.len());
    plan
}

/// Every weight name this model's decode plan binds.
///
/// The DECODE plan and not the prefill one, because they bind the same set —
/// `tests/checkpoint.rs` asserts that over all six texts — and decode is the
/// plan a driver runs 99 times out of 100.
///
/// A `scale.` marker is left out: it is a constant riding the weight slot rather
/// than a tensor, so no loader publishes one and no binder looks one up.
fn names_a_decode_binds() -> Vec<String> {
    use model_compiler::lower::{Arg, Fire, Row, lower};

    let text = llama_like_metal(&facts(), &backend_facts(), FireClass::Decode);
    let low = lower(
        &text,
        &[Row::default()],
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the plan lowers");
    let names: BTreeSet<String> = low
        .args
        .iter()
        .filter_map(|a| match a {
            Arg::Weight(n) if !n.starts_with("scale.") => Some(n.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        names.len(),
        BOUND,
        "the {MODEL} decode text binds {} weights, not {BOUND}",
        names.len()
    );
    names.into_iter().collect()
}

/// The bytes, under the names a plan states, read once and held for the life of
/// the suite.
///
/// Leaked rather than copied per call: 335 MiB and a 105-second encode, against
/// four tests that all want the same map.
fn weights() -> Option<&'static BTreeMap<String, Vec<u8>>> {
    static HELD: OnceLock<Option<&'static BTreeMap<String, Vec<u8>>>> = OnceLock::new();
    *HELD.get_or_init(|| load().map(|m| &*Box::leak(Box::new(m))))
}

/// The read itself, separated so [`weights`] is only the caching.
///
/// **The loader's own executor, not a read of each tensor's source span.** A
/// verbatim copy is what a `Binding::MLX_IN_PLACE` plan over a pre-quantised
/// repo happens to be, and it is not what this plan is: the encode-at-load
/// policy states hundreds of `TileMap`s that quantize bf16 into affine-U4
/// groups. Reading spans would hand the card raw bf16 under names the kernels
/// read as packed nibbles — which on this backend is not a fault, it is a wrong
/// number. `model_loader::executor::Execution` is a production path (`pie model
/// build` materializes through it), so running the plan is both less code here
/// and the thing a real driver does.
fn load() -> Option<BTreeMap<String, Vec<u8>>> {
    let dir = checkpoint()?;
    println!("measuring {MODEL} at {dir}");
    let began = std::time::Instant::now();
    let plan = plan_for(&dir);
    let storage =
        match model_loader::executor::Execution::new(&plan, std::path::Path::new(&dir)).run() {
            Ok(storage) => storage,
            Err(e) => {
                println!("the loader would not execute `{MODEL}`'s plan: {e}");
                return None;
            }
        };
    let naming = driver_wgpu::names::Naming::mlx();
    let mut out = BTreeMap::new();
    let mut bytes = 0u64;
    for traced in names_a_decode_binds() {
        // `spellings` answers with a LIST, in try order, and with an EMPTY one
        // for a name outside the table's shape. Both are panics here rather than
        // skips: a weight this shell is not given is a symbol the fire cannot
        // bind, and `tests/checkpoint.rs` has already measured that all 704 of
        // these resolve.
        let held = naming
            .spellings(&traced)
            .iter()
            .find_map(|s| storage.tensors.get(s.as_str()))
            .unwrap_or_else(|| panic!("`{traced}` resolves to nothing the loader produced"));
        bytes += held.len() as u64;
        out.insert(traced, held.clone());
    }
    println!(
        "{} weights, {bytes} bytes, staged in {:.1}s",
        out.len(),
        began.elapsed().as_secs_f32()
    );
    Some(out)
}

// ---------------------------------------------------------------------------
// The model, as this driver receives it
// ---------------------------------------------------------------------------

/// The architecture, from `crates/model`'s own fixture.
///
/// Not from the `.toml` beside the checkpoint and not from `config.json`: this
/// crate depends on `model` only as a dev kind, and a driver that read an
/// architecture would be a driver with an opinion about which models exist.
/// What the fixture states — 28 layers, 16 q heads over 8 kv heads, head_dim
/// 128, hidden 1024, vocab 151936, per-head qk-norm, tied embeddings and no qkv
/// bias — is what `~/.cache/pie/models/qwen-3-0.6b.toml` states, line for line.
fn facts() -> LlamaLikeFacts {
    LlamaLikeFacts::qwen3_0_6b()
}

/// The backend facts the text is lowered under.
///
/// `tests/arena.rs` and `tests/checkpoint.rs` build the same pair and say why:
/// `synthetic()` is `driver-metal`'s answer sheet, and this backend disagrees
/// with it on exactly one line, `add_bias`, because that driver's binder does
/// not resolve `Source::OutWidth` and this one does.
///
/// It has to be the same pair the loader was asked for. `add_bias` decides
/// whether the text states three bias weights a layer; qwen3-0.6B publishes
/// none, so the two agree here — but a shell whose text and whose weight list
/// came from different facts meets an unbound symbol at dispatch time, which is
/// a failure with no useful message.
fn backend_facts() -> LlamaLikeMetalFacts {
    LlamaLikeMetalFacts {
        add_bias: true,
        // MIRRORS THE DEPLOYMENT STAMP: `engine::driver::backend::wgpu`
        // sets this `false` because the tiled GEMM does not read back what
        // `cast_qmm_input` staged. A fixture that left `synthetic()`'s `true`
        // here would put eleven real-model tests -- every one of which had
        // SKIPPED for want of a checkpoint, and so had never said anything --
        // on the broken path, and they fail exactly there:
        // `a_weight_this_shell_was_never_given_is_a_different_answer`'s
        // control reads argmax 220 where it states 88204.
        qmm_fp16_precast: false,
        // AND MIRRORS THE OTHER HALF OF THAT STAMP, which it did not until
        // the tile sweep below was re-run and caught the divergence.
        //
        // `engine::driver::backend::wgpu` states `qmm_tile: Some((32, 64))`.
        // This fixture took `synthetic()`'s `(32, 32)`, so every timing in
        // this file -- the prefill tables, the per-rectangle attribution, the
        // whole `PIE_TX` sweep -- was measured on a GEMM THIS BACKEND DOES
        // NOT SHIP. At 512 rows the two read 464.3 ms and 376.3: the notebook
        // was 23% pessimistic about its own product, and every ratio in it
        // was taken against a denominator nobody runs.
        //
        // The comment above this one is the whole argument for why that
        // matters and it was already written, one field earlier, about a
        // fixture that put eleven tests on a path the deployment does not
        // take. The same reasoning reaches `qmm_tile` and nobody carried it
        // across. A fixture is only an answer sheet if it answers the
        // question the shipping build asks.
        qmm_tile: (32, 64),
        ..LlamaLikeMetalFacts::synthetic()
    }
}

/// A shell serving qwen3-0.6B on its own adapter, with its weights held.
///
/// The four pieces [`Shell::on`] checks against each other are assembled here
/// rather than derived, for the reason `shell::Text` states: deriving assumes
/// one set of facts went in and cannot notice when two did. Every number below
/// comes off [`facts`], so a cache shaped for the wrong model is impossible
/// rather than merely unlikely — and a cache shaped wrong is not refused, it is
/// a page whose rows are read at the wrong stride, which still fires and still
/// returns finite logits.
fn shelled(real: &BTreeMap<String, Vec<u8>>, pages: u32) -> Shell {
    shelled_with(real, pages, false)
}

/// [`shelled`], optionally with the DECODE plan in the prefill slot.
///
/// `Serving` picks a plan by row count, so a many-row fire always takes the
/// prefill text and its tiled GEMM. Putting the decode text there instead is
/// what lets the same rows be fired through the matvec kernels -- the only way
/// this harness can ask whether the two families agree, which is
/// `driver-vulkan`'s `the_tiled_gemm_answers_the_way_the_vector_kernel_does`
/// asked here.
fn shelled_with(real: &BTreeMap<String, Vec<u8>>, pages: u32, vector: bool) -> Shell {
    shelled_tuned(real, pages, vector, None)
}

/// [`shelled_with`], optionally overriding the GEMM tile the text is lowered
/// at.
///
/// `project::QMM_TILE` is a SHARED constant, and its doc says it was chosen
/// against a cooperative-matrix build: *"32 rather than 16 because 16 has no
/// cooperative-matrix build to compile into."* This backend stamps no such
/// build, so that reason cannot apply here and the value it produced is
/// inherited rather than measured. The same doc already names the remedy —
/// *"a backend wanting its own tile has `qmm_tile` on `MetalBinding`"* — so
/// this parameter exists to find out whether this one does, before anybody
/// sets it.
///
/// (It says "stamps no such build" and not "cannot": the adapter under this
/// suite offers `EXPERIMENTAL_COOPERATIVE_MATRIX` at 16x16x16, so the shared
/// constant's reason may yet become this backend's reason too. See
/// `kernels-wgpu`'s
/// `whether_this_adapter_offers_the_cooperative_matrix_this_tree_calls_absent`.)
///
/// RETIRED WITH `kernels-wgpu`'s TEST TREE. That name is a record of a
/// measurement now, not a live proof: the crate lost `tests/` and every
/// in-file `mod tests` when the three shader planes moved their numbers to
/// the fire that reads them, and nothing in this workspace re-runs it. What
/// it reported is still why the sentence above says what it says; what is
/// gone is the thing that would notice if it stopped being true.
fn shelled_tuned(
    real: &BTreeMap<String, Vec<u8>>,
    pages: u32,
    vector: bool,
    tile: Option<(u32, u32)>,
) -> Shell {
    let facts = facts();
    let backend = match tile {
        Some(qmm_tile) => LlamaLikeMetalFacts {
            qmm_tile,
            ..backend_facts()
        },
        None => backend_facts(),
    };
    shelled_facts(real, pages, vector, &facts, &backend)
}

/// [`shelled_tuned`], with both fact sets handed in rather than derived.
///
/// A caller varying `qmm_partial_rows` needs this: that fact decides the
/// projections' guard, and a guard that refuses the fire turns a comparison
/// of two kernel families into a comparison of one with itself.
fn shelled_facts(
    real: &BTreeMap<String, Vec<u8>>,
    pages: u32,
    vector: bool,
    facts: &LlamaLikeFacts,
    backend: &LlamaLikeMetalFacts,
) -> Shell {
    let facts = facts.clone();
    let backend = backend.clone();
    let text = Text {
        decode: llama_like_metal(&facts, &backend, FireClass::Decode),
        prefill: llama_like_metal(
            &facts,
            &backend,
            if vector {
                FireClass::Decode
            } else {
                FireClass::Prefill
            },
        ),
        geometry: Geometry {
            q_heads: facts.q_heads,
            kv_heads: facts.kv_heads,
            head_dim: facts.head_dim,
            // Qwen3 rotates the whole head. A partial rope would be a rotation
            // over part of a head and an identity over the rest, which is finite
            // and wrong.
            rotary_dims: facts.head_dim,
            n_experts: 0,
            experts_per_token: 0,
            ..Default::default()
        },
        layers: facts.layers as u16,
    };
    let device = opened().expect("an adapter answered once already");
    let mut shell = Shell::on(
        device,
        text,
        Deployment {
            pages,
            // 1e6, which `~/.cache/pie/models/qwen-3-0.6b.toml` and the HF
            // `config.json` both state and which `Deployment::default` happens
            // to agree with. Written out because a rope base wrong by a factor
            // of a hundred does not fault — it attends at the wrong wavelengths
            // and stays fluent.
            theta: 1_000_000.0,
            ..Deployment::default()
        },
    )
    .unwrap_or_else(|e| panic!("the shell: {e}"));
    for (name, bytes) in real {
        shell
            .hold(name, bytes)
            .unwrap_or_else(|e| panic!("`{name}` would not stage: {e}"));
    }
    shell
}

/// **A measurement in the dev profile is a measurement of the dev profile.**
///
/// `cargo test` builds `dev` and this workspace sets no `[profile.dev]`
/// opt-level, so a timing taken the obvious way is host-arithmetic-bound in a
/// way the shipped binary never is. Three conclusions on this branch were
/// published off debug timings before anybody noticed, and they went three
/// different ways: the llama.cpp gap was overstated 2-4x, the batching lever
/// was understated by half, and a kernel comparison came out backwards. **A
/// debug number is not a slower version of a release number, it is a different
/// measurement.**
///
/// So the measurement tests refuse rather than mislead. This is why they are
/// all `#[ignore]`d as well: nothing in the gate runs them, and the only way
/// to reach one is to ask for it, at which point being told the profile is
/// wrong costs a rerun instead of a retraction.
fn release_only() {
    // `if` rather than `assert!`: in release the condition folds to a
    // constant and `clippy::assertions_on_constants` is right that asserting
    // one says nothing. The refusal is the point, not the assertion.
    if cfg!(debug_assertions) {
        panic!(
            "this is a measurement and `cargo test` builds the dev profile; \
             rerun it with `--release`"
        );
    }
}

/// The prompt: [`PERIOD`] five times, then the first two of a sixth repeat, so
/// the next token the model should want is `PERIOD[2]`.
///
/// THIRTY-TWO, and the length is not free. `geometry.rs` refuses a prefill whose
/// rows are not a whole number of 16-row tiles — the tiled GEMM is compiled at
/// `bm = 16` and a driver may not pad a fire it did not author — so a caller
/// above this crate owes the batching. Recorded here rather than worked around.
fn prompt() -> Vec<u32> {
    let mut prompt: Vec<u32> = Vec::new();
    for _ in 0..5 {
        prompt.extend_from_slice(&PERIOD);
    }
    prompt.push(PERIOD[0]);
    prompt.push(PERIOD[1]);
    assert_eq!(
        prompt.len() % 16,
        0,
        "the tiled GEMM takes whole 16-row tiles"
    );
    prompt
}

/// The highest-scoring id of a distribution.
fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("a non-empty distribution")
        .0 as u32
}

/// How far apart two routes to THE SAME ROW are allowed to be.
///
/// # Why this is not zero
///
/// Because a row's logits depend on how many OTHER rows were fired beside it,
/// and that is the matvec's, not the cache's. `quant/qmv.wgsl`'s `reduce_store`
/// gives one workgroup `PIE_MT` = 2 activation rows, so every row of an even
/// batch is summed by `block_dot2` and the tail of an odd one by `block_dot1`.
/// The two bodies multiply the same products into the same accumulators in the
/// same order and STILL do not agree: the two-row body holds twice the
/// accumulators live across the unpack, so the backend contracts a different
/// subset of its `a + b * c` into fused multiply-adds.
///
/// Fired directly at the kernel, the two arms part by two bf16 ulps on about
/// five outputs in a hundred thousand, and only at a projection as wide as an
/// lm head. Twenty-eight layers turn that into 139561 of 151936 logits
/// differing by **0.79% of the row's peak**, argmax unchanged — which is what
/// these tests were reading when they demanded channel-exact equality and got
/// it from every row but the last.
/// [`a_seats_answer_does_not_depend_on_how_many_seats_fired_with_it`] has the
/// sweep and the table.
///
/// # Why 2% and not 0.79%
///
/// Headroom for a different prompt and a different batch, and still an order of
/// magnitude under anything that has ever been a real defect here: the staged
/// fp16 GEMM this suite caught missed by 120% of the peak, and the guard that
/// let a wrong kernel fire missed by 155%. A tolerance is only worth having if
/// the failures it would have caught are the ones that happened.
const BATCH_SPREAD: f32 = 0.02;

/// Two routes to the same row answered the same thing.
///
/// The argmax EXACTLY -- a difference small enough to be the batch's is small
/// enough not to reorder the peak -- and every channel to within
/// [`BATCH_SPREAD`] of the row's peak. Peak-relative and not per element, for
/// the reason `worst_disagreement_with` states: a logit near zero has no scale
/// of its own, and a relative test on it reports a hundred percent for a
/// difference of nothing.
fn answers_the_same(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: rows of different widths");
    assert_eq!(
        argmax(a),
        argmax(b),
        "{what}: different argmax, which no amount of batching accounts for"
    );
    let peak = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= BATCH_SPREAD * peak,
        "{what}: {:.2}% of the row's peak ({peak}), where the matvec's own \
         batch spread is 0.79%",
        100.0 * worst / peak
    );
}

/// Two routes that must NOT answer the same thing.
///
/// The mirror of [`answers_the_same`], and every control in this file needs it:
/// a control that only asks for a differing BIT is satisfied by the batch
/// spread above, which would leave the claim untested by a test that passes.
fn answers_differently(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: rows of different widths");
    let peak = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst > BATCH_SPREAD * peak,
        "{what}: the two rows are within {:.2}% of the peak ({peak}), which is \
         inside what one fire's own batching costs, so this comparison cannot \
         see what it is for",
        100.0 * worst / peak
    );
}

// ---------------------------------------------------------------------------
// The three feedings
// ---------------------------------------------------------------------------

/// How the prompt reaches the model.
///
/// Three ways to say the same thing to a server, which a server is entitled to
/// assume mean the same thing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Feeding {
    /// One step, the whole prompt, which is what a server does with a new
    /// conversation.
    Prefilled,
    /// One step per token, which is what a server does with a conversation it is
    /// already decoding — and which fires the DECODE plan for every position
    /// rather than the prefill plan once.
    OneAtATime,
    /// Prefilled, but with a second conversation in every batch.
    Alongside,
}

/// What a run of [`continued`] produced.
///
/// The distributions come back beside the tokens because one caller checks the
/// whole of them against an independent implementation and the others check only
/// which token won. Reading them off the same run they all use is the point: a
/// separate helper that fired its own prompt would be a second setup to drift.
struct Continuation {
    /// The four tokens, greedily.
    tokens: Vec<u32>,
    /// Every logit of the row the FIRST fire answered — the last row of the
    /// prompt, before anything was fed back.
    first: Vec<f32>,
    /// Every logit of the row the fire AFTER that answered: one token, fed back,
    /// through the decode plan and against a cache the prefill wrote.
    ///
    /// Separate from `first` because they are not the same claim. A prefill row
    /// is computed from tokens the same fire attended over; a decode row is
    /// computed from a cache written by an earlier fire, which is where paging,
    /// the page table and the cache's layout enter — none of which a
    /// prefill-only comparison can reach.
    second: Vec<f32>,
}

/// The four tokens this model produces after being shown [`PERIOD`] five times,
/// fed the given way.
///
/// Returns rather than asserts, because what the callers compare is the three
/// ways against EACH OTHER as much as against the pattern.
fn continued(real: &BTreeMap<String, Vec<u8>>, how: Feeding) -> Continuation {
    // EIGHT PAGES of sixteen rows. `Alongside` seats two conversations at once —
    // 35 rows and 19, which is five pages — and a pool sized to the exact need
    // would make the batched run the one that could not be added to.
    let mut shell = shelled(real, 8);
    let prompt = prompt();

    // The second conversation: sixteen tokens so the batch stays a whole number
    // of tiles, and deliberately NOT the pattern. A distraction that agreed with
    // A would not distinguish a shared cache from a private one.
    let other: Vec<u32> = (0..16).map(|i| 5_000 + i * 37).collect();

    let mut fires = 0usize;
    let mut widest = 0usize;
    let mut first: Vec<f32> = Vec::new();
    // A cell rather than a plain local: the closure holds it for the whole run,
    // and the loop below reads it BETWEEN calls.
    let latest: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
    // The row read is the caller's, absolute within the batch. A is put SECOND
    // in every mixed batch below on purpose: a conversation whose answer
    // depended on where in the batch it sat would be a driver that could not be
    // given work in the order it arrived, and A-first would leave A's rows at
    // index 0 either way and never say so.
    let mut fire = |turns: &[Turn], a_rows: usize| -> u32 {
        let step = shell.step(turns).unwrap_or_else(|e| panic!("{e}"));
        // The premise of reading row `a_rows - 1`: a batch that came back
        // narrower than the turns asked for would have this reading someone
        // else's distribution, or reading past the end.
        assert_eq!(
            step.rows,
            turns.iter().map(|t| t.tokens.len()).sum::<usize>(),
            "the fire answered a different number of rows than the turns state"
        );
        fires += 1;
        widest = widest.max(step.rows);
        // THROUGH `readout_of`, NOT BY FIRE ROW. This read used to slice
        // `logits.values` at `(a_rows - 1) * vocab` -- the absolute fire row --
        // and that stopped being where A's distribution is when the readback
        // was narrowed: `Logits` now holds one row per SAMPLED row, and
        // `turns.rs` puts `readout_of` through `frame.sampling_indices` for
        // exactly this reason. `serve.rs`'s `Logits::rows` doc names this call
        // site as one of the five that index by fire row and would each need
        // reading. It had never run to say so -- the checkpoint was absent --
        // and the first run panicked with `range start index 4710016 out of
        // range for slice of length 151936`, which is 31 rows past a readback
        // holding one.
        //
        // A IS THE LAST TURN IN EVERY BATCH THIS HELPER FIRES, which is the
        // premise that makes `readout_of.last()` A's row, and `a_rows` is what
        // states it: the caller passes A's last row's absolute index, so on a
        // batch where A is last that number IS the batch's row count.
        assert_eq!(
            a_rows, step.rows,
            "this helper reads the LAST turn's distribution, so A has to be \
             the last turn"
        );
        let row = step
            .logits
            .row(*step.readout_of.last().expect("a readout row"))
            .expect("the readout row");
        if first.is_empty() {
            first = row.to_vec();
        }
        *latest.borrow_mut() = row.to_vec();
        argmax(row)
    };

    let mut got = Vec::new();
    match how {
        Feeding::Prefilled => {
            got.push(fire(
                &[Turn {
                    who: 1,
                    tokens: prompt.clone(),
                }],
                prompt.len(),
            ));
        }
        Feeding::OneAtATime => {
            // Every position through the DECODE plan, one fire each. Only the
            // LAST one's distribution has seen the whole prompt, so the earlier
            // answers are read and dropped — reading them at all is the point,
            // since a fire whose distribution nobody reads is a fire whose arena
            // could have been anything.
            let mut answer = 0;
            for t in &prompt {
                answer = fire(
                    &[Turn {
                        who: 1,
                        tokens: vec![*t],
                    }],
                    1,
                );
            }
            got.push(answer);
        }
        Feeding::Alongside => {
            got.push(fire(
                &[
                    Turn {
                        who: 2,
                        tokens: other.clone(),
                    },
                    Turn {
                        who: 1,
                        tokens: prompt.clone(),
                    },
                ],
                other.len() + prompt.len(),
            ));
        }
    }

    // Three more, each fed back, so the decode plan and the cache carry the
    // pattern forward rather than the prefill answering everything.
    let mut second: Vec<f32> = Vec::new();
    for (round, filler) in other.iter().take(3).enumerate() {
        let fed = *got.last().expect("a token");
        let (turns, at) = if how == Feeding::Alongside {
            (
                vec![
                    Turn {
                        who: 2,
                        // Whatever B says, as long as it is not what A says.
                        tokens: vec![*filler],
                    },
                    Turn {
                        who: 1,
                        tokens: vec![fed],
                    },
                ],
                2,
            )
        } else {
            (
                vec![Turn {
                    who: 1,
                    tokens: vec![fed],
                }],
                1,
            )
        };
        got.push(fire(&turns, at));
        if round == 0 {
            second = latest.borrow().clone();
        }
    }
    // THE PREMISES, checked after the fact because they are about the whole run
    // rather than any one fire. Without them a helper that quietly ignored `how`
    // would make the comparison between the three ways vacuous — three identical
    // runs agree perfectly.
    match how {
        Feeding::Prefilled => {
            assert_eq!(fires, 4, "one prefill and three decodes");
            assert_eq!(widest, 32, "the prefill was not one fire");
        }
        Feeding::OneAtATime => {
            assert_eq!(fires, 35, "thirty-two single tokens and three decodes");
            assert_eq!(widest, 1, "something was fed more than one token");
        }
        Feeding::Alongside => {
            assert_eq!(fires, 4, "one prefill and three decodes");
            assert_eq!(widest, 48, "the second conversation was not in the batch");
        }
    }
    // The premise of the decode comparison: a row of the wrong width, or one no
    // fire ever wrote, would be held against the oracle as zeros and read as a
    // driver that answers nothing.
    assert_eq!(
        second.len(),
        first.len(),
        "the decode row is not the width of the prefill's"
    );
    Continuation {
        tokens: got,
        first,
        second,
    }
}

/// The prefilled run, fired once for the whole process.
///
/// Two tests want it and it is 25 seconds of card time. Sharing it is not a
/// weakening: the one thing a second identical run would prove — that the shell
/// answers the same twice — is asserted in
/// [`a_conversation_is_answered_the_same_however_it_reaches_the_driver`], which
/// fires its own and holds it against this one.
///
/// The caller must hold [`ONE_AT_A_TIME`]: this opens a device.
fn prefilled() -> Option<&'static Continuation> {
    static HELD: OnceLock<Option<&'static Continuation>> = OnceLock::new();
    *HELD.get_or_init(|| {
        let real = weights()?;
        Some(&*Box::leak(Box::new(continued(real, Feeding::Prefilled))))
    })
}

// ---------------------------------------------------------------------------
// The oracle
// ---------------------------------------------------------------------------

/// A CPU forward's answer to the [`PERIOD`] prompt, in enough detail to hold a
/// whole distribution against and not so much that it is a golden file.
///
/// **Not this repository's arithmetic.** Copied from
/// `driver-vulkan/tests/device.rs`, where it is recorded as the output of a
/// numpy forward that reads safetensors directly and dequantizes MLX's 4-bit
/// groups itself — no code, no kernel and no crate in common with this one, and
/// a different artifact besides: that file's fixture is the pre-quantised
/// `mlx-community/Qwen3-0.6B-4bit` where this one encodes an unquantised release
/// at load.
///
/// Eight ranked ids and five fixed indices rather than 151_936 logits: a golden
/// vector of the whole row would be a file nobody could check by reading, and
/// the two things worth pinning are WHICH tokens win and whether the numbers
/// away from the peak are the same numbers.
struct Oracle {
    /// The eight highest-scoring ids, in order.
    top: &'static [u32],
    /// Their logits, by the same index.
    vals: &'static [f32],
    /// The logits at [`PROBE`] — chosen for being spread across the vocabulary
    /// and nothing else. Away from the peak, so a driver that got the argmax
    /// right by luck does not.
    probe: &'static [f32],
    /// The row's whole range, which no single logit states.
    span: f32,
}

/// The ids [`Oracle::probe`] states.
const PROBE: [usize; 5] = [0, 1_000, 50_000, 100_000, 151_935];

/// What numpy says the last row of the prompt is.
const PREFILL: Oracle = Oracle {
    top: &[88_204, 33_032, 62_949, 14, 78_329, 42_746, 57_428, 17_521],
    vals: &[
        20.8004, 15.7309, 15.5539, 15.2734, 15.2257, 14.8423, 14.4461, 14.2924,
    ],
    probe: &[6.3329, -2.3533, -1.5004, 2.0615, 0.1445],
    span: 31.192,
};

/// ...and of the prompt with that answer appended, which is the row this
/// driver's first DECODE fire produces.
const DECODE: Oracle = Oracle {
    top: &[
        6_100, 16_997, 25_948, 18_062, 6_094, 20_405, 101_203, 65_069,
    ],
    vals: &[
        23.9178, 16.0206, 15.9419, 15.8716, 15.5314, 15.1679, 14.9816, 14.824,
    ],
    probe: &[7.1273, -1.677, 0.4846, 0.6105, 0.953],
    span: 36.8039,
};

/// How far a logit here may be from what numpy says.
///
/// **A budget for two 4-bit ENCODINGS disagreeing, not for this backend's
/// arithmetic.** The oracle quantised a different artifact with a different
/// quantizer, so the two never had the same weights. Measured, the worst of the
/// 26 numbers compared below is 0.27 — id 101_203 in the decode row, 14.98
/// against 15.25 — on rows spanning 31 and 37. Half a logit leaves room for that
/// and is still a real constraint: a row read from the wrong cache page, or a
/// layer whose scales were bound where its zero points belong, is out by tens.
const SLACK: f32 = 0.5;

/// The eight highest-scoring ids of a row, with their logits.
fn ranked(row: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|a, b| row[*b].total_cmp(&row[*a]));
    (
        order.iter().take(8).map(|i| *i as u32).collect(),
        order.iter().take(8).map(|i| row[*i]).collect(),
    )
}

/// Print what a row says, in the shape [`Oracle`] states it.
fn describe(what: &str, row: &[f32]) {
    let (top, vals) = ranked(row);
    let lo = row.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probe: Vec<f32> = PROBE.iter().map(|i| row[*i]).collect();
    println!("{what}: top {top:?}");
    println!("{what}: vals {vals:.4?}");
    println!("{what}: probe {probe:.4?}, span {:.4}", hi - lo);
}

/// One row against the oracle, matched by id rather than by rank.
///
/// **By id**, because the two encodings do not agree about the ORDER of near
/// ties: the prefill row's second and third places are 15.7309 and 15.5539 to
/// numpy and 15.5000 and 15.5625 here, which is a swap and is not a defect. What
/// has to hold is that the same eight tokens win and that each one's logit is
/// the same number — a distribution that ranked some OTHER id second is one this
/// comparison rejects.
fn agrees(what: &str, oracle: &Oracle, row: &[f32]) {
    assert_eq!(row.len(), 151_936, "{what}: the row is not the vocabulary");
    assert!(
        row.iter().all(|v| v.is_finite()),
        "{what}: the row is not finite"
    );
    let (top, _) = ranked(row);
    assert_eq!(
        top[0], oracle.top[0],
        "{what}: this driver's most likely token is not the oracle's"
    );
    assert_eq!(
        top.iter().copied().collect::<BTreeSet<u32>>(),
        oracle.top.iter().copied().collect::<BTreeSet<u32>>(),
        "{what}: a different eight tokens win"
    );
    let mut worst = 0.0f32;
    let mut worst_at = String::new();
    let mut check = |at: String, want: f32, got: f32| {
        let off = (want - got).abs();
        if off > worst {
            worst = off;
            worst_at.clone_from(&at);
        }
        assert!(
            off <= SLACK,
            "{what}: {at} is {got} here and {want} to an independent implementation, which is \
             {off} apart and past the {SLACK} two 4-bit encodings are allowed"
        );
    };
    for (id, want) in oracle.top.iter().zip(oracle.vals) {
        check(format!("id {id}"), *want, row[*id as usize]);
    }
    for (at, want) in PROBE.iter().zip(oracle.probe) {
        check(format!("the logit at {at}"), *want, row[*at]);
    }
    let lo = row.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    check("the span".to_owned(), oracle.span, hi - lo);
    println!("{what}: worst disagreement {worst:.4} ({worst_at}), budget {SLACK}");
}

// ---------------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------------

/// **The headline: a real model, through this shell, continues what it was
/// shown.**
///
/// Four tokens, greedily, from a 32-row prefill and three one-token decodes
/// against the cache that prefill wrote. The claim is not that the model says
/// any particular thing — it is that it copies a sequence it was given, which a
/// broken forward pass does not do and a random one does with probability
/// 151_936 to the fourth.
///
/// [`a_weight_this_shell_was_never_given_is_a_different_answer`] is the control
/// that says this check has teeth.
#[test]
fn a_real_model_continues_a_pattern_it_was_shown() {
    let Some(_held) = gpu() else { return };
    let Some(run) = prefilled() else { return };
    describe("prefill", &run.first);
    describe("decode", &run.second);
    assert_eq!(
        run.tokens,
        PERIOD[2..].to_vec(),
        "the model was shown {PERIOD:?} five times and did not continue it"
    );
}

/// The same conversation, said three ways, answered the same way.
///
/// # The claim a server actually needs
///
/// The test above proves one conversation alone, prefilled. A server never runs
/// that. It runs conversations in batches it did not choose, at row counts that
/// change every step, and it is entitled to assume that a conversation's answer
/// is its own.
///
/// So this fires the same prompt three ways and requires one answer:
///
///   - **prefilled**, thirty-two rows in one fire;
///   - **one token at a time**, thirty-two fires through the DECODE plan — a
///     different plan, different kernels, and a KV cache written one row per
///     fire instead of thirty-two at once;
///   - **alongside** a second conversation that shares every batch, every fire,
///     the same arena and the same cache, and says something else.
///
/// # What each one can catch that the others cannot
///
/// The one-at-a-time run is the prefill/decode equivalence. The two plans state
/// different matmuls above sixteen rows — `affine_qmv_fast` against
/// `affine_qmm_t_..._bm_16_bn_32`, the tiled GEMM — and different attention
/// paths, and nothing before this held their ANSWERS against each other on a
/// real model across a real cache.
///
/// The alongside run is page ownership and per-row positions. A batch that let
/// one conversation read another's pages, or that gave row 0 row 32's position,
/// still fires, still records, and still returns finite logits.
///
/// Neither is a claim any single fire can make. Together they are what a KV
/// cache written at the wrong offset, a page table misread and a positional
/// embedding applied at the wrong index all fail.
///
/// # The controls
///
/// A's turn is put SECOND in every mixed batch, so its rows do not begin at zero
/// and "the same answer" is not the same offset read twice.
///
/// And the prefilled arm is fired AGAIN here rather than taken from
/// [`prefilled`], which costs 25 seconds and buys the only determinism check in
/// this file: two shells, two adapters, two uploads of the same 335 MiB, one
/// answer.
#[test]
fn a_conversation_is_answered_the_same_however_it_reaches_the_driver() {
    let Some(_held) = gpu() else { return };
    let Some(real) = weights() else { return };
    let want = PERIOD[2..].to_vec();

    let alone = continued(real, Feeding::Prefilled).tokens;
    println!("prefilled:     {alone:?}");
    let stepped = continued(real, Feeding::OneAtATime).tokens;
    println!("one at a time: {stepped:?}");
    let batched = continued(real, Feeding::Alongside).tokens;
    println!("alongside:     {batched:?}");

    assert_eq!(alone, want, "prefilled");
    assert_eq!(stepped, want, "one token at a time");
    assert_eq!(batched, want, "alongside a second conversation");
    // The determinism half, and it is about a shell rather than about a feeding:
    // the run above and the one this file cached are two openings of two
    // adapters with two uploads of the same weights.
    if let Some(cached) = prefilled() {
        assert_eq!(
            alone, cached.tokens,
            "two shells over one checkpoint answered differently"
        );
    }
}

/// The whole distribution, against an implementation that shares no code with
/// this one.
///
/// The three feedings agreeing says this driver is consistent with itself, and a
/// driver can be consistently wrong: every path could read the cache at the same
/// wrong stride. This is the other side of that — 26 numbers a row, held against
/// a numpy forward recorded in `driver-vulkan/tests/device.rs` over the same
/// prompt.
///
/// **Both rows**, and they are not the same claim. The prefill row is computed
/// from tokens the same fire attended over; the decode row is computed from a
/// cache an earlier fire wrote, which is where paging, the page table and the
/// cache's layout enter. An oracle held only against the prefill would pass on a
/// driver whose KV writes went to the wrong page.
///
/// See [`SLACK`] for what the tolerance is a budget for, which is not this
/// backend's arithmetic.
#[test]
fn the_distribution_this_shell_answers_with_is_the_one_an_independent_implementation_states() {
    let Some(_held) = gpu() else { return };
    let Some(run) = prefilled() else { return };
    agrees("the prefill row", &PREFILL, &run.first);
    agrees("the decode row", &DECODE, &run.second);
}

/// The control: change the weights and the numbers change.
///
/// # Why this file needs it
///
/// Everything above would pass on a shell that ignored [`Shell::hold`] entirely
/// and answered out of some other source — "the model continues the pattern" is
/// a claim about the OUTPUT, and nothing in it says the output came from the
/// bytes that were staged. This is the negative half, in two stages, and the
/// first stage's measurement is why there are two.
///
/// # Stage one: a middle layer's attention landing
///
/// `layer.13.o_proj` zeroed — and it is a packed affine-U4 tensor, so zeroing it
/// does not zero the weights, it pins every one of them to its group's zero
/// point, which is exactly the kind of plausible-and-wrong a driver has to be
/// able to tell from correct.
///
/// **The argmax does not move**, and that is the finding rather than a
/// disappointment. 151_248 of 151_936 logits change and the worst of them moves
/// by 4.95, the second and fifth places swap and a new id enters the top eight —
/// but `88204` leads by five logits and one of twenty-eight attention landings
/// is not five logits. So this asserts what it actually measures, which is that
/// the ROW depends on the staged bytes, and the induction check above is not
/// weakened by learning that it is robust.
///
/// A control that had asserted the token flips would have gone red here, and
/// the honest repair was to measure what the perturbation does rather than to
/// pick a bigger hammer and keep the assertion.
///
/// # Stage two: the table the head reads
///
/// So a bigger hammer, second and separately: `embed` is the input table AND the
/// output head (this model is tied), and zeroing it is a model that cannot see
/// its prompt or score its vocabulary. The continuation breaks, which is the
/// claim stage one is too gentle to make.
///
/// # The third thing this measures
///
/// Every fire is the same shell and each stage uses a FRESH conversation id. So
/// this also says a weight can be replaced under a live shell and the next fire
/// sees the new one — which is [`Shell::hold`]'s documented behaviour
/// ("replacing whatever was there") and was otherwise unmeasured on a real
/// model.
#[test]
fn a_weight_this_shell_was_never_given_is_a_different_answer() {
    let Some(_held) = gpu() else { return };
    let Some(real) = weights() else { return };
    let mut shell = shelled(real, 8);
    let prompt = prompt();

    // Three conversations of 32 rows, two pages each, in a pool of eight.
    let fire = |shell: &mut Shell, who: u64| -> Vec<f32> {
        let step = shell
            .step(&[Turn {
                who,
                tokens: prompt.clone(),
            }])
            .unwrap_or_else(|e| panic!("{e}"));
        step.logits
            .row(step.readout_of[0])
            .expect("the turn's own row")
            .to_vec()
    };
    let zero = |shell: &mut Shell, name: &str| {
        let bytes = real
            .get(name)
            .unwrap_or_else(|| panic!("`{name}` is not a name this text binds"));
        shell
            .hold(name, &vec![0u8; bytes.len()])
            .expect("the zeroed weight");
    };

    let before = fire(&mut shell, 1);
    let was = argmax(&before);
    assert_eq!(
        was, PERIOD[2],
        "the control's own prefill did not continue the pattern"
    );

    zero(&mut shell, "layer.13.o_proj");
    let row = fire(&mut shell, 2);
    let moved = before
        .iter()
        .zip(&row)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let held = before.iter().zip(&row).filter(|(a, b)| a == b).count();
    println!(
        "`layer.13.o_proj` zeroed: argmax {} (was {was}), worst logit moved {moved:.4}, {held} of \
         {} logits unchanged",
        argmax(&row),
        row.len()
    );
    describe("that row", &row);
    assert!(
        row.iter().all(|v| v.is_finite()),
        "a zeroed projection made the row non-finite, which is a different failure from the one \
         this control is about"
    );
    // ONE, against a measured 4.95. A `> 0.0` floor would pass on a fire that
    // moved a single logit by a bf16 ulp, which is what a partially-applied
    // `hold` would look like; the point of the number is that a whole
    // projection's worth of the row moved.
    assert!(
        moved > 1.0,
        "one of the 704 weights was zeroed and the widest a logit moved was {moved}, so this \
         shell is not reading what was staged into it"
    );

    zero(&mut shell, "embed");
    let row = fire(&mut shell, 3);
    println!("...and `embed` too: argmax {}", argmax(&row));
    describe("that row", &row);
    assert!(
        row.iter().all(|v| v.is_finite()),
        "a zeroed embedding table made the row non-finite, which is a different failure from the \
         one this control is about"
    );
    assert_ne!(
        argmax(&row),
        was,
        "the table this model reads its prompt through AND scores its vocabulary with was zeroed \
         and the answer did not change, so nothing above measures what was staged"
    );
}

// ---------------------------------------------------------------------------
// The engine's own path
// ---------------------------------------------------------------------------

/// A frame the ENGINE built, on pages the ENGINE chose, answers what the
/// driver's own turns do.
///
/// # Why this is the test that says `launch` works
///
/// Everything above reaches the model through [`Shell::step`], which grows a
/// conversation through the driver's own `pages::Book`. The engine does not
/// work that way: its scheduler owns eviction, prefix sharing and the copy
/// plans, and it hands down a `kv_page_indices` CSR naming physical pages it
/// picked. `Shell::launch` fires over those, touching no book.
///
/// The two paths must agree, and the interesting disagreements are small: a
/// page off by one holds another conversation's keys and the model stays
/// fluent. So this fires one prompt both ways and holds the whole
/// distribution — not the argmax — of one against the other.
///
/// # The controls, each of which the main assertion would pass without
///
/// 1. A decode over pages holding a HISTORY must differ from the same decode
///    over pages nothing ever wrote. Without it, a fire that read no cache at
///    all would agree with itself perfectly.
/// 2. Two conversations in ONE frame must both answer what the single-request
///    frame did. A single request's page span IS the whole page list, so a
///    conversion that ignored `kv_page_indptr` entirely passes every
///    single-request assertion.
/// 3. A demand past what this adapter could ever bind is `Impossible` rather
///    than an error or a wait — a scheduler that waited on it waits forever.
/// 4. A frame whose CSR does not close is refused BEFORE it appends anything,
///    checked by firing the conversation again and getting the same answer.
#[test]
fn a_frame_the_engine_built_answers_what_the_drivers_own_turns_do() {
    let Some(_held) = gpu() else { return };
    let Some(real) = weights() else { return };
    // Twenty-four, so the frames below that name pages in the fifties make the
    // pool GROW -- `Shell::launch` sizing the pool to the highest page a frame
    // names is half of what it is for, and a pool opened wide enough would
    // never exercise it.
    let mut shell = shelled(real, 24);
    let prompt = prompt();

    // ── The driver's own path, so the number compared against was not made by
    //    the machinery under test. ──
    let step = shell
        .step(&[Turn {
            who: 1,
            tokens: prompt.clone(),
        }])
        .unwrap_or_else(|e| panic!("the book-served prefill: {e}"));
    let want: Vec<f32> = step
        .logits
        .row(step.readout_of[0])
        .expect("the readout row")
        .to_vec();
    assert_eq!(
        argmax(&want),
        PERIOD[2],
        "the control's own prefill did not continue the pattern, so there is \
         nothing worth comparing a frame against"
    );
    let seated: Vec<u32> = shell.book().pages(1).expect("its pages").to_vec();
    assert!(
        seated.len() >= 2,
        "the premise: 32 rows over 16-row pages is more than one page"
    );

    // ── The engine's path, over a conversation the book knows nothing about,
    //    on pages the frame itself names. ──
    let frame = |pages: &[u32]| driver_api::FrameSubmission {
        instance_ids: vec![1],
        kv_translation: pages.to_vec(),
        kv_translation_indptr: vec![0, pages.len() as u32],
        required_kv_pages: pages.len() as u32,
        steps: vec![driver_api::StepSubmission {
            plan: driver_api::LaunchPlan {
                token_ids: prompt.clone(),
                position_ids: (0..prompt.len() as u32).collect(),
                kv_page_indices: pages.to_vec(),
                kv_page_indptr: vec![0, pages.len() as u32],
                kv_last_page_lens: vec![prompt.len() as u32 % 16],
                qo_indptr: vec![0, prompt.len() as u32],
                sampling_indices: vec![prompt.len() as u32 - 1],
                sampling_indptr: vec![0, 1],
                ..driver_api::LaunchPlan::default()
            },
            roster_rows: vec![0],
            sub_batch_indptr: vec![0, 1],
            sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE],
            terminal_cells: Vec::new(),
            program_row_indptr: vec![0, 1],
            logical_fire_ids: vec![0],
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: vec![0, 0],
            region_row_indptr: Vec::new(),
            region_sig: Vec::new(),
            region_k: Vec::new(),
        }],
    };
    let ran = |shell: &mut Shell, f: &driver_api::FrameSubmission, what: &str| match shell
        .launch(f)
        .unwrap_or_else(|e| panic!("{what}: {e}"))
    {
        driver_wgpu::frames::Launched::Ran(steps) => steps,
        other => panic!("{what} did not run: {other:?}"),
    };

    // Pages nobody else holds, so the frame's fire and the book's cannot be
    // the same memory read twice. High ones, since the book hands out low
    // ones first.
    let fresh: Vec<u32> = (20..20 + seated.len() as u32).collect();
    let steps = ran(&mut shell, &frame(&fresh), "the frame");
    assert_eq!(steps.len(), 1, "one step in, one step out");
    let got = steps[0]
        .logits
        .row(steps[0].readout_of[0])
        .expect("the readout row");
    describe("the frame's row", got);
    assert_eq!(
        got,
        want.as_slice(),
        "the same conversation on scheduler-chosen pages answered differently \
         from the same conversation on book-chosen pages"
    );

    // ── The pages a frame names are the pages it writes. ──
    //
    // Same tokens, same everything, different physical pages -- and the pages
    // above still hold this prompt's keys, so a fire that ignored the frame's
    // page list would answer identically and prove nothing. It must still be
    // `want`, because these pages hold this conversation's own freshly written
    // keys; what must DIFFER is a decode over somebody else's, below.
    let elsewhere: Vec<u32> = (0..seated.len() as u32)
        .map(|i| 20 + seated.len() as u32 + i)
        .collect();
    let other = ran(&mut shell, &frame(&elsewhere), "the same frame elsewhere");
    assert_eq!(
        other[0]
            .logits
            .row(other[0].readout_of[0])
            .expect("the readout row"),
        want.as_slice(),
        "the same tokens written to different empty pages answered differently"
    );

    // ── Control 1: a decode reads the history its pages hold. ──
    //
    // A THIRD page: 32 tokens fill two 16-row pages exactly, so the next token
    // has nowhere to go without one.
    let held: Vec<u32> = fresh.iter().copied().chain([40]).collect();
    let decode = |pages: &[u32]| driver_api::FrameSubmission {
        instance_ids: vec![1],
        kv_translation: pages.to_vec(),
        kv_translation_indptr: vec![0, pages.len() as u32],
        required_kv_pages: pages.len() as u32,
        steps: vec![driver_api::StepSubmission {
            plan: driver_api::LaunchPlan {
                token_ids: vec![PERIOD[0]],
                position_ids: vec![prompt.len() as u32],
                kv_page_indices: pages.to_vec(),
                kv_page_indptr: vec![0, pages.len() as u32],
                kv_last_page_lens: vec![1],
                qo_indptr: vec![0, 1],
                sampling_indices: vec![0],
                sampling_indptr: vec![0, 1],
                ..driver_api::LaunchPlan::default()
            },
            ..frame(pages).steps.remove(0)
        }],
    };
    let history = ran(&mut shell, &decode(&held), "the decode with a history");
    let blank = ran(&mut shell, &decode(&[15, 16, 17]), "the decode without one");
    assert_ne!(
        history[0].logits.row(history[0].readout_of[0]),
        blank[0].logits.row(blank[0].readout_of[0]),
        "a decode over 32 tokens of history answered the same as one over \
         pages nothing ever wrote, so the frame's pages are not what attention \
         reads"
    );

    // ── Control 2: two conversations in one frame, split by the page CSR. ──
    //
    // Pages in the fifties, which the pool opened at 24 does not have: the
    // growth is `Shell::launch`'s and a scheduler is entitled to name a page
    // above a mark the trim task left.
    let mine: Vec<u32> = vec![50, 51];
    let theirs: Vec<u32> = vec![52, 53];
    let batched = driver_api::FrameSubmission {
        instance_ids: vec![1, 2],
        kv_translation: mine.iter().chain(&theirs).copied().collect(),
        kv_translation_indptr: vec![0, 2, 4],
        required_kv_pages: 4,
        steps: vec![driver_api::StepSubmission {
            plan: driver_api::LaunchPlan {
                token_ids: prompt.iter().chain(&prompt).copied().collect(),
                position_ids: (0..prompt.len() as u32)
                    .chain(0..prompt.len() as u32)
                    .collect(),
                kv_page_indices: mine.iter().chain(&theirs).copied().collect(),
                kv_page_indptr: vec![0, 2, 4],
                kv_last_page_lens: vec![0, 0],
                qo_indptr: vec![0, prompt.len() as u32, prompt.len() as u32 * 2],
                // Each request reads its own LAST row, numbered within
                // itself: the scheduler states these per request and this
                // fixture used to spell the second one as the plan's row
                // `2L - 1`. Both readings agree for request 0 (its rows start
                // at zero) and only the second says which convention is
                // meant, which is why the fixture could be wrong for as long
                // as the driver was wrong the same way.
                sampling_indices: vec![prompt.len() as u32 - 1; 2],
                sampling_indptr: vec![0, 1, 2],
                ..driver_api::LaunchPlan::default()
            },
            roster_rows: vec![0, 1],
            sub_batch_indptr: vec![0, 2],
            sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE],
            terminal_cells: Vec::new(),
            program_row_indptr: vec![0, 1, 2],
            logical_fire_ids: vec![0, 1],
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: vec![0, 0, 0],
            region_row_indptr: Vec::new(),
            region_sig: Vec::new(),
            region_k: Vec::new(),
        }],
    };
    let was = shell.shape().pages;
    let both = ran(&mut shell, &batched, "the batched frame");
    assert_eq!(
        shell.shape().pages,
        54,
        "the pool was {was} pages and the frame named page 53, so it had to \
         grow to 54 and did not"
    );
    assert_eq!(
        both[0].readout_of.len(),
        2,
        "two requests in, two readouts out"
    );
    for (which, row) in both[0].readout_of.iter().enumerate() {
        // To the BATCH SPREAD and not exactly: `want` was fired alone and
        // these two were fired together, and `answers_the_same`'s doc has why
        // that is a difference at all. The claim -- that the page CSR was
        // split where it says -- survives it: a CSR split at the wrong
        // boundary reads another request's pages, which is a different
        // conversation and nowhere near 2% of the peak.
        answers_the_same(
            both[0].logits.row(*row).expect("a readout row"),
            &want,
            &format!(
                "request {which} of a two-request frame against the same prompt \
                 fired alone, which is the page CSR not split at the boundary \
                 it names"
            ),
        );
    }

    // ── Control 3: a demand no adapter could meet is Impossible, not a wait. ──
    //
    // The number is the pool's own ceiling plus one, read the way
    // `Pool::ceiling` reads it, so this asserts the boundary rather than a
    // constant that happens to be past one particular card's.
    let ceiling = shell.shape().pages_within(shell.device().budget());
    println!(
        "budget {} bytes a buffer, so a cache of this shape tops out at {ceiling} pages",
        shell.device().budget()
    );
    assert!(
        ceiling < u32::MAX,
        "this adapter states no bound on a buffer at all, so `Impossible` \
         cannot be reached and control 3 measures nothing"
    );
    let vast = driver_api::FrameSubmission {
        required_kv_pages: ceiling + 1,
        ..frame(&fresh)
    };
    assert!(
        matches!(
            shell.launch(&vast).expect("an answer, not an error"),
            driver_wgpu::frames::Launched::Impossible
        ),
        "a demand one page past what this adapter could bind was not refused \
         as impossible, so a scheduler would wait for room that cannot exist"
    );

    // ── Control 4: a malformed frame appends nothing. ──
    let mut broken = frame(&fresh);
    // A CSR claiming more rows than there are positions.
    broken.steps[0].plan.qo_indptr = vec![0, prompt.len() as u32 + 4];
    shell
        .launch(&broken)
        .expect_err("a CSR that does not close");
    // ...and the conversation still answers what it did, so nothing was
    // half-written on the way to the refusal.
    let after = ran(&mut shell, &decode(&held), "the decode again");
    assert_eq!(
        after[0].logits.row(after[0].readout_of[0]),
        history[0].logits.row(history[0].readout_of[0]),
        "a refused frame changed the cache on its way out"
    );

    // ── And a plan naming a field this driver does not implement is refused
    //    by that field's name rather than served without it. ──
    let mut truncated = frame(&fresh);
    truncated.steps[0].plan.max_layers = Some(4);
    let why = shell
        .launch(&truncated)
        .expect_err("a layer truncation this driver would run past");
    assert!(
        format!("{why}").contains("max_layers"),
        "a frame asking for a layer truncation was refused, and the refusal \
         says {why} instead of naming the field"
    );
}

/// A fork gives the new seat the old one's history, byte for byte.
///
/// # Why this is the test forking wants
///
/// `Shell::fork` is two halves in two places: `Book::fork` decides which pages
/// move, and `Pool::copy_page` moves them. Neither half can be checked by
/// itself -- the book's answer is a list of numbers and the pool's is bytes in
/// a buffer nobody reads -- and the failure they produce together is the one
/// this crate is written against: a seat whose pages hold SOME of another
/// conversation's history answers, fluently, with a blend of two.
///
/// So the check is behavioural and needs no oracle. Feed a prompt to seat 1,
/// fork it to seat 2, then feed BOTH the same next token. Two seats holding
/// the same history over the same token must produce the same distribution,
/// EXACTLY -- not within a tolerance, because this is the same arithmetic over
/// the same bytes on the same device, and anything that makes it differ is a
/// difference in what was read.
///
/// The control is the other direction, and it is what makes the equality mean
/// something: a seat that was NOT forked, fed only that one token, must
/// DISAGREE. Without it a fork that copied nothing at all would pass -- two
/// empty seats also agree.
#[test]
fn a_forked_seat_reads_the_history_it_was_given() {
    let Some(_guard) = gpu() else {
        return;
    };
    let Some(real) = weights() else {
        return;
    };
    // FOUR seats over sixteen pages: the original, its fork, a fork of that
    // fork, and the control. Sixteen and not eight because a fork here COPIES
    // -- there is no copy-on-write in this driver -- so every seat costs the
    // prompt's pages again, and eight of them ran out at the third fork with
    // `NoPages { wanted: 3, spare: 2 }`.
    let mut shell = shelled(real, 16);
    let prompt = prompt();

    let row_of = |step: &driver_wgpu::turns::Step, at: usize| -> Vec<f32> {
        step.logits
            .row(step.readout_of[at])
            .expect("the fire read out the turn it was asked for")
            .to_vec()
    };

    // Seat 1 hears the whole prompt.
    let first = shell
        .step(&[Turn {
            who: 1,
            tokens: prompt.clone(),
        }])
        .expect("the prompt fires");
    let next = argmax(&row_of(&first, 0));

    // Seat 2 is given seat 1's history. The count is asserted because a fork
    // that moved NOTHING is the failure this whole test is about, and it would
    // otherwise be indistinguishable from a fork that worked until the
    // distributions were compared.
    let moved = shell.fork(1, 2).expect("seat 1 has a history to give");
    assert!(
        moved > 0,
        "a fork of a {}-token conversation moved no pages",
        prompt.len()
    );

    // A FORK OF A FORK, taken BEFORE anyone hears the next token so all three
    // seats hold exactly the same history. `prefix-tree-kv-cache` builds two
    // levels -- root, two children, four leaves -- so its first leaf is the
    // first fork taken FROM a seat that was itself a fork, and a book that
    // handed out a child's pages without noticing they were already shared
    // would show up there and nowhere else.
    let deeper = shell.fork(2, 4).expect("a forked seat can be forked");
    assert!(deeper > 0, "a fork of a fork moved no pages");

    // All three seats hear the same next token, in ONE fire, so any difference
    // is the cache and not the fire.
    let both = shell
        .step(&[
            Turn {
                who: 1,
                tokens: vec![next],
            },
            Turn {
                who: 2,
                tokens: vec![next],
            },
            Turn {
                who: 4,
                tokens: vec![next],
            },
        ])
        .expect("all three seats fire");
    let original = row_of(&both, 0);
    let forked = row_of(&both, 1);
    let grandchild = row_of(&both, 2);
    answers_the_same(
        &original,
        &forked,
        "the forked seat against the seat it was forked from",
    );
    answers_the_same(
        &original,
        &grandchild,
        "a seat forked FROM a fork against the seat both came from",
    );

    // THE CONTROL. Seat 3 was never forked, so it hears `next` with no history
    // at all and must answer something else. If it agrees, this test cannot
    // tell a copied cache from an ignored one.
    let alone = shell
        .step(&[Turn {
            who: 3,
            tokens: vec![next],
        }])
        .expect("an empty seat fires");
    answers_differently(
        &row_of(&alone, 0),
        &forked,
        "a seat with NO history against the forked seat, which is this test \
         comparing the fire with itself rather than the cache",
    );
}

/// A seat's answer does not depend on how many OTHER seats fired with it --
/// beyond what the matvec's own row tiling costs.
///
/// # The measurement this is the driver-level half of
///
/// `a_forked_seat_reads_the_history_it_was_given` fires three seats holding
/// identical histories in one step. Two agreed to the bit and the third did
/// not, and the third was not the forked one -- it was the LAST ROW. Forking
/// never entered into it:
///
/// ```text
///   rows in the fire    rows disagreeing with row 0
///          1                       []
///          2                       []
///          3                       [2]
///          4                       []
///          5                       [4]
///          6                       []
///          7                       [6]
///          8                       []
/// ```
///
/// At an odd row count the last row disagrees with every other; at an even one
/// nobody does; and there are only ever two answers. Reordering the seats moves
/// the disagreement to whichever seat is last, so it is the ROW and not the
/// seat.
///
/// # Why
///
/// `quant/qmv.wgsl`'s `reduce_store` gives one workgroup `PIE_MT` = 2
/// activation rows, so every row of an even batch is summed by `block_dot2`
/// and the tail of an odd one alone by `block_dot1`. Setting `PIE_MT` to 1
/// makes every row count agree bit for bit, and so does calling `block_dot1`
/// twice where the `mt == 2` arm calls `block_dot2` -- which puts the
/// difference inside that body and not in the reduction tree or the grid.
///
/// Fired directly at the kernel, one row's activations repeated, the two arms
/// part by two bf16 ulps on about five outputs in a hundred thousand, and only
/// a projection as wide as the lm head rolls the dice often enough to show it:
///
/// ```text
///   n_out    outputs differing from the lone row    worst
///     2048          0 of      2048                  0
///     8192          0 of      8192                  0
///    16384          1 of     16384                  0.125 at -26.5
///    32768          3 of     32768                  0.125
///   131072          7 of    131072                  0.125
///   151936          7 of    151936                  0.125
/// ```
///
/// The rate is flat, so there is no threshold and no provoking shape. Two
/// bf16 ulps at the head, twenty-eight layers deep, is 0.79% of the row's peak
/// across 92% of the logits with the argmax unchanged -- which is
/// [`BATCH_SPREAD`], and which is why that constant is not zero.
///
/// # What this asserts
///
/// That the parity of the batch is the ONLY thing that moves, and that it
/// moves by no more than the spread. A kernel change that made the two arms
/// compute different sums rather than round them differently would miss by
/// far more, and would be read here as it was read the first time: as the last
/// row of an odd fire.
#[test]
fn a_seats_answer_does_not_depend_on_how_many_seats_fired_with_it() {
    let Some(_guard) = gpu() else {
        return;
    };
    let Some(real) = weights() else {
        return;
    };
    // A fork COPIES here, and the sweep below wants a fresh set of seats per
    // row count -- a seat that has already heard `next` has moved on -- so
    // this is 1 + (1 + 2 + ... + 6) copies of a two-page history.
    let mut shell = shelled(real, 128);

    let row_of = |step: &driver_wgpu::turns::Step, at: usize| -> Vec<f32> {
        step.logits
            .row(step.readout_of[at])
            .expect("the fire read out the turn it was asked for")
            .to_vec()
    };

    let first = shell
        .step(&[Turn {
            who: 1,
            tokens: prompt(),
        }])
        .expect("the prompt fires");
    let next = argmax(&row_of(&first, 0));

    // The lone answer, which every batched row below is compared against --
    // and which is the one arm the sweep would otherwise never fire, since a
    // batch of one is the only fire whose FIRST row is also its tail.
    let mut who = 100u64;
    who += 1;
    shell.fork(1, who).expect("the prompt can be forked");
    let alone = row_of(
        &shell
            .step(&[Turn {
                who,
                tokens: vec![next],
            }])
            .expect("one seat fires"),
        0,
    );

    for rows in 2..=6usize {
        let seats: Vec<u64> = (0..rows)
            .map(|_| {
                who += 1;
                shell
                    .fork(1, who)
                    .expect("the prompt can be forked again");
                who
            })
            .collect();
        let turns: Vec<Turn> = seats
            .iter()
            .map(|w| Turn {
                who: *w,
                tokens: vec![next],
            })
            .collect();
        let fired = shell.step(&turns).expect("every seat fires");
        for at in 0..rows {
            answers_the_same(
                &row_of(&fired, at),
                &alone,
                &format!("row {at} of a {rows}-row fire against the same seat fired alone"),
            );
        }
    }
}

/// The prefix tree this driver is asked for, built seat by seat, against a
/// seat that was never forked at all.
///
/// # What this reproduces
///
/// `prefix-tree-kv-cache` prefills a common root, forks it, APPENDS to the
/// fork, forks THAT, and appends again — then generates from the leaves. Its
/// run through a real `pie serve` dies after the fourth append, which is the
/// first leaf: the first fork taken from a seat that has itself been forked
/// AND written to since. `a_forked_seat_reads_the_history_it_was_given` covers
/// the fork and the fork-of-a-fork; neither of them writes to a seat between
/// the two forks, and writing is what makes a copied page diverge from the one
/// it was copied from.
///
/// # The reference is not another fork
///
/// Seat D hears `root ++ child ++ leaf` in one fire and was never forked, so
/// it shares no page with anybody. If the tree's leaf agrees with it, every
/// page the leaf reads holds what the tokens that wrote it put there —
/// whichever seat wrote them and whichever copy it wrote into. That is the
/// whole claim, and no oracle is needed for it.
#[test]
fn a_two_level_prefix_tree_reads_what_a_seat_that_never_forked_reads() {
    let Some(_guard) = gpu() else {
        return;
    };
    let Some(real) = weights() else {
        return;
    };
    // A fork COPIES here, so five seats over a 21-token history want room for
    // five copies of it. Twenty-four pages is that with margin.
    let mut shell = shelled(real, 24);

    let row_of = |step: &driver_wgpu::turns::Step, at: usize| -> Vec<f32> {
        step.logits
            .row(step.readout_of[at])
            .expect("the fire read out the turn it was asked for")
            .to_vec()
    };

    // Three segments of the tree, all different, none a prefix of another.
    let root: Vec<u32> = prompt().into_iter().take(11).collect();
    let child: Vec<u32> = (0..5).map(|i| 6_000 + i * 13).collect();
    let leaf: Vec<u32> = (0..5).map(|i| 7_000 + i * 29).collect();

    // Seat 1 hears the root.
    shell
        .step(&[Turn {
            who: 1,
            tokens: root.clone(),
        }])
        .expect("the root fires");

    // Fork it, and WRITE to the fork. This is the step the other fork test
    // does not take.
    assert!(shell.fork(1, 2).expect("the root can be forked") > 0);
    shell
        .step(&[Turn {
            who: 2,
            tokens: child.clone(),
        }])
        .expect("the child fires");

    // Fork the written-to fork, and write to that.
    assert!(
        shell.fork(2, 3).expect("the child can be forked") > 0,
        "a fork of a written-to fork moved no pages"
    );
    shell
        .step(&[Turn {
            who: 3,
            tokens: leaf.clone(),
        }])
        .expect("the leaf fires");

    // Seat 4 hears the whole path at once and was never forked.
    let mut whole = root.clone();
    whole.extend(&child);
    whole.extend(&leaf);
    shell
        .step(&[Turn {
            who: 4,
            tokens: whole.clone(),
        }])
        .expect("the unforked seat fires");

    // Both hear the same next token, in ONE fire.
    let next = 1234u32;
    let both = shell
        .step(&[
            Turn {
                who: 3,
                tokens: vec![next],
            },
            Turn {
                who: 4,
                tokens: vec![next],
            },
        ])
        .expect("the leaf and the unforked seat fire");
    let tree = row_of(&both, 0);
    let flat = row_of(&both, 1);
    answers_the_same(
        &tree,
        &flat,
        &format!(
            "a leaf of a two-level fork tree against a seat that heard the same \
             {} tokens and was never forked",
            whole.len()
        ),
    );

    // THE CONTROL. A seat that heard only the ROOT must disagree, or the
    // comparison above is insensitive to what the appends wrote.
    assert!(shell.fork(1, 5).expect("the root can be forked again") > 0);
    let short = shell
        .step(&[Turn {
            who: 5,
            tokens: vec![next],
        }])
        .expect("the short seat fires");
    answers_differently(
        &row_of(&short, 0),
        &tree,
        "a seat holding only the root against the leaf, which is this test \
         unable to see what the appends wrote",
    );
}

/// A custom mask is APPLIED, and applying the causal one changes nothing.
///
/// # Why this test and not a CPU oracle
///
/// There is no independent implementation of masked attention here to compare
/// against, and writing one would be comparing this driver to a second thing
/// this session wrote. What there IS, exactly, is a mask whose meaning is
/// already known: the causal rectangle. `attn/sdpa_paged.wgsl` applies
/// `kp > q_pos || kp < start` before it ever looks at the mask, so a mask that
/// allows every key the causal rule allows must produce the SAME BITS as no
/// mask at all.
///
/// That is a strong check rather than a weak one, because almost every way of
/// getting the rectangle wrong breaks it:
///
/// * a pitch off by one indexes row `r` at row `r - 1`'s offset and forbids
///   real keys;
/// * the byte packing reversed puts row 3's byte where row 0's belongs;
/// * the enable table written per REQUEST instead of per ROW enables the wrong
///   rows;
/// * the runs decoded inverted forbids everything.
///
/// Each of those changes the answer, and the identity is what says none of
/// them happened. The second half then proves the mask is not simply being
/// dropped -- which would pass the identity trivially -- by forbidding a key
/// the causal rule allows and requiring the answer to MOVE.
///
/// # What the controls actually showed, including the one that did not fire
///
/// Both halves were checked by breaking the driver and watching them go red:
///
/// * clearing every row's enable byte fails the SECOND half -- *"forbidding 16
///   of 32 keys changed nothing"* -- and passes the first, which is exactly
///   the failure mode the second half exists for;
/// * reporting a pitch one larger than the bytes were packed at fails the
///   FIRST half, because row `r` is then read at row `r`'s offset plus `r`,
///   and by the last row that is a whole row out.
///
/// A third attempt did NOT fire, and it is the useful one: widening the
/// rectangle CONSISTENTLY -- writer and reader both -- changes no answer, and
/// should not. The extra column is always zero, and the causal rule has
/// already forbidden every key it could apply to. So what this test pins is
/// that the two pitches AGREE, not that either has a particular value. A test
/// that claimed the latter would be claiming something the shader does not
/// depend on.
///
/// The third half is the one the rectangle's shape makes possible: with two
/// requests in one fire, masking the first must leave the second's answer
/// exactly where it was. A pitch or a row base that is per-request rather than
/// per-fire-row passes the first two checks and fails this one.
#[test]
fn a_custom_mask_is_applied_and_the_causal_one_is_the_identity() {
    let Some(_held) = gpu() else { return };
    let Some(real) = weights() else { return };
    let mut shell = shelled(real, 24);
    let prompt = prompt();
    let n = prompt.len() as u32;

    // One request over its own pages, with whatever mask is handed in.
    let frame = |pages: &[u32],
                 masks: Vec<driver_api::EncodedMask>,
                 mask_indptr: Vec<u32>|
     -> driver_api::FrameSubmission {
        driver_api::FrameSubmission {
            instance_ids: vec![1],
            kv_translation: pages.to_vec(),
            kv_translation_indptr: vec![0, pages.len() as u32],
            required_kv_pages: pages.len() as u32,
            steps: vec![driver_api::StepSubmission {
                plan: driver_api::LaunchPlan {
                    token_ids: prompt.clone(),
                    position_ids: (0..n).collect(),
                    kv_page_indices: pages.to_vec(),
                    kv_page_indptr: vec![0, pages.len() as u32],
                    kv_last_page_lens: vec![n % 16],
                    kv_len: vec![n],
                    qo_indptr: vec![0, n],
                    sampling_indices: vec![n - 1],
                    sampling_indptr: vec![0, 1],
                    has_user_mask: !masks.is_empty(),
                    masks,
                    mask_indptr,
                    ..driver_api::LaunchPlan::default()
                },
                roster_rows: vec![0],
                sub_batch_indptr: vec![0, 1],
                sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE],
                terminal_cells: Vec::new(),
                program_row_indptr: vec![0, 1],
                logical_fire_ids: vec![0],
                channel_expected_head: Vec::new(),
                channel_expected_tail: Vec::new(),
                channel_ticket_indptr: vec![0, 0],
                region_row_indptr: Vec::new(),
                region_sig: Vec::new(),
                region_k: Vec::new(),
            }],
        }
    };
    let ran = |shell: &mut Shell, f: &driver_api::FrameSubmission, what: &str| -> Vec<f32> {
        match shell.launch(f).unwrap_or_else(|e| panic!("{what}: {e}")) {
            driver_wgpu::frames::Launched::Ran(steps) => {
                let step = steps.into_iter().next().expect("one step");
                step.logits
                    .row(step.readout_of[0])
                    .expect("the readout row")
                    .to_vec()
            }
            other => panic!("{what} did not run: {other:?}"),
        }
    };

    // ── The control: no mask table at all. ──
    let bare = ran(
        &mut shell,
        &frame(&[40, 41, 42], Vec::new(), Vec::new()),
        "no mask",
    );
    assert_eq!(
        argmax(&bare),
        PERIOD[2],
        "the control did not continue the pattern, so there is nothing worth \
         comparing a masked fire against"
    );

    // ── The causal mask, spelled the way `wire.rs` spells it: row j of an
    //    n-row request over n keys attends to `j + 1` of them. ──
    let causal: Vec<driver_api::EncodedMask> = (0..n)
        .map(|j| driver_api::EncodedMask::new(vec![0, j + 1], u64::from(j + 1)))
        .collect();
    let masked = ran(
        &mut shell,
        &frame(&[43, 44, 45], causal.clone(), vec![0, n]),
        "the causal mask",
    );
    assert_eq!(
        masked, bare,
        "applying the mask the attention already computes must change no bit"
    );

    // ── And a mask that forbids something must MOVE the answer. ──
    //
    // The LAST row is the one read out, so its mask is the one that can reach
    // the readout. Forbid the first half of its history and keep the rest.
    let half = n / 2;
    let mut forbidding = causal;
    forbidding[(n - 1) as usize] = driver_api::EncodedMask::new(vec![half, n - half], u64::from(n));
    let restricted = ran(
        &mut shell,
        &frame(&[46, 47, 48], forbidding, vec![0, n]),
        "a restricting mask",
    );
    let moved = restricted
        .iter()
        .zip(&bare)
        .filter(|(a, b)| (*a - *b).abs() > 1e-3)
        .count();
    assert!(
        moved > 0,
        "forbidding {half} of {n} keys changed nothing, so the mask is being \
         dropped rather than applied"
    );
}

/// A request naming SEVERAL readout rows is handed exactly those rows.
///
/// # Why this test and what it isolates
///
/// The multi-readout path used to be a named refusal, and lifting it made
/// `cacheback-speculative-decoding` run and then disagree with its own
/// sequential control. That disagreement has two possible homes: the DRIVER
/// hands back the wrong distributions, or something above it mishandles a
/// rejected window. This settles the first half.
///
/// The oracle is the fire's own `logits`: `Step::readouts_of[r]` names rows of
/// `Step::logits`, and a fire where every row samples has every row there. So
/// what a program is handed for request `r` must be those rows, in that order,
/// bit for bit. Nothing here is approximate and nothing needs a second
/// implementation of attention.
///
/// The control is the ordinary single-row case in the same fire: if the
/// gather had gone wrong in a way that also broke one row, every other test in
/// this file would already be red, so the interesting assertion is that the
/// MANY-row request is right while the one-row request beside it still is.
#[test]
fn a_request_that_names_several_readout_rows_is_handed_exactly_those_rows() {
    let Some(_held) = gpu() else { return };
    let Some(real) = weights() else { return };
    let mut shell = shelled(real, 24);
    let prompt = prompt();
    let n = prompt.len() as u32;

    // One request over its own pages, reading out `rows` of its own rows.
    let frame = |pages: &[u32], readouts: &[u32]| driver_api::FrameSubmission {
        instance_ids: vec![1],
        kv_translation: pages.to_vec(),
        kv_translation_indptr: vec![0, pages.len() as u32],
        required_kv_pages: pages.len() as u32,
        steps: vec![driver_api::StepSubmission {
            plan: driver_api::LaunchPlan {
                token_ids: prompt.clone(),
                position_ids: (0..n).collect(),
                kv_page_indices: pages.to_vec(),
                kv_page_indptr: vec![0, pages.len() as u32],
                kv_last_page_lens: vec![n % 16],
                kv_len: vec![n],
                qo_indptr: vec![0, n],
                sampling_indices: readouts.to_vec(),
                sampling_indptr: vec![0, readouts.len() as u32],
                ..driver_api::LaunchPlan::default()
            },
            roster_rows: vec![0],
            sub_batch_indptr: vec![0, 1],
            sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE],
            terminal_cells: Vec::new(),
            program_row_indptr: vec![0, 1],
            logical_fire_ids: vec![0],
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: vec![0, 0],
            region_row_indptr: Vec::new(),
            region_sig: Vec::new(),
            region_k: Vec::new(),
        }],
    };
    let step = |shell: &mut Shell, f: &driver_api::FrameSubmission| -> driver_wgpu::turns::Step {
        match shell.launch(f).expect("the frame launches") {
            driver_wgpu::frames::Launched::Ran(steps) => {
                steps.into_iter().next().expect("one step")
            }
            other => panic!("did not run: {other:?}"),
        }
    };

    // Three of its own rows, which is a speculative verifier's shape.
    let many = step(&mut shell, &frame(&[50, 51, 52], &[n - 3, n - 2, n - 1]));
    // THREE ROWS, DISTINCT, AND ALL OF THEM HELD. This stated the fire rows
    // themselves -- `vec![n - 3, n - 2, n - 1]` -- and cannot, since the
    // readback was narrowed: `turns.rs` remaps `readouts_of` through
    // `frame.sampling_indices`, so it names rows of `logits`. A request for
    // fire rows 29, 30 and 31 whose sampling table holds exactly those three
    // reads `[0, 1, 2]`, which is the remap working.
    //
    // The order claim survives in the only form still checkable here: the
    // readback holds three distributions, the span names three distinct ones,
    // and the value comparison below says they are three different rows of a
    // real prompt rather than one row gathered three times.
    assert_eq!(
        many.readouts_of[0].len(),
        3,
        "the span names {} rows where the request named three",
        many.readouts_of[0].len()
    );
    assert_eq!(
        many.logits.rows, 3,
        "the readback holds {} distributions where the request named three",
        many.logits.rows
    );
    {
        let mut seen = many.readouts_of[0].clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len(),
            3,
            "the span names the same readback row more than once: {:?}",
            many.readouts_of[0]
        );
    }
    // Every one of them is a row of this fire's own logits, and they are
    // DIFFERENT rows -- a gather that returned the last row three times would
    // otherwise pass the assertion above.
    let rows: Vec<Vec<f32>> = many.readouts_of[0]
        .iter()
        .map(|&at| many.logits.row(at).expect("the row").to_vec())
        .collect();
    assert!(
        rows[0] != rows[1] && rows[1] != rows[2],
        "three consecutive rows of a real prompt are three different \
         distributions; identical ones mean the gather read one row thrice"
    );

    // And the one-row case in the same shell still answers its LAST row, which
    // is what every decode asks for.
    let one = step(&mut shell, &frame(&[53, 54, 55], &[n - 1]));
    // Both name a row of the narrowed readback rather than of the fire, and a
    // request naming ONE row narrows it to one -- so both are 0, and the claim
    // worth making is that they agree with each other and with what the
    // readback holds. Asking for `n - 1` here is still what makes the value
    // comparison below a comparison of the LAST row.
    assert_eq!(one.logits.rows, 1);
    assert_eq!(one.readouts_of[0], vec![one.readout_of[0]]);
    assert_eq!(
        one.logits.row(one.readout_of[0]).expect("the row").to_vec(),
        rows[2],
        "the same prompt on different pages answers the same last row"
    );
}

/// A row's answer does not depend on tokens that come AFTER it.
///
/// # Why this is the question
///
/// It is what causal attention means, and it is the premise every speculative
/// verifier rests on: a verification fire embeds `committed + draft` and reads
/// the row at the end of `committed`, expecting the distribution that row
/// would have had on its own. If a longer fire changes an earlier row's
/// answer, greedy verification stops agreeing with sequential decoding and the
/// two diverge at the first rejection -- which is exactly what
/// `cacheback-speculative-decoding`'s curated control reports.
///
/// So this asks it directly, with no speculation in sight: the same prefix,
/// once alone and once with three more tokens after it, must give the same
/// row bit for bit.
///
/// The control is the last assertion: the LONGER fire's own last row is a
/// different distribution, so a test that compared two identical buffers by
/// accident would fail it.
#[test]
fn a_rows_answer_does_not_depend_on_the_tokens_after_it() {
    let Some(_held) = gpu() else { return };
    let Some(real) = weights() else { return };
    let mut shell = shelled(real, 24);
    let prompt = prompt();
    let n = prompt.len() as u32;

    let frame = |pages: &[u32], tokens: &[u32], readout: u32| driver_api::FrameSubmission {
        instance_ids: vec![1],
        kv_translation: pages.to_vec(),
        kv_translation_indptr: vec![0, pages.len() as u32],
        required_kv_pages: pages.len() as u32,
        steps: vec![driver_api::StepSubmission {
            plan: driver_api::LaunchPlan {
                token_ids: tokens.to_vec(),
                position_ids: (0..tokens.len() as u32).collect(),
                kv_page_indices: pages.to_vec(),
                kv_page_indptr: vec![0, pages.len() as u32],
                // `((len - 1) % 16) + 1`, not `len % 16`: a fire whose length
                // is a multiple of the page size fills its last page, and
                // `% 16` says ZERO there. Two fires whose lengths differ then
                // attend different spans and the comparison below measures
                // that instead of what it means to.
                kv_last_page_lens: vec![(tokens.len() as u32 - 1) % 16 + 1],
                kv_len: vec![tokens.len() as u32],
                qo_indptr: vec![0, tokens.len() as u32],
                sampling_indices: vec![readout],
                sampling_indptr: vec![0, 1],
                ..driver_api::LaunchPlan::default()
            },
            roster_rows: vec![0],
            sub_batch_indptr: vec![0, 1],
            sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE],
            terminal_cells: Vec::new(),
            program_row_indptr: vec![0, 1],
            logical_fire_ids: vec![0],
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: vec![0, 0],
            region_row_indptr: Vec::new(),
            region_sig: Vec::new(),
            region_k: Vec::new(),
        }],
    };
    // `readouts_of`, not `readout_of`. The second is the request's LAST row,
    // which is what a decode wants and is NOT what the sampling table asked
    // for here -- reading it compared row 34 of two fires whose last tokens
    // differ, which differs for the most ordinary reason there is. The
    // distinction is the one this file's multi-readout support introduced,
    // and getting it wrong in the test that hunts a causality bug is how a
    // fixture manufactures the very symptom it is looking for.
    let read = |shell: &mut Shell, f: &driver_api::FrameSubmission, want: u32| -> Vec<f32> {
        match shell.launch(f).expect("the frame launches") {
            driver_wgpu::frames::Launched::Ran(steps) => {
                let step = steps.into_iter().next().expect("one step");
                // A COUNT AND A BOUND, NOT THE FIRE ROW. This asserted
                // `readouts_of[0] == vec![want]` and could not, once the
                // readback was narrowed: `turns.rs` puts `readouts_of`
                // through `frame.sampling_indices`, so it names a row of
                // `logits` and not a row of the fire. Asking for fire row 31
                // when 31 is the only sampled row now reads `[0]`, which is
                // the remap working rather than the wrong row.
                //
                // What survives is the claim that actually guards this test:
                // ONE row was read out, and the readback holds exactly one.
                // A sampling table that named a second row, or none, moves
                // both numbers.
                assert_eq!(
                    step.readouts_of[0].len(),
                    1,
                    "the fire read out {} rows where this test asked for one \
                     (fire row {want})",
                    step.readouts_of[0].len()
                );
                assert_eq!(
                    step.logits.rows, 1,
                    "the readback holds {} distributions where the sampling \
                     table named one (fire row {want})",
                    step.logits.rows
                );
                assert!(
                    step.logits.row(step.readouts_of[0][0]).is_some(),
                    "the readout names a row the readback does not hold"
                );
                step.logits
                    .row(step.readouts_of[0][0])
                    .expect("the readout row")
                    .to_vec()
            }
            other => panic!("did not run: {other:?}"),
        }
    };

    // Two fires of the SAME LENGTH whose last three tokens differ.
    //
    // Same length, because the row count picks the KERNEL: `Rule::Qmm`'s
    // guard is `TokensMultipleOf(16)`, so a 32-row fire takes the tiled
    // `affine_qmm_t` and a 35-row fire takes the matvec fallback. Comparing
    // those two measures bf16 rounding between two kernel families and not
    // causality -- which is what the first version of this test did, and it
    // failed for that reason rather than for the one it names.
    let mut one = prompt.clone();
    one.extend_from_slice(&[PERIOD[1], PERIOD[2], PERIOD[0]]);
    let mut other = prompt.clone();
    other.extend_from_slice(&[PERIOD[0], PERIOD[0], PERIOD[1]]);
    assert_eq!(one.len(), other.len());
    assert_ne!(one[n as usize..], other[n as usize..]);

    let alone = read(&mut shell, &frame(&[60, 61, 62, 71], &one, n - 1), n - 1);
    let with_tail = read(&mut shell, &frame(&[63, 64, 65, 66], &other, n - 1), n - 1);

    assert_eq!(
        alone,
        with_tail,
        "row {} answered differently when the three tokens AFTER it changed, \
         so attention here is not causal within a fire -- which is the \
         premise every speculative verifier rests on",
        n - 1
    );

    // The control: the fire's OWN last row is a different answer, so
    // the equality above is not two copies of the same buffer.
    let tail_row = read(
        &mut shell,
        &frame(&[67, 68, 69, 70], &one, one.len() as u32 - 1),
        one.len() as u32 - 1,
    );
    assert!(
        tail_row != alone,
        "the fire's last row is the same distribution as the row this test \
         compares, so it cannot see what it is for"
    );

    // And the OTHER half, which is what a speculative verifier actually
    // does: the same row read from fires of DIFFERENT length.
    //
    // Not bit-identical, and it cannot be. `Rule::Qmm`'s guard is
    // `TokensMultipleOf(16)`, so a 32-row fire takes the tiled
    // `affine_qmm_t` and a 35-row fire takes the matvec fallback -- two
    // kernel families over the same numbers in bf16. What is asserted is
    // that they agree to a TOLERANCE, which says the difference is rounding
    // rather than a different computation.
    //
    // This is why `cacheback-speculative-decoding`'s curated control is not
    // exact on this backend. `draft_length = 0` fires L rows per step and
    // `draft_length = k` fires L + k, so the two take different projections
    // whenever the guard falls differently, and an argmax at a near-tie
    // flips. The test's premise -- "an exact control" -- holds only for a
    // backend whose kernel choice does not depend on the row count.
    let short = read(&mut shell, &frame(&[72, 73, 74], &prompt, n - 1), n - 1);
    let long_same_row = read(&mut shell, &frame(&[75, 76, 77, 78], &one, n - 1), n - 1);
    // Against the ROW's largest magnitude, not per element.
    //
    // A per-element relative error was tried and is not a measurement of
    // anything here: a logit near zero makes the denominator tiny, and two
    // kernels that agree to a hundredth of the row's scale score 1.99 on an
    // element whose value is 0.3. `driver-vulkan`'s
    // `the_tiled_gemm_answers_the_way_the_vector_kernel_does` does normalise
    // per element (`max(|a|, |b|, 1e-3)`) and holds its pair to 0.05 -- but it
    // is asking a different question, over the SAME rows through two plans,
    // where the near-zero elements agree too. This compares two fires of
    // different LENGTH, which is the thing a speculative verifier actually
    // does, and the row scale is the honest denominator for it.
    let worst = short
        .iter()
        .zip(&long_same_row)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = short.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        worst <= 0.05 * scale,
        "the same row from a 32-row and a 35-row fire differs by {worst}, \
         which is {:.1}% of the row's largest magnitude ({scale}) -- too much \
         to be bf16 rounding between two kernel families",
        100.0 * worst / scale
    );
    assert!(
        worst > 0.0,
        "the two kernel families answered bit-identically, so this assertion \
         is not measuring what it claims"
    );

    // And the third measurement, which says the cause is the KERNEL and not
    // the row count: the same row from a 32-row and a 64-row fire, both
    // MULTIPLES of the tile, is bit-identical.
    //
    // If a longer fire changed an earlier row's answer by itself, this would
    // differ too. It does not, so what the pair above measures is the switch
    // between `affine_qmm_t` and the matvec fallback -- and a backend that
    // SPLIT a partial fire into tile-shaped pieces instead of falling back,
    // the way `driver-vulkan`'s `Serving::tiled` does, would not have the
    // switch at all. That is why `cacheback-speculative-decoding` passes its
    // control there and not here.
    // SIXTY-FOUR, not 48. The tile moved from `bm_16` to `bm_32` upstream, so
    // 48 rows is one and a half tiles and takes the fallback just as 35 does
    // -- which is what the first version of this measurement compared, and it
    // failed for that reason rather than the one it names. 64 is a multiple of
    // every tile the GEMM is compiled for.
    let mut padded = prompt.clone();
    padded.extend(std::iter::repeat_n(PERIOD[0], 32));
    assert_eq!(padded.len(), 64);
    assert_eq!(padded.len() % 32, 0, "the premise: both are tile multiples");
    let longer_tile_multiple = read(&mut shell, &frame(&[79, 80, 81, 82], &padded, n - 1), n - 1);
    assert_eq!(
        short, longer_tile_multiple,
        "the same row from a 32-row and a 64-row fire differs, so the row \
         count alone changes an answer and the kernel switch is not the whole \
         story"
    );
}

/// The tiled GEMM and the matrix-vector kernel answer the same rows the same
/// way.
///
/// # Why this is the check that was missing
///
/// `Serving` picks a plan by row count: a many-row fire states
/// `affine_qmm_t` and its residual twin where a one-row fire states
/// `affine_qmv_fast`. Same weights, same activations, different code -- and
/// nothing in this crate had ever asked whether they agree. Every other whole-
/// plan claim here compares a plan against itself, which measures ordering; a
/// matmul that transposed its operands would pass all of them.
///
/// It is also the question behind `cacheback-speculative-decoding`'s curated
/// control. A speculative verifier fires `L` rows and then `L + k`, the tile
/// guard sends those to different families, and whether the answer changes
/// depends on how closely the two agree. `driver-vulkan` asks exactly this and
/// holds its pair to 0.05 relative
/// (`the_tiled_gemm_answers_the_way_the_vector_kernel_does`); its control
/// passes and this backend's does not, so the number is worth having on both
/// sides.
///
/// # The two fires differ in ONE thing
///
/// The same prompt, the same rows, the same pages, read at the same row --
/// and the plan in the prefill slot swapped. Anything else would put a second
/// difference in a comparison that exists to isolate one.
/// # WHAT IT FOUND THE FIRST TIME IT EVER RAN
///
/// It had never run: the fixture it needs is an unquantised `Qwen/Qwen3-0.6B`
/// checkpoint and `weights()` returned `None`, so it SKIPPED, along with ten
/// other real-model tests in this file. Downloading the checkpoint is the
/// whole of what changed, and the first real run said:
///
/// ```text
///   134533 of 151936 logits differ; the worst by 27.07 at token 41777
///   (tiled 1.05 against vector 28.13), 155.8% of the row's peak
/// ```
///
/// Not rounding, and not a near miss: 88% of the row was a different row.
///
/// ## Which family was wrong
///
/// Perturb ONE token of a 480-token prompt and read the worst change in the
/// readout row. A causal model's last row attends to every token, so every
/// position must move it and the last must move it most:
///
/// ```text
///   position     0    64   128   256   320   384   448   470   479
///   tiled     28.38  0.12  0.06  0.00  0.00  0.00  0.00  0.00  0.16
///   vector     5.02  2.06  2.03  3.09  2.19  7.74  4.47  3.69 13.81
/// ```
///
/// The vector family answers the way a transformer must. The tiled family
/// answered a row that responds to token 0 and to nothing after position 128
/// -- so the direction of blame was settled by measurement rather than
/// assumed, which mattered, because the answer was not where it looked.
///
/// ## It was not the GEMM
///
/// `kernels-wgpu`'s
/// `a_tiled_gemm_agrees_over_every_tile_shape_and_quantization_point` agrees
/// with a host reference over all nine tiles and six codecs, and it still
/// does at m = 32 and m = 64 as well as the 33 it fires by default -- so the
/// tiling and the arithmetic are right at an aligned row count and a ragged
/// one alike. The defect was the BUFFER: `qmm_fp16_precast` stages the
/// activation through `cast_qmm_input` into `half_in` and the GEMM reads
/// `half_in` instead of `x`. One fact flipped, nothing else:
///
/// ```text
///   precast=true    120.4% of the row's peak, 134533/151936 logits differ
///   precast=false     0.9% of the row's peak,      0/151936 logits differ
/// ```
///
/// `engine::driver::backend::wgpu` therefore stamps `qmm_fp16_precast: false`
/// and carries where to look when someone repairs it. `backend_facts()`
/// mirrors that stamp, which is why this test passes.
///
/// ## The blast radius, and why the row bound is not to blame
///
/// It reproduced at `e9843424b` -- the commit before `Params` carried `m` --
/// with the identical logit at the identical index, so it is older than the
/// row bound. What `qmm_partial_rows` did was widen it, from one row count in
/// thirty-two to thirty-one in thirty-two, which is how a defect that had sat
/// under the aligned lengths became visible from the daemon.
///
///
/// RETIRED WITH `kernels-wgpu`'s TEST TREE. That name is a record of a
/// measurement now, not a live proof: the crate lost `tests/` and every
/// in-file `mod tests` when the three shader planes moved their numbers to
/// the fire that reads them, and nothing in this workspace re-runs it. What
/// it reported is still why the sentence above says what it says; what is
/// gone is the thing that would notice if it stopped being true.
#[test]
fn the_tiled_gemm_answers_the_way_the_vector_kernel_does() {
    the_two_families_agree(prompt());
}

/// The same comparison at a row count the tile does NOT divide.
///
/// This is the whole correctness claim behind bounding `write_out` by
/// `params.m`. `qmm_grid` rounds the row axis up with `div_ceil`, so a fire
/// of 495 rows launches 512 rows' worth of tiles and the last seventeen are
/// arithmetic on whatever lies past the activation. Before the bound those
/// rows were STORED, over the next value in the arena.
///
/// 495 and not 496: it is odd, it is not a multiple of 16, 32 or 64, and so
/// it exercises the overhang at every tile this backend can pick rather than
/// only the one it picks today.
///
/// # IT MUST SET `qmm_partial_rows` ITSELF, OR IT MEASURES NOTHING
///
/// `backend_facts()` is `LlamaLikeMetalFacts::synthetic()`, whose
/// `qmm_partial_rows` is `false`. With it false the projections' guard is
/// `TokensMultipleOf(32)`, which REFUSES 495 -- so the fire this test calls
/// "the tiled GEMM" falls to the matvec, and the comparison it makes is the
/// matvec against the matvec. It read 0.9% and looked like a pass.
///
/// Turning the flag on is what makes the two sides different kernels, and the
/// number then reads the same as every aligned length does:
///
/// ```text
///   n=480  partial_rows=false  1.5580   (aligned: the GEMM runs regardless)
///   n=495  partial_rows=false  0.0092   (VACUOUS: matvec against matvec)
///   n=495  partial_rows=true   1.4046
///   n=496  partial_rows=true   1.0806
///   n=512  partial_rows=true   1.7608
/// ```
///
/// Which is the finding: the divergence tracks whether the GEMM RAN, and not
/// whether its last tile was ragged. Varying the tile instead of the length
/// says it from the other side -- `n=480` diverges at `bm=16` and `bm=32`,
/// which divide it, and agrees at `bm=64`, which does not and therefore
/// refuses the fire.
///
/// So the partial tile was never the defect -- the staged fp16 activation
/// was, and the stamp that turns it off is what makes both of these pass.
/// This one still earns its place: it is the only test in the tree that fires
/// the GEMM at a row count no tile divides, which is the claim the `row >=
/// params.m` bound in `write_out` exists to make.
#[test]
fn the_tiled_gemm_answers_the_way_the_vector_kernel_does_at_a_partial_tile() {
    let mut p = prompt();
    while p.len() < 495 {
        p.extend_from_slice(&PERIOD);
    }
    p.truncate(495);
    let Some(worst) = worst_disagreement_with(p, true) else {
        return;
    };
    assert!(
        worst <= 0.05,
        "at 495 rows -- a count no tile this backend can pick divides -- the \
         tiled GEMM and the matvec part by {:.1}% of the row's peak",
        100.0 * worst
    );
}

/// THE READOUT MOVES WHEN THE LAST TOKEN MOVES, WITH THE TILED GEMM FIRING.
///
/// This is the property `e0a2f6e20` withdrew the GEMM over, and it is a
/// different question from [`the_tiled_gemm_answers_the_way_the_vector_kernel_does`]
/// even though both fire the same kernel. That one asks whether two families
/// agree; this asks whether the answer is a function of the input at all. A
/// GEMM that returned one fixed row would pass neither, but a GEMM that
/// returned the WRONG row -- row 0's product where the readout row belongs --
/// would agree with the matvec on row 0 and still answer the same thing to
/// every prompt.
///
/// A causal model's last row attends to every token including the last, so
/// changing the last token MUST change the readout. `e0a2f6e20` measured this
/// against the daemon on Llama-3.2-1B q4 and read:
///
/// ```text
///     n= 16  differs      (matvec: the projections' guard needs 32 rows)
///     n= 31  differs      (matvec)
///     n= 32  IDENTICAL    (tiled GEMM) -- and degenerate, `1494` x8
///     n= 64  IDENTICAL    (tiled GEMM)
///     n= 96  IDENTICAL    (tiled GEMM)
/// ```
///
/// It is written here, in the tree, rather than left as a shell transcript,
/// because the transcript is what let the defect be diagnosed once and then
/// re-diagnosed from a commit message.
#[test]
fn the_tiled_gemms_readout_moves_when_the_last_token_moves() {
    let Some(_held) = gpu() else { return };
    let Some(real) = weights() else { return };
    let facts = facts();
    let backend = LlamaLikeMetalFacts {
        qmm_multi_batch: true,
        // Or 33 and 495 below measure the MATVEC: `backend_facts()` inherits
        // `synthetic()`'s `false`, under which the projections' guard is
        // `TokensMultipleOf(32)` and refuses every count no tile divides.
        qmm_partial_rows: true,
        ..backend_facts()
    };

    let read = |tokens: &[u32]| -> Vec<f32> {
        let mut shell = shelled_facts(real, 96, false, &facts, &backend);
        let step = shell
            .step(&[Turn {
                who: 1,
                tokens: tokens.to_vec(),
            }])
            .unwrap_or_else(|e| panic!("the fire: {e}"));
        step.logits
            .row(step.readout_of[0])
            .expect("the readout row")
            .to_vec()
    };

    // Every length the GEMM's guard admits, including counts no tile divides:
    // the withdrawn defect was invisible below 32 because the guard refused
    // the fire there, so a sweep that stopped at one length would have read
    // the matvec and called it the GEMM.
    let base = prompt();
    for n in [32usize, 33, 64, 96, 128, 255, 256, 495, 512] {
        let mut a = Vec::new();
        while a.len() < n {
            a.extend_from_slice(&base);
        }
        a.truncate(n);
        let mut b = a.clone();
        // The LAST token, and to a value the prompt does not already end in.
        let last = b.len() - 1;
        b[last] = if a[last] == PERIOD[0] {
            PERIOD[1]
        } else {
            PERIOD[0]
        };
        assert_ne!(a[last], b[last], "the perturbation must perturb");

        let (ra, rb) = (read(&a), read(&b));
        let peak = ra.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        let moved = ra
            .iter()
            .zip(&rb)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        eprintln!("  n={n:4} last token moved the readout by {:.3} ({:.1}% of peak)",
            moved, 100.0 * moved / peak);
        assert!(
            moved > 0.01 * peak,
            "at {n} rows the readout moved by {:.4} -- {:.3}% of its own peak \
             -- when the LAST token changed. A causal model's last row attends \
             to the last token, so this is the wrong row and not a near miss.",
            moved,
            100.0 * moved / peak
        );
    }
}

fn the_two_families_agree(prompt: Vec<u32>) {
    let Some(worst) = worst_disagreement(prompt) else {
        return;
    };
    assert!(
        worst <= 0.05,
        "the tiled GEMM and the matvec differ by {:.1}% of the row's peak \
         -- too much to be bf16 rounding between two matmul orders",
        100.0 * worst
    );
}

/// The widest gap between the two families on one prompt, as a fraction of
/// the tiled row's own peak. `None` when the fixture is not here.
fn worst_disagreement(prompt: Vec<u32>) -> Option<f32> {
    worst_disagreement_with(prompt, false)
}

/// [`worst_disagreement`], with `qmm_partial_rows` set -- which a caller
/// firing a row count the tile does not divide MUST do, or the guard refuses
/// the GEMM and both sides run the matvec.
fn worst_disagreement_with(prompt: Vec<u32>, partial_rows: bool) -> Option<f32> {
    let _held = gpu()?;
    let real = weights()?;
    let facts = facts();
    let backend = LlamaLikeMetalFacts {
        qmm_partial_rows: partial_rows,
        // The point of these two tests: they compare the two FAMILIES, so the
        // tiled one has to be reachable. `backend_facts()` inherits the
        // deployment's `qmm_fp16_precast: false` with it.
        qmm_multi_batch: true,
        ..backend_facts()
    };

    let answer = |vector: bool| -> Vec<f32> {
        // ENOUGH PAGES FOR THE LONGEST PROMPT THIS FILE FIRES, which is 495
        // and not 32: 24 pages refused one with "this growth needs 31 more
        // pages and 24 are free".
        let mut shell = shelled_facts(real, 96, vector, &facts, &backend);
        let step = shell
            .step(&[Turn {
                who: 1,
                tokens: prompt.clone(),
            }])
            .unwrap_or_else(|e| panic!("the fire: {e}"));
        step.logits
            .row(step.readout_of[0])
            .expect("the readout row")
            .to_vec()
    };

    let tiled = answer(false);
    let vector = answer(true);
    assert_eq!(tiled.len(), vector.len());
    // Not a constant row either way: an affine dequantisation of a degenerate
    // weight block is a constant, and two matmuls of a constant agree whatever
    // they do with it.
    assert!(
        tiled.iter().any(|v| (*v - tiled[0]).abs() > 1e-3),
        "the tiled answer is one value repeated, so this comparison is vacuous"
    );

    // ABSOLUTE, against the row's peak -- not per element.
    //
    // `driver-vulkan` normalises per element with a flat `1e-3` floor, which
    // is right for the synthetic fill it runs on and wrong for a real logit
    // row: this one spans about ±15 and has thousands of entries near zero, so
    // `0.021` against `-0.021` -- four hundredths apart, which is rounding --
    // scores 1.99 relative and fails a 0.05 check on nothing at all. Measured
    // here before the form was changed, and it is exactly the trap
    // `.wiki/new-driver/wgpu.md` §8 records: *"scale the tolerance by the
    // row's own largest magnitude"*. Flooring the denominator at 2% of the
    // peak instead still reported 0.33, on `0.227` against `0.363` -- an
    // absolute gap of fourteen hundredths, under one percent of the peak.
    //
    // So the claim is absolute and the scale is the row's. What it measures is
    // whether the two families agree to a fraction of what the model is
    // actually distinguishing.
    let peak = tiled.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (a, b)) in tiled.iter().zip(&vector).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "a non-finite logit at {i}");
        let off = (a - b).abs();
        if off > worst {
            worst = off;
            at = i;
        }
    }
    println!(
        "the two families part by {worst} at token {at} ({} vs {}), which is \
         {:.1}% of the row's peak ({peak})",
        tiled[at],
        vector[at],
        100.0 * worst / peak
    );
    // And they are not the SAME kernel by accident: a difference of exactly
    // zero would mean the plan swap did not change which code ran, and every
    // number above would be a comparison of a buffer with itself.
    assert!(
        worst > 0.0,
        "the two plans answered bit-identically, so the swap did not change \
         which kernels ran"
    );
    Some(worst / peak)
}

/// A copy plan whose destination is above the pool GROWS it rather than being
/// refused.
///
/// # What found this, and why nothing here could
///
/// `driver-vulkan`'s curated sweep, on `prefix-tree-kv-cache`, and only when
/// it ran after the other thirty-eight:
///
/// ```text
///   pre-launch KV copy rejected: driver-vulkan: page move 0's destination
///   names page 3 row 0, and the pool has 3 pages of 16 rows
/// ```
///
/// Run alone it passed. That is the signature of a driver whose answer
/// depends on what preceded it, and the reason is that this pool is ELASTIC:
/// it holds what the frames so far have needed, not what the scheduler is
/// entitled to hand out. `Shell::admit` knows that and grows to the highest
/// page a frame NAMES. `Shell::copy_kv` was the other door a page number
/// comes through and did not: it went straight to `Pool::copy_plan`, whose
/// bounds check is right about the pool as it IS and has no way to know what
/// it could be.
///
/// This backend's pool is elastic in the same way and had the same gap. The
/// defect was ported here by reading the sibling's fix rather than by waiting
/// for the sweep to reproduce it, because the sweep would have -- the two
/// drivers share the engine that builds these plans.
///
/// Nothing in this crate could have caught it: the pool's elasticity is
/// tested in `frames::pages_named` and `Shell::admit`, because that is where
/// it was written, and `copy_plan`'s tests are arithmetic on a pool big
/// enough for them. Neither suite could ask the other's question.
///
/// # What this measures
///
/// Both directions of the asymmetry, because the fix is not "grow for
/// anything named":
///
/// 1. a DESTINATION above the pool grows it, and the bytes land -- read back
///    and compared against the source page, so a growth that reallocated
///    without carrying the contents over fails here too;
/// 2. a SOURCE above the pool is still REFUSED, and the pool does not grow
///    for it. A page this pool has never held is a page nothing has ever
///    written, so growing would turn a refusal into a copy of fresh zeros:
///    history-shaped silence rather than an error.
#[test]
fn a_copy_plan_that_names_a_page_past_the_pool_grows_it_instead_of_refusing() {
    let Some(_held) = gpu() else { return };
    let Some(real) = weights() else { return };
    // Small on purpose: the sweep's pool was three pages because three was
    // all its prefills had asked for.
    let mut shell = shelled(real, 3);

    // Real history in page 0, so "the bytes land" is about a cache and not
    // about a buffer somebody wrote a pattern into.
    shell
        .step(&[Turn {
            who: 1,
            tokens: PERIOD[..4].to_vec(),
        }])
        .expect("a first turn");
    assert_eq!(
        shell.shape().pages,
        3,
        "this test wants a pool it can name past the end of"
    );

    let page_of = |shell: &Shell, page: u32| -> Vec<u8> {
        let shape = shell.shape();
        let buffer = shell.pool().cache(0, false).expect("layer 0 keys");
        let at = shape.slot(page, 0, 0, 0) * shape.bytes as u64;
        let n = shape.page_size as u64 * shape.row() * shape.bytes as u64;
        shell
            .device()
            .read_at(buffer, at, n)
            .expect("a page of keys")
    };
    let source = page_of(&shell, 0);
    assert!(
        source.iter().any(|b| *b != 0),
        "page 0 holds no history, so a copy of it proves nothing"
    );

    // The sweep's plan, in miniature: page 0 to a page the pool does not have.
    let moved = shell
        .copy_kv(&driver_api::KvCopyPlan {
            src_domain: driver_api::PIE_MEMORY_DOMAIN_WEBGPU_DEVICE,
            dst_domain: driver_api::PIE_MEMORY_DOMAIN_WEBGPU_DEVICE,
            src_page_ids: vec![0],
            dst_page_ids: vec![3],
            ..driver_api::KvCopyPlan::default()
        })
        .expect("a copy to page 3 of a 3-page pool grows the pool");
    assert_eq!(moved, 1, "one page move");
    assert_eq!(
        shell.shape().pages,
        4,
        "the pool grew to something other than the page the plan named"
    );
    assert_eq!(
        page_of(&shell, 3),
        source,
        "the destination does not hold the source's bytes, so either the copy \
         did not happen or the growth dropped what the pool was holding"
    );
    assert_eq!(
        page_of(&shell, 0),
        source,
        "the growth lost the page the copy read from"
    );

    // The other direction stays a refusal.
    let refused = shell
        .copy_kv(&driver_api::KvCopyPlan {
            src_domain: driver_api::PIE_MEMORY_DOMAIN_WEBGPU_DEVICE,
            dst_domain: driver_api::PIE_MEMORY_DOMAIN_WEBGPU_DEVICE,
            src_page_ids: vec![9],
            dst_page_ids: vec![1],
            ..driver_api::KvCopyPlan::default()
        })
        .expect_err("a source the pool has never held holds no history");
    assert!(
        format!("{refused:?}").contains("page 9"),
        "the refusal does not name the page that caused it: {refused:?}"
    );
    assert_eq!(
        shell.shape().pages,
        4,
        "a refused copy grew the pool anyway"
    );
}

/// A shell reads each shader module ONCE, however many steps it serves and
/// however many launches each step fires.
///
/// # What this is a control for
///
/// A 25x regression, and a quiet one. `serve::fire` used to expand the WGSL
/// source and run a `naga` parse once per LAUNCH -- 452 times a step over ten
/// distinct symbols -- which was 95% of a decode. It was fixed in two steps,
/// deduplicating within a fire and then caching across fires, and neither of
/// those changes anything a correctness test can see. The suite went from
/// 252 s to 157 s and every assertion in it stayed exactly as green as before.
///
/// So the control is a COUNT. `modules_read` is the number of cache misses,
/// because a miss is the only thing that inserts, and this pins two things:
///
/// * it stops growing -- a fire that re-reads a module it has already read
///   moves the number, and thirty more steps do not;
/// * it is the number of distinct SYMBOLS, not launches. A step fires
///   hundreds of dispatches over ten or so kernels, so a count in the
///   hundreds means the per-launch dedup is gone even if the cross-step cache
///   is still there.
///
/// It also pins the cache against the pipelines' own. Not by COUNT, which is
/// what this used to do and which was wrong in the direction nobody checked:
/// a fire reads a module for every symbol its LOWERING names and builds a
/// pipeline only for the ones a guard lets dispatch, so twelve reads against
/// eleven builds is a plan carrying both arms of the decode-attention switch
/// and firing one. `sdpa_paged_decode_bfloat16_d_128` is the arm not taken --
/// `sdpa_paged_decode_split` and `..._merge` are, because split-K wins at
/// these key counts -- and its module is expanded and reflected all the same,
/// since that happens before any guard has spoken.
///
/// What must hold is the SUBSET: every pipeline built came from a module this
/// cache read. The other way round is the drift worth naming, because the two
/// are keyed on the REQUESTED tier and the LANDED one -- which differ whenever
/// an adapter asks for a tier the tree has no variant of, as this one does at
/// every symbol: `Subgroup` asked, `Baseline` landed, twelve times out of
/// twelve.
#[test]
fn a_shell_reads_each_module_once_however_many_steps_it_serves() {
    let Some(_held) = gpu() else { return };
    let Some(real) = weights() else { return };
    let mut shell = shelled(real, 8);
    assert_eq!(shell.modules_read(), 0, "nothing is read before a fire");

    let prompt = prompt();
    shell
        .step(&[Turn {
            who: 1,
            tokens: prompt[..8].to_vec(),
        }])
        .expect("a prefill");
    let after_one = shell.modules_read();
    assert!(
        (1..=32).contains(&after_one),
        "one fire read {after_one} modules; a step fires hundreds of dispatches \
         over a dozen or so kernels, so a number in the hundreds is the \
         per-launch expansion back again"
    );

    // A DECODE, which lowers the other plan and so may name symbols the
    // prefill did not: the count may rise once and then must settle.
    for token in 0..12u32 {
        shell
            .step(&[Turn {
                who: 1,
                tokens: vec![prompt[(token as usize) % prompt.len()]],
            }])
            .expect("a decode");
    }
    let after_decodes = shell.modules_read();
    let asked_then = shell.modules_asked();

    for token in 0..12u32 {
        shell
            .step(&[Turn {
                who: 1,
                tokens: vec![prompt[(token as usize) % prompt.len()]],
            }])
            .expect("another decode");
    }
    assert_eq!(
        shell.modules_read(),
        after_decodes,
        "twelve more decodes EXPANDED more modules, so the cache is not \
         holding what it read and every step is paying the parse again"
    );
    // The other regression, which the miss count alone cannot see: a fire that
    // consults the cache once per LAUNCH instead of once per distinct symbol
    // is cheap -- every consult hits -- and is still hundreds of lookups and
    // hundreds of clones of a module's source per step.
    let per_step = (shell.modules_asked() - asked_then) / 12;
    assert!(
        per_step <= 32,
        "a step consulted the module cache {per_step} times; there are a dozen \
         or so distinct kernels in a step and hundreds of launches, so this is \
         the per-launch lookup back again"
    );
    let read = shell.read_symbols();
    let built = shell.built_symbols();
    let unread: Vec<&str> = built.difference(&read).copied().collect();
    assert!(
        unread.is_empty(),
        "{unread:?} were compiled from modules this cache never read, so the \
         module key and the pipeline key have drifted apart"
    );
    // The other direction is expected and BOUNDED: a guard may refuse an arm,
    // and a plan that named a dozen symbols and fired none of them is a shell
    // that is not serving.
    let unfired: Vec<&str> = read.difference(&built).copied().collect();
    assert!(
        unfired.len() < read.len() / 2,
        "{} of {} symbols this shell expanded were never dispatched ({unfired:?}), \
         which is more arms than any guard in this plan refuses",
        unfired.len(),
        read.len()
    );
}

/// Every row count from one to forty is servable, not just the multiples of
/// sixteen this file otherwise fires.
///
/// # The gap this closes
///
/// `prompt()` asserts `len() % 16 == 0` and every other proof here uses it, so
/// the whole serving suite has only ever exercised row counts the tiled GEMM's
/// tile divides. That is not an accident of the fixture -- it is written down
/// as a constraint -- and it means the arm a row count SELECTS has never been
/// swept.
///
/// It matters because the selection is not a threshold. `model`'s
/// `TokensMultipleOf(tile)` guard takes the tiled GEMM only when the tile
/// DIVIDES the row count, and `geometry::grid` refuses `Rule::Qmm` at a row
/// count it does not -- `Ungeometric::PartialTile`, chosen over falling back
/// to a matvec grid because `affine_qmm_t` reads its tile from the grid and a
/// two-token prefill came back entirely NaN when it did.
///
/// Those two rules have to agree for every row count, and they are in
/// different crates. When they did not, a real `pie run` of a 35-token prompt
/// died -- on Metal, Vulkan and wgpu alike -- because the guard was
/// `TokensGT(tile - 1)`, which is true for 35 and does not imply `35 % 16 ==
/// 0`. That is fixed upstream; nothing in THIS crate would notice it coming
/// back.
///
/// # What it asserts, and what it deliberately does not
///
/// That the fire is SERVED and returns a finite distribution of the right
/// width. Not what it says: forty prompts of different lengths have forty
/// different answers, and pinning them would be pinning the model. The
/// distributions are checked for being distributions -- finite, and not one
/// value repeated, which is what an unstaged buffer reads as.
#[test]
fn every_row_count_up_to_forty_is_servable_and_not_just_the_tile_multiples() {
    let Some(_held) = gpu() else { return };
    let Some(real) = weights() else { return };
    // Every count gets its own conversation so its pages are its own, which
    // is `sum(ceil(r / 16))` for r in 1..=40 -- seventy-two pages. Ninety-six
    // leaves room rather than making this a pool test by accident.
    let mut shell = shelled(real, 96);
    let period = PERIOD;

    let mut refused = Vec::new();
    for rows in 1..=40usize {
        let tokens: Vec<u32> = (0..rows).map(|i| period[i % period.len()]).collect();
        match shell.step(&[Turn {
            who: 1000 + rows as u64,
            tokens,
        }]) {
            Err(why) => refused.push(format!("{rows} rows: {why}")),
            Ok(step) => {
                let row = step
                    .logits
                    .row(step.readout_of[0])
                    .expect("a readout row for the fire that ran");
                assert!(
                    row.iter().all(|v| v.is_finite()),
                    "{rows} rows produced a non-finite logit"
                );
                assert!(
                    row.iter().any(|v| (*v - row[0]).abs() > 1e-3),
                    "{rows} rows produced one value repeated, which is what an \
                     unstaged buffer reads as rather than a distribution"
                );
            }
        }
    }
    assert!(
        refused.is_empty(),
        "these row counts are not servable, and the tile divides none of them: \
         {refused:#?}"
    );
}

/// A copy plan naming a page past what the device could EVER hold is refused,
/// and does not try to allocate it first.
///
/// # The gap this closes, which I opened
///
/// `Shell::copy_kv` grows the pool for a destination above it -- that fix is
/// two days old and it is right, because this pool holds what the frames so
/// far have needed and not what the scheduler is entitled to hand out.
///
/// `Shell::admit` does the same growth for a FRAME and asks one more question
/// first: `need > pool.ceiling(device)` answers `Launched::Impossible`, which
/// is "no growth could ever make room" rather than "not yet". `copy_kv` did
/// not ask it. The ceiling is derived from the adapter's own
/// `buffer_size`/`storage_binding_size`, so without that question a plan
/// naming a large page number sends `Pool::resize` to allocate a cache for it
/// -- `layers * 2` buffers of `pages * page_size * row * bytes` -- and the
/// refusal that comes back is an allocator's, after the attempt, not a
/// driver's before it.
///
/// This is the shape `engine`'s upload audit found twice on the same day: a
/// guard written for one dimension of a threat with the neighbouring
/// dimension left open. Here the growth was the new dimension and the ceiling
/// check stayed where it was.
///
/// # What it asserts
///
/// That the refusal happens, that it names the pool, and -- the part worth
/// having -- that the POOL IS UNCHANGED afterwards. A pool that half-grew
/// before failing would leave some layers at the new page count and some at
/// the old, and `Shape::slot` would index every one of them wrongly.
#[test]
fn a_copy_plan_past_what_the_device_could_hold_is_refused_before_it_allocates() {
    let Some(_held) = gpu() else { return };
    let Some(real) = weights() else { return };
    let mut shell = shelled(real, 4);
    let was = shell.shape().pages;

    // Past any adapter's `buffer_size`: one page of this model is 8 KiB of
    // keys per layer, so a billion of them is terabytes.
    let refused = shell
        .copy_kv(&driver_api::KvCopyPlan {
            src_domain: driver_api::PIE_MEMORY_DOMAIN_WEBGPU_DEVICE,
            dst_domain: driver_api::PIE_MEMORY_DOMAIN_WEBGPU_DEVICE,
            src_page_ids: vec![0],
            dst_page_ids: vec![1_000_000_000],
            ..driver_api::KvCopyPlan::default()
        })
        .expect_err("a page past the device's own budget is not a page to grow to");
    let text = format!("{refused:?}");
    assert!(
        text.contains("1000000000") && text.contains("pool"),
        "the refusal names neither the page the plan asked for nor the pool's \
         own ceiling, so a caller is left with an allocator's buffer size: \
         {refused:?}"
    );
    assert_eq!(
        shell.shape().pages,
        was,
        "a refused copy resized the pool anyway"
    );
}

/// A pool resize past what the adapter could hold is refused before the BOOK
/// allocates for it.
///
/// # The dimension that was open
///
/// `Shell::resize_pool` had two guards and they cover the neighbours of the
/// problem rather than the problem. `u32::try_from` catches a target past
/// `u32::MAX`; `Device::zeroed` catches one past the adapter's buffer limit,
/// in a comparison, before allocating. Between them sits every number that
/// fits in a `u32`, is past what the adapter could hold, and is large enough
/// to hurt on the way there.
///
/// Because the BOOK moves first. `Book::resize` builds the free list for the
/// new size -- a `Vec<u32>` with one entry per page it grew by -- so a target
/// of a billion allocates four gigabytes of HOST memory and only then reaches
/// the device that was always going to refuse it.
///
/// This is `engine`'s upload shape again -- "a guard written for one dimension
/// of a threat and left the neighbouring dimension open" -- and the third
/// place this driver has had it in two days. The other two were `copy_kv`'s
/// missing ceiling and `pages_named` reading declarations instead of
/// bindings; all three are the elastic pool, which is the youngest thing here.
///
/// # The number this uses, and why it is not a billion
///
/// A test that allocated four gigabytes to prove a point would be a test that
/// fails on a small machine for the wrong reason. This backend's ceiling on
/// this model is `4 GiB / (16 rows x 1024 elements x 2 bytes)` = 131072
/// pages, so 200_000 is past it while the free list it would have built is
/// under a megabyte. What the test pins is the REFUSAL and the book, not the
/// size of the allocation avoided.
#[test]
fn a_resize_past_the_adapters_ceiling_is_refused_before_the_book_grows() {
    let Some(_held) = gpu() else { return };
    let Some(real) = weights() else { return };
    let mut shell = shelled(real, 8);
    let pages_before = shell.shape().pages;
    let free_before = shell.book().spare();

    let refused = shell
        .resize_pool(&driver_api::PoolResizePlan {
            pool_id: driver_api::PIE_ELASTIC_POOL_KV,
            target_pages: 200_000,
            ..driver_api::PoolResizePlan::default()
        })
        .expect_err("a target past the adapter's own budget is not a target");
    let text = format!("{refused:?}");
    assert!(
        text.contains("200000"),
        "the refusal does not name the target that caused it: {refused:?}"
    );

    assert_eq!(
        shell.shape().pages,
        pages_before,
        "a refused resize moved the pool"
    );
    assert_eq!(
        shell.book().spare(),
        free_before,
        "a refused resize grew the BOOK, which is the allocation this check \
         exists to skip"
    );

    // ...and the pool still works afterwards, which is what "unchanged"
    // has to mean.
    shell
        .step(&[Turn {
            who: 7,
            tokens: PERIOD[..4].to_vec(),
        }])
        .expect("a turn after a refused resize");
}

/// A frame whose LAST step is malformed appends none of the first two.
///
/// `Shell::launch` states this and nothing checked it:
///
/// > convert every step's CSRs BEFORE firing any of them, so a frame with a
/// > malformed third step does not append the first two
///
/// with the reason beside the loop: "a frame whose third step does not close
/// its CSR would otherwise have appended the first two steps' keys, and the
/// scheduler's retry of the same frame would append them TWICE." That is a
/// corrupted cache, and the corruption is silent — doubled keys are attended
/// as history, so the run stays fluent and answers the wrong thing.
///
/// # Why the good frame is fired first
///
/// Because "the pages are still zero" proves nothing on its own: a pair of
/// steps that write nowhere satisfies it exactly as well as a refusal that
/// unwound. So the same two steps are fired alone first and the page is
/// required to CHANGE, and only then are they fired again behind a step that
/// cannot convert. Two shells, because the point is what the second one did
/// not do.
#[test]
fn a_frame_whose_last_step_is_malformed_appends_none_of_the_others() {
    let Some(_held) = gpu() else { return };
    let Some(real) = weights() else { return };

    let prompt = prompt();
    let n = prompt.len() as u32;
    // Distinct pages per step, so a step that appends is visible on ITS page
    // rather than sharing one with the step after it.
    let step = |pages: &[u32], rows: u32| driver_api::StepSubmission {
        plan: driver_api::LaunchPlan {
            token_ids: prompt.clone(),
            position_ids: (0..n).collect(),
            kv_page_indices: pages.to_vec(),
            kv_page_indptr: vec![0, pages.len() as u32],
            kv_last_page_lens: vec![n % 16],
            qo_indptr: vec![0, rows],
            sampling_indices: vec![n - 1],
            sampling_indptr: vec![0, 1],
            ..driver_api::LaunchPlan::default()
        },
        roster_rows: vec![0],
        sub_batch_indptr: vec![0, 1],
        sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE],
        terminal_cells: Vec::new(),
        program_row_indptr: vec![0, 1],
        logical_fire_ids: vec![0],
        channel_expected_head: Vec::new(),
        channel_expected_tail: Vec::new(),
        channel_ticket_indptr: vec![0, 0],
        region_row_indptr: Vec::new(),
        region_sig: Vec::new(),
        region_k: Vec::new(),
    };
    let pages_of = |steps: Vec<driver_api::StepSubmission>| driver_api::FrameSubmission {
        instance_ids: vec![1],
        kv_translation: (0..6).collect(),
        kv_translation_indptr: vec![0, 6],
        required_kv_pages: 6,
        steps,
    };
    let keys = |shell: &Shell, page: u32| -> Vec<u8> {
        let shape = shell.shape();
        let buffer = shell.pool().cache(0, false).expect("layer 0 keys");
        let at = shape.slot(page, 0, 0, 0) * shape.bytes as u64;
        let n = shape.page_size as u64 * shape.row() * shape.bytes as u64;
        shell
            .device()
            .read_at(buffer, at, n)
            .expect("a page of keys")
    };

    // ── The premise: these two steps DO append, so a zero page afterwards is
    //    a refusal that unwound and not a pair of steps that write nowhere. ──
    let mut ok = shelled(real, 32);
    assert!(
        keys(&ok, 0).iter().all(|b| *b == 0),
        "a fresh pool starts zeroed, or the comparison below means nothing"
    );
    ok.launch(&pages_of(vec![step(&[0, 1], n), step(&[2, 3], n)]))
        .expect("two well-formed steps");
    let (wrote_first, wrote_second) = (keys(&ok, 0), keys(&ok, 2));
    assert!(
        wrote_first.iter().any(|b| *b != 0) && wrote_second.iter().any(|b| *b != 0),
        "neither step appended anything, so this test cannot see the difference \
         it exists to see"
    );

    // ── The claim: the same two, behind a step whose CSR does not close. ──
    let mut refused = shelled(real, 32);
    // `qo_indptr` says three rows over a plan holding `n` tokens: the CSR does
    // not describe the fire, which is what `prepare` refuses.
    let bad = step(&[4, 5], n + 3);
    let e = refused
        .launch(&pages_of(vec![step(&[0, 1], n), step(&[2, 3], n), bad]))
        .expect_err("a frame whose last step does not convert is not launchable");

    for page in 0..6 {
        assert!(
            keys(&refused, page).iter().all(|b| *b == 0),
            "page {page} was written by a frame that was refused ({e}). The \
             first two steps appended before the third was checked, so the \
             scheduler's retry of this frame appends them a second time and \
             the conversation attends its own keys twice"
        );
    }
}

/// A step of no turns is refused by name, before anything is staged.
///
/// One of twenty-three refusals this crate constructs and no test named --
/// see `every_refusal_this_crate_builds_is_one_a_test_names` in
/// `tests/citations.rs`, which is the census that found it.
///
/// `device.rs`'s `Failed` says why naming matters: it is compared BY VALUE,
/// "because a test that asserts WHICH refusal came back is the only way an
/// alignment failure stays distinguishable from a length one". A refusal
/// nothing names is one whose condition could be inverted, or whose message
/// could describe a different fault, with every suite still green.
///
/// This one needs no weights, which is the reason it is worth having beyond
/// the census: it is the whole `Shell::on` -> `step` path on a shell that has
/// been given nothing, so it runs on any machine with an adapter and would
/// catch a `turns.is_empty()` that had been dropped or inverted.
#[test]
fn a_step_of_no_turns_is_refused_and_not_served() {
    let Some(_held) = gpu() else { return };
    let Ok(device) = opened() else { return };

    let facts = facts();
    let mut shell = Shell::on(
        device,
        Text {
            decode: llama_like_metal(&facts, &backend_facts(), FireClass::Decode),
            prefill: llama_like_metal(&facts, &backend_facts(), FireClass::Prefill),
            geometry: Geometry {
                q_heads: facts.q_heads,
                kv_heads: facts.kv_heads,
                head_dim: facts.head_dim,
                rotary_dims: facts.head_dim,
                n_experts: 0,
                experts_per_token: 0,
                ..Default::default()
            },
            layers: facts.layers as u16,
        },
        Deployment {
            pages: 4,
            theta: 1_000_000.0,
            ..Deployment::default()
        },
    )
    .expect("a shell with no weights is still a shell");

    let refused = shell
        .step(&[])
        .expect_err("a step of no turns has nothing to serve");
    assert!(
        matches!(refused, driver_wgpu::turns::Unstepped::Nothing),
        "a step of no turns came back as `{refused}` rather than `Nothing`, so \
         whatever it did instead ran on an empty roster"
    );
    assert!(
        refused.to_string().contains("no turns"),
        "the refusal must say what was wrong: {refused}"
    );
}

/// A run of decodes derives ONE lowering, and the answers do not change.
///
/// `lower` is a pure function of the plan, the rows and the fire flag, and a
/// one-token decode varies none of them: `Row` carries flags only — no
/// position, no length — so the graph of the token at position 40 IS the
/// graph of the token at position 33. This driver re-derived it anyway, 0.765
/// ms of 452 launches per token, which `lowering::cached` now keeps.
///
/// Two claims, and the second is the one that matters:
///
/// * the cache HITS — one prefill shape and one decode shape over a whole
///   generation, so `lowerings_derived()` reaches 2 and stops;
/// * the tokens are the SAME ones the uncached driver produced. A cache that
///   served a stale or foreign graph would still generate fluent text, which
///   is why this compares against a second shell that sees each shape once
///   and therefore never takes a hit at all.
///
/// Falsified by keying [`Shape`] on the rows alone: the second shell's
/// prefill and the first shell's decode share a key, and the token sequences
/// part on the first decode.
#[test]
fn a_run_of_decodes_derives_one_lowering_and_says_the_same_thing() {
    let _lock = gpu();
    let Some(real) = weights() else {
        eprintln!("no checkpoint; skipped");
        return;
    };
    let prompt = prompt();

    let mut shell = shelled(real, 64);
    assert_eq!(
        shell.lowerings_derived(),
        0,
        "nothing lowered before a step"
    );

    let mut cached_tokens = Vec::new();
    let first = shell
        .step(&[Turn {
            who: 1,
            tokens: prompt.clone(),
        }])
        .expect("the prompt fires");
    assert_eq!(
        shell.lowerings_derived(),
        1,
        "the prefill is the first shape"
    );
    let mut next = argmax(
        first
            .logits
            .row(first.readout_of[0])
            .expect("the prefill read out"),
    );
    cached_tokens.push(next);

    for step in 0..8 {
        let out = shell
            .step(&[Turn {
                who: 1,
                tokens: vec![next],
            }])
            .expect("a decode fires");
        next = argmax(
            out.logits
                .row(out.readout_of[0])
                .expect("a decode reads out"),
        );
        cached_tokens.push(next);
        assert_eq!(
            shell.lowerings_derived(),
            2,
            "decode {step} derived a lowering; the shape is supposed to be a \
             constant, so either `Row` gained a per-step field or the key did"
        );
    }

    // THE SAME ANSWERS, from this driver with the cache switched off.
    //
    // Not a fresh shell re-prefilling a growing history: that would compare
    // the tiled GEMM against the matvec (two kernel families the oracle only
    // holds to 0.05 * peak, so their argmaxes may honestly part) and would
    // anyway be REFUSED at the first odd length -- `geometry.rs` takes whole
    // 16-row tiles. Clearing before each step keeps every other thing equal
    // and makes each step a miss.
    let mut plain = shelled(real, 64);
    let mut plain_tokens = Vec::new();
    plain.forget_lowerings();
    let out = plain
        .step(&[Turn {
            who: 1,
            tokens: prompt.clone(),
        }])
        .expect("the prompt fires");
    let mut tok = argmax(out.logits.row(out.readout_of[0]).expect("it read out"));
    plain_tokens.push(tok);
    for _ in 0..8 {
        plain.forget_lowerings();
        let before = plain.lowerings_derived();
        let out = plain
            .step(&[Turn {
                who: 1,
                tokens: vec![tok],
            }])
            .expect("a decode fires");
        assert_eq!(
            plain.lowerings_derived(),
            before + 1,
            "a cleared cache must MISS, or this is not the uncached driver \
             and the comparison below proves nothing"
        );
        tok = argmax(
            out.logits
                .row(out.readout_of[0])
                .expect("a decode reads out"),
        );
        plain_tokens.push(tok);
    }

    assert_eq!(
        cached_tokens, plain_tokens,
        "the cached driver and the uncached one disagree about what this \
         model says"
    );
}

/// A real fire is ONE command buffer and shadows NOTHING.
///
/// Two findings, one after the other, and this is what is left of both.
///
/// # 735 command buffers
///
/// `Device::run_all` opened a fresh encoder either side of every shadow point
/// — a `copy_buffer_to_buffer` cannot be encoded inside a compute pass — and
/// 451 of a 452-launch decode's rectangles shadowed something, so the queue
/// was given 735 command buffers for one token. Ending a PASS is not ending
/// an ENCODER, and nothing was bought by the split: command buffers in one
/// `submit` run in order, commands in one command buffer run in order, and
/// `wgpu-core` emits the barrier between a copy and the pass that reads it
/// either way. That took a fire from 31.9 ms to 20.5 ms.
///
/// `Fired::submissions` did not catch it because it was the literal `1`,
/// counting `queue.submit` CALLS under a name and a doc that said command
/// buffers.
///
/// # And then no shadow at all
///
/// The copies were there because WebGPU refuses one buffer bound both
/// readable and writable in one dispatch — but two `read_write` bindings are
/// the same usage BIT, so the whole workaround was avoidable by declaring the
/// read side `read_write` too. The shader tree now does
/// (`kernels-wgpu`'s `no_shader_declares_a_read_only_storage_binding`), and a
/// decode went 25.1 ms to 11.2 ms, 39.8 to 89.3 tok/s.
///
/// So `shadowed` is asserted ZERO, which is the assertion that would catch
/// one new `var<storage, read>` anywhere in the tree — the change whose only
/// other symptom is that decoding got twice as slow.
///
/// Falsified by restoring the per-segment encoder (735 buffers) and by
/// restoring one `read` declaration (451 shadows).
///
/// RETIRED WITH `kernels-wgpu`'s TEST TREE. That name is a record of a
/// measurement now, not a live proof: the crate lost `tests/` and every
/// in-file `mod tests` when the three shader planes moved their numbers to
/// the fire that reads them, and nothing in this workspace re-runs it. What
/// it reported is still why the sentence above says what it says; what is
/// gone is the thing that would notice if it stopped being true.
#[test]
fn a_real_fire_is_one_command_buffer_and_shadows_nothing() {
    let _lock = gpu();
    let Some(real) = weights() else {
        eprintln!("no checkpoint; skipped");
        return;
    };
    let mut shell = shelled(real, 64);
    let out = shell
        .step(&[Turn {
            who: 1,
            tokens: prompt(),
        }])
        .expect("the prompt fires");
    assert_eq!(
        out.fired.submissions, 1,
        "a prefill of {} dispatches went to the queue as {} command buffers",
        out.fired.dispatches, out.fired.submissions
    );
    assert_eq!(
        out.fired.shadowed, 0,
        "{} of {} rectangles copied a read operand out of the arena. Some \
         shader declares `var<storage, read>` again, and the only other \
         symptom is that this got twice as slow.",
        out.fired.shadowed, out.fired.dispatches
    );

    let next = argmax(out.logits.row(out.readout_of[0]).expect("read out"));
    let decode = shell
        .step(&[Turn {
            who: 1,
            tokens: vec![next],
        }])
        .expect("a decode fires");
    assert_eq!(decode.fired.submissions, 1);
    assert_eq!(decode.fired.shadowed, 0);
    // And the fire is still 452 rectangles, so this is not one command buffer
    // for the uninteresting reason that the plan shrank.
    assert!(
        decode.fired.dispatches > 400,
        "a decode of {} dispatches is not the plan this test was written \
         against",
        decode.fired.dispatches
    );
}

/// What a decode costs at a long context.
///
/// **`#[ignore]`, and it is a measurement rather than an assertion.** The rest
/// of this file prompts [`prompt`]'s thirty-two tokens, where attention is a
/// rounding error and every per-key cost in `sdpa_paged` is invisible. Two
/// findings ported from `kernels-metal` were deferred for exactly that reason
/// — there was nothing here that could tell whether they helped.
///
/// At 512 of context, an RTX 4090, medians of forty decodes, three runs each:
///
/// | | ms |
/// | --- | --- |
/// | before | 25.4, 25.7, 27.1 |
/// | V load and page base hoisted | 23.6, 22.2, 22.1 |
///
/// ~3.5 ms, 13.5 %. Three runs because ONE is not enough to say anything: the
/// same binary measured 22.36 and 23.77 on consecutive runs of this probe, and
/// a single-sample comparison across a change of that size says whatever the
/// machine was doing. A third finding — caching the physical page across the
/// positions that share it — was tried, measured inside that noise, and
/// reverted rather than kept on the strength of one sample.
///
/// Run with `--ignored --nocapture`.
///
/// # WHERE A DECODE'S 10 ms GOES: NOWHERE IN PARTICULAR
///
/// The prefix sweep below steps the same knob
/// [`where_a_prefills_time_goes_across_its_plan`] uses. On a 512-key decode
/// of qwen3-0.6b, Apple M4 Pro, `--release`, 480 rectangles:
///
/// ```text
///   first    0 rectangles    0.276 ms
///   first   30 rectangles    1.027 ms   (+0.750)
///   first  120 rectangles    2.687 ms   (+0.542)
///   first  240 rectangles    5.023 ms   (+0.497)
///   first  360 rectangles    7.224 ms   (+0.667)
///   first  480 rectangles    9.665 ms   (+0.642)
/// ```
///
/// **Flat.** Every block of thirty rectangles costs 0.45 to 0.75 ms wherever
/// it sits, and one layer taken a rectangle at a time is flatter still --
/// deltas of 0.03 to 0.2 ms against a per-point spread of 0.1, with no line
/// standing out the way `sdpa_paged_tiled` stands out of a prefill's layer.
///
/// So: `(9.665 - 0.276) / 480` is **19.6 microseconds a rectangle**, and that
/// number does not care what the rectangle computes. A 512-row prefill is
/// 445 ms over 452 rectangles, 984 microseconds each, which is work. A decode
/// is 480 dispatches of almost nothing. **The decode is dispatch-bound**, and
/// the lever is the COUNT -- which is what [`fuse-qkv-quant`]-shaped work is
/// for -- rather than any kernel in it.
///
/// ## The tail this first reported was the measurement drifting under itself
///
/// Taken in order, the same sweep read +1.635 and +1.540 ms over the last
/// sixty rectangles and looked like a third of a decode sitting in the lm
/// head. It was not. **Every fire appends a key**, so a sweep of sixteen
/// points at fifteen fires each is hundreds of tokens of context growth from
/// its first point to its last, and taken in order all of that growth lands
/// on the last points. Visiting the points round-robin and keeping the
/// fastest of three rounds makes the drift common to every point instead of
/// ordered by it, and the tail disappears completely.
#[test]
#[ignore = "measurement"]
fn what_a_decode_costs_at_length() {
    release_only();
    let _lock = gpu();
    let Some(real) = weights() else {
        eprintln!("no checkpoint; skipped");
        return;
    };
    // 512, and a whole number of 16-row tiles for the reason `prompt` gives.
    let mut long: Vec<u32> = Vec::new();
    while long.len() + PERIOD.len() <= 510 {
        long.extend_from_slice(&PERIOD);
    }
    while !long.len().is_multiple_of(16) {
        long.push(PERIOD[0]);
    }
    assert_eq!(long.len(), 512);

    let mut shell = shelled(real, 1024);
    let f = shell
        .step(&[Turn {
            who: 1,
            tokens: long,
        }])
        .expect("the prompt fires");
    let mut next = argmax(f.logits.row(f.readout_of[0]).expect("it read out"));
    for _ in 0..5 {
        let o = shell
            .step(&[Turn {
                who: 1,
                tokens: vec![next],
            }])
            .expect("a decode fires");
        next = argmax(o.logits.row(o.readout_of[0]).expect("it read out"));
    }
    let mut ms = Vec::new();
    for _ in 0..40 {
        let t = std::time::Instant::now();
        let o = shell
            .step(&[Turn {
                who: 1,
                tokens: vec![next],
            }])
            .expect("a decode fires");
        ms.push(t.elapsed().as_secs_f64() * 1000.0);
        next = argmax(o.logits.row(o.readout_of[0]).expect("it read out"));
    }
    ms.sort_by(f64::total_cmp);
    let median = ms[ms.len() / 2];
    println!(
        "decode @512: median {median:.3} ms -> {:.1} tok/s",
        1000.0 / median
    );
    println!("  fastest {:.3}, slowest {:.3}", ms[0], ms[ms.len() - 1]);

    // WHERE THE 10 ms GOES, by the same prefix knob the prefill uses.
    //
    // A decode is small enough that a whole sweep is seconds rather than a
    // minute, so this takes the MEDIAN of many fires at each point rather than
    // the fastest of two -- a 10 ms fire's noise is absolute, not
    // proportional, and a median over 15 is steadier here than a minimum.
    // EVERY FIRE APPENDS A KEY, so a sweep drifts under itself: by the end of
    // one the context is hundreds of tokens longer than at the start and the
    // attention rectangles cost more for that reason alone. Taken in order,
    // the drift lands entirely on the last points and reads as a tail. So the
    // points are visited ROUND-ROBIN and the fastest of the rounds is kept:
    // the drift is then common to every point instead of ordered by it, which
    // is the same reason the prefill's sweep rounds.
    let mut decode_at = |shell: &mut Shell, n: Option<usize>, next: &mut u32| -> f64 {
        shell.fire_prefix(n);
        let mut ms = Vec::new();
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let o = shell
                .step(&[Turn {
                    who: 1,
                    tokens: vec![*next],
                }])
                .expect("a decode fires");
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
            if n.is_none() {
                *next = argmax(o.logits.row(o.readout_of[0]).expect("it read out"));
            }
        }
        shell.fire_prefix(None);
        ms.sort_by(f64::total_cmp);
        ms[ms.len() / 2]
    };
    let total = shell
        .step(&[Turn {
            who: 1,
            tokens: vec![next],
        }])
        .expect("a decode fires")
        .fired
        .dispatches;
    println!("  a decode records {total} rectangles");
    let step = total.div_ceil(16);
    let coarse: Vec<usize> = (0..=total).step_by(step).chain(std::iter::once(total)).collect();
    let lo = step;
    let hi = (step * 2).min(total);
    let fine: Vec<usize> = (lo..=hi).collect();
    let mut best = vec![f64::INFINITY; coarse.len() + fine.len()];
    for _ in 0..3 {
        for (i, &n) in coarse.iter().chain(fine.iter()).enumerate() {
            let at = decode_at(&mut shell, Some(n), &mut next);
            best[i] = best[i].min(at);
        }
    }
    println!("  cumulative, fastest of 3 round-robin rounds:");
    let mut prev = 0.0;
    for (i, &n) in coarse.iter().enumerate() {
        println!(
            "    first {n:4} rectangles {:8.3} ms   (+{:6.3})",
            best[i],
            best[i] - prev
        );
        prev = best[i];
    }
    println!("  one layer, rectangle by rectangle, fastest of 3:");
    for (j, &n) in fine.iter().enumerate().skip(1) {
        let i = coarse.len() + j;
        println!(
            "    rectangle {n:4} {:8.3} ms   (+{:6.3})",
            best[i],
            best[i] - best[i - 1]
        );
    }
}

/// **What a 512-token prefill costs, which is the half a decode cannot show.**
///
/// `#[ignore]`, and a measurement rather than an assertion.
///
/// # Why this exists beside [`what_a_decode_costs_at_length`]
///
/// A decode fires ONE row, so it reaches none of this backend's arithmetic:
/// `serve::record`'s table puts a decode's 12.7 ms in 452 launches and ~5.5 ms
/// of host work, and the kernels themselves are a rounding error inside it. A
/// prefill fires 512 rows through the tiled GEMM on the same weights, so it is
/// the only shape in this file where the WGSL is what is being timed.
///
/// The two therefore answer different questions and a single "tok/s" for this
/// backend is a category error. Both are reported per token so that neither
/// can be quoted as the other.
///
/// # The cross-runtime baseline, measured rather than recalled
///
/// llama.cpp b8994, CUDA, the same RTX 4090, the same model at a comparable
/// width — this driver quantises every weight to MLX affine-U4 at group 64
/// with an fp16 scale and an fp16 bias, which is 4.5 bits a weight and exactly
/// the 335,372,288 bytes it stages; `Q4_0` is 4.5 bits a weight too, and
/// `Q4_1` is the one that also carries a bias:
///
/// | | llama.cpp | here | |
/// | --- | --- | --- | --- |
/// | decode @512 | 676-702 tok/s | 46.4 tok/s | **15x** |
/// | prefill 512 | 40,369-41,430 tok/s | 628 tok/s | **65x** |
///
/// # Both sides in the same profile, which took two goes to get right
///
/// **Every driver figure above is `--release`, and the first set published
/// here was not.** `cargo test` builds the dev profile, this workspace sets no
/// `[profile.dev]` opt-level, and llama.cpp was configured
/// `CMAKE_BUILD_TYPE=Release` — so the original table compared a debug Rust
/// binary against an optimised C++ one and reported 34x and 270x. Rebuilt:
///
/// | | debug | release | |
/// | --- | --- | --- | --- |
/// | decode @512 | 49.2 ms | 21.6 ms | 2.3x |
/// | prefill 512 | 2709 ms | 815 ms | 3.3x |
/// | staging the weights | 105 s | 9.4 s | 11x |
///
/// The tell was in the tree the whole time and was read past: `serve::record`'s
/// cost table says *"release"* in its first line, and this file's numbers
/// disagreed with it by twelve. **A ratio between two runtimes is a claim about
/// two BUILDS**, and the profile belongs beside the hardware and the model in
/// anything that quotes one.
///
/// **That comparison is only worth the conditions it was taken in.** This
/// machine carries a permanent unrelated load — 72 % of 32 cores, nine days
/// old — and the two runtimes do not feel it equally: llama.cpp repeated
/// within 6 % (678.5, 676.1, 692.1, 701.6) while this driver's median moved
/// 49.2, 51.4 and 127.6 ms across three runs of ONE binary. The llama.cpp
/// figures above bracket the driver's runs on both sides for that reason.
///
/// So the decode's honest span is 8.8x at this driver's best recorded 12.7 ms
/// and 15x at what a loaded machine delivers, and the spread between those two
/// is itself the finding: **what varies is host time**. The prefill is 4x
/// worse than the decode's gap, and it is the half of this backend nobody had
/// measured until this test.
///
/// # Where the gap is NOT
///
/// It is not the GPU sitting behind a wrong plan. A 512-row fire records
/// **564 dispatches** against a one-row decode's 452, flat across twelve
/// consecutive fires — so the batched tiled GEMM is being chosen, and this is
/// not 512 matvecs wearing a prefill's name.
///
/// It is not the GPU being busy, either. `nvidia-smi dmon` over a whole run of
/// the decode probe returned 0 % for 93 of 96 samples and never exceeded 56 %,
/// and the card is otherwise empty (43 MiB, 0 %). The weights are 335 MB and a
/// 4090 reads them in 0.33 ms, so a 49 ms token is 148x the bandwidth floor.
/// A prefill does peg the card — 100 % for about 7 of its 44 seconds — which
/// is the one place in this driver where the WGSL, not the host, is the cost.
///
/// It is not mainly the absence of command-buffer reuse either, which was the
/// obvious guess: WebGPU has no CUDA-graph equivalent, so the whole graph is
/// re-encoded every token. But `GGML_CUDA_DISABLE_GRAPHS=1` costs llama.cpp
/// only 692 -> 459 tok/s, **1.5x of the 34x**. Graph replay is worth having and
/// is not what separates these two.
///
/// # Where it is
///
/// Two different places, which is why one number could not have found them.
///
/// The decode is launch COUNT times per-launch cost: 452 dispatches at ~15 us
/// against roughly 250 fused kernels on the other side. `serve::record`'s
/// table already names the lever — *"the way down from here is FEWER
/// LAUNCHES"* — and it is a lowering problem, not a WGSL one.
///
/// The prefill is the opposite, and the sweep this test prints is what says
/// so. Fastest of three interleaved rounds, `--release`, an RTX 4090:
///
/// | rows | ms | tok/s | ms a row |
/// | --- | --- | --- | --- |
/// | 32 | 122.4 | 261 | 3.825 |
/// | 64 | 152.5 | 420 | 2.383 |
/// | 128 | 232.1 | 551 | 1.814 |
/// | 256 | 437.7 | 585 | 1.710 |
/// | 512 | 815.3 | 628 | 1.592 |
///
/// # ON AN APPLE M4 PRO, AND IT IS THE ATTENTION -- NOT THE GEMM
///
/// Everything above was taken on an RTX 4090 and none of it transfers. Same
/// test, same checkpoint, `--release`, one sitting:
///
/// | rows | ms | tok/s | ms a row |
/// | --- | --- | --- | --- |
/// | 32 | 57.1 | 561 | 1.783 |
/// | 64 | 130.0 | 492 | 2.031 |
/// | 128 | 337.2 | 380 | 2.634 |
/// | 256 | 975.6 | 262 | 3.811 |
/// | 512 | 3031.1 | 169 | 5.920 |
///
/// The per-row cost RISES, where the 4090's fell. Fitting `t = a*n + b*n^2`
/// to the ends puts `b*n^2` at **75% of a 512-row prefill**, so this is not a
/// tuning gap, it is a quadratic term that the shorter lengths hide.
///
/// ## Three fires that say which kernel it is
///
/// The plan has two knobs that can be turned separately -- `qmm_multi_batch`
/// picks the projections' family, and putting the DECODE text in the prefill
/// slot ([`shelled_with`]'s `vector`) picks the attention's -- so the two can
/// be told apart rather than argued about. At 512 rows:
///
/// | plan | projections | attention | ms |
/// | --- | --- | --- | --- |
/// | prefill | tiled GEMM | `sdpa_paged_tiled` | 3031 |
/// | prefill | matvec | `sdpa_paged_tiled` | 3387 |
/// | decode | matvec | `sdpa_paged_decode` | 932 |
///
/// Rows 2 and 3 share their projections exactly, so their difference is one
/// kernel: **`sdpa_paged_tiled_bfloat16_d_128` costs 2455 ms of a 3387 ms
/// prefill, 72% of it, and the decode attention does the same work 3.6x
/// cheaper.** The same fit on the decode-attention plan puts its quadratic
/// term at 23% rather than 75% -- a 10.6x smaller `b`.
///
/// And rows 1 and 2 say the thing that would otherwise have been assumed:
/// **the tiled GEMM is 12% FASTER than the matvec here**, so it is not the
/// prefill's problem and `qmm_multi_batch: true` is earning its stamp. The
/// first reading of this session went the other way -- decode text against
/// prefill text, 932 against 3031, "the matvec wins 3.3x" -- because that
/// comparison moved both knobs at once. A two-way comparison over a plan with
/// two knobs cannot attribute anything.
///
/// [`which_tile_a_512_row_prefill_wants`] closes the same door from the other
/// side: nine tiles spanning a 16x range of area all read 156-173 tok/s. A
/// GEMM whose cost moved with its tile would have moved more than 11%.
///
/// # AND THE ATTENTION'S COST WAS THIRTY-TWO LANES RECOMPUTING ONE DOT
///
/// `attn/sdpa_paged.wgsl`'s tiled arm ran `@workgroup_size(32, 8)` and every
/// one of the 32 x-lanes of a query row called `dot_page` -- a full 128-term
/// inner product at `d_128` -- for the same (row, key), to own two output
/// channels. 128 multiply-adds recomputed 32 times against the 2 the
/// accumulator costs. The decode arm had already fixed the same waste with a
/// barrier tree; the tiled arm was left because it "cannot barrier", and it
/// does not need to: shrinking the x extent and giving each lane more channel
/// pairs cuts the redundancy with no synchronisation at all.
///
/// `PIE_TX` is that extent. Its sweep lives in the shader; at `d_128` the
/// optimum is **2**, and this test then reads:
///
/// | rows | ms | tok/s | ms a row |
/// | --- | --- | --- | --- |
/// | 32 | 43.9 | 729 | 1.372 |
/// | 64 | 73.2 | 875 | 1.143 |
/// | 128 | 136.8 | 936 | 1.068 |
/// | 256 | 263.4 | 972 | 1.029 |
/// | 512 | 570.7 | 897 | 1.115 |
///
/// **169 -> 897 tok/s at 512, 5.3x**, and the shape is the finding rather
/// than the factor: the per-row cost no longer RISES. It falls to 256 rows
/// and turns over by 8% -- the quadratic term that was 75% of a 512-row
/// prefill is now a minority of it, and whatever is next is not this.
///
/// # AND THE TABLE ABOVE IS OF A GEMM THIS BACKEND DOES NOT SHIP
///
/// Every number in this file up to here was taken through
/// `affine_qmm_t_..._bm_32_bn_32`, because `backend_facts()` inherited
/// `synthetic()`'s `qmm_tile` while `engine::driver::backend::wgpu` states
/// `Some((32, 64))`. The fixture already mirrors that stamp on
/// `qmm_fp16_precast`, with a comment explaining that a fixture off the
/// deployment path puts eleven tests on a code path nobody runs; the same
/// argument reaches the tile and nobody carried it across.
///
/// Mirrored, at 512 rows, and the whole curve moves:
///
/// | rows | ms | tok/s | ms a row |
/// | --- | --- | --- | --- |
/// | 32 | 40.0 | 800 | 1.249 |
/// | 64 | 56.5 | 1133 | 0.882 |
/// | 128 | 94.2 | 1359 | 0.736 |
/// | 256 | 179.3 | 1428 | 0.700 |
/// | 512 | 356.7 | **1436** | 0.697 |
///
/// **169 -> 1436 tok/s, 8.5x**, of which the last 1.24x was free: it was
/// already shipping and the notebook could not see it. The SHAPE is the
/// better news. Under the old tile the per-row cost fell to 256 rows and
/// turned back up by 8%; here it falls to 256 and then does not move --
/// 0.700 against 0.697 -- so the quadratic term that was 75% of a 512-row
/// prefill when this file opened is now too small to read off this curve at
/// all. The remaining cost is linear in rows, which is the shape of the
/// projections rather than of the attention.
///
/// The lesson is not about GEMM tiles. A measurement fixture is only an
/// answer sheet if it answers the question the shipping build asks, and this
/// one had drifted on a field whose neighbour carried a comment about
/// exactly that hazard. Every ratio taken in this file before now has a
/// denominator nobody runs.
///
/// The first version of this measurement claimed 1623 tok/s and was wrong.
/// `kernels-wgpu`'s `tiled_lanes` is a second copy of the shader's `PIE_TX`
/// and had not been moved with it; a `Fire` divides the LANES `apply` is
/// given by the module's real `@workgroup_size`, so the mismatch dispatched a
/// quarter of the query heads and simply did not compute most of the
/// attention. Fast, wrong, and invisible to a timing test. What caught it was
/// `arena`'s workgroup census, which is the reason that number is pinned.
///
/// ## And it is STILL the attention, at a third rather than at 72%
///
/// The layer sweep below says every layer of the 28 costs the same ~30 ms, so
/// there is no hot layer and it never named a kernel. Stepping the same
/// prefix ONE rectangle at a time across a single layer does, and the layer
/// is fifteen launches:
///
/// | rectangle | kernel | +ms |
/// | --- | --- | --- |
/// | 30 | `affine_qmm_t` (gate) | 1.9 |
/// | 31 | `affine_qmm_t` (up) | 1.9 |
/// | 32 | `silu_mul` | 0.3 |
/// | 33 | `affine_qmm_t_residual` (down) | 1.7 |
/// | 35 | `affine_qmm_t` (q) | 1.2 |
/// | 36 | `affine_qmm_t` (k) | 0.7 |
/// | 37 | `affine_qmm_t` (v) | 0.7 |
/// | 39-40 | `neox_mb` x2 | 0.1 |
/// | 42 | `kv_append_paged` | 0.1 |
/// | **43** | **`sdpa_paged_tiled_bfloat16_d_128`** | **10.0** |
/// | 44 | `affine_qmm_t_residual` (o) | 1.4 |
/// | 29, 34, 38, 41 | `rms_single_row` x4 | 0.1-0.2 each |
///
/// One rectangle is a third of a layer and half of everything the sweep can
/// name; the seven projections together are 9.5. So the lane narrowing moved
/// the attention from 72% of the prefill to about a third of it and did not
/// change which kernel is first.
///
/// The differences are of two ~500 ms fires, so a single rectangle's number
/// is a difference of large numbers and single milliseconds are noise. It is
/// read for the one line that stands out by 5x, not as a profile.
///
/// ## Then the dot went word-at-a-time, for another 28%
///
/// `q_at(i)` and `k_at(i)` each load `buf[i >> 1]` and select a half, so
/// `dot_page`'s scalar loop over 128 channels issued 512 loads and fetched
/// every word twice. The same loop over `PIE_PAIRS` words, keeping the two
/// multiply-adds as two separate statements so the f32 rounding is
/// unchanged, issues 128. The value accumulate had the same defect -- two
/// `v_at` calls for one word -- and is one load now.
///
/// | rows | ms | tok/s | ms a row |
/// | --- | --- | --- | --- |
/// | 32 | 38.1 | 841 | 1.189 |
/// | 64 | 60.6 | 1056 | 0.947 |
/// | 128 | 113.5 | 1128 | 0.887 |
/// | 256 | 214.6 | 1193 | 0.838 |
/// | 512 | 445.0 | 1151 | 0.869 |
///
/// **897 -> 1151 tok/s**, and 169 -> 1151 over the two changes together,
/// **6.8x**. Not one answer moved, which is what the same-order,
/// same-rounding restatement was for.
///
/// The lane sweep was re-taken on top of it and the optimum did not move: 1
/// and 2 now read the same (445 ms) where 2 used to win, and 4 is 541.
///
/// ## The 1064 tok/s this file used to be chasing was never measured here
///
/// It was an RTX 4090 figure, and `e0a2f6e20` withdrew the kernel that
/// produced it as wrong. See
/// [`the_tiled_gemms_readout_moves_when_the_last_token_moves`]: the defect it
/// withdrew the GEMM over is gone, the stamp is back on, and the GEMM now
/// agrees with the matvec to about 1% at every length from 31 to 512.
///
/// **Still falling at 512**, so a prefill has not yet reached its per-row cost
/// and is carrying a fixed cost worth roughly 60 ms. It is nonetheless
/// dominated by per-row work where the decode is dominated by fixed cost — the
/// two halves of this backend fail for different reasons, and a fix aimed at
/// one does little for the other.
///
/// (In the debug profile this table read 428.6 / 412.1 / 702.7 / 1332.0 /
/// 2709.3 and was FLAT from 64 rows on at ~5.2 ms a row. Optimisation did not
/// merely scale it — it changed the shape, because what it removed was host
/// work. A curve's shape is a claim about a build too.)
///
/// The ceiling I named here was wrong. This said *"WebGPU's baseline tier has
/// no cooperative-matrix instruction at all"* and treated 815 ms against a
/// tensor-core ~4 ms as partly structural. The baseline half is true and the
/// inference is not: `kernels-wgpu`'s
/// `whether_this_adapter_offers_the_cooperative_matrix_this_tree_calls_absent`
/// asks the adapter in front of this suite and it answers
/// `EXPERIMENTAL_COOPERATIVE_MATRIX: true`, six shapes, including
/// **16x16x16 F16 in, F32 accumulate** — from `wgpu 30`, the version already
/// in this tree's lock file, on `VK_KHR_cooperative_matrix`. A device opens
/// with it too, so this is not an advertised bit that request-time refuses.
/// What stands between this backend and the tensor cores is `device.rs`
/// asking for `ExperimentalFeatures::disabled()`, not the standard — and the
/// price of asking otherwise is signing wgpu's `unsafe` experimental token.
///
/// So 815 ms is 200x off a reachable number rather than an unreachable one,
/// and "structural" was doing work that "unimplemented" should have done.
/// Measured, not merely available: `kernels-wgpu`'s
/// `what_the_cooperative_matrix_is_worth_at_a_projections_shape` runs a
/// `coop_mat16x16` f16->f32 GEMM at this exact projection shape in **0.168 ms
/// against the shipped quantised kernel's 1.412 — 8.4x — with every one of
/// 1,835,008 output elements verified exact**, from a naive tile with no
/// staging and no register blocking.
///
/// It is not the projections, and it is not the tile — and ~30 % of it is not
/// the plan at all: `where_a_prefills_time_goes_across_its_plan` fires the same
/// prompt recording ZERO rectangles and still pays 176 ms of the 815.
///
/// # It is not the projections, and it is not the tile
///
/// `kernels-wgpu`'s isolated sweep does the gate/up projection `[3584, 1024]`
/// at m=512 in 1.382 ms in release, which is 2.7 TFLOP/s. If every GEMM in the
/// model ran at that rate a 512-row prefill would take 224 ms. It takes 815, of
/// which 176 is host work outside the plan — so the projections are roughly a
/// third of what a prefill spends on the GPU and the rest is the other
/// kernels, with no large unexplained remainder. (In debug the same
/// subtraction left ~85 % unaccounted, which was the profile and not the
/// kernels.)
///
/// The tile is the same story measured a second way, and in release there is
/// barely a story: the shipped `(32, 32)` is 1.412 ms against the best tile's
/// 1.382, two percent. End to end, fired through the whole shell interleaved,
/// four tiles land within 7 % of each other
/// (`which_tile_a_512_row_prefill_wants`). The debug run of the kernel sweep
/// showed a 1.34x that neither survived release nor reached the shell.
///
/// Run with `--ignored --nocapture`.
///
/// RETIRED WITH `kernels-wgpu`'s TEST TREE. That name is a record of a
/// measurement now, not a live proof: the crate lost `tests/` and every
/// in-file `mod tests` when the three shader planes moved their numbers to
/// the fire that reads them, and nothing in this workspace re-runs it. What
/// it reported is still why the sentence above says what it says; what is
/// gone is the thing that would notice if it stopped being true.
#[test]
#[ignore = "measurement"]
fn what_a_prefill_costs_at_length() {
    release_only();
    let _lock = gpu();
    let Some(real) = weights() else {
        eprintln!("no checkpoint; skipped");
        return;
    };
    let mut long: Vec<u32> = Vec::new();
    while long.len() + PERIOD.len() <= 510 {
        long.extend_from_slice(&PERIOD);
    }
    while !long.len().is_multiple_of(16) {
        long.push(PERIOD[0]);
    }
    assert_eq!(long.len(), 512);

    // A fresh `who` per fire, so every one is a prefill rather than a
    // continuation, and enough pages for all of them to coexist.
    let mut shell = shelled(real, 8192);
    let mut who = 0u64;
    let mut fire = |shell: &mut Shell, rows: usize| -> f64 {
        who += 1;
        let t = std::time::Instant::now();
        let f = shell
            .step(&[Turn {
                who,
                tokens: long[..rows].to_vec(),
            }])
            .expect("the prompt fires");
        let took = t.elapsed().as_secs_f64() * 1000.0;
        // Read the row out, so the timing cannot exclude the readback that a
        // caller always pays.
        let _ = argmax(f.logits.row(f.readout_of[0]).expect("it read out"));
        took
    };
    // Every length is a whole number of 32-row tiles, so `TokensMultipleOf`
    // admits all of them and the sweep is of SIZE rather than of admission.
    let rows: [usize; 5] = [32, 64, 128, 256, 512];
    let mut best = vec![f64::INFINITY; rows.len()];
    const ROUNDS: usize = 4;
    for round in 0..ROUNDS {
        for (i, &r) in rows.iter().enumerate() {
            let took = fire(&mut shell, r);
            // The first round builds pipelines and grows buffers.
            if round > 0 && took < best[i] {
                best[i] = took;
            }
        }
    }
    println!(
        "\n  a prefill, fastest of {} interleaved rounds:",
        ROUNDS - 1
    );
    for (i, &r) in rows.iter().enumerate() {
        println!(
            "    {r:4} rows {:8.1} ms  {:6.0} tok/s  {:7.3} ms a row",
            best[i],
            r as f64 * 1000.0 / best[i],
            best[i] / r as f64
        );
    }
    let at_512 = best[rows.len() - 1];
    println!(
        "\n  prefill 512: {at_512:.3} ms -> {:.0} tok/s",
        512_000.0 / at_512
    );
}

/// **Where a 512-row prefill's time is, walked one plan prefix at a time.**
///
/// `#[ignore]`, and a measurement rather than an assertion.
///
/// # Why a prefix walk and not a guess
///
/// `what_a_prefill_costs_at_length` establishes that a prefill is linear in
/// its rows at ~5.2 ms each, and arithmetic then rules the obvious suspect
/// out: at the isolated GEMM's 1.52 TFLOP/s the whole 610-GFLOP fire should
/// take 401 ms and it takes 2709, so ~85 % is somewhere the projections are
/// not. Three candidates read equally well from the source — the paged
/// attention, the readout over all 512 rows (`Step::logits` is *"one
/// distribution per row of the fire"*, which is 159 GFLOP of `lm_head`), and
/// host work that scales with the arena.
///
/// `Shell::fire_prefix` settles it without picking one. It records only the
/// first `n` rectangles, so firing the same prompt at a rising `n` gives a
/// CUMULATIVE cost curve over the plan and the jumps name their own
/// rectangles. The answer is not a fire — a truncated plan computes nothing
/// anybody wants — but the timing is the timing.
///
/// # And the plan is not all of where the time is
///
/// A fire recording ZERO rectangles, fastest of two interleaved rounds,
/// `--release`:
///
/// | rows | ms | ms a row |
/// | --- | --- | --- |
/// | 1 | 0.3 | 0.317 |
/// | 16 | 2.0 | 0.124 |
/// | 32 | 3.9 | 0.123 |
/// | 128 | 36.4 | 0.284 |
/// | 256 | 119.2 | 0.466 |
/// | 512 | 249.9 | 0.488 |
///
/// So of an 815 ms prefill, ~250 ms is paid before any rectangle runs, and the
/// walk from 0 to 564 rectangles climbs 174 -> 846, putting all 564 dispatches
/// together at ~600 ms with no single one a hog. **Call it 70/30 between the
/// plan and the host.**
///
/// # The row that answers the decode
///
/// **One row with zero rectangles is 0.3 ms**, against a real decode's 21.6.
/// So 1.4 % of a decode is overhead that is not per-dispatch, and the other
/// 98.6 % is the 452 rectangles — about 47 us each. That is the decode's whole
/// story in one number, and it is why nothing in the kernel tree moves it:
/// `serve::record`'s table splits those 47 us into host recording (`plan_one`,
/// the bind group, the encode) and the GPU's own wait, and both are PER
/// LAUNCH. Against llama.cpp's ~250 fused kernels replayed from a CUDA graph
/// at ~6 us, that is 1.8x the launches at ~8x the price.
///
/// Note what this measurement does NOT separate: recording no rectangles skips
/// the per-dispatch host work too, so the 0.3 ms is the per-FIRE floor and not
/// "the host share". The host share lives inside the 47 us with the GPU.
///
/// # A prefill's budget, and the third of it nobody has attributed
///
/// Assembling the pieces above into one account, because the interesting
/// question is not "what is slow" but "what would fixing each buy". Every line
/// names where it comes from; the last one names that it does not.
///
/// qwen3-0.6B is hidden 1024, 28 layers, intermediate 3072, 16 q-heads and 8
/// kv-heads of 128, vocab 151,936. That is 440.4 M parameters in the layers
/// plus a 155.6 M tied readout = 596.0 M, which is the number this driver
/// stages: 596.0 M at 4.5 bits is 335,372,288 bytes, and 335,372,288 is
/// exactly what it prints. So 512 rows is `2 * 596.0 M * 512` = **610 GFLOP of
/// weight GEMM**, the readout's 159 GFLOP included, plus ~30-60 GFLOP of
/// attention that no weight touches.
///
/// | | ms | where from |
/// | --- | --- | --- |
/// | whole fire | 815 | this test's sibling |
/// | per-fire host | 250 | zero-rectangle row above |
/// | the 564 rectangles | 565 | the difference |
/// | — of which weight GEMM | ~226 | 610 GFLOP at the 2.7 TFLOP/s `qmm_t` measures |
/// | — **of which unattributed** | **~339** | **nothing measured this** |
///
/// The residual is 60 % of the GPU time and it is NOT the projections. Two
/// guesses that do not cover it: attention is 30-60 GFLOP, ~11-22 ms at the
/// same rate; and 564 launches at the decode's 47 us is ~26 ms. So most of
/// that 339 ms is real work in the non-GEMM kernels — norm, rope, softmax,
/// `kv_append`, the elementwise chain — and **that is the next measurement,
/// not the next conclusion.**
///
/// # What this budget says about the tensor cores
///
/// `kernels-wgpu` measures `coop_mat16x16` at 8.4x the shipped quantised
/// `qmm_t`. Put it in the table: 226 ms becomes ~27, the other 339 does not
/// move, the host 250 does not move, and 815 ms becomes ~616. **1.3x**, for
/// the single largest lever in the tree.
///
/// That is the tile result again in a bigger costume — an isolated 8.4x that
/// reaches the shell as 1.3x — and it is worth writing down before anyone
/// spends a month on a cooperative-matrix GEMM. The order the numbers argue
/// for is: attribute the 339 first, narrow the readout second, and reach for
/// the tensor cores when the thing they multiply is the majority of the fire.
///
/// # The debug reading of this table said 80 %, and it was the profile
///
/// The first version of this measurement ran the dev profile, and reported
/// 125.1 / 518.2 / 1074.0 / **2150.5** ms for the same four row counts —
/// linear at ~4.2 ms a row, 80 % of a 2709 ms fire, with the conclusion that
/// a prefill is *"paid before a single rectangle runs"*. Rebuilt `--release`
/// the same measurement is 176.0 ms, **twelve times less**, and the conclusion
/// does not survive: the host share is under a third.
///
/// The reason it moved so far is what it was measuring. `serve::logits` widens
/// `rows * vocab` values element by element through
/// `chunks_exact().map().collect()` — 77.8 million of them at 512 rows — and a
/// scalar loop over 77.8 million elements is exactly the code an optimiser
/// helps most and a debug build punishes most. **The bigger a number's
/// debug-to-release ratio, the more of it was host arithmetic**, which makes
/// that ratio a diagnostic rather than merely an embarrassment.
///
/// # What is still per-row and outside a dispatch
///
/// The readout, materialised for EVERY row. `Step::logits` is documented as
/// *"one distribution per row of the fire"*, so a 512-token prompt whose
/// caller wants one row gets 512, and at this vocabulary that is
/// 512 x 151,936. Asked rather than inferred, and this test prints it:
/// `logits.rows 512, 77,791,232 values, 296.8 MB as f32`. It is paid twice
/// over — the plane sits in the arena, which is a fresh allocation every step
/// (`serve::record` says so), and then `serve::logits` reads all of it back
/// and widens it into that `Vec`.
///
/// `Logits::rows`'s own doc said the opposite — *"one per readout"* — which is
/// how a 296.8 MB return value went unremarked. It now says what it does, and
/// carries the counted blast radius of narrowing it: seventeen call sites
/// reach rows through `readout_of`/`readouts_of` and would survive a remap,
/// five index by fire row and would each need reading.
///
/// 176 ms for 155.6 MB read back and 77.8 M values widened is ~0.9 GB/s, which
/// no longer disagrees with the 0.35 ms `serve::record` measured for a one-row
/// decode the way the debug figure did. So the remaining question is not "why
/// is this twelve times slower than it should be" but the plainer one: a
/// caller that wants one row is charged for 512.
///
/// The comparison that makes it worth doing: llama.cpp's prompt-processing
/// path computes logits for the LAST token of a batch, not all of them, and
/// turns in 40,400 tok/s where this turns in 628.
///
/// Run with `--ignored --nocapture --release`.
#[test]
#[ignore = "measurement"]
fn where_a_prefills_time_goes_across_its_plan() {
    release_only();
    let _lock = gpu();
    let Some(real) = weights() else {
        eprintln!("no checkpoint; skipped");
        return;
    };
    let mut long: Vec<u32> = Vec::new();
    while long.len() + PERIOD.len() <= 510 {
        long.extend_from_slice(&PERIOD);
    }
    while !long.len().is_multiple_of(16) {
        long.push(PERIOD[0]);
    }
    assert_eq!(long.len(), 512);

    let mut shell = shelled(real, 8192);
    let mut who = 0u64;
    let mut fire = |shell: &mut Shell, rows: usize, n: Option<usize>| -> (f64, usize) {
        who += 1;
        shell.fire_prefix(n);
        let t = std::time::Instant::now();
        let f = shell
            .step(&[Turn {
                who,
                tokens: long[..rows].to_vec(),
            }])
            .expect("the prompt fires");
        let took = t.elapsed().as_secs_f64() * 1000.0;
        shell.fire_prefix(None);
        if n.is_none() {
            // WHICH OF TWO DOCS IS RIGHT ABOUT THE READOUT'S WIDTH.
            //
            // `Logits::rows` says "one per readout, so `Frame::readouts` and
            // `Lowered::n_requests` and this are the same number"; `Step::
            // readout_of` says "every row samples -- so a prefill of four
            // tokens produces four distributions". For a one-turn prefill
            // those are 1 and `rows`, and the difference is the whole
            // question of whether a prefill pays its vocabulary once or
            // once per token. Printed rather than reasoned.
            println!(
                "    {rows} rows -> logits.rows {}, {} values, {:.1} MB as f32",
                f.logits.rows,
                f.logits.values.len(),
                (f.logits.values.len() * 4) as f64 / (1024.0 * 1024.0)
            );
        }
        (took, f.fired.dispatches)
    };
    // What the whole plan is, asked rather than assumed.
    println!("\n  what a full fire hands back:");
    let (_, total) = fire(&mut shell, 512, None);
    let _ = fire(&mut shell, 32, None);
    println!("  a 512-row prefill records {total} rectangles");
    // A fire that records NOTHING still pays whatever a fire pays per row:
    // the arena is allocated and zeroed by the row count, and `serve::logits`
    // reads back and widens `rows * vocab` whether or not anybody wants more
    // than one row of it. Sweeping rows at n=0 separates "per fire" from "per
    // row" without attributing either.
    let empty_rows: [usize; 6] = [1, 16, 32, 128, 256, 512];
    let mut empty_best = vec![f64::INFINITY; empty_rows.len()];
    for round in 0..3 {
        for (i, &r) in empty_rows.iter().enumerate() {
            let (took, _) = fire(&mut shell, r, Some(0));
            if round > 0 && took < empty_best[i] {
                empty_best[i] = took;
            }
        }
    }
    println!("  a fire recording ZERO rectangles, fastest of two:");
    for (i, &r) in empty_rows.iter().enumerate() {
        println!(
            "    {r:4} rows {:8.1} ms  {:7.3} ms a row",
            empty_best[i],
            empty_best[i] / r as f64
        );
    }
    let step = total.div_ceil(16);
    let points: Vec<usize> = (0..=total)
        .step_by(step)
        .chain(std::iter::once(total))
        .collect();
    let mut best = vec![f64::INFINITY; points.len()];
    const ROUNDS: usize = 3;
    for round in 0..ROUNDS {
        for (i, &n) in points.iter().enumerate() {
            let (took, _) = fire(&mut shell, 512, Some(n));
            // The first round builds pipelines and grows buffers.
            if round > 0 && took < best[i] {
                best[i] = took;
            }
        }
    }
    println!("  cumulative, fastest of {} rounds:", ROUNDS - 1);
    let mut prev = 0.0;
    for (i, &n) in points.iter().enumerate() {
        let d = best[i] - prev;
        println!(
            "    first {n:4} rectangles {:8.1} ms   (+{d:7.1} over the last {step})",
            best[i]
        );
        prev = best[i];
    }

    // ONE LAYER, ONE RECTANGLE AT A TIME.
    //
    // The sweep above steps by a whole layer and every layer costs the same,
    // which is the answer to "is there a hot layer" (no) and no answer at all
    // to "which kernel". A prefix is a prefix, so the same knob resolves as
    // finely as it is asked to; the only reason it was not is that 452 points
    // at three rounds is a minute of wall clock. One layer is 29 of them.
    //
    // The differences are of TWO fires each ~500 ms apart, so a single
    // rectangle's cost is a difference of large numbers and the noise floor is
    // whole milliseconds. It is read for the ONE rectangle that stands out of
    // its layer, not as a profile.
    let lo = step;
    let hi = (step * 2).min(total);
    let mut fine = vec![f64::INFINITY; hi - lo + 1];
    for round in 0..ROUNDS {
        for (i, n) in (lo..=hi).enumerate() {
            let (took, _) = fire(&mut shell, 512, Some(n));
            if round > 0 && took < fine[i] {
                fine[i] = took;
            }
        }
    }
    println!("  one layer, rectangle by rectangle, fastest of {}:", ROUNDS - 1);
    for (i, n) in (lo..=hi).enumerate() {
        if i == 0 {
            continue;
        }
        println!(
            "    rectangle {n:4} {:8.1} ms   (+{:7.1})",
            fine[i],
            fine[i] - fine[i - 1]
        );
    }
}

/// **Which GEMM tile a 512-row prefill wants, fired through the whole shell.**
///
/// `#[ignore]`, and a measurement rather than an assertion.
///
/// # Why the isolated sweep does not settle this
///
/// `kernels-wgpu`'s `which_tile_the_batched_projection_wants` times one
/// `[3584, 1024]` projection at m=512 and reports `bm=32 bn=64` fastest at
/// 2.475 ms against the shipped `(32, 32)`'s 3.304. But `project::QMM_TILE`'s
/// own doc reports an END-TO-END pair that inverts an isolated ordering —
/// *"a 1024-token prefill takes 2563 ms at (16, 32) and 565 ms at (32, 32)"* —
/// where the isolated numbers put `(16, 32)` AHEAD of `(32, 32)`, 2.886 to
/// 3.304. One projection is not one prefill: a fire is 564 dispatches over
/// many shapes, and `TokensMultipleOf(bm)` moves which prompts are even
/// admitted. So the tile has to be chosen where it is used.
///
/// # Interleaved, because this machine will not hold still
///
/// Every shell is built and staged FIRST and the fires then go round-robin,
/// tile after tile, so a slow minute lands on all of them rather than on
/// whichever went last. Taking all of one tile's fires and then all of the
/// next's is how a 2.6x machine drift gets published as a tile result: this
/// file has already watched one binary's decode median move 49.2, 51.4 and
/// 127.6 ms.
///
/// # The answer, and it is that the question was aimed wrong
///
/// Fastest of five interleaved rounds, `--release`, an RTX 4090:
///
/// | tile | ms | tok/s | vs shipped |
/// | --- | --- | --- | --- |
/// | 32x32 (shipped) | 802.1 | 638 | 1.00x |
/// | 32x64 | 855.0 | 599 | 0.94x |
/// | 16x32 | 798.3 | 641 | 1.00x |
/// | 16x64 | 810.4 | 632 | 0.99x |
///
/// **Within 7 %, and the isolated 1.34x is gone.** The shipped tile is not
/// wrong here — the isolated win was real and simply too small a part of a
/// fire to see. So `qmm_tile` stays at `project::QMM_TILE` and this backend
/// does NOT need its own, which is the negative that stops the next person
/// setting one off a kernel bench.
///
/// Measured twice for a reason. The first run of this sweep was in the dev
/// profile, where a fire's host share is three times what release pays, and a
/// null result under a large constant is the cheapest kind of false negative.
/// Rebuilt `--release` the ordering is unchanged and the spread is if anything
/// tighter, so the conclusion is the tile's rather than the profile's — unlike
/// this file's prefill headline, which the same recheck overturned.
///
/// It also retires the reason this test was written. The doc quoted above was
/// right that a shared constant chosen against a cooperative-matrix build is
/// suspect on a backend with no such instruction; it was wrong that the
/// suspicion would pay.
///
/// # THAT CONCLUSION IS NOW WRONG, and the premise is why
///
/// `engine::driver::backend::wgpu` sets `qmm_tile: Some((32, 64))`, which is
/// exactly the "own tile" this test said the backend did not need. Nothing
/// above was mismeasured -- 32x64 really did read 0.94x -- but the number it
/// measured was a property of the KERNEL, not of the tile.
///
/// `quant/qmm_t.wgsl` had one fast inner loop, guarded on a lane holding
/// exactly two accumulators, and a per-accumulator sweep for everything else.
/// Every tile with `BM * BN4 == 256` took the fast loop and every other tile
/// took the slow one, so the sweep above was comparing (32, 32) on the good
/// path against (32, 64) on the bad one and reading the difference as the
/// tile's. Unrolling the loop at 1, 2, 4 and 8 accumulators moved 32x64 from
/// 0.99 to 2.54 TFLOP/s isolated and took an M4 Pro's pp512 from 905.6 to
/// 1235.8 tok/s end to end.
///
/// So this test is worth RE-RUNNING rather than believing, and its real
/// lesson is the one it did not draw: a tile sweep over a kernel with a
/// shape-conditional fast path measures the condition, not the shape.
///
/// # RE-RUN, AND IT CAUGHT A DIVERGENCE RATHER THAN A TILE
///
/// Nine tiles, fastest of five interleaved rounds, M4 Pro, pp512, after the
/// attention stopped being 72% of a prefill:
///
/// | tile | ms | tok/s |
/// | --- | --- | --- |
/// | 16x16 | 614.5 | 833 |
/// | 16x32 | 694.5 | 737 |
/// | 16x64 | 489.5 | 1046 |
/// | 32x16 | 630.4 | 812 |
/// | 32x32 | 464.3 | 1103 |
/// | **32x64 (shipped)** | **376.3** | **1360** |
/// | 64x16 | 443.6 | 1154 |
/// | 64x32 | 369.8 | 1385 |
/// | 64x64 | 362.4 | 1413 |
///
/// The spread the RTX run called "within 7%" is 1.9x here, and the reason
/// the first run could not see it is the reason a tile sweep has to be run
/// LAST: with the attention taking 72% of the fire, a 2x on the GEMM was a
/// 20% on the total and sat inside the noise this harness has at that scale.
/// A tile sweep measures the tile only once the tile is a majority of what
/// is being timed.
///
/// What it found is not a better tile. `bn = 64` was already shipping; what
/// was not shipping was the FIXTURE, which took `synthetic()`'s (32, 32) and
/// so timed every number in this file on a GEMM the deployment does not
/// build. See `backend_facts()`, which now mirrors the stamp.
///
/// (64, 64) does read a further 3.7%, and it is NOT taken. `bm` is the
/// `TokensMultipleOf` guard: at 64 a prompt whose row count is 32 mod 64
/// has no kernel that will accept it, and this backend would be trading the
/// prompts it can serve for four percent on the ones it can. `bn` is free
/// in that sense, which is why the shipped tile widens on `bn` alone.
///
/// Run with `--ignored --nocapture --release`.
///
/// RETIRED WITH `kernels-wgpu`'s TEST TREE. That name is a record of a
/// measurement now, not a live proof: the crate lost `tests/` and every
/// in-file `mod tests` when the three shader planes moved their numbers to
/// the fire that reads them, and nothing in this workspace re-runs it. What
/// it reported is still why the sentence above says what it says; what is
/// gone is the thing that would notice if it stopped being true.
#[test]
#[ignore = "measurement"]
fn which_tile_a_512_row_prefill_wants() {
    release_only();
    let _lock = gpu();
    let Some(real) = weights() else {
        eprintln!("no checkpoint; skipped");
        return;
    };
    let mut long: Vec<u32> = Vec::new();
    while long.len() + PERIOD.len() <= 510 {
        long.extend_from_slice(&PERIOD);
    }
    while !long.len().is_multiple_of(16) {
        long.push(PERIOD[0]);
    }
    assert_eq!(long.len(), 512);

    // Every `bm` here divides 512, so `TokensMultipleOf` admits this prompt
    // under all of them and the comparison is of speed rather than of who was
    // allowed to run.
    // ALL NINE, not the four this started with. A four-tile sweep that moves
    // 7% invites "the tile is nearly right"; nine tiles spanning a 16x range
    // of area and moving 11% says the cost is not in the tile at all. See
    // `what_a_prefill_costs_at_length` for where it is.
    let tiles: [(u32, u32); 9] = [
        (16, 16),
        (16, 32),
        (16, 64),
        (32, 16),
        (32, 32),
        (32, 64),
        (64, 16),
        (64, 32),
        (64, 64),
    ];
    for (bm, _) in tiles {
        assert_eq!(
            long.len() % bm as usize,
            0,
            "a tile whose guard refuses this prompt would be timed as absent"
        );
    }
    let mut shells: Vec<Shell> = tiles
        .iter()
        .map(|&t| shelled_tuned(real, 2048, false, Some(t)))
        .collect();
    let mut best = vec![f64::INFINITY; tiles.len()];
    let mut who = 0u64;
    const ROUNDS: usize = 6;
    for round in 0..ROUNDS {
        for (i, shell) in shells.iter_mut().enumerate() {
            who += 1;
            let t = std::time::Instant::now();
            let f = shell
                .step(&[Turn {
                    who,
                    tokens: long.clone(),
                }])
                .expect("the prompt fires");
            let took = t.elapsed().as_secs_f64() * 1000.0;
            let _ = argmax(f.logits.row(f.readout_of[0]).expect("it read out"));
            // The first round builds pipelines and grows buffers.
            if round > 0 && took < best[i] {
                best[i] = took;
            }
        }
    }
    println!("\n  a 512-token prefill, fastest of {} rounds:", ROUNDS - 1);
    let shipped = best[0];
    for (i, (bm, bn)) in tiles.iter().enumerate() {
        println!(
            "    bm={bm:<3} bn={bn:<3} {:8.1} ms  {:5.0} tok/s  {:.2}x the shipped tile",
            best[i],
            512_000.0 / best[i],
            shipped / best[i]
        );
    }
}

/// **Every rectangle a 512-row prefill fires, counted by symbol.**
///
/// `where_a_prefills_time_goes_across_its_plan` budgets 815 ms as 250 host +
/// ~226 weight GEMM + **~339 ms of non-GEMM GPU work nobody had named**, which
/// is the largest line in the tree and the only one reached purely by
/// subtraction. This names what is in it — exactly, without timing anything.
///
/// # Naming a rectangle, which `fire_prefix` alone cannot
///
/// `Shell::fire_prefix` says a caller that walks `n` *"finds the rectangle
/// rather than the subsystem"* — but a walk alone finds an INDEX. The names
/// are in the lowering: `Lowered::launches` carries one per rectangle and each
/// indexes `Lowered::kernels`. `lower` is a pure function of the plan and the
/// row shape, so this derives the same lowering the shell will and reads the
/// symbols off it. **No adapter and no checkpoint**, which is why it is a gate
/// test and not a measurement.
///
/// # What a prefill actually fires
///
/// | count | symbol |
/// | --- | --- |
/// | 140 | `affine_qmm_t` |
/// | 113 | `rms_single_row` |
/// | **112** | **`cast_qmm_input_strided_bfloat16_to_float16`** |
/// | 56 | `affine_qmm_t_residual` |
/// | 56 | `neox_mb` |
/// | 28 | `kv_append_paged` |
/// | 28 | `sdpa_paged_tiled` |
/// | 28 | `silu_mul` |
/// | 1 each | `affine_qmv_fast`, `embed_gather_mb_4bit`, `row_gather` |
/// | 564 | total |
///
/// **112 of 564 rectangles — one launch in five — are a CAST**, bf16 to f16
/// with no arithmetic in them at all. Add the 113 norms and 225 of 564 are
/// norm-or-cast. The 196 that are GEMM (`affine_qmm_t` and its residual
/// variant) are 35 %.
///
/// That is a fusion target with a name, and llama.cpp's WebGPU backend ships
/// the shape of the answer: its `rms_norm_mul.wgsl` puts a norm and the
/// multiply after it in one entrypoint. A cast that exists only to feed the
/// next kernel's operand type is the easiest thing in this table to stop
/// launching.
///
/// # And the decode, which is where the 15x is
///
/// | count | symbol |
/// | --- | --- |
/// | 141 | `affine_qmv_fast` |
/// | 113 | `rms_single_row` |
/// | 56 | `affine_qmv_fast_residual` |
/// | 56 | `neox_mb` |
/// | 28 | `kv_append_paged` |
/// | 28 | `sdpa_paged_decode` |
/// | 28 | `silu_mul` |
/// | 1 each | `embed_gather_mb_4bit`, `row_gather` |
/// | 452 | total |
///
/// **No casts at all** — those 112 are the prefill's, because the matvec path
/// takes bf16 where the tiled GEMM wants f16, so a fifth of a prefill's
/// launches exist only to bridge that. And `where_a_prefills_time_goes_across
/// _its_plan` shows a decode is 98.6 % its 452 rectangles at ~47 us each, so
/// this table is the decode's cost almost exactly.
///
/// **169 of 452 — 37 % — are a norm or a rope.** 197 are the matvec. That is
/// what `serve::record`'s *"the way down from here is FEWER LAUNCHES"* is
/// pointing at, and upstream is already there: `Name the merge that survives
/// every constraint: rms into rope` is a commit on this branch. The counts
/// say how much it is worth before anybody measures it.
///
/// # The timing this test does NOT do, and why
///
/// It first tried to price each symbol by walking prefixes one rectangle at a
/// time and differencing. That does not work and the test's own arithmetic
/// said so: the per-symbol estimates summed to **4269 ms against a plan that
/// takes ~565**, 7.5x over. Differencing two ~600 ms fires cannot resolve a
/// ~1 ms rectangle, and flooring the negative steps at zero — which seemed
/// like the careful choice — turns symmetric noise into a systematic credit.
///
/// **Kept as a refutation rather than as a table**, because a plausible-looking
/// per-kernel cost table is worse than none. The right instrument is
/// `Features::TIMESTAMP_QUERY`, which wgpu has and this driver does not yet
/// ask for; the counts above need no instrument at all, which is why they are
/// what this test returns.
#[test]
fn which_kernels_a_prefill_spends_its_gpu_time_in() {
    // NO ADAPTER AND NO CHECKPOINT. `lower` is a pure function of the plan and
    // the row shape, so the rectangle list is available on a build box, which
    // is why this is a gate test rather than a measurement.
    let long_len = 512usize;

    // The same lowering the shell derives, read for its names.
    let facts = facts();
    let plan = llama_like_metal(&facts, &backend_facts(), FireClass::Prefill);
    let mut lowerings = driver_wgpu::lowering::cached::Lowerings::new();
    let shape = driver_wgpu::lowering::cached::Shape {
        prefill: true,
        rows: vec![
            model_compiler::lower::Row {
                multi_token: true,
                ..Default::default()
            };
            long_len
        ],
    };
    let lowered = lowerings
        .get(&plan, shape, model_compiler::lower::Fire::default())
        .expect("the prefill lowers");
    let names: Vec<String> = lowered
        .launches
        .iter()
        .map(|l| lowered.kernels[l.kernel as usize].clone())
        .collect();
    println!(
        "\n  {} rectangles over {} distinct symbols",
        names.len(),
        lowered.kernels.len()
    );
    let mut per_symbol_count: BTreeMap<&str, usize> = BTreeMap::new();
    for nm in &names {
        *per_symbol_count.entry(nm.as_str()).or_default() += 1;
    }

    let mut rows: Vec<(usize, &str)> = per_symbol_count.iter().map(|(nm, c)| (*c, *nm)).collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
    println!("\n  every rectangle a 512-row prefill fires, by symbol:");
    for (count, nm) in &rows {
        println!("    {count:4}  {nm}");
    }
    let total: usize = rows.iter().map(|r| r.0).sum();
    assert_eq!(
        total,
        names.len(),
        "the histogram must account for every rectangle"
    );
    println!("    {total:4}  total");

    // AND THE DECODE, which is the shape the 15x against llama.cpp is measured
    // at. One row, the decode text, everything else the same read.
    let dplan = llama_like_metal(&facts, &backend_facts(), FireClass::Decode);
    let dshape = driver_wgpu::lowering::cached::Shape {
        prefill: false,
        rows: vec![model_compiler::lower::Row::default(); 1],
    };
    let dlowered = lowerings
        .get(&dplan, dshape, model_compiler::lower::Fire::default())
        .expect("the decode lowers");
    let mut dcount: BTreeMap<&str, usize> = BTreeMap::new();
    for l in &dlowered.launches {
        *dcount
            .entry(dlowered.kernels[l.kernel as usize].as_str())
            .or_default() += 1;
    }
    let mut drows: Vec<(usize, &str)> = dcount.iter().map(|(nm, c)| (*c, *nm)).collect();
    drows.sort_by_key(|r| std::cmp::Reverse(r.0));
    println!("\n  and a ONE-ROW decode:");
    for (count, nm) in &drows {
        println!("    {count:4}  {nm}");
    }
    let dtotal: usize = drows.iter().map(|r| r.0).sum();
    assert_eq!(
        dtotal,
        dlowered.launches.len(),
        "the histogram must account for every rectangle"
    );
    println!("    {dtotal:4}  total");

    // AND THE SAME PREFILL AT 64 ROWS, because the lane depends on it.
    // `tests/arena.rs` sweeps its prefills at 64 and stopped reaching
    // `sdpa_paged_tiled_bfloat16_d_128` when upstream gave a batched decode
    // its own attention lane; at 512 the table above still reaches it. Both
    // printed here so the row count is visible as the variable it is.
    let sshape = driver_wgpu::lowering::cached::Shape {
        prefill: true,
        rows: vec![
            model_compiler::lower::Row {
                multi_token: true,
                ..Default::default()
            };
            64
        ],
    };
    let slowered = lowerings
        .get(&plan, sshape, model_compiler::lower::Fire::default())
        .expect("the short prefill lowers");
    let mut scount: BTreeMap<&str, usize> = BTreeMap::new();
    for l in &slowered.launches {
        *scount
            .entry(slowered.kernels[l.kernel as usize].as_str())
            .or_default() += 1;
    }
    println!("\n  a 64-row prefill's attention:");
    for (nm, c) in scount.iter().filter(|(nm, _)| nm.contains("sdpa")) {
        println!("    {c:4}  {nm}");
    }
    println!("    of {} rectangles", slowered.launches.len());
}

/// **A prefill that is longer than the first one this shell saw.**///
/// # Where this came from
///
/// `tests/hybrid_probe.rs` found qwen3.5 going permanently dark after a
/// three-token prefill that followed a two-token one: every later fire — a
/// fresh row, a length that had already answered, a decode — returns a row
/// with a span of zero, with no refusal and nothing in `wgpu`'s error sink.
/// Re-holding every weight does not bring it back.
///
/// **Nothing in that description mentions the gated DeltaNet**, and this file
/// is the place to find out whether it needs to. The suite here fires this
/// shell in dozens of shapes and has never once GROWN a prefill: `prompt()` is
/// a fixed 26 tokens, so the first lowering is always the widest one, and a
/// buffer sized on first use would never be asked for more.
///
/// So the order matters and it is the whole test: **short first, then longer.**
/// Reversed, this passes on a shell with the defect.
#[test]
fn a_prefill_longer_than_the_first_one_is_still_answered() {
    let Some(_held) = gpu() else { return };
    let Some(real) = weights() else { return };
    let mut shell = shelled(real, 64);

    let mut spans = Vec::new();
    for (i, n) in [2usize, 3, 4, 2].into_iter().enumerate() {
        let step = shell
            .step(&[Turn {
                who: 10 + i as u64,
                tokens: PERIOD[..n.min(PERIOD.len())].to_vec(),
            }])
            .unwrap_or_else(|e| panic!("the {n}-token prefill was refused: {e}"));
        let row = step
            .logits
            .row(step.readout_of[0])
            .expect("the turn's own row");
        let span = row.iter().copied().fold(0.0f32, |m, v| m.max(v.abs()));
        println!("  {n} tokens: span {span:.3}, argmax {}", argmax(row));
        spans.push((n, span));
    }

    // A span, not a token: what a dark fire produces is not garbage text, it
    // is a hidden state of zeros through a quantized `lm_head`, which reads
    // out as a quarter of a million tiny constants. Asking for a plausible
    // argmax would pass on that; asking for a distribution does not.
    for (n, span) in &spans {
        assert!(
            *span > 1.0,
            "the {n}-token prefill answered with a span of {span}, which is a \
             row nothing wrote: {spans:?}"
        );
    }
}

