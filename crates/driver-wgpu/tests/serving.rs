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
//! **18 are `linear_attention`**; `kernels-wgpu`'s `ssm.rs` rows (`gdn_core`,
//! `gdn_core_recurrent`, `gdn_core_recurrent_prefill`) declare axes and no
//! operands, and `geometry.rs` refuses `Rule::RecurrentScan` and the rest of
//! that family as `Ungeometric::Unruled`. `pie model list` says the same thing
//! one level up: *unsupported type: qwen3_5_text*. A GDN model needs kernels
//! this tree does not have, which is a coverage statement and not a defect.
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
use model_compiler::trace::FireClass;

/// One device at a time, for the whole suite.
///
/// `tests/device.rs` states the measurement this stands on: with `cargo test`'s
/// default parallelism, ten `wgpu::Device`s open at once wedges roughly one run
/// in three on the NVIDIA proprietary driver. It matters more here than there —
/// each shell holds 335 MiB of weights, and three racing is a card that is out
/// of memory for a reason no message would explain.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

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
    let facts = facts();
    let text = Text {
        decode: llama_like_metal(&facts, &backend_facts(), FireClass::Decode),
        prefill: llama_like_metal(&facts, &backend_facts(), FireClass::Prefill),
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
        let vocab = step.logits.vocab;
        let at = (a_rows - 1) * vocab;
        let row = &step.logits.values[at..at + vocab];
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
                sampling_indices: vec![prompt.len() as u32 - 1, prompt.len() as u32 * 2 - 1],
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
        assert_eq!(
            both[0].logits.row(*row).expect("a readout row"),
            want.as_slice(),
            "request {which} of a two-request frame answered differently from \
             the same prompt fired alone, so the page CSR was not split at the \
             boundary it names"
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
    assert_eq!(
        original.len(),
        forked.len(),
        "two seats of one fire read rows of different widths"
    );
    let differing = original
        .iter()
        .zip(&forked)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing,
        0,
        "the forked seat disagrees with the seat it was forked from in {differing} \
         of {} channels; argmax {} against {}",
        original.len(),
        argmax(&forked),
        argmax(&original),
    );

    let differing_again = original
        .iter()
        .zip(&grandchild)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing_again,
        0,
        "a seat forked FROM a fork disagrees with the seat both came from in \
         {differing_again} of {} channels",
        original.len()
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
    let empty = row_of(&alone, 0);
    let same = empty
        .iter()
        .zip(&forked)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    assert!(
        same < empty.len(),
        "a seat with NO history answered exactly what the forked seat did, so \
         this test is comparing the fire with itself rather than the cache"
    );
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
    let differing = tree
        .iter()
        .zip(&flat)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing,
        0,
        "a leaf of a two-level fork tree disagrees with a seat that heard the \
         same {} tokens and was never forked, in {differing} of {} channels; \
         argmax {} against {}",
        whole.len(),
        tree.len(),
        argmax(&tree),
        argmax(&flat),
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
    let rooted = row_of(&short, 0);
    let same = rooted
        .iter()
        .zip(&tree)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    assert!(
        same < rooted.len(),
        "a seat holding only the root answered exactly what the leaf did, so \
         this test cannot see what the appends wrote"
    );
}
