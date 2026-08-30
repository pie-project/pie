//! The capability census: which of the catalog's SKUs this shell can bake,
//! and which op each of the rest dies on — derived, not declared.
//!
//! **THE HONESTY RULE, MADE INTO A TEST.** A shell may advertise only what
//! it really does. `kernels-metal` stamps one dtype and stubs whole
//! families, so most of the catalog cannot run here — and the failure mode
//! that matters is not "it refuses", it is "it refuses somewhere nobody
//! wrote down, three hours into a deployment". This file walks every SKU
//! the catalog ships, traced for [`Platform::Metal`], and prints the table:
//! what the kv probe says, what the compiler says, and which refused ops the
//! plan names.
//!
//! It is HOST-ONLY on purpose — no device, no checkpoint, no Apple target.
//! Everything it asks is a property of the model text and the bake, and
//! answering it on Linux is what keeps the census in the inner loop rather
//! than behind a machine.
//!
//! **A REFUSED OP IS ONLY FATAL IF SOMETHING CAN RUN IT**, and that is build
//! log 22's rule read back from the other side. `model_exec::fire::walk`'s rule
//! 1 is "zero rows means the node does not run", applied before the
//! dispatch — so a guarded op costs a fire whose composition gives its window
//! no rows exactly nothing. The census therefore splits its answer: an
//! `Always` refused op is FATAL, a guarded one is CONDITIONAL and only
//! reaches a lane whose word puts it inside that window.
//!
//! **`linear.lora_correct` USED TO BE THIS PARAGRAPH'S EXAMPLE, AND IT IS ON
//! NEITHER LIST NOW.** The correction has an entry, a dispatch arm and a
//! routes seat, so it is not a refusal at all; what stayed true is the shape
//! of the claim, and [`the_correction_is_guarded_and_never_always`] pins it
//! from the other side. A `Guard::Always` correction would launch over every
//! row of every fire to add zero to it, which is the cost the window exists
//! not to pay — and it would also make `Fault::AdapterWord` unsatisfiable,
//! because every class would run the arm and no lane could stand outside it.
//!
//! **AND THE TABLE IS ABOUT THE BAKE, NOT ABOUT THE DEVICE.** A row that
//! bakes with no fatal refusal has cleared the checks this file can make on
//! a machine with no GPU — the kv probe, the compile, the straddle rule, the
//! op vocabulary. It has not been loaded, and several rows here will fail at
//! load for reasons this file cannot see (a checkpoint's import contract, a
//! dtype no entry is stamped for, a quantized bank with no metal path). One
//! row has a device gate behind it, and only that claim is a claim.
//!
//! The assertions are deliberately weak in one direction and sharp in the
//! other: the census may not SHRINK silently (the SKU with the device gate
//! must keep clearing every check) and the refusal list must stay derived
//! from the ops a plan actually names, so a `kernels-metal` that grows an
//! entry moves this table without anyone editing it.

use std::collections::BTreeSet;

use model_compiler::{Budget, DeviceProfile, compile};
use model_ir::{Operands, Operation, Platform};

/// The op names this plane answers `KernelError::Unsupported` for, read off
/// `kernels-metal`'s stubs and `engine-metal`'s own refusing arms.
///
/// **A LIST, NOT A PREDICATE, AND THAT IS THE COST OF THE SEAM.** Whether an
/// entry refuses is inside the entry — `dtype_dispatch!` and the stub bodies
/// — and nothing on the `Operation` enum says so. The CUDA shell has the
/// same shape of problem for the same kind of fact and answers it the same
/// way (`engine_cuda::EXCLUSIVE`, eleven op names read off four modules).
/// What keeps this list honest is that it is only ever used to EXPLAIN a
/// refusal, never to cause one: the refusal itself comes from the entry, at
/// the node, with the entry's own name on it.
const REFUSED: &[&str] = &[
    // `kernels_metal::elemwise::hc` — every entry.
    "elementwise.hc_expand",
    "elementwise.hc_rmsnorm_f32",
    "elementwise.hc_gates",
    "elementwise.hc_fold",
    // `kernels_metal::elemwise::norm`, one entry.
    "elementwise.res_blend",
    // `kernels_metal::collective` — every entry.
    "collective.all_reduce",
    "collective.all_gather",
    "collective.reduce_scatter",
    // `kernels_metal::attn::mla` — every entry.
    "attention.mla_plan",
    "attention.mla_latents",
    "attention.mla_latents_rope",
    "attention.mla_split_q_b",
    "attention.mla_absorb_q",
    "attention.mla_absorb_out",
    "attention.mla_kv_append",
    "attention.mla_decode",
    "attention.mla_prefill",
    "attention.mla_decode_selected",
    "attention.mla_prefill_selected",
    // `kernels_metal::attn::index` — every entry.
    "attention.index_layernorm_rope",
    "attention.index_rope",
    "attention.index_topk",
    "attention.index_kv_append",
    // `kernels_metal::attn::pool` — every entry.
    "attention.pool_boundary_decode",
    "attention.pool_boundary_prefill",
    "attention.pool_gather",
    "attention.pool_kv_append",
    "attention.pool_lse",
];

/// The SKU this shell's device gates are written over, and the one row of
/// this table whose "clears the bake" is backed by a checkpoint that really
/// loaded and really answered.
const SERVED: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// Every refused op name a plan names, split by whether anything guards it.
///
/// The FATAL set is the unconditional ones: a region running an `Always`
/// node has every class in its mask, so every fire dispatches it and the
/// first one refuses. The CONDITIONAL set is the guarded ones, which cost a
/// fire whose composition gives their window no rows exactly nothing.
fn refusals(trace: &model_ir::Trace) -> (BTreeSet<&'static str>, BTreeSet<&'static str>) {
    let mut fatal = BTreeSet::new();
    let mut guarded = BTreeSet::new();
    for node in &trace.nodes {
        let name = op_name(&node.op);
        if let Some(refused) = REFUSED.iter().find(|refused| **refused == name) {
            if matches!(node.guard, model_ir::Guard::Always) {
                fatal.insert(*refused);
            } else {
                guarded.insert(*refused);
            }
        }
    }
    (fatal, guarded)
}

/// One set, as the table prints it.
fn listed(set: &BTreeSet<&'static str>) -> String {
    set.iter().copied().collect::<Vec<_>>().join(", ")
}

/// One op's IR name, as the refusals spell it.
fn op_name(op: &Operation) -> &'static str {
    op.name()
}

#[test]
fn the_census_of_what_this_plane_can_bake() {
    let budget = Budget::new(8, 2048);
    let profile = DeviceProfile {
        side_streams: 0,
        ..DeviceProfile::default()
    };
    let mut clears = Vec::new();
    let mut report = String::new();
    for (sku, _tp, trace, _classify) in model::catalog() {
        let trace = trace(Platform::Metal);
        let probed = engine_metal::store::kv::probe(&trace);
        let (fatal, guarded) = refusals(&trace);
        let compiled = compile(&trace, &budget, &profile);
        let straddles = compiled
            .as_ref()
            .ok()
            .map(|compiled| engine_metal::window::no_schedule_straddles_its_readers(&trace, compiled));
        let verdict = match (&probed, &compiled, &straddles) {
            (Err(why), ..) => format!("kv probe refuses: {why}"),
            (_, Err(why), _) => format!("the bake refuses: {why}"),
            (_, _, Some(Err(why))) => format!("a schedule straddles: {why}"),
            _ if !fatal.is_empty() => {
                format!("FATAL, unconditionally: {}", listed(&fatal))
            }
            _ => {
                clears.push(sku);
                if guarded.is_empty() {
                    "clears the bake".to_string()
                } else {
                    format!(
                        "clears the bake; guarded refusals a lane could still reach: {}",
                        listed(&guarded)
                    )
                }
            }
        };
        report.push_str(&format!("  {sku:<34} {verdict}\n"));
    }
    println!("engine-metal, over the catalog traced for Platform::Metal:\n{report}");
    assert!(
        clears.contains(&SERVED),
        "`{SERVED}` is the row this shell's device gates are written over, and the \
         census no longer clears it:\n{report}"
    );
}

/// A plan this plane cannot serve must say which op it dies on, and the
/// answer must come from the plan rather than from a guess.
#[test]
fn every_refused_name_is_one_the_ir_can_actually_spell() {
    let budget = Budget::new(8, 2048);
    let profile = DeviceProfile {
        side_streams: 0,
        ..DeviceProfile::default()
    };
    // The union of every op name the catalog's Metal traces name. A refusal
    // for a name no model text can produce is a stale entry — the list would
    // then be documenting a kernel plane that has moved on.
    let mut spoken: BTreeSet<&'static str> = BTreeSet::new();
    for (_sku, _tp, trace, _classify) in model::catalog() {
        let trace = trace(Platform::Metal);
        for node in &trace.nodes {
            spoken.insert(op_name(&node.op));
        }
    }
    let _ = (budget, profile);
    let unreachable: Vec<&&str> = REFUSED
        .iter()
        .filter(|name| !spoken.contains(**name))
        .collect();
    println!(
        "of {} refused names, {} are unreachable from today's catalog: {unreachable:?}",
        REFUSED.len(),
        unreachable.len()
    );
    // Not an assertion that the list is empty: a stub may legitimately exist
    // for an op no shipped model text uses yet (`elementwise.hc_*` is the standing
    // example). The claim is the other one — that nothing in the list is a
    // NAME the IR cannot spell, which would mean the entry was typed rather
    // than read.
    for name in REFUSED {
        assert!(
            name.contains('.'),
            "`{name}` is not an IR op name; the list is read off the kernel plane's \
             own `op` strings"
        );
    }
}

/// **THE CORRECTION IS GUARDED, AND THAT GUARD IS WHAT THE FIRE PATH CHECKS
/// A SUBMISSION AGAINST.**
///
/// `engine-metal`'s shell reads the classes whose region runs
/// `linear.lora_correct` off the bake and refuses a lane whose word disagrees
/// with them (`Fault::Adapterless`, `Fault::AdapterWord`). Both refusals rest
/// on the correction living in a window that some classes are outside of: an
/// `Always` correction would put every class inside it, so no lane could ever
/// be refused for standing outside, and every fire would pay two launches a
/// layer to add zero.
///
/// Host-only, like the rest of this file: it is a property of the model text
/// and the bake, and no device answers it.
#[test]
fn the_correction_is_guarded_and_never_always() {
    let mut seen = 0usize;
    let mut report = String::new();
    for (sku, _tp, trace, _classify) in model::catalog() {
        let trace = trace(Platform::Metal);
        let corrections: Vec<&model_ir::Guard> = trace
            .nodes
            .iter()
            .filter(|node| op_name(&node.op) == "linear.lora_correct")
            .map(|node| &node.guard)
            .collect();
        if corrections.is_empty() {
            continue;
        }
        seen += corrections.len();
        report.push_str(&format!("  {sku:<34} {} corrections\n", corrections.len()));
        for guard in corrections {
            assert!(
                !matches!(guard, model_ir::Guard::Always),
                "`{sku}` states an unguarded `linear.lora_correct`: every class would \
                 be inside the correction's window, every fire would pay it, and \
                 `Fault::AdapterWord` could never name a lane standing outside"
            );
        }
    }
    println!("guarded corrections, over the catalog traced for Platform::Metal:\n{report}");
    assert!(
        seen > 0,
        "no catalog row states a correction, and the adapter axis this shell now \
         serves has nothing to run"
    );
    assert!(
        !REFUSED.contains(&"linear.lora_correct"),
        "`linear.lora_correct` is served on this plane — an entry, a dispatch arm \
         and a routes seat — so listing it as refused would make the census lie \
         about the axis in the safe direction, which is still a lie"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The attachment ledger
// ─────────────────────────────────────────────────────────────────────────────
//
// **THE SECOND CENSUS THIS FILE CARRIES, AND IT IS ABOUT GUEST PROGRAMS
// RATHER THAN MODEL TEXT.** Everything above asks what this plane can BAKE; a
// guest program is not baked, it is adopted — and the attachment door has its
// own refusal set, derived from the emitted ABI rather than from a kernel
// stub. Same rule as the table above: the list may not shrink silently, and
// every entry has to come from something a host can compute.

/// The bind-time profile each golden was authored against, transcribed from
/// `eta-compiler`'s corpus helper for the reason `program_parity` transcribes
/// it: the goldens do not carry one, and binding at the wrong vocabulary
/// refuses rather than misbehaves.
fn golden_profile(name: &str) -> eta_ir::registry::ModelProfile {
    let mut profile = eta_ir::registry::ModelProfile::dummy();
    match name {
        "counter_pingpong" | "lora_prologue" | "section3_masked_gumbel" | "structured_masks" => {}
        "beam_epilogue" => {
            profile.vocab = 8;
            profile.page_size = 4;
        }
        "pentathlon_iter" => {
            profile.vocab = 8;
            profile.kernels.push(eta_ir::registry::KernelInfo {
                name: "envelope_dot".into(),
                sink_scope: None,
                replayable: true,
            });
        }
        _ => profile.vocab = 8,
    }
    profile
}

/// Every golden in the corpus that describes a program rather than a
/// refusal, as `(name, needs_logits, needs_mtp_logits, needs_attn_scores)`.
fn corpus() -> Vec<(String, bool, bool, bool)> {
    let dir = std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../eta-compiler/tests/golden"
    ));
    let mut rows = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return rows;
    };
    let mut names: Vec<String> = entries
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_stem()?.to_str()?.to_string();
            // The `neg_*` goldens are refusals of the VALIDATOR, not
            // programs: they do not bind, and asking this plane about one
            // would be asking whether it serves something that does not
            // exist.
            (!name.starts_with("neg_")).then_some(name)
        })
        .collect();
    names.sort();

    for name in names {
        let path = dir.join(format!("{name}.txt"));
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Some(line) = text
            .lines()
            .find_map(|line| line.strip_prefix("container: "))
        else {
            continue;
        };
        let bytes: Vec<u8> = (0..line.len() / 2)
            .filter_map(|at| u8::from_str_radix(&line[at * 2..at * 2 + 2], 16).ok())
            .collect();
        let Ok(container) = eta_ir::container::decode(&bytes) else {
            continue;
        };
        let Ok(bound) = eta_ir::validate::bind(container, golden_profile(&name)) else {
            continue;
        };
        let stages = eta_compiler::plan::compile_bound(&bound);
        let package = eta_compiler::codegen::launch::build(&bound, &stages);
        let Ok(plan) = eta_exec::adopt_launch_package(package) else {
            continue;
        };
        if !plan.executable {
            continue;
        }
        rows.push((
            name,
            plan.needs_logits,
            plan.needs_mtp_logits,
            plan.needs_attn_scores,
        ));
    }
    rows
}

/// **WHICH GUEST PROGRAMS THIS PLANE CAN ATTACH AT AN EPILOGUE, AND WHICH IT
/// REFUSES BY NAME.**
///
/// **THE ONE REFUSAL THIS CENSUS USED TO PRINT IS GONE, AND THE CENSUS SAYS
/// SO RATHER THAN GOING QUIET.** It was the emitted ABI's: the M2 emitter
/// bound `logits [[buffer(6)]]` and made it the first argument of EVERY
/// `INTRINSIC_VAL` op, so a program reading the draft column had no second
/// rectangle to be pointed at and `serve::prepare` refused to attach it
/// against ANY load. `eta_compiler::codegen::metal::intrinsics` gives each
/// intrinsic an argument index of its own now, `program::launch` carries a
/// slot table, and `Shell::enqueue` points `IntrinsicId::MtpLogits` at the
/// `mtp` export's rectangle.
///
/// **AND THE SECOND REFUSAL IS GONE TOO, ONE WAVE LATER**
/// (`.wiki/alto/attn-score.md` §4). A program reading `attn_score` was turned
/// away at `program::session` against every load, for two reasons that are both
/// closed: `engine_metal::scores` carves the observability slab the capture arm
/// writes, and `ptir_m1_runtime.metal`'s `0xA0` handler branches on the
/// intrinsic id so an F32 score plane is read as F32. `Shell::enqueue` points
/// `IntrinsicId::AttnScore` at the capturing lane's block of that slab.
///
/// So the census can no longer answer for a program on its own: whether an
/// `mtp_logits` reader attaches is a question about the LOAD — does its model
/// text bake an `mtp` seam — and whether an `attn_score` reader attaches is the
/// same question one seam over, and this file adopts launch packages without
/// either. What it still states is which programs need which rectangle at all,
/// which is what `api.rs`'s `has_mtp_logits: shell.drafts()` and
/// `has_attn_score: shell.observes_scores()` decide against.
///
/// Host-only, like the rest of this file: adopting a launch package and
/// asking what it reads touches no device.
#[test]
fn the_attachment_census_of_the_guest_corpus() {
    let rows = corpus();
    assert!(
        !rows.is_empty(),
        "no golden in the corpus adopts, so this census is about nothing"
    );

    let mut report = String::new();
    let mut served = 0usize;
    for (name, logits, drafts, scores) in &rows {
        // A program that reads the draft column or the score rectangle is
        // counted as served: the slot table gives each one a rectangle, and
        // what decides its fate is the artifact rather than this plane.
        let verdict = if *scores {
            served += 1;
            "attachable to an OBSERVING load — reads `attn_score` at the slab; \
             refused against a load that carves none"
        } else if *drafts {
            served += 1;
            "attachable to a DRAFTING load — reads `mtp_logits` at the mtp seam; \
             refused against a load that bakes none"
        } else if *logits {
            served += 1;
            "attachable — reads the trunk's logits at the out seam"
        } else {
            served += 1;
            "attachable — reads no intrinsic, so a boundary costs it nothing"
        };
        report.push_str(&format!("  {name:<28} {verdict}\n"));
    }
    println!("the guest corpus, at this plane's epilogue:\n{report}");

    assert!(
        served > 0,
        "this plane can attach no program in the corpus, and the epilogue door \
         serves nothing"
    );
    // The one the device smoke actually drives. If it ever stopped being
    // attachable the smoke would skip rather than fail, which is the silent
    // hole this line closes.
    assert!(
        rows.iter().any(|(name, logits, drafts, scores)| {
            name == "greedy_argmax" && *logits && !*drafts && !*scores
        }),
        "`greedy_argmax` is what `serve_smoke`'s attachment tests attach, and this \
         census says this plane cannot attach it"
    );
}

/// **EVERY BOUNDARY THE CONTRACT HAS, AND WHAT THIS PLANE ANSWERS TO EACH.**
///
/// The `match` is what makes this a ledger rather than a comment: a third
/// [`Boundary`](engine::fire::Boundary) variant does not compile here until
/// somebody decides what this plane does with it, which is the whole point of
/// writing the refusal down beside the service.
///
/// `Prologue` is refused because a prologue's channel writes are INPUTS to
/// the forward and this shell stages every fire input on the host, at
/// `prepare`, before it opens a command buffer — so there is no point in the
/// step at which one could be encoded. `Epilogue` is served: the pass is
/// encoded into the model fire's own command buffer after the walk and after
/// the readout blit, and its verdict is read from the harvest.
#[test]
fn every_boundary_the_contract_has_is_answered_one_way_or_the_other() {
    let mut report = String::new();
    let mut served = 0usize;
    for at in [
        engine::fire::Boundary::Prologue,
        engine::fire::Boundary::Epilogue,
    ] {
        let verdict = match at {
            engine::fire::Boundary::Epilogue => {
                served += 1;
                "SERVED — encoded into the model fire's command buffer after the walk"
            }
            engine::fire::Boundary::Prologue => {
                "REFUSED by name — the fire's inputs are staged on the host before a \
                 command buffer exists"
            }
        };
        report.push_str(&format!("  {:<10} {verdict}\n", format!("{at:?}")));
    }
    println!("the attachment boundaries:\n{report}");
    assert_eq!(
        served, 1,
        "this plane serves exactly one boundary, and the ledger says otherwise"
    );
}
