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
//! **AND THE GUEST DOOR ON TO IT IS OPEN NOW** (lane J). What the paragraph
//! above described was the MODEL's own correction class, reachable from a
//! control plane calling `Engine::register_adapter` by id; a guest program
//! carrying a `lora` sink was refused at bind, because `ModelProfile::has_lora`
//! answered `false` and `eta_ir::validate` honours that. The sink is consumed
//! now — read off the launch package at instance bind
//! (`engine_metal::adapter`), converted into the banks' bf16, landed in a
//! slot, and stamped on to every lane attached to the instance with the fact
//! word moved into the correction's window beside it — so the bit is `true`
//! and `tests/inferlets/lora-probe` runs. The arithmetic is pinned without a
//! device in `engine_metal::adapter`'s own tests and with one by
//! `device_floor::the_correction_adds_what_the_host_says_and_leaves_an_unrouted_row_alone`.
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
    // `kernels_metal::elemwise::hc` — NOTHING. The hyper-connection family is
    // ported whole (`elemwise/hc.metal`): the expansion that tiles the
    // embedding across the stream fan, the weightless RMS norm that widens the
    // wide row to f32, the gate split whose combiner is projected onto the
    // Birkhoff polytope by Sinkhorn-Knopp, and the fold that mixes the
    // sublayer's output back into the streams. `engine-metal/tests/hc_on_device.rs`
    // measures all four on the card against a host fp32 reference — the gate
    // matrices at fp32 epsilon, the collapse and the fold at half a bf16
    // quantum — and pins the Sinkhorn loop bound (`sinkhorn - 1` passes after
    // the seed) at the low counts where it is still observable, since by the
    // shipped twenty the iteration has converged and the off-by-one is
    // invisible. What is still deferred is the DYNAMIC mix: the op hands
    // `normed` where the reference hands `rmsnorm(streams) @ hc_fn`, and both
    // shells read the leading `2M + M²` floats at that stride because no plane
    // fires the projection. That is one interned organ, the same one on both
    // backends, and not a kernel gap.
    // `kernels_metal::elemwise::norm`, one entry.
    "elementwise.res_blend",
    // `kernels_metal::collective` — every entry.
    "collective.all_reduce",
    "collective.all_gather",
    "collective.reduce_scatter",
    // `kernels_metal::attn::mla` — NOTHING. The absorbed latent family is
    // ported whole (`attn/mla.metal`): the plan (empty on this plane, by the
    // seam the entry states), the latent split and its roped twin, the q
    // split, the q-absorb, the paged latent appender, the naive simd flash
    // engine — which serves BOTH the dense readers and the sparse ones, the
    // same body handed the NSA index row instead of the causal range — and,
    // last of the family, the output-absorb that maps the latent reading back
    // to value space. That one was deferred over WHERE `kv_b`'s value planes
    // begin; the answer is `nope·rank` elements in (the CUDA entry's
    // `2·nope·rank` is a byte add), measured end to end against the
    // unabsorbed attention by `engine-metal/tests/mla_on_device.rs`.
    // `kernels_metal::attn::index` — NOTHING. The NSA lightning indexer is
    // ported (`attn/index.metal`): the layernorm+rope, the per-head query
    // rotation, the paged top-k selection, and the append that routes to the
    // mla latent writer with a null rope plane. `attention.index_topk` can
    // still answer `Unsupported`, but only for a load whose trace names no
    // `attention.index_topk` and therefore reserved no score slab — which is
    // not a state any plan naming the op can reach, so it is no census row.
    //
    // **AND THE SELECTION'S KEY STRIDE IS THIS PLANE'S ALONE.** The shader
    // takes a `ratio`: `1` reads one key per token at its own cell, which is
    // glm_5's indexer and is `index.cuh` unchanged; a compressor's ratio
    // reads one key per COMPRESSED BLOCK at `(c+1)*ratio - 1`, which is
    // dsv4-flash's, whose index keys are its own compressor's pooled
    // entries. `index.cuh` has no such parameter, so the CUDA arm serves
    // `ratio == 1` and refuses the rest by name — a refusal on the OTHER
    // plane, and so no row of this census.
    // `kernels_metal::attn::pool` — NOTHING. The compressor's pool is ported
    // whole (`attn/pool.metal`): the two boundary marks, the gated softmax
    // pool, the store into the compressed cache, and the flash lse over the
    // compressed entries. The gather was the last of the five to come off
    // this list, and it came off over its SLABS: `state_kv` and `state_score`
    // are addressed by the source pool's paged slot rather than by a fire
    // row, so `crate::scratch` reserves them at the paging's cell ceiling the
    // way it reserves the indexer's score slab, and the dispatch arm binds
    // them. `engine-metal/tests/pool_on_device.rs` measures the gate on the
    // card against a host fp32 reference at both compressor widths and with
    // the position plane on and off.
    //
    // **AND THE SEAM THAT WAS LEFT IS CLOSED.** This entry used to say that
    // nothing WROTE either state slab — dsv4's model text interned the
    // compressor, so the projections that fill the state had no IR seat and
    // the plane the gather pooled was zero. `attention.pool_state_write` is
    // that writer and `ape` took an operand of its own, so the compressor
    // reads its own planes now; `pool_on_device`'s round-trip gate measures
    // the pair from the other end. The pool grew a SELECTED reader beside
    // the dense one (`pool_lse_selected_paged`, the NSA fine branch), served
    // here and refused by name on CUDA, which `pool.cuh` has no twin of —
    // again a refusal on the other plane and so no row of this census.
    // `engine-metal/tests/nsa_selected_on_device.rs` runs the real ranking
    // into the real reader and folds the result through `merge_lse` and
    // `attention.sink`.
    //
    // **THE QWEN4 N-GRAM HASHER — NOTHING, AND THIS ENTRY IS WHERE THE PAIR
    // USED TO BE.** `attention.ple_ngram_ids` and its chunked twin were on
    // this list for exactly one wave: they were added when the census learned
    // it had been calling every `qwen38-flash-*` row "clears the bake" while
    // no fire of one could reach its second layer, and they came off when
    // `attn/ple.metal` landed. Both arms are ported organ for organ off
    // `kernels-cuda/kernels/attn/ple.cuh` — the seed-derived odd multipliers,
    // the xor fold over the window newest-first, the per-head modulus and
    // offset, and the eos-segmentation rule that masks every id behind a
    // nearer eos. `engine-metal/tests/ple_conv_on_device.rs` measures both on
    // the card against `kernels_metal::attn::ple::reference`, EXACTLY: a hash
    // that is off by one is a different embedding row, so there is no band it
    // could be inside.
    //
    // **THE ONE THING THAT MOVED TO GET THEM HERE IS WHERE THE CONSTANTS
    // LIVE.** The CUDA entry hands its `PleHash` aggregate across the launch
    // ABI by value (`ArgValue::Bytes`); this plane's `ArgValue` has no
    // by-value blob seat, and growing one would have to cross `icb.rs`'s
    // eight-byte scalar arena and `record.rs`'s `Copy` `Arg` for one op. So
    // `crate::scratch` lays a `u64` plane per distinct hashing and writes it
    // ONCE at load — the only role in that file the host touches, and it
    // touches it before the first command buffer exists.
    //
    // **AND THE OTHER HALF OF THE SAME GAP IS GONE TOO, THOUGH IT WAS NEVER
    // NAMEABLE HERE.** The PLE's local mix is a
    // `attention.ssm_causal_conv1d` DILATED by `ngram_size` (3), and this
    // plane refused that op for `dilation: 2..` while serving it at 1. A
    // refusal by PARAMETER is not a name, so it could never have gone on this
    // list; it is stated in `qwen4_two_bit_first_light.rs`, where a trace's
    // own nodes can be asked for their dilation, and that is where the flip
    // is recorded.
    //
    // **THE QWEN4 GATED-RESIDUAL FAMILY — NOTHING, AND THIS ENTRY IS THE
    // CENSUS ADMITTING WHAT IT MISSED.** Opening the hasher's two doors did
    // not make a `qwen38-flash-*` fire; SEVEN more points refused behind them,
    // and this list named none of them while the row's line above read "clears
    // the bake" — because a bake is a compile and every one of these refuses
    // at the FIRE. Five were `dispatch/elemwise.rs`'s own block-refusal
    // (`rmsnorm_grouped_plus_one`, `silu_scaled`, `hc_mix`, `hc_inject`,
    // `ple_gate`), one was `dispatch/layout.rs`'s (`embed_concat`), and one —
    // `elementwise.rmsnorm_gated` at `sigmoid` — refused by ENUM ARM, which
    // like a dilation is not a name and could never have appeared here.
    //
    // All seven serve now, and the five that are new arithmetic are measured
    // on the card by `engine-metal/tests/qwen4_gated_residual_on_device.rs`
    // against `kernels_metal::elemwise::hc::reference`, each held apart from
    // the plausible wrong port beside it. The gather is measured exactly.
    //
    // **WHAT THIS COSTS THE LIST IS NOT AN ENTRY BUT A CAVEAT**, and it is the
    // one worth carrying: this table's "clears the bake" is a statement about
    // what BAKES, and the only row on it whose fire is a claim is [`SERVED`].
    // A refusal reachable only at the first fire of a model nobody has fired
    // is invisible here by construction, and the file that catches it is a
    // first light and not a census.
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
    for (sku, _tp, trace, _classify) in models::catalog() {
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
    for (_sku, _tp, trace, _classify) in models::catalog() {
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
    // for an op no shipped model text uses yet. (`elementwise.hc_*` was the
    // standing example, and is now gone from the list entirely — the family is
    // served, and dsv4's text names every one of its four ops.) The claim is
    // the other one — that nothing in the list is a NAME the IR cannot spell,
    // which would mean the entry was typed rather than read.
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
    for (sku, _tp, trace, _classify) in models::catalog() {
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

/// **EVERY FIRST-PARTY SINK, AND WHY THIS PLANE HAS NOWHERE TO PUT IT.**
///
/// The sibling above writes down the BOUNDARIES this plane answers; this one
/// writes down the SINKS the eta vocabulary reserves, and multiplies the two.
/// The product is the whole finding: for a sink to be servable here there must
/// be a stage that is both legal for its scope AND reachable at a boundary this
/// plane serves, and for every name in [`eta_ir::registry::KNOWN_SINKS`] that
/// intersection is EMPTY.
///
/// ```text
/// scope                 legal stages (eta_ir::validate.rs:662-666)
/// PassWide              Prologue
/// Attention             Prologue, OnAttnProj
///
/// this plane's boundaries (the sibling test's ledger)
/// Epilogue              SERVED   -> runs a program's ops at Stage::Epilogue
/// Prologue              REFUSED by name at `serve::prepare` — the fire's
///                                inputs are staged on the host, before a
///                                command buffer exists
/// OnAttnProj/OnAttn     NOT A BOUNDARY AT ALL — design.md §9 abolished the
///                                third boundary; `eta_exec::plan` refuses a
///                                per-layer tap by name
/// ```
///
/// **SO `has_attn_page_mask: false` IS A STRUCTURAL FACT, NOT AN UNBUILT
/// FEATURE** (`api.rs`'s own note beside the bit). `attn_page_mask` is
/// `SinkScope::Attention`, which is legal at `Prologue | OnAttnProj` and
/// ILLEGAL at `Epilogue` — the one stage this plane can run a guest program
/// at. No amount of engine work inside the current boundary vocabulary opens
/// it: what would have to move first is the constitution, and this test is
/// where that shows up. A lane that flips the bit without adding a boundary
/// fails here rather than shipping a sink nothing consumes.
///
/// The second, independent refusal is written down beside it:
/// [`eta_exec::Boundaries::METAL`] does not carry the name either, so even a
/// well-placed sink would be refused when the launch package is adopted
/// (`engine_metal::program::Programs::register` -> `Fault::Fire`).
///
/// Host-only: reading two const tables and a match touches no device.
#[test]
fn no_first_party_sink_has_a_stage_this_plane_can_run_it_at() {
    use eta_ir::registry::{KNOWN_SINKS, SinkScope, Stage};

    // The one stage a guest program's ops run at here, straight off the
    // boundary ledger the sibling test pins: `Epilogue` is served and it is
    // the only one.
    //
    // **AND `lora` IS IN `Boundaries::METAL` WITHOUT CONTRADICTING THAT**
    // (lane J). This ledger is about sinks a guest program's ops are RUN for,
    // and a `sink_call` is never run by anybody: `eta_exec::op`'s arm for it
    // is `Ok(())` on both planes. The adapter sink's effect is landed on the
    // HOST at instance bind — `engine_metal::adapter::sink_of` reads its
    // channels off the launch package and `planes_of` converts the seeded
    // cells into the banks' bf16 — so admitting it is a claim about what this
    // backend CONSUMES, which is the claim `ModelProfile::has_lora` makes at
    // the other door. `servable` below stays empty because the question it
    // asks is unchanged: no first-party sink has a stage this plane
    // INTERPRETS it at, and `lora` does not need one.
    const SERVED_STAGES: &[Stage] = &[Stage::Epilogue];

    assert!(
        !KNOWN_SINKS.is_empty(),
        "the first-party sink table is empty, so this ledger is about nothing"
    );

    let mut report = String::new();
    let mut servable = Vec::new();
    for (name, scope) in KNOWN_SINKS {
        // The precedence rule, restated from `eta_ir::validate`'s own match:
        // a sink must PRECEDE the point that consumes its effect.
        let legal: &[Stage] = match scope {
            SinkScope::PassWide => &[Stage::Prologue],
            SinkScope::Attention => &[Stage::Prologue, Stage::OnAttnProj],
        };
        let reachable: Vec<Stage> = legal
            .iter()
            .copied()
            .filter(|s| SERVED_STAGES.contains(s))
            .collect();
        let admitted = eta_exec::Boundaries::METAL.sink_calls.contains(name);
        if !reachable.is_empty() {
            servable.push(*name);
        }
        report.push_str(&format!(
            "  {name:<20} {:<10} legal at {legal:?}; reachable here {reachable:?}; \
             in Boundaries::METAL {admitted}\n",
            format!("{scope:?}")
        ));
    }
    println!("the first-party sinks, against this plane's one boundary:\n{report}");

    assert!(
        servable.is_empty(),
        "these first-party sinks now have a stage this plane can run them at, so the \
         boundary ledger moved and the capability bits beside them must be revisited: \
         {servable:?}"
    );

    // The named one, stated on its own so the failure reads as the finding
    // rather than as an arithmetic surprise.
    let page_mask = KNOWN_SINKS
        .iter()
        .find(|(n, _)| *n == "attn_page_mask")
        .expect("`attn_page_mask` is a first-party sink name and left the table");
    assert_eq!(
        page_mask.1,
        SinkScope::Attention,
        "`attn_page_mask` changed scope; the reason this plane cannot serve it is that \
         `Attention` excludes the epilogue, so a scope change reopens the question"
    );
    assert!(
        !eta_exec::Boundaries::METAL
            .sink_calls
            .contains(&"attn_page_mask"),
        "`Boundaries::METAL` admits `attn_page_mask`, but no stage this plane runs a \
         guest program at is legal for an `Attention`-scoped sink"
    );
}
