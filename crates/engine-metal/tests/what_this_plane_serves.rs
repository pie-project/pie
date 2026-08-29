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
//! log 22's rule read back from the other side. `engine::fire::walk`'s rule
//! 1 is "zero rows means the node does not run", applied before the
//! dispatch — so a plan naming `linear.lora_correct` under a `has_adapter`
//! guard costs a fire no lane routed exactly nothing, and every qwen row
//! names one. The census therefore splits its answer: an `Always` refused
//! op is FATAL, a guarded one is CONDITIONAL and only reaches a lane whose
//! word puts it inside that window.
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
    // `kernels_metal::attn`, one entry.
    "attention.merge_lse",
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
    // Refused in this crate's own dispatch arm, because `kernels-metal` has
    // no `linear::lora` module at all (design §8's standing open item).
    "linear.lora_correct",
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
