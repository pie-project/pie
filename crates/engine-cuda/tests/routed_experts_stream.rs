//! **Routed experts stream, and the logits do not notice** (alto design §7,
//! wave D2).
//!
//! The claim under test is design §7's one sentence about the dynamic demand
//! shape: *residency is a performance promotion, never a correctness
//! condition*. Routing is computed on device, so no host decision can precede
//! a fire and arrange for the experts it will need to be there. The engine's
//! answer is an indirection table — a device-resident `expert_id -> base
//! address` row per bank, whose entries point into a device slab when the
//! expert is resident and at PINNED HOST bytes over UVA when it is not — plus
//! per-expert usage counters the fire path notes its routing in and the host
//! reads between fires.
//!
//! ```text
//! (a) a load whose device budget holds HALF the experts fires, and its
//!     logits are the logits full residency produces
//! (b) a fire that routes to non-resident experts completes with no sync on
//!     the fire path — asserted by construction (below) and witnessed by the
//!     counters showing hits on experts the slab does not hold
//! (c) after repeated fires the promotion moves what was used on-device — the
//!     resident set changes — and the logits do not move with it
//! (d) the refusals: a budget under the dense planes, and a budget under the
//!     pinned tier
//! (e) and the refusal §M-3 added: a streamed load offered no weight cache
//!     directory does not serve, and its sentence names the field to set and
//!     the command to run
//! ```
//!
//! # (b) is a claim about a call graph, and this is where it is stated
//!
//! No test can prove the absence of a synchronize by watching a fire succeed.
//! What makes (b) true is that the fire path grew exactly TWO new operations
//! and neither of them can block:
//!
//! * **one read of `expert_table[expert]`** inside
//!   `moe_matmul_select_gemv_body` — a device load from a device address,
//!   replacing the `weight_base + expert * expert_stride` arithmetic that was
//!   there. When the table pointer is null (full residency) the arithmetic is
//!   what it always was and no load happens at all.
//! * **one `atomicAdd` per routed expert per fire**, from one thread of one
//!   block, into a device counter buffer at a fixed address.
//!
//! Everything else the tier does happens on the HOST between fires
//! (`experts::Tier::promote`, called at the top of `enqueue`) or on the NOTIFY
//! stream behind a settlement event (`experts::Tier::drain`, called in
//! `settle_step`). There is no `cudaLaunchHostFunc` on the compute stream, no
//! `cudaMemcpy` without a stream, no `cudaStreamSynchronize`, no readback the
//! next wave waits on. A miss reads pinned memory over PCIe and the kernel
//! keeps going — which is exactly what article 2 asks for and exactly what
//! `d(a)` measures the *result* of.
//!
//! # (a) boots THREE times now, and the middle one is a prepare
//!
//! §M wave M-3 made the streamed load WARM-ONLY. Under `Intent::Serve` a plan
//! that streams is served out of a prepared serving artifact — `<key>.tiers`
//! under the weight cache directory — or it is REFUSED before the pinned tier
//! is allocated; `Shell::prepare` is the only door in the process that writes
//! one. The fully-resident load is untouched, so the golden below is the load
//! it always was.
//!
//! This file used to hand its streamed boot `//! comment saying the cache was off for a gate and a streamed load formed no
//! key anyway. The first half was a choice; the second was a fact about §K's
//! engine, and M-3 is exactly the wave that deleted it. An unkeyed streamed
//! load is now the loudest refusal in the loader, and it is (e).
//!
//! So (a) prepares into a scratch directory and boots warm out of it, at the
//! SAME plan and the SAME `Boot` document — the artifact's key is a function
//! of the whole document, so two documents would name two files and the warm
//! boot would refuse against the one it just wrote. The cost is one extra
//! landing of a 58 MiB synthetic checkpoint, which is why this claim can be
//! made here and not beside the gpt-oss gates.
//!
//! # The fixture, and why it is not a catalog SKU
//!
//! `Model::a3b_micro` is `qwen35-a3b`'s own text at a size two loads of which
//! fit on one card: 4 layers, 32 experts, hidden 512, vocab 2048, ~58 MiB.
//! `a3b` itself is 64 GiB and this file's central claim — that a HALF-resident
//! load says what a FULLY resident one says — needs both loads on one device.
//! Its checkpoint is written here, from the trace's own params, with
//! deterministic pseudo-random bytes: what is under test is the residency
//! machinery, and a machinery that moves the wrong bytes fails against
//! arbitrary weights exactly as it fails against trained ones.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda --test routed_experts_stream -- --nocapture
//! ```
//!
//! # Gating
//!
//! As `serve_smoke.rs`: skipped at run time when the machine has no device,
//! rather than `#[ignore]`d — an ignored test on the one box that could run it
//! is a test nobody runs. Nothing here needs a checkpoint on disk either: the
//! one it reads it writes itself, as it does the serving artifact §M-3 now
//! requires of a streamed boot — tens of megabytes under `TMPDIR`, removed
//! however the test leaves, which is small enough that this file states no
//! disk condition.


use engine_cuda::experts::{Attachments, Budgets, Plan};
use model_dsl::{Dtype, Platform};
use model_ir::Trace;

// ── the fixture ──────────────────────────────────────────────────────────

/// The reduced routed text, traced for this shell.
fn micro() -> (models::qwen_3::model::Model, Trace) {
    let m = models::qwen_3::model::Model::a3b_micro(Dtype::Bf16, Dtype::Bf16, 1);
    let trace = model_dsl::trace_hybrid("qwen35-a3b-micro", &m, Platform::Cuda);
    (m, trace)
}

/// What the whole table demands on the device, off the trace alone.
fn full_demand(trace: &Trace) -> u64 {
    Plan::of(trace, &Attachments::new(), Budgets::uncapped())
        .expect("a bf16 routed text plans")
        .device_demand()
}

// ── (a) and (c) ──────────────────────────────────────────────────────────

// ── (d) and (e) the refusals ───────────────────────────────────────────────

#[test]
fn a_budget_under_the_planes_that_cannot_move_is_refused_by_name() {
    // **THE FLOOR MOVED AT D2b.** It used to be the DENSE planes — none of
    // them could leave the device, so a budget under them was the end of the
    // conversation. They can leave now (streaming §2's static demand shape),
    // and what is left under any budget is the planes that genuinely cannot:
    // a REGISTERED adapter bank, whose store offset `register_adapter` writes
    // at, plus one expert slot of every routed bank.
    let (_, trace) = micro();
    let why = Plan::of(&trace, &Attachments::new(), Budgets::device(1 << 16))
        .expect_err("64 KiB holds no model");
    let said = why.to_string();
    assert!(
        said.contains("REGISTERED") && said.contains("cannot be moved to another tier"),
        "the refusal names the planes that cannot hold less: {said}"
    );
}

#[test]
fn a_host_budget_under_the_pinned_tier_is_refused_by_name() {
    let (_, trace) = micro();
    let full = full_demand(&trace);
    let plan = Plan::of(&trace, &Attachments::new(), Budgets::device(full * 3 / 4))
        .expect("three quarters streams");
    assert!(plan.streams());
    let residency = engine::load::Residency {
        device_weight_budget: Some(full * 3 / 4),
        host_weight_budget: Some(plan.host_demand() - 1),
    };
    let why = residency
        .admit(plan.device_demand(), plan.host_demand())
        .expect_err("a pinned tier one byte short does not admit");
    let said = why.to_string();
    assert!(
        said.contains("host_weight_budget") && said.contains("pinned host"),
        "the refusal names the tier and the field: {said}"
    );
}

#[test]
fn an_uncapped_budget_opens_no_tier_at_all() {
    let (_, trace) = micro();
    let plan = Plan::of(&trace, &Attachments::new(), Budgets::uncapped()).expect("uncapped plans");
    assert!(!plan.streams());
    assert_eq!(
        plan.host_demand(),
        0,
        "a fully-resident load pins nothing — dev's `place_all` allocates no host tier"
    );
}

