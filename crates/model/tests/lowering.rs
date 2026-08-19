//! Lowering, checked against real declarations.
//!
//! Moved out of `model-compiler` with `kernels_table.rs` and for the same
//! reason: these lower an actual family's traced form, and a dev-dependency
//! cycle back onto `model` gives the test a second, incompatible copy of the
//! toolchain's types.

use model::shared::llama_like::forward::facts::LlamaLikeCudaFacts;
use model::shared::llama_like::forward::facts::LlamaLikeFacts;
use model_compiler::lower::*;
use model_ir::kernels::Backend;
use model_ir::trace::FireClass;
use model_ir::trace::{ForwardPlan, OpKind, PeelWindow, ValueId};
use std::ops::Range;

/// A fire whose rows are all plain AND all sampled — the ordinary
/// decode shape, and the one every row-axis test wants.
fn plain(n: usize) -> Vec<Row> {
    sampled(n)
}

fn sampled(n: usize) -> Vec<Row> {
    vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ]
}

/// A prefill-shaped fire: `n` token rows, one of them sampled, so
/// the epilogue gathers.
fn gathered(n: usize) -> Vec<Row> {
    let mut rows = vec![Row::default(); n];
    rows[n - 1].samples = true;
    rows
}

fn decode_plan() -> ForwardPlan {
    model::shared::llama_like::forward::llama_like_cuda(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        FireClass::Decode,
    )
}

/// The five live-verified llama_like deployments, each class — the
/// same set the goldens and the committed `.inc`s cover.
fn live_plans() -> Vec<(String, ForwardPlan)> {
    let cuda = LlamaLikeCudaFacts::qwen3_0_6b_l40s();
    let deployments = [
        ("qwen3_0_6b", LlamaLikeFacts::qwen3_0_6b()),
        ("qwen2_5_1_5b", LlamaLikeFacts::qwen2_5_1_5b()),
        ("phi3_mini", LlamaLikeFacts::phi3_mini()),
        ("mistral_7b_v03", LlamaLikeFacts::mistral_7b_v03()),
        ("olmo2_1b", LlamaLikeFacts::olmo2_1b()),
    ];
    let mut out = Vec::new();
    for (name, facts) in deployments {
        for class in [FireClass::Decode, FireClass::Prefill] {
            out.push((
                format!("{name}.{class:?}"),
                model::shared::llama_like::forward::llama_like_cuda(&facts, &cuda, class),
            ));
        }
    }
    out
}

/// The qwen3_5 family's residue LEDGER, pinned by kind and count.
///
/// llama_like's cutover was driven by exactly this: a ledger that
/// names what the flat list still does not carry, so each rung can
/// be read as a line leaving it. qwen3_5 has never had one — its
/// executor still walks, and "it walks" was the whole of what was
/// written down.
///
/// The counts are per fire, not per layer, so they move when a body
/// changes and stay put when a fixture does.
#[test]
fn the_qwen3_5_residue_ledger() {
    let facts = model::qwen_3_5::forward::facts::Qwen35HybridFacts::qwen3_5_0_8b();
    let cuda = model::qwen_3_5::forward::facts::Qwen35CudaFacts::qwen3_5_0_8b_synthetic();
    let plan = model::qwen_3_5::forward::qwen3_5_hybrid_cuda(&facts, &cuda, FireClass::Decode);
    let out = lower(&plan, &sampled(4), Fire::default()).expect("the plan lowers");
    let mut ledger: std::collections::BTreeMap<String, usize> = Default::default();
    for u in &out.residue {
        *ledger.entry(format!("{}: {}", u.kind, u.why)).or_default() += 1;
    }
    let seen: Vec<String> = ledger.iter().map(|(k, n)| format!("{n:>4}  {k}")).collect();
    let expected: Vec<String> = LEDGER_QWEN35_DECODE.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        seen, expected,
        "the qwen3_5 residue ledger moved.\n\
         Every line here is a statement the flat list does not carry. \
         If a rung removed one, update the constant and say which \
         statement now names its kernel; if a rung ADDED one, that is \
         a body stating something the lowering cannot read."
    );
}

/// The ledger's current contents — see [`the_qwen3_5_residue_ledger`].
/// One entry per (kind, reason), counted per DECODE fire.
const LEDGER_QWEN35_DECODE: &[&str] = &[];

/// The ALIGNED MoE leg lowers — the wall the north-star doc named.
///
/// `the_moe_block_covers_itself_only_in_its_cuda_reading` above already
/// covered the FUSED CUTLASS leg, which is the one a decode fire inside the
/// row bound takes. Every other fire fell back to the semantic body, and the
/// semantic body cannot lower — it names no kernels. That was the wall: the
/// aligned path's intermediates are `ceil((N·k + min(E, N·k)·(block-1)) /
/// block) · block` rows tall, an extent no `Dim` spelled.
///
/// `Dim::MoeAlignedRoutes` spells it, so the leg has a CUDA text and this
/// asserts what that buys: residue empty, coverage 1.0, on a deployment
/// whose facts disqualify the fused leg.
#[test]
fn the_aligned_moe_leg_lowers() {
    let facts = model::qwen_3_5::forward::facts::Qwen35MoeMlpFacts::qwen3_5_35b_a3b();
    // No CUTLASS workspace: the fused leg does not exist for this
    // deployment, so the text takes the aligned one.
    let mut cuda = model::qwen_3_5::forward::facts::Qwen35CudaFacts::qwen3_5_0_8b_synthetic();
    cuda.moe_cutlass_max_rows = 0;

    let plan = model::qwen_3_5::forward::qwen3_5_moe_mlp_block_cuda(&facts, &cuda);
    let out = lower(&plan, &sampled(4), Fire::default())
        .unwrap_or_else(|e| panic!("the aligned MoE leg must lower: {e:?}"));
    assert!(
        out.residue.is_empty(),
        "{} aligned-leg statements still owe a declaration: {:#?}",
        out.residue.len(),
        out.residue
    );
    assert_eq!(out.coverage(), 1.0);
}

/// THE QWEN3_5 CUTOVER GATE, in the shape llama_like's takes: every
/// statement a live fire executes is a rectangle in the flat list,
/// on both geometries and in both classes.
///
/// This was a CONTAINMENT test while the ledger was non-empty — 27B
/// owes nothing 0.8B does not — because asserting coverage would
/// have asserted something false about 0.8B too. With the ledger
/// empty the stronger claim is available, so it is the one made.
///
/// 27B earns its own row: it is the first geometry whose GDN half is
/// GQA (48 value heads over 16 key heads), which 0.8B cannot prove
/// either way.
#[test]
fn the_qwen3_5_flat_list_covers_every_statement() {
    let cuda = model::qwen_3_5::forward::facts::Qwen35CudaFacts::qwen3_5_0_8b_synthetic();
    let geometries = [
        (
            "0.8b",
            model::qwen_3_5::forward::facts::Qwen35HybridFacts::qwen3_5_0_8b(),
        ),
        (
            "27b",
            model::qwen_3_5::forward::facts::Qwen35HybridFacts::qwen3_6_27b(),
        ),
    ];
    for (name, facts) in geometries {
        for class in [FireClass::Decode, FireClass::Prefill] {
            let plan = model::qwen_3_5::forward::qwen3_5_hybrid_cuda(&facts, &cuda, class);
            for (shape, rows) in [("all-sampled", sampled(4)), ("gathered", gathered(4))] {
                let out = lower(&plan, &rows, Fire::default())
                    .unwrap_or_else(|e| panic!("{name}/{class:?}/{shape}: {e:?}"));
                assert!(
                    out.residue.is_empty(),
                    "{name}/{class:?}/{shape}: {} statements still owe a \
                     declaration: {:#?}",
                    out.residue.len(),
                    out.residue
                );
                assert_eq!(out.coverage(), 1.0, "{name}/{class:?}/{shape}");
                assert!(
                    !out.launches.is_empty(),
                    "{name}/{class:?}/{shape}: a fire that executes nothing \
                     is not a fire"
                );
            }
        }
    }
}

/// gemma-4's residue LEDGER — the third family's, opened the way
/// qwen3_5's was and for the same reason: a rung is legible when it
/// is a line leaving this list.
///
/// Empty means the body is a list of rectangles. It does NOT mean the
/// numbers are right: five of the six defects the executor found were
/// in a declaration whose ledger was already empty. This gate asks
/// whether statements are WELL FORMED; only a live fire asks whether
/// each one consumes what the pass produces.
#[test]
fn the_gemma4_residue_ledger() {
    let facts = model::gemma_4::forward::facts::Gemma4Facts::gemma_4_e4b();
    let cuda = model::gemma_4::forward::facts::Gemma4CudaFacts::gemma_4_e4b_synthetic();
    for (class, expected) in [
        (FireClass::Decode, LEDGER_GEMMA4_DECODE),
        (FireClass::Prefill, LEDGER_GEMMA4_PREFILL),
    ] {
        let plan = model::gemma_4::forward::gemma4_cuda(&facts, &cuda, class);
        let out = lower(&plan, &sampled(4), Fire::default()).expect("the plan lowers");
        let mut ledger: std::collections::BTreeMap<String, usize> = Default::default();
        for u in &out.residue {
            *ledger.entry(format!("{}: {}", u.kind, u.why)).or_default() += 1;
        }
        let seen: Vec<String> = ledger.iter().map(|(k, n)| format!("{n:>4}  {k}")).collect();
        let want: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
        assert_eq!(seen, want, "the gemma-4 {class:?} residue ledger moved");
        assert!(
            !out.launches.is_empty(),
            "a fire that executes nothing is not a fire"
        );
    }
}

/// See [`the_gemma4_residue_ledger`]. One entry per (kind, reason).
const LEDGER_GEMMA4_DECODE: &[&str] = &[];

/// The prefill class's, which differs from the decode ledger only in
/// the dispatch — and states two kernels of its own, so an empty
/// ledger here is a claim about those two as much as about the body.
const LEDGER_GEMMA4_PREFILL: &[&str] = &[];

/// gpt-oss's residue LEDGER — the fourth family's, opened the day the
/// text was written and before any executor exists.
///
/// gpt-oss is the first family whose MoE block is stated end to end,
/// which is the whole reason to open this list here: the decode leg
/// is seven rectangles because two GEMVs carry the expert axis
/// INSIDE the value, and if any of that were wrong it would show up
/// as a line below rather than as a wrong number later.
#[test]
fn the_gpt_oss_residue_ledger() {
    let facts = model::gpt_oss::forward::facts::GptOssFacts::gpt_oss_20b();
    let cuda = model::gpt_oss::forward::facts::GptOssCudaFacts::gpt_oss_20b_synthetic();
    for class in [FireClass::Decode, FireClass::Prefill] {
        let plan = model::gpt_oss::forward::gpt_oss_cuda(&facts, &cuda, class);
        let out = lower(&plan, &sampled(4), Fire::default()).expect("lowers");
        assert!(
            out.residue.is_empty() && out.coverage() == 1.0,
            "gpt-oss {class:?}: {:#?}",
            out.residue
        );
    }
    let plan = model::gpt_oss::forward::gpt_oss_cuda(&facts, &cuda, FireClass::Decode);
    let out = lower(&plan, &sampled(4), Fire::default()).expect("the plan lowers");
    let mut ledger: std::collections::BTreeMap<String, usize> = Default::default();
    for u in &out.residue {
        *ledger.entry(format!("{}: {}", u.kind, u.why)).or_default() += 1;
    }
    let seen: Vec<String> = ledger.iter().map(|(k, n)| format!("{n:>4}  {k}")).collect();
    let expected: Vec<String> = LEDGER_GPT_OSS_DECODE
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        seen, expected,
        "the gpt-oss residue ledger moved.\n\
         Every line here is a statement the flat list does not carry."
    );
    assert!(
        !out.launches.is_empty(),
        "a fire that executes nothing is not a fire"
    );
}

/// See [`the_gpt_oss_residue_ledger`]. One entry per (kind, reason).
const LEDGER_GPT_OSS_DECODE: &[&str] = &[];

/// THE GEMMA-4 CUTOVER GATE, in the shape the other two families'
/// take: every statement a live fire executes is a rectangle in the
/// flat list, in both classes and both logit shapes.
///
/// One geometry only, and honestly so: E4B is the sole gemma-4 fact
/// set anything has been read against. A second (E2B, the 31B) would
/// earn its own row the way 27B earned qwen3_5's.
#[test]
fn the_gemma4_flat_list_covers_every_statement() {
    // BOTH geometries. E2B is not a formality: 35 layers, MQA, 20/35
    // KV-shared and a DOUBLE-WIDE MLP, so it exercises
    // `intermediate_of`, the odd-layer interval and the unfused MLP
    // arm — three things E4B cannot say anything about. It found
    // three real gaps the day it was first booted.
    for (name, facts) in [
        (
            "e4b",
            model::gemma_4::forward::facts::Gemma4Facts::gemma_4_e4b(),
        ),
        (
            "e2b",
            model::gemma_4::forward::facts::Gemma4Facts::gemma_4_e2b(),
        ),
    ] {
        let _ = name;
        let cuda = model::gemma_4::forward::facts::Gemma4CudaFacts::gemma_4_e4b_synthetic();
        for class in [FireClass::Decode, FireClass::Prefill] {
            let plan = model::gemma_4::forward::gemma4_cuda(&facts, &cuda, class);
            for (shape, rows) in [("all-sampled", sampled(4)), ("gathered", gathered(4))] {
                let out = lower(&plan, &rows, Fire::default())
                    .unwrap_or_else(|e| panic!("{class:?}/{shape}: {e:?}"));
                assert!(
                    out.residue.is_empty(),
                    "{class:?}/{shape}: {} statements still owe a declaration: {:#?}",
                    out.residue.len(),
                    out.residue
                );
                assert_eq!(out.coverage(), 1.0, "{class:?}/{shape}");
                assert!(
                    !out.launches.is_empty(),
                    "{class:?}/{shape}: a fire that executes nothing is not a fire"
                );
            }
        }
    }
}

/// The MoE block's own ledger, and the argument for the fused leg.
///
/// The SEMANTIC reading is residue — a selector, a combine and a
/// shared-expert landing that no kernel is named for. The CUDA
/// reading of the same fragment is a list of rectangles. Both halves
/// are asserted here because either one alone is half the claim: a
/// covered CUDA reading proves the statements exist, and an
/// uncovered semantic one proves they were needed.
#[test]
fn the_moe_block_covers_itself_only_in_its_cuda_reading() {
    let facts = model::qwen_3_5::forward::facts::Qwen35MoeMlpFacts::qwen3_5_35b_a3b();
    let cuda = model::qwen_3_5::forward::facts::Qwen35CudaFacts::qwen3_5_0_8b_synthetic();

    // The semantic fragment names no backend at all, so it does not
    // reach the residue ledger — it is refused before any op is
    // read. That is the honest baseline: the MoE block had no CUDA
    // reading, not a partial one.
    let semantic = model::qwen_3_5::forward::qwen3_5_moe_mlp_block(&facts);
    assert!(
        matches!(
            lower(&semantic, &sampled(4), Fire::default()),
            Err(Uncovered::UnknownBackend(_))
        ),
        "the semantic MoE block named a backend — if it was given a \
         CUDA reading, this test is the one that should say so"
    );

    let declared = model::qwen_3_5::forward::qwen3_5_moe_mlp_block_cuda(&facts, &cuda);
    let out = lower(&declared, &sampled(4), Fire::default())
        .unwrap_or_else(|e| panic!("the CUDA MoE block must lower: {e:?}"));
    assert!(
        out.residue.is_empty(),
        "{} statements still owe a declaration: {:#?}",
        out.residue.len(),
        out.residue
    );
    assert_eq!(out.coverage(), 1.0);
}

/// THE CUTOVER GATE. Every statement a live fire executes is a
/// rectangle in the flat list — no residue, on every deployment the
/// driver serves, in both classes, sampled and unsampled.
///
/// This started as a ledger (88.7%-93.8%, residue `Swiglu` per
/// layer + `LmHead` per fire) and is now the gate itself: `launches`
/// is the WHOLE of what a fire runs, which is the property the driver
/// needs before it can stop walking. A regression here is a
/// statement that would silently not execute.
#[test]
fn the_flat_list_covers_every_statement() {
    for (name, plan) in live_plans() {
        // Both epilogue shapes: a decode fire samples every row, a
        // prefill fire samples one row per request and gathers.
        for (shape, rows) in [("all-sampled", sampled(4)), ("gathered", gathered(4))] {
            let out = lower(&plan, &rows, Fire::default())
                .unwrap_or_else(|e| panic!("{name}/{shape}: {e:?}"));
            assert!(
                out.residue.is_empty(),
                "{name}/{shape}: {} statements still owe a declaration: {:#?}",
                out.residue.len(),
                out.residue
            );
            assert_eq!(out.coverage(), 1.0, "{name}/{shape}");
            assert!(
                !out.launches.is_empty(),
                "{name}/{shape}: a fire that executes nothing is not a fire"
            );
        }
    }
}

/// A site inside an arm the guards did not take must NOT fire, and
/// the rectangles alone cannot say which those are — so the list
/// carries the live ones. This is what a form driven by the list
/// needs in order to bracket a layer's sideband correctly.
#[test]
fn the_live_sites_are_named_and_the_dead_ones_are_not() {
    let plan = decode_plan();
    let sites: Vec<usize> = plan
        .ops
        .iter()
        .enumerate()
        .filter(|(_, op)| matches!(op.kind, OpKind::HookSite { .. }))
        .map(|(i, _)| i)
        .collect();
    assert!(
        !sites.is_empty(),
        "the class trace carries observation sites"
    );

    // A plain fire takes the else arm; a MASKED fire takes the mask
    // arm. Both bracket their layers, and they are DIFFERENT sites —
    // which is the whole reason the list has to say.
    let plain_out = lower(&plan, &sampled(4), Fire::default()).unwrap();
    let mut masked = sampled(4);
    for r in &mut masked {
        r.custom_mask = true;
    }
    let masked_out = lower(&plan, &masked, Fire::default()).unwrap();

    for out in [&plain_out, &masked_out] {
        assert!(
            !out.structural.is_empty(),
            "a live fire brackets its layers"
        );
        assert!(
            out.structural
                .iter()
                .all(|s| sites.contains(&(s.at_op as usize))),
            "only sites are structural"
        );
        // Ordered, because a bracket opens before it closes.
        assert!(out.structural.windows(2).all(|w| w[0].at_op < w[1].at_op));
        // And every site brackets a NON-EMPTY window — an empty one
        // would be a retired layer's, which does not fire at all.
        assert!(out.structural.iter().all(|s| !s.rows.is_empty()));
    }
    assert_ne!(
        plain_out.structural, masked_out.structural,
        "the two arms bracket through different sites"
    );
    // And a dead arm's sites are absent from BOTH.
    assert!(plain_out.structural.len() < sites.len());
    assert!(masked_out.structural.len() < sites.len());
}

/// A CAPTURED fire emits both peel regions whatever its split is,
/// and says so on the launches — the one place a rectangle is not a
/// pair of numbers.
///
/// The shadow comparison is what asked for this: the walk emits both
/// regions under device-window capture (an empty one early-outs on
/// the device word, so one graph replays across every split) and the
/// flat list described only the non-empty one.
#[test]
fn a_captured_fire_emits_both_peel_regions() {
    let plan = decode_plan();
    // fast_rows == 0: every row hooked, so the hook-free prefix is
    // empty. Uncaptured, it contributes nothing.
    let mut rows = sampled(4);
    for r in &mut rows {
        r.hooked = true;
    }
    let host = lower(&plan, &rows, Fire::default()).expect("coverable");
    assert!(
        host.launches
            .iter()
            .all(|l| l.peel.is_none_or(|p| !p.rows_device))
    );
    assert!(
        !host
            .launches
            .iter()
            .any(|l| l.kernel_is(&host, "attn::qkv_decode_qk_norm_rope_write_kv_bf16")),
        "an empty prefix launches nothing when the host's count is the truth"
    );

    // Captured: the prefix's launches ARE in the list, marked as
    // reading the fire's split rather than these counts.
    let captured = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: true,
        },
    )
    .expect("coverable");
    let fused: Vec<_> = captured
        .launches
        .iter()
        .filter(|l| l.kernel_is(&captured, "attn::qkv_decode_qk_norm_rope_write_kv_bf16"))
        .collect();
    assert!(!fused.is_empty(), "the captured graph carries the prefix");
    assert!(fused.iter().all(|l| {
        l.peel
            .is_some_and(|p| p.axis == PeelWindow::HookFreePrefix && p.rows_device)
    }));
    // And its rows are the WHOLE window, not the empty prefix half:
    // a captured region launches a full-window grid and reads the
    // split off the device word. Naming the half would describe a
    // grid nobody launches, and an executor that believed it would
    // bake this fire's split into the graph — wrong only on the
    // REPLAY, which is why this is asserted here rather than left to
    // a parity run to notice.
    assert!(
        fused
            .iter()
            .all(|l| l.rows.start == 0 && l.rows.end == rows.len() as u32),
        "a captured peel region's rectangle is the full window"
    );

    // And ONLY the peel's regions are marked: everything outside is
    // still a plain count, which is what keeps the list readable.
    assert!(
        captured
            .launches
            .iter()
            .filter(|l| l.peel.is_some_and(|p| p.rows_device))
            .count()
            < captured.launches.len()
    );
}

/// The epilogue is three statements over a ROW COUNT, and the two
/// runtime branches the executor takes are the count being zero and
/// the count being short.
#[test]
fn the_epilogue_is_a_row_count_not_a_branch() {
    let plan = decode_plan();
    // The epilogue's launches are the ones carrying the LmHead
    // statement's index. Identifying them by SYMBOL would not work:
    // its projection is `kernels::gemm::act_x_w`, the same launcher every
    // body matmul takes.
    let at_op = plan
        .ops
        .iter()
        .position(|op| matches!(op.kind, OpKind::LmHead { .. }))
        .expect("the class trace has an epilogue") as u32;
    let epilogue = |rows: &[Row]| -> Vec<(String, Range<u32>)> {
        let out = lower(&plan, rows, Fire::default()).expect("coverable");
        out.launches
            .iter()
            .filter(|l| l.op == at_op)
            .map(|l| (out.kernels[l.kernel as usize].clone(), l.rows.clone()))
            .collect()
    };

    // Every row sampled: project over all four rows, no gather — there is
    // nothing to skip past.
    //
    // NO NORM. The epilogue used to emit `norm::rmsnorm_bf16` here and it
    // was dead on every fire: each text applies the final norm itself and
    // hands `logits()` the normed value, so the epilogue's own norm read
    // an already-normed input and wrote into the logits buffer that the
    // projection overwrites on the next launch. See
    // `the_epilogue_binds_one_ops_two_operands_to_the_launches` for how
    // that was invisible — this assertion named the symbols, and the
    // symbols were right.
    let all = epilogue(&sampled(4));
    assert_eq!(all, vec![("gemm::act_x_w".to_string(), 0..4)]);

    // One sampled row of four: the gather appears, and both statements
    // run over ONE row while the body ran over four — the epilogue's row
    // space is Requests.
    assert_eq!(
        epilogue(&gathered(4)),
        vec![
            ("layout::gather_bf16_rows".to_string(), 0..1),
            ("gemm::act_x_w".to_string(), 0..1),
        ]
    );

    // Nothing sampled (`emit_logits == false`, a fire whose logits
    // nobody reads): no rectangle at all, while the body still runs.
    let none = vec![Row::default(); 4];
    assert!(epilogue(&none).is_empty());
    assert!(
        !lower(&plan, &none, Fire::default())
            .unwrap()
            .launches
            .is_empty()
    );
}

/// A plain fire lowers, and every launch covers every row — the
/// degenerate rectangle, which is what today's fires are.
#[test]
fn a_plain_fire_is_one_rectangle_per_statement() {
    let plan = decode_plan();
    let rows = plain(8);
    let out = lower(&plan, &rows, Fire::default()).expect("a plain fire is coverable");
    assert!(out.rectangles > 0);
    assert!(out.launches.iter().all(|l| l.rows == (0..8)));
    // The frame's kernel table is what the driver would index.
    assert!(
        out.kernels
            .contains(&"attn::dispatch_attention_flashinfer_decode".to_string())
    );
    // Every launch names a layer the trace tagged.
    assert!(
        out.launches
            .iter()
            .all(|l| l.layers.end == l.layers.start + 1)
    );
}

/// The MASK arm selects only the masked rows, and the rest take the
/// plain body — one statement, two rectangles. This is the thing the
/// flat ABI buys: today the same fire is a guard the driver walks.
#[test]
fn a_masked_suffix_splits_the_rectangle() {
    let plan = decode_plan();
    // The seriation puts masked rows last.
    let mut rows = plain(8);
    for r in &mut rows[6..] {
        r.custom_mask = true;
    }
    let out = lower(&plan, &rows, Fire::default()).expect("mask + plain is coverable");
    let masked = out.launches.iter().filter(|l| l.rows == (6..8)).count();
    let plain_rows = out.launches.iter().filter(|l| l.rows == (0..6)).count();
    assert!(masked > 0, "the masked rows got their own rectangles");
    assert!(plain_rows > 0, "and the plain rows theirs");
    // More rectangles than the unsplit fire — what the row order
    // costs, reported rather than acted on.
    let flat = lower(&plan, &plain(8), Fire::default()).unwrap();
    assert!(out.rectangles > flat.rectangles);
}

/// A DISCONTIGUOUS order is refused rather than silently mis-served.
/// The engine's seriation guarantees contiguity per axis; if it ever
/// stops, this is the answer, and it is an admission answer.
#[test]
fn a_discontiguous_order_is_uncovered() {
    let plan = decode_plan();
    let mut rows = plain(8);
    rows[1].custom_mask = true;
    rows[5].custom_mask = true;
    assert!(matches!(
        lower(&plan, &rows, Fire::default()),
        Err(Uncovered::Discontiguous { .. })
    ));
}

/// `whole` CONSUMED: an XQA deployment's fire may not be lowered
/// with the kernel over a subset. Statically the check refuses it
/// inside a Peel; here it refuses the dynamic case too.
#[test]
fn a_whole_kernel_refuses_a_row_window() {
    let facts = LlamaLikeFacts::qwen3_0_6b();
    let cuda = LlamaLikeCudaFacts {
        xqa_decode: true,
        decode_fused_post: false,
        ..LlamaLikeCudaFacts::qwen3_0_6b_l40s()
    };
    let plan =
        model::shared::llama_like::forward::llama_like_cuda(&facts, &cuda, FireClass::Decode);
    assert!(
        plan.ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Launch { kernel, .. }
                if kernel == "attn::attention_xqa_decode_bf16_prepared"
        )),
        "this deployment states XQA"
    );
    // Whole fire: fine.
    assert!(lower(&plan, &plain(8), Fire::default()).is_ok());
    // And a MASKED fire is fine too, which is the point: a guard is
    // a fire fact, so the mask arm takes the whole fire and XQA — in
    // the else arm — does not run at all. Nothing hands a kernel a
    // row window except a Peel, and a `whole` kernel inside a Peel
    // is refused STATICALLY at trace time (`kernels::check_plan`),
    // so this dynamic check is a backstop rather than the rule's
    // live enforcement. It stays because the flat list is about to
    // become the thing that executes.
    let mut rows = plain(8);
    for r in &mut rows[6..] {
        r.custom_mask = true;
    }
    assert!(lower(&plan, &rows, Fire::default()).is_ok());
}

/// Liveness reuse is the point of assigning buffers here: a
/// 28-layer unrolled plan names 28 distinct normed-activation values
/// whose ranges never overlap, so the arena must be far smaller than
/// the naive sum.
#[test]
fn the_arena_reuses_across_layers() {
    let plan = decode_plan();
    let rows = plain(8);
    let buffers = Buffers::assign(&plan, &rows);
    let naive: usize = (0..plan.values.len())
        .map(|v| value_bytes(&plan, v as ValueId, rows.len(), rows.len()))
        .sum();
    assert!(buffers.bytes > 0);
    assert!(
        buffers.bytes * 4 < naive,
        "arena {} vs naive {naive}",
        buffers.bytes
    );
    // Pinned values are the backend's to bind, not the arena's.
    assert!(
        buffers
            .pinned
            .iter()
            .all(|&v| buffers.offset[v as usize] == Buffers::NAMED)
    );
    // Pins come off the seam statements, not a per-family table.
    assert!(
        !buffers.pinned.is_empty(),
        "this text states observation seams, so some values are exposed"
    );
}

/// FOUR distinct truncations lower fine. The driver's
/// `derive_depth_bands` refuses a fourth band (`if (count == 3)
/// return 0`) because its walk carries per-band plans; here a
/// layer's live row count is a number, so the ceiling has nowhere to
/// live. This is step 5's driver half, on the host side.
#[test]
fn depth_has_no_band_ceiling() {
    let plan = decode_plan();
    // Seriation order: full-depth first, then truncated deepest-first.
    let mut rows = plain(10);
    for (i, k) in [(2usize, 24u32), (4, 20), (6, 16), (8, 8)] {
        for r in &mut rows[i..] {
            r.depth_k = Some(k);
        }
    }
    let out = lower(&plan, &rows, Fire::default()).expect("four bands is not a special case");
    // Layer 0 runs over everybody; layer 23 only over the rows whose
    // k is past it (the full-depth prefix plus the k=24 block).
    let at = |l: u16| {
        out.launches
            .iter()
            .filter(|x| x.layers.start == l)
            .map(|x| x.rows.end)
            .max()
            .unwrap_or(0)
    };
    // rows 0-1 full depth, 2-3 k=24, 4-5 k=20, 6-7 k=16, 8-9 k=8;
    // a row is live at layer l while l < k, so it dies AT l == k.
    assert_eq!(at(0), 10);
    assert_eq!(at(7), 10);
    assert_eq!(at(8), 8, "the k=8 pair dies at layer 8");
    assert_eq!(at(16), 6);
    assert_eq!(at(20), 4);
    assert_eq!(at(23), 4);
    assert_eq!(at(24), 2, "only the full-depth rows are left");
    assert_eq!(at(27), 2);
}

/// A uniform truncation SKIPS the tail layers entirely — no launch
/// is emitted where nothing is live.
#[test]
fn a_uniform_truncation_skips_the_tail() {
    let plan = decode_plan();
    let rows = vec![
        Row {
            depth_k: Some(12),
            ..Row::default()
        };
        4
    ];
    let out = lower(&plan, &rows, Fire::default()).unwrap();
    assert!(
        out.launches
            .iter()
            .all(|l| l.layers.start < 12 || l.layers.start >= 28 || l.rows.is_empty())
    );
    let full = lower(&plan, &plain(4), Fire::default()).unwrap();
    assert!(out.rectangles < full.rectangles, "truncation costs less");
}

/// The arena is DETERMINISTIC in ask order — the property a replayed
/// graph needs, since the same plan must land the same value at the
/// same address on every fire.
#[test]
fn the_arena_is_deterministic() {
    let plan = decode_plan();
    let a = Buffers::assign(&plan, &plain(8));
    let b = Buffers::assign(&plan, &plain(8));
    assert_eq!(a.offset, b.offset);
    assert_eq!(a.bytes, b.bytes);
}

/// Not an assertion so much as a printout: `-- --ignored --nocapture` dumps
/// the aligned MoE leg's op sequence, which is the list the driver's declared
/// executor has to answer arm for arm.
#[test]
#[ignore]
fn dump_the_aligned_moe_leg() {
    let facts = model::qwen_3_5::forward::facts::Qwen35MoeMlpFacts::qwen3_5_35b_a3b();
    let mut cuda = model::qwen_3_5::forward::facts::Qwen35CudaFacts::qwen3_5_0_8b_synthetic();
    cuda.moe_cutlass_max_rows = 0;
    let plan = model::qwen_3_5::forward::qwen3_5_moe_mlp_block_cuda(&facts, &cuda);
    for (i, op) in plan.ops.iter().enumerate() {
        println!("{i:3}  {:?}", op.kind);
    }
}

/// Same printout for gemma-4's CUDA decode text — the driver refuses this
/// plan on a kernel it does not have an arm for, and the trace is where to
/// see which.
#[test]
#[ignore]
fn dump_gemma4_cuda_kernels() {
    let f = model::gemma_4::forward::facts::Gemma4Facts::gemma_4_e4b();
    let cuda = model::gemma_4::forward::facts::Gemma4CudaFacts::gemma_4_e4b_synthetic();
    let plan = model::gemma_4::forward::gemma4_cuda(&f, &cuda, FireClass::Decode);
    let mut names: Vec<String> = plan
        .ops
        .iter()
        .filter_map(|o| match &o.kind {
            OpKind::Launch { kernel, .. } => Some(kernel.clone()),
            _ => None,
        })
        .collect();
    names.sort();
    names.dedup();
    for n in names {
        println!("KERNEL {n}");
    }
}

/// gpt-oss's stated CUDA kernels. The hand-written pass is the broken side
/// for this family (`.wiki/tart/status.md`), and the declared drive is
/// clean, so the DIFFERENCE between what the declaration says and what
/// `mixtral.cpp` fires is the suspect set -- a directed search, where the
/// env knobs have run out.
#[test]
#[ignore]
fn dump_gpt_oss_cuda_kernels() {
    let f = model::gpt_oss::forward::facts::GptOssFacts::gpt_oss_20b();
    let cuda = model::gpt_oss::forward::facts::GptOssCudaFacts::gpt_oss_20b_synthetic();
    for class in [FireClass::Decode, FireClass::Prefill] {
        let plan = model::gpt_oss::forward::gpt_oss_cuda(&f, &cuda, class);
        let mut names: Vec<String> = plan
            .ops
            .iter()
            .filter_map(|o| match &o.kind {
                OpKind::Launch { kernel, .. } => Some(kernel.clone()),
                _ => None,
            })
            .collect();
        names.sort();
        names.dedup();
        println!("== {class:?}");
        for n in names {
            println!("  {n}");
        }
    }
}

/// glm5's CUDA decode text lowers with nothing left over.
///
/// The gate every declaration is measured against: residue empty means
/// the flat list IS the whole of what the fire executes, and only then
/// could a driver stop walking. glm5 has no declared executor yet, so
/// this is the check that the TEXT is finished — the executor is the
/// next question, not this one.
#[test]
fn the_glm5_decode_text_lowers() {
    let facts = model::glm_5::forward::facts::Glm5Facts::glm5_106b_a12b();
    let plan = model::glm_5::forward::glm5_cuda(&facts, FireClass::Decode);
    let out = lower(&plan, &sampled(1), Fire::default())
        .unwrap_or_else(|e| panic!("glm5's decode text must lower: {e:?}"));
    assert!(
        out.residue.is_empty(),
        "{} glm5 statement(s) still owe a declaration: {:#?}",
        out.residue.len(),
        out.residue
    );
    assert_eq!(out.coverage(), 1.0);
}

/// kimi's CUDA decode text lowers with nothing left over — the fused
/// latent binding and the split one both, since which is bound is a fact
/// and a fact that only lowers one way is a fact stated wrong.
#[test]
fn the_kimi_decode_text_lowers() {
    use model::kimi_k2::forward::facts::{KimiCudaFacts, KimiFacts};
    let facts = KimiFacts::kimi_k2();
    for fused in [true, false] {
        let cuda = KimiCudaFacts {
            q_kv_a_fused: fused,
            ..KimiCudaFacts::kimi_k2_synthetic()
        };
        let plan = model::kimi_k2::forward::kimi_cuda(&facts, &cuda, FireClass::Decode);
        let out = lower(&plan, &sampled(1), Fire::default())
            .unwrap_or_else(|e| panic!("kimi (fused={fused}) must lower: {e:?}"));
        assert!(
            out.residue.is_empty(),
            "fused={fused}: {} statement(s) still owe a declaration: {:#?}",
            out.residue.len(),
            out.residue
        );
        assert_eq!(out.coverage(), 1.0);
    }
}

/// kimi_k3's CUDA decode text lowers with nothing left over.
///
/// The hybrid matters here: the fixture's schedule puts both an MLA layer
/// and a KDA layer in the plan, so this covers both halves — a text that
/// only lowered one would pass a single-kind fixture and fail the first
/// real deployment.
#[test]
fn the_kimi_k3_decode_text_lowers() {
    use model::kimi_k3::forward::facts::KimiK3Facts;
    let facts = KimiK3Facts::kimi_k3_synthetic();
    assert!(
        (0..facts.layers).any(|l| facts.is_full_attn(l))
            && (0..facts.layers).any(|l| !facts.is_full_attn(l)),
        "the fixture must exercise BOTH halves of the hybrid"
    );
    let plan = model::kimi_k3::forward::kimi_k3_cuda(&facts, FireClass::Decode);
    let out = lower(&plan, &sampled(1), Fire::default())
        .unwrap_or_else(|e| panic!("kimi_k3's decode text must lower: {e:?}"));
    assert!(
        out.residue.is_empty(),
        "{} statement(s) still owe a declaration: {:#?}",
        out.residue.len(),
        out.residue
    );
    assert_eq!(out.coverage(), 1.0);
}

/// deepseek_v4's CUDA decode text lowers with nothing left over.
///
/// The two schemes this family alone carries are what the gate is for:
/// a rank-K residual that never spells `y += ...`, and a two-pass
/// attention combined by its LSEs.
#[test]
fn the_deepseek_v4_decode_text_lowers() {
    use model::deepseek_v4::forward::facts::Dsv4Facts;
    let facts = Dsv4Facts::dsv4_synthetic();
    let plan = model::deepseek_v4::forward::dsv4_cuda(&facts, FireClass::Decode);
    let out = lower(&plan, &sampled(1), Fire::default())
        .unwrap_or_else(|e| panic!("deepseek_v4's decode text must lower: {e:?}"));
    assert!(
        out.residue.is_empty(),
        "{} statement(s) still owe a declaration: {:#?}",
        out.residue.len(),
        out.residue
    );
    assert_eq!(out.coverage(), 1.0);
}

/// nemotron_h's CUDA decode text lowers with nothing left over.
///
/// The fixture's own test asserts all THREE layer kinds are present, so
/// this covers the mamba scan, the attention mixer and the mixer-less MLP
/// layer in one plan — which is the only way to be sure a list-shaped
/// schedule was read as a list.
#[test]
fn the_nemotron_h_decode_text_lowers() {
    use model::nemotron_h::forward::facts::NemotronHFacts;
    let facts = NemotronHFacts::nemotron_h_synthetic();
    let plan = model::nemotron_h::forward::nemotron_h_cuda(&facts, FireClass::Decode);
    let out = lower(&plan, &sampled(1), Fire::default())
        .unwrap_or_else(|e| panic!("nemotron_h's decode text must lower: {e:?}"));
    assert!(
        out.residue.is_empty(),
        "{} statement(s) still owe a declaration: {:#?}",
        out.residue.len(),
        out.residue
    );
    assert_eq!(out.coverage(), 1.0);
}

/// gemma3n's CUDA decode text lowers with nothing left over — the sixth
/// and last of the families that existed only as C++.
///
/// It is also the text that needed two primitives nothing else did:
/// `select` for the window AltUp's body reads, and `in_place` for the
/// per-layer embedding's K-1 adds landing back in the windows they read.
#[test]
fn the_gemma3n_decode_text_lowers() {
    use model::gemma_3n::forward::facts::Gemma3nFacts;
    let facts = Gemma3nFacts::gemma3n_synthetic();
    let plan = model::gemma_3n::forward::gemma3n_cuda(&facts, FireClass::Decode);
    let out = lower(&plan, &sampled(1), Fire::default())
        .unwrap_or_else(|e| panic!("gemma3n's decode text must lower: {e:?}"));
    assert!(
        out.residue.is_empty(),
        "{} statement(s) still owe a declaration: {:#?}",
        out.residue.len(),
        out.residue
    );
    assert_eq!(out.coverage(), 1.0);
}

/// gemma-2's CUDA decode text lowers with nothing left over — the last
/// family in the driver that had no declaration at all.
#[test]
fn the_gemma_2_decode_text_lowers() {
    use model::gemma_2::forward::facts::Gemma2Facts;
    let facts = Gemma2Facts::gemma_2_9b();
    let plan = model::gemma_2::forward::gemma2_cuda(&facts, FireClass::Decode);
    let out = lower(&plan, &sampled(1), Fire::default())
        .unwrap_or_else(|e| panic!("gemma-2's decode text must lower: {e:?}"));
    assert!(
        out.residue.is_empty(),
        "{} statement(s) still owe a declaration: {:#?}",
        out.residue.len(),
        out.residue
    );
    assert_eq!(out.coverage(), 1.0);
}

/// Every rectangle carries its own operands, and they agree with the
/// arena.
///
/// This is the host half of the family-independent driver (north-star
/// step 6). Today's four `declared_forward.cpp` walk the TRACED ops and
/// answer "which buffer is this operand?" with a per-family workspace
/// field — `ws.norm_x`, `la.mixed_qkv` — which is the only reason they
/// cannot be one file. A launch that carries its operands answers it in
/// the lowering, once, for every family.
///
/// So the claim is narrow and total: no launch has an empty operand run,
/// and every arena operand names the offset `Buffers` assigned it.
#[test]
fn every_launch_carries_operands_that_match_the_arena() {
    use model_compiler::lower::Arg;

    let plan = decode_plan();
    let rows = plain(8);
    let out = lower(&plan, &rows, Fire::default()).expect("must lower");
    let buffers = Buffers::assign(&plan, &rows);

    assert!(!out.launches.is_empty());
    let mut arena_args = 0usize;
    for l in &out.launches {
        assert!(
            l.args.end > l.args.start,
            "a rectangle with no operands cannot be driven"
        );
        for a in &out.args[l.args.start as usize..l.args.end as usize] {
            match a {
                Arg::Arena { at, width, .. } => {
                    arena_args += 1;
                    assert!(
                        *at < out.arena_bytes,
                        "an operand outside the arena ({at} vs {})",
                        out.arena_bytes
                    );
                    // A zero width is the lowering saying "this operand
                    // has no fixed row width", which no statement in the
                    // tree produces — so it reads as a resolver bug.
                    assert!(*width > 0, "an activation operand with no width");
                }
                Arg::Named { value: v, .. } => assert_eq!(
                    buffers.offset[*v as usize],
                    Buffers::NAMED,
                    "a Named operand must be one the arena declined"
                ),
                Arg::Weight(n) => assert!(!n.is_empty()),
            }
        }
    }
    assert!(
        arena_args > out.launches.len(),
        "most operands are activations; {arena_args} for {} launches reads \
         like the resolver is not running",
        out.launches.len()
    );
}

/// The operand slots name weights from the PLAN's table, not the
/// lowering's.
///
/// The confusion this guards is real and easy: a lowering hands back TWO
/// name tables — its own, holding launcher SYMBOLS, and the plan's,
/// holding weights. An operand resolved against the wrong one gives the
/// driver a kernel name where a tensor belongs, and both are valid u32.
#[test]
fn the_operand_slots_name_weights_from_the_plan() {
    use model_compiler::lower::Arg;

    let plan = decode_plan();
    let rows = plain(4);
    let out = lower(&plan, &rows, Fire::default()).expect("must lower");

    // At least one launch names a weight, or this proves nothing.
    let weights: Vec<&String> = out
        .args
        .iter()
        .filter_map(|a| match a {
            Arg::Weight(n) => Some(n),
            _ => None,
        })
        .collect();
    assert!(
        !weights.is_empty(),
        "this text's launches name weights; none reached the slots"
    );
    // Every one is a name the PLAN states, not a launcher symbol.
    for w in &weights {
        assert!(
            plan.ops.iter().any(|o| match &o.kind {
                OpKind::Launch { weights, .. } => weights.iter().any(|n| &n == w),
                _ => false,
            }),
            "`{w}` is not a weight any statement names"
        );
        assert!(
            !out.kernels.iter().any(|k| &k == w),
            "`{w}` is a LAUNCHER symbol — the two name tables were crossed"
        );
    }
}

/// The buffer table crosses beside the operands, and the two AGREE.
///
/// They are two views of one assignment: `args` carries an offset per
/// operand a rectangle names, `value_offsets` carries one per value id.
/// A driver mid-migration reads the table (it walks ops, so it has no
/// rectangle to read `args` from) and a driver on the flat list reads
/// the operands, so the families would disagree about where a value
/// lives if these ever drifted — and nothing else would say so, because
/// each is internally consistent.
#[test]
fn the_buffer_table_crosses_and_agrees_with_the_operands() {
    use model_compiler::lower::Arg;

    let plan = decode_plan();
    let rows = plain(4);
    let out = lower(&plan, &rows, Fire::default()).expect("must lower");

    assert_eq!(
        out.value_offset.len(),
        plan.values.len(),
        "the table is indexed by value id, so it is one entry per value"
    );

    // Every operand slot resolves to the same bytes the table names.
    let mut checked = 0usize;
    for l in &out.launches {
        for a in &out.args[l.args.start as usize..l.args.end as usize] {
            match a {
                Arg::Named { value, .. } => {
                    assert_eq!(
                        out.value_offset[*value as usize],
                        Buffers::NAMED,
                        "value {value} crossed as Named but the table places it"
                    );
                    checked += 1;
                }
                // An arena operand carries the offset directly, so the
                // agreement to check is that the table is not NAMED
                // there — the offset itself came from the same vector.
                Arg::Arena { at, .. } => {
                    assert!(
                        *at < out.arena_bytes,
                        "an operand outside the arena it reports"
                    );
                    checked += 1;
                }
                Arg::Weight(_) => {}
            }
        }
    }
    assert!(checked > 0, "no activation operands reached the check");

    // Nothing the table places sits past the block it reports.
    for (v, &at) in out.value_offset.iter().enumerate() {
        if at == Buffers::NAMED {
            continue;
        }
        assert!(
            at < out.arena_bytes,
            "value {v} is placed at {at}, past the reported {} bytes",
            out.arena_bytes
        );
    }
}

/// The epilogue's launches all bind ONE op's two operands, which is what
/// the gather still needs and what killed the norm.
///
/// `lower::epilogue` emits its statements over a single `OpKind::LmHead`,
/// whose `inputs` is `[hidden]` and `outputs` is `[logits]`. There is no
/// third value, so every launch sees exactly `[hidden, logits]`. Two
/// consequences, and only one of them is fixed.
///
/// **The rmsnorm was dead, and is gone.** Every text applies the final
/// norm itself and hands `logits()` the normed value —
/// `m.logits(&dsl::cuda::rmsnorm(&y, &m.final_norm()))` in llama_like,
/// `lm_head_tied(t, &normed, ..)` in gemma-4, `lm_head(rmsnorm(y,
/// final_norm))` in qwen3.5 — so the epilogue's own norm read an
/// already-normed input and wrote `rows x hidden` bf16 into the LOGITS
/// buffer, which the projection overwrites on the next launch. A wasted
/// kernel on every fire of every family.
///
/// **The gather still has nowhere to write.** It must compact `[sampled,
/// hidden]` rows for the head to read, and its only output is the logits
/// allocation, at the wrong width and stride. That is why
/// `driver-cuda` over-claims `Row::samples`: reading the step's real
/// readout list makes every prefill state this gather, and running it
/// produces all-zero logits. Fixing it needs a temp value, which is a
/// TRACE change — `OpKind::LmHead` has to name the gather's destination —
/// rather than a lowering one.
///
/// This test reads the ARGS. The one above reads the symbols, and that is
/// exactly why neither problem was visible: the symbols were right.
#[test]
fn the_epilogue_binds_one_ops_two_operands_to_the_launches() {
    let plan = decode_plan();
    let at_op = plan
        .ops
        .iter()
        .position(|op| matches!(op.kind, OpKind::LmHead { .. }))
        .expect("the class trace has an epilogue") as u32;

    let args_of = |rows: &[Row]| -> Vec<(String, Vec<String>)> {
        let out = lower(&plan, rows, Fire::default()).expect("coverable");
        out.launches
            .iter()
            .filter(|l| l.op == at_op)
            .map(|l| {
                (
                    out.kernels[l.kernel as usize].clone(),
                    out.args[l.args.start as usize..l.args.end as usize]
                        .iter()
                        .map(|a| match a {
                            Arg::Arena { width, .. } => format!("arena/{width}"),
                            Arg::Named { width, .. } => format!("named/{width}"),
                            Arg::Weight(w) => format!("weight/{w}"),
                        })
                        .collect(),
                )
            })
            .collect()
    };

    // One sampled row of four: the gather appears, and it writes into a
    // TEMP the trace never named — `Buffers::epilogue_gather`, sized from
    // this statement and carried on `Lowered` all along, which nothing
    // ever bound.
    //
    // It used to get the head's own two operands, because
    // `OpKind::LmHead` states `inputs=[hidden] outputs=[logits]` and the
    // plain emit binds exactly those. So the compaction's destination was
    // the LOGITS buffer at the wrong width, and the head then read what it
    // had overwritten: all-zero logits on gemma-4 and the hybrid. The
    // shell worked around it by forcing every row to sample, which costs a
    // prefill its whole head over every token.
    //
    // This test pinned the defect for as long as it asserted only the
    // SYMBOLS and the row ranges. It reads the ARGUMENTS now, which is the
    // only reason the hand-off can be held.
    let mut gathered = vec![Row::default(); 4];
    gathered[3].samples = true;
    let got = args_of(&gathered);
    let names: Vec<&str> = got.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        names,
        ["layout::gather_bf16_rows", "gemm::act_x_w"],
        "the epilogue is a compaction and a projection — the norm the texts \
         already did is not one of its statements"
    );
    let widths: Vec<&Vec<String>> = got.iter().map(|(_, a)| a).collect();
    assert_eq!(widths[0].len(), 2, "gather: the stream in, the temp out");
    assert_eq!(widths[1].len(), 2, "head: the temp in, the logits out");
    assert_eq!(
        widths[0][0], widths[0][1],
        "the gather reads the hidden stream and writes the temp — both at \
         the HIDDEN width, which is what makes it a compaction rather than \
         a projection"
    );
    assert_eq!(
        widths[0][1], widths[1][0],
        "THE HAND-OFF: the gather's destination is the head's source. \
         Anything else and the head reads rows nobody compacted"
    );
    assert_ne!(
        widths[1][0], widths[1][1],
        "the head's output is the logits buffer, at the vocabulary width — \
         so the temp is not it"
    );

    // Every row sampling: no compaction is needed, so none is stated, and
    // the head reads the stream directly.
    let all = vec![
        Row {
            samples: true,
            ..Default::default()
        };
        4
    ];
    let got = args_of(&all);
    assert_eq!(
        got.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
        ["gemm::act_x_w"],
        "a fire that reads every row has nothing to gather"
    );
}

/// A row that carries an adapter reaches the correction launch.
///
/// The whole point of reading the step's region table. `attn.qv` opens a
/// `HasLora` guard whose then-arm is `cuda::lora_qkv_correction`, and
/// `lower::select` resolves that guard with `rows.iter().any(|r|
/// r.lora)`. `driver-cuda` built every row with `Row::default()` and
/// never read `region_sig`, so the answer was NO on every fire no matter
/// what the engine sent — which is the whole of "LoRA is ported and
/// never applied".
///
/// The adapter itself is not a fixed form: `fwd.adapter(site, |x, y|
/// expr)` recognises LoRA, IA3 and DoRA and lowers them to a pass-wide
/// `lora` SINK in the prologue whose A/B are channel cells. This test is
/// about the OTHER half — the backbone's correction launch, which is what
/// the guard gates and what the region bit selects.
#[test]
fn a_lora_row_states_the_correction_and_a_plain_one_does_not() {
    let plan = decode_plan();
    let launches = |rows: &[Row]| -> Vec<String> {
        let out = lower(&plan, rows, Fire::default()).expect("coverable");
        out.launches
            .iter()
            .map(|l| out.kernels[l.kernel as usize].clone())
            .collect()
    };

    let plain = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        4
    ];
    assert!(
        !launches(&plain).iter().any(|k| k.contains("lora")),
        "a fire with no adapter states no correction"
    );

    // ONE row carrying it is enough: the guard is fire-wide
    // (`rows.iter().any`), because a correction launch spans the fire and
    // the lanes that carry no adapter get an identity from the staged
    // table rather than a different launch list.
    let mut adapted = plain.clone();
    adapted[2].lora = true;
    let with = launches(&adapted);
    assert!(
        with.iter().any(|k| k.contains("lora")),
        "a row carrying an adapter states the correction; got {with:?}"
    );

    // And it is the region bit that decides, end to end: the same rows
    // built the way the wire states them.
    let from_wire = model_compiler::lower::rows_from_regions(
        4,
        model_compiler::lower::Readouts {
            indices: &[3],
            indptr: &[0, 1],
            qo_indptr: &[0, 4],
        },
        &[0, 2, 4],
        &[0, model_compiler::lower::region_sig::LORA],
        &[model_compiler::lower::region_sig::MAX_LAYERS_FULL; 2],
    )
    .expect("a tiling table");
    assert!(
        launches(&from_wire).iter().any(|k| k.contains("lora")),
        "PIE_REGION_SIG_LORA on a region reaches the correction launch"
    );
    let no_bit = model_compiler::lower::rows_from_regions(
        4,
        model_compiler::lower::Readouts {
            indices: &[3],
            indptr: &[0, 1],
            qo_indptr: &[0, 4],
        },
        &[0, 2, 4],
        &[0, 0],
        &[model_compiler::lower::region_sig::MAX_LAYERS_FULL; 2],
    )
    .expect("a tiling table");
    assert!(
        !launches(&no_bit).iter().any(|k| k.contains("lora")),
        "and its absence does not"
    );
}

// ---------------------------------------------------------------------------
// The SEMANTIC GDN fragment, read against the Metal table.
// ---------------------------------------------------------------------------

/// What a `qwen3_5_hybrid_metal` would have to state for itself.
///
/// `LEDGER_QWEN35_DECODE` above is empty, and that is a fact about
/// `qwen3_5_hybrid_cuda` rather than about the family: that text names every
/// GDN kernel as a `Launch`, so the lowering never reads a semantic GDN kind
/// at all. The kinds are still there — `qwen3_5_gdn_block` is the same body
/// without the `Some(lower)` arms — and nobody had asked what they lower to.
///
/// This asks, and it asks against METAL, because that is the backend with
/// eight dark `gdn_*` rows waiting on the answer
/// (`driver-metal/tests/text_conformance.rs`'s `DARK`, whose reason on all
/// eight is "no plan in this workspace names a GDN symbol").
///
/// # Why the family is renamed
///
/// `lower` picks its kernel table from the plan's family name —
/// `Backend::of_family` reads the segment after the first `.` and knows
/// `cuda` and `metal`. The fragment is deliberately backend-free, so it has
/// no such segment and lowering refuses it outright with `UnknownBackend`.
/// Renaming is therefore not a trick around the check; it is how a
/// backend-free trace is asked a backend question at all, and it is the same
/// question a real `qwen3_5_hybrid_metal` would ask by being named that way.
///
/// # Two ledgers, because there are two ways to have no kernel
///
/// A kind can have no rule (`residue` — `semantic()` falls through), or it
/// can have a rule that names a symbol **this backend's table does not
/// declare**. The second is invisible to `lower`: `emit_bound` looks the
/// symbol up only to enforce the `whole` rule and treats a miss as "nothing
/// to enforce", so a Metal fire can be lowered to CUDA symbol names without
/// a word. `driver-metal`'s `every_symbol_every_text_states_has_a_row_...`
/// catches it for the texts that crate ships; nothing catches it here, which
/// is exactly why the second list is measured beside the first.
#[test]
fn the_semantic_gdn_fragment_states_what_metal_cannot_serve() {
    let facts = model::qwen_3_5::forward::facts::Qwen35GdnFacts::qwen3_5_0_8b();
    let mut plan = model::qwen_3_5::forward::qwen3_5_gdn_block(&facts);
    plan.family = format!("{}.metal", plan.family);
    let out = lower(&plan, &sampled(4), Fire::default()).expect("the fragment lowers");

    let mut unlowered: std::collections::BTreeMap<String, usize> = Default::default();
    for u in &out.residue {
        *unlowered
            .entry(format!("{}: {}", u.kind, u.why))
            .or_default() += 1;
    }
    let seen: Vec<String> = unlowered
        .iter()
        .map(|(k, n)| format!("{n:>4}  {k}"))
        .collect();
    let want: Vec<String> = GDN_FRAGMENT_NO_RULE.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        seen, want,
        "the GDN fragment's UNLOWERED ledger moved. A line leaving it means \
         `semantic()` learned a GDN kind; a line arriving means the fragment \
         grew a statement no backend-neutral rule can read."
    );

    let undeclared: Vec<String> = out
        .kernels
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .filter(|s| model_ir::kernels::stated_in(Backend::Metal, s).is_none())
        .map(String::from)
        .collect();
    let want: Vec<String> = GDN_FRAGMENT_NOT_IN_METAL
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        undeclared, want,
        "the GDN fragment's UNDECLARED ledger moved. Every line is a symbol \
         `semantic()` names for a Metal fire that Metal does not state, from \
         either plane -- a launch that would reach `check_plan` only if some \
         text stated it, and reach a device never."
    );
}

/// Kinds of [`the_semantic_gdn_fragment_states_what_metal_cannot_serve`]
/// that `semantic()` has no rule for at all.
const GDN_FRAGMENT_NO_RULE: &[&str] = &[
    "   1  CausalConv1d: no lowering rule for this kind",
    "   1  GatedDelta: no lowering rule for this kind",
];

/// Symbols that test's fragment lowers to which Metal does not state.
const GDN_FRAGMENT_NOT_IN_METAL: &[&str] = &[
    // The GEMM, the two norms and the prep -- CUDA spellings, every one,
    // and NOT because the fragment asked for CUDA. `semantic()`'s own doc
    // says where its list comes from ("read off the executor that launches
    // them today, `driver-cuda/csrc/.../declared_forward.cpp`"), and
    // nothing downstream re-reads it per backend. So this list is not a
    // fact about gated DeltaNet; it is what ANY Metal text would get the
    // moment it left one op semantic, and the reason every Metal text in
    // this workspace states all of its launches.
    "gemm::act_x_w",
    // D2's accumulating twin (`.wiki/kilimanjaro.md` §5). `act_x_w` took an
    // `Option<&str>` accumulator and the two behaviours it selected are two
    // symbols now, so this ledger gains a line for the same reason the table
    // did -- and gains it HERE because neither twin is in `KERNELS_METAL`,
    // which is the property this list is about.
    "gemm::act_x_w_acc",
    "norm::rmsnorm_gated_fp32_in_bf16",
    "norm::rmsnorm_gemma_bf16",
    "ssm::qwen_gdn_post_conv_prep_bf16",
];

/// The workspace root, from this test binary's own manifest.
fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/model has a workspace above it")
        .to_path_buf()
}

/// Every symbol `semantic()` can name, and which backend tables declare it.
///
/// [`the_semantic_gdn_fragment_states_what_metal_cannot_serve`] found four
/// symbols the Metal table does not carry, and the obvious reading -- "Metal
/// is behind" -- is wrong. Three of those four are not in the CUDA table
/// either. `sig_in` answering `None` does not mean "this backend cannot do
/// it"; it means "this TABLE states no contract for that name", and the CUDA
/// table is mid-migration (`kernels-cuda`, the JIT cutover) and
/// legitimately partial.
///
/// So the useful artefact is not a refusal, it is a census: for anyone
/// writing the first Metal text that leaves an op semantic, exactly which
/// arms of `semantic()` are already backed on their backend and which are
/// not. Both columns, side by side, so neither can be read as a fact about
/// the other.
///
/// # Why this parses the source
///
/// `semantic` is a private function of `OpKind`, and `OpKind`'s variants
/// carry payloads, so there is no value-level way to ask it for its whole
/// range. The symbols are string literals in one match, and reading them out
/// of it is the same move `driver-metal`'s `declared_buffers` makes against a
/// shader and `every_workflow_feature_names_a_feature_its_package_declares`
/// makes against YAML: parse the file that KNOWS, rather than keep a second
/// copy that can disagree. `the_census_reads_the_function_it_claims_to`
/// below is the check that the parse still finds anything at all.
#[test]
fn the_semantic_rules_name_symbols_each_backend_declares_or_does_not() {
    /// The leader column, wide enough for the longest symbol plus a gap.
    const WIDTH: usize = 44;
    let mut seen: Vec<String> = Vec::new();
    for symbol in semantic_symbols() {
        let cuda = model_ir::kernels::sig(&symbol).is_some();
        let metal = model_ir::kernels::stated_in(Backend::Metal, &symbol).is_some();
        // Dot leaders and not space padding: a run of six or more spaces
        // inside a string literal is what `no_message_carries_a_swallowed_
        // indent` looks for, and it is right to -- it cannot tell an aligned
        // COLUMN from a source line joined without dropping its indent. A
        // leader aligns and says which it is.
        let mark = |b: bool| if b { "yes" } else { "-" };
        let lead = ".".repeat(WIDTH.saturating_sub(symbol.len() + 1));
        seen.push(format!(
            "{symbol} {lead} cuda {} metal {}",
            mark(cuda),
            mark(metal)
        ));
    }
    let want: Vec<String> = SEMANTIC_CENSUS.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        seen, want,
        "the semantic-rule census moved. A `-` becoming `yes` is a table \
         gaining a contract; a `yes` becoming `-` is a table LOSING one, \
         which for CUDA means a row was deleted while a semantic rule still \
         names it, and for Metal means the same. A new line is a new arm in \
         `semantic()`."
    );
}

/// The string literals inside `semantic()`'s `Semantic::Kernels(&[..])`
/// arms, in source order, deduplicated.
fn semantic_symbols() -> Vec<String> {
    // `lower.rs` is a DIRECTORY since the lowering was split by what a reader
    // asks for, and the semantic rules are the file this scan is about. The
    // markers below are unchanged: `semantic()` is still spelled the same way
    // and `kind_name` still follows it, which is what the scan actually pins.
    let src = std::fs::read_to_string(
        workspace_root().join("crates/model-compiler/src/lower/semantics.rs"),
    )
    .expect("lower/semantics.rs is readable");
    let body = {
        let from = src
            .find("fn semantic(kind: &OpKind")
            .expect("`semantic` is still spelled that way");
        let to = src[from..]
            .find("\n/// The kind's name")
            .expect("`kind_name` still follows `semantic`");
        &src[from..from + to]
    };
    let mut out: Vec<String> = Vec::new();
    // Only the literals INSIDE a `Kernels(&[..])` list: the function's prose
    // names files and kernels too, and a comment is not a rule.
    for (n, line) in body.lines().enumerate() {
        let code = line.trim_start();
        if code.starts_with("//") {
            continue;
        }
        let _ = n;
        let mut rest = line;
        while let Some(i) = rest.find('"') {
            let after = &rest[i + 1..];
            let Some(j) = after.find('"') else { break };
            let lit = &after[..j];
            if lit.contains("::") && !out.iter().any(|s| s == lit) {
                out.push(lit.to_string());
            }
            rest = &after[j + 1..];
        }
    }
    out
}

/// The census of
/// [`the_semantic_rules_name_symbols_each_backend_declares_or_does_not`].
const SEMANTIC_CENSUS: &[&str] = &[
    // READ THE METAL COLUMN FIRST, and read it as one fact rather than
    // eighteen. It is not that Metal is behind on nine symbols and level on
    // none; it is that `semantic()` speaks CUDA's NAMESPACE and Metal's
    // census does not contain a single `module::name` spelling -- its symbols
    // are bare Metal entrypoints (`sdpa_paged_mma_sink`,
    // `affine_qmm_t_bfloat16_gs_64_...`). So no semantic arm can ever resolve
    // on Metal, and that is structural, not a gap someone forgot to fill.
    //
    // What follows from it is the useful part, and it is the answer to
    // `driver-metal`'s eight dark `gdn_*` stems: a `qwen3_5_hybrid_metal`
    // buys NOTHING from the semantic arms. Every op it wants has to be a
    // `Launch` naming a Metal symbol, exactly as `llama_like_metal` already
    // does for every op it has -- which is why that text has never tripped
    // over this and why the whole question was invisible.
    //
    // The CUDA column is the other half of the same care. ONE of the
    // eighteen is absent there now, and that is NOT the same phenomenon:
    // that spelling is in CUDA's namespace and simply has no row yet,
    // because `kernels-cuda` is mid-JIT-cutover and its table is
    // partial by construction. A `-` in this column is a row that has not
    // arrived; a `-` in the Metal column is a row that cannot.
    //
    // FOUR ARRIVED WHEN THE JIT BRANCH LANDED: `layout::embed_bf16`,
    // `norm::add_bias_bf16`, `norm::rmsnorm_gated_fp32_in_bf16` and
    // `attn::split_qkv_bf16_devwin` read `-` when this census was written
    // and read `yes` now. That is the direction the assertion calls "a table
    // gaining a contract", and it is the cutover doing exactly what the
    // paragraph above predicts, so the four are updated rather than
    // exempted. The Metal column did not move and could not have.
    //
    // AND FOUR MORE HAVE JUST ARRIVED, for a reason worth separating from
    // that one. `attn::split_qkv_bf16`, `layout::split_q_gate_bf16`,
    // `mlp::sigmoid_gate_inplace_bf16` and `ssm::qwen_gdn_post_conv_prep_bf16`
    // did not get a kernel or a port -- their host programs have been in
    // `kernels_cuda::driver_internal` all along, and that module DECLARED
    // nothing, so four symbols these very arms name had no row on the backend
    // every one of them is spelled for. They are `untraced!` lines in
    // `attn`, `layout`, `mlp` and `ssm` now. Nothing about the lowering
    // changed; what changed is that the census can no longer be read as "CUDA
    // cannot do these", which is what four `-`s in a table headed "declares or
    // does not" invited.
    //
    // A ROW IS NOT AN ARM. All four still refuse at the fire with `NoArm` --
    // `driver-cuda/tests/executor_bind.rs` is where that half is recorded --
    // and this census has never measured arms. `sig_in` answering `Some` is
    // what it asks and all it asks.
    "layout::embed_bf16 ......................... cuda yes metal -",
    "norm::add_bias_bf16 ........................ cuda yes metal -",
    "norm::residual_add_bf16 .................... cuda yes metal -",
    "ssm::qwen_gdn_post_conv_prep_bf16 .......... cuda yes metal -",
    "norm::rmsnorm_gated_fp32_in_bf16 ........... cuda yes metal -",
    "layout::split_q_gate_bf16 .................. cuda yes metal -",
    "mlp::sigmoid_gate_inplace_bf16 ............. cuda yes metal -",
    "norm::rmsnorm_bf16 ......................... cuda yes metal -",
    "norm::rmsnorm_gemma_bf16 ................... cuda yes metal -",
    "attn::split_qkv_bf16_devwin ................ cuda yes metal -",
    "attn::split_qkv_bf16 ....................... cuda yes metal -",
    "rope::rope_partial_bf16 .................... cuda yes metal -",
    "rope::rope_bf16 ............................ cuda yes metal -",
    // `cuda yes`, AND IT WAS ALREADY: both are `#[routine]`s in
    // `kernels-cuda/src/gemm`, and both had a `routine!` row before that.
    // The `-` here was a census taken when the pair had no row and never
    // retaken — the kind of staleness this list exists to catch, caught on
    // itself the first time the tree compiled well enough to run it.
    "gemm::act_x_w_acc .......................... cuda yes metal -",
    "gemm::act_x_w .............................. cuda yes metal -",
    "moe::moe_grouped_gemm_bf16 ................. cuda yes metal -",
    "moe::topk_softmax_bf16 ..................... cuda yes metal -",
    "moe::token_batched_weighted_sum_bf16 ....... cuda yes metal -",
    "mlp::sigmoid_dot_scalar_gate_add_bf16 ...... cuda yes metal -",
];

/// The census's own control: the parse finds a plausible number of symbols
/// and finds one it can name.
///
/// A source scan that silently matched nothing would make the census above
/// pass with an empty list forever, which is the failure mode every parsing
/// gate has. Two numbers rather than one: an exact symbol proves the shape
/// is right, and a floor proves the walk did not stop at it.
#[test]
fn the_census_reads_the_function_it_claims_to() {
    let found = semantic_symbols();
    assert!(
        found.len() >= 12,
        "the scan of `semantic()` found {} symbols, which is fewer than the \
         arms that function visibly has -- the parse has probably lost its \
         anchor. Found: {found:?}",
        found.len()
    );
    assert!(
        found.iter().any(|s| s == "norm::rmsnorm_bf16"),
        "the scan did not find `norm::rmsnorm_bf16`, which `semantic()` \
         names for a plain `Rmsnorm`. Found: {found:?}"
    );
}

/// The two namespaces do not overlap, and could not.
///
/// The census above says so in a comment, and a comment is checked by
/// nothing -- which is the failure this workspace keeps rediscovering (the
/// seven kernel rows that named `quantized_qmm_t.metal` years after that
/// file became `quant/qmm_t.metal`; the `--features forward` that four CI
/// steps named after the feature was deleted). So the claim is a test.
///
/// Both directions, because either one alone is a coincidence: every symbol
/// `semantic()` names is `module::name`, and NO symbol Metal can dispatch
/// is. Together they say a semantic arm cannot resolve on Metal for a
/// reason no amount of filling in rows would change -- someone would have
/// to make `semantic` take a backend, and THAT is the change the eight dark
/// `gdn_*` stems are actually waiting on if a Metal text ever wants to leave
/// an op semantic.
///
/// The Metal half asked `Backend::Metal.table()` until every Metal family
/// retired its rows, at which point the table answered nothing and the
/// control below -- the one that keeps "disjoint" from being a statement
/// about the empty set -- went red. It went red rather than quiet because
/// the control was there, which is the only reason this is a correction and
/// not a silently weakened test. The namespace outlived the rows: it is the
/// CENSUS, every entrypoint the backend ships, which is both the set a text
/// may name and a wider set than the rows ever were.
#[test]
fn the_semantic_namespace_and_the_metal_namespace_are_disjoint_by_shape() {
    let unqualified: Vec<String> = semantic_symbols()
        .into_iter()
        .filter(|s| !s.contains("::"))
        .collect();
    assert!(
        unqualified.is_empty(),
        "`semantic()` names {} symbol(s) with no `module::` qualifier, so the \
         census's claim that its namespace is CUDA's no longer holds by \
         shape: {unqualified:?}",
        unqualified.len()
    );

    let metal = model_ir::kernels::metal_entrypoints();
    let qualified: Vec<&String> = metal.iter().filter(|s| s.contains("::")).collect();
    assert!(
        qualified.is_empty(),
        "{} Metal entrypoint(s) now spell a `module::name` symbol, so the two \
         namespaces have started to overlap and a semantic arm could resolve \
         on Metal by accident: {qualified:?}",
        qualified.len()
    );

    // A control on both halves: the sets are not empty, so "disjoint" is a
    // statement about two populated namespaces rather than about nothing.
    // This is the assertion that caught the retirement.
    assert!(!semantic_symbols().is_empty() && !metal.is_empty());
}
