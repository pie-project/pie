//! Lowering, checked against real declarations.
//!
//! Moved out of `model-compiler` with `kernels_table.rs` and for the same
//! reason: these lower an actual family's traced form, and a dev-dependency
//! cycle back onto `model` gives the test a second, incompatible copy of the
//! toolchain's types.

use model_compiler::lower::*;
use model_compiler::trace::{ForwardPlan, OpKind, PeelWindow, ValueId};
use std::ops::Range;
use model::families::llama_like::forward::facts::LlamaLikeCudaFacts;
use model::families::llama_like::forward::facts::LlamaLikeFacts;
use model_compiler::trace::FireClass;

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
    model::families::llama_like::forward::llama_like_cuda(
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
                model::families::llama_like::forward::llama_like_cuda(&facts, &cuda, class),
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
        *ledger
            .entry(format!("{}: {}", u.kind, u.why))
            .or_default() += 1;
    }
    let seen: Vec<String> = ledger
        .iter()
        .map(|(k, n)| format!("{n:>4}  {k}"))
        .collect();
    let expected: Vec<String> = LEDGER_QWEN35_DECODE.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        seen,
        expected,
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
        ("0.8b", model::qwen_3_5::forward::facts::Qwen35HybridFacts::qwen3_5_0_8b()),
        ("27b", model::qwen_3_5::forward::facts::Qwen35HybridFacts::qwen3_6_27b()),
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
            *ledger
                .entry(format!("{}: {}", u.kind, u.why))
                .or_default() += 1;
        }
        let seen: Vec<String> = ledger
            .iter()
            .map(|(k, n)| format!("{n:>4}  {k}"))
            .collect();
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
        *ledger
            .entry(format!("{}: {}", u.kind, u.why))
            .or_default() += 1;
    }
    let seen: Vec<String> = ledger
        .iter()
        .map(|(k, n)| format!("{n:>4}  {k}"))
        .collect();
    let expected: Vec<String> = LEDGER_GPT_OSS_DECODE.iter().map(|s| s.to_string()).collect();
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
        ("e4b", model::gemma_4::forward::facts::Gemma4Facts::gemma_4_e4b()),
        ("e2b", model::gemma_4::forward::facts::Gemma4Facts::gemma_4_e2b()),
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
/// This started as a ledger (88.7%-93.8%, residue `Swiglu` per layer
/// + `LmHead` per fire) and is now the gate itself: `launches` is
/// the WHOLE of what a fire runs, which is the property the driver
/// needs before it can stop walking. A regression here is a
/// statement that would silently not execute.
#[test]
fn the_flat_list_covers_every_statement() {
    for (name, plan) in live_plans() {
        // Both epilogue shapes: a decode fire samples every row, a
        // prefill fire samples one row per request and gathers.
        for (shape, rows) in [("all-sampled", sampled(4)), ("gathered", gathered(4))] {
            let out = lower(&plan, &rows, Fire::default()).unwrap_or_else(|e| panic!("{name}/{shape}: {e:?}"));
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
    assert!(!sites.is_empty(), "the class trace carries observation sites");

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
        assert!(!out.structural.is_empty(), "a live fire brackets its layers");
        assert!(
            out.structural
                .iter()
                .all(|s| sites.contains(&(s.at_op as usize))),
            "only sites are structural"
        );
        // Ordered, because a bracket opens before it closes.
        assert!(out
            .structural
            .windows(2)
            .all(|w| w[0].at_op < w[1].at_op));
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
    assert!(host.launches.iter().all(|l| l.peel.is_none_or(|p| !p.rows_device)));
    assert!(
        !host
            .launches
            .iter()
            .any(|l| l.kernel_is(&host, "launch_qkv_decode_qk_norm_rope_write_kv_bf16")),
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
        .filter(|l| l.kernel_is(&captured, "launch_qkv_decode_qk_norm_rope_write_kv_bf16"))
        .collect();
    assert!(!fused.is_empty(), "the captured graph carries the prefix");
    assert!(fused
        .iter()
        .all(|l| l.peel.is_some_and(|p| p.axis == PeelWindow::HookFreePrefix
            && p.rows_device)));
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
    assert!(captured
        .launches
        .iter()
        .filter(|l| l.peel.is_some_and(|p| p.rows_device))
        .count()
        < captured.launches.len());
}

/// The epilogue is three statements over a ROW COUNT, and the two
/// runtime branches the executor takes are the count being zero and
/// the count being short.
#[test]
fn the_epilogue_is_a_row_count_not_a_branch() {
    let plan = decode_plan();
    // The epilogue's launches are the ones carrying the LmHead
    // statement's index. Identifying them by SYMBOL would not work:
    // its projection is `gemm_act_x_w`, the same launcher every
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
            .filter(|l| l.args == at_op)
            .map(|l| (out.kernels[l.kernel as usize].clone(), l.rows.clone()))
            .collect()
    };

    // Every row sampled: norm and project over all four rows, no
    // gather — there is nothing to skip past.
    let all = epilogue(&sampled(4));
    assert_eq!(
        all,
        vec![
            ("launch_rmsnorm_bf16".to_string(), 0..4),
            ("gemm_act_x_w".to_string(), 0..4),
        ]
    );

    // One sampled row of four: the gather appears, and all three
    // statements run over ONE row while the body ran over four —
    // the epilogue's row space is Requests.
    assert_eq!(
        epilogue(&gathered(4)),
        vec![
            ("launch_gather_bf16_rows".to_string(), 0..1),
            ("launch_rmsnorm_bf16".to_string(), 0..1),
            ("gemm_act_x_w".to_string(), 0..1),
        ]
    );

    // Nothing sampled (`emit_logits == false`, a fire whose logits
    // nobody reads): no rectangle at all, while the body still runs.
    let none = vec![Row::default(); 4];
    assert!(epilogue(&none).is_empty());
    assert!(!lower(&plan, &none, Fire::default()).unwrap().launches.is_empty());
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
    assert!(out.kernels.contains(&"dispatch_attention_flashinfer_decode".to_string()));
    // Every launch names a layer the trace tagged.
    assert!(out.launches.iter().all(|l| l.layers.end == l.layers.start + 1));
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
    let masked = out
        .launches
        .iter()
        .filter(|l| l.rows == (6..8))
        .count();
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
    let plan = model::families::llama_like::forward::llama_like_cuda(&facts, &cuda, FireClass::Decode);
    assert!(
        plan.ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Launch { kernel, .. }
                if kernel == "launch_attention_xqa_decode_bf16_prepared"
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
    assert!(buffers
        .pinned
        .iter()
        .all(|&v| buffers.offset[v as usize] == Buffers::NAMED));
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
    assert!(out.launches.iter().all(|l| l.layers.start < 12
        || l.layers.start >= 28
        || l.rows.is_empty()));
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
    let facts = model::glm5::forward::facts::Glm5Facts::glm5_106b_a12b();
    let plan = model::glm5::forward::glm5_cuda(&facts, FireClass::Decode);
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
