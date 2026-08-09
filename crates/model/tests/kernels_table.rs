//! The kernel signature tables, checked against the declarations that state
//! them.
//!
//! These were `model-compiler`'s own unit tests and could not stay there. Half
//! of them trace a REAL family to see that every symbol it launches is
//! declared, and the families are here now; a dev-dependency back on `model`
//! looked like it would work and does not — cargo builds `model_compiler`
//! twice in a dependency cycle, so `OpKind` from the crate under test is a
//! different type from `OpKind` in the plan `model` hands back.
//!
//! Being an integration test costs nothing here: `check_plan`, `Backend` and
//! `sig_in` are all public, because a driver-side consumer reads them too.

use model::qwen_3_5::forward::facts::Qwen35CudaFacts;
use model::qwen_3_5::forward::facts::Qwen35HybridFacts;
use model::shared::llama_like::forward::facts::LlamaLikeCudaFacts;
use model::shared::llama_like::forward::facts::LlamaLikeFacts;
use model_compiler::kernels::*;
use model_compiler::trace::FireClass;
use model_compiler::trace::ForwardPlan;

use model_compiler::trace::{Op, OpKind};

fn launch(symbol: &str) -> Op {
    Op {
        kind: OpKind::Launch {
            kernel: symbol.to_string(),
            weights: vec![],
            state: None,
            params: vec![],
        },
        inputs: vec![],
        outputs: vec![],
        layer: Some(0),
    }
}

fn plan_of(ops: Vec<Op>) -> ForwardPlan {
    ForwardPlan {
        // A family name that says whose kernels these are — the
        // check resolves the table from it.
        family: "llama_like.cuda.decode".to_string(),
        values: vec![],
        ops,
        depth_window: false,
        seams: vec![],
    }
}

/// The rules FIRE — without this, a green `live_traces_satisfy_the_table`
/// proves nothing but that the walk found no launches.
#[test]
fn the_check_is_not_vacuous() {
    // An undeclared symbol.
    let problems = check_plan(&plan_of(vec![launch("launch_something_new")]));
    assert_eq!(problems.len(), 1, "{problems:#?}");
    assert!(problems[0].contains("launch_something_new"));

    // A `whole` kernel given a row window by a Peel.
    let peel = Op {
        kind: OpKind::Peel {
            prefix_ops: 1,
            tail_ops: 1,
            window: model_compiler::trace::PeelWindow::HookFreePrefix,
        },
        inputs: vec![],
        outputs: vec![],
        layer: Some(0),
    };
    let xqa = "attn::attention_xqa_decode_bf16_prepared";
    let problems = check_plan(&plan_of(vec![
        peel,
        launch(xqa),
        launch("attn::dispatch_attention_flashinfer_decode"),
    ]));
    assert_eq!(problems.len(), 1, "{problems:#?}");
    assert!(problems[0].contains("whole"), "{}", problems[0]);

    // The same kernel OUTSIDE a peel is fine — the fire-level
    // statement the model body takes today.
    assert!(check_plan(&plan_of(vec![launch(xqa)])).is_empty());
}

/// A family name says whose kernels a text states.
#[test]
fn the_backend_is_read_off_the_family() {
    assert_eq!(
        Backend::of_family("llama_like.cuda.decode"),
        Some(Backend::Cuda)
    );
    assert_eq!(
        Backend::of_family("qwen3_5_hybrid.cuda.commit_advance"),
        Some(Backend::Cuda)
    );
    assert_eq!(
        Backend::of_family("llama_like.metal.decode"),
        Some(Backend::Metal)
    );
    // Semantic traces state no kernels, so no table applies.
    assert_eq!(Backend::of_family("llama_like"), None);
    assert_eq!(Backend::of_family("qwen3_5_moe_mlp_block"), None);
}

/// The backend table is a GATE, not a wall: it admits exactly the symbols
/// it declares and refuses everything else, so a `llama_like.metal.*` text
/// cannot state a kernel nobody wrote a row for.
///
/// Metal's table holds only the rows a first such text would need, so most
/// of the MSL entrypoints `decode_psos.cpp` compiles are still undeclared.
/// That is safe precisely because of the refusal half: an undeclared symbol
/// fails the trace at load rather than silently resolving to nothing.
#[test]
fn the_metal_table_admits_its_rows_and_refuses_the_rest() {
    let mut p = plan_of(vec![launch("metal_gemm_bf16")]);
    p.family = "llama_like.metal.decode".to_string();
    // (the same symbol under CUDA's table is refused too — this is
    // about WHICH table, not about permissiveness)
    let problems = check_plan(&p);
    assert_eq!(problems.len(), 1, "{problems:#?}");
    assert!(problems[0].contains("metal"), "{}", problems[0]);

    // And the other half, without which the above would also pass on a table
    // that refuses everything: a declared entrypoint goes through.
    //
    // Spelled from `entrypoints()` rather than from `symbol`, because a Metal
    // row's symbol is a BASE and a base is not something a text can launch —
    // every point of every axis contributes text, so `attn_gate` names a
    // kernel and `attn_gate_bfloat16` names the dispatch.
    let declared = KERNELS_METAL
        .first()
        .expect("Metal's table declares at least one kernel")
        .entrypoints();
    let declared = declared.first().expect("and that kernel has an entrypoint");
    let mut ok = plan_of(vec![launch(declared)]);
    ok.family = "llama_like.metal.decode".to_string();
    assert_eq!(check_plan(&ok), Vec::<String>::new());
}

/// Rows no `dsl::cuda` text states, and why each one is in the table.
///
/// This list is the seam between the table's TWO jobs, which stopped being
/// one job when the ABI pilot landed. The compiler's job is to plan against
/// symbols a declaration can record; `driver-cuda`'s
/// `every_launcher_the_header_declares_has_a_row` gives the table a second
/// one — being the operand contract for every launcher a HEADER declares,
/// whether a declaration reaches it or not. A row can now be real and
/// unstated, so the invariant below is a containment plus this pinned
/// remainder rather than an equality.
///
/// Sorted, because it is compared against a sorted difference.
/// SORTED, because the assertion compares against a sorted remainder.
const UNSTATED_ROWS: &[&str] = &[
    // The collectives came OUT (3). `dist::` and `comm::` joined the
    // prefix scan above, which is what this test measures: a symbol a
    // `dsl::cuda` wrapper RECORDS. Whether a model text calls one is a
    // different question, and the goldens are where it is answered --
    // `mistral_7b_v03.cuda.tp2.decode` is llama_like's sharded trace,
    // and it fires both all-reduce spellings 32 times each.
    //
    // Two remain recorded-but-uncalled, and neither has an entry here
    // because the scan cannot tell: `comm::all_reduce_residual_rmsnorm_bf16`
    // (the fused landing, waiting on a guard whose arms produce a PAIR)
    // and `dist::all_gather_bf16` (no text gathers; column-parallel
    // outputs here are consumed shard-local).
    // The epilogue's row gather has no `dsl::cuda` wrapper because no
    // model text states it: `lower::epilogue` emits it when the fire
    // samples fewer rows than it computes (a prefill reads one
    // distribution per request out of a stream of one row per token).
    // A statement the LOWERING makes is real without a text stating it,
    // which is precisely what this list is for.
    "layout::gather_bf16_rows",
    // The LOADER's two quantizers, called from `loader/arena.rs` rather
    // than recorded by any forward text: a weight transform runs once at
    // load and never appears in a fire's op list. Real without a text
    // stating it, which is what this list is for.
    "quant::quantize_bf16_to_fp8_e4m3_per_channel",
    "quant::quantize_bf16_to_mxfp4_e2m1_per_block",
    "rope::rope_partial_bf16_position_delta",
];

/// The table covers every symbol `dsl::cuda` can record.
///
/// This is the argument that [`check_plan`]'s coverage rule — which
/// runs at LOAD and fails the trace — can never fire spuriously on a
/// live deployment: reachability is a property of the dsl surface,
/// not of which fact combinations a test happens to exercise. And it
/// is the guard that makes the table's other three declarations get
/// written: a new `cuda::` wrapper fails this test until its
/// contract exists.
///
/// The containment direction is the load-bearing one and takes no
/// exceptions. The reverse is pinned to [`UNSTATED_ROWS`] rather than
/// asserted empty, for the reason that list gives — and it still fires on
/// a new wrapper, which lands in the remainder until its author either
/// states it or names it there with a reason. This is the same shape the
/// Metal table has carried all along (see
/// `the_metal_table_admits_its_rows_and_refuses_the_rest`): declared ⊇
/// stated is safe precisely because of the refusal half, since a symbol
/// nothing states is a symbol nothing can reach.
#[test]
fn the_table_covers_the_dsl_surface() {
    let dsl = include_str!("../../model-compiler/src/dsl.rs");
    let mut stated: Vec<&str> = dsl
        .split('"')
        .skip(1)
        .step_by(2)
        .filter(|s| {
            // The prefixes a kernel symbol can start with. `ops::` and
            // `marlin_moe::` are C++ NAMESPACES the symbol genuinely carries
            // -- the launcher lives in the vendored tree, and the table
            // records the name a caller writes.
            //
            // This list is a GUESS about naming, and it has been wrong twice:
            // once when `marlin_moe::` arrived, and again when
            // `scripts/kernel-vocabulary-audit.py` found seventeen launchers
            // named none of these ways (`mla_absorb_*`,
            // `merge_attention_states_*`, `gemm_*`) by reading the HEADERS
            // instead of guessing at prefixes.
            //
            // It stays a list because the principled alternative -- reading
            // the symbol out of `record`'s argument slot -- needs a parser to
            // tell a symbol from an `.expect` message, and got that wrong
            // when tried. So the division is: this test pins table<->dsl
            // drift cheaply, and the audit script is the exhaustive check.
            // Run the script when adding a family; this list alone will not
            // tell you what is missing.
            [
                "launch_",
                "dispatch_",
                "ops::",
                "marlin_moe::",
                "gemm_",
                // `mla_absorb_`, not `mla_`: an `.expect` message reading
                // "mla_prepare states four outputs" matched the shorter one.
                // Every prefix here is as long as the symbols it must admit
                // and no longer.
                "mla_absorb_",
                "merge_",
                "flashinfer_",
                "pie_lora",
                "qwen35_verify",
                // One line per family as step 3 lands; when the last
                // `launch_` is gone the first five entries can go too.
                "rope::",
                "gemm::",
                "attn::",
                "moe::",
                "quant::",
                "layout::",
                "norm::",
                "ssm::",
                "mlp::",
                "sample::",
                // The COLLECTIVES' namespaces. Their absence here was a
                // hole in the coverage rule rather than a fact about
                // them: `dsl::cuda::all_reduce` and friends record
                // these symbols like any other, and without the prefix
                // the scan simply could not see them.
                "dist::",
                "comm::",
            ]
            .iter()
            .any(|p| s.starts_with(p))
        })
        .collect();
    stated.sort_unstable();
    stated.dedup();
    let mut declared: Vec<&str> = KERNELS.iter().map(|k| k.symbol).collect();
    declared.sort_unstable();

    let unbacked: Vec<&str> = stated
        .iter()
        .filter(|s| !declared.contains(s))
        .copied()
        .collect();
    assert!(
        unbacked.is_empty(),
        "dsl::cuda records symbols the kernel! table does not declare, so \
         `check_plan` would refuse them at LOAD: {unbacked:?}"
    );

    let unstated: Vec<&str> = declared
        .iter()
        .filter(|d| !stated.contains(d))
        .copied()
        .collect();
    assert_eq!(
        unstated, UNSTATED_ROWS,
        "the table's rows that no dsl::cuda text states have changed. A row \
         that arrived here needs either a text that states it or an entry in \
         `UNSTATED_ROWS` saying why it is real without one; a row that left \
         needs its entry deleted"
    );
}

/// The retired `DepthRole`'s two facts, DERIVED, on a live
/// depth-declaring trace: membership is the layer tag, and exactly
/// one launch per layer swaps to the prefix plan.
///
/// The wire word `ffi::arena` writes is computed from these two, so
/// this pins the C ABI's `depth_role` byte without the IR carrying
/// it. (The one-off proof that the derivation reproduced the stored
/// word was 11,399 ops across all 23 goldens, zero mismatches.)
#[test]
fn the_depth_axis_derives_from_the_layer_tag() {
    let facts = LlamaLikeFacts::qwen3_0_6b();
    let plan = model::shared::llama_like::forward::llama_like_cuda(
        &facts,
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        FireClass::Decode,
    );
    assert!(plan.depth_window, "this deployment declares the axis");

    let windowed = plan.ops.iter().filter(|op| plan.depth_windowed(op)).count();
    let layered = plan.ops.iter().filter(|op| op.layer.is_some()).count();
    assert_eq!(windowed, layered, "every layer-tagged op is on the axis");
    assert!(
        plan.ops
            .iter()
            .all(|op| op.layer.is_some() || !plan.depth_windowed(op)),
        "the prologue/epilogue are outside it"
    );

    // Three planned-decode dispatches per layer take the swap: the
    // mask arm's unmasked-prefix rows, and the plain body's
    // score-capturing and plain arms.
    let swaps = plan
        .ops
        .iter()
        .filter(|op| plan.depth_prefix_plan(op))
        .count();
    assert_eq!(swaps, 3 * facts.layers as usize);
    assert!(
        plan.ops
            .iter()
            .filter(|op| plan.depth_prefix_plan(op))
            .all(|op| matches!(
                &op.kind,
                OpKind::Launch { kernel, .. }
                    if kernel == "attn::dispatch_attention_flashinfer_decode"
            )),
        "only the planned decode dispatch swaps"
    );

    // PREFILL declares the axis too (the cutover's last decline
    // class was a truncated prefill), and its layer-tagged ops are
    // on it — but NOTHING there takes the prefix-plan swap, because
    // that is a property of the planned DECODE dispatch and a
    // prefill fire does not run one. Which is the whole difference
    // between the two halves of the axis: stopping after layer `k`
    // costs a prefill nothing, and narrowing rows under it would
    // cost it a plan it has no way to build.
    let prefill = model::shared::llama_like::forward::llama_like_cuda(
        &facts,
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        FireClass::Prefill,
    );
    assert!(prefill.depth_window);
    assert!(prefill.ops.iter().any(|op| prefill.depth_windowed(op)));
    // Asked of the LOWERED form, not the traced one, and that is the
    // window-class merge (`.wiki/driver/graph.md` §4.1): the trace now
    // carries BOTH window classes as arms of a `GuardPred::WindowOne`
    // guard, so a prefill TRACE does contain the planned decode
    // dispatch — it is the arm this fire will not take. Which arm runs
    // is a lowering answer, and `Resolve` is where it is given.
    let prefill_rows = vec![
        model_compiler::lower::Row {
            multi_token: true,
            ..Default::default()
        };
        7
    ];
    let lowered = model_compiler::lower::lower_with(
        &prefill,
        &prefill_rows,
        model_compiler::lower::Fire::default(),
        model_compiler::lower::GuardMode::Resolve,
    )
    .expect("a prefill fire lowers");
    assert_eq!(
        lowered
            .launches
            .iter()
            .filter(|l| lowered.kernels[l.kernel as usize]
                == "attn::dispatch_attention_flashinfer_decode")
            .count(),
        0,
        "a prefill fire runs no planned decode dispatch, so nothing swaps"
    );

    // A PADDED-HEAD deployment declares the axis too. It cannot serve
    // the narrowing half — its staging offsets are physical width
    // while a row window's are logical — but stopping after layer `k`
    // addresses nothing, and `k` is a runtime input the trace does
    // not have. So the trace states the axis and the DRIVER refuses
    // the shapes that narrow (`PaddedHeadNarrowing`), which is the
    // same division of labour the Prefill class settled.
    let padded = model::shared::llama_like::forward::llama_like_cuda(
        &facts,
        &LlamaLikeCudaFacts {
            head_dim_padded: true,
            // SYNTHETIC: this fixture's model facts are qwen3-0.6B's
            // (head_dim 128), which pads nowhere. The width only has to
            // be wider than the logical one for the pad statements to
            // be well-formed; what the test is about is the AXIS.
            head_dim_kernel: 256,
            ..LlamaLikeCudaFacts::qwen3_0_6b_l40s()
        },
        FireClass::Prefill,
    );
    assert!(padded.depth_window);

    // The XQA decode deployment is the one that still withholds it:
    // its prepare is fire-wide and R-shaped, so even the free half
    // has nothing to stand on.
    let xqa = model::shared::llama_like::forward::llama_like_cuda(
        &facts,
        &LlamaLikeCudaFacts {
            xqa_decode: true,
            ..LlamaLikeCudaFacts::qwen3_0_6b_l40s()
        },
        FireClass::Decode,
    );
    assert!(!xqa.depth_window);
    assert!(xqa.ops.iter().all(|op| !xqa.depth_windowed(op)));
}

/// No symbol is declared twice, and no dsl-side name is either.
#[test]
fn table_is_unambiguous() {
    for (i, k) in KERNELS.iter().enumerate() {
        for other in &KERNELS[i + 1..] {
            assert_ne!(k.symbol, other.symbol, "symbol declared twice");
            assert_ne!(k.name, other.name, "name declared twice");
        }
    }
}

/// Every kernel every live deployment states is declared, and no
/// live trace puts a `whole` kernel under a row split. This is the
/// check running against real traces — the table is not decorative.
#[test]
fn live_traces_satisfy_the_table() {
    let mut plans = Vec::new();
    for class in [FireClass::Decode, FireClass::Prefill] {
        plans.push(model::shared::llama_like::forward::llama_like_cuda(
            &LlamaLikeFacts::qwen3_0_6b(),
            &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
            class,
        ));
        plans.push(model::shared::llama_like::forward::llama_like_cuda(
            &LlamaLikeFacts::mistral_7b_v03(),
            &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
            class,
        ));
    }
    for class in [FireClass::Decode, FireClass::Prefill] {
        plans.push(model::qwen_3_5::forward::qwen3_5_hybrid_cuda(
            &Qwen35HybridFacts::qwen3_5_0_8b(),
            &Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
            class,
        ));
        // Qwen3.6-27B: the same text at a different geometry, and
        // the first one whose GDN half is GQA.
        plans.push(model::qwen_3_5::forward::qwen3_5_hybrid_cuda(
            &Qwen35HybridFacts::qwen3_6_27b(),
            &Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
            class,
        ));
    }
    // gemma-4: every symbol its decode reading states has a
    // contract here.
    plans.push(model::gemma_4::forward::gemma4_cuda(
        &model::gemma_4::forward::facts::Gemma4Facts::gemma_4_e4b(),
        &model::gemma_4::forward::facts::Gemma4CudaFacts::gemma_4_e4b_synthetic(),
        FireClass::Decode,
    ));
    for plan in &plans {
        let problems = check_plan(plan);
        assert!(problems.is_empty(), "{problems:#?}");
    }
}

/// A quantized weight makes its statement name MORE tensors, and a
/// dense one names exactly what it did before.
///
/// The quantization axis lives on the weight handle
/// (`MatW::repr`), so `matmul(x, &w)` resolves to a stated symbol at
/// TRACE time and the scales ride as declared weights. This asserts the
/// two halves that matter: the dense path is untouched (every existing
/// golden depends on that), and each representation names a symbol the
/// `kernel!` table declares — which is what stops `check_plan` refusing
/// it at load.
#[test]
fn a_weight_representation_states_its_kernel() {
    use model_compiler::dsl::{MatW, ScaleLayout, WeightRepr};

    let dense = MatW::dense("layer.0.q_proj".into(), 2048, Some(0));
    assert_eq!(dense.gemm_symbol(), None, "a dense weight chooses nothing");
    assert!(dense.scale_names().is_empty());

    let cases = [
        (
            WeightRepr::Scaled {
                layout: ScaleLayout::PerGroup,
                group: 128,
                axis: 0,
                zero_point: true,
            },
            "gemm::act_x_wt_grouped_scaled",
            2,
        ),
        (
            WeightRepr::Scaled {
                layout: ScaleLayout::PerChannel,
                group: 0,
                axis: 0,
                zero_point: false,
            },
            "gemm::act_x_wt_channel_scaled",
            1,
        ),
        (WeightRepr::Mxfp4Marlin, "gemm::act_x_wt_mxfp4_marlin", 1),
    ];
    for (repr, symbol, extra) in cases {
        let w = dense.clone().with_repr(repr);
        assert_eq!(
            w.gemm_symbol(),
            Some(symbol),
            "{repr:?} must name the kernel that can read it"
        );
        assert_eq!(
            w.scale_names().len(),
            extra,
            "{repr:?} names its scales (and zero-points) as weights"
        );
        // The name the loader already looks for, derived off the
        // weight's own — not a second naming convention.
        assert!(w.scale_names()[0].starts_with("layer.0.q_proj."));
        assert!(
            sig_in(Backend::Cuda, symbol).is_some(),
            "{symbol} needs a kernel! row or `check_plan` refuses it at load"
        );
    }
}

/// Every kernel a SEMANTIC op kind can fan to has a row.
///
/// A semantic kind names no symbol, so the driver picks one — and the
/// table's coverage rule cannot see those picks, because `check_plan`
/// only walks `OpKind::Launch`. That is the hole this closes: a kernel
/// reachable only through a driver's fan has no operand contract
/// anywhere, and nothing notices.
///
/// It found exactly one pair when written — `norm::rmsnorm_bf16` and
/// `norm::rmsnorm_gemma_bf16`, the two `OpKind::Rmsnorm` chooses
/// between from its variant. Every other fan target is also stated by
/// some `dsl::cuda` wrapper, so it already had a row for that reason.
///
/// The list is written by hand because there is no machine-readable
/// link from a kind to the kernels its arms call; a kind that grows a
/// third spelling has to be added here, and that is the point — the
/// addition is where someone notices the driver is choosing.
#[test]
fn the_kernels_a_semantic_kind_fans_to_are_declared() {
    // (kind, the symbols its driver arms pick between)
    const FANS: &[(&str, &[&str])] = &[
        (
            "Rmsnorm",
            &["norm::rmsnorm_bf16", "norm::rmsnorm_gemma_bf16"],
        ),
        (
            "RmsnormPerHead",
            &["norm::rmsnorm_bf16", "norm::rmsnorm_gemma_bf16"],
        ),
        ("Rope", &["rope::rope_bf16", "rope::rope_partial_bf16"]),
        (
            "SplitGdn",
            &["layout::split_bf16_rows", "layout::split_qwen_gdn_ba_bf16"],
        ),
    ];
    let mut missing: Vec<String> = Vec::new();
    for (kind, symbols) in FANS {
        for s in *symbols {
            if sig_in(Backend::Cuda, s).is_none() {
                missing.push(format!("{kind} -> {s}"));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "a semantic kind fans to kernels with no `kernel!` row, so their \
         operand contract is written nowhere and `check_plan` cannot see \
         them (it walks Launch only): {missing:?}"
    );
}
