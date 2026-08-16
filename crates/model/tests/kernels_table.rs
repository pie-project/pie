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
//! Being an integration test costs nothing here: `check_plan`, `Backend`,
//! `sig` and `stated_in` are all public, because a driver-side consumer reads
//! them too.

use model::qwen_3_5::forward::facts::Qwen35CudaFacts;
use model::qwen_3_5::forward::facts::Qwen35HybridFacts;
use model::shared::llama_like::forward::facts::LlamaLikeCudaFacts;
use model::shared::llama_like::forward::facts::LlamaLikeFacts;
use model_ir::kernels::*;
use model_ir::trace::FireClass;
use model_ir::trace::ForwardPlan;

use model_ir::trace::{Op, OpKind};

/// A statement of `symbol` that places exactly what its routine binds.
///
/// The fixtures below are about the declaration rule and the `whole` rule, and
/// §6.2's arity rule fires on any statement that places nothing for a routine
/// that reads a pointer, so a bare `Op` would trip the wrong rule and bury the
/// one under test. An UNDECLARED symbol answers nothing and gets nothing: the
/// declaration rule refuses it before arity is reached.
fn launch(symbol: &str) -> Op {
    launch_on(Backend::Cuda, symbol)
}

/// [`launch`] against a named plane, for the fixtures that retag the family.
fn launch_on(backend: Backend, symbol: &str) -> Op {
    let (reads, writes) = stated_in(backend, symbol)
        .map_or((0, 0), |k| if k.args.is_empty() { (0, 0) } else { k.places() });
    Op {
        kind: OpKind::Launch {
            kernel: symbol.to_string(),
            weights: vec![],
            state: None,
            params: vec![],
            param_extents: vec![],
        },
        inputs: (0..reads as u32).collect(),
        outputs: (0..writes as u32).collect(),
        layer: Some(0),
        dest: Vec::new(),
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
            window: model_ir::trace::PeelWindow::HookFreePrefix,
        },
        inputs: vec![],
        outputs: vec![],
        layer: Some(0),
        dest: Vec::new(),
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

/// The backend table is a GATE, not a wall: it admits exactly the symbols it
/// declares and refuses everything else, so a `llama_like.metal.*` text cannot
/// state a kernel nobody wrote a row for. Metal's table holds only the rows a
/// first such text would need; that is safe precisely because of the refusal
/// half — an undeclared symbol fails the trace at load rather than silently
/// resolving to nothing.
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
    // that refuses everything: a declared entrypoint goes through. Spelled from
    // the CENSUS, not the table — Metal's table is empty now, every family
    // having crossed to a routine, and a row's symbol was a BASE, which is not
    // something a text can launch (`attn_gate` names a kernel,
    // `attn_gate_bfloat16` names the dispatch).
    let entrypoints = metal_entrypoints();
    let declared = entrypoints.first().expect("Metal's census names a kernel");
    let mut ok = plan_of(vec![launch_on(Backend::Metal, declared)]);
    ok.family = "llama_like.metal.decode".to_string();
    assert_eq!(check_plan(&ok), Vec::<String>::new());
}

/// Rows no `dsl::cuda` text states, and why each one is in the table.
///
/// The table has TWO jobs: planning against symbols a declaration can record,
/// and — since `driver-cuda`'s `every_launcher_the_header_declares_has_a_row`
/// — being the operand contract for every launcher a HEADER declares, stated
/// or not. A row can therefore be real and unstated, so the invariant below is
/// a containment plus this pinned remainder rather than an equality.
///
/// SORTED, because the assertion compares against a sorted remainder.
const UNSTATED_ROWS: &[&str] = &[
    // The collectives came out: `dist::` and `comm::` are in the prefix scan
    // now, so a `dsl::cuda` wrapper that RECORDS one is visible here. Two stay
    // recorded-but-uncalled without entries, because the scan cannot tell:
    // `comm::all_reduce_residual_rmsnorm_bf16` waits on a guard whose arms
    // produce a PAIR, and no text gathers `dist::all_gather_bf16`.

    // THE f16 TWINS (4). Each of `attn::logit_softcap_f16`,
    // `norm::residual_add_f16`, `norm::tanh_f16` and `rope::rope_partial_f16`
    // is the same body as a stated bf16 original over the other element type.
    // No text names one because nothing here stays fp16 across an op: the only
    // fp16 activations are MXFP4/marlin GEMM OPERANDS, cast in and out around
    // the call, so nothing is capped, rotated or residual-added in fp16.
    // Nothing calls the four `fn`s either — checked, not assumed. They are
    // contracts for kernels this crate compiles and this tree does not fire;
    // what a twin is missing is a deployment, not a caller.
    "attn::logit_softcap_f16",
    // AN INNER LEG IS NOT A STATEMENT (7). Seven symbols here are called by
    // ANOTHER routine in this crate rather than by anything holding an op
    // list: a text states the OUTER symbol, whose `fn` picks a shape or a
    // staging step on the host and calls the leg by path. The leg has a row
    // because its `fn` is in a family's `ROUTINES` and `sigs()` derives a row
    // from every line there — a compiled body has an operand contract whether
    // or not a text wants one. What it will never have is a statement: a
    // caller that has already chosen is not a declaration.
    //
    // This one is the fused decode QKV path's shared body, called with a null
    // window (read as "no window") by
    // `attn::qkv_decode_qk_norm_rope_write_kv_bf16`. The window parameter that
    // earns it a separate `fn` has no caller in this tree.
    "attn::qkv_decode_fused_dispatch",
    // The epilogue's row gather has no wrapper because no model text states
    // it: `lower::epilogue` emits it when the fire samples fewer rows than it
    // computes. A statement the LOWERING makes is real without a text, which
    // is what this list is for.
    //
    // SEVEN MORE OF THAT KIND: `lower::semantic` maps `OpKind::SplitQkv`,
    // `SplitQGate`, `SigmoidGateMul`, `GdnPrep`, `Embed`, `AddBias` and
    // `RmsnormGated` onto seven symbols, and those op kinds carry no kernel
    // string, so nothing records one and the scan cannot see them. `SplitQkv`
    // maps onto TWO because the peel's tail serves absolute row offsets in a
    // full-N buffer — the lowering states the pick rather than making the
    // driver derive it from a window pointer. Four of the seven are declared
    // by `driver_bound!` lines against host programs in
    // `kernels_cuda::driver_internal`, the other three by ordinary
    // `routine!`s; both spellings land in `declared` and neither can land in
    // `stated`. `bind/arms/` arms six of the seven for real, and the seventh,
    // `norm::rmsnorm_gated_fp32_in_bf16`, is listed there as `arm: None` with
    // a sentence saying what a `Cx` would have to grow to bind it.
    "attn::split_qkv_bf16",
    // The peel-tail half of `OpKind::SplitQkv`. See the block above.
    "attn::split_qkv_bf16_devwin",
    // §5 D5's TWO LEGS. `attn::write_kv_to_pages` was one `fn` branching on
    // `is_native_bf16`; a dispatcher is a map resolved once at load, so the
    // branch is now `kv_scheme::write_kv_to_pages(bool) -> &'static str` and
    // the two bodies are two symbols. Every model text still states the outer
    // name — that is the point of resolving at load — so the two it resolves
    // TO are stated by nobody.
    "attn::write_kv_to_pages_bf16",
    "attn::write_kv_to_pages_quantised",
    // `gemm::act_x_wt_bf16`'s `m == 1` leg, chosen from the shape on the host
    // and never by a text; the dense tuner's `GemmKind::Gemv` tactic is the
    // same call under timing. Note while reading this and the four `quant::`
    // legs below: none of the five is fired today, because no `gemm` matmul
    // symbol has an arm — `gemm::act_x_w` is `arm: None` with a reason that
    // `Route::refusal` turns into a load-time `Unfireable`. That is a gap in
    // the fire path, not a fact about this list.
    "gemm::gemv_bf16",
    // §54's FOUR DELETED WRAPPERS, BACK AS ROWS AND NOT AS DEMAND (4).
    // `copy_if_valid_slot`, `concat_rows`, `deinterleave_rows` and
    // `deinterleave_vec` existed because the dsl surface had been GENERATED
    // from launcher headers, and each had zero callers. `sigs()` derives a row
    // from every line of a family's `ROUTINES` and `layout.rs` still lists all
    // four `fn`s, so deleting a wrapper removes the demand and leaves the
    // contract. Nothing reaches them: no wrapper, no `lower::semantic` arm, no
    // entry in `bind/arms/layout.rs`, no host program. DECLARED AND UNREACHED.
    "layout::concat_bf16_rows",
    "layout::copy_if_valid_slot",
    "layout::deinterleave_rows_bf16",
    "layout::deinterleave_vec_bf16",
    // `OpKind::Embed`, and one of the seven. See the block above
    // `attn::split_qkv_bf16`.
    "layout::embed_bf16",
    // THE KEY-ENVELOPE TIER (3), opt-in and OFF: `Boot::kv_envelopes` defaults
    // false and nothing binds them, so on every deployment this tree ships all
    // three are compiled and none is launched. Their callers exist and are
    // exact; what gates them is a knob. Two are legs of the KV writes —
    // `attn::write_kv_explicit_bf16` and `attn::write_kv_to_pages_bf16` each
    // fire one after their own launch when the layer view answers
    // `has_envelopes`, refreshing the tier that write just invalidated. The
    // third, `layout::envelope_seed_empty`, is fired BY PATH from
    // `bind/abi.rs`'s `seed_envelopes_empty` while the KV pool materialises a
    // cache's envelope planes — no op list exists for a text to be part of.
    "layout::envelope_merge_written",
    "layout::envelope_seed_empty",
    "layout::envelope_update_appended",
    // `lower::epilogue`'s gather — the paragraph that opens the block above
    // `attn::split_qkv_bf16` is this entry's, and the seven that follow it
    // there are the same argument at scale.
    "layout::gather_bf16_rows",
    // `OpKind::SplitQGate` and `OpKind::SigmoidGateMul`, two of the seven.
    // See the block above `attn::split_qkv_bf16`.
    "layout::split_q_gate_bf16",
    "mlp::sigmoid_gate_inplace_bf16",
    // THE QUANTISED GEMM'S FOUR STAGING LAUNCHES, all legs. `gemm/quant.rs`
    // fires them by path from the int8 and fp8 arms: quantise the activation,
    // run the low-precision GEMM, dequantise the accumulator. A text states
    // the matmul and the weight REPRESENTATION it carries; the staging is what
    // reading that representation costs, chosen inside the body from the
    // dtypes rather than by anything a trace could say.
    "moe::scalar_weighted_add_bf16",
    // `OpKind::AddBias`, and one of the seven. See the block above
    // `attn::split_qkv_bf16`.
    "norm::add_bias_bf16",
    // An f16 twin. See the block above `attn::logit_softcap_f16`.
    "norm::residual_add_f16",
    // `OpKind::RmsnormGated`, and the one of the seven the driver refuses by
    // name. See the block above `attn::split_qkv_bf16`.
    "norm::rmsnorm_gated_fp32_in_bf16",
    // An f16 twin. See the block above `attn::logit_softcap_f16`.
    "norm::tanh_f16",
    // The LOADER's two quantizers, fired from `model-loader`'s
    // `executor/cuda.rs` against an arena-addressed transform plan rather than
    // recorded by any forward text: a weight transform runs once at load and
    // never appears in a fire's op list. (`executor/arena.rs` is the transform
    // DRIVER; `plan/passes/tile.rs` names both symbols as plan strings.)
    "norm::unstrided_bf16",
    // THE QUANTISED GEMM'S FOUR STAGING LAUNCHES, all legs. `gemm/quant.rs`
    // fires them by path from inside the int8 and fp8 arms of `gemm::act_x_w`
    // and the scaled-weight bodies: quantise the activation, run the low
    // precision GEMM, dequantise the accumulator. A text states the matmul
    // and the weight REPRESENTATION it carries -- which is what
    // `a_weight_representation_states_its_kernel` below is about -- and the
    // staging is what reading that representation costs, chosen inside the
    // body from the dtypes rather than by anything a trace could say. See
    // the leg block above `attn::qkv_decode_fused_dispatch`, and the arm gap
    // recorded at `gemm::gemv_bf16`.
    "quant::dequant_int32_w8a8_to_bf16",
    "quant::dequant_int8_to_bf16_per_channel",
    // An f16 twin, and the rope family's only one. See the block above
    // `attn::logit_softcap_f16`. `rope::rope_partial_bf16_position_delta` stood
    // beside it until the symbol left `sigs()` in the kernel-x sweep — a
    // remainder of `declared` minus `stated` cannot hold what nothing declares.
    // It is unreachable from BOTH ends (its arm is `unbound`, no builder
    // records it), so a draft/verify deployment that wants it needs a
    // statement, a row and an arm together.
    "quant::quantize_bf16_to_fp8_e4m3_per_channel",
    // Two more of the quantised GEMM's staging four, in sort order rather
    // than beside their pair. See the block above
    // `quant::dequant_int32_w8a8_to_bf16`.
    "quant::quantize_bf16_to_fp8_e4m3_per_token_group",
    "quant::quantize_bf16_to_int8_per_channel",
    // The second loader quantizer. See the block above
    // `quant::quantize_bf16_to_fp8_e4m3_per_channel`.
    "quant::quantize_bf16_to_mxfp4_e2m1_per_block",
    // D2's THREE-TENSOR FORM, LEFT WITHOUT A TEXT (1). §5 D2 split every
    // symbol whose operand COUNT decided what it did, and
    // `rope::rope_partial_last_bf16` is the one split whose TWO-tensor half no
    // text wanted: deepseek-v4 is the only family that rotates the last
    // channels, and its statement names `rope::rope_partial_last_q_bf16`. NOT
    // dead — the Q-alone routine calls this one with `num_kv_heads = 0`, so it
    // runs on every deepseek-v4 fire; what it lacks is a STATEMENT.
    "rope::rope_partial_f16",
    // THE THREE PLAIN ARGMAXES (3), missing not a caller but a JOB. Sampling
    // is not a model text's to state: the greedy readout is a tensor program,
    // `tensor-ir`'s `Op::ReduceArgmax` lowered by `tensor-compiler` into the
    // region's own generated kernel, which never enters this crate. The one
    // `sample` routine a text DOES state is the contrast:
    // `sample::lm_head_gemv_argmax_int8` folds the int8 LM head into the
    // readout, and a weight is the kind of thing a model text owns.
    "rope::rope_partial_last_bf16",
    "sample::argmax_bf16",
    "sample::argmax_compact_scatter_bf16",
    "sample::argmax_f32",
    // `OpKind::GdnPrep`, and one of the seven, in sort order rather than
    // beside the rest. See the block above `attn::split_qkv_bf16`.
    "ssm::qwen_gdn_post_conv_prep_bf16",
];

/// The table covers every symbol `dsl::cuda` can record.
///
/// This is the argument that [`check_plan`]'s coverage rule — which runs at
/// LOAD and fails the trace — can never fire spuriously on a live deployment:
/// reachability is a property of the dsl surface, not of which fact
/// combinations a test exercises. It is also what makes the table's other
/// declarations get written, since a new `cuda::` wrapper fails this test
/// until its contract exists.
///
/// The containment direction is load-bearing and takes no exceptions. The
/// reverse is pinned to [`UNSTATED_ROWS`] rather than asserted empty, and it
/// still fires on a new wrapper: that wrapper lands in the remainder until its
/// author either states it or names it there with a reason.
#[test]
fn the_table_covers_the_dsl_surface() {
    // The authoring surface is SIX files since `model-dsl` stopped being one
    // `lib.rs`, and this is a TEXT scan: reading only the root would see none
    // of the CUDA wrappers, which is the entire subject. `concat!` puts the
    // files back into one string rather than this test learning where a symbol
    // lives — what it asserts is a property of the SURFACE, so a wrapper
    // moving between files must not change the answer. Scanning source rather
    // than calling anything is what makes a symbol that stops being stated
    // show up here.
    let dsl = concat!(
        include_str!("../../model-dsl/src/lib.rs"),
        include_str!("../../model-dsl/src/ops.rs"),
        include_str!("../../model-dsl/src/guard.rs"),
        include_str!("../../model-dsl/src/rows.rs"),
        include_str!("../../model-dsl/src/cuda/mod.rs"),
        include_str!("../../model-dsl/src/cuda/attn.rs"),
        include_str!("../../model-dsl/src/cuda/base.rs"),
        include_str!("../../model-dsl/src/cuda/deepseek_v4.rs"),
        include_str!("../../model-dsl/src/cuda/gemma.rs"),
        include_str!("../../model-dsl/src/cuda/mla.rs"),
        include_str!("../../model-dsl/src/cuda/moe.rs"),
        include_str!("../../model-dsl/src/cuda/qwen_3_5.rs"),
        include_str!("../../model-dsl/src/cuda/rope.rs"),
        include_str!("../../model-dsl/src/cuda/ssm.rs"),
        include_str!("../../model-dsl/src/cuda/tp.rs"),
        include_str!("../../model-dsl/src/metal.rs"),
    );
    let mut stated: Vec<&str> = dsl
        .split('"')
        .skip(1)
        .step_by(2)
        .filter(|s| {
            // The prefixes a kernel symbol can start with. `ops::` and
            // `marlin_moe::` are C++ NAMESPACES the symbol genuinely carries.
            // The list is a GUESS about naming and has been wrong twice, so it
            // pins table<->dsl drift cheaply while
            // `scripts/kernel-vocabulary-audit.py` -- which reads the HEADERS
            // -- is the exhaustive check; run it when adding a family.
            //
            // `mla_absorb_`, not `mla_`: an `.expect` message reading
            // "mla_prepare states four outputs" matched the shorter one. Every
            // prefix here is as long as the symbols it must admit, no longer.
            [
                "launch_",
                "dispatch_",
                "ops::",
                "marlin_moe::",
                "gemm_",
                // One line per family as step 3 lands; when the last
                // `launch_` is gone the first five entries can go too.
                "mla_absorb_",
                "merge_",
                "flashinfer_",
                // The COLLECTIVES' namespaces. Their absence was a hole in
                // the coverage rule rather than a fact about them:
                // `dsl::cuda::all_reduce` and friends record these symbols
                // like any other, and the scan could not see them.
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
    let mut declared: Vec<&str> = sigs().iter().map(|k| k.symbol).collect();
    declared.sort_unstable();

    let unbacked: Vec<&str> = stated
        .iter()
        .filter(|s| !declared.contains(s))
        .copied()
        .collect();
    assert!(
        unbacked.is_empty(),
        "dsl::cuda records symbols the CUDA backend does not declare, so \
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

/// The retired `DepthRole`'s two facts, DERIVED, on a live depth-declaring
/// trace: membership is the layer tag, and exactly one launch per layer swaps
/// to the prefix plan. The wire word `ffi::arena` writes is computed from
/// these two, so this pins the C ABI's `depth_role` byte without the IR
/// carrying it.
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

    // PREFILL declares the axis too and its layer-tagged ops are on it, but
    // NOTHING there takes the prefix-plan swap, that being a property of the
    // planned DECODE dispatch. Which is the difference between the two halves
    // of the axis: stopping after layer `k` costs a prefill nothing, and
    // narrowing rows under it would cost it a plan it cannot build.
    let prefill = model::shared::llama_like::forward::llama_like_cuda(
        &facts,
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        FireClass::Prefill,
    );
    assert!(prefill.depth_window);
    assert!(prefill.ops.iter().any(|op| prefill.depth_windowed(op)));
    // Asked of the LOWERED form, not the traced one, because of the
    // window-class merge: the trace carries BOTH window classes as arms of a
    // `GuardPred::WindowOne` guard, so a prefill TRACE does contain the
    // planned decode dispatch — it is the arm this fire will not take.
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

    // A PADDED-HEAD deployment declares the axis too. It cannot serve the
    // narrowing half — its staging offsets are physical width while a row
    // window's are logical — but stopping after layer `k` addresses nothing.
    // So the trace states the axis and the DRIVER refuses the shapes that
    // narrow (`PaddedHeadNarrowing`).
    let padded = model::shared::llama_like::forward::llama_like_cuda(
        &facts,
        &LlamaLikeCudaFacts {
            head_dim_padded: true,
            // SYNTHETIC: this fixture's facts are qwen3-0.6B's (head_dim
            // 128), which pads nowhere. The width only has to exceed the
            // logical one for the pad statements to be well-formed.
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
    let rows = sigs();
    for (i, k) in rows.iter().enumerate() {
        for other in &rows[i + 1..] {
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

/// A quantized weight makes its statement name MORE tensors, and a dense one
/// names exactly what it did before.
///
/// The quantization axis lives on the weight handle
/// (`MatW::repr`), so `matmul(x, &w)` resolves to a stated symbol at
/// TRACE time and the scales ride as declared weights. This asserts the
/// two halves that matter: the dense path is untouched (every existing
/// golden depends on that), and each representation names a symbol the
/// CUDA backend declares — which is what stops `check_plan` refusing
/// it at load.
#[test]
fn a_weight_representation_states_its_kernel() {
    use model_dsl::{MatW, ScaleLayout, WeightRepr};

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
            sig(symbol).is_some(),
            "{symbol} needs a declaration in `kernels-cuda` or `check_plan` refuses it \
             at load"
        );
    }
}

/// Every kernel a SEMANTIC op kind can fan to has a row.
///
/// A semantic kind names no symbol, so the driver picks one, and the table's
/// coverage rule cannot see those picks because `check_plan` only walks
/// `OpKind::Launch`. Without this, a kernel reachable only through a driver's
/// fan has no operand contract anywhere and nothing notices.
///
/// The list is written by hand because there is no machine-readable link from
/// a kind to the kernels its arms call; a kind that grows a third spelling has
/// to be added here, and that is the point — the addition is where someone
/// notices the driver is choosing.
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
            if sig(s).is_none() {
                missing.push(format!("{kind} -> {s}"));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "a semantic kind fans to kernels `kernels-cuda` does not declare, so their \
         operand contract is written nowhere and `check_plan` cannot see \
         them (it walks Launch only): {missing:?}"
    );
}
