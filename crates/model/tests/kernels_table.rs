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

fn launch(symbol: &str) -> Op {
    Op {
        kind: OpKind::Launch {
            kernel: symbol.to_string(),
            weights: vec![],
            state: None,
            params: vec![],
            param_extents: vec![],
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
            window: model_ir::trace::PeelWindow::HookFreePrefix,
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
    // From the CENSUS, not from the table: Metal's is empty, every family
    // having crossed to a routine. A Metal row's symbol was a BASE and a base
    // is not something a text can launch -- every point of every axis
    // contributes text, so `attn_gate` names a kernel and `attn_gate_bfloat16`
    // names the dispatch -- and the census is the list of dispatches.
    let entrypoints = metal_entrypoints();
    let declared = entrypoints.first().expect("Metal's census names a kernel");
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

    // THE f16 TWINS (4). Each of `attn::logit_softcap_f16`,
    // `norm::residual_add_f16`, `norm::tanh_f16` and `rope::rope_partial_f16`
    // is declared beside a bf16 original that a `dsl::cuda` builder DOES
    // record -- `attn::logit_softcap_bf16`, `norm::residual_add_bf16`,
    // `norm::tanh_bf16`, `rope::rope_partial_bf16` are all in `stated` -- and
    // the twin is the same body over the other element type.
    //
    // No text names one because nothing in this tree stays fp16 across an
    // op. The only fp16 activations here are MXFP4/marlin GEMM OPERANDS:
    // gpt-oss and kimi-k2 cast with `quant::bf16_to_fp16` immediately before
    // the call and read bf16 back out of it, so nothing is ever capped,
    // rotated or residual-added in fp16. Nothing calls the four `fn`s
    // either, which was checked rather than assumed -- they are reached from
    // neither end, and `.wiki/kernel-x/northstar-old.md` recorded the same of
    // the rope one when that family crossed: *"exactly one rope row is f16
    // (`rope::rope_partial_f16`), it has no ahead-of-time twin and no bind."*
    //
    // They are here as contracts for kernels this crate compiles and this
    // tree does not fire. Which is the honest shape of a dtype variant, and
    // is why it is worth keeping them separate from the two DEAD groups
    // below: what a twin is missing is a deployment, not a caller.
    "attn::logit_softcap_f16",
    // AN INNER LEG IS NOT A STATEMENT (7). Seven symbols on this list are
    // called by ANOTHER routine in this crate rather than by anything
    // holding an op list. A text states the OUTER symbol; the outer `fn`
    // then picks a shape or a staging step on the host and calls the leg by
    // path. The leg has a row because someone put its `fn` in its family's
    // `ROUTINES`, and `sigs()` derives a row from every line there -- which
    // is right, since a body this crate compiles has an operand contract
    // whether or not a text wants one. What it does not have is a statement,
    // and it never will: a caller that has already chosen is not a
    // declaration.
    //
    // This one is the fused decode QKV path's shared body.
    // `attn::qkv_decode_qk_norm_rope_write_kv_bf16` -- which `dsl::cuda`
    // records at three sites, including the Peel's output-less form -- calls
    // it with a null window, the kernel reading null as "no window". The
    // window parameter that earns it a separate `fn` has no caller in this
    // tree that passes one.
    "attn::qkv_decode_fused_dispatch",
    // The epilogue's row gather has no `dsl::cuda` wrapper because no
    // model text states it: `lower::epilogue` emits it when the fire
    // samples fewer rows than it computes (a prefill reads one
    // distribution per request out of a stream of one row per token).
    // A statement the LOWERING makes is real without a text stating it,
    // which is precisely what this list is for.
    //
    // SEVEN MORE OF EXACTLY THAT KIND, and they are the clearest instance of
    // the sentence above. `lower::semantic` maps `OpKind::SplitQkv`,
    // `SplitQGate`, `SigmoidGateMul`, `GdnPrep`, `Embed`, `AddBias` and
    // `RmsnormGated` onto seven symbols; those op kinds carry no kernel
    // string, so no `dsl::cuda` wrapper records one and the scan cannot see
    // them. `SplitQkv` alone maps onto TWO, because the peel's tail serves
    // absolute row offsets in a full-N buffer and that is a different
    // kernel -- the lowering states the pick rather than making the driver
    // derive it from a window pointer -- so `attn::split_qkv_bf16` and
    // `attn::split_qkv_bf16_devwin` are the same statement's two regions.
    // gemma-4 and the llama-like anchor lower the split, gemma3n and the
    // qwen3.5 hybrid the gate/sigmoid/GDN three, and every text alive lowers
    // an embed; every one of the seven is named by a live deployment.
    //
    // They are DECLARED two different ways and the difference is worth one
    // line, because it is not the reason any of them is here. Four --
    // `attn::split_qkv_bf16`, `layout::split_q_gate_bf16`,
    // `mlp::sigmoid_gate_inplace_bf16`, `ssm::qwen_gdn_post_conv_prep_bf16`
    // -- have host programs in `kernels_cuda::driver_internal` and are
    // declared by a `driver_bound!` line each. The other three are ordinary
    // `routine!`s whose bodies live in their own families. Both spellings
    // land in `declared`; neither can land in `stated`, because what is
    // missing in all seven cases is a kernel string on the op kind.
    //
    // That the driver really fires them is checked from its side, not
    // assumed here: `bind/arms/` arms six of the seven for real
    // (`embed_arm`, `add_bias_arm`, `split_qkv_devwin_arm` and the rest),
    // and the seventh, `norm::rmsnorm_gated_fp32_in_bf16`, is listed there
    // as `arm: None` with a sentence saying what a `Cx` would have to grow
    // to bind it -- a refusal that names the gap, which is the answer this
    // list wants and could not give itself.
    "attn::split_qkv_bf16",
    // The peel-tail half of `OpKind::SplitQkv`. See the block above.
    "attn::split_qkv_bf16_devwin",
    // `gemm::act_x_wt_bf16`'s `m == 1` leg, chosen from the shape on the
    // host and never by a text -- and the dense tuner's `GemmKind::Gemv`
    // tactic is the same call under timing. See the leg block above
    // `attn::qkv_decode_fused_dispatch`.
    //
    // WORTH KNOWING WHILE READING THIS AND THE FOUR `quant::` LEGS BELOW:
    // none of the five is fired by a fire today, because none of `gemm`'s
    // matmul symbols has an arm. `bind/arms/gemm.rs` is a file written to
    // say so -- `gemm::act_x_w`, the portable spelling `lower::semantic`
    // emits, is `arm: None` with a reason, which `Route::refusal` turns into
    // a load-time `Unfireable` -- because the join was written by a
    // generated dispatch that has been deleted. That is a gap in the fire
    // path and not a fact about this list: the reason these five have no
    // text is that they are legs, and a leg will have no text on the day the
    // arm is written either.
    "gemm::gemv_bf16",
    // §54's FOUR DELETED WRAPPERS, BACK AS ROWS AND NOT AS DEMAND (4).
    // `model-dsl/src/lib.rs` records the deletion in place, and the record
    // is why this group needs no guessing: `copy_if_valid_slot`,
    // `concat_rows`, `deinterleave_rows` and `deinterleave_vec` were
    // wrappers that existed because the dsl surface had been GENERATED from
    // launcher headers, so a wrapper existed for every launcher whether or
    // not a model wanted one, and each of the four had zero callers in
    // `crates/model/src`. The note says the `table::layout` rows went in the
    // same edit, *"because [this test] asserts this surface and that table
    // are the same set, and half the edit fails it."*
    //
    // The rows are back, and nobody typed them. `sigs()` derives a row from
    // every line of a family's `ROUTINES` and `layout.rs` still lists all
    // four `fn`s -- so deleting a wrapper now removes the demand and leaves
    // the contract, which is the derivation working rather than failing.
    // What changed is only who has to say why: it used to be the edit, and
    // it is this list.
    //
    // And the answer is that nothing reaches them. No `dsl::cuda` wrapper,
    // no `lower::semantic` arm, no entry in `bind/arms/layout.rs`, no host
    // program in this crate -- and the witness the dsl's own note points at
    // is gone too: `kernels-cuda/tests/launch_rules.rs` fired
    // `copy_if_valid_slot` three times as the tree's only `LaunchRule::
    // Single` and was deleted in 1a08b179a, which makes that sentence stale
    // where it stands. These four are DECLARED AND UNREACHED and the entry
    // says so, rather than inventing a mechanism to make them look alive.
    "layout::concat_bf16_rows",
    "layout::copy_if_valid_slot",
    "layout::deinterleave_rows_bf16",
    "layout::deinterleave_vec_bf16",
    // `OpKind::Embed`, and one of the seven. See the block above
    // `attn::split_qkv_bf16`.
    "layout::embed_bf16",
    // THE KEY-ENVELOPE TIER (3), which is opt-in and OFF. `Boot::
    // kv_envelopes` defaults false and `boot.rs`'s own default test spells
    // the reason -- *"envelopes are opt-in; nothing binds them"* -- so on
    // every deployment this tree ships, all three are compiled and none is
    // launched. They are not in the dead groups below because their callers
    // exist and are exact; what gates them is a knob, and a knob is a fact
    // about deployments rather than about the code.
    //
    // Two of the three are legs of the KV WRITES: `attn::write_kv_explicit_
    // bf16` and `attn::write_kv_to_pages_bf16` each fire one after their own
    // launch when the layer view answers `has_envelopes`. A text states the
    // write, and refreshing the tier that write just invalidated is the
    // write's own obligation rather than a second statement.
    //
    // The third, `layout::envelope_seed_empty`, is fired BY PATH and outside
    // any fire at all: `bind/abi.rs`'s `seed_envelopes_empty` wraps it, and
    // the live KV pool calls that while materialising a cache's envelope
    // planes. A cache materialisation has no op list for a text to be part
    // of.
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
    // I COULD NOT FIND A CALLER, and the tree had already reached the same
    // answer. `moe::scalar_weighted_add_bf16` has device text, a host
    // program and now a row, and nothing calls it: not a `dsl::cuda`
    // builder, not `lower::semantic`, not `bind/arms/moe.rs`, not another
    // routine in this crate, not the loader. `.wiki/kernel-x/
    // refactor-plan.md` §12d lists it under "defects the ports found, not
    // yet acted on" in those very terms -- *"device kernel, host fn, no
    // contract, no caller anywhere. A second `scatter_add_weighted`; wants
    // the same evidence pass before deletion"* -- and the first
    // `scatter_add_weighted` was a confirmed orphan deleted whole, kernel
    // and row and builder in one commit, which `stated_columns.rs`'s
    // `DEPARTED` still records.
    //
    // So this entry is a FINDING and not a mechanism. The row is real
    // because the `fn` is; the `fn` is reached by nothing, and the pending
    // decision is whether it follows its twin out.
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
    // `norm::rmsnorm_bf16_with_fp16`'s leg for the two cases that do not
    // take the fused vec8 arm -- no fp16 copy asked for, or operands that do
    // not vectorise, in which case it norms and then casts. Its own doc
    // comment says it IS `norm::rmsnorm_bf16`, which is the sharper way to
    // put why no text names it: the trace symbol for that instantiation is
    // already spoken for by the `fn` `OpKind::Rmsnorm` lowers to, and this
    // is the same launch under a second name. See the leg block above
    // `attn::qkv_decode_fused_dispatch`.
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
    // The LOADER's two quantizers, fired from `model-loader`'s
    // `executor/cuda.rs` against an arena-addressed transform plan rather
    // than recorded by any forward text: a weight transform runs once at
    // load and never appears in a fire's op list. Real without a text
    // stating it, which is what this list is for.
    //
    // (The address used to read `loader/arena.rs`, which resolves to
    // nothing. `executor/arena.rs` is the transform DRIVER and
    // `executor/cuda.rs` holds the launches, with `plan/passes/tile.rs`
    // naming both symbols as the strings the plan carries.)
    "quant::quantize_bf16_to_fp8_e4m3_per_channel",
    // Two more of the quantised GEMM's staging four, in sort order rather
    // than beside their pair. See the block above
    // `quant::dequant_int32_w8a8_to_bf16`.
    "quant::quantize_bf16_to_fp8_e4m3_per_token_group",
    "quant::quantize_bf16_to_int8_per_channel",
    // The second loader quantizer. See the block above
    // `quant::quantize_bf16_to_fp8_e4m3_per_channel`.
    "quant::quantize_bf16_to_mxfp4_e2m1_per_block",
    // `rope::rope_partial_bf16_position_delta` STOOD HERE, and the line goes
    // because the symbol did: it left `sigs()` in the kernel-x sweep, and a
    // remainder of `declared` minus `stated` cannot hold what nothing
    // declares. Its entry predicted this exact failure and was left standing
    // to carry the evidence until someone decided which way it went. This is
    // that decision, and it goes the way the entry expected.
    //
    // What the entry established is still true, which is why nothing here
    // argues for bringing the row back: the symbol is unreachable from BOTH
    // ends. Its arm is `unbound` at `driver-cuda/src/bind/arms/rope.rs:261`
    // -- *"the offset added to every position. A fact about a draft/verify
    // pairing that no statement carries"* -- and no `dsl::cuda` builder
    // records it, which is what put it on this list to begin with. The
    // device text is untouched, so a draft/verify deployment that wants it
    // needs a statement, a row and an arm TOGETHER, exactly as §54's four
    // `layout` wrappers above do.
    //
    // `kernels-cuda/tests/stated_columns.rs`'s `DEPARTED` pins the same
    // departure from the other side, and its `why` still points here for
    // *"the second, currently-masked failure this departure causes"*. That
    // sentence is what this edit makes stale; the departure it records is
    // unchanged and correct.
    //
    // An f16 twin, and the rope family's only one. See the block above
    // `attn::logit_softcap_f16`.
    "rope::rope_partial_f16",
    // THE THREE PLAIN ARGMAXES (3), and what they are missing is not a
    // caller but a JOB. Sampling is not a model text's to state: the greedy
    // readout a fire performs is a tensor program, `tensor-ir`'s
    // `Op::ReduceArgmax` lowered by `tensor-compiler`'s CUDA codegen into
    // the region's own generated kernel, which never enters this crate at
    // all. So no `dsl::cuda` builder records these, no arm binds them, and
    // no host program here calls them -- checked, not assumed.
    //
    // The one `sample` routine a text DOES state is the contrast that makes
    // the rule readable. `sample::lm_head_gemv_argmax_int8` is stated
    // because it is a GEMM as much as an argmax: it folds the int8 LM head
    // into the readout, and a weight is exactly the kind of thing a model
    // text owns. Take the head away and what is left belongs to the sampler.
    //
    // Declared and unreached, then, like §54's four `layout` wrappers -- but
    // for the opposite reason. Those lost a wrapper that had existed; these
    // never had one, because the work moved to another compiler.
    "sample::argmax_bf16",
    "sample::argmax_compact_scatter_bf16",
    "sample::argmax_f32",
    // `OpKind::GdnPrep`, and one of the seven, in sort order rather than
    // beside the rest. See the block above `attn::split_qkv_bf16`.
    "ssm::qwen_gdn_post_conv_prep_bf16",
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
    // The authoring surface is SIX files since `model-dsl` stopped being one
    // 7,756-line `lib.rs`, and this scan is a text scan: reading only the
    // root would now see none of the CUDA wrappers, which is the entire
    // subject. `concat!` puts the files back into one string rather than
    // this test learning where a symbol lives -- what it asserts is a
    // property of the SURFACE, not of any file in it.
    //
    // Every file is included, not just `cuda.rs`, because a wrapper moving
    // between them must not change the answer. That is the same reason the
    // scan reads source at all instead of calling anything: a symbol that
    // stops being stated has to show up here.
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
                // `"pie_lora"` STOOD HERE and matches nothing now. It existed
                // for one symbol, `pie_lora_qkv_correction`, which was bare --
                // no namespace at all -- so no family prefix could reach it and
                // this list had to name it. It is `gemm::lora_qkv_correction`,
                // caught by `"gemm::"` two lines down, and the entry it needed
                // dies with it. That is what a namespace buys, stated as a
                // deletion.
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
            if sig(s).is_none() {
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
