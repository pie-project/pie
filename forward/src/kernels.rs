//! ② KERNEL SIGNATURES — one per kernel, backend-owned
//! (`.wiki/tart/dsl.md` ②).
//!
//! `dsl::cuda` has ten wrappers over five attention kernels because
//! `_region` / `_planned` / `_capture` / `_dequant` encode the DISPATCH
//! CONTEXT in the wrapper name. The context is a property of the call
//! site; what belongs to the kernel is its symbol and its contract. This
//! module holds the contract, once per symbol.
//!
//! Four declarations, each replacing something that is a hand-written
//! runtime rule today:
//!
//! | declaration | replaces |
//! |---|---|
//! | `whole`   | `if c.head_dim_padded \|\| (window_one && c.xqa_decode)` in the model body |
//! | `lacks`   | "a score-wanting program under XQA fails loudly PTIR-side" (a C++ throw) |
//! | `needs`   | the prepare a stated kernel obligates, named nowhere |
//! | `sink`    | `emit_cuda::emit_masked_pages_bracket`'s hardcoded page substitution |
//!
//! `whole` is CHECKED HERE, at trace time — which is load time, since a
//! declaration is traced when the model loads. The other three are
//! declared but not yet consumed: `needs`/`sink` are the emitter's
//! knowledge until the launch ABI flattens (migration step 6), and
//! `lacks` needs the deployment's servable-seam set, which is the
//! support-matrix work. Declaring them here first is the point — the
//! table is where they land, and it exists.
//!
//! The table is kept honest by [`check_plan`]'s second rule: every
//! `OpKind::Launch` symbol a trace records must be declared here. A new
//! kernel cannot be stated without its contract.

use crate::trace::{ForwardPlan, OpKind};

/// Which backend's kernels a lowered trace states.
///
/// The table is per-BACKEND because a kernel signature is backend-owned
/// (`.wiki/tart/dsl.md` ②: `driver/cuda/kernels.rs`). A model text is
/// written for one backend and states that backend's symbols; the
/// family name says which — `llama_like.cuda.decode` is CUDA's,
/// `llama_like.metal.decode` would be Metal's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Metal,
}

impl Backend {
    /// The backend a traced family name names, or `None` for a SEMANTIC
    /// trace — which states no kernels at all, so no table applies.
    pub fn of_family(family: &str) -> Option<Backend> {
        let mut parts = family.split('.').skip(1);
        match parts.next() {
            Some("cuda") => Some(Backend::Cuda),
            Some("metal") => Some(Backend::Metal),
            _ => None,
        }
    }

    pub fn table(self) -> &'static [KernelSig] {
        match self {
            Backend::Cuda => KERNELS,
            Backend::Metal => KERNELS_METAL,
        }
    }
}

/// A capability a seam may ask of the kernel covering its rows. Named
/// after the seam vocabulary (`.wiki/tart/dsl.md` ①), because that is
/// what a `lacks` line refuses to serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cap {
    /// The attention scores, published for an `attn.out` observer.
    Scores,
    /// The page-mask sink an `attn.q` tap writes.
    PageMaskSink,
}

/// The host-side plan a kernel's contract obligates: stated so a reader
/// of the model text can see which prepare a launch drags in, rather
/// than reading the driver to find out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prepare {
    /// No host plan.
    None,
    /// The FlashInfer decode plan (per fire, per layer group).
    DecodePlan,
    /// The FlashInfer ragged prefill plan.
    PrefillPlan,
    /// The custom-mask plan (`attn_page_mask`'s consumer).
    CustomPlan,
    /// XQA's fire-wide prepare — R-shaped, so it cannot be built per
    /// row window. This is why `xqa_decode` is also `whole`.
    FireWide,
}

/// One kernel's contract.
pub struct KernelSig {
    /// The dsl-side name (what a model text spells).
    pub name: &'static str,
    /// The C++ launcher symbol the trace records.
    pub symbol: &'static str,
    /// The kernel REFUSES a row split: it may not be stated inside a
    /// [`OpKind::Peel`]'s regions, because its addressing (a fire-wide
    /// prepare, a padded staging buffer) is not row-offsettable.
    pub whole: bool,
    /// The host plan its contract obligates.
    pub needs: Prepare,
    /// Capabilities this kernel cannot serve — a seam asking for one of
    /// these over rows this kernel covers is unservable.
    pub lacks: &'static [Cap],
    /// Where a sink-writing seam's output lands, if this kernel accepts
    /// one (`sink pages -> kv.pages`).
    pub sink: Option<&'static str>,
    /// On a union tail layer this dispatch pairs the DEPTH PREFIX plan
    /// (and its dedicated workspace) instead of the fire's own plan.
    ///
    /// This was the `PrefixPlanSwap` half of the retired per-op
    /// `DepthRole` — a word the IR carried on one launch per layer of
    /// every depth-declaring trace, restating a fact about the KERNEL.
    /// Migration step 5 moved it here.
    pub depth_prefix_plan: bool,
}

/// Declare one kernel. The syntax is `.wiki/tart/dsl.md` ②'s, minus the
/// operand shapes: those stay with the emitter until the launch ABI
/// flattens, and stating them twice would be the duplication this
/// redesign exists to remove.
macro_rules! kernel {
    ($name:ident $symbol:literal $(, $key:ident = $value:expr)* $(,)?) => {
        KernelSig {
            name: stringify!($name),
            symbol: $symbol,
            $($key: $value,)*
            ..KernelSig {
                name: "",
                symbol: "",
                whole: false,
                needs: Prepare::None,
                lacks: &[],
                sink: None,
                depth_prefix_plan: false,
            }
        }
    };
}

/// Every kernel a lowered declaration may state.
pub static KERNELS: &[KernelSig] = &[
    // ── attention ──────────────────────────────────────────────────
    kernel!(flashinfer_decode "dispatch_attention_flashinfer_decode",
        needs = Prepare::DecodePlan, sink = Some("kv.pages"),
        depth_prefix_plan = true),
    kernel!(flashinfer_decode_capture "dispatch_attention_flashinfer_decode_capture",
        needs = Prepare::DecodePlan, sink = Some("kv.pages")),
    kernel!(flashinfer_prefill "dispatch_attention_flashinfer_prefill_bf16",
        needs = Prepare::PrefillPlan, sink = Some("kv.pages")),
    kernel!(flashinfer_prefill_capture "dispatch_attention_flashinfer_prefill_capture_bf16",
        needs = Prepare::PrefillPlan, sink = Some("kv.pages")),
    kernel!(flashinfer_custom "dispatch_attention_flashinfer_prefill_custom",
        needs = Prepare::CustomPlan, sink = Some("kv.pages")),
    // XQA: its prepare is fire-wide (R-shaped), so the kernel cannot be
    // given a row window — `whole`. And no capture variant of it
    // exists, so it cannot publish scores — `lacks Scores`. Both are
    // hand-written rules today: the first is the model body's
    // `window_one && c.xqa_decode` test, the second a C++ throw.
    kernel!(xqa_decode "launch_attention_xqa_decode_bf16_prepared",
        whole = true, needs = Prepare::FireWide, lacks = &[Cap::Scores]),
    kernel!(dequant "launch_dequant_kv_cache_layer_to_bf16_active"),

    // ── qkv / norms / rope / kv write ──────────────────────────────
    kernel!(rope_standard_table "launch_rope_standard_table"),
    kernel!(qk_rmsnorm_rope "launch_qk_rmsnorm_rope_bf16"),
    kernel!(qkv_decode_fused "launch_qkv_decode_qk_norm_rope_write_kv_bf16"),
    kernel!(write_kv_explicit "launch_write_kv_explicit_bf16"),
    kernel!(write_kv_to_pages "launch_write_kv_to_pages"),

    // ── mlp ────────────────────────────────────────────────────────
    // Two spellings of one arithmetic, and the BINDING picks: a packed
    // gate‖up bank feeds the chunked form, two narrow buffers the pair
    // form. A load-time fact, so the declaration states it.
    kernel!(chunked_swiglu "launch_chunked_swiglu_bf16"),
    kernel!(swiglu "launch_swiglu_bf16"),

    // ── MoE ────────────────────────────────────────────────────────
    // The router's top-k, then the decode GEMV leg's two routed
    // projections and its combine. The expert axis rides INSIDE the
    // value on this leg, so the whole branch stays a list of rectangles;
    // the grouped-GEMM and host-routed legs reach the same numbers by
    // shapes no `Dim` spells, and are named refusals, not entries.
    kernel!(topk_softmax "launch_topk_softmax_bf16"),
    // The whole routed block as one call — permute, both grouped GEMMs,
    // the activation and the weighted finalize. The leg decode actually
    // takes, and the only one that is a single rectangle.
    // Namespaced because it is not a `kernels::launch_*` at all: it is an
    // `ops::` entry point that installs tactics and runs a CUTLASS
    // pipeline. The symbol says so.
    kernel!(moe_fused_cutlass "ops::flashinfer_cutlass_moe_bf16"),
    kernel!(moe_gate_up_gemv "launch_moe_gate_up_decode_gemv_bf16"),
    kernel!(moe_down_gemv "launch_moe_down_decode_gemv_bf16"),
    kernel!(moe_shared_gate_dot "launch_sigmoid_dot_scalar_gate_add_bf16"),
    kernel!(residual_add_cuda "launch_residual_add_bf16"),
    // The combine folds the residual when the MoE output lands straight
    // on the stream (tp=1) — one launch where the semantic text has a
    // WeightedSum and a ResidualAdd.
    kernel!(moe_weighted_sum "launch_token_batched_weighted_sum_bf16"),
    kernel!(moe_weighted_sum_add "launch_token_batched_weighted_sum_add_bf16"),

    // ── adapters ───────────────────────────────────────────────────
    kernel!(lora_qkv_correction "pie_lora_qkv_correction"),

    // ── gdn: conv, recurrence, stash ───────────────────────────────
    kernel!(gdn_conv_update "launch_causal_conv1d_update_batched_bf16"),
    kernel!(gdn_conv_prefill "launch_causal_conv1d_prefill_batched_bf16"),
    kernel!(gdn_step "launch_recurrent_gated_delta_step_batched"),
    kernel!(gdn_step_gqa "launch_recurrent_gated_delta_step_batched_gqa"),
    kernel!(gdn_step_state_bf16 "launch_recurrent_gated_delta_step_batched_state_bf16"),
    kernel!(gdn_step_gqa_state_bf16 "launch_recurrent_gated_delta_step_batched_gqa_state_bf16"),
    kernel!(gdn_prefill_fla "launch_chunk_gated_delta_prefill_batched"),
    kernel!(gdn_prefill_fla_state_bf16 "launch_chunk_gated_delta_prefill_batched_state_bf16"),
    kernel!(gdn_prefill_cached "launch_chunk_gated_delta_prefill_batched_cached"),
    kernel!(gdn_prefill_cached_state_bf16
        "launch_chunk_gated_delta_prefill_batched_cached_state_bf16"),
    kernel!(gdn_prefill_warp_tiled "launch_chunk_gated_delta_prefill_batched_warp_tiled"),
    kernel!(gdn_prefill_warp_tiled_gqa "launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa"),
    kernel!(gdn_prefill_warp_tiled_state_bf16
        "launch_chunk_gated_delta_prefill_batched_warp_tiled_state_bf16"),
    kernel!(gdn_prefill_warp_tiled_gqa_state_bf16
        "launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16"),
    kernel!(repeat_interleave_heads "launch_repeat_interleave_heads_fp32"),
    kernel!(verify_stash_store "qwen35_verify_stash_store"),
    kernel!(verify_stash_load "qwen35_verify_stash_load"),
];

/// METAL's kernel signatures.
///
/// EMPTY, and deliberately so: Metal has no lowered text yet. It
/// consumes the SEMANTIC trace and re-derives its dispatch selection in
/// C++ (`driver/metal/src/model/llama_like/declared_dag.hpp`) — the same
/// "the driver decides" shape the CUDA side is being cured of, from the
/// other end.
///
/// An empty table is not a placeholder that does nothing. Because
/// [`check_plan`]'s coverage rule refuses any undeclared symbol, a
/// `llama_like.metal.*` text CANNOT be written without declaring the
/// kernels it states, which is exactly the discipline the CUDA table
/// enforces. The first Metal text fills this in.
pub static KERNELS_METAL: &[KernelSig] = &[
    // ── io ─────────────────────────────────────────────────────────
    kernel!(embed_gather "embed_gather_4bit"),
    kernel!(embed_gather_mb "embed_gather_mb_4bit"),

    // ── norms / activation / residual ──────────────────────────────
    // One entrypoint serves attn_norm, mlp_norm, q_norm, k_norm and
    // final_norm — the driver fans five `Kernel` kinds onto it.
    kernel!(rms_norm "rms_single_row_bfloat16"),
    kernel!(silu_mul "silu_mul_bfloat16"),
    kernel!(residual_add "residual_add_bfloat16"),

    // ── projections ────────────────────────────────────────────────
    // The `_residual` forms fold the block residual in the GEMV/GEMM
    // epilogue, which is what a `beta_one` matmul is on this backend.
    // The readout takes this one too — `lm_head` is a projection, and
    // the driver has no separate entrypoint for it.
    kernel!(qmv "affine_qmv_fast"),
    kernel!(qmv_residual "affine_qmv_fast_residual"),
    kernel!(qmm "affine_qmm_t"),
    kernel!(qmm_residual "affine_qmm_t_residual"),

    // ── rope / kv ──────────────────────────────────────────────────
    kernel!(rope_decode "rope_neox_decode_bfloat16"),
    kernel!(rope_mb "rope_neox_mb_bfloat16"),
    kernel!(kv_append "kv_append_bfloat16"),
    kernel!(kv_append_paged "kv_append_paged_bfloat16"),

    // ── attention ──────────────────────────────────────────────────
    // No `sink` on either: Metal has no page-mask substitution path, so
    // an `attn.q` tap with PageMaskSink is unservable here — the
    // declaration says so instead of a C++ throw discovering it. No
    // capture variant exists either, so neither can publish scores.
    kernel!(sdpa_vector "sdpa_vector_decode_bfloat16_d_256",
        lacks = &[Cap::Scores, Cap::PageMaskSink]),
    kernel!(sdpa_paged "sdpa_paged_decode_bfloat16_d_256",
        lacks = &[Cap::Scores, Cap::PageMaskSink]),
];

/// The contract for one recorded symbol, in `backend`'s table.
pub fn sig_in(backend: Backend, symbol: &str) -> Option<&'static KernelSig> {
    backend.table().iter().find(|k| k.symbol == symbol)
}

/// The contract for one CUDA symbol.
pub fn sig(symbol: &str) -> Option<&'static KernelSig> {
    sig_in(Backend::Cuda, symbol)
}

/// LOAD-TIME check of a traced form against the kernel table.
///
/// Two rules, both of which are runtime failures today:
///
/// 1. a `whole` kernel may not be stated inside a [`OpKind::Peel`]'s
///    regions — the peel gives each region a row window, and a
///    fire-wide-prepared kernel has no way to honour one;
/// 2. every launched symbol must be declared, so the table cannot rot
///    while the model texts move on.
///
/// Returns the failures rather than panicking, so a caller can name the
/// family it was loading.
pub fn check_plan(plan: &ForwardPlan) -> Vec<String> {
    let mut problems = Vec::new();
    let backend = Backend::of_family(&plan.family);
    // Ops inside a Peel's two regions, as a countdown over the flat op
    // list (regions are consecutive: prefix then tail, right after the
    // op — `OpKind::Peel`'s doc).
    let mut peeled = 0usize;
    for op in &plan.ops {
        let inside_peel = peeled > 0;
        peeled = peeled.saturating_sub(1);
        match &op.kind {
            OpKind::Peel {
                prefix_ops,
                tail_ops,
                ..
            } => {
                peeled = peeled.max(*prefix_ops as usize + *tail_ops as usize);
            }
            OpKind::Launch { kernel, .. } => match backend.and_then(|b| sig_in(b, kernel)) {
                None => problems.push(format!(
                    "{}: launches `{kernel}`, which no {} kernel! signature declares",
                    plan.family,
                    match backend {
                        Some(b) => format!("{b:?}").to_lowercase(),
                        // A semantic trace states no kernels; one that
                        // does has a family name that does not say
                        // whose they are.
                        None => "backend's".to_string(),
                    }
                )),
                Some(k) if k.whole && inside_peel => problems.push(format!(
                    "{}: `{kernel}` is declared `whole` (needs {:?}) but is stated \
                     inside a Peel region, which gives it a row window it cannot honour",
                    plan.family, k.needs
                )),
                Some(_) => {}
            },
            _ => {}
        }
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::{LlamaLikeCudaFacts, LlamaLikeFacts, Qwen35CudaFacts, Qwen35HybridFacts};
    use crate::family;
    use crate::trace::FireClass;

    use crate::trace::{Op, OpKind};

    fn launch(symbol: &str) -> Op {
        Op {
            kind: OpKind::Launch {
                kernel: symbol.to_string(),
                weights: vec![],
                state: None,
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
                window: crate::trace::PeelWindow::HookFreePrefix,
            },
            inputs: vec![],
            outputs: vec![],
            layer: Some(0),
        };
        let xqa = "launch_attention_xqa_decode_bf16_prepared";
        let problems = check_plan(&plan_of(vec![
            peel,
            launch(xqa),
            launch("dispatch_attention_flashinfer_decode"),
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
        assert_eq!(Backend::of_family("llama_like.cuda.decode"), Some(Backend::Cuda));
        assert_eq!(
            Backend::of_family("qwen3_5_hybrid.cuda.commit_advance"),
            Some(Backend::Cuda)
        );
        assert_eq!(Backend::of_family("llama_like.metal.decode"), Some(Backend::Metal));
        // Semantic traces state no kernels, so no table applies.
        assert_eq!(Backend::of_family("llama_like"), None);
        assert_eq!(Backend::of_family("qwen3_5_moe_mlp_block"), None);
    }

    /// Metal's table is empty, and that REFUSES rather than permits: a
    /// `llama_like.metal.*` text cannot state a kernel it has not
    /// declared. This is the discipline that will fill the table when
    /// the first Metal text is written.
    #[test]
    fn an_empty_backend_table_refuses() {
        let mut p = plan_of(vec![launch("metal_gemm_bf16")]);
        p.family = "llama_like.metal.decode".to_string();
        // (the same symbol under CUDA's table is refused too — this is
        // about WHICH table, not about permissiveness)
        let problems = check_plan(&p);
        assert_eq!(problems.len(), 1, "{problems:#?}");
        assert!(problems[0].contains("metal"), "{}", problems[0]);
    }

    /// The table is exactly the set of symbols `dsl::cuda` can record.
    ///
    /// This is the argument that [`check_plan`]'s coverage rule — which
    /// runs at LOAD and fails the trace — can never fire spuriously on a
    /// live deployment: reachability is a property of the dsl surface,
    /// not of which fact combinations a test happens to exercise. And it
    /// is the guard that makes the table's other three declarations get
    /// written: a new `cuda::` wrapper fails this test until its
    /// contract exists.
    #[test]
    fn the_table_is_exactly_the_dsl_surface() {
        let dsl = include_str!("dsl.rs");
        let mut stated: Vec<&str> = dsl
            .split('"')
            .skip(1)
            .step_by(2)
            .filter(|s| {
                ["launch_", "dispatch_", "ops::", "pie_lora", "qwen35_verify"]
                    .iter()
                    .any(|p| s.starts_with(p))
            })
            .collect();
        stated.sort_unstable();
        stated.dedup();
        let mut declared: Vec<&str> = KERNELS.iter().map(|k| k.symbol).collect();
        declared.sort_unstable();
        assert_eq!(
            stated, declared,
            "the kernel! table and dsl::cuda's stated symbols have drifted"
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
        let plan = family::llama_like_cuda(
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
        let swaps = plan.ops.iter().filter(|op| plan.depth_prefix_plan(op)).count();
        assert_eq!(swaps, 3 * facts.layers as usize);
        assert!(
            plan.ops.iter().filter(|op| plan.depth_prefix_plan(op)).all(|op| matches!(
                &op.kind,
                OpKind::Launch { kernel, .. }
                    if kernel == "dispatch_attention_flashinfer_decode"
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
        let prefill = family::llama_like_cuda(
            &facts,
            &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
            FireClass::Prefill,
        );
        assert!(prefill.depth_window);
        assert!(prefill
            .ops
            .iter()
            .any(|op| prefill.depth_windowed(op)));
        assert_eq!(
            prefill.ops.iter().filter(|op| prefill.depth_prefix_plan(op)).count(),
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
        let padded = family::llama_like_cuda(
            &facts,
            &LlamaLikeCudaFacts {
                head_dim_padded: true,
                ..LlamaLikeCudaFacts::qwen3_0_6b_l40s()
            },
            FireClass::Prefill,
        );
        assert!(padded.depth_window);

        // The XQA decode deployment is the one that still withholds it:
        // its prepare is fire-wide and R-shaped, so even the free half
        // has nothing to stand on.
        let xqa = family::llama_like_cuda(
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
            plans.push(family::llama_like_cuda(
                &LlamaLikeFacts::qwen3_0_6b(),
                &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
                class,
            ));
            plans.push(family::llama_like_cuda(
                &LlamaLikeFacts::mistral_7b_v03(),
                &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
                class,
            ));
        }
        for class in [
            FireClass::Decode,
            FireClass::Prefill,
            FireClass::StateOnly,
            FireClass::CommitAdvance,
            FireClass::FrozenVerify,
        ] {
            plans.push(family::qwen3_5_hybrid_cuda(
                &Qwen35HybridFacts::qwen3_5_0_8b(),
                &Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
                class,
            ));
            // Qwen3.6-27B: the same text at a different geometry, and
            // the first one whose GDN half is GQA.
            plans.push(family::qwen3_5_hybrid_cuda(
                &Qwen35HybridFacts::qwen3_6_27b(),
                &Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
                class,
            ));
        }
        for plan in &plans {
            let problems = check_plan(plan);
            assert!(problems.is_empty(), "{problems:#?}");
        }
    }
}
