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
            }
        }
    };
}

/// Every kernel a lowered declaration may state.
pub static KERNELS: &[KernelSig] = &[
    // ── attention ──────────────────────────────────────────────────
    kernel!(flashinfer_decode "dispatch_attention_flashinfer_decode",
        needs = Prepare::DecodePlan, sink = Some("kv.pages")),
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

/// The contract for one recorded symbol.
pub fn sig(symbol: &str) -> Option<&'static KernelSig> {
    KERNELS.iter().find(|k| k.symbol == symbol)
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
            OpKind::Launch { kernel, .. } => match sig(kernel) {
                None => problems.push(format!(
                    "{}: launches `{kernel}`, which no kernel! signature declares",
                    plan.family
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
            depth_role: None,
        }
    }

    fn plan_of(ops: Vec<Op>) -> ForwardPlan {
        ForwardPlan {
            family: "test".to_string(),
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
            depth_role: None,
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
        }
        for plan in &plans {
            let problems = check_plan(plan);
            assert!(problems.is_empty(), "{problems:#?}");
        }
    }
}
