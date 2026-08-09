//! What a Metal load OBSERVED, and the one door to the row's Metal text.
//!
//! # This file is what is left of `model/text.rs`
//!
//! That module held an eleven-entry table of architecture STRING names
//! (`LLAMA_LIKE`), a `canonical()` that reduced a spelling to one of them, a
//! `serves()` predicate over it, and a `facts_from_with()` that rebuilt the
//! model's own twenty-nine `LlamaLikeFacts` from NINE `has_tensor` probes. It
//! was **the third dispatch key** the catalog refactor exists to delete: a
//! checkpoint's identity was decided once by `catalog::identify` from the
//! tensors, and then decided again here from a string the row had handed out.
//!
//! Two keys for one identity is not a tidiness problem, and this crate has
//! the receipts for what it costs:
//!
//! * `LLAMA_LIKE` listed `"gemma4"` and `"gpt_oss"`, so `serve/load.rs`
//!   CLAIMED to serve them at its pre-staging gate and then refused them
//!   ninety lines later on a completely different ground — the checkpoint
//!   ships a `pre_feedforward_layernorm` and the projected geometry could
//!   only state a SiLU gate. One driver, two answers, and the second one
//!   arrived after 17 GB of weights had been staged.
//! * `facts_from_with` decided the norm variant by asking whether
//!   `layers.0.pre_feedforward_layernorm.weight` exists. gemma-4 publishes
//!   that norm and stores its gains as a plain multiplier, so the probe read
//!   `(1 + w)` for a stack that means `w` — a norm whose LARGEST value agreed
//!   with MLX to three digits while its ordinary ones were off by a third.
//!   A row states the fold; a tensor's existence never could.
//!
//! Both are unrepresentable now. `catalog::Variant::trace` takes a
//! `catalog::Deployed` whose `backend` says which driver is asking, so the
//! Metal text is the ROW's answer for the Metal backend, and a row this build
//! has no Metal text for refuses in the one place the CUDA driver's refusal
//! also lives.
//!
//! # What a driver may still say about a model, and why it is only this
//!
//! [`MetalBinding`] is six values and every one of them is about the BYTES
//! that arrived or about the KERNELS this binary was built with. None is
//! about the model:
//!
//! * the affine group and bit width, because `mlx-community` publishes one
//!   model at g64/b4 and at g128/b8 and the two pack to identical extents —
//!   no tensor can be asked which it is, and no row may be split four ways to
//!   record it;
//! * whether the expert bank reached the device still in MXFP4, which is a
//!   statement about what the LOADER did and is only knowable after staging;
//! * and the three kernel capabilities, which are facts about
//!   `crates/kernels-metal` as compiled into this binary.
//!
//! [`observed`] takes an [`AffineFormat`](crate::batch::AffineFormat) and a
//! single tensor probe, and that narrowness is the design rather than an
//! economy: **it cannot see a `DecodeGeometry`, so it cannot smuggle a model
//! fact into a binding.** The nine probes that used to decide head norms,
//! fused projections, attention biases, routers, sinks, per-layer scalars and
//! the activation have nowhere to be written down here.

use model::catalog::MetalBinding;
use model_compiler::trace::{FireClass, ForwardPlan};

/// The one tensor this driver still asks a question of, and the question is
/// about an ENCODING.
///
/// `mlx-community/gpt-oss-20b-MXFP4-Q4` names 98 tensors as affine/64/4 and
/// leaves the expert banks out, so they take the top-level default, mxfp4/32.
/// A checkpoint need not quantize uniformly, and reading a bank with the
/// dense format is 909,207 NaNs rather than a near miss — the first of them
/// in `affine_qmv_routed_bfloat16_gs_64_b_4`, layer 0.
///
/// It is named as a constant rather than spelled at the call site so that the
/// claim "one probe, and it decides an encoding" is something a test can
/// check by reading this file. Whether the model HAS experts is the row's
/// answer and is not asked here; this asks only what format the bytes for
/// them arrived in.
pub const EXPERT_BANK: &str = "layers.0.mlp.experts.gate_proj.weight";

/// What this build's kernels can do, at any encoding.
///
/// Three booleans that are true of the BINARY: the projection GEMV folds the
/// block residual in its epilogue (`affine_qmv_fast_residual`), the M>1 lane
/// addresses the KV cache through a page table (`sdpa_paged_decode`), and the
/// M>1 projections take MLX's steel quantized GEMM (`affine_qmm_t`). They
/// were hardcoded `true` inside `facts_from_with` beside twenty-six values
/// that were the MODEL's; separating them is most of what this file is.
///
/// A `const fn` so that [`ANY_ENCODING`] and [`observed`] cannot drift: a
/// build capability answered one way for a real load and another way for the
/// pre-staging question would make the refusal at the door disagree with the
/// text at the fire, which is exactly the two-answers defect this module's
/// predecessor was.
#[must_use]
pub const fn build_kernels(quant_group: u32, quant_bits: u32, moe_mxfp4: bool) -> MetalBinding {
    MetalBinding {
        quant_group,
        quant_bits,
        moe_mxfp4,
        fuse_residual_gemv: true,
        paged_multi_batch: true,
        qmm_multi_batch: true,
    }
}

/// The binding to ask with when the question is "is there a Metal text at
/// all".
///
/// # The refusal must not depend on the binding facts
///
/// `serve/load.rs` refuses an unservable row BEFORE it stages a byte, and
/// that placement is load-bearing: on the 31B gemma the alternative is 17 GB
/// of staging spent to reach an answer identification had already settled.
/// But `moe_mxfp4` cannot be known until the tensors are on the device, so
/// the full binding does not exist yet at the moment the question is asked.
///
/// That is not a gap, because **a row that refuses Metal refuses it for every
/// encoding.** `Refusal::Unsupported` from `trace` means this build has no
/// Metal text for the row — a statement about which texts were written, which
/// no group size, bit width or expert-bank format can change. So the probe
/// below stands for every encoding, and the g64/b4 it names is the shipped
/// body format rather than a fact anything reads.
///
/// `a_row_is_served_the_same_way_at_every_encoding` in this file's tests is
/// that sentence as an assertion, over the whole catalog: if some row's
/// answer ever DID move with the binding, the pre-staging refusal would be
/// lying and this constant would have to go.
pub const ANY_ENCODING: MetalBinding = build_kernels(64, 4, false);

/// What THIS load observed, once its weights are on the device.
///
/// `quant` is the geometry's affine point and not the encoding's: a
/// checkpoint that declares no quantization arrives here as g64/b4, because
/// `geometry_from_deployment` has already replaced an unset format with the
/// one a shader is actually instantiated at — `gs_0_b_0` is a symbol no
/// kernel exports. Reading the encoding directly would name it.
///
/// `is_mxfp4` is `Loaded::mxfp4`, the set of names the LOAD PLAN left in the
/// checkpoint's own format. Exactly one name is put to it ([`EXPERT_BANK`]),
/// and the signature is why that is checkable rather than promised: nothing
/// else about the deployment is in scope here to be asked about.
#[must_use]
pub fn observed(
    quant: crate::batch::AffineFormat,
    is_mxfp4: impl Fn(&str) -> bool,
) -> MetalBinding {
    build_kernels(quant.group, quant.bits, is_mxfp4(EXPERT_BANK))
}

/// A row that answered a Metal load with some other backend's text.
///
/// Named as a constant because it is the sentence an operator would read,
/// and it should say what happened rather than name a symbol. No row in this
/// build produces it — every generation states its own Metal answer, so a
/// row either refuses in its own words or traces `llama_like.metal.*` — and
/// this is deliberately the driver's sentence rather than the row's, because
/// reaching it means a row said something about Metal that was not true.
const NOT_A_METAL_TEXT: &str =
    "this row answered a Metal load with another backend's text; the text \
     exists but it was written for other kernels, so a fire would lower \
     symbols no Metal shader exports";

/// The row's Metal text for one fire class.
///
/// Nearly a one-line function on purpose: it is the ONLY place this driver
/// reaches a forward text, so "which text does this checkpoint run" has
/// exactly one answer site, and that answer is a row's. What it replaces —
/// `text::plan_for(arch, class, &facts, &metal)` — took an architecture
/// STRING and a facts struct the driver had rebuilt itself, which is two
/// chances to describe a different model than the one that was loaded.
///
/// # The one thing it checks, and why it is not a second list
///
/// `Deployed::backend` is a REQUEST, and the answer is checked against it.
/// Every generation now states its own Metal answer: eight trace one
/// (`llama_3`, `qwen_2`, `qwen_3`, `mistral_3`, `phi_3`, `olmo_2`, `olmo_3`,
/// `gemma_3`) and the rest refuse by name, each with a `Refusal::Unsupported`
/// literal that names the generation and the CUDA text it has instead. So
/// this check does not currently fire for any row, and that is the state it
/// is meant to hold.
///
/// What it is FOR is the way a Metal arm goes wrong. `gemma_3`'s is a
/// `match` on `load.backend` whose two arms differ by one call —
/// `family::trace(f, row, class, load)` against
/// `llama_like_metal(f, &metal_facts(..), class)` — and both return
/// `Ok(ForwardPlan)`. A generation that grows a Metal arm and reaches for
/// the neighbouring builder compiles, deploys, projects a geometry, and
/// answers `trace` with a text whose every symbol is looked up in the Metal
/// table at fire time. `Backend::of_family` reads the backend the trace
/// stamped on ITSELF, so that slip is a refusal at this door instead of a
/// missing shader at the first dispatch.
///
/// That is not the string table coming back. The table said which
/// ARCHITECTURES a text had been written for and had to be edited by hand
/// every time one was; this reads what the text ITSELF says it is, from the
/// same value that is about to be lowered, so there is nothing to keep in
/// step and nothing to be wrong about. The distinction is the whole point:
/// `LLAMA_LIKE` could name `gemma4` while no gemma-4 text existed, and this
/// cannot. It also cannot refuse a row the catalog serves: `gemma3` was
/// never IN that table, and it traces a Metal text today.
///
/// # Errors
///
/// [`Refusal::Unsupported`](model::deployment::Refusal) when this build has
/// no Metal text for the row — almost always the row's own refusal, carried
/// out unchanged. Beyond that the driver does not second-guess the row and
/// holds no list of its own to check first: `LLAMA_LIKE` naming `gemma4`
/// while the load path refused every gemma-4, and omitting `gemma3` which
/// this build serves, is what a second list is worth.
pub fn text(
    row: &dyn model::catalog::Variant,
    class: FireClass,
    binding: &MetalBinding,
) -> Result<ForwardPlan, model::deployment::Refusal> {
    let plan = row.trace(class, model::catalog::Deployed::metal(binding))?;
    if model_compiler::kernels::Backend::of_family(&plan.family)
        != Some(model_compiler::kernels::Backend::Metal)
    {
        return Err(model::deployment::Refusal::Unsupported(NOT_A_METAL_TEXT));
    }
    Ok(plan)
}

/// Whether this build has a Metal text for `row`, asked without the weights.
///
/// Answered by the ROW, through [`ANY_ENCODING`] — see that constant for why
/// asking before the binding is known is sound. `FireClass::Decode` because
/// every text states a decode; a row that had one class and not the other
/// would be a row that loads and dies at its first prefill, which is the
/// failure `Variant::trace`'s own doc records `unbuilt_kv_store()` existing
/// for.
///
/// # Errors
///
/// The row's own [`Refusal`](model::deployment::Refusal), carried unchanged
/// so the caller can print what the row said rather than a sentence this
/// driver made up about it — or [`text`]'s, when the row answered for a
/// backend that is not this one.
pub fn serves(row: &dyn model::catalog::Variant) -> Result<(), model::deployment::Refusal> {
    text(row, FireClass::Decode, &ANY_ENCODING).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::AffineFormat;

    /// The checkpoint's own affine point reaches the binding, both numbers.
    ///
    /// Neither is inferable from a tensor: g64/b8 and g128/b4 pack to
    /// identical extents, so a pipeline built for the wrong point returns
    /// fluent nonsense rather than failing. That is the whole reason an
    /// encoding is asked of the LOAD and the whole reason a row must not be
    /// split four ways to hold one.
    #[test]
    fn the_binding_carries_the_checkpoints_own_affine_point() {
        let b = observed(AffineFormat { bits: 8, group: 128 }, |_| false);
        assert_eq!(b.quant_bits, 8);
        assert_eq!(b.quant_group, 128);

        let g64 = observed(AffineFormat { bits: 4, group: 64 }, |_| false);
        assert_eq!(g64.quant_bits, 4);
        assert_eq!(g64.quant_group, 64);
    }

    /// ONE tensor is probed, and it is the expert bank.
    ///
    /// The count is the assertion. `facts_from_with` made nine probes and
    /// eight of them decided a MODEL fact — the per-head q-norm, the fused
    /// QKV, the attention bias, the router, the shared expert, the sandwich
    /// norm, the attention sink, the per-layer scalar. Every one is a row's
    /// answer now, and a regression would show up here as a second name
    /// being asked for.
    #[test]
    fn exactly_one_tensor_is_asked_about_and_it_decides_an_encoding() {
        let asked = std::cell::RefCell::new(Vec::new());
        let b = observed(AffineFormat { bits: 4, group: 64 }, |name| {
            asked.borrow_mut().push(name.to_string());
            name == EXPERT_BANK
        });
        assert_eq!(
            asked.into_inner(),
            vec![EXPERT_BANK.to_string()],
            "the load's only question of the tensors is what format the \
             expert bank arrived in"
        );
        assert!(b.moe_mxfp4, "the bank answered mxfp4, so the binding says so");

        let dense = observed(AffineFormat { bits: 4, group: 64 }, |_| false);
        assert!(
            !dense.moe_mxfp4,
            "a checkpoint whose bank was transcoded reads as the dense format"
        );
    }

    /// The three kernel capabilities are the BUILD's, so the probe binding
    /// and a real one cannot disagree about them.
    #[test]
    fn the_build_capabilities_do_not_move_with_the_encoding() {
        let real = observed(AffineFormat { bits: 8, group: 32 }, |_| true);
        assert_eq!(real.fuse_residual_gemv, ANY_ENCODING.fuse_residual_gemv);
        assert_eq!(real.paged_multi_batch, ANY_ENCODING.paged_multi_batch);
        assert_eq!(real.qmm_multi_batch, ANY_ENCODING.qmm_multi_batch);
    }

    /// **A row that refuses Metal at the probe refuses it at every
    /// encoding.**
    ///
    /// The premise [`ANY_ENCODING`] rests on, asserted over the whole
    /// catalog rather than argued for in prose. `serves` is asked before a
    /// byte is staged and therefore before `moe_mxfp4` can be known; if a
    /// row that refused at the probe could be served at some other group
    /// size, the pre-staging refusal would be turning away a checkpoint
    /// this build can run, and the honest response would be to delete the
    /// constant rather than keep the placement.
    ///
    /// # Why this is one-directional and was not
    ///
    /// It asserted EQUALITY, and the other direction is false: a routed
    /// row that serves at the probe is refused at g128/b8, because
    /// `quant/qmv.metal` instantiates `affine_qmv_routed` at group 64
    /// alone — `AffineQ::group_size` is a template constant, so a second
    /// point would name a kernel that dequantises at 64 whatever it
    /// claimed. `qwen3-30b-a3b` is the row that says so.
    ///
    /// That direction was never the premise. `serves` is allowed to be
    /// PERMISSIVE — it exists so 17 GB of gemma is not staged to reach an
    /// answer identification already had — and a later refusal at the
    /// real encoding costs a load, not a wrong answer. What it may not be
    /// is wrong in the other direction, and that is what this holds.
    #[test]
    fn a_row_is_served_the_same_way_at_every_encoding() {
        let bindings = [
            ANY_ENCODING,
            build_kernels(32, 4, true),
            build_kernels(128, 8, false),
            build_kernels(64, 8, true),
        ];
        let mut rows = 0usize;
        for row in model::catalog::catalog() {
            let want = serves(*row).is_ok();
            for b in &bindings {
                assert!(
                    want || text(*row, FireClass::Decode, b).is_err(),
                    "`{}` is refused at the probe binding and served at \
                     g{}/b{} (bank mxfp4: {}). `serve/load.rs` refuses \
                     before a byte is staged, so a refusal there must hold \
                     for every encoding a load could turn out to have",
                    row.id(),
                    b.quant_group,
                    b.quant_bits,
                    b.moe_mxfp4
                );
            }
            rows += 1;
        }
        assert!(
            rows > 10,
            "only {rows} rows were enumerated, so this gate is not reading \
             the catalog"
        );
    }

    /// A refusal reaches the operator as the ROW's own sentence.
    ///
    /// The last hop of the path this module exists to build. `serve/load.rs`
    /// refuses at the door and `serve/launch.rs` propagates at the fire, and
    /// both do it by `Error::from`, so what an operator reads is what the row
    /// said — `Refusal::Unsupported`'s `Display` — rather than a paraphrase
    /// this driver invented from a lookup that failed.
    ///
    /// The `what` matters as much as the message and is asserted separately:
    /// `"deployment unsupported"` is a statement about the BUILD, and an
    /// operator's next move for it (run a pie built with that text) is a
    /// different move than for a malformed checkpoint. The deleted
    /// `plan_for` call site flattened every refusal into `"no text for this
    /// fire: {why:?}"`, which is a `Debug` of an error the caller then could
    /// not classify.
    #[test]
    fn a_refused_row_reaches_the_operator_as_the_rows_own_sentence() {
        let refusal = model::deployment::Refusal::Unsupported("no Metal text for this row");
        let crate::error::Error::Unserved { what, message } = crate::error::Error::from(refusal)
        else {
            panic!("a refusal is an Unserved, which is what the boundary reports");
        };
        assert_eq!(what, "deployment unsupported");
        assert!(
            message.contains("no Metal text for this row"),
            "the row's own words must survive the conversion; got `{message}`"
        );
    }

    /// The text is the ROW's, and it is a METAL text.
    ///
    /// Two claims in one test, and the second is the one with teeth. [`text`]
    /// exists to be the one door and not a translation, so equality with the
    /// row's own answer is the first. The second reads the family name:
    /// `Backend::of_family` parts `llama_like.metal.decode` from
    /// `llama_like.cuda.decode`, which is one call's difference inside the
    /// generation's own `match load.backend` — so a Metal arm that reached
    /// for the neighbouring CUDA builder fails here rather than at a missing
    /// symbol on a device nobody in CI has.
    #[test]
    fn the_plan_is_the_rows_own_answer_and_it_is_the_metal_one() {
        use model_compiler::kernels::Backend;

        let Some(row) = model::catalog::find("qwen3-0.6b") else {
            panic!("`qwen3-0.6b` is the row the device gates open");
        };
        let b = observed(AffineFormat { bits: 4, group: 64 }, |_| false);
        for class in [FireClass::Decode, FireClass::Prefill] {
            let mine = text(row, class, &b).expect("a metal text for a served row");
            let theirs = row
                .trace(class, model::catalog::Deployed::metal(&b))
                .expect("the row's own answer");
            assert_eq!(mine, theirs, "the door adds nothing to the row's text");
            assert!(!mine.ops.is_empty(), "a text of no operations is not a text");
            assert_eq!(
                Backend::of_family(&mine.family),
                Some(Backend::Metal),
                "`{}` answered a METAL load with `{}`. The backend is a \
                 parameter of the question now — if this fails, the row is \
                 still tracing one backend's text for both, and this driver \
                 would be lowering CUDA symbols against Metal pipelines",
                row.id(),
                mine.family
            );
        }
    }

    /// A row whose answer is not this backend's is REFUSED, not served.
    ///
    /// The half of [`text`] that is not a pass-through, asserted over the
    /// whole catalog in both directions: every row that states a Metal text
    /// is served, and every row that does not is refused. The driver holds no
    /// list, so this is the only thing that can be checked — and it is
    /// exactly the thing the deleted `LLAMA_LIKE` got wrong in both
    /// directions at once. It listed `gemma4`, which no Metal text has ever
    /// existed for, and omitted `gemma3`, which `llama_like_metal` serves
    /// today: sandwich norms, the `(1+w)` weight fold, the scaled gather, two
    /// rope bases and the per-layer window are all in that text. A hand-kept
    /// table drifts in whichever direction nobody is looking.
    ///
    /// Eleven generations refuse in their own words — `qwen_3_5` because its
    /// linear-attention interleave is a different shape, `gpt_oss` and
    /// `gemma_4` because their texts are `gpt_oss_cuda` and `gemma4_cuda` —
    /// and `serve/control.rs`'s `copy_state` refusal still rests on the first
    /// of those: no model this backend serves has recurrent state to copy.
    /// That sentence is true only while this passes.
    #[test]
    fn a_row_that_answers_for_another_backend_is_not_served() {
        use model_compiler::kernels::Backend;

        let mut served = 0usize;
        let mut refused = 0usize;
        for row in model::catalog::catalog() {
            let raw = row.trace(FireClass::Decode, model::catalog::Deployed::metal(&ANY_ENCODING));
            let ours = serves(*row).is_ok();
            match raw {
                Ok(plan) if Backend::of_family(&plan.family) == Some(Backend::Metal) => {
                    served += 1;
                    assert!(
                        ours,
                        "`{}` states a Metal text (`{}`) and this driver \
                         refuses it. The driver holds no list of its own — if \
                         it is refusing a text that exists, the door has grown \
                         an opinion",
                        row.id(),
                        plan.family
                    );
                }
                Ok(plan) => {
                    refused += 1;
                    assert!(
                        !ours,
                        "`{}` answered a Metal load with `{}` and this driver \
                         SERVED it. `load_model` would stage its weights, \
                         admit a request, and fault at the first dispatch on a \
                         symbol no Metal shader exports",
                        row.id(),
                        plan.family
                    );
                }
                Err(_) => {
                    refused += 1;
                    assert!(!ours, "`{}` refused `trace` and was served anyway", row.id());
                }
            }
        }
        assert!(served > 0, "no row in the catalog states a Metal text at all");
        assert!(
            refused > 0,
            "every row in the catalog states a Metal text, so the pre-staging \
             refusal in `serve/load.rs` has nothing left to refuse. That is a \
             good day and not a failure — but check it before relaxing this, \
             because `serve/control.rs`'s `copy_state` refusal is written on \
             the assumption that the recurrent generations are among the \
             refused"
        );
    }

    /// An ENCODING is not a model fact, measured on the plan itself.
    ///
    /// Two loads of one row that differ ONLY in the bytes their weights
    /// arrived as must state the same PROGRAM: the same operations, in the
    /// same order, over the same values, with the same dataflow between
    /// them. What may differ is which instantiation of a kernel each names —
    /// `_gs_64_b_4` against `_gs_128_b_8` — and that is the whole of what an
    /// affine point is, which is why the kernel strings are deliberately not
    /// compared.
    ///
    /// This is the property the deleted `facts_from_with` could not have.
    /// It built the model's facts and the encoding's facts in ONE pass over
    /// one `DecodeGeometry`, so nothing separated them and nothing could
    /// have checked that they were separate — which is how a `norm_variant`
    /// came to be decided by which norms a checkpoint shipped.
    #[test]
    fn two_encodings_of_one_row_state_the_same_program() {
        let row = model::catalog::find("qwen3-0.6b").expect("a row the device gates open");
        let four = observed(AffineFormat { bits: 4, group: 64 }, |_| false);
        let eight = observed(AffineFormat { bits: 8, group: 128 }, |_| false);
        let (a, b) = (
            text(row, FireClass::Decode, &four).expect("a metal text"),
            text(row, FireClass::Decode, &eight).expect("a metal text"),
        );
        assert_eq!(a.family, b.family, "an encoding does not change the family");
        assert_eq!(
            a.values, b.values,
            "an encoding changed a value's shape or dtype, which makes it a \
             model fact — and a model fact belongs to the row"
        );
        assert_eq!(
            a.ops.len(),
            b.ops.len(),
            "an encoding added or removed an operation"
        );
        for (i, (x, y)) in a.ops.iter().zip(&b.ops).enumerate() {
            assert_eq!(
                std::mem::discriminant(&x.kind),
                std::mem::discriminant(&y.kind),
                "op {i} became a different kind of operation when only the \
                 bytes changed"
            );
            assert_eq!(x.layer, y.layer, "op {i} moved layers");
            assert_eq!(x.inputs, y.inputs, "op {i} reads different values");
            assert_eq!(x.outputs, y.outputs, "op {i} writes different values");
        }
    }
}
