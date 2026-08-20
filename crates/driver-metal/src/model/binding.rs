//! What a Metal load observed, and the one door to the row's Metal text.
//!
//! [`MetalBinding`] is six values, and every one of them is about the BYTES
//! that arrived or about the KERNELS this binary was built with. None is
//! about the model:
//!
//! * the affine group and bit width, because a model may be published at
//!   more than one affine point that packs to identical extents — no tensor
//!   can be asked which it is, and no row may be split four ways to record it;
//! * whether the expert bank reached the device still in MXFP4, which is a
//!   statement about what the LOADER did and is only knowable after staging;
//! * and the three kernel capabilities, which are facts about
//!   `crates/kernels-metal` as compiled into this binary.
//!
//! [`observed`] takes an [`AffineFormat`](crate::batch::AffineFormat) and a
//! single tensor probe: it cannot see a `DecodeGeometry`, so it cannot
//! smuggle a model fact into a binding.

use model::catalog::MetalBinding;
use model_ir::trace::{FireClass, ForwardPlan};

/// The tensor probed to learn the expert bank's own affine encoding.
///
/// `mlx-community/gpt-oss-20b-MXFP4-Q4` names 98 tensors as affine/64/4 and
/// leaves the expert banks out, so they take the top-level default, mxfp4/32.
/// Its `quantization` block holds 122 entries and not 98: the other 24 are
/// the `mlp.router` gates at 64/**8**, which is the SECOND tensor below and
/// the reason there is a second one.
/// A checkpoint need not quantize uniformly, and reading a bank with the
/// dense format is 909,207 NaNs rather than a near miss — the first of them
/// in `affine_qmv_routed_bfloat16_gs_64_b_4`, layer 0.
///
/// The 98 is a count that stops one entry short. The block carries 122
/// overrides; the other 24 are the `mlp.router` gates at affine/64/**8**, and
/// [`ROUTER_GATES`] holds the name that asks after those. Theirs are the only
/// entries in the block with no `mode` key, which is why every independent
/// copy of this count stopped in the same place.
///
/// It is named as a constant rather than spelled at the call site so that the
/// claim "two probes, and each decides an encoding" is something a test can
/// check by reading this file. Whether the model HAS experts is the row's
/// answer and is not asked here; this asks only what format the bytes for
/// them arrived in.
pub const EXPERT_BANK: &str = "layers.0.mlp.experts.gate_proj.weight";

/// The ROUTER GATES this load asks the affine point of.
///
/// The second and last question this driver puts to a checkpoint, and it asks
/// the same KIND of thing [`EXPERT_BANK`] does: what format did the bytes for
/// this tensor arrive in. Whether the model HAS a router is the row's answer
/// and is not asked here.
///
/// THREE SPELLINGS, because a router is not always a member of the mlp and
/// is not always called a router:
///
/// - `layers.0.mlp.router.weight` — gpt-oss, whose routed block replaces the
///   dense one, so the router hangs where the mlp was.
/// - `layers.0.router.proj.weight` — gemma-4, whose routed block sits BESIDE
///   a dense one that is still there, so the router is a sibling of both and
///   is a module with a `proj` inside it rather than a bare weight.
/// - `layers.0.mlp.gate.weight` — qwen3.6, which calls the thing a GATE.
///   `mlx-community` publishes it and `mlp.shared_expert_gate` at eight bits
///   inside a four-bit stack, for all forty layers, and this list not having
///   the name refused the whole checkpoint as *"2 affine points beside its
///   router gate's"* with one of the two being that gate — the same failure
///   gemma-4 had, under the third spelling.
///
/// A list rather than a suffix rule for the reason
/// [`ROUTER_GATE_AT_ANY_LAYER`] gives at length: these names are not nested,
/// `layers.0.router.proj.weight` does not end with `.router.weight`, and a
/// constant that assumed it did would match nothing and refuse silently.
/// That is exactly what happened — `gemma-4-26b-a4b` was refused as
/// *"2 affine points beside its router gate's"* while one of the two WAS the
/// router gate, under the spelling this list did not have.
///
/// It is a list of constants rather than names spelled at the call sites so
/// that "which tensors does the load ask about" stays answerable by reading
/// one file, and `no_probe_decides_a_fact` is the test that keeps it so.
pub const ROUTER_GATES: &[&str] = &[
    "layers.0.mlp.router.weight",
    "layers.0.router.proj.weight",
    "layers.0.mlp.gate.weight",
];

/// The affine point of whichever router gate this checkpoint spells.
///
/// `None` when the checkpoint has no router at all, which is every dense row
/// and is not an error: the callers both treat "no gate" and "a gate at the
/// stack's own point" the same way, because they mean the same thing.
pub fn router_point(
    point_of: impl Fn(&str) -> Option<crate::batch::AffineFormat>,
) -> Option<crate::batch::AffineFormat> {
    ROUTER_GATES.iter().find_map(|n| point_of(n))
}

/// The router gate as a STATEMENT names it, at any layer.
///
/// Not a suffix of any [`ROUTER_GATES`] entry, and the difference is the
/// point. This
/// tensor has three spellings in this tree and they are not nested:
///
/// - `layers.0.mlp.router.weight` — the CHECKPOINT name, which is what
///   [`ROUTER_GATES`] lists and what a load plan indexes tensors by.
/// - `layer.{n}.mlp.router` — the tensor SPEC name, which `gpt_oss::project`
///   declares the contract with.
/// - `layer.{n}.router` — the name a lowered STATEMENT carries, which is the
///   only one `lowering::dispatch` ever sees.
///
/// So a constant that assumed one contained another would match nothing and
/// refuse silently, which is what the first two attempts at this did.
///
/// `lowering::dispatch` is what puts it to a statement: a checkpoint whose
/// gates arrived at their own affine point projects them at that point, and
/// the number reaches the routine through `Geometry::router_bits`. The
/// statement also carries `layer.{n}.router.scales` and `.zeros`, which are
/// the same tensor's companions and match too — harmless, because they answer
/// the same question about the same weight.
pub const ROUTER_GATE_AT_ANY_LAYER: &str = ".router";

/// Every statement-side spelling of a gate that arrived at the router's
/// affine point.
///
/// [`ROUTER_GATE_AT_ANY_LAYER`] is one entry of it and the reason there are
/// two is qwen3.6: `mlx-community` lists `mlp.gate` AND
/// `mlp.shared_expert_gate` at eight bits for all forty layers, and the text
/// names the second `layer.{n}.shared_gate_proj`, which does not end with
/// `.router` under any reading. A statement matched by neither is composed at
/// the STACK's width, so the trace asks for `_gs_64_b_8` and the body builds
/// `_gs_64_b_4` -- caught here as `Misspelled`, and the reason that check
/// exists is that the two pack to identical extents and would otherwise have
/// returned fluent nonsense.
///
/// The shared expert's gate is a projection to ONE row -- a scalar per token
/// that scales the dense expert's contribution -- so reading it at the wrong
/// width does not fault either: it scales by the wrong number.
pub const ROUTER_POINT_AT_ANY_LAYER: &[&str] = &[ROUTER_GATE_AT_ANY_LAYER, ".shared_gate_proj"];

/// What this build's kernels can do, at any encoding: the projection GEMV
/// folds the block residual in its epilogue, the M>1 lane addresses the KV
/// cache through a page table, and M>1 projections use the quantized GEMM.
/// A `const fn` so [`ANY_ENCODING`] and [`observed`] cannot disagree about a
/// build capability.
#[must_use]
pub const fn build_kernels(quant_group: u32, quant_bits: u32, moe_mxfp4: bool) -> MetalBinding {
    build_kernels_at(quant_group, quant_bits, 0, 0, moe_mxfp4)
}

/// [`build_kernels`], for a checkpoint whose router gate has its own affine
/// point; `(0, 0)` means "the same as the dense projections".
#[must_use]
pub const fn build_kernels_at(
    quant_group: u32,
    quant_bits: u32,
    router_quant_group: u32,
    router_quant_bits: u32,
    moe_mxfp4: bool,
) -> MetalBinding {
    MetalBinding {
        qmm_partial_rows: false,
        qmm_tile: None,
        quant_group,
        quant_bits,
        router_quant_group,
        router_quant_bits,
        moe_mxfp4,
        fuse_residual_gemv: true,
        paged_multi_batch: true,
        qmm_multi_batch: true,
        // A statement about this binary, not the model: it requires
        // `lowering::dispatch` to resolve every scalar source the table
        // names, or a kernel's width argument silently gets whatever the
        // encoder last left at that index.
        add_bias: true,
        // FALSE: there is no Metal `rms_rope`. The name resolves through this
        // backend's census so that `model-ir` can check a VULKAN text -- which
        // consumes the metal-flavoured plan -- and there is no shader behind
        // it here.
        fused_qk_rope: false,
    }
}

/// The binding to probe "is there a Metal text at all" before any weights
/// are staged: `moe_mxfp4` cannot be known until the tensors are on the
/// device, but a row that refuses Metal refuses it at every encoding, so
/// g64/b4 stands in for all of them.
/// `a_row_is_served_the_same_way_at_every_encoding` asserts this over the
/// whole catalog.
pub const ANY_ENCODING: MetalBinding = build_kernels(64, 4, false);

/// What THIS load observed, once its weights are on the device.
///
/// `quant` is the resolved affine point: `geometry_from_deployment` has
/// already replaced an unset format with the one a shader is instantiated
/// at, so `gs_0_b_0` never appears here. `is_mxfp4` is asked about exactly
/// one name, [`EXPERT_BANK`].
#[must_use]
pub fn observed(
    quant: crate::batch::AffineFormat,
    point_of: impl Fn(&str) -> Option<crate::batch::AffineFormat>,
    is_mxfp4: impl Fn(&str) -> bool,
) -> MetalBinding {
    // The gate's point, and ONLY when it differs. Equal is not "no second
    // point", it is the same point twice, and stating it would put a
    // `Some(repr)` on the text where `None` means the same thing -- two
    // spellings of one encoding, which is how a text stops being comparable
    // to itself across two checkpoints of the same row.
    let router = router_point(&point_of)
        .filter(|p| *p != quant)
        .unwrap_or(crate::batch::AffineFormat { bits: 0, group: 0 });
    build_kernels_at(
        quant.group,
        quant.bits,
        router.group,
        router.bits,
        is_mxfp4(EXPERT_BANK),
    )
}

/// A row that answered a Metal load with another backend's text — never
/// produced today, since every generation states its own Metal answer or
/// refuses in its own words.
const NOT_A_METAL_TEXT: &str = "this row answered a Metal load with another backend's text; the text \
     exists but it was written for other kernels, so a fire would lower \
     symbols no Metal shader exports";

/// The row's Metal text for one fire class.
///
/// The only place this driver reaches a forward text, so "which text does
/// this checkpoint run" has exactly one answer site, and that answer is a
/// row's.
///
/// # The one thing it checks, and why it is not a second list
///
/// `Deployed::backend` is a request, and the answer is checked against it:
/// `Backend::of_family` reads the backend the trace stamped on itself, so a
/// generation whose Metal arm mistakenly reaches for another backend's
/// builder is refused here rather than failing at a missing shader on the
/// device. This driver holds no list of architectures of its own to check
/// first — `catalog_backends`'s test enumerates which generations serve
/// Metal and must be kept in sync with reality.
///
/// # Errors
///
/// [`Refusal::Unsupported`](model::deployment::Refusal) when this build has
/// no Metal text for the row — almost always the row's own refusal, carried
/// out unchanged.
pub fn text(
    row: &dyn model::catalog::Variant,
    class: FireClass,
    binding: &MetalBinding,
) -> Result<ForwardPlan, model::deployment::Refusal> {
    let plan = row.trace(class, model::catalog::Deployed::metal(binding))?;
    if model_ir::kernels::Backend::of_family(&plan.family)
        != Some(model_ir::kernels::Backend::Metal)
    {
        return Err(model::deployment::Refusal::Unsupported(NOT_A_METAL_TEXT));
    }
    Ok(plan)
}

/// Whether this build has a Metal text for `row`, asked without the weights.
///
/// Uses [`ANY_ENCODING`] and `FireClass::Decode`, since every text states a
/// decode.
///
/// # Errors
///
/// The row's own [`Refusal`](model::deployment::Refusal), carried unchanged
/// — or [`text`]'s, when the row answered for a backend that is not this one.
pub fn serves(row: &dyn model::catalog::Variant) -> Result<(), model::deployment::Refusal> {
    text(row, FireClass::Decode, &ANY_ENCODING).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::AffineFormat;

    /// Either router spelling reaches the binding, at its own point.
    ///
    /// gemma-4-26b-a4b was refused as *"2 affine points beside its router
    /// gate's"* while one of those two points WAS its router gate's -- the
    /// probe asked only for gpt-oss's `layers.0.mlp.router.weight` and
    /// gemma-4 ships `layers.0.router.proj.weight`. The count that refuses
    /// subtracts what this returns, so a spelling missing here is a
    /// checkpoint refused for having the second point this driver was built
    /// to carry.
    #[test]
    fn either_router_spelling_answers_the_probe() {
        for name in ROUTER_GATES {
            let asked = std::cell::RefCell::new(Vec::new());
            let b = observed(
                AffineFormat { bits: 4, group: 64 },
                |t| {
                    asked.borrow_mut().push(t.to_string());
                    (t == *name).then_some(AffineFormat { bits: 8, group: 64 })
                },
                |_| false,
            );
            assert_eq!(
                (b.router_quant_group, b.router_quant_bits),
                (64, 8),
                "the gate at `{name}` arrived at g64/b8 and the binding did \
                 not carry it; the load asked {:?}",
                asked.into_inner()
            );
        }
    }

    /// A gate at the stack's own point states nothing, under either spelling.
    #[test]
    fn a_router_that_shares_the_stacks_point_is_not_a_second_one() {
        for name in ROUTER_GATES {
            let b = observed(
                AffineFormat { bits: 4, group: 64 },
                |t| (t == *name).then_some(AffineFormat { bits: 4, group: 64 }),
                |_| false,
            );
            assert_eq!(
                (b.router_quant_group, b.router_quant_bits),
                (0, 0),
                "`{name}` at the stack's own point is the same point twice"
            );
        }
    }

    /// The checkpoint's own affine point reaches the binding, both numbers.
    ///
    /// Neither is inferable from a tensor: g64/b8 and g128/b4 pack to
    /// identical extents, so a pipeline built for the wrong point returns
    /// fluent nonsense rather than failing. That is the whole reason an
    /// encoding is asked of the LOAD and the whole reason a row must not be
    /// split four ways to hold one.
    #[test]
    fn the_binding_carries_the_checkpoints_own_affine_point() {
        let b = observed(
            AffineFormat {
                bits: 8,
                group: 128,
            },
            |_| None,
            |_| false,
        );
        assert_eq!(b.quant_bits, 8);
        assert_eq!(b.quant_group, 128);

        let g64 = observed(AffineFormat { bits: 4, group: 64 }, |_| None, |_| false);
        assert_eq!(g64.quant_bits, 4);
        assert_eq!(g64.quant_group, 64);
    }

    /// Exactly one tensor is probed, and it is the expert bank; a
    /// regression would show up here as a second name being asked for.
    #[test]
    fn exactly_one_tensor_is_asked_about_and_it_decides_an_encoding() {
        let asked = std::cell::RefCell::new(Vec::new());
        let b = observed(
            AffineFormat { bits: 4, group: 64 },
            |_| None,
            |name| {
                asked.borrow_mut().push(name.to_string());
                name == EXPERT_BANK
            },
        );
        assert_eq!(
            asked.into_inner(),
            vec![EXPERT_BANK.to_string()],
            "the load's only question of the tensors is what format the \
             expert bank arrived in"
        );
        assert!(
            b.moe_mxfp4,
            "the bank answered mxfp4, so the binding says so"
        );

        let dense = observed(AffineFormat { bits: 4, group: 64 }, |_| None, |_| false);
        assert!(
            !dense.moe_mxfp4,
            "a checkpoint whose bank was transcoded reads as the dense format"
        );
    }

    /// The three kernel capabilities are the BUILD's, so the probe binding
    /// and a real one cannot disagree about them.
    #[test]
    fn the_build_capabilities_do_not_move_with_the_encoding() {
        let real = observed(AffineFormat { bits: 8, group: 32 }, |_| None, |_| true);
        assert_eq!(real.fuse_residual_gemv, ANY_ENCODING.fuse_residual_gemv);
        assert_eq!(real.paged_multi_batch, ANY_ENCODING.paged_multi_batch);
        assert_eq!(real.qmm_multi_batch, ANY_ENCODING.qmm_multi_batch);
    }

    /// A row refused at the probe binding is refused at every real encoding
    /// — asserted over the whole catalog, since `serves` is asked before a
    /// byte is staged and `moe_mxfp4` cannot yet be known.
    ///
    /// The converse is not asserted and is false: a routed row served at the
    /// probe can still be refused at a real encoding, since
    /// `affine_qmv_routed` is templated at group 64 alone. `serves` may be
    /// permissive — a later refusal costs a load, not a wrong answer — but
    /// it may never under-refuse.
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

    /// A refusal reaches the operator as the ROW's own sentence:
    /// `serve/load.rs` and `serve/launch.rs` both propagate it by
    /// `Error::from`, so `"deployment unsupported"` and the row's own
    /// message survive the conversion unchanged.
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

    /// The plan is the row's own answer, and it names a METAL family:
    /// [`text`] must add nothing to the row's answer, and `Backend::of_family`
    /// must read `Metal` from it — otherwise a Metal arm that reached for a
    /// CUDA builder would only fail at a missing symbol on the device.
    #[test]
    fn the_plan_is_the_rows_own_answer_and_it_is_the_metal_one() {
        use model_ir::kernels::Backend;

        let Some(row) = model::catalog::find("qwen3-0.6b") else {
            panic!("`qwen3-0.6b` is the row the device gates open");
        };
        let b = observed(AffineFormat { bits: 4, group: 64 }, |_| None, |_| false);
        for class in [FireClass::Decode, FireClass::Prefill] {
            let mine = text(row, class, &b).expect("a metal text for a served row");
            let theirs = row
                .trace(class, model::catalog::Deployed::metal(&b))
                .expect("the row's own answer");
            assert_eq!(mine, theirs, "the door adds nothing to the row's text");
            assert!(
                !mine.ops.is_empty(),
                "a text of no operations is not a text"
            );
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

    /// A row whose answer is not this backend's is refused, not served —
    /// asserted over the whole catalog in both directions, since the driver
    /// keeps no list of its own to check against.
    #[test]
    fn a_row_that_answers_for_another_backend_is_not_served() {
        use model_ir::kernels::Backend;

        let mut served = 0usize;
        let mut refused = 0usize;
        for row in model::catalog::catalog() {
            let raw = row.trace(
                FireClass::Decode,
                model::catalog::Deployed::metal(&ANY_ENCODING),
            );
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
                    assert!(
                        !ours,
                        "`{}` refused `trace` and was served anyway",
                        row.id()
                    );
                }
            }
        }
        assert!(
            served > 0,
            "no row in the catalog states a Metal text at all"
        );
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

    /// An encoding is not a model fact, measured on the plan itself: two
    /// loads of one row that differ only in the bytes their weights arrived
    /// as must state the same program — same operations, order, values and
    /// dataflow. Only the kernel instantiation may differ, so kernel
    /// strings are deliberately not compared.
    #[test]
    fn two_encodings_of_one_row_state_the_same_program() {
        let row = model::catalog::find("qwen3-0.6b").expect("a row the device gates open");
        let four = observed(AffineFormat { bits: 4, group: 64 }, |_| None, |_| false);
        let eight = observed(
            AffineFormat {
                bits: 8,
                group: 128,
            },
            |_| None,
            |_| false,
        );
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
