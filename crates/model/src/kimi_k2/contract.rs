//! What the MLA families bind.
//!
//! Ported from `crates/driver-cuda/csrc/src/model/kimi/kimi_contract.hpp`. DeepSeek-V2/V3
//! and Kimi-K2 share a binder and a forward. They differ in the contract:
//! Kimi hides the decoder under `language_model.` and wants `embed_tokens`
//! sharded and `lm_head` replicated, which is a memory trade the driver
//! makes and the checkpoint knows nothing about.

use model_loader::contract::Expr;
use model_loader::error::Error;
use model_loader::types::{DType, Encoding};

use crate::shared::builder::{Builder, int4b8_encoding, is_raw};

fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

/// deepseek_v2, deepseek_v3.
pub fn author_deepseek_mla(b: &mut Builder<'_>) -> Result<(), Error> {
    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    mla_fused_projection_joins(b)?;
    b.publish_remaining()
}

/// kimi_k2. Keeping `lm_head` whole costs ~1.7 GB a rank and buys not
/// needing a TP greedy argmax on the logits path.
///
/// The bf16 expert slabs are additive on top of the packed W4A16 experts,
/// which the per-step GEMV path still reads, so they are only published when
/// the whole model's worth fits in the budget.
pub fn author_kimi(b: &mut Builder<'_>) -> Result<(), Error> {
    b.source_prefix("language_model.");
    b.shard_embed_tokens();
    b.replicate_lm_head();
    bf16_expert_stacks(b, /*budget=*/ 4u64 << 30)?;
    author_deepseek_mla(b)
}

/// Fuse the two MLA projection pairs that share an input.
fn mla_fused_projection_joins(b: &mut Builder<'_>) -> Result<(), Error> {
    let mut candidates = Vec::new();
    for layer in 0..b.shape().layers {
        let p = format!("model.layers.{layer}.");
        let s = b.source_name(&p);
        // q_a_proj + kv_a_proj_with_mqa share an input (norm_x, unsharded).
        if let Some(candidate) = b.fused_join_candidate(
            format!("{p}self_attn.q_kv_a_proj.fused.weight"),
            &[
                format!("{s}self_attn.q_a_proj.weight"),
                format!("{s}self_attn.kv_a_proj_with_mqa.weight"),
            ],
        ) {
            candidates.push(candidate);
        }
        // Shared gate + up share an input (norm_y).
        if let Some(candidate) = b.fused_join_candidate(
            format!("{p}mlp.shared_experts.gate_up_proj.fused.weight"),
            &[
                format!("{s}mlp.shared_experts.gate_proj.weight"),
                format!("{s}mlp.shared_experts.up_proj.weight"),
            ],
        ) {
            candidates.push(candidate);
        }
    }
    b.publish_fused(candidates)
}

/// Dequantize Kimi's routed experts and stack them, at load time.
///
/// The checkpoint stores each expert separately as W4A16: `weight_packed` is
/// `I32 [out, in/8]` holding eight 4-bit codes per word, low nibble first,
/// and `weight_scale` is `BF16 [out, in/32]`. An element is `code - 8` times
/// its group's factor, which is what `QuantScheme::Int4B8` names — the bias
/// is the scheme's, not a zero-point tensor's.
///
/// The batched MoE path wants one dense bf16 slab per layer, `[E, 2I, H]`
/// over `[E, H, I]`. Building it is `Transmute` to say what the packed words
/// are, `Concat` to stack, and one `scale_per_block` per slab to dequantize
/// — with the sharding inside, so each rank dequantizes only the slice it
/// keeps.
///
/// Unlike DeepSeek-V4 this does **not** consume the packed originals. Kimi
/// picks between the two forms per step: the W4A16 GEMVs win below a token
/// count the batched path's 4x weight traffic cannot amortize, so both have
/// to be resident, and the stacks are additive. That is what `budget` is
/// for.
fn bf16_expert_stacks(b: &mut Builder<'_>, budget: u64) -> Result<(), Error> {
    const GROUP: i64 = 32;
    const CODES_PER_WORD: i64 = 8;

    for layer in 0..b.shape().layers {
        let mlp = format!("model.layers.{layer}.mlp.");
        let mut gate_up = Vec::new();
        let mut gate_up_scales = Vec::new();
        let mut down = Vec::new();
        let mut down_scales = Vec::new();
        let mut local_inter = 0i64;
        let mut hidden = 0i64;

        let mut expert = 0u32;
        loop {
            let ep = format!("{mlp}experts.{expert}.");
            if b.find(&b.source_name(&format!("{ep}gate_proj.weight_packed")))
                .is_none()
            {
                break;
            }
            let names = [
                format!("{ep}gate_proj.weight_packed"),
                format!("{ep}gate_proj.weight_scale"),
                format!("{ep}up_proj.weight_packed"),
                format!("{ep}up_proj.weight_scale"),
                format!("{ep}down_proj.weight_packed"),
                format!("{ep}down_proj.weight_scale"),
            ];
            let mut parts = Vec::with_capacity(6);
            for name in &names {
                let Some(part) = b.find(&b.source_name(name)) else {
                    return fail(format!(
                        "kimi expert stack: {ep} is missing a weight or scale"
                    ));
                };
                parts.push(part);
            }
            // Eight codes to a 32-bit word, and BF16 factors. A checkpoint
            // that stores its experts some other way is not this pass's to
            // rewrite, and stacking it anyway would hand the GEMM garbage —
            // so leave the whole checkpoint alone rather than half of it,
            // and let the W4A16 path run off the packed weights as before.
            if [0usize, 2, 4]
                .iter()
                .any(|&i| !is_raw(&parts[i].encoding, DType::I32))
            {
                return Ok(());
            }
            if [1usize, 3, 5]
                .iter()
                .any(|&i| !is_raw(&parts[i].encoding, DType::BF16))
            {
                return Ok(());
            }

            let up_raw = &parts[0].shape;
            let down_raw = &parts[4].shape;
            if up_raw.len() != 2 || down_raw.len() != 2 {
                return fail(format!(
                    "kimi expert stack: {ep} expects rank-2 expert weights"
                ));
            }
            let inter_full = up_raw[0];
            let h = up_raw[1] * CODES_PER_WORD;
            let inter = b.local_extent(inter_full);
            if h % GROUP != 0 || inter % GROUP != 0 {
                return fail(format!(
                    "kimi expert stack: {ep} expects both expert dims to be a multiple of 32"
                ));
            }
            if local_inter != 0 && (inter != local_inter || h != hidden) {
                return fail(format!(
                    "kimi expert stack: {ep} disagrees with its siblings on shape"
                ));
            }
            local_inter = inter;
            hidden = h;

            // Every leg is declared rank 3 with a leading 1, so that the
            // outer concatenation over axis 0 is a stack. The transmute
            // carries the rank lift as well as the unpacking, because
            // reshaping a packed tensor is not meaningful: its byte layout
            // is a function of the shape it was packed for.
            //
            // `gate`/`up` shard the out dim and `down` the in dim, the same
            // split the packed tensors take, applied to the logical shapes.
            let packed = |b: &Builder<'_>, name: &str, shape: Vec<i64>, axis: u8| {
                b.shard(
                    Expr::src(name).transmute(model_loader::contract::TensorType::new(
                        shape.clone(),
                        int4b8_encoding(2),
                    )),
                    shape,
                    Some(axis),
                )
                .0
            };
            let factors = |b: &Builder<'_>, name: &str, shape: Vec<i64>, axis: u8| {
                b.shard(
                    Expr::src(name).transmute(model_loader::contract::TensorType::new(
                        shape.clone(),
                        Encoding::Raw(DType::BF16),
                    )),
                    shape,
                    Some(axis),
                )
                .0
            };

            let gate = packed(b, &parts[0].name, vec![1, inter_full, h], 1);
            let up = packed(b, &parts[2].name, vec![1, inter_full, h], 1);
            let gate_s = factors(b, &parts[1].name, vec![1, inter_full, h / GROUP], 1);
            let up_s = factors(b, &parts[3].name, vec![1, inter_full, h / GROUP], 1);
            // `[gate | up]` — the order `mlp::chunked_swiglu_bf16` reads
            // when its `gate_second` argument is left at its default, which
            // is what the trace's `dsl::cuda::swiglu` records: it states no
            // params, so the launch takes the default. `y[n,i] =
            // silu(packed[n,i]) * packed[n,I+i]`, so the LEADING half is
            // the one that gets the silu. Swapping the legs here without
            // teaching the trace to pass `gate_second` would silu the up
            // projection and multiply by the gate — no shape error, no NaN,
            // a different model. Undoing it later would cost a copy of the
            // whole slab, so it is decided here, once.
            gate_up.push(Expr::concat(1, vec![gate, up]));
            gate_up_scales.push(Expr::concat(1, vec![gate_s, up_s]));
            down.push(packed(
                b,
                &parts[4].name,
                vec![1, down_raw[0], inter_full],
                2,
            ));
            down_scales.push(factors(
                b,
                &parts[5].name,
                vec![1, down_raw[0], inter_full / GROUP],
                2,
            ));
            expert += 1;
        }
        // Kimi's leading layers are dense, and a dense layer simply has no
        // expert names — which is why the pass walks rather than reading
        // `first_k_dense_replace`. The walk finding nothing at expert 0 IS
        // that answer; probing for expert 0 before the walk would ask the
        // same question twice and make this line unreachable.
        if gate_up.is_empty() {
            continue;
        }
        let experts = gate_up.len() as i64;

        // The loop's probe is `experts.{n}.gate_proj.weight_packed`, so a
        // checkpoint missing exactly that one name does not look like a
        // hole — it looks like the END of the bank, and the loop stops
        // there. Every other missing part is refused by name above; this
        // one would silently stack a SHORTER slab.
        //
        // Nothing downstream catches it. The manifest measures the
        // ROUTER, `[num_experts, hidden]`, and a checkpoint with a whole
        // expert missing still carries a full-width router — so the row
        // matches, the load succeeds, and the grouped GEMM indexes a slab
        // with fewer experts than the router emits indices for. Experts
        // are not sharded here (only `inter` is, through `local_extent`),
        // so the row's count is the count this slab must have.
        if experts != i64::from(b.shape().n_experts) {
            return fail(format!(
                "kimi expert stack: layer {layer} stacked {experts} experts but the \
                 row states {}; the router emits indices this slab has no rows for",
                b.shape().n_experts
            ));
        }

        // Both forms stay resident, so a model whose slabs do not fit is
        // better off with only the packed one: the W4A16 GEMVs are slower
        // per token but they are not a fallback, they are the path that wins
        // below the crossover. Checked per layer against the whole model's
        // cost, so the answer cannot come out different for different
        // layers.
        let slab_bytes = (experts as u64)
            * 3
            * (local_inter as u64)
            * (hidden as u64)
            * 2
            * u64::from(b.shape().layers);
        if slab_bytes > budget {
            return Ok(());
        }

        // Named but not bound. `scale_per_block` takes its factors by output
        // name — a scale is a tensor the contract declared, not a companion
        // the lowering guesses at from a suffix — and the stacked slab is
        // dequantized here, so no kernel ever reads these again. Left public
        // they would sit in the persistent arena for the process's lifetime
        // and appear in the driver's bind table under a name nothing asks
        // for.
        let gu_scale = format!("{mlp}experts.gate_up.scale");
        let dn_scale = format!("{mlp}experts.down.scale");
        let gu = b.define(
            gu_scale.clone(),
            Expr::concat(0, gate_up_scales),
            Encoding::Raw(DType::BF16),
            Some(vec![experts, 2 * local_inter, hidden / GROUP]),
        );
        b.mark_internal(gu);
        let dn = b.define(
            dn_scale.clone(),
            Expr::concat(0, down_scales),
            Encoding::Raw(DType::BF16),
            Some(vec![experts, hidden, local_inter / GROUP]),
        );
        b.mark_internal(dn);
        b.define(
            format!("{mlp}experts.gate_up.weight"),
            Expr::concat(0, gate_up).scale_per_block(Expr::out(&gu_scale)),
            Encoding::Raw(DType::BF16),
            Some(vec![experts, 2 * local_inter, hidden]),
        );
        b.define(
            format!("{mlp}experts.down.weight"),
            Expr::concat(0, down).scale_per_block(Expr::out(&dn_scale)),
            Encoding::Raw(DType::BF16),
            Some(vec![experts, hidden, local_inter]),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::LoadShape;
    use crate::encoding::Encoding as StoredEncoding;
    use crate::shared::policy::Policy;
    use model_loader::checkpoint::{CheckpointMetadata, RawTensor};
    use model_loader::contract::{ModelContract, Visibility};
    use model_loader::plan::StorageTarget;
    use model_loader::types::{FileId, TensorId};

    /// `h` is the LOGICAL hidden width; the packed tensor stores it as
    /// `h / 8`, because eight 4-bit codes ride in one `i32` word. Both
    /// it and `INTER` are multiples of 32, which is the group the pass
    /// requires and which the refusal tests below deliberately break.
    const HIDDEN: i64 = 64;
    const INTER: i64 = 32;
    const CODES: i64 = 8;
    const GROUP: i64 = 32;
    const EXPERTS: u32 = 3;

    fn i32e() -> Encoding {
        Encoding::Raw(DType::I32)
    }

    fn bf16() -> Encoding {
        Encoding::Raw(DType::BF16)
    }

    fn tensor(id: u32, name: &str, shape: Vec<i64>, encoding: Encoding) -> RawTensor {
        RawTensor {
            id: TensorId(id),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 0,
            shape,
            encoding,
        }
    }

    /// One layer of W4A16 experts, exactly as kimi ships them: a packed
    /// `i32` weight and a BF16 `weight_scale` for each of gate, up and
    /// down.
    fn experts(count: u32) -> Vec<RawTensor> {
        let mut out = Vec::new();
        let mut id = 1u32;
        for e in 0..count {
            let ep = format!("model.layers.0.mlp.experts.{e}.");
            for member in ["gate_proj", "up_proj"] {
                out.push(tensor(
                    id,
                    &format!("{ep}{member}.weight_packed"),
                    vec![INTER, HIDDEN / CODES],
                    i32e(),
                ));
                id += 1;
                out.push(tensor(
                    id,
                    &format!("{ep}{member}.weight_scale"),
                    vec![INTER, HIDDEN / GROUP],
                    bf16(),
                ));
                id += 1;
            }
            out.push(tensor(
                id,
                &format!("{ep}down_proj.weight_packed"),
                vec![HIDDEN, INTER / CODES],
                i32e(),
            ));
            id += 1;
            out.push(tensor(
                id,
                &format!("{ep}down_proj.weight_scale"),
                vec![HIDDEN, INTER / GROUP],
                bf16(),
            ));
            id += 1;
        }
        out
    }

    fn build(tensors: &[RawTensor], f: impl FnOnce(&mut Builder<'_>)) -> ModelContract {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: tensors.to_vec(),
        };
        let target = StorageTarget {
            preferred_alignment: 256,
            ..StorageTarget::default()
        };
        let encoding = StoredEncoding::dense();
        let policy = Policy::default();
        let mut b = Builder::new(
            &meta,
            "kimi-test",
            LoadShape::mixture(1, 128, EXPERTS, false),
            &encoding,
            &target,
            &policy,
        );
        f(&mut b);
        b.publish_remaining().expect("publish");
        b.finish().expect("finish")
    }

    /// Run the stacker with a budget large enough that it never declines.
    fn stack(tensors: &[RawTensor]) -> ModelContract {
        build(tensors, |b| {
            bf16_expert_stacks(b, 4u64 << 30).expect("the pass does not refuse");
        })
    }

    fn refusal(tensors: &[RawTensor]) -> String {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: tensors.to_vec(),
        };
        let target = StorageTarget {
            preferred_alignment: 256,
            ..StorageTarget::default()
        };
        let encoding = StoredEncoding::dense();
        let policy = Policy::default();
        let mut b = Builder::new(
            &meta,
            "kimi-test",
            LoadShape::mixture(1, 128, EXPERTS, false),
            &encoding,
            &target,
            &policy,
        );
        match bf16_expert_stacks(&mut b, 4u64 << 30) {
            Err(Error::Contract(why)) => why,
            other => panic!("expected a contract refusal, got {other:?}"),
        }
    }

    fn find<'a>(
        c: &'a ModelContract,
        name: &str,
    ) -> Option<&'a model_loader::contract::TensorContract> {
        c.tensors.iter().find(|t| t.name == name)
    }

    /// `(bank, first row, row count)` for an expression that is a slice of
    /// another declared tensor, and `None` for anything else — a part read
    /// straight from the checkpoint gives `None`, which is the answer that
    /// distinguishes a join from a pass-through.
    fn band(e: &Expr) -> Option<(String, i64, i64)> {
        match e {
            Expr::Slice {
                src,
                axis: model_loader::types::Axis(0),
                start,
                len,
            } => match src.as_ref() {
                Expr::Out(name) => Some((name.clone(), *start, *len)),
                _ => None,
            },
            _ => None,
        }
    }

    const GU: &str = "model.layers.0.mlp.experts.gate_up.weight";
    const DN: &str = "model.layers.0.mlp.experts.down.weight";
    const GU_S: &str = "model.layers.0.mlp.experts.gate_up.scale";
    const DN_S: &str = "model.layers.0.mlp.experts.down.scale";

    /// The shape of what the fused grouped GEMM reads.
    ///
    /// Three experts fold into one `[E, 2*inter, hidden]` slab and one
    /// `[E, hidden, inter]`. The declared extents are what the driver
    /// allocates and strides by, so a slab whose declaration disagrees
    /// with its own contents is not a load error -- it is a GEMM reading
    /// the wrong rows.
    #[test]
    fn the_stacked_slabs_are_declared_at_the_extents_the_gemm_strides_by() {
        let c = stack(&experts(EXPERTS));
        let gu = find(&c, GU).expect("a gate_up slab");
        let dn = find(&c, DN).expect("a down slab");
        assert_eq!(gu.shape, Some(vec![EXPERTS as i64, 2 * INTER, HIDDEN]));
        assert_eq!(dn.shape, Some(vec![EXPERTS as i64, HIDDEN, INTER]));
        assert_eq!(gu.encoding, bf16(), "the slab is dequantized, not packed");
        assert_eq!(dn.encoding, bf16());
    }

    /// The factors are DEFINED and then hidden.
    ///
    /// `scale_per_block` takes its factors by output NAME, so the scale
    /// has to be a declared tensor rather than a companion the lowering
    /// guesses from a suffix. Left `Public` it would sit in the
    /// persistent arena for the process's lifetime and appear in the
    /// driver's bind table under a name nothing ever asks for -- the
    /// slab is already dequantized, so no kernel reads these again.
    ///
    /// `Visibility` is the only thing that separates the two outcomes:
    /// both declare the same BF16 encoding at the same extents.
    #[test]
    fn the_factors_the_slab_consumed_are_declared_and_then_hidden() {
        let c = stack(&experts(EXPERTS));
        for name in [GU_S, DN_S] {
            let s = find(&c, name).unwrap_or_else(|| panic!("{name} is declared"));
            assert_eq!(
                s.visibility,
                Visibility::Internal,
                "{name} survives into the driver's bind table"
            );
        }
        for name in [GU, DN] {
            assert_eq!(
                find(&c, name).expect("a slab").visibility,
                Visibility::Public,
                "{name} is what the GEMM binds"
            );
        }
    }

    /// A model whose slabs do not fit publishes NONE of them.
    ///
    /// The BF16 slabs are additive on top of the packed W4A16 experts,
    /// which the per-step GEMV path still reads. Over budget, the packed
    /// path is not a fallback -- it is the path that wins below the
    /// crossover -- so declining is a choice and not a failure, and it
    /// must leave the checkpoint whole rather than half-stacked.
    #[test]
    fn a_model_whose_slabs_do_not_fit_the_budget_publishes_no_slab_at_all() {
        let ck = experts(EXPERTS);
        let c = build(&ck, |b| {
            bf16_expert_stacks(b, 1).expect("declining is not refusing");
        });
        for name in [GU, DN, GU_S, DN_S] {
            assert!(
                find(&c, name).is_none(),
                "{name} was published against a budget of one byte"
            );
        }
        assert!(
            find(&c, "model.layers.0.mlp.experts.0.gate_proj.weight_packed").is_some(),
            "the packed experts must survive the decline -- they are the \
             path that runs when the slabs do not"
        );
    }

    /// The budget is the WHOLE MODEL's, not one layer's.
    ///
    /// Multiplied by `layers` so the answer cannot come out different
    /// for different layers of one model: a per-layer test would stack
    /// the early layers and decline the late ones, which is the
    /// half-stacked checkpoint the decline exists to avoid.
    #[test]
    fn the_budget_is_measured_against_every_layer_and_not_just_this_one() {
        let ck = experts(EXPERTS);
        let one_layer = u64::from(EXPERTS) * 3 * (INTER as u64) * (HIDDEN as u64) * 2;
        let c = build(&ck, |b| {
            bf16_expert_stacks(b, one_layer).expect("declining is not refusing");
        });
        assert!(
            find(&c, GU).is_some(),
            "one layer's worth of slab fits a budget of exactly that"
        );

        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: ck.clone(),
        };
        let target = StorageTarget {
            preferred_alignment: 256,
            ..StorageTarget::default()
        };
        let encoding = StoredEncoding::dense();
        let policy = Policy::default();
        let mut b = Builder::new(
            &meta,
            "kimi-test",
            // The SAME single layer of tensors, in a model that says it
            // has four. Nothing else changes.
            LoadShape::mixture(4, 128, EXPERTS, false),
            &encoding,
            &target,
            &policy,
        );
        bf16_expert_stacks(&mut b, one_layer).expect("declining is not refusing");
        b.publish_remaining().expect("publish");
        assert!(
            find(&b.finish().expect("finish"), GU).is_none(),
            "four layers of the same slab do not fit one layer's budget"
        );
    }

    /// A dense layer has no expert names, and the loop must pass over it.
    ///
    /// Kimi's leading layers are dense. The pass probes for expert zero
    /// rather than reading `first_k_dense_replace`, so a checkpoint with
    /// no experts anywhere must simply produce nothing -- not a refusal
    /// naming a tensor a dense layer was never going to ship.
    #[test]
    fn a_layer_with_no_experts_is_passed_over_rather_than_refused() {
        let dense = vec![tensor(
            1,
            "model.layers.0.mlp.gate_proj.weight",
            vec![INTER, HIDDEN],
            bf16(),
        )];
        let c = stack(&dense);
        assert!(find(&c, GU).is_none() && find(&c, DN).is_none());
    }

    /// An expert missing one of its six parts is refused BY NAME.
    ///
    /// Stated for all six, because the lookup is a loop over six names
    /// and a fixture that removes one exercises one iteration. A hole
    /// that got past here would stack a slab with a leg missing, and
    /// `concat` would build it at whatever extents the survivors imply.
    #[test]
    fn an_expert_missing_any_of_its_six_parts_is_refused_and_named() {
        for (member, part) in [
            ("gate_proj", "weight_scale"),
            ("up_proj", "weight_packed"),
            ("up_proj", "weight_scale"),
            ("down_proj", "weight_packed"),
            ("down_proj", "weight_scale"),
        ] {
            let mut ck = experts(EXPERTS);
            let gone = format!("model.layers.0.mlp.experts.1.{member}.{part}");
            ck.retain(|t| t.name != gone);
            let why = refusal(&ck);
            assert!(
                why.contains("missing a weight or scale") && why.contains("experts.1."),
                "a missing {member}.{part}: {why}"
            );
        }
    }

    /// The SIXTH name is the loop's own probe, and it fails differently.
    ///
    /// `experts.{n}.gate_proj.weight_packed` is what the loop tests to
    /// decide whether expert `n` exists, so removing it does not look
    /// like a hole -- it looks like the end of the bank. The loop stops
    /// and stacks a SHORTER slab.
    ///
    /// Nothing downstream would have caught it. The manifest measures
    /// the ROUTER, `[num_experts, hidden]`, and a checkpoint missing a
    /// whole expert still carries a full-width router, so the row still
    /// matches and the load still succeeds. The grouped GEMM would then
    /// index a slab with fewer rows than the router emits indices for.
    /// The count check exists for this one name.
    #[test]
    fn an_expert_missing_the_very_name_the_loop_probes_is_caught_by_the_count() {
        let mut ck = experts(EXPERTS);
        ck.retain(|t| {
            !t.name
                .starts_with("model.layers.0.mlp.experts.1.gate_proj.weight_packed")
        });
        let why = refusal(&ck);
        assert!(
            why.contains("stacked 1 experts") && why.contains("router emits indices"),
            "the bank stopped at expert 1 and nothing said so: {why}"
        );
    }

    /// A bank LONGER than the row states is refused too.
    ///
    /// The check is an equality and not a floor: a slab with more rows
    /// than the router can address is a different checkpoint from the
    /// one this row identifies, and stacking it would bind extra
    /// resident weight nothing routes to.
    #[test]
    fn a_bank_longer_than_the_row_states_is_refused_as_well() {
        let why = refusal(&experts(EXPERTS + 1));
        assert!(
            why.contains(&format!("stacked {} experts", EXPERTS + 1)),
            "{why}"
        );
    }

    /// A checkpoint packed some other way is LEFT ALONE, not refused.
    ///
    /// Stacking a differently-encoded expert would hand the GEMM
    /// garbage, and rewriting it is not this pass's job -- so the whole
    /// checkpoint stays packed and the W4A16 path runs off it as before.
    /// A refusal here would fail a load that has a perfectly good path.
    #[test]
    fn experts_encoded_some_other_way_leave_the_whole_checkpoint_alone() {
        for (member, part, wrong) in [
            ("gate_proj", "weight_packed", bf16()),
            ("up_proj", "weight_packed", bf16()),
            ("down_proj", "weight_packed", bf16()),
            ("gate_proj", "weight_scale", i32e()),
            ("up_proj", "weight_scale", i32e()),
            ("down_proj", "weight_scale", i32e()),
        ] {
            let mut ck = experts(EXPERTS);
            let target = format!("model.layers.0.mlp.experts.1.{member}.{part}");
            for t in &mut ck {
                if t.name == target {
                    t.encoding = wrong.clone();
                }
            }
            let c = stack(&ck);
            assert!(
                find(&c, GU).is_none(),
                "a {part} of the wrong type on {member} still stacked a slab"
            );
        }
    }

    /// Both expert dims must be a multiple of the 32-code group.
    ///
    /// The scale slab is declared at `hidden / 32` and `inter / 32`. A
    /// width that does not divide truncates, and the declared factor
    /// count then covers fewer codes than the slab holds.
    #[test]
    fn an_expert_whose_dims_do_not_divide_the_group_is_refused() {
        let mut ck = experts(EXPERTS);
        for t in &mut ck {
            if t.name.ends_with("experts.1.gate_proj.weight_packed") {
                // 3 words is 24 codes -- not a multiple of 32.
                t.shape = vec![INTER, 3];
            }
        }
        let why = refusal(&ck);
        assert!(why.contains("multiple of 32"), "{why}");

        let mut ck = experts(EXPERTS);
        for t in &mut ck {
            if t.name.ends_with("experts.1.gate_proj.weight_packed") {
                t.shape = vec![INTER + 1, HIDDEN / CODES];
            }
        }
        let why = refusal(&ck);
        assert!(why.contains("multiple of 32"), "an odd inter: {why}");
    }

    /// A rank-1 expert weight is refused before anything indexes it.
    ///
    /// The pass reads `shape[0]` and `shape[1]` immediately after, so a
    /// rank that got past here is a panic and not a bad slab.
    #[test]
    fn a_rank_1_expert_weight_is_refused() {
        for member in ["gate_proj", "down_proj"] {
            let mut ck = experts(EXPERTS);
            for t in &mut ck {
                if t.name
                    .ends_with(&format!("experts.1.{member}.weight_packed"))
                {
                    t.shape = vec![INTER];
                }
            }
            let why = refusal(&ck);
            assert!(why.contains("rank-2"), "a rank-1 {member}: {why}");
        }
    }

    /// Siblings that disagree on shape are refused, naming the sibling.
    ///
    /// Every leg is declared `[1, ..]` and concatenated over axis 0, so
    /// the stack's extents come from the FIRST expert. A sibling of a
    /// different width would be concatenated at the first one's
    /// declaration.
    #[test]
    fn an_expert_that_disagrees_with_its_siblings_on_shape_is_refused() {
        for (member, shape) in [
            ("gate_proj", vec![INTER * 2, HIDDEN / CODES]),
            ("gate_proj", vec![INTER, HIDDEN * 2 / CODES]),
        ] {
            let mut ck = experts(EXPERTS);
            for t in &mut ck {
                if t.name
                    .ends_with(&format!("experts.1.{member}.weight_packed"))
                {
                    t.shape = shape.clone();
                }
            }
            let why = refusal(&ck);
            assert!(
                why.contains("disagrees with its siblings") && why.contains("experts.1."),
                "{why}"
            );
        }
    }

    /// The two halves are stacked GATE FIRST, and nothing else in this file
    /// reads the order.
    ///
    /// `mlp::chunked_swiglu_bf16` computes `y[n,i] = silu(packed[n,i]) *
    /// packed[n,I+i]` when its `gate_second` argument is left at its
    /// default, and the trace's `dsl::cuda::swiglu` records no params, so
    /// the launch takes that default. The LEADING half is therefore the one
    /// that gets the silu. Stacking `[up|gate]` here would silu the up
    /// projection and multiply by the gate: same shapes, same dtypes, no
    /// NaN, a different model — which is why the order is asserted rather
    /// than left to the concat's argument order to keep.
    ///
    /// The scales ride the same stack, so their order has to match the
    /// weights' or every column is dequantized by its neighbour's factor.
    #[test]
    fn the_gate_leads_the_stack_and_the_scales_ride_in_the_same_order() {
        let c = stack(&experts(EXPERTS));
        for (name, first, second) in [
            (GU, "gate_proj.weight_packed", "up_proj.weight_packed"),
            (GU_S, "gate_proj.weight_scale", "up_proj.weight_scale"),
        ] {
            let t = find(&c, name).expect("stacked");
            let srcs = t.expr.sources();
            // Expert 0's two legs are the first two names the axis-1 concat
            // reads; every later expert repeats the pair.
            assert!(
                srcs[0].ends_with(first) && srcs[1].ends_with(second),
                "{name} stacks {:?} first, but the leading half is the one \
                 chunked_swiglu applies silu to",
                &srcs[..2.min(srcs.len())]
            );
        }
    }

    /// The author itself runs the pass, and under kimi's source prefix.
    ///
    /// Every test above calls `bf16_expert_stacks` directly, so deleting
    /// its call from `author_kimi` would be invisible to all of them.
    /// The prefix matters as much as the call: kimi's checkpoint names
    /// are under `language_model.`, and the pass probes through
    /// `source_name`, so a pass that ran without the prefix would find
    /// no experts and silently stack nothing.
    #[test]
    fn the_author_itself_stacks_and_does_it_under_kimis_prefix() {
        let prefixed: Vec<RawTensor> = experts(EXPERTS)
            .into_iter()
            .map(|mut t| {
                t.name = format!("language_model.{}", t.name);
                t
            })
            .collect();
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: prefixed,
        };
        let target = StorageTarget {
            preferred_alignment: 256,
            ..StorageTarget::default()
        };
        let encoding = StoredEncoding::dense();
        let policy = Policy::default();
        let mut b = Builder::new(
            &meta,
            "kimi-test",
            LoadShape::mixture(1, 128, EXPERTS, false),
            &encoding,
            &target,
            &policy,
        );
        author_kimi(&mut b).expect("the author does not refuse");
        let c = b.finish().expect("finish");
        assert!(
            find(&c, GU).is_some() && find(&c, DN).is_some(),
            "the author published no slab, so either it does not run the \
             pass or it does not set the prefix the pass probes through"
        );
    }

    /// The MLA joins fuse TWO pairs, and only one of them had ever run.
    ///
    /// `mla_fused_projection_joins` joins `q_a_proj + kv_a_proj_with_mqa`
    /// (which share `norm_x`) and the shared expert's `gate_proj +
    /// up_proj` (which share `norm_y`). Every other test in this file
    /// builds a checkpoint of routed experts only, so the second join saw
    /// no candidate and its arm never ran — and kimi-k2 rides a shared
    /// expert on every MoE layer, which `spec.rs` asserts of the row.
    ///
    /// A join is a fusion, not a rename: the two source weights become one
    /// `[2I, H]` tensor under a `.fused.` name and are consumed. Losing the
    /// second arm would leave `gate_proj` and `up_proj` published
    /// separately, which is not a load error — it is one extra GEMM launch
    /// per layer per step, over the widest matrices in the shared MLP,
    /// binding names the fused forward does not ask for.
    #[test]
    fn the_shared_experts_gate_and_up_are_joined_the_same_way_the_mla_pair_is() {
        let mut tensors = Vec::new();
        // The two pairs the pass joins, under kimi's source prefix.
        for (id, (name, shape)) in [
            ("self_attn.q_a_proj.weight", vec![INTER, HIDDEN]),
            ("self_attn.kv_a_proj_with_mqa.weight", vec![INTER, HIDDEN]),
            ("mlp.shared_experts.gate_proj.weight", vec![INTER, HIDDEN]),
            ("mlp.shared_experts.up_proj.weight", vec![INTER, HIDDEN]),
        ]
        .into_iter()
        .enumerate()
        {
            tensors.push(tensor(
                100 + id as u32,
                &format!("language_model.model.layers.0.{name}"),
                shape,
                bf16(),
            ));
        }
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors,
        };
        let target = StorageTarget {
            preferred_alignment: 256,
            ..StorageTarget::default()
        };
        let encoding = StoredEncoding::dense();
        let policy = Policy::default();
        let mut b = Builder::new(
            &meta,
            "kimi-test",
            LoadShape::mixture(1, 128, EXPERTS, false),
            &encoding,
            &target,
            &policy,
        );
        author_kimi(&mut b).expect("the author does not refuse");
        let c = b.finish().expect("finish");

        for (fused, parts) in [
            (
                "model.layers.0.self_attn.q_kv_a_proj.fused.weight",
                [
                    "model.layers.0.self_attn.q_a_proj.weight",
                    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                ],
            ),
            (
                "model.layers.0.mlp.shared_experts.gate_up_proj.fused.weight",
                [
                    "model.layers.0.mlp.shared_experts.gate_proj.weight",
                    "model.layers.0.mlp.shared_experts.up_proj.weight",
                ],
            ),
        ] {
            let t = find(&c, fused).unwrap_or_else(|| {
                panic!(
                    "the pass published no {fused}; tensors: {:?}",
                    c.tensors.iter().map(|t| &t.name).collect::<Vec<_>>()
                )
            });
            assert_eq!(
                t.shape,
                Some(vec![2 * INTER, HIDDEN]),
                "{fused} is the two legs stacked on the output axis"
            );
            // The parts stay published, as VIEWS into the bank: a binder
            // that reads them by name still finds them, and the offset of
            // each leg is stated once here rather than recomputed. Asserting
            // their presence would pass on a pass that never joined at all,
            // so read the band each one names.
            for (leg, part) in parts.iter().enumerate() {
                let p = find(&c, part).unwrap_or_else(|| panic!("no {part}"));
                assert_eq!(
                    (band(&p.expr), p.shape.clone()),
                    (
                        Some((fused.to_string(), leg as i64 * INTER, INTER)),
                        Some(vec![INTER, HIDDEN])
                    ),
                    "{part} should be rows [{}, {}) of {fused}, not a tensor of \
                     its own — a part read from the checkpoint instead of the \
                     bank means the join did not happen and the forward binds \
                     both forms",
                    leg as i64 * INTER,
                    (leg as i64 + 1) * INTER
                );
            }
        }
    }
}
