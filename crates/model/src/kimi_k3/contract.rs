//! What Kimi-K3 binds.
//!
//! Ported from `crates/driver-cuda/csrc/src/model/kimi_k3/kimi_k3_contract.hpp`. Three
//! things the generic name-pattern rules cannot state:
//!
//! * K3's attention introduces tensor names no other family has (`g_proj`,
//!   `f_a_proj`, `f_b_proj`, `b_proj`, `dt_bias`, `A_log`,
//!   `{q,k,v}_conv1d`), and getting their TP axis wrong is silent: the model
//!   still loads and still emits tokens, just from the wrong heads.
//! * `A_log` ships as F32[128] while the layer has 96 KDA heads. A uniform
//!   row shard would hand rank 1 entries [64:128) — half real heads, half
//!   storage padding. The band takes [0:96) first and shards *that*.
//! * The latent MoE's three projections sit *outside* the expert bank: the
//!   experts shard on their intermediate dim and all-reduce, so the latent
//!   that enters and leaves them has to be full width on every rank. They
//!   are replicated, and it only looks like a default — HF's own
//!   `..._down_proj.weight` spelling misses `.down_proj.weight` by one
//!   character. State it rather than inherit it.

use model_loader::checkpoint::RawTensor;
use model_loader::contract::{Expr, TensorType};
use model_loader::error::Error;
use model_loader::types::{DType, Encoding, TensorId};

use crate::shared::builder::{Builder, is_raw, mxfp4_encoding};
use crate::shared::probe::hf_shard_axis;

fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

/// kimi_k3. `embed_tokens` is sharded on axis 0 under TP to save per-rank
/// memory, matching the Kimi-K2 row; the decoder lives under
/// `language_model.model.layers.`.
pub fn author_kimi_k3(b: &mut Builder<'_>) -> Result<(), Error> {
    // The multimodal checkpoints nest the decoder under `language_model.`,
    // the text-only ones do not. `source_prefix` asks the checkpoint rather
    // than declaring it, so one row covers both.
    b.source_prefix("language_model.");
    b.shard_axis_fn(kimi_k3_shard_axis);
    b.shard_embed_tokens();
    a_log_bands(b)?;
    // Checkpoint order, kept -- `[gate; up]`, the order
    // `mlp::chunked_swiglu_bf16` reads with its `gate_second` at the
    // default. A load that swaps the halves while the matmul does not is
    // silently wrong output rather than a load error.
    //
    // NOT settled, and stated here rather than assumed: the CUTLASS
    // grouped GEMM wants the opposite (`moe/flashinfer_moe.hpp`: "fc1
    // weights must be stacked as [up; gate]") and its enum lists `kimi`
    // under `Swiglu`. This crate cannot tell which epilogue a given fire
    // reaches -- `driver-cuda`'s own `gate_second` is written `false`
    // once and read nowhere -- so the two conventions have no checkable
    // meeting point. Whichever way it resolves, it resolves HERE and on
    // the sibling that states `true`, together.
    bf16_expert_stacks(b, /*gate_second=*/ false)?;
    // Deliberately *not* `author_dense_contract`: its
    // `dense_fused_projection_joins` would join `self_attn.{q,k,v}_proj`
    // into one QKV weight, which is right for a llama-like layer and wrong
    // here — KDA's q, k and v each go through their own short convolution
    // before they ever meet, so a fused projection would have to be split
    // straight back apart. The MoE slice still runs first, because a family
    // can have both a fused expert weight and fused dense projections.
    b.fused_moe_gate_up_tp_slices(false)?;
    b.publish_remaining()
}

/// TP axis policy for Kimi-K3.
fn kimi_k3_shard_axis(name: &str) -> Result<Option<u8>, Error> {
    // The latent MoE tail is replicated; see the module comment.
    if [
        ".routed_expert_down_proj.weight",
        ".routed_expert_up_proj.weight",
        ".routed_expert_norm.weight",
    ]
    .iter()
    .any(|tail| name.ends_with(tail))
    {
        return Ok(None);
    }
    // Replicated by shape: `f_a_proj` is the rank-`head_dim` bottleneck
    // every head reads, `o_norm` is per-channel within a head, and the
    // AttnRes projections are a single score row over the full-width
    // residual.
    if [
        ".self_attn.f_a_proj.weight",
        ".self_attn.o_norm.weight",
        ".self_attention_res_proj.weight",
        ".mlp_res_proj.weight",
        ".self_attention_res_norm.weight",
        ".mlp_res_norm.weight",
    ]
    .iter()
    .any(|tail| name.ends_with(tail))
    {
        return Ok(None);
    }
    // Per-head / per-head-channel tensors that follow the head split.
    // `b_proj` is [num_heads, hidden] — one beta row per head — and must not
    // be confused with `q_b_proj`, which the generic list already claims.
    if [
        ".self_attn.g_proj.weight",
        ".self_attn.f_b_proj.weight",
        ".self_attn.b_proj.weight",
        ".self_attn.dt_bias",
        ".self_attn.q_conv1d.weight",
        ".self_attn.k_conv1d.weight",
        ".self_attn.v_conv1d.weight",
    ]
    .iter()
    .any(|tail| name.ends_with(tail))
    {
        return Ok(Some(0));
    }
    // Routed experts fall through on purpose. This family's spelling is
    // `block_sparse_moe.experts.N.w{1,2,3}.weight_packed`, and what decides
    // their axis is the generic rule's companion-suffix step: a
    // `.weight_packed` or `.weight_scale` is asked about as the `.weight` it
    // packs, and `.w1`/`.w3` are row-parallel there while `.w2` is
    // column-parallel — the same intermediate-dim split, for the same
    // reason, already written down once.
    //
    // A branch matching `.w1.weight` used to sit here. It never fired: no
    // K3 checkpoint contains that name, because every expert ships packed.
    // It returned the answers the fallback returns anyway, so deleting it
    // changed nothing except that the rule is now in one place instead of
    // looking like it is in two.
    // `a_packed_expert_shards_the_way_its_weight_would` pins the answers.
    Ok(hf_shard_axis(name))
}

/// Shard `A_log` past its storage padding.
///
/// `[0:num_heads)` is the real gate bank; the tail exists only because the
/// checkpoint rounded the allocation up to `head_dim`. The head count comes
/// from `b_proj.weight`, which is `[num_heads, hidden]` — `ModelFacts` does
/// not carry it and `head_dim` there is MLA's 192, not KDA's 128, so the
/// checkpoint is both the only source and the right one.
fn a_log_bands(b: &mut Builder<'_>) -> Result<(), Error> {
    for layer in 0..b.shape().layers {
        let layer_prefix = format!("{}{layer}.self_attn.", b.decoder_layer_prefix_value());
        let (Some(raw), Some(beta)) = (
            b.find(&b.source_name(&format!("{layer_prefix}A_log"))),
            b.find(&b.source_name(&format!("{layer_prefix}b_proj.weight"))),
        ) else {
            continue; // an MLA layer.
        };
        if raw.shape.len() != 1 || beta.shape.is_empty() {
            return fail(format!(
                "kimi_k3 A_log band: layer {layer} has an unexpected A_log / b_proj rank"
            ));
        }
        let heads = beta.shape[0];
        if raw.shape[0] < heads {
            return fail(format!(
                "kimi_k3 A_log band: layer {layer} has {} gate entries for {heads} heads",
                raw.shape[0]
            ));
        }
        let (banded, rows) = b.band(Expr::src(&raw.name), 0, 0, heads);
        let encoding = raw.encoding.clone();
        let id = raw.id;
        b.define(b.output_name(&raw.name), banded, encoding, Some(vec![rows]));
        b.consume(id);
    }
    Ok(())
}

/// Dequantize K3's routed experts and stack them, at load time.
///
/// Each expert ships MXFP4: `weight_packed` is `U8 [out, in/2]` holding two
/// E2M1 codes per byte, and `weight_scale` is `U8 [out, in/32]` holding one
/// E8M0 exponent per group of 32 along the input axis. The batched MoE path
/// wants one dense bf16 slab per layer — `[E, 2I, L]` over `[E, L, I]` — so
/// a grouped GEMM sees a base pointer and a stride. The sharding lives
/// *inside* the stack, so each rank dequantizes only the intermediate slice
/// it keeps.
///
/// The same four sources are also republished per expert in the shape the
/// decode GEMV addresses — one `[2I, L/2]` interleaved slab per expert with
/// gate over up — because decode is bandwidth-bound and the four-bit form is
/// worth four times its weight there. Both stay resident and the forward
/// picks per step, the way Kimi-K2 does.
fn bf16_expert_stacks(b: &mut Builder<'_>, gate_second: bool) -> Result<(), Error> {
    const GROUP: i64 = 32;
    let experts = i64::from(b.shape().n_experts);
    if experts <= 0 {
        return Ok(());
    }

    for layer in 0..b.shape().layers {
        let moe = format!(
            "{}{layer}.block_sparse_moe.",
            b.decoder_layer_prefix_value()
        );
        let prefix = b.source_name(&moe);
        // K3's leading layers are dense and simply have no expert names,
        // which is why this probes rather than reading
        // `first_k_dense_replace`.
        if b.find(&format!("{prefix}experts.0.w1.weight_packed"))
            .is_none()
        {
            continue;
        }

        let mut gate_up = Vec::new();
        let mut gate_up_scales = Vec::new();
        let mut down = Vec::new();
        let mut down_scales = Vec::new();
        let mut consumed: Vec<TensorId> = Vec::new();
        let mut local_inter = 0i64;
        let mut latent = 0i64;

        // A width-changing `Transmute` may only rename a whole tensor, so
        // each source is reinterpreted on its own — U8 bytes as 4-bit MXFP4
        // elements, U8 bytes as E8M0 exponents — and the stacking happens
        // afterwards, over expressions that already have their final
        // encoding.
        let packed = |b: &Builder<'_>, raw: &RawTensor, shape: Vec<i64>, axis: u8| {
            b.shard(
                Expr::src(&raw.name).transmute(TensorType::new(shape.clone(), mxfp4_encoding(2))),
                shape,
                Some(axis),
            )
            .0
        };
        let factors = |b: &Builder<'_>, raw: &RawTensor, shape: Vec<i64>, axis: u8| {
            b.shard(
                Expr::src(&raw.name)
                    .transmute(TensorType::new(shape.clone(), Encoding::Raw(DType::E8M0))),
                shape,
                Some(axis),
            )
            .0
        };

        for e in 0..experts {
            let ep = format!("{prefix}experts.{e}.");
            let names = [
                format!("{ep}w1.weight_packed"),
                format!("{ep}w1.weight_scale"),
                format!("{ep}w3.weight_packed"),
                format!("{ep}w3.weight_scale"),
                format!("{ep}w2.weight_packed"),
                format!("{ep}w2.weight_scale"),
            ];
            let mut parts = Vec::with_capacity(6);
            for name in &names {
                let Some(part) = b.find(name) else {
                    return fail(format!(
                        "kimi_k3 expert stack: layer {layer} expert {e} is missing a \
                         weight or scale"
                    ));
                };
                parts.push(part);
            }
            // A checkpoint that packs its experts some other way is not this
            // pass's to rewrite — leave the whole model alone rather than
            // half of it.
            if parts
                .iter()
                .any(|part| !is_raw(&part.encoding, DType::U8) || part.shape.len() != 2)
            {
                return Ok(());
            }

            // w1/w3 are [I, L/2] packed with [I, L/32] scales; w2 is
            // [L, I/2] with [L, I/32]. The declared shapes below are in
            // *elements*, not bytes.
            let inter = parts[0].shape[0];
            let latent_here = parts[0].shape[1] * 2;
            if parts[4].shape[0] != latent_here
                || parts[4].shape[1] * 2 != inter
                || parts[1].shape[1] != latent_here / GROUP
                || parts[5].shape[1] != inter / GROUP
            {
                return fail(format!(
                    "kimi_k3 expert stack: layer {layer} expert {e} has inconsistent \
                     MXFP4 shapes"
                ));
            }
            if e == 0 {
                latent = latent_here;
                local_inter = b.local_extent(inter);
            } else if latent_here != latent {
                return fail(format!(
                    "kimi_k3 expert stack: layer {layer} expert {e} changes the latent width"
                ));
            }

            // `[up | gate]` is what flashinfer's grouped GEMM reads fc1 as.
            let w1 = packed(b, parts[0], vec![1, inter, latent], 1);
            let w3 = packed(b, parts[2], vec![1, inter, latent], 1);
            let w1s = factors(b, parts[1], vec![1, inter, latent / GROUP], 1);
            let w3s = factors(b, parts[3], vec![1, inter, latent / GROUP], 1);
            // The decode GEMV reads gate and up as *adjacent* rows — row 2i
            // is gate row i, row 2i+1 is up row i — because that is how
            // gpt-oss ships them, and it is what lets one warp pull a slab
            // of both with no gather. K3 ships them as two tensors, so the
            // interleaving is built here: reshape each to `[I, 1, cols]` and
            // join on the new middle axis, which is a rename plus a
            // concatenation rather than a permutation.
            let pair = |b: &Builder<'_>, a: &RawTensor, c: &RawTensor, cols: i64| {
                let u8enc = Encoding::Raw(DType::U8);
                let an = b
                    .split(Expr::src(&a.name), 0)
                    .transmute(TensorType::new(vec![local_inter, 1, cols], u8enc.clone()));
                let cn = b
                    .split(Expr::src(&c.name), 0)
                    .transmute(TensorType::new(vec![local_inter, 1, cols], u8enc));
                Expr::concat(1, vec![an, cn])
            };
            gate_up.push(Expr::concat(
                1,
                if gate_second {
                    vec![w3, w1]
                } else {
                    vec![w1, w3]
                },
            ));
            gate_up_scales.push(Expr::concat(
                1,
                if gate_second {
                    vec![w3s, w1s]
                } else {
                    vec![w1s, w3s]
                },
            ));
            down.push(packed(b, parts[4], vec![1, latent, inter], 2));
            down_scales.push(factors(b, parts[5], vec![1, latent, inter / GROUP], 2));

            // Gate over up, always — `gate_second` reorders the *bf16* stack
            // for flashinfer's fc1, and this path reads its two halves by
            // name into separate outputs.
            let ep_out = format!("{moe}experts.{e}.");
            let gu_packed = pair(b, parts[0], parts[2], latent / 2);
            let gu_scale = pair(b, parts[1], parts[3], latent / GROUP);
            let dn_packed = b.split(Expr::src(&parts[4].name), 1);
            let dn_scale = b.split(Expr::src(&parts[5].name), 1);
            let u8enc = Encoding::Raw(DType::U8);
            b.define(
                format!("{ep_out}gate_up.weight_packed"),
                gu_packed,
                u8enc.clone(),
                Some(vec![local_inter, 2, latent / 2]),
            );
            b.define(
                format!("{ep_out}gate_up.weight_scale"),
                gu_scale,
                u8enc.clone(),
                Some(vec![local_inter, 2, latent / GROUP]),
            );
            b.define(
                format!("{ep_out}down.weight_packed"),
                dn_packed,
                u8enc.clone(),
                Some(vec![latent, local_inter / 2]),
            );
            b.define(
                format!("{ep_out}down.weight_scale"),
                dn_scale,
                u8enc,
                Some(vec![latent, local_inter / GROUP]),
            );
            consumed.extend(parts.iter().map(|part| part.id));
        }

        // No `gate_up.is_empty()` guard here: `experts <= 0` returned above,
        // so this loop ran at least once, and every path through its body
        // either returned or pushed. The guard that used to sit here was the
        // same question asked a second time, from a place that could not see
        // the answer.
        // Named but not bound: `scale_per_block` takes its factors by output
        // name, and the stacked slab is dequantized here, so nothing reads
        // these again.
        let e8m0 = Encoding::Raw(DType::E8M0);
        let gu_scale = format!("{moe}experts.gate_up.scale");
        let dn_scale = format!("{moe}experts.down.scale");
        let gu = b.define(
            gu_scale.clone(),
            Expr::concat(0, gate_up_scales),
            e8m0.clone(),
            Some(vec![experts, 2 * local_inter, latent / GROUP]),
        );
        b.mark_internal(gu);
        let dn = b.define(
            dn_scale.clone(),
            Expr::concat(0, down_scales),
            e8m0,
            Some(vec![experts, latent, local_inter / GROUP]),
        );
        b.mark_internal(dn);
        b.define(
            format!("{moe}experts.gate_up_proj"),
            Expr::concat(0, gate_up).scale_per_block(Expr::out(&gu_scale)),
            Encoding::Raw(DType::BF16),
            Some(vec![experts, 2 * local_inter, latent]),
        );
        b.define(
            format!("{moe}experts.down_proj"),
            Expr::concat(0, down).scale_per_block(Expr::out(&dn_scale)),
            Encoding::Raw(DType::BF16),
            Some(vec![experts, latent, local_inter]),
        );
        for id in consumed {
            b.consume(id);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::LoadShape;
    use crate::encoding::Encoding as StoredEncoding;
    use crate::shared::policy::{Component, Policy};
    use model_loader::checkpoint::CheckpointMetadata;
    use model_loader::contract::ModelContract;
    use model_loader::plan::StorageTarget;
    use model_loader::types::FileId;

    const HIDDEN: i64 = 64;
    const LATENT: i64 = 64;
    const INTER: i64 = 32;
    const P: &str = "language_model.model.layers.0.";

    fn bf16() -> Encoding {
        Encoding::Raw(DType::BF16)
    }
    fn u8e() -> Encoding {
        Encoding::Raw(DType::U8)
    }

    /// A K3 checkpoint with one KDA layer and two MXFP4 experts.
    ///
    /// The same shape `family_contracts.rs` pins the golden against, built
    /// again here because these tests need to break it one tensor at a time
    /// and a golden fixture is the wrong place to keep damaged variants.
    fn checkpoint() -> Vec<RawTensor> {
        let mut ck = Vec::new();
        let mut push = |name: String, shape: Vec<i64>, encoding: Encoding| {
            let elements: i64 = shape.iter().product();
            ck.push(RawTensor {
                id: TensorId(u32::try_from(ck.len()).expect("a small fixture")),
                name,
                file_id: FileId(0),
                file_offset: 0,
                span_bytes: u64::try_from(elements).unwrap_or(0),
                shape,
                encoding,
            });
        };
        push(
            "language_model.model.embed_tokens.weight".into(),
            vec![128, HIDDEN],
            bf16(),
        );
        // 8 real gate entries in a 16-entry bank: the padding this family
        // exists to band past.
        push(
            format!("{P}self_attn.A_log"),
            vec![16],
            Encoding::Raw(DType::F32),
        );
        push(
            format!("{P}self_attn.b_proj.weight"),
            vec![8, HIDDEN],
            bf16(),
        );
        for expert in 0..2 {
            let e = format!("{P}block_sparse_moe.experts.{expert}.");
            for half in ["w1", "w3"] {
                push(
                    format!("{e}{half}.weight_packed"),
                    vec![INTER, LATENT / 2],
                    u8e(),
                );
                push(
                    format!("{e}{half}.weight_scale"),
                    vec![INTER, LATENT / 32],
                    u8e(),
                );
            }
            push(
                format!("{e}w2.weight_packed"),
                vec![LATENT, INTER / 2],
                u8e(),
            );
            push(
                format!("{e}w2.weight_scale"),
                vec![LATENT, INTER / 32],
                u8e(),
            );
        }
        push(
            "language_model.model.norm.weight".into(),
            vec![HIDDEN],
            bf16(),
        );
        ck
    }

    fn without(name: &str) -> Vec<RawTensor> {
        let mut ck = checkpoint();
        let before = ck.len();
        ck.retain(|raw| raw.name != name);
        assert_eq!(before - 1, ck.len(), "'{name}' was not in the fixture");
        ck
    }

    fn reshaped(name: &str, shape: Vec<i64>) -> Vec<RawTensor> {
        let mut ck = checkpoint();
        let raw = ck
            .iter_mut()
            .find(|raw| raw.name == name)
            .unwrap_or_else(|| panic!("'{name}' was not in the fixture"));
        raw.shape = shape;
        ck
    }

    fn retyped(name: &str, encoding: Encoding) -> Vec<RawTensor> {
        let mut ck = checkpoint();
        let raw = ck
            .iter_mut()
            .find(|raw| raw.name == name)
            .unwrap_or_else(|| panic!("'{name}' was not in the fixture"));
        raw.encoding = encoding;
        ck
    }

    fn run(
        tensors: Vec<RawTensor>,
        shape: LoadShape,
        policy: &Policy,
        author: impl FnOnce(&mut Builder<'_>) -> Result<(), Error>,
    ) -> Result<ModelContract, Error> {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors,
        };
        let target = StorageTarget {
            preferred_alignment: 256,
            ..StorageTarget::default()
        };
        let enc = StoredEncoding::dense();
        let mut b = Builder::new(&meta, "kimi-k3-test", shape, &enc, &target, policy);
        author(&mut b)?;
        b.finish()
    }

    /// The whole author, over a checkpoint that may have been damaged.
    fn author_over(tensors: Vec<RawTensor>) -> Result<ModelContract, Error> {
        run(
            tensors,
            LoadShape::mixture(1, 0, 2, true),
            &Policy::default(),
            author_kimi_k3,
        )
    }

    fn refusal(result: Result<ModelContract, Error>) -> String {
        match result {
            Err(Error::Contract(msg)) => msg,
            Err(other) => panic!("expected a contract refusal, got {other:?}"),
            Ok(_) => panic!("expected a refusal, and the author succeeded"),
        }
    }

    fn names(contract: &ModelContract) -> Vec<&str> {
        contract.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    /// The axis `name` gets on the path a load really walks.
    ///
    /// `Builder::shard_axis` is the entry point: it answers `embed_tokens`
    /// and `lm_head` itself, strips a companion suffix, and only then
    /// consults the family's rule. Asking `kimi_k3_shard_axis` directly
    /// skips all three.
    fn sharded(name: &str) -> Option<u8> {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: Vec::new(),
        };
        let target = StorageTarget {
            tp_size: 2,
            preferred_alignment: 256,
            ..StorageTarget::default()
        };
        let enc = StoredEncoding::dense();
        let policy = Policy::default();
        let mut b = Builder::new(
            &meta,
            "kimi-k3-test",
            LoadShape::mixture(1, 0, 2, true),
            &enc,
            &target,
            &policy,
        );
        b.shard_axis_fn(kimi_k3_shard_axis);
        b.shard_axis(name).expect("the policy does not refuse")
    }

    fn declared<'a>(
        contract: &'a ModelContract,
        name: &str,
    ) -> &'a model_loader::contract::TensorContract {
        contract
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("'{name}' was not declared in {:?}", names(contract)))
    }

    // ─── the TP axis policy ──────────────────────────────────────────

    /// Every name this family had to state, and the axis it states.
    ///
    /// The module comment opens by saying why this table is worth a test:
    /// getting one of these wrong is *silent*. The model still loads, the
    /// shapes still line up, and it still emits tokens -- from the wrong
    /// heads. There is no crash to notice and no golden to diff, because a
    /// golden pinned at tp_size 1 never calls this function at all.
    #[test]
    fn the_axis_of_every_name_this_family_had_to_state() {
        let cases: &[(&str, Option<u8>, &str)] = &[
            // Replicated: the latent MoE's three projections live outside
            // the expert bank, and the latent that enters and leaves the
            // experts has to be full width on every rank.
            (".mlp.routed_expert_down_proj.weight", None, "latent tail"),
            (".mlp.routed_expert_up_proj.weight", None, "latent tail"),
            (".mlp.routed_expert_norm.weight", None, "latent tail"),
            // Replicated by shape.
            (".self_attn.f_a_proj.weight", None, "the shared bottleneck"),
            (".self_attn.o_norm.weight", None, "per-channel in a head"),
            (".self_attention_res_proj.weight", None, "one score row"),
            (".mlp_res_proj.weight", None, "one score row"),
            (".self_attention_res_norm.weight", None, "one score row"),
            (".mlp_res_norm.weight", None, "one score row"),
            // Per-head, following the head split.
            (".self_attn.g_proj.weight", Some(0), "per head"),
            (".self_attn.f_b_proj.weight", Some(0), "per head"),
            (".self_attn.b_proj.weight", Some(0), "one beta row per head"),
            (".self_attn.dt_bias", Some(0), "per head"),
            (".self_attn.q_conv1d.weight", Some(0), "per head"),
            (".self_attn.k_conv1d.weight", Some(0), "per head"),
            (".self_attn.v_conv1d.weight", Some(0), "per head"),
            // The generic rules, still reached.
            (".self_attn.q_b_proj.weight", Some(0), "the generic list"),
            (".self_attn.o_proj.weight", Some(1), "the generic list"),
            (".input_layernorm.weight", None, "the generic list"),
        ];
        for (tail, axis, why) in cases {
            let name = format!("language_model.model.layers.3{tail}");
            assert_eq!(
                kimi_k3_shard_axis(&name).expect("the policy does not refuse"),
                *axis,
                "{name} ({why})"
            );
        }
    }

    /// `b_proj` and `q_b_proj` are one character apart and one is this
    /// family's own.
    ///
    /// Both answer 0, so the assertion is not the axis -- it is that the
    /// family's list matches the shorter name without swallowing the
    /// longer one, which a `contains` would.
    #[test]
    fn b_proj_does_not_swallow_q_b_proj() {
        assert!(
            !".self_attn.q_b_proj.weight".ends_with(".self_attn.b_proj.weight"),
            "the two names must stay distinguishable by suffix"
        );
        assert_eq!(
            kimi_k3_shard_axis("m.layers.0.self_attn.q_b_proj.weight").expect("no refusal"),
            Some(0)
        );
    }

    /// The latent tail's spelling misses the generic rule by one character.
    ///
    /// This is the module comment's third point, and it is why the tail is
    /// stated rather than inherited: `routed_expert_down_proj.weight` is
    /// not `down_proj.weight`, so the generic rule -- which would shard it
    /// on axis 1 -- does not fire, and *neither* does anything else. The
    /// name would fall through to `None` by accident. Stating it makes the
    /// same answer mean something.
    #[test]
    fn the_latent_tail_is_replicated_on_purpose_and_not_by_a_near_miss() {
        for tail in [
            ".mlp.routed_expert_down_proj.weight",
            ".mlp.routed_expert_up_proj.weight",
        ] {
            let name = format!("model.layers.0{tail}");
            assert_eq!(hf_shard_axis(&name), None, "the generic rule misses {name}");
            assert_eq!(
                kimi_k3_shard_axis(&name).expect("no refusal"),
                None,
                "and this family agrees, deliberately"
            );
        }
        // The one character that separates them.
        assert_eq!(
            hf_shard_axis("model.layers.0.mlp.down_proj.weight"),
            Some(1)
        );
    }

    /// K3's experts ship as `weight_packed`/`weight_scale`, and the axis of
    /// a companion is the axis of the weight it packs.
    ///
    /// Asked through [`Builder::shard_axis`] and not through the policy
    /// function, because the stripping happens *there* -- once, ahead of
    /// every family's own rule, which is what makes the pairing
    /// unforgettable for a family that supplied one. A test calling
    /// `kimi_k3_shard_axis` with a packed name directly would pass for the
    /// wrong reason: the generic fallback strips as well, so the answer
    /// comes out right through a path production never takes.
    #[test]
    fn a_packed_expert_shards_the_way_its_weight_would() {
        let e = "language_model.model.layers.0.block_sparse_moe.experts.1.";
        let a = "language_model.model.layers.0.self_attn.";
        for (name, axis) in [
            (format!("{e}w1.weight_packed"), Some(0)),
            (format!("{e}w1.weight_scale"), Some(0)),
            (format!("{e}w3.weight_packed"), Some(0)),
            (format!("{e}w2.weight_packed"), Some(1)),
            (format!("{e}w2.weight_scale"), Some(1)),
            // This family's own names arrive stripped by the same step,
            // which is the half a family-supplied rule could forget.
            (format!("{a}g_proj.weight_scale"), Some(0)),
            (format!("{a}f_a_proj.weight_scale"), None),
            // A companion whose base is not a `.weight` at all: the strip
            // asks about `dt_bias.weight`, gets nothing, and has to ask
            // again about `dt_bias` itself. Without that second question a
            // scaled bias would replicate while its heads sharded.
            (format!("{a}dt_bias.scale"), Some(0)),
        ] {
            assert_eq!(sharded(&name), axis, "{name} splits like what it packs");
        }
    }

    /// The branch that used to state the expert axes said what the generic
    /// rule already said.
    #[test]
    fn the_generic_rule_already_gave_the_expert_axes() {
        let e = "language_model.model.layers.0.block_sparse_moe.experts.1.";
        for (tail, axis) in [
            ("w1.weight", Some(0)),
            ("w3.weight", Some(0)),
            ("w2.weight", Some(1)),
        ] {
            let name = format!("{e}{tail}");
            assert_eq!(hf_shard_axis(&name), axis, "{name}");
            assert_eq!(
                kimi_k3_shard_axis(&name).expect("no refusal"),
                axis,
                "{name}"
            );
        }
    }

    /// Every name this family replicates is one the generic rule is silent
    /// about -- today.
    ///
    /// Stating a replication and falling through to one give the same
    /// answer, so no edit to that list can be caught by asking it a
    /// question. What *can* be caught is the generic rule growing an
    /// opinion that contradicts one of them: this family would still win,
    /// and the two would have quietly come to mean different things.
    #[test]
    fn every_stated_replication_is_one_the_generic_rule_is_silent_about() {
        for tail in [
            ".mlp.routed_expert_down_proj.weight",
            ".mlp.routed_expert_up_proj.weight",
            ".mlp.routed_expert_norm.weight",
            ".self_attn.f_a_proj.weight",
            ".self_attn.o_norm.weight",
            ".self_attention_res_proj.weight",
            ".mlp_res_proj.weight",
            ".self_attention_res_norm.weight",
            ".mlp_res_norm.weight",
        ] {
            let name = format!("model.layers.0{tail}");
            assert_eq!(
                hf_shard_axis(&name),
                None,
                "{name}: the generic rule now has an opinion this family \
                 overrides. Decide which is right rather than letting the \
                 two drift."
            );
        }
    }

    // ─── the A_log band ──────────────────────────────────────────────

    /// The band takes the real heads and leaves the storage padding.
    ///
    /// A uniform row shard of the 16-entry bank would hand rank 1 entries
    /// [8:16) -- every one of them padding -- so the band runs first and the
    /// shard applies to its result.
    ///
    /// The start is asserted and not just the length: a band of 8 taken
    /// from 1 is also 8 long, and would be off by one head everywhere.
    #[test]
    fn a_log_is_banded_to_the_head_count_the_checkpoint_states() {
        let contract = author_over(checkpoint()).expect("the fixture authors");
        let banded = declared(&contract, "model.layers.0.self_attn.A_log");
        assert_eq!(
            banded.shape,
            Some(vec![8]),
            "the 16-entry bank holds 8 real gates"
        );
        let expr = format!("{:?}", banded.expr);
        assert!(
            expr.contains("start: 0") || expr.contains("start: 0,"),
            "the band starts at the first real gate: {expr}"
        );
    }

    /// The head count comes from `b_proj`, so a `b_proj` that cannot state
    /// one is refused rather than guessed at.
    #[test]
    fn a_gate_bank_whose_head_count_cannot_be_read_is_refused() {
        for (case, tensors) in [
            (
                "A_log is not a vector",
                reshaped(&format!("{P}self_attn.A_log"), vec![16, 1]),
            ),
            (
                "b_proj is a scalar",
                reshaped(&format!("{P}self_attn.b_proj.weight"), vec![]),
            ),
        ] {
            let msg = refusal(author_over(tensors));
            assert!(
                msg.contains("unexpected A_log / b_proj rank"),
                "{case}: {msg}"
            );
            assert!(msg.contains("layer 0"), "{case} names the layer: {msg}");
        }
    }

    /// A bank smaller than the head count is a checkpoint this band cannot
    /// describe, and taking [0:heads) from it would read past the end.
    #[test]
    fn a_gate_bank_shorter_than_its_head_count_is_refused() {
        let msg = refusal(author_over(reshaped(
            &format!("{P}self_attn.A_log"),
            vec![4],
        )));
        assert!(
            msg.contains("4 gate entries for 8 heads"),
            "the refusal states both numbers: {msg}"
        );
    }

    /// An MLA layer has no `A_log` at all, and is passed over rather than
    /// refused -- K3 mixes KDA and MLA layers in one checkpoint.
    #[test]
    fn a_layer_with_no_gate_bank_is_passed_over() {
        let contract = author_over(without(&format!("{P}self_attn.A_log")))
            .expect("an MLA layer is not an error");
        assert!(
            !names(&contract).iter().any(|name| name.contains("A_log")),
            "nothing was banded"
        );
    }

    // ─── the expert stacks ───────────────────────────────────────────

    /// The two experts become one bf16 slab per projection, plus the
    /// per-expert pairs the decode GEMV addresses.
    #[test]
    fn the_experts_stack_into_one_slab_and_republish_per_expert() {
        let contract = author_over(checkpoint()).expect("the fixture authors");
        let moe = "model.layers.0.block_sparse_moe.";

        let gate_up = declared(&contract, &format!("{moe}experts.gate_up_proj"));
        assert_eq!(gate_up.shape, Some(vec![2, 2 * INTER, LATENT]));
        assert_eq!(gate_up.encoding, bf16());
        let down = declared(&contract, &format!("{moe}experts.down_proj"));
        assert_eq!(down.shape, Some(vec![2, LATENT, INTER]));

        // The factors are named so `scale_per_block` can take them by name,
        // and internal because nothing else reads them.
        for tail in ["experts.gate_up.scale", "experts.down.scale"] {
            let scale = declared(&contract, &format!("{moe}{tail}"));
            assert_eq!(scale.encoding, Encoding::Raw(DType::E8M0), "{tail}");
            assert_eq!(
                scale.visibility,
                model_loader::contract::Visibility::Internal,
                "{tail} is named but not bound"
            );
        }

        // Gate and up are adjacent rows in the republished pair: [I, 2, ...].
        let pair = declared(&contract, &format!("{moe}experts.0.gate_up.weight_packed"));
        assert_eq!(pair.shape, Some(vec![INTER, 2, LATENT / 2]));
        let dn = declared(&contract, &format!("{moe}experts.1.down.weight_packed"));
        assert_eq!(dn.shape, Some(vec![LATENT, INTER / 2]));
    }

    /// `gate_second` reorders the *stack*, and nothing else.
    ///
    /// It has to agree with whichever epilogue the halves reach: the
    /// CUTLASS grouped GEMM reads gate from the SECOND half of fc1
    /// (`moe/flashinfer_moe.hpp`) and `mlp::chunked_swiglu_bf16` reads it
    /// from the first. A load that swaps while the matmul does not is
    /// silently wrong output, which is the whole reason the flag is passed
    /// explicitly rather than defaulted. The per-expert republish is gate
    /// over up either way, because that path reads its halves by name.
    #[test]
    fn gate_second_reorders_the_stack_and_leaves_the_republish_alone() {
        let of = |gate_second: bool| {
            run(
                checkpoint(),
                LoadShape::mixture(1, 0, 2, true),
                &Policy::default(),
                |b| {
                    b.source_prefix("language_model.");
                    bf16_expert_stacks(b, gate_second)
                },
            )
            .expect("the stacks author")
        };
        let moe = "model.layers.0.block_sparse_moe.";
        let (first, second) = (of(false), of(true));

        let expr = |c: &ModelContract, n: &str| format!("{:?}", declared(c, n).expr);
        assert_ne!(
            expr(&first, &format!("{moe}experts.gate_up_proj")),
            expr(&second, &format!("{moe}experts.gate_up_proj")),
            "the stack is built in the other order"
        );
        assert_ne!(
            expr(&first, &format!("{moe}experts.gate_up.scale")),
            expr(&second, &format!("{moe}experts.gate_up.scale")),
            "and so are its factors"
        );
        assert_eq!(
            expr(&first, &format!("{moe}experts.0.gate_up.weight_packed")),
            expr(&second, &format!("{moe}experts.0.gate_up.weight_packed")),
            "the republish is gate over up either way"
        );
    }

    /// A dense row does no expert work at all.
    ///
    /// The observable is `finish`'s own refusal: a contract that declares
    /// nothing is refused, so "the pass wrote nothing" and "the builder had
    /// nothing to finish" are one fact.
    #[test]
    fn a_row_with_no_experts_stacks_nothing() {
        let msg = refusal(run(
            checkpoint(),
            LoadShape::dense(1, 0, true),
            &Policy::default(),
            |b| {
                b.source_prefix("language_model.");
                bf16_expert_stacks(b, false)
            },
        ));
        assert!(msg.contains("no contract was authored"), "{msg}");
    }

    /// A layer with no expert names is passed over: K3's leading layers are
    /// dense, which is why this probes rather than reading a layer index.
    #[test]
    fn a_dense_layer_is_probed_for_and_passed_over() {
        let contract = author_over(without(&format!(
            "{P}block_sparse_moe.experts.0.w1.weight_packed"
        )))
        .expect("a dense layer is not an error");
        assert!(
            !names(&contract).iter().any(|n| n.contains("gate_up_proj")),
            "nothing was stacked"
        );
    }

    /// An expert missing one of its six sources is refused by name.
    #[test]
    fn an_expert_missing_a_source_is_refused() {
        let msg = refusal(author_over(without(&format!(
            "{P}block_sparse_moe.experts.1.w3.weight_scale"
        ))));
        assert!(
            msg.contains("layer 0 expert 1") && msg.contains("missing a weight or scale"),
            "{msg}"
        );
    }

    /// A checkpoint that packs its experts some other way is left whole.
    ///
    /// Not a refusal: this pass declines, the generic publisher takes the
    /// tensors as they are, and the alternative -- rewriting the experts it
    /// understood and leaving the rest -- would be half a model.
    #[test]
    fn experts_packed_some_other_way_are_left_alone() {
        for (case, tensors) in [
            (
                "not U8",
                retyped(
                    &format!("{P}block_sparse_moe.experts.0.w1.weight_packed"),
                    bf16(),
                ),
            ),
            (
                "not rank 2",
                reshaped(
                    &format!("{P}block_sparse_moe.experts.0.w1.weight_packed"),
                    vec![1, INTER, LATENT / 2],
                ),
            ),
        ] {
            let contract =
                author_over(tensors).unwrap_or_else(|e| panic!("{case} is not a refusal: {e}"));
            assert!(
                !names(&contract).iter().any(|n| n.contains("gate_up_proj")),
                "{case}: nothing was stacked"
            );
            // ...and the sources are still published, rather than dropped.
            assert!(
                names(&contract)
                    .iter()
                    .any(|n| n.ends_with("experts.1.w2.weight_packed")),
                "{case}: the experts are still in the contract"
            );
        }
    }

    /// The six shapes have to agree with each other, and each disagreement
    /// is refused by layer and expert.
    #[test]
    fn an_expert_whose_shapes_disagree_is_refused() {
        let e = format!("{P}block_sparse_moe.experts.1.");
        for (case, tensors) in [
            (
                "w2 does not see w1's latent",
                reshaped(&format!("{e}w2.weight_packed"), vec![LATENT / 2, INTER / 2]),
            ),
            (
                "w2's intermediate is not w1's",
                reshaped(&format!("{e}w2.weight_packed"), vec![LATENT, INTER]),
            ),
            (
                "w1's factors are not one per group of 32",
                reshaped(&format!("{e}w1.weight_scale"), vec![INTER, 1]),
            ),
            (
                "w2's factors are not one per group of 32",
                reshaped(&format!("{e}w2.weight_scale"), vec![LATENT, 2]),
            ),
        ] {
            let msg = refusal(author_over(tensors));
            assert!(
                msg.contains("layer 0 expert 1") && msg.contains("inconsistent MXFP4 shapes"),
                "{case}: {msg}"
            );
        }
    }

    /// One layer's experts have to agree on the latent they read.
    ///
    /// Reached only by an expert that is internally consistent and still
    /// disagrees with expert 0, which is why every one of its four shapes
    /// moves together here.
    #[test]
    fn an_expert_that_changes_the_latent_is_refused() {
        let e = format!("{P}block_sparse_moe.experts.1.");
        let mut tensors = checkpoint();
        let half = LATENT / 2;
        for raw in &mut tensors {
            let Some(tail) = raw.name.strip_prefix(&e) else {
                continue;
            };
            raw.shape = match tail {
                "w1.weight_packed" | "w3.weight_packed" => vec![INTER, half / 2],
                "w1.weight_scale" | "w3.weight_scale" => vec![INTER, half / 32],
                "w2.weight_packed" => vec![half, INTER / 2],
                "w2.weight_scale" => vec![half, INTER / 32],
                other => panic!("unexpected expert tensor '{other}'"),
            };
        }
        let msg = refusal(author_over(tensors));
        assert!(
            msg.contains("layer 0 expert 1") && msg.contains("changes the latent width"),
            "{msg}"
        );
    }

    /// An encode component declares no decoder weight, and the pass runs to
    /// the end anyway.
    ///
    /// `define` answers `None` for everything here, which is why
    /// `mark_internal` takes an `Option`: that function *indexes* the
    /// contract, so an index that was never handed out would panic rather
    /// than be skipped. `allow_encode_scope` gets past the
    /// first guard in `finish` so the second one — the empty contract — is
    /// what reports, which is the thing being asserted.
    #[test]
    fn an_encode_component_declares_no_expert_factors() {
        let policy = Policy {
            component: Component::Encode,
            ..Policy::default()
        };
        let msg = refusal(run(
            checkpoint(),
            LoadShape::mixture(1, 0, 2, true),
            &policy,
            |b| {
                b.source_prefix("language_model.");
                b.allow_encode_scope()?;
                bf16_expert_stacks(b, false)
            },
        ));
        assert!(msg.contains("no contract was authored"), "{msg}");
    }
}
