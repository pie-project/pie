//! What Kimi-K3 binds.
//!
//! Ported from `driver/cuda/src/model/kimi_k3/kimi_k3_contract.hpp`. Three
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

use pie_loader::checkpoint::RawTensor;
use pie_loader::contract::{Expr, TensorType};
use pie_loader::error::Error;
use pie_loader::types::{DType, Encoding, TensorId};

use pie_model_common::builder::{Builder, is_raw, mxfp4_encoding};
use pie_model_common::probe::hf_shard_axis;

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
    // Checkpoint order, matching `kimi_k3_moe_gate_up_swapped()` on the
    // forward side. The two have to agree: a load that swaps the halves and
    // a matmul that does not is silently wrong output.
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
    // Routed experts under K3's `block_sparse_moe` spelling: shard the
    // intermediate dim inside each expert, w1/w3 on their output axis and w2
    // on its input axis, and all-reduce the partial expert outputs.
    if name.contains(".block_sparse_moe.experts.") {
        if name.ends_with(".w1.weight") || name.ends_with(".w3.weight") {
            return Ok(Some(0));
        }
        if name.ends_with(".w2.weight") {
            return Ok(Some(1));
        }
    }
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
    for layer in 0..b.facts().num_hidden_layers {
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
    let experts = i64::from(b.facts().num_experts);
    if experts <= 0 {
        return Ok(());
    }

    for layer in 0..b.facts().num_hidden_layers {
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

        if gate_up.is_empty() {
            continue;
        }
        // Named but not bound: `scale_per_block` takes its factors by output
        // name, and the stacked slab is dequantized here, so nothing reads
        // these again.
        let e8m0 = Encoding::Raw(DType::E8M0);
        let gu_scale = format!("{moe}experts.gate_up.scale");
        let dn_scale = format!("{moe}experts.down.scale");
        if let Some(gu) = b.define(
            gu_scale.clone(),
            Expr::concat(0, gate_up_scales),
            e8m0.clone(),
            Some(vec![experts, 2 * local_inter, latent / GROUP]),
        ) {
            b.mark_internal(gu);
        }
        if let Some(dn) = b.define(
            dn_scale.clone(),
            Expr::concat(0, down_scales),
            e8m0,
            Some(vec![experts, latent, local_inter / GROUP]),
        ) {
            b.mark_internal(dn);
        }
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
