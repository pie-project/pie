//! What the MLA families bind.
//!
//! Ported from `driver/cuda/src/model/kimi/kimi_contract.hpp`. DeepSeek-V2/V3
//! and Kimi-K2 share a binder and a forward. They differ in the contract:
//! Kimi hides the decoder under `language_model.` and wants `embed_tokens`
//! sharded and `lm_head` replicated, which is a memory trade the driver
//! makes and the checkpoint knows nothing about.

use pie_loader::contract::Expr;
use pie_loader::error::Error;
use pie_loader::types::{DType, Encoding};

use pie_model_common::builder::{Builder, int4b8_encoding, is_raw};

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
    for layer in 0..b.facts().num_hidden_layers {
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

    // `[up | gate]` is what flashinfer's fused grouped GEMM reads fc1 as,
    // and choosing the order here costs nothing — it is which `concat` leg
    // goes first. Undoing it later would cost a copy of the whole slab.
    // Kimi's grouped GEMM does not take that path, so the halves stay in
    // checkpoint order; `kimi_moe_gate_up_swapped()` on the forward side is
    // this same constant, and the two have to agree.
    const GATE_SECOND: bool = false;

    for layer in 0..b.facts().num_hidden_layers {
        let mlp = format!("model.layers.{layer}.mlp.");
        // Kimi's leading layers are dense, and a dense layer simply has no
        // expert names — which is why the loop probes rather than reading
        // `first_k_dense_replace`.
        if b.find(&b.source_name(&format!("{mlp}experts.0.gate_proj.weight_packed")))
            .is_none()
        {
            continue;
        }
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
                    Expr::src(name).transmute(pie_loader::contract::TensorType::new(
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
                    Expr::src(name).transmute(pie_loader::contract::TensorType::new(
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
            gate_up.push(Expr::concat(
                1,
                if GATE_SECOND {
                    vec![up, gate]
                } else {
                    vec![gate, up]
                },
            ));
            gate_up_scales.push(Expr::concat(
                1,
                if GATE_SECOND {
                    vec![up_s, gate_s]
                } else {
                    vec![gate_s, up_s]
                },
            ));
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
        if gate_up.is_empty() {
            continue;
        }
        let experts = gate_up.len() as i64;

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
            * u64::from(b.facts().num_hidden_layers);
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
        if let Some(gu) = b.define(
            gu_scale.clone(),
            Expr::concat(0, gate_up_scales),
            Encoding::Raw(DType::BF16),
            Some(vec![experts, 2 * local_inter, hidden / GROUP]),
        ) {
            b.mark_internal(gu);
        }
        if let Some(dn) = b.define(
            dn_scale.clone(),
            Expr::concat(0, down_scales),
            Encoding::Raw(DType::BF16),
            Some(vec![experts, hidden, local_inter / GROUP]),
        ) {
            b.mark_internal(dn);
        }
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
