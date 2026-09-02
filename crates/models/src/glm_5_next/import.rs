use checkpoint::contract::{Expr, ModelContract, TensorType};
use model_dsl::{Platform, Shard, Weight};

use super::model::{Indexer, Kda, Mixer, Mla, Mlp, Model};
use checkpoint_dsl::{Builder, Error};

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        self.import_from_mlx(src, platform)
    }

    /// The `Vontra/GLM-5.3-Flash-MLX-2bit-MTP` names: a text tower under
    /// `model.language_model.`, its head at the root, the vision tower and the
    /// `mtp.` block unread.
    pub fn import_from_mlx(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp, platform);
        b.read(&self.embed, "model.language_model.embed_tokens.weight")?;
        b.read(&self.final_norm, "model.language_model.norm.weight")?;
        b.read(&self.head, "lm_head.weight")?;

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("model.language_model.layers.{l}.{s}");

            b.read(&w.attn_mix.scale, n("hc_attn_scale"))?;
            b.read(&w.attn_mix.base, n("hc_attn_base"))?;
            b.read(&w.attn_mix.dynamic, n("hc_attn_fn"))?;
            b.read(&w.mlp_mix.scale, n("hc_ffn_scale"))?;
            b.read(&w.mlp_mix.base, n("hc_ffn_base"))?;
            b.read(&w.mlp_mix.dynamic, n("hc_ffn_fn"))?;
            b.read(&w.mixer_norm, n("input_layernorm.weight"))?;
            b.read(&w.mlp_norm, n("post_attention_layernorm.weight"))?;

            match &w.mixer {
                Mixer::Mla(a) => mla(&mut b, &n, a)?,
                Mixer::Kda(k) => kda(src, &mut b, &n, k)?,
            }

            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    b.read_concat(gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])?;
                    b.read(down, n("mlp.down_proj.weight"))?;
                }
                Mlp::Routed {
                    router,
                    bias,
                    gate_up,
                    down,
                    shared,
                    experts,
                    ..
                } => {
                    b.read(router, n("mlp.gate.weight"))?;
                    b.read(bias, n("mlp.gate.e_score_correction_bias"))?;

                    // Stored one expert at a time; the rows stack on axis 0.
                    let pair = |e: u32| {
                        vec![
                            n(&format!("mlp.experts.{e}.gate_proj.weight")),
                            n(&format!("mlp.experts.{e}.up_proj.weight")),
                        ]
                    };
                    b.read_stack(gate_up, (0..*experts).map(pair))?;
                    let one = |e: u32| vec![n(&format!("mlp.experts.{e}.down_proj.weight"))];
                    b.read_stack(down, (0..*experts).map(one))?;

                    if let Some(s) = shared {
                        b.read_concat(
                            &s.gate_up,
                            [
                                n("mlp.shared_experts.gate_proj.weight"),
                                n("mlp.shared_experts.up_proj.weight"),
                            ],
                        )?;
                        b.read(&s.down, n("mlp.shared_experts.down_proj.weight"))?;
                    }
                }
            }
        }
        Ok(b.build())
    }
}

fn mla(b: &mut Builder, n: &dyn Fn(&str) -> String, a: &Mla) -> Result<(), Error> {
    b.read(&a.q_a_proj, n("self_attn.q_a_proj.weight"))?;
    b.read(&a.q_a_norm, n("self_attn.q_a_layernorm.weight"))?;
    b.read(&a.q_b_proj, n("self_attn.q_b_proj.weight"))?;
    b.read(&a.kv_a_proj, n("self_attn.kv_a_proj_with_mqa.weight"))?;
    b.read(&a.kv_a_norm, n("self_attn.kv_a_layernorm.weight"))?;
    b.read(&a.kv_b_proj, n("self_attn.kv_b_proj.weight"))?;
    b.read(&a.o_proj, n("self_attn.o_proj.weight"))?;
    indexer(b, n, &a.indexer)
}

fn indexer(b: &mut Builder, n: &dyn Fn(&str) -> String, ix: &Indexer) -> Result<(), Error> {
    b.read(&ix.wq_b, n("self_attn.indexer.wq_b.weight"))?;
    b.read(&ix.wk, n("self_attn.indexer.wk.weight"))?;
    b.read(&ix.weights_proj, n("self_attn.indexer.weights_proj.weight"))?;
    b.read(&ix.k_norm, n("self_attn.indexer.k_norm.weight"))?;
    b.read(&ix.k_norm_bias, n("self_attn.indexer.k_norm.bias"))?;
    b.read(&ix.kpool_ape, n("self_attn.indexer.index_kpool_compress_ape"))?;
    b.read(
        &ix.kpool_gate,
        n("self_attn.indexer.index_kpool_compress_gate"),
    )?;
    Ok(())
}

fn kda(
    src: &ztensor::Source,
    b: &mut Builder,
    n: &dyn Fn(&str) -> String,
    k: &Kda,
) -> Result<(), Error> {
    b.read_concat(
        &k.qkv,
        [
            n("self_attn.q_proj.weight"),
            n("self_attn.k_proj.weight"),
            n("self_attn.v_proj.weight"),
        ],
    )?;
    // Each conv bank is stored [channels, 1, kernel] (the 1 is `groups`); the
    // declared type is rank-2, so each leg is squeezed before concatenation.
    let conv = Expr::concat(
        as_axis(cut_axis(&k.conv), &k.conv.name),
        vec![
            squeezed(src, n("self_attn.q_conv1d.weight"))?,
            squeezed(src, n("self_attn.k_conv1d.weight"))?,
            squeezed(src, n("self_attn.v_conv1d.weight"))?,
        ],
    );
    b.read_expr(&k.conv, conv)?;
    b.read(&k.f_a, n("self_attn.f_a_proj.weight"))?;
    b.read(&k.f_b, n("self_attn.f_b_proj.weight"))?;
    b.read(&k.g_a, n("self_attn.g_a_proj.weight"))?;
    b.read(&k.g_b, n("self_attn.g_b_proj.weight"))?;
    b.read(&k.b, n("self_attn.b_proj.weight"))?;
    // Stored flat [heads * head_dim]; the text states it per head.
    b.read_expr(
        &k.dt_bias,
        Expr::src(n("self_attn.dt_bias")).transmute(TensorType::new(
            vec![extent(u64::from(k.heads)), extent(u64::from(k.head_dim))],
            checkpoint_dsl::encoding(model_dsl::Dtype::F32),
        )),
    )?;
    b.read(&k.a_log, n("self_attn.A_log"))?;
    b.read(&k.o_norm, n("self_attn.o_norm.weight"))?;
    b.read(&k.o_proj, n("self_attn.o_proj.weight"))?;
    Ok(())
}

/// Drops the singleton `groups` axis: [channels, 1, kernel] -> [channels, kernel].
fn squeezed(src: &ztensor::Source, from: String) -> Result<Expr, Error> {
    let Some(tensor) = src.get(&from) else {
        return Err(Error::Missing(from));
    };
    let illegible = |why: &dyn std::fmt::Display| Error::Illegible {
        name: from.clone(),
        detail: why.to_string(),
    };
    let shape = tensor.shape();
    let [channels, 1, kernel] = *shape else {
        return Err(illegible(&format!(
            "a depthwise convolution bank is stored [channels, 1, kernel] and \
             this one is stored {shape:?}"
        )));
    };
    let part = tensor.part("data").map_err(|why| illegible(&why))?;
    let stored = checkpoint::file::encoding_of(&tensor, &part).map_err(|why| illegible(&why))?;
    Ok(Expr::src(from).transmute(TensorType::new(
        vec![extent(channels), extent(kernel)],
        stored,
    )))
}

fn extent(of: u64) -> i64 {
    i64::try_from(of).expect("an extent no i64 holds")
}

fn as_axis(axis: usize, name: &str) -> u8 {
    u8::try_from(axis)
        .unwrap_or_else(|_| panic!("`{name}` is packed on axis {axis}, which is no axis"))
}

fn cut_axis(w: &Weight) -> usize {
    match &w.shard {
        Shard::Replicated => panic!("`{}` is replicated and has no cut axis", w.name),
        Shard::Cut { axis, .. } => usize::try_from(*axis).expect("an axis inside a shape"),
    }
}
