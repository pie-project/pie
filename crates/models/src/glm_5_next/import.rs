use checkpoint::contract::{Expr, ModelContract, TensorType};
use model_dsl::{Platform, Shard, Weight};

use super::model::{Indexer, Kda, Mixer, Mla, Mlp, Model};
use checkpoint_dsl::{Builder, Error};

/// The checkpoint's draft head, `layers.45` of the language model.
const HEAD: &str = "model.language_model.layers.45.";

/// Where a trunk plane is read from: the checkpoint's names, or (for an
/// overlay onto an artifact) the artifact's own — every trunk plane by its
/// declared name, as the artifact already holds it.
#[derive(Clone, Copy)]
enum From {
    Source,
    Own,
}

/// A [`Builder`] whose trunk reads honour [`From`]; head planes always read
/// by name, since they are the new bytes either way.
struct Land<'a> {
    b: Builder<'a>,
    from: From,
}

impl Land<'_> {
    fn read(&mut self, w: &Weight, name: String) -> Result<(), Error> {
        match self.from {
            From::Source => self.b.read(w, name),
            From::Own => self.b.read_own(w),
        }
    }

    fn read_concat(&mut self, w: &Weight, names: Vec<String>) -> Result<(), Error> {
        match self.from {
            From::Source => self.b.read_concat(w, names),
            From::Own => self.b.read_own(w),
        }
    }

    fn read_stack(
        &mut self,
        w: &Weight,
        rows: impl IntoIterator<Item = Vec<String>>,
    ) -> Result<(), Error> {
        match self.from {
            From::Source => self.b.read_stack(w, rows),
            From::Own => self.b.read_own(w),
        }
    }

    fn read_expr(
        &mut self,
        w: &Weight,
        expr: impl FnOnce() -> Result<Expr, Error>,
    ) -> Result<(), Error> {
        match self.from {
            From::Source => self.b.read_expr(w, expr()?),
            From::Own => self.b.read_own(w),
        }
    }
}

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        // An artifact with the head overlaid (`pie model import <artifact.zt>
        // --aux <shards>`): the head under `aux.`, every trunk plane its own.
        if self.mtp.is_some() && src.get(&format!("aux.{HEAD}enorm.weight")).is_some() {
            return self.import_from_own_with_aux(src, platform);
        }
        self.import_from_mlx(src, platform)
    }

    /// The `Vontra/GLM-5.3-Flash-MLX-2bit-MTP` names: a text tower under
    /// `model.language_model.`, its head at the root, the draft head (on a
    /// row that declares one) at `layers.45`, the vision tower unread.
    pub fn import_from_mlx(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        self.land(src, platform, From::Source, HEAD)
    }

    /// Reads a STAMPED ARTIFACT of this family's text row with the draft head
    /// overlaid beside it: every trunk plane by its own name, the head's
    /// through the `aux.` reading. What lets a head be put onto a
    /// hundred-gigabyte artifact whose source snapshot is gone.
    pub fn import_from_own_with_aux(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        if self.mtp.is_none() {
            return Err(Error::Illegible {
                name: "mtp".to_string(),
                detail: "this row declares no draft head, so there is no overlay to land \
                         on an artifact"
                    .to_string(),
            });
        }
        self.land(src, platform, From::Own, &format!("aux.{HEAD}"))
    }

    fn land(
        &self,
        src: &ztensor::Source,
        platform: Platform,
        from: From,
        head: &str,
    ) -> Result<ModelContract, Error> {
        let mut b = Land {
            b: Builder::new(src, self.tp, platform),
            from,
        };
        b.read(&self.embed, "model.language_model.embed_tokens.weight".into())?;
        b.read(&self.final_norm, "model.language_model.norm.weight".into())?;
        b.read(&self.head, "lm_head.weight".into())?;

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
                    b.read_concat(
                        gate_up,
                        vec![n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
                    )?;
                    b.read(down, n("mlp.down_proj.weight"))?;
                }
                routed @ Mlp::Routed { .. } => moe(&mut b, &n, routed)?,
            }
        }

        // The draft head: new bytes on either road, so read by name.
        let Some(mtp) = &self.mtp else {
            return Ok(b.b.build());
        };
        let n = |s: &str| format!("{head}{s}");
        let hidden = i64::from(self.hidden);
        let mut b = Land {
            b: b.b,
            from: From::Source,
        };
        b.read(&mtp.enorm, n("enorm.weight"))?;
        b.read(&mtp.hnorm, n("hnorm.weight"))?;
        // `eh_proj` is one `[hidden, 2·hidden]` plane over `[e; h]`: its first
        // `hidden` columns multiply the embedding, the rest the residual.
        b.read_expr(&mtp.e_proj, || {
            Ok(Expr::src(n("eh_proj.weight")).slice(1, 0, hidden))
        })?;
        b.read_expr(&mtp.h_proj, || {
            Ok(Expr::src(n("eh_proj.weight")).slice(1, hidden, hidden))
        })?;
        b.read(&mtp.mixer_norm, n("input_layernorm.weight"))?;
        b.read(&mtp.mlp_norm, n("post_attention_layernorm.weight"))?;
        mla(&mut b, &n, &mtp.attn)?;
        moe(&mut b, &n, &mtp.mlp)?;
        b.read(&mtp.norm, n("shared_head.norm.weight"))?;
        Ok(b.b.build())
    }
}

fn moe(b: &mut Land, n: &dyn Fn(&str) -> String, mlp: &Mlp) -> Result<(), Error> {
    let Mlp::Routed {
        router,
        bias,
        gate_up,
        down,
        shared,
        experts,
        ..
    } = mlp
    else {
        return Ok(());
    };
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
            vec![
                n("mlp.shared_experts.gate_proj.weight"),
                n("mlp.shared_experts.up_proj.weight"),
            ],
        )?;
        b.read(&s.down, n("mlp.shared_experts.down_proj.weight"))?;
    }
    Ok(())
}

fn mla(b: &mut Land, n: &dyn Fn(&str) -> String, a: &Mla) -> Result<(), Error> {
    b.read(&a.q_a_proj, n("self_attn.q_a_proj.weight"))?;
    b.read(&a.q_a_norm, n("self_attn.q_a_layernorm.weight"))?;
    b.read(&a.q_b_proj, n("self_attn.q_b_proj.weight"))?;
    b.read(&a.kv_a_proj, n("self_attn.kv_a_proj_with_mqa.weight"))?;
    b.read(&a.kv_a_norm, n("self_attn.kv_a_layernorm.weight"))?;
    b.read(&a.kv_b_proj, n("self_attn.kv_b_proj.weight"))?;
    b.read(&a.o_proj, n("self_attn.o_proj.weight"))?;
    indexer(b, n, &a.indexer)
}

fn indexer(b: &mut Land, n: &dyn Fn(&str) -> String, ix: &Indexer) -> Result<(), Error> {
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
    b: &mut Land,
    n: &dyn Fn(&str) -> String,
    k: &Kda,
) -> Result<(), Error> {
    b.read_concat(
        &k.qkv,
        vec![
            n("self_attn.q_proj.weight"),
            n("self_attn.k_proj.weight"),
            n("self_attn.v_proj.weight"),
        ],
    )?;
    // Each conv bank is stored [channels, 1, kernel] (the 1 is `groups`); the
    // declared type is rank-2, so each leg is squeezed before concatenation.
    b.read_expr(&k.conv, || {
        Ok(Expr::concat(
            as_axis(cut_axis(&k.conv), &k.conv.name),
            vec![
                squeezed(src, n("self_attn.q_conv1d.weight"))?,
                squeezed(src, n("self_attn.k_conv1d.weight"))?,
                squeezed(src, n("self_attn.v_conv1d.weight"))?,
            ],
        ))
    })?;
    b.read(&k.f_a, n("self_attn.f_a_proj.weight"))?;
    b.read(&k.f_b, n("self_attn.f_b_proj.weight"))?;
    b.read(&k.g_a, n("self_attn.g_a_proj.weight"))?;
    b.read(&k.g_b, n("self_attn.g_b_proj.weight"))?;
    b.read(&k.b, n("self_attn.b_proj.weight"))?;
    // Stored flat [heads * head_dim]; the text states it per head.
    b.read_expr(&k.dt_bias, || {
        Ok(Expr::src(n("self_attn.dt_bias")).transmute(TensorType::new(
            vec![extent(u64::from(k.heads)), extent(u64::from(k.head_dim))],
            checkpoint_dsl::encoding(model_dsl::Dtype::F32),
        )))
    })?;
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
    let stored = checkpoint::file::encoding_of(&tensor).map_err(|why| illegible(&why))?;
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
