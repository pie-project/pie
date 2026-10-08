use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};
use checkpoint::types::Encoding;
use poem::import::{Builder, Error, encoding, extents, stored_encoding};
use poem::{Platform, Weight};

use super::model::{Conv, Embedder, GroupNorm, Linear, Model, ResBlock};

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, 1, platform);
        let d = &self.dims;

        b.read(&self.embed, "model.wte.weight")?;
        b.read(&self.head, "lm_head.weight")?;
        b.read(&self.final_norm, "model.ln_f.weight")?;
        signs(&mut b, src, &self.ones, "model.ln_f.weight")?;

        let head_dim = d.head_dim;
        let groups = d.q_heads / d.kv_heads;
        let q_perm = head_permutation(d.q_heads, head_dim);
        let k_perm = head_permutation(d.kv_heads, head_dim);
        let rope_perm = rope_channels(head_dim);

        for (l, w) in self.layers.iter().enumerate() {
            let at = |tail: &str| format!("model.layers.{l}.{tail}");
            b.read(&w.attn_norm, at("input_layernorm.weight"))?;
            b.read(&w.mlp_norm, at("post_attention_layernorm.weight"))?;
            b.read_expr(
                &w.qkv,
                Expr::src(at("self_attn.qkv_proj.weight"))
                    .gather(0, qkv_rows(d.kv_heads, groups, head_dim, &q_perm, &k_perm)),
            )?;
            b.read_expr(
                &w.q_norm,
                Expr::src(at("self_attn.query_layernorm.weight")).gather(0, rope_perm.clone()),
            )?;
            b.read_expr(
                &w.k_norm,
                Expr::src(at("self_attn.key_layernorm.weight")).gather(0, rope_perm.clone()),
            )?;
            b.read(&w.o_proj, at("self_attn.o_proj.weight"))?;
            b.read(&w.router, at("mlp.gate.wg.weight"))?;

            let inter = i64::from(self.moe_inter);
            let hidden = i64::from(d.hidden);
            let shared = i64::from(self.shared_inter);
            b.read_expr(
                &w.shared_gate_up,
                swap_halves(
                    Expr::src(at("mlp.shared_mlp.gate_and_up_proj.weight")),
                    shared,
                ),
            )?;
            b.read(&w.shared_down, at("mlp.shared_mlp.down_proj.weight"))?;

            let stored = stored_encoding(src, &at("mlp.experts.0.down_proj.weight"))?;
            b.read_expr(
                &w.experts_gate_up,
                Expr::concat(
                    0,
                    (0..d.experts)
                        .map(|e| {
                            slab(
                                swap_halves(
                                    Expr::src(at(&format!(
                                        "mlp.experts.{e}.gate_and_up_proj.weight"
                                    ))),
                                    inter,
                                ),
                                vec![1, 2 * inter, hidden],
                                stored.clone(),
                            )
                        })
                        .collect(),
                ),
            )?;
            b.read_expr(
                &w.experts_down,
                Expr::concat(
                    0,
                    (0..d.experts)
                        .map(|e| {
                            slab(
                                Expr::src(at(&format!("mlp.experts.{e}.down_proj.weight"))),
                                vec![1, hidden, inter],
                                stored.clone(),
                            )
                        })
                        .collect(),
                ),
            )?;
        }

        b.read(&self.timestep_emb.mlp_in.w, "timestep_emb.mlp.0.weight")?;
        b.read(&self.timestep_emb.mlp_in.bias, "timestep_emb.mlp.0.bias")?;
        b.read_expr(
            &self.timestep_emb.mlp_out.w,
            doubled(Expr::src("timestep_emb.mlp.2.weight")),
        )?;
        b.read_expr(
            &self.timestep_emb.mlp_out.bias,
            doubled(Expr::src("timestep_emb.mlp.2.bias")),
        )?;
        embedder(&mut b, &self.time_embed, "time_embed")?;
        embedder(&mut b, &self.time_embed_2, "time_embed_2")?;

        conv(
            &mut b,
            src,
            &self.patch_embed.conv_in,
            "patch_embed.model.0",
        )?;
        resblock(&mut b, src, &self.patch_embed.res, "patch_embed.model.1")?;
        resblock(&mut b, src, &self.final_layer.res, "final_layer.model.0")?;
        group_norm(&mut b, &self.final_layer.norm_out, "final_layer.model.1.0")?;
        conv(
            &mut b,
            src,
            &self.final_layer.conv_out,
            "final_layer.model.1.2",
        )?;

        Ok(b.build())
    }
}

fn embedder(b: &mut Builder, e: &Embedder, prefix: &str) -> Result<(), Error> {
    b.read(&e.mlp_in.w, format!("{prefix}.mlp.0.weight"))?;
    b.read(&e.mlp_in.bias, format!("{prefix}.mlp.0.bias"))?;
    b.read(&e.mlp_out.w, format!("{prefix}.mlp.2.weight"))?;
    b.read(&e.mlp_out.bias, format!("{prefix}.mlp.2.bias"))
}

fn conv(b: &mut Builder, src: &ztensor::Source, c: &Conv, name: &str) -> Result<(), Error> {
    let shape = extents(&c.w);
    let stored = stored_encoding(src, &format!("{name}.weight"))?;
    b.read_expr(
        &c.w,
        Expr::src(format!("{name}.weight")).transmute(TensorType::new(shape, stored)),
    )?;
    b.read(&c.bias, format!("{name}.bias"))
}

fn group_norm(b: &mut Builder, g: &GroupNorm, name: &str) -> Result<(), Error> {
    b.read(&g.weight, format!("{name}.weight"))?;
    b.read(&g.bias, format!("{name}.bias"))
}

fn resblock(
    b: &mut Builder,
    src: &ztensor::Source,
    r: &ResBlock,
    prefix: &str,
) -> Result<(), Error> {
    group_norm(b, &r.norm_in, &format!("{prefix}.in_layers.0"))?;
    conv(b, src, &r.conv_in, &format!("{prefix}.in_layers.2"))?;
    linear(b, &r.emb, &format!("{prefix}.emb_layers.1"))?;
    group_norm(b, &r.norm_out, &format!("{prefix}.out_layers.0"))?;
    conv(b, src, &r.conv_out, &format!("{prefix}.out_layers.3"))?;
    match &r.skip {
        Some(c) => conv(b, src, c, &format!("{prefix}.skip_connection")),
        None => Ok(()),
    }
}

fn linear(b: &mut Builder, w: &Linear, name: &str) -> Result<(), Error> {
    b.read(&w.w, format!("{name}.weight"))?;
    b.read(&w.bias, format!("{name}.bias"))
}

fn swap_halves(src: Expr, inter: i64) -> Expr {
    Expr::concat(
        0,
        vec![src.clone().slice(0, inter, inter), src.slice(0, 0, inter)],
    )
}

fn doubled(src: Expr) -> Expr {
    Expr::concat(0, vec![src.clone(), src])
}

fn slab(expr: Expr, shape: Vec<i64>, stored: Encoding) -> Expr {
    expr.transmute(TensorType::new(shape, stored))
}

fn rope_channels(head_dim: u32) -> Vec<i64> {
    let d = i64::from(head_dim);
    let quarter = d / 4;
    let mut perm = Vec::with_capacity(d as usize);
    for j in 0..quarter {
        perm.push(2 * j);
    }
    for j in 0..quarter {
        perm.push(2 * j + d / 2);
    }
    for j in 0..quarter {
        perm.push(2 * j + 1);
    }
    for j in 0..quarter {
        perm.push(2 * j + 1 + d / 2);
    }
    perm
}

fn head_permutation(heads: u32, head_dim: u32) -> Vec<i64> {
    let channels = rope_channels(head_dim);
    let d = i64::from(head_dim);
    (0..i64::from(heads))
        .flat_map(|h| channels.iter().map(move |c| h * d + c))
        .collect()
}

fn qkv_rows(kv_heads: u32, groups: u32, head_dim: u32, q_perm: &[i64], k_perm: &[i64]) -> Vec<i64> {
    let d = i64::from(head_dim);
    let stride = i64::from(groups + 2) * d;
    let mut rows = Vec::new();
    for q in q_perm {
        let head = q / d;
        let c = q % d;
        let (g, j) = (head / i64::from(groups), head % i64::from(groups));
        rows.push(g * stride + j * d + c);
    }
    for k in k_perm {
        let head = k / d;
        let c = k % d;
        rows.push(head * stride + i64::from(groups) * d + c);
    }
    for g in 0..i64::from(kv_heads) {
        for c in 0..d {
            rows.push(g * stride + (i64::from(groups) + 1) * d + c);
        }
    }
    rows
}

fn signs(b: &mut Builder, src: &ztensor::Source, w: &Weight, seed: &str) -> Result<(), Error> {
    let stored = stored_encoding(src, seed)?;
    let Encoding::Raw(dtype) = stored.clone() else {
        return Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!("`{seed}` is stored {stored:?}; a constant is stated in a raw dtype"),
        });
    };
    let shape = extents(w);
    let half = usize::try_from(shape[0] / 2).expect("a sign table's half fits");
    let values: Vec<f32> = [1.0f32, -1.0].iter().flat_map(|v| vec![*v; half]).collect();
    let joined = poem::import::constant(&w.name, &values, vec![2 * half as i64, 1], dtype)?;
    let want = encoding(w.dtype);
    let expr = if want == stored {
        joined
    } else {
        joined.cast(want.clone())
    };
    b.push(TensorContract::new(w.name.clone(), expr, shape, want));
    Ok(())
}
