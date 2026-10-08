use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType, UnaryOp};
use checkpoint::types::Encoding;
use poem::import::format::{Format, read_one};
use poem::import::{Builder, Error, encoding, extents, stored_encoding};
use poem::{Platform, Weight};

use super::model::{
    Attn, DOUBLE_MOD_SLICES, Dit, Model, SINGLE_MOD_SLICES, Swiglu, TE_HIDDEN, TextEncoder,
};
use super::vae::{BN_EPS, ConvW, Mid, Norm, POSTERIOR_STORED, Proj, ResBlock, Vae};

#[derive(Clone, Copy)]
enum Layout {
    Diffusers,
    Bare,
}

impl Layout {
    fn spelling(self) -> &'static str {
        match self {
            Self::Diffusers => "a diffusers pipeline (`dit.`/`te.`/`vae.` prefixes)",
            Self::Bare => "a bare transformer state_dict",
        }
    }

    fn dit(self, tail: &str) -> String {
        match self {
            Self::Diffusers => format!("dit.{tail}"),
            Self::Bare => tail.to_string(),
        }
    }

    fn component(self, prefix: &str, tail: &str) -> Result<String, Error> {
        match self {
            Self::Diffusers => Ok(format!("{prefix}{tail}")),
            Self::Bare => Err(Error::Illegible {
                name: format!("{prefix}{tail}"),
                detail: format!(
                    "{} carries no `{prefix}` component, and this row declares one",
                    self.spelling()
                ),
            }),
        }
    }
}

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        // A pipeline's components ride under their prefixes; a bare
        // transformer file holds the denoiser's names as they are.
        let pipeline = |src: &ztensor::Source| src.names().any(|name| name.starts_with("dit."));
        let mut formats = vec![Format::new(Layout::Diffusers.spelling(), pipeline, || {
            self.import_from(src, platform, Layout::Diffusers)
        })];
        if self.te.is_none() && self.vae.is_none() {
            formats.push(Format::new(
                Layout::Bare.spelling(),
                move |src| !pipeline(src),
                || self.import_from(src, platform, Layout::Bare),
            ));
        }
        read_one("flux_2", src, formats)
    }

    pub fn import_vae(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        let v = self.vae.as_ref().ok_or_else(|| Error::Illegible {
            name: "vae".to_string(),
            detail: "this row declares no VAE".to_string(),
        })?;
        let mut b = Builder::new(src, 1, platform);
        vae(&mut b, src, v, Layout::Diffusers)?;
        Ok(b.build())
    }

    fn import_from(
        &self,
        src: &ztensor::Source,
        platform: Platform,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, 1, platform);
        dit(&mut b, src, &self.dit, self.dims.dim, layout)?;
        if let Some(te) = &self.te {
            text_encoder(&mut b, src, te, layout)?;
        }
        if let Some(vae) = &self.vae {
            self::vae(&mut b, src, vae, layout)?;
        }
        Ok(b.build())
    }
}

fn dit(
    b: &mut Builder,
    src: &ztensor::Source,
    m: &Dit,
    dim: u32,
    layout: Layout,
) -> Result<(), Error> {
    let at = |tail: &str| layout.dit(tail);
    let w = |tail: &str| at(&format!("{tail}.weight"));

    b.read(&m.x_embed, w("x_embedder"))?;
    if let Some(ctx) = &m.context_embed {
        b.read(ctx, w("context_embedder"))?;
    }
    b.read(
        &m.t_embed.linear_1,
        w("time_guidance_embed.timestep_embedder.linear_1"),
    )?;
    b.read(
        &m.t_embed.linear_2,
        w("time_guidance_embed.timestep_embedder.linear_2"),
    )?;
    if let Some(g) = &m.g_embed {
        b.read(
            &g.linear_1,
            w("time_guidance_embed.guidance_embedder.linear_1"),
        )?;
        b.read(
            &g.linear_2,
            w("time_guidance_embed.guidance_embedder.linear_2"),
        )?;
    }

    reordered(
        b,
        &m.mod_img,
        &w("double_stream_modulation_img.linear"),
        DOUBLE_MOD_SLICES,
        dim,
    )?;
    reordered(
        b,
        &m.mod_txt,
        &w("double_stream_modulation_txt.linear"),
        DOUBLE_MOD_SLICES,
        dim,
    )?;
    reordered(
        b,
        &m.mod_single,
        &w("single_stream_modulation.linear"),
        SINGLE_MOD_SLICES,
        dim,
    )?;

    for (i, block) in m.double.iter().enumerate() {
        let stem = at(&format!("transformer_blocks.{i}"));
        attn(
            b,
            &block.img.attn,
            &stem,
            ["to_q", "to_k", "to_v"],
            ["norm_q", "norm_k"],
            "to_out.0",
        )?;
        swiglu(b, &block.img.ff, &format!("{stem}.ff"))?;
        attn(
            b,
            &block.txt.attn,
            &stem,
            ["add_q_proj", "add_k_proj", "add_v_proj"],
            ["norm_added_q", "norm_added_k"],
            "to_add_out",
        )?;
        swiglu(b, &block.txt.ff, &format!("{stem}.ff_context"))?;
    }

    for (i, block) in m.single.iter().enumerate() {
        let stem = at(&format!("single_transformer_blocks.{i}.attn"));
        b.read(&block.in_proj, format!("{stem}.to_qkv_mlp_proj.weight"))?;
        b.read(&block.q_norm, format!("{stem}.norm_q.weight"))?;
        b.read(&block.k_norm, format!("{stem}.norm_k.weight"))?;
        let out = format!("{stem}.to_out.weight");
        let inter = block.out_mlp.shape[1] as i64;
        column_block(b, src, &block.out_attn, &out, 0, i64::from(dim))?;
        column_block(b, src, &block.out_mlp, &out, i64::from(dim), inter)?;
    }

    b.read(&m.norm_out, w("norm_out.linear"))?;
    b.read(&m.proj_out, w("proj_out"))
}

fn attn(
    b: &mut Builder,
    a: &Attn,
    stem: &str,
    qkv: [&str; 3],
    norms: [&str; 2],
    out: &str,
) -> Result<(), Error> {
    let n = |s: &str| format!("{stem}.attn.{s}.weight");
    b.read_concat(&a.qkv, qkv.map(n))?;
    b.read(&a.q_norm, n(norms[0]))?;
    b.read(&a.k_norm, n(norms[1]))?;
    b.read(&a.out, n(out))
}

fn swiglu(b: &mut Builder, ff: &Swiglu, stem: &str) -> Result<(), Error> {
    b.read(&ff.linear_in, format!("{stem}.linear_in.weight"))?;
    b.read(&ff.linear_out, format!("{stem}.linear_out.weight"))
}

fn reordered(b: &mut Builder, w: &Weight, from: &str, slices: u32, dim: u32) -> Result<(), Error> {
    let take = |i: i64| Expr::src(from.to_string()).slice(0, i * i64::from(dim), i64::from(dim));
    let order: Vec<i64> = match slices {
        3 => vec![1, 0, 2],
        6 => vec![1, 0, 2, 4, 3, 5],
        other => panic!("FLUX.2 states modulation vectors of 3 or 6 slices, not {other}"),
    };
    b.read_expr(w, Expr::concat(0, order.into_iter().map(take).collect()))
}

fn text_encoder(
    b: &mut Builder,
    src: &ztensor::Source,
    te: &TextEncoder,
    layout: Layout,
) -> Result<(), Error> {
    let at = |tail: &str| layout.component("te.model.", tail);
    b.read(&te.embed, at("embed_tokens.weight")?)?;
    for (l, w) in te.layers.iter().enumerate() {
        let n = |s: &str| at(&format!("layers.{l}.{s}"));
        b.read(&w.attn_norm, n("input_layernorm.weight")?)?;
        b.read(&w.q, n("self_attn.q_proj.weight")?)?;
        b.read(&w.k, n("self_attn.k_proj.weight")?)?;
        b.read(&w.v, n("self_attn.v_proj.weight")?)?;
        b.read(&w.o, n("self_attn.o_proj.weight")?)?;
        b.read(&w.q_norm, n("self_attn.q_norm.weight")?)?;
        b.read(&w.k_norm, n("self_attn.k_norm.weight")?)?;
        b.read(&w.mlp_norm, n("post_attention_layernorm.weight")?)?;
        b.read_concat(
            &w.gate_up,
            [n("mlp.gate_proj.weight")?, n("mlp.up_proj.weight")?],
        )?;
        b.read(&w.down, n("mlp.down_proj.weight")?)?;
    }
    let embedder = layout.dit("context_embedder.weight");
    for (i, w) in te.context_embed.iter().enumerate() {
        let start = i64::from(TE_HIDDEN) * i as i64;
        column_block(b, src, w, &embedder, start, i64::from(TE_HIDDEN))?;
    }
    Ok(())
}

fn column_block(
    b: &mut Builder,
    src: &ztensor::Source,
    w: &Weight,
    from: &str,
    start: i64,
    len: i64,
) -> Result<(), Error> {
    let stored = stored_encoding(src, from)?;
    let want = encoding(w.dtype);
    let sliced = Expr::src(from.to_string()).slice(1, start, len);
    if stored == want {
        return b.read_expr(w, sliced);
    }
    let staged = format!("{}.read", w.name);
    b.push(TensorContract::new(staged.clone(), sliced, extents(w), stored).internal());
    b.push(TensorContract::new(
        w.name.clone(),
        Expr::out(staged).cast(want.clone()),
        extents(w),
        want,
    ));
    Ok(())
}

fn vae(b: &mut Builder, src: &ztensor::Source, v: &Vae, layout: Layout) -> Result<(), Error> {
    let at = |tail: &str| layout.component("vae.", tail);

    batch_norm(b, src, v, &at("bn.running_mean")?, &at("bn.running_var")?)?;
    conv(b, src, &v.post_quant_conv, &at("post_quant_conv")?)?;
    conv_head(b, src, &v.quant_conv, &at("quant_conv")?, POSTERIOR_STORED)?;

    let d = &v.decoder;
    conv(b, src, &d.conv_in, &at("decoder.conv_in")?)?;
    mid_block(b, src, &d.mid, &at("decoder.mid_block")?)?;
    for (i, up) in d.up.iter().enumerate() {
        for (r, block) in up.resnets.iter().enumerate() {
            resnet(
                b,
                src,
                block,
                &at(&format!("decoder.up_blocks.{i}.resnets.{r}"))?,
            )?;
        }
        if let Some(c) = &up.upsample {
            conv(
                b,
                src,
                c,
                &at(&format!("decoder.up_blocks.{i}.upsamplers.0.conv"))?,
            )?;
        }
    }
    group_norm(b, &d.norm_out, &at("decoder.conv_norm_out")?)?;
    conv(b, src, &d.conv_out, &at("decoder.conv_out")?)?;

    let e = &v.encoder;
    conv(b, src, &e.conv_in, &at("encoder.conv_in")?)?;
    for (i, down) in e.down.iter().enumerate() {
        for (r, block) in down.resnets.iter().enumerate() {
            resnet(
                b,
                src,
                block,
                &at(&format!("encoder.down_blocks.{i}.resnets.{r}"))?,
            )?;
        }
        if let Some(c) = &down.downsample {
            conv(
                b,
                src,
                c,
                &at(&format!("encoder.down_blocks.{i}.downsamplers.0.conv"))?,
            )?;
        }
    }
    mid_block(b, src, &e.mid, &at("encoder.mid_block")?)?;
    group_norm(b, &e.norm_out, &at("encoder.conv_norm_out")?)?;
    conv(b, src, &e.conv_out, &at("encoder.conv_out")?)
}

fn conv(b: &mut Builder, src: &ztensor::Source, c: &ConvW, stem: &str) -> Result<(), Error> {
    let name = format!("{stem}.weight");
    let stored = stored_encoding(src, &name)?;
    b.read_expr(
        &c.w,
        Expr::src(name).transmute(TensorType::new(extents(&c.w), stored)),
    )?;
    b.read(&c.bias, format!("{stem}.bias"))
}

fn conv_head(
    b: &mut Builder,
    src: &ztensor::Source,
    c: &ConvW,
    stem: &str,
    stored_rows: u32,
) -> Result<(), Error> {
    let kernel = format!("{stem}.weight");
    let stored = stored_encoding(src, &kernel)?;
    let mut natural = extents(&c.w);
    natural[0] = i64::from(stored_rows);
    let rows = i64::from(c.c_out);
    b.read_expr(
        &c.w,
        Expr::src(&kernel)
            .transmute(TensorType::new(natural, stored))
            .slice(0, 0, rows),
    )?;
    let bias = format!("{stem}.bias");
    let stored = stored_encoding(src, &bias)?;
    let want = encoding(c.bias.dtype);
    let head = format!("{}.head", c.bias.name);
    b.push(
        TensorContract::new(
            head.clone(),
            Expr::src(&bias).slice(0, 0, rows),
            extents(&c.bias),
            stored.clone(),
        )
        .internal(),
    );
    let expr = if want == stored {
        Expr::out(head)
    } else {
        Expr::out(head).cast(want.clone())
    };
    b.push(TensorContract::new(
        c.bias.name.clone(),
        expr,
        extents(&c.bias),
        want,
    ));
    Ok(())
}

fn group_norm(b: &mut Builder, n: &Norm, stem: &str) -> Result<(), Error> {
    b.read(&n.weight, format!("{stem}.weight"))?;
    b.read(&n.bias, format!("{stem}.bias"))
}

fn resnet(b: &mut Builder, src: &ztensor::Source, r: &ResBlock, stem: &str) -> Result<(), Error> {
    group_norm(b, &r.norm1, &format!("{stem}.norm1"))?;
    conv(b, src, &r.conv1, &format!("{stem}.conv1"))?;
    group_norm(b, &r.norm2, &format!("{stem}.norm2"))?;
    conv(b, src, &r.conv2, &format!("{stem}.conv2"))?;
    if let Some(s) = &r.shortcut {
        conv(b, src, s, &format!("{stem}.conv_shortcut"))?;
    }
    Ok(())
}

fn mid_block(b: &mut Builder, src: &ztensor::Source, m: &Mid, stem: &str) -> Result<(), Error> {
    resnet(b, src, &m.res0, &format!("{stem}.resnets.0"))?;
    let a = &m.attn;
    let n = |s: &str| format!("{stem}.attentions.0.{s}");
    group_norm(b, &a.norm, &n("group_norm"))?;
    biased(b, &a.q, &n("to_q"))?;
    biased(b, &a.k, &n("to_k"))?;
    biased(b, &a.v, &n("to_v"))?;
    biased(b, &a.out, &n("to_out.0"))?;
    resnet(b, src, &m.res1, &format!("{stem}.resnets.1"))
}

fn biased(b: &mut Builder, p: &Proj, stem: &str) -> Result<(), Error> {
    b.read(&p.w, format!("{stem}.weight"))?;
    b.read(&p.bias, format!("{stem}.bias"))
}

fn batch_norm(
    b: &mut Builder,
    src: &ztensor::Source,
    v: &Vae,
    mean: &str,
    var: &str,
) -> Result<(), Error> {
    let stored = stored_encoding(src, var)?;
    let shape = extents(&v.bn_scale);
    let var_eps = format!("{}.var_eps", v.bn_scale.name);
    b.push(
        TensorContract::new(
            var_eps.clone(),
            Expr::src(var.to_string()).bias(BN_EPS),
            shape.clone(),
            stored.clone(),
        )
        .internal(),
    );
    for (plane, op) in [(&v.bn_scale, UnaryOp::Sqrt), (&v.bn_rscale, UnaryOp::Rsqrt)] {
        let root = format!("{}.of_var", plane.name);
        b.push(
            TensorContract::new(
                root.clone(),
                Expr::out(var_eps.clone()).unary(op),
                shape.clone(),
                stored.clone(),
            )
            .internal(),
        );
        let want = encoding(plane.dtype);
        let published = if want == stored {
            Expr::out(root)
        } else {
            Expr::out(root).cast(want.clone())
        };
        b.push(TensorContract::new(
            plane.name.clone(),
            published,
            shape.clone(),
            want,
        ));
    }
    b.read(&v.bn_mean, mean.to_string())?;
    let want = encoding(v.bn_zero.dtype);
    let Encoding::Raw(dtype) = want.clone() else {
        return Err(Error::Illegible {
            name: v.bn_zero.name.clone(),
            detail: format!("a zero plane is stated raw, not {want:?}"),
        });
    };
    let shape = extents(&v.bn_zero);
    b.push(TensorContract::new(
        v.bn_zero.name.clone(),
        Expr::fill(0.0, TensorType::raw(shape.clone(), dtype)),
        shape,
        want,
    ));
    Ok(())
}
