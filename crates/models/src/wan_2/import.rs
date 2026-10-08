use checkpoint::contract::{Expr, ModelContract, TensorType};
use checkpoint_dsl::format::{Format, read_one};
use checkpoint_dsl::{Builder, Error, extents, stored_encoding};
use poem_dsl::{Platform, Weight};

use super::model::{
    Block, Conv, Dims, Dit, Downsampler, HEAD_SLICES, Linear, MOD_SLICES, Model, Resnet,
    TextEncoder, VAE_LATENTS_MEAN, VAE_LATENTS_STD, VAE_PATCH, VAE_RGB, VAE_Z, Vae, VaeEncoder,
};

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
        read_one("wan_2", src, formats)
    }

    fn import_from(
        &self,
        src: &ztensor::Source,
        platform: Platform,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, 1, platform);
        dit(&mut b, src, &self.dit, &self.dims, layout)?;
        if let Some(te) = &self.te {
            text_encoder(&mut b, te, layout)?;
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
    d: &Dims,
    layout: Layout,
) -> Result<(), Error> {
    let at = |tail: &str| layout.dit(tail);
    let dim = d.dim;

    transmuted(b, src, &m.patch_embed.w, &at("patch_embedding.weight"))?;
    b.read(&m.patch_embed.bias, at("patch_embedding.bias"))?;

    let cond = |s: &str| at(&format!("condition_embedder.{s}"));
    biased(b, &m.text_embed.linear_1, &cond("text_embedder.linear_1"))?;
    biased(b, &m.text_embed.linear_2, &cond("text_embedder.linear_2"))?;
    biased(b, &m.time_embed.linear_1, &cond("time_embedder.linear_1"))?;
    biased(b, &m.time_embed.linear_2, &cond("time_embedder.linear_2"))?;
    let l2 = cond("time_embedder.linear_2");
    let doubled = |e: Expr| Expr::concat(0, vec![e.clone(), e]);
    b.read_over(&m.head_proj.w, format!("{l2}.weight"), doubled)?;
    b.read_over(&m.head_proj.bias, format!("{l2}.bias"), doubled)?;
    let proj = cond("time_proj");
    let swap = |axis: u8, width: i64| move |e: Expr| slices_reordered(&e, MOD_SLICES, width, axis);
    b.read_over(
        &m.time_proj.w,
        format!("{proj}.weight"),
        swap(0, i64::from(dim)),
    )?;
    b.read_over(
        &m.time_proj.bias,
        format!("{proj}.bias"),
        swap(0, i64::from(dim)),
    )?;

    for (i, block) in m.blocks.iter().enumerate() {
        transformer_block(b, src, block, &at(&format!("blocks.{i}")))?;
    }

    table(b, src, &m.head_table, &at("scale_shift_table"), HEAD_SLICES)?;
    let rows = head_rows(d.out_channels);
    b.read_over(&m.proj_out.w, at("proj_out.weight"), |e| {
        e.gather(0, rows.clone())
    })?;
    b.read_over(&m.proj_out.bias, at("proj_out.bias"), |e| e.gather(0, rows))
}

fn transformer_block(
    b: &mut Builder,
    src: &ztensor::Source,
    block: &Block,
    stem: &str,
) -> Result<(), Error> {
    let n = |s: &str| format!("{stem}.{s}");
    table(b, src, &block.table, &n("scale_shift_table"), MOD_SLICES)?;

    let a = &block.self_attn;
    packed(
        b,
        &a.qkv,
        &[n("attn1.to_q"), n("attn1.to_k"), n("attn1.to_v")],
    )?;
    b.read(&a.norm_q, n("attn1.norm_q.weight"))?;
    b.read(&a.norm_k, n("attn1.norm_k.weight"))?;
    biased(b, &a.out, &n("attn1.to_out.0"))?;

    b.read(&block.norm2, n("norm2.weight"))?;
    b.read(&block.norm2_bias, n("norm2.bias"))?;

    let c = &block.cross;
    biased(b, &c.q, &n("attn2.to_q"))?;
    packed(b, &c.kv, &[n("attn2.to_k"), n("attn2.to_v")])?;
    b.read(&c.norm_q, n("attn2.norm_q.weight"))?;
    b.read(&c.norm_k, n("attn2.norm_k.weight"))?;
    biased(b, &c.out, &n("attn2.to_out.0"))?;

    biased(b, &block.ffn.up, &n("ffn.net.0.proj"))?;
    biased(b, &block.ffn.down, &n("ffn.net.2"))
}

fn text_encoder(b: &mut Builder, te: &TextEncoder, layout: Layout) -> Result<(), Error> {
    let at = |tail: &str| layout.component("te.", tail);
    b.read(&te.embed, at("shared.weight")?)?;
    for (l, w) in te.layers.iter().enumerate() {
        let attn = |s: &str| at(&format!("encoder.block.{l}.layer.0.{s}"));
        let ffn = |s: &str| at(&format!("encoder.block.{l}.layer.1.{s}"));
        b.read(&w.attn_norm, attn("layer_norm.weight")?)?;
        b.read(&w.q, attn("SelfAttention.q.weight")?)?;
        b.read(&w.k, attn("SelfAttention.k.weight")?)?;
        b.read(&w.v, attn("SelfAttention.v.weight")?)?;
        b.read(&w.o, attn("SelfAttention.o.weight")?)?;
        b.read(
            &w.rel_bias,
            attn("SelfAttention.relative_attention_bias.weight")?,
        )?;
        b.read(&w.ffn_norm, ffn("layer_norm.weight")?)?;
        b.read(&w.wi_0, ffn("DenseReluDense.wi_0.weight")?)?;
        b.read(&w.wi_1, ffn("DenseReluDense.wi_1.weight")?)?;
        b.read(&w.wo, ffn("DenseReluDense.wo.weight")?)?;
    }
    b.read(&te.final_norm, at("encoder.final_layer_norm.weight")?)
}

fn vae(b: &mut Builder, src: &ztensor::Source, v: &Vae, layout: Layout) -> Result<(), Error> {
    let at = |tail: &str| layout.component("vae.", tail);

    row_of(b, src, &v.denorm_scale, &at("post_quant_conv.bias")?, |i| {
        VAE_LATENTS_STD[i]
    })?;
    row_of(b, src, &v.denorm_bias, &at("post_quant_conv.bias")?, |i| {
        -VAE_LATENTS_MEAN[i] / VAE_LATENTS_STD[i]
    })?;

    conv(b, src, &v.post_quant, &at("post_quant_conv")?, None)?;
    conv(b, src, &v.conv_in, &at("decoder.conv_in")?, None)?;

    resnet(b, src, &v.mid_res0, &at("decoder.mid_block.resnets.0")?)?;
    let attn = at("decoder.mid_block.attentions.0")?;
    gamma(b, src, &v.mid_attn.norm, &format!("{attn}.norm.gamma"))?;
    transmuted(b, src, &v.mid_attn.qkv.w, &format!("{attn}.to_qkv.weight"))?;
    b.read(&v.mid_attn.qkv.bias, format!("{attn}.to_qkv.bias"))?;
    transmuted(b, src, &v.mid_attn.proj.w, &format!("{attn}.proj.weight"))?;
    b.read(&v.mid_attn.proj.bias, format!("{attn}.proj.bias"))?;
    resnet(b, src, &v.mid_res1, &at("decoder.mid_block.resnets.1")?)?;

    for (i, up) in v.up.iter().enumerate() {
        for (r, block) in up.resnets.iter().enumerate() {
            resnet(
                b,
                src,
                block,
                &at(&format!("decoder.up_blocks.{i}.resnets.{r}"))?,
            )?;
        }
        if let Some(u) = &up.upsampler {
            let stem = at(&format!("decoder.up_blocks.{i}.upsampler"))?;
            if let Some(tc) = &u.time_conv {
                conv(
                    b,
                    src,
                    tc,
                    &format!("{stem}.time_conv"),
                    Some(time_conv_rows(tc.c_in)),
                )?;
            }
            conv(b, src, &u.resample, &format!("{stem}.resample.1"), None)?;
        }
    }

    gamma(b, src, &v.norm_out, &at("decoder.norm_out.gamma")?)?;
    conv(
        b,
        src,
        &v.conv_out,
        &at("decoder.conv_out")?,
        Some(conv_out_rows()),
    )?;
    encoder(b, src, &v.enc, layout)
}

fn encoder(
    b: &mut Builder,
    src: &ztensor::Source,
    e: &VaeEncoder,
    layout: Layout,
) -> Result<(), Error> {
    let at = |tail: &str| layout.component("vae.", tail);

    row_of(b, src, &e.norm_bias, &at("quant_conv.bias")?, |i| {
        VAE_LATENTS_MEAN[i]
    })?;
    row_of(b, src, &e.norm_scale, &at("quant_conv.bias")?, |i| {
        1.0 / VAE_LATENTS_STD[i]
    })?;

    conv_over_channels(b, src, &e.conv_in, &at("encoder.conv_in")?, conv_out_rows())?;

    for (i, block) in e.down.iter().enumerate() {
        for (r, res) in block.resnets.iter().enumerate() {
            resnet(
                b,
                src,
                res,
                &at(&format!("encoder.down_blocks.{i}.resnets.{r}"))?,
            )?;
        }
        if let Some(Downsampler {
            resample,
            time_conv,
        }) = &block.downsampler
        {
            let stem = at(&format!("encoder.down_blocks.{i}.downsampler"))?;
            conv(b, src, resample, &format!("{stem}.resample.1"), None)?;
            if let Some(tc) = time_conv {
                conv(b, src, tc, &format!("{stem}.time_conv"), None)?;
            }
        }
    }

    resnet(b, src, &e.mid_res0, &at("encoder.mid_block.resnets.0")?)?;
    let attn = at("encoder.mid_block.attentions.0")?;
    gamma(b, src, &e.mid_attn.norm, &format!("{attn}.norm.gamma"))?;
    transmuted(b, src, &e.mid_attn.qkv.w, &format!("{attn}.to_qkv.weight"))?;
    b.read(&e.mid_attn.qkv.bias, format!("{attn}.to_qkv.bias"))?;
    transmuted(b, src, &e.mid_attn.proj.w, &format!("{attn}.proj.weight"))?;
    b.read(&e.mid_attn.proj.bias, format!("{attn}.proj.bias"))?;
    resnet(b, src, &e.mid_res1, &at("encoder.mid_block.resnets.1")?)?;

    gamma(b, src, &e.norm_out, &at("encoder.norm_out.gamma")?)?;
    conv(b, src, &e.conv_out, &at("encoder.conv_out")?, None)?;
    let quant = at("quant_conv")?;
    let name = format!("{quant}.weight");
    let stored = stored_encoding(src, &name)?;
    let shape = TensorType::new(extents(&e.quant.w), stored);
    let rows = i64::from(VAE_Z);
    b.read_over(&e.quant.w, name, move |x| {
        x.slice(0, 0, rows).transmute(shape)
    })?;
    b.read_over(&e.quant.bias, format!("{quant}.bias"), move |x| {
        x.slice(0, 0, rows)
    })
}

fn conv_over_channels(
    b: &mut Builder,
    src: &ztensor::Source,
    c: &Conv,
    stem: &str,
    cols: Vec<i64>,
) -> Result<(), Error> {
    let name = format!("{stem}.weight");
    let stored = stored_encoding(src, &name)?;
    let shape = TensorType::new(extents(&c.w), stored);
    b.read_over(&c.w, name, move |e| e.gather(1, cols).transmute(shape))?;
    b.read(&c.bias, format!("{stem}.bias"))
}

fn biased(b: &mut Builder, w: &Linear, stem: &str) -> Result<(), Error> {
    b.read(&w.w, format!("{stem}.weight"))?;
    b.read(&w.bias, format!("{stem}.bias"))
}

fn packed(b: &mut Builder, w: &Linear, stems: &[String]) -> Result<(), Error> {
    b.read_concat(&w.w, stems.iter().map(|s| format!("{s}.weight")))?;
    b.read_concat(&w.bias, stems.iter().map(|s| format!("{s}.bias")))
}

fn transmuted(b: &mut Builder, src: &ztensor::Source, w: &Weight, name: &str) -> Result<(), Error> {
    let stored = stored_encoding(src, name)?;
    let shape = TensorType::new(extents(w), stored);
    b.read_over(w, name.to_string(), |e| e.transmute(shape))
}

fn table(
    b: &mut Builder,
    src: &ztensor::Source,
    w: &Weight,
    name: &str,
    slices: u32,
) -> Result<(), Error> {
    let stored = stored_encoding(src, name)?;
    let shape = TensorType::new(extents(w), stored);
    b.read_over(w, name.to_string(), |e| {
        slices_reordered(&e, slices, 1, 1).transmute(shape)
    })
}

fn slices_reordered(from: &Expr, slices: u32, width: i64, axis: u8) -> Expr {
    let take = |i: i64| from.clone().slice(axis, i * width, width);
    let order: Vec<i64> = match slices {
        2 => vec![1, 0],
        6 => vec![1, 0, 2, 4, 3, 5],
        other => panic!("Wan states modulation vectors of 2 or 6 slices, not {other}"),
    };
    Expr::concat(axis, order.into_iter().map(take).collect())
}

fn head_rows(c_out: u32) -> Vec<i64> {
    let c_out = i64::from(c_out);
    let mut rows = Vec::with_capacity((4 * c_out) as usize);
    for c in 0..c_out {
        for ph in 0..2 {
            for pw in 0..2 {
                rows.push((ph * 2 + pw) * c_out + c);
            }
        }
    }
    rows
}

fn time_conv_rows(c: u32) -> Vec<i64> {
    let c = i64::from(c);
    (0..c).flat_map(|ch| [ch, c + ch]).collect()
}

fn conv_out_rows() -> Vec<i64> {
    let p = i64::from(VAE_PATCH);
    let mut rows = Vec::new();
    for c in 0..i64::from(VAE_RGB) {
        for ph in 0..p {
            for pw in 0..p {
                rows.push(c * p * p + pw * p + ph);
            }
        }
    }
    rows
}

fn conv(
    b: &mut Builder,
    src: &ztensor::Source,
    c: &Conv,
    stem: &str,
    rows: Option<Vec<i64>>,
) -> Result<(), Error> {
    let name = format!("{stem}.weight");
    let stored = stored_encoding(src, &name)?;
    let shape = TensorType::new(extents(&c.w), stored);
    let gathered = rows.clone();
    b.read_over(&c.w, name, move |e| {
        let kernel = match gathered {
            Some(rows) => e.gather(0, rows),
            None => e,
        };
        kernel.transmute(shape)
    })?;
    let bias = format!("{stem}.bias");
    match rows {
        Some(rows) => b.read_over(&c.bias, bias, move |e| e.gather(0, rows)),
        None => b.read(&c.bias, bias),
    }
}

fn row_of(
    b: &mut Builder,
    src: &ztensor::Source,
    w: &Weight,
    seed: &str,
    value: impl Fn(usize) -> f32,
) -> Result<(), Error> {
    let stored = stored_encoding(src, seed)?;
    let checkpoint::types::Encoding::Raw(dtype) = stored.clone() else {
        return Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!("`{seed}` is stored {stored:?}; a stated row wants a raw dtype"),
        });
    };
    let shape = extents(w);
    let n = shape.iter().product::<i64>();
    let values: Vec<f32> = (0..n as usize).map(value).collect();
    let want = checkpoint_dsl::encoding(w.dtype);
    let row = checkpoint_dsl::constant(&w.name, &values, vec![n], dtype)?;
    let row = if want == stored {
        row
    } else {
        row.cast(want.clone())
    };
    b.push(checkpoint::contract::TensorContract::new(
        w.name.clone(),
        row,
        shape,
        want,
    ));
    Ok(())
}

fn gamma(b: &mut Builder, src: &ztensor::Source, w: &Weight, name: &str) -> Result<(), Error> {
    transmuted(b, src, w, name)
}

fn resnet(b: &mut Builder, src: &ztensor::Source, r: &Resnet, stem: &str) -> Result<(), Error> {
    gamma(b, src, &r.norm1, &format!("{stem}.norm1.gamma"))?;
    conv(b, src, &r.conv1, &format!("{stem}.conv1"), None)?;
    gamma(b, src, &r.norm2, &format!("{stem}.norm2.gamma"))?;
    conv(b, src, &r.conv2, &format!("{stem}.conv2"), None)?;
    if let Some(s) = &r.shortcut {
        conv(b, src, s, &format!("{stem}.conv_shortcut"), None)?;
    }
    Ok(())
}
