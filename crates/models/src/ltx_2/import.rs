use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};
use checkpoint::types::Encoding;
use checkpoint_dsl::{Builder, Error, extents, stored_encoding};
use model_dsl::{Platform, Weight};

use super::model::{
    AV_GATE_SLICES, AV_SS_SLICES, AdaLn, Attn, Block, Connector, Dims, Dit, Ffn, HEAD_SLICES,
    Linear, MOD_SLICES, Model, PROMPT_SLICES, Side, Stream, VAE_PATCH, VAE_RGB, Vae, VaeConv,
    VaeResnet,
};

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp, platform);
        dit(&mut b, src, &self.dit, &self.dims)?;
        connector(&mut b, &self.connectors.0, "video", self.dims.caption)?;
        connector(&mut b, &self.connectors.1, "audio", self.dims.caption)?;
        if let Some(v) = &self.vae {
            vae(&mut b, src, v)?;
        }
        Ok(b.build())
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
        let mut b = Builder::new(src, self.tp, platform);
        vae(&mut b, src, v)?;
        Ok(b.build())
    }
}

fn vae(b: &mut Builder, src: &ztensor::Source, v: &Vae) -> Result<(), Error> {
    let at = |tail: &str| format!("vae.{tail}");
    b.read(&v.latents_mean, at("latents_mean"))?;
    b.read(&v.latents_std, at("latents_std"))?;
    zero_row(b, &v.zero)?;

    vae_conv(b, src, &v.conv_in, &at("decoder.conv_in"), None)?;
    for (r, res) in v.mid.iter().enumerate() {
        vae_resnet(b, src, res, &at(&format!("decoder.mid_block.resnets.{r}")))?;
    }
    for (i, up) in v.up.iter().enumerate() {
        let stem = at(&format!("decoder.up_blocks.{i}"));
        vae_conv(
            b,
            src,
            &up.upsampler,
            &format!("{stem}.upsamplers.0.conv"),
            None,
        )?;
        for (r, res) in up.resnets.iter().enumerate() {
            vae_resnet(b, src, res, &format!("{stem}.resnets.{r}"))?;
        }
    }
    vae_conv(
        b,
        src,
        &v.conv_out,
        &at("decoder.conv_out"),
        Some(conv_out_rows()),
    )
}

fn vae_resnet(
    b: &mut Builder,
    src: &ztensor::Source,
    r: &VaeResnet,
    stem: &str,
) -> Result<(), Error> {
    vae_conv(b, src, &r.conv1, &format!("{stem}.conv1"), None)?;
    vae_conv(b, src, &r.conv2, &format!("{stem}.conv2"), None)
}

fn vae_conv(
    b: &mut Builder,
    src: &ztensor::Source,
    c: &VaeConv,
    stem: &str,
    rows: Option<Vec<i64>>,
) -> Result<(), Error> {
    let name = format!("{stem}.conv.weight");
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
    let bias = format!("{stem}.conv.bias");
    match rows {
        Some(rows) => b.read_over(&c.bias, bias, move |e| e.gather(0, rows)),
        None => b.read(&c.bias, bias),
    }
}

fn conv_out_rows() -> Vec<i64> {
    let p = i64::from(VAE_PATCH);
    let mut rows = Vec::with_capacity((VAE_RGB * VAE_PATCH * VAE_PATCH) as usize);
    for c in 0..i64::from(VAE_RGB) {
        for ph in 0..p {
            for pw in 0..p {
                rows.push(c * p * p + pw * p + ph);
            }
        }
    }
    rows
}

fn zero_row(b: &mut Builder, w: &Weight) -> Result<(), Error> {
    let want = checkpoint_dsl::encoding(w.dtype);
    let Encoding::Raw(dtype) = want.clone() else {
        return Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!("declared {want:?}; a stated zero row wants a raw dtype"),
        });
    };
    b.push(TensorContract::new(
        w.name.clone(),
        Expr::fill(0.0, TensorType::raw(extents(w), dtype)),
        extents(w),
        want,
    ));
    Ok(())
}

fn dit(b: &mut Builder, src: &ztensor::Source, m: &Dit, d: &Dims) -> Result<(), Error> {
    let at = |tail: &str| format!("dit.{tail}");

    stream(
        b,
        src,
        &m.video,
        &at("proj_in"),
        &at("time_embed"),
        "",
        d.dim(),
    )?;
    stream(
        b,
        src,
        &m.audio,
        &at("audio_proj_in"),
        &at("audio_time_embed"),
        "audio_",
        d.audio_dim(),
    )?;
    adaln(b, &m.prompt, &at("prompt_adaln"), d.dim(), &[1, 0])?;
    adaln(
        b,
        &m.audio_prompt,
        &at("audio_prompt_adaln"),
        d.audio_dim(),
        &[1, 0],
    )?;

    for (i, block) in m.blocks.iter().enumerate() {
        transformer_block(b, src, block, &at(&format!("transformer_blocks.{i}")), d)?;
    }
    Ok(())
}

fn stream(
    b: &mut Builder,
    src: &ztensor::Source,
    s: &Stream,
    proj_in: &str,
    time_embed: &str,
    prefix: &str,
    dim: u32,
) -> Result<(), Error> {
    biased(b, &s.patchify, proj_in)?;
    adaln(b, &s.adaln, time_embed, dim, &[1, 0, 2, 4, 3, 5, 7, 6, 8])?;
    let l2 = format!("{time_embed}.emb.timestep_embedder.linear_2");
    let rows: Vec<i64> = (0..i64::from(dim)).chain(0..i64::from(dim)).collect();
    let twice = |tail: &str| Expr::src(format!("{l2}.{tail}")).gather(0, rows.clone());
    b.read_expr(&s.head_proj.w, twice("weight"))?;
    b.read_expr(
        s.head_proj
            .bias
            .as_ref()
            .expect("the head projection is biased"),
        twice("bias"),
    )?;
    table(
        b,
        src,
        &s.head_table,
        &format!("dit.{prefix}scale_shift_table"),
        HEAD_SLICES,
        &[1, 0],
    )?;
    adaln(
        b,
        &s.av_ss,
        &format!(
            "dit.av_cross_attn_{stem}_scale_shift",
            stem = av_stem(prefix)
        ),
        dim,
        &[0, 1, 2, 3],
    )?;
    adaln(
        b,
        &s.av_gate,
        &format!(
            "dit.av_cross_attn_{stem}_{gate}_gate",
            stem = av_stem(prefix),
            gate = av_gate(prefix)
        ),
        dim,
        &[0],
    )?;
    biased(b, &s.proj_out, &format!("dit.{prefix}proj_out"))
}

fn av_stem(prefix: &str) -> &'static str {
    if prefix.is_empty() { "video" } else { "audio" }
}

fn av_gate(prefix: &str) -> &'static str {
    if prefix.is_empty() { "a2v" } else { "v2a" }
}

fn adaln(b: &mut Builder, head: &AdaLn, stem: &str, dim: u32, order: &[i64]) -> Result<(), Error> {
    debug_assert_eq!(order.len(), head.slices as usize);
    let emb = format!("{stem}.emb.timestep_embedder");
    biased(b, &head.embed.linear_1, &format!("{emb}.linear_1"))?;
    biased(b, &head.embed.linear_2, &format!("{emb}.linear_2"))?;
    let proj = format!("{stem}.linear");
    b.read_expr(
        &head.proj.w,
        reordered(&format!("{proj}.weight"), order, i64::from(dim), 0),
    )?;
    b.read_expr(
        head.proj.bias.as_ref().expect("an adaLN linear is biased"),
        reordered(&format!("{proj}.bias"), order, i64::from(dim), 0),
    )
}

fn transformer_block(
    b: &mut Builder,
    src: &ztensor::Source,
    block: &Block,
    stem: &str,
    d: &Dims,
) -> Result<(), Error> {
    side(b, src, &block.video, stem, "", d.dim())?;
    side(b, src, &block.audio, stem, "audio_", d.audio_dim())?;
    attention(b, &block.a2v, &format!("{stem}.audio_to_video_attn"))?;
    attention(b, &block.v2a, &format!("{stem}.video_to_audio_attn"))
}

fn side(
    b: &mut Builder,
    src: &ztensor::Source,
    s: &Side,
    stem: &str,
    prefix: &str,
    dim: u32,
) -> Result<(), Error> {
    table(
        b,
        src,
        &s.table,
        &format!("{stem}.{prefix}scale_shift_table"),
        MOD_SLICES,
        &[1, 0, 2, 4, 3, 5, 7, 6, 8],
    )?;
    let av = format!(
        "{stem}.{stream}_a2v_cross_attn_scale_shift_table",
        stream = av_stem(prefix)
    );
    banded(b, src, &s.av_ss_table, &av, dim, 0, AV_SS_SLICES)?;
    banded(
        b,
        src,
        &s.av_gate_table,
        &av,
        dim,
        i64::from(AV_SS_SLICES),
        AV_GATE_SLICES,
    )?;
    table(
        b,
        src,
        &s.prompt_table,
        &format!("{stem}.{prefix}prompt_scale_shift_table"),
        PROMPT_SLICES,
        &[1, 0],
    )?;
    attention(b, &s.self_attn, &format!("{stem}.{prefix}attn1"))?;
    attention(b, &s.cross, &format!("{stem}.{prefix}attn2"))?;
    feed_forward(b, &s.ffn, &format!("{stem}.{prefix}ff"))
}

fn attention(b: &mut Builder, a: &Attn, stem: &str) -> Result<(), Error> {
    match &a.kv {
        None => packed(
            b,
            &a.qkv,
            &[
                format!("{stem}.to_q"),
                format!("{stem}.to_k"),
                format!("{stem}.to_v"),
            ],
        )?,
        Some(kv) => {
            biased(b, &a.qkv, &format!("{stem}.to_q"))?;
            packed(b, kv, &[format!("{stem}.to_k"), format!("{stem}.to_v")])?;
        }
    }
    b.read(&a.q_norm, format!("{stem}.norm_q.weight"))?;
    b.read(&a.k_norm, format!("{stem}.norm_k.weight"))?;
    biased(b, &a.gate, &format!("{stem}.to_gate_logits"))?;
    biased(b, &a.out, &format!("{stem}.to_out.0"))
}

fn feed_forward(b: &mut Builder, ff: &Ffn, stem: &str) -> Result<(), Error> {
    biased(b, &ff.up, &format!("{stem}.net.0.proj"))?;
    biased(b, &ff.down, &format!("{stem}.net.2"))
}

fn connector(b: &mut Builder, conn: &Connector, stem: &str, caption: u32) -> Result<(), Error> {
    let _ = caption;
    biased(
        b,
        &conn.aggregate,
        &format!("connectors.{stem}_text_proj_in"),
    )?;
    for (l, block) in conn.blocks.iter().enumerate() {
        let at = format!("connectors.{stem}_connector.transformer_blocks.{l}");
        attention(b, &block.attn, &format!("{at}.attn1"))?;
        feed_forward(b, &block.ffn, &format!("{at}.ff"))?;
    }
    Ok(())
}

fn biased(b: &mut Builder, w: &Linear, stem: &str) -> Result<(), Error> {
    b.read(&w.w, format!("{stem}.weight"))?;
    match &w.bias {
        Some(bias) => b.read(bias, format!("{stem}.bias")),
        None => Ok(()),
    }
}

fn packed(b: &mut Builder, w: &Linear, stems: &[String]) -> Result<(), Error> {
    b.read_concat(&w.w, stems.iter().map(|s| format!("{s}.weight")))?;
    b.read_concat(
        w.bias.as_ref().expect("a packed projection is biased"),
        stems.iter().map(|s| format!("{s}.bias")),
    )
}

fn table(
    b: &mut Builder,
    src: &ztensor::Source,
    w: &Weight,
    name: &str,
    slices: u32,
    order: &[i64],
) -> Result<(), Error> {
    debug_assert_eq!(order.len(), slices as usize);
    let stored = stored_encoding(src, name)?;
    b.read_expr(
        w,
        reordered(name, order, 1, 0).transmute(TensorType::new(extents(w), stored)),
    )
}

fn banded(
    b: &mut Builder,
    src: &ztensor::Source,
    w: &Weight,
    name: &str,
    _dim: u32,
    start: i64,
    len: u32,
) -> Result<(), Error> {
    let stored = stored_encoding(src, name)?;
    b.read_expr(
        w,
        Expr::src(name.to_string())
            .slice(0, start, i64::from(len))
            .transmute(TensorType::new(extents(w), stored)),
    )
}

fn reordered(from: &str, order: &[i64], width: i64, axis: u8) -> Expr {
    if order.iter().enumerate().all(|(at, &i)| i as usize == at) {
        return Expr::src(from.to_string());
    }
    let take = |i: i64| Expr::src(from.to_string()).slice(axis, i * width, width);
    Expr::concat(axis, order.iter().map(|&i| take(i)).collect())
}
