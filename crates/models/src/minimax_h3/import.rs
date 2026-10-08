use checkpoint::contract::{Expr, ModelContract, TensorType};
use checkpoint::types::Encoding;
use checkpoint_dsl::format::{Format, read_one};
use checkpoint_dsl::{Builder, Error, extents, stored_encoding};
use poem_dsl::{Platform, Weight};

use super::model::{
    ADALN_SLICES, Attn, Block, Dit, Linear, MODALITIES, Mlp, Model, Refiner, TextEncoder,
};

#[derive(Clone, Copy)]
enum Layout {
    Diffusers,
    Bare,
}

impl Layout {
    fn spelling(self) -> &'static str {
        match self {
            Self::Diffusers => "a MiniMax H3 partition (`dit.`/`te.` prefixes)",
            Self::Bare => "a bare transformer state_dict",
        }
    }

    fn dit(self, tail: &str) -> String {
        match self {
            Self::Diffusers => format!("dit.{tail}"),
            Self::Bare => tail.to_string(),
        }
    }

    fn te(self, tail: &str) -> Option<String> {
        match self {
            Self::Diffusers => Some(format!("te.model.language_model.{tail}")),
            Self::Bare => None,
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
        if self.te.is_none() {
            formats.push(Format::new(
                Layout::Bare.spelling(),
                move |src| !pipeline(src),
                || self.import_from(src, platform, Layout::Bare),
            ));
        }
        read_one("minimax_h3", src, formats)
    }

    fn import_from(
        &self,
        src: &ztensor::Source,
        platform: Platform,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, 1, platform);
        dit(&mut b, src, &self.dims, &self.dit, layout)?;
        if let Some(te) = &self.te {
            text_encoder(&mut b, te, layout)?;
        }
        Ok(b.build())
    }
}

fn dit(
    b: &mut Builder,
    src: &ztensor::Source,
    d: &super::model::Dims,
    m: &Dit,
    layout: Layout,
) -> Result<(), Error> {
    let at = |tail: &str| layout.dit(tail);

    biased(b, &m.video_patch, &at("video_patch_proj"))?;
    biased(b, &m.audio_patch, &at("audio_patch_proj"))?;
    biased(b, &m.condition, &at("condition_proj"))?;
    biased(b, &m.t_in, &at("time_embedder.proj_in"))?;
    biased(b, &m.t_out, &at("time_embedder.proj_out"))?;

    for (i, block) in m.refine.iter().enumerate() {
        refiner(b, src, d, block, &at(&format!("token_refiner.blocks.{i}")))?;
    }
    b.read(&m.refine_norm, at("token_refiner.final_norm.weight"))?;

    for (i, block) in m.blocks.iter().enumerate() {
        dit_block(b, src, d, block, &at(&format!("blocks.{i}")))?;
    }

    b.read(&m.final_norm, at("final_layer.norm.weight"))?;
    pairs(
        b,
        &m.final_adaln,
        &at("final_layer.adaln_proj.linear"),
        &[1, 0],
        i64::from(d.dim),
    )?;
    biased(b, &m.video_out, &at("final_layer.video_out"))?;
    biased(b, &m.audio_out, &at("final_layer.audio_out"))?;
    Ok(())
}

fn dit_block(
    b: &mut Builder,
    src: &ztensor::Source,
    d: &super::model::Dims,
    block: &Block,
    stem: &str,
) -> Result<(), Error> {
    b.read(&block.norm1, format!("{stem}.norm1.weight"))?;
    b.read(&block.norm2, format!("{stem}.norm2.weight"))?;
    attention(b, src, d, &block.attn, &format!("{stem}.attn"))?;
    feedforward(b, &block.mlp, &format!("{stem}.mlp"))?;
    let dim = i64::from(d.dim);
    let bank = format!("{stem}.adaln_proj.linear");
    for (m, w) in block.adaln.iter().enumerate() {
        let base = i64::try_from(m).unwrap_or(0) * i64::from(ADALN_SLICES);
        let order = [base + 1, base, base + 2, base + 4, base + 3, base + 5];
        pairs(b, w, &bank, &order, dim)?;
    }
    debug_assert_eq!(block.adaln.len(), MODALITIES as usize);
    Ok(())
}

fn refiner(
    b: &mut Builder,
    src: &ztensor::Source,
    d: &super::model::Dims,
    block: &Refiner,
    stem: &str,
) -> Result<(), Error> {
    b.read(&block.norm1, format!("{stem}.norm1.weight"))?;
    b.read(&block.norm2, format!("{stem}.norm2.weight"))?;
    attention(b, src, d, &block.attn, &format!("{stem}.attn"))?;
    feedforward(b, &block.mlp, &format!("{stem}.mlp"))
}

fn attention(
    b: &mut Builder,
    src: &ztensor::Source,
    d: &super::model::Dims,
    a: &Attn,
    stem: &str,
) -> Result<(), Error> {
    let name = format!("{stem}.qkv_proj.weight");
    let dtype = raw_dtype(src, &name, &a.qkv)?;
    let heads = i64::from(d.heads);
    let group = i64::from(d.head_dim) * i64::from(d.dim);
    let want = extents(&a.qkv);
    b.read_over(&a.qkv, name, |e| {
        let grouped = e.transmute(TensorType::raw(vec![3 * heads, group], dtype));
        Expr::concat(
            0,
            (0..3)
                .map(|which| grouped.clone().stride(0, which, heads, 3))
                .collect(),
        )
        .transmute(TensorType::raw(want, dtype))
    })?;
    b.read(&a.q_norm, format!("{stem}.q_norm.weight"))?;
    b.read(&a.k_norm, format!("{stem}.k_norm.weight"))?;
    b.read(&a.out, format!("{stem}.out_proj.weight"))
}

fn feedforward(b: &mut Builder, m: &Mlp, stem: &str) -> Result<(), Error> {
    b.read(&m.fc1, format!("{stem}.fc1.weight"))?;
    b.read(&m.fc2, format!("{stem}.fc2.weight"))
}

fn biased(b: &mut Builder, w: &Linear, stem: &str) -> Result<(), Error> {
    b.read(&w.w, format!("{stem}.weight"))?;
    b.read(&w.bias, format!("{stem}.bias"))
}

fn pairs(b: &mut Builder, w: &Linear, stem: &str, order: &[i64], block: i64) -> Result<(), Error> {
    let rows = |e: &Expr| {
        Expr::concat(
            0,
            order
                .iter()
                .map(|&slice| e.clone().slice(0, slice * block, block))
                .collect(),
        )
    };
    b.read_over(&w.w, format!("{stem}.weight"), |e| rows(&e))?;
    b.read_over(&w.bias, format!("{stem}.bias"), |e| rows(&e))
}

fn raw_dtype(
    src: &ztensor::Source,
    name: &str,
    w: &Weight,
) -> Result<checkpoint::types::DType, Error> {
    match stored_encoding(src, name)? {
        Encoding::Raw(dtype) => Ok(dtype),
        other => Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!("`{name}` is stored {other:?}; a fused qkv bank is a raw plane"),
        }),
    }
}

fn text_encoder(b: &mut Builder, te: &TextEncoder, layout: Layout) -> Result<(), Error> {
    let at = |tail: &str| -> Result<String, Error> {
        layout.te(tail).ok_or_else(|| Error::Illegible {
            name: format!("te.{tail}"),
            detail: format!(
                "{} carries no text encoder, and this row declares one",
                layout.spelling()
            ),
        })
    };
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
    Ok(())
}
