use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};
use checkpoint::types::{DType, Encoding};
use poem_dsl::{Dtype, Weight};

use super::model::Model;
use checkpoint_dsl::format::{Format, Stated, config, read_one};
use checkpoint_dsl::{
    Builder, Error, divided, encoding, extents, grouped, scaling, stored_encoding,
};
use poem_dsl::Platform;

const BANK_ROWS: u8 = 1;

#[derive(Clone, Copy)]
enum Layout {
    Transformers,
    Mlx,
}

impl Layout {
    fn codes(self) -> DType {
        match self {
            Self::Transformers => DType::U8,
            Self::Mlx => DType::U32,
        }
    }
}

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        // The two spellings share every name; the expert codes say which one a
        // checkpoint is, bytes under transformers and words under mlx_lm.
        let codes = |want: DType| {
            move |src: &ztensor::Source| {
                stored_encoding(src, "model.layers.0.mlp.experts.gate_up_proj_blocks")
                    .is_ok_and(|stored| stored == Encoding::Raw(want))
            }
        };
        read_one(
            "gpt_oss",
            src,
            vec![
                Format::new("transformers", codes(Layout::Transformers.codes()), || {
                    self.import_from(src, platform, Layout::Transformers)
                })
                .stating(self.configured()),
                Format::new("mlx_lm", codes(Layout::Mlx.codes()), || {
                    self.import_from(src, platform, Layout::Mlx)
                })
                .stating(self.configured()),
            ],
        )
    }

    /// The shape a transformers configuration states of this model.
    fn configured(&self) -> Vec<Stated> {
        let mut states = vec![
            config("hidden_size", self.hidden),
            config("vocab_size", self.vocab),
            config("num_attention_heads", self.q_heads),
            config("num_key_value_heads", self.kv_heads),
            config("head_dim", self.head_dim),
            config("num_hidden_layers", self.layers.len() as u32).or_deeper(),
            config("rms_norm_eps", self.final_norm_eps),
            config("sliding_window", self.window),
        ];
        if let Some(layer) = self.layers.first() {
            states.push(config("num_local_experts", layer.mlp.experts).or_deeper());
            states.push(config("rope_theta", layer.attn.theta));
        }
        states
    }

    fn import_from(
        &self,
        src: &ztensor::Source,
        platform: Platform,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, 1, platform);
        b.read(&self.embed, "model.embed_tokens.weight")?;
        b.read(&self.final_norm, "model.norm.weight")?;
        b.read(&self.head, "lm_head.weight")?;
        for (l, layer) in self.layers.iter().enumerate() {
            let ck = |what: &str| format!("model.layers.{l}.{what}");
            let attn = &layer.attn;
            let mlp = &layer.mlp;

            b.read(&layer.attn_norm, ck("input_layernorm.weight"))?;
            b.read(&layer.mlp_norm, ck("post_attention_layernorm.weight"))?;
            b.read(&attn.q_proj, ck("self_attn.q_proj.weight"))?;
            b.read(&attn.q_bias, ck("self_attn.q_proj.bias"))?;
            b.read(&attn.k_proj, ck("self_attn.k_proj.weight"))?;
            b.read(&attn.k_bias, ck("self_attn.k_proj.bias"))?;
            b.read(&attn.v_proj, ck("self_attn.v_proj.weight"))?;
            b.read(&attn.v_bias, ck("self_attn.v_proj.bias"))?;
            b.read(&attn.o_proj, ck("self_attn.o_proj.weight"))?;
            b.read(&attn.o_bias, ck("self_attn.o_proj.bias"))?;
            b.read(&attn.sinks, ck("self_attn.sinks"))?;
            b.read(&mlp.router, ck("mlp.router.weight"))?;
            b.read(&mlp.router_bias, ck("mlp.router.bias"))?;

            match layout {
                Layout::Transformers => {
                    let rows = i64::from(mlp.inter);
                    b.extend({
                        banked_interleaved(
                            src,
                            &mlp.gate_up,
                            ck("mlp.experts.gate_up_proj_blocks"),
                            ck("mlp.experts.gate_up_proj_scales"),
                            rows,
                            layout,
                        )
                    }?);
                    b.read_expr(
                        &mlp.gate_up_bias,
                        deinterleaved(Expr::src(ck("mlp.experts.gate_up_proj_bias")), rows),
                    )?;

                    b.extend({
                        banked(
                            src,
                            &mlp.down,
                            ck("mlp.experts.down_proj_blocks"),
                            ck("mlp.experts.down_proj_scales"),
                            layout,
                        )
                    }?);
                    b.read(&mlp.down_bias, ck("mlp.experts.down_proj_bias"))?;
                }
                Layout::Mlx => {
                    b.extend({
                        banked_split(
                            src,
                            &mlp.gate_up,
                            &[ck("mlp.experts.gate_proj"), ck("mlp.experts.up_proj")],
                            layout,
                        )
                    }?);
                    b.read_concat(
                        &mlp.gate_up_bias,
                        [
                            ck("mlp.experts.gate_proj.bias"),
                            ck("mlp.experts.up_proj.bias"),
                        ],
                    )?;

                    b.extend({
                        banked_split(src, &mlp.down, &[ck("mlp.experts.down_proj")], layout)
                    }?);
                    b.read(&mlp.down_bias, ck("mlp.experts.down_proj.bias"))?;
                }
            }
        }
        if let Some(dflash) = &self.dflash {
            dflash.bind_aux(&mut b, src, &|from| Expr::src(from).bias(-1.0))?;
        }
        Ok(b.build())
    }
}

fn banked(
    src: &ztensor::Source,
    w: &Weight,
    blocks: String,
    scales: String,
    layout: Layout,
) -> Result<Vec<TensorContract>, Error> {
    bank_planes(src, w, blocks, scales, layout, |expr| expr)
}

fn banked_interleaved(
    src: &ztensor::Source,
    w: &Weight,
    blocks: String,
    scales: String,
    rows: i64,
    layout: Layout,
) -> Result<Vec<TensorContract>, Error> {
    bank_planes(src, w, blocks, scales, layout, |expr| {
        deinterleaved(expr, rows)
    })
}

fn banked_split(
    src: &ztensor::Source,
    w: &Weight,
    stems: &[String],
    layout: Layout,
) -> Result<Vec<TensorContract>, Error> {
    let legs = i64::try_from(stems.len()).expect("a stem count inside i64");
    let axis = usize::from(BANK_ROWS);
    let leg = |whole: &[i64]| -> Vec<i64> {
        let mut cut = whole.to_vec();
        cut[axis] /= legs;
        cut
    };
    let codes = bank_codes(w);
    let scaled = bank_scales(w);
    let leg_codes = TensorType::new(leg(&codes.shape), codes.encoding.clone());
    let leg_scales = TensorType::new(leg(&scaled.shape), scaled.encoding.clone());

    let mut code_legs = Vec::with_capacity(stems.len());
    let mut scale_legs = Vec::with_capacity(stems.len());
    for stem in stems {
        let weight = format!("{stem}.weight");
        let scales = format!("{stem}.scales");
        stored_as(src, w, &weight, layout.codes())?;
        stored_as(src, w, &scales, DType::U8)?;
        code_legs.push(Expr::src(weight).transmute(leg_codes.clone()));
        scale_legs.push(Expr::src(scales).transmute(leg_scales.clone()));
    }
    let joined = |mut legs: Vec<Expr>| {
        if legs.len() == 1 {
            legs.remove(0)
        } else {
            Expr::concat(BANK_ROWS, legs)
        }
    };
    Ok(vec![
        TensorContract::inferred(w.name.clone(), joined(code_legs), codes.encoding),
        TensorContract::new(
            poem_dsl::scales_name(&w.name),
            joined(scale_legs),
            scaled.shape,
            scaled.encoding,
        )
        .scaling(scaling(w)),
    ])
}

fn stored_as(src: &ztensor::Source, w: &Weight, plane: &str, want: DType) -> Result<(), Error> {
    let stored = stored_encoding(src, plane)?;
    if stored == Encoding::Raw(want) {
        return Ok(());
    }
    Err(Error::Illegible {
        name: w.name.clone(),
        detail: format!(
            "`{plane}` is stored {stored:?}, and this spelling of the bank \
             carries it as raw {want:?}"
        ),
    })
}

fn bank_planes(
    src: &ztensor::Source,
    w: &Weight,
    blocks: String,
    scales: String,
    layout: Layout,
    lay: impl Fn(Expr) -> Expr,
) -> Result<Vec<TensorContract>, Error> {
    stored_as(src, w, &blocks, layout.codes())?;
    stored_as(src, w, &scales, DType::U8)?;
    let codes = bank_codes(w);
    let scaled = bank_scales(w);
    Ok(vec![
        TensorContract::inferred(
            w.name.clone(),
            lay(Expr::src(blocks).transmute(codes.clone())),
            codes.encoding,
        ),
        TensorContract::new(
            poem_dsl::scales_name(&w.name),
            lay(Expr::src(scales).transmute(scaled.clone())),
            scaled.shape,
            scaled.encoding,
        )
        .scaling(scaling(w)),
    ])
}

fn deinterleaved(src: Expr, rows: i64) -> Expr {
    Expr::concat(
        BANK_ROWS,
        vec![
            src.clone().stride(BANK_ROWS, 0, rows, 2),
            src.stride(BANK_ROWS, 1, rows, 2),
        ],
    )
}

fn bank_codes(w: &Weight) -> TensorType {
    TensorType::new(extents(w), grouped(w))
}

fn bank_scales(w: &Weight) -> TensorType {
    let shape = extents(w);
    let pairing = scaling(w);
    TensorType::new(
        divided(&shape, pairing.channel_axis, pairing.group_size, &w.name),
        encoding(Dtype::E8m0),
    )
}
