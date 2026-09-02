use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};
use checkpoint::types::{DType, Encoding};
use model_dsl::{Dtype, Weight};

use super::model::Model;
use model_dsl::Platform;
use checkpoint_dsl::{Builder, Error, divided, encoding, extents, grouped, scaling, stored_encoding};

const BANK_ROWS: u8 = 1;

/// How this family spells its expert banks; every name outside the MoE is
/// shared between the two spellings. `mlx_lm` un-fuses the row-interleaved
/// `gate_up_proj` into two contiguous tensors (gate first) joined by
/// `concat`, and its codes container is `Raw(U32)` instead of `Raw(U8)` —
/// same bytes, reinterpreted ([`Expr::transmute`]). `_bias` splits the same
/// way; `down_proj` only renames.
#[derive(Clone, Copy)]
enum Layout {
    /// `mlp.experts.gate_up_proj_blocks` / `_scales` / `_bias`, fused and
    /// row-interleaved, codes in `u8` — transformers.
    Transformers,
    /// `mlp.experts.{gate,up,down}_proj.{weight,scales,bias}`, split, codes in
    /// `u32` — `mlx_lm`.
    Mlx,
}

impl Layout {
    /// The container the expert code planes arrive in. Scales are `u8` in
    /// both spellings: one E8M0 exponent byte per block.
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
        // `load` fails on the first plane not found under this contract's
        // own names, which is how a foreign file is detected.
        let mut refusals: Vec<String> = Vec::new();
        for (what, layout) in [
            ("transformers", Layout::Transformers),
            ("mlx_lm", Layout::Mlx),
        ] {
            match self.import_from(src, platform, layout) {
                Ok(contract) => return Ok(contract),
                Err(why) => refusals.push(format!("as {what}, {why}")),
            }
        }
        Err(Error::Illegible {
            name: "gpt_oss".to_string(),
            detail: format!(
                "no reading of this file lands every plane this family \
                 declares — {}",
                refusals.join("; "),
            ),
        })
    }

    fn import_from(
        &self,
        src: &ztensor::Source, platform: Platform,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        // How many tensors one `read` covers depends on the weight's dtype,
        // not the layout: bf16 reads one `.weight`, u4g64 reads the
        // `.weight`/`.scales`/`.biases` triplet.
        let mut b = Builder::new(src, self.tp, platform);
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
            // `Moe::router` is declared `U8g64` wherever the stack is
            // `U4g64` (`gpt_oss.py`'s `quant_predicate`, read at
            // `Model::new`).
            b.read(&mlp.router, ck("mlp.router.weight"))?;
            b.read(&mlp.router_bias, ck("mlp.router.bias"))?;

            // The one place the two spellings part — see [`Layout`].
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
                    // Halves of the declared `[experts, 2*inter]` bias,
                    // joined gate-first, same axis as `deinterleaved` above.
                    b.read_concat(
                        &mlp.gate_up_bias,
                        [ck("mlp.experts.gate_proj.bias"), ck("mlp.experts.up_proj.bias")],
                    )?;

                    b.extend({
                        banked_split(src, &mlp.down, &[ck("mlp.experts.down_proj")], layout)
                    }?);
                    b.read(&mlp.down_bias, ck("mlp.experts.down_proj.bias"))?;
                }
            }
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

/// A declared bank from `mlx_lm`'s split stems (`gate_up` = 2 tensors,
/// `down` = 1). Each stem gives a `<stem>.weight` of codes and a
/// `<stem>.scales` of E8M0 exponents; legs join on [`BANK_ROWS`], the axis
/// `Weight::bank` cut. Leg extents are the bank's with that axis divided by
/// the stem count; a mismatched byte count is caught by `infer_transmute`.
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
            model_dsl::scales_name(&w.name),
            joined(scale_legs),
            scaled.shape,
            scaled.encoding,
        )
        .scaling(scaling(w)),
    ])
}

/// The plane is on disk in the container this spelling promises, or the
/// refusal names both.
fn stored_as(
    src: &ztensor::Source,
    w: &Weight,
    plane: &str,
    want: DType,
) -> Result<(), Error> {
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
            model_dsl::scales_name(&w.name),
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
