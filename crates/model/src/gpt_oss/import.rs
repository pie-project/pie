use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};
use checkpoint::types::{DType, Encoding};
use model_dsl::{Dtype, Weight};

use super::model::Model;
use crate::contract::{self, ModelError};
use crate::encoding;

const BANK_ROWS: u8 = 1;

/// **HOW A CHECKPOINT OF THIS FAMILY SPELLS ITS EXPERT BANKS**, which is the
/// only thing two otherwise identical files disagree about.
///
/// The trunk does NOT move here, unlike qwen's and gemma's: gpt-oss is a plain
/// causal LM with no multimodal wrapper, so `mlx_lm`'s `sanitize` leaves
/// `model.layers.{l}.*`, `model.embed_tokens.*` and `lm_head.*` exactly where
/// transformers put them, and every name outside the MoE below is shared.
///
/// What `GptOssModel.sanitize` does do is take the fused, row-interleaved
/// `gate_up_proj` apart:
///
/// ```python
/// if "gate_up_proj" in k and "bias" not in k:
///     if "_blocks" in k:
///         v = v.view(mx.uint32).flatten(-2)
///         k = k.replace("_blocks", ".weight")
///     if "_scales" in k:
///         k = k.replace("_scales", ".scales")
///     new_weights[k.replace("gate_up_proj", "gate_proj")] = mx.contiguous(v[..., ::2, :])
///     new_weights[k.replace("gate_up_proj", "up_proj")]   = mx.contiguous(v[..., 1::2, :])
/// ```
///
/// Three changes in one pass, and all three have to be read back out:
///
/// 1. **The seam moves from a stride to a name.** Transformers interleaves
///    gate and up rows inside one tensor, which is what [`deinterleaved`]
///    un-picks with a pair of strided reads. `mlx_lm` writes two tensors whose
///    rows are already contiguous — so the same declared bank is a `concat`
///    of two whole sources instead, on the same axis, gate leg first, which is
///    the order the strided pair produced.
///
/// 2. **The codes change container.** `v.view(mx.uint32).flatten(-2)` turns
///    the OCP `_blocks` layout — a `[.., 16]` tail of bytes per 32-code block
///    — into `u32` words, so what transformers ships as `Raw(U8)` this ships
///    as `Raw(U32)`. **The BYTES are identical**: OCP packs E2M1 low nibble
///    first, `mx.uint32` is little-endian, so byte `b` of a word still holds
///    codes `2b` and `2b+1` in that order. It is a reinterpret and
///    [`Expr::transmute`] is exactly the node for one — which is why
///    [`bank_planes`] takes the container it should expect rather than
///    asserting one.
///
/// 3. **`_bias` splits the same way**, into `gate_proj.bias` and
///    `up_proj.bias`, and joins back with a plain `concat` on the same axis.
///
/// The `down_proj` bank takes changes 2 and 3 and not 1: it was never fused,
/// so it is one source under a renamed suffix.
///
/// **THE DISCRIMINATOR IS THE FIRST LAYER'S GATE-UP CODES**, because that is
/// the one name the two spellings do not share. Everything else this import
/// reads is either identically named or answered by the weight's own dtype.
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
    /// The name whose presence decides — see the type's doc.
    fn witness(self) -> &'static str {
        match self {
            Self::Transformers => "model.layers.0.mlp.experts.gate_up_proj_blocks",
            Self::Mlx => "model.layers.0.mlp.experts.gate_proj.weight",
        }
    }

    /// The container the expert CODE planes arrive in. The scales are `u8` in
    /// both spellings — one OCP E8M0 exponent byte per block, which neither
    /// conversion repacks.
    fn codes(self) -> DType {
        match self {
            Self::Transformers => DType::U8,
            Self::Mlx => DType::U32,
        }
    }
}

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        assert!(
            self.tp == 1,
            "an import states the whole checkpoint; build the model at tp = 1"
        );
        for layout in [Layout::Transformers, Layout::Mlx] {
            if src.get(layout.witness()).is_some() {
                return self.import_from(src, layout);
            }
        }
        Err(ModelError::Illegible {
            name: "gpt_oss".to_string(),
            detail: format!(
                "it holds neither `{}` nor `{}`, so its expert banks are \
                 written in no spelling this family reads",
                Layout::Transformers.witness(),
                Layout::Mlx.witness(),
            ),
        })
    }

    fn import_from(
        &self,
        src: &ztensor::Source,
        layout: Layout,
    ) -> Result<ModelContract, ModelError> {
        // **`planes` AND NOT `copy`, AT EVERY MATMUL BANK.** These names are
        // the same in both spellings; what differs is how many TENSORS each
        // one is, and that is a fact about the weight's declared dtype rather
        // than about the layout. A bf16 SKU reads one `.weight` here exactly
        // as it always did; an `mlxu4` SKU reads the `.weight`/`.scales`/
        // `.biases` triplet MLX ships beside it. `planes` is the call that
        // says the logical name once and lets the dtype answer.
        let mut tensors = contract::planes(src, &self.embed, "model.embed_tokens.weight")?;
        tensors.push(contract::copy(src, &self.final_norm, "model.norm.weight")?);
        tensors.extend(contract::planes(src, &self.head, "lm_head.weight")?);
        for (l, layer) in self.layers.iter().enumerate() {
            let ck = |what: &str| format!("model.layers.{l}.{what}");
            let attn = &layer.attn;
            let mlp = &layer.mlp;

            tensors.push(contract::copy(
                src,
                &layer.attn_norm,
                ck("input_layernorm.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &layer.mlp_norm,
                ck("post_attention_layernorm.weight"),
            )?);
            tensors.extend(contract::planes(
                src,
                &attn.q_proj,
                ck("self_attn.q_proj.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.q_bias,
                ck("self_attn.q_proj.bias"),
            )?);
            tensors.extend(contract::planes(
                src,
                &attn.k_proj,
                ck("self_attn.k_proj.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.k_bias,
                ck("self_attn.k_proj.bias"),
            )?);
            tensors.extend(contract::planes(
                src,
                &attn.v_proj,
                ck("self_attn.v_proj.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.v_bias,
                ck("self_attn.v_proj.bias"),
            )?);
            tensors.extend(contract::planes(
                src,
                &attn.o_proj,
                ck("self_attn.o_proj.weight"),
            )?);
            tensors.push(contract::copy(
                src,
                &attn.o_bias,
                ck("self_attn.o_proj.bias"),
            )?);
            tensors.push(contract::copy(src, &attn.sinks, ck("self_attn.sinks"))?);
            // **THE ROUTER GATE, AT ITS OWN WIDTH.** `Moe::router` is declared
            // `MlxU8` wherever the stack is `MlxU4` — `gpt_oss.py`'s
            // `quant_predicate`, read at `Model::new` — and `planes` asks the
            // weight rather than the file, so the same call reads a bf16
            // tensor here from a transformers checkpoint and an eight-bit
            // affine triplet from an MLX one.
            tensors.extend(contract::planes(src, &mlp.router, ck("mlp.router.weight"))?);
            tensors.push(contract::copy(
                src,
                &mlp.router_bias,
                ck("mlp.router.bias"),
            )?);

            // The one place the two spellings part — see [`Layout`].
            match layout {
                Layout::Transformers => {
                    let rows = i64::from(mlp.inter);
                    tensors.extend(banked_interleaved(
                        src,
                        &mlp.gate_up,
                        ck("mlp.experts.gate_up_proj_blocks"),
                        ck("mlp.experts.gate_up_proj_scales"),
                        rows,
                        layout,
                    )?);
                    tensors.push(contract::declare(
                        src,
                        &mlp.gate_up_bias,
                        deinterleaved(Expr::src(ck("mlp.experts.gate_up_proj_bias")), rows),
                    )?);

                    tensors.extend(banked(
                        src,
                        &mlp.down,
                        ck("mlp.experts.down_proj_blocks"),
                        ck("mlp.experts.down_proj_scales"),
                        layout,
                    )?);
                    tensors.push(contract::copy(
                        src,
                        &mlp.down_bias,
                        ck("mlp.experts.down_proj_bias"),
                    )?);
                }
                Layout::Mlx => {
                    tensors.extend(banked_split(
                        src,
                        &mlp.gate_up,
                        &[
                            ck("mlp.experts.gate_proj"),
                            ck("mlp.experts.up_proj"),
                        ],
                        layout,
                    )?);
                    // Two contiguous halves of the declared `[experts,
                    // 2*inter]` bias, joined on the bank's own seam — the same
                    // axis and the same gate-first order `deinterleaved`
                    // produces above.
                    tensors.push(contract::fused(
                        src,
                        &mlp.gate_up_bias,
                        [
                            ck("mlp.experts.gate_proj.bias"),
                            ck("mlp.experts.up_proj.bias"),
                        ],
                    )?);

                    tensors.extend(banked_split(
                        src,
                        &mlp.down,
                        &[ck("mlp.experts.down_proj")],
                        layout,
                    )?);
                    tensors.push(contract::copy(
                        src,
                        &mlp.down_bias,
                        ck("mlp.experts.down_proj.bias"),
                    )?);
                }
            }
        }
        Ok(ModelContract {
            alignment: contract::ALIGNMENT,
            tensors,

            groups: Vec::new(),
        })
    }
}

fn banked(
    src: &ztensor::Source,
    w: &Weight,
    blocks: String,
    scales: String,
    layout: Layout,
) -> Result<Vec<TensorContract>, ModelError> {
    bank_planes(src, w, blocks, scales, layout, |expr| expr)
}

fn banked_interleaved(
    src: &ztensor::Source,
    w: &Weight,
    blocks: String,
    scales: String,
    rows: i64,
    layout: Layout,
) -> Result<Vec<TensorContract>, ModelError> {
    bank_planes(src, w, blocks, scales, layout, |expr| {
        deinterleaved(expr, rows)
    })
}

/// **ONE DECLARED BANK OUT OF THE `stem`s THE CHECKPOINT SPLIT IT INTO** —
/// `mlx_lm`'s spelling, where a `gate_up` is two whole tensors rather than one
/// tensor read at a stride, and a `down` is the degenerate one-stem case.
///
/// Each stem contributes a `<stem>.weight` of codes and a `<stem>.scales` of
/// E8M0 exponents; the legs join on [`BANK_ROWS`], which is the axis
/// `Weight::bank` cut and therefore the axis the declared shape's `2 * inter`
/// spans. **THE SCALES JOIN ON THE SAME AXIS AS THE CODES**, because a block's
/// exponent belongs to the row it scales, and this axis is not the blocked one
/// — the contracted axis is the last, and each leg's group count is untouched
/// by the seam.
///
/// The leg's own extents are the declared bank's with that axis divided by the
/// number of stems, which is what makes the arithmetic checkable rather than
/// assumed: a transmute whose byte count did not match the source would be
/// refused by `infer_transmute` at the plane it was wrong about.
fn banked_split(
    src: &ztensor::Source,
    w: &Weight,
    stems: &[String],
    layout: Layout,
) -> Result<Vec<TensorContract>, ModelError> {
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
        .scaling(contract::scaling(w)),
    ])
}

/// The plane is on disk in the container this spelling promises, or the
/// refusal names both.
fn stored_as(
    src: &ztensor::Source,
    w: &Weight,
    plane: &str,
    want: DType,
) -> Result<(), ModelError> {
    let stored = contract::stored_encoding(src, plane)?;
    if stored == Encoding::Raw(want) {
        return Ok(());
    }
    Err(ModelError::Illegible {
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
) -> Result<Vec<TensorContract>, ModelError> {
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
        .scaling(contract::scaling(w)),
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
    TensorType::new(contract::extents(w), contract::grouped(w))
}

fn bank_scales(w: &Weight) -> TensorType {
    let shape = contract::extents(w);
    let pairing = contract::scaling(w);
    TensorType::new(
        contract::divided(&shape, pairing.channel_axis, pairing.group_size, &w.name),
        encoding(Dtype::E8m0),
    )
}
