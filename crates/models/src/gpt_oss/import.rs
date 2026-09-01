use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};
use checkpoint::types::{DType, Encoding};
use model_dsl::{Dtype, Weight};

use super::model::Model;
use checkpoint_dsl::{Builder, Error, divided, encoding, extents, grouped, scaling, stored_encoding};

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
/// **THE DISCRIMINATOR WAS THE FIRST LAYER'S GATE-UP CODES AND IS NOW THE
/// READING ITSELF** (§M-4a-3). That one name is the only one the two
/// spellings do not share, which made it the cheapest possible witness — and
/// a witness is a proxy for "does this arm build", which promotion can
/// falsify: the plane it names is one `pie model import` moves. So both arms
/// are built and the first that succeeds is the answer, and the name that
/// used to decide is now just one of the names an arm looks for. The argument
/// in full, and the file it was measured on, is at `qwen_3::Model::import`.
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
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, Error> {
        // **THE NATIVE DOOR, ASKED BEFORE THE WITNESS SNIFF** (§M-4a). A file
        // holding every plane this contract declares, under this contract's
        // names, is an artifact `pie model import` wrote out of this very
        // text, and [`Model::load`] is its reader: `read_own` throughout, no
        // transform at all. `load` failing is what says the file is foreign,
        // and it fails on the first plane it cannot find. The argument in full
        // is at `qwen_3::Model::import`.
        if let Ok(native) = self.load(src) {
            return Ok(native);
        }
        // **AND THE ARM IS CHOSEN BY BUILDING IT, NOT BY SNIFFING A NAME.**
        // The witness this used to look for — the embedding, spelled the way
        // each layout spells it — is one of the planes a promotion MOVES, so
        // an artifact this build wrote could satisfy neither door. The
        // argument in full, and the file it was measured on, is at
        // `qwen_3::Model::import`.
        let mut refusals: Vec<String> = Vec::new();
        for (what, layout) in [
            ("transformers", Layout::Transformers),
            ("mlx_lm", Layout::Mlx),
        ] {
            match self.import_from(src, layout) {
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
        src: &ztensor::Source,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        // **ONE `read` PER BANK, HOWEVER MANY TENSORS THAT IS.** These
        // names are the same in both spellings; what differs is how many
        // TENSORS each one is, and that is a fact about the weight's declared
        // dtype rather than about the layout. A bf16 SKU reads one `.weight`
        // here exactly as it always did; an `mlxu4` SKU reads the
        // `.weight`/`.scales`/`.biases` triplet MLX ships beside it —
        // `read` says the logical name once and lets the dtype answer.
        let mut b = Builder::new(src, self.tp);
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
            // **THE ROUTER GATE, AT ITS OWN WIDTH.** `Moe::router` is declared
            // `U8g64` wherever the stack is `U4g64` — `gpt_oss.py`'s
            // `quant_predicate`, read at `Model::new` — and `read` asks the
            // weight rather than the file, so the same call reads a bf16
            // tensor here from a transformers checkpoint and an eight-bit
            // affine triplet from an MLX one.
            b.read(&mlp.router, ck("mlp.router.weight"))?;
            b.read(&mlp.router_bias, ck("mlp.router.bias"))?;

            // The one place the two spellings part — see [`Layout`].
            match layout {
                Layout::Transformers => {
                    let rows = i64::from(mlp.inter);
                    b.extend(banked_interleaved(
                        src,
                        &mlp.gate_up,
                        ck("mlp.experts.gate_up_proj_blocks"),
                        ck("mlp.experts.gate_up_proj_scales"),
                        rows,
                        layout,
                    )?);
                    b.read_expr(
                        &mlp.gate_up_bias,
                        deinterleaved(Expr::src(ck("mlp.experts.gate_up_proj_bias")), rows),
                    )?;

                    b.extend(banked(
                        src,
                        &mlp.down,
                        ck("mlp.experts.down_proj_blocks"),
                        ck("mlp.experts.down_proj_scales"),
                        layout,
                    )?);
                    b.read(&mlp.down_bias, ck("mlp.experts.down_proj_bias"))?;
                }
                Layout::Mlx => {
                    b.extend(banked_split(
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
                    b.read_concat(
                        &mlp.gate_up_bias,
                        [ck("mlp.experts.gate_proj.bias"), ck("mlp.experts.up_proj.bias")],
                    )?;

                    b.extend(banked_split(
                        src,
                        &mlp.down,
                        &[ck("mlp.experts.down_proj")],
                        layout,
                    )?);
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
