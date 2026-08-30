use checkpoint::contract::{Expr, ModelContract, Scales, TensorContract, TensorType};
use checkpoint::types::{Axis, DType, Encoding, QuantGranularity, QuantSpec, ScaleForm};
use model_dsl::{Dtype, Shard, Weight};

/// How many codes one group of a packed bank shares a scale with.
///
/// A NUMBER PER SCHEME, NOT A NUMBER FOR THE TREE. It was `const GROUP: u32 =
/// 32` — mxfp4's, back when mxfp4 was the only packed weight here — and every
/// site that blocked an axis or sized a scales plane read it. MLX's affine U4
/// groups sixty-four, so a single constant would have declared a scales plane
/// twice as long as the checkpoint ships and failed at the byte count, which
/// is late and far from the sentence that was wrong.
fn group_of(dtype: Dtype) -> u32 {
    match dtype {
        Dtype::Mxfp4 => 32,
        // Both affine widths group sixty-four CODES, not sixty-four bytes:
        // the group is a property of the scheme and the width is a property
        // of the code. See `dtype::Dtype::MlxU8`.
        Dtype::MlxU4 | Dtype::MlxU8 => 64,
        other => panic!("`{other:?}` blocks no axis; only a packed bank has groups"),
    }
}

/// How many affine codes the checkpoint packs into one `u32` word — MLX's own
/// packing, least-significant code first, which is a fact the loader reads off
/// `QuantScheme::MlxAffineU4` and this file only has to count with.
///
/// **THE COUNT IS THE WIDTH'S, NOT THE SCHEME'S**, and it had been a constant
/// while four bits was the only width MLX wrote here. A u32 holds eight
/// four-bit codes and FOUR eight-bit ones, so a router gate stored `[32, 720]`
/// unpacks to `[32, 2880]` through this and to `[32, 5760]` through the
/// constant — a bank twice as wide as the checkpoint holds, caught late and
/// far from the sentence that was wrong.
fn word_codes(dtype: Dtype) -> i64 {
    let bits = i64::try_from(dtype.bits()).expect("a code width inside i64");
    assert!(
        bits > 0 && 32 % bits == 0,
        "`{dtype:?}` is {bits} bits wide, which does not divide a 32-bit word"
    );
    32 / bits
}

pub(crate) const ALIGNMENT: u32 = 256;

pub struct Claim {
    pub name: String,
    pub shape: Vec<i64>,
    pub bands: Option<(u32, Vec<i64>)>,
    pub encoding: Encoding,
    pub scales: Option<Scales>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelError {
    Missing(String),

    Illegible {
        name: String,
        detail: String,
    },

    Incompatible {
        name: String,
        stored: Encoding,
        want: Encoding,
    },
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(name) => write!(
                f,
                "this model reads a plane called `{name}` and the checkpoint \
                 holds no tensor under that name"
            ),
            Self::Illegible { name, detail } => write!(
                f,
                "`{name}`: this checkpoint is stated in terms no reader here \
                 can name ({detail})"
            ),
            Self::Incompatible { name, stored, want } => write!(
                f,
                "`{name}` is stored {stored:?} and this model wants {want:?}; \
                 one quantization is not decoded into another on the way in"
            ),
        }
    }
}

impl std::error::Error for ModelError {}

#[must_use]
pub fn claim(w: &Weight, tp: u32) -> Claim {
    let (encoding, scales) = match w.dtype {
        Dtype::Mxfp4 | Dtype::MlxU4 | Dtype::MlxU8 => (grouped(w), Some(scaling(w))),
        Dtype::Bf16
        | Dtype::F16
        | Dtype::F32
        | Dtype::I32
        | Dtype::U32
        | Dtype::U8
        | Dtype::I8
        | Dtype::Fp8E4m3
        | Dtype::E8m0
        | Dtype::Fp8E5m2
        | Dtype::I64
        | Dtype::I16
        | Dtype::U64
        | Dtype::U16
        | Dtype::Bool => (crate::encoding(w.dtype), None),
        Dtype::Fp4 => panic!(
            "`Dtype::Fp4` names a kv-page quantization scheme, not a stored \
             weight plane; no load contract declares one"
        ),
    };
    Claim {
        name: w.name.clone(),
        shape: whole(w, tp),
        bands: banding(w, tp),
        encoding,
        scales,
    }
}

pub fn elaborate(src: &ztensor::Source, claims: Vec<Claim>) -> Result<ModelContract, ModelError> {
    let mut tensors = Vec::new();
    for claim in claims {
        let Claim {
            name,
            shape,
            bands,
            encoding,
            scales,
        } = claim;
        let stored = stored_encoding(src, &name)?;
        let read = banded(&name, bands.as_ref());
        let expr = ladder(&name, read, &stored, &encoding)?;
        tensors.push(match &encoding {
            Encoding::Raw(_) => {
                TensorContract::new(name.clone(), expr, shape.clone(), encoding.clone())
            }
            Encoding::Quant(_) => TensorContract::inferred(name.clone(), expr, encoding.clone()),
        });
        match (&stored, scales) {
            // The checkpoint SHIPPED both planes: the second one is bytes on
            // disk, so the contract says where they are.
            (Encoding::Quant(_), Some(pairing)) => {
                tensors.extend(interned(&name, &shape, bands, pairing));
            }
            // The checkpoint ships the weight unquantized and the model wants
            // it quantized, so `ladder` above put an honest
            // `Cast { to: Quant(..) }` in the expression and the LOADER
            // encodes on the way in. It also publishes the scales plane the
            // codes cannot be read without, under `<w>.scales` — the one name
            // this tree binds an mxfp4 second plane by, the one
            // `model_dsl::scales_name` writes and `Weight::planes` interns
            // into `Trace::params`. That accord is settled in the loader
            // (`plan::build::ScaleLayout::for_encode`'s MXFP4 arm carries the
            // ruling and `executor::walk`'s
            // `an_expert_bank_encodes_to_the_same_bytes_as_the_rows_it_stacks`
            // proves the bytes and the name), which is why there is nothing
            // to declare here: an entry of our own would be a SECOND producer
            // for a plane that already has exactly one.
            //
            // THIS ARM WAS A REFUSAL — `ModelError::Incompatible`, "one
            // quantization is not decoded into another on the way in", which
            // is a sentence about the Quant-to-Quant case wrongly applied to
            // this one. It made runtime quantization unreachable through
            // `Model::load` at all, against M18's own ruling that a declared
            // encoding the checkpoint does not hold is exactly what makes the
            // loader cast or encode. The Quant-to-Quant refusal is unaffected:
            // it lives in `ladder`, where it belongs.
            (Encoding::Raw(_), Some(_)) => {}
            (Encoding::Quant(_), None) | (Encoding::Raw(_), None) => {}
        }
    }
    Ok(ModelContract {
        alignment: ALIGNMENT,
        tensors,

        groups: Vec::new(),
    })
}

pub fn declare(
    src: &ztensor::Source,
    w: &Weight,
    expr: Expr,
) -> Result<TensorContract, ModelError> {
    let want = crate::encoding(w.dtype);
    let stored = agreed(src, &w.name, &expr)?;
    let expr = ladder(&w.name, expr, &stored, &want)?;
    Ok(match &stored {
        Encoding::Raw(_) => TensorContract::new(w.name.clone(), expr, extents(w), want),
        Encoding::Quant(_) => TensorContract::inferred(w.name.clone(), expr, want),
    })
}

pub fn copy(
    src: &ztensor::Source,
    w: &Weight,
    from: impl Into<String>,
) -> Result<TensorContract, ModelError> {
    declare(src, w, Expr::src(from))
}

pub fn fused(
    src: &ztensor::Source,
    w: &Weight,
    parts: impl IntoIterator<Item = String>,
) -> Result<TensorContract, ModelError> {
    let legs = parts.into_iter().map(Expr::src).collect();
    declare(src, w, Expr::concat(pack_axis(w), legs))
}

/// One declared weight, said as the one tensor it is or the three the
/// checkpoint splits it into.
///
/// **A FAMILY IMPORT NAMES A WEIGHT ONCE, WHATEVER IT IS STORED AS.** [`copy`]
/// answers with one contract because a bf16 projection IS one tensor; an MLX
/// affine-U4 projection is three — `q_proj.weight` holding the codes eight to
/// a `u32` word, `q_proj.scales` and `q_proj.biases` holding one bf16 each per
/// sixty-four codes — and a family that had to know which case it was in would
/// be two imports pretending to be one. So the call site says the logical name
/// and this says how many tensors that is.
///
/// **THE TRIPLET IS REQUIRED, NOT PREFERRED.** A weight declared `MlxU4`
/// against a checkpoint that ships bf16 would otherwise fall through to
/// [`declare`], whose `Raw -> Quant` ladder makes the LOADER encode — which is
/// a real and wanted path through `Model::load`, and a disaster through
/// `model::identify`: the u4 SKU would claim every bf16 checkpoint of its own
/// family, and being listed ahead of the bf16 row (which it must be, see
/// `qwen_3::IMPORTS`) it would claim it first. An import states what the FILE
/// holds, so a missing `.scales` is a miss and the next row gets its turn.
pub fn planes(
    src: &ztensor::Source,
    w: &Weight,
    from: impl Into<String>,
) -> Result<Vec<TensorContract>, ModelError> {
    let from = from.into();
    match w.dtype {
        Dtype::MlxU4 | Dtype::MlxU8 => affine_planes(src, w, vec![from]),
        _ => Ok(vec![copy(src, w, from)?]),
    }
}

/// [`planes`] for a weight this text states as one bank and the checkpoint
/// ships as several — a gate and an up, a q and a k and a v.
///
/// Each part carries its own triplet and the legs are joined on the bank's own
/// cut axis, which is the axis the parts were split along: the codes join
/// there, and so do the scales and the biases, because a group belongs to the
/// row it scales and rows are what a pack seam separates.
pub fn planes_fused(
    src: &ztensor::Source,
    w: &Weight,
    parts: impl IntoIterator<Item = String>,
) -> Result<Vec<TensorContract>, ModelError> {
    let parts: Vec<String> = parts.into_iter().collect();
    match w.dtype {
        Dtype::MlxU4 | Dtype::MlxU8 => affine_planes(src, w, parts),
        _ => Ok(vec![fused(src, w, parts)?]),
    }
}

/// The codes, the scales and the biases of one MLX affine-U4 bank, read out of
/// the `<stem>.weight` / `<stem>.scales` / `<stem>.biases` triplet each part
/// names.
fn affine_planes(
    src: &ztensor::Source,
    w: &Weight,
    parts: Vec<String>,
) -> Result<Vec<TensorContract>, ModelError> {
    let axis = if parts.len() > 1 { pack_axis(w) } else { 0 };
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for part in &parts {
        let stem = part
            .strip_suffix(".weight")
            .ok_or_else(|| ModelError::Illegible {
                name: w.name.clone(),
                detail: format!(
                    "`{part}` holds MLX affine codes, whose scales and biases \
                     are named beside a `.weight`, and it does not end in one"
                ),
            })?;
        let unpacked = unpacked_extents(src, w, part)?;
        codes.push(Expr::src(part.clone()).transmute(TensorType::new(unpacked, grouped(w))));
        scales.push(model_dsl::scales_name(stem));
        biases.push(model_dsl::biases_name(stem));
    }
    let pairing = scaling(w);
    let counted = divided(
        &extents(w),
        pairing.channel_axis,
        pairing.group_size,
        &w.name,
    );
    Ok(vec![
        // `inferred`, as `gpt_oss`'s bank planes are: the transmute above
        // already stated the logical shape of every leg, and a Quant
        // declaration that also predicted the joined shape would state the
        // same rectangle twice in two arithmetics.
        TensorContract::inferred(w.name.clone(), joined(axis, codes), grouped(w)),
        factors(
            src,
            model_dsl::scales_name(&w.name),
            &scales,
            counted.clone(),
            axis,
        )?
        .scaling(pairing),
        // `offsetting` and not a second `scaling`: the two companions complete
        // ONE attachment, and it is the zero-point entry that says which
        // weight it centres. A biases plane that named nothing would land as
        // a bound tensor no kernel reaches, and the codes beside it would
        // dequantize around zero — right spread, wrong centre, no NaN to
        // notice it by.
        factors(src, model_dsl::biases_name(&w.name), &biases, counted, axis)?
            .offsetting(w.name.clone()),
    ])
}

/// One companion plane of an affine bank, joined across the parts and brought
/// to bf16.
///
/// The cast is not a formality: mlx-community ships some conversions with F16
/// scales and biases and others with BF16, and this tree reads a bank's
/// factors as bf16 in one spelling. `ladder` is what makes that a stated
/// conversion rather than a reinterpretation of the bytes.
fn factors(
    src: &ztensor::Source,
    name: String,
    legs: &[String],
    shape: Vec<i64>,
    axis: u8,
) -> Result<TensorContract, ModelError> {
    let expr = joined(axis, legs.iter().cloned().map(Expr::src).collect());
    let stored = agreed(src, &name, &expr)?;
    let want = crate::encoding(Dtype::Bf16);
    let expr = ladder(&name, expr, &stored, &want)?;
    Ok(TensorContract::new(name, expr, shape, want))
}

fn joined(axis: u8, mut legs: Vec<Expr>) -> Expr {
    if legs.len() == 1 {
        legs.pop().expect("one leg")
    } else {
        Expr::concat(axis, legs)
    }
}

/// The logical shape of a stored affine-U4 plane: what the checkpoint holds,
/// with its contracted axis multiplied back out of the words it was packed
/// into.
///
/// Read off the FILE and not off the declaration, because the declaration is
/// the whole joined bank and this is one of its legs — a fused `gate_up` is
/// one `Weight` and two stored tensors, and neither leg's width is derivable
/// from the sum without assuming they are equal, which for a qkv pack they are
/// not.
fn unpacked_extents(src: &ztensor::Source, w: &Weight, name: &str) -> Result<Vec<i64>, ModelError> {
    let Some(tensor) = src.get(name) else {
        return Err(ModelError::Missing(name.to_string()));
    };
    let stored = stored_encoding(src, name)?;
    if stored != Encoding::Raw(DType::U32) {
        return Err(ModelError::Illegible {
            name: w.name.clone(),
            detail: format!(
                "`{name}` is stored {stored:?}, and MLX affine codes are read \
                 as raw u32 words of {} codes each",
                word_codes(w.dtype),
            ),
        });
    }
    let mut dims: Vec<i64> = tensor
        .shape()
        .iter()
        .map(|extent| i64::try_from(*extent).expect("an extent no i64 holds"))
        .collect();
    let Some(words) = dims.last_mut() else {
        return Err(ModelError::Illegible {
            name: w.name.clone(),
            detail: format!("`{name}` is a scalar and a bank has a contracted axis"),
        });
    };
    *words *= word_codes(w.dtype);
    Ok(dims)
}

/// The companion planes the checkpoint SHIPPED beside `of`, said as
/// declarations of their own.
///
/// **A SCHEME'S SECOND PLANE IS NOT ALWAYS ITS LAST.** This answered one
/// contract, because mxfp4 has one companion: a byte of exponent per block.
/// MLX's affine U4 has two — `code * scale + bias` reads a scale AND an
/// offset — and they are declared as siblings rather than one wide tensor
/// because that is how MLX ships them and how
/// `plan::build::QuantAttachment` binds them: the zero point of a shipped
/// triplet is a tensor the contract states in its own right, and only an
/// encode the loader performs has an id to record on the attachment instead.
///
/// The form is what says how many there are. `Scales::form` already carries
/// the scheme's whole answer to "what do these numbers mean", so asking it
/// "and how many tensors is that" costs nothing and cannot drift from the
/// pairing the same call site built.
fn interned(
    of: &str,
    shape: &[i64],
    bands: Option<(u32, Vec<i64>)>,
    pairing: Scales,
) -> Vec<TensorContract> {
    seams_clear_the_blocked_axis(of, bands.as_ref(), pairing.channel_axis, pairing.group_size);
    let declared = divided(shape, pairing.channel_axis, pairing.group_size, of);
    let plane = |name: String, dtype: Dtype, shape: Vec<i64>| {
        let expr = banded(&name, bands.as_ref());
        TensorContract::new(name, expr, shape, crate::encoding(dtype))
    };
    match pairing.form {
        ScaleForm::RawE8M0 => {
            vec![plane(model_dsl::scales_name(of), Dtype::E8m0, declared).scaling(pairing)]
        }
        ScaleForm::Bf16AffineFactors => vec![
            plane(model_dsl::scales_name(of), Dtype::Bf16, declared.clone()).scaling(pairing),
            plane(model_dsl::biases_name(of), Dtype::Bf16, declared).offsetting(of),
        ],
        other => panic!(
            "`{of}` pairs its codes with {other:?} scales, and no family here \
             declares a bank in those terms"
        ),
    }
}

fn agreed(src: &ztensor::Source, name: &str, expr: &Expr) -> Result<Encoding, ModelError> {
    let mut read: Vec<(&str, Encoding)> = Vec::new();
    for source in expr.sources() {
        let stored = stored_encoding(src, source)?;
        read.push((source, stored));
    }
    let Some((whose, first)) = read.first() else {
        return Err(ModelError::Illegible {
            name: name.to_string(),
            detail: "it is built from no checkpoint tensor at all".to_string(),
        });
    };
    for (source, seen) in &read {
        if seen != first {
            return Err(ModelError::Illegible {
                name: name.to_string(),
                detail: format!(
                    "`{whose}` is stored {first:?} and `{source}` is stored \
                     {seen:?}; one weight is not read out of two representations"
                ),
            });
        }
    }
    Ok(first.clone())
}

pub(crate) fn stored_encoding(src: &ztensor::Source, name: &str) -> Result<Encoding, ModelError> {
    let Some(tensor) = src.get(name) else {
        return Err(ModelError::Missing(name.to_string()));
    };
    let illegible = |why: &dyn std::fmt::Display| ModelError::Illegible {
        name: name.to_string(),
        detail: why.to_string(),
    };
    let part = tensor.part("data").map_err(|why| illegible(&why))?;
    checkpoint::file::encoding_of(&tensor, &part).map_err(|why| illegible(&why))
}

fn ladder(name: &str, expr: Expr, stored: &Encoding, want: &Encoding) -> Result<Expr, ModelError> {
    match (stored, want) {
        (s, w) if s == w => Ok(expr),
        (Encoding::Raw(_), Encoding::Raw(_)) => Ok(expr.cast(want.clone())),
        // **PACKED CODES ARE NOT VALUES TO QUANTIZE.** A `Raw -> Quant` rung
        // is the loader encoding on the way in, which is a real path — kimi
        // declares mxfp4 expert banks over a bf16 checkpoint and means exactly
        // that. It is nonsense for a checkpoint that already ships the codes:
        // MLX writes them as `u32` words, and encoding those integers as if
        // they were weights lands a bank whose every element is a code read as
        // a number, with no name in the plan being wrong. Such a file is bound
        // by naming its three planes — `contract::planes` — so this rung is a
        // refusal and not a conversion.
        (Encoding::Raw(DType::U32), Encoding::Quant(_)) => Err(ModelError::Illegible {
            name: name.to_string(),
            detail: "it is stored as raw u32 words and this model wants it \
                     quantized; a checkpoint that already ships packed codes is \
                     read by naming its planes, not by encoding its words"
                .to_string(),
        }),
        (Encoding::Raw(_), Encoding::Quant(_)) => Ok(expr.cast(want.clone())),
        (Encoding::Quant(_), Encoding::Raw(_)) => Ok(expr.cast(want.clone())),
        (Encoding::Quant(_), Encoding::Quant(_)) => Err(ModelError::Incompatible {
            name: name.to_string(),
            stored: stored.clone(),
            want: want.clone(),
        }),
    }
}

fn banding(w: &Weight, tp: u32) -> Option<(u32, Vec<i64>)> {
    match &w.shard {
        Shard::Replicated => None,
        Shard::Cut { axis, segments } => Some((
            *axis,
            segments
                .iter()
                .map(|segment| leg_extent(*segment, tp, &w.name))
                .collect(),
        )),
    }
}

fn banded(name: &str, bands: Option<&(u32, Vec<i64>)>) -> Expr {
    let Some((axis, extents)) = bands else {
        return Expr::src(name);
    };
    let at = as_axis(*axis, name);
    match extents.as_slice() {
        [] => panic!("`{name}` is cut at no seam at all"),
        [_lone] => Expr::src(name).shard(at),
        many => {
            let mut start = 0;
            let legs = many
                .iter()
                .map(|extent| {
                    let leg = Expr::src(name).slice(at, start, *extent).shard(at);
                    start += *extent;
                    leg
                })
                .collect();
            Expr::concat(at, legs)
        }
    }
}

pub(crate) fn extents(w: &Weight) -> Vec<i64> {
    w.shape
        .iter()
        .map(|extent| i64::try_from(*extent).expect("an extent no i64 holds"))
        .collect()
}

pub(crate) fn whole(w: &Weight, tp: u32) -> Vec<i64> {
    let mut dims = extents(w);
    match &w.shard {
        Shard::Replicated => dims,
        Shard::Cut { axis, segments } => {
            let at = *axis as usize;
            let dim = dims.get_mut(at).unwrap_or_else(|| {
                panic!("`{}` is {:?} and its cut names axis {at}", w.name, w.shape)
            });
            let seams: u64 = segments.iter().sum();
            assert_eq!(
                u64::try_from(*dim).expect("an extent no u64 holds"),
                seams,
                "`{}`: its segments sum to {seams} and its axis {at} is {dim}",
                w.name,
            );
            *dim = dim
                .checked_mul(i64::from(tp))
                .unwrap_or_else(|| panic!("`{}` is {tp} times wider than an i64", w.name));
            dims
        }
    }
}

pub(crate) fn divided(shape: &[i64], axis: u32, group: u32, name: &str) -> Vec<i64> {
    let mut dims = shape.to_vec();
    let at = axis as usize;
    let extent = *dims
        .get(at)
        .unwrap_or_else(|| panic!("`{name}` is {shape:?} and its blocks count along axis {at}"));
    let width = i64::from(group);
    assert!(
        extent % width == 0,
        "`{name}` contracts over {extent}, which is not a whole number of \
         {group}-code blocks",
    );
    dims[at] = extent / width;
    dims
}

fn seams_clear_the_blocked_axis(
    name: &str,
    bands: Option<&(u32, Vec<i64>)>,
    channel: u32,
    group: u32,
) {
    let Some((axis, extents)) = bands else {
        return;
    };
    assert!(
        extents.len() < 2 || *axis != channel,
        "`{name}` is cut at {} seams along axis {axis}, which is the axis its \
         scales count in {group}-code blocks",
        extents.len(),
    );
}

/// How `w`'s codes are paired with the numbers that read them.
///
/// The scheme decides both halves and they are not independent: an mxfp4 block
/// is thirty-two codes under one exponent byte, an MLX affine group is
/// sixty-four codes under one bf16 scale and one bf16 offset. Reading the
/// group width off one scheme and the form off another would produce a pairing
/// no kernel implements and no checkpoint ships.
pub(crate) fn scaling(w: &Weight) -> Scales {
    let form = match w.dtype {
        Dtype::Mxfp4 => ScaleForm::RawE8M0,
        Dtype::MlxU4 | Dtype::MlxU8 => ScaleForm::Bf16AffineFactors,
        other => panic!(
            "`{}` is {other:?}, which pairs with nothing; only a packed bank \
             has scales",
            w.name
        ),
    };
    Scales {
        of: w.name.clone(),
        granularity: QuantGranularity::PerGroup,
        group_size: group_of(w.dtype),
        channel_axis: u32::from(channel_axis(w)),
        form,
    }
}

pub(crate) fn grouped(w: &Weight) -> Encoding {
    match crate::encoding(w.dtype) {
        Encoding::Quant(spec) => Encoding::Quant(QuantSpec {
            channel_axis: Some(Axis(channel_axis(w))),
            ..spec
        }),
        Encoding::Raw(dtype) => panic!(
            "`{}` is {dtype:?}, which groups nothing; only a quantized bank \
             has a blocked axis",
            w.name
        ),
    }
}

pub(crate) fn channel_axis(w: &Weight) -> u8 {
    let last = w
        .shape
        .len()
        .checked_sub(1)
        .unwrap_or_else(|| panic!("`{}` is a bank and has no contracted axis", w.name));
    u8::try_from(last).expect("an axis inside a shape")
}

fn pack_axis(w: &Weight) -> u8 {
    match &w.shard {
        Shard::Replicated => panic!("`{}` is replicated and has no cut axis", w.name),
        Shard::Cut { axis, .. } => as_axis(*axis, &w.name),
    }
}

fn leg_extent(segment: u64, tp: u32, name: &str) -> i64 {
    let whole = segment
        .checked_mul(u64::from(tp))
        .unwrap_or_else(|| panic!("`{name}`: a segment of {segment} is not {tp} times anything"));
    i64::try_from(whole).expect("an extent no i64 holds")
}

fn as_axis(axis: u32, name: &str) -> u8 {
    u8::try_from(axis)
        .unwrap_or_else(|_| panic!("`{name}` is cut on axis {axis}, which is no axis"))
}
