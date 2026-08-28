use model_dsl::{Dtype, Shard, Weight};
use model_loader::contract::{Expr, ModelContract, Scales, TensorContract};
use model_loader::types::{Axis, Encoding, QuantGranularity, QuantSpec, ScaleForm};

const GROUP: u32 = 32;

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
        Dtype::Mxfp4 => (grouped(w), Some(scaling(w))),
        Dtype::Bf16
        | Dtype::F16
        | Dtype::F32
        | Dtype::I32
        | Dtype::U32
        | Dtype::U8
        | Dtype::I8
        | Dtype::Fp8E4m3
        | Dtype::E8m0 => (crate::encoding(w.dtype), None),
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
                tensors.push(interned(&name, &shape, bands, pairing));
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

fn interned(
    of: &str,
    shape: &[i64],
    bands: Option<(u32, Vec<i64>)>,
    pairing: Scales,
) -> TensorContract {
    seams_clear_the_blocked_axis(of, bands.as_ref(), pairing.channel_axis);
    let name = model_dsl::scales_name(of);
    let declared = divided(shape, pairing.channel_axis, of);
    let want = crate::encoding(Dtype::E8m0);
    let expr = banded(&name, bands.as_ref());
    TensorContract::new(name, expr, declared, want).scaling(pairing)
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
    model_loader::checkpoint::encoding_of(&tensor, &part).map_err(|why| illegible(&why))
}

fn ladder(name: &str, expr: Expr, stored: &Encoding, want: &Encoding) -> Result<Expr, ModelError> {
    match (stored, want) {
        (s, w) if s == w => Ok(expr),
        (Encoding::Raw(_), Encoding::Raw(_)) => Ok(expr.cast(want.clone())),
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

pub(crate) fn divided(shape: &[i64], axis: u32, name: &str) -> Vec<i64> {
    let mut dims = shape.to_vec();
    let at = axis as usize;
    let extent = *dims
        .get(at)
        .unwrap_or_else(|| panic!("`{name}` is {shape:?} and its blocks count along axis {at}"));
    let group = i64::from(GROUP);
    assert!(
        extent % group == 0,
        "`{name}` contracts over {extent}, which is not a whole number of \
         {GROUP}-code blocks",
    );
    dims[at] = extent / group;
    dims
}

fn seams_clear_the_blocked_axis(name: &str, bands: Option<&(u32, Vec<i64>)>, channel: u32) {
    let Some((axis, extents)) = bands else {
        return;
    };
    assert!(
        extents.len() < 2 || *axis != channel,
        "`{name}` is cut at {} seams along axis {axis}, which is the axis its \
         scales count in {GROUP}-code blocks",
        extents.len(),
    );
}

pub(crate) fn scaling(w: &Weight) -> Scales {
    Scales {
        of: w.name.clone(),
        granularity: QuantGranularity::PerGroup,
        group_size: GROUP,
        channel_axis: u32::from(channel_axis(w)),
        form: ScaleForm::RawE8M0,
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
