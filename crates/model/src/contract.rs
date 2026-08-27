use model_dsl::{Dtype, Shard, Weight};
use model_loader::contract::{Expr, ModelContract, Scales, TensorContract};
use model_loader::types::{Axis, Encoding, QuantGranularity, QuantSpec, ScaleForm};

const GROUP: u32 = 32;

const ALIGNMENT: u32 = 256;

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
                "`{name}`: the checkpoint states this plane in terms no reader \
                 here can name ({detail})"
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
        | Dtype::E8m0
        | Dtype::Fp4 => (crate::encoding(w.dtype), None),
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
        let stored = stored_encoding(src, &claim.name)?;
        let read = banded(&claim.name, claim.bands.as_ref());
        let expr = ladder(&claim.name, read, &stored, &claim.encoding)?;
        tensors.push(match &claim.encoding {
            Encoding::Raw(_) => TensorContract::new(
                claim.name.clone(),
                expr,
                claim.shape.clone(),
                claim.encoding.clone(),
            ),
            Encoding::Quant(_) => {
                TensorContract::inferred(claim.name.clone(), expr, claim.encoding.clone())
            }
        });
        let pairing = match stored {
            Encoding::Quant(_) => claim.scales,
            Encoding::Raw(_) => None,
        };
        if let Some(pairing) = pairing {
            tensors.push(interned(&claim.name, &claim.shape, claim.bands, pairing));
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
    tp: u32,
    expr: Expr,
) -> Result<TensorContract, ModelError> {
    let want = crate::encoding(w.dtype);
    let stored = agreed(src, &w.name, &expr)?;
    let expr = ladder(&w.name, expr, &stored, &want)?;
    Ok(match &stored {
        Encoding::Raw(_) => TensorContract::new(w.name.clone(), expr, whole(w, tp), want),
        Encoding::Quant(_) => TensorContract::inferred(w.name.clone(), expr, want),
    })
}

pub fn copy(
    src: &ztensor::Source,
    w: &Weight,
    tp: u32,
    from: impl Into<String>,
) -> Result<TensorContract, ModelError> {
    declare(src, w, tp, Expr::src(from))
}

pub fn fused(
    src: &ztensor::Source,
    w: &Weight,
    tp: u32,
    parts: impl IntoIterator<Item = String>,
) -> Result<TensorContract, ModelError> {
    let legs = parts.into_iter().map(Expr::src).collect();
    declare(src, w, tp, Expr::concat(pack_axis(w), legs))
}

fn interned(
    of: &str,
    shape: &[i64],
    bands: Option<(u32, Vec<i64>)>,
    pairing: Scales,
) -> TensorContract {
    seams_clear_the_blocked_axis(of, bands.as_ref(), pairing.channel_axis);
    let name = format!("{of}.scales");
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
        panic!("`{name}` is built from no checkpoint tensor at all")
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

fn stored_encoding(src: &ztensor::Source, name: &str) -> Result<Encoding, ModelError> {
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

fn whole(w: &Weight, tp: u32) -> Vec<i64> {
    let mut dims: Vec<i64> = w
        .shape
        .iter()
        .map(|extent| i64::try_from(*extent).expect("an extent no i64 holds"))
        .collect();
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

fn divided(shape: &[i64], axis: u32, name: &str) -> Vec<i64> {
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

fn scaling(w: &Weight) -> Scales {
    Scales {
        of: w.name.clone(),
        granularity: QuantGranularity::PerGroup,
        group_size: GROUP,
        channel_axis: u32::from(channel_axis(w)),
        form: ScaleForm::RawE8M0,
    }
}

fn grouped(w: &Weight) -> Encoding {
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

fn channel_axis(w: &Weight) -> u8 {
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
