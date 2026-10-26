pub mod format;
mod int4;

use crate::{Dtype, Platform, Shard, Weight};
use checkpoint::contract::{Expr, ModelContract, Scales, TensorContract, TensorType};
use checkpoint::types::{
    Axis, DType, Encoding, QuantGranularity, QuantScheme, QuantSpec, RepackLayout, ScaleForm,
    TILED_BAND,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
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

impl std::fmt::Display for Error {
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

impl std::error::Error for Error {}

thread_local! {
    static RECORDING: std::cell::RefCell<Option<Vec<String>>> =
        const { std::cell::RefCell::new(None) };
}

/// What `f` asks of the builders it reads through, each read stated rather
/// than performed: two imports that ask the same of every builder build the
/// same contract of any checkpoint.
pub fn recording<R>(f: impl FnOnce() -> R) -> (R, Vec<String>) {
    RECORDING.with(|r| *r.borrow_mut() = Some(Vec::new()));
    let out = f();
    let log = RECORDING
        .with(|r| r.borrow_mut().take())
        .unwrap_or_default();
    (out, log)
}

fn recorded(what: impl FnOnce() -> String) -> bool {
    RECORDING.with(|r| match r.borrow_mut().as_mut() {
        Some(log) => {
            log.push(what());
            true
        }
        None => false,
    })
}

pub struct Builder<'a> {
    src: &'a ztensor::Source,
    tp: u32,
    platform: Platform,
    tensors: Vec<TensorContract>,
    /// The weights the compiler lays out tiled (an affine projection a
    /// package states row-major).
    tiled: std::collections::BTreeSet<String>,
}

impl<'a> Builder<'a> {
    #[must_use]
    pub fn new(src: &'a ztensor::Source, tp: u32, platform: Platform) -> Builder<'a> {
        Builder {
            src,
            tp,
            platform,
            tensors: Vec::new(),
            tiled: std::collections::BTreeSet::new(),
        }
    }

    /// The builder storing the weights `tiled` names in the tiled order.
    #[must_use]
    pub fn tiling(mut self, tiled: &[String]) -> Builder<'a> {
        self.tiled = tiled.iter().cloned().collect();
        self
    }

    /// `w` as this platform stores it, tiled where the compiler lays it so.
    fn laid(&self, w: &Weight) -> Weight {
        let mut w = w.placed(self.platform);
        if w.dtype == Dtype::U4g64 && self.tiled.contains(&w.name) {
            w.dtype = self.platform.placement(Dtype::U4g64tiled);
        }
        w
    }

    pub fn read(&mut self, w: &Weight, from: impl Into<String>) -> Result<(), Error> {
        let w = &self.laid(w);
        let from = from.into();
        if recorded(|| format!("read {w:?} {from:?}")) {
            return Ok(());
        }
        self.whole_checkpoint(w)?;
        let read = planes(self.src, w, from)?;
        self.tensors.extend(read);
        Ok(())
    }

    pub fn read_concat(
        &mut self,
        w: &Weight,
        parts: impl IntoIterator<Item = String>,
    ) -> Result<(), Error> {
        let w = &self.laid(w);
        let parts: Vec<String> = parts.into_iter().collect();
        if recorded(|| format!("read_concat {w:?} {parts:?}")) {
            return Ok(());
        }
        self.whole_checkpoint(w)?;
        let read = planes_fused(self.src, w, parts)?;
        self.tensors.extend(read);
        Ok(())
    }

    pub fn read_stack(
        &mut self,
        w: &Weight,
        rows: impl IntoIterator<Item = Vec<String>>,
    ) -> Result<(), Error> {
        let w = &self.laid(w);
        let rows: Vec<Vec<String>> = rows.into_iter().collect();
        if recorded(|| format!("read_stack {w:?} {rows:?}")) {
            return Ok(());
        }
        self.whole_checkpoint(w)?;
        let read = affine_stacked(self.src, w, rows)?;
        self.tensors.extend(read);
        Ok(())
    }

    pub fn read_over(
        &mut self,
        w: &Weight,
        from: impl Into<String>,
        over: impl FnOnce(Expr) -> Expr,
    ) -> Result<(), Error> {
        let w = &self.laid(w);
        let from = from.into();
        let over = over(Expr::src(from.clone()));
        if recorded(|| format!("read_over {w:?} {from:?} {over:?}")) {
            return Ok(());
        }
        self.whole_checkpoint(w)?;
        let read = if placed_road_covers(w) {
            placed_from_encode(w, |canon| read_over(self.src, canon, &from, over))?
        } else {
            read_over(self.src, w, &from, over)?
        };
        self.tensors.extend(read);
        Ok(())
    }

    pub fn read_expr(&mut self, w: &Weight, expr: Expr) -> Result<(), Error> {
        let w = &self.laid(w);
        if recorded(|| format!("read_expr {w:?} {expr:?}")) {
            return Ok(());
        }
        self.whole_checkpoint(w)?;
        let read = computed(self.src, w, expr)?;
        self.tensors.extend(read);
        Ok(())
    }

    /// `w` lands columns `start .. start + len` of the checkpoint's tensor
    /// `from`, its last axis. A raw tensor is sliced; MLX affine codes are
    /// read by naming their planes, each sliced over the same columns: the
    /// words, the scales and the biases, a range that keeps whole groups.
    pub fn read_columns(
        &mut self,
        w: &Weight,
        from: impl Into<String>,
        start: i64,
        len: i64,
    ) -> Result<(), Error> {
        let w = &self.laid(w);
        let from = from.into();
        if recorded(|| format!("read_columns {w:?} {from:?} {start} {len}")) {
            return Ok(());
        }
        self.whole_checkpoint(w)?;
        if !stored_mlx_codes(self.src, &from)? || matches!(encoding(w.dtype), Encoding::Raw(_)) {
            let axis = column_axis(self.src, &from)?;
            let read = computed(self.src, w, Expr::src(from).slice(axis, start, len))?;
            self.tensors.extend(read);
            return Ok(());
        }
        let read = affine_columns(self.src, w, &from, start, len)?;
        self.tensors.extend(read);
        Ok(())
    }

    pub fn read_own(&mut self, w: &Weight) -> Result<(), Error> {
        let w = &self.laid(w);
        if recorded(|| format!("read_own {w:?}")) {
            return Ok(());
        }
        let read = resolve(self.src, claim(w, self.tp))?;
        self.tensors.extend(read);
        Ok(())
    }

    pub fn push(&mut self, tensor: TensorContract) -> Result<(), Error> {
        if recorded(|| format!("push {tensor:?}")) {
            return Ok(());
        }
        self.untiled(&tensor)?;
        self.tensors.push(tensor);
        Ok(())
    }

    pub fn extend(
        &mut self,
        tensors: impl IntoIterator<Item = TensorContract>,
    ) -> Result<(), Error> {
        let tensors: Vec<TensorContract> = tensors.into_iter().collect();
        if recorded(|| format!("extend {tensors:?}")) {
            return Ok(());
        }
        for t in &tensors {
            self.untiled(t)?;
        }
        self.tensors.extend(tensors);
        Ok(())
    }

    /// A stated tensor the compiler lays tiled must state the tiled repack.
    fn untiled(&self, t: &TensorContract) -> Result<(), Error> {
        if self.tiled.contains(&t.name)
            && t.visibility == checkpoint::types::Visibility::Public
            && !matches!(
                t.expr,
                Expr::Repack {
                    layout: RepackLayout::TiledAffineU4Weight,
                    ..
                }
            )
        {
            return Err(Error::Illegible {
                name: t.name.clone(),
                detail: "the compiler lays it tiled and this read states it row-major; \
                         read it with a verb that takes the weight"
                    .to_string(),
            });
        }
        Ok(())
    }

    #[must_use]
    pub fn build(self) -> ModelContract {
        ModelContract {
            alignment: ALIGNMENT,
            tensors: self.tensors,
            groups: Vec::new(),
        }
    }

    fn whole_checkpoint(&self, w: &Weight) -> Result<(), Error> {
        if self.tp != 1 {
            return Err(Error::Illegible {
                name: w.name.clone(),
                detail: format!(
                    "an import states the WHOLE checkpoint and this contract is \
                     built for {} ranks; nothing has banded the file it is \
                     reading, so there is no rank {}'s share of `{}` in it to \
                     land",
                    self.tp,
                    self.tp - 1,
                    w.name,
                ),
            });
        }
        Ok(())
    }
}

pub fn own_contract(
    src: &ztensor::Source,
    params: &[crate::Param],
    tp: u32,
    platform: Platform,
) -> Result<ModelContract, Error> {
    let companions: std::collections::BTreeSet<String> = params
        .iter()
        .filter(|param| matches!(claim_kind(param.dtype), Kind::Packed))
        .flat_map(|param| {
            [
                crate::scales_name(&param.name),
                crate::biases_name(&param.name),
                crate::global_scales_name(&param.name),
                crate::input_scales_name(&param.name),
            ]
        })
        .collect();
    let mut b = Builder::new(src, tp, platform);
    for param in params {
        if param.source != crate::ParamSource::Checkpoint || companions.contains(&param.name) {
            continue;
        }
        let before = b.tensors.len();
        b.read_own(&Weight::of_plane(param))?;
        if let Some(cut) = param.experts {
            for tensor in &mut b.tensors[before..] {
                let expr = std::mem::replace(&mut tensor.expr, Expr::src(""));
                tensor.expr = slotted(src, expr, cut).map_err(|why| Error::Illegible {
                    name: param.name.clone(),
                    detail: format!("{why}"),
                })?;
            }
        }
    }
    Ok(b.build())
}

/// `contract` without the weights `read` (the params a trace reads, by
/// name) holds none of: a public tensor neither a param nor a plane of one,
/// and the internal ones only those read.
pub fn prune(contract: &mut ModelContract, read: &[String]) {
    let read: std::collections::BTreeSet<&str> = read.iter().map(String::as_str).collect();
    let held = |name: &str| {
        read.contains(name)
            || name
                .rsplit_once('.')
                .is_some_and(|(stem, _)| read.contains(stem))
    };
    let mut keep: Vec<bool> = contract
        .tensors
        .iter()
        .map(|t| t.visibility != checkpoint::types::Visibility::Public || held(&t.name))
        .collect();
    // An internal tensor stays only where a kept one reads it.
    let at: std::collections::BTreeMap<&str, usize> = contract
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.as_str(), i))
        .collect();
    let mut needed = vec![false; contract.tensors.len()];
    for i in (0..contract.tensors.len()).rev() {
        let public = contract.tensors[i].visibility == checkpoint::types::Visibility::Public;
        if !(keep[i] && (public || needed[i])) {
            continue;
        }
        let mut outs = Vec::new();
        outs_of(&contract.tensors[i].expr, &mut outs);
        for name in outs {
            if let Some(&j) = at.get(name.as_str()) {
                needed[j] = true;
            }
        }
    }
    for (i, t) in contract.tensors.iter().enumerate() {
        if t.visibility != checkpoint::types::Visibility::Public {
            keep[i] = needed[i];
        }
    }
    let mut i = 0;
    contract.tensors.retain(|_| {
        i += 1;
        keep[i - 1]
    });
}

fn outs_of(expr: &Expr, into: &mut Vec<String>) {
    if let Expr::Out(name) = expr {
        into.push(name.clone());
        return;
    }
    let _ = expr.clone().map_children(|child| {
        outs_of(&child, into);
        Ok(child)
    });
}

/// One static input scale for legs joined into `of`, each stating its own
/// (`names`, one f32 each): the first leg's where they agree, else the
/// largest, under which no leg's activations overflow (as vLLM fuses them).
fn input_scale(src: &ztensor::Source, names: &[String], of: &str) -> Result<Expr, Error> {
    let value = |name: &str| -> Result<f32, Error> {
        let bytes = src
            .tensor(name)
            .ok()
            .and_then(|t| t.bytes().ok().map(|b| b.into_owned()))
            .ok_or_else(|| Error::Missing(name.to_string()))?;
        let word: [u8; 4] = bytes.as_slice().try_into().map_err(|_| Error::Illegible {
            name: of.to_string(),
            detail: format!("`{name}` is not one f32"),
        })?;
        Ok(f32::from_le_bytes(word))
    };
    // A checkpoint of headers alone (a plan's dry run) holds no values: its
    // legs' first scale stands, as an import that cannot compare them reads.
    let Ok(values) = names
        .iter()
        .map(|n| value(n))
        .collect::<Result<Vec<_>, _>>()
    else {
        return Ok(Expr::src(names[0].clone()));
    };
    if values.iter().all(|v| v.to_bits() == values[0].to_bits()) {
        return Ok(Expr::src(names[0].clone()));
    }
    let largest = values.iter().copied().fold(f32::MIN, f32::max);
    constant(of, &[largest], Vec::new(), checkpoint::types::DType::F32)
}

/// `contract` with the legs of `join` (each read as its own weight, its
/// planes beside it) joined along their rows under the joined name, plane by
/// plane, the legs kept as the internal reads it is made of; nothing where the
/// codec's rows read them joined already.
pub fn join_read(
    src: &ztensor::Source,
    contract: &mut ModelContract,
    join: &crate::Join,
) -> Result<(), String> {
    let at = |name: &str| contract.tensors.iter().position(|t| t.name == name);
    if at(&join.name).is_some() {
        return Ok(());
    }
    let mut planes: Vec<Vec<usize>> = Vec::new();
    for leg in &join.legs {
        let main = at(leg).ok_or_else(|| {
            format!(
                "`{leg}` is joined into `{}` and no read lands it",
                join.name
            )
        })?;
        let mut mine = vec![main];
        mine.extend(contract.tensors.iter().enumerate().filter_map(|(i, t)| {
            let suffix = t.name.strip_prefix(leg.as_str())?.strip_prefix('.')?;
            (!suffix.contains('.') || suffix.starts_with("leg")).then_some(i)
        }));
        planes.push(mine);
    }
    let suffix = |i: usize, leg: &str| contract.tensors[i].name[leg.len()..].to_string();
    let first: Vec<String> = planes[0]
        .iter()
        .map(|&i| suffix(i, &join.legs[0]))
        .collect();
    for (leg, mine) in join.legs.iter().zip(&planes) {
        let theirs: Vec<String> = mine.iter().map(|&i| suffix(i, leg)).collect();
        if theirs != first {
            return Err(format!(
                "`{leg}` lands planes {theirs:?} and `{}` lands {first:?}, so no joined `{}` \
                 holds both",
                join.legs[0], join.name
            ));
        }
    }
    let rename = |name: &str| -> String {
        join.legs
            .iter()
            .find_map(|leg| {
                let rest = name.strip_prefix(leg.as_str())?;
                (rest.is_empty() || rest.starts_with('.')).then(|| format!("{}{rest}", join.name))
            })
            .unwrap_or_else(|| name.to_string())
    };
    let mut made = Vec::new();
    let mut inlined: Vec<String> = Vec::new();
    for (p, plane) in first.iter().enumerate() {
        let legs: Vec<&TensorContract> = planes
            .iter()
            .map(|mine| &contract.tensors[mine[p]])
            .collect();
        let lead = legs[0];
        if legs.iter().any(|t| t.encoding != lead.encoding) {
            return Err(format!(
                "the legs of `{}` land `{plane}` at two encodings",
                join.name
            ));
        }
        // A plane of no axis (a static input scale) is one for the product:
        // the first leg's, the others the same value.
        if lead.shape.as_deref() == Some(&[]) {
            let names: Vec<String> = legs
                .iter()
                .map(|t| match &t.expr {
                    Expr::Src(name) => Ok(name.clone()),
                    other => Err(format!(
                        "`{}` reads {other:?}, not one stored scale",
                        t.name
                    )),
                })
                .collect::<Result<_, _>>()?;
            let expr = input_scale(src, &names, &join.name).map_err(|why| format!("{why}"))?;
            made.push(TensorContract {
                name: format!("{}{plane}", join.name),
                expr,
                ..lead.clone()
            });
            continue;
        }
        // A plane of no axis to join along (a routed bank's input scale, one
        // an expert) is the first leg's, as a scalar one is: the legs read
        // one activation.
        if lead
            .shape
            .as_ref()
            .is_some_and(|shape| shape.len() <= join.axis as usize)
        {
            made.push(TensorContract {
                name: format!("{}{plane}", join.name),
                ..lead.clone()
            });
            continue;
        }
        let shape = legs
            .iter()
            .map(|t| t.shape.clone())
            .collect::<Option<Vec<_>>>()
            .and_then(|shapes| {
                let at = join.axis as usize;
                let mut whole = shapes.first()?.clone();
                *whole.get_mut(at)? = shapes
                    .iter()
                    .map(|s| s.get(at).copied())
                    .sum::<Option<i64>>()?;
                Some(whole)
            });
        // Legs read straight from the checkpoint are joined as their reads,
        // the extents landing in the joined plane as a stated pack's do: a
        // serving load lands no leg apart to move it after. A leg the load
        // computes is joined as what it lands, which only an import moves.
        let read = legs.iter().all(|t| t.expr.is_affine());
        let axis = u8::try_from(join.axis).unwrap_or(0);
        // Quantized legs are joined before quantizing, so the joined weight's scales are the loader's.
        fn quantized(t: &TensorContract) -> Option<(&Expr, &checkpoint::types::Encoding)> {
            match &t.expr {
                Expr::Cast { src, to, chunks: 0 } if t.scales.is_none() => Some((&**src, to)),
                _ => None,
            }
        }
        let cast = legs
            .iter()
            .map(|t| quantized(t))
            .collect::<Option<Vec<(&Expr, &checkpoint::types::Encoding)>>>()
            .filter(|casts| plane.is_empty() && casts.iter().all(|(_, to)| *to == casts[0].1));
        let expr = if let Some(casts) = cast {
            inlined.extend(legs.iter().map(|t| t.name.clone()));
            let to = casts[0].1.clone();
            let parts = casts.into_iter().map(|(src, _)| src.clone()).collect();
            Expr::Cast {
                src: Box::new(Expr::concat(axis, parts)),
                to,
                chunks: 0,
            }
        } else {
            let parts = legs
                .iter()
                .map(|t| {
                    if read {
                        t.expr.clone()
                    } else {
                        Expr::out(t.name.clone())
                    }
                })
                .collect();
            if read {
                inlined.extend(legs.iter().map(|t| t.name.clone()));
            }
            Expr::concat(axis, parts)
        };
        made.push(TensorContract {
            name: format!("{}{plane}", join.name),
            expr,
            shape,
            encoding: lead.encoding.clone(),
            scales: lead.scales.clone().map(|mut s| {
                s.of = rename(&s.of);
                s
            }),
            zero_points: lead.zero_points.as_deref().map(rename),
            visibility: lead.visibility,
        });
    }
    let last = planes.iter().flatten().copied().max().unwrap_or(0);
    for i in planes.iter().flatten() {
        contract.tensors[*i].visibility = checkpoint::types::Visibility::Internal;
    }
    let tail = contract.tensors.split_off(last + 1);
    contract.tensors.extend(made);
    contract.tensors.extend(tail);
    // A leg joined as its reads lands nothing of its own, unless something
    // else reads what it lands.
    let read: std::collections::BTreeSet<String> = contract
        .tensors
        .iter()
        .flat_map(|t| t.expr.outputs())
        .map(str::to_string)
        .collect();
    contract
        .tensors
        .retain(|t| !inlined.contains(&t.name) || read.contains(&t.name));
    Ok(())
}

/// `expr` reading an expert-parallel rank's slots of each bank it reads
/// whole: a source whose leading axis is the bank's experts (its codes and
/// the planes beside them, one row an expert), not one scale for them all.
fn slotted(
    src: &ztensor::Source,
    expr: Expr,
    cut: poem_ir::ExpertSlots,
) -> Result<Expr, checkpoint::error::Error> {
    let experts = i64::from(cut.experts);
    match expr {
        Expr::Src(name) => {
            let leading = stored_shape(src, &name).and_then(|shape| shape.first().copied());
            Ok(if leading == Some(experts) {
                Expr::src(name).experts(experts, i64::from(cut.slots))
            } else {
                Expr::Src(name)
            })
        }
        expr => expr.map_children(|child| slotted(src, child, cut)),
    }
}

enum Kind {
    Packed,
    Plain,
}

fn claim_kind(dtype: Dtype) -> Kind {
    match dtype {
        Dtype::Mxfp4
        | Dtype::Nvfp4
        | Dtype::E4m3tensor
        | Dtype::U4g64
        | Dtype::U8g64
        | Dtype::U4g32
        | Dtype::U4g64tiled
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128 => Kind::Packed,
        _ => Kind::Plain,
    }
}

fn group_of(dtype: Dtype) -> u32 {
    match dtype {
        Dtype::Mxfp4 => 32,
        Dtype::Nvfp4 => 16,
        Dtype::E4m3tensor => 1,
        Dtype::U4g64 | Dtype::U8g64 | Dtype::U2g64 | Dtype::U4g64tiled => 64,
        Dtype::U4g32 | Dtype::U2g32 => 32,
        Dtype::U2g128 | Dtype::E4m3tile128 | Dtype::E4m3g128 => 128,
        other => panic!("`{other:?}` blocks no axis; only a packed bank has groups"),
    }
}

fn word_codes(dtype: Dtype) -> i64 {
    let bits = i64::try_from(dtype.bits()).expect("a code width inside i64");
    assert!(
        bits > 0 && 32 % bits == 0,
        "`{dtype:?}` is {bits} bits wide, which does not divide a 32-bit word"
    );
    32 / bits
}

const ALIGNMENT: u32 = 256;

/// The axis a weight is cut along, and each segment's whole extent with how
/// many ranks in a row read the same share of it.
type Bands = (u32, Vec<(i64, u32)>);

struct Claim {
    pub name: String,
    pub shape: Vec<i64>,
    pub bands: Option<Bands>,
    pub encoding: Encoding,
    pub scales: Option<Scales>,
}

#[must_use]
fn claim(w: &Weight, tp: u32) -> Claim {
    let (encoding, scales) = match w.dtype {
        Dtype::Mxfp4
        | Dtype::Nvfp4
        | Dtype::E4m3tensor
        | Dtype::U4g64
        | Dtype::U8g64
        | Dtype::U4g32
        | Dtype::U4g64tiled
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128 => (grouped(w), Some(scaling(w))),
        Dtype::Bf16
        | Dtype::F16
        | Dtype::F32
        | Dtype::I32
        | Dtype::U32
        | Dtype::U8
        | Dtype::I8
        | Dtype::E4m3
        | Dtype::E8m0
        | Dtype::E5m2
        | Dtype::I64
        | Dtype::I16
        | Dtype::U64
        | Dtype::U16
        | Dtype::Bool => (encoding(w.dtype), None),
        // Self-contained blocks (K-quants and PTQ1_0): one inline plane, no
        // separate scales/biases — the block carries its own scale.
        Dtype::U2g16k
        | Dtype::I3g16k
        | Dtype::U4g32k
        | Dtype::U5g32k
        | Dtype::I6g16k
        | Dtype::Ptq1_0 => (encoding(w.dtype), None),
        Dtype::E2m1 | Dtype::KvU4 => panic!(
            "this dtype names a kv-page quantization scheme, not a stored \
             weight plane; no load contract declares one"
        ),
        Dtype::E4m3row | Dtype::E4m3tile128 | Dtype::E4m3g128 => panic!(
            "a {:?} weight is served but no load contract declares one",
            w.dtype
        ),
        Dtype::Fp8b128 | Dtype::Fp8b32 | Dtype::Mxfp8 => panic!(
            "`{}`: a block-fp8 weight is claimed plane by plane, its raw codes and \
             exponents each their own",
            w.name
        ),
    };
    Claim {
        name: w.name.clone(),
        shape: banded_rows(w, whole(w, tp)),
        bands: banding(w, tp),
        encoding,
        scales,
    }
}

fn banded_rows(w: &Weight, shape: Vec<i64>) -> Vec<i64> {
    if w.dtype != Dtype::U4g64tiled {
        return shape;
    }
    let mut shape = shape;
    let band = i64::from(TILED_BAND);
    let rows = shape.first_mut().unwrap_or_else(|| {
        panic!(
            "`{}` is a tiled affine weight declared with no rows",
            w.name
        )
    });
    *rows = (*rows + band - 1) / band * band;
    shape
}

fn resolve(src: &ztensor::Source, claim: Claim) -> Result<Vec<TensorContract>, Error> {
    let mut tensors = Vec::new();
    let Claim {
        name,
        shape,
        bands,
        encoding,
        scales,
    } = claim;
    let stored = stored_encoding(src, &name)?;
    let read = padded(
        &name,
        bands.as_ref(),
        &shape,
        stored_shape(src, &name).as_deref(),
        &stored,
    );
    let expr = ladder(&name, read, &stored, &encoding)?;
    tensors.push(match &encoding {
        Encoding::Raw(_) => {
            TensorContract::new(name.clone(), expr, shape.clone(), encoding.clone())
        }
        Encoding::Quant(_) => TensorContract::inferred(name.clone(), expr, encoding.clone()),
    });
    match (&stored, scales) {
        (Encoding::Quant(_), Some(pairing)) => {
            tensors.extend(interned(src, &name, &shape, bands, pairing)?);
        }
        (Encoding::Raw(_), Some(_)) => {}
        (Encoding::Quant(_), None) | (Encoding::Raw(_), None) => {}
    }
    Ok(tensors)
}
fn declare(src: &ztensor::Source, w: &Weight, expr: Expr) -> Result<TensorContract, Error> {
    let want = match encoding(w.dtype) {
        // A self-contained block (K-quants, PTQ1_0) already carries its grouping
        // in the payload; it takes no channel axis. Leaving it ungrouped lets its
        // `want` compare EQUAL to the stored spec so `ladder` copies it through
        // unchanged (no decode, no re-encode) instead of forcing a conversion.
        Encoding::Quant(spec) if spec.scheme.is_self_contained() => Encoding::Quant(spec),
        Encoding::Quant(_) => grouped(w),
        raw => raw,
    };
    placed_wants_its_own_order(w, "an expression")?;
    let stored = agreed(src, &w.name, &expr)?;
    if matches!(stored, Encoding::Quant(_)) {
        holds_the_declared_count(src, w, &expr)?;
    }
    let expr = ladder(&w.name, expr, &stored, &want)?;
    Ok(match &stored {
        Encoding::Raw(_) => TensorContract::new(w.name.clone(), expr, extents(w), want),
        Encoding::Quant(_) => TensorContract::inferred(w.name.clone(), expr, want),
    })
}

fn read_over(
    src: &ztensor::Source,
    w: &Weight,
    from: &str,
    over: Expr,
) -> Result<Vec<TensorContract>, Error> {
    let want = encoding(w.dtype);
    let stored = stored_encoding(src, from)?;
    let applied = format!("{}.read", w.name);
    let adapted = ladder(from, Expr::out(applied.clone()), &stored, &want)?;
    Ok(vec![
        TensorContract::new(applied, over, extents(w), stored).internal(),
        TensorContract::new(w.name.clone(), adapted, extents(w), want),
    ])
}

fn holds_the_declared_count(src: &ztensor::Source, w: &Weight, expr: &Expr) -> Result<(), Error> {
    let mut stored: i128 = 0;
    for source in expr.sources() {
        let Some(tensor) = src.get(source) else {
            return Err(Error::Missing(source.to_string()));
        };
        stored += tensor
            .shape()
            .iter()
            .map(|&n| i128::from(n))
            .product::<i128>();
    }
    let asked: i128 = extents(w).iter().map(|&n| i128::from(n)).product();
    if stored > 1 && asked > 0 && stored != asked {
        return Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!(
                "the checkpoint holds {stored} values under this name and this \
                 model reads it as {:?} ({asked} values)",
                extents(w)
            ),
        });
    }
    Ok(())
}

fn copy(
    src: &ztensor::Source,
    w: &Weight,
    from: impl Into<String>,
) -> Result<TensorContract, Error> {
    declare(src, w, Expr::src(from))
}

fn fused(
    src: &ztensor::Source,
    w: &Weight,
    parts: impl IntoIterator<Item = String>,
) -> Result<TensorContract, Error> {
    let parts: Vec<String> = parts.into_iter().collect();
    let legs = parts.iter().cloned().map(Expr::src).collect();
    declare(src, w, Expr::concat(pack_axis(w), legs))
}

fn decoded_legs(
    src: &ztensor::Source,
    w: &Weight,
    parts: &[String],
) -> Option<Result<Vec<TensorContract>, Error>> {
    let want = encoding(w.dtype);
    if !matches!(want, Encoding::Raw(_)) {
        return None;
    }
    let mut tensors = Vec::with_capacity(parts.len() + 1);
    let mut legs = Vec::with_capacity(parts.len());
    for (at, part) in parts.iter().enumerate() {
        let stored = match stored_encoding(src, part) {
            Ok(stored) => stored,
            Err(why) => return Some(Err(why)),
        };
        let decoded = format!("{}.leg{at}", w.name);
        let expr = match ladder(part, Expr::src(part.clone()), &stored, &want) {
            Ok(expr) => expr,
            Err(why) => return Some(Err(why)),
        };
        tensors.push(TensorContract::inferred(decoded.clone(), expr, want.clone()).internal());
        legs.push(Expr::out(decoded));
    }
    tensors.push(TensorContract::new(
        w.name.clone(),
        Expr::concat(pack_axis(w), legs),
        extents(w),
        want,
    ));
    Some(Ok(tensors))
}

fn stored_raw(src: &ztensor::Source, name: &str) -> Result<bool, Error> {
    Ok(matches!(
        stored_encoding(src, name)?,
        Encoding::Raw(DType::Bf16 | DType::F16 | DType::F32)
    ))
}

fn stored_block(src: &ztensor::Source, name: &str) -> Result<bool, Error> {
    Ok(match stored_encoding(src, name)? {
        Encoding::Quant(spec) => spec.scheme.is_self_contained(),
        Encoding::Raw(_) => false,
    })
}

/// `w` encoded from `values`, an expression over self-contained blocks: each
/// block is decoded to bf16 first, so `values` computes over plain elements
/// (a gather may cut a block's groups) and its result is encoded once.
fn reencoded_from_block(
    src: &ztensor::Source,
    w: &Weight,
    values: Expr,
) -> Result<Vec<TensorContract>, Error> {
    let bf16 = encoding(Dtype::Bf16);
    let want = grouped(w);
    let mut tensors = Vec::new();
    let values = match over_decoded(src, w, values, &mut tensors)? {
        Expr::Out(decoded) => decoded,
        values => {
            let joined = format!("{}.values", w.name);
            tensors.push(TensorContract::new(joined.clone(), values, extents(w), bf16).internal());
            joined
        }
    };
    tensors.push(TensorContract::new(
        w.name.clone(),
        Expr::out(values).cast(want.clone()),
        extents(w),
        want,
    ));
    Ok(tensors)
}

/// `w` landing what `expr` computes. Over self-contained blocks of another
/// scheme it computes over their decoded values, as a plain read of them does
/// (`planes`); a gather may then cut their groups.
fn computed(src: &ztensor::Source, w: &Weight, expr: Expr) -> Result<Vec<TensorContract>, Error> {
    if placed_road_covers(w) {
        return placed_from_encode(w, |canon| computed(src, canon, expr));
    }
    let want = encoding(w.dtype);
    let blocks = expr.sources().into_iter().try_fold(true, |all, part| {
        Ok::<bool, Error>(all && stored_block(src, part)? && stored_encoding(src, part)? != want)
    })?;
    if !blocks || !matches!(want, Encoding::Quant(_)) {
        return Ok(vec![declare(src, w, expr)?]);
    }
    reencoded_from_block(src, w, expr)
}

/// `expr` with each checkpoint tensor it reads replaced by that tensor
/// decoded to bf16, one internal tensor per source however often it is read.
/// A gather over a decoded tensor lands as its own tensor too: a plan moves
/// only whole computed buffers into another.
fn over_decoded(
    src: &ztensor::Source,
    w: &Weight,
    expr: Expr,
    tensors: &mut Vec<TensorContract>,
) -> Result<Expr, Error> {
    let bf16 = encoding(Dtype::Bf16);
    let mut parts: Vec<String> = Vec::new();
    for part in expr.sources() {
        if !parts.iter().any(|p| p == part) {
            parts.push(part.to_string());
        }
    }
    for (at, part) in parts.iter().enumerate() {
        let stored = stored_encoding(src, part)?;
        let decoded = ladder(part, Expr::src(part.clone()), &stored, &bf16)?;
        tensors.push(
            TensorContract::inferred(format!("{}.block{at}", w.name), decoded, bf16.clone())
                .internal(),
        );
    }
    let mut gathered = 0;
    Ok(rebuilt(expr, &mut |expr| match expr {
        Expr::Src(part) => {
            let at = parts
                .iter()
                .position(|p| *p == part)
                .expect("every source was decoded");
            Expr::out(format!("{}.block{at}", w.name))
        }
        gather @ Expr::Gather { .. } => {
            let name = format!("{}.gathered{gathered}", w.name);
            gathered += 1;
            tensors.push(TensorContract::inferred(name.clone(), gather, bf16.clone()).internal());
            Expr::out(name)
        }
        expr => expr,
    }))
}

/// `expr` rebuilt bottom-up through `node`.
fn rebuilt(expr: Expr, node: &mut impl FnMut(Expr) -> Expr) -> Expr {
    let expr = expr
        .map_children(|child| Ok(rebuilt(child, node)))
        .expect("rebuilding an expression cannot fail");
    node(expr)
}

fn placed_from_encode(
    w: &Weight,
    canonical: impl FnOnce(&Weight) -> Result<Vec<TensorContract>, Error>,
) -> Result<Vec<TensorContract>, Error> {
    let canon_name = format!("{}.canon", w.name);
    let canon = Weight {
        name: canon_name.clone(),
        dtype: w.dtype.canonical(),
        ..w.clone()
    };
    let mut out: Vec<TensorContract> = canonical(&canon)?
        .into_iter()
        .map(TensorContract::internal)
        .collect();
    let pairing = scaling(w);
    let landed = banded_rows(w, extents(w));
    let counted = divided(&landed, pairing.channel_axis, pairing.group_size, &w.name);
    let factor_target = |shape: Vec<i64>| TensorType::new(shape, encoding(Dtype::Bf16));
    let relabel = |name: String, from: String| {
        TensorContract::new(
            name,
            Expr::out(from).repack(
                RepackLayout::TiledAffineFactor,
                factor_target(counted.clone()),
            ),
            counted.clone(),
            encoding(Dtype::Bf16),
        )
    };
    out.push(TensorContract::inferred(
        w.name.clone(),
        Expr::out(canon_name.clone()).repack(
            RepackLayout::TiledAffineU4Weight,
            TensorType::new(landed, grouped(w)),
        ),
        grouped(w),
    ));
    out.push(
        relabel(crate::scales_name(&w.name), crate::scales_name(&canon_name)).scaling(pairing),
    );
    out.push(
        relabel(crate::biases_name(&w.name), crate::biases_name(&canon_name))
            .offsetting(w.name.clone()),
    );
    Ok(out)
}

fn placed_road_covers(w: &Weight) -> bool {
    w.dtype == Dtype::U4g64tiled
}

fn placed_wants_its_own_order(w: &Weight, road: &str) -> Result<(), Error> {
    if !w.dtype.placed() {
        return Ok(());
    }
    Err(Error::Illegible {
        name: w.name.clone(),
        detail: format!(
            "it is declared {:?}, which names a byte order and not just a width, and \
             this checkpoint states it as {road} -- a road that encodes, and an encode \
             publishes the canonical order. Import the row that declares {:?} instead, \
             or state this weight as MLX affine codes (`.weight` beside `.scales` and \
             `.biases`), which is the one source shape that lands the placed order.",
            w.dtype,
            w.dtype.canonical()
        ),
    })
}

fn planes(
    src: &ztensor::Source,
    w: &Weight,
    from: impl Into<String>,
) -> Result<Vec<TensorContract>, Error> {
    let from = from.into();
    if int4::stored_int4(src, &from) {
        return int4::int4_planes(src, w, &[from]);
    }
    if w.dtype.fp8_blocks().is_some() {
        return fp8_block_planes(src, w, vec![from]);
    }
    if stored_raw(src, &from)? {
        if placed_road_covers(w) {
            return placed_from_encode(w, |canon| Ok(vec![copy(src, canon, from)?]));
        }
        placed_wants_its_own_order(w, "a raw plane")?;
        return Ok(vec![copy(src, w, from)?]);
    }
    if stored_block(src, &from)? && matches!(encoding(w.dtype), Encoding::Quant(_)) {
        // Native pass-through: the file already holds this exact self-contained
        // scheme (e.g. a PTQ1_0 ternary block served as-is). Keep the raw block
        // bytes — no decode to fp, no re-encode. `ladder` copies the source expr
        // through unchanged because stored == want.
        if stored_encoding(src, &from)? == encoding(w.dtype) {
            return Ok(vec![copy(src, w, from)?]);
        }
        if placed_road_covers(w) {
            return placed_from_encode(w, |canon| {
                reencoded_from_block(src, canon, Expr::src(from))
            });
        }
        placed_wants_its_own_order(w, "a self-contained block")?;
        return reencoded_from_block(src, w, Expr::src(from));
    }
    if matches!(encoding(w.dtype), Encoding::Raw(_)) && stored_mlx_codes(src, &from)? {
        return affine_decoded(src, w, from);
    }
    match w.dtype {
        Dtype::U4g64tiled => tiled_planes(src, w, vec![from]),
        Dtype::U4g64
        | Dtype::U8g64
        | Dtype::U4g32
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128 => affine_planes(src, w, vec![from]),
        Dtype::Mxfp4 => mx_planes(src, w, vec![from]),
        Dtype::Nvfp4 => nv_planes(src, w, vec![from]),
        _ => Ok(vec![copy(src, w, from)?]),
    }
}

fn planes_fused(
    src: &ztensor::Source,
    w: &Weight,
    parts: impl IntoIterator<Item = String>,
) -> Result<Vec<TensorContract>, Error> {
    let parts: Vec<String> = parts.into_iter().collect();
    if parts.iter().all(|part| int4::stored_int4(src, part)) {
        return int4::int4_planes(src, w, &parts);
    }
    if w.dtype.fp8_blocks().is_some() {
        return fp8_block_planes(src, w, parts);
    }
    if parts.iter().try_fold(true, |all, part| {
        Ok::<bool, Error>(all && stored_raw(src, part)?)
    })? {
        if placed_road_covers(w) {
            return placed_from_encode(w, |canon| Ok(vec![fused(src, canon, parts)?]));
        }
        placed_wants_its_own_order(w, "raw planes to fuse")?;
        return Ok(vec![fused(src, w, parts)?]);
    }
    if parts.iter().try_fold(true, |all, part| {
        Ok::<bool, Error>(all && stored_block(src, part)?)
    })? && matches!(encoding(w.dtype), Encoding::Quant(_))
    {
        // Native pass-through of fused self-contained blocks: when every leg is
        // already stored as exactly the declared scheme (e.g. PTQ1_0), concatenate
        // the raw block bytes along the row axis — no decode, no re-encode.
        let want = encoding(w.dtype);
        if parts.iter().try_fold(true, |all, part| {
            Ok::<bool, Error>(all && stored_encoding(src, part)? == want)
        })? {
            return Ok(vec![fused(src, w, parts)?]);
        }
        let legs =
            |w: &Weight| joined(pack_axis(w), parts.iter().cloned().map(Expr::src).collect());
        if placed_road_covers(w) {
            return placed_from_encode(w, |canon| reencoded_from_block(src, canon, legs(canon)));
        }
        placed_wants_its_own_order(w, "self-contained blocks to fuse")?;
        return reencoded_from_block(src, w, legs(w));
    }
    match w.dtype {
        Dtype::U4g64tiled => tiled_planes(src, w, parts),
        Dtype::U4g64
        | Dtype::U8g64
        | Dtype::U4g32
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128 => affine_planes(src, w, parts),
        Dtype::Mxfp4 => mx_planes(src, w, parts),
        Dtype::Nvfp4 => nv_planes(src, w, parts),
        _ => match fused(src, w, parts.clone()) {
            Ok(contract) => Ok(vec![contract]),
            Err(disagreement) => decoded_legs(src, w, &parts).unwrap_or(Err(disagreement)),
        },
    }
}

fn affine_planes(
    src: &ztensor::Source,
    w: &Weight,
    parts: Vec<String>,
) -> Result<Vec<TensorContract>, Error> {
    let axis = if parts.len() > 1 { pack_axis(w) } else { 0 };
    let stored = stored_affine(src, w, &parts[0], axis)?;
    if stored != w.dtype {
        return affine_reencoded(src, w, parts, axis, stored);
    }
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    let mut legs = Vec::new();
    for part in &parts {
        let stem = part
            .strip_suffix(".weight")
            .ok_or_else(|| Error::Illegible {
                name: w.name.clone(),
                detail: format!(
                    "`{part}` holds MLX affine codes, whose scales and biases \
                     are named beside a `.weight`, and it does not end in one"
                ),
            })?;
        let unpacked = unpacked_extents(src, w, part)?;
        legs.push(unpacked.clone());
        codes.push(Expr::src(part.clone()).transmute(TensorType::new(unpacked, grouped(w))));
        scales.push(crate::scales_name(stem));
        biases.push(crate::biases_name(stem));
    }
    holds_the_declared_rectangle(w, axis, &legs)?;
    let pairing = scaling(w);
    let counted = divided(
        &extents(w),
        pairing.channel_axis,
        pairing.group_size,
        &w.name,
    );
    Ok(vec![
        TensorContract::inferred(w.name.clone(), joined(axis, codes), grouped(w)),
        factors(
            src,
            crate::scales_name(&w.name),
            &scales,
            counted.clone(),
            axis,
        )?
        .scaling(pairing),
        factors(src, crate::biases_name(&w.name), &biases, counted, axis)?
            .offsetting(w.name.clone()),
    ])
}

fn column_axis(src: &ztensor::Source, name: &str) -> Result<u8, Error> {
    let Some(tensor) = src.get(name) else {
        return Err(Error::Missing(name.to_string()));
    };
    Ok(u8::try_from(tensor.shape().len().saturating_sub(1)).expect("a rank inside u8"))
}

/// Columns `start .. start + len` of the MLX affine bank `from`, read as
/// `w`'s planes: its words, scales and biases each sliced over the columns.
fn affine_columns(
    src: &ztensor::Source,
    w: &Weight,
    from: &str,
    start: i64,
    len: i64,
) -> Result<Vec<TensorContract>, Error> {
    let refuse = |detail: String| Error::Illegible {
        name: w.name.clone(),
        detail,
    };
    if !matches!(
        w.dtype,
        Dtype::U4g64 | Dtype::U8g64 | Dtype::U4g32 | Dtype::U2g32 | Dtype::U2g64 | Dtype::U2g128
    ) {
        return Err(refuse(format!(
            "`{from}` holds MLX affine codes and this model reads its columns as {:?}, \
             which is not an affine bank",
            w.dtype
        )));
    }
    let stem = from
        .strip_suffix(".weight")
        .expect("MLX codes are named `.weight`");
    let (scales, biases) = (crate::scales_name(stem), crate::biases_name(stem));
    let per_word = word_codes(w.dtype);
    let group = i64::from(group_of(w.dtype));
    let codes = unpacked_extents(src, w, from)?;
    let groups = src
        .get(&scales)
        .and_then(|t| t.shape().last().copied())
        .map(|n| i64::try_from(n).expect("an extent inside i64"));
    let columns = *codes.last().expect("a bank has a contracted axis");
    if groups.map(|g| g * group) != Some(columns) {
        return Err(refuse(format!(
            "`{from}` holds {columns} columns of codes against {groups:?} groups in \
             `{scales}`, which is not one group of {group} per scale: it is stored \
             at another width or grouping than {:?}",
            w.dtype
        )));
    }
    if start % group != 0 || len % group != 0 || start + len > columns {
        return Err(refuse(format!(
            "columns {start}..{} of `{from}` ({columns} wide) do not keep whole groups \
             of {group}, so they are not a slice of its codes and its scales",
            start + len
        )));
    }
    let axis = u8::try_from(codes.len() - 1).expect("a rank inside u8");
    let mut shape = codes.clone();
    *shape.last_mut().expect("a contracted axis") = len;
    if shape != extents(w) {
        return Err(refuse(format!(
            "columns {start}..{} of `{from}` are {shape:?} and this text declares {:?}",
            start + len,
            extents(w)
        )));
    }
    let pairing = scaling(w);
    let counted = divided(&shape, pairing.channel_axis, pairing.group_size, &w.name);
    let factor = |name: String, of: &str| -> Result<TensorContract, Error> {
        let expr = Expr::src(of.to_string()).slice(axis, start / group, len / group);
        let stored = agreed(src, &name, &expr)?;
        let want = encoding(Dtype::Bf16);
        Ok(TensorContract::new(
            name.clone(),
            ladder(&name, expr, &stored, &want)?,
            counted.clone(),
            want,
        ))
    };
    // A transmute renames a whole tensor: the sliced words are published
    // first, and their name is what reads as codes.
    let mut words = shape.clone();
    *words.last_mut().expect("a contracted axis") = len / per_word;
    let sliced = format!("{}.words", w.name);
    Ok(vec![
        TensorContract::new(
            sliced.clone(),
            Expr::src(from.to_string()).slice(axis, start / per_word, len / per_word),
            words,
            Encoding::Raw(DType::U32),
        )
        .internal(),
        TensorContract::inferred(
            w.name.clone(),
            Expr::out(sliced).transmute(TensorType::new(shape.clone(), grouped(w))),
            grouped(w),
        ),
        factor(crate::scales_name(&w.name), &scales)?.scaling(pairing),
        factor(crate::biases_name(&w.name), &biases)?.offsetting(w.name.clone()),
    ])
}

fn stored_affine(src: &ztensor::Source, w: &Weight, part: &str, axis: u8) -> Result<Dtype, Error> {
    let declared = extents(w);
    let Some(&k) = declared.last() else {
        return Ok(w.dtype);
    };
    if declared.len() > 1 && usize::from(axis) == declared.len() - 1 {
        return Ok(w.dtype);
    }
    let Some(tensor) = src.get(part) else {
        return Err(Error::Missing(part.to_string()));
    };
    let Some(&words) = tensor.shape().last() else {
        return Ok(w.dtype);
    };
    let words = i64::try_from(words).expect("an extent no i64 holds");
    if words <= 0 || k % words != 0 || 32 % (k / words) != 0 {
        return Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!(
                "`{part}` packs {words} words a row over a declared width of {k}; \
                 MLX affine codes fill a 32-bit word at 2, 4 or 8 bits"
            ),
        });
    }
    let bits = 32 / (k / words);
    let group = match encoding(w.dtype) {
        Encoding::Quant(spec) => spec.normalized_group_size(),
        Encoding::Raw(_) => return Ok(w.dtype),
    };
    match (bits, group) {
        (8, 64) => Ok(Dtype::U8g64),
        (4, 64) => Ok(Dtype::U4g64),
        (4, 32) => Ok(Dtype::U4g32),
        (2, 32) => Ok(Dtype::U2g32),
        (2, 64) => Ok(Dtype::U2g64),
        (2, 128) => Ok(Dtype::U2g128),
        _ => Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!(
                "`{part}` is stored at {bits} bits in groups of {group}, a width \
                 no affine row here names"
            ),
        }),
    }
}

fn affine_reencoded(
    src: &ztensor::Source,
    w: &Weight,
    parts: Vec<String>,
    axis: u8,
    stored: Dtype,
) -> Result<Vec<TensorContract>, Error> {
    let stored_w = Weight {
        dtype: stored,
        ..w.clone()
    };
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    let mut legs = Vec::new();
    for part in &parts {
        let stem = part
            .strip_suffix(".weight")
            .ok_or_else(|| Error::Illegible {
                name: w.name.clone(),
                detail: format!(
                    "`{part}` holds MLX affine codes, whose scales and biases \
                     are named beside a `.weight`, and it does not end in one"
                ),
            })?;
        let unpacked = unpacked_extents(src, &stored_w, part)?;
        legs.push(unpacked.clone());
        codes
            .push(Expr::src(part.clone()).transmute(TensorType::new(unpacked, grouped(&stored_w))));
        scales.push(crate::scales_name(stem));
        biases.push(crate::biases_name(stem));
    }
    holds_the_declared_rectangle(w, axis, &legs)?;
    let pairing = scaling(&stored_w);
    let counted = divided(
        &extents(w),
        pairing.channel_axis,
        pairing.group_size,
        &w.name,
    );
    let stored_name = format!("{}.stored", w.name);
    let stored_scales = format!("{stored_name}.scales");
    let stored_biases = format!("{stored_name}.biases");
    let scaled = format!("{stored_name}.scaled");
    let values = format!("{stored_name}.values");
    let bf16 = encoding(Dtype::Bf16);
    Ok(vec![
        TensorContract::inferred(stored_name.clone(), joined(axis, codes), grouped(&stored_w))
            .internal(),
        factors(src, stored_scales.clone(), &scales, counted.clone(), axis)?.internal(),
        factors(src, stored_biases.clone(), &biases, counted, axis)?.internal(),
        TensorContract::new(
            scaled.clone(),
            Expr::out(stored_name).scale_per_block(Expr::out(stored_scales)),
            extents(w),
            bf16.clone(),
        )
        .internal(),
        TensorContract::new(
            values.clone(),
            Expr::out(scaled).bias_per_block(Expr::out(stored_biases)),
            extents(w),
            bf16,
        )
        .internal(),
        TensorContract::inferred(
            w.name.clone(),
            Expr::out(values).cast(grouped(w)),
            grouped(w),
        ),
    ])
}

fn stored_mlx_codes(src: &ztensor::Source, name: &str) -> Result<bool, Error> {
    let Some(stem) = name.strip_suffix(".weight") else {
        return Ok(false);
    };
    Ok(
        matches!(stored_encoding(src, name)?, Encoding::Raw(DType::U32))
            && src.get(&crate::scales_name(stem)).is_some(),
    )
}

fn affine_decoded(
    src: &ztensor::Source,
    w: &Weight,
    from: String,
) -> Result<Vec<TensorContract>, Error> {
    let illegible = |detail: String| Error::Illegible {
        name: w.name.clone(),
        detail,
    };
    let stem = from
        .strip_suffix(".weight")
        .ok_or_else(|| {
            illegible(format!(
                "`{from}` holds MLX affine codes, whose scales and biases \
                 are named beside a `.weight`, and it does not end in one"
            ))
        })?
        .to_string();
    let last = |name: &str| -> Result<i64, Error> {
        let tensor = src
            .get(name)
            .ok_or_else(|| Error::Missing(name.to_string()))?;
        let extent = *tensor.shape().last().ok_or_else(|| {
            illegible(format!(
                "`{name}` is a scalar and a bank has a contracted axis"
            ))
        })?;
        Ok(i64::try_from(extent).expect("an extent no i64 holds"))
    };
    let k = *extents(w).last().ok_or_else(|| {
        illegible(format!(
            "`{}` is a scalar and a bank has a contracted axis",
            w.name
        ))
    })?;
    let words = last(&from)?;
    let scale_groups = last(&crate::scales_name(&stem))?;
    if words <= 0
        || k % words != 0
        || 32 % (k / words) != 0
        || scale_groups <= 0
        || k % scale_groups != 0
    {
        return Err(illegible(format!(
            "`{from}` packs {words} words and {scale_groups} scale groups a row over a \
             declared width of {k}; MLX affine codes fill a 32-bit word at 2, 4 or 8 bits"
        )));
    }
    let bits = 32 / (k / words);
    let group = k / scale_groups;
    let stored = match (bits, group) {
        (8, 64) => Dtype::U8g64,
        (4, 64) => Dtype::U4g64,
        (4, 32) => Dtype::U4g32,
        (2, 32) => Dtype::U2g32,
        (2, 64) => Dtype::U2g64,
        (2, 128) => Dtype::U2g128,
        _ => {
            return Err(illegible(format!(
                "`{from}` is stored at {bits} bits in groups of {group}, a width \
                 no affine row here names"
            )));
        }
    };
    let stored_w = Weight {
        dtype: stored,
        ..w.clone()
    };
    let stem = stem.as_str();
    let unpacked = unpacked_extents(src, &stored_w, &from)?;
    holds_the_declared_rectangle(w, 0, std::slice::from_ref(&unpacked))?;
    let codes = Expr::src(from.clone()).transmute(TensorType::new(unpacked, grouped(&stored_w)));
    let pairing = scaling(&stored_w);
    let counted = divided(
        &extents(w),
        pairing.channel_axis,
        pairing.group_size,
        &w.name,
    );
    let stored_name = format!("{}.stored", w.name);
    let stored_scales = format!("{stored_name}.scales");
    let stored_biases = format!("{stored_name}.biases");
    let scaled = format!("{stored_name}.scaled");
    let values = format!("{stored_name}.values");
    let bf16 = encoding(Dtype::Bf16);
    let want = encoding(w.dtype);
    let mut out = vec![
        TensorContract::inferred(stored_name.clone(), codes, grouped(&stored_w)).internal(),
        factors(
            src,
            stored_scales.clone(),
            &[crate::scales_name(stem)],
            counted.clone(),
            0,
        )?
        .internal(),
        factors(
            src,
            stored_biases.clone(),
            &[crate::biases_name(stem)],
            counted,
            0,
        )?
        .internal(),
        TensorContract::new(
            scaled.clone(),
            Expr::out(stored_name).scale_per_block(Expr::out(stored_scales)),
            extents(w),
            bf16.clone(),
        )
        .internal(),
    ];
    if want == bf16 {
        out.push(TensorContract::new(
            w.name.clone(),
            Expr::out(scaled).bias_per_block(Expr::out(stored_biases)),
            extents(w),
            bf16,
        ));
    } else {
        out.push(
            TensorContract::new(
                values.clone(),
                Expr::out(scaled).bias_per_block(Expr::out(stored_biases)),
                extents(w),
                bf16,
            )
            .internal(),
        );
        out.push(TensorContract::new(
            w.name.clone(),
            Expr::out(values).cast(want.clone()),
            extents(w),
            want,
        ));
    }
    Ok(out)
}

fn affine_stacked(
    src: &ztensor::Source,
    w: &Weight,
    rows: Vec<Vec<String>>,
) -> Result<Vec<TensorContract>, Error> {
    let illegible = |detail: String| Error::Illegible {
        name: w.name.clone(),
        detail,
    };
    let Encoding::Quant(QuantSpec {
        scheme: QuantScheme::MlxAffineU4,
        ..
    }) = encoding(w.dtype)
    else {
        return Err(illegible(format!(
            "it is {:?}, and only an MLX affine bank stacks from `.weight/.scales/.biases` rows",
            w.dtype
        )));
    };
    let declared = extents(w);
    let pairing = scaling(w);
    let counted = divided(&declared, pairing.channel_axis, pairing.group_size, &w.name);
    let group = i64::from(pairing.group_size);
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    let mut factor_stored: Option<Encoding> = None;
    let mut row_shape: Option<Vec<i64>> = None;
    for parts in &rows {
        let axis = if parts.len() > 1 { pack_axis(w) } else { 0 };
        let mut row_codes = Vec::new();
        let mut row_scales = Vec::new();
        let mut row_biases = Vec::new();
        let mut shape: Option<Vec<i64>> = None;
        for part in parts {
            let stem = part.strip_suffix(".weight").ok_or_else(|| {
                illegible(format!(
                    "`{part}` holds MLX affine codes, whose scales and biases \
                     are named beside a `.weight`, and it does not end in one"
                ))
            })?;
            let mut leg = unpacked_extents(src, w, part)?;
            leg.insert(0, 1);
            let mut factor_shape = leg.clone();
            let last = factor_shape.len() - 1;
            if factor_shape[last] % group != 0 {
                return Err(illegible(format!(
                    "`{part}` contracts over {}, which is not a whole number of {group}-code blocks",
                    factor_shape[last]
                )));
            }
            factor_shape[last] /= group;
            match &mut shape {
                None => shape = Some(leg.clone()),
                Some(joined) => {
                    if joined.len() != leg.len() {
                        return Err(illegible(format!(
                            "its stored parts are rank {} and rank {}, and parts that join \
                             into one row have one rank",
                            joined.len(),
                            leg.len()
                        )));
                    }
                    let at = usize::from(axis);
                    for (i, (into, part_extent)) in joined.iter_mut().zip(&leg).enumerate() {
                        if i == at {
                            *into += *part_extent;
                        } else if *into != *part_extent {
                            return Err(illegible(format!(
                                "`{part}` differs at axis {i} ({into} against {part_extent}) \
                                 from the part it joins on axis {at}"
                            )));
                        }
                    }
                }
            }
            row_codes.push(Expr::src(part.clone()).transmute(TensorType::new(leg, grouped(w))));
            for (name, into) in [
                (crate::scales_name(stem), &mut row_scales),
                (crate::biases_name(stem), &mut row_biases),
            ] {
                let stored = stored_encoding(src, &name)?;
                match &factor_stored {
                    None => factor_stored = Some(stored.clone()),
                    Some(first) if *first != stored => {
                        return Err(illegible(format!(
                            "`{name}` is stored {stored:?} and an earlier factor plane {first:?}; \
                             one bank's factors share one representation"
                        )));
                    }
                    Some(_) => {}
                }
                into.push(Expr::src(name).transmute(TensorType::new(factor_shape.clone(), stored)));
            }
        }
        let shape =
            shape.ok_or_else(|| illegible("a row is built from no checkpoint tensor".into()))?;
        match &row_shape {
            None => row_shape = Some(shape),
            Some(first) if *first != shape => {
                return Err(illegible(format!(
                    "its rows are stored {first:?} and {shape:?}; the rows of one bank share one shape"
                )));
            }
            Some(_) => {}
        }
        codes.push(joined(axis, row_codes));
        scales.push(joined(axis, row_scales));
        biases.push(joined(axis, row_biases));
    }
    let mut shape = row_shape.ok_or_else(|| illegible("it is built from no rows at all".into()))?;
    shape[0] = i64::try_from(rows.len()).expect("a row count inside i64");
    if shape != declared {
        return Err(illegible(format!(
            "the file stores it {shape:?} and this text declares it {declared:?}; a text reads \
             the widths it states, so this row is not the one that reads this checkpoint"
        )));
    }
    let stored = factor_stored.expect("a row names its factors");
    let want = encoding(Dtype::Bf16);
    let scales_name = crate::scales_name(&w.name);
    let biases_name = crate::biases_name(&w.name);
    let scales = ladder(&scales_name, joined(0, scales), &stored, &want)?;
    let biases = ladder(&biases_name, joined(0, biases), &stored, &want)?;
    Ok(vec![
        TensorContract::inferred(w.name.clone(), joined(0, codes), grouped(w)),
        TensorContract::new(scales_name, scales, counted.clone(), want.clone()).scaling(pairing),
        TensorContract::new(biases_name, biases, counted, want).offsetting(w.name.clone()),
    ])
}

fn mx_planes(
    src: &ztensor::Source,
    w: &Weight,
    parts: Vec<String>,
) -> Result<Vec<TensorContract>, Error> {
    let axis = if parts.len() > 1 { pack_axis(w) } else { 0 };
    let pairing = scaling(w);
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    let mut legs = Vec::new();
    for part in &parts {
        let stem = part
            .strip_suffix(".weight")
            .ok_or_else(|| Error::Illegible {
                name: w.name.clone(),
                detail: format!(
                    "`{part}` holds MLX mxfp4 codes, whose exponents are named \
                     beside a `.weight`, and it does not end in one"
                ),
            })?;
        let unpacked = unpacked_extents(src, w, part)?;
        let scale = crate::scales_name(stem);
        let stored = stored_encoding(src, &scale)?;
        if stored != Encoding::Raw(DType::U8) {
            return Err(Error::Illegible {
                name: w.name.clone(),
                detail: format!(
                    "`{scale}` is stored {stored:?}, and MLX mxfp4 exponents are \
                     read as one raw u8 (e8m0) per 32-code block"
                ),
            });
        }
        let counted = divided(&unpacked, pairing.channel_axis, pairing.group_size, &w.name);
        legs.push(unpacked.clone());
        codes.push(Expr::src(part.clone()).transmute(TensorType::new(unpacked, grouped(w))));
        scales.push(Expr::src(scale).transmute(TensorType::new(counted, encoding(Dtype::E8m0))));
    }
    holds_the_declared_rectangle(w, axis, &legs)?;
    let counted = divided(
        &extents(w),
        pairing.channel_axis,
        pairing.group_size,
        &w.name,
    );
    Ok(vec![
        TensorContract::inferred(w.name.clone(), joined(axis, codes), grouped(w)),
        TensorContract::new(
            crate::scales_name(&w.name),
            joined(axis, scales),
            counted,
            encoding(Dtype::E8m0),
        )
        .scaling(pairing),
    ])
}

/// An fp8b128 weight as DeepSeek releases it: raw e4m3 `{stem}.weight` beside
/// `{stem}.scale` (one e8m0 exponent a 128x128 block), landed as stored; legs
/// join along their rows, each a whole number of blocks.
fn fp8_block_planes(
    src: &ztensor::Source,
    w: &Weight,
    parts: Vec<String>,
) -> Result<Vec<TensorContract>, Error> {
    let illegible = |detail: String| Error::Illegible {
        name: w.name.clone(),
        detail,
    };
    let (br, bc) = w
        .dtype
        .fp8_blocks()
        .expect("only a block-fp8 weight reads its exponent planes");
    let (br, bc) = (i64::from(br), i64::from(bc));
    if w.shape.len() != 2 {
        return Err(illegible(format!(
            "it is declared {:?}; a block-fp8 weight is one projection",
            w.shape
        )));
    }
    let axis = if parts.len() > 1 { pack_axis(w) } else { 0 };
    if axis != 0 {
        return Err(illegible(
            "its legs join along the contracted axis, and its blocks are read whole".into(),
        ));
    }
    let shape_of = |name: &str| -> Vec<i64> {
        src.get(name)
            .map(|tensor| tensor.shape().iter().map(|&n| n as i64).collect())
            .unwrap_or_default()
    };
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    let mut legs = Vec::new();
    for (at, part) in parts.iter().enumerate() {
        let stem = part.strip_suffix(".weight").ok_or_else(|| {
            illegible(format!(
                "`{part}` holds fp8 codes, whose exponents are named beside a `.weight`, \
                 and it does not end in one"
            ))
        })?;
        let held = shape_of(part);
        let stored = stored_encoding(src, part)?;
        let [rows, cols] = held[..] else {
            return Err(illegible(format!(
                "`{part}` is stored {held:?}, not a matrix"
            )));
        };
        if stored != Encoding::Raw(DType::E4m3) || (at + 1 < parts.len() && rows % br != 0) {
            return Err(illegible(format!(
                "`{part}` is stored {stored:?} {held:?}; a block-fp8 leg is e4m3, a whole \
                 number of {br}-row blocks but the last"
            )));
        }
        let scale = format!("{stem}.scale");
        let blocks = vec![(rows + br - 1) / br, (cols + bc - 1) / bc];
        let stored = stored_encoding(src, &scale)?;
        if stored != Encoding::Raw(DType::E8m0) || shape_of(&scale) != blocks {
            return Err(illegible(format!(
                "`{scale}` is stored {stored:?} {:?}; block-fp8 exponents are one e8m0 a \
                 {br}x{bc} block, {blocks:?}",
                shape_of(&scale)
            )));
        }
        codes.push(Expr::src(part.clone()));
        scales.push(Expr::src(scale));
        legs.push(held);
    }
    holds_the_declared_rectangle(w, axis, &legs)?;
    let [n, k] = extents(w)[..] else {
        unreachable!("a two-dimensional weight has two extents")
    };
    Ok(vec![
        TensorContract::new(
            w.name.clone(),
            joined(axis, codes),
            vec![n, k],
            encoding(Dtype::E4m3),
        ),
        TensorContract::new(
            crate::scales_name(&w.name),
            joined(axis, scales),
            vec![(n + br - 1) / br, (k + bc - 1) / bc],
            encoding(Dtype::E8m0),
        ),
    ])
}

/// An NVFP4 weight from modelopt's planes: `{stem}.weight` (two e2m1 codes a
/// byte), `.weight_scale` (e4m3 per 16), `.weight_scale_2` (f32 tensor scale)
/// and `.input_scale` (static activation scale). Legs joined along rows keep
/// their own tensor scale (it lands per row) but read one activation, so the
/// first leg's input scale is the weight's.
fn nv_planes(
    src: &ztensor::Source,
    w: &Weight,
    parts: Vec<String>,
) -> Result<Vec<TensorContract>, Error> {
    let illegible = |detail: String| Error::Illegible {
        name: w.name.clone(),
        detail,
    };
    if w.shape.len() != 2 {
        return Err(illegible(format!(
            "it is declared {:?}; an NVFP4 bank of more than one slab is read leg by \
             leg in its format's own terms",
            w.shape
        )));
    }
    let axis = if parts.len() > 1 { pack_axis(w) } else { 0 };
    if axis != 0 {
        return Err(illegible(
            "its legs join along the contracted axis, and a row holds one tensor scale".into(),
        ));
    }
    let pairing = scaling(w);
    let f32_raw = encoding(Dtype::F32);
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    let mut tensors = Vec::new();
    let mut legs = Vec::new();
    for part in &parts {
        let stem = part.strip_suffix(".weight").ok_or_else(|| {
            illegible(format!(
                "`{part}` holds NVFP4 codes, whose scales are named beside a `.weight`, \
                 and it does not end in one"
            ))
        })?;
        let stored = stored_encoding(src, part)?;
        let held: Vec<i64> = src
            .get(part)
            .map(|tensor| tensor.shape().iter().map(|&n| n as i64).collect())
            .unwrap_or_default();
        let [rows, bytes] = held[..] else {
            return Err(illegible(format!(
                "`{part}` is stored {held:?}; a leg of NVFP4 codes is [rows, cols / 2] bytes"
            )));
        };
        if stored != Encoding::Raw(DType::U8) {
            return Err(illegible(format!(
                "`{part}` is stored {stored:?}, and NVFP4 codes are packed two to a u8"
            )));
        }
        let unpacked = vec![rows, 2 * bytes];
        let scale = format!("{stem}.weight_scale");
        let stored = stored_encoding(src, &scale)?;
        let counted = divided(&unpacked, pairing.channel_axis, pairing.group_size, &w.name);
        let held: Vec<i64> = src
            .get(&scale)
            .map(|tensor| tensor.shape().iter().map(|&n| n as i64).collect())
            .unwrap_or_default();
        if stored != Encoding::Raw(DType::E4m3) || held != counted {
            return Err(illegible(format!(
                "`{scale}` is stored {stored:?} {held:?}, and NVFP4 scales are one e4m3 \
                 per 16-code group, {counted:?}"
            )));
        }
        let tensor = format!("{stem}.weight_scale_2");
        one_f32(src, &tensor, &w.name)?;
        let rows = usize::try_from(unpacked[0]).expect("a row count inside usize");
        codes
            .push(Expr::src(part.clone()).transmute(TensorType::new(unpacked.clone(), grouped(w))));
        scales.push(Expr::src(scale));
        let one = Expr::src(tensor).transmute(TensorType::new(vec![1], f32_raw.clone()));
        tensors.push(if rows > 1 {
            one.gather(0, vec![0; rows])
        } else {
            one
        });
        legs.push(unpacked);
    }
    holds_the_declared_rectangle(w, axis, &legs)?;
    let first = parts[0].strip_suffix(".weight").unwrap_or(&parts[0]);
    let input = format!("{first}.input_scale");
    one_f32(src, &input, &w.name)?;
    // One product quantizes its activations under one static scale.
    let inputs: Vec<String> = parts
        .iter()
        .map(|part| {
            format!(
                "{}.input_scale",
                part.strip_suffix(".weight").unwrap_or(part)
            )
        })
        .collect();
    let input_expr = input_scale(src, &inputs, &w.name)?;
    let counted = divided(
        &extents(w),
        pairing.channel_axis,
        pairing.group_size,
        &w.name,
    );
    let rows = extents(w)[..1].to_vec();
    Ok(vec![
        TensorContract::inferred(w.name.clone(), joined(axis, codes), grouped(w)),
        TensorContract::new(
            crate::scales_name(&w.name),
            joined(axis, scales),
            counted,
            encoding(Dtype::E4m3),
        )
        .scaling(pairing),
        TensorContract::new(
            crate::global_scales_name(&w.name),
            joined(axis, tensors),
            rows,
            f32_raw.clone(),
        )
        .offsetting(w.name.clone()),
        TensorContract::new(
            crate::input_scales_name(&w.name),
            input_expr,
            Vec::new(),
            f32_raw,
        ),
    ])
}

/// That `name` is one stored f32.
fn one_f32(src: &ztensor::Source, name: &str, of: &str) -> Result<(), Error> {
    let stored = stored_encoding(src, name)?;
    let count: u64 = src
        .get(name)
        .map_or(0, |tensor| tensor.shape().iter().product());
    if stored != Encoding::Raw(DType::F32) || count != 1 {
        return Err(Error::Illegible {
            name: of.to_string(),
            detail: format!("`{name}` is stored {stored:?} ({count} values), and it is one f32"),
        });
    }
    Ok(())
}

fn tiled_planes(
    src: &ztensor::Source,
    w: &Weight,
    parts: Vec<String>,
) -> Result<Vec<TensorContract>, Error> {
    if w.dtype != Dtype::U4g64tiled {
        return Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!(
                "it is declared {:?} and this verb states a repack; a relabelled plane \
                 is declared `U4g64tiled`, because the declaration is what says which \
                 order the artifact holds",
                w.dtype
            ),
        });
    }
    let axis = if parts.len() > 1 { pack_axis(w) } else { 0 };
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    let mut legs = Vec::new();
    for part in &parts {
        let stem = part
            .strip_suffix(".weight")
            .ok_or_else(|| Error::Illegible {
                name: w.name.clone(),
                detail: format!(
                    "`{part}` holds MLX affine codes, whose scales and biases \
                     are named beside a `.weight`, and it does not end in one"
                ),
            })?;
        let unpacked = unpacked_extents(src, w, part)?;
        legs.push(unpacked.clone());
        codes.push(Expr::src(part.clone()).transmute(TensorType::new(unpacked, grouped(w))));
        scales.push(crate::scales_name(stem));
        biases.push(crate::biases_name(stem));
    }
    holds_the_declared_rectangle(w, axis, &legs)?;
    let pairing = scaling(w);
    let flat = extents(w);
    let landed = banded_rows(w, flat.clone());
    let counted = divided(&landed, pairing.channel_axis, pairing.group_size, &w.name);
    let factor_target = |shape: Vec<i64>| TensorType::new(shape, encoding(Dtype::Bf16));
    Ok(vec![
        TensorContract::inferred(
            w.name.clone(),
            joined(axis, codes).repack(
                RepackLayout::TiledAffineU4Weight,
                TensorType::new(landed, grouped(w)),
            ),
            grouped(w),
        ),
        relaid(
            src,
            crate::scales_name(&w.name),
            &scales,
            counted.clone(),
            axis,
            factor_target(counted.clone()),
        )?
        .scaling(pairing),
        relaid(
            src,
            crate::biases_name(&w.name),
            &biases,
            counted.clone(),
            axis,
            factor_target(counted),
        )?
        .offsetting(w.name.clone()),
    ])
}

fn relaid(
    src: &ztensor::Source,
    name: String,
    legs: &[String],
    shape: Vec<i64>,
    axis: u8,
    to: TensorType,
) -> Result<TensorContract, Error> {
    let plane = factors(src, name.clone(), legs, shape.clone(), axis)?;
    Ok(TensorContract::new(
        name,
        plane.expr.repack(RepackLayout::TiledAffineFactor, to),
        shape,
        encoding(Dtype::Bf16),
    ))
}

fn factors(
    src: &ztensor::Source,
    name: String,
    legs: &[String],
    shape: Vec<i64>,
    axis: u8,
) -> Result<TensorContract, Error> {
    let expr = joined(axis, legs.iter().cloned().map(Expr::src).collect());
    let stored = agreed(src, &name, &expr)?;
    let want = encoding(Dtype::Bf16);
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

fn unpacked_extents(src: &ztensor::Source, w: &Weight, name: &str) -> Result<Vec<i64>, Error> {
    let Some(tensor) = src.get(name) else {
        return Err(Error::Missing(name.to_string()));
    };
    let stored = stored_encoding(src, name)?;
    if stored != Encoding::Raw(DType::U32) {
        return Err(Error::Illegible {
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
        return Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!("`{name}` is a scalar and a bank has a contracted axis"),
        });
    };
    *words *= word_codes(w.dtype);
    Ok(dims)
}

fn holds_the_declared_rectangle(w: &Weight, axis: u8, legs: &[Vec<i64>]) -> Result<(), Error> {
    let declared = extents(w);
    let refuse = |detail: String| Error::Illegible {
        name: w.name.clone(),
        detail,
    };
    let Some(first) = legs.first() else {
        return Ok(());
    };
    let mut joined = first.clone();
    let at = axis as usize;
    for leg in &legs[1..] {
        if leg.len() != joined.len() {
            return Err(refuse(format!(
                "its stored parts are rank {} and rank {}, and parts that join \
                 into one bank have one rank",
                joined.len(),
                leg.len(),
            )));
        }
        for (i, (into, part)) in joined.iter_mut().zip(leg).enumerate() {
            if i == at {
                *into += *part;
            } else if *into != *part {
                return Err(refuse(format!(
                    "its stored parts differ at axis {i} ({into} against \
                     {part}), which is not the axis {at} they join on, so they \
                     are not two halves of one rectangle"
                )));
            }
        }
    }
    if joined != declared {
        return Err(refuse(format!(
            "the file stores it {joined:?} and this text declares it \
             {declared:?}; a text reads the widths it states, so this row is \
             not the one that reads this checkpoint"
        )));
    }
    Ok(())
}

fn interned(
    src: &ztensor::Source,
    of: &str,
    shape: &[i64],
    bands: Option<Bands>,
    pairing: Scales,
) -> Result<Vec<TensorContract>, Error> {
    seams_clear_the_blocked_axis(of, bands.as_ref(), pairing.channel_axis, pairing.group_size);
    if pairing.form == ScaleForm::F32Tensor {
        // One scale for the whole tensor, which every rank holds whole.
        let name = crate::scales_name(of);
        return Ok(vec![
            TensorContract::new(
                name.clone(),
                Expr::src(name),
                Vec::new(),
                encoding(Dtype::F32),
            )
            .scaling(pairing),
        ]);
    }
    let declared = divided(shape, pairing.channel_axis, pairing.group_size, of);
    // A scale plane is stored beside its codes, a block of them each.
    let stored = stored_shape(src, of).filter(|codes| {
        codes
            .get(pairing.channel_axis as usize)
            .is_some_and(|extent| extent % i64::from(pairing.group_size) == 0)
    });
    let stored = stored.map(|codes| divided(&codes, pairing.channel_axis, pairing.group_size, of));
    let plane = |name: String, dtype: Dtype, shape: Vec<i64>| {
        let expr = padded(
            &name,
            bands.as_ref(),
            &shape,
            stored.as_deref(),
            &encoding(dtype),
        );
        TensorContract::new(name, expr, shape, encoding(dtype))
    };
    Ok(match pairing.form {
        ScaleForm::RawE8M0 => {
            vec![plane(crate::scales_name(of), Dtype::E8m0, declared).scaling(pairing)]
        }
        ScaleForm::Bf16AffineFactors => vec![
            plane(crate::scales_name(of), Dtype::Bf16, declared.clone()).scaling(pairing),
            plane(crate::biases_name(of), Dtype::Bf16, declared).offsetting(of),
        ],
        ScaleForm::E4m3TensorScaled => {
            // The tensor and input scales stop short of the contracted axis,
            // so a cut there holds them whole; a cut before it bands them.
            let lead = &shape[..shape.len() - 1];
            let short = |name: String, shape: &[i64]| {
                let expr = match &bands {
                    Some((axis, _)) if (*axis as usize) < shape.len() => {
                        banded(&name, bands.as_ref())
                    }
                    _ => Expr::src(name.clone()),
                };
                TensorContract::new(name, expr, shape.to_vec(), encoding(Dtype::F32))
            };
            let input = crate::input_scales_name(of);
            if src.get(&input).is_none() {
                return Err(Error::Missing(input));
            }
            vec![
                plane(crate::scales_name(of), Dtype::E4m3, declared).scaling(pairing),
                short(crate::global_scales_name(of), lead).offsetting(of),
                short(input, &lead[..lead.len().saturating_sub(1)]),
            ]
        }
        other => panic!(
            "`{of}` pairs its codes with {other:?} scales, and no family here \
             declares a bank in those terms"
        ),
    })
}

fn agreed(src: &ztensor::Source, name: &str, expr: &Expr) -> Result<Encoding, Error> {
    let mut read: Vec<(&str, Encoding)> = Vec::new();
    for source in expr.sources() {
        let stored = stored_encoding(src, source)?;
        read.push((source, stored));
    }
    let Some((whose, first)) = read.first() else {
        return Err(Error::Illegible {
            name: name.to_string(),
            detail: "it is built from no checkpoint tensor at all".to_string(),
        });
    };
    for (source, seen) in &read {
        if seen != first {
            return Err(Error::Illegible {
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

pub(crate) fn stored_encoding(src: &ztensor::Source, name: &str) -> Result<Encoding, Error> {
    let Some(tensor) = src.get(name) else {
        return Err(Error::Missing(name.to_string()));
    };
    let illegible = |why: &dyn std::fmt::Display| Error::Illegible {
        name: name.to_string(),
        detail: why.to_string(),
    };
    checkpoint::file::encoding_of(&tensor).map_err(|why| illegible(&why))
}

fn ladder(name: &str, expr: Expr, stored: &Encoding, want: &Encoding) -> Result<Expr, Error> {
    match (stored, want) {
        (s, w) if s == w => Ok(expr),
        (Encoding::Raw(_), Encoding::Raw(_)) => Ok(expr.cast(want.clone())),
        (Encoding::Raw(DType::U32), Encoding::Quant(_)) => Err(Error::Illegible {
            name: name.to_string(),
            detail: "it is stored as raw u32 words and this model wants it \
                     quantized; a checkpoint that already ships packed codes is \
                     read by naming its planes, not by encoding its words"
                .to_string(),
        }),
        (Encoding::Raw(_), Encoding::Quant(_)) => Ok(expr.cast(want.clone())),
        (Encoding::Quant(_), Encoding::Raw(_)) => Ok(expr.cast(want.clone())),
        (Encoding::Quant(_), Encoding::Quant(_)) => Err(Error::Incompatible {
            name: name.to_string(),
            stored: stored.clone(),
            want: want.clone(),
        }),
    }
}

/// Each segment of `w`'s cut as the checkpoint holds it whole, and how many
/// ranks in a row read the same share of it.
fn banding(w: &Weight, tp: u32) -> Option<Bands> {
    match &w.shard {
        // A weight whose cut was left to the compiler and reaches the
        // import without one is read whole, as one stated whole is.
        Shard::Replicated | Shard::Inferred => None,
        Shard::Cut {
            axis,
            segments,
            heads,
        } => Some((
            *axis,
            segments
                .iter()
                .enumerate()
                .map(|(i, segment)| {
                    let parts = cut_parts(w, heads.get(i).copied(), tp);
                    (leg_extent(*segment, parts, &w.name), tp / parts)
                })
                .collect(),
        )),
    }
}

/// The ways `tp` ranks cut a segment of `heads` heads.
fn cut_parts(w: &Weight, heads: Option<u64>, tp: u32) -> u32 {
    let parts =
        Shard::parts(heads, u64::from(tp)).unwrap_or_else(|why| panic!("`{}`: {why}", w.name));
    u32::try_from(parts).expect("no more parts than ranks")
}

fn banded(name: &str, bands: Option<&Bands>) -> Expr {
    let Some((axis, extents)) = bands else {
        return Expr::src(name);
    };
    let at = as_axis(*axis, name);
    match extents.as_slice() {
        [] => panic!("`{name}` is cut at no seam at all"),
        [(_, replicas)] => Expr::src(name).shard_among(at, *replicas),
        many => {
            let mut start = 0;
            let legs = many
                .iter()
                .map(|(extent, replicas)| {
                    let leg = Expr::src(name)
                        .slice(at, start, *extent)
                        .shard_among(at, *replicas);
                    start += *extent;
                    leg
                })
                .collect();
            Expr::concat(at, legs)
        }
    }
}

/// [`banded`], each segment zero-padded at its end when the file stores
/// `name` (as `stored`, encoded `held`) narrower along the cut than the plan
/// holds it `whole`: a width whose blocks the ranks cannot split, which the
/// model pads with rows and columns that contract to nothing, read out of an
/// artifact imported whole.
fn padded(
    name: &str,
    bands: Option<&Bands>,
    whole: &[i64],
    stored: Option<&[i64]>,
    held: &Encoding,
) -> Expr {
    let read = banded(name, bands);
    let (Some((axis, extents)), Some(stored)) = (bands, stored) else {
        return read;
    };
    let at = *axis as usize;
    let narrower = stored.len() == whole.len()
        && at < whole.len()
        && stored[at] < whole[at]
        && (0..whole.len()).all(|i| i == at || stored[i] == whole[i]);
    if !narrower {
        return read;
    }
    let legs = extents.len() as i64;
    let short = whole[at] - stored[at];
    let pad = short / legs;
    // A cut of one segment is the whole axis, in the units this plane counts
    // it in; the seams of many clear any blocked axis.
    let seg = |extent: i64| if legs == 1 { whole[at] } else { extent };
    // Each segment padded alike, by less than it stores; any other shortfall
    // is a plane this plan does not read, for the load to refuse.
    if short % legs != 0 || extents.iter().any(|(extent, _)| seg(*extent) <= 2 * pad) {
        return read;
    }
    let mut zeros = whole.to_vec();
    zeros[at] = pad;
    // A zero MXFP4 code is zero under any scale, and the scales beside it
    // are zeroed as bytes.
    let zeros = match held {
        Encoding::Raw(DType::E8m0) => Expr::fill(
            0.0,
            TensorType::new(zeros.clone(), Encoding::Raw(DType::U8)),
        )
        .transmute(TensorType::new(zeros, held.clone())),
        held => Expr::fill(0.0, TensorType::new(zeros, held.clone())),
    };
    let cut = as_axis(*axis, name);
    let mut start = 0;
    let mut parts = Vec::with_capacity(extents.len());
    for (extent, replicas) in extents {
        let kept = seg(*extent) - pad;
        let leg = match legs {
            1 => Expr::src(name),
            _ => Expr::src(name).slice(cut, start, kept),
        };
        start += kept;
        parts.push(Expr::concat(cut, vec![leg, zeros.clone()]).shard_among(cut, *replicas));
    }
    match legs {
        1 => parts.remove(0),
        _ => Expr::concat(cut, parts),
    }
}

/// The shape the file stores `name` in, if it holds it.
fn stored_shape(src: &ztensor::Source, name: &str) -> Option<Vec<i64>> {
    let tensor = src.get(name)?;
    Some(
        tensor
            .shape()
            .iter()
            .map(|extent| i64::try_from(*extent).expect("an extent no i64 holds"))
            .collect(),
    )
}

pub fn extents(w: &Weight) -> Vec<i64> {
    w.shape
        .iter()
        .map(|extent| i64::try_from(*extent).expect("an extent no i64 holds"))
        .collect()
}

/// `w`'s shape as the checkpoint holds it whole.
fn whole(w: &Weight, tp: u32) -> Vec<i64> {
    let mut dims = extents(w);
    if let Some((axis, bands)) = banding(w, tp) {
        let at = axis as usize;
        let seams: u64 = match &w.shard {
            Shard::Cut { segments, .. } => segments.iter().sum(),
            Shard::Replicated | Shard::Inferred => unreachable!("a band is a cut's"),
        };
        let dim = dims
            .get_mut(at)
            .unwrap_or_else(|| panic!("`{}` is {:?} and its cut names axis {at}", w.name, w.shape));
        assert_eq!(
            u64::try_from(*dim).expect("an extent no u64 holds"),
            seams,
            "`{}`: its segments sum to {seams} and its axis {at} is {dim}",
            w.name,
        );
        *dim = bands.iter().map(|(extent, _)| extent).sum();
    }
    dims
}

pub fn divided(shape: &[i64], axis: u32, group: u32, name: &str) -> Vec<i64> {
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

fn seams_clear_the_blocked_axis(name: &str, bands: Option<&Bands>, channel: u32, group: u32) {
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

pub fn scaling(w: &Weight) -> Scales {
    let form = match w.dtype {
        Dtype::Mxfp4 => ScaleForm::RawE8M0,
        Dtype::Nvfp4 => ScaleForm::E4m3TensorScaled,
        Dtype::E4m3tensor => ScaleForm::F32Tensor,
        // A block-FP8 release's f32 factors, one a 128x128 block.
        Dtype::E4m3tile128 | Dtype::E4m3g128 => ScaleForm::F32Factors,
        Dtype::U4g64
        | Dtype::U8g64
        | Dtype::U4g32
        | Dtype::U4g64tiled
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128 => ScaleForm::Bf16AffineFactors,
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

pub fn grouped(w: &Weight) -> Encoding {
    match encoding(w.dtype) {
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
        Shard::Replicated | Shard::Inferred => {
            panic!("`{}` is replicated and has no cut axis", w.name)
        }
        Shard::Cut { axis, .. } => as_axis(*axis, &w.name),
    }
}

fn leg_extent(segment: u64, parts: u32, name: &str) -> i64 {
    let whole = segment.checked_mul(u64::from(parts)).unwrap_or_else(|| {
        panic!("`{name}`: a segment of {segment} is not {parts} times anything")
    });
    i64::try_from(whole).expect("an extent no i64 holds")
}

fn as_axis(axis: u32, name: &str) -> u8 {
    u8::try_from(axis)
        .unwrap_or_else(|_| panic!("`{name}` is cut on axis {axis}, which is no axis"))
}

pub fn encoding(dtype: Dtype) -> Encoding {
    match dtype {
        Dtype::Bf16 => Encoding::Raw(DType::Bf16),
        Dtype::F16 => Encoding::Raw(DType::F16),
        Dtype::F32 => Encoding::Raw(DType::F32),
        Dtype::I32 => Encoding::Raw(DType::I32),
        Dtype::U32 => Encoding::Raw(DType::U32),
        Dtype::U8 => Encoding::Raw(DType::U8),
        Dtype::I8 => Encoding::Raw(DType::I8),
        Dtype::E4m3 => Encoding::Raw(DType::E4m3),
        Dtype::E8m0 => Encoding::Raw(DType::E8m0),
        Dtype::E5m2 => Encoding::Raw(DType::E5m2),
        Dtype::I64 => Encoding::Raw(DType::I64),
        Dtype::I16 => Encoding::Raw(DType::I16),
        Dtype::U64 => Encoding::Raw(DType::U64),
        Dtype::U16 => Encoding::Raw(DType::U16),
        Dtype::Bool => Encoding::Raw(DType::Bool),
        Dtype::Mxfp4 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: None,
        }),
        Dtype::U4g64 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 64,
            channel_axis: None,
        }),
        Dtype::U8g64 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 8,
            group_size: 64,
            channel_axis: None,
        }),
        Dtype::U4g32 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: None,
        }),
        Dtype::U4g64tiled => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 64,
            channel_axis: None,
        }),
        Dtype::U2g32 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 2,
            group_size: 32,
            channel_axis: None,
        }),
        Dtype::U2g64 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 2,
            group_size: 64,
            channel_axis: None,
        }),
        Dtype::U2g128 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 2,
            group_size: 128,
            channel_axis: None,
        }),
        Dtype::E2m1 | Dtype::KvU4 => panic!(
            "this dtype names a kv-page quantization scheme, not a stored \
             weight plane; no load contract declares one"
        ),
        Dtype::Nvfp4 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::Nvfp4E2M1E4M3,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 16,
            channel_axis: None,
        }),
        Dtype::E4m3tensor => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::Fp8E4M3Tensor,
            logical_dtype: DType::Bf16,
            bits_per_element: 8,
            group_size: 1,
            channel_axis: None,
        }),
        Dtype::E4m3row | Dtype::E4m3tile128 | Dtype::E4m3g128 => panic!(
            "a {:?} weight is served but no load contract declares one",
            dtype
        ),
        Dtype::Fp8b128 | Dtype::Fp8b32 | Dtype::Mxfp8 => panic!(
            "a block-fp8 weight is encoded plane by plane, its raw codes and exponents \
             each their own"
        ),
        // PTQ1_0 joins the self-contained block family: its `g128_t3_f16_n` term
        // resolves to `QuantScheme::Ptq1_0` (ternary, one inline fp16 scale).
        Dtype::U2g16k
        | Dtype::I3g16k
        | Dtype::U4g32k
        | Dtype::U5g32k
        | Dtype::I6g16k
        | Dtype::Ptq1_0 => {
            Encoding::Quant(checkpoint::spec_of_term(dtype.repr()).unwrap_or_else(|| {
                panic!(
                    "`{dtype}` is not the arithmetic of any self-contained scheme this \
                     checkpoint vocabulary knows; a term whose factors live in \
                     companion planes is declared by the variant that names them"
                )
            }))
        }
    }
}

/// The tensor the layout states `w` holds: its values at the dtype it
/// computes in, cast to the encoding it is stored in.
pub fn stated(w: &Weight) -> Result<TensorContract, Error> {
    let values = w.values.as_deref().unwrap_or_default();
    let Encoding::Raw(raw) = encoding(w.compute_dtype()) else {
        return Err(Error::Illegible {
            name: w.name.clone(),
            detail: "a stated weight computes in no raw dtype".to_string(),
        });
    };
    let shape: Vec<i64> = w.shape.iter().map(|&n| n as i64).collect();
    let mut expr = constant(&w.name, values, shape.clone(), raw)?;
    let want = encoding(w.dtype);
    if want != Encoding::Raw(raw) {
        expr = expr.cast_chunked(want.clone(), 0);
    }
    Ok(TensorContract::new(w.name.clone(), expr, shape, want))
}

/// The tensor of `values` in `shape`, stored as raw `dtype`: a table a model
/// states rather than reads, carried by the contract itself. Each value is
/// rounded to `dtype` as the plan's own float transforms round theirs.
pub fn constant(
    name: &str,
    values: &[f32],
    shape: Vec<i64>,
    dtype: checkpoint::types::DType,
) -> Result<Expr, Error> {
    let wide: Vec<f64> = values.iter().map(|v| f64::from(*v)).collect();
    let bytes =
        checkpoint::codec::cast::encode_values(&wide, dtype).map_err(|why| Error::Illegible {
            name: name.to_string(),
            detail: format!("a constant stated as {dtype:?}: {why}"),
        })?;
    Ok(Expr::constant(TensorType::raw(shape, dtype), bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Weight;
    use checkpoint::types::Visibility;

    const ROWS: u64 = 32;
    const K: u64 = 128;

    fn raw_source(dir: &std::path::Path) -> ztensor::Source {
        let path = dir.join("dense.zt");
        let mut writer = ztensor::Writer::create(&path).expect("the container opens");
        let values = vec![0u8; (ROWS * K * 2) as usize];
        writer
            .add("proj", vec![ROWS, K], ztensor::Leaf::BF16, &values)
            .expect("the plane lands");
        writer.finish().expect("the container closes");
        ztensor::Source::open(&path).expect("it reads back")
    }

    #[test]
    fn lib_every_case() {
        a_placed_row_lands_the_placement_over_the_codes_an_encode_wrote();
        a_placed_slice_lands_the_placement_too();
        an_uncovered_placement_still_refuses_by_name();
        the_unplaced_sibling_of_that_row_still_reads_it();
        the_mlx_road_lands_the_placed_order_and_bands_its_rows();
        a_dense_row_is_untouched_by_the_guard();
        modelopt_nvfp4_legs_land_their_codes_scales_and_a_tensor_scale_a_row();
        deepseek_fp8_legs_land_their_codes_and_block_exponents_as_stored();
        a_gather_over_k_quants_computes_over_their_decoded_values();
        an_expert_rank_loads_its_slots_of_a_bank();
    }

    /// A bank of four experts over two ranks of three slots: rank 1 holds
    /// experts 3, 0 and 1 (the placement wraps), bytes as stored.
    fn an_expert_rank_loads_its_slots_of_a_bank() {
        use checkpoint::executor::Execution;
        use checkpoint::executor::sink::MemorySink;
        use checkpoint::plan::{StorageTarget, compile};

        let dir = tempfile::tempdir().expect("a scratch directory");
        let mut writer =
            ztensor::Writer::create(dir.path().join("model.zt")).expect("the container opens");
        // Expert `e`'s four bf16 values are each `e + 1` in their low byte.
        let bytes: Vec<u8> = (0..4u8).flat_map(|e| [e + 1, 0].repeat(4)).collect();
        writer
            .add("bank", vec![4, 2, 2], ztensor::Leaf::BF16, &bytes)
            .expect("the bank lands");
        writer.finish().expect("the container closes");
        let src = ztensor::Source::open(dir.path().join("model.zt")).expect("it reads back");
        let param = crate::Param {
            name: "bank".into(),
            shape: vec![3, 2, 2],
            shard: crate::Shard::Replicated,
            dtype: Dtype::Bf16,
            source: crate::ParamSource::Checkpoint,
            layout: poem_ir::ParamLayout::Natural,
            placement: poem_ir::ParamPlacement::Device,
            experts: Some(poem_ir::ExpertSlots {
                layer: 0,
                ep: 2,
                experts: 4,
                slots: 3,
            }),
        };
        let contract =
            own_contract(&src, &[param], 1, Platform::Cuda).expect("the bank's contract");
        let metadata = checkpoint::file::read::parse_metadata(dir.path()).expect("it parses");
        for (rank, held) in [(0u32, [1u8, 2, 3]), (1, [4, 1, 2])] {
            let target = StorageTarget {
                ep_rank: rank,
                ..StorageTarget::default()
            };
            let plan = compile(&metadata, &contract, target).expect("it compiles");
            let mut sink = MemorySink::default();
            Execution::new(&plan, dir.path())
                .sink(&mut sink)
                .run()
                .expect("the plan executes");
            let want: Vec<u8> = held.iter().flat_map(|e| [*e, 0].repeat(4)).collect();
            assert_eq!(sink.tensors["bank"], want, "rank {rank}");
        }
    }

    fn deepseek_fp8_legs_land_their_codes_and_block_exponents_as_stored() {
        use checkpoint::executor::Execution;
        use checkpoint::executor::sink::MemorySink;
        use checkpoint::plan::{StorageTarget, compile};

        const LEG: u64 = 128;
        const WIDTH: u64 = 256;
        let dir = tempfile::tempdir().expect("a scratch directory");
        let mut writer =
            ztensor::Writer::create(dir.path().join("model.zt")).expect("the container opens");
        for (leg, fill) in [("w1", 1u8), ("w3", 2)] {
            writer
                .add(
                    format!("ffn.{leg}.scale"),
                    vec![1, 2],
                    ztensor::Leaf::E8M0,
                    &[120 + fill, 124 + fill],
                )
                .expect("exponents land");
            writer
                .add(
                    format!("ffn.{leg}.weight"),
                    vec![LEG, WIDTH],
                    ztensor::Leaf::E4M3,
                    &vec![fill; (LEG * WIDTH) as usize],
                )
                .expect("codes land");
        }
        writer.finish().expect("the container closes");
        let src = ztensor::Source::open(dir.path().join("model.zt")).expect("it reads back");

        let w = Weight::sym("gate_up", [2 * LEG, WIDTH], Dtype::Fp8b128).packed([LEG, LEG]);
        let parts = ["ffn.w1.weight", "ffn.w3.weight"].map(String::from);
        let read = planes_fused(&src, &w, parts).expect("the legs read as one fp8b128 weight");
        let names: Vec<&str> = read.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["gate_up", "gate_up.scales"]);
        let contract = ModelContract {
            alignment: ALIGNMENT,
            tensors: read,
            groups: Vec::new(),
        };
        let metadata = checkpoint::file::read::parse_metadata(dir.path()).expect("it parses");
        let plan = compile(&metadata, &contract, StorageTarget::default()).expect("it compiles");
        let mut sink = MemorySink::default();
        Execution::new(&plan, dir.path())
            .sink(&mut sink)
            .run()
            .expect("the plan executes");
        let codes = &sink.tensors["gate_up"];
        let half = (LEG * WIDTH) as usize;
        assert_eq!(codes.len(), 2 * half);
        assert!(codes[..half].iter().all(|b| *b == 1) && codes[half..].iter().all(|b| *b == 2));
        assert_eq!(sink.tensors["gate_up.scales"], [121, 125, 122, 126]);

        let short = Weight::sym("gate_up", [2 * LEG, 2 * WIDTH], Dtype::Fp8b128).packed([LEG, LEG]);
        let parts = ["ffn.w1.weight", "ffn.w3.weight"].map(String::from);
        assert!(
            planes_fused(&src, &short, parts).is_err(),
            "legs narrower than the declared weight are refused"
        );
    }

    fn modelopt_nvfp4_legs_land_their_codes_scales_and_a_tensor_scale_a_row() {
        use checkpoint::executor::Execution;
        use checkpoint::executor::sink::MemorySink;
        use checkpoint::plan::{StorageTarget, compile};

        const LEG: u64 = 4;
        const WIDTH: u64 = 64;
        let dir = tempfile::tempdir().expect("a scratch directory");
        let mut writer =
            ztensor::Writer::create(dir.path().join("model.zt")).expect("the container opens");
        let legs = [("gate", 0.5f32, 0.25f32, 1u8), ("up", 2.0, 0.75, 2)];
        for (leg, tensor, input, fill) in legs {
            let at = |plane: &str| format!("mlp.{leg}_proj.{plane}");
            writer
                .add(
                    at("input_scale"),
                    vec![],
                    ztensor::Leaf::F32,
                    &input.to_le_bytes(),
                )
                .expect("a scalar lands");
            writer
                .add(
                    at("weight"),
                    vec![LEG, WIDTH / 2],
                    ztensor::Leaf::U8,
                    &vec![fill; (LEG * WIDTH / 2) as usize],
                )
                .expect("codes land");
            writer
                .add(
                    at("weight_scale"),
                    vec![LEG, WIDTH / 16],
                    ztensor::Leaf::E4M3,
                    &vec![0x38 + fill; (LEG * WIDTH / 16) as usize],
                )
                .expect("scales land");
            writer
                .add(
                    at("weight_scale_2"),
                    vec![],
                    ztensor::Leaf::F32,
                    &tensor.to_le_bytes(),
                )
                .expect("a scalar lands");
        }
        writer.finish().expect("the container closes");
        let src = ztensor::Source::open(dir.path().join("model.zt")).expect("it reads back");

        let w = Weight::sym("proj", [2 * LEG, WIDTH], Dtype::Nvfp4).packed([LEG, LEG]);
        let parts = ["mlp.gate_proj.weight", "mlp.up_proj.weight"].map(String::from);
        let read = planes_fused(&src, &w, parts).expect("the legs read as one nvfp4 weight");
        let names: Vec<&str> = read.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "proj",
                "proj.scales",
                "proj.global_scales",
                "proj.input_scales"
            ]
        );
        assert_eq!(
            read[1].scales.as_ref().map(|s| s.form),
            Some(ScaleForm::E4m3TensorScaled)
        );
        assert_eq!(read[2].zero_points.as_deref(), Some("proj"));

        let contract = ModelContract {
            alignment: ALIGNMENT,
            tensors: read,
            groups: Vec::new(),
        };
        let metadata = checkpoint::file::read::parse_metadata(dir.path()).expect("it parses");
        let plan = compile(&metadata, &contract, StorageTarget::default()).expect("it compiles");
        let attached = &plan.attachments[0];
        assert_eq!(attached.scale_form, ScaleForm::E4m3TensorScaled);
        assert!(
            attached.zero_point_tensor.is_some(),
            "the tensor scales ride as the codes' second companion"
        );
        let mut sink = MemorySink::default();
        Execution::new(&plan, dir.path())
            .sink(&mut sink)
            .run()
            .expect("the plan executes");
        let f32s = |name: &str| -> Vec<f32> {
            sink.tensors[name]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let rows = |value: f32| vec![value; LEG as usize];
        assert_eq!(f32s("proj.global_scales"), [rows(0.5), rows(2.0)].concat());
        assert_eq!(
            f32s("proj.input_scales"),
            [0.75],
            "the larger of the legs' input scales"
        );
        let codes = &sink.tensors["proj"];
        let half = (LEG * WIDTH / 2) as usize;
        assert!(codes[..half].iter().all(|b| *b == 1) && codes[half..].iter().all(|b| *b == 2));
        let scales = &sink.tensors["proj.scales"];
        assert_eq!(scales.len(), (2 * LEG * WIDTH / 16) as usize);
        assert_eq!((scales[0], scales[scales.len() - 1]), (0x39, 0x3a));
    }

    /// A container of K-quant blocks, `(name, [rows, cols], scheme)`.
    fn blocks(dir: &std::path::Path, tensors: &[(&str, [i64; 2], QuantScheme)]) -> ztensor::Source {
        use checkpoint::types::{TensorDecl, TensorId};
        let path = dir.join("blocks.zt");
        let mut writer = checkpoint::file::write::Writer::create(&path, &Default::default())
            .expect("the container opens");
        for (name, shape, scheme) in tensors {
            let decl = TensorDecl {
                id: TensorId(0),
                name: name.to_string(),
                shape: shape.to_vec(),
                encoding: Encoding::Quant(QuantSpec {
                    scheme: *scheme,
                    logical_dtype: DType::Bf16,
                    bits_per_element: 0,
                    group_size: 0,
                    channel_axis: None,
                }),
                alignment: 64,
                visibility: Visibility::default(),
            };
            let block = match scheme {
                QuantScheme::GgufQ4K => 144,
                _ => 210,
            };
            let bytes = vec![0u8; (shape[0] * shape[1] / 256) as usize * block];
            writer.add_tensor(&decl, &bytes).expect("the blocks land");
        }
        writer.finish().expect("the container closes");
        ztensor::Source::open(&path).expect("it reads back")
    }

    /// llama.cpp's qwen35 GGUFs tile the GDN v-heads; un-tiling `out_proj`
    /// gathers 128-wide column spans, which cut a K-quant's 256-wide blocks,
    /// and its legs are of two schemes.
    fn a_gather_over_k_quants_computes_over_their_decoded_values() {
        const COLS: u64 = 256;
        let dir = tempfile::tempdir().expect("a scratch directory");
        let shape = [ROWS as i64, COLS as i64];
        let src = blocks(
            dir.path(),
            &[
                ("q4", shape, QuantScheme::GgufQ4K),
                ("q6", shape, QuantScheme::GgufQ6K),
            ],
        );
        let halves: Vec<i64> = (128..256).chain(0..128).collect();
        let expr = Expr::concat(
            0,
            vec![
                Expr::src("q6").gather(1, halves.clone()),
                Expr::src("q4").gather(1, halves),
            ],
        );
        let w = Weight::sym("out_proj", [2 * ROWS, COLS], Dtype::U4g64);
        let landed = computed(&src, &w, expr.clone()).expect("the blocks are read as values");
        let names: Vec<&str> = landed.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "out_proj.block0",
                "out_proj.block1",
                "out_proj.gathered0",
                "out_proj.gathered1",
                "out_proj.values",
                "out_proj"
            ],
            "each leg decodes once and is gathered as its own tensor"
        );
        assert!(
            matches!(&landed[5].expr, Expr::Cast { src, .. } if **src == Expr::out("out_proj.values")),
            "the values are encoded once: {:?}",
            landed[5].expr
        );
        let metadata = checkpoint::file::read::parse_metadata(&dir.path().join("blocks.zt"))
            .expect("the container's metadata reads");
        let contract = ModelContract {
            alignment: 64,
            tensors: landed,
            groups: Vec::new(),
        };
        let target = checkpoint::plan::StorageTarget {
            tile_map_mask: checkpoint::plan::CONVERT_TILE_MAP_MASK,
            ..Default::default()
        };
        let plan = checkpoint::plan::compile(&metadata, &contract, target)
            .expect("a gather over a decoded leg lands");
        assert!(
            plan.instrs.iter().any(|i| matches!(
                i,
                checkpoint::plan::StorageInstr::TileMap {
                    kind: checkpoint::plan::TileMapKind::Gather,
                    ..
                }
            )),
            "the column gather runs over the decoded buffer"
        );
        assert!(
            declare(&src, &w, expr).is_err(),
            "read as stored, two schemes are not one weight's representation"
        );
    }

    fn a_placed_row_lands_the_placement_over_the_codes_an_encode_wrote() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let src = raw_source(dir.path());
        let w = Weight::sym("proj", [ROWS, K], Dtype::U4g64tiled);
        let landed = planes(&src, &w, "proj").expect("the encode road states the placement");
        let public: Vec<&TensorContract> = landed
            .iter()
            .filter(|c| c.visibility == Visibility::Public)
            .collect();
        assert_eq!(
            public.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            vec!["proj", "proj.scales", "proj.biases"],
            "a placed bank publishes its three planes and nothing else"
        );
        assert!(
            matches!(
                &public[0].expr,
                Expr::Repack { layout, .. } if *layout == RepackLayout::TiledAffineU4Weight
            ),
            "the public codes must be the fragment repack, not the encode: {:?}",
            public[0].expr
        );
        assert!(
            landed
                .iter()
                .any(|c| c.name == "proj.canon" && c.visibility == Visibility::Internal),
            "the encode itself stays internal"
        );
    }

    fn a_placed_slice_lands_the_placement_too() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let src = raw_source(dir.path());
        let w = Weight::sym("proj", [ROWS, K / 2], Dtype::U4g64tiled);
        let landed = computed(
            &src,
            &w,
            Expr::src("proj").slice(1, K as i64 / 2, K as i64 / 2),
        )
        .expect("a slice of a raw plane encodes");
        let codes = landed
            .iter()
            .find(|c| c.name == "proj")
            .expect("the bank lands");
        assert!(
            matches!(&codes.expr, Expr::Repack { layout, .. } if *layout == RepackLayout::TiledAffineU4Weight),
            "{:?}",
            codes.expr
        );
    }

    fn an_uncovered_placement_still_refuses_by_name() {
        let w = Weight::sym("proj", [ROWS, K], Dtype::U4g64tiled);
        let why = placed_wants_its_own_order(&w, "a raw plane")
            .expect_err("a placed declaration this road cannot state is a refusal");
        let said = format!("{why}");
        assert!(
            said.contains("U4g64tiled") && said.contains("byte order"),
            "the refusal must name the placement and why it cannot be met: {said}"
        );
    }

    fn the_unplaced_sibling_of_that_row_still_reads_it() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let src = raw_source(dir.path());
        let w = Weight::sym("proj", [ROWS, K], Dtype::U4g64);
        planes(&src, &w, "proj").expect("the canonical order is what an encode publishes");
    }

    fn the_mlx_road_lands_the_placed_order_and_bands_its_rows() {
        const ODD: u64 = 24;
        let dir = tempfile::tempdir().expect("a scratch directory");
        let path = dir.path().join("affine.zt");
        let mut writer = ztensor::Writer::create(&path).expect("the container opens");
        let groups = (K / 64) as usize;
        for plane in ["proj.biases", "proj.scales"] {
            writer
                .add(
                    plane,
                    vec![ODD, groups as u64],
                    ztensor::Leaf::BF16,
                    &vec![0u8; ODD as usize * groups * 2],
                )
                .expect("factors land");
        }
        writer
            .add(
                "proj.weight",
                vec![ODD, K / 8],
                ztensor::Leaf::U32,
                &vec![0u8; (ODD * K / 8 * 4) as usize],
            )
            .expect("codes land");
        writer.finish().expect("the container closes");
        let src = ztensor::Source::open(&path).expect("it reads back");

        let w = Weight::sym("proj", [ODD, K], Dtype::U4g64tiled);
        let read = planes(&src, &w, "proj.weight").expect("MLX codes take the placed road");
        let codes = read
            .iter()
            .find(|t| t.name == "proj")
            .expect("the bank keeps its declared name");
        let checkpoint::contract::Expr::Repack { layout, to, .. } = &codes.expr else {
            panic!(
                "the placed road puts a Repack on the codes, got {:?}",
                codes.expr
            );
        };
        assert_eq!(*layout, RepackLayout::TiledAffineU4Weight);
        assert_eq!(
            to.shape,
            vec![32, K as i64],
            "24 rows band up to 32; no other road here rounds a row count"
        );
    }

    fn a_dense_row_is_untouched_by_the_guard() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let src = raw_source(dir.path());
        let w = Weight::sym("proj", [ROWS, K], Dtype::Bf16);
        planes(&src, &w, "proj").expect("bf16 in, bf16 out");
    }
}
