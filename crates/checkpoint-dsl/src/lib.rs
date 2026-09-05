//! The checkpoint-provenance authoring eDSL: a family's import calls
//! [`Builder`]'s verbs to produce a [`ModelContract`].

use checkpoint::contract::{Expr, ModelContract, Scales, TensorContract, TensorType};
use checkpoint::types::{
    Axis, DType, Encoding, QuantGranularity, QuantScheme, QuantSpec, RepackLayout, ScaleForm,
    TILED_BAND,
};
use model_dsl::{Dtype, Platform, Shard, Weight};

/// Why a read refused: the checkpoint lacks the name, states it in terms no
/// reader here can name, or holds it in a representation the declared one is
/// not decoded from.
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

/// One checkpoint being read into one [`ModelContract`]. Each `read*` line
/// states where one declared weight's bytes come from.
pub struct Builder<'a> {
    src: &'a ztensor::Source,
    tp: u32,
    platform: Platform,
    tensors: Vec<TensorContract>,
}

impl<'a> Builder<'a> {
    /// `tp` feeds [`read_own`](Builder::read_own) alone; the foreign verbs
    /// read a whole checkpoint and refuse `tp > 1`.
    #[must_use]
    pub fn new(src: &'a ztensor::Source, tp: u32, platform: Platform) -> Builder<'a> {
        Builder {
            src,
            tp,
            platform,
            tensors: Vec::new(),
        }
    }

    /// Read `w` from the checkpoint tensor named `from`: one stored tensor
    /// for bf16, the `.weight/.scales/.biases` triplet for an MLX affine bank,
    /// relabelled into fragment order for a bank placed `U4g64tiled`.
    pub fn read(&mut self, w: &Weight, from: impl Into<String>) -> Result<(), Error> {
        let w = &w.placed(self.platform);
        self.whole_checkpoint(w)?;
        let read = planes(self.src, w, from)?;
        self.tensors.extend(read);
        Ok(())
    }

    /// Read `w` from several checkpoint tensors, concatenated on the weight's
    /// own cut axis.
    pub fn read_concat(
        &mut self,
        w: &Weight,
        parts: impl IntoIterator<Item = String>,
    ) -> Result<(), Error> {
        let w = &w.placed(self.platform);
        self.whole_checkpoint(w)?;
        let read = planes_fused(self.src, w, parts)?;
        self.tensors.extend(read);
        Ok(())
    }

    /// Read `w` via a stated expression. Every source it names must exist and
    /// agree on one stored representation.
    /// One MLX affine bank stored a row at a time: `rows[i]` names the
    /// `.weight` planes that fuse along the cut axis into row `i` of axis 0.
    pub fn read_stack(
        &mut self,
        w: &Weight,
        rows: impl IntoIterator<Item = Vec<String>>,
    ) -> Result<(), Error> {
        let w = &w.placed(self.platform);
        self.whole_checkpoint(w)?;
        let read = affine_stacked(self.src, w, rows.into_iter().collect())?;
        self.tensors.extend(read);
        Ok(())
    }

    /// Read `w` from `from` with `over` applied to the plane AS STORED, then
    /// adapted to the encoding the row declares.
    ///
    /// Two things make this its own verb rather than a use of
    /// [`read_expr`](Builder::read_expr). The order is one: `read_expr`
    /// states its expression over the stored bytes and lets the ladder wrap
    /// a cast around the result, which buries the transform's kernel under
    /// that cast where nothing lowers it. The arithmetic is the other: a
    /// transform that undoes what a converter folded in — subtracting the
    /// one an RMSNorm's weight was stored with, say — is a small difference
    /// of near-equal numbers, and doing it after narrowing to the row's
    /// dtype cancels away the very residual it is recovering. So `over`
    /// applies at the checkpoint's own width, under its own internal
    /// tensor, and the adaptation happens above it.
    pub fn read_over(
        &mut self,
        w: &Weight,
        from: impl Into<String>,
        over: impl FnOnce(Expr) -> Expr,
    ) -> Result<(), Error> {
        let w = &w.placed(self.platform);
        self.whole_checkpoint(w)?;
        let from = from.into();
        let want = encoding(w.dtype);
        let stored = stored_encoding(self.src, &from)?;
        let applied = format!("{}.read", w.name);
        self.tensors.push(
            TensorContract::new(
                applied.clone(),
                over(Expr::src(from.clone())),
                extents(w),
                stored.clone(),
            )
            .internal(),
        );
        let adapted = ladder(&from, Expr::out(applied), &stored, &want)?;
        self.tensors.push(TensorContract::new(
            w.name.clone(),
            adapted,
            extents(w),
            want,
        ));
        Ok(())
    }

    pub fn read_expr(&mut self, w: &Weight, expr: Expr) -> Result<(), Error> {
        let w = &w.placed(self.platform);
        self.whole_checkpoint(w)?;
        let read = declare(self.src, w, expr)?;
        self.tensors.push(read);
        Ok(())
    }

    /// Read `w` from the tensor of its own name, banded into `tp` ranks when
    /// the weight is declared cut.
    pub fn read_own(&mut self, w: &Weight) -> Result<(), Error> {
        let w = &w.placed(self.platform);
        let read = resolve(self.src, claim(w, self.tp))?;
        self.tensors.extend(read);
        Ok(())
    }

    /// A contract this language cannot say, stated raw.
    pub fn push(&mut self, tensor: TensorContract) {
        self.tensors.push(tensor);
    }

    pub fn extend(&mut self, tensors: impl IntoIterator<Item = TensorContract>) {
        self.tensors.extend(tensors);
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

/// The load contract of a serving artifact: every checkpoint-sourced weight
/// of the trace read from the tensor of its own name. Companion planes come
/// with their codes.
pub fn own_contract(
    src: &ztensor::Source,
    params: &[model_dsl::Param],
    tp: u32,
    platform: Platform,
) -> Result<ModelContract, Error> {
    let companions: std::collections::BTreeSet<String> = params
        .iter()
        .filter(|param| matches!(claim_kind(param.dtype), Kind::Packed))
        .flat_map(|param| {
            [
                model_dsl::scales_name(&param.name),
                model_dsl::biases_name(&param.name),
            ]
        })
        .collect();
    let mut b = Builder::new(src, tp, platform);
    for param in params {
        if param.source != model_dsl::ParamSource::Checkpoint || companions.contains(&param.name) {
            continue;
        }
        b.read_own(&Weight::of_plane(param))?;
    }
    Ok(b.build())
}

enum Kind {
    Packed,
    Plain,
}

fn claim_kind(dtype: Dtype) -> Kind {
    match dtype {
        Dtype::Mxfp4
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

/// How many codes one group of a packed bank shares a scale with — a number
/// per scheme, not a single constant for the tree, since MLX's affine U4
/// groups a different count than mxfp4.
fn group_of(dtype: Dtype) -> u32 {
    match dtype {
        Dtype::Mxfp4 => 32,
        // Groups sixty-four codes, not sixty-four bytes.
        Dtype::U4g64 | Dtype::U8g64 | Dtype::U2g64 | Dtype::U4g64tiled => 64,
        Dtype::U4g32 | Dtype::U2g32 => 32,
        Dtype::U2g128 => 128,
        other => panic!("`{other:?}` blocks no axis; only a packed bank has groups"),
    }
}

/// How many affine codes the checkpoint packs into one `u32` word — MLX's own
/// packing, least-significant code first. Derived from the code's bit width,
/// not a fixed constant: a u32 holds eight four-bit codes but only four
/// eight-bit ones.
fn word_codes(dtype: Dtype) -> i64 {
    let bits = i64::try_from(dtype.bits()).expect("a code width inside i64");
    assert!(
        bits > 0 && 32 % bits == 0,
        "`{dtype:?}` is {bits} bits wide, which does not divide a 32-bit word"
    );
    32 / bits
}

const ALIGNMENT: u32 = 256;

struct Claim {
    pub name: String,
    pub shape: Vec<i64>,
    pub bands: Option<(u32, Vec<i64>)>,
    pub encoding: Encoding,
    pub scales: Option<Scales>,
}

#[must_use]
fn claim(w: &Weight, tp: u32) -> Claim {
    let (encoding, scales) = match w.dtype {
        Dtype::Mxfp4
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
        // Self-contained: factors live in the payload, no `.scales`/`.biases`.
        Dtype::U2g16k | Dtype::I3g16k | Dtype::U4g32k | Dtype::U5g32k | Dtype::I6g16k => {
            (encoding(w.dtype), None)
        }
        Dtype::E2m1 => panic!(
            "`Dtype::E2m1` names a kv-page quantization scheme, not a stored \
             weight plane; no load contract declares one"
        ),
        // Served, but no load contract declares one yet — the same
        // statement `model_dsl::Weight::planes` makes, one crate over.
        Dtype::Nvfp4 | Dtype::E4m3row | Dtype::E4m3tile128 => panic!(
            "a {:?} weight is served but no load contract declares one",
            w.dtype
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

/// A tiled affine weight claims the rectangle it was repacked into: output
/// columns rounded up to a whole mma band ([`TILED_BAND`]), matching the
/// padded shape the engine checks arriving tensors against.
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

/// One claim, checked against the source and stated as the one, two or
/// three tensors the checkpoint stores it as.
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
    let read = banded(&name, bands.as_ref());
    let expr = ladder(&name, read, &stored, &encoding)?;
    tensors.push(match &encoding {
        Encoding::Raw(_) => {
            TensorContract::new(name.clone(), expr, shape.clone(), encoding.clone())
        }
        Encoding::Quant(_) => TensorContract::inferred(name.clone(), expr, encoding.clone()),
    });
    match (&stored, scales) {
        // Checkpoint shipped both planes: say where they are.
        (Encoding::Quant(_), Some(pairing)) => {
            tensors.extend(interned(&name, &shape, bands, pairing));
        }
        // Raw stored, quantized wanted: the loader encodes on the way in and
        // publishes the scales plane itself; nothing more to declare here.
        (Encoding::Raw(_), Some(_)) => {}
        (Encoding::Quant(_), None) | (Encoding::Raw(_), None) => {}
    }
    Ok(tensors)
}
fn declare(src: &ztensor::Source, w: &Weight, expr: Expr) -> Result<TensorContract, Error> {
    // A quantized want states its blocked axis (the contracted, last one),
    // so a raw source encoded on the way in lands the grouping the bank's
    // own contract asks for.
    let want = match encoding(w.dtype) {
        Encoding::Quant(_) => grouped(w),
        raw => raw,
    };
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

/// That a quantized source holds as many values as the row declares.
///
/// A quantized plane's contract is `inferred` — the shape comes from the file,
/// because a packed bank's stored extents are the authority on its own
/// blocking. That leaves nothing comparing the file against the model, so a
/// row of another width reads a checkpoint it has no business claiming, and
/// identification hands back the wrong SKU rather than a refusal. Counting is
/// the check that costs nothing and catches that: a bank fused from several
/// parts sums them, and a source that states no extents (every shape `[1]`)
/// is not judged, exactly as [`flattened`](crate::flattened) does not judge it.
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

/// A fused bank whose legs are stored under DIFFERENT encodings, wanted raw.
///
/// llama.cpp mixes schemes tensor by tensor — a `Q4_K_M` file holds one
/// projection at q5_k and the gate beside it at q4_k — and a row that fuses
/// the two into one bf16 plane is not asking for the impossible: each leg
/// decodes on its own, and raw legs concatenate. A decode is a KERNEL, and a
/// kernel nested inside a `Concat` has no byte-run lowering, so each leg is
/// taken to the wanted dtype under its own internal name (never the engine's,
/// as [`affine_reencoded`] does the same for a re-quantized bank) and the
/// published contract joins those outputs, which are plain byte runs.
///
/// Only a QUANTIZED want has no answer here, because codes of two schemes do
/// not join, so this returns `None` and the caller's own refusal stands.
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

/// Whether `name` is stored as a raw floating-point tensor — what a bank the
/// text wants quantized has to be ENCODED from, rather than transmuted.
fn stored_raw(src: &ztensor::Source, name: &str) -> Result<bool, Error> {
    Ok(matches!(
        stored_encoding(src, name)?,
        Encoding::Raw(DType::Bf16 | DType::F16 | DType::F32)
    ))
}

/// Whether `name` is stored in a scheme that keeps its factors INSIDE the
/// payload — every ggml block type, `q4_k` and friends.
///
/// The distinction that matters to a reader: an MLX affine bank names its
/// scales and biases in companion tensors, so another affine form is reached
/// by reading those planes; a ggml block names nothing beside itself, so the
/// only road out of it is through values.
fn stored_block(src: &ztensor::Source, name: &str) -> Result<bool, Error> {
    Ok(match stored_encoding(src, name)? {
        Encoding::Quant(spec) => spec.scheme.is_self_contained(),
        Encoding::Raw(_) => false,
    })
}

/// A bank the row wants quantized, read out of ggml block codes.
///
/// llama.cpp picks a scheme per tensor — a `Q4_K_M` of one model holds `q4_k`,
/// `q5_k`, `q6_k` and `q8_0` side by side — and none of those is a form this
/// catalog's rows name. Codes of two schemes do not transmute into each other,
/// so the value is what carries over: each part decodes to bf16 under its own
/// internal name, the join is materialised (a decode is a KERNEL, and a kernel
/// nested inside a `Concat` has no byte-run lowering), and the ladder encodes
/// that join into the bank the row declares. The encode is paid ONCE, at
/// import, which is the same trade [`affine_reencoded`] makes for a bank
/// stored wider than its row, and what `refuse_a_decode_of_packed_codes` means
/// by "a re-encode takes its stored codes through bf16 on the way to the codes
/// it publishes".
fn reencoded_from_block(
    src: &ztensor::Source,
    w: &Weight,
    parts: &[String],
) -> Result<Vec<TensorContract>, Error> {
    let bf16 = encoding(Dtype::Bf16);
    let want = grouped(w);
    let mut tensors = Vec::with_capacity(parts.len() + 2);
    let mut legs = Vec::with_capacity(parts.len());
    for (at, part) in parts.iter().enumerate() {
        let stored = stored_encoding(src, part)?;
        let decoded = format!("{}.block{at}", w.name);
        let expr = ladder(part, Expr::src(part.clone()), &stored, &bf16)?;
        tensors.push(TensorContract::inferred(decoded.clone(), expr, bf16.clone()).internal());
        legs.push(Expr::out(decoded));
    }
    // One part is already its own byte run; several join on the cut axis, and
    // the join needs a name before anything reads it.
    let values = if legs.len() == 1 {
        format!("{}.block0", w.name)
    } else {
        let joined = format!("{}.values", w.name);
        tensors.push(
            TensorContract::new(
                joined.clone(),
                Expr::concat(pack_axis(w), legs),
                extents(w),
                bf16,
            )
            .internal(),
        );
        joined
    };
    tensors.push(TensorContract::new(
        w.name.clone(),
        Expr::out(values).cast(want.clone()),
        extents(w),
        want,
    ));
    Ok(tensors)
}

/// A placed dtype names a byte ORDER, and [`tiled_planes`] is the only door
/// here that writes one. Every other road to a quantized want publishes the
/// CANONICAL order — an encode's `Cast` emits a bank's three planes together,
/// and the grammar has no way to name one of them and repack it — so a placed
/// row reaching one of those doors would land row-major bytes under a
/// fragment-order declaration.
///
/// That combination is unrefusable downstream and answers nonsense:
/// `kernels_cuda::linear::tiled` says so itself ("an un-repacked plane is not
/// refusable and answers nonsense"), and it is what
/// `qwen35-d0.8b-u4g64-kv-bf16` did on CUDA — importing cleanly from a bf16
/// snapshot and then decoding gibberish, while the same row on a backend
/// whose kernels read the canonical order served correctly. So refuse here,
/// where the road is still named, rather than write bytes no reader can
/// question.
///
/// Lifting this needs the encode to publish its planes under names a
/// `Repack` can take as operands, the way [`affine_reencoded`] already
/// publishes a stored bank's three planes internally; until then a dense
/// source and a placed row have no shared road.
/// The three public planes of a placed bank whose codes an ENCODE produced.
///
/// [`tiled_planes`] states the placement over source codes it can name. An
/// encode cannot be taken apart that way — one `Cast` writes the codes and
/// their scales and biases together — so the encode runs under an internal
/// name at the canonical dtype and the placement is stated over what it
/// wrote. The companions are nameable because the plan publishes them
/// (`plan::build`), which is the whole reason this road exists.
///
/// The factor relabel is the same [`RepackLayout::TiledAffineFactor`]
/// [`relaid`] uses: rows are padded up to a band, never reordered, so a
/// weight whose rows already divide the band gets an identity pass.
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
        relabel(
            model_dsl::scales_name(&w.name),
            model_dsl::scales_name(&canon_name),
        )
        .scaling(pairing),
    );
    out.push(
        relabel(
            model_dsl::biases_name(&w.name),
            model_dsl::biases_name(&canon_name),
        )
        .offsetting(w.name.clone()),
    );
    Ok(out)
}

/// Whether [`placed_from_encode`] knows this placement's relabelling.
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
    // A quantized bank stated by a RAW source (a bf16 head overlaid onto a
    // quantized trunk) is not MLX codes to transmute: it takes the raw
    // reader, whose ladder has the loader encode it on the way in.
    if stored_raw(src, &from)? {
        if placed_road_covers(w) {
            return placed_from_encode(w, |canon| Ok(vec![copy(src, canon, from)?]));
        }
        placed_wants_its_own_order(w, "a raw plane")?;
        return Ok(vec![copy(src, w, from)?]);
    }
    // Codes with no companion planes: whatever the row wants, the road runs
    // through values (see [`reencoded_from_block`]). A raw want takes the
    // ordinary ladder below, which already decodes.
    if stored_block(src, &from)? && matches!(encoding(w.dtype), Encoding::Quant(_)) {
        if placed_road_covers(w) {
            let parts = [from];
            return placed_from_encode(w, |canon| reencoded_from_block(src, canon, &parts));
        }
        placed_wants_its_own_order(w, "a self-contained block")?;
        return reencoded_from_block(src, w, &[from]);
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
        _ => Ok(vec![copy(src, w, from)?]),
    }
}

fn planes_fused(
    src: &ztensor::Source,
    w: &Weight,
    parts: impl IntoIterator<Item = String>,
) -> Result<Vec<TensorContract>, Error> {
    let parts: Vec<String> = parts.into_iter().collect();
    if parts.iter().try_fold(true, |all, part| {
        Ok::<bool, Error>(all && stored_raw(src, part)?)
    })? {
        if placed_road_covers(w) {
            return placed_from_encode(w, |canon| Ok(vec![fused(src, canon, parts)?]));
        }
        placed_wants_its_own_order(w, "raw planes to fuse")?;
        return Ok(vec![fused(src, w, parts)?]);
    }
    // Every leg a self-contained block, and a quantized want: the legs need
    // not agree on a scheme, since each decodes on its own before the join.
    if parts.iter().try_fold(true, |all, part| {
        Ok::<bool, Error>(all && stored_block(src, part)?)
    })? && matches!(encoding(w.dtype), Encoding::Quant(_))
    {
        if placed_road_covers(w) {
            return placed_from_encode(w, |canon| reencoded_from_block(src, canon, &parts));
        }
        placed_wants_its_own_order(w, "self-contained blocks to fuse")?;
        return reencoded_from_block(src, w, &parts);
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
        // The ordinary case is one ladder over the whole join. Legs stored
        // under different encodings have no single ladder, and a raw want
        // decodes each of them instead; anything else keeps the refusal.
        _ => match fused(src, w, parts.clone()) {
            Ok(contract) => Ok(vec![contract]),
            Err(disagreement) => decoded_legs(src, w, &parts).unwrap_or(Err(disagreement)),
        },
    }
}

/// The codes, the scales and the biases of one MLX affine-U4 bank, read out of
/// the `<stem>.weight` / `<stem>.scales` / `<stem>.biases` triplet each part
/// names.
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
        scales.push(model_dsl::scales_name(stem));
        biases.push(model_dsl::biases_name(stem));
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
        // inferred: the transmute above already stated each leg's shape.
        TensorContract::inferred(w.name.clone(), joined(axis, codes), grouped(w)),
        factors(
            src,
            model_dsl::scales_name(&w.name),
            &scales,
            counted.clone(),
            axis,
        )?
        .scaling(pairing),
        // offsetting: the zero-point entry, names which weight it centres.
        factors(src, model_dsl::biases_name(&w.name), &biases, counted, axis)?
            .offsetting(w.name.clone()),
    ])
}

/// The width the file stores an affine bank at, read off its word count: a
/// `[N, K / codes-a-word]` u32 plane spans the declared `K`, so the codes a
/// word holds — and the bits — are the ratio. The row's dtype names the
/// SERVED form; where the two differ, [`affine_reencoded`] takes the bank
/// through its values. A pack joined on the contracted axis has no one `K`
/// a leg spans, so it is read as stored at the row's own width.
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

/// [`affine_planes`] for a bank the file stores at another width than the
/// row serves (an 8-bit conversion read by a 4-bit row): no kernel re-encodes
/// codes in one step, so the stored triplet is read under `<name>.stored`
/// (internal: never the engine's), taken to bf16 by its own scales and
/// biases — the affine fragment's per-block `Scale` decodes MLX codes — and
/// that value plane is encoded as the row's form, which publishes the new
/// scales and biases itself. A second quantization: a row takes it on
/// purpose (pinned), never by identification.
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
        scales.push(model_dsl::scales_name(stem));
        biases.push(model_dsl::biases_name(stem));
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

/// [`affine_planes`] for a bank stored one row at a time: each row's parts
/// fuse along the cut axis, and the rows stack on a new leading axis.
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
                (model_dsl::scales_name(stem), &mut row_scales),
                (model_dsl::biases_name(stem), &mut row_biases),
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
    let scales_name = model_dsl::scales_name(&w.name);
    let biases_name = model_dsl::biases_name(&w.name);
    let scales = ladder(&scales_name, joined(0, scales), &stored, &want)?;
    let biases = ladder(&biases_name, joined(0, biases), &stored, &want)?;
    Ok(vec![
        TensorContract::inferred(w.name.clone(), joined(0, codes), grouped(w)),
        TensorContract::new(scales_name, scales, counted.clone(), want.clone()).scaling(pairing),
        TensorContract::new(biases_name, biases, counted, want).offsetting(w.name.clone()),
    ])
}

/// The codes and the exponents of one MLX mxfp4 bank, read out of the
/// `<stem>.weight` (u32 words, eight e2m1 codes each) / `<stem>.scales` (one
/// e8m0 byte per 32 codes) pair each part names. MLX packs the words the way
/// the artifact holds its mxfp4 planes, so both are transmuted, not
/// re-encoded; there is no bias plane — mxfp4 centres on zero.
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
        let scale = model_dsl::scales_name(stem);
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
            model_dsl::scales_name(&w.name),
            joined(axis, scales),
            counted,
            encoding(Dtype::E8m0),
        )
        .scaling(pairing),
    ])
}

/// The same triplet, relaid — [`affine_planes`]'s three entries with an
/// [`Expr::Repack`] on the end of each, target rows rounded up to a whole
/// [`TILED_BAND`] as [`claim`] claims. Padding is zero codes beside zero
/// factors, decoding to a zero weight.
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
        scales.push(model_dsl::scales_name(stem));
        biases.push(model_dsl::biases_name(stem));
    }
    // Checked against the flat rectangle the legs join into; the band order
    // below is a relabelling of that same shape.
    holds_the_declared_rectangle(w, axis, &legs)?;
    let pairing = scaling(w);
    // The flat rectangle, and the banded one it's relaid into.
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
            model_dsl::scales_name(&w.name),
            &scales,
            counted.clone(),
            axis,
            factor_target(counted.clone()),
        )?
        .scaling(pairing),
        relaid(
            src,
            model_dsl::biases_name(&w.name),
            &biases,
            counted.clone(),
            axis,
            factor_target(counted),
        )?
        .offsetting(w.name.clone()),
    ])
}

/// [`factors`] with the relabelling on the end — one companion plane, joined
/// at its seams and then put into band order.
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

/// One companion plane of an affine bank, joined across the parts and brought
/// to bf16 — a real cast: mlx-community ships some conversions with F16
/// factors and others with BF16.
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

/// The logical shape of a stored affine-U4 plane: what the checkpoint holds,
/// with its contracted axis multiplied back out of the words it was packed
/// into. Read off the file, not the declaration: the declaration is the
/// whole joined bank and this is one leg, and a qkv pack's legs are not
/// equal width.
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

/// Checks that the legs join into the rectangle this text declared.
/// [`TensorContract::inferred`] never compares stored width to the declared
/// one on its own, so this is what catches a bank read by the wrong row
/// (e.g. a 64-expert bank silently served as 16).
///
/// Checked against the joined shape, not per leg: a fused bank's legs need
/// not be equal width (a qkv pack), so they must agree everywhere off the
/// seam and sum to the declaration on it.
///
/// The contract is always built at `tp == 1` ([`Builder::read`] and siblings
/// call `whole_checkpoint` first), so [`extents`] here is the whole model's
/// rectangle.
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

/// The companion planes the checkpoint shipped beside `of`, declared in their
/// own right. mxfp4 has one companion (an exponent byte per block); MLX
/// affine U4 has two (scale and offset, `code * scale + bias`), declared as
/// siblings the way MLX ships them. `pairing.form` says how many there are.
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
        TensorContract::new(name, expr, shape, encoding(dtype))
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

/// How the checkpoint stores `name`, read off the file's own header — the
/// fact every conversion decision here starts from.
pub fn stored_encoding(src: &ztensor::Source, name: &str) -> Result<Encoding, Error> {
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
        // Packed codes are not values to quantize: a stored u32-word bank
        // must be read by naming its planes, not by encoding its words.
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

/// `w`'s declared shape, as the signed extents a contract states.
pub fn extents(w: &Weight) -> Vec<i64> {
    w.shape
        .iter()
        .map(|extent| i64::try_from(*extent).expect("an extent no i64 holds"))
        .collect()
}

fn whole(w: &Weight, tp: u32) -> Vec<i64> {
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

/// `shape` with `axis` counted in `group`-code blocks — the shape of a scales
/// plane, derived from the bank it reads.
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

/// How `w`'s codes are paired with the numbers that read them. Group width
/// and form both come from the same scheme (e.g. mxfp4: 32 codes under one
/// exponent byte) — mixing them would produce a pairing nothing ships.
pub fn scaling(w: &Weight) -> Scales {
    let form = match w.dtype {
        Dtype::Mxfp4 => ScaleForm::RawE8M0,
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

/// `w`'s quantized encoding with its blocked axis stated — the channel axis
/// is the bank's own last, because a rank is not a fact about a scheme.
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

/// The stored representation each weight [`Dtype`] declares -- raw of
/// itself for everything that stores itself verbatim, and the quantization
/// spec of the scheme for a packed bank's codes.
///
/// # Panics
///
/// On a [`Dtype::Quant`] whose term no self-contained scheme here has.
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
        // MLX affine U4: 64 codes under one bf16 scale and one bf16 offset.
        Dtype::U4g64 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 64,
            channel_axis: None,
        }),
        // Same scheme, twice the code width (`bits_per_element` says how wide).
        Dtype::U8g64 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 8,
            group_size: 64,
            channel_axis: None,
        }),
        // Same scheme, half the group (`group_size` codes share a scale).
        Dtype::U4g32 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: None,
        }),
        // Same encoding as `U4g64`; a repack moves no value, only layout
        // (tensor-core fragment order), which lives on the `Dtype` instead.
        Dtype::U4g64tiled => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 64,
            channel_axis: None,
        }),
        // Same scheme at two bits, over its three group sizes.
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
        Dtype::E2m1 => panic!(
            "`Dtype::E2m1` names a kv-page quantization scheme, not a stored \
             weight plane; no load contract declares one"
        ),
        Dtype::Nvfp4 | Dtype::E4m3row | Dtype::E4m3tile128 => panic!(
            "a {:?} weight is served but no load contract declares one",
            dtype
        ),
        // This variant carries the arithmetic itself; look up which block
        // scheme has it rather than respelling the numbers here.
        Dtype::U2g16k | Dtype::I3g16k | Dtype::U4g32k | Dtype::U5g32k | Dtype::I6g16k => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use checkpoint::types::Visibility;
    use model_dsl::Weight;

    const ROWS: u64 = 32;
    const K: u64 = 128;

    /// One bf16 plane and nothing beside it — the shape a dense snapshot
    /// has, and the shape [`stored_raw`] answers `true` for.
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

    /// **A PLACED ROW AND A DENSE SOURCE NOW SHARE A ROAD, AND IT LANDS THE
    /// PLACEMENT.** `U4g64tiled` names a byte ORDER, and for a while
    /// [`tiled_planes`] was the only door that wrote one — so a raw source
    /// took the encode road, which publishes the canonical order, the import
    /// wrote row-major bytes under a fragment-order declaration,
    /// `kernels_cuda::linear::tiled` read them as fragments, and
    /// `qwen35-d0.8b-u4g64-kv-bf16` answered gibberish on CUDA while the
    /// same row served correctly everywhere else. The encode runs under an
    /// internal name now and the placement is stated over what it wrote, so
    /// what a reader gets is fragments. The assertion is that the public
    /// codes carry the fragment `Repack` and that the encode itself is not
    /// public — a canonical plane left visible is the old bug wearing a
    /// different name.
    #[test]
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

    /// The fallback CUDAQ left, tested where it lives: `U4g64tiled` is the
    /// only placed variant today, so nothing reaches this through
    /// [`planes`] — and the day a second one lands, it refuses by name
    /// rather than quietly publishing canonical bytes again.
    #[test]
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

    /// The same source and the same width, unplaced: the encode road is
    /// exactly right, and nothing above refuses it. This is the half that
    /// keeps the refusal from being a blanket one — every backend whose
    /// kernels read the canonical order still imports this checkpoint.
    #[test]
    fn the_unplaced_sibling_of_that_row_still_reads_it() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let src = raw_source(dir.path());
        let w = Weight::sym("proj", [ROWS, K], Dtype::U4g64);
        planes(&src, &w, "proj").expect("the canonical order is what an encode publishes");
    }

    /// **THE ROAD THAT DOES LAND THE PLACED ORDER STILL DOES.** MLX affine
    /// codes name their factors in companion tensors, so [`tiled_planes`]
    /// can put a `Repack` on each of the three planes — and it band-pads the
    /// rows while it is there, which no other road does. 24 rows landing 32
    /// is the cheapest proof that this contract went through the repack and
    /// not through an encode; nothing else in this crate rounds a row count
    /// up to a whole `TILED_BAND`.
    #[test]
    fn the_mlx_road_lands_the_placed_order_and_bands_its_rows() {
        const ODD: u64 = 24;
        let dir = tempfile::tempdir().expect("a scratch directory");
        let path = dir.path().join("affine.zt");
        let mut writer = ztensor::Writer::create(&path).expect("the container opens");
        let groups = (K / 64) as usize;
        // Sorted insertion, which is what canonical `.zt` form requires:
        // the two factor planes sort before the codes.
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
        // MLX codes are raw u32 words, eight four-bit codes to a word.
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

    /// A dense row over the same plane, which is neither placed nor
    /// quantized: the guard is on the placement, not on the door.
    #[test]
    fn a_dense_row_is_untouched_by_the_guard() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let src = raw_source(dir.path());
        let w = Weight::sym("proj", [ROWS, K], Dtype::Bf16);
        planes(&src, &w, "proj").expect("bf16 in, bf16 out");
    }
}
