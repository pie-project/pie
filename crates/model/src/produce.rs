//! The Import interpreter: a family's production table run over host
//! tensors, one THIS RANK'S tensor per row.
//!
//! [`produce`] is the executable half of the Load contract's supply side.
//! The demand side is a plan's `params` column; the two are joined by name
//! and shape, and `bin/baker_load.rs` is that join run against a real
//! checkpoint.
//!
//! THE DEMAND COLUMN IS AN ARGUMENT HERE, and that is the rank cut. A
//! `-tp2` catalog row states per-rank shapes (`Param::shape` IS what this
//! rank holds) and a shard mark saying which axis the knife ran along; the
//! checkpoint holds the whole tensor. So the last thing a production row
//! does is take this rank's slice of what it just built, and the interpreter
//! reads the SAME `params` the join is about to run against — see [`cut`]
//! for why the slice lives here and not at the driver's upload.
//!
//! The interpreter reads the checkpoint through a closure rather than a
//! reader type on purpose. A production table names checkpoint tensors in
//! the family's own spelling -- `layer.3.self_attn.q_proj`, not
//! `model.language_model.layers.3.self_attn.q_proj.weight` -- so whoever
//! opens the file owns the naming convention, and the verbs stay about
//! bytes. That also keeps this module free of every checkpoint format:
//! safetensors, GGUF and zt all reduce to `&dyn Fn(&str) ->
//! Option<HostTensor>`.

use model_dsl::load::{Import, Source};
use model_ir::plan::{Param, Shard};

/// A checkpoint tensor's element type, in the safetensors spelling.
///
/// This is the *storage* axis, which is not the plan's `repr` column: a
/// plan says `bf16` for a bank the checkpoint may ship as f32 (qwen's
/// `A_log` and its gdn norm both do). The join reports that; the
/// interpreter only refuses when a verb cannot be performed at the dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
    F64,
    F8E4M3,
    F8E5M2,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

impl Dtype {
    /// Bytes per element.
    #[must_use]
    pub fn width(self) -> usize {
        match self {
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
            Dtype::Bf16 | Dtype::F16 | Dtype::I16 | Dtype::U16 => 2,
            Dtype::F8E4M3 | Dtype::F8E5M2 | Dtype::I8 | Dtype::U8 | Dtype::Bool => 1,
        }
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Dtype::Bf16 => "BF16",
            Dtype::F16 => "F16",
            Dtype::F32 => "F32",
            Dtype::F64 => "F64",
            Dtype::F8E4M3 => "F8_E4M3",
            Dtype::F8E5M2 => "F8_E5M2",
            Dtype::I8 => "I8",
            Dtype::I16 => "I16",
            Dtype::I32 => "I32",
            Dtype::I64 => "I64",
            Dtype::U8 => "U8",
            Dtype::U16 => "U16",
            Dtype::U32 => "U32",
            Dtype::U64 => "U64",
            Dtype::Bool => "BOOL",
        }
    }

    /// The safetensors header spelling, or `None` for one this crate has
    /// no width for.
    #[must_use]
    pub fn parse(s: &str) -> Option<Dtype> {
        Some(match s {
            "BF16" => Dtype::Bf16,
            "F16" => Dtype::F16,
            "F32" => Dtype::F32,
            "F64" => Dtype::F64,
            "F8_E4M3" => Dtype::F8E4M3,
            "F8_E5M2" => Dtype::F8E5M2,
            "I8" => Dtype::I8,
            "I16" => Dtype::I16,
            "I32" => Dtype::I32,
            "I64" => Dtype::I64,
            "U8" => Dtype::U8,
            "U16" => Dtype::U16,
            "U32" => Dtype::U32,
            "U64" => Dtype::U64,
            "BOOL" => Dtype::Bool,
            _ => return None,
        })
    }
}

/// One tensor on the host: shape, dtype, bytes. Row-major, dense, no
/// strides -- a checkpoint part and a canonical part are both this, which
/// is what lets a verb be a byte move.
#[derive(Clone, Debug)]
pub struct HostTensor {
    pub shape: Vec<u64>,
    pub dtype: Dtype,
    pub bytes: Vec<u8>,
}

impl HostTensor {
    #[must_use]
    pub fn new(shape: impl IntoIterator<Item = u64>, dtype: Dtype, bytes: Vec<u8>) -> HostTensor {
        HostTensor {
            shape: shape.into_iter().collect(),
            dtype,
            bytes,
        }
    }

    /// Element count. A rank-0 part holds one element, which is what makes
    /// an empty shape well formed rather than empty.
    #[must_use]
    pub fn elems(&self) -> u64 {
        self.shape.iter().product()
    }

    /// The leading axis's extent -- what `Pack` sums and `Deinterleave`
    /// permutes. Rank 0 counts as one row.
    #[must_use]
    pub fn rows(&self) -> u64 {
        self.shape.first().copied().unwrap_or(1)
    }

    /// Everything under the leading axis, as bytes.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        let rows = self.rows().max(1) as usize;
        self.bytes.len() / rows
    }

    fn well_formed(&self) -> bool {
        self.bytes.len() as u64 == self.elems() * self.dtype.width() as u64
    }
}

/// A production row that could not be run, and which row it was.
#[derive(Clone, Debug)]
pub struct ProduceError {
    pub target: String,
    pub fault: Fault,
}

/// Why one row refused. Every variant names the checkpoint tensor it was
/// reading, because "the import table is wrong" and "the checkpoint is not
/// the one this table was written for" are the same message otherwise.
#[derive(Clone, Debug)]
pub enum Fault {
    /// The checkpoint holds no tensor under that name.
    Absent { name: String },
    /// The header's shape and the payload's length disagree.
    Ragged {
        name: String,
        shape: Vec<u64>,
        dtype: Dtype,
        bytes: usize,
    },
    /// `Pack`/`Stack` over nothing produces nothing.
    Empty { verb: &'static str },
    /// Two operands of one fold at two dtypes.
    Mixed {
        verb: &'static str,
        a: Dtype,
        b: Dtype,
    },
    /// `Pack`: the operands disagree under the leading axis.
    Unpackable { a: Vec<u64>, b: Vec<u64> },
    /// `Stack`: the operands are not the same rectangle.
    Unstackable { a: Vec<u64>, b: Vec<u64> },
    /// `Squeeze`: no such axis, or its extent is not 1.
    Unsqueezable {
        name: String,
        axis: u32,
        shape: Vec<u64>,
    },
    /// `Deinterleave`: no such axis, or its extent does not divide by the
    /// group count.
    Ungroupable {
        name: String,
        axis: u32,
        shape: Vec<u64>,
        groups: u32,
    },
    /// A verb this interpreter does not perform yet.
    Refused { verb: &'static str, why: String },
    /// The plan's shard column and the checkpoint's own rectangle do not
    /// describe one cut of one tensor: a missing axis, a disagreement off
    /// the cut axis, an extent this rank's does not divide, or segments
    /// that do not cover the axis they partition.
    Uncuttable {
        axis: u32,
        whole: Vec<u64>,
        mine: Vec<u64>,
    },
    /// A rank outside the world the checkpoint says this weight was cut
    /// into -- including rank 1 of a plan that cuts nothing, which is a
    /// whole-model SKU handed to a two-way deployment.
    Unranked { rank: u32, world: u32 },
    /// Two weights of one plan, cut two different ways by one checkpoint.
    Uneven { world: u32, saw: u32 },
}

impl std::fmt::Display for ProduceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "`{}`: ", self.target)?;
        match &self.fault {
            Fault::Absent { name } => write!(f, "the checkpoint holds no `{name}`"),
            Fault::Ragged {
                name,
                shape,
                dtype,
                bytes,
            } => write!(
                f,
                "`{name}` is {shape:?} {} but carries {bytes} bytes",
                dtype.name()
            ),
            Fault::Empty { verb } => write!(f, "{verb} over no sources"),
            Fault::Mixed { verb, a, b } => {
                write!(f, "{verb} mixes {} with {}", a.name(), b.name())
            }
            Fault::Unpackable { a, b } => {
                write!(f, "pack: {a:?} and {b:?} disagree under the leading axis")
            }
            Fault::Unstackable { a, b } => {
                write!(f, "stack: {a:?} and {b:?} are not the same rectangle")
            }
            Fault::Unsqueezable { name, axis, shape } => write!(
                f,
                "`{name}` is {shape:?}, whose axis {axis} is not an extent of 1"
            ),
            Fault::Ungroupable {
                name,
                axis,
                shape,
                groups,
            } => write!(
                f,
                "`{name}` is {shape:?}, whose axis {axis} is not a whole {groups} groups"
            ),
            Fault::Refused { verb, why } => write!(f, "{verb} is refused: {why}"),
            Fault::Uncuttable { axis, whole, mine } => write!(
                f,
                "the checkpoint holds {whole:?} and this rank holds {mine:?}, \
                 which is not that rectangle cut on axis {axis}"
            ),
            Fault::Unranked { rank, world } => write!(
                f,
                "the checkpoint cuts this weight {world} way(s) and this load \
                 was asked for rank {rank}"
            ),
            Fault::Uneven { world, saw } => write!(
                f,
                "this weight is cut {saw} ways where an earlier one is cut \
                 {world}; a checkpoint is ONE cut of one plan"
            ),
        }
    }
}

impl std::error::Error for ProduceError {}

/// Run every row of `import` against a checkpoint, in table order, and hand
/// back what `rank` holds.
///
/// `read` answers with the checkpoint's tensor under the family's own
/// spelling, or `None` when it holds none. `params` is the plan's demand
/// column -- the same one the join is about to run against -- and it is what
/// makes this a rank's load rather than a checkpoint's: each row's tensor is
/// [`cut`] by the param of the same name before it is pushed.
///
/// A target no param names is passed through WHOLE. There is nothing to cut
/// it by, the join already reports it as a produced row nobody demands, and
/// guessing an axis for it would be the one thing worse than moving the
/// bytes.
///
/// # THE WORLD IS DERIVED, ONCE, AND CHECKED
///
/// Nothing states the degree. Every cut row says one -- `checkpoint extent /
/// Param::shape[axis]` -- and they must all say the same one, because a
/// checkpoint is ONE cut of one plan. A row that disagrees is
/// [`Fault::Uneven`] and names both numbers; a `rank` no row's world holds
/// is [`Fault::Unranked`]. At world 1 (every single-GPU row) the cut is the
/// identity and no byte moves, which is why a `-tp1` load through here is
/// the same bytes it was before this argument existed.
///
/// # Errors
///
/// The first row that could not be run or could not be cut, named.
pub fn produce(
    import: &Import,
    params: &[Param],
    rank: u32,
    read: &dyn Fn(&str) -> Option<HostTensor>,
) -> Result<Vec<(String, HostTensor)>, ProduceError> {
    let mut out = Vec::with_capacity(import.rows.len());
    let mut world: Option<u32> = None;
    for row in &import.rows {
        let named = |fault| ProduceError {
            target: row.target.clone(),
            fault,
        };
        let t = run(&row.source, read).map_err(named)?;
        let Some(p) = params.iter().find(|p| p.name == row.target) else {
            out.push((row.target.clone(), t));
            continue;
        };
        let (t, saw) = cut(t, p, rank).map_err(named)?;
        if let Some(saw) = saw {
            match world {
                Some(seen) if seen != saw => return Err(named(Fault::Uneven { world: seen, saw })),
                _ => world = Some(saw),
            }
        }
        out.push((row.target.clone(), t));
    }
    Ok(out)
}

/// One rank's share of a canonical tensor, and the world the checkpoint says
/// it was cut into -- `None` for a weight every rank holds whole.
///
/// # WHY THE SLICE IS HERE
///
/// Three places could hold it and only one of them can be checked.
///
/// * **The import table.** No: `model-dsl`'s `load` module states the rule
///   in its first line -- import may rewrite bytes, load may only view -- and
///   a cut verb there would put the degree in the production table, where a
///   `-tp2` row would then need a table of its own that differed from its
///   sibling's by nothing a checkpoint can see. A `-tp2` row takes its
///   sibling's table VERBATIM and this function is the whole difference.
/// * **The driver, at upload.** No, twice over. The join `bin/baker_load.rs`
///   runs is `plan.params` against what production produced, and
///   `Param::shape` is what a rank holds -- so a driver-side cut would leave
///   every sharded row of a `-tp2` join reading MISMATCH, and the only way
///   to a green gate would be teaching the join to accept "a multiple of the
///   demanded extent", which is the join softened into saying nothing. And
///   the upload loop's whole documented virtue is that it has no decision in
///   it: one contiguous H2D per bank, no restride, no repack, no cast. A
///   slice IS a restride.
/// * **Here**, in the interpreter, fused into the production loop. The
///   driver and the CLI reach one implementation, so the gate exercises the
///   arithmetic the driver runs; the join stays the join it already was; and
///   peak host memory is a rank's model plus one whole tensor rather than
///   the whole model plus a rank's (a3b: 34 GiB + one bank, not 66 + 34).
///
/// # The arithmetic, and what it checks
///
/// EACH SEGMENT IS CUT, which is what [`Shard::Cut`]'s segment list is for.
/// A `[gate | up]` bank at half width is `[gate/2 | up/2]`, so this walks the
/// segments and takes `rank`'s share out of each one where it sits in the
/// whole -- never `rank`'s half of the concatenated axis, which would hand
/// rank 0 the whole gate and rank 1 the whole up.
///
/// A QUANTIZED BANK NEEDS NO ARM. `model_dsl`'s `restated` already says a
/// bank's cut in each stored plane's own extents (mxfp4 codes are
/// `[.., K/32, 16]` and the segments come divided by the block), so this
/// slices axis `axis` of the shape the checkpoint actually holds and the
/// whole-code-block check is the divisibility check below -- a cut that
/// landed on half a block would show up as an extent this rank's does not
/// divide.
///
/// What is NOT checked here is a `Replicated` row whose rectangle is not the
/// plan's. There is nothing this could do about it that the join does not do
/// better, with both shapes in the message; the rule is that the interpreter
/// refuses what it cannot PERFORM and the join reports everything else.
fn cut(t: HostTensor, p: &Param, rank: u32) -> Result<(HostTensor, Option<u32>), Fault> {
    let Shard::Cut { axis, segments } = &p.shard else {
        return Ok((t, None));
    };
    let at = *axis as usize;
    let (Some(&whole), Some(&mine)) = (t.shape.get(at), p.shape.get(at)) else {
        return Err(uncuttable(&t, p, *axis));
    };
    let off_axis = |(i, (a, b)): (usize, (&u64, &u64))| i != at && a != b;
    if t.shape.len() != p.shape.len()
        || t.shape.iter().zip(&p.shape).enumerate().any(off_axis)
        || mine == 0
        || !whole.is_multiple_of(mine)
        || segments.iter().sum::<u64>() != mine
    {
        return Err(uncuttable(&t, p, *axis));
    }
    let world = u32::try_from(whole / mine).map_err(|_| uncuttable(&t, p, *axis))?;
    if rank >= world {
        return Err(Fault::Unranked { rank, world });
    }
    if world == 1 {
        // The identity, taken as one. A single-rank load must not pay a copy
        // of every weight in the model to slice `[0, whole)` out of
        // `[0, whole)` -- and this is also what makes a `-tp1` load byte-
        // identical by construction rather than by inspection.
        return Ok((t, Some(1)));
    }

    // Below the cut axis: the bytes one position of it carries, which move as
    // a unit and are never inspected. Above it: every combination of the
    // leading extents, each holding one whole copy of the partitioned axis.
    let inner = t.shape[at + 1..].iter().product::<u64>() as usize * t.dtype.width();
    let outer = t.shape[..at].iter().product::<u64>() as usize;
    let stride = whole as usize * inner;
    let mut bytes = Vec::with_capacity(outer * mine as usize * inner);
    for o in 0..outer {
        let base = o * stride;
        // Where the WHOLE segment starts on the axis; this rank's share of it
        // begins `rank` shares in.
        let mut start = 0usize;
        for seg in segments {
            let take = *seg as usize;
            let from = base + (start + rank as usize * take) * inner;
            bytes.extend_from_slice(&t.bytes[from..from + take * inner]);
            start += take * world as usize;
        }
    }
    let mut shape = t.shape;
    shape[at] = mine;
    Ok((
        HostTensor {
            shape,
            dtype: t.dtype,
            bytes,
        },
        Some(world),
    ))
}

/// The one refusal [`cut`] has for a cut it cannot perform, carrying both
/// rectangles: every way the arithmetic can fail is the same sentence, and
/// which of the two tables is wrong is the reader's to decide from them.
fn uncuttable(t: &HostTensor, p: &Param, axis: u32) -> Fault {
    Fault::Uncuttable {
        axis,
        whole: t.shape.clone(),
        mine: p.shape.clone(),
    }
}

fn fetch(name: &str, read: &dyn Fn(&str) -> Option<HostTensor>) -> Result<HostTensor, Fault> {
    let t = read(name).ok_or_else(|| Fault::Absent {
        name: name.to_string(),
    })?;
    if !t.well_formed() {
        return Err(Fault::Ragged {
            name: name.to_string(),
            shape: t.shape.clone(),
            dtype: t.dtype,
            bytes: t.bytes.len(),
        });
    }
    Ok(t)
}

fn run(s: &Source, read: &dyn Fn(&str) -> Option<HostTensor>) -> Result<HostTensor, Fault> {
    match s {
        Source::Copy(name) => fetch(name, read),
        Source::Pack(sources) => pack(sources, read),
        Source::Stack(sources) => stack(sources, read),
        Source::Deinterleave(name, axis, groups) => {
            deinterleave(fetch(name, read)?, name, *axis, *groups)
        }
        Source::Squeeze(name, axis) => squeeze(fetch(name, read)?, name, *axis),
    }
}

/// Concatenate along the OUT axis -- axis 0, the axis a projection's rows
/// live on. `pack([q, k, v])` is exactly the fused qkv bank gemma's
/// `Qkv::Packed` declares, and `Tensor::packed`'s segment list is the same
/// row counts in the same order.
fn pack(
    sources: &[Source],
    read: &dyn Fn(&str) -> Option<HostTensor>,
) -> Result<HostTensor, Fault> {
    let mut parts = Vec::with_capacity(sources.len());
    for s in sources {
        parts.push(run(s, read)?);
    }
    let (first, rest) = parts.split_first().ok_or(Fault::Empty { verb: "pack" })?;
    let mut rows = first.rows();
    let mut bytes = Vec::with_capacity(parts.iter().map(|p| p.bytes.len()).sum());
    bytes.extend_from_slice(&first.bytes);
    for p in rest {
        if p.dtype != first.dtype {
            return Err(Fault::Mixed {
                verb: "pack",
                a: first.dtype,
                b: p.dtype,
            });
        }
        if p.shape.get(1..) != first.shape.get(1..) {
            return Err(Fault::Unpackable {
                a: first.shape.clone(),
                b: p.shape.clone(),
            });
        }
        rows += p.rows();
        bytes.extend_from_slice(&p.bytes);
    }
    let mut shape = first.shape.clone();
    if shape.is_empty() {
        shape.push(rows);
    } else {
        shape[0] = rows;
    }
    Ok(HostTensor {
        shape,
        dtype: first.dtype,
        bytes,
    })
}

/// Stack under a NEW leading axis -- what turns per-expert banks into the
/// one `[experts, ..]` tensor a routed bank declares.
fn stack(
    sources: &[Source],
    read: &dyn Fn(&str) -> Option<HostTensor>,
) -> Result<HostTensor, Fault> {
    let mut parts = Vec::with_capacity(sources.len());
    for s in sources {
        parts.push(run(s, read)?);
    }
    let (first, rest) = parts.split_first().ok_or(Fault::Empty { verb: "stack" })?;
    let mut bytes = Vec::with_capacity(parts.iter().map(|p| p.bytes.len()).sum());
    bytes.extend_from_slice(&first.bytes);
    for p in rest {
        if p.dtype != first.dtype {
            return Err(Fault::Mixed {
                verb: "stack",
                a: first.dtype,
                b: p.dtype,
            });
        }
        if p.shape != first.shape {
            return Err(Fault::Unstackable {
                a: first.shape.clone(),
                b: p.shape.clone(),
            });
        }
        bytes.extend_from_slice(&p.bytes);
    }
    let mut shape = vec![parts.len() as u64];
    shape.extend_from_slice(&first.shape);
    Ok(HostTensor {
        shape,
        dtype: first.dtype,
        bytes,
    })
}

/// Ungroup `groups`-way interleaving along `axis`. A checkpoint that stores a
/// fused bank as `g0 g1 g0 g1 ...` down that axis becomes `g0 g0 ... g1 g1
/// ...`, which is what a `Packed` segment list means. The shape does not move
/// -- only the rows do.
///
/// THE AXIS IS NOT ALWAYS ZERO, and assuming it was is what this fn used to
/// do. gpt-oss's fused gate/up bank is `[experts, 2 * inter, ..]`: the
/// interleaving is under the expert fan, and a leading-axis permutation there
/// shuffles WHICH EXPERT IS WHICH while leaving every shape and byte count
/// intact -- undetectable by the join, by the shape walk, and by anything
/// short of a numeric read. So the axis is stated and the axes ABOVE it are
/// walked: `outer` positions, each holding one `groups * seg` run of `inner`
/// bytes.
fn deinterleave(t: HostTensor, name: &str, axis: u32, groups: u32) -> Result<HostTensor, Fault> {
    let at = axis as usize;
    let ungroupable = || Fault::Ungroupable {
        name: name.to_string(),
        axis,
        shape: t.shape.clone(),
        groups,
    };
    let Some(&extent) = t.shape.get(at) else {
        return Err(ungroupable());
    };
    if groups == 0 || !extent.is_multiple_of(u64::from(groups)) {
        return Err(ungroupable());
    }
    // Above the axis: every combination of the leading extents, each of which
    // holds one whole interleaved run. Below it: the bytes one position of the
    // axis carries, which move as a unit and are never inspected.
    let outer: u64 = t.shape[..at].iter().product();
    let below: u64 = t.shape[at + 1..].iter().product();
    let inner = (below * t.dtype.width() as u64) as usize;
    let run = extent as usize * inner;
    let seg = (extent / u64::from(groups)) as usize;
    let groups = groups as usize;
    let mut bytes = Vec::with_capacity(t.bytes.len());
    for o in 0..outer as usize {
        let base = o * run;
        for g in 0..groups {
            for j in 0..seg {
                let src = base + (j * groups + g) * inner;
                bytes.extend_from_slice(&t.bytes[src..src + inner]);
            }
        }
    }
    Ok(HostTensor { bytes, ..t })
}

/// Drop one extent-1 axis. Bytes do not move; the rectangle loses a
/// degenerate axis the checkpoint's framework put there.
fn squeeze(mut t: HostTensor, name: &str, axis: u32) -> Result<HostTensor, Fault> {
    let a = axis as usize;
    if t.shape.get(a) != Some(&1) {
        return Err(Fault::Unsqueezable {
            name: name.to_string(),
            axis,
            shape: t.shape.clone(),
        });
    }
    t.shape.remove(a);
    Ok(t)
}

#[must_use]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}
