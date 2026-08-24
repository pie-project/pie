//! The Import interpreter: a family's production table run over host
//! tensors, one canonical tensor per row.
//!
//! [`produce`] is the executable half of the Load contract's supply side.
//! The demand side is a plan's `params` column; the two are joined by name
//! and shape, and `bin/baker_load.rs` is that join run against a real
//! checkpoint.
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
    /// `PlusOne` at a dtype with no float fold.
    Unfoldable { name: String, dtype: Dtype },
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
            Fault::Unfoldable { name, dtype } => write!(
                f,
                "the (1 + w) fold has no reading at {} (`{name}`)",
                dtype.name()
            ),
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
        }
    }
}

impl std::error::Error for ProduceError {}

/// Run every row of `import` against a checkpoint, in table order.
///
/// `read` answers with the checkpoint's tensor under the family's own
/// spelling, or `None` when it holds none. The result is the canonical
/// tensors, named as the plan's `params` column names them -- which is
/// what makes the join a lookup and not a translation.
pub fn produce(
    import: &Import,
    read: &dyn Fn(&str) -> Option<HostTensor>,
) -> Result<Vec<(String, HostTensor)>, ProduceError> {
    let mut out = Vec::with_capacity(import.rows.len());
    for row in &import.rows {
        let t = run(&row.source, read).map_err(|fault| ProduceError {
            target: row.target.clone(),
            fault,
        })?;
        out.push((row.target.clone(), t));
    }
    Ok(out)
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
        Source::PlusOne(name) => plus_one(fetch(name, read)?, name),
        Source::Pack(sources) => pack(sources, read),
        Source::Stack(sources) => stack(sources, read),
        Source::ScalarOf(name) => Err(Fault::Refused {
            verb: "scalar_of",
            why: format!(
                "no shipping checkpoint has been read that stores a scalar beside \
                 `{name}` -- gemma's HF release files it as a `[1]` tensor of its \
                 own (`layer.{{l}}.layer_scalar`), which is a `copy`, and the GGUF \
                 release's form is unmeasured. Until one is read, a guess here \
                 would be a silently wrong weight rather than a refusal"
            ),
        }),
        Source::Deinterleave(name, axis, groups) => {
            deinterleave(fetch(name, read)?, name, *axis, *groups)
        }
        Source::Squeeze(name, axis) => squeeze(fetch(name, read)?, name, *axis),
    }
}

/// The `(1 + w)` norm fold. Gemma ships its rmsnorm weights centred on
/// zero and the canonical weight is plain, so the fold is the import's
/// and not the kernel's. Computed in f32 and written back at the source
/// dtype: a bf16 `1 + w` done in bf16 would lose the whole mantissa of a
/// small `w`.
fn plus_one(mut t: HostTensor, name: &str) -> Result<HostTensor, Fault> {
    match t.dtype {
        Dtype::Bf16 => {
            for c in t.bytes.chunks_exact_mut(2) {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                let folded = f32_to_bf16(bf16_to_f32(bits) + 1.0);
                c.copy_from_slice(&folded.to_le_bytes());
            }
        }
        Dtype::F32 => {
            for c in t.bytes.chunks_exact_mut(4) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) + 1.0;
                c.copy_from_slice(&v.to_le_bytes());
            }
        }
        // NO f16 ARM, and it is a refusal rather than an omission. Every
        // norm this fold has been run against ships bf16 (gemma) or f32
        // (qwen's gdn rows); an f16 codec written for nobody would be
        // thirty lines of unexercised rounding whose first exercise would
        // be a wrong weight on a GPU. A refusal names the row instead.
        dtype => {
            return Err(Fault::Unfoldable {
                name: name.to_string(),
                dtype,
            });
        }
    }
    Ok(t)
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
/// one `[experts, ...]` tensor an `.experts()` weight declares.
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

/// Round-to-nearest-even, the same rule every bf16 cast in this tree uses.
#[must_use]
pub fn f32_to_bf16(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7FC0;
    }
    let bits = x.to_bits();
    let round = 0x7FFF + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}
