//! A layout's state: what a model remembers from one fire to the next,
//! declared where the layout puts it, as a weight is (`state(shape,
//! dtype)`), and named by its path in the layout (`layers.9.delta`).
//!
//! A state whose first axis is `tokens` (or `tokens // r`, a pool keeping
//! one entry every `r` tokens) grows a row a token; any other is a fixed
//! slab a sequence holds one of. A forward hands the state itself to the ops
//! that read and write it. What the ops say of it (the window an attention
//! reads it through, its head width, the axis its heads or channels lie on)
//! is not declared but read off a first trace of the forward, and the state
//! is then lowered to the cache rows a trace declares: a per-token state to
//! a kv row in the space of its dtype and window, a fixed one to a state
//! row.
//!
//! What the ops say, by op: a grouped-query attention (decode, prefill, a
//! score capture, a sink window) its window and head width, and that the
//! row holds heads the ranks part; a latent attention its window and the
//! latent's width (the rest of the row the rotated dims, the head width); an
//! indexer or a pool the width of the keys it scores; a pool appending its
//! own entries that the row pages alone; a pool gathering from the row that
//! no rank holds a part of it; a recurrence that its slab is cut by head; a
//! convolution whose channels feed heads that its taps are cut by channel.

use std::fmt;

use allocative::Allocative;
use starlark::any::ProvidesStaticType;
use starlark::environment::GlobalsBuilder;
use starlark::starlark_simple_value;
use starlark::values::dict::DictRef;
use starlark::values::list::{ListRef, UnpackList};
use starlark::values::structs::StructRef;
use starlark::values::tuple::TupleRef;
use starlark::values::{Heap, NoSerialize, StarlarkValue, Value as Star, ValueLike};
use starlark_derive::{starlark_module, starlark_value};

use crate::star::values::{DtypeValue, word};
use crate::{Dtype, HybridSpec, KvSpace};
use poem_ir::{Attn, Def, Operands, Operation, Ssm, Trace, ValueId};

/// The token axis a per-token state's shape starts with: one row every
/// `rate` tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ProvidesStaticType, NoSerialize, Allocative)]
pub struct TokensAxis {
    pub rate: u32,
}

starlark_simple_value!(TokensAxis);

impl fmt::Display for TokensAxis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.rate {
            1 => write!(f, "tokens"),
            r => write!(f, "tokens // {r}"),
        }
    }
}

#[starlark_value(type = "tokens")]
impl<'v> StarlarkValue<'v> for TokensAxis {
    fn floor_div(&self, other: Star<'v>, heap: Heap<'v>) -> starlark::Result<Star<'v>> {
        match other.unpack_i32().and_then(|r| u32::try_from(r).ok()) {
            Some(r) if r > 0 => Ok(heap.alloc(TokensAxis {
                rate: self.rate * r,
            })),
            _ => Err(starlark::Error::new_other(anyhow::anyhow!(
                "`{self} // {other}`: a pool keeps a row every whole number of tokens"
            ))),
        }
    }

    fn equals(&self, other: Star<'v>) -> starlark::Result<bool> {
        Ok(other.downcast_ref::<TokensAxis>() == Some(self))
    }
}

/// A block-scaled row format a per-token state's rows are stored in: the
/// dtype of a state whose every row is one such row, a u8 plane of its
/// bytes (`kv_rows.v4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ProvidesStaticType, NoSerialize, Allocative)]
pub enum RowFormat {
    /// V4's rows: 512 dims in 584 bytes.
    V4,
    /// MXFP8 rows: 512 dims in 528 bytes.
    Mx8,
    /// V4.1's compressed rows: 512 dims in 288 bytes.
    Nv4,
    /// V4.1's FP4 index keys: 128 dims in 68 bytes.
    Fp4Keys,
    /// vLLM's standard fp8 rows: a byte a dim, of any width.
    Fp8,
}

impl RowFormat {
    const ALL: [RowFormat; 5] = [Self::V4, Self::Mx8, Self::Nv4, Self::Fp4Keys, Self::Fp8];

    fn word(self) -> &'static str {
        match self {
            Self::V4 => "v4",
            Self::Mx8 => "mx8",
            Self::Nv4 => "nv4",
            Self::Fp4Keys => "fp4_keys",
            Self::Fp8 => "fp8",
        }
    }

    /// The dims a row encodes, where the format fixes them.
    fn dims(self) -> Option<u64> {
        match self {
            Self::V4 | Self::Mx8 | Self::Nv4 => Some(512),
            Self::Fp4Keys => Some(128),
            Self::Fp8 => None,
        }
    }

    /// The format of a row of `dims`.
    fn rows(self, dims: u64) -> anyhow::Result<poem_ir::KvRows> {
        Ok(match self {
            Self::V4 => poem_ir::KvRows::V4,
            Self::Mx8 => poem_ir::KvRows::Mx8,
            Self::Nv4 => poem_ir::KvRows::Nv4,
            Self::Fp4Keys => poem_ir::KvRows::Fp4Keys,
            Self::Fp8 => poem_ir::KvRows::Fp8 {
                width: u32::try_from(dims)?,
            },
        })
    }
}

starlark_simple_value!(RowFormat);

impl fmt::Display for RowFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "kv_rows.{}", self.word())
    }
}

#[starlark_value(type = "kv_rows")]
impl<'v> StarlarkValue<'v> for RowFormat {
    fn equals(&self, other: Star<'v>) -> starlark::Result<bool> {
        Ok(other.downcast_ref::<RowFormat>() == Some(self))
    }
}

/// A state a layout declares: its name (its path, once the layout is
/// named), whether it grows a row every `rate` tokens, the shape of a row
/// (or of the whole slab) and its dtype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateDecl {
    pub name: String,
    pub per_token: Option<u32>,
    /// A slab's axes; a per-token state's axes after `tokens`, where they
    /// are counts.
    pub shape: Vec<u64>,
    /// A per-token state's planes, the dims of each a token.
    pub planes: Vec<u64>,
    pub dtype: Dtype,
    /// The block-scaled format a per-token state's rows are stored in.
    pub rows: Option<RowFormat>,
}

impl StateDecl {
    /// The dims a token's row holds, every plane's.
    fn dims(&self) -> u64 {
        self.planes.iter().sum()
    }
}

/// A state a layout holds.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, Allocative)]
pub struct StateValue(#[allocative(skip)] pub StateDecl);

starlark_simple_value!(StateValue);

impl fmt::Display for StateValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.0;
        let mut axes: Vec<String> = s.shape.iter().map(u64::to_string).collect();
        if let Some(rate) = s.per_token {
            if axes.is_empty() {
                let planes: Vec<String> = s.planes.iter().map(u64::to_string).collect();
                axes.push(format!("[{}]", planes.join(", ")));
            }
            axes.insert(0, TokensAxis { rate }.to_string());
        }
        let dtype = match s.rows {
            Some(rows) => rows.to_string(),
            None => format!("dtype.{}", word(s.dtype)),
        };
        write!(f, "state({:?}, [{}], {dtype})", s.name, axes.join(", "))
    }
}

#[starlark_value(type = "state")]
impl<'v> StarlarkValue<'v> for StateValue {}

/// A positive count, an axis of a state's shape.
fn count(axis: Star<'_>) -> anyhow::Result<u64> {
    <i64 as starlark::values::UnpackValue>::unpack_value(axis)
        .ok()
        .flatten()
        .and_then(|n| u64::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| anyhow::anyhow!("a state's axis is a count, not {axis}"))
}

#[starlark_module]
fn declare(builder: &mut GlobalsBuilder) {
    /// A state of `shape` and `dtype` the model keeps from one fire to the
    /// next, named by where the layout puts it. A shape starting with
    /// `tokens` (or `tokens // r`) grows a row a token (one every `r`): one
    /// plane of the width that follows, as many planes as the axis after
    /// `tokens` counts, each of the width the rest make, or, where a list
    /// follows `tokens`, a plane of each width it lists. Any other shape is a
    /// slab a sequence holds one of. A per-token state's dtype may be a
    /// block-scaled row format (`kv_rows.v4`, ...): its one plane, of the
    /// dims a row of the format encodes, is stored as the format's rows.
    fn state<'v>(
        #[starlark(require = pos)] shape: UnpackList<Star<'v>>,
        #[starlark(require = pos)] dtype: Star<'v>,
    ) -> anyhow::Result<StateValue> {
        let (dtype, rows) = if let Some(d) = dtype.downcast_ref::<DtypeValue>() {
            (d.0, None)
        } else if let Some(r) = dtype.downcast_ref::<RowFormat>() {
            (Dtype::U8, Some(*r))
        } else {
            anyhow::bail!("a state's dtype is a dtype or a `kv_rows` format, not {dtype}");
        };
        let mut axes = shape.items.into_iter();
        let per_token = match axes.clone().next() {
            Some(first) => first.downcast_ref::<TokensAxis>().map(|t| t.rate),
            None => anyhow::bail!("a state has a shape of at least one axis"),
        };
        if per_token.is_some() {
            axes.next();
        }
        let axes: Vec<Star<'v>> = axes.collect();
        if axes
            .iter()
            .any(|a| a.downcast_ref::<TokensAxis>().is_some())
        {
            anyhow::bail!("`tokens` is a state's first axis, or none of its axes");
        }
        let listed = match axes[..] {
            [only] => ListRef::from_value(only)
                .map(|l| l.iter().map(count).collect::<anyhow::Result<Vec<u64>>>())
                .transpose()?,
            _ => None,
        };
        if per_token.is_none() && listed.is_some() {
            anyhow::bail!("a slab's axes are counts; a per-token state lists its planes");
        }
        let shape = match listed {
            Some(_) => Vec::new(),
            None => axes
                .into_iter()
                .map(count)
                .collect::<anyhow::Result<Vec<u64>>>()?,
        };
        let planes = match (per_token, listed, &shape[..]) {
            (None, _, _) => Vec::new(),
            (Some(_), Some(planes), _) => planes,
            (Some(_), None, []) => {
                anyhow::bail!("a per-token state states the width of its rows after `tokens`")
            }
            (Some(_), None, &[width]) => vec![width],
            (Some(_), None, &[count, ref rest @ ..]) => {
                vec![rest.iter().product(); usize::try_from(count)?]
            }
        };
        if per_token.is_some() && planes.is_empty() {
            anyhow::bail!("a per-token state holds at least one plane");
        }
        if let Some(format) = rows {
            let (Some(_), &[dims]) = (per_token, &planes[..]) else {
                anyhow::bail!(
                    "a state of {format} rows grows a row a token, one plane of the dims a \
                     row encodes"
                );
            };
            if let Some(fixed) = format.dims()
                && fixed != dims
            {
                anyhow::bail!("{format} rows encode {fixed} dims, not {dims}");
            }
        }
        Ok(StateValue(StateDecl {
            name: String::new(),
            per_token,
            shape,
            planes,
            dtype,
            rows,
        }))
    }
}

/// `state`, its `tokens` axis and the `kv_rows` formats.
pub(crate) fn globals(builder: &mut GlobalsBuilder) {
    declare(builder);
    builder.set("tokens", TokensAxis { rate: 1 });
    builder.namespace("kv_rows", |ns| {
        for format in RowFormat::ALL {
            ns.set(format.word(), format);
        }
    });
}

/// The states the layout `m` holds, in the order its paths walk them, each
/// once.
pub(crate) fn declared(m: Star<'_>) -> Vec<StateDecl> {
    fn gather(v: Star<'_>, out: &mut Vec<StateDecl>) {
        if let Some(s) = v.downcast_ref::<StateValue>() {
            if !out.iter().any(|o| o.name == s.0.name) {
                out.push(s.0.clone());
            }
        } else if let Some(s) = StructRef::from_value(v) {
            s.iter().for_each(|(_, x)| gather(x, out));
        } else if let Some(l) = ListRef::from_value(v) {
            l.iter().for_each(|x| gather(x, out));
        } else if let Some(t) = TupleRef::from_value(v) {
            t.iter().for_each(|x| gather(x, out));
        } else if let Some(d) = DictRef::from_value(v) {
            d.iter().for_each(|(_, x)| gather(x, out));
        }
    }
    let mut out = Vec::new();
    gather(m, &mut out);
    out
}

/// What the ops of a trace say of one state they read.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Reads {
    /// The window an attention reads it through (`Some(None)`: whole).
    window: Option<Option<u32>>,
    /// The head width an attention reads it as.
    head_dim: Option<u32>,
    /// The head width a pool or an indexer reads it as, where no attention
    /// says one.
    keys_dim: Option<u32>,
    /// A latent attention reads it: its rows the latent of this width, then
    /// the rotated dims, the width of its heads.
    latent: Option<u32>,
    /// A grouped-query attention reads it, its planes holding heads.
    heads: bool,
    /// A pool gathers its entries from the whole row: no rank holds a part.
    whole: bool,
    /// An attention reads it through a plan of the forward's rows.
    planned: bool,
    /// A pool appends entries to it at its own positions: it pages alone.
    pooled: bool,
    /// An indexer's keys: scored, not attended.
    index: bool,
    /// The slab axis its heads or channels lie on: from the end where
    /// `Err`, `Ok` from the front.
    split: Option<Result<u32, u32>>,
}

impl Reads {
    fn window(&mut self, name: &str, window: Option<u32>) -> anyhow::Result<()> {
        match self.window {
            Some(seen) if seen != window => anyhow::bail!(
                "`{name}` is read through a window of {seen:?} tokens by one attention and \
                 {window:?} by another"
            ),
            _ => self.window = Some(window),
        }
        Ok(())
    }

    fn head_dim(&mut self, name: &str, head_dim: u32) -> anyhow::Result<()> {
        match self.head_dim {
            Some(seen) if seen != head_dim => anyhow::bail!(
                "`{name}` is read as heads of {seen} by one attention and of {head_dim} by another"
            ),
            _ => self.head_dim = Some(head_dim),
        }
        Ok(())
    }

    fn latent(&mut self, name: &str, rank: u32) -> anyhow::Result<()> {
        match self.latent {
            Some(seen) if seen != rank => anyhow::bail!(
                "`{name}` is read as a latent of {seen} by one attention and of {rank} by another"
            ),
            _ => self.latent = Some(rank),
        }
        Ok(())
    }

    fn keys_dim(&mut self, head_dim: u32) {
        self.keys_dim.get_or_insert(head_dim);
    }

    fn split(&mut self, name: &str, axis: Result<u32, u32>) -> anyhow::Result<()> {
        match self.split {
            Some(seen) if seen != axis => {
                anyhow::bail!("`{name}` is cut between ranks along two axes by the ops reading it")
            }
            _ => self.split = Some(axis),
        }
        Ok(())
    }

    /// The width of the heads a row of `dims` holds.
    fn heads_of(&self, dims: u64) -> Option<u32> {
        let rotated = self
            .latent
            .and_then(|rank| u32::try_from(dims).ok()?.checked_sub(rank));
        self.head_dim.or(rotated).or(self.keys_dim)
    }
}

/// Whether what `trace` computes from a value reaches, before a projection
/// takes it, an op holding heads (an attention or a recurrence). A
/// convolution whose channels feed heads is cut between ranks as they are;
/// one over the residual stream is held whole.
fn feeds_heads(trace: &Trace) -> impl Fn(ValueId) -> bool + '_ {
    // each value's users: the ops it is an input of, the merges it is an
    // arm of
    let mut users: Vec<Vec<Result<usize, ValueId>>> = vec![Vec::new(); trace.values.len()];
    let mut inputs = Vec::new();
    for (n, node) in trace.nodes.iter().enumerate() {
        inputs.clear();
        node.op.inputs(&mut inputs);
        for v in &inputs {
            users[v.0 as usize].push(Ok(n));
        }
    }
    for (v, decl) in trace.values.iter().enumerate() {
        if let Def::Merge(arms) = &decl.def {
            for (arm, _) in arms {
                users[arm.0 as usize].push(Err(ValueId(v as u32)));
            }
        }
    }
    move |from| {
        let mut seen = vec![false; trace.values.len()];
        let mut todo = vec![from];
        let mut outputs = Vec::new();
        while let Some(v) = todo.pop() {
            if std::mem::replace(&mut seen[v.0 as usize], true) {
                continue;
            }
            for user in &users[v.0 as usize] {
                match *user {
                    Err(merged) => todo.push(merged),
                    Ok(n) => match &trace.nodes[n].op {
                        Operation::Attn(_)
                        | Operation::Ssm(
                            Ssm::GatedDelta { .. }
                            | Ssm::GatedDeltaChunked { .. }
                            | Ssm::Kda { .. }
                            | Ssm::KdaChunked { .. }
                            | Ssm::Mamba2 { .. }
                            | Ssm::Mamba2Chunked { .. },
                        ) => return true,
                        Operation::Linear(_) => {}
                        op => {
                            outputs.clear();
                            op.outputs(&mut outputs);
                            todo.extend(outputs.iter().copied());
                        }
                    },
                }
            }
        }
        false
    }
}

/// What the ops of `trace` say of each cache row it reads, by row.
fn reads(trace: &Trace) -> anyhow::Result<Vec<Reads>> {
    let mut out = vec![Reads::default(); trace.caches.len()];
    let row = |id: ValueId| match trace.values[id.0 as usize].def {
        Def::Cache(index) => Some(index as usize),
        _ => None,
    };
    let name = |index: usize| crate::record::cache_name(&trace.caches[index]).to_string();
    let feeds_heads = feeds_heads(trace);
    for node in &trace.nodes {
        match &node.op {
            Operation::Attn(
                Attn::Decode {
                    cache,
                    window,
                    head_dim,
                    ..
                }
                | Attn::Prefill {
                    cache,
                    window,
                    head_dim,
                    ..
                },
            ) => {
                if let Some(i) = row(*cache) {
                    out[i].window(&name(i), *window)?;
                    out[i].head_dim(&name(i), *head_dim)?;
                    out[i].heads = true;
                    out[i].planned = true;
                }
            }
            Operation::Attn(Attn::ScoreCapture {
                cache, head_dim, ..
            }) => {
                if let Some(i) = row(*cache) {
                    out[i].head_dim(&name(i), *head_dim)?;
                    out[i].heads = true;
                    out[i].planned = true;
                }
            }
            // a sink attention reads its window without a plan
            Operation::Attn(Attn::WindowSink {
                cache,
                window,
                head_dim,
                ..
            }) => {
                if let Some(i) = row(*cache) {
                    out[i].window(&name(i), Some(*window))?;
                    out[i].head_dim(&name(i), *head_dim)?;
                    out[i].heads = true;
                }
            }
            // a latent attention reads its latent and rotated dims whole
            Operation::Attn(
                Attn::MlaDecode {
                    cache,
                    window,
                    kv_lora_rank,
                    ..
                }
                | Attn::MlaPrefill {
                    cache,
                    window,
                    kv_lora_rank,
                    ..
                },
            ) => {
                if let Some(i) = row(*cache) {
                    out[i].window(&name(i), *window)?;
                    out[i].latent(&name(i), *kv_lora_rank)?;
                    out[i].planned = true;
                }
            }
            Operation::Attn(Attn::IndexTopk { keys, head_dim, .. }) => {
                if let Some(i) = row(*keys) {
                    out[i].keys_dim(*head_dim);
                    out[i].index = true;
                }
            }
            Operation::Attn(Attn::IndexKvAppend { keys, .. }) => {
                if let Some(i) = row(*keys) {
                    out[i].index = true;
                }
            }
            Operation::Attn(Attn::PoolLse {
                entries, head_dim, ..
            }) => {
                if let Some(i) = row(*entries) {
                    out[i].keys_dim(*head_dim);
                }
            }
            Operation::Attn(
                Attn::PoolGather {
                    pages, head_dim, ..
                }
                | Attn::PoolStateWrite {
                    pages, head_dim, ..
                },
            ) => {
                if let Some(i) = row(*pages) {
                    out[i].keys_dim(*head_dim);
                    out[i].whole = true;
                }
            }
            Operation::Attn(Attn::PoolKvAppend { pool, .. }) => {
                if let Some(i) = row(*pool) {
                    out[i].pooled = true;
                }
            }
            // a conv's state holds its taps by channel: where the channels
            // feed heads, the ranks cut them, its last axis
            Operation::Ssm(
                Ssm::CausalConv1d { state, y, .. }
                | Ssm::CausalConv1dChunked { state, y, .. }
                | Ssm::ShortConv { state, y, .. }
                | Ssm::ShortConvChunked { state, y, .. },
            ) => {
                if let Some(i) = row(*state)
                    && feeds_heads(*y)
                {
                    out[i].split(&name(i), Err(1))?;
                }
            }
            // a recurrence's state holds a slab a head: the ranks cut the
            // heads, its first axis
            Operation::Ssm(
                Ssm::GatedDelta { state, .. }
                | Ssm::GatedDeltaChunked { state, .. }
                | Ssm::Kda { state, .. }
                | Ssm::KdaChunked { state, .. }
                | Ssm::Mamba2 { state, .. }
                | Ssm::Mamba2Chunked { state, .. },
            ) => {
                if let Some(i) = row(*state) {
                    out[i].split(&name(i), Ok(0))?;
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// The rates whose pools pack their entries a row each: every page size an
/// engine serves at is a multiple of them. A pool keeping a row every `r`
/// tokens holds `1 / r` of a row a token where `r` is one of them (and
/// parts its planes whole), else a whole row a token, at each group's last.
const COMPACT_RATES: [u32; 2] = [2, 4];

/// The planes a per-token state's row holds, a row every `rate` tokens, in
/// elements of its dtype: a compact pool's plane is its rows' share of a
/// token's.
fn planes(s: &StateDecl, rate: u32) -> anyhow::Result<Vec<u64>> {
    let widths: Vec<u64> = match s.rows {
        Some(format) => vec![format.rows(s.dims())?.bytes()],
        None => s.planes.clone(),
    };
    let r = u64::from(rate);
    Ok(
        if COMPACT_RATES.contains(&rate) && widths.iter().all(|w| w % r == 0) {
            widths.iter().map(|w| w / r).collect()
        } else {
            widths
        },
    )
}

/// The cache rows `states` lower to before the forward has been traced:
/// their attentions' heads and windows, their cuts not yet known.
pub(crate) fn provisional(states: &[StateDecl]) -> anyhow::Result<HybridSpec> {
    lower(states, &vec![Reads::default(); states.len()])
}

/// The cache rows `states` lower to, as the ops of `first`, a trace of the
/// forward over [`provisional`]'s rows, read them.
pub(crate) fn lowered(states: &[StateDecl], first: &Trace) -> anyhow::Result<HybridSpec> {
    let by_row = reads(first)?;
    let mut read: Vec<Reads> = states
        .iter()
        .map(|s| {
            first
                .caches
                .iter()
                .position(|row| crate::record::cache_name(row) == s.name)
                .map(|i| by_row[i].clone())
                .unwrap_or_default()
        })
        .collect();
    // A per-token state no op reads the heads of is laid out as the first
    // one of its shape and dtype that one does.
    for i in 0..states.len() {
        let s = &states[i];
        if s.per_token.is_none() || read[i].heads_of(s.dims()).is_some() {
            continue;
        }
        let like = (0..states.len()).find(|&j| {
            let t = &states[j];
            (t.per_token, &t.planes, t.dtype, t.rows) == (s.per_token, &s.planes, s.dtype, s.rows)
                && read[j].heads_of(t.dims()).is_some()
        });
        match like {
            Some(j) => read[i] = read[j].clone(),
            None => anyhow::bail!(
                "no attention reads the per-token state `{}`, nor one of its shape, so the \
                 width of its heads is unknown",
                s.name
            ),
        }
    }
    lower(states, &read)
}

/// What parts per-token rows into spaces: their dtype, window and rate, and
/// what reads them: attentions or an indexer (`Ok(index)`), or, appending
/// its own entries, a pool alone (`Err(the state)`).
type SpaceKey = (Dtype, Option<u32>, u32, Result<bool, usize>);

/// The order spaces are numbered in: those an attention reads through a
/// plan of the forward's rows first (the whole sequence's before a
/// window's), then every other.
fn rank(r: &Reads) -> u8 {
    match (r.planned, r.window.flatten()) {
        (true, None) => 0,
        (true, Some(_)) => 1,
        (false, _) => 2,
    }
}

fn lower(states: &[StateDecl], read: &[Reads]) -> anyhow::Result<HybridSpec> {
    let key = |i: usize, rate: u32| -> SpaceKey {
        let r = &read[i];
        let class = if r.pooled { Err(i) } else { Ok(r.index) };
        (states[i].dtype, r.window.flatten(), rate, class)
    };
    // every space, in the order its first row is declared, then by rank
    let mut keys: Vec<(SpaceKey, u8)> = Vec::new();
    for (i, s) in states.iter().enumerate() {
        if let Some(rate) = s.per_token {
            let k = key(i, rate);
            if !keys.iter().any(|(seen, _)| *seen == k) {
                keys.push((k, rank(&read[i])));
            }
        }
    }
    keys.sort_by_key(|(_, rank)| *rank);
    let mut spec = HybridSpec::new();
    let spaces: Vec<(SpaceKey, KvSpace)> = keys
        .into_iter()
        .map(|(k, _)| {
            let space = match k.1 {
                Some(w) => spec.windowed_kv_space(k.0, w),
                None => spec.kv_space(k.0),
            };
            (k, space)
        })
        .collect();
    for (i, (s, r)) in states.iter().zip(read).enumerate() {
        match s.per_token {
            Some(rate) => {
                let k = key(i, rate);
                let space = spaces
                    .iter()
                    .find_map(|(seen, space)| (*seen == k).then_some(*space))
                    .expect("every per-token state's space is made");
                let head_dim = r.heads_of(s.dims()).unwrap_or(0);
                let mut row = spec.kv(space, s.name.clone(), planes(s, rate)?, head_dim);
                if let Some(format) = s.rows {
                    row.rows(format.rows(s.dims())?);
                }
                // One plane of a single head, its key and its value both, is
                // no head the ranks could part.
                let parted = s.planes.len() > 1 || s.dims() > u64::from(head_dim);
                if r.heads && !r.whole && parted {
                    row.heads();
                }
            }
            None => {
                let row = spec.state(s.name.clone(), s.shape.iter().copied(), s.dtype);
                match r.split {
                    Some(Ok(axis)) => row.split(axis),
                    Some(Err(from_end)) => {
                        let Some(axis) = (s.shape.len() as u32).checked_sub(from_end) else {
                            anyhow::bail!(
                                "`{}`, a slab of {:?}, has too few axes for the op reading it",
                                s.name,
                                s.shape
                            );
                        };
                        row.split(axis);
                    }
                    None => {}
                }
            }
        }
    }
    Ok(spec)
}
