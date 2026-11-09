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
use poem_ir::{Attn, Def, Operation, Ssm, Trace};

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

/// A state a layout declares: its name (its path, once the layout is
/// named), whether it grows a row every `rate` tokens, the shape of a row
/// (or of the whole slab) and its dtype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateDecl {
    pub name: String,
    pub per_token: Option<u32>,
    pub shape: Vec<u64>,
    pub dtype: Dtype,
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
            axes.insert(0, TokensAxis { rate }.to_string());
        }
        write!(
            f,
            "state({:?}, [{}], dtype.{})",
            s.name,
            axes.join(", "),
            word(s.dtype)
        )
    }
}

#[starlark_value(type = "state")]
impl<'v> StarlarkValue<'v> for StateValue {}

#[starlark_module]
fn declare(builder: &mut GlobalsBuilder) {
    /// A state of `shape` and `dtype` the model keeps from one fire to the
    /// next, named by where the layout puts it. A shape starting with
    /// `tokens` (or `tokens // r`) grows a row a token (one every `r`): one
    /// plane of the width that follows, or as many planes as the axis after
    /// `tokens` counts, each of the width the rest make. Any other shape is a
    /// slab a sequence holds one of.
    fn state<'v>(
        #[starlark(require = pos)] shape: UnpackList<Star<'v>>,
        #[starlark(require = pos)] dtype: &DtypeValue,
    ) -> anyhow::Result<StateValue> {
        let mut axes = shape.items.into_iter();
        let per_token = match axes.clone().next() {
            Some(first) => first.downcast_ref::<TokensAxis>().map(|t| t.rate),
            None => anyhow::bail!("a state has a shape of at least one axis"),
        };
        if per_token.is_some() {
            axes.next();
        }
        let shape = axes
            .map(|axis| {
                if axis.downcast_ref::<TokensAxis>().is_some() {
                    anyhow::bail!("`tokens` is a state's first axis, or none of its axes");
                }
                <i64 as starlark::values::UnpackValue>::unpack_value(axis)
                    .ok()
                    .flatten()
                    .and_then(|n| u64::try_from(n).ok())
                    .filter(|&n| n > 0)
                    .ok_or_else(|| anyhow::anyhow!("a state's axis is a count, not {axis}"))
            })
            .collect::<anyhow::Result<Vec<u64>>>()?;
        if per_token.is_some() && shape.is_empty() {
            anyhow::bail!("a per-token state states the width of its rows after `tokens`");
        }
        Ok(StateValue(StateDecl {
            name: String::new(),
            per_token,
            shape,
            dtype: dtype.0,
        }))
    }
}

/// `state` and its `tokens` axis.
pub(crate) fn globals(builder: &mut GlobalsBuilder) {
    declare(builder);
    builder.set("tokens", TokensAxis { rate: 1 });
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
    head_dim: Option<u32>,
    /// A grouped-query attention reads it, its planes holding heads.
    heads: bool,
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
        self.heads = true;
        Ok(())
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
}

/// What the ops of `trace` say of each cache row it reads, by row.
fn reads(trace: &Trace) -> anyhow::Result<Vec<Reads>> {
    let mut out = vec![Reads::default(); trace.caches.len()];
    let row = |id: poem_ir::ValueId| match trace.values[id.0 as usize].def {
        Def::Cache(index) => Some(index as usize),
        _ => None,
    };
    let name = |index: usize| crate::record::cache_name(&trace.caches[index]).to_string();
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
                }
            }
            Operation::Attn(Attn::ScoreCapture {
                cache, head_dim, ..
            }) => {
                if let Some(i) = row(*cache) {
                    out[i].head_dim(&name(i), *head_dim)?;
                }
            }
            // a conv's state holds its taps by channel: the ranks cut the
            // channels, its last axis
            Operation::Ssm(
                Ssm::CausalConv1d { state, .. }
                | Ssm::CausalConv1dChunked { state, .. }
                | Ssm::ShortConv { state, .. }
                | Ssm::ShortConvChunked { state, .. },
            ) => {
                if let Some(i) = row(*state) {
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

/// The planes a per-token state's row of `shape` holds, a row every
/// `rate` tokens: a pool's plane is its rows' share of a token's.
fn planes(s: &StateDecl, rate: u32) -> anyhow::Result<Vec<u64>> {
    let (count, width) = match s.shape[..] {
        [width] => (1, width),
        [count, ref rest @ ..] => (count, rest.iter().product()),
        [] => unreachable!("a per-token state has a row width"),
    };
    let rate = u64::from(rate);
    if width % rate != 0 {
        anyhow::bail!(
            "`{}` keeps a row every {rate} tokens, and its planes of {width} part into no \
             whole share a token",
            s.name
        );
    }
    Ok(vec![width / rate; count as usize])
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
    // A per-token state no attention reads is laid out as the first one of
    // its shape and dtype that one does.
    for i in 0..states.len() {
        let s = &states[i];
        if s.per_token.is_none() || read[i].head_dim.is_some() {
            continue;
        }
        let like = (0..states.len()).find(|&j| {
            let t = &states[j];
            (t.per_token, &t.shape, t.dtype) == (s.per_token, &s.shape, s.dtype)
                && read[j].head_dim.is_some()
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

/// What parts per-token rows into spaces: their dtype, window and rate.
type SpaceKey = (Dtype, Option<u32>, u32);

fn lower(states: &[StateDecl], read: &[Reads]) -> anyhow::Result<HybridSpec> {
    let mut spec = HybridSpec::new();
    let mut spaces: Vec<(SpaceKey, KvSpace)> = Vec::new();
    for (s, r) in states.iter().zip(read) {
        match s.per_token {
            Some(rate) => {
                let window = r.window.flatten();
                let key = (s.dtype, window, rate);
                let space = match spaces.iter().find(|(k, _)| *k == key) {
                    Some((_, space)) => *space,
                    None => {
                        let space = match window {
                            Some(w) => spec.windowed_kv_space(s.dtype, w),
                            None => spec.kv_space(s.dtype),
                        };
                        spaces.push((key, space));
                        space
                    }
                };
                let planes = planes(s, rate)?;
                let row = spec.kv(space, s.name.clone(), planes, r.head_dim.unwrap_or(0));
                if r.heads {
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
