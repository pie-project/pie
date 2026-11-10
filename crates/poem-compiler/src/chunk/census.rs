//! What a trace remembers, read off today's IR: each cache row's bytes, its
//! retention, the layer that reads it and the axis its readers let a piece
//! cut, and the rs buffer the deferred-commit recurrences and the n-gram id
//! windows carry: a per-token row each, laid in the rs list.

use std::collections::{BTreeMap, HashMap};

use poem_ir::{
    Attn, CacheRow, Conv, Def, Dim, Dtype, Gather, KvRows, Operands, Operation, Shard, Ssm, Trace,
    Ty, ValueId,
};

use super::{ChunkError, RowId};
use crate::arena::elem_bytes;

/// The rs buffer's element: the engine buffers a recurrence's planes in
/// bf16 (token ids in their own i32).
const RS_ELEMENT: u64 = 2;

/// The most ranks a deployment is cut across ([`crate::catalog::RANKS`]): a
/// row the ranks cut on its atom axis is cut by a piece only at multiples of
/// this many atoms, so each rank's share of a piece is whole atoms.
pub const MOST_RANKS: u64 = 8;

/// Which axis of a state row a piece may cut, by the op kinds that read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Contract {
    /// A recurrent state (GDN, KDA, Mamba-2): heads, axis 0.
    Heads,
    /// A conv's carried taps: the channel axis.
    Channels,
    /// A conv3d frame cache: frame positions, axis 0.
    Frames,
    /// Attention rows: tokens, axis 0.
    Tokens,
    /// Read whole (n-gram id windows, the replay floor, anything unknown).
    Whole,
}

/// A row's cuttable axis: `extent` atoms of `bytes` each along `axis`; a piece
/// cuts it only at multiples of `grain` atoms. `extent == 1`: uncuttable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Atom {
    pub contract: Contract,
    pub axis: u32,
    pub extent: u64,
    pub bytes: u64,
    pub grain: u64,
}

impl Atom {
    fn whole(bytes: u64) -> Atom {
        Atom {
            contract: Contract::Whole,
            axis: 0,
            extent: 1,
            bytes,
            grain: 1,
        }
    }
}

/// A per-token row (a kv row): its bytes a token, plane by plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenRow {
    pub id: RowId,
    pub layer: Option<u32>,
    pub planes: Vec<u64>,
    pub window: Option<u32>,
    /// The positions a compressed pool folds into one entry (1: none).
    pub ratio: u32,
    pub shard: Shard,
}

impl TokenRow {
    #[must_use]
    pub fn per_token(&self) -> u64 {
        self.planes.iter().sum()
    }
}

/// A fixed row of a working set's record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedRow {
    pub id: RowId,
    pub layer: Option<u32>,
    pub bytes: u64,
    pub atom: Atom,
    pub shard: Shard,
}

/// What one rs row buffers, as an engine lays it in an rs-list slot of
/// `tokens` tokens: `planes` one after another, plane `i` at
/// `tokens x (the bytes a token of planes 0..i)` past the row's offset, token
/// `t` of a plane `t x` its bytes a token past the plane's start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsRow {
    /// The state cache row the buffered tokens commit into: the recurrence's
    /// own state, or the n-gram id window.
    pub state: Option<u32>,
    pub planes: Vec<RsPlane>,
}

impl RsRow {
    /// The bytes a token takes across every plane.
    #[must_use]
    pub fn per_token(&self) -> u64 {
        self.planes.iter().map(|p| p.bytes).sum()
    }

    /// Where plane `i` starts past the row's offset, in a slot of `tokens`.
    #[must_use]
    pub fn plane_at(&self, i: usize, tokens: u64) -> u64 {
        tokens * self.planes[..i].iter().map(|p| p.bytes).sum::<u64>()
    }
}

/// A plane of an rs row: the trace value whose token rows it buffers, and
/// their bytes a token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RsPlane {
    pub value: ValueId,
    pub bytes: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Census {
    pub tokens: Vec<TokenRow>,
    pub fixed: Vec<FixedRow>,
    /// The rs buffer's rows: per-token rows of the rs list, a token's bytes
    /// every plane's.
    pub rs_rows: Vec<TokenRow>,
    /// What each of `rs_rows` buffers.
    pub rs: BTreeMap<RowId, RsRow>,
}

/// Reads every row `trace` remembers; with `state_f32`, a state row takes
/// four bytes an element but an integer one (the backend widens it).
pub fn census(trace: &Trace, state_f32: bool) -> Result<Census, ChunkError> {
    let readers = readers(trace);
    let mut out = Census::default();
    for (at, row) in trace.caches.iter().enumerate() {
        let id = RowId::Cache(u32::try_from(at).unwrap_or(u32::MAX));
        let read = &readers[at];
        match row {
            CacheRow::Kv {
                name,
                planes,
                dtype,
                window,
                shard,
                head_dim,
                rows,
                ..
            } => {
                let planes = planes
                    .iter()
                    .map(|&width| stored(name, width, *dtype, *rows, *head_dim))
                    .collect::<Result<Vec<_>, _>>()?;
                out.tokens.push(TokenRow {
                    id,
                    layer: read.layer,
                    planes,
                    window: *window,
                    ratio: read.ratio.max(1),
                    shard: shard.clone(),
                });
            }
            CacheRow::State {
                name,
                slab,
                dtype,
                shard,
            } => {
                let element = if state_f32 && *dtype != Dtype::I32 {
                    4
                } else {
                    element(name, *dtype)?
                };
                let bytes = slab.iter().product::<u64>() * element;
                let atom = atom(slab, bytes, read, shard);
                out.fixed.push(FixedRow {
                    id,
                    layer: read.layer,
                    bytes,
                    atom,
                    shard: shard.clone(),
                });
            }
        }
    }
    for (row, rs) in rs_rows(trace, &readers)? {
        out.rs.insert(row.id, rs);
        out.rs_rows.push(row);
    }
    // A compressor keeps a token's projection and score beside the kv row
    // its gather keys by its pages, in the same pages.
    for (row, width) in pooled(trace) {
        let Some(keyed) = out
            .tokens
            .iter()
            .find(|r| r.id == RowId::Cache(row))
            .cloned()
        else {
            continue;
        };
        out.tokens.push(TokenRow {
            id: RowId::Pool(row),
            planes: vec![width * POOL_ELEMENT; 2],
            ratio: 1,
            ..keyed
        });
    }
    Ok(out)
}

/// The bytes a token of a kv plane `width` wide takes as the engines store
/// it: `width` elements of `dtype`; a block-scaled row's own bytes (the
/// plane is declared in them); a `KvU4` head of `head_dim` packed to
/// `head_dim / 2 + 2` bytes (the codec's 256-wide anchor where the trace
/// states no head_dim); another packed format by its blocks.
fn stored(
    name: &str,
    width: u64,
    dtype: Dtype,
    rows: Option<KvRows>,
    head_dim: u32,
) -> Result<u64, ChunkError> {
    if rows.is_some() {
        return Ok(width);
    }
    if let Some(element) = elem_bytes(dtype) {
        return Ok(width * element);
    }
    let block = u64::from(head_dim);
    if dtype == Dtype::KvU4 && block != 0 && width.is_multiple_of(block) {
        return Ok(width / block * (block / 2 + 2));
    }
    u32::try_from(width)
        .ok()
        .and_then(|k| dtype.row_bytes(k))
        .ok_or_else(|| ChunkError::Unsized {
            row: name.to_string(),
            dtype,
        })
}

fn element(name: &str, dtype: Dtype) -> Result<u64, ChunkError> {
    elem_bytes(dtype).ok_or_else(|| ChunkError::Unsized {
        row: name.to_string(),
        dtype,
    })
}

/// The compressor state's element: bf16.
const POOL_ELEMENT: u64 = 2;

/// Each cache row a pool gather keys its compressor state by, and the
/// state's width a token: the gather's head width, twice it where a ratio-4
/// compressor overlaps two groups.
#[must_use]
pub fn pooled(trace: &Trace) -> Vec<(u32, u64)> {
    let mut rows: Vec<(u32, u64)> = Vec::new();
    for node in &trace.nodes {
        // A gather, or one the compressor's fused op holds.
        let gather = match &node.op {
            Operation::Attn(a) => a,
            Operation::Fused(poem_ir::Fused::PoolCompress { gather, .. }) => &**gather,
            _ => continue,
        };
        let Attn::PoolGather {
            pages,
            head_dim,
            ratio,
            ..
        } = gather
        else {
            continue;
        };
        let Some(Def::Cache(row)) = trace.values.get(pages.0 as usize).map(|v| &v.def) else {
            continue;
        };
        let width = if *ratio == 4 { 2 } else { 1 } * u64::from(*head_dim);
        if width == 0 {
            continue;
        }
        match rows.iter_mut().find(|(held, _)| *held == *row) {
            Some((_, held)) => *held = (*held).max(width),
            None => rows.push((*row, width)),
        }
    }
    rows
}

/// What the ops reading one row say of it.
#[derive(Debug, Clone, Default)]
struct Read {
    layer: Option<u32>,
    contracts: Vec<Contract>,
    /// The channels a conv reading the row convolves.
    channels: Option<u64>,
    ratio: u32,
}

fn readers(trace: &Trace) -> Vec<Read> {
    let mut out = vec![Read::default(); trace.caches.len()];
    let mut inputs = Vec::new();
    for node in &trace.nodes {
        inputs.clear();
        node.op.inputs(&mut inputs);
        for value in &inputs {
            let Some(Def::Cache(row)) = trace.values.get(value.0 as usize).map(|d| &d.def) else {
                continue;
            };
            let Some(read) = out.get_mut(*row as usize) else {
                continue;
            };
            if read.layer.is_none() {
                read.layer = node.layer;
            }
            if let Some(ratio) = ratio_of(&node.op) {
                read.ratio = lcm(read.ratio.max(1), ratio.max(1));
            }
            // A fused op restates its parts' reads; the trace this pass
            // reads is unfused, and a fused one keeps its parts' contracts.
            let Some(contract) = contract_of(&node.op) else {
                continue;
            };
            if !read.contracts.contains(&contract) {
                read.contracts.push(contract);
            }
            if contract == Contract::Channels
                && let Some(x) = conv_input(&node.op)
            {
                read.channels = width_of(trace, x);
            }
        }
    }
    out
}

/// The atom contract of an op kind over the state rows it reads.
fn contract_of(op: &Operation) -> Option<Contract> {
    Some(match op {
        Operation::Ssm(
            Ssm::GatedDelta { .. }
            | Ssm::GatedDeltaChunked { .. }
            | Ssm::Kda { .. }
            | Ssm::KdaChunked { .. }
            | Ssm::Mamba2 { .. }
            | Ssm::Mamba2Chunked { .. },
        ) => Contract::Heads,
        Operation::Ssm(
            Ssm::CausalConv1d { .. }
            | Ssm::CausalConv1dChunked { .. }
            | Ssm::ShortConv { .. }
            | Ssm::ShortConvChunked { .. },
        ) => Contract::Channels,
        Operation::Conv(Conv::Conv3d { .. } | Conv::CacheStore { .. }) => Contract::Frames,
        Operation::Gather(Gather::NgramIds { .. } | Gather::NgramIdsChunked { .. }) => {
            Contract::Whole
        }
        Operation::Attn(Attn::ReplayFloor { .. }) => Contract::Whole,
        Operation::Attn(_) => Contract::Tokens,
        Operation::Fused(_) => return None,
        _ => Contract::Whole,
    })
}

fn conv_input(op: &Operation) -> Option<ValueId> {
    match op {
        Operation::Ssm(
            Ssm::CausalConv1d { x, .. }
            | Ssm::CausalConv1dChunked { x, .. }
            | Ssm::ShortConv { x, .. }
            | Ssm::ShortConvChunked { x, .. },
        ) => Some(*x),
        _ => None,
    }
}

/// The positions a compressed pool's reader folds into one entry.
fn ratio_of(op: &Operation) -> Option<u32> {
    match op {
        Operation::Attn(attn) => attn_ratio(attn),
        Operation::Fused(poem_ir::Fused::PoolAttention { pool, .. }) => attn_ratio(pool),
        _ => None,
    }
}

fn attn_ratio(attn: &Attn) -> Option<u32> {
    match attn {
        Attn::PoolLse { ratio, .. }
        | Attn::IndexTopk { ratio, .. }
        | Attn::IndexBlockMean { ratio, .. } => Some(*ratio),
        _ => None,
    }
}

/// The cuttable axis of a state row, if every reader agrees on one.
fn atom(slab: &[u64], bytes: u64, read: &Read, shard: &Shard) -> Atom {
    let [contract] = read.contracts.as_slice() else {
        return Atom::whole(bytes);
    };
    let axis = match contract {
        Contract::Heads | Contract::Frames | Contract::Tokens => Some(0),
        // The axis as wide as the conv's input, the last such.
        Contract::Channels => read
            .channels
            .and_then(|c| slab.iter().rposition(|&e| e == c))
            .or_else(|| slab.len().checked_sub(1)),
        Contract::Whole => None,
    };
    let Some(axis) = axis.filter(|_| slab.len() > 1) else {
        return Atom::whole(bytes);
    };
    let extent = slab[axis].max(1);
    let grain = match shard {
        Shard::Cut { axis: cut, .. } if *cut as usize == axis => gcd(extent, MOST_RANKS),
        _ => 1,
    };
    Atom {
        contract: *contract,
        axis: u32::try_from(axis).unwrap_or(0),
        extent,
        bytes: bytes / extent,
        grain,
    }
}

/// The rs buffer: a per-token row of `width` a chunked recurrence whose
/// commit is deferred, `width` the planes it replays (as the engines'
/// `store::rs::read` lay them): the conv's input, then the gate prep's
/// `[b | a]` (gated delta) or the forget and beta projections (KDA); then a
/// row an n-gram id window, its unfolded token ids (as Metal's rs layout
/// buffers them, in their own dtype). `RowId::Rs(n)`: the recurrences in
/// node order, then the windows in the order their first reader stands.
fn rs_rows(trace: &Trace, readers: &[Read]) -> Result<Vec<(TokenRow, RsRow)>, ChunkError> {
    let mut defined: HashMap<u32, usize> = HashMap::new();
    let mut outs = Vec::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        outs.clear();
        node.op.outputs(&mut outs);
        for out in &outs {
            defined.insert(out.0, at);
        }
    }
    let op_of = |id: ValueId| defined.get(&id.0).map(|n| &trace.nodes[*n].op);
    let conv_x = |id| match op_of(id) {
        Some(Operation::Ssm(Ssm::CausalConv1dChunked { x, .. })) => Some(*x),
        _ => None,
    };
    let cache = |id: ValueId| match trace.values.get(id.0 as usize).map(|d| &d.def) {
        Some(Def::Cache(row)) => Some(*row),
        _ => None,
    };
    let mut recurrences = Vec::new();
    let mut windows: Vec<(u32, ValueId, Option<u32>)> = Vec::new();
    for node in &trace.nodes {
        match &node.op {
            Operation::Ssm(Ssm::GatedDeltaChunked {
                qkv, gates, state, ..
            }) => {
                let ba = match op_of(*gates) {
                    Some(Operation::Ssm(Ssm::GdnGates { ba, .. })) => Some(*ba),
                    _ => None,
                };
                if let Some((x, ba)) = conv_x(*qkv).zip(ba) {
                    recurrences.push((vec![x, ba], *state, node.layer));
                }
            }
            Operation::Ssm(Ssm::KdaChunked {
                mixed, f, b, state, ..
            }) => {
                if let Some(x) = conv_x(*mixed) {
                    recurrences.push((vec![x, *f, *b], *state, node.layer));
                }
            }
            Operation::Gather(
                Gather::NgramIds { ids, state, .. } | Gather::NgramIdsChunked { ids, state, .. },
            ) => {
                if let Some(row) = cache(*state)
                    && !windows.iter().any(|(seen, ..)| *seen == row)
                {
                    windows.push((row, *ids, node.layer));
                }
            }
            _ => {}
        }
    }
    let mut rows = Vec::new();
    let mut push = |planes: Vec<RsPlane>, state: Option<u32>, layer: Option<u32>, shard| {
        let id = RowId::Rs(u32::try_from(rows.len()).unwrap_or(u32::MAX));
        let layer = state
            .and_then(|row| readers.get(row as usize))
            .and_then(|r| r.layer)
            .or(layer);
        rows.push((
            TokenRow {
                id,
                layer,
                planes: planes.iter().map(|p| p.bytes).collect(),
                window: None,
                ratio: 1,
                shard,
            },
            RsRow { state, planes },
        ));
    };
    for (values, state, layer) in recurrences {
        let mut planes = Vec::with_capacity(values.len());
        let mut width = 0u64;
        for value in values {
            let w = width_of(trace, value).ok_or(ChunkError::UnsizedRs)?;
            width += w;
            planes.push(RsPlane {
                value,
                bytes: w * RS_ELEMENT,
            });
        }
        let row = cache(state);
        // The rs planes are cut as the recurrence's own state is: as many
        // parts as its heads let the ranks cut.
        let shard = match row.and_then(|row| trace.caches.get(row as usize)) {
            Some(CacheRow::State { shard, .. }) if *shard != Shard::Replicated => Shard::Cut {
                axis: 1,
                segments: vec![width],
                heads: match shard {
                    Shard::Cut { heads, .. } => heads.clone(),
                    _ => Vec::new(),
                },
            },
            _ => Shard::Replicated,
        };
        push(planes, row, layer, shard);
    }
    for (row, ids, layer) in windows {
        let element = match trace.values.get(ids.0 as usize).map(|d| &d.ty) {
            Some(Ty::Tensor { dtype, .. }) => elem_bytes(*dtype),
            _ => None,
        };
        let bytes = width_of(trace, ids)
            .zip(element)
            .map(|(w, e)| w * e)
            .ok_or(ChunkError::UnsizedRs)?;
        push(
            vec![RsPlane { value: ids, bytes }],
            Some(row),
            layer,
            Shard::Replicated,
        );
    }
    Ok(rows)
}

/// The elements a row of `value` holds (its dims past the first).
fn width_of(trace: &Trace, value: ValueId) -> Option<u64> {
    let Ty::Tensor { shape, .. } = &trace.values.get(value.0 as usize)?.ty else {
        return None;
    };
    shape.iter().skip(1).try_fold(1u64, |width, dim| match dim {
        Dim::Const(n) => Some(width * n),
        _ => None,
    })
}

#[must_use]
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[must_use]
pub fn lcm(a: u32, b: u32) -> u32 {
    let g = gcd(u64::from(a), u64::from(b)).max(1);
    u32::try_from(u64::from(a) / g * u64::from(b)).unwrap_or(u32::MAX)
}
