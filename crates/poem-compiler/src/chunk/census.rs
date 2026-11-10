//! What a trace remembers, read off today's IR: each cache row's bytes, its
//! retention, the layer that reads it and the axis its readers let a piece
//! cut, and the rs buffer the deferred-commit recurrences carry.

use std::collections::HashMap;

use poem_ir::{Attn, CacheRow, Conv, Def, Dim, Gather, Operands, Operation, Shard, Ssm, Trace, Ty};

use super::{ChunkError, RowId};
use crate::arena::elem_bytes;

/// The tokens the rs buffer holds a recurrent layer: the most a lane carries
/// unfolded past its state (a fire's speculative rows before they commit).
pub const UNFOLDED_ROWS: u32 = 16;

/// The rs buffer's element: the engine buffers its planes in bf16.
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
    /// Attention rows and the rs buffer: tokens, axis 0.
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
    /// A window ring folded in: the tokens it holds.
    pub ring_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct Census {
    pub tokens: Vec<TokenRow>,
    pub fixed: Vec<FixedRow>,
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
                ..
            } => {
                let element = element(name, *dtype)?;
                out.tokens.push(TokenRow {
                    id,
                    layer: read.layer,
                    planes: planes.iter().map(|w| w * element).collect(),
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
                let element = if state_f32 && *dtype != poem_ir::Dtype::I32 {
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
                    ring_tokens: None,
                });
            }
        }
    }
    out.fixed.extend(rs_rows(trace, &readers)?);
    Ok(out)
}

fn element(name: &str, dtype: poem_ir::Dtype) -> Result<u64, ChunkError> {
    elem_bytes(dtype).ok_or_else(|| ChunkError::Unsized {
        row: name.to_string(),
        dtype,
    })
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

fn conv_input(op: &Operation) -> Option<poem_ir::ValueId> {
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

/// The rs buffer: a `[UNFOLDED_ROWS, width]` row a chunked recurrence whose
/// commit is deferred, `width` the planes it replays (as the engine's
/// `store::rs::read` lays them): the conv's input, then the gate prep's
/// `[b | a]` (gated delta) or the forget and beta projections (KDA).
fn rs_rows(trace: &Trace, readers: &[Read]) -> Result<Vec<FixedRow>, ChunkError> {
    let mut defined: HashMap<u32, usize> = HashMap::new();
    let mut outs = Vec::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        outs.clear();
        node.op.outputs(&mut outs);
        for out in &outs {
            defined.insert(out.0, at);
        }
    }
    let op_of = |id: poem_ir::ValueId| defined.get(&id.0).map(|n| &trace.nodes[*n].op);
    let conv_x = |id| match op_of(id) {
        Some(Operation::Ssm(Ssm::CausalConv1dChunked { x, .. })) => Some(*x),
        _ => None,
    };
    let mut rows = Vec::new();
    for node in &trace.nodes {
        let (planes, state) = match &node.op {
            Operation::Ssm(Ssm::GatedDeltaChunked {
                qkv, gates, state, ..
            }) => {
                let ba = match op_of(*gates) {
                    Some(Operation::Ssm(Ssm::GdnGates { ba, .. })) => Some(*ba),
                    _ => None,
                };
                (conv_x(*qkv).zip(ba).map(|(x, ba)| vec![x, ba]), *state)
            }
            Operation::Ssm(Ssm::KdaChunked {
                mixed, f, b, state, ..
            }) => (conv_x(*mixed).map(|x| vec![x, *f, *b]), *state),
            _ => continue,
        };
        let Some(planes) = planes else {
            continue;
        };
        let mut width = 0u64;
        for plane in planes {
            width += width_of(trace, plane).ok_or(ChunkError::UnsizedRs)?;
        }
        // The rs planes are cut as the recurrence's own state is.
        let shard = match trace.values.get(state.0 as usize).map(|d| &d.def) {
            Some(Def::Cache(row)) => match trace.caches.get(*row as usize) {
                Some(CacheRow::State { shard, .. }) => shard.clone(),
                _ => Shard::Replicated,
            },
            _ => Shard::Replicated,
        };
        let shard = match shard {
            Shard::Replicated => Shard::Replicated,
            _ => Shard::Cut {
                axis: 1,
                segments: vec![width],
                heads: Vec::new(),
            },
        };
        let layer = match trace.values.get(state.0 as usize).map(|d| &d.def) {
            Some(Def::Cache(row)) => readers.get(*row as usize).and_then(|r| r.layer),
            _ => None,
        }
        .or(node.layer);
        let per_token = width * RS_ELEMENT;
        rows.push(FixedRow {
            id: RowId::Rs(u32::try_from(rows.len()).unwrap_or(u32::MAX)),
            layer,
            bytes: u64::from(UNFOLDED_ROWS) * per_token,
            atom: Atom {
                contract: Contract::Tokens,
                axis: 0,
                extent: u64::from(UNFOLDED_ROWS),
                bytes: per_token,
                grain: 1,
            },
            shard,
            ring_tokens: None,
        });
    }
    Ok(rows)
}

/// The elements a row of `value` holds (its dims past the first).
fn width_of(trace: &Trace, value: poem_ir::ValueId) -> Option<u64> {
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
