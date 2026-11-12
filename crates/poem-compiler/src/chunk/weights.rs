//! The weights a chunk pool may hold as read-only **weight slots**: routed
//! experts and token-id-gathered table rows, as [`poem_ir::residency`]
//! classifies them by the ops that read them.
//!
//! An **expert** slot holds one expert of one MoE layer: its plane of every
//! bank the layer's selects read (gate/up and down, and each one's scales,
//! biases and per-expert scales), packed back to back. A layer's banks share
//! their experts, so an expert's whole FFN streams as one atom. A slot larger
//! than the chunk is cut where the profile lets it ([`ChunkProfile::cut_experts`],
//! on-device Metal) into equal **pieces along each bank's output axis**
//! (axis 1 of an `[experts, N, ..]` plane, the rows of its product): piece `j`
//! holds rows `j * ceil(N / k) ..` of every plane, so each piece is a
//! self-contained slice of the expert's products; a plane of no output axis
//! (one scale an expert) rides in the first piece. No piece is smaller than
//! [`MIN_PIECE`]. Where the profile does not cut, the chunk is never smaller
//! than an expert.
//!
//! A **rows** slot holds a block of a table's rows: tables gathered alike (the
//! same row count, ids a token and hashing) share one atom, a token id's row
//! of each, so one id's rows land together; a slot holds the fewest ids
//! whose rows fill an alignment.
//!
//! The **floor** is what the cache never drops below: for experts, `top_k`
//! experts of the current layer and of the next (the double-buffered
//! staging); for rows, the rows one fire of the budget's tokens may read.
//!
//! [`ChunkProfile::cut_experts`]: super::ChunkProfile::cut_experts

use std::collections::BTreeMap;

use poem_ir::residency::{Residency, leads, residency};
use poem_ir::{Dtype, Param, Trace, ValueId};

use super::{MIN_PIECE, align_up};
use crate::arena::elem_bytes;

/// What a weight slot holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WeightKind {
    /// One routed expert of an MoE layer (or a piece of one).
    Expert,
    /// A block of token ids' rows of the tables gathered alike.
    Rows,
}

/// The planes one atom spans: a layer's banks (experts), or the tables a
/// token id reads (rows).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightBank {
    /// The layer that reads it, where the trace states one.
    pub layer: Option<u32>,
    /// Atoms along axis 0: experts, or rows.
    pub atoms: u64,
    /// Every plane (param index) the atom spans, companions included.
    pub params: Vec<u32>,
}

/// One weight slot shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightSlot {
    pub kind: WeightKind,
    /// One atom's bytes over every plane: an expert of a layer, or a token
    /// id's rows of every table in the group.
    pub atom_bytes: u64,
    /// Atoms a slot holds: 1 for an expert, the ids of a row block.
    pub atoms_per_slot: u64,
    /// Slots an atom is cut into along its banks' output axis (1: whole).
    pub pieces: u32,
    /// A slot's bytes, aligned.
    pub bytes: u64,
    /// Slots that hold every atom of every bank.
    pub slots: u64,
    /// Slots the floor holds.
    pub floor: u64,
    /// The banks (one a layer) that share this shape.
    pub banks: Vec<WeightBank>,
}

impl WeightSlot {
    /// The slots of this shape a chunk of `chunk` bytes holds.
    #[must_use]
    pub fn per_chunk(&self, chunk: u64) -> u64 {
        chunk / self.bytes.max(1)
    }

    /// The bytes every atom takes held.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.slots * self.bytes
    }

    /// The chunks every atom takes.
    #[must_use]
    pub fn chunks(&self, chunk: u64) -> u64 {
        self.slots.div_ceil(self.per_chunk(chunk).max(1))
    }
}

/// One plane of an expert atom, for cutting it on its output axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Plane {
    /// Output rows (axis 1); 0 for a plane of none.
    rows: u64,
    /// The plane's bytes an atom.
    bytes: u64,
}

/// The output rows a piece of a cut quantized expert holds a multiple of:
/// the widest column tile the on-device Metal routed arms read from one
/// expert's rows at once (`kernels_metal::linear::moe::BankTable::ROWS`).
/// A piece of an expert whose layer holds a packed plane spans
/// [`piece_rows`] rows; a bf16 layer's planes are read a row at a time and
/// cut at `ceil(N / k)`.
pub const PIECE_ROW_GRAIN: u64 = 128;

/// The output rows piece `j` of `k` pieces holds of a plane of `rows`
/// rows, from `j * piece_rows(..)` on: `ceil(rows / k)`, rounded up to
/// `grain` where cut (the last pieces may hold fewer, or none).
#[must_use]
pub fn piece_rows(rows: u64, k: u64, grain: u64) -> u64 {
    let rows_a_piece = rows.div_ceil(k.max(1));
    if k <= 1 {
        rows_a_piece
    } else {
        rows_a_piece.next_multiple_of(grain.max(1)).min(rows)
    }
}

/// Whether a plane stored as `dtype` is packed codes (or their factors'
/// companions ride with them): an expert layer holding one is cut on
/// [`PIECE_ROW_GRAIN`] rows.
#[must_use]
pub fn packed(dtype: Dtype) -> bool {
    !matches!(dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32)
}

/// A slot shape before the chunk size is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Group {
    kind: WeightKind,
    atom_bytes: u64,
    planes: Vec<Plane>,
    /// The rows a piece holds a multiple of (see [`piece_rows`]).
    grain: u64,
    /// Rows: ids a fire may read (`max_tokens x ids a token`), else 0.
    fire_ids: u64,
    /// Experts: the most a token's route picks.
    top_k: u64,
    banks: Vec<WeightBank>,
    /// What a slot's bytes are a whole number of: the lcm of its planes'
    /// boundaries where the profile lays them on their TMA rows, else 1.
    boundary: u64,
}

impl Group {
    /// The bytes of an atom whole, aligned.
    pub(super) fn whole(&self, align: u64) -> u64 {
        align_up(self.atom_bytes, lcm(align.max(1), self.boundary.max(1)))
    }

    /// What a chunk holding this group's slots is a whole number of, so every
    /// slot's planes lie on their boundaries past the pool's base (1: none).
    pub(super) fn boundary(&self) -> u64 {
        self.boundary
    }

    pub(super) fn is_expert(&self) -> bool {
        self.kind == WeightKind::Expert
    }
}

/// A plane's stored bytes: its codes at their packed width, or its elements.
#[must_use]
pub fn plane_bytes(param: &Param) -> u64 {
    let (rows, width) = match param.shape.split_first() {
        Some((rows, rest)) => (*rows, rest.iter().product::<u64>()),
        None => (1, 1),
    };
    let elements = rows.saturating_mul(width);
    match param.dtype {
        // An MXFP4 plane states its packed bytes as its last axis.
        Dtype::Mxfp4 | Dtype::U8g64 | Dtype::E4m3tensor => elements,
        Dtype::U4g64 | Dtype::U4g32 | Dtype::U4g64tiled | Dtype::Nvfp4 => elements.div_ceil(2),
        Dtype::U2g32 | Dtype::U2g64 | Dtype::U2g128 => elements.div_ceil(4),
        Dtype::U2g16k | Dtype::I3g16k | Dtype::U4g32k | Dtype::U5g32k | Dtype::I6g16k => elements,
        other => match elem_bytes(other) {
            Some(element) => elements.saturating_mul(element),
            None => u32::try_from(width)
                .ok()
                .and_then(|k| other.row_bytes(k))
                .map_or_else(
                    || elements.saturating_mul(other.bits()).div_ceil(8),
                    |row| rows.saturating_mul(row),
                ),
        },
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

pub(super) fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        a.max(b)
    } else {
        a / gcd(a, b) * b
    }
}

/// The rows a TMA map of an expert plane, `param` of a bank whose planes
/// are `bank`, indexes it by (an entry's row past its table's window is its
/// offset over the plane's bytes an expert over these): its rows (axis 1),
/// or for the scales of a bank's codes, rows alike, its 128-row blocks,
/// which the tcgen05 kernels read in atoms (a block-FP8 bank's laid scales
/// state their blocks as their rows already). 0: none.
#[must_use]
pub fn tma_rows(trace: &Trace, param: usize, bank: &[usize]) -> u64 {
    let plane = &trace.params[param];
    let rows = plane.shape.get(1).copied().unwrap_or(0);
    let scales = bank.iter().any(|&codes| {
        poem::declare::scales_name(&trace.params[codes].name) == plane.name
            && trace.params[codes].shape.get(1).copied() == Some(rows)
    });
    if scales && rows.is_multiple_of(128) {
        rows / 128
    } else {
        rows
    }
}

/// The boundary an expert plane of `bytes` an expert lies on in its slot
/// where the profile lays planes on their TMA rows: its row (`bytes` over
/// [`tma_rows`]) and 16 bytes (a vector load) both, or 16 where it states no
/// row.
#[must_use]
pub fn plane_boundary(bytes: u64, rows: u64) -> u64 {
    if rows > 0 && bytes.is_multiple_of(rows) && bytes > 0 {
        lcm(bytes / rows, 16)
    } else {
        16
    }
}

/// A MoE layer's banks: the routes its selects share, and its experts.
type LayerKey = (ValueId, u32);

/// Tables gathered alike: rows, ids a token, hashed.
type TableKey = (u64, u32, bool);

/// The weight slot groups of `trace`, a fire reading at most `max_tokens`
/// tokens.
pub(super) fn census(trace: &Trace, max_tokens: u32, tma: bool) -> Vec<Group> {
    let classes = residency(trace);
    let lead = leads(trace);
    // Each streamable weight's planes that stream with it.
    let mut planes: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (q, class) in classes.iter().enumerate() {
        if !class.is_resident() {
            planes.entry(lead[q]).or_default().push(q);
        }
    }

    // Experts: the banks the selects of one route read (a MoE layer) share
    // their experts, an atom each.
    let mut layers: BTreeMap<LayerKey, (u64, Option<u32>, Vec<usize>)> = BTreeMap::new();
    // Rows: tables gathered alike share an atom, a token id's row of each.
    let mut tables: BTreeMap<TableKey, (Option<u32>, Vec<usize>)> = BTreeMap::new();
    for (&p, qs) in &planes {
        match classes[p] {
            Residency::Expert(atom) => {
                let entry = layers.entry((atom.routes, atom.experts)).or_insert((
                    0,
                    atom.layer,
                    Vec::new(),
                ));
                entry.0 = entry.0.max(u64::from(atom.top_k));
                entry.2.extend(qs);
            }
            Residency::Rows(atom) => {
                let entry = tables
                    .entry((atom.rows, atom.ids_per_token, atom.hashed))
                    .or_insert((atom.layer, Vec::new()));
                entry.1.extend(qs);
            }
            Residency::Resident => {}
        }
    }

    let mut groups: Vec<Group> = Vec::new();
    let atoms_of = |q: &usize| {
        let param = &trace.params[*q];
        param.experts.map_or_else(
            || param.shape.first().copied().unwrap_or(1),
            |s| u64::from(s.experts),
        )
    };
    for ((_, experts), (top_k, layer, qs)) in layers {
        let experts = u64::from(experts).max(1);
        let mut shape: Vec<Plane> = qs
            .iter()
            .map(|q| {
                let param = &trace.params[*q];
                // Per-rank shapes hold `slots` experts: bytes an expert
                // follow from axis 0 as stored.
                let held = param.shape.first().copied().unwrap_or(1).max(1);
                Plane {
                    rows: param.shape.get(1).copied().unwrap_or(0),
                    bytes: plane_bytes(param) / held,
                }
            })
            .collect();
        shape.sort_unstable();
        // In the bank's order, as an engine lays a slot: back to back, or
        // each plane on its boundary where the profile maps windows.
        let (atom_bytes, boundary) = if tma {
            let params: Vec<usize> = qs.iter().copied().collect();
            qs.iter().fold((0u64, 1u64), |(at, every), &q| {
                let param = &trace.params[q];
                let held = param.shape.first().copied().unwrap_or(1).max(1);
                let bytes = plane_bytes(param) / held;
                let edge = plane_boundary(bytes, tma_rows(trace, q, &params));
                (align_up(at, edge) + bytes, lcm(every, edge))
            })
        } else {
            (shape.iter().map(|p| p.bytes).sum(), 1)
        };
        let grain = if qs.iter().any(|q| packed(trace.params[*q].dtype)) {
            PIECE_ROW_GRAIN
        } else {
            1
        };
        let bank = WeightBank {
            layer,
            atoms: experts,
            params: qs.iter().map(|q| *q as u32).collect(),
        };
        // Layers whose experts are alike share one shape.
        if let Some(group) = groups
            .iter_mut()
            .find(|g| g.kind == WeightKind::Expert && g.planes == shape && g.grain == grain)
        {
            group.top_k = group.top_k.max(top_k);
            group.banks.push(bank);
        } else {
            groups.push(Group {
                kind: WeightKind::Expert,
                atom_bytes,
                planes: shape,
                grain,
                fire_ids: 0,
                top_k,
                banks: vec![bank],
                boundary,
            });
        }
    }
    for ((rows, ids_per_token, _), (layer, qs)) in tables {
        let atom_bytes = qs
            .iter()
            .map(|q| plane_bytes(&trace.params[*q]) / atoms_of(q).max(1))
            .sum();
        groups.push(Group {
            kind: WeightKind::Rows,
            atom_bytes,
            planes: Vec::new(),
            grain: 1,
            fire_ids: u64::from(max_tokens.max(1)) * u64::from(ids_per_token.max(1)),
            top_k: 0,
            boundary: 1,
            banks: vec![WeightBank {
                layer,
                atoms: rows,
                params: qs.iter().map(|q| *q as u32).collect(),
            }],
        });
    }
    groups
}

/// The weight slots of `groups` in chunks of `chunk` bytes, experts cut on
/// their output axis where `cut` lets them: `None` where an expert does
/// not fit whole and may not be cut, or a piece would fall under
/// [`MIN_PIECE`].
pub(super) fn layout(
    groups: &[Group],
    chunk: u64,
    align: u64,
    cut: bool,
) -> Option<Vec<WeightSlot>> {
    groups
        .iter()
        .map(|group| {
            let whole = group.whole(align);
            let atoms: u64 = group.banks.iter().map(|b| b.atoms).sum();
            match group.kind {
                WeightKind::Expert => {
                    let (pieces, bytes) = if whole <= chunk {
                        (1, whole)
                    } else if cut {
                        let k = whole.div_ceil(chunk.max(1));
                        let bytes = (k..=k.saturating_mul(2)).find_map(|k| {
                            let piece = piece_bytes(&group.planes, k, align, group.grain);
                            (piece <= chunk).then_some((k, piece))
                        })?;
                        if bytes.1 < MIN_PIECE {
                            return None;
                        }
                        (u32::try_from(bytes.0).ok()?, bytes.1)
                    } else {
                        return None;
                    };
                    let staged = u64::try_from(group.banks.len().min(2)).unwrap_or(2);
                    Some(WeightSlot {
                        kind: WeightKind::Expert,
                        atom_bytes: group.atom_bytes,
                        atoms_per_slot: 1,
                        pieces,
                        bytes,
                        slots: atoms * u64::from(pieces),
                        floor: staged * group.top_k.max(1) * u64::from(pieces),
                        banks: group.banks.clone(),
                    })
                }
                WeightKind::Rows => {
                    let per = (align / group.atom_bytes.max(1)).max(1);
                    let bytes = align_up(per * group.atom_bytes, align);
                    if bytes > chunk {
                        return None;
                    }
                    let slots = atoms.div_ceil(per);
                    Some(WeightSlot {
                        kind: WeightKind::Rows,
                        atom_bytes: group.atom_bytes,
                        atoms_per_slot: per,
                        pieces: 1,
                        bytes,
                        slots,
                        floor: slots.min(group.fire_ids),
                        banks: group.banks.clone(),
                    })
                }
            }
        })
        .collect()
}

/// The largest of `k` pieces of an expert cut on its planes' output axis:
/// [`piece_rows`] rows of each plane, the planes of no output axis in the
/// first piece.
fn piece_bytes(planes: &[Plane], k: u64, align: u64, grain: u64) -> u64 {
    let mut piece = 0;
    for plane in planes {
        piece = align_up(piece, align);
        piece += if plane.rows == 0 {
            plane.bytes
        } else {
            piece_rows(plane.rows, k, grain) * (plane.bytes / plane.rows)
        };
    }
    align_up(piece, align)
}

/// The share of the chunks `slots` fill that holds nothing: each chunk's
/// room past its last whole slot, over every slot of every shape.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn tail(slots: &[WeightSlot], chunk: u64) -> f64 {
    let (mut held, mut taken) = (0u64, 0u64);
    for slot in slots {
        held += slot.total_bytes();
        taken += slot.chunks(chunk) * chunk;
    }
    if taken == 0 {
        0.0
    } else {
        1.0 - held as f64 / taken as f64
    }
}
