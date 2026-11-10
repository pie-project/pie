//! The weight cache (CONTRACT: `ChunkFacts::weights`, `Engine::weight_residency`,
//! `Engine::weight_uses`) on unified memory: token-id-gathered table rows and
//! routed experts in read-only pool slots, read through tables of GPU
//! addresses.
//!
//! **Rows.** A table read only by the plain embedding gather, bf16, out of the
//! checkpoint, is stated as a [`WeightKind::Rows`] shape when the compiler's
//! on-device Metal recipe gives it (alone, or with the tables gathered alike:
//! gemma's per-layer tables) a rows slot. Its atoms are blocks of
//! `block_rows` token ids; a slot holds one block of every table of the
//! group, table after table (each table's `block_rows` rows laid as in the
//! table, so a row is read the same way wherever its block lies). The gather
//! (`gather/embed.metal`) reads a row through its block's address.
//!
//! **Experts.** A dense bf16 bank read only at the `bank` port of a plain
//! routed select (no bias, not quantized) is streamed an expert at a time
//! when the recipe gives its layer an expert slot of one piece and every bank
//! of every layer of the slot qualifies. The on-device recipe cuts an expert
//! larger than the chunk into `pieces` slots on its banks' output axis
//! (piece `j` holds rows `j * ceil(N / pieces) ..` of each bank's `N`): atom
//! `(layer * experts + expert) * pieces + piece`, a slot holding that piece
//! of each of the layer's banks, back to back in the recipe's order. The
//! select (`linear/moe/select.metal`) reads an output row of an expert
//! through its piece's address; a layer's reads are counted at its first
//! bank, an expert at a time, and reported for every piece of it.
//!
//! **The host copy is the weight itself, read in place.** Metal memory is
//! unified, so there is no pinned staging copy as on CUDA: an atom no slot
//! holds is read from the weight's own place (the address table names it).
//! A cold load lands each streamed weight in an allocation of its own
//! (`Weights::hosted`), so it can be given back; a warm load reads it out of
//! the checkpoint's mapped window, which the load keeps for its other
//! planes. The kernels bind only the address table under a table (never the
//! weight), so no recorded program holds the weight, and dropping it needs
//! no re-arm.
//!
//! **Residency** (`Engine::weight_residency`): every frame drains, the atoms
//! newly named a slot are blitted from the weight into their slots, and every
//! address table is rewritten on the host before the next fire. Evacuation
//! rewrites them again. A use of an atom that is not named reads the weight
//! in place (no staging slot: on unified memory the in-place read is the
//! stream). **Held** (`WeightResidency::held`, a deployment that does not
//! stream) with every atom of every shape named is the fast path: every
//! read lands in the pool, and each weight's own allocation (the cold road's)
//! is given back; from then on a set naming a new slot for an atom, or a set
//! that is not held, is refused. The kernels do not change between the cases:
//! one kernel reads through the table whatever the entries say, and reads
//! the contiguous weight where nothing is stated, so a load that states no
//! shape runs the kernels it ran before the cache.
//!
//! The address tables and hit counters are buffers the kernels bind; a
//! weight's own allocation is reached only through addresses, so it is kept
//! in the queue's residency set while it is read (the pool's sparse buffer
//! is in that set already). A device without that set (not Metal 4) states
//! no weight shape.

use engine::chunk::{ShapeId, WeightAtom, WeightHeld, WeightKind, WeightShape};
use kernels_metal::Tensor;
use kernels_metal::gather::RowTable;
use kernels_metal::linear::moe::ExpertTable;

/// The address table a streamed weight is read through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Streamed {
    Rows(RowTable),
    Experts(ExpertTable),
    /// A quantized routed bank's planes (codes, scales, zero points, its
    /// select's bias), at its codes param.
    Bank(kernels_metal::linear::moe::BankTable),
    /// A quantized table's planes (codes, scales, zero points), at its
    /// codes param.
    BankRows(kernels_metal::gather::BankRows),
}
use poem_compiler::chunk::ChunkRecipe;
use poem_ir::{Def, Operands, Operation, Trace};

use crate::device::alloc::slab_address;
use crate::device::ctx::Frame;
use crate::device::handles::Binding;
use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};

/// The first weight shape's id: past the state shapes (the kv list, the
/// fixed record, up to four window lists, the rs list).
pub const WEIGHTS: u32 = 8;

/// Whether every node of `trace` that reads param `at` is one `reads`
/// accepts, and one does.
fn read_only_by(trace: &Trace, at: usize, reads: impl Fn(&Operation) -> bool) -> bool {
    let mut inputs = Vec::new();
    let mut read = false;
    for node in &trace.nodes {
        inputs.clear();
        node.op.inputs(&mut inputs);
        let touches = inputs.iter().any(|v| {
            matches!(trace.values.get(v.0 as usize).map(|d| &d.def),
                Some(Def::Weight(w)) if *w as usize == at)
        });
        if !touches {
            continue;
        }
        if !reads(&node.op) {
            return false;
        }
        read = true;
    }
    read
}

/// The tables of `trace` this engine streams by their rows: read only by the
/// plain embedding gather, out of the checkpoint (as `engine-cuda`'s
/// `layout::streamed_rows`): a bf16 plane of its own with no companion, or
/// affine 4/8-bit codes (the gather `embed_gather.metal` reads through a
/// table) with their scales and zero points.
#[must_use]
pub fn streamed_rows(trace: &Trace) -> Vec<usize> {
    use poem_ir::Gather;
    use poem_ir::residency::{Residency, leads, residency};
    use poem_ir::Dtype;
    let classes = residency(trace);
    let mut tables: Vec<usize> = (0..trace.params.len())
        .filter(|&at| {
            let param = &trace.params[at];
            matches!(classes[at], Residency::Rows(_))
                && matches!(param.dtype, Dtype::Bf16 | Dtype::U4g64 | Dtype::U4g32 | Dtype::U8g64)
                && param.shape.len() == 2
                && param.source == poem_ir::ParamSource::Checkpoint
                && read_only_by(trace, at, |op| {
                    matches!(op, Operation::Gather(Gather::Embed {
                        weights: None,
                        scales: None,
                        concat: false,
                        vocab,
                        ..
                    }) if u64::from(*vocab) == param.shape[0])
                })
        })
        .collect();
    let lead = leads(trace);
    let companions: Vec<usize> = (0..trace.params.len())
        .filter(|&at| {
            lead[at] != at
                && tables.contains(&lead[at])
                && trace.params[lead[at]].dtype != Dtype::Bf16
                && matches!(classes[at], Residency::Rows(_))
                && trace.params[at].source == poem_ir::ParamSource::Checkpoint
        })
        .collect();
    tables.extend(companions);
    tables.sort_unstable();
    tables
}

/// Whether a routed bank stored as `dtype` is one every routed arm reads
/// through an expert table: affine codes at 64/{2,4,8}, 32/2, 128/2 and
/// MXFP4 (`kernels_metal::linear::moe::BankTable`).
fn tabled_codes(dtype: poem_ir::Dtype) -> bool {
    use poem_ir::Dtype;
    matches!(
        dtype,
        Dtype::U4g64 | Dtype::U8g64 | Dtype::U2g64 | Dtype::U2g32 | Dtype::U2g128 | Dtype::Mxfp4
    )
}

/// The routed banks of `trace` this engine streams an expert at a time:
/// planes out of the checkpoint read only at the `bank` port of a routed
/// select whose kernels read an expert table (as `engine-cuda`'s
/// `layout::streamed_experts`, before its recipe grouping, which
/// [`Shape::experts`] applies): a dense bf16 bank of a plain select (no
/// bias, not quantized), or a quantized bank ([`tabled_codes`]) of any
/// select of no `quant` scheme; with each quantized bank its companions
/// (scales, zero points) and its select's expert bias.
#[must_use]
pub fn streamed_experts(trace: &Trace) -> Vec<usize> {
    use poem_ir::Linear;
    use poem_ir::residency::{Residency, leads, residency};
    let classes = residency(trace);
    let weight = |id: &poem_ir::ValueId, at: usize| {
        matches!(trace.values.get(id.0 as usize).map(|d| &d.def),
            Some(Def::Weight(w)) if *w as usize == at)
    };
    let expert = |at: usize| {
        let param = &trace.params[at];
        matches!(classes[at], Residency::Expert(_))
            && param.shape.len() >= 2
            && param.source == poem_ir::ParamSource::Checkpoint
    };
    let mut banks: Vec<usize> = (0..trace.params.len())
        .filter(|&at| {
            let dtype = trace.params[at].dtype;
            expert(at)
                && (dtype == poem_ir::Dtype::Bf16 || tabled_codes(dtype))
                && read_only_by(trace, at, |op| match op {
                    Operation::Linear(Linear::MoeSelect {
                        bank,
                        bias,
                        quantized,
                        quant: None,
                        ..
                    }) if weight(bank, at) => {
                        tabled_codes(dtype) || (bias.is_none() && !*quantized)
                    }
                    _ => false,
                })
        })
        .collect();
    // A quantized bank's companions, and its selects' expert biases.
    let quantized: Vec<usize> = banks
        .iter()
        .copied()
        .filter(|&at| tabled_codes(trace.params[at].dtype))
        .collect();
    let lead = leads(trace);
    for at in 0..trace.params.len() {
        if !expert(at) || banks.contains(&at) {
            continue;
        }
        let companion = lead[at] != at && quantized.contains(&lead[at]);
        let bias = trace.params[at].dtype == poem_ir::Dtype::Bf16
            && read_only_by(trace, at, |op| {
                matches!(op, Operation::Linear(Linear::MoeSelect {
                    bank,
                    bias: Some(bias),
                    quant: None,
                    ..
                }) if weight(bias, at) && quantized.iter().any(|&b| weight(bank, b)))
            });
        if companion || bias {
            banks.push(at);
        }
    }
    banks.sort_unstable();
    banks
}

/// Every param [`streamed_rows`] or [`streamed_experts`] names: the ones a
/// cold load lands in allocations of their own.
#[must_use]
pub fn streamed(trace: &Trace) -> Vec<usize> {
    let mut all = streamed_rows(trace);
    all.extend(streamed_experts(trace));
    all.sort_unstable();
    all.dedup();
    all
}

/// What `trace`'s weights demand of the device however the runtime runs the
/// weight cache, its params' stored bytes `plane_bytes` (aligned): every
/// weight no shape of `recipe` streams, whole, and of the streamable ones
/// only the floor each shape's slots keep; and the streamable weights'
/// whole bytes, which a held set seats in pool slots and a streaming one
/// reads a few atoms of at a time.
#[must_use]
pub fn resident_demand(trace: &Trace, recipe: &ChunkRecipe, plane_bytes: &[u64]) -> (u64, u64) {
    let shapes = Shape::of(trace, recipe);
    let streamed: std::collections::BTreeSet<usize> = shapes
        .iter()
        .flat_map(|shape| shape.planes.iter().map(|plane| plane.param))
        .collect();
    let aligned = |at: usize| plane_bytes[at].next_multiple_of(crate::weights::ALIGN);
    let kept: u64 = (0..plane_bytes.len())
        .filter(|at| !streamed.contains(at))
        .map(aligned)
        .sum();
    let floor: u64 = shapes
        .iter()
        .map(|shape| u64::from(shape.floor) * shape.slot_bytes)
        .sum();
    let streamable: u64 = streamed.iter().map(|&at| aligned(at)).sum();
    (kept + floor, streamable)
}

/// One weight's part of a shape. Its groups (row blocks, or a layer's
/// experts) are `stride` bytes apart in the weight's own copy, each cut into
/// `pieces` runs of `piece` bytes (an expert's rows on the output axis; a
/// row block is one piece); entry `i` (group `i / pieces`, piece
/// `i % pieces`) is atom `atom + i` and lies `at` bytes into that atom's
/// slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    pub param: usize,
    pub atom: u32,
    pub entries: u32,
    pub stride: u64,
    pub pieces: u32,
    pub piece: u64,
    /// Rows a piece (experts: on the output axis); rows a block (rows).
    pub piece_rows: u32,
    pub at: u64,
    /// The weight's bytes: the last group copies what is left of them.
    pub bytes: u64,
    /// Whether its reads are the atoms' use record.
    pub counts: bool,
}

impl Plane {
    /// Where entry `entry` starts in the weight's own copy.
    fn entry_at(&self, entry: u32) -> u64 {
        let pieces = self.pieces.max(1);
        u64::from(entry / pieces) * self.stride + u64::from(entry % pieces) * self.piece
    }

    /// The bytes entry `entry` copies: its piece of what its group holds.
    fn entry_bytes(&self, entry: u32) -> u64 {
        let pieces = self.pieces.max(1);
        let group = u64::from(entry / pieces) * self.stride;
        let held = self.stride.min(self.bytes.saturating_sub(group));
        let start = u64::from(entry % pieces) * self.piece;
        self.piece.min(held.saturating_sub(start))
    }
}

/// One weight shape: the planes of a recipe weight slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub id: ShapeId,
    pub kind: WeightKind,
    pub slot_bytes: u64,
    pub per_chunk: u32,
    /// The atoms: row blocks, or every layer's experts.
    pub count: u32,
    pub floor: u32,
    /// Slots an expert is cut into (1 for rows): atom `group * pieces +
    /// piece`.
    pub pieces: u32,
    pub planes: Vec<Plane>,
    /// The weights' names, for logs.
    pub name: String,
}

impl Shape {
    /// The shapes of `recipe` this engine streams: rows shapes whose every
    /// table is in [`streamed_rows`], then expert shapes (one piece an
    /// expert) whose every bank of every layer is in [`streamed_experts`].
    /// A slot that holds a weight this engine does not stream is not
    /// stated, its weights read whole.
    #[must_use]
    pub fn of(trace: &Trace, recipe: &ChunkRecipe) -> Vec<Shape> {
        let mut out = Vec::new();
        let rows = streamed_rows(trace);
        for slot in &recipe.weights {
            if slot.kind != poem_compiler::chunk::WeightKind::Rows || slot.pieces != 1 {
                continue;
            }
            if let Some(shape) = Shape::rows(trace, recipe, slot, &rows, out.len()) {
                out.push(shape);
            }
        }
        let banks = streamed_experts(trace);
        for slot in &recipe.weights {
            if slot.kind != poem_compiler::chunk::WeightKind::Expert {
                continue;
            }
            if let Some(shape) = Shape::experts(trace, recipe, slot, &banks, out.len()) {
                out.push(shape);
            }
        }
        out
    }

    fn named(trace: &Trace, params: impl Iterator<Item = usize>) -> String {
        params
            .map(|p| trace.params[p].name.as_str())
            .collect::<Vec<_>>()
            .join(",")
    }

    fn rows(
        trace: &Trace,
        recipe: &ChunkRecipe,
        slot: &poem_compiler::chunk::WeightSlot,
        streamed: &[usize],
        at: usize,
    ) -> Option<Shape> {
        let [bank] = slot.banks.as_slice() else {
            return None;
        };
        let params: Vec<usize> = bank.params.iter().map(|&p| p as usize).collect();
        let rows = bank.atoms;
        let block_rows = u32::try_from(slot.atoms_per_slot).ok()?;
        if params.is_empty()
            || block_rows == 0
            || rows == 0
            || !params.iter().all(|p| {
                streamed.contains(p) && trace.params[*p].shape.first().copied() == Some(rows)
            })
        {
            return None;
        }
        let blocks = u32::try_from(rows.div_ceil(u64::from(block_rows))).ok()?;
        let mut planes = Vec::with_capacity(params.len());
        let mut used = 0u64;
        for (k, &param) in params.iter().enumerate() {
            // A row's stored bytes (codes at their packed width).
            let full = poem_compiler::chunk::weights::plane_bytes(&trace.params[param]);
            let row = full.checked_div(rows)?;
            if row * rows != full {
                return None;
            }
            let stride = u64::from(block_rows) * row;
            planes.push(Plane {
                param,
                atom: 0,
                entries: blocks,
                stride,
                pieces: 1,
                piece: stride,
                piece_rows: block_rows,
                at: used,
                bytes: rows * row,
                counts: k == 0,
            });
            used += stride;
        }
        if used > slot.bytes || slot.bytes > recipe.chunk_bytes {
            return None;
        }
        Some(Shape {
            id: ShapeId(WEIGHTS + u32::try_from(at).ok()?),
            kind: WeightKind::Rows,
            slot_bytes: slot.bytes,
            per_chunk: u32::try_from(recipe.chunk_bytes / slot.bytes).ok()?.max(1),
            count: blocks,
            floor: u32::try_from(slot.floor).unwrap_or(u32::MAX).min(blocks),
            pieces: 1,
            name: Shape::named(trace, params.iter().copied()),
            planes,
        })
    }

    /// An expert slot: each layer's banks, an expert of each a slot (or
    /// `pieces` slots, piece `j` holding rows `j * ceil(N / pieces) ..` of
    /// every bank's `N` output rows), the banks back to back in the
    /// recipe's order inside a piece.
    fn experts(
        trace: &Trace,
        recipe: &ChunkRecipe,
        slot: &poem_compiler::chunk::WeightSlot,
        streamed: &[usize],
        at: usize,
    ) -> Option<Shape> {
        if slot.atoms_per_slot != 1 || slot.pieces == 0 || slot.bytes > recipe.chunk_bytes {
            return None;
        }
        let pieces = slot.pieces;
        let experts = slot.banks.first()?.atoms;
        let experts32 = u32::try_from(experts).ok()?;
        if experts == 0 {
            return None;
        }
        let entries = experts32.checked_mul(pieces)?;
        let mut planes = Vec::new();
        for (layer, bank) in slot.banks.iter().enumerate() {
            if bank.atoms != experts || bank.params.is_empty() {
                return None;
            }
            let mut used = 0u64;
            let layer_quantized = bank
                .params
                .iter()
                .any(|&p| poem_compiler::chunk::weights::packed(trace.params[p as usize].dtype));
            for (k, &param) in bank.params.iter().enumerate() {
                let param = param as usize;
                let held = &trace.params[param];
                if !streamed.contains(&param) || held.shape.first().copied() != Some(experts) {
                    return None;
                }
                let rows = *held.shape.get(1)?;
                // A row's stored bytes: the plane's over its experts' rows
                // (codes at their packed width).
                let full = poem_compiler::chunk::weights::plane_bytes(held);
                let row = full.checked_div(experts * rows)?;
                if row * experts * rows != full {
                    return None;
                }
                // As the recipe cuts it: a layer holding a packed plane on
                // whole tiles of every routed arm.
                let grain = if layer_quantized {
                    poem_compiler::chunk::weights::PIECE_ROW_GRAIN
                } else {
                    1
                };
                let piece_rows =
                    poem_compiler::chunk::weights::piece_rows(rows, u64::from(pieces), grain);
                let piece = piece_rows * row;
                // A quantized plane lies 16-byte aligned, and its pieces hold
                // whole tiles (or every row).
                if layer_quantized
                    && (!used.is_multiple_of(16)
                        || (piece_rows < rows
                            && !piece_rows.is_multiple_of(u64::from(
                                kernels_metal::linear::moe::BankTable::ROWS,
                            ))))
                {
                    return None;
                }
                planes.push(Plane {
                    param,
                    atom: u32::try_from(layer).ok()?.checked_mul(entries)?,
                    entries,
                    stride: rows * row,
                    pieces,
                    piece,
                    piece_rows: u32::try_from(piece_rows).ok()?,
                    at: used,
                    bytes: experts * rows * row,
                    counts: k == 0,
                });
                used += piece;
            }
            if used > slot.bytes {
                return None;
            }
        }
        let count = entries.checked_mul(u32::try_from(slot.banks.len()).ok()?)?;
        Some(Shape {
            id: ShapeId(WEIGHTS + u32::try_from(at).ok()?),
            kind: WeightKind::Expert,
            slot_bytes: slot.bytes,
            per_chunk: u32::try_from(recipe.chunk_bytes / slot.bytes).ok()?.max(1),
            count,
            floor: u32::try_from(slot.floor).unwrap_or(u32::MAX).min(count),
            pieces,
            name: Shape::named(trace, planes.iter().map(|plane| plane.param)),
            planes,
        })
    }

    #[must_use]
    pub fn facts(&self) -> WeightShape {
        WeightShape {
            id: self.id,
            kind: self.kind,
            slot_bytes: self.slot_bytes,
            slots_per_chunk: self.per_chunk,
            count: self.count,
            floor_slots: self.floor,
            name: self.name.clone(),
            seated: None,
        }
    }
}

/// The byte offset in the pool of a slot ref.
type SlotAt<'a> = &'a dyn Fn(u32) -> Result<u64>;

/// A weight's own copy: where a kernel reads an atom no slot holds.
#[derive(Debug, Clone)]
struct Host {
    binding: Binding,
    /// The handle the load bound the weight at (released with the copy).
    handle: u32,
    /// Where the weight lies in the serving file, where the load reads it
    /// through a window of its own: a slot is then filled from the file on
    /// the host, which leaves the mapping's pages untouched.
    file: Option<(std::sync::Arc<crate::mapping::Mapping>, u64)>,
    /// Whether `binding` is the weight's own memory (a file weight's is a
    /// placeholder until [`Cache::open_files`]).
    opened: bool,
}

/// Slots filled on the host from the serving file: `(from, to, bytes)`
/// each, `bytes` at `from` in the file to `to` bytes into the pool, in the
/// file's order.
#[derive(Debug, Clone)]
pub struct FileFills {
    pub file: std::sync::Arc<crate::mapping::Mapping>,
    pub fills: Vec<(u64, u64, u64)>,
}

impl FileFills {
    /// `bytes` bytes at `from`, read from the file (not through its
    /// mapping).
    pub fn read(&self, from: u64, bytes: u64, into: &mut Vec<u8>) -> Result<()> {
        use std::os::unix::fs::FileExt;
        into.resize(usize::try_from(bytes).unwrap_or(usize::MAX), 0);
        self.file
            .file()
            .read_exact_at(into, from)
            .map_err(|why| Fault::Mapped {
                step: "read",
                what: self.file.path().display().to_string(),
                why: why.to_string(),
            })
    }
}

/// One weight shape's cache: its address tables, its hit counters and
/// which atom each slot holds.
#[derive(Debug)]
pub struct Cache {
    pub shape: Shape,
    /// Each plane's own copy; `None` once given back.
    hosts: Vec<Option<Host>>,
    /// Each plane's address table, a u64 an entry, and its handle.
    tables: Vec<(Buffer, u32)>,
    /// A u32 a group (a row block, or an expert: every piece of it), and
    /// each counting plane's handle on its groups.
    hits: Buffer,
    hit_handles: Vec<Option<u32>>,
    /// The slot ref each atom is named, if any.
    slots: Vec<Option<u32>>,
    /// The load seated every atom itself (`WeightShape::seated`), until the
    /// runtime's first residency.
    preseated: bool,
    /// Whether the weights' own copies are in the queue's residency set
    /// (they are read in place by a miss).
    hosts_resident: bool,
}

impl Cache {
    /// The cache of `shape` over the weights the load bound (`handle_of` a
    /// param), every atom read from its weight in place; its buffers bound
    /// in `handles` (before the load seals them).
    pub fn open(
        device: &Context,
        handles: &Handles,
        shape: Shape,
        handle_of: impl Fn(usize) -> Option<u32>,
        file_of: impl Fn(usize) -> Option<(std::sync::Arc<crate::mapping::Mapping>, u64)>,
    ) -> Result<Cache> {
        let mut hosts = Vec::with_capacity(shape.planes.len());
        let mut tables = Vec::with_capacity(shape.planes.len());
        let pieces = shape.pieces.max(1);
        let hits = Buffer::zeroed(device, u64::from(shape.count / pieces) * 4)?;
        let mut hit_handles = Vec::with_capacity(shape.planes.len());
        for plane in &shape.planes {
            let param = plane.param;
            let handle = handle_of(param).ok_or_else(|| Fault::Unbound {
                what: format!("param {param}, a streamed weight this load binds no plane for"),
            })?;
            let binding = handles
                .get(handle)
                .map(|b| b.clone())
                .ok_or_else(|| Fault::Unbound {
                    what: format!("handle {handle} of streamed param {param}"),
                })?;
            hosts.push(Some(Host {
                binding,
                handle,
                opened: false,
                file: file_of(param),
            }));
            let entries = u64::from(plane.entries);
            let table = Buffer::zeroed(device, entries * 8)?;
            let bound = handles.bind(&table, 0, entries * 8)?;
            tables.push((table, bound));
            let groups = u64::from(plane.entries / pieces);
            hit_handles.push(if plane.counts {
                Some(handles.bind(&hits, u64::from(plane.atom / pieces) * 4, groups * 4)?)
            } else {
                None
            });
        }
        let mut cache = Cache {
            slots: vec![None; shape.count as usize],
            shape,
            hosts,
            tables,
            hits,
            hit_handles,
            preseated: false,
            hosts_resident: false,
        };
        cache.stage(None)?;
        Ok(cache)
    }

    /// The address table each of the shape's weights is read through, by
    /// param.
    pub fn tables(&self) -> impl Iterator<Item = (usize, Streamed)> + '_ {
        self.shape.planes.iter().enumerate().map(|(k, plane)| {
            let (_, handle) = self.tables[k];
            let groups = plane.entries / plane.pieces.max(1);
            let at = Tensor::new(handle, plane.entries, 1, poem_ir::Dtype::I64);
            let hits =
                self.hit_handles[k].map(|hits| Tensor::new(hits, groups, 1, poem_ir::Dtype::U32));
            (
                plane.param,
                match self.shape.kind {
                    WeightKind::Rows => Streamed::Rows(RowTable {
                        at,
                        block_rows: plane.piece_rows,
                        hits,
                    }),
                    WeightKind::Expert => Streamed::Experts(ExpertTable {
                        at,
                        pieces: plane.pieces,
                        piece_rows: plane.piece_rows,
                        hits,
                    }),
                },
            )
        })
    }

    /// The weights' own copies the queue must keep resident: the slabs the
    /// kernels reach only through addresses.
    pub fn host_slabs(&self) -> impl Iterator<Item = &crate::device::alloc::Slab> {
        self.hosts.iter().flatten().map(|host| host.binding.slab())
    }

    /// Rewrites every address table: an atom named a slot reads it through
    /// `pool` (base address and the byte offset of a slot ref), the others
    /// read their weight in place. The caller has drained every frame.
    fn stage(&mut self, pool: Option<(u64, SlotAt<'_>)>) -> Result<()> {
        for (k, (table, _)) in self.tables.iter_mut().enumerate() {
            let plane = &self.shape.planes[k];
            let host = self.hosts[k].as_ref();
            let mut bytes = Vec::with_capacity(plane.entries as usize * 8);
            for entry in 0..plane.entries {
                let slot = self.slots[(plane.atom + entry) as usize];
                let address = match (slot, pool) {
                    (Some(slot), Some((base, offset))) => base + offset(slot)? + plane.at,
                    _ => {
                        let Some(host) = host else {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "atom {} of param {}, named no slot after its weight's own \
                                     copy was given back",
                                    plane.atom + entry,
                                    plane.param
                                ),
                            });
                        };
                        slab_address(host.binding.slab())
                            + host.binding.offset()
                            + plane.entry_at(entry)
                    }
                };
                bytes.extend_from_slice(&address.to_le_bytes());
            }
            table.write(0, &bytes)?;
        }
        Ok(())
    }

    /// Takes the shape's part of a residency: atoms newly named a slot are
    /// blitted from their weights into `frame` (the pool `into`, a slot
    /// ref's byte offset by `offset`). The tables are rewritten by
    /// [`Cache::restage`] once `frame` is committed.
    ///
    /// A weight read from the serving file is not blitted: its fills are
    /// answered, for the caller to write on the host.
    pub fn take(
        &mut self,
        frame: &mut Frame,
        named: Vec<Option<u32>>,
        into: &Buffer,
        offset: SlotAt<'_>,
    ) -> Result<Vec<FileFills>> {
        // Each atom newly named a slot, and the slot's byte offset.
        let mut to: Vec<Option<u64>> = vec![None; named.len()];
        for (atom, slot) in named.iter().enumerate() {
            let Some(slot) = *slot else { continue };
            if self.slots[atom] != Some(slot) {
                to[atom] = Some(offset(slot)?);
            }
        }
        // A plane at a time, its entries in order: a file weight's fills
        // follow one another in the file.
        let mut fills = Vec::new();
        for (plane, host) in self.shape.planes.iter().zip(&self.hosts) {
            let mut file_fills = Vec::new();
            for entry in 0..plane.entries {
                let atom = plane.atom + entry;
                let Some(to) = to.get(atom as usize).copied().flatten() else {
                    continue;
                };
                let Some(host) = host else {
                    return Err(Fault::Unbound {
                        what: format!(
                            "atom {atom} of param {} named a new slot after the held set's \
                             weights were given back: a held atom stays where it was named",
                            plane.param
                        ),
                    });
                };
                let bytes = plane.entry_bytes(entry);
                if bytes == 0 {
                    continue;
                }
                if let Some((_, at)) = &host.file {
                    file_fills.push((at + plane.entry_at(entry), to + plane.at, bytes));
                    continue;
                }
                frame.copy(
                    host.binding.slab(),
                    host.binding.offset() + plane.entry_at(entry),
                    into.slab(),
                    to + plane.at,
                    bytes,
                )?;
            }
            if let Some(Host {
                file: Some((file, _)),
                ..
            }) = host
                && !file_fills.is_empty()
            {
                fills.push(FileFills {
                    file: std::sync::Arc::clone(file),
                    fills: file_fills,
                });
            }
        }
        self.slots = named;
        Ok(fills)
    }

    /// Rewrites the address tables for the slots the atoms are named now,
    /// through the pool at `base` (see [`Cache::stage`]).
    pub fn restage(&mut self, base: u64, offset: SlotAt<'_>) -> Result<()> {
        self.stage(Some((base, offset)))
    }

    /// The shape's facts, stating where the load seated it while it holds
    /// its own seating.
    #[must_use]
    pub fn facts(&self) -> WeightShape {
        WeightShape {
            seated: if self.preseated {
                self.slots.first().copied().flatten()
            } else {
                None
            },
            ..self.shape.facts()
        }
    }

    /// Marks the slots named as the load's own seating (or not).
    pub fn set_preseated(&mut self, preseated: bool) {
        self.preseated = preseated;
    }

    #[must_use]
    pub fn preseated(&self) -> bool {
        self.preseated
    }

    /// Whether the weights' own copies are held in the queue's residency
    /// set; marked so by the caller who put them there.
    #[must_use]
    pub fn hosts_resident(&self) -> bool {
        self.hosts_resident
    }

    pub fn set_hosts_resident(&mut self, resident: bool) {
        self.hosts_resident = resident;
    }

    /// Gives each weight read from the serving file, and bound to no
    /// memory yet, the binding `open` answers for it (a window over the
    /// file a miss reads in place), by param and handle. The caller
    /// restages the tables after.
    pub fn open_files(
        &mut self,
        mut open: impl FnMut(usize, u32) -> Result<Option<Binding>>,
    ) -> Result<()> {
        for (host, plane) in self.hosts.iter_mut().zip(&self.shape.planes) {
            let Some(host) = host else { continue };
            if host.file.is_none() || host.opened {
                continue;
            }
            if let Some(binding) = open(plane.param, host.handle)? {
                host.binding = binding;
            }
            host.opened = true;
        }
        Ok(())
    }

    /// Whether every atom is named a slot.
    #[must_use]
    pub fn whole(&self) -> bool {
        self.slots.iter().all(Option::is_some)
    }

    /// Gives back the weights' own copies: answers each param's handle and
    /// slab, for the caller to unbind and let go. Every atom is named.
    pub fn release(&mut self) -> Vec<(usize, u32, crate::device::alloc::Slab)> {
        if !self.whole() {
            return Vec::new();
        }
        self.hosts
            .iter_mut()
            .zip(&self.shape.planes)
            .filter_map(|(host, plane)| {
                host.take()
                    .map(|host| (plane.param, host.handle, host.binding.slab().clone()))
            })
            .collect()
    }

    /// Whether the weights' own copies were given back.
    #[must_use]
    pub fn released(&self) -> bool {
        self.hosts.iter().all(Option::is_none)
    }

    /// The atoms named a slot, and all of them.
    #[must_use]
    pub fn atoms(&self) -> (u64, u64) {
        (
            self.slots.iter().flatten().count() as u64,
            self.slots.len() as u64,
        )
    }

    /// Where the shape's atoms stand.
    #[must_use]
    pub fn held(&self) -> WeightHeld {
        let (resident, all) = self.atoms();
        WeightHeld {
            shape: self.shape.id,
            resident: u32::try_from(resident).unwrap_or(u32::MAX),
            host: u32::try_from(all - resident).unwrap_or(u32::MAX),
        }
    }

    /// Each atom's reads since the last call, zeroed. The caller has
    /// drained every frame.
    /// Each atom's reads since the last call, zeroed: a group's count for
    /// every piece of it (a route reads every piece of its expert). The
    /// caller has drained every frame.
    pub fn uses(&mut self, into: &mut Vec<(WeightAtom, u32)>) -> Result<()> {
        let pieces = self.shape.pieces.max(1);
        let groups = self.slots.len() / pieces as usize;
        let mut counted = vec![0u8; groups * 4];
        self.hits.read(0, &mut counted)?;
        self.hits.write(0, &vec![0u8; groups * 4])?;
        for (group, reads) in counted.as_chunks::<4>().0.iter().enumerate() {
            let reads = u32::from_le_bytes(*reads);
            if reads == 0 {
                continue;
            }
            let first = u32::try_from(group).unwrap_or(u32::MAX) * pieces;
            into.extend((first..first + pieces).map(|atom| {
                (
                    WeightAtom {
                        shape: self.shape.id,
                        atom,
                    },
                    reads,
                )
            }));
        }
        Ok(())
    }

    /// The slot refs atoms are named, by atom.
    #[must_use]
    pub fn slots(&self) -> &[Option<u32>] {
        &self.slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_last_entry_copies_only_what_is_left() {
        let block = Plane {
            param: 0,
            atom: 0,
            entries: 3,
            stride: 256,
            pieces: 1,
            piece: 256,
            piece_rows: 4,
            at: 0,
            bytes: 640,
            counts: true,
        };
        assert_eq!(block.entry_bytes(0), 256);
        assert_eq!(block.entry_bytes(2), 128);
        assert_eq!(block.entry_at(2), 512);
        // Two experts of ten 8-byte rows, in four pieces of three rows.
        let expert = Plane {
            param: 0,
            atom: 0,
            entries: 8,
            stride: 80,
            pieces: 4,
            piece: 24,
            piece_rows: 3,
            at: 0,
            bytes: 160,
            counts: true,
        };
        assert_eq!(expert.entry_at(5), 80 + 24);
        assert_eq!(expert.entry_bytes(3), 8, "the last piece holds one row");
        assert_eq!(expert.entry_bytes(7), 8);
    }
}
