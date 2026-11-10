//! How the cache rows sit in the state pool's chunks: the compiler's chunk
//! recipe ([`poem_compiler::chunk::plan`]) read for this engine's paging.
//!
//! The pool is `total` chunks of `chunk_bytes`, chunk `p` at
//! `base + p * chunk_bytes`. A chunk holds slots of one shape. A **list**
//! slot is one kv page of every full-retention row, layer after layer, and a
//! **window list** slot one page of every windowed row; a row's key plane and then
//! its value plane run where the recipe puts the row.
//!
//! A paged kernel reads page `q` of a plane at `plane base + q * grain`
//! ([`kernels_cuda::KvPool::page_stride`]). `grain` is the largest step both
//! a chunk and the shape's slot are whole numbers of, so every slot of every
//! chunk has a `q`: slot `i` of physical chunk `p` is
//! `(p * chunk_bytes + i * slot_bytes) / grain`. The recipe makes a chunk a
//! whole number of list slots, so a list slot's `q` is its slot ref.

use engine::chunk::ShapeId;
use poem_compiler::chunk::{ChunkRecipe, RowId, SlotLayout};

use crate::error::{Fault, Result};

/// The list shape's id in the facts this engine states.
pub const LIST: ShapeId = ShapeId(0);
/// The window list's.
pub const WINDOW: ShapeId = ShapeId(1);
/// The fixed shape's.
pub const FIXED: ShapeId = ShapeId(2);
/// The rs list's.
pub const RS: ShapeId = ShapeId(3);

/// Which of a load's slot shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A page of every full-retention kv row.
    List,
    /// A page of every windowed kv row.
    Window,
    /// A working set's record of every recurrent row.
    Fixed,
    /// A page of the rs buffer: the rows a lane lands past its state before
    /// they fold, every recurrent layer's.
    Rs,
}

impl Kind {
    /// The shape a kv row's pages are slots of.
    #[must_use]
    pub const fn paged(windowed: bool) -> Kind {
        if windowed { Kind::Window } else { Kind::List }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Kind::List => "kv page",
            Kind::Window => "windowed page",
            Kind::Fixed => "state slot",
            Kind::Rs => "rs buffer page",
        }
    }
}

/// One slot shape: the pages of the kv rows that share a page table, or the
/// record of the recurrent rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotShape {
    pub id: ShapeId,
    /// Tokens a slot holds: the page size (0 for the fixed shape).
    pub tokens: u32,
    pub slot_bytes: u64,
    pub per_chunk: u32,
    /// The step page ids count in.
    pub grain: u64,
    /// The window its rows are read through; `None` for the list.
    pub window: Option<u32>,
    /// Each cache row laid in the slot, and where its run starts.
    pub rows: Vec<(u32, u64)>,
    /// Each compressor state laid in the slot, by the cache row its gather
    /// keys it by, and where its two planes start.
    pub pools: Vec<(u32, u64)>,
}

impl SlotShape {
    fn of(layout: &SlotLayout, chunk_bytes: u64, id: ShapeId) -> Result<SlotShape> {
        let (tokens, bytes, window, runs) = match layout {
            SlotLayout::List {
                tokens,
                bytes,
                rows,
                window,
            } => (*tokens, *bytes, *window, rows),
            SlotLayout::RsList {
                tokens,
                bytes,
                rows,
            } => (*tokens, *bytes, None, rows),
            SlotLayout::Fixed { .. } => {
                return Err(Fault::program(
                    "store::layout",
                    "a fixed slot read as a token slot",
                ));
            }
        };
        let per_chunk = u32::try_from(chunk_bytes / bytes.max(1)).unwrap_or(u32::MAX);
        if bytes == 0 || per_chunk == 0 {
            return Err(Fault::program(
                "store::layout",
                format!("a token slot of {bytes} bytes fits no chunk of {chunk_bytes}"),
            ));
        }
        Ok(SlotShape {
            id,
            tokens,
            slot_bytes: bytes,
            per_chunk,
            grain: gcd(chunk_bytes, bytes),
            window,
            rows: runs
                .iter()
                .filter_map(|run| match run.row {
                    RowId::Cache(row) => Some((row, run.offset)),
                    RowId::Rs(_) | RowId::Pool(_) => None,
                })
                .collect(),
            pools: runs
                .iter()
                .filter_map(|run| match run.row {
                    RowId::Pool(row) => Some((row, run.offset)),
                    RowId::Cache(_) | RowId::Rs(_) => None,
                })
                .collect(),
        })
    }

    /// Where the compressor state keyed by cache row `row` starts in a slot.
    #[must_use]
    pub fn pool_of(&self, row: u32) -> Option<u64> {
        self.pools
            .iter()
            .find(|(held, _)| *held == row)
            .map(|&(_, offset)| offset)
    }

    /// Where cache row `row`'s run starts in a slot.
    #[must_use]
    pub fn offset_of(&self, row: u32) -> Option<u64> {
        self.rows
            .iter()
            .find(|(held, _)| *held == row)
            .map(|&(_, offset)| offset)
    }

    /// Bytes from the pool's base to slot `index` of physical chunk `chunk`.
    #[must_use]
    pub fn slot_at(&self, chunk_bytes: u64, chunk: u32, index: u32) -> u64 {
        u64::from(chunk) * chunk_bytes + u64::from(index) * self.slot_bytes
    }

    /// The page id a kernel reads that slot by.
    #[must_use]
    pub fn page_id(&self, chunk_bytes: u64, chunk: u32, index: u32) -> u64 {
        self.slot_at(chunk_bytes, chunk, index) / self.grain
    }

    #[must_use]
    pub fn slot_ref(&self, chunk: u32, index: u32) -> u32 {
        chunk.saturating_mul(self.per_chunk).saturating_add(index)
    }
}

/// The slot shapes of a load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub chunk_bytes: u64,
    /// Chunks a segment, by the recipe's device profile.
    pub segment_chunks: u32,
    pub list: Option<SlotShape>,
    /// The window list: this engine pages every windowed row by one table.
    pub window: Option<SlotShape>,
    pub fixed: Option<FixedShape>,
    pub rs: Option<RsShape>,
    /// The tables the weight cache streams by their rows, a shape each.
    pub rows: Vec<RowsShape>,
}

/// The first weight shape's id; the shapes after it follow.
pub const WEIGHTS: u32 = 4;

/// A table a token-id gather reads, streamed by its rows: a slot holds a
/// block of `block_rows` rows of `row_bytes`, of the table's `rows`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowsShape {
    pub id: ShapeId,
    /// The trace's param the table is.
    pub param: u32,
    pub slot_bytes: u64,
    pub per_chunk: u32,
    pub block_rows: u32,
    pub row_bytes: u64,
    /// The table's rows.
    pub rows: u64,
    /// The blocks the table makes: its atoms.
    pub blocks: u32,
    /// The blocks one fire may read: what the cache never drops below.
    pub floor: u32,
}

/// The tables of `trace` this engine streams by their rows: read only by the
/// plain embedding gather (the one whose kernel reads a row table), a bf16
/// plane of its own with no companion, out of the checkpoint.
#[must_use]
pub fn streamed_rows(trace: &poem_ir::Trace) -> Vec<usize> {
    use poem_ir::residency::{Residency, residency};
    use poem_ir::{Def, Gather, Operands, Operation};
    // The HIP gather reads a table's rows contiguous: none streams there.
    if cfg!(feature = "rocm") {
        return Vec::new();
    }
    let classes = residency(trace);
    let mut out = Vec::new();
    let mut inputs = Vec::new();
    'params: for (at, param) in trace.params.iter().enumerate() {
        if !matches!(classes[at], Residency::Rows(_))
            || param.dtype != poem_ir::Dtype::Bf16
            || param.shape.len() != 2
            || param.source != poem_ir::ParamSource::Checkpoint
        {
            continue;
        }
        let mut read = false;
        for node in &trace.nodes {
            inputs.clear();
            node.op.inputs(&mut inputs);
            let reads = inputs.iter().any(|v| {
                matches!(trace.values.get(v.0 as usize).map(|d| &d.def),
                    Some(Def::Weight(w)) if *w as usize == at)
            });
            if !reads {
                continue;
            }
            match &node.op {
                Operation::Gather(Gather::Embed {
                    weights: None,
                    scales: None,
                    concat: false,
                    vocab,
                    ..
                }) if u64::from(*vocab) == param.shape[0] => read = true,
                _ => continue 'params,
            }
        }
        if read {
            out.push(at);
        }
    }
    out
}

/// The rs list: a slot holds `slot.tokens` buffered rows of every recurrent
/// layer's rs planes, a layer's rows `layers[k]` bytes into it and a token
/// `per_token[k]` bytes of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsShape {
    pub slot: SlotShape,
    /// Where each rs row's run starts in a slot and its bytes a token, in
    /// the recipe's order (a recurrent layer each, then what else buffers).
    pub layers: Vec<(u64, u64)>,
}

/// One recurrent row of a working set's record: `pieces` equal pieces of
/// `atoms_per_piece` atoms (the last padded past the row's `atoms`), each
/// `offset` bytes into a slot of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedRow {
    /// The cache row.
    pub row: u32,
    pub pieces: u32,
    pub atoms_per_piece: u64,
    pub atoms: u64,
    pub atom_bytes: u64,
    pub offset: u64,
    /// The record's slots its pieces lie in, one a piece.
    pub slots: Vec<u32>,
    /// Where its ids start among a lane's staged ids ([`FixedShape::tables`]).
    pub column: u32,
}

impl FixedRow {
    /// The bytes piece `piece` holds of the row.
    #[must_use]
    pub fn piece_bytes(&self, piece: u32) -> u64 {
        let start = (u64::from(piece) * self.atoms_per_piece).min(self.atoms);
        ((start + self.atoms_per_piece).min(self.atoms) - start) * self.atom_bytes
    }
}

/// The fixed shape: a working set's record of every recurrent row, `pieces`
/// slots of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedShape {
    pub slot: SlotShape,
    /// The slots a record takes.
    pub pieces: u32,
    pub rows: Vec<FixedRow>,
    /// The id tables a fire stages a lane: each a run of the record's slots
    /// some row's pieces lie in, the rows that share a run sharing a table.
    pub tables: Vec<Vec<u32>>,
}

impl FixedShape {
    fn of(recipe: &ChunkRecipe, layout: &SlotLayout) -> Result<FixedShape> {
        let chunk_bytes = recipe.chunk_bytes;
        let SlotLayout::Fixed {
            bytes,
            pieces_per_ws,
            rows: runs,
            ..
        } = layout
        else {
            return Err(Fault::program(
                "store::layout",
                "a token slot read as the fixed slot",
            ));
        };
        let per_chunk = u32::try_from(chunk_bytes / (*bytes).max(1)).unwrap_or(u32::MAX);
        let mut tables: Vec<Vec<u32>> = Vec::new();
        let mut columns: Vec<u32> = Vec::new();
        let mut rows = Vec::new();
        for run in runs {
            let atom = recipe.atoms.get(&run.row).ok_or_else(|| {
                Fault::program(
                    "store::layout",
                    format!("the chunk recipe states no atom for {:?}", run.row),
                )
            })?;
            match run.row {
                RowId::Cache(row) => {
                    let table = match tables.iter().position(|held| *held == run.slots) {
                        Some(at) => at,
                        None => {
                            columns.push(tables.iter().map(|t| t.len() as u32).sum());
                            tables.push(run.slots.clone());
                            tables.len() - 1
                        }
                    };
                    rows.push(FixedRow {
                        row,
                        pieces: run.pieces,
                        atoms_per_piece: run.atoms_per_piece,
                        atoms: atom.extent,
                        atom_bytes: atom.bytes,
                        offset: run.offset,
                        slots: run.slots.clone(),
                        column: columns[table],
                    });
                }
                RowId::Rs(_) => {
                    return Err(Fault::program(
                        "store::layout",
                        "an rs row in the fixed record: the rs buffer is a list shape",
                    ));
                }
                RowId::Pool(_) => {}
            }
        }
        Ok(FixedShape {
            slot: SlotShape {
                id: FIXED,
                tokens: 0,
                slot_bytes: *bytes,
                per_chunk: per_chunk.max(1),
                grain: gcd(chunk_bytes, *bytes),
                window: None,
                rows: Vec::new(),
                pools: Vec::new(),
            },
            pieces: (*pieces_per_ws).max(1),
            rows,
            tables,
        })
    }

    /// Cache row `row`'s place in the record.
    #[must_use]
    pub fn row(&self, row: u32) -> Option<&FixedRow> {
        self.rows.iter().find(|held| held.row == row)
    }

    /// The ids a fire stages a lane: every table's.
    #[must_use]
    pub fn ids_a_lane(&self) -> u32 {
        self.tables.iter().map(|table| table.len() as u32).sum()
    }
}

impl Layout {
    /// The recipe's token shapes. This engine pages every windowed row by one
    /// table, so a recipe with two window lists is refused, and by its kv
    /// page, so a window list of another page size is.
    pub fn of(recipe: &ChunkRecipe, streamed: &[usize]) -> Result<Layout> {
        let chunk_bytes = recipe.chunk_bytes;
        let list = recipe
            .list()
            .map(|layout| SlotShape::of(layout, chunk_bytes, LIST))
            .transpose()?;
        let mut windows = recipe.windows();
        let window = windows
            .next()
            .map(|layout| SlotShape::of(layout, chunk_bytes, WINDOW))
            .transpose()?;
        if let (Some(list), Some(window)) = (&list, &window)
            && list.tokens != window.tokens
        {
            return Err(Fault::Unbound {
                what: format!(
                    "a window list of {} tokens a slot beside kv pages of {}, which this \
                     engine's kernels page alike",
                    window.tokens, list.tokens
                ),
            });
        }
        if windows.next().is_some() {
            return Err(Fault::Unbound {
                what: "kv rows read through windows of two widths, which this engine pages \
                       by one windowed table"
                    .to_string(),
            });
        }
        let fixed = recipe
            .fixed()
            .map(|layout| FixedShape::of(recipe, layout))
            .transpose()?;
        let rs = recipe
            .rs_list()
            .map(|layout| -> Result<RsShape> {
                let mut slot = SlotShape::of(layout, chunk_bytes, RS)?;
                slot.rows.clear();
                Ok(RsShape {
                    slot,
                    layers: recipe
                        .rs
                        .runs
                        .iter()
                        .map(|run| (run.offset, run.per_token))
                        .collect(),
                })
            })
            .transpose()?;
        let (_, segment_chunks) = recipe.profile.segment(chunk_bytes);
        let mut rows = Vec::new();
        for &param in streamed {
            let slot = recipe.weights.iter().find(|slot| {
                slot.kind == poem_compiler::chunk::WeightKind::Rows
                    && slot.pieces == 1
                    && matches!(slot.banks.as_slice(), [bank] if bank.params == [param as u32])
            });
            // A table the recipe lays with others is served from its host
            // copy, no row of it cached.
            let Some(slot) = slot else { continue };
            let per_chunk = u32::try_from(chunk_bytes / slot.bytes.max(1)).unwrap_or(u32::MAX);
            let block_rows = u32::try_from(slot.atoms_per_slot)
                .unwrap_or(u32::MAX)
                .max(1);
            let table_rows = slot.banks[0].atoms;
            rows.push(RowsShape {
                id: ShapeId(WEIGHTS + rows.len() as u32),
                param: param as u32,
                slot_bytes: slot.bytes,
                per_chunk: per_chunk.max(1),
                block_rows,
                row_bytes: slot.atom_bytes,
                rows: table_rows,
                blocks: u32::try_from(table_rows.div_ceil(u64::from(block_rows)))
                    .unwrap_or(u32::MAX),
                floor: u32::try_from(slot.floor).unwrap_or(u32::MAX),
            });
        }
        Ok(Layout {
            chunk_bytes,
            segment_chunks: u32::try_from(segment_chunks).unwrap_or(u32::MAX).max(1),
            list,
            window,
            fixed,
            rs,
            rows,
        })
    }

    #[must_use]
    pub fn shape(&self, kind: Kind) -> Option<&SlotShape> {
        match kind {
            Kind::List => self.list.as_ref(),
            Kind::Window => self.window.as_ref(),
            Kind::Fixed => self.fixed.as_ref().map(|fixed| &fixed.slot),
            Kind::Rs => self.rs.as_ref().map(|rs| &rs.slot),
        }
    }

    /// Makes `kind`'s pages step `packed` bytes, the step of kernels that
    /// read a plane's pages one right after another (the HIP kernels: a page
    /// id times a page's rows of the plane). `strides` are the packed page
    /// strides of every plane in the shape's slot. One table names a slot
    /// for all of them, so they must agree, and a slot and a chunk must be
    /// whole numbers of the step for every slot to have a page id.
    pub fn pack(&mut self, kind: Kind, strides: &[u64]) -> Result<()> {
        let chunk_bytes = self.chunk_bytes;
        let Some(&packed) = strides.first() else {
            return Ok(());
        };
        let shape = match kind {
            Kind::List => self.list.as_mut(),
            Kind::Window => self.window.as_mut(),
            Kind::Fixed | Kind::Rs => None,
        };
        let Some(shape) = shape else {
            return Ok(());
        };
        if packed == 0
            || strides.iter().any(|&stride| stride != packed)
            || !shape.slot_bytes.is_multiple_of(packed)
            || !chunk_bytes.is_multiple_of(packed)
        {
            return Err(Fault::Unbound {
                what: format!(
                    "{}s whose planes step {strides:?} bytes a page in slots of {} bytes and \
                     chunks of {chunk_bytes}: this backend's kernels step every plane of a \
                     page table by one packed page, which a slot and a chunk must be whole \
                     numbers of",
                    kind.name(),
                    shape.slot_bytes
                ),
            });
        }
        shape.grain = packed;
        Ok(())
    }

    /// The chunks `slots` slots of a shape take.
    #[must_use]
    pub fn chunks_for(&self, kind: Kind, slots: u64) -> u64 {
        self.shape(kind)
            .map_or(0, |shape| slots.div_ceil(u64::from(shape.per_chunk.max(1))))
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout_of(list: SlotShape, chunk_bytes: u64) -> Layout {
        Layout {
            chunk_bytes,
            segment_chunks: 1,
            list: Some(list),
            window: None,
            fixed: None,
            rs: None,
            rows: Vec::new(),
        }
    }

    #[test]
    fn a_packed_shape_steps_its_pages_by_a_plane_and_refuses_planes_that_disagree() {
        // Four planes of 3 units a page in a slot of 12, sixteen slots a chunk.
        let unit = 256u64;
        let chunk = 192 * unit;
        let list = SlotShape {
            id: LIST,
            tokens: 16,
            slot_bytes: 12 * unit,
            per_chunk: 16,
            grain: gcd(chunk, 12 * unit),
            window: None,
            rows: vec![(0, 0), (1, 6 * unit)],
            pools: Vec::new(),
        };
        let mut layout = layout_of(list.clone(), chunk);
        layout
            .pack(Kind::List, &[3 * unit; 4])
            .expect("the planes pack");
        let packed = layout.list.as_ref().unwrap();
        assert_eq!(packed.grain, 3 * unit);
        for (chunk_id, index) in [(1u32, 0u32), (1, 5), (7, 15)] {
            // A plane `at` bytes into the slot: its page id's packed step
            // lands on the slot.
            let q = packed.page_id(chunk, chunk_id, index);
            assert_eq!(q * 3 * unit, packed.slot_at(chunk, chunk_id, index));
        }
        // Planes of two widths share no page id.
        let mut layout = layout_of(list.clone(), chunk);
        assert!(layout.pack(Kind::List, &[3 * unit, 6 * unit]).is_err());
        // Nor does a slot that is no whole number of the step.
        let mut layout = layout_of(list, chunk);
        assert!(layout.pack(Kind::List, &[5 * unit; 2]).is_err());
    }

    #[test]
    fn a_list_slot_is_read_by_its_slot_ref_and_a_window_slot_by_its_grain() {
        let chunk = 48 * 256;
        let list = SlotShape {
            id: LIST,
            tokens: 16,
            slot_bytes: 12 * 256,
            per_chunk: 4,
            grain: gcd(chunk, 12 * 256),
            window: None,
            rows: vec![(0, 0), (3, 1024)],
            pools: Vec::new(),
        };
        assert_eq!(list.grain, list.slot_bytes);
        assert_eq!(list.page_id(chunk, 5, 3), u64::from(list.slot_ref(5, 3)));
        assert_eq!(list.offset_of(3), Some(1024));
        assert_eq!(list.offset_of(1), None);

        // A window list slot of 20 units in a chunk of 48: two a chunk, eight spare.
        let window = SlotShape {
            id: WINDOW,
            tokens: 16,
            slot_bytes: 20 * 256,
            per_chunk: 2,
            grain: gcd(chunk, 20 * 256),
            window: Some(64),
            rows: vec![(1, 0)],
            pools: Vec::new(),
        };
        assert_eq!(window.grain, 4 * 256);
        for (chunk_id, index) in [(0u32, 0u32), (1, 1), (7, 0), (7, 1)] {
            let q = window.page_id(chunk, chunk_id, index);
            assert_eq!(q * window.grain, window.slot_at(chunk, chunk_id, index));
        }
    }
}
