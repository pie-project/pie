//! The weight cache's row-streamed tables (CONTRACT: `ChunkFacts::weights`,
//! `Engine::weight_residency`, `Engine::weight_uses`), on unified memory.
//!
//! A table read only by the plain embedding gather, bf16, out of the
//! checkpoint, is stated as a [`WeightKind::Rows`] shape when the compiler's
//! on-device Metal recipe gives it (alone, or with the tables gathered
//! alike: gemma's per-layer tables) a rows slot. Its atoms are blocks of
//! `block_rows` token ids; a slot holds one block of every table of the
//! group, table after table (each table's `block_rows` rows laid as in the
//! table, so a row is read the same way wherever its block lies).
//!
//! **The host copy is the table itself, read in place.** Metal memory is
//! unified, so there is no pinned staging copy as on CUDA: the gather reads
//! a block that is not in a pool slot from the table's own place in the
//! weights (`gather/embed.metal` through a [`RowTable`] of GPU addresses, one
//! a block). A cold load lands each such table in an allocation of its own
//! (`Weights::hosted`), so it can be given back; a warm load reads it out of
//! the checkpoint's mapped window, which the load keeps for its other planes.
//! Because the gather binds only the address table (never the table), no
//! recorded program holds the table, and dropping it needs no re-arm.
//!
//! **Residency** (`Engine::weight_residency`): every frame drains, the blocks
//! newly named a slot are blitted from the table into their slots, and every
//! address table is rewritten on the host before the next fire. Evacuation
//! rewrites them again. A use of a block that is not named reads the table
//! in place (no staging slot: on unified memory the in-place read is the
//! stream). **Held** (`WeightResidency::held`, a deployment that does not
//! stream) with every block of every shape named is the fast path: every
//! read lands in the pool, and each table's own allocation (the cold road's)
//! is given back; from then on a set naming a new slot for a block, or a
//! set that is not held, is refused. The kernels do not change between the
//! cases: the one `embed` kernel reads through the table whatever the
//! entries say, and reads the contiguous table where nothing is stated.
//!
//! The address table and hit counters are buffers the gather binds; the
//! table's own allocation is reached only through addresses, so it is kept
//! in the queue's residency set while it is read (the pool's sparse buffer
//! is in that set already). A device without that set (not Metal 4) states
//! no rows shape.

use engine::chunk::{ShapeId, WeightAtom, WeightHeld, WeightKind, WeightShape};
use kernels_metal::Tensor;
use kernels_metal::gather::RowTable;
use poem_compiler::chunk::ChunkRecipe;
use poem_ir::Trace;

use crate::device::alloc::slab_address;
use crate::device::ctx::Frame;
use crate::device::handles::Binding;
use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};

/// The first weight shape's id: past the state shapes (the kv list, the
/// fixed record, up to four window lists, the rs list).
pub const WEIGHTS: u32 = 8;

/// The tables of `trace` this engine streams by their rows: read only by the
/// plain embedding gather, a bf16 plane of its own with no companion, out
/// of the checkpoint (as `engine-cuda`'s `layout::streamed_rows`).
#[must_use]
pub fn streamed(trace: &Trace) -> Vec<usize> {
    use poem_ir::residency::{Residency, residency};
    use poem_ir::{Def, Gather, Operands, Operation};
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

/// One rows shape: the tables of a recipe rows slot, a slot one block of
/// `block_rows` ids of each, table after table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowsShape {
    pub id: ShapeId,
    /// The tables (trace params), in slot order.
    pub params: Vec<usize>,
    pub slot_bytes: u64,
    pub per_chunk: u32,
    pub block_rows: u32,
    /// The tables' rows (one count: tables gathered alike).
    pub rows: u64,
    /// Each table's bytes a row.
    pub row_bytes: Vec<u64>,
    /// Each table's run in a slot: `block_rows` of its rows from here.
    pub at: Vec<u64>,
    /// The blocks the tables make: the atoms.
    pub blocks: u32,
    pub floor: u32,
    /// The tables' names, for logs.
    pub name: String,
}

impl RowsShape {
    /// The rows shapes of `recipe` (chunks of `chunk_bytes`) whose every
    /// table is in `streamed`; a rows slot that holds a table this engine
    /// does not stream (or cuts its atoms) is not stated, its tables read
    /// whole.
    #[must_use]
    pub fn of(trace: &Trace, recipe: &ChunkRecipe, streamed: &[usize]) -> Vec<RowsShape> {
        let mut out = Vec::new();
        for slot in &recipe.weights {
            if slot.kind != poem_compiler::chunk::WeightKind::Rows || slot.pieces != 1 {
                continue;
            }
            let [bank] = slot.banks.as_slice() else {
                continue;
            };
            let params: Vec<usize> = bank.params.iter().map(|&p| p as usize).collect();
            if params.is_empty() || !params.iter().all(|p| streamed.contains(p)) {
                continue;
            }
            let rows = bank.atoms;
            if params
                .iter()
                .any(|&p| trace.params[p].shape.first().copied() != Some(rows))
            {
                continue;
            }
            let block_rows = u32::try_from(slot.atoms_per_slot).unwrap_or(0);
            if block_rows == 0 || rows == 0 {
                continue;
            }
            let row_bytes: Vec<u64> = params
                .iter()
                .map(|&p| trace.params[p].shape[1] * 2)
                .collect();
            let mut at = Vec::with_capacity(params.len());
            let mut used = 0u64;
            for bytes in &row_bytes {
                at.push(used);
                used += u64::from(block_rows) * bytes;
            }
            if used > slot.bytes || slot.bytes > recipe.chunk_bytes {
                continue;
            }
            let blocks = u32::try_from(rows.div_ceil(u64::from(block_rows))).unwrap_or(u32::MAX);
            let name = params
                .iter()
                .map(|&p| trace.params[p].name.as_str())
                .collect::<Vec<_>>()
                .join(",");
            out.push(RowsShape {
                id: ShapeId(WEIGHTS + u32::try_from(out.len()).unwrap_or(0)),
                params,
                slot_bytes: slot.bytes,
                per_chunk: u32::try_from(recipe.chunk_bytes / slot.bytes)
                    .unwrap_or(u32::MAX)
                    .max(1),
                block_rows,
                rows,
                row_bytes,
                at,
                blocks,
                floor: u32::try_from(slot.floor).unwrap_or(u32::MAX).min(blocks),
                name,
            });
        }
        out
    }

    #[must_use]
    pub fn facts(&self) -> WeightShape {
        WeightShape {
            id: self.id,
            kind: WeightKind::Rows,
            slot_bytes: self.slot_bytes,
            slots_per_chunk: self.per_chunk,
            count: self.blocks,
            floor_slots: self.floor,
            name: self.name.clone(),
        }
    }

    /// Table `k`'s rows in block `block` (the last block may be short).
    fn block_rows_of(&self, block: u32) -> u64 {
        let first = u64::from(block) * u64::from(self.block_rows);
        u64::from(self.block_rows).min(self.rows.saturating_sub(first))
    }
}

/// The byte offset in the pool of a slot ref.
type SlotAt<'a> = &'a dyn Fn(u32) -> Result<u64>;

/// A table's own copy: where the gather reads a block no slot holds.
#[derive(Debug, Clone)]
struct Host {
    binding: Binding,
    /// The handle the load bound the table at (released with the copy).
    handle: u32,
}

/// One rows shape's cache: its address tables, its hit counters and which
/// block each slot holds.
#[derive(Debug)]
pub struct RowCache {
    pub shape: RowsShape,
    /// Each table's own copy; `None` once given back.
    hosts: Vec<Option<Host>>,
    /// Each table's address table, a u64 a block, and its handle.
    tables: Vec<(Buffer, u32)>,
    /// A u32 a block, counted at the first table.
    hits: (Buffer, u32),
    /// The slot ref each block is named, if any.
    slots: Vec<Option<u32>>,
}

impl RowCache {
    /// The cache of `shape` over the tables `weights` binds, every block
    /// read from its table in place; its buffers bound in `handles` (before
    /// the load seals them).
    pub fn open(
        device: &Context,
        handles: &Handles,
        shape: RowsShape,
        table_handle: impl Fn(usize) -> Option<u32>,
    ) -> Result<RowCache> {
        let blocks = u64::from(shape.blocks);
        let mut hosts = Vec::with_capacity(shape.params.len());
        let mut tables = Vec::with_capacity(shape.params.len());
        for &param in &shape.params {
            let handle = table_handle(param).ok_or_else(|| Fault::Unbound {
                what: format!("param {param}, a row-streamed table this load binds no plane for"),
            })?;
            let binding = handles
                .get(handle)
                .map(|b| b.clone())
                .ok_or_else(|| Fault::Unbound {
                    what: format!("handle {handle} of row-streamed param {param}"),
                })?;
            hosts.push(Some(Host { binding, handle }));
            let table = Buffer::zeroed(device, blocks * 8)?;
            let bound = handles.bind(&table, 0, blocks * 8)?;
            tables.push((table, bound));
        }
        let hits = Buffer::zeroed(device, blocks * 4)?;
        let hits_handle = handles.bind(&hits, 0, blocks * 4)?;
        let mut cache = RowCache {
            slots: vec![None; shape.blocks as usize],
            shape,
            hosts,
            tables,
            hits: (hits, hits_handle),
        };
        cache.stage(None)?;
        Ok(cache)
    }

    /// The row table each of the shape's tables is read through, by param.
    pub fn row_tables(&self) -> impl Iterator<Item = (usize, RowTable)> + '_ {
        self.shape.params.iter().enumerate().map(|(k, &param)| {
            let (_, handle) = self.tables[k];
            let blocks = self.shape.blocks;
            (
                param,
                RowTable {
                    at: Tensor::new(handle, blocks, 1, poem_ir::Dtype::I64),
                    block_rows: self.shape.block_rows,
                    hits: (k == 0)
                        .then(|| Tensor::new(self.hits.1, blocks, 1, poem_ir::Dtype::U32)),
                },
            )
        })
    }

    /// The table's own copies the queue must keep resident: the slabs the
    /// gather reaches only through addresses.
    pub fn host_slabs(&self) -> impl Iterator<Item = &crate::device::alloc::Slab> {
        self.hosts.iter().flatten().map(|host| host.binding.slab())
    }

    /// Rewrites every address table: a block named a slot reads it through
    /// `pool` (base address and the byte offset of a slot ref), the others
    /// read their table in place. The caller has drained every frame.
    fn stage(&mut self, pool: Option<(u64, SlotAt<'_>)>) -> Result<()> {
        for (k, (table, _)) in self.tables.iter_mut().enumerate() {
            let host = self.hosts[k].as_ref();
            let mut bytes = Vec::with_capacity(self.slots.len() * 8);
            for (block, slot) in self.slots.iter().enumerate() {
                let address = match (slot, pool) {
                    (Some(slot), Some((base, offset))) => base + offset(*slot)? + self.shape.at[k],
                    _ => {
                        let Some(host) = host else {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "block {block} of param {}, named no slot after its table's \
                                     own copy was given back",
                                    self.shape.params[k]
                                ),
                            });
                        };
                        slab_address(host.binding.slab())
                            + host.binding.offset()
                            + block as u64
                                * u64::from(self.shape.block_rows)
                                * self.shape.row_bytes[k]
                    }
                };
                bytes.extend_from_slice(&address.to_le_bytes());
            }
            table.write(0, &bytes)?;
        }
        Ok(())
    }

    /// Takes the shape's part of a residency: blocks newly named a slot
    /// are blitted from their tables into `frame` (the pool `into`, a slot
    /// ref's byte offset by `offset`); answers the copies. The tables are
    /// rewritten by [`RowCache::restage`] once `frame` is committed.
    pub fn take(
        &mut self,
        frame: &mut Frame,
        named: Vec<Option<u32>>,
        into: &Buffer,
        offset: &dyn Fn(u32) -> Result<u64>,
    ) -> Result<u32> {
        let mut copies = 0;
        for (block, slot) in named.iter().enumerate() {
            let Some(slot) = *slot else { continue };
            if self.slots[block] == Some(slot) {
                continue;
            }
            let at = offset(slot)?;
            let block32 = u32::try_from(block).unwrap_or(u32::MAX);
            let rows = self.shape.block_rows_of(block32);
            for (k, host) in self.hosts.iter().enumerate() {
                let Some(host) = host else {
                    return Err(Fault::Unbound {
                        what: format!(
                            "block {block} of param {} named a new slot after the held set's \
                             tables were given back: a held block stays where it was named",
                            self.shape.params[k]
                        ),
                    });
                };
                let row = self.shape.row_bytes[k];
                let from =
                    host.binding.offset() + block as u64 * u64::from(self.shape.block_rows) * row;
                frame.copy(
                    host.binding.slab(),
                    from,
                    into.slab(),
                    at + self.shape.at[k],
                    rows * row,
                )?;
            }
            copies += 1;
        }
        self.slots = named;
        Ok(copies)
    }

    /// Rewrites the address tables for the slots the blocks are named now,
    /// through the pool at `base` (see [`RowCache::stage`]).
    pub fn restage(&mut self, base: u64, offset: &dyn Fn(u32) -> Result<u64>) -> Result<()> {
        self.stage(Some((base, offset)))
    }

    /// Whether every block is named a slot.
    #[must_use]
    pub fn whole(&self) -> bool {
        self.slots.iter().all(Option::is_some)
    }

    /// Gives back the tables' own copies: answers each param's handle and
    /// slab, for the caller to unbind and let go. Every block is named.
    pub fn release(&mut self) -> Vec<(usize, u32, crate::device::alloc::Slab)> {
        if !self.whole() {
            return Vec::new();
        }
        let params = self.shape.params.clone();
        self.hosts
            .iter_mut()
            .zip(params)
            .filter_map(|(host, param)| {
                host.take()
                    .map(|host| (param, host.handle, host.binding.slab().clone()))
            })
            .collect()
    }

    /// Whether the tables' own copies were given back.
    #[must_use]
    pub fn released(&self) -> bool {
        self.hosts.iter().all(Option::is_none)
    }

    /// The blocks named a slot, and all of them.
    #[must_use]
    pub fn blocks(&self) -> (u64, u64) {
        (
            self.slots.iter().flatten().count() as u64,
            self.slots.len() as u64,
        )
    }

    /// Where the shape's blocks stand.
    #[must_use]
    pub fn held(&self) -> WeightHeld {
        let (resident, all) = self.blocks();
        WeightHeld {
            shape: self.shape.id,
            resident: u32::try_from(resident).unwrap_or(u32::MAX),
            host: u32::try_from(all - resident).unwrap_or(u32::MAX),
        }
    }

    /// Each block's reads since the last call, zeroed. The caller has
    /// drained every frame.
    pub fn uses(&mut self, into: &mut Vec<(WeightAtom, u32)>) -> Result<()> {
        let blocks = self.slots.len();
        let mut counted = vec![0u8; blocks * 4];
        self.hits.0.read(0, &mut counted)?;
        self.hits.0.write(0, &vec![0u8; blocks * 4])?;
        for (atom, reads) in counted.as_chunks::<4>().0.iter().enumerate() {
            let reads = u32::from_le_bytes(*reads);
            if reads > 0 {
                into.push((
                    WeightAtom {
                        shape: self.shape.id,
                        atom: u32::try_from(atom).unwrap_or(u32::MAX),
                    },
                    reads,
                ));
            }
        }
        Ok(())
    }

    /// The slot refs blocks are named, by block.
    #[must_use]
    pub fn slots(&self) -> &[Option<u32>] {
        &self.slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_last_block_copies_only_its_rows() {
        let shape = RowsShape {
            id: ShapeId(WEIGHTS),
            params: vec![0],
            slot_bytes: 1024,
            per_chunk: 4,
            block_rows: 4,
            rows: 10,
            row_bytes: vec![64],
            at: vec![0],
            blocks: 3,
            floor: 3,
            name: String::new(),
        };
        assert_eq!(shape.block_rows_of(0), 4);
        assert_eq!(shape.block_rows_of(2), 2);
    }
}
