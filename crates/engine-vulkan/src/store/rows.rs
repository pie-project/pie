//! The weight cache's row tables (SPEC "Weight cache"): bf16 tables a plain
//! token-id gather alone reads (per-layer embeddings) streamed by their rows
//! through read-only weight slots of the chunk pool.
//!
//! The recipe groups the tables gathered alike into one weight shape: an
//! atom is a token id's row of each table, side by side in the bank's param
//! order, a slot `block_rows` atoms. The tables themselves stay on the host
//! (the artifact, mapped): the runtime names which blocks are resident and
//! in which slot ([`RowCaches::residency`]); the engine fills a newly named
//! slot from the host and points the block's entry at it. A fire reads a
//! row through its block's entry ([`kernels_slang::gather::embed::RowTable`]);
//! a row of a block not resident is staged for that fire from the host,
//! never cached: where the gather reads the fire's own token ids (the host
//! wrote them) the host lays the rows of every token of the fire in the
//! stage of the fire's arm before it is submitted ([`RowCaches::prestage`]),
//! a ring of stages an arm each, so the fire binds only buffers that stand
//! across fires and records like any other; else the ids are read back
//! after the frame so far lands. Each read counts a use of its block, which
//! [`RowCaches::uses`] reports.

use std::collections::BTreeMap;

use engine::chunk::{ShapeId, WeightAtom, WeightHeld, WeightKind, WeightResidency, WeightShape};
use kernels_slang::Tensor;
use poem_compiler::chunk::{ChunkRecipe, WeightKind as RecipeKind};
use poem_ir::{Def, Dtype, Gather, Operands, Operation, ParamSource, Trace};

use crate::device::ctx::Frame;
use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};
use crate::experts::{HostPlane, HostTier};
use crate::store::Pools;

/// The uses a weight shape's atoms took, and where each shape's lie.
pub type Uses = (Vec<(WeightAtom, u32)>, Vec<WeightHeld>);

/// A block no slot holds.
pub const NOT_HELD: u32 = u32::MAX;

/// The first weight shape's id; the shapes after it follow.
pub const WEIGHTS: u32 = 8;

/// Bytes one fill moves through its host-visible upload at most.
const UPLOAD: u64 = 64 << 20;

/// The tables a plain token-id gather alone reads: bf16, two axes, from
/// the checkpoint, every reader a `Gather::Embed` with no weights, scales or
/// concat over the table's own rows.
#[must_use]
pub fn streamed(trace: &Trace) -> Vec<usize> {
    use poem_ir::residency::{Residency, residency};
    let classes = residency(trace);
    let mut out = Vec::new();
    let mut inputs = Vec::new();
    'params: for (at, param) in trace.params.iter().enumerate() {
        if !matches!(classes[at], Residency::Rows(_))
            || param.dtype != Dtype::Bf16
            || param.shape.len() != 2
            || param.source != ParamSource::Checkpoint
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

/// One table of a row shape: its param, where its row lies in an atom and
/// the bytes a row takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Table {
    pub param: usize,
    pub origin: u64,
    pub row_bytes: u64,
}

/// The tables gathered alike, streamed by blocks of `block_rows` ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowsShape {
    pub id: ShapeId,
    pub tables: Vec<Table>,
    pub slot_bytes: u64,
    pub per_chunk: u32,
    pub block_rows: u32,
    pub atom_bytes: u64,
    /// The ids (rows of each table).
    pub rows: u64,
    pub blocks: u32,
    pub floor: u32,
    pub name: String,
}

impl RowsShape {
    /// The recipe's row slots whose tables are all `streamed`, a shape each.
    pub fn of(trace: &Trace, recipe: &ChunkRecipe, streamed: &[usize]) -> Result<Vec<RowsShape>> {
        let mut shapes = Vec::new();
        for slot in &recipe.weights {
            if slot.kind != RecipeKind::Rows || slot.pieces != 1 {
                continue;
            }
            let [bank] = slot.banks.as_slice() else {
                continue;
            };
            if bank.params.is_empty()
                || !bank
                    .params
                    .iter()
                    .all(|&p| streamed.contains(&(p as usize)))
            {
                continue;
            }
            let mut origin = 0u64;
            let mut tables = Vec::with_capacity(bank.params.len());
            for &p in &bank.params {
                let param = &trace.params[p as usize];
                let row_bytes = param.shape[1] * 2;
                if !origin.is_multiple_of(4) || !row_bytes.is_multiple_of(4) {
                    return Err(Fault::Unbound {
                        what: format!(
                            "row table `{}`, whose {row_bytes}-byte rows at byte {origin} of an \
                             atom are no whole words",
                            param.name
                        ),
                    });
                }
                tables.push(Table {
                    param: p as usize,
                    origin,
                    row_bytes,
                });
                origin += row_bytes;
            }
            if origin != slot.atom_bytes {
                return Err(Fault::Unbound {
                    what: format!(
                        "the recipe's row atom of {} bytes, where its tables' rows take {origin}",
                        slot.atom_bytes
                    ),
                });
            }
            let block_rows = u32::try_from(slot.atoms_per_slot).unwrap_or(u32::MAX);
            let id = ShapeId(WEIGHTS + u32::try_from(shapes.len()).unwrap_or(u32::MAX));
            shapes.push(RowsShape {
                id,
                name: trace.params[tables[0].param].name.clone(),
                tables,
                slot_bytes: slot.bytes,
                per_chunk: u32::try_from(slot.per_chunk(recipe.chunk_bytes)).unwrap_or(u32::MAX),
                block_rows,
                atom_bytes: slot.atom_bytes,
                rows: bank.atoms,
                blocks: u32::try_from(slot.slots).unwrap_or(u32::MAX),
                floor: u32::try_from(slot.floor).unwrap_or(u32::MAX),
            });
        }
        Ok(shapes)
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
            seated: None,
        }
    }
}

/// One row shape's cache: which slot each block lies in, the device's view
/// of that, the uses counted and the rows staged a fire.
#[derive(Debug)]
struct Cache {
    shape: RowsShape,
    /// Each block's slot ref, [`NOT_HELD`] where none.
    slots: Vec<u32>,
    /// Each block's pool unit as kernels read it (`blocks`' contents).
    blocks: Buffer,
    /// A use a row read, a block each (the first table's reads).
    hits: Buffer,
    /// Each table's rows staged for a fire whose ids are read back, and
    /// each row's stage row.
    stage: Vec<Buffer>,
    staged: Vec<Buffer>,
    /// The same an arm each, laid by the host before the arm's fire by the
    /// fire's own token ids ([`RowCaches::prestage`]): `[arm][table]`.
    ring: Vec<(Vec<Buffer>, Vec<Buffer>)>,
    /// The set is the load's for life.
    held: bool,
    /// The bytes a pool unit of this shape counts: `gcd(chunk, slot)`.
    unit: u64,
}

impl Cache {
    fn resident(&self) -> u32 {
        u32::try_from(self.slots.iter().filter(|&&s| s != NOT_HELD).count()).unwrap_or(u32::MAX)
    }
}

/// The load's row tables.
#[derive(Debug)]
pub struct RowCaches {
    caches: Vec<Cache>,
    /// Each streamed table's cache and its place among the cache's tables.
    by_param: BTreeMap<usize, (usize, usize)>,
    /// Where a table that counts no use lands its counts.
    spare_hits: Buffer,
    rows: u32,
    /// Every gather of a streamed table reads the fire's token ids.
    by_tokens: bool,
}

impl RowCaches {
    /// The caches of `shapes` of `trace`, nothing resident; a fire stages up
    /// to `rows` rows a table, in a stage of each of `arms` fire arms.
    pub fn new(
        device: &Context,
        trace: &Trace,
        shapes: Vec<RowsShape>,
        rows: u32,
        chunk_bytes: u64,
        arms: usize,
    ) -> Result<RowCaches> {
        let mut caches = Vec::with_capacity(shapes.len());
        let mut by_param = BTreeMap::new();
        let mut widest = 0u32;
        for shape in shapes {
            let blocks = shape.blocks.max(1);
            widest = widest.max(blocks);
            let mut units = Buffer::host(device, u64::from(blocks) * 4)?;
            units.write(0, &vec![0xffu8; blocks as usize * 4])?;
            for (k, table) in shape.tables.iter().enumerate() {
                by_param.insert(table.param, (caches.len(), k));
            }
            caches.push(Cache {
                slots: vec![NOT_HELD; shape.blocks as usize],
                blocks: units,
                hits: Buffer::host(device, u64::from(blocks) * 4)?,
                stage: shape
                    .tables
                    .iter()
                    .map(|t| Buffer::host(device, u64::from(rows.max(1)) * t.row_bytes))
                    .collect::<Result<_>>()?,
                staged: shape
                    .tables
                    .iter()
                    .map(|_| Buffer::host(device, u64::from(rows.max(1)) * 4))
                    .collect::<Result<_>>()?,
                ring: (0..arms)
                    .map(|_| {
                        let stage = shape
                            .tables
                            .iter()
                            .map(|t| Buffer::host(device, u64::from(rows.max(1)) * t.row_bytes))
                            .collect::<Result<_>>()?;
                        let staged = shape
                            .tables
                            .iter()
                            .map(|_| Buffer::host(device, u64::from(rows.max(1)) * 4))
                            .collect::<Result<_>>()?;
                        Ok((stage, staged))
                    })
                    .collect::<Result<_>>()?,
                held: false,
                unit: crate::store::layout::gcd(chunk_bytes, shape.slot_bytes),
                shape,
            });
        }
        let by_tokens = trace.nodes.iter().all(|node| match &node.op {
            Operation::Gather(Gather::Embed { table, ids, .. }) => {
                let streamed = matches!(trace.values.get(table.0 as usize).map(|d| &d.def),
                    Some(Def::Weight(w)) if by_param.contains_key(&(*w as usize)));
                !streamed
                    || matches!(
                        trace.values.get(ids.0 as usize).map(|d| &d.def),
                        Some(Def::Input(poem_ir::RuntimeInput::Tokens))
                    )
            }
            _ => true,
        });
        Ok(RowCaches {
            caches,
            by_param,
            spare_hits: Buffer::host(device, u64::from(widest.max(1)) * 4)?,
            rows,
            by_tokens,
        })
    }

    /// Whether every gather of a streamed table reads the fire's own token
    /// ids, which the host stages a fire's misses by ([`RowCaches::prestage`]).
    #[must_use]
    pub fn by_tokens(&self) -> bool {
        self.by_tokens
    }

    /// Lays in arm `arm`'s stage the rows of every token of a fire (`tokens`,
    /// its rows in order) whose block no slot holds, each at its row, and
    /// each row's stage row (`NOT_HELD` where resident): what that fire's
    /// gathers of its token ids read for a block not resident. The arm's
    /// last fire has landed (its inputs are written now too).
    pub fn prestage(
        &self,
        arm: usize,
        host: &HostTier,
        planes: &BTreeMap<usize, &HostPlane>,
        tokens: &[i32],
    ) -> Result<()> {
        if tokens.len() > self.rows as usize {
            return Err(Fault::Ceiling {
                what: "rows a fire stages of a row table",
                need: tokens.len() as u64,
                have: u64::from(self.rows),
            });
        }
        for cache in &self.caches {
            if !cache.slots.contains(&NOT_HELD) {
                continue;
            }
            let (stage, staged) = cache.ring.get(arm).ok_or(Fault::Ceiling {
                what: "fire arms a row table stages for",
                need: arm as u64 + 1,
                have: cache.ring.len() as u64,
            })?;
            let mut rows = vec![NOT_HELD; tokens.len()];
            let mut last = 0usize;
            let ids: Vec<u64> = tokens
                .iter()
                .map(|&raw| {
                    if raw >= 0 && u64::from(raw.unsigned_abs()) < cache.shape.rows {
                        u64::from(raw.unsigned_abs())
                    } else {
                        0
                    }
                })
                .collect();
            for (n, &id) in ids.iter().enumerate() {
                let block = id / u64::from(cache.shape.block_rows.max(1));
                if cache.slots[block as usize] == NOT_HELD {
                    rows[n] = u32::try_from(n).unwrap_or(NOT_HELD);
                    last = n + 1;
                }
            }
            for (k, table) in cache.shape.tables.iter().enumerate() {
                let plane = planes.get(&table.param).ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "row table param {}, which the host tier does not hold",
                        table.param
                    ),
                })?;
                let row_bytes = usize::try_from(table.row_bytes).unwrap_or(0);
                let mut image = vec![0u8; last * row_bytes];
                for (n, &id) in ids[..last].iter().enumerate() {
                    if rows[n] != NOT_HELD {
                        plane.row_into(
                            host.map(),
                            id,
                            &mut image[n * row_bytes..(n + 1) * row_bytes],
                        )?;
                    }
                }
                stage[k].write_shared(0, &image)?;
                let bytes: Vec<u8> = rows.iter().flat_map(|r| r.to_le_bytes()).collect();
                staged[k].write_shared(0, &bytes)?;
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.caches.is_empty()
    }

    /// The weight shapes the runtime counts.
    #[must_use]
    pub fn facts(&self) -> Vec<WeightShape> {
        self.caches.iter().map(|c| c.shape.facts()).collect()
    }

    /// Whether every block of every table is resident: no fire stages.
    #[must_use]
    pub fn whole(&self) -> bool {
        self.caches.iter().all(|c| !c.slots.contains(&NOT_HELD))
    }

    #[must_use]
    pub fn streams(&self, param: usize) -> bool {
        self.by_param.contains_key(&param)
    }

    /// Whether any slot of these caches lies in logical chunks
    /// `chunks` (a seat run an engine's own fire writes).
    #[must_use]
    pub fn holds_in(&self, chunks: std::ops::Range<u32>) -> bool {
        self.caches.iter().any(|c| {
            c.slots
                .iter()
                .any(|&s| s != NOT_HELD && chunks.contains(&(s / c.shape.per_chunk.max(1))))
        })
    }

    /// Takes `residency` as the whole set (its atoms of these shapes): fills
    /// each slot newly named from
    /// the host (`host`, the tables' planes by param in `planes`) through
    /// `frame`s of its own, and rewrites the blocks' entries. The caller has
    /// drained every fire. The slots it filled.
    pub fn residency(
        &mut self,
        device: &Context,
        pools: &Pools,
        host: &HostTier,
        planes: &BTreeMap<usize, &HostPlane>,
        residency: &WeightResidency,
    ) -> Result<usize> {
        let mut named: Vec<Vec<u32>> = self
            .caches
            .iter()
            .map(|c| vec![NOT_HELD; c.shape.blocks as usize])
            .collect();
        for resident in &residency.resident {
            let Some(at) = self
                .caches
                .iter()
                .position(|c| c.shape.id == resident.atom.shape)
            else {
                continue;
            };
            let cache = &self.caches[at];
            let block = resident.atom.atom as usize;
            if block >= named[at].len() {
                return Err(Fault::Ceiling {
                    what: "blocks of a row table",
                    need: u64::from(resident.atom.atom) + 1,
                    have: u64::from(cache.shape.blocks),
                });
            }
            pools.weight_slot(cache.shape.per_chunk, cache.shape.slot_bytes, resident.slot)?;
            named[at][block] = resident.slot;
        }
        for (cache, named) in self.caches.iter().zip(&named) {
            if cache.held
                && (!residency.held || named.iter().zip(&cache.slots).any(|(n, s)| n != s))
            {
                return Err(Fault::Unbound {
                    what: format!(
                        "a weight residency that moves row table `{}`, held for the load's life",
                        cache.shape.name
                    ),
                });
            }
        }
        let mut fills: Vec<(u64, usize, u32)> = Vec::new();
        for (at, (cache, named)) in self.caches.iter().zip(&named).enumerate() {
            for (block, (&want, &have)) in named.iter().zip(&cache.slots).enumerate() {
                if want != NOT_HELD && want != have {
                    let to =
                        pools.weight_slot(cache.shape.per_chunk, cache.shape.slot_bytes, want)?;
                    fills.push((to, at, u32::try_from(block).unwrap_or(u32::MAX)));
                }
            }
        }
        self.fill(device, pools, host, planes, &mut fills)?;
        let filled = fills.len();
        for (cache, named) in self.caches.iter_mut().zip(named) {
            cache.slots = named;
            cache.held = residency.held;
        }
        self.restage(pools)?;
        Ok(filled)
    }

    /// Fills the slots `fills` names (virtual byte, cache, block) with their
    /// blocks' rows from the host, in uploads of [`UPLOAD`] bytes at most.
    fn fill(
        &self,
        device: &Context,
        pools: &Pools,
        host: &HostTier,
        planes: &BTreeMap<usize, &HostPlane>,
        fills: &mut [(u64, usize, u32)],
    ) -> Result<()> {
        fills.sort_unstable();
        let mut start = 0usize;
        while start < fills.len() {
            let mut bytes = 0u64;
            let mut end = start;
            while end < fills.len() {
                let slot = self.caches[fills[end].1].shape.slot_bytes;
                if bytes > 0 && bytes + slot > UPLOAD {
                    break;
                }
                bytes += slot;
                end += 1;
            }
            let mut upload = Buffer::host(device, bytes.max(4))?;
            let mut at = 0u64;
            let mut image = Vec::new();
            for &(_, cache, block) in &fills[start..end] {
                let shape = &self.caches[cache].shape;
                image.clear();
                image.resize(usize::try_from(shape.slot_bytes).unwrap_or(0), 0);
                for k in 0..u64::from(shape.block_rows) {
                    let id = u64::from(block) * u64::from(shape.block_rows) + k;
                    if id >= shape.rows {
                        break;
                    }
                    for table in &shape.tables {
                        let plane = planes.get(&table.param).ok_or_else(|| Fault::Unbound {
                            what: format!(
                                "row table param {}, which the host tier does not hold",
                                table.param
                            ),
                        })?;
                        let from = usize::try_from(k * shape.atom_bytes + table.origin)
                            .unwrap_or(usize::MAX);
                        let len = usize::try_from(table.row_bytes).unwrap_or(0);
                        plane.row_into(host.map(), id, &mut image[from..from + len])?;
                    }
                }
                upload.write(at, &image)?;
                at += shape.slot_bytes;
            }
            // One copy of many regions a segment, slots side by side merged.
            let mut frame = device.frame()?;
            let mut at = 0u64;
            let mut regions: Vec<(u64, u64, u64)> = Vec::new();
            let mut into: Option<(&Buffer, u64)> = None;
            for &(to, cache, _) in &fills[start..end] {
                let slot = self.caches[cache].shape.slot_bytes;
                let (buffer, local) = pools.locate(to)?;
                let segment = to - local;
                if into.is_some_and(|(_, held)| held != segment) {
                    let (buffer, _) = into.take().expect("checked");
                    frame.copy_regions(&upload, buffer, &regions)?;
                    regions.clear();
                }
                into = Some((buffer, segment));
                match regions.last_mut() {
                    Some(last) if last.0 + last.2 == at && last.1 + last.2 == local => {
                        last.2 += slot;
                    }
                    _ => regions.push((at, local, slot)),
                }
                at += slot;
            }
            if let Some((buffer, _)) = into {
                frame.copy_regions(&upload, buffer, &regions)?;
            }
            frame.commit()?;
            start = end;
        }
        Ok(())
    }

    /// Rewrites every block's entry from its slot ref through the pool's
    /// chunk table: after a residency, and after an evacuation moves a
    /// chunk. The caller has drained every fire that reads the entries.
    pub fn restage(&mut self, pools: &Pools) -> Result<()> {
        for cache in &mut self.caches {
            let mut units = Vec::with_capacity(cache.slots.len() * 4);
            for &slot in &cache.slots {
                let unit = if slot == NOT_HELD {
                    NOT_HELD
                } else {
                    let at =
                        pools.weight_slot(cache.shape.per_chunk, cache.shape.slot_bytes, slot)?;
                    u32::try_from(at / cache.unit).map_err(|_| Fault::Ceiling {
                        what: "pool units a row table's entry names",
                        need: at / cache.unit,
                        have: u64::from(u32::MAX),
                    })?
                };
                units.extend_from_slice(&unit.to_le_bytes());
            }
            cache.blocks.write(0, &units)?;
        }
        Ok(())
    }

    /// The uses counted since the last call, and where each shape's blocks
    /// lie; the counts start over. The caller has drained every fire.
    pub fn uses(&mut self) -> Result<Uses> {
        let mut uses = Vec::new();
        let mut held = Vec::with_capacity(self.caches.len());
        for cache in &mut self.caches {
            let blocks = cache.shape.blocks as usize;
            let mut raw = vec![0u8; blocks * 4];
            cache.hits.read(0, &mut raw)?;
            for (block, word) in raw.as_chunks::<4>().0.iter().enumerate() {
                let n = u32::from_le_bytes(*word);
                if n > 0 {
                    uses.push((
                        WeightAtom {
                            shape: cache.shape.id,
                            atom: u32::try_from(block).unwrap_or(u32::MAX),
                        },
                        n,
                    ));
                }
            }
            cache.hits.zero_span(0, raw.len() as u64)?;
            let resident = cache.resident();
            held.push(WeightHeld {
                shape: cache.shape.id,
                resident,
                host: cache.shape.blocks - resident,
            });
        }
        Ok((uses, held))
    }

    /// Table `param`'s row table for a fire gathering `ids`, the pool as
    /// `pool` binds it: rows of blocks not resident staged from the host
    /// (`host`, the table's plane `plane`) for this fire. Where the ids are
    /// the fire's tokens from row `first` (`ring`: `(arm, first)`), the host
    /// staged them in the arm's stage before the fire
    /// ([`RowCaches::prestage`]); else after the frame recorded so far lands
    /// (`ids` may be written by it), the ids are read back and staged. `None`
    /// where `param` is not streamed.
    #[allow(clippy::too_many_arguments)]
    pub fn table(
        &self,
        handles: &Handles,
        host: &HostTier,
        plane: &HostPlane,
        param: usize,
        ids: Tensor,
        ring: Option<(usize, u32)>,
        read: &std::cell::RefCell<Option<(u32, Vec<i32>)>>,
        pool: kernels_slang::Segments,
    ) -> Result<Option<kernels_slang::gather::embed::RowTable>> {
        let Some(&(at, k)) = self.by_param.get(&param) else {
            return Ok(None);
        };
        let cache = &self.caches[at];
        let table = cache.shape.tables[k];
        let rows = ids.rows;
        if rows > self.rows {
            return Err(Fault::Ceiling {
                what: "rows a fire stages of a row table",
                need: u64::from(rows),
                have: u64::from(self.rows),
            });
        }
        let missing = cache.slots.contains(&NOT_HELD);
        if missing && ring.is_none() {
            let mut held = read.borrow_mut();
            if held.as_ref().is_none_or(|(buf, _)| *buf != ids.buf) {
                crate::probe::with_frame(Frame::flush_settled).ok_or_else(|| {
                    Fault::Residency("no frame is recording on this thread".into())
                })??;
                let raw = handles.read(ids.buf, u64::from(rows) * 4)?;
                let got = raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|w| i32::from_le_bytes(*w))
                    .collect();
                *held = Some((ids.buf, got));
            }
            let ids_host: &[i32] = &held.as_ref().expect("read above").1;
            let row_bytes = usize::try_from(table.row_bytes).unwrap_or(0);
            let mut stage_rows = vec![NOT_HELD; rows as usize];
            let mut image = vec![0u8; ids_host.len() * row_bytes];
            let mut last = 0usize;
            for (n, &raw) in ids_host.iter().enumerate() {
                let id = if raw >= 0 && u64::from(raw.unsigned_abs()) < cache.shape.rows {
                    u64::from(raw.unsigned_abs())
                } else {
                    0
                };
                let block = id / u64::from(cache.shape.block_rows.max(1));
                if cache.slots[block as usize] != NOT_HELD {
                    continue;
                }
                plane.row_into(
                    host.map(),
                    id,
                    &mut image[n * row_bytes..(n + 1) * row_bytes],
                )?;
                stage_rows[n] = u32::try_from(n).unwrap_or(NOT_HELD);
                last = n + 1;
            }
            let bytes: Vec<u8> = stage_rows.iter().flat_map(|r| r.to_le_bytes()).collect();
            cache.stage[k].write_shared(0, &image[..last * row_bytes])?;
            cache.staged[k].write_shared(0, &bytes)?;
        }
        let (stage, staged, first) = match ring {
            Some((arm, first)) => {
                let (stage, staged) = cache.ring.get(arm).ok_or(Fault::Ceiling {
                    what: "fire arms a row table stages for",
                    need: arm as u64 + 1,
                    have: cache.ring.len() as u64,
                })?;
                (&stage[k], &staged[k], first)
            }
            None => (&cache.stage[k], &cache.staged[k], 0),
        };
        let bind = |buffer: &Buffer, len: u64, width: u32, dtype: Dtype| -> Result<Tensor> {
            let handle = handles.bind(buffer, 0, len.max(4))?;
            Ok(Tensor::new(
                handle,
                u32::try_from(len / (u64::from(width) * 4).max(1)).unwrap_or(u32::MAX),
                width,
                dtype,
            ))
        };
        let words = u32::try_from(table.row_bytes / 4).unwrap_or(u32::MAX);
        Ok(Some(kernels_slang::gather::embed::RowTable {
            blocks: bind(
                &cache.blocks,
                u64::from(cache.shape.blocks) * 4,
                1,
                Dtype::U32,
            )?,
            staged: bind(staged, u64::from(self.rows) * 4, 1, Dtype::U32)?,
            stage: bind(
                stage,
                u64::from(self.rows) * table.row_bytes,
                words,
                Dtype::U32,
            )?,
            hits: bind(
                if k == 0 {
                    &cache.hits
                } else {
                    &self.spare_hits
                },
                u64::from(cache.shape.blocks) * 4,
                1,
                Dtype::U32,
            )?,
            pool,
            block_rows: cache.shape.block_rows,
            unit_words: u32::try_from(cache.unit / 4).unwrap_or(u32::MAX),
            origin: u32::try_from(table.origin / 4).unwrap_or(u32::MAX),
            atom_words: u32::try_from(cache.shape.atom_bytes / 4).unwrap_or(u32::MAX),
            first,
        }))
    }
}
