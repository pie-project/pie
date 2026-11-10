//! A fire's view of the host pool: the slots its lanes name, each under a
//! fire-local id in the planes the fire's program reads and writes.
//!
//! The planes are the fire's activation (sized at load from its lanes, not
//! from the pool): a kv plane holds `kv_pages` local pages and the sink
//! page past them, a windowed plane the null page 0, `window_pages - 1`
//! local pages and its sink, a state slab `records` local records and the
//! sink record, an rs plane `rs_pages` local pages. A slot ref is staged
//! once however many lanes name it, so lanes sharing a prefix read one copy
//! and a fire reads and writes exactly what it would in place.

use std::borrow::Cow;
use std::collections::HashMap;

use engine::chunk::ShapeId;

use super::pool::{FIXED, HostPool, LIST, RS, WINDOW};
use crate::error::{Fault, Result};

/// How many of each a fire stages at most.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Capacity {
    pub kv_pages: u32,
    /// The null page 0 included.
    pub window_pages: u32,
    pub records: u32,
    pub rs_pages: u32,
}

/// Slot refs of one list shape under local ids from `first`; a scratch id
/// stands for no slot (a kv-less lane's own rows), zeroed and never kept.
#[derive(Debug, Clone)]
pub struct Local {
    staged: Vec<Option<u32>>,
    of: HashMap<u32, u32>,
    first: u32,
    cap: u32,
    what: &'static str,
}

impl Local {
    fn new(first: u32, cap: u32, what: &'static str) -> Local {
        Local {
            staged: Vec::new(),
            of: HashMap::new(),
            first,
            cap,
            what,
        }
    }

    fn next(&mut self, slot: Option<u32>) -> Result<u32> {
        let local = self.first + u32::try_from(self.staged.len()).unwrap_or(u32::MAX);
        if local >= self.cap {
            return Err(Fault::Ceiling {
                what: self.what,
                need: u64::from(local) + 1,
                have: u64::from(self.cap),
            });
        }
        self.staged.push(slot);
        Ok(local)
    }

    /// The local id of slot ref `slot`, staged on first sight.
    pub fn of(&mut self, slot: u32) -> Result<u32> {
        if let Some(&local) = self.of.get(&slot) {
            return Ok(local);
        }
        let local = self.next(Some(slot))?;
        self.of.insert(slot, local);
        Ok(local)
    }

    /// A local id of its own, backed by no slot.
    pub fn scratch(&mut self) -> Result<u32> {
        self.next(None)
    }

    /// Each local id staged and the slot ref it holds.
    pub fn staged(&self) -> impl Iterator<Item = (u32, Option<u32>)> + '_ {
        (self.first..).zip(self.staged.iter().copied())
    }
}

/// One staged record: its pieces (none for scratch) and whether the fire
/// writes it back.
#[derive(Debug, Clone)]
struct Record {
    pieces: Option<Vec<u32>>,
    keeps: bool,
}

#[derive(Debug, Clone)]
pub struct Records {
    staged: Vec<Record>,
    of: HashMap<Vec<u32>, u32>,
    cap: u32,
}

impl Records {
    fn next(&mut self, record: Record) -> Result<u32> {
        let local = u32::try_from(self.staged.len()).unwrap_or(u32::MAX);
        if local >= self.cap {
            return Err(Fault::Ceiling {
                what: "state records one fire stages",
                need: u64::from(local) + 1,
                have: u64::from(self.cap),
            });
        }
        self.staged.push(record);
        Ok(local)
    }

    /// The local id of the record whose pieces are `pieces`; `keeps` says
    /// the fire's state is written back to it (a record a lane only reads,
    /// a group's `run` beside the `cow` it folds into, is not).
    pub fn of(&mut self, pieces: &[u32], keeps: bool) -> Result<u32> {
        if let Some(&local) = self.of.get(pieces) {
            self.staged[local as usize].keeps |= keeps;
            return Ok(local);
        }
        let local = self.next(Record {
            pieces: Some(pieces.to_vec()),
            keeps,
        })?;
        self.of.insert(pieces.to_vec(), local);
        Ok(local)
    }

    /// A zeroed record of its own (a lane that carries no record).
    pub fn scratch(&mut self) -> Result<u32> {
        self.next(Record {
            pieces: None,
            keeps: false,
        })
    }
}

/// Every slot one fire stages.
#[derive(Debug, Clone)]
pub struct Stage {
    pub kv: Local,
    /// Local 0 is the null page.
    pub window: Local,
    pub records: Records,
    pub rs: Local,
}

impl Stage {
    #[must_use]
    pub fn new(capacity: Capacity) -> Stage {
        Stage {
            kv: Local::new(0, capacity.kv_pages, "kv pages one fire stages"),
            window: Local::new(1, capacity.window_pages, "windowed pages one fire stages"),
            records: Records {
                staged: Vec::new(),
                of: HashMap::new(),
                cap: capacity.records,
            },
            rs: Local::new(0, capacity.rs_pages, "rs buffer pages one fire stages"),
        }
    }
}

/// A staged unit's local id and words, `None` for zeros.
pub(super) type Source<'p> = (u32, Option<Cow<'p, [u32]>>);

/// Each staged unit of shape `id` (a list slot, or a record reassembled
/// from its pieces).
pub(super) fn sources<'p>(
    pool: &'p HostPool,
    stage: &Stage,
    id: ShapeId,
) -> Result<Vec<Source<'p>>> {
    match id {
        FIXED => {
            let shape = pool.shape(FIXED)?;
            let words = usize::try_from(shape.record_words).unwrap_or(usize::MAX);
            (0u32..)
                .zip(&stage.records.staged)
                .map(|(local, record)| {
                    let Some(pieces) = &record.pieces else {
                        return Ok((local, None));
                    };
                    let mut out = Vec::with_capacity(words);
                    for &piece in pieces {
                        if let Some(slot) = pool.slot(FIXED, piece)? {
                            out.extend_from_slice(slot);
                        }
                    }
                    if out.is_empty() {
                        return Ok((local, None));
                    }
                    out.truncate(words);
                    Ok((local, Some(Cow::Owned(out))))
                })
                .collect()
        }
        _ => {
            let local = list_of(stage, id);
            let mut out: Vec<Source<'p>> = Vec::new();
            if id == WINDOW {
                // The null page reads zeros, every fire.
                out.push((0, None));
            }
            for (at, slot) in local.staged() {
                let words = match slot {
                    Some(slot) => pool.slot(id, slot)?.map(Cow::Borrowed),
                    None => None,
                };
                out.push((at, words));
            }
            Ok(out)
        }
    }
}

/// The local ids of list shape `id`.
pub(super) fn list_of(stage: &Stage, id: ShapeId) -> &Local {
    match id {
        WINDOW => &stage.window,
        RS => &stage.rs,
        _ => {
            debug_assert_eq!(id, LIST);
            &stage.kv
        }
    }
}

/// The staged units of shape `id` the fire writes back: `(local, slot
/// refs)`, one ref a list slot, a record's pieces.
pub(super) fn kept(stage: &Stage, id: ShapeId) -> Vec<(u32, Vec<u32>)> {
    match id {
        FIXED => (0u32..)
            .zip(&stage.records.staged)
            .filter(|(_, record)| record.keeps)
            .filter_map(|(local, record)| Some((local, record.pieces.clone()?)))
            .collect(),
        _ => list_of(stage, id)
            .staged()
            .filter_map(|(local, slot)| Some((local, vec![slot?])))
            .collect(),
    }
}

#[cfg(test)]
mod tests;
