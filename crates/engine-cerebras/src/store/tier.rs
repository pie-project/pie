//! The tiers below the host pool (CONTRACT-BC 22): host chunks a working
//! set's slots are parked in, and a slot file of kv list slots on disk.
//!
//! The pool itself is host memory here, so the host tier is a second region
//! of the same DRAM, apart from the pool's budget: it is seated only where a
//! host budget is stated. A host chunk is `chunk_bytes`, ids from 0, no
//! table; slot ref `r` of shape `s` lies at `s.tier_offset(chunk_bytes, r)`,
//! laid out word for word as the same slot in the pool, so a slot moves
//! whole. A host chunk is allocated the first time a slot in it is written
//! and reads zeros before. Disk ref `r` is slot `r` of the kv slot file.

use engine::chunk::{ShapeId, SlotCopy};
use engine::disk::{Aligned, SlotFile};
use engine::transfer::MemoryDomain;

use super::pool::{HostPool, LIST};
use crate::error::{Fault, Result};

/// The host chunks and the kv slot file.
#[derive(Debug, Default)]
pub struct Tiers {
    host: Vec<Option<Box<[u32]>>>,
    disk: Option<Disk>,
}

#[derive(Debug)]
struct Disk {
    file: SlotFile,
    stage: Aligned,
    /// The file is kept for the next run: every write is made durable.
    kept: bool,
}

fn io(what: &str) -> impl FnOnce(std::io::Error) -> Fault + '_ {
    move |error| Fault::Unbound {
        what: format!("the kv slot file ({what}): {error}"),
    }
}

/// Where one end of a copy lies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum End {
    Pool,
    Host,
    Disk,
}

impl HostPool {
    /// Seats `budget` bytes of host chunks; none where it holds no chunk.
    pub fn seat_host(&mut self, budget: u64) {
        let chunk = self.layout().chunk_words * 4;
        let chunks = budget.checked_div(chunk).unwrap_or(0);
        self.tiers.host = (0..chunks).map(|_| None).collect();
    }

    /// Opens `dir`'s kv slot file of `budget` bytes of kv list slots; `keep`
    /// names the load when the file is kept for the next run. A pool with
    /// no kv list, or a budget short of one slot, seats none.
    pub fn seat_disk(
        &mut self,
        dir: &std::path::Path,
        budget: u64,
        keep: Option<&str>,
    ) -> Result<()> {
        let Ok(list) = self.shape(LIST) else {
            return Ok(());
        };
        let bytes = list.slot_words * 4;
        let file = match keep {
            None => SlotFile::open(dir, "r0.kv", bytes, budget),
            Some(load) => {
                let sections: Vec<(u32, u8, u32)> = list
                    .sections
                    .iter()
                    .map(|s| (s.row, s.plane, s.width))
                    .collect();
                let layout = format!(
                    "cerebras {load} list tokens {} words {} sections {sections:?}",
                    list.tokens, list.slot_words
                );
                SlotFile::open_kept(dir, "r0.kv", bytes, budget, &layout)
            }
        }
        .map_err(io("open"))?;
        self.tiers.disk = file.map(|file| Disk {
            stage: Aligned::zeroed(file.stride()),
            file,
            kept: keep.is_some(),
        });
        Ok(())
    }

    /// The host chunks seated.
    #[must_use]
    pub fn host_chunks(&self) -> u32 {
        u32::try_from(self.tiers.host.len()).unwrap_or(u32::MAX)
    }

    /// The kv list's slots the slot file holds, in whole chunks of them.
    #[must_use]
    pub fn disk_chunks(&self) -> u32 {
        let per = self.shape(LIST).map_or(0, |list| list.per_chunk);
        self.tiers
            .disk
            .as_ref()
            .map_or(0, |disk| disk.file.slots())
            .checked_div(per)
            .unwrap_or(0)
    }

    /// The slot file is kept for the next run.
    #[must_use]
    pub fn disk_persistent(&self) -> bool {
        self.tiers.disk.as_ref().is_some_and(|disk| disk.kept)
    }

    /// The kept slot file holds what the last run left.
    #[must_use]
    pub fn disk_kept(&self) -> bool {
        self.tiers
            .disk
            .as_ref()
            .is_some_and(|disk| disk.file.kept())
    }

    /// Host ref `slot` of shape `id`: its chunk and the words it takes there.
    fn host_span(&self, id: ShapeId, slot: u32) -> Result<(usize, std::ops::Range<usize>)> {
        let shape = self.shape(id)?;
        let per = shape.per_chunk.max(1);
        let chunk = slot / per;
        if chunk as usize >= self.tiers.host.len() {
            return Err(Fault::Ceiling {
                what: "chunks of the host tier",
                need: u64::from(chunk) + 1,
                have: self.tiers.host.len() as u64,
            });
        }
        let first = usize::try_from(u64::from(slot % per) * shape.slot_words).unwrap_or(0);
        let words = usize::try_from(shape.slot_words).unwrap_or(0);
        Ok((chunk as usize, first..first + words))
    }

    fn disk_slot(&self, slot: u32) -> Result<()> {
        let per = self.shape(LIST)?.per_chunk;
        let slots = self.disk_chunks() * per;
        if slot >= slots {
            return Err(Fault::Ceiling {
                what: "kv list slots of the slot file",
                need: u64::from(slot) + 1,
                have: u64::from(slots),
            });
        }
        Ok(())
    }

    /// Slot `slot` of shape `id` at `end`, read whole; `None` on a dry pool.
    fn read_end(&mut self, end: End, id: ShapeId, slot: u32) -> Result<Option<Vec<u32>>> {
        match end {
            End::Pool => Ok(self.slot(id, slot)?.map(<[u32]>::to_vec)),
            End::Host => {
                let (chunk, range) = self.host_span(id, slot)?;
                Ok(Some(match &self.tiers.host[chunk] {
                    Some(words) => words[range].to_vec(),
                    None => vec![0; range.len()],
                }))
            }
            End::Disk => {
                self.disk_slot(slot)?;
                let words = self.shape(LIST)?.slot_words as usize;
                let disk = self
                    .tiers
                    .disk
                    .as_mut()
                    .expect("a slot file holds the slot");
                disk.file
                    .read(slot, disk.stage.as_mut_slice())
                    .map_err(io("read"))?;
                Ok(Some(
                    disk.stage.as_slice()[..words * 4]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| u32::from_le_bytes(*b))
                        .collect(),
                ))
            }
        }
    }

    /// Writes `words` over slot `slot` of shape `id` at `end`.
    fn write_end(&mut self, end: End, id: ShapeId, slot: u32, words: &[u32]) -> Result<()> {
        match end {
            End::Pool => {
                if let Some(to) = self.slot_mut(id, slot)? {
                    to.copy_from_slice(words);
                }
            }
            End::Host => {
                let (chunk, range) = self.host_span(id, slot)?;
                let len = usize::try_from(self.layout().chunk_words).unwrap_or(0);
                self.tiers.host[chunk].get_or_insert_with(|| vec![0; len].into_boxed_slice())
                    [range]
                    .copy_from_slice(words);
            }
            End::Disk => {
                self.disk_slot(slot)?;
                let disk = self
                    .tiers
                    .disk
                    .as_mut()
                    .expect("a slot file holds the slot");
                let stage = disk.stage.as_mut_slice();
                stage.fill(0);
                for (at, word) in words.iter().enumerate() {
                    stage[at * 4..at * 4 + 4].copy_from_slice(&word.to_le_bytes());
                }
                disk.file
                    .write(slot, disk.stage.as_slice())
                    .map_err(io("write"))?;
            }
        }
        Ok(())
    }

    /// Moves whole slots of one shape between the pool (`device`), the
    /// host chunks and the slot file (kv list only), `(src, dst)` in order;
    /// token runs only within the pool. Every source is read before its
    /// destination is written, pair by pair; a kept slot file is made
    /// durable before this returns.
    pub fn copy_tiers(&mut self, copy: &SlotCopy, device: MemoryDomain) -> Result<()> {
        let end = |domain: MemoryDomain| -> Result<End> {
            match domain {
                d if d == device => Ok(End::Pool),
                MemoryDomain::HostPinned => Ok(End::Host),
                MemoryDomain::LocalDisk => Ok(End::Disk),
                other => Err(Fault::Unbound {
                    what: format!("slots in {other:?}, which this load holds none of"),
                }),
            }
        };
        let (src, dst) = (end(copy.src)?, end(copy.dst)?);
        if (src, dst) == (End::Pool, End::Pool) {
            self.copy_slots(copy.shape, &copy.slots)?;
            for run in &copy.runs {
                self.copy_rows(
                    copy.shape,
                    (run.src, run.src_token),
                    (run.dst, run.dst_token),
                    run.tokens,
                )?;
            }
            return Ok(());
        }
        if !copy.runs.is_empty() {
            return Err(Fault::Unbound {
                what: format!(
                    "token runs from {:?} to {:?}: runs move only within the pool",
                    copy.src, copy.dst
                ),
            });
        }
        if (src == End::Disk || dst == End::Disk) && copy.shape != LIST {
            return Err(Fault::Unbound {
                what: format!(
                    "slots of shape {} to or from disk: only the kv list goes there",
                    copy.shape
                ),
            });
        }
        for &(from, to) in &copy.slots {
            match self.read_end(src, copy.shape, from)? {
                Some(words) => self.write_end(dst, copy.shape, to, &words)?,
                // A dry pool: the refs are checked, nothing moves.
                None if dst == End::Pool => {
                    self.span(copy.shape, to)?;
                }
                None => {}
            }
        }
        if dst == End::Disk
            && let Some(disk) = &self.tiers.disk
            && disk.kept
        {
            disk.file.sync().map_err(io("sync"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
