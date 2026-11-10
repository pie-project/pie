//! The host and disk tiers of the shared pool (CONTRACT-BC 22): host chunks
//! of the device's chunk size in host memory, every state shape laid in them
//! exactly as on the device (`SlotShape::tier_offset`); and the slot file,
//! kv list slots only, slot `r` at `r · S`.
//!
//! A device slot moves out by one gather program whose result is downloaded,
//! and back in by one upload landed by a `dynamic_update_slice` a slot
//! (`Pools::read_slots` / `Pools::write_slots`). Both run in submission
//! order on the device's stream, so a copy reads what every earlier fire
//! wrote and lands before every later fire.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use crate::error::{Fault, Result};

/// The host tier: `chunks` chunks of `chunk_bytes`, zeroed lazily.
#[derive(Debug, Default)]
pub struct Host {
    bytes: Vec<u8>,
    chunks: u32,
    chunk_bytes: u64,
}

impl Host {
    #[must_use]
    pub fn new(chunks: u32, chunk_bytes: u64) -> Host {
        let len = usize::try_from(u64::from(chunks) * chunk_bytes).unwrap_or(0);
        Host {
            // calloc: pages are touched as copies first land in them.
            bytes: vec![0; len],
            chunks: if len == 0 { 0 } else { chunks },
            chunk_bytes,
        }
    }

    #[must_use]
    pub fn chunks(&self) -> u32 {
        self.chunks
    }

    fn span(&self, offset: u64, len: u64) -> Result<std::ops::Range<usize>> {
        let end = offset + len;
        if end > self.bytes.len() as u64 {
            return Err(Fault::Ceiling {
                what: "host tier bytes",
                need: end,
                have: self.bytes.len() as u64,
            });
        }
        Ok(offset as usize..end as usize)
    }

    pub fn read(&self, offset: u64, len: u64) -> Result<&[u8]> {
        Ok(&self.bytes[self.span(offset, len)?])
    }

    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        let span = self.span(offset, bytes.len() as u64)?;
        self.bytes[span].copy_from_slice(bytes);
        Ok(())
    }

    #[must_use]
    pub fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }
}

/// The slot file: kv list slots of `slot_bytes`, `slots` of them.
#[derive(Debug)]
pub struct Disk {
    file: File,
    path: PathBuf,
    slots: u32,
    slot_bytes: u64,
    persistent: bool,
    kept: bool,
}

/// The layout a kept slot file was written in.
fn meta(slot_bytes: u64, slots: u32) -> String {
    format!("pie-xla kv slots v1 slot_bytes={slot_bytes} slots={slots}\n")
}

impl Disk {
    /// The slot file under `dir`, `budget` bytes of whole slots. A
    /// persistent file whose meta names this layout is kept as it was.
    pub fn open(
        dir: &std::path::Path,
        budget: u64,
        slot_bytes: u64,
        persistent: bool,
    ) -> Result<Disk> {
        let slots = u32::try_from(budget / slot_bytes.max(1)).unwrap_or(u32::MAX);
        std::fs::create_dir_all(dir).map_err(|e| io(dir, &e))?;
        let path = dir.join("xla-kv.slots");
        let meta_path = dir.join("xla-kv.meta");
        let want = meta(slot_bytes, slots);
        let kept = persistent
            && std::fs::read_to_string(&meta_path).is_ok_and(|held| held == want)
            && std::fs::metadata(&path).is_ok_and(|m| m.len() == u64::from(slots) * slot_bytes);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(!kept)
            .open(&path)
            .map_err(|e| io(&path, &e))?;
        file.set_len(u64::from(slots) * slot_bytes)
            .map_err(|e| io(&path, &e))?;
        if persistent {
            std::fs::write(&meta_path, want).map_err(|e| io(&meta_path, &e))?;
        }
        Ok(Disk {
            file,
            path,
            slots,
            slot_bytes,
            persistent,
            kept,
        })
    }

    #[must_use]
    pub fn slots(&self) -> u32 {
        self.slots
    }

    #[must_use]
    pub fn kept(&self) -> bool {
        self.kept
    }

    #[must_use]
    pub fn persistent(&self) -> bool {
        self.persistent
    }

    fn at(&self, slot: u32) -> Result<u64> {
        if slot >= self.slots {
            return Err(Fault::Ceiling {
                what: "slot file slots",
                need: u64::from(slot) + 1,
                have: u64::from(self.slots),
            });
        }
        Ok(u64::from(slot) * self.slot_bytes)
    }

    pub fn read(&mut self, slot: u32) -> Result<Vec<u8>> {
        let at = self.at(slot)?;
        let mut out = vec![0u8; self.slot_bytes as usize];
        self.file
            .seek(SeekFrom::Start(at))
            .and_then(|_| self.file.read_exact(&mut out))
            .map_err(|e| io(&self.path, &e))?;
        Ok(out)
    }

    pub fn write(&mut self, slot: u32, bytes: &[u8]) -> Result<()> {
        let at = self.at(slot)?;
        self.file
            .seek(SeekFrom::Start(at))
            .and_then(|_| self.file.write_all(bytes))
            .map_err(|e| io(&self.path, &e))
    }
}

impl Drop for Disk {
    fn drop(&mut self) {
        if !self.persistent {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn io(path: &std::path::Path, e: &std::io::Error) -> Fault {
    Fault::Residency(format!("the slot file {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kept_slot_file_answers_what_it_held() {
        let dir = std::env::temp_dir().join(format!("pie-xla-disk-{}", std::process::id()));
        {
            let mut disk = Disk::open(&dir, 4 * 1024, 1024, true).expect("the file opens");
            assert!(!disk.kept());
            disk.write(2, &[7u8; 1024]).expect("a slot lands");
            assert!(disk.write(4, &[0u8; 1024]).is_err(), "past the file");
        }
        let mut disk = Disk::open(&dir, 4 * 1024, 1024, true).expect("the file reopens");
        assert!(disk.kept(), "same layout, kept");
        assert_eq!(disk.read(2).expect("the slot reads"), vec![7u8; 1024]);
        drop(disk);
        let disk = Disk::open(&dir, 8 * 1024, 1024, true).expect("a new layout");
        assert!(!disk.kept(), "another layout starts afresh");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn host_slots_lie_where_their_offset_says() {
        let mut host = Host::new(2, 4096);
        host.write(4096 + 512, &[3u8; 512]).expect("lands");
        assert_eq!(host.read(4096 + 512, 512).expect("reads"), &[3u8; 512][..]);
        assert!(host.write(8192 - 100, &[0u8; 512]).is_err());
    }
}
