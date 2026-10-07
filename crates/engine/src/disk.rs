//! The slot file an engine suspends kv pages into below host memory: one
//! per rank, locked, one kv page per fixed slot, written and read around the
//! page cache where the filesystem allows it. Preallocated where blocks are
//! written in place; removed when the engine lets go of it.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// Slot offsets and lengths are multiples of this, as direct I/O asks.
pub const ALIGN: usize = 4096;

#[derive(Debug)]
pub struct SlotFile {
    file: File,
    path: PathBuf,
    stride: usize,
    slots: u32,
}

impl SlotFile {
    /// `dir/r<rank>.kv` with as many slots of `slot_bytes` as `budget`
    /// holds; `None` when that is none. The file is locked for the life of
    /// the load, so two servers never share one.
    pub fn open(dir: &Path, rank: u32, slot_bytes: u64, budget: u64) -> io::Result<Option<Self>> {
        let stride = usize::try_from(slot_bytes)
            .map_err(io::Error::other)?
            .next_multiple_of(ALIGN);
        let slots = u32::try_from(budget / stride as u64).unwrap_or(u32::MAX);
        if slot_bytes == 0 || slots == 0 {
            return Ok(None);
        }
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("r{rank}.kv"));
        let file = open_direct(&path)?;
        file.try_lock().map_err(|_| {
            io::Error::other(format!(
                "{} is in use by another server; give this one its own disk_kv_path",
                path.display()
            ))
        })?;
        let len = u64::from(slots) * stride as u64;
        file.set_len(len)?;
        // APFS copies on write: a page written lands in new blocks and the
        // preallocated ones stay reserved, so preallocating there doubles
        // the file's footprint instead of guaranteeing it.
        #[cfg(not(target_vendor = "apple"))]
        if let Err(error) =
            rustix::fs::fallocate(&file, rustix::fs::FallocateFlags::empty(), 0, len)
            && error != rustix::io::Errno::OPNOTSUPP
        {
            return Err(error.into());
        }
        Ok(Some(Self {
            file,
            path,
            stride,
            slots,
        }))
    }

    #[must_use]
    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// The bytes one slot takes in the file and in a page's stage.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Writes one slot's `stage`, an `ALIGN`-aligned buffer of `stride` bytes.
    pub fn write(&self, slot: u32, stage: &[u8]) -> io::Result<()> {
        self.file.write_all_at(stage, self.offset(slot, stage)?)
    }

    /// Reads slot `slot` into `stage`.
    pub fn read(&self, slot: u32, stage: &mut [u8]) -> io::Result<()> {
        self.file.read_exact_at(stage, self.offset(slot, stage)?)
    }

    fn offset(&self, slot: u32, stage: &[u8]) -> io::Result<u64> {
        if slot >= self.slots
            || stage.len() != self.stride
            || stage.as_ptr().align_offset(ALIGN) != 0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{} bytes at slot {slot} is not one aligned slot of the file's {} x {}",
                    stage.len(),
                    self.slots,
                    self.stride
                ),
            ));
        }
        Ok(u64::from(slot) * self.stride as u64)
    }
}

#[cfg(target_os = "linux")]
fn open_direct(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    let open = |flags| {
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .custom_flags(flags)
            .open(path)
    };
    // tmpfs and some overlays refuse O_DIRECT; the page cache then serves.
    match open(rustix::fs::OFlags::DIRECT.bits() as i32) {
        Err(error) if error.raw_os_error() == Some(rustix::io::Errno::INVAL.raw_os_error()) => {
            open(0)
        }
        opened => opened,
    }
}

#[cfg(not(target_os = "linux"))]
fn open_direct(path: &Path) -> io::Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    #[cfg(target_vendor = "apple")]
    rustix::fs::fcntl_nocache(&file, true)?;
    Ok(file)
}

/// What a slot file holds means nothing once the engine that wrote it lets
/// go, so it does not outlive the engine on disk.
impl Drop for SlotFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// A heap buffer of `len` bytes that starts on an `ALIGN` boundary: the
/// stage one page goes through on an engine whose kv lives in host memory.
#[derive(Debug, Default)]
pub struct Aligned {
    bytes: Vec<u8>,
    at: usize,
    len: usize,
}

impl Aligned {
    #[must_use]
    pub fn zeroed(len: usize) -> Self {
        let bytes = vec![0u8; len + ALIGN];
        let at = bytes.as_ptr().align_offset(ALIGN);
        Self { bytes, at, len }
    }

    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes[self.at..self.at + self.len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes[self.at..self.at + self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slot_reads_back_what_was_written_and_the_file_is_one_servers() {
        let dir = std::env::temp_dir().join(format!("pie-slot-file-{}", std::process::id()));
        let file = SlotFile::open(&dir, 0, 5000, 4 * 8192).unwrap().unwrap();
        assert_eq!((file.slots(), file.stride()), (4, 8192));
        let mut stage = Aligned::zeroed(8192);
        stage
            .as_mut_slice()
            .iter_mut()
            .enumerate()
            .for_each(|(i, b)| *b = i as u8);
        file.write(3, stage.as_slice()).unwrap();
        assert!(SlotFile::open(&dir, 0, 5000, 4 * 8192).is_err());
        let mut back = Aligned::zeroed(8192);
        file.read(3, back.as_mut_slice()).unwrap();
        assert_eq!(back.as_slice(), stage.as_slice());
        assert!(file.read(4, back.as_mut_slice()).is_err());
        drop(file);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rewriting_every_slot_keeps_the_file_within_its_budget_and_dropping_it_removes_it() {
        use std::os::unix::fs::MetadataExt;
        let dir = std::env::temp_dir().join(format!("pie-slot-budget-{}", std::process::id()));
        let path = dir.join("r0.kv");
        let budget = 64 * 8192;
        let file = SlotFile::open(&dir, 0, 8192, budget).unwrap().unwrap();
        let used = || std::fs::metadata(&path).unwrap().blocks() * 512;
        let stage = Aligned::zeroed(file.stride());
        for _ in 0..3 {
            for slot in 0..file.slots() {
                file.write(slot, stage.as_slice()).unwrap();
            }
            assert!(
                used() <= budget,
                "{} bytes on disk for a budget of {budget}",
                used()
            );
        }
        drop(file);
        assert!(!path.exists(), "the slot file outlived its engine");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
