//! The slot file an engine suspends kv pages into below host memory: one
//! per rank, preallocated, one kv page per fixed slot, written and read
//! around the page cache where the filesystem allows it.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;

/// Slot offsets and lengths are multiples of this, as direct I/O asks.
pub const ALIGN: usize = 4096;

/// The host bytes an engine stages a disk copy through when its kv does not
/// live in host-visible memory; larger copies go in turns of this.
pub const STAGE_BYTES: usize = 64 << 20;

#[derive(Debug)]
pub struct SlotFile {
    file: File,
    stride: usize,
    sums: Vec<u64>,
}

impl SlotFile {
    /// `dir/r<rank>.kv` with as many slots of `slot_bytes` as `budget`
    /// holds; `None` when that is none. The file is locked for the life of
    /// the load, so two servers never share one.
    pub fn open(dir: &Path, rank: u32, slot_bytes: u64, budget: u64) -> io::Result<Option<Self>> {
        let stride = usize::try_from(slot_bytes)
            .map_err(io::Error::other)?
            .next_multiple_of(ALIGN);
        let slots = usize::try_from(budget / stride as u64)
            .unwrap_or(usize::MAX)
            .min(u32::MAX as usize);
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
        let len = (slots * stride) as u64;
        file.set_len(len)?;
        if let Err(error) =
            rustix::fs::fallocate(&file, rustix::fs::FallocateFlags::empty(), 0, len)
            && error != rustix::io::Errno::OPNOTSUPP
        {
            return Err(error.into());
        }
        Ok(Some(Self {
            file,
            stride,
            sums: vec![0; slots],
        }))
    }

    #[must_use]
    pub fn slots(&self) -> u32 {
        self.sums.len() as u32
    }

    /// The bytes one slot takes in the file and in a staging buffer.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Writes `buf`, whole slots from `first` on, and keeps their checksums.
    pub fn write(&mut self, first: u32, buf: &[u8]) -> io::Result<()> {
        let range = self.span(first, buf)?;
        self.file
            .write_all_at(buf, first as u64 * self.stride as u64)?;
        for (sum, slot) in self.sums[range]
            .iter_mut()
            .zip(buf.chunks_exact(self.stride))
        {
            *sum = xxhash_rust::xxh3::xxh3_64(slot);
        }
        Ok(())
    }

    /// Reads whole slots from `first` on into `buf`, refusing any whose
    /// bytes are not the ones written.
    pub fn read(&self, first: u32, buf: &mut [u8]) -> io::Result<()> {
        let range = self.span(first, buf)?;
        self.file
            .read_exact_at(buf, first as u64 * self.stride as u64)?;
        for (at, slot) in range.zip(buf.chunks_exact(self.stride)) {
            if xxhash_rust::xxh3::xxh3_64(slot) != self.sums[at] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("kv slot {at} does not hold the bytes written to it"),
                ));
            }
        }
        Ok(())
    }

    fn span(&self, first: u32, buf: &[u8]) -> io::Result<std::ops::Range<usize>> {
        let count = buf.len() / self.stride;
        let range = first as usize..first as usize + count;
        if !buf.len().is_multiple_of(self.stride) || range.end > self.sums.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{} bytes at slot {first} are not whole slots within the file's {}",
                    buf.len(),
                    self.sums.len()
                ),
            ));
        }
        if buf.as_ptr().align_offset(ALIGN) != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a slot buffer must start on an ALIGN boundary",
            ));
        }
        Ok(range)
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

/// A heap buffer that starts on an `ALIGN` boundary.
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

/// Splits `ids` into runs of consecutive ids: `(index into ids, first id,
/// count)`.
#[must_use]
pub fn runs(ids: &[u32]) -> Vec<(usize, u32, usize)> {
    let mut runs: Vec<(usize, u32, usize)> = Vec::new();
    for (at, &id) in ids.iter().enumerate() {
        match runs.last_mut() {
            Some((_, first, count)) if *first as usize + *count == id as usize => *count += 1,
            _ => runs.push((at, id, 1)),
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slot_reads_back_what_was_written_and_refuses_torn_bytes() {
        let dir = std::env::temp_dir().join(format!("pie-slot-file-{}", std::process::id()));
        let mut file = SlotFile::open(&dir, 0, 5000, 4 * 8192).unwrap().unwrap();
        assert_eq!((file.slots(), file.stride()), (4, 8192));
        let mut buf = Aligned::zeroed(2 * 8192);
        buf.as_mut_slice()
            .iter_mut()
            .enumerate()
            .for_each(|(i, b)| *b = i as u8);
        file.write(1, buf.as_slice()).unwrap();
        assert!(SlotFile::open(&dir, 0, 5000, 4 * 8192).is_err());
        let mut back = Aligned::zeroed(2 * 8192);
        file.read(1, back.as_mut_slice()).unwrap();
        assert_eq!(back.as_slice(), buf.as_slice());
        let mut torn = Aligned::zeroed(ALIGN);
        torn.as_mut_slice().fill(0xff);
        file.file.write_all_at(torn.as_slice(), 8192).unwrap();
        assert!(file.read(1, back.as_mut_slice()).is_err());
        assert_eq!(
            runs(&[3, 4, 9, 10, 11, 2]),
            vec![(0, 3, 2), (2, 9, 3), (5, 2, 1)]
        );
        drop(file);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
