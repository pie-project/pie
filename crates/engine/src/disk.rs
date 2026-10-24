//! Slot files below host memory, one per rank and kind: unlinked once open so
//! they never outlive the process, or kept, for a persistent tier.

use std::fs::{File, OpenOptions};
use std::io;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::Path;

/// Slot offsets and lengths are multiples of this, as direct I/O asks.
pub const ALIGN: usize = 4096;

#[derive(Debug)]
pub struct SlotFile {
    file: File,
    stride: usize,
    slots: u32,
    kept: bool,
}

impl SlotFile {
    /// `dir/<name>` (`r<rank>.kv`, `r<rank>.rs`) with as many slots of
    /// `slot_bytes` as `budget` holds; `None` when that is none. The path is unlinked once the file
    /// is open and locked: what it holds means nothing to the next run, and
    /// the space comes back when the process exits, however it exits. The
    /// lock covers the moment before the unlink, so two servers never share
    /// one file.
    pub fn open(dir: &Path, name: &str, slot_bytes: u64, budget: u64) -> io::Result<Option<Self>> {
        let Some((file, stride, slots)) = Self::locked(dir, name, slot_bytes, budget)? else {
            return Ok(None);
        };
        std::fs::remove_file(dir.join(name))?;
        Self::reserve(&file, slots, stride)?;
        Ok(Some(Self {
            file,
            stride,
            slots,
            kept: false,
        }))
    }

    /// [`Self::open`], but the path stays for the next run: `dir/<name>`
    /// and beside it `<name>.meta`, which names the file's slots and the
    /// `layout` its engine wrote them in. When the meta found matches,
    /// [`Self::kept`] is true and the slots hold what the last run left
    /// there; otherwise the file is taken afresh. Which slots mean anything
    /// is the caller's to remember.
    pub fn open_kept(
        dir: &Path,
        name: &str,
        slot_bytes: u64,
        budget: u64,
        layout: &str,
    ) -> io::Result<Option<Self>> {
        let Some((file, stride, slots)) = Self::locked(dir, name, slot_bytes, budget)? else {
            return Ok(None);
        };
        let meta_path = dir.join(format!("{name}.meta"));
        let meta = format!("{layout}\n{stride}\n{slots}\n");
        let len = u64::from(slots) * stride as u64;
        let kept = std::fs::read_to_string(&meta_path).is_ok_and(|found| found == meta)
            && file.metadata()?.len() == len;
        if !kept {
            // No meta while the file is taken afresh: a run that dies here
            // leaves nothing that reads as kept.
            match std::fs::remove_file(&meta_path) {
                Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error),
                _ => {}
            }
            file.set_len(0)?;
            Self::reserve(&file, slots, stride)?;
            file.sync_all()?;
            write_synced(&meta_path, meta.as_bytes())?;
        }
        Ok(Some(Self {
            file,
            stride,
            slots,
            kept,
        }))
    }

    /// The file at `dir/<name>`, open and locked, with the slots `budget`
    /// holds; `None` when that is none.
    fn locked(
        dir: &Path,
        name: &str,
        slot_bytes: u64,
        budget: u64,
    ) -> io::Result<Option<(File, usize, u32)>> {
        let stride = usize::try_from(slot_bytes)
            .map_err(io::Error::other)?
            .next_multiple_of(ALIGN);
        let slots =
            u32::try_from(budget.checked_div(stride as u64).unwrap_or(0)).unwrap_or(u32::MAX);
        if slot_bytes == 0 || slots == 0 {
            return Ok(None);
        }
        std::fs::create_dir_all(dir)?;
        let path = dir.join(name);
        let file = open_direct(&path)?;
        file.try_lock().map_err(|_| {
            io::Error::other(format!(
                "{} is in use by another server; give this one its own disk_kv_dir",
                path.display()
            ))
        })?;
        Ok(Some((file, stride, slots)))
    }

    fn reserve(file: &File, slots: u32, stride: usize) -> io::Result<()> {
        let len = u64::from(slots) * stride as u64;
        file.set_len(len)?;
        // APFS copies on write: a page written lands in new blocks and the
        // preallocated ones stay reserved, so preallocating there doubles
        // the file's footprint instead of guaranteeing it.
        #[cfg(all(unix, not(target_vendor = "apple")))]
        if let Err(error) = rustix::fs::fallocate(file, rustix::fs::FallocateFlags::empty(), 0, len)
            && error != rustix::io::Errno::OPNOTSUPP
        {
            return Err(error.into());
        }
        Ok(())
    }

    #[must_use]
    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// Whether the slots hold what the last run left: a kept file whose
    /// meta matched.
    #[must_use]
    pub fn kept(&self) -> bool {
        self.kept
    }

    /// Makes every slot written so far durable.
    pub fn sync(&self) -> io::Result<()> {
        self.file.sync_data()
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

/// The bytes a kept slot file `dir/<name>` already takes: room a
/// persistent tier counts as its own when it sizes itself again.
#[must_use]
pub fn kept_bytes(dir: &Path, name: &str) -> u64 {
    std::fs::metadata(dir.join(name)).map_or(0, |meta| meta.len())
}

fn write_synced(path: &Path, bytes: &[u8]) -> io::Result<()> {
    use std::io::Write;
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Zero when the filesystem does not say.
#[must_use]
pub fn free_space(dir: &Path) -> u64 {
    dir.ancestors()
        .find(|path| path.exists())
        .and_then(available_bytes)
        .unwrap_or(0)
}

#[cfg(unix)]
fn available_bytes(at: &Path) -> Option<u64> {
    let stat = rustix::fs::statvfs(at).ok()?;
    Some(stat.f_bavail.saturating_mul(stat.f_frsize))
}

/// Windows states free space only through an unsafe call this crate
/// forbids.
#[cfg(not(unix))]
fn available_bytes(_at: &Path) -> Option<u64> {
    None
}

#[must_use]
pub fn unstated_budget(
    dir: &Path,
    rows: u64,
    row_bytes: u64,
    pages: u64,
    seated: u64,
    page_bytes: u64,
) -> u64 {
    let slot = |bytes: u64| bytes.next_multiple_of(ALIGN as u64);
    let wanted = rows
        .saturating_mul(slot(row_bytes))
        .saturating_add(pages.max(seated).saturating_mul(slot(page_bytes)));
    wanted.min(free_space(dir) / 4)
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

/// Windows reads and writes at an offset through `seek_read`/`seek_write`,
/// which may stop short: the whole slot is looped through, as Unix's
/// `read_exact_at`/`write_all_at` do.
#[cfg(windows)]
trait FileExt {
    fn write_all_at(&self, buf: &[u8], at: u64) -> io::Result<()>;
    fn read_exact_at(&self, buf: &mut [u8], at: u64) -> io::Result<()>;
}

#[cfg(windows)]
impl FileExt for File {
    fn write_all_at(&self, mut buf: &[u8], mut at: u64) -> io::Result<()> {
        use std::os::windows::fs::FileExt as _;
        while !buf.is_empty() {
            match self.seek_write(buf, at)? {
                0 => return Err(io::ErrorKind::WriteZero.into()),
                n => (buf, at) = (&buf[n..], at + n as u64),
            }
        }
        Ok(())
    }

    fn read_exact_at(&self, mut buf: &mut [u8], mut at: u64) -> io::Result<()> {
        use std::os::windows::fs::FileExt as _;
        while !buf.is_empty() {
            match self.seek_read(buf, at)? {
                0 => return Err(io::ErrorKind::UnexpectedEof.into()),
                n => (buf, at) = (&mut buf[n..], at + n as u64),
            }
        }
        Ok(())
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
    fn a_slot_reads_back_what_was_written_and_each_server_has_its_own_file() {
        let dir = std::env::temp_dir().join(format!("pie-slot-file-{}", std::process::id()));
        let file = SlotFile::open(&dir, "r0.kv", 5000, 4 * 8192)
            .unwrap()
            .unwrap();
        assert_eq!((file.slots(), file.stride()), (4, 8192));
        let mut stage = Aligned::zeroed(8192);
        stage
            .as_mut_slice()
            .iter_mut()
            .enumerate()
            .for_each(|(i, b)| *b = i as u8);
        file.write(3, stage.as_slice()).unwrap();
        let other = SlotFile::open(&dir, "r0.kv", 5000, 4 * 8192)
            .unwrap()
            .unwrap();
        other.write(3, Aligned::zeroed(8192).as_slice()).unwrap();
        let mut back = Aligned::zeroed(8192);
        file.read(3, back.as_mut_slice()).unwrap();
        assert_eq!(back.as_slice(), stage.as_slice());
        assert!(file.read(4, back.as_mut_slice()).is_err());
        drop(file);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_kept_file_holds_its_slots_across_opens_until_its_layout_or_size_changes() {
        let dir = std::env::temp_dir().join(format!("pie-slot-kept-{}", std::process::id()));
        let mut stage = Aligned::zeroed(8192);
        stage.as_mut_slice().fill(7);
        {
            let file = SlotFile::open_kept(&dir, "r0.kv", 8192, 4 * 8192, "a")
                .unwrap()
                .unwrap();
            assert!(!file.kept(), "a new file holds nothing");
            file.write(2, stage.as_slice()).unwrap();
            file.sync().unwrap();
            assert!(
                SlotFile::open_kept(&dir, "r0.kv", 8192, 4 * 8192, "a").is_err(),
                "two servers share one file"
            );
        }
        assert_eq!(kept_bytes(&dir, "r0.kv"), 4 * 8192);
        {
            let file = SlotFile::open_kept(&dir, "r0.kv", 8192, 4 * 8192, "a")
                .unwrap()
                .unwrap();
            assert!(file.kept());
            let mut back = Aligned::zeroed(8192);
            file.read(2, back.as_mut_slice()).unwrap();
            assert_eq!(back.as_slice(), stage.as_slice());
        }
        let other_layout = SlotFile::open_kept(&dir, "r0.kv", 8192, 4 * 8192, "b")
            .unwrap()
            .unwrap();
        assert!(
            !other_layout.kept(),
            "another layout's slots read as this one's"
        );
        drop(other_layout);
        let other_size = SlotFile::open_kept(&dir, "r0.kv", 8192, 8 * 8192, "b")
            .unwrap()
            .unwrap();
        assert!(!other_size.kept(), "slots of another count read as kept");
        drop(other_size);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rewriting_every_slot_keeps_the_file_within_its_budget_and_it_leaves_no_path() {
        use std::os::unix::fs::MetadataExt;
        let dir = std::env::temp_dir().join(format!("pie-slot-budget-{}", std::process::id()));
        let path = dir.join("r0.kv");
        let budget = 64 * 8192;
        let file = SlotFile::open(&dir, "r0.kv", 8192, budget)
            .unwrap()
            .unwrap();
        assert!(!path.exists(), "the slot file outlives its process");
        let used = || file.file.metadata().unwrap().blocks() * 512;
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
        std::fs::remove_dir_all(dir).unwrap();
    }
}
