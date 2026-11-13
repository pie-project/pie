//! Slot files on the host's filesystem.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

use super::{SlotFile, geometry};

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
        let Some((stride, slots)) = geometry(slot_bytes, budget)? else {
            return Ok(None);
        };
        std::fs::create_dir_all(dir)?;
        let path = dir.join(name);
        let file = open_direct(&path)?;
        file.try_lock().map_err(|_| {
            io::Error::other(format!(
                "{} is in use by another server; give this one its own `[engine.memory] disk_path`",
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

#[cfg(test)]
mod tests {
    use super::super::{Aligned, SlotFile};
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
