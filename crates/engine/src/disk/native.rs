//! Slot files on the host's filesystem.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

use super::{Room, SlotFile, geometry};

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

    /// The file's length set, sparse: its budget is a ceiling, its blocks
    /// written as slots are, never reserved up front (a tier derived to all
    /// the free disk left none for logs or other processes). A write that
    /// finds the disk full fails that copy by name.
    fn reserve(file: &File, slots: u32, stride: usize) -> io::Result<()> {
        file.set_len(u64::from(slots) * stride as u64)
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

/// The room under `dir` (its nearest existing ancestor): the free space
/// and the platform's own reserve within it ([`Room`]). Nothing when the
/// filesystem does not say.
#[must_use]
pub fn room(dir: &Path) -> Room {
    let Some(at) = dir.ancestors().find(|path| path.exists()) else {
        return Room::unstated(&format!("no part of {} exists", dir.display()));
    };
    read_room(at).unwrap_or_else(|why| Room::unstated(&why))
}

#[cfg(all(unix, not(target_os = "android")))]
fn read_room(at: &Path) -> Result<Room, String> {
    let stat = rustix::fs::statvfs(at).map_err(|error| format!("statvfs: {error}"))?;
    Ok(Room::statvfs(
        stat.f_bfree.saturating_mul(stat.f_frsize),
        stat.f_bavail.saturating_mul(stat.f_frsize),
    ))
}

/// The global settings as the settings provider keeps them; readable only
/// to a process the platform lets read them (a text file before Android
/// 12, binary ABX since), else AOSP's defaults stand.
#[cfg(target_os = "android")]
const ANDROID_GLOBAL_SETTINGS: &str = "/data/system/users/0/settings_global.xml";

#[cfg(target_os = "android")]
fn read_room(at: &Path) -> Result<Room, String> {
    let stat = rustix::fs::statvfs(at).map_err(|error| format!("statvfs: {error}"))?;
    let settings = std::fs::read_to_string(ANDROID_GLOBAL_SETTINGS).ok();
    Ok(Room::android(
        stat.f_blocks.saturating_mul(stat.f_frsize),
        stat.f_bfree.saturating_mul(stat.f_frsize),
        stat.f_bavail.saturating_mul(stat.f_frsize),
        &super::LowStorage::from_settings(settings.as_deref()),
    ))
}

#[cfg(windows)]
#[allow(unsafe_code)]
fn read_room(at: &Path) -> Result<Room, String> {
    use std::os::windows::ffi::OsStrExt;

    unsafe extern "system" {
        fn GetDiskFreeSpaceExW(
            directory: *const u16,
            available_to_caller: *mut u64,
            total: *mut u64,
            total_free: *mut u64,
        ) -> i32;
    }
    let wide: Vec<u16> = at.as_os_str().encode_wide().chain([0]).collect();
    let (mut available, mut total, mut free) = (0u64, 0u64, 0u64);
    // SAFETY: `wide` is a NUL-terminated UTF-16 path that outlives the call;
    // the three out-parameters are plain integers owned here.
    let ok = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &raw mut available,
            &raw mut total,
            &raw mut free,
        )
    };
    if ok == 0 {
        return Err(format!(
            "GetDiskFreeSpaceExW: {}",
            io::Error::last_os_error()
        ));
    }
    Ok(Room::windows(free, available))
}

#[cfg(not(any(unix, windows)))]
fn read_room(_at: &Path) -> Result<Room, String> {
    Err("this platform states no free space".to_string())
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

    /// The reader names a reserve the platform states and never offers
    /// more than is free; a path not yet made reads its nearest ancestor.
    #[test]
    fn the_room_under_a_path_is_read_from_its_nearest_existing_ancestor() {
        let room = room(&std::env::temp_dir().join("pie-room-not-made/kv"));
        assert!(room.free > 0, "{room}");
        assert!(room.for_us() <= room.free, "{room}");
        assert!(!room.reserve_is.starts_with("nothing"), "{room}");
    }

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
    fn a_span_reads_back_what_was_written_and_stays_aligned_within_the_slots() {
        let dir = std::env::temp_dir().join(format!("pie-slot-span-{}", std::process::id()));
        let file = SlotFile::open(&dir, "r0.state", 3 * 4096, 2 * 3 * 4096)
            .unwrap()
            .unwrap();
        let mut stage = Aligned::zeroed(2 * 4096);
        stage
            .as_mut_slice()
            .iter_mut()
            .enumerate()
            .for_each(|(i, b)| *b = (i * 7) as u8);
        // Across the boundary of slots 0 and 1.
        file.write_span(2 * 4096, stage.as_slice()).unwrap();
        let mut back = Aligned::zeroed(2 * 4096);
        file.read_span(2 * 4096, back.as_mut_slice()).unwrap();
        assert_eq!(back.as_slice(), stage.as_slice());
        // Unaligned, short, or past the slots: refused.
        assert!(file.read_span(100, back.as_mut_slice()).is_err());
        assert!(file.read_span(0, &mut back.as_mut_slice()[..100]).is_err());
        assert!(file.read_span(5 * 4096, back.as_mut_slice()).is_err());
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
