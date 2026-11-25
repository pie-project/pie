//! Slot files on the host's filesystem.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};

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
        Self::reserve(&file, dir, name, slots, stride)?;
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
            && file.metadata()?.len() == len
            && !laid_whole_for_nothing(&file, dir)?;
        if !kept {
            // No meta while the file is taken afresh: a run that dies here
            // leaves nothing that reads as kept.
            match std::fs::remove_file(&meta_path) {
                Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error),
                _ => {}
            }
            Self::reserve(&file, dir, name, slots, stride)?;
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
    /// finds the disk full fails that copy by name. A filesystem that keeps
    /// no holes lays it whole, so there it must fit what is available.
    fn reserve(file: &File, dir: &Path, name: &str, slots: u32, stride: usize) -> io::Result<()> {
        let bytes = u64::from(slots) * stride as u64;
        // What an older run left in it goes first.
        empty(file)?;
        if keeps_holes(dir)? {
            mark_sparse(file)?;
        } else {
            let available = read_room(dir).map_err(io::Error::other)?.for_us();
            fits_whole(&dir.join(name), bytes, available)?;
        }
        file.set_len(bytes)
    }
}

/// Whether a slot file of `bytes`, laid whole, fits the `available` bytes;
/// refused by name when not.
fn fits_whole(path: &Path, bytes: u64, available: u64) -> io::Result<()> {
    if bytes <= available {
        return Ok(());
    }
    // Not `StorageFull`: no copy found the tier full, the load states more
    // than the volume has.
    Err(io::Error::other(format!(
        "{}: the filesystem keeps no holes, so the slot file is laid whole: {bytes} bytes asked, \
         {available} available; state a smaller `[engine.memory] budget_disk` or another \
         `disk_path`",
        path.display()
    )))
}

/// The length the probe file is set to: past any filesystem's allocation
/// unit, so one that keeps holes takes less of it.
const PROBE_BYTES: u64 = 4 << 20;

/// Whether the filesystem under `dir` keeps holes: a probe file, set long
/// and never written, takes less than its length.
fn keeps_holes(dir: &Path) -> io::Result<bool> {
    static PROBES: AtomicU32 = AtomicU32::new(0);
    let path = dir.join(format!(
        ".pie-holes-{}-{}",
        std::process::id(),
        PROBES.fetch_add(1, Ordering::Relaxed)
    ));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)?;
    std::fs::remove_file(&path)?;
    if mark_sparse(&file).is_err() {
        return Ok(false);
    }
    match file.set_len(PROBE_BYTES) {
        // Laid whole, and not even that fits.
        Err(error) if error.kind() == io::ErrorKind::StorageFull => Ok(false),
        set => {
            set?;
            Ok(allocated(&file)? < PROBE_BYTES)
        }
    }
}

/// Gives back every byte `file` takes.
#[cfg(not(windows))]
fn empty(file: &File) -> io::Result<()> {
    file.set_len(0)
}

/// Gives back every byte `file` takes: NTFS keeps what a shorter length
/// leaves allocated until the file closes, so the allocation is set too.
#[cfg(windows)]
#[allow(unsafe_code)]
fn empty(file: &File) -> io::Result<()> {
    use std::os::windows::io::{AsRawHandle, RawHandle};

    const FILE_ALLOCATION_INFO: i32 = 5;
    unsafe extern "system" {
        fn SetFileInformationByHandle(
            file: RawHandle,
            class: i32,
            info: *const i64,
            bytes: u32,
        ) -> i32;
    }
    file.set_len(0)?;
    let allocation = 0i64;
    // SAFETY: the handle is open for the call, and `FILE_ALLOCATION_INFO`
    // is one 64-bit integer, owned here.
    let ok = unsafe {
        SetFileInformationByHandle(
            file.as_raw_handle(),
            FILE_ALLOCATION_INFO,
            &raw const allocation,
            size_of::<i64>() as u32,
        )
    };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Makes unwritten ranges of `file` holes: NTFS allocates what a length
/// covers unless the file is sparse; other platforms' filesystems decide
/// alone.
#[cfg(not(windows))]
#[allow(clippy::unnecessary_wraps)]
fn mark_sparse(_file: &File) -> io::Result<()> {
    Ok(())
}

#[cfg(windows)]
#[allow(unsafe_code)]
fn mark_sparse(file: &File) -> io::Result<()> {
    use std::os::windows::io::{AsRawHandle, RawHandle};

    const FSCTL_SET_SPARSE: u32 = 0x0009_00C4;
    unsafe extern "system" {
        fn DeviceIoControl(
            device: RawHandle,
            code: u32,
            input: *const u8,
            input_bytes: u32,
            output: *mut u8,
            output_bytes: u32,
            returned: *mut u32,
            overlapped: *mut u8,
        ) -> i32;
    }
    let mut returned = 0u32;
    // SAFETY: the handle is open for the call and not overlapped; no input
    // sets the file sparse, and `returned` is a plain integer owned here.
    let ok = unsafe {
        DeviceIoControl(
            file.as_raw_handle(),
            FSCTL_SET_SPARSE,
            std::ptr::null(),
            0,
            std::ptr::null_mut(),
            0,
            &raw mut returned,
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// The bytes `file` takes on its disk.
#[cfg(unix)]
fn allocated(file: &File) -> io::Result<u64> {
    use std::os::unix::fs::MetadataExt;
    Ok(file.metadata()?.blocks() * 512)
}

#[cfg(windows)]
#[allow(unsafe_code)]
fn allocated(file: &File) -> io::Result<u64> {
    use std::os::windows::io::{AsRawHandle, RawHandle};

    /// `FILE_STANDARD_INFO`.
    #[repr(C)]
    #[derive(Default)]
    struct Standard {
        allocation: i64,
        end: i64,
        links: u32,
        delete_pending: u8,
        directory: u8,
    }
    const FILE_STANDARD_INFO: i32 = 1;
    unsafe extern "system" {
        fn GetFileInformationByHandleEx(
            file: RawHandle,
            class: i32,
            info: *mut Standard,
            bytes: u32,
        ) -> i32;
    }
    let mut info = Standard::default();
    // SAFETY: the handle is open for the call, and `info` is the class's
    // struct with its own size.
    let ok = unsafe {
        GetFileInformationByHandleEx(
            file.as_raw_handle(),
            FILE_STANDARD_INFO,
            &raw mut info,
            size_of::<Standard>() as u32,
        )
    };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(u64::try_from(info.allocation).unwrap_or(0))
}

#[cfg(not(any(unix, windows)))]
fn allocated(_file: &File) -> io::Result<u64> {
    Err(io::Error::other(
        "this platform states no allocated size of a file",
    ))
}

/// Whether a kept `file` was laid whole on a filesystem that keeps holes:
/// marking it sparse frees nothing it already takes, so it is taken afresh.
#[cfg(windows)]
fn laid_whole_for_nothing(file: &File, dir: &Path) -> io::Result<bool> {
    use std::os::windows::fs::MetadataExt;

    const FILE_ATTRIBUTE_SPARSE_FILE: u32 = 0x200;
    Ok(file.metadata()?.file_attributes() & FILE_ATTRIBUTE_SPARSE_FILE == 0 && keeps_holes(dir)?)
}

#[cfg(not(windows))]
#[allow(clippy::unnecessary_wraps)]
fn laid_whole_for_nothing(_file: &File, _dir: &Path) -> io::Result<bool> {
    Ok(false)
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
/// filesystem does not say, or keeps no holes ([`Room::laid_whole`]).
#[must_use]
pub fn room(dir: &Path) -> Room {
    let Some(at) = dir.ancestors().find(|path| path.exists()) else {
        return Room::unstated(&format!("no part of {} exists", dir.display()));
    };
    let room = match read_room(at) {
        Ok(room) => room,
        Err(why) => return Room::unstated(&why),
    };
    match keeps_holes(at) {
        Ok(true) => room,
        Ok(false) => Room::laid_whole(room.free),
        Err(error) => Room::unstated(&format!("probing {} for holes: {error}", at.display())),
    }
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

    /// A slot file's budget is a ceiling: nothing of it is taken as it is
    /// opened, and one slot written takes that slot. The probe leaves no
    /// file behind.
    #[test]
    fn a_slot_file_takes_only_what_is_written() {
        let dir = std::env::temp_dir().join(format!("pie-slot-holes-{}", std::process::id()));
        let slot = 1 << 20;
        for file in [
            SlotFile::open(&dir, "r0.kv", slot, 64 * slot),
            SlotFile::open_kept(&dir, "r1.kv", slot, 64 * slot, "a"),
        ] {
            let file = file.unwrap().unwrap();
            let opened = allocated(&file.file).unwrap();
            assert_eq!(opened, 0, "bytes taken as the file is opened");
            let mut stage = Aligned::zeroed(file.stride());
            stage.as_mut_slice().fill(7);
            file.write(40, stage.as_slice()).unwrap();
            file.file.sync_all().unwrap();
            let written = allocated(&file.file).unwrap();
            assert!((slot..2 * slot).contains(&written), "{written}");
        }
        let left: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert!(
            left.iter()
                .all(|name| !name.to_string_lossy().starts_with(".pie-holes")),
            "{left:?}"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_file_laid_whole_past_the_available_bytes_is_refused_by_name() {
        let path = Path::new("somewhere/r0.kv");
        fits_whole(path, 10, 10).unwrap();
        let refused = fits_whole(path, 11, 10).unwrap_err();
        assert_ne!(refused.kind(), io::ErrorKind::StorageFull);
        let why = refused.to_string();
        for part in ["r0.kv", "11 bytes asked", "10 available"] {
            assert!(why.contains(part), "{why}");
        }
        assert_eq!(Room::laid_whole(50).for_us(), 0);
    }

    /// A full disk's codes are the kind the engines tell as the tier full.
    #[cfg(windows)]
    #[test]
    fn windows_disk_full_codes_are_storage_full() {
        for code in [39, 112] {
            let error = io::Error::from_raw_os_error(code);
            assert_eq!(error.kind(), io::ErrorKind::StorageFull, "{code}");
        }
    }

    /// A kept file an older run laid whole is taken afresh, its bytes given
    /// back; one that is sparse is kept.
    #[cfg(windows)]
    #[test]
    fn a_kept_file_laid_whole_is_taken_afresh_as_holes() {
        let dir = std::env::temp_dir().join(format!("pie-slot-whole-{}", std::process::id()));
        let budget = 8u64 << 20;
        std::fs::create_dir_all(&dir).unwrap();
        let whole = File::create(dir.join("r0.kv")).unwrap();
        whole.set_len(budget).unwrap();
        assert_eq!(allocated(&whole).unwrap(), budget, "laid whole unmarked");
        drop(whole);
        std::fs::write(dir.join("r0.kv.meta"), format!("a\n{}\n{}\n", 1 << 20, 8)).unwrap();
        for kept in [false, true] {
            let file = SlotFile::open_kept(&dir, "r0.kv", 1 << 20, budget, "a")
                .unwrap()
                .unwrap();
            assert_eq!(file.kept(), kept);
            assert_eq!(allocated(&file.file).unwrap(), 0);
        }
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
