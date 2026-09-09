use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::plan::{LoadPlan, SourceExtent, StorageInstr};

pub fn release(file: &std::fs::File, offset: u64, len: u64) {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: `fallocate` only changes the allocation of the range it is
        // given on a file descriptor the caller owns; it writes no memory.
        // Within the range, whole blocks are deallocated and partial ones are
        // zeroed, so no byte outside `offset..offset + len` is touched --
        // which is what lets a caller release a range whose neighbours have
        // not been read yet. `KEEP_SIZE` is redundant with `PUNCH_HOLE`, which
        // never moves the end of the file, and is passed because the manual
        // requires the pair.
        unsafe {
            libc::fallocate(
                file.as_raw_fd(),
                libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                offset as libc::off_t,
                len as libc::off_t,
            );
        }
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        let block = match file.metadata() {
            Ok(metadata) => {
                use std::os::unix::fs::MetadataExt;
                let reported = metadata.blksize();
                if reported > 0 { reported } else { 4096 }
            }
            Err(_) => 4096,
        };
        let start = offset.div_ceil(block) * block;
        let end = (offset + len) / block * block;
        if end > start {
            let mut hole = libc::fpunchhole_t {
                fp_flags: 0,
                reserved: 0,
                fp_offset: start as libc::off_t,
                fp_length: (end - start) as libc::off_t,
            };
            // SAFETY: `F_PUNCHHOLE` reads the `fpunchhole_t` this call owns and
            // deallocates the range it names on a descriptor the caller owns;
            // it writes no memory of ours and never moves the file's end. The
            // range is block-aligned and rounded inward, so it lies strictly
            // within `offset..offset + len`.
            unsafe {
                libc::fcntl(file.as_raw_fd(), libc::F_PUNCHHOLE, &raw mut hole);
            }
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (file, offset, len);
    }
}

type Range = (u64, u64);

#[derive(Debug, Default)]
struct FileReads {
    reads: Vec<Range>,
    blocked: Vec<Range>,
    longest: u64,
    longest_blocked: u64,
}

#[derive(Debug, Default)]
pub struct SourceLedger {
    by_file: HashMap<PathBuf, FileReads>,
    sorted: bool,
}

impl SourceLedger {
    #[must_use]
    pub fn of(plan: &LoadPlan, base: &Path) -> Self {
        let mut ledger = Self::default();
        let paths: HashMap<u32, PathBuf> = plan
            .files
            .iter()
            .map(|file| {
                let path = PathBuf::from(&file.path);
                let path = if path.is_absolute() {
                    path
                } else {
                    base.join(path)
                };
                (file.id.0, path)
            })
            .collect();

        let mut runs: HashMap<u32, usize> = HashMap::new();
        for id in &plan.schedule {
            *runs.entry(id.0).or_default() += 1;
        }
        for instr in &plan.instrs {
            let (id, source) = match instr {
                StorageInstr::ExtentWrite { id, source, .. }
                | StorageInstr::BulkExtentWrite { id, source, .. }
                | StorageInstr::GatherWrite { id, source, .. } => (id, Some(source)),
                StorageInstr::TileMap { id, source, .. } => (id, source.as_ref()),
                _ => continue,
            };
            let Some(source) = source else { continue };
            let Some(path) = paths.get(&source.file_id.0) else {
                continue;
            };
            let Some(range) = physical_range(source) else {
                continue;
            };
            let times = runs.get(&id.0).copied().unwrap_or(1).max(1);
            let entry = ledger.file_mut(path);
            for _ in 0..times {
                entry.reads.push(range);
            }
            entry.longest = entry.longest.max(range.1 - range.0);
        }

        for instr in &plan.instrs {
            let StorageInstr::TileMap { transform, .. } = instr else {
                continue;
            };
            let Some(metadata_source) = transform.metadata_source else {
                continue;
            };
            let Some(decl) = plan.sources.iter().find(|decl| decl.id == metadata_source) else {
                continue;
            };
            let Some(path) = paths.get(&decl.file_id.0) else {
                continue;
            };
            let range = (decl.file_offset, decl.file_offset + decl.span_bytes);
            let entry = ledger.file_mut(path);
            entry.blocked.push(range);
            entry.longest_blocked = entry.longest_blocked.max(decl.span_bytes);
        }
        ledger.sort();
        ledger
    }

    pub fn also_read(&mut self, path: &Path, offset: u64, len: u64) {
        if len == 0 {
            return;
        }
        let entry = self.file_mut(path);
        entry.reads.push((offset, offset + len));
        entry.longest = entry.longest.max(len);
        self.sorted = false;
    }

    #[must_use]
    pub fn last_read(&self, path: &Path, offset: u64, len: u64) -> bool {
        if len == 0 {
            return false;
        }
        debug_assert!(self.sorted, "ask `sort` before `last_read`");
        let Some(entry) = self.by_file.get(&canonical(path)) else {
            return false;
        };
        let (start, end) = (offset, offset + len);
        if overlaps(&entry.blocked, entry.longest_blocked, start, end) > 0 {
            return false;
        }
        let mut covering = 0usize;
        let mut touching = 0usize;
        for &(read_start, read_end) in window(&entry.reads, entry.longest, start, end) {
            if read_start < end && read_end > start {
                touching += 1;
                if read_start <= start && read_end >= end {
                    covering += 1;
                }
            }
        }
        touching == 1 && covering == 1
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_file.is_empty()
    }

    pub fn sort(&mut self) {
        for entry in self.by_file.values_mut() {
            entry.reads.sort_unstable();
            entry.blocked.sort_unstable();
        }
        self.sorted = true;
    }

    fn file_mut(&mut self, path: &Path) -> &mut FileReads {
        self.by_file.entry(canonical(path)).or_default()
    }
}

fn physical_range(source: &SourceExtent) -> Option<Range> {
    let mut normalized = source.stride.clone();
    let base_offset = normalized.base_offset;
    normalized.base_offset = 0;
    let len = crate::executor::walk::physical_source_bytes(&normalized).ok()?;
    if len == 0 {
        return None;
    }
    let start = source.file_offset.checked_add(base_offset)?;
    Some((start, start.checked_add(len)?))
}

fn window(ranges: &[Range], longest: u64, start: u64, end: u64) -> &[Range] {
    let floor = start.saturating_sub(longest);
    let from = ranges.partition_point(|(range_start, _)| *range_start < floor);
    let rest = &ranges[from..];
    let to = rest.partition_point(|(range_start, _)| *range_start < end);
    &rest[..to]
}

fn overlaps(ranges: &[Range], longest: u64, start: u64, end: u64) -> usize {
    window(ranges, longest, start, end)
        .iter()
        .filter(|(range_start, range_end)| *range_start < end && *range_end > start)
        .count()
}

fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}
