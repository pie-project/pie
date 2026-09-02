//! `--consume-source`'s mechanism: giving a source's bytes back to the
//! filesystem as the import reads them, so the old copy shrinks while the new
//! one grows and a conversion needs room for one checkpoint rather than two.
//!
//! Two halves, and the second is the whole difficulty:
//!
//! * [`release`] is the syscall — `fallocate(PUNCH_HOLE)` on Linux,
//!   `F_PUNCHHOLE` on Darwin — which deallocates a range without moving the
//!   file's end or touching a byte outside it.
//! * [`SourceLedger`] is the question the syscall may not be asked without an
//!   answer to: *is this the last read of these bytes?* Punching a range some
//!   later read still needs does not fail — the read succeeds and returns
//!   ZEROS — so the artifact would come out plausible and wrong. The ledger is
//!   what makes the answer a proof rather than a hope.
//!
//! # The proof the ledger carries
//!
//! Every byte an import reads out of the source is read at one of exactly
//! three places, and the ledger accounts for all three:
//!
//! 1. `executor::walk::read_extent`, the single funnel every *decode* read
//!    passes through. Its ranges are enumerable from the plan before the first
//!    instruction runs: one [`SourceExtent`] per source-naming instruction,
//!    at `file_offset + stride.base_offset` for `physical_source_bytes(stride)`
//!    bytes.
//! 2. `walk::fp8_block_operand`, which reads a *second* tensor the instruction
//!    names by [`TransformSpec::metadata_source`] — the block-scale sibling of
//!    an FP8 payload — once per shard, OUT OF BAND of the instruction's own
//!    extent. This is the shared-blob case, and the reason the ledger has a
//!    blocked list at all: those spans are never releasable and neither is
//!    anything that overlaps them.
//! 3. `pie model import`'s own passthrough copy loop, whose spans the importer
//!    declares with [`SourceLedger::also_read`] because the plan does not know
//!    about them.
//!
//! A range is released only when the ledger can say that exactly one read in
//! the whole import covers it and nothing blocked touches it. Anything it
//! cannot prove — a range read twice, a range overlapping a scale tensor, a
//! range in a file the ledger never heard of — is simply not released, which
//! costs disk and nothing else. **Partial credit is the design.** The
//! conversion-heavy checkpoint this was written for
//! (`mlx-community/DeepSeek-V4-Flash-2bit-DQ`, 89.9 GiB, 99.7% decode) has its
//! 89.7 GiB of expert banks in the provable set, and that is where the peak
//! lives.
//!
//! # What an aborted import leaves behind
//!
//! A punched range is gone: `--consume-source`'s stated contract is that the
//! source does not survive the run. What the ledger preserves is the weaker
//! property that still matters — **an unpunched range is intact**, byte for
//! byte, because [`release`] never touches anything outside the range it is
//! given and rounds INWARD when it must align. So an import killed halfway
//! leaves a source that is holed where it was read and whole where it was not,
//! and the operator can tell which is which by reading it.
//!
//! [`SourceExtent`]: crate::plan::SourceExtent
//! [`TransformSpec::metadata_source`]: crate::plan::TransformSpec::metadata_source

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::plan::{LoadPlan, SourceExtent, StorageInstr};

/// Gives a range of a file back to the filesystem, without changing the file's
/// length or the bytes outside the range.
///
/// Three callers, one situation: bytes that have just been read for the last
/// time, in a file that is going to be deleted. Releasing as the read passes
/// over them means the old copy shrinks while the new one grows, instead of
/// both standing at full size until the end — which is the difference between
/// needing room for two copies of a checkpoint and needing room for one.
///
/// Best-effort by design, and unchecked for that reason. A filesystem that
/// cannot do this fails the call and the file stays whole, which costs space
/// and nothing else: the bytes were read either way, and the caller deletes
/// the file either way.
///
/// **THE CALLER OWES THE PROOF, NOT THIS FUNCTION.** Punching a range a later
/// read still needs raises no error — the read returns zeros. [`SourceLedger`]
/// is where that proof is kept.
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
    // **AND THE SAME THING ON APFS**, so the mechanism is not Linux-only.
    // Without this arm `--consume-source` would report that it is consuming
    // the source as it reads while giving nothing back, and peak use would be
    // source PLUS artifact — exactly what the flag exists to avoid.
    //
    // `F_PUNCHHOLE` is `fallocate`'s deallocating half under another name, with
    // one difference that decides the arithmetic below: Linux zeroes the
    // partial blocks at the ends of the range, and Darwin REFUSES a range whose
    // ends are not block-aligned (`EINVAL`). Rounding outward to fix that would
    // free blocks this call was not given — and the callers release a range
    // whose NEIGHBOURS HAVE NOT BEEN READ YET, so outward rounding would hand
    // back bytes the import still needs and the artifact would take zeros. So
    // it rounds INWARD: the first whole block at or after `offset`, the last
    // whole block at or before the end. What that costs is up to one block at
    // each end left allocated, per call; what it buys is the invariant the
    // Linux arm states in its own SAFETY note — no byte outside the range is
    // touched — holding on both platforms for the same reason.
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        // The allocation block this file is actually made of, not a guess.
        // 4 KiB is the APFS default and the fallback; a filesystem that
        // reports something else (or nothing) is served by asking.
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

/// One half-open byte range of one file.
type Range = (u64, u64);

/// Every read one import will make of one source file, and every range of it
/// that may not be released whatever the reads say.
#[derive(Debug, Default)]
struct FileReads {
    /// Sorted by start. One entry per read, WITH MULTIPLICITY: a range two
    /// instructions both read appears twice, which is what makes it fail the
    /// "exactly one" test below.
    reads: Vec<Range>,
    /// Ranges read at a place the plan does not enumerate. Nothing overlapping
    /// one of these is ever released.
    blocked: Vec<Range>,
    /// The longest read on this file, which bounds how far back a query has to
    /// scan to find every range that could overlap it.
    longest: u64,
    /// The longest blocked range, for the same reason.
    longest_blocked: u64,
}

/// Which source ranges an import may hand back to the filesystem as it reads
/// them.
///
/// Built from the plan before a byte moves, extended by the importer with the
/// reads the plan does not know about, and then asked one question at each
/// read site: [`SourceLedger::last_read`]. See the module note for the proof
/// this is complete against the executor's read sites.
///
/// Keyed by CANONICAL path. A HuggingFace snapshot is a directory of symlinks
/// into `blobs/`, so two names can be one file and the ledger has to know it —
/// a read counted under one spelling and a blocker recorded under the other
/// would be a proof about two files that are one.
#[derive(Debug, Default)]
pub struct SourceLedger {
    by_file: HashMap<PathBuf, FileReads>,
    sorted: bool,
}

impl SourceLedger {
    /// The reads `plan` will perform, and the spans it reads out of band.
    ///
    /// `base` resolves the plan's relative file paths the same way
    /// [`executor::walk`](crate::executor::walk) does, because a ledger that
    /// named the files differently would answer about a different checkpoint.
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

        // A schedule may name an instruction more than once; the count is what
        // the executor will actually do, and one is the floor because an
        // instruction absent from the schedule reads nothing but must not make
        // a range look like it is read fewer times than it is.
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

        // **THE OUT-OF-BAND READ**, which is why this is not just a multiset of
        // the instructions' own extents. `walk::fp8_block_operand` reads the
        // whole of the tensor an instruction names by `metadata_source` — the
        // block-scale sibling of an FP8 payload — once per shard that
        // references it, at a call site the plan's instruction list does not
        // describe. Blocking the span is the honest answer: it is small, it is
        // shared, and a single-read proof about it would be a guess.
        for instr in &plan.instrs {
            let StorageInstr::TileMap { transform, .. } = instr else {
                continue;
            };
            let Some(metadata_source) = transform.metadata_source else {
                continue;
            };
            let Some(decl) = plan
                .sources
                .iter()
                .find(|decl| decl.id == metadata_source)
            else {
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

    /// Declare a read the plan does not describe.
    ///
    /// `pie model import` copies its passthrough tensors itself, outside the
    /// executor entirely, and those spans have to be in the same ledger as the
    /// decode's — otherwise the decode would release a range the copy loop has
    /// not read yet, or the copy loop would release one the decode still
    /// wants. One ledger, one answer, both paths.
    pub fn also_read(&mut self, path: &Path, offset: u64, len: u64) {
        if len == 0 {
            return;
        }
        let entry = self.file_mut(path);
        entry.reads.push((offset, offset + len));
        entry.longest = entry.longest.max(len);
        self.sorted = false;
    }

    /// Whether the range `offset..offset + len` of `path` has now been read for
    /// the last time this import will read it — and may therefore be released.
    ///
    /// True when exactly one recorded read overlaps the range AND that read
    /// CONTAINS it, and nothing blocked overlaps it. Containment rather than
    /// equality because the passthrough loop reads a tensor's span in chunks:
    /// the chunks partition the span, each is read once, so a chunk inside a
    /// span nothing else touches is as dead as the span would be.
    ///
    /// Everything it cannot prove is false, which costs disk and never a byte.
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

    /// Whether this ledger knows about any file at all. An import with nothing
    /// to release opens its sources read-only, which is one fewer way to
    /// damage a checkpoint the operator asked to keep.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_file.is_empty()
    }

    /// Put every file's ranges in start order. Called by [`Self::of`] and after
    /// the importer has finished declaring its own reads.
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

/// The physical byte range one source extent reads, computed exactly the way
/// `walk::read_extent` computes the arguments it hands `read_file` — the base
/// offset folded into the start, the strided span measured with a zero base.
/// A ledger that measured this differently would answer about ranges the
/// executor never asks it about.
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

/// The slice of a sorted range list that can contain anything overlapping
/// `start..end`: a range overlapping it starts before `end` and ends after
/// `start`, and no range is longer than `longest`, so none of them starts
/// before `start - longest`.
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

/// The path a ledger entry is filed under. `canonicalize` is what makes a
/// snapshot's symlink and its blob one key; a path that cannot be resolved is
/// filed under itself, which is still consistent as long as every caller
/// spells it the same way — and a spelling nobody matches only loses a
/// release.
fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

