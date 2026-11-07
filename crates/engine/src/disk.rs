//! Slot files below host memory, one per rank and kind. On a host they sit
//! on its filesystem, unlinked once open so they never outlive the process,
//! or kept, for a persistent tier. In a browser they sit in the origin's
//! private file system, which can do neither.

use std::io;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod opfs;

#[cfg(not(target_arch = "wasm32"))]
pub use native::{free_space, kept_bytes};
#[cfg(target_arch = "wasm32")]
pub use opfs::free_space;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(target_arch = "wasm32")]
use web_std::opfs::File;

/// Slot offsets and lengths are multiples of this, as direct I/O asks.
pub const ALIGN: usize = 4096;

#[derive(Debug)]
pub struct SlotFile {
    file: File,
    stride: usize,
    slots: u32,
    kept: bool,
}

/// The stride and the slots `budget` holds of `slot_bytes` each; `None` when
/// that is none.
fn geometry(slot_bytes: u64, budget: u64) -> io::Result<Option<(usize, u32)>> {
    let stride = usize::try_from(slot_bytes)
        .map_err(io::Error::other)?
        .next_multiple_of(ALIGN);
    let slots = u32::try_from(budget.checked_div(stride as u64).unwrap_or(0)).unwrap_or(u32::MAX);
    Ok((slot_bytes != 0 && slots != 0).then_some((stride, slots)))
}

impl SlotFile {
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
