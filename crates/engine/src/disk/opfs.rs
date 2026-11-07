//! Slot files in the origin's private file system. A file there cannot be
//! unlinked while open, and a browser may not offer one at all: `web_std::opfs`
//! says what it holds to, and a tab it refuses serves with no disk tier.

use std::io;
use std::path::Path;

use web_std::opfs::File;

use super::{SlotFile, geometry};

impl SlotFile {
    /// A file of its own under `dir` with as many slots of `slot_bytes` as
    /// `budget` holds, `name` naming it for whoever reads the directory;
    /// `None` when that is none, or when the browser gives no file: a tab
    /// boots on what its platform has. Must run on a green thread.
    pub fn open(dir: &Path, name: &str, slot_bytes: u64, budget: u64) -> io::Result<Option<Self>> {
        let Some((stride, slots)) = geometry(slot_bytes, budget)? else {
            return Ok(None);
        };
        let len = u64::from(slots) * stride as u64;
        let file = File::create(&dir.to_string_lossy(), name, len);
        Ok(file.map(|file| Self {
            file,
            stride,
            slots,
            kept: false,
        }))
    }
}

/// Zero when the browser offers no synchronous OPFS here, so an unstated
/// budget comes to nothing and the tab serves without the tier. Must run on
/// a green thread.
#[must_use]
pub fn free_space(dir: &Path) -> u64 {
    web_std::opfs::free_space(&dir.to_string_lossy())
}
