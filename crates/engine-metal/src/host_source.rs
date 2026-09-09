use std::ops::{Deref, DerefMut};

use crate::error::{Fault, Result};

static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[must_use]
#[allow(clippy::unnecessary_fallible_conversions, clippy::useless_conversion)]
pub fn free_bytes(at: &std::path::Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(at.as_os_str().as_bytes()).ok()?;
    let mut said: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: a `statvfs` of a NUL-terminated path into a zeroed struct this
    // frame owns; the call reads the path and writes only that struct.
    if unsafe { libc::statvfs(path.as_ptr(), &raw mut said) } != 0 {
        return None;
    }
    u64::try_from(said.f_bavail)
        .ok()?
        .checked_mul(u64::try_from(said.f_frsize).ok()?)
}

#[must_use]
pub fn descriptors() -> u64 {
    NEXT.load(std::sync::atomic::Ordering::Relaxed)
}

pub struct HostSource {
    at: *mut u8,
    len: usize,
    file: Option<std::fs::File>,
}

// SAFETY: `at` is a `MAP_SHARED` mapping over a file this type created and
// unlinked, so no other process can reach the storage and no other owner
// exists in this one. What `Send` buys is the same thing it buys `Buffer`
// (`device::alloc`): the MOVE from the thread that loaded the model onto the
// thread that will fire it.
unsafe impl Send for HostSource {}

impl HostSource {
    pub fn open(bytes: u64) -> Result<HostSource> {
        let len = usize::try_from(bytes).unwrap_or(usize::MAX);
        if len == 0 {
            return Ok(HostSource {
                at: std::ptr::NonNull::<u8>::dangling().as_ptr(),
                len: 0,
                file: None,
            });
        }
        if let Some(free) = free_bytes(&std::env::temp_dir())
            && bytes > free
        {
            return Err(Fault::Backing {
                step: "admit",
                bytes,
                why: format!(
                    "staging this load's routed bands wants {:.2} GiB under {} and the \
                     volume has {:.2} GiB free. The file would size without complaint — \
                     `ftruncate` is sparse — and fill the pool as the landing wrote it. \
                     A load this size wants the WARM arm, which stages nothing: check \
                     the sentence the warm arm printed on its way past.",
                    bytes as f64 / (1u64 << 30) as f64,
                    std::env::temp_dir().display(),
                    free as f64 / (1u64 << 30) as f64,
                ),
            });
        }
        let at = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("pie-experts-{}-{at}", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|why| Fault::Backing {
                step: "open",
                bytes,
                why: format!("{} does not open: {why}", path.display()),
            })?;
        let _ = std::fs::remove_file(&path);
        file.set_len(bytes).map_err(|why| Fault::Backing {
            step: "size",
            bytes,
            why: why.to_string(),
        })?;
        // SAFETY: a fresh shared mapping over a file this function just created
        // and sized to `len`; the protections and length are stated here, and
        // the descriptor stays open beside it.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                std::os::fd::AsRawFd::as_raw_fd(&file),
                0,
            )
        };
        if at == libc::MAP_FAILED {
            return Err(Fault::Backing {
                step: "map",
                bytes,
                why: std::io::Error::last_os_error().to_string(),
            });
        }
        Ok(HostSource {
            at: at.cast(),
            len,
            file: Some(file),
        })
    }

    #[must_use]
    pub fn file(&self) -> Option<&std::fs::File> {
        self.file.as_ref()
    }

    pub fn settle(&mut self) {
        if self.len == 0 {
            return;
        }
        // SAFETY: both calls address exactly this type's own mapping, at its
        // stated length, and neither can invalidate it: `MS_ASYNC` schedules
        // writeback and `MADV_DONTNEED` over a `MAP_SHARED` file mapping
        // deactivates or drops frames whose contents the file still holds.
        unsafe {
            libc::msync(self.at.cast(), self.len, libc::MS_ASYNC);
            libc::madvise(self.at.cast(), self.len, libc::MADV_DONTNEED);
        }
    }

    #[must_use]
    pub fn backing(&self) -> Option<(u64, u64)> {
        let file = self.file.as_ref()?;
        // SAFETY: `stat` is written by `fstat` before it is read, over a
        // descriptor this type owns and keeps open.
        let stat = unsafe {
            let mut stat = std::mem::zeroed::<libc::stat>();
            if libc::fstat(std::os::fd::AsRawFd::as_raw_fd(file), &raw mut stat) != 0 {
                return None;
            }
            stat
        };
        Some((stat.st_size as u64, u64::from(stat.st_nlink)))
    }
}

impl Deref for HostSource {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: `len` readable bytes this type owns, or an empty slice over
        // a dangling-but-aligned pointer, which is what `from_raw_parts`
        // requires for a zero length.
        unsafe { std::slice::from_raw_parts(self.at, self.len) }
    }
}

impl DerefMut for HostSource {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: as `deref`, and `&mut self` is the exclusive access the
        // mapping's `PROT_WRITE` needs.
        unsafe { std::slice::from_raw_parts_mut(self.at, self.len) }
    }
}

impl std::fmt::Debug for HostSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostSource")
            .field("bytes", &self.len)
            .field("mapped", &self.file.is_some())
            .finish()
    }
}

impl Drop for HostSource {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        // SAFETY: unmapping the mapping this type created, at its own length.
        // The descriptor closes with `self.file` immediately after, and the
        // file is already unlinked — so this is the last reference to the
        // storage and the bytes go back to the filesystem here.
        unsafe {
            libc::munmap(self.at.cast(), self.len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_source_every_case() {
        a_staging_larger_than_the_volume_is_refused_by_the_numbers();
        a_source_that_streams_nothing_maps_nothing();
    }

    fn a_staging_larger_than_the_volume_is_refused_by_the_numbers() {
        let Some(free) = free_bytes(&std::env::temp_dir()) else {
            eprintln!("skipping: this filesystem does not answer statvfs");
            return;
        };
        let want = free + (1 << 30);
        let said = HostSource::open(want)
            .expect_err("a staging past the disk does not open")
            .to_string();
        assert!(
            said.contains("GiB free") && said.contains("wants"),
            "the refusal carries BOTH numbers: {said}"
        );

        let path = std::env::temp_dir().join(format!("pie-sparse-{}", std::process::id()));
        let file = std::fs::File::create(&path).expect("a scratch file");
        let took = file.set_len(want).is_ok();
        let allocated = std::fs::metadata(&path)
            .map(|it| std::os::unix::fs::MetadataExt::blocks(&it) * 512)
            .unwrap_or(0);
        let _ = std::fs::remove_file(&path);
        assert!(
            took && allocated < (1 << 20),
            "`ftruncate` of {want} bytes on a volume with {free} free is what this \
             refusal exists to get in front of, and it took it ({allocated} allocated)"
        );
    }

    fn a_source_that_streams_nothing_maps_nothing() {
        let source = HostSource::open(0).expect("the empty source is free");
        assert!(source.is_empty());
        assert!(
            source.backing().is_none(),
            "a full-residency load opens no file at all"
        );
    }
}
