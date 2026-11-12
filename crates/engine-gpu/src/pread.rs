//! Positional reads of a file into memory the caller holds, spread over
//! worker threads that share one `&File`.
#![allow(unsafe_code)]

use crate::error::{Fault, Result};

/// # Safety
///
/// Each job's `(dst, _, len)` must be `len` writable bytes, live for the call
/// and disjoint from every other job's.
pub unsafe fn pread_jobs(
    file: &std::fs::File,
    jobs: &[(usize, u64, u64)],
    threads: usize,
) -> Result<()> {
    let threads = threads.clamp(1, jobs.len().max(1));
    let per = jobs.len().div_ceil(threads).max(1);
    let failed: std::sync::Mutex<Option<Fault>> = std::sync::Mutex::new(None);
    std::thread::scope(|scope| {
        for chunk in jobs.chunks(per) {
            let failed = &failed;
            scope.spawn(move || {
                for &(dst, from, len) in chunk {
                    // SAFETY: per this fn's contract, `dst` is `len` writable
                    // bytes no other job touches.
                    if let Err(why) = unsafe { pread_all(file, dst as *mut u8, from, len) } {
                        *failed.lock().expect("failed lock poisoned") = Some(why);
                        return;
                    }
                }
            });
        }
    });
    match failed.into_inner().expect("failed lock poisoned") {
        Some(why) => Err(why),
        None => Ok(()),
    }
}

unsafe fn pread_all(file: &std::fs::File, dst: *mut u8, from: u64, len: u64) -> Result<()> {
    let mut done = 0u64;
    while done < len {
        let want = usize::try_from(len - done)
            .unwrap_or(usize::MAX)
            .min(1 << 30);
        // SAFETY: `dst + done` is `want` writable bytes inside the span.
        let got = match unsafe { pread_chunk(file, dst.add(done as usize), from + done, want) } {
            Ok(got) => got,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) => {
                return Err(Fault::Device {
                    call: "pread",
                    why: err.to_string(),
                });
            }
        };
        if got == 0 {
            return Err(Fault::Device {
                call: "pread",
                why: format!("short read at {}: {} of {len} bytes", from + done, done),
            });
        }
        done += got as u64;
    }
    Ok(())
}
/// Reads one positional chunk. Both arms are positional and leave the file
/// cursor alone, so the worker threads may share one `&File`.
#[cfg(unix)]
unsafe fn pread_chunk(
    file: &std::fs::File,
    dst: *mut u8,
    at: u64,
    want: usize,
) -> std::io::Result<usize> {
    use std::os::fd::AsRawFd;
    // SAFETY: `dst` is `want` writable bytes the caller vouches for.
    let got = unsafe {
        libc::pread(
            file.as_raw_fd(),
            dst.cast::<libc::c_void>(),
            want,
            libc::off_t::try_from(at).unwrap_or(libc::off_t::MAX),
        )
    };
    if got < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(got as usize)
    }
}

#[cfg(windows)]
unsafe fn pread_chunk(
    file: &std::fs::File,
    dst: *mut u8,
    at: u64,
    want: usize,
) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    // SAFETY: `dst` is `want` writable bytes the caller vouches for.
    let buf = unsafe { std::slice::from_raw_parts_mut(dst, want) };
    file.seek_read(buf, at)
}
