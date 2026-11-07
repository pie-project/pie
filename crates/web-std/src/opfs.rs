#![allow(unsafe_code)]
//! Files in the origin's private file system, through the page's
//! `FileSystemSyncAccessHandle`s (`platform.mjs`). Opening one and asking for
//! the free space are promises, so those park a green thread. Reads, writes
//! and flushes are synchronous, and only a Worker has them.
//!
//! OPFS cannot unlink an open file, so nothing removes a file whose tab died.
//! Each open takes a name of its own and removes what it finds in the
//! directory instead, one tab at a time: the handle a live tab holds is an
//! exclusive lock, which refuses the removal, so only a dead tab's files go.
//!
//! The quota is an estimate and a file's length is sparse, so neither promises
//! room: a [`File`] that opened can still fail a write, which is an error.

use std::cell::RefCell;
use std::io;
use std::task::{Poll, Waker};

#[link(wasm_import_module = "./platform.mjs")]
unsafe extern "C" {
    fn pie_opfs_open(
        dir: *const u8,
        dir_len: usize,
        name: *const u8,
        name_len: usize,
        len: f64,
        request: u32,
    );
    fn pie_opfs_free_space(dir: *const u8, dir_len: usize, request: u32);
    fn pie_opfs_read(handle: u32, at: f64, bytes: *mut u8, len: usize) -> f64;
    fn pie_opfs_write(handle: u32, at: f64, bytes: *const u8, len: usize) -> f64;
    fn pie_opfs_flush(handle: u32) -> i32;
    fn pie_opfs_close(handle: u32);
}

#[derive(Default)]
struct Pending {
    answer: Option<f64>,
    waker: Option<Waker>,
}

thread_local! {
    static PENDING: RefCell<slab::Slab<Pending>> = const { RefCell::new(slab::Slab::new()) };
}

/// Parks this green thread until the page answers the request `start` makes.
fn ask(start: impl FnOnce(u32)) -> f64 {
    let request = PENDING.with_borrow_mut(|pending| pending.insert(Pending::default()));
    start(request as u32);
    let answer = crate::block_on_poll(|cx| {
        PENDING.with_borrow_mut(|pending| {
            let slot = &mut pending[request];
            match slot.answer {
                Some(answer) => Poll::Ready(answer),
                None => {
                    slot.waker = Some(cx.waker().clone());
                    Poll::Pending
                }
            }
        })
    });
    PENDING.with_borrow_mut(|pending| pending.remove(request));
    answer
}

/// The page's answer to a request of [`ask`].
#[unsafe(no_mangle)]
pub extern "C" fn pie_opfs_answer(request: u32, answer: f64) {
    let waker = PENDING.with_borrow_mut(|pending| {
        let slot = &mut pending[request as usize];
        slot.answer = Some(answer);
        slot.waker.take()
    });
    if let Some(waker) = waker {
        waker.wake();
    }
}

/// What `navigator.storage.estimate()` says `dir` can still take, quota less
/// usage, once the files of dead tabs are gone: not what the browser enforces. Zero when this context has no synchronous
/// handles (the main thread) or no storage. The page logs why.
#[must_use]
pub fn free_space(dir: &str) -> u64 {
    ask(|request| {
        // SAFETY: the page decodes `dir` before this call returns.
        unsafe { pie_opfs_free_space(dir.as_ptr(), dir.len(), request) }
    }) as u64
}

/// A zero-filled file only this tab can reach, removed when dropped.
#[derive(Debug)]
pub struct File {
    handle: u32,
}

impl File {
    /// A new file of `len` bytes under `dir`, its name unique to this open
    /// and ending in `name`. `None` when the browser refuses it, for want of
    /// the API or the quota. The page logs why.
    #[must_use]
    pub fn create(dir: &str, name: &str, len: u64) -> Option<Self> {
        let handle = ask(|request| {
            // SAFETY: the page decodes both strings before this call returns.
            unsafe {
                pie_opfs_open(
                    dir.as_ptr(),
                    dir.len(),
                    name.as_ptr(),
                    name.len(),
                    len as f64,
                    request,
                );
            }
        }) as u32;
        (handle != 0).then(|| Self { handle })
    }

    pub fn read_exact_at(&self, buf: &mut [u8], at: u64) -> io::Result<()> {
        // SAFETY: the page writes at most `len` bytes into a view of this
        // memory it builds and drops within the call.
        let read = unsafe { pie_opfs_read(self.handle, at as f64, buf.as_mut_ptr(), buf.len()) };
        whole("read", read, buf.len(), at)
    }

    pub fn write_all_at(&self, buf: &[u8], at: u64) -> io::Result<()> {
        // SAFETY: the page reads `len` bytes through a view of this memory
        // it builds and drops within the call.
        let wrote = unsafe { pie_opfs_write(self.handle, at as f64, buf.as_ptr(), buf.len()) };
        whole("write", wrote, buf.len(), at)
    }

    pub fn sync_data(&self) -> io::Result<()> {
        // SAFETY: the page takes only the handle's number.
        match unsafe { pie_opfs_flush(self.handle) } {
            0 => Ok(()),
            _ => Err(io::Error::other(
                "OPFS refused a flush; the page logged why",
            )),
        }
    }
}

impl Drop for File {
    fn drop(&mut self) {
        // SAFETY: the page takes only the handle's number.
        unsafe { pie_opfs_close(self.handle) }
    }
}

/// Past its real limit a handle does not throw: a write answers a huge count
/// (-8 as unsigned, in Chromium 153). Anything but the whole length is lost
/// data, and kv cannot be computed again.
fn whole(what: &str, done: f64, len: usize, at: u64) -> io::Result<()> {
    if done == len as f64 {
        return Ok(());
    }
    Err(io::Error::other(format!(
        "an OPFS {what} of {len} bytes at {at} answered {done}"
    )))
}
