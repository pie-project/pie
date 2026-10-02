use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::encode::Mark;
use crate::error::{Fault, Result};

use super::ctx::{Context, Core, note_reservation};
use super::host::Signal;

#[cfg(not(target_arch = "wasm32"))]
const PREAD_CHUNK: u64 = 256 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Memory {
    Device,

    Host,

    Staging,

    Readback,
}

static IDS: AtomicU64 = AtomicU64::new(1);

pub(crate) struct Raw {
    pub(crate) core: Arc<Core>,
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) size: u64,
    pub(crate) kind: Memory,
    pub(crate) mappable: bool,
    pub(crate) id: u64,
}

pub(crate) type Slab = Arc<Raw>;

impl Raw {
    pub(crate) fn new(core: &Arc<Core>, bytes: u64, kind: Memory) -> Result<Slab> {
        note_reservation();
        let size = bytes.max(4).next_multiple_of(4);
        if size > core.limits.max_buffer_size {
            return Err(Fault::Ceiling {
                what: "bytes of one device reservation",
                need: size,
                have: core.limits.max_buffer_size,
            });
        }
        let mappable =
            kind == Memory::Readback || (kind != Memory::Device && core.enabled.mappable_primary);
        let mut usage = if kind == Memory::Readback {
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
        } else {
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
        };
        if mappable && kind != Memory::Readback {
            usage |= wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE;
        }
        let buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        });
        if let Err(fault) = core.take_error("create_buffer") {
            return Err(match fault {
                Fault::Wgpu { why, .. } if why.contains("emory") => Fault::Ceiling {
                    what: "device memory",
                    need: size,
                    have: core
                        .device_local
                        .saturating_sub(core.allocated.load(Ordering::Relaxed)),
                },
                other => other,
            });
        }
        core.allocated.fetch_add(size, Ordering::Relaxed);

        let mut encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pie clear"),
            });
        encoder.clear_buffer(&buffer, 0, None);
        core.queue.submit(std::iter::once(encoder.finish()));
        core.take_error("clear_buffer")?;
        Ok(Arc::new(Raw {
            core: Arc::clone(core),
            buffer,
            size,
            kind,
            mappable,
            id: IDS.fetch_add(1, Ordering::Relaxed),
        }))
    }

    pub(crate) fn span(&self, offset: u64, len: u64) -> Result<()> {
        match offset.checked_add(len) {
            Some(end) if end <= self.size => Ok(()),
            _ => Err(Fault::Ceiling {
                what: "bytes of a device reservation",
                need: offset.saturating_add(len),
                have: self.size,
            }),
        }
    }

    pub(crate) fn write(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        self.span(offset, bytes.len() as u64)?;
        if bytes.is_empty() {
            return Ok(());
        }
        let len = bytes.len() as u64;
        if offset.is_multiple_of(4) && len.is_multiple_of(4) {
            self.core.queue.write_buffer(&self.buffer, offset, bytes);
            return Ok(());
        }
        self.write_unaligned(offset, bytes)
    }

    fn write_unaligned(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        let len = bytes.len() as u64;
        let start = offset & !3;
        let end = (offset + len).next_multiple_of(4).min(self.size);
        let mut window = vec![0u8; (end - start) as usize];
        self.read(start, &mut window)?;
        let skip = (offset - start) as usize;
        window[skip..skip + bytes.len()].copy_from_slice(bytes);
        self.core.queue.write_buffer(&self.buffer, start, &window);
        Ok(())
    }

    pub(crate) fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        self.span(offset, into.len() as u64)?;
        if into.is_empty() {
            return Ok(());
        }
        let back = Signal::new();
        let deliver = back.clone();
        self.read_async(offset, into.len() as u64, move |raw| deliver.notify(raw));
        let raw = back.park("Buffer::read")??;
        into.copy_from_slice(&raw[..into.len()]);
        Ok(())
    }

    pub(crate) fn read_async(
        &self,
        offset: u64,
        len: u64,
        on_read: impl FnOnce(Result<Vec<u8>>) + Send + 'static,
    ) {
        if let Err(fault) = self.span(offset, len) {
            return on_read(Err(fault));
        }
        if len == 0 {
            return on_read(Ok(Vec::new()));
        }
        let core = &self.core;
        let start = offset & !3;
        let end = (offset + len).next_multiple_of(4).min(self.size);
        let span = end - start;
        let skip = (offset - start) as usize;
        let take = len as usize;
        if self.kind == Memory::Readback {
            let mapped = self.buffer.clone();
            let core = Arc::clone(core);
            self.buffer
                .slice(start..end)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let read = result
                        .map_err(|e| core.fault("map_async", e))
                        .and_then(|()| {
                            let view = mapped
                                .slice(start..end)
                                .get_mapped_range()
                                .map_err(|e| core.fault("get_mapped_range", e))?;
                            Ok(view[skip..skip + take].to_vec())
                        });
                    mapped.unmap();
                    on_read(read);
                });
            self.core.kick();
            return;
        }
        core.destroy_spent();
        let staging = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pie read"),
            size: span,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pie read"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, start, &staging, 0, Some(span));
        core.queue.submit(std::iter::once(encoder.finish()));
        let mapped = staging.clone();
        let core = Arc::clone(core);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let read = result
                    .map_err(|e| core.fault("map_async", e))
                    .and_then(|()| {
                        let view = mapped
                            .slice(..)
                            .get_mapped_range()
                            .map_err(|e| core.fault("get_mapped_range", e))?;
                        Ok(view[skip..skip + take].to_vec())
                    });
                mapped.unmap();
                core.spend(mapped);
                on_read(read);
            });
        self.core.kick();
    }

    pub(crate) fn zero(&self, offset: u64, len: u64) -> Result<()> {
        self.span(offset, len)?;
        if len == 0 {
            return Ok(());
        }
        let start = offset & !3;
        let end = (offset + len).next_multiple_of(4).min(self.size);
        let core = &self.core;
        let mut encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pie zero"),
            });
        encoder.clear_buffer(&self.buffer, start, Some(end - start));
        core.queue.submit(std::iter::once(encoder.finish()));
        core.take_error("clear_buffer")
    }
}

impl Drop for Raw {
    fn drop(&mut self) {
        self.core.allocated.fetch_sub(self.size, Ordering::Relaxed);
        // A browser frees the memory only when the JS wrapper is collected.
        self.buffer.destroy();
    }
}

#[derive(Clone)]
pub struct Buffer {
    slab: Slab,
    bytes: u64,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("bytes", &self.bytes)
            .field("kind", &self.slab.kind)
            .finish()
    }
}

impl Buffer {
    pub fn zeroed(device: &Context, bytes: u64) -> Result<Buffer> {
        Buffer::with(device, bytes, Memory::Device)
    }

    pub fn host(device: &Context, bytes: u64) -> Result<Buffer> {
        Buffer::with(device, bytes, Memory::Host)
    }

    pub fn with(device: &Context, bytes: u64, kind: Memory) -> Result<Buffer> {
        let slab = Raw::new(device.core(), bytes, kind)?;
        Ok(Buffer { slab, bytes })
    }

    #[must_use]
    pub fn is_mapped(&self) -> bool {
        false
    }

    #[must_use]
    pub fn is_host(&self) -> bool {
        self.slab.mappable
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn slab(&self) -> &Slab {
        &self.slab
    }

    pub fn span(&self, offset: u64, len: u64) -> Result<()> {
        match offset.checked_add(len) {
            Some(end) if end <= self.bytes => Ok(()),
            _ => Err(Fault::Ceiling {
                what: "bytes of a device reservation",
                need: offset.saturating_add(len),
                have: self.bytes,
            }),
        }
    }

    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        self.span(offset, bytes.len() as u64)?;
        let started = Mark::now();
        let out = self.slab.write(offset, bytes);
        crate::encode::record_io(true, started.ns());
        out
    }

    pub fn zero_span(&mut self, offset: u64, len: u64) -> Result<()> {
        self.span(offset, len)?;
        self.slab.zero(offset, len)
    }

    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        self.span(offset, into.len() as u64)?;
        let started = Mark::now();
        let out = self.slab.read(offset, into);
        crate::encode::record_io(false, started.ns());
        out
    }

    pub fn read_async(
        &self,
        offset: u64,
        len: u64,
        on_read: impl FnOnce(Result<Vec<u8>>) + Send + 'static,
    ) {
        if let Err(fault) = self.span(offset, len) {
            return on_read(Err(fault));
        }
        self.slab.read_async(offset, len, on_read);
    }

    pub(crate) fn queue_verdict(&self) -> Result<()> {
        self.slab.core.take_error("submission")
    }

    pub fn write_from_file(
        &mut self,
        file: &std::fs::File,
        jobs: &[(u64, u64, u64)],
        threads: usize,
    ) -> Result<()> {
        let writer = self.file_writer(jobs)?;
        writer.pread(file, jobs, threads)
    }

    pub fn file_writer(&mut self, jobs: &[(u64, u64, u64)]) -> Result<FileWriter> {
        for &(into, _, len) in jobs {
            self.span(into, len)?;
        }
        Ok(FileWriter {
            slab: Arc::clone(&self.slab),
        })
    }
}

pub struct FileWriter {
    slab: Slab,
}

impl FileWriter {
    #[cfg(target_arch = "wasm32")]
    pub fn pread(
        &self,
        _file: &std::fs::File,
        jobs: &[(u64, u64, u64)],
        _threads: usize,
    ) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        for &(into, _, len) in jobs {
            self.slab.span(into, len)?;
        }
        Err(Fault::Device {
            call: "pread",
            why: "a browser host has no file to read positionally".to_string(),
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn pread(
        &self,
        file: &std::fs::File,
        jobs: &[(u64, u64, u64)],
        threads: usize,
    ) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        for &(into, _, len) in jobs {
            self.slab.span(into, len)?;
        }
        let threads = threads.clamp(1, 16);
        let core = &self.slab.core;
        for &(into, from, len) in jobs {
            let mut at = 0u64;
            while at < len {
                let piece = (len - at).min(PREAD_CHUNK);
                let dst = into + at;
                let src = from + at;

                if !dst.is_multiple_of(4)
                    || piece < 4
                    || (!piece.is_multiple_of(4) && at + piece == len && piece < 8)
                {
                    let mut bytes = vec![0u8; piece as usize];
                    pread_jobs(file, &[(bytes.as_mut_ptr() as usize, src, piece)], 1)?;
                    self.slab.write(dst, &bytes)?;
                    at += piece;
                    continue;
                }
                let whole = piece & !3;
                let size = std::num::NonZeroU64::new(whole).expect("at least 4");
                {
                    let mut view = core
                        .queue
                        .write_buffer_with(&self.slab.buffer, dst, size)
                        .ok_or_else(|| core.fault("write_buffer_with", "no staging view"))?;
                    let base = view.slice(..).as_raw_ptr().as_ptr() as *mut u8 as usize;
                    let per = whole
                        .div_ceil(threads as u64)
                        .next_multiple_of(4096)
                        .max(4096);
                    let mut split = Vec::new();
                    let mut off = 0u64;
                    while off < whole {
                        let n = (whole - off).min(per);
                        split.push((base + off as usize, src + off, n));
                        off += n;
                    }
                    pread_jobs(file, &split, threads)?;
                }
                if whole < piece {
                    let tail = piece - whole;
                    let mut bytes = vec![0u8; tail as usize];
                    pread_jobs(file, &[(bytes.as_mut_ptr() as usize, src + whole, tail)], 1)?;
                    self.slab.write(dst + whole, &bytes)?;
                }
                at += piece;
            }
        }
        core.take_error("write_buffer_with")
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn pread_jobs(file: &std::fs::File, jobs: &[(usize, u64, u64)], threads: usize) -> Result<()> {
    let threads = threads.clamp(1, jobs.len().max(1));
    let per = jobs.len().div_ceil(threads).max(1);
    let failed: std::sync::Mutex<Option<Fault>> = std::sync::Mutex::new(None);
    std::thread::scope(|scope| {
        for chunk in jobs.chunks(per) {
            let failed = &failed;
            scope.spawn(move || {
                for &(dst, from, len) in chunk {
                    if let Err(why) = unsafe { pread_all(file, dst as *mut u8, from, len) } {
                        *failed
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(why);
                        return;
                    }
                }
            });
        }
    });
    match failed
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        Some(why) => Err(why),
        None => Ok(()),
    }
}

#[cfg(not(target_arch = "wasm32"))]
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
