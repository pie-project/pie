#![allow(unsafe_code)]
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(not(target_arch = "wasm32"))]
use engine_gpu::pread::pread_jobs;

use crate::encode::Mark;
use engine_gpu::Owner;
use engine_gpu::error::{Fault, Result};

use super::ctx::{Context, Core, note_reservation};
use super::host::Signal;

#[cfg(not(target_arch = "wasm32"))]
const PREAD_CHUNK: u64 = 256 << 20;

pub use engine_gpu::Memory;

static IDS: AtomicU64 = AtomicU64::new(1);

pub(crate) struct Raw {
    pub(crate) core: Arc<Core>,
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) size: u64,
    pub(crate) kind: Memory,
    pub(crate) mappable: bool,
    pub(crate) id: u64,
    /// Whose it is in the device's books; `None` for a buffer the host maps,
    /// which lies in host memory.
    owner: Option<Owner>,
}

pub(crate) type Slab = Arc<Raw>;

impl Raw {
    pub(crate) fn new(core: &Arc<Core>, bytes: u64, kind: Memory, owner: Owner) -> Result<Slab> {
        note_reservation();
        let size = bytes.max(4).next_multiple_of(4);
        if size > core.limits.max_buffer_size {
            return Err(Fault::Ceiling {
                what: "bytes of one device reservation",
                need: size,
                have: core.limits.max_buffer_size,
            });
        }
        let mappable = kind == Memory::Readback;
        let mut usage = if mappable {
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
        } else {
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
        };
        // The pool's segments hold guest configs a stage binds as a uniform
        // (`guest::session`), beside everything bound as storage.
        if owner == engine_gpu::books::owners::POOL && kind == Memory::Device {
            usage |= wgpu::BufferUsages::UNIFORM;
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
                    have: core.device_local.saturating_sub(core.books.device()),
                },
                other => other,
            });
        }
        let owner = (!mappable).then_some(owner);
        if let Some(owner) = owner {
            core.books.take(owner, size);
        }

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
            owner,
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
        if let Some(owner) = self.owner {
            self.core.books.give(owner, self.size);
        }
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
    pub fn zeroed(device: &Context, bytes: u64, owner: Owner) -> Result<Buffer> {
        Buffer::with(device, bytes, Memory::Device, owner)
    }

    pub fn host(device: &Context, bytes: u64, owner: Owner) -> Result<Buffer> {
        Buffer::with(device, bytes, Memory::Host, owner)
    }

    pub fn with(device: &Context, bytes: u64, kind: Memory, owner: Owner) -> Result<Buffer> {
        let slab = Raw::new(device.core(), bytes, kind, owner)?;
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

    /// [`Buffer::write`] through a shared reference: the queue orders it
    /// before the work submitted next.
    pub(crate) fn write_shared(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        self.span(offset, bytes.len() as u64)?;
        self.slab.write(offset, bytes)
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

impl engine_gpu::Buffer<super::Api> for Buffer {
    fn with(device: &Context, bytes: u64, kind: Memory, owner: Owner) -> Result<Buffer> {
        Buffer::with(device, bytes, kind, owner)
    }

    fn bytes(&self) -> u64 {
        self.bytes
    }

    fn slab_size(&self) -> u64 {
        self.slab.size
    }

    fn slab_id(&self) -> u64 {
        self.slab.id
    }

    fn is_host(&self) -> bool {
        Buffer::is_host(self)
    }

    fn mapped_ptr(&self) -> Option<usize> {
        None
    }

    fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        Buffer::write(self, offset, bytes)
    }

    fn write_shared(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        Buffer::write_shared(self, offset, bytes)
    }

    fn zero_span(&mut self, offset: u64, len: u64) -> Result<()> {
        Buffer::zero_span(self, offset, len)
    }

    fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        Buffer::read(self, offset, into)
    }

    fn write_from_file(
        &mut self,
        file: &std::fs::File,
        jobs: &[(u64, u64, u64)],
        threads: usize,
    ) -> Result<()> {
        Buffer::write_from_file(self, file, jobs, threads)
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
                    // SAFETY: the one job fills `bytes`, `piece` bytes long.
                    unsafe { pread_jobs(file, &[(bytes.as_mut_ptr() as usize, src, piece)], 1)? };
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
                    // SAFETY: the jobs tile `[base, base + whole)`, the staging
                    // view `view` maps and keeps alive until this scope ends.
                    unsafe { pread_jobs(file, &split, threads)? };
                }
                if whole < piece {
                    let tail = piece - whole;
                    let mut bytes = vec![0u8; tail as usize];
                    // SAFETY: the one job fills `bytes`, `tail` bytes long.
                    unsafe {
                        pread_jobs(file, &[(bytes.as_mut_ptr() as usize, src + whole, tail)], 1)?;
                    }
                    self.slab.write(dst + whole, &bytes)?;
                }
                at += piece;
            }
        }
        core.take_error("write_buffer_with")
    }
}
