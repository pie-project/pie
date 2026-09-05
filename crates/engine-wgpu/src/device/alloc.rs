use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{Fault, Result};

use super::ctx::{Context, Core, STAGING_BYTES, note_reservation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Memory {
    Device,

    Host,

    Staging,
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
        let mappable = kind != Memory::Device && core.enabled.mappable_primary;
        let mut usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        if mappable {
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

        let core = &self.core;
        let mut guard = core
            .staging
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let staging = core.staging(&mut guard);
        let mut at = 0u64;
        let total = into.len() as u64;
        while at < total {
            let want = (total - at).min(STAGING_BYTES - 8);
            let from = offset + at;
            let start = from & !3;
            let end = (from + want).next_multiple_of(4).min(self.size);
            let len = end - start;
            let mut encoder = core
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pie read"),
                });
            encoder.copy_buffer_to_buffer(&self.buffer, start, staging, 0, Some(len));

            let t0 = std::time::Instant::now();
            let index = core.queue.submit(std::iter::once(encoder.finish()));
            crate::encode::record_read_phase(0, t0.elapsed().as_nanos() as u64);
            let (tx, rx) = std::sync::mpsc::channel();
            staging
                .slice(0..len)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send(result);
                });
            let t1 = std::time::Instant::now();
            core.wait_for(&index)?;
            crate::encode::record_read_phase(1, t1.elapsed().as_nanos() as u64);
            let t2 = std::time::Instant::now();
            let mapped = loop {
                match rx.try_recv() {
                    Ok(result) => break result,
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        core.device
                            .poll(wgpu::PollType::Poll)
                            .map_err(|e| core.fault("Device::poll", e))?;
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        return Err(core.fault("map_async", "the map callback never ran"));
                    }
                }
            };
            mapped.map_err(|e| core.fault("map_async", e))?;
            crate::encode::record_read_phase(2, t2.elapsed().as_nanos() as u64);
            {
                let view = staging
                    .slice(0..len)
                    .get_mapped_range()
                    .map_err(|e| core.fault("get_mapped_range", e))?;
                let skip = (from - start) as usize;
                let take = (end - from).min(want) as usize;
                into[at as usize..at as usize + take].copy_from_slice(&view[skip..skip + take]);
                at += take as u64;
            }
            staging.unmap();
        }
        Ok(())
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
        let started = std::time::Instant::now();
        let out = self.slab.write(offset, bytes);
        crate::encode::record_io(true, started.elapsed().as_nanos() as u64);
        out
    }

    pub fn zero_span(&mut self, offset: u64, len: u64) -> Result<()> {
        self.span(offset, len)?;
        self.slab.zero(offset, len)
    }

    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        self.span(offset, into.len() as u64)?;
        let started = std::time::Instant::now();
        let out = self.slab.read(offset, into);
        crate::encode::record_io(false, started.elapsed().as_nanos() as u64);
        out
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
        use std::os::fd::AsRawFd;
        let fd = file.as_raw_fd();
        let threads = threads.clamp(1, 16);
        let core = &self.slab.core;
        for &(into, from, len) in jobs {
            let mut at = 0u64;
            while at < len {
                let piece = (len - at).min(STAGING_BYTES);
                let dst = into + at;
                let src = from + at;

                if !dst.is_multiple_of(4)
                    || piece < 4
                    || (!piece.is_multiple_of(4) && at + piece == len && piece < 8)
                {
                    let mut bytes = vec![0u8; piece as usize];
                    pread_jobs(fd, &[(bytes.as_mut_ptr() as usize, src, piece)], 1)?;
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
                    pread_jobs(fd, &split, threads)?;
                }
                if whole < piece {
                    let tail = piece - whole;
                    let mut bytes = vec![0u8; tail as usize];
                    pread_jobs(fd, &[(bytes.as_mut_ptr() as usize, src + whole, tail)], 1)?;
                    self.slab.write(dst + whole, &bytes)?;
                }
                at += piece;
            }
        }
        core.take_error("write_buffer_with")
    }
}

fn pread_jobs(fd: i32, jobs: &[(usize, u64, u64)], threads: usize) -> Result<()> {
    let threads = threads.clamp(1, jobs.len().max(1));
    let per = jobs.len().div_ceil(threads).max(1);
    let failed: std::sync::Mutex<Option<Fault>> = std::sync::Mutex::new(None);
    std::thread::scope(|scope| {
        for chunk in jobs.chunks(per) {
            let failed = &failed;
            scope.spawn(move || {
                for &(dst, from, len) in chunk {
                    if let Err(why) = unsafe { pread_all(fd, dst as *mut u8, from, len) } {
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

unsafe fn pread_all(fd: i32, dst: *mut u8, from: u64, len: u64) -> Result<()> {
    let mut done = 0u64;
    while done < len {
        let want = usize::try_from(len - done)
            .unwrap_or(usize::MAX)
            .min(1 << 30);
        let got = unsafe {
            libc::pread(
                fd,
                dst.add(done as usize).cast::<libc::c_void>(),
                want,
                libc::off_t::try_from(from + done).unwrap_or(libc::off_t::MAX),
            )
        };
        if got < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(Fault::Device {
                call: "pread",
                why: err.to_string(),
            });
        }
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
