use std::ptr::NonNull;
use std::sync::Arc;

use ash::vk;

use crate::error::{Fault, Result};

use super::ctx::{Context, Core, STAGING_BYTES, note_reservation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Memory {
    Device,

    Host,

    Staging,
}

pub(crate) struct Raw {
    pub(crate) core: Arc<Core>,
    pub(crate) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub(crate) size: u64,
    pub(crate) mapped: Option<NonNull<u8>>,
    pub(crate) kind: Memory,
}

unsafe impl Send for Raw {}
unsafe impl Sync for Raw {}

pub(crate) type Slab = Arc<Raw>;

impl Raw {
    fn memory_type(core: &Core, need: &vk::MemoryRequirements, kind: Memory) -> Option<u32> {
        let types = &core.memory.memory_types[..core.memory.memory_type_count as usize];
        let prefers = |flags: vk::MemoryPropertyFlags| {
            (0..core.memory.memory_type_count).find(|&i| {
                need.memory_type_bits & (1 << i) != 0
                    && types[i as usize].property_flags.contains(flags)
            })
        };
        use vk::MemoryPropertyFlags as F;
        match kind {
            Memory::Device => prefers(F::DEVICE_LOCAL).or_else(|| prefers(F::empty())),
            Memory::Host | Memory::Staging => {
                prefers(F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED)
                    .or_else(|| prefers(F::HOST_VISIBLE | F::HOST_COHERENT))
                    .or_else(|| prefers(F::HOST_VISIBLE))
            }
        }
    }

    pub(crate) fn new(core: &Arc<Core>, bytes: u64, kind: Memory) -> Result<Slab> {
        note_reservation();
        let size = bytes.max(4).next_multiple_of(4);
        let d = &core.device;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer =
            unsafe { d.create_buffer(&info, None) }.map_err(|e| core.fault("vkCreateBuffer", e))?;
        let need = unsafe { d.get_buffer_memory_requirements(buffer) };
        let Some(index) = Self::memory_type(core, &need, kind) else {
            unsafe { d.destroy_buffer(buffer, None) };
            return Err(Fault::NoDevice {
                detail: format!("no memory type serves a {kind:?} buffer"),
            });
        };
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(need.size)
            .memory_type_index(index);
        let memory = match unsafe { d.allocate_memory(&alloc, None) } {
            Ok(m) => m,
            Err(e) => {
                unsafe { d.destroy_buffer(buffer, None) };
                if e == vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                    || e == vk::Result::ERROR_OUT_OF_HOST_MEMORY
                {
                    return Err(Fault::Ceiling {
                        what: if kind == Memory::Device {
                            "device memory"
                        } else {
                            "host-visible memory"
                        },
                        need: need.size,
                        have: core.device_local.saturating_sub(
                            core.allocated.load(std::sync::atomic::Ordering::Relaxed),
                        ),
                    });
                }
                return Err(core.fault("vkAllocateMemory", e));
            }
        };
        if let Err(e) = unsafe { d.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                d.free_memory(memory, None);
                d.destroy_buffer(buffer, None);
            }
            return Err(core.fault("vkBindBufferMemory", e));
        }
        let mapped = if kind == Memory::Device {
            None
        } else {
            match unsafe { d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) } {
                Ok(p) => NonNull::new(p.cast::<u8>()),
                Err(e) => {
                    unsafe {
                        d.free_memory(memory, None);
                        d.destroy_buffer(buffer, None);
                    }
                    return Err(core.fault("vkMapMemory", e));
                }
            }
        };
        core.allocated
            .fetch_add(need.size, std::sync::atomic::Ordering::Relaxed);
        Ok(Arc::new(Raw {
            core: Arc::clone(core),
            buffer,
            memory,
            size,
            mapped,
            kind,
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
        if let Some(base) = self.mapped {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    base.as_ptr().add(offset as usize),
                    bytes.len(),
                );
            }
            return Ok(());
        }
        let core = &self.core;
        let mut transfer = core
            .transfer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let staging = Arc::clone(core.staging(&mut transfer)?);
        let base = staging.mapped.expect("staging is mapped");
        let mut at = 0usize;
        while at < bytes.len() {
            let len = (bytes.len() - at).min(STAGING_BYTES as usize);
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr().add(at), base.as_ptr(), len);
            }
            let region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(offset + at as u64)
                .size(len as u64);
            core.submit_once(&transfer, |d, cmd| unsafe {
                d.cmd_copy_buffer(cmd, staging.buffer, self.buffer, &[region]);
            })?;
            at += len;
        }
        Ok(())
    }

    pub(crate) fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        self.span(offset, into.len() as u64)?;
        if into.is_empty() {
            return Ok(());
        }
        if let Some(base) = self.mapped {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    base.as_ptr().add(offset as usize),
                    into.as_mut_ptr(),
                    into.len(),
                );
            }
            return Ok(());
        }
        let core = &self.core;
        let mut transfer = core
            .transfer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let staging = Arc::clone(core.staging(&mut transfer)?);
        let base = staging.mapped.expect("staging is mapped");
        let mut at = 0usize;
        while at < into.len() {
            let len = (into.len() - at).min(STAGING_BYTES as usize);
            let region = vk::BufferCopy::default()
                .src_offset(offset + at as u64)
                .dst_offset(0)
                .size(len as u64);
            core.submit_once(&transfer, |d, cmd| unsafe {
                d.cmd_copy_buffer(cmd, self.buffer, staging.buffer, &[region]);
            })?;
            unsafe {
                std::ptr::copy_nonoverlapping(base.as_ptr(), into.as_mut_ptr().add(at), len);
            }
            at += len;
        }
        Ok(())
    }

    pub(crate) fn zero(&self, offset: u64, len: u64) -> Result<()> {
        self.span(offset, len)?;
        if len == 0 {
            return Ok(());
        }
        if let Some(base) = self.mapped {
            unsafe {
                std::ptr::write_bytes(base.as_ptr().add(offset as usize), 0, len as usize);
            }
            return Ok(());
        }

        let core = &self.core;
        let transfer = core
            .transfer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let start = offset & !3;
        let end = (offset + len).next_multiple_of(4).min(self.size);
        core.submit_once(&transfer, |d, cmd| unsafe {
            d.cmd_fill_buffer(cmd, self.buffer, start, end - start, 0);
        })
    }
}

impl Drop for Raw {
    fn drop(&mut self) {
        let d = &self.core.device;
        unsafe {
            if self.mapped.is_some() {
                d.unmap_memory(self.memory);
            }
            d.destroy_buffer(self.buffer, None);
            d.free_memory(self.memory, None);
        }
        self.core
            .allocated
            .fetch_sub(self.size, std::sync::atomic::Ordering::Relaxed);
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
        let buffer = Buffer { slab, bytes };
        buffer.slab.zero(0, buffer.slab.size)?;
        Ok(buffer)
    }

    #[must_use]
    pub fn is_mapped(&self) -> bool {
        false
    }

    #[must_use]
    pub fn is_host(&self) -> bool {
        self.slab.mapped.is_some()
    }

    #[must_use]
    pub(crate) fn mapped_ptr(&self) -> Option<usize> {
        self.slab.mapped.map(|p| p.as_ptr() as usize)
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
        self.slab.write(offset, bytes)
    }

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
        self.slab.read(offset, into)
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
        if let Some(base) = self.slab.mapped {
            let base = base.as_ptr() as usize;
            let jobs: Vec<(usize, u64, u64)> = jobs
                .iter()
                .map(|&(into, from, len)| (base + into as usize, from, len))
                .collect();
            return pread_jobs(fd, &jobs, threads);
        }
        let core = &self.slab.core;
        let mut transfer = core
            .transfer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let staging = Arc::clone(core.staging(&mut transfer)?);
        let base = staging.mapped.expect("staging is mapped").as_ptr() as usize;

        let mut pieces: Vec<(u64, u64, u64)> = Vec::new();
        for &(into, from, len) in jobs {
            let mut at = 0u64;
            while at < len {
                let piece = (len - at).min(STAGING_BYTES);
                pieces.push((into + at, from + at, piece));
                at += piece;
            }
        }
        let mut i = 0usize;
        while i < pieces.len() {
            let mut used = 0u64;
            let mut window: Vec<(usize, u64, u64)> = Vec::new();
            let mut regions: Vec<vk::BufferCopy> = Vec::new();
            while i < pieces.len() {
                let (into, from, len) = pieces[i];
                let slot = used.next_multiple_of(4);
                if slot + len > STAGING_BYTES {
                    break;
                }
                window.push((base + slot as usize, from, len));
                regions.push(
                    vk::BufferCopy::default()
                        .src_offset(slot)
                        .dst_offset(into)
                        .size(len),
                );
                used = slot + len;
                i += 1;
            }
            pread_jobs(fd, &window, threads)?;
            core.submit_once(&transfer, |d, cmd| unsafe {
                d.cmd_copy_buffer(cmd, staging.buffer, self.slab.buffer, &regions);
            })?;
        }
        Ok(())
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
