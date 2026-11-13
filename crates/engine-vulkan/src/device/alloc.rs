#![allow(unsafe_code)]
use std::ptr::NonNull;
use std::sync::Arc;

use ash::vk;
use engine_gpu::pread::pread_jobs;

use engine_gpu::error::{Fault, Result};

use super::ctx::{Context, Core, STAGING_BYTES, note_reservation};

pub use engine_gpu::Memory;

pub(crate) struct Raw {
    pub(crate) core: Arc<Core>,
    pub(crate) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub(crate) size: u64,
    /// What the driver took for it (its requirement, at least `size`):
    /// what `Core::allocated` counts.
    charged: u64,
    pub(crate) mapped: Option<NonNull<u8>>,
    pub(crate) kind: Memory,
}

// SAFETY: `Raw` owns its buffer, memory and mapping outright; the device
// it names lives in the shared `Core`, and Vulkan objects are not bound to
// the thread that created them, so dropping it on another thread is fine.
unsafe impl Send for Raw {}
// SAFETY: needed because slabs travel as `Arc<Raw>` to completion and
// staging threads. Through `&Raw` the handles and sizes are only read, and
// every device-side copy is recorded and submitted under `core.transfer`'s
// lock. The host copies through `mapped` (`write`, `read`, `zero`) are not
// synchronized here: callers keep concurrent writes to one slab on disjoint
// ranges (each tensor owns its span), which this type does not check.
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
            Memory::Host | Memory::Staging | Memory::Readback => {
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
        // SAFETY: `d` is the live device `core` owns, and `info` is a valid
        // create info for a nonzero size.
        let buffer =
            unsafe { d.create_buffer(&info, None) }.map_err(|e| core.fault("vkCreateBuffer", e))?;
        // SAFETY: `buffer` was just created on `d`.
        let need = unsafe { d.get_buffer_memory_requirements(buffer) };
        let Some(index) = Self::memory_type(core, &need, kind) else {
            // SAFETY: `buffer` is ours, unbound and unused by any command.
            unsafe { d.destroy_buffer(buffer, None) };
            return Err(Fault::NoDevice {
                detail: format!("no memory type serves a {kind:?} buffer"),
            });
        };
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(need.size)
            .memory_type_index(index);
        // SAFETY: `index` is a memory type of this device that `need` allows,
        // and the size is the buffer's own requirement.
        let memory = match unsafe { d.allocate_memory(&alloc, None) } {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: `buffer` is ours, unbound and unused by any command.
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
        // SAFETY: `memory` was sized and typed from `buffer`'s requirements,
        // offset 0 meets any alignment, and neither is bound yet.
        if let Err(e) = unsafe { d.bind_buffer_memory(buffer, memory, 0) } {
            // SAFETY: both objects are ours and no command has used them.
            unsafe {
                d.free_memory(memory, None);
                d.destroy_buffer(buffer, None);
            }
            return Err(core.fault("vkBindBufferMemory", e));
        }
        let mapped = if kind == Memory::Device {
            None
        } else {
            // SAFETY: `memory` is a host-visible allocation of ours (the
            // non-device kinds only pick host-visible types), not yet mapped.
            match unsafe { d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) } {
                Ok(p) => NonNull::new(p.cast::<u8>()),
                Err(e) => {
                    // SAFETY: both objects are ours and no command has used them.
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
            charged: need.size,
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
            // SAFETY: `span` checked `offset + len <= size`, and the mapping
            // covers the whole allocation, which is at least `size` bytes; a
            // caller's slice cannot alias device-mapped memory.
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
        let mut transfer = core.transfer.lock().expect("transfer lock poisoned");
        let staging = Arc::clone(core.staging(&mut transfer)?);
        let base = staging.mapped.expect("staging is mapped");
        let mut at = 0usize;
        while at < bytes.len() {
            let len = (bytes.len() - at).min(STAGING_BYTES as usize);
            // SAFETY: `len` fits both the rest of `bytes` and the mapped
            // staging slab of `STAGING_BYTES`; the held transfer lock keeps
            // every other user off the staging slab, and the previous
            // `submit_once` waited for the copy that read it.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr().add(at), base.as_ptr(), len);
            }
            let region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(offset + at as u64)
                .size(len as u64);
            // SAFETY: `cmd` is the transfer command buffer in recording state;
            // both buffers are live slabs of this device and the region lies
            // inside each (`span` bounds the destination).
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
            // SAFETY: `span` checked `offset + len <= size` inside the mapping,
            // and `into` is a caller's exclusive slice of that length.
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
        let mut transfer = core.transfer.lock().expect("transfer lock poisoned");
        let staging = Arc::clone(core.staging(&mut transfer)?);
        let base = staging.mapped.expect("staging is mapped");
        let mut at = 0usize;
        while at < into.len() {
            let len = (into.len() - at).min(STAGING_BYTES as usize);
            let region = vk::BufferCopy::default()
                .src_offset(offset + at as u64)
                .dst_offset(0)
                .size(len as u64);
            // SAFETY: `cmd` is the transfer command buffer in recording state;
            // both buffers are live slabs of this device and the region lies
            // inside each (`span` bounds the source, `len <= STAGING_BYTES`).
            core.submit_once(&transfer, |d, cmd| unsafe {
                d.cmd_copy_buffer(cmd, self.buffer, staging.buffer, &[region]);
            })?;
            // SAFETY: `submit_once` waited on the copy's fence, so the staging
            // bytes are written; the transfer lock keeps others off them, and
            // `len` fits both the staging slab and the rest of `into`.
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
            // SAFETY: `span` checked `offset + len <= size` inside the mapping.
            unsafe {
                std::ptr::write_bytes(base.as_ptr().add(offset as usize), 0, len as usize);
            }
            return Ok(());
        }

        let core = &self.core;
        let transfer = core.transfer.lock().expect("transfer lock poisoned");
        let start = offset & !3;
        let end = (offset + len).next_multiple_of(4).min(self.size);
        // SAFETY: `cmd` is the transfer command buffer in recording state;
        // `start` and `end` are 4-aligned (`size` is a multiple of 4) and
        // inside this live buffer, as `vkCmdFillBuffer` requires.
        core.submit_once(&transfer, |d, cmd| unsafe {
            d.cmd_fill_buffer(cmd, self.buffer, start, end - start, 0);
        })
    }
}

impl Drop for Raw {
    fn drop(&mut self) {
        let d = &self.core.device;
        // SAFETY: this is the last `Arc` of the slab, and every frame that
        // bound it retains its own `Arc` until its fence signals, so no
        // pending command uses the buffer; `core` keeps the device alive.
        unsafe {
            if self.mapped.is_some() {
                d.unmap_memory(self.memory);
            }
            d.destroy_buffer(self.buffer, None);
            d.free_memory(self.memory, None);
        }
        self.core
            .allocated
            .fetch_sub(self.charged, std::sync::atomic::Ordering::Relaxed);
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

impl engine_gpu::Buffer<super::Api> for Buffer {
    fn with(device: &Context, bytes: u64, kind: Memory) -> Result<Buffer> {
        Buffer::with(device, bytes, kind)
    }

    fn bytes(&self) -> u64 {
        self.bytes
    }

    fn slab_size(&self) -> u64 {
        self.slab.size
    }

    fn slab_id(&self) -> u64 {
        {
            use ash::vk::Handle;
            self.slab.buffer.as_raw()
        }
    }

    fn is_host(&self) -> bool {
        Buffer::is_host(self)
    }

    fn mapped_ptr(&self) -> Option<usize> {
        Buffer::mapped_ptr(self)
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
        if let Some(base) = self.slab.mapped {
            let base = base.as_ptr() as usize;
            let jobs: Vec<(usize, u64, u64)> = jobs
                .iter()
                .map(|&(into, from, len)| (base + into as usize, from, len))
                .collect();
            // SAFETY: each job's `dst..dst + len` lies inside this slab's
            // mapping (`span` checked every job above), held for the call;
            // the jobs are the caller's and are not checked for overlap.
            return unsafe { pread_jobs(file, &jobs, threads) };
        }
        let core = &self.slab.core;
        let mut transfer = core.transfer.lock().expect("transfer lock poisoned");
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
            // SAFETY: each window slot lies inside the staging slab
            // (`slot + len <= STAGING_BYTES`), held under the transfer lock,
            // and the slots are disjoint.
            unsafe { pread_jobs(file, &window, threads)? };
            // SAFETY: `cmd` is the transfer command buffer in recording state;
            // each region's source sits inside the staging slab (`slot + len
            // <= STAGING_BYTES`) and its destination inside this slab (`span`
            // checked every job above).
            core.submit_once(&transfer, |d, cmd| unsafe {
                d.cmd_copy_buffer(cmd, staging.buffer, self.slab.buffer, &regions);
            })?;
        }
        Ok(())
    }
}
