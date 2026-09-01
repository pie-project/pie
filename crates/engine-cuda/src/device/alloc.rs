//! One owned device allocation, and the two transfers a shell needs.
//!
//! **NOT AN ALLOCATOR.** A shell makes a handful of allocations in its whole
//! life — the arena, the weight store, one pool per cache space, the resident
//! fire inputs, the plan workspace — and every one of them is sized once from
//! a budget and lives until the model is unloaded. What varies per fire is
//! which OFFSETS a kernel is pointed at, and that arithmetic belongs to the
//! compiler's carve (`ArenaMap`), not to a heap. So this is `cudaMalloc`,
//! `cudaFree`, and the memcpys, with a length on the front of each.
//!
//! **Bounds are checked here and nowhere else.** A device pointer is a `u64`
//! by the time it reaches [`Tensor`](kernels_cuda::Tensor), and past that
//! point an off-by-one does not fault — it writes into a neighbour's
//! rectangle and the model produces slightly wrong numbers forever. Every
//! offset therefore meets its length at this door.

use crate::error::{Fault, Result};

/// A device allocation of a stated size.
#[derive(Debug)]
pub struct Buffer {
    ptr: u64,
    bytes: usize,
}

/// A failed allocation, re-said with numbers when it was a shortfall.
///
/// **THE ONE STATUS WORTH TRANSLATING, AND ONLY IT.** `cudaErrorMemoryAllocation`
/// is a fact about a size and a device, and both numbers are askable; every
/// other status is a fact about the runtime and is left as the call and the
/// code. [`Fault::Ceiling`] is deliberately not what comes back: that one is a
/// fire wanting more than the load reserved, and this is a load wanting more
/// than the card has. The free figure is what `cudaMemGetInfo` says AFTER the failure,
/// which is the honest one — it is what was actually available to the ask —
/// and a query that itself fails leaves the original fault alone rather than
/// replacing a true sentence with a guess.
#[cfg(feature = "_cuda")]
fn out_of_room(fault: Fault, bytes: usize) -> Fault {
    use cudarc::runtime::sys as rt;

    let shortfall = matches!(
        fault,
        Fault::Device { code, .. } if code == rt::cudaError::cudaErrorMemoryAllocation as i32
    );
    if !shortfall {
        return fault;
    }
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: two live locals; the call only writes them.
    let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
    if asked != rt::cudaError::cudaSuccess {
        return fault;
    }
    Fault::OutOfMemory {
        need: bytes as u64,
        have: free as u64,
    }
}

impl Buffer {
    /// Allocate `bytes` and zero them.
    ///
    /// ZEROED, NOT FRESH. `cudaMalloc` hands back whatever the last tenant
    /// left, and the places this crate allocates are exactly the places where
    /// unwritten bytes are read as numbers: a recurrent state slab at slot
    /// open, an arena rectangle a kernel writes only part of, a page table
    /// past this fire's lanes. The loader makes the same call for the same
    /// reason and says so.
    ///
    /// A zero-byte request is a real answer — a plan with no recurrent
    /// caches wants no recurrent pool — and comes back as a null handle no
    /// launch is pointed at.
    ///
    /// **AND A REFUSAL THAT DOES NOT FIT CARRIES BOTH NUMBERS** (palo C3b).
    /// Every allocation this crate makes is sized once from a budget, so the
    /// interesting failure is always the same one — this model does not fit
    /// this device — and `cudaMalloc answered 2` is the least useful sentence
    /// there is about it. `cudaErrorMemoryAllocation` is therefore turned into
    /// a [`Fault::Ceiling`] with the ask and the free, which is the shape a
    /// ceiling refusal already has everywhere else in this shell and which a
    /// caller can act on: a 27B checkpoint against a 46 GiB card says so in
    /// gibibytes instead of in an errno. Every other CUDA status stays what it
    /// was.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a request the device has no room for,
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn zeroed(bytes: usize) -> Result<Buffer> {
        if bytes == 0 {
            return Ok(Buffer { ptr: 0, bytes: 0 });
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            let mut base: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `base` is a live local; the allocation is this
            // buffer's, freed exactly once in `Drop`.
            let allocated =
                unsafe { crate::device::ctx::check("cudaMalloc", rt::cudaMalloc(&raw mut base, bytes)) };
            if let Err(fault) = allocated {
                return Err(out_of_room(fault, bytes));
            }
            // SAFETY: `base` is the allocation just made, of `bytes` bytes.
            unsafe {
                crate::device::ctx::check("cudaMemset", rt::cudaMemset(base, 0, bytes))?;
            }
            Ok(Buffer {
                ptr: base as u64,
                bytes,
            })
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// The base address.
    #[must_use]
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// How many bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The address `offset` bytes in, checked against the length.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for an offset past the end.
    pub fn at(&self, offset: u64) -> Result<u64> {
        if offset > self.bytes as u64 {
            return Err(Fault::Ceiling {
                what: "bytes into a device buffer",
                need: offset,
                have: self.bytes as u64,
            });
        }
        Ok(self.ptr + offset)
    }

    /// Copy `bytes` in at `offset`, and wait.
    ///
    /// The load path's transfer: 260 weight planes, once, in front of a fire
    /// that has not started. Synchronous on purpose — there is nothing to
    /// overlap it with, and a load that returns before its bytes have landed
    /// is a first fire that reads poison.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a write past the end, [`Fault::Device`] for the
    /// copy.
    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        self.span(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the span was just checked against this allocation, and
            // `bytes` is a live host slice for the duration of a synchronous
            // copy.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemcpy",
                    rt::cudaMemcpy(
                        (self.ptr + offset) as *mut core::ffi::c_void,
                        bytes.as_ptr().cast(),
                        bytes.len(),
                        rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    ),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// Zero `len` bytes at `offset`, and wait.
    ///
    /// The same `cudaMemset` [`Buffer::zeroed`] makes at allocation, aimed at
    /// one span rather than the whole buffer. It exists because a recurrent
    /// slot has to be re-zeroed every time a fresh sequence takes it
    /// (`Pools::clear`), and staging ten megabytes of host zeros across PCIe
    /// to say "nothing" is the wrong shape for something on the fire path.
    ///
    /// Synchronous, like [`Buffer::write`] and for the same reason: the
    /// caller is about to launch kernels that read these bytes.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a span past the end, [`Fault::Device`] for the
    /// fill.
    pub fn zero_span(&mut self, offset: u64, len: usize) -> Result<()> {
        self.span(offset, len)?;
        if len == 0 {
            return Ok(());
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the span was just checked against this allocation.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemset",
                    rt::cudaMemset((self.ptr + offset) as *mut core::ffi::c_void, 0, len),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// Copy `bytes` in at `offset` on `stream`, ordered with the fire.
    ///
    /// The fire path's transfer: the descriptor's derived vectors — tokens,
    /// positions, the indptr, the page tables — written in front of launches
    /// that read them. It must be ON THE STREAM, not synchronous: a
    /// synchronous copy is ordered against every stream in the process, which
    /// is both slower and a lie about what this fire depends on.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] or [`Fault::Device`].
    pub fn stage(&mut self, stream: *mut core::ffi::c_void, offset: u64, bytes: &[u8]) -> Result<()> {
        self.span(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the span is checked; `bytes` outlives the enqueue, and
            // the caller synchronizes the stream before it is dropped —
            // every caller in this crate stages inside one `fire`.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemcpyAsync",
                    rt::cudaMemcpyAsync(
                        (self.ptr + offset) as *mut core::ffi::c_void,
                        bytes.as_ptr().cast(),
                        bytes.len(),
                        rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                        stream.cast(),
                    ),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    /// Copy `len` bytes from a HOST address at `offset` on `stream`.
    ///
    /// [`Buffer::stage`]'s twin, and the only difference is where the source
    /// lives — which is the whole of what the staging ring changed. `stage`
    /// takes a `&[u8]` that is almost always a `Vec` the fire built: pageable
    /// memory, so the driver copies it into its own staging buffer before
    /// `cudaMemcpyAsync` returns, which makes the call synchronous in the
    /// source and is why a single-buffered `Inputs` was survivable at depth 1.
    /// This one's source is a claimed ring slot's PINNED bytes: the copy is
    /// genuinely asynchronous, the host must not touch those bytes again until
    /// the GPU has passed it, and the slot's lifetime — claimed at prepare,
    /// released by the step's settlement callback — is exactly that bound.
    ///
    /// # Safety
    ///
    /// `src .. src + len` must be a live, page-locked host allocation that
    /// stays unwritten until the copy completes on `stream`.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a span past this allocation, [`Fault::Device`]
    /// for the copy.
    pub unsafe fn stage_from(
        &mut self,
        stream: *mut core::ffi::c_void,
        offset: u64,
        src: *const u8,
        len: usize,
    ) -> Result<()> {
        self.span(offset, len)?;
        if len == 0 {
            return Ok(());
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the destination span is checked; the source is the
            // caller's promise above.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemcpyAsync",
                    rt::cudaMemcpyAsync(
                        (self.ptr + offset) as *mut core::ffi::c_void,
                        src.cast(),
                        len,
                        rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                        stream.cast(),
                    ),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = (stream, src);
            Err(Fault::Runtimeless)
        }
    }

    /// Zero a span on `stream`, ordered with the fire.
    ///
    /// [`Buffer::zero_span`]'s asynchronous twin, and the fire path wants this
    /// one for the reason F2b made sharp: `cudaMemset` is SYNCHRONOUS, so the
    /// fresh-slot clear that begins a sequence drained everything airborne —
    /// a host wait between two waves, which is exactly what article 2
    /// forbids. On the stream it is what it always meant to be: a zeroing
    /// ordered in front of the launches that read the bank, costing the
    /// pipeline nothing.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a span past this allocation, [`Fault::Device`]
    /// for the memset.
    pub fn zero_span_on(
        &mut self,
        stream: *mut core::ffi::c_void,
        offset: u64,
        len: usize,
    ) -> Result<()> {
        self.span(offset, len)?;
        if len == 0 {
            return Ok(());
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the span was just checked against this allocation, and
            // the caller keeps it alive past the enqueue.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemsetAsync",
                    rt::cudaMemsetAsync(
                        (self.ptr + offset) as *mut core::ffi::c_void,
                        0,
                        len,
                        stream.cast(),
                    ),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    /// Zero the whole allocation on `stream`, ordered with the fire.
    ///
    /// THE FIRE-PATH COUNTERPART OF [`Buffer::zeroed`], and the guest-program
    /// plane is what wants it: an ETA fire's scratch, its per-channel pending
    /// flags and its commit slot are re-used allocation-for-allocation across
    /// fires (nothing allocates on the fire path), so each fire has to start
    /// from the state the emitted kernels assume — zeros — rather than from
    /// what the previous fire left. Synchronous zeroing would be ordered
    /// against every stream in the process, which is both slower and a lie
    /// about what this fire depends on.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn clear(&mut self, stream: *mut core::ffi::c_void) -> Result<()> {
        if self.bytes == 0 {
            return Ok(());
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the span is this allocation's own, and the caller
            // synchronizes the stream before the buffer is dropped.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemsetAsync",
                    rt::cudaMemsetAsync(
                        self.ptr as *mut core::ffi::c_void,
                        0,
                        self.bytes,
                        stream.cast(),
                    ),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    /// Copy `into.len()` bytes out from `offset`, and wait.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] or [`Fault::Device`].
    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        self.span(offset, into.len())?;
        if into.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the span is checked and `into` is a live host slice.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemcpy",
                    rt::cudaMemcpy(
                        into.as_mut_ptr().cast(),
                        (self.ptr + offset) as *const core::ffi::c_void,
                        into.len(),
                        rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    ),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    fn span(&self, offset: u64, len: usize) -> Result<()> {
        let end = offset.saturating_add(len as u64);
        if end > self.bytes as u64 {
            return Err(Fault::Ceiling {
                what: "bytes of a device buffer",
                need: end,
                have: self.bytes as u64,
            });
        }
        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        #[cfg(feature = "_cuda")]
        if self.ptr != 0 {
            // SAFETY: the pointer came from this buffer's own `cudaMalloc`
            // and is freed exactly once.
            unsafe {
                let _ = cudarc::runtime::sys::cudaFree(self.ptr as *mut core::ffi::c_void);
            }
        }
    }
}

/// **ONE MAPPED PINNED ALLOCATION: THE SAME BYTES ON BOTH SIDES.**
///
/// `cudaHostAlloc(.., cudaHostAllocMapped)` gives a host pointer the guest's
/// own thread stores through and a device address a kernel dereferences —
/// under UVA the same number, which `cudaHostGetDevicePointer` is asked rather
/// than assumed. That equality is the whole of alto's channel crossing
/// (survey §7, invariant I5): a guest's cell reaches the device because
/// `channel::pull_validate` READS it where the guest wrote it, and a pass's
/// cell reaches the guest because `channel::scatter_publish` WRITES it where
/// the guest will read it. Neither direction is a `cudaMemcpy`, and a full
/// guest round trip makes no CUDA call at all on the guest's thread.
///
/// **NOT A [`Buffer`], AND THE DIFFERENCE IS THE POINT.** A `Buffer`'s bytes
/// are device-only and reach the host through a copy; these are one
/// allocation two agents address concurrently. So there is no `write`/`read`
/// pair here: a caller takes the slice and stores into it.
#[derive(Debug)]
pub struct Pinned {
    host: *mut u8,
    device: u64,
    bytes: usize,
    /// Which door made these bytes, and therefore which one gives them back.
    /// The two are not interchangeable: `cudaFreeHost` on a mapping this
    /// process made itself is a fault, and `munmap` on a `cudaHostAlloc`
    /// leaks the driver's own bookkeeping.
    origin: Origin,
}

/// **HOW A [`Pinned`] CAME TO BE PAGE-LOCKED**, which is the whole of what
/// [`Pinned::drop`] needs to know.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Origin {
    /// `cudaHostAlloc` made the allocation and page-locked it in one call.
    Allocated,
    /// This process mapped the pages and `cudaHostRegister` locked them —
    /// [`Pinning`], and its doc says why an image ever takes that road.
    /// Unreachable without a runtime, where `Pinning::lock` is a refusal.
    #[cfg_attr(not(feature = "_cuda"), allow(dead_code))]
    Registered,
}

// SAFETY: a `Pinned` is an allocation and a length. What makes concurrent
// access sound is the SPSC discipline of the channel plane above it — the
// guest owns one control word and the engine the other, and neither writes
// the other's — not anything this type could enforce; moving the allocation
// between threads is no less sound than moving a `Box<[u8]>`.
unsafe impl Send for Pinned {}
// SAFETY: as `Send`; `&Pinned` hands out only the base addresses and the
// length, and the byte access below is `unsafe` at the caller's door.
unsafe impl Sync for Pinned {}

impl Pinned {
    /// Allocate `bytes` of zeroed mapped pinned memory.
    ///
    /// A zero-byte request answers a null allocation, which is what a
    /// channel with no host end wants: nothing to address and nothing to
    /// free.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] without a CUDA runtime, [`Fault::Device`] for
    /// whatever `cudaHostAlloc` said.
    pub fn mapped(bytes: usize) -> Result<Pinned> {
        Pinned::alloc(bytes, true)
    }

    /// **THE SAME ALLOCATION WITHOUT THE MEMSET** — for a caller that will
    /// overwrite every byte before it reads one (§K.4).
    ///
    /// [`Pinned::mapped`]'s `write_bytes(0)` is a serving cost nobody has
    /// named until now: the weight tier of a streamed deployment is TENS OF
    /// GIGABYTES of page-locked memory, and zeroing it costs tens of seconds
    /// of memory bandwidth on EVERY boot — including the warm one whose whole
    /// promise is that it does not touch the bytes twice. A tier restored
    /// from the artifact reads its whole image in over itself, so the zeros
    /// it is written on top of are pure loss.
    ///
    /// **WHAT THE CALLER OWES.** Every byte of this allocation is
    /// indeterminate until the caller writes it, and the doors that hand the
    /// bytes out ([`Pinned::read`], [`Pinned::view`]) do not know that. So a
    /// caller may take this only when it can point at the write that covers
    /// the WHOLE length — not the parts an index names, the whole length,
    /// padding and alignment gaps included — and only when that write happens
    /// before any read, any digest and any kernel launch. It is a safe
    /// signature for the reason [`Pinned::read`] is one: this type's
    /// discipline is stated and checked by its callers rather than by its
    /// borrows (see the `unsafe impl Sync` above), and a second convention
    /// here would not make the first one true.
    ///
    /// **The streamed COLD load is not such a caller and does not take it.**
    /// The landing sink writes each plane's published bytes; the tier seats
    /// each plane at `reserved`, which is those bytes rounded up to
    /// [`weights::ALIGN`](crate::weights::ALIGN) — so up to 255 bytes per
    /// plane, plus the gap that aligns the groups behind the banks, are never
    /// written by anything. Zeroed, they are a deterministic image the
    /// artifact can carry; unzeroed, they are whatever the kernel handed out,
    /// which a write path would hash and publish.
    ///
    /// # Errors
    ///
    /// As [`Pinned::mapped`].
    pub fn mapped_uninit(bytes: usize) -> Result<Pinned> {
        Pinned::alloc(bytes, false)
    }

    /// **The memset [`Pinned::mapped_uninit`] did not do, done late.**
    ///
    /// The other half of that door's contract, and the reason it can be taken
    /// at all. A caller takes `mapped_uninit` on a PLAN to overwrite the whole
    /// allocation — the serving artifact's warm boot plans exactly that — and
    /// a plan can fail: one of the images it reads can hash to something other
    /// than what the file's block table states, or the disk can refuse a read
    /// halfway through one. What is left then is a mixture of restored bytes and bytes the
    /// kernel handed out, and the load that falls through to the cold path is
    /// about to write only the spans an index names.
    ///
    /// So this is called on exactly that fall-through, and it puts the
    /// allocation back into the state [`Pinned::mapped`] would have handed
    /// over — which is what makes the padding between spans a deterministic
    /// image again, and therefore what keeps the artifact the cold path is
    /// about to write from carrying bytes nobody wrote.
    ///
    /// **Sound where [`Pinned::write`] is sound**, and no wider: the span is
    /// the whole allocation, so a caller may take it only where the discipline
    /// says the whole allocation is its own.
    pub fn zero(&self) {
        if self.host.is_null() || self.bytes == 0 {
            return;
        }
        // SAFETY: the span is the allocation, which outlives the write; what
        // makes it writable AT ALL is the discipline the doc above states.
        unsafe { core::ptr::write_bytes(self.host, 0, self.bytes) }
    }

    fn alloc(bytes: usize, zeroed: bool) -> Result<Pinned> {
        if bytes == 0 {
            return Ok(Pinned {
                host: core::ptr::null_mut(),
                device: 0,
                bytes: 0,
                origin: Origin::Allocated,
            });
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            let mut host: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `host` is a live local; the allocation is this
            // structure's, freed exactly once in `Drop`.
            unsafe {
                crate::device::ctx::check(
                    "cudaHostAlloc",
                    rt::cudaHostAlloc(&raw mut host, bytes, rt::cudaHostAllocMapped),
                )?;
                if zeroed {
                    core::ptr::write_bytes(host.cast::<u8>(), 0, bytes);
                }
            }
            let mut device: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `host` is the allocation just made and mapped.
            unsafe {
                crate::device::ctx::check(
                    "cudaHostGetDevicePointer",
                    rt::cudaHostGetDevicePointer(&raw mut device, host, 0),
                )?;
            }
            Ok(Pinned {
                host: host.cast(),
                device: device as u64,
                bytes,
                origin: Origin::Allocated,
            })
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = (bytes, zeroed);
            Err(Fault::Runtimeless)
        }
    }

    /// The address the host stores through.
    #[must_use]
    pub fn host(&self) -> *mut u8 {
        self.host
    }

    /// The address a kernel dereferences. Equal to [`Pinned::host`] under
    /// UVA, asked rather than assumed.
    #[must_use]
    pub fn device(&self) -> u64 {
        self.device
    }

    /// How many bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The bytes, as the host sees them right now.
    ///
    /// **A SNAPSHOT OF SOMETHING ANOTHER AGENT MAY BE WRITING.** Sound only
    /// where the channel plane's SPSC discipline says this span is the
    /// caller's: a cell below the reader's tail, or a cell above the writer's
    /// head.
    #[must_use]
    pub fn read(&self, offset: usize, len: usize) -> Vec<u8> {
        if len == 0 || self.host.is_null() || offset + len > self.bytes {
            return vec![0u8; len];
        }
        // SAFETY: the span is inside this allocation, which outlives the copy.
        unsafe { core::slice::from_raw_parts(self.host.add(offset).cast_const(), len).to_vec() }
    }

    /// **BORROW A SPAN OF THE PAGE-LOCKED BYTES**, or `None` for one this
    /// allocation does not hold.
    ///
    /// [`Pinned::read`]'s window, and the difference is the whole reason it
    /// exists: `read` COPIES, and the caller that needs this one is the tier
    /// artifact's writer, whose image is tens of gigabytes and whose chunk is
    /// 64 MiB. Asking `read` for it would allocate the tier a second time on
    /// the host, to hand it to a `write_all` that copies it again (§K.3).
    ///
    /// **THE SPSC CAVEAT IS `read`'s, AND IT IS LONGER HERE.** These bytes
    /// are the same bytes a device kernel and a guest thread may be storing
    /// through, and nothing about a `&[u8]` says otherwise. So this is sound
    /// exactly where the discipline above says the span is the caller's — and
    /// for a borrow, for as long as the borrow lives, not for the instant of
    /// a copy. The writer that takes it holds it across one `write_all` of
    /// one chunk, on the load thread, after the landing and before the first
    /// fire: no kernel is enqueued and no guest exists.
    ///
    /// An `at` or a `len` past the allocation is `None` rather than a clamp,
    /// because a short span silently written into a file is the failure this
    /// door is on the path of.
    #[must_use]
    pub fn view(&self, at: u64, len: u64) -> Option<&[u8]> {
        if len == 0 {
            return Some(&[]);
        }
        let at = usize::try_from(at).ok()?;
        let len = usize::try_from(len).ok()?;
        if self.host.is_null() || at.checked_add(len)? > self.bytes {
            return None;
        }
        // SAFETY: the span is inside this allocation, which outlives the
        // borrow; what makes the bytes readable AT ALL is the discipline the
        // doc above states, exactly as for `Pinned::read`.
        Some(unsafe { core::slice::from_raw_parts(self.host.add(at).cast_const(), len) })
    }

    /// Store `bytes` at `offset`, answering `false` when they would not fit.
    ///
    /// As [`Pinned::read`]: sound where the discipline says the span is the
    /// caller's.
    pub fn write(&self, offset: usize, bytes: &[u8]) -> bool {
        if self.host.is_null() || offset + bytes.len() > self.bytes {
            return false;
        }
        // SAFETY: the span is inside this allocation, which outlives the copy.
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), self.host.add(offset), bytes.len()) }
        true
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        if self.host.is_null() {
            return;
        }
        match self.origin {
            #[cfg(feature = "_cuda")]
            Origin::Allocated => {
                // SAFETY: the pointer came from this structure's own
                // `cudaHostAlloc` and is freed exactly once.
                unsafe {
                    let _ = cudarc::runtime::sys::cudaFreeHost(self.host.cast());
                }
            }
            // **UNREGISTER, THEN UNMAP, AND THE ORDER IS THE ORDER**: the
            // driver's page-lock is a claim on pages this process owns, so
            // dropping the pages first would leave the claim pointing at
            // whatever the address range becomes next.
            Origin::Registered => {
                #[cfg(feature = "_cuda")]
                // SAFETY: the span is the one this structure registered.
                unsafe {
                    let _ = cudarc::runtime::sys::cudaHostUnregister(self.host.cast());
                }
                // SAFETY: unmapping the mapping `Pinning::uninit` made, once.
                unsafe {
                    libc::munmap(self.host.cast(), self.bytes.max(1));
                }
            }
            #[cfg(not(feature = "_cuda"))]
            Origin::Allocated => {}
        }
    }
}

/// **AN IMAGE THAT IS FILLED FIRST AND PAGE-LOCKED LAST** — the road a
/// sixty-gigabyte tier takes, and the measurement that put it here.
///
/// `cudaHostAlloc` holds the CUDA runtime's memory-manager lock for the whole
/// of a large allocation, and *every* other CUDA call on every other thread
/// waits behind it — not just another allocation. Measured on this box at
/// 40 GiB: one `cudaHostAlloc` runs 23.7 s and stalls a one-megabyte
/// `cudaMalloc` and a pre-allocated `cudaMemcpyAsync` for all 23.7 s of it.
/// That is hazard H1 (§L.7), and at qwen4's 60 GiB it was ~48 s of a warm
/// deferred boot: the load's own remaining allocations, and any fire the seat
/// was supposed to be serving during the window, sat behind the page-lock of
/// an image nobody was waiting for.
///
/// **The lock is not the pages, it is the CALL.** So the image is made the
/// way the kernel makes any other sixty gigabytes — an anonymous mapping,
/// advised into huge pages, faulted in BY THE READ THAT FILLS IT, which takes
/// no CUDA lock at all — and one `cudaHostRegister` locks the finished thing
/// at the end. Same box, same 40 GiB: 7.2 s in total with a 4.2 s stall, and
/// at qwen4's 60 GiB the register itself is **3.4 s** (17.7 GiB/s) where the
/// `cudaHostAlloc` it replaces was ~36. The huge pages are why: locking is
/// per-page work, and `MADV_HUGEPAGE` gives the driver five hundred times
/// fewer of them to walk.
///
/// **CHUNKING THE REGISTER WAS THE OTHER CANDIDATE AND IT IS WORSE**, which
/// is worth saying because §L.7 named it: registering the same 40 GiB in
/// 256 MiB pieces took 45.8 s (128.4 s at 64 MiB) and still stalled a
/// neighbouring `cudaMalloc` for 28.6 s, because the registering thread
/// re-takes the lock faster than anyone else can win it. One call over huge
/// pages beats many calls over small ones twice over.
///
/// **AND THE ADDRESS IS THE SAME NUMBER ON BOTH SIDES**, as it is for
/// `cudaHostAlloc(cudaHostAllocMapped)`: [`Pinning::lock`] asks
/// `cudaHostGetDevicePointer` rather than assuming it, so an image whose two
/// bases disagree is a refusal and not a wrong address.
#[derive(Debug)]
pub struct Pinning {
    host: *mut u8,
    bytes: usize,
}

// SAFETY: as `Pinned`'s — an address and a length, whose sole writer is the
// thread that holds it. `Pinning` hands out no aliases at all: `host` is the
// one door, and `lock` consumes the value.
unsafe impl Send for Pinning {}

impl Pinning {
    /// **Map `bytes` of pages nothing has written yet**, advised into huge
    /// pages so that the page-lock at the end of the fill is cheap.
    ///
    /// Uninitialized in the sense [`Pinned::mapped_uninit`] is — the kernel
    /// hands out zeros, and a caller takes this door only on a plan to
    /// overwrite every byte before reading one.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] naming `mmap`, for a mapping the kernel refused.
    pub fn uninit(bytes: usize) -> Result<Pinning> {
        // SAFETY: a fresh private anonymous mapping of a stated length; no fd
        // and no offset are involved, and the pages belong to nobody else.
        let at = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                bytes.max(1),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if at == libc::MAP_FAILED {
            return Err(Fault::Device {
                call: "mmap",
                code: -1,
            });
        }
        // Advisory in both directions: a kernel that will not give huge pages
        // says nothing and the image is locked page by page instead, which is
        // the cost `Pinning`'s doc measures and not a failure.
        // SAFETY: the range is the mapping this call just made.
        unsafe { libc::madvise(at, bytes.max(1), libc::MADV_HUGEPAGE) };
        Ok(Pinning {
            host: at.cast(),
            bytes,
        })
    }

    /// The base of the mapping, for the reader that fills it.
    #[must_use]
    pub fn host(&self) -> *mut u8 {
        self.host
    }

    /// **PAGE-LOCK THE FINISHED IMAGE** — one `cudaHostRegister` over the
    /// whole mapping, and the device address asked for rather than assumed.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] without a CUDA runtime, [`Fault::Device`] for
    /// whatever `cudaHostRegister` or `cudaHostGetDevicePointer` said. **The
    /// mapping is returned to the kernel on every one of them**, so a refusal
    /// leaves nothing behind.
    pub fn lock(self) -> Result<Pinned> {
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            if self.bytes == 0 {
                return Pinned::mapped(0);
            }
            // SAFETY: the span is this structure's own mapping, live until
            // the `forget` below hands it to the `Pinned`.
            let locked = unsafe {
                crate::device::ctx::check(
                    "cudaHostRegister",
                    rt::cudaHostRegister(self.host.cast(), self.bytes, rt::cudaHostRegisterMapped),
                )
            };
            locked?;
            let mut device: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `host` is the mapping just registered.
            let asked = unsafe {
                crate::device::ctx::check(
                    "cudaHostGetDevicePointer",
                    rt::cudaHostGetDevicePointer(&raw mut device, self.host.cast(), 0),
                )
            };
            if asked.is_err() {
                // SAFETY: undoing the registration this call just made.
                unsafe { let _ = rt::cudaHostUnregister(self.host.cast()); }
                asked?;
            }
            let host = self.host;
            let bytes = self.bytes;
            // The mapping belongs to the `Pinned` from here on, and its `Drop`
            // is the one that unregisters and unmaps it.
            core::mem::forget(self);
            Ok(Pinned {
                host,
                device: device as u64,
                bytes,
                origin: Origin::Registered,
            })
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }
}

impl Drop for Pinning {
    fn drop(&mut self) {
        // SAFETY: unmapping the mapping `Pinning::uninit` made, exactly once —
        // `lock` forgets the value rather than reaching here.
        unsafe {
            libc::munmap(self.host.cast(), self.bytes.max(1));
        }
    }
}

/// **Copy `bytes` to a device address the caller resolved, and wait.**
///
/// [`Buffer::write`]'s free-function twin, for the one owner that cannot hand
/// out a `&mut Buffer`: a channel's SHARED device ring lives behind an
/// `Arc<Endpoint>` (`program::endpoint`), so every holder has `&self` and the
/// seeds a bind plants have to reach it anyway. The bounds check is the
/// caller's for the same reason — the address is already resolved when it
/// arrives here.
///
/// Synchronous, like [`Buffer::write`], and for its reason: the only caller is
/// bind-time seeding, which is control plane and orders against nothing.
///
/// # Errors
///
/// [`Fault::Device`] for the copy, [`Fault::Runtimeless`] with no runtime
/// selected.
pub fn write_raw(at: u64, bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: `at` is an address the caller resolved against an allocation
        // it owns, and `bytes` is a live host slice for the duration of a
        // synchronous copy.
        unsafe {
            crate::device::ctx::check(
                "cudaMemcpy",
                rt::cudaMemcpy(
                    at as *mut core::ffi::c_void,
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                ),
            )
        }
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = at;
        Err(Fault::Runtimeless)
    }
}

/// **One device-to-device copy, on `stream`** (alto F3).
///
/// A free function rather than a [`Buffer`] method because the two callers do
/// not hold one: the buffered-activation scatter moves bytes between an ARENA
/// rectangle and a POOL slab, and `copy_state` moves them between two slots of
/// the same slab. Both hold resolved device addresses, which is what this
/// takes.
///
/// On the stream, never synchronous, for the reason
/// [`Buffer::zero_span_on`] gives: a synchronous copy is ordered against every
/// stream in the process, which is a host wait between two waves (article 2)
/// and a lie about what this fire depends on.
///
/// # Errors
///
/// [`Fault::Device`] for the copy, [`Fault::Runtimeless`] with no runtime
/// selected.
pub fn copy_d2d(
    stream: *mut core::ffi::c_void,
    dst: u64,
    src: u64,
    bytes: usize,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: both addresses are resolutions the caller made against
        // allocations this shell owns for the load's lifetime, and the copy is
        // enqueued on the same stream as the launches around it.
        unsafe {
            crate::device::ctx::check(
                "cudaMemcpyAsync",
                rt::cudaMemcpyAsync(
                    dst as *mut core::ffi::c_void,
                    src as *const core::ffi::c_void,
                    bytes,
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                    stream.cast(),
                ),
            )
        }
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (stream, dst, src);
        Err(Fault::Runtimeless)
    }
}

/// **Copy `bytes` between two addresses of ANY kind, on `stream`** — the
/// transfer whose ends the caller resolved and whose direction the driver
/// works out.
///
/// [`copy_d2d`]'s twin for the one pair that is not device-to-device: a
/// routed expert's promotion reads MAPPED PINNED HOST memory (alto design
/// §7's T1) and writes the device slab, and both ends are plain `u64`
/// addresses under UVA. `cudaMemcpyDefault` is the kind that reads the
/// direction off the addresses themselves, which is the only honest spelling
/// when one end is a `cudaHostAlloc`ed page and the other is a `cudaMalloc`ed
/// one — naming `DeviceToDevice` for that pair would be a claim about the
/// allocations that this module cannot make.
///
/// Asynchronous, like everything the promotion path enqueues: the source must
/// stay unwritten until the copy completes, which for the tier means the
/// pinned mirror it owns for the load's whole life.
///
/// # Errors
///
/// [`Fault::Device`] for the copy, [`Fault::Runtimeless`] with no runtime
/// selected.
pub fn copy_any(
    stream: *mut core::ffi::c_void,
    dst: u64,
    src: u64,
    bytes: usize,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: both addresses are resolutions the caller made against
        // allocations this shell owns for the load's lifetime, and the copy is
        // enqueued on a stream the caller keeps ordered around it.
        unsafe {
            crate::device::ctx::check(
                "cudaMemcpyAsync",
                rt::cudaMemcpyAsync(
                    dst as *mut core::ffi::c_void,
                    src as *const core::ffi::c_void,
                    bytes,
                    rt::cudaMemcpyKind::cudaMemcpyDefault,
                    stream.cast(),
                ),
            )
        }
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (stream, dst, src);
        Err(Fault::Runtimeless)
    }
}

/// **Zero `len` bytes at a device address, and wait.**
///
/// [`Buffer::zero_span`]'s free-function twin, for the bytes whose bounds
/// were checked somewhere else: an [`Arena`](crate::device::Arena) span
/// checks against what is COMMITTED rather than against a fixed length, and
/// that check cannot live in a method on a `Buffer` that no longer owns the
/// bytes.
///
/// # Errors
///
/// [`Fault::Device`] for the fill, [`Fault::Runtimeless`] with no runtime.
pub fn zero_span(at: u64, len: usize) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: the address is a span the caller resolved against an
        // allocation this shell owns.
        unsafe {
            crate::device::ctx::check(
                "cudaMemset",
                rt::cudaMemset(at as *mut core::ffi::c_void, 0, len),
            )
        }
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = at;
        Err(Fault::Runtimeless)
    }
}

/// **Zero `len` bytes at a device address on `stream`**, ordered with the
/// fire — [`Buffer::zero_span_on`]'s free-function twin.
///
/// # Errors
///
/// As [`zero_span`].
pub fn zero_span_on(stream: *mut core::ffi::c_void, at: u64, len: usize) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: as `zero_span`, and the caller keeps the span alive past
        // the enqueue.
        unsafe {
            crate::device::ctx::check(
                "cudaMemsetAsync",
                rt::cudaMemsetAsync(at as *mut core::ffi::c_void, 0, len, stream.cast()),
            )
        }
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (stream, at);
        Err(Fault::Runtimeless)
    }
}

/// **Copy `into.len()` bytes back from a device address, and wait** —
/// [`Buffer::read`]'s free-function twin.
///
/// # Errors
///
/// [`Fault::Device`] for the copy, [`Fault::Runtimeless`] with no runtime.
pub fn copy_d2h(at: u64, into: &mut [u8]) -> Result<()> {
    if into.is_empty() {
        return Ok(());
    }
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: the address is a checked span and `into` is a live host
        // slice of exactly the length copied.
        unsafe {
            crate::device::ctx::check(
                "cudaMemcpy",
                rt::cudaMemcpy(
                    into.as_mut_ptr().cast(),
                    at as *const core::ffi::c_void,
                    into.len(),
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                ),
            )
        }
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = at;
        Err(Fault::Runtimeless)
    }
}

/// **What the card says is free, in bytes**, or `None` with no runtime.
///
/// The one observation that can settle whether an elastic commit actually put
/// physical pages on the device rather than only moving a counter: a gate
/// reads it either side of a fire and diffs. Not on any serving path — a
/// `cudaMemGetInfo` is a driver round trip, and the only production caller is
/// [`PhysicalPool`](crate::device::PhysicalPool)'s budget, which asks once at
/// load and again only when a commit has to grow.
#[must_use]
pub fn free_bytes() -> Option<u64> {
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: two live locals; the call only writes them.
        let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
        (asked == rt::cudaError::cudaSuccess).then_some(free as u64)
    }
    #[cfg(not(feature = "_cuda"))]
    {
        None
    }
}
