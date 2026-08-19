//! `moe::build_moe_ptrs_aligned_bf16` — the aligned MoE leg's pointer build,
//! kept as a driver op since the six pointer arrays must outlive normal trace
//! liveness across both grouped GEMMs, kept alive via [`MoePtrArena`].


use std::ffi::c_void;

use crate::fire::sideband_arena::DeviceMemory;

/// The six device pointers one batched GEMM per projection needs.
#[derive(Debug, Clone, Copy)]
pub struct Arrays {
    /// gate/up activations — the aligned rectangle, one row block per entry.
    pub a_gu: *mut *const c_void,
    /// gate/up weights — the expert's slice of the `[E, 2·I, H]` bank.
    pub b_gu: *mut *const c_void,
    /// gate/up output — into the `gate_up` staging.
    pub c_gu: *mut *mut c_void,
    /// down activations — into the `act` staging.
    pub a_dn: *mut *const c_void,
    /// down weights — the expert's slice of the `[E, H, I]` bank.
    pub b_dn: *mut *const c_void,
    /// down output — into the `out` staging.
    pub c_dn: *mut *mut c_void,
    /// Which bank `b_gu` slices, so an arm can tell the two GEMMs apart by address.
    pub bank_gu: *const c_void,
    /// See [`Self::bank_gu`].
    pub bank_dn: *const c_void,
}

/// Which projection a grouped GEMM is; no `Unknown` variant, since an
/// unrecognized bank is a refusal, not a third state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Half {
    /// `[aligned, H] @ [E, 2·I, H]^T -> [aligned, 2·I]`, the gate/up pair.
    GateUp,
    /// `[aligned, I] @ [E, H, I]^T -> [aligned, H]`, the down projection.
    Down,
}

impl Arrays {
    /// Which of the two triples serves a GEMM that names `bank`. `None` is a
    /// refusal, not a default guess, for a bank neither half was built for.
    #[must_use]
    pub fn select(&self, bank: *const c_void) -> Option<Half> {
        if bank.is_null() {
            return None;
        }
        if std::ptr::eq(bank, self.bank_gu) {
            Some(Half::GateUp)
        } else if std::ptr::eq(bank, self.bank_dn) {
            Some(Half::Down)
        } else {
            None
        }
    }

    /// The `(activations, weights, output)` triple for a half, in cuBLAS's order.
    #[must_use]
    pub fn triple(
        &self,
        half: Half,
    ) -> (
        *const *const c_void,
        *const *const c_void,
        *const *mut c_void,
    ) {
        match half {
            Half::GateUp => (
                self.a_gu.cast_const(),
                self.b_gu.cast_const(),
                self.c_gu.cast_const(),
            ),
            Half::Down => (
                self.a_dn.cast_const(),
                self.b_dn.cast_const(),
                self.c_dn.cast_const(),
            ),
        }
    }
}

/// The four weight bases, with the shared-expert pair kept nullable.
///
/// The nullity is load-bearing: the host program rewrites `routed_blocks` to
/// `max_blocks` when a shared base is null; don't reproduce that rewrite here.
#[derive(Debug, Clone, Copy)]
pub struct Banks {
    /// The routed experts' `[E, 2·I, H]` gate/up bank.
    pub gate_up: *const c_void,
    /// The routed experts' `[E, H, I]` down bank.
    pub down: *const c_void,
    /// The shared expert's gate/up projection, or null.
    pub shared_gate_up: *const c_void,
    /// The shared expert's down projection, or null.
    pub shared_down: *const c_void,
}

/// The three staging buffers the statement declares as results, not scratch —
/// the build must fix their bases before anything writes them.
#[derive(Debug, Clone, Copy)]
pub struct Stage {
    /// `[aligned, 2·I]` bf16 — grouped GEMM 1's destination.
    pub gate_up: *mut c_void,
    /// `[aligned, I]` bf16 — the swiglu's destination.
    pub act: *mut c_void,
    /// `[aligned, H]` bf16 — grouped GEMM 2's destination.
    pub out: *mut c_void,
}

/// The five load-time constants the statement can't carry as operands:
/// `build_moe_ptrs_aligned` is built on `record_many`, which has no params channel.
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    /// The padded block count the arrays are sized for (`MOE_MAX_BLOCKS`, 1024); also the grid.
    pub max_blocks: i32,
    /// Rows per padded block (`MOE_ALIGNED_BLOCK`, 16); the grouped GEMM's `FRAG`
    /// requires the same 16 of `M`.
    pub block_size: i32,
    /// The model's hidden width. Also the `out` staging's row width.
    pub hidden: i32,
    /// The routed experts' intermediate width. Also the `act` staging's.
    pub moe_intermediate: i32,
    /// Where the routed blocks end and the shared tail begins, before the host
    /// program's null rewrite.
    pub routed_blocks: i32,
}

/// Why a build handed back no arrays: a shape the kernel declines, or a
/// device that would not give the arena its next block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decline {
    /// The host program declined — today only `max_blocks <= 0`, an empty padded batch.
    Refused(kernels_cuda::Refusal),
    /// The bump arena could not grow to hold six arrays of this many pointers.
    NoArena {
        /// What the six arrays needed, in bytes.
        bytes: usize,
    },
}

/// A build's outcome. `#[must_use]` — an ignored decline would launch two
/// GEMMs over pointer arrays that were never written.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub enum Built {
    /// The launch went to the device; the arrays are live on the stream.
    Ready(Arrays),
    /// Nothing was launched.
    Declined(Decline),
}

/// The per-fire bump arena the six pointer arrays are carved from.
///
/// 256-aligned, doubling growth, retire-on-grow: the old block is kept, not
/// freed, since an in-flight fire may still read it ([`Self::release`] frees it).
#[derive(Debug, Default)]
pub struct MoePtrArena {
    buf: *mut c_void,
    buf_size: usize,
    used: usize,
    retired: Vec<*mut c_void>,
}

impl MoePtrArena {
    /// Alignment every carve starts on; costs at most 248 bytes once per fire.
    const ALIGN: usize = 256;

    /// Reclaim the space, at the top of a fire once prior launches are ordered
    /// on the stream ahead of this one's.
    pub const fn reset(&mut self) {
        self.used = 0;
    }

    /// The current backing block's base — never dereferenced, only fingerprinted.
    #[must_use]
    pub const fn base(&self) -> *mut c_void {
        self.buf
    }

    /// Bump-allocate `bytes`, growing the backing when it does not fit.
    /// Returns null when the device refuses growth — the caller must test it.
    pub fn alloc<M: DeviceMemory>(&mut self, mem: &mut M, bytes: usize) -> *mut c_void {
        let at = self.used.div_ceil(Self::ALIGN) * Self::ALIGN;
        if at + bytes > self.buf_size {
            let want = ((at + bytes) * 2).max(1 << 20);
            if self.buf_size > 0 {
                self.retired.push(self.buf);
            }
            self.buf = mem.alloc(want).unwrap_or(std::ptr::null_mut());
            self.buf_size = if self.buf.is_null() { 0 } else { want };
            self.used = 0;
            if self.buf.is_null() {
                return std::ptr::null_mut();
            }
            // SAFETY: fresh block is at least `bytes` long at offset zero;
            // only handed to a launch, never read here.
            self.used = bytes;
            return self.buf;
        }
        self.used = at + bytes;
        // SAFETY: `at + bytes <= buf_size` by construction; only handed to a
        // launch, never dereferenced here.
        unsafe { self.buf.cast::<u8>().add(at).cast() }
    }

    /// Free the backing and every retired block through the seam.
    pub fn release<M: DeviceMemory>(&mut self, mem: &mut M) {
        if !self.buf.is_null() {
            mem.free(self.buf);
            self.buf = std::ptr::null_mut();
        }
        for p in self.retired.drain(..) {
            mem.free(p);
        }
        self.buf_size = 0;
        self.used = 0;
    }

    /// Carve six arrays of `max_blocks` pointers, or nothing. One `alloc` call
    /// and six offsets, so a growth cannot split the six across two blocks.
    fn carve<M: DeviceMemory>(
        &mut self,
        mem: &mut M,
        max_blocks: i32,
        banks: Banks,
    ) -> Option<Arrays> {
        let slots = usize::try_from(max_blocks).ok()?;
        let one = slots.checked_mul(size_of::<*const c_void>())?;
        let stride = one.div_ceil(Self::ALIGN) * Self::ALIGN;
        let base = self.alloc(mem, stride.checked_mul(6)?);
        if base.is_null() {
            return None;
        }
        // SAFETY: block is `6 * stride` long; every offset is a multiple of
        // `stride` under six.
        let at = |i: usize| unsafe { base.cast::<u8>().add(i * stride).cast::<c_void>() };
        Some(Arrays {
            a_gu: at(0).cast(),
            b_gu: at(1).cast(),
            c_gu: at(2).cast(),
            a_dn: at(3).cast(),
            b_dn: at(4).cast(),
            c_dn: at(5).cast(),
            // Carried here, not re-derived in `build`: the arrays outlive this call.
            bank_gu: banks.gate_up,
            bank_dn: banks.down,
        })
    }

    /// What [`Self::carve`] will ask the device for, to size the arena before a fire.
    #[must_use]
    pub fn carve_bytes(max_blocks: i32) -> usize {
        let slots = usize::try_from(max_blocks).unwrap_or(0);
        let one = slots.saturating_mul(size_of::<*const c_void>());
        one.div_ceil(Self::ALIGN) * Self::ALIGN * 6
    }
}

/// `moe::build_moe_ptrs_aligned_bf16` — carve the six arrays, then fill them on
/// the device; returned via `DispatchCtx::moe_ptrs` for later dispatch statements.
///
/// # Safety
///
/// Every pointer in `banks`/`stage`/`expert_ids`/`aligned_in` must be a live
/// device allocation of the aligned leg's shapes until the launch completes.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn build<M: DeviceMemory>(
    arena: &mut MoePtrArena,
    mem: &mut M,
    expert_ids: *const c_void,
    aligned_in: *const c_void,
    banks: Banks,
    stage: Stage,
    bounds: Bounds,
    stream: *mut c_void,
) -> Built {
    use kernels_cuda::jit::abi::bf16;

    // The only refusal here is a device that won't grow the arena; carving
    // first means a decline allocated nothing.
    let Some(arrays) = arena.carve(mem, bounds.max_blocks, banks) else {
        return Built::Declined(Decline::NoArena {
            bytes: MoePtrArena::carve_bytes(bounds.max_blocks),
        });
    };
    // SAFETY: the caller's obligation, above.
    let ctx = unsafe { kernels_cuda::jit::Ctx::on(stream) };
    // Argument order doesn't match parameter order: `act` takes
    // `moe_intermediate`, `out` takes `hidden`.
    let fired = kernels_cuda::moe::build_moe_ptrs_aligned_bf16(
        &ctx,
        kernels::routine::In { ptr: expert_ids.cast::<i32>(), rows: 0, width: 0 },
        kernels::routine::Const { v: banks.gate_up.cast::<bf16>() },
        kernels::routine::Const { v: banks.down.cast::<bf16>() },
        kernels::routine::In { ptr: aligned_in.cast::<bf16>(), rows: 0, width: 0 },
        kernels::routine::Out { ptr: stage.gate_up.cast::<bf16>(), rows: 0, width: 0 },
        kernels::routine::Out { ptr: stage.act.cast::<bf16>(), rows: 0, width: bounds.moe_intermediate },
        kernels::routine::Out { ptr: stage.out.cast::<bf16>(), rows: 0, width: bounds.hidden },
        arrays.a_gu.cast::<*const bf16>(),
        arrays.b_gu.cast::<*const bf16>(),
        arrays.c_gu.cast::<*mut bf16>(),
        arrays.a_dn.cast::<*const bf16>(),
        arrays.b_dn.cast::<*const bf16>(),
        arrays.c_dn.cast::<*mut bf16>(),
        bounds.max_blocks,
        bounds.block_size,
        bounds.routed_blocks,
        banks.shared_gate_up.cast::<bf16>(),
        banks.shared_down.cast::<bf16>(),
    );
    match fired {
        Ok(()) => Built::Ready(arrays),
        Err(why) => Built::Declined(Decline::Refused(why)),
    }
}

/// The arena's allocator: the three CUDA calls [`MoePtrArena`] needs, on the
/// fire's raw stream. Separate from `sideband_arena::LiveDeviceMemory`, which
/// wants a `device::StreamRef` a dispatch arm (holding a raw pointer) lacks.
#[cfg(feature = "_cuda")]
#[derive(Debug, Clone, Copy)]
pub struct LiveMoePtrMemory {
    stream: *mut c_void,
}

#[cfg(feature = "_cuda")]
impl LiveMoePtrMemory {
    /// Allocations ordered on `stream` — the fire's.
    #[must_use]
    pub const fn new(stream: *mut c_void) -> Self {
        Self { stream }
    }
}

#[cfg(feature = "_cuda")]
impl DeviceMemory for LiveMoePtrMemory {
    fn alloc(&mut self, bytes: usize) -> Option<*mut c_void> {
        use cudarc::runtime::sys::{cudaError, cudaMalloc};
        let mut p: *mut c_void = core::ptr::null_mut();
        let code = unsafe { cudaMalloc(std::ptr::from_mut(&mut p), bytes) };
        (code == cudaError::cudaSuccess && !p.is_null()).then_some(p)
    }

    // Safe by the trait's design: only pointers this impl's `alloc` produced
    // come back here.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn free(&mut self, ptr: *mut c_void) {
        let _ = unsafe { cudarc::runtime::sys::cudaFree(ptr) };
    }

    fn synchronize(&mut self) -> bool {
        use cudarc::runtime::sys::{cudaError, cudaStreamSynchronize};
        unsafe { cudaStreamSynchronize(self.stream.cast()) == cudaError::cudaSuccess }
    }
}

#[cfg(feature = "_cuda")]
thread_local! {
    /// The arena, per thread: retire-on-grow is only sound if one fire owns
    /// it, and a fire is a thread.
    static ARENA: std::cell::RefCell<MoePtrArena> =
        std::cell::RefCell::new(MoePtrArena::default());
}

// Arrays belong to the dispatch that built them; a second dispatch has its
// own `ctx`, so nothing reads stale across fires.

/// [`build`] against the per-thread arena — the entry point a dispatch arm
/// calls, holding a raw stream and no arena.
///
/// # Safety
///
/// [`build`]'s, unchanged.
#[cfg(feature = "_cuda")]
pub unsafe fn build_for_fire(
    expert_ids: *const c_void,
    aligned_in: *const c_void,
    banks: Banks,
    stage: Stage,
    bounds: Bounds,
    stream: *mut c_void,
) -> Built {
    let mut mem = LiveMoePtrMemory::new(stream);
    ARENA.with(|arena| {
        let mut arena = arena.borrow_mut();
        // SAFETY: the caller's obligation, forwarded unchanged.
        unsafe {
            build(
                &mut arena, &mut mem, expert_ids, aligned_in, banks, stage, bounds, stream,
            )
        }
    })
}
