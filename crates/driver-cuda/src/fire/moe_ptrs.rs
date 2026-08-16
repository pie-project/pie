//! `moe::build_moe_ptrs_aligned_bf16` — the aligned MoE leg's pointer build,
//! as a **driver op**: `x::moe`'s second `contract!` with no `Entry`.
//!
//! # Why this file exists, and it is not a preference
//!
//! The aligned leg is the only leg qwen3.5 has once the fused CUTLASS call is
//! retired — every condition in `model/src/qwen_3_5/forward/mod.rs:341-349`
//! that turns the fused leg off already returns `moe_mlp_body_aligned_cuda`.
//! And the aligned leg cannot start without this call: the pointer build is
//! what DECLARES `gu_stage`/`act_stage`/`out_stage`, the three destinations
//! every op below it writes into. It is not an optimisation, it is step 3
//! of 8.
//!
//! It was a `none:` arm, and it had been unsourced in `table::moe` before
//! that — it has never had an arm in either world, and the fused leg has been
//! covering for it. That is the whole reason the retirement is gated on this
//! file.
//!
//! # Why a driver op and not a dtype
//!
//! The refusal's own sentence offered two ways out: *a dtype for an array of
//! device addresses*, or *the driver-op shape, since the arrays are the
//! driver's arena.* The dtype is the larger change and the second reason is
//! the one that decides it.
//!
//! **The size argument.** A `Ty` for `void**` is a widening of the SHARED
//! vocabulary — `kernels::Ty` and `model-compiler`'s `DType` both — against a
//! step 9 that is measured at *shrinking* it. And it would not be enough on
//! its own: `max_blocks`, `block_size` and `routed_blocks` are unsourced too,
//! and so are both shared-expert bank handles. One dtype buys six of eleven
//! missing operands.
//!
//! **The correctness argument, which is the real one.** The six arrays have
//! no stated consumer. They are read by the batched-cuBLAS fallback *inside*
//! `moe::moe_grouped_gemm_bf16`, which is a LOWERING of that statement and
//! not a statement of its own — the grouped GEMM's own parameter list is
//! `(a, weight_base, c, expert_ids)` and names no pointer array. So declaring
//! the six as trace results would give the plan six values nothing reads, and
//! `lower.rs:1911`'s liveness frees a value at the first op past its last
//! use: the arrays would return to the pool immediately after the build and
//! the next allocation would take their bytes. The batched GEMM would then
//! dereference pointer arrays overwritten with bf16 activations. That is a
//! wrong answer, not a refusal, and it is the same failure `lower.rs:1949`
//! already records for the rotated `k` — *"liveness then freed one buffer for
//! what placement had made two."*
//!
//! Stating them properly therefore means stating an operand only ONE of two
//! lowerings reads, which is the thing this migration is retiring. So the
//! arrays stay the driver's, and the symbol takes the shape
//! `moe::flashinfer_cutlass_moe_bf16` and `gemm::lora_qkv_correction` already
//! have: `contract!` yes, `Entry` no, body here.
//!
//! # What the arena is, and why not the sideband one
//!
//! [`SidebandArena`](crate::fire::sideband_arena::SidebandArena)'s `Region`
//! discriminants match the C++ `HookSidebandArena` slot array, so a fourth
//! region is a change to a pinned enum for a caller that wants none of its
//! machinery — this arena has one client, no fingerprint and no
//! free-before-realloc dance.
//!
//! [`MoePtrArena`] is `fire::lora::LoraStageArena`'s idiom instead, and for
//! its stated reason: 256-aligned bump allocation, doubling growth with a
//! 1 MiB floor, **retire-on-grow** rather than free-on-grow because an
//! in-flight fire may still be reading the old block, and a per-fire reset.
//! `cublasGemmGroupedBatchedEx` does not consume its pointer arrays
//! synchronously (`fire::lora`'s `upload_slab` says so about the same class
//! of buffer), so the six must outlive the launch that builds them and stay
//! live across BOTH grouped GEMMs. A per-fire bump arena is exactly that
//! lifetime and nothing narrower would be.
//!
//! Unlike lora's slab these arrays are filled **on the device**, by the
//! kernel, so there is no upload here and no host mirror. The arena only has
//! to hand out six correctly sized, correctly aligned blocks.

use std::ffi::c_void;

use crate::fire::sideband_arena::DeviceMemory;

/// The six device pointer arrays one batched GEMM per projection needs.
///
/// `a`/`b`/`c` are cuBLAS's operand roles — activations, weights, output —
/// and `gu`/`dn` are the two projections. Each array holds `max_blocks`
/// entries, one per padded block, and a padding block's slot is written with
/// whatever the kernel decides; the GEMM's group count is what bounds the
/// read, not the array's length.
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
    /// The `[E, 2·I, H]` bank whose expert slices `b_gu`'s entries address,
    /// and [`Self::bank_dn`] the `[E, H, I]` one behind `b_dn`.
    ///
    /// Carried so a later arm can say WHICH projection it is holding. The two
    /// grouped GEMMs are the same symbol with the same parameter list —
    /// `dsl::cuda::moe_grouped_gemm(act, expert_ids, stage, .., bank, ..)`
    /// twice, at `model/src/qwen_3_5/forward/mod.rs:198` and `:210` — and
    /// nothing else in the statement tells them apart. `N` would, on a model
    /// where `2·I != H`, which is a property of qwen3.5 and not of the shape.
    /// The BANK is exact: the build and both GEMMs are handed the same two
    /// weight NAMES (`w.expert_gate_up.name`, `w.expert_down.name`), so one
    /// `Resolver::weight` answer per name is one address, and the arm
    /// compares the address it was given against the two carved from.
    pub bank_gu: *const c_void,
    /// See [`Self::bank_gu`].
    pub bank_dn: *const c_void,
}

/// Which projection a grouped GEMM is, once [`Arrays::select`] has named it.
///
/// Two variants and no `Unknown`: a bank that is neither is a refusal at the
/// call site, not a third state carried around.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Half {
    /// `[aligned, H] @ [E, 2·I, H]^T -> [aligned, 2·I]`, the gate/up pair.
    GateUp,
    /// `[aligned, I] @ [E, H, I]^T -> [aligned, H]`, the down projection.
    Down,
}

impl Arrays {
    /// Which of the two triples serves a GEMM that names `bank`.
    ///
    /// `None` for a bank neither half was built for — a third projection, or
    /// a build whose banks are another layer's. That is a refusal and not a
    /// default: picking a triple by "it must be the other one" launches a
    /// GEMM over pointer arrays addressed to a tensor the statement does not
    /// name, which is the wrong-answer direction rather than the refusing
    /// one.
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

    /// The `(activations, weights, output)` triple for a half, in the order
    /// [`kernels_cuda::gemm::dense::batched_act_x_wt_bf16`] takes them.
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
/// The nullity is LOAD-BEARING and the host program owns what it means:
/// `x::moe::build_moe_ptrs_aligned_bf16` rewrites `routed_blocks` to
/// `max_blocks` when either shared base is null, so the kernel's
/// `is_shared = (b >= routed_blocks)` is false for every block and the shared
/// tail is never addressed. That WAS `execution.rs`'s `Control::Supplies` for
/// this symbol — *a host decision about an operand from a POINTER's nullity* —
/// and the walk that carried the sentence is deleted, because the sentence is
/// code now. It is stated once, in the host program. **Do not reproduce it
/// here.**
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

/// The three staging buffers the statement DECLARES as its results, in the
/// order `dsl::cuda::build_moe_ptrs_aligned` returns them.
///
/// They are results and not scratch precisely because the build has to know
/// their base addresses before anything writes them — a statement cannot take
/// an operand a later statement produces, so the build is what fixes where
/// the aligned staging lives and everything below fills a buffer it named.
#[derive(Debug, Clone, Copy)]
pub struct Stage {
    /// `[aligned, 2·I]` bf16 — grouped GEMM 1's destination.
    pub gate_up: *mut c_void,
    /// `[aligned, I]` bf16 — the swiglu's destination.
    pub act: *mut c_void,
    /// `[aligned, H]` bf16 — grouped GEMM 2's destination.
    pub out: *mut c_void,
}

/// The five numbers the statement does not carry.
///
/// Every one of them is a load-time constant of the aligned leg, and the
/// reason they arrive as parameters rather than as operands is that
/// `dsl::cuda::build_moe_ptrs_aligned` is built on `record_many`, which has
/// no params channel — where `moe_align` next door uses
/// `record_with_params` and rides its three load-time numbers there.
/// Widening the wrapper is a one-line change to `dsl.rs` and a REGENERATED
/// golden (`qwen3_5_moe_mlp_35b_a3b_cuda_aligned.json`), which is why it is
/// not done in the same change as the shape.
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    /// The padded block count the arrays are sized for — `MOE_MAX_BLOCKS`,
    /// 1024 at `model/src/qwen_3_5/forward/mod.rs:19`. It is the grid too.
    pub max_blocks: i32,
    /// Rows per padded block — `MOE_ALIGNED_BLOCK`, 16, and the same 16 that
    /// `x::moe::supported`'s `FRAG` requires of the grouped GEMM's `M`.
    pub block_size: i32,
    /// The model's hidden width. Also the `out` staging's row width.
    pub hidden: i32,
    /// The routed experts' intermediate width. Also the `act` staging's.
    pub moe_intermediate: i32,
    /// Where the routed blocks end and the shared expert's tail begins —
    /// **before** the null rewrite, which is the host program's.
    pub routed_blocks: i32,
}

/// Why a build handed back no arrays.
///
/// Two states rather than one because they are not the same event: one is a
/// shape the kernel declines, one is a device that would not give the arena
/// its next block. `fire::flashinfer_moe`'s `Fused`/`Decline` pair next door
/// is the shape being followed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decline {
    /// The host program declined — today only `max_blocks <= 0`
    /// (`moe_dispatch.cu:245`), an empty padded batch.
    Refused(kernels_cuda::Refusal),
    /// The bump arena could not grow to hold six arrays of this many
    /// pointers.
    NoArena {
        /// What the six arrays needed, in bytes.
        bytes: usize,
    },
}

/// A build's outcome. `#[must_use]` for the reason `Fired` is: the caller
/// that ignores a decline goes on to launch two GEMMs over pointer arrays
/// that were never written.
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
/// Ports `fire::lora::LoraStageArena` field for field, including the growth
/// rule, and for the same three reasons: 256-aligned allocations, doubling
/// growth with a 1 MiB floor so a fire that grows once does not grow again,
/// and **retire-on-grow** — the old block is kept, not freed, because an
/// in-flight fire may still be reading it. [`Self::release`] is where the
/// retired blocks go back.
///
/// Arithmetic over a [`DeviceMemory`], with no CUDA call of its own, which is
/// what lets the growth rule be checked without a card.
#[derive(Debug, Default)]
pub struct MoePtrArena {
    buf: *mut c_void,
    buf_size: usize,
    used: usize,
    retired: Vec<*mut c_void>,
}

impl MoePtrArena {
    /// Alignment every carve starts on. 256 is lora's and the C++ staging
    /// arena's; a pointer array only needs 8, and taking the larger number
    /// costs at most 248 bytes once per fire.
    const ALIGN: usize = 256;

    /// Reclaim the space. Called at the top of a fire, when the previous
    /// fire's launches have been ordered on the stream ahead of this one's.
    pub const fn reset(&mut self) {
        self.used = 0;
    }

    /// The current backing block's base — nothing dereferences it; it is the
    /// value a fingerprint would mix, as lora's does.
    #[must_use]
    pub const fn base(&self) -> *mut c_void {
        self.buf
    }

    /// Bump-allocate `bytes`, growing the backing when it does not fit.
    ///
    /// Returns null when the device refuses the growth, and the caller is
    /// expected to test it: this is the one failure that is not a shape.
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
            // SAFETY: the fresh block is at least `bytes` long and the offset
            // is zero; the pointer is handed to a launch, never read here.
            self.used = bytes;
            return self.buf;
        }
        self.used = at + bytes;
        // SAFETY: `at + bytes <= buf_size` by construction; the pointer is
        // only ever handed to a launch, never dereferenced here.
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

    /// Carve six arrays of `max_blocks` pointers, or nothing.
    ///
    /// One `alloc` and six offsets rather than six `alloc`s, so the six
    /// cannot be split across a growth: a partial carve would leave three
    /// arrays in a retired block the next fire's reset hands back.
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
        // SAFETY: the block is `6 * stride` long and every offset below is a
        // whole multiple of `stride` under six.
        let at = |i: usize| unsafe { base.cast::<u8>().add(i * stride).cast::<c_void>() };
        Some(Arrays {
            a_gu: at(0).cast(),
            b_gu: at(1).cast(),
            c_gu: at(2).cast(),
            a_dn: at(3).cast(),
            b_dn: at(4).cast(),
            c_dn: at(5).cast(),
            // The two bases are carried rather than re-derived because the
            // arrays outlive this call and the banks do not travel with the
            // GEMM that reads them. Taking them here rather than in `build`
            // is so that an `Arrays` is never half-filled: every field is
            // set by the one expression that mints the value.
            bank_gu: banks.gate_up,
            bank_dn: banks.down,
        })
    }

    /// What [`Self::carve`] will ask the device for, for a caller that wants
    /// to size the arena before a fire rather than during one.
    #[must_use]
    pub fn carve_bytes(max_blocks: i32) -> usize {
        let slots = usize::try_from(max_blocks).unwrap_or(0);
        let one = slots.saturating_mul(size_of::<*const c_void>());
        one.div_ceil(Self::ALIGN) * Self::ALIGN * 6
    }
}

/// `moe::build_moe_ptrs_aligned_bf16` — carve the six arrays, then fill them
/// on the device.
///
/// The arrays are the return value because they are the driver's and nothing
/// else can hold them: the two grouped GEMMs that read them are the same
/// dispatch's later statements, so `DispatchCtx::moe_ptrs` holds this
/// `Arrays` for the fire and [`crate::fire::moe_grouped`] hands it to the
/// batched fallback when
/// [`x::moe::supported`](kernels_cuda::moe::supported) refuses the
/// WMMA form. On qwen3.5 that is not hypothetical: gate/up has `K = 2048`
/// against a `SHORT_K` of 512, so the fallback is the only GEMM on that half.
///
/// # Safety
///
/// Every pointer in `banks` and `stage`, plus `expert_ids` and `aligned_in`,
/// must be a device allocation of the aligned leg's shapes on the current
/// device, live on `stream` until the launch completes. The two shared-expert
/// bases may be null and mean something when they are — see [`Banks`].
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

    // The shape test is the HOST PROGRAM'S and is made there, not here. What
    // is made here is the one refusal the host program cannot see, because
    // the arena is the caller's: a device that would not grow it. Carving
    // first also means a declined build has allocated nothing.
    let Some(arrays) = arena.carve(mem, bounds.max_blocks, banks) else {
        return Built::Declined(Decline::NoArena {
            bytes: MoePtrArena::carve_bytes(bounds.max_blocks),
        });
    };
    // SAFETY: the caller's obligation, above.
    let ctx = unsafe { kernels_cuda::jit::Ctx::on(stream) };
    let fired = kernels_cuda::moe::build_moe_ptrs_aligned_bf16(
        &ctx,
        expert_ids.cast::<i32>(),
        banks.gate_up.cast::<bf16>(),
        banks.down.cast::<bf16>(),
        aligned_in.cast::<bf16>(),
        stage.gate_up.cast::<bf16>(),
        stage.act.cast::<bf16>(),
        stage.out.cast::<bf16>(),
        arrays.a_gu.cast::<*const bf16>(),
        arrays.b_gu.cast::<*const bf16>(),
        arrays.c_gu.cast::<*mut bf16>(),
        arrays.a_dn.cast::<*const bf16>(),
        arrays.b_dn.cast::<*const bf16>(),
        arrays.c_dn.cast::<*mut bf16>(),
        bounds.max_blocks,
        bounds.block_size,
        bounds.hidden,
        bounds.moe_intermediate,
        bounds.routed_blocks,
        banks.shared_gate_up.cast::<bf16>(),
        banks.shared_down.cast::<bf16>(),
    );
    match fired {
        Ok(()) => Built::Ready(arrays),
        // Two spellings of an outcome, and this is not a `bind!` body: a
        // driver op answers its OWN two-state type, as `fire::flashinfer_moe`
        // does, and the arrays ride the `Ready` arm because a caller that got
        // a `Declined` must not be holding them.
        Err(why) => Built::Declined(Decline::Refused(why)),
    }
}

/// The arena's allocator: the three CUDA calls [`MoePtrArena`] needs, on the
/// fire's raw stream.
///
/// A second impl rather than a use of `sideband_arena::LiveDeviceMemory`
/// because that one takes a `device::StreamRef<'a>` and a dispatch arm holds
/// a raw `*mut c_void`. `fire::lora::LiveLoraOps` made the same call for the
/// same reason, and the three bodies are identical in all three places —
/// which is a fact about `cudaMalloc` having no interesting variations, not
/// duplication worth a fourth abstraction.
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

    // Safe by the trait's design: the arena hands back only pointers this
    // impl's `alloc` produced.
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
    /// The arena, per thread, for the reason `LoraStageArena` is per state:
    /// a bump allocator with retire-on-grow is only sound if one fire owns
    /// it, and a fire is a thread.
    static ARENA: std::cell::RefCell<MoePtrArena> =
        std::cell::RefCell::new(MoePtrArena::default());
}

// A SECOND `thread_local!` STOOD HERE — `LAST`, the carve stashed for the
// GEMM arm that reads it — and it is GONE, in the one edit its own doc said
// it was written for.
//
// It called itself a SEAM and named the better home: *"a field on
// `DispatchCtx`, beside `ctx.lora`, which is that struct's precedent for
// exactly this: state one arm builds and another consumes"*. That field
// landed as `DispatchCtx::moe_ptrs`, and as a `Cell<Option<Arrays>>` rather
// than the plain `Option` the ask asked for — `ctx.lora`'s precedent does
// not transfer, because `lora` is filled at CONSTRUCTION and this cannot be
// (the build is step 3 of 8) while every dispatch arm holds `&DispatchCtx`.
// A plain `Option` would have been `None` for the life of the fire.
//
// What the stash could never state is the part worth keeping: it was correct
// only because the build and both GEMMs are the same fire on the same thread
// in issue order, and NOTHING SAID SO — the next layer's build overwrote the
// cell and only the plan's shape kept that from being read. The `ctx` field
// says it: the arrays belong to the dispatch that built them, and a second
// dispatch has a second ctx.

/// [`build`] against the per-thread arena.
///
/// The entry point a dispatch arm calls: it holds a raw stream and no arena,
/// which is the whole difference from [`build`]. The `Ready` arrays are the
/// CALLER'S to keep — `DispatchCtx::moe_ptrs` is where they go, and the two
/// grouped GEMMs below this statement read them from there.
///
/// # Safety
///
/// [`build`]'s, unchanged — every pointer must be a live device allocation of
/// the aligned leg's shapes on `stream`.
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
