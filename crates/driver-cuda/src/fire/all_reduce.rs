//! `comm/custom_all_reduce.cu`'s LIFECYCLE, in Rust: peer access, the IPC
//! handle exchange, the workspace slabs, and the destructor that closes what
//! they opened.
//!
//! The `.cu` was 664 lines with **zero** `__global__` and **zero** `<<<>>>` —
//! named for linkage, not for content. Exactly **two lines** of it reached
//! device text, `impl_->allreduce<__nv_bfloat16>` at `:614-620` and
//! `allreduce_fusion_kernel_launcher` at `:157-162`, and both live in headers
//! this repository does not carry.
//!
//! # The split, and which half this is
//!
//! The whole host program came here first. `.wiki/kernel-x/refactor-plan.md`
//! §6.3 then divided it, and the dividing question was **what owns
//! something**:
//!
//! * **[`kernels_cuda::comm`]** took the half a LAUNCH reads — the
//!   240-point template cross product as data, the two `switch`es that pick a
//!   point out of it, the `AllReduceFusionParams` mirror, both refusals and
//!   both bodies. None of it touches a device; the tests that came with it
//!   said so in as many words. `comm::all_reduce_bf16` and
//!   `comm::all_reduce_residual_rmsnorm_bf16` are `driver_bound!` rows
//!   derived from `fn`s there. **Its header carries the `kernels.def`
//!   measurement, the 240/24/1 arithmetic and the JIT argument** — this file
//!   no longer restates any of it.
//! * **This file** kept everything with a lifetime: peer-access enablement,
//!   the IPC handle memo, the `Signal` + staging slab, the `RankData` slab,
//!   the fusion plane's four allocations and its Lamport initialisation,
//!   [`CustomAllReduce::register_buffer`],
//!   [`CustomAllReduce::register_graph_buffers`] and [`Drop`]. Three other
//!   processes hold the far end of those handles.
//!
//! The line is not a preference. Every function here reports through
//! [`crate::error::Error`], which carries a `cudaError` and a `String`;
//! `kernels-cuda` can name neither, and `kernels::Refusal` deliberately
//! carries no driver code. Moving the lifecycle would have meant deciding
//! what a failed IPC exchange IS, which is a design question and not a port.
//!
//! [`CustomAllReduce::plane`] is the seam: the five facts a launch reads off
//! this side, `Copy`, filled per call.
//!
//! # What was missing has landed
//!
//! The two launch points named `flashinfer/comm/vllm_custom_all_reduce.cuh`
//! and `flashinfer/comm/trtllm_allreduce_fusion.cuh`, which were CPM-*fetched*
//! at configure time and not vendored, so both bodies answered a
//! `Decline::NoDeviceText`. Both headers are internalised now at
//! `csrc/src/attn/flashinfer/comm/`, `kernels/comm/all_reduce.cuh` is the
//! root over them, and `kernels_cuda::comm::CAN_LAUNCH` is `true`. That
//! variant is gone with the absence it described.
//!
//! What that changed on THIS side is one field and one method.
//! [`CustomAllReduce::plane`] used to carry the group size, the rank and the
//! fusion workspace, which is all the FUSED launcher reads. The PLAIN one
//! reads three more things — the peer `Signal*` array by value, this rank's
//! own `Signal*`, and a `RankData*` naming the input's eight peer addresses —
//! so [`PeerPlane`] is the widening and [`CustomAllReduce::plane_for`] is
//! where the per-call one is resolved.
//!
//! # What got BETTER in the crossing, and it stayed here
//!
//! `custom_all_reduce.cu:340-342` initialised the Lamport buffer by launching
//! `flashinfer::trtllm_allreduce::lamportInitialize<__nv_bfloat16>`, which
//! writes negative zero into every slot. Negative zero in bf16 is `0x8000`, a
//! 16-bit pattern and not a byte pattern, so `cudaMemset` cannot express it —
//! but `cuMemsetD16_v2` can, exactly, and it is a driver-API call with no
//! device text behind it. See [`CustomAllReduce::new`]. That is one launch
//! point removed rather than deferred, and it is why the count above is two
//! and not three.
//!
//! # Two leaks the C++ had, closed by the crossing
//!
//! `custom_all_reduce.cu:520-538` and `:355-375` opened peer handles with
//! bare `cudaIpcOpenMemHandle` calls and recorded them nowhere, so the
//! destructor (`:403-427`, which walks only `signal_peers_`) could not close
//! them. Every open here goes through `CustomAllReduce::open_ipc_handle`,
//! which memoises by handle bytes, and [`Drop`] closes the memo. A throwing
//! C++ constructor also freed nothing; [`CustomAllReduce::new`] builds the
//! value first and lets [`Drop`] run on the error path.
//!
//! # Reachability today
//!
//! **The missing arm is written.** `bind/arms/comm.rs` binds both symbols and
//! reaches this file's [`CustomAllReduce`] through [`ResidentPlane`], the
//! thread-local the shell publishes its plane into;
//! `serve::load::build_tp_plane` is what constructs one, out of
//! `layout::rendezvous::tp_host_allgather`. So every line below is now
//! reachable BY CONSTRUCTION, and so is a fire that gets past it:
//! `kernels_cuda::comm::CAN_LAUNCH` is `true`, so
//! `serve::load::tp_serving_refusal` no longer turns `tp_size > 1` away at
//! `create` — it refuses a world size no plane can be built for, and a group
//! with no key, and nothing else.
//!
//! **Nothing in this module has been run against a second GPU, and that is
//! now the only thing standing between this and a working collective.** The
//! box this was written on has one, so `enable_peer_access`, the IPC
//! exchange, `build_fusion`'s Lamport initialisation, [`Drop`],
//! [`CustomAllReduce::plane_for`]'s slot arithmetic and every launch
//! `kernels_cuda::comm` makes out of them are correct by reading and by their
//! C++ ancestry, and by nothing else. A two-rank fire is the first thing that
//! would test any of it, and this repository has never run one.
//!
//! [`kernels_cuda::comm`]: kernels_cuda::comm

// `initialise` prints ONE line to stderr at construction, which the archive
// did at `custom_all_reduce.cu:398-404` and which is the only trace a
// deployment gets that the P2P plane came up at all. `serve/mod.rs` and
// `fire/attn_score.rs` carry the same allow for the same reason.
#![allow(clippy::print_stderr)]

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::fmt;
use std::sync::Arc;

use cudarc::driver::sys::{
    CUdeviceptr, CUpointer_attribute, cuMemsetD16_v2, cuPointerGetAttribute,
};
use cudarc::runtime::sys::{
    cudaDeviceCanAccessPeer, cudaDeviceEnablePeerAccess, cudaDeviceSynchronize, cudaError,
    cudaFree, cudaGetDevice, cudaGetLastError, cudaIpcCloseMemHandle, cudaIpcGetMemHandle,
    cudaIpcMemHandle_t, cudaIpcMemLazyEnablePeerAccess, cudaIpcOpenMemHandle, cudaMalloc,
    cudaMemcpy, cudaMemcpyKind, cudaMemset, cudaStream_t, cudaStreamCaptureStatus,
    cudaStreamIsCapturing,
};

use crate::error::{Error, check_cu, check_rt, ignore_in_drop};

// ── the constants the archive carried, each cited to its line ────────────

/// `sizeof(vllm::Signal)` — 3,456 bytes.
///
/// `flashinfer/comm/vllm_custom_all_reduce.cuh:52-60`:
///
/// ```text
///   struct Signal {
///     alignas(128) FlagType self_counter[kMaxBlocks][8];      // 36*8*4 = 1152
///     alignas(128) FlagType peer_counter[2][kMaxBlocks][8];   // 2*36*8*4 = 2304
///   };
/// ```
///
/// with `kMaxBlocks = 36` (`:46`) and `using FlagType = uint32_t` (`:51`).
/// Both members are already whole multiples of the 128-byte alignment, so
/// the struct needs no tail padding and the sum is exact. **This number is
/// an ABI fact of a header this tree does not vendor**, which is why it is
/// quoted here with the derivation rather than referenced.
pub const SIGNAL_BYTES: usize = 1152 + 2304;

// `MAX_BLOCKS` AND `ALL_REDUCE_THREADS` STOOD HERE and are `comm`'s, because
// they are the launch RECTANGLE and the launch descended. They are re-exported
// below with the rest of it. The constants that remain in this file are the
// ones the CONSTRUCTOR spends -- the signal slab, the rank-data slot width,
// the Lamport cap and the 2 MiB fusion alignment -- which is the same line the
// whole split is drawn on.

/// `sizeof(vllm::RankData)` — `vllm_custom_all_reduce.cuh:62-64`,
/// `struct __align__(16) RankData { void* ptrs[8]; }`.
pub const RANK_DATA_BYTES: usize = 8 * 8;

/// `custom_all_reduce.hpp:69` — the default `max_bytes`, 8 MiB.
pub const DEFAULT_MAX_BYTES: usize = 8 * 1024 * 1024;

/// `custom_all_reduce.hpp:70` — the default `rank_data_bytes`, 8 MiB.
///
/// `custom_all_reduce.cu:302` calls that *"enough for ~131k graph
/// addresses"*, which is `8 MiB / 64 B` exactly.
pub const DEFAULT_RANK_DATA_BYTES: usize = 8 * 1024 * 1024;

/// `custom_all_reduce.cu:313` — `constexpr std::size_t kAlign = 1ull << 21`.
///
/// Every fusion allocation is rounded up to 2 MiB, which is the large-page
/// granularity the Lamport protocol's address arithmetic assumes.
pub const FUSION_ALIGN: usize = 1 << 21;

/// `custom_all_reduce.cu:314` — `constexpr std::size_t kBarrierFlagCount = 256`.
pub const BARRIER_FLAG_COUNT: usize = 256;

/// `custom_all_reduce.cu:329-333` — the Lamport communication cap,
/// 2,145,386,496 bytes.
///
/// The archive wrote it as a bare literal. It is `2^31 - 2 MiB`: the largest
/// 2 MiB-aligned byte count that still fits a SIGNED 32-bit integer, which
/// is what the flag word at index 3 of the five-word flag block is
/// (`custom_all_reduce.cu:345-349` casts it to `std::uint32_t`, and the
/// device side reads it as an offset). A cap chosen for the width of the
/// field that carries it, and stated that way so the next reader does not
/// have to factor it.
pub const LAMPORT_COMM_CAP: usize = 2_145_386_496;

/// bf16 negative zero — the Lamport "empty slot" sentinel.
///
/// `custom_all_reduce.cu:339-342` got this by launching
/// `flashinfer::trtllm_allreduce::lamportInitialize<__nv_bfloat16>`. It is a
/// 16-bit fill, so `cuMemsetD16_v2` writes it with no kernel at all.
const LAMPORT_EMPTY_BF16: u16 = 0x8000;

// ── THE LAUNCH HALF STOOD HERE ──────────────────────────────────────────
//
// `FusionPattern`, `INSTANTIATED`, `NRANKS`, `Leaf`, `Instantiation`,
// `REACHED`, `resolve`, `Decline`, `AllReduce` and the
// `AllReduceFusionParams` mirror. All of it is `kernels_cuda::comm`; the
// module header above is the argument for the line the split was drawn on,
// and `comm`'s own header carries the 240-point measurement that used to be
// stated here.
//
// One thing is worth repeating at the point of removal rather than only at
// the top, because it is the test a reader will want to apply to the next
// split: **not one line of what left touched a device.** The module's own
// tests said so -- *"host arithmetic over an upstream struct layout, and none
// of it touches a device"* -- and four of the six went with the cross product
// while two stayed with the constants the constructor spends.

// FOUR NAMES AND NOT SIXTEEN. The first draft of this line re-exported the
// whole of `comm` -- the cross product, the constants, `resolve`, every
// instantiation type -- on the reasoning that a reader arriving here should
// not have to go looking. Measured: eleven of the sixteen had no use in
// `driver-cuda` outside this file's own prose, so re-exporting them would
// have been publishing a second address for a module that has one, and the
// second address is the one that goes stale. These four are the ones this
// file's own code speaks: the outcome, the refusal, and the two halves of the
// descriptor `CustomAllReduce::plane` fills.
pub use kernels_cuda::comm::{AllReduce, Decline, FusionPlane, PeerPlane, Plane};

/// `custom_all_reduce.cu:84-86` — `align_up(n, a)`.
fn align_up(n: usize, a: usize) -> usize {
    n.div_ceil(a) * a
}

/// `custom_all_reduce.cu:71-82` — `get_base_ptr`.
///
/// *"The vllm kernel needs the base pointer for the IPC handle exchange —
/// sub-allocation pointers won't round-trip across processes correctly."*
///
/// # Errors
///
/// The driver's code, when the pointer has no queryable range.
fn base_ptr(ptr: *const c_void) -> crate::error::Result<*mut c_void> {
    let mut base: *mut c_void = std::ptr::null_mut();
    check_cu(
        // SAFETY: `&mut base` is a live `void*` slot, which is the width
        // `CU_POINTER_ATTRIBUTE_RANGE_START_ADDR` writes.
        unsafe {
            cuPointerGetAttribute(
                std::ptr::addr_of_mut!(base).cast::<c_void>(),
                CUpointer_attribute::CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                ptr as usize as CUdeviceptr,
            )
        },
        "cuPointerGetAttribute(RANGE_START_ADDR)",
    )?;
    Ok(base)
}

/// `custom_all_reduce.cu:31-56` — `enable_peer_access`.
///
/// Idempotent: `cudaErrorPeerAccessAlreadyEnabled` is swallowed and the
/// sticky error reset, exactly as `:53` did with its `(void)cudaGetLastError()`.
///
/// **`peers` holds real device ORDINALS, never rank indices** — the archive's
/// warning at `:28-30`, kept because it is the kind of mistake that works on
/// every single-group box and corrupts the second group on a four-GPU one:
/// *"a TP group is not necessarily devices 0..world_size-1 (a second group on
/// a 4-GPU box runs on devices 2 and 3)"*.
///
/// # Errors
///
/// [`Error::Invalid`] naming the ordered pair, when peer access is
/// unavailable or cannot be enabled.
fn enable_peer_access(self_device: i32, peers: &[i32]) -> crate::error::Result<()> {
    for &peer in peers {
        if peer == self_device {
            continue;
        }
        let mut can_access: c_int = 0;
        // SAFETY: both ordinals are plain integers; the out-parameter is live.
        let can_err = unsafe { cudaDeviceCanAccessPeer(&mut can_access, self_device, peer) };
        if can_err != cudaError::cudaSuccess || can_access == 0 {
            return Err(Error::invalid(
                "custom_all_reduce",
                format!(
                    "peer access unavailable from {self_device} to {peer}{}",
                    if can_err == cudaError::cudaSuccess {
                        String::new()
                    } else {
                        format!(": {can_err:?}")
                    }
                ),
            ));
        }
        // SAFETY: as above.
        let err = unsafe { cudaDeviceEnablePeerAccess(peer, 0) };
        if err != cudaError::cudaSuccess && err != cudaError::cudaErrorPeerAccessAlreadyEnabled {
            return Err(Error::invalid(
                "custom_all_reduce",
                format!("cudaDeviceEnablePeerAccess {self_device}->{peer} failed: {err:?}"),
            ));
        }
        // Reset the sticky error -- `custom_all_reduce.cu:53`.
        // SAFETY: no arguments, no aliasing.
        let _ = unsafe { cudaGetLastError() };
    }
    Ok(())
}

/// `custom_all_reduce.cu:58-69` — `has_full_peer_access`.
///
/// Both directions of every ordered pair, because peer access is not
/// symmetric on every topology.
fn has_full_peer_access(group_devices: &[i32]) -> bool {
    for &src in group_devices {
        for &dst in group_devices {
            if src == dst {
                continue;
            }
            let mut can_access: c_int = 0;
            // SAFETY: plain integers and a live out-parameter.
            if unsafe { cudaDeviceCanAccessPeer(&mut can_access, src, dst) }
                != cudaError::cudaSuccess
            {
                return false;
            }
            if can_access == 0 {
                return false;
            }
        }
    }
    true
}

/// `cudaIpcMemHandle_t` from its 64 opaque bytes.
fn to_handle(bytes: &[u8; 64]) -> cudaIpcMemHandle_t {
    let mut handle = cudaIpcMemHandle_t { reserved: [0; 64] };
    for (dst, src) in handle.reserved.iter_mut().zip(bytes.iter()) {
        *dst = *src as c_char;
    }
    handle
}

/// The 64 opaque bytes of a `cudaIpcMemHandle_t`.
///
/// Handles travel as bytes here, not as the struct, because they are the
/// payload of an all-gather written by the caller and the caller has no
/// business naming a CUDA type. It is also what makes the handle hashable:
/// upstream's `open_ipc_handle` memo is keyed on the raw bytes for exactly
/// this reason.
fn from_handle(handle: &cudaIpcMemHandle_t) -> [u8; 64] {
    let mut bytes = [0u8; 64];
    for (dst, src) in bytes.iter_mut().zip(handle.reserved.iter()) {
        *dst = *src as u8;
    }
    bytes
}

// ── the seam ─────────────────────────────────────────────────────────────

/// One bootstrap-time all-gather over HOST buffers.
///
/// `send` is this rank's contribution; `recv` is `send.len() *
/// world_size` bytes, **rank-major**. `custom_all_reduce.hpp:55` states the
/// same contract in C++.
pub type Allgather = Arc<dyn Fn(&[u8], &mut [u8]) + Send + Sync>;

/// What this needs from the collective, and nothing more —
/// `custom_all_reduce.hpp:39-57`, kept whole because the reasoning is the
/// reason the type exists. It is quoted verbatim, so the `kernels-cuda` it
/// names is the ARCHIVE crate it was written against, deleted at
/// `85c6c674b`.
///
/// > The wrapper used to take an `NcclComm&`. It reads exactly two things off
/// > it — the world size, and one bootstrap-time all-gather of IPC handles —
/// > and taking the class instead of those two made a compute kernel depend
/// > on the driver's comm plane. It is a compute kernel:
/// > `all_reduce_residual_rmsnorm_bf16` fuses a reduction with a residual add
/// > and an RMSNorm, and the unfused halves of that live in `kernels-cuda`.
/// >
/// > So the seam is a callback. `gather` takes HOST buffers; whatever H2D
/// > dance a given collective needs is the caller's business, which is where
/// > NCCL knowledge belongs. Who decides custom-vs-NCCL by message size stays
/// > the caller's too — `can_handle()` only reports.
#[derive(Clone)]
pub struct HostAllgather {
    /// This rank's index in the group.
    pub rank: i32,
    /// The group size.
    pub world_size: i32,
    /// The collective.
    pub gather: Allgather,
}

impl fmt::Debug for HostAllgather {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostAllgather")
            .field("rank", &self.rank)
            .field("world_size", &self.world_size)
            .finish_non_exhaustive()
    }
}

/// The constructor's remaining arguments — `custom_all_reduce.hpp:66-72`.
#[derive(Debug, Clone)]
pub struct Config {
    /// Skip IPC entirely and exchange raw pointers, for a single-process
    /// multi-GPU deployment. `custom_all_reduce.cu:264-272`.
    pub same_process: bool,
    /// **The CUDA device ordinal of every rank, indexed by rank** —
    /// `custom_all_reduce.hpp:62-65`. Required, and required to be
    /// `world_size` long.
    pub group_devices: Vec<i32>,
    /// The largest message the plain P2P path will take.
    pub max_bytes: usize,
    /// The `RankData` slab. Floored at one slot.
    pub rank_data_bytes: usize,
    /// Zero disables the fused landing entirely.
    pub fusion_max_tokens: i32,
    /// Zero disables the fused landing entirely.
    pub fusion_hidden: i32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            same_process: false,
            group_devices: Vec::new(),
            max_bytes: DEFAULT_MAX_BYTES,
            rank_data_bytes: DEFAULT_RANK_DATA_BYTES,
            fusion_max_tokens: 0,
            fusion_hidden: 0,
        }
    }
}

/// The fusion plane, built only for `world_size == 2` with both fusion
/// dimensions positive — `custom_all_reduce.cu:308-397`.
#[derive(Debug)]
struct Fusion {
    /// `fusion_buffers_` — `[buffer, flag, lamport]`, `:336-338`.
    buffers: [*mut c_void; 3],
    /// `fusion_workspace_dev_` — the `3 * world + 1` pointer array, `:391`.
    workspace_dev: *mut c_void,
    /// `fusion_flag_dev_` — the five-word flag block, `:344`.
    flag_dev: *mut c_void,
    /// `fusion_max_tokens_`.
    max_tokens: i32,
    /// `fusion_hidden_`.
    hidden: i32,
    // `lamport_comm_bytes` STOOD HERE, mirroring the archive's
    // `fusion_lamport_comm_bytes_` (`:329-333`). The port keeps it as a LOCAL
    // in `ensure_fusion`, where it sizes the Lamport span and goes into the
    // flag block's fourth word; storing a second copy on the state gave it a
    // reader-free lifetime, and a mirrored field is only worth its risk when
    // something later asks it.
}

// ── the lifecycle ────────────────────────────────────────────────────────

/// The custom P2P all-reduce's whole host state —
/// `custom_all_reduce.hpp:59-141` and the half of `vllm::CustomAllreduce`
/// that never launched anything.
///
/// # Why the vllm half is absorbed rather than wrapped
///
/// The C++ held a `std::unique_ptr<vllm::CustomAllreduce> impl_` and reached
/// THROUGH it for state it also duplicated: `registered_bases_` beside
/// `impl_->buffers_`, and `register_graph_buffers` poking
/// `impl_->d_rank_data_base_` and `impl_->graph_unreg_buffers_` directly
/// (`custom_all_reduce.cu:557-575`). Every one of those members is plain
/// host bookkeeping — `open_ipc_handle`, `get_graph_buffer_ipc_meta`,
/// `check_rank_data_capacity`, `register_buffer(void**)` and
/// `register_graph_buffers` are host functions in upstream too, and only
/// `allreduce<T>()` launches. Absorbing them removes the duplication and
/// leaves exactly one thing on the other side of the seam: a launch.
///
/// # Two leaks the C++ had, closed by the crossing
///
/// `custom_all_reduce.cu:520-538` and `:355-375` opened peer handles with
/// bare `cudaIpcOpenMemHandle` calls and recorded them nowhere, so the
/// destructor (`:403-427`, which walks only `signal_peers_`) could not close
/// them. Every open here goes through [`CustomAllReduce::open_ipc_handle`],
/// which memoises by handle bytes — upstream's own shape — and [`Drop`]
/// closes the memo. A throwing C++ constructor also freed nothing;
/// [`CustomAllReduce::new`] builds the value first and lets [`Drop`] run on
/// the error path.
///
/// # Not `Send`, not `Sync`
///
/// The raw device pointers make it neither, which matches the C++: the class
/// is non-copyable and move-only, and every method assumes the calling thread
/// holds the device context the constructor ran on.
#[derive(Debug)]
pub struct CustomAllReduce {
    rank: i32,
    world_size: i32,
    fully_connected: bool,
    same_process: bool,
    max_bytes: usize,
    ag: HostAllgather,

    /// `signal_self_` — `sizeof(Signal) + max_bytes` of zeroed device
    /// memory, `custom_all_reduce.cu:259-261`.
    signal_self: *mut c_void,
    /// `signal_peers_`, one per rank, self included.
    signal_peers: Vec<*mut c_void>,

    /// `rank_data_` — the `RankData` slab.
    rank_data: *mut c_void,
    /// `d_rank_data_end_ - d_rank_data_base_` at construction, in slots.
    rank_data_slots: usize,
    /// upstream's `d_rank_data_base_`, as an index rather than a pointer.
    rank_data_next: usize,

    /// upstream's `buffers_` merged with the wrapper's `registered_bases_`:
    /// local base address -> the device `RankData*` it was registered into.
    buffers: HashMap<usize, *mut c_void>,
    /// upstream's `ipc_handles_`, keyed on the handle's 64 opaque bytes.
    ipc_handles: HashMap<[u8; 64], *mut c_void>,
    /// upstream's `graph_unreg_buffers_`.
    ///
    /// Appended to by the vllm host launcher when an all-reduce runs on a
    /// CAPTURING stream against a buffer that was never registered; drained
    /// by [`CustomAllReduce::register_graph_buffers`]. **Nothing appends to
    /// it in this tree**, because the launcher is the one thing that has not
    /// crossed, and a decline must not append: a deferred registration for a
    /// launch that never happened would bind the next real one to the wrong
    /// slot.
    graph_unreg_buffers: Vec<*mut c_void>,

    fusion: Option<Fusion>,
}

impl CustomAllReduce {
    /// `custom_all_reduce.cu:222-401`, the constructor.
    ///
    /// # Errors
    ///
    /// Every `throw` of the C++ constructor, as an [`Error`]: an unsupported
    /// world size, a missing or mis-sized `group_devices`, unavailable peer
    /// access, and any failing CUDA call.
    pub fn new(ag: HostAllgather, cfg: &Config) -> crate::error::Result<Self> {
        let mut me = Self {
            rank: ag.rank,
            world_size: ag.world_size,
            fully_connected: false,
            same_process: cfg.same_process,
            max_bytes: cfg.max_bytes,
            ag,
            signal_self: std::ptr::null_mut(),
            signal_peers: Vec::new(),
            rank_data: std::ptr::null_mut(),
            rank_data_slots: 0,
            rank_data_next: 0,
            buffers: HashMap::new(),
            ipc_handles: HashMap::new(),
            graph_unreg_buffers: Vec::new(),
            fusion: None,
        };
        // Built first so that a failure part-way returns through `Drop` and
        // frees whatever was allocated. The C++ constructor threw and freed
        // nothing.
        me.initialise(cfg)?;
        Ok(me)
    }

    fn initialise(&mut self, cfg: &Config) -> crate::error::Result<()> {
        // `custom_all_reduce.cu:231-235`.
        if self.world_size < 2 || self.world_size > 8 || (self.world_size % 2) != 0 {
            return Err(Error::invalid(
                "custom_all_reduce",
                format!(
                    "the vllm kernel supports world_size in {{2,4,6,8}}; got {}",
                    self.world_size
                ),
            ));
        }

        // `:239-251`.
        let mut dev: c_int = 0;
        // SAFETY: a live out-parameter.
        check_rt(unsafe { cudaGetDevice(&mut dev) }, "cudaGetDevice")?;
        if cfg.group_devices.is_empty() {
            return Err(Error::invalid("custom_all_reduce", "group device ordinals are required"));
        }
        if cfg.group_devices.len() != self.world_size as usize {
            return Err(Error::invalid(
                "custom_all_reduce",
                format!(
                    "group device list has {} entries for world_size {}",
                    cfg.group_devices.len(),
                    self.world_size
                ),
            ));
        }
        enable_peer_access(dev, &cfg.group_devices)?;

        // `:252-254`: the vllm kernel handles larger TP groups only when
        // every rank has direct peer access to every other rank.
        self.fully_connected = self.world_size <= 2 || has_full_peer_access(&cfg.group_devices);

        // `:256-261`. The staging region past the `Signal` is what
        // flashinfer's 2-stage algorithm needs; TP=2 takes the 1-stage path
        // and never touches it, and the layout is matched anyway so this
        // wrapper stays valid for fully-connected larger groups.
        let signal_bytes = SIGNAL_BYTES + self.max_bytes;
        // SAFETY: a live out-parameter and a positive size.
        check_rt(unsafe { cudaMalloc(&mut self.signal_self, signal_bytes) }, "cudaMalloc(signal)")?;
        // SAFETY: `signal_self` now addresses `signal_bytes` writable bytes.
        check_rt(unsafe { cudaMemset(self.signal_self, 0, signal_bytes) }, "cudaMemset(signal)")?;

        // `:263-297`, the signal exchange.
        self.signal_peers = self.exchange_pointers(self.signal_self)?;

        // `:299-304`. vLLM uses 8 MiB, "enough for ~131k graph addresses".
        let rank_data_bytes = cfg.rank_data_bytes.max(RANK_DATA_BYTES);
        // SAFETY: a live out-parameter and a positive size.
        check_rt(
            unsafe { cudaMalloc(&mut self.rank_data, rank_data_bytes) },
            "cudaMalloc(rank_data)",
        )?;
        self.rank_data_slots = rank_data_bytes / RANK_DATA_BYTES;
        self.rank_data_next = 0;

        if self.world_size == 2 && cfg.fusion_max_tokens > 0 && cfg.fusion_hidden > 0 {
            self.build_fusion(cfg)?;
        }

        // `:398-404`, kept verbatim. One line on stderr at construction, and
        // the four facts a support ticket needs.
        eprintln!(
            "[custom_all_reduce] initialised (world={}, rank={}, mode={}, fully_connected={})",
            self.world_size,
            self.rank,
            if self.same_process { "same-process" } else { "ipc" },
            if self.fully_connected { "yes" } else { "no" }
        );
        Ok(())
    }

    /// One all-gather of `local`'s address across the group, by whichever of
    /// the two mechanisms this deployment configured.
    ///
    /// `custom_all_reduce.cu:263-297` for the signal and `:355-375` for the
    /// fusion buffers were the same twelve lines twice; this is them once.
    /// In `same_process` mode the raw `u64` crosses; otherwise a
    /// `cudaIpcMemHandle_t` does and each peer's is opened.
    fn exchange_pointers(&mut self, local: *mut c_void) -> crate::error::Result<Vec<*mut c_void>> {
        let world = self.world_size as usize;
        if self.same_process {
            let send = (local as usize as u64).to_ne_bytes();
            let gathered = self.allgather(&send);
            let mut out = Vec::with_capacity(world);
            for r in 0..world {
                let mut word = [0u8; 8];
                word.copy_from_slice(&gathered[r * 8..r * 8 + 8]);
                out.push(u64::from_ne_bytes(word) as usize as *mut c_void);
            }
            return Ok(out);
        }

        let mut self_handle = cudaIpcMemHandle_t { reserved: [0; 64] };
        // SAFETY: `local` is a base allocation of this process.
        check_rt(unsafe { cudaIpcGetMemHandle(&mut self_handle, local) }, "cudaIpcGetMemHandle")?;
        let gathered = self.allgather(&from_handle(&self_handle));

        let mut out = Vec::with_capacity(world);
        for r in 0..world {
            if r == self.rank as usize {
                out.push(local);
                continue;
            }
            let mut key = [0u8; 64];
            key.copy_from_slice(&gathered[r * 64..r * 64 + 64]);
            out.push(self.open_ipc_handle(key)?);
        }
        Ok(out)
    }

    /// The collective, with the rank-major receive buffer sized here so no
    /// caller can get it wrong.
    fn allgather(&self, send: &[u8]) -> Vec<u8> {
        let mut recv = vec![0u8; send.len() * self.world_size as usize];
        (self.ag.gather)(send, &mut recv);
        recv
    }

    /// upstream `vllm::CustomAllreduce::open_ipc_handle`, memoised by the
    /// handle's bytes.
    ///
    /// The memo is not an optimisation: `cudaIpcOpenMemHandle` on a handle
    /// already open in this process returns the SAME mapping and increments
    /// nothing a second `cudaIpcCloseMemHandle` would balance, so opening
    /// twice and closing twice is a double free. Keying on the bytes is how
    /// upstream avoids it and how [`Drop`] here knows the exact set to close.
    fn open_ipc_handle(&mut self, key: [u8; 64]) -> crate::error::Result<*mut c_void> {
        if let Some(existing) = self.ipc_handles.get(&key) {
            return Ok(*existing);
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `key` is 64 bytes produced by `cudaIpcGetMemHandle` on a
        // peer, and the out-parameter is live.
        check_rt(
            unsafe {
                cudaIpcOpenMemHandle(
                    &mut ptr,
                    to_handle(&key),
                    cudaIpcMemLazyEnablePeerAccess as c_uint,
                )
            },
            "cudaIpcOpenMemHandle",
        )?;
        self.ipc_handles.insert(key, ptr);
        Ok(ptr)
    }

    /// `custom_all_reduce.cu:308-397` — the fusion plane's four allocations,
    /// its Lamport initialisation and its device workspace.
    fn build_fusion(&mut self, cfg: &Config) -> crate::error::Result<()> {
        let world = self.world_size as usize;
        let max_tokens = cfg.fusion_max_tokens;
        let hidden = cfg.fusion_hidden;

        // `:315-326`. `elem_bytes` is `sizeof(__nv_bfloat16)`.
        const ELEM_BYTES: usize = 2;
        let span = world * max_tokens as usize * hidden as usize * ELEM_BYTES;
        let buffer_bytes = align_up(span, FUSION_ALIGN);
        let flag_bytes = align_up(world * BARRIER_FLAG_COUNT * 4, FUSION_ALIGN);
        let lamport_comm_bytes = span.min(LAMPORT_COMM_CAP);
        let lamport_bytes = align_up(lamport_comm_bytes * 3, FUSION_ALIGN);

        let mut buffers = [std::ptr::null_mut::<c_void>(); 3];
        for (slot, bytes) in buffers.iter_mut().zip([buffer_bytes, flag_bytes, lamport_bytes]) {
            // SAFETY: a live out-parameter and a positive size.
            check_rt(unsafe { cudaMalloc(slot, bytes) }, "cudaMalloc(fusion)")?;
        }

        let mut flag_dev: *mut c_void = std::ptr::null_mut();
        // SAFETY: a live out-parameter.
        check_rt(unsafe { cudaMalloc(&mut flag_dev, 5 * 4) }, "cudaMalloc(fusion flags)")?;

        // Everything allocated: park it on `self` so `Drop` owns it from
        // here, before anything else can fail.
        self.fusion = Some(Fusion {
            buffers,
            workspace_dev: std::ptr::null_mut(),
            flag_dev,
            max_tokens,
            hidden,
        });

        // `:339-342`, and the one place this port does LESS device work than
        // the archive. `lamportInitialize<__nv_bfloat16>` launched a kernel
        // to write bf16 negative zero into every slot; negative zero is the
        // 16-bit pattern `0x8000`, so `cuMemsetD16_v2` writes it directly and
        // there is no device text to compile, fetch or vendor. The archive
        // passed a null stream and synchronised at `:396`; this is the
        // synchronous form, which is the same thing without the pair.
        // SAFETY: `buffers[2]` addresses `lamport_bytes` writable bytes, and
        // `lamport_bytes` is 2 MiB-aligned so the 16-bit count is exact.
        check_cu(
            unsafe {
                cuMemsetD16_v2(
                    buffers[2] as usize as CUdeviceptr,
                    LAMPORT_EMPTY_BF16,
                    lamport_bytes / ELEM_BYTES,
                )
            },
            "cuMemsetD16_v2(lamport)",
        )?;

        // `:345-351`. Index 3 carries the Lamport communication size; the
        // other four words start at zero.
        let flags: [u32; 5] = [0, 0, 0, lamport_comm_bytes as u32, 0];
        // SAFETY: `flag_dev` addresses 20 writable bytes and `flags` is 20
        // bytes of initialised host memory.
        check_rt(
            unsafe {
                cudaMemcpy(
                    flag_dev,
                    flags.as_ptr().cast::<c_void>(),
                    std::mem::size_of_val(&flags),
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            },
            "cudaMemcpy(fusion flags)",
        )?;

        // `:353-390`. `3 * world + 1` pointers: every rank's view of each of
        // the three buffers, rank-major within buffer, then the local flag
        // block.
        let mut workspace: Vec<*mut c_void> = Vec::with_capacity(3 * world + 1);
        for i in 0..3 {
            let peers = self.exchange_pointers(buffers[i])?;
            workspace.extend_from_slice(&peers);
        }
        workspace.push(flag_dev);

        let mut workspace_dev: *mut c_void = std::ptr::null_mut();
        // SAFETY: a live out-parameter and a positive size.
        check_rt(
            unsafe {
                cudaMalloc(&mut workspace_dev, workspace.len() * std::mem::size_of::<*mut c_void>())
            },
            "cudaMalloc(fusion workspace)",
        )?;
        if let Some(fusion) = self.fusion.as_mut() {
            fusion.workspace_dev = workspace_dev;
        }
        // SAFETY: both sides address `workspace.len()` pointers.
        check_rt(
            unsafe {
                cudaMemcpy(
                    workspace_dev,
                    workspace.as_ptr().cast::<c_void>(),
                    workspace.len() * std::mem::size_of::<*mut c_void>(),
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            },
            "cudaMemcpy(fusion workspace)",
        )?;
        // `:396`. The archive synchronised here because `lamportInitialize`
        // was asynchronous on the null stream; kept because the workspace
        // copy is too and every peer must see it before the first fire.
        // SAFETY: no arguments.
        check_rt(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")?;
        Ok(())
    }

    /// This rank's index.
    #[must_use]
    pub const fn rank(&self) -> i32 {
        self.rank
    }

    /// The group size.
    #[must_use]
    pub const fn world_size(&self) -> i32 {
        self.world_size
    }

    /// Whether every ordered pair of the group has direct peer access —
    /// `custom_all_reduce.cu:252-254`.
    #[must_use]
    pub const fn fully_connected(&self) -> bool {
        self.fully_connected
    }

    /// The largest message the plain P2P path will take.
    #[must_use]
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Whether the fusion plane was built — `custom_all_reduce.cu:308`.
    #[must_use]
    pub const fn has_fusion(&self) -> bool {
        self.fusion.is_some()
    }

    /// `custom_all_reduce.cu:498-541` — `register_buffer`.
    ///
    /// `buf_bytes` was `/*buf_bytes*/` in the C++ too (`:499`): the vllm
    /// kernel registers a BASE address and does its own offset arithmetic
    /// against the registered `RankData`, so the extent never mattered. It
    /// stays in the signature because the caller has it and the day a bounds
    /// check becomes possible is the day it is wanted.
    ///
    /// Idempotent per base address, which is what makes it safe to call on
    /// every step.
    ///
    /// # Errors
    ///
    /// An unresolvable base pointer, a failing IPC exchange, or an exhausted
    /// `RankData` slab.
    pub fn register_buffer(
        &mut self,
        buf: *mut c_void,
        _buf_bytes: usize,
    ) -> crate::error::Result<()> {
        let self_base = base_ptr(buf)?;
        if self.buffers.contains_key(&(self_base as usize)) {
            return Ok(());
        }
        let peer_bases = self.exchange_pointers(self_base)?;
        let slot = self.write_rank_data(&[peer_bases])?;
        self.buffers.insert(self_base as usize, slot);
        Ok(())
    }

    /// upstream `vllm::CustomAllreduce::check_rank_data_capacity` +
    /// the `cudaMemcpy` into `d_rank_data_base_`, for a run of `rows`
    /// consecutive slots.
    ///
    /// One function because both callers wanted exactly this and the C++ had
    /// it twice — once inside `impl_->register_buffer` and once open-coded at
    /// `custom_all_reduce.cu:565-575`, where the wrapper reached past its
    /// own abstraction to advance `impl_->d_rank_data_base_` by hand.
    ///
    /// Returns the device address of the FIRST slot written.
    fn write_rank_data(&mut self, rows: &[Vec<*mut c_void>]) -> crate::error::Result<*mut c_void> {
        let n = rows.len();
        if self.rank_data_next + n > self.rank_data_slots {
            // upstream threw "Rank data buffer is overflowed by X"; here it
            // is the shared exhaustion error, which names the same two
            // numbers by naming the want.
            return Err(Error::exhausted(
                "custom_all_reduce rank_data slots",
                self.rank_data_next + n,
            ));
        }
        // `RankData` is `void* ptrs[8]` (`vllm_custom_all_reduce.cuh:62-64`)
        // regardless of world size; the tail past `world_size` is padding the
        // kernel never reads, and is zeroed here rather than left undefined.
        let mut flat: Vec<*mut c_void> = vec![std::ptr::null_mut(); n * 8];
        for (i, row) in rows.iter().enumerate() {
            for (r, ptr) in row.iter().enumerate() {
                flat[i * 8 + r] = *ptr;
            }
        }
        let first =
            (self.rank_data as usize + self.rank_data_next * RANK_DATA_BYTES) as *mut c_void;
        // SAFETY: the capacity check above proves `n` slots fit, and `flat`
        // is exactly `n * RANK_DATA_BYTES` bytes of initialised host memory.
        check_rt(
            unsafe {
                cudaMemcpy(
                    first,
                    flat.as_ptr().cast::<c_void>(),
                    n * RANK_DATA_BYTES,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            },
            "cudaMemcpy(rank_data)",
        )?;
        self.rank_data_next += n;
        Ok(first)
    }

    /// upstream `vllm::CustomAllreduce::get_graph_buffer_ipc_meta`.
    ///
    /// Returns the concatenated 64-byte handles of every unregistered graph
    /// buffer's BASE allocation, and each buffer's byte offset within it.
    fn graph_buffer_ipc_meta(&self) -> crate::error::Result<(Vec<u8>, Vec<i64>)> {
        let n = self.graph_unreg_buffers.len();
        let mut handles = vec![0u8; n * 64];
        let mut offsets = vec![0i64; n];
        for (i, &ptr) in self.graph_unreg_buffers.iter().enumerate() {
            let base = base_ptr(ptr)?;
            offsets[i] = (ptr as usize as i64) - (base as usize as i64);
            let mut handle = cudaIpcMemHandle_t { reserved: [0; 64] };
            // SAFETY: `base` is a base allocation of this process.
            check_rt(
                unsafe { cudaIpcGetMemHandle(&mut handle, base) },
                "cudaIpcGetMemHandle(graph)",
            )?;
            handles[i * 64..(i + 1) * 64].copy_from_slice(&from_handle(&handle));
        }
        Ok((handles, offsets))
    }

    /// `custom_all_reduce.cu:543-601` — `register_graph_buffers`.
    ///
    /// Registers, in one collective, every buffer an all-reduce met on a
    /// CAPTURING stream and found unregistered. It is called once after a
    /// capture closes.
    ///
    /// **It is a no-op in this tree today**, and will be until the launcher
    /// crosses: [`CustomAllReduce::graph_unreg_buffers`] is fed by the vllm
    /// host launcher and by nothing else, and a decline must not feed it —
    /// see the field's own note. It is ported whole anyway because it is the
    /// half of the graph path that has no device text in it, and porting it
    /// later would mean reconstructing this collective from a deleted file.
    ///
    /// # The capture path WILL need this, and here is where the call goes
    ///
    /// Checked rather than assumed, because "does a bind arm ever see a
    /// capturing stream?" decides it and the answer is **yes**.
    /// `bind::run_captured` is not a separate dispatch surface — it walks the
    /// same launches and calls the same `bind`/`dispatch` as `run` — and it
    /// overwrites `ctx.stream` with the graph builder's stream per region, so
    /// `bind/arms/comm.rs` issues onto a capturing stream during a capture.
    /// [`CustomAllReduce::can_handle`] answers `Ok` immediately for a
    /// capturing stream WITHOUT the registration check (the address will be
    /// replayed, not dereferenced now), which is exactly the case this
    /// function settles afterwards.
    ///
    /// The capture is `fire::launch::capture_or_replay`'s, per fire rather
    /// than one-time: it opens whenever an `(R, N, class, model, lora)`
    /// bucket is cold or its epoch went stale. The call belongs between
    /// `scope.end()` and `graph.instantiate()` — after the capture closes,
    /// because the collective is not capturable, and before instantiate,
    /// because this writes the `RankData` slots the replayed addresses
    /// resolve through.
    ///
    /// Two things a future caller must not miss. `CaptureScope` also closes
    /// on `Drop` (the abandoned-capture path), and the two error arms after
    /// `scope.end()` take it — a capture abandoned with buffers queued would
    /// leave them queued into the next capture's registration. And that file
    /// is not this one's to edit here, which is why this is a note and not a
    /// call.
    ///
    /// # Errors
    ///
    /// A failing IPC exchange or an exhausted `RankData` slab.
    pub fn register_graph_buffers(&mut self) -> crate::error::Result<()> {
        let n = self.graph_unreg_buffers.len();
        if n == 0 {
            return Ok(());
        }
        let world = self.world_size as usize;

        if self.same_process {
            // `:552-577`. Gather every rank's raw pointers, buffer-minor.
            let mut send = Vec::with_capacity(n * 8);
            for &ptr in &self.graph_unreg_buffers {
                send.extend_from_slice(&(ptr as usize as u64).to_ne_bytes());
            }
            let gathered = self.allgather(&send);
            let mut rows: Vec<Vec<*mut c_void>> = Vec::with_capacity(n);
            for i in 0..n {
                let mut row = Vec::with_capacity(world);
                for r in 0..world {
                    // `:568-570`: rank-major outer, buffer-minor inner.
                    let idx = (r * n + i) * 8;
                    let mut word = [0u8; 8];
                    word.copy_from_slice(&gathered[idx..idx + 8]);
                    row.push(u64::from_ne_bytes(word) as usize as *mut c_void);
                }
                rows.push(row);
            }
            self.write_rank_data(&rows)?;
            self.graph_unreg_buffers.clear();
            return Ok(());
        }

        // `:579-600`, then upstream's `register_graph_buffers`.
        let (self_handles, self_offsets) = self.graph_buffer_ipc_meta()?;
        let all_handles = self.allgather(&self_handles);
        let mut offset_bytes = Vec::with_capacity(n * 8);
        for off in &self_offsets {
            offset_bytes.extend_from_slice(&off.to_ne_bytes());
        }
        let all_offsets = self.allgather(&offset_bytes);

        let handle_bytes = n * 64;
        let mut rows: Vec<Vec<*mut c_void>> = vec![Vec::with_capacity(world); n];
        for r in 0..world {
            for i in 0..n {
                if r == self.rank as usize {
                    rows[i].push(self.graph_unreg_buffers[i]);
                    continue;
                }
                let at = r * handle_bytes + i * 64;
                let mut key = [0u8; 64];
                key.copy_from_slice(&all_handles[at..at + 64]);
                let peer = self.open_ipc_handle(key)?;
                let mut word = [0u8; 8];
                let off_at = (r * n + i) * 8;
                word.copy_from_slice(&all_offsets[off_at..off_at + 8]);
                let offset = i64::from_ne_bytes(word);
                rows[i].push((peer as usize).wrapping_add(offset as usize) as *mut c_void);
            }
        }
        self.write_rank_data(&rows)?;
        self.graph_unreg_buffers.clear();
        Ok(())
    }

    /// `custom_all_reduce.cu:464-486` — `can_handle`, which returned `bool`.
    ///
    /// The `bool` is a [`Decline`] here, because every one of the eight
    /// `return false`s meant something different and the caller could not
    /// tell them apart. It is still a QUERY, not a refusal: the header's
    /// `:88-92` says *"above the threshold the kernel falls off NCCL on
    /// bandwidth, so we short-circuit and return false — caller should fall
    /// back to ncclAllReduce"*. A `Decline` from here is the caller's cue to
    /// use the collective, not an error.
    ///
    /// # Errors
    ///
    /// The [`Decline`] that the corresponding `return false` stood for.
    pub fn can_handle(
        &self,
        input: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> std::result::Result<(), Decline> {
        // `:467`.
        if input.is_null() {
            return Err(Decline::NullInput);
        }
        // `:469-471`. The 16-byte multiple is the kernel's vector width.
        if bytes == 0 || bytes > self.max_bytes || bytes % 16 != 0 {
            return Err(Decline::Bytes { bytes, max_bytes: self.max_bytes });
        }
        // `:473`.
        if self.world_size > 2 && !self.fully_connected {
            return Err(Decline::NotFullyConnected { world_size: self.world_size });
        }
        // `:475-479`. During capture the pointer query is meaningless --
        // the address will be replayed, not dereferenced now -- so the
        // registration check is deferred to `register_graph_buffers` and
        // capture answers YES immediately.
        let mut status = cudaStreamCaptureStatus::cudaStreamCaptureStatusNone;
        // SAFETY: `stream` is a `cudaStream_t` from the caller (null is the
        // legal default stream) and the out-parameter is live.
        if unsafe { cudaStreamIsCapturing(stream as cudaStream_t, &mut status) }
            != cudaError::cudaSuccess
        {
            return Err(Decline::CaptureUnknown);
        }
        if status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive {
            return Ok(());
        }
        // `:481-483`. A throwing `get_base_ptr` was caught and turned into
        // `false`; here the error is simply not distinguished from "not
        // registered", which is what the C++ meant by catching it.
        let Ok(base) = base_ptr(input) else {
            return Err(Decline::Unregistered);
        };
        if !self.buffers.contains_key(&(base as usize)) {
            return Err(Decline::Unregistered);
        }
        // `:485`. The measured crossover with NCCL, which is the only reason
        // this class is optional at all. Below it the P2P kernel wins on
        // latency; above it NCCL wins on bandwidth, and the wider the group
        // the sooner that happens.
        let crossover = if self.world_size <= 2 {
            self.max_bytes
        } else if self.world_size <= 4 {
            1 << 20
        } else {
            256 << 10
        };
        if bytes < crossover {
            Ok(())
        } else {
            Err(Decline::AboveCrossover { bytes, crossover, world_size: self.world_size })
        }
    }

    /// The five facts a LAUNCH reads off this instance.
    ///
    /// `can_fuse_residual_rmsnorm` and the two reduction bodies stood here and
    /// are `kernels_cuda::comm`'s. This is what replaces the `&mut self`
    /// they took: a `Copy` descriptor, filled per call, carrying the group
    /// size, this rank and — when the constructor built one — the fusion
    /// workspace's address and the two extents it was sized for.
    ///
    /// **Three of `Fusion`'s five fields are deliberately not in it.**
    /// `buffers` and `flag_dev` are what `build_fusion` allocated and
    /// initialised, and no launch reads either; carrying them would make this
    /// a mirror of the private struct instead of a statement of what a launch
    /// needs, and a mirror is a thing that can start disagreeing.
    ///
    /// [`CustomAllReduce::can_handle`] is NOT expressible through this and
    /// stayed a method, which is the second half of the same argument: it
    /// walks `self.buffers` for the registration check and queries the
    /// stream's capture state, so it is a function of the instance rather than
    /// of the plane.
    #[must_use]
    pub fn plane(&self) -> Plane {
        Plane {
            world_size: self.world_size,
            rank: self.rank,
            fusion: self.fusion.as_ref().map(|f| FusionPlane {
                workspace: f.workspace_dev,
                max_tokens: f.max_tokens,
                hidden: f.hidden,
            }),
            peers: PeerPlane {
                signals: self.signal_array(),
                self_signal: usize::try_from(self.rank)
                    .ok()
                    .and_then(|at| self.signal_peers.get(at).copied())
                    .unwrap_or(std::ptr::null_mut()),
                // Per CALL, and this is the accessor that has no call to
                // answer for. [`CustomAllReduce::plane_for`] is the one the
                // plain reduction goes through.
                rank_data: std::ptr::null_mut(),
                fully_connected: self.fully_connected,
            },
        }
    }

    /// `vllm::RankSignals` — `Signal* signals[8]`, self included.
    ///
    /// Eight and not `world_size`, because upstream's struct is eight
    /// pointers wide at every world size and the kernel is written against
    /// that: `sg.signals[threadIdx.x]` under `if (threadIdx.x < ngpus)`.
    /// Entries at or past `world_size` are never read and are zeroed here, so
    /// a kernel that read one anyway faults instead of following a stale
    /// address.
    fn signal_array(&self) -> [*mut c_void; 8] {
        let mut out = [std::ptr::null_mut(); 8];
        for (slot, peer) in out.iter_mut().zip(self.signal_peers.iter()) {
            *slot = *peer;
        }
        out
    }

    /// The plane a PLAIN reduction of `input` reduces through.
    ///
    /// [`CustomAllReduce::plane`] plus the one fact that is per call: the
    /// `RankData*` slot `register_buffer` wrote for this input's base
    /// allocation. `vllm::CustomAllreduce::allreduce` looked this up itself
    /// (`vllm_custom_all_reduce.cuh:454-467`), and it is here because the
    /// launcher descended and the map did not.
    ///
    /// Null `rank_data` is `Decline::Unregistered`, which is what upstream
    /// threw. Two ways to get one, and they are upstream's two:
    ///
    /// * the input's base address is not in `buffers` — never registered;
    /// * `cuPointerGetAttribute` will not name a base — not a device
    ///   allocation this process made.
    ///
    /// # Capture is the third way, and it does NOT mutate here
    ///
    /// On a capturing stream upstream takes `d_rank_data_base_ +
    /// graph_unreg_buffers_.size()` and pushes `input` onto
    /// `graph_unreg_buffers_` in the same expression, so the slot is claimed
    /// whether or not a kernel is enqueued. This computes the same address
    /// and pushes NOTHING; [`CustomAllReduce::note_graph_buffer`] is the
    /// push, and the caller makes it only after the launch actually happened.
    /// A deferred registration for a launch that declined would bind the next
    /// real one to the wrong slot, which is silent and rank-dependent.
    ///
    /// # Errors
    ///
    /// Never — an unresolvable input is a null `rank_data`, not an error,
    /// because the launch half has a [`Decline`] for it and a caller that
    /// gets an `Err` here cannot tell it from a broken stream.
    #[must_use]
    pub fn plane_for(&self, input: *const c_void, capturing: bool) -> Plane {
        let mut plane = self.plane();
        plane.peers.rank_data = if capturing {
            // `register_graph_buffers` drains `graph_unreg_buffers` into
            // `write_rank_data`, which starts at `rank_data_next` and lays the
            // rows down in order -- so the slot this launch replays out of is
            // that base plus however many captures are already queued.
            let slot = self.rank_data_next + self.graph_unreg_buffers.len();
            if self.rank_data.is_null() || slot >= self.rank_data_slots {
                std::ptr::null_mut()
            } else {
                (self.rank_data as usize + slot * RANK_DATA_BYTES) as *mut c_void
            }
        } else {
            base_ptr(input)
                .ok()
                .and_then(|base| self.buffers.get(&(base as usize)).copied())
                .unwrap_or(std::ptr::null_mut())
        };
        plane
    }

    /// Record that a CAPTURED launch claimed the next `RankData` slot for
    /// `input` — upstream's `graph_unreg_buffers_.push_back(input)`.
    ///
    /// Split out of [`CustomAllReduce::plane_for`] so that the slot is
    /// claimed only when a kernel was actually enqueued. `register_graph_buffers`
    /// drains this list and fills the slots in order, so a push with no
    /// launch behind it shifts every later replay onto the wrong peer
    /// addresses.
    pub fn note_graph_buffer(&mut self, input: *mut c_void) {
        self.graph_unreg_buffers.push(input);
    }
}

// ── how an ARM reaches it ────────────────────────────────────────────────

thread_local! {
    /// The plane this thread's fires reduce through — see [`ResidentPlane`].
    static CURRENT: std::cell::Cell<*mut CustomAllReduce> =
        const { std::cell::Cell::new(std::ptr::null_mut()) };
}

/// One rank's plane, owned by the shell and published to the thread that
/// fires.
///
/// # Why a thread-local and not a `DispatchCtx` field
///
/// A bind arm is a `fn(&Cx, *mut c_void)`, and [`Cx`](crate::bind::cx::Cx) is
/// query-only by `northstar.md` §3.3: it answers facts about a STATEMENT, and
/// a communicator is not one. `DispatchCtx` is where the driver's other
/// per-fire handle lives — `ctx.cublas` — and that is the precedent this
/// would otherwise follow.
///
/// It does not follow it, for a reason about ownership rather than taste. A
/// `cublasHandle_t` is a `*mut c_void` that any thread may hold;
/// `CustomAllReduce` is neither `Send` nor `Sync`, because every one of its
/// methods assumes the calling thread holds the device context the
/// constructor ran on, and because three other ranks hold the far end of its
/// IPC handles. **A TP group's ranks are threads of one process** — the whole
/// of `layout::rendezvous` is built on that — so "this thread's plane" is not
/// an approximation of "this rank's plane", it is the same statement.
///
/// # What the type guarantees
///
/// Construction publishes and [`Drop`] retracts, so the raw pointer in
/// `CURRENT` is live for exactly as long as the value behind it. The address
/// is a `Box`'s, so it survives the shell being moved; and the shell holding
/// a `!Send` field is what stops it being moved to a thread the publication
/// would not follow.
#[derive(Debug)]
pub struct ResidentPlane(Box<CustomAllReduce>);

impl ResidentPlane {
    /// Publish `car` as the calling thread's plane.
    #[must_use]
    pub fn publish(car: CustomAllReduce) -> Self {
        let mut boxed = Box::new(car);
        let at: *mut CustomAllReduce = &raw mut *boxed;
        CURRENT.with(|slot| slot.set(at));
        Self(boxed)
    }

    /// The plane, for a caller that already has the shell.
    #[must_use]
    pub fn plane(&self) -> Plane {
        self.0.plane()
    }
}

impl Drop for ResidentPlane {
    fn drop(&mut self) {
        CURRENT.with(|slot| slot.set(std::ptr::null_mut()));
    }
}

/// Run `f` against this thread's plane, or answer `None` when it has none.
///
/// **Re-entrant calls see `None`.** The pointer is taken out of the slot for
/// the duration and restored after, so two overlapping `&mut` borrows of one
/// `CustomAllReduce` cannot be produced even if an arm somehow re-entered the
/// dispatch. Nothing does today; the property is what makes the `&mut` sound
/// without a `RefCell`'s runtime cost on every fire.
///
/// `&mut` and not `&` because [`CustomAllReduce::register_buffer`] takes one,
/// and an arm firing on a buffer the plane has never seen has to be able to
/// register it — that is the call `can_handle`'s [`Decline::Unregistered`]
/// exists to send a caller to.
pub fn with_current<R>(f: impl FnOnce(&mut CustomAllReduce) -> R) -> Option<R> {
    let at = CURRENT.with(|slot| slot.replace(std::ptr::null_mut()));
    if at.is_null() {
        return None;
    }
    // SAFETY: `ResidentPlane` publishes the address of its own `Box` and
    // retracts it in `Drop`, so a non-null slot names a live value; the slot
    // is nulled for the duration above, so no second `&mut` can be minted
    // while this one is alive.
    let out = f(unsafe { &mut *at });
    CURRENT.with(|slot| slot.set(at));
    Some(out)
}

/// `custom_all_reduce.cu:403-427` — the destructor.
///
/// Order matters and is upstream's: peer mappings close before the memory
/// they were opened against is freed.
impl Drop for CustomAllReduce {
    fn drop(&mut self) {
        // Every peer mapping this object ever opened, by construction --
        // signal peers, registered buffers, fusion peers and graph buffers
        // all went through `open_ipc_handle`. The C++ walked `signal_peers_`
        // only (`:410-419`) and leaked the rest.
        for (_, ptr) in self.ipc_handles.drain() {
            if !ptr.is_null() {
                // SAFETY: opened by `cudaIpcOpenMemHandle`, closed once.
                ignore_in_drop(unsafe { cudaIpcCloseMemHandle(ptr) });
            }
        }
        let mut owned = vec![self.signal_self, self.rank_data];
        if let Some(fusion) = self.fusion.as_ref() {
            owned.extend_from_slice(&fusion.buffers);
            owned.push(fusion.workspace_dev);
            owned.push(fusion.flag_dev);
        }
        for ptr in owned {
            if !ptr.is_null() {
                // SAFETY: each came from `cudaMalloc` in this object and is
                // freed once.
                ignore_in_drop(unsafe { cudaFree(ptr) });
            }
        }
    }
}

// ── the ABI forms ────────────────────────────────────────────────────────
//
// `custom_all_reduce.hpp:143-201` declared four free functions so the shim
// could name them without a C++ type in the signature. They survive as the
// same four shapes, `car` still an opaque handle, because `table::gemm`'s
// two rows spell their first operand `KernelParam::CustomAllReduce`, which
// `kernels/src/lib.rs:1082-1132` spells `*mut c_void` -- and the owner's
// constraint is that the model compiler must not be able to tell whether a
// symbol is cuBLAS or a JIT'd kernel. It equally must not be able to tell
// whether it is a Rust struct.

/// Reborrow an opaque `car` handle.
///
/// # Safety
///
/// `car` must be null or a pointer to a live [`CustomAllReduce`] owned by
/// the caller, not aliased for the duration of the call.
unsafe fn reborrow<'a>(car: *mut c_void) -> Option<&'a mut CustomAllReduce> {
    if car.is_null() {
        return None;
    }
    // SAFETY: the caller's contract.
    Some(unsafe { &mut *car.cast::<CustomAllReduce>() })
}

// THESE TWO ARE NOW THE WHOLE OF THE DRIVER'S SIDE, and that is what the
// descent bought. Each used to be one of THREE layers -- `bind/service.rs`'s
// `comm_all_reduce_bf16` called this, which called
// `CustomAllReduce::all_reduce_bf16`, which built the refusal -- and the
// middle layer's entire content was `reborrow` plus a forward. `bind/
// service.rs` is deleted and the innermost layer is `kernels_cuda::comm`,
// so what is left here is the one thing neither of the others could do:
// **turn an opaque handle back into a Rust value.** `car` is a `*mut c_void`
// because a row's operand is spelled `KernelParam::CustomAllReduce` and the
// model compiler must not be able to tell whether a symbol is cuBLAS, a JIT'd
// kernel or a Rust struct.
//
// Neither has a caller today, and neither did before: `bind/service.rs`'s
// wrappers were called by a generated dispatch that no longer exists, so
// `bind::route` answers `Route::Rows` for both symbols and the hand match has
// no arm. That is a real gap and it is `bind/arms/comm.rs`-shaped, not this
// file's; recording it here is the most a descent can do about it.

/// `custom_all_reduce.hpp:164-180` — the plain P2P reduction, by handle.
///
/// # Safety
///
/// `car` is an opaque [`CustomAllReduce`] handle; `input` and `output`
/// address at least `count` bf16 elements on the device, and `stream` names a
/// live CUDA stream for the duration of the call.
#[must_use]
pub unsafe fn all_reduce_bf16(
    car: *mut c_void,
    input: *const c_void,
    output: *mut c_void,
    count: usize,
    stream: *mut c_void,
) -> AllReduce {
    // SAFETY: the caller's contract.
    let Some(car) = (unsafe { reborrow(car) }) else {
        return AllReduce::Declined(Decline::NoInstance);
    };
    // The capture query is the driver's because the plane is: it decides
    // WHICH `RankData` slot the launch addresses, and under capture that slot
    // is one nothing has filled yet -- `register_graph_buffers` fills it once
    // the peers have exchanged handles for the replayed addresses.
    let Some(capturing) = capturing(stream) else {
        return AllReduce::Declined(Decline::CaptureUnknown);
    };
    let plane = car.plane_for(input, capturing);
    // SAFETY: `stream` is the caller's, live across the call.
    let ctx = unsafe { kernels_cuda::jit::Ctx::on(stream) };
    let fired = kernels_cuda::comm::all_reduce_bf16(&ctx, plane, input, output, count);
    // Upstream pushed BEFORE the launch and could not tell the two apart; the
    // push is here, after, so a decline claims no slot. `plane_for`'s doc has
    // the argument.
    if capturing && fired == AllReduce::Launched {
        car.note_graph_buffer(input.cast_mut());
    }
    fired
}

/// Whether `stream` is capturing, or `None` if the driver would not say.
///
/// `custom_all_reduce.cu:470`'s query, hoisted out of `can_handle` so the
/// launch path can ask it too: both readers want the same fact and the C++
/// asked it twice.
fn capturing(stream: *mut c_void) -> Option<bool> {
    let mut status = cudaStreamCaptureStatus::cudaStreamCaptureStatusNone;
    // SAFETY: `stream` is a `cudaStream_t` from the caller (null is the legal
    // default stream) and the out-parameter is live.
    if unsafe { cudaStreamIsCapturing(stream as cudaStream_t, &mut status) }
        != cudaError::cudaSuccess
    {
        return None;
    }
    Some(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive)
}

/// `custom_all_reduce.hpp:186-201` — the fused landing, by handle.
///
/// # Safety
///
/// `car` is an opaque [`CustomAllReduce`] handle; `input`, `residual_inout`
/// and `norm_out` address at least `tokens * hidden` bf16 elements,
/// `rms_gamma` at least `hidden`, and `stream` names a live CUDA stream.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub unsafe fn all_reduce_residual_rmsnorm_bf16(
    car: *mut c_void,
    input: *const c_void,
    residual_inout: *mut c_void,
    rms_gamma: *const c_void,
    norm_out: *mut c_void,
    tokens: c_int,
    hidden: c_int,
    eps: f32,
    stream: *mut c_void,
) -> AllReduce {
    // SAFETY: the caller's contract.
    let Some(car) = (unsafe { reborrow(car) }) else {
        return AllReduce::Declined(Decline::NoInstance);
    };
    // SAFETY: as above.
    let ctx = unsafe { kernels_cuda::jit::Ctx::on(stream) };
    kernels_cuda::comm::all_reduce_residual_rmsnorm_bf16(
        &ctx,
        car.plane(),
        input,
        residual_inout,
        rms_gamma,
        norm_out,
        tokens,
        hidden,
        eps,
    )
}

/// `custom_all_reduce.hpp:150-158` — the free forms of the two lifecycle
/// calls, for a caller holding the handle rather than the struct.
///
/// # Safety
///
/// `car` is an opaque [`CustomAllReduce`] handle.
///
/// # Errors
///
/// [`Error::Invalid`] when `car` is null; otherwise whatever
/// [`CustomAllReduce::register_buffer`] refuses.
pub unsafe fn register_buffer(
    car: *mut c_void,
    buf: *mut c_void,
    buf_bytes: usize,
) -> crate::error::Result<()> {
    // SAFETY: the caller's contract.
    let Some(car) = (unsafe { reborrow(car) }) else {
        return Err(Error::invalid("custom_all_reduce", "null handle"));
    };
    car.register_buffer(buf, buf_bytes)
}

/// # Safety
///
/// `car` is an opaque [`CustomAllReduce`] handle.
///
/// # Errors
///
/// [`Error::Invalid`] when `car` is null; otherwise whatever
/// [`CustomAllReduce::register_graph_buffers`] refuses.
pub unsafe fn register_graph_buffers(car: *mut c_void) -> crate::error::Result<()> {
    // SAFETY: the caller's contract.
    let Some(car) = (unsafe { reborrow(car) }) else {
        return Err(Error::invalid("custom_all_reduce", "null handle"));
    };
    car.register_graph_buffers()
}

#[cfg(test)]
mod tests {
    use super::{LAMPORT_COMM_CAP, RANK_DATA_BYTES, SIGNAL_BYTES, align_up};

    /// Every constant this file carries is host arithmetic over an upstream
    /// struct layout, and none of it touches a device. That is the whole
    /// reason these are tests and not comments.
    #[test]
    fn the_signal_slab_is_the_upstream_struct() {
        // `vllm_custom_all_reduce.cuh:52-60`: `self_counter[36][8]` at 128B
        // alignment plus `peer_counter[2][36][8]`, `FlagType = uint32_t`.
        assert_eq!(SIGNAL_BYTES, 36 * 8 * 4 + 2 * 36 * 8 * 4);
        // `:62-64`: `struct __align__(16) RankData { void* ptrs[8]; }`.
        assert_eq!(RANK_DATA_BYTES, 8 * std::mem::size_of::<*mut u8>());
    }

    #[test]
    fn the_lamport_cap_is_the_largest_aligned_count_a_signed_word_holds() {
        // The flag block's word 3 is `uint32_t` but read as a signed size
        // downstream; `custom_all_reduce.cu:329-333` capped it here.
        assert_eq!(LAMPORT_COMM_CAP, (1usize << 31) - (1 << 21));
        assert_eq!(align_up(LAMPORT_COMM_CAP, 1 << 21), LAMPORT_COMM_CAP);
    }

    // FOUR MORE TESTS STOOD HERE AND ARE `kernels_cuda::comm`'s, with the
    // cross product they assert on: `the_cross_product_is_the_number_kernels_\
    // def_measured`, `the_one_reached_point_resolves`,
    // `an_uninstantiated_pattern_declines_with_its_code` and
    // `an_unsupported_world_size_declines_before_the_pattern_is_read`. They
    // did not stay behind the re-export, and that is the point: a test that
    // reaches its subject through a `pub use` is testing the re-export.
    //
    // The two above stayed because their subjects did. `SIGNAL_BYTES`,
    // `RANK_DATA_BYTES`, `LAMPORT_COMM_CAP` and `align_up` size the PLANE and
    // are spent by the constructor in this file; the cross product sizes a
    // LAUNCH. That is the same line the whole split is drawn on, and these
    // six tests are where it is easiest to see.
}
