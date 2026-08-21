//! The custom P2P all-reduce's host lifecycle, in Rust: peer access, the IPC
//! handle exchange, the `Signal`/`RankData` and fusion slabs with their
//! Lamport initialisation, and the destructor that closes what they opened.
//! [`kernels_cuda::comm`] owns the launches; this file owns everything with a
//! lifetime. Nothing here has run against a second GPU — it is correct by
//! review and by its `custom_all_reduce.cu` ancestry only.

// `initialise` logs one line to stderr when the P2P plane comes up.
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
use kernels::Refusal;

// ── constants the constructor spends ──────────────────────────────────────

/// `sizeof(vllm::Signal)`: `self_counter[36][8]` (1152) +
/// `peer_counter[2][36][8]` (2304), `FlagType = uint32_t`. An ABI fact of a
/// header not vendored here.
pub const SIGNAL_BYTES: usize = 1152 + 2304;

/// `sizeof(vllm::RankData)` — `struct __align__(16) RankData { void* ptrs[8]; }`.
pub const RANK_DATA_BYTES: usize = 8 * 8;

/// The default `max_bytes`, 8 MiB.
pub const DEFAULT_MAX_BYTES: usize = 8 * 1024 * 1024;

/// The default `rank_data_bytes`, 8 MiB — `8 MiB / 64 B`, ~131k graph slots.
pub const DEFAULT_RANK_DATA_BYTES: usize = 8 * 1024 * 1024;

/// Fusion allocations round up to 2 MiB — the large-page granularity the
/// Lamport protocol's address arithmetic assumes.
pub const FUSION_ALIGN: usize = 1 << 21;

/// `kBarrierFlagCount`, 256.
pub const BARRIER_FLAG_COUNT: usize = 256;

/// The Lamport communication cap `2^31 - 2 MiB`: the largest 2 MiB-aligned
/// byte count that fits the signed 32-bit offset the flag block's word 3 is
/// read as.
pub const LAMPORT_COMM_CAP: usize = 2_145_386_496;

/// bf16 negative zero (`0x8000`), the Lamport empty-slot sentinel.
const LAMPORT_EMPTY_BF16: u16 = 0x8000;

pub use kernels_cuda::comm::{AllReduce, Decline, FusionPlane, PeerPlane, Plane};

fn align_up(n: usize, a: usize) -> usize {
    n.div_ceil(a) * a
}

/// The base allocation of `ptr`. The IPC handle exchange needs the base:
/// sub-allocation pointers do not round-trip across processes.
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

/// Enable peer access. Idempotent: `cudaErrorPeerAccessAlreadyEnabled` is
/// swallowed and the sticky error reset. `peers` holds real device ordinals,
/// never rank indices — a TP group need not be devices `0..world_size`.
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
        // Reset the sticky error so a later unrelated call does not inherit it.
        // SAFETY: no arguments, no aliasing.
        let _ = unsafe { cudaGetLastError() };
    }
    Ok(())
}

/// Both directions of every ordered pair: peer access is not symmetric on
/// every topology.
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

/// The 64 opaque bytes of a `cudaIpcMemHandle_t` — bytes travel, not the
/// struct, and are what makes the handle hashable for the `open_ipc_handle`
/// memo.
fn from_handle(handle: &cudaIpcMemHandle_t) -> [u8; 64] {
    let mut bytes = [0u8; 64];
    for (dst, src) in bytes.iter_mut().zip(handle.reserved.iter()) {
        *dst = *src as u8;
    }
    bytes
}

// ── the seam ─────────────────────────────────────────────────────────────

/// One bootstrap-time all-gather over host buffers. `send` is this rank's
/// contribution; `recv` is `send.len() * world_size` bytes, rank-major.
pub type Allgather = Arc<dyn Fn(&[u8], &mut [u8]) + Send + Sync>;

/// What this needs from the collective, and nothing more — a callback, not a
/// communicator type, so a compute kernel does not depend on the driver's comm
/// plane. `gather` takes host buffers; the H2D dance and custom-vs-NCCL choice
/// are the caller's.
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

/// The constructor's remaining arguments.
#[derive(Debug, Clone)]
pub struct Config {
    /// Skip IPC entirely and exchange raw pointers, for a single-process
    /// multi-GPU deployment.
    pub same_process: bool,
    /// The CUDA device ordinal of every rank, indexed by rank. Required, and
    /// required to be `world_size` long.
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
/// dimensions positive.
#[derive(Debug)]
struct Fusion {
    /// `[buffer, flag, lamport]`.
    buffers: [*mut c_void; 3],
    workspace_dev: *mut c_void,
    flag_dev: *mut c_void,
    max_tokens: i32,
    hidden: i32,
}

// ── the lifecycle ────────────────────────────────────────────────────────

/// The custom P2P all-reduce's whole host state — the half of
/// `vllm::CustomAllreduce` that never launched anything.
///
/// Every peer handle opened here goes through
/// [`CustomAllReduce::open_ipc_handle`], which memoises by handle bytes, and
/// [`Drop`] closes the memo — closing two leaks the C++ destructor had.
///
/// Not `Send`, not `Sync`: the raw device pointers make it neither, and every
/// method assumes the calling thread holds the constructor's device context.
#[derive(Debug)]
pub struct CustomAllReduce {
    rank: i32,
    world_size: i32,
    fully_connected: bool,
    same_process: bool,
    max_bytes: usize,
    ag: HostAllgather,

    signal_self: *mut c_void,
    /// `signal_peers_`, one per rank, self included.
    signal_peers: Vec<*mut c_void>,

    rank_data: *mut c_void,
    /// `d_rank_data_end_ - d_rank_data_base_` at construction, in slots.
    rank_data_slots: usize,
    /// upstream's `d_rank_data_base_`, as an index rather than a pointer.
    rank_data_next: usize,

    /// local base address -> the device `RankData*` it was registered into.
    buffers: HashMap<usize, *mut c_void>,
    /// upstream's `ipc_handles_`, keyed on the handle's 64 opaque bytes.
    ipc_handles: HashMap<[u8; 64], *mut c_void>,
    /// Drained by [`CustomAllReduce::register_graph_buffers`]. Nothing appends
    /// to it in this tree; a decline must not append, or a deferred
    /// registration for a launch that never happened would bind the next real
    /// one to the wrong slot.
    graph_unreg_buffers: Vec<*mut c_void>,

    fusion: Option<Fusion>,
}

impl CustomAllReduce {
    /// The constructor. Errors on an unsupported world size, a missing or
    /// mis-sized `group_devices`, unavailable peer access, any failing CUDA
    /// call.
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
        // Built first so a failure part-way still returns through `Drop`,
        // which frees whatever was allocated.
        me.initialise(cfg)?;
        Ok(me)
    }

    fn initialise(&mut self, cfg: &Config) -> crate::error::Result<()> {
        if self.world_size < 2 || self.world_size > 8 || (self.world_size % 2) != 0 {
            return Err(Error::invalid(
                "custom_all_reduce",
                format!(
                    "the vllm kernel supports world_size in {{2,4,6,8}}; got {}",
                    self.world_size
                ),
            ));
        }

        let mut dev: c_int = 0;
        // SAFETY: a live out-parameter.
        check_rt(unsafe { cudaGetDevice(&mut dev) }, "cudaGetDevice")?;
        if cfg.group_devices.is_empty() {
            return Err(Error::invalid(
                "custom_all_reduce",
                "group device ordinals are required",
            ));
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

        // the vllm kernel handles larger TP groups only when every rank has
        // direct peer access to every other rank.
        self.fully_connected = self.world_size <= 2 || has_full_peer_access(&cfg.group_devices);

        // the staging region past the `Signal` is what the 2-stage algorithm
        // needs; TP=2 takes the 1-stage path but the layout is matched anyway.
        let signal_bytes = SIGNAL_BYTES + self.max_bytes;
        // SAFETY: a live out-parameter and a positive size.
        check_rt(
            unsafe { cudaMalloc(&mut self.signal_self, signal_bytes) },
            "cudaMalloc(signal)",
        )?;
        // SAFETY: `signal_self` now addresses `signal_bytes` writable bytes.
        check_rt(
            unsafe { cudaMemset(self.signal_self, 0, signal_bytes) },
            "cudaMemset(signal)",
        )?;

        self.signal_peers = self.exchange_pointers(self.signal_self)?;

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

        eprintln!(
            "[custom_all_reduce] initialised (world={}, rank={}, mode={}, fully_connected={})",
            self.world_size,
            self.rank,
            if self.same_process {
                "same-process"
            } else {
                "ipc"
            },
            if self.fully_connected { "yes" } else { "no" }
        );
        Ok(())
    }

    /// One all-gather of `local`'s address across the group. In `same_process`
    /// mode the raw `u64` crosses; otherwise a `cudaIpcMemHandle_t` does.
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
        check_rt(
            unsafe { cudaIpcGetMemHandle(&mut self_handle, local) },
            "cudaIpcGetMemHandle",
        )?;
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

    /// The collective; the rank-major receive buffer is sized here.
    fn allgather(&self, send: &[u8]) -> Vec<u8> {
        let mut recv = vec![0u8; send.len() * self.world_size as usize];
        (self.ag.gather)(send, &mut recv);
        recv
    }

    /// Open a peer IPC handle, memoised by the handle's bytes. The memo is not
    /// an optimisation: `cudaIpcOpenMemHandle` on a handle already open returns
    /// the SAME mapping and balances no second `cudaIpcCloseMemHandle`, so
    /// opening twice and closing twice is a double free; the key is how
    /// [`Drop`] knows the exact set to close.
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

    /// The fusion plane's four allocations, its Lamport initialisation and its
    /// device workspace.
    fn build_fusion(&mut self, cfg: &Config) -> crate::error::Result<()> {
        let world = self.world_size as usize;
        let max_tokens = cfg.fusion_max_tokens;
        let hidden = cfg.fusion_hidden;

        // `elem_bytes` is `sizeof(__nv_bfloat16)`.
        const ELEM_BYTES: usize = 2;
        let span = world * max_tokens as usize * hidden as usize * ELEM_BYTES;
        let buffer_bytes = align_up(span, FUSION_ALIGN);
        let flag_bytes = align_up(world * BARRIER_FLAG_COUNT * 4, FUSION_ALIGN);
        let lamport_comm_bytes = span.min(LAMPORT_COMM_CAP);
        let lamport_bytes = align_up(lamport_comm_bytes * 3, FUSION_ALIGN);

        let mut buffers = [std::ptr::null_mut::<c_void>(); 3];
        for (slot, bytes) in buffers
            .iter_mut()
            .zip([buffer_bytes, flag_bytes, lamport_bytes])
        {
            // SAFETY: a live out-parameter and a positive size.
            check_rt(unsafe { cudaMalloc(slot, bytes) }, "cudaMalloc(fusion)")?;
        }

        let mut flag_dev: *mut c_void = std::ptr::null_mut();
        // SAFETY: a live out-parameter.
        check_rt(
            unsafe { cudaMalloc(&mut flag_dev, 5 * 4) },
            "cudaMalloc(fusion flags)",
        )?;

        // Park it on `self` now so `Drop` owns it before anything else fails.
        self.fusion = Some(Fusion {
            buffers,
            workspace_dev: std::ptr::null_mut(),
            flag_dev,
            max_tokens,
            hidden,
        });

        // bf16 negative zero (`0x8000`) writes with `cuMemsetD16_v2`, no kernel.
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

        // Word 3 carries the Lamport communication size; the other four are zero.
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

        // `3 * world + 1` pointers: every rank's view of each of the three
        // buffers, rank-major within buffer, then the local flag block.
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
                cudaMalloc(
                    &mut workspace_dev,
                    workspace.len() * std::mem::size_of::<*mut c_void>(),
                )
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
        // The workspace copy above is async; sync so every peer sees it first.
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

    /// Whether every ordered pair of the group has direct peer access.
    #[must_use]
    pub const fn fully_connected(&self) -> bool {
        self.fully_connected
    }

    /// The largest message the plain P2P path will take.
    #[must_use]
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Whether the fusion plane was built.
    #[must_use]
    pub const fn has_fusion(&self) -> bool {
        self.fusion.is_some()
    }

    /// Register a buffer. `_buf_bytes` is unused: the kernel registers a BASE
    /// address and does its own offset arithmetic against the registered
    /// `RankData`, so the extent never mattered; it stays in the signature for
    /// the caller. Idempotent per base address, so safe to call every step.
    /// Errors on an unresolvable base, a failing IPC exchange, or an exhausted
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

    /// upstream `vllm::CustomAllreduce::check_rank_data_capacity` + the
    /// `cudaMemcpy` into `d_rank_data_base_`, for a run of `rows` consecutive
    /// slots. Returns the device address of the FIRST slot written.
    fn write_rank_data(&mut self, rows: &[Vec<*mut c_void>]) -> crate::error::Result<*mut c_void> {
        let n = rows.len();
        if self.rank_data_next + n > self.rank_data_slots {
            return Err(Error::exhausted(
                "custom_all_reduce rank_data slots",
                self.rank_data_next + n,
            ));
        }
        // `RankData` is `void* ptrs[8]` regardless of world size; the tail
        // past `world_size` is padding the kernel never reads, and is
        // zeroed here rather than left undefined.
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

    /// The concatenated 64-byte handles of every unregistered graph buffer's
    /// BASE allocation, and each buffer's byte offset within it.
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

    /// Register, in one collective, every buffer an all-reduce met unregistered
    /// on a CAPTURING stream. A no-op today. When wired in, it belongs after
    /// the capture closes (the collective is not capturable) and before
    /// `graph.instantiate()` (it writes the `RankData` slots replayed addresses
    /// resolve through). Errors on a failing IPC exchange or exhausted slab.
    pub fn register_graph_buffers(&mut self) -> crate::error::Result<()> {
        let n = self.graph_unreg_buffers.len();
        if n == 0 {
            return Ok(());
        }
        let world = self.world_size as usize;

        if self.same_process {
            // Gather every rank's raw pointers, buffer-minor.
            let mut send = Vec::with_capacity(n * 8);
            for &ptr in &self.graph_unreg_buffers {
                send.extend_from_slice(&(ptr as usize as u64).to_ne_bytes());
            }
            let gathered = self.allgather(&send);
            let mut rows: Vec<Vec<*mut c_void>> = Vec::with_capacity(n);
            for i in 0..n {
                let mut row = Vec::with_capacity(world);
                for r in 0..world {
                    // rank-major outer, buffer-minor inner.
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

    /// Whether this plane will take the message. A [`Decline`] is a routing
    /// query, not a failure: above a bandwidth threshold the kernel falls off
    /// to NCCL on purpose, and the `Decline` is the caller's cue to.
    pub fn can_handle(
        &self,
        input: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> std::result::Result<(), Decline> {
        if input.is_null() {
            return Err(Decline::NullInput);
        }
        // the 16-byte multiple is the kernel's vector width.
        if bytes == 0 || bytes > self.max_bytes || bytes % 16 != 0 {
            return Err(Decline::Bytes {
                bytes,
                max_bytes: self.max_bytes,
            });
        }
        if self.world_size > 2 && !self.fully_connected {
            return Err(Decline::NotFullyConnected {
                world_size: self.world_size,
            });
        }
        // Under capture the address is replayed, not dereferenced now, so the
        // registration check is deferred to `register_graph_buffers` and
        // capture answers yes immediately.
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
        // A throwing `base_ptr` is not distinguished from "not registered".
        let Ok(base) = base_ptr(input) else {
            return Err(Decline::Unregistered);
        };
        if !self.buffers.contains_key(&(base as usize)) {
            return Err(Decline::Unregistered);
        }
        // The measured NCCL crossover: below it the P2P kernel wins on
        // latency, above it NCCL wins on bandwidth, and the wider the group
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
            Err(Decline::AboveCrossover {
                bytes,
                crossover,
                world_size: self.world_size,
            })
        }
    }

    /// The facts a launch reads off this instance: group size, rank, and (when
    /// built) the fusion workspace address and its two extents. Excludes
    /// `Fusion::buffers`/`flag_dev`: no launch reads them.
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
                // Per call; this accessor has none. See `plane_for`.
                rank_data: std::ptr::null_mut(),
                fully_connected: self.fully_connected,
            },
        }
    }

    /// `Signal* signals[8]`, self included — eight at every world size,
    /// because the kernel indexes `signals[threadIdx.x]`. Entries past
    /// `world_size` are zeroed, so a stray read faults, not follows a stale
    /// address.
    fn signal_array(&self) -> [*mut c_void; 8] {
        let mut out = [std::ptr::null_mut(); 8];
        for (slot, peer) in out.iter_mut().zip(self.signal_peers.iter()) {
            *slot = *peer;
        }
        out
    }

    /// The plane a plain reduction of `input` reduces through:
    /// [`CustomAllReduce::plane`] plus the per-call `RankData*` slot
    /// `register_buffer` wrote for this input's base; a null slot is
    /// `Decline::Unregistered`. Under capture it computes the slot the replay
    /// will land in but pushes nothing — see
    /// [`CustomAllReduce::note_graph_buffer`].
    #[must_use]
    pub fn plane_for(&self, input: *const c_void, capturing: bool) -> Plane {
        let mut plane = self.plane();
        plane.peers.rank_data = if capturing {
            // The replay slot is `rank_data_next` plus the captures already
            // queued.
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
    /// `input`. Split from [`CustomAllReduce::plane_for`] so a slot is claimed
    /// only once a kernel was enqueued: `register_graph_buffers` fills slots in
    /// order, so a push with no launch behind it shifts later replays wrong.
    pub fn note_graph_buffer(&mut self, input: *mut c_void) {
        self.graph_unreg_buffers.push(input);
    }
}

// ── how an ARM reaches it ────────────────────────────────────────────────

thread_local! {
    /// The `CustomAllReduce` this thread's fires reduce through — the
    /// lifecycle object, not the [`Plane`]. [`admitted`] is the only reader.
    static CURRENT: std::cell::Cell<*mut CustomAllReduce> =
        const { std::cell::Cell::new(std::ptr::null_mut()) };
}

/// One rank's plane, published to the thread that fires. Thread-local because
/// `CustomAllReduce` is `!Send`/`!Sync`: a TP group's ranks are threads of one
/// process. Construction publishes and [`Drop`] retracts, so the `Box`'s
/// address in `CURRENT` is live exactly as long as the value behind it.
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

/// The plane this thread's fire reduces through, when the plane admits the
/// message. Calls [`CustomAllReduce::plane`], not
/// [`CustomAllReduce::plane_for`], so the per-call `RankData*` stays null and
/// under capture the launch declines [`Decline::Unregistered`]; the ABI form
/// [`all_reduce_bf16`] resolves the slot instead. [`Decline::NoInstance`] when
/// this thread has no plane.
pub fn admitted(input: *const c_void, bytes: usize, stream: *mut c_void) -> Result<Plane, Decline> {
    let at = CURRENT.with(|slot| slot.get());
    if at.is_null() {
        return Err(Decline::NoInstance);
    }
    // SAFETY: `ResidentPlane` publishes its `Box`'s address and retracts it in
    // `Drop`, so a non-null slot names a live value borrowed only shared here.
    let car = unsafe { &*at };
    car.can_handle(input, bytes, stream)?;
    Ok(car.plane())
}

/// This thread's `car`, as the opaque handle the ABI forms take.
///
/// Null when no rank published one, which every form below reads as
/// [`Decline::NoInstance`] rather than as an error: a deployment with no
/// custom all-reduce is a routing answer.
///
/// The handle and not a [`Plane`]: `plane_for` needs the launch's input and
/// the stream's capture state, and `note_graph_buffer` needs `&mut` -- neither
/// of which a resolved plane can still offer.
#[must_use]
pub fn resident_car() -> *mut c_void {
    CURRENT.with(|slot| slot.get()).cast()
}

/// One [`Decline`] as the [`Refusal`] that carries the most of it.
///
/// Here and not in `bind/arms/comm.rs`: the mapping is about `Decline`, and
/// the driver-op table is now the only caller. A decline's
/// numbers survive only into a variant with integer fields.
pub fn refusal_for(decline: &Decline) -> Refusal {
    match decline {
        Decline::NoInstance | Decline::NotInitialised => Refusal::Absent {
            what: "a constructed custom all-reduce for this rank",
        },
        Decline::NullInput => Refusal::Null {
            what: "the all-reduce's input",
        },
        Decline::Bytes { bytes, max_bytes } => Refusal::Wide {
            what: "the P2P all-reduce's message (or it is not a multiple of 16 bytes)",
            at: i64::try_from(*bytes).unwrap_or(i64::MAX),
            max: i64::try_from(*max_bytes).unwrap_or(i64::MAX),
        },
        Decline::NotFullyConnected { .. } => Refusal::Absent {
            what: "peer access between every ordered pair of a group wider than two",
        },
        Decline::CaptureUnknown => Refusal::Device {
            why: "`cudaStreamIsCapturing` failed on the fire's stream",
        },
        Decline::Unregistered => Refusal::Absent {
            what: "a `register_buffer` for the all-reduce's input",
        },
        Decline::AboveCrossover {
            bytes, crossover, ..
        } => Refusal::Wide {
            what: "the P2P all-reduce's message, above the crossover where NCCL wins",
            at: i64::try_from(*bytes).unwrap_or(i64::MAX),
            max: i64::try_from(*crossover).unwrap_or(i64::MAX),
        },
        Decline::NoFusionWorkspace => Refusal::Absent {
            what: "a fusion workspace (world size 2 with both fusion extents positive builds one)",
        },
        Decline::FusionTokens { tokens, max_tokens } => Refusal::Wide {
            what: "the fused landing's token count",
            at: i64::from(*tokens),
            max: i64::from(*max_tokens),
        },
        Decline::FusionHidden { .. } => Refusal::Unstated {
            what: "a hidden size equal to the one the fusion workspace was sized for",
        },
        Decline::FusionWorldSize { .. } => Refusal::Unstated {
            what: "a world size of two, which is all the fused landing takes",
        },
        Decline::FusionHiddenNotOctet { .. } => Refusal::Unstated {
            what: "a hidden size that is a multiple of 8, the kernel's vector width in bf16",
        },
        Decline::PatternNotInstantiated { .. } => Refusal::Unstated {
            what: "an `AllReduceFusionPattern` in `kernels_cuda::comm::INSTANTIATED`",
        },
        Decline::WorldSizeUnsupported { .. } => Refusal::Unstated {
            what: "a TP world size the kernel is instantiated at (the fused landing takes 2, 4, \
                   8, 16; the plain reduction takes 2, 4, 6, 8)",
        },
        Decline::Vector { count, .. } => Refusal::Narrow {
            what: "the all-reduce's element count, which must be a non-zero multiple of 8 -- \
                   the kernel's 16-byte vector width in bf16",
            at: i64::try_from(*count).unwrap_or(i64::MAX),
        },
        Decline::FusionBlockWidth { threads, max, .. } => Refusal::Wide {
            what: "the fused landing's threads per block, which is `hidden / 8` because \
                   `comm::CLUSTER_SIZE` is pinned to 1",
            at: i64::from(*threads),
            max: i64::from(*max),
        },
        Decline::FusionBlockNarrow { threads, .. } => Refusal::Narrow {
            what: "the two-shot fused kernel's threads per block, which must cover one per rank",
            at: i64::from(*threads),
        },
        Decline::NoTemplateId { .. } => Refusal::Absent {
            what: "a template-id in `kernels_cuda::comm::inst` for the resolved point",
        },
        Decline::DeviceQuery { what } => Refusal::Absent { what },
        // Already a `Refusal` -- `jit::Ctx::launch`'s. Forwarded whole rather
        // than flattened, so a caller sees the layer that refused.
        Decline::Launch(why) => *why,
        // NCCL's refusal, reaching here from another crate:
        // `comm::fall_back_out_of_place` decides which of the two absences to
        // carry.
        Decline::FellBack(why) => *why,
    }
}

/// The destructor. Order matters: peer mappings close before the memory they
/// were opened against is freed.
impl Drop for CustomAllReduce {
    fn drop(&mut self) {
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
                // SAFETY: from `cudaMalloc` in this object, freed once.
                ignore_in_drop(unsafe { cudaFree(ptr) });
            }
        }
    }
}

// ── the ABI forms ────────────────────────────────────────────────────────
//
// `car` stays an opaque handle so the model compiler cannot tell a Rust
// struct from cuBLAS or a JIT kernel.

/// Reborrow an opaque `car` handle.
///
/// # Safety
///
/// `car` is null or a live [`CustomAllReduce`] the caller owns, unaliased.
unsafe fn reborrow<'a>(car: *mut c_void) -> Option<&'a mut CustomAllReduce> {
    if car.is_null() {
        return None;
    }
    // SAFETY: the caller's contract.
    Some(unsafe { &mut *car.cast::<CustomAllReduce>() })
}

// Neither ABI form below has a caller yet — a gap for `bind/arms/comm.rs`.

/// The plain P2P reduction, by handle.
///
/// # Safety
///
/// `car` is an opaque [`CustomAllReduce`] handle; `input`/`output` address at
/// least `count` bf16 device elements; `stream` is a live CUDA stream.
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
    // The capture query decides which `RankData` slot the launch addresses;
    // under capture that slot stays empty until `register_graph_buffers`.
    let Some(capturing) = capturing(stream) else {
        return AllReduce::Declined(Decline::CaptureUnknown);
    };
    let plane = car.plane_for(input, capturing);
    // SAFETY: `stream` is the caller's, live across the call; `plane` is
    // `car`'s own and `car` outlives this frame.
    let ctx = unsafe { kernels_cuda::jit::Ctx::on(stream).with_comm(plane) };
    let fired = kernels_cuda::comm::all_reduce_bf16(&ctx, input, output, count);
    // The push is here, after the launch, so a decline claims no slot.
    if capturing && fired == AllReduce::Launched {
        car.note_graph_buffer(input.cast_mut());
    }
    fired
}

/// Whether `stream` is capturing, or `None` if the driver would not say.
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

/// The fused landing, by handle.
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
    // SAFETY: as above; stream and plane (the caller's and `car`'s) are live.
    let ctx = unsafe { kernels_cuda::jit::Ctx::on(stream).with_comm(car.plane()) };
    kernels_cuda::comm::all_reduce_residual_rmsnorm_bf16(
        &ctx,
        input,
        residual_inout,
        rms_gamma,
        norm_out,
        tokens,
        hidden,
        eps,
    )
}

/// The free form of `register_buffer`, for a caller holding the handle.
///
/// # Safety
///
/// `car` is an opaque [`CustomAllReduce`] handle.
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

    #[test]
    fn the_signal_slab_is_the_upstream_struct() {
        assert_eq!(SIGNAL_BYTES, 36 * 8 * 4 + 2 * 36 * 8 * 4);
        assert_eq!(RANK_DATA_BYTES, 8 * std::mem::size_of::<*mut u8>());
    }

    #[test]
    fn the_lamport_cap_is_the_largest_aligned_count_a_signed_word_holds() {
        // The flag block's word 3 is `uint32_t` but read as a signed size
        // downstream.
        assert_eq!(LAMPORT_COMM_CAP, (1usize << 31) - (1 << 21));
        assert_eq!(align_up(LAMPORT_COMM_CAP, 1 << 21), LAMPORT_COMM_CAP);
    }
}
