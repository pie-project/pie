#![allow(unsafe_code)]
use core::ffi::c_void;
use std::fmt;

use crate::error::{Fault, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Transport {
    #[default]
    Shm,
    Peer,
    Nccl,
}

impl std::str::FromStr for Transport {
    type Err = String;

    fn from_str(word: &str) -> std::result::Result<Transport, String> {
        match word {
            "shm" => Ok(Transport::Shm),
            "peer" => Ok(Transport::Peer),
            "nccl" => Ok(Transport::Nccl),
            other => Err(format!(
                "`{other}` does not name an NCCL transport; the spellings are \
                 `shm` (no device-to-device path: P2P and NVLS off, the \
                 default), `peer` (P2P on), and `nccl` (whatever NCCL's own \
                 environment says)"
            )),
        }
    }
}

#[derive(Clone)]
pub struct Id(pub [u8; 128]);

impl Id {
    pub fn new(transport: Transport) -> Result<Id> {
        #[cfg(feature = "device")]
        {
            use crate::device::nccl;
            transport_defaults(transport);
            let mut id = nccl::ncclUniqueId { internal: [0; 128] };
            // SAFETY: a live out-parameter of the exact type NCCL writes.
            let code = unsafe { nccl::ncclGetUniqueId(&raw mut id) };
            answered("ncclGetUniqueId", code)?;
            Ok(Id(id.internal.map(|byte| byte as u8)))
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = transport;
            Err(Fault::Runtimeless)
        }
    }
}

pub struct Comm {
    raw: *mut c_void,
    rank: u32,
    size: u32,
    peers: Option<kernels_cuda::collective::Peers>,
}

// SAFETY: an NCCL communicator may be used from any thread as long as calls
// on it are not concurrent; the group drives each rank's collectives from that
// rank's one lane thread.
unsafe impl Send for Comm {}
// SAFETY: the group shares `Arc<Comm>` with its abort closures. Through `&self`
// a `Comm` only hands out its immutable fields (`raw`, `peers`, `rank`,
// `size`) or calls `ncclCommAbort`, which NCCL allows from another thread
// while a collective runs on the communicator; `Poison::set` runs the aborts
// once.
unsafe impl Sync for Comm {}

impl Comm {
    pub fn open(id: &Id, rank: u32, size: u32) -> Result<Comm> {
        #[cfg(feature = "device")]
        {
            use crate::device::nccl;
            let unique = nccl::ncclUniqueId {
                internal: id.0.map(|byte| byte as core::ffi::c_char),
            };
            let mut raw: nccl::ncclComm_t = core::ptr::null_mut();
            // SAFETY: a live out-parameter; `unique` is by value, as the
            // binding declares it.
            let code = unsafe {
                nccl::ncclCommInitRank(
                    &raw mut raw,
                    size as core::ffi::c_int,
                    unique,
                    rank as core::ffi::c_int,
                )
            };
            answered("ncclCommInitRank", code)?;
            Ok(Comm {
                raw: raw.cast(),
                rank,
                size,
                peers: None,
            })
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = (id, rank, size);
            Err(Fault::Runtimeless)
        }
    }

    #[must_use]
    pub fn raw(&self) -> *mut c_void {
        self.raw
    }

    pub fn abort(&self) {
        #[cfg(feature = "device")]
        {
            use crate::device::nccl;
            if self.raw.is_null() {
                return;
            }
            // SAFETY: a live communicator this group opened; NCCL allows an
            // abort from any thread while other threads are inside calls on
            // the same communicator — that is what it is for.
            let _ = unsafe { nccl::ncclCommAbort(self.raw.cast()) };
        }
    }

    #[must_use]
    pub fn peers(&self) -> Option<kernels_cuda::collective::Peers> {
        self.peers
    }

    pub fn set_peers(&mut self, peers: kernels_cuda::collective::Peers) {
        self.peers = Some(peers);
    }

    #[must_use]
    pub fn rank(&self) -> u32 {
        self.rank
    }

    #[must_use]
    pub fn size(&self) -> u32 {
        self.size
    }
}

impl Drop for Comm {
    fn drop(&mut self) {
        self.raw = core::ptr::null_mut();
    }
}

impl fmt::Debug for Comm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Comm")
            .field("rank", &self.rank)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl PartialEq for Comm {
    fn eq(&self, other: &Comm) -> bool {
        self.raw == other.raw
    }
}

/// Bytes of each rank's peer stage; a longer message goes in stage-sized
/// chunks.
#[cfg(feature = "cuda")]
const STAGE_BYTES: u64 = 16 << 20;

/// Bytes of the stage [`reach`] maps: the handshake's one word, aligned.
#[cfg(feature = "cuda")]
const PROBE_BYTES: u64 = 256;

/// Maps every rank's stage into every other rank when each pair of devices
/// reaches the other: the group's ranks share this process, so a peer's
/// buffer is a plain device pointer once peer access is on. At two ranks the
/// handshake below is the whole decision: a link that carries the kernels'
/// stores and loads carries every message, since the peer all-reduce reads
/// no more than NCCL's ring moves and never crosses host memory (#777).
/// `None` otherwise, and the collectives stay on NCCL; always `None` on ROCm,
/// which has no peer kernels and carries every message on RCCL.
#[must_use]
pub fn open_peers(ordinals: &[i32]) -> Option<Vec<kernels_cuda::collective::Peers>> {
    #[cfg(feature = "cuda")]
    {
        match wire_peers(ordinals) {
            Ok(peers) => {
                eprintln!("engine-cuda: peer collectives carry the group's bf16 messages");
                peers
            }
            Err(fault) => {
                eprintln!("engine-cuda: peer collectives are off, NCCL carries them: {fault}");
                None
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ordinals;
        None
    }
}

/// Whether every rank of `ordinals` loads and stores every other's memory,
/// as the peer kernels do: each pair checked and a handshake run through
/// the mapping, on a stage of one word. What an expert-parallel group
/// needs, at any width; whether the peer all-reduce beats NCCL is
/// [`open_peers`]'s question.
pub fn reach(ordinals: &[i32]) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        mapped(ordinals, PROBE_BYTES).map(drop)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ordinals;
        Err(Fault::program(
            "comm::reach",
            "ROCm has no peer kernels".to_string(),
        ))
    }
}

#[cfg(feature = "cuda")]
fn wire_peers(ordinals: &[i32]) -> Result<Option<Vec<kernels_cuda::collective::Peers>>> {
    use crate::device::rt;

    let off = |why: String| Fault::program("comm::open_peers", why);
    let world = u32::try_from(ordinals.len()).unwrap_or(u32::MAX);
    // Past two ranks every rank reads several peers at once; over PCIe that
    // loses to NCCL at every size, where any pair alone wins (#777). Over
    // NVLink (every pair with native peer atomics, which PCIe does not carry)
    // the one-stage reduce wins at four and eight as vLLM's does.
    if !matches!(world, 2 | 4 | 8) {
        return Err(off(format!("no peer path for a group of {world}")));
    }
    let nvlink = world == 2 || {
        let mut all = true;
        for &a in ordinals {
            for &b in ordinals.iter().filter(|&&b| b != a) {
                let mut atomic = 0;
                // SAFETY: a live out-parameter and two ordinals the runtime counts.
                let asked = unsafe {
                    rt::cudaDeviceGetP2PAttribute(
                        &raw mut atomic,
                        rt::cudaDeviceP2PAttr::cudaDevP2PAttrNativeAtomicSupported,
                        a,
                        b,
                    )
                };
                all &= asked == rt::cudaError::cudaSuccess && atomic != 0;
            }
        }
        all
    };
    if !nvlink {
        return Err(off(format!(
            "no peer path for a group of {world}: its pairs are not all NVLink"
        )));
    }
    mapped(ordinals, STAGE_BYTES).map(Some)
}

/// Every rank's stage of `stage_bytes` (and its signals) mapped into every
/// other's, once each pair of devices reaches the other and a handshake
/// through the mapping finds what was posted.
#[cfg(feature = "cuda")]
fn mapped(ordinals: &[i32], stage_bytes: u64) -> Result<Vec<kernels_cuda::collective::Peers>> {
    use crate::device::rt;

    use crate::device::ctx::check;

    let off = |why: String| Fault::program("comm::peers", why);
    let world = u32::try_from(ordinals.len()).unwrap_or(u32::MAX);
    for &a in ordinals {
        let mut vmm = 0;
        // SAFETY: a live out-parameter; a driver device is its ordinal.
        let asked = unsafe {
            crate::device::dr::cuDeviceGetAttribute(
                &raw mut vmm,
                crate::device::dr::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                a,
            )
        };
        if asked != crate::device::dr::CUresult::CUDA_SUCCESS || vmm == 0 {
            return Err(off(format!("cuda:{a} does not map memory for peers")));
        }
        for &b in ordinals.iter().filter(|&&b| b != a) {
            let mut reach = 0;
            let mut native = 0;
            // SAFETY: live out-parameters and two ordinals the runtime counts.
            unsafe {
                check(
                    "cudaDeviceCanAccessPeer",
                    rt::cudaDeviceCanAccessPeer(&raw mut reach, a, b),
                )?;
                check(
                    "cudaDeviceGetP2PAttribute",
                    rt::cudaDeviceGetP2PAttribute(
                        &raw mut native,
                        rt::cudaDeviceP2PAttr::cudaDevP2PAttrAccessSupported,
                        a,
                        b,
                    ),
                )?;
            }
            if reach == 0 || native == 0 {
                return Err(off(format!("cuda:{a} cannot access cuda:{b}")));
            }
        }
    }
    let table = 2 * size_of::<[u64; 8]>();
    let mut stages = [0u64; 8];
    let mut signals = [0u64; 8];
    let mut tables = Vec::with_capacity(ordinals.len());
    for (rank, &a) in ordinals.iter().enumerate() {
        // SAFETY: each call takes live out-parameters or pointers this loop
        // just allocated on device `a`.
        unsafe {
            check("cudaSetDevice", rt::cudaSetDevice(a))?;
            let signal_bytes = kernels_cuda::collective::SIGNAL_BYTES as usize + table;
            let stage =
                crate::device::elastic::shared(a, ordinals, stage_bytes + signal_bytes as u64)?;
            let signal = stage + stage_bytes;
            crate::device::zero_span(signal, signal_bytes)?;
            stages[rank] = stage;
            signals[rank] = signal;
            tables.push(signal + kernels_cuda::collective::SIGNAL_BYTES);
        }
    }
    let mut bytes = Vec::with_capacity(table);
    for word in stages.iter().chain(&signals) {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let mut peers = Vec::with_capacity(ordinals.len());
    for (rank, &a) in ordinals.iter().enumerate() {
        // SAFETY: the table region is `table` bytes past this rank's signal,
        // and the stage is at least a word.
        unsafe {
            check("cudaSetDevice", rt::cudaSetDevice(a))?;
            let own = crate::device::alloc::OWN_STREAM;
            check(
                "cudaMemcpyAsync",
                rt::cudaMemcpyAsync(
                    tables[rank] as *mut c_void,
                    bytes.as_ptr().cast(),
                    table,
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    own.cast(),
                ),
            )?;
            check(
                "cudaMemsetAsync",
                rt::cudaMemsetAsync(
                    stages[rank] as *mut c_void,
                    0x40 + rank as i32,
                    4,
                    own.cast(),
                ),
            )?;
            crate::device::ctx::sync(own)?;
        }
        peers.push(kernels_cuda::collective::Peers {
            rank: rank as u32,
            world,
            stage: stages[rank],
            stage_bytes,
            signal: signals[rank],
            stages: tables[rank],
            signals: tables[rank] + size_of::<[u64; 8]>() as u64,
        });
    }
    // What the driver advertises is not always what the link carries (#757).
    // Every rank posts to and reads from every peer with the stores and loads
    // the collectives use, but in a kernel that never waits: the first round
    // posts, and the second, after every device has drained, finds what
    // arrived.
    let mut statuses = Vec::with_capacity(ordinals.len());
    for (&a, _) in ordinals.iter().zip(&peers) {
        crate::device::ctx::bind_thread(a)?;
        statuses.push(crate::device::Buffer::zeroed(4)?);
    }
    for _ in 0..2 {
        for ((&a, peers), status) in ordinals.iter().zip(&peers).zip(&statuses) {
            crate::device::ctx::bind_thread(a)?;
            // SAFETY: this thread's own stream on the device; the stages and
            // tables are live.
            let ctx = unsafe { kernels_cuda::Ctx::on(crate::device::alloc::OWN_STREAM) };
            peers
                .answer(&ctx, status.ptr())
                .map_err(crate::error::kernel)?;
        }
        for &a in ordinals {
            crate::device::ctx::bind_thread(a)?;
            // The rounds' kernels, on this thread's own stream of the device.
            crate::device::ctx::sync(crate::device::alloc::OWN_STREAM)?;
        }
    }
    for (&a, status) in ordinals.iter().zip(&statuses) {
        crate::device::ctx::bind_thread(a)?;
        let mut word = [0u8; 4];
        status.read(0, &mut word)?;
        match u32::from_le_bytes(word) {
            0 => {}
            1 => {
                return Err(off(format!(
                    "cuda:{a} heard nothing from a peer through the mapping"
                )));
            }
            _ => {
                return Err(off(format!(
                    "cuda:{a} read the wrong word from a peer's stage"
                )));
            }
        }
    }
    Ok(peers)
}

/// `shm` takes no device-to-device path, and NVLS (NVSwitch multicast) is
/// one: left on, NCCL tries it even with P2P off, and in a container
/// without the multicast capability every communicator fails to open
/// (#500).
#[cfg(feature = "device")]
fn transport_defaults(transport: Transport) {
    let set: &[(&str, &str)] = match transport {
        Transport::Shm => &[("NCCL_P2P_DISABLE", "1"), ("NCCL_NVLS_ENABLE", "0")],
        Transport::Peer => &[("NCCL_P2P_DISABLE", "0")],
        Transport::Nccl => return,
    };
    for (key, value) in set {
        // SAFETY: called before any communicator exists, from the group
        // opener, on the thread that starts the rank threads.
        unsafe { std::env::set_var(key, value) };
    }
}

#[cfg(feature = "device")]
fn answered(call: &'static str, code: crate::device::nccl::ncclResult_t) -> Result<()> {
    if code == crate::device::nccl::ncclResult_t::ncclSuccess {
        Ok(())
    } else {
        Err(Fault::Device {
            call,
            code: code as i32,
        })
    }
}
