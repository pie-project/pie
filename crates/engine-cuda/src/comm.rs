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
                 `shm` (P2P off, the default), `peer` (P2P on), and `nccl` \
                 (whatever NCCL's own environment says)"
            )),
        }
    }
}

#[derive(Clone)]
pub struct Id(pub [u8; 128]);

impl Id {
    pub fn new(transport: Transport) -> Result<Id> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::nccl::sys as nccl;
            transport_defaults(transport);
            let mut id = nccl::ncclUniqueId { internal: [0; 128] };
            // SAFETY: a live out-parameter of the exact type NCCL writes.
            let code = unsafe { nccl::ncclGetUniqueId(&raw mut id) };
            answered("ncclGetUniqueId", code)?;
            Ok(Id(id.internal.map(|byte| byte as u8)))
        }
        #[cfg(not(feature = "cuda"))]
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

// SAFETY: NCCL communicators are used from any thread as long as calls on one
// communicator are not concurrent, which the group guarantees by driving each
// rank from one thread at a time.
unsafe impl Send for Comm {}
unsafe impl Sync for Comm {}

impl Comm {
    pub fn open(id: &Id, rank: u32, size: u32) -> Result<Comm> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::nccl::sys as nccl;
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
        #[cfg(not(feature = "cuda"))]
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
        #[cfg(feature = "cuda")]
        {
            use cudarc::nccl::sys as nccl;
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
const STAGE_BYTES: u64 = 16 << 20;

/// Maps every rank's stage into every other rank when each pair of devices
/// reaches the other: the group's ranks share this process, so a peer's
/// buffer is a plain device pointer once peer access is on. At two ranks the
/// handshake below is the whole decision: a link that carries the kernels'
/// stores and loads carries every message, since the peer all-reduce reads
/// no more than NCCL's ring moves and never crosses host memory (#777).
/// `None` otherwise, and the collectives stay on NCCL.
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

#[cfg(feature = "cuda")]
fn wire_peers(ordinals: &[i32]) -> Result<Option<Vec<kernels_cuda::collective::Peers>>> {
    use cudarc::runtime::sys as rt;

    use crate::device::ctx::check;

    let off = |why: String| Fault::program("comm::open_peers", why);
    let world = u32::try_from(ordinals.len()).unwrap_or(u32::MAX);
    // Past two ranks every rank reads several peers at once; over PCIe that
    // loses to NCCL at every size, where any pair alone wins (#777).
    if world != 2 {
        return Err(off(format!("no peer path for a group of {world}")));
    }
    for &a in ordinals {
        let mut vmm = 0;
        // SAFETY: a live out-parameter; a driver device is its ordinal.
        let asked = unsafe {
            cudarc::driver::sys::cuDeviceGetAttribute(
                &raw mut vmm,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                a,
            )
        };
        if asked != cudarc::driver::sys::CUresult::CUDA_SUCCESS || vmm == 0 {
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
                crate::device::elastic::shared(a, ordinals, STAGE_BYTES + signal_bytes as u64)?;
            let signal = stage + STAGE_BYTES;
            check(
                "cudaMemset",
                rt::cudaMemset(signal as *mut c_void, 0, signal_bytes),
            )?;
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
            check(
                "cudaMemcpy",
                rt::cudaMemcpy(
                    tables[rank] as *mut c_void,
                    bytes.as_ptr().cast(),
                    table,
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                ),
            )?;
            check(
                "cudaMemset",
                rt::cudaMemset(stages[rank] as *mut c_void, 0x40 + rank as i32, 4),
            )?;
        }
        peers.push(kernels_cuda::collective::Peers {
            rank: rank as u32,
            world,
            stage: stages[rank],
            stage_bytes: STAGE_BYTES,
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
            // SAFETY: the device's legacy stream; the stages and tables are live.
            let ctx = unsafe { kernels_cuda::Ctx::on(core::ptr::null_mut()) };
            peers
                .answer(&ctx, status.ptr())
                .map_err(crate::error::kernel)?;
        }
        for &a in ordinals {
            crate::device::ctx::bind_thread(a)?;
            // SAFETY: no arguments; waits for the rounds' kernels.
            check("cudaDeviceSynchronize", unsafe {
                rt::cudaDeviceSynchronize()
            })?;
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
    Ok(Some(peers))
}

#[cfg(feature = "cuda")]
fn transport_defaults(transport: Transport) {
    let disable_p2p = match transport {
        Transport::Shm => "1",
        Transport::Peer => "0",
        Transport::Nccl => return,
    };
    // SAFETY: called before any communicator exists, from the group opener,
    // on the thread that starts the rank threads.
    unsafe { std::env::set_var("NCCL_P2P_DISABLE", disable_p2p) };
}

#[cfg(feature = "cuda")]
fn answered(call: &'static str, code: cudarc::nccl::sys::ncclResult_t) -> Result<()> {
    if code == cudarc::nccl::sys::ncclResult_t::ncclSuccess {
        Ok(())
    } else {
        Err(Fault::Device {
            call,
            code: code as i32,
        })
    }
}
