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
/// buffer is a plain device pointer once peer access is on. `None` when some
/// pair does not, and the collectives stay on NCCL.
#[must_use]
pub fn open_peers(ordinals: &[i32]) -> Option<Vec<kernels_cuda::collective::Peers>> {
    #[cfg(feature = "cuda")]
    {
        match wire_peers(ordinals) {
            Ok(peers) => peers,
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

    const PROBE: usize = 4096;
    let world = u32::try_from(ordinals.len()).unwrap_or(u32::MAX);
    if !matches!(world, 2 | 4 | 8) {
        return Ok(None);
    }
    for &a in ordinals {
        for &b in ordinals.iter().filter(|&&b| b != a) {
            let mut reach = 0;
            // SAFETY: a live out-parameter and two ordinals the runtime counts.
            check("cudaDeviceCanAccessPeer", unsafe {
                rt::cudaDeviceCanAccessPeer(&raw mut reach, a, b)
            })?;
            if reach == 0 {
                return Ok(None);
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
            for &b in ordinals.iter().filter(|&&b| b != a) {
                let code = rt::cudaDeviceEnablePeerAccess(b, 0);
                if code != rt::cudaError::cudaErrorPeerAccessAlreadyEnabled {
                    check("cudaDeviceEnablePeerAccess", code)?;
                }
            }
            let mut stage: *mut c_void = core::ptr::null_mut();
            check(
                "cudaMalloc",
                rt::cudaMalloc(&raw mut stage, STAGE_BYTES as usize),
            )?;
            let mut signal: *mut c_void = core::ptr::null_mut();
            let signal_bytes = kernels_cuda::collective::SIGNAL_BYTES as usize + table;
            check("cudaMalloc", rt::cudaMalloc(&raw mut signal, signal_bytes))?;
            check("cudaMemset", rt::cudaMemset(signal, 0, signal_bytes))?;
            stages[rank] = stage as u64;
            signals[rank] = signal as u64;
            tables.push(signal as u64 + kernels_cuda::collective::SIGNAL_BYTES);
        }
    }
    // cudaDeviceCanAccessPeer is what the driver advertises; a copy through
    // the mapping is what the link delivers.
    let mut seen = vec![0u8; PROBE];
    for (a, &from) in ordinals.iter().enumerate() {
        for (b, &to) in ordinals.iter().enumerate().filter(|&(_, &to)| to != from) {
            // SAFETY: both stages are live allocations of at least PROBE bytes.
            unsafe {
                check("cudaSetDevice", rt::cudaSetDevice(to))?;
                check(
                    "cudaMemset",
                    rt::cudaMemset(stages[b] as *mut c_void, 0, PROBE),
                )?;
                check("cudaSetDevice", rt::cudaSetDevice(from))?;
                check(
                    "cudaMemset",
                    rt::cudaMemset(stages[a] as *mut c_void, 0x5a, PROBE),
                )?;
                check(
                    "cudaMemcpyPeer",
                    rt::cudaMemcpyPeer(
                        stages[b] as *mut c_void,
                        to,
                        stages[a] as *const c_void,
                        from,
                        PROBE,
                    ),
                )?;
                check("cudaSetDevice", rt::cudaSetDevice(to))?;
                check(
                    "cudaMemcpy",
                    rt::cudaMemcpy(
                        seen.as_mut_ptr().cast(),
                        stages[b] as *const c_void,
                        PROBE,
                        rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    ),
                )?;
            }
            if seen.iter().any(|&byte| byte != 0x5a) {
                return Ok(None);
            }
        }
    }
    let mut bytes = Vec::with_capacity(table);
    for word in stages.iter().chain(&signals) {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let mut peers = Vec::with_capacity(ordinals.len());
    for (rank, &a) in ordinals.iter().enumerate() {
        // SAFETY: the table region is `table` bytes past this rank's signal.
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
