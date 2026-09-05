//! The NCCL communicator a tensor-parallel rank fires its collectives on.
//!
//! One [`Comm`] per rank, opened together by [`open_group`](crate::open_group)
//! and carried onto that rank's kernel context
//! (`kernels_cuda::jit::Ctx::with_comm`). The collectives themselves live in
//! `kernels-cuda`; this module only owns the handle's life: opened once, on a
//! thread bound to the rank's device, and destroyed idle — never aborted,
//! since an abort followed by a destroy was observed to spin forever.

use core::ffi::c_void;
use std::fmt;

use crate::error::{Fault, Result};

/// `ncclUniqueId`: the 128 bytes every rank of one group opens with.
#[derive(Clone)]
pub struct Id(pub [u8; 128]);

impl Id {
    /// A fresh group identity, from rank 0's process.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] with no runtime, [`Fault::Device`] when NCCL
    /// refused.
    pub fn new() -> Result<Id> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::nccl::sys as nccl;
            let mut id = nccl::ncclUniqueId { internal: [0; 128] };
            // SAFETY: a live out-parameter of the exact type NCCL writes.
            let code = unsafe { nccl::ncclGetUniqueId(&raw mut id) };
            answered("ncclGetUniqueId", code)?;
            Ok(Id(id.internal.map(|byte| byte as u8)))
        }
        #[cfg(not(feature = "cuda"))]
        Err(Fault::Runtimeless)
    }
}

/// One rank's open communicator.
pub struct Comm {
    raw: *mut c_void,
    rank: u32,
    size: u32,
}

// SAFETY: NCCL communicators are used from any thread as long as calls on one
// communicator are not concurrent, which the group guarantees by driving each
// rank from one thread at a time.
unsafe impl Send for Comm {}
unsafe impl Sync for Comm {}

impl Comm {
    /// Opens rank `rank` of a `size`-wide group. Collective: every rank must
    /// call this at the same time with the same `id`, from a thread whose
    /// current device is that rank's — `ncclCommInitRank` blocks until the
    /// whole group has arrived.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] with no runtime, [`Fault::Device`] when NCCL
    /// refused.
    pub fn open(id: &Id, rank: u32, size: u32) -> Result<Comm> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::nccl::sys as nccl;
            transport_defaults();
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
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (id, rank, size);
            Err(Fault::Runtimeless)
        }
    }

    /// The `ncclComm_t`, for the kernel context.
    #[must_use]
    pub fn raw(&self) -> *mut c_void {
        self.raw
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
    /// The communicator is left to the process. `ncclCommDestroy` waits for
    /// the whole clique and for anything still queued on the rank's stream;
    /// a group torn down after a refused boot has ranks whose peers never
    /// arrived, and the destroy then spins forever (observed on both this
    /// engine and the driver before it). A deployment opens one group for
    /// its life, so the leak is bounded by the process.
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

/// The ranks of one group are threads of one process. On PCIe boxes NCCL's
/// P2P transport can wedge before it falls back (observed on a 2x L40S pair),
/// so shared memory is the default transport; an operator who has stated a
/// policy keeps it.
#[cfg(feature = "cuda")]
fn transport_defaults() {
    if std::env::var_os("NCCL_P2P_DISABLE").is_none() {
        // SAFETY: called before any communicator exists, from the group
        // opener, on the thread that starts the rank threads.
        unsafe { std::env::set_var("NCCL_P2P_DISABLE", "1") };
    }
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
