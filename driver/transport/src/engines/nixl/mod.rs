//! NIXL engine — the cross-node KV-tensor data path.
//!
//! Wraps NVIDIA's NIXL (via its C-API, `libnixl_capi.so`) to move KV pages
//! between workers over UCX (RDMA where a NIC exists, otherwise `shm`/`tcp`).
//! Built only under `--features nixl`; the on-device build ships without it.
//!
//! # Mechanism, not policy
//!
//! The remote-access credential is **not** an ibverbs `rkey` — NIXL exposes
//! none. Instead each agent publishes an opaque *metadata blob*
//! ([`get_local_md`]) that a peer loads ([`load_remote_md`]); that blob is the
//! [`PeerConn::metadata`] the controller's pairing hands over. Combined with the
//! peer's exported [`KvHandle`] (its region addresses), it's everything a
//! one-sided WRITE/READ needs. The handle/region stays backend-neutral.
//!
//! # Lifecycle → NIXL C-API
//!
//! - [`register`](NixlEngine::register) → `register_mem` over a reg-dlist built
//!   from the handle's regions.
//! - [`local_metadata`](NixlEngine::local_metadata) → `get_local_md`.
//! - [`connect`](NixlEngine::connect) → `load_remote_md` (caches the peer's
//!   agent name + exported handle).
//! - [`send`](NixlEngine::send)/[`recv`](NixlEngine::recv) → `create_xfer_req`
//!   (WRITE/READ) + `post_xfer_req`.
//! - [`poll`](NixlEngine::poll) → `get_xfer_status`.
//!
//! NIXL is not safe to drive from multiple threads concurrently (it can
//! deadlock), so all agent calls are serialized behind one mutex.
//!
//! [`get_local_md`]: https://github.com/ai-dynamo/nixl
//! [`load_remote_md`]: https://github.com/ai-dynamo/nixl

mod ffi;

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;
use std::sync::Mutex;

use crate::core::{
    Completion, Engine, EngineKind, PageSet, PeerConn, RegisteredHandle, TransferId, WorkerId,
};
use crate::error::{Result, TransportError};
use ffi::*;
use pie_driver_abi::kv::{KvHandle, MemoryDomain};

const OK: nixl_capi_status_t = nixl_capi_status_t_NIXL_CAPI_SUCCESS;
const INPROG: nixl_capi_status_t = nixl_capi_status_t_NIXL_CAPI_IN_PROG;
const WRITE: nixl_capi_xfer_op_t = nixl_capi_xfer_op_t_NIXL_CAPI_XFER_OP_WRITE;
const READ: nixl_capi_xfer_op_t = nixl_capi_xfer_op_t_NIXL_CAPI_XFER_OP_READ;

fn err(ctx: &str, st: nixl_capi_status_t) -> TransportError {
    TransportError::Transfer(format!("nixl {ctx}: status {st}"))
}

fn mem_type(domain: MemoryDomain) -> nixl_capi_mem_type_t {
    match domain {
        MemoryDomain::HostPinned => nixl_capi_mem_type_t_NIXL_CAPI_MEM_DRAM,
        MemoryDomain::CudaDevice(_) | MemoryDomain::RocmDevice(_) => {
            nixl_capi_mem_type_t_NIXL_CAPI_MEM_VRAM
        }
    }
}

fn dev_id(domain: MemoryDomain) -> u64 {
    match domain {
        MemoryDomain::HostPinned => 0,
        MemoryDomain::CudaDevice(n) | MemoryDomain::RocmDevice(n) => n as u64,
    }
}

/// A connected peer: its NIXL agent name and its exported handle (region
/// addresses to target).
struct Remote {
    agent: CString,
    handle: KvHandle,
}

struct Inner {
    agent: nixl_capi_agent_t,
    backend: nixl_capi_backend_t,
    /// Locally registered handles, by owning worker.
    locals: HashMap<u64, KvHandle>,
    /// Connected remote peers, by worker.
    remotes: HashMap<u64, Remote>,
    /// In-flight transfer requests, by this engine's inner id.
    reqs: HashMap<u64, nixl_capi_xfer_req_t>,
    next_id: u64,
}

/// Cross-node NIXL engine. One NIXL agent + UCX backend per instance.
pub struct NixlEngine {
    inner: Mutex<Inner>,
}

// SAFETY: every NIXL call goes through `inner`'s mutex, so the agent is never
// touched concurrently (NIXL is not thread-safe). The raw handles are only
// dereferenced by NIXL itself under that lock.
unsafe impl Send for NixlEngine {}
unsafe impl Sync for NixlEngine {}

impl NixlEngine {
    /// Create a NIXL agent named `agent_name` with a UCX backend. The agent
    /// name must be unique within the cluster (it's how peers address it).
    pub fn new(agent_name: &str) -> Result<Self> {
        let name = CString::new(agent_name)
            .map_err(|_| TransportError::Transfer("agent name contains a nul byte".into()))?;
        let ucx = CString::new("UCX").expect("UCX literal");
        unsafe {
            let mut agent: nixl_capi_agent_t = ptr::null_mut();
            let st = nixl_capi_create_agent(name.as_ptr(), &mut agent);
            if st != OK {
                return Err(err("create_agent", st));
            }
            let mut mems: nixl_capi_mem_list_t = ptr::null_mut();
            let mut params: nixl_capi_params_t = ptr::null_mut();
            let st = nixl_capi_get_plugin_params(agent, ucx.as_ptr(), &mut mems, &mut params);
            if st != OK {
                return Err(err("get_plugin_params(UCX)", st));
            }
            let mut backend: nixl_capi_backend_t = ptr::null_mut();
            let st = nixl_capi_create_backend(agent, ucx.as_ptr(), params, &mut backend);
            if st != OK {
                return Err(err("create_backend(UCX)", st));
            }
            Ok(Self {
                inner: Mutex::new(Inner {
                    agent,
                    backend,
                    locals: HashMap::new(),
                    remotes: HashMap::new(),
                    reqs: HashMap::new(),
                    next_id: 0,
                }),
            })
        }
    }

    fn xfer(
        &self,
        op: nixl_capi_xfer_op_t,
        local: &RegisteredHandle,
        pages: &PageSet,
        peer: WorkerId,
    ) -> Result<TransferId> {
        let mut g = self.inner.lock().unwrap();
        let (rname, remote_handle) = {
            let r = g
                .remotes
                .get(&peer.0)
                .ok_or(TransportError::UnknownPeer { worker: peer.0 })?;
            (r.agent.clone(), r.handle.clone())
        };
        if !local.handle().layout.compatible_with(&remote_handle.layout) {
            return Err(TransportError::LayoutMismatch);
        }

        unsafe {
            let local_dl = build_xfer_dlist(local.handle(), pages)?;
            let remote_dl = match build_xfer_dlist(&remote_handle, pages) {
                Ok(dl) => dl,
                Err(e) => {
                    nixl_capi_destroy_xfer_dlist(local_dl);
                    return Err(e);
                }
            };

            let mut req: nixl_capi_xfer_req_t = ptr::null_mut();
            let st = nixl_capi_create_xfer_req(
                g.agent,
                op,
                local_dl,
                remote_dl,
                rname.as_ptr(),
                &mut req,
                ptr::null_mut(),
            );
            nixl_capi_destroy_xfer_dlist(local_dl);
            nixl_capi_destroy_xfer_dlist(remote_dl);
            if st != OK {
                return Err(err("create_xfer_req", st));
            }

            let st = nixl_capi_post_xfer_req(g.agent, req, ptr::null_mut());
            if st != OK && st != INPROG {
                nixl_capi_destroy_xfer_req(req);
                return Err(err("post_xfer_req", st));
            }

            let id = g.next_id;
            g.next_id += 1;
            g.reqs.insert(id, req);
            Ok(TransferId(id))
        }
    }
}

/// Build a NIXL transfer descriptor list for `pages` of `handle`. Addresses
/// only `regions.first()` — the single-contiguous-region assumption shared with
/// the local engine; multi-region handles are out of current scope.
unsafe fn build_xfer_dlist(handle: &KvHandle, pages: &PageSet) -> Result<nixl_capi_xfer_dlist_t> {
    let region = handle
        .regions
        .first()
        .ok_or(TransportError::Unsupported("handle has no KV region"))?;
    let page_bytes = handle.page_bytes();
    let mut dl: nixl_capi_xfer_dlist_t = ptr::null_mut();
    let st = unsafe { nixl_capi_create_xfer_dlist(mem_type(region.domain), &mut dl) };
    if st != OK {
        return Err(err("create_xfer_dlist", st));
    }
    for &page in &pages.pages {
        let offset = page as u64 * page_bytes;
        if offset + page_bytes > region.len {
            unsafe { nixl_capi_destroy_xfer_dlist(dl) };
            return Err(TransportError::PageOutOfBounds { page });
        }
        let st = unsafe {
            nixl_capi_xfer_dlist_add_desc(
                dl,
                (region.base + offset) as usize,
                page_bytes as usize,
                dev_id(region.domain),
            )
        };
        if st != OK {
            unsafe { nixl_capi_destroy_xfer_dlist(dl) };
            return Err(err("xfer_dlist_add_desc", st));
        }
    }
    Ok(dl)
}

impl Engine for NixlEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::Nixl
    }

    fn register(&self, owner: WorkerId, handle: KvHandle) -> Result<RegisteredHandle> {
        let mut g = self.inner.lock().unwrap();
        for region in &handle.regions {
            unsafe {
                let mut dl: nixl_capi_reg_dlist_t = ptr::null_mut();
                let st = nixl_capi_create_reg_dlist(mem_type(region.domain), &mut dl);
                if st != OK {
                    return Err(err("create_reg_dlist", st));
                }
                let st = nixl_capi_reg_dlist_add_desc(
                    dl,
                    region.base as usize,
                    region.len as usize,
                    dev_id(region.domain),
                    ptr::null(),
                    0,
                );
                if st != OK {
                    nixl_capi_destroy_reg_dlist(dl);
                    return Err(err("reg_dlist_add_desc", st));
                }
                let st = nixl_capi_register_mem(g.agent, dl, ptr::null_mut());
                nixl_capi_destroy_reg_dlist(dl);
                if st != OK {
                    return Err(err("register_mem", st));
                }
            }
        }
        g.locals.insert(owner.0, handle.clone());
        Ok(RegisteredHandle {
            engine: EngineKind::Nixl,
            owner,
            handle,
        })
    }

    fn send(
        &self,
        handle: &RegisteredHandle,
        pages: &PageSet,
        dst: WorkerId,
    ) -> Result<TransferId> {
        self.xfer(WRITE, handle, pages, dst)
    }

    fn recv(&self, slot: &RegisteredHandle, pages: &PageSet, src: WorkerId) -> Result<TransferId> {
        self.xfer(READ, slot, pages, src)
    }

    fn poll(&self, id: TransferId) -> Result<Completion> {
        let g = self.inner.lock().unwrap();
        let req = *g
            .reqs
            .get(&id.0)
            .ok_or(TransportError::UnknownTransfer { id: id.0 })?;
        let st = unsafe { nixl_capi_get_xfer_status(g.agent, req) };
        match st {
            s if s == OK => Ok(Completion::Done),
            s if s == INPROG => Ok(Completion::Pending),
            other => Ok(Completion::Failed(format!("nixl xfer status {other}"))),
        }
    }

    fn connect(&self, peer: &PeerConn) -> Result<()> {
        let mut g = self.inner.lock().unwrap();
        let mut rname: *mut c_char = ptr::null_mut();
        let agent = unsafe {
            let st = nixl_capi_load_remote_md(
                g.agent,
                peer.metadata.as_ptr() as *const c_void,
                peer.metadata.len(),
                &mut rname,
            );
            if st != OK {
                return Err(err("load_remote_md", st));
            }
            CStr::from_ptr(rname).to_owned()
        };
        g.remotes.insert(
            peer.worker.0,
            Remote {
                agent,
                handle: peer.handle.clone(),
            },
        );
        Ok(())
    }

    fn local_metadata(&self) -> Result<Vec<u8>> {
        let g = self.inner.lock().unwrap();
        unsafe {
            let mut data: *mut c_void = ptr::null_mut();
            let mut len: usize = 0;
            let st = nixl_capi_get_local_md(g.agent, &mut data, &mut len);
            if st != OK {
                return Err(err("get_local_md", st));
            }
            Ok(std::slice::from_raw_parts(data as *const u8, len).to_vec())
        }
    }
}

impl Drop for NixlEngine {
    fn drop(&mut self) {
        let g = self.inner.get_mut().unwrap();
        unsafe {
            for (_, req) in g.reqs.drain() {
                nixl_capi_destroy_xfer_req(req);
            }
            if !g.backend.is_null() {
                nixl_capi_destroy_backend(g.backend);
            }
            if !g.agent.is_null() {
                nixl_capi_destroy_agent(g.agent);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_driver_abi::kv::{KvDtype, KvLayout, KvLayoutKind};

    fn layout() -> KvLayout {
        // 1 layer · KvSeparate(2 planes) · 1 head · head_dim 64 · page_size 16 ·
        // Bf16 → page_bytes() = 4096 (matches the dummy host-DRAM export).
        KvLayout {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 64,
            page_size: 16,
            dtype: KvDtype::Bf16,
            kind: KvLayoutKind::KvSeparate,
        }
    }

    fn host_handle(buf: &[u8]) -> KvHandle {
        KvHandle {
            regions: vec![pie_driver_abi::kv::KvRegion {
                base: buf.as_ptr() as u64,
                len: buf.len() as u64,
                domain: MemoryDomain::HostPinned,
            }],
            layout: layout(),
        }
    }

    /// Real cross-engine NIXL transfer over UCX `shm,tcp` — no RDMA NIC, no GPU.
    /// Two agents in one process: A WRITEs host-DRAM pages into B's buffer.
    ///
    /// Skips (does not fail) when NIXL isn't runnable here. Run it with:
    /// ```text
    /// NIXL_PREFIX=/path/to/nixl_prefix \
    /// LD_LIBRARY_PATH=$NIXL_PREFIX/lib:$NIXL_PREFIX/lib/ucx \
    ///   cargo test -p pie-transport --features nixl -- --test-threads=1 nixl_
    /// ```
    #[test]
    fn nixl_ucx_shm_tcp_transfer() {
        let Ok(prefix) = std::env::var("NIXL_PREFIX") else {
            eprintln!("SKIP: NIXL_PREFIX unset");
            return;
        };
        // Read lazily by NIXL/UCX at backend creation, so setting them here works.
        unsafe {
            std::env::set_var("NIXL_PLUGIN_DIR", format!("{prefix}/lib/plugins"));
            std::env::set_var("UCX_MODULE_DIR", format!("{prefix}/lib/ucx"));
            if std::env::var("UCX_TLS").is_err() {
                std::env::set_var("UCX_TLS", "shm,tcp");
            }
        }

        let a = match NixlEngine::new("A-e2e") {
            Ok(a) => a,
            Err(e) => {
                eprintln!("SKIP: NIXL agent init failed ({e}) — check LD_LIBRARY_PATH");
                return;
            }
        };
        let b = NixlEngine::new("B-e2e").expect("B agent");

        let page_bytes = layout().page_bytes() as usize;
        let src = vec![0xABu8; page_bytes * 8];
        let dst = vec![0u8; page_bytes * 8];

        let ra = a
            .register(WorkerId(1), host_handle(&src))
            .expect("register A");
        b.register(WorkerId(2), host_handle(&dst))
            .expect("register B");

        let md_b = b.local_metadata().expect("B local_metadata");
        a.connect(&PeerConn {
            worker: WorkerId(2),
            handle: host_handle(&dst),
            metadata: md_b,
        })
        .expect("A connect B");

        let pages = PageSet::new(vec![0, 1]);
        let id = a.send(&ra, &pages, WorkerId(2)).expect("send");

        let mut spins = 0;
        loop {
            match a.poll(id).expect("poll") {
                Completion::Done => break,
                Completion::Pending => {
                    spins += 1;
                    assert!(spins < 5_000_000, "transfer timed out");
                }
                Completion::Failed(m) => panic!("transfer failed: {m}"),
            }
        }

        assert!(
            dst[..page_bytes * 2].iter().all(|&x| x == 0xAB),
            "transferred pages must hold the 0xAB pattern"
        );
        assert!(
            dst[page_bytes * 2..].iter().all(|&x| x == 0),
            "untransferred pages must stay zero"
        );
    }
}
