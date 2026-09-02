//! NIXL backend: moves KV pages between workers over UCX. Built only under `--features nixl`.
//! Credential is an opaque metadata blob (not an ibverbs rkey); all agent calls are serialized behind one mutex since NIXL is not thread-safe.

mod ffi;

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;
use std::sync::Mutex;

use crate::core::{
    Backend, BackendKind, Completion, PageSet, PeerConn, RegisteredHandle, TransferId, WorkerId,
};
use crate::error::{Result, TransportError};
use engine::{KvHandle, MemoryDomain};
use ffi::*;

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
    /// In-flight transfer requests, by this backend's inner id.
    reqs: HashMap<u64, Request>,
    next_id: u64,
}

#[derive(Clone, Copy)]
struct Request {
    handle: nixl_capi_xfer_req_t,
    released: bool,
}

/// Cross-node NIXL backend. One NIXL agent + UCX plugin per instance.
pub struct NixlBackend {
    inner: Mutex<Inner>,
}

// SAFETY: every NIXL call goes through `inner`'s mutex, so the agent is never
// touched concurrently (NIXL is not thread-safe). The raw handles are only
// dereferenced by NIXL itself under that lock.
unsafe impl Send for NixlBackend {}
unsafe impl Sync for NixlBackend {}

impl NixlBackend {
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
        local_pages: &PageSet,
        remote_pages: &PageSet,
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
            if local_pages.len() != remote_pages.len() {
                return Err(TransportError::LayoutMismatch);
            }
            let local_dl = build_xfer_dlist(local.handle(), local_pages)?;
            let remote_dl = match build_xfer_dlist(&remote_handle, remote_pages) {
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

            let _post_status = nixl_capi_post_xfer_req(g.agent, req, ptr::null_mut());
            // A post error can still leave an agent-owned request; keep it pollable so the normal path releases it.

            let id = g.next_id;
            g.next_id += 1;
            g.reqs.insert(
                id,
                Request {
                    handle: req,
                    released: false,
                },
            );
            Ok(TransferId(id))
        }
    }
}

/// Build a NIXL transfer descriptor list for every physical region slice of
/// each logical KV page.
unsafe fn build_xfer_dlist(handle: &KvHandle, pages: &PageSet) -> Result<nixl_capi_xfer_dlist_t> {
    let first = handle
        .regions
        .first()
        .ok_or(TransportError::Unsupported("handle has no KV region"))?;
    let mut dl: nixl_capi_xfer_dlist_t = ptr::null_mut();
    let st = unsafe { nixl_capi_create_xfer_dlist(mem_type(first.domain), &mut dl) };
    if st != OK {
        return Err(err("create_xfer_dlist", st));
    }
    for &page in &pages.pages {
        for region in &handle.regions {
            if mem_type(region.domain) != mem_type(first.domain) || region.page_stride == 0 {
                unsafe { nixl_capi_destroy_xfer_dlist(dl) };
                return Err(TransportError::LayoutMismatch);
            }
            let offset = page as u64 * region.page_stride;
            if offset + region.page_stride > region.len {
                unsafe { nixl_capi_destroy_xfer_dlist(dl) };
                return Err(TransportError::PageOutOfBounds { page });
            }
            let st = unsafe {
                nixl_capi_xfer_dlist_add_desc(
                    dl,
                    (region.base + offset) as usize,
                    region.page_stride as usize,
                    dev_id(region.domain),
                )
            };
            if st != OK {
                unsafe { nixl_capi_destroy_xfer_dlist(dl) };
                return Err(err("xfer_dlist_add_desc", st));
            }
        }
    }
    Ok(dl)
}

impl Backend for NixlBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Nixl
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
            backend: BackendKind::Nixl,
            owner,
            handle,
        })
    }

    fn send_mapped(
        &self,
        handle: &RegisteredHandle,
        src_pages: &PageSet,
        dst_pages: &PageSet,
        dst: WorkerId,
    ) -> Result<TransferId> {
        self.xfer(WRITE, handle, src_pages, dst_pages, dst)
    }

    fn recv_mapped(
        &self,
        slot: &RegisteredHandle,
        dst_pages: &PageSet,
        src_pages: &PageSet,
        src: WorkerId,
    ) -> Result<TransferId> {
        self.xfer(READ, slot, dst_pages, src_pages, src)
    }

    fn poll(&self, id: TransferId) -> Result<Completion> {
        let mut g = self.inner.lock().unwrap();
        let request = *g
            .reqs
            .get(&id.0)
            .ok_or(TransportError::UnknownTransfer { id: id.0 })?;
        if request.released {
            let status = unsafe { nixl_capi_destroy_xfer_req(request.handle) };
            if status != OK {
                return Err(err("destroy_xfer_req", status));
            }
            g.reqs.remove(&id.0);
            return Ok(Completion::Done);
        }
        let st = unsafe { nixl_capi_get_xfer_status(g.agent, request.handle) };
        match st {
            s if s == OK => {
                let status = unsafe { nixl_capi_release_xfer_req(g.agent, request.handle) };
                if status != OK {
                    return Err(err("release_xfer_req", status));
                }
                g.reqs.get_mut(&id.0).expect("request remains").released = true;
                let status = unsafe { nixl_capi_destroy_xfer_req(request.handle) };
                if status != OK {
                    return Err(err("destroy_xfer_req", status));
                }
                g.reqs.remove(&id.0);
                Ok(Completion::Done)
            }
            s if s == INPROG => Ok(Completion::Pending),
            other => {
                let status = unsafe { nixl_capi_release_xfer_req(g.agent, request.handle) };
                if status != OK {
                    return Err(err("release_xfer_req", status));
                }
                g.reqs.get_mut(&id.0).expect("request remains").released = true;
                let status = unsafe { nixl_capi_destroy_xfer_req(request.handle) };
                if status != OK {
                    return Err(err("destroy_xfer_req", status));
                }
                g.reqs.remove(&id.0);
                Ok(Completion::Failed(format!("nixl xfer status {other}")))
            }
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

impl Drop for NixlBackend {
    fn drop(&mut self) {
        let g = self.inner.get_mut().unwrap();
        unsafe {
            for (_, request) in g.reqs.drain() {
                if !request.released {
                    let _ = nixl_capi_release_xfer_req(g.agent, request.handle);
                }
                let _ = nixl_capi_destroy_xfer_req(request.handle);
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

