mod ffi;
pub mod objects;

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

/// The NIXL segment a domain's memory registers as, and its device id: host
/// memory as DRAM, a CUDA or ROCm device's as VRAM. The rest (Metal, Vulkan,
/// wgpu, XLA memory, a slot file) NIXL cannot reach.
fn segment(domain: MemoryDomain) -> Result<(nixl_capi_mem_type_t, u64)> {
    match domain {
        MemoryDomain::HostPinned => Ok((nixl_capi_mem_type_t_NIXL_CAPI_MEM_DRAM, 0)),
        MemoryDomain::CudaDevice(n) | MemoryDomain::RocmDevice(n) => {
            Ok((nixl_capi_mem_type_t_NIXL_CAPI_MEM_VRAM, u64::from(n)))
        }
        MemoryDomain::MetalShared
        | MemoryDomain::MetalPrivate
        | MemoryDomain::VulkanDevice(_)
        | MemoryDomain::WgpuDevice(_)
        | MemoryDomain::XlaDevice(_)
        | MemoryDomain::LocalDisk => Err(TransportError::Unsupported(
            "NIXL reaches host memory and CUDA or ROCm device memory only",
        )),
    }
}

struct Remote {
    agent: CString,
    handle: KvHandle,
}

struct Inner {
    agent: nixl_capi_agent_t,
    backend: nixl_capi_backend_t,
    locals: HashMap<u64, KvHandle>,
    remotes: HashMap<u64, Remote>,
    reqs: HashMap<u64, Request>,
    next_id: u64,
}

#[derive(Clone, Copy)]
struct Request {
    handle: nixl_capi_xfer_req_t,
    released: bool,
}

pub struct NixlBackend {
    inner: Mutex<Inner>,
}

// SAFETY: every NIXL call goes through `inner`'s mutex, so the agent is never
// touched concurrently (NIXL is not thread-safe). The raw handles are only
// dereferenced by NIXL itself under that lock.
unsafe impl Send for NixlBackend {}
unsafe impl Sync for NixlBackend {}

impl NixlBackend {
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

unsafe fn build_xfer_dlist(handle: &KvHandle, pages: &PageSet) -> Result<nixl_capi_xfer_dlist_t> {
    let first = handle
        .regions
        .first()
        .ok_or(TransportError::Unsupported("handle has no KV region"))?;
    let (kind, _) = segment(first.domain)?;
    let mut dl: nixl_capi_xfer_dlist_t = ptr::null_mut();
    let st = unsafe { nixl_capi_create_xfer_dlist(kind, &mut dl) };
    if st != OK {
        return Err(err("create_xfer_dlist", st));
    }
    for &page in &pages.pages {
        for region in &handle.regions {
            let device = match segment(region.domain) {
                Ok((other, device)) if other == kind && region.page_stride != 0 => device,
                _ => {
                    unsafe { nixl_capi_destroy_xfer_dlist(dl) };
                    return Err(TransportError::LayoutMismatch);
                }
            };
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
                    device,
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
            let (kind, device) = segment(region.domain)?;
            unsafe {
                let mut dl: nixl_capi_reg_dlist_t = ptr::null_mut();
                let st = nixl_capi_create_reg_dlist(kind, &mut dl);
                if st != OK {
                    return Err(err("create_reg_dlist", st));
                }
                let st = nixl_capi_reg_dlist_add_desc(
                    dl,
                    region.base as usize,
                    region.len as usize,
                    device,
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

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{KvLayout, KvLayoutKind, KvRegion};

    const PAGES: u64 = 8;
    const STRIDE: u64 = 4096;

    fn layout() -> KvLayout {
        KvLayout {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 64,
            page_size: 16,
            dtype: dtype::Dtype::Bf16,
            kind: KvLayoutKind::KvSeparate,
            storage_format: "test".to_string(),
            region_page_bytes: vec![STRIDE],
        }
    }

    fn host(bytes: &[u8]) -> KvHandle {
        KvHandle {
            regions: vec![KvRegion {
                base: bytes.as_ptr() as u64,
                len: bytes.len() as u64,
                page_stride: STRIDE,
                domain: MemoryDomain::HostPinned,
            }],
            layout: layout(),
        }
    }

    #[test]
    fn a_read_brings_the_peers_pages_into_the_named_slots() {
        let theirs: Vec<u8> = (0..PAGES * STRIDE)
            .map(|i| (i / STRIDE) as u8 + 1)
            .collect();
        let mut ours = vec![0u8; (PAGES * STRIDE) as usize];
        let (a, b) = (
            NixlBackend::new("home").unwrap(),
            NixlBackend::new("peer").unwrap(),
        );
        let mine = a.register(WorkerId(1), host(&ours)).unwrap();
        b.register(WorkerId(2), host(&theirs)).unwrap();
        a.connect(&PeerConn {
            worker: WorkerId(2),
            handle: host(&theirs),
            metadata: b.local_metadata().unwrap(),
        })
        .unwrap();
        let id = a
            .recv_mapped(
                &mine,
                &PageSet::new(vec![0, 5]),
                &PageSet::new(vec![3, 6]),
                WorkerId(2),
            )
            .unwrap();
        let started = std::time::Instant::now();
        loop {
            match a.poll(id).unwrap() {
                Completion::Done => break,
                Completion::Pending => assert!(started.elapsed().as_secs() < 10, "never landed"),
                Completion::Failed(why) => panic!("{why}"),
            }
        }
        let page = |at: u64| &ours[(at * STRIDE) as usize..((at + 1) * STRIDE) as usize];
        assert!(page(0).iter().all(|&b| b == 4), "slot 0 holds their page 3");
        assert!(page(5).iter().all(|&b| b == 7), "slot 5 holds their page 6");
        assert!(page(1).iter().all(|&b| b == 0), "slot 1 was not asked for");
        ours.clear();
    }
}
