#![allow(unsafe_code)]
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
use ffi::{
    nixl_capi_agent_t, nixl_capi_backend_t, nixl_capi_create_agent, nixl_capi_create_backend,
    nixl_capi_create_reg_dlist, nixl_capi_create_xfer_dlist, nixl_capi_create_xfer_req,
    nixl_capi_destroy_agent, nixl_capi_destroy_backend, nixl_capi_destroy_reg_dlist,
    nixl_capi_destroy_xfer_dlist, nixl_capi_destroy_xfer_req, nixl_capi_get_local_md,
    nixl_capi_get_plugin_params, nixl_capi_get_xfer_status, nixl_capi_load_remote_md,
    nixl_capi_mem_list_t, nixl_capi_mem_type_t, nixl_capi_mem_type_t_NIXL_CAPI_MEM_DRAM,
    nixl_capi_mem_type_t_NIXL_CAPI_MEM_VRAM, nixl_capi_params_t, nixl_capi_post_xfer_req,
    nixl_capi_reg_dlist_add_desc, nixl_capi_reg_dlist_t, nixl_capi_register_mem,
    nixl_capi_release_xfer_req, nixl_capi_status_t, nixl_capi_status_t_NIXL_CAPI_IN_PROG,
    nixl_capi_status_t_NIXL_CAPI_SUCCESS, nixl_capi_xfer_dlist_add_desc, nixl_capi_xfer_dlist_t,
    nixl_capi_xfer_op_t, nixl_capi_xfer_op_t_NIXL_CAPI_XFER_OP_READ,
    nixl_capi_xfer_op_t_NIXL_CAPI_XFER_OP_WRITE, nixl_capi_xfer_req_t,
};

const OK: nixl_capi_status_t = nixl_capi_status_t_NIXL_CAPI_SUCCESS;
const INPROG: nixl_capi_status_t = nixl_capi_status_t_NIXL_CAPI_IN_PROG;
const WRITE: nixl_capi_xfer_op_t = nixl_capi_xfer_op_t_NIXL_CAPI_XFER_OP_WRITE;
const READ: nixl_capi_xfer_op_t = nixl_capi_xfer_op_t_NIXL_CAPI_XFER_OP_READ;

fn err(ctx: &str, st: nixl_capi_status_t) -> TransportError {
    TransportError::Transfer(format!("nixl {ctx}: status {st}"))
}

/// The NIXL segment a domain's memory registers as, and its device id: host
/// memory as DRAM, a CUDA or ROCm device's as VRAM. The rest (Metal, Vulkan,
/// wgpu, XLA, Cerebras memory, a slot file) NIXL cannot reach.
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
        | MemoryDomain::CerebrasDevice(_)
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

// SAFETY: the raw handles are NIXL heap objects with no tie to the thread
// that made them, only dereferenced by NIXL itself, and only under `inner`'s
// mutex (or `&mut self` in `Drop`); moving the backend moves sole ownership.
unsafe impl Send for NixlBackend {}
// SAFETY: `&NixlBackend` reaches the agent only through `inner`'s mutex, so
// NIXL (which is not thread-safe) never sees two calls at once.
unsafe impl Sync for NixlBackend {}

impl NixlBackend {
    pub fn new(agent_name: &str) -> Result<Self> {
        let name = CString::new(agent_name)
            .map_err(|_| TransportError::Transfer("agent name contains a nul byte".into()))?;
        let ucx = CString::new("UCX").expect("UCX literal");
        let mut agent: nixl_capi_agent_t = ptr::null_mut();
        // SAFETY: `name` is NUL-terminated and outlives the call; `agent` is a
        // live out-parameter.
        let st = unsafe { nixl_capi_create_agent(name.as_ptr(), &mut agent) };
        if st != OK {
            return Err(err("create_agent", st));
        }
        let mut mems: nixl_capi_mem_list_t = ptr::null_mut();
        let mut params: nixl_capi_params_t = ptr::null_mut();
        // SAFETY: `agent` was just created; `ucx` is NUL-terminated; `mems`
        // and `params` are live out-parameters.
        let st =
            unsafe { nixl_capi_get_plugin_params(agent, ucx.as_ptr(), &mut mems, &mut params) };
        if st != OK {
            return Err(err("get_plugin_params(UCX)", st));
        }
        let mut backend: nixl_capi_backend_t = ptr::null_mut();
        // SAFETY: `agent` is live and `params` is the list NIXL just handed
        // back for this plugin; `backend` is a live out-parameter.
        let st = unsafe { nixl_capi_create_backend(agent, ucx.as_ptr(), params, &mut backend) };
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

        if local_pages.len() != remote_pages.len() {
            return Err(TransportError::LayoutMismatch);
        }
        let (locals, remotes) = runs(local_pages, remote_pages);
        let local_dl = build_xfer_dlist(local.handle(), &locals)?;
        let remote_dl = match build_xfer_dlist(&remote_handle, &remotes) {
            Ok(dl) => dl,
            Err(e) => {
                // SAFETY: `local_dl` was built above and is destroyed once.
                unsafe { nixl_capi_destroy_xfer_dlist(local_dl) };
                return Err(e);
            }
        };

        let mut req: nixl_capi_xfer_req_t = ptr::null_mut();
        // SAFETY: `g.agent` is live while `self` is and is used under its
        // lock; both lists were just built; `rname` is the NUL-terminated name
        // `connect` copied; `req` is a live out-parameter. NIXL checks each
        // descriptor against the memory registered on its side.
        let st = unsafe {
            nixl_capi_create_xfer_req(
                g.agent,
                op,
                local_dl,
                remote_dl,
                rname.as_ptr(),
                &mut req,
                ptr::null_mut(),
            )
        };
        // SAFETY: the request copied the lists; each is destroyed once, here.
        unsafe {
            nixl_capi_destroy_xfer_dlist(local_dl);
            nixl_capi_destroy_xfer_dlist(remote_dl);
        }
        if st != OK {
            return Err(err("create_xfer_req", st));
        }

        // SAFETY: `req` was just created on `g.agent`; no notification is sent.
        let st = unsafe { nixl_capi_post_xfer_req(g.agent, req, ptr::null_mut()) };
        if st != OK && st != INPROG {
            // SAFETY: `req` failed to post, so nothing is in flight; it is
            // released and destroyed once, and never recorded.
            unsafe {
                nixl_capi_release_xfer_req(g.agent, req);
                nixl_capi_destroy_xfer_req(req);
            }
            return Err(err("post_xfer_req", st));
        }

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

/// Runs of consecutive pages, each as `(first page, pages)`.
type Runs = Vec<(u32, u32)>;

/// The pages paired index by index, cut where either side stops being
/// consecutive: each side's runs, one per run, so a run moves as one
/// descriptor per region rather than one per page.
fn runs(local: &PageSet, remote: &PageSet) -> (Runs, Runs) {
    let (mut locals, mut remotes) = (Runs::new(), Runs::new());
    for (&l, &r) in local.pages.iter().zip(&remote.pages) {
        match (locals.last_mut(), remotes.last_mut()) {
            (Some((lf, ln)), Some((rf, rn)))
                if lf.checked_add(*ln) == Some(l) && rf.checked_add(*rn) == Some(r) =>
            {
                *ln += 1;
                *rn += 1;
            }
            _ => {
                locals.push((l, 1));
                remotes.push((r, 1));
            }
        }
    }
    (locals, remotes)
}

/// The transfer list naming `runs` of every region of `handle`, each run
/// checked to lie inside its region. The list only records addresses: no
/// memory is touched until a request built from it is posted, and NIXL
/// checks each descriptor against registered memory then, so this asks
/// nothing of its caller. The caller owns the list and destroys it.
fn build_xfer_dlist(handle: &KvHandle, runs: &[(u32, u32)]) -> Result<nixl_capi_xfer_dlist_t> {
    let first = handle
        .regions
        .first()
        .ok_or(TransportError::Unsupported("handle has no KV region"))?;
    let (kind, _) = segment(first.domain)?;
    let mut dl: nixl_capi_xfer_dlist_t = ptr::null_mut();
    // SAFETY: `dl` is a live out-parameter.
    let st = unsafe { nixl_capi_create_xfer_dlist(kind, &mut dl) };
    if st != OK {
        return Err(err("create_xfer_dlist", st));
    }
    let discard = |dl| {
        // SAFETY: `dl` was created above, is not returned, and is destroyed
        // once, on the way out.
        unsafe { nixl_capi_destroy_xfer_dlist(dl) };
    };
    for region in &handle.regions {
        let device = match segment(region.domain) {
            Ok((other, device)) if other == kind && region.page_stride != 0 => device,
            _ => {
                discard(dl);
                return Err(TransportError::LayoutMismatch);
            }
        };
        for &(page, count) in runs {
            // Checked: a peer's page numbers must not wrap past the bound.
            let inside = u64::from(page)
                .checked_mul(region.page_stride)
                .zip(u64::from(count).checked_mul(region.page_stride))
                .filter(|&(offset, len)| {
                    offset.checked_add(len).is_some_and(|end| end <= region.len)
                });
            let Some((offset, len)) = inside else {
                discard(dl);
                return Err(TransportError::PageOutOfBounds {
                    page: page + count - 1,
                });
            };
            // SAFETY: `dl` is the live list created above; a descriptor is
            // plain data, read by NIXL only when a request is built from it.
            let st = unsafe {
                nixl_capi_xfer_dlist_add_desc(
                    dl,
                    (region.base + offset) as usize,
                    len as usize,
                    device,
                )
            };
            if st != OK {
                discard(dl);
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

    unsafe fn register(&self, owner: WorkerId, handle: KvHandle) -> Result<RegisteredHandle> {
        let mut g = self.inner.lock().unwrap();
        for region in &handle.regions {
            let (kind, device) = segment(region.domain)?;
            let mut dl: nixl_capi_reg_dlist_t = ptr::null_mut();
            // SAFETY: `dl` is a live out-parameter.
            let st = unsafe { nixl_capi_create_reg_dlist(kind, &mut dl) };
            if st != OK {
                return Err(err("create_reg_dlist", st));
            }
            // SAFETY: `dl` was just created; the descriptor carries no
            // metadata (null, 0), and its range is plain data here.
            let st = unsafe {
                nixl_capi_reg_dlist_add_desc(
                    dl,
                    region.base as usize,
                    region.len as usize,
                    device,
                    ptr::null(),
                    0,
                )
            };
            if st != OK {
                // SAFETY: `dl` is live and destroyed once, here.
                unsafe { nixl_capi_destroy_reg_dlist(dl) };
                return Err(err("reg_dlist_add_desc", st));
            }
            // SAFETY: `g.agent` is live and used under its lock, and `dl` was
            // just built. The region stays allocated and unraced for the
            // agent's life, as `register`'s caller promised.
            let st = unsafe { nixl_capi_register_mem(g.agent, dl, ptr::null_mut()) };
            // SAFETY: registering copied the list; `dl` is destroyed once.
            unsafe { nixl_capi_destroy_reg_dlist(dl) };
            if st != OK {
                return Err(err("register_mem", st));
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
            // SAFETY: an earlier poll released `request` and its destroy
            // failed, so it is live, finished, and still ours to destroy.
            let status = unsafe { nixl_capi_destroy_xfer_req(request.handle) };
            if status != OK {
                return Err(err("destroy_xfer_req", status));
            }
            g.reqs.remove(&id.0);
            return Ok(Completion::Done);
        }
        // SAFETY: `request.handle` is a live, unreleased request posted on
        // `g.agent`, used under its lock.
        let st = unsafe { nixl_capi_get_xfer_status(g.agent, request.handle) };
        match st {
            s if s == OK => {
                // SAFETY: the request is live and unreleased, and no longer in
                // flight (its status is final); it is marked released below.
                let status = unsafe { nixl_capi_release_xfer_req(g.agent, request.handle) };
                if status != OK {
                    return Err(err("release_xfer_req", status));
                }
                g.reqs.get_mut(&id.0).expect("request remains").released = true;
                // SAFETY: released just above; destroyed once, as its entry is
                // removed on success.
                let status = unsafe { nixl_capi_destroy_xfer_req(request.handle) };
                if status != OK {
                    return Err(err("destroy_xfer_req", status));
                }
                g.reqs.remove(&id.0);
                Ok(Completion::Done)
            }
            s if s == INPROG => Ok(Completion::Pending),
            other => {
                // SAFETY: the request is live and unreleased, and no longer in
                // flight (its status is final); it is marked released below.
                let status = unsafe { nixl_capi_release_xfer_req(g.agent, request.handle) };
                if status != OK {
                    return Err(err("release_xfer_req", status));
                }
                g.reqs.get_mut(&id.0).expect("request remains").released = true;
                // SAFETY: released just above; destroyed once, as its entry is
                // removed on success.
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
        // SAFETY: `g.agent` is live and used under its lock; the metadata is
        // `peer.metadata.len()` readable bytes; `rname` is a live
        // out-parameter.
        let st = unsafe {
            nixl_capi_load_remote_md(
                g.agent,
                peer.metadata.as_ptr() as *const c_void,
                peer.metadata.len(),
                &mut rname,
            )
        };
        if st != OK {
            return Err(err("load_remote_md", st));
        }
        // SAFETY: on success NIXL hands back the peer's agent name as a
        // NUL-terminated string, which is copied here.
        let agent = unsafe { CStr::from_ptr(rname) }.to_owned();
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
        let mut data: *mut c_void = ptr::null_mut();
        let mut len: usize = 0;
        // SAFETY: `g.agent` is live and used under its lock; `data` and `len`
        // are live out-parameters.
        let st = unsafe { nixl_capi_get_local_md(g.agent, &mut data, &mut len) };
        if st != OK {
            return Err(err("get_local_md", st));
        }
        // SAFETY: on success `data` is a buffer of `len` initialised bytes
        // that NIXL allocated for this call; it is copied out here.
        Ok(unsafe { std::slice::from_raw_parts(data as *const u8, len) }.to_vec())
    }
}

impl Drop for NixlBackend {
    fn drop(&mut self) {
        let g = self.inner.get_mut().unwrap();
        for (_, request) in g.reqs.drain() {
            if !request.released {
                // SAFETY: a live request of this agent not yet released;
                // releasing it stops whatever is still in flight.
                let _ = unsafe { nixl_capi_release_xfer_req(g.agent, request.handle) };
            }
            // SAFETY: released, now or by an earlier poll; drained, so
            // destroyed once.
            let _ = unsafe { nixl_capi_destroy_xfer_req(request.handle) };
        }
        if !g.backend.is_null() {
            // SAFETY: the backend `new` created, destroyed once, after every
            // request that used it.
            unsafe { nixl_capi_destroy_backend(g.backend) };
        }
        if !g.agent.is_null() {
            // SAFETY: the agent `new` created, destroyed once and last.
            unsafe { nixl_capi_destroy_agent(g.agent) };
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

    /// A host region at `base`, `len` bytes: NIXL reads or writes it.
    fn host(base: *mut u8, len: usize) -> KvHandle {
        KvHandle {
            regions: vec![KvRegion {
                base: base as u64,
                len: len as u64,
                page_stride: STRIDE,
                domain: MemoryDomain::HostPinned,
            }],
            layout: layout(),
        }
    }

    #[test]
    fn a_read_brings_the_peers_pages_into_the_named_slots() {
        let mut theirs: Vec<u8> = (0..PAGES * STRIDE)
            .map(|i| (i / STRIDE) as u8 + 1)
            .collect();
        let mut ours = vec![0u8; (PAGES * STRIDE) as usize];
        let (a, b) = (
            NixlBackend::new("home").unwrap(),
            NixlBackend::new("peer").unwrap(),
        );
        let ours_at = (ours.as_mut_ptr(), ours.len());
        let theirs_at = (theirs.as_mut_ptr(), theirs.len());
        // SAFETY: `ours` and `theirs` are declared before `a` and `b`, so
        // they outlive both agents, and neither is touched until the read
        // below has completed.
        let mine = unsafe { a.register(WorkerId(1), host(ours_at.0, ours_at.1)) }.unwrap();
        // SAFETY: as above.
        unsafe { b.register(WorkerId(2), host(theirs_at.0, theirs_at.1)) }.unwrap();
        a.connect(&PeerConn {
            worker: WorkerId(2),
            handle: host(theirs_at.0, theirs_at.1),
            metadata: b.local_metadata().unwrap(),
        })
        .unwrap();
        let id = a
            .recv_mapped(
                &mine,
                &PageSet::new(vec![0, 1, 5]),
                &PageSet::new(vec![2, 3, 6]),
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
        assert!(page(0).iter().all(|&b| b == 3), "slot 0 holds their page 2");
        assert!(page(1).iter().all(|&b| b == 4), "slot 1 holds their page 3");
        assert!(page(5).iter().all(|&b| b == 7), "slot 5 holds their page 6");
        assert!(page(2).iter().all(|&b| b == 0), "slot 2 was not asked for");
        // The peer's region stays alive until the read is checked.
        drop(theirs);
    }

    #[test]
    fn pages_consecutive_on_both_sides_move_as_one_run() {
        let (ours, theirs) = runs(
            &PageSet::new(vec![0, 1, 2, 5, 6, 9]),
            &PageSet::new(vec![4, 5, 7, 8, 9, 3]),
        );
        assert_eq!(ours, vec![(0, 2), (2, 1), (5, 2), (9, 1)]);
        assert_eq!(theirs, vec![(4, 2), (7, 1), (8, 2), (3, 1)]);
    }
}
