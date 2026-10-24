//! Named objects in a store a NIXL storage plugin fronts (`OBJ` for S3 and
//! what speaks it, `AZURE_BLOB`, ...): a buffer is written to a name whole,
//! and read back from it given its length. The agent moves each batch
//! between host memory and the store itself; no peer takes part.
#![allow(unsafe_code)]

use std::ffi::{CString, c_void};
use std::ptr;
use std::sync::Mutex;
use std::time::Duration;

use super::ffi::{
    nixl_capi_agent_t, nixl_capi_backend_t, nixl_capi_create_agent, nixl_capi_create_backend,
    nixl_capi_create_opt_args, nixl_capi_create_params, nixl_capi_create_query_resp_list,
    nixl_capi_create_reg_dlist, nixl_capi_create_xfer_dlist, nixl_capi_create_xfer_req,
    nixl_capi_deregister_mem, nixl_capi_destroy_agent, nixl_capi_destroy_backend,
    nixl_capi_destroy_opt_args, nixl_capi_destroy_params, nixl_capi_destroy_query_resp_list,
    nixl_capi_destroy_reg_dlist, nixl_capi_destroy_xfer_dlist, nixl_capi_destroy_xfer_req,
    nixl_capi_get_xfer_status, nixl_capi_mem_type_t, nixl_capi_mem_type_t_NIXL_CAPI_MEM_DRAM,
    nixl_capi_mem_type_t_NIXL_CAPI_MEM_OBJECT, nixl_capi_opt_args_add_backend,
    nixl_capi_opt_args_t, nixl_capi_params_add, nixl_capi_params_t, nixl_capi_post_xfer_req,
    nixl_capi_query_mem, nixl_capi_query_resp_list_has_value, nixl_capi_query_resp_list_t,
    nixl_capi_reg_dlist_add_desc, nixl_capi_reg_dlist_t, nixl_capi_register_mem,
    nixl_capi_release_xfer_req, nixl_capi_xfer_dlist_add_desc, nixl_capi_xfer_dlist_t,
    nixl_capi_xfer_op_t, nixl_capi_xfer_req_t,
};
use super::{OK, READ, WRITE, err};
use crate::error::{Result, TransportError};

const OBJECT: nixl_capi_mem_type_t = nixl_capi_mem_type_t_NIXL_CAPI_MEM_OBJECT;
const DRAM: nixl_capi_mem_type_t = nixl_capi_mem_type_t_NIXL_CAPI_MEM_DRAM;
/// How long a batch is polled between looks.
const POLL: Duration = Duration::from_micros(200);

struct Inner {
    agent: nixl_capi_agent_t,
    backend: nixl_capi_backend_t,
    name: CString,
    /// The next device id an object is registered under: an object is
    /// addressed by it for one batch.
    next: u64,
}

pub struct Objects {
    inner: Mutex<Inner>,
}

// SAFETY: the agent and backend are NIXL heap objects with no tie to the
// thread that made them, only dereferenced by NIXL itself, and only under
// `inner`'s mutex (or `&mut self` in `Drop`), as in `NixlBackend`.
unsafe impl Send for Objects {}
// SAFETY: `&Objects` reaches the agent only through `inner`'s mutex, so NIXL
// never sees two calls at once.
unsafe impl Sync for Objects {}

/// One object of a batch: its name, and the host bytes it is written from
/// or read into.
struct Span {
    name: CString,
    addr: usize,
    len: usize,
}

impl Objects {
    /// An agent named `agent` over the storage plugin `plugin`, set up by
    /// `params` (the plugin's own: for `OBJ`, `bucket`, `endpoint_override`,
    /// `scheme`, `region`; credentials from the environment unless given).
    pub fn open(agent: &str, plugin: &str, params: &[(String, String)]) -> Result<Self> {
        let cstr = |s: &str| {
            CString::new(s).map_err(|_| TransportError::Transfer(format!("`{s}` holds a nul")))
        };
        let (name, plugin) = (cstr(agent)?, cstr(plugin)?);
        let params = params
            .iter()
            .map(|(key, value)| Ok((cstr(key)?, cstr(value)?)))
            .collect::<Result<Vec<_>>>()?;
        let mut handle: nixl_capi_agent_t = ptr::null_mut();
        // SAFETY: `name` is NUL-terminated and outlives the call; `handle` is
        // a live out-parameter.
        let st = unsafe { nixl_capi_create_agent(name.as_ptr(), &mut handle) };
        if st != OK {
            return Err(err("create_agent", st));
        }
        let fail = |what: &str, st| {
            // SAFETY: `handle` is the agent created above, which nothing else
            // holds yet; only an early return calls this, once.
            unsafe { nixl_capi_destroy_agent(handle) };
            err(what, st)
        };
        let mut given: nixl_capi_params_t = ptr::null_mut();
        // SAFETY: `given` is a live out-parameter.
        let st = unsafe { nixl_capi_create_params(&mut given) };
        if st != OK {
            return Err(fail("create_params", st));
        }
        for (key, value) in &params {
            // SAFETY: `given` is live; `key` and `value` are NUL-terminated and
            // outlive the call.
            let st = unsafe { nixl_capi_params_add(given, key.as_ptr(), value.as_ptr()) };
            if st != OK {
                // SAFETY: `given` is live and destroyed once, here.
                unsafe { nixl_capi_destroy_params(given) };
                return Err(fail("params_add", st));
            }
        }
        let mut backend: nixl_capi_backend_t = ptr::null_mut();
        // SAFETY: `handle` and `given` are live; `plugin` is NUL-terminated;
        // `backend` is a live out-parameter.
        let st = unsafe { nixl_capi_create_backend(handle, plugin.as_ptr(), given, &mut backend) };
        // SAFETY: the backend copied what it needs; `given` is destroyed once.
        unsafe { nixl_capi_destroy_params(given) };
        if st != OK {
            return Err(fail("create_backend", st));
        }
        Ok(Self {
            inner: Mutex::new(Inner {
                agent: handle,
                backend,
                name,
                next: 0,
            }),
        })
    }

    /// Writes each buffer to its name, replacing what was there.
    pub fn put(&self, objects: &[(&str, &[u8])]) -> Result<()> {
        let spans = objects
            .iter()
            .map(|(name, bytes)| span(name, bytes.as_ptr() as usize, bytes.len()))
            .collect::<Result<Vec<_>>>()?;
        self.batch(WRITE, &spans)
    }

    /// Reads each name's first `len` bytes. An object shorter than asked
    /// for leaves the rest zero: the length is the caller's to know.
    pub fn get(&self, objects: &[(&str, usize)]) -> Result<Vec<Vec<u8>>> {
        let mut buffers: Vec<Vec<u8>> = objects.iter().map(|(_, len)| vec![0; *len]).collect();
        let spans = objects
            .iter()
            .zip(&mut buffers)
            .map(|((name, len), buffer)| span(name, buffer.as_mut_ptr() as usize, *len))
            .collect::<Result<Vec<_>>>()?;
        self.batch(READ, &spans)?;
        Ok(buffers)
    }

    /// Whether each name is in the store.
    pub fn exists(&self, names: &[&str]) -> Result<Vec<bool>> {
        let names = names
            .iter()
            .map(|name| span(name, 0, 0))
            .collect::<Result<Vec<_>>>()?;
        let g = self.inner.lock().unwrap();
        let descs: Vec<_> = names.iter().map(|s| (0, 0, 0, s.name.as_bytes())).collect();
        let descs = reg_dlist(OBJECT, &descs)?;
        let mut resp: nixl_capi_query_resp_list_t = ptr::null_mut();
        // SAFETY: `resp` is a live out-parameter.
        let st = unsafe { nixl_capi_create_query_resp_list(&mut resp) };
        if st != OK {
            // SAFETY: `descs` was built above and is destroyed once, here.
            unsafe { nixl_capi_destroy_reg_dlist(descs) };
            return Err(err("create_query_resp_list", st));
        }
        // SAFETY: `g.backend` is the live backend `open` created.
        let opts = match unsafe { with_backend(g.backend) } {
            Ok(opts) => opts,
            Err(error) => {
                // SAFETY: both were made above and are destroyed once, here.
                unsafe {
                    nixl_capi_destroy_query_resp_list(resp);
                    nixl_capi_destroy_reg_dlist(descs);
                }
                return Err(error);
            }
        };
        // SAFETY: `g.agent` is live and used under its lock; `descs`, `resp`
        // and `opts` were all made above.
        let st = unsafe { nixl_capi_query_mem(g.agent, descs, resp, opts) };
        // SAFETY: the query is over; each is destroyed once.
        unsafe {
            nixl_capi_destroy_opt_args(opts);
            nixl_capi_destroy_reg_dlist(descs);
        }
        let answer = if st == OK {
            (0..names.len())
                .map(|at| {
                    let mut has = false;
                    // SAFETY: an answered `resp` holds one entry per
                    // descriptor, `names.len()` of them; `has` is a live
                    // out-parameter.
                    match unsafe { nixl_capi_query_resp_list_has_value(resp, at, &mut has) } {
                        OK => Ok(has),
                        st => Err(err("query_resp_list_has_value", st)),
                    }
                })
                .collect()
        } else {
            Err(err("query_mem", st))
        };
        // SAFETY: `resp` is read no more; it is destroyed once.
        unsafe { nixl_capi_destroy_query_resp_list(resp) };
        answer
    }

    /// Registers the batch's host buffers and objects, moves it, and
    /// deregisters both, whatever came of it.
    fn batch(&self, op: nixl_capi_xfer_op_t, spans: &[Span]) -> Result<()> {
        if spans.is_empty() {
            return Ok(());
        }
        let mut g = self.inner.lock().unwrap();
        // Each object is addressed by a device id of its own for the batch.
        let first = g.next;
        g.next += spans.len() as u64;
        let host: Vec<_> = spans.iter().map(|s| (s.addr, s.len, 0, &[][..])).collect();
        let store: Vec<_> = spans
            .iter()
            .zip(first..)
            .map(|(s, device)| (0, s.len, device, s.name.as_bytes()))
            .collect();
        let places = |descs: &[(usize, usize, u64, &[u8])]| {
            descs
                .iter()
                .map(|&(addr, len, device, _)| (addr, len, device))
                .collect::<Vec<_>>()
        };
        // SAFETY: `g.backend` is the live backend `open` created.
        let opts = unsafe { with_backend(g.backend) }?;
        let mut registered = Vec::new();
        let mut moved = Ok(());
        for (kind, descs) in [(DRAM, &host), (OBJECT, &store)] {
            let dl = match reg_dlist(kind, descs) {
                Ok(dl) => dl,
                Err(error) => {
                    moved = Err(error);
                    break;
                }
            };
            // SAFETY: `g.agent` and `opts` are live, used under the lock; the
            // host ranges are the caller's buffers, borrowed for this whole
            // call and deregistered below before it returns.
            let st = unsafe { nixl_capi_register_mem(g.agent, dl, opts) };
            if st != OK {
                // SAFETY: `dl` did not register; it is destroyed once, here.
                unsafe { nixl_capi_destroy_reg_dlist(dl) };
                moved = Err(err("register_mem", st));
                break;
            }
            registered.push(dl);
        }
        if moved.is_ok() {
            moved = match (
                xfer_dlist(DRAM, &places(&host)),
                xfer_dlist(OBJECT, &places(&store)),
            ) {
                // SAFETY: both lists were just built and are handed over;
                // every range they name was registered above, and the host
                // buffers outlive this call.
                (Ok(local), Ok(remote)) => unsafe { transfer(&g, op, local, remote) },
                (local, remote) => {
                    for dl in [&local, &remote].into_iter().flatten() {
                        // SAFETY: a list built above and not handed on; it is
                        // destroyed once, here.
                        unsafe { nixl_capi_destroy_xfer_dlist(*dl) };
                    }
                    local.and(remote).map(|_| ())
                }
            };
        }
        for dl in registered {
            // SAFETY: `dl` was registered above with `opts`, and the transfer
            // is over (`transfer` waits it out); it is deregistered and
            // destroyed once.
            unsafe {
                nixl_capi_deregister_mem(g.agent, dl, opts);
                nixl_capi_destroy_reg_dlist(dl);
            }
        }
        // SAFETY: `opts` is used no more; it is destroyed once.
        unsafe { nixl_capi_destroy_opt_args(opts) };
        moved
    }
}

/// Moves the batch between `host` and `store` and waits until it is over.
///
/// # Safety
///
/// `g.agent` must be live and used by no one else for the call. `host` and
/// `store` must be live transfer lists, which this consumes; every range they
/// name must be registered with `g.agent`, and the host memory must stay
/// alive until this returns.
unsafe fn transfer(
    g: &Inner,
    op: nixl_capi_xfer_op_t,
    host: nixl_capi_xfer_dlist_t,
    store: nixl_capi_xfer_dlist_t,
) -> Result<()> {
    let mut req: nixl_capi_xfer_req_t = ptr::null_mut();
    // SAFETY: the caller hands over a live agent and two live lists; `g.name`
    // is NUL-terminated (the agent's own: the store is local); `req` is a live
    // out-parameter.
    let st = unsafe {
        nixl_capi_create_xfer_req(
            g.agent,
            op,
            host,
            store,
            g.name.as_ptr(),
            &mut req,
            ptr::null_mut(),
        )
    };
    // SAFETY: the request copied the lists, which are ours to destroy once.
    unsafe {
        nixl_capi_destroy_xfer_dlist(host);
        nixl_capi_destroy_xfer_dlist(store);
    }
    if st != OK {
        return Err(err("create_xfer_req", st));
    }
    // SAFETY: `req` was just created on `g.agent`; no notification is sent.
    let mut st = unsafe { nixl_capi_post_xfer_req(g.agent, req, ptr::null_mut()) };
    while st == super::INPROG {
        std::thread::sleep(POLL);
        // SAFETY: `req` is live and posted on `g.agent`.
        st = unsafe { nixl_capi_get_xfer_status(g.agent, req) };
    }
    // SAFETY: `req`'s status is final, so nothing is in flight; it is
    // released and destroyed once.
    unsafe {
        nixl_capi_release_xfer_req(g.agent, req);
        nixl_capi_destroy_xfer_req(req);
    }
    if st != OK {
        return Err(err("moving objects", st));
    }
    Ok(())
}

impl Drop for Objects {
    fn drop(&mut self) {
        let g = self.inner.get_mut().unwrap();
        // SAFETY: the backend and agent `open` created, each destroyed once,
        // the backend first; `&mut self` rules out any call in flight.
        unsafe {
            nixl_capi_destroy_backend(g.backend);
            nixl_capi_destroy_agent(g.agent);
        }
    }
}

fn span(name: &str, addr: usize, len: usize) -> Result<Span> {
    let name = CString::new(name)
        .map_err(|_| TransportError::Transfer(format!("object name `{name}` holds a nul")))?;
    Ok(Span { name, addr, len })
}

/// Transfer options naming `backend`; the caller destroys them.
///
/// # Safety
///
/// `backend` must be a live NIXL backend handle.
unsafe fn with_backend(backend: nixl_capi_backend_t) -> Result<nixl_capi_opt_args_t> {
    let mut opts: nixl_capi_opt_args_t = ptr::null_mut();
    // SAFETY: `opts` is a live out-parameter.
    let st = unsafe { nixl_capi_create_opt_args(&mut opts) };
    if st != OK {
        return Err(err("create_opt_args", st));
    }
    // SAFETY: `opts` was just created; the caller vouches for `backend`.
    let st = unsafe { nixl_capi_opt_args_add_backend(opts, backend) };
    if st != OK {
        // SAFETY: `opts` is live and destroyed once, here.
        unsafe { nixl_capi_destroy_opt_args(opts) };
        return Err(err("opt_args_add_backend", st));
    }
    Ok(opts)
}

/// A registration list of `kind` holding `descs` as `(addr, len, device,
/// metadata)`. The list only records them: nothing is read until it is
/// registered, which is the caller's to answer for. The caller destroys it.
fn reg_dlist(
    kind: nixl_capi_mem_type_t,
    descs: &[(usize, usize, u64, &[u8])],
) -> Result<nixl_capi_reg_dlist_t> {
    let mut dl: nixl_capi_reg_dlist_t = ptr::null_mut();
    // SAFETY: `dl` is a live out-parameter.
    let st = unsafe { nixl_capi_create_reg_dlist(kind, &mut dl) };
    if st != OK {
        return Err(err("create_reg_dlist", st));
    }
    for &(addr, len, device, meta) in descs {
        // SAFETY: `dl` is live; `meta` is `meta.len()` readable bytes, which
        // NIXL copies.
        let st = unsafe {
            nixl_capi_reg_dlist_add_desc(
                dl,
                addr,
                len,
                device,
                meta.as_ptr() as *const c_void,
                meta.len(),
            )
        };
        if st != OK {
            // SAFETY: `dl` is not returned; it is destroyed once, here.
            unsafe { nixl_capi_destroy_reg_dlist(dl) };
            return Err(err("reg_dlist_add_desc", st));
        }
    }
    Ok(dl)
}

/// A transfer list of `kind` holding `descs` as `(addr, len, device)`. The
/// list only records them: NIXL checks them against registered memory when
/// a request is built from it. The caller destroys it.
fn xfer_dlist(
    kind: nixl_capi_mem_type_t,
    descs: &[(usize, usize, u64)],
) -> Result<nixl_capi_xfer_dlist_t> {
    let mut dl: nixl_capi_xfer_dlist_t = ptr::null_mut();
    // SAFETY: `dl` is a live out-parameter.
    let st = unsafe { nixl_capi_create_xfer_dlist(kind, &mut dl) };
    if st != OK {
        return Err(err("create_xfer_dlist", st));
    }
    for &(addr, len, device) in descs {
        // SAFETY: `dl` is live; a descriptor is plain data until a request is
        // built from the list.
        let st = unsafe { nixl_capi_xfer_dlist_add_desc(dl, addr, len, device) };
        if st != OK {
            // SAFETY: `dl` is not returned; it is destroyed once, here.
            unsafe { nixl_capi_destroy_xfer_dlist(dl) };
            return Err(err("xfer_dlist_add_desc", st));
        }
    }
    Ok(dl)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An S3 endpoint to write to (`PIE_TEST_S3`, the URL of one that
    /// takes any credentials, holding the bucket `pie`), else none.
    fn store() -> Option<Objects> {
        let endpoint = std::env::var("PIE_TEST_S3").ok()?;
        let params = [
            ("bucket", "pie"),
            ("endpoint_override", endpoint.as_str()),
            ("scheme", "http"),
            ("region", "us-east-1"),
            ("access_key", "test"),
            ("secret_key", "test"),
        ]
        .map(|(k, v)| (k.to_string(), v.to_string()));
        Some(Objects::open("objects-test", "OBJ", &params).unwrap())
    }

    #[test]
    fn objects_come_back_as_they_were_written_and_are_known_by_name() {
        let Some(store) = store() else {
            eprintln!("no PIE_TEST_S3: skipping");
            return;
        };
        let name = |n: u32| format!("objects-test/{}/{n}", std::process::id());
        let (a, b) = (name(1), name(2));
        let (one, two): (Vec<u8>, Vec<u8>) = ((0..5000).map(|i| i as u8).collect(), vec![9; 300]);
        store.put(&[(&a, &one), (&b, &two)]).unwrap();
        let back = store.get(&[(&b, 300), (&a, 5000)]).unwrap();
        assert_eq!(back, [two, one]);
        assert_eq!(
            store.exists(&[&a, &name(3), &b]).unwrap(),
            [true, false, true]
        );
    }
}
