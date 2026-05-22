//! `pie_driver_dummy` — Rust dummy driver, C ABI.
//!
//! Exports the same shape as `driver/portable` and `driver/cuda`:
//!
//! ```c
//! int pie_driver_dummy_run(int argc, char** argv,
//!                          int install_signal_handlers,
//!                          ready_cb_t ready_cb, void* ready_ctx);
//! void pie_driver_dummy_request_stop(void);
//! ```
//!
//! `argv[1]` is `--config <path>`; everything else is ignored.

mod config;
mod handler;

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};

use crate::handler::Handler;
use pie_bridge::ArchivedRequestPayload;
use pie_bridge::SCHEMA_HASH;
use pie_bridge::ipc::ShmemServer;
use pie_bridge::wire::parse_request;

pub type ReadyCb = unsafe extern "C" fn(caps_json: *const c_char, ctx: *mut c_void);

/// Process-global handles to every running dummy shmem server. The Rust
/// server can embed multiple same-flavor DP replicas in one process, so
/// `request_stop` must stop all live instances.
static SERVERS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_driver_dummy_run(
    argc: c_int,
    argv: *mut *mut c_char,
    _install_signal_handlers: c_int,
    ready_cb: ReadyCb,
    ready_ctx: *mut c_void,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_impl(argc, argv, ready_cb, ready_ctx)
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            eprintln!("[pie-driver-dummy] fatal: {e:#}");
            -1
        }
        Err(_) => {
            eprintln!("[pie-driver-dummy] fatal: panic");
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_driver_dummy_request_stop() {
    let servers = SERVERS.lock().map(|g| g.clone()).unwrap_or_default();
    for addr in servers {
        let p = addr as *mut ShmemServer;
        if !p.is_null() {
            // SAFETY: each registered ShmemServer is kept live until its
            // drop guard removes the pointer.
            unsafe { (*p).stop() };
        }
    }
}

fn run_impl(
    argc: c_int,
    argv: *mut *mut c_char,
    ready_cb: ReadyCb,
    ready_ctx: *mut c_void,
) -> Result<()> {
    let config_path = parse_argv_for_config(argc, argv)?;
    let cfg = config::load(&config_path)
        .with_context(|| format!("loading dummy startup TOML {config_path:?}"))?;

    eprintln!(
        "[pie-driver-dummy] config loaded\n\
         \x20 shmem.name      = {}\n\
         \x20 shmem.num_slots = {}\n\
         \x20 vocab_size      = {}\n\
         \x20 arch_name       = {}",
        cfg.shmem.name, cfg.shmem.num_slots, cfg.dummy.vocab_size, cfg.dummy.arch_name,
    );

    let server = ShmemServer::create(
        &cfg.shmem.name,
        cfg.shmem.num_slots,
        cfg.shmem.req_buf,
        cfg.shmem.resp_buf,
        cfg.shmem.spin_budget_us,
        SCHEMA_HASH,
    )?;

    let server_ptr = &server as *const _ as usize;
    SERVERS
        .lock()
        .map_err(|_| anyhow!("dummy server registry poisoned"))?
        .push(server_ptr);

    struct ClearOnDrop(usize);
    impl Drop for ClearOnDrop {
        fn drop(&mut self) {
            if let Ok(mut servers) = SERVERS.lock() {
                servers.retain(|addr| *addr != self.0);
            }
        }
    }
    let _clear = ClearOnDrop(server_ptr);

    let total_pages = cfg.dummy.derived_total_pages();

    // Capability handshake.
    let caps = serde_json::json!({
        "total_pages":      total_pages,
        "kv_page_size":     config::KV_PAGE_SIZE,
        "swap_pool_size":   0u32,
        "max_forward_tokens": cfg.dummy.max_forward_tokens,
        "max_forward_requests": cfg.dummy.max_forward_requests,
        "max_page_refs": total_pages,
        "max_logit_rows": u32::MAX,
        "max_prob_rows": u32::MAX,
        "max_custom_mask_bytes": u32::MAX,
        "max_sampler_rows": u32::MAX,
        "max_logprob_labels": u32::MAX,
        "arch_name":        cfg.dummy.arch_name,
        "vocab_size":       cfg.dummy.vocab_size,
        "max_model_len":    cfg.dummy.max_model_len,
        "activation_dtype": cfg.dummy.activation_dtype,
        "snapshot_dir":     cfg.dummy.snapshot_dir,
        "shmem_name":       cfg.shmem.name,
    })
    .to_string();
    let caps_c = CString::new(caps).map_err(|e| anyhow!("caps JSON contains NUL: {e}"))?;
    // SAFETY: ready_cb is a non-null fn pointer per the contract; caps_c
    // outlives the call.
    unsafe { ready_cb(caps_c.as_ptr(), ready_ctx) };

    eprintln!(
        "[pie-driver-dummy] serving on {} ({} slots, req_buf={}, resp_buf={})",
        server.name(),
        server.num_slots(),
        server.req_buf_size(),
        server.resp_buf_size(),
    );

    let mut handler = Handler::new(cfg.dummy.random_seed, cfg.dummy.vocab_size);
    let mut handled: u64 = 0;

    while !server.stopped() {
        let Some(lease) = server.poll() else {
            std::thread::sleep(Duration::from_micros(200));
            continue;
        };
        handled += 1;
        match parse_request(lease.payload()) {
            Ok(frame) => {
                let driver_id: u32 = frame.driver_id.into();
                match &frame.payload {
                    ArchivedRequestPayload::Forward(fr) => {
                        match handler.handle_forward(driver_id, fr) {
                            Ok(resp) => {
                                if let Err(e) = lease.commit(&resp) {
                                    eprintln!("[pie-driver-dummy] commit failed: {e:#}");
                                }
                            }
                            Err(e) => {
                                eprintln!("[pie-driver-dummy] handle_forward failed: {e:#}");
                                let _ = lease.commit_status(-1);
                            }
                        }
                    }
                    ArchivedRequestPayload::Health => {
                        let _ = lease.commit_status(0);
                    }
                    _ => {
                        eprintln!("[pie-driver-dummy] unsupported payload variant");
                        let _ = lease.commit_status(-1);
                    }
                }
            }
            Err(e) => {
                eprintln!("[pie-driver-dummy] parse failed: {e}");
                drop(lease);
            }
        }
    }

    eprintln!("[pie-driver-dummy] shutting down (handled {handled} requests)");
    Ok(())
}

fn parse_argv_for_config(argc: c_int, argv: *mut *mut c_char) -> Result<PathBuf> {
    if argc < 1 || argv.is_null() {
        return Err(anyhow!("invalid argv"));
    }
    let n = argc as usize;
    let args: Vec<&str> = (0..n)
        .map(|i| {
            let p = unsafe { *argv.add(i) };
            if p.is_null() {
                ""
            } else {
                unsafe { CStr::from_ptr(p) }.to_str().unwrap_or("")
            }
        })
        .collect();

    let mut iter = args.iter().skip(1);
    while let Some(arg) = iter.next() {
        match *arg {
            "-c" | "--config" => {
                let val = iter
                    .next()
                    .ok_or_else(|| anyhow!("--config needs a value"))?;
                return Ok(PathBuf::from(val));
            }
            _ if arg.starts_with("--config=") => {
                return Ok(PathBuf::from(&arg["--config=".len()..]));
            }
            _ => {}
        }
    }
    Err(anyhow!("missing --config <path>; got argv = {args:?}"))
}
