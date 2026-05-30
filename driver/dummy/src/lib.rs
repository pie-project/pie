//! `pie_driver_dummy` — Rust dummy driver, C ABI.
//!
//! Exports the same shape as `driver/portable` and `driver/cuda`:
//!
//! ```c
//! int pie_driver_dummy_run(int argc, char** argv,
//!                          int install_signal_handlers,
//!                          ready_cb_t ready_cb, void* ready_ctx,
//!                          fatal_cb_t fatal_cb, void* fatal_ctx);
//! void pie_driver_dummy_request_stop(void);
//! ```
//!
//! `argv[1]` is `--config <path>`; everything else is ignored. The
//! driver reads the startup TOML, opens the shmem region, emits caps
//! via `ready_cb`, and serves random-token responses until `request_stop`
//! is called. See README.md for the config schema and limitations.

mod config;
mod handler;
mod schema;
mod shmem;

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow};

use crate::handler::Handler;
use crate::shmem::{METHOD_TAG_FIRE_BATCH, ShmemServer};

pub type ReadyCb = unsafe extern "C" fn(caps_json: *const c_char, ctx: *mut c_void);

/// Fatal callback type. Invoked at most once with the failure reason
/// (NUL-terminated) just before a nonzero return. Non-nullable — pass a
/// no-op callback for the legacy stderr-only behavior. Mirrors the C++
/// drivers' `pie_driver_*_fatal_cb`.
pub type FatalCb = unsafe extern "C" fn(reason: *const c_char, ctx: *mut c_void);

/// Process-global handles to every running dummy shmem server. The Rust
/// server can embed multiple same-flavor DP replicas in one process, so
/// `request_stop` must stop all live instances.
static SERVERS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Route a fatal reason through `fatal_cb` if one was provided. Mirrors the
/// `if (fatal_cb) fatal_cb(...)` pattern in the C++ entries; the stderr line
/// is printed separately by the caller for back-compat.
///
/// # Safety
/// `fatal_cb` must be a valid function pointer and `fatal_ctx` the context it
/// expects; the callback receives a transient NUL-terminated string.
unsafe fn report_fatal(reason: &str, fatal_cb: FatalCb, fatal_ctx: *mut c_void) {
    // A NUL inside the reason would truncate the C string; fall back to a fixed
    // message so the callback still fires with something meaningful.
    match CString::new(reason) {
        Ok(c) => unsafe { fatal_cb(c.as_ptr(), fatal_ctx) },
        Err(_) => {
            let fallback = CString::new("fatal reason contained a NUL byte").unwrap();
            unsafe { fatal_cb(fallback.as_ptr(), fatal_ctx) };
        }
    }
}

/// Library entry point. Mirrors the C++ `pie_driver_portable_run`.
///
/// # Safety
/// `argv` must point to `argc` C strings; `ready_cb` must be a valid
/// function pointer or null. The string handed to `ready_cb` is owned
/// by this function and freed before return. `fatal_cb` is invoked once with
/// `fatal_ctx` and the failure reason just before a nonzero return.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_driver_dummy_run(
    argc: c_int,
    argv: *mut *mut c_char,
    _install_signal_handlers: c_int,
    ready_cb: ReadyCb,
    ready_ctx: *mut c_void,
    fatal_cb: FatalCb,
    fatal_ctx: *mut c_void,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_impl(argc, argv, ready_cb, ready_ctx)
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            let reason = format!("{e:#}");
            eprintln!("[pie-driver-dummy] fatal: {reason}");
            unsafe { report_fatal(&reason, fatal_cb, fatal_ctx) };
            -1
        }
        Err(_) => {
            eprintln!("[pie-driver-dummy] fatal: panic");
            unsafe { report_fatal("panic", fatal_cb, fatal_ctx) };
            -1
        }
    }
}

/// Stop the running serve loop. Idempotent. Safe to call from any thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_driver_dummy_request_stop() {
    let servers = SERVERS.lock().map(|g| g.clone()).unwrap_or_default();
    for addr in servers {
        let p = addr as *mut ShmemServer;
        if !p.is_null() {
            // SAFETY: `pie_driver_dummy_run` keeps each registered
            // ShmemServer live until its drop guard removes the pointer.
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

    // Open the shmem region (we are the server — runtime attaches as client).
    let server = ShmemServer::create(
        &cfg.shmem.name,
        cfg.shmem.num_slots,
        cfg.shmem.req_buf,
        cfg.shmem.resp_buf,
        cfg.shmem.spin_us,
    )?;

    // Register the global pointer so `pie_driver_dummy_request_stop`
    // can reach this instance. Multiple concurrent instances are valid
    // for DP > 1.
    let server_ptr = &server as *const _ as usize;
    SERVERS
        .lock()
        .map_err(|_| anyhow!("dummy server registry poisoned"))?
        .push(server_ptr);
    // RAII clear on early-return / panic so we never dangle a stale pointer.
    struct ClearOnDrop(usize);
    impl Drop for ClearOnDrop {
        fn drop(&mut self) {
            if let Ok(mut servers) = SERVERS.lock() {
                servers.retain(|addr| *addr != self.0);
            }
        }
    }
    let _clear = ClearOnDrop(server_ptr);

    // Capability handshake. Mirror `driver/portable/src/entry.cpp`'s
    // caps shape exactly so `embedded_driver::DriverCapabilities` parses
    // it without changes.
    let caps = serde_json::json!({
        "total_pages":      cfg.dummy.max_num_kv_pages,
        "kv_page_size":     cfg.dummy.kv_page_size,
        "swap_pool_size":   0u32,
        "max_batch_tokens": cfg.dummy.max_batch_tokens,
        "max_batch_size":   cfg.dummy.max_batch_size,
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

    server.serve_forever(|req, resp| {
        handled += 1;
        if req.method_tag != METHOD_TAG_FIRE_BATCH {
            eprintln!(
                "[pie-driver-dummy] unsupported method_tag={} req_id={}",
                req.method_tag, req.req_id
            );
            return 0;
        }
        handler.handle_fire_batch(req.payload, resp)
    });

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
