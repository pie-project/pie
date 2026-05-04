//! `pie check` / `pie smoke` — diagnostics that run without booting
//! the full engine. Used during driver/standalone development; not
//! part of the Python `pie_cli` surface.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;

use anyhow::Result;

use crate::{config, driver_ffi};

/// `pie check <toml>` — parse + validate a config TOML and pretty-
/// print the resulting struct. Exits non-zero on parse / validation
/// failure.
pub fn check(path: &Path) -> Result<()> {
    match config::Config::from_toml_file(path) {
        Ok(cfg) => {
            println!("[ok] {path:?}");
            println!("{cfg:#?}");
            Ok(())
        }
        Err(e) => {
            eprintln!("[err] {path:?}: {e}");
            std::process::exit(1);
        }
    }
}

/// `pie smoke [--rpc]` — exercise the FFI / RPC plumbing without a
/// real model load. `rpc=false` invokes the embedded driver's entry
/// with `--help` and reports its rc; `rpc=true` constructs an
/// `RpcServer` and confirms it can be opened + closed.
pub fn smoke(rpc: bool) -> Result<()> {
    if rpc { smoke_rpc() } else { smoke_ffi() }
}

unsafe extern "C" fn smoke_ready_cb(caps_json: *const c_char, _ctx: *mut c_void) {
    let json = unsafe { CStr::from_ptr(caps_json) }
        .to_string_lossy()
        .into_owned();
    println!("[smoke] ready_cb fired with {json}");
}

fn smoke_ffi() -> Result<()> {
    println!(
        "[smoke] invoking pie_driver_{}_run(--help)…\n",
        driver_ffi::FLAVOR
    );
    let argv_strs = vec![
        CString::new(driver_ffi::ARGV0).unwrap(),
        CString::new("--help").unwrap(),
    ];
    let mut argv_ptrs: Vec<*mut c_char> = argv_strs
        .iter()
        .map(|s| s.as_ptr() as *mut c_char)
        .collect();
    let rc = unsafe {
        driver_ffi::run(
            argv_ptrs.len() as c_int,
            argv_ptrs.as_mut_ptr(),
            0,
            smoke_ready_cb,
            std::ptr::null_mut(),
        )
    };
    println!("\n[smoke] driver entry returned rc={rc}");
    Ok(())
}

fn smoke_rpc() -> Result<()> {
    use pie::device::RpcServer;
    match RpcServer::create() {
        Ok(server) => {
            println!(
                "[smoke-rpc] RpcServer ready, server_name={}",
                server.server_name()
            );
            server.close();
            println!("[smoke-rpc] closed cleanly");
            Ok(())
        }
        Err(e) => {
            eprintln!("[smoke-rpc] RpcServer::create failed: {e}");
            std::process::exit(1);
        }
    }
}
