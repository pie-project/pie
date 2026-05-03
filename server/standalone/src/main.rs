//! Pie — standalone (no-Python) server entry point.
//!
//! Usage:
//!   pie --config <toml>     Boot the runtime + embedded driver and serve.
//!   pie --check <toml>      Parse + validate a TOML config.
//!   pie --smoke             FFI smoke test: drive the embedded driver
//!                           entry with `--help` (no model load).
//!   pie --smoke-rpc         Construct a pie::device::RpcServer.

mod bootstrap_translate;
mod config;
mod driver_ffi;
mod embedded_driver;
mod rpc_loop;
mod serve;

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

unsafe extern "C" fn smoke_ready_cb(caps_json: *const c_char, _ctx: *mut c_void) {
    let json = unsafe { CStr::from_ptr(caps_json) }
        .to_string_lossy()
        .into_owned();
    println!("[smoke] ready_cb fired with {json}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--smoke") {
        std::process::exit(smoke_test());
    }
    if args.iter().any(|a| a == "--smoke-rpc") {
        std::process::exit(smoke_rpc());
    }
    if let Some(i) = args.iter().position(|a| a == "--check") {
        let path = match args.get(i + 1) {
            Some(p) => std::path::PathBuf::from(p),
            None => {
                eprintln!("--check requires a path argument");
                std::process::exit(2);
            }
        };
        std::process::exit(check_config(&path));
    }
    if let Some(i) = args.iter().position(|a| a == "--config") {
        let path = match args.get(i + 1) {
            Some(p) => std::path::PathBuf::from(p),
            None => {
                eprintln!("--config requires a path argument");
                std::process::exit(2);
            }
        };
        match serve::run(&path) {
            Ok(()) => std::process::exit(0),
            Err(e) => {
                eprintln!("pie: {e:#}");
                std::process::exit(1);
            }
        }
    }

    print_usage();
}

fn print_usage() {
    let build_dir = env!("PIE_DRIVER_BUILD_DIR");
    println!("pie-standalone (M2.4)");
    println!("  driver build dir: {build_dir}");
    println!("  embedded driver:  pie_driver_portable_lib (statically linked)");
    println!();
    println!("Usage:");
    println!("  pie --config <toml>   Boot the runtime + embedded driver and serve.");
    println!("  pie --check <toml>    Parse + validate a TOML config.");
    println!("  pie --smoke           FFI smoke test: invoke driver entry with --help.");
    println!("  pie --smoke-rpc       Construct a pie::device::RpcServer and report.");
}

fn check_config(path: &std::path::Path) -> i32 {
    match config::Config::from_toml_file(path) {
        Ok(cfg) => {
            println!("[ok] {path:?}");
            println!("{cfg:#?}");
            0
        }
        Err(e) => {
            eprintln!("[err] {path:?}: {e}");
            1
        }
    }
}

fn smoke_test() -> i32 {
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
    0
}

fn smoke_rpc() -> i32 {
    use pie::device::RpcServer;
    match RpcServer::create() {
        Ok(server) => {
            println!(
                "[smoke-rpc] RpcServer ready, server_name={}",
                server.server_name()
            );
            server.close();
            println!("[smoke-rpc] closed cleanly");
            0
        }
        Err(e) => {
            eprintln!("[smoke-rpc] RpcServer::create failed: {e}");
            1
        }
    }
}
