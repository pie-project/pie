//! `pie check` / `pie smoke` — diagnostics that run without booting
//! the full engine. Used during driver/standalone development; not
//! part of the Python `pie_cli` surface.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;

use anyhow::{Result, anyhow};

use pie_worker::config::{self, DriverKind};
use pie_worker::driver_ffi::{self, Flavor};

/// `pie check <toml>` — parse + validate the standalone config TOML (all three
/// `[controller]`/`[gateway]`/`[worker]` sections via their role `Config::parse`).
/// Exits non-zero on parse / validation failure.
pub fn check(path: &Path, debug: bool) -> Result<()> {
    let combined = match crate::derive::read_config_file(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[err] {path:?}: {e}");
            std::process::exit(1);
        }
    };
    match crate::derive::derive_standalone(&combined) {
        Ok((_controller, _gateway, worker)) => {
            println!("{}", check_summary(path, &worker));
            if debug {
                println!();
                println!("{worker:#?}");
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("[err] {path:?}: {e:#}");
            std::process::exit(1);
        }
    }
}

fn check_summary(path: &Path, cfg: &config::Config) -> String {
    let model_count = cfg.models.len();
    let model_word = if model_count == 1 { "model" } else { "models" };
    let model_summary = cfg
        .models
        .iter()
        .map(|m| format!("{}:{}", m.name, m.driver.kind.as_str()))
        .collect::<Vec<_>>()
        .join(", ");
    let auth = if cfg.auth.enabled {
        "auth enabled"
    } else {
        "auth disabled"
    };
    format!(
        "✓ {} valid: {model_count} {model_word}; {auth}; {model_summary}",
        path.display(),
    )
}

/// `pie smoke [--flavor <name>]` — exercise the FFI plumbing without a
/// real model load. Invokes the requested driver's entry with `--help`
/// and reports its rc.
///
/// `rpc` is accepted for backwards compatibility with older invocation
/// scripts but is now a no-op — the cold-path RPC infrastructure has
/// been retired in favor of the unified DriverChannel.
pub fn smoke(rpc: bool, flavor_name: Option<&str>) -> Result<()> {
    if rpc {
        eprintln!("[smoke] --rpc is a no-op; cold-path RpcServer has been retired");
    }
    let flavor = pick_smoke_flavor(flavor_name)?;
    smoke_ffi(flavor)
}

fn pick_smoke_flavor(name: Option<&str>) -> Result<Flavor> {
    if let Some(n) = name {
        // Reuse the TOML kind parser for symmetry: same string set.
        let kind: DriverKind = match n {
            "portable" => DriverKind::Portable,
            "cuda" | "cuda_native" => DriverKind::CudaNative,
            "dummy" => DriverKind::Dummy,
            other => {
                return Err(anyhow!(
                    "--flavor must be one of \"portable\" | \"cuda\" | \"dummy\" \
                     (got {other:?}). Compiled flavors: {compiled}.",
                    compiled = driver_ffi::compiled_summary(),
                ));
            }
        };
        Flavor::from_kind(kind).map_err(|m| anyhow!("{m}"))
    } else {
        driver_ffi::default_flavor()
            .ok_or_else(|| anyhow!("no driver flavor compiled into this binary"))
    }
}

unsafe extern "C" fn smoke_ready_cb(caps_json: *const c_char, _ctx: *mut c_void) {
    let json = unsafe { CStr::from_ptr(caps_json) }
        .to_string_lossy()
        .into_owned();
    println!("[smoke] ready_cb fired with {json}");
}

fn smoke_ffi(flavor: Flavor) -> Result<()> {
    println!(
        "[smoke] invoking pie_driver_{}_run(--help)…\n",
        flavor.as_str()
    );
    let argv_strs = [
        CString::new(flavor.argv0()).unwrap(),
        CString::new("--help").unwrap(),
    ];
    let mut argv_ptrs: Vec<*mut c_char> = argv_strs
        .iter()
        .map(|s| s.as_ptr() as *mut c_char)
        .collect();
    let rc = unsafe {
        driver_ffi::run(
            flavor,
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
