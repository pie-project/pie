//! `pie check` / `pie smoke` — diagnostics that run without booting
//! the full engine. Used during driver/standalone development; not
//! part of the Python `pie_cli` surface.

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
    let m = &cfg.model;
    let model_summary = format!("{}:{}", m.name, m.driver.kind.as_str());
    let auth = if cfg.auth.enabled {
        "auth enabled"
    } else {
        "auth disabled"
    };
    format!(
        "✓ {} valid: 1 model; {auth}; {model_summary}",
        path.display(),
    )
}

/// `pie smoke [--flavor <name>]` — verify the requested direct driver flavor is
/// compiled. Full create/launch smoke tests use a real model configuration.
///
/// `rpc` is accepted for backwards compatibility with older invocation
/// scripts but is now a no-op.
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
            "cuda" | "cuda_native" => DriverKind::CudaNative,
            "metal" => DriverKind::Metal,
            "dummy" => DriverKind::Dummy,
            other => {
                return Err(anyhow!(
                    "--flavor must be one of \"cuda\" | \"metal\" | \"dummy\" \
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

fn smoke_ffi(flavor: Flavor) -> Result<()> {
    println!("[smoke] direct {} driver is compiled", flavor.as_str());
    Ok(())
}
