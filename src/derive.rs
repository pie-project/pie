//! Derive the three **typed** role Configs from the single standalone TOML.
//!
//! Domain parsing stays in the role libs: `pie` owns the standalone-config
//! *schema* but never re-parses domain types — it hands each role's section to
//! that role's `Config::parse`. Compose then boots from the typed Configs (it
//! never sees a raw string).

use anyhow::{Context, Result};

/// Read the combined standalone config file (the `[controller]/[gateway]/[worker]`
/// TOML the CLI ops operate on). A `pie` concern: the skeleton sources config for
/// the daemon boot path; ops read it directly here.
pub fn read_config_file(path: &std::path::Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("reading config file {}", path.display()))
}

/// Load just the `[worker]` role Config from the combined standalone file — for
/// ops that need worker-domain settings (registry, engines) without booting the
/// cluster. Replaces the old worker-only `Config::from_toml_file`.
pub fn load_worker_config(path: &std::path::Path) -> Result<worker::Config> {
    worker::Config::parse(&read_config_file(path)?).context("parsing config")
}

/// Extract one top-level `[section]` from the combined standalone config as a
/// standalone TOML string (its contents promoted to top level, e.g.
/// `[worker.engine]` → `[engine]`). A **missing** section yields an empty string
/// — the role lib then applies its own defaults (matching
/// `bootstrap::config::source`'s empty-on-missing contract). A present section
/// that isn't a table is a config error.
pub fn extract_section(combined: &str, section: &str) -> Result<String> {
    let root: toml::Table = combined.parse().context("parsing standalone config TOML")?;

    match root.get(section) {
        None => Ok(String::new()),
        Some(toml::Value::Table(t)) => {
            toml::to_string(t).with_context(|| format!("re-serializing [{section}] section"))
        }
        Some(_) => anyhow::bail!("standalone config key `{section}` must be a table ([{section}])"),
    }
}

/// The three typed role Configs for the in-proc standalone, parsed from the
/// combined `config_str`. Each role's own `Config::parse` does the domain
/// validation; loopback/in-proc address wiring is applied by `compose`
/// (`run_standalone` binds ephemeral and cross-wires worker↔gateway).
pub fn derive_standalone(
    combined: &str,
) -> Result<(controller::Config, gateway::Config, worker::Config)> {
    // The file IS the worker config: a single-node deployment states nothing
    // for the other two roles, so both take their own defaults here. `compose`
    // then wires the addresses, and `[server] host:port` is what the client
    // edge binds.
    let worker = worker::Config::parse(combined).context("parsing config")?;
    let controller = controller::Config::parse("").context("controller defaults")?;
    let gateway = gateway::Config::parse("").context("gateway defaults")?;
    Ok((controller, gateway, worker))
}

