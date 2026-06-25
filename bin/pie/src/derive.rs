//! P5b: derive the three **typed** role Configs from the single standalone TOML.
//!
//! Per the manager's ruling, domain parsing stays in the role libs: bin/pie owns
//! the standalone-config *schema* (one TOML with `[controller]`/`[gateway]`/
//! `[worker]` sections) but never re-parses domain types — it splits each section
//! into a standalone TOML string and hands it to that role's `Config::parse`.
//! Compose then boots from the typed Configs (it never sees a raw string).

use anyhow::{Context, Result};

/// Read the combined standalone config file (the `[controller]/[gateway]/[worker]`
/// TOML the CLI ops operate on). A bin/pie concern: bootstrap sources config for
/// the daemon boot path; ops read it directly here.
pub fn read_config_file(path: &std::path::Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("reading config file {}", path.display()))
}

/// Load just the `[worker]` role Config from the combined standalone file — for
/// ops that need worker-domain settings (registry, drivers) without booting the
/// cluster. Replaces the old worker-only `Config::from_toml_file`.
pub fn load_worker_config(path: &std::path::Path) -> Result<pie_worker::Config> {
    let combined = read_config_file(path)?;
    pie_worker::Config::parse(&extract_section(&combined, "worker")?)
        .context("parsing [worker] section of the standalone config")
}

/// Extract one top-level `[section]` from the combined standalone config as a
/// standalone TOML string (its contents promoted to top level, e.g.
/// `[worker.driver]` → `[driver]`). A **missing** section yields an empty string
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
/// (golf's `run_standalone` binds ephemeral and cross-wires worker↔gateway).
pub fn derive_standalone(
    combined: &str,
) -> Result<(
    pie_controller::Config,
    pie_gateway::Config,
    pie_worker::Config,
)> {
    let controller = pie_controller::Config::parse(&extract_section(combined, "controller")?)
        .context("parsing [controller] section")?;
    let gateway = pie_gateway::Config::parse(&extract_section(combined, "gateway")?)
        .context("parsing [gateway] section")?;
    let worker = pie_worker::Config::parse(&extract_section(combined, "worker")?)
        .context("parsing [worker] section")?;
    Ok((controller, gateway, worker))
}

#[cfg(test)]
mod tests {
    use super::*;

    const STANDALONE: &str = r#"
[controller]
[gateway]
[worker]
name = "w0"
[worker.driver]
kind = "dummy"
"#;

    #[test]
    fn promotes_nested_tables_to_top_level() {
        let w = extract_section(STANDALONE, "worker").unwrap();
        let parsed: toml::Table = w.parse().unwrap();
        assert_eq!(parsed["name"].as_str(), Some("w0"));
        assert!(parsed["driver"].is_table(), "driver promoted: {w}");
        assert_eq!(parsed["driver"]["kind"].as_str(), Some("dummy"));
        assert!(parsed.get("worker").is_none());
    }

    #[test]
    fn missing_section_is_empty() {
        assert_eq!(extract_section("[worker]\n", "controller").unwrap(), "");
        assert_eq!(extract_section("", "gateway").unwrap(), "");
    }

    #[test]
    fn non_table_section_errors() {
        assert!(extract_section("worker = 3\n", "worker").is_err());
    }
}
