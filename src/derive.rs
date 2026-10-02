use anyhow::{Context, Result};

pub fn read_config_file(path: &std::path::Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("reading config file {}", path.display()))
}

pub fn load_worker_config(path: &std::path::Path) -> Result<worker::Config> {
    worker::Config::parse(&read_config_file(path)?).context("parsing config")
}

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

pub fn derive_standalone(
    combined: &str,
) -> Result<(controller::Config, gateway::Config, worker::Config)> {
    let worker = worker::Config::parse(combined).context("parsing config")?;
    let controller = controller::Config::parse("").context("controller defaults")?;
    let mut gateway = gateway::Config::parse("").context("gateway defaults")?;
    gateway.admission_wait_ms = worker.runtime.admission_wait.as_micros() / 1000;
    Ok((controller, gateway, worker))
}
