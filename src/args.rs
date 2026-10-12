//! The flags every pie binary takes, and the config file they name.

use std::path::PathBuf;

use crate::paths;

#[derive(clap::Args, Clone, Debug)]
pub struct GlobalArgs {
    #[arg(short = 'c', long, value_name = "PATH")]
    pub config: Option<String>,
    #[arg(long, value_name = "LEVEL", default_value = "info")]
    pub log_level: String,
    #[arg(long, value_name = "ADDR")]
    pub metrics_addr: Option<String>,
}

/// Where the config path came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    Flag,
    Env,
    Default,
}

impl Origin {
    pub fn is_explicit(self) -> bool {
        matches!(self, Self::Flag | Self::Env)
    }

    pub fn describe(self) -> &'static str {
        match self {
            Self::Flag => "--config flag",
            Self::Env => "$PIE_CONFIG",
            Self::Default => "$PIE_HOME default",
        }
    }
}

/// The config `pie` reads: `--config`, else `$PIE_CONFIG`, else
/// `$PIE_HOME/config.toml`.
pub fn config_path(global: &GlobalArgs) -> (PathBuf, Origin) {
    config_path_or(global, "config.toml")
}

/// [`config_path`], with `default` under `$PIE_HOME` when neither the flag
/// nor the environment names one.
pub fn config_path_or(global: &GlobalArgs, default: &str) -> (PathBuf, Origin) {
    if let Some(flag) = global.config.as_deref() {
        return (PathBuf::from(flag), Origin::Flag);
    }
    if let Ok(env) = std::env::var("PIE_CONFIG")
        && !env.trim().is_empty()
    {
        return (PathBuf::from(env), Origin::Env);
    }
    (paths::pie_home_file(default), Origin::Default)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        global: GlobalArgs,
        #[arg(long)]
        listen: Option<String>,
    }

    #[test]
    fn the_global_flags_flatten_beside_a_binarys_own() {
        let cli = TestCli::try_parse_from(["bin", "--listen", "1.2.3.4:5"]).unwrap();
        assert_eq!(cli.global.log_level, "info");
        assert_eq!(cli.global.config, None);
        assert_eq!(cli.listen.as_deref(), Some("1.2.3.4:5"));

        let cli = TestCli::try_parse_from([
            "bin",
            "-c",
            "/tmp/x.toml",
            "--log-level",
            "debug",
            "--metrics-addr",
            "0.0.0.0:9",
        ])
        .unwrap();
        assert_eq!(cli.global.config.as_deref(), Some("/tmp/x.toml"));
        assert_eq!(cli.global.log_level, "debug");
        assert_eq!(cli.global.metrics_addr.as_deref(), Some("0.0.0.0:9"));
    }
}
