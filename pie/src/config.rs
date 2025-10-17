//! Configuration file structure for the Pie CLI.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// Helper struct for parsing the TOML config file
#[derive(Deserialize, Serialize, Debug)]
pub struct ConfigFile {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub enable_auth: Option<bool>,
    pub auth_secret: Option<String>,
    pub cache_dir: Option<PathBuf>,
    pub verbose: Option<bool>,
    pub log: Option<PathBuf>,
    #[serde(default)]
    pub backend: Vec<toml::Value>,
}
