//! Program Repository - Two-tier program storage
//!
//! Provides a 2-tier lookup system for programs:
//! 1. Disk index (Manifest with path/hash info)
//! 2. Binary cache (WASM bytes for user-registered programs)

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;

use super::ProgramName;
use super::manifest::Manifest;

// =============================================================================
// Repository Types
// =============================================================================

/// Get the directory path for a program in the cache.
fn program_dir(cache_dir: &Path, name: &ProgramName) -> PathBuf {
    cache_dir.join("programs").join(&name.name)
}

/// Get the WASM file path for a program.
fn wasm_path(cache_dir: &Path, name: &ProgramName) -> PathBuf {
    program_dir(cache_dir, name).join(format!("{}.wasm", name.version))
}

/// Get the manifest file path for a program.
fn manifest_path(cache_dir: &Path, name: &ProgramName) -> PathBuf {
    program_dir(cache_dir, name).join(format!("{}.toml", name.version))
}

/// Two-tier program repository: disk index + binary cache.
pub struct Repository {
    /// Index: manifests for programs on disk
    index: HashMap<ProgramName, Manifest>,
    /// Binary cache: WASM bytes for registered programs (consumed on fetch)
    wasm_binary_cache: HashMap<ProgramName, Vec<u8>>,
    /// Registry URL for fallback downloads
    registry_url: String,
    /// Cache directory for disk storage
    cache_dir: PathBuf,
}

impl Repository {
    /// Create a new empty repository.
    pub fn new(registry_url: String, cache_dir: PathBuf) -> Self {
        Self {
            wasm_binary_cache: HashMap::new(),
            index: HashMap::new(),
            registry_url,
            cache_dir,
        }
    }

    /// Get manifest from index.
    pub fn fetch_manifest(&self, name: &ProgramName) -> Option<Manifest> {
        self.index.get(name).cloned()
    }


    /// Fetch WASM bytes for a program (consuming from wasm_binary_cache or loading from disk).
    pub async fn fetch_wasm_binary(&mut self, name: &ProgramName) -> Result<Vec<u8>> {
        // Consume from wasm_binary_cache if present (e.g., user-registered program)
        if let Some(wasm_binary) = self.wasm_binary_cache.remove(name) {
            return Ok(wasm_binary);
        }

        // Load from disk if in index
        if self.index.contains_key(name) {
            let wasm = wasm_path(&self.cache_dir, name);
            let wasm_binary = tokio::fs::read(&wasm)
                .await
                .map_err(|e| anyhow!("Failed to read WASM file: {}", e))?;
            return Ok(wasm_binary);
        }

        bail!("Program not found: {}", name)
    }


    /// Check if program is cached in repository.
    pub fn exists(&self, name: &ProgramName) -> bool {
        self.index.contains_key(name) || self.wasm_binary_cache.contains_key(name)
    }

    /// Add a program by name (downloads from registry).
    pub async fn add_from_registry(&mut self, name: &ProgramName, force_overwrite: bool) -> Result<()> {
        // Check if already exists (unless force_overwrite)
        if !force_overwrite && self.index.contains_key(name) {
            return Ok(()); // Already added
        }

        // Download manifest
        let manifest_url = format!(
            "{}/api/v1/inferlets/{}/{}/manifest",
            self.registry_url.trim_end_matches('/'),
            name.name,
            name.version
        );
        let manifest_response = reqwest::get(&manifest_url).await
            .map_err(|e| anyhow!("Failed to download manifest from {}: {}", manifest_url, e))?;

        if !manifest_response.status().is_success() {
            bail!("Failed to download manifest: {} returned {}", manifest_url, manifest_response.status());
        }

        let manifest_content = manifest_response.text().await
            .map_err(|e| anyhow!("Failed to read manifest response: {}", e))?;
        let manifest = Manifest::parse(&manifest_content)?;

        // Download WASM
        let wasm_url = format!(
            "{}/api/v1/inferlets/{}/{}/download",
            self.registry_url.trim_end_matches('/'),
            name.name,
            name.version
        );
        let wasm_response = reqwest::get(&wasm_url).await
            .map_err(|e| anyhow!("Failed to download WASM from {}: {}", wasm_url, e))?;

        if !wasm_response.status().is_success() {
            bail!("Failed to download WASM: {} returned {}", wasm_url, wasm_response.status());
        }

        let wasm_binary = wasm_response.bytes().await
            .map_err(|e| anyhow!("Failed to read WASM response: {}", e))?
            .to_vec();

        // Save to disk (updates index)
        self.store_program_cache(&wasm_binary, manifest).await?;

        // Store in binary cache
        self.wasm_binary_cache.insert(name.clone(), wasm_binary);

        Ok(())
    }

    /// Add a program manually with provided WASM binary and manifest.
    pub async fn add(
        &mut self,
        wasm_binary: Vec<u8>,
        manifest: Manifest,
        force_overwrite: bool,
    ) -> Result<()> {
        let name = manifest.program_name();

        // Check if already registered (unless force_overwrite)
        if !force_overwrite && self.index.contains_key(&name) {
            return Ok(());
        }

        // Save to disk (updates index)
        self.store_program_cache(&wasm_binary, manifest).await?;

        // Store in binary cache
        self.wasm_binary_cache.insert(name, wasm_binary);

        Ok(())
    }

    /// Load programs from disk into the index.
    /// Scans `{cache_dir}/programs` for cached programs.
    pub fn load_program_cache(&mut self) {
        let dir = self.cache_dir.join("programs");
        if !dir.exists() {
            return;
        }

        // List all subdirectories (each is a program name)
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let program_name_str = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            // Look for version files: {version}.wasm
            let version_entries = match std::fs::read_dir(&path) {
                Ok(e) => e,
                Err(_) => continue,
            };

            for version_entry in version_entries.flatten() {
                let version_path = version_entry.path();
                if version_path.extension().and_then(|e| e.to_str()) != Some("wasm") {
                    continue;
                }

                let version = match version_path.file_stem().and_then(|s| s.to_str()) {
                    Some(v) => v.to_string(),
                    None => continue,
                };

                let program_name = ProgramName {
                    name: program_name_str.clone(),
                    version: version.clone(),
                };

                let manifest_file = manifest_path(&self.cache_dir, &program_name);

                let manifest_content = match std::fs::read_to_string(&manifest_file) {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                // Parse manifest
                let manifest = match Manifest::parse(&manifest_content) {
                    Ok(m) => m,
                    Err(_) => continue,
                };

                self.index.insert(program_name, manifest);
            }
        }
    }

    /// Save program to disk and update index.
    async fn store_program_cache(
        &mut self,
        wasm_binary: &[u8],
        manifest: Manifest,
    ) -> Result<()> {
        let name = manifest.program_name();
        let dir = program_dir(&self.cache_dir, &name);
        let wasm = wasm_path(&self.cache_dir, &name);
        let manifest_file = manifest_path(&self.cache_dir, &name);

        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| anyhow!("Failed to create directory {:?}: {}", dir, e))?;

        tokio::fs::write(&wasm, wasm_binary)
            .await
            .map_err(|e| anyhow!("Failed to write WASM file: {}", e))?;
        tokio::fs::write(&manifest_file, manifest.to_string())
            .await
            .map_err(|e| anyhow!("Failed to write manifest file: {}", e))?;

        // Update index
        self.index.insert(name, manifest);

        Ok(())
    }
}

impl std::fmt::Debug for Repository {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Repository")
            .field("wasm_binary_cache_count", &self.wasm_binary_cache.len())
            .field("index_count", &self.index.len())
            .field("registry_url", &self.registry_url)
            .finish()
    }
}

