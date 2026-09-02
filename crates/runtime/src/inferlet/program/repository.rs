//! Program repository: two-tier program storage, a disk index (manifest with
//! path/hash info) plus a binary cache (WASM bytes for user-registered
//! programs).

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;

use super::ProgramName;
use super::manifest::{Manifest, manifest_url};

/// `programs_dir` is the directory this repository owns outright, so the
/// wasm and manifest paths below cannot disagree about a "programs" prefix.
fn wasm_path(programs_dir: &Path, name: &ProgramName) -> PathBuf {
    programs_dir
        .join(&name.name)
        .join(format!("{}.wasm", name.version))
}

fn manifest_path(programs_dir: &Path, name: &ProgramName) -> PathBuf {
    programs_dir
        .join(&name.name)
        .join(format!("{}.toml", name.version))
}

fn wasm_url(registry_url: &str, name: &ProgramName) -> String {
    format!(
        "{}/api/v1/inferlets/{}/{}/download",
        registry_url.trim_end_matches('/'),
        name.name,
        name.version
    )
}

/// Two-tier program repository: disk index + binary cache.
pub struct Repository {
    index: HashMap<ProgramName, Manifest>,
    /// WASM bytes staged for immediate first-use, consumed on fetch.
    preloaded_binaries: HashMap<ProgramName, Vec<u8>>,
    registry_url: String,
    /// The directory holding `<name>/<version>.{wasm,toml}`.
    programs_dir: PathBuf,
}

impl Repository {
    pub fn new(registry_url: String, programs_dir: PathBuf) -> Self {
        Self {
            preloaded_binaries: HashMap::new(),
            index: HashMap::new(),
            registry_url,
            programs_dir,
        }
    }

    pub fn fetch_manifest(&self, name: &ProgramName) -> Option<Manifest> {
        self.index.get(name).cloned()
    }

    pub async fn fetch_wasm_binary(&mut self, name: &ProgramName) -> Result<Vec<u8>> {
        if let Some(wasm_binary) = self.preloaded_binaries.remove(name) {
            return Ok(wasm_binary);
        }
        if self.index.contains_key(name) {
            let wasm = wasm_path(&self.programs_dir, name);
            let wasm_binary = tokio::fs::read(&wasm)
                .await
                .map_err(|e| anyhow!("Failed to read WASM file: {}", e))?;
            return Ok(wasm_binary);
        }

        bail!("Program not found: {}", name)
    }

    /// Every program on disk, name-then-version ordered, with the bytes each
    /// one occupies. Exists so `pie inferlet list` can enumerate the cache
    /// without knowing the `<name>/<version>.wasm` layout itself.
    pub fn cached(&self) -> Vec<(ProgramName, Manifest, u64)> {
        let mut out: Vec<(ProgramName, Manifest, u64)> = self
            .index
            .iter()
            .map(|(name, manifest)| {
                let size = std::fs::metadata(wasm_path(&self.programs_dir, name))
                    .map(|m| m.len())
                    .unwrap_or(0);
                (name.clone(), manifest.clone(), size)
            })
            .collect();
        out.sort_by(|a, b| (&a.0.name, &a.0.version).cmp(&(&b.0.name, &b.0.version)));
        out
    }

    /// Delete a cached program. Returns whether it was there to delete.
    ///
    /// Removes the manifest before the wasm: a program whose manifest is gone
    /// is skipped by `load_program_cache`, so an interrupted removal leaves
    /// something invisible rather than something half-loadable.
    pub fn remove(&mut self, name: &ProgramName) -> Result<bool> {
        if self.index.remove(name).is_none() && !wasm_path(&self.programs_dir, name).exists() {
            return Ok(false);
        }
        self.preloaded_binaries.remove(name);
        for path in [
            manifest_path(&self.programs_dir, name),
            wasm_path(&self.programs_dir, name),
        ] {
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(anyhow!("removing {:?}: {}", path, e)),
            }
        }
        // Prune the name directory once its last version goes; failure isn't
        // an error, since it just means another version is still there.
        let _ = std::fs::remove_dir(self.programs_dir.join(&name.name));
        Ok(true)
    }

    pub fn exists(&self, name: &ProgramName) -> bool {
        self.index.contains_key(name)
    }

    pub async fn add_from_registry(
        &mut self,
        name: &ProgramName,
        force_overwrite: bool,
    ) -> Result<()> {
        if !force_overwrite && self.index.contains_key(name) {
            return Ok(());
        }

        let url = manifest_url(&self.registry_url, name);
        let manifest_response = reqwest::get(&url)
            .await
            .map_err(|e| anyhow!("Failed to download manifest from {}: {}", url, e))?;

        if !manifest_response.status().is_success() {
            bail!(
                "Failed to download manifest: {} returned {}",
                url,
                manifest_response.status()
            );
        }

        let manifest_content = manifest_response
            .text()
            .await
            .map_err(|e| anyhow!("Failed to read manifest response: {}", e))?;
        let manifest = Manifest::parse(&manifest_content)?;

        let url = wasm_url(&self.registry_url, name);
        let wasm_response = reqwest::get(&url)
            .await
            .map_err(|e| anyhow!("Failed to download WASM from {}: {}", url, e))?;

        if !wasm_response.status().is_success() {
            bail!(
                "Failed to download WASM: {} returned {}",
                url,
                wasm_response.status()
            );
        }

        let wasm_binary = wasm_response
            .bytes()
            .await
            .map_err(|e| anyhow!("Failed to read WASM response: {}", e))?
            .to_vec();

        self.store_program_cache(&wasm_binary, manifest).await?;
        self.preloaded_binaries.insert(name.clone(), wasm_binary);

        Ok(())
    }

    pub async fn add(
        &mut self,
        wasm_binary: Vec<u8>,
        manifest: Manifest,
        force_overwrite: bool,
    ) -> Result<()> {
        let name = manifest.program_name();

        if !force_overwrite && self.index.contains_key(&name) {
            return Ok(());
        }

        self.store_program_cache(&wasm_binary, manifest).await?;
        self.preloaded_binaries.insert(name, wasm_binary);

        Ok(())
    }

    /// Scans `programs_dir` for `<name>/<version>.wasm` + its sibling manifest.
    pub fn load_program_cache(&mut self) {
        self.lift_doubled_programs_dir();
        let dir = self.programs_dir.clone();
        if !dir.exists() {
            return;
        }

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

                let manifest_file = manifest_path(&self.programs_dir, &program_name);

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

    /// Move programs out of the `programs/programs/` directory the old
    /// `cache_dir` doubling wrote them to.
    ///
    /// Temporary: migrates the old doubled `programs/programs/` layout.
    fn lift_doubled_programs_dir(&self) {
        let nested = self.programs_dir.join("programs");
        let Ok(entries) = std::fs::read_dir(&nested) else {
            return;
        };
        for entry in entries.flatten() {
            let destination = self.programs_dir.join(entry.file_name());
            // Never over a live one.
            if !destination.exists() {
                let _ = std::fs::rename(entry.path(), destination);
            }
        }
        // Only when empty, so anything left behind stays findable.
        let _ = std::fs::remove_dir(&nested);
    }

    async fn store_program_cache(&mut self, wasm_binary: &[u8], manifest: Manifest) -> Result<()> {
        let name = manifest.program_name();
        let dir = self.programs_dir.join(&name.name);
        let wasm = wasm_path(&self.programs_dir, &name);
        let manifest_file = manifest_path(&self.programs_dir, &name);

        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| anyhow!("Failed to create directory {:?}: {}", dir, e))?;

        tokio::fs::write(&wasm, wasm_binary)
            .await
            .map_err(|e| anyhow!("Failed to write WASM file: {}", e))?;
        tokio::fs::write(&manifest_file, manifest.to_toml()?)
            .await
            .map_err(|e| anyhow!("Failed to write manifest file: {}", e))?;

        self.index.insert(name, manifest);

        Ok(())
    }
}

impl std::fmt::Debug for Repository {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Repository")
            .field("preloaded_binaries_count", &self.preloaded_binaries.len())
            .field("index_count", &self.index.len())
            .field("registry_url", &self.registry_url)
            .finish()
    }
}

