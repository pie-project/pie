//! Program Repository - Two-tier program storage
//!
//! Provides a 2-tier lookup system for programs:
//! 1. Disk index (Manifest with path/hash info)
//! 2. Binary cache (WASM bytes for user-registered programs)

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;

use super::ProgramName;
use super::manifest::{Manifest, manifest_url};

// =============================================================================
// Repository Types
// =============================================================================

/// Get the WASM file path for a program.
///
/// `programs_dir` is the directory this repository owns outright. It used to
/// be a `cache_dir` that these helpers joined `"programs"` onto -- while the
/// only caller already passed `$PIE_HOME/programs`, so every program landed in
/// `$PIE_HOME/programs/programs/<name>/`. Taking the directory itself leaves
/// nowhere for the two ends to disagree.
fn wasm_path(programs_dir: &Path, name: &ProgramName) -> PathBuf {
    programs_dir
        .join(&name.name)
        .join(format!("{}.wasm", name.version))
}

/// Get the manifest file path for a program.
fn manifest_path(programs_dir: &Path, name: &ProgramName) -> PathBuf {
    programs_dir
        .join(&name.name)
        .join(format!("{}.toml", name.version))
}

/// Build WASM download URL for a program from registry.
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
    /// Index: manifests for programs on disk
    index: HashMap<ProgramName, Manifest>,
    /// Preloaded binaries: WASM bytes staged for immediate first-use (consumed on fetch)
    preloaded_binaries: HashMap<ProgramName, Vec<u8>>,
    /// Registry URL for fallback downloads
    registry_url: String,
    /// The directory holding `<name>/<version>.{wasm,toml}`.
    programs_dir: PathBuf,
}

impl Repository {
    /// Create a new empty repository.
    pub fn new(registry_url: String, programs_dir: PathBuf) -> Self {
        Self {
            preloaded_binaries: HashMap::new(),
            index: HashMap::new(),
            registry_url,
            programs_dir,
        }
    }

    /// Get manifest from index.
    pub fn fetch_manifest(&self, name: &ProgramName) -> Option<Manifest> {
        self.index.get(name).cloned()
    }

    /// Fetch WASM bytes for a program (consuming from preloaded_binaries or loading from disk).
    pub async fn fetch_wasm_binary(&mut self, name: &ProgramName) -> Result<Vec<u8>> {
        // Consume from preloaded_binaries if present (e.g., user-registered program)
        if let Some(wasm_binary) = self.preloaded_binaries.remove(name) {
            return Ok(wasm_binary);
        }

        // Load from disk if in index
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
    /// one occupies.
    ///
    /// Exists so `pie inferlet list` can enumerate the cache without knowing
    /// the `<name>/<version>.wasm` layout. A second implementation of that
    /// layout is a second thing to keep in step with this one.
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
        // Prune the name directory once its last version goes, so the listing
        // does not accumulate empty shells. Failure is not an error: an
        // occupied directory means another version is still there.
        let _ = std::fs::remove_dir(self.programs_dir.join(&name.name));
        Ok(true)
    }

    /// Check if program is cached in repository.
    pub fn exists(&self, name: &ProgramName) -> bool {
        self.index.contains_key(name)
    }

    /// Add a program by name (downloads from registry).
    pub async fn add_from_registry(
        &mut self,
        name: &ProgramName,
        force_overwrite: bool,
    ) -> Result<()> {
        // Check if already exists (unless force_overwrite)
        if !force_overwrite && self.index.contains_key(name) {
            return Ok(()); // Already added
        }

        // Download manifest
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

        // Download WASM
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

        // Save to disk (updates index)
        self.store_program_cache(&wasm_binary, manifest).await?;

        // Store in preloaded binaries for first-use optimization
        self.preloaded_binaries.insert(name.clone(), wasm_binary);

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

        // Store in preloaded binaries for first-use optimization
        self.preloaded_binaries.insert(name, wasm_binary);

        Ok(())
    }

    /// Load programs from disk into the index.
    /// Scans `programs_dir` for `<name>/<version>.wasm` + its sibling manifest.
    pub fn load_program_cache(&mut self) {
        self.lift_doubled_programs_dir();
        let dir = self.programs_dir.clone();
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
    /// TEMPORARY -- delete once no tree in use predates the path fix. Without
    /// it every already-downloaded program becomes invisible at once and is
    /// silently re-fetched, which needs a reachable registry to recover from
    /// and leaves the old copies occupying disk with nothing pointing at them.
    fn lift_doubled_programs_dir(&self) {
        let nested = self.programs_dir.join("programs");
        let Ok(entries) = std::fs::read_dir(&nested) else {
            return;
        };
        for entry in entries.flatten() {
            let destination = self.programs_dir.join(entry.file_name());
            // Never over a live one: if both layouts hold a program, the one
            // already in the right place is the one the engine has been using.
            if !destination.exists() {
                let _ = std::fs::rename(entry.path(), destination);
            }
        }
        // Only when empty, so anything left behind stays findable.
        let _ = std::fs::remove_dir(&nested);
    }

    /// Save program to disk and update index.
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

        // Update index
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

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_toml(name: &str, version: &str) -> String {
        format!("[package]\nname = \"{name}\"\nversion = \"{version}\"\n")
    }

    /// Write a program straight to disk in the layout the repository reads,
    /// without going through the repository itself -- so these tests exercise
    /// the reader rather than agreeing with the writer.
    fn plant(root: &Path, name: &str, version: &str, bytes: usize) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{version}.wasm")), vec![0u8; bytes]).unwrap();
        std::fs::write(
            dir.join(format!("{version}.toml")),
            manifest_toml(name, version),
        )
        .unwrap();
    }

    fn open(root: &Path) -> Repository {
        let mut repo = Repository::new(String::new(), root.to_path_buf());
        repo.load_program_cache();
        repo
    }

    #[test]
    fn cached_reports_each_program_with_its_size() {
        let dir = tempfile::tempdir().unwrap();
        plant(dir.path(), "beta", "0.2.0", 200);
        plant(dir.path(), "alpha", "0.1.0", 100);
        plant(dir.path(), "alpha", "0.10.0", 150);

        let cached = open(dir.path()).cached();
        let ids: Vec<String> = cached
            .iter()
            .map(|(n, _, _)| format!("{}@{}", n.name, n.version))
            .collect();
        // Sorted, and two versions of one name are two rows.
        assert_eq!(ids, ["alpha@0.1.0", "alpha@0.10.0", "beta@0.2.0"]);
        assert_eq!(cached[0].2, 100);
        assert_eq!(cached[2].2, 200);
    }

    #[test]
    fn remove_takes_both_files_and_prunes_only_the_emptied_directory() {
        let dir = tempfile::tempdir().unwrap();
        plant(dir.path(), "alpha", "0.1.0", 10);
        plant(dir.path(), "alpha", "0.2.0", 10);
        let mut repo = open(dir.path());

        let first = ProgramName {
            name: "alpha".into(),
            version: "0.1.0".into(),
        };
        assert!(repo.remove(&first).unwrap());
        assert!(!dir.path().join("alpha/0.1.0.wasm").exists());
        assert!(!dir.path().join("alpha/0.1.0.toml").exists());
        // The sibling version still lives here, so the directory stays.
        assert!(dir.path().join("alpha/0.2.0.wasm").exists());
        assert_eq!(repo.cached().len(), 1);

        // Removing what is already gone is reported, not an error.
        assert!(!repo.remove(&first).unwrap());

        repo.remove(&ProgramName {
            name: "alpha".into(),
            version: "0.2.0".into(),
        })
        .unwrap();
        assert!(!dir.path().join("alpha").exists());
    }

    #[test]
    fn programs_left_in_the_old_doubled_directory_are_lifted_out() {
        // The layout `cache_dir.join("programs")` produced when the caller had
        // already appended `programs`.
        let dir = tempfile::tempdir().unwrap();
        plant(&dir.path().join("programs"), "alpha", "0.1.0", 42);

        let repo = open(dir.path());
        assert_eq!(repo.cached().len(), 1, "the old copy should be found");
        assert_eq!(repo.cached()[0].2, 42);
        assert!(dir.path().join("alpha/0.1.0.wasm").exists());
        assert!(!dir.path().join("programs").exists(), "emptied, so pruned");
    }

    #[test]
    fn lifting_never_overwrites_a_program_already_in_the_right_place() {
        // Both layouts hold `alpha`. The one the engine has been loading is
        // the one in the right place, so it must survive.
        let dir = tempfile::tempdir().unwrap();
        plant(dir.path(), "alpha", "0.1.0", 999);
        plant(&dir.path().join("programs"), "alpha", "0.1.0", 1);

        let repo = open(dir.path());
        assert_eq!(repo.cached()[0].2, 999);
        // Not pruned: something is still in there, so it stays findable.
        assert!(dir.path().join("programs/alpha/0.1.0.wasm").exists());
    }
}
