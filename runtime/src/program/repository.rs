//! Program Repository - Three-tier program storage
//!
//! Provides a 3-tier lookup system for programs:
//! 1. In-memory (WASM bytes loaded)
//! 2. On-disk (path + metadata)
//! 3. Registry (remote download)

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use dashmap::DashMap;

use super::{ProgramName, ProgramMetadata};

// =============================================================================
// Repository Types
// =============================================================================

/// Entry in the in-memory tier of the repository.
#[derive(Clone, Debug)]
pub struct InMemoryEntry {
    pub wasm_bytes: Vec<u8>,
    pub metadata: ProgramMetadata,
}

/// Three-tier program repository: memory -> disk -> registry.
pub struct ProgramRepository {
    /// Tier 1: In-memory (WASM bytes loaded)
    in_memory: DashMap<ProgramName, InMemoryEntry>,
    /// Tier 2: On-disk (path + metadata, scanned on startup)
    on_disk: DashMap<ProgramName, ProgramMetadata>,
    /// Tier 3 config: Registry URL for fallback downloads
    pub registry_url: String,
    /// Cache directory for disk storage
    pub cache_dir: PathBuf,
}

impl ProgramRepository {
    /// Create a new empty repository.
    pub fn new(registry_url: String, cache_dir: PathBuf) -> Self {
        Self {
            in_memory: DashMap::new(),
            on_disk: DashMap::new(),
            registry_url,
            cache_dir,
        }
    }

    /// Get metadata from any tier (memory first, then disk).
    pub fn get_metadata(&self, name: &ProgramName) -> Option<ProgramMetadata> {
        // Check in-memory first
        if let Some(entry) = self.in_memory.get(name) {
            return Some(entry.metadata.clone());
        }
        // Then check on-disk
        self.on_disk.get(name).map(|e| e.value().clone())
    }

    /// Check if program exists in repository (any tier).
    pub fn exists(&self, name: &ProgramName) -> bool {
        self.in_memory.contains_key(name) || self.on_disk.contains_key(name)
    }

    /// Check if program exists with optional hash verification.
    pub fn exists_with_hash(&self, name: &ProgramName, expected: Option<(String, String)>) -> bool {
        let metadata = self.get_metadata(name);
        match (metadata, expected) {
            (Some(m), Some((wasm, manifest))) => m.wasm_hash == wasm && m.manifest_hash == manifest,
            (Some(_), None) => true,
            (None, _) => false,
        }
    }

    /// Register a program in both memory and disk tiers.
    pub fn register(&self, name: ProgramName, wasm_bytes: Vec<u8>, metadata: ProgramMetadata) {
        self.in_memory.insert(
            name.clone(),
            InMemoryEntry {
                wasm_bytes,
                metadata: metadata.clone(),
            },
        );
        self.on_disk.insert(name, metadata);
    }

    /// Fetch a program into in-memory tier (from disk or registry).
    pub async fn fetch(&self, name: &ProgramName) -> Result<InMemoryEntry, String> {
        // Already in memory?
        if let Some(entry) = self.in_memory.get(name) {
            return Ok(entry.clone());
        }

        // On disk?
        if let Some(metadata) = self.on_disk.get(name).map(|e| e.clone()) {
            let wasm_bytes = tokio::fs::read(&metadata.wasm_path)
                .await
                .map_err(|e| format!("Failed to read WASM file: {}", e))?;

            let entry = InMemoryEntry {
                wasm_bytes,
                metadata: metadata.clone(),
            };
            self.in_memory.insert(name.clone(), entry.clone());
            return Ok(entry);
        }

        // Download from registry
        let metadata = try_download_inferlet_from_registry(
            &self.registry_url,
            &self.cache_dir,
            name,
            &self.on_disk,
        )
        .await
        .map_err(|e| e.to_string())?;

        // Load into memory
        let wasm_bytes = tokio::fs::read(&metadata.wasm_path)
            .await
            .map_err(|e| format!("Failed to read WASM file: {}", e))?;

        let entry = InMemoryEntry {
            wasm_bytes,
            metadata: metadata.clone(),
        };
        self.in_memory.insert(name.clone(), entry.clone());

        Ok(entry)
    }

    /// Get reference to on_disk DashMap (for internal use during startup scan).
    pub(super) fn on_disk_mut(&self) -> &DashMap<ProgramName, ProgramMetadata> {
        &self.on_disk
    }
}

impl std::fmt::Debug for ProgramRepository {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProgramRepository")
            .field("in_memory_count", &self.in_memory.len())
            .field("on_disk_count", &self.on_disk.len())
            .field("registry_url", &self.registry_url)
            .finish()
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Load programs from a directory into the on_disk DashMap.
pub fn load_programs_from_dir(dir: &Path, programs: &DashMap<ProgramName, ProgramMetadata>) {
    // List all subdirectories (each is a program name)
    let entries = match std::fs::read_dir(dir) {
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

            let manifest_path = path.join(format!("{}.toml", version));
            let wasm_hash_path = path.join(format!("{}.wasm_hash", version));
            let manifest_hash_path = path.join(format!("{}.toml_hash", version));

            // Read hashes
            let wasm_hash = match std::fs::read_to_string(&wasm_hash_path) {
                Ok(h) => h.trim().to_string(),
                Err(_) => continue,
            };
            let manifest_hash = match std::fs::read_to_string(&manifest_hash_path) {
                Ok(h) => h.trim().to_string(),
                Err(_) => continue,
            };

            // Parse dependencies from manifest
            let dependencies = if manifest_path.exists() {
                if let Ok(manifest_content) = std::fs::read_to_string(&manifest_path) {
                    super::parse_program_dependencies_from_manifest(&manifest_content)
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            let program_name = ProgramName {
                name: program_name_str.clone(),
                version: version.clone(),
            };

            let metadata = ProgramMetadata {
                wasm_path: version_path,
                wasm_hash,
                manifest_hash,
                dependencies,
            };

            programs.insert(program_name, metadata);
        }
    }
}

/// Downloads an inferlet from the registry, with local caching.
/// Uses flat namespace structure: {cache_dir}/registry/{name}/{version}.{wasm,toml,wasm_hash,toml_hash}
pub async fn try_download_inferlet_from_registry(
    registry_url: &str,
    cache_dir: &Path,
    program_name: &ProgramName,
    registry_programs_in_disk: &DashMap<ProgramName, ProgramMetadata>,
) -> Result<ProgramMetadata> {
    let cache_base = cache_dir.join("registry").join(&program_name.name);
    let wasm_cache_path = cache_base.join(format!("{}.wasm", program_name.version));
    let manifest_cache_path = cache_base.join(format!("{}.toml", program_name.version));
    let wasm_hash_cache_path = cache_base.join(format!("{}.wasm_hash", program_name.version));
    let manifest_hash_cache_path = cache_base.join(format!("{}.toml_hash", program_name.version));

    // Check if already cached
    if wasm_cache_path.exists()
        && manifest_cache_path.exists()
        && wasm_hash_cache_path.exists()
        && manifest_hash_cache_path.exists()
    {
        tracing::info!(
            "Using cached inferlet: {} @ {} from {:?}",
            program_name.name,
            program_name.version,
            wasm_cache_path
        );
        let wasm_hash = tokio::fs::read_to_string(&wasm_hash_cache_path)
            .await
            .map_err(|e| {
                anyhow!(
                    "Failed to read cached WASM hash at {:?}: {}",
                    wasm_hash_cache_path,
                    e
                )
            })?;
        let manifest_hash = tokio::fs::read_to_string(&manifest_hash_cache_path)
            .await
            .map_err(|e| {
                anyhow!(
                    "Failed to read cached manifest hash at {:?}: {}",
                    manifest_hash_cache_path,
                    e
                )
            })?;
        let manifest_content = tokio::fs::read_to_string(&manifest_cache_path)
            .await
            .map_err(|e| {
                anyhow!(
                    "Failed to read cached manifest at {:?}: {}",
                    manifest_cache_path,
                    e
                )
            })?;
        let dependencies = super::parse_program_dependencies_from_manifest(&manifest_content);

        let metadata = ProgramMetadata {
            wasm_path: wasm_cache_path,
            wasm_hash: wasm_hash.trim().to_string(),
            manifest_hash: manifest_hash.trim().to_string(),
            dependencies,
        };
        registry_programs_in_disk.insert(program_name.clone(), metadata.clone());
        return Ok(metadata);
    }

    tracing::info!(
        "Downloading inferlet from registry: {} @ {}",
        program_name.name,
        program_name.version
    );

    // Download manifest first
    let manifest_url = format!(
        "{}/v2/inferlet/{}/{}/manifest.toml",
        registry_url.trim_end_matches('/'),
        program_name.name,
        program_name.version
    );
    let manifest_response = reqwest::get(&manifest_url).await.map_err(|e| {
        anyhow!(
            "Failed to download manifest from {}: {}",
            manifest_url,
            e
        )
    })?;

    if !manifest_response.status().is_success() {
        bail!(
            "Failed to download manifest: {} returned {}",
            manifest_url,
            manifest_response.status()
        );
    }

    let manifest_content = manifest_response.text().await.map_err(|e| {
        anyhow!("Failed to read manifest response: {}", e)
    })?;
    let manifest_hash = blake3::hash(manifest_content.as_bytes())
        .to_hex()
        .to_string();
    let dependencies = super::parse_program_dependencies_from_manifest(&manifest_content);

    // Download WASM
    let wasm_url = format!(
        "{}/v2/inferlet/{}/{}/program.wasm",
        registry_url.trim_end_matches('/'),
        program_name.name,
        program_name.version
    );
    let wasm_response = reqwest::get(&wasm_url).await.map_err(|e| {
        anyhow!("Failed to download WASM from {}: {}", wasm_url, e)
    })?;

    if !wasm_response.status().is_success() {
        bail!(
            "Failed to download WASM: {} returned {}",
            wasm_url,
            wasm_response.status()
        );
    }

    let wasm_bytes = wasm_response.bytes().await.map_err(|e| {
        anyhow!("Failed to read WASM response: {}", e)
    })?;
    let wasm_hash = blake3::hash(&wasm_bytes).to_hex().to_string();

    // Cache to disk
    tokio::fs::create_dir_all(&cache_base).await.map_err(|e| {
        anyhow!("Failed to create cache directory {:?}: {}", cache_base, e)
    })?;

    tokio::fs::write(&wasm_cache_path, &wasm_bytes)
        .await
        .map_err(|e| anyhow!("Failed to write WASM cache: {}", e))?;
    tokio::fs::write(&manifest_cache_path, &manifest_content)
        .await
        .map_err(|e| anyhow!("Failed to write manifest cache: {}", e))?;
    tokio::fs::write(&wasm_hash_cache_path, &wasm_hash)
        .await
        .map_err(|e| anyhow!("Failed to write WASM hash cache: {}", e))?;
    tokio::fs::write(&manifest_hash_cache_path, &manifest_hash)
        .await
        .map_err(|e| anyhow!("Failed to write manifest hash cache: {}", e))?;

    let metadata = ProgramMetadata {
        wasm_path: wasm_cache_path,
        wasm_hash,
        manifest_hash,
        dependencies,
    };
    registry_programs_in_disk.insert(program_name.clone(), metadata.clone());

    Ok(metadata)
}
