//! Program Manager Service - Inferlet program caching and loading
//!
//! This module provides a singleton actor for managing program (inferlet) metadata,
//! caching, downloading from registry, and compilation.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Result, anyhow, bail};
use dashmap::DashMap;
use tokio::sync::oneshot;
use wasmtime::Engine as WasmEngine;
use wasmtime::component::Component;

use crate::actor::{Actor, Handle, SendError};
use crate::runtime;

// =============================================================================
// Program Actor
// =============================================================================

/// Global singleton Program Manager actor.
static ACTOR: LazyLock<Actor<Message>> = LazyLock::new(Actor::new);

/// Spawns the Program Manager actor with configuration.
pub fn spawn(config: ProgramManagerConfig) {
    ACTOR.spawn_with::<ProgramManagerActor, _>(|| ProgramManagerActor::with_config(config));
}

/// Sends a message to the Program Manager actor.
pub fn send(msg: Message) -> Result<(), SendError> {
    ACTOR.send(msg)
}

/// Check if the program manager actor is spawned.
pub fn is_spawned() -> bool {
    ACTOR.is_spawned()
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the Program Manager.
#[derive(Debug)]
pub struct ProgramManagerConfig {
    pub wasm_engine: WasmEngine,
    pub registry_url: String,
    pub cache_dir: PathBuf,
}

// =============================================================================
// Messages
// =============================================================================

/// Messages for the Program Manager actor.
#[derive(Debug)]
pub enum Message {
    /// Get program metadata by name
    GetMetadata {
        name: ProgramName,
        response: oneshot::Sender<Option<ProgramMetadata>>,
    },

    /// Register a newly uploaded program
    RegisterUpload {
        name: ProgramName,
        metadata: ProgramMetadata,
    },

    /// Ensure a program and its dependencies are loaded in runtime
    EnsureLoaded {
        name: ProgramName,
        response: oneshot::Sender<Result<ProgramMetadata, String>>,
    },

    /// Download a program from registry
    DownloadFromRegistry {
        name: ProgramName,
        response: oneshot::Sender<Result<ProgramMetadata>>,
    },
}

impl Message {
    pub fn send(self) -> Result<(), SendError> {
        ACTOR.send(self)
    }
}

// =============================================================================
// Program Metadata Types
// =============================================================================

/// Identifier for an inferlet (name, version).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProgramName {
    pub name: String,
    pub version: String,
}

impl ProgramName {
    /// Parses an inferlet identifier from a string.
    ///
    /// Supported formats:
    /// - `name@version` -> (name, version)
    /// - `name` -> (name, "latest")
    pub fn parse(s: &str) -> Self {
        // Split on @ to get name and version
        let (name, version) = if let Some((n, v)) = s.split_once('@') {
            (n.to_string(), v.to_string())
        } else {
            (s.to_string(), "latest".to_string())
        };

        Self { name, version }
    }
}

impl std::fmt::Display for ProgramName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.name, self.version)
    }
}

/// Metadata for a cached inferlet program on disk.
#[derive(Clone, Debug)]
pub struct ProgramMetadata {
    /// Path to the WASM binary file
    pub wasm_path: PathBuf,
    /// Blake3 hash of the WASM binary
    pub wasm_hash: String,
    /// Blake3 hash of the manifest
    pub manifest_hash: String,
    /// Dependencies of this inferlet
    pub dependencies: Vec<ProgramName>,
}

// =============================================================================
// Program Service
// =============================================================================

/// The program service handles program caching and loading.
/// This is the core business logic, separate from the actor message handling.
#[derive(Debug)]
struct ProgramManager {
    wasm_engine: WasmEngine,
    registry_url: String,
    cache_dir: PathBuf,
    /// Uploaded programs on disk, keyed by program name
    uploaded_programs: DashMap<ProgramName, ProgramMetadata>,
    /// Registry-downloaded programs on disk, keyed by program name
    registry_programs: DashMap<ProgramName, ProgramMetadata>,
}

impl ProgramManager {
    fn new(config: ProgramManagerConfig) -> Self {
        let uploaded_programs = DashMap::new();
        let registry_programs = DashMap::new();

        let programs_dir = config.cache_dir.join("programs");
        if programs_dir.exists() {
            load_programs_from_dir(&programs_dir, &uploaded_programs);
        }

        let registry_dir = config.cache_dir.join("registry");
        if registry_dir.exists() {
            load_programs_from_dir(&registry_dir, &registry_programs);
        }

        ProgramManager {
            wasm_engine: config.wasm_engine,
            registry_url: config.registry_url,
            cache_dir: config.cache_dir,
            uploaded_programs,
            registry_programs,
        }
    }

    fn get_metadata(&self, name: &ProgramName) -> Option<ProgramMetadata> {
        self.uploaded_programs
            .get(name)
            .or_else(|| self.registry_programs.get(name))
            .map(|e| e.value().clone())
    }

    fn register_upload(&self, name: ProgramName, metadata: ProgramMetadata) {
        self.uploaded_programs.insert(name, metadata);
    }

    /// Access uploaded programs for external use
    pub fn uploaded_programs(&self) -> &DashMap<ProgramName, ProgramMetadata> {
        &self.uploaded_programs
    }

    /// Access registry programs for external use
    pub fn registry_programs(&self) -> &DashMap<ProgramName, ProgramMetadata> {
        &self.registry_programs
    }

    async fn ensure_loaded(&self, name: &ProgramName) -> Result<ProgramMetadata, String> {
        let metadata = match self.get_metadata(name) {
            Some(m) => m,
            None => {
                // Try to download from registry
                try_download_inferlet_from_registry(
                    &self.registry_url,
                    &self.cache_dir,
                    name,
                    &self.registry_programs,
                )
                .await
                .map_err(|e| e.to_string())?
            }
        };

        ensure_program_loaded_with_dependencies(
            &self.wasm_engine,
            &metadata,
            name,
            &self.uploaded_programs,
            &self.registry_programs,
            &self.registry_url,
            &self.cache_dir,
        )
        .await?;

        Ok(metadata)
    }

    async fn download_from_registry(&self, name: &ProgramName) -> Result<ProgramMetadata> {
        try_download_inferlet_from_registry(
            &self.registry_url,
            &self.cache_dir,
            name,
            &self.registry_programs,
        )
        .await
    }
}

// =============================================================================
// Program Actor
// =============================================================================

struct ProgramManagerActor {
    service: ProgramManager,
}

impl ProgramManagerActor {
    fn with_config(config: ProgramManagerConfig) -> Self {
        ProgramManagerActor {
            service: ProgramManager::new(config),
        }
    }
}

impl Handle for ProgramManagerActor {
    type Message = Message;

    fn new() -> Self {
        panic!("ProgramManagerActor requires config; use spawn() instead")
    }

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::GetMetadata { name, response } => {
                let result = self.service.get_metadata(&name);
                let _ = response.send(result);
            }
            Message::RegisterUpload { name, metadata } => {
                self.service.register_upload(name, metadata);
            }
            Message::EnsureLoaded { name, response } => {
                let result = self.service.ensure_loaded(&name).await;
                let _ = response.send(result);
            }
            Message::DownloadFromRegistry { name, response } => {
                let result = self.service.download_from_registry(&name).await;
                let _ = response.send(result);
            }
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Parses a manifest TOML string to extract the program name and version.
pub fn parse_program_name_from_manifest(manifest: &str) -> Result<ProgramName> {
    let table: toml::Table =
        toml::from_str(manifest).map_err(|e| anyhow!("Failed to parse manifest TOML: {}", e))?;

    let package = table
        .get("package")
        .and_then(|p| p.as_table())
        .ok_or_else(|| anyhow!("Manifest missing [package] section"))?;

    let name = package
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| anyhow!("Manifest missing package.name field"))?;

    let version = package
        .get("version")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Manifest missing package.version field"))?;

    Ok(ProgramName {
        name: name.to_string(),
        version: version.to_string(),
    })
}

/// Parses a manifest TOML string to extract dependencies as ProgramName entries.
pub fn parse_program_dependencies_from_manifest(manifest: &str) -> Vec<ProgramName> {
    let table: toml::Table = match toml::from_str(manifest) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    let Some(dependencies) = table.get("dependencies").and_then(|d| d.as_table()) else {
        return Vec::new();
    };

    dependencies
        .iter()
        .filter_map(|(name, value)| {
            let version = value
                .as_table()
                .and_then(|t| t.get("version"))
                .and_then(|v| v.as_str())
                .unwrap_or("latest");

            Some(ProgramName {
                name: name.clone(),
                version: version.to_string(),
            })
        })
        .collect()
}

/// Compiles WASM bytes to a Component in a blocking thread.
pub async fn compile_wasm_component(engine: &WasmEngine, wasm_bytes: Vec<u8>) -> Result<Component> {
    let engine = engine.clone();
    match tokio::task::spawn_blocking(move || Component::from_binary(&engine, &wasm_bytes)).await {
        Ok(Ok(component)) => Ok(component),
        Ok(Err(e)) => Err(anyhow!("Failed to compile WASM: {}", e)),
        Err(e) => Err(anyhow!("Compilation task failed: {}", e)),
    }
}

/// Ensures a program and all its dependencies are loaded in the runtime.
///
/// This function recursively loads all dependencies first, then loads the program itself.
/// It includes cycle detection to prevent infinite loops.
pub async fn ensure_program_loaded_with_dependencies(
    wasm_engine: &WasmEngine,
    program_metadata: &ProgramMetadata,
    program_name: &ProgramName,
    uploaded_programs: &DashMap<ProgramName, ProgramMetadata>,
    registry_programs: &DashMap<ProgramName, ProgramMetadata>,
    registry_url: &str,
    cache_dir: &Path,
) -> Result<(), String> {
    // Use a work queue to avoid async recursion
    // Each entry is (program_name, program_metadata, already_processed_deps)
    let mut work_stack: Vec<(ProgramName, ProgramMetadata, bool)> = Vec::new();
    let mut visited: HashSet<ProgramName> = HashSet::new();
    let mut loaded: HashSet<String> = HashSet::new();

    // Start with the main program
    work_stack.push((program_name.clone(), program_metadata.clone(), false));

    while let Some((current_name, current_metadata, deps_processed)) = work_stack.pop() {
        if deps_processed {
            // All dependencies are loaded, now load this program
            if loaded.contains(&current_metadata.wasm_hash) {
                continue;
            }

            let (loaded_tx, loaded_rx) = oneshot::channel();
            runtime::Message::ProgramLoaded {
                program_hash: runtime::ProgramHash::new(
                    current_metadata.wasm_hash.clone(),
                    current_metadata.manifest_hash.clone(),
                ),
                response: loaded_tx,
            }
            .send()
            .unwrap();

            let is_loaded = loaded_rx.await.unwrap();

            if !is_loaded {
                let raw_bytes = tokio::fs::read(&current_metadata.wasm_path)
                    .await
                    .map_err(|e| {
                        format!(
                            "Failed to read program from disk at {:?}: {}",
                            current_metadata.wasm_path, e
                        )
                    })?;

                let component = compile_wasm_component(wasm_engine, raw_bytes)
                    .await
                    .map_err(|e| e.to_string())?;

                let (load_tx, load_rx) = oneshot::channel();
                runtime::Message::LoadProgram {
                    program_hash: runtime::ProgramHash::new(
                        current_metadata.wasm_hash.clone(),
                        current_metadata.manifest_hash.clone(),
                    ),
                    component,
                    dependencies: current_metadata.dependencies.iter().map(|dep_name| {
                        // Look up each dependency's metadata to get its hashes
                        uploaded_programs.get(dep_name)
                            .or_else(|| registry_programs.get(dep_name))
                            .map(|m| runtime::ProgramHash::new(m.wasm_hash.clone(), m.manifest_hash.clone()))
                            .expect("dependency should be loaded by this point")
                    }).collect(),
                    response: load_tx,
                }
                .send()
                .unwrap();

                load_rx.await.unwrap();
            }

            loaded.insert(current_metadata.wasm_hash.clone());
        } else {
            // First visit: check for cycles and queue dependencies
            if visited.contains(&current_name) {
                return Err(format!(
                    "Dependency cycle detected for {}@{}",
                    current_name.name, current_name.version
                ));
            }
            visited.insert(current_name.clone());

            // Re-add this program with deps_processed = true (to load after deps)
            work_stack.push((current_name, current_metadata.clone(), true));

            // Add all dependencies to process (in reverse order so they're processed first)
            for dep in current_metadata.dependencies.iter().rev() {
                // Skip if already visited
                if visited.contains(dep) {
                    continue;
                }

                // Get dependency metadata
                let dep_metadata = if let Some(meta) = uploaded_programs
                    .get(dep)
                    .or_else(|| registry_programs.get(dep))
                    .map(|e| e.value().clone())
                {
                    meta
                } else {
                    // Download dependency from registry
                    try_download_inferlet_from_registry(registry_url, cache_dir, dep, registry_programs)
                        .await
                        .map_err(|e| {
                            format!("Failed to download dependency {}@{}: {}", dep.name, dep.version, e)
                        })?
                };

                work_stack.push((dep.clone(), dep_metadata, false));
            }
        }
    }

    Ok(())
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
            })?
            .trim()
            .to_string();
        let manifest_hash = tokio::fs::read_to_string(&manifest_hash_cache_path)
            .await
            .map_err(|e| {
                anyhow!(
                    "Failed to read cached manifest hash at {:?}: {}",
                    manifest_hash_cache_path,
                    e
                )
            })?
            .trim()
            .to_string();
        let manifest_data = tokio::fs::read_to_string(&manifest_cache_path)
            .await
            .map_err(|e| {
                anyhow!(
                    "Failed to read cached manifest at {:?}: {}",
                    manifest_cache_path,
                    e
                )
            })?;

        let dependencies = parse_program_dependencies_from_manifest(&manifest_data);

        let metadata = ProgramMetadata {
            wasm_path: wasm_cache_path,
            wasm_hash,
            manifest_hash,
            dependencies,
        };
        return Ok(metadata);
    }

    // Download from registry (using flat namespace)
    let base_url = registry_url.trim_end_matches('/');
    let wasm_download_url = format!(
        "{}/api/v1/inferlets/{}/{}/download",
        base_url, program_name.name, program_name.version
    );
    let manifest_download_url = format!(
        "{}/api/v1/inferlets/{}/{}/manifest",
        base_url, program_name.name, program_name.version
    );

    tracing::info!(
        "Downloading inferlet: {} @ {} from {}",
        program_name.name,
        program_name.version,
        wasm_download_url
    );

    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

    // Download WASM
    let wasm_response = client
        .get(&wasm_download_url)
        .send()
        .await
        .map_err(|e| anyhow!("Failed to download inferlet from registry: {}", e))?;

    if !wasm_response.status().is_success() {
        let status = wasm_response.status();
        let body = wasm_response.text().await.unwrap_or_default();
        bail!(
            "Registry returned error {} for {} @ {}: {}",
            status,
            program_name.name,
            program_name.version,
            body
        );
    }

    let wasm_data = wasm_response
        .bytes()
        .await
        .map_err(|e| anyhow!("Failed to read inferlet data: {}", e))?
        .to_vec();

    if wasm_data.is_empty() {
        bail!(
            "Registry returned empty data for {} @ {}",
            program_name.name,
            program_name.version
        );
    }

    // Download manifest
    tracing::info!(
        "Downloading manifest for {} @ {} from {}",
        program_name.name,
        program_name.version,
        manifest_download_url
    );

    let manifest_response = client
        .get(&manifest_download_url)
        .send()
        .await
        .map_err(|e| anyhow!("Failed to download manifest from registry: {}", e))?;

    if !manifest_response.status().is_success() {
        let status = manifest_response.status();
        let body = manifest_response.text().await.unwrap_or_default();
        bail!(
            "Registry returned error {} for manifest {} @ {}: {}",
            status,
            program_name.name,
            program_name.version,
            body
        );
    }

    let manifest_data = manifest_response
        .text()
        .await
        .map_err(|e| anyhow!("Failed to read manifest data: {}", e))?;

    // Compute hashes
    let wasm_hash = blake3::hash(&wasm_data).to_hex().to_string();
    let manifest_hash = blake3::hash(manifest_data.as_bytes()).to_hex().to_string();

    // Parse dependencies from manifest
    let dependencies = parse_program_dependencies_from_manifest(&manifest_data);

    // Cache to disk
    tokio::fs::create_dir_all(&cache_base)
        .await
        .map_err(|e| anyhow!("Failed to create cache directory {:?}: {}", cache_base, e))?;

    tokio::fs::write(&wasm_cache_path, &wasm_data)
        .await
        .map_err(|e| anyhow!("Failed to cache inferlet at {:?}: {}", wasm_cache_path, e))?;

    tokio::fs::write(&manifest_cache_path, &manifest_data)
        .await
        .map_err(|e| {
            anyhow!(
                "Failed to cache manifest at {:?}: {}",
                manifest_cache_path,
                e
            )
        })?;

    tokio::fs::write(&wasm_hash_cache_path, &wasm_hash)
        .await
        .map_err(|e| {
            anyhow!(
                "Failed to cache WASM hash at {:?}: {}",
                wasm_hash_cache_path,
                e
            )
        })?;

    tokio::fs::write(&manifest_hash_cache_path, &manifest_hash)
        .await
        .map_err(|e| {
            anyhow!(
                "Failed to cache manifest hash at {:?}: {}",
                manifest_hash_cache_path,
                e
            )
        })?;

    tracing::info!(
        "Cached inferlet {} @ {} to {:?} (wasm_hash: {}, manifest_hash: {})",
        program_name.name,
        program_name.version,
        wasm_cache_path,
        wasm_hash,
        manifest_hash
    );

    let metadata = ProgramMetadata {
        wasm_path: wasm_cache_path,
        wasm_hash,
        manifest_hash,
        dependencies,
    };

    // Add to in-memory map
    registry_programs_in_disk.insert(program_name.clone(), metadata.clone());

    Ok(metadata)
}

/// Helper to load programs from a directory with structure {dir}/{name}/{version}.wasm
/// Uses flat namespace structure (no namespace subdirectory).
pub fn load_programs_from_dir(dir: &Path, programs_in_disk: &DashMap<ProgramName, ProgramMetadata>) {
    let name_entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };

    for name_entry in name_entries.flatten() {
        let name_path = name_entry.path();
        if !name_path.is_dir() {
            continue;
        }
        let name = match name_path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        let file_entries = match std::fs::read_dir(&name_path) {
            Ok(entries) => entries,
            Err(_) => continue,
        };

        for file_entry in file_entries.flatten() {
            let file_path = file_entry.path();
            if file_path.extension().is_some_and(|ext| ext == "wasm") {
                let version = match file_path.file_stem().and_then(|s| s.to_str()) {
                    Some(v) => v.to_string(),
                    None => continue,
                };

                // Read WASM hash
                let wasm_hash_path = name_path.join(format!("{}.wasm_hash", version));
                let wasm_hash = match std::fs::read_to_string(&wasm_hash_path) {
                    Ok(h) => h.trim().to_string(),
                    Err(_) => continue,
                };

                // Read manifest hash
                let manifest_hash_path = name_path.join(format!("{}.toml_hash", version));
                let manifest_hash = match std::fs::read_to_string(&manifest_hash_path) {
                    Ok(h) => h.trim().to_string(),
                    Err(_) => continue,
                };

                // Read manifest for dependencies
                let manifest_path = name_path.join(format!("{}.toml", version));
                let dependencies = match std::fs::read_to_string(&manifest_path) {
                    Ok(manifest) => parse_program_dependencies_from_manifest(&manifest),
                    Err(_) => Vec::new(),
                };

                let key = ProgramName {
                    name: name.clone(),
                    version,
                };
                let metadata = ProgramMetadata {
                    wasm_path: file_path,
                    wasm_hash,
                    manifest_hash,
                    dependencies,
                };
                programs_in_disk.insert(key, metadata);
            }
        }
    }
}
