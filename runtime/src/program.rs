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

mod repository;
pub use repository::{InMemoryEntry, ProgramRepository};
use repository::{load_programs_from_dir, try_download_inferlet_from_registry};

// =============================================================================
// Program Actor
// =============================================================================

/// Global singleton Program Manager actor.
static ACTOR: LazyLock<Actor<Message>> = LazyLock::new(Actor::new);

/// Spawns the Program Manager actor with configuration.
pub fn spawn(config: ProgramManagerConfig) {
    let repository = std::sync::Arc::new(ProgramRepository::new(
        config.registry_url.clone(),
        config.cache_dir.clone(),
    ));

    // Scan disk on startup: load existing programs into on_disk tier
    let programs_dir = config.cache_dir.join("programs");
    if programs_dir.exists() {
        load_programs_from_dir(&programs_dir, repository.on_disk_mut());
    }
    let registry_dir = config.cache_dir.join("registry");
    if registry_dir.exists() {
        load_programs_from_dir(&registry_dir, repository.on_disk_mut());
    }

    ACTOR.spawn_with::<ProgramManagerActor, _>(|| {
        ProgramManagerActor::new(config.wasm_engine, repository)
    });
}

/// Sends a message to the Program Manager actor.
pub fn send(msg: Message) -> Result<(), SendError> {
    ACTOR.send(msg)
}

/// Check if the program manager actor is spawned.
pub fn is_spawned() -> bool {
    ACTOR.is_spawned()
}

/// Upload a new program. Stores in memory + disk (does NOT install).
pub async fn upload(
    wasm_bytes: Vec<u8>,
    manifest: String,
    expected_hash: String,
) -> Result<UploadResult, String> {
    let (tx, rx) = oneshot::channel();
    Message::UploadProgram {
        wasm_bytes,
        manifest,
        expected_hash,
        response: tx,
    }
    .send()
    .map_err(|_| "Program manager not running".to_string())?;
    rx.await.map_err(|_| "Program manager did not respond".to_string())?
}

/// Check if a program exists in repository (any tier) with optional hash verification.
pub async fn program_exists(name: &ProgramName, expected_hashes: Option<(String, String)>) -> bool {
    let (tx, rx) = oneshot::channel();
    let _ = Message::ProgramExists {
        name: name.clone(),
        expected_hashes,
        response: tx,
    }
    .send();
    rx.await.unwrap_or(false)
}

/// Check if a program is installed (JIT compiled and ready to run).
pub async fn is_program_installed(name: &ProgramName) -> bool {
    let (tx, rx) = oneshot::channel();
    let _ = Message::IsProgramInstalled {
        name: name.clone(),
        response: tx,
    }
    .send();
    rx.await.unwrap_or(false)
}

/// Install a program: JIT compile + link, auto-downloads from registry if needed, resolves dependencies.
pub async fn install(name: &ProgramName) -> Result<ProgramMetadata, String> {
    let (tx, rx) = oneshot::channel();
    Message::InstallProgram {
        name: name.clone(),
        response: tx,
    }
    .send()
    .map_err(|_| "Program manager not running".to_string())?;
    rx.await.map_err(|_| "Program manager did not respond".to_string())?
}

/// Get program metadata by name.
pub async fn get_metadata(name: &ProgramName) -> Option<ProgramMetadata> {
    let (tx, rx) = oneshot::channel();
    let _ = Message::GetMetadata {
        name: name.clone(),
        response: tx,
    }
    .send();
    rx.await.ok().flatten()
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

    /// Upload a new program: store in memory + disk (does NOT install)
    UploadProgram {
        wasm_bytes: Vec<u8>,
        manifest: String,
        expected_hash: String,
        response: oneshot::Sender<Result<UploadResult, String>>,
    },

    /// Check if a program exists in repository (any tier) with optional hash verification
    ProgramExists {
        name: ProgramName,
        expected_hashes: Option<(String, String)>,
        response: oneshot::Sender<bool>,
    },

    /// Check if a program is installed (JIT compiled and ready to run)
    IsProgramInstalled {
        name: ProgramName,
        response: oneshot::Sender<bool>,
    },

    /// Install a program: JIT compile + link, auto-downloads from registry if needed, resolves dependencies
    InstallProgram {
        name: ProgramName,
        response: oneshot::Sender<Result<ProgramMetadata, String>>,
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

/// Result of a successful program upload.
#[derive(Clone, Debug)]
pub struct UploadResult {
    pub wasm_hash: String,
    pub manifest_hash: String,
}

// =============================================================================
// Program Manager (Service)
// =============================================================================

/// The program service handles program caching, installation, and loading.
/// This is the core business logic, separate from the actor message handling.
struct ProgramManager {
    wasm_engine: WasmEngine,
    repository: std::sync::Arc<ProgramRepository>,
    /// Installed (JIT compiled) programs, keyed by program hash
    installed_programs: std::collections::HashMap<runtime::ProgramHash, Component>,
}

impl ProgramManager {
    fn new(wasm_engine: WasmEngine, repository: std::sync::Arc<ProgramRepository>) -> Self {
        ProgramManager {
            wasm_engine,
            repository,
            installed_programs: std::collections::HashMap::new(),
        }
    }

    fn get_metadata(&self, name: &ProgramName) -> Option<ProgramMetadata> {
        self.repository.get_metadata(name)
    }

    fn program_exists(&self, name: &ProgramName, expected_hashes: Option<(String, String)>) -> bool {
        self.repository.exists_with_hash(name, expected_hashes)
    }

    fn is_program_installed(&self, name: &ProgramName) -> bool {
        if let Some(metadata) = self.repository.get_metadata(name) {
            let hash = runtime::ProgramHash::new(metadata.wasm_hash, metadata.manifest_hash);
            self.installed_programs.contains_key(&hash)
        } else {
            false
        }
    }

    /// Upload a new program: verify hash, store in memory + disk (does NOT install).
    async fn upload_program(
        &mut self,
        wasm_bytes: Vec<u8>,
        manifest: String,
        expected_hash: String,
    ) -> Result<UploadResult, String> {
        // Verify the hash
        let wasm_hash = blake3::hash(&wasm_bytes).to_hex().to_string();
        if wasm_hash != expected_hash {
            return Err(format!(
                "Hash mismatch: expected {}, got {}",
                expected_hash, wasm_hash
            ));
        }

        // Parse manifest for name and dependencies
        let program_name = parse_program_name_from_manifest(&manifest)
            .map_err(|e| format!("Failed to parse manifest: {}", e))?;
        let dependencies = parse_program_dependencies_from_manifest(&manifest);

        // Compute manifest hash
        let manifest_hash = blake3::hash(manifest.as_bytes()).to_hex().to_string();

        // Write to disk: {cache_dir}/programs/{name}/{version}.{wasm,toml,wasm_hash,toml_hash}
        let dir_path = self.repository.cache_dir.join("programs").join(&program_name.name);
        tokio::fs::create_dir_all(&dir_path)
            .await
            .map_err(|e| format!("Failed to create directory {:?}: {}", dir_path, e))?;

        let wasm_file_path = dir_path.join(format!("{}.wasm", program_name.version));
        let manifest_file_path = dir_path.join(format!("{}.toml", program_name.version));
        let wasm_hash_file_path = dir_path.join(format!("{}.wasm_hash", program_name.version));
        let manifest_hash_file_path = dir_path.join(format!("{}.toml_hash", program_name.version));

        tokio::fs::write(&wasm_file_path, &wasm_bytes)
            .await
            .map_err(|e| format!("Failed to write WASM file: {}", e))?;
        tokio::fs::write(&manifest_file_path, &manifest)
            .await
            .map_err(|e| format!("Failed to write manifest file: {}", e))?;
        tokio::fs::write(&wasm_hash_file_path, &wasm_hash)
            .await
            .map_err(|e| format!("Failed to write WASM hash file: {}", e))?;
        tokio::fs::write(&manifest_hash_file_path, &manifest_hash)
            .await
            .map_err(|e| format!("Failed to write manifest hash file: {}", e))?;

        // Create metadata
        let metadata = ProgramMetadata {
            wasm_path: wasm_file_path.clone(),
            wasm_hash: wasm_hash.clone(),
            manifest_hash: manifest_hash.clone(),
            dependencies: dependencies.clone(),
        };

        // Register in repository (memory + disk tiers)
        self.repository.register(program_name, wasm_bytes, metadata);

        Ok(UploadResult {
            wasm_hash,
            manifest_hash,
        })
    }

    /// Install a program: JIT compile + link, auto-downloads from registry if needed, resolves dependencies.
    async fn install_program(&mut self, name: &ProgramName) -> Result<ProgramMetadata, String> {
        // Step 1: Find program in repository, or download from registry
        let metadata = match self.repository.get_metadata(name) {
            Some(m) => m,
            None => {
                // Download from registry (this will also get dependencies)
                try_download_inferlet_from_registry(
                    &self.repository.registry_url,
                    &self.repository.cache_dir,
                    name,
                    self.repository.on_disk_mut(),
                )
                .await
                .map_err(|e| e.to_string())?
            }
        };

        // Step 2: Install dependencies first (recursive)
        for dep_name in &metadata.dependencies {
            if !self.is_program_installed(dep_name) {
                Box::pin(self.install_program(dep_name)).await?;
            }
        }

        // Step 3: Check if already installed
        let program_hash = runtime::ProgramHash::new(metadata.wasm_hash.clone(), metadata.manifest_hash.clone());
        if self.installed_programs.contains_key(&program_hash) {
            return Ok(metadata);
        }

        // Step 4: Fetch WASM bytes (uses repository.fetch() internally)
        let entry = self.repository.fetch(name).await?;
        let wasm_bytes = entry.wasm_bytes;

        // Step 5: JIT compile
        let component = compile_wasm_component(&self.wasm_engine, wasm_bytes)
            .await
            .map_err(|e| e.to_string())?;

        // Step 6: Register with runtime
        let dep_hashes: Vec<runtime::ProgramHash> = metadata
            .dependencies
            .iter()
            .map(|dep_name| {
                let dep_meta = self.repository.get_metadata(dep_name)
                    .expect("dependency should exist after installation");
                runtime::ProgramHash::new(dep_meta.wasm_hash, dep_meta.manifest_hash)
            })
            .collect();

        let (evt_tx, evt_rx) = oneshot::channel();
        runtime::Message::LoadProgram {
            program_hash: program_hash.clone(),
            component: component.clone(),
            dependencies: dep_hashes,
            response: evt_tx,
        }
        .send()
        .map_err(|_| "Failed to send LoadProgram message")?;

        evt_rx.await.map_err(|_| "Failed to receive LoadProgram response")?;

        // Step 7: Track as installed
        self.installed_programs.insert(program_hash, component);

        Ok(metadata)
    }
}

// =============================================================================
// Program Actor
// =============================================================================

struct ProgramManagerActor {
    service: ProgramManager,
}

impl ProgramManagerActor {
    fn new(wasm_engine: WasmEngine, repository: std::sync::Arc<ProgramRepository>) -> Self {
        ProgramManagerActor {
            service: ProgramManager::new(wasm_engine, repository),
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
            Message::UploadProgram { wasm_bytes, manifest, expected_hash, response } => {
                let result = self.service.upload_program(wasm_bytes, manifest, expected_hash).await;
                let _ = response.send(result);
            }
            Message::ProgramExists { name, expected_hashes, response } => {
                let result = self.service.program_exists(&name, expected_hashes);
                let _ = response.send(result);
            }
            Message::IsProgramInstalled { name, response } => {
                let result = self.service.is_program_installed(&name);
                let _ = response.send(result);
            }
            Message::InstallProgram { name, response } => {
                let result = self.service.install_program(&name).await;
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
