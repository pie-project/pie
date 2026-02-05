//! Program Manager Service - Inferlet program caching and loading
//!
//! This module provides a singleton actor for managing program (inferlet) metadata,
//! caching, downloading from registry, and compilation.

use std::path::PathBuf;
use std::sync::LazyLock;

use anyhow::{Result, anyhow};
use tokio::sync::oneshot;
use wasmtime::Engine as WasmEngine;
use wasmtime::component::Component;

use crate::actor::{Actor, Handle, SendError};

mod manifest;
mod repository;
pub use manifest::Manifest;
pub use repository::Repository;

// =============================================================================
// Program Actor
// =============================================================================

/// Global singleton Program Manager actor.
static ACTOR: LazyLock<Actor<Message>> = LazyLock::new(Actor::new);

/// Spawns the Program Manager actor with configuration.
pub fn spawn(config: ProgramManagerConfig) {
    let mut repository = Repository::new(
        config.registry_url.clone(),
        config.cache_dir.clone(),
    );

    // Scan disk on startup: load existing programs into index
    repository.load_program_cache();

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

/// Register a new program. Stores in memory + disk (does NOT install).
pub async fn register(
    wasm_binary: Vec<u8>,
    manifest: String,
    force_overwrite: bool,
) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    Message::Register {
        wasm_binary,
        manifest,
        force_overwrite,
        response: tx,
    }
    .send()
    .map_err(|_| anyhow!("Program manager not running"))?;
    rx.await.map_err(|_| anyhow!("Program manager did not respond"))?
}

/// Check if a program is registered in repository.
pub async fn is_registered(name: &ProgramName) -> bool {
    let (tx, rx) = oneshot::channel();
    let _ = Message::IsRegistered {
        name: name.clone(),
        response: tx,
    }
    .send();
    rx.await.unwrap_or(false)
}

/// Check if a program is installed (JIT compiled and ready to run).
pub async fn is_installed(name: &ProgramName) -> bool {
    let (tx, rx) = oneshot::channel();
    let _ = Message::IsInstalled {
        name: name.clone(),
        response: tx,
    }
    .send();
    rx.await.unwrap_or(false)
}

/// Install a program: JIT compile + link, auto-downloads from registry if needed, resolves dependencies.
pub async fn install(name: &ProgramName) -> Result<Manifest> {
    let (tx, rx) = oneshot::channel();
    Message::Install {
        name: name.clone(),
        response: tx,
    }
    .send()
    .map_err(|_| anyhow!("Program manager not running"))?;
    rx.await.map_err(|_| anyhow!("Program manager did not respond"))?
}

/// Uninstall a program: remove from installed programs (does NOT remove from cache).
pub async fn uninstall(name: &ProgramName) -> bool {
    let (tx, rx) = oneshot::channel();
    let _ = Message::Uninstall {
        name: name.clone(),
        response: tx,
    }
    .send();
    rx.await.unwrap_or(false)
}

/// Get program metadata by name.
pub async fn fetch_manifest(name: &ProgramName) -> Option<Manifest> {
    let (tx, rx) = oneshot::channel();
    let _ = Message::GetMetadata {
        name: name.clone(),
        response: tx,
    }
    .send();
    rx.await.ok().flatten()
}

/// Get the compiled component for an installed program.
pub async fn get_component(name: &ProgramName) -> Option<Component> {
    let (tx, rx) = oneshot::channel();
    let _ = Message::GetWasmComponent {
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
///
/// Note: Component doesn't implement Debug, so we manually implement it.
pub enum Message {
    /// Get program metadata by name
    GetMetadata {
        name: ProgramName,
        response: oneshot::Sender<Option<Manifest>>,
    },

    /// Register a new program: store in memory + disk (does NOT install)
    Register {
        wasm_binary: Vec<u8>,
        manifest: String,
        force_overwrite: bool,
        response: oneshot::Sender<Result<()>>,
    },

    /// Check if a program is registered in repository
    IsRegistered {
        name: ProgramName,
        response: oneshot::Sender<bool>,
    },

    /// Check if a program is installed (JIT compiled and ready to run)
    IsInstalled {
        name: ProgramName,
        response: oneshot::Sender<bool>,
    },

    /// Install a program: JIT compile + link, auto-downloads from registry if needed, resolves dependencies
    Install {
        name: ProgramName,
        response: oneshot::Sender<Result<Manifest>>,
    },

    /// Uninstall a program: remove from installed programs (does NOT remove from cache)
    Uninstall {
        name: ProgramName,
        response: oneshot::Sender<bool>,
    },

    /// Get the compiled component for an installed program
    GetWasmComponent {
        name: ProgramName,
        response: oneshot::Sender<Option<Component>>,
    },
}

impl std::fmt::Debug for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Message::GetMetadata { name, .. } => write!(f, "GetMetadata {{ name: {:?} }}", name),
            Message::Register { force_overwrite, .. } => write!(f, "Register {{ force_overwrite: {} }}", force_overwrite),
            Message::IsRegistered { name, .. } => write!(f, "IsRegistered {{ name: {:?} }}", name),
            Message::IsInstalled { name, .. } => write!(f, "IsInstalled {{ name: {:?} }}", name),
            Message::Install { name, .. } => write!(f, "Install {{ name: {:?} }}", name),
            Message::Uninstall { name, .. } => write!(f, "Uninstall {{ name: {:?} }}", name),
            Message::GetWasmComponent { name, .. } => write!(f, "GetWasmComponent {{ name: {:?} }}", name),
        }
    }
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
// =============================================================================
// Program Manager (Service)
// =============================================================================

/// The program service handles program caching, installation, and loading.
/// This is the core business logic, separate from the actor message handling.
struct ProgramManager {
    wasm_engine: WasmEngine,
    repository: Repository,
    /// Installed (JIT compiled) programs, keyed by program name
    installed: std::collections::HashMap<ProgramName, (Component, Manifest)>,
}

impl ProgramManager {
    fn new(wasm_engine: WasmEngine, repository: Repository) -> Self {
        ProgramManager {
            wasm_engine,
            repository,
            installed: std::collections::HashMap::new(),
        }
    }

    fn is_installed(&self, name: &ProgramName) -> bool {
        self.installed.contains_key(name)
    }

    fn get_installed_metadata(&self, name: &ProgramName) -> Option<&Manifest> {
        self.installed.get(name).map(|(_, m)| m)
    }

    fn get_component(&self, name: &ProgramName) -> Option<Component> {
        self.installed.get(name).map(|(c, _)| c.clone())
    }

    fn is_registered(&self, name: &ProgramName) -> bool {
        self.repository.exists(name)
    }

    /// Uninstall a program: remove from installed programs (does NOT remove from cache).
    fn uninstall(&mut self, name: &ProgramName) -> bool {
        self.installed.remove(name).is_some()
    }

    /// Register a new program: store in memory + disk (does NOT install).
    async fn register(
        &mut self,
        wasm_binary: Vec<u8>,
        manifest_content: String,
        force_overwrite: bool,
    ) -> Result<()> {
        let manifest = Manifest::parse(&manifest_content)?;
        self.repository.add(wasm_binary, manifest, force_overwrite).await
    }

    /// Install a program: JIT compile + link, auto-downloads from registry if needed, resolves dependencies.
    async fn install(&mut self, name: &ProgramName) -> Result<Manifest> {
        // Step 0: Check if already installed (early exit)
        if let Some((_, metadata)) = self.installed.get(name) {
            return Ok(metadata.clone());
        }

        // Step 1: Fetch WASM bytes (handles binary_cache -> disk -> registry fallback)
        let wasm_binary = self.repository.fetch_wasm_binary(name).await?;

        // Step 2: Get metadata from index
        let metadata = self.repository.fetch_manifest(name)
            .ok_or_else(|| anyhow!("Metadata not found for program: {}", name))?;

        // Step 3: Install dependencies
        for dep_name in metadata.dependency_names() {
            if !self.is_installed(&dep_name) {
                // Fetch and compile each dependency
                let dep_wasm = self.repository.fetch_wasm_binary(&dep_name).await?;
                let dep_metadata = self.repository.fetch_manifest(&dep_name)
                    .ok_or_else(|| anyhow!("Metadata not found for dependency: {}", dep_name))?;
                let dep_component = compile_wasm_component(&self.wasm_engine, dep_wasm)
                    .await?;
                self.installed.insert(dep_name, (dep_component, dep_metadata));
            }
        }

        // Step 4: JIT compile
        let component = compile_wasm_component(&self.wasm_engine, wasm_binary)
            .await?;

        // Step 5: Track as installed (program.rs now owns compiled programs)
        self.installed.insert(name.clone(), (component, metadata.clone()));

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
    fn new(wasm_engine: WasmEngine, repository: Repository) -> Self {
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
                let result = self.service.get_installed_metadata(&name).cloned();
                let _ = response.send(result);
            }
            Message::Register { wasm_binary, manifest, force_overwrite, response } => {
                let result = self.service.register(wasm_binary, manifest, force_overwrite).await;
                let _ = response.send(result);
            }
            Message::IsRegistered { name, response } => {
                let result = self.service.is_registered(&name);
                let _ = response.send(result);
            }
            Message::IsInstalled { name, response } => {
                let result = self.service.is_installed(&name);
                let _ = response.send(result);
            }
            Message::Install { name, response } => {
                let result = self.service.install(&name).await;
                let _ = response.send(result);
            }
            Message::Uninstall { name, response } => {
                let result = self.service.uninstall(&name);
                let _ = response.send(result);
            }
            Message::GetWasmComponent { name, response } => {
                let result = self.service.get_component(&name);
                let _ = response.send(result);
            }
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compiles WASM bytes to a Component in a blocking thread.
pub async fn compile_wasm_component(engine: &WasmEngine, wasm_binary: Vec<u8>) -> Result<Component> {
    let engine = engine.clone();
    match tokio::task::spawn_blocking(move || Component::from_binary(&engine, &wasm_binary)).await {
        Ok(Ok(component)) => Ok(component),
        Ok(Err(e)) => Err(anyhow!("Failed to compile WASM: {}", e)),
        Err(e) => Err(anyhow!("Compilation task failed: {}", e)),
    }
}
