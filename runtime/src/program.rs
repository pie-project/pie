//! Program Manager Service - Inferlet program caching and loading
//!
//! This module provides a singleton actor for managing program (inferlet) metadata,
//! caching, downloading from registry, and compilation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::LazyLock;

use anyhow::{Result, anyhow};
use tokio::sync::oneshot;
use wasmtime::Engine as WasmEngine;
use wasmtime::component::Component;

use crate::service::{Service, ServiceHandler};

mod manifest;
mod repository;
pub use manifest::Manifest;
pub use repository::Repository;

// =============================================================================
// Program Actor
// =============================================================================

/// Global singleton Program Manager actor.
static ACTOR: LazyLock<Service<Message>> = LazyLock::new(Service::new);

/// Spawns the Program Manager actor with configuration.
pub fn spawn(config: ProgramManagerConfig) {
    let mut repository = Repository::new(
        config.registry_url.clone(),
        config.cache_dir.clone(),
    );

    // Scan disk on startup: load existing programs into index
    repository.load_program_cache();

    ACTOR.spawn(|| {
        ProgramManagerActor::new(config.wasm_engine, repository)
    }).expect("Program manager already spawned");
}

/// Sends a message to the Program Manager actor.
pub fn send(msg: Message) -> anyhow::Result<()> {
    ACTOR.send(msg)
}

/// Check if the program manager actor is spawned.
pub fn is_spawned() -> bool {
    ACTOR.is_spawned()
}

/// Add a program with WASM binary and manifest. Stores in repository + disk (does NOT install).
pub async fn add(
    wasm_binary: Vec<u8>,
    manifest: Manifest,
    force_overwrite: bool,
) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    Message::Add {
        wasm_binary,
        manifest,
        force_overwrite,
        response: tx,
    }
    .send()
    .map_err(|_| anyhow!("Program manager not running"))?;
    rx.await.map_err(|_| anyhow!("Program manager did not respond"))?
}

/// Add a program from registry by name. Downloads and stores in repository + disk (does NOT install).
pub async fn add_from_registry(name: &ProgramName, force_overwrite: bool) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    Message::AddFromRegistry {
        name: name.clone(),
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
    let _ = Message::Exists {
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
pub async fn install(name: &ProgramName) -> Result<()> {
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

    /// Add a program with WASM binary and manifest: store in repository + disk (does NOT install)
    Add {
        wasm_binary: Vec<u8>,
        manifest: Manifest,
        force_overwrite: bool,
        response: oneshot::Sender<Result<()>>,
    },

    /// Add a program from registry by name: download and store in repository + disk (does NOT install)
    AddFromRegistry {
        name: ProgramName,
        force_overwrite: bool,
        response: oneshot::Sender<Result<()>>,
    },

    /// Check if a program exists in repository
    Exists {
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
        response: oneshot::Sender<Result<()>>,
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
            Message::Add { force_overwrite, .. } => write!(f, "Add {{ force_overwrite: {} }}", force_overwrite),
            Message::AddFromRegistry { name, force_overwrite, .. } => write!(f, "AddFromRegistry {{ name: {:?}, force_overwrite: {} }}", name, force_overwrite),
            Message::Exists { name, .. } => write!(f, "Exists {{ name: {:?} }}", name),
            Message::IsInstalled { name, .. } => write!(f, "IsInstalled {{ name: {:?} }}", name),
            Message::Install { name, .. } => write!(f, "Install {{ name: {:?} }}", name),
            Message::Uninstall { name, .. } => write!(f, "Uninstall {{ name: {:?} }}", name),
            Message::GetWasmComponent { name, .. } => write!(f, "GetWasmComponent {{ name: {:?} }}", name),
        }
    }
}

impl Message {
    pub fn send(self) -> anyhow::Result<()> {
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
    installed: HashMap<ProgramName, Component>,
    /// Programs that were explicitly installed (not pulled as dependencies)
    explicit_installs: std::collections::HashSet<ProgramName>,
}

impl ProgramManager {
    fn new(wasm_engine: WasmEngine, repository: Repository) -> Self {
        ProgramManager {
            wasm_engine,
            repository,
            installed: HashMap::new(),
            explicit_installs: std::collections::HashSet::new(),
        }
    }

    fn is_installed(&self, name: &ProgramName) -> bool {
        self.installed.contains_key(name)
    }

    fn get_manifest(&self, name: &ProgramName) -> Option<Manifest> {
        self.repository.fetch_manifest(name)
    }

    fn get_component(&self, name: &ProgramName) -> Option<Component> {
        self.installed.get(name).cloned()
    }

    fn is_registered(&self, name: &ProgramName) -> bool {
        self.repository.exists(name)
    }

    /// Uninstall a program and cascade remove orphaned dependencies.
    fn uninstall(&mut self, name: &ProgramName) -> bool {
        if self.installed.remove(name).is_none() {
            return false;
        }
        self.explicit_installs.remove(name);

        // Cascade: find and remove orphaned dependencies
        loop {
            let orphans = self.find_orphaned_dependencies();
            if orphans.is_empty() {
                break;
            }
            for orphan in orphans {
                self.installed.remove(&orphan);
            }
        }

        true
    }


    /// Add a program with WASM binary and manifest: store in repository + disk (does NOT install).
    async fn add(
        &mut self,
        wasm_binary: Vec<u8>,
        manifest: Manifest,
        force_overwrite: bool,
    ) -> Result<()> {
        self.repository.add(wasm_binary, manifest, force_overwrite).await
    }

    /// Add a program from registry by name: download and store in repository + disk (does NOT install).
    async fn add_from_registry(&mut self, name: &ProgramName, force_overwrite: bool) -> Result<()> {
        self.repository.add_from_registry(name, force_overwrite).await
    }

    /// Install a program: JIT compile + link, resolves transitive dependencies.
    async fn install(&mut self, name: &ProgramName) -> Result<()> {
        // Step 0: Check if already installed (mark as explicit and exit)
        if self.installed.contains_key(name) {
            self.explicit_installs.insert(name.clone());
            return Ok(());
        }

        // Step 1: Ensure program is in repository (downloads from registry if needed)
        if !self.repository.exists(name) {
            self.repository.add_from_registry(name, false).await?;
        }

        // Step 2: Resolve all transitive dependencies (flattened, deduplicated, topological order)
        let dependencies = self.resolve_dependencies(name).await?;

        // Step 3: Install each dependency in order
        for dep_name in dependencies {
            if !self.installed.contains_key(&dep_name) {
                let dep_wasm = self.repository.fetch_wasm_binary(&dep_name).await?;
                let dep_component = compile_wasm_component(&self.wasm_engine, dep_wasm).await?;
                self.installed.insert(dep_name, dep_component);
            }
        }

        // Step 4: JIT compile the main program
        let wasm_binary = self.repository.fetch_wasm_binary(name).await?;
        let component = compile_wasm_component(&self.wasm_engine, wasm_binary).await?;

        // Step 5: Track as installed and mark as explicitly installed
        self.installed.insert(name.clone(), component);
        self.explicit_installs.insert(name.clone());

        Ok(())
    }

    /// Resolve transitive dependencies iteratively and return flattened, deduplicated list.
    /// Dependencies are returned in topological order (dependencies before dependents).
    async fn resolve_dependencies(&mut self, name: &ProgramName) -> Result<Vec<ProgramName>> {
        use std::collections::HashSet;

        let mut resolved: Vec<ProgramName> = Vec::new();
        let mut visited: HashSet<ProgramName> = HashSet::new();
        // Stack entries: (program_name, children_processed)
        let mut stack: Vec<(ProgramName, bool)> = vec![(name.clone(), false)];

        while let Some((current, children_processed)) = stack.pop() {
            if children_processed {
                // Second visit: all children processed, add to resolved
                resolved.push(current);
                continue;
            }

            // Skip if already visited (handles cycles and duplicates)
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            // Ensure program is in repository (downloads from registry if needed)
            if !self.repository.exists(&current) {
                self.repository.add_from_registry(&current, false).await?;
            }

            // Get manifest to find direct dependencies
            let manifest = self.repository.fetch_manifest(&current)
                .ok_or_else(|| anyhow!("Manifest not found for program: {}", current))?;

            // Push current back with children_processed=true (for post-order)
            stack.push((current, true));

            // Push children (dependencies) to process first
            for dep_name in manifest.dependency_names() {
                if !visited.contains(&dep_name) {
                    stack.push((dep_name, false));
                }
            }
        }

        // Remove the root program itself from the dependency list
        resolved.retain(|dep| dep != name);

        Ok(resolved)
    }

    /// Find dependencies that are no longer needed:
    /// - Not explicitly installed
    /// - No other installed program depends on them
    fn find_orphaned_dependencies(&self) -> Vec<ProgramName> {
        // Build reverse dependency map: program -> installed programs that depend on it
        let mut reverse_deps: HashMap<ProgramName, Vec<ProgramName>> = HashMap::new();
        for name in self.installed.keys() {
            if let Some(manifest) = self.repository.fetch_manifest(name) {
                for dep in manifest.dependency_names() {
                    reverse_deps.entry(dep).or_default().push(name.clone());
                }
            }
        }

        self.installed
            .keys()
            .filter(|name| {
                // Not explicitly installed
                !self.explicit_installs.contains(*name) &&
                // No other installed program depends on it
                reverse_deps.get(*name).map_or(true, |dependents| dependents.is_empty())
            })
            .cloned()
            .collect()
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

impl ServiceHandler for ProgramManagerActor {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::GetMetadata { name, response } => {
                let result = self.service.get_manifest(&name);
                let _ = response.send(result);
            }
            Message::Add { wasm_binary, manifest, force_overwrite, response } => {
                let result = self.service.add(wasm_binary, manifest, force_overwrite).await;
                let _ = response.send(result);
            }
            Message::AddFromRegistry { name, force_overwrite, response } => {
                let result = self.service.add_from_registry(&name, force_overwrite).await;
                let _ = response.send(result);
            }
            Message::Exists { name, response } => {
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
