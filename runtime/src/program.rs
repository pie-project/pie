//! Program Manager Service - Inferlet program caching and loading
//!
//! This module provides a singleton actor for managing program (inferlet) metadata,
//! caching, downloading from registry, and compilation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::LazyLock;

use anyhow::{Result, anyhow, bail};
use tokio::sync::oneshot;
use wasmtime::Engine as WasmEngine;
use wasmtime::component::Component;

use crate::service::{Service, ServiceHandler};

mod manifest;
mod repository;
pub use manifest::Manifest;
pub use repository::Repository;

// =============================================================================
// Public API
// =============================================================================

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

/// Spawns the program manager service.
pub fn spawn(wasm_engine: &WasmEngine, registry_url: String, cache_dir: PathBuf) {
    let mut repository = Repository::new(
        registry_url,
        cache_dir,
    );

    // Scan disk on startup: load existing programs into index
    repository.load_program_cache();

    SERVICE.spawn(|| {
        ProgramService::new(&wasm_engine, repository)
    }).expect("Program manager already spawned");
}

/// Add a program with WASM binary and manifest. Stores in repository + disk (does NOT install).
pub async fn add(
    wasm_binary: Vec<u8>,
    manifest: Manifest,
    force_overwrite: bool,
) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Add {
        wasm_binary,
        manifest,
        force_overwrite,
        response: tx,
    })?;
    rx.await?
}

/// Add a program from registry by name. Downloads and stores in repository + disk (does NOT install).
pub async fn add_from_registry(name: &ProgramName, force_overwrite: bool) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::AddFromRegistry {
        name: name.clone(),
        force_overwrite,
        response: tx,
    })?;
    rx.await?
}

/// Check if a program is registered in repository.
pub async fn is_registered(name: &ProgramName) -> bool {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Exists {
        name: name.clone(),
        response: tx,
    }).ok();
    rx.await.unwrap_or(false)
}

/// Check if a program is installed (JIT compiled and ready to run).
pub async fn is_installed(name: &ProgramName) -> bool {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::IsInstalled {
        name: name.clone(),
        response: tx,
    }).ok();
    rx.await.unwrap_or(false)
}

/// Install a program: JIT compile + link, auto-downloads from registry if needed, resolves dependencies.
pub async fn install(name: &ProgramName) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Install {
        name: name.clone(),
        response: tx,
    })?;
    rx.await?
}

/// Uninstall a program: remove from installed programs (does NOT remove from cache).
pub async fn uninstall(name: &ProgramName) -> bool {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Uninstall {
        name: name.clone(),
        response: tx,
    }).ok();
    rx.await.unwrap_or(false)
}

/// Get program metadata by name.
pub async fn fetch_manifest(name: &ProgramName) -> Option<Manifest> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::GetMetadata {
        name: name.clone(),
        response: tx,
    }).ok();
    rx.await.ok().flatten()
}

/// Get the compiled component for an installed program.
pub async fn get_wasm_component(name: &ProgramName) -> Option<Component> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::GetWasmComponent {
        name: name.clone(),
        response: tx,
    }).ok();
    rx.await.ok().flatten()
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
    /// Parses an inferlet identifier from a `name@major.minor.patch` string.
    ///
    /// The name must contain only alphanumeric characters, hyphens, and underscores.
    /// The version must be valid semver (e.g., `1.0.0`).
    pub fn parse(s: &str) -> Result<Self> {
        static RE: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
            fancy_regex::Regex::new(r"^([a-zA-Z0-9][a-zA-Z0-9_-]*)@(\d+\.\d+\.\d+)$").unwrap()
        });

        let caps = RE.captures(s)?
            .ok_or_else(|| anyhow!(
                "Invalid program identifier '{}': expected 'name@major.minor.patch'", s
            ))?;

        Ok(Self {
            name: caps.get(1).unwrap().as_str().to_string(),
            version: caps.get(2).unwrap().as_str().to_string(),
        })
    }
}

impl std::fmt::Display for ProgramName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.name, self.version)
    }
}
// =============================================================================
// Program Service
// =============================================================================

/// Program service: caching, installation, and loading of inferlet programs.
struct ProgramService {
    wasm_engine: WasmEngine,
    repository: Repository,
    /// Installed (JIT compiled) programs, keyed by program name
    installed: HashMap<ProgramName, Component>,
    /// Programs that were explicitly installed (not pulled as dependencies)
    explicit_installs: std::collections::HashSet<ProgramName>,
}

impl ProgramService {
    fn new(wasm_engine: &WasmEngine, repository: Repository) -> Self {
        ProgramService {
            wasm_engine: wasm_engine.clone(),
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
// ServiceHandler
// =============================================================================

enum Message {
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


impl ServiceHandler for ProgramService {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::GetMetadata { name, response } => {
                let _ = response.send(self.get_manifest(&name));
            }
            Message::Add { wasm_binary, manifest, force_overwrite, response } => {
                let _ = response.send(self.add(wasm_binary, manifest, force_overwrite).await);
            }
            Message::AddFromRegistry { name, force_overwrite, response } => {
                let _ = response.send(self.add_from_registry(&name, force_overwrite).await);
            }
            Message::Exists { name, response } => {
                let _ = response.send(self.is_registered(&name));
            }
            Message::IsInstalled { name, response } => {
                let _ = response.send(self.is_installed(&name));
            }
            Message::Install { name, response } => {
                let _ = response.send(self.install(&name).await);
            }
            Message::Uninstall { name, response } => {
                let _ = response.send(self.uninstall(&name));
            }
            Message::GetWasmComponent { name, response } => {
                let _ = response.send(self.get_component(&name));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_name_parse_validation() {
        // ── Valid inputs ──────────────────────────────────────────────
        let valid = [
            ("text-completion@0.1.0",  "text-completion", "0.1.0"),
            ("my_inferlet@1.2.3",      "my_inferlet",     "1.2.3"),
            ("a@0.0.0",               "a",               "0.0.0"),
            ("X@99.99.99",            "X",               "99.99.99"),
            ("foo-bar_baz@10.20.30",  "foo-bar_baz",     "10.20.30"),
            ("A1-b2_C3@0.0.1",       "A1-b2_C3",        "0.0.1"),
        ];
        for (input, expected_name, expected_version) in valid {
            let p = ProgramName::parse(input)
                .unwrap_or_else(|e| panic!("Expected '{}' to be valid, got: {}", input, e));
            assert_eq!(p.name, expected_name, "name mismatch for '{}'", input);
            assert_eq!(p.version, expected_version, "version mismatch for '{}'", input);
            // Display roundtrip
            assert_eq!(p.to_string(), input, "Display roundtrip failed for '{}'", input);
        }

        // ── Invalid inputs ───────────────────────────────────────────
        let invalid = [
            // Missing version (bare name)
            "text-completion",
            "foo",
            // Missing name
            "@0.1.0",
            // Empty string
            "",
            // No semver
            "foo@latest",
            "foo@v1.0.0",
            "foo@1.0",
            "foo@1",
            "foo@1.0.0.0",
            "foo@abc",
            "foo@1.0.0-beta",
            // Path traversal / unsafe characters
            "../evil@0.1.0",
            "foo/bar@0.1.0",
            "foo\\bar@0.1.0",
            "foo@../0.1.0",
            // Special characters in name
            "foo bar@0.1.0",
            "foo!@0.1.0",
            "foo.bar@0.1.0",
            // Name starting with hyphen or underscore
            "-foo@0.1.0",
            "_foo@0.1.0",
            // Multiple @
            "foo@bar@0.1.0",
            // Whitespace
            " foo@0.1.0",
            "foo@0.1.0 ",
            "foo @0.1.0",
        ];
        for input in invalid {
            assert!(
                ProgramName::parse(input).is_err(),
                "Expected '{}' to be rejected, but it was accepted", input
            );
        }
    }
}
