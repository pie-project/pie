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
pub use manifest::{Manifest, ParameterType};
pub use repository::Repository;

use super::python::runtime as py_runtime;
use super::python::snapshot;

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

pub fn spawn(wasm_engine: &WasmEngine, registry_url: String, programs_dir: PathBuf) {
    let mut repository = Repository::new(registry_url, programs_dir);

    repository.load_program_cache();

    SERVICE
        .spawn(|| ProgramService::new(wasm_engine, repository))
        .expect("Program manager already spawned");
}

pub async fn add(wasm_binary: Vec<u8>, manifest: Manifest, force_overwrite: bool) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Add {
        wasm_binary,
        manifest,
        force_overwrite,
        response: tx,
    })?;
    rx.await?
}

pub async fn add_from_registry(name: &ProgramName, force_overwrite: bool) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::AddFromRegistry {
        name: name.clone(),
        force_overwrite,
        response: tx,
    })?;
    rx.await?
}

pub async fn is_registered(name: &ProgramName) -> bool {
    let (tx, rx) = oneshot::channel();
    SERVICE
        .send(Message::Exists {
            name: name.clone(),
            response: tx,
        })
        .ok();
    rx.await.unwrap_or(false)
}

pub async fn is_installed(name: &ProgramName) -> bool {
    let (tx, rx) = oneshot::channel();
    SERVICE
        .send(Message::IsInstalled {
            name: name.clone(),
            response: tx,
        })
        .ok();
    rx.await.unwrap_or(false)
}

pub async fn install(name: &ProgramName) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Install {
        name: name.clone(),
        response: tx,
    })?;
    rx.await?
}

pub async fn uninstall(name: &ProgramName) -> bool {
    let (tx, rx) = oneshot::channel();
    SERVICE
        .send(Message::Uninstall {
            name: name.clone(),
            response: tx,
        })
        .ok();
    rx.await.unwrap_or(false)
}

pub async fn fetch_manifest(name: &ProgramName) -> Option<Manifest> {
    let (tx, rx) = oneshot::channel();
    SERVICE
        .send(Message::GetMetadata {
            name: name.clone(),
            response: tx,
        })
        .ok();
    rx.await.ok().flatten()
}

pub async fn get_wasm_component(name: &ProgramName) -> Option<InstalledComponent> {
    let (tx, rx) = oneshot::channel();
    SERVICE
        .send(Message::GetWasmComponent {
            name: name.clone(),
            response: tx,
        })
        .ok();
    rx.await.ok().flatten()
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProgramName {
    pub name: String,
    pub version: String,
}

impl ProgramName {
    pub fn parse(s: &str) -> Result<Self> {
        static RE: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
            fancy_regex::Regex::new(r"^([a-zA-Z0-9][a-zA-Z0-9_-]*)@(\d+\.\d+\.\d+)$").unwrap()
        });

        let caps = RE.captures(s)?.ok_or_else(|| {
            anyhow!(
                "Invalid program identifier '{}': expected 'name@major.minor.patch'",
                s
            )
        })?;

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

struct ProgramService {
    wasm_engine: WasmEngine,
    repository: Repository,
    installed: HashMap<ProgramName, InstalledProgram>,
    explicit_installs: std::collections::HashSet<ProgramName>,
    generation: u64,
}

#[derive(Clone)]
struct InstalledProgram {
    component: Component,
    snapshotted: bool,
    python_runtime: Option<String>,
}

#[derive(Clone)]
pub struct InstalledComponent {
    pub component: Component,
    pub generation: u64,
    pub snapshotted: bool,
    pub python_runtime: Option<String>,
}

impl ProgramService {
    fn new(wasm_engine: &WasmEngine, repository: Repository) -> Self {
        ProgramService {
            wasm_engine: wasm_engine.clone(),
            repository,
            installed: HashMap::new(),
            explicit_installs: std::collections::HashSet::new(),
            generation: 0,
        }
    }

    fn is_installed(&self, name: &ProgramName) -> bool {
        self.installed.contains_key(name)
    }

    fn get_manifest(&self, name: &ProgramName) -> Option<Manifest> {
        self.repository.fetch_manifest(name)
    }

    fn get_component(&self, name: &ProgramName) -> Option<InstalledComponent> {
        self.installed.get(name).map(|p| InstalledComponent {
            component: p.component.clone(),
            generation: self.generation,
            snapshotted: p.snapshotted,
            python_runtime: p.python_runtime.clone(),
        })
    }

    fn is_registered(&self, name: &ProgramName) -> bool {
        self.repository.exists(name)
    }

    fn uninstall(&mut self, name: &ProgramName) -> bool {
        if self.installed.remove(name).is_none() {
            return false;
        }
        super::linker::invalidate(name);
        self.explicit_installs.remove(name);

        loop {
            let orphans = self.find_orphaned_dependencies();
            if orphans.is_empty() {
                break;
            }
            for orphan in orphans {
                self.installed.remove(&orphan);
                super::linker::invalidate(&orphan);
            }
        }

        self.bump_generation();
        true
    }

    fn bump_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    async fn add(
        &mut self,
        wasm_binary: Vec<u8>,
        manifest: Manifest,
        force_overwrite: bool,
    ) -> Result<()> {
        let program_name = manifest.program_name();
        self.repository
            .add(wasm_binary, manifest, force_overwrite)
            .await?;
        if force_overwrite {
            self.uninstall(&program_name);
        }
        Ok(())
    }

    async fn add_from_registry(&mut self, name: &ProgramName, force_overwrite: bool) -> Result<()> {
        self.repository
            .add_from_registry(name, force_overwrite)
            .await?;
        if force_overwrite {
            self.uninstall(name);
        }
        Ok(())
    }

    async fn install(&mut self, name: &ProgramName) -> Result<()> {
        if self.installed.contains_key(name) {
            self.explicit_installs.insert(name.clone());
            return Ok(());
        }

        if !self.repository.exists(name) {
            self.repository.add_from_registry(name, false).await?;
        }

        let dependencies = self.resolve_dependencies(name).await?;

        for dep_name in &dependencies {
            if !self.installed.contains_key(dep_name) {
                let dep_wasm = self.repository.fetch_wasm_binary(dep_name).await?;
                let dep_component = compile_wasm_component(&self.wasm_engine, dep_wasm).await?;
                let dep_python_runtime = self
                    .repository
                    .fetch_manifest(dep_name)
                    .and_then(|m| m.python_runtime().map(str::to_string));
                self.installed.insert(
                    dep_name.clone(),
                    InstalledProgram {
                        component: dep_component,
                        snapshotted: false,
                        python_runtime: dep_python_runtime,
                    },
                );
            }
        }

        let wasm_binary = self.repository.fetch_wasm_binary(name).await?;

        let python_runtime = self
            .repository
            .fetch_manifest(name)
            .and_then(|m| m.python_runtime().map(str::to_string));

        let should_snapshot = python_runtime.is_some()
            && py_runtime::is_available()
            && py_runtime::is_snapshot_enabled();

        let (component, snapshotted) = if should_snapshot {
            let manifest = self
                .repository
                .fetch_manifest(name)
                .ok_or_else(|| anyhow!("Manifest disappeared mid-install: {}", name))?;
            let dep_components: Vec<Component> = manifest
                .dependency_names()
                .into_iter()
                .filter_map(|n| self.installed.get(&n).map(|p| p.component.clone()))
                .collect();

            match snapshot::snapshot_from_bytes(&self.wasm_engine, &wasm_binary, dep_components)
                .await
            {
                Ok(snap_bytes) => match compile_wasm_component(&self.wasm_engine, snap_bytes).await
                {
                    Ok(c) => (c, true),
                    Err(e) => {
                        tracing::warn!(
                            "Compile of snapshotted component failed for {}, falling back to non-snapshotted: {e:#}",
                            name,
                        );
                        (
                            compile_wasm_component(&self.wasm_engine, wasm_binary).await?,
                            false,
                        )
                    }
                },
                Err(e) => {
                    tracing::warn!(
                        "Snapshot pipeline failed for {}, falling back to non-snapshotted: {e:#}",
                        name,
                    );
                    (
                        compile_wasm_component(&self.wasm_engine, wasm_binary).await?,
                        false,
                    )
                }
            }
        } else {
            (
                compile_wasm_component(&self.wasm_engine, wasm_binary).await?,
                false,
            )
        };

        self.installed.insert(
            name.clone(),
            InstalledProgram {
                component,
                snapshotted,
                python_runtime,
            },
        );
        self.explicit_installs.insert(name.clone());
        self.bump_generation();

        Ok(())
    }

    async fn resolve_dependencies(&mut self, name: &ProgramName) -> Result<Vec<ProgramName>> {
        use std::collections::HashSet;

        let mut resolved: Vec<ProgramName> = Vec::new();
        let mut visited: HashSet<ProgramName> = HashSet::new();
        let mut stack: Vec<(ProgramName, bool)> = vec![(name.clone(), false)];

        while let Some((current, children_processed)) = stack.pop() {
            if children_processed {
                resolved.push(current);
                continue;
            }

            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            if !self.repository.exists(&current) {
                self.repository.add_from_registry(&current, false).await?;
            }

            let manifest = self
                .repository
                .fetch_manifest(&current)
                .ok_or_else(|| anyhow!("Manifest not found for program: {}", current))?;

            stack.push((current, true));

            for dep_name in manifest.dependency_names() {
                if !visited.contains(&dep_name) {
                    stack.push((dep_name, false));
                }
            }
        }

        resolved.retain(|dep| dep != name);

        Ok(resolved)
    }

    fn find_orphaned_dependencies(&self) -> Vec<ProgramName> {
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
                !self.explicit_installs.contains(*name)
                    && reverse_deps
                        .get(*name)
                        .is_none_or(|dependents| dependents.is_empty())
            })
            .cloned()
            .collect()
    }
}

enum Message {
    GetMetadata {
        name: ProgramName,
        response: oneshot::Sender<Option<Manifest>>,
    },

    Add {
        wasm_binary: Vec<u8>,
        manifest: Manifest,
        force_overwrite: bool,
        response: oneshot::Sender<Result<()>>,
    },

    AddFromRegistry {
        name: ProgramName,
        force_overwrite: bool,
        response: oneshot::Sender<Result<()>>,
    },

    Exists {
        name: ProgramName,
        response: oneshot::Sender<bool>,
    },

    IsInstalled {
        name: ProgramName,
        response: oneshot::Sender<bool>,
    },

    Install {
        name: ProgramName,
        response: oneshot::Sender<Result<()>>,
    },

    Uninstall {
        name: ProgramName,
        response: oneshot::Sender<bool>,
    },

    GetWasmComponent {
        name: ProgramName,
        response: oneshot::Sender<Option<InstalledComponent>>,
    },
}

impl ServiceHandler for ProgramService {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::GetMetadata { name, response } => {
                let _ = response.send(self.get_manifest(&name));
            }
            Message::Add {
                wasm_binary,
                manifest,
                force_overwrite,
                response,
            } => {
                let _ = response.send(self.add(wasm_binary, manifest, force_overwrite).await);
            }
            Message::AddFromRegistry {
                name,
                force_overwrite,
                response,
            } => {
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

pub async fn compile_wasm_component(
    engine: &WasmEngine,
    wasm_binary: Vec<u8>,
) -> Result<Component> {
    let engine = engine.clone();
    match tokio::task::spawn_blocking(move || Component::from_binary(&engine, &wasm_binary)).await {
        Ok(Ok(component)) => Ok(component),
        Ok(Err(e)) => Err(anyhow!("Failed to compile WASM: {}", e)),
        Err(e) => Err(anyhow!("Compilation task failed: {}", e)),
    }
}
