use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use anyhow::{Context, Result, anyhow, bail};
use tokio::sync::oneshot;
use wasmtime::Engine as WasmEngine;
use wasmtime::component::Component;

use crate::service::{Service, ServiceHandler};

mod manifest;
mod repository;
pub use manifest::{Call, Language, Manifest, Package, ParameterType, Runtime};
pub use repository::Repository;

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

/// Start the program service over `programs_dir`. Language components come
/// from `languages_dir` (`<language>.wasm`) on first use, or from bytes via
/// [`add_language`]. Each of `builtins` is registered unless the directory
/// already holds that name and version: a copy installed on disk shadows
/// the built-in one.
pub fn spawn(
    wasm_engine: &WasmEngine,
    programs_dir: PathBuf,
    languages_dir: PathBuf,
    builtins: &[crate::bootstrap::BuiltinProgram],
) -> Result<()> {
    let mut repository = Repository::new(programs_dir);

    repository.refresh();
    for builtin in builtins {
        let manifest =
            Manifest::parse(builtin.manifest).context("a built-in inferlet's manifest")?;
        repository.add_builtin(manifest, builtin.component);
    }

    SERVICE
        .spawn(|| ProgramService::new(wasm_engine, repository, languages_dir))
        .expect("Program manager already spawned");
    Ok(())
}

pub async fn add(wasm_binary: Vec<u8>, manifest: Manifest, force_overwrite: bool) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Add {
        wasm_binary,
        manifest: Box::new(manifest),
        force_overwrite,
        response: tx,
    })?;
    rx.await?
}

/// A language component from bytes, for a host with no languages directory
/// to read (the browser).
pub async fn add_language(language: Language, wasm_binary: Vec<u8>) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::AddLanguage {
        language,
        wasm_binary,
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

/// The newest installed version of a program named `name`.
pub async fn newest(name: &str) -> Option<ProgramName> {
    let (tx, rx) = oneshot::channel();
    SERVICE
        .send(Message::Newest {
            name: name.to_string(),
            response: tx,
        })
        .ok();
    rx.await.ok().flatten()
}

pub async fn install(name: &ProgramName) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Install {
        name: name.clone(),
        response: tx,
    })?;
    rx.await?
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

/// A script program's source, handed to its language component at launch.
#[derive(Debug)]
pub struct Script {
    /// `name@version`, for the language component's messages.
    pub name: String,
    /// The file name tracebacks quote.
    pub file: String,
    pub source: String,
    /// The function the language component calls with the input.
    pub entry: String,
    /// How that function takes it.
    pub call: Call,
}

impl Script {
    /// The launch input the language component unwraps: the script and
    /// the caller's input, verbatim, in one JSON envelope.
    pub fn envelope(&self, input: &str) -> String {
        serde_json::json!({
            "__pie_script__": {
                "name": self.name,
                "file": self.file,
                "source": self.source,
                "entry": self.entry,
                "call": self.call.name(),
            },
            "input": input,
        })
        .to_string()
    }
}

struct ProgramService {
    wasm_engine: WasmEngine,
    repository: Repository,
    installed: HashMap<ProgramName, InstalledProgram>,
    explicit_installs: HashSet<ProgramName>,
    generation: u64,
    languages_dir: PathBuf,
    /// Handed over as bytes, compiled on first use.
    language_binaries: HashMap<Language, Vec<u8>>,
    /// Compiled once, shared by every script program in that language.
    languages: HashMap<Language, Component>,
}

#[derive(Clone)]
struct InstalledProgram {
    component: Component,
    script: Option<Arc<Script>>,
}

#[derive(Clone)]
pub struct InstalledComponent {
    pub component: Component,
    pub generation: u64,
    /// `Some` for a script program: the component is its language's
    /// language component, and this is what that component runs.
    pub script: Option<Arc<Script>>,
}

impl ProgramService {
    fn new(wasm_engine: &WasmEngine, repository: Repository, languages_dir: PathBuf) -> Self {
        ProgramService {
            wasm_engine: wasm_engine.clone(),
            repository,
            installed: HashMap::new(),
            explicit_installs: HashSet::new(),
            generation: 0,
            languages_dir,
            language_binaries: HashMap::new(),
            languages: HashMap::new(),
        }
    }

    fn get_component(&self, name: &ProgramName) -> Option<InstalledComponent> {
        self.installed.get(name).map(|p| InstalledComponent {
            component: p.component.clone(),
            generation: self.generation,
            script: p.script.clone(),
        })
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

    fn add_language(&mut self, language: Language, wasm_binary: Vec<u8>) {
        self.languages.remove(&language);
        self.language_binaries.insert(language, wasm_binary);
    }

    /// The compiled language component for `language`: compiled once from the
    /// bytes handed over, else from `<languages_dir>/<language>.wasm`.
    async fn language_component(&mut self, language: Language) -> Result<Component> {
        if let Some(component) = self.languages.get(&language) {
            return Ok(component.clone());
        }
        let path = self.languages_dir.join(language.component_file());
        let binary = match self.language_binaries.remove(&language) {
            Some(binary) => binary,
            #[cfg(not(target_arch = "wasm32"))]
            None => tokio::fs::read(&path).await.map_err(|e| {
                anyhow!(
                    "no {language} language component at {}: {e}; `pie language install \
                     pie-language-{language}.tar.gz` puts one there",
                    path.display()
                )
            })?,
            #[cfg(target_arch = "wasm32")]
            None => bail!(
                "no {language} language component: this host reads no files, so the page \
                 must add it from bytes before a {language} program runs"
            ),
        };
        let component = compile_wasm_component(&self.wasm_engine, binary)
            .await
            .with_context(|| {
                format!(
                    "compiling the {language} language component {}",
                    path.display()
                )
            })?;
        self.languages.insert(language, component.clone());
        Ok(component)
    }

    async fn install(&mut self, name: &ProgramName) -> Result<()> {
        if self.installed.contains_key(name) {
            self.explicit_installs.insert(name.clone());
            return Ok(());
        }

        let manifest = self
            .repository
            .fetch_manifest(name)
            .ok_or_else(|| anyhow!("{}", self.repository.not_installed(name)))?;

        let dependencies = self.resolve_dependencies(name).await?;

        for dep_name in &dependencies {
            if !self.installed.contains_key(dep_name) {
                let dep_manifest = self
                    .repository
                    .fetch_manifest(dep_name)
                    .ok_or_else(|| anyhow!("{}", self.repository.not_installed(dep_name)))?;
                if let Some(language) = dep_manifest.language() {
                    bail!(
                        "{name} depends on {dep_name}, which is a {language} script; only a \
                         component can be a dependency"
                    );
                }
                let dep_wasm = self.repository.fetch_binary(dep_name).await?;
                let dep_component = compile_wasm_component(&self.wasm_engine, dep_wasm).await?;
                self.installed.insert(
                    dep_name.clone(),
                    InstalledProgram {
                        component: dep_component,
                        script: None,
                    },
                );
            }
        }

        let binary = self.repository.fetch_binary(name).await?;

        let (component, script) = match manifest.language() {
            None => (
                compile_wasm_component(&self.wasm_engine, binary).await?,
                None,
            ),
            Some(language) => {
                let source = String::from_utf8(binary)
                    .map_err(|e| anyhow!("{name} is not UTF-8 {language} source: {e}"))?;
                let script = Script {
                    name: name.to_string(),
                    file: manifest.script_file(),
                    source,
                    entry: manifest.entry().to_string(),
                    call: manifest.call(),
                };
                (
                    self.language_component(language).await?,
                    Some(Arc::new(script)),
                )
            }
        };

        self.installed
            .insert(name.clone(), InstalledProgram { component, script });
        self.explicit_installs.insert(name.clone());
        self.bump_generation();

        Ok(())
    }

    async fn resolve_dependencies(&mut self, name: &ProgramName) -> Result<Vec<ProgramName>> {
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
                bail!(
                    "{}, and {name} depends on it",
                    self.repository.not_installed(&current)
                );
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
        manifest: Box<Manifest>,
        force_overwrite: bool,
        response: oneshot::Sender<Result<()>>,
    },

    AddLanguage {
        language: Language,
        wasm_binary: Vec<u8>,
        response: oneshot::Sender<Result<()>>,
    },

    Exists {
        name: ProgramName,
        response: oneshot::Sender<bool>,
    },

    Newest {
        name: String,
        response: oneshot::Sender<Option<ProgramName>>,
    },

    Install {
        name: ProgramName,
        response: oneshot::Sender<Result<()>>,
    },

    GetWasmComponent {
        name: ProgramName,
        response: oneshot::Sender<Option<InstalledComponent>>,
    },
}

impl ServiceHandler for ProgramService {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        // An install, removal or replacement on disk since the last message
        // is seen now; a replaced program's compiled form is dropped.
        for stale in self.repository.refresh() {
            self.uninstall(&stale);
        }
        match msg {
            Message::GetMetadata { name, response } => {
                let _ = response.send(self.repository.fetch_manifest(&name));
            }
            Message::Add {
                wasm_binary,
                manifest,
                force_overwrite,
                response,
            } => {
                let _ = response.send(self.add(wasm_binary, *manifest, force_overwrite).await);
            }
            Message::AddLanguage {
                language,
                wasm_binary,
                response,
            } => {
                self.add_language(language, wasm_binary);
                let _ = response.send(Ok(()));
            }
            Message::Exists { name, response } => {
                let _ = response.send(self.repository.exists(&name));
            }
            Message::Newest { name, response } => {
                let _ = response.send(self.repository.newest(&name));
            }
            Message::Install { name, response } => {
                let _ = response.send(self.install(&name).await);
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
    match crate::rt::spawn_blocking(move || Component::from_binary(&engine, &wasm_binary)).await {
        Ok(Ok(component)) => Ok(component),
        Ok(Err(e)) => Err(anyhow!("Failed to compile WASM: {}", e)),
        Err(e) => Err(anyhow!("Compilation task failed: {}", e)),
    }
}
