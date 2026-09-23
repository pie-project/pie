use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use anyhow::{Context, Result, anyhow};
use tokio::sync::oneshot;
use wasmtime::Engine as WasmEngine;
use wasmtime::component::Component;

use crate::service::{Service, ServiceHandler};

mod language;
mod repository;
pub use language::{Language, artifact_extension};
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
        let name = ProgramName::parse(&format!("{}@{}", builtin.name, builtin.version))
            .context("a built-in inferlet's name")?;
        repository.add_builtin(name, builtin.component);
    }

    SERVICE
        .spawn(|| ProgramService::new(wasm_engine, repository, languages_dir))
        .expect("Program manager already spawned");
    Ok(())
}

pub async fn add(
    binary: Vec<u8>,
    file: &str,
    version: Option<&str>,
    force_overwrite: bool,
) -> Result<ProgramName> {
    let identity = identify(file, version, &binary)?;
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Add {
        binary,
        identity: identity.clone(),
        force_overwrite,
        response: tx,
    })?;
    rx.await??;
    Ok(identity.name)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    pub name: ProgramName,
    pub language: Option<Language>,
}

pub const PACKAGE_SECTION: &str = "pie.package";

pub fn identify(file: &str, version: Option<&str>, bytes: &[u8]) -> Result<Identity> {
    let path = std::path::Path::new(file);
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let language = match extension {
        "wasm" => None,
        other => Some(Language::from_extension(other).ok_or_else(|| {
            anyhow!(
                "{file}: not a program pie can install; a program is a `.wasm` component or a \
                 `.py` / `.js` script"
            )
        })?),
    };
    if language.is_none()
        && let Some(package) = package_section(bytes)
    {
        let name = ProgramName::parse(&package)
            .with_context(|| format!("{file}: its {PACKAGE_SECTION} section"))?;
        if let Some(version) = version
            && version != name.version
        {
            anyhow::bail!(
                "{file} names itself {name}; a version of {version} would be the package's \
                 Cargo.toml saying so"
            );
        }
        return Ok(Identity { name, language });
    }
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("{file}: no file name to name the program by"))?;
    let name = stem.replace('_', "-");
    let version = match version {
        Some(version) => version.to_string(),
        None => language
            .and_then(|language| declared_version(bytes, language))
            .unwrap_or_else(|| hashed_version(bytes)),
    };
    let name = ProgramName::parse(&format!("{name}@{version}"))
        .with_context(|| format!("{file} cannot name a program"))?;
    Ok(Identity { name, language })
}

pub fn package_section(bytes: &[u8]) -> Option<String> {
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        let Ok(wasmparser::Payload::CustomSection(section)) = payload else {
            continue;
        };
        if section.name() == PACKAGE_SECTION {
            return std::str::from_utf8(section.data())
                .ok()
                .map(|s| s.trim().to_string());
        }
    }
    None
}

fn declared_version(source: &[u8], language: Language) -> Option<String> {
    static PYTHON: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
        fancy_regex::Regex::new(r#"(?m)^__version__\s*=\s*["'](\d+\.\d+\.\d+)["']"#).unwrap()
    });
    static JAVASCRIPT: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
        fancy_regex::Regex::new(r#"(?m)^export\s+const\s+version\s*=\s*["'](\d+\.\d+\.\d+)["']"#)
            .unwrap()
    });
    let source = std::str::from_utf8(source).ok()?;
    let re = match language {
        Language::Python => &*PYTHON,
        Language::JavaScript => &*JAVASCRIPT,
    };
    re.captures(source)
        .ok()
        .flatten()
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
}

pub fn hashed_version(bytes: &[u8]) -> String {
    let digest = blake3::hash(bytes);
    let d = digest.as_bytes();
    format!(
        "0.{}.{}",
        u16::from_be_bytes([d[0], d[1]]),
        u16::from_be_bytes([d[2], d[3]])
    )
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
        self.bump_generation();
        true
    }

    fn bump_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    async fn add(
        &mut self,
        binary: Vec<u8>,
        identity: Identity,
        force_overwrite: bool,
    ) -> Result<()> {
        let stored = self
            .repository
            .add(
                binary,
                identity.name.clone(),
                identity.language,
                force_overwrite,
            )
            .await?;
        if stored {
            self.uninstall(&identity.name);
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
            None => anyhow::bail!(
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
            return Ok(());
        }

        let language = self
            .repository
            .language(name)
            .ok_or_else(|| anyhow!("{}", self.repository.not_installed(name)))?;

        let binary = self.repository.fetch_binary(name).await?;

        let (component, script) = match language {
            None => (
                compile_wasm_component(&self.wasm_engine, binary).await?,
                None,
            ),
            Some(language) => {
                let source = String::from_utf8(binary)
                    .map_err(|e| anyhow!("{name} is not UTF-8 {language} source: {e}"))?;
                let script = Script {
                    name: name.to_string(),
                    file: format!("{}.{}", name.name, language.extension()),
                    source,
                };
                (
                    self.language_component(language).await?,
                    Some(Arc::new(script)),
                )
            }
        };

        self.installed
            .insert(name.clone(), InstalledProgram { component, script });
        self.bump_generation();

        Ok(())
    }
}

enum Message {
    Add {
        binary: Vec<u8>,
        identity: Identity,
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
            Message::Add {
                binary,
                identity,
                force_overwrite,
                response,
            } => {
                let _ = response.send(self.add(binary, identity, force_overwrite).await);
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
