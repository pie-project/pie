#![allow(unsafe_code)]
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use anyhow::{Context, Result, anyhow};
use tokio::sync::oneshot;
use wasmtime::Engine as WasmEngine;
use wasmtime::component::Component;

use crate::service::{Service, ServiceHandler};

mod identity;
mod language;
mod repository;
pub use identity::{
    Identity, NAME_SECTION, Namespace, Program, ProgramHash, ProgramRef, check_name, identify,
    name_section,
};
pub use language::{Language, kind};
pub use repository::{Listed, Reclaimed, Repository};

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

/// Start the program service over `programs_dir`. Language components come
/// from `languages_dir` (`<language>.wasm`) on first use, or from bytes via
/// [`add_language`]. The built-ins are the system's, under whatever the
/// operator installs of the same name.
pub fn spawn(
    wasm_engine: &WasmEngine,
    programs_dir: PathBuf,
    languages_dir: PathBuf,
    builtins: &[crate::bootstrap::BuiltinProgram],
) -> Result<()> {
    let mut repository = Repository::new(programs_dir);
    for builtin in builtins {
        check_name(builtin.name).context("a built-in inferlet's name")?;
        repository.add_builtin(builtin.name, builtin.component);
    }

    SERVICE
        .spawn(|| ProgramService::new(wasm_engine, repository, languages_dir))
        .expect("Program manager already spawned");
    Ok(())
}

/// Install `binary`, which came as `file`, under its name in `namespace`.
/// It is compiled first: what does not compile is refused, and the name
/// keeps the build it ran.
pub async fn add(namespace: Namespace, file: &str, binary: Vec<u8>) -> Result<Program> {
    let identity = identify(file, &binary)?;
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Add {
        namespace,
        identity: identity.clone(),
        binary,
        response: tx,
    })?;
    rx.await??;
    Ok(identity.program)
}

/// The program `reference` names for `tenant`; `None` looks in the
/// system's alone.
pub async fn resolve(tenant: Option<&str>, reference: &ProgramRef) -> Result<Program> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Resolve {
        tenant: tenant.map(str::to_string),
        reference: reference.clone(),
        response: tx,
    })?;
    rx.await?
}

/// Compile `program`, unless it already is: once per hash, whoever runs
/// it and under whatever name.
pub async fn load(program: &Program) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Load {
        program: program.clone(),
        response: tx,
    })?;
    rx.await?
}

pub async fn get_wasm_component(program: &Program) -> Option<InstalledComponent> {
    let (tx, rx) = oneshot::channel();
    SERVICE
        .send(Message::GetWasmComponent {
            program: program.clone(),
            response: tx,
        })
        .ok();
    rx.await.ok().flatten()
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

/// Hand over `language`'s component already compiled for this runtime's
/// engine (`precompile_component`), alongside its source from
/// `add_language`. The first program in that language loads it instead of
/// compiling the source, which in a browser tab takes seconds; if the engine
/// refuses it (another wasmtime version or engine settings), the source is
/// compiled as before.
pub async fn add_precompiled_language(language: Language, precompiled: Vec<u8>) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::AddPrecompiledLanguage {
        language,
        precompiled,
        response: tx,
    })?;
    rx.await?
}

/// Compile a component for `engine` and serialize it, for
/// `add_precompiled_language`.
pub fn precompile_component(engine: &WasmEngine, wasm_binary: &[u8]) -> Result<Vec<u8>> {
    engine
        .precompile_component(wasm_binary)
        .map_err(|e| anyhow!("precompiling the component: {e}"))
}

/// A script program's source, handed to its language component at launch.
#[derive(Debug)]
pub struct Script {
    /// `name@<short hash>`, for the language component's messages.
    pub name: String,
    /// The file name tracebacks quote.
    pub file: String,
    pub source: Arc<str>,
}

impl Script {
    /// The launch input the language component unwraps: the script and
    /// the caller's input, verbatim, in one JSON envelope.
    pub fn envelope(&self, input: &str) -> String {
        serde_json::json!({
            "__pie_script__": {
                "name": self.name,
                "file": self.file,
                "source": &*self.source,
            },
            "input": input,
        })
        .to_string()
    }
}

struct ProgramService {
    wasm_engine: WasmEngine,
    repository: Repository,
    /// Compiled, by the hash of what was compiled.
    loaded: HashMap<ProgramHash, Loaded>,
    languages_dir: PathBuf,
    /// Handed over as bytes, compiled on first use.
    language_binaries: HashMap<Language, Vec<u8>>,
    /// Handed over already compiled, loaded on first use in place of the
    /// source in `language_binaries`.
    language_precompiled: HashMap<Language, Vec<u8>>,
    /// Compiled once, shared by every script program in that language.
    languages: HashMap<Language, Component>,
}

#[derive(Clone)]
struct Loaded {
    component: Component,
    /// A script's source and language: the component is its language's.
    script: Option<(Language, Arc<str>)>,
}

#[derive(Clone)]
pub struct InstalledComponent {
    pub component: Component,
    /// `Some` for a script program: the component is its language's
    /// language component, and this is what that component runs.
    pub script: Option<Arc<Script>>,
}

impl ProgramService {
    fn new(wasm_engine: &WasmEngine, repository: Repository, languages_dir: PathBuf) -> Self {
        ProgramService {
            wasm_engine: wasm_engine.clone(),
            repository,
            loaded: HashMap::new(),
            languages_dir,
            language_binaries: HashMap::new(),
            language_precompiled: HashMap::new(),
            languages: HashMap::new(),
        }
    }

    fn get_component(&self, program: &Program) -> Option<InstalledComponent> {
        let loaded = self.loaded.get(&program.hash)?;
        let script = loaded.script.as_ref().map(|(language, source)| {
            Arc::new(Script {
                name: program.to_string(),
                file: format!("{}.{}", program.name, language.extension()),
                source: Arc::clone(source),
            })
        });
        Some(InstalledComponent {
            component: loaded.component.clone(),
            script,
        })
    }

    fn add_language(&mut self, language: Language, wasm_binary: Vec<u8>) {
        self.languages.remove(&language);
        self.language_binaries.insert(language, wasm_binary);
    }

    fn add_precompiled_language(&mut self, language: Language, precompiled: Vec<u8>) {
        self.languages.remove(&language);
        self.language_precompiled.insert(language, precompiled);
    }

    /// The compiled language component for `language`: compiled once from the
    /// bytes handed over, else from `<languages_dir>/<language>.wasm`.
    async fn language_component(&mut self, language: Language) -> Result<Component> {
        if let Some(component) = self.languages.get(&language) {
            return Ok(component.clone());
        }
        if let Some(precompiled) = self.language_precompiled.remove(&language) {
            match deserialize_component(&self.wasm_engine, &precompiled) {
                Ok(component) => {
                    self.languages.insert(language, component.clone());
                    return Ok(component);
                }
                Err(e) => tracing::warn!(
                    "the precompiled {language} language component does not load ({e:#}); \
                     compiling its source instead"
                ),
            }
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

    async fn add(
        &mut self,
        namespace: Namespace,
        identity: Identity,
        binary: Vec<u8>,
    ) -> Result<()> {
        self.repository.admit(&namespace, &identity)?;
        let program = &identity.program;
        if !self.loaded.contains_key(&program.hash) {
            let loaded = self
                .compile(program, binary.clone(), identity.language)
                .await?;
            self.loaded.insert(program.hash.clone(), loaded);
        }
        self.repository.add(namespace, &identity, binary).await?;
        self.unload_unnamed();
        Ok(())
    }

    /// Drop what is compiled of builds no name runs any more (removed or
    /// pruned since), here and in the linker. A process running one keeps
    /// its own reference.
    fn unload_unnamed(&mut self) {
        let named = self.repository.named();
        let unnamed: Vec<ProgramHash> = self
            .loaded
            .keys()
            .filter(|hash| !named.contains(*hash))
            .cloned()
            .collect();
        for hash in &unnamed {
            self.loaded.remove(hash);
        }
        super::linker::forget(unnamed);
    }

    async fn load(&mut self, program: &Program) -> Result<()> {
        if self.loaded.contains_key(&program.hash) {
            return Ok(());
        }
        let (binary, language) = self.repository.fetch(&program.hash).await?;
        let loaded = self.compile(program, binary, language).await?;
        self.loaded.insert(program.hash.clone(), loaded);
        Ok(())
    }

    async fn compile(
        &mut self,
        program: &Program,
        binary: Vec<u8>,
        language: Option<Language>,
    ) -> Result<Loaded> {
        Ok(match language {
            None => Loaded {
                component: compile_wasm_component(&self.wasm_engine, binary)
                    .await
                    .with_context(|| format!("compiling {program}"))?,
                script: None,
            },
            Some(language) => {
                let source = String::from_utf8(binary)
                    .map_err(|e| anyhow!("{program} is not UTF-8 {language} source: {e}"))?;
                Loaded {
                    component: self.language_component(language).await?,
                    script: Some((language, source.into())),
                }
            }
        })
    }
}

enum Message {
    Add {
        namespace: Namespace,
        identity: Identity,
        binary: Vec<u8>,
        response: oneshot::Sender<Result<()>>,
    },

    AddLanguage {
        language: Language,
        wasm_binary: Vec<u8>,
        response: oneshot::Sender<Result<()>>,
    },

    AddPrecompiledLanguage {
        language: Language,
        precompiled: Vec<u8>,
        response: oneshot::Sender<Result<()>>,
    },

    Resolve {
        tenant: Option<String>,
        reference: ProgramRef,
        response: oneshot::Sender<Result<Program>>,
    },

    Load {
        program: Program,
        response: oneshot::Sender<Result<()>>,
    },

    GetWasmComponent {
        program: Program,
        response: oneshot::Sender<Option<InstalledComponent>>,
    },
}

impl ServiceHandler for ProgramService {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Add {
                namespace,
                identity,
                binary,
                response,
            } => {
                let _ = response.send(self.add(namespace, identity, binary).await);
            }
            Message::AddLanguage {
                language,
                wasm_binary,
                response,
            } => {
                self.add_language(language, wasm_binary);
                let _ = response.send(Ok(()));
            }
            Message::AddPrecompiledLanguage {
                language,
                precompiled,
                response,
            } => {
                self.add_precompiled_language(language, precompiled);
                let _ = response.send(Ok(()));
            }
            Message::Resolve {
                tenant,
                reference,
                response,
            } => {
                // The disk as it stands: what another process installed or
                // removed is seen. Nothing compiled goes stale; a hash is
                // its bytes.
                let _ = response.send(self.repository.resolve(tenant.as_deref(), &reference));
            }
            Message::Load { program, response } => {
                let _ = response.send(self.load(&program).await);
            }
            Message::GetWasmComponent { program, response } => {
                let _ = response.send(self.get_component(&program));
            }
        }
    }
}

/// A component serialized by `precompile_component`. Wasmtime checks the
/// artifact's header against this engine's version and settings and refuses
/// a mismatch.
fn deserialize_component(engine: &WasmEngine, precompiled: &[u8]) -> Result<Component> {
    if WasmEngine::detect_precompiled(precompiled) != Some(wasmtime::Precompiled::Component) {
        anyhow::bail!("not a component precompiled by wasmtime");
    }
    // SAFETY: deserializing trusts the bytes to be wasmtime's own output, as
    // wasmtime cannot verify the code inside. They come from the host that
    // embeds this runtime (the page, in a browser), which is trusted as much
    // as the runtime itself: it supplies the language component's source too.
    unsafe { Component::deserialize(engine, precompiled) }.map_err(|e| anyhow!("{e}"))
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
