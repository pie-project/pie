use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};

use runtime::inferlet::program::{
    Language, Manifest, Package, ParameterType, ProgramName, Repository, Runtime,
};

use crate::ui::{self, Align, Answer, Mark, Palette, Row, Table};

#[derive(Subcommand, Debug)]
pub enum InferletCmd {
    List,

    Info(TargetArgs),

    Install(InstallArgs),

    Remove(TargetArgs),
}

#[derive(Args, Debug)]
pub struct TargetArgs {
    pub inferlet: String,
}

#[derive(Args, Debug)]
pub struct InstallArgs {
    /// A built `.wasm`, or a `.py` script (which needs no build).
    pub artifact: PathBuf,

    /// Its manifest. Defaults to `<stem>.toml` beside the artifact, then
    /// `Pie.toml` beside it; a bare script needs none.
    #[arg(long, short = 'm')]
    pub manifest: Option<PathBuf>,

    /// Replace an installed program of the same name and version.
    #[arg(long)]
    pub force: bool,
}

pub async fn run(cmd: InferletCmd) -> Result<Answer> {
    match cmd {
        InferletCmd::List => list(),
        InferletCmd::Info(args) => info(args),
        InferletCmd::Install(args) => install(args).await,
        InferletCmd::Remove(args) => remove(args),
    }
}

/// What is installed under the inferlets directory, plus the programs built
/// into the binary (which a disk install of the same name and version
/// shadows).
fn open() -> Repository {
    let mut repo = Repository::new(bootstrap::paths::inferlets_dir());
    repo.refresh();
    for builtin in builtins::all() {
        if let Ok(manifest) = Manifest::parse(builtin.manifest) {
            repo.add_builtin(manifest, builtin.component);
        }
    }
    repo
}

/// Resolve `name` (its newest installed version) or `name@version` against
/// what is installed. Nothing is fetched: an absent program is an error
/// that says what is here instead.
pub(crate) fn resolve_installed(spec: &str) -> Result<ProgramName> {
    resolve_in(&open(), spec)
}

fn resolve_in(repo: &Repository, spec: &str) -> Result<ProgramName> {
    let name = match spec.split_once('@') {
        None => spec,
        Some((name, _)) => {
            let program = ProgramName::parse(spec)?;
            if repo.exists(&program) {
                return Ok(program);
            }
            let others: Vec<String> = repo
                .cached()
                .into_iter()
                .filter(|(program, _, _)| program.name == name)
                .map(|(program, _, _)| program.version)
                .collect();
            if !others.is_empty() {
                bail!(
                    "{spec} is not installed; {name} is here as {}",
                    others.join(", ")
                );
            }
            name
        }
    };
    repo.newest(name).ok_or_else(|| {
        anyhow::anyhow!(
            "{spec} is not installed; `pie inferlet list` shows what is, \
             `pie inferlet install <file.wasm>` adds one"
        )
    })
}

#[derive(serde::Serialize)]
#[serde(transparent)]
pub struct InferletList {
    inferlets: Vec<InstalledInferlet>,
}

#[derive(serde::Serialize)]
struct InstalledInferlet {
    name: String,
    version: String,
    description: Option<String>,
    bytes: u64,
    /// Served from the binary, not the inferlets directory.
    builtin: bool,
}

impl ui::Report for InferletList {
    fn render(&self, palette: &Palette) {
        if self.inferlets.is_empty() {
            println!("nothing installed yet");
            println!(
                "  `pie inferlet install <file.wasm | file.py>` adds one; `pie run <file>` runs one without installing it"
            );
            return;
        }
        let mut table = Table::new([Align::Left, Align::Right, Align::Left, Align::Left], 2);
        for inferlet in &self.inferlets {
            let description = inferlet
                .description
                .as_deref()
                .unwrap_or("")
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            table.push(Row::new(
                Mark::Plain,
                [
                    format!("{}@{}", inferlet.name, inferlet.version),
                    ui::bytes(inferlet.bytes),
                    if inferlet.builtin { "built-in" } else { "" }.to_string(),
                    description,
                ],
            ));
        }
        table.print(palette);
    }
}

fn list() -> Result<Answer> {
    let repo = open();
    Ok(Answer::report(InferletList {
        inferlets: repo
            .cached()
            .into_iter()
            .map(|(name, manifest, bytes)| InstalledInferlet {
                builtin: repo.is_builtin(&name),
                name: name.name,
                version: name.version,
                description: manifest.package.description,
                bytes,
            })
            .collect(),
    }))
}

async fn install(args: InstallArgs) -> Result<Answer> {
    let artifact = load_artifact(&args.artifact, args.manifest.as_deref())?;
    let name = artifact.manifest.program_name();

    let mut repo = open();
    if repo.exists(&name) && !args.force {
        return Ok(Answer::noop(format!(
            "{name} is already installed; `--force` replaces it"
        )));
    }
    repo.add(artifact.bytes, artifact.manifest, args.force)
        .await
        .with_context(|| format!("installing {name}"))?;
    Ok(Answer::did(format!(
        "installed {name} into {}",
        ui::short_path(repo.programs_dir())
    )))
}

/// A program read from disk with the manifest that names it: a component
/// beside its manifest, or a script, whose manifest may be synthesized from
/// the file alone.
#[derive(Debug)]
pub struct Artifact {
    pub bytes: Vec<u8>,
    pub manifest: Manifest,
    /// The manifest as the engine will receive it.
    pub manifest_toml: String,
}

/// Read `path` and settle its manifest. A `.wasm` needs a manifest beside
/// it (or named with `manifest`). A script (`.py`) takes the manifest beside
/// it when there is one, else one synthesized from the file name; either
/// way the manifest states the script's language.
pub(crate) fn load_artifact(path: &Path, manifest: Option<&Path>) -> Result<Artifact> {
    if !path.is_file() {
        bail!("no file at {}", path.display());
    }
    let language = path
        .extension()
        .and_then(|e| e.to_str())
        .and_then(Language::from_extension);
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;

    let manifest_path = match manifest {
        Some(named) => Ok(named.to_path_buf()),
        None => manifest_beside(path),
    };
    let mut manifest = match (manifest_path, language) {
        (Ok(manifest_path), _) => {
            let content = std::fs::read_to_string(&manifest_path)
                .with_context(|| format!("reading {}", manifest_path.display()))?;
            Manifest::parse(&content)
                .with_context(|| format!("reading {}", manifest_path.display()))?
        }
        (Err(_), Some(language)) => synthesized_manifest(path, language)?,
        (Err(error), None) => return Err(error),
    };

    match (language, manifest.language()) {
        (Some(from_file), None) => manifest.runtime.language = Some(from_file),
        (Some(from_file), Some(declared)) if from_file != declared => bail!(
            "{} is {from_file} source but its manifest says `language = \"{declared}\"`",
            path.display()
        ),
        (None, Some(declared)) => bail!(
            "{} is a component but its manifest says `language = \"{declared}\"`; a \
             {declared} inferlet is its source file, not a build",
            path.display()
        ),
        _ => {}
    }

    let manifest_toml = manifest.to_toml()?;
    Ok(Artifact {
        bytes,
        manifest,
        manifest_toml,
    })
}

/// The manifest a bare script implies: named after the file, version
/// 0.0.0, in the file's language.
fn synthesized_manifest(path: &Path, language: Language) -> Result<Manifest> {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("{} has no file name", path.display()))?;
    let name = stem.replace('_', "-");
    ProgramName::parse(&format!("{name}@0.0.0")).with_context(|| {
        format!(
            "{} cannot name a program; a `Pie.toml` beside it can",
            path.display()
        )
    })?;
    Ok(Manifest {
        package: Package {
            name,
            version: "0.0.0".to_string(),
            description: None,
            authors: Vec::new(),
            repository: None,
            readme: None,
            tier: None,
        },
        runtime: Runtime {
            language: Some(language),
            ..Default::default()
        },
        parameters: Default::default(),
        dependencies: Default::default(),
    })
}

/// The manifest an artifact carries beside it: `<stem>.toml` (the installed
/// layout) or `Pie.toml` (a project directory).
fn manifest_beside(artifact: &Path) -> Result<PathBuf> {
    let sibling = artifact.with_extension("toml");
    if sibling.is_file() {
        return Ok(sibling);
    }
    if let Some(dir) = artifact.parent() {
        let pie_toml = dir.join("Pie.toml");
        if pie_toml.is_file() {
            return Ok(pie_toml);
        }
    }
    bail!(
        "no manifest beside {}: expected {} or a Pie.toml in the same directory; \
         `--manifest` names one elsewhere",
        artifact.display(),
        sibling.display()
    )
}

fn remove(args: TargetArgs) -> Result<Answer> {
    let mut repo = open();
    let name = match args.inferlet.split_once('@') {
        Some(_) => ProgramName::parse(&args.inferlet)?,
        None => {
            let matching: Vec<ProgramName> = repo
                .cached()
                .into_iter()
                .map(|(name, _, _)| name)
                .filter(|name| name.name == args.inferlet)
                .collect();
            match matching.as_slice() {
                [] => bail!(
                    "{} is not installed; `pie inferlet list` shows what is",
                    args.inferlet
                ),
                [one] => one.clone(),
                many => {
                    let versions: Vec<&str> = many.iter().map(|n| n.version.as_str()).collect();
                    bail!(
                        "{} has {} versions installed ({}); name the one to remove",
                        args.inferlet,
                        versions.len(),
                        versions.join(", ")
                    );
                }
            }
        }
    };
    if repo.is_builtin(&name) {
        bail!(
            "{name} is built into this pie and cannot be removed; installing another version \
             of {} shadows it",
            name.name
        );
    }
    Ok(if repo.remove(&name)? {
        Answer::did(format!("removed {name}"))
    } else {
        Answer::noop(format!("{name} was not installed"))
    })
}

fn info(args: TargetArgs) -> Result<Answer> {
    let repo = open();
    let program = resolve_in(&repo, &args.inferlet)?;
    let manifest = repo
        .fetch_manifest(&program)
        .ok_or_else(|| anyhow::anyhow!("{}", repo.not_installed(&program)))?;

    Ok(Answer::report(InferletInfo {
        name: program.name,
        version: program.version,
        description: manifest.package.description,
        authors: manifest.package.authors,
        repository: manifest.package.repository,
        runtime: serde_json::to_value(&manifest.runtime)?,
        dependencies: serde_json::to_value(&manifest.dependencies)?,
        parameters: manifest
            .parameters
            .into_iter()
            .map(|(name, p)| Parameter {
                name,
                r#type: match p.param_type {
                    ParameterType::String => "string",
                    ParameterType::Int => "int",
                    ParameterType::Float => "float",
                    ParameterType::Bool => "bool",
                },
                optional: p.optional,
                description: p.description,
            })
            .collect(),
    }))
}

#[derive(serde::Serialize)]
pub struct InferletInfo {
    name: String,
    version: String,
    description: Option<String>,
    authors: Vec<String>,
    repository: Option<String>,
    runtime: serde_json::Value,
    dependencies: serde_json::Value,
    parameters: Vec<Parameter>,
}

#[derive(serde::Serialize)]
struct Parameter {
    name: String,
    r#type: &'static str,
    optional: bool,
    description: Option<String>,
}

impl ui::Report for InferletInfo {
    fn render(&self, palette: &Palette) {
        println!(
            "{}",
            palette.bold(format!("{}@{}", self.name, self.version))
        );
        if let Some(description) = &self.description {
            println!("{description}");
        }
        if let Some(repository) = &self.repository {
            println!("{}", palette.dim(repository));
        }

        if self.parameters.is_empty() {
            println!("\n{}", palette.dim("(no parameters)"));
            return;
        }

        println!("\n{}", palette.bold("Parameters"));
        let name_width = self
            .parameters
            .iter()
            .map(|p| p.name.chars().count())
            .max()
            .unwrap_or(4)
            .max("name".len());
        let type_width = self
            .parameters
            .iter()
            .map(|p| p.r#type.chars().count())
            .max()
            .unwrap_or(4)
            .max("type".len());

        println!(
            "{}",
            palette.dim(format!(
                "{:<name_width$}  {:<type_width$}  required  description",
                "name", "type"
            ))
        );
        for parameter in &self.parameters {
            let required = format!(
                "{:<8}",
                if parameter.optional {
                    "optional"
                } else {
                    "yes"
                }
            );
            let required = if parameter.optional {
                palette.dim(required).to_string()
            } else {
                palette.green(required).to_string()
            };
            println!(
                "{}  {:<type_width$}  {required}  {}",
                palette.accent(format!("{:<name_width$}", parameter.name)),
                parameter.r#type,
                palette.dim(parameter.description.as_deref().unwrap_or("")),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(name: &str, version: &str) -> Manifest {
        Manifest::parse(&format!(
            "[package]\nname = \"{name}\"\nversion = \"{version}\"\n"
        ))
        .unwrap()
    }

    async fn repo_with(dir: &Path, programs: &[(&str, &str)]) -> Repository {
        let mut repo = Repository::new(dir.to_path_buf());
        for (name, version) in programs {
            repo.add(b"\0asm".to_vec(), manifest(name, version), false)
                .await
                .unwrap();
        }
        repo
    }

    #[tokio::test]
    async fn inferlet_every_case() {
        a_bare_name_resolves_to_the_newest_installed_version().await;
        an_absent_program_is_refused_with_what_is_here().await;
        the_manifest_beside_a_wasm_is_its_stem_toml_then_pie_toml();
        a_bare_script_is_named_by_its_file_in_its_language();
        a_script_beside_a_manifest_takes_the_manifest_and_states_its_language();
        a_component_whose_manifest_claims_a_language_is_refused();
    }

    fn a_bare_script_is_named_by_its_file_in_its_language() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("beam_search.py");
        std::fs::write(&script, "async def main(input): return input\n").unwrap();
        let artifact = load_artifact(&script, None).unwrap();
        assert_eq!(
            artifact.manifest.program_name().to_string(),
            "beam-search@0.0.0"
        );
        assert_eq!(artifact.manifest.language(), Some(Language::Python));
        assert!(artifact.manifest_toml.contains("language = \"python\""));
        assert_eq!(artifact.bytes, b"async def main(input): return input\n");
    }

    fn a_script_beside_a_manifest_takes_the_manifest_and_states_its_language() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("main.py");
        std::fs::write(&script, "x = 1\n").unwrap();
        std::fs::write(
            dir.path().join("Pie.toml"),
            "[package]\nname = \"twin\"\nversion = \"0.2.0\"\n",
        )
        .unwrap();
        let artifact = load_artifact(&script, None).unwrap();
        assert_eq!(artifact.manifest.program_name().to_string(), "twin@0.2.0");
        assert_eq!(artifact.manifest.language(), Some(Language::Python));
    }

    fn a_component_whose_manifest_claims_a_language_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let wasm = dir.path().join("0.1.0.wasm");
        std::fs::write(&wasm, b"\0asm").unwrap();
        std::fs::write(
            dir.path().join("0.1.0.toml"),
            "[package]\nname = \"x\"\nversion = \"0.1.0\"\n[runtime]\nlanguage = \"python\"\n",
        )
        .unwrap();
        let err = load_artifact(&wasm, None).unwrap_err().to_string();
        assert!(err.contains("component"), "{err}");
    }

    async fn a_bare_name_resolves_to_the_newest_installed_version() {
        let dir = tempfile::tempdir().unwrap();
        let repo = repo_with(dir.path(), &[("beam", "0.1.0"), ("beam", "0.2.0")]).await;
        assert_eq!(resolve_in(&repo, "beam").unwrap().version, "0.2.0");
        assert_eq!(resolve_in(&repo, "beam@0.1.0").unwrap().version, "0.1.0");
    }

    async fn an_absent_program_is_refused_with_what_is_here() {
        let dir = tempfile::tempdir().unwrap();
        let repo = repo_with(dir.path(), &[("beam", "0.1.0")]).await;
        let err = resolve_in(&repo, "beam@0.9.0").unwrap_err().to_string();
        assert!(err.contains("0.1.0"), "{err}");
        let err = resolve_in(&repo, "sink").unwrap_err().to_string();
        assert!(err.contains("pie inferlet install"), "{err}");
        let err = resolve_in(&repo, "sink@0.1.0").unwrap_err().to_string();
        assert!(err.contains("pie inferlet install"), "{err}");
    }

    fn the_manifest_beside_a_wasm_is_its_stem_toml_then_pie_toml() {
        let dir = tempfile::tempdir().unwrap();
        let wasm = dir.path().join("0.1.0.wasm");
        std::fs::write(&wasm, b"\0asm").unwrap();
        assert!(manifest_beside(&wasm).is_err());
        std::fs::write(dir.path().join("Pie.toml"), "").unwrap();
        assert_eq!(manifest_beside(&wasm).unwrap(), dir.path().join("Pie.toml"));
        std::fs::write(dir.path().join("0.1.0.toml"), "").unwrap();
        assert_eq!(
            manifest_beside(&wasm).unwrap(),
            dir.path().join("0.1.0.toml")
        );
    }
}
