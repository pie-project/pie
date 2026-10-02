use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};

use runtime::inferlet::program::{self, Identity, ProgramName, Repository};

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
    pub artifact: PathBuf,

    #[arg(long)]
    pub version: Option<String>,

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
        if let Ok(name) = ProgramName::parse(&format!("{}@{}", builtin.name, builtin.version)) {
            repo.add_builtin(name, builtin.component);
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
    language: Option<String>,
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
            table.push(Row::new(
                Mark::Plain,
                [
                    format!("{}@{}", inferlet.name, inferlet.version),
                    ui::bytes(inferlet.bytes),
                    inferlet.language.clone().unwrap_or_default(),
                    if inferlet.builtin { "built-in" } else { "" }.to_string(),
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
            .map(|(name, language, bytes)| InstalledInferlet {
                builtin: repo.is_builtin(&name),
                name: name.name,
                version: name.version,
                language: language.map(|l| l.name().to_string()),
                bytes,
            })
            .collect(),
    }))
}

async fn install(args: InstallArgs) -> Result<Answer> {
    let artifact = load_artifact(&args.artifact, args.version.as_deref())?;
    let name = artifact.identity.name.clone();

    let mut repo = open();
    if repo.exists(&name) && !args.force {
        return Ok(Answer::noop(format!(
            "{name} is already installed; `--force` replaces it"
        )));
    }
    repo.add(
        artifact.bytes,
        name.clone(),
        artifact.identity.language,
        args.force,
    )
    .await
    .with_context(|| format!("installing {name}"))?;
    Ok(Answer::did(format!(
        "installed {name} into {}",
        ui::short_path(repo.programs_dir())
    )))
}

#[derive(Debug)]
pub struct Artifact {
    pub bytes: Vec<u8>,
    pub file: String,
    pub identity: Identity,
}

pub(crate) fn load_artifact(path: &Path, version: Option<&str>) -> Result<Artifact> {
    if !path.is_file() {
        bail!("no file at {}", path.display());
    }
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let file = program_file(path)?;
    let identity = program::identify(&file, version, &bytes)
        .with_context(|| format!("{} is not a program pie can install", path.display()))?;
    Ok(Artifact {
        bytes,
        file,
        identity,
    })
}

pub(crate) fn program_file(path: &Path) -> Result<String> {
    let extension = path.extension().and_then(|e| e.to_str()).ok_or_else(|| {
        anyhow::anyhow!(
            "{} has no extension; a program is a `.wasm`, `.py` or `.js` file",
            path.display()
        )
    })?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow::anyhow!("{} has no file name", path.display()))?;
    let name = if matches!(stem, "main" | "index") && extension != "wasm" {
        std::fs::canonicalize(path)
            .ok()
            .and_then(|p| p.parent()?.file_name()?.to_str().map(str::to_string))
            .filter(|dir| !dir.is_empty())
            .unwrap_or_else(|| stem.to_string())
    } else {
        stem.to_string()
    };
    Ok(format!("{}.{extension}", name.replace('_', "-")))
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
    let (_, language, bytes) = repo
        .cached()
        .into_iter()
        .find(|(name, _, _)| *name == program)
        .ok_or_else(|| anyhow::anyhow!("{}", repo.not_installed(&program)))?;

    Ok(Answer::report(InferletInfo {
        builtin: repo.is_builtin(&program),
        name: program.name,
        version: program.version,
        language: language.map(|l| l.name().to_string()),
        bytes,
    }))
}

#[derive(serde::Serialize)]
pub struct InferletInfo {
    name: String,
    version: String,
    language: Option<String>,
    bytes: u64,
    builtin: bool,
}

impl ui::Report for InferletInfo {
    fn render(&self, palette: &Palette) {
        println!(
            "{}",
            palette.bold(format!("{}@{}", self.name, self.version))
        );
        let kind = match &self.language {
            Some(language) => format!("{language} script"),
            None => "component".to_string(),
        };
        let from = if self.builtin {
            "built into this pie"
        } else {
            "installed"
        };
        println!(
            "{}",
            palette.dim(format!("{kind}, {}, {from}", ui::bytes(self.bytes)))
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime::inferlet::program::Language;

    async fn repo_with(dir: &Path, programs: &[(&str, &str)]) -> Repository {
        let mut repo = Repository::new(dir.to_path_buf());
        for (name, version) in programs {
            let name = ProgramName::parse(&format!("{name}@{version}")).unwrap();
            repo.add(b"\0asm".to_vec(), name, None, false)
                .await
                .unwrap();
        }
        repo
    }

    fn component_named(package: &str) -> Vec<u8> {
        let name = program::PACKAGE_SECTION.as_bytes();
        let mut payload = vec![name.len() as u8];
        payload.extend_from_slice(name);
        payload.extend_from_slice(package.as_bytes());
        let mut bytes = b"\0asm\x01\0\0\0".to_vec();
        bytes.push(0);
        bytes.push(payload.len() as u8);
        bytes.extend_from_slice(&payload);
        bytes
    }

    #[tokio::test]
    async fn inferlet_every_case() {
        a_bare_name_resolves_to_the_newest_installed_version().await;
        an_absent_program_is_refused_with_what_is_here().await;
        a_bare_script_is_named_by_its_file_in_its_language();
        an_entry_script_is_named_by_its_directory();
        a_component_is_named_by_the_package_it_carries();
        a_declared_version_beats_the_hash_and_a_given_one_beats_both();
    }

    fn a_bare_script_is_named_by_its_file_in_its_language() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("beam_search.py");
        std::fs::write(&script, "async def main(input): return input\n").unwrap();
        let artifact = load_artifact(&script, None).unwrap();
        assert_eq!(artifact.file, "beam-search.py");
        assert_eq!(artifact.identity.name.name, "beam-search");
        assert_eq!(
            artifact.identity.name.version,
            program::hashed_version(b"async def main(input): return input\n")
        );
        assert_eq!(artifact.identity.language, Some(Language::Python));
        assert_eq!(artifact.bytes, b"async def main(input): return input\n");
    }

    fn an_entry_script_is_named_by_its_directory() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("text_completion_js");
        std::fs::create_dir(&project).unwrap();
        let script = project.join("index.js");
        std::fs::write(&script, "export function main(input) { return input; }\n").unwrap();
        let artifact = load_artifact(&script, Some("0.2.0")).unwrap();
        assert_eq!(artifact.file, "text-completion-js.js");
        assert_eq!(
            artifact.identity.name.to_string(),
            "text-completion-js@0.2.0"
        );
        assert_eq!(artifact.identity.language, Some(Language::JavaScript));
    }

    fn a_component_is_named_by_the_package_it_carries() {
        let dir = tempfile::tempdir().unwrap();
        let wasm = dir.path().join("text_completion.wasm");
        std::fs::write(&wasm, component_named("text-completion@0.3.0")).unwrap();
        let artifact = load_artifact(&wasm, None).unwrap();
        assert_eq!(artifact.identity.name.to_string(), "text-completion@0.3.0");
        assert_eq!(artifact.identity.language, None);
        let err = load_artifact(&wasm, Some("9.9.9")).unwrap_err();
        assert!(format!("{err:#}").contains("names itself"), "{err:#}");

        let bare = dir.path().join("probe.wasm");
        std::fs::write(&bare, b"\0asm\x01\0\0\0").unwrap();
        let artifact = load_artifact(&bare, Some("0.1.0")).unwrap();
        assert_eq!(artifact.identity.name.to_string(), "probe@0.1.0");
    }

    fn a_declared_version_beats_the_hash_and_a_given_one_beats_both() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("probe.py");
        std::fs::write(
            &script,
            "__version__ = \"1.2.3\"\nasync def main(i): return i\n",
        )
        .unwrap();
        assert_eq!(
            load_artifact(&script, None).unwrap().identity.name.version,
            "1.2.3"
        );
        assert_eq!(
            load_artifact(&script, Some("2.0.0"))
                .unwrap()
                .identity
                .name
                .version,
            "2.0.0"
        );
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
}
