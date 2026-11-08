use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};

use runtime::inferlet::program::{
    self, Identity, Listed, Namespace, Program, ProgramRef, Reclaimed, Repository,
};

use crate::ui::{self, Align, Answer, Mark, Palette, Row, Table};

#[derive(Subcommand, Debug)]
pub enum InferletCmd {
    /// Every installed build, by namespace and name, and the built-ins.
    List,

    /// What a name, or `name@<hash>`, runs.
    Info(TargetArgs),

    /// Install a `.wasm`, `.py` or `.js` file: the system's, unless
    /// `--tenant` names whose.
    Install(InstallArgs),

    /// Remove a build (`name@<hash>`), or every build of a name.
    Remove(TargetArgs),

    /// Remove every build but each name's latest, and the bytes nothing
    /// names any more.
    Prune,
}

#[derive(Args, Debug)]
pub struct TargetArgs {
    /// `name`, or `name@<hash prefix>` for one build.
    pub inferlet: String,

    /// A tenant's own installs, rather than the system's.
    #[arg(long)]
    pub tenant: Option<String>,
}

#[derive(Args, Debug)]
pub struct InstallArgs {
    pub artifact: PathBuf,

    /// Install among this tenant's programs, rather than the system's.
    #[arg(long)]
    pub tenant: Option<String>,
}

pub async fn run(cmd: InferletCmd) -> Result<Answer> {
    match cmd {
        InferletCmd::List => list(),
        InferletCmd::Info(args) => info(args),
        InferletCmd::Install(args) => install(args).await,
        InferletCmd::Remove(args) => remove(args),
        InferletCmd::Prune => prune(),
    }
}

/// What the inferlets directory holds, and the programs built into the
/// binary under it.
fn open() -> Repository {
    let mut repo = Repository::new(crate::paths::inferlets_dir());
    for builtin in builtins::all() {
        repo.add_builtin(builtin.name, builtin.component);
    }
    repo
}

fn namespace(tenant: Option<String>) -> Namespace {
    tenant.map_or(Namespace::System, Namespace::Tenant)
}

/// The program `spec` names for a `pie run` session, whose tenant is the
/// default one. Nothing is fetched: an absent program is an error.
pub(crate) fn resolve_installed(spec: &str) -> Result<Program> {
    resolve_in(&open(), Some(runtime::server::DEFAULT_TENANT), spec)
}

fn resolve_in(repo: &Repository, tenant: Option<&str>, spec: &str) -> Result<Program> {
    let reference = ProgramRef::parse(spec)?;
    repo.resolve(tenant, &reference).with_context(|| {
        format!(
            "`pie inferlet list` shows what is installed, `pie inferlet install <file.wasm>` \
             adds {}",
            reference.name
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
    hash: String,
    /// `None`: the system's.
    tenant: Option<String>,
    language: Option<String>,
    bytes: u64,
    /// Served from the binary, not the inferlets directory.
    builtin: bool,
    /// What the bare name runs.
    current: bool,
    #[serde(skip)]
    shown: String,
    #[serde(skip)]
    kind: String,
}

impl From<Listed> for InstalledInferlet {
    fn from(listed: Listed) -> Self {
        Self {
            shown: listed.program.to_string(),
            kind: program::kind(listed.language),
            name: listed.program.name,
            hash: listed.program.hash.to_string(),
            tenant: match listed.namespace {
                Namespace::System => None,
                Namespace::Tenant(tenant) => Some(tenant),
            },
            language: listed.language.map(|l| l.name().to_string()),
            bytes: listed.bytes,
            builtin: listed.builtin,
            current: listed.current,
        }
    }
}

impl InstalledInferlet {
    fn owner(&self) -> String {
        match (&self.tenant, self.builtin) {
            (Some(tenant), _) => format!("tenant {tenant}"),
            (None, true) => "built-in".to_string(),
            (None, false) => "system".to_string(),
        }
    }
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
        let mut table = Table::new([Align::Left, Align::Left, Align::Right, Align::Left], 2);
        for inferlet in &self.inferlets {
            let mark = if inferlet.current {
                Mark::Chosen
            } else {
                Mark::Plain
            };
            table.push(Row::new(
                mark,
                [
                    inferlet.shown.clone(),
                    inferlet.owner(),
                    ui::bytes(inferlet.bytes),
                    inferlet.kind.clone(),
                ],
            ));
        }
        table.print(palette);
        println!(
            "{}",
            palette.dim(format!(
                "{} is what a bare name runs; `name@<hash>` runs any build",
                Mark::Chosen.render(palette)
            ))
        );
    }
}

fn list() -> Result<Answer> {
    Ok(Answer::report(InferletList {
        inferlets: open().list().into_iter().map(Into::into).collect(),
    }))
}

async fn install(args: InstallArgs) -> Result<Answer> {
    let artifact = load_artifact(&args.artifact)?;
    let program = artifact.identity.program.clone();
    let namespace = namespace(args.tenant);
    let mut repo = open();
    repo.add(namespace.clone(), &artifact.identity, artifact.bytes)
        .await
        .with_context(|| format!("installing {program}"))?;
    Ok(Answer::did(format!(
        "installed {program} for the {namespace}, into {}",
        ui::short_path(repo.root())
    )))
}

#[derive(Debug)]
pub struct Artifact {
    pub bytes: Vec<u8>,
    pub file: String,
    pub identity: Identity,
}

pub(crate) fn load_artifact(path: &Path) -> Result<Artifact> {
    if !path.is_file() {
        bail!("no file at {}", path.display());
    }
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let file = program_file(path)?;
    let identity = program::identify(&file, &bytes)
        .with_context(|| format!("{} is not a program pie can install", path.display()))?;
    Ok(Artifact {
        bytes,
        file,
        identity,
    })
}

/// The file name a program goes up as, which names it: an entry script
/// (`main.py`, `index.js`) takes its directory's name.
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
    let reference = ProgramRef::parse(&args.inferlet)?;
    let namespace = namespace(args.tenant);
    let reclaimed = open().remove(&namespace, &reference)?;
    Ok(Answer::did(reclaimed_line("removed", &reclaimed)))
}

fn prune() -> Result<Answer> {
    let reclaimed = open().prune()?;
    Ok(if reclaimed.installs.is_empty() && reclaimed.blobs == 0 {
        Answer::noop("nothing to prune: every name holds its latest build alone".to_string())
    } else {
        Answer::did(reclaimed_line("pruned", &reclaimed))
    })
}

fn reclaimed_line(verb: &str, reclaimed: &Reclaimed) -> String {
    let builds: Vec<String> = reclaimed
        .installs
        .iter()
        .map(|(namespace, program)| match namespace {
            Namespace::System => program.to_string(),
            Namespace::Tenant(tenant) => format!("{program} ({tenant})"),
        })
        .collect();
    let builds = if builds.is_empty() {
        "no build".to_string()
    } else {
        builds.join(", ")
    };
    format!(
        "{verb} {builds}; {} blob(s), {} reclaimed",
        reclaimed.blobs,
        ui::bytes(reclaimed.bytes)
    )
}

fn info(args: TargetArgs) -> Result<Answer> {
    let repo = open();
    let program = resolve_in(&repo, args.tenant.as_deref(), &args.inferlet)?;
    let listed = repo
        .list()
        .into_iter()
        .filter(|listed| listed.program == program)
        .min_by_key(|listed| !listed.current)
        .ok_or_else(|| anyhow::anyhow!("{program} is not installed"))?;
    Ok(Answer::report(InferletInfo(listed.into())))
}

#[derive(serde::Serialize)]
#[serde(transparent)]
pub struct InferletInfo(InstalledInferlet);

impl ui::Report for InferletInfo {
    fn render(&self, palette: &Palette) {
        let inferlet = &self.0;
        println!(
            "{}",
            palette.bold(format!("{}@{}", inferlet.name, inferlet.hash))
        );
        println!(
            "{}",
            palette.dim(format!(
                "{}, {}, {}",
                inferlet.kind,
                ui::bytes(inferlet.bytes),
                inferlet.owner()
            ))
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inferlet_every_case() {
        a_bare_script_is_named_by_its_file_in_its_language();
        an_entry_script_is_named_by_its_directory();
    }

    fn a_bare_script_is_named_by_its_file_in_its_language() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("beam_search.py");
        let source = b"async def main(input): return input\n";
        std::fs::write(&script, source).unwrap();
        let artifact = load_artifact(&script).unwrap();
        assert_eq!(artifact.file, "beam-search.py");
        assert_eq!(artifact.identity.program.name, "beam-search");
        assert_eq!(
            artifact.identity.program.hash,
            program::ProgramHash::of(source)
        );
        assert_eq!(artifact.identity.language, Some(program::Language::Python));
        assert_eq!(artifact.bytes, source);
    }

    fn an_entry_script_is_named_by_its_directory() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("text_completion_js");
        std::fs::create_dir(&project).unwrap();
        let script = project.join("index.js");
        std::fs::write(&script, "export function main(input) { return input; }\n").unwrap();
        let artifact = load_artifact(&script).unwrap();
        assert_eq!(artifact.file, "text-completion-js.js");
        assert_eq!(artifact.identity.program.name, "text-completion-js");
        assert_eq!(
            artifact.identity.language,
            Some(program::Language::JavaScript)
        );
    }
}
