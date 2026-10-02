//! `pie language`: the language components Python and JavaScript inferlets
//! run under, installed from local files: the release asset
//! `pie-language-<language>.tar.gz` (holding `languages/<language>.wasm`),
//! or a bare `.wasm` from `<language>/inferlet/language/build.sh`.

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};

use runtime::inferlet::program::Language;

use crate::ui::{self, Align, Answer, Mark, Palette, Row, Table};

#[derive(Subcommand, Debug)]
pub enum LanguageCmd {
    /// The languages pie can run script inferlets in, and which are installed.
    List,

    /// Install a language component from a release archive or a `.wasm`.
    Install(InstallArgs),

    /// Remove an installed language component.
    Remove(RemoveArgs),
}

#[derive(Args, Debug)]
pub struct InstallArgs {
    /// `pie-language-<language>.tar.gz` from a pie release, or a built
    /// `<language>.wasm`.
    pub file: PathBuf,

    /// The language a bare `.wasm` is for, when its file name does not say
    /// (`python` or `javascript`).
    #[arg(long, short = 'l')]
    pub language: Option<String>,

    /// Replace a component that is already installed.
    #[arg(long)]
    pub force: bool,
}

#[derive(Args, Debug)]
pub struct RemoveArgs {
    /// `python` or `javascript`.
    pub language: String,
}

/// Where the components live: `$PIE_HOME/languages/<language>.wasm`, which
/// is also where install.sh puts them.
pub(crate) fn dir() -> PathBuf {
    bootstrap::paths::pie_home().join("languages")
}

pub(crate) fn path(language: Language) -> PathBuf {
    dir().join(language.component_file())
}

pub fn run(cmd: LanguageCmd) -> Result<Answer> {
    let dir = dir();
    match cmd {
        LanguageCmd::List => list(&dir),
        LanguageCmd::Install(args) => install(&dir, &args),
        LanguageCmd::Remove(args) => remove(&dir, &args),
    }
}

#[derive(serde::Serialize)]
#[serde(transparent)]
struct LanguageList {
    languages: Vec<LanguageRow>,
}

#[derive(serde::Serialize)]
struct LanguageRow {
    language: String,
    installed: bool,
    path: String,
    bytes: u64,
}

impl ui::Report for LanguageList {
    fn render(&self, palette: &Palette) {
        let mut table = Table::new([Align::Left, Align::Right, Align::Left], 2);
        for row in &self.languages {
            if row.installed {
                table.push(Row::new(
                    Mark::Did,
                    [row.language.clone(), ui::bytes(row.bytes), row.path.clone()],
                ));
            } else {
                table.push(Row::new(
                    Mark::Absent,
                    [
                        row.language.clone(),
                        String::new(),
                        format!(
                            "not installed; `pie language install pie-language-{}.tar.gz` adds it",
                            row.language
                        ),
                    ],
                ));
            }
        }
        table.print(palette);
    }
}

fn list(dir: &Path) -> Result<Answer> {
    let languages = Language::ALL
        .iter()
        .map(|&language| {
            let path = dir.join(language.component_file());
            let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            LanguageRow {
                language: language.name().to_string(),
                installed: path.is_file(),
                path: ui::short_path(&path),
                bytes,
            }
        })
        .collect();
    Ok(Answer::report(LanguageList { languages }))
}

fn install(dir: &Path, args: &InstallArgs) -> Result<Answer> {
    let file = &args.file;
    if !file.is_file() {
        bail!("{} is not a file", file.display());
    }
    let named = match &args.language {
        Some(word) => Some(Language::parse(word)?),
        None => None,
    };
    let bytes = std::fs::read(file).with_context(|| format!("reading {}", file.display()))?;
    let installed = if bytes.starts_with(WASM_MAGIC) {
        let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let language = match named.or_else(|| Language::parse(stem).ok()) {
            Some(language) => language,
            None => bail!(
                "{} does not say which language it is; name it `python.wasm` or \
                 `javascript.wasm`, or pass --language",
                file.display()
            ),
        };
        vec![place(dir, language, &bytes, args.force)?]
    } else {
        let found = install_archive(dir, file, &bytes, args.force)?;
        if let Some(language) = named
            && !found.contains(&language)
        {
            bail!(
                "{} holds no {language} component (it has {})",
                file.display(),
                names(&found)
            );
        }
        found
    };
    Ok(Answer::did(format!(
        "installed {} into {}",
        names(&installed),
        ui::short_path(dir)
    )))
}

fn remove(dir: &Path, args: &RemoveArgs) -> Result<Answer> {
    let language = Language::parse(&args.language)?;
    let path = dir.join(language.component_file());
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(Answer::did(format!(
            "removed {language} ({})",
            ui::short_path(&path)
        ))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Answer::noop(format!(
            "{language} was not installed (no {})",
            ui::short_path(&path)
        ))),
        Err(e) => Err(e).with_context(|| format!("removing {}", path.display())),
    }
}

const WASM_MAGIC: &[u8] = b"\0asm";

/// Install every `languages/<language>.wasm` the gzipped tar `bytes` holds.
fn install_archive(dir: &Path, file: &Path, bytes: &[u8], force: bool) -> Result<Vec<Language>> {
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(bytes));
    let mut found = Vec::new();
    let entries = archive
        .entries()
        .with_context(|| format!("{} is neither wasm nor a gzipped tar", file.display()))?;
    for entry in entries {
        let mut entry = entry.with_context(|| format!("reading {}", file.display()))?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry.path()?.into_owned();
        let Some(language) = archived_language(&path) else {
            continue;
        };
        let mut component = Vec::new();
        entry
            .read_to_end(&mut component)
            .with_context(|| format!("reading {} from {}", path.display(), file.display()))?;
        found.push(place(dir, language, &component, force)?);
    }
    if found.is_empty() {
        bail!(
            "{} holds no `languages/<language>.wasm`; a release's pie-language-<language>.tar.gz does",
            file.display()
        );
    }
    Ok(found)
}

/// `languages/python.wasm` → Python; any other entry is not a component.
fn archived_language(path: &Path) -> Option<Language> {
    let file = path.strip_prefix("languages").ok()?.to_str()?;
    Language::ALL
        .into_iter()
        .find(|language| language.component_file() == file)
}

/// Write `bytes` as the `language` component, atomically, refusing to
/// replace one already there unless `force`.
fn place(dir: &Path, language: Language, bytes: &[u8], force: bool) -> Result<Language> {
    if !bytes.starts_with(WASM_MAGIC) {
        bail!("the {language} component is not wasm");
    }
    let path = dir.join(language.component_file());
    if path.exists() && !force {
        bail!(
            "{language} is already installed at {}; --force replaces it",
            ui::short_path(&path)
        );
    }
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let staging = dir.join(format!(".{}.tmp", language.component_file()));
    std::fs::write(&staging, bytes).with_context(|| format!("writing {}", staging.display()))?;
    std::fs::rename(&staging, &path)
        .with_context(|| format!("moving {} into place", staging.display()))?;
    Ok(language)
}

fn names(languages: &[Language]) -> String {
    languages
        .iter()
        .map(|l| l.name())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn archive(dir: &Path, entries: &[(&str, &[u8])]) -> PathBuf {
        let path = dir.join("asset.tar.gz");
        let file = std::fs::File::create(&path).unwrap();
        let gz = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        let mut tar = tar::Builder::new(gz);
        for (name, bytes) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            tar.append_data(&mut header, name, *bytes).unwrap();
        }
        tar.into_inner().unwrap().finish().unwrap();
        path
    }

    fn install_args(file: PathBuf, language: Option<&str>, force: bool) -> InstallArgs {
        InstallArgs {
            file,
            language: language.map(str::to_string),
            force,
        }
    }

    fn error_of(answer: Result<Answer>) -> String {
        answer.err().expect("an error").to_string()
    }

    fn remove_args(language: &str) -> RemoveArgs {
        RemoveArgs {
            language: language.to_string(),
        }
    }

    #[test]
    fn language_every_case() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("languages");

        // A release archive installs what it holds and nothing else.
        let asset = archive(
            tmp.path(),
            &[
                ("languages/python.wasm", b"\0asm\x01\0\0\0py"),
                ("languages/README", b"not a component"),
                ("elsewhere/javascript.wasm", b"\0asm\x01\0\0\0no"),
            ],
        );
        install(&home, &install_args(asset.clone(), None, false)).unwrap();
        assert_eq!(
            std::fs::read(home.join("python.wasm")).unwrap(),
            b"\0asm\x01\0\0\0py"
        );
        assert!(!home.join("javascript.wasm").exists());
        assert!(!home.join("README").exists());

        // Already installed: refused without --force, replaced with it.
        let err = error_of(install(&home, &install_args(asset.clone(), None, false)));
        assert!(err.contains("already installed"), "{err}");
        install(&home, &install_args(asset.clone(), None, true)).unwrap();

        // Asking for a language the archive lacks is an error.
        let err = error_of(install(
            &home,
            &install_args(asset, Some("javascript"), true),
        ));
        assert!(err.contains("no javascript component"), "{err}");

        // A bare .wasm: named by its stem, or by --language.
        let js = tmp.path().join("javascript.wasm");
        std::fs::write(&js, b"\0asm\x01\0\0\0js").unwrap();
        install(&home, &install_args(js, None, false)).unwrap();
        assert!(home.join("javascript.wasm").is_file());
        let odd = tmp.path().join("build-output.wasm");
        std::fs::write(&odd, b"\0asm\x01\0\0\0js2").unwrap();
        let err = error_of(install(&home, &install_args(odd.clone(), None, true)));
        assert!(err.contains("--language"), "{err}");
        install(&home, &install_args(odd, Some("javascript"), true)).unwrap();
        assert_eq!(
            std::fs::read(home.join("javascript.wasm")).unwrap(),
            b"\0asm\x01\0\0\0js2"
        );

        // Not wasm, not an archive: refused.
        let junk = tmp.path().join("junk.tar.gz");
        std::fs::write(&junk, b"hello").unwrap();
        assert!(install(&home, &install_args(junk, None, false)).is_err());

        // Remove, twice.
        assert!(remove(&home, &remove_args("python")).is_ok());
        assert!(!home.join("python.wasm").exists());
        assert!(remove(&home, &remove_args("python")).is_ok());
        assert!(remove(&home, &remove_args("cobol")).is_err());

        // What is left is listed.
        assert!(list(&home).is_ok());
        assert_eq!(
            archived_language(Path::new("languages/python.wasm")),
            Some(Language::Python)
        );
        assert_eq!(
            archived_language(Path::new("languages/x/python.wasm")),
            None
        );
        assert_eq!(archived_language(Path::new("python.wasm")), None);
    }
}
