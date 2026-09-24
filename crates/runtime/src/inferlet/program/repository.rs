use std::collections::HashMap;
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::time::SystemTime;

#[cfg(not(target_arch = "wasm32"))]
use anyhow::anyhow;
use anyhow::{Result, bail};

use super::ProgramName;
use super::language::Language;
#[cfg(not(target_arch = "wasm32"))]
use super::language::artifact_extension;

/// `major.minor.patch` as a sortable triple; anything else sorts first.
fn semver_key(version: &str) -> (u64, u64, u64) {
    let mut parts = version.split('.').map(|p| p.parse::<u64>().unwrap_or(0));
    (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    )
}

/// The extensions an installed artifact can carry: a component, or a
/// script in a language the host has a component for.
#[cfg(not(target_arch = "wasm32"))]
const ARTIFACT_EXTENSIONS: [&str; 3] = ["wasm", "py", "js"];

#[cfg(not(target_arch = "wasm32"))]
fn artifact_path(programs_dir: &Path, name: &ProgramName, extension: &str) -> PathBuf {
    programs_dir
        .join(&name.name)
        .join(format!("{}.{extension}", name.version))
}

pub struct Repository {
    index: HashMap<ProgramName, Option<Language>>,
    preloaded_binaries: HashMap<ProgramName, Vec<u8>>,
    /// What the binary carries: not removable, and shadowed by a disk copy
    /// of the same name and version.
    builtin: HashMap<ProgramName, &'static [u8]>,
    /// The index entries that came from disk, with their artifact's mtime,
    /// so [`Repository::refresh`] can tell a replaced artifact from a kept one.
    #[cfg(not(target_arch = "wasm32"))]
    on_disk: HashMap<ProgramName, Option<SystemTime>>,
    programs_dir: PathBuf,
}

impl Repository {
    pub fn new(programs_dir: PathBuf) -> Self {
        Self {
            preloaded_binaries: HashMap::new(),
            builtin: HashMap::new(),
            index: HashMap::new(),
            #[cfg(not(target_arch = "wasm32"))]
            on_disk: HashMap::new(),
            programs_dir,
        }
    }

    pub fn programs_dir(&self) -> &Path {
        &self.programs_dir
    }

    pub fn language(&self, name: &ProgramName) -> Option<Option<Language>> {
        self.index.get(name).copied()
    }

    /// A copy of the same name and version already here (installed on
    /// disk) wins.
    pub fn add_builtin(&mut self, name: ProgramName, component: &'static [u8]) {
        self.index.entry(name.clone()).or_insert(None);
        self.builtin.insert(name, component);
    }

    /// Served from the binary: built in, and not shadowed by a disk copy.
    pub fn is_builtin(&self, name: &ProgramName) -> bool {
        self.builtin.contains_key(name) && !self.shadowed(name)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn shadowed(&self, name: &ProgramName) -> bool {
        self.on_disk.contains_key(name)
    }

    #[cfg(target_arch = "wasm32")]
    fn shadowed(&self, name: &ProgramName) -> bool {
        self.preloaded_binaries.contains_key(name)
    }

    /// The program's artifact bytes: a component, or a script's source.
    pub async fn fetch_binary(&mut self, name: &ProgramName) -> Result<Vec<u8>> {
        #[cfg(target_arch = "wasm32")]
        if let Some(binary) = self.preloaded_binaries.get(name) {
            return Ok(binary.clone());
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(binary) = self.preloaded_binaries.remove(name) {
            return Ok(binary);
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(language) = self.index.get(name).filter(|_| self.shadowed(name)) {
            let path = artifact_path(&self.programs_dir, name, artifact_extension(*language));
            let binary = tokio::fs::read(&path)
                .await
                .map_err(|e| anyhow!("reading {}: {e}", path.display()))?;
            return Ok(binary);
        }
        if let Some(component) = self.builtin.get(name) {
            return Ok(component.to_vec());
        }

        bail!("{}", self.not_installed(name))
    }

    /// The sentence for a program this repository does not hold.
    pub fn not_installed(&self, name: &ProgramName) -> String {
        format!(
            "program {name} is not installed (looked in {})",
            self.programs_dir.display()
        )
    }

    pub fn cached(&self) -> Vec<(ProgramName, Option<Language>, u64)> {
        let mut out: Vec<(ProgramName, Option<Language>, u64)> = self
            .index
            .iter()
            .map(|(name, language)| {
                #[cfg(not(target_arch = "wasm32"))]
                let size = match self.builtin.get(name).filter(|_| !self.shadowed(name)) {
                    Some(component) => component.len() as u64,
                    None => std::fs::metadata(artifact_path(
                        &self.programs_dir,
                        name,
                        artifact_extension(*language),
                    ))
                    .map(|m| m.len())
                    .unwrap_or(0),
                };
                #[cfg(target_arch = "wasm32")]
                let size = self
                    .preloaded_binaries
                    .get(name)
                    .map_or(0, |bytes| bytes.len() as u64);
                (name.clone(), *language, size)
            })
            .collect();
        out.sort_by(|a, b| (&a.0.name, &a.0.version).cmp(&(&b.0.name, &b.0.version)));
        out
    }

    #[cfg(target_arch = "wasm32")]
    pub fn remove(&mut self, name: &ProgramName) -> Result<bool> {
        if self.is_builtin(name) {
            bail!("{name} is built into this pie; it cannot be removed");
        }
        let known = self.index.remove(name).is_some();
        let removed = self.preloaded_binaries.remove(name).is_some() || known;
        self.restore_builtin(name);
        Ok(removed)
    }

    /// Removing a disk copy uncovers the built-in it shadowed.
    fn restore_builtin(&mut self, name: &ProgramName) {
        if self.builtin.contains_key(name) {
            self.index.insert(name.clone(), None);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn remove(&mut self, name: &ProgramName) -> Result<bool> {
        if self.is_builtin(name) {
            bail!("{name} is built into this pie; it cannot be removed");
        }
        let on_disk = ARTIFACT_EXTENSIONS
            .iter()
            .any(|extension| artifact_path(&self.programs_dir, name, extension).exists());
        if self.index.remove(name).is_none() && !on_disk {
            return Ok(false);
        }
        self.preloaded_binaries.remove(name);
        remove_artifacts(&self.programs_dir, name, None)?;
        let _ = std::fs::remove_dir(self.programs_dir.join(&name.name));
        self.on_disk.remove(name);
        self.restore_builtin(name);
        Ok(true)
    }

    pub fn exists(&self, name: &ProgramName) -> bool {
        self.index.contains_key(name)
    }

    /// The newest installed version of `name`, by semantic version order.
    pub fn newest(&self, name: &str) -> Option<ProgramName> {
        self.index
            .keys()
            .filter(|p| p.name == name)
            .max_by_key(|p| semver_key(&p.version))
            .cloned()
    }

    pub async fn add(
        &mut self,
        binary: Vec<u8>,
        name: ProgramName,
        language: Option<Language>,
        force_overwrite: bool,
    ) -> Result<bool> {
        if !force_overwrite && self.index.contains_key(&name) {
            return Ok(false);
        }

        self.store_program_cache(&binary, &name, language).await?;
        self.preloaded_binaries.insert(name, binary);

        Ok(true)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn refresh(&mut self) -> Vec<ProgramName> {
        Vec::new()
    }

    /// Bring the index up to date with the programs directory, so an
    /// install, removal or replacement made while this host runs is seen.
    /// Returns the programs whose artifact changed or went away, whose
    /// compiled form is stale.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn refresh(&mut self) -> Vec<ProgramName> {
        let mut found: HashMap<ProgramName, Vec<(String, Option<SystemTime>)>> = HashMap::new();
        for program_dir in read_dir(&self.programs_dir).filter(|p| p.is_dir()) {
            let Some(program) = program_dir.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            for artifact in read_dir(&program_dir) {
                let (Some(extension), Some(version)) = (
                    artifact.extension().and_then(|e| e.to_str()),
                    artifact.file_stem().and_then(|s| s.to_str()),
                ) else {
                    continue;
                };
                if !ARTIFACT_EXTENSIONS.contains(&extension) {
                    continue;
                }
                let mtime = std::fs::metadata(&artifact).and_then(|m| m.modified()).ok();
                found
                    .entry(ProgramName {
                        name: program.to_string(),
                        version: version.to_string(),
                    })
                    .or_default()
                    .push((extension.to_string(), mtime));
            }
        }

        let mut changed = Vec::new();
        let gone: Vec<ProgramName> = self
            .on_disk
            .keys()
            .filter(|name| !found.contains_key(name))
            .cloned()
            .collect();
        for name in gone {
            self.on_disk.remove(&name);
            self.index.remove(&name);
            self.preloaded_binaries.remove(&name);
            self.restore_builtin(&name);
            changed.push(name);
        }
        for (name, artifacts) in found {
            let Some((extension, mtime)) = artifacts.iter().max_by_key(|(_, mtime)| *mtime) else {
                continue;
            };
            let language = Language::from_extension(extension);
            match self.on_disk.get(&name) {
                Some(seen) if seen == mtime && self.index.get(&name) == Some(&language) => {
                    continue;
                }
                Some(_) => {
                    self.preloaded_binaries.remove(&name);
                    changed.push(name.clone());
                }
                None => {}
            }
            self.on_disk.insert(name.clone(), *mtime);
            self.index.insert(name, language);
        }
        changed
    }

    #[cfg(target_arch = "wasm32")]
    async fn store_program_cache(
        &mut self,
        _binary: &[u8],
        name: &ProgramName,
        language: Option<Language>,
    ) -> Result<()> {
        self.index.insert(name.clone(), language);
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn store_program_cache(
        &mut self,
        binary: &[u8],
        name: &ProgramName,
        language: Option<Language>,
    ) -> Result<()> {
        let dir = self.programs_dir.join(&name.name);
        let extension = artifact_extension(language);
        let artifact = artifact_path(&self.programs_dir, name, extension);

        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| anyhow!("Failed to create directory {:?}: {}", dir, e))?;

        tokio::fs::write(&artifact, binary)
            .await
            .map_err(|e| anyhow!("writing {}: {e}", artifact.display()))?;
        remove_artifacts(&self.programs_dir, name, Some(extension))?;

        let mtime = std::fs::metadata(&artifact).and_then(|m| m.modified()).ok();
        self.on_disk.insert(name.clone(), mtime);
        self.index.insert(name.clone(), language);

        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn remove_artifacts(programs_dir: &Path, name: &ProgramName, keep: Option<&str>) -> Result<()> {
    for extension in ARTIFACT_EXTENSIONS {
        if Some(extension) == keep {
            continue;
        }
        let path = artifact_path(programs_dir, name, extension);
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(anyhow!("removing {:?}: {}", path, e)),
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn read_dir(dir: &Path) -> impl Iterator<Item = PathBuf> {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn named(name: &str, version: &str) -> ProgramName {
        ProgramName::parse(&format!("{name}@{version}")).unwrap()
    }

    #[tokio::test]
    async fn repository_every_case() {
        a_script_is_stored_as_its_source_and_found_again_on_reload().await;
        a_version_is_one_artifact_and_the_newer_kind_replaces_the_older().await;
        a_bare_name_resolves_to_its_newest_version().await;
        a_builtin_is_listed_served_and_shadowed_but_not_removed().await;
        a_running_host_sees_installs_removals_and_replacements().await;
    }

    async fn a_running_host_sees_installs_removals_and_replacements() {
        let dir = tempfile::tempdir().unwrap();
        let mut host = Repository::new(dir.path().to_path_buf());
        host.refresh();
        let name = named("late", "0.1.0");
        assert!(!host.exists(&name));

        // `pie inferlet install`, from another process.
        let mut cli = Repository::new(dir.path().to_path_buf());
        cli.add(b"\0asm-one".to_vec(), name.clone(), None, true)
            .await
            .unwrap();
        assert!(
            host.refresh().is_empty(),
            "a new program is not a stale one"
        );
        assert!(host.exists(&name));
        assert_eq!(host.fetch_binary(&name).await.unwrap(), b"\0asm-one");

        // A replacement of the same version is reported, so its compiled
        // form is dropped.
        std::thread::sleep(std::time::Duration::from_millis(20));
        cli.add(b"\0asm-two".to_vec(), name.clone(), None, true)
            .await
            .unwrap();
        let path = dir.path().join("late").join("0.1.0.wasm");
        let later = std::time::SystemTime::now() + std::time::Duration::from_secs(2);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(later)
            .unwrap();
        assert_eq!(host.refresh(), vec![name.clone()]);
        assert_eq!(host.fetch_binary(&name).await.unwrap(), b"\0asm-two");
        assert!(host.refresh().is_empty());

        // `pie inferlet remove`, from another process.
        assert!(cli.remove(&name).unwrap());
        assert_eq!(host.refresh(), vec![name.clone()]);
        assert!(!host.exists(&name));
    }

    async fn a_builtin_is_listed_served_and_shadowed_but_not_removed() {
        let dir = tempfile::tempdir().unwrap();
        let mut repo = Repository::new(dir.path().to_path_buf());
        let name = named("compat", "0.1.0");
        repo.add_builtin(name.clone(), b"\0asm-builtin");
        assert!(repo.exists(&name) && repo.is_builtin(&name));
        assert_eq!(repo.fetch_binary(&name).await.unwrap(), b"\0asm-builtin");
        assert_eq!(repo.cached()[0].2, b"\0asm-builtin".len() as u64);
        assert!(
            repo.remove(&name)
                .unwrap_err()
                .to_string()
                .contains("built into")
        );
        assert!(
            !dir.path().join("compat").exists(),
            "a built-in touches no disk"
        );

        // The same name and version installed on disk takes over.
        repo.add(b"\0asm-disk".to_vec(), name.clone(), None, true)
            .await
            .unwrap();
        assert!(!repo.is_builtin(&name));
        // Once from the bytes just added, once from the file they went to.
        assert_eq!(repo.fetch_binary(&name).await.unwrap(), b"\0asm-disk");
        assert_eq!(repo.fetch_binary(&name).await.unwrap(), b"\0asm-disk");
        assert!(repo.remove(&name).unwrap());
        // Removing the copy uncovers the built-in again.
        assert!(repo.is_builtin(&name));
        assert_eq!(repo.fetch_binary(&name).await.unwrap(), b"\0asm-builtin");

        // And a disk install already there at boot is not replaced.
        let mut repo = Repository::new(dir.path().to_path_buf());
        repo.add(b"\0asm-disk".to_vec(), name.clone(), None, true)
            .await
            .unwrap();
        repo.add_builtin(name.clone(), b"\0asm-builtin");
        assert!(!repo.is_builtin(&name));
    }

    async fn a_bare_name_resolves_to_its_newest_version() {
        let dir = tempfile::tempdir().unwrap();
        let mut repo = Repository::new(dir.path().to_path_buf());
        for version in ["0.9.0", "0.10.0", "0.2.5"] {
            repo.add(
                b"async def main(i): return i".to_vec(),
                named("probe", version),
                Some(Language::Python),
                false,
            )
            .await
            .unwrap();
        }
        assert_eq!(
            repo.newest("probe").map(|p| p.version).as_deref(),
            Some("0.10.0")
        );
        assert_eq!(repo.newest("nobody"), None);
        assert_eq!(semver_key("1.2.3"), (1, 2, 3));
        assert_eq!(semver_key("x"), (0, 0, 0));
    }

    async fn a_script_is_stored_as_its_source_and_found_again_on_reload() {
        let dir = tempfile::tempdir().unwrap();
        let mut repo = Repository::new(dir.path().to_path_buf());
        let name = named("probe", "0.1.0");
        assert!(
            repo.add(
                b"def main(input): return input\n".to_vec(),
                name.clone(),
                Some(Language::Python),
                false,
            )
            .await
            .unwrap()
        );
        assert!(
            !repo
                .add(b"x".to_vec(), name.clone(), Some(Language::Python), false)
                .await
                .unwrap(),
            "a held name is left alone"
        );
        assert!(dir.path().join("probe").join("0.1.0.py").is_file());
        assert!(!dir.path().join("probe").join("0.1.0.wasm").exists());

        let mut reloaded = Repository::new(dir.path().to_path_buf());
        reloaded.refresh();
        assert!(reloaded.exists(&name));
        let source = reloaded.fetch_binary(&name).await.unwrap();
        assert_eq!(source, b"def main(input): return input\n");
        assert_eq!(reloaded.language(&name), Some(Some(Language::Python)));

        assert!(reloaded.remove(&name).unwrap());
        assert!(!dir.path().join("probe").join("0.1.0.py").exists());
        assert!(!reloaded.exists(&name));
    }

    async fn a_version_is_one_artifact_and_the_newer_kind_replaces_the_older() {
        let dir = tempfile::tempdir().unwrap();
        let mut repo = Repository::new(dir.path().to_path_buf());
        let name = named("twin", "0.1.0");
        repo.add(b"\0asm".to_vec(), name.clone(), None, false)
            .await
            .unwrap();
        assert_eq!(repo.language(&name), Some(None));
        repo.add(b"x".to_vec(), name.clone(), Some(Language::Python), true)
            .await
            .unwrap();
        assert!(!dir.path().join("twin").join("0.1.0.wasm").exists());
        assert_eq!(repo.language(&name), Some(Some(Language::Python)));

        let mut reloaded = Repository::new(dir.path().to_path_buf());
        reloaded.refresh();
        let cached = reloaded.cached();
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].1, Some(Language::Python));
        assert_eq!(cached[0].2, 1, "the script's size");
    }
}
