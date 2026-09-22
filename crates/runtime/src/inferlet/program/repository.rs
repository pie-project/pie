use std::collections::HashMap;
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::time::SystemTime;

#[cfg(not(target_arch = "wasm32"))]
use anyhow::anyhow;
use anyhow::{Result, bail};

use super::ProgramName;
use super::manifest::Manifest;

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

#[cfg(not(target_arch = "wasm32"))]
fn manifest_path(programs_dir: &Path, name: &ProgramName) -> PathBuf {
    programs_dir
        .join(&name.name)
        .join(format!("{}.toml", name.version))
}

/// The programs this host can run: a local directory of
/// `<name>/<version>.{wasm,py,js}` + `<version>.toml` pairs, the built-in
/// inferlets the binary carries, and whatever was added from bytes. Nothing
/// is fetched; a program that is not here is an error that names where it
/// was expected.
pub struct Repository {
    index: HashMap<ProgramName, Manifest>,
    preloaded_binaries: HashMap<ProgramName, Vec<u8>>,
    /// What the binary carries: not removable, and shadowed by a disk copy
    /// of the same name and version.
    builtin: HashMap<ProgramName, (Manifest, &'static [u8])>,
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

    pub fn fetch_manifest(&self, name: &ProgramName) -> Option<Manifest> {
        self.index.get(name).cloned()
    }

    /// A copy of the same name and version already here (installed on
    /// disk) wins.
    pub fn add_builtin(&mut self, manifest: Manifest, component: &'static [u8]) {
        let name = manifest.program_name();
        self.index
            .entry(name.clone())
            .or_insert_with(|| manifest.clone());
        self.builtin.insert(name, (manifest, component));
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
        if let Some(manifest) = self.index.get(name).filter(|_| self.shadowed(name)) {
            let path = artifact_path(&self.programs_dir, name, manifest.artifact_extension());
            let binary = tokio::fs::read(&path)
                .await
                .map_err(|e| anyhow!("reading {}: {e}", path.display()))?;
            return Ok(binary);
        }
        if let Some((_, component)) = self.builtin.get(name) {
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

    pub fn cached(&self) -> Vec<(ProgramName, Manifest, u64)> {
        let mut out: Vec<(ProgramName, Manifest, u64)> = self
            .index
            .iter()
            .map(|(name, manifest)| {
                #[cfg(not(target_arch = "wasm32"))]
                let size = match self.builtin.get(name).filter(|_| !self.shadowed(name)) {
                    Some((_, component)) => component.len() as u64,
                    None => std::fs::metadata(artifact_path(
                        &self.programs_dir,
                        name,
                        manifest.artifact_extension(),
                    ))
                    .map(|m| m.len())
                    .unwrap_or(0),
                };
                #[cfg(target_arch = "wasm32")]
                let size = self
                    .preloaded_binaries
                    .get(name)
                    .map_or(0, |bytes| bytes.len() as u64);
                (name.clone(), manifest.clone(), size)
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
        if let Some((manifest, _)) = self.builtin.get(name) {
            self.index.insert(name.clone(), manifest.clone());
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
        let mut paths = vec![manifest_path(&self.programs_dir, name)];
        paths.extend(
            ARTIFACT_EXTENSIONS
                .iter()
                .map(|extension| artifact_path(&self.programs_dir, name, extension)),
        );
        for path in paths {
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(anyhow!("removing {:?}: {}", path, e)),
            }
        }
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
        wasm_binary: Vec<u8>,
        manifest: Manifest,
        force_overwrite: bool,
    ) -> Result<()> {
        let name = manifest.program_name();

        if !force_overwrite && self.index.contains_key(&name) {
            return Ok(());
        }

        self.store_program_cache(&wasm_binary, manifest).await?;
        self.preloaded_binaries.insert(name, wasm_binary);

        Ok(())
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
            // The manifest says what the artifact is; a `.py` beside a
            // manifest that claims a component (or the reverse) is not this
            // program.
            let Some(manifest) = std::fs::read_to_string(manifest_path(&self.programs_dir, &name))
                .ok()
                .and_then(|text| Manifest::parse(&text).ok())
            else {
                continue;
            };
            let Some((_, mtime)) = artifacts
                .iter()
                .find(|(extension, _)| extension == manifest.artifact_extension())
            else {
                continue;
            };
            match self.on_disk.get(&name) {
                Some(seen) if seen == mtime => continue,
                Some(_) => {
                    self.preloaded_binaries.remove(&name);
                    changed.push(name.clone());
                }
                None => {}
            }
            self.on_disk.insert(name.clone(), *mtime);
            self.index.insert(name, manifest);
        }
        changed
    }

    #[cfg(target_arch = "wasm32")]
    async fn store_program_cache(&mut self, _wasm_binary: &[u8], manifest: Manifest) -> Result<()> {
        self.index.insert(manifest.program_name(), manifest);
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn store_program_cache(&mut self, binary: &[u8], manifest: Manifest) -> Result<()> {
        let name = manifest.program_name();
        let dir = self.programs_dir.join(&name.name);
        let artifact = artifact_path(&self.programs_dir, &name, manifest.artifact_extension());
        let manifest_file = manifest_path(&self.programs_dir, &name);

        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| anyhow!("Failed to create directory {:?}: {}", dir, e))?;

        tokio::fs::write(&artifact, binary)
            .await
            .map_err(|e| anyhow!("writing {}: {e}", artifact.display()))?;
        tokio::fs::write(&manifest_file, manifest.to_toml()?)
            .await
            .map_err(|e| anyhow!("Failed to write manifest file: {}", e))?;

        let mtime = std::fs::metadata(&artifact).and_then(|m| m.modified()).ok();
        self.on_disk.insert(name.clone(), mtime);
        self.index.insert(name, manifest);

        Ok(())
    }
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
    use crate::inferlet::program::Language;

    fn manifest(toml: &str) -> Manifest {
        Manifest::parse(toml).unwrap()
    }

    #[tokio::test]
    async fn repository_every_case() {
        a_script_is_stored_as_its_source_and_found_again_on_reload().await;
        a_component_and_a_script_of_one_name_do_not_shadow_each_other().await;
        an_unknown_language_is_refused_at_add().await;
        a_bare_name_resolves_to_its_newest_version().await;
        a_builtin_is_listed_served_and_shadowed_but_not_removed().await;
        a_running_host_sees_installs_removals_and_replacements().await;
    }

    async fn a_running_host_sees_installs_removals_and_replacements() {
        let dir = tempfile::tempdir().unwrap();
        let mut host = Repository::new(dir.path().to_path_buf());
        host.refresh();
        let manifest = manifest("[package]\nname = \"late\"\nversion = \"0.1.0\"\n");
        let name = manifest.program_name();
        assert!(!host.exists(&name));

        // `pie inferlet install`, from another process.
        let mut cli = Repository::new(dir.path().to_path_buf());
        cli.add(b"\0asm-one".to_vec(), manifest.clone(), true)
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
        cli.add(b"\0asm-two".to_vec(), manifest.clone(), true)
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
        let manifest = manifest("[package]\nname = \"compat\"\nversion = \"0.1.0\"\n");
        let name = manifest.program_name();
        repo.add_builtin(manifest.clone(), b"\0asm-builtin");
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
        repo.add(b"\0asm-disk".to_vec(), manifest.clone(), true)
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
        repo.add(b"\0asm-disk".to_vec(), manifest.clone(), true)
            .await
            .unwrap();
        repo.add_builtin(manifest, b"\0asm-builtin");
        assert!(!repo.is_builtin(&name));
    }

    async fn a_bare_name_resolves_to_its_newest_version() {
        let dir = tempfile::tempdir().unwrap();
        let mut repo = Repository::new(dir.path().to_path_buf());
        for version in ["0.9.0", "0.10.0", "0.2.5"] {
            let manifest = manifest(&format!(
                "[package]\nname = \"probe\"\nversion = \"{version}\"\n[runtime]\nlanguage = \"python\"\n"
            ));
            repo.add(b"async def main(i): return i".to_vec(), manifest, false)
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
        let manifest = manifest(
            "[package]\nname = \"probe\"\nversion = \"0.1.0\"\n[runtime]\nlanguage = \"python\"\nentry = \"go\"\n",
        );
        let name = manifest.program_name();
        repo.add(b"def go(input): return input\n".to_vec(), manifest, false)
            .await
            .unwrap();
        assert!(dir.path().join("probe").join("0.1.0.py").is_file());
        assert!(!dir.path().join("probe").join("0.1.0.wasm").exists());

        let mut reloaded = Repository::new(dir.path().to_path_buf());
        reloaded.refresh();
        assert!(reloaded.exists(&name));
        let source = reloaded.fetch_binary(&name).await.unwrap();
        assert_eq!(source, b"def go(input): return input\n");
        let found = reloaded.fetch_manifest(&name).unwrap();
        assert_eq!(found.language(), Some(Language::Python));
        assert_eq!(found.entry(), "go");
        assert_eq!(found.script_file(), "probe.py");

        assert!(reloaded.remove(&name).unwrap());
        assert!(!dir.path().join("probe").join("0.1.0.py").exists());
        assert!(!reloaded.exists(&name));
    }

    async fn a_component_and_a_script_of_one_name_do_not_shadow_each_other() {
        let dir = tempfile::tempdir().unwrap();
        let mut repo = Repository::new(dir.path().to_path_buf());
        let wasm = manifest("[package]\nname = \"twin\"\nversion = \"0.1.0\"\n");
        repo.add(b"\0asm".to_vec(), wasm, false).await.unwrap();
        // A stray source file beside a component manifest is not the program.
        std::fs::write(dir.path().join("twin").join("0.1.0.py"), "x").unwrap();
        let mut reloaded = Repository::new(dir.path().to_path_buf());
        reloaded.refresh();
        let cached = reloaded.cached();
        assert_eq!(cached.len(), 1);
        assert_eq!(
            cached[0].2, 4,
            "the component's size, not the stray source's"
        );
    }

    async fn an_unknown_language_is_refused_at_add() {
        let err = Manifest::parse(
            "[package]\nname = \"x\"\nversion = \"0.1.0\"\n[runtime]\nlanguage = \"cobol\"\n",
        )
        .unwrap_err();
        assert!(err.to_string().contains("cobol"), "{err}");
    }
}
