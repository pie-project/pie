//! Where programs are kept: their bytes once each, by hash, and the names
//! each namespace installed them under.
//!
//! On disk, under the programs directory:
//!
//! ```text
//! blobs/<hash>.<wasm|py|js>          the bytes, the extension their kind
//! system/<name>/<hash>               an install by the operator
//! tenants/<tenant>/<name>/<hash>     an install by a tenant
//! ```
//!
//! An install's file holds when it was made, and a name runs its latest
//! install. Nothing is ever overwritten: other bytes are another blob, and
//! installing a build again only makes it the latest. The disk is read as
//! it stands, so what another process (`pie inferlet install`, `remove`,
//! `prune`) did is seen at once. A host with no files (the browser) keeps
//! the same in memory.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use anyhow::anyhow;
use anyhow::{Result, bail};

use super::identity::{Identity, Namespace, Program, ProgramHash, ProgramRef};
use super::language::{Language, kind};

/// A blob no install names is kept this long before a removal reclaims it,
/// so one an install is still writing, or reusing, is not taken from under it.
#[cfg(not(target_arch = "wasm32"))]
const BLOB_GRACE: std::time::Duration = std::time::Duration::from_secs(60);

/// One install of a build under a name. `at` orders a name's installs:
/// nanoseconds since the epoch, and always past the name's latest before.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Install {
    hash: ProgramHash,
    at: u64,
}

/// A program as `pie inferlet list` shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Listed {
    pub namespace: Namespace,
    pub program: Program,
    pub language: Option<Language>,
    pub bytes: u64,
    pub builtin: bool,
    /// What the bare name runs, in its namespace.
    pub current: bool,
}

/// What a prune or removal reclaimed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reclaimed {
    pub installs: Vec<(Namespace, Program)>,
    pub blobs: usize,
    pub bytes: u64,
}

pub struct Repository {
    root: PathBuf,
    /// The binary's own programs, by name: the system's, under whatever it
    /// installs of the same name.
    builtins: BTreeMap<String, (ProgramHash, &'static [u8])>,
    #[cfg(target_arch = "wasm32")]
    memory: memory::Memory,
}

impl Repository {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            builtins: BTreeMap::new(),
            #[cfg(target_arch = "wasm32")]
            memory: memory::Memory::default(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn add_builtin(&mut self, name: &str, component: &'static [u8]) {
        self.builtins
            .insert(name.to_string(), (ProgramHash::of(component), component));
    }

    pub fn is_builtin_name(&self, name: &str) -> bool {
        self.builtins.contains_key(name)
    }

    /// Whether `identity` may be installed in `namespace`: a tenant cannot
    /// take a built-in's name (the routes that run it would run its own
    /// instead), and one blob is one kind of program.
    pub fn admit(&self, namespace: &Namespace, identity: &Identity) -> Result<()> {
        let Identity { program, language } = identity;
        if matches!(namespace, Namespace::Tenant(_)) && self.is_builtin_name(&program.name) {
            bail!(
                "{:?} is the name of a built-in program; install under another name",
                program.name
            );
        }
        if let Some(held) = self.language(&program.hash)
            && held != *language
        {
            bail!(
                "these bytes are already installed as a {}; one blob is one kind of program",
                kind(held)
            );
        }
        Ok(())
    }

    /// Install `bytes` under their name in `namespace`, as its latest build.
    pub async fn add(
        &mut self,
        namespace: Namespace,
        identity: &Identity,
        bytes: Vec<u8>,
    ) -> Result<()> {
        self.admit(&namespace, identity)?;
        let program = &identity.program;
        let after = self
            .builds(&namespace, &program.name)
            .last()
            .map_or(0, |latest| latest.at + 1);
        self.store_blob(&program.hash, identity.language, bytes)
            .await?;
        self.store_install(&namespace, program, now().max(after))
            .await
    }

    /// The program `reference` names for `tenant` (`None`: the system
    /// alone): its own installs first, then the system's, then the
    /// built-ins. A bare name is the first of these to hold it, at its
    /// latest install; a pinned one is the build whose hash it starts.
    pub fn resolve(&self, tenant: Option<&str>, reference: &ProgramRef) -> Result<Program> {
        let program = |hash: ProgramHash| Program {
            name: reference.name.clone(),
            hash,
        };
        let tiers = self.visible(tenant, &reference.name);
        let mut found: Vec<Program> = Vec::new();
        if reference.hash.is_none() {
            found.extend(
                tiers
                    .into_iter()
                    .find_map(|tier| tier.last().cloned())
                    .map(program),
            );
        } else {
            for candidate in tiers.into_iter().flatten().map(program) {
                if reference.matches(&candidate) && !found.contains(&candidate) {
                    found.push(candidate);
                }
            }
        }
        match found.len() {
            1 => Ok(found.remove(0)),
            0 => bail!("{}", not_installed(reference)),
            n => bail!("{}", ambiguous(reference, n)),
        }
    }

    /// The builds `tenant` sees under `name`, a namespace at a time in the
    /// order it looks, each oldest first.
    fn visible(&self, tenant: Option<&str>, name: &str) -> Vec<Vec<ProgramHash>> {
        let tenant = tenant.map(|t| Namespace::Tenant(t.to_string()));
        let mut tiers: Vec<Vec<ProgramHash>> = tenant
            .iter()
            .chain([&Namespace::System])
            .map(|ns| self.builds(ns, name).into_iter().map(|i| i.hash).collect())
            .collect();
        if let Some((hash, _)) = self.builtins.get(name) {
            tiers.push(vec![hash.clone()]);
        }
        tiers
    }

    /// The kind of program a blob is, if it is here.
    pub fn language(&self, hash: &ProgramHash) -> Option<Option<Language>> {
        if self.builtins.values().any(|(h, _)| h == hash) {
            return Some(None);
        }
        self.blob_facts(hash).map(|(language, _)| language)
    }

    /// A blob's bytes: a component, or a script's source.
    pub async fn fetch(&self, hash: &ProgramHash) -> Result<(Vec<u8>, Option<Language>)> {
        if let Some((_, bytes)) = self.builtins.values().find(|(h, _)| h == hash) {
            return Ok((bytes.to_vec(), None));
        }
        self.read_blob(hash).await
    }

    /// Every hash something runs under a name: an install, or a built-in.
    pub fn named(&self) -> HashSet<ProgramHash> {
        self.installs()
            .into_iter()
            .flat_map(|(_, _, builds)| builds)
            .map(|install| install.hash)
            .chain(self.builtins.values().map(|(hash, _)| hash.clone()))
            .collect()
    }

    /// Everything installed, and the built-ins, by namespace and name; a
    /// name's latest build first.
    pub fn list(&self) -> Vec<Listed> {
        let mut out = Vec::new();
        let mut system_names = HashSet::new();
        for (namespace, name, builds) in self.installs() {
            if namespace == Namespace::System {
                system_names.insert(name.clone());
            }
            for (i, install) in builds.into_iter().rev().enumerate() {
                let (language, bytes) = self.blob_facts(&install.hash).unwrap_or((None, 0));
                out.push(Listed {
                    namespace: namespace.clone(),
                    program: Program {
                        name: name.clone(),
                        hash: install.hash,
                    },
                    language,
                    bytes,
                    builtin: false,
                    current: i == 0,
                });
            }
        }
        for (name, (hash, bytes)) in &self.builtins {
            out.push(Listed {
                namespace: Namespace::System,
                program: Program {
                    name: name.clone(),
                    hash: hash.clone(),
                },
                language: None,
                bytes: bytes.len() as u64,
                builtin: true,
                current: !system_names.contains(name),
            });
        }
        out
    }

    /// Remove the installs of `namespace` that `reference` names: one build
    /// when it pins one, else every build of the name. Blobs no install
    /// names any more go with them.
    pub fn remove(&mut self, namespace: &Namespace, reference: &ProgramRef) -> Result<Reclaimed> {
        let removed: Vec<Program> = self
            .builds(namespace, &reference.name)
            .into_iter()
            .map(|install| Program {
                name: reference.name.clone(),
                hash: install.hash,
            })
            .filter(|program| reference.matches(program))
            .collect();
        if removed.is_empty() {
            if *namespace == Namespace::System && self.is_builtin_name(&reference.name) {
                bail!("{reference} is built into this pie; it cannot be removed");
            }
            bail!("{}", not_installed(reference));
        }
        if reference.hash.is_some() && removed.len() > 1 {
            bail!("{}", ambiguous(reference, removed.len()));
        }
        let removed: Vec<(Namespace, Program)> = removed
            .into_iter()
            .map(|program| (namespace.clone(), program))
            .collect();
        self.reclaim(removed)
    }

    /// Keep each name's latest build alone, and the blobs those name.
    pub fn prune(&mut self) -> Result<Reclaimed> {
        let stale = self
            .installs()
            .into_iter()
            .flat_map(|(namespace, name, mut builds)| {
                builds.pop();
                builds.into_iter().map(move |install| {
                    let program = Program {
                        name: name.clone(),
                        hash: install.hash,
                    };
                    (namespace.clone(), program)
                })
            })
            .collect();
        self.reclaim(stale)
    }

    fn reclaim(&mut self, installs: Vec<(Namespace, Program)>) -> Result<Reclaimed> {
        for (namespace, program) in &installs {
            self.remove_install(namespace, program)?;
        }
        let named = self.named();
        let mut reclaimed = self.collect_blobs(&named)?;
        reclaimed.installs = installs;
        Ok(reclaimed)
    }
}

fn not_installed(reference: &ProgramRef) -> String {
    format!("no program {reference} is installed")
}

fn ambiguous(reference: &ProgramRef, builds: usize) -> String {
    format!("{reference} is ambiguous: {builds} builds start with that hash; give more of it")
}

/// Oldest first.
fn sort(builds: &mut [Install]) {
    builds.sort_by(|a, b| (a.at, &a.hash).cmp(&(b.at, &b.hash)));
}

/// Where there is no clock, installs are ordered by arrival alone.
#[cfg(target_arch = "wasm32")]
fn now() -> u64 {
    0
}

#[cfg(not(target_arch = "wasm32"))]
fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

#[cfg(target_arch = "wasm32")]
mod memory {
    use std::collections::{BTreeMap, HashMap};

    use super::{Install, Language, Namespace, ProgramHash};

    #[derive(Default)]
    pub(super) struct Memory {
        pub installs: BTreeMap<Namespace, BTreeMap<String, Vec<Install>>>,
        pub blobs: HashMap<ProgramHash, (Vec<u8>, Option<Language>)>,
    }
}

#[cfg(target_arch = "wasm32")]
impl Repository {
    fn builds(&self, namespace: &Namespace, name: &str) -> Vec<Install> {
        let mut builds = self
            .memory
            .installs
            .get(namespace)
            .and_then(|names| names.get(name))
            .cloned()
            .unwrap_or_default();
        sort(&mut builds);
        builds
    }

    fn installs(&self) -> Vec<(Namespace, String, Vec<Install>)> {
        let mut out = Vec::new();
        for (namespace, names) in &self.memory.installs {
            for name in names.keys() {
                out.push((
                    namespace.clone(),
                    name.clone(),
                    self.builds(namespace, name),
                ));
            }
        }
        out
    }

    async fn store_blob(
        &mut self,
        hash: &ProgramHash,
        language: Option<Language>,
        bytes: Vec<u8>,
    ) -> Result<()> {
        self.memory.blobs.insert(hash.clone(), (bytes, language));
        Ok(())
    }

    async fn store_install(
        &mut self,
        namespace: &Namespace,
        program: &Program,
        at: u64,
    ) -> Result<()> {
        let builds = self
            .memory
            .installs
            .entry(namespace.clone())
            .or_default()
            .entry(program.name.clone())
            .or_default();
        builds.retain(|install| install.hash != program.hash);
        builds.push(Install {
            hash: program.hash.clone(),
            at,
        });
        Ok(())
    }

    fn remove_install(&mut self, namespace: &Namespace, program: &Program) -> Result<()> {
        if let Some(names) = self.memory.installs.get_mut(namespace) {
            if let Some(builds) = names.get_mut(&program.name) {
                builds.retain(|install| install.hash != program.hash);
                if builds.is_empty() {
                    names.remove(&program.name);
                }
            }
            if names.is_empty() {
                self.memory.installs.remove(namespace);
            }
        }
        Ok(())
    }

    fn blob_facts(&self, hash: &ProgramHash) -> Option<(Option<Language>, u64)> {
        let (bytes, language) = self.memory.blobs.get(hash)?;
        Some((*language, bytes.len() as u64))
    }

    async fn read_blob(&self, hash: &ProgramHash) -> Result<(Vec<u8>, Option<Language>)> {
        match self.memory.blobs.get(hash) {
            Some(blob) => Ok(blob.clone()),
            None => bail!("no blob {hash}"),
        }
    }

    fn collect_blobs(&mut self, named: &HashSet<ProgramHash>) -> Result<Reclaimed> {
        let mut reclaimed = Reclaimed::default();
        self.memory.blobs.retain(|hash, (bytes, _)| {
            let keep = named.contains(hash);
            if !keep {
                reclaimed.blobs += 1;
                reclaimed.bytes += bytes.len() as u64;
            }
            keep
        });
        Ok(reclaimed)
    }
}

/// The extensions a blob can carry: a component, or a script.
#[cfg(not(target_arch = "wasm32"))]
const BLOB_EXTENSIONS: [&str; 3] = ["wasm", "py", "js"];

/// A tenant's directory name: the tenant itself when it is a plain
/// lowercase word, else `~` and its bytes in hex. No tenant names a path,
/// and none shares another's directory where case does not count.
#[cfg(not(target_arch = "wasm32"))]
fn tenant_dir(tenant: &str) -> String {
    let plain = !tenant.is_empty()
        && !tenant.starts_with('.')
        && tenant.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'-' | b'_' | b'.')
        });
    if plain {
        tenant.to_string()
    } else {
        let hex: String = tenant.bytes().map(|b| format!("{b:02x}")).collect();
        format!("~{hex}")
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn tenant_of_dir(dir: &str) -> Option<String> {
    let Some(hex) = dir.strip_prefix('~') else {
        return Some(dir.to_string());
    };
    let bytes = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(hex.get(i..i + 2)?, 16).ok())
        .collect::<Option<Vec<u8>>>()?;
    String::from_utf8(bytes).ok()
}

#[cfg(not(target_arch = "wasm32"))]
impl Repository {
    fn blob_dir(&self) -> PathBuf {
        self.root.join("blobs")
    }

    fn namespace_dir(&self, namespace: &Namespace) -> PathBuf {
        match namespace {
            Namespace::System => self.root.join("system"),
            Namespace::Tenant(tenant) => self.root.join("tenants").join(tenant_dir(tenant)),
        }
    }

    fn install_path(&self, namespace: &Namespace, program: &Program) -> PathBuf {
        self.namespace_dir(namespace)
            .join(&program.name)
            .join(program.hash.as_str())
    }

    fn blob_path(&self, hash: &ProgramHash) -> Option<(PathBuf, Option<Language>)> {
        BLOB_EXTENSIONS.iter().find_map(|extension| {
            let path = self.blob_dir().join(format!("{hash}.{extension}"));
            path.is_file()
                .then(|| (path, Language::from_extension(extension)))
        })
    }

    /// A name's installs in a namespace, read from its directory.
    fn builds(&self, namespace: &Namespace, name: &str) -> Vec<Install> {
        let mut builds: Vec<Install> = read_dir(&self.namespace_dir(namespace).join(name))
            .filter_map(|path| {
                let hash = file_name(&path).and_then(ProgramHash::parse)?;
                let at = std::fs::read_to_string(&path).ok()?.trim().parse().ok()?;
                Some(Install { hash, at })
            })
            .collect();
        sort(&mut builds);
        builds
    }

    fn installs(&self) -> Vec<(Namespace, String, Vec<Install>)> {
        let mut namespaces = vec![Namespace::System];
        for dir in read_dir(&self.root.join("tenants")) {
            if let Some(tenant) = file_name(&dir).and_then(tenant_of_dir) {
                namespaces.push(Namespace::Tenant(tenant));
            }
        }
        namespaces.sort();
        let mut out = Vec::new();
        for namespace in namespaces {
            let mut names: Vec<String> = read_dir(&self.namespace_dir(&namespace))
                .filter_map(|dir| file_name(&dir).map(str::to_string))
                .collect();
            names.sort();
            for name in names {
                let builds = self.builds(&namespace, &name);
                if !builds.is_empty() {
                    out.push((namespace.clone(), name, builds));
                }
            }
        }
        out
    }

    /// Write the blob unless it is here; if it is, make it young again, so
    /// a removal elsewhere does not reclaim it before its install lands.
    async fn store_blob(
        &mut self,
        hash: &ProgramHash,
        language: Option<Language>,
        bytes: Vec<u8>,
    ) -> Result<()> {
        if let Some((path, _)) = self.blob_path(hash) {
            return std::fs::File::options()
                .write(true)
                .open(&path)
                .and_then(|file| file.set_modified(std::time::SystemTime::now()))
                .map_err(|e| anyhow!("touching {}: {e}", path.display()));
        }
        let dir = self.blob_dir();
        let extension = super::language::artifact_extension(language);
        let path = dir.join(format!("{hash}.{extension}"));
        let partial = dir.join(format!(".{hash}.{}.partial", std::process::id()));
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| anyhow!("creating {}: {e}", dir.display()))?;
        tokio::fs::write(&partial, &bytes)
            .await
            .map_err(|e| anyhow!("writing {}: {e}", partial.display()))?;
        tokio::fs::rename(&partial, &path)
            .await
            .map_err(|e| anyhow!("writing {}: {e}", path.display()))
    }

    async fn store_install(
        &mut self,
        namespace: &Namespace,
        program: &Program,
        at: u64,
    ) -> Result<()> {
        let path = self.install_path(namespace, program);
        let dir = path
            .parent()
            .expect("an install sits in its name's directory");
        tokio::fs::create_dir_all(dir)
            .await
            .map_err(|e| anyhow!("creating {}: {e}", dir.display()))?;
        tokio::fs::write(&path, at.to_string())
            .await
            .map_err(|e| anyhow!("writing {}: {e}", path.display()))
    }

    fn remove_install(&mut self, namespace: &Namespace, program: &Program) -> Result<()> {
        let path = self.install_path(namespace, program);
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(anyhow!("removing {}: {e}", path.display())),
        }
        if let Some(dir) = path.parent() {
            let _ = std::fs::remove_dir(dir);
        }
        Ok(())
    }

    fn blob_facts(&self, hash: &ProgramHash) -> Option<(Option<Language>, u64)> {
        let (path, language) = self.blob_path(hash)?;
        Some((language, std::fs::metadata(path).map_or(0, |m| m.len())))
    }

    async fn read_blob(&self, hash: &ProgramHash) -> Result<(Vec<u8>, Option<Language>)> {
        let (path, language) = self
            .blob_path(hash)
            .ok_or_else(|| anyhow!("no blob {hash} under {}", self.root.display()))?;
        let bytes = tokio::fs::read(&path)
            .await
            .map_err(|e| anyhow!("reading {}: {e}", path.display()))?;
        Ok((bytes, language))
    }

    /// Delete the blobs `named` leaves out, past their grace.
    fn collect_blobs(&mut self, named: &HashSet<ProgramHash>) -> Result<Reclaimed> {
        let mut reclaimed = Reclaimed::default();
        let now = std::time::SystemTime::now();
        for blob in read_dir(&self.blob_dir()) {
            let Some(hash) = file_name(&blob)
                .and_then(|f| f.split_once('.'))
                .and_then(|(hash, _)| ProgramHash::parse(hash))
            else {
                continue;
            };
            if named.contains(&hash) {
                continue;
            }
            let Ok(meta) = std::fs::metadata(&blob) else {
                continue;
            };
            let young = meta
                .modified()
                .ok()
                .and_then(|m| now.duration_since(m).ok())
                .is_none_or(|age| age < BLOB_GRACE);
            if young {
                continue;
            }
            std::fs::remove_file(&blob).map_err(|e| anyhow!("removing {}: {e}", blob.display()))?;
            reclaimed.blobs += 1;
            reclaimed.bytes += meta.len();
        }
        Ok(reclaimed)
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

#[cfg(not(target_arch = "wasm32"))]
fn file_name(path: &Path) -> Option<&str> {
    path.file_name().and_then(|n| n.to_str())
}
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::super::identity::identify;
    use super::*;

    fn tenant(name: &str) -> Namespace {
        Namespace::Tenant(name.to_string())
    }

    fn reference(spec: &str) -> ProgramRef {
        ProgramRef::parse(spec).unwrap()
    }

    async fn install(
        repo: &mut Repository,
        namespace: Namespace,
        file: &str,
        bytes: &[u8],
    ) -> Program {
        let identity = identify(file, bytes).unwrap();
        repo.add(namespace, &identity, bytes.to_vec())
            .await
            .unwrap();
        identity.program
    }

    /// Age every blob past its grace, as if the installs were long ago.
    fn age_blobs(root: &Path) {
        let then = std::time::SystemTime::now() - 2 * BLOB_GRACE;
        for blob in read_dir(&root.join("blobs")) {
            std::fs::File::options()
                .write(true)
                .open(&blob)
                .unwrap()
                .set_modified(then)
                .unwrap();
        }
    }

    #[tokio::test]
    async fn repository_every_case() {
        a_rebuild_is_another_program_and_the_name_runs_the_latest().await;
        tenants_see_their_own_and_the_systems_but_not_each_others().await;
        the_same_bytes_are_one_blob_whoever_installs_them().await;
        a_builtin_is_the_systems_under_what_the_operator_installs().await;
        a_running_host_sees_what_another_process_did().await;
        removing_and_pruning_reclaim_the_blobs_nothing_names().await;
    }

    async fn a_rebuild_is_another_program_and_the_name_runs_the_latest() {
        let dir = tempfile::tempdir().unwrap();
        let mut repo = Repository::new(dir.path().to_path_buf());
        let a = install(&mut repo, tenant("t"), "pico.wasm", b"\0asm-a").await;
        let b = install(&mut repo, tenant("t"), "pico.wasm", b"\0asm-b").await;
        assert_ne!(a.hash, b.hash);
        assert_eq!(repo.resolve(Some("t"), &reference("pico")).unwrap(), b);
        // Each build stays launchable by its hash, a short one or all of it.
        assert_eq!(
            repo.resolve(Some("t"), &reference(&a.to_string())).unwrap(),
            a
        );
        assert_eq!(repo.resolve(Some("t"), &reference(&b.pinned())).unwrap(), b);
        // Installing a build again makes it the one the name runs.
        install(&mut repo, tenant("t"), "pico.wasm", b"\0asm-a").await;
        assert_eq!(repo.resolve(Some("t"), &reference("pico")).unwrap(), a);
        // A new install is the latest even when the clock says otherwise.
        let ahead = (now() + 3_600_000_000_000).to_string();
        std::fs::write(repo.install_path(&tenant("t"), &a), ahead).unwrap();
        install(&mut repo, tenant("t"), "pico.wasm", b"\0asm-b").await;
        assert_eq!(repo.resolve(Some("t"), &reference("pico")).unwrap(), b);
        let err = repo
            .resolve(Some("t"), &reference("pico@00000000"))
            .unwrap_err();
        assert!(err.to_string().contains("no program"), "{err}");
    }

    async fn tenants_see_their_own_and_the_systems_but_not_each_others() {
        let dir = tempfile::tempdir().unwrap();
        let mut repo = Repository::new(dir.path().to_path_buf());
        let alice = install(&mut repo, tenant("alice"), "pico.py", b"alice").await;
        let bob = install(&mut repo, tenant("bob"), "pico.py", b"bob").await;
        let shared = install(&mut repo, Namespace::System, "tool.py", b"tool").await;
        assert_eq!(
            repo.resolve(Some("alice"), &reference("pico")).unwrap(),
            alice
        );
        assert_eq!(repo.resolve(Some("bob"), &reference("pico")).unwrap(), bob);
        // Knowing another tenant's hash does not run its program.
        assert!(
            repo.resolve(Some("bob"), &reference(&alice.pinned()))
                .is_err()
        );
        assert!(repo.resolve(None, &reference("pico")).is_err());
        for who in [Some("alice"), Some("bob"), None] {
            assert_eq!(repo.resolve(who, &reference("tool")).unwrap(), shared);
        }
        // A tenant's own install of a system name is its own to run.
        let own = install(&mut repo, tenant("bob"), "tool.py", b"bob's tool").await;
        assert_eq!(repo.resolve(Some("bob"), &reference("tool")).unwrap(), own);
        assert_eq!(
            repo.resolve(Some("bob"), &reference(&shared.to_string()))
                .unwrap(),
            shared
        );
        assert_eq!(
            repo.resolve(Some("alice"), &reference("tool")).unwrap(),
            shared
        );
        // A tenant whose name is no plain word still gets a directory of
        // its own, and one differing only in case is another tenant.
        let odd = install(&mut repo, tenant("org/a b"), "pico.py", b"odd").await;
        let upper = install(&mut repo, tenant("Alice"), "pico.py", b"Alice").await;
        let reloaded = Repository::new(dir.path().to_path_buf());
        let resolve = |who| reloaded.resolve(Some(who), &reference("pico")).unwrap();
        assert_eq!(resolve("org/a b"), odd);
        assert_eq!(resolve("Alice"), upper);
        assert_eq!(resolve("alice"), alice);
    }

    async fn the_same_bytes_are_one_blob_whoever_installs_them() {
        let dir = tempfile::tempdir().unwrap();
        let mut repo = Repository::new(dir.path().to_path_buf());
        let a = install(&mut repo, tenant("alice"), "pico.wasm", b"\0asm").await;
        let b = install(&mut repo, tenant("bob"), "other_name.wasm", b"\0asm").await;
        assert_eq!(a.hash, b.hash);
        assert_eq!(read_dir(&dir.path().join("blobs")).count(), 1);
        let identity = identify("pico.py", b"\0asm").unwrap();
        let err = repo
            .add(tenant("carol"), &identity, b"\0asm".to_vec())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("component"), "{err}");
        assert_eq!(
            repo.fetch(&a.hash).await.unwrap(),
            (b"\0asm".to_vec(), None)
        );
    }

    async fn a_builtin_is_the_systems_under_what_the_operator_installs() {
        let dir = tempfile::tempdir().unwrap();
        let mut repo = Repository::new(dir.path().to_path_buf());
        repo.add_builtin("compat", b"\0asm-builtin");
        let builtin = repo.resolve(Some("t"), &reference("compat")).unwrap();
        assert_eq!(builtin.hash, ProgramHash::of(b"\0asm-builtin"));
        assert_eq!(repo.fetch(&builtin.hash).await.unwrap().0, b"\0asm-builtin");
        let identity = identify("compat.wasm", b"\0asm-mine").unwrap();
        let err = repo
            .add(tenant("t"), &identity, b"\0asm-mine".to_vec())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("built-in"), "{err}");
        let err = repo
            .remove(&Namespace::System, &reference("compat"))
            .unwrap_err();
        assert!(err.to_string().contains("built into"), "{err}");
        assert!(
            !dir.path().join("blobs").exists(),
            "a built-in touches no disk"
        );

        // The operator's install of the name runs in its place, and its
        // removal brings the built-in back.
        let operator = install(&mut repo, Namespace::System, "compat.wasm", b"\0asm-op").await;
        assert_eq!(
            repo.resolve(Some("t"), &reference("compat")).unwrap(),
            operator
        );
        assert_eq!(
            repo.resolve(Some("t"), &reference(&builtin.pinned()))
                .unwrap(),
            builtin
        );
        let listed = repo.list();
        let current: Vec<bool> = listed.iter().map(|l| l.current).collect();
        assert_eq!(current, [true, false], "{listed:?}");
        repo.remove(&Namespace::System, &reference("compat"))
            .unwrap();
        assert_eq!(
            repo.resolve(Some("t"), &reference("compat")).unwrap(),
            builtin
        );
    }

    async fn a_running_host_sees_what_another_process_did() {
        let dir = tempfile::tempdir().unwrap();
        let host = Repository::new(dir.path().to_path_buf());
        assert!(host.resolve(None, &reference("late")).is_err());

        // `pie inferlet install`, from another process.
        let mut cli = Repository::new(dir.path().to_path_buf());
        let program = install(&mut cli, Namespace::System, "late.py", b"late").await;
        assert_eq!(host.resolve(None, &reference("late")).unwrap(), program);
        assert_eq!(
            host.fetch(&program.hash).await.unwrap(),
            (b"late".to_vec(), Some(Language::Python))
        );

        // `pie inferlet remove`, from another process.
        cli.remove(&Namespace::System, &reference("late")).unwrap();
        assert!(host.resolve(None, &reference("late")).is_err());
    }

    async fn removing_and_pruning_reclaim_the_blobs_nothing_names() {
        let dir = tempfile::tempdir().unwrap();
        let mut repo = Repository::new(dir.path().to_path_buf());
        let old = install(&mut repo, tenant("t"), "pico.wasm", b"\0asm-old").await;
        let new = install(&mut repo, tenant("t"), "pico.wasm", b"\0asm-new").await;
        let kept = install(&mut repo, tenant("u"), "pico.wasm", b"\0asm-old").await;
        age_blobs(dir.path());

        // The old build is gone from `t`, but `u` still names its blob.
        let pruned = repo.prune().unwrap();
        assert_eq!(pruned.installs, vec![(tenant("t"), old.clone())]);
        assert_eq!(pruned.blobs, 0);
        assert!(repo.resolve(Some("t"), &reference(&old.pinned())).is_err());
        assert_eq!(repo.resolve(Some("u"), &reference("pico")).unwrap(), kept);

        let removed = repo.remove(&tenant("u"), &reference("pico")).unwrap();
        assert_eq!(removed.installs, vec![(tenant("u"), kept)]);
        assert_eq!(
            (removed.blobs, removed.bytes),
            (1, b"\0asm-old".len() as u64)
        );
        assert_eq!(repo.resolve(Some("t"), &reference("pico")).unwrap(), new);

        // A blob written moments ago is spared: its install may be on the
        // way. So is an old one an install has just reused.
        install(&mut repo, tenant("t"), "fresh.wasm", b"\0asm-fresh").await;
        repo.remove(&tenant("t"), &reference("fresh")).unwrap();
        assert_eq!(read_dir(&dir.path().join("blobs")).count(), 2);
        age_blobs(dir.path());
        install(&mut repo, tenant("v"), "again.wasm", b"\0asm-fresh").await;
        repo.remove(&tenant("v"), &reference("again")).unwrap();
        assert_eq!(read_dir(&dir.path().join("blobs")).count(), 2);
    }
}
