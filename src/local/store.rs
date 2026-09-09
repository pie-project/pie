use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};

use checkpoint::file::read::parse_metadata;

use checkpoint::file::meta::{SOURCE_KEY, VERSION_KEY};

pub const ARCHIVE_FILE: &str = "archive.zt";

pub const RUNTIME_DIR: &str = "runtime";

pub fn dir() -> PathBuf {
    bootstrap::paths::pie_home().join("models")
}

pub fn model_dir(name: &str) -> PathBuf {
    dir().join(name)
}

pub fn archive_path(name: &str) -> PathBuf {
    model_dir(name).join(ARCHIVE_FILE)
}

pub struct Runtime {
    pub key: String,
    pub files: Vec<PathBuf>,
    pub bytes: u64,
    pub runtime_quant: Option<String>,
}

pub struct Entry {
    pub name: String,
    pub qualified: Option<String>,
    pub siblings: usize,
    pub sku: Option<String>,
    pub backend: Option<String>,
    pub dir: Option<PathBuf>,
    pub root: PathBuf,
    pub files: Vec<PathBuf>,
    pub bytes: u64,
    pub tensors: usize,
    pub written_by: Option<String>,
    pub source: Option<String>,
    pub runtimes: Vec<Runtime>,
}

impl Entry {
    pub fn shards(&self) -> usize {
        self.files.len().saturating_sub(1)
    }

    pub fn address(&self) -> &str {
        match &self.qualified {
            Some(qualified) if self.siblings > 1 => qualified,
            _ => &self.name,
        }
    }

    pub fn total_bytes(&self) -> u64 {
        self.bytes + self.runtimes.iter().map(|r| r.bytes).sum::<u64>()
    }
}

pub fn entries() -> Result<Vec<Entry>> {
    entries_in(&dir())
}

fn entries_in(dir: &Path) -> Result<Vec<Entry>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut children: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|err| anyhow!("cannot read {}: {err}", dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .collect();
    children.sort();

    let mut found = Vec::new();
    for child in children.iter().filter(|path| path.is_dir()) {
        let name = child
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let roots = checkpoint::file::read::discover_zt_files(child);
        let mut in_dir = entries_from(roots);
        let siblings = in_dir.len();
        for entry in &mut in_dir {
            entry.qualified = qualified_name(&entry.root);
            entry.name = name.clone();
            entry.siblings = siblings;
            entry.dir = Some(child.clone());
            if siblings == 1 {
                entry.runtimes = read_runtimes(&child.join(RUNTIME_DIR));
            }
        }
        found.extend(in_dir);
    }

    let taken: BTreeSet<String> = found.iter().map(|entry| entry.name.clone()).collect();
    let flat: Vec<PathBuf> = children
        .iter()
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "zt"))
        .cloned()
        .collect();
    found.extend(
        entries_from(flat)
            .into_iter()
            .filter(|entry| !taken.contains(&entry.name)),
    );

    found.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(found)
}

fn qualified_name(root: &Path) -> Option<String> {
    let file_name = root.file_name()?.to_str()?;
    checkpoint::serving::Name::parse(file_name).ok()?;
    Some(file_name.strip_suffix(".zt")?.to_string())
}

fn read_entry(root: &Path, name: String) -> Option<Entry> {
    let metadata = parse_metadata(root).ok()?;
    let files: Vec<PathBuf> = metadata
        .files
        .iter()
        .map(|file| PathBuf::from(&file.path))
        .collect();
    let attributes = checkpoint::file::zt::read_attributes(root).unwrap_or_default();
    let stamp = checkpoint::file::serve::stamp_of(root).ok().flatten();
    Some(Entry {
        name,
        qualified: qualified_name(root),
        siblings: 1,
        sku: stamp.as_ref().map(|stamp| stamp.sku.clone()),
        backend: stamp.as_ref().map(|stamp| stamp.backend.clone()),
        dir: None,
        bytes: files
            .iter()
            .filter_map(|f| std::fs::metadata(f).ok())
            .map(|m| m.len())
            .sum(),
        tensors: metadata.weights().count(),
        written_by: attributes.get(VERSION_KEY).cloned(),
        source: attributes.get(SOURCE_KEY).cloned(),
        root: root.to_path_buf(),
        files,
        runtimes: Vec::new(),
    })
}

fn read_runtimes(dir: &Path) -> Vec<Runtime> {
    let Ok(read) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut candidates: Vec<PathBuf> = read
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "zt"))
        .collect();
    candidates.sort();

    entries_from(candidates)
        .into_iter()
        .map(|entry| Runtime {
            key: entry.name,
            bytes: entry.bytes,
            runtime_quant: checkpoint::file::zt::read_attributes(&entry.root)
                .unwrap_or_default()
                .get(checkpoint::file::meta::RUNTIME_QUANT_KEY)
                .cloned(),
            files: entry.files,
        })
        .collect()
}

fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn entries_from(candidates: Vec<PathBuf>) -> Vec<Entry> {
    let mut parsed = Vec::new();
    let mut claimed: BTreeSet<PathBuf> = BTreeSet::new();
    for path in candidates {
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let Some(entry) = read_entry(&path, name) else {
            continue;
        };
        for shard in entry.files.iter().skip(1) {
            claimed.insert(canonical(shard));
        }
        parsed.push(entry);
    }
    parsed.retain(|entry| !claimed.contains(&canonical(&entry.root)));
    parsed
}

pub enum Resolved {
    Missing,
    One(Box<Entry>),
    Ambiguous(Vec<String>),
}

pub fn find(name: &str) -> Result<Resolved> {
    Ok(resolve_in(entries()?, name))
}

fn resolve_in(entries: Vec<Entry>, name: &str) -> Resolved {
    if let Some(exact) = entries
        .iter()
        .position(|entry| entry.qualified.as_deref() == Some(name))
    {
        return Resolved::One(Box::new(
            entries
                .into_iter()
                .nth(exact)
                .expect("the position just found is in the vector it was found in"),
        ));
    }
    let mut named: Vec<Entry> = entries
        .into_iter()
        .filter(|entry| entry.name == name)
        .collect();
    match named.len() {
        0 => Resolved::Missing,
        1 => Resolved::One(Box::new(named.remove(0))),
        _ => Resolved::Ambiguous(
            named
                .iter()
                .map(|entry| entry.address().to_string())
                .collect(),
        ),
    }
}

pub fn remove(entry: &Entry) -> Result<()> {
    for runtime in &entry.runtimes {
        for file in runtime.files.iter().skip(1) {
            std::fs::remove_file(file)
                .map_err(|err| anyhow!("cannot delete {}: {err}", file.display()))?;
        }
        std::fs::remove_file(&runtime.files[0])
            .map_err(|err| anyhow!("cannot delete {}: {err}", runtime.files[0].display()))?;
    }
    for shard in entry.files.iter().skip(1) {
        std::fs::remove_file(shard)
            .map_err(|err| anyhow!("cannot delete {}: {err}", shard.display()))?;
    }
    std::fs::remove_file(&entry.root)
        .map_err(|err| anyhow!("cannot delete {}: {err}", entry.root.display()))?;
    if let Some(dir) = &entry.dir {
        let _ = std::fs::remove_dir(dir.join(RUNTIME_DIR));
        let _ = std::fs::remove_dir(dir);
    }
    Ok(())
}

pub fn staging_dir(repo_id: &str) -> Option<PathBuf> {
    let dir = crate::local::hf::resolve_cache_dir()
        .join(format!("models--{}", repo_id.replace('/', "--")));
    dir.is_dir().then_some(dir)
}

pub fn staging_bytes(dir: &Path) -> u64 {
    fn walk(dir: &Path) -> std::io::Result<u64> {
        let mut total = 0;
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_dir() {
                total += walk(&entry.path())?;
            } else if metadata.is_file() {
                total += metadata.len();
            }
        }
        Ok(total)
    }
    walk(dir).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use checkpoint::file::write::Writer;
    use checkpoint::types::{DType, Encoding, TensorDecl, TensorId, Visibility};

    fn decl(name: &str) -> TensorDecl {
        TensorDecl {
            id: TensorId(0),
            name: name.to_string(),
            shape: vec![32_000],
            encoding: Encoding::Raw(DType::U8),
            alignment: 1,
            visibility: Visibility::default(),
        }
    }

    #[test]
    fn store_every_case() {
        shards_are_not_entries_of_their_own();
        a_model_directory_is_one_entry_with_its_builds_beneath_it();
        a_leftover_flat_file_does_not_shadow_the_archive_that_replaced_it();
        an_artifact_named_for_its_specialization_is_listed();
        a_directory_of_three_backends_is_three_entries();
        a_lone_artifact_is_still_addressed_by_its_model_name();
        a_name_resolves_exactly_or_names_what_it_could_not_choose_between();
    }

    fn shards_are_not_entries_of_their_own() {
        let dir = tempfile::tempdir().unwrap();
        let payload = vec![7u8; 32_000];

        let single = dir.path().join("solo.zt");
        let mut writer = Writer::create(&single, &Default::default()).unwrap();
        writer.add_tensor(&decl("w"), &payload).unwrap();
        writer.finish().unwrap();

        let sharded = dir.path().join("split.zt");
        let mut writer = Writer::create_sharded(&sharded, &Default::default(), 40_000).unwrap();
        for i in 0..3 {
            writer
                .add_tensor(&decl(&format!("w{i}")), &payload)
                .unwrap();
        }
        writer.finish().unwrap();

        let on_disk = std::fs::read_dir(dir.path()).unwrap().count();
        assert_eq!(
            on_disk, 5,
            "expected a root plus three shards plus the solo"
        );

        let mut found: Vec<Entry> = {
            let mut candidates: Vec<std::path::PathBuf> = std::fs::read_dir(dir.path())
                .unwrap()
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "zt"))
                .collect();
            candidates.sort();
            entries_from(candidates)
        };
        found.sort_by(|a, b| a.name.cmp(&b.name));

        let names: Vec<&str> = found.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["solo", "split"]);
        assert_eq!(found[0].shards(), 0);
        assert_eq!(found[1].shards(), 3);
        assert_eq!(found[1].tensors, 3);
        assert!(found[1].bytes > 3 * 32_000);
    }

    fn a_model_directory_is_one_entry_with_its_builds_beneath_it() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("qwen--qwen3-0.6b");
        let payload = vec![7u8; 32_000];

        let archive = model.join(ARCHIVE_FILE);
        let mut writer = Writer::create(&archive, &Default::default()).unwrap();
        writer.add_tensor(&decl("w"), &payload).unwrap();
        writer.finish().unwrap();

        let plain = model.join(RUNTIME_DIR).join("0123456789abcdef.zt");
        let mut writer = Writer::create(&plain, &Default::default()).unwrap();
        writer.add_tensor(&decl("w"), &payload).unwrap();
        writer.finish().unwrap();

        let split = model.join(RUNTIME_DIR).join("fedcba9876543210.zt");
        let mut writer = Writer::create_sharded(&split, &Default::default(), 40_000).unwrap();
        for i in 0..3 {
            writer
                .add_tensor(&decl(&format!("w{i}")), &payload)
                .unwrap();
        }
        writer.finish().unwrap();

        let found = entries_in(root.path()).unwrap();
        assert_eq!(found.len(), 1, "one model, not one per file");
        let entry = &found[0];
        assert_eq!(entry.name, "qwen--qwen3-0.6b");

        assert_eq!(entry.tensors, 1, "the archive is one tensor");
        let on_disk = std::fs::read_dir(model.join(RUNTIME_DIR)).unwrap().count();
        assert_eq!(
            on_disk, 5,
            "a root plus three shards plus the unsharded one"
        );
        let keys: Vec<&str> = entry.runtimes.iter().map(|r| r.key.as_str()).collect();
        assert_eq!(keys, ["0123456789abcdef", "fedcba9876543210"]);
        assert_eq!(entry.runtimes[1].files.len(), 4, "root plus three shards");

        assert!(entry.total_bytes() > 3 * entry.bytes);
    }

    fn a_leftover_flat_file_does_not_shadow_the_archive_that_replaced_it() {
        let root = tempfile::tempdir().unwrap();
        let payload = vec![7u8; 32_000];

        let flat = root.path().join("qwen.zt");
        let mut writer = Writer::create(&flat, &Default::default()).unwrap();
        writer.add_tensor(&decl("w"), &payload).unwrap();
        writer.finish().unwrap();

        let archive = root.path().join("qwen").join(ARCHIVE_FILE);
        let mut writer = Writer::create(&archive, &Default::default()).unwrap();
        writer.add_tensor(&decl("a"), &payload).unwrap();
        writer.add_tensor(&decl("b"), &payload).unwrap();
        writer.finish().unwrap();

        let found = entries_in(root.path()).unwrap();
        assert_eq!(found.len(), 1, "one model, not two");
        assert_eq!(found[0].name, "qwen");
        assert_eq!(found[0].tensors, 2, "the archive, not the flat leftover");
        assert_eq!(found[0].root, archive);

        let other = root.path().join("legacy.zt");
        let mut writer = Writer::create(&other, &Default::default()).unwrap();
        writer.add_tensor(&decl("w"), &payload).unwrap();
        writer.finish().unwrap();
        let names: Vec<String> = entries_in(root.path())
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(names, ["legacy", "qwen"]);
    }

    fn an_artifact_named_for_its_specialization_is_listed() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("deepseek");
        let specialized = model.join("deepseek.dsv4-flash-full-u4g64-u2g64-kv-bf16.metal.zt");
        let mut writer = Writer::create(&specialized, &Default::default()).unwrap();
        writer.add_tensor(&decl("w"), &vec![7u8; 32_000]).unwrap();
        writer.finish().unwrap();

        let found = entries_in(root.path()).unwrap();
        assert_eq!(found.len(), 1, "the store holds one model and shows it");
        assert_eq!(
            found[0].name, "deepseek",
            "the entry is named for its directory, as every entry is"
        );
        assert_eq!(found[0].root, specialized);
        assert_eq!(found[0].tensors, 1);
    }

    fn serving(dir: &Path, slug: &str, sku: &str, backend: &str) -> std::path::PathBuf {
        let stamp = checkpoint::serving::Stamp::of(backend, sku);
        let path = dir.join(checkpoint::serving::Name::of(&stamp, slug).render());
        let mut writer = Writer::create_serving(&path, &Default::default(), stamp).unwrap();
        writer.add_tensor(&decl("w"), &vec![7u8; 32_000]).unwrap();
        writer.finish().unwrap();
        path
    }

    fn a_directory_of_three_backends_is_three_entries() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("gemma");
        let sku = "gemma4-e4b-bf16-kv-bf16";
        for backend in ["cuda", "vulkan", "wgpu"] {
            serving(&model, "gemma", sku, backend);
        }

        let found = entries_in(root.path()).unwrap();
        assert_eq!(found.len(), 3, "three artifacts, three entries");
        for entry in &found {
            assert_eq!(entry.name, "gemma", "every one is an artifact of gemma");
            assert_eq!(entry.siblings, 3);
            assert_eq!(entry.sku.as_deref(), Some(sku), "read off the stamp");
            assert_eq!(entry.tensors, 1);
        }
        let addresses: Vec<&str> = found.iter().map(Entry::address).collect();
        assert_eq!(
            addresses,
            [
                "gemma.gemma4-e4b-bf16-kv-bf16.cuda",
                "gemma.gemma4-e4b-bf16-kv-bf16.vulkan",
                "gemma.gemma4-e4b-bf16-kv-bf16.wgpu",
            ]
        );
        let backends: Vec<&str> = found
            .iter()
            .filter_map(|entry| entry.backend.as_deref())
            .collect();
        assert_eq!(backends, ["cuda", "vulkan", "wgpu"]);
    }

    fn a_lone_artifact_is_still_addressed_by_its_model_name() {
        let root = tempfile::tempdir().unwrap();
        serving(
            &root.path().join("gemma"),
            "gemma",
            "gemma4-e4b-bf16-kv-bf16",
            "cuda",
        );
        let found = entries_in(root.path()).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].address(), "gemma");
        assert_eq!(
            found[0].qualified.as_deref(),
            Some("gemma.gemma4-e4b-bf16-kv-bf16.cuda"),
            "the file still names itself, whether or not anything needs it to"
        );
    }

    fn a_name_resolves_exactly_or_names_what_it_could_not_choose_between() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("gemma");
        let sku = "gemma4-e4b-bf16-kv-bf16";
        for backend in ["cuda", "vulkan"] {
            serving(&model, "gemma", sku, backend);
        }
        serving(
            &root.path().join("qwen"),
            "qwen",
            "qwen3-0-6b-bf16-kv-bf16",
            "cuda",
        );

        let resolve = |name: &str| resolve_in(entries_in(root.path()).unwrap(), name);

        let Resolved::One(qwen) = resolve("qwen") else {
            panic!("one artifact, one answer");
        };
        assert_eq!(qwen.backend.as_deref(), Some("cuda"));

        let Resolved::Ambiguous(candidates) = resolve("gemma") else {
            panic!("two artifacts of one model is not a pick");
        };
        assert_eq!(
            candidates,
            [
                "gemma.gemma4-e4b-bf16-kv-bf16.cuda",
                "gemma.gemma4-e4b-bf16-kv-bf16.vulkan"
            ]
        );

        let Resolved::One(vulkan) = resolve("gemma.gemma4-e4b-bf16-kv-bf16.vulkan") else {
            panic!("a fully specified name names one file");
        };
        assert_eq!(vulkan.backend.as_deref(), Some("vulkan"));

        assert!(matches!(resolve("llama"), Resolved::Missing));
    }
}
