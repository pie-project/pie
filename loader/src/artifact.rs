//! Identity of the materialized weights a plan produces.
//!
//! A driver that caches finished device weights needs a name for them that
//! changes whenever the bytes would. This computes that name.
//!
//! It used to live in the CUDA driver, where it hashed a hand-written list of
//! plan fields — a list that had to be extended by hand every time the plan grew
//! one, with a silent stale-cache hit as the penalty for forgetting. Here the
//! plan's own `Serialize` supplies the field list, so a new field is covered the
//! moment it exists. It also means both drivers get the same key instead of only
//! the one that happened to implement it.

use std::io::{self, Write};
use std::path::Path;

use serde::Serialize;

use crate::plan::LoadPlan;

/// The request-side inputs that decide the bytes but are not visible in a plan.
pub struct ArtifactInputs<'a> {
    pub snapshot_dir: &'a Path,
    /// The runtime quantization policy, *after* the target's capabilities have
    /// been applied — a policy the device cannot execute is already dropped by
    /// the time a plan exists, and two requests that differ only there produce
    /// identical weights.
    pub runtime_quant: &'a str,
    /// Which slice of the model was compiled, as the request's own discriminant.
    ///
    /// Unlike the MoE lowering — which resolves into `plan.target` and is hashed
    /// with it — nothing in the plan states the component directly, so it has to
    /// be mixed in on its own.
    pub component: u32,
}

/// Name the artifact this plan will materialize.
///
/// Three independent inputs decide the bytes that land on the device:
///
/// 1. **The whole compiled plan** — every instruction, buffer, tensor, source
///    span and the target it was lowered for. The plan *is* the recipe, so
///    hashing it whole is the only formulation that cannot be out of date: any
///    request field that changes the bytes does so by changing the plan, and any
///    field that does not change the plan cannot change the bytes.
/// 2. **The request facts no plan states** — the snapshot path and the component,
///    in [`ArtifactInputs`].
/// 3. **The weight files' size and mtime** — a checkpoint whose payload bytes are
///    rewritten in place without changing any header produces an identical plan,
///    so (1) alone would still hit a stale artifact.
///
/// A false miss costs one re-materialization. A false hit puts silently wrong
/// weights on the device, so this errs toward missing.
pub fn artifact_cache_key(plan: &LoadPlan, inputs: &ArtifactInputs<'_>) -> String {
    let mut hash = Fnv1a::new();
    hash.mix_bytes(inputs.snapshot_dir.as_os_str().as_encoded_bytes());
    hash.mix_bytes(inputs.runtime_quant.as_bytes());
    hash.mix_u64(u64::from(inputs.component));
    hash.mix_json(plan);
    hash.mix_snapshot_stat(inputs.snapshot_dir);
    format!("{:016x}", hash.0)
}

/// Whether a file in the snapshot directory can change what a recompile produces.
///
/// Broader than the plan's file table on purpose: the table names what the plan
/// *reads*, while the key must cover anything that could change what a recompile
/// would decide — including `config.json` and an index file the plan itself never
/// opens. Over-approximating only costs a cache miss.
fn affects_compilation(name: &str) -> bool {
    name.ends_with(".safetensors")
        || name.ends_with(".gguf")
        || name == "config.json"
        || name == "model.safetensors.index.json"
}

struct Fnv1a(u64);

impl Fnv1a {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn mix_raw(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn mix_u64(&mut self, value: u64) {
        self.mix_raw(&value.to_le_bytes());
    }

    /// Mix a run, length-prefixed so adjacent fields cannot be confused for one
    /// another ("ab" then "c" must not collide with "a" then "bc").
    fn mix_bytes(&mut self, bytes: &[u8]) {
        self.mix_u64(bytes.len() as u64);
        self.mix_raw(bytes);
    }

    /// Mix a value's serialized form, so its field list is whatever the type
    /// declares rather than whatever this function remembers to name.
    fn mix_json(&mut self, value: &impl Serialize) {
        // Serializing straight into the hash keeps the plan's source table from
        // being materialized as a second copy on every compile.
        if serde_json::to_writer(&mut *self, value).is_err() {
            // `Serialize` on a plan cannot fail, but a sentinel beats a panic on
            // the compile path if one ever does.
            self.mix_u64(0xBADD_12EB_ADD1_2EBA);
        }
    }

    /// Mix the size and mtime of every file that could change a recompile.
    ///
    /// The stat follows symlinks, and must: a HuggingFace snapshot directory is
    /// a farm of links into `blobs/`, so `DirEntry::metadata` — which is `lstat`
    /// — would report the length of the link target *string* (52 or 76 bytes,
    /// the same for every model) and the link's own mtime. That carries no
    /// information about the checkpoint at all.
    ///
    /// Any failure to read a file's metadata mixes a sentinel rather than being
    /// skipped: an unreadable checkpoint must not hash the same as a readable
    /// one. Materialization will produce the real error moments later.
    fn mix_snapshot_stat(&mut self, dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            self.mix_u64(0xDEAD_BEEF_DEAD_BEEF);
            return;
        };
        let mut relevant: Vec<_> = entries
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_str().is_some_and(affects_compilation))
            .collect();
        // Directory order is unspecified.
        relevant.sort_by_key(std::fs::DirEntry::file_name);

        self.mix_u64(relevant.len() as u64);
        for entry in relevant {
            self.mix_bytes(entry.file_name().as_encoded_bytes());
            let Ok(meta) = std::fs::metadata(entry.path()) else {
                self.mix_u64(u64::MAX);
                self.mix_u64(u64::MAX);
                continue;
            };
            self.mix_u64(meta.len());
            let mtime = meta
                .modified()
                .ok()
                .and_then(|at| at.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(u64::MAX, |since| since.as_nanos() as u64);
            self.mix_u64(mtime);
        }
    }
}

impl Write for Fnv1a {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.mix_raw(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{CheckpointFileDecl, StorageTarget};
    use crate::types::{CheckpointFormat, FileId};

    /// A directory of its own per test.
    ///
    /// The key stats the snapshot directory, so sharing one with the rest of the
    /// suite would make every assertion here depend on what other tests happened
    /// to leave lying around.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("pie-loader-artifact-{}-{tag}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn plan_with(files: Vec<CheckpointFileDecl>) -> LoadPlan {
        let mut plan = LoadPlan::empty(StorageTarget::default());
        plan.files = files;
        plan
    }

    fn file(path: &str) -> CheckpointFileDecl {
        CheckpointFileDecl {
            id: FileId(0),
            path: path.to_string(),
            size_bytes: 1,
            format: CheckpointFormat::Safetensors,
        }
    }

    fn inputs(dir: &Path) -> ArtifactInputs<'_> {
        ArtifactInputs {
            snapshot_dir: dir,
            runtime_quant: "",
            component: 0,
        }
    }

    #[test]
    fn a_key_is_sixteen_hex_digits() {
        let dir = TempDir::new("hex");
        let key = artifact_cache_key(&plan_with(Vec::new()), &inputs(&dir.0));
        assert_eq!(key.len(), 16);
        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn a_different_plan_is_a_different_key() {
        let dir = TempDir::new("plan");
        let one = artifact_cache_key(&plan_with(vec![file("a.safetensors")]), &inputs(&dir.0));
        let two = artifact_cache_key(&plan_with(vec![file("b.safetensors")]), &inputs(&dir.0));
        assert_ne!(one, two);
    }

    #[test]
    fn a_different_component_is_a_different_key() {
        let dir = TempDir::new("component");
        let plan = plan_with(Vec::new());
        let mut other = inputs(&dir.0);
        other.component = 1;
        assert_ne!(
            artifact_cache_key(&plan, &inputs(&dir.0)),
            artifact_cache_key(&plan, &other)
        );
    }

    #[test]
    fn the_same_request_is_the_same_key() {
        let dir = TempDir::new("stable");
        let plan = plan_with(vec![file("a.safetensors")]);
        assert_eq!(
            artifact_cache_key(&plan, &inputs(&dir.0)),
            artifact_cache_key(&plan, &inputs(&dir.0))
        );
    }

    /// The plan alone cannot see a checkpoint whose payload bytes were rewritten
    /// in place, which is what the directory stat is there to catch.
    #[test]
    fn a_rewritten_weight_file_is_a_different_key() {
        let dir = TempDir::new("rewrite");
        let plan = plan_with(vec![file("model.safetensors")]);
        let weights = dir.0.join("model.safetensors");
        std::fs::write(&weights, b"one").expect("write");
        let before = artifact_cache_key(&plan, &inputs(&dir.0));
        std::fs::write(&weights, b"two different bytes").expect("rewrite");
        assert_ne!(before, artifact_cache_key(&plan, &inputs(&dir.0)));
    }

    /// `config.json` is not in the plan's file table — the loader reads it and
    /// keeps the answer — so only the directory scan can notice an edit to it.
    #[test]
    fn an_edited_config_is_a_different_key() {
        let dir = TempDir::new("config");
        let plan = plan_with(Vec::new());
        let config = dir.0.join("config.json");
        std::fs::write(&config, b"{}").expect("write");
        let before = artifact_cache_key(&plan, &inputs(&dir.0));
        std::fs::write(&config, b"{\"model_type\":\"llama\"}").expect("rewrite");
        assert_ne!(before, artifact_cache_key(&plan, &inputs(&dir.0)));
    }

    /// The device the plan was lowered for decides the bytes as surely as the
    /// checkpoint does — two ranks of one group must not share an artifact.
    ///
    /// Every field of the target reaches the key through its `Serialize`, so
    /// this checks the mixing is wired at all rather than enumerating fields;
    /// enumerating them is the hand-maintained list this replaced.
    #[test]
    fn a_different_target_is_a_different_key() {
        let dir = TempDir::new("target");
        let plan = plan_with(Vec::new());
        let mut other = plan_with(Vec::new());
        other.target.tp_rank = 1;
        other.target.tp_size = 2;
        assert_ne!(
            artifact_cache_key(&plan, &inputs(&dir.0)),
            artifact_cache_key(&other, &inputs(&dir.0))
        );
    }

    /// The source table is what makes the key checkpoint-sensitive: a header
    /// change lands here and nowhere else.
    #[test]
    fn a_different_source_is_a_different_key() {
        use crate::plan::SourceTensorDecl;
        use crate::types::{DType, Encoding, TensorId};

        let source = |offset| SourceTensorDecl {
            id: TensorId(0),
            name: "w".to_string(),
            file_id: FileId(0),
            file_offset: offset,
            span_bytes: 4,
            shape: vec![2, 2],
            encoding: Encoding::Raw(DType::BF16),
        };
        let dir = TempDir::new("source");
        let mut plan = plan_with(Vec::new());
        plan.sources = vec![source(0)];
        let mut other = plan_with(Vec::new());
        other.sources = vec![source(64)];
        assert_ne!(
            artifact_cache_key(&plan, &inputs(&dir.0)),
            artifact_cache_key(&other, &inputs(&dir.0))
        );
    }

    /// Length prefixing is what keeps two adjacent strings from being confused
    /// for one differently-split pair.
    #[test]
    fn adjacent_fields_do_not_run_together() {
        let dir = TempDir::new("prefix");
        let plan = plan_with(Vec::new());
        let ab = ArtifactInputs {
            snapshot_dir: &dir.0,
            runtime_quant: "ab",
            component: 0,
        };
        let a = ArtifactInputs {
            snapshot_dir: &dir.0,
            runtime_quant: "a",
            component: 0,
        };
        assert_ne!(
            artifact_cache_key(&plan, &ab),
            artifact_cache_key(&plan, &a)
        );
    }

    /// A HuggingFace snapshot directory is a farm of symlinks into `blobs/`, so
    /// a key that stats the *link* learns nothing: every model's `config.json`
    /// link is the same 52 bytes, and rewriting the blob leaves the link
    /// untouched. This is the layout the driver actually passes as
    /// `snapshot_dir`, so a regression here silently reuses another
    /// configuration's weights.
    #[test]
    #[cfg(unix)]
    fn an_edited_config_behind_a_symlink_is_a_different_key() {
        let dir = TempDir::new("symlink");
        let blobs = dir.0.join("blobs");
        std::fs::create_dir_all(&blobs).expect("create blobs");
        let blob = blobs.join("abc123");
        std::fs::write(&blob, b"{\"num_hidden_layers\": 28}").expect("write blob");

        let snapshot = dir.0.join("snapshot");
        std::fs::create_dir_all(&snapshot).expect("create snapshot");
        std::os::unix::fs::symlink(&blob, snapshot.join("config.json")).expect("link");

        let plan = plan_with(vec![file("model.safetensors")]);
        let before = artifact_cache_key(&plan, &inputs(&snapshot));

        std::fs::write(&blob, b"{\"num_hidden_layers\": 36, \"quant\": \"awq\"}")
            .expect("rewrite blob");
        let after = artifact_cache_key(&plan, &inputs(&snapshot));

        assert_ne!(
            before, after,
            "a config edit through the snapshot's symlink must move the key"
        );
    }

    /// The plan is the recipe, so anything that reaches it must reach the key —
    /// including the parts the old hand-written field list never named.
    #[test]
    fn a_different_instruction_stream_is_a_different_key() {
        let dir = TempDir::new("instrs");
        let mut plan = plan_with(vec![file("model.safetensors")]);
        let before = artifact_cache_key(&plan, &inputs(&dir.0));

        plan.memory.device_write_bytes += 1;
        assert_ne!(before, artifact_cache_key(&plan, &inputs(&dir.0)));
    }
}
