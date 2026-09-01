//! **The warm-boot weight artifact cache, end to end** (alto design §7's T2
//! tier; ported from `origin/dev`'s `weight_artifact_cache.hpp`).
//!
//! The materialized device weight table is a deterministic function of three
//! things — which checkpoint, which load contract compiled against it, which
//! layout this shell chose — so the second boot of the same deployment
//! recomputes, byte for byte, what the first one already produced. This gate
//! asserts that it does not have to, and the three claims that make skipping
//! it safe:
//!
//! ```text
//!  1. cold  -> the transforms run, and the artifact is written
//!  2. warm  -> the transforms DO NOT run, and the resident bytes are
//!              byte-identical to the cold load's
//!  3. rot   -> a corrupt artifact is CAUGHT, said out loud under its own
//!              counter, thrown away, and followed by the full load
//! ```
//!
//! Claim 2 is asserted with a digest of what is actually on the device, not
//! with a size and not with the cache's own word for it. Claim 3 is the one a
//! cache earns its keep by: dev's rule, kept verbatim in spirit — *"a
//! silently-corrupt weight artifact produces garbage tokens with no error,
//! which is not a trade any operator should be offered for a few seconds of
//! load time."* There is no unverified restore.
//!
//! ```bash
//! cargo test -p engine-cuda --features cuda-13 \
//!   --test a_warm_boot_reads_the_weights_it_wrote -- --nocapture
//! ```
//!
//! **IT WRITES THE WEIGHTS TO DISK.** One artifact the size of the model's
//! device table (~1.6 GiB for the 0.8B smoke SKU) lands in a temporary
//! directory and is removed at the end, including when an assertion fails.

use std::path::{Path, PathBuf};

use engine_cuda::{Boot, Graphs, Shell};
use model_compiler::Budget;
use model_dsl::Platform;

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// A temporary directory that removes itself, however the test leaves.
///
/// Not a crate: this is four lines and the alternative is a dev-dependency
/// whose whole job is `Drop`.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch(what: &str) -> Scratch {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos());
    let dir = std::env::temp_dir().join(format!("pie-{what}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temporary directory");
    Scratch(dir)
}

#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_warm_boot_reads_the_weights_it_wrote_and_refuses_the_ones_that_rotted() {
    let Some((checkpoint, container)) = ready("the weight artifact cache gate") else {
        return;
    };
    let cache = scratch("weight-cache");

    // ── 1. THE COLD LOAD. Nothing is on the disk, so the whole host-side
    //    transform pipeline runs and the artifact is written on the way out.
    let before = Shell::weight_cache_observed();
    let (cold_digest, cold_bytes) = {
        let shell = load(&checkpoint, &container, Some(&cache.0));
        assert!(
            !shell.weights_from_cache(),
            "the first load of an empty cache cannot have restored anything"
        );
        (
            shell.weight_digest().expect("the store reads back"),
            shell.footprint().0,
        )
    };
    let after_cold = Shell::weight_cache_observed();
    assert_eq!(
        after_cold.missed,
        before.missed + 1,
        "an empty cache is a miss, and it is counted: {after_cold:?}"
    );
    assert_eq!(
        after_cold.stored,
        before.stored + 1,
        "and the cold load wrote what it materialized: {after_cold:?}"
    );
    assert_eq!(after_cold.restored, before.restored);
    assert_eq!(
        after_cold.declined, before.declined,
        "the scratch disk had room; a declined write here means it did not"
    );

    let artifact = one_artifact(&cache.0);
    eprintln!(
        "cold load: {:.2} GiB resident, digest {cold_digest:016x}, artifact {:.2} GiB at {artifact:?}",
        cold_bytes as f64 / (1u64 << 30) as f64,
        std::fs::metadata(&artifact).map_or(0, |meta| meta.len()) as f64 / (1u64 << 30) as f64,
    );

    // ── 2. THE WARM BOOT. The same recipe keys the same artifact, so the
    //    transforms are skipped entirely — and the bytes that end up on the
    //    device are the SAME BYTES, which is the only claim that makes
    //    skipping them safe.
    let warm_digest = {
        let shell = load(&checkpoint, &container, Some(&cache.0));
        assert!(
            shell.weights_from_cache(),
            "the second load of the same recipe should have read the artifact"
        );
        assert_eq!(
            shell.footprint().0,
            cold_bytes,
            "a restore that resized the table restored the wrong thing"
        );
        shell.weight_digest().expect("the store reads back")
    };
    let after_warm = Shell::weight_cache_observed();
    assert_eq!(
        after_warm.restored,
        after_cold.restored + 1,
        "the warm boot is counted as a restore: {after_warm:?}"
    );
    assert_eq!(
        after_warm.missed, after_cold.missed,
        "and it is not also counted as a miss"
    );
    assert_eq!(
        after_warm.stored, after_cold.stored,
        "a restore rewrites nothing — the artifact it read is already the answer"
    );
    assert_eq!(
        warm_digest, cold_digest,
        "the warm boot's resident bytes differ from the cold boot's; \
         a cache that changes the weights is worse than no cache"
    );

    // ── 3. THE ROT. Flip bytes in the middle of the blob and load again. The
    //    digest is checked on every restore — there is no `verify = false` —
    //    so this is caught, counted under its own name, said out loud, and
    //    followed by the FULL load, which lands the same bytes as the cold one.
    rot(&artifact);
    let rotted_digest = {
        let shell = load(&checkpoint, &container, Some(&cache.0));
        assert!(
            !shell.weights_from_cache(),
            "a corrupt artifact must not be trusted"
        );
        shell.weight_digest().expect("the store reads back")
    };
    let after_rot = Shell::weight_cache_observed();
    assert_eq!(
        after_rot.corrupt,
        after_warm.corrupt + 1,
        "corruption has its own counter, and a silent retry would move the \
         miss counter instead: {after_rot:?}"
    );
    assert_eq!(
        after_rot.missed, after_warm.missed,
        "a corrupt artifact is NOT a miss — the distinction is the whole point \
         of counting it"
    );
    assert_eq!(
        rotted_digest, cold_digest,
        "the fallback load must land exactly what the cold load landed"
    );
    assert_eq!(
        after_rot.stored,
        after_warm.stored + 1,
        "and the discarded artifact is replaced by a good one"
    );

    // ── 4. AND THE REPLACEMENT IS GOOD. The fourth load restores again,
    //    which is what says step 3 rewrote rather than merely deleted.
    {
        let shell = load(&checkpoint, &container, Some(&cache.0));
        assert!(
            shell.weights_from_cache(),
            "the artifact rewritten after the rot should restore"
        );
        assert_eq!(
            shell.weight_digest().expect("the store reads back"),
            cold_digest
        );
    }

    // ── 5. AND NO DIRECTORY IS THE FEATURE OFF: no read, no write, and no
    //    counter moved. A deployment that named nothing did not miss a cache.
    let before_off = Shell::weight_cache_observed();
    {
        let shell = load(&checkpoint, &container, None);
        assert!(!shell.weights_from_cache());
    }
    assert_eq!(
        Shell::weight_cache_observed(),
        before_off,
        "a load offered no cache directory did not miss one"
    );
}

// ── the load ─────────────────────────────────────────────────────────────

/// Everything the gate needs, or `None` and a sentence saying what is missing.
fn ready(what: &str) -> Option<(PathBuf, PathBuf)> {
    if !engine_cuda::device::present() {
        eprintln!("skipping {what}: no CUDA device on this machine");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping {what}: no Qwen3.5-0.8B snapshot in the hugging face cache \
             (set PIE_SMOKE_SNAPSHOT)"
        );
        return None;
    };
    let Some(container) = container(&checkpoint) else {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    };
    Some((checkpoint, container))
}

/// One load, through the same door production comes through.
fn load(checkpoint: &Path, container: &Path, cache_dir: Option<&Path>) -> Shell {
    let trace = models::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
    let source = ztensor_compat::index(container).expect("the checkpoint opens");
    let contract = models::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);
    Shell::load(Boot {
        // Full residency: the whole weight table on the device, which is what
        // an uncapped `Residency` plans (alto design §7).
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint,
        budget: Budget::new(2, 64),
        patches: None,
        profile: None,
        page_size: 16,
        context: 128,
        slots: 2,
        ordinal: 0,
        // Nothing here fires: the claim is about which bytes are resident,
        // and recording a graph would only cost the gate time.
        graphs: Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        weight_cache_dir: cache_dir,
    })
    .expect("the shell loads")
}

/// The one artifact this gate's cache directory holds.
fn one_artifact(dir: &Path) -> PathBuf {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("the cache directory exists")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            path.extension()
                .is_some_and(|extension| extension == "weights")
                .then_some(path)
        })
        .collect();
    found.sort();
    assert_eq!(
        found.len(),
        1,
        "one load of one recipe writes one artifact, not {found:?}"
    );
    found.into_iter().next().expect("checked above")
}

/// Flip bytes in the middle of the blob — past the header, so what the file
/// CLAIMS about itself is untouched and only the bytes disagree. That is the
/// case a checksum exists for; a mangled header would be caught by the magic.
fn rot(artifact: &Path) {
    use std::io::{Seek, SeekFrom, Write};

    let len = std::fs::metadata(artifact).expect("the artifact exists").len();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(artifact)
        .expect("the artifact opens for writing");
    file.seek(SeekFrom::Start(len / 2)).expect("seek");
    file.write_all(&[0x5au8; 4096]).expect("write");
    file.sync_all().expect("sync");
}

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        Path::new(&home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .find(|path| path.join("tokenizer.json").exists())
}

fn container(snapshot: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    found.into_iter().next()
}
