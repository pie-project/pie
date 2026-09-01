//! **§M wave M-3: the PREPARE writes the tiers, and a bare boot refuses.**
//!
//! This gate was `a_streamed_boot_writes_its_tiers`, and its subject changed
//! under it. §K phase T-2 gave a streamed load a key and made the first boot
//! write the artifact; §M-1 moved that write into `pie model import` as a
//! shortcut, with the boot-side write kept as a fallback; §M-3 removes the
//! fallback. A serving load that streams is warm or it is REFUSED, and
//! `Shell::prepare` is the only door in the process that writes one of these
//! files.
//!
//! So the same two boots test the same file from the other side:
//!
//! ```text
//!  0. refused  -> a streamed load against an EMPTY cache directory does not
//!                 stream, does not transform and does not write. It refuses,
//!                 and its sentence names `pie model import --prepare-only`
//!  1. written  -> one `Shell::prepare` under a 4 GiB device budget leaves
//!                 exactly one file, at `tier::path(dir, key)`, under the key
//!                 the load forms — and the counter says `stored`
//!  2. whole    -> it opens, and every block digest verifies against the
//!                 bytes that crossed
//!  3. true     -> the header's payload is what the RANKING lays out, and it
//!                 flags its registered planes as the zeros they are
//!  4. indexed  -> the index IS `Ranking::images`, image for image; every
//!                 plane the pinned tier holds is in it at the span the tier
//!                 seated it in, and so is every plane the store reserved for
//!  5. once     -> a boot at the same seat forms the same key, RESTORES out
//!                 of the file, and does not write it again
//! ```
//!
//! **(0) IS THE CLAIM THIS WAVE ADDS AND IT IS THE WHOLE OF IT.** Before it,
//! the sentence "the first boot writes" was measured; after it, the sentence
//! is "nothing but an import writes", and the only way to assert a path is
//! dead is to stand on it. The refusal is `Error::Impossible` — nothing the
//! deployment frees changes the answer — and the operator's remedy is in the
//! message rather than in a runbook.
//!
//! **AND NOT ONE OF THESE CLAIMS MENTIONS A BUDGET** (§M.3). Formats 1 and 2
//! wrote three budget-shaped sections and this gate asserted their three
//! lengths against `plan.device_demand()`, `plan.host_image()` and
//! `plan.mapped_image()` — the claim that the file WAS the split. It is not
//! any more: the file is the ranking, the split is the boot's, and what this
//! gate checks is that the writer transcribed the ranking. The positive form
//! of the new property — one artifact, two budgets — is
//! `a_second_streamed_boot_maps_the_tiers_it_wrote`'s last section.
//!
//! **(5) OUTLIVED ITS OWN PHASE TWICE.** When this file was written the
//! second boot still loaded cold, so the only thing the same key could buy was
//! a skipped write, and the counter it moved was `skipped`. T-3 turned the
//! file into a boot: the load now RESTORES out of it and never reaches the
//! writer at all. M-3 makes that the ONLY thing it can do. What (5) asserts
//! throughout is the thing it was written for — the key is stable across the
//! prepare and the boot of one deployment, and the bytes are not written
//! twice.
//!
//! ```bash
//! cargo test -p engine-cuda --features cuda-13 --release \
//!     --test a_prepare_writes_the_tiers_a_bare_boot_refuses \
//!     -- --ignored --nocapture
//! ```
//!
//! # Gating
//!
//! `#[ignore]`d: it wants a CUDA device, the gpt-oss-20b snapshot on disk, and
//! room under `TMPDIR` for the artifact (~15 GiB at this budget). It lands the
//! model TWICE, sequentially — once as a prepare, once as a boot — plus one
//! refusal that costs a plan compile and no bytes. Skips with a sentence when
//! any of it is missing, the same convention its siblings use.

use std::path::{Path, PathBuf};
use std::time::Instant;

use engine_cuda::experts::{Budgets, Held, Plan};
use engine_cuda::weight_cache::tier;
use engine_cuda::{Boot, Graphs, Shell};
use model_compiler::Budget;
use model_dsl::Platform;

const SKU: &str = "gptoss-20b-bf16-mxfp4-kv-bf16";

/// **The device budget, and it is the point of the gate.** Four gibibytes
/// under a table of roughly thirteen: whole mxfp4 groups land on the pinned
/// tier, the host tier is uncapped so none of them reach the mapping, and the
/// load streams — which is the only shape that reaches this write at all.
const DEVICE: u64 = 4 << 30;

/// How much room the artifact wants under `TMPDIR`, plus the writer's own
/// margin. The gate skips rather than filling a disk somebody else is using.
const ROOM: u64 = 24 << 30;

/// A temporary directory that removes itself, however the test leaves —
/// including the fifteen gigabytes the artifact costs.
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

fn snapshot() -> Option<PathBuf> {
    for key in ["PIE_GPTOSS_SNAPSHOT", "PIE_MASKLESS_SNAPSHOT"] {
        if let Ok(stated) = std::env::var(key) {
            let path = PathBuf::from(stated);
            return path.is_dir().then_some(path);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        Path::new(&home).join(".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .find(|path| path.join("tokenizer.json").exists())
}

/// Every container in the snapshot, sorted — gpt-oss-20b ships three shards
/// and an import built over shard zero refuses for an unrelated reason.
fn shards(snapshot: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    found
}

/// The trace, the contract and the checkpoint — everything both boots share,
/// read once.
struct Rig {
    trace: model_ir::Trace,
    contract: checkpoint::contract::ModelContract,
    checkpoint: PathBuf,
}

fn rig(what: &str) -> Option<Rig> {
    if !engine_cuda::device::present() {
        eprintln!("skipping {what}: no CUDA device on this machine");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping {what}: no gpt-oss-20b snapshot in the hugging face cache \
             (set PIE_GPTOSS_SNAPSHOT)"
        );
        return None;
    };
    let shards = shards(&checkpoint);
    if shards.is_empty() {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    }
    let trace = models::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
    let source = ztensor_compat::index_all(&shards).expect("the checkpoint's shards open as one");
    let contract = models::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);
    Some(Rig {
        trace,
        contract,
        checkpoint,
    })
}

/// **ONE DOCUMENT, TWO DOORS** (§M-3). The prepare and the boot have to
/// describe the same deployment in every field or they name two different
/// files — the artifact's key is a function of the trace, the recipe and the
/// ranking, and `Cuda::prepare` states the whole `Boot` for exactly this
/// reason. So the gate states it once and hands it to both.
fn doc<'a>(rig: &'a Rig, plan: Plan, dir: &'a Path) -> Boot<'a> {
    Boot {
        trace: rig.trace.clone(),
        contract: &rig.contract,
        checkpoint: &rig.checkpoint,
        budget: Budget::new(4, 256),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        // **THE FEATURE UNDER TEST.** A directory, and a plan that streams:
        // the two conditions the tier write is gated on.
        weight_cache_dir: Some(dir),
        residency: plan,
    }
}

/// A shell over the rig at a stated residency. Loaded and dropped: this gate
/// asserts about the LOAD, and — since §M-3 — about the fact that it leaves
/// no file behind.
fn boot(rig: &Rig, plan: Plan, dir: &Path) -> engine_cuda::Result<Shell> {
    Shell::load(doc(rig, plan, dir))
}

/// **THE WRITER**, and the only one there is. `pie model import` reaches this
/// through `Cuda::prepare`; the gate reaches it directly, because what it is
/// asserting about is the file and not the plumbing.
fn prepare(rig: &Rig, plan: Plan, dir: &Path) -> engine_cuda::Result<()> {
    Shell::prepare(doc(rig, plan, dir))
}

/// Free bytes under `dir`, so the gate skips instead of half-filling a disk.
fn free(dir: &Path) -> u64 {
    let Ok(text) = std::process::Command::new("df")
        .arg("-B1")
        .arg("--output=avail")
        .arg(dir)
        .output()
    else {
        return u64::MAX;
    };
    String::from_utf8_lossy(&text.stdout)
        .lines()
        .nth(1)
        .and_then(|line| line.trim().parse().ok())
        .unwrap_or(u64::MAX)
}

// ─────────────────────────────────────────────────────────────────────────────

/// **All six claims in one refusal, one prepare and one boot.**
#[test]
#[ignore = "real-hardware: needs a CUDA device, a local gpt-oss-20b snapshot \
            and ~24 GiB under TMPDIR; run it with `-- --ignored`"]
fn a_prepare_writes_the_tiers_a_bare_boot_refuses() {
    let Some(rig) = rig("the tier-write gate") else {
        return;
    };
    let dir = scratch("tier-write");
    if free(&dir.0) < ROOM {
        eprintln!(
            "skipping the tier-write gate: {:?} has {} GiB free and the artifact wants {} GiB",
            dir.0,
            free(&dir.0) >> 30,
            ROOM >> 30
        );
        return;
    }

    // ── THE PLAN. The pairing is the load plan's, read before a byte lands.
    let prospect = engine_cuda::weights::prospect(&rig.trace, &rig.contract, &rig.checkpoint)
        .expect("the load plan pairs every packed bank with its scales");
    let plan = Plan::of(&rig.trace, &prospect.planes, Budgets::device(DEVICE))
        .expect("a capped mxfp4 MoE plans rather than refusing");
    assert!(plan.streams(), "a 4 GiB budget under this table has to stream");
    assert_eq!(
        plan.spill_demand(),
        0,
        "the host tier is uncapped, so nothing reaches the mapping and this \
         phase has a file to write"
    );
    let pinned: Vec<&engine_cuda::experts::GroupPlan> = plan
        .groups()
        .iter()
        .filter(|group| group.held == Held::Pinned)
        .collect();
    assert!(
        !pinned.is_empty(),
        "the whole point of the budget is that whole groups land on T1"
    );
    let layout = plan.host_layout();
    assert_eq!(
        layout.len(),
        pinned.iter().map(|group| group.planes.len()).sum::<usize>() + plan.banks().len(),
        "the host layout states one span per pinned plane and per dense bank"
    );
    eprintln!(
        "gpt-oss-20b at a {} GiB device budget: {} bytes on the device, {} on the \
         pinned tier ({} groups, {} spans)",
        DEVICE >> 30,
        plan.device_demand(),
        plan.host_image(),
        pinned.len(),
        layout.len(),
    );

    // ── THE KEY, FORMED FROM OUTSIDE THE LOAD. The same statement the load
    //    forms inside `Weights::resident`, which is what makes naming the
    //    file an assertion rather than a search.
    //    It no longer takes the plan: the key stopped being a function of the
    //    budgets when the file did (§M.3), so `prospect` already has it.
    let key = engine_cuda::weights::tier_key(&rig.trace, &rig.contract, &rig.checkpoint)
        .expect("the key is a function of the trace and the recipe")
        .expect("this plan serializes, so this deployment forms a key");
    assert_eq!(
        key, prospect.tier_key,
        "the door and the prospect state one number"
    );
    let path = tier::path(&dir.0, key);
    assert!(!path.exists(), "nothing is cached before the prepare");

    // ── (0) AND A SERVING BOOT AGAINST THAT EMPTY DIRECTORY REFUSES (§M-3).
    //    The whole of the wave, stated first because it is the claim the rest
    //    of the file rests on: if this load succeeded there would be a second
    //    writer and every "only an import writes" sentence below would be
    //    prose. It costs a metadata parse and a plan compile — no checkpoint
    //    bytes are read, because the refusal is raised before the pinned tier
    //    is allocated and long before the executor would run.
    let before = tier::observed();
    let refused = boot(&rig, plan.clone(), &dir.0)
        .err()
        .expect("a streamed load with no serving artifact does not serve");
    let sentence = format!("{refused:?}");
    eprintln!("the bare streamed boot refused: {sentence}");
    assert!(
        sentence.contains("pie model import --prepare-only"),
        "the refusal names the command that fixes it: {sentence}"
    );
    assert!(
        sentence.contains("never been prepared"),
        "and an empty cache directory is the never-prepared case, not the \
         changed-recipe one: {sentence}"
    );
    assert!(!path.exists(), "and it wrote nothing: {path:?}");
    assert_eq!(
        std::fs::read_dir(&dir.0)
            .expect("the cache directory")
            .flatten()
            .count(),
        0,
        "not even a `.part`"
    );
    let after = tier::observed();
    assert_eq!(
        after.stored, before.stored,
        "a refusal is not a write: {before:?} -> {after:?}"
    );

    // ── (1) AND THE PREPARE WRITES.
    let before = tier::observed();
    let clock = Instant::now();
    prepare(&rig, plan.clone(), &dir.0).expect("the prepare lands and writes");
    let cold = clock.elapsed();
    let after = tier::observed();
    eprintln!("the prepare took {cold:.1?} and left {after:?}");
    assert_eq!(
        after.stored,
        before.stored + 1,
        "one prepare, one tier artifact: {before:?} -> {after:?}"
    );
    assert_eq!(after.declined, before.declined, "and nothing was declined");
    assert_eq!(after.skipped, before.skipped, "and nothing was already there");
    assert!(path.exists(), "under the key the load forms: {path:?}");
    assert_eq!(
        std::fs::read_dir(&dir.0)
            .expect("the cache directory")
            .flatten()
            .count(),
        1,
        "one file, and no `.part` left behind"
    );

    // ── (2) IT OPENS AND EVERY BLOCK VERIFIES.
    let artifact = tier::Artifact::open(&path).expect("what the load published opens");
    assert_eq!(artifact.key(), key, "under the key it was asked for");
    artifact
        .verify()
        .expect("every block hashes to what the table states");

    // ── (3) THE HEADER SAYS WHAT THE RANKING LAYS OUT.
    let head = artifact.head();
    let images = prospect.ranking.images();
    assert_eq!(
        head.payload_total,
        images.iter().map(|(_, _, _, reserved)| reserved).sum::<u64>(),
        "the payload is exactly what the ranking's spans tile"
    );
    assert!(
        head.payload_total >= plan.device_demand(),
        "and it holds at least the store this plan demanded — every plane the \
         store reserves for is an image, and so is every one it does not"
    );
    assert!(
        head.flags & tier::FLAG_ADAPTERS_ZEROED != 0,
        "the file states that its registered planes were the zeros \
         `Buffer::zeroed` left, which is why it does not carry them: the \
         snapshot is taken inside `Weights::resident`, before a \
         `register_adapter` can exist to call"
    );

    // ── (4) THE INDEX IS THE RANKING, IMAGE FOR IMAGE.
    assert_eq!(
        artifact.entries().len(),
        images.len(),
        "one entry per image the ranking states"
    );
    for (at, &(param, offset, bytes, reserved)) in images.iter().enumerate() {
        let group = artifact.entries()[at];
        assert_eq!(
            (u64::from(group.id), group.offset, group.bytes, group.reserved),
            (param, offset, bytes, reserved),
            "image {at} is not the image the ranking ranks there"
        );
        let seen = artifact
            .plane(group.id)
            .unwrap_or_else(|| panic!("param {param}'s bytes are inside the payload"));
        assert_eq!(seen.len() as u64, bytes, "and the window is its published length");
    }

    // Every plane the PINNED tier holds is an image, at the span the tier
    // seated it in — which is what makes the cut a lookup and not a search.
    for &(param, _, bytes, reserved) in &layout {
        let id = u32::try_from(param).expect("a param ordinal");
        let group = artifact
            .resolve(id)
            .unwrap_or_else(|| panic!("the index carries no image for param {id}"));
        assert_eq!(group.bytes, bytes, "param {id} is that long in the plan");
        assert_eq!(group.reserved, reserved, "in the span the tier gave it");
    }
    for group in &pinned {
        for plane in &group.planes {
            let id = u32::try_from(plane.param).expect("a param ordinal");
            let seen = artifact.resolve(id).unwrap_or_else(|| {
                panic!("`{}`'s plane {id} is on T1 and not in the file", group.name)
            });
            assert_eq!(
                seen.bytes, plane.bytes,
                "`{}`'s plane {id} is {} bytes in the plan",
                group.name, plane.bytes
            );
        }
    }
    // And so is every plane the STORE holds — except the registered ones,
    // which are the zeros the flag states and the file deliberately omits.
    let mut on_device = 0usize;
    for (at, param) in rig.trace.params.iter().enumerate() {
        let id = u32::try_from(at).expect("a param ordinal");
        let held = artifact.resolve(id);
        if param.source == model_dsl::ParamSource::Registered {
            assert!(held.is_none(), "param {id} is registered and is not in the file");
            continue;
        }
        let held = held.unwrap_or_else(|| panic!("param {id} is not in the file"));
        assert!(
            held.bytes <= held.reserved,
            "param {id} publishes more than its span holds"
        );
        if !plan.streamed_whole(at) {
            on_device += 1;
        }
    }
    assert!(on_device > 0, "the store holds planes, or this budget is not a budget");
    drop(artifact);
    let stat = std::fs::metadata(&path).expect("the file");
    let stamp = stat.modified().expect("a modification time");

    // ── (5) AND THE BOOT AT THAT SEAT SERVES OUT OF IT AND WRITES NOTHING.
    //    The counter below is `restored` and not `skipped` because the load
    //    never reaches the writer — and since §M-3 it COULD not: `write_tiers`
    //    is behind `Intent::Prepare`. What this still asserts is the thing it
    //    was written for — the prepare and the boot form one key, and the
    //    bytes on the disk are not written twice. It is also the positive half
    //    of (0): the same call that refused an empty directory serves out of a
    //    prepared one.
    let before = tier::observed();
    let clock = Instant::now();
    let shell = boot(&rig, plan.clone(), &dir.0).expect("the prepared shell loads");
    let again = clock.elapsed();
    assert!(
        !shell.weights_resident(),
        "a 4 GiB budget under a 13 GiB table streams, or this gate tests nothing"
    );
    assert!(
        shell.weights_from_cache(),
        "and it came off the artifact the prepare wrote, not out of a checkpoint"
    );
    drop(shell);
    let after = tier::observed();
    eprintln!("the prepared streamed boot took {again:.1?} and left {after:?}");
    assert_eq!(
        after.stored, before.stored,
        "the same seat forms the same key, and a serving load cannot write one"
    );
    assert_eq!(
        after.restored,
        before.restored + 1,
        "it read them instead, which is counted where an operator can see it: \
         {before:?} -> {after:?}"
    );
    assert_eq!(
        after.skipped, before.skipped,
        "and it never reached the writer, so nothing was skipped there either"
    );
    assert_eq!(after.declined, before.declined, "and is not a decline");
    assert_eq!(
        std::fs::metadata(&path)
            .expect("the file is still there")
            .modified()
            .expect("a modification time"),
        stamp,
        "the bytes were not rewritten"
    );
    assert_eq!(
        tier::Artifact::open(&path)
            .expect("and it still opens")
            .key(),
        key,
        "under the same key"
    );
}
