//! **§K phase T-4: the tier artifact carries the third image too, and a
//! spilled seat boots off its own file.**
//!
//! T-3 gave a streamed load a warm boot out of two sections. It could not give
//! one to a SPILLED load — a plan whose groups fit neither budget — because
//! the writer declined the moment `spill_demand` moved: a file describing two
//! of three images would key, open, verify and restore two tiers of three.
//! Every boot of such a deployment therefore ran the whole executor, every
//! time, and needed a hundred-gigabyte whole-table artifact beside it to map
//! T2 out of. This is the phase that closes it.
//!
//! ```text
//!  1. seed   -> one uncapped boot writes the whole-table artifact and sets
//!               the golden tokens; that file is the bootstrap and nothing
//!               else
//!  2. made   -> `Shell::prepare` lands the spilled plan off it and writes a
//!               SERVING artifact carrying every plane of the trace —
//!               including, at their own spans, the ones this budget pair
//!               spills — every block verifying. A serving boot cannot do
//!               this and refuses instead (§M-3), which claim 0 stands on
//!  3. alone  -> the bootstrap is moved away, and the spilled plan boots
//!               again anyway: `admit_tiers` is sourced by the serving
//!               artifact, the device and pinned images come off it, and T2 is
//!               mapped out of the same file
//!  4. same   -> the tokens are the same tokens and the logits are the same
//!               floats, across all three boots
//!  5. faster -> the executor is gone from the spilled load for the first
//!               time; the warm boot beats the cold one by `SPEEDUP`
//!  6. rot    -> one flipped byte in an image this budget pair SPILLS is
//!               hashed at the door, counted, named, and the file LEFT ON THE
//!               DISK (§M.4) — the load refuses rather than serving out of it,
//!               and once the bootstrap is back a PREPARE replaces it and the
//!               boot after that answers the same answer
//! ```
//!
//! # Why claim 3 is the point
//!
//! T2 is not a speed feature. `Tier::open` resolves a spilled group's
//! addresses against a mapping, and before this phase the only file that could
//! answer was the snapshot of a device store that had held the WHOLE table —
//! so a deployment's T2 source was a file only a boot with enough VRAM for the
//! entire model could ever have written. After it, one capped boot transcribes
//! every plane into a file of its own, and that file is a complete load: one
//! image per plane, one key, one artifact. The move-aside in step 3 is what
//! makes the claim decidable rather than asserted — the bootstrap is not on the
//! disk when the shell boots.
//!
//! **AND SINCE §M THE FILE IS NOT KEYED TO THIS BUDGET PAIR.** It was, and
//! that made "one capped boot writes the source" a sentence about ONE budget
//! pair: change either and the whole transcription happened again under
//! another key. The file carries the ranking now and the budgets only cut it,
//! so what step 2 writes is a source for any budget pair on this machine —
//! which is the claim `a_second_streamed_boot_maps_the_tiers_it_wrote` proves
//! directly.
//!
//! # And since §M-3 the capped BOOT does not write it — the PREPARE does
//!
//! Step 2 said "one capped boot writes the source", and the boot that wrote it
//! was a serving load running the whole executor. That road is closed:
//! `Shell::load` passes `Intent::Serve` and a streamed load under that intent
//! is warm or refused. So step 2 is a `Shell::prepare` — `pie model import`'s
//! own door — and everything the phase claimed about the FILE is unchanged,
//! because it is the same landing writing the same images.
//!
//! **THE BOOTSTRAP SURVIVES, AND IT SURVIVES AS PREPARE'S ROAD IN.** It is
//! tempting to read "the boot is warm-only" as "the whole-table artifact is
//! dead". It is the reverse. A spilled deployment's prepare has no serving
//! artifact by definition — the file it is about to write is the one that does
//! not exist — and its landing still has to read the spilled planes from
//! somewhere. That somewhere is step 1's file, exactly as it was in §K.6-T4.
//! `Residency::admit_tiers` still counts either file as a source for the same
//! reason: it is asked by `Cuda::prepare` as well as by `Engine::load`, and a
//! statute that demanded the tier artifact would refuse the run that creates
//! it. Step 3's move-aside is still what makes the claim decidable.
//!
//! ```bash
//! cargo test -p engine-cuda --features cuda-13 --release \
//!     --test a_spilled_boot_carries_the_tier_no_budget_held -- --ignored --nocapture
//! ```
//!
//! # Gating and cost
//!
//! `#[ignore]`d. It wants a CUDA device that reports `pageableMemoryAccess`
//! (CUDA 12.2+ HMM) with ~15 GiB free, the gpt-oss-20b snapshot, and room
//! under `TMPDIR` for two artifacts at once. It boots the model four times and
//! refuses once, sequentially. Skips with a sentence when any of that is
//! missing, the same convention its siblings use.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use engine_cuda::experts::{Budgets, Held, Plan};
use engine_cuda::weight_cache::tier;
use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "gptoss-20b-bf16-mxfp4-kv-bf16";

/// **THE TWO CEILINGS**, verbatim from the W-1 capability gate: six GiB
/// between them over a ~12.8 GiB table, so what neither holds is most of the
/// model and the third section is most of the file.
const DEVICE: u64 = 4 << 30;
const HOST: u64 = 2 << 30;

/// Room under `TMPDIR` for BOTH artifacts at once — the ~13 GiB whole-table
/// bootstrap and the ~13 GiB tier file whose three sections mirror it — plus
/// the writer's own margin.
const ROOM: u64 = 40 << 30;

/// **What the warm spilled boot has to beat.**
///
/// Lower than the tier-read gate's four, and the reason is the section this
/// phase adds: a warm spilled boot pays for a full hash of the MAPPED section
/// at the door (`open_tiers` argues why it is up front and whole) and the cold
/// boot it is measured against does not run the transform pipeline for the T2
/// planes' sake alone — it runs it for the whole table. Two is a real bar all
/// the same: the cold branch running at all takes the ratio to one, and the
/// executor is what separates the two boots.
const SPEEDUP: f64 = 2.0;

/// And the ceiling, whatever the cold load cost — §K.6-T3's number, kept.
const CEILING: Duration = Duration::from_secs(120);

const PROMPT: &str = "<|start|>user<|message|>What is the capital of France? \
                      Answer in one word.<|end|>\
                      <|start|>assistant<|channel|>final<|message|>";

/// How many greedy decodes follow the prefill. Short: every fire reads the
/// mapped banks, so the claim is stated in as few steps as can state it.
const STEPS: usize = 6;

/// A temporary directory that removes itself, however the test leaves —
/// including the gigabytes the two artifacts cost.
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

fn word(query_len: u32) -> u64 {
    models::gpt_oss::forward::Facts::of(&Request::new(query_len, false)).word()
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

/// Everything all four boots share, read once.
struct Rig {
    trace: model_ir::Trace,
    contract: checkpoint::contract::ModelContract,
    checkpoint: PathBuf,
    tokenizer: tokenizer::Tokenizer,
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
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let trace = models::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
    let source = ztensor_compat::index_all(&shards).expect("the checkpoint's shards open as one");
    let contract = models::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);
    Some(Rig {
        trace,
        contract,
        checkpoint,
        tokenizer,
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
        weight_cache_dir: Some(dir),
        residency: plan,
    }
}

fn boot(rig: &Rig, plan: Plan, dir: &Path) -> engine_cuda::Result<Shell> {
    Shell::load(doc(rig, plan, dir))
}

/// **THE WRITER**, and the only one there is (§M-3). A spilled prepare reads
/// its T2 planes out of the whole-table bootstrap and writes every plane of
/// the trace back out as one serving artifact.
fn prepare(rig: &Rig, plan: Plan, dir: &Path) -> engine_cuda::Result<()> {
    Shell::prepare(doc(rig, plan, dir))
}

/// A prefill and `STEPS` greedy decodes, feeding the argmax back.
fn run(shell: &mut Shell, prompt: &[u32]) -> (Vec<u32>, Vec<Vec<f32>>) {
    shell.open(0).expect("slot 0 opens");
    let mut chosen = Vec::with_capacity(STEPS + 1);
    let mut rows = Vec::with_capacity(STEPS + 1);
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the prefill fires");
    let mut fed = argmax(&prefill[0]);
    chosen.push(fed);
    rows.push(prefill[0].clone());
    for step in 0..STEPS {
        let decode = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &[fed],
            }])
            .unwrap_or_else(|why| panic!("decode step {step} fires: {why}"));
        fed = argmax(&decode[0]);
        chosen.push(fed);
        rows.push(decode[0].clone());
    }
    (chosen, rows)
}

fn argmax(logits: &[f32]) -> u32 {
    assert!(!logits.is_empty(), "a fire produced no logits at all");
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        assert!(value.is_finite(), "logit {at} is {value}");
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

/// The same floats, not nearly the same floats.
fn same_logits(golden: &[Vec<f32>], found: &[Vec<f32>], what: &str) {
    assert_eq!(golden.len(), found.len(), "{what} produced a different number of rows");
    for (step, (a, b)) in golden.iter().zip(found).enumerate() {
        assert_eq!(a.len(), b.len(), "{what} step {step} has a different width");
        for (at, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{what} step {step}, logit {at}: uncapped {x}, this load {y} — a tier \
                 moved a number"
            );
        }
    }
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

/// **Flip one bit in the middle of an image this budget pair SPILLS**, in
/// place.
///
/// The middle rather than the first byte, for the tier-read gate's reason: a
/// reader that checked only its first block would pass a corruption at byte
/// zero for the wrong reason. And an image the cut MAPS rather than any image,
/// because that is the rung this phase adds — the one no restore reads and no
/// kernel bounds-checks, whose bytes a GPU faults in one page at a time long
/// after any door could have looked at them.
fn flip_a_mapped_byte(path: &Path, plan: &Plan) -> u64 {
    use std::os::unix::fs::FileExt;

    let artifact = tier::Artifact::open(path).expect("the artifact opens");
    let head = artifact.head();
    let mapped = plan.mapped_layout();
    let (param, _, _, _) = *mapped
        .get(mapped.len() / 2)
        .expect("this plan seats bytes on the mapping");
    let group = artifact
        .resolve(u32::try_from(param).expect("a param ordinal"))
        .expect("the file carries every plane this plan spills");
    let at = head.payload_at + group.offset + group.reserved / 2;
    drop(artifact);
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .expect("the artifact is writable");
    let mut byte = [0u8; 1];
    file.read_exact_at(&mut byte, at).expect("the byte reads");
    byte[0] ^= 0x01;
    file.write_all_at(&byte, at).expect("the byte writes");
    file.sync_all().expect("the flip reaches the disk");
    at
}

// ─────────────────────────────────────────────────────────────────────────────

/// **The six claims, in one process, on one seat.**
#[test]
// **THIS SUITE ASSERTS A ROAD §M-4d REMOVED, AND IT IS RE-AIMED AND NOT
// DELETED.** The PROPERTY it holds — a spilled boot serves planes no budget
// held — is exactly what §M-4 is for and still true. What moved is the SOURCE:
// the tier artifact this file opens by key is no longer what a spilled load
// reads, because the model's own `.zt` is (`experts::Spill::Serving`), and the
// two doors were measured equal within noise before the old one went.
//
// So every `tier_spill` assertion below is about a file nothing opens now, and
// running this as it stands fails for that reason and not for a regression.
// `tests/gpu/tests/cuda_serving_spill.rs` holds the same property against the
// new source and is cheap enough to run per-change; this one covers what that
// cannot — a 20B at HMM, three rungs, a real ladder — and is worth re-pointing
// rather than losing.
#[ignore = "SUPERSEDED PENDING RE-AIM (§M-4d): asserts the tier road, which no \
            longer carries a spill — see the note above. Also real-hardware: a \
            CUDA device with HMM and ~15 GiB free, a local gpt-oss-20b snapshot, \
            and ~40 GiB under TMPDIR"]
fn a_spilled_boot_carries_the_tier_no_budget_held() {
    let Some(rig) = rig("the T-4 third-section gate") else {
        return;
    };
    if !engine_cuda::experts::pageable_access() {
        eprintln!(
            "skipping the T-4 third-section gate: this device does not report \
             `pageableMemoryAccess`, so a GPU touch of a mapped page cannot fault it \
             in — the T2 arm's one hardware precondition"
        );
        return;
    }
    let dir = scratch("tier-spill");
    if free(&dir.0) < ROOM {
        eprintln!(
            "skipping the T-4 third-section gate: {:?} has {} GiB free and two \
             artifacts want {} GiB",
            dir.0,
            free(&dir.0) >> 30,
            ROOM >> 30
        );
        return;
    }
    let prompt = rig.tokenizer.encode(PROMPT);

    // ── THE PLAN, AND BOTH KEYS, READ FROM OUTSIDE THE LOAD.
    let prospect = engine_cuda::weights::prospect(&rig.trace, &rig.contract, &rig.checkpoint)
        .expect("the load plan pairs every packed bank with its scales");
    let plan = Plan::of(
        &rig.trace,
        &prospect.planes,
        Budgets {
            device: Some(DEVICE),
            host: Some(HOST),
        },
    )
    .expect("a plan past both budgets is planned, not refused");
    assert!(
        plan.spill_demand() > 0,
        "two budgets under the table have to spill, or this gate tests nothing"
    );
    assert!(
        plan.mapped_image() >= plan.spill_demand(),
        "the section reserves at least what the groups publish"
    );
    let key = engine_cuda::weights::tier_key(&rig.trace, &rig.contract, &rig.checkpoint)
        .expect("the key is a function of the trace and the recipe")
        .expect("this plan serializes, so this deployment forms a key");
    let tiers = tier::path(&dir.0, key);
    let bootstrap = engine_cuda::weight_cache::artifact_path(&dir.0, prospect.resident_key);
    let aside = dir.0.join("bootstrap.moved-away");
    eprintln!(
        "gpt-oss-20b at {} + {} GiB: T0 {} bytes, T1 {} bytes, T2 {} bytes in a \
         {}-byte section; tier key {key:016x}",
        DEVICE >> 30,
        HOST >> 30,
        plan.device_demand(),
        plan.host_image(),
        plan.spill_demand(),
        plan.mapped_image(),
    );

    // ── (0) THE REFUSAL IS STILL THE REFUSAL. An empty cache is neither a
    //    bootstrap nor a tier file, and a spilled plan with neither is a load
    //    this machine cannot serve. The W-1 gate's claim (c), restated here
    //    because this phase is the one that widens what counts as a source and
    //    a widening that admitted NOTHING would be the same widening.
    //
    //    `Shell::load` is the SHELL's door and asks no statute — the runtime's
    //    `Residency::admit_tiers` sits one level up, in `api.rs`, and is not
    //    on this path. **AND WHICH OF THE SHELL'S DOORS REFUSES HAS MOVED.**
    //    It used to be `Tier::open`, finding no mapping to seat a spilled
    //    group against, which is a refusal about T2 in particular. §M-3 puts a
    //    door in front of it: the serving artifact is opened before the pinned
    //    tier is allocated, and a streamed load with none refuses THERE, for
    //    every rung at once and before a byte is spent. So the sentence names
    //    the artifact and the command that writes it rather than the tier.
    //
    //    `tier_spill` below is the exact predicate the statute's `sourced` is
    //    computed from, asserted on its own beside the load, because that
    //    statute is unchanged and is what `Cuda::prepare` is admitted by.
    assert!(
        engine_cuda::weights::tier_spill(Some(&dir.0), key).is_none(),
        "nothing is under the tier key before the prepare"
    );
    let said = match boot(&rig, plan.clone(), &dir.0) {
        Err(why) => format!("{why}"),
        Ok(_) => panic!("a spilled plan with an empty cache cannot be served"),
    };
    assert!(
        said.contains("pie model import --prepare-only"),
        "the refusal names the command that writes one: {said}"
    );
    assert!(
        said.contains("never been prepared"),
        "and an empty cache directory is the never-prepared case, not the \
         changed-recipe one: {said}"
    );
    eprintln!("with an empty cache: {said}");

    // ── (1) THE BOOTSTRAP, AND THE GOLDEN. The one boot this deployment needs
    //    a machine large enough for, once, ever.
    let mut resident =
        boot(&rig, Plan::default(), &dir.0).expect("the uncapped shell loads");
    assert!(resident.weights_resident());
    let (golden, golden_rows) = run(&mut resident, &prompt);
    let says = rig.tokenizer.decode(&golden, false);
    drop(resident);
    eprintln!("uncapped answers: {says:?}");
    assert!(
        bootstrap.is_file(),
        "the uncapped boot writes the whole-table artifact at {}",
        bootstrap.display()
    );
    assert!(
        golden.iter().collect::<std::collections::BTreeSet<_>>().len() > 1,
        "the uncapped load answered {golden:?}, which is one token repeated"
    );

    // ── (2) THE SPILLED PREPARE, AND THE FILE IT LEAVES. Sourced by the
    //    bootstrap, and writing the artifact that replaces it.
    //
    //    **THIS WAS A `Shell::load` AND IT CANNOT BE ONE** (§M-3). The landing
    //    is identical — same bind, same compile, same transforms, same write —
    //    and what is gone is the shell built on top of it, which is device
    //    memory a run that is about to exit has no use for. `cold` is that
    //    landing's cost, and claim 5 measures the warm boot against it.
    let before = tier::observed();
    let clock = Instant::now();
    prepare(&rig, plan.clone(), &dir.0)
        .expect("a model larger than both budgets prepares out of the bootstrap");
    let cold = clock.elapsed();
    let after_cold = tier::observed();
    eprintln!(
        "SPILLED PREPARE: {:.1} s, left {after_cold:?}",
        cold.as_secs_f64()
    );
    assert_eq!(
        after_cold.stored,
        before.stored + 1,
        "one prepare, one tier artifact: {before:?} -> {after_cold:?}"
    );
    assert_eq!(
        after_cold.declined, before.declined,
        "and the writer no longer declines a plan that spills"
    );
    assert!(tiers.is_file(), "under the key the load forms: {tiers:?}");

    // ── (2b) AND EVERY SPILLED PLANE IS AN IMAGE IN IT, at its own span, with
    //     every block hashing to what the table states.
    //
    //     What this checked before §M was that the file's THIRD SECTION was
    //     `plan.mapped_image()` bytes long — the file's shape being this
    //     budget pair's split. There is no third section: a spilled plane is
    //     an image like every other, at the span the ranking gave it, and what
    //     makes it T2 is where THIS boot's cut put it. So the claim is asked
    //     one plane at a time instead, which is also what `Tier::open` asks.
    let artifact = tier::Artifact::open(&tiers).expect("the serving artifact opens");
    let head = artifact.head();
    let mapped = plan.mapped_layout();
    assert!(!mapped.is_empty(), "this plan spills, or the gate tests nothing");
    let mut spilled = 0u64;
    for (param, _, bytes, reserved) in &mapped {
        let id = u32::try_from(*param).expect("a param ordinal");
        let entry = artifact
            .resolve(id)
            .unwrap_or_else(|| panic!("the file carries no image for spilled param {id}"));
        assert_eq!(entry.plane, 0, "one image per param on this plane");
        assert_eq!(entry.bytes, *bytes, "publishing what the plan publishes");
        assert_eq!(entry.reserved, *reserved, "and reserving what it reserves");
        let seen = artifact
            .plane(id)
            .unwrap_or_else(|| panic!("param {id}'s bytes are inside the payload"));
        assert_eq!(seen.len() as u64, *bytes, "and the window is its published length");
        spilled += entry.reserved;
    }
    assert_eq!(
        spilled,
        plan.mapped_image(),
        "the spilled images add up to what this plan asks the mapping for"
    );
    assert_eq!(
        head.payload_at % tier::TIER_ALIGN,
        0,
        "and the payload starts on the format's own boundary"
    );
    eprintln!(
        "serving artifact: {} ({} bytes on disk), {} images tiling {} payload bytes at \
         {}, of which {} images / {spilled} bytes are what this cut maps",
        tiers.display(),
        std::fs::metadata(&tiers).map(|meta| meta.len()).unwrap_or(0),
        head.entries,
        head.payload_total,
        head.payload_at,
        mapped.len(),
    );
    artifact
        .verify()
        .expect("every block hashes to what the table states");
    drop(artifact);
    // **AND THE STATUTE'S OWN PREDICATE HAS FLIPPED.** `Residency::admit_tiers`
    // is asked with a `sourced` the shell computes from these two doors; the
    // tier file now answers where only the bootstrap did, which is the
    // acceptance half of this phase.
    assert!(
        engine_cuda::weights::tier_spill(Some(&dir.0), key).is_some(),
        "the tier artifact this boot wrote is a T2 source for the plan that wrote it"
    );

    // ── (3) ALONE. The bootstrap goes away, and the spilled plan boots
    //    anyway: `admit_tiers` is sourced by the tier file, both restorable
    //    images come off it, and T2 is mapped out of its own third section.
    std::fs::rename(&bootstrap, &aside).expect("the bootstrap moves aside");
    assert!(!bootstrap.exists(), "and it is not on the disk any more");
    let before_warm = engine_cuda::experts::observed();
    let clock = Instant::now();
    let mut warm_shell = boot(&rig, plan.clone(), &dir.0)
        .expect("the tier artifact is a source of its own for the plan that wrote it");
    let warm = clock.elapsed();
    let after_tiers = tier::observed();
    let after_warm = engine_cuda::experts::observed();
    assert!(
        warm_shell.weights_from_cache(),
        "the second spilled boot reads the two restorable images off the file"
    );
    assert!(!warm_shell.weights_resident(), "and it is still a streamed load");
    assert!(
        after_warm.seated > before_warm.seated && after_warm.bytes > before_warm.bytes,
        "and it seated the mapped tier: {before_warm:?} -> {after_warm:?}"
    );
    assert_eq!(after_warm.absent, before_warm.absent, "refusing no plane");
    assert_eq!(
        after_tiers.restored,
        after_cold.restored + 1,
        "counted as a restore: {after_cold:?} -> {after_tiers:?}"
    );
    assert_eq!(
        after_tiers.stored, after_cold.stored,
        "and a restore rewrites nothing — the file it read is already the answer"
    );
    assert_eq!(after_tiers.corrupt, after_cold.corrupt, "and nothing was wrong");
    let banks = warm_shell.expert_residency();
    assert!(
        banks.iter().any(|bank| bank.held == Some(Held::Mapped)),
        "and the tier reports which banks it mapped"
    );
    eprintln!(
        "WARM spilled boot, bootstrap deleted: {:.1} s ({:.1}x the prepare's \
         {:.1} s); T2 seated {} planes, {} bytes",
        warm.as_secs_f64(),
        cold.as_secs_f64() / warm.as_secs_f64().max(f64::MIN_POSITIVE),
        cold.as_secs_f64(),
        after_warm.seated - before_warm.seated,
        after_warm.bytes - before_warm.bytes,
    );

    // ── (4) THE SAME ANSWER, OUT OF A DIFFERENT FILE.
    let (warm_tokens, warm_rows) = run(&mut warm_shell, &prompt);
    drop(warm_shell);
    assert_eq!(
        warm_tokens, golden,
        "the boot that mapped its own spilled images chose {warm_tokens:?}"
    );
    same_logits(&golden_rows, &warm_rows, "the warm spilled boot");

    // ── (5) AND IT IS FASTER, WHICH IS WHAT WAS NEVER TRUE BEFORE.
    assert!(
        warm.as_secs_f64() * SPEEDUP < cold.as_secs_f64(),
        "the warm spilled boot took {:.1} s against the prepare's {:.1} s, which \
         is {:.1}x and this gate asks for {SPEEDUP}x",
        warm.as_secs_f64(),
        cold.as_secs_f64(),
        cold.as_secs_f64() / warm.as_secs_f64().max(f64::MIN_POSITIVE),
    );
    assert!(
        warm < CEILING,
        "the warm spilled boot took {:.1} s and the ceiling is {:.1} s",
        warm.as_secs_f64(),
        CEILING.as_secs_f64(),
    );

    // ── (6) THE ROT. One flipped bit in an image this cut maps, hashed at
    //    the door. With the bootstrap still aside there is no other source, so
    //    the load REFUSES by name — which is the honest outcome and not a
    //    fallback: the file that was the whole load cannot be trusted.
    let at = flip_a_mapped_byte(&tiers, &plan);
    eprintln!("flipped one bit at byte {at} of {tiers:?}, inside an image T2 maps");
    let before_rot = tier::observed();
    let said = match boot(&rig, plan.clone(), &dir.0) {
        Err(why) => format!("{why}"),
        Ok(_) => panic!("a corrupt mapped section is never served out of"),
    };
    let after_rot = tier::observed();
    eprintln!("with a rotted mapped section: {said}\nleft {after_rot:?}");
    assert_eq!(
        after_rot.corrupt,
        before_rot.corrupt + 1,
        "the corruption is counted where an operator can see it: \
         {before_rot:?} -> {after_rot:?}"
    );
    assert_eq!(
        after_rot.restored, before_rot.restored,
        "and it is not also counted as a restore"
    );
    // **AND THE FILE IS STILL THERE** (§M.4), which is the wave's flip. It
    // used to be deleted by the door that hashed it, and `tier_spill` — the
    // statute's own predicate — went back to false with it. Nothing deletes
    // now, so the predicate stays TRUE: a parseable file is still a candidate
    // source, and what refuses the load is the door that hashed the bytes.
    // The two questions were always different and the deletion had been
    // answering both.
    assert!(
        tiers.exists(),
        "the file that lied about its bytes is refused, not removed: {tiers:?}"
    );
    assert!(
        engine_cuda::weights::tier_spill(Some(&dir.0), key).is_some(),
        "a file that still parses is still a candidate source; existence is the \
         statute's question and the bytes are the load's"
    );
    assert!(
        said.contains("pie model import --prepare-only"),
        "and the refusal names the command that rebuilds it: {said}"
    );
    assert_eq!(
        after_rot.stored, before_rot.stored,
        "AND NOTHING WAS WRITTEN (§M-3). The serving path cannot reach \
         `tier::store` at all now, so a rotted spilled image costs a refusal \
         and not a hundred gigabytes rewritten under a deployment nobody asked \
         to rebuild: {after_rot:?}"
    );

    // ── (6b) AND THE RECOVERY, WHICH IS A COMMAND AND NOT A BOOT (§M-3). The
    //     bootstrap comes back — the prepare needs it, because the file it is
    //     about to write is the one that rotted — and `Shell::prepare` runs
    //     step 2 again: a fresh serving artifact, every spilled image back at
    //     its own span, every block hashing. The boot AFTER it is the one that
    //     answers, and it answers the golden.
    std::fs::rename(&aside, &bootstrap).expect("the bootstrap comes back");
    prepare(&rig, plan.clone(), &dir.0)
        .expect("with a source again, the prepare replaces the rotted artifact");
    let after_again = tier::observed();
    let mut again_shell = boot(&rig, plan.clone(), &dir.0)
        .expect("and the boot after the rebuild serves");
    assert!(
        again_shell.weights_from_cache(),
        "off the artifact the rebuild wrote"
    );
    let (again_tokens, again_rows) = run(&mut again_shell, &prompt);
    drop(again_shell);
    assert_eq!(
        after_again.stored,
        after_rot.stored + 1,
        "the rotted file is REPLACED, by the one door that ever overwrites one \
         (§M.4) and by the one command that reaches it (§M-3): \
         {after_rot:?} -> {after_again:?}"
    );
    assert_eq!(
        after_again.skipped, after_rot.skipped,
        "and the writer did not mistake it for the file it was about to write"
    );
    assert!(tiers.is_file(), "back under the same key");
    let rewritten = tier::Artifact::open(&tiers).expect("the rewritten artifact opens");
    let mut back = 0u64;
    for (param, _, _, _) in &plan.mapped_layout() {
        let id = u32::try_from(*param).expect("a param ordinal");
        back += rewritten
            .resolve(id)
            .unwrap_or_else(|| panic!("param {id} is missing from the rewrite"))
            .reserved;
    }
    assert_eq!(back, plan.mapped_image(), "with every spilled image back");
    rewritten
        .verify()
        .expect("and every block of it hashes to what its table states");
    assert_eq!(
        again_tokens, golden,
        "the boot after the rebuild chose {again_tokens:?}"
    );
    same_logits(&golden_rows, &again_rows, "the boot after the rebuild");
    assert!(
        std::fs::read_dir(&dir.0)
            .expect("the cache directory")
            .flatten()
            .all(|entry| !entry.file_name().to_string_lossy().contains(".part")),
        "and no `.part` is left behind"
    );
}
