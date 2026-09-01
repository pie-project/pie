//! **§L phase L-1: the warm boot answers BEFORE its pinned tier exists.**
//!
//! T-3 made the second boot of a streamed seat read its three images off one
//! file instead of recomputing them. What it did not change is the ORDER: the
//! whole page-locked tier is allocated and filled before the first token, and
//! on the SKU §L.0 measured that is forty to sixty seconds of a seventy-five
//! second boot — `cudaHostAlloc` over sixty-four gigabytes, then the read that
//! fills it. Neither term produces an answer.
//!
//! Neither term is needed to serve, either. A packed select loads its two
//! plane bases out of a cell and dereferences them; a dense bank's entry is
//! one address in a table. Neither can tell a page-locked pointer from a
//! mapped one — under `pageableMemoryAccess` both are host pointers the GPU
//! faults through — and the serving artifact carries every plane the pinned
//! tier holds, byte for byte (§M.3). So the deferred seat verifies THOSE
//! IMAGES WHERE THEY LIE, opens the tier over the mapping, and answers; a
//! background thread page-locks and fills the real image behind it, and one
//! inter-fire gap installs it.
//!
//! **THE INSTALL MOVES ONE ADDRESS PER SPAN AND NOT ONE BASE** (§M.3's change
//! to §L.1). Format 2's host section was the pinned allocation's whole image
//! at the same offsets, so the swap was a single number. §M's file is the
//! ranking, and the images a budget puts on T1 are a subset of that order
//! rather than a run of it — there is no base to swap. What §L.2's atomicity
//! argument actually rested on is untouched: every one of them moves inside
//! the same `drained`/`ready` bracket, followed by one `publish_all` and one
//! `publish_cells`, and both rungs are byte-identical verified content, so a
//! cell caught mid-flight names one of two addresses of the same bytes.
//!
//! ```text
//!  1. sooner   -> the deferred boot's first token beats the eager boot's and
//!                 lands under `TTFB`
//!  2. same     -> every logit it serves through the window is BIT-IDENTICAL
//!                 to the eager boot's — the window is a speed, not an answer
//!  3. installed-> `Shell::settle_tier_refill` closes it; `promoted` moves
//!                 once and `window_ms` says how long it was open
//!  4. still same-> after the install the census by rung is what it was, every
//!                 group reports the tier the eager boot reported for it, and
//!                 the logits are still bit-identical
//!  5. boot rot -> a byte flipped in an image this cut PINS, before the boot,
//!                 is caught by the in-place verify, counted, named and the
//!                 file LEFT ON THE DISK (§M.4) — and the boot REFUSES
//!                 (§M-3), because a rotted serving artifact is no longer a
//!                 load that runs cold. One `Shell::prepare` replaces it and
//!                 the seat serves the same answers again
//!  6. late rot -> a file replaced AFTER the boot verified it is caught by the
//!                 background fill, counted, named, and LEFT WHERE IT IS — and
//!                 the seat goes on serving the mapping it already checked,
//!                 never promoting
//! ```
//!
//! **(6) IS THE ARGUMENT FOR VERIFYING FIRST**, and it is why §L.3 rejected
//! the cheaper window. Because every byte this seat serves was hashed before a
//! kernel was pointed at it, a background fill that finds the file rotted under
//! it is a PERFORMANCE failure and nothing else: the load keeps its answers and
//! loses its promotion. A design that trusted the mapping and checked it lazily
//! would have had no moment to check it at — nothing hooks a first touch — and
//! its failure mode would have been tokens already handed to a caller.
//!
//! # What "eager" means here, and since §M-3 it is a boot that WAITS
//!
//! `defer_tiers` is unconditional: a warm streamed boot on a device with HMM
//! takes the deferred road, which is the entire point of the wave. So an eager
//! seat has to be made, and this file used to get one by accident — the eager
//! reference WAS the boot that materialized the artifact, and it page-locked
//! and filled its tier before it answered because there was nothing on the
//! disk to defer over. The paragraph that stood here said that process could
//! not produce an eager WARM boot at all, and pointed at
//! `a_second_streamed_boot_maps_the_tiers_it_wrote` for the number.
//!
//! §M-3 took the cold boot away — the write is `Shell::prepare`'s, and a
//! serving load that streams is warm or refused — so every boot below is a
//! warm one and every warm one defers. The eager reference is therefore made
//! ON PURPOSE: `Shell::settle_tier_refill` before the first fire, which joins
//! the fill thread and installs the page-locked image, so the seat that
//! answers is the seat the whole allocation-and-fill was paid for.
//!
//! **AND IT IS A BETTER REFERENCE THAN THE ONE IT REPLACES.** The two boots
//! now differ by exactly the `cudaHostAlloc` and the fill — the term this wave
//! claims to move — where before they differed by that plus a whole cold
//! landing, which no amount of deferring was ever going to explain. The
//! arithmetic to hold the printed numbers against is the sibling gate's
//! decomposition of a warm streamed boot: ~2.2 s metadata and compile, ~5.3 s
//! `cudaHostAlloc`, ~2.8 s restore.
//!
//! # What is REPORTED and not gated
//!
//! The window's throughput. §L.5 is explicit that this wave buys the first
//! token and not the tokens after it: while the fill runs, every T1 plane a
//! fire reads is an NVMe page fault over HMM, the fill owns the same disk, and
//! its `cudaHostAlloc` holds the runtime's memory-manager lock. Total tokens by
//! the minute mark are close to a wash. Printing the two step times is how that
//! sentence stays honest; gating on them would be gating on the disk.
//!
//! ```bash
//! cargo test -p engine-cuda --features cuda-13 --release \
//!     --test a_deferred_tier_serves_before_it_is_pinned -- --ignored --nocapture
//! ```
//!
//! # Gating
//!
//! `#[ignore]`d: it wants a CUDA device reporting `pageableMemoryAccess`, the
//! gpt-oss-20b snapshot on disk, and room under `TMPDIR` for the artifact
//! (~15 GiB at this budget). It runs TWO prepares and FIVE boots,
//! sequentially — the prepares are the two cold landings (the first write and
//! the rot's replacement), and one of the five boots is a refusal that costs a
//! plan compile and a verify rather than a load. Skips with a sentence when
//! any of it is missing.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use engine_cuda::experts::{Budgets, Held, Plan};
use engine_cuda::weight_cache::tier;
use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "gptoss-20b-bf16-mxfp4-kv-bf16";

/// **The device budget, and it is the T-3 triangle's exactly.** Four gibibytes
/// under a table of roughly thirteen: whole mxfp4 groups land on the pinned
/// tier, the host tier is uncapped so none of them reach the mapping, and the
/// load streams — which is the only shape that has a host section to defer.
const DEVICE: u64 = 4 << 30;

/// How much room the artifact wants under `TMPDIR`, plus the writer's margin.
const ROOM: u64 = 24 << 30;

/// **THE PIN** (§L.8, phase L-2): the deferred boot's first token, from the
/// call to `Shell::load` to the prefill's logits.
///
/// **Twenty, where L-1 pinned twenty-five.** The number moved with
/// [`TIER_STRIPES`](engine_cuda::weight_cache::tier::TIER_STRIPES): the host
/// section's in-place verify is what the deferred boot pays before it answers,
/// and eight chains hide more of it under the device pump than four did.
/// Measured on the L40S box, back to back, at this SKU's 9.59 GB host section:
///
/// ```text
///   stripes  load    first token  window
///   4        9.6 s   12.8 s       8745 ms
///   8        7.9 s   10.9 s       8108 ms
///   8        8.3 s   11.6 s       8765 ms   <- a second run, same build
/// ```
///
/// Three runs and not two, because the spread is most of a second and a
/// one-sample delta of 1.9 s would have been a claim the box does not support:
/// what is repeatable is between one and two seconds off the first token.
///
/// The delta is that small because gpt-oss is SMALL — its host section
/// is a seventh of the SKU §L.0 measured, and a hash that was never the
/// binding term cannot give up much when it halves. §L.3's 18.6 s -> 12.7 s is
/// the number at the scale the wave is for. So this pin is at twenty with room
/// under it, and what it really gates is the shape: a boot that stopped
/// deferring, or a verify that stopped overlapping, lands the other side of it.
const TTFB: Duration = Duration::from_secs(20);

const PROMPT: &str = "<|start|>user<|message|>What is the capital of France? \
                      Answer in one word.<|end|>\
                      <|start|>assistant<|channel|>final<|message|>";

/// Long enough for a step time to be an average rather than a sample of the
/// first cold fire, short enough that the window is still open at the end of
/// it — the fill's own `cudaHostAlloc` is seconds and the read behind it is
/// seconds more, and twelve decodes of this SKU are not.
const STEPS: usize = 12;

fn word(query_len: u32) -> u64 {
    models::gpt_oss::forward::Facts::of(&Request::new(query_len, false)).word()
}

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

/// The trace, the contract, the checkpoint and the tokenizer — everything the
/// four boots share, read once.
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

/// **ONE DOCUMENT, TWO DOORS** (§M-3). The prepare and the boots have to
/// describe the same deployment in every field or they name two different
/// files — the serving artifact's key is a function of the trace, the recipe
/// and the ranking — so the gate states it once and hands it to both.
fn doc<'a>(rig: &'a Rig, residency: Plan, dir: &'a Path) -> Boot<'a> {
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
        residency,
    }
}

fn load(rig: &Rig, residency: Plan, dir: &Path) -> engine_cuda::Result<Shell> {
    Shell::load(doc(rig, residency, dir))
}

/// **THE WRITER**, and since §M-3 the only one there is. `pie model import
/// --prepare-only` reaches it through `Cuda::prepare`; the gate reaches it
/// directly, because what these claims are about is the file and not the
/// plumbing. It runs the bake and the landing a cold boot used to run, writes
/// the artifact and keeps nothing — no arena, no pools, nothing on the device.
fn prepare(rig: &Rig, residency: Plan, dir: &Path) -> engine_cuda::Result<()> {
    Shell::prepare(doc(rig, residency, dir))
}

/// **A boot, timed to its FIRST TOKEN** — the number this wave exists to move.
///
/// The clock starts before `Shell::load` and stops when the prefill's logits
/// are in hand, because that is what an operator waits for: a load that
/// returned and a slot that has not answered is not a served deployment.
struct Booted {
    shell: Shell,
    loaded: Duration,
    first: Duration,
    row: Vec<f32>,
}

fn boot(rig: &Rig, plan: Plan, dir: &Path, prompt: &[u32]) -> engine_cuda::Result<Booted> {
    let clock = Instant::now();
    let shell = load(rig, plan, dir)?;
    answer(shell, clock, prompt)
}

/// **A BOOT THAT WAITS FOR ITS PINNED IMAGE BEFORE IT ANSWERS** — the eager
/// seat, and since §M-3 this is the only way to make one (the header argues
/// why at length). `settle_tier_refill` joins the fill thread and installs
/// synchronously, so what comes back has paid the `cudaHostAlloc` and the fill
/// the deferred boot below is racing, and its `loaded` includes both.
///
/// It is asserted rather than reported because a `false` here would mean the
/// seat never deferred at all — a warm streamed boot on an HMM device that
/// took the eager road is a policy change, not a slower run — and every
/// comparison in this file would then be one seat against itself.
fn eager_boot(rig: &Rig, plan: Plan, dir: &Path, prompt: &[u32]) -> engine_cuda::Result<Booted> {
    let clock = Instant::now();
    let mut shell = load(rig, plan, dir)?;
    assert!(
        shell.settle_tier_refill()?,
        "the reference boot deferred nothing to install, so there is no eager seat \
         here to race against"
    );
    answer(shell, clock, prompt)
}

/// The half both doors share: open slot 0, prefill, and stop the clock on the
/// logits.
fn answer(mut shell: Shell, clock: Instant, prompt: &[u32]) -> engine_cuda::Result<Booted> {
    let loaded = clock.elapsed();
    shell.open(0).expect("slot 0 opens");
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the prefill fires");
    let first = clock.elapsed();
    finite(&prefill[0], "prefill");
    Ok(Booted {
        shell,
        loaded,
        first,
        row: prefill[0].clone(),
    })
}

/// `STEPS` greedy decodes fed back into slot 0, from `seed`'s argmax.
///
/// Answers the tokens, the rows they were chosen from — `seed` first, so a
/// comparison covers the prefill too — and the mean decode step time.
fn decode(shell: &mut Shell, seed: &[f32], what: &str) -> (Vec<u32>, Vec<Vec<f32>>, f64) {
    let mut chosen = Vec::with_capacity(STEPS + 1);
    let mut rows = Vec::with_capacity(STEPS + 1);
    let mut fed = argmax(seed);
    chosen.push(fed);
    rows.push(seed.to_vec());
    let began = Instant::now();
    for step in 0..STEPS {
        let out = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &[fed],
            }])
            .unwrap_or_else(|why| panic!("{what} decode step {step} fires: {why}"));
        finite(&out[0], what);
        fed = argmax(&out[0]);
        chosen.push(fed);
        rows.push(out[0].clone());
    }
    let each = began.elapsed().as_secs_f64() * 1e3 / STEPS as f64;
    (chosen, rows, each)
}

/// The same slot, re-opened, so a second turn starts where the first did.
fn reopen(shell: &mut Shell, prompt: &[u32], what: &str) -> Vec<f32> {
    shell.open(0).expect("slot 0 re-opens");
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .unwrap_or_else(|why| panic!("{what} prefill fires: {why}"));
    finite(&prefill[0], what);
    prefill[0].clone()
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

fn finite(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits at all");
    for (at, value) in logits.iter().enumerate() {
        assert!(value.is_finite(), "{what} logit {at} is {value}");
    }
    let spread = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-3,
        "{what} logits span {spread}, which is a rectangle nothing wrote"
    );
}

/// **The same logits, bit for bit** — the one claim a residency change, or a
/// base address moving under a running model, may not move.
fn identical(golden: &[Vec<f32>], also: &[Vec<f32>], what: &str) {
    assert_eq!(golden.len(), also.len(), "{what} served {} rows", also.len());
    for (step, (a, b)) in golden.iter().zip(also).enumerate() {
        assert_eq!(
            a.len(),
            b.len(),
            "step {step} produced {} logits eagerly and {} {what}",
            a.len(),
            b.len()
        );
        for (at, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "step {step}, logit {at}: eager {x}, {what} {y} — the seat moved a number"
            );
        }
    }
}

/// Every packed group's `(name, tier)`, from the tier's own live report.
fn groups(shell: &Shell) -> Vec<(String, Held)> {
    shell
        .expert_residency()
        .into_iter()
        .filter_map(|bank| Some((bank.name, bank.held?)))
        .collect()
}

/// How many groups sit on each rung: device, pinned, mapped.
fn census(report: &[(String, Held)]) -> [usize; 3] {
    let mut rungs = [0usize; 3];
    for (_, held) in report {
        rungs[held.rung() as usize] += 1;
    }
    rungs
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

/// **Flip one bit in the middle of an image this cut PINS, IN PLACE** — the
/// T-3 gate's own rot, for the boot-path claim.
///
/// The middle rather than the first byte: a reader that checked only its first
/// block, or only its header, would pass a corruption at byte zero for the
/// wrong reason.
fn flip_a_pinned_byte(path: &Path, plan: &Plan) -> u64 {
    use std::os::unix::fs::FileExt;

    let artifact = tier::Artifact::open(path).expect("the artifact opens");
    let head = artifact.head();
    let layout = plan.host_layout();
    let (param, _, _, _) = *layout
        .get(layout.len() / 2)
        .expect("this plan seats bytes on the pinned tier");
    let group = artifact
        .resolve(u32::try_from(param).expect("a param ordinal"))
        .expect("the file carries every plane this plan pins");
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

/// **REPLACE the file with a rotten copy of itself** — the late rot, and it has
/// to be a NEW INODE.
///
/// The mapping a deferred seat serves out of is `PROT_READ, MAP_PRIVATE` over
/// this path: a write into the same inode would be visible through it, which
/// would corrupt the bytes being served rather than testing what happens when
/// the FILE goes bad behind them. A rename swaps the directory entry and leaves
/// the mapped inode exactly as it was, which is the property a seat serving out
/// of a mapping has always stood on.
///
/// The header, the index and the block table are copied verbatim and the
/// PAYLOAD is left sparse, so the copy parses as an artifact under this key and
/// states the ORIGINAL digests over bytes that are zeros. It is caught twice
/// over: by its own table, and by the block digests the fill thread carries
/// across from the inode the seat is serving.
///
/// **AND IT HAS TO BE INSTANT.** The window it has to land inside is the fill
/// thread's `cudaHostAlloc`, which is seconds and not minutes; a copy of
/// fourteen gigabytes takes longer than the whole window and would be racing
/// the read rather than beating it. A sparse `set_len` over a two-megabyte
/// prefix is two syscalls.
/// **HOW MANY MAPPINGS OF `path` THIS PROCESS HOLDS**, out of the kernel's
/// own table — the observable arm (6) times its rename off.
///
/// A mapping whose file has been renamed away is listed with a ` (deleted)`
/// on the end and stops matching, which is what makes a COUNT of the live
/// name the right question: the number rises as this load seats itself over
/// the artifact and falls again when the shell that held it goes.
fn mappings_of(path: &Path) -> usize {
    let name = path.to_string_lossy().into_owned();
    std::fs::read_to_string("/proc/self/maps")
        .map(|maps| maps.lines().filter(|line| line.ends_with(&name)).count())
        .unwrap_or(0)
}

fn replace_with_a_rotten_copy(path: &Path) -> u64 {
    use std::io::Write;

    let holds = std::fs::metadata(path).expect("the artifact stats").len();
    let head = tier::Artifact::open(path).expect("the artifact opens").head();
    // Everything before the payload: the header, the index and the block
    // table, which is what makes the result an artifact and not a blank file.
    let keep = usize::try_from(head.payload_at).expect("a payload start");
    let mut prefix = vec![0u8; keep];
    {
        use std::os::unix::fs::FileExt;
        std::fs::File::open(path)
            .expect("the artifact reads")
            .read_exact_at(&mut prefix, 0)
            .expect("the header and the index runs read");
    }
    let beside = path.with_extension("rotten");
    let mut file = std::fs::File::create(&beside).expect("the rotten copy is created");
    file.write_all(&prefix).expect("its header writes");
    file.set_len(holds).expect("and it is the length it claims");
    file.sync_all().expect("the rotten copy reaches the disk");
    drop(file);
    std::fs::rename(&beside, path).expect("the rotten copy takes the name");
    head.payload_at
}

// ─────────────────────────────────────────────────────────────────────────────

/// **All six claims over two prepares, five boots and one artifact.**
#[test]
#[ignore = "real-hardware: needs a CUDA device with HMM, a local gpt-oss-20b \
            snapshot and ~24 GiB under TMPDIR; run it with `-- --ignored`"]
fn a_deferred_tier_serves_before_it_is_pinned() {
    let Some(rig) = rig("the deferred-seat gate") else {
        return;
    };
    if !engine_cuda::experts::pageable_access() {
        eprintln!(
            "skipping the deferred-seat gate: this device does not report \
             `pageableMemoryAccess`, so a mapped host pointer is not a device pointer \
             and the deferred road falls back to the eager one by policy (§L, H4)"
        );
        return;
    }
    let dir = scratch("deferred-tier");
    if free(&dir.0) < ROOM {
        eprintln!(
            "skipping the deferred-seat gate: {:?} has {} GiB free and the artifact \
             wants {} GiB",
            dir.0,
            free(&dir.0) >> 30,
            ROOM >> 30
        );
        return;
    }
    let prompt = rig.tokenizer.encode(PROMPT);
    assert!(prompt.len() > 4, "the harmony turn encodes to something");

    // ── THE PLAN AND THE KEY, BOTH FORMED FROM OUTSIDE THE LOAD.
    let prospect = engine_cuda::weights::prospect(&rig.trace, &rig.contract, &rig.checkpoint)
        .expect("the load plan pairs every packed bank with its scales");
    let plan = Plan::of(&rig.trace, &prospect.planes, Budgets::device(DEVICE))
        .expect("a capped mxfp4 MoE plans rather than refusing");
    assert!(plan.streams(), "a 4 GiB budget under this table has to stream");
    assert!(
        plan.host_image() > 0,
        "a seat with nothing on T1 has nothing to defer"
    );
    assert_eq!(
        plan.spill_demand(),
        0,
        "the host tier is uncapped, so nothing reaches the mapping and every \
         deferred address below is a T1 address"
    );
    let key = engine_cuda::weights::tier_key(&rig.trace, &rig.contract, &rig.checkpoint)
        .expect("the key is a function of the trace and the recipe")
        .expect("this plan serializes, so this deployment forms a key");
    let path = tier::path(&dir.0, key);
    assert!(!path.exists(), "nothing is cached before the prepare");
    eprintln!(
        "gpt-oss-20b at a {} GiB device budget: {} bytes on the device, {} on the \
         pinned tier; key {key:016x}",
        DEVICE >> 30,
        plan.device_demand(),
        plan.host_image(),
    );

    // ── THE PREPARE, WHICH IS WHERE THE ARTIFACT COMES FROM NOW (§M-3).
    //    This used to be the eager boot's side effect: nothing was on the
    //    disk, so the first streamed load ran the executor and wrote what it
    //    had materialized on its way to serving. `Shell::load` cannot do that
    //    any more — a streamed serve is warm or refused — so the cold landing
    //    is here, in the one door that still runs it, and every boot below
    //    reads what it leaves.
    let before = tier::observed();
    let cold = Instant::now();
    prepare(&rig, plan.clone(), &dir.0).expect("the prepare lands and writes");
    let landing = cold.elapsed();
    let stored = tier::observed();
    eprintln!(
        "PREPARE: landed and wrote in {:.1} s, {stored:?}",
        landing.as_secs_f64()
    );
    assert_eq!(
        stored.stored,
        before.stored + 1,
        "one prepare, one serving artifact: {before:?} -> {stored:?}"
    );
    assert!(path.exists(), "under the key the boots form: {path:?}");

    // ── THE EAGER BOOT, AND IT IS EAGER BECAUSE IT WAITS. It restores like
    //    every warm streamed boot, takes the deferred seat like every warm
    //    streamed boot on this device, and then closes its own window before
    //    it fires a single token — so what answers is the page-locked image
    //    and not the mapping. Its logits are the golden and its first token is
    //    the number to beat.
    let before = tier::observed();
    let eager = eager_boot(&rig, plan.clone(), &dir.0, &prompt).expect("the eager shell loads");
    let (eager_first, eager_loaded) = (eager.first, eager.loaded);
    let mut shell = eager.shell;
    assert!(
        !shell.weights_resident(),
        "a 4 GiB budget under a 13 GiB table streams, or this gate tests nothing"
    );
    assert!(
        shell.weights_from_cache(),
        "and since §M-3 there is nowhere else it could have come from: the boot \
         that used to land this table cold is the prepare above"
    );
    let (golden, golden_rows, eager_ms) = decode(&mut shell, &eager.row, "eager");
    let says = rig.tokenizer.decode(&golden, false);
    let eager_groups = groups(&shell);
    drop(shell);
    let after_eager = tier::observed();
    eprintln!(
        "EAGER boot: loaded and installed in {:.1} s, first token at {:.1} s, \
         {eager_ms:.2} ms/step, answered {says:?}",
        eager_loaded.as_secs_f64(),
        eager_first.as_secs_f64(),
    );
    assert_eq!(
        after_eager.stored, before.stored,
        "a serving load cannot write a serving artifact (§M-3), whichever seat it \
         takes: {before:?} -> {after_eager:?}"
    );
    assert_eq!(
        after_eager.restored,
        before.restored + 1,
        "it read the prepare's file instead: {before:?} -> {after_eager:?}"
    );
    assert_eq!(
        after_eager.deferred,
        before.deferred + 1,
        "and it deferred like every warm streamed boot on an HMM device — what \
         makes it the EAGER reference is the install below, not a road it declined"
    );
    assert_eq!(
        after_eager.promoted,
        before.promoted + 1,
        "which it took before it fired a token, so its answers are the pinned \
         image's: {before:?} -> {after_eager:?}"
    );
    assert!(path.exists(), "under the key the load forms: {path:?}");
    assert!(
        golden.iter().collect::<std::collections::BTreeSet<_>>().len() > 1,
        "the eager load answered {golden:?}, which is one token repeated"
    );
    assert!(
        !eager_groups.is_empty(),
        "the tier reports its packed groups, or claim (4) compares two empty lists"
    );

    // ── (1) THE DEFERRED BOOT ANSWERS SOONER.
    let before_warm = tier::observed();
    let warm = boot(&rig, plan.clone(), &dir.0, &prompt).expect("the deferred shell loads");
    let (warm_first, warm_loaded) = (warm.first, warm.loaded);
    let mut shell = warm.shell;
    let after_warm = tier::observed();
    assert!(
        shell.weights_from_cache(),
        "the second boot at the same seat should have read the artifact"
    );
    assert!(!shell.weights_resident(), "and it is still a streamed load");
    assert_eq!(
        after_warm.deferred,
        before_warm.deferred + 1,
        "the warm boot took the deferred seat: {before_warm:?} -> {after_warm:?}"
    );
    assert_eq!(
        after_warm.restored,
        before_warm.restored + 1,
        "and it is a restore like any other — it skipped the cold branch"
    );
    assert_eq!(
        after_warm.promoted, before_warm.promoted,
        "the window is still open the instant the boot returns"
    );
    assert_eq!(
        after_warm.corrupt, before_warm.corrupt,
        "and nothing about the file was wrong"
    );
    eprintln!(
        "DEFERRED boot: loaded in {:.1} s, first token at {:.1} s (eager {:.1} s)",
        warm_loaded.as_secs_f64(),
        warm_first.as_secs_f64(),
        eager_first.as_secs_f64(),
    );
    assert!(
        warm_first < eager_first,
        "the deferred boot's first token came at {:.1} s and the eager boot's at \
         {:.1} s; the whole wave is that inequality",
        warm_first.as_secs_f64(),
        eager_first.as_secs_f64(),
    );
    assert!(
        warm_first < TTFB,
        "the deferred boot's first token came at {:.1} s and §L.8 pins {:.1} s at \
         eight stripes",
        warm_first.as_secs_f64(),
        TTFB.as_secs_f64(),
    );

    // ── (2) AND IT IS THE SAME ANSWER THROUGH THE WINDOW. Every T1 plane
    //    these fires read is a page of the artifact, faulted in over HMM, at
    //    the offset the pinned image would have held it at.
    let (window, window_rows, window_ms) = decode(&mut shell, &warm.row, "the window");
    assert_eq!(
        window, golden,
        "the deferred load chose {window:?} through the window and the eager one \
         chose {golden:?}"
    );
    identical(&golden_rows, &window_rows, "the window");
    let window_groups = groups(&shell);
    assert_eq!(
        window_groups, eager_groups,
        "a deferred seat reports the tier the PLAN gave each group — the base moved, \
         not the rung"
    );

    // ── (3) THE WINDOW CLOSES, THROUGH THE DOOR.
    //
    //    The install rides the inter-fire gap, so a shell nobody is firing
    //    never takes it and a gate cannot observe one by waiting (§L, H5). The
    //    door joins the fill thread and installs synchronously; it answers
    //    `false` when a gap got there first, which is a stronger outcome and
    //    not a different one — what is asserted is the COUNTER either way.
    let by_the_door = shell
        .settle_tier_refill()
        .expect("the deferred seat installs");
    let settled = tier::observed();
    assert_eq!(
        settled.promoted,
        before_warm.promoted + 1,
        "one deferred seat, one install, however it was reached: \
         {before_warm:?} -> {settled:?}"
    );
    assert!(
        settled.window_ms > 0,
        "the window was open for {} ms, which is a window that never opened",
        settled.window_ms
    );
    eprintln!(
        "the window closed {} after {} ms; {settled:?}",
        match by_the_door {
            true => "through the door",
            false => "in an inter-fire gap before the door was reached",
        },
        settled.window_ms,
    );
    assert!(
        !shell.settle_tier_refill().expect("a closed window closes once"),
        "a window that is already closed cannot close again"
    );

    // ── (4) AND THE INSTALLED SEAT IS THE EAGER SEAT.
    let after_groups = groups(&shell);
    assert_eq!(
        after_groups, eager_groups,
        "after the install every group reports the tier the eager boot reported \
         for it, name for name"
    );
    assert_eq!(
        census(&after_groups),
        census(&window_groups),
        "the install moved a base address; it may not move a group between rungs"
    );
    let seed = reopen(&mut shell, &prompt, "post-install");
    let (again, again_rows, installed_ms) = decode(&mut shell, &seed, "post-install");
    assert_eq!(
        again, golden,
        "after the install the load chose {again:?} and the eager one chose {golden:?}"
    );
    identical(&golden_rows, &again_rows, "post-install");
    drop(shell);
    eprintln!(
        "step time: eager {eager_ms:.2} ms, deferred window {window_ms:.2} ms \
         ({:.2} tok/s), after the install {installed_ms:.2} ms ({:.2} tok/s) — \
         REPORTED, not pinned: this wave buys the first token, not the throughput",
        1e3 / window_ms,
        1e3 / installed_ms,
    );

    // ── (5) BOOT-PATH ROT: THE SEAT REFUSES, AND A PREPARE IS THE REPAIR.
    //
    //    **THE DETECTION IS T-3's AND IT HAS NOT MOVED.** The host section is
    //    not hashed by `open_tiers` — the eager road checks it as it crosses
    //    and the deferred road checks it where it lies — so this flip is
    //    caught by the in-place verify, in the same scope, beside the same
    //    device pump, and counted at the door that hashed it.
    //
    //    **WHAT THE LOAD DOES ABOUT IT IS §M-3's AND IT IS THE OPPOSITE.**
    //    T-3 wrote this claim as a RECOVERY: the file was left where it was
    //    (§M.4) and the boot ran the full cold load, replacing it by its own
    //    write. There is no cold serving path left, so a rotted artifact buys
    //    a serving load the same refusal a missing one buys — the sentence,
    //    with `pie model import --prepare-only` in it — and the rewrite is
    //    that remedy being run, below. The deferred arm's extra debt (there is
    //    no pinned image to zero, so `Tier::undefer` makes one) is the
    //    PREPARE's now: a load that is about to return an error pays no
    //    recovery, because it continues nowhere.
    let at = flip_a_pinned_byte(&path, &plan);
    eprintln!("flipped one bit at byte {at} of {path:?}, inside an image T1 holds");
    let rotted = std::fs::metadata(&path).expect("the rotted file stats");
    let (rotted_len, rotted_at) = (
        rotted.len(),
        rotted.modified().expect("a modification time"),
    );
    let before_rot = tier::observed();
    let refused = boot(&rig, plan.clone(), &dir.0, &prompt)
        .err()
        .expect("a rotted serving artifact does not serve");
    let sentence = format!("{refused:?}");
    let after_rot = tier::observed();
    eprintln!("BOOT-ROT boot refused: {sentence}");
    assert!(
        sentence.contains("pie model import --prepare-only"),
        "the refusal names the command that fixes it: {sentence}"
    );
    assert_eq!(
        after_rot.corrupt,
        before_rot.corrupt + 1,
        "the corruption is counted where an operator can see it: \
         {before_rot:?} -> {after_rot:?}"
    );
    assert_eq!(
        after_rot.deferred, before_rot.deferred,
        "a window that never opened is not counted as one"
    );
    assert_eq!(
        after_rot.promoted, before_rot.promoted,
        "and nothing was installed"
    );
    assert_eq!(
        after_rot.restored, before_rot.restored,
        "and a restore that refused is not counted as one"
    );
    assert_eq!(
        after_rot.stored, before_rot.stored,
        "and the boot that found the bytes wrong wrote nothing: since §M-3 it \
         could not — `write_tiers` is behind `Intent::Prepare`: {after_rot:?}"
    );
    // **AND THE FILE IS EXACTLY AS THE FLIP LEFT IT** (§M.4). Not deleted,
    // not truncated, not rewritten — the boot that cannot read a hundred
    // gigabytes is not the boot that gets to replace them.
    let still = std::fs::metadata(&path).expect("the rotted file is still there");
    assert_eq!(still.len(), rotted_len, "the refusal changed its length");
    assert_eq!(
        still.modified().expect("a modification time"),
        rotted_at,
        "the refusal rewrote it"
    );

    // ── AND THE PREPARE REPLACES IT, WHICH IS THE ONE DOOR THAT EVER DOES.
    //    It reaches the same verify at the same offset and says the sentence
    //    out loud this time — an import about to overwrite a rotted artifact
    //    owes an operator that line — then throws away what crossed, lands the
    //    checkpoint and writes the file again through `tier::store`'s
    //    verify-then-replace.
    let before_fix = tier::observed();
    prepare(&rig, plan.clone(), &dir.0).expect("the prepare replaces a rotted artifact");
    let after_fix = tier::observed();
    eprintln!("the prepare replaced the rotted artifact: {before_fix:?} -> {after_fix:?}");
    assert_eq!(
        after_fix.corrupt,
        before_fix.corrupt + 1,
        "it hashed the same image and found the same flip"
    );
    assert_eq!(
        after_fix.stored,
        before_fix.stored + 1,
        "and then wrote the replacement: {before_fix:?} -> {after_fix:?}"
    );
    assert_eq!(
        after_fix.skipped, before_fix.skipped,
        "and the writer did not mistake a rotted file for the file it was writing"
    );
    assert!(path.exists(), "and the fresh artifact is back under the same key");
    assert!(
        std::fs::metadata(&path)
            .expect("the fresh file stats")
            .modified()
            .expect("a modification time")
            > rotted_at,
        "under the same name, and it is not the same bytes"
    );

    // ── AND THE SEAT SERVES AGAIN, out of the replacement, with the answers
    //    the eager boot gave. This is the half of T-3's claim that survives
    //    intact: a flipped bit costs an import and not a deployment.
    let before_healed = tier::observed();
    let healed = boot(&rig, plan.clone(), &dir.0, &prompt).expect("the repaired shell loads");
    let mut shell = healed.shell;
    assert!(
        shell.weights_from_cache(),
        "and it came off the artifact the prepare wrote"
    );
    let (rot_tokens, rot_rows, _) = decode(&mut shell, &healed.row, "the repaired load");
    drop(shell);
    let after_healed = tier::observed();
    eprintln!(
        "REPAIRED boot: first token at {:.1} s, left {after_healed:?}",
        healed.first.as_secs_f64()
    );
    assert_eq!(
        after_healed.corrupt, before_healed.corrupt,
        "and nothing was wrong with the replacement: {after_healed:?}"
    );
    assert_eq!(
        after_healed.restored,
        before_healed.restored + 1,
        "it is a restore like any other: {before_healed:?} -> {after_healed:?}"
    );
    assert_eq!(
        rot_tokens, golden,
        "the repaired load chose {rot_tokens:?} and the eager one chose {golden:?}"
    );
    identical(&golden_rows, &rot_rows, "the load that was repaired");

    // ── (6) LATE ROT: THE FILE GOES BAD UNDER A SEAT THAT VERIFIED IT.
    //
    //    **THE RENAME IS TIMED OFF THE SEAT'S OWN MAPPING**, which the kernel
    //    will state to anybody who asks: a deferred boot maps this path
    //    TWICE, once for the restore and once for the tier that will serve
    //    out of it (`weights::resident` argues why the second one exists),
    //    and the second mapping appearing in `/proc/self/maps` is the moment
    //    the seat is seated. Everything the property needs sits after it —
    //    the boot's verify reads the MAPPED INODE and a rename cannot touch
    //    it, so the seat still serves bytes that were checksummed, while the
    //    fill re-opens the PATH and gets the rotten file. What separates the
    //    two is the whole of `fill_tiers`, seconds of pump and digest, so the
    //    syscall is not racing anything.
    //
    //    **AND IT USED TO BE TIMED OFF THE `deferred` COUNTER**, into the
    //    beginning of a `cudaHostAlloc` that took seconds to page-lock the
    //    image before the fill read a byte. §L-3 deleted that window on
    //    purpose — the fill now maps, reads and locks LAST — so the trigger
    //    moved to the one that is still seconds wide. Nothing here reaches
    //    into the engine to arrange it either way.
    let before_late = tier::observed();
    let seated = mappings_of(&path);
    let watching = std::thread::spawn({
        let path = path.clone();
        move || {
            let deadline = Instant::now() + Duration::from_secs(300);
            while mappings_of(&path) < seated + 2 && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
            replace_with_a_rotten_copy(&path)
        }
    });
    let late = boot(&rig, plan, &dir.0, &prompt).expect("the deferred shell loads again");
    let mut shell = late.shell;
    let at = watching.join().expect("the watcher renames the file");
    let opened = tier::observed();
    assert_eq!(
        opened.deferred,
        before_late.deferred + 1,
        "the fourth boot takes the deferred seat too: {opened:?}"
    );
    assert_eq!(
        opened.promoted, before_late.promoted,
        "the window has to still be open for this claim to be about anything"
    );
    eprintln!(
        "replaced {path:?} with a same-length copy whose payload is zeros, from \
         byte {at} on — under a seat that had already mapped the good inode, and \
         before the fill re-opened the name"
    );

    // **AND THE SEAT GOES ON SERVING.** Its addresses point into the mapping
    // of the inode the rename unlinked, which the tier holds open for its
    // whole life — the property a seat serving out of a mapping stands on.
    let (late_tokens, late_rows, _) = decode(&mut shell, &late.row, "the late-rot window");
    assert_eq!(
        late_tokens, golden,
        "a file rotting under a verified mapping changed the answer: {late_tokens:?}"
    );
    identical(&golden_rows, &late_rows, "the late-rot window");

    assert!(
        !shell
            .settle_tier_refill()
            .expect("a refused fill is not an error"),
        "the background fill read a rotted file and must never install it"
    );
    let after_late = tier::observed();
    assert_eq!(
        after_late.promoted, before_late.promoted,
        "a deferred seat whose fill was refused never promotes: {after_late:?}"
    );
    assert_eq!(
        after_late.corrupt,
        before_late.corrupt + 1,
        "and the refusal is counted where an operator can see it: \
         {before_late:?} -> {after_late:?}"
    );
    // **AND THE FILE THAT LIED IS STILL THERE** (§M.4). The fill named it and
    // stopped; it is the model on this machine and the thread that could not
    // read it is not the thread that can rebuild it. The next boot refuses the
    // same bytes at the same door — and since §M-3 that refusal is where it
    // ends, exactly as (5) has just shown: an import replaces the file, and a
    // boot after it serves.
    assert!(
        path.exists(),
        "the background fill removed a file it only failed to read: {path:?}"
    );
    assert!(
        tier::Artifact::open(&path)
            .expect("the rotten copy still parses — it kept the real header")
            .verify()
            .is_err(),
        "and it is still the rotten copy, which is why the next boot goes cold"
    );

    // And it is still serving, after the refusal, out of the same mapping.
    let seed = reopen(&mut shell, &prompt, "after the refusal");
    let (still, still_rows, still_ms) = decode(&mut shell, &seed, "after the refusal");
    assert_eq!(still, golden, "the seat stopped answering after its fill was refused");
    identical(&golden_rows, &still_rows, "after the refusal");
    drop(shell);
    eprintln!(
        "LATE-ROT boot: first token at {:.1} s, still serving at {still_ms:.2} ms/step; \
         {:?}",
        late.first.as_secs_f64(),
        tier::observed(),
    );
}
