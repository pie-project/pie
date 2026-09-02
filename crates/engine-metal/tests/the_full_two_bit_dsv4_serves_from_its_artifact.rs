//! **FIRST LIGHT FOR THE FULL TWO-BIT DeepSeek-V4-Flash**, off its serving
//! artifact and nothing else.
//!
//! `dsv4-flash-full-u4g64-u2g64-kv-bf16` — 43 layers, 256 routed experts of which six
//! fire, over the 129 280-token vocabulary — is 89.9 GiB on a box with 32 GiB
//! of unified memory. Every other file in this suite that fires this family
//! fires the `mini-l5-e16` miniature, which stands in for exactly this
//! checkpoint and is five layers and sixteen experts of it. This is the one
//! that reads the real thing.
//!
//! # The artifact, and why there is no snapshot arm
//!
//! ```text
//! cargo build --release -p pie --features metal
//! ./target/release/pie model import \
//!     ~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-2bit-DQ/\
//! snapshots/<hash> --force --consume-source \
//!     --out ~/.pie/models/dsv4-flash-full-u4g64-u2g64-kv-bf16/dsv4-flash-full-u4g64-u2g64-kv-bf16.zt
//! ```
//!
//! `--consume-source` is not a convenience here, it is the only way the import
//! fits: the source is 89.9 GiB, the artifact is 89.9 GiB, and this pool had 38
//! GiB free. The flag releases each source range as the decode reads it, so the
//! two never stand at full size together — free space held flat at 38 GiB for
//! the whole 4m14s of it. The other side of that is that THE SNAPSHOT IS GONE:
//! its nineteen weight shards were consumed, and the cold arms
//! `a_warm_load_streams_the_bands_it_maps` fires beside its warm ones are not
//! available for this model and never will be on this box. What is left beside
//! the artifact is the tokenizer and the config, which the import does not eat.
//!
//! So the claim here is narrower than that file's four-road identity, and it is
//! the claim that was open: **this artifact loads on this box and answers.** The
//! four-road argument is the miniature's, made on the same code path, and it is
//! about the MECHANISM. What a full checkpoint adds is that the mechanism holds
//! at 43 layers and 256 experts — where the residency plan actually has to hold
//! something back, because full residency is not on the table at any budget.
//!
//! # WHAT THIS FILE FOUND, AND WHAT LIFTED IT: `maxBufferLength`
//!
//! For one commit it did not run. The warm arm wrapped the whole mapped
//! artifact in ONE Metal buffer — `newBufferWithBytesNoCopy` — and one buffer
//! is bounded by `maxBufferLength`, which on this box is 20 100 448 256 bytes
//! (18.72 GiB). The artifact is 96 566 753 286 (89.93 GiB). So `weights::warm`
//! failed with a `Fault::Ceiling` and the load DOWNGRADED TO COLD rather than
//! refusing.
//!
//! That downgrade was the dangerous part, and it was measured by letting it
//! run: the cold arm reads the whole artifact into a host store and stages its
//! bands beside it, and the free pool fell 38 → 21 GiB in forty seconds with
//! swap rising 705 → 1003 MB, on a box with 32 GiB of memory. It was stopped
//! above its own floor. Both halves of that are answered now and neither is
//! answered here: `mapping::cut` binds the mapping as SEVERAL windows so the
//! ceiling bounds a reservation instead of the model, and `HostSource::open`
//! admits the cold road's staging against the disk so the fall is a refusal
//! with numbers rather than a pool that empties.
//!
//! **THE CEILING IS STILL ASKED BEFORE THE LOAD**, off the same device the
//! load will bind, and it is now a fact this file PRINTS rather than a wall it
//! stops at: an artifact of `n` ceilings is expected to bind as at least `n`
//! windows, and `Shell::weight_windows` is where that is read back. This one
//! takes 44 — cut around the 4.02 GiB of resident planes and NOT around the
//! 85.88 GiB of routed bands they are interleaved with, because a reservation
//! wires whole (`what_a_mapped_window_wires`) and windows tiling this file
//! wire the file. The first cut that tiled it did exactly that and its first
//! window alone was 18.72 GiB: free to 0.06 GiB, swap past 14 GiB, killed by
//! hand. This is the first artifact on this box big enough to need more than
//! one window — the next largest in the store is 17.2 GiB, and its own gate
//! asserts it still takes exactly one.
//!
//! # WHAT THIS FILE FOUND SECOND: A SEAT IS A COUNT, AND SIXTEEN WAS NOT IT
//!
//! With the ceiling lifted the load ran and the FIRE refused, by name and with
//! its numbers:
//!
//! ```text
//! one segment of this fire routes to more than 16 distinct experts of
//! `layer.0.experts_gate`, and the wired slab seats 16: every seat is pinned
//! by a matmul this same segment will run, so no seat can be reused.
//! ```
//!
//! Which is right, and it is the accounting doing its job. A five-token
//! prefill at `top_k` six can route to thirty distinct experts in one segment,
//! and every one of them has to be seated at once. So the seat count is the
//! lever and its price is device memory, measured here and printed by the run:
//!
//! ```text
//!    16 seats ->  9.46 GiB device demand      40 seats -> 17.52 GiB
//!    24 seats -> 12.15 GiB                    48 seats -> 20.20 GiB
//!    32 seats -> 14.83 GiB                    64 seats -> 25.57 GiB
//! ```
//!
//! Forty is what this file fires at (`PIE_U2_FULL_SEATS` moves it): thirty
//! distinct experts with room, at 17.52 GiB on a 32 GiB box.
//!
//! # FIRST LIGHT, MEASURED
//!
//! ```text
//! run-1  load 27 417 ms  prefill 23 209 ms (0.2 tok/s)  decode 25 734 ms (0.6 tok/s)
//! run-2  load 45 232 ms  prefill 24 601 ms (0.2 tok/s)  decode 27 618 ms (0.6 tok/s)
//!        warm=true  windows=44  source=artifact  slabs=43  motion=(8019, 731)
//!        wired: boot +0.023 GiB, first fire +15.455 GiB
//!        swap: 0.945 GiB idle -> 0.820 GiB after both runs — it went DOWN
//! ```
//!
//! Both runs answered the same seventeen tokens. The boot wiring is +0.023 GiB
//! because a warm load binds and reads nothing; the +15.5 GiB arrives on the
//! first FIRE, which is the store's seats and the resident planes the kernels
//! touch, and it is the number the 17.52 GiB plan predicted.
//!
//! **THE TOKENS ARE NOT COHERENT TEXT**, and this file does not claim they
//! are. It asserts finite, spread, and identical across two independent loads
//! — the properties a wrong band offset or a wrong window view would break —
//! and the four-road identity argument that would settle fidelity belongs to
//! the miniature, whose snapshot still exists. This one's does not.
//!
//! # What it asserts
//!
//! * the load takes the WARM arm (`Shell::weights_warm`) — the artifact is
//!   mapped, not read into a store the size of the model, which on this box is
//!   the difference between loading and not — and it is mapped as MORE THAN
//!   ONE window, because 89.93 GiB does not fit one `MTLBuffer`;
//! * the plan STREAMS, and its seats come from the artifact's own mapping
//!   (`expert_source_kind() == "artifact"`), so nothing staged a second copy of
//!   the expert bank onto a disk with 38 GiB free;
//! * every logit is finite and the row is not flat — a wrong band offset seats
//!   another expert's bytes and answers finite, deterministic nonsense, so
//!   finiteness alone is not the check; the spread is;
//! * two independent loads of the same artifact answer the SAME tokens.
//!
//! Reported and not asserted: load wall clock, prefill and decode rates, and
//! the `vm_stat` wired and swap readings. Those are facts about this box on
//! this day, and a threshold on them would be a gate on the machine rather than
//! on the code.
//!
//! ```text
//! cargo test -p engine-metal --release \
//!     --test the_full_two_bit_dsv4_serves_from_its_artifact -- --nocapture --test-threads 1
//! ```

#![cfg(target_vendor = "apple")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use engine_metal::experts::{Attachments, Plan};
use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};
use model_ir::Trace;

const SKU: &str = "dsv4-flash-full-mtp-u4g64-u2g64-mxfp4-kv-bf16";

const REPO: &str = "models--mlx-community--DeepSeek-V4-Flash-2bit-DQ";

/// How many of the 256 routed experts one slab seats.
///
/// The number has to hold every DISTINCT expert one segment routes to, all at
/// once: a seat is pinned by a matmul the same segment will run, so no seat is
/// reusable within a segment. With `top_k` of six that is six per token in the
/// worst case, and a prefill of `n` tokens in one segment can want `6n`.
/// Sixteen was this file's first guess and the accounting refused it BY NAME on
/// this five-token prompt — the sentence is in the header, and it is the right
/// answer rather than a failure. It is read from `PIE_U2_FULL_SEATS` so the
/// trade against `device_weight_budget` can be moved without a rebuild.
fn seats() -> u32 {
    std::env::var("PIE_U2_FULL_SEATS")
        .ok()
        .and_then(|it| it.parse().ok())
        .unwrap_or(40)
}

/// A SHORT prompt, deliberately. A wide prefill routes more distinct experts
/// per segment than the slab can hold at once, and the accounting refuses
/// rather than seating a fire it cannot serve — which is the right answer and
/// not a failure of this file; it fired here at sixteen seats and the header
/// carries the sentence. Five tokens against forty seats has room. See
/// `a_wide_prefill_says_what_it_would_need` in the streamed suite for that
/// refusal read from its own side.
const PROMPT: &str = "The capital of France is";

/// Decode fires after the prefill.
const STEPS: usize = 16;

/// Global `Pages wired down`, in bytes, off `vm_stat` — a mapped page wires
/// against the KERNEL rather than against this process, so a task-level counter
/// does not see it.
fn wired() -> Option<u64> {
    let said = std::process::Command::new("vm_stat").output().ok()?;
    let text = String::from_utf8(said.stdout).ok()?;
    let mut page = 4096u64;
    let mut pages = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Mach Virtual Memory Statistics: (page size of ") {
            if let Some(n) = rest.split_whitespace().next() {
                page = n.parse().unwrap_or(page);
            }
        }
        if let Some(rest) = line.strip_prefix("Pages wired down:") {
            pages = rest.trim().trim_end_matches('.').parse::<u64>().ok();
        }
    }
    Some(pages? * page)
}

/// Swap in use, in bytes, off `sysctl vm.swapusage`. Printed beside the wired
/// numbers because on a box this size the interesting failure is not "it did
/// not load" but "it loaded by paging the machine out from under itself".
fn swap() -> Option<u64> {
    let said = std::process::Command::new("sysctl")
        .arg("-n")
        .arg("vm.swapusage")
        .output()
        .ok()?;
    let text = String::from_utf8(said.stdout).ok()?;
    let used = text.split("used = ").nth(1)?;
    let megabytes: f64 = used.trim_start().split('M').next()?.parse().ok()?;
    Some((megabytes * (1u64 << 20) as f64) as u64)
}

fn gib(bytes: Option<u64>) -> String {
    match bytes {
        Some(bytes) => format!("{:.3} GiB", bytes as f64 / (1u64 << 30) as f64),
        None => "unavailable".to_string(),
    }
}

fn delta(before: Option<u64>, after: Option<u64>) -> String {
    match (before, after) {
        (Some(before), Some(after)) => format!(
            "{:+.3} GiB",
            (after as f64 - before as f64) / (1u64 << 30) as f64
        ),
        _ => "unavailable".to_string(),
    }
}

/// The serving artifact, found by its own STAMP and not by its name — the same
/// three-answer door `weights::serves_this_deployment` asks, so a candidate
/// this finds is one that gate accepts.
fn artifact() -> Option<PathBuf> {
    let stamped = |path: &Path| {
        let stamp = checkpoint::file::serve::stamp_of(path).ok().flatten()?;
        (stamp.backend == "metal" && stamp.sku == SKU).then(|| path.to_path_buf())
    };
    if let Ok(stated) = std::env::var("PIE_METAL_FULL_ARTIFACT") {
        return stamped(Path::new(&stated));
    }
    let homes = [
        format!("{}/models", std::env::var("PIE_HOME").unwrap_or_default()),
        format!("{}/.pie/models", std::env::var("HOME").unwrap_or_default()),
        "/Users/ingim/.pie/models".to_string(),
    ];
    homes.iter().find_map(|home| {
        let mut found = walk(Path::new(home));
        found.sort();
        found.iter().find_map(|path| stamped(path))
    })
}

/// Every `.zt` under `at`, one directory deep — the model store nests an
/// artifact under a directory named for its row.
fn walk(at: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(at).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|it| it == "zt") {
            out.push(path);
        } else if path.is_dir() {
            out.extend(
                std::fs::read_dir(&path)
                    .into_iter()
                    .flatten()
                    .flatten()
                    .map(|entry| entry.path())
                    .filter(|path| path.extension().is_some_and(|it| it == "zt")),
            );
        }
    }
    out
}

/// The tokenizer, from whatever the import LEFT behind.
///
/// `--consume-source` eats the weight shards and keeps the tokenizer and the
/// config, so the snapshot directory is still the place to ask — it is simply
/// no longer a checkpoint. Named explicitly by `PIE_U2_FULL_SNAPSHOT` where the
/// cache is somewhere else.
fn tokenizer_file() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_U2_FULL_SNAPSHOT") {
        let path = PathBuf::from(stated).join("tokenizer.json");
        return path.is_file().then_some(path);
    }
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
    homes.iter().find_map(|home| {
        let snapshots = Path::new(home)
            .join(".cache/huggingface/hub")
            .join(REPO)
            .join("snapshots");
        let mut found: Vec<PathBuf> = std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path().join("tokenizer.json")))
            .filter(|path| path.is_file())
            .collect();
        found.sort();
        // The full checkpoint's snapshot, not the miniature's: they share a
        // repo directory and the miniature's vocabulary is the same file, but
        // asking for the one whose directory is a hash keeps the two apart if
        // they ever diverge.
        found
            .iter()
            .find(|path| {
                path.parent()
                    .and_then(Path::file_name)
                    .and_then(|it| it.to_str())
                    .is_some_and(|name| name.len() == 40 && name.chars().all(|c| c.is_ascii_hexdigit()))
            })
            .or_else(|| found.first())
            .cloned()
    })
}

/// One artifact, read for a load: the trace, the contract fitted against it,
/// and the load plan's pairing.
struct Read {
    trace: Trace,
    contract: checkpoint::contract::ModelContract,
    planes: Attachments,
}

fn read(artifact: &Path) -> Read {
    let trace = (models::sku(SKU).expect("the catalog ships the full 2-bit row").trace)(Platform::Metal);
    let source = ztensor_compat::index(artifact).expect("the artifact opens");
    let contract = checkpoint_dsl::own_contract(&source, &trace.params, 1, Platform::Metal)
        .unwrap_or_else(|why| panic!("the artifact holds every plane of {SKU}: {why}"));
    drop(source);
    let planes = engine_metal::weights::attachments(&trace, &contract, artifact)
        .expect("the load plan pairs this artifact's quantized banks");
    Read {
        trace,
        contract,
        planes,
    }
}

/// **THE STREAMED PLAN, SIZED BY A SEAT COUNT** — the streamed suite's
/// bisection and its argument: a fire's floor is a COUNT (`top_k` seats pinned
/// at once) while a budget is bytes, so the budget that seats a stated count is
/// searched for and the plan is asserted to seat exactly it.
fn seating(read: &Read, want: u32) -> Plan {
    let full = Plan::of(&read.trace, &read.planes, None)
        .expect("an uncapped plan is full residency")
        .device_demand();
    let plan_at = |budget: u64| Plan::of(&read.trace, &read.planes, Some(budget));
    let (mut lo, mut hi) = (0u64, full - 1);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match plan_at(mid) {
            Ok(plan) if plan.slots() >= want => hi = mid,
            _ => lo = mid + 1,
        }
    }
    let plan = plan_at(lo).unwrap_or_else(|why| {
        panic!("no budget under {full} seats {want} of this artifact's experts: {why}")
    });
    assert!(plan.slots() >= want, "the bisected budget seats what was asked");
    assert!(plan.streams(), "a plan that holds nothing back is not a slab");
    plan
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

/// Finite AND spread. A wrong band offset seats another expert's bytes and the
/// model answers finite, deterministic nonsense, so finiteness is a necessary
/// check and not a sufficient one; a row whose whole range is a rounding error
/// is a row nothing chose from.
fn finite_and_spread(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits at all");
    let bad = logits.iter().position(|value| !value.is_finite());
    assert!(
        bad.is_none(),
        "{what} logit {} is {}, and a single NaN means the whole row is noise",
        bad.unwrap_or(0),
        logits[bad.unwrap_or(0)],
    );
    let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
    for value in logits {
        low = low.min(*value);
        high = high.max(*value);
    }
    assert!(
        high - low > 1.0,
        "{what} logits span only {} over {} classes, which is a row nothing chose \
         from rather than an answer",
        high - low,
        logits.len(),
    );
}

fn word(len: u32) -> u64 {
    models::deepseek_v4::forward::Facts::of(&Request::new(len, false)).word()
}

/// One run: load, prefill, decode, drop. **ONE MODEL RESIDENT AT A TIME** —
/// this is an 89.9 GiB artifact on a 32 GiB box and a second live shell is not
/// a measurement, it is a swap storm.
struct Run {
    tokens: Vec<u32>,
    load_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    warm: bool,
    windows: usize,
    kind: Option<&'static str>,
    source: Option<(u64, u64)>,
    motion: (u64, u64),
    slabs: usize,
    boot_wired: String,
    fire_wired: String,
    swap_after: Option<u64>,
}

fn run(what: &str, artifact: &Path, residency: Plan, prompt: &[u32]) -> Run {
    let read = read(artifact);
    let idle = wired();
    let at = Instant::now();
    let mut shell = Shell::load(Boot {
        trace: read.trace.clone(),
        contract: &read.contract,
        checkpoint: artifact,
        budget: Budget::new(4, 512),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        pages: (4) * (512) / (16),
        // F1: a streamed load's segment cuts block deeper runahead, so the
        // depth the shallowest arm can reach is the depth every number here is
        // measured at.
        runahead: engine::runahead::Runahead::F1,
        residency,
    })
    .unwrap_or_else(|why| panic!("the {what} shell loads: {why}"));
    let load_ms = at.elapsed().as_secs_f64() * 1000.0;
    let booted = wired();

    shell.open(0).expect("the slot opens");
    let at = Instant::now();
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .unwrap_or_else(|why| panic!("the {what} prefill fires: {why}"));
    let prefill_ms = at.elapsed().as_secs_f64() * 1000.0;
    let fired = wired();
    finite_and_spread(&prefill[0], what);
    let mut tokens = vec![argmax(&prefill[0])];

    let at = Instant::now();
    for step in 0..STEPS {
        let fed = [*tokens.last().expect("a step feeds the last token back")];
        let decode = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &fed,
            }])
            .unwrap_or_else(|why| panic!("{what} decode step {step} fires: {why}"));
        finite_and_spread(&decode[0], what);
        tokens.push(argmax(&decode[0]));
    }
    let decode_ms = at.elapsed().as_secs_f64() * 1000.0;

    Run {
        tokens,
        load_ms,
        prefill_ms,
        decode_ms,
        warm: shell.weights_warm(),
        windows: shell.weight_windows(),
        kind: shell.expert_source_kind(),
        source: shell.expert_source(),
        motion: shell.expert_motion(),
        slabs: shell.expert_residency().len(),
        boot_wired: delta(idle, booted),
        fire_wired: delta(booted, fired),
        swap_after: swap(),
    }
}

/// **THE FULL CHECKPOINT LOADS ON THIS BOX AND ANSWERS.**
#[test]
fn the_full_dsv4_artifact_loads_warm_streams_its_experts_and_answers_twice_the_same() {
    if !engine_metal::device::present() {
        eprintln!("skipping: this machine publishes no Metal device");
        return;
    }
    let Some(artifact) = artifact() else {
        eprintln!(
            "skipping: no `metal`-stamped {SKU} artifact found — import one with an \
             ENGINE-METAL-FEATURE binary (see this file's header; `-p pie`, never a \
             workspace build) and name it in PIE_METAL_FULL_ARTIFACT"
        );
        return;
    };
    let Some(vocabulary) = tokenizer_file() else {
        eprintln!(
            "skipping: no tokenizer.json beside a {REPO} snapshot — `--consume-source` \
             keeps it, so this is a cache that was cleared rather than consumed; name \
             the directory in PIE_U2_FULL_SNAPSHOT"
        );
        return;
    };
    let bytes = std::fs::metadata(&artifact).map(|it| it.len()).unwrap_or(0);
    eprintln!(
        "artifact {artifact:?} ({:.1} GiB)\ntokenizer {vocabulary:?}",
        bytes as f64 / (1u64 << 30) as f64
    );
    eprintln!(
        "idle: wired {} swap {}",
        gib(wired()),
        gib(swap())
    );

    let tokenizer =
        tokenizer::Tokenizer::from_file(&vocabulary).expect("the checkpoint's tokenizer loads");
    let prompt = tokenizer.encode(PROMPT);
    assert!(!prompt.is_empty(), "the prompt encodes to at least one token");

    // **THE CEILING IS ASKED BEFORE THE LOAD, NOT DISCOVERED INSIDE IT.**
    //
    // It is no longer a wall — `mapping::cut` binds the artifact as several
    // windows and `maxBufferLength` bounds one of them — but it is still the
    // number that decides the shape of this load, so it is read here, from
    // the same device the load will bind, and printed beside what it implies.
    // **WHAT A SEAT COSTS, BEFORE ONE IS TAKEN.** The trade this row is
    // actually bound by: a segment's floor is a COUNT of distinct experts and
    // the budget is BYTES, so the operator's lever is the seat count and its
    // price is device memory. Printed as a curve rather than asserted,
    // because the affordable point is a fact about the box.
    let sized = read(&artifact);
    for want in [16u32, 24, 32, 40, 48, 64] {
        let plan = seating(&sized, want);
        eprintln!(
            "  {want:>3} seats -> {} slots, {:.2} GiB device demand",
            plan.slots(),
            plan.device_demand() as f64 / (1u64 << 30) as f64,
        );
    }
    let want = seats();
    let plan = seating(&sized, want);
    eprintln!(
        "firing with {} seats ({:.2} GiB device demand); a segment of {} token(s) can \
         route to at most {} distinct experts",
        plan.slots(),
        plan.device_demand() as f64 / (1u64 << 30) as f64,
        prompt.len(),
        prompt.len() * 6,
    );
    eprintln!(
        "the streamed plan seats {} experts across {} groups, {} bands; device demand {:.2} GiB",
        plan.slots(),
        plan.groups().len(),
        plan.bands().len(),
        plan.device_demand() as f64 / (1u64 << 30) as f64,
    );

    let ceiling = engine_metal::device::Context::bind()
        .expect("the device binds")
        .max_buffer();
    let least = bytes.div_ceil(ceiling);
    eprintln!(
        "maxBufferLength {ceiling} ({:.2} GiB) against an artifact of {bytes} ({:.2} \
         GiB): one `MTLBuffer` holds {:.1}% of this row, so the mapping binds as at \
         least {least} window(s) cut at the manifest's own blob boundaries",
        ceiling as f64 / (1u64 << 30) as f64,
        bytes as f64 / (1u64 << 30) as f64,
        100.0 * ceiling as f64 / bytes as f64,
    );

    let first = run("run-1", &artifact, plan.clone(), &prompt);
    let second = run("run-2", &artifact, plan.clone(), &prompt);

    for (what, at) in [("run-1", &first), ("run-2", &second)] {
        eprintln!(
            "{what}  load {:>8.0} ms  prefill {:>7.1} ms ({:.1} tok/s)  decode {:>7.1} ms \
             ({:.1} tok/s)\n       warm={} windows={} source={} {:?} motion={:?} \
             slabs={}  wired boot {} first fire {}  swap {}",
            at.load_ms,
            at.prefill_ms,
            prompt.len() as f64 / (at.prefill_ms / 1000.0),
            at.decode_ms,
            STEPS as f64 / (at.decode_ms / 1000.0),
            at.warm,
            at.windows,
            at.kind.unwrap_or("none"),
            at.source,
            at.motion,
            at.slabs,
            at.boot_wired,
            at.fire_wired,
            gib(at.swap_after),
        );
    }
    eprintln!("run-1 tokens {:?}", first.tokens);
    eprintln!("run-1 text   {:?}", tokenizer.decode(&first.tokens, false));

    // (1) THE WARM ARM. The artifact is mapped, not read into a store the size
    //     of the model — on this box that is the difference between loading and
    //     being killed.
    assert!(
        first.warm,
        "the {SKU} artifact did not take the warm arm, so this load read 89.9 GiB into \
         a host store on a 32 GiB box rather than mapping it"
    );

    // (1b) AND IT IS BOUND IN WINDOWS. 89.93 GiB does not fit one `MTLBuffer`
    //      on any Apple device that ships, so a warm load of it that reported
    //      ONE window would be reporting a ceiling that had stopped applying
    //      rather than a mapping that had been cut.
    assert!(
        first.windows >= least as usize,
        "the {SKU} artifact bound as {} window(s) against a {ceiling}-byte          `maxBufferLength` and {bytes} bytes of file, which needs at least {least}",
        first.windows,
    );
    assert_eq!(
        first.windows, second.windows,
        "two loads of one artifact cut it into different numbers of windows"
    );

    // (2) AND IT STREAMS OUT OF THAT SAME MAPPING. `"landed"` would mean the
    //     load wrote a staging file of expert bands — a second copy of ~80 GiB,
    //     on a pool with 38 GiB free.
    assert!(
        first.slabs > 0,
        "a load with no slab is a full-residency load wearing the word, and full \
         residency of this bank does not fit this box"
    );
    assert_eq!(
        first.kind,
        Some("artifact"),
        "the seats came from {:?} rather than from the artifact's own mapping, which \
         means this load staged a second copy of the expert bank",
        first.kind,
    );

    // (3) TWO LOADS, ONE ANSWER. Not a fidelity claim — see the streamed
    //     suite's note on why a step count is a shape check — but the check
    //     that catches a slab whose seating depends on what the clock happened
    //     to be holding.
    assert_eq!(
        first.tokens, second.tokens,
        "two loads of one artifact answered differently, so something in the seat \
         bookkeeping is reading state a load does not own"
    );
    assert_eq!(
        first.tokens.len(),
        STEPS + 1,
        "the prefill's token and one per decode step"
    );
}
