//! **THE WARM-STREAMED ARM** (§M-6) — a serving artifact whose residency plan
//! streams its routed bands loads by MAPPING, seats its experts out of that
//! same mapping, and answers what all three of the other arms answer.
//!
//! This file is the composition of two halves that were each gated alone and
//! could not be taken together until now:
//!
//! * **`a_warm_load_is_the_artifact_mapped`** (§M-5) proved the warm bind: the
//!   artifact is mapped once and every plane is a `(buffer, offset)` view onto
//!   its own pages, with no store the size of the model and no `memcpy` into
//!   frames the GPU was about to wire anyway. It refused a STREAMED plan
//!   outright, on the fits-in-memory rule.
//! * **`two_bit_experts_stream`** proved the streamed path over this same
//!   artifact's snapshot: fifteen 2-bit routed banks, two router kinds, seated
//!   a fraction at a time out of a file-backed host mapping, bit-identical to
//!   full residency.
//!
//! The refusal was never the rule, only the shape of it. The fits-in-memory
//! rule is about what the DEVICE READS — a GPU-touched shared page wires and
//! the pager takes none of it back — and a streamed band is not read by the
//! device off the mapping at all. It gets `slots` seats in the writable
//! reservation exactly as the cold path gives it, and the tier copies experts
//! into them with the CPU. So the two compose exactly, and what the warm arm
//! removes from a streamed load is the thing that was always silly about it:
//! **staging a second file.** The cold path lands the bands into a `MAP_SHARED`
//! temporary and copies seats out of that; the bytes were already on this
//! machine, in landed form, in the artifact the load is holding open.
//!
//! # The four arms, and why all four
//!
//! ```text
//!               full residency          streamed
//!   cold      the snapshot, read      the snapshot, read + staged
//!   warm      the artifact, mapped    the artifact, mapped + seated
//! ```
//!
//! Token identity across all four is the claim, and it is not three claims
//! plus one: every arm reaches the same weights by a different road, and the
//! two roads this file adds are the two that could disagree QUIETLY. A wrong
//! band offset does not crash — it seats another expert's bytes, and the model
//! answers finite, deterministic nonsense — which is exactly the failure the
//! per-band source table exists to make impossible (`experts::Source`).
//!
//! # What separates the arms as OBSERVABLES
//!
//! * `Shell::weights_warm` — did this load MAP its checkpoint, or read it.
//! * `Shell::expert_source_kind` — `"artifact"` when the seat copies read the
//!   mapped serving artifact, `"landed"` when they read a staging file this
//!   load wrote, `None` when nothing streams.
//! * `Shell::expert_source` — the pair, whose second number reads OPPOSITE
//!   ways under the two arms: `0` links is the cold arm's unlinked temporary,
//!   and at least one is the artifact the operator named. Asserting zero
//!   without asking the kind is asserting the load was cold, which is why this
//!   file asks the kind first.
//!
//! # The artifact
//!
//! ```text
//! cargo build --release -p pie --features engine-metal
//! ./target/release/pie model import \
//!     ~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-2bit-DQ/\
//! snapshots/mini-l5-e16 --out /tmp/warmstream/dsv4-mini-metal.zt
//! ```
//!
//! `-p pie` and not a workspace build: a workspace build of this feature fails
//! on another crate and still exits zero through a pipe, which leaves a STALE
//! binary that stamps the wrong backend. `this_box()` is Metal only in a
//! binary built with the feature, and the stamp is what
//! `weights::serves_this_deployment` compares.
//!
//! `PIE_METAL_MOE_ARTIFACT` names it; absent that, any `.zt` under
//! `/tmp/warmstream`, `$PIE_HOME/models` or `~/.pie/models` whose stamp reads
//! back for this backend and this SKU.
//!
//! ```text
//! cargo test -p engine-metal --release \
//!     --test a_warm_load_streams_the_bands_it_maps -- --nocapture --test-threads 1
//! ```

#![cfg(target_vendor = "apple")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use engine_metal::experts::{Attachments, Plan};
use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};
use model_ir::Trace;

/// The row this file serves — `two_bit_experts_stream`'s dsv4 row, because it
/// is the one whose routed bands are a REAL fraction of the table (40%, so a
/// slab bounds something) and whose two router kinds exercise both segment
/// triggers.
const SKU: &str = "dsv4-flash-mlxu2-kv-bf16";

const REPO: &str = "models--mlx-community--DeepSeek-V4-Flash-2bit-DQ";

/// How many experts of sixteen the slab this file fires seats — the sibling's
/// number, and it is bounded below by this row's `top_k` of 6: every distinct
/// expert one SEGMENT routes to must be seated at once.
const SEATS: u32 = 8;

/// How many greedy decode fires follow the one-token prefill. Long enough that
/// the routing wanders off the identity prefix the slab opened at and the
/// clock has to evict, which is what makes the token comparison a claim about
/// SEAT COPIES rather than about the prefix.
const STEPS: usize = 8;

/// One token, for `two_bit_experts_stream`'s reason: the first light's
/// eight-token prompt reaches all sixteen experts at the hash gate, and no
/// streaming budget serves that fire. The parity is therefore fired one token
/// at a time on every arm, so that the four are comparable.
const PROMPT: &str = "The";

/// Global `Pages wired down`, in bytes, off `vm_stat` — a mapped page wires
/// against the KERNEL rather than against this process, so a task-level
/// counter does not see it.
fn wired() -> Option<u64> {
    let said = std::process::Command::new("vm_stat").output().ok()?;
    let text = String::from_utf8(said.stdout).ok()?;
    let mut page = 4096u64;
    let mut pages = None;
    for line in text.lines() {
        if let Some(rest) =
            line.strip_prefix("Mach Virtual Memory Statistics: (page size of ")
        {
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

/// A signed GiB delta between two `vm_stat` readings, or `unavailable`.
fn delta(before: Option<u64>, after: Option<u64>) -> String {
    match (before, after) {
        (Some(before), Some(after)) => format!(
            "{:+.3} GiB",
            (after as f64 - before as f64) / (1u64 << 30) as f64
        ),
        _ => "unavailable".to_string(),
    }
}

/// The snapshot: the checkpoint AND the tokenizer beside it.
fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_U2_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let usable = |path: &Path| path.join("tokenizer.json").exists() && !shards(path).is_empty();
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
            .filter_map(|entry| Some(entry.ok()?.path()))
            .filter(|path| usable(path))
            .collect();
        found.sort();
        found.into_iter().next()
    })
}

/// Every shard of a snapshot, sorted.
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

/// **THE SERVING ARTIFACT, FOUND BY ITS OWN STAMP AND NOT BY ITS NAME** —
/// `a_warm_load_is_the_artifact_mapped`'s finder, over this file's SKU. A file
/// named for the model and stamped for another shell is not one this box can
/// serve, and the difference is a field rather than a spelling.
fn artifact() -> Option<PathBuf> {
    let stamped = |path: &Path| {
        let stamp = checkpoint::file::serve::stamp_of(path).ok().flatten()?;
        (stamp.backend == "metal" && stamp.sku == SKU).then(|| path.to_path_buf())
    };
    if let Ok(stated) = std::env::var("PIE_METAL_MOE_ARTIFACT") {
        return stamped(Path::new(&stated));
    }
    let homes = [
        "/tmp/warmstream".to_string(),
        format!(
            "{}/models",
            std::env::var("PIE_HOME").unwrap_or_default()
        ),
        format!("{}/.pie/models", std::env::var("HOME").unwrap_or_default()),
        "/Users/ingim/.pie/models".to_string(),
    ];
    homes.iter().find_map(|home| {
        let mut found: Vec<PathBuf> = walk(Path::new(home));
        found.sort();
        found.iter().find_map(|path| stamped(path))
    })
}

/// Every `.zt` under `at`, one directory deep — the model store nests an
/// artifact under a directory named for its repo, and `/tmp/warmstream` does
/// not.
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
    let bad = logits.iter().position(|value| !value.is_finite());
    assert!(
        bad.is_none(),
        "{what} logit {} is {}, and a single NaN means the whole row is noise",
        bad.unwrap_or(0),
        logits[bad.unwrap_or(0)],
    );
}

fn word(len: u32) -> u64 {
    models::deepseek_v4::forward::Facts::of(&Request::new(len, false)).word()
}

/// One checkpoint, read for a load: the trace, the contract fitted against
/// THAT container, and the load plan's pairing.
///
/// Read per-checkpoint rather than once, because the two roads read two
/// different files: the snapshot publishes source planes the load transforms,
/// the artifact publishes the landed results. `models::import_of` answers a
/// contract for either — its arms try each family reading and the first that
/// BUILDS wins — and the trace they are both checked against is one trace.
struct Read {
    trace: Trace,
    contract: checkpoint::contract::ModelContract,
    planes: Attachments,
}

fn read(checkpoint: &Path) -> Read {
    let files = if checkpoint.is_dir() {
        shards(checkpoint)
    } else {
        vec![checkpoint.to_path_buf()]
    };
    assert!(!files.is_empty(), "{checkpoint:?} holds a tensor container");
    let trace = models::trace_of(SKU).expect("the catalog ships the 2-bit SKU")(Platform::Metal);
    let source = ztensor_compat::index_all(&files).expect("the container opens");
    let contract = models::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
        .unwrap_or_else(|why| panic!("the SKU's import contract fits {checkpoint:?}: {why}"));
    drop(source);
    let planes = engine_metal::weights::attachments(&trace, &contract, checkpoint)
        .expect("the load plan pairs this checkpoint's quantized banks");
    Read {
        trace,
        contract,
        planes,
    }
}

/// **THE STREAMED PLAN, SIZED BY A SEAT COUNT** — `two_bit_experts_stream`'s
/// bisection, and its argument: the floor a fire needs is a COUNT (`top_k`
/// seats pinned at once) while a budget is bytes, so the budget that seats a
/// stated count is searched for and the plan is asserted to seat exactly it.
fn seating(read: &Read, want: u32) -> Plan {
    let full = Plan::of(&read.trace, &read.planes, None)
        .expect("an uncapped 2-bit plan is full residency")
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
    assert_eq!(plan.slots(), want, "the bisected budget seats the stated count");
    assert!(plan.streams(), "a plan that holds nothing back is not a slab");
    plan
}

/// One arm: load, fire, drop — and everything the four are compared on.
struct Arm {
    what: &'static str,
    tokens: Vec<u32>,
    millis: f64,
    warm: bool,
    kind: Option<&'static str>,
    source: Option<(u64, u64)>,
    motion: (u64, u64),
    slabs: usize,
    boot_wired: String,
    fire_wired: String,
}

/// **ONE MODEL RESIDENT AT A TIME.** Each arm loads, fires, is read, and is
/// DROPPED before the next is asked for — this is a 1.6 GiB artifact on a
/// 32 GiB box, and a global `Pages wired down` delta means nothing measured
/// against a baseline this process is still holding a model against.
fn arm(what: &'static str, checkpoint: &Path, residency: Plan, prompt: &[u32]) -> Arm {
    let read = read(checkpoint);
    let idle = wired();
    let at = Instant::now();
    let mut shell = Shell::load(Boot {
        trace: read.trace.clone(),
        contract: &read.contract,
        checkpoint,
        tp_size: 1,
        precision: models::precision_of(SKU)
            .expect("the catalog states this row's precision")
            .to_string(),
        budget: Budget::new(4, 512),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        // F1 everywhere: the streamed arms collapse to it (the segment cuts
        // block), so all four are fired at the depth the shallowest can reach.
        runahead: engine::runahead::Runahead::F1,
        residency,
    })
    .unwrap_or_else(|why| panic!("the {what} shell loads: {why}"));
    let millis = at.elapsed().as_secs_f64() * 1000.0;
    let booted = wired();

    shell.open(0).expect("the slot opens");
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .unwrap_or_else(|why| panic!("the {what} prefill fires: {why}"));
    let fired = wired();
    finite(&prefill[0], what);
    let mut tokens = vec![argmax(&prefill[0])];
    for step in 0..STEPS {
        let fed = [*tokens.last().expect("a step feeds the last token back")];
        let decode = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &fed,
            }])
            .unwrap_or_else(|why| panic!("{what} decode step {step} fires: {why}"));
        finite(&decode[0], what);
        tokens.push(argmax(&decode[0]));
    }
    Arm {
        what,
        tokens,
        millis,
        warm: shell.weights_warm(),
        kind: shell.expert_source_kind(),
        source: shell.expert_source(),
        motion: shell.expert_motion(),
        slabs: shell.expert_residency().len(),
        boot_wired: delta(idle, booted),
        fire_wired: delta(booted, fired),
    }
}

/// **THE CLAIM.** Four roads to one set of weights, and they answer the same
/// nine tokens — with the fourth mapping its artifact AND streaming out of it.
///
/// One test rather than four, because the four arms are one comparison and a
/// box that holds two of this model at once is a box measuring the wrong
/// thing.
#[test]
fn a_streamed_artifact_seats_its_experts_off_the_pages_it_mapped() {
    if !engine_metal::device::present() {
        eprintln!("skipping the warm-streamed arm: this machine publishes no Metal device");
        return;
    }
    let Some(snapshot) = snapshot() else {
        eprintln!(
            "skipping the warm-streamed arm: no {REPO} snapshot with a tokenizer beside \
             it — name one in PIE_U2_SNAPSHOT"
        );
        return;
    };
    let Some(artifact) = artifact() else {
        eprintln!(
            "skipping the warm-streamed arm: no `metal`-stamped {SKU} artifact found — \
             import one with an ENGINE-METAL-FEATURE binary (see this file's header; \
             `-p pie`, never a workspace build) and name it in PIE_METAL_MOE_ARTIFACT"
        );
        return;
    };
    eprintln!("artifact {artifact:?}\nsnapshot {snapshot:?}");
    let tokenizer = tokenizer::Tokenizer::from_file(&snapshot.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let prompt = tokenizer.encode(PROMPT);

    // **THE RESIDENCY PLAN IS THE SNAPSHOT'S, AND BOTH STREAMED ARMS FIRE
    // IT.** A plan is a function of the trace and the load plan's pairing, and
    // the two roads must not be allowed to disagree about how many experts a
    // slab seats — otherwise the token comparison below is between two
    // different mechanisms rather than between two sources for one.
    let plan = seating(&read(&snapshot), SEATS);
    eprintln!(
        "the streamed plan seats {} of 16 experts across {} groups, {} bands",
        plan.slots(),
        plan.groups().len(),
        plan.bands().len(),
    );

    // ── the four arms, ONE RESIDENT AT A TIME, warm first.
    //
    //    The order is the LOAD TIME claim's, and it is the conservative one:
    //    the warm arm reads the artifact with a cold page cache and the cold
    //    arms read a snapshot two of them have already walked, so every
    //    millisecond of the ratio printed below is one the warm arm earned
    //    against a handicap.
    //
    //    **AND IT IS WHY THE FIRST FIRE'S WIRED NUMBER IS PRINTED AND NOT
    //    ASSERTED.** Whichever arm fires first pays this process's one-time GPU
    //    state — the pipeline library, the shader heaps — and it is the larger
    //    term. Measured, by running this same file with the order reversed:
    //    the +2.4 GiB moved to `cold-full` and `warm-streamed`'s first fire
    //    fell from +2.718 to +0.890 GiB, while every other number held. The
    //    reading that IS about the mechanism is the BOOT one, which is flat on
    //    all four arms in either order: binding a mapping costs nothing and the
    //    GPU touch is what pays (`crate::mapping`'s header).
    let warm_streamed = arm("warm-streamed", &artifact, plan.clone(), &prompt);
    let warm_full = arm("warm-full", &artifact, Plan::default(), &prompt);
    let cold_streamed = arm("cold-streamed", &snapshot, plan.clone(), &prompt);
    let cold_full = arm("cold-full", &snapshot, Plan::default(), &prompt);
    let arms = [&warm_streamed, &warm_full, &cold_streamed, &cold_full];

    for arm in arms {
        eprintln!(
            "{:<14} {:>7.0} ms  warm={:<5} source={:<9} {:?}  motion={:?} slabs={}  \
             wired boot {} first fire {}",
            arm.what,
            arm.millis,
            arm.warm,
            arm.kind.unwrap_or("none"),
            arm.source,
            arm.motion,
            arm.slabs,
            arm.boot_wired,
            arm.fire_wired,
        );
    }

    // ── (1) THE WARM-STREAMED ARM IS BOTH WARM AND STREAMED. Either half
    //    alone was already gated; what is new is that one load is both.
    assert!(
        warm_streamed.warm,
        "the streamed load of {artifact:?} did not take the warm arm — the reason was \
         printed by `weights::warm` on the way past, and every one of them is a fact \
         about the file or the plan rather than a flake"
    );
    assert_eq!(
        warm_streamed.kind,
        Some("artifact"),
        "a warm streamed load's seat copies must read the artifact it already mapped; \
         `landed` here would mean it staged a second file, which is the copy this arm \
         exists to remove"
    );
    assert!(
        warm_streamed.slabs > 0,
        "a streamed load with no slab is a full-residency load wearing the word"
    );
    assert_eq!(
        warm_streamed.slabs,
        plan.groups().len(),
        "one slab per routed group, as the plan states"
    );
    // **SEGMENTS ARE STRUCTURAL AND ARE ASSERTED EXACTLY**: one cut per routed
    // group per fire, which is a number no name-matched encode trigger can hit
    // by accident (`two_bit_experts_stream`'s header tells the story of the one
    // that missed three of five groups and served finite nonsense).
    let fires = STEPS as u64 + 1;
    assert_eq!(
        warm_streamed.motion.1,
        fires * plan.groups().len() as u64,
        "a warm streamed load cuts its command buffer after EVERY router of every \
         fire, or the selects behind an uncut mixture read expert ids against a slab \
         indexed by seat"
    );
    assert_eq!(
        cold_streamed.motion.1, warm_streamed.motion.1,
        "the two streamed arms run the same plan and cut the same segments"
    );
    assert!(
        warm_streamed.motion.0 > 0,
        "the routing wandered off the identity prefix, so a streamed load that copied \
         no seat is one whose tier never moved"
    );

    // **AND THE SEAT COPIES ARE NOT ASSERTED EQUAL BETWEEN THE ARMS, WHICH IS
    // A FACT ABOUT THE IMPORT AND NOT ABOUT THIS MECHANISM.** They come out
    // deterministic and CLOSE — 819 against 837 as this is written, which is
    // 91 seat copies against 93 over nine bands a group — and the gap is two
    // near-ties in a router's top-k falling the other way. `pie model import`
    // NARROWS 35 of this checkpoint's planes to bf16 on the way past, and the
    // planes the model text declares f32 are cast back by the warm arm's
    // residue landing — so the warm arm's gated-delta scan runs on bf16-rounded
    // coefficients where the cold arm's runs on the source f32. It moves an
    // expert score by an ulp, it occasionally reorders two adjacent experts of
    // sixteen, and it changes which seat the clock evicts.
    //
    // What it does NOT move is the answer, which is the assert that matters
    // and is made below: nine tokens, four arms, identical. A swap-count
    // equality here would be gating the import's narrowing policy through a
    // residency test, and the narrowing is a question for the importer.
    eprintln!(
        "seat copies: warm-streamed {} band copies, cold-streamed {} — the gap is the \
         import's bf16 narrowing reordering a near-tie in a router's top-k, not the \
         tier (see this file's note); the segments and the TOKENS agree exactly",
        warm_streamed.motion.0, cold_streamed.motion.0,
    );

    // ── (2) AND THE SOURCE IS THE ARTIFACT ITSELF, not a copy of it. The
    //    pair's first number is the mapped file's own size and its second is
    //    the link count — at least one, because this load opened a file the
    //    operator named and did not create it. The cold arm's zero says the
    //    opposite thing about a different file, which is why the KIND is
    //    asserted above before the pair is read at all.
    let bytes = std::fs::metadata(&artifact)
        .expect("the artifact stats")
        .len();
    assert_eq!(
        warm_streamed.source,
        Some((bytes, 1)),
        "the warm arm's seat source must fstat as the artifact itself ({bytes} bytes, \
         linked) — a different size is a second file, and zero links is the cold arm's \
         unlinked temporary"
    );
    assert_eq!(
        cold_streamed.kind,
        Some("landed"),
        "the cold arm stages the bands into a file it writes"
    );
    assert_eq!(
        cold_streamed.source.map(|(_, links)| links),
        Some(0),
        "and that file is UNLINKED, so nothing outside this process can reach it"
    );

    // ── (3) THE OTHER TWO ARMS ARE WHAT THEY SAY THEY ARE.
    assert!(warm_full.warm, "the artifact maps at full residency too");
    assert_eq!(
        warm_full.kind, None,
        "a full-residency load streams nothing and opens no seat source"
    );
    assert_eq!(warm_full.motion, (0, 0), "and its tier does not exist to move");
    assert!(
        !cold_full.warm && !cold_streamed.warm,
        "a raw snapshot has no serving artifact to map, and these loads claimed it did"
    );
    assert_eq!(cold_full.kind, None, "nor does the cold full arm stream");

    // ── (4) FOUR ROADS, ONE ANSWER. The claim.
    for arm in arms {
        eprintln!(
            "{:<14} {:?} {:?}",
            arm.what,
            arm.tokens,
            tokenizer.decode(&arm.tokens, false)
        );
    }
    for arm in [&warm_full, &cold_streamed, &cold_full] {
        assert_eq!(
            warm_streamed.tokens, arm.tokens,
            "the warm-streamed arm answered differently from the {} one. Every band \
             offset this arm binds is read out of the serving manifest rather than \
             computed by the plan that reserved the seats, so a wrong reading seats \
             another expert's bytes and the model answers finite, deterministic \
             nonsense — which is what these tokens are here to notice",
            arm.what,
        );
    }

    // ── (5) AND IT IS FASTER THAN THE COLD STREAM, BY THE STAGING. What the
    //    warm arm removes from a streamed load is a read of every plane, a
    //    transform of it, and a write of the routed bands into a second file
    //    the seat copies then read back out of.
    eprintln!(
        "load: cold-streamed {:.0} ms, warm-streamed {:.0} ms — {:.1}x",
        cold_streamed.millis,
        warm_streamed.millis,
        cold_streamed.millis / warm_streamed.millis.max(f64::MIN_POSITIVE),
    );
    assert!(
        warm_streamed.millis < cold_streamed.millis,
        "the mapped streamed load took {:.0} ms against the staged one's {:.0} ms — the \
         arm exists to remove a read of the whole model and a staging file, so a warm \
         load that is not faster is one that did neither",
        warm_streamed.millis,
        cold_streamed.millis,
    );
}
