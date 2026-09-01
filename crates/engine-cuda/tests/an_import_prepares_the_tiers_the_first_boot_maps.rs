//! **§M wave M-1: the IMPORT writes the tiers, and the first boot maps them.**
//!
//! §K/§L made a streamed boot write an artifact and the next boot read it. What
//! neither wave moved is WHICH boot pays: the file is written by the first load
//! that materializes the weights, and on a fresh deployment that load is the
//! first SERVE. So an operator who runs `pie model import` and then `pie serve`
//! waits out the whole cold path on their first request — 290-440 s at qwen4
//! scale, against 21.5 s once the file exists.
//!
//! `Shell::prepare` is the door that moves it. It runs the cold half of a load
//! — the same `bake`, the same `Weights::resident`, the same write — and then
//! tears the device down without arming anything a serve would need. This gate
//! is the claim:
//!
//! ```text
//!  1. written -> `prepare` against an EMPTY cache directory leaves exactly one
//!                file, at `tier::path(dir, key)`, under the key a load of the
//!                same seat forms — and the counter says `stored`
//!  2. whole   -> it opens, and every block digest verifies
//!  3. clean   -> nothing is left behind: the device's free bytes come back to
//!                where they were, no `.part` survives, and a full `Shell::load`
//!                at the same seat still fits on the card afterwards
//!  4. warm    -> that load RESTORES (`restored` rises, `stored` does not),
//!                which is the whole point: the first serve after an import is
//!                the warm kind
//!  5. said    -> and it answers: a store that reads back, tokens that are not
//!                one token repeated, and the logit rows claim 6 is measured
//!                against
//!  6. ANY BUDGET -> and so does a boot at a DIFFERENT device budget, off the
//!                SAME file: it restores rather than stores, no second
//!                artifact appears, its store is a DIFFERENT store (the cut is
//!                a cut), and its logits are bit-identical to the first
//!                budget's
//!  0. refused -> and none of it is optional. A boot of the same seat against
//!                the EMPTY directory does not fall back to a cold load; it
//!                refuses, naming `pie model import --prepare-only` (§M-3)
//! ```
//!
//! **(6) IS M-2, AND IT IS WHAT THE IMPORT PROMISED ALL ALONG.** M-1 moved the
//! cold path into `pie model import` while the artifact was still keyed to a
//! budget pair, which meant an operator who raised `device_weight_budget`
//! silently went back to paying the cold path on their first request — the
//! import had prepared a file that seat no longer named. §M.3 takes the rungs
//! and the budgets out of the key: the file is the ranking, the budget is a
//! LOAD-TIME CUT, and one `pie model import` serves every budget this machine
//! will ever be configured with.
//!
//! **(0) IS M-3, AND IT TURNS THIS GATE'S SUBJECT FROM AN OPTIMIZATION INTO A
//! REQUIREMENT.** M-1 moved the cold path into `pie model import` and kept the
//! boot-side one as a fallback, so everything above was about WHO PAYS. M-3
//! removes the fallback: `Shell::load` passes `Intent::Serve`, a streamed load
//! under that intent is warm or refused, and `Shell::prepare` is the only
//! writer of one of these files in the process. The import does not save the
//! first serve a wait any more — it is the reason there is a first serve.
//!
//! # What claim 5 used to be, and where that claim went
//!
//! It was *the warm boot says exactly what a COLD boot of the same seat said*,
//! and it was measured by booting the same streamed plan with NO cache
//! directory at all — a load that neither read nor wrote, and was therefore
//! the pre-§M baseline. That load refuses now, and correctly: a streamed plan
//! with no artifact to cut has nothing to serve out of. So the baseline is
//! gone, and it is gone in both places this gate used it (claim 5 and claim
//! 6's per-budget arm).
//!
//! What replaces it is not weaker, because the two halves it was carrying have
//! better homes. **Absolute correctness** — that a streamed load says what
//! this model says — wants an UNCAPPED RESIDENT load to compare against, whose
//! path this wave does not touch, and that gate is
//! `a_capped_moe_serves_the_tokens_it_would_have`. **Cut-invariance** — that
//! moving a plane between rungs cannot move a logit, which is §M.3's whole
//! measured premise — is claim 6, and it needs no cold boot at all: two
//! budgets, two stores that differ by digest, one file, and identical floats.
//! That is the comparison this gate is actually for, and it is now the only
//! one it makes.
//!
//! # What this gate does NOT do
//!
//! It does not drive the `pie model import` CLI. That command is in the `pie`
//! package, needs a serving `config.toml` to read a device and a budget out of,
//! and converts a checkpoint before it prepares anything — three things that
//! have nothing to do with the claim above and one of which (the config) would
//! make this gate depend on the machine's `$PIE_HOME`. What import contributes
//! is the CALL, and the call is `Shell::prepare`; the plumbing that reaches it
//! (`worker::embedded_engine::prepare_weight_artifact` ->
//! `runtime::engine::backend::open::prepare_cuda` -> `Cuda::prepare`) is
//! ordinary config threading with no device in it.
//!
//! ```bash
//! cargo test -p engine-cuda --features cuda-13 --release \
//!     --test an_import_prepares_the_tiers_the_first_boot_maps -- --ignored --nocapture
//! ```
//!
//! # Gating
//!
//! `#[ignore]`d: it wants a CUDA device, the gpt-oss-20b snapshot on disk, and
//! room under `TMPDIR` for the artifact (~15 GiB at this budget). It runs one
//! refusal, one prepare and two warm boots, sequentially — half of what it
//! cost before, because the two cold baselines are the thing §M-3 deleted.
//! Skips with a sentence when any of it is missing, the same convention its
//! siblings use.

use std::path::{Path, PathBuf};
use std::time::Instant;

use engine_cuda::experts::{Budgets, Plan};
use engine_cuda::weight_cache::tier;
use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "gptoss-20b-bf16-mxfp4-kv-bf16";

/// **The device budget, and it is the point of the gate.** Four gibibytes
/// under a table of roughly thirteen: whole mxfp4 groups land on the pinned
/// tier, the host tier is uncapped so none of them reach the mapping, and the
/// load streams — which is the only shape that writes a tier artifact at all.
const DEVICE: u64 = 4 << 30;

/// **THE SECOND BUDGET, AND IT IS CLAIM 6.** Eight gibibytes under the same
/// thirteen: still a streamed load, still whole mxfp4 groups on the pinned
/// tier, and roughly twice as many of them staying on the device. Nothing
/// prepares at this budget — the whole claim is that nothing has to.
const OTHER: u64 = 8 << 30;

/// How much room the artifact wants under `TMPDIR`, plus the writer's margin.
const ROOM: u64 = 24 << 30;

/// **How much of the card `prepare` may still be holding when it returns.**
///
/// Zero is not assertable: `cudaFree` returns pages to the driver's allocator
/// and `cudaMemGetInfo` is a whole-device figure that a context teardown, a
/// module unload or another process moves under the test's feet. What IS
/// assertable is the order of magnitude — the store this plan demands is four
/// gibibytes, so a `prepare` that leaked it would show up hundreds of times
/// over this margin.
const LEAK: u64 = 512 << 20;

/// The harmony turn, written out rather than templated, as its siblings write
/// it: this binary is its own crate and what it needs is a deterministic
/// prompt.
const PROMPT: &str = "<|start|>user<|message|>What is the capital of France? \
                      Answer in one word.<|end|>\
                      <|start|>assistant<|channel|>final<|message|>";

/// How many greedy decodes follow the prefill.
const STEPS: usize = 8;

/// A temporary directory that removes itself, however the test leaves —
/// including the gigabytes the artifact costs.
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

/// The trace, the contract, the checkpoint and the tokenizer — everything the
/// three passes share, read once.
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

/// Everything a load states, at this seat.
///
/// **ONE BUILDER FOR BOTH DOORS, AND THAT IS DELIBERATE.** The claim this gate
/// makes is that `prepare` and `load` are the same cold half; a `Boot` spelled
/// twice would let the two drift in a field neither assertion would catch.
/// `dir` is the only thing that varies — `None` for the cold baseline, which
/// reads no artifact and writes none.
fn boot<'a>(rig: &'a Rig, plan: Plan, dir: Option<&'a Path>) -> Boot<'a> {
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
        weight_cache_dir: dir,
        residency: plan,
    }
}

/// A prefill and `STEPS` greedy decodes, feeding the argmax back. Answers the
/// tokens it chose and the logit rows it chose them from.
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
    assert_eq!(
        golden.len(),
        found.len(),
        "{what} produced a different number of rows"
    );
    for (step, (a, b)) in golden.iter().zip(found).enumerate() {
        assert_eq!(
            a.len(),
            b.len(),
            "{what} step {step} has {} logits where cold had {}",
            b.len(),
            a.len()
        );
        for (at, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{what} step {step}, logit {at}: cold {x}, this load {y} — an \
                 import-written artifact moved a number"
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

// ─────────────────────────────────────────────────────────────────────────────

/// **All five claims, in one cold boot, one prepare and one warm boot.**
#[test]
#[ignore = "real-hardware: needs a CUDA device, a local gpt-oss-20b snapshot \
            and ~24 GiB under TMPDIR; run it with `-- --ignored`"]
fn an_import_prepares_the_tiers_the_first_boot_maps() {
    let Some(rig) = rig("the import-prepare gate") else {
        return;
    };
    let dir = scratch("import-prepare");
    if free(&dir.0) < ROOM {
        eprintln!(
            "skipping the import-prepare gate: {:?} has {} GiB free and the artifact \
             wants {} GiB",
            dir.0,
            free(&dir.0) >> 30,
            ROOM >> 30
        );
        return;
    }

    // ── THE PLAN AND THE KEY, FORMED FROM OUTSIDE. The same statements a load
    //    forms inside `Weights::resident`, which is what makes naming the file
    //    an assertion rather than a search.
    let prospect = engine_cuda::weights::prospect(&rig.trace, &rig.contract, &rig.checkpoint)
        .expect("the load plan pairs every packed bank with its scales");
    let plan = Plan::of(&rig.trace, &prospect.planes, Budgets::device(DEVICE))
        .expect("a capped mxfp4 MoE plans rather than refusing");
    assert!(
        plan.streams(),
        "a 4 GiB budget under this table has to stream"
    );
    assert_eq!(
        plan.spill_demand(),
        0,
        "the host tier is uncapped, so nothing reaches the mapping and this \
         seat has a file to write"
    );
    let key = engine_cuda::weights::tier_key(&rig.trace, &rig.contract, &rig.checkpoint)
        .expect("the key is a function of the trace and the recipe")
        .expect("this plan serializes, so this deployment forms a key");
    assert_eq!(key, prospect.tier_key, "the door and the prospect state one number");
    let path = tier::path(&dir.0, key);
    assert!(!path.exists(), "nothing is cached before the prepare");

    let prompt = rig.tokenizer.encode(PROMPT);
    assert!(prompt.len() > 4, "the harmony turn encodes to something");

    // ── (0) THE REFUSAL, WHICH USED TO BE THE COLD BASELINE (§M-3).
    //
    //    A boot with NO cache directory at all stood here: it read nothing and
    //    wrote nothing, so it was exactly the load this deployment had before
    //    §M, and every claim below was measured against it. It is not a load
    //    any more. A streamed plan with no artifact to cut has nothing to
    //    serve out of, and the honest answer to that is a refusal rather than
    //    a hundred gigabytes of transforms the operator did not ask for.
    //
    //    So the same call is made and the opposite thing is asserted, and it
    //    is asserted FIRST, because it is the premise: if this served, the
    //    prepare below would be an optimization rather than the only writer.
    let refused = Shell::load(boot(&rig, plan.clone(), None))
        .err()
        .expect("a streamed load with no weight cache directory does not serve");
    let sentence = format!("{refused:?}");
    eprintln!("with no cache directory at all: {sentence}");
    assert!(
        sentence.contains("weight_cache_dir"),
        "the refusal names the setting that is missing: {sentence}"
    );
    assert!(
        sentence.contains("pie model import --prepare-only"),
        "and the command that writes the file: {sentence}"
    );

    //    AND THE SAME SEAT AGAINST THE EMPTY DIRECTORY, which is the case an
    //    operator actually meets: the config is right, the model has simply
    //    never been prepared on this box.
    let refused = Shell::load(boot(&rig, plan.clone(), Some(&dir.0)))
        .err()
        .expect("an empty weight cache is not a serving artifact");
    let sentence = format!("{refused:?}");
    eprintln!("with an empty cache directory: {sentence}");
    assert!(
        sentence.contains("never been prepared"),
        "an empty directory is the never-prepared case, not the changed-recipe \
         one: {sentence}"
    );
    assert!(
        sentence.contains("pie model import --prepare-only"),
        "and it names the command either way: {sentence}"
    );
    assert!(
        !path.exists(),
        "and a refused load writes no artifact"
    );

    // ── (1) THE PREPARE WRITES, AND IT IS THE ONLY THING IT LEAVES.
    let idle = engine_cuda::device::free_bytes();
    let before = tier::observed();
    let clock = Instant::now();
    Shell::prepare(boot(&rig, plan.clone(), Some(&dir.0))).expect("the prepare runs");
    let prepared = clock.elapsed();
    let after = tier::observed();
    eprintln!("PREPARE: {prepared:.1?}, and it left {after:?}");
    assert_eq!(
        after.stored,
        before.stored + 1,
        "one prepare, one tier artifact: {before:?} -> {after:?}"
    );
    assert_eq!(after.declined, before.declined, "and nothing was declined");
    assert_eq!(
        after.skipped, before.skipped,
        "and nothing was already there"
    );
    assert_eq!(
        after.restored, before.restored,
        "and a prepare against an empty directory restores nothing"
    );
    assert!(path.exists(), "under the key a load forms: {path:?}");
    assert_eq!(
        std::fs::read_dir(&dir.0)
            .expect("the cache directory")
            .flatten()
            .count(),
        1,
        "one file, and no `.part` left behind"
    );

    // ── (2) IT OPENS AND EVERY BLOCK VERIFIES.
    let artifact = tier::Artifact::open(&path).expect("what the prepare published opens");
    assert_eq!(artifact.key(), key, "under the key it was asked for");
    artifact
        .verify()
        .expect("every block hashes to what the table states");
    let head = artifact.head();
    let images = prospect.ranking.images();
    assert_eq!(
        head.entries as usize,
        images.len(),
        "one image per plane the ranking ranks"
    );
    assert_eq!(
        head.payload_total,
        images.iter().map(|(_, _, _, reserved)| reserved).sum::<u64>(),
        "and the payload is exactly what their spans tile"
    );
    drop(artifact);
    let stamp = std::fs::metadata(&path)
        .expect("the file")
        .modified()
        .expect("a modification time");

    // ── (3) AND IT LEFT NOTHING ON THE CARD.
    //
    //    A prepare that forgot to drop its store, or that detached the refill
    //    thread §L arms instead of joining it, would still be holding the four
    //    gibibytes this plan demands — and the warm load below would be asking
    //    the card for them a second time. `Refill`'s `Drop` is a join for
    //    exactly this reason, and this is the observation that says so.
    if let (Some(idle), Some(now)) = (idle, engine_cuda::device::free_bytes()) {
        let held = idle.saturating_sub(now);
        assert!(
            held < LEAK,
            "the prepare is still holding {held} bytes of device memory ({} free \
             before, {now} after); the store or the tier outlived the call",
            idle
        );
    }

    // ── (4) THE NEXT BOOT IS THE WARM ONE — which is the whole of M-1.
    let before = tier::observed();
    let clock = Instant::now();
    let mut shell =
        Shell::load(boot(&rig, plan.clone(), Some(&dir.0))).expect("the warm shell loads");
    let warm = clock.elapsed();
    let after = tier::observed();
    eprintln!("WARM boot after the prepare: {warm:.1?}, and it left {after:?}");
    assert_eq!(
        after.restored,
        before.restored + 1,
        "the first boot after an import reads what the import wrote: \
         {before:?} -> {after:?}"
    );
    assert_eq!(
        after.stored, before.stored,
        "and writes nothing, the file being the one it just read"
    );
    assert_eq!(
        after.corrupt, before.corrupt,
        "and finds nothing wrong with it"
    );
    assert_eq!(
        std::fs::metadata(&path)
            .expect("the file is still there")
            .modified()
            .expect("a modification time"),
        stamp,
        "the bytes were not rewritten"
    );

    // ── (5) AND IT SAYS SOMETHING — the golden claim 6 is measured against.
    //    The cold comparator that stood here died with the cold serving path;
    //    the header argues where the two things it was proving went.
    assert!(
        !shell.weights_resident(),
        "a 4 GiB budget under a 13 GiB table streams, or this gate tests nothing"
    );
    assert!(
        shell.weights_from_cache(),
        "and it came off the file the prepare wrote"
    );
    let warm_digest = shell.weight_digest().expect("the store reads back");
    let (warm_said, warm_rows) = run(&mut shell, &prompt);
    let warm_text = rig.tokenizer.decode(&warm_said, false);
    drop(shell);
    assert!(
        warm_said.iter().collect::<std::collections::BTreeSet<_>>().len() > 1,
        "the warm load answered {warm_said:?}, which is one token repeated"
    );
    eprintln!(
        "prepare {prepared:.1?} + warm {warm:.1?}; digest {warm_digest:016x}, \
         answered {warm_text:?} — and the deployment's first serve pays the warm \
         figure because there is no other figure to pay"
    );

    // ── (6) AND A DIFFERENT BUDGET READS THE SAME FILE (§M.3).
    //
    //    Nothing is prepared for this budget. Under M-1's format it would have
    //    formed another key, found nothing, and paid the cold path on its
    //    first request — the exact failure the import exists to remove, moved
    //    one configuration change downstream.
    let other = Plan::of(&rig.trace, &prospect.planes, Budgets::device(OTHER))
        .expect("a capped mxfp4 MoE plans rather than refusing");
    assert!(other.streams(), "the second budget streams too");
    assert_ne!(
        other.groups().len(),
        plan.groups().len(),
        "the two budgets have to put different numbers of groups off the device, \
         or claim 6 is one budget asserted twice"
    );

    //    A COLD BASELINE AT THIS BUDGET STOOD HERE, with no cache directory —
    //    "the load this deployment would have had at THIS budget before §M".
    //    §M-3 refuses it, and the claim below no longer needs it: what makes
    //    the cut a cut is that the two STORES differ, which is checked by
    //    digest against the first budget's, and what makes it safe is that the
    //    LOGITS do not, which is checked against the first budget's too. Both
    //    comparators are warm boots off the one file, which is the only thing
    //    this deployment has.
    let before = tier::observed();
    let clock = Instant::now();
    let mut shell =
        Shell::load(boot(&rig, other, Some(&dir.0))).expect("the other warm shell loads");
    let other_warm = clock.elapsed();
    let after = tier::observed();
    eprintln!(
        "WARM boot at {} GiB off the {} GiB prepare: {other_warm:.1?}, and it left \
         {after:?}",
        OTHER >> 30,
        DEVICE >> 30,
    );
    assert_eq!(
        after.restored,
        before.restored + 1,
        "a boot at another budget RESTORED out of the imported file: \
         {before:?} -> {after:?}"
    );
    assert_eq!(
        after.stored, before.stored,
        "and wrote nothing — one import, one file, whatever the budget"
    );
    assert_eq!(
        after.corrupt, before.corrupt,
        "and found nothing wrong with a file another budget wrote"
    );
    assert_eq!(
        std::fs::read_dir(&dir.0)
            .expect("the cache directory")
            .flatten()
            .count(),
        1,
        "STILL one file: the second budget did not name a second one"
    );
    assert_eq!(
        std::fs::metadata(&path)
            .expect("the file is still there")
            .modified()
            .expect("a modification time"),
        stamp,
        "and not one byte of it was rewritten"
    );

    //    A DIFFERENT STORE, WHICH IS WHAT MAKES THE CUT A CUT. One file, two
    //    budgets, two layouts — if these hashed the same the second budget was
    //    not putting anything anywhere different and claim 6 would be one
    //    budget asserted twice.
    let other_warm_digest = shell.weight_digest().expect("the store reads back");
    assert_ne!(
        other_warm_digest, warm_digest,
        "the two budgets have to lay out different stores, or the cut is not a cut"
    );

    //    AND THE SAME FLOATS ANYWAY. §M.3's measured fact, asserted: a plane's
    //    bytes are identical on all three rungs, so which rung a budget puts
    //    it on cannot move a logit. This is the comparison the whole gate is
    //    for, and since §M-3 it is the only one it makes.
    let (other_said, other_rows) = run(&mut shell, &prompt);
    drop(shell);
    assert_eq!(
        other_said, warm_said,
        "the {} GiB boot chose {other_said:?} and the {} GiB one chose {warm_said:?}",
        OTHER >> 30,
        DEVICE >> 30,
    );
    same_logits(&warm_rows, &other_rows, "the other budget, against the first");
    eprintln!(
        "ONE IMPORT, TWO BUDGETS: {} GiB warm {warm:.1?}, {} GiB warm \
         {other_warm:.1?}, one prepare of {prepared:.1?}, one file of {} bytes",
        DEVICE >> 30,
        OTHER >> 30,
        std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0),
    );
}
