//! **THE WARM ARM, MEASURED AND ANSWERED** (§M-5) — a serving artifact
//! stamped for this deployment loads by MAPPING, and says the same thing the
//! cold load says.
//!
//! `four_bit_first_light` is the vehicle and this is its second road. That
//! suite loads the raw `mlx_lm.convert` snapshot — every plane read off disk,
//! transformed host-side and copied into a device store the size of the model
//! — and asserts the continuation of its prompt begins " Paris". This file
//! loads `pie model import`'s artifact of the SAME checkpoint through
//! `weights::warm`, where nothing is read and nothing is copied: the file is
//! mapped once, bound through `newBufferWithBytesNoCopy`, and every weight row
//! is a handle into it at the offset the serving manifest states.
//!
//! Four claims, and the first three are the whole of the arm:
//!
//! 1. **IT MAPS.** `Shell::weights_warm` is true for the artifact and false
//!    for the snapshot beside it — the field `LoadFacts::weights_from_cache`
//!    publishes, which was `false` unconditionally until this landed.
//! 2. **IT IS FASTER, AND BY THE COPY.** Both loads are timed and both
//!    numbers are printed. What the warm arm removes is a read of the whole
//!    model plus a `memcpy` of it into pages the GPU was about to wire
//!    anyway; what it keeps is the manifest parse and the load-plan compile,
//!    because a quantized bank's three planes are a fact about the compiled
//!    plan (`weights::pairings`) and not about where the bytes came from.
//! 3. **IT SAYS THE SAME THING.** Nine greedy tokens, and they are the SAME
//!    NINE. This is the claim that separates a mapped load from a
//!    mapped-shaped one: every offset in this file is read out of a manifest
//!    rather than computed by the plan that reserved the store, so a wrong
//!    reading binds every plane shifted by a constant and answers finite,
//!    deterministic nonsense. The tokens are what notice.
//! 4. **AND THE WIRED PAGES ARE OBSERVED.** A `StorageModeShared` page wires
//!    when the GPU TOUCHES it, not when it is bound (`crate::mapping`'s
//!    header, and the measurement in `.wiki/alto/streaming.md`). So a warm
//!    boot is cheap and the first fire pays — this file prints the global
//!    `Pages wired down` delta across each, because the two-trap shape is
//!    the thing an operator has to be told rather than something to assert
//!    on a shared counter.
//!
//! # The artifact
//!
//! ```text
//! cargo run --release --features engine-metal -- \
//!     model import ~/.cache/huggingface/hub/\
//! models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/local \
//!     --out /tmp/qwen35-mlxu4-metal.zt
//! ```
//!
//! `this_box()` is Metal only in a binary built with the feature, and the
//! stamp it writes is what `weights::serves_this_deployment` compares — so an
//! artifact imported by a CUDA-featured or featureless `pie` is refused at the
//! door and never reaches the warm arm at all (`a_cross_recipe_artifact_-
//! refuses_before_it_allocates` is that gate's own file).
//!
//! `PIE_METAL_ARTIFACT` names it; absent that, any `.zt` under `$PIE_HOME`'s
//! model store whose stamp reads back for this backend and this SKU.
//!
//! ```text
//! cargo test -p engine-metal --release --test a_warm_load_is_the_artifact_mapped \
//!     -- --nocapture --test-threads 1
//! ```

#![cfg(target_vendor = "apple")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// The row this file serves — `four_bit_first_light`'s, because the point is
/// that the two roads answer the same.
const SKU: &str = "qwen35-d0.8b-mlxu4-kv-bf16";

/// `four_bit_first_light::PROMPT`, and it is the same for the reason that
/// suite gives at length: on the bare prompt this checkpoint's top two logits
/// are exactly tied, and an assertion on the argmax would be reading which way
/// the rounding fell. Four words more and the same fact is decided by 2.06
/// logits.
const PROMPT: &str = "The capital of France is the city of";

/// What a correct load produces here, cold or warm.
const EXPECTED: &str = " Paris";

/// How many decode fires follow the prefill.
const STEPS: usize = 8;

/// Global `Pages wired down`, in bytes, off `vm_stat` — the same reading
/// `a_mapped_artifact_is_the_bytes_without_the_copy` takes, and for the same
/// reason: a mapped page wires against the KERNEL rather than against this
/// process, so a task-level counter does not see it.
///
/// `None` when `vm_stat` is not there or does not parse, which is a reason to
/// print nothing rather than to fail.
fn wired() -> Option<u64> {
    let said = std::process::Command::new("vm_stat").output().ok()?;
    let text = String::from_utf8_lossy(&said.stdout);
    let page: u64 = text
        .lines()
        .next()?
        .split("page size of ")
        .nth(1)?
        .split(' ')
        .next()?
        .parse()
        .ok()?;
    let pages: u64 = text
        .lines()
        .find(|line| line.starts_with("Pages wired down:"))?
        .split(':')
        .nth(1)?
        .trim()
        .trim_end_matches('.')
        .parse()
        .ok()?;
    Some(pages * page)
}

/// A signed GiB delta between two `vm_stat` readings, as a sentence — or
/// `unavailable` when either reading was.
fn delta(before: Option<u64>, after: Option<u64>) -> String {
    match (before, after) {
        (Some(before), Some(after)) => format!(
            "{:+.3} GiB",
            (after as f64 - before as f64) / (1u64 << 30) as f64
        ),
        _ => "unavailable".to_string(),
    }
}

/// The raw MLX snapshot — `four_bit_first_light`'s finder, kept in step with
/// it deliberately: the cold arm below must load THE SAME checkpoint the
/// artifact was imported from, or the token comparison is about two models.
fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_U4_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let usable = |path: &Path| path.join("tokenizer.json").exists() && container(path).is_some();
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
    homes.iter().find_map(|home| {
        let hub = Path::new(home).join(".cache/huggingface/hub");
        let mut repos: Vec<PathBuf> = std::fs::read_dir(&hub)
            .ok()?
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let name = path.file_name()?.to_str()?.to_string();
                (name.starts_with("models--mlx-community--Qwen3.5") && name.ends_with("-4bit"))
                    .then_some(path)
            })
            .collect();
        repos.sort();
        repos.into_iter().find_map(|repo| {
            std::fs::read_dir(repo.join("snapshots"))
                .ok()?
                .filter_map(|entry| Some(entry.ok()?.path()))
                .find(|path| usable(path))
        })
    })
}

/// The container a contract is read against — one file of a snapshot,
/// whichever holds the tensors.
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

/// **THE SERVING ARTIFACT, FOUND BY ITS OWN STAMP AND NOT BY ITS NAME.**
///
/// A file that says `pie.serving/1`, `backend = "metal"` and this SKU is one
/// this box's warm arm can serve; a file named for the model and stamped for
/// another shell is not, and the difference is a field rather than a spelling.
/// `stamp_of` is the same three-answer door `weights::serves_this_deployment`
/// asks, so a candidate this finds is one that gate will accept.
fn artifact() -> Option<PathBuf> {
    let stamped = |path: &Path| {
        let stamp = checkpoint::file::serve::stamp_of(path).ok().flatten()?;
        (stamp.backend == "metal" && stamp.sku == SKU).then(|| path.to_path_buf())
    };
    if let Ok(stated) = std::env::var("PIE_METAL_ARTIFACT") {
        return stamped(Path::new(&stated));
    }
    let homes = [
        std::env::var("PIE_HOME").unwrap_or_default(),
        format!("{}/.pie", std::env::var("HOME").unwrap_or_default()),
        "/Users/ingim/.pie".to_string(),
    ];
    homes.iter().find_map(|home| {
        let mut found: Vec<PathBuf> = std::fs::read_dir(Path::new(home).join("models"))
            .ok()?
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                (path.extension()? == "zt").then_some(path)
            })
            .collect();
        found.sort();
        found.iter().find_map(|path| stamped(path))
    })
}

/// One load, timed, off `checkpoint` — a directory for the cold arm, the
/// artifact for the warm one.
///
/// The CONTRACT is read against whichever container that path holds, which is
/// what makes one function serve both: a raw MLX snapshot publishes planes the
/// load transforms, and a serving artifact publishes the landed results under
/// the same names, so `models::import_of` answers a contract for either and
/// the trace it is checked against is one trace.
fn load(checkpoint: &Path) -> (Shell, f64) {
    let container = if checkpoint.is_dir() {
        container(checkpoint).expect("the snapshot holds a tensor container")
    } else {
        checkpoint.to_path_buf()
    };
    let trace = models::trace_of(SKU).expect("the catalog ships the 4-bit SKU")(Platform::Metal);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    // Read FOR THIS SHELL, as `four_bit_first_light::ready` argues: a family's
    // text may state a `Dtype` placement, so the setup the contract is read
    // under has to be the setup the trace was taken for.
    let contract = models::placing_for(Platform::Metal, || {
        models::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
    })
    .expect("the SKU's import contract fits this checkpoint");
    drop(source);

    let at = Instant::now();
    let shell = Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint,
        tp_size: 1,
        precision: models::precision_of(SKU)
            .expect("the catalog states this row's precision")
            .to_string(),
        budget: Budget::new(4, 256),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the shell loads");
    (shell, at.elapsed().as_secs_f64() * 1000.0)
}

/// Greedy: the highest logit.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

/// The lane word the model's own `Classify` computes.
fn word(query_len: u32) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

fn finite(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits at all");
    assert!(
        logits.iter().all(|value| value.is_finite()),
        "{what} produced a non-finite logit, and one NaN means the whole row is noise"
    );
}

/// One prefill and [`STEPS`] decodes in one slot, greedy throughout, plus what
/// the FIRST fire cost in wired pages — which is the number the two-trap
/// account wants beside the load's.
fn run(shell: &mut Shell, slot: u32, prompt: &[u32]) -> (Vec<u32>, String) {
    shell.open(slot).expect("the slot opens");

    let before = wired();
    let prefill = shell
        .fire(&[Lane {
            slot,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the prefill fires");
    let fired = wired();
    assert_eq!(prefill.len(), 1, "one lane in, one row of logits out");
    finite(&prefill[0], "prefill");

    let mut produced = vec![argmax(&prefill[0])];
    for step in 0..STEPS {
        let fed = [*produced.last().expect("a step feeds the last token back")];
        let decode = shell
            .fire(&[Lane {
                slot,
                word: word(1),
                tokens: &fed,
            }])
            .unwrap_or_else(|why| panic!("decode step {step} fires: {why}"));
        finite(&decode[0], "decode");
        produced.push(argmax(&decode[0]));
    }
    (produced, delta(before, fired))
}

/// **THE CLAIM.** The artifact loads by mapping, and answers what the
/// checkpoint it was imported from answers.
///
/// One test rather than four, because the four claims are about ONE pair of
/// loads and a box that holds two of this model resident at once is a box that
/// measured the wrong thing.
#[test]
fn a_stamped_artifact_maps_and_answers_what_the_snapshot_answers() {
    if !engine_metal::device::present() {
        eprintln!("skipping the warm arm: this machine publishes no Metal device");
        return;
    }
    let Some(snapshot) = snapshot() else {
        eprintln!(
            "skipping the warm arm: no MLX 4-bit Qwen3.5 snapshot found — see \
             `four_bit_first_light`'s header, and name one in PIE_U4_SNAPSHOT"
        );
        return;
    };
    let Some(artifact) = artifact() else {
        eprintln!(
            "skipping the warm arm: no `metal`-stamped {SKU} artifact found — import one \
             with an ENGINE-METAL-FEATURE binary (`cargo run --release --features \
             engine-metal -- model import <snapshot> --out <path>.zt`) and name it in \
             PIE_METAL_ARTIFACT. A featureless or cuda-featured `pie` stamps another \
             backend and this deployment refuses it at the door, which is what \
             `a_cross_recipe_artifact_refuses_before_it_allocates` is for"
        );
        return;
    };
    let tokenizer = tokenizer::Tokenizer::from_file(&snapshot.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let prompt = tokenizer.encode(PROMPT);
    eprintln!("artifact {artifact:?}\nsnapshot {snapshot:?}");

    // ── THE WARM ARM, FIRST. The order is the wired reading's: a global
    //    `Pages wired down` delta only means anything from a baseline this
    //    process is not still holding a model against, and the cold shell's
    //    411 MiB come back to the kernel lazily after it drops. Measured
    //    first, dropped, and then the cold arm answers the same prompt.
    let (warm_tokens, warm_millis, warm_reservations) = {
        let idle = wired();
        let reserved = engine_metal::device::reservations();
        let (mut shell, millis) = load(&artifact);
        let reserved = engine_metal::device::reservations() - reserved;
        let bound = wired();
        assert!(
            shell.weights_warm(),
            "the {SKU} artifact at {artifact:?} did not take the warm arm — the reason \
             was printed by `weights::warm` on the way past, and every one of them is a \
             fact about the file or the plan rather than a flake"
        );
        let (tokens, first_fire) = run(&mut shell, 0, &prompt);
        eprintln!(
            "OBSERVED wired (global `Pages wired down`, this box, not an assertion): the \
             warm BOOT {}, and the first fire {first_fire} — the bind is cheap and the \
             GPU touch is what pays, which is `crate::mapping`'s whole header",
            delta(idle, bound),
        );
        (tokens, millis, reserved)
    };

    // ── THE COLD ARM: the raw snapshot, read and copied, exactly as
    //    `four_bit_first_light` loads it.
    let (cold_tokens, cold_millis, cold_reservations) = {
        let reserved = engine_metal::device::reservations();
        let (mut shell, millis) = load(&snapshot);
        let reserved = engine_metal::device::reservations() - reserved;
        assert!(
            !shell.weights_warm(),
            "a raw snapshot has no serving artifact to map, and this load claimed it did"
        );
        let (tokens, _) = run(&mut shell, 0, &prompt);
        (tokens, millis, reserved)
    };

    // ── what it cost, printed. The reservation counts are the process's and
    //    include the arena, the pools and the inputs — what separates the two
    //    arms is the WEIGHT reservation, which is a copy of the model on one
    //    road and a mapping plus a few hundred bytes of residue on the other.
    eprintln!(
        "load: cold {cold_millis:.0} ms ({cold_reservations} reservations), warm \
         {warm_millis:.0} ms ({warm_reservations} reservations) — {:.1}x",
        cold_millis / warm_millis.max(f64::MIN_POSITIVE),
    );

    // ── THE PARITY. Nine greedy tokens, and they are the same nine.
    let cold_text = tokenizer.decode(&cold_tokens, false);
    let warm_text = tokenizer.decode(&warm_tokens, false);
    eprintln!("cold {cold_tokens:?} {cold_text:?}\nwarm {warm_tokens:?} {warm_text:?}");
    assert_eq!(
        cold_tokens, warm_tokens,
        "the mapped load answered differently from the copied one, which means the \
         offsets this arm binds at are not the bytes the manifest names"
    );
    assert!(
        warm_text.starts_with(EXPECTED),
        "the mapped continuation is {warm_text:?}, and a correct load begins it {EXPECTED:?}"
    );
    assert!(
        warm_millis < cold_millis,
        "the mapped load took {warm_millis:.0} ms against the copied load's \
         {cold_millis:.0} ms — the arm exists to remove a read of the whole model and a \
         `memcpy` of it, so a warm load that is not faster is one that did neither"
    );
}

