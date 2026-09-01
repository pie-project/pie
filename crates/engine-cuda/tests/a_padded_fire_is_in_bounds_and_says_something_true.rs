//! **D4's gate: the tail a padded gemm writes belongs to nobody**
//! (`.wiki/palo/cuda-abi.md` §3, refined form; kill factors 1 and 2).
//!
//! `Knobs::pad` rounds an Always launch's `M` up to the fire's lattice point
//! before handing it to cuBLASLt, so the library's unpublished shape→kernel
//! table stops being a function of the batch. The rows `[rows, bucket)` that
//! buys are read and written garbage. Two claims stand between that and a
//! wrong answer, and this file is both of them run end to end:
//!
//! 1. **In bounds.** The arena reserves every `Dim::Tokens` column at
//!    `max_tokens` rows and P0 refuses a lattice above that ceiling, so the
//!    tail is reserved bytes. A padded fire that walked off a column would
//!    come back as a fault, a NaN, or a continuation that is not the pinned
//!    one — which is why the assertion here is `serve_smoke`'s own pin rather
//!    than a checksum: the smoke's " Paris" is the cheapest oracle in this
//!    tree for "the arithmetic is still the model's".
//! 2. **Harmless.** A gemm is row-independent, so tail garbage stays in tail
//!    rows. Measured on this device while the wave was wired, at the shapes
//!    this SKU runs: eight identical activation rows through one `M=8` call
//!    come back eight identical output rows, bit for bit, at every `M` from
//!    two to eight. Row-independence is a property of the call, not a hope
//!    about it.
//!
//! # What this file deliberately does NOT assert, and the measurement that
//! # says why
//!
//! Kill factor 1 asks for **byte-identical tokens across the `Knobs::pad`
//! A/B**, and that claim is FALSE — not because the tail leaks, but because
//! the feature works. Padding's whole purpose is to move which kernel
//! cuBLASLt picks; a different kernel is a different reduction order; a
//! different reduction order is a different bf16 number in the LIVE rows.
//! Measured on the L40S at the `[rows, 32]` projection this SKU's gated-delta
//! layers run: `M=6` and `M=8` disagree by up to 9 bf16 ulps on rows both of
//! them compute, and `M=1` (the gemv arm) disagrees with every other `M` by 2.
//! The last of those is TODAY's behaviour and has nothing to do with padding.
//!
//! So the honest statement of what padding does to determinism is not "it
//! preserves it" but **"it quantizes it"**: two fires agree bit-for-bit iff
//! they land in the same bucket, where today they agree iff cuBLASLt happened
//! to pick the same kernel for their two exact row counts. That is a stronger
//! and more predictable property than the one it replaces, and it is still a
//! CHANGE — [`a_solo_lane_and_a_batched_one_agree_inside_one_bucket`] is the
//! half of it this file can gate.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 --release \
//!   --test a_padded_fire_is_in_bounds_and_says_something_true -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use engine_cuda::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";
const PROMPT: &str = "The capital of France is";
const EXPECTED: &str = " Paris";
const STEPS: usize = 16;

/// One shell at a time per process — `serve_smoke.rs` argues it whole (four
/// copies of 1.7 GB on one card is arithmetic, not correctness).
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
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

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

fn word(query_len: u32) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

fn finite(logits: &[f32], what: &str) {
    let bad = logits.iter().position(|value| !value.is_finite());
    assert!(
        bad.is_none(),
        "{what} logit {} is {}, and one NaN out of a padded tail row would be \
         the whole of what this file exists to catch",
        bad.unwrap_or(0),
        logits[bad.unwrap_or(0)],
    );
}

/// **THE ARM, AND IT IS STATED ON THE `Boot` BECAUSE THAT IS WHERE IT IS
/// READ.** The pad and the lattice are load-time words on this plane — the
/// lattice is BAKED, and the pad is armed before each walk from a word read
/// once — so an A/B is two `Shell::load`s and not two settings on one shell.
///
/// They were `PIE_CUDA_PAD` and `PIE_CUDA_BUCKETS` and are `Knobs::pad` and
/// `Budget::buckets` now (alto wave P, article 9): the arm is a value the test
/// hands over, not a variable it sets in its own process.
fn load(pad: bool, lattice: Option<&[u32]>) -> Option<(Shell, tokenizer::Tokenizer)> {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return None;
    }
    let checkpoint = snapshot()?;
    let container = container(&checkpoint)?;
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");

    let trace = models::trace_of(SKU).expect("the catalog ships the SKU");
    let trace = trace(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract =
        models::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
            .expect("the SKU's import contract fits its own checkpoint");
    drop(source);

    let shell = Shell::load(Boot {
        // Full residency: the whole weight table on the device, which is what
        // an uncapped `Residency` plans (alto design §7).
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        // A stated lattice is kept and an unstated one is filled with the
        // shell's own default (`engine_cuda::api::lattice`), which is what
        // `None` here asks for.
        budget: {
            let mut budget = Budget::new(4, 256);
            if let Some(points) = lattice {
                budget.buckets = points.to_vec();
            }
            budget
        },
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        graphs: engine_cuda::Graphs::Off,
        knobs: engine_cuda::Knobs {
            pad,
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        // F1's depth, kept: these gates fire one step at a time and
        // read its numbers, so a deeper ring would carve slots nothing
        // claims. `Runahead::of` is the door a deployment comes through.
        runahead: engine::runahead::Runahead::F1,
        // The warm-boot weight artifact cache is off for a gate: a test
        // that shared one would be asserting about the last run.
        weight_cache_dir: None,
    })
    .expect("the shell loads");
    Some((shell, tokenizer))
}

/// One prefill and `STEPS` greedy decodes for `lanes` slots at once, and how
/// long each fire took.
///
/// `lanes` is the axis the waste measurement rides: with the default lattice's
/// floor at eight, a one-lane decode computes eight rows and a three-lane one
/// computes eight, so the two lane counts price the tail at 8x and 2.7x.
fn run(shell: &mut Shell, prompt: &[u32], lanes: u32) -> (Vec<Vec<u32>>, Vec<f64>) {
    for slot in 0..lanes {
        shell.open(slot).expect("the slot opens");
    }
    let seats: Vec<Lane<'_>> = (0..lanes)
        .map(|slot| Lane {
            slot,
            word: word(prompt.len() as u32),
            tokens: prompt,
        })
        .collect();
    let prefill = shell.fire(&seats).expect("the prefill fires");
    let mut said: Vec<Vec<u32>> = prefill
        .iter()
        .map(|row| {
            finite(row, "prefill");
            vec![argmax(row)]
        })
        .collect();

    let mut millis = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        let fed: Vec<[u32; 1]> = said
            .iter()
            .map(|lane| [*lane.last().expect("a step has a last token")])
            .collect();
        let seats: Vec<Lane<'_>> = (0..lanes as usize)
            .map(|slot| Lane {
                slot: slot as u32,
                word: word(1),
                tokens: &fed[slot],
            })
            .collect();
        let at = Instant::now();
        let out = shell
            .fire(&seats)
            .unwrap_or_else(|why| panic!("decode step {step} fires: {why}"));
        millis.push(at.elapsed().as_secs_f64() * 1000.0);
        for (lane, row) in out.iter().enumerate() {
            finite(row, "decode");
            said[lane].push(argmax(row));
        }
    }
    (said, millis)
}

/// The warm half's mean, which is what a per-fire cost is: the first fires of
/// a shape pay a jit compile and a cuBLAS heuristic query nobody is measuring.
fn warm(millis: &[f64]) -> f64 {
    let warm = &millis[millis.len() / 2..];
    warm.iter().sum::<f64>() / warm.len() as f64
}

/// **KILL FACTOR 1, AS MUCH OF IT AS IS TRUE.** A padded fire is finite, in
/// bounds, and still the model: sixteen greedy steps off a real checkpoint,
/// with `Knobs::pad` on and the shell's own lattice under it, land the
/// continuation this tree pinned before D4 existed.
///
/// A padded gemm that walked off its column would take the arena's next value
/// with it, and every one of those is upstream of the logits — so the pin is a
/// bounds check with a vocabulary attached.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_padded_fire_stays_in_bounds_and_says_the_pinned_thing() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = load(true, None) else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let (said, millis) = run(&mut shell, &prompt, 1);
    let text = tokenizer.decode(&said[0], false);
    eprintln!("padded: {text:?} at {:.2} ms/fire", warm(&millis));
    assert!(
        text.starts_with(EXPECTED),
        "a padded fire's greedy continuation of {PROMPT:?} began {text:?}, and \
         this shell answers {EXPECTED:?} when its gemms are in bounds"
    );
}

/// **KILL FACTOR 2: WHAT THE TAIL COSTS.** The default lattice's floor is
/// eight, so a one-lane decode computes eight rows and a three-lane one
/// computes eight — 8x and 2.7x the rows, through every Always projection.
/// The design's claim is that the linear layers are weight-bound at decode
/// scale (1.40 GiB of weight reads against a handful of activation rows), so
/// those rows ride reads that were happening anyway; this is that claim with
/// two numbers under it.
///
/// The assertion is deliberately loose. This is a MEASUREMENT with a floor
/// under it, not a benchmark: it fails only if the padded arm costs so much
/// more than the unpadded one that the weight-bound argument is wrong by an
/// order of magnitude, which is the shape of a regression (a bucket rounding
/// a decode fire up to a prefill-sized gemm) rather than of noise.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn the_tail_a_padded_decode_computes_rides_the_weight_reads() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = load(true, None) else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let padded: Vec<(u32, f64)> = [1u32, 3]
        .iter()
        .map(|lanes| (*lanes, warm(&run(&mut shell, &prompt, *lanes).1)))
        .collect();
    drop(shell);

    let Some((mut shell, _)) = load(false, None) else {
        return;
    };
    let plain: Vec<(u32, f64)> = [1u32, 3]
        .iter()
        .map(|lanes| (*lanes, warm(&run(&mut shell, &prompt, *lanes).1)))
        .collect();

    for ((lanes, on), (_, off)) in padded.iter().zip(plain.iter()) {
        eprintln!(
            "{lanes} decode lane(s), {lanes} rows -> bucket 8: padded {on:.3} ms/fire, \
             unpadded {off:.3} ms/fire ({:+.1}%)",
            (on / off - 1.0) * 100.0
        );
        assert!(
            *on < off * 2.0,
            "padding {lanes} rows to eight cost {on:.3} ms against {off:.3} — the \
             tail is not riding the weight reads and the lattice is the wrong shape"
        );
    }
}

/// **THE DETERMINISM PADDING ACTUALLY BUYS**, stated as the test the false
/// version of kill factor 1 should have been.
///
/// A lane fired alone and the same lane fired beside two others run different
/// gemms today — `M=1` takes the gemv arm, `M=3` takes cuBLAS — and agree only
/// where the library's arms happen to. Under a lattice with ONE point above
/// every row count these fires reach, both are `M=16`, the arm is the same
/// arm, and the agreement stops being an accident.
///
/// This is also the census-freeze mechanism (`.wiki/palo/cuda-abi.md` §3's
/// kill factor 3) observed from the outside: what freezes inside a bucket is
/// not only the kernel NAME the probe counts but the numbers it computes.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_solo_lane_and_a_batched_one_agree_inside_one_bucket() {
    let _serial = serialized();
    // Sixteen holds every fire below — a 5-token prompt on three lanes is 15
    // rows, and every decode after it is 3 — so the whole run is one bucket.
    let Some((mut shell, tokenizer)) = load(true, Some(&[16, 256])) else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    assert!(
        prompt.len() * 3 <= 16,
        "this gate needs every fire it runs inside one bucket, and the prompt \
         grew to {} tokens",
        prompt.len()
    );

    let (solo, _) = run(&mut shell, &prompt, 1);
    let (batched, _) = run(&mut shell, &prompt, 3);
    for (lane, said) in batched.iter().enumerate() {
        assert_eq!(
            *said, solo[0],
            "lane {lane} of a three-lane fire said {:?} and the same lane alone \
             said {:?}, inside one bucket where both run the same arm",
            tokenizer.decode(said, false),
            tokenizer.decode(&solo[0], false),
        );
    }
    eprintln!(
        "one bucket, three lanes and one lane: {:?}",
        tokenizer.decode(&solo[0], false)
    );
}
