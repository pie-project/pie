//! **THE STREAMED PATH, OVER A REAL 2-BIT ARTIFACT.** Every piece of wave
//! W-a/W-b was gated, and every gate but this one was fired over a synthetic
//! bf16 fixture. This file is the composition: `mlx-community/
//! DeepSeek-V4-Flash-2bit-DQ`'s `mini-l5-e16` snapshot — the same 1.57 GiB of
//! mixed-group `MlxU2G{32,64}` the first light reads — loaded onto a device
//! weight budget too small to hold it, so its FIFTEEN routed 2-bit expert
//! banks (five layers, three halves, three planes each) are seated a fraction
//! at a time out of a file-backed host mapping.
//!
//! `routed_experts_stream` is the sibling that owns the mechanism; it proves
//! parity, motion and the `fstat` door over `a3b_micro`, a bf16 text with one
//! plane per bank and one router kind. This file changes three things at once
//! and only asks what those three change:
//!
//! ```text
//! (1) the bank is a TRIPLET, not a plane — codes at 2 bits, scales and
//!     biases beside them, and the seat is all three or it is somebody else's
//!     expert dequantized against this one's centre
//! (2) the groups are MIXED — `gate_proj` at group 32 on layers 0-3 and at
//!     group 64 on the last, `up`/`down` at 64 throughout — so three bands of
//!     one group carry three different seat strides, and the layer-4 gate
//!     carries a fourth
//! (3) the routers are TWO KINDS — layers 0-2 route by `linear.moe_hash_route`
//!     off a `[vocab, top_k]` table and layers 3-4 by
//!     `linear.moe_topk_sqrt_softplus` — and a residency mechanism that knows
//!     only one of them holds half a model
//! ```
//!
//! # What this composition found, and it was (3)
//!
//! **A DEEPSEEK-V4-FLASH LOAD COULD NOT STREAM AT ALL, AND THEN IT STREAMED
//! WRONG.** Two faults, one cause, and the second was hiding behind the first:
//!
//! 1. **`experts::Plan::of` refused the model.** The router scan matched the
//!    four ranked `Linear::MoeTopk*` arms and not `Linear::MoeHashRoute`, so
//!    the hash-routed layers' selects were "routed reads whose router this
//!    plan does not state" and every budget under full residency was a
//!    refusal. `scratch::routers` had the same blind spot, where its cost was
//!    quieter: the expert count read 0 for those layers and the sorted-MoE arm
//!    reserved nothing for them, so three fifths of this model's mixtures took
//!    the matvec arm whatever the rectangle. That one is worth a number —
//!    `what_a_two_bit_prefill_costs` reads dsv4's 512-token prefill at 555
//!    tok/s before and 768 after, which is a 1.38x nobody was looking for.
//!
//!    The fix is the expert count on the router that states it:
//!    `MoeHashRoute` now carries `experts`, as its four ranked siblings do,
//!    and the two scans read it. The kernel does not — the table names ids
//!    outright — which is exactly why nobody noticed the field was missing.
//!
//! 2. **And then the cut fell on two of five mixtures.** The segment boundary
//!    is triggered at ENCODE by a shader-point name, and the name was one
//!    prefix: `router_topk`. The hash router's point is `hash_route_gather`.
//!    So the three hash-routed groups were planned, seated, and never cut —
//!    the tier never rewrote their routing vectors from EXPERT id to SEAT
//!    index, and the selects behind them indexed an eight-seat slab with ids
//!    in `0..16`. Not a refusal: a fire that completes, returns finite spread
//!    logits, and has read another band's bytes for three of five layers.
//!
//! This is the fault this file exists to have caught, and the shape of it is
//! worth naming: the plan is derived from the IR (`experts::cuts` found all
//! five routers correctly, before and after) and the trigger is derived from a
//! KERNEL NAME, and nothing held the two together. What holds them together
//! now is the motion assert below — `segments == fires · groups`, exactly the
//! sibling's assert — which is a claim no name-matched trigger can satisfy by
//! accident. It was already written in `routed_experts_stream`; it had simply
//! never been asked of a model with two router kinds.
//!
//! # Every fire is ONE TOKEN wide, and the wide one is a gate of its own
//!
//! The sibling's header gives the rule: every distinct expert one SEGMENT
//! routes to must be seated at once, because the segment's matmuls all run
//! behind its cut — so a slab under a row's `top_k` cannot serve even a
//! one-token fire, which is why the budget below is sized by a SEAT COUNT and
//! not by a fraction of the bytes (`seating` argues it). dsv4 routes at
//! `top_k` 6 over 16 experts, and the
//! first light's eight-token prompt reaches all sixteen at the hash gate — so
//! there is no streaming budget that serves that prefill, and
//! [`a_wide_prefill_over_a_small_slab_is_refused_by_name`] asserts that it is
//! REFUSED, by the hash layer's own name and with both numbers, rather than
//! served wrong. The parity claim is therefore fired one token at a time,
//! which is the same shape the sibling fires and for the same reason.
//!
//! # And a second row, for the half dsv4 cannot exercise
//!
//! [`a_streamed_fused_two_bit_bank_says_what_a_resident_one_says`] fires the
//! same parity over `qwen38-flash-mlxu2` — a FUSED `experts_gate_up` bank at a
//! uniform `MlxU2G128`, four layers, top-k 10, and only the ranked router. It
//! is a seat-arithmetic claim and NOT a second footprint claim: 94% of that
//! artifact's 4.49 GiB is `ple.table`, `embed` and `lm_head`, which no tier may
//! hold less of, so its slab holds back half a percent where dsv4's holds back
//! twenty. [`QWEN4`] carries that sentence in full, and design §7's answer to
//! it — the static demand shape, dense overflow on a compiler-emitted prefetch
//! schedule — is not built.
//!
//! # Gating
//!
//! Apple-only at compile time, and SKIPS at run time naming which precondition
//! was missing. `PIE_U2_SNAPSHOT` and `PIE_QWEN4_U2_SNAPSHOT` override where
//! the two snapshots are looked for.
//!
//! ```text
//! cargo test -p engine-metal --release --test two_bit_experts_stream -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use engine_metal::experts::{Attachments, Plan};
use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};
use model_ir::Trace;

/// **One artifact this file streams**: which catalog row, where its snapshot
/// is, and which family's `Classify` computes its lane word.
///
/// Two rows are named below because the composition has two independent
/// shapes, not because the second is a copy of the first — see
/// [`QWEN4`] for what it covers and what it deliberately does not claim.
struct Artifact {
    sku: &'static str,
    repo: &'static str,
    /// The environment variable that overrides where the snapshot is looked
    /// for, so each row keeps the name its first light already published.
    stated: &'static str,
    /// The lane word, which is the model family's own and not a shell fact.
    word: fn(u32) -> u64,
    /// How many experts of sixteen the slab this file fires seats.
    ///
    /// **BOUNDED ON BOTH SIDES, AND THE LOWER BOUND IS THE ROW'S `top_k`.**
    /// Every distinct expert one SEGMENT routes to must be seated at once, so a
    /// slab under `top_k` cannot serve even a one-token fire — dsv4 routes at 6
    /// and qwen4 at 10, which is why one number could not serve both. Above the
    /// bank's arity there is no slab at all.
    seats: u32,
}

/// **THE MIXED-GROUP SPLIT-BANK ROW**, and the one this file is written for:
/// five layers, sixteen experts, `gate`/`up`/`down` as three separate 2-bit
/// banks at group 32 and 64, and two router kinds.
const DSV4: Artifact = Artifact {
    sku: "dsv4-flash-mlxu2-kv-bf16",
    repo: "models--mlx-community--DeepSeek-V4-Flash-2bit-DQ",
    stated: "PIE_U2_SNAPSHOT",
    word: |len| models::deepseek_v4::forward::Facts::of(&Request::new(len, false)).word(),
    // Half the bank, and comfortably over this row's top-k of 6.
    seats: 8,
};

/// **THE FUSED UNIFORM-GROUP ROW**, which is a different band shape over the
/// same mechanism: four layers, sixteen experts, one FUSED `experts_gate_up`
/// bank beside `experts_down`, uniform `MlxU2G128`, and only the ranked router.
///
/// **AND ITS FOOTPRINT CLAIM IS NOT THE DSV4 ONE.** This artifact's 4.49 GiB is
/// 94% planes no tier may hold less of — `ple.table` and its two companions
/// are 1.86 GiB, `embed` and `lm_head` 1.18 GiB each — against 0.08 GiB of
/// routed bands. Streaming half its experts holds back 0.9% of the table, so
/// this row is NOT evidence that the tier bounds a load's footprint, and the
/// gate below does not pretend it is. What it is evidence of is that the seat
/// arithmetic is right for a FUSED bank at a third group width, which is the
/// half of the mechanism dsv4 cannot exercise. Design §7's answer to the rest
/// is the static demand shape — the dense overflow with a compiler-emitted
/// prefetch schedule — and it is not built.
const QWEN4: Artifact = Artifact {
    sku: "qwen38-flash-mlxu2-kv-bf16",
    repo: "models--Sawfwair--Qwen3.8-Flash-Next-MLX-Mixed-2bit",
    stated: "PIE_QWEN4_U2_SNAPSHOT",
    word: |len| models::qwen_4::forward::Facts::of(&Request::new(len, false)).word(),
    // This row routes at top-k 10 of 16, so the slab has four seats of room
    // between "cannot serve one token" and "is not a slab".
    seats: 12,
};

/// How many greedy decode fires follow the one-token prefill.
///
/// Long enough that the routing wanders off the identity prefix the slab was
/// opened at and the clock has to evict — which is what the motion assert
/// reads, and what a seat bookkeeping that agreed with nothing would fail.
const STEPS: usize = 8;

/// The token every parity fire starts from, and the ones it feeds back. One
/// token, for the reason the header gives; the id is the first token of the
/// first light's own prompt so that the two files enter the same tower at the
/// same place.
const PROMPT: &str = "The";

/// The first light's prompt, which is the WIDE fire — eight tokens, six
/// experts each, sixteen experts in the bank.
const WIDE: &str = "The capital of France is the city of";

/// One shell at a time per process: two whole-model loads and the wired
/// numbers are only readable one at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The snapshot: the checkpoint AND the tokenizer beside it. The artifact's
/// own environment variable overrides where it is looked for.
fn snapshot(of: &Artifact) -> Option<PathBuf> {
    if let Ok(stated) = std::env::var(of.stated) {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let usable = |path: &Path| path.join("tokenizer.json").exists() && !shards(path).is_empty();
    // The suite runs as root over tailscale ssh, so `HOME` is not always the
    // owner's — the cache is named explicitly beside it.
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
    homes.iter().find_map(|home| {
        let snapshots = Path::new(home)
            .join(".cache/huggingface/hub")
            .join(of.repo)
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

/// **EVERY shard**, and not the first one. The qwen4 miniature spans two files
/// with the n-gram table across the seam, so one file's header is not the
/// artifact and an import fitted against it is refused by name.
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

/// **THE WIRED FOOTPRINT, AS THE KERNEL REPORTS IT**, in bytes — the first
/// light's reader, for its reason: what the shell can tell us is how many
/// bytes it asked for, and what a streamed load raises is how many the KERNEL
/// then wired. `None` where `vm_stat` is not readable, and a missing number is
/// printed as missing rather than as zero.
fn wired() -> Option<u64> {
    let out = std::process::Command::new("vm_stat").output().ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
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

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1 << 30) as f64
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
    let spread = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-3,
        "{what} logits span {spread}, which is a rectangle nothing wrote"
    );
}

/// Everything both arms need, read once: the trace, the fitted contract, the
/// load plan's pairing, and the tokenizer.
struct Fixture {
    of: &'static Artifact,
    trace: Trace,
    contract: checkpoint::contract::ModelContract,
    snapshot: PathBuf,
    planes: Attachments,
    tokenizer: tokenizer::Tokenizer,
}

fn fixture(of: &'static Artifact, what: &str) -> Option<Fixture> {
    let Some(snapshot) = snapshot(of) else {
        eprintln!(
            "skipping {what}: no {} snapshot with a tokenizer beside it under \
             $HOME/.cache/huggingface/hub — name one in {}",
            of.repo, of.stated
        );
        return None;
    };
    let files = shards(&snapshot);
    assert!(!files.is_empty(), "the snapshot holds tensor shards");
    let tokenizer = tokenizer::Tokenizer::from_file(&snapshot.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");

    let sku = of.sku;
    let trace = models::trace_of(sku).expect("the catalog ships the 2-bit SKU")(Platform::Metal);
    let source = ztensor_compat::index_all(&files).expect("the shards open as one source");
    let contract = models::import_of(sku).expect("the catalog ships an import for the SKU")(&source)
        .expect("the 2-bit SKU's import contract fits the real checkpoint");
    drop(source);

    // **THE PAIRING IS READ, NOT DERIVED** (`experts::Plan::of` argues it):
    // a 2-bit bank's scales and zero points are part of an expert's seat, and
    // this is the door `api.rs` opens to learn which planes those are. It is
    // the whole of what makes this fixture different from the sibling's, whose
    // honest map is empty.
    let planes = engine_metal::weights::attachments(&trace, &contract, &snapshot)
        .expect("the load plan pairs this artifact's quantized banks");
    Some(Fixture {
        of,
        trace,
        contract,
        snapshot,
        planes,
        tokenizer,
    })
}

/// What the whole table demands on the device.
fn full_demand(fixture: &Fixture) -> u64 {
    Plan::of(&fixture.trace, &fixture.planes, None)
        .expect("an uncapped 2-bit plan is full residency")
        .device_demand()
}

/// **How many bytes of this trace are routed bands**, read off a plan that had
/// to form one: a budget one byte under full residency is the smallest ask
/// that still streams, and its bands state the arity and the stride each whole
/// band is the product of.
fn band_bytes(fixture: &Fixture) -> u64 {
    Plan::of(
        &fixture.trace,
        &fixture.planes,
        Some(full_demand(fixture) - 1),
    )
    .expect("one byte under full residency streams")
    .bands()
    .iter()
    .map(|band| u64::from(band.experts) * band.stride)
    .sum()
}

/// **The streamed plan this file fires**: every dense plane whole, and a slab
/// of exactly [`Artifact::seats`] experts a group.
///
/// **SIZED BY THE SEAT COUNT AND NOT BY A FRACTION OF THE BYTES**, because the
/// two are not the same question and only one of them is answerable in advance.
/// The floor a fire needs is a count — `top_k` seats pinned at once — while the
/// budget is bytes, and the bytes per seat differ by an order of magnitude
/// between these two rows: dsv4's bands are 40% of its table and qwen4's are
/// 1.8% of its, so "half the band bytes" seats 8 in one and 8 in the other by
/// coincidence and serves only the first. `Plan::of`'s `slots` is monotone in
/// the budget, so the budget that seats a stated count is bisected for, and the
/// plan is asserted to seat exactly it rather than approximately.
fn seating(fixture: &Fixture, want: u32) -> Plan {
    let full = full_demand(fixture);
    let plan_at = |budget: u64| Plan::of(&fixture.trace, &fixture.planes, Some(budget));
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
    assert_eq!(
        plan.slots(),
        want,
        "the smallest budget that seats {want} seats {} instead, so the seat count \
         this file fires is not the one it states",
        plan.slots()
    );
    assert!(
        plan.streams(),
        "a plan that seats {want} of a bank and holds nothing back is not a slab"
    );
    plan
}

/// The plan this file's parity arms fire, for this artifact.
fn half(fixture: &Fixture) -> Plan {
    seating(fixture, fixture.of.seats)
}

/// A shell over the artifact, at a stated residency.
fn load(fixture: &Fixture, residency: Plan) -> engine_metal::Result<Shell> {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.snapshot,
        tp_size: 1,
        precision: models::precision_of(fixture.of.sku)
            .expect("the catalog states this row's precision")
            .to_string(),
        // The first light's budgets: a miniature on a 32 GiB box, four lanes
        // and a short context.
        budget: Budget::new(4, 512),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        // F1: the streamed arm collapses to this anyway (the segment cuts
        // block), so the golden is fired at the same depth its comparand can
        // reach — a parity test at two depths would measure two things.
        runahead: engine::runahead::Runahead::F1,
        residency,
    })
}

/// A one-token prefill and [`STEPS`] greedy decodes, feeding the argmax back.
/// Every fire one token wide, for the reason the header gives.
fn run(fixture: &Fixture, shell: &mut Shell, slot: u32, first: &[u32]) -> (Vec<u32>, Vec<Vec<f32>>) {
    let word = fixture.of.word;
    shell.open(slot).expect("the slot opens");
    let mut rows = Vec::with_capacity(STEPS + 1);
    let prefill = shell
        .fire(&[Lane {
            slot,
            word: word(first.len() as u32),
            tokens: first,
        }])
        .expect("the prefill fires");
    finite(&prefill[0], "prefill");
    let mut produced = vec![argmax(&prefill[0])];
    rows.push(prefill[0].clone());
    for step in 0..STEPS {
        let fed = [*produced.last().expect("a step feeds the last token back")];
        let decode = shell
            .fire(&[Lane {
                slot,
                word: word(1),
                tokens: &fed,
            }])
            .unwrap_or_else(|why| panic!("decode step {step} fires: {why}"));
        finite(&decode[0], &format!("decode step {step}"));
        produced.push(argmax(&decode[0]));
        rows.push(decode[0].clone());
    }
    (produced, rows)
}

fn ready(what: &str) -> bool {
    if engine_metal::device::present() {
        return true;
    }
    eprintln!("skipping {what}: this machine publishes no Metal device");
    false
}

// ── the plan, off the artifact's own pairing ─────────────────────────────

/// **THE SLAB IS CUT OVER TRIPLETS AT MIXED GROUPS**, and this is the gate
/// that says so before a device is asked for anything.
///
/// Needs no device — it is the trace, the contract and the artifact's header —
/// so it runs wherever the snapshot is.
///
/// Three claims the sibling's bf16 fixture cannot make:
///
/// 1. every routed band of every layer is in the plan, and a band is a
///    TRIPLET: 5 layers × 3 halves × 3 planes = 45, not 15. A plan that
///    seated the codes and left the scales resident would move an expert's
///    bits and dequantize them against another expert's centre, and it would
///    look exactly like a correct plan from `slots()`;
/// 2. the seat strides are NOT uniform across a group — group 32 doubles a
///    gate half's scales and biases against group 64 — so the plan's own
///    per-band `stride` is what the tier must copy by, and the layer-4
///    landmine is where the gate half changes its mind;
/// 3. the same contract lowers through the ARENALESS pipeline, which is what
///    a streamed load runs (`weights::Weights::resident` takes that branch on
///    `plan.streams()`).
#[test]
fn a_two_bit_load_plans_a_slab_over_triplets_at_mixed_groups() {
    let Some(fixture) = fixture(&DSV4, "the 2-bit residency plan") else {
        return;
    };
    let full = full_demand(&fixture);
    let bands = band_bytes(&fixture);
    let plan = half(&fixture);

    // `seating` proved `streams()` already; what is worth a line here is that
    // the bands are a REAL fraction of this row's table, which is the property
    // that makes dsv4 the artifact this file's footprint claim is made on.
    assert!(
        bands * 4 > full,
        "this row's routed bands are {bands} of a {full}-byte table, under a quarter \
         of it — a tier over them bounds nothing, and the claim below would be about \
         a rounding error"
    );
    assert_eq!(
        plan.host_demand(),
        0,
        "unified memory has no second tier to demand of"
    );

    // ── (1) FIVE GROUPS, FORTY-FIVE BANDS. One group per mixture layer, and
    //    nine bands in each: `{gate,up,down}` × `{codes, scales, biases}`.
    let groups = plan.groups();
    assert_eq!(
        groups.len(),
        5,
        "this snapshot is five layers and every one of them is a mixture"
    );
    for group in groups {
        assert_eq!(group.experts, 16, "sixteen routed experts a layer");
        assert_eq!(
            group.bands.len(),
            9,
            "three halves of three planes each — a 2-bit seat is the codes AND \
             the scales AND the zero points, and a plan that seated fewer would \
             dequantize one expert's bits against another's centre"
        );
    }
    let seated = plan.slots();
    assert!(
        seated > 0 && seated < 16,
        "the slab seats {seated} of 16, which is neither a fraction nor whole"
    );
    // A one-token fire pins `top_k` seats at once — see the header. Stated
    // here so that a budget change that broke it fails on this line rather
    // than inside a fire.
    assert!(
        seated >= 6,
        "dsv4-flash routes at top-k 6 and the slab seats {seated}; one segment \
         cannot pin more seats than it has"
    );

    // Every routed plane the first light's census names is a band here, by
    // name — the codes and both companions, on all five layers.
    let named: BTreeSet<&str> = plan.bands().iter().map(|b| b.name.as_str()).collect();
    for layer in 0..5 {
        for half in ["gate", "up", "down"] {
            for plane in ["", ".scales", ".biases"] {
                let want = format!("layer.{layer}.experts_{half}{plane}");
                assert!(
                    named.contains(want.as_str()),
                    "`{want}` is a routed 2-bit plane and the slab does not seat it; \
                     a companion left resident is an expert's centre read off \
                     whichever expert happens to sit in that seat"
                );
            }
        }
    }
    assert_eq!(named.len(), 45, "and nothing else is banded");

    // ── (2) THE STRIDES, AND THE LANDMINE IN THEM. A group-32 gate half
    //    carries twice the scale and zero-point bytes per expert that a
    //    group-64 one does, and the codes are the same width either way — so
    //    the gate companions on layers 0-3 are exactly twice the layer-4 ones,
    //    and `up`/`down` never move. Read off the plan's own per-band stride,
    //    which is what the tier copies by.
    let stride = |name: &str| -> u64 {
        plan.bands()
            .iter()
            .find(|b| b.name == name)
            .unwrap_or_else(|| panic!("`{name}` is a band"))
            .stride
    };
    for plane in [".scales", ".biases"] {
        let landmine = stride(&format!("layer.4.experts_gate{plane}"));
        for layer in 0..4 {
            let grouped32 = stride(&format!("layer.{layer}.experts_gate{plane}"));
            assert_eq!(
                grouped32,
                landmine * 2,
                "layer {layer}'s gate `{plane}` seat is {grouped32} bytes and layer 4's \
                 is {landmine}; the gate half groups by 32 on layers 0-3 and by 64 on \
                 the last, so the companions are exactly twice as wide there and a \
                 tier that copied one stride for the group would move the wrong bytes \
                 on four layers of five"
            );
        }
        // And the halves that do NOT move say so, on the same line.
        for half in ["up", "down"] {
            let at0 = stride(&format!("layer.0.experts_{half}{plane}"));
            let at4 = stride(&format!("layer.4.experts_{half}{plane}"));
            assert_eq!(
                at0, at4,
                "`{half}` is group 64 throughout and its `{plane}` seat may not change"
            );
        }
    }
    // The codes are 2-bit on every one of them and the group is no part of
    // their width, so every layer's gate codes are one number.
    let codes: BTreeSet<u64> = (0..5)
        .map(|l| stride(&format!("layer.{l}.experts_gate")))
        .collect();
    assert_eq!(
        codes.len(),
        1,
        "the gate codes are 2 bits on every layer and the GROUP is a property of \
         the companions, so the code seat may not move: {codes:?}"
    );

    // ── (3) THE ARENALESS LOWERING. The same contract against the same Metal
    //    target, minus the two passes that exist to serve an arena — which is
    //    the branch `Weights::resident` takes when the plan streams. Compiled
    //    here so that an artifact the arenaless pipeline could not lower fails
    //    on this line rather than inside a load; its own `arena_bytes` is not
    //    the saving and is not read.
    let metadata = checkpoint::file::read::parse_metadata(&fixture.snapshot)
        .expect("the snapshot's metadata parses");
    let target =
        checkpoint::plan::StorageTarget::for_backend(checkpoint::types::BackendKind::Metal, 0, 1);
    let arena = checkpoint::plan::compile(&metadata, &fixture.contract, target.clone())
        .expect("the 2-bit artifact compiles a load plan")
        .memory
        .arena_bytes();
    checkpoint::plan::compile_streaming(&metadata, &fixture.contract, target)
        .expect("and compiles an arenaless one for the same target");
    assert!(
        arena > 0,
        "the arena arm plans no host image at all, so the streamed arm's saving is \
         unmeasurable and this artifact is not the one that measures it"
    );

    eprintln!(
        "dsv4-flash-mlxu2: {:.2} GiB whole, {:.2} GiB planned — {seated} of 16 experts \
         per group over {} groups, {:.2} GiB of source bytes behind them, and {:.2} GiB \
         of host image the arena arm would have taken and the streamed arm does not",
        gib(full),
        gib(plan.device_demand()),
        groups.len(),
        gib(plan.source_bytes()),
        gib(arena),
    );
}

// ── the composition, on the card ─────────────────────────────────────────

/// **THE CLAIM**, asked of one artifact: a real 2-bit MoE checkpoint, seated a
/// fraction of its experts at a time out of a file-backed mapping, answers BIT
/// FOR BIT what the same checkpoint answers with its whole table resident.
///
/// This is the sibling's claim (a) over an artifact that changes what the
/// header lists, plus (b) its motion — which is the assert that catches a
/// router whose segment is never cut — plus (b') the `fstat` door, which is
/// W-b's honest half.
fn parity(of: &'static Artifact, what: &str) {
    let _one = serialized();
    if !ready(what) {
        return;
    }
    let Some(fixture) = fixture(of, what) else {
        return;
    };
    let first = fixture.tokenizer.encode(PROMPT);
    assert!(
        !first.is_empty(),
        "the prompt encodes to no tokens, and a prefill of nothing proves nothing"
    );

    // ── THE GOLDEN. Uncapped: the whole table on the device, no tier, one
    //    command buffer per fire — the load this row's first light fires.
    let base = wired();
    let mut resident = load(&fixture, Plan::default()).expect("the resident 2-bit shell loads");
    assert!(
        resident.weights_resident(),
        "an uncapped load holds the whole table"
    );
    assert!(
        resident.expert_residency().is_empty(),
        "and opens no tier to report on"
    );
    assert_eq!(
        resident.expert_motion(),
        (0, 0),
        "and cuts no segment and copies no seat"
    );
    let (resident_tokens, golden) = run(&fixture, &mut resident, 0, &first);
    let (resident_weights, resident_arena, ..) = resident.footprint();
    let after_resident = wired();
    drop(resident);

    // ── THE STREAMED LOAD.
    let plan = half(&fixture);
    let seated = plan.slots();
    let source = plan.source_bytes();
    let planned = plan.device_demand();
    let plan_groups = plan.groups().len();
    let plan_bands = plan.bands().len();

    let before = wired();
    let booted = Instant::now();
    let mut streamed = load(&fixture, plan).expect("the streamed 2-bit shell loads");
    let wall = booted.elapsed().as_secs_f64();
    let loaded = wired();

    assert!(
        !streamed.weights_resident(),
        "a streamed load says so rather than claiming the table"
    );
    let groups = streamed.expert_residency();
    assert_eq!(
        groups.len(),
        plan_groups,
        "one slab per mixture — a tier that opened fewer is a mixture reading \
         whatever the store held, which is what an unrecognised router looks like \
         from here"
    );
    let opened: Vec<Vec<Option<u32>>> = groups.iter().map(|g| g.in_seat.clone()).collect();
    for group in &groups {
        assert_eq!(
            group.in_seat.len(),
            seated as usize,
            "`{}` seats what the plan said",
            group.name
        );
        assert!(
            group.in_seat.iter().all(Option::is_some),
            "`{}` opened with an empty seat; the identity prefix is copied in at \
             `Tier::open`",
            group.name
        );
    }

    // ── (b') THE SOURCE IS A MAPPING OF AN UNLINKED FILE OF THE PLANNED SIZE.
    //    Not "could be": `fstat`'s own two numbers — the size, which must be
    //    exactly what the plan planned, and the link count, which must be zero
    //    because the path was removed the instant it was opened. That a
    //    file-backed page is one the kernel MAY reclaim is the property W-b
    //    bought; that it DOES reclaim is the kernel's, and a pressure test on
    //    a 32 GiB box would measure the box. What the fires below add is that
    //    the bytes come back identical after every `HostSource::settle`.
    assert_eq!(
        streamed.expert_source(),
        Some((source, 0)),
        "the streamed source is not a mapping of an unlinked file of the planned \
         size; a source the kernel cannot page out is the term W-b exists to bound"
    );

    let (streamed_tokens, produced) = run(&fixture, &mut streamed, 0, &first);
    let fired = wired();

    // ── (a) THE PARITY, BIT FOR BIT. The arithmetic is the same kernel over
    //    the same 2-bit codes against the same scales and zero points in the
    //    same order; only the seat each expert's TRIPLET was copied into
    //    differs. A tolerance here would be a tolerance for the machinery
    //    having moved the wrong bytes — which, at 2 bits with mixed groups, is
    //    a wrong CENTRE and not a wrong bit, and lands finite spread logits
    //    that no `finite` check can see.
    //
    //    **AND IT COVERS THE ARENALESS LOWERING TOO.** The golden landed
    //    through the executor's arena and this load through its streaming
    //    residency, off a second plan compiled without the two arena passes.
    assert_eq!(golden.len(), produced.len());
    for (step, (want, got)) in golden.iter().zip(&produced).enumerate() {
        assert_eq!(
            want.len(),
            got.len(),
            "step {step}: two readouts of one vocabulary"
        );
        let differ = want
            .iter()
            .zip(got)
            .enumerate()
            .find(|(_, (a, b))| a.to_bits() != b.to_bits());
        assert!(
            differ.is_none(),
            "step {step}: the streamed load's logits differ from the resident load's at \
             column {:?} — residency changed a number, which is the one thing it may \
             never do",
            differ.map(|(at, (a, b))| (at, *a, *b)),
        );
    }
    // And the sentence a reader wants said in the vocabulary they asked it in.
    assert_eq!(
        resident_tokens, streamed_tokens,
        "the streamed load produced different tokens; bit parity above should have \
         caught this first, and one of the two claims is lying"
    );

    // ── (b) THE MECHANISM MOVED — AND THIS IS THE ASSERT THAT CAUGHT THE
    //    UNCUT HASH ROUTER. One cut per mixture per fire is what a streamed
    //    fire owes, and a trigger that recognised only `router_topk` delivered
    //    two of five. Seats were copied, and the occupancy is no longer the
    //    identity prefix the tier opened at.
    let (swaps, segments) = streamed.expert_motion();
    let after = streamed.expert_residency();
    let moved = after
        .iter()
        .zip(&opened)
        .any(|(group, was)| &group.in_seat != was);
    let fires = (STEPS + 1) as u64;
    assert_eq!(
        segments,
        fires * groups.len() as u64,
        "{segments} segment cuts over {fires} fires of {} mixtures — a streamed fire \
         cuts ONCE PER MIXTURE, and a mixture whose router this shell does not \
         recognise is cut zero times, seated never, and read at an expert id the slab \
         has no seat for",
        groups.len()
    );
    assert!(
        swaps > 0,
        "{segments} segments and not one band was copied; the tier is opened and the \
         swap is not connected"
    );
    assert!(
        moved,
        "seats were copied and no group's occupancy changed; the bookkeeping and the \
         copies are not the same seats"
    );

    // ── (4) THE NUMBERS, OBSERVED AND PRINTED. Not pinned: a wired delta is a
    //    fact about this box at this instant, and the claim above is the one
    //    that must hold everywhere.
    let (weights, arena, pools, inputs) = streamed.footprint();
    let whole = full_demand(&fixture);
    eprintln!(
        "streamed {} on {} in {wall:.1}s\n  \
         slab: {seated} of {} experts per group over {plan_groups} groups, {plan_bands} \
         bands, {:.2} GiB planned against {:.2} GiB whole ({:.2} GiB held back, {:.1}% \
         of the table)\n  \
         store: weights {:.2} GiB (resident arm {:.2} GiB), arena {:.1} MiB (resident \
         arm {:.1} MiB), pools {:.1} MiB, inputs {:.1} MiB\n  \
         host source: {:.2} GiB, file-backed and unlinked\n  \
         motion: {segments} segments cut, {swaps} bands copied, occupancy moved {moved}\n  \
         tokens: {resident_tokens:?}",
        fixture.of.sku,
        streamed.device_name(),
        groups[0].experts,
        gib(planned),
        gib(whole),
        gib(whole - planned),
        100.0 * (whole - planned) as f64 / whole as f64,
        gib(weights),
        gib(resident_weights),
        arena as f64 / (1 << 20) as f64,
        resident_arena as f64 / (1 << 20) as f64,
        pools as f64 / (1 << 20) as f64,
        inputs as f64 / (1 << 20) as f64,
        gib(source),
    );
    match (base, after_resident, before, loaded, fired) {
        (Some(base), Some(gold), Some(before), Some(loaded), Some(fired)) => eprintln!(
            "wired down: {:.2} GiB at rest -> {:.2} GiB with the resident arm loaded and \
             fired (delta {:+.2}); {:.2} GiB after it was dropped -> {:.2} GiB streamed \
             load (delta {:+.2}) -> {:.2} GiB after {fires} fires (delta {:+.2} over the \
             load, {:+.2} over rest)",
            gib(base),
            gib(gold),
            gib(gold) - gib(base),
            gib(before),
            gib(loaded),
            gib(loaded) - gib(before),
            gib(fired),
            gib(fired) - gib(loaded),
            gib(fired) - gib(base),
        ),
        _ => eprintln!("wired down: not readable on this box"),
    }
}

/// **THE GATE THIS FILE EXISTS FOR**: the mixed-group split-bank row, whose
/// three routed halves seat at three strides and whose five mixtures are
/// decided by two different router kinds.
#[test]
fn a_streamed_two_bit_load_says_what_a_resident_one_says() {
    parity(&DSV4, "the 2-bit streaming parity gate");
}

/// **THE SAME MECHANISM OVER A FUSED BANK AT A THIRD GROUP WIDTH.**
///
/// `qwen38-flash-mlxu2` is not a second footprint claim and [`QWEN4`] says why
/// — 94% of its table is `ple.table`, `embed` and `lm_head`, which no tier may
/// hold less of, so streaming half its experts holds back under a percent. It
/// is a second SEAT-ARITHMETIC claim, and that is worth its own load: the bank
/// is FUSED (`experts_gate_up`, one rectangle where dsv4 declares two), the
/// point is a uniform `MlxU2G128` where dsv4 mixes 32 and 64, and the only
/// router is the ranked one. A seat stride derived from a 4-bit-era assumption
/// about how a group divides a row fails here and passes there, or the reverse.
#[test]
fn a_streamed_fused_two_bit_bank_says_what_a_resident_one_says() {
    parity(&QWEN4, "the fused 2-bit streaming parity gate");
}

/// **THE BOUNDARY, NAMED RATHER THAN SERVED WRONG.** The first light's
/// eight-token prompt routes to more distinct experts than any streamed slab
/// of this artifact can seat, so it is refused — and the refusal names the
/// layer, both numbers, and the two levers.
///
/// Two things ride on this beyond the refusal itself. It is the fire that
/// proves the HASH router is cut at all: the refusal is raised inside
/// `Tier::segment`, and before the segment trigger learned
/// `hash_route_gather`, this same fire completed and returned finite logits it
/// had read off the wrong bands. And it is the measured reason the parity gate
/// above fires one token at a time rather than the first light's prefill.
#[test]
fn a_wide_prefill_over_a_small_slab_is_refused_by_name() {
    let _one = serialized();
    if !ready("the wide-prefill refusal") {
        return;
    }
    let Some(fixture) = fixture(&DSV4, "the wide-prefill refusal") else {
        return;
    };
    let wide = fixture.tokenizer.encode(WIDE);
    assert!(
        wide.len() >= 3,
        "the wide prompt is {} tokens, which is not wide",
        wide.len()
    );

    let plan = half(&fixture);
    let seated = plan.slots();
    let mut shell = load(&fixture, plan).expect("the streamed 2-bit shell loads");
    shell.open(0).expect("the slot opens");
    let why = shell
        .fire(&[Lane {
            slot: 0,
            word: (fixture.of.word)(wide.len() as u32),
            tokens: &wide,
        }])
        .expect_err(
            "an eight-token prefill at top-k 6 over 16 experts reaches every one of \
             them, and a slab that seats fewer cannot pin them all at once",
        )
        .to_string();

    assert!(
        why.contains("distinct experts") && why.contains(&format!("{seated}")),
        "the refusal names the mechanism and the seat count it has: {why}"
    );
    assert!(
        why.contains("experts_gate") || why.contains("experts_up") || why.contains("experts_down"),
        "the refusal names the band whose slab ran out: {why}"
    );
    assert!(
        why.contains("device_weight_budget") && why.contains("fewer tokens"),
        "the refusal names both levers: {why}"
    );
    // **AND IT IS THE HASH ROUTER'S OWN POINT THAT RAISED IT** on this
    // artifact — layer 0 is one of the three `linear.moe_hash_route` layers,
    // and a shell whose segment trigger did not recognise that point would
    // have served this fire instead of refusing it.
    assert!(
        why.contains("hash_route_gather"),
        "the fire was refused somewhere other than the hash router's own segment cut, \
         which is where a five-layer artifact with three hash-routed mixtures must \
         run out of seats first: {why}"
    );
    eprintln!("the wide prefill over a {seated}-seat slab is refused: {why}");
}
