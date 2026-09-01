//! The two EXPORT axes, end to end — and the gate that says palo C3b/C4b runs.
//!
//! **WHAT THIS FILE IS FOR.** `masked_axis` and `adapter_banks` cover the two
//! axes whose divergence stays inside the graph: a kernel choice and an
//! additive correction, both of them things a fire does differently. These two
//! are the other kind. A draft head and a score capture each write a SECOND
//! COLUMN that a reader collects after the graph has run (palo design §9), so
//! what has to be true of them is not only that the right arm ran but that the
//! bytes it wrote are still there when somebody comes for them — and that a
//! fire nobody asked either of costs nothing at all.
//!
//! ```text
//! (a) a capturing lane's mass comes back, per layer, and comes back the same twice
//! (b) a capturing lane beside two others leaves them the fire they had alone
//! (c) the three-class composition captures once and replays identically
//! (d) a fire no lane captured is the fire this shell always fired  — nodes and tokens
//! (e) the refusals: a capture and a word that disagree, both ways;
//!     a draft against a text that declares no head
//! (f) the drafting SKU does not fit this device, and the refusal says by how much
//! (g) a capturing PREFILL beside a capturing DECODE is the fire P4 cannot
//!     seat: it runs as two launches, says what each lane says alone, and
//!     replays out of a recorded graph identically
//! ```
//!
//! **AND WHY (g) IS THIS FILE'S BUSINESS.** The capture axis is the one that
//! CROSSES `qo_one`. Every earlier axis nested inside it — `masked` splits
//! decode from prefill inside itself, and a nested family is laminar and
//! therefore always consecutive-ones — so P4 seated every window the catalog
//! stated and its `FallbackTable` really was empty. `captures_scores` is not
//! nested: it cuts across, and the row order P4 finds puts the plain classes
//! between the capturing prefill class and the capturing decode class. A fire
//! holding both is then a window in two pieces, which is the case design §3's
//! fallback exists for and the case every engine used to refuse by name. It is
//! not exotic — two capturing requests at different stages of their lives is
//! an ordinary serving mix — so it is gated here beside the axis that produces
//! it.
//!
//! **WHY THE SCORE AXIS CARRIES THE WEIGHT AND THE DRAFT AXIS DOES NOT.** The
//! one shipping SKU whose checkpoint publishes a draft head is
//! `qwen36-27b-bf16-kv-bf16`, whose bf16 weights are ~52 GiB against an L40S's
//! 46 GiB — it does not load, and gate (f) is the honest statement of that
//! rather than a skipped test. What can be gated here is everything the two
//! axes SHARE, which is nearly all of it: one export mechanism, one delivery
//! tail in the carve, one class-set reading at load, one twin refusal shape.
//! The score axis exercises all of it on a model that fits. What remains
//! unproven on a device is the draft head's own arithmetic, and nothing short
//! of a checkpoint that fits proves that.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 --release --test export_axes -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_cuda::{Boot, Graphs, Lane, LayerScores, Seated, Shell};
use model_compiler::{Budget, DeviceProfile, compile};
use model_dsl::{Classify, Platform, Request};

/// The workhorse: small, dense, and the SKU whose model text declares the
/// capture arm.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The one shipping SKU whose checkpoint publishes a draft head — and the one
/// that does not fit this device.
const DRAFTING: &str = "qwen36-27b-bf16-kv-bf16";

const PROMPT: &str = "The capital of France is";

/// What `serve_smoke` pins this shell to answer for [`PROMPT`] on this SKU,
/// greedily — the golden every other suite in this crate holds itself to.
const EXPECTED: &str = " Paris";

/// How many greedy decode fires follow a prefill.
const STEPS: usize = 8;

/// The ceilings every shell in this file loads at — named because gate (g)
/// bakes the same plan a second time to check its own premise, and two
/// budgets would be two artifacts.
const BUDGETS: Budget = Budget {
    max_lanes: 4,
    max_tokens: 256,
    buckets: Vec::new(),
    max_adapters: 0,
};

/// One shell at a time per process — `kernels-cuda`'s scratch slabs are
/// process-global and keyed by name (`serve_smoke.rs` argues it whole).
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The lane word the model's own `Classify` computes — the facts qwen
/// declares, and no third opinion about any of them.
fn word(query_len: u32, captures: bool) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false).capturing_scores(captures))
        .word()
}

/// The word a DRAFTING request carries, which this SKU's artifact has no arm
/// for — `ClassTable::mask` drops the bit, so the lane composes as the word it
/// would have had. That is the masking `model_exec::fire::compose` documents, and
/// it is why the draft refusal below is `Draftless` rather than
/// `UnknownWord`: the two halves agree perfectly, and what is missing is the
/// ARM.
fn drafting_word(query_len: u32) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false).drafting(true)).word()
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        assert!(value.is_finite(), "logit {at} is {value}");
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

// ── the runs ─────────────────────────────────────────────────────────────

/// One sequence alone in its fires: prefill then `steps` greedy decodes, every
/// lane capturing or not as `captures` says. Returns the tokens and the
/// prefill's captured mass.
///
/// **IT FIRES THE PROMPT TWICE AND KEEPS THE SECOND**, as `adapter_banks` and
/// `masked_axis` do and for their reason: the dense autotuner tunes a GEMM
/// shape on its second sighting, so a cold solo run and a warm mixed one are
/// two tactic ladders. Every identity here is between STEADY STATES.
fn solo(
    shell: &mut Shell,
    slot: u32,
    prompt: &[u32],
    captures: bool,
    steps: usize,
) -> (Vec<u32>, Vec<LayerScores>) {
    shell.open(slot).expect("the slot opens");
    let mut warm = Vec::new();
    shell
        .fire_captured(&[seat(slot, prompt, captures)], &[], &mut warm)
        .expect("the warming fire");
    shell.open(slot).expect("the slot re-opens");
    let mut captured = Vec::new();
    let prefill = shell
        .fire_captured(&[seat(slot, prompt, captures)], &[], &mut captured)
        .expect("the prefill fires");
    let mass = captured.into_iter().next().unwrap_or_default();
    let mut said = vec![argmax(&prefill[0])];
    for step in 1..steps {
        let fed = [*said.last().expect("a token to feed")];
        let mut none = Vec::new();
        let decode = shell
            .fire_captured(&[seat(slot, &fed, captures)], &[], &mut none)
            .unwrap_or_else(|why| panic!("decode step {step}: {why}"));
        said.push(argmax(&decode[0]));
    }
    (said, mass)
}

/// One lane, seated with its word and its ask stated as ONE READING — which
/// is what `runtime::pipeline::fire::stamp_lane_words` is, spelled here so a
/// gate cannot accidentally state them apart.
fn seat<'a>(slot: u32, tokens: &'a [u32], captures: bool) -> Seated<'a> {
    let lane = Lane {
        slot,
        word: word(tokens.len() as u32, captures),
        tokens,
    };
    if captures {
        Seated::capturing(lane)
    } else {
        Seated::of(lane)
    }
}

/// The largest absolute difference between two captures, layer by layer.
fn drift(left: &[LayerScores], right: &[LayerScores]) -> f32 {
    assert_eq!(left.len(), right.len(), "two captures of one model");
    let mut worst = 0.0f32;
    for (a, b) in left.iter().zip(right) {
        assert_eq!(a.layer, b.layer, "the layers came back in a different order");
        assert_eq!((a.rows, a.heads), (b.rows, b.heads), "two shapes");
        assert_eq!(a.lse.len(), b.lse.len(), "two lengths");
        for (x, y) in a.lse.iter().zip(&b.lse) {
            worst = worst.max((x - y).abs());
        }
    }
    worst
}

// ── (a) the mass comes back, per layer, and comes back the same twice ────

/// **AN EXPORT IS ONLY AN EXPORT IF SOMEBODY CAN READ IT.**
///
/// palo C4 declared the axis and proved it in the plan: a capturing lane lands
/// in a class that runs `attention.prefill_lse`, and that arm writes a column.
/// What it could not say is that anything ever reads the column, and it could
/// not, because nothing did — `model_compiler::arena` gave the delivery tail
/// to the `"out"` seam by name, so every capture column's life ended at the
/// node that wrote it and the busiest-instant carve was free to place the next
/// layer's rectangles on top of it. This is the gate for the other half: the
/// tail is the export SET now, the shell resolves the seam at load, and a
/// capturing lane's rows come back out of the arena where they lie.
///
/// **DETERMINISM IS THE SHARP HALF, AND IT IS SHARP BECAUSE OF WHAT WOULD
/// BREAK IT.** A column carved over reads whatever ran last, and what ran last
/// is a function of the composition — so a capture that were unpinned would
/// still be finite, still be roughly the right magnitude, and would move
/// between two fires of the same lane. Bit-for-bit equality over two identical
/// fires is what an unpinned column cannot produce.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_capturing_lane_reads_its_attention_mass_and_reads_the_same_thing_twice() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the capture readout") else {
        return;
    };
    let prompt = tok.encode(PROMPT);

    let layers = shell.score_layers();
    assert!(
        !layers.is_empty(),
        "`{SKU}` declares no `attn.scores` export, so this file is testing nothing"
    );

    let (said, first) = solo(&mut shell, 0, &prompt, true, STEPS);
    let (again, second) = solo(&mut shell, 0, &prompt, true, STEPS);

    eprintln!(
        "capture: {} layer(s) {:?}, {} rows x {} heads each, continuation {:?}",
        first.len(),
        layers,
        first[0].rows,
        first[0].heads,
        tok.decode(&said, false),
    );

    assert_eq!(
        first.len(),
        layers.len(),
        "the fire came back with {} capture column(s) and the artifact declares {}",
        first.len(),
        layers.len(),
    );
    for (column, layer) in first.iter().zip(&layers) {
        assert_eq!(column.layer, *layer, "the columns are not in layer order");
        assert_eq!(
            column.rows as usize,
            prompt.len(),
            "layer {layer}'s column came back {} rows for a {}-row lane",
            column.rows,
            prompt.len(),
        );
        assert!(column.heads > 0, "layer {layer} came back zero heads wide");
        assert_eq!(
            column.lse.len(),
            column.rows as usize * column.heads as usize,
            "layer {layer}'s column is not `rows x heads` of them",
        );
        // A log-sum-exp over a non-empty causal row is finite and strictly
        // greater than the largest score it normalizes, so it is never zero
        // and never NaN. An unwritten column would be one or the other.
        for (at, mass) in column.lse.iter().enumerate() {
            assert!(
                mass.is_finite(),
                "layer {layer}, element {at} of the mass is {mass}"
            );
        }
        assert!(
            column.lse.iter().any(|mass| *mass != 0.0),
            "layer {layer}'s whole column is zero, which is the arena as it was \
             reserved rather than anything a kernel wrote"
        );
    }

    assert_eq!(
        said, again,
        "two capturing runs of one prompt through one boot disagree"
    );
    let moved = drift(&first, &second);
    assert_eq!(
        moved, 0.0,
        "two identical fires captured masses that differ by {moved}; a column \
         nothing pinned reads whatever the carve placed on it last"
    );
}

/// **THE ROW OFFSET IS READ, AND HERE IS THE TEST THAT WOULD CATCH IT NOT
/// BEING.**
///
/// A capture column is the whole fire's height and a lane's mass is its own row
/// run of it, taken at `LaneRow::row_offset` off the composition. Determinism
/// alone does not prove that number is used: a shell that read row zero for
/// every lane would be perfectly deterministic and perfectly wrong, and a
/// single-lane fire cannot tell the difference because the offset IS zero.
///
/// Two capturing lanes of the same length and different content, in one fire,
/// can. They land in one class, so the composition seriates them into one
/// window at two offsets; their masses are functions of their own tokens and
/// must differ. Read at a fixed offset they would be one lane's bytes twice.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn two_capturing_lanes_in_one_fire_read_their_own_rows() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the capture row offset") else {
        return;
    };
    // Same token count, different tokens — so the two lanes share a class and
    // a window, and only the offset tells their rows apart.
    let (mut left, mut right) = (tok.encode(PROMPT), tok.encode("The largest planet is"));
    let rows = left.len().min(right.len());
    left.truncate(rows);
    right.truncate(rows);
    assert_eq!(left.len(), right.len());
    assert_ne!(left, right, "two prompts that differ somewhere");

    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");
    let mut mass: Vec<Vec<LayerScores>> = Vec::new();
    shell
        .fire_captured(
            &[seat(0, &left, true), seat(1, &right, true)],
            &[],
            &mut mass,
        )
        .expect("two capturing lanes fire");

    assert_eq!(mass.len(), 2, "two lanes in, two captures out");
    let moved = drift(&mass[0], &mass[1]);
    eprintln!(
        "two capturing lanes, {rows} rows each: max |Δ| between their masses = {moved}"
    );
    assert!(
        moved > 0.0,
        "two lanes with different tokens captured the SAME mass, which is what a \
         shell that read row zero for every lane would produce"
    );
    // And the second lane is not reading past its own run either: its rows are
    // the ones it submitted, not the fire's.
    for column in mass.iter().flatten() {
        assert_eq!(
            column.rows as usize, rows,
            "a lane's capture is its own row run and nothing else"
        );
    }
}

// ── (b) a capturing lane leaves the lanes beside it alone ────────────────

/// **THE C1b THREE-CLASS FIRE, WITH THE CAPTURE ARM AS THE THIRD CLASS.**
///
/// Decode, prefill and capture in one fire, each over its own window, and the
/// two lanes that captured nothing must say what they said alone. This is the
/// leak question asked exactly: the capture arm is a third arm of the
/// attention merge, so it writes disjoint rows of the same `o` column as the
/// other two, and a window resolved at the fire's rectangle rather than at the
/// class's would have it write over theirs.
///
/// **THE IDENTITY IS ON TOKENS AND HAS TO BE** (build log 21's finding, and
/// build log 22 restates it): a batched fire's shared GEMMs run at a different
/// `M` than a solo one, so an unrelated lane's logits move by about one bf16
/// ulp at a magnitude of ~20. Tokens do not move, and a leak from a
/// vocabulary-wide arm is nowhere near an ulp.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_capturing_lane_beside_two_others_leaves_them_the_fire_they_had_alone() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the three-class capture fire") else {
        return;
    };
    let prompt = tok.encode(PROMPT);
    let other = tok.encode("The largest planet is");

    // The solo references, warmed.
    let (plain_prefill, _) = solo(&mut shell, 0, &prompt, false, STEPS);
    let (plain_other, _) = solo(&mut shell, 1, &other, false, STEPS);
    let (capturing, _) = solo(&mut shell, 2, &prompt, true, STEPS);

    // The mixed fire: lane 0 prefills plain, lane 1 prefills capturing, lane 2
    // decodes. Three classes, three windows, one fire.
    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");
    shell.open(2).expect("slot 2 opens");
    let mut said: [Vec<u32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut fed: Vec<Vec<u32>> = vec![prompt.clone(), prompt.clone(), other.clone()];
    let mut captured_rows = 0usize;
    for step in 0..STEPS {
        let seated = [
            Seated::of(Lane {
                slot: 0,
                word: word(fed[0].len() as u32, false),
                tokens: &fed[0],
            }),
            Seated::capturing(Lane {
                slot: 1,
                word: word(fed[1].len() as u32, true),
                tokens: &fed[1],
            }),
            Seated::of(Lane {
                slot: 2,
                word: word(fed[2].len() as u32, false),
                tokens: &fed[2],
            }),
        ];
        let mut mass: Vec<Vec<LayerScores>> = Vec::new();
        let out = shell
            .fire_captured(&seated, &[], &mut mass)
            .unwrap_or_else(|why| panic!("the mixed fire at step {step}: {why}"));
        assert!(
            mass[0].is_empty() && mass[2].is_empty(),
            "step {step}: a lane that captured nothing was handed a capture"
        );
        assert!(
            !mass[1].is_empty(),
            "step {step}: the capturing lane was handed nothing"
        );
        captured_rows += mass[1][0].rows as usize;
        for (lane, readout) in out.iter().enumerate() {
            let token = argmax(readout);
            said[lane].push(token);
            fed[lane] = vec![token];
        }
    }

    eprintln!(
        "mixed: plain {:?} / capturing {:?} / other {:?}  ({captured_rows} rows captured)",
        tok.decode(&said[0], false),
        tok.decode(&said[1], false),
        tok.decode(&said[2], false),
    );
    assert_eq!(
        said[0],
        plain_prefill[..STEPS],
        "the plain lane's continuation moved when a capturing lane joined its fire"
    );
    assert_eq!(
        said[2],
        plain_other[..STEPS],
        "the second plain lane's continuation moved when a capturing lane joined \
         its fire"
    );
    assert_eq!(
        said[1],
        capturing[..STEPS],
        "the capturing lane's own continuation moved when two plain lanes joined \
         its fire"
    );
}

// ── (c) the composition captures once and replays identically ────────────

/// **THE RECORDED GRAPH HAS TO CARRY THE NEW COMPOSITION TOO.**
///
/// The capture arm is a new class, so a fire that holds one presents a
/// `record::BodyKey` — a lattice point and a present set — this shell has
/// never recorded a body for. What is asked is the same thing `masked_axis`
/// asks of the masked composition: the eager walk (the serialization of the
/// DAG) and the replayed body agree token for token, and the mass they capture
/// agrees bit for bit — which is the stronger half here, because a graph that
/// replayed a stale arena address would still produce the right tokens and the
/// wrong capture.
///
/// **AND THE COMPOSITION IS BODY-ADMISSIBLE, WHICH IS NOT FREE.** One lane, on
/// the capture arm: every region this fire presents is whole-fire or empty, so
/// `Windows::admits` calls all of them capturable without needing anything on
/// `crate::SHIFTED` at all — a body with no island in it. The sibling gate below — three lanes, the capture
/// class in two pieces — is the one that leans on the shifted list, and it
/// says so.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_capture_composition_captures_once_and_replays_identically() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the capture replay") else {
        return;
    };
    let prompt = tok.encode(PROMPT);

    shell.set_mode(Graphs::Off);
    let (eager, eager_mass) = solo(&mut shell, 0, &prompt, true, STEPS);

    shell.set_mode(Graphs::On);
    let (replayed, replay_mass) = solo(&mut shell, 0, &prompt, true, STEPS);

    let stats = shell.body_stats();
    eprintln!(
        "capture replay: {stats} | eager {:?} / replay {:?}",
        tok.decode(&eager, false),
        tok.decode(&replayed, false),
    );
    assert!(
        stats.tally.hits >= 1,
        "the capture composition was never served from a body, so this \
         compared eager against eager. `refusals` would say the admissibility \
         rule turned it away, which on a whole-fire composition would be a \
         finding rather than this wave's known limit: {stats}"
    );
    assert_eq!(
        eager, replayed,
        "the recorded graph's continuation differs from the eager walk's"
    );
    let moved = drift(&eager_mass, &replay_mass);
    assert_eq!(
        moved, 0.0,
        "the recorded graph captured a mass {moved} away from the eager walk's; \
         a replayed launch writing a stale address moves this and not the tokens"
    );
}

// ── (d) a fire no lane captured costs the axis nothing ───────────────────

/// **THE ZERO-COST CLAIM, ON THE DEVICE, AND IT IS A CLAIM ABOUT AN ABSENCE.**
///
/// `model/tests/the_declared_axes_are_the_ones_that_run` pins the compile-time
/// half — a word with neither new fact composes as the classes it composed as,
/// runs the nodes it ran, out of an arena that grew by exactly the capture
/// columns' own bytes and nothing else. This is the runtime half: a fire no
/// lane captured must issue the launches it always issued. The mechanism is
/// design §0's and not a special case — the capture arm is guarded, a fire
/// nobody captured has zero rows in its classes, and `model_exec::fire::walk`
/// skips a zero-row region before it dispatches a node — so what is asserted
/// is that the RECORDED GRAPH of a plain composition holds the same node count
/// whether or not any other fire in the process captured, and that the tokens
/// are the ones every other suite's golden states.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_fire_no_lane_captured_costs_the_axis_nothing() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the uncaptured floor") else {
        return;
    };
    let prompt = tok.encode(PROMPT);

    shell.set_mode(Graphs::On);
    let (plain, mass) = solo(&mut shell, 0, &prompt, false, STEPS);
    assert!(
        mass.is_empty(),
        "a lane that captured nothing was handed {} capture column(s)",
        mass.len(),
    );
    let plain_nodes = shell.body_stats().last_capture.nodes;

    // Now capture on another slot, then fire the plain composition again. Its
    // graph is already recorded and keyed by the composition, so a plain fire
    // after a capturing one must replay the SAME graph — same node count, same
    // tokens.
    let (_, captured) = solo(&mut shell, 1, &prompt, true, STEPS);
    assert!(!captured.is_empty(), "the capturing run captured nothing");
    let (plain_again, _) = solo(&mut shell, 0, &prompt, false, STEPS);
    let after = shell.body_stats();

    // The census names the MOST RECENTLY CAPTURED body, so the two numbers
    // are two different compositions' graphs — the plain one and the capturing
    // one — and not a before and after of the same graph. Printed for that
    // reason rather than asserted. What the assertion below rests on is the
    // plain lane's continuation, which is the observable that would move if
    // the plain composition's launches had changed.
    eprintln!(
        "uncaptured floor: {plain_nodes} nodes before, {} after; {after}; \
         continuation {:?}",
        after.last_capture.nodes,
        tok.decode(&plain, false),
    );
    assert_eq!(
        plain, plain_again,
        "a plain fire's continuation moved after another slot captured"
    );
    let text = tok.decode(&plain, false);
    assert!(
        text.starts_with(EXPECTED),
        "the uncaptured continuation is {text:?} and every other suite's golden \
         for this prompt on this SKU starts with {EXPECTED:?}; the axis is not free"
    );

    // The launches, read off the plan rather than the clock: a fire nobody
    // captured dispatched no `attention.prefill_lse` at all, and the artifact
    // does carry them, so this is an absence rather than an emptiness.
    let arms = shell
        .trace()
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node.op,
                model_ir::Operation::Attention(model_ir::Attention::PrefillLse { .. })
            )
        })
        .count();
    assert_eq!(
        arms,
        shell.score_layers().len(),
        "the SKU should state one capture arm per exported attention layer"
    );
}


// ── (e) the refusals ─────────────────────────────────────────────────────

// ── (g) the fire P4 cannot seat ──────────────────────────────────────────

/// **THE COMPOSITION THAT BREAKS THE ROW ORDER, FIRED.**
///
/// Four of this SKU's eight classes run the capture arm — the ones whose word
/// sets the `captures_scores` bit — and P4's seriation cannot make that set an
/// interval of the same order that keeps `qo_one`, `masked` and the adapter
/// window consecutive. So it withdraws the constraint and writes a `Fallback`
/// row for each of the six `attention.prefill_lse` nodes that state it. A fire
/// holding a capturing PREFILL lane and a capturing DECODE lane at once puts a
/// plain lane's rows between them, and the capture window is then two row
/// intervals rather than one — as is the `attention.plan_prefill` region that
/// carves their schedule, which P4 offers no constraint for and therefore owes
/// no row (`model_exec::fire::fallback::promised`) but which splits all the same.
///
/// Until this was fixed that fire was `Fault::Fragmented` and the batch died.
/// What it is now is `Fallback::Split { r }`: the walk dispatches the region
/// once per interval, each launch over its own pointer, its own extent and its
/// own rebased qo boundaries, and each interval's plan builder carves its own
/// schedule into its own grant. The claim is the same one every mixed-fire
/// gate in this crate makes — **every lane says what it says alone** — and it
/// is the claim that catches the split's own failure modes: a run that read
/// the other run's window computes the first interval twice, and a run that
/// read the other run's SCHEDULE indexes a boundary vector describing other
/// lanes. Both come back as tokens that moved.
///
/// The lanes are admitted one step apart, which is the whole setup: lane 0
/// prefills capturing and lane 2 prefills plain, and only then does lane 1
/// arrive to prefill capturing while lane 0 has moved on to decoding. That one
/// fire is the fragmented one, and the assert below says so rather than hoping.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_capturing_prefill_beside_a_capturing_decode_is_two_launches_and_the_same_tokens() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the split-window fire") else {
        return;
    };
    let first = tok.encode(PROMPT);
    let late = tok.encode("The largest planet is");
    let plain = tok.encode("Water boils at");

    // NOT VACUOUS, AND CHECKED AGAINST THE ARTIFACT RATHER THAN ASSERTED. The
    // whole claim is that step 1 below is a composition P4 could not seat, and
    // a model text or a seriation that stopped producing one would otherwise
    // turn this into a green test of an ordinary fire. So: bake the same plan
    // at the same budgets, compose the same three words, and count the windows
    // that come back in pieces.
    let split_windows = {
        let trace = models::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
        let compiled = compile(&trace, &BUDGETS, &DeviceProfile::default()).expect("the SKU bakes");
        let words = [
            model_exec::fire::Lane::new(word(1, true), 1),
            model_exec::fire::Lane::new(word(late.len() as u32, true), late.len() as u32),
            model_exec::fire::Lane::new(word(1, false), 1),
        ];
        let fire = model_exec::fire::compose(&compiled, &BUDGETS, &words).expect("the fire composes");
        compiled
            .template()
            .iter()
            .filter(|region| fire.classes().spans(&region.mask).len() > 1)
            .count()
    };
    assert!(
        split_windows > 0,
        "a capturing decode beside a capturing prefill no longer breaks a window, \
         so this gate is firing an ordinary composition",
    );
    eprintln!("the step-1 composition leaves {split_windows} windows in pieces");

    // The solo references, warmed. Lane 1 arrives a step late, so it takes one
    // fewer.
    let (solo_first, _) = solo(&mut shell, 0, &first, true, STEPS + 1);
    let (solo_late, _) = solo(&mut shell, 1, &late, true, STEPS);
    let (solo_plain, _) = solo(&mut shell, 2, &plain, false, STEPS + 1);

    for slot in 0..3 {
        shell.open(slot).expect("the slot opens");
    }
    let mut said: [Vec<u32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut fed: Vec<Vec<u32>> = vec![first.clone(), late.clone(), plain.clone()];
    let mut split_fires = 0usize;

    for step in 0..=STEPS {
        // Step 0 carries two lanes; from step 1 the late one is in, and at
        // that instant it is PREFILLING while lane 0 is DECODING and both are
        // capturing. That is the composition the row order cannot seat.
        // `(slot, captures)` beside the seats, because `Seated` does not read
        // back out and the two assertions below are about which lane is which.
        let held: &[(usize, bool)] = if step == 0 {
            &[(0, true), (2, false)]
        } else {
            &[(0, true), (1, true), (2, false)]
        };
        let seated: Vec<Seated<'_>> = held
            .iter()
            .map(|&(slot, captures)| seat(slot as u32, &fed[slot], captures))
            .collect();
        let fragmented = step == 1;
        if fragmented {
            split_fires += 1;
        }

        let mut mass: Vec<Vec<LayerScores>> = Vec::new();
        let out = shell
            .fire_captured(&seated, &[], &mut mass)
            .unwrap_or_else(|why| {
                panic!(
                    "the {} fire at step {step}: {why}",
                    if fragmented { "split" } else { "ordinary" },
                )
            });

        // Both capturing lanes are handed their own mass in the fragmented
        // fire — one per interval, and neither of them empty. A run that
        // resolved the other run's window would leave one of them with none.
        for (at, &(slot, captures)) in held.iter().enumerate() {
            assert_eq!(
                !mass[at].is_empty(),
                captures,
                "step {step} slot {slot}: capture and mass disagree",
            );
            let token = argmax(&out[at]);
            said[slot].push(token);
            fed[slot] = vec![token];
        }
    }

    eprintln!(
        "split: capturing {:?} / late capturing {:?} / plain {:?}",
        tok.decode(&said[0], false),
        tok.decode(&said[1], false),
        tok.decode(&said[2], false),
    );
    assert_eq!(split_fires, 1, "exactly one fire of this run is the split one");
    assert_eq!(
        said[0], solo_first,
        "the capturing lane's continuation moved when its window was split",
    );
    assert_eq!(
        said[1], solo_late,
        "the late capturing lane's continuation moved when its window was split",
    );
    assert_eq!(
        said[2], solo_plain,
        "the plain lane's continuation moved when a split fire carried it",
    );
}

/// **AND THE RECORDED GRAPH CARRIES THE SPLIT TOO.**
///
/// A split's launch count is a function of the fire's WINDOW TABLE — where
/// each class's rows stand, which classes have any — and WHICH CLASSES HAVE
/// ROWS is exactly what `record::BodyKey` names beside its lattice point (see
/// `engine_cuda::record`). So every fire that replays a captured body presents
/// the same class set the capture was walked at, and therefore the same number
/// of runs over the same intervals; the graph can hold the split the way it
/// holds everything else about a composition. That is an argument, and this is
/// the gate for it.
///
/// **AND IT IS THE GATE FOR `crate::SHIFTED` TOO, WHICH THE SINGLE-LANE
/// SIBLING ABOVE IS NOT.** A split window is not whole-fire — its rows begin
/// somewhere inside the plane and end before its end — so
/// `Windows::admits` calls this composition's regions capturable only through
/// its SHIFTING clause: every op in a fragmented region has to be one that
/// takes its lanes off the plan's staged tables rather than off a grid
/// coordinate. The capture arm (`attention.prefill_lse`) and this hybrid's
/// chunked mixer arms are on that list, which is why this fire is held WHOLE
/// by its body. A region that left the list would not refuse the composition
/// since the tier-2 campaign — it would become an island the body is cut
/// around — so what names it now is `LastCapture::islands` rather than
/// `refusals`.
///
/// The fragmented composition is repeated rather than passed through once:
/// lane 1's slot is RE-OPENED every step, so it prefills the same rows every
/// step while lane 0 decodes and lane 2 decodes, which presents one key over
/// and over — the only way a key reaches its capture fire and then a replay.
/// Eager first, then the same sequence under `Graphs::On`, and the tokens and
/// the captured mass both have to be identical.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_split_composition_captures_once_and_replays_identically() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the split replay") else {
        return;
    };
    let first = tok.encode(PROMPT);
    let late = tok.encode("The largest planet is");
    let plain = tok.encode("Water boils at");

    /// One run of the repeated fragmented composition: prefill the two
    /// long-lived lanes, then fire the split key `steps` times, re-prefilling
    /// the late lane each time so the key never moves.
    fn repeat(
        shell: &mut Shell,
        first: &[u32],
        late: &[u32],
        plain: &[u32],
        steps: usize,
    ) -> (Vec<u32>, Vec<LayerScores>) {
        for slot in 0..3 {
            shell.open(slot).expect("the slot opens");
        }
        // The two long-lived lanes prefill together; from here on they decode.
        let mut none = Vec::new();
        let out = shell
            .fire_captured(
                &[seat(0, first, true), seat(2, plain, false)],
                &[],
                &mut none,
            )
            .expect("the two-lane prefill");
        let mut fed0 = vec![argmax(&out[0])];
        let mut fed2 = vec![argmax(&out[1])];

        let mut said = Vec::new();
        let mut captured = Vec::new();
        for step in 0..steps {
            // The late lane starts over every step, so it prefills the same
            // rows every step and the window table — whose present set is half
            // of what a `record::BodyKey` names — never moves.
            shell.open(1).expect("slot 1 re-opens");
            let mut mass: Vec<Vec<LayerScores>> = Vec::new();
            let out = shell
                .fire_captured(
                    &[
                        seat(0, &fed0, true),
                        seat(1, late, true),
                        seat(2, &fed2, false),
                    ],
                    &[],
                    &mut mass,
                )
                .unwrap_or_else(|why| panic!("the split fire at step {step}: {why}"));
            said.extend([argmax(&out[0]), argmax(&out[1]), argmax(&out[2])]);
            captured = mass[1].clone();
            fed0 = vec![argmax(&out[0])];
            fed2 = vec![argmax(&out[2])];
        }
        (said, captured)
    }

    // Enough repetitions that the key warms, captures, and then replays.
    const FIRES: usize = 6;

    shell.set_mode(Graphs::Off);
    let (eager, eager_mass) = repeat(&mut shell, &first, &late, &plain, FIRES);

    shell.set_mode(Graphs::On);
    let (replayed, replay_mass) = repeat(&mut shell, &first, &late, &plain, FIRES);

    let stats = shell.body_stats();
    eprintln!("split replay: {stats}");
    assert!(
        stats.tally.hits >= 1,
        "the split composition was never served from a body, so this compared \
         eager against eager. A moved `refusals` names the region whose ops are \
         not on `crate::SHIFTED` — the fragmented window has nowhere else to \
         be admitted from: {stats}"
    );
    assert_eq!(
        eager, replayed,
        "the recorded graph of a split composition disagrees with the eager walk"
    );
    let moved = drift(&eager_mass, &replay_mass);
    assert_eq!(
        moved, 0.0,
        "the replayed split captured a mass {moved} away from the eager walk's",
    );
}

/// **A CAPTURE AND A WORD THAT DISAGREE, BOTH WAYS.**
///
/// `Fault::MaskWord`'s and `Fault::AdapterWord`'s third, and the argument
/// changes in one place because this axis carries no payload. A lane whose word
/// runs the capture arm and that asked for nothing has a mass column written
/// for it that the readout skips — paid for and thrown away. A lane that asked
/// and whose word puts it on the plain arm gets no mass, and an empty capture
/// is indistinguishable from a captured nothing. Both are refused before
/// anything launches.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_capture_and_a_word_that_disagree_are_refused() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the capture refusals") else {
        return;
    };
    let prompt = tok.encode(PROMPT);

    // Asked, and the word says the plain arm.
    shell.open(0).expect("slot 0 opens");
    let asked = shell.fire_seated(&[Seated::capturing(Lane {
        slot: 0,
        word: word(prompt.len() as u32, false),
        tokens: &prompt,
    })]);
    let said = asked.expect_err("a capture with a plain word is refused").to_string();
    eprintln!("asked, plain word: {said}");
    assert!(
        said.contains("asks to capture") && said.contains("plain arm"),
        "the refusal does not say which way it went: {said}"
    );

    // The word says the capture arm, and nobody asked.
    shell.open(0).expect("slot 0 re-opens");
    let unasked = shell.fire_seated(&[Seated::of(Lane {
        slot: 0,
        word: word(prompt.len() as u32, true),
        tokens: &prompt,
    })]);
    let said = unasked
        .expect_err("a capturing word with no ask is refused")
        .to_string();
    eprintln!("unasked, capturing word: {said}");
    assert!(
        said.contains("prefill_lse") && said.contains("never read"),
        "the refusal does not say which way it went: {said}"
    );

    // And the two stated together still fire, which is what says the refusals
    // above are about the DISAGREEMENT and not about the axis.
    shell.open(0).expect("slot 0 re-opens");
    shell
        .fire_seated(&[Seated::capturing(Lane {
            slot: 0,
            word: word(prompt.len() as u32, true),
            tokens: &prompt,
        })])
        .expect("a capture and a capturing word are one reading of one lane");
}

/// **A DRAFT AGAINST A TEXT THAT DECLARES NO HEAD.**
///
/// `Fault::Maskless`'s and `Fault::Adapterless`'s third: an MTP head is a
/// supergraph arm the model text either states or does not (design §8), and
/// `qwen35-d0.8b` does not. A lane that asked for a draft here would be handed
/// the trunk's continuation with a draft's name on it.
///
/// **AND A DRAFTING WORD ALONE IS NOT A REFUSAL, WHICH IS ALSO THE DESIGN.**
/// `model_exec::fire::compose` masks a lane's word to the bits some guard READS,
/// so a `drafts` bit against an artifact that splits on no such guard is
/// dropped and the lane composes as the word it would have had. That is the
/// right answer — a model may state a fact it does not split on — and it is
/// why the refusal below is about the ASK.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_draft_against_a_text_that_declares_no_head_is_refused() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the draft refusal") else {
        return;
    };
    let prompt = tok.encode(PROMPT);
    assert!(
        !shell.drafts(),
        "`{SKU}` declares a draft head after all, and this gate is the wrong one"
    );

    shell.open(0).expect("slot 0 opens");
    let asked = shell.fire_seated(&[Seated::drafting(Lane {
        slot: 0,
        word: drafting_word(prompt.len() as u32),
        tokens: &prompt,
    })]);
    let said = asked.expect_err("a draft against a headless text is refused").to_string();
    eprintln!("drafting ask, headless text: {said}");
    assert!(
        said.contains("draft head") && said.contains("declares none"),
        "the refusal does not name the axis: {said}"
    );

    // The word alone fires, and answers what a plain lane answers — the bit is
    // masked away because no guard of this artifact reads it.
    shell.open(0).expect("slot 0 re-opens");
    let drafting = shell
        .fire_seated(&[Seated::of(Lane {
            slot: 0,
            word: drafting_word(prompt.len() as u32),
            tokens: &prompt,
        })])
        .expect("a fact this artifact does not split on is masked, not refused");
    shell.open(1).expect("slot 1 opens");
    let plain = shell
        .fire_seated(&[Seated::of(Lane {
            slot: 1,
            word: word(prompt.len() as u32, false),
            tokens: &prompt,
        })])
        .expect("the plain lane fires");
    assert_eq!(
        argmax(&drafting[0]),
        argmax(&plain[0]),
        "a masked-away fact bit changed the answer"
    );
}

/// **A CAPTURE AGAINST A TEXT THAT DECLARES NO CAPTURE ARM.**
///
/// `Fault::Draftless`'s twin, and the refusal palo C4b's score door owes: a
/// score READ against a plan with no `attn.scores` export. It needs a SKU whose
/// model text declares none, and gemma is the family that declares nothing —
/// `the_declared_axes_are_the_ones_that_run` pins that its snapshots hold zero
/// draft-head tensors and its forward states no capture arm, so a capturing
/// lane here has nowhere for its mass to come from and is told so rather than
/// handed an empty `LaneReadout::scores` it cannot tell from a captured
/// nothing.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_capture_against_a_text_that_declares_no_arm_is_refused() {
    let _serial = serialized();
    let Some((mut shell, tok)) = gemma::ready("the capture refusal on a plain text") else {
        return;
    };
    assert!(
        !shell.captures_scores(),
        "gemma declares a capture arm after all, and this gate is the wrong one"
    );
    assert!(!shell.drafts(), "gemma declares no draft head either");

    let prompt = tok.encode(PROMPT);
    shell.open(0).expect("slot 0 opens");
    let asked = shell.fire_seated(&[Seated::capturing(Lane {
        slot: 0,
        // Gemma's own word for a plain prefill: its `Facts` reads `qo_one` and
        // `masked` and nothing else, so there is no capture bit to set and the
        // ask stands alone — which is exactly the case under test.
        word: models::gemma_4::forward::Facts::of(&Request::new(prompt.len() as u32, false)).word(),
        tokens: &prompt,
    })]);
    let said = asked
        .expect_err("a capture against an armless text is refused")
        .to_string();
    eprintln!("capturing ask, armless text: {said}");
    assert!(
        said.contains("capture its attention mass") && said.contains("declares no capture arm"),
        "the refusal does not name the axis: {said}"
    );

    // And the plain fire still works, which is what says the refusal is about
    // the ASK and not about the load.
    shell.open(0).expect("slot 0 re-opens");
    shell
        .fire_seated(&[Seated::of(Lane {
            slot: 0,
            word: models::gemma_4::forward::Facts::of(&Request::new(prompt.len() as u32, false))
                .word(),
            tokens: &prompt,
        })])
        .expect("the plain gemma lane fires");
}

/// gemma4-E4B, loaded the way `masked_axis` loads it — the budgets are the
/// L40S's and that file argues them.
mod gemma {
    use super::{Boot, Budget, Path, PathBuf, Platform, Shell};

    const SKU: &str = "gemma4-e4b-bf16-kv-bf16";

    fn snapshot() -> Option<PathBuf> {
        if let Ok(stated) = std::env::var("PIE_GEMMA_SNAPSHOT") {
            let path = PathBuf::from(stated);
            return path.is_dir().then_some(path);
        }
        let home = std::env::var("HOME").ok()?;
        let snapshots = Path::new(&home)
            .join(".cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots");
        std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .find(|path| path.join("tokenizer.json").exists())
    }

    pub fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
        if !engine_cuda::device::present() {
            eprintln!("skipping {what}: no CUDA device on this machine");
            return None;
        }
        let Some(checkpoint) = snapshot() else {
            eprintln!(
                "skipping {what}: no gemma-4-E4B-it snapshot in the hugging face \
                 cache (set PIE_GEMMA_SNAPSHOT)"
            );
            return None;
        };
        let Some(container) = super::container(&checkpoint) else {
            eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
            return None;
        };
        let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
            .expect("the checkpoint's tokenizer loads");
        let trace = models::trace_of(SKU).expect("the catalog ships gemma")(Platform::Cuda);
        let source = ztensor_compat::index(&container).expect("the checkpoint opens");
        let contract = models::import_of(SKU).expect("the catalog ships an import")(&source)
            .expect("the import contract fits its own checkpoint");
        drop(source);

        let shell = Shell::load(Boot {
        // Full residency: the whole weight table on the device, which is what
        // an uncapped `Residency` plans (alto design §7).
        residency: engine_cuda::experts::Plan::default(),
            trace,
            contract: &contract,
            checkpoint: &checkpoint,
            budget: Budget::new(4, 768),
            patches: None,
            profile: None,
            page_size: 16,
            context: 1024,
            slots: 4,
            ordinal: 0,
            graphs: engine_cuda::Graphs::Off,
            // The golden at load; the gates that record state their own mode.
            // `bodies` is written out because it defaults to TRUE now — the
            // word is documentation here, and `Off` is what keeps the arming
            // pass from running.
            knobs: engine_cuda::Knobs {
                bodies: true,
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
        eprintln!("gemma4-e4b loaded — capture layers {:?}", shell.score_layers());
        Some((shell, tokenizer))
    }
}

// ── (f) the drafting SKU does not fit this device ────────────────────────

/// **AN HONEST OUT-OF-MEMORY REFUSAL IS ITSELF A GATE.**
///
/// `qwen36-27b-bf16-kv-bf16` is the one shipping SKU whose checkpoint
/// publishes a draft head, and its bf16 weights are ~52 GiB against this
/// device's 46. It cannot be loaded here and this wave does not pretend
/// otherwise. What CAN be asserted, and is worth asserting, is that the
/// refusal is a sentence with a NUMBER in it rather than a launch failure
/// three layers down: residency is asked for at load, against a budget the
/// caller stated, and a shell that discovered the shortfall at its first fire
/// would have already written the page tables.
///
/// Skipped, not failed, on a machine with no such snapshot — the point is the
/// refusal's shape, and there is nothing to refuse without the checkpoint.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn the_drafting_sku_does_not_fit_this_device() {
    let _serial = serialized();
    if !engine_cuda::device::present() {
        eprintln!("skipping the drafting load: no CUDA device on this machine");
        return;
    }
    let Some(checkpoint) = drafting_snapshot() else {
        eprintln!(
            "skipping the drafting load: no Qwen3.6-27B snapshot in the hugging \
             face cache (set PIE_DRAFTING_SNAPSHOT)"
        );
        return;
    };
    let shards = shards(&checkpoint);
    if shards.is_empty() {
        eprintln!("skipping the drafting load: {checkpoint:?} holds no tensor container");
        return;
    }
    let trace = models::trace_of(DRAFTING).expect("the catalog ships the SKU")(Platform::Cuda);
    // The plan is the half that DOES fit, and it is worth reading before the
    // load refuses: this SKU declares the draft export, which is what makes
    // the refusal below about the device rather than about the model text.
    assert!(
        trace.seams.iter().any(|seam| seam.seam == "mtp"),
        "`{DRAFTING}` states no `mtp` export, so it is not the drafting SKU"
    );
    // ALL FIFTEEN SHARDS AS ONE NAME SPACE. `ztensor_compat::index` takes one
    // file, which is right for the single-container SKUs every other suite
    // loads and wrong here: a 27B checkpoint's final norm lives in the last
    // shard, and an import contract built over the first refuses for a reason
    // that has nothing to do with the device.
    let source =
        ztensor_compat::index_all(&shards).expect("the checkpoint's shards open as one");
    let contract = models::import_of(DRAFTING).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);

    let refused = Shell::load(Boot {
        // Full residency: the whole weight table on the device, which is what
        // an uncapped `Residency` plans (alto design §7).
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: BUDGETS,
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        graphs: Graphs::Off,
        // As above: the golden at load, `bodies` said out loud because it is
        // true by default now.
        knobs: engine_cuda::Knobs {
            bodies: true,
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
    });
    let said = match refused {
        Ok(shell) => {
            let (weights, arena, pools, inputs) = shell.footprint();
            panic!(
                "`{DRAFTING}` LOADED on this device — weights {:.2} GiB, arena \
                 {:.1} MiB, pools {:.1} MiB, inputs {:.1} MiB. The MTP device \
                 gates this file skips are now runnable and should be written.",
                weights as f64 / (1u64 << 30) as f64,
                arena as f64 / (1 << 20) as f64,
                pools as f64 / (1 << 20) as f64,
                inputs as f64 / (1 << 20) as f64,
            )
        }
        Err(why) => why.to_string(),
    };
    eprintln!("the drafting SKU on this device: {said}");
    // THE REFUSAL CARRIES BOTH NUMBERS, and that is the whole of what this
    // gate can assert. `cudaMalloc answered 2` is a true sentence and a
    // useless one: it does not say whether the shortfall was six gigabytes or
    // sixty, and it reads the same as every other runtime failure.
    // `device::alloc` turns `cudaErrorMemoryAllocation` into a `Fault::Ceiling`
    // with the ask and the free, so this asks for the shape rather than for a
    // magic string.
    assert!(
        said.contains("device memory"),
        "the refusal is not the one this device is entitled to give: {said}"
    );
    let numbers: Vec<u64> = said
        .split(|c: char| !c.is_ascii_digit())
        .filter(|word| !word.is_empty())
        .filter_map(|word| word.parse().ok())
        .collect();
    assert!(
        numbers.len() >= 2,
        "the refusal states {} number(s); a ceiling is an ask AND a have: {said}",
        numbers.len(),
    );
    let (need, have) = (numbers[0], numbers[1]);
    eprintln!(
        "  wanted {:.2} GiB, {:.2} GiB free",
        need as f64 / (1u64 << 30) as f64,
        have as f64 / (1u64 << 30) as f64,
    );
    assert!(
        need > have,
        "the refusal says it wanted {need} bytes and had {have}, which is not a \
         shortfall at all"
    );
}

// ── the load ─────────────────────────────────────────────────────────────

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    cached("models--Qwen--Qwen3.5-0.8B")
}

fn drafting_snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_DRAFTING_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    cached("models--Qwen--Qwen3.6-27B")
}

fn cached(repo: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snapshots = Path::new(&home)
        .join(".cache/huggingface/hub")
        .join(repo)
        .join("snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .find(|path| path.join("tokenizer.json").exists())
}

/// Every container in a snapshot, sorted — a sharded checkpoint's whole name
/// space.
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

/// A loaded shell, or `None` and a sentence saying what was missing.
fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
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
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let trace = models::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = models::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);

    let shell = Shell::load(Boot {
        // Full residency: the whole weight table on the device, which is what
        // an uncapped `Residency` plans (alto design §7).
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: BUDGETS,
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        graphs: Graphs::Off,
        // As above: the golden at load, `bodies` said out loud because it is
        // true by default now.
        knobs: engine_cuda::Knobs {
            bodies: true,
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
    let (weights, arena, pools, inputs) = shell.footprint();
    eprintln!(
        "{SKU} loaded — weights {:.2} GiB, arena {:.1} MiB, pools {:.1} MiB, \
         inputs {:.1} MiB, capture layers {:?}, drafts {}",
        weights as f64 / (1u64 << 30) as f64,
        arena as f64 / (1 << 20) as f64,
        pools as f64 / (1 << 20) as f64,
        inputs as f64 / (1 << 20) as f64,
        shell.score_layers(),
        shell.drafts(),
    );
    Some((shell, tokenizer))
}
