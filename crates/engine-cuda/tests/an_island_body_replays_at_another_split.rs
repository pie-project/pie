//! **AN ISLAND BODY, ARMED AT BOOT, REPLAYED AT A SPLIT THE CAPTURE NEVER SAW
//! — AND THE SIBLING COMPOSITION ONE CLASS AWAY FROM IT, WALKING AND SAYING SO
//! BY NAME** — the tier-2 campaign's gate.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 --release \
//!     --test an_island_body_replays_at_another_split -- --nocapture
//! ```
//!
//! # What tier 2 is, in one paragraph
//!
//! A body is one recorded graph per COMPOSITION, replayed for every fire that
//! keys to it. Some regions cannot be in that graph at all: a gathered one,
//! whose rows were compacted into a scratch slab and live at a fire-dependent
//! offset inside it; a grouped one, whose span is a union with foreign rows in
//! the gaps; a windowed one whose ops do not all read the staged seat's start.
//! Tier 1 refused the whole composition over any of them and walked. Tier 2
//! cuts the template AROUND them (`record::cuts`): each maximal stretch of
//! capturable regions becomes its own exec, and the ISLANDS between them are
//! re-issued by the eager walk on the same stream, fire after fire —
//! `exec₁ → island → exec₂ → …`, one host for-loop, no runtime capture.
//!
//! # Why THIS composition, and which three classes it is
//!
//! P4 withdraws qwen3.5's `captures_scores` window — the axis that crosses
//! `qo_one` where every earlier axis nested inside it — and writes a
//! `Fallback` row for each of its six `attention.prefill_lse` nodes. Put a
//! plain lane's rows between two capturing classes and that window is in
//! pieces; below the copy/split crossover the table asks for `Fallback::Copy`,
//! so the shell gathers them into one compacted rectangle. That rectangle is
//! the island.
//!
//! WHICH capturing classes is the whole of what this file had to get right,
//! and it is not a matter of taste. The bake states twelve classes and P4
//! seriates them once; `a_copied_window_and_a_split_one_are_the_same_bytes.rs`
//! reads the shipped order of the eight an unmasked lane can land in off the
//! artifact, and names the five this file needs:
//!
//! ```text
//! the shipped order:   4   0   2   6   7   3   1   5
//! the withdrawn mask:  ─           ─   ─           ─     {4,5,6,7}
//!   4  capturing prefill, no adapter
//!   0  plain prefill              ← the separator this file fires
//!   6  capturing prefill WITH an adapter
//!   1  plain decode
//!   5  capturing decode
//! ```
//!
//! So the three lanes below are a capturing prefill (`4`), a plain prefill
//! (`0`) standing inside the mask's span, and a capturing prefill that ROUTES
//! TO AN ADAPTER (`6`) — `Facts::has_adapter` is the only bit between `4` and
//! `6`, and it is what puts a second capturing class BEHIND the separator
//! instead of in front of it.
//!
//! **AND THE SPLIT MOVING IS THE WHOLE POINT.** A body promises to serve every
//! fire of its key whatever the rows do, and its key carries no row counts at
//! all — so the honest test of a segmented body is not "does the fire that
//! captured it replay" but "does a DIFFERENT split of the same present set and
//! the same bucket replay, and answer what the eager walk answers". Both
//! halves of the machine are on trial there: the captured stretches, whose
//! grids and schedules were carved at the key's ceilings and must dominate a
//! split they never saw; and the island, whose launches are re-issued from the
//! host every fire and must plan, grid and address at the FIRE's own live
//! geometry (`Run::captured` is the one gate that stands every ceiling down
//! inside one — an island that took a ceiling would be gridded past the
//! rectangle the gather actually filled).
//!
//! # The lattice has an EDGE, and this file now stands on both sides of it
//!
//! `Shell::arm_bodies`' fragmenting arm does not enumerate present sets. It
//! enumerates MINIMAL WITNESSES (`Shell::fragmenting`, `Shell::witness`): per
//! (mask with two or more classes, separator outside it), the three classes
//! that witness the break — the separator, the nearest mask class in FRONT of
//! it and the nearest one BEHIND it. Over the order above, mask `{4,5,6,7}`
//! and separator `0` derive `{0,4,6}`: 4 is in front, and the nearest behind
//! is 6, not 5.
//!
//! `{0,4,5}` — a capturing prefill, the same plain separator, and a capturing
//! DECODE — breaks the very same window (its order is `4 0 5`, which is two
//! runs of the mask) and is not a witness of it. Nothing is wrong with that
//! fire. It is one of the exponentially many present sets a lattice cannot
//! hold a body for, and what the engine owes it is the behaviour it has: past
//! `Graphs::seal_bodies` it WALKS, it is counted where an operator can see it
//! (`BodyStats::sealed_declines`), and the first one is named on stderr with
//! its key spelled out. So the gate is two claims and not one, because the
//! lattice's edge is a designed thing and an untested designed thing is a
//! hope:
//!
//! ```text
//! (a) the boot ARMS this composition — `BodyStats::armed_at_load` moves, some
//!     armed body is SEGMENTED, and the map seals before any caller fires
//! (b) two different splits of the WITNESS both REPLAY — `hits` moves twice,
//!     `captures` does not move once past the boot, and `sealed_declines`
//!     does not move at all
//! (c) every fire of it really does GATHER — `FireCost::copied` is nonzero on
//!     the eager arm and on the replayed one, which is the island's premise
//!     and the thing a seriation change would quietly take away
//! (d) and each split answers, bit for bit, what the eager walk of that same
//!     split answers
//! (e) while the SIBLING one class away — `{0,4,5}`, the same three prompts
//!     with the routed lane's adapter dropped and its rows cut to one — is
//!     turned away: `sealed_declines` moves, `hits` does not, `captures` does
//!     not, and its bytes are still the eager walk's
//! ```
//!
//! (a) is what no other file in this suite can claim: the three enumerations
//! that predate this campaign — decode-only, prefill-only, mixed — top out at
//! TWO present classes, and two classes can never break a window (a fire's
//! class order is the shipped order with the absent classes dropped, and
//! dropping a class can only close a gap). It takes a third class standing
//! between two of a mask's own, which is `Shell::fragmenting`'s enumeration
//! and this file's composition.
//!
//! (d) is the load-bearing half. A replay that answered different bytes would
//! be worse than no replay: the same kernels run over the same rows through
//! the same page table, so a difference of one ULP is a bug and not noise.
//! (e) asserts it too, for the fire that is NOT replayed: an eager walk the
//! router took because it had no body must be the eager walk, and the cheapest
//! way for the decline path to be wrong is to be half a body.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_cuda::{Boot, FireCost, Graphs, LayerScores, Seated, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The three prompts, one per lane and one per class. They are sliced rather
/// than fed whole (see [`head`]): a split is a statement about ROW COUNTS, and
/// a file whose splits are whatever the tokenizer happens to produce is one
/// whose bucket moves when the tokenizer does.
const CAPTURING: &str = "The capital of France is";
const PLAIN: &str = "Water boils at one hundred degrees";
const ROUTED: &str = "The largest planet in our solar system is";

/// The adapter every routed lane here binds to. Slot zero, because the arming
/// pass's own synthetic for class `6` binds slot zero too
/// (`Shell::synthetic_lanes`: `self.corrected.contains(class).then_some(0)`) —
/// and a bank's CONTENTS are not in the `record::BodyKey` (decision 17), which
/// is why registering the loud one below after the boot invalidates nothing.
const ADAPTER: u32 = 0;

/// **THE TWO SPLITS, AS ROWS PER LANE** — `[class 4, class 0, class 6]`.
///
/// The same three classes and the same lattice point (ten rows and eleven both
/// round up to sixteen), with every lane's row count different between them,
/// and NO lane at one row: a one-row capturing lane is class `5` rather than
/// class `4` (`Facts::captures_scores` precedes `qo_one` in the split, so a
/// capturing lane takes the capture arm whatever its row count — but `qo_one`
/// is still a fact of its own and still names a class of its own), and a fire
/// that let one drift to one row would be firing (e)'s composition and
/// calling it (b)'s.
const FIRST: [usize; 3] = [4, 3, 3];
const SECOND: [usize; 3] = [2, 5, 4];

/// One shell at a time per process — `kernels-cuda`'s scratch slabs are
/// process-global and keyed by name, and this file's copy slab is one of them
/// (`serve_smoke.rs` argues the rule whole).
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// **THE LATTICE IS STATED, WHICH IS THIS FILE'S ONE PREREQUISITE.**
/// `Shell::arm_bodies` enumerates prefill, mixed and FRAGMENTED present sets
/// over `Budget::buckets`, so a deployment that declares no lattice arms only
/// its decode keys — and the composition below would then never be armed at
/// boot, which is claim (a). Sixteen is the point both splits' rows round up
/// to, and every rung here sits under the copy/split crossover
/// (`model_compiler::layout`'s `CROSSOVER_ROWS`), which is what makes the
/// withdrawn window a GATHERED one rather than a split.
///
/// **AND IT IS TWO POINTS RATHER THAN THREE, BECAUSE THE MAP'S SEATS ARE THE
/// ENUMERATION'S BUDGET.** `record::MAX_BODIES` is sixty-four and this bake
/// states twelve classes, so one lattice point already enumerates thirty
/// prefill and mixed keys plus one fragmenting witness per (mask, separator)
/// pair — and the fragmenting arm is the LAST one attempted inside each
/// bucket. At three points the map filled inside the second one and the key
/// this file's fires actually present was named in the boot line's `never
/// attempted` warning instead of being armed. Dropping the rung no fire here
/// lands on is the deployment saying what it fires; it is not a tuning of the
/// engine, and the boot line is where the same sentence is available to an
/// operator.
///
/// **THE BOOT LINE IS ALSO WHERE THE PREMISE OF (b) IS READABLE**, which is
/// worth stating because this file's fires all land at SIXTEEN. A load of
/// these budgets prints `fragmented 14/14`, and fourteen is exactly the number
/// of witnesses this bake has in TOTAL — one per (mask, separator) pair,
/// deduplicated — so every one of them was attempted and armed inside the
/// FIRST bucket, and the second bucket's fragmenting arm is what the map ran
/// out of seats before reaching.
///
/// **AND `max_adapters` IS ONE RATHER THAN ZERO, WHICH IS NOT A WIDENING.**
/// The witness needs class `6`, class `6` is the capturing class whose word
/// sets `Facts::has_adapter`, and a lane may not claim that bit without naming
/// a bank to route to (`Fault::AdapterWord`). The number is a BAKE-time
/// capacity check against the shape the model text declared
/// (`model_compiler::compile`) and feeds no other pass: it moves no window, no
/// class, no order and no bucket. What it buys is the right to register the
/// one adapter [`register_loud`] writes.
fn budgets() -> Budget {
    Budget {
        max_lanes: 4,
        max_tokens: 256,
        buckets: vec![16, 32],
        max_adapters: 1,
    }
}

/// One lane's word: the model's own `Classify::of`, packed. Three of the six
/// facts move here — the row count (`qo_one`), the capture and the adapter —
/// and the class table is what turns them into the ids the header names.
fn word(query_len: u32, captures: bool, adapted: bool) -> u64 {
    model::qwen_3::forward::Facts::of(
        &Request::new(query_len, false)
            .capturing_scores(captures)
            .adapted(adapted),
    )
    .word()
}

/// A lane, with its word derived from the same three facts the seat states —
/// which is the standing rule and not a convenience: the word decides the
/// CLASS and the seat carries the PAYLOAD, and a shell that found them
/// disagreeing refuses the fire by name (`Fault::AdapterWord`,
/// `Fault::ScoreWord`).
fn seat<'a>(slot: u32, tokens: &'a [u32], captures: bool, adapter: Option<u32>) -> Seated<'a> {
    let lane = engine_cuda::Lane {
        slot,
        word: word(tokens.len() as u32, captures, adapter.is_some()),
        tokens,
    };
    Seated {
        captures_scores: captures,
        adapter,
        ..Seated::of(lane)
    }
}

/// `(tokens, captures, adapter)` per lane — one fire's whole composition.
type Composition = Vec<(Vec<u32>, bool, Option<u32>)>;

/// **THE FIRST `rows` TOKENS OF `text`**, and it says so when the prompt is
/// too short. A bare slice would panic with an index and read as a tokenizer
/// fault; what it actually means is that a prompt can no longer make the split
/// this file asks for, which is a sentence about this file.
fn head(tok: &tokenizer::Tokenizer, text: &str, rows: usize) -> Vec<u32> {
    let all = tok.encode(text);
    assert!(
        all.len() >= rows,
        "{text:?} encodes to {} tokens and this split wants {rows} of them",
        all.len(),
    );
    all[..rows].to_vec()
}

/// **THE ARMED WITNESS, AT ONE SPLIT** — present set `{0,4,6}`, the triple
/// `Shell::witness` derives for mask `{4,5,6,7}` and separator `0`.
fn witness_at(tok: &tokenizer::Tokenizer, rows: [usize; 3]) -> Composition {
    vec![
        // class 4 — capturing prefill, no adapter
        (head(tok, CAPTURING, rows[0]), true, None),
        // class 0 — the plain separator, standing inside the mask's span
        (head(tok, PLAIN, rows[1]), false, None),
        // class 6 — capturing prefill, routed
        (head(tok, ROUTED, rows[2]), true, Some(ADAPTER)),
    ]
}

/// **THE SIBLING THE LATTICE DOES NOT HOLD** — present set `{0,4,5}`, and the
/// diff against [`witness_at`] is ONE LANE: the routed one drops its adapter
/// and all but one of its rows, so class `6` becomes class `5` and the fire
/// keeps every other thing about itself. Same three prompts, same three slots,
/// same mask broken into the same two runs, same bucket — and no body.
fn sibling(tok: &tokenizer::Tokenizer) -> Composition {
    vec![
        // class 4, and class 0, exactly as the witness seats them
        (head(tok, CAPTURING, FIRST[0]), true, None),
        (head(tok, PLAIN, FIRST[1]), false, None),
        // class 5 — capturing DECODE, which is where the triple leaves the
        // lattice: it is the mask's last class in the baked order, not 0's
        // nearest one behind.
        (head(tok, ROUTED, 1), true, None),
    ]
}

/// One composition, fired once from a clean set of slots.
///
/// **THE SLOTS ARE RE-OPENED EVERY TIME**, which is what makes the two arms of
/// each A/B the same fire rather than two consecutive steps of one
/// conversation: an open slot carries kv from whatever ran before it.
fn fire_it(shell: &mut Shell, lanes: &Composition) -> (Vec<Vec<f32>>, FireCost) {
    for slot in 0..lanes.len() as u32 {
        shell.open(slot).expect("the slot opens");
    }
    let seated: Vec<Seated<'_>> = lanes
        .iter()
        .enumerate()
        .map(|(at, (tokens, captures, adapter))| seat(at as u32, tokens, *captures, *adapter))
        .collect();
    let mut mass: Vec<Vec<LayerScores>> = Vec::new();
    let out = shell
        .fire_captured(&seated, &[], &mut mass)
        .expect("the island-bearing fire");
    (out, shell.last_fire_cost())
}

/// Bit-for-bit, and it says WHICH number moved when it does not hold.
fn same_logits(left: &[Vec<f32>], right: &[Vec<f32>], what: &str) {
    assert_eq!(left.len(), right.len(), "{what}: lane counts");
    for (lane, (a, b)) in left.iter().zip(right).enumerate() {
        assert_eq!(a.len(), b.len(), "{what}: lane {lane} vocabulary");
        for (at, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{what}: lane {lane} logit {at} — eager {x} against body {y}",
            );
        }
    }
}

// ── the gate ─────────────────────────────────────────────────────────────

/// **THE FIVE CLAIMS, ON ONE LOAD.**
///
/// One load and two modes, for `bodies_gate.rs`'s reason: two loads would
/// differ in their weight residency, their arena carve and their autotuner
/// state, and the diff would be about those instead of about the router.
///
/// The eager arm runs FIRST and is warmed, because the dense tuner tunes a
/// GEMM shape on its second sighting and a cold arm against a warm one is two
/// tactic ladders rather than two routers.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn an_armed_island_body_replays_at_another_split_and_its_sibling_walks() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the segmented body gate") else {
        return;
    };
    // BEFORE ANY FIRE, so that both arms of every A/B read the same bank. The
    // boot armed its bodies against whatever the bank held then, which is the
    // point of decision 17 and not a caveat to it: the graph key is the
    // composition and the bank's addresses were reserved at load, so writing
    // the slot afterwards invalidates nothing and changes every fire's
    // arithmetic identically.
    register_loud(&mut shell, ADAPTER);

    // **CLAIM (a): THE BOOT ARMED IT.** Read before any fire, so nothing here
    // can be traffic's. `armed_at_load` counts keys the enumeration seated and
    // pinned; `segmented` counts how many of the RESIDENT bodies hold an
    // island, which is the number that says the fragmenting arm reached a
    // composition P4 wrote a `Fallback` row for.
    let armed = shell.body_stats();
    eprintln!("at boot: {armed}");
    assert!(
        armed.armed_at_load >= 1,
        "the boot armed nothing at all, so nothing below is about arming: {armed}"
    );
    assert!(
        armed.segmented >= 1,
        "the boot armed no SEGMENTED body. `Shell::arm_bodies`' fragmenting arm \
         is what enumerates a present set that breaks a mask — three classes, \
         with one standing between two of the mask's own — and without it no \
         load ever arms a tier-2 body: {armed}"
    );

    // The two splits of the armed witness, and the sibling one class away.
    let first = witness_at(&tok, FIRST);
    let second = witness_at(&tok, SECOND);
    let sibling = sibling(&tok);

    // ── the golden, and it is the eager walk of each fire ────────────────
    shell.set_mode(Graphs::Off);
    for _ in 0..2 {
        let _ = fire_it(&mut shell, &first);
        let _ = fire_it(&mut shell, &second);
        let _ = fire_it(&mut shell, &sibling);
    }
    let (eager_first, cost_first) = fire_it(&mut shell, &first);
    let (eager_second, cost_second) = fire_it(&mut shell, &second);
    let (eager_sibling, cost_sibling) = fire_it(&mut shell, &sibling);

    // **CLAIM (c), THE PREMISE HALF.** A seriation that seated the capture
    // window, or a crossover that put sixteen on the split side, would leave
    // every assertion below comparing a fire against itself and passing.
    assert!(
        cost_first.copied > 0 && cost_second.copied > 0,
        "neither split gathered anything, so this composition has no island and \
         the file is testing the wrong thing ({} / {} gathered)",
        cost_first.copied,
        cost_second.copied,
    );
    assert!(
        cost_sibling.copied > 0,
        "the sibling gathered nothing, so it is not the same window in pieces and \
         claim (e) is about some other fire",
    );

    // ── the tiered router, on the body the BOOT captured ─────────────────
    let before = shell.body_stats();
    shell.set_mode(Graphs::On);
    let (bodied_first, replay_first) = fire_it(&mut shell, &first);
    let (bodied_second, replay_second) = fire_it(&mut shell, &second);
    let after = shell.body_stats();
    eprintln!("after two splits of the witness: {after}");

    // **CLAIM (b): BOTH SPLITS REPLAYED, AND NOTHING CAPTURED.** `hits` is the
    // evidence a fire was SERVED from a body — `captures` was already nonzero
    // before either fire, which is what an armed load means — and the capture
    // counter standing still is the other half of "upfront": past the seal the
    // serving path records nothing at all.
    assert!(
        after.hits >= before.hits + 2,
        "two splits of one armed key produced fewer than two hits. A key whose \
         body was armed at boot serves every split of its bucket, because the \
         grids and the schedules were carved at the KEY's ceilings and the seat \
         retires the rows this fire did not bring: {after}"
    );
    assert_eq!(
        after.captures, before.captures,
        "the serving path captured {} body/bodies. The map is sealed after \
         `Shell::arm_bodies`, so a fire that mints one is a fire the boot did \
         not arm — read `sealed_declines` beside it: {after}",
        after.captures - before.captures,
    );
    // **AND THE WITNESS IS NOT SILENTLY WALKING**, which is the way claim (b)
    // failed before this file fired the triple the enumeration actually
    // derives: a present set that breaks the same mask without BEING the
    // minimal witness reaches the sealed map, finds nothing, walks, and every
    // byte below still matches. This is the assertion that can tell the two
    // apart, and it belongs to the ARMED half.
    assert_eq!(
        after.sealed_declines, before.sealed_declines,
        "the sealed map turned this composition away {} time(s) — so the key \
         these fires present is not one `Shell::arm_bodies` armed, and the \
         stderr line beside this one spells the key it wanted: {after}",
        after.sealed_declines - before.sealed_declines,
    );

    // **CLAIM (c), THE REPLAY HALF.** The window table is derived every fire,
    // in `prepare`, whatever the mode — so a replayed fire still says whether
    // it gathered, and a body serving a composition whose table gathers is a
    // body with an island in it (`record::Graphs::fire_body`'s island
    // `debug_assert` is the engine-side statement of the same thing).
    assert!(
        replay_first.copied > 0 && replay_second.copied > 0,
        "the replayed fires gathered nothing while their eager twins gathered \
         {} and {} — the body served a composition it was not cut for",
        cost_first.copied,
        cost_second.copied,
    );

    // **CLAIM (d): AND THE BYTES ARE THE EAGER WALK'S.** Per split, because
    // that is where a segmented body can go wrong: the captured stretches were
    // recorded at ONE split and are replayed at both, and the island between
    // them is re-issued from the host at whatever geometry each fire brings.
    same_logits(&eager_first, &bodied_first, "the first split");
    same_logits(&eager_second, &bodied_second, "the second split");

    // ── and the composition the lattice does not hold ────────────────────
    //
    // **CLAIM (e): THE SIBLING WALKS, AND THE ENGINE SAYS SO.** `{0,4,5}`
    // fragments the same window as `{0,4,6}` — its order is `4 0 5`, two runs
    // of mask `{4,5,6,7}` — and it is not the witness the arming pass derived,
    // because 0's nearest mask class BEHIND it is 6. The lattice cannot cover
    // every present set; what it owes the ones it misses is this: no capture
    // on a caller's critical path, a counter an operator can read, and the key
    // named once on stderr.
    let (bodied_sibling, replay_sibling) = fire_it(&mut shell, &sibling);
    let declined = shell.body_stats();
    eprintln!("after the unarmed sibling: {declined}");
    assert!(
        declined.sealed_declines >= after.sealed_declines + 1,
        "the sealed map served a composition the enumeration never armed, or \
         counted the decline somewhere else. `BodyStats::sealed_declines` is \
         the one surface that says how far the traffic stands outside the \
         lattice: {declined}"
    );
    assert_eq!(
        declined.hits, after.hits,
        "the unarmed sibling was served from a body. Its present set is not a \
         key `Shell::arm_bodies` armed, so a hit here means some other key's \
         body answered for it — which is the one failure a byte diff cannot be \
         relied on to catch: {declined}"
    );
    assert_eq!(
        declined.captures, after.captures,
        "the declined fire minted a body. A sealed map warms toward no capture: \
         that is what `sealed_declines` means and `misses` does not: {declined}"
    );
    assert!(
        replay_sibling.copied > 0,
        "the declined fire stopped gathering when the mode changed, so the two \
         arms of its byte diff are not the same walk",
    );
    // AND ITS NUMBERS ARE STILL THE ORACLE'S. An eager walk the router took
    // because it had no body must be the eager walk — the cheapest way for the
    // decline path to be wrong is to be half a body.
    same_logits(&eager_sibling, &bodied_sibling, "the unarmed sibling");
}

// ── the adapter, which is what makes class 6 reachable ───────────────────

/// A LOUD adapter — big enough that its correction is visibly not the
/// identity, so a replay that dropped the routed lane's rows, or gathered
/// them from the wrong offset, would not pass the byte diff by accident.
///
/// Lifted from `a_copied_window_and_a_split_one_are_the_same_bytes.rs`'s
/// three-run gate, which reaches class `6` for the same reason and by the same
/// route. The planes are sized off [`Shell::banks`], because a bank's capacity
/// and slot are the MODEL TEXT's shapes and a registration that guessed them
/// would be a test asserting its own arithmetic.
fn register_loud(shell: &mut Shell, id: u32) {
    let built: Vec<(String, Vec<u8>)> = shell
        .banks()
        .iter()
        .map(|&(name, _, slot)| {
            let count = usize::try_from(slot).expect("a slot fits this host") / 2;
            let mut bytes = Vec::with_capacity(count * 2);
            for at in 0..count {
                let value = if name.ends_with(".lora_a") {
                    0.20 - ((at % 5) as f32) * 0.07
                } else {
                    0.15 + ((at % 3) as f32) * 0.05
                };
                bytes.extend_from_slice(&bf16_bits(value).to_le_bytes());
            }
            (name.to_string(), bytes)
        })
        .collect();
    assert!(
        !built.is_empty(),
        "this plan declares no adapter bank, so class 6 is unreachable and the \
         witness this file fires cannot be composed"
    );
    let planes: Vec<engine_cuda::AdapterPlane<'_>> = built
        .iter()
        .map(|(bank, bytes)| engine_cuda::AdapterPlane {
            bank: bank.as_str(),
            bytes,
        })
        .collect();
    shell
        .register_adapter(id, &planes)
        .unwrap_or_else(|why| panic!("registering adapter {id}: {why}"));
}

/// f32 to bf16, round-to-nearest-even — the loader's conversion, restated so
/// the adapter registered is the one described.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

// ── the load ─────────────────────────────────────────────────────────────

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
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);

    let shell = Shell::load(Boot {
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: budgets(),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        // **THE TIERED MODE AT LOAD, WHICH IS WHAT MAKES CLAIM (a) SAYABLE.**
        // `Shell::arm_bodies` refuses to run at all unless the mode records, so
        // a load that stated `Off` and turned the mode on afterwards would mint
        // its bodies from TRAFFIC and this file would be a different test
        // (`a_copied_window_and_a_split_one_are_the_same_bytes.rs` is that one).
        graphs: Graphs::On,
        // **AND `grouped: false`, WHICH IS THIS FILE'S PREMISE AND NOT A
        // TUNING.** `crate::GROUPED` is named by default now, and a groupable
        // consumer is nearly free to withdraw — so `layout::choose` takes the
        // LoRA correction and seats the score window, leaving nothing for a
        // copy to gather. Emptying the list puts the score window back on the
        // losing side, which is the artifact that has an island in it.
        knobs: engine_cuda::Knobs {
            grouped: false,
            bodies: true,
            copies: true,
            ..engine_cuda::Knobs::default()
        },
        program_cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        weight_cache_dir: None,
    })
    .expect("the shell loads");
    Some((shell, tokenizer))
}
