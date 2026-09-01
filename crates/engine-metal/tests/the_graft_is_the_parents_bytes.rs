//! **`copy_kv` on an Apple GPU: the bytes arrive, the parent keeps its own,
//! and the queue is the only ordering there is.**
//!
//! The host half of the graft is settled and runs anywhere —
//! `a_forked_page_is_one_run_and_not_one_move_per_token` is arithmetic over
//! two integer lists and needs no device. What it cannot say is that the
//! bytes ARRIVE: `Pools::copy_kv` walks `rows × planes` and encodes one
//! `copyFromBuffer:` per run per plane into a command buffer of its own on
//! the fire queue, and a mover that copied a subset of the planes would leave
//! a fork attending to the parent's keys at some layers and its own at
//! others — which reads as fluent garbage rather than as an error. That is
//! the claim this file settles, and it is session G of
//! `.wiki/alto/metal-verify-queue.md`.
//!
//! Five gates, in the order they can fail:
//!
//! 1. **The graft is the parent's bytes.** Prefill, `copy_kv` the whole pages
//!    onto fresh ids, fire the same continuation against the copies, and diff
//!    the logits row for row. Byte-identical, or the plane loop is copying a
//!    subset.
//! 2. **The partial tail.** The boundary page half filled, its live cells
//!    copied one `KvMove` per token — the shape the runtime's own
//!    `pipeline::fire::copy_into` builds — and then the CHILD appends past
//!    them. The parent's next answer is unchanged, which is what
//!    distinguishes a correct graft from one that shares the page.
//! 3. **The ordering, which is the queue's and not a fence's.** A step, then
//!    a `copy_kv` reading the pages that step writes, then a step reading the
//!    copies, with NO drain anywhere. The claim is that command buffers
//!    execute in commit order. Run at `runahead` 2, where it can fail, and at
//!    1, where it cannot, and diff. **This is the gate that can actually
//!    fail.**
//! 4. **Refusals cost nothing.** A page past the pool is `Impossible` and
//!    leaves the queue untouched; a host-pinned or `MetalPrivate` end is
//!    `Unsupported` and names the pair; an overlapping move is `Invalid`. And
//!    the shell still fires after each.
//! 5. **The capability record is a promise a verb keeps.**
//!    `Capabilities::kv_copy.device_to_device` and the other three false,
//!    read back from a real load.
//!
//! # The two doors, and why both are used
//!
//! Gates 1–3 reach the shell through `Shell`/`Seated`, because a graft is
//! only observable to a caller that OWNS its page table: `Seated::pages` and
//! `Seated::held` are the contract's "the runtime keeps its own table"
//! spelling, and a `Lane` alone cannot name the page a fork wrote onto.
//! Gates 4 and 5 reach it through `Engine::copy_kv` on a loaded `Metal`,
//! because the refusal TAXONOMY (`Impossible`/`Unsupported`/`Invalid`) and
//! the `Capabilities` record are the contract's own vocabulary and the shell
//! does not speak them.
//!
//! # Gating
//!
//! An Apple target is not a machine with a GPU, and neither is a machine
//! with a GPU one with a 1.4 GB checkpoint on its disk. So the file is
//! `cfg`'d to Apple and SKIPS at run time, saying which of the three was
//! missing, rather than being `#[ignore]`d — an ignored test on the one box
//! that could run it is a test nobody runs.
//!
//! ```text
//! cargo test -p engine-metal --release --test the_graft_is_the_parents_bytes -- --nocapture
//! ```
//!
//! `PIE_SMOKE_SNAPSHOT` overrides where it looks.

#![cfg(target_vendor = "apple")]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use engine::Engine;
use engine::runahead::Runahead;
use engine::transfer::{KvCopy, KvMove, MemoryDomain};
use engine_metal::store::Move;
use engine_metal::{Boot, Lane, Seated, Shell, StepView};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// The catalog row this file serves, spelled as the catalog spells it — the
/// same one `serve_smoke` pins, so a divergence here is the graft's and not
/// the load's.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// Tokens per kv page, stated rather than derived because every page id below
/// is arithmetic over it.
const PAGE: u32 = 16;

/// The most tokens one sequence may hold, and therefore `512 / 16 = 32` pages
/// per slot, `4 × 32 = 128` pages in the pool.
const CONTEXT: u32 = 512;

/// How many sequences the pools seat.
const SLOTS: u32 = 4;

/// **The fixture's length, and it is chosen for gate 2.** Two whole pages and
/// half of a third: `40 = 2 × 16 + 8`, so the boundary page has eight live
/// cells and eight the parent has not written yet — which is the shape a fork
/// has to copy out rather than share.
const FIXTURE: usize = 40;

/// The prompt the fixture is cut from. Longer than `serve_smoke`'s on purpose:
/// a graft that copies one page is a graft whose plane loop is never asked to
/// walk two.
const TEXT: &str = "The capital of France is Paris, and the capital of Japan is Tokyo. \
     The largest planet in our solar system is Jupiter, and water boils at one \
     hundred degrees Celsius at sea level. The Pacific is the largest ocean, and \
     the tallest mountain above sea level is Everest.";

/// The parent's page table: four pages of slot 0's own block, which is more
/// than the fixture covers so that the parent can append past it.
const PARENT: [u32; 4] = [0, 1, 2, 3];

/// **ONE SHELL AT A TIME, PER PROCESS**, for `serve_smoke`'s reason: each of
/// these holds ~1.5 GiB resident on a 32 GiB unified machine, and gate 3
/// holds two loads in one test function (sequentially, the first dropped
/// before the second is asked for).
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The snapshot directory: the checkpoint AND the tokenizer that goes with
/// it, because a vocabulary from another snapshot decodes the right ids into
/// the wrong words.
fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    // The suite runs as root over tailscale ssh, so `HOME` is not the
    // owner's — the cache the checkpoint actually lives in is named
    // explicitly beside it.
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
    homes.iter().find_map(|home| {
        let snapshots =
            Path::new(home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
        std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .find(|path| path.join("tokenizer.json").exists())
    })
}

/// The container the contract is checked against — one file of the snapshot,
/// whichever one holds the tensors.
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

/// The lane word the model's own `Classify` computes — runtime-side work,
/// done here because this test IS the runtime for the length of one fire.
fn word(query_len: u32) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
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

/// **Where two logit rows first differ, by BITS**, or `None` when they are
/// the same row.
///
/// Bits and not epsilon, and that is the whole point of every diff in this
/// file: a graft either moved the parent's cells or it did not, and two
/// identical fires over identical cells on one device produce identical
/// floats. A tolerance here would hide exactly the failure the file is for —
/// a plane loop that copied seventeen layers of eighteen answers something
/// *close*.
fn first_difference(left: &[f32], right: &[f32]) -> Option<usize> {
    if left.len() != right.len() {
        return Some(left.len().min(right.len()));
    }
    left.iter()
        .zip(right)
        .position(|(a, b)| a.to_bits() != b.to_bits())
}

/// The two rows, and a sentence naming where they parted.
fn diverged(left: &[f32], right: &[f32]) -> String {
    match first_difference(left, right) {
        None => "identical".to_string(),
        Some(at) => format!(
            "first differ at logit {at}: {} vs {}",
            left.get(at).copied().unwrap_or(f32::NAN),
            right.get(at).copied().unwrap_or(f32::NAN),
        ),
    }
}

/// One lane whose page table and length are the CALLER'S — which is the only
/// shape a graft is observable through, because a `Lane` alone cannot name
/// the page a fork wrote onto.
fn owned<'a>(slot: u32, tokens: &'a [u32], pages: &'a [u32], held: u32) -> Seated<'a> {
    Seated {
        pages,
        held: Some(held),
        ..Seated::of(Lane {
            slot,
            word: word(tokens.len() as u32),
            tokens,
        })
    }
}

/// One fire of one caller-owned lane, blocking, in call order — **and the
/// slot is opened first, which is a statement about what `copy_kv` is FOR
/// and not a tidy-up.**
///
/// The smoke SKU is `trace_hybrid`: three quarters of `qwen35-d0.8b`'s stack
/// is gated-delta, and a gated-delta layer's history is not in a kv page at
/// all — it is in the slot's RECURRENT bank, which `Pools::copy_kv` walks
/// straight past (`let Shape::Kv { .. } = *shape else { continue }`).
/// Deliberately: moving recurrent state is `copy_state`, a different verb
/// with a different contract, and a `copy_kv` that quietly moved it too
/// would be doing something its name does not say.
///
/// So a fork of THIS checkpoint is not complete with a graft alone, and a
/// diff between a parent slot that has been decoding and a child slot that
/// has not is dominated by the recurrent banks — it would answer "the graft
/// is broken" whatever the graft did. `open` zeroes those banks, so every
/// fire below starts from the same recurrent state and the ONLY thing that
/// differs between a parent row and a child row is which kv pages the
/// attention layers read. That is exactly the half `copy_kv` owns, and it is
/// the half these gates are about.
///
/// **WHAT THIS FILE THEREFORE DOES NOT COVER**: that a fork of a hybrid
/// model is correct end to end. It is not, on `copy_kv` alone, and that is
/// a `copy_state` question banked for whoever asks it.
fn fire_owned(
    shell: &mut Shell,
    slot: u32,
    tokens: &[u32],
    pages: &[u32],
    held: u32,
) -> Vec<f32> {
    shell
        .open(slot)
        .unwrap_or_else(|why| panic!("slot {slot} opens to a zeroed recurrent bank: {why}"));
    let lanes = [owned(slot, tokens, pages, held)];
    let mut rows = shell
        .fire_seated(&lanes)
        .unwrap_or_else(|why| panic!("the lane at slot {slot}, held {held}, fires: {why}"));
    assert_eq!(rows.len(), 1, "one lane in, one row of logits out");
    rows.pop().expect("one row")
}

/// **A whole-page graft**: page for page, both ends at offset zero, which is
/// what `KvCopy`'s two parallel lists mean and what a prefix graft states.
fn whole_pages(src: &[u32], dst: &[u32]) -> Vec<Move> {
    Move::plan(
        &KvCopy {
            src: MemoryDomain::MetalShared,
            dst: MemoryDomain::MetalShared,
            src_page_ids: src.to_vec(),
            dst_page_ids: dst.to_vec(),
            moves: Vec::new(),
        },
        PAGE,
    )
    .expect("a whole-page graft is a plan the page arithmetic admits")
}

/// **A partial tail, one `KvMove` PER TOKEN** — the vector
/// `runtime::pipeline::fire`'s `copy_into` actually builds, submitted in that
/// shape on purpose: the coalescing that turns it into one run per plane is
/// part of what is under test.
fn live_cells(pairs: &[(u32, u32)], live: u32) -> Vec<Move> {
    let mut moves = Vec::new();
    for &(src_page, dst_page) in pairs {
        moves.extend((0..live).map(|at| KvMove {
            src_page_id: src_page,
            src_token_offset: at,
            dst_page_id: dst_page,
            dst_token_offset: at,
        }));
    }
    Move::plan(
        &KvCopy {
            src: MemoryDomain::MetalShared,
            dst: MemoryDomain::MetalShared,
            src_page_ids: Vec::new(),
            dst_page_ids: Vec::new(),
            moves,
        },
        PAGE,
    )
    .expect("a partial tail is a plan the page arithmetic admits")
}

/// Everything the tests below share: a loaded shell at the stated run-ahead
/// depth and the fixture's token ids, or `None` and a sentence saying which
/// of the machine, the checkpoint and the tokenizer was missing.
fn ready(what: &str, runahead: Runahead) -> Option<(Shell, Vec<u32>)> {
    if !engine_metal::device::present() {
        eprintln!("skipping {what}: this machine publishes no Metal device");
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
    let mut prompt = tokenizer.encode(TEXT);
    assert!(
        prompt.len() >= FIXTURE,
        "the fixture text encodes to {} tokens and the gates are written at {FIXTURE} — \
         a shorter fixture would not half-fill a boundary page, which is gate 3's whole \
         subject",
        prompt.len()
    );
    prompt.truncate(FIXTURE);

    let trace = models::trace_of(SKU).expect("the catalog ships this file's SKU");
    let trace = trace(Platform::Metal);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract =
        models::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
            .expect("the SKU's import contract fits its own checkpoint");
    drop(source);

    let mut shell = Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        // §M-4c, as `serve_smoke` states it: an unstamped snapshot proceeds,
        // and the deployment's facts are stated honestly all the same.
        tp_size: 1,
        precision: models::precision_of(SKU)
            .expect("the catalog states this row's precision")
            .to_string(),
        // Small on purpose: the arena reserves `max_tokens` rows of a
        // 248320-wide logit column, and these gates need a fixture, not a
        // batch. Sized for a 32 GiB unified machine.
        budget: Budget::new(4, 256),
        patches: None,
        profile: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: SLOTS,
        runahead,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the shell loads");
    eprintln!(
        "{what}: loaded on {} at runahead {}, {} pages of {PAGE} tokens in the pool",
        shell.device_name(),
        runahead.frames_in_flight,
        shell.paging().pages(),
    );
    // **EVERY SLOT THIS FILE FIRES DOWN IS OPENED ONCE, HERE, AND THE
    // CONTROL BELOW IS WHY.** `Seated::pages` and `Seated::held` say where a
    // lane's cells live and how many of them are readable, but they are not
    // the whole of a slot: `open` is what clears the recurrent banks and
    // resets the extent, and a slot that has never been opened carries
    // whatever the load left in it. A fire down such a slot answers a
    // plausible, finite, WRONG row — which, in a file whose every gate is a
    // diff between two slots, reads exactly like a bad graft.
    // `two_slots_down_one_page_table_answer_one_row` is the control that
    // pins this, and it fails without these lines.
    Some((shell, prompt))
}

/// **GATE 2 — THE GRAFT IS THE PARENT'S BYTES.**
///
/// One shell, one checkpoint. The parent prefills the fixture into pages it
/// owns; `copy_kv` grafts those whole pages onto fresh ids; the SAME
/// continuation is fired against the copies and against the parent, and the
/// two logit rows are diffed bit for bit.
///
/// **BYTE-IDENTICAL OR THE PLANE LOOP IS COPYING A SUBSET.** A page is not
/// one allocation: page `p` exists once per PLANE of every `CacheRow::Kv` the
/// plan declares — eighteen layers times a key half and a value half — and
/// this shell cuts one reservation into a key half at `0` and a value half at
/// `values_at`. A loop that walked the rows and forgot the second base, or
/// walked the planes and stopped at the first row, produces a child that
/// attends its own keys at some layers and the parent's at others. That is
/// fluent garbage, not an error, and only a diff catches it.
#[test]
fn the_whole_page_graft_answers_what_the_parent_answers() {
    let _serial = serialized();
    let Some((mut shell, prompt)) = ready("the whole-page graft", Runahead::F1) else {
        return;
    };
    // Slot 2's block, which nothing else in this test addresses. Four pages,
    // so the child can append past what was grafted onto it.
    let child: [u32; 4] = [64, 65, 66, 67];

    // 1. The parent prefills the fixture into its own pages.
    let prefill = fire_owned(&mut shell, 0, &prompt, &PARENT, 0);
    finite(&prefill, "the parent's prefill");
    let first = argmax(&prefill);

    // 2. The graft, at the instant the parent holds exactly the fixture: the
    //    pages the fixture covers, onto fresh ids. The last of them is half
    //    live and moves whole — a page's LIVE length is the runtime's
    //    bookkeeping and not a number this verb is handed.
    let covered = FIXTURE.div_ceil(PAGE as usize);
    let moves = whole_pages(&PARENT[..covered], &child[..covered]);
    shell.copy_kv(&moves).expect("the whole-page graft encodes");

    // 3. The same continuation, twice: once down the parent's own pages and
    //    once down the copies. Fed the same tokens in the same order, so the
    //    only thing that differs between the two runs is which pages the
    //    attention reads.
    let mut fed = first;
    let mut parent_rows = Vec::new();
    let mut child_rows = Vec::new();
    for step in 0..4u32 {
        let token = [fed];
        let from_parent = fire_owned(&mut shell, 0, &token, &PARENT, FIXTURE as u32 + step);
        let from_child = fire_owned(&mut shell, 1, &token, &child, FIXTURE as u32 + step);
        finite(&from_parent, "the parent's continuation");
        finite(&from_child, "the child's continuation");
        fed = argmax(&from_parent);
        parent_rows.push(from_parent);
        child_rows.push(from_child);
    }

    for (step, (parent, child_row)) in parent_rows.iter().zip(&child_rows).enumerate() {
        assert_eq!(
            first_difference(parent, child_row),
            None,
            "continuation step {step} off the GRAFTED pages is not the parent's own row \
             ({}). The graft copied something, because the row is finite and spread — so \
             what it did not copy is a subset of the planes: `Pools::copy_kv` walks \
             `rows × planes` with `values_at` as the second base, and a child attending \
             its own keys at some layers and the parent's at others answers exactly this \
             way, fluently and wrongly.",
            diverged(parent, child_row)
        );
    }
    eprintln!(
        "the graft: {covered} pages copied, {} continuation rows byte-identical",
        parent_rows.len()
    );
}

/// **GATE 3 — THE PARTIAL TAIL, AND THE PAGE IS NOT SHARED.**
///
/// The boundary page is half filled — `40 = 2 × 16 + 8`, eight live cells and
/// eight the parent has not written — so a fork must copy those eight OUT
/// rather than share the page, or the child's first append lands in a cell
/// the parent is about to want.
///
/// Two children, because there are two things to say and they need different
/// pages: one is fed the parent's own next token and must answer the parent's
/// own row (the copy really moved the LIVE cells); the other is fed a
/// different token and appends past the copy, after which the parent's NEXT
/// answer must be unchanged (the child's append did not land in the parent's
/// page). The second is the gate that distinguishes a correct graft from one
/// that shares.
#[test]
fn a_forked_tail_appends_past_the_copy_and_the_parent_keeps_its_own() {
    let _serial = serialized();
    let Some((mut shell, prompt)) = ready("the partial tail", Runahead::F1) else {
        return;
    };
    // The two full pages are SHARED — the same ids in both tables, which is
    // what a prefix-tree fork does — and only the boundary page is forked.
    let echo: [u32; 4] = [PARENT[0], PARENT[1], 64, 65];
    let appender: [u32; 4] = [PARENT[0], PARENT[1], 66, 67];
    let live = FIXTURE as u32 % PAGE;
    assert_eq!(
        live, 8,
        "this gate is written at a HALF-FILLED boundary page; {live} live cells is a \
         different fixture and a different claim"
    );

    // ── The control run: the parent alone, no fork anywhere, two steps past
    //    the fixture. These two rows are what the test run is held against.
    let prefill = fire_owned(&mut shell, 0, &prompt, &PARENT, 0);
    finite(&prefill, "the control prefill");
    let next = argmax(&prefill);
    // A token that is NOT the one the parent appends, so that a child sharing
    // the parent's page writes something the parent can notice. Taken from
    // the fixture rather than invented: it is a real id in this vocabulary.
    let other = prompt
        .iter()
        .copied()
        .find(|&token| token != next)
        .expect("the fixture holds a token that is not the parent's own next one");
    let control_first = fire_owned(&mut shell, 0, &[next], &PARENT, FIXTURE as u32);
    let control_second = fire_owned(&mut shell, 0, &[next], &PARENT, FIXTURE as u32 + 1);
    finite(&control_first, "the control's first step");
    finite(&control_second, "the control's second step");

    // ── The test run. The parent re-prefills over its own pages (the cells
    //    past the live tail are stale and unread — `kv_len` bounds what the
    //    attention sees), and then the fork happens at exactly the instant
    //    the parent holds the fixture and nothing more.
    let again = fire_owned(&mut shell, 0, &prompt, &PARENT, 0);
    assert_eq!(
        first_difference(&prefill, &again),
        None,
        "the same prefill over the same pages answered differently the second time ({}), \
         so this fixture is not repeatable and nothing below it means anything",
        diverged(&prefill, &again)
    );

    let moves = live_cells(&[(PARENT[2], echo[2]), (PARENT[2], appender[2])], live);
    assert_eq!(
        moves.len(),
        2,
        "eight cells stated one `KvMove` per token are ONE run per destination, and \
         `Move::plan` answered {} — the coalescing is what makes a fork 36 blits rather \
         than 180",
        moves.len()
    );
    shell.copy_kv(&moves).expect("the partial tail encodes");

    // The parent takes its own step, writing cell 8 of its boundary page.
    let parent_first = fire_owned(&mut shell, 0, &[next], &PARENT, FIXTURE as u32);
    assert_eq!(
        first_difference(&control_first, &parent_first),
        None,
        "the parent's own next answer changed merely because a fork was grafted off it \
         ({}) — a `copy_kv` reads the parent's pages and writes somebody else's, so a \
         parent that notices one is a graft writing into its source",
        diverged(&control_first, &parent_first)
    );

    // The echo child, fed the SAME token off the COPIED tail: the eight live
    // cells are the parent's or this row is not the parent's row.
    let echoed = fire_owned(&mut shell, 1, &[next], &echo, FIXTURE as u32);
    assert_eq!(
        first_difference(&control_first, &echoed),
        None,
        "the child reading a COPIED boundary page did not answer what the parent answered \
         off the original ({}). The two full pages are shared ids, so the only cells that \
         can differ are the eight the token-granular moves were supposed to carry: this \
         is a partial-tail copy that moved fewer cells, the wrong offset, or fewer planes \
         than a page has",
        diverged(&control_first, &echoed)
    );

    // The appending child writes cell 8 of ITS boundary page — the cell the
    // parent has just written in its own. A fork that shared the page would
    // land here.
    let appended = fire_owned(&mut shell, 2, &[other], &appender, FIXTURE as u32);
    finite(&appended, "the appending child's step");

    // And now the parent's SECOND step, which reads the cell the child would
    // have clobbered.
    let parent_second = fire_owned(&mut shell, 0, &[next], &PARENT, FIXTURE as u32 + 1);
    assert_eq!(
        first_difference(&control_second, &parent_second),
        None,
        "after the child appended past its copied tail, the parent's next answer changed \
         ({}) — which means the child's append landed in the PARENT's boundary page. That \
         is a fork that shared the page instead of copying its live cells out, and every \
         sequence forked off a partial page would silently take the other one's token",
        diverged(&control_second, &parent_second)
    );
    eprintln!("the partial tail: {live} live cells copied per child, the parent kept its own");
}

/// **GATE 4 — THE ORDERING IS THE QUEUE'S, AND THIS IS THE ONE THAT CAN
/// ACTUALLY FAIL.**
///
/// The shell has no stream to hang a graft on — a `Frame` is opened by
/// `enqueue` and committed inside it — so `Shell::copy_kv` takes a command
/// buffer of its own on the same queue and rests the whole claim on one
/// property: **command buffers execute in the order they were committed.**
/// There is no fence, no event and no drain anywhere in it (article 2).
///
/// So: submit a step, then `copy_kv` reading the pages that step writes, then
/// a step that reads the copies — nothing synchronized between them — and
/// hold the child's answer against the parent's own. A wrong answer here
/// looks like the child attending the parent's PRE-fire cells, which is a
/// perfectly finite row of the right shape.
///
/// Run at `runahead` 2, where the first step is genuinely still airborne when
/// the graft is committed, and at 1, where `stage`'s own seat harvest has
/// already waited for it — and diff the two, because a shell whose ordering
/// depends on the depth is a shell whose ordering is an accident.
#[test]
fn a_graft_committed_behind_an_airborne_step_reads_what_that_step_wrote() {
    let _serial = serialized();

    let mut arms: Vec<(u8, Vec<f32>, Vec<f32>)> = Vec::new();
    for depth in [2u8, 1] {
        let Some((mut shell, prompt)) = ready("the graft's ordering", Runahead::of(depth)) else {
            return;
        };
        let child: [u32; 4] = [64, 65, 66, 67];

        // Setup, and everything blocking about this test happens here: the
        // parent holds the fixture, and the two token ids the probe feeds are
        // in hand so that nothing below has to read a row to decide what to
        // fire next.
        let prefill = fire_owned(&mut shell, 0, &prompt, &PARENT, 0);
        finite(&prefill, "the parent's prefill");
        let appended = argmax(&prefill);
        let probe = prompt[0];

        // ── THE PROBE. Three commits, in this order, with no drain, no
        //    `rows_of` and no `reap` between them:
        //
        //      step A   parent appends `appended` at token 40 -> page 2, cell 8
        //      graft    pages 0,1,2 -> 64,65,66, whole pages
        //      step B   child appends `probe` at token 41, reading the copies
        //
        //    At depth 2 the seat ring has a free arm, so `stage` does not
        //    harvest and step A is still on the device when the graft and
        //    step B are committed behind it. At depth 1 the ring is the eager
        //    shell's and step B's own staging waits for A, which is why that
        //    arm cannot fail and is the golden model this one is bisected
        //    against.
        let a_token = [appended];
        let a_lanes = [owned(0, &a_token, &PARENT, FIXTURE as u32)];
        let landed_a = {
            use engine::frame::Shell as FrameShell;
            let prepared = FrameShell::prepare(
                &mut shell,
                StepView {
                    lanes: &a_lanes,
                    attachments: &[],
                    media: &[],
                    done: None,
                },
                None,
            )
            .expect("step A stages");
            let enqueued = FrameShell::enqueue(&mut shell, prepared).expect("step A commits");
            FrameShell::settle(&mut shell, enqueued).expect("step A files")
        };

        // The pages the parent covers AFTER step A's append: 41 tokens is
        // three pages, and the third of them is the one step A just wrote
        // into. That page is the whole subject of this gate.
        let covered = (FIXTURE + 1).div_ceil(PAGE as usize);
        let moves = whole_pages(&PARENT[..covered], &child[..covered]);
        shell
            .copy_kv(&moves)
            .expect("the graft encodes behind an airborne step");

        let b_token = [probe];
        let b_lanes = [owned(1, &b_token, &child, FIXTURE as u32 + 1)];
        let landed_b = {
            use engine::frame::Shell as FrameShell;
            let prepared = FrameShell::prepare(
                &mut shell,
                StepView {
                    lanes: &b_lanes,
                    attachments: &[],
                    media: &[],
                    done: None,
                },
                None,
            )
            .expect("step B stages");
            let enqueued = FrameShell::enqueue(&mut shell, prepared).expect("step B commits");
            FrameShell::settle(&mut shell, enqueued).expect("step B files")
        };

        // ── Only now does the host wait for anything.
        let rows_a = shell.rows_of(&landed_a).expect("step A's rows");
        let rows_b = shell.rows_of(&landed_b).expect("step B's rows");
        finite(&rows_a[0], "step A");
        finite(&rows_b[0], "step B");

        // The reference: the parent takes the SAME step B down its own pages,
        // which hold step A's append by construction. Blocking, ordinary, and
        // after everything above has landed — this row is what step B's row
        // has to be.
        let reference = fire_owned(&mut shell, 0, &[probe], &PARENT, FIXTURE as u32 + 1);
        finite(&reference, "the reference step");

        assert_eq!(
            first_difference(&reference, &rows_b[0]),
            None,
            "at runahead {depth}: a step read pages a `copy_kv` copied, and the copy did \
             not carry what the step in front of it had written ({}).\n\
             \n\
             WHAT A DIVERGENCE HERE MEANS. Three command buffers went onto one \
             `MTLCommandQueue` in this order — the parent's append, the graft, the \
             child's step — with nothing synchronized between them, and the shell's whole \
             claim is that they therefore EXECUTE in that order. A child that answers \
             something else is a child attending the parent's PRE-fire cells: the graft \
             read page {} before the append landed in it, so cell {} of that page is \
             whatever was there before. Nothing faults, nothing is out of bounds and the \
             row is finite — a fork would simply lose the last token of its parent's \
             prefix, forever and silently. If this arm fails at depth 2 and passes at \
             depth 1, the ordering was never the queue's: it was the eager shell's \
             accidental drain, and `Shell::copy_kv` needs a real dependency rather than a \
             commit order.",
            diverged(&reference, &rows_b[0]),
            PARENT[covered - 1],
            FIXTURE % PAGE as usize,
        );
        arms.push((depth, rows_b[0].clone(), reference));
        // The shell goes back before the next depth's load asks the machine
        // for another 1.5 GiB.
        drop(shell);
    }

    if let [(deep, deep_child, _), (eager, eager_child, _)] = arms.as_slice() {
        assert_eq!(
            first_difference(deep_child, eager_child),
            None,
            "the same graft answered differently at runahead {deep} and runahead {eager} \
             ({}) — the two depths differ only in whether `stage`'s seat harvest happened \
             to wait for the step in front of the graft, so an answer that depends on the \
             depth is an answer that depends on a wait nobody wrote down",
            diverged(deep_child, eager_child)
        );
        eprintln!("the ordering: runahead {deep} and runahead {eager} agree, bit for bit");
    }
}

// ── The contract's own door, for the two gates that are about the contract's
//    vocabulary rather than about bytes.

/// The load door a shell is opened with: how a checkpoint's tensors become a
/// plan's params is the model's declaration, resolved by the party that links
/// the catalog — which, for the length of this file, is the test.
fn contract_for(
    trace: &model_ir::Trace,
    path: &Path,
) -> std::result::Result<checkpoint::contract::ModelContract, String> {
    let import = models::import_of(&trace.name).ok_or_else(|| {
        format!(
            "this build ships no import contract for {:?}, so a checkpoint's tensors \
             cannot be mapped onto its params",
            trace.name
        )
    })?;
    let file = container(path).ok_or_else(|| format!("{path:?} holds no tensor container"))?;
    let source = ztensor_compat::index(&file).map_err(|error| format!("{error}"))?;
    import(&source).map_err(|error| format!("{error}"))
}

/// A loaded `Metal` — the contract's own door — and the fixture, or `None`
/// and the sentence saying what was missing.
fn engine_ready(what: &str) -> Option<(engine_metal::Metal, Vec<u32>)> {
    if !engine_metal::device::present() {
        eprintln!("skipping {what}: this machine publishes no Metal device");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping {what}: no Qwen3.5-0.8B snapshot in the hugging face cache \
             (set PIE_SMOKE_SNAPSHOT)"
        );
        return None;
    };
    if container(&checkpoint).is_none() {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    }
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let mut prompt = tokenizer.encode(TEXT);
    prompt.truncate(FIXTURE);

    let trace = models::trace_of(SKU).expect("the catalog ships this file's SKU");
    let trace = trace(Platform::Metal);

    // `::default()` and not the unit literal, though `DeviceBoot` is empty
    // today: `api.rs` spells the door this way on purpose, so that the first
    // knob to arrive there does not break its callers.
    #[allow(clippy::default_constructed_unit_structs)]
    let boot = engine_metal::DeviceBoot::default();
    let mut metal = engine_metal::Metal::new(boot, contract_for);
    let loaded = metal
        .load(engine::LoadRequest {
            trace,
            checkpoint: engine::Checkpoint::Path(checkpoint.clone()),
            budgets: engine::Budgets {
                max_lanes: 4,
                max_tokens: 256,
                buckets: Vec::new(),
                max_adapters: 0,
                page_size: PAGE,
                max_context: CONTEXT,
                slots: SLOTS,
                max_patches: None,
                max_images: None,
            },
            residency: engine::Residency::default(),
            // The system default device, and the only one this plane
            // addresses: anything above zero is refused by name.
            ordinal: 0,
            frames_in_flight: 1,
            // §M-4c: what `runtime::engine::load::request_of` would fill in
            // for this row. The snapshot below is a raw one and carries no
            // `pie.serving/1` stamp, so the check proceeds — but the request
            // states the facts anyway, because that is what the door this
            // gate is exercising receives, and an empty precision is refused.
            tp_size: 1,
            precision: models::precision_of(SKU)
                .expect("the catalog states this row's precision")
                .to_string(),
        })
        .expect("the metal engine loads the smoke's checkpoint");
    eprintln!(
        "{what}: loaded {:?}, {} kv pages of {} tokens",
        loaded.facts.trace_name, loaded.caps.pools.kv_pages, loaded.caps.pools.kv_page_size,
    );
    // As in `ready`: a seated fire down a never-opened slot reads stale
    // state, and the refusal gates below fire to prove the queue survives.
    {
        let shell = metal.shell_mut().expect("the engine has a shell");
        for slot in 0..SLOTS {
            shell.open(slot).expect("every slot this file uses opens clean");
        }
    }
    Some((metal, prompt))
}

/// One fire through the shell behind the engine, to say that the queue is
/// still flying after a refusal.
fn still_fires(metal: &mut engine_metal::Metal, prompt: &[u32], after: &str) {
    let shell = metal.shell_mut().expect("the engine has a shell");
    let row = fire_owned(shell, 0, prompt, &PARENT, 0);
    finite(&row, &format!("the fire after {after}"));
}

/// **GATE 5 — REFUSALS COST NOTHING, AND EACH IS REFUSED UNDER ITS OWN
/// NAME.**
///
/// The taxonomy is the point. `Impossible` is a scheduling answer the
/// runtime's lane loop acts on — every ceiling this shell states was reserved
/// at LOAD, so no amount of freeing makes a pool carved for 128 pages address
/// a 129th — `Unsupported` says the plan is a plan this contract describes
/// and what is missing is storage on THIS engine, and `Invalid` says the
/// caller's own statement is wrong. A caller that cannot tell the three apart
/// retries the one it should drop and drops the two it should route around.
///
/// And each of them leaves the queue exactly as it found it: a `Frame` that
/// is dropped before `commit_async` puts NOTHING on the device, so the fire
/// after a refusal is a fire like any other.
#[test]
fn a_refused_graft_leaves_the_queue_untouched_and_names_what_it_refused() {
    let _serial = serialized();
    let Some((mut metal, prompt)) = engine_ready("the graft's refusals") else {
        return;
    };
    let pages = metal
        .capabilities()
        .expect("a loaded engine publishes its capabilities")
        .pools
        .kv_pages;

    let device = MemoryDomain::MetalShared;
    let plan = |src: MemoryDomain, dst: MemoryDomain, page_pairs: (Vec<u32>, Vec<u32>), moves: Vec<KvMove>| KvCopy {
        src,
        dst,
        src_page_ids: page_pairs.0,
        dst_page_ids: page_pairs.1,
        moves,
    };

    // ── A PAGE PAST THE POOL IS `Impossible`. The pages a fork addresses are
    //    stated through the same `Supply` the fire path states its union
    //    through — a fork names a destination page before any fire has
    //    admitted one — so a page past the reservation comes back as a
    //    ceiling rather than as a blit into nothing.
    let past = plan(device, device, (vec![0], vec![pages]), Vec::new());
    match metal.copy_kv(&past) {
        Err(engine::Error::Impossible(why)) => {
            assert!(
                why.contains(&pages.to_string()),
                "the refusal for a page past a pool of {pages} does not name the number it \
                 was held against: {why}"
            );
        }
        other => panic!(
            "page {pages} of a {pages}-page pool was answered with {other:?}. `Impossible` \
             is the answer a runtime acts on — every ceiling this shell states was \
             reserved at LOAD, so `Exhausted` would send a scheduler off to free pages \
             that would never help",
        ),
    }
    still_fires(&mut metal, &prompt, "a page past the pool");

    // ── A HOST-PINNED END IS `Unsupported`, AND IT NAMES THE PAIR. Not
    //    `Invalid`: the plan is a plan this contract describes, and what is
    //    missing is a swap pool this load does not reserve. Unified memory
    //    does not supply one — it only makes the pools themselves
    //    host-addressable.
    for (src, dst, expected) in [
        (MemoryDomain::HostPinned, device, "out of host-pinned"),
        (device, MemoryDomain::HostPinned, "into host-pinned"),
        (device, MemoryDomain::MetalPrivate, "PRIVATE"),
        (MemoryDomain::MetalPrivate, device, "PRIVATE"),
    ] {
        let refused = plan(src, dst, (vec![0], vec![1]), Vec::new());
        match metal.copy_kv(&refused) {
            Err(engine::Error::Unsupported { verb, engine }) => {
                assert_eq!(
                    engine, "metal",
                    "the refusal for {src:?} -> {dst:?} does not say which engine refused it"
                );
                assert!(
                    verb.contains(expected),
                    "the refusal for {src:?} -> {dst:?} is {verb:?}, which does not name \
                     the pair — a refusal a caller can only print is worth less than one \
                     it can match on, and the enumeration in `kv_copy_direction` exists \
                     precisely so this sentence is about THESE two domains"
                );
            }
            other => panic!(
                "{src:?} -> {dst:?} was answered with {other:?}. `Unsupported` and not \
                 `Invalid`: the submission is well formed and what is absent is storage \
                 on this engine, which is exactly the difference the two variants carry — \
                 and `Capabilities::kv_copy` promises the same thing ahead of time",
            ),
        }
    }
    still_fires(&mut metal, &prompt, "a domain pair this load has no bytes in");

    // ── A MOVE WHOSE TWO ENDS OVERLAP IS `Invalid`. Both ends live in one
    //    reservation, so a run that reads and writes overlapping cells of one
    //    page is a blit whose regions overlap — undefined, and silently so. A
    //    caller that means "shift a page's tokens" states a staging page and
    //    two moves.
    let overlapping: Vec<KvMove> = (0..3)
        .map(|at| KvMove {
            src_page_id: 2,
            src_token_offset: at,
            dst_page_id: 2,
            dst_token_offset: at + 1,
        })
        .collect();
    match metal.copy_kv(&plan(device, device, (Vec::new(), Vec::new()), overlapping)) {
        Err(engine::Error::Invalid(why)) => {
            assert!(
                why.contains("overlap"),
                "the refusal for a run whose ends overlap does not say so: {why}"
            );
        }
        other => panic!(
            "three cells shifted one place inside one page were answered with {other:?}. \
             `Invalid` is the caller's own statement being wrong — the shift it meant is \
             two moves through a staging page — and a shell that ENCODED it would hand \
             the blit two overlapping regions, which is undefined and would not fault",
        ),
    }
    still_fires(&mut metal, &prompt, "an overlapping move");

    // ── And the served pair still is served, which is what makes the three
    //    refusals above statements about those plans rather than about the
    //    verb.
    metal
        .copy_kv(&plan(device, device, (vec![0], vec![64]), Vec::new()))
        .expect("the one pair this load has bytes in is still served after three refusals");
    still_fires(&mut metal, &prompt, "a graft that was accepted");
}

/// **GATE 6 — THE CAPABILITY RECORD IS A PROMISE A VERB KEEPS.**
///
/// `Capabilities::kv_copy` used to be a record nothing stood behind. It is
/// now the ahead-of-time spelling of `Metal::copy_kv`'s own door — one served
/// pair, three refusals — so it is read back from a REAL load rather than
/// asserted against the constructor, and each `false` is checked beside the
/// reason there is no storage for it.
#[test]
fn a_real_load_promises_device_to_device_and_refuses_the_other_three() {
    let _serial = serialized();
    let Some((metal, _prompt)) = engine_ready("the capability record") else {
        return;
    };
    let caps = metal
        .capabilities()
        .expect("a loaded engine publishes its capabilities");

    assert!(
        caps.kv_copy.device_to_device,
        "this load says it cannot move kv cells between its own pages, and the verb \
         above it does exactly that — a caller reading capabilities would route a fork \
         to a copy-fallback that is not needed"
    );
    assert!(
        !caps.kv_copy.device_to_host,
        "this load promises an eviction path out of its pools, and there is none: every \
         reservation it made is `StorageModeShared` in this process and there is no swap \
         pool to write to"
    );
    assert!(
        !caps.kv_copy.host_to_device,
        "this load promises a restore path into its pools, and there is none — the same \
         missing swap pool, read the other way"
    );
    assert!(
        !caps.kv_copy.host_to_host,
        "this load promises staging between pinned buffers, which is the caller's own \
         memmove and not a verb this engine has any part in"
    );
    assert_eq!(
        caps.device.domain,
        MemoryDomain::MetalShared,
        "the served pair is `device.domain` to `device.domain`, so a load whose own \
         domain is not `MetalShared` is one whose `device_to_device: true` promises a \
         direction `Metal::copy_kv` would refuse"
    );
    assert_eq!(
        caps.pools.kv_page_size, PAGE,
        "the page size the record publishes is the one every `KvMove`'s token offset is \
         checked against, so a record that disagreed with the pools would refuse legal \
         forks and admit illegal ones"
    );
    eprintln!(
        "the record: kv_copy {:?}, domain {:?}",
        caps.kv_copy, caps.device.domain
    );
}

/// **THE FIXTURE'S OWN CONTROL, AND IT IS NOT A GATE ABOUT `copy_kv`.**
///
/// Every claim in this file diffs a row fired down one slot against a row
/// fired down another, and reads a difference as the graft's. That inference
/// is only sound if two slots handed the SAME pages and the same stated
/// `held` answer the same row — i.e. if `Seated::pages` and `Seated::held`
/// really are the whole of what a fire reads, and a slot carries no attention
/// state of its own beside them. If this fails, nothing else in this file is
/// evidence about `copy_kv`; it is evidence about `fire_seated`.
#[test]
fn two_slots_down_one_page_table_answer_one_row() {
    let _serial = serialized();
    let Some((mut shell, prompt)) = ready("the fixture's control", Runahead::F1) else {
        return;
    };
    let prefill = fire_owned(&mut shell, 0, &prompt, &PARENT, 0);
    finite(&prefill, "the control prefill");
    let token = [argmax(&prefill)];

    // No graft anywhere: both fires name the PARENT's own pages and the same
    // stated length, and differ only in which slot they ride.
    let from_zero = fire_owned(&mut shell, 0, &token, &PARENT, FIXTURE as u32);
    let from_one = fire_owned(&mut shell, 1, &token, &PARENT, FIXTURE as u32);
    assert_eq!(
        first_difference(&from_zero, &from_one),
        None,
        "the same token, the same pages and the same stated held answered two different rows \
         from slot 0 and slot 1. The page table is not the whole of what a seated fire reads, \
         so every other diff in this file is measuring the slot and not the graft."
    );
}
