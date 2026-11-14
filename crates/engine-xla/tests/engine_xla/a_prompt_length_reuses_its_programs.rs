//! Prompts of nearby lengths run the programs the first one traced: a prefill
//! is padded to its row rung, and the runtime's cut at the last whole page
//! before the prompt's end (where a hybrid keeps its recurrent state) leaves
//! a tail whose length reaches no program text. Nor do the slots and pages a
//! maintenance program (a slot's reset, a state or kv copy) moves: a new
//! request lands in a new slot. Nor does a decode's lane count grow its
//! program past a round of recurrent state. Over a dry device: nothing runs,
//! every new program is counted.

use std::path::Path;

use crate::common_llm::{Refs, Seq};
use engine_xla::{Boot, DeviceBoot, Lane, Seated, Shell};
use poem::{Platform, Request};
use poem_compiler::Budget;

const PAGE: u32 = 16;
const CONTEXT: u32 = 2048;
const SLOTS: u32 = 8;

fn shell(name: &str) -> Shell {
    let deployment = poem_compiler::catalog::deployment(name).expect("the deployment is listed");
    let trace = deployment.trace(Platform::Xla);
    let paging =
        engine_xla::store::kv::Paging::of(PAGE, CONTEXT, SLOTS, u64::from(SLOTS * CONTEXT / PAGE))
            .expect("the paging lays out");
    let kv_budget = engine_xla::store::pool_demand(&trace, paging).expect("the pools size");
    let contract = checkpoint::contract::ModelContract {
        alignment: 1,
        tensors: Vec::new(),
        groups: Vec::new(),
    };
    Shell::load_dry(
        Boot {
            trace,
            contract: &contract,
            checkpoint: Path::new("/dev/null"),
            budget: Budget {
                max_lanes: 64,
                max_tokens: 2048,
                buckets: Vec::new(),
                max_adapters: 0,
                readouts_per_lane: poem_compiler::arena::READOUTS_PER_LANE,
            },
            patches: None,
            page_size: PAGE,
            context: CONTEXT,
            slots: SLOTS,
            kv_budget: Some(kv_budget),
            allowance: engine::Allowance::WHOLE,
            device: &DeviceBoot::default(),
        },
        None,
        false,
    )
    .expect("the dry shell loads")
}

/// One prompt of `n` tokens as the runtime fires it for a hybrid: the rows
/// to the last whole page, then the tail behind them (one fire when the
/// prompt ends on a page).
fn prefill(shell: &mut Shell, n: u32) {
    let facts = shell.trace().facts.clone();
    let mut refs = Refs::of_facts(shell.chunk_facts());
    let mut seq = Seq::new(0);
    let cut = n / PAGE * PAGE;
    let tokens = vec![7u32; n as usize];
    let mut pieces = vec![(0, cut)];
    if cut < n {
        pieces.push((cut, n));
    }
    if cut == 0 {
        pieces = vec![(0, n)];
    }
    shell.open(0).expect("the slot opens");
    for (from, to) in pieces {
        let word = facts.word(&Request::new(to - from, false));
        let lane = seq.lane_of(CONTEXT, word, &mut refs, vec![0; (to - from) as usize]);
        let mut seated = seated(&lane, word, &tokens[from as usize..to as usize], to);
        seated.held = Some(from);
        shell.fire_full(&[seated]).expect("the piece fires");
        seq.held = to;
    }
}

/// `lane`'s slot refs as the shell seats them, reaching `after` rows.
fn seated<'a>(
    lane: &'a engine::fire::Lane,
    word: u64,
    tokens: &'a [u32],
    after: u32,
) -> Seated<'a> {
    let mut seated = Seated::of(Lane {
        slot: lane.slot,
        word,
        tokens,
    });
    let pages = after.div_ceil(PAGE) as usize;
    seated.pages = &lane.kv.pages[..pages.min(lane.kv.pages.len())];
    seated.fixed = &lane.fixed;
    seated
}

#[test]
fn prompts_of_nearby_lengths_trace_no_new_program() {
    let mut shell = shell("qwen35-d0.8b-bf16-kv-bf16");
    prefill(&mut shell, 551);
    let first = shell.device().dry_texts().len();
    // A tail of one row is a decode fire, which runs the decode programs.
    for n in (540..=572).filter(|n| n % PAGE != 1) {
        prefill(&mut shell, n);
    }
    let texts = shell.device().dry_texts();
    assert_eq!(
        texts.len(),
        first,
        "prompts of 540 to 572 tokens traced {} programs past the first prompt's {first}",
        texts.len() - first
    );
}

#[test]
fn the_slots_a_request_lands_in_trace_no_new_program() {
    let mut shell = shell("qwen35-d0.8b-bf16-kv-bf16");
    let facts = shell.chunk_facts();
    let mut refs = Refs::of_facts(facts.clone());
    let kv = facts.shapes[0].id;
    let fixed = facts
        .shapes
        .iter()
        .find(|s| {
            matches!(s.kind, engine::chunk::SlotKind::Fixed { .. })
                && s.role == engine::chunk::SlotRole::State
        })
        .expect("a hybrid states a record")
        .id;
    let pages: Vec<u32> = (0..120).map(|_| refs.take(kv)).collect();
    let records: Vec<u32> = (0..SLOTS).map(|_| refs.take(fixed)).collect();
    let run = |src_page, dst_page| engine::transfer::KvRun {
        src_page,
        src_token: 0,
        dst_page,
        dst_token: 0,
        tokens: PAGE,
    };
    let part = |src_page, dst_page| engine::transfer::KvRun {
        src_page,
        src_token: 3,
        dst_page,
        dst_token: 3,
        tokens: 5,
    };
    shell
        .copy_slots(&[(fixed, records[0], records[1])])
        .expect("the record copies");
    shell
        .copy_kv(&[run(pages[0], pages[9]), run(pages[1], pages[10])])
        .expect("the kv copies");
    shell
        .copy_kv(&[part(pages[2], pages[11])])
        .expect("a partial page copies");
    let first = shell.device().dry_texts().len();
    for slot in 1..SLOTS as usize {
        shell
            .copy_slots(&[(fixed, records[slot], records[(slot + 3) % SLOTS as usize])])
            .expect("the record copies");
        let page = slot * 8;
        shell
            .copy_kv(&[
                run(pages[page], pages[page + 40]),
                run(pages[page + 1], pages[page + 41]),
            ])
            .expect("the kv copies");
        shell
            .copy_kv(&[part(pages[page + 2], pages[page + 42])])
            .expect("a partial page copies");
    }
    let texts = shell.device().dry_texts();
    assert_eq!(
        texts.len(),
        first,
        "{} programs past the first request's {first}",
        texts.len() - first
    );
}

/// The text of a decode of `lanes` lanes, each at 300 rows.
fn decode_text(shell: &mut Shell, lanes: u32) -> usize {
    let facts = shell.trace().facts.clone();
    let mut refs = Refs::of_facts(shell.chunk_facts());
    let token = [7u32];
    let word = facts.word(&Request::new(1, false));
    // Lanes share the seats' slots (`lane % SLOTS`), as the old test did.
    let seats: Vec<engine::fire::Lane> = (0..SLOTS)
        .map(|slot| {
            let mut seq = Seq::new(slot);
            seq.held = 300;
            seq.lane_of(CONTEXT, word, &mut refs, vec![7])
        })
        .collect();
    let held: Vec<engine::fire::Lane> = (0..lanes)
        .map(|lane| seats[(lane % SLOTS) as usize].clone())
        .collect();
    let seated: Vec<Seated<'_>> = held
        .iter()
        .map(|lane| {
            let mut seated = seated(lane, word, &token, 301);
            seated.held = Some(300);
            seated
        })
        .collect();
    shell.fire_full(&seated).expect("the decode fires");
    let texts = shell.device().dry_texts();
    texts
        .last()
        .map(String::len)
        .expect("the decode traced a program")
}

#[test]
fn a_hybrid_decode_program_does_not_grow_with_its_lanes() {
    let mut shell = shell("qwen35-d0.8b-bf16-kv-bf16");
    let at_32 = decode_text(&mut shell, 32);
    let at_64 = decode_text(&mut shell, 64);
    // Unrolled, each round of 16 lanes added its gather, step and scatter to
    // every recurrent layer: 64 lanes traced 1.6 times 32's text.
    assert!(
        at_64 * 10 <= at_32 * 11,
        "a decode of 64 lanes traced {at_64} bytes and one of 32 lanes {at_32}"
    );
}
