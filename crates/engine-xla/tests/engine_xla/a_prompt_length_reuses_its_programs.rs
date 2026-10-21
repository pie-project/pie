//! Prompts of nearby lengths run the programs the first one traced: a prefill
//! is padded to its row rung, and the runtime's cut at the last whole page
//! before the prompt's end (where a hybrid keeps its recurrent state) leaves
//! a tail whose length reaches no program text. Nor do the slots and pages a
//! maintenance program (a slot's reset, a state or kv copy) moves: a new
//! request lands in a new slot. Over a dry device: nothing runs, every new
//! program is counted.

use std::path::Path;

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
            },
            patches: None,
            page_size: PAGE,
            context: CONTEXT,
            slots: SLOTS,
            kv_budget: Some(kv_budget),
            memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
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
    let pages: Vec<u32> = (0..n.div_ceil(PAGE)).collect();
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
        let mut seated = Seated::of(Lane {
            slot: 0,
            word: facts.word(&Request::new(to - from, false)),
            tokens: &tokens[from as usize..to as usize],
        });
        seated.pages = &pages[..to.div_ceil(PAGE) as usize];
        seated.held = Some(from);
        shell.fire_full(&[seated]).expect("the piece fires");
    }
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
    let run = |src_page, dst_page| engine::transfer::KvRun {
        src_page,
        src_token: 0,
        dst_page,
        dst_token: 0,
        tokens: PAGE,
    };
    shell.open(0).expect("the slot opens");
    shell.copy_state(0, 1).expect("the state copies");
    shell
        .copy_kv(&[run(0, 9), run(1, 10), run(2, 11)])
        .expect("the kv copies");
    let first = shell.device().dry_texts().len();
    for slot in 1..SLOTS {
        shell.open(slot).expect("the slot opens");
        shell
            .copy_state(slot, (slot + 3) % SLOTS)
            .expect("the state copies");
        let page = slot * 8;
        shell
            .copy_kv(&[
                run(page, page + 40),
                run(page + 1, page + 41),
                run(page + 2, page + 2 + 41),
            ])
            .expect("the kv copies");
        shell
            .copy_kv(&[run(page, page + 40), run(page + 3, page + 50)])
            .expect("the kv copies");
    }
    let texts = shell.device().dry_texts();
    assert_eq!(
        texts.len(),
        first + 1,
        "{} programs past the first slot's {first} and the four-run copy's one",
        texts.len() - first
    );
}
