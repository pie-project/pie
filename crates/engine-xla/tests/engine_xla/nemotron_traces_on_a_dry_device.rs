//! Nemotron-3's programs trace on a dry device: a prompt, a decode step,
//! and a fire mixing a prompt's continuation with a fresh prompt reach no
//! op this engine refuses (its Mamba-2 scans, the conv's bias, the f32
//! router logits, the relu2 experts). Nothing runs.

use std::path::Path;

use engine_xla::{Boot, DeviceBoot, Lane, Seated, Shell};
use poem::{Platform, Request};
use poem_compiler::Budget;

const PAGE: u32 = 16;
const CONTEXT: u32 = 2048;
const SLOTS: u32 = 4;

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
            memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
            device: &DeviceBoot::default(),
        },
        None,
        false,
    )
    .expect("the dry shell loads")
}

fn fire(shell: &mut Shell, what: &str, lanes: &[(u32, &[u32], &[u32], u32)]) {
    let facts = shell.trace().facts.clone();
    let seated: Vec<Seated> = lanes
        .iter()
        .map(|&(slot, tokens, pages, held)| {
            let mut s = Seated::of(Lane {
                slot,
                word: facts.word(&Request::new(tokens.len() as u32, false)),
                tokens,
            });
            s.pages = pages;
            s.held = Some(held);
            s
        })
        .collect();
    if let Err(e) = shell.fire_full(&seated) {
        panic!("the {what} fire: {e:?}");
    }
}

fn traces(name: &str) {
    let mut shell = shell(name);
    let short = [7u32; 40];
    let long = [9u32; 288];
    let pages: Vec<u32> = (0..64).collect();
    shell.open(0).expect("the slot opens");
    fire(&mut shell, "prompt", &[(0, &short[..32], &pages[..2], 0)]);
    fire(
        &mut shell,
        "decode",
        &[(0, &short[32..33], &pages[..3], 32)],
    );
    shell.open(1).expect("the slot opens");
    fire(
        &mut shell,
        "mixed",
        &[
            (0, &short[33..38], &pages[..3], 33),
            (1, &long, &pages[40..58], 0),
        ],
    );
}

#[test]
fn nemotron_nano_traces_on_a_dry_device() {
    traces("nemotron3-nano-mini-bf16-kv-bf16");
}

#[test]
fn nemotron_ultra_traces_on_a_dry_device() {
    traces("nemotron3-ultra-mini-bf16-kv-bf16");
}
