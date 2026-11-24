//! A dry load (no weights, no device memory read) of the boot a real load
//! takes traces the real load's program texts byte for byte: the fires and
//! programs it would compile, so a build step can fill the
//! program cache ahead of the first boot. Skipped without a PJRT plugin.

use std::collections::BTreeSet;
use std::path::Path;

use engine::chunk::{SlotKind, SlotRole};
use engine_xla::{Boot, DeviceBoot, Lane, Seated, Shell};
use poem::{Platform, Request};
use poem_compiler::Budget;

const PAGE: u32 = 16;
const CONTEXT: u32 = 256;
const SLOTS: u32 = 4;

fn traced() -> poem_ir::Trace {
    let name = "qwen3-0.6b";
    let package = poem_compiler::catalog::package_of(name).expect("listed");
    let deploy = poem::star::Deploy {
        layers: Some(2),
        ..poem_compiler::catalog::one_rank(poem::Dtype::Bf16, poem::Dtype::Bf16)
    };
    package.trace(name, &deploy, name, Platform::Xla).expect("traces")
}

fn boot<'a>(
    trace: poem_ir::Trace,
    contract: &'a checkpoint::contract::ModelContract,
    checkpoint: &'a Path,
    device: &'a DeviceBoot,
) -> Boot<'a> {
    let kv_budget = crate::common::kv_budget(&trace, CONTEXT, SLOTS, SLOTS * CONTEXT / PAGE);
    Boot {
        trace,
        contract,
        checkpoint,
        budget: Budget {
            max_lanes: 8,
            max_tokens: 128,
            buckets: Vec::new(),
            max_adapters: 0,
            readouts_per_lane: poem_compiler::arena::READOUTS_PER_LANE,
        },
        patches: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: SLOTS,
        kv_budget,
        allowance: engine::Allowance::WHOLE,
        device,
        steps: None,
    }
}

/// A prompt of 20 rows, then a decode of two lanes.
fn fires(shell: &mut Shell) {
    let facts = shell.trace().facts.clone();
    let mut refs = crate::common_llm::Refs::of_facts(shell.chunk_facts());
    let kv = refs
        .shapes(|s| matches!(s.kind, SlotKind::List { window: None, .. }) && s.role == SlotRole::State)[0]
        .id;
    let pages: Vec<u32> = (0..CONTEXT / PAGE).map(|_| refs.take(kv)).collect();
    for (lanes, rows, held) in [(1usize, 20u32, 0u32), (2, 1, 20)] {
        let tokens: Vec<Vec<u32>> = (0..lanes).map(|_| vec![3; rows as usize]).collect();
        let word = facts.word(&Request::new(rows, false));
        let seated: Vec<Seated<'_>> = (0..lanes)
            .map(|at| {
                let mut s = Seated::of(Lane {
                    slot: at as u32,
                    word,
                    tokens: &tokens[at],
                });
                s.pages = &pages[..(held + rows).div_ceil(PAGE) as usize];
                s.held = Some(held);
                s
            })
            .collect();
        for at in 0..lanes as u32 {
            shell.open(at).expect("opens");
        }
        shell.fire_full(&seated).expect("fires");
    }
}

fn digests(texts: impl IntoIterator<Item = String>) -> BTreeSet<String> {
    texts
        .into_iter()
        .map(|t| blake3::hash(t.as_bytes()).to_hex().to_string())
        .collect()
}

#[test]
fn a_dry_load_traces_the_real_load_programs() {
    if crate::common_dit::device().is_none() {
        return;
    }
    let trace = traced();
    let dir = std::env::temp_dir().join(format!("pie-xla-dry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let path = dir.join("weights.zt");
    crate::common_llm::write_checkpoint(&path, &trace, 7);
    let contract = crate::common_dit::contract_for(&trace, &path).expect("a contract");
    // `PIE_XLA_DUMP` names where the real load writes its texts; read
    // before the dry load (which writes there too) opens.
    let Some(dumped) = std::env::var_os("PIE_XLA_DUMP").map(std::path::PathBuf::from) else {
        eprintln!("skipping: PIE_XLA_DUMP names no directory");
        return;
    };
    let device_boot = DeviceBoot::default();
    let real = {
        let mut shell = Shell::load(boot(trace.clone(), &contract, &path, &device_boot))
            .expect("the real load");
        fires(&mut shell);
        drop(shell);
        let texts: Vec<String> = std::fs::read_dir(&dumped)
            .expect("the dump")
            .filter_map(Result::ok)
            .filter_map(|e| std::fs::read_to_string(e.path()).ok())
            .collect();
        digests(texts)
    };
    let dry = {
        let mut shell = Shell::load_dry(boot(trace, &contract, &path, &device_boot), None, true)
            .expect("the dry load");
        fires(&mut shell);
        digests(shell.device().dry_texts())
    };
    let _ = std::fs::remove_dir_all(&dir);
    eprintln!(
        "DRY real {} program(s), dry {}, in both {}",
        real.len(),
        dry.len(),
        real.intersection(&dry).count()
    );
    assert_eq!(real, dry, "a dry load traces the real load's program texts");
}
