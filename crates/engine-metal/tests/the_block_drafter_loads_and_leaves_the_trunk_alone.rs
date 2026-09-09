#![cfg(target_vendor = "apple")]

use std::path::PathBuf;
use std::time::Instant;

use engine::fire::{Mask, Masking};
use engine_metal::{Boot, Lane, Seated, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "qwen36-27b-dflash-u4g64-kv-bf16";
const PLAIN_SKU: &str = "qwen36-27b-mtp-u4g64-kv-bf16";
const PROMPT: &[u32] = &[9707, 11, 847, 829, 374, 264, 1602, 2613, 3364];
const STEPS: usize = 4;

fn artifact() -> Option<PathBuf> {
    if let Ok(named) = std::env::var("PIE_DFLASH_ARTIFACT") {
        let path = PathBuf::from(shellexpand(&named));
        return path.is_file().then_some(path);
    }
    let store = PathBuf::from(std::env::var("HOME").ok()?).join(".pie/models");
    for entry in std::fs::read_dir(store).ok()?.flatten() {
        for file in std::fs::read_dir(entry.path()).ok()?.flatten() {
            let name = file.file_name().to_string_lossy().into_owned();
            if name.contains(SKU) && name.ends_with(".zt") {
                return Some(file.path());
            }
        }
    }
    None
}

fn plain_artifact() -> Option<PathBuf> {
    let store = PathBuf::from(std::env::var("HOME").ok()?).join(".pie/models");
    for entry in std::fs::read_dir(store).ok()?.flatten() {
        for file in std::fs::read_dir(entry.path()).ok()?.flatten() {
            let name = file.file_name().to_string_lossy().into_owned();
            if name.contains(PLAIN_SKU) && name.ends_with(".zt") {
                return Some(file.path());
            }
        }
    }
    None
}

fn shellexpand(path: &str) -> String {
    match path.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => path.to_string(),
    }
}

fn word(query_len: u32, drafts: bool) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false).drafting(drafts)).word()
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
    assert!(!logits.is_empty(), "{what} produced no logits");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "{what} produced a non-finite logit"
    );
    let spread = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-3,
        "{what} logits span {spread}, which nothing wrote"
    );
}

fn run(shell: &mut Shell, slot: u32, drafts: bool) -> Vec<Vec<f32>> {
    shell.open(slot).expect("the slot opens");
    let mut rows = Vec::with_capacity(STEPS + 1);
    let got = shell
        .fire(&[Lane {
            slot,
            word: word(PROMPT.len() as u32, drafts),
            tokens: PROMPT,
        }])
        .expect("the prefill fires");
    rows.push(got.into_iter().next().expect("one row"));
    for step in 0..STEPS {
        let fed = [argmax(rows.last().expect("a row"))];
        let got = shell
            .fire(&[Lane {
                slot,
                word: word(1, drafts),
                tokens: &fed,
            }])
            .unwrap_or_else(|why| panic!("decode step {step} (drafts={drafts}) fires: {why}"));
        rows.push(got.into_iter().next().expect("one row"));
    }
    rows
}

#[test]
fn the_block_drafter_loads_and_leaves_the_trunk_alone_every_case() {
    the_drafters_planes_bind_and_its_context_arm_moves_no_trunk_logit();
    a_draft_block_fires_and_the_drafter_answers_it();
}

fn the_drafters_planes_bind_and_its_context_arm_moves_no_trunk_logit() {
    if !engine_metal::device::present() {
        eprintln!("skipping: this machine publishes no Metal device");
        return;
    }
    let Some(artifact) = artifact() else {
        eprintln!("not asked: no dflash artifact (PIE_DFLASH_ARTIFACT, or one in ~/.pie/models)");
        return;
    };
    let sku = models::sku(SKU).expect("the catalog ships the block-drafter row");
    let trace = (sku.trace)(Platform::Metal);
    let source = ztensor_compat::index(&artifact).expect("the artifact opens");
    let contract = checkpoint_dsl::own_contract(&source, &trace.params, 1, Platform::Metal)
        .unwrap_or_else(|why| panic!("the artifact holds every plane of {SKU}: {why}"));
    drop(source);

    let booted = Instant::now();
    let mut shell = Shell::load(Boot {
        voxels: None,
        trace,
        contract: &contract,
        checkpoint: &artifact,
        budget: Budget::new(4, 512),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        pages: 4 * 512 / 16,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the block drafter's shell loads");
    eprintln!(
        "loaded {SKU} in {:.1}s; drafts advertised: {}, depth {}",
        booted.elapsed().as_secs_f64(),
        shell.drafts(),
        shell.mtp_depth(),
    );
    assert!(
        shell.drafts(),
        "the load carries a block drafter and advertises no draft head"
    );

    let drafted = run(&mut shell, 1, true);
    for (step, row) in drafted.iter().enumerate() {
        finite(row, &format!("drafting step {step}"));
    }
    let tokens: Vec<u32> = drafted.iter().map(|r| argmax(r)).collect();

    let Some(plain_artifact) = plain_artifact() else {
        eprintln!(
            "the drafted run decoded {tokens:?}; no plain artifact beside it, so the \
             trunk is unchecked against one"
        );
        return;
    };
    let plain_sku = models::sku(PLAIN_SKU).expect("the catalog ships the plain row");
    let plain_trace = (plain_sku.trace)(Platform::Metal);
    let plain_source = ztensor_compat::index(&plain_artifact).expect("the artifact opens");
    let plain_contract =
        checkpoint_dsl::own_contract(&plain_source, &plain_trace.params, 1, Platform::Metal)
            .expect("the plain artifact holds every plane");
    drop(plain_source);
    let mut plain_shell = Shell::load(Boot {
        voxels: None,
        trace: plain_trace,
        contract: &plain_contract,
        checkpoint: &plain_artifact,
        budget: Budget::new(4, 512),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        pages: 4 * 512 / 16,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the plain shell loads");
    let plain_word = |len: u32| {
        models::qwen_3::forward::Facts::of(&Request::new(len, false).drafting(false)).word()
    };
    plain_shell.open(0).expect("the slot opens");
    let mut plain_tokens = Vec::with_capacity(STEPS + 1);
    let got = plain_shell
        .fire(&[Lane {
            slot: 0,
            word: plain_word(PROMPT.len() as u32),
            tokens: PROMPT,
        }])
        .expect("the plain prefill fires");
    let mut fed = argmax(&got[0]);
    plain_tokens.push(fed);
    for step in 0..STEPS {
        let got = plain_shell
            .fire(&[Lane {
                slot: 0,
                word: plain_word(1),
                tokens: &[fed],
            }])
            .unwrap_or_else(|why| panic!("plain decode step {step}: {why}"));
        fed = argmax(&got[0]);
        plain_tokens.push(fed);
    }
    assert_eq!(
        tokens, plain_tokens,
        "the trunk decoded different tokens with a block drafter over it than the same \
         weights decode without one — the drafter's fusion has reached the residual stream"
    );
    eprintln!(
        "drafted run tokens {tokens:?} — the same tokens the plain artifact decodes over {STEPS} steps"
    );
}

fn all_visible(extent: u64) -> Masking {
    Masking::Extent(Mask::new(
        vec![0, u32::try_from(extent).expect("an extent that fits")],
        extent,
    ))
}

fn a_draft_block_fires_and_the_drafter_answers_it() {
    if !engine_metal::device::present() {
        eprintln!("skipping: this machine publishes no Metal device");
        return;
    }
    let Some(artifact) = artifact() else {
        eprintln!("not asked: no dflash artifact (PIE_DFLASH_ARTIFACT, or one in ~/.pie/models)");
        return;
    };
    let sku = models::sku(SKU).expect("the catalog ships the block-drafter row");
    let trace = (sku.trace)(Platform::Metal);
    let source = ztensor_compat::index(&artifact).expect("the artifact opens");
    let contract = checkpoint_dsl::own_contract(&source, &trace.params, 1, Platform::Metal)
        .expect("the artifact holds every plane");
    drop(source);
    let mut shell = Shell::load(Boot {
        voxels: None,
        trace,
        contract: &contract,
        checkpoint: &artifact,
        budget: Budget::new(4, 512),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        pages: 4 * 512 / 16,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the block drafter's shell loads");

    shell.open(0).expect("the slot opens");
    let seeded = shell
        .fire(&[Lane {
            slot: 0,
            word: word(PROMPT.len() as u32, true),
            tokens: PROMPT,
        }])
        .expect("the prefill fires");
    let anchor = argmax(&seeded[0]);

    let block = models::qwen_3::model::QWEN36_27B_DFLASH.block as usize;
    let mut tokens = vec![models::qwen_3::model::QWEN36_27B_DFLASH.mask_token; block];
    tokens[0] = anchor;
    let extent = PROMPT.len() as u64 + block as u64;
    let masking = all_visible(extent);
    let mut seat = Seated::of(Lane {
        slot: 0,
        word: models::qwen_3::forward::Facts::of(
            &Request::new(block as u32, true).drafting_a_block(true),
        )
        .word(),
        tokens: &tokens,
    });
    seat.mask = Some(&masking);
    let drafted = shell.fire_seated(&[seat]).expect("the draft block fires");

    let row = drafted.into_iter().next().expect("one readout row");
    finite(&row, "the draft block");
    eprintln!(
        "anchor {anchor} -> block of {block}; the drafter's last-row proposal is {}",
        argmax(&row)
    );
}
