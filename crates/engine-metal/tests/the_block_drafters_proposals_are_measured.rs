#![cfg(target_vendor = "apple")]

use std::path::PathBuf;

use engine::fire::{Mask, Masking};
use engine_metal::{Boot, Lane, Seated, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "qwen36-27b-dflash-u4g64-kv-bf16";

const PROMPTS: &[(&str, &[u32])] = &[
    (
        "prose",
        &[9707, 11, 847, 829, 374, 264, 1602, 2613, 3364, 911, 264],
    ),
    (
        "repetition",
        &[16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220],
    ),
    (
        "code",
        &[750, 17984, 1198, 262, 470, 508, 87, 353, 220, 17, 369],
    ),
];

fn artifact() -> Option<PathBuf> {
    if let Ok(named) = std::env::var("PIE_DFLASH_ARTIFACT") {
        let path = PathBuf::from(named.replace('~', &std::env::var("HOME").unwrap_or_default()));
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

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

#[test]
fn the_target_keeps_a_measured_prefix_of_every_block() {
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

    let block = models::qwen_3::model::QWEN36_27B_DFLASH.block as usize;
    let mask_token = models::qwen_3::model::QWEN36_27B_DFLASH.mask_token;
    let drafting = |len: u32| {
        models::qwen_3::forward::Facts::of(&Request::new(len, false).drafting(true)).word()
    };
    let block_word = models::qwen_3::forward::Facts::of(
        &Request::new(block as u32, true).drafting_a_block(true),
    )
    .word();

    const ANCHORS: usize = 4;

    let mut agreed = 0usize;
    let mut asked = 0usize;
    for (name, prompt) in PROMPTS {
        shell.open(0).expect("the slot opens");
        let seeded = shell
            .fire(&[Lane {
                slot: 0,
                word: drafting(prompt.len() as u32),
                tokens: prompt,
            }])
            .expect("the prefill fires");
        let first = argmax(&seeded[0]);
        let mut truth = vec![first];
        let mut fed = first;
        for _ in 0..ANCHORS + block {
            let got = shell
                .fire(&[Lane {
                    slot: 0,
                    word: drafting(1),
                    tokens: &[fed],
                }])
                .expect("a decode step fires");
            fed = argmax(&got[0]);
            truth.push(fed);
        }

        shell.open(1).expect("the slot opens");
        shell
            .fire(&[Lane {
                slot: 1,
                word: drafting(prompt.len() as u32),
                tokens: prompt,
            }])
            .expect("the prefill fires");
        let mut marks = String::new();
        for anchor_at in 0..ANCHORS {
            if anchor_at > 0 {
                shell
                    .fire(&[Lane {
                        slot: 1,
                        word: drafting(1),
                        tokens: &[truth[anchor_at - 1]],
                    }])
                    .expect("a context step fires");
            }
            let anchor = truth[anchor_at];
            let held = u32::try_from(prompt.len() + anchor_at).expect("a context that fits");
            let mut tokens = vec![mask_token; block];
            tokens[0] = anchor;
            let extent = u64::from(held) + block as u64;
            let masking = Masking::Extent(Mask::new(
                vec![0, u32::try_from(extent).expect("an extent that fits")],
                extent,
            ));
            let mut seat = Seated::of(Lane {
                slot: 1,
                word: block_word,
                tokens: &tokens,
            });
            seat.mask = Some(&masking);
            seat.held = Some(held);
            let drafted = shell
                .fire_seated(&[seat])
                .unwrap_or_else(|why| panic!("the draft block fires at anchor {anchor_at}: {why}"));
            let guess = argmax(&drafted[0]);
            let want = truth[anchor_at + block - 1];
            asked += 1;
            if guess == want {
                agreed += 1;
                marks.push('#');
            } else {
                marks.push('.');
            }
        }
        eprintln!(
            "{name:12} {ANCHORS} anchors, the token {} ahead: {marks}  (truth head {:?})",
            block - 1,
            &truth[..truth.len().min(6)]
        );
    }

    eprintln!(
        "\nthe block's LAST row agreed with the target at {agreed} of {asked} anchors \
         — the hardest position in the block; the accepted-prefix profile needs the inferlet"
    );
    assert!(
        agreed >= 2,
        "the block's last row agreed at {agreed} of {asked} anchors, which is a \
         drafter that has lost its context rather than one that is merely wrong far ahead"
    );
}
