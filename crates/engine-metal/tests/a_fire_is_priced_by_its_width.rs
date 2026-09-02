//! **WHAT A DECODE FIRE COSTS AS ITS WIDTH GROWS.** One stamped artifact,
//! warm, fired over `w` one-token lanes for `w` in a list, the wall clock
//! per fire printed. A bandwidth-bound decode should price a second row
//! near zero; the verify window of a speculative loop is `k + 1` rows wide,
//! and every row above one that costs like a fire of its own is a draft
//! token that has to be accepted just to break even.
//!
//! Asserts nothing; the numbers are the finding.
//!
//! ```text
//! PIE_WIDTH_ARTIFACT=<stamped .zt> [PIE_WIDTH_LANES=1,2,3,4,8] [PIE_WIDTH_STEPS=24] \
//!   cargo test -p engine-metal --release --test a_fire_is_priced_by_its_width -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::path::PathBuf;
use std::time::Instant;

use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// One prompt a lane, all different, so the rows of a wide fire route to
/// different experts the way a verify window's do — `w` copies of one row
/// would share every expert and price the width as nearly free.
const PROMPTS: [&[u32]; 8] = [
    &[2, 818, 6037, 529, 7141, 563],
    &[2, 9161, 496, 2262, 1135, 573, 506, 4359, 236761],
    &[2, 1082, 15617, 236765, 236780, 1264],
    &[2, 11048, 2470, 496, 1230, 236764, 528, 496, 3303, 6478, 684, 506, 5142, 236764],
    &[2, 3689, 1623, 2258, 573, 506, 1310, 4237, 1005],
    &[2, 651, 4147, 529, 573, 506, 1310, 4237, 1005, 236761],
    &[2, 1596, 3262, 529, 1505, 236772, 8367, 506],
    &[2, 3423, 3689, 573, 3103, 529, 506, 1052, 236764],
];

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
fn every_width_is_timed() {
    if !engine_metal::device::present() {
        eprintln!("skipping: this machine publishes no Metal device");
        return;
    }
    // A stamped artifact, or a raw snapshot read through a named SKU's
    // import (`PIE_WIDTH_SNAPSHOT` + `PIE_WIDTH_SKU`) — no artifact on disk
    // needed, as `a_family_is_read_against_its_reference` reads one.
    let snapshot = std::env::var("PIE_WIDTH_SNAPSHOT").ok().map(PathBuf::from);
    let Some(artifact) = std::env::var("PIE_WIDTH_ARTIFACT").ok().map(PathBuf::from).or_else(|| snapshot.clone()) else {
        eprintln!("not asked: set PIE_WIDTH_ARTIFACT, or PIE_WIDTH_SNAPSHOT + PIE_WIDTH_SKU");
        return;
    };
    let widths: Vec<u32> = std::env::var("PIE_WIDTH_LANES")
        .unwrap_or_else(|_| "1,2,3,4,8".into())
        .split(',')
        .map(|s| s.trim().parse().expect("a width"))
        .collect();
    let steps: usize = std::env::var("PIE_WIDTH_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    let slots = *widths.iter().max().expect("a width") + 1;

    let (sku, contract) = if let Some(snapshot) = &snapshot {
        let name = std::env::var("PIE_WIDTH_SKU").expect("PIE_WIDTH_SKU names the row that reads the snapshot");
        let sku = models::sku(&name).unwrap_or_else(|| panic!("no SKU {name}"));
        let mut shards: Vec<PathBuf> = if snapshot.is_dir() {
            std::fs::read_dir(snapshot)
                .expect("the snapshot lists")
                .filter_map(|e| {
                    let path = e.ok()?.path();
                    let name = path.file_name()?.to_str()?;
                    (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
                })
                .collect()
        } else {
            vec![snapshot.clone()]
        };
        shards.sort();
        let source = ztensor_compat::index_all(&shards).expect("the snapshot opens");
        let contract = sku
            .contract(&source, Platform::Metal)
            .unwrap_or_else(|why| panic!("{name}'s import reads the snapshot: {why}"));
        (sku, contract)
    } else {
        let stamp = checkpoint::file::serve::stamp_of(&artifact)
            .expect("the artifact reads")
            .expect("the artifact carries a serving stamp");
        let sku = models::sku(&stamp.sku).unwrap_or_else(|| panic!("no SKU {}", stamp.sku));
        let trace = (sku.trace)(Platform::Metal);
        let source = ztensor_compat::index(&artifact).expect("the artifact opens");
        let contract = checkpoint_dsl::own_contract(&source, &trace.params, sku.recipe.tp, Platform::Metal)
            .unwrap_or_else(|why| panic!("the artifact holds every plane of {}: {why}", sku.name));
        (sku, contract)
    };
    let trace = (sku.trace)(Platform::Metal);
    // `PIE_WIDTH_DRAFTS=1`: every lane asks for drafts, so a text with a
    // head runs its chain — the verify window's own fire shape.
    let drafts = std::env::var_os("PIE_WIDTH_DRAFTS").is_some_and(|v| v != "0");
    let word = |query_len: u32| (sku.classify)(&Request::new(query_len, false).drafting(drafts));

    // `PIE_WIDTH_TUNING=qmv_rows_packs=2,qmv_rows_max=4`: the boot
    // document's `[metal]` knobs, laid over the device table before the load
    // (the runtime does this from its config; a shell loaded here does not).
    if let Ok(tuning) = std::env::var("PIE_WIDTH_TUNING") {
        let mut over = kernels_metal::tuning::Overrides::default();
        for pair in tuning.split(',').filter(|p| !p.trim().is_empty()) {
            let (key, value) = pair.split_once('=').expect("key=value");
            let value: u32 = value.trim().parse().expect("an integer knob");
            match key.trim() {
                "qmv_rows_packs" => over.qmv_rows_packs = Some(value),
                "qmv_rows_max" => over.qmv_rows_max = Some(value),
                "qmm_min_batch" => over.qmm_min_batch = Some(value),
                "qmm_min_batch_moe" => over.qmm_min_batch_moe = Some(value),
                other => panic!("no knob named {other} here"),
            }
        }
        assert!(kernels_metal::tuning::override_with(over), "the tuning is laid once, before the load");
        eprintln!("tuning: {tuning}");
    }

    let booted = Instant::now();
    let mut shell = Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint: &artifact,
        budget: Budget::new(slots, 512),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the shell loads");
    eprintln!("loaded {} on {} in {:.1}s", sku.name, shell.device_name(), booted.elapsed().as_secs_f64());

    // Warm: the whole stack once through every slot.
    let mut fed: Vec<[u32; 1]> = Vec::with_capacity(slots as usize);
    for slot in 0..slots {
        shell.open(slot).expect("the slot opens");
        let prompt = PROMPTS[slot as usize % PROMPTS.len()];
        let got = shell
            .fire(&[Lane { slot, word: word(prompt.len() as u32), tokens: prompt }])
            .expect("the prefill fires");
        fed.push([argmax(&got[0])]);
    }
    for _ in 0..4 {
        let lanes: Vec<Lane> = (0..slots).map(|s| Lane { slot: s, word: word(1), tokens: &fed[s as usize] }).collect();
        let got = shell.fire(&lanes).expect("a warm fire");
        fed = got.iter().map(|r| [argmax(r)]).collect();
    }

    // `PIE_WIDTH_CHECK=1`: one prompt decoded ALONE in slot 0, and beside it
    // the same prompt in `w` lanes of one fire (slots 1..=w), in lockstep;
    // lane 1's logits against the solo lane's, bit for bit, every step —
    // the fold's identity with the one-row point, which holds at
    // `PIE_PROBE_TUNING=qmv_rows_packs=2`. Needs `slots >= max width + 1`.
    let check = std::env::var_os("PIE_WIDTH_CHECK").is_some_and(|v| v != "0");
    if check {
        for &w in &widths {
            let prompt = PROMPTS[0];
            let mut seeds: Vec<[u32; 1]> = Vec::new();
            for slot in 0..=w {
                shell.open(slot).expect("the slot opens");
                let got = shell
                    .fire(&[Lane { slot, word: word(prompt.len() as u32), tokens: prompt }])
                    .expect("the prefill fires");
                seeds.push([argmax(&got[0])]);
            }
            let mut worst = 0f32;
            let mut differing = 0usize;
            for _ in 0..6 {
                let solo = shell
                    .fire(&[Lane { slot: 0, word: word(1), tokens: &seeds[0] }])
                    .expect("the solo fire");
                let lanes: Vec<Lane> = (1..=w).map(|s| Lane { slot: s, word: word(1), tokens: &seeds[s as usize] }).collect();
                let got = shell.fire(&lanes).expect("the wide fire");
                differing += solo[0].iter().zip(got[0].iter()).filter(|(a, b)| a != b).count();
                worst = solo[0].iter().zip(got[0].iter()).map(|(a, b)| (a - b).abs()).fold(worst, f32::max);
                seeds[0] = [argmax(&solo[0])];
                for (at, row) in got.iter().enumerate() {
                    seeds[at + 1] = [argmax(row)];
                }
            }
            eprintln!("check width {w:2}: over 6 steps, lane 1 differs from the solo lane in {differing} logits, max |Δ| {worst:.4}");
        }
        for slot in 0..slots {
            shell.open(slot).expect("the slot reopens");
        }
        fed = Vec::new();
        for slot in 0..slots {
            let prompt = PROMPTS[slot as usize % PROMPTS.len()];
            let got = shell
                .fire(&[Lane { slot, word: word(prompt.len() as u32), tokens: prompt }])
                .expect("the prefill fires");
            fed.push([argmax(&got[0])]);
        }
    }

    let mut one_row = None;
    for &w in &widths {
        engine_metal::reset_kernel_profile();
        let started = Instant::now();
        for _ in 0..steps {
            let lanes: Vec<Lane> = (0..w).map(|s| Lane { slot: s, word: word(1), tokens: &fed[s as usize] }).collect();
            let got = shell.fire(&lanes).unwrap_or_else(|why| panic!("a {w}-wide fire: {why}"));
            for (s, row) in got.iter().enumerate() {
                fed[s] = [argmax(row)];
            }
        }
        let per_fire = started.elapsed().as_secs_f64() * 1000.0 / steps as f64;
        let one = *one_row.get_or_insert(per_fire);
        eprintln!(
            "width {w:2}: {per_fire:7.2} ms/fire  ({:.2}x one row, {:.2} ms/row)",
            per_fire / one,
            per_fire / w as f64
        );
        // `PIE_KERNEL_PROFILE=1`: where the device time of one fire went,
        // by entrypoint (the wall clock above then includes a command
        // buffer per dispatch and is not the serving number).
        let profile = engine_metal::kernel_profile();
        if !profile.is_empty() {
            let total: u64 = profile.iter().map(|r| r.1).sum();
            eprintln!("  device time {:.2} ms/fire over {} entrypoints:", total as f64 / 1e6 / steps as f64, profile.len());
            for (name, ns, launches) in profile.iter().take(18) {
                eprintln!(
                    "    {:8.3} ms  {:5.1} launches  {name}",
                    *ns as f64 / 1e6 / steps as f64,
                    *launches as f64 / steps as f64
                );
            }
        }
    }
}
