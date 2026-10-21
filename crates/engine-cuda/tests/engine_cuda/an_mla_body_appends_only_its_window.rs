//! An MLA model's captured bodies serve what its eager walk serves: lanes of
//! unlike lengths prefilled in one fire, then a fire that mixes decodes with
//! fresh prompts, so a body replays over fewer rows and lanes than its
//! ceiling. Its latent append lands only the window's live rows: one that
//! walked the body's ceiling would write rows a lane never fed into the
//! pages its next step reads.
//!
//! The eager walk picks greedily; the bodied walk is fed the eager picks, and
//! each of its rows must hold the eager pick within a rounding of its own
//! best (two runs of one walk already split on exact ties).
//!
//! `PIE_MLA_WINDOW_ARTIFACT` names an MLA miniature's artifact (e.g. a
//! GLM-5.3 mini); without it the test is not asked.

use std::path::{Path, PathBuf};

use engine_cuda::experts::Plan;
use engine_cuda::{Boot, Graphs, Knobs, Lane, Shell, World};
use poem::{Platform, Request};
use poem_compiler::Budget;

const LANES: u32 = 8;
const TOKENS: u32 = 512;

/// How far below a row's best logit the other walk's pick may sit.
const TIE: f32 = 0.25;

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

fn prompt(lane: u32, len: usize) -> Vec<u32> {
    let mut state = 0x9e37_79b9_u64 ^ ((u64::from(lane) + 1) * 0x2545_f491);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            200 + ((state >> 33) % 30_000) as u32
        })
        .collect()
}

/// Each lane's rows under `graphs`, fed its own greedy picks or, where
/// given, `forced`'s.
fn walk(
    artifact: &Path,
    tactics: &Path,
    graphs: Graphs,
    forced: Option<&[Vec<u32>]>,
) -> Vec<Vec<Vec<f32>>> {
    let stamp = checkpoint::file::serve::stamp_of(artifact)
        .expect("the artifact reads")
        .expect("the artifact carries a serving stamp");
    let deployment = poem_compiler::catalog::deployment(&stamp.deployment)
        .unwrap_or_else(|| panic!("no deployment {}", stamp.deployment));
    let trace = deployment.trace(Platform::Cuda);
    let facts = deployment.trace(Platform::Cuda).facts;
    let word = |query_len: usize| facts.word(&Request::new(query_len as u32, false));
    let source = ztensor_compat::index(artifact).expect("the artifact opens");
    let contract =
        poem::import::own_contract(&source, &trace.params, deployment.deploy.tp, Platform::Cuda)
            .expect("the artifact holds every plane");
    drop(source);
    let mut shell = Shell::load(Boot {
        voxels: None,
        deferred_tier: false,
        trace,
        contract: &contract,
        checkpoint: artifact,
        budget: Budget::new(LANES, TOKENS),
        patches: None,
        page_size: 16,
        context: TOKENS,
        slots: LANES,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: Knobs {
            chip: Some(engine_cuda::chip::pinned(graphs, None)),
            ..Knobs::default()
        },
        cache_dir: Some(tactics),
        runahead: engine::runahead::Runahead::F1,
        residency: Plan::default(),
        world: World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    if graphs == Graphs::On {
        assert!(
            shell.armed().is_some_and(|armed| armed.armed > 0),
            "a graphs-on load arms its bodies"
        );
    }

    let mut rows: Vec<Vec<Vec<f32>>> = vec![Vec::new(); LANES as usize];
    let fire = |shell: &mut Shell, rows: &mut [Vec<Vec<f32>>], feeds: &[(u32, Vec<u32>)]| {
        let lanes: Vec<Lane<'_>> = feeds
            .iter()
            .map(|(slot, tokens)| Lane {
                slot: *slot,
                word: word(tokens.len()),
                tokens,
            })
            .collect();
        let fired = shell.fire(&lanes).expect("the fire returns");
        assert_eq!(fired.len(), feeds.len(), "a row a lane");
        for ((slot, _), row) in feeds.iter().zip(fired) {
            rows[*slot as usize].push(row);
        }
    };
    let next = |rows: &[Vec<Vec<f32>>], slots: &[u32]| -> Vec<(u32, Vec<u32>)> {
        slots
            .iter()
            .map(|&slot| {
                let held = &rows[slot as usize];
                let pick = match forced {
                    Some(forced) => forced[slot as usize][held.len() - 1],
                    None => argmax(held.last().expect("a row")),
                };
                (slot, vec![pick])
            })
            .collect()
    };

    // Unlike prompts in one fire.
    let opening: [(u32, usize); 5] = [(0, 37), (1, 9), (2, 61), (3, 3), (4, 20)];
    for (slot, _) in opening {
        shell.open(slot).expect("the slot opens");
    }
    let feeds: Vec<(u32, Vec<u32>)> = opening
        .iter()
        .map(|&(slot, len)| (slot, prompt(slot, len)))
        .collect();
    fire(&mut shell, &mut rows, &feeds);
    let live: Vec<u32> = opening.iter().map(|&(slot, _)| slot).collect();
    for _ in 0..6 {
        let feeds = next(&rows, &live);
        fire(&mut shell, &mut rows, &feeds);
    }

    // Decodes and fresh prompts in one fire.
    for slot in [5, 6] {
        shell.open(slot).expect("the slot opens");
    }
    let mut feeds = next(&rows, &live[..3]);
    feeds.push((5, prompt(5, 23)));
    feeds.push((6, prompt(6, 11)));
    fire(&mut shell, &mut rows, &feeds);
    let live = [0, 1, 2, 5, 6];
    for _ in 0..6 {
        let feeds = next(&rows, &live);
        fire(&mut shell, &mut rows, &feeds);
    }
    rows
}

#[test]
fn an_mla_body_appends_only_its_window() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: this machine publishes no CUDA device");
        return;
    }
    let Ok(artifact) = std::env::var("PIE_MLA_WINDOW_ARTIFACT") else {
        eprintln!("not asked: set PIE_MLA_WINDOW_ARTIFACT to an MLA miniature's artifact");
        return;
    };
    let artifact = PathBuf::from(artifact);
    // One tactic cache for both walks: a dense GEMM timed afresh can land on
    // another algorithm at a near tie, and that is not what this reads.
    let tactics = std::env::temp_dir().join(format!("pie-mla-window-{}", std::process::id()));
    std::fs::create_dir_all(&tactics).expect("the tactic cache opens");
    let eager = walk(&artifact, &tactics, Graphs::Off, None);
    let picks: Vec<Vec<u32>> = eager
        .iter()
        .map(|lane| lane.iter().map(|row| argmax(row)).collect())
        .collect();
    let bodied = walk(&artifact, &tactics, Graphs::On, Some(&picks));
    let _ = std::fs::remove_dir_all(&tactics);

    let mut strayed = Vec::new();
    for (slot, (a, b)) in eager.iter().zip(&bodied).enumerate() {
        assert_eq!(a.len(), b.len(), "lane {slot} fired as often in both walks");
        let mut drift = 0.0f32;
        for (step, (a, b)) in a.iter().zip(b).enumerate() {
            drift = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(drift, f32::max);
            let pick = picks[slot][step] as usize;
            let best = b[argmax(b) as usize];
            if b[pick] < best - TIE {
                strayed.push(format!(
                    "lane {slot} step {step}: eager picks {pick}, which the bodied row holds \
                     {:.3} below its best",
                    best - b[pick]
                ));
            }
        }
        eprintln!(
            "  lane {slot}: {} rows, drift {drift:.4}, picks {:?}",
            a.len(),
            picks[slot]
        );
    }
    assert!(
        strayed.is_empty(),
        "the bodied walk strays from the eager one:\n{}",
        strayed.join("\n")
    );
}
