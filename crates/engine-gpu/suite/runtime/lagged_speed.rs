//! The lagged fold's decode step, timed (`new-memory/designs/lagged-fold.md`,
//! the L = 0 / 1 / 2 bench): a load of Qwen3.5-0.8B's shapes (its seeded
//! checkpoint; a dense model's step does not read its weights' values)
//! decodes 1 and 64 lanes, each step buffering its row and folding the one
//! `L` back, so every step replays `L` rows; `L = 0` folds every row as a
//! plain decode does. Each step is a fire the host waits on (logits read
//! back), so it times the whole step. A measurement, not a check:
//! `-- --ignored --nocapture the_lagged_decode_step_costs`; the model is
//! `$PIE_LAG_MODEL` where set.

use engine::fire::{FoldLen, RsReset, RsVerb};

use crate::backend::{Lane, Seated};
use crate::model_logits::{PAGE, fixture, load_lanes, word};

const CONTEXT: u32 = 1024;
const PROMPT: usize = 16;
const STEPS: usize = 32;
const SETTLE: usize = 3;
const ROUNDS: usize = 5;

#[test]
#[ignore = "a measurement of the lagged decode step"]
fn the_lagged_decode_step_costs() {
    let model = std::env::var("PIE_LAG_MODEL").unwrap_or_else(|_| "qwen35-d0.8b".to_string());
    let fixture = fixture(&model, None);
    let Some(mut shell) = load_lanes(&fixture, PAGE, CONTEXT, 64, 256) else {
        return;
    };
    let carry = shell
        .decode_replay_carry()
        .expect("the load lays the decode replay");
    let per_page = shell.pools().rs_rows().expect("an rs list").tokens.max(1);
    eprintln!("{model}: carry {carry}, {per_page} tokens an rs page");
    let prompt: Vec<u32> = (0..PROMPT as u32).map(|at| (at * 37 + 11) % 1000).collect();
    let tokens: Vec<u32> = (0..64u32).map(|at| (at * 53 + 7) % 1000).collect();
    for lanes in [1usize, 64] {
        for seat in 0..lanes as u32 {
            shell.open(seat).expect("the seat opens");
            let seated = Seated {
                rs_reset: RsReset::Fresh,
                held: Some(0),
                ..Seated::of(Lane {
                    slot: seat,
                    word: word(&fixture.trace, PROMPT as u32),
                    tokens: &prompt,
                })
            };
            shell.fire_seated(&[seated]).expect("the prompt fires");
        }
        let mut have = PROMPT as u32;
        let mut folded = PROMPT as u32;
        let mut step = |shell: &mut crate::backend::Shell, lag: u32| {
            let verb = if lag == 0 && folded == have {
                RsVerb::Fold
            } else {
                let fold = (have + 1).saturating_sub(lag).saturating_sub(folded);
                RsVerb::Buffer {
                    pages: vec![0; (have + 1).div_ceil(per_page) as usize],
                    base: 0,
                    folded,
                    fold: FoldLen::Host(fold),
                    lineage: engine::fire::Lineage::Set,
                }
            };
            let seated: Vec<Seated<'_>> = (0..lanes)
                .map(|seat| Seated {
                    rs: &verb,
                    rs_reset: RsReset::Held,
                    held: Some(have),
                    ..Seated::of(Lane {
                        slot: seat as u32,
                        word: word(&fixture.trace, 1),
                        tokens: &tokens[seat..=seat],
                    })
                })
                .collect();
            let start = std::time::Instant::now();
            shell.fire_seated(&seated).expect("a decode step fires");
            let took = start.elapsed().as_secs_f64();
            folded = match &verb {
                RsVerb::Buffer {
                    fold: FoldLen::Host(fold),
                    ..
                } => folded + fold,
                _ => have + 1,
            };
            have += 1;
            took
        };
        let lags = [0u32, 1, 2];
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); lags.len()];
        for _ in 0..ROUNDS {
            for (at, &lag) in lags.iter().enumerate() {
                for _ in 0..SETTLE {
                    step(&mut shell, lag);
                }
                let total: f64 = (0..STEPS).map(|_| step(&mut shell, lag)).sum();
                times[at].push(total / STEPS as f64);
            }
        }
        let median = |v: &[f64]| {
            let mut v = v.to_vec();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let base = median(&times[0]);
        for (at, &lag) in lags.iter().enumerate() {
            let s = median(&times[at]);
            eprintln!(
                "LAGSTEP {} {model} c{lanes}: L={lag}: {:.3} ms a step, {:.1} tok/s ({:+.1}% of L=0); rounds {:?}",
                crate::backend::NAME,
                s * 1e3,
                lanes as f64 / s,
                (s / base - 1.0) * 100.0,
                times[at]
                    .iter()
                    .map(|t| format!("{:.3}", t * 1e3))
                    .collect::<Vec<_>>()
            );
        }
    }
}
