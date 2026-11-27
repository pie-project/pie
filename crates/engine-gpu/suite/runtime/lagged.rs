//! A lagged lane answers the folding lane (CUDA's `golden_lagged`, on this
//! shell): a prefill folded but for its last `lag` rows (whose keys and
//! corrected values the prompt's chunked step lands in the buffer), then
//! [`STEPS`] decodes of a fixed
//! script, each buffering its row and folding so that `lag` rows stay
//! unfolded behind it (a run-ahead decode whose rows fold once they are
//! accepted). From the lag on, each decode carries `lag` rows into its step
//! and commits the oldest of them; every seventh step commits them all and
//! every fifth one none. Lag 0 folds each row as it lands.
//!
//! The lagged lane's decode step reads its buffered rows as a linear
//! attention over their stored keys and corrected values from the committed
//! state (`kernels/ssm/gdn_carried.slang`), where the folding lane re-steps
//! and stores the state a row; this shell keeps the state in f32, so the two
//! forms round apart by f32 sums only. Their logits lie within a twentieth
//! of their scale (CUDA's bar, whose bf16 state rounds them further apart),
//! and no greedy token flips off a gap wider than 0.02 of the scale.
//!
//! On a backend that records its fires, the lagged decodes replay one
//! recording whatever rows they carry: the rows are the staged table's
//! data, never a recording's key.

use engine::fire::{FoldLen, Lineage, RsReset, RsVerb};

use crate::backend::{Lane, Seated, Shell};
use crate::model_logits::{CONTEXT, argmax, fixture, load, word};

/// Decode steps a lagged lane takes.
const STEPS: u32 = 30;

/// The prompt's tokens.
const PROMPT: u32 = 37;

/// The model: Qwen3.5's miniature (gated delta, 128-wide heads).
const MODEL: &str = "qwen35-tiny";

struct Served {
    steps: Vec<Vec<f32>>,
    /// Recordings the lagged decodes made, and fires they walked unkept.
    recorded: u64,
    unkept: u64,
}

fn fire(
    shell: &mut Shell,
    trace: &poem_ir::Trace,
    tokens: &[u32],
    rs: &RsVerb,
    reset: RsReset,
    held: u32,
) -> Vec<f32> {
    let seated = [Seated {
        rs,
        rs_reset: reset,
        held: Some(held),
        ..Seated::of(Lane {
            slot: 0,
            word: word(trace, tokens.len() as u32),
            tokens,
        })
    }];
    shell
        .fire_seated(&seated)
        .expect("the step fires")
        .remove(0)
}

fn served(lag: u32) -> Option<Served> {
    let fixture = fixture(MODEL, None);
    let mut shell = load(&fixture, CONTEXT)?;
    // The bodies armed before the load serves, as an engine's load arms them:
    // a serving fire fills one and replays it, recording nothing.
    shell.record_bodies(None).expect("the load arms its bodies");
    let _ = (
        shell.recorded_serving(),
        shell.unkept_serving(),
        shell.eager_serving(),
    );
    shell.serving = true;
    let page = shell
        .pools()
        .rs_rows()
        .expect("the pool lays an rs list")
        .tokens;
    shell.open(0).expect("slot 0 opens");
    let prompt: Vec<u32> = (0..PROMPT).map(|t| (t * 7 + 11) % 1000).collect();
    // A lagged lane's prompt folds all but its last `lag` rows: the prompt
    // fire lands their keys and corrected values for the decodes to read.
    let unfolded = lag.min(PROMPT);
    let verb = if lag == 0 {
        RsVerb::Fold
    } else {
        RsVerb::Buffer {
            pages: vec![0; PROMPT.div_ceil(page) as usize],
            base: 0,
            folded: 0,
            fold: FoldLen::Host(PROMPT - unfolded),
            lineage: Lineage::Set,
        }
    };
    let mut steps = vec![fire(
        &mut shell,
        &fixture.trace,
        &prompt,
        &verb,
        RsReset::Fresh,
        0,
    )];
    let (mut fed, mut folded) = (PROMPT, PROMPT - unfolded);
    // The prompt's own recordings are not the decodes'.
    let _ = (shell.recorded_serving(), shell.unkept_serving());
    for at in 0..STEPS {
        let token = (fed * 7 + 11) % 1000;
        if lag == 0 {
            steps.push(fire(
                &mut shell,
                &fixture.trace,
                &[token],
                &RsVerb::Fold,
                RsReset::Held,
                fed,
            ));
            fed += 1;
            folded += 1;
            continue;
        }
        let end = fed + 1;
        let fold = match at {
            _ if at % 7 == 6 => end - folded,
            _ if at % 5 == 4 => 0,
            _ => (end - folded).saturating_sub(lag),
        };
        // A seat-addressed lane's pages are its seat's: only their count is
        // read, from the buffer's base at 0.
        let verb = RsVerb::Buffer {
            pages: vec![0; end.div_ceil(page) as usize],
            base: 0,
            folded,
            fold: FoldLen::Host(fold),
            lineage: Lineage::Set,
        };
        steps.push(fire(
            &mut shell,
            &fixture.trace,
            &[token],
            &verb,
            RsReset::Held,
            fed,
        ));
        fed = end;
        folded += fold;
    }
    // The same prompt again, fresh, buffering another split of its rows
    // (half folded): the armed body its first fire replayed replays this
    // one too, the split the fire's data.
    if lag > 0 {
        let verb = RsVerb::Buffer {
            pages: vec![0; PROMPT.div_ceil(page) as usize],
            base: 0,
            folded: 0,
            fold: FoldLen::Host(PROMPT / 2),
            lineage: Lineage::Set,
        };
        fire(
            &mut shell,
            &fixture.trace,
            &prompt,
            &verb,
            RsReset::Fresh,
            0,
        );
    }
    Some(Served {
        steps,
        recorded: shell.recorded_serving(),
        unkept: shell.unkept_serving(),
    })
}

#[test]
fn a_lagged_lane_answers_the_folding_lane() {
    let Some(folding) = served(0) else {
        return;
    };
    let lags: Vec<u32> = std::env::var("PIE_GOLDEN_LAGS").map_or_else(
        |_| vec![1, 2, 3, 9],
        |lags| {
            lags.split(',')
                .map(|lag| lag.parse().expect("a lag"))
                .collect()
        },
    );
    for lag in lags {
        let lagged = served(lag).expect("the shell loaded once");
        assert_eq!(lagged.steps.len(), folding.steps.len());
        let mut worst = 0f32;
        let mut moved = 0usize;
        let mut flips = 0usize;
        for (got, want) in lagged.steps.iter().zip(&folding.steps) {
            let scale = want.iter().fold(0f32, |most, v| most.max(v.abs()));
            let apart = got
                .iter()
                .zip(want)
                .fold(0f32, |most, (a, b)| most.max((a - b).abs()));
            worst = worst.max(apart / scale.max(f32::MIN_POSITIVE));
            moved += usize::from(got != want);
            if std::env::var_os("PIE_GOLDEN_LAG_STEPS").is_some() {
                println!(
                    "lag {lag} step: apart {:e}",
                    apart / scale.max(f32::MIN_POSITIVE)
                );
            }
            if argmax(got) != argmax(want) {
                let mut sorted = want.clone();
                sorted.sort_by(|x, y| y.total_cmp(x));
                if (sorted[0] - sorted[1]) / scale.max(f32::MIN_POSITIVE) > 0.02 {
                    flips += 1;
                }
            }
        }
        println!(
            "{} {MODEL} lag {lag}: {} steps, {moved} differ, logits apart by {worst:e} of their \
             scale, {flips} greedy token(s) flipped off a near tie; the lagged decodes made {} \
             recording(s) and walked {} fire(s) unkept",
            crate::backend::NAME,
            lagged.steps.len(),
            lagged.recorded,
            lagged.unkept
        );
        assert!(
            worst < 0.05 && flips == 0,
            "at a lag of {lag} the lagged lane's logits lie {worst:e} of their scale from the \
             folding lane's, {flips} greedy token(s) flipped"
        );
        if crate::backend::NAME == "vulkan" {
            assert!(
                lagged.recorded == 0 && lagged.unkept == 0,
                "at a lag of {lag} the lagged decodes made {} recordings and walked {} fires \
                 unkept: the armed bodies serve every carried count",
                lagged.recorded,
                lagged.unkept
            );
        }
    }
}
