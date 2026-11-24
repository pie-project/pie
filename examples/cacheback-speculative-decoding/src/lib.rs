//! CacheBack-style speculative decoding with a prompt-lookup draft cache.
//!
//! The drafter searches the committed token history for the longest matching
//! suffix and reuses the tokens that followed its previous occurrence. One
//! target-model forward per round verifies the whole window — the pending
//! correct token, then the drafts — and supplies the next correction.
//!
//! ## The correctness property, and how to test it
//!
//! Verification is greedy (`reduce_argmax`), and a draft token is kept only
//! when it equals what the target model would itself have produced. So
//! speculation here is a pure latency optimization: it must change *how many
//! forward passes* run and nothing else about the output.
//!
//! `draft_length = 0` makes that testable without a second inferlet.
//! `draft_from_cache` returns empty immediately, every window is one row, and
//! the loop degenerates to sequential greedy decoding — through the same
//! prompt, the same stop tokens and the same fire. Setting it to 0 versus 4
//! is therefore a controlled A/B in which the token sequence must come out
//! **identical**. If a rejected draft ever leaked into later state, the two
//! would diverge.

use inferlet::chat;
use inferlet::eta::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default)]
    prompt_tokens: Option<Vec<u32>>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_draft_length")]
    draft_length: usize,
    #[serde(default = "default_max_ngram")]
    max_ngram: usize,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    ignore_eos: bool,
    /// `rollback_mode: "ckpt_sync"` when `rollback_mode` is left out.
    #[serde(default)]
    ckpt_sync: bool,
    /// `flush_free` (the next fire overwrites the rejected slots),
    /// `ckpt_sync` (checkpoint before each verify, restore and replay the
    /// accepted rows after) or `stop_wait` (verify one row a fire). Any other
    /// name is refused.
    #[serde(default)]
    rollback_mode: Option<String>,
    /// DIAGNOSTIC: report, for every generated token, the top-1 and top-2
    /// logits of the row that produced it (`top_trace`, `margin_trace`), so a
    /// parting between two runs reads as a near-tie or a real gap, and two
    /// runs' logits can be compared row by row. A `top_k` over the vocabulary
    /// in every fire's epilogue.
    #[serde(default)]
    margin: bool,
}

fn default_prompt() -> String {
    "Repeat this pattern: red green blue, red green blue, red green".into()
}

fn default_max_tokens() -> usize {
    256
}

fn default_draft_length() -> usize {
    4
}

fn default_max_ngram() -> usize {
    8
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    /// Generated token ids. Reported so an equivalence test can compare the
    /// exact sequence rather than its detokenization, since distinct token
    /// sequences can render to the same string.
    tokens: Vec<u32>,
    prompt_tokens: usize,
    count: usize,
    draft_length: usize,
    /// Target-model forward passes actually run past the prefill. At
    /// `draft_length = 0` this is `count + stopped as usize`, which is
    /// precisely what makes that setting a sequential greedy control: same
    /// prompt, same stop tokens, same fire, no speculation. Below `count`
    /// means speculation is paying off.
    ///
    /// **THE `+ stopped` IS THE STOP TOKEN'S OWN PASS.** A run the model ends
    /// itself spends one forward pass producing the stop token, and that token
    /// is not part of the answer so it is not in `count`. A run truncated by
    /// `max_tokens` never pays it. Reading the invariant as a bare
    /// `verification_steps == count` therefore holds only on prompts long
    /// enough to run out the budget — which is a property of the model and the
    /// prompt, not of this loop.
    verification_steps: usize,
    /// Whether the model ended the run with a stop token rather than the
    /// caller's `max_tokens` doing it.
    stopped: bool,
    drafted: usize,
    accepted: usize,
    acceptance_rate: f64,
    /// With `margin`: per token, the top-1 logit of the row that chose it.
    top_trace: Vec<f32>,
    /// With `margin`: per token, top-1 minus top-2 of that row.
    margin_trace: Vec<f32>,
}

fn draft_from_cache(tokens: &[u32], draft_length: usize, max_ngram: usize) -> Vec<u32> {
    if draft_length == 0 || tokens.len() < 2 {
        return Vec::new();
    }

    let max_match = max_ngram.min(tokens.len() - 1);
    let min_w = if tokens.len() >= 16 {
        2.min(max_match)
    } else {
        1
    };
    for width in (min_w..=max_match).rev() {
        let suffix = &tokens[tokens.len() - width..];
        for start in (0..tokens.len() - width).rev() {
            if &tokens[start..start + width] != suffix {
                continue;
            }
            let continuation = start + width;
            let end = (continuation + draft_length).min(tokens.len());
            if continuation < end {
                return tokens[continuation..end].to_vec();
            }
        }
    }
    Vec::new()
}

async fn fire(
    ws: &WorkingSet,
    pipeline: &Pipeline,
    tokens: &[u32],
    base: u32,
    fold: Option<u32>,
    last_only: bool,
    margin: bool,
) -> Result<(Vec<u32>, Vec<[f32; 2]>)> {
    let rows = tokens.len() as u32;
    let total = base + rows;

    let ids = Channel::from_iter(tokens.iter().map(|&t| t as i32));
    let embed_indptr = Channel::from([0u32, rows]);
    let slots = Channel::from_iter(base..total);
    let read = if last_only { rows - 1 } else { 0 };
    let readout = Channel::from_iter(read..rows);
    let truth_out = Channel::new([rows - read], dtype::i32);
    let read_rows = rows - read;
    let top_out = margin.then(|| Channel::new([read_rows * 2], dtype::f32));
    let fold_len = fold.map(|n| Channel::from([n]));
    let fold = match &fold_len {
        Some(count) => Fold::Count(count),
        None => Fold::All,
    };

    let fwd = ForwardPass::new();
    // A verify window reads every draft row, and the next round writes the
    // rejected tail's slots again: it takes rows back (CONTRACT-BC 25).
    if !last_only {
        fwd.retractable()?;
    }
    fwd.embed(&ids, &embed_indptr)?;
    fwd.readout(&readout)?;
    fwd.state(
        &[ws],
        StateGeometry {
            slots: Some(&slots),
            fold,
            ..StateGeometry::default()
        },
    )?;
    fwd.epilogue(move || {
        let logits = intrinsics::output(0, "logits");
        truth_out.put(reduce_argmax(&logits));
        if let Some(top_out) = &top_out {
            let (top, _) = top_k(&logits, 2);
            top_out.put(reshape(top, [read_rows * 2]));
        }
    });
    fwd.submit(pipeline).context("verify-and-extend")?;

    let truth = truth_out
        .take_host::<Vec<i32>>()
        .await?
        .into_iter()
        .map(|t| t as u32)
        .collect();
    let tops = match top_out {
        Some(top_out) => top_out
            .take_host::<Vec<f32>>()
            .await?
            .chunks_exact(2)
            .map(|pair| [pair[0], pair[1]])
            .collect(),
        None => Vec::new(),
    };
    Ok((truth, tops))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rollback {
    FlushFree,
    CkptSync,
    StopWait,
}

/// The rollback the request names; a name this program does not run is
/// refused, never taken for the default.
fn rollback(input: &Input) -> Result<Rollback> {
    match input.rollback_mode.as_deref() {
        None if input.ckpt_sync => Ok(Rollback::CkptSync),
        None | Some("flush_free") => Ok(Rollback::FlushFree),
        Some("ckpt_sync") => Ok(Rollback::CkptSync),
        Some("stop_wait") => Ok(Rollback::StopWait),
        Some(other) => Err(format!(
            "unknown rollback_mode `{other}`: this program runs flush_free, ckpt_sync or stop_wait"
        )),
    }
}

/// What the decode loop needs from the target model: one greedy verify fire,
/// and the checkpoint `ckpt_sync` takes before it and restores after.
trait Target {
    /// Writes `tokens` to slots `base..`, folding `fold` survivors (all when
    /// `None`), and answers each read row's argmax — only the last row's when
    /// `last_only`.
    async fn fire(
        &mut self,
        tokens: &[u32],
        base: u32,
        fold: Option<u32>,
        last_only: bool,
    ) -> Result<Vec<u32>>;
    fn checkpoint(&mut self) -> Result<()>;
    /// Returns to the last checkpoint; `false` when none was taken.
    fn restore(&mut self) -> bool;
    /// The last fire's rows' top-1 and top-2 logits, when it reports them.
    fn tops(&mut self) -> Vec<[f32; 2]> {
        Vec::new()
    }
}

struct Live {
    ws: WorkingSet,
    pipeline: Pipeline,
    ckpt: Option<WorkingSet>,
    margin: bool,
    tops: Vec<[f32; 2]>,
}

impl Target for Live {
    async fn fire(
        &mut self,
        tokens: &[u32],
        base: u32,
        fold: Option<u32>,
        last_only: bool,
    ) -> Result<Vec<u32>> {
        let (truth, tops) = fire(
            &self.ws,
            &self.pipeline,
            tokens,
            base,
            fold,
            last_only,
            self.margin,
        )
        .await?;
        self.tops = tops;
        Ok(truth)
    }

    fn tops(&mut self) -> Vec<[f32; 2]> {
        std::mem::take(&mut self.tops)
    }

    fn checkpoint(&mut self) -> Result<()> {
        self.ckpt = Some(
            self.ws
                .fork(&self.pipeline)
                .context("ckpt_sync fork before verify")?,
        );
        Ok(())
    }

    fn restore(&mut self) -> bool {
        match self.ckpt.take() {
            Some(restored) => {
                self.ws = restored;
                true
            }
            None => false,
        }
    }
}

struct Decoded {
    generated: Vec<u32>,
    /// Per generated token, its row's top-1 and top-2 logits, when the
    /// target reports them.
    tops: Vec<[f32; 2]>,
    verification_steps: usize,
    stopped: bool,
    drafted: usize,
    accepted: usize,
}

/// The prefill, then speculative rounds until a stop token or `max_tokens`.
/// `emit(n)` hears each batch of `n > 0` newly committed tokens.
async fn decode<T: Target>(
    target: &mut T,
    input: &Input,
    prompt: &[u32],
    stop_tokens: &[u32],
    mut emit: impl FnMut(usize),
) -> Result<Decoded> {
    let w_max = input.draft_length + 1;
    let n = prompt.len() as u32;
    let mode = rollback(input)?;
    let is_ckpt_sync = mode == Rollback::CkptSync;
    let is_stop_wait = mode == Rollback::StopWait;

    let truth = target.fire(prompt, 0, None, true).await?;
    let first = *truth.last().expect("a prefill answers its last row");
    let seed_top = target.tops().last().copied();
    let mut tops: Vec<[f32; 2]> = Vec::new();

    let mut committed: Vec<u32> = prompt.to_vec();
    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let (mut verification_steps, mut drafted, mut accepted) = (1usize, 0usize, 0usize);

    // The seed token is the first thing the model said: nothing drafted it,
    // so nothing can reject it.
    let mut x = first;
    let mut stopped = stop_tokens.contains(&x);
    if !stopped {
        committed.push(x);
        generated.push(x);
        tops.extend(seed_top);
        emit(1);
    }
    let mut base = n;
    let mut survivors: u32 = 0;


    while !stopped && generated.len() < input.max_tokens {
        let prev_len = generated.len();
        // ── The window: the pending correct token, then the drafts.
        let drafts = draft_from_cache(&committed, input.draft_length, input.max_ngram);
        let mut window = Vec::with_capacity(w_max);
        window.push(x);
        window.extend(drafts.iter().copied());

        let proposed = window.len() - 1;
        drafted += proposed;

        // Each window row's top-1/top-2, row `i` the one choosing window[i + 1].
        let mut row_tops: Vec<[f32; 2]> = Vec::new();
        let (m, correction) = if is_stop_wait {
            let mut m = 0usize;
            let mut corr = 0u32;
            for idx in 0..window.len() {
                let step_truth = target
                    .fire(
                        &window[idx..=idx],
                        base + idx as u32,
                        Some(if idx == 0 { survivors } else { 1 }),
                        true,
                    )
                    .await?;
                verification_steps += 1;
                row_tops.extend(target.tops().first().copied());
                let pred = step_truth[0];
                if idx < proposed && window[idx + 1] == pred {
                    m += 1;
                } else {
                    corr = pred;
                    break;
                }
            }
            (m, corr)
        } else {
            if is_ckpt_sync {
                target.checkpoint()?;
            }
            let fold = if is_ckpt_sync { None } else { Some(survivors) };
            let truth = target.fire(&window, base, fold, false).await?;
            verification_steps += 1;
            row_tops = target.tops();

            // ── Verify: the longest matching prefix, and nothing after it.
            let mut m = 0usize;
            while m < proposed && window[m + 1] == truth[m] {
                m += 1;
            }
            if is_ckpt_sync && target.restore() {
                let _ = target.fire(&window[..=m], base, None, true).await?;
            }
            (m, truth[m])
        };
        accepted += m;

        // No explicit rollback: the next fire overwrites the rejected tail's slots.

        // ── Commit `window[1 ..= m]` — `window[0]` was committed last round as
        //    the correction that produced it — then the new correction.
        for (row, &token) in window[1..=m].iter().enumerate() {
            if stop_tokens.contains(&token) {
                stopped = true;
                break;
            }
            committed.push(token);
            generated.push(token);
            tops.extend(row_tops.get(row).copied());
            if generated.len() == input.max_tokens {
                break;
            }
        }
        if stopped || generated.len() == input.max_tokens {
            let emitted = generated.len() - prev_len;
            if emitted > 0 {
                emit(emitted);
            }
            break;
        }
        if stop_tokens.contains(&correction) {
            stopped = true;
            let emitted = generated.len() - prev_len;
            if emitted > 0 {
                emit(emitted);
            }
            break;
        }
        committed.push(correction);
        generated.push(correction);
        tops.extend(row_tops.get(m).copied());
        emit(generated.len() - prev_len);

        // Only the accepted run advances the sequence and survives in the
        // buffer; the next fire folds it ahead of its own rows.
        base += (m + 1) as u32;
        survivors = if is_stop_wait { 1 } else { (m + 1) as u32 };
        x = correction;
    }

    Ok(Decoded {
        generated,
        tops,
        verification_steps,
        stopped,
        drafted,
        accepted,
    })
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let empty = || Output {
        sampler: "cacheback-speculative",
        text: String::new(),
        tokens: Vec::new(),
        prompt_tokens: 0,
        count: 0,
        draft_length: input.draft_length,
        verification_steps: 0,
        stopped: false,
        drafted: 0,
        accepted: 0,
        acceptance_rate: 0.0,
        top_trace: Vec::new(),
        margin_trace: Vec::new(),
    };
    rollback(&input)?;
    if input.max_tokens == 0 {
        return Ok(empty());
    }

    let mut prompt = if let Some(toks) = input.prompt_tokens.clone() {
        toks
    } else {
        let mut p = chat::system_user("Continue the requested text.", &input.prompt);
        p.extend(chat::cue_without_thinking());
        p
    };
    if prompt.is_empty() {
        prompt.push(0);
    }
    let stop_tokens = if input.ignore_eos {
        Vec::new()
    } else {
        chat::stop_tokens()
    };
    let stream = input.stream;

    let mut target = Live {
        ws: WorkingSet::new(),
        pipeline: Pipeline::new(),
        ckpt: None,
        margin: input.margin,
        tops: Vec::new(),
    };
    let Decoded {
        generated,
        tops,
        verification_steps,
        stopped,
        drafted,
        accepted,
    } = decode(&mut target, &input, &prompt, &stop_tokens, |emitted| {
        if stream {
            inferlet::session::send(&format!("t:{emitted}"));
        }
    })
    .await?;

    let acceptance_rate = if drafted == 0 {
        0.0
    } else {
        accepted as f64 / drafted as f64
    };
    Ok(Output {
        sampler: "cacheback-speculative",
        text: model::decode(&generated)?,
        tokens: generated.clone(),
        prompt_tokens: prompt.len(),
        count: generated.len(),
        draft_length: input.draft_length,
        verification_steps,
        stopped,
        drafted,
        accepted,
        acceptance_rate,
        top_trace: tops.iter().map(|t| t[0]).collect(),
        margin_trace: tops.iter().map(|t| t[0] - t[1]).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A key the request does not name is refused by name, not ignored: a
    /// removed option (`bench_accept`) cannot quietly run something else.
    #[test]
    fn an_unknown_key_is_refused() {
        let why = serde_json::from_str::<Input>(r#"{"bench_accept": true}"#)
            .err()
            .expect("an unknown key is refused")
            .to_string();
        assert!(why.contains("unknown field `bench_accept`"), "{why}");
    }
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    /// A greedy model with no device: the next token is a fixed function of
    /// the three before it. Its cache is the slot array a fire writes, so a
    /// rejected draft left in a slot the next fire does not overwrite would
    /// change what it answers — exactly the leak the equivalence would catch.
    struct Toy {
        slots: Vec<u32>,
        saved: Option<Vec<u32>>,
        /// The last fire's rows, top-1 standing for the token each chose.
        tops: Vec<[f32; 2]>,
    }

    fn next(context: &[u32]) -> u32 {
        let at = |back: usize| u64::from(context[context.len().saturating_sub(back)]);
        let h = at(1) * 31 + at(2) * 17 + at(3) * 7;
        100 + (h % 5) as u32
    }

    impl Target for Toy {
        async fn fire(
            &mut self,
            tokens: &[u32],
            base: u32,
            _fold: Option<u32>,
            last_only: bool,
        ) -> Result<Vec<u32>> {
            let base = base as usize;
            assert!(base <= self.slots.len(), "a fire past the written slots");
            self.slots.truncate(base);
            self.slots.extend_from_slice(tokens);
            let rows = tokens.len();
            let from = if last_only { rows - 1 } else { 0 };
            let truth: Vec<u32> = (from..rows)
                .map(|row| next(&self.slots[..=base + row]))
                .collect();
            self.tops = truth.iter().map(|&t| [t as f32, 0.0]).collect();
            Ok(truth)
        }

        fn tops(&mut self) -> Vec<[f32; 2]> {
            std::mem::take(&mut self.tops)
        }

        fn checkpoint(&mut self) -> Result<()> {
            self.saved = Some(self.slots.clone());
            Ok(())
        }

        fn restore(&mut self) -> bool {
            match self.saved.take() {
                Some(saved) => {
                    self.slots = saved;
                    true
                }
                None => false,
            }
        }
    }

    fn ready<F: Future>(f: F) -> F::Output {
        let mut f = pin!(f);
        match f.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(out) => out,
            Poll::Pending => panic!("the toy target never waits"),
        }
    }

    fn request(max_tokens: usize, draft_length: usize, mode: &str) -> Input {
        Input {
            prompt: String::new(),
            prompt_tokens: None,
            max_tokens,
            draft_length,
            max_ngram: default_max_ngram(),
            stream: false,
            ignore_eos: false,
            ckpt_sync: false,
            rollback_mode: Some(mode.into()),
            margin: false,
        }
    }

    fn run(input: &Input, stop_tokens: &[u32]) -> Decoded {
        let mut toy = Toy {
            slots: Vec::new(),
            saved: None,
            tops: Vec::new(),
        };
        ready(decode(&mut toy, input, &[100, 101, 102], stop_tokens, |_| {})).unwrap()
    }

    /// A 128-token request with drafts says exactly what sequential greedy
    /// decoding says.
    #[test]
    fn a_long_request_with_drafts_says_what_greedy_says() {
        for mode in ["flush_free", "ckpt_sync", "stop_wait"] {
            let greedy = run(&request(128, 0, mode), &[]);
            let drafted = run(&request(128, 4, mode), &[]);
            assert_eq!(greedy.generated.len(), 128, "{mode}");
            assert!(
                drafted.accepted > 0 && drafted.accepted < drafted.drafted,
                "{mode}: speculation must both land and miss ({} of {})",
                drafted.accepted,
                drafted.drafted
            );
            assert_eq!(drafted.generated, greedy.generated, "{mode}");
        }
    }

    /// The same request honors its stop tokens: it ends where greedy ends.
    #[test]
    fn a_long_request_with_drafts_stops_where_greedy_stops() {
        let unbounded = run(&request(128, 0, "flush_free"), &[]);
        let stop = unbounded.generated[40];
        let end = unbounded.generated.iter().position(|&t| t == stop).unwrap();
        let greedy = run(&request(128, 0, "flush_free"), &[stop]);
        let drafted = run(&request(128, 4, "flush_free"), &[stop]);
        assert!(greedy.stopped && greedy.generated.len() == end);
        assert!(drafted.stopped);
        assert_eq!(drafted.generated, greedy.generated);
    }

    /// Every generated token's margin is that of the row which chose it.
    #[test]
    fn each_tokens_margin_is_its_own_rows() {
        for mode in ["flush_free", "ckpt_sync", "stop_wait"] {
            for draft_length in [0, 4] {
                let out = run(&request(128, draft_length, mode), &[]);
                let chose: Vec<u32> = out.tops.iter().map(|t| t[0] as u32).collect();
                assert_eq!(chose, out.generated, "{mode} at draft {draft_length}");
            }
        }
    }

    /// A rollback the program does not run is refused by name, not flushed.
    #[test]
    fn an_unknown_rollback_mode_is_refused() {
        let mut toy = Toy {
            slots: Vec::new(),
            saved: None,
            tops: Vec::new(),
        };
        let err = ready(decode(&mut toy, &request(8, 4, "ckpt"), &[100], &[], |_| {}))
            .err()
            .expect("an unknown rollback_mode is refused");
        assert!(err.contains("unknown rollback_mode `ckpt`"), "{err}");
        assert!(toy.slots.is_empty(), "nothing was fired");
    }
}
