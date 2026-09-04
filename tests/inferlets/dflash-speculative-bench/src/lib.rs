//! **A BLOCK DRAFTER'S SPECULATIVE LOOP: ONE DRAFT FIRE, ONE VERIFY FIRE.**
//!
//! `dflash-block-acceptance` measures how much of a block the target keeps
//! and pays `block` ordinary decodes to learn it. This spends that number:
//! a round is TWO fires, and it emits every token the target agreed with
//! plus the one it corrected.
//!
//! ```text
//! draft    [anchor, MASK x block-1]      one fire, `set-drafting-block`
//!          -> d[0..block-1]              row i proposes position held+i
//! verify   [anchor, d[0], .., d[b-2]]    one fire, the TRUNK, plainly causal
//!          -> t[i] = argmax(row i)       the target's own token for held+i+1
//! keep     the longest prefix where d[j] == t[j], then t at the break
//! ```
//!
//! The break's token is the target's, so a round emits `kept + 1` tokens and
//! is never wrong: the sequence this writes is the sequence a one-token-a-
//! fire decode would have written. That is the whole claim a speculative
//! loop makes, and `--baseline` runs the other loop over the same prompt so
//! the ratio is measured rather than argued.
//!
//! # What it measured
//!
//! `qwen36-27b-dflash` on an M4 Pro, greedy, the same 192 tokens either way
//! (a 64-token run subtracted from a 256-token one, so the load is out of
//! the number) and the two roads token for token identical:
//!
//! ```text
//! counting  "1, 2, 3, ..."     14.00 tok/round   62.1 tok/s   vs 15.0   4.1x
//! code      "write a function" 6.14 tok/round    28.4 tok/s   vs 15.1   1.9x
//! ```
//!
//! # THOSE ARE RAW CONTINUATIONS, AND THEY ARE NOT SERVING NUMBERS
//!
//! The two lines above were measured before this file took
//! `text-completion-bench`'s envelope: no chat template, no system prompt,
//! the prompt handed to the model as raw text to continue. A served request
//! is templated, and the model then answers a QUESTION rather than
//! continuing a pattern — which is a different distribution and the drafter
//! feels it. Through `benches/pie_bench.py latency` (4 requests x 256
//! tokens, `--ignore-eos`, the same knobs the three-way uses), one arm
//! against the other on this box:
//!
//! ```text
//!                                      base    spec    ratio   tok/round
//! "Write a short story about a robot"  14.52   15.39   1.06x     3.21
//! "Write a Python function that ..."   14.58   23.95   1.64x     5.80
//! "Count from 1 to 200, ..."           14.32   21.10   1.47x     7.08
//! "List the capitals of every ..."     14.65   12.96   0.88x     2.88
//! ```
//!
//! **THE LAST ROW IS A LOSS, AND IT IS THE POINT.** A round costs roughly
//! three one-row decodes — a sixteen-row verify fire is 2.79 of them on this
//! box, and the five-layer draft fire through the target's `lm_head` is the
//! rest — so the loop pays only above about THREE TOKENS A ROUND. Prose sits
//! at 3.21 and is a wash; a list of capitals sits at 2.88 and is worse than
//! decoding one token a fire. The block width does not fix this (see the
//! staircase below): the fix is a drafter that accepts more, or a guest that
//! notices its own tokens-a-round and stops speculating. **The engine must
//! not make that call** — which rounds are worth a fire is the inferlet's.
//!
//! At 64 tokens a request the same pairs read 13.53/15.15 (1.12x),
//! 13.44/22.39 (1.67x) and 13.41/17.62 (1.31x): the ratio is set by content,
//! not by how long the answer runs.
//!
//! # THE FULL BLOCK IS THE WORST WIDTH THERE IS
//!
//! The drafter proposes fifteen and the loop verified all sixteen rows,
//! which is what every number above was taken at. Swept through
//! `benches/pie_bench.py latency --requests 4 --max-tokens 256`, that is
//! the worst choice available on all four workloads:
//!
//! ```text
//!              w=16     w=4      w=8      auto
//! prose        1.06x    1.41x    1.27x    1.29x
//! code         1.62x    1.62x    1.90x    1.80x
//! capitals     0.93x    1.30x    1.19x    1.16x
//! counting     1.47x    1.51x    1.83x    1.89x
//! mean         1.27     1.46     1.55     1.54
//! ```
//!
//! **And no fixed width dominates either**: four wins prose and the list,
//! eight wins code, and on counting the ladder beats every fixed width there
//! is (1.89x) because it climbs to sixteen once the target starts taking
//! whole windows. `auto_width` is therefore the default — it costs about 8%
//! against knowing the right width in advance (prose 1.29x where a pinned
//! four reads 1.41x) and never picks the width that loses. Pin one with
//! `verify_rows` to reproduce a row of the table.
//!
//! Why sixteen loses is the staircase and the censoring together: a
//! sixteen-row fire costs 2.79 one-row decodes and prose only ever keeps
//! about 2.8 tokens, so twelve of those rows are bought and thrown away
//! every round.
//!
//! # The waste is paid TWICE, and the draft fire pays it too
//!
//! A block drafter has no head of its own — its rows go through the TARGET's
//! `lm_head`, the widest matrix in the model — so a draft fire that reads out
//! all sixteen rows to then verify four buys twelve rows of that head and
//! throws them away, exactly as the verify side did. The width is known
//! BEFORE the draft whenever it comes from the ladder or from `verify_rows`
//! (only `margin_width` picks it after, off margins the readout would not
//! then contain), so the readout narrows with it. Measured, same shape:
//!
//! ```text
//!              readout 16   readout = width
//! prose          1.29x          1.38x
//! code           1.80x          1.82x
//! capitals       1.16x          1.32x
//! counting       1.89x          1.88x
//! ```
//!
//! **The gain tracks how narrow the width got** — prose and the list run at
//! four rows and gain 7% and 14%, counting runs at sixteen and gains nothing,
//! which is the same arithmetic read from the other end. It also closes most
//! of the ladder's gap against a pinned width: prose reads 1.38x where a
//! pinned four read 1.41x.
//!
//! # AT EIGHT CONCURRENT THE LOOP LOSES, AND IT IS NOT THE DRAFTER
//!
//! `tput --num-requests 32 --concurrency 8 --max-tokens 64`, dflash SKU:
//!
//! ```text
//! text-completion-bench, mtp SKU      47.67 tok/s
//! text-completion-bench, dflash SKU   46.27          the ctx arm costs 3%
//! this loop, --baseline               23.52          the STRUCTURE costs 49%
//! this loop, auto width               18.32   0.78x
//! ```
//!
//! **The drafter's context arm is nearly free even at concurrency** — three
//! percent against the plain SKU through the same plain inferlet — so the
//! half that is missing is this loop's own shape: it awaits a host readback
//! every round, and a wave waits on its straggler lane, so eight
//! host-in-the-loop guests each put a turnaround on the critical path.
//! `text-completion-bench` runs ahead instead and never lands one there.
//!
//! Sizing the take-side rings the way that file and `mtp-speculative-bench`
//! size theirs was tried and moved NOTHING (23.52 against 23.44): those
//! loops hoist their channels out of the round and this one builds them
//! fresh, and the ticket check is not what is binding. Closing this needs a
//! DEVICE-RESIDENT accept and commit — what `mtp-speculative-bench` gets
//! from `Lane::drafts` — not another knob on the guest, because the compare
//! that decides a round is a host compare by construction.
//!
//! So: **speculation is a single-stream lever on this box.** At one stream
//! it is worth 1.32x-1.88x; at eight concurrent plain decode wins, and the
//! serving path should pick per shape.
//!
//! # The gate, and what it costs to never lose
//!
//! `min_tokens_per_round` (see `Gate`) is the guest reading its OWN yield
//! and declining to draft when it is under water. A gated round is the same
//! verify fire at ONE row, so it carries the pending fold and buffers its
//! own row exactly as a wide one does — nothing about the fold-commit
//! contract changes. Proven rather than argued: with the floor set high
//! enough that the gate never opens, 128 tokens of the code prompt come back
//! TOKEN FOR TOKEN identical to `--baseline`.
//!
//! Swept on this box, same four-request 256-token shape, `tok/s` against a
//! `--baseline` arm of 14.52 (prose) / 14.58 (code) / 14.65 (capitals):
//!
//! ```text
//! floor      capitals        code          prose
//! off      12.96  0.88x   23.95  1.64x   15.39  1.06x
//! 2.5      14.79  1.01x   22.57  1.55x
//! 3.0      15.05  1.03x   21.32  1.46x
//! 3.5      14.28  0.97x   20.91  1.43x   15.83  1.09x
//! 2.5, WINDOW 8
//!          14.74  1.01x   22.91  1.57x   15.71  1.08x
//! ```
//!
//! **The last row is the setting this file ships**, and the trade it makes
//! is stated plainly: the loss row goes to parity and prose gains a little,
//! and the workload where drafting wins most pays 4% for it (1.64x -> 1.57x).
//! A floor ABOVE break-even is worse on both ends — 3.5 closes on code often
//! enough to cost 13% and still lands capitals under 1.0x, because a window
//! that dips is not a workload that changed. Left at zero the gate never
//! closes, which is the loop every number above this section was taken with.
//!
//! # One thing the loop does NOT promise
//!
//! Token identity with one-token-a-fire decode holds until the first
//! near-tie, not forever. A sixteen-row verify fire folds its dot products
//! in a different order than the one-row point, and on a list of European
//! capitals at 256 tokens the two roads pick different capitals at index 154
//! — the same bf16 accumulation-order floor this repo already records for
//! whole-prefill against one-token-at-a-time. Short runs (64 and 128 tokens,
//! code and prose) do come back identical, and that is what the gate above
//! was verified against.
//!
//! # The verify width is worth choosing, and nothing chose it well
//!
//! The drafter proposes fifteen whatever the target reads — a block diffusion
//! model is out of distribution at any other block width — but the VERIFY
//! fire is priced by its rows, and the price is a STAIRCASE: the tile point
//! pads a fire up to a row block, so twelve rows cost what sixteen do and
//! twenty-four cost MORE than thirty-two. `a_fire_is_priced_by_its_width` on
//! this box, in one-row fires: 1.00 / 1.83 / 2.79 / 5.19 / 5.00 at
//! 1 / 8 / 16 / 24 / 32 rows. 256 tokens, wall clock:
//!
//! ```text
//!                      v=4     v=6     v=8     v=12    v=16
//! counting            9.96s   8.51s   7.14s   8.06s   6.37s
//! code                11.26   10.87    9.83   13.77   11.18
//! ```
//!
//! So the best width is a property of the WORKLOAD — sixteen where the block
//! is nearly all kept, eight where five tokens are — and it is worth up to
//! 20%. THREE rules to pick it per round were written and all three are a
//! wash, which is why the parameter is still the caller's to state:
//!
//! - **A smoothed accepted prefix** against the stated prices RATCHETS: at
//!   eight rows a prefix can never be observed above seven, so nothing ever
//!   argues for going back to sixteen. Recall went 10.37s to 15.19.
//! - **A two-armed bandit on the guest's own clock**, tokens per nanosecond
//!   with a probe floor on each arm, picked the losing arm on two of three
//!   workloads (code 10.67s against 9.31 fixed; recall 14.87 against 10.37).
//!   A run that MIXES widths keeps less per round than either pure run: the
//!   round boundaries move, so the arms do not sample one distribution.
//! - **The drafter's own logit margin** (`--margin_width`), the signal Dspark
//!   spends a trained confidence head on. Pooled it separates cleanly — over
//!   sixty anchors on five prompts a position inside the accepted prefix
//!   carries a margin of 6.86 and one outside 1.10, and the first seven
//!   positions' mean reads 7.95 where the prefix reached eight against 2.16
//!   where it did not. It is MISCALIBRATED exactly where the width matters:
//!   on recall the drafter is unsure and RIGHT — a factual token has many
//!   plausible rivals — so it sends long rounds to the narrow width. At a
//!   threshold of 4 recall goes 10.74s to 15.38; pulled back to 2 the three
//!   workloads come to 28.73s against a fixed sixteen's 29.15 and an oracle's
//!   27.27.
//!
//! What survives is the MEASUREMENT, not a policy. The margin costs 9% of a
//! round (3.28s against 3.01 over 64 tokens) because `top_k(logits, 2)` reads
//! the plane ONCE and returns the runner-up beside the proposal. Two consumers
//! of that plane cost about a second a round — 3.0s to 7.4s for a bare second
//! reduction, 46s with a softmax on it — which is the thing to know before
//! reaching for any other signal off a fire's logits.
//!
//! # A candidate tree does not pay here, and the arithmetic says so
//!
//! The drafter denoises every position from the SAME context in one pass, so
//! its guess at `j + 1` does not depend on its guess at `j` — a second
//! candidate at one position costs a parallel chain, not a second draft.
//! Branching at `j` lays out `j + 2 * (block - j)` rows: 31 for `j = 1`.
//!
//! At the break, the target's token was the drafter's RUNNER-UP 14 times in
//! 45 (31%), and `top_k` already carries it. But 31 rows are 5.00 + 0.27 =
//! 5.27 one-row fires against the linear block's 3.09, so the tree must
//! return 14.1 tokens a round where the block returns 8.27 — a prefix of 13.1
//! against 7.27. Recovering a third of the breaks is worth about one token.
//! Not close, and the reason is the width curve: on a GPU where sixteen rows
//! cost two one-row fires rather than 2.79, the same tree would be arguable.
//!
//! # The recurrence is a fold, not a cell
//!
//! This SKU is hybrid, so a verify fire that folded its rows would fold the
//! REJECTED ones too and no `kv_len` could take them back —
//! `rs-speculative-decoding`'s header is the argument in full. So the verify
//! window is BUFFERED (`fold-len` leaves it unfolded), the rejected tail is
//! forgotten with `discard-buffered`, and the NEXT round's verify folds
//! exactly the accepted prefix ahead of its own rows.
//!
//! The draft fire in between is the trap: a fire must bind one recurrent
//! working set per request and is CHARGED ITS ROWS in the buffer, even though
//! the trunk that owns the recurrence runs over none of a draft fire's rows.
//! Binding a scratch working set instead corrupts the sequence's state; the
//! fix is to fold nothing and forget the fire's own rows again, which leaves
//! the buffer exactly as it was found.
//!
//! **THE KV ROLLBACK IS THE SUBTLE PART.** The verify fire writes a row's
//! keys at `held + i` for all `block` rows, and the rows past the break
//! carried tokens the target rejected — so the next round states
//! `kv_len = held + kept + 1` and the stale rows are simply never read.
//! The drafter's own context rides the same rows (its context arm runs on
//! every trunk fire), so one length rolls both streams back.

use inferlet::eta::hybrid::prelude::*;
use inferlet::{chat, session};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    /// Pre-tokenized prompt (the harness's, template already applied); wins
    /// over `prompt` when given.
    #[serde(default)]
    prompt_tokens: Option<Vec<u32>>,
    #[serde(default = "default_system")]
    system: String,
    /// Drop the template's stop tokens so the loop runs to `max_tokens`.
    #[serde(default)]
    ignore_eos: bool,
    /// Announce `ready` and wait for the harness's `start` before the clock.
    #[serde(default)]
    wait_for_start: bool,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Decode one token a fire instead — the same sequence by a different
    /// road, and the denominator of the speedup.
    #[serde(default)]
    baseline: bool,
    /// **HOW MANY OF THE BLOCK'S ROWS TO VERIFY**, the anchor's included.
    /// Zero (the default) verifies all of them.
    ///
    /// The drafter is trained at one block width and proposes fifteen
    /// whatever this says — a block diffusion model is out of distribution
    /// at any other width — but the TARGET need not read them all, and a
    /// verify fire is priced by its rows: sixteen cost 2.82 one-row fires on
    /// this box, eight cost 1.83. A workload whose accepted prefix rarely
    /// reaches eight pays for ten rows it was never going to keep. Which
    /// rows are worth a fire is the guest's call, not the engine's.
    #[serde(default)]
    verify_rows: u32,
    /// Let the drafter's own logit margin choose the width per round, rather
    /// than verifying the whole block — see `wide_enough`. Off by default,
    /// because on three workloads it is a WASH.
    #[serde(default)]
    margin_width: bool,
    /// **STOP DRAFTING BELOW THIS MANY TOKENS A ROUND.** Zero (the default)
    /// never stops, which is the loop every number in the header was taken
    /// with. Around 3.0 is break-even on this box; see `Gate`.
    #[serde(default)]
    min_tokens_per_round: f64,
    /// **LET THE LOOP'S OWN YIELD PICK THE VERIFY WIDTH.** See `Gate::width`.
    /// **ON by default**, because the full block is the WORST width on every
    /// workload measured — see the table in the header. `verify_rows` pins a
    /// width and wins over this; `--verify_rows 16` is the loop every number
    /// above the width section was taken with.
    #[serde(default = "default_true")]
    auto_width: bool,
}

fn default_prompt() -> String {
    "The quick brown fox jumps over".into()
}

fn default_max_tokens() -> usize {
    64
}

fn default_true() -> bool {
    true
}

/// **A TAKE-SIDE RING WITH A FRAME OF MARGIN ABOVE THE ADVERTISED
/// CAPACITY**, as `text-completion-bench` and `mtp-speculative-bench` size
/// theirs: sized at exactly `channel_capacity()` the runtime's ticket check
/// skips continuations that land inside its staging margin and the run-ahead
/// collapses.
///
/// **MEASURED NEUTRAL HERE**, and kept for the convention rather than for a
/// gain: 23.52 against 23.44 tok/s at eight concurrent. This loop's channels
/// are built fresh a round rather than hoisted like the mtp loop's, and its
/// concurrency wall is elsewhere — see the header.
fn ring() -> u32 {
    (channel_capacity() + 7 * live_slots()) as u32
}

fn default_system() -> String {
    "You are a helpful benchmarking assistant.".into()
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    tokens: Vec<u32>,
    count: usize,
    /// `text-completion-bench`'s envelope, for `benches/pie_bench.py`.
    num_prompt_tokens: usize,
    num_output_tokens: usize,
    token_ids: Vec<u32>,
    /// Draft+verify fire pairs after the prefill; zero under `--baseline`.
    rounds: usize,
    /// Proposals made, `block - 1` a round.
    drafted: usize,
    /// Proposals the target's own argmax agreed with.
    accepted: usize,
    /// `accepted / drafted`, or zero when nothing was drafted.
    acceptance_rate: f64,
    /// Tokens emitted per round, the number the economics turns on: a round
    /// costs two fires whatever this is.
    tokens_per_round: f64,
    /// The block width this run used.
    block: u32,
    /// The widths the rounds chose, in order.
    verify: Vec<u32>,
    /// Rows replayed through the buffer read path, summed over the rounds —
    /// a run whose `replayed` is zero exercised none of the fold-commit path.
    replayed: usize,
    /// Rows the buffer forgot because the target rejected them.
    discarded: usize,
    /// The recurrent buffer's page width, for reading the grant arithmetic.
    rs_page: u32,
    /// Fires the gate spent on one token because a draft was not worth it.
    /// Zero unless `min_tokens_per_round` was stated.
    plain_fires: usize,
}

/// **IS A DRAFT WORTH ITS FIRE AT ALL? THE LOOP'S OWN YIELD SAYS.**
///
/// A round costs about three one-row decodes on this box — a sixteen-row
/// verify fire is 2.79 of them and the five-layer draft fire, which still
/// goes through the target's `lm_head`, is the rest — so a workload that
/// yields fewer than about three tokens a round is SLOWER than decoding one
/// token a fire. Measured through `benches/pie_bench.py`, a list of European
/// capitals yields 2.88 and runs at 0.88x; prose yields 3.21 and is a wash.
///
/// This reads the loop's own recent yield and stops drafting when it falls
/// under a floor the CALLER states, then probes again after a run of plain
/// fires. **The floor is the guest's number, not the engine's** — the engine
/// dispatches the fires it is given and has no opinion about which rounds
/// were worth one. Left at zero, the gate never closes and the loop is the
/// one every measurement above was taken with.
struct Gate {
    /// Tokens a round below which drafting stops. Zero never closes.
    floor: f64,
    /// What the last rounds that DID draft emitted, newest last, each with
    /// whether the target took the WHOLE window — see `Gate::width`.
    recent: Vec<(u32, bool)>,
    /// The width the last drafting round used, the ladder's starting rung.
    last: u32,
    /// Plain fires since the gate closed, counted towards the next probe.
    since: u32,
    /// Whether the verify width follows the yield. Off leaves it at `block`.
    auto: bool,
}

impl Gate {
    /// Rounds averaged before the gate will act. Fewer reads noise as a
    /// verdict; more spends the fires it is trying to save.
    const WINDOW: usize = 8;
    /// Plain fires between probes. One probe costs a round; at sixteen the
    /// probe is under 7% of the fires it is deciding about.
    const PROBE: u32 = 16;
    /// **THE WIDTHS WORTH BUYING, AND THERE ARE ONLY THREE.** A verify fire is
    /// priced by its rows as a STAIRCASE — the vector fold answers 1-3 rows
    /// for about what one costs, 4 steps to 1.38, 5-8 share the tile's first
    /// step at 1.67-1.83, and 9-16 all cost 2.79 — so a width between rungs
    /// pays the rung above it and keeps nothing extra. Twelve rows measured
    /// 0.81x where sixteen measured 0.88x, which is the staircase, not noise.
    ///
    /// **THE LADDER STARTS AT FOUR AND NEVER GOES BELOW IT.** Two rows are
    /// nearly free but yield about 1.7 tokens a round against four rows' 2.8,
    /// and 1.7/1.00 loses to 2.8/1.38: measured, letting prose sink to two on
    /// a sixth of its rounds cost 1.41x -> 1.30x.
    const RUNGS: [u32; 3] = [4, 8, 16];

    fn new(floor: f64, auto: bool) -> Self {
        Gate { floor, recent: Vec::new(), last: 0, since: 0, auto }
    }

    /// Whether this round drafts. A probe is allowed through so a workload
    /// that turns list-shaped into code-shaped is noticed.
    fn drafts(&mut self) -> bool {
        if self.floor <= 0.0 || self.recent.len() < Self::WINDOW {
            return true;
        }
        if self.mean() >= self.floor {
            return true;
        }
        if self.since >= Self::PROBE {
            self.since = 0;
            return true;
        }
        self.since += 1;
        false
    }

    /// **VERIFY AS MANY ROWS AS THE LOOP HAS BEEN KEEPING, ROUNDED UP TO A
    /// RUNG.** The signal is the loop's own tokens a round, which is exactly
    /// the accepted prefix plus the anchor — no prediction, no second pass
    /// over a logits plane, just what the last rounds actually did.
    ///
    /// The comparison is STRICT, so a width the yield has saturated steps up
    /// rather than pinning the loop under its own ceiling: at width four a
    /// mean of 4.0 asks for eight, and the loop climbs back on a workload
    /// that turns code-shaped. Measured against the shipped sixteen on this
    /// box, a fixed width picked this way is the per-workload optimum in all
    /// three cases (`prose` 4, `code` 8, `capitals` 4).
    fn width(&self, block: u32) -> u32 {
        if !self.auto {
            return block;
        }
        // **START CHEAP AND CLIMB, RATHER THAN WIDE AND FALL.** With no
        // history the ladder has nothing to read, and eight rounds at the
        // full block is eight fires at 2.79 where the first rung costs 1.38
        // — on a 91-round request that alone was 9% of the rounds spent at
        // the most expensive width the loop offers.
        if self.recent.len() < Self::WINDOW {
            return Self::RUNGS[0].min(block);
        }
        // **A WINDOW THE TARGET TOOK WHOLE SAYS NOTHING ABOUT HOW MUCH MORE
        // IT WOULD HAVE TAKEN.** The yield is CENSORED at the width, so
        // reading the mean alone is a ratchet: narrow once and the mean can
        // never again exceed the width that produced it, and a request that
        // turns code-shaped stays pinned at four rows. Measured: without
        // this the code prompt sits at width 4 and 1.62x where width 8 is
        // 1.86x. So a window that saturated asks for the NEXT RUNG UP
        // instead, and the ladder climbs until the target stops taking
        // everything.
        let saturated = self.recent.iter().filter(|(_, whole)| *whole).count();
        let over = f64::from(self.last);
        // Three quarters, not half: a rung is left only on strong evidence,
        // because the step up costs 1.5-2x and a window that merely brushed
        // its ceiling has not earned it. At half the ladder reached sixteen
        // on a fifth of the prose rounds and gave back most of the win.
        let want = if saturated * 4 >= self.recent.len() * 3 { over } else { self.mean() };
        Self::RUNGS
            .iter()
            .copied()
            .find(|w| f64::from(*w) > want)
            .unwrap_or(block)
            .min(block)
    }

    fn mean(&self) -> f64 {
        self.recent.iter().map(|(t, _)| f64::from(*t)).sum::<f64>() / self.recent.len() as f64
    }

    /// What a drafting round emitted, the anchor's token included, and
    /// whether the target kept every row it was shown.
    fn yielded(&mut self, tokens: u32, width: u32, whole: bool) {
        self.last = width;
        self.recent.push((tokens, whole));
        if self.recent.len() > Self::WINDOW {
            self.recent.remove(0);
        }
    }
}

/// **IS THIS ROUND WORTH A WIDE VERIFY? THE DRAFTER ALREADY SAID.**
///
/// The signal is the LOGIT MARGIN, top-1 less top-2, because that is what
/// comes free: `top_k` reads the logits plane once and returns both, where a
/// probability would want a softmax over the whole vocabulary and a second
/// pass over the plane.
///
/// A verify fire's price is a staircase — the tile point pads a fire's rows up
/// to a row block, so a round costs 2.10 one-row fires at eight rows and 3.09
/// at sixteen — and the question is whether the prefix will reach eight. The
/// drafter's own top-1 probability answers it: over sixty anchors on five
/// prompts, a position INSIDE the accepted prefix carries 0.936 and one
/// outside 0.383, and the first seven positions' mean reads 0.959 where the
/// prefix reached eight against 0.553 where it did not. At this threshold
/// that calls the wide round on 28 of 30 long anchors and on 2 of 30 short
/// ones.
///
/// This is the signal Dspark spends a trained confidence head on. A block
/// drafter that reads out through the target's own `lm_head` has it for one
/// reduction over a plane the fire already computed.
fn wide_enough(confidence: &[f32]) -> bool {
    const THRESHOLD: f64 = MARGIN_THRESHOLD;
    let head = &confidence[..confidence.len().min(NARROW as usize - 1)];
    let mean = head.iter().map(|c| f64::from(*c)).sum::<f64>() / head.len().max(1) as f64;
    mean >= THRESHOLD
}

/// The narrow rung, the only other width whose price differs (twelve rows
/// cost what sixteen do).
const NARROW: u32 = 8;

/// Where the margin separates a round that will reach eight from one that
/// will not — fitted offline, `scratchpad/dflash_ref/confidence.py`.
const MARGIN_THRESHOLD: f64 = 2.0;

/// The pages the recurrent buffer must hold for one fire: the survivors (at
/// most a page's worth of head offset before them), plus the window.
/// A fold releases only the WHOLE head pages its prefix covers and rebases
/// `buffer_head` by the remainder, so a run whose folds are not page-aligned
/// keeps a head offset the grant has to sit above — hence the extra page on
/// top of the survivors and the window.
fn buffer_pages_for(survivors: u32, window: u32, page: u32) -> u32 {
    (page.saturating_sub(1) + survivors + window)
        .div_ceil(page.max(1))
        .max(1)
        + 1
}

/// Rows one draft block carries — the model text's `DFLASH_BLOCK`. A guest
/// cannot read it off the load yet (`mtp_depth` advertises the seam's DEPTH,
/// which is one: a block drafter plants one proposal a ROW), so it is stated
/// in one place until the load advertises it.
const BLOCK_ROWS: u32 = 16;

/// The drafter's own mask token, `dflash_config.mask_token_id`. Stated here
/// for the same reason and with the same caveat.
const MASK_TOKEN: i32 = 248_070;

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::pass_kind() != model::ForwardKind::Hybrid {
        return Err("this inferlet drives a hybrid model's recurrent state".into());
    }
    if model::mtp_depth() == 0 {
        return Err("this SKU ships no draft head".into());
    }
    let block = BLOCK_ROWS;
    let pinned = (input.verify_rows != 0).then(|| input.verify_rows.clamp(2, block));
    let page_size = kv_page_size();
    let rs_page = model::rs_buffer_page_size().max(1);
    // The harness's tokens when it sent them; else the chat template the
    // plain bench applies (system + user + the assistant cue). Both benches
    // must see the same prompt or the ratio is against a different prefill.
    let mut prompt: Vec<u32> = match &input.prompt_tokens {
        Some(tokens) => tokens.clone(),
        None => {
            let mut p = chat::system_user(&input.system, &input.prompt);
            p.extend(chat::cue());
            p
        }
    };
    if prompt.is_empty() {
        prompt.push(0);
    }
    let stop_tokens: Vec<u32> = if input.ignore_eos { Vec::new() } else { chat::stop_tokens() };
    if input.wait_for_start {
        session::send("ready");
        let _ = session::receive().await;
    }
    let n = prompt.len() as u32;
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();

    // The prompt, every token the loop commits, and one block of speculative
    // rows above the committed length that the next fire writes over.
    let ws = WorkingSet::new();
    let max_extent = n + input.max_tokens as u32 + 2 * block;
    let max_pages = max_extent.div_ceil(page_size);
    ws.reserve(max_pages).context("reserve KV")?;
    let pool = max_pages * page_size;
    let rs = RsWorkingSet::new();
    let rs_set = vec![rs];

    let pipe = Pipeline::new();

    // ── PREFILL: the prompt, chunked. Every chunk leaves the drafter's
    //    context behind it, because the context arm rides every trunk fire.
    let mut anchor: i32 = 0;
    for &(base, end) in &prefill_chunks(n, None) {
        let len = end - base;
        let toks = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
        let indptr = Channel::from([0u32, len]).named("embed_indptr_p");
        let positions = Channel::from_iter(base..end).named("positions_p");
        let pages = Channel::from_iter(0..max_pages).named("pages_p");
        let page_indptr = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_p");
        let w_slot = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_p");
        let w_off = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_p");
        let kv_len = Channel::from([end]).named("kv_len_p");
        let readout = Channel::from([len - 1]).named("readout_p");
        let next = Channel::new([1], dtype::i32).named("next_p");

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &indptr)?;
        fwd.readout(&readout)?;
        fwd.attention(
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: ..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: None,
                },
            }),
            &rs_set,
            RsGeometry { fold_len: None, buffer: 0..0 },
        )?;
        fwd.epilogue(move || {
            next.put(&reshape(reduce_argmax(intrinsics::logits()), [1]));
        });
        fwd.submit(&pipe).context("prefill submit")?;
        anchor = next.take_host::<Vec<i32>>().await.context("prefill readback")?[0];
    }

    // ── THE LOOP ────────────────────────────────────────────────────────
    // `held` is the committed length: positions `0..held` are in the cache
    // and `anchor` is the token AT position `held`, not yet written.
    let mut held = n;
    let mut generated: Vec<u32> = vec![anchor as u32];
    let mut rounds = 0usize;
    let mut drafted = 0usize;
    let mut accepted = 0usize;
    let mut replayed = 0usize;
    let mut discarded = 0usize;
    // Tokens sitting in the recurrent buffer unfolded. The prefill folded
    // everything and the anchor is the first window's own row, so the first
    // round replays nothing.
    let mut survivors: u32 = 0;
    let mut widths: Vec<u32> = Vec::new();
    let mut gate = Gate::new(input.min_tokens_per_round, input.auto_width);
    let mut plain_fires: usize = 0;

    // A round commits several tokens at once, so the stop is read off the
    // committed run after the fact rather than one token at a time.
    while generated.len() < input.max_tokens
        && !generated.iter().any(|t| stop_tokens.contains(t))
    {
        if input.baseline {
            // One token a fire, the road the speedup is measured against.
            let toks = Channel::from([anchor]).named("toks_b");
            let indptr = Channel::from([0u32, 1]).named("embed_indptr_b");
            let positions = Channel::from([held]).named("positions_b");
            let pages = Channel::from_iter(0..max_pages).named("pages_b");
            let page_indptr =
                Channel::from([0u32, (held + 1).div_ceil(page_size)]).named("page_indptr_b");
            let w_slot = Channel::from([held / page_size]).named("w_slot_b");
            let w_off = Channel::from([held % page_size]).named("w_off_b");
            let kv_len = Channel::from([held + 1]).named("kv_len_b");
            let next = Channel::new([1], dtype::i32)
                .capacity(ring())
                .named("next_b");

            let fwd = ForwardPass::new();
            fwd.embed(&toks, &indptr)?;
            fwd.attention(
                Some(KvBinding {
                    working_set: &ws,
                    geometry: KvGeometry {
                        readable_pages: ..,
                        writable_pages: ..,
                        kv_len: &kv_len,
                        pages: &pages,
                        page_indptr: &page_indptr,
                        w_slot: &w_slot,
                        w_off: &w_off,
                        positions: &positions,
                        mask: None,
                    },
                }),
                &rs_set,
                RsGeometry { fold_len: None, buffer: 0..0 },
            )?;
            fwd.epilogue(move || {
                next.put(&reshape(reduce_argmax(intrinsics::logits()), [1]));
            });
            fwd.submit(&pipe).context("baseline submit")?;
            anchor = next.take_host::<Vec<i32>>().await.context("baseline readback")?[0];
            held += 1;
            generated.push(anchor as u32);
            continue;
        }

        // The width is chosen AFTER the draft fire, from what the drafter
        // itself says — see `wide_enough`. Pinned, it is stated here.
        let mut verify = pinned.unwrap_or_else(|| gate.width(block));
        // The buffer must hold the survivors and this window; the grant is
        // the guest's one allocation decision.
        let buffer_pages = buffer_pages_for(survivors, block, rs_page);
        let have = rs_set[0].buffer_size();
        if have < buffer_pages {
            rs_set[0]
                .alloc_buffer(buffer_pages - have)
                .map_err(|why| format!("alloc {} rs buffer page(s): {why}", buffer_pages - have))?;
        }
        let fold_none = Channel::from([0u32]).named("fold_none");
        let fold_len = Channel::from([survivors]).named("fold_len_v");

        // ── the gate: a round the loop is losing on drafts nothing ──────
        //    A gated round is the SAME verify fire at one row, so it carries
        //    the pending fold and buffers its own row exactly as a wide one
        //    does. Nothing about the fold-commit contract changes; only the
        //    width, and whether a draft fire ran before it.
        let drafting = gate.drafts();
        let mut proposals_owned: Vec<i32> = Vec::new();
        if !drafting {
            verify = 1;
            plain_fires += 1;
        } else {
        // ── the draft: ONE pass over `[anchor, MASK x block-1]` ──────────
        let mut ids = vec![MASK_TOKEN; block as usize];
        ids[0] = anchor;
        let toks = Channel::from(ids.as_slice()).named("toks_d");
        let indptr = Channel::from([0u32, block]).named("embed_indptr_d");
        let positions = Channel::from_iter(held..held + block).named("positions_d");
        let pages = Channel::from_iter(0..max_pages).named("pages_d");
        let page_indptr =
            Channel::from([0u32, (held + block).div_ceil(page_size)]).named("page_indptr_d");
        let w_slot = Channel::from_iter((held..held + block).map(|p| p / page_size)).named("w_slot_d");
        let w_off = Channel::from_iter((held..held + block).map(|p| p % page_size)).named("w_off_d");
        let kv_len = Channel::from([held + block]).named("kv_len_d");
        // The drafter's full-attention layer is BIDIRECTIONAL over the
        // block, which only a stated mask says; every key up to the block's
        // end is visible to every block row.
        let visible: Vec<bool> = (0..block)
            .flat_map(|_| (0..pool).map(move |j| j < held + block))
            .collect();
        let mask = Channel::from_shaped([block, pool], visible).named("mask_d");
        // **ONLY THE ROWS THE VERIFY WILL READ.** A block drafter's rows go
        // through the TARGET's one `lm_head`, so there is no separate draft
        // plane — and that head is the widest matrix in the model. Reading
        // all sixteen rows of it to then verify four is the same waste the
        // width staircase found on the verify side, paid a second time.
        //
        // The width is known BEFORE this fire whenever it comes from the
        // ladder or from `verify_rows`; only `margin_width` picks it after,
        // off margins this readout would otherwise not contain, so that one
        // still reads the block.
        let shown = if input.margin_width { block } else { verify };
        let readout = Channel::from_iter(0..shown).named("readout_d");
        let out = Channel::new([shown * 2], dtype::i32)
            .capacity(ring())
            .named("drafts_d");
        // **THE DRAFTER'S OWN CONFIDENCE, OFF THE SAME LOGITS.** One more
        // reduction over a plane the fire already computed, read back beside
        // the proposals in the same round trip.
        let conf = Channel::new([shown * 2], dtype::f32)
            .capacity(ring())
            .named("conf_d");

        let fwd = ForwardPass::new();
        fwd.set_drafting_block(true)
            .map_err(|why| format!("stating the draft block: {why}"))?;
        fwd.embed(&toks, &indptr)?;
        fwd.readout(&readout)?;
        fwd.attention(
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: ..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: Some(&mask),
                },
            }),
            &rs_set,
            // **FOLDS NOTHING, AND ITS OWN ROWS ARE DISCARDED BELOW.** A
            // fire binds one recurrent working set per request and is
            // charged its rows in the buffer, but the trunk that owns the
            // recurrence runs over NONE of a draft fire's rows — they are
            // the drafter's. Folding "everything" here would fold the
            // previous round's accepted prefix at the wrong instant and the
            // draft rows on top of it; folding nothing and forgetting the
            // rows again leaves the buffer exactly as this fire found it.
            RsGeometry { fold_len: Some(&fold_none), buffer: 0..buffer_pages },
        )?;
        {
            let out = out.clone();
            let conf = conf.clone();
            fwd.epilogue(move || {
                // **ONE OP, ONE CONSUMER, BOTH ANSWERS.** A second reduction
                // over the logits plane costs about a second a round even
                // with the plane bound once (3.0 s to 7.4 s over a 64-token
                // run; 46 s with a softmax on it) — two consumers take it off
                // the device. `top_k` reads it once and returns the values
                // beside the indices, so the proposal and the margin that
                // says how sure of it the drafter is come out together.
                let (value, index) = top_k(intrinsics::logits(), 2);
                out.put(&reshape(cast(index, dtype::i32), [shown * 2]));
                conf.put(&reshape(value, [shown * 2]));
            });
        }
        fwd.submit(&pipe).context("draft submit")?;
        // **ROW 0 IS THE ANCHOR, NOT A PREDICTION.** A block diffusion model
        // denoises each mask into the token AT ITS OWN POSITION, so row `i`
        // proposes position `held + i` and the anchor's row proposes nothing
        // new. The proposals are rows `1..block`.
        let top = out.take_host::<Vec<i32>>().await.context("draft readback")?;
        let value = conf.take_host::<Vec<f32>>().await.context("margin readback")?;
        // Row `r` occupies `[2r, 2r + 1]`: the proposal and its runner-up.
        proposals_owned = (1..shown as usize).map(|r| top[2 * r]).collect();
        let margin: Vec<f32> = (1..shown as usize)
            .map(|r| value[2 * r] - value[2 * r + 1])
            .collect();
        if pinned.is_none() && input.margin_width {
            verify = if wide_enough(&margin) { block } else { NARROW };
        }
        // The buffer is back to the accepted prefix the verify is about to
        // fold — see the draft fire's geometry.
        rs_set[0]
            .discard_buffered(block)
            .map_err(|why| format!("forget the draft fire's {block} row(s): {why}"))?;
        }
        let proposals = proposals_owned.as_slice();
        widths.push(verify);

        // ── the verify: ONE trunk fire over `[anchor, proposals]` ────────
        let mut fed = vec![anchor];
        fed.extend_from_slice(&proposals[..verify as usize - 1]);
        let toks = Channel::from(fed.as_slice()).named("toks_v");
        let indptr = Channel::from([0u32, verify]).named("embed_indptr_v");
        let positions = Channel::from_iter(held..held + verify).named("positions_v");
        let pages = Channel::from_iter(0..max_pages).named("pages_v");
        let page_indptr =
            Channel::from([0u32, (held + verify).div_ceil(page_size)]).named("page_indptr_v");
        let w_slot = Channel::from_iter((held..held + verify).map(|p| p / page_size)).named("w_slot_v");
        let w_off = Channel::from_iter((held..held + verify).map(|p| p % page_size)).named("w_off_v");
        let kv_len = Channel::from([held + verify]).named("kv_len_v");
        let readout = Channel::from_iter(0..verify).named("readout_v");
        let truth = Channel::new([verify], dtype::i32)
            .capacity(ring())
            .named("truth_v");

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &indptr)?;
        fwd.readout(&readout)?;
        fwd.attention(
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: ..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    // No mask: the verify is an ordinary causal prefill of
                    // the rows the drafter guessed.
                    mask: None,
                },
            }),
            &rs_set,
            RsGeometry {
                fold_len: Some(&fold_len),
                buffer: 0..buffer_pages,
            },
        )?;
        {
            let truth = truth.clone();
            fwd.epilogue(move || {
                truth.put(&reshape(reduce_argmax(intrinsics::logits()), [verify]));
            });
        }
        fwd.submit(&pipe).context("verify submit")?;
        let truth = truth.take_host::<Vec<i32>>().await.context("verify readback")?;

        // ── what the target kept ────────────────────────────────────────
        // Row `i` of the verify predicts position `held + i + 1`, which is
        // what proposal `i` claims — so the prefix is read off one zip.
        let kept = proposals[..verify as usize - 1]
            .iter()
            .zip(&truth)
            .take_while(|(p, t)| p == t)
            .count();
        if drafting {
            rounds += 1;
            drafted += verify as usize - 1;
            accepted += kept;
            gate.yielded(kept as u32 + 1, verify, verify > 1 && kept as u32 == verify - 1);
        }
        replayed += survivors as usize;
        // The rejected tail never happened: forget it before the next fire,
        // whose fold reaches exactly the accepted prefix.
        let rejected = (verify as usize - 1 - kept) as u32;
        if rejected > 0 {
            rs_set[0]
                .discard_buffered(rejected)
                .map_err(|why| format!("discard {rejected} rejected row(s): {why}"))?;
            discarded += rejected as usize;
        }
        survivors = kept as u32 + 1;
        for tok in proposals[..kept].iter() {
            generated.push(*tok as u32);
        }
        // The break's token is the TARGET's, so the round is never wrong.
        anchor = truth[kept];
        generated.push(anchor as u32);
        // Positions `held ..= held + kept` carried tokens the target agreed
        // with and stay; everything above them is stale and never read.
        held += kept as u32 + 1;
    }

    // A stop token ends the text; anything the last block committed past it
    // is not the answer.
    if let Some(at) = generated.iter().position(|t| stop_tokens.contains(t)) {
        generated.truncate(at + 1);
    }
    generated.truncate(input.max_tokens);
    let count = generated.len();
    Ok(Output {
        sampler: "dflash-speculative-bench",
        text: model::decode(&generated)?,
        tokens: generated.clone(),
        count,
        num_prompt_tokens: prompt.len(),
        num_output_tokens: count,
        token_ids: generated,
        rounds,
        drafted,
        accepted,
        acceptance_rate: if drafted == 0 {
            0.0
        } else {
            accepted as f64 / drafted as f64
        },
        tokens_per_round: if rounds == 0 {
            0.0
        } else {
            (accepted + rounds) as f64 / rounds as f64
        },
        block,
        verify: widths,
        replayed,
        discarded,
        rs_page,
        plain_fires,
    })
}
