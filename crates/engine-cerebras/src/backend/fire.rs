//! A fire's step on the cerebras engine: each lane's rows through the model,
//! the guest passes over its logits, the lanes' readouts.

use std::collections::BTreeMap;

use engine::StepOutcome;
use engine::error::{Error, Result as EngineResult};
use engine::fire::{LaneReadout, Readout};
use engine::program::InstanceId;
use eta_exec::Value;
use eta_ir::registry::{GeometryClass, Stage};

use super::kv::Kv;
use super::launch::{self, MODEL_NS};
use super::{CerebrasEngine, ring};
use crate::guest::{port_error, refusal};
use crate::wafer;

/// Steps a sequence's first token loop runs past its row.
pub(super) const FIRST_LOOP: usize = 8;

/// The token loop's most steps past a greedy lane's row when the config states none and the
/// program names no loop of its own.
const DEFAULT_LOOKAHEAD: usize = 32;

/// The lookahead when the config states none: the program's posted loop (`LOOP`), else
/// [`DEFAULT_LOOKAHEAD`].
pub(super) fn default_lookahead(posted_loop: usize) -> usize {
    if posted_loop > 0 {
        posted_loop
    } else {
        DEFAULT_LOOKAHEAD
    }
}

/// The frames the runtime keeps posted when its config states none: with a posted loop, the most
/// the engine takes ([`Runahead::UNSTAGED_MAX_FRAMES`]), so every lane keeps posted past a loop's
/// end (0: no posted loop, the runtime's own).
///
/// [`Runahead::UNSTAGED_MAX_FRAMES`]: engine::runahead::Runahead::UNSTAGED_MAX_FRAMES
pub(super) fn dispatch_depth(posted_loop: usize, lookahead: usize) -> u32 {
    if posted_loop.min(lookahead) > 1 {
        u32::from(engine::runahead::Runahead::UNSTAGED_MAX_FRAMES)
    } else {
        0
    }
}

/// A logits row of `width` standing in for an argmax: 1 at `token`, 0
/// elsewhere (all 0 without one).
fn one_hot(width: usize, token: Option<u32>) -> Vec<f32> {
    let mut row = vec![0f32; width];
    if let Some(cell) = token.and_then(|t| row.get_mut(t as usize)) {
        *cell = 1.0;
    }
    row
}

/// The most steps a greedy lane's token loop runs past its row, decided at
/// load: the config's `lookahead` (`stated`), else [`default_lookahead`] of
/// the program's posted loop, where the program has a token loop (`WL`); 0
/// where it has none, and a lookahead stated for such a program is refused
/// rather than every greedy lane at its fire.
pub(super) fn lookahead_of(
    stated: Option<usize>,
    token_loop: bool,
    posted_loop: usize,
) -> Result<usize, String> {
    match (stated, token_loop) {
        (Some(steps), false) if steps > 0 => Err(format!(
            "lookahead = {steps}: the wafer program has no token loop (a WL build) to run it"
        )),
        (_, false) => Ok(0),
        (Some(steps), true) => Ok(steps),
        (None, true) => Ok(default_lookahead(posted_loop)),
    }
}

/// The logits a lane of `n` rows asks the model for when `rows` are read:
/// none, the last row's alone, or every row's.
fn logits_wanted(rows: &[u32], n: usize) -> wafer::Logits {
    if rows.is_empty() {
        wafer::Logits::None
    } else if rows.len() == 1 && rows[0] as usize + 1 == n {
        wafer::Logits::Last
    } else {
        wafer::Logits::All
    }
}

/// Whether lane `at` of `step` has one reader: one attachment that runs
/// stages covers it (others in the step read their own lanes).
fn one_reader(step: &engine::fire::Step, at: usize) -> bool {
    let at = at as u32;
    step.attachments
        .iter()
        .filter(|a| !a.adapter_only && (a.lane..a.lane + a.lanes.max(1)).contains(&at))
        .count()
        == 1
}

/// Whether `mask` (alternating runs: excluded, admitted, ...) lets a row at
/// `position` see every position up to its own: the causal mask the model
/// runs. Anything else this engine cannot run.
/// The first of a lane's `rows` (at positions `start..`) whose mask is not
/// causal over the lane's own history, if any. A mask speaks of the slots
/// from the lane's committed boundary `folded` on: column `c` is slot
/// `folded + c`, and every slot below it is read. So row `r` is causal when
/// it admits columns `0..=start + r - folded` and, where the mask is dense,
/// none after.
fn first_non_causal(
    masking: Option<&engine::fire::Masking>,
    dense: Option<(&[bool], usize)>,
    folded: u32,
    start: u32,
    rows: usize,
) -> Option<usize> {
    let folded = folded.min(start);
    let own = |row: usize| u64::from(start - folded) + row as u64;
    if let Some(masking) = masking {
        for row in 0..rows {
            let mask = match masking {
                engine::fire::Masking::Extent(mask) => mask,
                engine::fire::Masking::Rows(masks) => match masks.get(row) {
                    Some(mask) => mask,
                    None => continue,
                },
            };
            if !causal(mask, own(row)) {
                return Some(row);
            }
        }
    }
    if let Some((cells, stride)) = dense {
        for row in 0..rows {
            let line = &cells[row * stride..(row + 1) * stride];
            let seen = own(row) as usize + 1;
            if line.len() < seen
                || line.iter().take(seen).any(|&admit| !admit)
                || line.iter().skip(seen).any(|&admit| admit)
            {
                return Some(row);
            }
        }
    }
    None
}

fn causal(mask: &engine::fire::Mask, position: u64) -> bool {
    let want = position + 1;
    let admitted: u64 = mask.admitted(0..want).map(|r| r.end - r.start).sum();
    // A mask over fewer positions says nothing of those past its end.
    admitted == want.min(mask.len())
}

/// What a lane's rows are answered with: the last one's argmax (a greedy lane's), or their logits.
pub(super) enum Forward {
    Argmax(u32),
    Logits(Vec<Vec<f32>>),
}

impl CerebrasEngine {
    /// Runs `tokens` at positions `start..` for the lane (`slot`, its `pages`), the cache first
    /// brought to the lane's history; the logits of `rows`, each the vocabulary wide. `greedy`: the
    /// model runs its token loop `lookahead` steps past the row and returns its argmax alone.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &mut self,
        slot: u32,
        pages: &[u32],
        tokens: &[u32],
        start: u32,
        rows: &[u32],
        greedy: bool,
        ahead: Option<u32>,
    ) -> EngineResult<Forward> {
        let history = self.kv.history(slot, pages, start)?;
        self.model()?;
        let last_only = rows.len() == 1 && rows[0] as usize + 1 == tokens.len();
        let lookahead = self.lookahead();
        let greedy = greedy && last_only && lookahead > 0;
        let fed: Vec<u32> = history.iter().chain(tokens).copied().collect();
        // The wafer program runs its token loops back to back: the next
        // is posted as one lands, so the wafer runs while rows are answered.
        let posting = self.loop_posting();
        if posting {
            self.land_loop(false)?;
        }
        loop {
            if greedy {
                // A token loop that already ran these rows answers them.
                let last = fed.len() - 1;
                let c = &self.kv.cached;
                if c.tokens.len() > last
                    && c.tokens[..=last] == fed[..]
                    && let Some(token) = c.argmax.get(last).copied().flatten()
                {
                    self.ahead_hits += 1;
                    self.kv.write(slot, pages, history, tokens)?;
                    if posting {
                        self.post_loop()?;
                    }
                    return Ok(Forward::Argmax(token));
                }
            }
            // Nothing else runs while a posted loop is out: it lands first.
            if posting && self.loop_posted.is_some() {
                self.land_loop(true)?;
                continue;
            }
            break;
        }
        let agree = self.kv.agreement(&history);
        // This greedy row picks up where the cache's last token loop ended
        // (fed its last argmax): the loop was used to its end.
        let continues = {
            let c = &self.kv.cached;
            agree == c.tokens.len()
                && agree == history.len()
                && agree > 0
                && c.argmax.last().copied().flatten() == tokens.first().copied()
        };
        self.align_cache(agree, &history)?;
        if greedy {
            let steps = {
                let c = &mut self.kv.cached;
                c.next_loop = if continues && c.next_loop > 0 {
                    (c.next_loop * 2).min(lookahead)
                } else {
                    FIRST_LOOP.min(lookahead)
                };
                // At least as far as the lane's frame will fire: those
                // fires are certain, so a large frame starts at its size.
                c.next_loop = c
                    .next_loop
                    .max(ahead.map_or(0, |fires| (fires as usize).min(lookahead)));
                c.next_loop
            };
            // A loop runs no further than the lane's pages: the rows past a sequence's end
            // are waste.
            let room = (pages.len() as u32 * self.kv.page_size)
                .saturating_sub(start + tokens.len() as u32) as usize;
            let bounded = !posting && !pages.is_empty();
            let steps = if bounded { steps.min(room) } else { steps };
            tracing::info!(
                rows = tokens.len(),
                steps,
                room,
                fires_after = ?ahead,
                "spatial: token loop"
            );
            let (ran, argmax) = launch::greedy(self.model()?, tokens, start, steps)?;
            self.ahead_runs += 1;
            if tracing::enabled!(target: "spatial_trace", tracing::Level::DEBUG) {
                tracing::debug!(target: "spatial_trace",
                    "spatial-trace token loop: {} run(s), {} row(s) answered from them",
                    self.ahead_runs, self.ahead_hits
                );
            }
            let cached = &mut self.kv.cached;
            cached.tokens.extend_from_slice(&ran);
            cached.argmax.extend(argmax);
            cached.argmax.resize(cached.tokens.len(), None);
            let token = cached.argmax[fed.len() - 1].unwrap_or_default();
            self.kv.write(slot, pages, history, tokens)?;
            if posting {
                self.post_loop()?;
            }
            return Ok(Forward::Argmax(token));
        }
        let which = logits_wanted(rows, tokens.len());
        let logits = launch::forward(self.model()?, tokens, start, which)?;
        self.kv.cached.extend(tokens);
        self.kv.write(slot, pages, history, tokens)?;
        Ok(Forward::Logits(match which {
            wafer::Logits::None => Vec::new(),
            wafer::Logits::Last => logits,
            wafer::Logits::All => rows
                .iter()
                .map(|&r| logits.get(r as usize).cloned().unwrap_or_default())
                .collect(),
        }))
    }

    /// Whether the model is a unit program (its lanes' rows run together, each over its lane's
    /// own pages and state slot).
    fn paged(&self) -> bool {
        self.model.as_ref().is_some_and(|w| w.lanes.is_some())
    }

    /// Whether greedy loops are posted back to back: wherever a model with a
    /// token loop is loaded.
    fn loop_posting(&self) -> bool {
        self.model.is_some() && self.lookahead() > 0
    }

    /// The posted loop's rows into the cache (waiting for it, or only if it
    /// has landed).
    fn land_loop(&mut self, wait: bool) -> EngineResult<()> {
        let Some((token, start)) = self.loop_posted else {
            return Ok(());
        };
        let Some(answer) = self.model()?.take_posted(wait) else {
            return Ok(());
        };
        self.loop_posted = None;
        let argmax = answer.map_err(Error::Device)?;
        let c = &mut self.kv.cached;
        // Every change to the cache comes after the posted loop has landed
        // (`forward` lands it first), and a loop answers its 1 + steps rows:
        // a loop that lands anywhere else, or empty, means the host's account
        // of the wafer's sequence and the wafer's own have parted.
        if c.tokens.len() != start || argmax.is_empty() {
            return Err(Error::Device(format!(
                "a posted token loop from position {start} landed on a cache of {} token(s) \
                 with {} argmax: the host's account of the wafer's sequence has parted from \
                 the wafer's",
                c.tokens.len(),
                argmax.len()
            )));
        }
        c.tokens.push(token);
        c.tokens.extend_from_slice(&argmax[..argmax.len() - 1]);
        c.argmax.extend(argmax.iter().map(|&t| Some(t)));
        self.ahead_runs += 1;
        Ok(())
    }

    /// The next loop posted from the cache's end (fed its last argmax),
    /// twice the last one's steps up to the lookahead.
    fn post_loop(&mut self) -> EngineResult<()> {
        if self.loop_posted.is_some() {
            return Ok(());
        }
        let lookahead = self.lookahead();
        let c = &mut self.kv.cached;
        let Some(Some(token)) = c.argmax.last().copied() else {
            return Ok(());
        };
        if c.argmax.len() != c.tokens.len() {
            return Ok(());
        }
        let start = c.tokens.len();
        // A loop's rows come back when it ends: the request's last rows wait
        // for its whole loop, so posted loops stay short (the program plan's
        // loop).
        let most = self.model()?.posted_loop;
        let c = &mut self.kv.cached;
        c.next_loop = (c.next_loop.max(FIRST_LOOP) * 2)
            .min(lookahead)
            .min(most.max(1));
        let steps = c.next_loop;
        let w = self.model()?;
        if start + 1 >= w.positions {
            return Ok(());
        }
        w.post_greedy(&[token], start, steps)
            .map_err(Error::Device)?;
        self.loop_posted = Some((token, start));
        tracing::info!(start, steps, "spatial: artifact token loop posted");
        Ok(())
    }

    /// The loaded model's readout: the float output its readout rows read,
    /// and its width (a load refuses a trace without one).
    fn readout(&self) -> EngineResult<(String, u32)> {
        self.readout
            .clone()
            .ok_or_else(|| Error::Load("the spatial engine has no model loaded".into()))
    }

    /// The most steps a greedy lane's token loop runs past its row (0: none):
    /// as the load decided it ([`lookahead_of`]); 0 with no model loaded.
    pub(super) fn lookahead(&self) -> usize {
        if self.model.is_some() {
            self.lookahead
        } else {
            0
        }
    }

    /// Refuses a lane whose attention is not causal over its own history:
    /// the model runs that alone, and a mask it ignored would answer wrong.
    pub(super) fn check_masks(
        &self,
        lane: &engine::fire::Lane,
        at: usize,
        start: u32,
        rows: usize,
        dense: Option<&(Vec<bool>, usize)>,
    ) -> EngineResult<()> {
        let refuse = |row: usize| {
            Err(Error::Invalid(format!(
                "lane {at} row {row} (position {}) attends a non-causal set of positions; the \
                 cerebras engine runs causal attention over a lane's own history only (no \
                 sinks, windows or custom masks)",
                start as usize + row
            )))
        };
        match first_non_causal(
            lane.mask.as_ref(),
            dense.map(|(cells, stride)| (cells.as_slice(), *stride)),
            lane.kv.folded,
            start,
            rows,
        ) {
            Some(row) => refuse(row),
            None => Ok(()),
        }
    }

    /// One step: prologues, the lanes' tokens and positions (a device
    /// geometry instance states them on its ports), the forward, the guest
    /// passes over its logits, and the lanes' readouts. A pass that refuses
    /// or faults (a prologue, an envelope, an epilogue, a lane its rows
    /// cannot run) faults its own lanes in the outcome; the other lanes run
    /// on and commit. An `Err` is the step lost whole.
    #[allow(clippy::too_many_lines)]
    pub(super) fn fire_step(
        &mut self,
        step: &engine::fire::Step,
    ) -> EngineResult<(Vec<LaneReadout>, StepOutcome)> {
        if self.stream.is_some() {
            return self
                .fire_stream(step)
                .map(|read| (read, StepOutcome::committed()));
        }
        let fire_t0 = std::time::Instant::now();
        self.next_fire = self.next_fire.wrapping_add(1);
        let mut outcome = StepOutcome::committed();
        for attachment in passes(step) {
            if self.programs.prologue_runs(attachment.instance)
                && let Err(refused) =
                    self.programs
                        .fire_interpreted(attachment.instance, &[], Stage::Prologue)
            {
                fault_pass(&mut outcome, attachment, "prologue", &refused.to_string());
            }
        }
        let lanes = &step.lanes;
        let pages: Vec<Vec<u32>> = lanes.iter().map(Kv::lane_pages).collect();
        if tracing::enabled!(target: "spatial_trace", tracing::Level::DEBUG) {
            for lane in lanes {
                tracing::debug!(target: "spatial_trace",
                    "spatial-trace lane: slot {} tokens {:?} positions {:?} kv {:?} kv_less {} group {:?} \
                     mask {} readout {:?} rs {:?}",
                    lane.slot,
                    lane.tokens,
                    lane.positions,
                    lane.kv,
                    lane.kv_less,
                    lane.group,
                    lane.mask.is_some(),
                    lane.readout,
                    lane.rs
                );
            }
        }
        // Each lane's tokens and positions: the fire's, or an instance's ports.
        let mut resolved = Resolved {
            tokens: lanes.iter().map(|l| l.tokens.clone()).collect(),
            dense_masks: BTreeMap::new(),
            positions: lanes.iter().map(|l| l.positions.clone()).collect(),
        };
        for attachment in passes(step) {
            if faulted(&outcome, attachment) {
                continue;
            }
            if let Err(refused) = self.envelope(step, attachment, &mut resolved) {
                fault_pass(&mut outcome, attachment, "envelope", &refused.to_string());
            }
        }
        let Resolved {
            tokens,
            dense_masks,
            positions,
        } = resolved;
        // Every lane's window-buffer copies land before any lane reads or writes.
        self.windows.copy_slots(lanes)?;
        // The forward, lane by lane, keeping the logits of the rows something reads.
        let attached: BTreeMap<usize, InstanceId> = step
            .attachments
            .iter()
            .filter(|a| !a.adapter_only)
            .flat_map(|a| {
                (0..a.lanes.max(1)).map(move |k| (a.lane as usize + k as usize, a.instance))
            })
            .collect();
        let (name, width) = self.readout()?;
        let mut logits: Vec<Vec<Vec<f32>>> = Vec::with_capacity(lanes.len());
        let mut read_rows: Vec<Vec<u32>> = Vec::with_capacity(lanes.len());
        // Greedy lanes the model answered with their argmax alone (no logits row made).
        let mut known: Vec<Option<u32>> = vec![None; lanes.len()];
        // a unit program's lanes, run together after: (lane, its first position, greedy)
        let mut paged: Vec<(usize, u32, bool)> = Vec::new();
        // a ring's, run together after, and each one's lane and history
        let mut wants: Vec<ring::Want> = Vec::new();
        let mut seated: Vec<(usize, Vec<u32>)> = Vec::new();
        for (at, lane) in lanes.iter().enumerate() {
            let n = tokens[at].len() as u32;
            let rows: Vec<u32> = match &lane.readout {
                Readout::Rows(rows) => rows.clone(),
                Readout::Last if n > 0 => vec![n - 1],
                Readout::None if attached.contains_key(&at) && n > 0 => vec![n - 1],
                _ => Vec::new(),
            };
            let out = if tokens[at].is_empty() || lane_faulted(&outcome, at) {
                Vec::new()
            } else {
                // The rows' positions: stated (consecutive), else after what the lane holds.
                let start = match positions[at].first() {
                    Some(&p) => p,
                    None if !pages[at].is_empty() => lane.kv.held,
                    None => self.kv.slots.get(&lane.slot).map_or(0, |h| h.len() as u32),
                };
                let runs = if positions[at]
                    .iter()
                    .enumerate()
                    .any(|(i, &p)| p != start + i as u32)
                {
                    Err(Error::Invalid(format!(
                        "the spatial engine runs a lane's rows at consecutive positions; lane \
                         {at} has {:?}",
                        positions[at]
                    )))
                } else {
                    // The model runs causal attention over the lane's own history only.
                    self.check_masks(lane, at, start, tokens[at].len(), dense_masks.get(&at))
                        .and_then(|()| self.windows.fold(lane, start, &tokens[at]))
                };
                if let Err(refused) = runs {
                    fault_lane(&mut outcome, step, at, &refused.to_string());
                    Vec::new()
                } else {
                    // Greedy: the lane's one reader takes only the argmax of
                    // the logits.
                    let greedy = attached
                        .get(&at)
                        .is_some_and(|&instance| self.programs.greedy_reads(instance, &name))
                        && one_reader(step, at);
                    if self.ring.is_some() {
                        // the lanes' rows run together below, each lane on its seat
                        let history = self.kv.history(lane.slot, &pages[at], start)?;
                        let last = greedy && self.lookahead() > 0 && rows == [n - 1];
                        wants.push(ring::Want {
                            slot: lane.slot,
                            instance: attached.get(&at).copied(),
                            fed: history.iter().chain(&tokens[at]).copied().collect(),
                            logits: if last {
                                Vec::new()
                            } else {
                                rows.iter().map(|&r| (start + r) as usize).collect()
                            },
                        });
                        seated.push((at, history));
                        Vec::new()
                    } else if self.paged() {
                        // the lanes' rows run together below (a launch B rows, each over its
                        // lane's pages)
                        paged.push((at, start, greedy));
                        Vec::new()
                    } else {
                        match self.forward(
                            lane.slot,
                            &pages[at],
                            &tokens[at],
                            start,
                            &rows,
                            greedy,
                            lane.fires_after,
                        )? {
                            Forward::Argmax(token) => {
                                known[at] = Some(token);
                                vec![Vec::new()]
                            }
                            Forward::Logits(rows) => rows,
                        }
                    }
                }
            };
            logits.push(out);
            read_rows.push(rows);
        }
        if !wants.is_empty() {
            let ahead = self.lookahead() > 0;
            let (Some(ring), Some(model)) = (self.ring.as_mut(), self.model.as_mut()) else {
                return Err(Error::Load("the wafer program has no ring".into()));
            };
            let got = ring.rows(model, &wants, ahead).map_err(launch::device)?;
            for ((at, history), got) in seated.into_iter().zip(got) {
                known[at] = got.argmax.filter(|_| !read_rows[at].is_empty());
                logits[at] = match known[at] {
                    Some(_) => vec![Vec::new()],
                    None => got.logits,
                };
                self.kv
                    .write(lanes[at].slot, &pages[at], history, &tokens[at])?;
            }
        }
        if !paged.is_empty() {
            // A lane's kv refs are the pool's: the wafer's pages they name. Its
            // seat is its recurrences' state slot, where the program has any.
            let wafer = self.wafer_pages.ok_or_else(|| {
                Error::Load("a unit program loaded without its wafer pages".into())
            })?;
            let stateful = self
                .facts
                .as_ref()
                .is_some_and(crate::facts::Facts::has_state);
            let mut all: Vec<crate::units::Row> = Vec::new();
            for &(at, start, greedy) in &paged {
                let lane_pages = pages[at]
                    .iter()
                    .map(|&slot| wafer.page(slot))
                    .collect::<Result<Vec<u32>, String>>()
                    .map_err(Error::Invalid)?;
                for (i, &token) in tokens[at].iter().enumerate() {
                    all.push(crate::units::Row {
                        token,
                        pos: start + i as u32,
                        pages: lane_pages.clone(),
                        state: if stateful { lanes[at].slot } else { 0 },
                        logits: !greedy && read_rows[at].contains(&(i as u32)),
                    });
                }
            }
            let (argmax, mut rows_logits) = self
                .model()?
                .rows(all)
                .map_err(|e| Error::Device(format!("the wafer model: {e}")))?;
            rows_logits.reverse();
            let mut k = 0;
            for &(at, _, greedy) in &paged {
                let n = tokens[at].len();
                let mine = &argmax[k..k + n];
                k += n;
                if greedy {
                    // its one reader takes the argmax: no logits row is made
                    let last = read_rows[at].first().map_or(n - 1, |&r| r as usize);
                    known[at] = mine.get(last).copied();
                    logits[at] = vec![Vec::new()];
                } else {
                    logits[at] = read_rows[at]
                        .iter()
                        .map(|_| rows_logits.pop().unwrap_or_default())
                        .collect();
                }
            }
        }
        // Guest passes in the interpreter, each output it reads from this
        // fire; a pass one of whose lanes faulted runs none.
        for attachment in passes(step) {
            if let Some(why) = outcome
                .faulted
                .iter()
                .find(|fault| pass_lanes(attachment).contains(&fault.lane))
                .map(|fault| fault.reason.clone())
            {
                fault_pass(&mut outcome, attachment, "epilogue", &why);
                continue;
            }
            match self.epilogue_outputs(attachment, &name, width, &mut logits, &known) {
                Ok((outputs, argmaxed)) => {
                    let fired = if argmaxed.is_empty() {
                        self.programs.fire_interpreted(
                            attachment.instance,
                            &outputs,
                            Stage::Epilogue,
                        )
                    } else {
                        self.programs.fire_argmaxed(
                            attachment.instance,
                            &outputs,
                            &argmaxed,
                            Stage::Epilogue,
                        )
                    };
                    if let Err(refused) = fired {
                        fault_pass(&mut outcome, attachment, "epilogue", &refused.to_string());
                    }
                }
                Err(refused) => {
                    fault_pass(&mut outcome, attachment, "epilogue", &refused.to_string());
                }
            }
        }
        self.fires += 1;
        tracing::info!(
            fires = self.fires,
            lanes = step.lanes.len(),
            rows = step.lanes.iter().map(|l| l.tokens.len()).sum::<usize>(),
            fire_ms = fire_t0.elapsed().as_secs_f64() * 1e3,
            model_ms = MODEL_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6,
            "spatial: model time"
        );
        let readouts = lanes
            .iter()
            .enumerate()
            .map(|(at, lane)| {
                // A token loop's row is a stand-in for its argmax: the lane's
                // answer is in its program's channels, as a device guest's.
                if matches!(lane.readout, Readout::None)
                    || logits[at].is_empty()
                    || known[at].is_some()
                    || lane_faulted(&outcome, at)
                {
                    return LaneReadout::default();
                }
                let values: Vec<f32> = logits[at].iter().flatten().copied().collect();
                LaneReadout {
                    output: name.clone(),
                    rows: u32::try_from(read_rows[at].len()).unwrap_or(u32::MAX),
                    width,
                    values,
                    ..LaneReadout::default()
                }
            })
            .collect();
        Ok((readouts, outcome))
    }

    /// The tokens, positions and masks `attachment`'s instance states for
    /// its lanes in `resolved`: a device geometry instance's ports, else
    /// its envelope's tokens.
    fn envelope(
        &mut self,
        step: &engine::fire::Step,
        attachment: &engine::fire::Attachment,
        resolved: &mut Resolved,
    ) -> EngineResult<()> {
        let lanes = &step.lanes;
        let instance = attachment.instance;
        let first = attachment.lane as usize;
        if self.programs.geometry_of(instance) != Some(GeometryClass::DeviceGeometry) {
            let count = self.programs.lane_count(instance).map_err(refusal)?;
            for at in 0..count {
                let source = first + at;
                let Some(lane) = lanes.get(source) else { break };
                if let Some(crate::guest::Tokens::Host(ids)) = self
                    .programs
                    .envelope_tokens(instance, lane.tokens.len(), at)
                    .map_err(refusal)?
                {
                    resolved.tokens[source] = ids;
                }
            }
            return Ok(());
        }
        let Some(mut envelope) = self.programs.device_envelope(instance).map_err(refusal)? else {
            return Ok(());
        };
        envelope.host_rows = lanes
            .iter()
            .skip(first)
            .take(attachment.lanes.max(1) as usize)
            .map(|lane| lane.tokens.len() as u32)
            .collect();
        for at in 0..envelope.lanes() {
            let source = first + at;
            let Some(lane) = lanes.get(source) else { break };
            let ports = envelope.lane(at, source).map_err(port_error)?;
            let rows = if ports.owns_slots() {
                ports.rows() as usize
            } else {
                lane.tokens.len()
            };
            let have = if ports.owns_slots() {
                ports
                    .reach(rows)
                    .map_err(port_error)?
                    .map(|after| after.saturating_sub(rows as u32))
            } else {
                (!Kv::lane_pages(lane).is_empty()).then_some(lane.kv.held)
            };
            resolved.tokens[source] = ports.tokens_for(rows).map_err(port_error)?.to_vec();
            if let Some((cells, stride)) = ports.mask(rows).map_err(port_error)? {
                resolved
                    .dense_masks
                    .insert(source, (cells.to_vec(), stride));
            }
            if let Some(have) = have {
                if let Some(p) = ports.positions_for(have, rows).map_err(port_error)? {
                    resolved.positions[source] = p.to_vec();
                } else if resolved.positions[source].len() != rows {
                    resolved.positions[source] = (have..have + rows as u32).collect();
                }
            }
        }
        Ok(())
    }

    /// The outputs `attachment`'s epilogue reads from this fire: the model's
    /// `name` (its logits) over the pass's lanes. A program reading another
    /// model output this engine does not serve is refused.
    fn epilogue_outputs(
        &self,
        attachment: &engine::fire::Attachment,
        name: &str,
        width: u32,
        logits: &mut [Vec<Vec<f32>>],
        known: &[Option<u32>],
    ) -> EngineResult<Argmaxed> {
        let mut outputs: Vec<(String, u32, Value)> = Vec::new();
        // outputs answered by their argmax alone (a greedy lane the model told only that)
        let mut argmaxed: Vec<(String, u32, Vec<i32>)> = Vec::new();
        for (out_name, k) in self.programs.outputs_of(attachment.instance) {
            let lane = attachment.lane as usize + k as usize;
            if out_name != name && out_name.starts_with("attn.") {
                return Err(Error::Invalid(format!(
                    "the program reads model output `{out_name}`; the spatial engine's \
                     model answers `{name}` (the logits) alone — attention scores are not \
                     served yet"
                )));
            }
            if out_name != name {
                continue;
            }
            // A pass over several lanes reads its rows as one output: the
            // lane's, then the pass's later lanes'.
            let end = attachment.lane as usize + attachment.lanes.max(1) as usize;
            let lanes_of = lane..end.max(lane + 1);
            if lanes_of
                .clone()
                .all(|l| known.get(l).copied().flatten().is_some())
            {
                let ids = lanes_of.map(|l| known[l].unwrap_or(0) as i32).collect();
                argmaxed.push((out_name.clone(), k, ids));
                continue;
            }
            for l in lanes_of {
                // (a lane known by its argmax read beside others: its one-hot row)
                if let Some(token) = known.get(l).copied().flatten() {
                    logits[l] = vec![one_hot(width as usize, Some(token))];
                }
            }
            if logits.get(lane).is_some() {
                let flat: Vec<f32> = if known.get(lane).copied().flatten().is_some()
                    && end <= lane + 1
                    && logits[lane].len() == 1
                {
                    // Its one row, handed over: nothing else reads it.
                    std::mem::take(&mut logits[lane][0])
                } else {
                    (lane..end.max(lane + 1))
                        .filter_map(|l| logits.get(l))
                        .flat_map(|rows| rows.iter().flatten().copied())
                        .collect()
                };
                outputs.push((out_name.clone(), k, Value::F32(flat)));
            }
        }
        Ok((outputs, argmaxed))
    }
}

/// An epilogue's outputs: its logits rows, and those answered by their argmax alone.
type Argmaxed = (Vec<(String, u32, Value)>, Vec<(String, u32, Vec<i32>)>);

/// The tokens, positions and masks of a step's lanes, as their instances
/// resolve them.
struct Resolved {
    tokens: Vec<Vec<u32>>,
    positions: Vec<Vec<u32>>,
    /// Masks a device geometry instance states on its port, by lane.
    dense_masks: BTreeMap<usize, (Vec<bool>, usize)>,
}

/// The step's attachments that run stages.
fn passes(step: &engine::fire::Step) -> impl Iterator<Item = &engine::fire::Attachment> {
    step.attachments.iter().filter(|a| !a.adapter_only)
}

/// The lanes `attachment`'s pass is of.
fn pass_lanes(attachment: &engine::fire::Attachment) -> std::ops::Range<u32> {
    attachment.lane..attachment.lane + attachment.lanes.max(1)
}

/// Whether one of `attachment`'s lanes faulted.
fn faulted(outcome: &StepOutcome, attachment: &engine::fire::Attachment) -> bool {
    outcome
        .faulted
        .iter()
        .any(|fault| pass_lanes(attachment).contains(&fault.lane))
}

/// Whether lane `at` faulted.
fn lane_faulted(outcome: &StepOutcome, at: usize) -> bool {
    outcome
        .faulted
        .iter()
        .any(|fault| fault.lane as usize == at)
}

/// `attachment`'s pass refused or faulted at `stage` for `why`: its lanes
/// fault, by its instance.
fn fault_pass(
    outcome: &mut StepOutcome,
    attachment: &engine::fire::Attachment,
    stage: &str,
    why: &str,
) {
    tracing::warn!(
        instance = attachment.instance.0,
        stage,
        %why,
        "spatial: a guest pass faulted; its lanes fault"
    );
    outcome.fault_lanes(pass_lanes(attachment), Some(attachment.instance), why);
}

/// Lane `at`'s rows cannot run, for `why`: the lanes of every pass it is
/// of fault, or it alone where it is of none.
fn fault_lane(outcome: &mut StepOutcome, step: &engine::fire::Step, at: usize, why: &str) {
    let lane = u32::try_from(at).unwrap_or(u32::MAX);
    let mut of = passes(step)
        .filter(|attachment| pass_lanes(attachment).contains(&lane))
        .peekable();
    if of.peek().is_none() {
        outcome.fault_lanes(lane..lane + 1, None, why);
    }
    for attachment in of {
        fault_pass(outcome, attachment, "forward", why);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lane_asks_for_the_logits_its_readers_read() {
        assert_eq!(logits_wanted(&[], 4), wafer::Logits::None);
        assert_eq!(logits_wanted(&[3], 4), wafer::Logits::Last);
        assert_eq!(logits_wanted(&[1], 4), wafer::Logits::All);
        assert_eq!(logits_wanted(&[2, 3], 4), wafer::Logits::All);
    }

    /// A mask speaks of the slots from the lane's committed boundary on: a
    /// causal mask past ten committed slots, two buffered and two rows, is
    /// four columns wide and is taken; one that hides a buffered slot, or
    /// admits a later row, is not. The same mask read from slot 0 (a
    /// boundary of 0) is short of the lane's history and says nothing of it.
    #[test]
    fn a_causal_mask_is_counted_from_the_lanes_boundary() {
        use engine::fire::{Mask, Masking};
        // Lane writes rows at slots 12 and 13; slots 0..10 are committed.
        let causal = Masking::Rows(vec![Mask::new(vec![0, 3], 4), Mask::new(vec![0, 4], 4)]);
        assert_eq!(first_non_causal(Some(&causal), None, 10, 12, 2), None);
        let hides = Masking::Rows(vec![Mask::new(vec![1, 2], 4), Mask::new(vec![0, 4], 4)]);
        assert_eq!(first_non_causal(Some(&hides), None, 10, 12, 2), Some(0));
        // A dense mask in the same columns, and one admitting the next row.
        let dense = [true, true, true, false, true, true, true, true];
        assert_eq!(first_non_causal(None, Some((&dense, 4)), 10, 12, 2), None);
        let ahead = [true, true, true, true, true, true, true, true];
        assert_eq!(
            first_non_causal(None, Some((&ahead, 4)), 10, 12, 2),
            Some(0)
        );
        // With no boundary the columns are slots, as before.
        let whole = Masking::Extent(Mask::new(vec![0, 14], 14));
        assert_eq!(first_non_causal(Some(&whole), None, 0, 12, 2), None);
    }

    /// An epilogue whose put is `word`'s cell, or, `reads_logits`, the
    /// model's logits over its lane: a pass with no rows holds none, so its
    /// epilogue faults.
    fn guest(spatial: &mut CerebrasEngine, reads_logits: bool) -> engine::program::ProgramId {
        use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
        use eta_ir::op::Op;
        use eta_ir::registry::{Axis, EngineCaps, OutputDecl};
        use eta_ir::types::{Dtype, Literal, Shape};
        let ops = if reads_logits {
            vec![
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: Shape::matrix(1, 4),
                    dtype: Dtype::F32,
                },
                Op::ChanPut { chan: 0, value: 0 },
            ]
        } else {
            vec![
                Op::Const(Literal::F32(7.0)),
                Op::Broadcast {
                    value: 0,
                    shape: Shape::matrix(1, 4),
                },
                Op::ChanPut { chan: 0, value: 1 },
            ]
        };
        let program = TraceContainer {
            names: vec!["logits".to_string()],
            channels: vec![ChannelDecl {
                shape: Shape::matrix(1, 4),
                dtype: ChanDType::Concrete(Dtype::F32),
                capacity: 2,
                host_role: HostRole::Reader,
                seeded: false,
            }],
            ports: Vec::new(),
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops,
            }],
            externs: Vec::new(),
            inputs: Vec::new(),
        };
        let decls = vec![OutputDecl {
            name: "logits".to_string(),
            axis: Axis::Lane,
            width: 4,
            dtype: Dtype::F32,
            layers: 0,
            readout: true,
        }];
        let bound = eta_ir::validate::bind(program, vec![decls], EngineCaps::default())
            .expect("the guest binds");
        let stages = eta_compiler::plan::compile_bound(&bound);
        spatial
            .programs
            .register(&engine::program::ProgramRegistration {
                program_hash: u64::from(reads_logits),
                launch: eta_compiler::codegen::launch::build(&bound, &stages),
                ..Default::default()
            })
            .expect("the guest registers")
    }

    /// Three instances' passes in one step, the middle one's epilogue
    /// faulting: its lane faults, by its instance, and the step does not
    /// err; the other two's epilogues run and their puts commit.
    #[test]
    fn one_instances_fault_leaves_the_others_outputs_committed() {
        use engine::channel::{ChannelId, ChannelRegistration};
        use engine::fire::{Attachment, Lane, Step};
        use engine::program::{BindExtents, InstanceBinding};
        use eta_ir::container::{ChanDType, HostRole};
        use eta_ir::types::Dtype;

        fn no_model(
            _: &poem_ir::Trace,
            _: &std::path::Path,
        ) -> std::result::Result<checkpoint::contract::ModelContract, String> {
            Err("no model in this test".into())
        }
        let mut spatial = CerebrasEngine::new(crate::Config::default(), no_model);
        spatial.readout = Some(("logits".to_string(), 4));
        let puts = guest(&mut spatial, false);
        let reads = guest(&mut spatial, true);
        let instances: Vec<InstanceId> = [puts, reads, puts]
            .into_iter()
            .enumerate()
            .map(|(at, program)| {
                let channel = ChannelId(at as u64 + 1);
                spatial
                    .programs
                    .register_channel(&ChannelRegistration {
                        id: channel,
                        shape: vec![1, 4],
                        dtype: ChanDType::Concrete(Dtype::F32),
                        host_role: HostRole::Reader,
                        seeded: false,
                        extern_dir: None,
                        capacity: 2,
                        extern_name: Vec::new(),
                        slots: Vec::new(),
                    })
                    .expect("the channel registers");
                spatial
                    .programs
                    .bind(&InstanceBinding {
                        program,
                        channels: vec![channel],
                        seeds: Vec::new(),
                        geometry: GeometryClass::Host,
                        extents: BindExtents::default(),
                        device_resolved: Default::default(),
                        instance_slots: Vec::new(),
                        ring_slots: Vec::new(),
                        adapter_slots: Vec::new(),
                        scratch: engine::ScratchPlace::None,
                        pair: engine::ScratchPlace::None,
                        pair_generation: 0,
                    })
                    .expect("the instance binds")
                    .id
            })
            .collect();
        // Lanes with no rows: no forward runs, and the wafer is not asked.
        let step = Step {
            lanes: vec![Lane::default(); 3],
            attachments: instances
                .iter()
                .enumerate()
                .map(|(at, &instance)| Attachment {
                    lane: at as u32,
                    lanes: 1,
                    instance,
                    adapter_only: false,
                    scratch: ::engine::ScratchPlace::None,
                })
                .collect(),
            media: Vec::new(),
            voxels: Vec::new(),
            shape: None,
        };
        let (_, outcome) = spatial
            .fire_step(&step)
            .expect("a guest's fault does not fail the step");
        let lanes: Vec<(u32, Option<InstanceId>)> = outcome
            .faulted
            .iter()
            .map(|fault| (fault.lane, fault.instance))
            .collect();
        assert_eq!(
            lanes,
            [(1, Some(instances[1]))],
            "the faulting pass's lane alone"
        );
        assert!(
            outcome.faulted[0].reason.contains("logits"),
            "faulted by its own reason: {}",
            outcome.faulted[0].reason
        );
        let seven: Vec<u8> = [7.0f32; 4].iter().flat_map(|v| v.to_le_bytes()).collect();
        for at in [0, 2] {
            let cell = spatial
                .programs
                .take(instances[at], 0)
                .expect("the channel reads");
            assert_eq!(
                cell.as_deref(),
                Some(&seven[..]),
                "instance {at}'s put committed"
            );
        }
        assert!(
            spatial.programs.take(instances[1], 0).is_err(),
            "the faulted instance is poisoned"
        );
    }

    #[test]
    fn an_answered_row_is_a_one_hot_of_its_argmax() {
        assert_eq!(one_hot(4, Some(2)), [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(one_hot(3, Some(7)), [0.0; 3]);
        assert_eq!(one_hot(2, None), [0.0; 2]);
    }
}
