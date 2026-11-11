//! A fire's step on the cerebras engine: each lane's rows through the model,
//! the guest passes over its logits, the lanes' readouts.

use std::collections::BTreeMap;

use engine::error::{Error, Result as EngineResult};
use engine::fire::{LaneReadout, Readout};
use engine::program::InstanceId;
use eta_exec::Value;
use eta_ir::registry::{GeometryClass, Stage};

use super::CerebrasEngine;
use super::launch::{self, MODEL_NS};
use crate::guest::{port_error, refusal};
use crate::wafer;

/// Steps a sequence's first token loop runs past its row.
const FIRST_LOOP: usize = 8;

/// A logits row of `width` standing in for an argmax: 1 at `token`, 0
/// elsewhere (all 0 without one).
fn one_hot(width: usize, token: Option<u32>) -> Vec<f32> {
    let mut row = vec![0f32; width];
    if let Some(cell) = token.and_then(|t| row.get_mut(t as usize)) {
        *cell = 1.0;
    }
    row
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
fn causal(mask: &engine::fire::Mask, position: u64) -> bool {
    let want = position + 1;
    let admitted: u64 = mask.admitted(0..want).map(|r| r.end - r.start).sum();
    // A mask over fewer positions says nothing of those past its end.
    admitted == want.min(mask.len())
}

impl CerebrasEngine {
    /// Runs `tokens` at positions `start..` for the lane (`slot`, its
    /// `pages`), the model's cache first brought to the lane's history; the
    /// logits of `rows` (indexes into `tokens`), each the vocabulary wide.
    ///
    /// `greedy`: what reads the lane's last row takes only its argmax. Then
    /// the model runs its token loop past the row (`lookahead` steps more,
    /// one launch on the wafer, no logits back), later rows the loop already
    /// ran answer from it without running, and the row comes back as a
    /// one-hot of its argmax.
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
    ) -> EngineResult<(Vec<Vec<f32>>, bool)> {
        let history = self.kv.history(slot, pages, start)?;
        self.model()?;
        let last_only = rows.len() == 1 && rows[0] as usize + 1 == tokens.len();
        let lookahead = self.lookahead();
        let greedy = greedy && last_only && lookahead > 0;
        let width = self.readout.as_ref().map_or(0, |(_, w)| *w as usize);
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
                    return Ok((vec![one_hot(width, Some(token))], true));
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
            // A loop runs no further than the lane's pages (debug `loop_room = false`: the
            // lookahead alone): the rows past a sequence's end are waste.
            let room = (pages.len() as u32 * self.kv.page_size)
                .saturating_sub(start + tokens.len() as u32) as usize;
            let bounded = !posting && self.debug.loop_room.unwrap_or(true) && !pages.is_empty();
            let steps = if bounded { steps.min(room) } else { steps };
            // With the prompt forced through the loop (debug `force_prompt`), the
            // waited-for launch is the prompt alone: its first token right after the
            // prefill, every later loop posted.
            let steps = if posting && self.debug.force_prompt {
                0
            } else {
                steps
            };
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
            return Ok((vec![one_hot(width, Some(token))], true));
        }
        let which = logits_wanted(rows, tokens.len());
        let logits = launch::forward(self.model()?, tokens, start, which)?;
        self.kv.cached.extend(tokens);
        self.kv.write(slot, pages, history, tokens)?;
        Ok((
            match which {
                wafer::Logits::None => Vec::new(),
                wafer::Logits::Last => logits,
                wafer::Logits::All => rows
                    .iter()
                    .map(|&r| logits.get(r as usize).cloned().unwrap_or_default())
                    .collect(),
            },
            false,
        ))
    }

    /// Whether greedy loops are posted back to back (debug `posted = false`: one
    /// loop at a time).
    fn loop_posting(&self) -> bool {
        self.model.is_some() && self.lookahead() > 0 && self.debug.posted.unwrap_or(true)
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
        if c.tokens.len() != start || argmax.is_empty() {
            // The cache moved on (it never does while a loop is out).
            tracing::warn!(
                start,
                held = c.tokens.len(),
                "spatial: a posted token loop dropped"
            );
            return Ok(());
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
        // for its whole loop, so posted loops stay short (the config's
        // `posted_loop`, else the program plan's loop).
        let plan_loop = self.model()?.posted_loop;
        let c = &mut self.kv.cached;
        let most = self.cfg.posted_loop.unwrap_or(plan_loop);
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

    /// The most steps a greedy lane's token loop runs past its row (0: none):
    /// the config's `lookahead`, else 32 once a model is loaded.
    pub(super) fn lookahead(&self) -> usize {
        let default = if self.model.is_some() { 32 } else { 0 };
        self.cfg.lookahead.unwrap_or(default)
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
        if let Some(masking) = &lane.mask {
            for row in 0..rows {
                let mask = match masking {
                    engine::fire::Masking::Extent(mask) => mask,
                    engine::fire::Masking::Rows(masks) => match masks.get(row) {
                        Some(mask) => mask,
                        None => continue,
                    },
                };
                if !causal(mask, u64::from(start) + row as u64) {
                    return refuse(row);
                }
            }
        }
        if let Some((cells, stride)) = dense {
            for row in 0..rows {
                let position = start as usize + row;
                let line = &cells[row * stride..(row + 1) * stride];
                let seen = position + 1;
                if line.len() < seen
                    || line.iter().take(seen).any(|&admit| !admit)
                    || line.iter().skip(seen).any(|&admit| admit)
                {
                    return refuse(row);
                }
            }
        }
        Ok(())
    }

    /// One step: prologues, the lanes' tokens and positions (a device
    /// geometry instance states them on its ports), the forward, the guest
    /// passes over its logits, and the lanes' readouts.
    #[allow(clippy::too_many_lines)]
    pub(super) fn fire_step(
        &mut self,
        step: &engine::fire::Step,
    ) -> EngineResult<Vec<LaneReadout>> {
        let fire_t0 = std::time::Instant::now();
        self.next_fire = self.next_fire.wrapping_add(1);
        for attachment in &step.attachments {
            if !attachment.adapter_only && self.programs.prologue_runs(attachment.instance) {
                self.programs
                    .fire_interpreted(attachment.instance, &[], Stage::Prologue)
                    .map_err(refusal)?;
            }
        }
        let lanes = &step.lanes;
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
        let mut tokens: Vec<Vec<u32>> = lanes.iter().map(|l| l.tokens.clone()).collect();
        // Masks a device geometry instance states on its port, by lane.
        let mut dense_masks: BTreeMap<usize, (Vec<bool>, usize)> = BTreeMap::new();
        let mut positions: Vec<Vec<u32>> = lanes.iter().map(|l| l.positions.clone()).collect();
        for attachment in &step.attachments {
            if attachment.adapter_only {
                continue;
            }
            let instance = attachment.instance;
            let first = attachment.lane as usize;
            if self.programs.geometry_of(instance) == Some(GeometryClass::DeviceGeometry) {
                let Some(mut envelope) =
                    self.programs.device_envelope(instance).map_err(refusal)?
                else {
                    continue;
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
                        (!lane.kv.pages.is_empty()).then_some(lane.kv.held)
                    };
                    tokens[source] = ports.tokens_for(rows).map_err(port_error)?.to_vec();
                    if let Some((cells, stride)) = ports.mask(rows).map_err(port_error)? {
                        dense_masks.insert(source, (cells.to_vec(), stride));
                    }
                    if let Some(have) = have {
                        if let Some(p) = ports.positions_for(have, rows).map_err(port_error)? {
                            positions[source] = p.to_vec();
                        } else if positions[source].len() != rows {
                            positions[source] = (have..have + rows as u32).collect();
                        }
                    }
                }
            } else {
                let count = self.programs.lane_count(instance).map_err(refusal)?;
                for at in 0..count {
                    let source = first + at;
                    let Some(lane) = lanes.get(source) else { break };
                    if let Some(crate::guest::Tokens::Host(ids)) = self
                        .programs
                        .envelope_tokens(instance, lane.tokens.len(), at)
                        .map_err(refusal)?
                    {
                        tokens[source] = ids;
                    }
                }
            }
        }
        // The forward, lane by lane, keeping the logits of the rows something reads.
        let attached: BTreeMap<usize, InstanceId> = step
            .attachments
            .iter()
            .filter(|a| !a.adapter_only)
            .flat_map(|a| {
                (0..a.lanes.max(1)).map(move |k| (a.lane as usize + k as usize, a.instance))
            })
            .collect();
        let (name, width) = self.readout.clone().ok_or_else(|| {
            Error::Load("the model reads no float output at its readout rows".into())
        })?;
        let mut logits: Vec<Vec<Vec<f32>>> = Vec::with_capacity(lanes.len());
        let mut read_rows: Vec<Vec<u32>> = Vec::with_capacity(lanes.len());
        // Lanes a token loop answered (their row a one-hot of the argmax).
        let mut fast: Vec<bool> = Vec::with_capacity(lanes.len());
        for (at, lane) in lanes.iter().enumerate() {
            let n = tokens[at].len() as u32;
            let rows: Vec<u32> = match &lane.readout {
                Readout::Rows(rows) => rows.clone(),
                Readout::Last if n > 0 => vec![n - 1],
                Readout::None if attached.contains_key(&at) && n > 0 => vec![n - 1],
                _ => Vec::new(),
            };
            let (out, ran_ahead) = if tokens[at].is_empty() {
                (Vec::new(), false)
            } else {
                // The rows' positions: stated (consecutive), else after what
                // the lane holds.
                let start = match positions[at].first() {
                    Some(&p) => p,
                    None if !lane.kv.pages.is_empty() => lane.kv.held,
                    None => self.kv.slots.get(&lane.slot).map_or(0, |h| h.len() as u32),
                };
                if positions[at]
                    .iter()
                    .enumerate()
                    .any(|(i, &p)| p != start + i as u32)
                {
                    return Err(Error::Invalid(format!(
                        "the spatial engine runs a lane's rows at consecutive positions; lane \
                         {at} has {:?}",
                        positions[at]
                    )));
                }
                // The model runs causal attention over the lane's own history only.
                self.check_masks(lane, at, start, tokens[at].len(), dense_masks.get(&at))?;
                // Greedy: the lane's one reader takes only the argmax of the
                // logits.
                let greedy = attached
                    .get(&at)
                    .is_some_and(|&instance| self.programs.greedy_reads(instance, &name))
                    && one_reader(step, at);
                self.forward(
                    lane.slot,
                    &lane.kv.pages,
                    &tokens[at],
                    start,
                    &rows,
                    greedy,
                    lane.fires_after,
                )?
            };
            logits.push(out);
            fast.push(ran_ahead);
            read_rows.push(rows);
        }
        // Guest passes in the interpreter, each output it reads from this fire.
        for attachment in &step.attachments {
            if attachment.adapter_only {
                continue;
            }
            let mut outputs: Vec<(String, u32, Value)> = Vec::new();
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
                if logits.get(lane).is_some() {
                    let flat: Vec<f32> = if fast.get(lane) == Some(&true)
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
            self.programs
                .fire_interpreted(attachment.instance, &outputs, Stage::Epilogue)
                .map_err(refusal)?;
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
        Ok(lanes
            .iter()
            .enumerate()
            .map(|(at, lane)| {
                // A token loop's row is a stand-in for its argmax: the lane's
                // answer is in its program's channels, as a device guest's.
                if matches!(lane.readout, Readout::None) || logits[at].is_empty() || fast[at] {
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
            .collect())
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

    #[test]
    fn an_answered_row_is_a_one_hot_of_its_argmax() {
        assert_eq!(one_hot(4, Some(2)), [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(one_hot(3, Some(7)), [0.0; 3]);
        assert_eq!(one_hot(2, None), [0.0; 2]);
    }
}
