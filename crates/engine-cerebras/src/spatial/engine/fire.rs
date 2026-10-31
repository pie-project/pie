//! A fire's step on the spatial engine: each lane's rows through the model,
//! the guest passes over its logits, the lanes' readouts.

use std::collections::BTreeMap;

use engine::error::{Error, Result as EngineResult};
use engine::fire::{LaneReadout, Readout};
use engine::program::InstanceId;
use eta_exec::Value;
use eta_ir::registry::{GeometryClass, Stage};

use super::Spatial;
use super::launch::{self, MODEL_NS, PagedLane};
use crate::api::{port_error, refusal};
use crate::spatial::wafer;

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

/// A paged lane's token loop past its row ([`Spatial::paged_ahead`]).
pub(super) struct PagedAhead {
    next: u32,
    pages: Vec<u32>,
    run: std::collections::VecDeque<(u32, u32)>,
}

impl PagedAhead {
    /// The argmax the loop already has for a row at `slot` on `pages` whose
    /// input is `token`, popped; `None` unless the row is exactly the one it
    /// ran (any other row leaves it to be dropped). Rows past the end of a
    /// stream are never asked for, so never answered.
    fn answer(&mut self, slot: u32, pages: &[u32], token: u32) -> Option<u32> {
        if self.next != slot
            || !pages.starts_with(&self.pages)
            || self.run.front().is_none_or(|&(input, _)| input != token)
        {
            return None;
        }
        self.next += 1;
        self.run.pop_front().map(|(_, answer)| answer)
    }
}

/// How far a greedy decode row's launch runs past it: the lookahead, or
/// the fires its frame still sends if more (those are certain to come), at
/// most CAP − 2 and never past the lane's own pages (`pages` of `size`
/// slots, the row at `slot`).
fn ahead_steps(
    fires_after: Option<u32>,
    lookahead: usize,
    cap: usize,
    pages: usize,
    size: u32,
    slot: u32,
) -> usize {
    if cap <= 2 {
        return 0;
    }
    let room = (pages as u32 * size).saturating_sub(slot + 1) as usize;
    lookahead
        .max(fires_after.map_or(0, |f| f as usize))
        .min(cap - 2)
        .min(room)
}

/// Per row of a lane's `rows` rows, the KV slots its mask admits, as runs: a
/// device geometry instance's dense mask, else `masking` (the fire's, or its
/// instance's port), else `None` (causal).
fn visible_runs(
    masking: Option<&engine::fire::Masking>,
    rows: usize,
    dense: Option<&(Vec<bool>, usize)>,
) -> Option<Vec<Vec<[u32; 2]>>> {
    let runs_of = |cells: &[bool]| -> Vec<[u32; 2]> {
        let mut runs = Vec::new();
        let mut at = 0;
        while at < cells.len() {
            if cells[at] {
                let start = at;
                while at < cells.len() && cells[at] {
                    at += 1;
                }
                runs.push([start as u32, at as u32]);
            } else {
                at += 1;
            }
        }
        runs
    };
    if let Some((cells, stride)) = dense {
        return Some(
            (0..rows)
                .map(|r| runs_of(&cells[r * stride..(r + 1) * stride]))
                .collect(),
        );
    }
    let masking = masking?;
    Some(
        (0..rows)
            .map(|r| {
                let mask = match masking {
                    engine::fire::Masking::Extent(mask) => mask,
                    engine::fire::Masking::Rows(masks) => match masks.get(r) {
                        Some(mask) => mask,
                        None => return Vec::new(),
                    },
                };
                mask.admitted(0..mask.len())
                    .map(|span| [span.start as u32, span.end as u32])
                    .collect()
            })
            .collect(),
    )
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

impl Spatial {
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
                return Ok((vec![one_hot(width, Some(token))], true));
            }
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
            tracing::info!(
                rows = tokens.len(),
                steps,
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

    /// The most steps a greedy lane's token loop runs past its row (0: none):
    /// the config's `lookahead`, else 32 once a model is loaded.
    pub(super) fn lookahead(&self) -> usize {
        let default = if self.model.is_some() { 32 } else { 0 };
        self.spatial.lookahead.unwrap_or(default)
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
                 spatial engine runs causal attention over a lane's own history only (no \
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
        // The KV slots a device geometry instance places a lane's rows at.
        let mut port_slots: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        // Masks an instance's port states for a lane the host places.
        let mut host_masks: BTreeMap<usize, engine::fire::Masking> = BTreeMap::new();
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
                if tracing::enabled!(target: "spatial_trace", tracing::Level::DEBUG) {
                    tracing::debug!(target: "spatial_trace",
                        "spatial-trace envelope: slots {:?} positions {:?} kv_len {:?} fold {:?} \
                         qo {:?} mask {}",
                        envelope.slots,
                        envelope.positions,
                        envelope.kv_len,
                        envelope.fold,
                        envelope.qo_indptr,
                        envelope.mask.as_ref().map_or(0, Vec::len)
                    );
                }
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
                    if let Some(slots) = ports.slots(rows).map_err(port_error)? {
                        port_slots.insert(source, slots.to_vec());
                    }
                    if let Some((cells, stride)) = ports.mask(rows).map_err(port_error)? {
                        if tracing::enabled!(target: "spatial_trace", tracing::Level::DEBUG) {
                            let seen: Vec<usize> = cells[..stride]
                                .iter()
                                .enumerate()
                                .filter(|(_, c)| **c)
                                .map(|(i, _)| i)
                                .collect();
                            tracing::debug!(target: "spatial_trace",
                                "spatial-trace mask: lane {source} rows {rows} stride {stride} row0 \
                                 sees {seen:?} slots {:?}",
                                ports.slots(rows).ok().flatten()
                            );
                        }
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
                    if let Some(crate::program::Tokens::Host(ids)) = self
                        .programs
                        .envelope_tokens(instance, lane.tokens.len(), at)
                        .map_err(refusal)?
                    {
                        tokens[source] = ids;
                    }
                    // The mask its `attn_mask` port states, when the fire
                    // carries none.
                    if lane.mask.is_none()
                        && let Some(mask) = self
                            .programs
                            .envelope_mask(instance, lane.tokens.len(), at)
                            .map_err(refusal)?
                    {
                        host_masks.insert(source, mask);
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
        // A paged model runs the fire's lanes together, each through its own
        // pages: (lane index, tokens, start, pages, logits asked, rows read).
        let paged = self.model.as_ref().is_some_and(|w| w.paged.is_some());
        let mut paged_lanes: Vec<(usize, PagedLane, Vec<u32>)> = Vec::new();
        // Lanes whose epilogue the wafer head's VM ran: (instance, the
        // step's `out` words, the VM's pool after it).
        let mut vm_out: std::collections::HashMap<usize, (InstanceId, Vec<u32>, Vec<u32>)> =
            std::collections::HashMap::new();
        // Cells the fire's paged lanes so far will have written.
        let mut planned: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
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
                // Greedy: the lane's one reader takes only the argmax of the
                // logits.
                let greedy = attached
                    .get(&at)
                    .is_some_and(|&instance| self.programs.greedy_reads(instance, &name))
                    && one_reader(step, at);
                let ahead = lane.fires_after;
                let first_slot = port_slots
                    .get(&at)
                    .and_then(|s| s.first().copied())
                    .unwrap_or(if positions[at].is_empty() {
                        start
                    } else {
                        lane.kv.held
                    });
                if first_slot == 0 {
                    self.kv.tainted.remove(&lane.slot);
                }
                // The model's pages must hold what the lane's rows will read;
                // a lane continuing on pages the model never got (a fire
                // that named another lane's pages) is re-prefilled into them
                // from its own tokens first.
                let mut on_pages = paged
                    && !lane.kv.pages.is_empty()
                    && !self.kv.tainted.contains(&lane.slot)
                    && self.kv.known(&lane.kv.pages, first_slot, &planned);
                if paged && !on_pages && !lane.kv.pages.is_empty() && first_slot > 0 {
                    let history = self
                        .kv
                        .slots
                        .get(&lane.slot)
                        .filter(|h| h.len() >= first_slot as usize)
                        .map(|h| h[..first_slot as usize].to_vec());
                    if let Some(history) = history {
                        let slots: Vec<u32> = (0..first_slot).collect();
                        let refill = PagedLane {
                            tokens: history.clone(),
                            slots: slots.clone(),
                            positions: slots.clone(),
                            pages: lane.kv.pages.clone(),
                            visible: None,
                            which: wafer::Logits::None,
                            argmax_only: false,
                            argmax_rows: false,
                            ahead: 0,
                            vm: None,
                        };
                        launch::fire_paged(self.model()?, &[&refill])?;
                        self.kv
                            .write_paged(lane.slot, &lane.kv.pages, &slots, &history);
                        self.kv.tainted.remove(&lane.slot);
                        self.refilled += u64::from(first_slot);
                        tracing::info!(
                            slot = lane.slot,
                            rows = first_slot,
                            total = self.refilled,
                            "spatial: lane re-prefilled into its own pages"
                        );
                        on_pages = self.kv.known(&lane.kv.pages, first_slot, &planned);
                    }
                }
                if paged && !on_pages && self.kv.tainted.insert(lane.slot) {
                    tracing::info!(
                        slot = lane.slot,
                        first_slot,
                        "spatial: the model's pages do not hold this lane's cache; it \
                             replays from its own tokens until it starts over"
                    );
                }
                if on_pages {
                    let last_only = rows.len() == 1 && rows[0] as usize + 1 == tokens[at].len();
                    let which = logits_wanted(&rows, tokens[at].len());
                    let n = tokens[at].len();
                    // Where the rows' K/V go (the instance's slots, else after
                    // what the lane holds), how they rotate (their positions,
                    // else their slots), and what each sees (its mask, else
                    // every slot up to its own).
                    let slots: Vec<u32> = port_slots.get(&at).cloned().unwrap_or_else(|| {
                        let held = if positions[at].is_empty() {
                            start
                        } else {
                            lane.kv.held
                        };
                        (held..held + n as u32).collect()
                    });
                    let rope = if positions[at].is_empty() {
                        slots.clone()
                    } else {
                        positions[at].clone()
                    };
                    let visible = visible_runs(
                        lane.mask.as_ref().or(host_masks.get(&at)),
                        n,
                        dense_masks.get(&at),
                    );
                    let size = self.kv.page_size.max(1);
                    for &q in &slots {
                        if let Some(&page) = lane.kv.pages.get((q / size) as usize) {
                            planned.insert((page, q % size));
                        }
                    }
                    // The multi-lane VM samples a decode row (one token, its
                    // one reader taking the last row) of a program it holds.
                    let vm = attached.get(&at).copied().filter(|&instance| {
                        n == 1
                            && last_only
                            && visible.is_none()
                            && one_reader(step, at)
                            && self.vm_lanes_take(instance)
                    });
                    // A greedy decode row: answered by a token loop that
                    // already ran it, else run ahead within its pages.
                    let loop_row =
                        greedy && last_only && vm.is_none() && n == 1 && visible.is_none();
                    let cap = match &self.model {
                        Some(w) => w.cap,
                        _ => 0,
                    };
                    if loop_row
                        && let Some(run) = self.paged_ahead.get_mut(&lane.slot)
                        && let Some(token) = run.answer(slots[0], &lane.kv.pages, tokens[at][0])
                    {
                        if run.run.is_empty() {
                            self.paged_ahead.remove(&lane.slot);
                        }
                        self.ahead_hits += 1;
                        self.kv
                            .write_paged(lane.slot, &lane.kv.pages, &slots, &tokens[at]);
                        logits.push(vec![one_hot(width as usize, Some(token))]);
                        fast.push(true);
                        read_rows.push(rows);
                        continue;
                    }
                    self.paged_ahead.remove(&lane.slot);
                    let ahead = if loop_row {
                        ahead_steps(
                            ahead,
                            self.lookahead(),
                            cap,
                            lane.kv.pages.len(),
                            size,
                            slots[0],
                        )
                    } else {
                        0
                    };
                    paged_lanes.push((
                        at,
                        PagedLane {
                            tokens: tokens[at].clone(),
                            slots,
                            positions: rope,
                            pages: lane.kv.pages.clone(),
                            visible,
                            which,
                            ahead,
                            argmax_only: greedy && last_only && vm.is_none(),
                            argmax_rows: greedy
                                && !last_only
                                && which == wafer::Logits::All
                                && vm.is_none(),
                            vm,
                        },
                        rows.clone(),
                    ));
                    (Vec::new(), false)
                } else {
                    // A model of one sequence a cache runs causal attention
                    // only (a paged model attends what each row's mask says).
                    self.check_masks(lane, at, start, tokens[at].len(), dense_masks.get(&at))?;
                    let vm = attached.get(&at).copied().filter(|&instance| {
                        rows.len() == 1
                            && rows[0] as usize + 1 == tokens[at].len()
                            && one_reader(step, at)
                            && self.vm_takes(instance)
                    });
                    match vm {
                        Some(instance) => {
                            let (row, pool) = self.forward_vm(
                                instance,
                                lane.slot,
                                &lane.kv.pages,
                                &tokens[at],
                                start,
                            )?;
                            vm_out.insert(at, (instance, row, pool));
                            (Vec::new(), true)
                        }
                        None => self.forward(
                            lane.slot,
                            &lane.kv.pages,
                            &tokens[at],
                            start,
                            &rows,
                            greedy,
                            ahead,
                        )?,
                    }
                }
            };
            logits.push(out);
            fast.push(ran_ahead);
            read_rows.push(rows);
        }
        // The multi-lane VM's lanes: one launch, each in its bank's lane.
        let (vm_lanes, mut paged_lanes): (Vec<_>, Vec<_>) = paged_lanes
            .into_iter()
            .partition(|(_, lane, _)| lane.vm.is_some());
        if !vm_lanes.is_empty() {
            let request: Vec<(InstanceId, &PagedLane)> = vm_lanes
                .iter()
                .filter_map(|(_, lane, _)| lane.vm.map(|instance| (instance, lane)))
                .collect();
            let answers = self.fire_vm_lanes(&request)?;
            for ((at, mut lane, rows), answer) in vm_lanes.into_iter().zip(answers) {
                let Some((instance, row, pool)) = answer else {
                    // No VM lane was free: its logits come back as any lane's.
                    lane.vm = None;
                    paged_lanes.push((at, lane, rows));
                    continue;
                };
                self.kv
                    .write_paged(lanes[at].slot, &lane.pages, &lane.slots, &lane.tokens);
                logits[at] = Vec::new();
                fast[at] = true;
                vm_out.insert(at, (instance, row, pool));
            }
        }
        if !paged_lanes.is_empty() {
            let request: Vec<&PagedLane> = paged_lanes.iter().map(|(_, lane, _)| lane).collect();
            let answers = launch::fire_paged(self.model()?, &request)?;
            self.paged_fires += 1;
            for ((at, lane, rows), (lane_logits, token)) in paged_lanes.into_iter().zip(answers) {
                self.kv
                    .write_paged(lanes[at].slot, &lane.pages, &lane.slots, &lane.tokens);
                let one_hot = |t: Option<u32>| one_hot(width as usize, t);
                if lane.argmax_only && lane.ahead > 0 && token.len() > 1 {
                    // Its row the first iteration's argmax; the rest ran ahead
                    // (each the next row's argmax, its input the one before).
                    let k = token.len() - 1;
                    let next = lane.slots[0] + 1;
                    let extra: Vec<u32> = (next..next + k as u32).collect();
                    self.kv
                        .write_paged(lanes[at].slot, &lane.pages, &extra, &token[..k]);
                    self.paged_ahead.insert(
                        lanes[at].slot,
                        PagedAhead {
                            next,
                            pages: lane.pages.clone(),
                            run: token[..k]
                                .iter()
                                .copied()
                                .zip(token[1..].iter().copied())
                                .collect(),
                        },
                    );
                    self.ahead_runs += 1;
                    logits[at] = vec![one_hot(token.first().copied())];
                    fast[at] = true;
                    continue;
                }
                if lane.argmax_only {
                    // Its row a one-hot of the argmax, as a token loop's.
                    logits[at] = vec![one_hot(token.last().copied())];
                    fast[at] = true;
                    continue;
                }
                if lane.argmax_rows {
                    // Each asked row a one-hot of its argmax.
                    logits[at] = rows
                        .iter()
                        .map(|&r| one_hot(token.get(r as usize).copied()))
                        .collect();
                    fast[at] = true;
                    continue;
                }
                logits[at] = match lane.which {
                    wafer::Logits::None => Vec::new(),
                    wafer::Logits::Last => lane_logits,
                    wafer::Logits::All => rows
                        .iter()
                        .map(|&r| lane_logits.get(r as usize).cloned().unwrap_or_default())
                        .collect(),
                };
            }
        }
        // Guest passes in the interpreter, each output it reads from this fire.
        let mut guest_read = vec![false; lanes.len()];
        for attachment in &step.attachments {
            if attachment.adapter_only {
                continue;
            }
            let at = attachment.lane as usize;
            if let Some((instance, row, pool)) = vm_out.remove(&at)
                && instance == attachment.instance
            {
                self.fire_vm(instance, &row, &pool)?;
                if let Some(read) = guest_read.get_mut(at) {
                    *read = true;
                }
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
                        // Its one row, handed over: nothing reads it again.
                        std::mem::take(&mut logits[lane][0])
                    } else {
                        (lane..end.max(lane + 1))
                            .filter_map(|l| logits.get(l))
                            .flat_map(|rows| rows.iter().flatten().copied())
                            .collect()
                    };
                    outputs.push((out_name.clone(), k, Value::F32(flat)));
                    if let Some(read) = guest_read.get_mut(lane) {
                        *read = true;
                    }
                }
            }
            if !self.fire_guest(attachment.instance, &outputs)? {
                self.programs
                    .fire_interpreted(attachment.instance, &outputs, Stage::Epilogue)
                    .map_err(refusal)?;
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
    fn a_token_loop_never_runs_past_the_lane_s_pages() {
        // Two pages of 16: a row at slot 28 has 3 slots left after it.
        assert_eq!(ahead_steps(None, 32, 16, 2, 16, 28), 3);
        // The last slot of the last page: nothing past it.
        assert_eq!(ahead_steps(None, 32, 16, 2, 16, 31), 0);
        // CAP bounds it (two iterations of the launch are the row and its
        // first follower).
        assert_eq!(ahead_steps(None, 32, 8, 4, 16, 0), 6);
        assert_eq!(ahead_steps(None, 32, 2, 4, 16, 0), 0);
        // The fires the frame still sends are a floor, not a ceiling: a
        // small frame (more frames may follow) keeps the lookahead, a large
        // one runs past it, up to CAP − 2.
        assert_eq!(ahead_steps(Some(0), 32, 64, 4, 16, 0), 32);
        assert_eq!(ahead_steps(Some(2), 32, 64, 4, 16, 0), 32);
        assert_eq!(ahead_steps(Some(50), 32, 64, 4, 16, 0), 50);
        assert_eq!(ahead_steps(Some(90), 32, 64, 8, 16, 0), 62);
    }

    #[test]
    fn a_token_loop_answers_only_the_rows_it_ran() {
        // Rows from slot 10 on: inputs 5, 6, 7 with argmax 6, 7, 8.
        let mut run = PagedAhead {
            next: 10,
            pages: vec![3],
            run: [(5, 6), (6, 7), (7, 8)].into_iter().collect(),
        };
        // A row the stream did not send (another input, slot or page) gets
        // nothing: a stop or the token limit leaves the rest unanswered.
        assert_eq!(run.answer(10, &[3], 9), None);
        assert_eq!(run.answer(11, &[3], 5), None);
        assert_eq!(run.answer(10, &[4], 5), None);
        // The rows it ran, in order; a grown page list still matches.
        assert_eq!(run.answer(10, &[3], 5), Some(6));
        assert_eq!(run.answer(11, &[3, 4], 6), Some(7));
        assert_eq!(run.answer(12, &[3, 4], 7), Some(8));
        assert_eq!(run.answer(13, &[3, 4], 8), None);
    }
}
