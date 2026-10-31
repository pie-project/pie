//! Guest epilogues on the wafer's eta VM: which programs it can take, the
//! single-sequence VM, and the multi-lane VM's banks.

use engine::error::{Error, Result as EngineResult};
use engine::program::{InstanceId, ProgramId};
use eta_exec::Value;
use eta_ir::registry::Stage;

use super::Spatial;
use super::launch::{self, PagedLane};
use crate::api::refusal;
use crate::spatial::{guest4, serve2d, wafer};

/// The guest VM's limits a table must fit (guest4's simulator build: wide
/// registers, pool words).
const GUEST_REGS: u32 = 12;

const GUEST_POOL: u32 = 4096;

impl Spatial {
    /// Whether the wafer's guest VM could run `instance`'s epilogue
    /// (`guest::translate` at the VM's limits), told once a program: the
    /// engine still runs every epilogue on the host; this says which would
    /// move to the wafer.
    pub(super) fn probe_guest(&mut self, instance: InstanceId, program: ProgramId) {
        if !self.guest_probed.insert(program.0) {
            return;
        }
        let Some(plan) = self.programs.plan_of(instance) else {
            return;
        };
        let vocab = self.readout.as_ref().map_or(0, |(_, w)| *w);
        let host_takes: Vec<u32> = plan
            .package
            .channels
            .iter()
            .filter(|c| c.host_role == eta_ir::container::HostRole::Reader)
            .map(|c| c.id)
            .collect();
        let name = self.readout.as_ref().map_or("", |(n, _)| n.as_str());
        // Translated to the wafer VM's registers and pool when the model has
        // one (the translator re-makes draws at their readers to fit fewer
        // registers), else to the host's limits. A table whose pool overruns
        // is translated again with the big logits-free arrays sunk into
        // main, whose words are then live only while main runs.
        let (regs, pool) = match &self.model {
            Some(w) => w.vm.map_or((GUEST_REGS, GUEST_POOL), |(lay, _)| {
                (lay.nreg as u32, lay.pool_words as u32)
            }),
            _ => (GUEST_REGS, GUEST_POOL),
        };
        // Translated without a pool bound, so a table's size is known when
        // it overruns; the bound is applied below.
        let build = |sink: bool| {
            guest4::translate(&plan, vocab, &host_takes, regs, u32::MAX, sink)
                .and_then(|table| guest4::lane::Program::of(&plan, table, name))
        };
        let mut built = build(false);
        let over = built
            .as_ref()
            .is_ok_and(|guest| guest.table.pool_words > pool);
        if over {
            let sunk = build(true);
            let better = match (&sunk, &built) {
                (Ok(s), Ok(b)) => s.table.pool_words < b.table.pool_words,
                _ => false,
            };
            tracing::info!(
                program = program.0,
                pool = built.as_ref().map_or(0, |g| g.table.pool_words),
                sunk_pool = sunk.as_ref().map_or(0, |g| g.table.pool_words),
                taken = better,
                "spatial guest: the pool overruns; translated again with logits-free arrays sunk into main"
            );
            if better {
                built = sunk;
            }
        }
        let built = built.and_then(|guest| {
            if guest.table.pool_words > pool {
                Err(format!(
                    "{} pool words, more than {pool}",
                    guest.table.pool_words
                ))
            } else {
                Ok(guest)
            }
        });
        match built {
            Ok(guest) => {
                tracing::info!(
                    program = program.0,
                    instructions = guest.table.ins.len(),
                    pre = guest.table.pre_len,
                    registers = guest.table.nreg,
                    pool = guest.table.pool_words,
                    homes = guest.table.homes.len(),
                    "spatial guest: the epilogue would run on the wafer"
                );
                if let Some(w) = &self.model
                    && let Some((lay, ring)) = w.vm
                {
                    // A multi-lane VM drives batched lanes; one lane, a single sequence.
                    let refused = if ring.lanes > 1 {
                        lay.refuses(&guest.table)
                            .or_else(|| ring.refuses_outputs(&guest.table.outputs))
                    } else {
                        w.vm_refuses(&guest.table)
                    };
                    match refused {
                        Some(why) => {
                            tracing::info!(program = program.0, %why, "spatial guest: the VM cannot take it")
                        }
                        None => {
                            let placement = self.placement();
                            let f = guest4::cost::forecast(&guest.table, lay.vr as u32);
                            match placement.refuses(f) {
                                Some(why) => {
                                    tracing::info!(program = program.0, %why, pre = f.pre, main = f.main, token = placement.token_cycles, "spatial guest: the VM could take it; the forecast keeps it on the host");
                                    self.guest_costly.insert(program.0, why);
                                }
                                None => tracing::info!(
                                    program = program.0,
                                    pre = f.pre,
                                    main = f.main,
                                    token = placement.token_cycles,
                                    "spatial guest: the VM can take it"
                                ),
                            }
                        }
                    }
                }
                self.guest_programs
                    .insert(program.0, std::sync::Arc::new(guest));
            }
            Err(why) => tracing::info!(
                program = program.0,
                %why,
                "spatial guest: the epilogue stays on the host"
            ),
        }
    }

    /// When a table the VM can take runs there: the config's
    /// `token_cycles`, `pre_margin` and `main_fraction`, else the token from
    /// the model's size and the placement's default thresholds.
    pub(super) fn placement(&self) -> guest4::cost::Placement {
        use guest4::cost::{Placement, token_cycles};
        Placement {
            token_cycles: self
                .spatial
                .token_cycles
                .unwrap_or_else(|| token_cycles(self.params)),
            pre_margin: self.spatial.pre_margin.unwrap_or(Placement::PRE_MARGIN),
            main_fraction: self
                .spatial
                .main_fraction
                .unwrap_or(Placement::MAIN_FRACTION),
        }
    }

    /// Who runs translatable epilogues, the config's `guest`: `"off"` (the
    /// host interpreter, the default), `"check"` (the contract played on the
    /// host: [`guest4::lane::CheckVm`]) or `"wafer"` (the VM on the serving
    /// head, once the program has one; until then the host).
    pub(super) fn guest_mode(&self) -> String {
        let mode = self
            .spatial
            .guest
            .clone()
            .unwrap_or_else(|| "off".to_string());
        if matches!(mode.as_str(), "off" | "check" | "wafer") {
            mode
        } else {
            tracing::warn!(
                mode,
                "spatial guest: not off, check or wafer; epilogues stay on the host"
            );
            "off".to_string()
        }
    }

    /// Whether the wafer head's VM runs `instance`'s epilogue (`guest`
    /// "wafer"): a one-sequence program with a VM, the instance's program
    /// translated and within the VM's limits, and the VM free or already
    /// its.
    pub(super) fn vm_takes(&self, instance: InstanceId) -> bool {
        if self.guest_mode() != "wafer" {
            return false;
        }
        let Some(w) = &self.model else {
            return false;
        };
        if w.vm.is_none() || w.paged.is_some() {
            return false;
        }
        let Some((program, _, _)) = self.programs.instance_view(instance) else {
            return false;
        };
        let Some(guest) = self.guest_programs.get(&program.0) else {
            return false;
        };
        if let Some(why) = w
            .vm_refuses(&guest.table)
            .or_else(|| self.guest_costly.get(&program.0).cloned())
        {
            tracing::debug!(program = program.0, %why, "spatial guest: the VM cannot take it");
            return false;
        }
        // One lane's homes at a time; another instance's stay on the host
        // until the VM's owner is gone.
        self.vm_owner
            .is_none_or(|owner| owner == instance || self.programs.instance_view(owner).is_none())
    }

    /// `tokens` at `start` for the lane (`slot`, its `pages`) with the
    /// head's VM running `instance`'s epilogue on the last row: the model's
    /// cache brought to the lane's history, every row but the last run
    /// (the head's argmax mode), then the last as a VM step. The step's
    /// `out` words and the VM's pool after it.
    pub(super) fn forward_vm(
        &mut self,
        instance: InstanceId,
        slot: u32,
        pages: &[u32],
        tokens: &[u32],
        start: u32,
    ) -> EngineResult<(Vec<u32>, Vec<u32>)> {
        let device = |e: String| Error::Device(format!("the wafer model's VM: {e}"));
        if self.vm_owner != Some(instance) {
            let (program, _, inst) = self
                .programs
                .instance_view(instance)
                .ok_or_else(|| Error::Invalid(format!("instance {instance:?} is gone")))?;
            let guest = self
                .guest_programs
                .get(&program.0)
                .cloned()
                .ok_or_else(|| Error::Invalid("the program has no guest table".into()))?;
            // Its carried state: every home's value as it stands.
            let seeds: Vec<guest4::wafer::Seed> = guest
                .table
                .homes
                .iter()
                .map(|h| {
                    guest4::wafer::Seed::of(
                        h,
                        guest4::lane::words_of(&inst.channels[h.chan as usize].current()),
                    )
                })
                .collect();
            let Some(w) = self.model.as_mut() else {
                return Err(Error::Device("the VM path without a wafer model".into()));
            };
            w.vm_load(guest, seeds).map_err(device)?;
            self.vm_owner = Some(instance);
            tracing::info!(
                ?instance,
                program = program.0,
                "spatial guest: the program runs on the wafer's VM"
            );
        }
        let history = self.kv.history(slot, pages, start)?;
        let agree = self.kv.agreement(&history);
        self.align_cache(agree, &history)?;
        let n = tokens.len();
        if n > 1 {
            launch::forward(self.model()?, &tokens[..n - 1], start, wafer::Logits::None)?;
        }
        let Some(w) = self.model.as_mut() else {
            return Err(Error::Device("the VM path without a wafer model".into()));
        };
        let (_, row, pool) = w
            .vm_step(tokens[n - 1], (start as usize) + n - 1)
            .map_err(device)?;
        self.kv.cached.extend(tokens);
        self.kv.write(slot, pages, history, tokens)?;
        Ok((row, pool))
    }

    /// Whether the wafer head's multi-lane VM (serve2d_pipe `VM` 1, `gring`
    /// lanes L > 1) samples `instance`'s rows (`guest` "wafer"): its
    /// translated program fits the VM and is the one it holds (or the VM
    /// holds none a live request still uses), and its lane has a bank (it
    /// keeps its own, or one is free).
    pub(super) fn vm_lanes_take(&self, instance: InstanceId) -> bool {
        if self.guest_mode() != "wafer" {
            return false;
        }
        let Some(w) = &self.model else {
            return false;
        };
        let Some((lay, ring)) = w.vm else {
            return false;
        };
        if ring.lanes < 2 || w.paged.is_none() {
            return false;
        }
        let Some((program, _, _)) = self.programs.instance_view(instance) else {
            return false;
        };
        let Some(guest) = self.guest_programs.get(&program.0) else {
            return false;
        };
        // A program a lane: a set whose image takes it (A or B); placement
        // finds it a bank, evicting an idle lane if it must, or leaves it on
        // the host.
        if lay.lprog {
            return self
                .vm_sets()
                .iter()
                .any(|(lay, ring, _)| self.set_takes(lay, ring, instance));
        }
        if self.guest_costly.contains_key(&program.0)
            || lay
                .refuses(&guest.table)
                .or_else(|| ring.refuses_outputs(&guest.table.outputs))
                .is_some()
        {
            return false;
        }
        let live = |owner: &Option<InstanceId>| {
            owner.is_some_and(|o| self.programs.instance_view(o).is_some())
        };
        // One program for every lane, unless the image runs a program a lane.
        if !lay.lprog
            && self.vm_program.is_some_and(|p| p != program.0)
            && self.vm_banks.iter().any(live)
        {
            return false;
        }
        self.vm_banks.contains(&Some(instance))
            || self.vm_banks.len() < ring.lanes as usize
            || self.vm_banks.iter().any(|o| !live(o))
    }

    /// One multi-lane VM launch for `lanes` (each its instance and its
    /// decode row): the program loaded if the VM holds another, each
    /// instance's bank taken and seeded from its channels the first time,
    /// the launch's VM lanes filled in bank order (an idle one repeats a
    /// live lane, its bank kept as it was when a live request owns it).
    /// Per lane: its instance, the turn's `out` words, its pool after it;
    /// `None` for a lane no VM lane was free for (the host samples it).
    #[allow(clippy::type_complexity)]
    pub(super) fn fire_vm_lanes(
        &mut self,
        lanes: &[(InstanceId, &PagedLane)],
    ) -> EngineResult<Vec<Option<(InstanceId, Vec<u32>, Vec<u32>)>>> {
        let device = |e: String| Error::Device(format!("the wafer model's VM: {e}"));
        let Some(w) = &self.model else {
            return Err(Error::Device("the VM path without a wafer model".into()));
        };
        let (lay, ring) =
            w.vm.ok_or_else(|| Error::Device("the wafer program has no VM".into()))?;
        // Set A's lanes, then set B's: a global bank is a head lane.
        let sets = self.vm_sets();
        let n = if lay.lprog {
            sets.last().map_or(0, |(_, _, r)| r.end)
        } else {
            ring.lanes as usize
        };
        let lay_of = |bank: usize| {
            sets.iter()
                .find(|(_, _, r)| r.contains(&bank))
                .map_or(lay, |(l, _, _)| *l)
        };
        // Banks of requests that are gone are free.
        let alive: Vec<bool> = self
            .vm_banks
            .iter()
            .map(|o| o.is_some_and(|o| self.programs.instance_view(o).is_some()))
            .collect();
        for (owner, live) in self.vm_banks.iter_mut().zip(alive) {
            if !live {
                *owner = None;
            }
        }
        let programs = &self.programs;
        self.vm_parked
            .retain(|i, _| programs.instance_view(*i).is_some());
        let placed = if lay.lprog {
            self.place_vm_lanes(lanes, n)?
        } else {
            self.seed_vm_lanes(lanes, n)?
        };
        let Some(lead) = lanes
            .iter()
            .zip(&placed)
            .find(|(_, b)| b.is_some())
            .map(|(l, _)| l.1)
        else {
            return Ok(vec![None; lanes.len()]);
        };
        // The launch in bank order.
        let batch_of = |lane: &PagedLane| serve2d::BatchLane {
            pos: lane.positions[0],
            slot: lane.slots[0],
            pages: lane.pages.clone(),
            visible: None,
            forced: lane.tokens.clone(),
        };
        // Every head lane of the launch: the VM sets' are the first `n`;
        // any rest (a set this path does not drive) repeat a live lane,
        // skipped.
        let width = match &self.model {
            Some(w) => w.launch_lanes.max(n),
            _ => n,
        };
        let mut launch: Vec<serve2d::BatchLane> = vec![batch_of(lead); width];
        for (&(_, lane), bank) in lanes.iter().zip(&placed) {
            if let Some(bank) = *bank {
                launch[bank] = batch_of(lane);
            }
        }
        // Only this launch's lanes run a turn: an idle bank, or one whose
        // request sat this launch out, is skipped and its state untouched.
        let sampled: Vec<bool> = (0..n).map(|b| placed.contains(&Some(b))).collect();
        let mut inputs: Vec<wafer::StepInput> = vec![None; n];
        for (&(instance, _), bank) in lanes.iter().zip(&placed) {
            if let Some(bank) = *bank {
                inputs[bank] = self.step_input(instance, &lay_of(bank))?;
            }
        }
        let Some(w) = self.model.as_mut() else {
            return Err(Error::Device("the VM path without a wafer model".into()));
        };
        let steps = w.vm_fire(launch, sampled.clone(), inputs).map_err(device)?;
        self.vm_bank_used.resize(n, 0);
        for bank in placed.iter().flatten() {
            self.vm_bank_used[*bank] = self.paged_fires + 1;
        }
        for (bank, (token, row, _)) in steps.iter().enumerate().filter(|(b, _)| sampled[*b]) {
            tracing::debug!(
                bank,
                owner = ?self.vm_banks.get(bank).copied().flatten(),
                token,
                row = ?&row[..row.len().min(4)],
                "spatial guest: a VM lane's turn"
            );
        }
        self.paged_fires += 1;
        Ok(lanes
            .iter()
            .zip(&placed)
            .map(|(&(instance, _), bank)| {
                bank.map(|bank| {
                    let (_, row, pool) = steps[bank].clone();
                    (instance, row, pool)
                })
            })
            .collect())
    }

    /// A single-program multi-lane VM's banks for `lanes`: the program
    /// (the first lane's) loaded if the VM holds another, each lane its own
    /// bank, else a free one seeded from its channels; `None` for a lane no
    /// bank was free for.
    pub(super) fn seed_vm_lanes(
        &mut self,
        lanes: &[(InstanceId, &PagedLane)],
        n: usize,
    ) -> EngineResult<Vec<Option<usize>>> {
        let device = |e: String| Error::Device(format!("the wafer model's VM: {e}"));
        let first = lanes
            .first()
            .map(|(i, _)| *i)
            .ok_or_else(|| Error::Invalid("no VM lanes".into()))?;
        let (program, guest) = self.guest_of(first)?;
        if self.vm_program != Some(program) {
            let Some(w) = self.model.as_mut() else {
                return Err(Error::Device("the VM path without a wafer model".into()));
            };
            w.vm_load_lanes(std::sync::Arc::clone(&guest))
                .map_err(device)?;
            self.vm_program = Some(program);
            self.vm_banks = vec![None; n];
            tracing::info!(
                program,
                lanes = n,
                "spatial guest: the program runs on the wafer's multi-lane VM"
            );
        }
        self.vm_banks.resize(n, None);
        let mut placed: Vec<Option<usize>> = Vec::with_capacity(lanes.len());
        for &(instance, _) in lanes {
            let bank = if let Some(bank) = self.vm_banks.iter().position(|o| *o == Some(instance)) {
                bank
            } else {
                let Some(bank) = self.vm_banks.iter().position(Option::is_none) else {
                    placed.push(None);
                    continue;
                };
                let seeds = self.seeds_of(instance, &guest)?;
                let Some(w) = self.model.as_mut() else {
                    return Err(Error::Device("the VM path without a wafer model".into()));
                };
                w.vm_seed_bank(bank, seeds).map_err(device)?;
                self.vm_banks[bank] = Some(instance);
                tracing::info!(?instance, bank, "spatial guest: a request takes a VM lane");
                bank
            };
            placed.push(Some(bank));
        }
        Ok(placed)
    }

    /// A per-lane-program VM's banks for `lanes`: each lane its own bank,
    /// else a free one; when a lane comes in, every live lane's program
    /// placed again (the arena and windows compacted, the newcomers seeded
    /// from their channels). A newcomer the image has no room for stays on
    /// the host (`None`).
    pub(super) fn place_vm_lanes(
        &mut self,
        lanes: &[(InstanceId, &PagedLane)],
        n: usize,
    ) -> EngineResult<Vec<Option<usize>>> {
        let device = |e: String| Error::Device(format!("the wafer model's VM: {e}"));
        let sets = self.vm_sets();
        self.vm_banks.resize(n, None);
        self.vm_bank_used.resize(n, 0);
        let before = self.vm_banks.clone();
        let mut placed: Vec<Option<usize>> = Vec::with_capacity(lanes.len());
        let mut fresh: Vec<usize> = Vec::new();
        // Lanes this fire leaves out, least recently fired first: a newcomer
        // the banks or the image have no room for takes theirs (their windows
        // parked word for word; they place again on their next fire).
        let mut victims = {
            let current: Vec<InstanceId> = lanes.iter().map(|(i, _)| *i).collect();
            let owners: Vec<Option<InstanceId>> = self
                .vm_banks
                .iter()
                .map(|o| o.filter(|o| !current.contains(o)))
                .collect();
            eviction_order(&owners, &self.vm_bank_used)
        };
        let mut evicted: Vec<(usize, InstanceId)> = Vec::new();
        for &(instance, _) in lanes {
            if let Some(bank) = self.vm_banks.iter().position(|o| *o == Some(instance)) {
                placed.push(Some(bank));
                continue;
            }
            // Set A first, else set B: a free bank in a set whose image
            // takes the program, else an idle lane's there; else the host.
            let takes: Vec<bool> = sets
                .iter()
                .map(|(lay, ring, _)| self.set_takes(lay, ring, instance))
                .collect();
            let ranges: Vec<std::ops::Range<usize>> =
                sets.iter().map(|(_, _, r)| r.clone()).collect();
            let bank = pick_bank(&takes, &ranges, &self.vm_banks, &mut victims);
            if let Some(bank) = bank {
                evicted.extend(self.vm_banks[bank].map(|o| (bank, o)));
                self.vm_banks[bank] = Some(instance);
                fresh.push(bank);
            }
            placed.push(bank);
        }
        if fresh.is_empty() && !self.vm_bank_progs.is_empty() {
            return Ok(placed);
        }
        let mut programs: Vec<Option<guest4::slots::LaneProgram>> = vec![None; n];
        let mut seeds: Vec<Option<Vec<guest4::wafer::Seed>>> = (0..n).map(|_| None).collect();
        let mut parked: Vec<Option<guest4::slots::Parked>> = vec![None; n];
        for bank in 0..n {
            let Some(owner) = self.vm_banks[bank] else {
                continue;
            };
            let (key, program) = self.guest_of(owner)?;
            if fresh.contains(&bank) {
                // An evicted lane gets its window back; its channels seed it
                // if the window no longer fits its program.
                parked[bank] = self.vm_parked.get(&owner).cloned();
                seeds[bank] = Some(self.seeds_of(owner, &program)?);
            }
            programs[bank] = Some(guest4::slots::LaneProgram { key, program });
        }
        // Each set's arena and bank: evict there until its newcomers fit, or
        // no one is left to.
        for (lay, _, range) in &sets {
            while guest4::slots::Slots::plan(lay, &programs[range.clone()]).is_err() {
                let Some(at) = victims.iter().position(|b| range.contains(b)) else {
                    break;
                };
                let bank = victims.remove(at);
                evicted.extend(self.vm_banks[bank].map(|o| (bank, o)));
                self.vm_banks[bank] = None;
                programs[bank] = None;
            }
        }
        for (bank, instance) in &evicted {
            tracing::info!(
                ?instance,
                bank,
                "spatial guest: an idle lane leaves the VM for a newcomer"
            );
        }
        let changed = |range: &std::ops::Range<usize>| {
            self.vm_bank_progs.is_empty()
                || fresh
                    .iter()
                    .chain(evicted.iter().map(|(b, _)| b))
                    .any(|b| range.contains(b))
        };
        let touched: Vec<bool> = sets.iter().map(|(_, _, range)| changed(range)).collect();
        let mut seeds_of_set: Vec<Vec<Option<Vec<guest4::wafer::Seed>>>> = sets
            .iter()
            .map(|(_, _, range)| seeds.drain(..range.len()).collect())
            .collect();
        let mut left: Vec<Option<guest4::slots::Parked>> = vec![None; n];
        for (set, (_, _, range)) in sets.iter().enumerate() {
            if !touched[set] {
                continue;
            }
            let Some(w) = self.model.as_mut() else {
                return Err(Error::Device("the VM path without a wafer model".into()));
            };
            let seeds = std::mem::take(&mut seeds_of_set[set]);
            let out = match w.vm_place_lanes(
                set,
                programs[range.clone()].to_vec(),
                seeds,
                parked[range.clone()].to_vec(),
            ) {
                Ok(out) => {
                    for &bank in fresh.iter().filter(|b| range.contains(b)) {
                        tracing::info!(instance = ?self.vm_banks[bank], bank, set, "spatial guest: a request takes a VM lane");
                    }
                    out
                }
                Err(why) => {
                    // The set's newcomers stay on the host; its lanes already
                    // placed run on.
                    let mine: Vec<usize> = fresh
                        .iter()
                        .copied()
                        .filter(|b| range.contains(b))
                        .collect();
                    tracing::info!(%why, set, lanes = mine.len(), "spatial guest: no room on the VM for new lanes");
                    for &bank in &mine {
                        self.vm_banks[bank] = None;
                        programs[bank] = None;
                    }
                    for slot in &mut placed {
                        if slot.is_some_and(|b| mine.contains(&b)) {
                            *slot = None;
                        }
                    }
                    if self.vm_bank_progs.is_empty() {
                        continue;
                    }
                    let seeds = (0..range.len()).map(|_| None).collect();
                    let Some(w) = self.model.as_mut() else {
                        return Err(Error::Device("the VM path without a wafer model".into()));
                    };
                    w.vm_place_lanes(
                        set,
                        programs[range.clone()].to_vec(),
                        seeds,
                        vec![None; range.len()],
                    )
                    .map_err(device)?
                }
            };
            for (k, window) in out.into_iter().enumerate() {
                left[range.start + k] = window;
            }
        }
        // The windows of the lanes that left: an evicted instance's kept for
        // its next fire; one placed again gets a fresh entry when it leaves.
        for (bank, window) in left.into_iter().enumerate() {
            if let (Some(window), Some(Some(owner))) = (window, before.get(bank))
                && self.vm_banks.get(bank) != Some(&Some(*owner))
            {
                self.vm_parked.insert(*owner, window);
            }
        }
        for owner in self.vm_banks.iter().flatten() {
            self.vm_parked.remove(owner);
        }
        self.vm_bank_progs = programs.iter().map(|p| p.as_ref().map(|p| p.key)).collect();
        Ok(placed)
    }

    /// The wafer's VM sets with their rings and banks: set A's lanes
    /// first, then set B's (a global bank is a head lane).
    pub(super) fn vm_sets(
        &self,
    ) -> Vec<(
        guest4::wafer::VmLayout,
        guest4::lane::Ring,
        std::ops::Range<usize>,
    )> {
        let Some(w) = &self.model else {
            return Vec::new();
        };
        let Some((a, ring)) = w.vm else {
            return Vec::new();
        };
        let na = ring.lanes as usize;
        let mut sets = vec![(a, ring, 0..na)];
        if let Some((b, ring_b)) = w.vm_b {
            sets.push((b, ring_b, na..na + ring_b.lanes as usize));
        }
        sets
    }

    /// Whether a set's image (`lay`, `ring`) takes `instance`'s program:
    /// its limits, its ring's width, and a forecast that keeps it on the
    /// wafer.
    pub(super) fn set_takes(
        &self,
        lay: &guest4::wafer::VmLayout,
        ring: &guest4::lane::Ring,
        instance: InstanceId,
    ) -> bool {
        let Ok((program, guest)) = self.guest_of(instance) else {
            return false;
        };
        !self.guest_costly.contains_key(&program)
            && lay
                .refuses(&guest.table)
                .or_else(|| ring.refuses_outputs(&guest.table.outputs))
                .is_none()
    }

    /// `instance`'s program id and guest table.
    pub(super) fn guest_of(
        &self,
        instance: InstanceId,
    ) -> EngineResult<(u64, std::sync::Arc<guest4::lane::Program>)> {
        let (program, _, _) = self
            .programs
            .instance_view(instance)
            .ok_or_else(|| Error::Invalid(format!("instance {instance:?} is gone")))?;
        let guest = self
            .guest_programs
            .get(&program.0)
            .cloned()
            .ok_or_else(|| Error::Invalid("the program has no guest table".into()))?;
        Ok((program.0, guest))
    }

    /// `instance`'s homes' seeds: its channels as they stand.
    pub(super) fn seeds_of(
        &self,
        instance: InstanceId,
        guest: &guest4::lane::Program,
    ) -> EngineResult<Vec<guest4::wafer::Seed>> {
        let (_, _, inst) = self
            .programs
            .instance_view(instance)
            .ok_or_else(|| Error::Invalid(format!("instance {instance:?} is gone")))?;
        Ok(guest
            .table
            .homes
            .iter()
            .map(|h| {
                guest4::wafer::Seed::of(
                    h,
                    guest4::lane::words_of(&inst.channels[h.chan as usize].current()),
                )
            })
            .collect())
    }

    /// `instance`'s step input this launch: the homes its program has the
    /// host write each step, from their channels (what the inferlet put).
    pub(super) fn step_input(
        &self,
        instance: InstanceId,
        lay: &guest4::wafer::VmLayout,
    ) -> EngineResult<wafer::StepInput> {
        let (_, guest) = self.guest_of(instance)?;
        if guest.table.per_step.is_empty() {
            return Ok(None);
        }
        let entries = guest4::slots::step_entries(lay, &guest.table).map_err(Error::Device)?;
        let (_, _, inst) = self
            .programs
            .instance_view(instance)
            .ok_or_else(|| Error::Invalid(format!("instance {instance:?} is gone")))?;
        let values = entries
            .iter()
            .map(|e| guest4::lane::words_of(&inst.channels[e.chan as usize].current()))
            .collect();
        Ok(Some((entries, values)))
    }

    /// `instance`'s epilogue as the head's VM ran it: eta-exec steps the
    /// instance through [`guest4::lane::VmStage`] on the step's `out` words,
    /// and the pool homes (the next token's among them) come back from the
    /// VM's pool; wide homes stay on the wafer.
    pub(super) fn fire_vm(
        &mut self,
        instance: InstanceId,
        row: &[u32],
        pool: &[u32],
    ) -> EngineResult<()> {
        let Some((program, _, _)) = self.programs.instance_view(instance) else {
            return Ok(());
        };
        let guest = self
            .guest_programs
            .get(&program.0)
            .cloned()
            .ok_or_else(|| Error::Invalid("the program has no guest table".into()))?;
        // The pool homes the step put (the next token's among them), as the
        // VM's pool holds them: put on the instance; wide homes stay on the
        // wafer.
        let homes = {
            let Some((_, plan, _)) = self.programs.instance_view(instance) else {
                return Ok(());
            };
            let puts: std::collections::BTreeSet<u32> = plan.package.stages[guest.epilogue]
                .puts
                .iter()
                .map(|p| p.channel)
                .collect();
            guest
                .table
                .homes
                .iter()
                .filter(|h| h.kind == "pool" && puts.contains(&h.chan))
                .filter_map(|h| {
                    let (at, n) = (h.at as usize, h.n as usize);
                    pool.get(at..at + n)
                        .map(|words| (h.chan, guest4::lane::value_of(words, h.dtype)))
                })
                .collect::<Vec<_>>()
        };
        self.programs
            .fire_with(
                instance,
                &[],
                Stage::Epilogue,
                &mut guest4::lane::VmStage {
                    program: &guest,
                    out: row,
                    wide: &[],
                    homes: &homes,
                },
            )
            .map_err(refusal)?;
        self.guest_steps += 1;
        if self.guest_steps.is_power_of_two() {
            tracing::info!(
                steps = self.guest_steps,
                program = program.0,
                "spatial guest: epilogue steps run on the wafer's VM"
            );
        }
        Ok(())
    }

    /// `instance`'s epilogue through the guest contract, if its program has
    /// a table and the mode runs one; `false` when the host interpreter
    /// should run it as usual.
    pub(super) fn fire_guest(
        &mut self,
        instance: InstanceId,
        outputs: &[(String, u32, Value)],
    ) -> EngineResult<bool> {
        if self.guest_mode() != "check" {
            return Ok(false);
        }
        let Some((program, plan, inst)) = self.programs.instance_view(instance) else {
            return Ok(false);
        };
        let Some(guest) = self.guest_programs.get(&program.0).cloned() else {
            return Ok(false);
        };
        let vm = self
            .guest_vms
            .entry(instance)
            .or_insert_with(|| guest4::lane::CheckVm::load(&plan, inst));
        let (out, wide, homes) = vm
            .step(&plan, &guest, inst, outputs)
            .map_err(|why| Error::Device(format!("the guest VM: {why}")))?;
        self.guest_steps += 1;
        if self.guest_steps.is_power_of_two() {
            tracing::info!(
                steps = self.guest_steps,
                program = program.0,
                "spatial guest: epilogue steps run through the VM contract"
            );
        }
        self.programs
            .fire_with(
                instance,
                outputs,
                Stage::Epilogue,
                &mut guest4::lane::VmStage {
                    program: &guest,
                    out: &out,
                    wide: &wide,
                    homes: &homes,
                },
            )
            .map_err(refusal)?;
        if let (Some(vm), Some((_, _, inst))) = (
            self.guest_vms.get(&instance),
            self.programs.instance_view(instance),
        ) {
            vm.read_back(&guest, inst);
        }
        Ok(true)
    }
}

/// A newcomer's bank: a free one in the first set whose image takes it
/// (`takes`, set A before set B; `ranges`, each set's banks), else the
/// least recently fired idle lane's (`victims`, in that order) in a set
/// that takes it, which leaves `victims`; `None`: the host samples it.
fn pick_bank(
    takes: &[bool],
    ranges: &[std::ops::Range<usize>],
    banks: &[Option<InstanceId>],
    victims: &mut Vec<usize>,
) -> Option<usize> {
    let set_of = |bank: usize| ranges.iter().position(|r| r.contains(&bank));
    let free = ranges
        .iter()
        .zip(takes)
        .filter(|(_, t)| **t)
        .find_map(|(range, _)| {
            range
                .clone()
                .find(|&b| banks.get(b).is_some_and(Option::is_none))
        });
    free.or_else(|| {
        let at = victims
            .iter()
            .position(|&b| set_of(b).is_some_and(|s| takes.get(s) == Some(&true)))?;
        Some(victims.remove(at))
    })
}

/// The banks an eviction may take (`owners`: a bank's owner if it may go),
/// least recently fired (`used`) first.
fn eviction_order(owners: &[Option<InstanceId>], used: &[u64]) -> Vec<usize> {
    let mut banks: Vec<usize> = (0..owners.len()).filter(|&b| owners[b].is_some()).collect();
    banks.sort_by_key(|&b| (used.get(b).copied().unwrap_or(0), b));
    banks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_newcomer_takes_set_a_then_set_b_then_an_idle_lane_where_it_fits() {
        let (a, b) = (0..2, 2..4);
        let (x, y) = (Some(InstanceId(1)), Some(InstanceId(2)));
        // A free bank in A first; with A full, one in B.
        assert_eq!(
            pick_bank(
                &[true, true],
                &[a.clone(), b.clone()],
                &[None, x, None, None],
                &mut vec![]
            ),
            Some(0)
        );
        assert_eq!(
            pick_bank(
                &[true, true],
                &[a.clone(), b.clone()],
                &[x, y, None, None],
                &mut vec![]
            ),
            Some(2)
        );
        // Typical (B's image only) goes to B even with A free.
        assert_eq!(
            pick_bank(
                &[false, true],
                &[a.clone(), b.clone()],
                &[None, None, None, None],
                &mut vec![]
            ),
            Some(2)
        );
        // Tail-free (neither) stays on the host, whatever is free or idle.
        let mut idle = vec![0, 2];
        assert_eq!(
            pick_bank(
                &[false, false],
                &[a.clone(), b.clone()],
                &[x, None, y, None],
                &mut idle
            ),
            None
        );
        assert_eq!(idle, [0, 2]);
        // Every bank held: an idle lane's, only in a set that takes it.
        let full = [x, y, x, y];
        let mut idle = vec![0, 3];
        assert_eq!(
            pick_bank(&[false, true], &[a.clone(), b.clone()], &full, &mut idle),
            Some(3)
        );
        assert_eq!(idle, [0]);
        assert_eq!(pick_bank(&[false, true], &[a, b], &full, &mut idle), None);
    }

    #[test]
    fn an_eviction_takes_the_least_recently_fired_bank_that_may_go() {
        let (a, b, c) = (
            Some(InstanceId(1)),
            Some(InstanceId(2)),
            Some(InstanceId(3)),
        );
        // Bank 1 fired longest ago; bank 2 has no owner that may go (idle,
        // or its request is in the fire).
        assert_eq!(
            eviction_order(&[a, b, None, c], &[7, 3, 1, 5]),
            vec![1, 3, 0]
        );
        assert_eq!(eviction_order(&[None, None], &[0, 0]), Vec::<usize>::new());
        // Ties: the lower bank first.
        assert_eq!(eviction_order(&[a, b], &[4, 4]), vec![0, 1]);
    }
}
