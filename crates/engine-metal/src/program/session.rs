use eta_compiler::codegen::launch::LaunchPackage;
use eta_exec::{ExecPlan, Extents};
use eta_ir::op::IntrinsicId;
use eta_ir::registry::GeometryClass;
use eta_ir::types::name_or_unknown;
use eta_ir::validate::Direction;

use crate::device::ctx::Frame;
use crate::device::{Buffer, Context, Pipelines};
use crate::error::{Fault, Result};

use super::compile::Compiled;
use super::launch::{ChannelShape, Cursor, Prepared, Rings};
use super::ports::{self, Envelope};
use super::shared::SharedRing;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fired {
    Committed,
    Blocked(u32),
    Declined,
    Faulted(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blocked {
    pub channel: u32,
    pub needs: Option<Direction>,
    pub live: u64,
    pub capacity: u64,
}

impl std::fmt::Display for Blocked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.needs {
            Some(Direction::NeedsFull) => write!(
                f,
                "channel {} is empty and its program takes from it (needs a cell, \
                 holds {} of {})",
                self.channel, self.live, self.capacity
            ),
            Some(Direction::NeedsEmpty) => write!(
                f,
                "channel {} is full and its program puts into it (needs room, holds \
                 {} of {})",
                self.channel, self.live, self.capacity
            ),
            None => write!(
                f,
                "channel {} holds {} of {}",
                self.channel, self.live, self.capacity
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Launched {
    Airborne,
    Refused(Fired),
}

#[derive(Debug)]
pub struct Session {
    rings: Rings,
    shapes: Vec<ChannelShape>,
    cursors: Vec<Cursor>,
    prepared: Vec<Option<Prepared>>,
    bound: u64,
    poisoned: bool,
    fires: u64,
    airborne: Option<Vec<usize>>,
}

impl Session {
    pub fn bind(
        device: &Context,
        compiled: &Compiled,
        plan: &ExecPlan,
        seeds: &[(u32, Vec<u8>)],
        extents: Extents,
        adopted: &[Option<std::sync::Arc<SharedRing>>],
    ) -> Result<Session> {
        stages_and_plans_agree(compiled)?;
        if plan.package.channels.is_empty() {
            return Err(Fault::program(
                "program::session",
                "an instance with no channels: there is nothing for a fire to read \
                 or publish",
            ));
        }

        let shapes: Vec<ChannelShape> =
            plan.package.channels.iter().map(ChannelShape::of).collect();
        let rings = Rings::allocate(device, &shapes, adopted)?;
        let cursors = vec![Cursor { head: 0, tail: 0 }; shapes.len()];

        let mut prepared = Vec::with_capacity(compiled.plans.len());
        for (index, stage_plan) in compiled.plans.iter().enumerate() {
            let launches = compiled
                .stages
                .get(index)
                .is_some_and(|stage| !stage.regions.is_empty());
            prepared.push(if launches {
                Some(Prepared::build(device, stage_plan, &shapes, extents)?)
            } else {
                None
            });
        }

        let mut session = Session {
            rings,
            shapes,
            cursors,
            prepared,
            bound: 0,
            poisoned: false,
            fires: 0,
            airborne: None,
        };
        for (channel, wire) in seeds {
            if let Some(ring) = session.rings.shared(*channel as usize)
                && !ring.claim_seeding()
            {
                continue;
            }
            if !session.publish(*channel, wire)? {
                return Err(Fault::program(
                    "program::session",
                    format!(
                        "channel {channel}'s seed does not fit: its ring already holds \
                         a cell, so the seed would be the second value rather than the first"
                    ),
                ));
            }
        }
        Ok(session)
    }

    #[must_use]
    pub fn channels(&self) -> usize {
        self.shapes.len()
    }

    #[must_use]
    pub fn shape(&self, channel: u32) -> Option<ChannelShape> {
        self.shapes.get(channel as usize).copied()
    }

    #[must_use]
    pub fn cursor(&self, channel: u32) -> Option<Cursor> {
        let channel = channel as usize;
        if let Some(ring) = self.rings.shared(channel) {
            return Some(ring.cursor());
        }
        self.cursors.get(channel).copied()
    }

    #[must_use]
    pub fn cursors_now(&self) -> Vec<Cursor> {
        let mut cursors = self.cursors.clone();
        for (channel, cursor) in cursors.iter_mut().enumerate() {
            if let Some(ring) = self.rings.shared(channel) {
                *cursor = ring.cursor();
            }
        }
        cursors
    }

    #[must_use]
    pub const fn fires(&self) -> u64 {
        self.fires
    }

    #[must_use]
    pub const fn poisoned(&self) -> bool {
        self.poisoned
    }

    #[must_use]
    pub fn depth(&self, channel: u32) -> u64 {
        self.cursor(channel)
            .map_or(0, |cursor| cursor.tail.saturating_sub(cursor.head))
    }

    pub fn publish(&mut self, channel: u32, wire: &[u8]) -> Result<bool> {
        let shape = self.shape_of(channel)?;
        if self.depth(channel) >= u64::from(shape.capacity) {
            return Ok(false);
        }
        let want = shape.cell_bytes();
        if wire.len() != want {
            return Err(Fault::program(
                "program::session",
                format!(
                    "a {} cell of {} lane(s) on channel {channel} is {want} bytes and {} \
                     were offered",
                    name_or_unknown(shape.dtype),
                    shape.numel,
                    wire.len()
                ),
            ));
        }
        let slot = channel as usize;
        let tail = self.cursor(channel).unwrap_or_default().tail;
        self.rings.write_cell(slot, tail, wire)?;
        match self.rings.shared(slot) {
            Some(ring) => ring.bump_tail(),
            None => self.cursors[slot].tail = tail + 1,
        }
        Ok(true)
    }

    pub fn take(&mut self, channel: u32) -> Result<Option<Vec<u8>>> {
        if self.depth(channel) == 0 {
            return Ok(None);
        }
        let slot = channel as usize;
        let head = self.cursor(channel).unwrap_or_default().head;
        let cell = self.rings.read_cell(slot, head)?;
        match self.rings.shared(slot) {
            Some(ring) => ring.bump_head(),
            None => self.cursors[slot].head = head + 1,
        }
        Ok(Some(cell))
    }

    pub fn feed_cell(&self, channel: u32) -> Result<Option<(&Buffer, u64, u64, eta_ir::Dtype)>> {
        let slot = channel as usize;
        let shape = self.shape_of(channel)?;
        if self.depth(channel) == 0 {
            return Ok(None);
        }
        let head = self.cursor(channel).unwrap_or_default().head;
        let at = self.rings.cell_offset(slot, head)?;
        Ok(Some((
            self.rings.slab(slot)?,
            at,
            shape.cell_bytes() as u64,
            shape.dtype,
        )))
    }

    pub fn peek(&self, channel: u32, sequence: u64) -> Result<Vec<u8>> {
        self.shape_of(channel)?;
        self.rings.read_cell(channel as usize, sequence)
    }

    pub fn envelope(&self, plan: &ExecPlan, class: GeometryClass) -> Result<Envelope> {
        ports::resolve(plan, class, &self.rings, &self.cursors_now(), &self.shapes)
    }

    pub fn bind_intrinsic(
        &mut self,
        intrinsic: IntrinsicId,
        base: &Buffer,
        offset: u64,
        width: u32,
        dtype: eta_ir::Dtype,
    ) -> Result<()> {
        for prepared in self.prepared.iter_mut().flatten() {
            prepared.bind_intrinsic(intrinsic, base, offset, width, dtype)?;
        }
        self.bound |= 1u64 << (intrinsic as u32);
        Ok(())
    }

    pub fn fire(
        &mut self,
        device: &Context,
        pipelines: &Pipelines,
        compiled: &Compiled,
        plan: &ExecPlan,
    ) -> Result<Fired> {
        if self.airborne.is_some() {
            return Err(Fault::program(
                "program::session",
                "this instance has a pass airborne in someone else's command buffer: a \
                 standalone fire would reset the status word that pass has not been \
                 read from, and both fires would report the second's verdict",
            ));
        }
        if let Some(refused) = self.gate(compiled, plan)? {
            return Ok(refused);
        }

        let cursors = self.cursors_now();
        let mut launched = Vec::with_capacity(compiled.stages.len());
        for (index, stage) in compiled.stages.iter().enumerate() {
            let Some(prepared) = self.prepared.get_mut(index).and_then(Option::as_mut) else {
                continue;
            };
            prepared.refresh(&self.rings, &cursors)?;
            for region in stage.regions.iter() {
                prepared.launch_region(device, pipelines, &self.rings, region)?;
            }
            launched.push(index);
        }

        self.verdict(plan, &launched)
    }

    pub fn stage_into(
        &mut self,
        frame: &Frame,
        compiled: &Compiled,
        plan: &ExecPlan,
    ) -> Result<Launched> {
        if self.airborne.is_some() {
            return Err(Fault::program(
                "program::session",
                "this instance already has a pass airborne: a second staging would \
                 refresh the scratch and the status word of stages whose first pass \
                 has not been read, so both fires would report the second's verdict \
                 and the first's commit would be lost",
            ));
        }
        if let Some(refused) = self.gate(compiled, plan)? {
            return Ok(Launched::Refused(refused));
        }

        let cursors = self.cursors_now();
        let mut launched = Vec::with_capacity(compiled.stages.len());
        for (index, stage) in compiled.stages.iter().enumerate() {
            let Some(prepared) = self.prepared.get_mut(index).and_then(Option::as_mut) else {
                continue;
            };
            prepared.refresh(&self.rings, &cursors)?;
            for region in stage.regions.iter() {
                prepared.encode_into(frame, region)?;
            }
            launched.push(index);
        }

        self.airborne = Some(launched);
        Ok(Launched::Airborne)
    }

    pub fn prepare_airborne(
        &mut self,
        compiled: &Compiled,
        plan: &ExecPlan,
    ) -> Result<Option<Fired>> {
        if self.airborne.is_some() {
            return Err(Fault::program(
                "program::session",
                "this instance already has a pass airborne: a second staging would \
                 refresh the cells and the status word of stages whose first pass \
                 has not been read, so both fires would report the second's verdict \
                 and the first's commit would be lost",
            ));
        }
        if let Some(refused) = self.gate(compiled, plan)? {
            return Ok(Some(refused));
        }
        let cursors = self.cursors_now();
        let mut launched = Vec::with_capacity(compiled.stages.len());
        for (index, stage) in compiled.stages.iter().enumerate() {
            let Some(prepared) = self.prepared.get_mut(index).and_then(Option::as_mut) else {
                continue;
            };
            if stage.regions.is_empty() {
                continue;
            }
            prepared.refresh_cells(&self.rings, &cursors)?;
            launched.push(index);
        }
        self.airborne = Some(launched);
        Ok(None)
    }

    pub fn prepared_mut(&mut self, stage: usize) -> Option<&mut Prepared> {
        self.prepared.get_mut(stage).and_then(Option::as_mut)
    }

    pub fn settle_launched(&mut self, plan: &ExecPlan) -> Result<Fired> {
        let Some(launched) = self.airborne.take() else {
            return Ok(Fired::Committed);
        };
        self.verdict(plan, &launched)
    }

    pub fn abandon_launched(&mut self) {
        self.airborne = None;
    }

    #[must_use]
    pub const fn is_airborne(&self) -> bool {
        self.airborne.is_some()
    }

    fn gate(&mut self, compiled: &Compiled, plan: &ExecPlan) -> Result<Option<Fired>> {
        if self.poisoned {
            return Ok(Some(Fired::Faulted("instance is poisoned".to_string())));
        }
        stages_and_plans_agree(compiled)?;

        let readout_bound = [
            IntrinsicId::Logits,
            IntrinsicId::Velocity,
            IntrinsicId::Hidden,
        ]
        .iter()
        .any(|id| self.bound & (1u64 << (*id as u32)) != 0);
        if plan.needs_logits && !readout_bound {
            return Err(Fault::program(
                "program::session",
                "this program reads the fire's readout (`logits`, `velocity` or \
                 `hidden`) and no buffer has been bound to it; the emitted kernel \
                 reads the argument slot regardless of whether anything was ever \
                 encoded into it",
            ));
        }
        if plan.needs_mtp_logits && self.bound & (1u64 << (IntrinsicId::MtpLogits as u32)) == 0 {
            return Err(Fault::program(
                "program::session",
                "this program reads the `mtp_logits` intrinsic and no buffer has \
                 been bound to it; a model whose text declares no `mtp` export has \
                 no draft column for it to point at",
            ));
        }

        if plan.needs_mtp_drafts && self.bound & (1u64 << (IntrinsicId::MtpDrafts as u32)) == 0 {
            return Err(Fault::program(
                "program::session",
                "this program reads the `mtp_drafts` intrinsic and no buffer has \
                 been bound to it; a model whose text declares no `mtp.drafts` export \
                 has no token plane for it to point at",
            ));
        }

        if plan.needs_attn_scores && self.bound & (1u64 << (IntrinsicId::AttnScore as u32)) == 0 {
            return Err(Fault::program(
                "program::session",
                "this program reads the `attn_score` intrinsic and no buffer has been \
                 bound to it; a lane that did not ask to capture its attention has no \
                 block of the observability slab for it to point at",
            ));
        }

        if let Some(blocked) = self.blocked_channel(plan) {
            return Ok(Some(Fired::Blocked(blocked)));
        }

        Ok(None)
    }

    fn verdict(&mut self, plan: &ExecPlan, launched: &[usize]) -> Result<Fired> {
        let max_channel = u32::try_from(self.shapes.len().saturating_sub(1)).unwrap_or(u32::MAX);

        let mut verdict = Verdict::Committed;
        for &index in launched {
            let Some(prepared) = self.prepared.get(index).and_then(Option::as_ref) else {
                continue;
            };
            verdict = verdict.worse(verdict_of(
                &plan.package,
                index,
                prepared.status()?,
                max_channel,
            ));
        }

        match verdict {
            Verdict::Faulted(why) => {
                self.poisoned = true;
                return Ok(Fired::Faulted(why));
            }
            Verdict::Declined => return Ok(Fired::Declined),
            Verdict::Committed => {}
        }

        match self.commit(plan) {
            Ok(()) => {
                self.fires += 1;
                Ok(Fired::Committed)
            }
            Err(why) => {
                self.poisoned = true;
                Ok(Fired::Faulted(why))
            }
        }
    }

    #[must_use]
    pub fn blocked_channel(&self, plan: &ExecPlan) -> Option<u32> {
        self.readiness(plan).map(|blocked| blocked.channel)
    }

    #[must_use]
    pub fn readiness(&self, plan: &ExecPlan) -> Option<Blocked> {
        for channel in 0..self.shapes.len() {
            let needs = plan
                .package
                .channels
                .get(channel)
                .and_then(|declared| declared.readiness);
            let live = self.depth(channel as u32);
            let capacity = u64::from(self.shapes[channel].capacity);
            let ready = match needs {
                Some(Direction::NeedsFull) => live != 0,
                Some(Direction::NeedsEmpty) => live < capacity,
                None => true,
            };
            if !ready {
                return Some(Blocked {
                    channel: channel as u32,
                    needs,
                    live,
                    capacity,
                });
            }
        }
        None
    }

    fn commit(&mut self, plan: &ExecPlan) -> std::result::Result<(), String> {
        let now = self.cursors_now();
        let mut next = now.clone();
        let mut bumps: Vec<(usize, bool, bool)> = Vec::new();
        for (channel, cursor) in now.iter().enumerate() {
            if cursor.tail < cursor.head {
                return Err(format!("channel {channel}: tail precedes head at commit"));
            }
            let mut used = cursor.tail - cursor.head;
            let capacity = u64::from(self.shapes[channel].capacity);
            let took = plan.takes_channel(channel as u32) && used != 0;
            if took {
                next[channel].head = cursor.head + 1;
                used -= 1;
            }
            let put = plan.puts_channel(channel as u32);
            if put {
                if used >= capacity {
                    return Err(format!(
                        "channel {channel}: put overflows capacity {capacity} at commit"
                    ));
                }
                next[channel].tail = cursor.tail + 1;
            }
            if (took || put) && self.rings.shared(channel).is_some() {
                bumps.push((channel, took, put));
            }
        }
        for (channel, took, put) in bumps {
            let Some(ring) = self.rings.shared(channel) else {
                continue;
            };
            if took {
                ring.bump_head();
            }
            if put {
                ring.bump_tail();
            }
        }
        self.cursors = next;
        Ok(())
    }

    fn shape_of(&self, channel: u32) -> Result<ChannelShape> {
        self.shapes.get(channel as usize).copied().ok_or_else(|| {
            Fault::program(
                "program::session",
                format!("channel {channel} is not one this instance carries"),
            )
        })
    }
}

#[derive(Clone, Debug)]
enum Verdict {
    Committed,
    Declined,
    Faulted(String),
}

impl Verdict {
    fn worse(self, other: Verdict) -> Verdict {
        match (self, other) {
            (fault @ Verdict::Faulted(_), _) => fault,
            (_, fault @ Verdict::Faulted(_)) => fault,
            (Verdict::Declined, _) | (_, Verdict::Declined) => Verdict::Declined,
            (Verdict::Committed, Verdict::Committed) => Verdict::Committed,
        }
    }
}

fn verdict_of(
    package: &LaunchPackage,
    stage: usize,
    status: eta_exec::Status,
    max_channel: u32,
) -> Verdict {
    match status.state() {
        Some(eta_exec::State::Running) => Verdict::Committed,
        Some(eta_exec::State::Committed) => Verdict::Committed,
        Some(eta_exec::State::Retry) => Verdict::Declined,
        Some(eta_exec::State::Fault | eta_exec::State::Unset) | None => Verdict::Faulted(format!(
            "stage {stage}: {}",
            eta_exec::report_status(package, status, true, max_channel)
        )),
    }
}

#[must_use]
pub fn seeds_of(interp: &eta_exec::InterpInstance, plan: &ExecPlan) -> Vec<(u32, Vec<u8>)> {
    let mut seeds = Vec::new();
    for (channel, ring) in interp.channels.iter().enumerate() {
        let declared = match plan.package.channels.get(channel) {
            Some(declared) => declared,
            None => continue,
        };
        let dtype = eta_exec::concrete_dtype(declared.dtype);
        let numel = declared
            .shape
            .iter()
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1);
        for sequence in ring.head()..ring.tail() {
            let mut wire = vec![0u8; eta_exec::wire_cell_bytes(dtype, numel)];
            eta_exec::encode_wire(&ring.decode_sequence(sequence), &mut wire);
            seeds.push((channel as u32, wire));
        }
    }
    seeds
}

fn stages_and_plans_agree(compiled: &Compiled) -> Result<()> {
    stage_plans_are_parallel(
        &compiled
            .stages
            .iter()
            .map(|stage| (stage.signature_hash, !stage.regions.is_empty()))
            .collect::<Vec<_>>(),
        &compiled
            .plans
            .iter()
            .map(|plan| plan.signature_hash)
            .collect::<Vec<_>>(),
    )
}

fn stage_plans_are_parallel(stages: &[(u64, bool)], plans: &[u64]) -> Result<()> {
    if stages.len() != plans.len() {
        return Err(Fault::program(
            "program::session",
            format!(
                "this program has {} compiled stage(s) and {} plan(s): the fire pairs \
                 them by index to prepare each stage's own scratch, so it cannot tell \
                 which plan describes which stage",
                stages.len(),
                plans.len()
            ),
        ));
    }
    for (index, (&(signature, launches), &plan)) in stages.iter().zip(plans).enumerate() {
        if launches && signature != plan {
            return Err(Fault::program(
                "program::session",
                format!(
                    "stage {index} has regions to launch and the plan at that index \
                     describes a different stage: its regions would index scratch, \
                     descriptors and a channel table sized for someone else"
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use eta_compiler::codegen::launch::LaunchPackage;

    use super::{Verdict, verdict_of};

    fn package() -> LaunchPackage {
        LaunchPackage {
            fault_classes: eta_compiler::codegen::fault::classes(),
            ..LaunchPackage::default()
        }
    }

    #[test]
    fn session_every_case() {
        an_adapter_prologue_and_a_sampling_epilogue_are_one_fire();
        two_launching_stages_each_get_their_own_plan();
        a_stage_that_launches_nothing_is_not_compared();
        a_kernel_readiness_miss_is_a_decline_and_not_a_fault();
        a_fault_outranks_a_decline_in_either_order();
    }

    fn an_adapter_prologue_and_a_sampling_epilogue_are_one_fire() {
        super::stage_plans_are_parallel(&[(0xa11, false), (0xb22, true)], &[0xa11, 0xb22])
            .expect("the plans are parallel and the launching stage is its own");
    }

    fn two_launching_stages_each_get_their_own_plan() {
        super::stage_plans_are_parallel(&[(0xa11, true), (0xb22, true)], &[0xa11, 0xb22])
            .expect("both launch, both are prepared from their own plan");
    }

    fn a_stage_that_launches_nothing_is_not_compared() {
        super::stage_plans_are_parallel(&[(0xa11, false)], &[0xdead])
            .expect("nothing is prepared from it");
    }

    fn a_kernel_readiness_miss_is_a_decline_and_not_a_fault() {
        let status = eta_exec::Status {
            state: 2,
            fault: 0x480,
            reserved0: 0,
            reserved1: 0,
        };
        assert!(matches!(
            verdict_of(&package(), 0, status, 3),
            Verdict::Declined
        ));
    }

    fn a_fault_outranks_a_decline_in_either_order() {
        let fault = || Verdict::Faulted("stage 0: guard".to_string());
        assert!(matches!(
            fault().worse(Verdict::Declined),
            Verdict::Faulted(_)
        ));
        assert!(matches!(
            Verdict::Declined.worse(fault()),
            Verdict::Faulted(_)
        ));
        assert!(matches!(
            Verdict::Committed.worse(Verdict::Declined),
            Verdict::Declined
        ));
        assert!(matches!(
            Verdict::Committed.worse(Verdict::Committed),
            Verdict::Committed
        ));
    }
}
