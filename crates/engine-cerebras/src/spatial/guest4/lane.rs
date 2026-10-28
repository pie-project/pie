//! The engine's side of the guest VM contract (`spatial/eta/CONTRACT.md`):
//! what a lane's epilogue looks like to eta-exec when the VM runs it. The VM
//! owns the program's home channels (it keeps them between steps and the host
//! sees them only when it reads them back) and computes every value the
//! stage puts; the host is handed the puts of the channels it takes, as the
//! words of the VM's `out`.
//!
//! [`VmStage`] is that view as an eta-exec [`StageRunner`]: the logits are
//! the VM's (bound, never handed over), home channels are kept, and the
//! stage's puts are the `out` words for host-taken channels and empty (the
//! ring as it is) for homes. The other stages (a prologue) still run in the
//! interpreter; a program whose prologue touches a home is not eligible.

use std::collections::BTreeSet;

use eta_compiler::codegen::launch::{LaunchValue, ValueOrigin};
use eta_exec::{ExecPlan, Interpreted, StagePlan, StageRunner, Value};
use eta_ir::Dtype;
use eta_ir::registry::Stage;

use super::Table;

/// Words of `out` a step fills (the VM's out block).
pub const OUT_WORDS: u32 = 16;

/// A value as VM words: f32 as bits, ints as their two's complement, bool
/// 0/1.
#[must_use]
pub fn words_of(value: &Value) -> Vec<u32> {
    match value {
        Value::F32(v) => v.iter().map(|x| x.to_bits()).collect(),
        Value::I32(v) => v.iter().map(|&x| x as u32).collect(),
        Value::U32(v) => v.clone(),
        Value::Bool(v) => v.iter().map(|&b| u32::from(b != 0)).collect(),
    }
}

/// VM words as a value of `dtype`.
#[must_use]
pub fn value_of(words: &[u32], dtype: Dtype) -> Value {
    match dtype {
        Dtype::I32 => Value::I32(words.iter().map(|&w| w as i32).collect()),
        Dtype::U32 => Value::U32(words.to_vec()),
        Dtype::Bool => Value::Bool(words.iter().map(|&w| u8::from(w != 0)).collect()),
        _ => Value::F32(words.iter().map(|&w| f32::from_bits(w)).collect()),
    }
}

/// A program the VM can run, as the engine keeps it: its table and the
/// stage it replaces.
pub struct Program {
    pub table: Table,
    /// `plan.package.stages` index of the epilogue the table is.
    pub epilogue: usize,
    /// The channels the VM keeps (its homes).
    pub homes: BTreeSet<u32>,
    /// The name of the logits output the VM binds.
    pub logits: String,
}

impl Program {
    /// `table` for `plan`, when the engine can hand the stage over: one
    /// epilogue, every output it reads the logits, its host outputs inside
    /// `out`, and no other stage touching a home.
    pub fn of(plan: &ExecPlan, table: Table, logits: &str) -> Result<Program, String> {
        let stages = &plan.package.stages;
        let epilogues: Vec<usize> = (0..stages.len())
            .filter(|&i| stages[i].stage == Stage::Epilogue)
            .collect();
        let [epilogue] = epilogues[..] else {
            return Err(format!("{} epilogue stages", epilogues.len()));
        };
        if let Some(other) = plan
            .package
            .values
            .iter()
            .filter(|v| v.source == ValueOrigin::Output)
            .find_map(|v| v.output.as_deref().filter(|name| *name != logits))
        {
            return Err(format!("the program reads output `{other}`"));
        }
        if let Some((chan, slot, n, _)) =
            table.outputs.iter().find(|(_, slot, n, _)| slot + n > OUT_WORDS)
        {
            return Err(format!(
                "host channel {chan} takes out words {slot}..{} of {OUT_WORDS}",
                slot + n
            ));
        }
        let homes: BTreeSet<u32> = table.homes.iter().map(|h| h.chan).collect();
        for (at, stage) in stages.iter().enumerate() {
            if at == epilogue {
                continue;
            }
            let touched = stage
                .takes
                .iter()
                .chain(&stage.reads)
                .chain(stage.puts.iter().map(|p| &p.channel))
                .find(|c| homes.contains(c));
            if let Some(chan) = touched {
                return Err(format!(
                    "a {:?} stage uses channel {chan}, which the VM keeps",
                    stage.stage
                ));
            }
        }
        Ok(Program {
            table,
            epilogue,
            homes,
            logits: logits.to_string(),
        })
    }
}

/// One step of `program` as eta-exec sees it when the VM ran it and left
/// `out`, and the rows of its vocabulary-wide host outputs (read from the
/// VM's `wide` registers, `Table::wide_outputs`).
pub struct VmStage<'a> {
    pub program: &'a Program,
    pub out: &'a [u32],
    pub wide: &'a [(u32, Value)],
}

impl StageRunner for VmStage<'_> {
    fn run(
        &mut self,
        plan: &ExecPlan,
        sp: &StagePlan,
        roots: &[u32],
        wanted: &[u32],
        vals: &mut [Value],
    ) -> eta_exec::Result<()> {
        if sp.stage_index != self.program.epilogue {
            return Interpreted.run(plan, sp, roots, wanted, vals);
        }
        for put in &plan.package.stages[sp.stage_index].puts {
            let host = self
                .program
                .table
                .outputs
                .iter()
                .find(|(chan, ..)| *chan == put.channel);
            let wide = self.wide.iter().find(|(chan, _)| *chan == put.channel);
            vals[put.value as usize] = match (host, wide) {
                (_, Some((_, value))) => value.clone(),
                (Some(&(_, slot, n, dtype)), None) => {
                    let words = self
                        .out
                        .get(slot as usize..(slot + n) as usize)
                        .ok_or_else(|| eta_exec::Error {
                            message: format!(
                                "the VM's out holds {} words; channel {} reads {slot}..{}",
                                self.out.len(),
                                put.channel,
                                slot + n
                            ),
                        })?;
                    value_of(words, dtype)
                }
                // A home: the VM wrote it in place; the ring stays as it is.
                (None, None) => Value::F32(Vec::new()),
            };
        }
        Ok(())
    }

    fn binds(&self, root: &LaunchValue) -> bool {
        root.output.as_deref() == Some(self.program.logits.as_str())
    }

    fn keeps(&self, channel: u32) -> bool {
        self.program.homes.contains(&channel)
    }
}

/// `src`'s ring copied into `dst` (every cell, and where it stands).
pub fn copy_ring(dst: &eta_exec::ChannelState, src: &eta_exec::ChannelState) {
    for sequence in 0..=src.capacity() as u64 {
        dst.encode_sequence(sequence, &src.decode_sequence(sequence));
    }
    dst.store_head(src.head());
    dst.store_tail(src.tail());
}

/// The VM's side played by eta's interpreter on a copy of the instance (the
/// engine's check of this contract, `PIE_SPATIAL_GUEST=check`): the copy
/// holds the homes as the VM would, a step runs the epilogue on it and leaves
/// `out` as the VM would, and the instance itself steps through [`VmStage`].
/// The homes are copied back after each step (the host side keeps reading
/// them: an envelope's tokens) and the ones the host writes are copied in
/// before it.
pub struct CheckVm {
    shadow: eta_exec::InterpInstance,
}

impl CheckVm {
    /// The VM loaded with `inst`'s channels as they stand (CONTRACT.md §2:
    /// the homes seeded).
    #[must_use]
    pub fn load(plan: &ExecPlan, inst: &eta_exec::InterpInstance) -> CheckVm {
        let channels = plan
            .package
            .channels
            .iter()
            .zip(&inst.channels)
            .map(|(decl, ring)| {
                let numel = decl.shape.iter().product::<u32>().max(1) as usize;
                let copy = std::sync::Arc::new(eta_exec::ChannelState::host(
                    eta_exec::concrete_dtype(decl.dtype),
                    numel,
                    ring.capacity(),
                ));
                copy_ring(&copy, ring);
                copy
            })
            .collect();
        CheckVm {
            shadow: eta_exec::InterpInstance {
                channels,
                poisoned: false,
            },
        }
    }

    /// One step over `outputs` (the logits): the epilogue on the copy, the
    /// `out` words of the host-taken puts and the vocabulary-wide ones' rows.
    /// The host-written homes (`per_step` channels) are brought in from
    /// `inst` first.
    pub fn step(
        &mut self,
        plan: &ExecPlan,
        program: &Program,
        inst: &eta_exec::InterpInstance,
        outputs: &[(String, u32, Value)],
    ) -> Result<(Vec<u32>, Vec<(u32, Value)>), String> {
        for &chan in &program.table.per_step {
            copy_ring(&self.shadow.channels[chan as usize], &inst.channels[chan as usize]);
        }
        let inputs = eta_exec::PassInputs { outputs };
        match eta_exec::step_with(&mut self.shadow, plan, &inputs, &mut Interpreted, Stage::Epilogue) {
            eta_exec::StepOutcome::Committed => {}
            other => return Err(format!("the check VM's step: {other:?}")),
        }
        let mut out = vec![0u32; OUT_WORDS as usize];
        for &(chan, slot, n, _) in &program.table.outputs {
            // The put waits in the copy's ring; the host takes the real one.
            let ring = &self.shadow.channels[chan as usize];
            let value = ring.pop().ok_or_else(|| format!("channel {chan} got no put"))?;
            let words = words_of(&value);
            out[slot as usize..(slot + n) as usize]
                .copy_from_slice(&words[..n as usize]);
        }
        let mut wide = Vec::new();
        for (chan, _, _) in &program.table.wide_outputs {
            let ring = &self.shadow.channels[*chan as usize];
            let value = ring.pop().ok_or_else(|| format!("channel {chan} got no put"))?;
            wide.push((*chan, value));
        }
        Ok((out, wide))
    }

    /// The homes back into `inst` (what a read-back of the VM's pool and wide
    /// rows gives).
    pub fn read_back(&self, program: &Program, inst: &eta_exec::InterpInstance) {
        for &chan in &program.homes {
            copy_ring(&inst.channels[chan as usize], &self.shadow.channels[chan as usize]);
        }
    }
}
