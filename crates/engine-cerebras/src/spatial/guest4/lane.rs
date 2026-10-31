//! The engine's side of the guest VM contract:
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

/// The VM's out ring: `gout` holds
/// `depth` rows of `width` words, step s's host-taken words at row s % D,
/// word `slot`; a launch runs at most D steps before the host reads it. A
/// property of the compiled image, read once after the program loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ring {
    pub depth: u32,
    pub width: u32,
    /// Lanes the VM time-multiplexes (`gring` word 2; 1 on older images):
    /// turn t of a launch is lane t % L, each lane with its own bank.
    pub lanes: u32,
}

impl Ring {
    /// `gout`'s word for step `step`'s output word `slot` (lane 0).
    #[must_use]
    pub fn index(&self, step: u64, slot: u32) -> usize {
        self.index_lane(step, 0, slot)
    }

    /// `gout`'s word for lane `lane`'s step `step`, output word `slot`: its
    /// row is (s·L + l) % D.
    #[must_use]
    pub fn index_lane(&self, step: u64, lane: u32, slot: u32) -> usize {
        let row = (step * u64::from(self.lanes) + u64::from(lane)) % u64::from(self.depth);
        row as usize * self.width as usize + slot as usize
    }

    /// Why a table whose `outputs` are these (channel, slot, words, dtype)
    /// cannot run on this image, if it cannot: an output reaching past a
    /// step's `width` words.
    #[must_use]
    pub fn refuses_outputs(&self, outputs: &[(u32, u32, u32, Dtype)]) -> Option<String> {
        outputs
            .iter()
            .find(|&&(_, slot, n, _)| slot + n > self.width)
            .map(|&(chan, slot, n, _)| {
                format!(
                    "channel {chan} takes out words {slot}..{}; the VM image's ring holds {} a step",
                    slot + n,
                    self.width
                )
            })
    }
}

/// The words of the largest object `name` any PE image in `program_dir`
/// defines (`None`: none does).
pub fn symbol_words(program_dir: &std::path::Path, name: &str) -> Option<usize> {
    let entries = std::fs::read_dir(program_dir.join("bin")).ok()?;
    entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "elf"))
        .filter_map(|p| std::fs::read(&p).ok())
        .filter_map(|image| elf_symbol_size(&image, name))
        .map(|size| size as usize / 4)
        .max()
}

/// The size of the object `name` in a little-endian ELF64 image's symbol
/// table.
fn elf_symbol_size(image: &[u8], name: &str) -> Option<u64> {
    let u16_at = |o: usize| {
        image
            .get(o..o + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
    };
    let u32_at = |o: usize| {
        image
            .get(o..o + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    };
    let u64_at = |o: usize| {
        image
            .get(o..o + 8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    };
    if image.get(..6)? != b"\x7fELF\x02\x01" {
        return None;
    }
    let (shoff, shentsize, shnum) = (
        u64_at(0x28)? as usize,
        u16_at(0x3a)? as usize,
        u16_at(0x3c)? as usize,
    );
    let section = |i: usize| shoff + i * shentsize;
    for i in 0..shnum {
        let sh = section(i);
        // SHT_SYMTAB, its string table at sh_link.
        if u32_at(sh + 4)? != 2 {
            continue;
        }
        let (off, size, entsize) = (
            u64_at(sh + 0x18)? as usize,
            u64_at(sh + 0x20)? as usize,
            u64_at(sh + 0x38)? as usize,
        );
        let strtab = u64_at(section(u32_at(sh + 0x28)? as usize) + 0x18)? as usize;
        for at in (off..off + size).step_by(entsize.max(24)) {
            let start = strtab + u32_at(at)? as usize;
            let end = start + image.get(start..)?.iter().position(|&b| b == 0)?;
            if &image[start..end] == name.as_bytes() {
                return u64_at(at + 16);
            }
        }
    }
    None
}

/// `gring` read from a VM PE (`hub`: block 0's, (R2X + 1, F)), `words` long
/// (the program spec's VM ABI); a one-word ring is 16 words wide.
pub fn read_ring(
    rt: &mut dyn crate::spatial::device::Device,
    hub: (usize, usize),
    words: usize,
) -> Result<Ring, String> {
    let mut ring = vec![0u32; words];
    let at = crate::sdk::Rect {
        x: hub.0 as i32,
        y: hub.1 as i32,
        w: 1,
        h: 1,
    };
    rt.memcpy_d2h(
        "gring",
        &mut ring,
        at,
        words,
        crate::sdk::Order::RowMajor,
        crate::sdk::DataType::Bits32,
    )
    .map_err(|e| format!("reading the VM's ring (`gring`): {e}"))?;
    let ring = Ring {
        depth: ring[0],
        width: ring.get(1).copied().unwrap_or(OUT_WORDS),
        lanes: ring.get(2).copied().unwrap_or(1),
    };
    if ring.depth == 0 || ring.width == 0 || ring.lanes == 0 || ring.lanes > ring.depth {
        return Err(format!("the VM's `gring` reads {ring:?}"));
    }
    Ok(ring)
}

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
        if let Some((chan, slot, n, _)) = table
            .outputs
            .iter()
            .find(|(_, slot, n, _)| slot + n > OUT_WORDS)
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
    /// The values the step put to homes the host knows (the check VM's, or
    /// a pool home read back from the VM): put on the instance as eta puts
    /// them, so whatever watches a put (the next pass's token) sees it.
    /// Other homes stay as they are (kept).
    pub homes: &'a [(u32, Value)],
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
                // A home: its value when the host knows it, else the VM
                // wrote it in place and the ring stays as it is.
                (None, None) => self
                    .homes
                    .iter()
                    .find(|(chan, _)| *chan == put.channel)
                    .map_or_else(|| Value::F32(Vec::new()), |(_, value)| value.clone()),
            };
        }
        Ok(())
    }

    fn binds(&self, root: &LaunchValue) -> bool {
        root.output.as_deref() == Some(self.program.logits.as_str())
    }

    fn keeps(&self, channel: u32) -> bool {
        self.program.homes.contains(&channel)
            && !self.homes.iter().any(|(chan, _)| *chan == channel)
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

/// Values by channel.
pub type ChannelValues = Vec<(u32, Value)>;

/// The VM's side played by eta's interpreter on a copy of the instance (the
/// engine's check of this contract, `guest = "check"`): the copy
/// holds the homes as the VM would, a step runs the epilogue on it and leaves
/// `out` as the VM would, and the instance itself steps through [`VmStage`].
/// The homes are copied back after each step (the host side keeps reading
/// them: an envelope's tokens) and the ones the host writes are copied in
/// before it.
pub struct CheckVm {
    shadow: eta_exec::InterpInstance,
}

impl CheckVm {
    /// The VM loaded with `inst`'s channels as they stand (the homes
    /// seeded).
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
    ) -> Result<(Vec<u32>, ChannelValues, ChannelValues), String> {
        for &chan in &program.table.per_step {
            copy_ring(
                &self.shadow.channels[chan as usize],
                &inst.channels[chan as usize],
            );
        }
        let inputs = eta_exec::PassInputs { outputs };
        match eta_exec::step_with(
            &mut self.shadow,
            plan,
            &inputs,
            &mut Interpreted,
            Stage::Epilogue,
        ) {
            eta_exec::StepOutcome::Committed => {}
            other => return Err(format!("the check VM's step: {other:?}")),
        }
        let mut out = vec![0u32; OUT_WORDS as usize];
        for &(chan, slot, n, _) in &program.table.outputs {
            // The put waits in the copy's ring; the host takes the real one.
            let ring = &self.shadow.channels[chan as usize];
            let value = ring
                .pop()
                .ok_or_else(|| format!("channel {chan} got no put"))?;
            let words = words_of(&value);
            out[slot as usize..(slot + n) as usize].copy_from_slice(&words[..n as usize]);
        }
        let mut wide = Vec::new();
        for (chan, _, _) in &program.table.wide_outputs {
            let ring = &self.shadow.channels[*chan as usize];
            let value = ring
                .pop()
                .ok_or_else(|| format!("channel {chan} got no put"))?;
            wide.push((*chan, value));
        }
        // What the step put to each home: the copy's last put.
        let puts: BTreeSet<u32> = plan.package.stages[program.epilogue]
            .puts
            .iter()
            .map(|p| p.channel)
            .collect();
        let homes = program
            .homes
            .iter()
            .filter(|chan| puts.contains(chan))
            .filter_map(|&chan| {
                let ring = &self.shadow.channels[chan as usize];
                let tail = ring.tail();
                (tail > 0).then(|| (chan, ring.decode_sequence(tail - 1)))
            })
            .collect();
        Ok((out, wide, homes))
    }

    /// The homes back into `inst` (what a read-back of the VM's pool and wide
    /// rows gives).
    pub fn read_back(&self, program: &Program, inst: &eta_exec::InterpInstance) {
        for &chan in &program.homes {
            copy_ring(
                &inst.channels[chan as usize],
                &self.shadow.channels[chan as usize],
            );
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn ring_index_wraps_by_depth() {
        let ring = super::Ring {
            depth: 16,
            width: 1,
            lanes: 1,
        };
        assert_eq!((ring.index(3, 0), ring.index(17, 0)), (3, 1));
        // Three lanes: lane l's step s at row (3s + l) % 16; 5 steps a launch.
        let ring = super::Ring {
            depth: 16,
            width: 2,
            lanes: 3,
        };
        assert_eq!(
            (ring.index_lane(0, 2, 1), ring.index_lane(5, 1, 0)),
            (2 * 2 + 1, 0)
        );
    }

    #[test]
    fn a_home_whose_value_is_known_is_put_not_kept() {
        // A prompt pass's token channel is a pool home with no token home:
        // kept, the instance never sees its put (the first token read 0).
        use eta_exec::StageRunner;
        let program = super::Program {
            table: super::Table {
                ins: Vec::new(),
                pre_len: 0,
                homes: Vec::new(),
                outputs: Vec::new(),
                wide_outputs: Vec::new(),
                per_step: Vec::new(),
                tok_home: None,
                nreg: 1,
                pool_words: 1,
                feat: 0,
                remat: false,
                bank_regs: Vec::new(),
                bank_pool: Vec::new(),
            },
            epilogue: 0,
            homes: [0, 3].into(),
            logits: "logits".into(),
        };
        let homes = [(0, super::Value::U32(vec![7]))];
        let stage = super::VmStage {
            program: &program,
            out: &[],
            wide: &[],
            homes: &homes,
        };
        assert!(!stage.keeps(0));
        assert!(stage.keeps(3));
        assert!(!stage.keeps(5));
    }
}
