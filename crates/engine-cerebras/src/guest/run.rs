//! Running lowered eta stages on the fabric.
//!
//! [`OnDevice`] is an `eta_exec::StageRunner`: `eta_exec::step_with` drives
//! one instance's pass and hands each stage here, which packs the stage's
//! roots (and the readout rows the stage reads) into one upload, one row
//! per PE, runs the stage's program and unpacks the one download of what
//! the stage puts.

use std::collections::HashMap;
use std::sync::Arc;

use eta_compiler::codegen::launch::LaunchPackage;
use eta_exec::{BatchRunner, ExecPlan, Lane, StagePlan, StageRunner, Value};
use eta_ir::Dtype;
use eta_ir::op::IntrinsicId;

use super::lower::{self, Batch, Feed, Layout};
use crate::device::{Arg, Buffer, Device, Program};
use crate::readout::{Kept, Seat};
use crate::trace::{Signature, Source};

/// A lowered stage, compiled; `program` is `None` for a stage that puts
/// nothing (there is nothing to run).
pub struct Staged {
    pub program: Option<Arc<Program>>,
    pub layout: Layout,
    pub pes: u32,
}

type StageKey = ([u8; 32], usize, Batch);

/// Compiled stages by (program content, stage, batch shape): a pass that
/// was seen skips the lowering and the compile both.
#[derive(Default)]
pub struct Stages {
    map: HashMap<StageKey, Arc<Staged>>,
    /// Stage executions and lanes they carried, for the tally.
    pub runs: u64,
    pub lanes: u64,
}

impl Stages {
    pub fn get(
        &mut self,
        device: &Device,
        package: &LaunchPackage,
        key: [u8; 32],
        at: usize,
        batch: Batch,
    ) -> Result<Arc<Staged>, String> {
        let slot = (key, at, batch);
        if let Some(hit) = self.map.get(&slot) {
            return Ok(Arc::clone(hit));
        }
        let lowered = lower::lower(package, at, batch).map_err(|why| why.to_string())?;
        let program = if !lowered.layout.runs() {
            None
        } else {
            let (ins, outs) = lowered.layout.buffers();
            let params = ins
                .iter()
                .enumerate()
                .map(|(i, _)| Source::Input { input: i as u32 })
                .collect();
            let names = ins.iter().map(|(name, _)| name.clone()).collect();
            let sig = Signature {
                params,
                results: vec![None; outs.len()],
                names,
                outputs: outs
                    .iter()
                    .map(|(name, words)| {
                        (
                            name.clone(),
                            (dtype::Dtype::U32, lowered.pes, *words as u32),
                        )
                    })
                    .collect(),
                buffers: lowered.buffers.clone(),
                packed_views: Vec::new(),
            };
            Some(
                device
                    .program(&lowered.text, sig)
                    .map_err(|e| format!("stage {at} did not compile: {e}"))?,
            )
        };
        let staged = Arc::new(Staged {
            program,
            layout: lowered.layout,
            pes: lowered.pes,
        });
        self.map.insert(slot, Arc::clone(&staged));
        Ok(staged)
    }
}

/// The batch shape of stage `at` for `lanes` lanes over `cols` PEs each,
/// reading `kept`.
#[must_use]
pub fn batch_of(
    package: &LaunchPackage,
    at: usize,
    lanes: u32,
    cols: u32,
    kept: Option<&Kept>,
) -> Batch {
    let (logits, mtp) = lower::reads(package, at);
    Batch {
        lanes,
        cols,
        logits: kept.filter(|_| logits).map(|k| k.width),
        mtp: kept
            .filter(|_| mtp)
            .and_then(|k| k.mtp.as_ref().map(|(_, w)| *w)),
    }
}

/// The fewest PEs a lane of `package` spreads over so that every stage
/// fits (the intrinsics fed as values), or none when no spread fits.
#[must_use]
pub fn columns_of(package: &LaunchPackage) -> Option<u32> {
    columns_above(package, 0)
}

/// `columns_of` among the spreads over more than `above` PEs.
#[must_use]
pub fn columns_above(package: &LaunchPackage, above: u32) -> Option<u32> {
    let choices = lower::wide_axis(package).map_or_else(|| vec![1], lower::column_choices);
    choices.into_iter().filter(|&c| c > above).find(|&cols| {
        (0..package.stages.len()).all(|at| {
            lower::lower(
                package,
                at,
                Batch {
                    lanes: 1,
                    cols,
                    logits: None,
                    mtp: None,
                },
            )
            .is_ok()
        })
    })
}

/// Why a program's stages could not be readied on the device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unfit {
    /// A stage's code and data overflow the PE (the linker says so): a
    /// wider spread may fit.
    Memory(String),
    /// Anything else: the program has no device form.
    Other(String),
}

/// Compiles every stage of `package` for a batch of `lanes` lanes over
/// `cols` PEs each (reading `kept`), so a pass finds them ready.
pub fn prepare(
    device: &Device,
    stages: &mut Stages,
    package: &LaunchPackage,
    key: [u8; 32],
    cols: u32,
    lanes: u32,
    kept: Option<&Kept>,
) -> Result<(), Unfit> {
    for at in 0..package.stages.len() {
        let batch = batch_of(package, at, lanes, cols, kept);
        if let Err(why) = stages.get(device, package, key, at, batch) {
            return Err(if why.contains("ran out of PE memory") {
                Unfit::Memory(format!("stage {at}: {why}"))
            } else {
                Unfit::Other(format!("stage {at}: {why}"))
            });
        }
    }
    Ok(())
}

/// The most lanes one guest program carries (`PIE_CEREBRAS_GUEST_LANES`,
/// 16 by default: a rectangle's rows, one lane each).
#[must_use]
pub fn guest_lanes() -> usize {
    std::env::var("PIE_CEREBRAS_GUEST_LANES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(16)
}

fn encode(value: &Value, dtype: Dtype, n: usize, out: &mut Vec<u32>) -> Result<(), String> {
    if value.len() != n || value.dtype() != dtype {
        return Err(format!(
            "a root holds {} {:?} lane(s) where the stage reads {n} {dtype:?}",
            value.len(),
            value.dtype()
        ));
    }
    match value {
        Value::F32(v) => out.extend(v.iter().map(|x| x.to_bits())),
        Value::I32(v) => out.extend(v.iter().map(|&x| x as u32)),
        Value::U32(v) => out.extend_from_slice(v),
        Value::Bool(v) => {
            let base = out.len();
            out.resize(base + n.div_ceil(32), 0);
            for (j, &b) in v.iter().enumerate() {
                if b != 0 {
                    out[base + j / 32] |= 1 << (j % 32);
                }
            }
        }
    }
    Ok(())
}

fn decode(words: &[u32], dtype: Dtype, n: usize) -> Value {
    match dtype {
        Dtype::F32 => Value::F32(words[..n].iter().map(|&w| f32::from_bits(w)).collect()),
        Dtype::I32 => Value::I32(words[..n].iter().map(|&w| w as i32).collect()),
        Dtype::Bool => Value::Bool(
            (0..n)
                .map(|j| ((words[j / 32] >> (j % 32)) & 1) as u8)
                .collect(),
        ),
        _ => Value::U32(words[..n].to_vec()),
    }
}

/// PE `x`'s share of a wide value: every row's `[x · block, min(v, (x + 1)
/// · block))`, padded to `block` (a PE past the axis holds padding alone).
fn block_of(value: &Value, v: usize, block: usize, x: usize) -> Value {
    fn cut<T: Copy + Default>(src: &[T], v: usize, block: usize, x: usize) -> Vec<T> {
        let rows = src.len().checked_div(v).unwrap_or(0);
        let mut out = Vec::with_capacity(rows * block);
        for r in 0..rows {
            let from = (x * block).min(v);
            let to = ((x + 1) * block).min(v);
            out.extend_from_slice(&src[r * v + from..r * v + to]);
            out.resize((r + 1) * block, T::default());
        }
        out
    }
    match value {
        Value::F32(a) => Value::F32(cut(a, v, block, x)),
        Value::I32(a) => Value::I32(cut(a, v, block, x)),
        Value::U32(a) => Value::U32(cut(a, v, block, x)),
        Value::Bool(a) => Value::Bool(cut(a, v, block, x)),
    }
}

/// A wide value from its PEs' shares (`block_of`'s inverse): row by row,
/// each PE's held lanes in PE order.
fn stitch(parts: &[Value], v: usize, block: usize, rows: usize) -> Value {
    fn join<T: Copy>(parts: &[&[T]], v: usize, block: usize, rows: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(rows * v);
        for r in 0..rows {
            for (x, part) in parts.iter().enumerate() {
                let held = v.saturating_sub(x * block).min(block);
                out.extend_from_slice(&part[r * block..r * block + held]);
            }
        }
        out
    }
    match parts.first() {
        Some(Value::F32(_)) => Value::F32(join(
            &parts
                .iter()
                .map(|p| match p {
                    Value::F32(a) => a.as_slice(),
                    _ => &[],
                })
                .collect::<Vec<_>>(),
            v,
            block,
            rows,
        )),
        Some(Value::I32(_)) => Value::I32(join(
            &parts
                .iter()
                .map(|p| match p {
                    Value::I32(a) => a.as_slice(),
                    _ => &[],
                })
                .collect::<Vec<_>>(),
            v,
            block,
            rows,
        )),
        Some(Value::U32(_)) => Value::U32(join(
            &parts
                .iter()
                .map(|p| match p {
                    Value::U32(a) => a.as_slice(),
                    _ => &[],
                })
                .collect::<Vec<_>>(),
            v,
            block,
            rows,
        )),
        Some(Value::Bool(_)) => Value::Bool(join(
            &parts
                .iter()
                .map(|p| match p {
                    Value::Bool(a) => a.as_slice(),
                    _ => &[],
                })
                .collect::<Vec<_>>(),
            v,
            block,
            rows,
        )),
        None => Value::F32(Vec::new()),
    }
}

/// One lane's feed for a stage: its values (by package-global id; only the
/// stage's roots are read) and where its readout rows sit.
pub struct Feeding<'a> {
    pub vals: &'a [Value],
    pub seat: Option<Seat>,
}

/// Packs `lanes`' roots (padded to the staged batch by repeating lane 0),
/// runs the stage and unpacks every lane's puts. PE `(x, y)` holds lane
/// `y`'s local roots whole and block `x` of its wide ones.
pub fn run_stage(
    device: &Device,
    staged: &Staged,
    kept: Option<&Kept>,
    lanes: &[Feeding<'_>],
) -> Result<Vec<Vec<(u32, Value)>>, String> {
    let Some(program) = &staged.program else {
        return Ok(vec![Vec::new(); lanes.len()]);
    };
    let layout = &staged.layout;
    let cols = layout.cols.max(1) as usize;
    let block = layout.block.max(1);
    let batch = staged.pes as usize / cols;
    // Every input buffer's words, PE after PE: a direct slot's own, the
    // bit-packed bools' `pack`.
    let mut direct: Vec<Vec<u32>> = vec![Vec::new(); layout.inputs.len()];
    let mut pack_words: Vec<u32> = Vec::with_capacity(batch * cols * layout.in_words);
    for b in 0..batch {
        let lane = &lanes[if b < lanes.len() { b } else { 0 }];
        for x in 0..cols {
            for (k, slot) in layout.inputs.iter().enumerate() {
                let words = if slot.direct {
                    &mut direct[k]
                } else {
                    &mut pack_words
                };
                let before = words.len();
                match slot.feed {
                    Feed::Value(dtype, n) => {
                        let value = &lane.vals[slot.value as usize];
                        if slot.wide {
                            let (axis, block) = (slot.axis, slot.block);
                            if value.len() != n || axis == 0 || !n.is_multiple_of(axis) {
                                return Err(format!(
                                    "a wide root holds {} lane(s) where the stage reads {n} over a {axis}-wide axis",
                                    value.len()
                                ));
                            }
                            let share = block_of(value, axis, block, x);
                            encode(&share, dtype, (n / axis) * block, words)?;
                        } else {
                            encode(value, dtype, n, words)?;
                        }
                    }
                    Feed::Rows {
                        intrinsic,
                        rows,
                        width,
                    } => {
                        let seat = lane.seat.ok_or_else(|| {
                            format!(
                                "this pass reads the `{}` intrinsic and its lane offered no \
                                 readout rows",
                                intrinsic.name()
                            )
                        })?;
                        if rows > seat.count {
                            return Err(
                                "logits intrinsic row range exceeds the forward's readout rows"
                                    .to_owned(),
                            );
                        }
                        let kept =
                            kept.ok_or("this stage reads a readout and the fire kept none")?;
                        let plane: &Buffer = if intrinsic == IntrinsicId::Logits {
                            &kept.logits
                        } else {
                            &kept
                                .mtp
                                .as_ref()
                                .ok_or("this stage reads the draft head and the fire kept none")?
                                .0
                        };
                        let width = width as usize;
                        let (block, held) = if slot.wide {
                            (block, width.saturating_sub(x * block).min(block))
                        } else {
                            (width, width)
                        };
                        for r in 0..rows as usize {
                            let from = (seat.first as usize + r) * width + (x * block).min(width);
                            let cut = plane.words().get(from..from + held).ok_or(
                                "the readout rows the stage reads are past the kept plane",
                            )?;
                            words.extend_from_slice(cut);
                            words.resize(words.len() + block - held, 0);
                        }
                    }
                }
                if words.len() - before != slot.words {
                    return Err(format!(
                        "input slot {k} packed {} words for a PE where its layout holds {}",
                        words.len() - before,
                        slot.words
                    ));
                }
            }
        }
    }
    let pes = (batch * cols) as u32;
    let mut buffers: Vec<Buffer> = Vec::new();
    for (k, slot) in layout.inputs.iter().enumerate() {
        if slot.direct {
            let words = std::mem::take(&mut direct[k]);
            buffers.push(
                Buffer::new(dtype::Dtype::U32, pes, slot.words as u32, words)
                    .ok_or("an input buffer has no storage form")?,
            );
        }
    }
    if layout.in_words > 0 {
        buffers.push(
            Buffer::new(dtype::Dtype::U32, pes, layout.in_words as u32, pack_words)
                .ok_or("the pack has no storage form")?,
        );
    }
    let args: Vec<Arg<'_>> = buffers.iter().map(Arg::Keep).collect();
    // The stage's program stays loaded between runs: a lane runs it every
    // step.
    let outs = device
        .run_with(program, args, true)
        .map_err(|e| e.to_string())?;
    let (_, out_names) = layout.buffers();
    if outs.len() != out_names.len() {
        return Err(format!(
            "the stage returned {} buffer(s) for {}",
            outs.len(),
            out_names.len()
        ));
    }
    for (buffer, (name, words)) in outs.iter().zip(&out_names) {
        if buffer.words().len() != batch * cols * words {
            return Err(format!(
                "the stage returned {} words in `{name}` for {batch} lane(s) of {words} x {cols}",
                buffer.words().len()
            ));
        }
        if name.starts_with("dbg") {
            for pe in 0..batch * cols {
                let row = &buffer.words()[pe * words..(pe + 1) * words];
                eprintln!(
                    "{name} pe {pe}: {:?} (as i32 {:?}, as f32 {:?})",
                    row,
                    row.iter().map(|&w| w as i32).collect::<Vec<_>>(),
                    row.iter().map(|&w| f32::from_bits(w)).collect::<Vec<_>>()
                );
            }
        }
    }
    // Output slot k's buffer and the PE's words in it.
    let buffer_of = |k: usize| -> (&[u32], usize) {
        let name = layout.output_name(k);
        let at = out_names
            .iter()
            .position(|(n, _)| *n == name)
            .expect("an output buffer");
        (outs[at].words(), out_names[at].1)
    };
    Ok((0..lanes.len())
        .map(|b| {
            layout
                .outputs
                .iter()
                .enumerate()
                .map(|(k, slot)| {
                    let Feed::Value(dtype, n) = slot.feed else {
                        unreachable!("outputs are values");
                    };
                    // A local output from the row's first PE, a wide one
                    // stitched from the row's PEs' shares.
                    let (all, stride) = buffer_of(k);
                    let pe_words = |x: usize| {
                        let from = (b * cols + x) * stride + slot.at;
                        &all[from..from + slot.words]
                    };
                    let value = if slot.wide && slot.axis > 0 {
                        let (axis, block) = (slot.axis, slot.block);
                        let rows = n / axis;
                        let parts: Vec<Value> = (0..cols)
                            .map(|x| decode(pe_words(x), dtype, rows * block))
                            .collect();
                        stitch(&parts, axis, block, rows)
                    } else {
                        decode(pe_words(0), dtype, n)
                    };
                    (slot.value, value)
                })
                .collect()
        })
        .collect())
}

/// Several instances' stages on the fabric in lockstep, as `step_many`'s
/// runner: one program per stage, lane `l` of the batch on PE row `l`.
pub struct OnDeviceMany<'a> {
    device: &'a Device,
    stages: &'a mut Stages,
    key: [u8; 32],
    /// PEs a lane spreads over.
    cols: u32,
    kept: Option<&'a Kept>,
    /// Each lane's seat in the readout.
    seats: Vec<Option<Seat>>,
    ran: usize,
}

impl<'a> OnDeviceMany<'a> {
    pub fn new(
        device: &'a Device,
        stages: &'a mut Stages,
        key: [u8; 32],
        cols: u32,
        kept: Option<&'a Kept>,
        seats: Vec<Option<Seat>>,
    ) -> OnDeviceMany<'a> {
        OnDeviceMany {
            device,
            stages,
            key,
            cols,
            kept,
            seats,
            ran: 0,
        }
    }

    /// Stages this runner executed.
    #[must_use]
    pub fn ran(&self) -> usize {
        self.ran
    }
}

impl BatchRunner for OnDeviceMany<'_> {
    fn run(
        &mut self,
        plan: &ExecPlan,
        sp: &StagePlan,
        lanes: &mut [Lane<'_>],
    ) -> eta_exec::Result<()> {
        let at = sp.stage_index;
        let fail = |message: String| eta_exec::Error { message };
        let batch = batch_of(&plan.package, at, lanes.len() as u32, self.cols, self.kept);
        let staged = self
            .stages
            .get(self.device, &plan.package, self.key, at, batch)
            .map_err(fail)?;
        let feedings: Vec<Feeding<'_>> = lanes
            .iter()
            .map(|lane| Feeding {
                vals: lane.vals,
                seat: self.seats.get(lane.index).copied().flatten(),
            })
            .collect();
        let outs = run_stage(self.device, &staged, self.kept, &feedings).map_err(fail)?;
        self.stages.runs += 1;
        self.stages.lanes += lanes.len() as u64;
        for (lane, puts) in lanes.iter_mut().zip(outs) {
            for (id, value) in puts {
                lane.vals[id as usize] = value;
            }
        }
        self.ran += 1;
        Ok(())
    }

    fn binds(&self, lane: usize, id: Option<IntrinsicId>) -> bool {
        device_binds(id, self.kept, self.seats.get(lane).copied().flatten())
    }
}

/// Whether the device binds `id` for a lane seated at `seat` in `kept`.
#[must_use]
pub fn device_binds(id: Option<IntrinsicId>, kept: Option<&Kept>, seat: Option<Seat>) -> bool {
    match id {
        Some(IntrinsicId::Logits) => kept.is_some_and(|k| k.f32) && seat.is_some(),
        Some(IntrinsicId::MtpLogits | IntrinsicId::MtpDrafts) => {
            kept.is_some_and(|k| k.f32 && k.mtp.is_some()) && seat.is_some()
        }
        _ => false,
    }
}

/// One instance's stages on the fabric, as `step_with`'s runner.
pub struct OnDevice<'a> {
    device: &'a Device,
    stages: &'a mut Stages,
    key: [u8; 32],
    /// PEs the lane spreads over.
    cols: u32,
    kept: Option<&'a Kept>,
    seat: Option<Seat>,
    ran: usize,
}

impl<'a> OnDevice<'a> {
    pub fn new(
        device: &'a Device,
        stages: &'a mut Stages,
        key: [u8; 32],
        cols: u32,
        kept: Option<&'a Kept>,
        seat: Option<Seat>,
    ) -> OnDevice<'a> {
        OnDevice {
            device,
            stages,
            key,
            cols,
            kept,
            seat,
            ran: 0,
        }
    }

    /// Stages this runner executed.
    #[must_use]
    pub fn ran(&self) -> usize {
        self.ran
    }
}

impl StageRunner for OnDevice<'_> {
    fn run(
        &mut self,
        plan: &ExecPlan,
        sp: &StagePlan,
        _roots: &[u32],
        _wanted: &[u32],
        vals: &mut [Value],
    ) -> eta_exec::Result<()> {
        let at = sp.stage_index;
        let fail = |message: String| eta_exec::Error { message };
        let batch = batch_of(&plan.package, at, 1, self.cols, self.seat.and(self.kept));
        let staged = self
            .stages
            .get(self.device, &plan.package, self.key, at, batch)
            .map_err(fail)?;
        let outs = run_stage(
            self.device,
            &staged,
            self.kept,
            &[Feeding {
                vals,
                seat: self.seat,
            }],
        )
        .map_err(fail)?;
        self.stages.runs += 1;
        self.stages.lanes += 1;
        for (id, value) in outs.into_iter().next().unwrap_or_default() {
            vals[id as usize] = value;
        }
        self.ran += 1;
        Ok(())
    }

    fn binds(&self, id: Option<IntrinsicId>) -> bool {
        device_binds(id, self.kept, self.seat)
    }
}
