//! A stream reading's host side: what the engine does around a step the wafer runs, read off the
//! program's `host.json`. The host forward makes each entry's values from the step's inputs and
//! the trace's exit ops from the rows read back; [`Denoise`] runs the schedule between steps and
//! [`Reading::statics`] makes the static arrays from the checkpoint. A step runs on a [`Fabric`]:
//! the SDK runtime, or [`HostFabric`] (engine-cpu's rows, no wafer). Nothing here names a model's
//! op: the values are the trace's, by id.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use engine_cpu::Forward;
use engine_cpu::forward::{Captured, StreamLane, StreamStep, Values};
use kernels_cerebras::compile::emit::state::{Pe, StateRoot};
use kernels_cerebras::compile::stream::host::{Span, Word, apply, apply_spans};
use poem_ir::{Def, InputAxis, Operands, RuntimeInput, Trace, ValueId};
use serde_json::Value;

use crate::sdk::{DataType, Order, Rect, Runtime};

type R<T> = Result<T, String>;

/// A row class of the reading: one lane of a step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Class {
    pub name: String,
    pub word: u64,
    pub rows: u32,
}

/// A memcpy rectangle the host writes: its words from `file`, from `statics` (spans over values
/// the checkpoint makes), or, for an entry, gathered from the step's values each step.
#[derive(Clone, Debug)]
pub struct Array {
    pub sym: String,
    pub rect: Rect,
    pub len: usize,
    pub u16: bool,
    pub file: Option<PathBuf>,
    pub entry: Option<Vec<Word>>,
    pub statics: Option<Vec<Span>>,
    /// The weight group whose launch it is written before (a grid program in groups: a run of
    /// passes' tiles and their range), every step.
    pub group: Option<usize>,
    /// Each row's PEs take the same words (its first PE's), written once a row and broadcast along
    /// it.
    pub rowbcast: bool,
    /// The program never writes it: a write after the load's first sends each PE's words up to its
    /// last changed one alone.
    pub kept: bool,
}

impl Array {
    fn file(&self) -> R<&Path> {
        self.file.as_deref().ok_or_else(|| {
            format!(
                "`{}` at ({}, {}): no words on file (a program compiled with no capture)",
                self.sym, self.rect.x, self.rect.y
            )
        })
    }
}

/// Where a step's arrays are written and run: the SDK runtime, or [`HostFabric`].
pub const MEMCPY_WORDS: usize = 1 << 24;

pub trait Fabric {
    /// Writes array `a`'s words (each an f32's bits, or a 16-bit word).
    fn write(&mut self, a: &Array, words: &[u32]) -> R<()>;
    /// Runs `step` on the arrays written: the rows back (each [`Back`]'s words) and the step's
    /// cycles where the program stamps them.
    fn run(&mut self, reading: &Reading, step: &StreamStep<'_>) -> R<(Vec<Vec<f32>>, Option<u64>)>;
    /// Runs a batch's `steps` (at most [`Reading::batch`]) on the arrays written: the rows back
    /// laid out as the batch's.
    fn run_batch(
        &mut self,
        reading: &Reading,
        steps: &[StreamStep<'_>],
    ) -> R<(Vec<Vec<f32>>, Option<u64>)> {
        self.run(reading, steps.first().ok_or("a batch of no steps")?)
    }
    /// Opens the program linked in `link` in place of the one open. Returns whether it was loaded
    /// afresh (its arrays to be written again).
    fn switch(&mut self, link: &Path) -> R<bool> {
        let _ = link;
        Ok(false)
    }
    /// Launches `run` on the arrays written and waits, reading no rows back: its cycles where the
    /// program stamps them.
    fn launch(&mut self, reading: &Reading) -> R<Option<u64>> {
        let _ = reading;
        Ok(None)
    }
    /// A decode program's argmax a row after a launch ([`Reading::argmax`]): each row's largest
    /// logit's column, the first on a tie; `None` where the fabric reads none.
    fn argmax(&mut self, reading: &Reading) -> R<Option<Vec<u32>>> {
        let _ = reading;
        Ok(None)
    }
    /// `rowp` as the device holds it, where followed (`None`: written every step).
    fn rows_held(&self) -> Option<&[u32]> {
        None
    }
    /// `rowp` written (`Some`: followed, the program advances it).
    fn rows_written(&mut self, rows: Option<Vec<u32>>) {
        let _ = rows;
    }
}

/// `rowp` (lane slot, state slot, position a row) after a launch of a program that advances it.
#[must_use]
pub fn rows_advanced(rows: &[u32]) -> Vec<u32> {
    let mut v = rows.to_vec();
    for p in v.iter_mut().skip(2).step_by(3) {
        *p = p.wrapping_add(1);
    }
    v
}

/// The SDK runtime of one program at a time, each created on the thread that holds this (the
/// SDK wants every runtime of a process made on one thread).
pub struct Programs {
    pub sdk_lib: PathBuf,
    pub platform: crate::sdk::Platform,
    pub rt: Option<Runtime>,
}

impl Fabric for Programs {
    fn write(&mut self, a: &Array, words: &[u32]) -> R<()> {
        Fabric::write(self.rt.as_mut().ok_or("no program is open")?, a, words)
    }

    fn run(&mut self, reading: &Reading, step: &StreamStep<'_>) -> R<(Vec<Vec<f32>>, Option<u64>)> {
        Fabric::run(self.rt.as_mut().ok_or("no program is open")?, reading, step)
    }

    fn launch(&mut self, reading: &Reading) -> R<Option<u64>> {
        Fabric::launch(self.rt.as_mut().ok_or("no program is open")?, reading)
    }

    fn argmax(&mut self, reading: &Reading) -> R<Option<Vec<u32>>> {
        Fabric::argmax(self.rt.as_mut().ok_or("no program is open")?, reading)
    }

    fn rows_held(&self) -> Option<&[u32]> {
        self.rt.as_ref().and_then(|rt| Fabric::rows_held(rt))
    }

    fn rows_written(&mut self, rows: Option<Vec<u32>>) {
        if let Some(rt) = self.rt.as_mut() {
            Fabric::rows_written(rt, rows);
        }
    }

    fn switch(&mut self, link: &Path) -> R<bool> {
        if let Some(mut old) = self.rt.take() {
            old.stop();
        }
        let sdk = crate::sdk::Sdk::open_at(&self.sdk_lib).map_err(|e| e.to_string())?;
        let mut rt =
            Runtime::new(sdk, link, self.platform.clone(), "WARNING").map_err(|e| e.to_string())?;
        let t0 = std::time::Instant::now();
        rt.load();
        rt.run();
        tracing::info!(
            s = t0.elapsed().as_secs_f64(),
            link = %link.display(),
            "spatial: a stream program loaded"
        );
        self.rt = Some(rt);
        Ok(true)
    }
}

/// A root's rows read back: each of its rows (the step's row `rows[i]`) the `parts`
/// `(value, from, len)` side by side.
#[derive(Clone, Debug)]
pub struct Back {
    pub sym: String,
    pub x: i32,
    pub y: i32,
    /// The rectangle of PEs read back alike (1 x 1: one PE), its rows in row-major order.
    pub w: i32,
    pub h: i32,
    /// Words a PE.
    pub len: usize,
    pub rows: Vec<usize>,
    pub parts: Vec<(u32, u32, u32)>,
}

/// A stream program's host side, as `host.json` states it.
#[derive(Clone, Debug)]
pub struct Reading {
    pub dir: PathBuf,
    pub model: String,
    pub reading: String,
    pub classes: Vec<Class>,
    pub h2d: Vec<Array>,
    pub d2h: Vec<Back>,
    /// A decode program's argmax a row: each root's two words a row, its largest logit and that
    /// logit's column.
    pub argmax: Vec<Back>,
    /// The values read back whole: (value, rows, width).
    pub outs: Vec<(u32, usize, usize)>,
    /// The capture the program was compiled from, if any: each value read back, its file there.
    pub refs: Vec<(u32, PathBuf)>,
    pub w: u32,
    pub h: u32,
    /// A row attends the rows before it alone (a prefill's mask).
    pub causal: bool,
    /// No program runs it: the host's forward makes its outputs whole ([`Reading::host_only`]).
    pub host_only: bool,
    /// Whether a launch reads its PEs' stamps back (its cycles): a served decode step reads none.
    pub stamped: bool,
    /// Whether a launch is timed to its end: a one-word read of its first PE right after the call.
    pub poll: bool,
    /// Its program advances `rowp` itself (`state.advances`): written only where it differs.
    pub rows_advance: bool,
    /// Its PEs' first column and row in a program of co-resident readings (`stream::coresident`).
    pub origin: (i32, i32),
    /// The id its launch takes in a program of co-resident readings (`None`: a program its own).
    pub rd: Option<u32>,
    /// The clip a grid reading's program is compiled for (its rows are the clip's voxels).
    pub clip: Option<[u32; 3]>,
    /// A grid program's weight groups: a launch a group, each group's arrays written before it;
    /// 0: one launch.
    pub groups: usize,
    /// A decode program's K / V PEs: each told a step's first position, their caches kept across
    /// steps.
    pub pos: Vec<Rect>,
    /// A decode program's paging: `pages` pages of `size` positions a column, `lanes` lane slots;
    /// `rowp` its PEs told a step's rows' (lane slot, position), `ptab` each lane slot's pages.
    pub paging: Option<Paging>,
    pub rowp: Vec<Rect>,
    pub ptab: Vec<Rect>,
    /// The state slots every recurrence's column holds (`None`: no recurrence).
    pub slots: Option<u32>,
    /// Requests a launch (the GEMM regime's batch): the step's rows each request's rows in turn.
    pub batch: usize,
}

/// A decode program's paging ([`Reading::paging`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Paging {
    pub size: u32,
    pub pages: u32,
    pub lanes: u32,
}

/// PEs as rectangles: a column's consecutive PEs one.
fn rects(pes: &[Pe]) -> Vec<Rect> {
    let mut cs: Vec<(i32, i32)> = pes.iter().map(|p| (p.x as i32, p.y as i32)).collect();
    cs.sort_unstable();
    let mut out: Vec<Rect> = Vec::new();
    for (x, y) in cs {
        match out.last_mut() {
            Some(r) if r.x == x && r.y + r.h == y => r.h += 1,
            _ => out.push(Rect { x, y, w: 1, h: 1 }),
        }
    }
    out
}

fn num(v: &Value, k: &str) -> R<u64> {
    v[k].as_u64()
        .ok_or_else(|| format!("host.json: `{k}` is no number"))
}

fn text(v: &Value, k: &str) -> R<String> {
    v[k].as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("host.json: `{k}` is no string"))
}

fn npy_words(path: &Path) -> R<(Vec<usize>, Vec<u32>)> {
    let (shape, v) = kernels_cerebras::compile::npy::read_f32_or_i32(path)?;
    Ok((shape, v.iter().map(|x| x.to_bits()).collect()))
}

impl Reading {
    /// The reading `wafer-compile stream` wrote into `dir` (its `host.json`).
    pub fn open(dir: &Path) -> R<Reading> {
        let path = dir.join("host.json");
        let host: Value = serde_json::from_str(
            &std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?,
        )
        .map_err(|e| format!("{}: {e}", path.display()))?;
        // the program's state (`state`: its regions, slots, pages and PEs); a program that states
        // its K / V PEs as top-level keys is refused
        if let Some(k) = ["pos", "paging", "rowp", "ptab"]
            .iter()
            .find(|k| host.get(**k).is_some())
        {
            return Err(format!(
                "{}: `{k}`: compiled before the state root (host.json `state`); compile it anew",
                path.display()
            ));
        }
        let state: StateRoot = match host.get("state") {
            Some(v) => serde_json::from_value(v.clone())
                .map_err(|e| format!("{}: `state`: {e}", path.display()))?,
            None => StateRoot::default(),
        };
        let list = |k: &str| {
            host[k]
                .as_array()
                .cloned()
                .ok_or_else(|| format!("host.json: no `{k}` (compiled by an older build?)"))
        };
        let classes = list("classes")?
            .iter()
            .map(|c| {
                Ok(Class {
                    name: text(c, "name")?,
                    word: text(c, "word")?
                        .parse()
                        .map_err(|e| format!("a class word: {e}"))?,
                    rows: num(c, "rows")? as u32,
                })
            })
            .collect::<R<Vec<_>>>()?;
        let mut h2d = Vec::new();
        for a in list("h2d")? {
            let entry = match a["entry"].as_str() {
                Some(f) => {
                    let (_, w) = npy_words(&dir.join(f))?;
                    Some(w.as_chunks::<4>().0.to_vec())
                }
                None => None,
            };
            let statics = match a["static"].as_str() {
                Some(f) => {
                    let (_, w) = npy_words(&dir.join(f))?;
                    Some(w.as_chunks::<8>().0.to_vec())
                }
                None => None,
            };
            h2d.push(Array {
                sym: text(&a, "sym")?,
                rect: Rect {
                    x: num(&a, "x")? as i32,
                    y: num(&a, "y")? as i32,
                    w: num(&a, "w")? as i32,
                    h: num(&a, "h")? as i32,
                },
                len: num(&a, "len")? as usize,
                u16: a["u16"].as_bool().unwrap_or(false),
                file: a["file"].as_str().map(|f| dir.join(f)),
                entry,
                statics,
                group: a["group"].as_u64().map(|g| g as usize),
                rowbcast: a["rowbcast"].as_bool().unwrap_or(false),
                kept: a["kept"].as_bool().unwrap_or(false),
            });
        }
        let backs = |key: &str| -> R<Vec<Back>> {
            let mut d2h = Vec::new();
            for o in host[key].as_array().into_iter().flatten() {
                let ints = |k: &str| -> Vec<u64> {
                    o[k].as_array()
                        .map(|a| a.iter().filter_map(Value::as_u64).collect())
                        .unwrap_or_default()
                };
                let parts = o["parts"]
                    .as_array()
                    .map(|ps| {
                        ps.iter()
                            .filter_map(|p| {
                                let p: Vec<u32> = (p.as_array()?.iter())
                                    .map(|x| x.as_u64().map(|x| x as u32))
                                    .collect::<Option<_>>()?;
                                // (an argmax's parts: (from, len), of the logits)
                                match p[..] {
                                    [b, f, n] => Some((b, f, n)),
                                    [f, n] => Some((0, f, n)),
                                    _ => None,
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                d2h.push(Back {
                    sym: text(o, "sym")?,
                    x: num(o, "x")? as i32,
                    y: num(o, "y")? as i32,
                    w: o["w"].as_i64().unwrap_or(1) as i32,
                    h: o["h"].as_i64().unwrap_or(1) as i32,
                    len: num(o, "len")? as usize,
                    rows: ints("rows").into_iter().map(|r| r as usize).collect(),
                    parts,
                });
            }
            Ok(d2h)
        };
        let d2h = backs("d2h")?;
        let argmax = backs("argmax")?;
        let outs = list("outs")?
            .iter()
            .map(|o| {
                Ok((
                    num(o, "value")? as u32,
                    num(o, "rows")? as usize,
                    num(o, "width")? as usize,
                ))
            })
            .collect::<R<Vec<_>>>()?;
        let gold = host["gold"].as_str().map(PathBuf::from);
        let refs = (host["refs"].as_array().into_iter().flatten())
            .filter_map(|r| {
                let file = gold.as_ref()?.join(r["file"].as_str()?);
                Some((r["value"].as_u64()? as u32, file))
            })
            .collect();
        Ok(Reading {
            refs,
            dir: dir.to_path_buf(),
            model: text(&host, "model")?,
            reading: text(&host, "reading")?,
            classes,
            h2d,
            d2h,
            argmax,
            outs,
            w: num(&host, "w")? as u32,
            h: num(&host, "h")? as u32,
            causal: host["causal"].as_bool().unwrap_or(false),
            host_only: false,
            stamped: true,
            poll: false,
            rows_advance: state.advances,
            origin: (
                host["origin"][0].as_i64().unwrap_or(0) as i32,
                host["origin"][1].as_i64().unwrap_or(0) as i32,
            ),
            rd: host["rd"].as_u64().map(|r| r as u32),
            clip: host["clip"].as_array().and_then(|c| {
                let v: Vec<u32> = c
                    .iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as u32)
                    .collect();
                <[u32; 3]>::try_from(v).ok()
            }),
            groups: host["groups"].as_u64().unwrap_or(0) as usize,
            batch: host["batch"].as_u64().unwrap_or(1).max(1) as usize,
            pos: rects(&state.pes(|r| &r.pos)),
            paging: state.paging().map(|(g, lanes)| Paging {
                size: g.size,
                pages: g.pages,
                lanes,
            }),
            rowp: rects(&state.pes(|r| &r.rows)),
            ptab: rects(&state.pes(|r| &r.tables)),
            slots: state
                .regions
                .iter()
                .filter(|r| r.kind != "kv")
                .map(|r| r.slots)
                .min(),
        })
    }

    /// A reading of `model` no program runs (the host fabric's fallback): `classes` its lanes, no
    /// arrays; [`Reading::finish`] makes its outputs from the step's inputs alone.
    #[must_use]
    pub fn host_only(
        model: &str,
        reading: &str,
        classes: &[poem_compiler::spatial::layer::stream::RowClass],
    ) -> Reading {
        Reading {
            dir: PathBuf::new(),
            model: model.to_string(),
            reading: reading.to_string(),
            classes: classes
                .iter()
                .map(|c| Class {
                    name: c.name.clone(),
                    word: c.word,
                    rows: c.rows,
                })
                .collect(),
            h2d: Vec::new(),
            d2h: Vec::new(),
            argmax: Vec::new(),
            outs: Vec::new(),
            refs: Vec::new(),
            w: 0,
            h: 0,
            causal: false,
            host_only: true,
            stamped: true,
            poll: false,
            rows_advance: false,
            origin: (0, 0),
            rd: None,
            clip: None,
            groups: 0,
            pos: Vec::new(),
            paging: None,
            rowp: Vec::new(),
            ptab: Vec::new(),
            slots: None,
            batch: 1,
        }
    }

    /// The reading's trace: the catalog model the program was compiled from.
    pub fn trace(&self) -> R<Trace> {
        Ok(poem_compiler::catalog::repository()
            .parse(&self.model)
            .ok_or_else(|| format!("no catalog model {}", self.model))?
            .trace(poem_ir::Platform::Cerebras))
    }

    /// A step of the reading: a lane a class, a voxel lane on the grid `grid`, and the guest inputs
    /// by name.
    pub fn step<'a>(
        &self,
        trace: &Trace,
        grid: Option<[u32; 3]>,
        inputs: &'a [(String, Vec<f32>)],
    ) -> StreamStep<'a> {
        let voxel: Vec<ValueId> = (trace.values.iter().enumerate())
            .filter(|(_, d)| {
                matches!(
                    d.def,
                    Def::Input(RuntimeInput::Guest {
                        axis: InputAxis::Voxel,
                        ..
                    })
                )
            })
            .map(|(i, _)| ValueId(i as u32))
            .collect();
        let reads_voxels = |word: u64| {
            trace.nodes.iter().any(|n| {
                let mut i = Vec::new();
                n.op.inputs(&mut i);
                n.guard.holds(word) && i.iter().any(|v| voxel.contains(v))
            })
        };
        StreamStep {
            lanes: self
                .classes
                .iter()
                .map(|c| StreamLane {
                    word: c.word,
                    rows: c.rows as usize,
                    clip: grid.filter(|_| reads_voxels(c.word)),
                })
                .collect(),
            inputs: inputs
                .iter()
                .map(|(n, v)| (n.as_str(), v.clone()))
                .collect(),
            tokens: Vec::new(),
        }
    }

    /// A request's rows (every class's).
    #[must_use]
    pub fn request_rows(&self) -> usize {
        self.classes.iter().map(|c| c.rows as usize).sum()
    }

    /// [`Reading::entries`] of a batch: `steps` a request's each (fewer than [`Reading::batch`]:
    /// the last request's values stand for the rest), every value's rows the requests' in turn.
    pub fn entries_batch(
        &self,
        host: &mut Forward,
        steps: &[StreamStep<'_>],
    ) -> R<Vec<Option<Vec<f32>>>> {
        let wanted = self.entry_values();
        let mut held: BTreeMap<u32, (usize, usize, Vec<f32>)> = BTreeMap::new();
        let n = steps.len();
        for (k, step) in steps.iter().enumerate() {
            let copies = if k + 1 == n { self.batch.max(n) - k } else { 1 };
            for c in host.stream_values(step, &[], &wanted)? {
                let v = match c.values {
                    Values::F32(v) => v,
                    Values::I32(v) => v.into_iter().map(|x| x as f32).collect(),
                };
                let e = held.entry(c.value.0).or_insert((0, c.width, Vec::new()));
                for _ in 0..copies {
                    e.0 += c.rows;
                    e.2.extend_from_slice(&v);
                }
            }
        }
        let held: BTreeMap<u32, (usize, usize, Rc<Vec<f32>>)> = held
            .into_iter()
            .map(|(k, (r, w, v))| (k, (r, w, Rc::new(v))))
            .collect();
        self.h2d
            .iter()
            .map(|a| {
                let Some(g) = &a.entry else {
                    return Ok(None);
                };
                if a.u16 {
                    return Err(format!("entry `{}`: a 16-bit entry", a.sym));
                }
                let get = |v: u32| {
                    held.get(&v)
                        .cloned()
                        .ok_or_else(|| format!("value %{v}: not made"))
                };
                apply(g, &get).map(Some)
            })
            .collect()
    }

    /// Request `b`'s rows of a batched launch's values read back.
    #[must_use]
    pub fn request_of(&self, back: &[Captured], b: usize) -> Vec<Captured> {
        back.iter()
            .map(|c| {
                let rows = c.rows / self.batch.max(1);
                let Values::F32(v) = &c.values else {
                    return c.clone();
                };
                Captured {
                    rows,
                    values: Values::F32(v[b * rows * c.width..(b + 1) * rows * c.width].to_vec()),
                    ..c.clone()
                }
            })
            .collect()
    }

    /// The values the entries gather from: what the host's forward makes of a step's inputs.
    #[must_use]
    pub fn entry_values(&self) -> Vec<ValueId> {
        let mut wanted: Vec<ValueId> = Vec::new();
        for a in &self.h2d {
            for w in a.entry.iter().flatten() {
                let v = ValueId(w[0]);
                if w[0] != kernels_cerebras::compile::stream::host::NONE && !wanted.contains(&v) {
                    wanted.push(v);
                }
            }
        }
        wanted
    }

    /// Each array's words for `step`, computed on the weightless host forward `host`: an entry's
    /// from the step's values, `None` for an array whose words are the file's.
    pub fn entries(&self, host: &mut Forward, step: &StreamStep<'_>) -> R<Vec<Option<Vec<f32>>>> {
        self.entries_given(host, step, &[])
    }

    /// [`Reading::entries`] with `given` held (a decode step's positions: its tokens' own).
    pub fn entries_given(
        &self,
        host: &mut Forward,
        step: &StreamStep<'_>,
        given: &[Captured],
    ) -> R<Vec<Option<Vec<f32>>>> {
        let wanted = self.entry_values();
        let held: BTreeMap<u32, (usize, usize, Rc<Vec<f32>>)> = host
            .stream_values(step, given, &wanted)?
            .into_iter()
            .map(|c| {
                let v = match c.values {
                    Values::F32(v) => v,
                    Values::I32(v) => v.into_iter().map(|x| x as f32).collect(),
                };
                (c.value.0, (c.rows, c.width, Rc::new(v)))
            })
            .collect();
        self.h2d
            .iter()
            .map(|a| {
                let Some(g) = &a.entry else {
                    return Ok(None);
                };
                if a.u16 {
                    return Err(format!("entry `{}`: a 16-bit entry", a.sym));
                }
                let get = |v: u32| {
                    held.get(&v)
                        .cloned()
                        .ok_or_else(|| format!("value %{v}: not made"))
                };
                apply(g, &get).map(Some)
            })
            .collect()
    }

    /// The values the host reads back, whole (rows the program returns none of are NaN).
    pub fn rows_back(&self, got: &[Vec<f32>]) -> R<Vec<Captured>> {
        let mut vals: BTreeMap<u32, (usize, usize, Vec<f32>)> = self
            .outs
            .iter()
            .map(|&(v, rows, width)| (v, (rows, width, vec![f32::NAN; rows * width])))
            .collect();
        for (b, buf) in self.d2h.iter().zip(got) {
            let width: usize = b.parts.iter().map(|p| p.2 as usize).sum();
            for (i, &tr) in b.rows.iter().enumerate() {
                let mut off = 0;
                for &(base, from, n) in &b.parts {
                    let (_, w, v) = vals
                        .get_mut(&base)
                        .ok_or_else(|| format!("value %{base} is read back but not stated"))?;
                    let (w, from, n) = (*w, from as usize, n as usize);
                    v[tr * w + from..tr * w + from + n]
                        .copy_from_slice(&buf[i * width + off..i * width + off + n]);
                    off += n;
                }
            }
        }
        Ok(vals
            .into_iter()
            .map(|(v, (rows, width, values))| Captured {
                node: usize::MAX,
                input: false,
                value: ValueId(v),
                rows,
                width,
                values: Values::F32(values),
            })
            .collect())
    }

    /// The trace's outputs `names` from the rows read back (`given`): the exit's host ops.
    pub fn finish(
        &self,
        host: &mut Forward,
        trace: &Trace,
        step: &StreamStep<'_>,
        given: &[Captured],
        names: &[&str],
    ) -> R<Vec<Vec<f32>>> {
        let wanted: Vec<ValueId> = names
            .iter()
            .map(|n| {
                (trace.outputs.iter())
                    .find(|o| o.name == *n)
                    .map(|o| o.value)
                    .ok_or_else(|| format!("the trace has no output `{n}`"))
            })
            .collect::<R<_>>()?;
        self.finish_values(host, step, given, &wanted)
    }

    /// The values `wanted` (trace outputs) from the rows read back (`given`): the exit's host ops.
    pub fn finish_values(
        &self,
        host: &mut Forward,
        step: &StreamStep<'_>,
        given: &[Captured],
        wanted: &[ValueId],
    ) -> R<Vec<Vec<f32>>> {
        Ok(host
            .stream_values(step, given, wanted)?
            .into_iter()
            .map(|c| match c.values {
                Values::F32(v) => v,
                Values::I32(v) => v.into_iter().map(|x| x as f32).collect(),
            })
            .collect())
    }

    /// The value ids the host reads back.
    #[must_use]
    pub fn out_values(&self) -> Vec<ValueId> {
        self.outs.iter().map(|o| ValueId(o.0)).collect()
    }

    /// Every array no step's inputs make, from its file: written once on a loaded program, before
    /// its first [`Reading::launch`].
    pub fn write_static(&self, rt: &mut dyn Fabric) -> R<()> {
        for a in self
            .h2d
            .iter()
            .filter(|a| a.entry.is_none() && a.group.is_none())
        {
            rt.write(a, &npy_words(a.file()?)?.1)?;
        }
        Ok(())
    }

    /// The words of every array no step's inputs make, from the checkpoint: each array's spans over
    /// the values `weights` makes; an array with no spans its file's. `None` for an entry.
    pub fn statics(
        &self,
        weights: &mut Forward,
        step: &StreamStep<'_>,
    ) -> R<Vec<Option<Vec<u32>>>> {
        let mut wanted: Vec<ValueId> = Vec::new();
        for a in self.h2d.iter().filter(|a| a.entry.is_none()) {
            for s in a.statics.iter().flatten() {
                let v = ValueId(s[0]);
                let constant = s[0] == kernels_cerebras::compile::stream::host::NONE
                    || s[0] == kernels_cerebras::compile::stream::host::UNIT
                    || s[0] == kernels_cerebras::compile::stream::host::LIT;
                if !constant && !wanted.contains(&v) {
                    wanted.push(v);
                }
            }
        }
        let held: BTreeMap<u32, (usize, usize, Rc<Vec<f32>>)> = weights
            .stream_values(step, &[], &wanted)?
            .into_iter()
            .map(|c| {
                let v = match c.values {
                    Values::F32(v) => v,
                    Values::I32(v) => v.into_iter().map(|x| x as f32).collect(),
                };
                (c.value.0, (c.rows, c.width, Rc::new(v)))
            })
            .collect();
        let get = |v: u32| {
            held.get(&v)
                .cloned()
                .ok_or_else(|| format!("value %{v}: not made"))
        };
        self.h2d
            .iter()
            .map(|a| {
                if a.entry.is_some() {
                    return Ok(None);
                }
                let words = match &a.statics {
                    Some(s) => apply_spans(s, &get)?,
                    None => npy_words(a.file()?)?.1,
                };
                if words.len() != a.len * (a.rect.w * a.rect.h) as usize {
                    return Err(format!(
                        "`{}` at ({}, {}): {} words for {} PEs of {}",
                        a.sym,
                        a.rect.x,
                        a.rect.y,
                        words.len(),
                        a.rect.w * a.rect.h,
                        a.len
                    ));
                }
                Ok(Some(words))
            })
            .collect()
    }

    /// The static arrays' words handed to `write` a run of rows of PEs at a time (at most `chunk`
    /// words, at least a row), each value made only while a span reads it: bounded memory. Arrays
    /// of a weight group are not written here.
    pub fn statics_streamed(
        &self,
        weights: &mut Forward,
        step: &StreamStep<'_>,
        chunk: usize,
        write: &mut dyn FnMut(Array, Vec<u32>) -> R<()>,
    ) -> R<()> {
        let weights = std::cell::RefCell::new(weights);
        // values made a batch at a time (one pass of the weights' forward, up to PREFETCH of them),
        // each handed out once
        const PREFETCH: usize = 8;
        let made = |vs: &[u32]| -> R<Vec<(u32, kernels_cerebras::compile::stream::host::Rows)>> {
            let want: Vec<ValueId> = vs.iter().map(|&v| ValueId(v)).collect();
            Ok(weights
                .borrow_mut()
                .stream_values(step, &[], &want)?
                .into_iter()
                .map(|c| {
                    let vals = match c.values {
                        Values::F32(v) => v,
                        Values::I32(v) => v.into_iter().map(|x| x as f32).collect(),
                    };
                    (c.value.0, (c.rows, c.width, Rc::new(vals)))
                })
                .collect())
        };
        for a in &self.h2d {
            if a.entry.is_some() || a.group.is_some() {
                continue;
            }
            let row = a.len * a.rect.w as usize;
            let Some(spans) = &a.statics else {
                let words = npy_words(a.file()?)?.1;
                write(a.clone(), words)?;
                continue;
            };
            let rows = (chunk / row.max(1)).max(1);
            let mut y = 0i32;
            let mut words = 0usize;
            // the values the spans read, in the order they are first read
            let mut order: Vec<u32> = Vec::new();
            let mut seen = std::collections::BTreeSet::new();
            for sp in spans {
                let v = sp[0];
                let constant = v == kernels_cerebras::compile::stream::host::NONE
                    || v == kernels_cerebras::compile::stream::host::UNIT
                    || v == kernels_cerebras::compile::stream::host::LIT;
                if !constant && seen.insert(v) {
                    order.push(v);
                }
            }
            let ahead = std::cell::RefCell::new((
                0usize,
                BTreeMap::<u32, kernels_cerebras::compile::stream::host::Rows>::new(),
            ));
            let get = |v: u32| -> R<kernels_cerebras::compile::stream::host::Rows> {
                let mut a = ahead.borrow_mut();
                if let Some(r) = a.1.remove(&v) {
                    return Ok(r);
                }
                let at = order.iter().position(|&x| x == v).unwrap_or(a.0).max(a.0);
                let batch: Vec<u32> = std::iter::once(v)
                    .chain(
                        order[at..]
                            .iter()
                            .copied()
                            .filter(|&x| x != v)
                            .take(PREFETCH - 1),
                    )
                    .collect();
                a.0 = (at + PREFETCH).min(order.len());
                let mut got = None;
                for (k, r) in made(&batch)? {
                    if k == v {
                        got = Some(r);
                    } else {
                        a.1.insert(k, r);
                    }
                }
                got.ok_or_else(|| format!("value %{v}: not made"))
            };
            kernels_cerebras::compile::stream::host::apply_spans_into(
                spans,
                &get,
                rows * row,
                &mut |w| {
                    if w.is_empty() {
                        return Ok(());
                    }
                    if w.len() % row != 0 {
                        return Err(format!(
                            "`{}` at ({}, {}): a run of {} words is no whole row of {row}",
                            a.sym,
                            a.rect.x,
                            a.rect.y,
                            w.len()
                        ));
                    }
                    let h = (w.len() / row) as i32;
                    let mut part = a.clone();
                    part.rect.y = a.rect.y + y;
                    part.rect.h = h;
                    y += h;
                    words += w.len();
                    write(part, w)
                },
            )?;
            if words != row * a.rect.h as usize {
                return Err(format!(
                    "`{}` at ({}, {}): {words} words for {} PEs of {}",
                    a.sym,
                    a.rect.x,
                    a.rect.y,
                    a.rect.w * a.rect.h,
                    a.len
                ));
            }
        }
        Ok(())
    }

    /// The static arrays' words written, once a load (an array `words` leaves out its file's). One
    /// with no words on file is refused: a PE waiting on it hangs the launch.
    pub fn write_statics(&self, rt: &mut dyn Fabric, words: &[Option<Vec<u32>>]) -> R<()> {
        for (k, a) in self.h2d.iter().enumerate() {
            if a.group.is_some() || a.entry.is_some() {
                continue;
            }
            match words.get(k).and_then(Option::as_ref) {
                Some(w) => rt.write(a, w)?,
                None => rt.write(a, &npy_words(a.file()?)?.1)?,
            }
        }
        Ok(())
    }

    /// A decode step on a loaded program: `tokens` at `positions` (a row each), every array
    /// written, then [`Reading::launch`].
    pub fn decode_step(
        &self,
        host: &mut Forward,
        trace: &Trace,
        tokens: &[u32],
        positions: &[u32],
    ) -> R<(StreamStep<'static>, Vec<Option<Vec<f32>>>)> {
        let step = StreamStep {
            lanes: self
                .classes
                .iter()
                .map(|c| StreamLane {
                    word: c.word,
                    rows: c.rows as usize,
                    clip: None,
                })
                .collect(),
            inputs: Vec::new(),
            tokens: tokens.to_vec(),
        };
        let given: Vec<Captured> = (trace.values.iter().enumerate())
            .filter(|(_, d)| matches!(d.def, Def::Input(RuntimeInput::Positions)))
            .map(|(v, _)| Captured {
                node: usize::MAX,
                input: false,
                value: ValueId(v as u32),
                rows: tokens.len(),
                width: 1,
                values: Values::I32(positions.iter().map(|&p| p as i32).collect()),
            })
            .collect();
        let entries = self.entries_given(host, &step, &given)?;
        Ok((step, entries))
    }

    /// The K / V PEs told a decode step's first position ([`Reading::pos`]).
    pub fn write_pos(&self, rt: &mut dyn Fabric, at: u32) -> R<()> {
        self.position_held(at)?;
        write_all(rt, &self.pos, "pos", &[at])
    }

    /// A position the program's pages hold, refused past them (the wafer's own `fab.below` check is
    /// a memory safety net, not a range check).
    fn position_held(&self, at: u32) -> R<()> {
        match self.paging {
            Some(g) if u64::from(at) >= u64::from(g.pages) * u64::from(g.size) => Err(format!(
                "position {at} past the {} the program's {} pages of {} hold",
                g.pages * g.size,
                g.pages,
                g.size
            )),
            _ => Ok(()),
        }
    }

    /// A step's rows, each (lane slot, state slot, position), refused by name past the program's
    /// range.
    pub fn rows_held(&self, rows: &[(u32, u32, u32)]) -> R<()> {
        for &(lane, state, at) in rows {
            if let Some(g) = self.paging
                && lane >= g.lanes
            {
                return Err(format!(
                    "lane slot {lane} past the program's {} lane slots",
                    g.lanes
                ));
            }
            if let Some(n) = self.slots
                && state >= n
            {
                return Err(format!(
                    "state slot {state} past the {n} its recurrences' columns hold"
                ));
            }
            self.position_held(at)?;
        }
        Ok(())
    }

    /// Page tables in the program's range, refused by name past it.
    pub fn pages_held(&self, table: &[u32]) -> R<()> {
        let Some(g) = self.paging else {
            return Err("page tables for a program that pages nothing".into());
        };
        if table.len() != (g.lanes * g.pages) as usize {
            return Err(format!(
                "a page table of {} entries for {} lane slots of {} pages",
                table.len(),
                g.lanes,
                g.pages
            ));
        }
        if let Some((i, p)) = table.iter().enumerate().find(|(_, p)| **p >= g.pages) {
            return Err(format!(
                "page {p} (lane slot {}, entry {}) past the program's {} pages",
                i as u32 / g.pages,
                i as u32 % g.pages,
                g.pages
            ));
        }
        Ok(())
    }

    /// The attention and state PEs told a step's rows' (lane slot, state slot, position); `n` rows
    /// a step, those past `rows` slot 0 at position 0.
    pub fn write_rows(&self, rt: &mut dyn Fabric, n: usize, rows: &[(u32, u32, u32)]) -> R<()> {
        self.rows_held(&rows[..rows.len().min(n)])?;
        let mut words = vec![0u32; 3 * n];
        for (t, &(lane, state, at)) in rows.iter().enumerate().take(n) {
            words[3 * t] = lane;
            words[3 * t + 1] = state;
            words[3 * t + 2] = at;
        }
        if self.rows_advance && rt.rows_held() == Some(&words[..]) {
            return Ok(());
        }
        write_all(rt, &self.rowp, "rowp", &words)?;
        rt.rows_written(self.rows_advance.then_some(words));
        Ok(())
    }

    /// One sequence's page table: lane slot 0's pages in order.
    #[must_use]
    pub fn one_sequence(&self) -> Vec<u32> {
        let Some(g) = self.paging else {
            return Vec::new();
        };
        // (the rest, and past lane slot 0's, the scratch page: the column's last)
        let mut table = vec![g.pages - 1; (g.lanes * g.pages) as usize];
        for (j, w) in table.iter_mut().take(g.pages as usize - 1).enumerate() {
            *w = j as u32;
        }
        table
    }

    /// The attention PEs told each lane slot's pages (`pages` a slot).
    pub fn write_pages(&self, rt: &mut dyn Fabric, table: &[u32]) -> R<()> {
        self.pages_held(table)?;
        write_all(rt, &self.ptab, "ptab", table)
    }

    pub fn fire(
        &self,
        rt: &mut Runtime,
        entries: &[Option<Vec<f32>>],
    ) -> R<(Vec<Vec<f32>>, Option<u64>)> {
        self.write_static(rt)?;
        self.launch(rt, entries)
    }

    /// One step on a program whose static arrays are written: the step's entries written, `run`
    /// launched, the rows read back and the step's cycles off the PEs' stamps.
    pub fn launch(
        &self,
        rt: &mut Runtime,
        entries: &[Option<Vec<f32>>],
    ) -> R<(Vec<Vec<f32>>, Option<u64>)> {
        let step = StreamStep {
            lanes: Vec::new(),
            inputs: Vec::new(),
            tokens: Vec::new(),
        };
        self.launch_on(rt, &step, entries)
    }

    /// [`Reading::launch`] on any [`Fabric`]: `step` is the step the entries are of.
    pub fn launch_on(
        &self,
        rt: &mut dyn Fabric,
        step: &StreamStep<'_>,
        entries: &[Option<Vec<f32>>],
    ) -> R<(Vec<Vec<f32>>, Option<u64>)> {
        self.launch_with(rt, step, entries, &[])
    }

    /// A step's entries written (each from `entries`, else its file).
    pub fn write_entries(&self, rt: &mut dyn Fabric, entries: &[Option<Vec<f32>>]) -> R<()> {
        for (a, words) in self.h2d.iter().zip(entries) {
            if a.entry.is_none() {
                continue;
            }
            let data: Vec<u32> = match words {
                Some(w) => w.iter().map(|x| x.to_bits()).collect(),
                None => npy_words(a.file()?)?.1,
            };
            rt.write(a, &data)?;
        }
        Ok(())
    }

    /// [`Reading::launch_on`] for a grid program in weight groups: each group's arrays written
    /// before its launch, the rows read back after the last.
    pub fn launch_with(
        &self,
        rt: &mut dyn Fabric,
        step: &StreamStep<'_>,
        entries: &[Option<Vec<f32>>],
        statics: &[Option<Vec<u32>>],
    ) -> R<(Vec<Vec<f32>>, Option<u64>)> {
        self.launch_batch(rt, std::slice::from_ref(step), entries, statics)
    }

    /// [`Reading::launch_with`] of a batch ([`Reading::entries_batch`]).
    pub fn launch_batch(
        &self,
        rt: &mut dyn Fabric,
        steps: &[StreamStep<'_>],
        entries: &[Option<Vec<f32>>],
        statics: &[Option<Vec<u32>>],
    ) -> R<(Vec<Vec<f32>>, Option<u64>)> {
        self.write_entries(rt, entries)?;
        // the groups' launches' cycles, summed
        let mut before: Option<u64> = None;
        for k in 0..self.groups {
            for (i, a) in self.h2d.iter().enumerate() {
                if a.group != Some(k) {
                    continue;
                }
                match statics.get(i).and_then(Option::as_ref) {
                    Some(w) => rt.write(a, w)?,
                    None => rt.write(a, &npy_words(a.file()?)?.1)?,
                }
            }
            if k + 1 < self.groups
                && let Some(c) = rt.launch(self)?
            {
                before = Some(before.unwrap_or(0) + c);
            }
        }
        let (got, cycles) = rt.run_batch(self, steps)?;
        Ok((got, cycles.map(|c| c + before.unwrap_or(0))))
    }
}

impl Runtime {
    /// The launch's cycles off its PEs' stamps (`tsc`: start, end, passes a PE), where the program
    /// stamps them.
    pub fn tsc(&mut self, reading: &Reading) -> R<Option<Vec<u32>>> {
        if !self.rpc().symbols().contains_key("tsc") {
            return Ok(None);
        }
        let (w, h) = (reading.w as usize, reading.h as usize);
        let mut ts = vec![0u32; w * h * 3];
        let rect = Rect {
            x: reading.origin.0,
            y: reading.origin.1,
            w: w as i32,
            h: h as i32,
        };
        self.memcpy_d2h("tsc", &mut ts, rect, 3, Order::RowMajor, DataType::Bits32)
            .map_err(|x| x.to_string())?;
        Ok(Some(ts))
    }

    fn stamps(&mut self, reading: &Reading) -> R<Option<u64>> {
        if !reading.stamped {
            return Ok(None);
        }
        let Some(ts) = self.tsc(reading)? else {
            return Ok(None);
        };
        // (a PE whose line queues held words at the launch's end: the next launch reads them
        // first and stops; the GEMM PE sets the third word's top bit)
        if let Some(i) = ts.chunks(3).position(|s| s[2] & LINE_LEFT != 0) {
            let w = reading.w as usize;
            return Err(format!(
                "PE ({}, {}): a line's words were left queued at the launch's end (the next \
                 launch would read them first)",
                i % w,
                i / w
            ));
        }
        let live = ts.chunks(3).filter(|s| s[0] != 0 || s[1] != 0);
        let (lo, hi) = live.fold((u64::MAX, 0u64), |(lo, hi), s| {
            (lo.min(u64::from(s[0])), hi.max(u64::from(s[1])))
        });
        Ok((hi >= lo).then(|| hi - lo))
    }

    /// Lands the simulator's writes in flight (a word read back from the last's first PE).
    fn land(&mut self) -> R<()> {
        if let Some(one) = self.landing.filter(|_| self.unlanded > 0) {
            let mut ts = [0u32; 3];
            self.memcpy_d2h("tsc", &mut ts, one, 3, Order::RowMajor, DataType::Bits32)
                .map_err(|x| x.to_string())?;
            self.unlanded = 0;
        }
        Ok(())
    }
}

/// Writes the simulator holds in flight before one is landed (its command fan halts on a fourth
/// queued command).
const SIM_IN_FLIGHT: usize = 2;

impl Fabric for Runtime {
    fn launch(&mut self, reading: &Reading) -> R<Option<u64>> {
        self.land()?;
        let args: Vec<u32> = reading.rd.into_iter().collect();
        self.keep_for(reading);
        self.launch_followed(reading, &args)?;
        self.stamps(reading)
    }

    fn rows_held(&self) -> Option<&[u32]> {
        self.rows_held.as_deref()
    }

    fn rows_written(&mut self, rows: Option<Vec<u32>>) {
        self.rows_held = rows;
    }

    fn write(&mut self, a: &Array, words: &[u32]) -> R<()> {
        let dt = if a.u16 {
            DataType::Bits16
        } else {
            DataType::Bits32
        };
        // a band of the rectangle's rows a call, each at most MEMCPY_WORDS
        tracing::debug!(
            sym = %a.sym,
            x = a.rect.x,
            y = a.rect.y,
            w = a.rect.w,
            h = a.rect.h,
            len = a.len,
            bits = if a.u16 { 16 } else { 32 },
            words = words.len(),
            "cerebras: an array written"
        );
        let row = (a.rect.w.max(1) as usize) * a.len.max(1);
        // a kept array written since the load: only the columns any PE changed
        if a.kept
            && !a.rowbcast
            && !self.simulated()
            && words.len() == row * a.rect.h.max(0) as usize
        {
            let key = (a.sym.clone(), a.rect.x, a.rect.y);
            let (w, h) = (a.rect.w.max(0) as usize, a.rect.h.max(0) as usize);
            let plan = self
                .kept
                .get(&key)
                .filter(|old| old.len() == words.len())
                .and_then(|old| kept_writes(old, words, w, h, a.len));
            if let Some(plan) = plan {
                for kw in &plan {
                    let mut data = Vec::with_capacity(h * kw.w * kw.n);
                    for y in 0..h {
                        for x in kw.x..kw.x + kw.w {
                            let at = (y * w + x) * a.len;
                            data.extend_from_slice(&words[at..at + kw.n]);
                        }
                    }
                    let rect = Rect {
                        x: a.rect.x + kw.x as i32,
                        w: kw.w as i32,
                        ..a.rect
                    };
                    self.memcpy_h2d(&a.sym, &data, rect, kw.n, Order::RowMajor, dt)
                        .map_err(|x| x.to_string())?;
                }
                self.kept.insert(key, words.to_vec());
                return Ok(());
            }
            self.kept.insert(key, words.to_vec());
        } else {
            // (any other write of the symbol: what it held before no longer known)
            self.kept.retain(|k, _| k.0 != a.sym);
        }
        // each row's PEs the same words: written once a row, broadcast along it
        if a.rowbcast && words.len() == row * a.rect.h.max(0) as usize {
            let firsts: Vec<u32> = words
                .chunks(row)
                .flat_map(|r| r[..a.len].iter().copied())
                .collect();
            match self.memcpy_h2d_rowbcast(&a.sym, &firsts, a.rect, a.len, dt) {
                Ok(()) => return Ok(()),
                // (an SDK with no row broadcast: every PE's words, as any array's)
                Err(crate::sdk::Error::MissingSymbol(..)) => {}
                Err(x) => return Err(x.to_string()),
            }
        }
        let band = (MEMCPY_WORDS / row).max(1);
        let mut y = 0;
        while y < a.rect.h as usize {
            let n = band.min(a.rect.h as usize - y);
            let rect = Rect {
                y: a.rect.y + y as i32,
                h: n as i32,
                ..a.rect
            };
            let part = words.get(y * row..(y + n) * row).ok_or_else(|| {
                format!(
                    "`{}`: {} words for {} rows of {row}",
                    a.sym,
                    words.len(),
                    a.rect.h
                )
            })?;
            self.memcpy_h2d(&a.sym, part, rect, a.len, Order::RowMajor, dt)
                .map_err(|x| x.to_string())?;
            // the simulator halts on a fourth queued command: at most two writes in flight, then a
            // read (a word of the last's first PE) lands them
            if self.simulated() && self.rpc().symbols().contains_key("tsc") {
                self.unlanded += 1;
                self.landing = Some(Rect { w: 1, h: 1, ..rect });
                if self.unlanded >= SIM_IN_FLIGHT {
                    let one = Rect { w: 1, h: 1, ..rect };
                    let mut ts = [0u32; 3];
                    self.memcpy_d2h("tsc", &mut ts, one, 3, Order::RowMajor, DataType::Bits32)
                        .map_err(|x| x.to_string())?;
                    self.unlanded = 0;
                }
            }
            y += n;
        }
        Ok(())
    }

    fn run(
        &mut self,
        reading: &Reading,
        _step: &StreamStep<'_>,
    ) -> R<(Vec<Vec<f32>>, Option<u64>)> {
        let e = |x: crate::sdk::Error| x.to_string();
        self.land()?;
        let args: Vec<u32> = reading.rd.into_iter().collect();
        self.keep_for(reading);
        let t0 = std::time::Instant::now();
        self.launch_followed(reading, &args)?;
        if reading.poll && self.rpc().symbols().contains_key("tsc") {
            // (a word of the first PE's stamps: the read waits on the launch there)
            let mut one = [0u32; 1];
            let rect = Rect {
                x: reading.origin.0,
                y: reading.origin.1,
                w: 1,
                h: 1,
            };
            self.memcpy_d2h("tsc", &mut one, rect, 1, Order::RowMajor, DataType::Bits32)
                .map_err(e)?;
            self.io.done.0 += 1;
            self.io.done.1 += t0.elapsed().as_nanos() as u64;
        }
        let mut got: Vec<Vec<f32>> = vec![Vec::new(); reading.d2h.len()];
        // a band's backs one memcpy of the rectangle around them: a call's latency, not its words,
        // is a step's readback
        for (rect, members) in packs(&reading.d2h) {
            let len = reading.d2h[members[0]].len;
            let mut out = vec![0u32; len * (rect.w * rect.h) as usize];
            self.memcpy_d2h(
                &reading.d2h[members[0]].sym,
                &mut out,
                rect,
                len,
                Order::RowMajor,
                DataType::Bits32,
            )
            .map_err(e)?;
            for i in members {
                got[i] = cut(&out, rect, &reading.d2h[i])
                    .iter()
                    .map(|&x| f32::from_bits(x))
                    .collect();
            }
        }
        let cycles = self.stamps(reading)?;
        Ok((got, cycles))
    }

    fn argmax(&mut self, reading: &Reading) -> R<Option<Vec<u32>>> {
        if reading.argmax.is_empty() {
            return Ok(None);
        }
        let rows = reading
            .argmax
            .iter()
            .flat_map(|b| &b.rows)
            .max()
            .map_or(0, |r| r + 1);
        let mut best: Vec<(f32, u32)> = vec![(f32::NEG_INFINITY, u32::MAX); rows];
        for run in runs(&reading.argmax) {
            let (b, n) = (&reading.argmax[run.start], run.len() as i32);
            let mut out = vec![0u32; b.len * n as usize];
            let rect = Rect {
                x: b.x,
                y: b.y,
                w: n,
                h: 1,
            };
            self.memcpy_d2h(
                &b.sym,
                &mut out,
                rect,
                b.len,
                Order::RowMajor,
                DataType::Bits32,
            )
            .map_err(|x| x.to_string())?;
            for (back, words) in reading.argmax[run].iter().zip(out.chunks(b.len)) {
                for (i, &tr) in back.rows.iter().enumerate() {
                    let (m, k) = (f32::from_bits(words[2 * i]), words[2 * i + 1]);
                    let have = &mut best[tr];
                    if m > have.0 || (m == have.0 && k < have.1) {
                        *have = (m, k);
                    }
                }
            }
        }
        // a row with no logit past the PEs' floor (NaN logits) is refused, not taken as column 0
        if let Some(t) = best.iter().position(|&(m, _)| m.is_nan() || m <= -3.0e38) {
            return Err(format!(
                "row {t}: no finite logit (a refused attention query?)"
            ));
        }
        Ok(Some(best.into_iter().map(|(_, k)| k).collect()))
    }
}

/// `words` written to every PE of each of `rects` (array `sym`, the same words a PE).
fn write_all(rt: &mut dyn Fabric, rects: &[Rect], sym: &str, words: &[u32]) -> R<()> {
    for r in rects {
        let a = Array {
            sym: sym.into(),
            rect: *r,
            len: words.len(),
            u16: false,
            file: None,
            entry: None,
            statics: None,
            group: None,
            rowbcast: false,
            kept: false,
        };
        let n = (r.w * r.h) as usize;
        let all: Vec<u32> = (0..n).flat_map(|_| words.iter().copied()).collect();
        rt.write(&a, &all)?;
    }
    Ok(())
}

impl Runtime {
    /// `run` launched: the followed `rowp` advanced where the program advances it, else (or on
    /// failure) forgotten.
    fn launch_followed(&mut self, reading: &Reading, args: &[u32]) -> R<()> {
        match self.launch("run", args) {
            Ok(()) => {
                self.rows_held = if reading.rows_advance {
                    self.rows_held.as_deref().map(rows_advanced)
                } else {
                    None
                };
                Ok(())
            }
            Err(x) => {
                self.rows_held = None;
                Err(x.to_string())
            }
        }
    }

    /// Before `reading`'s launch: forgets the kept words of arrays it does not keep (another
    /// co-resident reading may write them).
    fn keep_for(&mut self, reading: &Reading) {
        self.kept.retain(|k, _| {
            reading
                .h2d
                .iter()
                .any(|a| a.kept && a.sym == k.0 && a.rect.x == k.1 && a.rect.y == k.2)
        });
    }
}

/// A kept array's write after its first: each PE's first `n` words over the columns `x..x + w`
/// of every row. Not a strided copy: `memcpy_h2d_stride` misplaces words on a CS-3.
#[derive(Clone, Debug, PartialEq, Eq)]
struct KeptWrite {
    x: usize,
    w: usize,
    n: usize,
}

/// More writes than a whole rectangle's one is worth.
const KEPT_WRITES: usize = 16;

/// The writes of a kept array (`w` x `h` PEs, `len` words each) from `old` to `new`: each
/// column's words up to the last any row changed, columns of a count one rectangle. `None` past
/// [`KEPT_WRITES`] writes or when no fewer words than the whole.
fn kept_writes(old: &[u32], new: &[u32], w: usize, h: usize, len: usize) -> Option<Vec<KeptWrite>> {
    let mut n = vec![0usize; w];
    for y in 0..h {
        for (x, nx) in n.iter_mut().enumerate() {
            let at = (y * w + x) * len;
            if let Some(i) = (0..len).rev().find(|&i| old[at + i] != new[at + i]) {
                *nx = (*nx).max(i + 1);
            }
        }
    }
    // (each count's columns' span)
    let mut by: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    for (x, &k) in n.iter().enumerate() {
        if k > 0 {
            let e = by.entry(k).or_insert((x, x));
            e.1 = x;
        }
    }
    let spans: Vec<(usize, (usize, usize))> = by.into_iter().collect();
    let out: Vec<KeptWrite> = spans
        .iter()
        .filter(|&&(k, (a, b))| {
            !spans
                .iter()
                .any(|&(k2, (a2, b2))| k2 > k && a2 <= a && b <= b2)
        })
        .map(|&(k, (a, b))| KeptWrite {
            x: a,
            w: b - a + 1,
            n: k,
        })
        .collect();
    let words: usize = out.iter().map(|kw| kw.w * kw.n * h).sum();
    (out.len() <= KEPT_WRITES && words < w * h * len).then_some(out)
}

/// The stamps' third word's top bit: a PE's line queues held words at the launch's end
/// (`emit::gemm`'s `LINE_LEFT`).
const LINE_LEFT: u32 = 0x8000_0000;

/// How many times a back's words the rectangle around a band's backs may hold before they are
/// read a back at a time.
const PACK_SLACK: usize = 4;

/// The reads of `d2h`: each band's backs one read of their rectangle where it holds at most
/// [`PACK_SLACK`] times their words, else a run ([`runs`]) a read.
fn packs(d2h: &[Back]) -> Vec<(Rect, Vec<usize>)> {
    // (symbol, words a PE, top row, height): a band
    type Band<'a> = (&'a str, usize, i32, i32);
    let mut bands: Vec<(Band<'_>, Vec<usize>)> = Vec::new();
    for (i, b) in d2h.iter().enumerate() {
        let k = (b.sym.as_str(), b.len, b.y, b.h);
        match bands.iter_mut().find(|(g, _)| *g == k) {
            Some((_, m)) => m.push(i),
            None => bands.push((k, vec![i])),
        }
    }
    let mut out = Vec::new();
    for ((_, len, y, h), m) in bands {
        let x0 = m.iter().map(|&i| d2h[i].x).min().unwrap_or(0);
        let x1 = m.iter().map(|&i| d2h[i].x + d2h[i].w).max().unwrap_or(0);
        let words: usize = m
            .iter()
            .map(|&i| (d2h[i].w * d2h[i].h) as usize * len)
            .sum();
        let around = ((x1 - x0) * h) as usize * len;
        if m.len() > 1 && around <= PACK_SLACK * words {
            out.push((
                Rect {
                    x: x0,
                    y,
                    w: x1 - x0,
                    h,
                },
                m,
            ));
            continue;
        }
        let sub: Vec<Back> = m.iter().map(|&i| d2h[i].clone()).collect();
        for run in runs(&sub) {
            let (b, n) = (&sub[run.start], run.len() as i32);
            let (w, hh) = if n > 1 { (n, 1) } else { (b.w, b.h) };
            out.push((
                Rect {
                    x: b.x,
                    y: b.y,
                    w,
                    h: hh,
                },
                run.map(|j| m[j]).collect(),
            ));
        }
    }
    out
}

/// `b`'s words (row-major, `len` a PE) out of a read of `rect` that holds it.
fn cut(out: &[u32], rect: Rect, b: &Back) -> Vec<u32> {
    let mut v = Vec::with_capacity((b.w * b.h) as usize * b.len);
    for yy in 0..b.h {
        let at = (((b.y - rect.y + yy) * rect.w + (b.x - rect.x)) as usize) * b.len;
        v.extend_from_slice(&out[at..at + b.w as usize * b.len]);
    }
    v
}

/// `d2h` in runs of one memcpy each: one-PE reads of one symbol and length at consecutive
/// columns of a row.
fn runs(d2h: &[Back]) -> Vec<std::ops::Range<usize>> {
    let mut out: Vec<std::ops::Range<usize>> = Vec::new();
    for (i, b) in d2h.iter().enumerate() {
        if let Some(r) = out.last_mut() {
            let a = &d2h[r.end - 1];
            let one = |b: &Back| b.w == 1 && b.h == 1;
            if one(a) && one(b) && a.sym == b.sym && a.len == b.len && a.y == b.y && b.x == a.x + 1
            {
                r.end = i + 1;
                continue;
            }
        }
        out.push(i..i + 1);
    }
    out
}

/// A fabric with no wafer: a step's rows as engine-cpu's forward makes them, laid into each
/// [`Back`]'s words. What it is written is kept, not run.
pub struct HostFabric {
    pub forward: Forward,
    pub written: BTreeMap<(String, i32, i32), Vec<u32>>,
}

impl HostFabric {
    #[must_use]
    pub fn new(forward: Forward) -> HostFabric {
        HostFabric {
            forward,
            written: BTreeMap::new(),
        }
    }
}

impl Fabric for HostFabric {
    fn write(&mut self, a: &Array, words: &[u32]) -> R<()> {
        self.written
            .insert((a.sym.clone(), a.rect.x, a.rect.y), words.to_vec());
        Ok(())
    }

    fn run(&mut self, reading: &Reading, step: &StreamStep<'_>) -> R<(Vec<Vec<f32>>, Option<u64>)> {
        self.run_batch(reading, std::slice::from_ref(step))
    }

    fn run_batch(
        &mut self,
        reading: &Reading,
        steps: &[StreamStep<'_>],
    ) -> R<(Vec<Vec<f32>>, Option<u64>)> {
        // each request's values; the batch's row `tr` its request's row `tr % rows`
        let mut made: Vec<BTreeMap<u32, Captured>> = Vec::new();
        for step in steps {
            made.push(
                self.forward
                    .stream_values(step, &[], &reading.out_values())?
                    .into_iter()
                    .map(|c| (c.value.0, c))
                    .collect(),
            );
        }
        let per = reading.request_rows().max(1);
        let mut got = Vec::new();
        for b in &reading.d2h {
            let width: usize = b.parts.iter().map(|p| p.2 as usize).sum();
            let mut buf = vec![0f32; b.len * (b.w * b.h) as usize];
            for (i, &tr) in b.rows.iter().enumerate() {
                // one request: its rows as they are; else request `tr / per`, a padded one the
                // last's
                let (made, tr) = if reading.batch <= 1 {
                    (&made[0], tr)
                } else {
                    (&made[(tr / per).min(made.len() - 1)], tr % per)
                };
                let mut off = 0;
                for &(base, from, n) in &b.parts {
                    let c = made
                        .get(&base)
                        .ok_or_else(|| format!("value %{base} is read back but not made"))?;
                    let Values::F32(v) = &c.values else {
                        return Err(format!("value %{base}: not a float"));
                    };
                    let (from, n) = (from as usize, n as usize);
                    let at = tr * c.width + from;
                    buf[i * width + off..i * width + off + n].copy_from_slice(&v[at..at + n]);
                    off += n;
                }
            }
            got.push(buf);
        }
        Ok((got, None))
    }
}

/// A denoise loop on the host around a device's steps: the poem's flow sigmas, an Euler step a
/// velocity.
#[derive(Clone, Debug, PartialEq)]
pub struct Denoise {
    pub sigmas: Vec<f32>,
    pub train_steps: u32,
    /// The guest inputs the loop sets each step, and the output it integrates.
    pub latent: String,
    pub timestep: String,
    pub velocity: String,
}

impl Denoise {
    /// `steps` steps of the schedule the poem states for `model` over `image_rows` rows.
    pub fn of(model: &str, steps: u32, image_rows: Option<u32>) -> R<Denoise> {
        let facts = poem_compiler::catalog::repository()
            .parse(model)
            .ok_or_else(|| format!("no catalog model {model}"))?
            .try_generative()
            .map_err(|e| format!("{model}: {e:?}"))?
            .ok_or_else(|| format!("{model} states no generative facts"))?;
        let s = facts
            .schedule
            .ok_or_else(|| format!("{model} states no schedule"))?;
        Ok(Denoise {
            sigmas: s.flow_sigmas(steps, image_rows)?,
            train_steps: s.train_steps.max(1),
            latent: "latents".into(),
            timestep: "timestep".into(),
            velocity: "velocity".into(),
        })
    }

    #[must_use]
    pub fn steps(&self) -> usize {
        self.sigmas.len().saturating_sub(1)
    }

    /// Runs the loop from `inputs`: `velocity(i, inputs)` is step `i`'s velocity. Returns the final
    /// latent and each step's velocity.
    pub fn run(
        &self,
        inputs: &mut [(String, Vec<f32>)],
        lanes: usize,
        mut velocity: impl FnMut(usize, &[(String, Vec<f32>)]) -> R<Vec<f32>>,
    ) -> R<(Vec<f32>, Vec<Vec<f32>>)> {
        let at = |inputs: &[(String, Vec<f32>)], n: &str| {
            inputs
                .iter()
                .position(|(k, _)| k == n)
                .ok_or_else(|| format!("no input `{n}`"))
        };
        let (li, ti) = (at(inputs, &self.latent)?, at(inputs, &self.timestep)?);
        let mut vs = Vec::new();
        for i in 0..self.steps() {
            inputs[ti].1 = vec![self.sigmas[i] * self.train_steps as f32; lanes];
            let v = velocity(i, inputs)?;
            let dt = self.sigmas[i + 1] - self.sigmas[i];
            let x = &mut inputs[li].1;
            if v.len() != x.len() {
                return Err(format!(
                    "a velocity of {} values for a latent of {}",
                    v.len(),
                    x.len()
                ));
            }
            for (x, v) in x.iter_mut().zip(&v) {
                *x += dt * v;
            }
            vs.push(v);
        }
        Ok((inputs[li].1.clone(), vs))
    }
}

#[cfg(test)]
mod held_tests {
    use super::*;

    /// A paged program of 2 lane slots, 3 pages of 64 a column, its recurrences 2 slots.
    fn paged() -> Reading {
        let mut r = Reading::host_only("m", "decode", &[]);
        r.paging = Some(Paging {
            size: 64,
            pages: 3,
            lanes: 2,
        });
        r.slots = Some(2);
        r
    }

    #[test]
    fn a_step_s_host_words_past_the_program_s_range_are_refused_by_name() {
        let r = paged();
        assert!(r.rows_held(&[(0, 0, 0), (1, 1, 191)]).is_ok());
        let e = |x: R<()>| x.expect_err("refused");
        assert!(e(r.rows_held(&[(2, 0, 0)])).contains("lane slot 2 past the program's 2"));
        assert!(e(r.rows_held(&[(0, 2, 0)])).contains("state slot 2 past the 2"));
        assert!(e(r.rows_held(&[(0, 0, 192)])).contains("position 192 past the 192"));
        assert!(r.pages_held(&[0, 1, 2, 2, 2, 2]).is_ok());
        assert!(e(r.pages_held(&[0, 1, 2])).contains("3 entries for 2 lane slots of 3 pages"));
        assert!(e(r.pages_held(&[0, 1, 2, 2, 3, 2])).contains("page 3 (lane slot 1, entry 1)"));
        assert!(e(r.position_held(500)).contains("position 500"));
    }

    #[test]
    fn a_bands_backs_read_as_one_rectangle_cut_as_each_alone() {
        // a 128² denoise's backs: 1 x 64 strips every third column in four bands, a lone 3 x 2 back
        let back = |x: i32, y: i32, w: i32, h: i32, len: usize| Back {
            sym: "hout".into(),
            x,
            y,
            w,
            h,
            len,
            rows: Vec::new(),
            parts: Vec::new(),
        };
        let mut d2h: Vec<Back> = Vec::new();
        for x in (2..384).step_by(3) {
            for y in [64, 192, 320, 448] {
                d2h.push(back(x, y, 1, 64, 1));
            }
        }
        d2h.push(back(5, 600, 3, 2, 2));
        let reads = packs(&d2h);
        assert_eq!(reads.len(), 5);
        let mut seen = vec![0; d2h.len()];
        // each PE's word k: its place and k, as a read of the rectangle returns them
        let word = |x: i32, y: i32, k: usize| (y as u32) << 16 | (x as u32) << 4 | k as u32;
        for (rect, members) in &reads {
            let len = d2h[members[0]].len;
            let mut out = Vec::new();
            for y in rect.y..rect.y + rect.h {
                for x in rect.x..rect.x + rect.w {
                    out.extend((0..len).map(|k| word(x, y, k)));
                }
            }
            for &i in members {
                let b = &d2h[i];
                let mut alone = Vec::new();
                for y in b.y..b.y + b.h {
                    for x in b.x..b.x + b.w {
                        alone.extend((0..b.len).map(|k| word(x, y, k)));
                    }
                }
                assert_eq!(cut(&out, *rect, b), alone);
                seen[i] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n == 1));
    }

    #[test]
    fn a_kept_arrays_write_is_its_changed_columns_words() {
        // a 1024² latent band: word 0 at columns 1 and 2 of each three, words 1-8 at column 2; then
        // a step's change
        let (w, h, len) = (12, 3, 9);
        let mut old = vec![0u32; w * h * len];
        let mut new = old.clone();
        for y in 0..h {
            for x in 0..w {
                let at = (y * w + x) * len;
                if x % 3 != 0 {
                    old[at] = 1;
                    new[at] = 2;
                }
                if x % 3 == 2 {
                    for i in 1..len {
                        old[at + i] = 10;
                        new[at + i] = 20 + i as u32;
                    }
                }
            }
        }
        let plan = kept_writes(&old, &new, w, h, len).expect("a plan");
        let kw = |x, w, n| KeptWrite { x, w, n };
        assert_eq!(plan, vec![kw(1, 10, 1), kw(2, 10, 9)]);
        // the words it sends make `old` `new`
        let mut dev = old.clone();
        for k in &plan {
            for y in 0..h {
                for x in k.x..k.x + k.w {
                    let at = (y * w + x) * len;
                    dev[at..at + k.n].copy_from_slice(&new[at..at + k.n]);
                }
            }
        }
        assert_eq!(dev, new);
        // nothing changed: no writes; everything: the whole rectangle's
        assert_eq!(kept_writes(&new, &new, w, h, len), Some(Vec::new()));
        let all: Vec<u32> = new.iter().map(|x| x + 1).collect();
        assert_eq!(kept_writes(&new, &all, w, h, len), None);
    }

    #[test]
    fn a_new_requests_context_mid_load_sends_its_columns_alone() {
        // two batch slots of 4 rows, 8 words a PE on 12 columns; the second slot's context changes
        // on columns 3-6, the timestep (word 0) on every column
        let (w, h, len) = (12, 8, 9);
        let mut old = vec![0u32; w * h * len];
        for (i, v) in old.iter_mut().enumerate() {
            *v = i as u32;
        }
        let mut new = old.clone();
        for y in 0..h {
            for x in 0..w {
                let at = (y * w + x) * len;
                new[at] += 1_000_000;
                if y >= 4 && (3..7).contains(&x) {
                    for i in 1..len {
                        new[at + i] += 2_000_000;
                    }
                }
            }
        }
        let plan = kept_writes(&old, &new, w, h, len).expect("a plan");
        let kw = |x, w, n| KeptWrite { x, w, n };
        assert_eq!(plan, vec![kw(0, 12, 1), kw(3, 4, 9)]);
        let sent: usize = plan.iter().map(|k| k.w * k.n * h).sum();
        assert!(sent < w * h * len / 2, "{sent} words");
        // the device after the writes holds what a whole write leaves
        let mut dev = old.clone();
        for k in &plan {
            for y in 0..h {
                for x in k.x..k.x + k.w {
                    let at = (y * w + x) * len;
                    dev[at..at + k.n].copy_from_slice(&new[at..at + k.n]);
                }
            }
        }
        assert_eq!(dev, new);
    }
}
