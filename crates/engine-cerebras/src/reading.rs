//! A stream reading's host side (GENERIC.md M1e): what the engine does around a step the wafer
//! runs (a diffusion transformer's denoise), read off the program's `host.json` and computed by
//! the host forward from the step's own inputs, not taken from a capture.
//!
//! `wafer-compile stream` states the reading: its classes (request words and rows), every array
//! the host writes (`h2d`), which of them a step's inputs make (`entry`: each word a value's
//! element, [`kernels_cerebras::stream::host::Gather`]) and the rows the host reads back (`d2h`,
//! `outs`). The engine:
//!
//! - lays the step out as the poem's host ops read it ([`Reading::step`]: a lane a class, its
//!   request word, a media lane's voxel grid);
//! - computes each entry's values with [`engine_cpu::Forward::stream_values`] on a weightless
//!   [`engine_cpu::Forward::host`]: the trace's own entry ops (patchify, the cell table and its
//!   split, the timestep's sinusoid, the request ids, the context's rows) over the step's inputs,
//!   gathered into the arrays' words with the same [`apply`] the capture-fed compile uses;
//! - after the step, the trace's own exit ops over the rows read back ([`Reading::finish`]: the
//!   velocity unpatchified);
//! - between steps, the poem's schedule ([`Denoise`]: its flow sigmas, an Euler step a velocity);
//! - the static arrays (the weight tiles, the regions' shared residents) from the checkpoint
//!   ([`Reading::statics`]): the values each array's spans read, made by an
//!   [`engine_cpu::Forward`] over the checkpoint's planes, cast to the array's words with the
//!   compile's own [`apply_spans`]. A program compiled with no capture runs on any checkpoint.
//!
//! A step runs on a [`Fabric`]: the SDK runtime (a CS system or the simulator), or
//! [`HostFabric`], the rows a program returns as engine-cpu's forward makes them (no fabric: the
//! engine's request path checked on the host).
//!
//! Nothing here names a model's op: the values are the trace's, by id, and the runner is the
//! same for any stream reading the compile states.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use engine_cpu::Forward;
use engine_cpu::forward::{Captured, StreamLane, StreamStep, Values};
use kernels_cerebras::emit::state::{Pe, StateRoot};
use kernels_cerebras::stream::host::{Span, Word, apply, apply_spans};
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

/// A memcpy rectangle the host writes: its words from `file` (a program's own array, or any
/// array of a program compiled from a capture), from `statics` (spans over values the
/// checkpoint makes), or, for an entry, gathered from the step's values each step.
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
/// The most words one host-to-device memcpy call carries ([`Fabric::write`] on the SDK
/// runtime cuts a larger array into bands of its rows).
pub const MEMCPY_WORDS: usize = 1 << 24;

pub trait Fabric {
    /// Writes array `a`'s words (each an f32's bits, or a 16-bit word).
    fn write(&mut self, a: &Array, words: &[u32]) -> R<()>;
    /// Runs `step` on the arrays written: the rows back (each [`Back`]'s words) and the step's
    /// cycles where the program stamps them.
    fn run(&mut self, reading: &Reading, step: &StreamStep<'_>) -> R<(Vec<Vec<f32>>, Option<u64>)>;
    /// Runs a batch's `steps` (a request's each, the program's [`Reading::batch`]) on the
    /// arrays written: a device's program runs the arrays (the steps are the entries'), the
    /// host's runs each step and lays their rows out as the batch's.
    fn run_batch(
        &mut self,
        reading: &Reading,
        steps: &[StreamStep<'_>],
    ) -> R<(Vec<Vec<f32>>, Option<u64>)> {
        self.run(reading, steps.first().ok_or("a batch of no steps")?)
    }
    /// Opens the program linked in `link` in place of the one open (a fabric runs one
    /// program at a time; the host's runs every reading's and opens nothing). Returns whether
    /// it was loaded afresh (its arrays to be written again).
    fn switch(&mut self, link: &Path) -> R<bool> {
        let _ = link;
        Ok(false)
    }
    /// Launches `run` on the arrays written and waits, reading no rows back (a weight group's
    /// launch before the reading's last): its cycles where the program stamps them. The host's
    /// runs a step whole in [`Fabric::run`].
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

/// Each stream program in a process of its own (`wafer-fabric`, the command `cmd`), the one
/// open at a time: the simulator holds one program a process, so readings each linked on their
/// own (`[cerebras.debug] stream_processes`) run in a child each, one engine process driving
/// them.
pub struct Children {
    pub cmd: Vec<String>,
    kids: BTreeMap<PathBuf, Child>,
    cur: Option<PathBuf>,
}

struct Child {
    proc: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: std::io::BufReader<std::process::ChildStdout>,
}

impl Children {
    #[must_use]
    pub fn new(cmd: Vec<String>) -> Children {
        Children {
            cmd,
            kids: BTreeMap::new(),
            cur: None,
        }
    }

    /// `req` (a JSON line) and `words` to the child open; its reply and its f32 words after.
    fn call(&mut self, req: &Value, words: &[u32]) -> R<(Value, Vec<f32>)> {
        use std::io::{BufRead, Read, Write};
        let cur = self.cur.clone().ok_or("no program is open")?;
        let kid = self.kids.get_mut(&cur).ok_or("no program is open")?;
        let io = |e: std::io::Error| format!("wafer-fabric: {e}");
        writeln!(kid.stdin, "{req}").map_err(io)?;
        let bytes: Vec<u8> = words.iter().flat_map(|x| x.to_le_bytes()).collect();
        kid.stdin.write_all(&bytes).map_err(io)?;
        kid.stdin.flush().map_err(io)?;
        let mut line = String::new();
        if kid.stdout.read_line(&mut line).map_err(io)? == 0 {
            return Err("wafer-fabric: the child ended".into());
        }
        let head: Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        if head["ok"].as_bool() != Some(true) {
            return Err(format!("wafer-fabric: {}", head["err"]));
        }
        let n: usize = (head["lens"].as_array().into_iter().flatten())
            .filter_map(Value::as_u64)
            .sum::<u64>() as usize;
        let mut data = vec![0u8; 4 * n];
        kid.stdout.read_exact(&mut data).map_err(io)?;
        Ok((
            head,
            data.as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        ))
    }
}

impl Drop for Children {
    fn drop(&mut self) {
        for (_, mut k) in std::mem::take(&mut self.kids) {
            drop(k.stdin);
            let _ = k.proc.wait();
        }
    }
}

impl Fabric for Children {
    fn write(&mut self, a: &Array, words: &[u32]) -> R<()> {
        let req = serde_json::json!({"op": "write", "sym": a.sym, "x": a.rect.x, "y": a.rect.y,
            "w": a.rect.w, "h": a.rect.h, "len": a.len, "u16": a.u16, "n": words.len()});
        self.call(&req, words).map(|_| ())
    }

    fn run(
        &mut self,
        reading: &Reading,
        _step: &StreamStep<'_>,
    ) -> R<(Vec<Vec<f32>>, Option<u64>)> {
        let req = serde_json::json!({"op": "run", "dir": reading.dir});
        let (head, data) = self.call(&req, &[])?;
        let mut got = Vec::new();
        let mut at = 0;
        for l in (head["lens"].as_array().into_iter().flatten()).filter_map(Value::as_u64) {
            got.push(data[at..at + l as usize].to_vec());
            at += l as usize;
        }
        Ok((got, head["cycles"].as_u64()))
    }

    fn launch(&mut self, reading: &Reading) -> R<Option<u64>> {
        let req = serde_json::json!({"op": "launch", "dir": reading.dir});
        self.call(&req, &[])
            .map(|(head, _)| head["cycles"].as_u64())
    }

    fn switch(&mut self, link: &Path) -> R<bool> {
        if self.cur.as_deref() == Some(link) && self.kids.contains_key(link) {
            return Ok(false);
        }
        // one child at a time (one simulator machine-wide): the one open ends
        for (_, mut k) in std::mem::take(&mut self.kids) {
            drop(k.stdin);
            let _ = k.proc.wait();
        }
        self.cur = Some(link.to_path_buf());
        let (bin, args) = self.cmd.split_first().ok_or("no wafer-fabric command")?;
        let mut proc = std::process::Command::new(bin)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("{bin}: {e}"))?;
        let stdin = proc.stdin.take().ok_or("no child stdin")?;
        let stdout = std::io::BufReader::new(proc.stdout.take().ok_or("no child stdout")?);
        self.kids.insert(
            link.to_path_buf(),
            Child {
                proc,
                stdin,
                stdout,
            },
        );
        self.call(&serde_json::json!({"op": "open", "link": link}), &[])?;
        Ok(true)
    }
}

/// A root's rows read back: each of its own rows (the step's row `rows[i]`) the `parts`
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
    /// A decode program's argmax a row (`host.json`'s `argmax`): each root's two words a row,
    /// its largest logit and that logit's column (`parts`: its columns, `(0, from, len)`).
    pub argmax: Vec<Back>,
    /// The values read back whole: (value, rows, width).
    pub outs: Vec<(u32, usize, usize)>,
    /// The capture the program was compiled from, if any: each value read back, its file there.
    pub refs: Vec<(u32, PathBuf)>,
    pub w: u32,
    pub h: u32,
    /// A row attends the rows before it alone (a prefill's mask): a lane's rows past its own
    /// tokens are padding no earlier row reads.
    pub causal: bool,
    /// No program runs it: the host's forward makes its outputs whole ([`Reading::host_only`]).
    pub host_only: bool,
    /// Whether a launch reads its PEs' stamps back (its cycles): a served decode step reads none.
    pub stamped: bool,
    /// Its PEs' first column and row in a program of co-resident readings (`stream::coresident`).
    pub origin: (i32, i32),
    /// The id its launch takes in a program of co-resident readings (`None`: a program its own).
    pub rd: Option<u32>,
    /// The clip a grid reading's program is compiled for (its rows are the clip's voxels, not
    /// token rows).
    pub clip: Option<[u32; 3]>,
    /// A grid program's weight groups: a step is a launch a group, each group's arrays written
    /// before it (the stores kept between); 0: one launch.
    pub groups: usize,
    /// A decode program's K / V PEs (GENERIC.md G1): each told a step's first position (`pos`),
    /// their caches kept across steps.
    pub pos: Vec<Rect>,
    /// A decode program's paging (GENERIC.md G1.8): its attention columns' pages, `size`
    /// positions each, `pages` a column, `lanes` lane slots (a step's rows); `rowp` its PEs told a
    /// step's rows' (lane slot, position), `ptab` each lane slot's pages.
    pub paging: Option<Paging>,
    pub rowp: Vec<Rect>,
    pub ptab: Vec<Rect>,
    /// The state slots every recurrence's column holds (the least; `None`: no recurrence): a
    /// row's state slot is below it.
    pub slots: Option<u32>,
    /// Requests a launch (the GEMM regime's batch): the step's rows each request's rows in
    /// turn, its lane values a request's lanes in turn.
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
    let (shape, v) = kernels_cerebras::npy::read_f32_or_i32(path)?;
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
        // the program's state (`state`: its regions, their slots, pages and PEs), the one
        // description of it the engine reads
        // (a program compiled before the state root states its K / V PEs as top-level keys: it
        // is refused, not read)
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
                .ok_or_else(|| format!("host.json: no `{k}` (compiled before M1e?)"))
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

    /// A reading of `model` no program runs (the host fabric's fallback for one the stream
    /// compiler refuses): `classes` its lanes, no arrays; [`Reading::finish`] makes its outputs
    /// from the step's inputs alone.
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

    /// A step of the reading: a lane a class (its request word and rows), a lane whose word
    /// reads a voxel input (the latent) on the voxel grid `grid`, and the guest inputs by name.
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

    /// [`Reading::entries`] of a batch: `steps` a request's each (at most the program's
    /// [`Reading::batch`]; fewer: the last request's values stand for the rest, made once),
    /// every value the entries gather from the requests' rows in turn (a value of a row a
    /// lane, its lanes in turn).
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

    /// Request `b`'s rows of a batch's values read back ([`Reading::rows_back`] of a batched
    /// launch): each value's rows a request's in turn.
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
                if w[0] != kernels_cerebras::stream::host::NONE && !wanted.contains(&v) {
                    wanted.push(v);
                }
            }
        }
        wanted
    }

    /// Each array's words for `step`, computed on the host forward `host` (holding no weights
    /// but those the host's own ops read, [`Forward::host_for`]): an entry's from the step's
    /// values, `None` for an array whose words are the file's.
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

    /// The values the host reads back, whole (rows the program returns none of are NaN), from
    /// each [`Back`]'s words in `got`.
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

    /// Every array no step's inputs make (the tiles, the regions' tables and static residents),
    /// from its file (a program compiled from a capture): written once on a loaded program,
    /// before its first [`Reading::launch`].
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

    /// The words of every array no step's inputs make, from the checkpoint: each array's spans
    /// over the values `weights` (a forward holding the checkpoint's planes) makes of them, at
    /// `step` (any step: these values are the weights' alone); an array with no spans (a table,
    /// a PE's config words) its file's. `None` for an entry.
    pub fn statics(
        &self,
        weights: &mut Forward,
        step: &StreamStep<'_>,
    ) -> R<Vec<Option<Vec<u32>>>> {
        let mut wanted: Vec<ValueId> = Vec::new();
        for a in self.h2d.iter().filter(|a| a.entry.is_none()) {
            for s in a.statics.iter().flatten() {
                let v = ValueId(s[0]);
                let constant = s[0] == kernels_cerebras::stream::host::NONE
                    || s[0] == kernels_cerebras::stream::host::UNIT
                    || s[0] == kernels_cerebras::stream::host::LIT;
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

    /// The static arrays' words ([`Reading::statics`]) handed to `write` a run of rows of PEs at a
    /// time, each run at most `chunk` words (at least a row), each value made from `weights`
    /// only while a span reads it: the arrays made in bounded memory, not all at once (klein's
    /// GEMM tapes are 8 GB of words over 16 GB of their values in f32). Arrays of a weight group
    /// (written a launch each) are not written here.
    pub fn statics_streamed(
        &self,
        weights: &mut Forward,
        step: &StreamStep<'_>,
        chunk: usize,
        write: &mut dyn FnMut(Array, Vec<u32>) -> R<()>,
    ) -> R<()> {
        let weights = std::cell::RefCell::new(weights);
        // values made a batch at a time (each batch one pass of the weights' forward: the next
        // values the spans read, in order, up to PREFETCH of them), each handed out once
        const PREFETCH: usize = 8;
        let made = |vs: &[u32]| -> R<Vec<(u32, kernels_cerebras::stream::host::Rows)>> {
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
                let constant = v == kernels_cerebras::stream::host::NONE
                    || v == kernels_cerebras::stream::host::UNIT
                    || v == kernels_cerebras::stream::host::LIT;
                if !constant && seen.insert(v) {
                    order.push(v);
                }
            }
            let ahead = std::cell::RefCell::new((
                0usize,
                BTreeMap::<u32, kernels_cerebras::stream::host::Rows>::new(),
            ));
            let get = |v: u32| -> R<kernels_cerebras::stream::host::Rows> {
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
            kernels_cerebras::stream::host::apply_spans_into(spans, &get, rows * row, &mut |w| {
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
            })?;
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

    /// The static arrays' words ([`Reading::statics`]) written, once a load: an array `words`
    /// leaves out its file's (as [`Reading::write_static`] writes it; one with no words on file
    /// refused, never left unwritten: a PE waiting on it hangs the launch).
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

    /// One step on a loaded program: every array written (an entry's words from `entries`, the
    /// rest from its file), then [`Reading::launch`].
    /// A decode program's step (GENERIC.md G1): `tokens` (a row each) at `positions` (a row
    /// each), the trace's positions given as theirs; the step and its entries.
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

    /// A position the program's pages hold (a lane's `pages` pages of `size`), refused past
    /// them: the wafer holds a word it is told below what it indexes (`fab.below`), a memory
    /// safety net, never a host's range check.
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

    /// A step's rows, each (lane slot, state slot, position) in the program's range, refused
    /// by name past it.
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

    /// Page tables in the program's range (a lane slot's `pages` entries each, every page id
    /// below `pages`), refused by name past it.
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

    /// The attention and state PEs told a step's rows' (lane slot, state slot, position)
    /// ([`Reading::rowp`]; `n` rows a step, those past `rows` slot 0 at position 0).
    pub fn write_rows(&self, rt: &mut dyn Fabric, n: usize, rows: &[(u32, u32, u32)]) -> R<()> {
        self.rows_held(&rows[..rows.len().min(n)])?;
        let mut words = vec![0u32; 3 * n];
        for (t, &(lane, state, at)) in rows.iter().enumerate().take(n) {
            words[3 * t] = lane;
            words[3 * t + 1] = state;
            words[3 * t + 2] = at;
        }
        write_all(rt, &self.rowp, "rowp", &words)
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

    /// The attention PEs told each lane slot's pages (`table`: lane slot by lane slot, `pages` a
    /// slot; [`Reading::ptab`]).
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

    /// One step on a program whose static arrays are written ([`Reading::write_static`], once:
    /// the program re-runs a step a launch, its tiles kept): the step's entries written from
    /// `entries`, `run` launched, the rows read back (each [`Back`]'s words) and the step's
    /// cycles off the PEs' stamps (`tsc`: start, end, passes a PE).
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

    /// [`Reading::launch`] on any [`Fabric`]: `step` is the step the entries are of (what a
    /// [`HostFabric`] runs).
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

    /// [`Reading::launch_on`], a grid program in weight groups a launch a group: each group's
    /// arrays written before its launch (a static's words from `statics`, as
    /// [`Reading::statics`] made them, else its file's), the rows read back after the last.
    pub fn launch_with(
        &self,
        rt: &mut dyn Fabric,
        step: &StreamStep<'_>,
        entries: &[Option<Vec<f32>>],
        statics: &[Option<Vec<u32>>],
    ) -> R<(Vec<Vec<f32>>, Option<u64>)> {
        self.launch_batch(rt, std::slice::from_ref(step), entries, statics)
    }

    /// [`Reading::launch_with`] of a batch: `steps` a request's each, `entries` the batch's
    /// ([`Reading::entries_batch`]).
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
    /// The launch's cycles off its PEs' stamps (`tsc`: start, end, passes a PE) over the
    /// reading's rectangle, where the program stamps them.
    /// Every PE's three stamp words (`tsc`, row major over the program's `w x h`), where the
    /// program stamps.
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
        let live = ts.chunks(3).filter(|s| s[0] != 0 || s[1] != 0);
        let (lo, hi) = live.fold((u64::MAX, 0u64), |(lo, hi), s| {
            (lo.min(u64::from(s[0])), hi.max(u64::from(s[1])))
        });
        Ok((hi >= lo).then(|| hi - lo))
    }

    /// The simulator's writes still in flight landed (a word read back from the last's first
    /// PE), before a program runs over them.
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

/// Writes the simulator holds in flight before one is landed (its command fan halts on a
/// fourth command queued: two writes and the read that lands them).
const SIM_IN_FLIGHT: usize = 2;

impl Fabric for Runtime {
    fn launch(&mut self, reading: &Reading) -> R<Option<u64>> {
        self.land()?;
        let args: Vec<u32> = reading.rd.into_iter().collect();
        self.launch("run", &args).map_err(|x| x.to_string())?;
        self.stamps(reading)
    }

    fn write(&mut self, a: &Array, words: &[u32]) -> R<()> {
        let dt = if a.u16 {
            DataType::Bits16
        } else {
            DataType::Bits32
        };
        // a band of the rectangle's rows a call, each at most MEMCPY_WORDS (one call of an
        // array a decode program's head holds, 155M words, the simulator's memcpy halts on)
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
            // the simulator's memcpy halo takes a write's command before the fabric has taken
            // the one before; a fourth command queued that way halts it (an `hcf` in the command
            // fan, GENERIC.md G1.6): at most two writes in flight, then a read (a word of the
            // last's first PE) lands them, the read the third
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
        self.launch("run", &args).map_err(e)?;
        let mut got = Vec::new();
        // a run of one-PE reads along a row (a head's column roots, the logits) one memcpy
        for run in runs(&reading.d2h) {
            let (b, n) = (&reading.d2h[run.start], run.len() as i32);
            let (w, h) = if n > 1 { (n, 1) } else { (b.w, b.h) };
            let mut out = vec![0u32; b.len * (w * h) as usize];
            let rect = Rect {
                x: b.x,
                y: b.y,
                w,
                h,
            };
            self.memcpy_d2h(
                &b.sym,
                &mut out,
                rect,
                b.len,
                Order::RowMajor,
                DataType::Bits32,
            )
            .map_err(e)?;
            let per = out.len() / run.len();
            for c in out.chunks(per) {
                got.push(c.iter().map(|&x| f32::from_bits(x)).collect());
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
        // (a row no logit of which passed the PEs' floor: its logits NaN, as a step whose
        // attention refused a query leaves them; refused, not taken as column 0)
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
        };
        let n = (r.w * r.h) as usize;
        let all: Vec<u32> = (0..n).flat_map(|_| words.iter().copied()).collect();
        rt.write(&a, &all)?;
    }
    Ok(())
}

/// `d2h` in runs read by one memcpy each: consecutive one-PE reads of one symbol and length, at
/// consecutive columns of one row.
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

/// A fabric with no wafer: a step's rows as engine-cpu's forward over the checkpoint makes the
/// values the program returns ([`Reading::out_values`]), laid into each [`Back`]'s words as the
/// program's roots would. What it is written is kept (`written`, by array), not run.
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
                // (one step: its rows as they are, a grid reading's its voxels)
                // (one request on a program of one: its rows as they are, a grid reading's its
                // voxels; else request `tr / per`, a padded one the last's)
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

/// A denoise loop on the host around a device's steps: the poem's flow sigmas, a step's
/// timestep `sigma · train_steps`, an Euler step a velocity.
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
    /// `steps` steps of the schedule the poem states for `model` (its generative facts) over
    /// `image_rows` rows; the latent, timestep and velocity named as the poem names them.
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

    /// Runs the loop from `inputs` (the latent at the first sigma, and every other input):
    /// `velocity(i, inputs)` is step `i`'s velocity. Returns the final latent and each step's
    /// velocity.
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
}
