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
pub trait Fabric {
    /// Writes array `a`'s words (each an f32's bits, or a 16-bit word).
    fn write(&mut self, a: &Array, words: &[u32]) -> R<()>;
    /// Runs `step` on the arrays written: the rows back (each [`Back`]'s words) and the step's
    /// cycles where the program stamps them.
    fn run(&mut self, reading: &Reading, step: &StreamStep<'_>) -> R<(Vec<Vec<f32>>, Option<u64>)>;
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
        let mut d2h = Vec::new();
        for o in list("d2h")? {
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
                            let p = p.as_array()?;
                            Some((
                                p.first()?.as_u64()? as u32,
                                p.get(1)?.as_u64()? as u32,
                                p.get(2)?.as_u64()? as u32,
                            ))
                        })
                        .collect()
                })
                .unwrap_or_default();
            d2h.push(Back {
                sym: text(&o, "sym")?,
                x: num(&o, "x")? as i32,
                y: num(&o, "y")? as i32,
                w: o["w"].as_i64().unwrap_or(1) as i32,
                h: o["h"].as_i64().unwrap_or(1) as i32,
                len: num(&o, "len")? as usize,
                rows: ints("rows").into_iter().map(|r| r as usize).collect(),
                parts,
            });
        }
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
            outs,
            w: num(&host, "w")? as u32,
            h: num(&host, "h")? as u32,
            causal: host["causal"].as_bool().unwrap_or(false),
            host_only: false,
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
            pos: (host["pos"].as_array().into_iter().flatten())
                .filter_map(|c| {
                    Some(Rect {
                        x: c["x"].as_i64()? as i32,
                        y: c["y"].as_i64()? as i32,
                        w: 1,
                        h: 1,
                    })
                })
                .collect(),
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
            outs: Vec::new(),
            refs: Vec::new(),
            w: 0,
            h: 0,
            causal: false,
            host_only: true,
            origin: (0, 0),
            rd: None,
            clip: None,
            groups: 0,
            pos: Vec::new(),
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
                    || s[0] == kernels_cerebras::stream::host::UNIT;
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

    /// The static arrays' words ([`Reading::statics`]) written, once a load.
    pub fn write_statics(&self, rt: &mut dyn Fabric, words: &[Option<Vec<u32>>]) -> R<()> {
        for (a, w) in self.h2d.iter().zip(words) {
            if a.group.is_some() {
                continue;
            }
            if let Some(w) = w {
                rt.write(a, w)?;
            }
        }
        Ok(())
    }

    /// One step on a loaded program: every array written (an entry's words from `entries`, the
    /// rest from its file), then [`Reading::launch`].
    /// A decode program's step (GENERIC.md G1): `tokens` (a lane's one each) at positions
    /// from `at`, the trace's positions given as theirs; the step and its entries.
    pub fn decode_step(
        &self,
        host: &mut Forward,
        trace: &Trace,
        tokens: &[u32],
        at: u32,
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
                values: Values::I32((0..tokens.len()).map(|i| (at as usize + i) as i32).collect()),
            })
            .collect();
        let entries = self.entries_given(host, &step, &given)?;
        Ok((step, entries))
    }

    /// The K / V PEs told a decode step's first position ([`Reading::pos`]).
    pub fn write_pos(&self, rt: &mut dyn Fabric, at: u32) -> R<()> {
        for r in &self.pos {
            let a = Array {
                sym: "pos".into(),
                rect: *r,
                len: 1,
                u16: false,
                file: None,
                entry: None,
                statics: None,
                group: None,
            };
            rt.write(&a, &[at])?;
        }
        Ok(())
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
        let (got, cycles) = rt.run(self, step)?;
        Ok((got, cycles.map(|c| c + before.unwrap_or(0))))
    }
}

impl Runtime {
    /// The launch's cycles off its PEs' stamps (`tsc`: start, end, passes a PE) over the
    /// reading's rectangle, where the program stamps them.
    fn stamps(&mut self, reading: &Reading) -> R<Option<u64>> {
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
        let live = ts.chunks(3).filter(|s| s[0] != 0 || s[1] != 0);
        let (lo, hi) = live.fold((u64::MAX, 0u64), |(lo, hi), s| {
            (lo.min(u64::from(s[0])), hi.max(u64::from(s[1])))
        });
        Ok((hi >= lo).then(|| hi - lo))
    }
}

impl Fabric for Runtime {
    fn launch(&mut self, reading: &Reading) -> R<Option<u64>> {
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
        self.memcpy_h2d(&a.sym, words, a.rect, a.len, Order::RowMajor, dt)
            .map_err(|x| x.to_string())
    }

    fn run(
        &mut self,
        reading: &Reading,
        _step: &StreamStep<'_>,
    ) -> R<(Vec<Vec<f32>>, Option<u64>)> {
        let e = |x: crate::sdk::Error| x.to_string();
        let args: Vec<u32> = reading.rd.into_iter().collect();
        self.launch("run", &args).map_err(e)?;
        let mut got = Vec::new();
        for b in &reading.d2h {
            let mut out = vec![0u32; b.len * (b.w * b.h) as usize];
            let rect = Rect {
                x: b.x,
                y: b.y,
                w: b.w,
                h: b.h,
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
            got.push(out.into_iter().map(f32::from_bits).collect());
        }
        let cycles = self.stamps(reading)?;
        Ok((got, cycles))
    }
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
        let made: BTreeMap<u32, Captured> = self
            .forward
            .stream_values(step, &[], &reading.out_values())?
            .into_iter()
            .map(|c| (c.value.0, c))
            .collect();
        let mut got = Vec::new();
        for b in &reading.d2h {
            let width: usize = b.parts.iter().map(|p| p.2 as usize).sum();
            let mut buf = vec![0f32; b.len * (b.w * b.h) as usize];
            for (i, &tr) in b.rows.iter().enumerate() {
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
