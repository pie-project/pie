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
//! - between steps, the poem's schedule ([`Denoise`]: its flow sigmas, an Euler step a velocity).
//!
//! Nothing here names a model's op: the values are the trace's, by id, and the runner is the
//! same for any stream reading the compile states.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use engine_cpu::Forward;
use engine_cpu::forward::{Captured, StreamLane, StreamStep, Values};
use kernels_cerebras::stream::host::{Word, apply};
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

/// A memcpy rectangle the host writes: its words from `file`, or, for an entry, gathered from
/// the step's values each step.
#[derive(Clone, Debug)]
pub struct Array {
    pub sym: String,
    pub rect: Rect,
    pub len: usize,
    pub u16: bool,
    pub file: PathBuf,
    pub entry: Option<Vec<Word>>,
}

/// A root's rows read back: each of its own rows (the step's row `rows[i]`) the `parts`
/// `(value, from, len)` side by side.
#[derive(Clone, Debug)]
pub struct Back {
    pub sym: String,
    pub x: i32,
    pub y: i32,
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
                file: dir.join(text(&a, "file")?),
                entry,
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
        })
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
        }
    }

    /// Each array's words for `step`, computed on the host forward `host` (no weights needed):
    /// an entry's from the step's values, `None` for an array whose words are the file's.
    pub fn entries(&self, host: &mut Forward, step: &StreamStep<'_>) -> R<Vec<Option<Vec<f32>>>> {
        let mut wanted: Vec<ValueId> = Vec::new();
        for a in &self.h2d {
            for w in a.entry.iter().flatten() {
                let v = ValueId(w[0]);
                if w[0] != kernels_cerebras::stream::host::NONE && !wanted.contains(&v) {
                    wanted.push(v);
                }
            }
        }
        let held: BTreeMap<u32, (usize, usize, Rc<Vec<f32>>)> = host
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
        Ok(host
            .stream_values(step, given, &wanted)?
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

    /// One step on a loaded program: every array written (an entry's words from `entries`, the
    /// rest from its file), `run` launched, the rows read back (each [`Back`]'s words) and the
    /// step's cycles off the PEs' stamps (`tsc`: start, end, passes a PE).
    pub fn fire(
        &self,
        rt: &mut Runtime,
        entries: &[Option<Vec<f32>>],
    ) -> R<(Vec<Vec<f32>>, Option<u64>)> {
        let e = |x: crate::sdk::Error| x.to_string();
        for (a, words) in self.h2d.iter().zip(entries) {
            let data: Vec<u32> = match words {
                Some(w) => w.iter().map(|x| x.to_bits()).collect(),
                None => npy_words(&a.file)?.1,
            };
            let dt = if a.u16 {
                DataType::Bits16
            } else {
                DataType::Bits32
            };
            rt.memcpy_h2d(&a.sym, &data, a.rect, a.len, Order::RowMajor, dt)
                .map_err(e)?;
        }
        rt.launch("run", &[]).map_err(e)?;
        let mut got = Vec::new();
        for b in &self.d2h {
            let mut out = vec![0u32; b.len];
            let rect = Rect {
                x: b.x,
                y: b.y,
                w: 1,
                h: 1,
            };
            rt.memcpy_d2h(
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
        let cycles = if rt.rpc().symbols().contains_key("tsc") {
            let (w, h) = (self.w as usize, self.h as usize);
            let mut ts = vec![0u32; w * h * 3];
            let rect = Rect {
                x: 0,
                y: 0,
                w: w as i32,
                h: h as i32,
            };
            rt.memcpy_d2h("tsc", &mut ts, rect, 3, Order::RowMajor, DataType::Bits32)
                .map_err(e)?;
            let live = ts.chunks(3).filter(|s| s[0] != 0 || s[1] != 0);
            let (lo, hi) = live.fold((u64::MAX, 0u64), |(lo, hi), s| {
                (lo.min(u64::from(s[0])), hi.max(u64::from(s[1])))
            });
            (hi >= lo).then(|| hi - lo)
        } else {
            None
        };
        Ok((got, cycles))
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
