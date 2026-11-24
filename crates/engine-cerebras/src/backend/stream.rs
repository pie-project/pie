//! A stream reading's fires: a diffusion transformer's denoise step on the wafer program
//! `wafer-compile stream` made, driven by the guest's passes as any engine's. A step's lanes are
//! the reading's classes; each lane's inputs come off the channels its attachment binds, the
//! entries are the host forward's and the exit's ops run over the rows read back. The denoise
//! loop is the guest's; the engine runs a step a fire.

use std::collections::BTreeMap;
use std::sync::{Arc, mpsc};

use engine::error::{Error, Result as EngineResult};
use engine::fire::{LaneReadout, Readout};
use engine_cpu::Forward;
use engine_cpu::forward::{StreamLane, StreamStep};
use eta_exec::Value;
use eta_ir::registry::Stage;
use poem_ir::{Def, InputAxis, RuntimeInput, Trace};

use super::CerebrasEngine;
use crate::guest::refusal;
use crate::reading::{Fabric, Reading};

type R<T> = std::result::Result<T, String>;

type Job = Box<dyn FnOnce(&mut dyn Fabric) + Send>;

/// A [`Fabric`] on a thread of its own (the SDK wants every runtime call on the thread that made
/// it).
pub(crate) struct Remote {
    jobs: Option<mpsc::Sender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Remote {
    /// The fabric `open` makes, on its thread.
    pub(crate) fn open(open: impl FnOnce() -> R<Box<dyn Fabric>> + Send + 'static) -> R<Remote> {
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::channel::<R<()>>();
        let thread = std::thread::Builder::new()
            .name("spatial-stream".into())
            .spawn(move || {
                let mut fabric = match open() {
                    Ok(f) => {
                        let _ = ready_tx.send(Ok(()));
                        f
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                while let Ok(job) = job_rx.recv() {
                    job(fabric.as_mut());
                }
            })
            .map_err(|e| e.to_string())?;
        ready_rx
            .recv()
            .map_err(|_| "the stream fabric's thread ended".to_string())??;
        Ok(Remote {
            jobs: Some(job_tx),
            thread: Some(thread),
        })
    }

    /// `f` run on the fabric, waited for.
    pub(crate) fn call<T: Send + 'static>(
        &self,
        f: impl FnOnce(&mut dyn Fabric) -> R<T> + Send + 'static,
    ) -> R<T> {
        let (tx, rx) = mpsc::channel();
        let job: Job = Box::new(move |fabric| {
            let _ = tx.send(f(fabric));
        });
        self.jobs
            .as_ref()
            .ok_or("the stream fabric is closed")?
            .send(job)
            .map_err(|_| "the stream fabric's thread ended".to_string())?;
        rx.recv()
            .map_err(|_| "the stream fabric's thread ended".to_string())?
    }
}

impl Drop for Remote {
    fn drop(&mut self) {
        self.jobs = None;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The most words a static array's run written at once holds (1 GiB).
const STATIC_RUN: usize = 1 << 28;

/// A loaded stream reading: its program's host side and link.
pub(crate) struct Stream {
    pub(crate) reading: Arc<Reading>,
    /// The host forward of the entries' and the exit's ops (holding only the weights those read).
    host: Forward,
    /// cslc's output for the SDK runtime.
    link: std::path::PathBuf,
    /// A grid program in weight groups: its static arrays' words, made at load. Every other
    /// program's are made as they are written ([`Streams::write`]).
    statics: Arc<Vec<Option<Vec<u32>>>>,
    /// Its static arrays are on the program open.
    written: bool,
    steps: u64,
}

/// A load's stream readings (text encoder, denoiser, VAE), each its own program, and the fabric
/// they run on.
pub(crate) struct Streams {
    pub(crate) trace: Trace,
    pub(crate) readings: Vec<Stream>,
    /// The fabric (the host's, or the SDK runtime's programs, one at a time), on its own thread.
    fabric: Remote,
    host_fabric: bool,
    /// The program open (the SDK's link), every reading's static arrays on it written.
    open: Option<std::path::PathBuf>,
    /// The checkpoint's weights the static arrays are made from each time a program opens.
    weights: Forward,
}

impl Streams {
    /// The reading whose classes run what `words` run, lane for lane.
    fn pick(&self, words: &[u64]) -> Option<usize> {
        let same = |a: u64, b: u64| {
            a == b
                || self
                    .trace
                    .nodes
                    .iter()
                    .all(|n| n.guard.holds(a) == n.guard.holds(b))
        };
        self.readings.iter().position(|s| {
            s.reading.classes.len() == words.len()
                && s.reading
                    .classes
                    .iter()
                    .zip(words)
                    .all(|(c, w)| same(c.word, *w))
        })
    }

    /// The fabric reading `ri` runs on, its program open and its static arrays written.
    fn fabric_for(&mut self, ri: usize) -> R<&Remote> {
        // the host fabric keeps each reading's arrays apart; the SDK's opens a program at a time
        // (co-resident readings share one)
        if self.host_fabric {
            if !self.readings[ri].written {
                self.write(ri)?;
            }
        } else {
            let link = self.readings[ri].link.clone();
            if self.open.as_ref() != Some(&link) {
                self.open_link(&link)?;
            }
        }
        Ok(&self.fabric)
    }

    /// Opens the program `link` and writes the static arrays of every reading on it at once, so a
    /// reading's first launch does not wait on its arrays still landing.
    fn open_link(&mut self, link: &std::path::Path) -> R<()> {
        self.open = None;
        let l = link.to_path_buf();
        // a program loaded afresh holds none of its readings' arrays
        let fresh = self.fabric.call(move |f| f.switch(&l))?;
        for ri in 0..self.readings.len() {
            if self.readings[ri].link == link && (fresh || !self.readings[ri].written) {
                self.write(ri)?;
            }
        }
        self.open = Some(link.to_path_buf());
        Ok(())
    }

    /// Writes reading `ri`'s static arrays: a weight-grouped program's as made at load, any other's
    /// a run of rows at a time ([`Reading::statics_streamed`]).
    fn write(&mut self, ri: usize) -> R<()> {
        let reading = Arc::clone(&self.readings[ri].reading);
        if reading.groups > 0 {
            let words = Arc::clone(&self.readings[ri].statics);
            self.fabric
                .call(move |f| reading.write_statics(f, &words))?;
        } else {
            let step = reading.step(&self.trace, None, &[]);
            let fabric = &self.fabric;
            reading.statics_streamed(&mut self.weights, &step, STATIC_RUN, &mut |a, w| {
                fabric.call(move |f| f.write(&a, &w))
            })?;
        }
        self.readings[ri].written = true;
        Ok(())
    }
}

/// A step's lanes and inputs, owned (sent to the fabric's thread).
#[derive(Clone)]
struct Owned {
    lanes: Vec<StreamLane>,
    inputs: Vec<(String, Vec<f32>)>,
    tokens: Vec<u32>,
}

impl Owned {
    fn step(&self) -> StreamStep<'_> {
        StreamStep {
            lanes: self.lanes.clone(),
            inputs: self
                .inputs
                .iter()
                .map(|(n, v)| (n.as_str(), v.clone()))
                .collect(),
            tokens: self.tokens.clone(),
        }
    }
}

impl Streams {
    /// Guest input `id`'s axis and width, as the trace reads it.
    fn input_shape(&self, id: u32) -> Option<(InputAxis, usize)> {
        self.trace.values.iter().find_map(|d| match d.def {
            Def::Input(RuntimeInput::Guest {
                id: i, axis, width, ..
            }) if i == id => Some((axis, width as usize)),
            _ => None,
        })
    }
}

impl CerebrasEngine {
    /// One step of the stream reading: the prologues, the lanes' inputs off their channels, the
    /// wafer's step, the epilogues over its outputs, the lanes' readouts.
    #[allow(clippy::too_many_lines)]
    pub(super) fn fire_stream(
        &mut self,
        step: &engine::fire::Step,
    ) -> EngineResult<Vec<LaneReadout>> {
        let t0 = std::time::Instant::now();
        self.next_fire = self.next_fire.wrapping_add(1);
        for attachment in &step.attachments {
            if !attachment.adapter_only && self.programs.prologue_runs(attachment.instance) {
                self.programs
                    .fire_interpreted(attachment.instance, &[], Stage::Prologue)
                    .map_err(refusal)?;
            }
        }
        let Some(streams) = self.stream.as_mut() else {
            return Err(Error::Load("no stream reading is loaded".into()));
        };
        let invalid = |why: String| Error::Invalid(why);
        let words: Vec<u64> = step.lanes.iter().map(|l| l.word).collect();
        // the step's requests: runs of lanes of one group (lanes of no group one request)
        let mut requests: Vec<std::ops::Range<usize>> = Vec::new();
        let mut at0 = 0;
        while at0 < step.lanes.len() {
            let g = step.lanes[at0].group;
            let mut end = at0 + 1;
            while end < step.lanes.len() && step.lanes[end].group == g {
                end += 1;
            }
            requests.push(at0..end);
            at0 = end;
        }
        let Some(ri) = requests
            .first()
            .and_then(|r| streams.pick(&words[r.clone()]))
        else {
            return Err(invalid(format!(
                "no stream program runs lanes of words {words:x?}: the programs' are {:?}",
                (streams.readings.iter())
                    .map(|s| (
                        &s.reading.reading,
                        s.reading.classes.iter().map(|c| c.word).collect::<Vec<_>>()
                    ))
                    .collect::<Vec<_>>()
            )));
        };
        if requests[1..]
            .iter()
            .any(|r| streams.pick(&words[r.clone()]) != Some(ri))
        {
            return Err(invalid(format!(
                "a step's groups read different readings: lanes of words {words:x?}"
            )));
        }
        let reading = Arc::clone(&streams.readings[ri].reading);
        let classes = &reading.classes;
        let mut clips: BTreeMap<usize, Vec<[u32; 3]>> = BTreeMap::new();
        for v in &step.voxels {
            clips.insert(v.lane as usize, v.clips.clone());
        }
        // Each lane's inputs off the channels its attachment binds.
        let attached: BTreeMap<usize, engine::program::InstanceId> = step
            .attachments
            .iter()
            .filter(|a| !a.adapter_only)
            .flat_map(|a| {
                (0..a.lanes.max(1)).map(move |k| (a.lane as usize + k as usize, a.instance))
            })
            .collect();
        // each request's step: a lane may bring fewer rows than its class's, the rest padding (a
        // causal reading's last token again; another's zero rows); the outputs its own rows alone
        let mut owned: Vec<Owned> = Vec::with_capacity(requests.len());
        let mut lanes: Vec<StreamLane> = Vec::with_capacity(step.lanes.len());
        let mut own: Vec<usize> = Vec::with_capacity(step.lanes.len());
        for range in &requests {
            let mut rlanes = Vec::with_capacity(classes.len());
            let mut tokens: Vec<u32> = Vec::new();
            for (at, lane) in step.lanes[range.clone()].iter().enumerate() {
                let (at, class) = (range.start + at, &classes[at]);
                // (a grid reading's rows, and a lane's clip's, are its voxels: never short)
                let short = lane.tokens.len() < class.rows as usize
                    && !reading.host_only
                    && reading.clip.is_none()
                    && (reading.causal || clips.get(&at).is_none_or(Vec::is_empty));
                own.push(lane.tokens.len());
                tokens.extend_from_slice(&lane.tokens);
                if short {
                    let pad = if reading.causal {
                        lane.tokens.last().copied().unwrap_or(0)
                    } else {
                        0
                    };
                    tokens.resize(tokens.len() + class.rows as usize - lane.tokens.len(), pad);
                }
                // (a grid reading's rows are its clip's voxels: its lane's tokens are none of them)
                if !reading.host_only
                    && reading.clip.is_none()
                    && lane.tokens.len() != class.rows as usize
                    && !short
                {
                    return Err(invalid(format!(
                        "lane {at} is word {:#x} of {} rows; the program's lane is `{}` (word \
                         {:#x}) of {} rows",
                        lane.word,
                        lane.tokens.len(),
                        class.name,
                        class.word,
                        class.rows
                    )));
                }
                let clip = match clips.get(&at).map(Vec::as_slice) {
                    None | Some([]) => None,
                    Some([c]) => Some(*c),
                    Some(more) => {
                        return Err(invalid(format!(
                            "lane {at} carries {} clips; the program's lane holds one",
                            more.len()
                        )));
                    }
                };
                if let (Some(want), Some(got)) = (reading.clip, clip)
                    && want != got
                {
                    return Err(invalid(format!(
                        "lane {at}'s clip is {got:?}; the `{}` program is compiled for {want:?}",
                        reading.reading
                    )));
                }
                rlanes.push(StreamLane {
                    word: lane.word,
                    rows: if reading.host_only {
                        lane.tokens.len()
                    } else {
                        class.rows as usize
                    },
                    clip,
                });
            }
            let n_rows: usize = rlanes.iter().map(|l| l.rows).sum();
            let n_voxels: usize = rlanes
                .iter()
                .filter_map(|l| l.clip)
                .map(|[t, h, w]| (t * h * w) as usize)
                .sum();
            let mut inputs: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
            let (mut row0, mut vox0) = (0usize, 0usize);
            for (k, lane) in step.lanes[range.clone()].iter().enumerate() {
                let at = range.start + k;
                for feed in &lane.inputs {
                    let (axis, width) = streams.input_shape(feed.id).ok_or_else(|| {
                        invalid(format!(
                            "lane {at} feeds input {}, which the trace has not",
                            feed.id
                        ))
                    })?;
                    let instance = attached.get(&at).ok_or_else(|| {
                        invalid(format!(
                            "lane {at} feeds inputs and no program is attached to it"
                        ))
                    })?;
                    let cell = match self
                        .programs
                        .feed_cell(*instance, feed.channel)
                        .map_err(refusal)?
                    {
                        Value::F32(v) => v,
                        Value::I32(v) => v.into_iter().map(|x| x as f32).collect(),
                        Value::U32(v) => v.into_iter().map(|x| x as f32).collect(),
                        // bf16 widens exactly: its bits are an f32's high half
                        Value::Bf16(v) => {
                            v.into_iter().map(|x| f32::from_bits(u32::from(x) << 16)).collect()
                        }
                        Value::Bool(_) => {
                            return Err(invalid(format!("input {}: a boolean cell", feed.id)));
                        }
                    };
                    // (a short lane's row inputs its own rows, zero rows after them)
                    let (len, off, rows) = match axis {
                        InputAxis::Lane => (classes.len() * width, k * width, 1),
                        InputAxis::Row => (
                            n_rows * width,
                            row0 * width,
                            rlanes[k].rows.min(own[own.len() - rlanes.len() + k]),
                        ),
                        InputAxis::Voxel => {
                            let [t, h, w] = rlanes[k].clip.unwrap_or([0; 3]);
                            (n_voxels * width, vox0 * width, (t * h * w) as usize)
                        }
                    };
                    if cell.len() < rows * width {
                        return Err(invalid(format!(
                            "lane {at}'s input `{}`: a cell of {} values for {rows} rows of {width}",
                            streams.trace.inputs[feed.id as usize].name,
                            cell.len()
                        )));
                    }
                    let whole = inputs.entry(feed.id).or_insert_with(|| vec![0.0; len]);
                    whole[off..off + rows * width].copy_from_slice(&cell[..rows * width]);
                }
                row0 += rlanes[k].rows;
                if let Some([t, h, w]) = rlanes[k].clip {
                    vox0 += (t * h * w) as usize;
                }
            }
            lanes.extend(rlanes.iter().cloned());
            owned.push(Owned {
                lanes: rlanes,
                tokens,
                inputs: inputs
                    .into_iter()
                    .map(|(id, v)| (streams.trace.inputs[id as usize].name.clone(), v))
                    .collect(),
            });
        }
        let device = |e: String| Error::Device(e);
        // each request's rows back: a launch a batch of the program's requests (the last batch
        // padded with its last request); a reading with no program runs every op in `finish`
        let batch = reading.batch.max(1);
        let mut backs: Vec<Vec<engine_cpu::forward::Captured>> = Vec::new();
        let mut cycles: Option<u64> = None;
        if reading.host_only {
            backs.resize(owned.len(), Vec::new());
        } else {
            for chunk in owned.chunks(batch) {
                // (a short batch: its last request's values stand for the rest, made once)
                let sent: Vec<Owned> = chunk.to_vec();
                let views: Vec<StreamStep<'_>> = sent.iter().map(Owned::step).collect();
                let entries = if batch > 1 {
                    reading.entries_batch(&mut streams.readings[ri].host, &views)
                } else {
                    reading.entries(&mut streams.readings[ri].host, &views[0])
                }
                .map_err(device)?;
                drop(views);
                // (a grid program in weight groups writes its groups' tiles a launch each)
                let words = (reading.groups > 0).then(|| Arc::clone(&streams.readings[ri].statics));
                let fabric = streams.fabric_for(ri).map_err(device)?;
                let run = Arc::clone(&reading);
                let (got, c) = fabric
                    .call(move |f| {
                        let views: Vec<StreamStep<'_>> = sent.iter().map(Owned::step).collect();
                        run.launch_batch(
                            f,
                            &views,
                            &entries,
                            words.as_deref().map_or(&[], |w| &w[..]),
                        )
                    })
                    .map_err(device)?;
                if let Some(c) = c {
                    cycles = Some(cycles.unwrap_or(0) + c);
                }
                let back = reading.rows_back(&got).map_err(device)?;
                for b in 0..chunk.len() {
                    backs.push(if batch > 1 {
                        reading.request_of(&back, b)
                    } else {
                        back.clone()
                    });
                }
            }
        }
        // the outputs the attachments read, by name
        let mut names: Vec<String> = Vec::new();
        for a in step.attachments.iter().filter(|a| !a.adapter_only) {
            for (name, _) in self.programs.outputs_of(a.instance) {
                if !names.contains(&name) {
                    names.push(name);
                }
            }
        }
        // each name's output the step's lanes write (two readings may each write one of a name)
        let trace = &streams.trace;
        let written = |v: poem_ir::ValueId| match &trace.values[v.0 as usize].def {
            Def::Op(n) => words
                .iter()
                .any(|&w| trace.nodes[*n as usize].guard.holds(w)),
            Def::Merge(arms) => arms.iter().any(|(_, g)| words.iter().any(|&w| g.holds(w))),
            _ => false,
        };
        let mut decls: Vec<(String, InputAxis, usize)> = Vec::new();
        let mut wanted: Vec<poem_ir::ValueId> = Vec::new();
        for o in trace.outputs.iter().filter(|o| names.contains(&o.name)) {
            if !written(o.value) || decls.iter().any(|d| d.0 == o.name) {
                continue;
            }
            let (rows_axis, width) = output_shape(trace, o.value);
            decls.push((o.name.clone(), rows_axis, width));
            wanted.push(o.value);
        }
        let stream = &mut streams.readings[ri];
        // each request's outputs (a causal lane's padding dropped), the requests' in turn
        let mut made: Vec<Vec<f32>> = vec![Vec::new(); wanted.len()];
        let mut lane0 = 0;
        for (o, back) in owned.iter().zip(&backs) {
            let got = reading
                .finish_values(&mut stream.host, &o.step(), back, &wanted)
                .map_err(device)?;
            let mine = &own[lane0..lane0 + o.lanes.len()];
            lane0 += o.lanes.len();
            for ((v, (_, axis, width)), all) in got.into_iter().zip(&decls).zip(&mut made) {
                if *axis != InputAxis::Row || mine.iter().zip(&o.lanes).all(|(n, l)| *n == l.rows) {
                    all.extend(v);
                    continue;
                }
                let mut r0 = 0;
                for (n, l) in mine.iter().zip(&o.lanes) {
                    all.extend_from_slice(&v[r0 * width..(r0 + n) * width]);
                    r0 += l.rows;
                }
            }
        }
        // a lane's rows of an output: its tokens', voxels' or own row
        let lanes = &lanes;
        tracing::debug!(
            reading = %reading.reading,
            clips = ?lanes.iter().map(|l| l.clip).collect::<Vec<_>>(),
            outputs = ?made.iter().map(Vec::len).collect::<Vec<_>>(),
            decls = ?decls,
            "spatial: a stream step's outputs"
        );
        let voxels_before = |k: usize| -> usize {
            lanes[..k]
                .iter()
                .filter_map(|l| l.clip)
                .map(|[t, h, w]| (t * h * w) as usize)
                .sum()
        };
        // an output on a voxel grid of its own (a decoder's pixels): each lane's share of it
        let span =
            |axis: InputAxis, width: usize, i: usize, from: usize, to: usize| -> (usize, usize) {
                let at = |k: usize| -> usize {
                    match axis {
                        InputAxis::Lane => k,
                        InputAxis::Row => own[..k].iter().sum(),
                        InputAxis::Voxel => {
                            let (all, rows) =
                                (voxels_before(lanes.len()), made[i].len() / width.max(1));
                            (voxels_before(k) * rows).checked_div(all).unwrap_or(0)
                        }
                    }
                };
                (at(from), at(to))
            };
        for a in step.attachments.iter().filter(|a| !a.adapter_only) {
            let mut outputs: Vec<(String, u32, Value)> = Vec::new();
            for (name, k) in self.programs.outputs_of(a.instance) {
                let Some(i) = decls.iter().position(|d| d.0 == name) else {
                    return Err(invalid(format!(
                        "the program reads `{name}`, which the `{}` reading does not write",
                        reading.reading
                    )));
                };
                let (_, axis, width) = decls[i];
                let lane = a.lane as usize + k as usize;
                let end = (a.lane + a.lanes.max(1)) as usize;
                let (r0, r1) = span(axis, width, i, lane.min(lanes.len()), end.min(lanes.len()));
                outputs.push((
                    name,
                    k,
                    Value::F32(made[i][r0 * width..r1 * width].to_vec()),
                ));
            }
            self.programs
                .fire_interpreted(a.instance, &outputs, Stage::Epilogue)
                .map_err(refusal)?;
        }
        stream.steps += 1;
        self.fires += 1;
        tracing::info!(
            steps = stream.steps,
            cycles = ?cycles,
            fire_ms = t0.elapsed().as_secs_f64() * 1e3,
            "spatial: a stream step"
        );
        // each lane's readout: the reading's first output, the lane's rows of it
        let first = decls.first().cloned();
        Ok(step
            .lanes
            .iter()
            .enumerate()
            .map(|(at, lane)| {
                let Some((name, axis, width)) = &first else {
                    return LaneReadout::default();
                };
                if matches!(lane.readout, Readout::None) {
                    return LaneReadout::default();
                }
                let (r0, r1) = span(*axis, *width, 0, at, at + 1);
                if r1 == r0 {
                    return LaneReadout::default();
                }
                LaneReadout {
                    output: name.clone(),
                    rows: u32::try_from(r1 - r0).unwrap_or(u32::MAX),
                    width: u32::try_from(*width).unwrap_or(u32::MAX),
                    values: made[0][r0 * width..r1 * width].to_vec(),
                    clips: lanes[at].clip.into_iter().collect(),
                }
            })
            .collect())
    }
}

/// The stream programs in `dir`: its own (linked in `link`, else `<dir>/out`), or one a reading
/// in its subdirectories, each linked in its `out`; `None` where it holds neither.
pub(crate) fn programs_in(
    dir: &std::path::Path,
    link: Option<&std::path::Path>,
) -> Option<Vec<(std::path::PathBuf, std::path::PathBuf)>> {
    if dir.join("host.json").is_file() {
        let link = link.map_or_else(|| dir.join("out"), std::path::Path::to_path_buf);
        return Some(vec![(dir.to_path_buf(), link)]);
    }
    let mut out: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("host.json").is_file())
        .map(|p| (p.clone(), p.join("out")))
        .collect();
    out.sort();
    (!out.is_empty()).then_some(out)
}

/// The axis a value's rows run over and its width.
fn output_shape(trace: &Trace, v: poem_ir::ValueId) -> (InputAxis, usize) {
    use poem_ir::{Dim, Ty};
    match &trace.values[v.0 as usize].ty {
        Ty::Tensor { shape, .. } => {
            let axis = match shape.first() {
                Some(Dim::Voxels | Dim::VoxelsScaled(..)) => InputAxis::Voxel,
                Some(Dim::Lanes | Dim::LanesPlus(_)) => InputAxis::Lane,
                _ => InputAxis::Row,
            };
            let width = (shape.iter().skip(1))
                .map(|d| match d {
                    Dim::Const(n) => *n as usize,
                    _ => 1,
                })
                .product();
            (axis, width)
        }
        Ty::Struct(_) => (InputAxis::Row, 0),
    }
}

impl CerebrasEngine {
    /// Loads a model's stream readings (`programs`: each compile's output and link): each reading's
    /// host side, the checkpoint landed for their static arrays, and the fabric.
    pub(super) fn load_stream(
        &mut self,
        trace: poem_ir::Trace,
        checkpoint: &std::path::Path,
        contract: &checkpoint::contract::ModelContract,
        budgets: &engine::load::Budgets,
        steps: &engine::StepRules,
        programs: &[(std::path::PathBuf, std::path::PathBuf)],
    ) -> EngineResult<engine::load::Loaded> {
        use engine::caps::{Capabilities, FireLimits, PoolFacts};
        use eta_ir::registry::{GeometryClass, PortMask};
        let load = |e: String| Error::Load(e);
        let t0 = std::time::Instant::now();
        let landed = crate::prepare::land(&trace, contract, checkpoint).map_err(load)?;
        let of = |name: &str, count: usize| crate::prepare::plane(&landed, name, count);
        let mut weights = Forward::new(trace.clone(), None, &of).map_err(load)?;
        tracing::info!(
            s = t0.elapsed().as_secs_f64(),
            "spatial: the checkpoint landed for a stream program"
        );
        let mut readings = Vec::new();
        for (dir, link) in programs {
            let reading = Reading::open(dir).map_err(load)?;
            if reading.model != trace.name {
                return Err(load(format!(
                    "the stream program in {} is `{}`'s; this load is `{}`",
                    dir.display(),
                    reading.model,
                    trace.name
                )));
            }
            let host =
                Forward::host_for(trace.clone(), &of, &reading.entry_values()).map_err(load)?;
            // the static arrays are the weights' alone: made from a step of no inputs
            let statics = if reading.groups > 0 {
                let s0 = std::time::Instant::now();
                let st = reading
                    .statics(&mut weights, &reading.step(&trace, None, &[]))
                    .map_err(load)?;
                tracing::info!(
                    s = s0.elapsed().as_secs_f64(),
                    reading = %reading.reading,
                    "spatial: a stream program's static arrays made from the checkpoint"
                );
                st
            } else {
                Vec::new()
            };
            readings.push(Stream {
                reading: Arc::new(reading),
                host,
                link: link.clone(),
                statics: Arc::new(statics),
                written: false,
                steps: 0,
            });
        }
        let host_fabric = self.debug.stream_fabric.as_deref() == Some("host");
        // On the host fabric, a reading no program runs (e.g. a VAE's convolutions) is served whole
        // by the host's forward: a debug fallback.
        if host_fabric {
            let served: Vec<String> = readings.iter().map(|s| s.reading.reading.clone()).collect();
            let all: Vec<poem_ir::ValueId> = trace.outputs.iter().map(|o| o.value).collect();
            for r in trace
                .facts
                .declared("reading")
                .map(|d| d.values.clone())
                .unwrap_or_default()
            {
                if served.contains(&r) {
                    continue;
                }
                let classes = poem_compiler::spatial::layer::stream::classes(&trace, &r, 0);
                tracing::warn!(reading = %r, "spatial: a reading no stream program runs: the host fabric's forward serves it whole");
                readings.push(Stream {
                    reading: Arc::new(Reading::host_only(&trace.name, &r, &classes)),
                    host: Forward::host_for(trace.clone(), &of, &all).map_err(load)?,
                    link: std::path::PathBuf::new(),
                    statics: Arc::new(Vec::new()),
                    written: false,
                    steps: 0,
                });
            }
        }
        let fabric = if host_fabric {
            let forward = Forward::new(trace.clone(), None, &of).map_err(load)?;
            Remote::open(move || Ok(Box::new(crate::reading::HostFabric::new(forward)) as _))
        } else {
            let (sdk_lib, platform) = (self.cfg.toolchain.sdk_lib(), self.platform()?);
            Remote::open(move || {
                Ok(Box::new(crate::reading::Programs {
                    sdk_lib,
                    platform,
                    rt: None,
                }) as _)
            })
        }
        .map_err(load)?;
        drop(landed);
        // a fire holds a launch's batch of requests (at least the budgets', so a larger stated one
        // stands)
        let held = |f: &dyn Fn(&Reading) -> usize| {
            readings
                .iter()
                .map(|s: &Stream| s.reading.batch.max(1) * f(&s.reading))
                .max()
                .unwrap_or(0)
        };
        let max_lanes = budgets
            .max_lanes
            .max(u32::try_from(held(&|r| r.classes.len())).unwrap_or(u32::MAX))
            .max(1);
        let max_tokens = budgets
            .max_tokens
            .max(u32::try_from(held(&Reading::request_rows)).unwrap_or(u32::MAX));
        let slots =
            poem_compiler::state_slots(budgets.slots, &trace, super::default_state_slots(budgets));
        let facts = crate::facts::Facts::of(trace.clone(), budgets, slots).map_err(load)?;
        let paging = facts.paging();
        // One shared pool on the host, kv pages of tokens alone: a stream step holds no cache.
        let chunks = super::pool::Pool {
            page_size: paging.page_size,
            kv_pages: paging.pages(),
            records: 0,
            state_bytes: 0,
            windowed: Vec::new(),
            buffer_pages: 0,
            wafer: None,
        }
        .facts();
        chunks
            .validate()
            .map_err(|e| load(format!("the stream load's pool: {e}")))?;
        let caps = Capabilities {
            device: super::device_facts(),
            pools: PoolFacts::default(),
            chunks,
            limits: FireLimits {
                max_lanes,
                max_tokens,
                max_page_refs: paging.pages_per_slot,
                max_context: paging.context(),
                prefill_rows: 0,
                prefill_lanes: 0,
                max_readouts: max_tokens,
                replayed_rows: 0,
                dispatch_depth: 0,
            },
            has_lora: false,
            has_attn_page_mask: false,
            draft_block: 0,
            draft_mask_token: 0,
            draft_proposals_from: 1,
            ports: PortMask::DEVICE_GEOMETRY,
            geometry: GeometryClass::DeviceGeometry,
            kv_handle: None,
            media_encode: false,
            rs_verbs: false,
            windowed_state: false,
            rs_snapshots: 0,
            // a stream step's lanes attend each other's rows
            bidirectional_attention: true,
            adapter_attachments: false,
            facts: facts.trace().facts.clone(),
            inputs: facts.input_decls(),
            classes: facts.classes(),
            // Cerebras states no bound on one submission.
            guest_cost: engine::GuestCost::default(),
            host_ops: engine::HostOps::NONE,
            mixed_lanes: engine::MixedLanes::Any,
            // one frame at a time: no run-ahead
            frames_in_flight: 1,
            steps: engine::StatedSteps::default(),
        };
        // The step compositions the runtime's builder forms over this load's
        // facts (CONTRACT-BC 24), as a decode load states them: a stream
        // load records no bodies either.
        let mut caps = caps;
        caps.steps = engine::StatedSteps::of(
            steps,
            &engine::StepFacts::stated(&caps, &engine::ReadoutFacts::default()),
        );
        for s in &readings {
            tracing::info!(
                program = %s.reading.dir.display(),
                reading = %s.reading.reading,
                classes = ?s.reading.classes.iter().map(|c| (&c.name, c.rows)).collect::<Vec<_>>(),
                host_fabric,
                "spatial: a stream reading loaded"
            );
        }
        let trace_name = trace.name.clone();
        let mut streams = Streams {
            trace,
            readings,
            fabric,
            host_fabric,
            open: None,
            weights,
        };
        // open the wafer's program now, so a reading's first step neither loads it nor waits on its
        // arrays
        if !host_fabric && let Some(link) = streams.readings.first().map(|s| s.link.clone()) {
            let s0 = std::time::Instant::now();
            streams.open_link(&link).map_err(load)?;
            tracing::info!(
                s = s0.elapsed().as_secs_f64(),
                link = %link.display(),
                "spatial: a stream program opened at load, its static arrays written"
            );
        }
        self.stream = Some(streams);
        self.facts = Some(facts);
        self.caps = Some(caps.clone());
        Ok(engine::load::Loaded {
            facts: engine::load::LoadFacts {
                trace_name,
                weight_bytes: 0,
                weights_resident: true,
                weights_from_cache: false,
                arena_bytes: 0,
                pool_bytes: 0,
                input_bytes: 0,
                pool_committed_bytes: 0,
                pool_high_water_bytes: 0,
            },
            caps,
        })
    }
}
