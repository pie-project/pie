//! A stream reading's fires (GENERIC.md M1g): a diffusion transformer's denoise step on the
//! wafer program `wafer-compile stream` made, driven by the guest's passes as any engine's.
//!
//! A step's lanes are the reading's classes (a lane a class, in the program's order and rows; a
//! media lane's clip its step's voxels). Each lane's guest inputs come off the channels its
//! attachment binds (the cell the instance's own `take` would read, after the prologues ran),
//! laid out as the trace's input axes do: a value a lane, a row a token, a row a voxel. The
//! entries are the host forward's ([`Reading::entries`]), the static arrays the checkpoint's
//! ([`Reading::statics`], written once a load, on the first step), and the exit's ops run over
//! the rows read back ([`Reading::finish`]); the epilogues read the outputs, a lane's rows and
//! the pass's later lanes' as for decode. The denoise loop is the guest's: its schedule, its
//! Euler steps and its channels; the engine runs a step a fire.

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

/// A [`Fabric`] on a thread of its own (the SDK wants every runtime call on the thread that
/// made it): jobs sent, their answers waited on.
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

/// A loaded stream reading.
pub(crate) struct Stream {
    pub(crate) reading: Arc<Reading>,
    pub(crate) trace: Trace,
    /// The weightless host forward: the entries' and the exit's ops.
    host: Forward,
    /// The forward over the checkpoint's planes the static arrays are made from, until they
    /// are written (the first step).
    weights: Option<Forward>,
    fabric: Remote,
    steps: u64,
}

/// A step's lanes and inputs, owned (sent to the fabric's thread).
struct Owned {
    lanes: Vec<StreamLane>,
    inputs: Vec<(String, Vec<f32>)>,
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
        }
    }
}

impl Stream {
    pub(crate) fn new(reading: Reading, trace: Trace, weights: Forward, fabric: Remote) -> Stream {
        Stream {
            reading: Arc::new(reading),
            host: Forward::host(trace.clone()),
            trace,
            weights: Some(weights),
            fabric,
            steps: 0,
        }
    }

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
        let Some(stream) = self.stream.as_mut() else {
            return Err(Error::Load("no stream reading is loaded".into()));
        };
        let classes = &stream.reading.classes;
        let invalid = |why: String| Error::Invalid(why);
        if step.lanes.len() != classes.len() {
            return Err(invalid(format!(
                "the wafer's `{}` program runs {} lanes a step (its classes {:?}); this step has {}",
                stream.reading.reading,
                classes.len(),
                classes.iter().map(|c| &c.name).collect::<Vec<_>>(),
                step.lanes.len()
            )));
        }
        let mut clips: BTreeMap<usize, Vec<[u32; 3]>> = BTreeMap::new();
        for v in &step.voxels {
            clips.insert(v.lane as usize, v.clips.clone());
        }
        let mut lanes = Vec::with_capacity(classes.len());
        for (at, (lane, class)) in step.lanes.iter().zip(classes).enumerate() {
            if lane.word != class.word || lane.tokens.len() != class.rows as usize {
                return Err(invalid(format!(
                    "lane {at} is word {:#x} of {} rows; the program's lane {at} is `{}` (word \
                     {:#x}) of {} rows",
                    lane.word,
                    lane.tokens.len(),
                    class.name,
                    class.word,
                    class.rows
                )));
            }
            if lane.group != step.lanes[0].group {
                return Err(invalid(
                    "a stream step's lanes attend each other: one group".into(),
                ));
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
            lanes.push(StreamLane {
                word: lane.word,
                rows: lane.tokens.len(),
                clip,
            });
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
        let n_rows: usize = lanes.iter().map(|l| l.rows).sum();
        let n_voxels: usize = lanes
            .iter()
            .filter_map(|l| l.clip)
            .map(|[t, h, w]| (t * h * w) as usize)
            .sum();
        let mut inputs: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
        let (mut row0, mut vox0) = (0usize, 0usize);
        for (at, lane) in step.lanes.iter().enumerate() {
            for feed in &lane.inputs {
                let (axis, width) = stream.input_shape(feed.id).ok_or_else(|| {
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
                    Value::Bool(_) => {
                        return Err(invalid(format!("input {}: a boolean cell", feed.id)));
                    }
                };
                let (len, off, rows) = match axis {
                    InputAxis::Lane => (classes.len() * width, at * width, 1),
                    InputAxis::Row => (n_rows * width, row0 * width, lanes[at].rows),
                    InputAxis::Voxel => {
                        let [t, h, w] = lanes[at].clip.unwrap_or([0; 3]);
                        (n_voxels * width, vox0 * width, (t * h * w) as usize)
                    }
                };
                if cell.len() < rows * width {
                    return Err(invalid(format!(
                        "lane {at}'s input `{}`: a cell of {} values for {rows} rows of {width}",
                        stream.trace.inputs[feed.id as usize].name,
                        cell.len()
                    )));
                }
                let whole = inputs.entry(feed.id).or_insert_with(|| vec![0.0; len]);
                whole[off..off + rows * width].copy_from_slice(&cell[..rows * width]);
            }
            row0 += lanes[at].rows;
            if let Some([t, h, w]) = lanes[at].clip {
                vox0 += (t * h * w) as usize;
            }
        }
        let owned = Owned {
            lanes,
            inputs: inputs
                .into_iter()
                .map(|(id, v)| (stream.trace.inputs[id as usize].name.clone(), v))
                .collect(),
        };
        let device = |e: String| Error::Device(e);
        let step_view = owned.step();
        let entries = stream
            .reading
            .entries(&mut stream.host, &step_view)
            .map_err(device)?;
        if let Some(mut weights) = stream.weights.take() {
            let s0 = std::time::Instant::now();
            let words = stream
                .reading
                .statics(&mut weights, &step_view)
                .map_err(device)?;
            let reading = Arc::clone(&stream.reading);
            stream
                .fabric
                .call(move |f| reading.write_statics(f, &words))
                .map_err(device)?;
            tracing::info!(
                s = s0.elapsed().as_secs_f64(),
                "spatial: a stream program's static arrays written from the checkpoint"
            );
        }
        let reading = Arc::clone(&stream.reading);
        let sent = Owned {
            lanes: owned.lanes.clone(),
            inputs: owned.inputs.clone(),
        };
        let (got, cycles) = stream
            .fabric
            .call(move |f| reading.launch_on(f, &sent.step(), &entries))
            .map_err(device)?;
        let back = stream.reading.rows_back(&got).map_err(device)?;
        // the outputs the attachments read, by name
        let mut names: Vec<String> = Vec::new();
        for a in step.attachments.iter().filter(|a| !a.adapter_only) {
            for (name, _) in self.programs.outputs_of(a.instance) {
                if !names.contains(&name) {
                    names.push(name);
                }
            }
        }
        let decls: Vec<(String, InputAxis, usize)> = stream
            .trace
            .outputs
            .iter()
            .filter(|o| names.contains(&o.name))
            .map(|o| {
                let (rows_axis, width) = output_shape(&stream.trace, o.value);
                (o.name.clone(), rows_axis, width)
            })
            .collect();
        let wanted: Vec<&str> = decls.iter().map(|d| d.0.as_str()).collect();
        let made = stream
            .reading
            .finish(&mut stream.host, &stream.trace, &step_view, &back, &wanted)
            .map_err(device)?;
        // a lane's rows of an output: its tokens', voxels' or own row
        let lanes = &owned.lanes;
        let span = |axis: InputAxis, from: usize, to: usize| -> (usize, usize) {
            let at = |k: usize| -> usize {
                match axis {
                    InputAxis::Lane => k,
                    InputAxis::Row => lanes[..k].iter().map(|l| l.rows).sum(),
                    InputAxis::Voxel => lanes[..k]
                        .iter()
                        .filter_map(|l| l.clip)
                        .map(|[t, h, w]| (t * h * w) as usize)
                        .sum(),
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
                        stream.reading.reading
                    )));
                };
                let (_, axis, width) = decls[i];
                let lane = a.lane as usize + k as usize;
                let end = (a.lane + a.lanes.max(1)) as usize;
                let (r0, r1) = span(axis, lane.min(lanes.len()), end.min(lanes.len()));
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
                let (r0, r1) = span(*axis, at, at + 1);
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

/// The axis a value's rows run over and its width.
fn output_shape(trace: &Trace, v: poem_ir::ValueId) -> (InputAxis, usize) {
    use poem_ir::{Dim, Ty};
    match &trace.values[v.0 as usize].ty {
        Ty::Tensor { shape, .. } => {
            let axis = match shape.first() {
                Some(Dim::Voxels) => InputAxis::Voxel,
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

/// A checkpoint plane's values as f32 (`landed` holds it bf16 or f32 as the contract lands it).
pub(crate) fn plane(landed: &crate::prepare::Landed, name: &str, count: usize) -> Vec<f32> {
    use kernels_cerebras::tiles::Params;
    let Some(p) = landed.get(name) else {
        return Vec::new();
    };
    let b = p.bytes;
    if b.len() == 2 * count {
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|h| f32::from_bits(u32::from(u16::from_le_bytes(*h)) << 16))
            .collect()
    } else if b.len() == 4 * count {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect()
    } else {
        Vec::new()
    }
}

impl CerebrasEngine {
    /// Loads a stream reading's program (`dir`: `wafer-compile stream`'s output, its link in
    /// `link`): the reading's host side, the checkpoint landed for its static arrays, the fabric
    /// opened. The facts the runtime is told are the trace's, as for decode.
    pub(super) fn load_stream(
        &mut self,
        trace: poem_ir::Trace,
        checkpoint: &std::path::Path,
        contract: &checkpoint::contract::ModelContract,
        budgets: &engine::load::Budgets,
        dir: &std::path::Path,
    ) -> EngineResult<engine::load::Loaded> {
        use engine::caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
        use engine::transfer::MemoryDomain;
        use eta_ir::registry::{GeometryClass, PortMask};
        let load = |e: String| Error::Load(e);
        let reading = Reading::open(dir).map_err(load)?;
        if reading.model != trace.name {
            return Err(load(format!(
                "the stream program in {} is `{}`'s; this load is `{}`",
                dir.display(),
                reading.model,
                trace.name
            )));
        }
        let t0 = std::time::Instant::now();
        let landed = crate::prepare::land(&trace, contract, checkpoint).map_err(load)?;
        let of = |name: &str, count: usize| plane(&landed, name, count);
        let weights = Forward::new(trace.clone(), None, &of).map_err(load)?;
        tracing::info!(
            s = t0.elapsed().as_secs_f64(),
            "spatial: the checkpoint landed for a stream program"
        );
        let host_fabric = self.debug.stream_fabric.as_deref() == Some("host");
        let fabric = if host_fabric {
            let forward = Forward::new(trace.clone(), None, &of).map_err(load)?;
            Remote::open(move || Ok(Box::new(crate::reading::HostFabric::new(forward)) as _))
        } else {
            let link = self.debug.link.clone().unwrap_or_else(|| dir.join("out"));
            let platform = self.platform();
            let sdk_lib = self.cfg.toolchain.sdk_lib();
            Remote::open(move || {
                let sdk = crate::sdk::Sdk::open_at(&sdk_lib).map_err(|e| e.to_string())?;
                let mut rt = crate::sdk::Runtime::new(sdk, &link, platform, "WARNING")
                    .map_err(|e| e.to_string())?;
                let t0 = std::time::Instant::now();
                rt.load();
                rt.run();
                tracing::info!(
                    s = t0.elapsed().as_secs_f64(),
                    "spatial: a stream program loaded"
                );
                Ok(Box::new(rt) as _)
            })
        }
        .map_err(load)?;
        drop(landed);
        let slots = poem_compiler::state_slots(budgets.slots, &trace, super::DEFAULT_STATE_SLOTS);
        let facts = crate::facts::Facts::of(trace.clone(), budgets, slots).map_err(load)?;
        let paging = facts.paging();
        let caps = Capabilities {
            device: DeviceFacts {
                backend: "cerebras".to_string(),
                domain: MemoryDomain::CerebrasDevice(0),
                sms: 1,
                unified_memory: false,
                fp8_native: false,
                native_mxfp4_moe: false,
                storage_alignment: 256,
                storage_max_tile_bytes: u64::MAX,
                codegen_backend: None,
            },
            pools: PoolFacts {
                kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
                kv_page_size: paging.page_size,
                ..PoolFacts::default()
            },
            limits: FireLimits {
                max_lanes: budgets.max_lanes.max(1),
                max_tokens: budgets.max_tokens,
                max_page_refs: paging.pages_per_slot,
                max_context: paging.context(),
                prefill_rows: 0,
                max_readouts: budgets.max_tokens,
            },
            has_lora: false,
            has_attn_page_mask: false,
            draft_block: 0,
            draft_mask_token: 0,
            draft_proposals_from: 1,
            ports: PortMask::DEVICE_GEOMETRY,
            geometry: GeometryClass::DeviceGeometry,
            kv_copy: KvCopyDomains::default(),
            kv_handle: None,
            media_encode: false,
            device_channel_commit: true,
            rs_verbs: false,
            rs_snapshots: 0,
            // a stream step's lanes attend each other's rows
            bidirectional_attention: true,
            adapter_attachments: false,
            facts: facts.trace().facts.clone(),
            inputs: facts.input_decls(),
            classes: facts.classes(),
        };
        tracing::info!(
            program = %dir.display(),
            reading = %reading.reading,
            classes = ?reading.classes.iter().map(|c| (&c.name, c.rows)).collect::<Vec<_>>(),
            host_fabric,
            "spatial: a stream reading loaded"
        );
        let trace_name = trace.name.clone();
        self.stream = Some(Stream::new(reading, trace, weights, fabric));
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
