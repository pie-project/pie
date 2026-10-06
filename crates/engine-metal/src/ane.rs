//! Prefill on the GPU and the Neural Engine at once (`PIE_ANE`).
//!
//! A dense SwiGLU MLP is split along its intermediate axis: the GPU keeps
//! columns `[0, keep)` and the Neural Engine runs `[keep, inter)` as a CoreML
//! program that `scripts/ane/build.py` writes, one per layer. For each layer
//! the GPU stages the MLP's input as fp16 and signals a shared event; a host
//! thread runs the CoreML program over it and signals back; the GPU works on
//! its own columns meanwhile and waits on the event only where it adds the
//! Neural Engine's half in.
//!
//! Short fires (decode, small prompts) keep the whole MLP on the GPU. After a
//! CoreML failure or a non-finite result the layer's Neural Engine half is
//! rerun on the GPU, so the fire stays correct, and the split stops for the
//! rest of the load.
//!
//! The GPU reads its gate and up rows straight out of the model's own banks,
//! and its columns of `down` as the leading columns of `down`'s rows where the
//! MPP matmul runs (they are copied out on devices without it).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, mpsc};

use kernels_metal::{Bank, Tensor};
use model_ir::Dtype;

use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{MTLBuffer, MTLDevice, MTLEvent, MTLSharedEvent};

/// Fires with fewer rows keep the whole MLP on the GPU: the Neural Engine's
/// fixed cost per call outweighs its share below this.
const MIN_ROWS: u32 = 256;

/// A process holds about 128 Neural Engine programs, so each layer loads
/// two: its largest bucket for full chunks and its smallest for the tiles a
/// shorter chunk is cut into.
const PROGRAMS_PER_LAYER: usize = 2;

/// How long the host waits for the GPU to stage a layer before it gives up
/// on that job (a fire that was encoded but never committed).
const STAGE_TIMEOUT_MS: u64 = 60_000;

/// The GPU's half of one layer's MLP: `gate` and `up` are views of rows
/// `[0, keep)` of the model's gate and up planes; `down` is read by its
/// columns `[0, keep)`, in place or from a copy.
#[derive(Clone, Copy)]
pub struct Split {
    pub gate: Bank,
    pub up: Bank,
    pub down: Bank,
    pub keep: u32,
}

/// One layer's split for one fire, handed to the dispatch.
pub struct Plan {
    pub split: Split,
    /// Scratch for the up projection's rows, beside `packed`'s gate rows.
    pub up_rows: Tensor,
    pub staged: Tensor,
    pub other: Tensor,
    pub stage: u32,
    pub done: u32,
    job: Job,
}

impl Plan {
    /// Queues the Neural Engine's half; call once the stage is encoded.
    pub fn submit(&self) {
        if let Some(ane) = ANE.get()
            && let Ok(jobs) = ane.jobs.lock()
        {
            let _ = jobs.send(self.job);
        }
    }
}

#[derive(Clone, Copy)]
struct Job {
    layer: u32,
    rows: u32,
    stage: u64,
    done: u64,
}

struct Meta {
    hidden: u32,
    intermediate: u32,
    ane: u32,
    buckets: Vec<u32>,
    input: String,
    output: String,
    layers: Vec<u32>,
}

struct Ane {
    splits: BTreeMap<u32, Split>,
    max_rows: u32,
    up_rows: Tensor,
    staged: Tensor,
    other: Tensor,
    #[cfg(target_vendor = "apple")]
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    next: AtomicU64,
    jobs: Mutex<mpsc::Sender<Job>>,
    ready: std::sync::Arc<AtomicBool>,
    off: std::sync::Arc<AtomicBool>,
    usable: std::sync::Arc<Vec<AtomicBool>>,
}

// SAFETY: `MTLSharedEvent` is documented thread-safe; the banks and tensors
// are plain handles; the sender sits behind a mutex.
unsafe impl Send for Ane {}
unsafe impl Sync for Ane {}

static ANE: OnceLock<Ane> = OnceLock::new();

thread_local! {
    static LIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Marks the walk this thread is encoding as a live forward into one
/// command buffer, the only kind the split may fence inside.
pub struct Live(bool);

impl Live {
    #[must_use]
    pub fn enter() -> Live {
        Live(LIVE.with(|live| live.replace(true)))
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        LIVE.with(|live| live.set(self.0));
    }
}

/// On unless `PIE_ANE=0`. `PIE_ANE` may name the directory `build.py`
/// wrote; otherwise the newest build under `~/.cache/pie/ane` whose shapes
/// match this model is used, and with none the MLP stays on the GPU.
fn requested() -> Option<String> {
    match std::env::var("PIE_ANE") {
        Ok(v) if v == "0" => None,
        Ok(v) if !v.is_empty() => Some(v),
        _ => Some("1".to_string()),
    }
}

fn find(hidden: u32, inter: u32, asked: &str) -> Option<(PathBuf, Meta)> {
    let read = |dir: &Path| -> Option<Meta> {
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("meta.json")).ok()?).ok()?;
        let int = |key: &str| meta.get(key)?.as_u64().and_then(|v| u32::try_from(v).ok());
        let ints = |key: &str| -> Option<Vec<u32>> {
            meta.get(key)?
                .as_array()?
                .iter()
                .map(|v| v.as_u64().and_then(|v| u32::try_from(v).ok()))
                .collect()
        };
        let text = |key: &str| meta.get(key)?.as_str().map(str::to_string);
        Some(Meta {
            hidden: int("hidden")?,
            intermediate: int("intermediate")?,
            ane: int("ane")?,
            buckets: ints("buckets")?,
            input: text("input")?,
            output: text("output")?,
            layers: ints("layers")?,
        })
    };
    if asked != "1" {
        let dir = PathBuf::from(asked);
        return read(&dir).map(|meta| (dir, meta));
    }
    let root = PathBuf::from(std::env::var("HOME").ok()?).join(".cache/pie/ane");
    let mut found: Vec<(std::time::SystemTime, PathBuf, Meta)> = Vec::new();
    for model in std::fs::read_dir(root).ok()?.flatten() {
        for split in std::fs::read_dir(model.path()).ok()?.flatten() {
            let dir = split.path();
            if let Some(meta) = read(&dir)
                && meta.hidden == hidden
                && meta.intermediate == inter
            {
                let when = std::fs::metadata(dir.join("meta.json"))
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::UNIX_EPOCH);
                found.push((when, dir, meta));
            }
        }
    }
    found.sort_by_key(|(when, _, _)| *when);
    found.pop().map(|(_, dir, meta)| (dir, meta))
}

/// The MLPs a trace asks to split: `(layer, gate_up row, down row, inter)`.
fn mlps(trace: &model_ir::Trace) -> Vec<(u32, u32, u32, u32)> {
    let weight = |id: model_ir::ValueId| match trace.values[id.0 as usize].def {
        model_ir::Def::Weight(at) => Some(at),
        _ => None,
    };
    trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            model_ir::Operation::Linear(model_ir::Linear::MlpAne {
                gate_up,
                down,
                intermediate,
                layer,
                ..
            }) => Some((*layer, weight(*gate_up)?, weight(*down)?, *intermediate)),
            _ => None,
        })
        .collect()
}

/// A view of rows `[start, start + rows)` of every plane of a bank: the
/// planes are row-major, so it is the same buffer from a later offset.
fn rows_of(handles: &Handles, bank: &Bank, start: u32, rows: u32) -> Result<Bank> {
    let plane = |t: Tensor, unit_bits: u32| -> Result<Tensor> {
        let row = u64::from(t.width) * u64::from(unit_bits) / 8;
        let buf = if start == 0 {
            t.buf
        } else {
            handles.cut(t.buf, row * u64::from(start), row * u64::from(rows))?
        };
        Ok(Tensor::new(buf, rows, t.width, t.dtype))
    };
    Ok(Bank {
        codes: plane(bank.codes, bank.bits)?,
        mpp_codes: None,
        scales: plane(bank.scales, 16)?,
        biases: match bank.biases {
            Some(b) => Some(plane(b, 16)?),
            None => None,
        },
        group: bank.group,
        bits: bank.bits,
    })
}

/// Copies `rows` row ranges of each plane of an affine 4-bit bank into a
/// fresh bank, keeping the first `cols` columns of every row.
fn carve(
    device: &Context,
    handles: &Handles,
    bank: &Bank,
    ranges: &[(u32, u32)],
    cols: u32,
    keep: &mut Vec<Buffer>,
) -> Result<Bank> {
    let rows: u32 = ranges.iter().map(|(_, n)| n).sum();
    let plane =
        |t: Tensor, unit_bits: u32, out_cols: u32, keep: &mut Vec<Buffer>| -> Result<Tensor> {
            let row_in = u64::from(t.width) * u64::from(unit_bits) / 8;
            let row_out = u64::from(out_cols) * u64::from(unit_bits) / 8;
            let whole = handles.read(t.buf, row_in * u64::from(t.rows))?;
            let mut bytes = Vec::with_capacity((row_out * u64::from(rows)) as usize);
            for &(start, n) in ranges {
                for r in start..start + n {
                    let at = (u64::from(r) * row_in) as usize;
                    bytes.extend_from_slice(&whole[at..at + row_out as usize]);
                }
            }
            let buffer = Buffer::zeroed(device, bytes.len() as u64)?;
            write(&buffer, &bytes)?;
            let handle = handles.bind(&buffer, 0, bytes.len() as u64)?;
            keep.push(buffer);
            Ok(Tensor::new(handle, rows, out_cols, t.dtype))
        };
    let groups = cols / bank.group;
    Ok(Bank {
        codes: plane(bank.codes, bank.bits, cols, keep)?,
        mpp_codes: None,
        scales: plane(bank.scales, 16, groups, keep)?,
        biases: match bank.biases {
            Some(b) => Some(plane(b, 16, groups, keep)?),
            None => None,
        },
        group: bank.group,
        bits: bank.bits,
    })
}

fn write(buffer: &Buffer, bytes: &[u8]) -> Result<()> {
    #[cfg(target_vendor = "apple")]
    {
        let base = buffer.raw().contents().as_ptr().cast::<u8>();
        // SAFETY: the buffer is host-visible and was sized to `bytes`.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), base, bytes.len()) };
        Ok(())
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (buffer, bytes);
        Err(Fault::Deviceless)
    }
}

/// Builds the GPU halves, stages the rows the two engines trade through,
/// and starts the host thread that drives CoreML. A no-op unless `PIE_ANE`
/// is set and the trace has an MLP to split.
pub fn load(
    device: &Context,
    handles: &Handles,
    trace: &model_ir::Trace,
    weights: &mut crate::weights::Weights,
) -> Result<()> {
    let Some(asked) = requested() else {
        return Ok(());
    };
    let mlps = mlps(trace);
    let Some(&(_, gate_up0, _, inter)) = mlps.first() else {
        return Ok(());
    };
    let hidden = match weights.table().0.get(gate_up0 as usize).copied().flatten() {
        Some(crate::run::WeightRow::Planes(bank)) => bank.codes.width,
        _ => {
            eprintln!("PIE_ANE: the MLP weights are not quantized banks; staying on the GPU");
            return Ok(());
        }
    };
    let Some((dir, meta)) = find(hidden, inter, &asked) else {
        eprintln!(
            "PIE_ANE: no Neural Engine build for hidden {hidden} / intermediate {inter}; \
             run scripts/ane/build.py. Staying on the GPU"
        );
        return Ok(());
    };
    let keep = inter - meta.ane;
    let max_rows = meta.buckets.iter().copied().max().unwrap_or(0);

    let mut buffers = Vec::new();
    let mut splits = BTreeMap::new();
    for &(layer, gate_up, down, layer_inter) in &mlps {
        if layer_inter != inter || !meta.layers.contains(&layer) {
            continue;
        }
        let table = &weights.table().0;
        let (Some(crate::run::WeightRow::Planes(gu)), Some(crate::run::WeightRow::Planes(dn))) = (
            table.get(gate_up as usize).copied().flatten(),
            table.get(down as usize).copied().flatten(),
        ) else {
            continue;
        };
        if gu.bits != 4 || dn.bits != 4 || gu.codes.dtype != Dtype::U4g64 || keep % gu.group != 0 {
            continue;
        }
        let gate = rows_of(handles, &gu, 0, keep)?;
        let up = rows_of(handles, &gu, inter, keep)?;
        // The MPP matmul reads a row's leading columns in place; elsewhere
        // the GPU's columns of `down` are copied out.
        let down = if kernels_metal::tuning::current().qmm_mpp {
            Bank {
                mpp_codes: None,
                ..dn
            }
        } else {
            let mut down = carve(
                device,
                handles,
                &dn,
                &[(0, dn.codes.rows)],
                keep,
                &mut buffers,
            )?;
            down.mpp_codes = crate::weights::mpp_pack(device, handles, &down, &mut buffers)?;
            down
        };
        splits.insert(
            layer,
            Split {
                gate,
                up,
                down,
                keep,
            },
        );
    }
    if splits.is_empty() {
        eprintln!(
            "PIE_ANE: {} has no layer this model can split",
            dir.display()
        );
        return Ok(());
    }

    let up_bytes = u64::from(max_rows) * u64::from(keep) * 2;
    let up_buf = Buffer::zeroed(device, up_bytes)?;
    let up_rows = Tensor::new(
        handles.bind(&up_buf, 0, up_bytes)?,
        max_rows,
        keep,
        Dtype::Bf16,
    );
    buffers.push(up_buf);
    let bytes = u64::from(max_rows) * u64::from(hidden) * 2;
    let staged_buf = Buffer::zeroed(device, bytes)?;
    let other_buf = Buffer::zeroed(device, bytes)?;
    let staged = Tensor::new(
        handles.bind(&staged_buf, 0, bytes)?,
        max_rows,
        hidden,
        Dtype::F16,
    );
    let other = Tensor::new(
        handles.bind(&other_buf, 0, bytes)?,
        max_rows,
        hidden,
        Dtype::F16,
    );

    #[cfg(target_vendor = "apple")]
    {
        let event = device.device().newSharedEvent().ok_or(Fault::Device {
            call: "newSharedEvent",
            why: "the device would not make a shared event".to_string(),
        })?;
        let ready = std::sync::Arc::new(AtomicBool::new(false));
        let off = std::sync::Arc::new(AtomicBool::new(false));
        let top = splits.keys().max().map_or(0, |l| *l as usize + 1);
        let usable =
            std::sync::Arc::new((0..top).map(|_| AtomicBool::new(false)).collect::<Vec<_>>());
        let (tx, rx) = mpsc::channel();
        let worker = Worker {
            dir: dir.clone(),
            meta,
            hidden,
            input: staged_buf.raw().contents().as_ptr() as usize,
            output: other_buf.raw().contents().as_ptr() as usize,
            event: SendEvent(event.clone()),
            ready: ready.clone(),
            off: off.clone(),
            usable: usable.clone(),
        };
        buffers.push(staged_buf);
        buffers.push(other_buf);
        weights.keep_buffers(buffers);
        let layers: Vec<u32> = splits.keys().copied().collect();
        std::thread::Builder::new()
            .name("pie-ane".into())
            .spawn(move || worker.run(&layers, rx))
            .map_err(|e| Fault::Device {
                call: "spawn",
                why: e.to_string(),
            })?;
        eprintln!(
            "PIE_ANE: {} of {} MLP layers split, GPU {keep} / Neural Engine {} columns, from {}",
            splits.len(),
            mlps.len(),
            inter - keep,
            dir.display()
        );
        let _ = ANE.set(Ane {
            splits,
            max_rows,
            up_rows,
            staged,
            other,
            event,
            next: AtomicU64::new(0),
            jobs: Mutex::new(tx),
            ready,
            off,
            usable,
        });
    }
    Ok(())
}

/// The split for `layer` in a fire of `rows` rows, or `None` to run the
/// whole MLP on the GPU.
#[must_use]
pub fn plan(layer: u32, rows: u32) -> Option<Plan> {
    let ane = ANE.get()?;
    if !LIVE.with(std::cell::Cell::get)
        || rows < MIN_ROWS
        || rows > ane.max_rows
        || !ane.ready.load(Ordering::Acquire)
        || ane.off.load(Ordering::Acquire)
        || crate::diag::on().kernel_profile.on()
    {
        return None;
    }
    if !ane.usable.get(layer as usize)?.load(Ordering::Acquire) {
        return None;
    }
    let split = *ane.splits.get(&layer)?;
    let base = ane.next.fetch_add(2, Ordering::AcqRel);
    let (stage, done) = (base + 1, base + 2);
    Some(Plan {
        split,
        up_rows: Tensor {
            rows,
            ..ane.up_rows
        },
        staged: Tensor { rows, ..ane.staged },
        other: Tensor { rows, ..ane.other },
        stage: u32::try_from(stage).ok()?,
        done: u32::try_from(done).ok()?,
        job: Job {
            layer,
            rows,
            stage,
            done,
        },
    })
}

/// The event the encoder fences on around a split.
#[cfg(target_vendor = "apple")]
#[must_use]
pub fn event() -> Option<&'static ProtocolObject<dyn MTLEvent>> {
    ANE.get().map(|ane| ProtocolObject::from_ref(&*ane.event))
}

/// Waits for the GPU to reach `value`: spins first, since a blocking wait
/// wakes up late on the path between the two engines, then blocks.
#[cfg(target_vendor = "apple")]
fn staged(event: &ProtocolObject<dyn MTLSharedEvent>, value: u64) -> bool {
    let spin = std::time::Instant::now();
    while spin.elapsed() < std::time::Duration::from_millis(200) {
        if event.signaledValue() >= value {
            return true;
        }
        std::hint::spin_loop();
    }
    event.waitUntilSignaledValue_timeoutMS(value, STAGE_TIMEOUT_MS)
}

#[cfg(target_vendor = "apple")]
struct SendEvent(Retained<ProtocolObject<dyn MTLSharedEvent>>);

// SAFETY: `MTLSharedEvent` is documented thread-safe.
#[cfg(target_vendor = "apple")]
unsafe impl Send for SendEvent {}

#[cfg(target_vendor = "apple")]
struct Worker {
    dir: PathBuf,
    meta: Meta,
    hidden: u32,
    input: usize,
    output: usize,
    event: SendEvent,
    ready: std::sync::Arc<AtomicBool>,
    off: std::sync::Arc<AtomicBool>,
    usable: std::sync::Arc<Vec<AtomicBool>>,
}

#[cfg(target_vendor = "apple")]
mod coreml {
    use std::ffi::{CString, c_char, c_int, c_long, c_void};

    unsafe extern "C" {
        fn pie_coreml_load(
            path: *const c_char,
            function: *const c_char,
            units: c_int,
            err: *mut c_char,
            cap: c_int,
        ) -> *mut c_void;
        fn pie_coreml_free(model: *mut c_void);
        fn pie_coreml_predict(
            model: *mut c_void,
            in_name: *const c_char,
            input: *mut c_void,
            rows: c_long,
            width_in: c_long,
            out_name: *const c_char,
            output: *mut c_void,
            width_out: c_long,
            err: *mut c_char,
            cap: c_int,
        ) -> c_int;
    }

    pub enum Units {
        NeuralEngine,
        Gpu,
    }

    pub struct Model(*mut c_void);

    // SAFETY: an `MLModel` may be used from any thread; this one is only
    // ever used from the worker.
    unsafe impl Send for Model {}

    fn text(err: &[c_char]) -> String {
        // SAFETY: the bridge writes a NUL-terminated string into `err`.
        unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    impl Model {
        pub fn load(path: &std::path::Path, function: &str, units: Units) -> Result<Model, String> {
            let path =
                CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
            let function = CString::new(function).map_err(|e| e.to_string())?;
            let mut err = [0 as c_char; 512];
            let units = match units {
                Units::NeuralEngine => 0,
                Units::Gpu => 2,
            };
            // SAFETY: both strings are NUL-terminated and `err` holds `cap` bytes.
            let model = unsafe {
                pie_coreml_load(
                    path.as_ptr(),
                    function.as_ptr(),
                    units,
                    err.as_mut_ptr(),
                    512,
                )
            };
            if model.is_null() {
                return Err(text(&err));
            }
            Ok(Model(model))
        }

        /// # Safety
        /// `input` and `output` hold `rows` rows of `width` fp16 values.
        pub unsafe fn predict(
            &self,
            input_name: &CString,
            input: *mut c_void,
            output_name: &CString,
            output: *mut c_void,
            rows: u32,
            width: u32,
        ) -> Result<(), String> {
            let mut err = [0 as c_char; 512];
            // SAFETY: the caller vouches for the two spans.
            let status = unsafe {
                pie_coreml_predict(
                    self.0,
                    input_name.as_ptr(),
                    input,
                    c_long::from(rows),
                    c_long::from(width),
                    output_name.as_ptr(),
                    output,
                    c_long::from(width),
                    err.as_mut_ptr(),
                    512,
                )
            };
            if status == 0 { Ok(()) } else { Err(text(&err)) }
        }
    }

    impl Drop for Model {
        fn drop(&mut self) {
            // SAFETY: the pointer came from `pie_coreml_load` and is freed once.
            unsafe { pie_coreml_free(self.0) };
        }
    }
}

#[cfg(target_vendor = "apple")]
impl Worker {
    fn path(&self, layer: u32) -> PathBuf {
        self.dir.join(format!("layer{layer:03}.mlmodelc"))
    }

    /// The buckets this worker loads: the largest and the smallest.
    fn loaded(&self) -> Vec<u32> {
        let mut buckets = self.meta.buckets.clone();
        buckets.sort_unstable();
        buckets.dedup();
        let (Some(&small), Some(&large)) = (buckets.first(), buckets.last()) else {
            return Vec::new();
        };
        let mut loaded = vec![large, small];
        loaded.dedup();
        loaded.truncate(PROGRAMS_PER_LAYER);
        loaded
    }

    /// Cuts `rows` into `(offset, bucket)` runs: whole large buckets, then
    /// small tiles, the last one padded.
    fn tiles(&self, rows: u32) -> Vec<(u32, u32)> {
        let loaded = self.loaded();
        let (large, small) = (loaded[0], *loaded.last().expect("one bucket at least"));
        let mut out = Vec::new();
        let mut at = 0;
        while rows - at >= large {
            out.push((at, large));
            at += large;
        }
        while at < rows {
            out.push((at, small));
            at += small;
        }
        out
    }

    fn run(self, layers: &[u32], jobs: mpsc::Receiver<Job>) {
        // The host is on the critical path between the two engines; keep it
        // on a performance core and out of timer coalescing.
        // SAFETY: sets this thread's own QoS class.
        unsafe {
            libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
        }
        let started = std::time::Instant::now();
        let mut models: BTreeMap<(u32, u32), coreml::Model> = BTreeMap::new();
        for &layer in layers {
            let mut loaded = true;
            for rows in self.loaded() {
                match coreml::Model::load(
                    &self.path(layer),
                    &format!("t{rows}"),
                    coreml::Units::NeuralEngine,
                ) {
                    Ok(model) => {
                        models.insert((layer, rows), model);
                    }
                    Err(why) => {
                        eprintln!(
                            "PIE_ANE: layer {layer} t{rows} did not load ({why}); it stays on the GPU"
                        );
                        loaded = false;
                        break;
                    }
                }
            }
            if loaded && let Some(usable) = self.usable.get(layer as usize) {
                usable.store(true, Ordering::Release);
            }
        }
        eprintln!(
            "PIE_ANE: {} Neural Engine programs ready in {:.1}s",
            models.len(),
            started.elapsed().as_secs_f64()
        );
        self.ready.store(true, Ordering::Release);

        let input_name = CString::new(self.meta.input.as_str()).expect("a plain name");
        let output_name = CString::new(self.meta.output.as_str()).expect("a plain name");
        let event = &self.event.0;
        let trace = std::env::var("PIE_ANE_TRACE").is_ok();
        let (mut waited, mut ran_for, mut count) = (0.0f64, 0.0f64, 0u32);
        let mut host = (Vec::new(), Vec::new());
        for job in jobs {
            let clock = std::time::Instant::now();
            if !staged(event, job.stage) {
                eprintln!("PIE_ANE: layer {} was never staged; skipping it", job.layer);
                event.setSignaledValue(job.done.max(event.signaledValue()));
                continue;
            }
            let staged = clock.elapsed().as_secs_f64();
            let ran = self.predict(&models, job, &input_name, &output_name, &mut host);
            if trace {
                waited += staged;
                ran_for += clock.elapsed().as_secs_f64() - staged;
                count += 1;
                if count % 64 == 0 {
                    eprintln!(
                        "PIE_ANE: {count} jobs ({} rows last): {:.2} ms waiting for the GPU, {:.2} ms on the Neural Engine per job",
                        job.rows,
                        waited * 1e3 / 64.0,
                        ran_for * 1e3 / 64.0
                    );
                    (waited, ran_for) = (0.0, 0.0);
                }
            }
            if let Err(why) = ran {
                eprintln!(
                    "PIE_ANE: layer {} failed on the Neural Engine ({why}); rerunning it on the GPU \
                     and stopping the split",
                    job.layer
                );
                self.off.store(true, Ordering::Release);
                if let Err(why) = self.on_gpu(job, &input_name, &output_name, &mut host) {
                    eprintln!("PIE_ANE: layer {} failed on the GPU too: {why}", job.layer);
                }
            }
            event.setSignaledValue(job.done);
        }
    }

    /// The layer's program over the job's rows, tile by tile. CoreML reads
    /// and writes host copies: handing it the Metal buffers' own pages
    /// stalls the GPU for as long as the Neural Engine holds them.
    fn predict(
        &self,
        models: &BTreeMap<(u32, u32), coreml::Model>,
        job: Job,
        input: &CString,
        output: &CString,
        host: &mut (Vec<u16>, Vec<u16>),
    ) -> std::result::Result<(), String> {
        let width = self.hidden as usize;
        let tiles = self.tiles(job.rows);
        let span = tiles.last().map_or(0, |(at, b)| (at + b) as usize) * width;
        if host.0.len() < span {
            host.0.resize(span, 0);
            host.1.resize(span, 0);
        }
        let rows = job.rows as usize * width;
        // SAFETY: the staging buffer holds at least `rows` fp16 values.
        let staged = unsafe { std::slice::from_raw_parts(self.input as *const u16, rows) };
        host.0[..rows].copy_from_slice(staged);
        for (at, bucket) in tiles {
            let model = models
                .get(&(job.layer, bucket))
                .ok_or_else(|| format!("no t{bucket} program"))?;
            let skip = at as usize * width;
            // SAFETY: both host copies hold `span >= skip + bucket * width` values.
            unsafe {
                model.predict(
                    input,
                    host.0.as_mut_ptr().add(skip).cast(),
                    output,
                    host.1.as_mut_ptr().add(skip).cast(),
                    bucket,
                    self.hidden,
                )?;
            }
        }
        // SAFETY: the output staging buffer holds at least `rows` fp16 values.
        let out = unsafe { std::slice::from_raw_parts_mut(self.output as *mut u16, rows) };
        out.copy_from_slice(&host.1[..rows]);
        self.finite(job.rows)
    }

    /// A strided look for NaN or infinity in the rows the GPU will read.
    fn finite(&self, rows: u32) -> std::result::Result<(), String> {
        let n = rows as usize * self.hidden as usize;
        // SAFETY: the output buffer holds at least `rows * hidden` fp16 values.
        let out = unsafe { std::slice::from_raw_parts(self.output as *const u16, n) };
        if out.iter().step_by(61).any(|&h| h & 0x7c00 == 0x7c00) {
            return Err("a non-finite value in its output".to_string());
        }
        Ok(())
    }

    fn on_gpu(
        &self,
        job: Job,
        input: &CString,
        output: &CString,
        host: &mut (Vec<u16>, Vec<u16>),
    ) -> std::result::Result<(), String> {
        let mut models = BTreeMap::new();
        for bucket in self.loaded() {
            let model = coreml::Model::load(
                &self.path(job.layer),
                &format!("t{bucket}"),
                coreml::Units::Gpu,
            )?;
            models.insert((job.layer, bucket), model);
        }
        self.predict(&models, job, input, output, host)
    }
}

#[cfg(target_vendor = "apple")]
use std::ffi::CString;
