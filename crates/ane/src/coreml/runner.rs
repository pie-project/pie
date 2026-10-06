use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLSharedEvent;

use super::model as coreml;
use super::{Build, Job};

const STAGE_TIMEOUT_MS: u64 = 60_000;

#[derive(Default)]
struct State {
    ready: AtomicBool,
    off: AtomicBool,
    lost: AtomicU64,
    usable: Vec<AtomicBool>,
}

pub struct Runner {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    jobs: Mutex<Option<mpsc::Sender<Job>>>,
    worker: Option<std::thread::JoinHandle<()>>,
    state: Arc<State>,
}

unsafe impl Send for Runner {}
unsafe impl Sync for Runner {}

impl Runner {
    /// # Safety
    /// `input` and `output` hold `max_rows * hidden` fp16 values and outlive the runner.
    pub unsafe fn start(
        build: Build,
        layers: &[u32],
        input: *const u16,
        output: *mut u16,
        event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    ) -> std::io::Result<Runner> {
        let top = layers.iter().max().map_or(0, |l| *l as usize + 1);
        let state = Arc::new(State {
            usable: (0..top).map(|_| AtomicBool::new(false)).collect(),
            ..State::default()
        });
        let (tx, rx) = mpsc::channel();
        let worker = Worker {
            build,
            input: input as usize,
            output: output as usize,
            event: SendEvent(event.clone()),
            state: state.clone(),
        };
        let layers = layers.to_vec();
        let worker = std::thread::Builder::new()
            .name("pie-ane".into())
            .spawn(move || worker.run(&layers, rx))?;
        Ok(Runner {
            event,
            jobs: Mutex::new(Some(tx)),
            worker: Some(worker),
            state,
        })
    }

    #[must_use]
    pub fn takes(&self, layer: u32) -> bool {
        self.state.ready.load(Ordering::Acquire)
            && !self.state.off.load(Ordering::Acquire)
            && self
                .state
                .usable
                .get(layer as usize)
                .is_some_and(|usable| usable.load(Ordering::Acquire))
    }

    pub fn submit(&self, job: Job) {
        if let Ok(jobs) = self.jobs.lock()
            && let Some(jobs) = jobs.as_ref()
        {
            let _ = jobs.send(job);
        }
    }

    #[must_use]
    pub fn event(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &self.event
    }

    pub fn lost(&self) -> impl Fn() -> u64 + Send + 'static {
        let state = self.state.clone();
        move || state.lost.load(Ordering::Acquire)
    }
}

impl Drop for Runner {
    fn drop(&mut self) {
        if let Ok(mut jobs) = self.jobs.lock() {
            jobs.take();
        }
        self.event.setSignaledValue(u64::MAX);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

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

struct SendEvent(Retained<ProtocolObject<dyn MTLSharedEvent>>);

unsafe impl Send for SendEvent {}

struct Worker {
    build: Build,
    input: usize,
    output: usize,
    event: SendEvent,
    state: Arc<State>,
}

type Programs = BTreeMap<(u32, u32), coreml::Model>;

impl Worker {
    fn path(&self, layer: u32) -> PathBuf {
        self.build.dir.join(format!("layer{layer:03}.mlmodelc"))
    }

    fn buckets(&self) -> (u32, u32) {
        let small = self.build.buckets.iter().copied().min().unwrap_or(0);
        (small, self.build.max_rows())
    }

    fn tiles(&self, rows: u32) -> Vec<(u32, u32)> {
        let (small, large) = self.buckets();
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

    fn programs(&self, layer: u32, units: coreml::Units) -> Result<Programs, String> {
        let (small, large) = self.buckets();
        let mut programs = Programs::new();
        for rows in [small, large] {
            let program = coreml::Model::load(&self.path(layer), &format!("t{rows}"), units)?;
            programs.insert((layer, rows), program);
        }
        Ok(programs)
    }

    fn run(self, layers: &[u32], jobs: mpsc::Receiver<Job>) {
        unsafe {
            libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
        }
        let mut programs = Programs::new();
        for &layer in layers {
            match self.programs(layer, coreml::Units::NeuralEngine) {
                Ok(loaded) => {
                    programs.extend(loaded);
                    self.state.usable[layer as usize].store(true, Ordering::Release);
                }
                Err(why) => eprintln!("PIE_ANE: layer {layer} stays on the GPU: {why}"),
            }
        }
        self.state.ready.store(true, Ordering::Release);
        eprintln!("PIE_ANE: {} Neural Engine programs ready", programs.len());

        let names = (
            CString::new(self.build.input.as_str()).expect("a plain name"),
            CString::new(self.build.output.as_str()).expect("a plain name"),
        );
        let event = &self.event.0;
        let mut host = (Vec::new(), Vec::new());
        let mut gpu = Programs::new();
        for job in jobs {
            if !staged(event, job.stage) {
                self.state.off.store(true, Ordering::Release);
                self.state.lost.fetch_add(1, Ordering::AcqRel);
                event.setSignaledValue(job.done.max(event.signaledValue()));
                continue;
            }
            let failed = self.state.off.load(Ordering::Acquire)
                || self
                    .predict(&programs, job, &names, &mut host)
                    .inspect_err(|why| {
                        eprintln!(
                            "PIE_ANE: layer {} failed on the Neural Engine ({why}); rerunning it on the GPU and stopping the split",
                            job.layer
                        );
                        self.state.off.store(true, Ordering::Release);
                    })
                    .is_err();
            if failed {
                if !gpu.keys().any(|(layer, _)| *layer == job.layer) {
                    match self.programs(job.layer, coreml::Units::Gpu) {
                        Ok(loaded) => gpu.extend(loaded),
                        Err(why) => {
                            eprintln!("PIE_ANE: layer {} has no GPU program: {why}", job.layer);
                        }
                    }
                }
                if let Err(why) = self.predict(&gpu, job, &names, &mut host) {
                    self.state.lost.fetch_add(1, Ordering::AcqRel);
                    eprintln!("PIE_ANE: layer {} failed on the GPU too: {why}", job.layer);
                }
            }
            event.setSignaledValue(job.done.max(event.signaledValue()));
        }
    }

    fn predict(
        &self,
        programs: &Programs,
        job: Job,
        names: &(CString, CString),
        host: &mut (Vec<u16>, Vec<u16>),
    ) -> Result<(), String> {
        let width = self.build.hidden as usize;
        let tiles = self.tiles(job.rows);
        let span = tiles.last().map_or(0, |(at, b)| (at + b) as usize) * width;
        if host.0.len() < span {
            host.0.resize(span, 0);
            host.1.resize(span, 0);
        }
        let rows = job.rows as usize * width;
        host.0[..rows]
            .copy_from_slice(unsafe { std::slice::from_raw_parts(self.input as *const u16, rows) });
        for (at, bucket) in tiles {
            let program = programs
                .get(&(job.layer, bucket))
                .ok_or_else(|| format!("no t{bucket} program"))?;
            let skip = at as usize * width;
            unsafe {
                program.predict(
                    (&names.0, &names.1),
                    host.0.as_mut_ptr().add(skip).cast(),
                    host.1.as_mut_ptr().add(skip).cast(),
                    bucket,
                    self.build.hidden,
                )?;
            }
        }
        let out = &host.1[..rows];
        if out.iter().step_by(61).any(|&h| h & 0x7c00 == 0x7c00) {
            return Err("a non-finite value in its output".to_string());
        }
        unsafe { std::slice::from_raw_parts_mut(self.output as *mut u16, rows) }
            .copy_from_slice(out);
        Ok(())
    }
}
