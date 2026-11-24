//! The wafer program ([`ArtifactModel`]) on a thread of its own (the SDK wants every runtime
//! call on the thread that made the first runtime): the engine sends it jobs ([`Wafer`]) and
//! waits on their answers.

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

use crate::artifact::{self, ArtifactModel};
use crate::sdk::{Platform, Runtime, Sdk};
use crate::units::UnitModel;

type R<T> = std::result::Result<T, String>;

/// Time spent waiting on the wafer thread's launches, all fires together.
pub(crate) static MODEL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Adds its lifetime to [`MODEL_NS`].
struct Busy(Instant);

impl Busy {
    fn start() -> Busy {
        Busy(Instant::now())
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        let ns = u64::try_from(self.0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        MODEL_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Which logits a forward hands back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Logits {
    None,
    Last,
    All,
}

enum Job {
    /// Rows of Pie's lanes, each over its own pages (a unit program's: [`UnitModel::rows`]).
    Rows(Vec<crate::units::Row>),
    Forward {
        tokens: Vec<u32>,
        start: usize,
        which: Logits,
    },
    Greedy {
        tokens: Vec<u32>,
        start: usize,
        extra: usize,
        /// Answered on the posted channel, not waited for.
        posted: bool,
    },
}

/// The argmax after each row, and the logits rows asked for.
type Answer = R<(Vec<u32>, Vec<Vec<f32>>)>;

/// The model a wafer program serves: the decode template's, or a program of stream units.
enum Model {
    Artifact(Box<ArtifactModel>),
    Units(Box<UnitModel>),
}

impl Model {
    /// (positions, vocab, hidden)
    fn sizes(&self) -> (usize, usize, usize) {
        match self {
            Model::Artifact(m) => (m.positions, m.vocab, m.hidden),
            Model::Units(m) => (m.positions, m.vocab, 0),
        }
    }

    /// Rows over their own pages: each row's argmax, and the logits of those that ask.
    fn rows(&mut self, rows: &[crate::units::Row]) -> Answer {
        match self {
            Model::Artifact(_) => {
                Err("the decode template runs one sequence, not paged rows".into())
            }
            Model::Units(m) => {
                let got = m.rows(rows)?;
                let argmax = got.iter().map(|g| g.0).collect();
                Ok((argmax, got.into_iter().filter_map(|g| g.1).collect()))
            }
        }
    }

    /// Its paging: (lane slots a launch, pages a lane may hold, positions a page); none for the
    /// template's.
    fn paging(&self) -> Option<(usize, u32, u32)> {
        match self {
            Model::Artifact(_) => None,
            Model::Units(m) => m.pages().map(|(p, s)| (m.lanes(), p, s)),
        }
    }

    /// A unit program's lane slots a launch; none for the template's.
    fn lanes(&self) -> Option<usize> {
        match self {
            Model::Artifact(_) => None,
            Model::Units(m) => Some(m.lanes()),
        }
    }

    fn counts(&self) -> (u64, u64) {
        match self {
            Model::Artifact(m) => (m.launches, m.steps),
            Model::Units(m) => (m.launches, m.steps),
        }
    }

    fn greedy(&mut self, tokens: &[u32], start: usize, extra: usize) -> R<Vec<u32>> {
        match self {
            Model::Artifact(m) => m.greedy(tokens, start, extra),
            Model::Units(m) => m.greedy(tokens, start, extra),
        }
    }

    fn forward(
        &mut self,
        tokens: &[u32],
        start: usize,
        logits: bool,
    ) -> R<(Vec<u32>, Option<Vec<f32>>)> {
        match self {
            Model::Artifact(m) => m.forward(tokens, start, logits),
            Model::Units(m) => m.forward(tokens, start, logits),
        }
    }
}

/// A [`Model`] on its own thread.
pub struct Wafer {
    jobs: Option<mpsc::Sender<Job>>,
    answers: mpsc::Receiver<Answer>,
    /// Posted token loops' answers, in order.
    posted: mpsc::Receiver<Answer>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// The KV positions the sequence may hold.
    pub positions: usize,
    pub vocab: usize,
    pub hidden: usize,
    /// The program's posted-loop length (its plan's).
    pub posted_loop: usize,
    /// A unit program's paging: (lane slots a launch, pages a lane may hold, positions a page).
    pub paging: Option<(usize, u32, u32)>,
    /// A unit program's lane slots a launch (its rows: Pie's lanes run together).
    pub lanes: Option<usize>,
    /// Whether the program runs the token loop (a template's WL build; a unit program's rows).
    pub token_loop: bool,
}

impl Wafer {
    /// Loads the compiled `program` (its [`artifact`]) on `platform` through the SDK in
    /// `sdk_lib`, places `weights` and returns once the model answers; `vocab`: the trace's.
    pub fn open(
        program: PathBuf,
        platform: Platform,
        vocab: usize,
        weights: artifact::Weights,
        sdk_lib: PathBuf,
    ) -> R<Wafer> {
        let art = artifact::read(&program)?;
        let posted_loop = art.host.loop_len as usize;
        let token_loop = art.host.token_loop;
        Wafer::spawn(posted_loop, token_loop, move || {
            let sdk = Sdk::open_at(&sdk_lib).map_err(|e| e.to_string())?;
            let mut rt =
                Runtime::new(sdk, &program, platform, "WARNING").map_err(|e| e.to_string())?;
            let t0 = Instant::now();
            rt.load();
            rt.run();
            tracing::info!(s = t0.elapsed().as_secs_f64(), "artifact: program loaded");
            let m = ArtifactModel::open(Box::new(rt), &weights, &art, vocab)?;
            Ok(Model::Artifact(Box::new(m)))
        })
    }

    /// A decode program of stream units ([`UnitModel`]): `program` its files, `link` its link, its
    /// statics made from `checkpoint` as `contract` lands it.
    #[allow(clippy::too_many_arguments)]
    pub fn open_units(
        program: PathBuf,
        link: PathBuf,
        platform: Platform,
        sdk_lib: PathBuf,
        trace: poem_ir::Trace,
        contract: checkpoint::contract::ModelContract,
        checkpoint: PathBuf,
        positions: usize,
    ) -> R<Wafer> {
        Wafer::spawn(0, true, move || {
            let reading = crate::reading::Reading::open(&program)?;
            let landed = crate::prepare::land(&trace, &contract, &checkpoint)?;
            let sdk = Sdk::open_at(&sdk_lib).map_err(|e| e.to_string())?;
            let mut rt =
                Runtime::new(sdk, &link, platform, "WARNING").map_err(|e| e.to_string())?;
            let t0 = Instant::now();
            rt.load();
            rt.run();
            tracing::info!(s = t0.elapsed().as_secs_f64(), "units: program loaded");
            let of = |name: &str, count: usize| crate::prepare::plane(&landed, name, count);
            let m = UnitModel::open(Box::new(rt), reading, trace, &of, positions)?;
            Ok(Model::Units(Box::new(m)))
        })
    }

    /// The model `open` makes, on a thread of its own taking jobs.
    fn spawn(
        posted_loop: usize,
        token_loop: bool,
        open: impl FnOnce() -> R<Model> + Send + 'static,
    ) -> R<Wafer> {
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (ans_tx, ans_rx) = mpsc::channel::<Answer>();
        let (post_tx, post_rx) = mpsc::channel::<Answer>();
        let (ready_tx, ready_rx) = mpsc::channel::<
            R<(
                (usize, usize, usize),
                Option<(usize, u32, u32)>,
                Option<usize>,
            )>,
        >();
        let thread = std::thread::Builder::new()
            .name("spatial-wafer".into())
            .spawn(move || {
                let mut m = match open() {
                    Ok(m) => {
                        let _ = ready_tx.send(Ok((m.sizes(), m.paging(), m.lanes())));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let mut idle_from = Instant::now();
                for job in job_rx {
                    let t0 = Instant::now();
                    let (answer, to) = match job {
                        Job::Greedy {
                            tokens,
                            start,
                            extra,
                            posted,
                        } => {
                            let a = m.greedy(&tokens, start, extra).map(|a| (a, Vec::new()));
                            let (rows, steps, pos) = (tokens.len(), extra + 1, start);
                            let idle_ms = (t0 - idle_from).as_secs_f64() * 1e3;
                            let ms = t0.elapsed().as_secs_f64() * 1e3;
                            if posted {
                                tracing::info!(
                                    rows,
                                    steps,
                                    pos,
                                    idle_ms,
                                    ms,
                                    "spatial wafer: artifact token loop launch (posted)"
                                );
                                (a, &post_tx)
                            } else {
                                tracing::info!(
                                    rows,
                                    steps,
                                    pos,
                                    idle_ms,
                                    ms,
                                    "spatial wafer: artifact token loop launch"
                                );
                                (a, &ans_tx)
                            }
                        }
                        Job::Forward {
                            tokens,
                            start,
                            which,
                        } => (forward(&mut m, &tokens, start, which), &ans_tx),
                        Job::Rows(rows) => (m.rows(&rows), &ans_tx),
                    };
                    idle_from = Instant::now();
                    let (launches, steps) = m.counts();
                    tracing::info!(launches, steps, "spatial wafer: totals");
                    if to.send(answer).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| format!("spawning the wafer thread: {e}"))?;
        let ((positions, vocab, hidden), paging, lanes) = ready_rx
            .recv()
            .map_err(|_| "the wafer thread died while loading".to_string())??;
        Ok(Wafer {
            jobs: Some(job_tx),
            answers: ans_rx,
            posted: post_rx,
            thread: Some(thread),
            positions,
            vocab,
            hidden,
            posted_loop,
            paging,
            lanes,
            token_loop,
        })
    }

    fn send(&self, job: Job) -> R<()> {
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(job)
            .map_err(|_| "the wafer thread went away".to_string())
    }

    fn answer(&self) -> Answer {
        self.answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
    }

    /// The token loop: `tokens` at `start..`, then `extra` steps each fed the argmax before; the
    /// argmax after each row.
    pub fn greedy(&mut self, tokens: &[u32], start: usize, extra: usize) -> R<Vec<u32>> {
        let _busy = Busy::start();
        self.send(Job::Greedy {
            tokens: tokens.to_vec(),
            start,
            extra,
            posted: false,
        })?;
        self.answer().map(|(argmax, _)| argmax)
    }

    /// Posts a token loop: its answer comes from [`Wafer::take_posted`], in order.
    pub fn post_greedy(&mut self, tokens: &[u32], start: usize, extra: usize) -> R<()> {
        self.send(Job::Greedy {
            tokens: tokens.to_vec(),
            start,
            extra,
            posted: true,
        })
    }

    /// A posted loop's argmax after each row (as [`Wafer::greedy`]'s): waiting for it, or `None`
    /// if it has not landed.
    pub fn take_posted(&mut self, wait: bool) -> Option<R<Vec<u32>>> {
        let got = if wait {
            let _busy = Busy::start();
            self.posted
                .recv()
                .map_err(|_| "the wafer thread went away".to_string())
        } else {
            match self.posted.try_recv() {
                Ok(a) => Ok(a),
                Err(mpsc::TryRecvError::Empty) => return None,
                Err(_) => Err("the wafer thread went away".to_string()),
            }
        };
        Some(got.and_then(|a| a.map(|(argmax, _)| argmax)))
    }

    /// Rows of Pie's lanes over their own pages: each row's argmax, and the logits of those that
    /// ask, in order.
    pub fn rows(&mut self, rows: Vec<crate::units::Row>) -> Answer {
        let _busy = Busy::start();
        self.send(Job::Rows(rows))?;
        self.answer()
    }

    pub fn forward(&mut self, tokens: &[u32], start: usize, which: Logits) -> Answer {
        let _busy = Busy::start();
        self.send(Job::Forward {
            tokens: tokens.to_vec(),
            start,
            which,
        })?;
        self.answer()
    }
}

/// A forward on the wafer's thread: every row's logits a row at a time.
fn forward(m: &mut Model, tokens: &[u32], start: usize, which: Logits) -> Answer {
    match which {
        Logits::None => m
            .forward(tokens, start, false)
            .map(|(a, _)| (a, Vec::new())),
        Logits::Last => m
            .forward(tokens, start, true)
            .map(|(a, l)| (a, l.into_iter().collect())),
        Logits::All => {
            let (mut argmax, mut rows) = (Vec::new(), Vec::new());
            for (i, &t) in tokens.iter().enumerate() {
                let (a, l) = m.forward(&[t], start + i, true)?;
                argmax.extend(a);
                rows.extend(l);
            }
            Ok((argmax, rows))
        }
    }
}

impl Drop for Wafer {
    fn drop(&mut self) {
        // Closing the job channel ends the thread, which stops the program.
        self.jobs = None;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
