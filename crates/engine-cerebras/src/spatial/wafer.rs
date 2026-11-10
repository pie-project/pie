//! The wafer program ([`ArtifactModel`]) on a thread of its own: the SDK
//! wants every runtime call on the thread that made the first runtime, so
//! the engine sends it jobs ([`Wafer`]) and waits on their answers. One
//! sequence: greedy rows by the wafer token loop (waited for, or posted to
//! run while the host answers rows), logits rows by `lgt`.

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

use super::artifact::{self, ArtifactModel};
use crate::sdk::{Platform, Runtime, Sdk};

type R<T> = std::result::Result<T, String>;

/// Time spent waiting on the wafer thread's launches, all fires together: the host's share of a token is
/// the rest of its wall time.
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

/// An [`ArtifactModel`] on its own thread.
pub struct Wafer {
    jobs: Option<mpsc::Sender<Job>>,
    answers: mpsc::Receiver<Answer>,
    /// Posted token loops' answers, in order.
    posted: mpsc::Receiver<Answer>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// Passes a launch runs at most.
    pub cap: usize,
    /// The KV positions the sequence may hold.
    pub positions: usize,
    pub vocab: usize,
    pub hidden: usize,
    /// The program's posted-loop length (its plan's; `PIE_CEREBRAS_POSTED_LOOP` overrides it).
    pub posted_loop: usize,
}

impl Wafer {
    /// Loads the compiled `program` (its [`artifact`]) on `platform`, places the prep's tiles
    /// (`PIE_CEREBRAS_WEIGHTS`) and returns once the model answers; `vocab`: the trace's.
    pub fn open(program: PathBuf, platform: Platform, vocab: usize) -> R<Wafer> {
        let weights = artifact::Weights::from_env()?;
        let art = artifact::read(&program)?;
        let posted_loop = art.host.loop_len as usize;
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (ans_tx, ans_rx) = mpsc::channel::<Answer>();
        let (post_tx, post_rx) = mpsc::channel::<Answer>();
        let (ready_tx, ready_rx) = mpsc::channel::<R<(usize, usize, usize, usize)>>();
        let thread = std::thread::Builder::new()
            .name("spatial-wafer".into())
            .spawn(move || {
                let open = || -> R<ArtifactModel> {
                    let sdk = Sdk::open().map_err(|e| e.to_string())?;
                    let mut rt = Runtime::new(sdk, &program, platform, "WARNING")
                        .map_err(|e| e.to_string())?;
                    let t0 = Instant::now();
                    rt.load();
                    rt.run();
                    tracing::info!(s = t0.elapsed().as_secs_f64(), "artifact: program loaded");
                    ArtifactModel::open(Box::new(rt), &weights, &art, vocab)
                };
                let mut m = match open() {
                    Ok(m) => {
                        let _ = ready_tx.send(Ok((m.cap, m.positions, m.vocab, m.hidden)));
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
                    };
                    idle_from = Instant::now();
                    tracing::info!(
                        launches = m.launches,
                        steps = m.steps,
                        "spatial wafer: totals"
                    );
                    if to.send(answer).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| format!("spawning the wafer thread: {e}"))?;
        let (cap, positions, vocab, hidden) = ready_rx
            .recv()
            .map_err(|_| "the wafer thread died while loading".to_string())??;
        Ok(Wafer {
            jobs: Some(job_tx),
            answers: ans_rx,
            posted: post_rx,
            thread: Some(thread),
            cap,
            positions,
            vocab,
            hidden,
            posted_loop,
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

    /// `tokens` at `start..`: the argmax after each row and the logits `which` asks for (one row
    /// for `Last`, a row a token for `All`).
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
fn forward(m: &mut ArtifactModel, tokens: &[u32], start: usize, which: Logits) -> Answer {
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
