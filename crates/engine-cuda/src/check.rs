//! CUDA's loads for the check of the estimate (`engine::load::check`): a
//! served-shape load of a model laying every rung to four times its fire
//! top, whose rungs and estimate the check weighs, and loads on the shell
//! the check sizes for each kind of its fires, every rung to the shell's
//! rows (`Knobs::every_rung`); their fires replayed as serving runs them
//! or walked eagerly (`Knobs::walked`) and dissected node by node
//! ([`crate::dissect`]).

use std::path::{Path, PathBuf};

use engine::load::CheckFire;
use engine::load::check::{Bench, Dissection, Fired, Loaded, Run, Shell, Weighed};

use crate::tune::{Shape, load, patches};

/// The lanes a served CUDA load is asked (`pie serve`'s); its tokens are
/// left to the engine (its fire top on the card).
const SERVED_LANES: u32 = engine::load::CUDA_FIRE_LANES;

/// The context a served-shape load is asked: the most tokens one prompt
/// holds, so the fire top it weighs is the model's, not a short context's.
/// The server card's working-set context (`ChunkProfile::SERVER_CUDA`).
const SERVED_CONTEXT: u32 = poem_compiler::chunk::ChunkProfile::SERVER_CUDA.context;

/// The page a check's loads lay their cache in: a served load's.
const PAGE: u32 = engine::load::CUDA_PAGE_SIZE;

/// One model's check loads on GPU 0, under the profile the device resolves
/// with the user's profiles.
pub struct Check {
    artifact: PathBuf,
    deployment: String,
    profiles: Option<PathBuf>,
}

/// The check bench of `artifact` (its deployment named), the user's
/// profiles under `profiles`.
#[must_use]
pub fn bench(artifact: &Path, deployment: &str, profiles: Option<&Path>) -> Check {
    Check {
        artifact: artifact.to_path_buf(),
        deployment: deployment.to_owned(),
        profiles: profiles.map(Path::to_path_buf),
    }
}

impl Bench for Check {
    fn load(&mut self, run: Run, shell: Option<Shell>) -> Result<Box<dyn Loaded + '_>, String> {
        // The served shape: a slot a lane, each one long prompt's context.
        let (slots, context, every_rung) = match shell {
            None => (SERVED_LANES, SERVED_CONTEXT, crate::EveryRung::PastTop),
            Some(shell) => (shell.slots, shell.context, crate::EveryRung::To(shell.rows)),
        };
        let knobs = crate::Knobs {
            profiles: self.profiles.clone(),
            every_rung: Some(every_rung),
            walked: run == Run::Walked,
            ..crate::Knobs::default()
        };
        let (shell, trace) = load(&self.artifact, &self.deployment, knobs, |trace| {
            let mut budget = poem_compiler::Budget::new(SERVED_LANES, 0);
            budget.readouts_per_lane = poem_compiler::arena::readouts_per_lane(trace);
            Shape {
                // A slot a lane: a decode-shaped rung's check fires a lane
                // a slot.
                slots,
                budget,
                patches: patches(trace, SERVED_LANES),
                page_size: PAGE,
                context,
            }
        })?;
        Ok(Box::new(CheckLoad {
            shell,
            trace,
            drain_ns: None,
        }))
    }
}

/// One check load: its shell and the trace it serves.
struct CheckLoad {
    shell: crate::Shell,
    trace: poem_ir::Trace,
    /// A drained pair around nothing (ns), taken at the first dissection.
    drain_ns: Option<f64>,
}

impl CheckLoad {
    /// One fire as the estimate weighs a rung: its lanes each on a fresh
    /// slot of its own, each its tokens; its wall time to its landing (ns)
    /// and whether it replayed a recorded body.
    fn fire_once(&mut self, fire: CheckFire) -> Result<Fired, String> {
        let token = |i: u32| 100 + (i * 7919) % 5000;
        let prompts: Vec<Vec<u32>> = (0..fire.lanes)
            .map(|lane| {
                (0..fire.tokens_per_lane)
                    .map(|i| token(lane * fire.tokens_per_lane + i))
                    .collect()
            })
            .collect();
        for slot in 0..fire.lanes {
            self.shell.open(slot).map_err(|e| e.to_string())?;
        }
        let word = self
            .trace
            .facts
            .word(&poem::Request::new(fire.tokens_per_lane, false));
        let lanes: Vec<crate::Lane> = prompts
            .iter()
            .enumerate()
            .map(|(slot, tokens)| crate::Lane {
                slot: slot as u32,
                word,
                tokens,
            })
            .collect();
        let hits = self.shell.body_stats().tally.hits;
        let (landed, _) = self.shell.fire_landed(&lanes).map_err(|e| e.to_string())?;
        Ok(Fired {
            ns: landed.as_secs_f64() * 1e9,
            replayed: self.shell.body_stats().tally.hits > hits,
        })
    }
}

impl Loaded for CheckLoad {
    fn weighed(&self) -> Result<Weighed, String> {
        let weighed = self.shell.weighed().ok_or_else(|| {
            "this device has no rates in its profile to estimate by: run `pie tune --rates`"
                .to_string()
        })?;
        Ok(Weighed {
            model: engine::load::model_class(&self.trace).to_owned(),
            rungs: weighed.rungs.clone(),
            estimate: weighed.estimate.clone(),
            live: weighed.live,
            fire_top: weighed.fire_top,
            laid: self.shell.budget().max_tokens,
        })
    }

    fn fire(&mut self, fire: CheckFire) -> Result<Fired, String> {
        self.fire_once(fire)
    }

    fn dissect(&mut self, fire: CheckFire) -> Result<Dissection, String> {
        if !matches!(self.shell.mode(), crate::Graphs::Off) {
            return Err("a dissection walks its fire, and this load captures its fires".into());
        }
        let drain_ns = match self.drain_ns {
            Some(ns) => ns,
            None => {
                // A drained pair around nothing: what each node's drains
                // cost beside its work.
                crate::dissect::start();
                for _ in 0..64 {
                    crate::dissect::enter();
                    crate::dissect::left("empty");
                }
                let ns = crate::dissect::take()
                    .first()
                    .map_or(0.0, |&(_, ns, n)| ns as f64 / f64::from(n));
                self.drain_ns = Some(ns);
                ns
            }
        };
        crate::dissect::start();
        let fired = self.fire_once(fire);
        let ops = crate::dissect::take();
        let fired = fired?;
        Ok(Dissection {
            wall_ns: fired.ns,
            ops: ops
                .into_iter()
                .map(|(op, ns, n)| (op.to_owned(), ns as f64, n))
                .collect(),
            drain_ns,
        })
    }
}
