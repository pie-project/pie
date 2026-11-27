//! The check of the estimate on a slang backend (`engine::load::check`):
//! its loads of one model, each laying its rungs past the fire top
//! ([`crate::serve::Boot::check`]), each check fire fired on fresh slots and
//! timed to its landing, and a walked fire dissected by the backend's
//! per-dispatch device timestamps.

use crate::Instant;
use std::path::{Path, PathBuf};

use engine::load::CheckFire;
use engine::load::check::{Dissection, Fired, Run, Shell as CheckShell, Weighed};
use poem_ir::{Platform, Request, Trace};

use crate::api::ContractFor;
use crate::serve::{Boot, CheckRungs, Lane, Seated, Serve, Shell};

/// The widest fire a check's load is asked: [`engine::load::check::PAST_TOP`]
/// times the widest row count the rates step measures, past any fire top a
/// device's rates can set (the binding bounds it below where it must).
pub const ASKED_TOP: u32 =
    engine::load::check::PAST_TOP * engine::rates::GEMM_ROWS[engine::rates::GEMM_ROWS.len() - 1];

/// The lanes a served load is asked where a check names no shell: the
/// compiler's default budget's.
fn served_lanes() -> u32 {
    poem_compiler::Budget::default().max_lanes
}

/// The context a served-shape load is asked: one long prompt's, so the
/// fire top it weighs is the model's, not a short context's.
const SERVED_CONTEXT: u32 = 32_768;

/// The page every check load pages its kv by.
const PAGE: u32 = 16;

/// How a backend times a walked fire's dispatches on the device: its
/// per-dispatch timestamps switched on and off, cleared, and read as
/// `(kernel, device ns, dispatches)`.
#[derive(Clone, Copy)]
pub struct Probe {
    pub timing: fn(bool),
    pub reset: fn(),
    pub rows: fn() -> Vec<(String, u64, u64)>,
}

/// A slang backend's loads of one model for the check. Its loads report a
/// fire `replayed` when it ran as serving runs it: a recorded body replayed
/// on a backend that records (Vulkan), a walk on one that records nothing
/// (wgpu), where serving walks every fire.
pub struct Bench<B: Serve> {
    artifact: PathBuf,
    trace: Trace,
    contract: checkpoint::contract::ModelContract,
    device: B::DeviceBoot,
    probe: Probe,
}

impl<B: Serve> Bench<B> {
    /// The loads of `deployment` (a catalog row) from `artifact`, on the
    /// device `device` opens, its trace for `platform`.
    pub fn new(
        artifact: &Path,
        deployment: &str,
        platform: Platform,
        contract_for: ContractFor,
        device: B::DeviceBoot,
        probe: Probe,
    ) -> Result<Bench<B>, String> {
        let row = poem_compiler::catalog::deployment(deployment)
            .ok_or_else(|| format!("no catalog row `{deployment}`"))?;
        let trace = row.trace(platform);
        let contract = contract_for(&trace, artifact)?;
        Ok(Bench {
            artifact: artifact.to_path_buf(),
            trace,
            contract,
            device,
            probe,
        })
    }
}

impl<B: Serve> engine::load::check::Bench for Bench<B> {
    fn load(
        &mut self,
        run: Run,
        shell: Option<CheckShell>,
    ) -> Result<Box<dyn engine::load::check::Loaded + '_>, String> {
        // The served-shape load fires nothing: it weighs the served lanes'
        // rungs on one slot of a long prompt's context, so it reserves no
        // state for slots it never seats.
        let (lanes, slots, context, rungs) = match shell {
            None => (served_lanes(), 1, SERVED_CONTEXT, CheckRungs::PastTop),
            Some(shell) => (
                shell.slots,
                shell.slots,
                shell.context,
                CheckRungs::To(shell.rows),
            ),
        };
        let mut loaded = Shell::<B>::load(Boot {
            trace: self.trace.clone(),
            contract: &self.contract,
            checkpoint: &self.artifact,
            budget: poem_compiler::Budget::new(lanes, ASKED_TOP),
            patches: None,
            voxels: None,
            profile: None,
            page_size: PAGE,
            context: context.next_multiple_of(PAGE),
            slots,
            runahead: engine::runahead::Runahead::F1,
            device: &self.device,
            residency: engine::load::Residency::default(),
            handed: None,
            check: Some(rungs),
            places_programs: false,
        })
        .map_err(|e| e.to_string())?;
        // No runtime commits segments for a check's fires as residency
        // grows: the load backs the whole pool before it fires.
        let segments = loaded
            .chunk_facts()
            .map_err(|e| e.to_string())?
            .segments
            .max;
        loaded
            .commit_segments(segments)
            .map_err(|e| e.to_string())?;
        if run == Run::Walked {
            loaded.walk_every_fire();
        }
        Ok(Box::new(Loaded {
            shell: loaded,
            probe: self.probe,
        }))
    }
}

/// One load of a [`Bench`]. A fire it reports `replayed` is the fire as
/// serving runs it: on a backend that records (Vulkan) its recorded body
/// replayed; on one that records nothing (wgpu) a walk, as serving walks.
pub struct Loaded<B: Serve> {
    shell: Shell<B>,
    probe: Probe,
}

impl<B: Serve> Loaded<B> {
    /// `fire` once on fresh slots, its time to its landing (ns), and
    /// whether any of it was walked.
    fn fired(&mut self, fire: CheckFire) -> Result<(f64, bool), String> {
        if fire.keys_per_lane != fire.tokens_per_lane {
            return Err(format!(
                "a check fire of {} keys a lane over {} new tokens: a fresh slot's lane attends \
                 its own tokens alone",
                fire.keys_per_lane, fire.tokens_per_lane
            ));
        }
        let tokens: Vec<u32> = (0..fire.tokens_per_lane)
            .map(|i| 100 + (i * 7919) % 5000)
            .collect();
        for slot in 0..fire.lanes {
            self.shell.open(slot).map_err(|e| e.to_string())?;
        }
        let word = self
            .shell
            .trace()
            .facts
            .word(&Request::new(fire.tokens_per_lane, false));
        // The fire serving makes: every lane's rows laid, and none read
        // back by the host, since a serving lane's rows are read on the
        // device by the program attached to it (or by nothing).
        let lanes: Vec<Seated<'_, B>> = (0..fire.lanes)
            .map(|slot| Seated {
                hosts: false,
                ..Seated::of(Lane {
                    slot,
                    word,
                    tokens: &tokens,
                })
            })
            .collect();
        self.shell.recorded_serving();
        self.shell.unkept_serving();
        let began = Instant::now();
        self.shell.fire_seated(&lanes).map_err(|e| e.to_string())?;
        let ns = began.elapsed().as_secs_f64() * 1e9;
        let walked = self.shell.recorded_serving() + self.shell.unkept_serving() > 0;
        Ok((ns, walked))
    }
}

impl<B: Serve> engine::load::check::Loaded for Loaded<B> {
    fn weighed(&self) -> Result<Weighed, String> {
        self.shell.weighed().cloned().ok_or_else(|| {
            format!(
                "device {} has no complete [rates] in its profile, so nothing weighs its rungs; \
                 `pie tune <backend> --rates` measures them",
                crate::backend::Context::name(&self.shell.device)
            )
        })
    }

    /// `replayed` means the fire as serving runs it: a recorded body
    /// replayed where the backend records, and a walk where it records
    /// nothing (wgpu), since serving walks every fire there.
    fn fire(&mut self, fire: CheckFire) -> Result<Fired, String> {
        let (ns, walked) = self.fired(fire)?;
        Ok(Fired {
            ns,
            replayed: !walked || !B::RECORDS,
        })
    }

    /// Each dispatch timed on the device by its own timestamps around it,
    /// with no drain between: `drain_ns` is nothing.
    fn dissect(&mut self, fire: CheckFire) -> Result<Dissection, String> {
        (self.probe.reset)();
        (self.probe.timing)(true);
        let fired = self.fired(fire);
        (self.probe.timing)(false);
        let (wall_ns, walked) = fired?;
        if !walked {
            return Err(
                "a dissected fire replayed a recording: its dispatches carry no timestamps"
                    .to_owned(),
            );
        }
        // A frame's own span (where the backend times one) is no op.
        let ops = (self.probe.rows)()
            .into_iter()
            .filter(|(kernel, ..)| !kernel.starts_with("frame."))
            .map(|(kernel, ns, n)| {
                (
                    op_of(&kernel),
                    ns as f64,
                    u32::try_from(n).unwrap_or(u32::MAX),
                )
            })
            .collect();
        (self.probe.reset)();
        Ok(Dissection {
            wall_ns,
            ops,
            drain_ns: 0.0,
        })
    }
}

/// A kernel's entry point named as the op it runs, in the words
/// `engine::load::check::kind_of` rolls ops up by.
#[must_use]
pub fn op_of(kernel: &str) -> String {
    let has = |words: &[&str]| words.iter().any(|w| kernel.contains(w));
    let kind = if has(&["gemm", "qmm", "matmul", "matvec", "gemv"]) {
        "linear"
    } else if has(&["attn", "sdpa"]) {
        "attn"
    } else if has(&["ssm", "scan", "conv", "delta"]) {
        "ssm"
    } else if has(&["sampl", "argmax", "topk", "top_k", "logits"]) {
        "sampling"
    } else if has(&["gather", "embed"]) {
        "gather"
    } else {
        return kernel.to_owned();
    };
    format!("{kind}.{kernel}")
}
