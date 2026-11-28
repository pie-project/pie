//! The check of the estimate (`pie tune --check-estimate`), one for every
//! engine: a fire of each rung width a load lays is timed as serving runs it
//! (its recorded body replayed), beside the estimate the load weighs its top
//! by ([`super::FireEstimate`]); the estimate's pick (the load's fire top,
//! [`super::fire_top`]) is held within [`CHECK_BOUND`] of the measured best
//! among every rung timed (to [`PAST_TOP`] times the fire top, so a top set
//! too low shows). The narrowest and widest rungs, both picks and any
//! width asked are dissected: replayed and walked node by node on fresh
//! loads, each node's device time rolled up by the kind of work it is. A
//! rung is dissected as its check fire shapes it ([`RungFacts::check_fire`]):
//! a decode-shaped rung as its decode step (a lane a token), a prompt-shaped
//! one as its prompt, each kind on a load of its own.
//!
//! An engine offers the check its loads through [`Bench`] and [`Loaded`];
//! everything else (the rungs timed, the warm fires, the median, the picks,
//! the report and the verdict) is here.

use std::fmt::Write as _;

use super::{CheckFire, FireEstimate, RungFacts};

/// How far the estimate's pick may fall short of the measured-best rung:
/// its score, under the measured times, within this share of the best's.
pub const CHECK_BOUND: f64 = crate::rates::AGREEMENT;

/// How far past the fire top a check's load lays its rungs: four times it
/// (two octaves of the lattice), so a top set too low fails the check.
pub const PAST_TOP: u32 = 4;

/// The fires timed a rung, after [`WARM_FIRES`].
pub const TIMED_FIRES: usize = 5;

/// The fires run a rung before it is timed.
pub const WARM_FIRES: usize = 2;

/// The widest rung a check's load lays: the lattice point at [`PAST_TOP`]
/// times `fire_top`, or the lattice's widest short of it.
#[must_use]
pub fn laid_top(lattice: &[u32], fire_top: u32) -> u32 {
    let past = u64::from(fire_top) * u64::from(PAST_TOP);
    lattice
        .iter()
        .copied()
        .find(|&rung| u64::from(rung) >= past)
        .or_else(|| lattice.iter().copied().max())
        .unwrap_or(fire_top)
}

/// How a check's load runs its fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Run {
    /// As serving runs them: each rung's recorded body replayed.
    Replayed,
    /// Eagerly, node by node: what a dissection times each node of.
    Walked,
}

/// What a check's load weighed its rungs by.
#[derive(Debug, Clone)]
pub struct Weighed {
    /// The model, as its checks name it ([`super::model_class`]).
    pub model: String,
    /// Every rung the load laid, the infeasible ones too.
    pub rungs: Vec<RungFacts>,
    pub estimate: FireEstimate,
    /// The live range kv and activation share at load.
    pub live: u64,
    /// The model's fire top ([`super::fire_top`]): the widest rung a load
    /// that is no check lays, the estimate's pick over its lattice.
    pub fire_top: u32,
    /// The widest rung the load laid.
    pub laid: u32,
}

/// One fire timed to its landing (ns), and whether it replayed a recorded
/// body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fired {
    pub ns: f64,
    pub replayed: bool,
}

/// One walked fire dissected node by node.
#[derive(Debug, Clone, PartialEq)]
pub struct Dissection {
    /// The fire's wall time while dissected (ns).
    pub wall_ns: f64,
    /// Each op: its name, its nodes' device time (ns, each node drained
    /// before and after) and its count of nodes.
    pub ops: Vec<(String, f64, u32)>,
    /// A drained pair around nothing (ns): what each node's drains add to
    /// its work, taken off every node.
    pub drain_ns: f64,
}

/// The shell a check's load lays: `slots` slots of `context` tokens each
/// (a fire on a fresh slot owns a run of its pages, no runtime page tables
/// between), every rung of the model's lattice to `rows` laid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shell {
    pub slots: u32,
    pub context: u32,
    pub rows: u32,
}

/// An engine's loads of one model, for the check.
///
/// A check fire is exactly the fire serving makes at its shape: it reads
/// back what a serving fire reads and no more. Where serving samples on the
/// device and reads back its tokens (or its programs' readouts), the check
/// reads those; where serving reads each lane's rows to the host, the check
/// reads those rows. A check that reads back more (every lane's full logits
/// row where serving reads tokens) times a fire serving never makes.
/// `replayed` ([`Fired`]) is likewise the fire as serving runs it: a
/// recorded body replayed where the engine records, a walk where it
/// records nothing and serving walks.
pub trait Bench {
    /// A load of the model run as `run`: on `shell` ([`shell_for`] the fires
    /// it runs: the prompts on one slot of the widest prompt, the decode
    /// steps on a slot a lane of a page each), or (`None`) on the shape a
    /// served load is asked, laying every rung to [`laid_top`] of its fire
    /// top: the load whose rungs and estimate the check weighs.
    fn load(&mut self, run: Run, shell: Option<Shell>) -> Result<Box<dyn Loaded + '_>, String>;
}

/// One load of a [`Bench`].
pub trait Loaded {
    /// The rungs it laid and the estimate it weighed them by; refused by
    /// name where the device has no `[rates]` to estimate by.
    fn weighed(&self) -> Result<Weighed, String>;
    /// `fire` once, timed to its landing.
    fn fire(&mut self, fire: CheckFire) -> Result<Fired, String>;
    /// `fire` once, walked, each node's device time taken.
    fn dissect(&mut self, fire: CheckFire) -> Result<Dissection, String>;
}

/// The [`Shell`] a load must hold to fire each of `fires` (of one kind:
/// prompts, or decode steps): its slots the most lanes one fires, its
/// context the keys a lane attends (its new tokens among them, on a fresh
/// slot) in whole pages of `page`, its rungs
/// laid to the widest fire's rows. Prompts and decode steps are fired on
/// shells of their own: one shell for both would lay the decode's lanes
/// each the widest prompt's pages.
#[must_use]
pub fn shell_for(fires: impl IntoIterator<Item = CheckFire>, page: u32) -> Shell {
    let fold = |shell: Shell, f: CheckFire| Shell {
        slots: shell.slots.max(f.lanes),
        context: shell.context.max(f.keys_per_lane.max(f.tokens_per_lane)),
        rows: shell.rows.max(f.rows),
    };
    let shell = fires.into_iter().fold(
        Shell {
            slots: 1,
            context: 1,
            rows: 1,
        },
        fold,
    );
    Shell {
        context: shell.context.next_multiple_of(page.max(1)),
        ..shell
    }
}

/// The page a check's shells round their context to.
const PAGE: u32 = 16;

/// One prompt of `width` tokens on a fresh slot: the dissection's fire.
#[must_use]
pub fn prompt(width: u32) -> CheckFire {
    CheckFire {
        rows: width,
        lanes: 1,
        tokens_per_lane: width,
        keys_per_lane: width,
        sampled: 1,
    }
}

/// `fire` timed as a rung's fires are: [`WARM_FIRES`] warm, then the median
/// of [`TIMED_FIRES`] back to back; and whether every timed fire replayed.
pub fn timed(load: &mut dyn Loaded, fire: CheckFire) -> Result<Fired, String> {
    for _ in 0..WARM_FIRES {
        load.fire(fire)?;
    }
    let mut times = Vec::with_capacity(TIMED_FIRES);
    let mut replayed = true;
    for _ in 0..TIMED_FIRES {
        let f = load.fire(fire)?;
        replayed &= f.replayed;
        times.push(f.ns);
    }
    times.sort_by(f64::total_cmp);
    Ok(Fired {
        ns: times[times.len() / 2],
        replayed,
    })
}

/// The kind of work an op is, for the dissection's rollup.
#[must_use]
pub fn kind_of(op: &str) -> &'static str {
    let moe = op.contains("moe") || op.contains("expert");
    match op.split('.').next().unwrap_or_default() {
        "linear" if moe => "gemm (experts)",
        "linear" => "gemm (dense)",
        "fused" if op.contains("anchored") => "gemm (dense)",
        "attn" => "attention",
        "ssm" | "conv" => "recurrent",
        "sampling" => "sampling/readout",
        "gather" => "gather/embed",
        "collective" => "collective",
        _ => "norms/elementwise",
    }
}

/// What a check found: the report, and whether the estimate's pick fell
/// past [`CHECK_BOUND`].
#[derive(Debug, Clone, PartialEq)]
pub struct Checked {
    pub model: String,
    pub report: String,
    /// The estimate's pick and the measured best (tokens).
    pub picks: (u32, u32),
    /// How far below the measured best's score the estimate's pick scores.
    pub short: f64,
}

impl Checked {
    #[must_use]
    pub fn passes(&self) -> bool {
        self.short <= CHECK_BOUND
    }

    /// The refusal of a pick past the bound, with the report.
    #[must_use]
    pub fn failure(&self) -> String {
        format!(
            "the estimate picks rung {} for {}, {:.1}% below the measured best {}: past the \
             {:.0}% bound\n{}",
            self.picks.0,
            self.model,
            100.0 * self.short,
            self.picks.1,
            100.0 * CHECK_BOUND,
            self.report
        )
    }
}

/// The feasible rungs of `weighed` the check times: those the load laid.
fn timed_rungs(weighed: &Weighed) -> Vec<RungFacts> {
    weighed
        .rungs
        .iter()
        .filter(|r| r.infeasible.is_none() && r.tokens <= weighed.laid)
        .copied()
        .collect()
}

/// The check of the estimate on `bench`'s model: every rung timed, the two
/// picks weighed, `also` (rung widths) dissected beside the narrowest, the
/// widest and both picks, each as its check fire shapes it (a decode-shaped
/// rung as decode). `say` hears the report as it is made; the verdict is
/// [`Checked::passes`].
pub fn check(
    bench: &mut dyn Bench,
    also: &[u32],
    say: &mut dyn FnMut(&str),
) -> Result<Checked, String> {
    // The rungs, the estimate and the live range come of one load on the
    // served shape; each kind of fire (decode steps, prompts) is then timed
    // on a load of its own, sized from its fires.
    let weighed = bench.load(Run::Replayed, None)?.weighed()?;
    let rungs = timed_rungs(&weighed);
    let model = weighed.model.clone();
    say(&format!(
        "timing a fire at each of {} rung widths of {model} (laid to {}, {PAST_TOP}x its fire top \
         {})",
        rungs.len(),
        weighed.laid,
        weighed.fire_top,
    ));
    let mut meas = vec![0.0; rungs.len()];
    for decode in [true, false] {
        let kind: Vec<usize> = (0..rungs.len())
            .filter(|&k| rungs[k].decode_shaped() == decode)
            .collect();
        if kind.is_empty() {
            continue;
        }
        let shell = shell_for(kind.iter().map(|&k| rungs[k].check_fire()), PAGE);
        say(&format!(
            "  {} {} on {} slots of {} tokens, rungs to {}",
            kind.len(),
            if decode { "decode steps" } else { "prompts" },
            shell.slots,
            shell.context,
            shell.rows
        ));
        let mut load = bench.load(Run::Replayed, Some(shell))?;
        for &k in &kind {
            let f = timed(&mut *load, rungs[k].check_fire())?;
            if !f.replayed {
                return Err(format!(
                    "a fire of {} tokens of {model} was walked, not replayed: its time is not \
                     what serving runs, so it is no measurement",
                    rungs[k].tokens
                ));
            }
            meas[k] = f.ns;
        }
    }
    let e = &weighed.estimate;
    let est: Vec<f64> = rungs.iter().map(|r| e.ns(r)).collect();
    // Both picks are the fire-top rule's over every rung timed: the estimate's
    // argmax (efficiency times live capacity) against the measured times'.
    // No cap stands between them: the load's fire top is this pick, one
    // function (`super::fire_top`), and a top set too low shows as a
    // measured best past it.
    let est_effs: Vec<f64> = rungs.iter().map(|r| e.eff(r)).collect();
    let meas_effs = super::measured_effs(&rungs, &meas);
    let by_est = super::pick(&rungs, &est_effs, weighed.live).map_err(|e| e.to_string())?;
    let by_meas = super::pick(&rungs, &meas_effs, weighed.live).map_err(|e| e.to_string())?;
    let scores = super::scored(&rungs, &meas_effs, weighed.live);
    let score = |tokens: u32| {
        scores
            .iter()
            .find(|&&(t, _)| t == tokens)
            .map_or(0.0, |&(_, s)| s)
    };
    let (at_est, best) = (score(by_est.tokens), score(by_meas.tokens));
    let short = if best > 0.0 { 1.0 - at_est / best } else { 0.0 };
    let mut out = format!(
        "{model}: rung, measured ms, estimated ms, measured/estimated, measured tok/s, estimated \
         tok/s; the estimate = work + the launch floors' share (each dispatch its work or a \
         launch) (ms): floor, work; work's terms: weight read, matmul, readout, attention, kv, \
         recurrent, elementwise\n"
    );
    for (k, r) in rungs.iter().enumerate() {
        let tokens = f64::from(r.tokens);
        let t = e.terms(r);
        let _ = writeln!(
            out,
            "  {:>6} {:>9.2} {:>9.2} {:>6.2}x {:>9.0} {:>9.0}   {:.2} {:.2}; {:.2} {:.2} {:.2} \
             {:.2} {:.2} {:.2} {:.2}",
            r.tokens,
            meas[k] / 1e6,
            est[k] / 1e6,
            meas[k] / est[k],
            tokens / meas[k] * 1e9,
            tokens / est[k] * 1e9,
            t.floor() / 1e6,
            t.work() / 1e6,
            t.weights / 1e6,
            t.flops / 1e6,
            t.readout / 1e6,
            t.attention / 1e6,
            t.kv / 1e6,
            t.recurrent / 1e6,
            t.elementwise / 1e6,
        );
    }
    out.push_str(&super::shape_table(e, &rungs, &meas));
    let _ = writeln!(
        out,
        "  pick by the estimate {}, by the measured times {}; the estimate's scores {:.1}% below \
         the best (bound {:.0}%)",
        by_est.tokens,
        by_meas.tokens,
        100.0 * short,
        100.0 * CHECK_BOUND
    );
    say(&out);

    let mut widths: Vec<u32> = vec![by_est.tokens, by_meas.tokens];
    widths.extend(rungs.first().map(|r| r.tokens));
    widths.extend(rungs.last().map(|r| r.tokens));
    widths.extend(also.iter().copied());
    widths.sort_unstable();
    widths.dedup();
    let dissected: Vec<(usize, u32)> = widths
        .iter()
        .filter_map(|&t| Some((rungs.iter().position(|r| r.tokens == t)?, t)))
        .collect();
    let launch_ns = e.rates.launch_ns;
    // Each kind of fire (decode steps, prompts) is dissected on loads of its
    // own, sized from its fires, as the rung loop timed it.
    for decode in [true, false] {
        let kind: Vec<(usize, u32)> = dissected
            .iter()
            .copied()
            .filter(|&(k, _)| rungs[k].decode_shaped() == decode)
            .collect();
        if kind.is_empty() {
            continue;
        }
        let shell = shell_for(kind.iter().map(|&(k, _)| rungs[k].check_fire()), PAGE);
        // The rung loop's fires ran back to back across every rung, and a
        // card's clocks fall as it heats: a walk timed after a fresh load
        // would meet a cooler card. Each dissected fire is replayed again on
        // a fresh load, as its walk is, by the same count of fires.
        let mut replayed = Vec::with_capacity(kind.len());
        {
            let mut load = bench.load(Run::Replayed, Some(shell))?;
            for &(k, width) in &kind {
                let f = timed(&mut *load, rungs[k].check_fire())?;
                if !f.replayed {
                    return Err(format!(
                        "a fire of {width} tokens of {model} was walked, not replayed, on the \
                         dissection's replay load"
                    ));
                }
                replayed.push(f.ns);
            }
        }
        let mut load = bench.load(Run::Walked, Some(shell))?;
        for (&(k, width), replayed) in kind.iter().zip(replayed) {
            let fire = rungs[k].check_fire();
            let walk = timed(&mut *load, fire)?.ns;
            let d = load.dissect(fire)?;
            say(&dissection(
                width,
                (meas[k], replayed, est[k], walk),
                rungs[k].launches,
                launch_ns,
                d,
            ));
        }
    }
    Ok(Checked {
        model,
        report: out,
        picks: (by_est.tokens, by_meas.tokens),
        short,
    })
}

/// A dissected fire of `width` tokens, rolled up by kind: `times` its time
/// in the rung loop, replayed on a fresh load, estimated and walked (ns);
/// `launches` the estimate's launches at it.
fn dissection(
    width: u32,
    (timed, replayed, estimated, walk): (f64, f64, f64, f64),
    launches: u32,
    launch_ns: f64,
    d: Dissection,
) -> String {
    let mut ops = d.ops;
    ops.sort_by(|a, b| b.1.total_cmp(&a.1));
    let nodes: u32 = ops.iter().map(|row| row.2).sum();
    let net = |ns: f64, n: u32| (ns - d.drain_ns * f64::from(n)).max(0.0);
    let device: f64 = ops.iter().map(|(_, ns, n)| net(*ns, *n)).sum();
    let mut kinds: Vec<(&str, f64, u32)> = Vec::new();
    for (op, ns, n) in &ops {
        let kind = kind_of(op);
        match kinds.iter_mut().find(|k| k.0 == kind) {
            Some(k) => {
                k.1 += net(*ns, *n);
                k.2 += n;
            }
            None => kinds.push((kind, net(*ns, *n), *n)),
        }
    }
    kinds.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut out = format!(
        "dissected: a fire of {width} tokens: replayed {:.2} ms ({:.2} ms in the rung loop), \
         estimated {:.2} ms; walked {:.2} ms ({:.2} ms dissected), {nodes} nodes, {:.2} ms in \
         them past the drains ({:.1} us a drained pair); the estimate's {launches} launches x \
         {launch_ns:.0} ns = {:.2} ms\n",
        replayed / 1e6,
        timed / 1e6,
        estimated / 1e6,
        walk / 1e6,
        d.wall_ns / 1e6,
        device / 1e6,
        d.drain_ns / 1e3,
        f64::from(launches) * launch_ns / 1e6,
    );
    for (kind, ns, n) in &kinds {
        let _ = writeln!(
            out,
            "  {:>9.3} ms {:>5.1}%  x{n:<5} {kind}",
            ns / 1e6,
            100.0 * ns / device.max(1.0)
        );
    }
    out.push_str("  by op:\n");
    for (op, ns, n) in ops.iter().take(14) {
        let _ = writeln!(out, "    {:>9.3} ms  x{n:<5} {op}", net(*ns, *n) / 1e6);
    }
    out
}
