//! The end-to-end search `pie tune` runs on every backend: decisions a
//! kernel bench cannot price are tried on served models, one at a time.

use std::fmt::Write as _;

use crate::profile::Profile;
use crate::spec::{Kind, Shape};

/// The serving every backend times: a prompt of [`PROMPT`] tokens, then
/// [`STEPS`] decode steps at each lane count in [`LANES`], in a context of
/// [`CONTEXT`].
pub const PROMPT: u32 = 512;
pub const CONTEXT: u32 = 1024;
pub const STEPS: usize = 16;
pub const LANES: [usize; 3] = [1, 4, 8];

/// Tuned decisions as `(decision id, TOML value)`.
pub type Entries = Vec<(String, String)>;

/// The fires `pie tune` times at each rung width beside the one before
/// them, which compiles or records: the median is kept.
pub const FIRE_REPS: usize = 3;

/// `(tokens, ns)` of a fire at each rung width, ascending in tokens: what a
/// check of the estimate measures. Never a profile's.
pub type FireTable = Vec<(u32, u64)>;

/// The fires a check of the estimate times on every backend: a fire of
/// each width in `widths` (the rungs its load lays, in tokens), `fire`
/// timing one in nanoseconds (told whether it is timed: the first a width,
/// which records, is not); the median of [`FIRE_REPS`] is kept.
pub fn time_fires(
    widths: &[u32],
    mut fire: impl FnMut(u32, bool) -> Result<u64, String>,
    say: &mut dyn FnMut(&str),
) -> Result<FireTable, String> {
    let mut widths = widths.to_vec();
    widths.sort_unstable();
    widths.dedup();
    let mut table = Vec::with_capacity(widths.len());
    for width in widths {
        fire(width, false)?;
        let mut times = (0..FIRE_REPS)
            .map(|_| fire(width, true))
            .collect::<Result<Vec<u64>, String>>()?;
        times.sort_unstable();
        let ns = times[times.len() / 2].max(1);
        say(&format!(
            "  a fire of {width} tokens: {:.2} ms",
            ns as f64 / 1e6
        ));
        table.push((width, ns));
    }
    Ok(table)
}

#[derive(Clone, Debug)]
pub struct Line {
    pub id: &'static str,
    pub text: String,
    /// The run measured it: compared it on a model that reads it, and
    /// changed it or kept it under a bar of [`crate::AGREEMENT`] or less. A
    /// tuned file's `[provenance] measured` lists these alone.
    pub measured: bool,
}

#[derive(Clone, Debug)]
pub struct Serving {
    /// Milliseconds of each timed prompt.
    pub prompt: Vec<f64>,
    /// Milliseconds of each timed step, per lane count in [`LANES`].
    pub steps: Vec<Vec<f64>>,
    /// The prompt's readout row.
    pub readout: Vec<f32>,
    /// The decisions the load read ([`Profile::read`]).
    pub read: Vec<&'static str>,
    /// Why a co-processor refused its programs, if it did: the load then
    /// served without it.
    pub refused: Option<String>,
}

/// What a backend gives the search.
pub trait Bench {
    fn models(&self) -> Vec<String>;
    /// The complete profile `entries` lay over today's, checked against
    /// the device.
    fn file(&self, entries: &[(String, String)]) -> Result<String, String>;
    fn serve(&mut self, model: usize, file: &str) -> Result<Serving, String>;
    /// A fingerprint of the plan model `model` compiles under `file`:
    /// equal fingerprints serve the same. `None` where it cannot say.
    fn plan(&self, model: usize, file: &str) -> Option<u64>;
    /// Whether [`Bench::plan`] covers every decision, not only the
    /// plan-only ones (a price, a fusion): its fingerprint is all the
    /// load would run (XLA's program texts), so a candidate whose
    /// fingerprint equals today's serves the same whatever its decision,
    /// and a decision no check model's plan moves is not served at all.
    fn plan_covers_all(&self) -> bool {
        false
    }
}

/// One decision the search tries.
#[derive(Clone, Debug)]
pub struct Knob {
    pub id: &'static str,
    /// The value the profile holds now, as TOML.
    pub current: String,
    /// `(label, overrides)`: the value as a reader sees it, and the TOML.
    pub candidates: Vec<(String, String)>,
    /// Whether a number in it ranges freely, so halfway between two tried
    /// values is a value too (not so for a set).
    pub halfway: bool,
    /// It can only matter by changing the compiled plan (a price, a
    /// fusion), so a candidate whose plan is unchanged is not served.
    pub plan_only: bool,
}

/// The decisions the search tries on `base`'s registry, in order: kernels,
/// the co-processor (`refusable`), the engine, fusions, then prices. `skip`
/// names decisions measured another way or meaningless on this backend; a
/// `never` decision is never touched.
#[must_use]
pub fn knobs(base: &Profile, skip: &[&str], refusable: &str) -> Vec<Knob> {
    let mut out: Vec<Knob> = Vec::new();
    for (spec, rungs, _) in base.entries() {
        if spec.tune == crate::Tune::Never || skip.contains(&spec.id) {
            continue;
        }
        let (domain, key) = spec.id.split_once('.').expect("an id is domain.name");
        let set = |value: &str| format!("[{domain}]\n{key} = {value}\n");
        let current = crate::render::value_toml(spec, rungs);
        let mut values: Vec<String> = Vec::new();
        let mut ranged = false;
        // A whole number is tried at half to double in quarter steps: fine
        // enough to place a share such as the Neural Engine's; a price only
        // at half and double, since its plan moves in steps anyway.
        let scaled = |v: f64, lo: f64, hi: f64, int: bool| -> Vec<f64> {
            let steps: &[f64] = if int {
                &[0.5, 0.75, 1.25, 1.5, 2.0]
            } else {
                &[0.5, 2.0]
            };
            steps
                .iter()
                .map(|m| (v * m).clamp(lo, hi))
                .map(|x| if int { x.round() } else { x })
                .filter(|x| (x - v).abs() > f64::EPSILON)
                .collect()
        };
        // The values a number is tried at, around `from`: every value of a
        // set, or half to double of a range.
        let around = |kind: Kind, from: &str| -> Vec<String> {
            let from: f64 = from.replace('_', "").parse().unwrap_or(1.0);
            match kind {
                Kind::Ints(set) => set.iter().map(i64::to_string).collect(),
                Kind::Int { lo, hi } => scaled(from, lo as f64, hi as f64, true)
                    .into_iter()
                    .map(|x| (x as i64).to_string())
                    .collect(),
                Kind::Float { lo, hi } => scaled(from, lo, hi, false)
                    .into_iter()
                    .map(|x| format!("{x:.1}"))
                    .collect(),
                _ => Vec::new(),
            }
        };
        match spec.shape {
            Shape::Flag => {
                if !spec.vars.is_empty() {
                    continue;
                }
                values.push(if current == "false" { "true" } else { "false" }.to_string());
            }
            Shape::Constant(c) => match c.kind {
                Kind::Ints(_) => values.extend(around(c.kind, &current)),
                Kind::Int { .. } | Kind::Float { .. } if spec.vars.is_empty() => {
                    ranged = matches!(c.kind, Kind::Int { .. });
                    values.extend(around(c.kind, &current));
                }
                _ => continue,
            },
            Shape::Choice(alternatives) => {
                if !spec.vars.is_empty() {
                    continue;
                }
                // The alternative the profile holds, and its constants.
                let held = match rungs.first().map(|r| &r.pick) {
                    Some(crate::Pick::Alternative { index, constants }) => {
                        Some((*index, constants.clone()))
                    }
                    _ => None,
                };
                for (at, alt) in alternatives.iter().enumerate() {
                    if alt.constants.is_empty() {
                        values.push(format!("\"{}\"", alt.name));
                        continue;
                    }
                    // The constants it starts from: the held ones for the held
                    // alternative, the defaults for another.
                    let base: Vec<String> = match &held {
                        Some((index, constants)) if *index == at => {
                            constants.iter().map(crate::Value::toml).collect()
                        }
                        _ => alt
                            .constants
                            .iter()
                            .map(|c| c.default.to_string())
                            .collect(),
                    };
                    let body = |with: &[String]| {
                        let fields: Vec<String> = alt
                            .constants
                            .iter()
                            .zip(with)
                            .map(|(c, v)| format!("{} = {v}", c.name))
                            .collect();
                        format!("{{ use = \"{}\", {} }}", alt.name, fields.join(", "))
                    };
                    values.push(body(&base));
                    // One constant at a time around where it starts.
                    for (i, c) in alt.constants.iter().enumerate() {
                        let options: Vec<String> = match c.kind {
                            Kind::Bool => vec![(base[i] != "true").to_string()],
                            Kind::Word(names) => names.iter().map(|n| format!("\"{n}\"")).collect(),
                            kind => {
                                ranged |= matches!(kind, Kind::Int { .. });
                                around(kind, &base[i])
                            }
                        };
                        for o in options {
                            let mut with = base.clone();
                            with[i] = o;
                            values.push(body(&with));
                        }
                    }
                }
            }
        }
        let normal = |v: &str| v.replace(' ', "");
        let mut seen: Vec<String> = Vec::new();
        values.retain(|v| {
            let fresh = !seen.contains(&normal(v));
            seen.push(normal(v));
            fresh
        });
        let candidates: Vec<(String, String)> = values
            .into_iter()
            .filter(|v| normal(v) != normal(&current))
            .map(|v| (v.clone(), set(&v)))
            .collect();
        if candidates.is_empty() {
            continue;
        }
        let plan_only = matches!(spec.tune, crate::Tune::Price) || domain == "fuse";
        out.push(Knob {
            id: spec.id,
            current,
            candidates,
            halfway: ranged,
            plan_only,
        });
    }
    let rank = |id: &str| match id.split('.').next() {
        _ if !refusable.is_empty() && id.starts_with(refusable) => 1,
        Some("kernel") => 0,
        Some("engine") => 2,
        Some("fuse") => 3,
        _ => 4,
    };
    out.sort_by_key(|k| rank(k.id));
    out
}

/// The complete profile `pie tune` writes: `base` with the tuned
/// decisions replaced, its target `device`, and a provenance that says how
/// it was made. The file assumes what `base` does, and each `assume` fact
/// this device has that `base` leaves open. Read back strictly and resolved
/// against `facts` before it is returned, so a file that would be refused is
/// never written.
pub fn assemble(
    registry: &crate::Registry,
    base: &Profile,
    device: &str,
    facts: &crate::Facts,
    tuned: &str,
    provenance: &[(&str, String)],
    assume: &[&str],
) -> Result<String, String> {
    use crate::{Source, Strict};
    let mut target = base.target().clone();
    target.device = Some(device.to_string());
    for name in assume {
        let Some(i) = registry.facts.iter().position(|v| v.name == *name) else {
            continue;
        };
        if let Some(v) = facts.get(name)
            && !target.assumes.iter().any(|&(j, _)| j == i)
        {
            target.assumes.push((i, v));
        }
    }
    let path = format!("tuned {device}");
    let profile = registry
        .overlay(base, tuned, target, Source::User(path.clone()))
        .map_err(|e| e.to_string())?;
    // A tuned file is a decision profile: the keyed measurements a stored
    // or committed layer gave `base` (`measured_on`, `measured_by`, the
    // dense kernels' curves and settles) are a device's own, never written
    // into it; a load of the device takes them from their own layers.
    let profile = {
        let mut tuned = profile.tuned().clone();
        tuned.rates.measured_on = None;
        tuned.rates.measured_by = None;
        tuned.rates.kernel_curves = None;
        tuned.rates.dense_settled = Vec::new();
        tuned.committed_settle = Vec::new();
        profile.with_tuned(tuned)
    };
    let text = crate::render::dump_with(&profile, provenance);
    registry
        .parse(&text, &path, Strict::CheckedIn)
        .map_err(|e| format!("the tuned file does not read back: {e}"))?;
    let (_, report) = registry.resolve(
        &[crate::Candidate {
            source: Source::User(path.clone()),
            text: text.clone(),
        }],
        facts,
    );
    if report.used != Source::User(path) {
        return Err(format!(
            "the tuned file is refused on this device: {:?}",
            report.rejected
        ));
    }
    Ok(text)
}

/// `(decision id, TOML value)` entries as profile TOML, a table a domain.
#[must_use]
pub fn render_overrides(entries: &[(String, String)]) -> String {
    let mut domains: Vec<&str> = Vec::new();
    for (id, _) in entries {
        let d = id.split('.').next().unwrap_or_default();
        if !domains.contains(&d) {
            domains.push(d);
        }
    }
    let mut out = String::new();
    for d in domains {
        let _ = writeln!(out, "[{d}]");
        for (id, value) in entries
            .iter()
            .filter(|(id, _)| id.split('.').next() == Some(d))
        {
            let _ = writeln!(out, "{} = {value}", &id[d.len() + 1..]);
        }
    }
    out
}

/// `entries` with `id` set to `value`, in place or appended.
fn with_entry(entries: &[(String, String)], id: &str, value: &str) -> Entries {
    let mut out = entries.to_vec();
    match out.iter_mut().find(|(i, _)| i == id) {
        Some(e) => e.1 = value.to_string(),
        None => out.push((id.to_string(), value.to_string())),
    }
    out
}

/// Halfway from `best` toward its nearest tried neighbours, for a
/// candidate that differs from them in one whole number (`percent = 71`
/// between 47 and 94): what a second look at a numeric decision tries.
/// Candidates already tried, and labels with fractions, give nothing.
fn halfway(best: &str, tried: &[(String, bool)]) -> Vec<String> {
    // The label as text with its numbers cut out, and the numbers.
    fn split(label: &str) -> (Vec<String>, Vec<i64>) {
        let mut text = vec![String::new()];
        let mut numbers = Vec::new();
        let mut digits = String::new();
        for c in label.chars().chain(std::iter::once(' ')) {
            if c.is_ascii_digit() {
                digits.push(c);
            } else {
                if !digits.is_empty() {
                    numbers.push(digits.parse().unwrap_or(0));
                    digits.clear();
                    text.push(String::new());
                }
                text.last_mut().expect("a piece").push(c);
            }
        }
        (text, numbers)
    }
    if best.contains('.') {
        return Vec::new();
    }
    let (text, numbers) = split(best);
    let mut out = Vec::new();
    for at in 0..numbers.len() {
        let v = numbers[at];
        let others: Vec<i64> = tried
            .iter()
            .map(|(l, _)| split(l))
            .filter(|(t, n)| *t == text && n.len() == numbers.len())
            .filter(|(_, n)| (0..n.len()).all(|i| i == at || n[i] == numbers[i]))
            .map(|(_, n)| n[at])
            .collect();
        let below = others.iter().copied().filter(|&x| x < v).max();
        let above = others.iter().copied().filter(|&x| x > v).min();
        for neighbour in [below, above].into_iter().flatten() {
            let mid = (v + neighbour) / 2;
            if mid == v || mid == neighbour || others.contains(&mid) {
                continue;
            }
            let mut n = numbers.clone();
            n[at] = mid;
            let mut label = String::new();
            for (i, piece) in text.iter().enumerate() {
                label.push_str(piece);
                if let Some(x) = n.get(i) {
                    label.push_str(&x.to_string());
                }
            }
            let label = label.trim_end().to_string();
            if !tried.iter().any(|(l, _)| *l == label) && !out.contains(&label) {
                out.push(label);
            }
        }
    }
    out
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
}

/// What a serving's time is weighed by: the prompt and the one-lane step
/// 0.3 each, the 4- and 8-lane steps 0.2 ([`Serving::measures`]'s order).
const WEIGHTS: [f64; 4] = [0.3, 0.3, 0.2, 0.2];

impl Serving {
    /// The prompt's timings, then each lane count's steps.
    fn measures(&self) -> [&[f64]; 4] {
        [&self.prompt, &self.steps[0], &self.steps[1], &self.steps[2]]
    }
}

/// One serving's weighted time against another's ([`WEIGHTS`]).
fn relative(got: &Serving, against: &Serving) -> f64 {
    let m = |v: &[f64]| median(v.to_vec());
    let (g, a) = (got.measures(), against.measures());
    (0..4).map(|i| WEIGHTS[i] * m(g[i]) / m(a[i])).sum()
}

/// A median's standard error relative to it, from the timings it is the
/// median of: `sqrt(pi / 2)` times the timings' standard deviation over
/// the root of their count (the median of normal noise). Zero where fewer
/// than two timings say nothing of their spread.
fn median_error(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let m = median(v.to_vec());
    if m > 0.0 {
        (std::f64::consts::PI / 2.0).sqrt() * sd / n.sqrt() / m
    } else {
        0.0
    }
}

/// What two servings' own timings can tell apart, as a relative error of
/// one against the other: each measure's two median errors joined (their
/// noise is independent), `(of the weighted time, of the worst measure)`.
/// A comparison holds a change to this at the least, where the incumbent's
/// drift between two servings says less.
fn resolution(got: &Serving, against: &Serving) -> (f64, f64) {
    let (g, a) = (got.measures(), against.measures());
    let each: Vec<f64> = (0..4)
        .map(|i| median_error(g[i]).hypot(median_error(a[i])))
        .collect();
    let weighted = (0..4)
        .map(|i| (WEIGHTS[i] * each[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    (weighted, each.iter().copied().fold(0.0, f64::max))
}

/// The worst single measure of `got` against `against`: the prompt, or a
/// lane count's step, whichever slowed most. A trade that wins the prompt
/// and loses decode is a loss for a server, whatever the weighted mean says.
fn worst(got: &Serving, against: &Serving) -> (f64, &'static str) {
    let m = |v: &[f64]| median(v.to_vec());
    let mut out = (m(&got.prompt) / m(&against.prompt), "the prompt");
    let names = ["the 1-lane step", "the 4-lane step", "the 8-lane step"];
    for (i, name) in names.iter().enumerate() {
        let r = m(&got.steps[i]) / m(&against.steps[i]);
        if r > out.0 {
            out = (r, name);
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x) * f64::from(y);
        aa += f64::from(x) * f64::from(x);
        bb += f64::from(y) * f64::from(y);
    }
    dot / (aa * bb).sqrt()
}

fn top(r: &[f32]) -> usize {
    (0..r.len()).fold(0, |b, i| if r[i] > r[b] { i } else { b })
}

/// Whether two readouts say the same: a cosine of 0.999 or more, or 0.99
/// or more with the same top token. A change of arithmetic (an fp16 format,
/// another accumulation order) can flip a near tie at the top; it cannot
/// move the whole row.
fn same(cosine: f64, same_top: bool) -> bool {
    cosine >= 0.999 || (cosine >= 0.99 && same_top)
}

/// Why two readouts disagree, or `None` ([`same`]).
fn disagrees(a: &[f32], b: &[f32]) -> Option<String> {
    let c = cosine(a, b);
    let (ta, tb) = (top(a), top(b));
    (!same(c, ta == tb)).then(|| format!("cosine {c:.4}, top token {tb} for {ta}"))
}

/// One model served under today's profile and under the tuned one.
#[derive(Clone, Debug)]
pub struct Check {
    pub model: String,
    /// Median milliseconds of a prompt, (today, tuned).
    pub prefill_ms: (f64, f64),
    /// Median milliseconds of a decode step at each lane count:
    /// (lanes, today, tuned). Several lanes put several rows through every
    /// matmul, where ladders differ most.
    pub decode_ms: Vec<(usize, f64, f64)>,
    pub cosine: f64,
    pub same_top: bool,
    /// How much slower the tuned file may read and still pass: today's own
    /// drift between its two servings, or what the servings' timings can
    /// resolve where that is more, twice over.
    pub allowance: f64,
}

impl Check {
    #[must_use]
    pub fn passes(&self) -> bool {
        let ok = |a: f64, b: f64| b <= a * self.allowance;
        ok(self.prefill_ms.0, self.prefill_ms.1)
            && self.decode_ms.iter().all(|(_, a, b)| ok(*a, *b))
            && same(self.cosine, self.same_top)
    }

    #[must_use]
    pub fn describe(&self) -> String {
        let steps: Vec<String> = self
            .decode_ms
            .iter()
            .map(|(lanes, a, b)| format!("{lanes}-lane step {a:.2} -> {b:.2} ms"))
            .collect();
        format!(
            "{}: {PROMPT}-token prompt {:.1} -> {:.1} ms, {}, readout cosine {:.4}{}",
            self.model,
            self.prefill_ms.0,
            self.prefill_ms.1,
            steps.join(", "),
            self.cosine,
            if self.same_top {
                ""
            } else {
                ", top token differs"
            }
        )
    }
}

/// Serve model `model` under `today`, `tuned`, and `today` again: today's
/// two servings bracket the tuned one, their mean is what it is compared
/// against, and their drift is the noise it is allowed.
pub fn check(
    bench: &mut dyn Bench,
    model: usize,
    today: &str,
    tuned: &str,
) -> Result<Check, String> {
    let a = bench.serve(model, today)?;
    let b = bench.serve(model, tuned)?;
    let c = bench.serve(model, today)?;
    let noise = (relative(&c, &a) - 1.0).abs();
    let both = |x: &[f64], y: &[f64]| 0.5 * (median(x.to_vec()) + median(y.to_vec()));
    Ok(Check {
        model: bench.models()[model].clone(),
        prefill_ms: (both(&a.prompt, &c.prompt), median(b.prompt.clone())),
        decode_ms: (0..LANES.len())
            .map(|i| {
                (
                    LANES[i],
                    both(&a.steps[i], &c.steps[i]),
                    median(b.steps[i].clone()),
                )
            })
            .collect(),
        cosine: cosine(&a.readout, &b.readout),
        same_top: top(&a.readout) == top(&b.readout),
        allowance: 1.0 + 2.0 * noise.max(resolution(&b, &a).1).max(resolution(&b, &c).1),
    })
}

/// A value as one line for the report: a ladder's rungs joined, and long
/// ones cut.
fn shown(label: &str) -> String {
    let one: String = label.split_whitespace().collect::<Vec<_>>().join(" ");
    if one.chars().count() > 90 {
        format!("{}…", one.chars().take(90).collect::<String>())
    } else {
        one
    }
}

/// A decision's best candidate so far: its mean relative time, label,
/// file, what each model it moved served, and the mean it must beat.
type Best = (f64, String, String, Vec<(usize, Serving)>, f64);

/// Of `served`, the lowest candidate within [`crate::AGREEMENT`] of the
/// fastest: by its value where every tied label reads as a number, else the
/// first served. `None` where none served.
fn pick_tied(served: Vec<Best>) -> Option<Best> {
    let fastest = served.iter().map(|b| b.0).fold(f64::INFINITY, f64::min);
    let tied: Vec<Best> = served
        .into_iter()
        .filter(|b| b.0 <= fastest * (1.0 + crate::AGREEMENT))
        .collect();
    let numbers: Option<Vec<f64>> = tied
        .iter()
        .map(|b| b.1.trim().parse::<f64>().ok())
        .collect();
    match numbers {
        Some(values) => {
            let at = (0..tied.len()).min_by(|&a, &b| values[a].total_cmp(&values[b]))?;
            tied.into_iter().nth(at)
        }
        None => tied.into_iter().next(),
    }
}

/// Coordinate descent over `knobs` on `bench`'s models: each decision's
/// candidates are served on the models that read it (or, for a price or
/// fusion, whose plan it changes), against the incumbent served again just
/// before them. The fastest on average is taken if its mean beats the
/// models' averaged noise (and 1%), no model loses more than twice its own
/// noise (and 2%), and every model reads the same. A decision under
/// `refusable` is left alone where the co-processor refuses its programs.
pub fn search(
    bench: &mut dyn Bench,
    mut chosen: Entries,
    knobs: &[Knob],
    refusable: &str,
    say: &mut dyn FnMut(&str),
) -> Result<(Entries, Vec<Line>), String> {
    let names = bench.models();
    let mut incumbent_text = bench.file(&chosen)?;
    // Each model's last serving of the incumbent; `None` where it did not
    // serve at all.
    let mut incumbent: Vec<Option<Serving>> = Vec::new();
    for (m, name) in names.iter().enumerate() {
        match bench.serve(m, &incumbent_text) {
            Ok(s) => incumbent.push(Some(s)),
            Err(why) => {
                say(&format!("  {name} did not serve, left out: {why}"));
                incumbent.push(None);
            }
        }
    }
    if incumbent.iter().all(Option::is_none) {
        return Err("no check model serves".into());
    }
    let refuses = |id: &str| !refusable.is_empty() && id.starts_with(refusable);
    let mut lines = Vec::new();
    for knob in knobs {
        let mut fresh = vec![false; names.len()];
        let by_plan = knob.plan_only || bench.plan_covers_all();
        let movable: Vec<usize> = (0..names.len())
            .filter(|&m| {
                incumbent[m].as_ref().is_some_and(|s| {
                    (by_plan || s.read.contains(&knob.id))
                        && !(refuses(knob.id) && s.refused.is_some())
                })
            })
            .collect();
        if movable.is_empty() {
            let text = if refuses(knob.id) {
                let why = incumbent
                    .iter()
                    .flatten()
                    .find_map(|s| s.refused.clone())
                    .unwrap_or_default();
                format!("kept; the co-processor refuses programs on this machine ({why})")
            } else {
                "kept; no check model reads it".to_string()
            };
            say(&format!("{}: {text}", knob.id));
            lines.push(Line {
                id: knob.id,
                text,
                measured: false,
            });
            continue;
        }
        let plans: Vec<Option<u64>> = (0..names.len())
            .map(|m| {
                if by_plan {
                    bench.plan(m, &incumbent_text)
                } else {
                    None
                }
            })
            .collect();
        let mut best: Option<Best> = None;
        // Every candidate that served, as `best` would hold it.
        let mut served_all: Vec<Best> = Vec::new();
        let mut notes: Vec<String> = Vec::new();
        // Each model's drift between two servings of the incumbent, and
        // what its last comparison's timings resolve.
        let mut noise: Vec<f64> = vec![0.0; names.len()];
        let mut floor: Vec<f64> = vec![0.0; names.len()];
        let mut queue: Vec<String> = knob.candidates.iter().map(|(l, _)| l.clone()).collect();
        // The incumbent counts as tried, so halfway never skips it.
        let mut tried: Vec<(String, bool)> = vec![(knob.current.clone(), true)];
        let mut bracketed = false;
        let mut at = 0;
        'candidate: while at < queue.len() || !bracketed {
            if at == queue.len() {
                // Once: where the fastest number sits between a neighbour
                // that lost or failed, try halfway to it.
                bracketed = true;
                if let Some((_, label, ..)) = best.as_ref().filter(|_| knob.halfway) {
                    queue.extend(halfway(label, &tried));
                }
                continue;
            }
            let label = queue[at].clone();
            at += 1;
            tried.push((label.clone(), false));
            let seen = shown(&label);
            let trial = with_entry(&chosen, knob.id, &label);
            let text = match bench.file(&trial) {
                Ok(t) => t,
                Err(why) => {
                    let why = why.lines().next().unwrap_or("").to_string();
                    // `origin:line:col: ` says nothing a reader needs.
                    let why = why
                        .splitn(4, ':')
                        .nth(3)
                        .map_or(why.clone(), |w| w.trim().to_string());
                    notes.push(format!("refused: {why}\u{1f}{seen}"));
                    continue;
                }
            };
            let on: Vec<usize> = movable
                .iter()
                .copied()
                .filter(|&m| !by_plan || plans[m].is_none() || bench.plan(m, &text) != plans[m])
                .collect();
            if on.is_empty() {
                notes.push(format!("plans the same\u{1f}{seen}"));
                continue;
            }
            let mut got: Vec<(usize, Serving)> = Vec::new();
            let mut r = 0.0;
            let mut worse: Option<String> = None;
            for &m in &on {
                // The incumbent again, just before its rivals: drift lands
                // on both, and how far it moved since its last serving is
                // the noise a rival must beat.
                if !fresh[m] {
                    // A device that will not serve the incumbent again (a
                    // lost device, memory taken by an earlier load) leaves
                    // this decision as it is; the next one tries again.
                    let again = match bench.serve(m, &incumbent_text) {
                        Ok(a) => a,
                        Err(why) => {
                            let why: String =
                                why.lines().next().unwrap_or("").chars().take(100).collect();
                            notes.push(format!(
                                "today's file did not serve again on {} ({why})",
                                names[m]
                            ));
                            best = None;
                            served_all.clear();
                            break 'candidate;
                        }
                    };
                    let last = incumbent[m].as_ref().expect("movable models served");
                    noise[m] = (relative(&again, last) - 1.0).abs();
                    incumbent[m] = Some(again);
                    fresh[m] = true;
                }
                let base = incumbent[m].as_ref().expect("served");
                let served = match bench.serve(m, &text) {
                    Ok(s) if s.refused.is_some() && base.refused.is_none() => {
                        let why = s.refused.unwrap_or_default();
                        notes.push(format!("{seen}: the co-processor refused it ({why})"));
                        continue 'candidate;
                    }
                    Ok(s) => s,
                    Err(why) => {
                        let why: String =
                            why.lines().next().unwrap_or("").chars().take(100).collect();
                        notes.push(format!("{seen} did not serve ({why})"));
                        continue 'candidate;
                    }
                };
                if let Some(why) = disagrees(&base.readout, &served.readout) {
                    notes.push(format!("{seen} reads differently on {} ({why})", names[m]));
                    continue 'candidate;
                }
                let rm = relative(&served, base);
                // What this pair's timings resolve, where the incumbent's
                // drift says less.
                let (mean_floor, worst_floor) = resolution(&served, base);
                floor[m] = mean_floor;
                // No measure of any model may lose more than its noise twice
                // over: the final check holds the file to the same.
                let (w, what) = worst(&served, base);
                if w > 1.0 + 2.0 * noise[m].max(worst_floor) {
                    worse = Some(format!("{what} {:+.1}% on {}", (w - 1.0) * 100.0, names[m]));
                }
                r += rm / on.len() as f64;
                got.push((m, served));
            }
            say(&format!(
                "  {} = {seen}: {:+.1}%",
                knob.id,
                (r - 1.0) * 100.0
            ));
            if let Some(last) = tried.last_mut() {
                last.1 = true;
            }
            if let Some(worse) = worse {
                notes.push(format!("{seen} costs a model ({worse})"));
                continue;
            }
            // The mean must beat the models' noise (each its drift, or what
            // its timings resolve where that is more), which averaging over
            // `n` of them shrinks by its root.
            let n = on.len() as f64;
            let rms = (on
                .iter()
                .map(|&m| noise[m].max(floor[m]).powi(2))
                .sum::<f64>()
                / n)
                .sqrt();
            let bar = 1.0 - 2.0 * rms / n.sqrt();
            served_all.push((r, label.clone(), text.clone(), got.clone(), bar));
            if best.as_ref().is_none_or(|(b, ..)| r < *b) {
                best = Some((r, label.clone(), text, got, bar));
            }
        }
        // Of the candidates within the agreement tolerance of the fastest,
        // the lowest is taken: a near-tie between two runs of a tune lands
        // on the same value, and two tunes write the same profile.
        if best.is_some() {
            best = pick_tied(served_all);
        }
        // A keep says the incumbent holds only where a change had to beat
        // no more than the agreement tolerance: past it, the run's noise
        // hid any difference, and nothing was measured.
        let measured = best
            .as_ref()
            .is_some_and(|(r, .., bar)| r < bar || 1.0 - bar <= crate::AGREEMENT);
        let text = match best {
            Some((r, label, text, got, bar)) if r < bar => {
                chosen = with_entry(&chosen, knob.id, &label);
                incumbent_text = text;
                for (m, served) in got {
                    incumbent[m] = Some(served);
                }
                format!(
                    "{}: {:.1}% faster end to end",
                    shown(&label),
                    (1.0 - r) * 100.0
                )
            }
            Some((r, label, _, _, bar)) => format!(
                "kept; the best other, {}, was {:+.1}% (a change must beat {:.1}%)",
                shown(&label),
                (r - 1.0) * 100.0,
                (1.0 - bar) * 100.0
            ),
            None => "kept; no other candidate served".to_string(),
        };
        // Notes that share a reason name it once: `reason: a, b, c`.
        let mut grouped: Vec<(String, Vec<String>)> = Vec::new();
        for note in notes {
            let (reason, label) = note
                .split_once('\u{1f}')
                .map_or((note.clone(), None), |(r, l)| {
                    (r.to_string(), Some(l.to_string()))
                });
            match grouped.iter_mut().find(|(r, _)| *r == reason) {
                Some((_, labels)) => labels.extend(label),
                None => grouped.push((reason, label.into_iter().collect())),
            }
        }
        let notes: Vec<String> = grouped
            .into_iter()
            .map(|(reason, labels)| {
                if labels.is_empty() {
                    reason
                } else {
                    format!("{reason}: {}", labels.join(", "))
                }
            })
            .collect();
        let text = if notes.is_empty() {
            text
        } else {
            format!("{text} ({})", notes.join("; "))
        };
        say(&format!("{}: {text}", knob.id));
        lines.push(Line {
            id: knob.id,
            text,
            measured,
        });
    }
    Ok((chosen, lines))
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_near_tie_takes_the_lowest_candidate() {
        let b = |r: f64, label: &str| (r, label.to_string(), String::new(), Vec::new(), 1.0);
        // 0.96, 0.97 and 1.00 sit within 5% of the fastest: the lowest value.
        let got = super::pick_tied(vec![b(1.00, "128"), b(0.97, "64"), b(0.96, "256")]);
        assert_eq!(got.map(|g| g.1).as_deref(), Some("64"));
        // A clear winner is taken whatever its value.
        let got = super::pick_tied(vec![b(1.00, "64"), b(0.90, "256")]);
        assert_eq!(got.map(|g| g.1).as_deref(), Some("256"));
        // Labels that are not numbers: the first served of the tied.
        let got = super::pick_tied(vec![b(0.99, "\"walk\""), b(0.98, "\"flash\"")]);
        assert_eq!(got.map(|g| g.1).as_deref(), Some("\"walk\""));
        assert!(super::pick_tied(Vec::new()).is_none());
    }

    use super::*;

    #[test]
    fn a_second_look_tries_halfway_to_the_neighbours() {
        let tried: Vec<(String, bool)> = [24, 35, 47, 59, 71, 94]
            .iter()
            .map(|p| (format!("{{ use = \"share\", percent = {p} }}"), *p != 94))
            .collect();
        assert_eq!(
            halfway("{ use = \"share\", percent = 71 }", &tried),
            vec![
                "{ use = \"share\", percent = 65 }".to_string(),
                "{ use = \"share\", percent = 82 }".to_string()
            ]
        );
        assert!(halfway("120.0", &[("60.0".into(), true)]).is_empty());
        assert_eq!(
            halfway("64", &[("32".into(), true), ("128".into(), true)]),
            vec!["48", "96"]
        );
    }

    /// A bench whose time is a function of one flag and one number, with
    /// no noise: the search finds both, and leaves an unread decision.
    struct Toy;

    fn value(file: &str, key: &str) -> Option<String> {
        file.lines()
            .find_map(|l| l.strip_prefix(&format!("{key} = ")).map(str::to_string))
    }

    impl Bench for Toy {
        fn models(&self) -> Vec<String> {
            vec!["toy".into()]
        }
        fn file(&self, entries: &[(String, String)]) -> Result<String, String> {
            Ok(render_overrides(entries))
        }
        fn serve(&mut self, _: usize, file: &str) -> Result<Serving, String> {
            let fast = value(file, "fast").is_some_and(|v| v == "true");
            let n: f64 = value(file, "n")
                .and_then(|v| v.parse().ok())
                .unwrap_or(10.0);
            let t = (if fast { 50.0 } else { 100.0 }) + (n - 30.0).abs();
            Ok(Serving {
                prompt: vec![t],
                steps: vec![vec![t]; 3],
                readout: vec![1.0, 0.0],
                read: vec!["kernel.fast", "kernel.n"],
                refused: None,
            })
        }
        fn plan(&self, _: usize, _: &str) -> Option<u64> {
            None
        }
    }

    #[test]
    fn the_search_takes_what_is_faster_and_leaves_what_is_not_read() {
        let knob = |id: &'static str, current: &str, values: &[&str], halfway: bool| Knob {
            id,
            current: current.into(),
            candidates: values
                .iter()
                .map(|v| (v.to_string(), String::new()))
                .collect(),
            halfway,
            plan_only: false,
        };
        let knobs = vec![
            knob("kernel.fast", "false", &["true"], false),
            knob("kernel.n", "10", &["5", "20", "40"], true),
            knob("kernel.unread", "1", &["2"], false),
        ];
        let start = vec![
            ("kernel.fast".to_string(), "false".to_string()),
            ("kernel.n".to_string(), "10".to_string()),
        ];
        let (chosen, lines) = search(&mut Toy, start, &knobs, "", &mut |_| {}).unwrap();
        assert_eq!(chosen[0].1, "true");
        // 20 and 40 tie at 10 from 30; halfway between them finds 30.
        assert_eq!(chosen[1].1, "30", "{lines:?}");
        assert!(
            lines[2].text.contains("no check model reads it"),
            "{lines:?}"
        );
        // Measured: what was compared and changed; not what no model reads.
        assert!(lines[0].measured && lines[1].measured, "{lines:?}");
        assert!(!lines[2].measured, "{lines:?}");
    }

    /// A bench whose every serving is 10% off the last: noise no change
    /// can beat within the agreement tolerance.
    struct Noisy(bool);

    impl Bench for Noisy {
        fn models(&self) -> Vec<String> {
            vec!["noisy".into()]
        }
        fn file(&self, entries: &[(String, String)]) -> Result<String, String> {
            Ok(render_overrides(entries))
        }
        fn serve(&mut self, _: usize, _: &str) -> Result<Serving, String> {
            self.0 = !self.0;
            let t = if self.0 { 100.0 } else { 110.0 };
            Ok(Serving {
                prompt: vec![t],
                steps: vec![vec![t]; 3],
                readout: vec![1.0, 0.0],
                read: vec!["kernel.n"],
                refused: None,
            })
        }
        fn plan(&self, _: usize, _: &str) -> Option<u64> {
            None
        }
    }

    #[test]
    fn a_keep_under_noise_past_the_agreement_is_not_measured() {
        let knobs = vec![Knob {
            id: "kernel.n",
            current: "10".into(),
            candidates: vec![("20".into(), String::new())],
            halfway: false,
            plan_only: false,
        }];
        let (chosen, lines) =
            search(&mut Noisy(false), Vec::new(), &knobs, "", &mut |_| {}).unwrap();
        assert!(chosen.is_empty(), "{lines:?}");
        assert!(lines[0].text.starts_with("kept"), "{lines:?}");
        assert!(!lines[0].measured, "{lines:?}");
    }

    /// A bench whose plan fingerprint covers every decision: `kernel.noop`
    /// is read by the model but moves nothing it runs, and noise would make
    /// one of its candidates look faster.
    struct Planned(usize);

    impl Bench for Planned {
        fn models(&self) -> Vec<String> {
            vec!["toy".into()]
        }
        fn file(&self, entries: &[(String, String)]) -> Result<String, String> {
            Ok(render_overrides(entries))
        }
        fn serve(&mut self, _: usize, file: &str) -> Result<Serving, String> {
            self.0 += 1;
            // Every other serve is 10% faster, whatever it serves.
            let t = if self.0.is_multiple_of(2) {
                90.0
            } else {
                100.0
            };
            let _ = value(file, "noop");
            Ok(Serving {
                prompt: vec![t],
                steps: vec![vec![t]; 3],
                readout: vec![1.0, 0.0],
                read: vec!["kernel.noop"],
                refused: None,
            })
        }
        fn plan(&self, _: usize, _: &str) -> Option<u64> {
            Some(7)
        }
        fn plan_covers_all(&self) -> bool {
            true
        }
    }

    #[test]
    fn a_decision_that_moves_no_plan_is_not_served() {
        let knobs = vec![Knob {
            id: "kernel.noop",
            current: "1".into(),
            candidates: vec![("2".into(), String::new()), ("3".into(), String::new())],
            halfway: false,
            plan_only: false,
        }];
        let start = vec![("kernel.noop".to_string(), "1".to_string())];
        let mut bench = Planned(0);
        let (chosen, lines) = search(&mut bench, start, &knobs, "", &mut |_| {}).unwrap();
        assert_eq!(chosen[0].1, "1", "{lines:?}");
        assert!(lines[0].text.contains("plans the same"), "{lines:?}");
        // Only the incumbent's first serving: no candidate was served.
        assert_eq!(bench.0, 1);
    }
}
