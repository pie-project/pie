//! The fire terms `pie tune` measures on Metal: the terms of a fire's time
//! ([`profile::curves::FireTerm`]) as Metal serves the model, each read off
//! fires of the model's own rungs on a load of its own weights: every
//! dispatch of a walked fire committed alone and timed by the GPU's clock
//! (`crate::encode::profile_kernels`), rolled up by the node it dispatches
//! for (`crate::encode::set_term`):
//!
//! * an attention by its heads, at the rung's rows and the keys its lane
//!   attends (the fire's held prefix and its own rows), a layer's;
//! * a dense plane's matmul `[n, k, dtype]`, a launch's (the output head's
//!   at the rows a fire reads out);
//! * a gated-delta layer's conv, scan, gates and gated norm, a layer's;
//! * every other node (norms, rope, residuals, the activation, the kv
//!   write, the splits) a layer's per-row pass, its share of the layers;
//! * the fire's own cost past its device work: the fire's wall time less
//!   its command buffers' device span, from fires timed as serving runs
//!   them.

use std::collections::{BTreeMap, BTreeSet};

use engine::load::check::{Bench as _, Run};
use engine::load::rung::CheckFire;
use poem_ir::{Attn, Dtype, Operation, Ssm, Trace};
use profile::curves::FireTerm;

use super::CheckBench;

/// The keys each rung's attention is timed at, below the model's context
/// (those no fewer than the rung's rows), and the context itself.
pub const KEYS: [u32; 3] = [4096, 16384, 32768];

/// The context the checkpoint `artifact` carries the config of states
/// (`max_position_embeddings`, its text model's where it has one).
#[must_use]
pub fn context_of(artifact: &std::path::Path) -> Option<u32> {
    let metadata = checkpoint::file::read::parse_metadata(artifact).ok()?;
    let config = checkpoint::file::read::read_meta(&metadata, "model/config")
        .ok()
        .flatten()?;
    let config: serde_json::Value = serde_json::from_slice(&config).ok()?;
    let text = config
        .get("text_config")
        .filter(|text| text.is_object())
        .unwrap_or(&config);
    text.get("max_position_embeddings")
        .and_then(serde_json::Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
}

/// The keys a model of `context` is timed at: [`KEYS`] below it, then it.
#[must_use]
pub fn keys_to(context: u32) -> Vec<u32> {
    let mut keys: Vec<u32> = KEYS.iter().copied().filter(|&k| k < context).collect();
    keys.push(context);
    keys
}
use crate::encode::Term;

/// The fires timed a point as serving runs them (their median wall and
/// device time), after one warm fire.
const TIMED: usize = 3;
/// The fires dissected a point (each term's median of them).
const DISSECTED: usize = 2;

/// A dense plane's dtype as [`FireTerm`] codes it (0: no code).
#[must_use]
pub fn dtype_code(dtype: Dtype) -> u32 {
    match dtype {
        Dtype::Bf16 => FireTerm::BF16,
        Dtype::F16 => FireTerm::FP16,
        Dtype::F32 => FireTerm::FP32,
        _ => 0,
    }
}

/// What the terms are shared over: the layers each kind runs in, the
/// per-row pass's `[hidden, inter]` and the scan's shape.
#[derive(Debug, Clone)]
pub struct Layers {
    pub layers: u32,
    pub attention: u32,
    pub scan: BTreeSet<u32>,
    /// `[k_heads, v_heads, k_dim, v_dim, conv_width]`.
    pub scan_shape: Option<Vec<u32>>,
    pub hidden: u32,
    pub inter: u32,
}

impl Layers {
    #[must_use]
    pub fn of(trace: &Trace, counts: &engine::load::Counts) -> Self {
        let mut all = BTreeSet::new();
        let mut attention = BTreeSet::new();
        let mut scan = BTreeSet::new();
        let (mut heads, mut conv) = (None, None);
        for node in &trace.nodes {
            let Some(layer) = node.layer else { continue };
            all.insert(layer);
            match &node.op {
                Operation::Attn(Attn::Prefill { .. }) => {
                    attention.insert(layer);
                }
                Operation::Ssm(op) => {
                    scan.insert(layer);
                    match op {
                        Ssm::GatedDeltaChunked {
                            k_heads,
                            v_heads,
                            k_dim,
                            v_dim,
                            ..
                        } => heads = Some([*k_heads, *v_heads, *k_dim, *v_dim]),
                        Ssm::CausalConv1dChunked { conv_width, .. } => conv = Some(*conv_width),
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        let hidden = counts.readout_depth;
        // The per-row pass's activation width: the widest depth a plane
        // writing the hidden width reads (the down projection's).
        let inter = counts
            .gemm_shapes
            .iter()
            .filter(|g| g.n == hidden)
            .map(|g| g.k)
            .max()
            .unwrap_or(0);
        Self {
            layers: u32::try_from(all.len()).unwrap_or(u32::MAX),
            attention: u32::try_from(attention.len()).unwrap_or(u32::MAX),
            scan_shape: heads.map(|h| {
                let mut shape = h.to_vec();
                shape.push(conv.unwrap_or(0));
                shape
            }),
            scan,
            hidden,
            inter,
        }
    }
}

/// One point timed: a fire of `rows` rows whose lane attends `keys` keys.
#[derive(Debug, Clone)]
pub struct Point {
    pub rows: u32,
    pub keys: u32,
    /// As serving runs it: the median wall time and device span (ns).
    pub wall_ns: f64,
    pub device_ns: f64,
    /// Walked and dissected: each term's device time (ns), its dispatches
    /// and the sources they ran, a fire's.
    pub terms: BTreeMap<(u32, Vec<u32>), (f64, u64, BTreeSet<String>)>,
}

impl Point {
    /// The dissected fire's device time, every term's.
    #[must_use]
    pub fn dissected_ns(&self) -> f64 {
        self.terms.values().map(|(ns, _, _)| ns).sum()
    }
}

/// The rolled-up term of one dissected dispatch group: its kind and shape
/// as the point keys it (the output head's matmul still marked).
fn rolled(term: &Term, layers: &Layers, rows: u32, keys: u32) -> (u32, Vec<u32>) {
    let Some((kind, shape)) = term else {
        return (FireTerm::ROW_OPS, vec![layers.hidden, layers.inter]);
    };
    match *kind {
        FireTerm::ATTENTION => {
            let mut shape = shape.clone();
            shape.push(rows.min(crate::FIRE.0));
            shape.push(keys);
            (FireTerm::ATTENTION, shape)
        }
        FireTerm::GEMM => (FireTerm::GEMM, shape.clone()),
        // A recurrent layer's conv and scan, and its gated norm; a gated
        // norm elsewhere is a row op.
        FireTerm::SCAN if shape.first().is_some_and(|l| layers.scan.contains(l)) => (
            FireTerm::SCAN,
            layers.scan_shape.clone().unwrap_or_default(),
        ),
        _ => (FireTerm::ROW_OPS, vec![layers.hidden, layers.inter]),
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Times each of `rows` at each of `keys` no fewer than its rows on
/// `bench`'s model, saying each point; `rows` `None`: the prompt-shaped
/// rungs the served load's weighing lays (to four times its fire top, as a
/// check lays them), the decode-shaped ones being the product times'.
///
/// # Errors
///
/// A load or a fire refused.
pub fn points(
    bench: &mut CheckBench<'_>,
    rows: Option<&[u32]>,
    keys: &[u32],
    say: &mut dyn FnMut(&str),
) -> Result<(Layers, Vec<Point>), String> {
    // The served load's weighing first: the check's shells are laid as its
    // pool lays them.
    let rungs: Vec<u32> = {
        let weighing = bench.load(Run::Walked, None)?;
        match rows {
            Some(rows) => rows.to_vec(),
            None => weighing
                .weighed()?
                .rungs
                .iter()
                .filter(|r| !r.decode_shaped() && r.infeasible.is_none())
                .map(|r| r.tokens)
                .collect(),
        }
    };
    let fires: Vec<CheckFire> = rungs
        .iter()
        .flat_map(|&r| {
            keys.iter().filter(move |&&k| k >= r).map(move |&k| CheckFire {
                rows: r,
                lanes: 1,
                tokens_per_lane: r,
                keys_per_lane: k,
                sampled: 1,
            })
        })
        .collect();
    let counts = crate::top::counts(&bench.trace).map_err(|e| e.to_string())?;
    let layers = Layers::of(&bench.trace, &counts);
    say(&format!(
        "  {} layers ({} attend, {} scan), per-row pass [{}, {}], scan {:?}",
        layers.layers,
        layers.attention,
        layers.scan.len(),
        layers.hidden,
        layers.inter,
        layers.scan_shape
    ));
    let shell = engine::load::check::shell_for(fires.iter().copied(), 16);
    let mut load = bench.load(Run::Walked, Some(shell))?;
    let mut points = Vec::with_capacity(fires.len());
    for fire in fires {
        load.fire(fire)?;
        let (mut walls, mut devices) = (Vec::new(), Vec::new());
        for _ in 0..TIMED {
            walls.push(load.fire(fire)?.ns);
            devices.push(crate::encode::last_device_us() as f64 * 1e3);
        }
        let mut dissections: Vec<BTreeMap<(u32, Vec<u32>), (f64, u64, BTreeSet<String>)>> =
            Vec::new();
        for _ in 0..DISSECTED {
            load.dissect(fire)?;
            let mut terms: BTreeMap<(u32, Vec<u32>), (f64, u64, BTreeSet<String>)> =
                BTreeMap::new();
            for (term, file, ns, n) in crate::encode::term_profile() {
                let id = rolled(&term, &layers, fire.rows, fire.keys_per_lane);
                let seat = terms.entry(id).or_default();
                seat.0 += ns as f64;
                seat.1 += n;
                seat.2.insert(file);
            }
            crate::encode::reset_kernel_profile();
            dissections.push(terms);
        }
        let mut terms = dissections[0].clone();
        for (id, seat) in &mut terms {
            seat.0 = median(
                dissections
                    .iter()
                    .map(|d| d.get(id).map_or(0.0, |s| s.0))
                    .collect(),
            );
        }
        let point = Point {
            rows: fire.rows,
            keys: fire.keys_per_lane,
            wall_ns: median(walls),
            device_ns: median(devices),
            terms,
        };
        say(&format!(
            "  {} rows at {} keys: {:.2} ms a fire ({:.2} ms on the device), {:.2} ms dissected",
            point.rows,
            point.keys,
            point.wall_ns / 1e6,
            point.device_ns / 1e6,
            point.dissected_ns() / 1e6
        ));
        points.push(point);
    }
    Ok((layers, points))
}

/// The terms `points` measure, each a layer's or a launch's (as
/// [`FireTerm`] states its kind), by rows: an attention at each keys of its
/// own, the others the median over the points' keys; the output head's
/// matmul at its one sampled row; the fire fixed its wall less
/// its device span.
#[must_use]
pub fn terms(layers: &Layers, points: &[Point]) -> Vec<FireTerm> {
    // (kind, shape) -> rows -> [ns over the points' keys], with sources.
    let mut by: BTreeMap<(u32, Vec<u32>), BTreeMap<u32, Vec<f64>>> = BTreeMap::new();
    let mut files: BTreeMap<(u32, Vec<u32>), BTreeSet<String>> = BTreeMap::new();
    for p in points {
        for ((kind, shape), (ns, n, sources)) in &p.terms {
            // The output head's matmul is marked by a fourth field
            // (`Run::linear`), its rows the fire's sampled ones.
            let is_head = *kind == FireTerm::GEMM && shape.len() == 4;
            let shape = &if is_head { shape[..3].to_vec() } else { shape.clone() };
            let share = match *kind {
                FireTerm::ATTENTION => f64::from(layers.attention.max(1)),
                FireTerm::SCAN => layers.scan.len().max(1) as f64,
                FireTerm::ROW_OPS => f64::from(layers.layers.max(1)),
                // A launch's: the plane's matmuls, one dispatch each.
                _ => (*n).max(1) as f64,
            };
            let rows = if is_head { 1 } else { p.rows };
            let id = (*kind, shape.clone());
            by.entry(id.clone()).or_default().entry(rows).or_default().push(ns / share);
            files.entry(id).or_default().extend(sources.iter().cloned());
        }
        by.entry((FireTerm::FIRE_FIXED, Vec::new()))
            .or_default()
            .entry(p.rows)
            .or_default()
            .push((p.wall_ns - p.device_ns).max(0.0));
    }
    by.into_iter()
        .map(|((kind, shape), curve)| {
            let sources: Vec<&str> = files
                .get(&(kind, shape.clone()))
                .map(|f| f.iter().map(String::as_str).collect())
                .unwrap_or_default();
            FireTerm {
                kind,
                shape,
                curve: curve.into_iter().map(|(rows, ns)| (rows, median(ns))).collect(),
                by: crate::stored::by(profile::rates::FIRE_TERMS, &sources),
            }
        })
        .collect()
}

/// `pie tune metal --fire-terms`: each of `models` (an artifact and its
/// deployment) loaded and its fire terms timed ([`points`] at its rungs
/// and [`keys_to`] its context, [`terms`]), the device's user file with them laid over
/// its `[rates]` (an entry of the same kind and shape replaced, a later
/// model's over an earlier's).
///
/// # Errors
///
/// No device, or a model that does not load or fire.
pub fn measure(
    profiles: Option<&std::path::Path>,
    models: &[(std::path::PathBuf, String)],
    say: &mut dyn FnMut(&str),
) -> Result<engine::rates::Rated, String> {
    let began = std::time::Instant::now();
    let name = crate::device::Context::bind()
        .map_err(|e| e.to_string())?
        .name()
        .to_string();
    let mut all: Vec<FireTerm> = Vec::new();
    for (artifact, deployment) in models {
        say(&format!("{name}: the fire terms of {deployment}"));
        let context = context_of(artifact).ok_or_else(|| {
            format!(
                "{deployment}: its artifact carries no config stating a context \
                 (`max_position_embeddings`) to time its attention to; import it again"
            )
        })?;
        say(&format!("  keys {:?} (its context {context})", keys_to(context)));
        let mut bench = super::check_bench(profiles, artifact, deployment)?;
        let (layers, points) = points(&mut bench, None, &keys_to(context), say)?;
        for term in terms(&layers, &points) {
            all.retain(|t| t.id() != term.id());
            all.push(term);
        }
    }
    let count = all.len();
    let stored = profile::rates::Rates {
        fire_terms: Some(all),
        measured_on: Some(crate::stored::measured_on(&name)),
        ..profile::rates::Rates::default()
    };
    let how = crate::rates::provenance(&name);
    let file = engine::rates::saved_text(super::user_file(profiles, &name).as_deref(), &stored, &how);
    let seconds = began.elapsed().as_secs_f64();
    let block = format!(
        "{}\n[provenance]\nrates.{} = {}\n",
        stored.toml().trim_end(),
        how.0,
        how.1
    );
    say(&format!("{name}: {count} fire terms of {} model(s) in {seconds:.0} s", models.len()));
    Ok(engine::rates::Rated {
        device: name,
        file,
        block,
        seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The terms of one served model on this Mac, written as JSON (each
    /// point and each term) to `PIE_FT_OUT`: `PIE_FT_ARTIFACT` its weights,
    /// `PIE_FT_DEPLOYMENT` its catalog row, `PIE_FT_ROWS` and `PIE_FT_KEYS`
    /// the rungs and keys timed.
    #[test]
    #[ignore = "times a served model on this Mac's GPU"]
    fn fire_terms_of_a_served_model() {
        let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k}"));
        let list = |k: &str| -> Vec<u32> { env(k).split(',').map(|v| v.parse().unwrap()).collect() };
        let artifact = std::path::PathBuf::from(env("PIE_FT_ARTIFACT"));
        let mut bench = super::super::check_bench(None, &artifact, &env("PIE_FT_DEPLOYMENT")).unwrap();
        let mut say = |line: &str| eprintln!("{line}");
        let rows = std::env::var("PIE_FT_ROWS").ok().map(|_| list("PIE_FT_ROWS"));
        let (layers, points) = points(&mut bench, rows.as_deref(), &list("PIE_FT_KEYS"), &mut say).unwrap();
        let counts = crate::top::counts(&bench.trace).unwrap();
        let terms = terms(&layers, &points);
        let mut out = String::from("{\"layers\": ");
        out += &format!(
            "{{\"layers\": {}, \"attention\": {}, \"scan\": {}, \"hidden\": {}, \"inter\": {}}}",
            layers.layers, layers.attention, layers.scan.len(), layers.hidden, layers.inter
        );
        out += ", \"points\": [";
        for (i, p) in points.iter().enumerate() {
            let terms: Vec<String> = p
                .terms
                .iter()
                .map(|((k, s), (ns, n, f))| format!("[{k}, {s:?}, {ns}, {n}, {:?}]", f.iter().collect::<Vec<_>>()))
                .collect();
            out += &format!(
                "{}{{\"rows\": {}, \"keys\": {}, \"wall_ns\": {}, \"device_ns\": {}, \"terms\": [{}]}}",
                if i > 0 { "," } else { "" },
                p.rows, p.keys, p.wall_ns, p.device_ns, terms.join(",")
            );
        }
        out += "], \"terms\": [";
        for (i, t) in terms.iter().enumerate() {
            out += &format!(
                "{}[{}, {:?}, {:?}, {:?}]",
                if i > 0 { "," } else { "" },
                t.kind, t.shape, t.curve.iter().map(|&(r, ns)| vec![f64::from(r), ns]).collect::<Vec<_>>(), t.by.kernels.iter().map(|(f, c)| format!("{f}:{c}")).collect::<Vec<_>>()
            );
        }
        out += "], \"gemm_planes\": [";
        out += &counts.gemm_shapes.iter().map(|g| format!("[{}, {}, {}, {}]", g.n, g.k, g.planes, u8::from(g.routed))).collect::<Vec<_>>().join(",");
        out += "]}";
        std::fs::write(env("PIE_FT_OUT"), out).unwrap();
    }
}
