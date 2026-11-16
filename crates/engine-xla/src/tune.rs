//! `pie tune xla`: the end-to-end search (`profile::search`) over the
//! decisions XLA's profile states, each candidate profile served on the
//! check models through the serving path (`Shell::load`, then a prompt and
//! decode steps at each of `search::LANES`), as `engine_cuda::tune::run`
//! does for CUDA. The device's `[rates]` are their own step.

use std::path::{Path, PathBuf};
use std::time::Instant;

use profile::search::{self, Bench, CONTEXT, Check, LANES, Line, PROMPT, STEPS, Serving};
use profile::{Profile, Report};

/// A check model, as the runtime resolved it: its trace on XLA, its
/// checkpoint's contract, the checkpoint.
pub struct Model {
    pub name: String,
    pub trace: poem_ir::Trace,
    pub contract: checkpoint::contract::ModelContract,
    pub checkpoint: PathBuf,
}

/// What a search produced.
pub struct Outcome {
    pub device: String,
    /// The complete profile, ready for the user's shelf.
    pub file: String,
    /// One line a decision searched.
    pub lines: Vec<Line>,
    /// The end-to-end check of the tuned file against today's, a model each.
    pub checks: Vec<Check>,
    pub changed: bool,
    pub seconds: f64,
}

/// Decisions the search leaves alone: none yet (every `tune = search`
/// decision XLA states is served).
const SEARCH_SKIPS: &[&str] = &[];

/// The device this machine would serve, by its kind (`TPU v6 lite`, ...).
fn device_kind() -> Result<String, String> {
    let client = crate::bench::client().ok_or("no PJRT device answers")?;
    let client = client.lock().map_err(|e| e.to_string())?;
    let device = client.device(0).ok_or("the client addresses no device")?;
    client.device_kind(device).map_err(|e| e.to_string())
}

struct Xla<'a> {
    models: &'a [Model],
    kind: &'a str,
    base: &'a Profile,
    /// Where each candidate file is laid for a load to resolve, under the
    /// device's own name (`Shelf::user_path`).
    shelf: PathBuf,
}

impl Bench for Xla<'_> {
    fn models(&self) -> Vec<String> {
        self.models.iter().map(|m| m.name.clone()).collect()
    }

    fn file(&self, entries: &[(String, String)]) -> Result<String, String> {
        let tpu = crate::chip::Tpu::of(self.kind);
        profile::search::assemble(
            crate::chip::registry(),
            self.base,
            self.kind,
            &tpu.facts(),
            &search::render_overrides(entries),
            &[],
            &[],
        )
    }

    fn serve(&mut self, model: usize, file: &str) -> Result<Serving, String> {
        let path = crate::chip::SHELF.user_path(&self.shelf, self.kind);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        std::fs::write(&path, file).map_err(|e| e.to_string())?;
        serve(&self.models[model], &self.shelf)
    }

    fn plan(&self, _model: usize, _file: &str) -> Option<u64> {
        // XLA cannot fingerprint a plan short of tracing it: every
        // candidate is served.
        None
    }
}

/// Serve `model` under the profile laid in `shelf`: a prompt, then steps at
/// each lane count; the decisions the load read.
fn serve(model: &Model, shelf: &Path) -> Result<Serving, String> {
    let slots = *LANES.iter().max().expect("lanes") as u32;
    let device = crate::api::DeviceBoot {
        profiles: Some(shelf.to_path_buf()),
        ..crate::api::DeviceBoot::default()
    };
    let mut shell = crate::Shell::load(crate::serve::Boot {
        trace: model.trace.clone(),
        contract: &model.contract,
        checkpoint: &model.checkpoint,
        budget: poem_compiler::Budget::new(slots, PROMPT.max(slots)),
        patches: None,
        page_size: 16,
        context: CONTEXT,
        slots,
        kv_budget: None,
        allowance: engine::Allowance::WHOLE,
        device: &device,
    })
    .map_err(|e| e.to_string())?;
    let token = |i: u32| 100 + (i * 7919) % 5000;
    let prompt: Vec<u32> = (0..PROMPT).map(token).collect();
    let facts = model.trace.facts.clone();
    let fire =
        |shell: &mut crate::Shell, lanes: &[(u32, &[u32])]| -> Result<(f64, Vec<f32>), String> {
            let lanes: Vec<crate::Lane<'_>> = lanes
                .iter()
                .map(|&(slot, tokens)| crate::Lane {
                    slot,
                    word: facts.word(&poem_ir::Request::new(tokens.len() as u32, false)),
                    tokens,
                })
                .collect();
            let began = Instant::now();
            let out = shell.fire(&lanes).map_err(|e| e.to_string())?;
            Ok((
                began.elapsed().as_secs_f64() * 1e3,
                out.into_iter().next().unwrap_or_default(),
            ))
        };
    // The first prompt compiles; the three after it are timed.
    let mut prefill = Vec::new();
    let mut readout = Vec::new();
    for round in 0..4 {
        shell.open(0).map_err(|e| e.to_string())?;
        let (ms, r) = fire(&mut shell, &[(0, &prompt)])?;
        if round > 0 {
            prefill.push(ms);
        }
        readout = r;
    }
    let history: Vec<u32> = (0..32).map(token).collect();
    for slot in 0..slots {
        shell.open(slot).map_err(|e| e.to_string())?;
        fire(&mut shell, &[(slot, &history)])?;
    }
    let mut decode = Vec::new();
    for &lanes in &LANES {
        let mut times = Vec::new();
        for step in 0..=STEPS {
            let next: Vec<[u32; 1]> = (0..lanes)
                .map(|l| [token((step * lanes + l) as u32)])
                .collect();
            let batch: Vec<(u32, &[u32])> = next
                .iter()
                .enumerate()
                .map(|(l, t)| (l as u32, &t[..]))
                .collect();
            let (ms, _) = fire(&mut shell, &batch)?;
            // The first step of a lane count compiles its program.
            if step > 0 {
                times.push(ms);
            }
        }
        decode.push(times);
    }
    let read = shell.chip_read();
    drop(shell);
    Ok(Serving {
        prompt: prefill,
        steps: decode,
        readout,
        read,
        refused: None,
    })
}

/// The search on this machine's TPU: every searchable decision served on
/// `models`, the tuned file checked end to end against today's, and the
/// file assembled with each decision's provenance. `say` hears progress.
pub fn run(
    profiles: Option<&Path>,
    models: &[Model],
    say: &mut dyn FnMut(&str),
) -> Result<Outcome, String> {
    let began = Instant::now();
    let kind = device_kind()?;
    let tpu = crate::chip::Tpu::of(&kind);
    if !tpu.known() {
        return Err(format!("`{kind}` is no TPU generation this build knows"));
    }
    let (base, report): (Profile, Report) = crate::chip::resolve(&kind, profiles);
    say(&format!("{kind}: starting from {}", report.used));
    let knobs = search::knobs(&base, SEARCH_SKIPS, "");
    let shelf = std::env::temp_dir().join(format!("pie-tune-xla-{}", std::process::id()));
    let mut bench = Xla {
        models,
        kind: &kind,
        base: &base,
        shelf: shelf.clone(),
    };
    let mut lines = Vec::new();
    let mut chosen = Vec::new();
    if models.is_empty() {
        for k in &knobs {
            lines.push(Line {
                id: k.id,
                text: "not searched: no model to serve (import one, or pass --check)".to_string(),
            });
        }
    } else {
        let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        say(&format!(
            "searching {} decisions end to end on {} (a candidate that changes a program \
             compiles it again)",
            knobs.len(),
            names.join(", ")
        ));
        let (found, searched) = search::search(&mut bench, chosen, &knobs, "", say)?;
        chosen = found;
        lines = searched;
    }
    let tuned = search::render_overrides(&chosen);
    let today = bench.file(&[])?;
    let draft = bench.file(&chosen)?;
    let unchanged = draft == today;
    let mut checks = Vec::new();
    for (at, m) in models.iter().enumerate().filter(|_| !unchanged) {
        say(&format!(
            "end to end: serving {} under today's profile and the tuned one",
            m.name
        ));
        match search::check(&mut bench, at, &today, &draft) {
            Ok(c) => {
                say(&format!("  {}", c.describe()));
                checks.push(c);
            }
            Err(why) => say(&format!("  {} did not serve: {why}", m.name)),
        }
    }
    let _ = std::fs::remove_dir_all(&shelf);
    // A searched decision is measured on this device where the search
    // decided it: a winner that beat the noise, or today's value kept under
    // a noise bar within AGREEMENT. One kept under a wider bar, or that no
    // candidate or model served, was not decided: its line (with its bar)
    // stays in `decisions`, but not in `measured`, so the load keeps naming
    // it (`chip::unmeasured`). The base's measured decisions the search did
    // not touch stay measured.
    let searched: Vec<&Line> = lines
        .iter()
        .filter(|l| !l.text.starts_with("not searched"))
        .collect();
    let mut measured: Vec<String> = report
        .measured
        .iter()
        .filter(|id| !searched.iter().any(|l| l.id == **id))
        .map(|id| format!("\"{id}\""))
        .collect();
    for l in searched.iter().filter(|l| decided(&l.text)) {
        let id = format!("\"{}\"", l.id);
        if !measured.contains(&id) {
            measured.push(id);
        }
    }
    measured.sort();
    let checked = if unchanged {
        "nothing changed".to_string()
    } else if checks.is_empty() {
        "no end-to-end run".to_string()
    } else {
        format!(
            "end to end: {}",
            checks
                .iter()
                .map(Check::describe)
                .collect::<Vec<_>>()
                .join("; ")
        )
    };
    // Each searched decision's outcome, as the search said it, by id.
    let decisions: Vec<String> = lines
        .iter()
        .filter(|l| !l.text.starts_with("not searched"))
        .map(|l| {
            format!(
                "\"{}\" = \"{}\"",
                l.id,
                l.text.replace('\\', "/").replace('"', "'")
            )
        })
        .collect();
    let seconds = began.elapsed().as_secs_f64();
    let provenance = [
        ("source", "\"pie tune xla on this machine\"".to_string()),
        ("measured", format!("[{}]", measured.join(", "))),
        ("decisions", format!("{{ {} }}", decisions.join(", "))),
        ("from", format!("\"{}\"", report.used)),
        ("pie", format!("\"{}\"", env!("CARGO_PKG_VERSION"))),
        ("seconds", format!("{seconds:.0}")),
        ("checked", format!("\"{}\"", checked.replace('"', "'"))),
    ];
    let file = search::assemble(
        crate::chip::registry(),
        &base,
        &kind,
        &tpu.facts(),
        &tuned,
        &provenance,
        &[],
    )?;
    Ok(Outcome {
        device: kind,
        file,
        lines,
        checks,
        changed: !unchanged,
        seconds,
    })
}

/// Whether a search line decided its decision: a winner beat the noise
/// ("…% faster end to end"), or today's value was kept under a noise bar
/// within `engine::rates::AGREEMENT` ("a change must beat X%"). A decision kept
/// under a wider bar, or that nothing served, was not decided.
fn decided(text: &str) -> bool {
    if text.contains("faster end to end") {
        return true;
    }
    let Some(at) = text.find("a change must beat ") else {
        return false;
    };
    let rest = &text[at + "a change must beat ".len()..];
    let bar: Option<f64> = rest.split('%').next().and_then(|n| n.trim().parse().ok());
    bar.is_some_and(|bar| bar <= engine::rates::AGREEMENT * 100.0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_line_is_measured_only_where_the_search_decided_it() {
        use super::decided;
        assert!(decided("64: 2.8% faster end to end"));
        assert!(decided(
            "kept; the best other, 64, was -4.1% (a change must beat 4.1%)"
        ));
        assert!(!decided(
            "kept; the best other, 128, was -13.6% (a change must beat 27.1%)"
        ));
        assert!(!decided(
            "kept; no other candidate served (64 costs a model)"
        ));
        assert!(!decided("kept; no check model reads it"));
    }
}
