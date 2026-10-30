//! The spatial engine: the model forward runs in a model daemon — the
//! resident wafer program on a CS-3, or a reference that answers the same
//! protocol — and everything else stays as the cerebras engine has it: guest
//! (eta) programs in the host interpreter, channels, the runtime's frames.
//!
//! The daemon (a test reference, e.g. the Python model; only a build with
//! the `spatial-daemon` feature runs one) is a child process speaking JSON
//! lines on stdin/stdout
//! (`spatial/serve/README.md` in the bring-up tree):
//!
//! ```text
//! {"op":"load","ckpt":dir,"layers":N?}   -> {"ok":true,"vocab":V,"hidden":H}
//! {"op":"reset"}                         -> {"ok":true}
//! {"op":"truncate","n":n}               -> {"ok":true}   (keep positions < n)
//! {"op":"forward","tokens":[..],"start_pos":p,"logits":"none"|"last"|"all"}
//!                                        -> {"ok":true,"argmax":[..],"logits":[[..]]?}
//! {"op":"generate","tokens":[..],"start_pos":p,"max_new":n,"stop":[..]}
//!                                        -> {"ok":true,"tokens":[..]}
//! ```
//!
//! The model holds one sequence's KV cache. The engine keeps every lane's
//! cache as the tokens that made it (by slot, and by page for `copy_kv`), and
//! before a lane's rows run brings the model's cache to that lane: kept as far
//! as it agrees, truncated there, the rest replayed. So rejected speculative
//! rows, forks, shared prefixes and several lanes a fire (run one after
//! another) all answer as a device cache would — slowly where lanes
//! alternate. Attention is causal over a lane's own history only: a lane
//! whose mask says otherwise is refused, as is a program reading attention
//! scores.
//!
//! A model that owns the KV cache as PIE's pages (its load answer says
//! `paged`: `fire` / `copy` in the protocol) runs a fire's lanes together,
//! each with its KV write slots, RoPE positions, pages and mask, and needs no
//! replay: `copy_kv` is a page copy, a truncate a smaller slot. A lane whose
//! pages the model never got (a fire that named another lane's pages) is
//! re-prefilled into them from its own tokens first; the replay above stays
//! the fallback for a model of single-sequence caches.
//!
//! Selected by `[cerebras.spatial]` in the engine config:
//!
//! ```toml
//! [cerebras.spatial]
//! daemon = "python3 spatial/serve/ref_daemon.py"   # run with `sh -c`
//! ckpt = "models/Qwen3-0.6B"   # the daemon's checkpoint (default: the load's)
//! layers = 2                   # optional: the first N layers only
//! ```
//!
//! Or, with no daemon, the wafer program in this process ([`wafer`]): the
//! compiled `serve28s` program's output directory, run on the `[cerebras]`
//! platform (`cmaddr`, else the simulator), its weights read through PIE's
//! checkpoint path (the load's contract over its checkpoint, by the model's
//! parameter names); `ckpt` names a Hugging Face directory to read instead
//! (`model.safetensors`, its shards, or `weights.npz`, with `config.json`):
//!
//! ```toml
//! [cerebras.spatial]
//! params = "kernel_cols:64,hidden:1024,..."   # serve28s' --params
//! ckpt = "models/Qwen3-0.6B"
//! ```
//!
//! `params` compiles the serve28s sources shipped in `spatial/serve28s` into
//! PIE's program cache (`PIE_CEREBRAS_CACHE`; see [`wafer::compile`]);
//! `program = "dir"` names an already compiled output directory instead.

pub mod ckpt;
pub mod guest4;
pub mod paged;
pub mod plan;
pub mod serve2d;
pub mod wafer;
// In-tree wafer programs: sources, generation, build, manifest (audit item 5).
pub mod build;
pub mod codegen;
pub mod manifest;
pub mod serve8b;
pub mod serve8b_region;

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use engine::Engine;
use engine::caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{
    FireId, FireTicket, FrameId, FrameSubmission, FrameTicket, LaneReadout, Readout,
};
use engine::load::{Checkpoint, LoadFacts, LoadRequest, Loaded};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::MemoryDomain;
use eta_exec::Value;
use eta_ir::registry::{GeometryClass, PortMask, Stage};

use crate::api::{ContractFor, DeviceBoot, bake_budgets, fault, patch_ladder, port_error, refusal};
use crate::serve::{Boot, Shell};

/// The `[cerebras.spatial]` table.
#[derive(Debug, Clone)]
pub struct SpatialBoot {
    pub daemon: Option<String>,
    pub program: Option<String>,
    pub params: Option<String>,
    /// `"args"` or `"run_cfg"`: how requests reach the wafer program
    /// (default: decided from the program).
    pub launch: Option<String>,
    /// Who runs translatable epilogues: `"off"`, `"check"` or `"wafer"`
    /// (`Spatial::guest_mode`).
    pub guest: Option<String>,
    /// The most steps a greedy lane's token loop runs past its row (0: off;
    /// loops start at 8 and double while a sequence uses them up).
    pub lookahead: Option<usize>,
    pub ckpt: Option<String>,
    pub layers: Option<u32>,
    /// The guest placement's forecast ([`guest4::cost::Placement`]): the
    /// model's token in cycles (default: from its parameter count), how
    /// much of it a table's `pre` may take and `main` may add.
    pub token_cycles: Option<u64>,
    pub pre_margin: Option<f64>,
    pub main_fraction: Option<f64>,
    /// `[cerebras.spatial.serve]`: with no `program` (and no serve28
    /// `params`), PIE builds the plan's program for the model itself
    /// ([`build::ServeConfig::from_toml`]; compiled once, then cached).
    pub serve: Option<toml::Table>,
}

impl SpatialBoot {
    pub fn of(doc: &toml::Table) -> Option<SpatialBoot> {
        let t = doc
            .get("cerebras")
            .and_then(toml::Value::as_table)?
            .get("spatial")
            .and_then(toml::Value::as_table)?;
        let text = |k: &str| t.get(k).and_then(toml::Value::as_str).map(str::to_string);
        let number = |k: &str| {
            t.get(k)
                .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        };
        let (daemon, program, params) = (text("daemon"), text("program"), text("params"));
        let serve = t.get("serve").and_then(toml::Value::as_table).cloned();
        if daemon.is_none() && program.is_none() && params.is_none() && serve.is_none() {
            return None;
        }
        Some(SpatialBoot {
            daemon,
            program,
            params,
            launch: text("launch"),
            guest: text("guest"),
            lookahead: t
                .get("lookahead")
                .and_then(toml::Value::as_integer)
                .and_then(|v| usize::try_from(v).ok()),
            ckpt: t
                .get("ckpt")
                .and_then(toml::Value::as_str)
                .map(str::to_string),
            layers: t
                .get("layers")
                .and_then(toml::Value::as_integer)
                .and_then(|v| u32::try_from(v).ok()),
            token_cycles: t
                .get("token_cycles")
                .and_then(toml::Value::as_integer)
                .and_then(|v| u64::try_from(v).ok()),
            pre_margin: number("pre_margin"),
            main_fraction: number("main_fraction"),
            serve,
        })
    }
}

/// The daemon child and its pipes.
struct Daemon {
    /// Sequences' caches it holds (its load answer's `caches`, else 1).
    caches: usize,
    /// It owns the KV cache as PIE's pages (its load answer's `paged`).
    paged: bool,
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Daemon {
    fn spawn(cmd: &str) -> EngineResult<Daemon> {
        let mut child = Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| Error::Load(format!("spawning the model daemon `{cmd}`: {e}")))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::Load("daemon stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::Load("daemon stdout".into()))?;
        Ok(Daemon {
            caches: 1,
            paged: false,
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    /// One request, its answer; an answer without `"ok": true` is an error.
    fn call(&mut self, request: &serde_json::Value) -> EngineResult<serde_json::Value> {
        let _busy = Busy::start();
        let json_ns = |t: std::time::Instant| {
            let ns = u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX);
            JSON_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
        };
        let t = std::time::Instant::now();
        let line = serde_json::to_string(request).expect("a JSON value serializes");
        json_ns(t);
        writeln!(self.stdin, "{line}")
            .and_then(|()| self.stdin.flush())
            .map_err(|e| Error::Device(format!("the model daemon went away: {e}")))?;
        let mut answer = String::new();
        let read = self
            .stdout
            .read_line(&mut answer)
            .map_err(|e| Error::Device(format!("reading the model daemon: {e}")))?;
        if read == 0 {
            return Err(Error::Device("the model daemon closed its output".into()));
        }
        let t = std::time::Instant::now();
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(answer.trim());
        json_ns(t);
        let value: serde_json::Value = parsed.map_err(|e| {
            Error::Device(format!(
                "the model daemon answered non-JSON ({e}): {answer}"
            ))
        })?;
        if value.get("ok").and_then(serde_json::Value::as_bool) != Some(true) {
            return Err(Error::Device(format!(
                "the model daemon refused {line}: {answer}"
            )));
        }
        Ok(value)
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Time spent waiting on the model (the daemon's answers, the wafer
/// thread's launches), all fires together: the host's share of a token is
/// the rest of its wall time.
pub(crate) static MODEL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Of [`MODEL_NS`], the host's own JSON work on a daemon's requests and
/// answers (serializing, parsing).
pub(crate) static JSON_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Adds its lifetime to [`MODEL_NS`].
pub(crate) struct Busy(std::time::Instant);

impl Busy {
    pub(crate) fn start() -> Busy {
        Busy(std::time::Instant::now())
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        let ns = u64::try_from(self.0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        MODEL_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Where the forward runs: the daemon, or the wafer program in process.
enum Model {
    Daemon(Daemon),
    Wafer(wafer::Wafer),
}

impl Model {
    /// Sequences' caches the model holds at once.
    /// Whether the model owns the KV cache as PIE's pages: a fire's lanes
    /// run with their page tables, no history replayed.
    fn paged(&self) -> bool {
        match self {
            Model::Daemon(d) => d.paged,
            Model::Wafer(w) => w.paged.is_some(),
        }
    }

    /// One fire's lanes on the paged cache: each lane's tokens at positions
    /// `start..` through its pages; the logits `which` asks for, per lane.
    /// Each lane's logits rows (as `which` asks) and, for an `argmax_only`
    /// lane, its last row's argmax instead.
    fn fire_paged(&mut self, lanes: &[&PagedLane]) -> EngineResult<Vec<PagedAnswer>> {
        let d = match self {
            Model::Wafer(w) => return fire_batched(w, lanes),
            Model::Daemon(d) => d,
        };
        let request: Vec<serde_json::Value> = lanes
            .iter()
            .map(|lane| {
                serde_json::json!({
                    "tokens": lane.tokens, "slots": lane.slots, "positions": lane.positions,
                    "pages": lane.pages, "visible": lane.visible,
                    // An argmax-only lane takes the daemon's argmax: no row
                    // of the vocabulary crosses the pipe.
                    "logits": match lane.which {
                        _ if lane.argmax_only => "none",
                        wafer::Logits::None => "none",
                        wafer::Logits::Last => "last",
                        wafer::Logits::All => "all",
                    },
                })
            })
            .collect();
        let answer = d.call(&serde_json::json!({"op": "fire", "lanes": request}))?;
        let lane_answers: &[serde_json::Value] = answer
            .get("lanes")
            .and_then(serde_json::Value::as_array)
            .map_or(&[], Vec::as_slice);
        let rows = |v: &serde_json::Value| -> Vec<Vec<f32>> {
            v.get("logits")
                .and_then(serde_json::Value::as_array)
                .map(|rows| {
                    rows.iter()
                        .map(|row| {
                            row.as_array()
                                .map(|v| {
                                    v.iter()
                                        .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32)
                                        .collect()
                                })
                                .unwrap_or_default()
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        Ok(lane_answers
            .iter()
            .zip(lanes)
            .map(|(answer, lane)| {
                if lane.argmax_only {
                    // The last row's argmax, as the daemon took it.
                    let token = answer
                        .get("argmax")
                        .and_then(serde_json::Value::as_array)
                        .and_then(|a| a.last())
                        .and_then(serde_json::Value::as_u64)
                        .map(|t| t as u32);
                    (Vec::new(), token)
                } else {
                    (rows(answer), None)
                }
            })
            .collect())
    }

    /// `copy_kv`'s token runs on the paged cache.
    fn copy_paged(&mut self, runs: &[engine::transfer::KvRun]) -> EngineResult<()> {
        let d = match self {
            Model::Wafer(w) => {
                let size = w.paged.map_or(0, |s| s.page_size) as u32;
                let mut pairs = Vec::new();
                for run in runs {
                    if run.src_token != 0 || run.dst_token != 0 || run.tokens != size {
                        return Err(Error::Unsupported {
                            verb: "`copy_kv` of part of a page on the wafer (whole pages only)",
                            engine: "cerebras-spatial",
                        });
                    }
                    pairs.push((run.src_page, run.dst_page));
                }
                return w
                    .copy_pages(pairs)
                    .map_err(|e| Error::Device(format!("the wafer model: {e}")));
            }
            Model::Daemon(d) => d,
        };
        let runs: Vec<[u32; 5]> = runs
            .iter()
            .map(|r| [r.src_page, r.src_token, r.dst_page, r.dst_token, r.tokens])
            .collect();
        d.call(&serde_json::json!({"op": "copy", "runs": runs})).map(drop)
    }

    fn caches(&self) -> usize {
        match self {
            Model::Daemon(d) => d.caches.max(1),
            Model::Wafer(_) => 1,
        }
    }

    fn reset(&mut self, cache: usize) -> EngineResult<()> {
        match self {
            Model::Daemon(d) => d.call(&with_cache(serde_json::json!({"op": "reset"}), cache)).map(drop),
            // A request at position 0 restarts the wafer's cache.
            Model::Wafer(_) => Ok(()),
        }
    }

    /// Keeps the cache's first `n` positions.
    fn truncate(&mut self, cache: usize, n: usize) -> EngineResult<()> {
        match self {
            Model::Daemon(d) => d
                .call(&with_cache(serde_json::json!({"op": "truncate", "n": n}), cache))
                .map(drop),
            // A request at position n attends positions below it alone and
            // writes over the rest.
            Model::Wafer(_) => Ok(()),
        }
    }

    /// The token loop: `tokens` at `start..`, then up to `extra` steps each
    /// fed the argmax before. The tokens the cache then holds past `start`
    /// and the argmax after each where known (the feed's last row on).
    fn greedy(
        &mut self,
        cache: usize,
        tokens: &[u32],
        start: u32,
        extra: usize,
    ) -> EngineResult<(Vec<u32>, Vec<Option<u32>>)> {
        let n = tokens.len();
        let new: Vec<u32> = match self {
            Model::Wafer(w) => {
                let argmax = w
                    .greedy(tokens, start as usize, extra)
                    .map_err(|e| Error::Device(format!("the wafer model: {e}")))?;
                argmax[n - 1..].to_vec()
            }
            Model::Daemon(d) => {
                let answer = d.call(&with_cache(
                    serde_json::json!({
                        "op": "generate", "tokens": tokens, "start_pos": start,
                        "max_new": extra + 1
                    }),
                    cache,
                ))?;
                let new: Vec<u32> = answer
                    .get("tokens")
                    .and_then(serde_json::Value::as_array)
                    .map(|v| v.iter().filter_map(|t| t.as_u64()).map(|t| t as u32).collect())
                    .unwrap_or_default();
                // A daemon may have fed its last token too: the cache ends
                // with the tokens whose argmax came back.
                let keep = start as usize + n + new.len().saturating_sub(1);
                d.call(&with_cache(serde_json::json!({"op": "truncate", "n": keep}), cache))?;
                new
            }
        };
        if new.is_empty() {
            return Err(Error::Device("the model's token loop answered nothing".into()));
        }
        // Fed: the tokens, then every new one but the last; the argmax after
        // the feed's last row is new[0], after new[i] it is new[i + 1].
        let mut fed = tokens.to_vec();
        fed.extend_from_slice(&new[..new.len() - 1]);
        let mut argmax = vec![None; n - 1];
        argmax.extend(new.iter().map(|&t| Some(t)));
        Ok((fed, argmax))
    }

    /// The argmax after each token and the logits `which` asks for (one row
    /// for `Last`, a row a token for `All`).
    fn forward(
        &mut self,
        cache: usize,
        tokens: &[u32],
        start: u32,
        which: wafer::Logits,
    ) -> EngineResult<Vec<Vec<f32>>> {
        match self {
            Model::Wafer(w) => w
                .forward(tokens, start as usize, which)
                .map(|(_, logits)| logits)
                .map_err(|e| Error::Device(format!("the wafer model: {e}"))),
            Model::Daemon(d) => {
                let mode = match which {
                    wafer::Logits::None => "none",
                    wafer::Logits::Last => "last",
                    wafer::Logits::All => "all",
                };
                let answer = d.call(&with_cache(
                    serde_json::json!({
                        "op": "forward", "tokens": tokens, "start_pos": start, "logits": mode
                    }),
                    cache,
                ))?;
                Ok(answer
                    .get("logits")
                    .and_then(serde_json::Value::as_array)
                    .map(|rows| {
                        rows.iter()
                            .map(|row| {
                                row.as_array()
                                    .map(|v| {
                                        v.iter()
                                            .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32)
                                            .collect()
                                    })
                                    .unwrap_or_default()
                            })
                            .collect()
                    })
                    .unwrap_or_default())
            }
        }
    }
}

/// A fire's paged lanes on the batched wafer build: each lane's rows as
/// launches of consecutive slots and positions (a prefill chunk, at most a
/// launch's iterations; a lane with a mask a row at a time; every row's
/// logits a row at a time), the lanes' n-th pieces together, in launches of
/// lanes of the same length: up to a launch's lanes (B, or MMAX
/// micro-batches of B on a ring) when no logits come back, the head's few
/// logit lanes when they do.
fn fire_batched(w: &mut wafer::Wafer, lanes: &[&PagedLane]) -> EngineResult<Vec<PagedAnswer>> {
    if w.paged.is_none() {
        return Err(Error::Device("a batched fire on a one-sequence build".into()));
    }
    let cap = w.cap.saturating_sub(1).max(1);
    // Each lane's pieces: (first row, rows, logits wanted).
    let mut pieces: Vec<Vec<(usize, usize, bool)>> = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let n = lane.tokens.len();
        let consecutive = |v: &[u32]| v.windows(2).all(|p| p[1] == p[0] + 1);
        let row_by_row = lane.visible.is_some()
            || lane.which == wafer::Logits::All
            || !consecutive(&lane.slots)
            || !consecutive(&lane.positions);
        let mut mine = Vec::new();
        let mut at = 0;
        while at < n {
            let rows = if row_by_row { 1 } else { (n - at).min(cap) };
            let last = at + rows == n;
            let want = match lane.which {
                _ if lane.argmax_only => false,
                wafer::Logits::None => false,
                wafer::Logits::Last => last,
                wafer::Logits::All => true,
            };
            mine.push((at, rows, want));
            at += rows;
        }
        pieces.push(mine);
    }
    let mut logits: Vec<Vec<Vec<f32>>> = vec![Vec::new(); lanes.len()];
    let mut last_argmax: Vec<Option<u32>> = vec![None; lanes.len()];
    let waves = pieces.iter().map(Vec::len).max().unwrap_or(0);
    for wave in 0..waves {
        // The lanes with an n-th piece, by (rows, logits wanted).
        let mut groups: BTreeMap<(usize, bool), Vec<usize>> = BTreeMap::new();
        for (li, mine) in pieces.iter().enumerate() {
            if let Some(&(_, rows, want)) = mine.get(wave) {
                groups.entry((rows, want)).or_default().push(li);
            }
        }
        for ((rows, want), members) in groups {
            // A launch's lanes (MMAX micro-batches of B on a ring); the head
            // keeps logits rows for the first few only.
            let most = if want { w.logit_lanes } else { w.launch_lanes };
            for chunk in members.chunks(most.max(1)) {
                let batch: Vec<crate::spatial::serve2d::BatchLane> = chunk
                    .iter()
                    .map(|&li| {
                        let lane = lanes[li];
                        let (first, _, _) = pieces[li][wave];
                        crate::spatial::serve2d::BatchLane {
                            pos: lane.positions[first],
                            slot: lane.slots[first],
                            pages: lane.pages.clone(),
                            visible: lane.visible.as_ref().map(|rows| {
                                rows[first]
                                    .iter()
                                    .flat_map(|&[a, b]| a..b)
                                    .collect()
                            }),
                            forced: lane.tokens[first..first + rows].to_vec(),
                        }
                    })
                    .collect();
                let (argmax_out, rows_out) = w
                    .fire_batch(batch, rows, want)
                    .map_err(|e| Error::Device(format!("the wafer model: {e}")))?;
                for (k, &li) in chunk.iter().enumerate() {
                    if let Some(&token) = argmax_out.get(k).and_then(|t| t.get(rows - 1)) {
                        last_argmax[li] = Some(token);
                    }
                }
                if want {
                    for (k, &li) in chunk.iter().enumerate() {
                        if let Some(row) = rows_out.get(k) {
                            logits[li].push(row.clone());
                        }
                    }
                }
            }
        }
    }
    Ok(logits
        .into_iter()
        .zip(last_argmax)
        .zip(lanes)
        .map(|((rows, token), lane)| if lane.argmax_only { (Vec::new(), token) } else { (rows, None) })
        .collect())
}

/// A paged lane's answer: its logits rows, or (argmax only) its last row's
/// argmax.
type PagedAnswer = (Vec<Vec<f32>>, Option<u32>);

/// One lane of a fire on a paged model.
struct PagedLane {
    tokens: Vec<u32>,
    /// The KV slot each row appends at.
    slots: Vec<u32>,
    /// The RoPE position each row rotates by.
    positions: Vec<u32>,
    pages: Vec<u32>,
    /// Per row, the slots it attends as `[start, end)` runs; `None`: every
    /// slot up to its own.
    visible: Option<Vec<Vec<[u32; 2]>>>,
    which: wafer::Logits,
    /// The lane's one reader takes only its last row's argmax: none of its
    /// logits come back (a launch carries more such lanes than lanes with
    /// logits).
    argmax_only: bool,
    /// The instance whose epilogue the wafer head's multi-lane VM samples
    /// for this lane (its one row, in its VM lane's bank).
    vm: Option<InstanceId>,
}

/// Per row of a lane's `rows` rows, the KV slots its mask admits, as runs: a
/// device geometry instance's dense mask, else `masking` (the fire's, or its
/// instance's port), else `None` (causal).
fn visible_runs(
    masking: Option<&engine::fire::Masking>,
    rows: usize,
    dense: Option<&(Vec<bool>, usize)>,
) -> Option<Vec<Vec<[u32; 2]>>> {
    let runs_of = |cells: &[bool]| -> Vec<[u32; 2]> {
        let mut runs = Vec::new();
        let mut at = 0;
        while at < cells.len() {
            if cells[at] {
                let start = at;
                while at < cells.len() && cells[at] {
                    at += 1;
                }
                runs.push([start as u32, at as u32]);
            } else {
                at += 1;
            }
        }
        runs
    };
    if let Some((cells, stride)) = dense {
        return Some(
            (0..rows)
                .map(|r| runs_of(&cells[r * stride..(r + 1) * stride]))
                .collect(),
        );
    }
    let masking = masking?;
    Some(
        (0..rows)
            .map(|r| {
                let mask = match masking {
                    engine::fire::Masking::Extent(mask) => mask,
                    engine::fire::Masking::Rows(masks) => match masks.get(r) {
                        Some(mask) => mask,
                        None => return Vec::new(),
                    },
                };
                mask.admitted(0..mask.len())
                    .map(|span| [span.start as u32, span.end as u32])
                    .collect()
            })
            .collect(),
    )
}

/// `request` for cache `cache` (named only past the first, so a daemon of
/// one cache never sees the key).
fn with_cache(mut request: serde_json::Value, cache: usize) -> serde_json::Value {
    if cache > 0 {
        request["cache"] = serde_json::json!(cache);
    }
    request
}

/// What the KV cache holds, as tokens: the model (daemon or wafer) keeps
/// one sequence's cache, so every lane's cache is kept here as the tokens
/// that made it — by slot, and by pool page and offset for what `copy_kv`
/// moves between pages — and the model's cache is brought to a lane's
/// history before its rows run: kept as far as it agrees, cut there
/// (`truncate`), and the rest replayed.
///
/// A lane's history is its slot's, unless the slot holds too little or a
/// copy landed in the lane's pages since (a fork, a shared prefix): then
/// its pages'. (Two lanes of one fire may name the same pool pages, so a
/// page's cells alone do not say whose a cache is.)
#[derive(Default)]
struct Kv {
    page_size: u32,
    /// Pool page -> the token at each offset (`None`: never written).
    pages: std::collections::HashMap<u32, Vec<Option<u32>>>,
    /// The slot's tokens.
    slots: std::collections::HashMap<u32, Vec<u32>>,
    /// Pages a copy wrote since a lane last fired on them.
    copied: std::collections::HashSet<u32>,
    /// On a paged model: page cells record what the model's pages hold, so
    /// only paged fires (and copies) write them; slots whose pages the
    /// model never got (a fire that named another lane's pages) replay
    /// from their own tokens until they start over.
    paged: bool,
    tainted: std::collections::HashSet<u32>,
    /// Each of the model's caches: the tokens it holds, position by
    /// position, the argmax after each where a token loop told it, and when
    /// it last ran (to give the least recent away).
    cached: Vec<Cached>,
    clock: u64,
    /// The host tier: suspended pages by host page id, the device page's
    /// words and its cells' tokens; at most `host_pages` of them.
    host: std::collections::HashMap<u32, (Vec<u32>, Option<Vec<Option<u32>>>)>,
    host_pages: u32,
}

impl Kv {
    /// The lane's first `n` positions' tokens.
    fn history(&self, slot: u32, pages: &[u32], n: u32) -> EngineResult<Vec<u32>> {
        let size = self.page_size.max(1);
        let copied_in = pages
            .iter()
            .take(n.div_ceil(size) as usize)
            .any(|page| self.copied.contains(page));
        let slot_holds = self.slots.get(&slot).is_some_and(|h| h.len() >= n as usize);
        if pages.is_empty() || (slot_holds && !copied_in) {
            let have = self.slots.get(&slot).map_or(&[][..], Vec::as_slice);
            return have.get(..n as usize).map(<[u32]>::to_vec).ok_or_else(|| {
                Error::Invalid(format!(
                    "slot {slot} fires at position {n} and the spatial engine holds {} of its \
                     tokens",
                    have.len()
                ))
            });
        }
        (0..n)
            .map(|p| {
                let page = pages.get((p / size) as usize).copied();
                page.and_then(|page| self.pages.get(&page))
                    .and_then(|cells| cells.get((p % size) as usize).copied().flatten())
                    .ok_or_else(|| {
                        Error::Invalid(format!(
                            "position {p} of slot {slot} (page {page:?}) holds no token this \
                             engine wrote or copied: the spatial engine keeps a cache as the \
                             tokens that made it"
                        ))
                    })
            })
            .collect()
    }

    /// Whether the cells of `pages` below slot `upto` all hold a token, or
    /// are in `planned` (written earlier in the same fire).
    fn known(
        &self,
        pages: &[u32],
        upto: u32,
        planned: &std::collections::HashSet<(u32, u32)>,
    ) -> bool {
        let size = self.page_size.max(1);
        (0..upto).all(|q| {
            let Some(&page) = pages.get((q / size) as usize) else {
                return false;
            };
            planned.contains(&(page, q % size))
                || self
                    .pages
                    .get(&page)
                    .and_then(|cells| cells.get((q % size) as usize).copied().flatten())
                    .is_some()
        })
    }

    /// A paged fire's rows: `tokens` at KV `slots` of `pages`, and the
    /// slot's own tokens where the rows continue them.
    fn write_paged(&mut self, slot: u32, pages: &[u32], slots: &[u32], tokens: &[u32]) {
        let size = self.page_size.max(1);
        for (&q, &tok) in slots.iter().zip(tokens) {
            if let Some(&page) = pages.get((q / size) as usize) {
                let cells = self.pages.entry(page).or_insert_with(|| vec![None; size as usize]);
                cells[(q % size) as usize] = Some(tok);
            }
        }
        for page in pages {
            self.copied.remove(page);
        }
        let first = slots.first().copied().unwrap_or(0);
        let contiguous = slots.iter().enumerate().all(|(i, &q)| q == first + i as u32);
        match self.slots.get_mut(&slot) {
            Some(have) if contiguous && have.len() >= first as usize => {
                have.truncate(first as usize);
                have.extend_from_slice(tokens);
            }
            _ if contiguous && first == 0 => {
                self.slots.insert(slot, tokens.to_vec());
            }
            _ => {
                self.slots.remove(&slot);
            }
        }
    }

    /// `tokens` written at positions `start..` of the lane.
    fn write(
        &mut self,
        slot: u32,
        pages: &[u32],
        history: Vec<u32>,
        tokens: &[u32],
    ) -> EngineResult<()> {
        let start = history.len() as u32;
        let mut have = history;
        have.extend_from_slice(tokens);
        self.slots.insert(slot, have);
        for page in pages {
            self.copied.remove(page);
        }
        if pages.is_empty() || self.paged {
            return Ok(());
        }
        let size = self.page_size.max(1);
        for (i, &tok) in tokens.iter().enumerate() {
            let p = start + i as u32;
            let page = *pages.get((p / size) as usize).ok_or_else(|| {
                Error::Invalid(format!(
                    "slot {slot} writes position {p} past its {} page(s)",
                    pages.len()
                ))
            })?;
            let cells = self.pages.entry(page).or_insert_with(|| vec![None; size as usize]);
            cells[(p % size) as usize] = Some(tok);
        }
        Ok(())
    }
}

#[derive(Default, Clone)]
struct Cached {
    tokens: Vec<u32>,
    argmax: Vec<Option<u32>>,
    used: u64,
    /// Steps the next token loop on this cache runs past its row: it
    /// doubles while loops are used to their end and starts over at
    /// [`FIRST_LOOP`] otherwise, so a sequence that ends wastes at most
    /// about what it used.
    next_loop: usize,
}

/// The guest VM's limits a table must fit (guest4's simulator build: wide
/// registers, pool words).
const GUEST_REGS: u32 = 12;
const GUEST_POOL: u32 = 4096;

/// Steps a sequence's first token loop runs past its row.
const FIRST_LOOP: usize = 8;

impl Cached {
    fn truncate(&mut self, n: usize) {
        self.tokens.truncate(n);
        self.argmax.truncate(n);
    }

    fn extend(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
        self.argmax.resize(self.tokens.len(), None);
    }
}

/// Whether `mask` (alternating runs: excluded, admitted, ...) lets a row at
/// `position` see every position up to its own: the causal mask the model
/// runs. Anything else this engine cannot run.
fn causal(mask: &engine::fire::Mask, position: u64) -> bool {
    let want = position + 1;
    let admitted: u64 = mask.admitted(0..want).map(|r| r.end - r.start).sum();
    // A mask over fewer positions says nothing of those past its end.
    admitted == want.min(mask.len())
}

pub struct Spatial {
    programs: crate::program::Plane,
    boot: DeviceBoot,
    spatial: SpatialBoot,
    contract_for: ContractFor,
    /// A dry shell: the trace, its classes, paging and readout, no device.
    shell: Option<Shell>,
    caps: Option<Capabilities>,
    readout: Option<(String, u32)>,
    daemon: Option<Model>,
    kv: Kv,
    /// Programs whose guest translation was told, and those the guest VM
    /// can run, by program id.
    guest_probed: std::collections::HashSet<u64>,
    guest_programs: std::collections::HashMap<u64, std::sync::Arc<guest4::lane::Program>>,
    /// Programs the VM could take that the cost forecast keeps on the host,
    /// and why.
    guest_costly: std::collections::HashMap<u64, String>,
    /// The model's parameter count (the forecast's default token time).
    params: f64,
    /// The guest VM a lane's instance runs on, while it does.
    guest_vms: std::collections::HashMap<InstanceId, guest4::lane::CheckVm>,
    /// The instance whose program the wafer head's VM holds (`guest`
    /// "wafer"): its homes live there while it does.
    vm_owner: Option<InstanceId>,
    /// A multi-lane VM: the program it holds and each lane's request.
    vm_program: Option<u64>,
    vm_banks: Vec<Option<InstanceId>>,
    /// A per-lane-program VM: each lane's program as placed (empty until
    /// the first placement).
    vm_bank_progs: Vec<Option<u64>>,
    /// Fires run (the model-time log's count).
    fires: u64,
    /// Epilogue steps the guest VM ran.
    guest_steps: u64,
    /// Rows of history replayed into the model's cache (the fallback for a
    /// model that owns no paged KV), and fires run on paged KV.
    replayed: u64,
    paged_fires: u64,
    /// Rows re-prefilled into a lane's own pages.
    refilled: u64,
    /// Greedy rows a token loop answered, and the loops run.
    ahead_hits: u64,
    ahead_runs: u64,
    next_fire: u64,
    next_frame: u64,
    sink: Option<engine::CompletionSink>,
    pending: Option<(FrameId, Vec<Vec<LaneReadout>>)>,
}

impl Spatial {
    #[must_use]
    pub fn new(boot: DeviceBoot, spatial: SpatialBoot, contract_for: ContractFor) -> Spatial {
        Spatial {
            programs: crate::program::Plane::default(),
            boot,
            spatial,
            contract_for,
            shell: None,
            caps: None,
            readout: None,
            daemon: None,
            kv: Kv::default(),
            guest_probed: std::collections::HashSet::new(),
            guest_programs: std::collections::HashMap::new(),
            guest_costly: std::collections::HashMap::new(),
            params: 0.0,
            guest_vms: std::collections::HashMap::new(),
            vm_owner: None,
            vm_program: None,
            vm_banks: Vec::new(),
            vm_bank_progs: Vec::new(),
            fires: 0,
            guest_steps: 0,
            replayed: 0,
            paged_fires: 0,
            refilled: 0,
            ahead_hits: 0,
            ahead_runs: 0,
            next_fire: 0,
            next_frame: 0,
            sink: None,
            pending: None,
        }
    }

    fn daemon(&mut self) -> EngineResult<&mut Model> {
        self.daemon
            .as_mut()
            .ok_or_else(|| Error::Load("the spatial engine has no model loaded".into()))
    }

    /// Runs `tokens` at positions `start..` for the lane (`slot`, its
    /// `pages`), the model's cache first brought to the lane's history; the
    /// logits of `rows` (indexes into `tokens`), each the vocabulary wide.
    ///
    /// `greedy`: what reads the lane's last row takes only its argmax. Then
    /// the model runs its token loop past the row (`lookahead` steps more,
    /// one launch on the wafer, no logits back), later rows the loop already
    /// ran answer from it without running, and the row comes back as a
    /// one-hot of its argmax.
    fn forward(
        &mut self,
        slot: u32,
        pages: &[u32],
        tokens: &[u32],
        start: u32,
        rows: &[u32],
        greedy: bool,
        ahead: Option<u32>,
    ) -> EngineResult<(Vec<Vec<f32>>, bool)> {
        let history = self.kv.history(slot, pages, start)?;
        let caches = self.daemon()?.caches();
        self.kv.cached.resize_with(caches, Default::default);
        let last_only = rows.len() == 1 && rows[0] as usize + 1 == tokens.len();
        let lookahead = self.lookahead();
        let greedy = greedy && last_only && lookahead > 0;
        let width = self.readout.as_ref().map_or(0, |(_, w)| *w as usize);
        let one_hot = |token: u32| {
            let mut row = vec![0f32; width];
            if let Some(cell) = row.get_mut(token as usize) {
                *cell = 1.0;
            }
            vec![row]
        };
        let fed: Vec<u32> = history.iter().chain(tokens).copied().collect();
        if greedy {
            // A token loop that already ran these rows answers them.
            let last = fed.len() - 1;
            let hit = self.kv.cached.iter().position(|c| {
                c.tokens.len() > last
                    && c.tokens[..=last] == fed[..]
                    && c.argmax.get(last).copied().flatten().is_some()
            });
            if let Some(cache) = hit {
                self.kv.clock += 1;
                self.kv.cached[cache].used = self.kv.clock;
                let token = self.kv.cached[cache].argmax[last].unwrap_or_default();
                self.ahead_hits += 1;
                self.kv.write(slot, pages, history, tokens)?;
                return Ok((one_hot(token), true));
            }
        }
        // The cache that agrees with the lane longest; of equals, the least
        // recently run.
        let agree_of = |cached: &[u32]| {
            cached
                .iter()
                .zip(&history)
                .take_while(|(a, b)| a == b)
                .count()
        };
        let (cache, agree) = self
            .kv
            .cached
            .iter()
            .enumerate()
            .map(|(c, cached)| (c, agree_of(&cached.tokens), cached.used))
            .max_by_key(|&(_, agree, used)| (agree, std::cmp::Reverse(used)))
            .map(|(c, agree, _)| (c, agree))
            .unwrap_or((0, 0));
        self.kv.clock += 1;
        self.kv.cached[cache].used = self.kv.clock;
        // This greedy row picks up where the cache's last token loop ended
        // (fed its last argmax): the loop was used to its end.
        let continues = {
            let c = &self.kv.cached[cache];
            agree == c.tokens.len()
                && agree == history.len()
                && agree > 0
                && c.argmax.last().copied().flatten() == tokens.first().copied()
        };
        if agree < self.kv.cached[cache].tokens.len() {
            // Another sequence, a rewind or rejected rows: keep what agrees.
            if agree == 0 {
                self.daemon()?.reset(cache)?;
            } else {
                self.daemon()?.truncate(cache, agree)?;
            }
            self.kv.cached[cache].truncate(agree);
        }
        if agree < history.len() {
            self.replayed += (history.len() - agree) as u64;
            tracing::info!(
                rows = history.len() - agree,
                total = self.replayed,
                "spatial: history replayed"
            );
            self.daemon()?
                .forward(cache, &history[agree..], agree as u32, wafer::Logits::None)?;
            self.kv.cached[cache].extend(&history[agree..]);
        }
        if greedy {
            let steps = {
                let c = &mut self.kv.cached[cache];
                c.next_loop = if continues && c.next_loop > 0 {
                    (c.next_loop * 2).min(lookahead)
                } else {
                    FIRST_LOOP.min(lookahead)
                };
                // No further than the lane's pipeline will fire.
                ahead.map_or(c.next_loop, |fires| c.next_loop.min(fires as usize))
            };
            tracing::info!(
                rows = tokens.len(),
                steps,
                fires_after = ?ahead,
                "spatial: token loop"
            );
            let (ran, argmax) = self.daemon()?.greedy(cache, tokens, start, steps)?;
            self.ahead_runs += 1;
            if tracing::enabled!(target: "spatial_trace", tracing::Level::DEBUG) {
                tracing::debug!(target: "spatial_trace",
                    "spatial-trace token loop: {} run(s), {} row(s) answered from them",
                    self.ahead_runs, self.ahead_hits
                );
            }
            let cached = &mut self.kv.cached[cache];
            cached.tokens.extend_from_slice(&ran);
            cached.argmax.extend(argmax);
            cached.argmax.resize(cached.tokens.len(), None);
            let token = cached.argmax[fed.len() - 1].unwrap_or_default();
            self.kv.write(slot, pages, history, tokens)?;
            return Ok((one_hot(token), true));
        }
        let which = if rows.is_empty() {
            wafer::Logits::None
        } else if last_only {
            wafer::Logits::Last
        } else {
            wafer::Logits::All
        };
        let logits = self.daemon()?.forward(cache, tokens, start, which)?;
        self.kv.cached[cache].extend(tokens);
        self.kv.write(slot, pages, history, tokens)?;
        Ok((
            match which {
                wafer::Logits::None => Vec::new(),
                wafer::Logits::Last => logits,
                wafer::Logits::All => rows
                    .iter()
                    .map(|&r| logits.get(r as usize).cloned().unwrap_or_default())
                    .collect(),
            },
            false,
        ))
    }

    /// The most steps a greedy lane's token loop runs past its row (0: none):
    /// the config's `lookahead`, else `PIE_SPATIAL_LOOKAHEAD` (a development
    /// override where the config says nothing), else 32 on the wafer and 0
    /// on a daemon (a CPU reference pays for every step it runs ahead).
    fn lookahead(&self) -> usize {
        let default = match &self.daemon {
            Some(Model::Wafer(_)) => 32,
            _ => 0,
        };
        self.spatial
            .lookahead
            .or_else(|| std::env::var("PIE_SPATIAL_LOOKAHEAD").ok().and_then(|v| v.parse().ok()))
            .unwrap_or(default)
    }

    /// Whether the wafer's guest VM could run `instance`'s epilogue
    /// (`guest::translate` at the VM's limits), told once a program: the
    /// engine still runs every epilogue on the host; this says which would
    /// move to the wafer (`spatial/eta/CONTRACT.md`).
    fn probe_guest(&mut self, instance: InstanceId, program: ProgramId) {
        if !self.guest_probed.insert(program.0) {
            return;
        }
        let Some(plan) = self.programs.plan_of(instance) else {
            return;
        };
        let vocab = self.readout.as_ref().map_or(0, |(_, w)| *w);
        let host_takes: Vec<u32> = plan
            .package
            .channels
            .iter()
            .filter(|c| c.host_role == eta_ir::container::HostRole::Reader)
            .map(|c| c.id)
            .collect();
        let name = self.readout.as_ref().map_or("", |(n, _)| n.as_str());
        // Translated to the wafer VM's registers and pool when the model has
        // one (the translator re-makes draws at their readers to fit fewer
        // registers), else to the host's limits. A table whose pool overruns
        // is translated again with the big logits-free arrays sunk into
        // main, whose words are then live only while main runs.
        let (regs, pool) = match &self.daemon {
            Some(Model::Wafer(w)) => w
                .vm
                .map_or((GUEST_REGS, GUEST_POOL), |(lay, _)| (lay.nreg as u32, lay.pool_words as u32)),
            _ => (GUEST_REGS, GUEST_POOL),
        };
        // Translated without a pool bound, so a table's size is known when
        // it overruns; the bound is applied below.
        let build = |sink: bool| {
            guest4::translate_sinking(&plan, vocab, &host_takes, regs, u32::MAX, sink)
                .and_then(|table| guest4::lane::Program::of(&plan, table, name))
        };
        let mut built = build(false);
        let over = built.as_ref().is_ok_and(|guest| guest.table.pool_words > pool);
        if over {
            let sunk = build(true);
            let better = match (&sunk, &built) {
                (Ok(s), Ok(b)) => s.table.pool_words < b.table.pool_words,
                _ => false,
            };
            tracing::info!(
                program = program.0,
                pool = built.as_ref().map_or(0, |g| g.table.pool_words),
                sunk_pool = sunk.as_ref().map_or(0, |g| g.table.pool_words),
                taken = better,
                "spatial guest: the pool overruns; translated again with logits-free arrays sunk into main"
            );
            if better {
                built = sunk;
            }
        }
        let built = built.and_then(|guest| {
            if guest.table.pool_words > pool {
                Err(format!("{} pool words, more than {pool}", guest.table.pool_words))
            } else {
                Ok(guest)
            }
        });
        match built {
            Ok(guest) => {
                tracing::info!(
                    program = program.0,
                    instructions = guest.table.ins.len(),
                    pre = guest.table.pre_len,
                    registers = guest.table.nreg,
                    pool = guest.table.pool_words,
                    homes = guest.table.homes.len(),
                    "spatial guest: the epilogue would run on the wafer"
                );
                if let Some(Model::Wafer(w)) = &self.daemon
                    && let Some((lay, ring)) = w.vm
                {
                    // A multi-lane VM drives batched lanes; one lane, a single sequence.
                    let refused = if ring.lanes > 1 {
                        lay.refuses(&guest.table).or_else(|| ring.refuses_outputs(&guest.table.outputs))
                    } else {
                        w.vm_refuses(&guest.table)
                    };
                    match refused {
                        Some(why) => tracing::info!(program = program.0, %why, "spatial guest: the VM cannot take it"),
                        None => {
                            let placement = self.placement();
                            let f = guest4::cost::forecast(&guest.table, lay.vr as u32);
                            match placement.refuses(f) {
                                Some(why) => {
                                    tracing::info!(program = program.0, %why, pre = f.pre, main = f.main, token = placement.token_cycles, "spatial guest: the VM could take it; the forecast keeps it on the host");
                                    self.guest_costly.insert(program.0, why);
                                }
                                None => tracing::info!(program = program.0, pre = f.pre, main = f.main, token = placement.token_cycles, "spatial guest: the VM can take it"),
                            }
                        }
                    }
                }
                self.guest_programs.insert(program.0, std::sync::Arc::new(guest));
            }
            Err(why) => tracing::info!(
                program = program.0,
                %why,
                "spatial guest: the epilogue stays on the host"
            ),
        }
    }

    /// When a table the VM can take runs there: the config's
    /// `token_cycles`, `pre_margin` and `main_fraction`, else the token from
    /// the model's size and `COVERAGE.md`'s thresholds.
    fn placement(&self) -> guest4::cost::Placement {
        use guest4::cost::{Placement, token_cycles};
        Placement {
            token_cycles: self.spatial.token_cycles.unwrap_or_else(|| token_cycles(self.params)),
            pre_margin: self.spatial.pre_margin.unwrap_or(Placement::PRE_MARGIN),
            main_fraction: self.spatial.main_fraction.unwrap_or(Placement::MAIN_FRACTION),
        }
    }

    /// Who runs translatable epilogues: the config's `guest`, else
    /// `PIE_SPATIAL_GUEST` (a development override where the config says
    /// nothing): `"off"` (the host interpreter, the default),
    /// `"check"` (the contract played on the host: [`guest4::lane::CheckVm`])
    /// or `"wafer"` (the VM on the serving head, once the program has one;
    /// until then the host).
    fn guest_mode(&self) -> String {
        let mode = self
            .spatial
            .guest
            .clone()
            .or_else(|| std::env::var("PIE_SPATIAL_GUEST").ok())
            .unwrap_or_else(|| "off".to_string());
        if matches!(mode.as_str(), "off" | "check" | "wafer") {
            mode
        } else {
            tracing::warn!(mode, "spatial guest: not off, check or wafer; epilogues stay on the host");
            "off".to_string()
        }
    }

    /// Whether the wafer head's VM runs `instance`'s epilogue (`guest`
    /// "wafer"): a one-sequence program with a VM, the instance's program
    /// translated and within the VM's limits, and the VM free or already
    /// its.
    fn vm_takes(&self, instance: InstanceId) -> bool {
        if self.guest_mode() != "wafer" {
            return false;
        }
        let Some(Model::Wafer(w)) = &self.daemon else {
            return false;
        };
        if w.vm.is_none() || w.paged.is_some() {
            return false;
        }
        let Some((program, _, _)) = self.programs.instance_view(instance) else {
            return false;
        };
        let Some(guest) = self.guest_programs.get(&program.0) else {
            return false;
        };
        if let Some(why) = w.vm_refuses(&guest.table).or_else(|| self.guest_costly.get(&program.0).cloned()) {
            tracing::debug!(program = program.0, %why, "spatial guest: the VM cannot take it");
            return false;
        }
        // One lane's homes at a time; another instance's stay on the host
        // until the VM's owner is gone.
        self.vm_owner.is_none_or(|owner| {
            owner == instance || self.programs.instance_view(owner).is_none()
        })
    }

    /// `tokens` at `start` for the lane (`slot`, its `pages`) with the
    /// head's VM running `instance`'s epilogue on the last row: the model's
    /// cache brought to the lane's history, every row but the last run
    /// (the head's argmax mode), then the last as a VM step. The step's
    /// `out` words and the VM's pool after it.
    fn forward_vm(
        &mut self,
        instance: InstanceId,
        slot: u32,
        pages: &[u32],
        tokens: &[u32],
        start: u32,
    ) -> EngineResult<(Vec<u32>, Vec<u32>)> {
        let device = |e: String| Error::Device(format!("the wafer model's VM: {e}"));
        if self.vm_owner != Some(instance) {
            let (program, _, inst) = self
                .programs
                .instance_view(instance)
                .ok_or_else(|| Error::Invalid(format!("instance {instance:?} is gone")))?;
            let guest = self
                .guest_programs
                .get(&program.0)
                .cloned()
                .ok_or_else(|| Error::Invalid("the program has no guest table".into()))?;
            // Its carried state: every home's value as it stands.
            let seeds: Vec<guest4::wafer::Seed> = guest
                .table
                .homes
                .iter()
                .map(|h| guest4::wafer::Seed::of(h, guest4::lane::words_of(&inst.channels[h.chan as usize].current())))
                .collect();
            let Some(Model::Wafer(w)) = self.daemon.as_mut() else {
                return Err(Error::Device("the VM path without a wafer model".into()));
            };
            w.vm_load(guest, seeds).map_err(device)?;
            self.vm_owner = Some(instance);
            tracing::info!(?instance, program = program.0, "spatial guest: the program runs on the wafer's VM");
        }
        let history = self.kv.history(slot, pages, start)?;
        let caches = self.daemon()?.caches();
        self.kv.cached.resize_with(caches, Default::default);
        let agree_of = |cached: &[u32]| cached.iter().zip(&history).take_while(|(a, b)| a == b).count();
        let (cache, agree) = self
            .kv
            .cached
            .iter()
            .enumerate()
            .map(|(c, cached)| (c, agree_of(&cached.tokens), cached.used))
            .max_by_key(|&(_, agree, used)| (agree, std::cmp::Reverse(used)))
            .map(|(c, agree, _)| (c, agree))
            .unwrap_or((0, 0));
        self.kv.clock += 1;
        self.kv.cached[cache].used = self.kv.clock;
        if agree < self.kv.cached[cache].tokens.len() {
            if agree == 0 {
                self.daemon()?.reset(cache)?;
            } else {
                self.daemon()?.truncate(cache, agree)?;
            }
            self.kv.cached[cache].truncate(agree);
        }
        if agree < history.len() {
            self.replayed += (history.len() - agree) as u64;
            self.daemon()?
                .forward(cache, &history[agree..], agree as u32, wafer::Logits::None)?;
            self.kv.cached[cache].extend(&history[agree..]);
        }
        let n = tokens.len();
        if n > 1 {
            self.daemon()?
                .forward(cache, &tokens[..n - 1], start, wafer::Logits::None)?;
        }
        let Some(Model::Wafer(w)) = self.daemon.as_mut() else {
            return Err(Error::Device("the VM path without a wafer model".into()));
        };
        let (_, row, pool) = w.vm_step(tokens[n - 1], (start as usize) + n - 1).map_err(device)?;
        self.kv.cached[cache].extend(tokens);
        self.kv.write(slot, pages, history, tokens)?;
        Ok((row, pool))
    }

    /// Whether the wafer head's multi-lane VM (serve2d_pipe `VM` 1, `gring`
    /// lanes L > 1) samples `instance`'s rows (`guest` "wafer"): its
    /// translated program fits the VM and is the one it holds (or the VM
    /// holds none a live request still uses), and its lane has a bank (it
    /// keeps its own, or one is free).
    fn vm_lanes_take(&self, instance: InstanceId) -> bool {
        if self.guest_mode() != "wafer" {
            return false;
        }
        let Some(Model::Wafer(w)) = &self.daemon else {
            return false;
        };
        let Some((lay, ring)) = w.vm else {
            return false;
        };
        if ring.lanes < 2 || w.paged.is_none() {
            return false;
        }
        let Some((program, _, _)) = self.programs.instance_view(instance) else {
            return false;
        };
        let Some(guest) = self.guest_programs.get(&program.0) else {
            return false;
        };
        if self.guest_costly.contains_key(&program.0)
            || lay.refuses(&guest.table).or_else(|| ring.refuses_outputs(&guest.table.outputs)).is_some()
        {
            return false;
        }
        let live = |owner: &Option<InstanceId>| owner.is_some_and(|o| self.programs.instance_view(o).is_some());
        // One program for every lane, unless the image runs a program a lane.
        if !lay.lprog && self.vm_program.is_some_and(|p| p != program.0) && self.vm_banks.iter().any(live) {
            return false;
        }
        self.vm_banks.iter().any(|o| *o == Some(instance))
            || self.vm_banks.len() < ring.lanes as usize
            || self.vm_banks.iter().any(|o| !live(o))
    }

    /// One multi-lane VM launch for `lanes` (each its instance and its
    /// decode row): the program loaded if the VM holds another, each
    /// instance's bank taken and seeded from its channels the first time,
    /// the launch's VM lanes filled in bank order (an idle one repeats a
    /// live lane, its bank kept as it was when a live request owns it).
    /// Per lane: its instance, the turn's `out` words, its pool after it;
    /// `None` for a lane no VM lane was free for (the host samples it).
    #[allow(clippy::type_complexity)]
    fn fire_vm_lanes(
        &mut self,
        lanes: &[(InstanceId, &PagedLane)],
    ) -> EngineResult<Vec<Option<(InstanceId, Vec<u32>, Vec<u32>)>>> {
        let device = |e: String| Error::Device(format!("the wafer model's VM: {e}"));
        let Some(Model::Wafer(w)) = &self.daemon else {
            return Err(Error::Device("the VM path without a wafer model".into()));
        };
        let (lay, ring) = w.vm.ok_or_else(|| Error::Device("the wafer program has no VM".into()))?;
        let n = ring.lanes as usize;
        // Banks of requests that are gone are free.
        let alive: Vec<bool> = self
            .vm_banks
            .iter()
            .map(|o| o.is_some_and(|o| self.programs.instance_view(o).is_some()))
            .collect();
        for (owner, live) in self.vm_banks.iter_mut().zip(alive) {
            if !live {
                *owner = None;
            }
        }
        let placed = if lay.lprog {
            self.place_vm_lanes(lanes, n)?
        } else {
            self.seed_vm_lanes(lanes, n)?
        };
        let Some(lead) = lanes.iter().zip(&placed).find(|(_, b)| b.is_some()).map(|(l, _)| l.1) else {
            return Ok(vec![None; lanes.len()]);
        };
        // The launch in bank order.
        let batch_of = |lane: &PagedLane| serve2d::BatchLane {
            pos: lane.positions[0],
            slot: lane.slots[0],
            pages: lane.pages.clone(),
            visible: None,
            forced: lane.tokens.clone(),
        };
        let mut launch: Vec<serve2d::BatchLane> = vec![batch_of(lead); n];
        for (&(_, lane), bank) in lanes.iter().zip(&placed) {
            if let Some(bank) = *bank {
                launch[bank] = batch_of(lane);
            }
        }
        let keep: Vec<usize> = (0..n)
            .filter(|b| !placed.contains(&Some(*b)) && self.vm_banks[*b].is_some())
            .collect();
        let mut inputs: Vec<wafer::StepInput> = vec![None; n];
        for (&(instance, _), bank) in lanes.iter().zip(&placed) {
            if let Some(bank) = *bank {
                inputs[bank] = self.step_input(instance, &lay)?;
            }
        }
        let Some(Model::Wafer(w)) = self.daemon.as_mut() else {
            return Err(Error::Device("the VM path without a wafer model".into()));
        };
        let steps = w.vm_fire(launch, keep, inputs).map_err(device)?;
        self.paged_fires += 1;
        Ok(lanes
            .iter()
            .zip(&placed)
            .map(|(&(instance, _), bank)| {
                bank.map(|bank| {
                    let (_, row, pool) = steps[bank].clone();
                    (instance, row, pool)
                })
            })
            .collect())
    }

    /// A suspension's pages to the host tier: whole pages only (the host
    /// never attends a page; a lane parks and resumes whole).
    fn page_out(&mut self, copy: &engine::transfer::KvCopy) -> EngineResult<()> {
        if !copy.moves.is_empty() || !copy.windowed.is_empty() {
            return Err(Error::Unsupported { verb: "`copy_kv` of part of a page to the host", engine: "cerebras-spatial" });
        }
        if let Some(&page) = copy.dst_page_ids.iter().find(|&&p| p >= self.kv.host_pages) {
            return Err(Error::Invalid(format!("host page {page} of a host tier of {}", self.kv.host_pages)));
        }
        let Some(Model::Wafer(w)) = self.daemon.as_mut() else {
            return Err(Error::Device("the host tier without a wafer model".into()));
        };
        let t0 = std::time::Instant::now();
        let words = w
            .page_out(copy.src_page_ids.clone())
            .map_err(|e| Error::Device(format!("pages to the host: {e}")))?;
        for ((&src, &dst), words) in copy.src_page_ids.iter().zip(&copy.dst_page_ids).zip(words) {
            let cells = self.kv.pages.get(&src).cloned();
            self.kv.host.insert(dst, (words, cells));
        }
        tracing::debug!(pages = copy.src_page_ids.len(), ms = t0.elapsed().as_secs_f64() * 1e3, "spatial: kv pages to the host");
        Ok(())
    }

    /// A resumption's pages back from the host tier.
    fn page_in(&mut self, copy: &engine::transfer::KvCopy) -> EngineResult<()> {
        if !copy.moves.is_empty() || !copy.windowed.is_empty() {
            return Err(Error::Unsupported { verb: "`copy_kv` of part of a page from the host", engine: "cerebras-spatial" });
        }
        let mut pages = Vec::with_capacity(copy.src_page_ids.len());
        for (&src, &dst) in copy.src_page_ids.iter().zip(&copy.dst_page_ids) {
            let (words, _) = self
                .kv
                .host
                .get(&src)
                .ok_or_else(|| Error::Invalid(format!("host page {src} holds nothing")))?;
            pages.push((dst, words.clone()));
        }
        let Some(Model::Wafer(w)) = self.daemon.as_mut() else {
            return Err(Error::Device("the host tier without a wafer model".into()));
        };
        let t0 = std::time::Instant::now();
        w.page_in(pages).map_err(|e| Error::Device(format!("pages from the host: {e}")))?;
        let size = self.kv.page_size.max(1) as usize;
        for (&src, &dst) in copy.src_page_ids.iter().zip(&copy.dst_page_ids) {
            let cells = self.kv.host.get(&src).and_then(|(_, c)| c.clone()).unwrap_or_else(|| vec![None; size]);
            self.kv.pages.insert(dst, cells);
            self.kv.copied.insert(dst);
        }
        tracing::debug!(pages = copy.src_page_ids.len(), ms = t0.elapsed().as_secs_f64() * 1e3, "spatial: kv pages from the host");
        Ok(())
    }

    /// A single-program multi-lane VM's banks for `lanes`: the program
    /// (the first lane's) loaded if the VM holds another, each lane its own
    /// bank, else a free one seeded from its channels; `None` for a lane no
    /// bank was free for.
    fn seed_vm_lanes(&mut self, lanes: &[(InstanceId, &PagedLane)], n: usize) -> EngineResult<Vec<Option<usize>>> {
        let device = |e: String| Error::Device(format!("the wafer model's VM: {e}"));
        let first = lanes.first().map(|(i, _)| *i).ok_or_else(|| Error::Invalid("no VM lanes".into()))?;
        let (program, guest) = self.guest_of(first)?;
        if self.vm_program != Some(program) {
            let Some(Model::Wafer(w)) = self.daemon.as_mut() else {
                return Err(Error::Device("the VM path without a wafer model".into()));
            };
            w.vm_load_lanes(std::sync::Arc::clone(&guest)).map_err(device)?;
            self.vm_program = Some(program);
            self.vm_banks = vec![None; n];
            tracing::info!(program, lanes = n, "spatial guest: the program runs on the wafer's multi-lane VM");
        }
        self.vm_banks.resize(n, None);
        let mut placed: Vec<Option<usize>> = Vec::with_capacity(lanes.len());
        for &(instance, _) in lanes {
            let bank = if let Some(bank) = self.vm_banks.iter().position(|o| *o == Some(instance)) {
                bank
            } else {
                let Some(bank) = self.vm_banks.iter().position(Option::is_none) else {
                    placed.push(None);
                    continue;
                };
                let seeds = self.seeds_of(instance, &guest)?;
                let Some(Model::Wafer(w)) = self.daemon.as_mut() else {
                    return Err(Error::Device("the VM path without a wafer model".into()));
                };
                w.vm_seed_bank(bank, seeds).map_err(device)?;
                self.vm_banks[bank] = Some(instance);
                tracing::info!(?instance, bank, "spatial guest: a request takes a VM lane");
                bank
            };
            placed.push(Some(bank));
        }
        Ok(placed)
    }

    /// A per-lane-program VM's banks for `lanes`: each lane its own bank,
    /// else a free one; when a lane comes in, every live lane's program
    /// placed again (the arena and windows compacted, the newcomers seeded
    /// from their channels). A newcomer the image has no room for stays on
    /// the host (`None`).
    fn place_vm_lanes(&mut self, lanes: &[(InstanceId, &PagedLane)], n: usize) -> EngineResult<Vec<Option<usize>>> {
        let device = |e: String| Error::Device(format!("the wafer model's VM: {e}"));
        self.vm_banks.resize(n, None);
        let mut placed: Vec<Option<usize>> = Vec::with_capacity(lanes.len());
        let mut fresh: Vec<usize> = Vec::new();
        for &(instance, _) in lanes {
            let bank = self.vm_banks.iter().position(|o| *o == Some(instance)).or_else(|| {
                let bank = self.vm_banks.iter().position(Option::is_none)?;
                self.vm_banks[bank] = Some(instance);
                fresh.push(bank);
                Some(bank)
            });
            placed.push(bank);
        }
        if fresh.is_empty() && !self.vm_bank_progs.is_empty() {
            return Ok(placed);
        }
        let mut programs: Vec<Option<guest4::slots::LaneProgram>> = vec![None; n];
        let mut seeds: Vec<Option<Vec<guest4::wafer::Seed>>> = (0..n).map(|_| None).collect();
        for bank in 0..n {
            let Some(owner) = self.vm_banks[bank] else {
                continue;
            };
            let (key, program) = self.guest_of(owner)?;
            if fresh.contains(&bank) {
                seeds[bank] = Some(self.seeds_of(owner, &program)?);
            }
            programs[bank] = Some(guest4::slots::LaneProgram { key, program });
        }
        let Some(Model::Wafer(w)) = self.daemon.as_mut() else {
            return Err(Error::Device("the VM path without a wafer model".into()));
        };
        if let Err(why) = w.vm_place_lanes(programs.clone(), seeds) {
            // The newcomers stay on the host; the lanes already placed run on.
            tracing::info!(%why, lanes = fresh.len(), "spatial guest: no room on the VM for new lanes");
            for &bank in &fresh {
                self.vm_banks[bank] = None;
                programs[bank] = None;
            }
            for slot in &mut placed {
                if slot.is_some_and(|b| fresh.contains(&b)) {
                    *slot = None;
                }
            }
            if self.vm_bank_progs.is_empty() {
                return Ok(placed);
            }
            let seeds = (0..n).map(|_| None).collect();
            let Some(Model::Wafer(w)) = self.daemon.as_mut() else {
                return Err(Error::Device("the VM path without a wafer model".into()));
            };
            w.vm_place_lanes(programs.clone(), seeds).map_err(device)?;
        } else {
            for &bank in &fresh {
                tracing::info!(instance = ?self.vm_banks[bank], bank, "spatial guest: a request takes a VM lane");
            }
        }
        self.vm_bank_progs = programs.iter().map(|p| p.as_ref().map(|p| p.key)).collect();
        Ok(placed)
    }

    /// `instance`'s program id and guest table.
    fn guest_of(&self, instance: InstanceId) -> EngineResult<(u64, std::sync::Arc<guest4::lane::Program>)> {
        let (program, _, _) = self
            .programs
            .instance_view(instance)
            .ok_or_else(|| Error::Invalid(format!("instance {instance:?} is gone")))?;
        let guest = self
            .guest_programs
            .get(&program.0)
            .cloned()
            .ok_or_else(|| Error::Invalid("the program has no guest table".into()))?;
        Ok((program.0, guest))
    }

    /// `instance`'s homes' seeds: its channels as they stand.
    fn seeds_of(&self, instance: InstanceId, guest: &guest4::lane::Program) -> EngineResult<Vec<guest4::wafer::Seed>> {
        let (_, _, inst) = self
            .programs
            .instance_view(instance)
            .ok_or_else(|| Error::Invalid(format!("instance {instance:?} is gone")))?;
        Ok(guest
            .table
            .homes
            .iter()
            .map(|h| guest4::wafer::Seed::of(h, guest4::lane::words_of(&inst.channels[h.chan as usize].current())))
            .collect())
    }

    /// `instance`'s step input this launch: the homes its program has the
    /// host write each step, from their channels (what the inferlet put).
    fn step_input(&self, instance: InstanceId, lay: &guest4::wafer::VmLayout) -> EngineResult<wafer::StepInput> {
        let (_, guest) = self.guest_of(instance)?;
        if guest.table.per_step.is_empty() {
            return Ok(None);
        }
        let entries = guest4::slots::step_entries(lay, &guest.table).map_err(Error::Device)?;
        let (_, _, inst) = self
            .programs
            .instance_view(instance)
            .ok_or_else(|| Error::Invalid(format!("instance {instance:?} is gone")))?;
        let values = entries
            .iter()
            .map(|e| guest4::lane::words_of(&inst.channels[e.chan as usize].current()))
            .collect();
        Ok(Some((entries, values)))
    }

    /// `instance`'s epilogue as the head's VM ran it: eta-exec steps the
    /// instance through [`guest4::lane::VmStage`] on the step's `out` words,
    /// and the pool homes (the next token's among them) come back from the
    /// VM's pool; wide homes stay on the wafer.
    fn fire_vm(&mut self, instance: InstanceId, row: &[u32], pool: &[u32]) -> EngineResult<()> {
        let Some((program, _, _)) = self.programs.instance_view(instance) else {
            return Ok(());
        };
        let guest = self
            .guest_programs
            .get(&program.0)
            .cloned()
            .ok_or_else(|| Error::Invalid("the program has no guest table".into()))?;
        // The pool homes the step put (the next token's among them), as the
        // VM's pool holds them: put on the instance; wide homes stay on the
        // wafer.
        let homes = {
            let Some((_, plan, _)) = self.programs.instance_view(instance) else {
                return Ok(());
            };
            let puts: std::collections::BTreeSet<u32> =
                plan.package.stages[guest.epilogue].puts.iter().map(|p| p.channel).collect();
            guest
                .table
                .homes
                .iter()
                .filter(|h| h.kind == "pool" && puts.contains(&h.chan))
                .filter_map(|h| {
                    let (at, n) = (h.at as usize, h.n as usize);
                    pool.get(at..at + n).map(|words| (h.chan, guest4::lane::value_of(words, h.dtype)))
                })
                .collect::<Vec<_>>()
        };
        self.programs
            .fire_with(
                instance,
                &[],
                Stage::Epilogue,
                &mut guest4::lane::VmStage {
                    program: &guest,
                    out: row,
                    wide: &[],
                    homes: &homes,
                },
            )
            .map_err(refusal)?;
        self.guest_steps += 1;
        if self.guest_steps.is_power_of_two() {
            tracing::info!(steps = self.guest_steps, program = program.0, "spatial guest: epilogue steps run on the wafer's VM");
        }
        Ok(())
    }

    /// `instance`'s epilogue through the guest contract, if its program has
    /// a table and the mode runs one; `false` when the host interpreter
    /// should run it as usual.
    fn fire_guest(
        &mut self,
        instance: InstanceId,
        outputs: &[(String, u32, Value)],
    ) -> EngineResult<bool> {
        if self.guest_mode() != "check" {
            return Ok(false);
        }
        let Some((program, plan, inst)) = self.programs.instance_view(instance) else {
            return Ok(false);
        };
        let Some(guest) = self.guest_programs.get(&program.0).cloned() else {
            return Ok(false);
        };
        let vm = self
            .guest_vms
            .entry(instance)
            .or_insert_with(|| guest4::lane::CheckVm::load(&plan, inst));
        let (out, wide, homes) = vm
            .step(&plan, &guest, inst, outputs)
            .map_err(|why| Error::Device(format!("the guest VM: {why}")))?;
        self.guest_steps += 1;
        if self.guest_steps.is_power_of_two() {
            tracing::info!(
                steps = self.guest_steps,
                program = program.0,
                "spatial guest: epilogue steps run through the VM contract"
            );
        }
        self.programs
            .fire_with(
                instance,
                outputs,
                Stage::Epilogue,
                &mut guest4::lane::VmStage {
                    program: &guest,
                    out: &out,
                    wide: &wide,
                    homes: &homes,
                },
            )
            .map_err(refusal)?;
        if let (Some(vm), Some((_, _, inst))) = (
            self.guest_vms.get(&instance),
            self.programs.instance_view(instance),
        ) {
            vm.read_back(&guest, inst);
        }
        Ok(true)
    }

    /// Refuses a lane whose attention is not causal over its own history:
    /// the model runs that alone, and a mask it ignored would answer wrong.
    fn check_masks(
        &self,
        lane: &engine::fire::Lane,
        at: usize,
        start: u32,
        rows: usize,
        dense: Option<&(Vec<bool>, usize)>,
    ) -> EngineResult<()> {
        let refuse = |row: usize| {
            Err(Error::Invalid(format!(
                "lane {at} row {row} (position {}) attends a non-causal set of positions; the \
                 spatial engine runs causal attention over a lane's own history only (no \
                 sinks, windows or custom masks)",
                start as usize + row
            )))
        };
        if let Some(masking) = &lane.mask {
            for row in 0..rows {
                let mask = match masking {
                    engine::fire::Masking::Extent(mask) => mask,
                    engine::fire::Masking::Rows(masks) => match masks.get(row) {
                        Some(mask) => mask,
                        None => continue,
                    },
                };
                if !causal(mask, u64::from(start) + row as u64) {
                    return refuse(row);
                }
            }
        }
        if let Some((cells, stride)) = dense {
            for row in 0..rows {
                let position = start as usize + row;
                let line = &cells[row * stride..(row + 1) * stride];
                let seen = position + 1;
                if line.len() < seen
                    || line.iter().take(seen).any(|&admit| !admit)
                    || line.iter().skip(seen).any(|&admit| admit)
                {
                    return refuse(row);
                }
            }
        }
        Ok(())
    }

    /// One step: prologues, the lanes' tokens and positions (a device
    /// geometry instance states them on its ports), the forward, the guest
    /// passes over its logits, and the lanes' readouts.
    #[allow(clippy::too_many_lines)]
    fn fire_step(&mut self, step: &engine::fire::Step) -> EngineResult<Vec<LaneReadout>> {
        let fire_t0 = std::time::Instant::now();
        let _ = FireId(self.next_fire);
        self.next_fire = self.next_fire.wrapping_add(1);
        for attachment in &step.attachments {
            if !attachment.adapter_only && self.programs.prologue_runs(attachment.instance) {
                self.programs
                    .fire_interpreted(attachment.instance, &[], Stage::Prologue)
                    .map_err(refusal)?;
            }
        }
        let lanes = &step.lanes;
        if tracing::enabled!(target: "spatial_trace", tracing::Level::DEBUG) {
            for lane in lanes {
                tracing::debug!(target: "spatial_trace",
                    "spatial-trace lane: slot {} tokens {:?} positions {:?} kv {:?} kv_less {} group {:?} \
                     mask {} readout {:?} rs {:?}",
                    lane.slot,
                    lane.tokens,
                    lane.positions,
                    lane.kv,
                    lane.kv_less,
                    lane.group,
                    lane.mask.is_some(),
                    lane.readout,
                    lane.rs
                );
            }
        }
        // Each lane's tokens and positions: the fire's, or an instance's ports.
        let mut tokens: Vec<Vec<u32>> = lanes.iter().map(|l| l.tokens.clone()).collect();
        // Masks a device geometry instance states on its port, by lane.
        let mut dense_masks: BTreeMap<usize, (Vec<bool>, usize)> = BTreeMap::new();
        // The KV slots a device geometry instance places a lane's rows at.
        let mut port_slots: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        // Masks an instance's port states for a lane the host places.
        let mut host_masks: BTreeMap<usize, engine::fire::Masking> = BTreeMap::new();
        let mut positions: Vec<Vec<u32>> = lanes.iter().map(|l| l.positions.clone()).collect();
        for attachment in &step.attachments {
            if attachment.adapter_only {
                continue;
            }
            let instance = attachment.instance;
            let first = attachment.lane as usize;
            if self.programs.geometry_of(instance) == Some(GeometryClass::DeviceGeometry) {
                let Some(mut envelope) =
                    self.programs.device_envelope(instance).map_err(refusal)?
                else {
                    continue;
                };
                if tracing::enabled!(target: "spatial_trace", tracing::Level::DEBUG) {
                    tracing::debug!(target: "spatial_trace",
                        "spatial-trace envelope: slots {:?} positions {:?} kv_len {:?} fold {:?} \
                         qo {:?} mask {}",
                        envelope.slots,
                        envelope.positions,
                        envelope.kv_len,
                        envelope.fold,
                        envelope.qo_indptr,
                        envelope.mask.as_ref().map_or(0, Vec::len)
                    );
                }
                envelope.host_rows = lanes
                    .iter()
                    .skip(first)
                    .take(attachment.lanes.max(1) as usize)
                    .map(|lane| lane.tokens.len() as u32)
                    .collect();
                for at in 0..envelope.lanes() {
                    let source = first + at;
                    let Some(lane) = lanes.get(source) else { break };
                    let ports = envelope.lane(at, source).map_err(port_error)?;
                    let rows = if ports.owns_slots() {
                        ports.rows() as usize
                    } else {
                        lane.tokens.len()
                    };
                    let have = if ports.owns_slots() {
                        ports
                            .reach(rows)
                            .map_err(port_error)?
                            .map(|after| after.saturating_sub(rows as u32))
                    } else {
                        (!lane.kv.pages.is_empty()).then_some(lane.kv.held)
                    };
                    tokens[source] = ports.tokens_for(rows).map_err(port_error)?.to_vec();
                    if let Some(slots) = ports.slots(rows).map_err(port_error)? {
                        port_slots.insert(source, slots.to_vec());
                    }
                    if let Some((cells, stride)) = ports.mask(rows).map_err(port_error)? {
                        if tracing::enabled!(target: "spatial_trace", tracing::Level::DEBUG) {
                            let seen: Vec<usize> = cells[..stride]
                                .iter()
                                .enumerate()
                                .filter(|(_, c)| **c)
                                .map(|(i, _)| i)
                                .collect();
                            tracing::debug!(target: "spatial_trace",
                                "spatial-trace mask: lane {source} rows {rows} stride {stride} row0 \
                                 sees {seen:?} slots {:?}",
                                ports.slots(rows).ok().flatten()
                            );
                        }
                        dense_masks.insert(source, (cells.to_vec(), stride));
                    }
                    if let Some(have) = have {
                        if let Some(p) = ports.positions_for(have, rows).map_err(port_error)? {
                            positions[source] = p.to_vec();
                        } else if positions[source].len() != rows {
                            positions[source] = (have..have + rows as u32).collect();
                        }
                    }
                }
            } else {
                let count = self.programs.lane_count(instance).map_err(refusal)?;
                for at in 0..count {
                    let source = first + at;
                    let Some(lane) = lanes.get(source) else { break };
                    if let Some(crate::program::Tokens::Host(ids)) = self
                        .programs
                        .envelope_tokens(instance, lane.tokens.len(), at)
                        .map_err(refusal)?
                    {
                        tokens[source] = ids;
                    }
                    // The mask its `attn_mask` port states, when the fire
                    // carries none.
                    if lane.mask.is_none()
                        && let Some(mask) = self
                            .programs
                            .envelope_mask(instance, lane.tokens.len(), at)
                            .map_err(refusal)?
                    {
                        host_masks.insert(source, mask);
                    }
                }
            }
        }
        // The forward, lane by lane, keeping the logits of the rows something reads.
        let attached: BTreeMap<usize, InstanceId> = step
            .attachments
            .iter()
            .filter(|a| !a.adapter_only)
            .flat_map(|a| {
                (0..a.lanes.max(1)).map(move |k| (a.lane as usize + k as usize, a.instance))
            })
            .collect();
        let (name, width) = self.readout.clone().ok_or_else(|| {
            Error::Load("the model reads no float output at its readout rows".into())
        })?;
        let mut logits: Vec<Vec<Vec<f32>>> = Vec::with_capacity(lanes.len());
        let mut read_rows: Vec<Vec<u32>> = Vec::with_capacity(lanes.len());
        // Lanes a token loop answered (their row a one-hot of the argmax).
        let mut fast: Vec<bool> = Vec::with_capacity(lanes.len());
        // A paged model runs the fire's lanes together, each through its own
        // pages: (lane index, tokens, start, pages, logits asked, rows read).
        let paged = self.daemon.as_ref().is_some_and(Model::paged);
        let mut paged_lanes: Vec<(usize, PagedLane, Vec<u32>)> = Vec::new();
        // Lanes whose epilogue the wafer head's VM ran: (instance, the
        // step's `out` words, the VM's pool after it).
        let mut vm_out: std::collections::HashMap<usize, (InstanceId, Vec<u32>, Vec<u32>)> =
            std::collections::HashMap::new();
        // Cells the fire's paged lanes so far will have written.
        let mut planned: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        for (at, lane) in lanes.iter().enumerate() {
            let n = tokens[at].len() as u32;
            let rows: Vec<u32> = match &lane.readout {
                Readout::Rows(rows) => rows.clone(),
                Readout::Last if n > 0 => vec![n - 1],
                Readout::None if attached.contains_key(&at) && n > 0 => vec![n - 1],
                _ => Vec::new(),
            };
            let (out, ran_ahead) = if tokens[at].is_empty() {
                (Vec::new(), false)
            } else {
                // The rows' positions: stated (consecutive), else after what
                // the lane holds.
                let start = match positions[at].first() {
                    Some(&p) => p,
                    None if !lane.kv.pages.is_empty() => lane.kv.held,
                    None => self.kv.slots.get(&lane.slot).map_or(0, |h| h.len() as u32),
                };
                if positions[at]
                    .iter()
                    .enumerate()
                    .any(|(i, &p)| p != start + i as u32)
                {
                    return Err(Error::Invalid(format!(
                        "the spatial engine runs a lane's rows at consecutive positions; lane \
                         {at} has {:?}",
                        positions[at]
                    )));
                }
                // Greedy: the lane's one reader takes only the argmax of the
                // logits.
                let greedy = attached
                    .get(&at)
                    .is_some_and(|&instance| self.programs.greedy_reads(instance, &name))
                    && step.attachments.iter().filter(|a| !a.adapter_only).count() == 1;
                let ahead = lane.ahead.as_ref().and_then(engine::ahead::Ahead::fires_left);
                let first_slot = port_slots
                    .get(&at)
                    .and_then(|s| s.first().copied())
                    .unwrap_or(if positions[at].is_empty() { start } else { lane.kv.held });
                if first_slot == 0 {
                    self.kv.tainted.remove(&lane.slot);
                }
                // The model's pages must hold what the lane's rows will read;
                // a lane continuing on pages the model never got (a fire
                // that named another lane's pages) is re-prefilled into them
                // from its own tokens first.
                let mut on_pages = paged
                    && !lane.kv.pages.is_empty()
                    && !self.kv.tainted.contains(&lane.slot)
                    && self.kv.known(&lane.kv.pages, first_slot, &planned);
                if paged && !on_pages && !lane.kv.pages.is_empty() && first_slot > 0 {
                    let history = self
                        .kv
                        .slots
                        .get(&lane.slot)
                        .filter(|h| h.len() >= first_slot as usize)
                        .map(|h| h[..first_slot as usize].to_vec());
                    if let Some(history) = history {
                        let slots: Vec<u32> = (0..first_slot).collect();
                        let refill = PagedLane {
                            tokens: history.clone(),
                            slots: slots.clone(),
                            positions: slots.clone(),
                            pages: lane.kv.pages.clone(),
                            visible: None,
                            which: wafer::Logits::None,
                            argmax_only: false,
                            vm: None,
                        };
                        self.daemon()?.fire_paged(&[&refill])?;
                        self.kv.write_paged(lane.slot, &lane.kv.pages, &slots, &history);
                        self.kv.tainted.remove(&lane.slot);
                        self.refilled += u64::from(first_slot);
                        tracing::info!(
                            slot = lane.slot,
                            rows = first_slot,
                            total = self.refilled,
                            "spatial: lane re-prefilled into its own pages"
                        );
                        on_pages = self.kv.known(&lane.kv.pages, first_slot, &planned);
                    }
                }
                if paged && !on_pages {
                    if self.kv.tainted.insert(lane.slot) {
                        tracing::info!(
                            slot = lane.slot,
                            first_slot,
                            "spatial: the model's pages do not hold this lane's cache; it \
                             replays from its own tokens until it starts over"
                        );
                    }
                }
                if on_pages {
                    let last_only = rows.len() == 1 && rows[0] as usize + 1 == tokens[at].len();
                    let which = if rows.is_empty() {
                        wafer::Logits::None
                    } else if last_only {
                        wafer::Logits::Last
                    } else {
                        wafer::Logits::All
                    };
                    let n = tokens[at].len();
                    // Where the rows' K/V go (the instance's slots, else after
                    // what the lane holds), how they rotate (their positions,
                    // else their slots), and what each sees (its mask, else
                    // every slot up to its own).
                    let slots: Vec<u32> = port_slots.get(&at).cloned().unwrap_or_else(|| {
                        let held = if positions[at].is_empty() { start } else { lane.kv.held };
                        (held..held + n as u32).collect()
                    });
                    let rope = if positions[at].is_empty() {
                        slots.clone()
                    } else {
                        positions[at].clone()
                    };
                    let visible = visible_runs(
                        lane.mask.as_ref().or(host_masks.get(&at)),
                        n,
                        dense_masks.get(&at),
                    );
                    let size = self.kv.page_size.max(1);
                    for &q in &slots {
                        if let Some(&page) = lane.kv.pages.get((q / size) as usize) {
                            planned.insert((page, q % size));
                        }
                    }
                    // The multi-lane VM samples a decode row (one token, its
                    // one reader taking the last row) of a program it holds.
                    let vm = attached.get(&at).copied().filter(|&instance| {
                        n == 1
                            && last_only
                            && visible.is_none()
                            && step.attachments.iter().filter(|a| !a.adapter_only).count() == 1
                            && self.vm_lanes_take(instance)
                    });
                    paged_lanes.push((
                        at,
                        PagedLane {
                            tokens: tokens[at].clone(),
                            slots,
                            positions: rope,
                            pages: lane.kv.pages.clone(),
                            visible,
                            which,
                            argmax_only: greedy && last_only && vm.is_none(),
                            vm,
                        },
                        rows.clone(),
                    ));
                    (Vec::new(), false)
                } else {
                    // A model of one sequence a cache runs causal attention
                    // only (a paged model attends what each row's mask says).
                    self.check_masks(lane, at, start, tokens[at].len(), dense_masks.get(&at))?;
                    let vm = attached.get(&at).copied().filter(|&instance| {
                        rows.len() == 1
                            && rows[0] as usize + 1 == tokens[at].len()
                            && step.attachments.iter().filter(|a| !a.adapter_only).count() == 1
                            && self.vm_takes(instance)
                    });
                    match vm {
                        Some(instance) => {
                            let (row, pool) =
                                self.forward_vm(instance, lane.slot, &lane.kv.pages, &tokens[at], start)?;
                            vm_out.insert(at, (instance, row, pool));
                            (Vec::new(), true)
                        }
                        None => self.forward(
                            lane.slot,
                            &lane.kv.pages,
                            &tokens[at],
                            start,
                            &rows,
                            greedy,
                            ahead,
                        )?,
                    }
                }
            };
            logits.push(out);
            fast.push(ran_ahead);
            read_rows.push(rows);
        }
        // The multi-lane VM's lanes: one launch, each in its bank's lane.
        let (vm_lanes, mut paged_lanes): (Vec<_>, Vec<_>) =
            paged_lanes.into_iter().partition(|(_, lane, _)| lane.vm.is_some());
        if !vm_lanes.is_empty() {
            let request: Vec<(InstanceId, &PagedLane)> = vm_lanes
                .iter()
                .filter_map(|(_, lane, _)| lane.vm.map(|instance| (instance, lane)))
                .collect();
            let answers = self.fire_vm_lanes(&request)?;
            for ((at, mut lane, rows), answer) in vm_lanes.into_iter().zip(answers) {
                let Some((instance, row, pool)) = answer else {
                    // No VM lane was free: its logits come back as any lane's.
                    lane.vm = None;
                    paged_lanes.push((at, lane, rows));
                    continue;
                };
                self.kv
                    .write_paged(lanes[at].slot, &lane.pages, &lane.slots, &lane.tokens);
                logits[at] = Vec::new();
                fast[at] = true;
                vm_out.insert(at, (instance, row, pool));
            }
        }
        if !paged_lanes.is_empty() {
            let request: Vec<&PagedLane> = paged_lanes.iter().map(|(_, lane, _)| lane).collect();
            let answers = self.daemon()?.fire_paged(&request)?;
            self.paged_fires += 1;
            for ((at, lane, rows), (lane_logits, token)) in paged_lanes.into_iter().zip(answers) {
                self.kv
                    .write_paged(lanes[at].slot, &lane.pages, &lane.slots, &lane.tokens);
                if lane.argmax_only {
                    // Its row a one-hot of the argmax, as a token loop's.
                    let mut row = vec![0f32; width as usize];
                    if let Some(cell) = token.and_then(|t| row.get_mut(t as usize)) {
                        *cell = 1.0;
                    }
                    logits[at] = vec![row];
                    fast[at] = true;
                    continue;
                }
                logits[at] = match lane.which {
                    wafer::Logits::None => Vec::new(),
                    wafer::Logits::Last => lane_logits,
                    wafer::Logits::All => rows
                        .iter()
                        .map(|&r| lane_logits.get(r as usize).cloned().unwrap_or_default())
                        .collect(),
                };
            }
        }
        // Guest passes in the interpreter, each output it reads from this fire.
        let mut guest_read = vec![false; lanes.len()];
        for attachment in &step.attachments {
            if attachment.adapter_only {
                continue;
            }
            let at = attachment.lane as usize;
            if let Some((instance, row, pool)) = vm_out.remove(&at)
                && instance == attachment.instance
            {
                self.fire_vm(instance, &row, &pool)?;
                if let Some(read) = guest_read.get_mut(at) {
                    *read = true;
                }
                continue;
            }
            let mut outputs: Vec<(String, u32, Value)> = Vec::new();
            for (out_name, k) in self.programs.outputs_of(attachment.instance) {
                let lane = attachment.lane as usize + k as usize;
                if out_name != name && out_name.starts_with("attn.") {
                    return Err(Error::Invalid(format!(
                        "the program reads model output `{out_name}`; the spatial engine's \
                         model answers `{name}` (the logits) alone — attention scores are not \
                         served yet"
                    )));
                }
                if out_name != name {
                    continue;
                }
                // A pass over several lanes reads its rows as one output: the
                // lane's, then the pass's later lanes'.
                let end = attachment.lane as usize + attachment.lanes.max(1) as usize;
                if logits.get(lane).is_some() {
                    let flat: Vec<f32> = if fast.get(lane) == Some(&true)
                        && end <= lane + 1
                        && logits[lane].len() == 1
                    {
                        // Its one row, handed over: nothing reads it again.
                        std::mem::take(&mut logits[lane][0])
                    } else {
                        (lane..end.max(lane + 1))
                            .filter_map(|l| logits.get(l))
                            .flat_map(|rows| rows.iter().flatten().copied())
                            .collect()
                    };
                    outputs.push((out_name.clone(), k, Value::F32(flat)));
                    if let Some(read) = guest_read.get_mut(lane) {
                        *read = true;
                    }
                }
            }
            if !self.fire_guest(attachment.instance, &outputs)? {
                self.programs
                    .fire_interpreted(attachment.instance, &outputs, Stage::Epilogue)
                    .map_err(refusal)?;
            }
        }
        self.fires += 1;
        tracing::info!(
            fires = self.fires,
            lanes = step.lanes.len(),
            rows = step.lanes.iter().map(|l| l.tokens.len()).sum::<usize>(),
            fire_ms = fire_t0.elapsed().as_secs_f64() * 1e3,
            model_ms = MODEL_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6,
            json_ms = JSON_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6,
            "spatial: model time"
        );
        Ok(lanes
            .iter()
            .enumerate()
            .map(|(at, lane)| {
                // A token loop's row is a stand-in for its argmax: the lane's
                // answer is in its program's channels, as a device guest's.
                if matches!(lane.readout, Readout::None) || logits[at].is_empty() || fast[at] {
                    return LaneReadout::default();
                }
                let values: Vec<f32> = logits[at].iter().flatten().copied().collect();
                LaneReadout {
                    output: name.clone(),
                    rows: u32::try_from(read_rows[at].len()).unwrap_or(u32::MAX),
                    width,
                    values,
                    ..LaneReadout::default()
                }
            })
            .collect())
    }
}

impl Engine for Spatial {
    fn kind(&self) -> &'static str {
        "cerebras-spatial"
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.caps.as_ref().map(|caps| &caps.device)
    }

    fn load(&mut self, request: LoadRequest) -> EngineResult<Loaded> {
        if self.shell.is_some() {
            return Err(Error::Load(
                "this spatial engine already has a model loaded".into(),
            ));
        }
        let LoadRequest {
            trace,
            checkpoint,
            budgets,
            residency,
            ordinal: _,
            frames_in_flight: _,
        } = request;
        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(
                "the spatial engine loads a checkpoint path".into(),
            ));
        };
        // The dry shell reads neither contract nor checkpoint; it wants one.
        let contract = (self.contract_for)(&trace, &path).map_err(Error::Load)?;
        let patches = patch_ladder(&trace, &budgets);
        let voxels = poem_compiler::VoxelLadder::of(
            &trace,
            budgets.max_voxels,
            budgets.max_clips,
            budgets.max_lanes,
        );
        let slots =
            poem_compiler::state_slots(budgets.slots, &trace, crate::api::DEFAULT_STATE_SLOTS);
        // The model as the spatial kernels see it: refused here unless its
        // structure is one they implement (spatial/plan.rs).
        let spec = plan::ModelSpec::read(&trace).map_err(|e| Error::Load(e.to_string()))?;
        // The weights land by the trace's params (the shell takes the trace).
        let weights_trace = trace.clone();
        self.params = {
            let (h, hd) = (spec.hidden as f64, spec.head_dim as f64);
            let attn = h * hd * (2 * spec.q_heads + 2 * spec.kv_heads) as f64;
            let heads = if spec.tied { 1.0 } else { 2.0 };
            spec.layers as f64 * (attn + 3.0 * h * spec.inter as f64) + heads * spec.vocab as f64 * h
        };
        let shell = Shell::load_dry(
            Boot {
                patches,
                trace,
                contract: &contract,
                checkpoint: &path,
                budget: bake_budgets(&budgets),
                page_size: budgets.page_size,
                context: budgets.max_context,
                slots,
                kv_budget: residency.device_kv_budget,
                memory_share: residency.device_memory_share,
                device: &self.boot,
            },
            voxels,
            false,
        )
        .map_err(fault)?;
        self.readout = shell.readout_output();
        // The model: the daemon, or the wafer program in this process.
        let ckpt = self
            .spatial
            .ckpt
            .clone()
            .unwrap_or_else(|| Path::new(&path).display().to_string());
        let program = match (&self.spatial.program, &self.spatial.params) {
            (Some(dir), _) => Some(dir.clone()),
            (None, Some(params)) => Some(
                wafer::compile(params, self.boot.target, self.boot.cmaddr.is_some())
                    .map_err(|e| Error::Load(format!("the spatial wafer program: {e}")))?
                    .display()
                    .to_string(),
            ),
            (None, None) => match &self.spatial.serve {
                // PIE builds the program from the model's plan.
                Some(t) => {
                    let cfg = build::ServeConfig::from_toml(t, budgets.page_size as usize)
                        .map_err(|e| Error::Load(e.to_string()))?;
                    Some(
                        build::build_for(&spec, &cfg, self.boot.target, self.boot.cmaddr.is_some())
                            .map_err(|e| Error::Load(format!("building the spatial program: {e}")))?
                            .display()
                            .to_string(),
                    )
                }
                None => None,
            },
        };
        // A compiled program must be the model's: dims, layers, vocabulary,
        // and what the placement assumes (tied head, eps, q/k norms, theta).
        if let Some(program) = &program {
            let path = Path::new(program).join("out.json");
            let json: serde_json::Value = std::fs::read_to_string(&path)
                .ok()
                .and_then(|text| serde_json::from_str(&text).ok())
                .ok_or_else(|| Error::Load(format!("{}: no readable out.json", path.display())))?;
            spec.check_program(plan::out_json_params(&json))
                .map_err(|e| Error::Load(e.to_string()))?;
        }
        let paged_model;
        let model = if let Some(program) = &program {
            let platform = match &self.boot.cmaddr {
                Some(cmaddr) => crate::sdk::Platform::System {
                    cmaddr: cmaddr.clone(),
                },
                None => crate::sdk::Platform::Simulator(crate::sdk::Simulator {
                    num_threads: self.boot.num_threads,
                    ..crate::sdk::Simulator::new(self.boot.target)
                }),
            };
            let launch = wafer::Launch::named(self.spatial.launch.as_deref());
            let theta = Some(f64::from(spec.rope_theta));
            // The weights through PIE's checkpoint path (the load's contract
            // over its checkpoint); `ckpt` names a Hugging Face directory
            // instead (checks against the Python hosts).
            let weights = match &self.spatial.ckpt {
                Some(dir) => ckpt::Ckpt::open(Path::new(dir)),
                None => ckpt::Ckpt::landed(&weights_trace, &contract, Path::new(&path)),
            }
            .map_err(|e| Error::Load(format!("the spatial wafer model's weights: {e}")))?;
            let w = wafer::Wafer::open(program.into(), weights, platform, launch, theta)
                .map_err(|e| Error::Load(format!("the spatial wafer model: {e}")))?;
            tracing::info!(program, ckpt, vocab = w.vocab, hidden = w.hidden, "spatial wafer model loaded");
            // A batched build owns the KV as PIE's pages, as a paged daemon.
            paged_model = w.paged.is_some();
            Model::Wafer(w)
        } else {
            // The daemon is a test reference (the Python model), not a
            // serving path: only a build with `spatial-daemon` runs one.
            if !cfg!(feature = "spatial-daemon") {
                return Err(Error::Load(
                    "[cerebras.spatial] names no wafer program (`program`, `params` or `serve`); a model \
                     `daemon` is a test reference, run only by a build with the `spatial-daemon` feature"
                        .into(),
                ));
            }
            let cmd = self.spatial.daemon.clone().unwrap_or_default();
            let mut daemon = Daemon::spawn(&cmd)?;
            let mut load = serde_json::json!({
                "op": "load", "ckpt": ckpt,
                "paged": {"page_size": shell.paging().page_size},
            });
            if let Some(layers) = self.spatial.layers {
                load["layers"] = serde_json::json!(layers);
            }
            let answer = daemon.call(&load)?;
            daemon.caches = answer
                .get("caches")
                .and_then(serde_json::Value::as_u64)
                .map_or(1, |n| n.max(1) as usize);
            daemon.paged = answer.get("paged").and_then(serde_json::Value::as_bool) == Some(true);
            paged_model = daemon.paged;
            tracing::info!(?answer, "spatial model daemon loaded");
            Model::Daemon(daemon)
        };
        self.daemon = Some(model);
        self.kv = Kv {
            paged: paged_model,
            ..Kv::default()
        };

        let trace_name = shell.trace().name.clone();
        let paging = shell.paging();
        self.kv.page_size = paging.page_size;
        // A batched wafer build's pool is PIE's: the same page size, and at
        // least the pages PIE hands out.
        if let Some(Model::Wafer(w)) = &self.daemon
            && let Some(shape) = w.paged
        {
            if paging.page_size as usize != shape.page_size || paging.pages() > shape.pages() as u64 {
                return Err(Error::Load(format!(
                    "the wafer program's KV pool is {} pages of {}; this load pages {} of {} \
                     (set the page size and the KV budget to fit it)",
                    shape.pages(),
                    shape.page_size,
                    paging.pages(),
                    paging.page_size
                )));
            }
        }
        // The host tier: `host_kv_budget` of whole pages, on a wafer build
        // whose pages the host can read and write.
        self.kv.host_pages = match &self.daemon {
            Some(Model::Wafer(w)) => match (w.paged, w.page_words, residency.host_kv_budget) {
                (Some(_), Some(words), Some(budget)) if words > 0 => {
                    u32::try_from(budget / (words as u64 * 4)).unwrap_or(u32::MAX)
                }
                _ => 0,
            },
            _ => 0,
        };
        if self.kv.host_pages > 0 {
            tracing::info!(pages = self.kv.host_pages, "spatial: the host keeps suspended lanes' kv pages");
        }
        let state_rows = shell.has_state();
        let wafer_pool = match &self.daemon {
            Some(Model::Wafer(w)) => w.paged,
            _ => None,
        };
        // The context the model holds: the wafer program's positions (a
        // lane's pages on a batched build); a daemon's is its own.
        let model_context = match &self.daemon {
            Some(Model::Wafer(w)) => u32::try_from(w.positions).unwrap_or(u32::MAX),
            _ => u32::MAX,
        };
        let caps = Capabilities {
            device: DeviceFacts {
                backend: "cerebras".to_string(),
                domain: MemoryDomain::CerebrasDevice(self.boot.ordinal),
                sms: 1,
                unified_memory: false,
                fp8_native: false,
                native_mxfp4_moe: false,
                storage_alignment: 256,
                storage_max_tile_bytes: u64::MAX,
                codegen_backend: None,
            },
            pools: PoolFacts {
                // A batched wafer build reads at most E of a lane's pages on a
                // row: pages below rows x E fill every row E times over, so no
                // lane can name more.
                kv_pages: u32::try_from(paging.pages())
                    .unwrap_or(u32::MAX)
                    .min(wafer_pool.map_or(u32::MAX, |s| (s.rows * s.entries) as u32)),
                kv_page_size: paging.page_size,
                state_slots: if state_rows { paging.slots } else { 0 },
                buffer_pages: shell.buffer_pages(),
                state_slot_bytes: shell.state_slot_bytes(),
                adapter_banks: 0,
                host_kv_pages: self.kv.host_pages,
                ..PoolFacts::default()
            },
            limits: FireLimits {
                // The model holds one sequence's cache; a fire's lanes run one
                // after another, each cache brought back from its tokens.
                max_lanes: budgets.max_lanes.max(1),
                max_tokens: budgets.max_tokens,
                max_page_refs: paging
                    .pages_per_slot
                    .min(wafer_pool.map_or(u32::MAX, |s| s.max_page_refs() as u32)),
                max_context: paging.context().min(model_context),
                prefill_rows: 0,
                max_readouts: budgets.max_tokens,
            },
            has_lora: false,
            has_attn_page_mask: false,
            draft_block: 0,
            draft_mask_token: 0,
            draft_proposals_from: 1,
            // A paged model attends what each row's mask says; a model of
            // one sequence a cache runs causal attention only, so it takes
            // no mask port.
            ports: if paged_model {
                PortMask::DEVICE_GEOMETRY.with(eta_ir::registry::Port::AttnMask)
            } else {
                PortMask::DEVICE_GEOMETRY
            },
            geometry: GeometryClass::DeviceGeometry,
            kv_copy: KvCopyDomains {
                // Pages are tokens here: a copy copies them. A wafer build
                // whose pages the host can read parks a suspended lane's
                // pages on the host (never attended there).
                device_to_device: true,
                device_to_host: self.kv.host_pages > 0,
                host_to_device: self.kv.host_pages > 0,
                host_to_host: false,
            },
            kv_handle: None,
            media_encode: false,
            device_channel_commit: true,
            rs_verbs: shell.serves_rs_verbs(),
            rs_snapshots: 0,
            bidirectional_attention: false,
            adapter_attachments: false,
            facts: shell.trace().facts.clone(),
            inputs: shell.input_decls(),
            classes: shell.classes(),
        };
        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                trace_name,
                weight_bytes: 0,
                weights_resident: true,
                weights_from_cache: false,
                arena_bytes: 0,
                pool_bytes: 0,
                input_bytes: 0,
                pool_committed_bytes: 0,
                pool_high_water_bytes: 0,
            },
            caps,
        })
    }

    fn copy_kv(&mut self, copy: &engine::transfer::KvCopy) -> EngineResult<()> {
        copy.validate()?;
        if tracing::enabled!(target: "spatial_trace", tracing::Level::DEBUG) {
            tracing::debug!(target: "spatial_trace", "spatial-trace copy_kv: {copy:?}");
        }
        let device = self.caps.as_ref().map(|caps| caps.device.domain);
        let host = engine::transfer::MemoryDomain::HostPinned;
        if self.kv.host_pages > 0 && Some(copy.src) == device && copy.dst == host {
            return self.page_out(copy);
        }
        if self.kv.host_pages > 0 && copy.src == host && Some(copy.dst) == device {
            return self.page_in(copy);
        }
        if Some(copy.src) != device || Some(copy.dst) != device {
            return Err(Error::Unsupported {
                verb: "`copy_kv` between domains other than this load's device",
                engine: "cerebras-spatial",
            });
        }
        let size = self.kv.page_size.max(1);
        if self.daemon.as_ref().is_some_and(Model::paged) {
            let runs = copy.runs(size)?;
            self.daemon()?.copy_paged(&runs)?;
        }
        for run in copy.runs(size)? {
            let src: Vec<Option<u32>> = self
                .kv
                .pages
                .get(&run.src_page)
                .cloned()
                .unwrap_or_else(|| vec![None; size as usize]);
            let dst = self
                .kv
                .pages
                .entry(run.dst_page)
                .or_insert_with(|| vec![None; size as usize]);
            self.kv.copied.insert(run.dst_page);
            for k in 0..run.tokens as usize {
                if let (Some(&cell), Some(slot)) = (
                    src.get(run.src_token as usize + k),
                    dst.get_mut(run.dst_token as usize + k),
                ) {
                    *slot = cell;
                }
            }
        }
        Ok(())
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        // The guest program as registered, for studying what the wafer's
        // guest region would have to run (`examples/eta_coverage.rs`).
        if let Some(dir) = std::env::var_os("PIE_SPATIAL_DUMP_PTIR") {
            let path = std::path::Path::new(&dir)
                .join(format!("{:016x}.ptir", registration.program_hash));
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(path, &registration.reference_ptir);
        }
        self.programs.register(registration).map_err(refusal)
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> EngineResult<RegisteredChannel> {
        self.programs
            .register_channel(registration)
            .map_err(refusal)
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
        let caps = self
            .caps
            .as_ref()
            .ok_or_else(|| Error::Program("bind_instance before load".to_string()))?;
        if !caps.admits(binding.geometry) {
            return Err(Error::Program(format!(
                "this load binds at most {:?} and not {:?}",
                caps.geometry, binding.geometry
            )));
        }
        let bound = self.programs.bind(binding).map_err(refusal)?;
        self.probe_guest(bound.id, bound.program);
        Ok(bound)
    }

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<()> {
        self.guest_vms.remove(&id);
        self.programs.close_instance(id).map_err(refusal)
    }

    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        self.programs.close_channel(id).map_err(refusal)
    }

    fn publish_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> EngineResult<bool> {
        self.programs
            .publish(instance, channel, cell)
            .map_err(refusal)
    }

    fn take_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
    ) -> EngineResult<Option<Vec<u8>>> {
        self.programs.take(instance, channel).map_err(refusal)
    }

    fn on_complete(&mut self, sink: engine::CompletionSink) {
        self.sink = Some(sink);
    }

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        frame.validate_for(
            self.caps
                .as_ref()
                .map_or(engine::fire::Serves::NONE, engine::fire::Serves::of),
        )?;
        let id = FrameId(self.next_frame);
        self.next_frame = self.next_frame.wrapping_add(1);
        if tracing::enabled!(target: "spatial_trace", tracing::Level::DEBUG) {
            tracing::debug!(target: "spatial_trace",
                "spatial-trace frame: {} step(s), attachments {:?}",
                frame.steps.len(),
                frame.steps.first().map(|s| s.attachments.len())
            );
        }
        let mut steps = Vec::with_capacity(frame.steps.len());
        let mut readouts = Vec::with_capacity(frame.steps.len());
        for (index, step) in frame.steps.iter().enumerate() {
            let read = self.fire_step(step)?;
            if let Some(sink) = &self.sink {
                sink(
                    engine::StepDone {
                        frame: id,
                        step: index as u32,
                    },
                    engine::StepOutcome::Committed,
                );
            }
            steps.push(FireTicket {
                id: FireId(self.next_fire),
                readouts: read.clone(),
            });
            readouts.push(read);
        }
        self.pending = Some((id, readouts));
        Ok(FrameTicket { id, steps })
    }

    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> EngineResult<()> {
        if let Some((id, readouts)) = &self.pending
            && *id == ticket.id
        {
            for (receipt, read) in ticket.steps.iter_mut().zip(readouts) {
                receipt.readouts = read.clone();
            }
        }
        Ok(())
    }
}
