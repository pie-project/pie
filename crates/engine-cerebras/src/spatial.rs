//! The spatial engine: the model forward runs in a model daemon — the
//! resident wafer program on a CS-3, or a reference that answers the same
//! protocol — and everything else stays as the cerebras engine has it: guest
//! (eta) programs in the host interpreter, channels, the runtime's frames.
//!
//! The daemon is a child process speaking JSON lines on stdin/stdout
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
//! platform (`cmaddr`, else the simulator), its weights placed from `ckpt`
//! (`model.safetensors` or `weights.npz`, with `config.json`):
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
pub mod wafer;

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
    /// Steps a greedy lane's token loop runs past its row (0: off).
    pub lookahead: Option<usize>,
    pub ckpt: Option<String>,
    pub layers: Option<u32>,
}

impl SpatialBoot {
    pub fn of(doc: &toml::Table) -> Option<SpatialBoot> {
        let t = doc
            .get("cerebras")
            .and_then(toml::Value::as_table)?
            .get("spatial")
            .and_then(toml::Value::as_table)?;
        let text = |k: &str| t.get(k).and_then(toml::Value::as_str).map(str::to_string);
        let (daemon, program, params) = (text("daemon"), text("program"), text("params"));
        if daemon.is_none() && program.is_none() && params.is_none() {
            return None;
        }
        Some(SpatialBoot {
            daemon,
            program,
            params,
            launch: text("launch"),
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
        })
    }
}

/// The daemon child and its pipes.
struct Daemon {
    /// Sequences' caches it holds (its load answer's `caches`, else 1).
    caches: usize,
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
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    /// One request, its answer; an answer without `"ok": true` is an error.
    fn call(&mut self, request: &serde_json::Value) -> EngineResult<serde_json::Value> {
        let line = serde_json::to_string(request).expect("a JSON value serializes");
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
        let value: serde_json::Value = serde_json::from_str(answer.trim()).map_err(|e| {
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

/// Where the forward runs: the daemon, or the wafer program in process.
enum Model {
    Daemon(Daemon),
    Wafer(wafer::Wafer),
}

impl Model {
    /// Sequences' caches the model holds at once.
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
    /// Each of the model's caches: the tokens it holds, position by
    /// position, the argmax after each where a token loop told it, and when
    /// it last ran (to give the least recent away).
    cached: Vec<Cached>,
    clock: u64,
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
        if pages.is_empty() {
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
}

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
    ) -> EngineResult<Vec<Vec<f32>>> {
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
                return Ok(one_hot(token));
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
            self.daemon()?
                .forward(cache, &history[agree..], agree as u32, wafer::Logits::None)?;
            self.kv.cached[cache].extend(&history[agree..]);
        }
        if greedy {
            let (ran, argmax) = self.daemon()?.greedy(cache, tokens, start, lookahead)?;
            self.ahead_runs += 1;
            if std::env::var_os("PIE_SPATIAL_TRACE").is_some() {
                eprintln!(
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
            return Ok(one_hot(token));
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
        Ok(match which {
            wafer::Logits::None => Vec::new(),
            wafer::Logits::Last => logits,
            wafer::Logits::All => rows
                .iter()
                .map(|&r| logits.get(r as usize).cloned().unwrap_or_default())
                .collect(),
        })
    }

    /// Steps a greedy lane's token loop runs past its row (0: none):
    /// `PIE_SPATIAL_LOOKAHEAD`, else the config's `lookahead`, else 16.
    fn lookahead(&self) -> usize {
        std::env::var("PIE_SPATIAL_LOOKAHEAD")
            .ok()
            .and_then(|v| v.parse().ok())
            .or(self.spatial.lookahead)
            .unwrap_or(16)
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
        if std::env::var_os("PIE_SPATIAL_TRACE").is_some() {
            for lane in lanes {
                eprintln!(
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
                if std::env::var_os("PIE_SPATIAL_TRACE").is_some() {
                    eprintln!(
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
                    if let Some((cells, stride)) = ports.mask(rows).map_err(port_error)? {
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
        for (at, lane) in lanes.iter().enumerate() {
            let n = tokens[at].len() as u32;
            let rows: Vec<u32> = match &lane.readout {
                Readout::Rows(rows) => rows.clone(),
                Readout::Last if n > 0 => vec![n - 1],
                Readout::None if attached.contains_key(&at) && n > 0 => vec![n - 1],
                _ => Vec::new(),
            };
            let out = if tokens[at].is_empty() {
                Vec::new()
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
                self.check_masks(lane, at, start, tokens[at].len(), dense_masks.get(&at))?;
                // Greedy: the lane's one reader takes only the argmax of the
                // logits.
                let greedy = attached
                    .get(&at)
                    .is_some_and(|&instance| self.programs.greedy_reads(instance, &name))
                    && step.attachments.iter().filter(|a| !a.adapter_only).count() == 1;
                self.forward(lane.slot, &lane.kv.pages, &tokens[at], start, &rows, greedy)?
            };
            logits.push(out);
            read_rows.push(rows);
        }
        // Guest passes in the interpreter, each output it reads from this fire.
        let mut guest_read = vec![false; lanes.len()];
        for attachment in &step.attachments {
            if attachment.adapter_only {
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
                    let flat: Vec<f32> = (lane..end.max(lane + 1))
                        .filter_map(|l| logits.get(l))
                        .flat_map(|rows| rows.iter().flatten().copied())
                        .collect();
                    outputs.push((out_name.clone(), k, Value::F32(flat)));
                    if let Some(read) = guest_read.get_mut(lane) {
                        *read = true;
                    }
                }
            }
            self.programs
                .fire_interpreted(attachment.instance, &outputs, Stage::Epilogue)
                .map_err(refusal)?;
        }
        Ok(lanes
            .iter()
            .enumerate()
            .map(|(at, lane)| {
                if matches!(lane.readout, Readout::None) || logits[at].is_empty() {
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
            (None, None) => None,
        };
        let model = if let Some(program) = &program {
            let platform = match &self.boot.cmaddr {
                Some(cmaddr) => crate::sdk::Platform::System {
                    cmaddr: cmaddr.clone(),
                },
                None => crate::sdk::Platform::Simulator(crate::sdk::Simulator::new(
                    self.boot.target,
                )),
            };
            let launch = wafer::Launch::named(self.spatial.launch.as_deref());
            let w = wafer::Wafer::open(program.into(), ckpt.clone().into(), platform, launch)
                .map_err(|e| Error::Load(format!("the spatial wafer model: {e}")))?;
            tracing::info!(program, ckpt, vocab = w.vocab, hidden = w.hidden, "spatial wafer model loaded");
            Model::Wafer(w)
        } else {
            let cmd = self.spatial.daemon.clone().unwrap_or_default();
            let mut daemon = Daemon::spawn(&cmd)?;
            let mut load = serde_json::json!({"op": "load", "ckpt": ckpt});
            if let Some(layers) = self.spatial.layers {
                load["layers"] = serde_json::json!(layers);
            }
            let answer = daemon.call(&load)?;
            daemon.caches = answer
                .get("caches")
                .and_then(serde_json::Value::as_u64)
                .map_or(1, |n| n.max(1) as usize);
            tracing::info!(?answer, "spatial model daemon loaded");
            Model::Daemon(daemon)
        };
        self.daemon = Some(model);
        self.kv = Kv::default();

        let trace_name = shell.trace().name.clone();
        let paging = shell.paging();
        self.kv.page_size = paging.page_size;
        let state_rows = shell.has_state();
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
                kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
                kv_page_size: paging.page_size,
                state_slots: if state_rows { paging.slots } else { 0 },
                buffer_pages: shell.buffer_pages(),
                state_slot_bytes: shell.state_slot_bytes(),
                adapter_banks: 0,
                ..PoolFacts::default()
            },
            limits: FireLimits {
                // The model holds one sequence's cache; a fire's lanes run one
                // after another, each cache brought back from its tokens.
                max_lanes: budgets.max_lanes.max(1),
                max_tokens: budgets.max_tokens,
                max_page_refs: paging.pages_per_slot,
                max_context: paging.context(),
                prefill_rows: 0,
                max_readouts: budgets.max_tokens,
            },
            has_lora: false,
            has_attn_page_mask: false,
            draft_block: 0,
            draft_mask_token: 0,
            draft_proposals_from: 1,
            // A mask on its port is read (and refused unless causal).
            ports: PortMask::DEVICE_GEOMETRY.with(eta_ir::registry::Port::AttnMask),
            geometry: GeometryClass::DeviceGeometry,
            kv_copy: KvCopyDomains {
                // Pages are tokens here: a copy copies them.
                device_to_device: true,
                device_to_host: false,
                host_to_device: false,
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
        if std::env::var_os("PIE_SPATIAL_TRACE").is_some() {
            eprintln!("spatial-trace copy_kv: {copy:?}");
        }
        let device = self.caps.as_ref().map(|caps| caps.device.domain);
        if Some(copy.src) != device || Some(copy.dst) != device {
            return Err(Error::Unsupported {
                verb: "`copy_kv` between domains other than this load's device",
                engine: "cerebras-spatial",
            });
        }
        let size = self.kv.page_size.max(1);
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
        self.programs.bind(binding).map_err(refusal)
    }

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<()> {
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
        if std::env::var_os("PIE_SPATIAL_TRACE").is_some() {
            eprintln!(
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
