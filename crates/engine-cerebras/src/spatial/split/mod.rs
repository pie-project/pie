//! The role-split wafer program in process ([`SplitModel`]): tokens at positions, a launch of S
//! passes, the wafer token loop's out-token ring. The program's build (its plan, blocks, floor,
//! fabric, kernels and bindings) is `kernels_cerebras::spatial`; [`place`] places it.
//!
//! The program keeps one sequence's KV and GDN state on the wafer, by
//! position; a pass at position 0 starts a new sequence (the kernels zero
//! their state). The host keeps the tokens the wafer has taken, so a run
//! that starts behind them (a fork, a rejected row) replays from 0.
//!
//! Selected by `PIE_CEREBRAS_SPLIT=<prep dir>:<prefix>[:k=v,...]`: the
//! prepared tiles (`<prefix>meta.json`, `<prefix>w<u>.npy`, ...; built by
//! the prep step from the checkpoint) and the geometry's overrides (the
//! build's `G2`, `WL`, `CAP`).

pub mod place;

use std::path::{Path, PathBuf};

use kernels_cerebras::spatial::npy::Npy;
use kernels_cerebras::spatial::{emit, floor, program};

use super::device::Device;
use crate::sdk::Rect;

type R<T> = std::result::Result<T, String>;

/// `PIE_CEREBRAS_SPLIT`: the prep directory, the file prefix, the overrides.
#[derive(Clone, Debug)]
pub struct SplitSpec {
    pub prep: PathBuf,
    pub prefix: String,
    pub meta: serde_json::Map<String, serde_json::Value>,
}

impl SplitSpec {
    pub fn from_env() -> R<Option<SplitSpec>> {
        match std::env::var("PIE_CEREBRAS_SPLIT") {
            Ok(v) if !v.is_empty() => SplitSpec::parse(&v).map(Some),
            _ => Ok(None),
        }
    }

    pub fn parse(v: &str) -> R<SplitSpec> {
        let mut it = v.splitn(3, ':');
        let prep = PathBuf::from(it.next().unwrap_or("."));
        let prefix = it.next().unwrap_or("").to_string();
        let path = prep.join(format!("{prefix}meta.json"));
        let mut meta: serde_json::Map<String, serde_json::Value> = serde_json::from_str(
            &std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?,
        )
        .map_err(|e| format!("{}: {e}", path.display()))?;
        for kv in it.next().unwrap_or("").split(',').filter(|s| !s.is_empty()) {
            let (k, x) = kv
                .split_once('=')
                .ok_or_else(|| format!("PIE_CEREBRAS_SPLIT override {kv}: k=v"))?;
            let n: i64 = x.parse().map_err(|e| format!("override {kv}: {e}"))?;
            let val = match meta.get(k) {
                Some(serde_json::Value::Bool(_)) => serde_json::Value::Bool(n != 0),
                _ => serde_json::Value::from(n),
            };
            meta.insert(k.to_string(), val);
        }
        Ok(SplitSpec { prep, prefix, meta })
    }

    pub fn int(&self, k: &str) -> Option<u32> {
        match self.meta.get(k)? {
            serde_json::Value::Bool(b) => Some(u32::from(*b)),
            v => v.as_u64().map(|x| x as u32),
        }
    }

    pub fn file(&self, name: &str) -> PathBuf {
        self.prep.join(format!("{}{name}", self.prefix))
    }

    pub fn npy(&self, name: &str) -> R<Npy> {
        Npy::open(&self.file(&format!("{name}.npy")))
    }

    pub fn has(&self, name: &str) -> bool {
        self.file(&format!("{name}.npy")).exists()
    }
}

/// The role-split model on a loaded program. Everything program-specific comes from its
/// [`emit::host::HostLayout`]; nothing here names a family.
pub struct SplitModel {
    pub dev: Box<dyn Device>,
    pub vocab: usize,
    pub hidden: usize,
    /// Passes a launch can run (the head control root's ring).
    pub cap: usize,
    pub positions: usize,
    /// Tokens the wafer has taken, from position 0.
    pub fed: Vec<u32>,
    pub launches: u64,
    pub steps: u64,
    hl: emit::host::HostLayout,
    /// Every CX row's x offset in the placed hidden vector ([`emit::host::HostLayout::x_offsets`]).
    x_off: Vec<usize>,
    embed: Embed,
    eps: f32,
    /// `lgt`'s prescale (the prep's `HSCALE`: the band head's 2^15 logits; 1 without one).
    lgt_scale: f32,
    /// Prompt rows forced through the token loop (`PIE_CEREBRAS_SPLIT_FORCE=1`).
    force: bool,
}

/// Where a token's embedding row comes from: the head's tiles (`hemb`, the head's layout:
/// `words` a head PE), else (a tied head with no tiles) the embedding table `embed` (bf16 [V][H]).
enum Embed {
    Tiles { hemb: Npy, words: u32 },
    Table(Npy),
}

impl SplitModel {
    /// Places the prep on `dev` (a loaded, running program) and runs init: the meta's
    /// [`program::Program`] (any family, `ARCHITECTURE.md`), placed by its bindings
    /// ([`place::place`]), the host's map of it ([`emit::host::HostLayout`]).
    pub fn open(mut dev: Box<dyn Device>, spec: &SplitSpec, vocab: usize) -> R<SplitModel> {
        let t0 = std::time::Instant::now();
        let meta = emit::meta::Meta {
            map: spec.meta.clone().into_iter().collect(),
        };
        let prog = program::Program::from_meta(&meta)?;
        let hl = emit::host::HostLayout::of(&prog)?;
        tracing::info!(
            ms = t0.elapsed().as_millis() as u64,
            w = hl.w,
            h = hl.h,
            family = prog.plan.shape.family.name(),
            "split: program built"
        );
        let t1 = std::time::Instant::now();
        place::place(dev.as_mut(), &prog, &spec.prep, &spec.prefix)?;
        tracing::info!(s = t1.elapsed().as_secs_f64(), "split: placed");
        dev.call("init", &[]).map_err(|e| format!("init: {e}"))?;
        // A launch can return before every PE ran it: a read from every PE
        // waits for each one's init.
        dev.get("rcfg", rect(&floor::Rect::new(0, 0, hl.w, hl.h)), 1)?;
        let shape = &prog.plan.shape;
        let embed = if spec.has("hemb") {
            let hemb = spec.npy("hemb")?;
            let words = if hl.band.is_none() && hl.vocab_rows > 0 {
                hemb.shape[2] as u32 / hl.vocab_rows
            } else {
                hl.hemb_words
            };
            Embed::Tiles { hemb, words }
        } else {
            Embed::Table(spec.npy("embed")?)
        };
        Ok(SplitModel {
            vocab: vocab.min(shape.vocab as usize),
            hidden: shape.hidden as usize,
            cap: hl.cap as usize,
            positions: hl.positions as usize,
            fed: Vec::new(),
            launches: 0,
            steps: 0,
            x_off: hl.x_offsets(),
            embed,
            eps: shape.norm.eps,
            lgt_scale: meta.float("HSCALE").unwrap_or(1.0) as f32,
            force: std::env::var("PIE_CEREBRAS_SPLIT_FORCE").is_ok_and(|v| v == "1"),
            hl,
            dev,
        })
    }

    /// The launch's first token id to the PLE storage's lookup (`tin`), where the program has
    /// one: its storage PEs send its per-layer-input words on the launch-start word.
    fn put_token(&mut self, v: u32) -> R<()> {
        let Some(r) = self.hl.token_in else {
            return Ok(());
        };
        let words = vec![v; (r.w * r.h) as usize];
        self.dev.put("tin", rect(&r), 1, &words)
    }

    /// Token `v`'s embedding row (bf16 bits), from the head's tiles or the table.
    pub fn embedding(&self, v: u32) -> R<Vec<u32>> {
        let (hemb, words) = match &self.embed {
            Embed::Tiles { hemb, words } => (hemb, *words as usize),
            Embed::Table(t) => {
                let mut out = Vec::with_capacity(self.hidden);
                t.words(v as usize * self.hidden, self.hidden, &mut out)?;
                return Ok(out);
            }
        };
        let vr = self.hl.vocab_rows;
        let (q, k) = ((v / vr) as usize, (v % vr) as usize);
        let mut out = Vec::new();
        match self.hl.band {
            // the band's tiles: [dims row][vocabulary column][VR][dims]
            Some((_, _, cols)) => {
                let hc = self.hl.row_dims as usize;
                for i in 0..self.hl.rows as usize {
                    let at = ((i * cols as usize + q) * vr as usize + k) * hc;
                    hemb.words(at, hc, &mut out)?;
                }
            }
            // a column head's tiles: [vocabulary row][dims column][VR][words]
            None => {
                let cols = self.hl.head_cols as usize;
                for c in 0..cols {
                    let at = (q * cols + c) * vr as usize * words + k * words;
                    hemb.words(at, words, &mut out)?;
                }
            }
        }
        Ok(out)
    }

    /// `v`'s embedding into unit 0's CX column (the next pass's x), by the x runs.
    fn put_x(&mut self, v: u32) -> R<()> {
        let mut e: Vec<u32> = self.embedding(v)?.into_iter().map(|b| b << 16).collect();
        let total = *self.x_off.last().unwrap();
        if e.len() < total {
            // (a padded hidden's dims past the model's are 0)
            e.resize(total, 0);
        }
        let cx = self.hl.first_cx;
        for &(r0, r1, dims, ly) in &self.hl.x_runs {
            let (a, b) = (self.x_off[r0 as usize], self.x_off[r1 as usize]);
            let r = floor::Rect::new(cx.x, cx.y + ly, 1, r1 - r0);
            self.dev.put("x", rect(&r), dims as usize, &e[a..b])?;
        }
        Ok(())
    }

    fn launch(&mut self, steps: usize, pos: usize) -> R<()> {
        self.dev
            .call("compute", &[steps as u32, pos as u32])
            .map_err(|e| format!("compute({steps}, {pos}): {e}"))?;
        self.launches += 1;
        self.steps += steps as u64;
        Ok(())
    }

    /// The wafer's sequence brought to `start` (what it holds agrees up to
    /// there, by the engine's account): a run behind it replays from 0.
    fn seat(&mut self, start: usize) -> R<()> {
        if start == self.fed.len() {
            return Ok(());
        }
        if start > self.fed.len() {
            return Err(format!(
                "split: a run at {start} past the wafer's {} tokens",
                self.fed.len()
            ));
        }
        let keep = self.fed[..start].to_vec();
        tracing::info!(
            rows = start,
            held = self.fed.len(),
            "split: sequence replayed from 0"
        );
        self.fed.clear();
        self.feed(&keep, 0)
    }

    /// `tokens` at `start..` one pass a launch (no reads).
    fn feed(&mut self, tokens: &[u32], start: usize) -> R<()> {
        if self.force && self.hl.token_loop && tokens.len() > 1 {
            for (i, chunk) in tokens.chunks(self.cap).enumerate() {
                self.forced(chunk, start + i * self.cap, 0, false)?;
            }
            return Ok(());
        }
        for (i, &t) in tokens.iter().enumerate() {
            self.put_x(t)?;
            self.put_token(t)?;
            self.launch(1, start + i)?;
            self.fed.push(t);
        }
        Ok(())
    }

    /// `tokens` and `extra` passes past them in one launch
    /// (`PIE_CEREBRAS_SPLIT_FORCE=1`, a build with the head's force table):
    /// the first token's x from the host, then the root's winner at steps
    /// s < n - 1 forced to `tokens[s + 1]` (the ring keeps the argmax). The
    /// ring's first `n + extra` words when `read`.
    fn forced(&mut self, tokens: &[u32], start: usize, extra: usize, read: bool) -> R<Vec<u32>> {
        let n = tokens.len();
        let root = rect(&floor::Rect::new(self.hl.root.x, self.hl.root.y, 1, 1));
        self.put_x(tokens[0])?;
        self.put_token(tokens[0])?;
        if n > 1 {
            self.dev.put("force", root, n - 1, &tokens[1..])?;
            self.dev.put("nforce", root, 1, &[(n - 1) as u32])?;
        }
        self.launch(n + extra, start)?;
        if n > 1 {
            self.dev.put("nforce", root, 1, &[0])?;
        }
        self.fed.extend_from_slice(tokens);
        if !read {
            return Ok(Vec::new());
        }
        let ring = self.ring(n + extra)?;
        self.fed.extend_from_slice(&ring[n - 1..n - 1 + extra]);
        Ok(ring)
    }

    /// The ring's first `n` winners. An id past the vocabulary (a padded
    /// head row: the prep pads with the last real row so none can win) is
    /// refused rather than handed on.
    fn ring(&mut self, n: usize) -> R<Vec<u32>> {
        let root = floor::Rect::new(self.hl.root.x, self.hl.root.y, 1, 1);
        let ring = self.dev.get("ring", rect(&root), n)?;
        if let Some(&id) = ring.iter().find(|&&id| id as usize >= self.vocab) {
            return Err(format!(
                "split: the token loop chose id {id}, past the vocabulary ({}): a padded head row \
                 won (re-prep: padded rows must copy the last real row)",
                self.vocab
            ));
        }
        Ok(ring)
    }

    /// The token loop: `tokens` at `start..`, then `extra` passes each fed
    /// the argmax before (one launch on the wafer). The argmax after each
    /// row (rows before the feed's last: 0, unknown).
    pub fn greedy(&mut self, tokens: &[u32], start: usize, extra: usize) -> R<Vec<u32>> {
        if !self.hl.token_loop {
            return Err("split: the token loop wants a WL build".into());
        }
        let n = tokens.len();
        self.seat(start)?;
        let extra = extra
            .min(self.cap - 1)
            .min(self.positions.saturating_sub(start + n));
        if self.force && n > 1 {
            // the feed's head in whole launches, its tail and the loop one launch
            let tail = n.min(self.cap - extra);
            let head = n - tail;
            self.feed(&tokens[..head], start)?;
            let ring = self.forced(&tokens[head..], start + head, extra, true)?;
            let mut out = vec![0u32; head];
            out.extend(ring);
            return Ok(out);
        }
        self.feed(&tokens[..n - 1], start)?;
        let last = start + n - 1;
        self.put_x(tokens[n - 1])?;
        self.put_token(tokens[n - 1])?;
        self.launch(1 + extra, last)?;
        self.fed.push(tokens[n - 1]);
        let ring = self.ring(1 + extra)?;
        self.fed.extend_from_slice(&ring[..extra]);
        let mut out = vec![0u32; n - 1];
        out.extend(ring);
        Ok(out)
    }

    /// `tokens` at `start..`; with `logits`, the last row's (the head rows'
    /// `lgt`, scaled by the final norm's 1/rms of the last unit's x).
    pub fn forward(
        &mut self,
        tokens: &[u32],
        start: usize,
        logits: bool,
    ) -> R<(Vec<u32>, Option<Vec<f32>>)> {
        self.seat(start)?;
        self.feed(tokens, start)?;
        if !logits {
            return Ok((vec![0; tokens.len()], None));
        }
        let w = self
            .dev
            .get("lgt", rect(&self.hl.logits), self.hl.vocab_rows as usize)?;
        // `lgt` leaves out the final norm's 1/rms (argmax needs no scale):
        // the rms of the head's input, the last unit's x.
        let mut ss = 0f64;
        let cx = self.hl.last_cx;
        for &(r0, r1, dims, ly) in &self.hl.x_runs {
            let r = floor::Rect::new(cx.x, cx.y + ly, 1, r1 - r0);
            let x = self.dev.get("x", rect(&r), dims as usize)?;
            ss += x
                .iter()
                .map(|&b| f64::from(f32::from_bits(b)).powi(2))
                .sum::<f64>();
        }
        // (over the model's hidden size: a padded hidden's dims past it hold 0)
        let n = self.hidden as f64;
        let scale = (1.0 / (ss / n + f64::from(self.eps)).sqrt()) as f32 / self.lgt_scale;
        let mut l: Vec<f32> = w.into_iter().map(|b| f32::from_bits(b) * scale).collect();
        // the head's padded rows (ids >= vocab) never reach sampling or the argmax
        l.truncate(self.vocab);
        if let Some(c) = self.hl.softcap {
            for v in l.iter_mut() {
                *v = c * (*v / c).tanh();
            }
        }
        let mut am = vec![0u32; tokens.len()];
        if let Some(best) = argmax(&l) {
            *am.last_mut().unwrap() = best;
        }
        Ok((am, Some(l)))
    }
}

/// The SDK's rectangle of a floor rectangle.
fn rect(r: &floor::Rect) -> Rect {
    Rect {
        x: r.x as i32,
        y: r.y as i32,
        w: r.w as i32,
        h: r.h as i32,
    }
}

/// The lowest index of the largest value (the wafer's tie rule).
pub fn argmax(l: &[f32]) -> Option<u32> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in l.iter().enumerate() {
        if best.is_none_or(|(_, b)| v > b) {
            best = Some((i, v));
        }
    }
    best.map(|(i, _)| i as u32)
}

/// A compiled split program: the env names a prep and the program's
/// out.json has the unit grid.
pub fn is_split(program: &Path) -> bool {
    std::env::var("PIE_CEREBRAS_SPLIT").is_ok_and(|v| !v.is_empty())
        && std::fs::read_to_string(program.join("out.json")).is_ok_and(|s| {
            (s.contains("\"UNX\"") && s.contains("\"UNY\""))
                || (s.contains("\"NTW\"") && s.contains("\"Ic1\"")) // Gemma's layout (mkfull_gemma.py)
                || (s.contains("\"NEX\"") && s.contains("\"QO\"")) // gpt-oss's (mkfull_gptoss.py)
                || (s.contains("\"HcA\"") && s.contains("\"HcW\"")) // Gemma 12B's (mkfull_g12.py)
                || s.contains("\"SPLIT_GEO\"") // the layered build's (emit::layout)
        })
}
