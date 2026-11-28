//! The wafer program in process ([`ArtifactModel`]): tokens at positions, a launch of S passes,
//! the token loop's out-token ring. The program is an [`Artifact`] (`wafer-compile link`'s
//! `artifact/`); the weights are the checkpoint's cut for its plan ([`Weights`]). It keeps one
//! sequence's state on the wafer; a pass at position 0 starts a new one, and a run that starts
//! behind the tokens the host has seen replays from 0.

pub mod place;

use std::path::{Path, PathBuf};
use std::time::Instant;

use kernels_cerebras::compile::artifact::Artifact;
use kernels_cerebras::compile::emit::host::{HostLayout, SeatsHost};
use kernels_cerebras::compile::floor;
use kernels_cerebras::compile::npy::Npy;

use crate::device::Device;
use crate::sdk::Rect;

type R<T> = std::result::Result<T, String>;

/// The artifact's directory in a compiled program.
pub const ARTIFACT: &str = "artifact";

/// Where a cut's tiles are, and their file prefix (`<dir>` or `<dir>:<prefix>`).
#[derive(Clone, Debug)]
pub struct Weights {
    pub prep: PathBuf,
    pub prefix: String,
}

impl Weights {
    pub fn parse(v: &str) -> R<Weights> {
        let (prep, prefix) = v.split_once(':').unwrap_or((v, ""));
        if prefix.contains(':') {
            return Err(format!("weights {v}: <dir>:<prefix>"));
        }
        Ok(Weights {
            prep: PathBuf::from(prep),
            prefix: prefix.to_string(),
        })
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

/// The artifact of the compiled program at `program`.
pub fn read(program: &Path) -> R<Artifact> {
    let dir = program.join(ARTIFACT);
    if !dir.join("program.json").exists() {
        return Err(format!(
            "{}: no artifact (wafer-compile build --plan <plan.json> --out <dir>, then wafer-compile link --build <dir>)",
            dir.display()
        ));
    }
    Artifact::read(&dir)
}

/// The wafer program on a loaded program. Everything program-specific comes from its
/// [`HostLayout`]; nothing here names a family.
pub struct ArtifactModel {
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
    /// Host time by part of a launch, all launches together (embedding row, `put`s, `compute`, the
    /// ring's read), ns.
    pub host_ns: [u64; 4],
    hl: HostLayout,
    /// Every CX row's x offset in the placed hidden vector ([`HostLayout::x_offsets`]).
    x_off: Vec<usize>,
    embed: Embed,
    eps: f32,
    /// `lgt`'s prescale (the plan's: the band head's 2^15 logits; 1 without one).
    lgt_scale: f32,
    /// A ring's seats' KV windows in its shared pool: (first position, positions), the pool split
    /// evenly until [`ArtifactModel::windows`] states others (empty: fixed windows or one seat).
    wins: Vec<(usize, usize)>,
    /// The wafer holds `wins`' first positions (`sbase`): put again only once they change.
    wins_put: bool,
    /// A ring's per-layer-input files, a storage column's (`pt<store>_<n>`, [`HostLayout`]'s
    /// `ple`): a launch's first lap's words are the host's.
    ple: Vec<Vec<Rows>>,
}

/// Where a token's embedding row comes from: the head's tiles (`hemb`), else (a tied head with
/// no tiles) the embedding table `embed` (bf16 [V][H]).
enum Embed {
    Tiles { hemb: Rows, words: u32 },
    Table(Rows),
}

/// A `.npy` the host reads a token's row from at every launch, through a map of the file (a
/// head's tiles hold the row as one slice a head PE), by `pread` where it can't be mapped.
struct Rows {
    npy: Npy,
    map: Option<ztensor::Store>,
}

impl Rows {
    fn open(npy: Npy, path: &Path) -> Rows {
        let map = ztensor::Store::map(path, "npy")
            .ok()
            .filter(|m| m.bytes().is_some());
        if let Some(m) = &map {
            // the tiles were just placed from this file: prefetch keeps the first launches' reads
            // off disk
            let _ = m.prefetch(0, m.len());
        }
        Rows { npy, map }
    }

    /// Elements `at .. at + n` (flat C order), one a word, as [`Npy::words`].
    fn words(&self, at: usize, n: usize, out: &mut Vec<u32>) -> R<()> {
        let Some(bytes) = self.map.as_ref().and_then(|m| m.bytes()) else {
            return self.npy.words(at, n, out);
        };
        let elem = self.npy.elem;
        let from = self.npy.data_offset() as usize + at * elem;
        let b = bytes
            .get(from..from + n * elem)
            .ok_or_else(|| format!("embedding row {at}+{n} past the file"))?;
        if elem == 2 {
            out.extend(
                b.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_le_bytes(*c) as u32),
            );
        } else {
            out.extend(b.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)));
        }
        Ok(())
    }
}

/// A ring's pool split evenly over its seats, each window from a multiple of the rows (none
/// without a pool).
fn even_windows(hl: &HostLayout) -> Vec<(usize, usize)> {
    let Some(s) = hl.seats.as_ref().filter(|s| s.pool > 0 && s.m > 1) else {
        return Vec::new();
    };
    let rows = hl.pool_rows() as usize;
    let span = s.pool as usize / s.m as usize / rows * rows;
    (0..s.m as usize)
        .map(|j| (j * span, span.min(hl.positions as usize)))
        .collect()
}

impl ArtifactModel {
    /// Places the prep on `dev` by `art`'s bindings ([`place::place`]) and runs init.
    pub fn open(
        mut dev: Box<dyn Device>,
        spec: &Weights,
        art: &Artifact,
        vocab: usize,
    ) -> R<ArtifactModel> {
        let hl = art.host.clone();
        let t1 = std::time::Instant::now();
        place::place(dev.as_mut(), art, &spec.prep, &spec.prefix)?;
        tracing::info!(s = t1.elapsed().as_secs_f64(), "artifact: placed");
        dev.call("init", &[]).map_err(|e| format!("init: {e}"))?;
        // A launch can return before every PE ran it: a read from every PE waits for each one's
        // init.
        dev.get("rcfg", rect(&floor::Rect::new(0, 0, hl.w, hl.h)), 1)?;
        let model = &art.header.model;
        let embed = if spec.has("hemb") {
            let hemb = spec.npy("hemb")?;
            let words = if hl.band.is_none() && hl.vocab_rows > 0 {
                hemb.shape[2] as u32 / hl.vocab_rows
            } else {
                hl.hemb_words
            };
            Embed::Tiles {
                hemb: Rows::open(hemb, &spec.file("hemb.npy")),
                words,
            }
        } else {
            Embed::Table(Rows::open(spec.npy("embed")?, &spec.file("embed.npy")))
        };
        let mut ple = Vec::new();
        for st in hl.seats.iter().flat_map(|s| &s.ple) {
            let mut files = Vec::new();
            while spec.has(&format!("pt{}_{}", st.store, files.len())) {
                let name = format!("pt{}_{}", st.store, files.len());
                files.push(Rows::open(spec.npy(&name)?, &spec.file(&format!("{name}.npy"))));
            }
            ple.push(files);
        }
        Ok(ArtifactModel {
            vocab: vocab.min(model.vocab as usize),
            hidden: model.hidden as usize,
            cap: hl.cap as usize,
            positions: hl.positions as usize,
            fed: Vec::new(),
            launches: 0,
            steps: 0,
            host_ns: [0; 4],
            x_off: hl.x_offsets(),
            embed,
            eps: model.eps,
            lgt_scale: model.logit_scale,
            wins: even_windows(&hl),
            wins_put: false,
            ple,
            hl,
            dev,
        })
    }

    /// The launch's first lap's token ids (a ring's seats', else one) to the PLE storage's lookup
    /// (`tin`), where the program has one.
    fn put_token(&mut self, v: &[u32]) -> R<()> {
        let Some(r) = self.hl.token_in else {
            return Ok(());
        };
        let words: Vec<u32> = (0..r.w * r.h).flat_map(|_| v.iter().copied()).collect();
        let t0 = Instant::now();
        let r = self.dev.put("tin", rect(&r), v.len(), &words);
        self.host_ns[1] += ns(t0);
        r
    }

    /// A ring's first lap's per-layer inputs, each seat's first token's (`pls`, a seat after
    /// another in every CY row PE of a storage column's units): the bus carries one token a pass.
    fn put_ple(&mut self, s: &SeatsHost, first: &[u32]) -> R<()> {
        let t0 = Instant::now();
        for (st, files) in s.ple.iter().zip(&self.ple) {
            let (nt, lv, cols) = (st.tokens as usize, st.words as usize, st.cols as usize);
            let mut col: Vec<Vec<u32>> = vec![Vec::with_capacity(first.len() * (lv + 1)); cols];
            let mut h = Vec::with_capacity(lv);
            for &t in first {
                let r = t as usize / nt;
                let f = files
                    .get(r / st.per as usize)
                    .ok_or_else(|| format!("artifact: no PLE file of store {} for token {t}", st.store))?;
                for (c, w) in col.iter_mut().enumerate() {
                    h.clear();
                    let at = ((r % st.per as usize * cols + c) * nt + t as usize % nt) * lv;
                    f.words(at, lv, &mut h)?;
                    w.extend(h.iter().map(|&b| f16_bits(b as u16).to_bits()));
                    w.push(1f32.to_bits());
                }
            }
            for &(r, c0) in &st.cy {
                let words: Vec<u32> = (0..r.w as usize)
                    .flat_map(|i| col[c0 as usize + i].iter().copied())
                    .collect();
                self.dev.put("pls", rect(&r), first.len() * (lv + 1), &words)?;
            }
        }
        self.host_ns[1] += ns(t0);
        Ok(())
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
        let t0 = Instant::now();
        let mut e: Vec<u32> = self.embedding(v)?.into_iter().map(|b| b << 16).collect();
        self.host_ns[0] += ns(t0);
        let t0 = Instant::now();
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
        self.host_ns[1] += ns(t0);
        Ok(())
    }

    fn launch(&mut self, steps: usize, pos: usize) -> R<()> {
        let t0 = Instant::now();
        self.dev
            .call("compute", &[steps as u32, pos as u32])
            .map_err(|e| format!("compute({steps}, {pos}): {e}"))?;
        self.host_ns[2] += ns(t0);
        self.launches += 1;
        self.steps += steps as u64;
        Ok(())
    }

    /// The wafer's sequence brought to `start`: a run behind what it holds replays from 0.
    fn seat(&mut self, start: usize) -> R<()> {
        if start == self.fed.len() {
            return Ok(());
        }
        if start > self.fed.len() {
            return Err(format!(
                "artifact: a run at {start} past the wafer's {} tokens",
                self.fed.len()
            ));
        }
        let keep = self.fed[..start].to_vec();
        tracing::info!(
            rows = start,
            held = self.fed.len(),
            "artifact: sequence replayed from 0"
        );
        self.fed.clear();
        self.feed(&keep, 0)
    }

    /// `tokens` at `start..` one pass a launch (no reads).
    fn feed(&mut self, tokens: &[u32], start: usize) -> R<()> {
        for (i, &t) in tokens.iter().enumerate() {
            self.put_x(t)?;
            self.put_token(&[t])?;
            self.launch(1, start + i)?;
            self.fed.push(t);
        }
        Ok(())
    }

    /// The ring's first `n` winners. An id past the vocabulary (a padded
    /// head row: the prep pads with the last real row so none can win) is
    /// refused rather than handed on.
    fn ring(&mut self, n: usize) -> R<Vec<u32>> {
        let root = floor::Rect::new(self.hl.root.x, self.hl.root.y, 1, 1);
        let t0 = Instant::now();
        let ring = self.dev.get("ring", rect(&root), n)?;
        self.host_ns[3] += ns(t0);
        if let Some(&id) = ring.iter().find(|&&id| id as usize >= self.vocab) {
            return Err(format!(
                "artifact: the token loop chose id {id}, past the vocabulary ({}): a padded head row \
                 won (re-prep: padded rows must copy the last real row)",
                self.vocab
            ));
        }
        Ok(ring)
    }

    /// The token loop: `tokens` at `start..`, then `extra` passes each fed the argmax before (one
    /// launch). The argmax after each row (rows before the feed's last: 0, unknown).
    pub fn greedy(&mut self, tokens: &[u32], start: usize, extra: usize) -> R<Vec<u32>> {
        // (a program without the token loop is refused at load unless the lookahead is 0)
        let n = tokens.len();
        self.seat(start)?;
        let extra = extra
            .min(self.cap - 1)
            .min(self.positions.saturating_sub(start + n));
        self.feed(&tokens[..n - 1], start)?;
        let last = start + n - 1;
        self.put_x(tokens[n - 1])?;
        self.put_token(&tokens[n - 1..n])?;
        self.launch(1 + extra, last)?;
        self.fed.push(tokens[n - 1]);
        let ring = self.ring(1 + extra)?;
        self.fed.extend_from_slice(&ring[..extra]);
        let mut out = vec![0u32; n - 1];
        out.extend(ring);
        Ok(out)
    }

    /// The seats' KV windows from here on, (first position, positions) a seat: each within the
    /// pool, from a multiple of the rows, none over another but the same (seats with no
    /// sequence sharing a scratch window: their cells no one's). A seat's window changes only as it
    /// starts again at position 0 (the attention learns a window's first entry there).
    pub fn windows(&mut self, wins: &[(usize, usize)]) -> R<()> {
        let pool = self.hl.seats.as_ref().map_or(0, |s| s.pool as usize);
        let rows = self.hl.pool_rows() as usize;
        if pool == 0 || wins.len() != self.seats() {
            return Err(format!(
                "artifact: {} windows for {} seats over a pool of {pool}",
                wins.len(),
                self.seats()
            ));
        }
        let mut by: Vec<(usize, usize)> = wins.to_vec();
        by.sort_unstable();
        let fits = by
            .iter()
            .all(|&(b, n)| b % rows == 0 && b + n <= pool && n <= self.positions)
            && by.windows(2).all(|w| w[0] == w[1] || w[0].0 + w[0].1 <= w[1].0);
        if !fits {
            return Err(format!(
                "artifact: windows {wins:?} over a pool of {pool} (rows {rows}, a seat at most {})",
                self.positions
            ));
        }
        self.wins_put &= self.wins == wins;
        self.wins = wins.to_vec();
        Ok(())
    }

    /// The positions seat `j` may run: its window's, else the program's.
    pub fn room(&self, j: usize) -> usize {
        self.wins.get(j).map_or(self.positions, |w| w.1)
    }

    /// The ring's shared KV pool: its positions and rows (none: fixed windows or one seat).
    pub fn pool(&self) -> Option<(usize, usize)> {
        let s = self.hl.seats.as_ref().filter(|s| s.pool > 0 && s.m > 1)?;
        Some((s.pool as usize, self.hl.pool_rows() as usize))
    }

    /// The sequences the program holds at once: its ring's seats (1 without one).
    pub fn seats(&self) -> usize {
        self.hl.seats.as_ref().map_or(1, |s| s.m as usize)
    }

    /// One launch of `n` passes round the seats (pass `s` seat `s % seats`'s): seat `j` starts at position `pos[j]` with token
    /// `first[j]`, its first `force[j].len()` laps take `force[j]` as their next token (then its
    /// own argmax). Every pass's winner (forced passes too). A program of
    /// one seat runs one sequence so (its x and its launch's position).
    pub fn ring_launch(
        &mut self,
        first: &[u32],
        pos: &[usize],
        force: &[Vec<u32>],
        n: usize,
    ) -> R<Vec<u32>> {
        let m = self.seats();
        if first.len() != m || pos.len() != m || force.len() != m {
            return Err(format!(
                "artifact: a ring launch of {} seats, the program's {m}",
                first.len()
            ));
        }
        if n == 0 || n > self.cap {
            return Err(format!("artifact: {n} passes past the ring's {}", self.cap));
        }
        let laps = n.div_ceil(m);
        if let Some(j) = (0..m).find(|&j| pos[j] + laps > self.room(j)) {
            return Err(format!(
                "artifact: {laps} laps from {} past seat {j}'s {} positions",
                pos[j],
                self.room(j)
            ));
        }
        let mut fw = vec![0u32; laps * m];
        let mut nf = vec![0u32; m];
        for (j, f) in force.iter().enumerate() {
            let k = f.len().min(laps);
            nf[j] = k as u32;
            for (lap, &t) in f[..k].iter().enumerate() {
                fw[lap * m + j] = t;
            }
        }
        let root = rect(&floor::Rect::new(self.hl.root.x, self.hl.root.y, 1, 1));
        let t0 = Instant::now();
        self.dev.put("force", root, n, &fw[..n])?;
        self.dev.put("nforce", root, m, &nf)?;
        self.host_ns[1] += ns(t0);
        match self.hl.seats.clone() {
            None => {
                self.put_x(first[0])?;
                self.put_token(&first[..1])?;
                self.launch(n, pos[0])?;
            }
            Some(s) => {
                self.put_ple(&s, first)?;
                let t0 = Instant::now();
                let total = *self.x_off.last().unwrap();
                let mut es = Vec::with_capacity(m);
                for &t in first {
                    let mut e: Vec<u32> = self.embedding(t)?.into_iter().map(|b| b << 16).collect();
                    e.resize(total.max(e.len()), 0);
                    es.push(e);
                }
                self.host_ns[0] += ns(t0);
                let t0 = Instant::now();
                let cx = self.hl.first_cx;
                for &(r0, r1, dims, ly) in &self.hl.x_runs {
                    let mut w = Vec::with_capacity((r1 - r0) as usize * m * dims as usize);
                    for r in r0 as usize..r1 as usize {
                        for e in &es {
                            w.extend_from_slice(&e[self.x_off[r]..self.x_off[r + 1]]);
                        }
                    }
                    let r = floor::Rect::new(cx.x, cx.y + ly, 1, r1 - r0);
                    self.dev.put("xs", rect(&r), m * dims as usize, &w)?;
                }
                let sp: Vec<u32> = pos.iter().map(|&p| p as u32).collect();
                let sb: Vec<u32> = self.wins.iter().map(|w| w.0 as u32).collect();
                for r in &s.cy {
                    let words: Vec<u32> = (0..r.w * r.h).flat_map(|_| sp.iter().copied()).collect();
                    self.dev.put("spos", rect(r), m, &words)?;
                    if !sb.is_empty() && !self.wins_put {
                        let words: Vec<u32> =
                            (0..r.w * r.h).flat_map(|_| sb.iter().copied()).collect();
                        self.dev.put("sbase", rect(r), m, &words)?;
                    }
                }
                self.wins_put = true;
                self.host_ns[1] += ns(t0);
                self.launch(n, 0)?;
            }
        }
        // (the one-sequence history no longer stands)
        self.fed.clear();
        self.ring(n)
    }

    /// The head's logits of the last pass as read (`lgt`, f32 bits, the final norm's 1/rms left
    /// out).
    pub fn last_logits(&mut self) -> R<Vec<u32>> {
        let r = rect(&self.hl.logits);
        self.dev.get("lgt", r, self.hl.vocab_rows as usize)
    }

    /// `tokens` at `start..`; with `logits`, the last row's (`lgt` scaled by the final norm's
    /// 1/rms).
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
        let l = self.logits()?;
        let mut am = vec![0u32; tokens.len()];
        if let Some(best) = argmax(&l) {
            *am.last_mut().unwrap() = best;
        }
        Ok((am, Some(l)))
    }

    /// The last pass's logits (`lgt` scaled by the final norm's 1/rms).
    pub fn logits(&mut self) -> R<Vec<f32>> {
        let w = self
            .dev
            .get("lgt", rect(&self.hl.logits), self.hl.vocab_rows as usize)?;
        // `lgt` leaves out the final norm's 1/rms (argmax needs no scale): the last unit's x's rms.
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
        Ok(l)
    }
}

/// An f16's value (bits) as f32, exactly.
fn f16_bits(h: u16) -> f32 {
    let (s, e, m) = (u32::from(h >> 15), u32::from((h >> 10) & 0x1f), u32::from(h & 0x3ff));
    let v = match e {
        0 => (m as f32) * (2f32).powi(-24),
        0x1f => {
            if m == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => f32::from_bits(((e + 127 - 15) << 23) | (m << 13)),
    };
    if s == 1 { -v } else { v }
}

/// `t0`'s age in ns.
fn ns(t0: Instant) -> u64 {
    u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX)
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
