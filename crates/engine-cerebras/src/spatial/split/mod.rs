//! The role-split wafer program (`csl/split`: Qwen3.5, Qwen3, Gemma 4, gpt-oss; one sequence) in
//! process: the layered build of its geometry and placement (`ARCHITECTURE.md`: a [`family`]'s
//! [`plan`] and [`block`]s -> [`floor`] -> [`wiring`] -> [`fabric`]; the kernels' [`abi`] and
//! [`kernels`]; [`emit`], [`weights`]) and
//! the model the engine drives ([`SplitModel`]): tokens at positions, a launch of S passes, the
//! wafer token loop's out-token ring.
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

pub mod abi;
pub mod block;
pub mod emit;
pub mod fabric;
pub mod family;
pub mod floor;
pub mod kernels;
pub mod npy;
pub mod plan;
pub mod prep;
pub mod program;
pub mod role;
pub mod weights;
pub mod wiring;

use std::path::{Path, PathBuf};

use super::device::Device;
use crate::sdk::Rect;
use npy::Npy;

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

/// The role-split model on a loaded program.
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
    wl: bool,
    /// The head's embedding tiles (untied heads; Gemma 12B reads `embed`).
    hemb: Option<Npy>,
    hr: u32,
    hc: u32,
    vr: u32,
    hdc: u32,
    hcc: u32,
    py: u32,
    yc: u32,
    x_cx: u32,
    hctl: u32,
    /// The last unit's CX column (the final residual x, the head's input).
    last_cx: (u32, u32),
    eps: f32,
    /// Prompt rows forced through the token loop (`PIE_CEREBRAS_SPLIT_FORCE=1`).
    force: bool,
    /// The head control root (ring, force, nforce).
    root: (u32, u32),
    /// Unit 0's top row (Gemma's and gpt-oss's units sit off the origin).
    x_y0: u32,
    /// gpt-oss: the head band (its control row's x0, y, count) and lm_head's prescale (`lgt` = 2^15 logits).
    band: Option<(u32, u32, u32)>,
    hscale: f32,
    /// Gemma 4: the per-layer-input words of a launch's first token, its softcap.
    gemma: Option<GemmaPle>,
    /// Gemma 4 12B: the CX rows' dims (attention / plain rows), the embedding table, the softcap.
    g12: Option<G12Rows>,
}

/// Gemma 4 12B's x layout: runs of CX rows of one width (the attention rows' Hc A, the plain rows' Hc W) on one
/// side of the CY row, each row's first dim; the embedding table (bf16 [V][H]); the final logit softcap.
pub struct G12Rows {
    /// (r0, r1, Hc, the run's first wafer-row offset from the unit's top)
    runs: Vec<(u32, u32, u32, u32)>,
    off: Vec<usize>,
    embed: Npy,
    softcap: f32,
}

/// Gemma 4's host side (its step-0 PLE words come from the storage PEs): the final logit softcap.
pub struct GemmaPle {
    softcap: f32,
}

impl SplitModel {
    /// Places the prep on `dev` (a loaded, running program) and runs init: the meta's
    /// [`program::Program`] (any family, `ARCHITECTURE.md`), placed by its bindings
    /// ([`weights::place`]), the host's map of it ([`emit::host::HostLayout`]).
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
            family = emit::meta::family_name(prog.plan.shape.family),
            "split: program built"
        );
        let t1 = std::time::Instant::now();
        weights::place(dev.as_mut(), &prog, &spec.prep, &spec.prefix)?;
        tracing::info!(s = t1.elapsed().as_secs_f64(), "split: placed");
        dev.call("init", &[]).map_err(|e| format!("init: {e}"))?;
        // A launch can return before every PE ran it: a read from every PE
        // waits for each one's init.
        dev.get(
            "rcfg",
            Rect {
                x: 0,
                y: 0,
                w: hl.w as i32,
                h: hl.h as i32,
            },
            1,
        )?;
        let shape = &prog.plan.shape;
        let family = prog.plan.shape.family;
        let softcap = shape.head.final_softcap;
        // the head's embedding tiles, else (Gemma 12B's tied head) the embedding table read by
        // CX row runs
        let hemb = if spec.has("hemb") {
            Some(spec.npy("hemb")?)
        } else {
            None
        };
        let g12 = if hemb.is_none() {
            Some(G12Rows {
                runs: hl.x_runs.clone(),
                off: hl.x_offsets(),
                embed: spec.npy("embed")?,
                softcap: softcap.unwrap_or(30.0),
            })
        } else {
            None
        };
        let gemma = (family == plan::FamilyId::Gemma).then(|| GemmaPle {
            softcap: softcap.unwrap_or(30.0),
        });
        let hdc = match &hemb {
            Some(h) if hl.band.is_none() && hl.vocab_rows > 0 => h.shape[2] as u32 / hl.vocab_rows,
            _ => hl.hemb_words,
        };
        Ok(SplitModel {
            vocab: vocab.min(shape.vocab as usize),
            hidden: shape.hidden as usize,
            cap: hl.cap as usize,
            positions: hl.positions as usize,
            fed: Vec::new(),
            launches: 0,
            steps: 0,
            wl: hl.token_loop,
            hr: hl.head_rows,
            hc: hl.head_cols,
            vr: hl.vocab_rows,
            hdc,
            hcc: hl.row_dims,
            py: hl.rows,
            yc: hl.cy_row,
            x_cx: hl.first_cx.x,
            last_cx: (hl.last_cx.x, hl.last_cx.y),
            force: std::env::var("PIE_CEREBRAS_SPLIT_FORCE").is_ok_and(|v| v == "1"),
            eps: shape.norm.eps,
            hctl: hl.head_ctl.x,
            root: (hl.root.x, hl.root.y),
            x_y0: hl.first_cx.y,
            band: hl.band,
            hscale: if family == plan::FamilyId::GptOss {
                spec.meta
                    .get("HSCALE")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(32768.0) as f32
            } else {
                1.0
            },
            hemb,
            gemma,
            g12,
            dev,
        })
    }

    /// Gemma: the launch's first token id to the head control column (`tin`): its storage PEs look up and send
    /// its PLE words on the launch-start TB word (one copy a launch).
    fn put_pl(&mut self, v: u32, _pos: usize) -> R<()> {
        if self.gemma.is_none() {
            return Ok(());
        }
        let words = vec![v; self.hr as usize];
        self.dev.put(
            "tin",
            Rect {
                x: self.hctl as i32,
                y: 0,
                w: 1,
                h: self.hr as i32,
            },
            1,
            &words,
        )
    }

    /// Token `v`'s embedding row (bf16 bits), from the head's tiles.
    pub fn embedding(&self, v: u32) -> R<Vec<u32>> {
        if let Some((_, _, hvc)) = self.band {
            // gpt-oss: [dims-row][vocab column][VR][Hc]
            let (j, k) = (v / self.vr, v % self.vr);
            let mut out = Vec::with_capacity((self.py * self.hcc) as usize);
            let he = self.hemb.as_ref().ok_or("split: no embedding tiles")?;
            for i in 0..self.py as usize {
                he.words(
                    ((i * hvc as usize + j as usize) * self.vr as usize + k as usize)
                        * self.hcc as usize,
                    self.hcc as usize,
                    &mut out,
                )?;
            }
            return Ok(out);
        }
        let (r, j) = (v / self.vr, v % self.vr);
        let per = (self.vr * self.hdc) as usize;
        let mut out = Vec::with_capacity((self.hc * self.hdc) as usize);
        for c in 0..self.hc as usize {
            self.hemb
                .as_ref()
                .ok_or("split: no embedding tiles")?
                .words(
                    (r as usize * self.hc as usize + c) * per + (j * self.hdc) as usize,
                    self.hdc as usize,
                    &mut out,
                )?;
        }
        Ok(out)
    }

    /// `v`'s embedding into unit 0's CX column (the next pass's x).
    fn put_x(&mut self, v: u32) -> R<()> {
        if let Some(gr) = &self.g12 {
            let h = self.hidden;
            let mut raw = Vec::with_capacity(h);
            gr.embed.words(v as usize * h, h, &mut raw)?;
            let mut e: Vec<u32> = raw.into_iter().map(|b| b << 16).collect();
            e.resize(*gr.off.last().unwrap(), 0);
            for &(r0, r1, hc, ly0) in &gr.runs {
                let (a, b) = (gr.off[r0 as usize], gr.off[r1 as usize]);
                self.dev.put(
                    "x",
                    Rect {
                        x: self.x_cx as i32,
                        y: (self.x_y0 + ly0) as i32,
                        w: 1,
                        h: (r1 - r0) as i32,
                    },
                    hc as usize,
                    &e[a..b],
                )?;
            }
            return Ok(());
        }
        let e: Vec<u32> = self.embedding(v)?.into_iter().map(|b| b << 16).collect();
        let hc = self.hcc as usize;
        for (r0, r1) in [(0, self.yc), (self.yc, self.py)] {
            let y0 = self.x_y0 + r0 + u32::from(r0 >= self.yc);
            self.dev.put(
                "x",
                Rect {
                    x: self.x_cx as i32,
                    y: y0 as i32,
                    w: 1,
                    h: (r1 - r0) as i32,
                },
                hc,
                &e[r0 as usize * hc..r1 as usize * hc],
            )?;
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
        if self.force && self.wl && tokens.len() > 1 {
            for (i, chunk) in tokens.chunks(self.cap).enumerate() {
                self.forced(chunk, start + i * self.cap, 0, false)?;
            }
            return Ok(());
        }
        for (i, &t) in tokens.iter().enumerate() {
            self.put_x(t)?;
            self.put_pl(t, start + i)?;
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
        let root = Rect {
            x: self.root.0 as i32,
            y: self.root.1 as i32,
            w: 1,
            h: 1,
        };
        self.put_x(tokens[0])?;
        self.put_pl(tokens[0], start)?;
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
        let ring = self.dev.get(
            "ring",
            Rect {
                x: self.root.0 as i32,
                y: self.root.1 as i32,
                w: 1,
                h: 1,
            },
            n,
        )?;
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
        if !self.wl {
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
        self.put_pl(tokens[n - 1], last)?;
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
        let lr = match self.band {
            // gpt-oss: the control row, one vocabulary column a PE
            Some((x0, y, n)) => Rect {
                x: x0 as i32,
                y: y as i32,
                w: n as i32,
                h: 1,
            },
            None => Rect {
                x: self.hctl as i32,
                y: 0,
                w: 1,
                h: self.hr as i32,
            },
        };
        let w = self.dev.get("lgt", lr, self.vr as usize)?;
        // `lgt` leaves out the final norm's 1/rms (argmax needs no scale):
        // the rms of the head's input, the last unit's x.
        let hc = self.hcc as usize;
        let mut ss = 0f64;
        let mut runs: Vec<(u32, u32, usize, u32)> = Vec::new();
        if let Some(gr) = &self.g12 {
            runs.extend(
                gr.runs
                    .iter()
                    .map(|&(r0, r1, h, ly0)| (r0, r1, h as usize, ly0)),
            );
        }
        for &(r0, r1, h, ly0) in &runs {
            let x = self.dev.get(
                "x",
                Rect {
                    x: self.last_cx.0 as i32,
                    y: (self.last_cx.1 + ly0) as i32,
                    w: 1,
                    h: (r1 - r0) as i32,
                },
                h,
            )?;
            ss += x
                .iter()
                .map(|&b| f64::from(f32::from_bits(b)).powi(2))
                .sum::<f64>();
        }
        for (r0, r1) in [(0, self.yc), (self.yc, self.py)] {
            if self.g12.is_some() {
                break;
            }
            let y0 = self.last_cx.1 + r0 + u32::from(r0 >= self.yc);
            let x = self.dev.get(
                "x",
                Rect {
                    x: self.last_cx.0 as i32,
                    y: y0 as i32,
                    w: 1,
                    h: (r1 - r0) as i32,
                },
                hc,
            )?;
            ss += x
                .iter()
                .map(|&b| f64::from(f32::from_bits(b)).powi(2))
                .sum::<f64>();
        }
        // (over the model's hidden size: a padded hidden's dims past it hold 0)
        let n = self.hidden as f64;
        let scale = (1.0 / (ss / n + f64::from(self.eps)).sqrt()) as f32 / self.hscale;
        let mut l: Vec<f32> = w.into_iter().map(|b| f32::from_bits(b) * scale).collect();
        // the head's padded rows (ids >= vocab) never reach sampling or the argmax
        l.truncate(self.vocab);
        let cap = self
            .gemma
            .as_ref()
            .map(|gp| gp.softcap)
            .or(self.g12.as_ref().map(|gr| gr.softcap));
        if let Some(c) = cap {
            // Gemma's final logit softcap (monotone: the argmax is the wafer's)
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
