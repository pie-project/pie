//! The role-split wafer program (`csl/split`: Qwen3.5-9B bf16, one
//! sequence) in process: its geometry ([`geo`]: routes and queue roles of
//! every PE), its placement from the prepared tiles ([`place`]) and the
//! model the engine drives ([`SplitModel`]): tokens at positions, a launch
//! of S passes, the wafer token loop's out-token ring.
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

pub mod geo;
pub mod npy;
pub mod place;
pub mod prep;

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

    fn need(&self, k: &str) -> R<u32> {
        self.int(k).ok_or_else(|| format!("split meta: no {k}"))
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

    /// The geometry the meta (and overrides) states, built.
    pub fn geo(&self) -> R<geo::Geo> {
        let g2 = self.int("G2").unwrap_or(0);
        let p = geo::Params {
            px: self.need("PX")?,
            py: self.need("PY")?,
            unx: self.need("UNX")?,
            uny: self.need("UNY")?,
            hc: self.need("HC")?,
            hr: self.need("HR")?,
            hcc: self.need("Hc")?,
            swa: self.need("SWA")?,
            swg: self.need("SWG")?,
            swv: self.need("SWV")?,
            dkg: self.need("DKG")?,
            dkh: self.need("DKH")?,
            g2x: self.int("G2X").unwrap_or(g2),
            g2y: self.int("G2Y").unwrap_or(g2),
            g2h: self.int("G2H").unwrap_or(g2),
            wl: self.int("WL").unwrap_or(0) != 0,
            g2w: self.int("G2W").unwrap_or(24),
        };
        let mut g = geo::Geo::new(p);
        g.build(
            self.int("has_att").unwrap_or(0) != 0,
            self.int("has_gdn").unwrap_or(0) != 0,
        )?;
        Ok(g)
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
    hemb: Npy,
    hr: u32,
    hc: u32,
    vr: u32,
    hdc: u32,
    hcc: u32,
    py: u32,
    yc: u32,
    x_cx: u32,
    hctl: u32,
}

impl SplitModel {
    /// Places the prep on `dev` (a loaded, running program) and runs init.
    pub fn open(mut dev: Box<dyn Device>, spec: &SplitSpec, vocab: usize) -> R<SplitModel> {
        let t0 = std::time::Instant::now();
        let g = spec.geo()?;
        tracing::info!(
            ms = t0.elapsed().as_millis() as u64,
            w = g.w,
            h = g.h,
            "split: geometry built"
        );
        let t1 = std::time::Instant::now();
        place::place(dev.as_mut(), &g, spec)?;
        tracing::info!(s = t1.elapsed().as_secs_f64(), "split: placed");
        dev.call("init", &[]).map_err(|e| format!("init: {e}"))?;
        // A launch can return before every PE ran it: a read from every PE
        // waits for each one's init.
        dev.get(
            "rcfg",
            Rect {
                x: 0,
                y: 0,
                w: g.w as i32,
                h: g.h as i32,
            },
            1,
        )?;
        let wl = g.p.wl;
        let hemb = spec.npy("hemb")?;
        let u0 = g.unit_origin(0);
        if u0 != (0, 0) {
            return Err("split: unit 0 off the origin".into());
        }
        let vr = spec.need("VR")?;
        Ok(SplitModel {
            dev,
            vocab,
            hidden: spec.need("H")? as usize,
            cap: spec.int("CAP").unwrap_or(256) as usize,
            positions: (spec.need("NE")? * g.p.py) as usize,
            fed: Vec::new(),
            launches: 0,
            steps: 0,
            wl,
            hr: g.p.hr,
            hc: g.p.hc,
            vr,
            hdc: hemb.shape[2] as u32 / vr,
            hcc: g.p.hcc,
            py: g.p.py,
            yc: g.yc,
            x_cx: u0.0 + g.xc,
            hctl: g.hx0 + g.hxc,
            hemb,
        })
    }

    /// Token `v`'s embedding row (bf16 bits), from the head's tiles.
    pub fn embedding(&self, v: u32) -> R<Vec<u32>> {
        let (r, j) = (v / self.vr, v % self.vr);
        let per = (self.vr * self.hdc) as usize;
        let mut out = Vec::with_capacity((self.hc * self.hdc) as usize);
        for c in 0..self.hc as usize {
            self.hemb.words(
                (r as usize * self.hc as usize + c) * per + (j * self.hdc) as usize,
                self.hdc as usize,
                &mut out,
            )?;
        }
        Ok(out)
    }

    /// `v`'s embedding into unit 0's CX column (the next pass's x).
    fn put_x(&mut self, v: u32) -> R<()> {
        let e: Vec<u32> = self.embedding(v)?.into_iter().map(|b| b << 16).collect();
        let hc = self.hcc as usize;
        for (r0, r1) in [(0, self.yc), (self.yc, self.py)] {
            let y0 = r0 + u32::from(r0 >= self.yc);
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
        for (i, &t) in tokens.iter().enumerate() {
            self.put_x(t)?;
            self.launch(1, start + i)?;
            self.fed.push(t);
        }
        Ok(())
    }

    /// The ring's first `n` winners.
    fn ring(&mut self, n: usize) -> R<Vec<u32>> {
        self.dev.get(
            "ring",
            Rect {
                x: self.hctl as i32,
                y: (self.hr / 2) as i32,
                w: 1,
                h: 1,
            },
            n,
        )
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
        self.feed(&tokens[..n - 1], start)?;
        let last = start + n - 1;
        self.put_x(tokens[n - 1])?;
        self.launch(1 + extra, last)?;
        self.fed.push(tokens[n - 1]);
        let ring = self.ring(1 + extra)?;
        self.fed.extend_from_slice(&ring[..extra]);
        let mut out = vec![0u32; n - 1];
        out.extend(ring);
        Ok(out)
    }

    /// `tokens` at `start..`; with `logits`, the last row's (the head rows'
    /// `lgt`: positive-scale logits, the final norm's rms left out).
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
        let w = self.dev.get(
            "lgt",
            Rect {
                x: self.hctl as i32,
                y: 0,
                w: 1,
                h: self.hr as i32,
            },
            self.vr as usize,
        )?;
        let mut l: Vec<f32> = w.into_iter().map(f32::from_bits).collect();
        l.truncate(self.vocab);
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
        && std::fs::read_to_string(program.join("out.json"))
            .is_ok_and(|s| s.contains("\"UNX\"") && s.contains("\"UNY\""))
}
