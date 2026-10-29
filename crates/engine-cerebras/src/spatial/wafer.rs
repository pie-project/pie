//! The resident wafer model in process: the compiled `serve28s` (or
//! `serve28`) program
//! (model28's layer strips running serve's request API) loaded through the
//! SDK runtime binding, its weights placed from the checkpoint, requests
//! answered by launches. The Rust form of the bring-up's `serve_lib28.py`.
//!
//! Layers sit in strips of `kernel_cols` columns (`lps` a strip, alternating
//! down and up), the head in the strip after them (up when the layer strips
//! are odd in number), relay rows above and below. A request is one launch:
//! `run_cfg` = [steps, start position] on every PE, the tokens to force on
//! the head's bottom-east PE (`in_tokens`, `SAMPLE` for its own argmax),
//! `compute`, then every step's argmax (`out_tokens`, as `f32`) and, when
//! asked, the last step's logits (`hlogits`, the head's east column).
//! Positions carry on across requests: the KV cache persists until a request
//! starts at 0.
//!
//! The SDK wants every runtime call on the thread that made the first
//! runtime, so the model lives on a thread of its own ([`Wafer`]).

use super::paged::SYMBOLS as SYM;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

use super::ckpt::{Ckpt, Tensor};
use crate::sdk::{DataType, Order, Platform, Rect, Runtime, Sdk};

/// `in_tokens`: take the step's own argmax.
pub const SAMPLE: u32 = 0xFFFF_FFFF;

type R<T> = std::result::Result<T, String>;

/// The program's shape, from the compiler's `out.json` params.
#[derive(Debug, Clone)]
pub struct Layout {
    pub pw: usize,
    pub ph: usize,
    pub hm: usize,
    pub layers: usize,
    pub hh: usize,
    pub lps: usize,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub positions: usize,
    pub rb: usize,
    pub inter: usize,
    pub vr: usize,
    pub vocab: usize,
    pub cap: usize,
    /// Layer strips; the head is strip `ns`.
    pub ns: usize,
    /// Rows a strip; relay rows 0 and `body + 1`.
    pub body: usize,
    pub width: usize,
    pub height: usize,
}

impl Layout {
    /// The shape the compiler recorded in the program's `out.json`.
    pub fn read(program: &Path) -> R<Layout> {
        let path = program.join("out.json");
        let v: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?,
        )
        .map_err(|e| format!("{}: {e}", path.display()))?;
        Layout::of(|k| {
            let x = &v["params"][k];
            x.as_u64()
                .or_else(|| x.as_str().and_then(|s| s.trim().parse().ok()))
                .map(|n| n as usize)
        })
        .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The shape a `--params` string (`name:value,...`) asks for.
    pub fn parse(params: &str) -> R<Layout> {
        let map: std::collections::HashMap<&str, &str> = params
            .split(',')
            .filter_map(|kv| kv.split_once(':'))
            .map(|(k, v)| (k.trim(), v.trim()))
            .collect();
        Layout::of(|k| map.get(k).and_then(|v| v.parse().ok()))
    }

    fn of(get: impl Fn(&str) -> Option<usize>) -> R<Layout> {
        let p = |k: &str| -> R<usize> { get(k).ok_or_else(|| format!("no param {k}")) };
        let (pw, ph, hm) = (p("kernel_cols")?, p("attn_rows")?, p("mlp_rows")?);
        let (layers, lps) = (p("layers")?, p("lps")?);
        let hh = p("head_rows")?;
        if lps == 0 || layers % lps != 0 || hh == 0 || hh > lps * (ph + hm) {
            return Err(format!(
                "{layers} layers at {lps} a strip with a head of {hh} rows do not make a \
                 serve28 layout ({} rows a strip)",
                lps * (ph + hm)
            ));
        }
        let (ns, body) = (layers / lps, lps * (ph + hm));
        Ok(Layout {
            pw,
            ph,
            hm,
            layers,
            hh,
            lps,
            hidden: p("hidden")?,
            heads: p("heads")?,
            head_dim: p("head_dim")?,
            positions: p("positions")?,
            rb: p("band_rows")?,
            inter: p("inter")?,
            vr: p("vocab_rows")?,
            vocab: p("vocab")?,
            cap: p("iters")?,
            ns,
            body,
            width: (ns + 1) * pw,
            height: body + 2,
        })
    }

    /// A strip's logical row on the fabric: even strips down from row 1,
    /// odd ones up from row `body`.
    pub fn prow(&self, flip: bool, ly: usize) -> usize {
        if flip { self.body - ly } else { 1 + ly }
    }

    fn hflip(&self) -> bool {
        self.ns % 2 == 1
    }

    /// The head's bottom-east PE (`in_tokens`, `out_tokens`).
    pub fn hbe(&self) -> (usize, usize) {
        (self.ns * self.pw + self.pw - 1, self.prow(self.hflip(), self.hh - 1))
    }

    /// The top row of the head's east column (`hlogits`).
    fn hcol_y0(&self) -> usize {
        self.prow(self.hflip(), 0).min(self.prow(self.hflip(), self.hh - 1))
    }
}

/// Which wafer program a compiled output directory holds.
pub enum Design {
    /// serve28 / serve28s*: layer strips and the strip head.
    Strips(Layout),
    /// serve2d: qwen2d grids and head2d.
    Grid(super::serve2d::Layout2d),
}

impl Design {
    pub fn read(program: &Path) -> R<Design> {
        let path = program.join("out.json");
        let v: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?,
        )
        .map_err(|e| format!("{}: {e}", path.display()))?;
        let get = |k: &str| {
            let x = &v["params"][k];
            x.as_u64()
                .or_else(|| x.as_str().and_then(|s| s.trim().parse().ok()))
                .map(|n| n as usize)
        };
        match super::serve2d::Layout2d::of(get) {
            Some(grid) => grid
                .map(Design::Grid)
                .map_err(|e| format!("{}: {e}", path.display())),
            None => Layout::read(program).map(Design::Strips),
        }
    }

    pub fn cap(&self) -> usize {
        match self {
            Design::Strips(l) => l.cap,
            Design::Grid(g) => g.cap,
        }
    }

    pub fn positions(&self) -> usize {
        match self {
            Design::Strips(l) => l.positions,
            Design::Grid(g) => g.positions(),
        }
    }

    pub fn vocab(&self) -> usize {
        match self {
            Design::Strips(l) => l.vocab,
            Design::Grid(g) => g.vocab,
        }
    }

    pub fn hidden(&self) -> usize {
        match self {
            Design::Strips(l) => l.hidden,
            Design::Grid(g) => g.hidden,
        }
    }
}

/// Where placed weights go: the runtime, or a recorder for checking.
pub trait Sink {
    fn h2d(&mut self, symbol: &str, rect: Rect, elems_per_pe: usize, words: &[u32]) -> R<()>;
}

impl Sink for Runtime {
    fn h2d(&mut self, symbol: &str, rect: Rect, elems_per_pe: usize, words: &[u32]) -> R<()> {
        self.memcpy_h2d(symbol, words, rect, elems_per_pe, Order::RowMajor, DataType::Bits32)
            .map_err(|e| format!("placing {symbol}: {e}"))
    }
}

/// `data`: `rows` logical rows from logical row `ly0` of `strip`, each `pw`
/// PEs of `n` elements, in logical order.
fn put_strip(
    lay: &Layout,
    sink: &mut dyn Sink,
    symbol: &str,
    data: &[f32],
    strip: usize,
    ly0: usize,
    rows: usize,
    n: usize,
) -> R<()> {
    let row = lay.pw * n;
    debug_assert_eq!(data.len(), rows * row);
    let flip = strip % 2 == 1;
    let (words, y0): (Vec<u32>, usize) = if flip {
        // Logical rows run up the fabric: send them bottom-up, reversed.
        (
            data.chunks_exact(row).rev().flatten().map(|x| x.to_bits()).collect(),
            lay.prow(true, ly0 + rows - 1),
        )
    } else {
        (data.iter().map(|x| x.to_bits()).collect(), lay.prow(false, ly0))
    };
    let rect = Rect {
        x: (strip * lay.pw) as i32,
        y: y0 as i32,
        w: lay.pw as i32,
        h: rows as i32,
    };
    sink.h2d(symbol, rect, n, &words)
}

fn expect(t: &Tensor, name: &str, shape: &[usize]) -> R<()> {
    if t.shape == shape {
        Ok(())
    } else {
        Err(format!(
            "{name} is {:?}; the compiled program wants {shape:?}",
            t.shape
        ))
    }
}

/// Logical head rows a copy of the embedding carries (bounds host memory).
const EMB_CHUNK_ROWS: usize = 64;

/// Places every weight of `ckpt` (and zeroed KV) for `lay`.
#[allow(clippy::too_many_lines)]
pub fn place(lay: &Layout, ckpt: &Ckpt, rope_theta: f64, sink: &mut dyn Sink) -> R<()> {
    let (pw, ph, hm, rb) = (lay.pw, lay.ph, lay.hm, lay.rb);
    let (hid, g, d) = (lay.hidden, lay.heads, lay.head_dim);
    let nb = ph / rb;
    let (ht, dr, tc, it) = (hid / pw, d / rb, lay.positions / pw, lay.inter / hm);
    let lh = ph + hm;
    let qr = (g + 2) * dr;
    let qdim = nb * g * d;
    // Each band row holds dr of a head's dims, RoPE pairs together: dr/2 from
    // the first half and their partners from the second.
    let perm: Vec<Vec<usize>> = (0..rb)
        .map(|r| {
            (r * dr / 2..(r + 1) * dr / 2)
                .chain(d / 2 + r * dr / 2..d / 2 + (r + 1) * dr / 2)
                .collect()
        })
        .collect();
    let inv_freq: Vec<f32> = (0..d / 2)
        .map(|i| rope_theta.powf(-((2 * i) as f64) / d as f64) as f32)
        .collect();
    let sqrt_d = (d as f64).sqrt() as f32;
    let get = |name: &str, shape: &[usize]| -> R<Tensor> {
        let t = ckpt.get(name)?;
        expect(&t, name, shape)?;
        Ok(t)
    };
    for li in 0..lay.layers {
        let started = Instant::now();
        let p = format!("model.layers.{li}.");
        let (strip, k) = (li / lay.lps, li % lay.lps);
        let base = k * lh;
        let wq = get(&format!("{p}self_attn.q_proj.weight"), &[qdim, hid])?;
        let wk = get(&format!("{p}self_attn.k_proj.weight"), &[nb * d, hid])?;
        let wv = get(&format!("{p}self_attn.v_proj.weight"), &[nb * d, hid])?;
        let wo = get(&format!("{p}self_attn.o_proj.weight"), &[hid, qdim])?;
        let qn = get(&format!("{p}self_attn.q_norm.weight"), &[d])?;
        let kn = get(&format!("{p}self_attn.k_norm.weight"), &[d])?;
        let ln = get(&format!("{p}input_layernorm.weight"), &[hid])?;
        let mut wqkv = vec![0f32; ph * pw * ht * qr];
        let mut wob = vec![0f32; ph * pw * g * dr * ht];
        let mut freq = vec![0f32; ph * pw * dr / 2];
        let mut qknw = vec![0f32; ph * pw * 2 * dr];
        let mut gamma = vec![0f32; ph * pw * ht];
        for py in 0..ph {
            let (b, rr) = (py / rb, py % rb);
            let dims = &perm[rr];
            let qrows: Vec<usize> = (0..g)
                .flat_map(|h| dims.iter().map(move |&j| (b * g + h) * d + j))
                .collect();
            // The band row's q, k, v rows of the projection.
            let blk: Vec<&[f32]> = qrows
                .iter()
                .map(|&r| &wq.data[r * hid..(r + 1) * hid])
                .chain(dims.iter().map(|&j| &wk.data[(b * d + j) * hid..(b * d + j + 1) * hid]))
                .chain(dims.iter().map(|&j| &wv.data[(b * d + j) * hid..(b * d + j + 1) * hid]))
                .collect();
            for px in 0..pw {
                let pe = py * pw + px;
                for c in 0..ht {
                    for (r, src) in blk.iter().enumerate() {
                        wqkv[(pe * ht + c) * qr + r] = src[px * ht + c];
                    }
                }
                for (q, &qrow) in qrows.iter().enumerate() {
                    for c in 0..ht {
                        wob[(pe * g * dr + q) * ht + c] = wo.data[(px * ht + c) * qdim + qrow];
                    }
                }
                for i in 0..dr / 2 {
                    freq[pe * dr / 2 + i] = inv_freq[dims[i]];
                }
                // q_norm follows the q projection and would cancel a scale
                // folded into Wq: 1/sqrt(D) goes into q_norm's weights.
                for (i, &j) in dims.iter().enumerate() {
                    qknw[pe * 2 * dr + i] = qn.data[j] / sqrt_d;
                    qknw[pe * 2 * dr + dr + i] = kn.data[j];
                }
                gamma[pe * ht..(pe + 1) * ht].copy_from_slice(&ln.data[px * ht..(px + 1) * ht]);
            }
        }
        let zeros_kv = vec![0f32; ph * pw * dr * tc];
        for (sym, arr, n) in [
            ("wqkv", &wqkv, qr * ht),
            ("wo", &wob, g * dr * ht),
            ("k", &zeros_kv, dr * tc),
            ("v", &zeros_kv, dr * tc),
            ("inv_freq", &freq, dr / 2),
            ("qkn_w", &qknw, 2 * dr),
            ("gamma", &gamma, ht),
        ] {
            put_strip(lay, sink, sym, arr, strip, base, ph, n)?;
        }
        drop((wq, wk, wv, wo, wqkv, wob));

        let gate = get(&format!("{p}mlp.gate_proj.weight"), &[lay.inter, hid])?;
        let up = get(&format!("{p}mlp.up_proj.weight"), &[lay.inter, hid])?;
        let down = get(&format!("{p}mlp.down_proj.weight"), &[hid, lay.inter])?;
        let ln2 = get(&format!("{p}post_attention_layernorm.weight"), &[hid])?;
        let mut gu = vec![0f32; hm * pw * ht * 2 * it];
        let mut dn = vec![0f32; hm * pw * it * ht];
        let mut gamma2 = vec![0f32; hm * pw * ht];
        for y in 0..hm {
            for px in 0..pw {
                let pe = y * pw + px;
                for c in 0..ht {
                    let col = px * ht + c;
                    for i in 0..it {
                        let row = y * it + i;
                        gu[(pe * ht + c) * 2 * it + i] = gate.data[row * hid + col];
                        gu[(pe * ht + c) * 2 * it + it + i] = up.data[row * hid + col];
                        dn[(pe * it + i) * ht + c] = down.data[col * lay.inter + row];
                    }
                }
                gamma2[pe * ht..(pe + 1) * ht].copy_from_slice(&ln2.data[px * ht..(px + 1) * ht]);
            }
        }
        for (sym, arr, n) in [
            ("gu", &gu, 2 * it * ht),
            ("down", &dn, ht * it),
            ("gamma2", &gamma2, ht),
        ] {
            put_strip(lay, sink, sym, arr, strip, base + ph, hm, n)?;
        }
        tracing::info!(
            layer = li,
            ms = started.elapsed().as_millis() as u64,
            "spatial wafer: layer placed"
        );
    }

    // The head: the (tied) embedding as the vocabulary's rows, padded.
    let emb = ckpt.get("model.embed_tokens.weight")?;
    if emb.shape.len() != 2 || emb.shape[1] != hid {
        return Err(format!("model.embed_tokens.weight is {:?}", emb.shape));
    }
    let rows_have = emb.shape[0].min(lay.vocab);
    let (vr, hh) = (lay.vr, lay.hh);
    let mut y0 = 0;
    while y0 < hh {
        let rows = EMB_CHUNK_ROWS.min(hh - y0);
        let mut chunk = vec![0f32; rows * pw * ht * vr];
        for yl in 0..rows {
            for px in 0..pw {
                let pe = yl * pw + px;
                for v in 0..vr {
                    let tok = (y0 + yl) * vr + v;
                    if tok >= rows_have {
                        continue;
                    }
                    let src = &emb.data[tok * hid + px * ht..tok * hid + (px + 1) * ht];
                    for (c, &x) in src.iter().enumerate() {
                        chunk[(pe * ht + c) * vr + v] = x;
                    }
                }
            }
        }
        put_strip(lay, sink, "emb", &chunk, lay.ns, y0, rows, vr * ht)?;
        y0 += rows;
    }
    drop(emb);
    let norm = get("model.norm.weight", &[hid])?;
    let hgamma: Vec<f32> = (0..hh * pw)
        .flat_map(|pe| norm.data[(pe % pw) * ht..(pe % pw + 1) * ht].iter().copied())
        .collect();
    put_strip(lay, sink, "hgamma", &hgamma, lay.ns, 0, hh, ht)
}

/// The program's sources, shipped with the engine: serve28s (serve28 with
/// shared program images: PEs derive their place, colors and routes from
/// their coordinates in an `init` launch, so the wafer takes a few hundred
/// ELF files instead of thousands).
const SOURCES: [(&str, &str); 8] = [
    ("layout.csl", include_str!("serve28s3/layout.csl")),
    ("attn_pe.csl", include_str!("serve28s3/attn_pe.csl")),
    ("mlp_pe.csl", include_str!("serve28s3/mlp_pe.csl")),
    ("head_pe.csl", include_str!("serve28s3/head_pe.csl")),
    ("relay_pe.csl", include_str!("serve28s3/relay_pe.csl")),
    ("filler_pe.csl", include_str!("serve28s3/filler_pe.csl")),
    ("colors.csl", include_str!("serve28s3/colors.csl")),
    ("rt.csl", include_str!("serve28s3/rt.csl")),
];

/// The serve28s program for `params` (`name:value,...`, `kernel_cols`
/// included), from PIE's program cache (`PIE_CEREBRAS_CACHE`) or compiled
/// into it: keyed by the sources, the parameters and the fabric, finished
/// when the RPC manifest and a PE binary are there, as the phase programs'.
/// A system's program is laid out on the whole wafer.
pub fn compile(params: &str, target: crate::sdk::Target, system: bool) -> R<PathBuf> {
    use crate::sdk::{Arch, Compile, Cslc};
    let lay = Layout::parse(params)?;
    let fabric_dims = if system {
        crate::device::wafer_dims(target)
    } else {
        ((lay.width + 7) as u32, (lay.height + 2) as u32)
    };
    let fabric_offsets = (4, 1);
    let mut key = blake3::Hasher::new();
    for (name, text) in SOURCES {
        key.update(name.as_bytes()).update(&[0]).update(text.as_bytes()).update(&[0]);
    }
    key.update(format!("{params}|{fabric_dims:?}|{fabric_offsets:?}|{target:?}|{system}").as_bytes());
    let hex = key.finalize().to_hex().to_string();
    let target_name = match target {
        crate::sdk::Target::Wse2 => "wse2",
        crate::sdk::Target::Wse3 => "wse3",
    };
    let dir = crate::device::disk_cache().join(format!(
        "{hex}-{target_name}{}",
        if system { "-system" } else { "" }
    ));
    let out = dir.join("out");
    let bin = out.join("bin");
    if bin.join("out_rpc.json").is_file() && bin.join("out_0_0.elf").is_file() {
        tracing::info!(dir = %out.display(), "spatial wafer: program cached");
        return Ok(out);
    }
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for (name, text) in SOURCES {
        std::fs::write(dir.join(name), text).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let compile = Compile {
        arch: Arch::from(target),
        layout: dir.join("layout.csl"),
        fabric_dims,
        fabric_offsets,
        params: params
            .split(',')
            .filter_map(|kv| kv.split_once(':'))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .collect(),
        memcpy: true,
        channels: 1,
        import_paths: Vec::new(),
        out: out.clone(),
    };
    let started = Instant::now();
    tracing::info!(dir = %out.display(), "spatial wafer: compiling the program");
    if let Err(e) = Cslc::find().and_then(|cslc| cslc.compile(&compile)) {
        let _ = std::fs::remove_dir_all(&out);
        return Err(format!("compiling the serve28s program: {e}"));
    }
    tracing::info!(
        s = started.elapsed().as_secs_f32(),
        "spatial wafer: program compiled"
    );
    Ok(out)
}

/// Which logits a forward hands back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Logits {
    None,
    Last,
    All,
}

/// The program on the fabric, weights placed.
pub struct WaferModel {
    rt: Runtime,
    design: Design,
    launch: Launch,
    /// Launches and layer steps run so far (the warm-up's included).
    launches: u64,
    steps: u64,
    /// Batched launches' lane-passes (tokens through the model).
    lane_steps: u64,
    /// The head's eta VM, when the build has one.
    vm: Option<Vm>,
}

/// The head VM's state on the host: its shape and ring, the `gmode` the
/// head PEs hold, and the steps run since the loaded program's homes were
/// seeded (the ring's position).
struct Vm {
    lay: super::guest4::wafer::VmLayout,
    ring: super::guest4::lane::Ring,
    gmode: u32,
    steps: u64,
    loaded: bool,
}

/// One VM step's answer: the token the VM sampled (the head embedded it),
/// the step's `out` words, the pool after it.
pub type VmStep = (u32, Vec<u32>, Vec<u32>);

/// How a request reaches the program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Launch {
    /// serve28s3: `in_tokens` and `compute(steps, start_pos)` queued without
    /// waiting; only reading the answer blocks.
    Args,
    /// serve28, serve28s: `run_cfg` written on every PE, then `compute()`.
    RunCfg,
}

impl Launch {
    /// `"args"`, `"run_cfg"`; anything else (or nothing) is decided from
    /// the program.
    pub fn named(name: Option<&str>) -> Option<Launch> {
        match name {
            Some("args") => Some(Launch::Args),
            Some("run_cfg") => Some(Launch::RunCfg),
            _ => None,
        }
    }

    /// From the RPC manifest: a `compute` that takes arguments.
    fn of(rt: &Runtime) -> Launch {
        if rt.rpc().get(SYM.compute).is_ok_and(|s| s.inputs >= 2) {
            Launch::Args
        } else {
            Launch::RunCfg
        }
    }
}

fn rect(x: usize, y: usize, w: usize, h: usize) -> Rect {
    Rect {
        x: x as i32,
        y: y as i32,
        w: w as i32,
        h: h as i32,
    }
}

impl WaferModel {
    /// [`WaferModel::open_with`], RoPE's theta from the checkpoint's config.
    pub fn open(
        program: &Path,
        ckpt_dir: &Path,
        platform: Platform,
        launch: Option<Launch>,
    ) -> R<WaferModel> {
        WaferModel::open_with(program, ckpt_dir, platform, launch, None)
    }

    /// Loads `program` and places `ckpt_dir`'s weights; RoPE's theta is
    /// `rope_theta` (the model's, from its trace), else the checkpoint
    /// config's (else 1e6).
    pub fn open_with(
        program: &Path,
        ckpt_dir: &Path,
        platform: Platform,
        launch: Option<Launch>,
        rope_theta: Option<f64>,
    ) -> R<WaferModel> {
        let design = Design::read(program)?;
        let ckpt = Ckpt::open(ckpt_dir)?;
        if !ckpt.tied_head()? {
            return Err(format!(
                "{}: an untied LM head (lm_head.weight apart from embed_tokens); the wafer \
                 programs compute the logits from the embedding and cannot place lm_head yet",
                ckpt_dir.display()
            ));
        }
        let config_theta = ckpt.config()["rope_theta"].as_f64();
        if let (Some(theta), Some(config)) = (rope_theta, config_theta)
            && (theta - config).abs() > 1e-6 * config.abs()
        {
            tracing::warn!(theta, config, "spatial wafer: the checkpoint config's rope_theta differs from the model's");
        }
        let rope_theta = rope_theta.or(config_theta).unwrap_or(1.0e6);
        let sdk = Sdk::open().map_err(|e| e.to_string())?;
        let started = Instant::now();
        let mut rt = Runtime::new(sdk, program, platform, "WARNING").map_err(|e| e.to_string())?;
        rt.load();
        rt.run();
        // serve28s: every PE takes its place, queue colors and routes before
        // any traffic (a serve28 program has no `init`).
        if rt.rpc().get("init").is_ok() {
            rt.launch("init", &[]).map_err(|e| e.to_string())?;
        }
        tracing::info!(
            s = started.elapsed().as_secs_f32(),
            grid = matches!(design, Design::Grid(_)),
            "spatial wafer: program loaded"
        );
        let started = Instant::now();
        match &design {
            Design::Strips(lay) => place(lay, &ckpt, rope_theta, &mut rt)?,
            Design::Grid(grid) => super::serve2d::place(grid, &ckpt, rope_theta, &mut rt)?,
        }
        tracing::info!(
            s = started.elapsed().as_secs_f32(),
            "spatial wafer: weights placed"
        );
        let launch = launch.unwrap_or_else(|| Launch::of(&rt));
        let mut model = WaferModel {
            rt,
            design,
            launch,
            launches: 0,
            lane_steps: 0,
            vm: None,
            steps: 0,
        };
        if let Design::Grid(grid) = &model.design
            && let Some(lay) = grid.vm
        {
            let words = super::guest4::lane::ring_words(program)?;
            let ring = super::guest4::lane::read_ring(&mut model.rt, lay.hub(), words)?;
            super::guest4::wafer::set_gmode(&mut model.rt, &lay, 0)?;
            tracing::info!(?ring, ops = lay.max_ops, registers = lay.nreg, pool = lay.pool_words, "spatial wafer: the head has the eta VM");
            model.vm = Some(Vm {
                lay,
                ring,
                gmode: 0,
                steps: 0,
                loaded: false,
            });
        }
        // One request (position 0; later requests overwrite its KV) so the
        // first real one does not pay first-launch costs.
        let started = Instant::now();
        let paged = match &model.design {
            Design::Grid(g) => g.paged,
            Design::Strips(_) => None,
        };
        if paged.is_some() {
            // Every lane one step at slot 0 of pool page 0 (later fires write
            // over it), as serve_lib2d's warm-up.
            let warm = super::serve2d::BatchLane {
                pos: 0,
                slot: 0,
                pages: vec![0],
                visible: None,
                forced: vec![0],
            };
            model.fire_batch(&[warm], 1, false)?;
        } else {
            model.run(&[0], 1, 0, false)?;
        }
        tracing::info!(
            ?launch,
            s = started.elapsed().as_secs_f32(),
            "spatial wafer: warm-up request"
        );
        Ok(model)
    }

    pub fn design(&self) -> &Design {
        &self.design
    }

    /// Why the head's VM cannot take `table` (no VM, or a table past its
    /// limits or its ring's width), if it cannot.
    pub fn vm_refuses(&self, table: &super::guest4::Table) -> Option<String> {
        let Some(vm) = &self.vm else {
            return Some("the wafer program has no VM".into());
        };
        if !table.per_step.is_empty() {
            return Some("the host writes some of its homes every step".into());
        }
        if vm.ring.lanes != 1 {
            return Some(format!("the VM time-multiplexes {} lanes; this path drives one", vm.ring.lanes));
        }
        vm.lay.refuses(table).or_else(|| vm.ring.refuses_outputs(&table.outputs))
    }

    /// Loads `table` on the head's VM, its homes seeded with `seeds` (a
    /// lane's carried state): the VM's ring starts over.
    pub fn vm_load(&mut self, table: &super::guest4::Table, seeds: &[super::guest4::wafer::Seed]) -> R<()> {
        if let Some(why) = self.vm_refuses(table) {
            return Err(why);
        }
        let vm = self.vm.as_mut().ok_or("the wafer program has no VM")?;
        super::guest4::wafer::load(&mut self.rt, &vm.lay, table, seeds)?;
        vm.gmode = 1;
        vm.steps = 0;
        vm.loaded = true;
        Ok(())
    }

    /// Loads `table` on a multi-lane VM (`gring` lanes L, serve2d_pipe's
    /// lanes m·B + b): the program once for all lanes, every bank zeroed
    /// (each lane is seeded when a request takes it), the ring starting over.
    pub fn vm_load_lanes(&mut self, table: &super::guest4::Table) -> R<()> {
        let vm = self.vm.as_mut().ok_or("the wafer program has no VM")?;
        if let Some(why) = vm.lay.refuses(table).or_else(|| vm.ring.refuses_outputs(&table.outputs)) {
            return Err(why);
        }
        super::guest4::wafer::load(&mut self.rt, &vm.lay, table, &[])?;
        let lanes = vm.ring.lanes as usize;
        let empty: Vec<Vec<super::guest4::wafer::Seed>> = (0..lanes).map(|_| Vec::new()).collect();
        super::guest4::wafer::write_banks(&mut self.rt, &vm.lay, &empty)?;
        vm.gmode = 1;
        vm.steps = 0;
        vm.loaded = true;
        Ok(())
    }

    /// Lane `lane`'s bank seeded with `seeds` (its request's carried state),
    /// the other lanes' banks as they stand.
    pub fn vm_seed_bank(&mut self, lane: usize, seeds: &[super::guest4::wafer::Seed]) -> R<()> {
        use super::guest4::wafer;
        let vm = self.vm.as_ref().ok_or("the wafer program has no VM")?;
        let lanes = vm.ring.lanes as usize;
        if lane >= lanes {
            return Err(format!("lane {lane} of a VM of {lanes}"));
        }
        let mut images = wafer::read_banks(&mut self.rt, &vm.lay, lanes)?;
        // The lane's bank alone, laid out as a VM of one lane, spread into
        // its slot.
        let one = wafer::bank_images(&vm.lay, &[seeds.iter().map(|s| wafer::Seed { words: s.words.clone(), ..*s }).collect()])?;
        let bank = vm.lay.pool_words + (vm.lay.nreg - 1) * vm.lay.vr;
        for (src, dst) in one.iter().zip(images.iter_mut()) {
            for pe in 0..=vm.lay.rows {
                let at = (pe * lanes + lane) * bank;
                dst[at..at + bank].copy_from_slice(&src[pe * bank..(pe + 1) * bank]);
            }
        }
        wafer::write_bank_images(&mut self.rt, &vm.lay, &images)
    }

    /// One lap of a multi-lane VM launch: `lanes` (one per VM lane, in lane
    /// order m·B + b; idle lanes repeat a live one) each forced its token,
    /// the VM sampling every lane's next one from its bank. The banks of
    /// `keep` (lanes whose requests sat this launch out) come back as they
    /// were: an idle lane's turn runs on its bank. Per lane: the token the VM
    /// sampled, the turn's `out` words, its pool after it.
    pub fn vm_fire(&mut self, lanes: &[super::serve2d::BatchLane], keep: &[usize]) -> R<Vec<VmStep>> {
        use super::guest4::wafer;
        let Design::Grid(grid) = &self.design else {
            return Err("a VM launch on a strip program".into());
        };
        let vm = self.vm.as_mut().ok_or("the wafer program has no VM")?;
        let n = vm.ring.lanes as usize;
        if !vm.loaded || lanes.len() != n || grid.launch_lanes() != n {
            return Err(format!(
                "a VM launch of {} lanes on a VM of {n} ({} lanes a launch, program loaded: {})",
                lanes.len(),
                grid.launch_lanes(),
                vm.loaded
            ));
        }
        if vm.gmode != 1 {
            wafer::set_gmode(&mut self.rt, &vm.lay, 1)?;
            vm.gmode = 1;
        }
        let saved = if keep.is_empty() { None } else { Some(wafer::read_banks(&mut self.rt, &vm.lay, n)?) };
        self.launches += 1;
        self.steps += 1;
        self.lane_steps += n as u64;
        let sampled = vec![1u32; n];
        let (tokens, _) = super::serve2d::fire_lanes(grid, &mut self.rt, lanes, 1, false, Some(&sampled))?;
        let rows = wafer::read_out_lanes(&mut self.rt, &vm.lay, vm.ring, vm.steps)?;
        vm.steps += 1;
        let pools = wafer::read_bank_pools(&mut self.rt, &vm.lay, n)?;
        if let Some(saved) = saved {
            let mut now = wafer::read_banks(&mut self.rt, &vm.lay, n)?;
            for &lane in keep {
                wafer::copy_bank(&vm.lay, n, lane, &saved, &mut now);
            }
            wafer::write_bank_images(&mut self.rt, &vm.lay, &now)?;
        }
        Ok(tokens
            .into_iter()
            .zip(rows)
            .zip(pools)
            .map(|((t, row), pool)| (t.first().copied().unwrap_or_default(), row, pool))
            .collect())
    }

    /// The last launch's logits rows of the first HLB lanes (y x 1/rms in
    /// f32: on a VM build, what the VM sampled from).
    pub fn lane_planes(&mut self) -> R<Vec<Vec<f32>>> {
        let Design::Grid(grid) = &self.design else {
            return Err("lane planes on a strip program".into());
        };
        super::serve2d::logits_rows(grid, &mut self.rt, grid.logit_lanes())
    }

    /// The logits row the VM sampled its last step from
    /// ([`super::guest4::wafer::read_plane`]): for checks.
    pub fn vm_plane(&mut self) -> R<Vec<f32>> {
        let vm = self.vm.as_ref().ok_or("the wafer program has no VM")?;
        super::guest4::wafer::read_plane(&mut self.rt, &vm.lay)
    }

    /// `token` at position `start` with the VM picking the next one: one
    /// launch of one iteration with `gmode` 1 (the VM runs the loaded
    /// program on that row's logits and the head embeds its token), then
    /// the step's `out` words and the VM's pool.
    pub fn vm_step(&mut self, token: u32, start: usize) -> R<VmStep> {
        let Design::Grid(grid) = &self.design else {
            return Err("a VM step on a strip program".into());
        };
        let vm = self.vm.as_mut().ok_or("the wafer program has no VM")?;
        if !vm.loaded {
            return Err("no program is loaded on the VM".into());
        }
        if vm.gmode != 1 {
            super::guest4::wafer::set_gmode(&mut self.rt, &vm.lay, 1)?;
            vm.gmode = 1;
        }
        self.launches += 1;
        self.steps += 1;
        let (sampled, _) = super::serve2d::run(grid, &mut self.rt, &[token], 1, start, false)?;
        let mut rows = super::guest4::wafer::read_out(&mut self.rt, &vm.lay, vm.ring, vm.steps, 1)?;
        vm.steps += 1;
        let pool = super::guest4::wafer::read_pool(&mut self.rt, &vm.lay)?;
        let token = sampled.first().copied().ok_or("the launch answered no token")?;
        Ok((token, rows.pop().unwrap_or_default(), pool))
    }

    /// A batched build's launch ([`super::serve2d::fire`]).
    pub fn fire_batch(
        &mut self,
        lanes: &[super::serve2d::BatchLane],
        steps: usize,
        want_logits: bool,
    ) -> R<(Vec<Vec<u32>>, Vec<Vec<f32>>)> {
        let Design::Grid(grid) = &self.design else {
            return Err("a batched launch on a strip program".into());
        };
        // The head's own argmax: the VM idles.
        if let Some(vm) = self.vm.as_mut()
            && vm.gmode != 0
        {
            super::guest4::wafer::set_gmode(&mut self.rt, &vm.lay, 0)?;
            vm.gmode = 0;
        }
        self.launches += 1;
        self.steps += steps as u64;
        self.lane_steps += (lanes.len() * steps) as u64;
        if self.launches % 256 == 0 {
            tracing::info!(
                launches = self.launches,
                steps = self.steps,
                lane_steps = self.lane_steps,
                lanes = lanes.len(),
                "spatial wafer: batch totals"
            );
        }
        super::serve2d::fire(grid, &mut self.rt, lanes, steps, want_logits)
    }

    /// A batched build's whole-page copies (every layer's pool).
    pub fn copy_pages(&mut self, pairs: &[(u32, u32)]) -> R<()> {
        let Design::Grid(grid) = &self.design else {
            return Err("page copies on a strip program".into());
        };
        super::serve2d::copy_pages(grid, &mut self.rt, pairs)
    }

    /// One launch: head iterations 0..=steps, iteration i forcing forced[i]
    /// (past its end, the argmax). The argmax after each step; the last
    /// step's logits when asked.
    fn run(
        &mut self,
        forced: &[u32],
        steps: usize,
        start: usize,
        want_logits: bool,
    ) -> R<(Vec<u32>, Option<Vec<f32>>)> {
        self.launches += 1;
        self.steps += steps as u64;
        let lay = match &self.design {
            Design::Strips(lay) => lay,
            Design::Grid(grid) => {
                // The head's own argmax: the VM idles.
                if let Some(vm) = self.vm.as_mut()
                    && vm.gmode != 0
                {
                    super::guest4::wafer::set_gmode(&mut self.rt, &vm.lay, 0)?;
                    vm.gmode = 0;
                }
                return super::serve2d::run(grid, &mut self.rt, forced, steps, start, want_logits);
            }
        };
        if steps + 1 > lay.cap || start + steps > lay.positions || forced.len() > lay.cap {
            return Err(format!(
                "a launch of {steps} steps at position {start} does not fit the program \
                 ({} iterations, {} positions)",
                lay.cap, lay.positions
            ));
        }
        let mut inn = vec![SAMPLE; lay.cap];
        inn[..forced.len()].copy_from_slice(forced);
        let (hx, hy) = lay.hbe();
        let one = rect(hx, hy, 1, 1);
        let (cap, hh, vr, vocab, hflip, y0) =
            (lay.cap, lay.hh, lay.vr, lay.vocab, lay.hflip(), lay.hcol_y0());
        let e = |e: crate::sdk::Error| e.to_string();
        let rt = &mut self.rt;
        if self.launch == Launch::Args {
            // Queued in order; the answer's read waits for both.
            rt.memcpy_h2d_nonblock(SYM.in_tokens, inn, one, cap, Order::RowMajor, DataType::Bits32)
                .map_err(e)?;
            rt.launch_nonblock(SYM.compute, &[steps as u32, start as u32])
                .map_err(e)?;
        } else {
            let cfg: Vec<u32> =
                std::iter::repeat_n([steps as u32, start as u32], lay.width * lay.height)
                    .flatten()
                    .collect();
            let whole = rect(0, 0, lay.width, lay.height);
            rt.memcpy_h2d("run_cfg", &cfg, whole, 2, Order::RowMajor, DataType::Bits32)
                .map_err(e)?;
            rt.memcpy_h2d(SYM.in_tokens, &inn, one, cap, Order::RowMajor, DataType::Bits32)
                .map_err(e)?;
            rt.launch(SYM.compute, &[]).map_err(e)?;
        }
        let mut out = vec![0u32; cap];
        let read = rt.memcpy_d2h(SYM.out_tokens, &mut out, one, cap, Order::RowMajor, DataType::Bits32);
        rt.wait_pending();
        read.map_err(e)?;
        // The head writes each argmax as an f32 value.
        let argmax = out[..steps]
            .iter()
            .map(|&w| f32::from_bits(w) as u32)
            .collect();
        let logits = if want_logits {
            let mut col = vec![0u32; hh * vr];
            rt.memcpy_d2h(
                SYM.logit_lanes,
                &mut col,
                rect(hx, y0, 1, hh),
                vr,
                Order::RowMajor,
                DataType::Bits32,
            )
            .map_err(e)?;
            // Physical rows ascending are logical rows descending.
            let rows: Vec<&[u32]> = if hflip {
                col.chunks_exact(vr).rev().collect()
            } else {
                col.chunks_exact(vr).collect()
            };
            Some(
                rows.into_iter()
                    .flatten()
                    .take(vocab)
                    .map(|&w| f32::from_bits(w))
                    .collect(),
            )
        } else {
            None
        };
        Ok((argmax, logits))
    }

    /// The wafer's token loop: `tokens` at positions `start..`, then `extra`
    /// more steps each fed the step before's argmax, in one launch where it
    /// fits (a long feed goes first in pieces). The argmax after every step:
    /// `tokens.len() + extra` of them, the last `extra + 1` the greedy
    /// continuation. Nothing but the argmaxes comes back.
    pub fn greedy(&mut self, tokens: &[u32], start: usize, extra: usize) -> R<Vec<u32>> {
        let piece = self.design.cap() - 1;
        let room = self.design.positions().saturating_sub(start + tokens.len());
        let extra = extra.min(piece.saturating_sub(1)).min(room);
        // The feed's head in pieces that leave the tail and the loop one launch.
        let tail = tokens.len().min(piece - extra).max(1);
        let head = tokens.len() - tail;
        let mut argmax = Vec::with_capacity(tokens.len() + extra);
        if head > 0 {
            let (a, _) = self.forward(&tokens[..head], start, Logits::None)?;
            argmax.extend(a);
        }
        let (a, _) = self.run(&tokens[head..], tail + extra, start + head, false)?;
        argmax.extend(a);
        Ok(argmax)
    }

    /// The tokens at positions `start..`: the argmax after each, and the
    /// logits `which` asks for. A launch takes at most `iters - 1` tokens;
    /// longer runs go in pieces (the cache carries on between them), and
    /// every row's logits take a launch a token.
    pub fn forward(
        &mut self,
        tokens: &[u32],
        start: usize,
        which: Logits,
    ) -> R<(Vec<u32>, Vec<Vec<f32>>)> {
        let mut argmax = Vec::with_capacity(tokens.len());
        let mut logits = Vec::new();
        if which == Logits::All {
            for (i, &tok) in tokens.iter().enumerate() {
                let (a, l) = self.run(&[tok], 1, start + i, true)?;
                argmax.extend(a);
                logits.extend(l);
            }
            return Ok((argmax, logits));
        }
        let piece = self.design.cap() - 1;
        let pieces = tokens.len().div_ceil(piece);
        for (i, chunk) in tokens.chunks(piece).enumerate() {
            let want = which == Logits::Last && i + 1 == pieces;
            let (a, l) = self.run(chunk, chunk.len(), start + i * piece, want)?;
            argmax.extend(a);
            logits.extend(l);
        }
        Ok((argmax, logits))
    }
}

enum Job {
    Forward {
        tokens: Vec<u32>,
        start: usize,
        which: Logits,
    },
    Greedy {
        tokens: Vec<u32>,
        start: usize,
        extra: usize,
    },
    /// A batched build's launch (answered on the batch channel).
    Batch {
        lanes: Vec<super::serve2d::BatchLane>,
        steps: usize,
        logits: bool,
    },
    /// A batched build's whole-page copies (answered on the batch channel).
    Copy { pairs: Vec<(u32, u32)> },
    /// A program onto the head's VM, and one VM step (the VM channel).
    VmLoad {
        program: std::sync::Arc<super::guest4::lane::Program>,
        seeds: Vec<super::guest4::wafer::Seed>,
    },
    VmStep { token: u32, start: usize },
    /// A multi-lane VM: a program for every lane, one lane's bank seeded,
    /// one lap of a launch.
    VmLoadLanes {
        program: std::sync::Arc<super::guest4::lane::Program>,
    },
    VmSeedBank {
        lane: usize,
        seeds: Vec<super::guest4::wafer::Seed>,
    },
    VmFire {
        lanes: Vec<super::serve2d::BatchLane>,
        keep: Vec<usize>,
    },
}

type Answer = R<(Vec<u32>, Vec<Vec<f32>>)>;
/// Each lane's tokens and logits rows.
type BatchAnswer = R<(Vec<Vec<u32>>, Vec<Vec<f32>>)>;

/// A [`WaferModel`] on its own thread.
pub struct Wafer {
    jobs: Option<mpsc::Sender<Job>>,
    answers: mpsc::Receiver<Answer>,
    batch_answers: mpsc::Receiver<BatchAnswer>,
    /// The batched build's paging (`None`: one sequence), and its
    /// iterations a launch.
    pub paged: Option<super::paged::Shape>,
    pub cap: usize,
    /// Lanes a batched launch carries (MMAX micro-batches of B on a ring),
    /// and how many of them may want logits.
    pub launch_lanes: usize,
    pub logit_lanes: usize,
    /// The KV positions one sequence may hold (a lane's pages on a batched
    /// build).
    pub positions: usize,
    /// The head's eta VM (its shape and out ring), when the build has one.
    pub vm: Option<(super::guest4::wafer::VmLayout, super::guest4::lane::Ring)>,
    vm_answers: mpsc::Receiver<R<Vec<VmStep>>>,
    thread: Option<std::thread::JoinHandle<()>>,
    pub vocab: usize,
    pub hidden: usize,
}

impl Wafer {
    /// Loads `program` (the compiler's output directory) on `platform` and
    /// places `ckpt` (RoPE's theta `rope_theta`, else the checkpoint
    /// config's); returns once the model answers.
    pub fn open(
        program: PathBuf,
        ckpt: PathBuf,
        platform: Platform,
        launch: Option<Launch>,
        rope_theta: Option<f64>,
    ) -> R<Wafer> {
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (ans_tx, ans_rx) = mpsc::channel::<Answer>();
        let (batch_tx, batch_rx) = mpsc::channel::<BatchAnswer>();
        let (vm_tx, vm_rx) = mpsc::channel::<R<Vec<VmStep>>>();
        let (ready_tx, ready_rx) = mpsc::channel::<R<Wafer>>();
        let thread = std::thread::Builder::new()
            .name("spatial-wafer".into())
            .spawn(move || {
                let mut model = match WaferModel::open_with(&program, &ckpt, platform, launch, rope_theta) {
                    Ok(m) => {
                        let (paged, (launch_lanes, logit_lanes)) = match &m.design {
                            Design::Grid(g) => (g.paged, (g.launch_lanes(), g.logit_lanes())),
                            Design::Strips(_) => (None, (1, 1)),
                        };
                        // The facts the engine reads; the channels are filled in below.
                        let _ = ready_tx.send(Ok(Wafer {
                            jobs: None,
                            answers: mpsc::channel().1,
                            batch_answers: mpsc::channel().1,
                            vm_answers: mpsc::channel().1,
                            paged,
                            cap: m.design.cap(),
                            launch_lanes,
                            logit_lanes,
                            positions: m.design.positions(),
                            vm: m.vm.as_ref().map(|vm| (vm.lay, vm.ring)),
                            thread: None,
                            vocab: m.design.vocab(),
                            hidden: m.design.hidden(),
                        }));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                for job in job_rx {
                    let answer = match job {
                        Job::Batch {
                            lanes,
                            steps,
                            logits,
                        } => {
                            let reply = model.fire_batch(&lanes, steps, logits);
                            if batch_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::VmLoad { program, seeds } => {
                            let reply = model.vm_load(&program.table, &seeds).map(|()| Vec::new());
                            if vm_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::VmLoadLanes { program } => {
                            let reply = model.vm_load_lanes(&program.table).map(|()| Vec::new());
                            if vm_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::VmSeedBank { lane, seeds } => {
                            let reply = model.vm_seed_bank(lane, &seeds).map(|()| Vec::new());
                            if vm_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::VmFire { lanes, keep } => {
                            let reply = model.vm_fire(&lanes, &keep);
                            if vm_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::VmStep { token, start } => {
                            let reply = model.vm_step(token, start).map(|step| vec![step]);
                            if vm_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::Copy { pairs } => {
                            let reply = model.copy_pages(&pairs).map(|()| (Vec::new(), Vec::new()));
                            if batch_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::Forward {
                            tokens,
                            start,
                            which,
                        } => model.forward(&tokens, start, which),
                        Job::Greedy {
                            tokens,
                            start,
                            extra,
                        } => model.greedy(&tokens, start, extra).map(|a| (a, Vec::new())),
                    };
                    tracing::info!(
                        launches = model.launches,
                        steps = model.steps,
                        "spatial wafer: totals"
                    );
                    if ans_tx.send(answer).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| format!("spawning the wafer thread: {e}"))?;
        let facts = ready_rx
            .recv()
            .map_err(|_| "the wafer thread died while loading".to_string())??;
        Ok(Wafer {
            jobs: Some(job_tx),
            answers: ans_rx,
            batch_answers: batch_rx,
            vm_answers: vm_rx,
            thread: Some(thread),
            ..facts
        })
    }

    /// [`WaferModel::fire_batch`] on the wafer's thread.
    pub fn fire_batch(
        &mut self,
        lanes: Vec<super::serve2d::BatchLane>,
        steps: usize,
        logits: bool,
    ) -> BatchAnswer {
        let _busy = super::Busy::start();
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::Batch {
                lanes,
                steps,
                logits,
            })
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.batch_answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
    }

    /// Why the head's VM cannot take `table`, if it cannot (as
    /// [`WaferModel::vm_refuses`]).
    pub fn vm_refuses(&self, table: &super::guest4::Table) -> Option<String> {
        let Some((lay, ring)) = &self.vm else {
            return Some("the wafer program has no VM".into());
        };
        if !table.per_step.is_empty() {
            return Some("the host writes some of its homes every step".into());
        }
        if ring.lanes != 1 {
            return Some(format!("the VM time-multiplexes {} lanes; this path drives one", ring.lanes));
        }
        lay.refuses(table).or_else(|| ring.refuses_outputs(&table.outputs))
    }

    /// [`WaferModel::vm_load`] on the wafer's thread.
    pub fn vm_load(
        &mut self,
        program: std::sync::Arc<super::guest4::lane::Program>,
        seeds: Vec<super::guest4::wafer::Seed>,
    ) -> R<()> {
        let _busy = super::Busy::start();
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::VmLoad { program, seeds })
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.vm_answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
            .map(drop)
    }

    /// [`WaferModel::vm_load_lanes`] on the wafer's thread.
    pub fn vm_load_lanes(&mut self, program: std::sync::Arc<super::guest4::lane::Program>) -> R<()> {
        let _busy = super::Busy::start();
        self.vm_job(Job::VmLoadLanes { program }).map(drop)
    }

    /// [`WaferModel::vm_seed_bank`] on the wafer's thread.
    pub fn vm_seed_bank(&mut self, lane: usize, seeds: Vec<super::guest4::wafer::Seed>) -> R<()> {
        let _busy = super::Busy::start();
        self.vm_job(Job::VmSeedBank { lane, seeds }).map(drop)
    }

    /// [`WaferModel::vm_fire`] on the wafer's thread.
    pub fn vm_fire(&mut self, lanes: Vec<super::serve2d::BatchLane>, keep: Vec<usize>) -> R<Vec<VmStep>> {
        let _busy = super::Busy::start();
        self.vm_job(Job::VmFire { lanes, keep })
    }

    fn vm_job(&mut self, job: Job) -> R<Vec<VmStep>> {
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(job)
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.vm_answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
    }

    /// [`WaferModel::vm_step`] on the wafer's thread.
    pub fn vm_step(&mut self, token: u32, start: usize) -> R<VmStep> {
        let _busy = super::Busy::start();
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::VmStep { token, start })
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.vm_answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())??
            .pop()
            .ok_or_else(|| "the VM answered no step".to_string())
    }

    /// [`WaferModel::copy_pages`] on the wafer's thread.
    pub fn copy_pages(&mut self, pairs: Vec<(u32, u32)>) -> R<()> {
        let _busy = super::Busy::start();
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::Copy { pairs })
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.batch_answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
            .map(drop)
    }

    /// [`WaferModel::greedy`] on the wafer's thread.
    pub fn greedy(&mut self, tokens: &[u32], start: usize, extra: usize) -> R<Vec<u32>> {
        let _busy = super::Busy::start();
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::Greedy {
                tokens: tokens.to_vec(),
                start,
                extra,
            })
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
            .map(|(argmax, _)| argmax)
    }

    pub fn forward(&mut self, tokens: &[u32], start: usize, which: Logits) -> Answer {
        let _busy = super::Busy::start();
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::Forward {
                tokens: tokens.to_vec(),
                start,
                which,
            })
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
    }
}

impl Drop for Wafer {
    fn drop(&mut self) {
        // Closing the job channel ends the thread, which stops the program.
        self.jobs = None;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
