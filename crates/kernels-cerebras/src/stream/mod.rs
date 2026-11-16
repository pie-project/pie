//! The stream regime's shape: what a reading that runs whole sequences a step (a diffusion
//! transformer's denoise) asks of the wafer, read off its trace ([`shape`]), as [`crate::target`]
//! reads a decoder's [`crate::plan::Shape`].
//!
//! The reading's stream program ([`poem_compiler::spatial::layer::stream`]) says what runs where:
//! the row classes, the units each layer's sides take, the lane vectors computed once a step and
//! the host's permutations. The shape adds the params the kernels take: the joint attention's
//! heads, the axial rotation, the MLP, the norms and modulations. An op whose params no stream
//! kernel takes is refused by name; nothing is dropped. GENERIC.md §4 (M1) says what builds it.

use poem_compiler::spatial::layer::stream::{self as program, RowClass, StreamProgram};
use poem_ir::{
    Attn, Elemwise, Linear, ModulateForm, NormAxis, Operation, RaggedMask, RopeForm, Trace,
};
use serde::Serialize;

use crate::plan::Act;

pub mod build;
pub mod coresident;
pub mod gemm;
pub mod grid;
pub mod host;
pub mod mamba;
pub mod moe;
pub mod place;
pub mod tiled;
pub mod unit;

type R<T> = Result<T, String>;

/// The attention over a step's rows.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct JointSpec {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub sm_scale: f32,
    /// q and k RMS-normed per head before the rotation.
    pub qk_norm: bool,
    /// A row attends the rows before it alone (a prefill's mask): the step is one request.
    pub causal: bool,
}

/// A rotation of up to four position axes, each a run of a head's dims.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AxesRope {
    pub dims: [u32; 4],
    pub thetas: [f32; 4],
    /// Pairs `(2i, 2i + 1)`; else NeoX's halves.
    pub interleaved: bool,
}

/// What a stream reading asks of the wafer.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StreamShape {
    pub reading: String,
    pub hidden: u32,
    /// Each row class's name and rows a step.
    pub classes: Vec<(String, u32)>,
    pub attn: JointSpec,
    pub rope: Option<AxesRope>,
    /// The gated MLP's width and its gate's activation.
    pub intermediate: u32,
    pub act: Act,
    pub eps: f32,
    /// The adaLN modulation's forms the rows run.
    pub modulations: Vec<String>,
    /// The units the rows flow through (a side of each stage) and the layers' distinct bodies.
    pub units: usize,
    pub layer_bodies: usize,
    pub program: StreamProgram,
    /// A decode step's cache: the positions each K / V column keeps across steps (its rows
    /// stored at the step's position, `pos` at its PEs); `None`, a stream step's own rows alone.
    pub cache: Option<u32>,
}

/// The shape of `trace`'s `reading` over a step of `classes`' rows: its stream program, and the
/// params of what it runs. A node the program runs whose params no stream kernel takes is refused
/// by name.
pub fn shape(trace: &Trace, reading: &str, classes: &[RowClass]) -> R<StreamShape> {
    let name = &trace.name;
    let analysis = poem_compiler::analysis::analyze(trace).map_err(|e| format!("{name}: {e}"))?;
    let prog = program::lower(trace, &analysis, classes)?;
    let runs = |g: &poem_ir::Guard| classes.iter().any(|c| g.holds(c.word));
    let (mut attn, mut rope, mut glu, mut eps) = (None, None, None, None);
    let (mut qk_norm, mut modulations) = (false, Vec::new());
    for (at, node) in trace.nodes.iter().enumerate() {
        if !runs(&node.guard) {
            continue;
        }
        let refuse = |what: String| format!("{name}: node {at}: {what}");
        match &node.op {
            Operation::Attn(Attn::Ragged {
                mask,
                head_dim,
                kv_heads,
                sm_scale,
                q,
                ..
            }) => {
                if matches!(mask, RaggedMask::RelativeBias { .. }) {
                    return Err(refuse(
                        "attn.ragged under a relative bias: no stream attention adds one".into(),
                    ));
                }
                let width = width_of(trace, *q);
                let spec = JointSpec {
                    q_heads: width / (*head_dim).max(1),
                    kv_heads: *kv_heads,
                    head_dim: *head_dim,
                    sm_scale: *sm_scale,
                    qk_norm: false,
                    causal: false,
                };
                one_attention(&mut attn, spec).map_err(refuse)?;
            }
            Operation::Attn(Attn::Prefill {
                q,
                head_dim,
                kv_heads,
                sm_scale,
                causal,
                ..
            }) => {
                if *causal && classes.len() > 1 {
                    return Err(refuse(format!(
                        "attn.prefill, causal, over {} row classes: the stream's causal mask is \
                         one request's",
                        classes.len()
                    )));
                }
                let spec = JointSpec {
                    q_heads: width_of(trace, *q) / (*head_dim).max(1),
                    kv_heads: *kv_heads,
                    head_dim: *head_dim,
                    sm_scale: *sm_scale,
                    qk_norm: false,
                    causal: *causal,
                };
                one_attention(&mut attn, spec).map_err(refuse)?;
            }
            // a decode step's attention over its cache: one row a step, causal over the rows
            // stored before it
            Operation::Attn(Attn::Decode {
                q,
                plan,
                head_dim,
                sm_scale,
                ..
            }) => {
                let kv_heads =
                    match trace
                        .values
                        .get(plan.0 as usize)
                        .map(|d| &d.def)
                        .and_then(|d| match d {
                            poem_ir::Def::Op(n) => trace.nodes.get(*n as usize).map(|n| &n.op),
                            _ => None,
                        }) {
                        Some(Operation::Attn(Attn::PlanDecode { kv_heads, .. })) => *kv_heads,
                        _ => return Err(refuse("attn.decode: its plan is no decode plan".into())),
                    };
                let spec = JointSpec {
                    q_heads: width_of(trace, *q) / (*head_dim).max(1),
                    kv_heads,
                    head_dim: *head_dim,
                    sm_scale: *sm_scale,
                    qk_norm: false,
                    causal: true,
                };
                one_attention(&mut attn, spec).map_err(refuse)?;
            }
            // (the rotations print as their regions do, `fuse::chain::lower_run`: the shape
            // states the first, a form the tables turn, for the reading's description alone)
            Operation::Elemwise(Elemwise::RopeAxes {
                dims,
                thetas,
                form,
                rotary_dim,
                head_dim,
                ..
            }) if rope.is_none() && rotary_dim == head_dim => {
                if let RopeForm::Interleaved | RopeForm::Neox = form {
                    rope = Some(AxesRope {
                        dims: *dims,
                        thetas: *thetas,
                        interleaved: matches!(form, RopeForm::Interleaved),
                    });
                }
            }
            Operation::Elemwise(Elemwise::Rmsnorm { axis, eps: e, .. }) => {
                eps.get_or_insert(*e);
                qk_norm |= matches!(axis, NormAxis::Head(_));
            }
            Operation::Elemwise(Elemwise::Layernorm { eps: e, .. }) => {
                eps.get_or_insert(*e);
            }
            Operation::Elemwise(Elemwise::Modulate { form, .. }) => {
                let f = match form {
                    ModulateForm::ScaleShift => "scale_shift",
                    ModulateForm::Scale => "scale",
                    ModulateForm::TanhGate => "tanh_gate",
                };
                if !modulations.contains(&f.to_string()) {
                    modulations.push(f.to_string());
                }
            }
            // (each MLP prints as its region does: the shape states the first, for the
            // reading's description alone)
            Operation::Linear(Linear::Glu {
                intermediate, act, ..
            }) if glu.is_none() => {
                glu = Some((*intermediate, crate::target::act(act).map_err(refuse)?));
            }
            Operation::Linear(Linear::Matmul {
                input_scales,
                scales,
                quant,
                ..
            }) if input_scales.is_some() || scales.is_some() || quant.is_some() => {
                return Err(refuse(
                    "linear.matmul over quantized weights: the stream matvec reads 16-bit tiles"
                        .into(),
                ));
            }
            _ => {}
        }
    }
    // (a step of no attention: its blocks are its recurrences', each its own spec)
    let mut attn = attn.unwrap_or(JointSpec {
        q_heads: 1,
        kv_heads: 1,
        head_dim: 1,
        sm_scale: 1.0,
        qk_norm: false,
        causal: true,
    });
    attn.qk_norm = qk_norm;
    // (no gated MLP: the shape states none)
    let (intermediate, act) = glu.unwrap_or((0, Act::Silu));
    Ok(StreamShape {
        reading: reading.to_string(),
        hidden: prog.hidden,
        classes: prog
            .classes
            .iter()
            .map(|c| (c.name.clone(), c.rows))
            .collect(),
        attn,
        rope,
        intermediate,
        act,
        eps: eps.unwrap_or(1e-6),
        modulations,
        units: prog.units(),
        layer_bodies: prog.layer_bodies().len(),
        program: prog,
        cache: None,
    })
}

/// `v`'s row width.
fn width_of(trace: &Trace, v: poem_ir::ValueId) -> u32 {
    match trace.values.get(v.0 as usize).map(|d| &d.ty) {
        Some(poem_ir::Ty::Tensor { shape, .. }) => match shape.last() {
            Some(poem_ir::Dim::Const(n)) => *n as u32,
            _ => 0,
        },
        _ => 0,
    }
}

/// `spec`, the reading's first attention shape (each block's K / V columns its own block's:
/// the shape states the first, for the reading's description and its defaults).
fn one_attention(attn: &mut Option<JointSpec>, spec: JointSpec) -> R<()> {
    if attn.is_none() {
        *attn = Some(spec);
    }
    Ok(())
}

/// What [`compile`] builds: a reading at `rows` rows a class, its roundings the build's 16-bit
/// type's (`round16`, a bf16 build), the first `upto` stages only, `taps` also read back.
#[derive(Clone, Debug)]
pub struct Options {
    pub reading: String,
    pub rows: u32,
    pub round16: bool,
    pub upto: Option<usize>,
    pub taps: Vec<u32>,
}

/// What [`compile_pinned`] pins besides [`Options`] (the default: nothing).
#[derive(Clone, Debug, Default)]
pub struct Pins {
    /// Row roots a unit and a weight PE's most tile words (`None`: the cost model's choice,
    /// [`build::program_py`]; `py 16`, `tile 12288`: flux2-mini as M1d built it).
    pub py: Option<u32>,
    pub tile: Option<u32>,
    /// The K / V columns' cut (`None`: the cost model's choice of [`build::KV_SPLITS`]).
    pub kv: Option<build::KvSplit>,
    /// Every unit's rows held over the barrier in its row store, not at its roots (a test of
    /// the store at rows the roots would hold).
    pub spill: bool,
    /// Rows of the classes named here in place of `rows` (flux at 1024²: context 512, latents
    /// 4096).
    pub class_rows: Vec<(String, u32)>,
    /// The fabric's width for the program (`None`: the system's, [`Bounds::system`]; narrower
    /// fills more bands: a test of the row stream's turn into a band on a small model).
    ///
    /// [`Bounds::system`]: crate::floorplan::stream::Bounds::system
    pub width: Option<u32>,
}

/// Compiles `trace`'s stream reading (`o`) into `out`: the program's CSL (`layout.csl`, the
/// kernels, the printed regions) and the host's files (`host.json`, the arrays' spans and
/// gathers, `h2d/`: every array's words given a step's captured values `gold`, else only the
/// program's own). Returns the program.
pub fn compile(
    trace: &Trace,
    o: &Options,
    out: &std::path::Path,
    gold: Option<&std::path::Path>,
) -> R<build::Program> {
    compile_pinned(trace, o, &Pins::default(), out, gold)
}

/// [`compile`] with `pins`.
pub fn compile_pinned(
    trace: &Trace,
    o: &Options,
    pins: &Pins,
    out: &std::path::Path,
    gold: Option<&std::path::Path>,
) -> R<build::Program> {
    let s = shape(
        trace,
        &o.reading,
        &with_rows(
            program::classes(trace, &o.reading, o.rows),
            &pins.class_rows,
        ),
    )?;
    let p = build::program_py(
        trace,
        s,
        o.round16,
        o.upto,
        &o.taps,
        pins.py,
        pins.tile,
        pins.kv,
        pins.spill,
        crate::floorplan::stream::Bounds::system().narrowed(pins.width),
    )?;
    write_stream(&p, o, out, gold)?;
    Ok(p)
}

/// A built stream program's files into `out` (as [`compile_pinned`] writes them).
fn write_stream(
    p: &build::Program,
    o: &Options,
    out: &std::path::Path,
    gold: Option<&std::path::Path>,
) -> R<()> {
    let e = crate::emit::stream::emit(p)?;
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    for (name, text) in &e.files {
        std::fs::write(out.join(name), text).map_err(|e| format!("{name}: {e}"))?;
    }
    std::fs::write(
        out.join("fabric.txt"),
        format!("{} {} {}\n", e.w, e.h, u8::from(o.round16)),
    )
    .map_err(|e| e.to_string())?;
    let meta = serde_json::json!({
        "fabric": [e.w, e.h],
        "units": p.units.iter().map(|b| serde_json::json!({
            "name": b.unit.name, "rows": b.unit.rows, "cols": b.layout.px(), "heads": b.layout.heads,
            "segs": b.sched.as_ref().map(|s| format!("{:?}", s.segs)),
        })).collect::<Vec<_>>(),
        "kv": p.kv,
        "palette": p.palette.map.iter().map(|(k, c)| (format!("{}{}", k.name, k.index), c.0)).collect::<std::collections::BTreeMap<_, _>>(),
        "lane_words": p.lane_words,
        "py": p.stream.py,
        "cost": crate::cost::stream::time(p),
    });
    std::fs::write(
        out.join("stream.json"),
        serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    host::write(p, &e.arrays, gold, out)?;
    Ok(())
}

/// Compiles `trace`'s decode step as a unit program (GENERIC.md G1: decode through the stream
/// regime's units, `lanes` rows a step, a row a lane, each attention layer's K / V column its
/// cache of `positions` across steps, in pages: G1.8) into `out`. Returns the program.
pub fn compile_decode(
    trace: &Trace,
    positions: u32,
    lanes: u32,
    round16: bool,
    out: &std::path::Path,
) -> R<build::Program> {
    let classes = program::decode_classes(trace, lanes.max(1));
    let mut s = shape(trace, "decode", &classes)?;
    s.cache = Some(positions);
    let p = build::program_py(
        trace,
        s,
        round16,
        None,
        &[],
        None,
        None,
        None,
        false,
        crate::floorplan::stream::Bounds::system(),
    )?;
    let e = crate::emit::stream::emit(&p)?;
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    for (name, text) in &e.files {
        std::fs::write(out.join(name), text).map_err(|e| format!("{name}: {e}"))?;
    }
    std::fs::write(
        out.join("fabric.txt"),
        format!("{} {} {}\n", e.w, e.h, u8::from(round16)),
    )
    .map_err(|e| e.to_string())?;
    host::write(&p, &e.arrays, None, out)?;
    // the K / V PEs the host tells a step's first position (`pos`, a word each)
    let at = out.join("host.json");
    let mut h: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&at).map_err(|e| format!("{}: {e}", at.display()))?,
    )
    .map_err(|e| e.to_string())?;
    let pes = |sym: &str| -> Vec<serde_json::Value> {
        (e.arrays.iter())
            .filter(|a| a.sym == sym)
            .map(|a| serde_json::json!({"x": a.at.x, "y": a.at.y}))
            .collect()
    };
    h["pos"] = serde_json::json!(pes("pos"));
    // the attention columns' pages (GENERIC.md G1.8): a step's rows' (lane slot, position) and
    // each lane slot's pages, at every K / V PE
    if let Some(g) = crate::emit::stream::paging(&p) {
        h["paging"] = serde_json::json!({"size": g.size, "pages": g.pages, "lanes": g.lanes});
        h["rowp"] = serde_json::json!(pes("rowp"));
        h["ptab"] = serde_json::json!(pes("ptab"));
    }
    std::fs::write(
        &at,
        serde_json::to_string_pretty(&h).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(p)
}

/// Compiles `trace`'s reading `o.reading` into `out` with no capture, whichever regime it reads
/// in: a stream reading's units ([`compile`]), or a grid reading ([`compile_grid`], on the clip
/// `o.rows` token rows make: [`grid::clip_of`]). Returns its fabric's size.
pub fn compile_reading(trace: &Trace, o: &Options, out: &std::path::Path) -> R<(u32, u32)> {
    if grid::is_grid(trace, &o.reading) {
        let clip = grid::clip_of(trace, &o.reading, o.rows);
        return compile_grid(trace, &o.reading, clip, o.round16, out);
    }
    compile_whole(trace, o, out)
}

/// A whole-sequence reading's regime, by price: the stream regime's program (rows through
/// units) and the GEMM regime's ([`gemm`]: every op over the whole fabric), the cheaper of those
/// that place within a system's fabric. Returns the fabric's size.
pub fn compile_whole(trace: &Trace, o: &Options, out: &std::path::Path) -> R<(u32, u32)> {
    let b = crate::floorplan::stream::Bounds::system();
    let st = shape(
        trace,
        &o.reading,
        &program::classes(trace, &o.reading, o.rows),
    )
    .and_then(|s| {
        build::program_py(
            trace, s, o.round16, o.upto, &o.taps, None, None, None, false, b,
        )
    });
    let gm = gemm::program(trace, o, &[], None, (b.w, b.h));
    let sc = st
        .as_ref()
        .map_or(f64::MAX, |p| crate::cost::stream::time(p).cycles);
    let gc = gm
        .as_ref()
        .map_or(f64::MAX, |p| crate::cost::gemm::time(p).cycles);
    match (st, gm) {
        (Ok(p), Ok(_)) if sc <= gc => {
            write_stream(&p, o, out, None)?;
            Ok((p.stream.w, p.stream.h))
        }
        (_, Ok(g)) => {
            gemm::compile_program(&g, out, None)?;
            Ok(g.fabric())
        }
        (Ok(p), Err(_)) => {
            write_stream(&p, o, out, None)?;
            Ok((p.stream.w, p.stream.h))
        }
        (Err(a), Err(b)) => Err(format!("{a}; {b}")),
    }
}

/// A grid reading's regime, by price: the tiled regime's program ([`tiled`]: the image over the
/// fabric, its top tile searched) and the rows regime's ([`grid`]: passes over one unit), the
/// cheaper of those that place within a system's fabric. Returns the fabric's size.
pub fn compile_grid(
    trace: &Trace,
    reading: &str,
    clip: [u32; 3],
    round16: bool,
    out: &std::path::Path,
) -> R<(u32, u32)> {
    let b = crate::floorplan::stream::Bounds::system();
    let t = tiled::program(trace, reading, clip, round16, None, Some((b.w, b.h)));
    let g = grid::program(trace, reading, clip, round16);
    let tc = t
        .as_ref()
        .map_or(f64::MAX, |p| crate::cost::tiled::time(p).cycles);
    let gc = g
        .as_ref()
        .map_or(f64::MAX, |g| crate::cost::grid::time(g).cycles);
    match (t, g) {
        (Ok(p), Err(_)) => {
            tiled::compile_program(&p, out)?;
            Ok(p.fabric())
        }
        (Ok(p), Ok(_)) if tc <= gc => {
            tiled::compile_program(&p, out)?;
            Ok(p.fabric())
        }
        (_, Ok(_)) => {
            let g = grid::compile(trace, reading, clip, round16, out)?;
            Ok((g.stream.w, g.stream.h))
        }
        (Err(a), Err(b)) => Err(format!("{a}; {b}")),
    }
}

/// `classes` with the rows of those `rows` names.
#[must_use]
pub fn with_rows(mut classes: Vec<RowClass>, rows: &[(String, u32)]) -> Vec<RowClass> {
    for c in &mut classes {
        if let Some((_, n)) = rows.iter().find(|(name, _)| *name == c.name) {
            c.rows = *n;
        }
    }
    classes
}

/// Parses `name=rows,...` (wafer-compile's `--class-rows`, the survey's `ROWS`).
pub fn parse_rows(s: &str) -> R<Vec<(String, u32)>> {
    s.split(',')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (n, r) = p
                .split_once('=')
                .ok_or_else(|| format!("{p}: not name=rows"))?;
            Ok((n.to_string(), r.parse().map_err(|e| format!("{p}: {e}"))?))
        })
        .collect()
}

/// The shape of `trace`'s `reading` over the conventional row classes
/// ([`program::classes`]), `rows` rows each.
pub fn shape_of(trace: &Trace, reading: &str, rows: u32) -> R<StreamShape> {
    shape(trace, reading, &program::classes(trace, reading, rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(name: &str) -> Trace {
        poem_compiler::catalog::repository()
            .parse(name)
            .unwrap_or_else(|| panic!("{name} parses"))
            .trace(poem_ir::Platform::Cerebras)
    }

    /// flux2-mini's denoise at 64 text and 64 image rows: two 128-wide heads under per-head q / k
    /// norms, the four-axis interleaved rotation at theta 2000, SwiGLU three times the width.
    #[test]
    fn flux_2_minis_denoise_has_a_stream_shape() {
        let s = shape_of(&trace("flux2-mini-bf16-kv-bf16"), "denoise", 64)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            s.classes,
            [("context".to_string(), 64), ("latents".to_string(), 64)]
        );
        assert_eq!(s.hidden, 256);
        assert_eq!(
            s.attn,
            JointSpec {
                q_heads: 2,
                kv_heads: 2,
                head_dim: 128,
                sm_scale: 0.088_388_35,
                qk_norm: true,
                causal: false,
            }
        );
        assert_eq!(
            s.rope,
            Some(AxesRope {
                dims: [32; 4],
                thetas: [2000.0; 4],
                interleaved: true,
            })
        );
        assert_eq!((s.intermediate, s.act), (768, Act::Silu));
        assert_eq!(s.modulations, ["scale_shift"]);
        assert_eq!((s.units, s.layer_bodies), (9, 3));
    }

    /// flux2-mini holds no VAE: its `vae.decode` reading runs no layers.
    #[test]
    fn a_reading_the_poem_does_not_run_is_refused() {
        assert!(shape_of(&trace("flux2-mini-bf16-kv-bf16"), "vae.decode", 64).is_err());
    }
}
