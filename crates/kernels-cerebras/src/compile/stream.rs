//! The stream regime's shape: what a reading that runs whole sequences a step (a diffusion
//! transformer's denoise) asks of the wafer, read off its trace ([`shape`]). The reading's stream
//! program says what runs where; the shape adds the params the kernels take. An op whose params
//! no stream kernel takes is refused by name; nothing is dropped.

use poem_compiler::spatial::layer::stream::{self as program, RowClass, StreamProgram};
use poem_ir::{
    Attn, Elemwise, Linear, ModulateForm, NormAxis, Operation, RaggedMask, RopeForm, Trace,
};
use serde::Serialize;

use crate::compile::plan::Act;

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
    let prog = program::lower_with(trace, &analysis, classes, &crate::kernel::claims)?;
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
                // (an activation the templates run none of, Kimi's SiTU, the regions print
                // alone: the shape states no MLP)
                if !matches!(act, poem_ir::GluAct::Situ { .. }) {
                    glu = Some((
                        *intermediate,
                        crate::compile::target::act(act).map_err(refuse)?,
                    ));
                }
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
    /// The annealing placer after the search ([`crate::compile::place::anneal::Anneal`]; off: the
    /// search's placement, byte for byte).
    pub anneal: crate::compile::place::anneal::Anneal,
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
    let s = shape(
        trace,
        &o.reading,
        &program::classes(trace, &o.reading, o.rows),
    )?;
    let p = build::program_py(
        trace,
        s,
        o.round16,
        o.upto,
        &o.taps,
        None,
        None,
        None,
        false,
        crate::compile::floorplan::stream::Bounds::system(),
    )?;
    write_stream(&p, o, out, gold)?;
    Ok(p)
}

/// A built stream program's files into `out` (as [`compile`] writes them).
fn write_stream(
    p: &build::Program,
    o: &Options,
    out: &std::path::Path,
    gold: Option<&std::path::Path>,
) -> R<()> {
    // (the annealer's placement where `o` asks)
    let annealed = crate::compile::place::anneal::maybe(p, &o.anneal)?;
    let p = annealed.as_ref().map_or(p, |(q, _)| q);
    let e = crate::compile::emit::stream::emit(p)?;
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    for (name, text) in &e.files {
        std::fs::write(out.join(name), text).map_err(|e| format!("{name}: {e}"))?;
    }
    crate::compile::check::written(&e.placed, &e.files, None);
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
        "cost": crate::compile::cost::stream::time(p),
    });
    std::fs::write(
        out.join("stream.json"),
        serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    host::write(p, &e.arrays, gold, out)?;
    Ok(())
}

/// Compiles `trace`'s decode step as a unit program into `out`: `lanes` rows a step, a row a
/// lane, each attention layer's K / V column its paged cache of `positions`. Returns the program.
pub fn compile_decode(
    trace: &Trace,
    positions: u32,
    lanes: u32,
    round16: bool,
    upto: Option<usize>,
    anneal: &crate::compile::place::anneal::Anneal,
    out: &std::path::Path,
) -> R<build::Program> {
    let classes = program::decode_classes(trace, lanes.max(1));
    let mut s = shape(trace, "decode", &classes)?;
    s.cache = Some(positions);
    let p = build::program_py(
        trace,
        s,
        round16,
        upto,
        &[],
        None,
        None,
        None,
        false,
        crate::compile::floorplan::stream::Bounds::system(),
    )?;
    // (the annealer's placement where `anneal` asks)
    let p = match crate::compile::place::anneal::maybe(&p, anneal)? {
        Some((q, _)) => q,
        None => p,
    };
    let e = crate::compile::emit::stream::emit(&p)?;
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    for (name, text) in &e.files {
        std::fs::write(out.join(name), text).map_err(|e| format!("{name}: {e}"))?;
    }
    crate::compile::check::written(&e.placed, &e.files, None);
    std::fs::write(
        out.join("fabric.txt"),
        format!("{} {} {}\n", e.w, e.h, u8::from(round16)),
    )
    .map_err(|e| e.to_string())?;
    host::write(&p, &e.arrays, None, out)?;
    // the program's state (its K / V and recurrent columns, their slots and pages, the PEs a
    // step's position, rows and page tables go to): the one description the engine reads
    let at = out.join("host.json");
    let mut h: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&at).map_err(|e| format!("{}: {e}", at.display()))?,
    )
    .map_err(|e| e.to_string())?;
    if let Some(root) = crate::compile::emit::stream::state_root(&p, &e.arrays) {
        h["state"] = serde_json::to_value(root).map_err(|e| e.to_string())?;
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
    compile_whole(trace, o, &[], out)
}

/// The most requests compile on load batches a launch: the price picks how many up to it,
/// where the fabric and a PE's memory hold them (a batch's step serves its requests at once).
pub const BATCH_MOST: u32 = 4;

/// A whole-sequence reading's regime, by price: the stream regime's program (rows through
/// units) and the GEMM regime's ([`gemm`]: every op over the whole fabric), the cheaper of those
/// that place within a system's fabric. Returns the fabric's size.
pub fn compile_whole(
    trace: &Trace,
    o: &Options,
    class_rows: &[(String, u32)],
    out: &std::path::Path,
) -> R<(u32, u32)> {
    let b = crate::compile::floorplan::stream::Bounds::system();
    let st = shape(
        trace,
        &o.reading,
        &with_rows(program::classes(trace, &o.reading, o.rows), class_rows),
    )
    .and_then(|s| {
        build::program_py(
            trace, s, o.round16, o.upto, &o.taps, None, None, None, false, b,
        )
    });
    // (the GEMM regime's batch by price: a request's share of its step)
    let gm = gemm::batched(trace, o, class_rows, (b.w, b.h), BATCH_MOST);
    // (each regime's price, a request's)
    let sc = st
        .as_ref()
        .map_or(f64::MAX, crate::compile::cost::stream::expected);
    let gc = gm.as_ref().map_or(f64::MAX, |g| {
        crate::compile::cost::gemm::expected(g) / f64::from(g.l.batch.max(1))
    });
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

/// A regime and shape a reading's program can take.
#[derive(Clone, Debug, PartialEq)]
pub enum Regime {
    /// Rows through units, the stream compiler's own floorplan.
    Stream,
    /// The GEMM regime over `c` x `r` PEs, `kbm` key-block members, `batch` requests a launch.
    Gemm {
        c: u32,
        r: u32,
        kbm: u32,
        batch: u32,
    },
    /// The tiled regime, its top tile `tile` (rows, columns) and its convolutions' staged
    /// channels and weight block at most.
    Tiled {
        tile: (u32, u32),
        chunks: tiled::Chunks,
    },
    /// The rows regime: passes over one unit.
    Rows,
}

/// Whether programs of these sizes (w, h) place together, or by how much they miss.
pub type Fit = dyn Fn(&[(u32, u32)]) -> Result<(), crate::compile::place::pack::Miss>;

/// The readings compiled together with their points, and those refused with why.
pub type Together = (Vec<(String, Point)>, Vec<(String, String)>);

/// A program a reading can be compiled to: its regime, its fabric and its price (cycles a
/// request's run of it).
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    pub regime: Regime,
    pub w: u32,
    pub h: u32,
    pub cycles: f64,
}

/// Every program `o.reading` compiles to within a system's fabric (a grid reading on the clip
/// `o.rows` voxels make), each regime at each shape its compiler builds, less those another is
/// no larger than and no dearer than; or why none is.
pub fn points(trace: &Trace, o: &Options, class_rows: &[(String, u32)]) -> R<Vec<Point>> {
    let b = crate::compile::floorplan::stream::Bounds::system();
    let mut all: Vec<Point> = Vec::new();
    let mut why: Vec<String> = Vec::new();
    if grid::is_grid(trace, &o.reading) {
        let clip = grid::clip_of(trace, &o.reading, o.rows);
        match tiled::points(trace, &o.reading, clip, o.round16, (b.w, b.h)) {
            Ok(v) => all.extend(v.into_iter().map(|(cycles, tile, chunks, (w, h))| Point {
                regime: Regime::Tiled { tile, chunks },
                w,
                h,
                cycles,
            })),
            Err(e) => why.push(e),
        }
        match grid::program(trace, &o.reading, clip, o.round16) {
            Ok(g) if g.stream.w <= b.w && g.stream.h <= b.h => all.push(Point {
                regime: Regime::Rows,
                w: g.stream.w,
                h: g.stream.h,
                cycles: crate::compile::cost::grid::time(&g).cycles,
            }),
            Ok(g) => why.push(format!(
                "the rows regime's {} x {} PEs past the fabric",
                g.stream.w, g.stream.h
            )),
            Err(e) => why.push(e),
        }
    } else {
        let st = shape(
            trace,
            &o.reading,
            &with_rows(program::classes(trace, &o.reading, o.rows), class_rows),
        )
        .and_then(|s| {
            build::program_py(
                trace, s, o.round16, o.upto, &o.taps, None, None, None, false, b,
            )
        });
        match st {
            Ok(p) => all.push(Point {
                regime: Regime::Stream,
                w: p.stream.w,
                h: p.stream.h,
                cycles: crate::compile::cost::stream::expected(&p),
            }),
            Err(e) => why.push(e),
        }
        for batch in 1..=BATCH_MOST {
            match gemm::candidates(trace, o, class_rows, None, (b.w, b.h), batch) {
                Ok((v, w)) => {
                    if v.is_empty() && !w.is_empty() {
                        why.push(w);
                    }
                    all.extend(v.into_iter().map(|(_, g)| {
                        let (w, h) = g.fabric();
                        Point {
                            regime: Regime::Gemm {
                                c: g.c,
                                r: g.r,
                                kbm: g.kbm,
                                batch,
                            },
                            w,
                            h,
                            cycles: crate::compile::cost::gemm::expected(&g) / f64::from(batch),
                        }
                    }));
                }
                Err(e) => why.push(e),
            }
        }
    }
    let kept = pareto(all);
    if kept.is_empty() {
        return Err(why.join("; "));
    }
    Ok(kept)
}

/// The points no other is as small as and as cheap as (the cheaper on a tie), cheapest first.
fn pareto(mut all: Vec<Point>) -> Vec<Point> {
    all.sort_by(|a, b| a.cycles.total_cmp(&b.cycles));
    let mut kept: Vec<Point> = Vec::new();
    for p in all {
        if !kept.iter().any(|k| k.w <= p.w && k.h <= p.h) {
            kept.push(p);
        }
    }
    kept
}

/// Compiles `o.reading` at `point` into `out`. Returns the fabric's size.
pub fn compile_point(
    trace: &Trace,
    o: &Options,
    class_rows: &[(String, u32)],
    point: &Point,
    out: &std::path::Path,
) -> R<(u32, u32)> {
    let b = crate::compile::floorplan::stream::Bounds::system();
    match &point.regime {
        Regime::Stream => {
            let s = shape(
                trace,
                &o.reading,
                &with_rows(program::classes(trace, &o.reading, o.rows), class_rows),
            )?;
            let p = build::program_py(
                trace, s, o.round16, o.upto, &o.taps, None, None, None, false, b,
            )?;
            write_stream(&p, o, out, None)?;
            Ok((p.stream.w, p.stream.h))
        }
        Regime::Gemm { c, r, kbm, batch } => {
            let g = gemm::program(
                trace,
                o,
                class_rows,
                Some((*c, *r, *kbm)),
                (b.w, b.h),
                *batch,
            )?;
            gemm::compile_program(&g, out, None)?;
            Ok(g.fabric())
        }
        Regime::Tiled { tile, chunks } => {
            let clip = grid::clip_of(trace, &o.reading, o.rows);
            let p = tiled::program_with(
                trace,
                &o.reading,
                clip,
                o.round16,
                *tile,
                *chunks,
                Some((b.w, b.h)),
            )?;
            tiled::compile_program(&p, out)?;
            Ok(p.fabric())
        }
        Regime::Rows => {
            let clip = grid::clip_of(trace, &o.reading, o.rows);
            let g = grid::compile(trace, &o.reading, clip, o.round16, out)?;
            Ok((g.stream.w, g.stream.h))
        }
    }
}

/// One point a reading (each reading's points, and the runs of it a request makes), the
/// combination whose request is cheapest (the runs' cycles summed) of those `fit` places
/// together; `fit` gives the rows a combination falls short by where it does not. Returns
/// the chosen point of each reading, or the nearest miss by name.
pub fn choose(readings: &[(String, Vec<Point>, f64)], fit: &Fit) -> R<Vec<usize>> {
    let mut combos: Vec<(f64, Vec<usize>)> = vec![(0.0, Vec::new())];
    for (_, pts, runs) in readings {
        let mut next = Vec::with_capacity(combos.len() * pts.len());
        for (c, at) in &combos {
            for (i, p) in pts.iter().enumerate() {
                let mut a = at.clone();
                a.push(i);
                next.push((c + runs * p.cycles, a));
            }
        }
        combos = next;
    }
    combos.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut nearest: Option<(crate::compile::place::pack::Miss, f64, Vec<usize>)> = None;
    for (cy, at) in combos {
        let sizes: Vec<(u32, u32)> = readings
            .iter()
            .zip(&at)
            .map(|((_, pts, _), &i)| (pts[i].w, pts[i].h))
            .collect();
        match fit(&sizes) {
            Ok(()) => return Ok(at),
            Err(miss) => {
                let by = |m: &crate::compile::place::pack::Miss| (m.dy, m.dx);
                if nearest.as_ref().is_none_or(|(s, _, _)| by(&miss) < by(s)) {
                    nearest = Some((miss, cy, at));
                }
            }
        }
    }
    let Some((miss, cy, at)) = nearest else {
        return Err("no reading to place".into());
    };
    let parts: Vec<String> = readings
        .iter()
        .zip(&at)
        .map(|((name, pts, _), &i)| {
            let p = &pts[i];
            format!("{name} {:?} {} x {}", p.regime, p.w, p.h)
        })
        .collect();
    Err(format!(
        "no combination of the readings' programs fits the fabric together: the nearest ({}; {:.1}M cycles a request) is {miss}",
        parts.join(", "),
        cy / 1e6
    ))
}

/// A reading compile on load serves: its options (`o.rows` its classes' rows, a grid
/// reading's the token rows its clip is of), the rows of the classes named in place of those,
/// and the runs of it a request makes.
#[derive(Clone, Debug)]
pub struct Wanted {
    pub o: Options,
    pub class_rows: Vec<(String, u32)>,
    pub runs: f64,
}

/// The readings `trace` declares as compile on load serves them at `rows` rows a class: where
/// `voxels` states the latent's voxels, a class (or a grid reading) that reads a voxel input
/// takes the token rows they make; a whole-sequence reading over the latent runs `steps` times
/// a request (the schedule's loop), every other once.
#[must_use]
pub fn wanted(trace: &Trace, rows: u32, voxels: Option<u32>, steps: u32) -> Vec<Wanted> {
    let Some(switch) = trace.facts.declared("reading") else {
        return Vec::new();
    };
    let latent_rows = voxels.map(|v| (v / grid::voxels_a_row(trace)).max(1));
    switch
        .values
        .iter()
        .map(|reading| {
            let is_grid = grid::is_grid(trace, reading);
            let classes = program::classes(trace, reading, rows);
            let latent: Vec<String> = classes
                .iter()
                .filter(|c| grid::reads_voxels(trace, c.word))
                .map(|c| c.name.clone())
                .collect();
            let o = Options {
                reading: reading.clone(),
                rows: if is_grid {
                    latent_rows.unwrap_or(rows)
                } else {
                    rows
                },
                round16: true,
                upto: None,
                taps: Vec::new(),
                anneal: Default::default(),
            };
            let class_rows = match (is_grid, latent_rows) {
                (false, Some(n)) => latent.iter().map(|c| (c.clone(), n)).collect(),
                _ => Vec::new(),
            };
            let runs = if !is_grid && !latent.is_empty() {
                f64::from(steps.max(1))
            } else {
                1.0
            };
            Wanted {
                o,
                class_rows,
                runs,
            }
        })
        .collect()
}

/// Compiles the `wanted` readings to run together: each one's [`points`], the cheapest
/// combination that places together on a system's fabric ([`choose`],
/// [`coresident::fits`]), each compiled into `dir(reading)`. A reading no point builds is left
/// out (with why). Returns the compiled readings with their points, and the refused ones.
pub fn compile_together(
    trace: &Trace,
    wanted: &[Wanted],
    dir: &dyn Fn(&str) -> std::path::PathBuf,
) -> R<Together> {
    let mut ok: Vec<(&Wanted, Vec<Point>)> = Vec::new();
    let mut refused = Vec::new();
    for w in wanted {
        match points(trace, &w.o, &w.class_rows) {
            Ok(p) => ok.push((w, p)),
            Err(e) => refused.push((w.o.reading.clone(), e)),
        }
    }
    let readings: Vec<(String, Vec<Point>, f64)> = ok
        .iter()
        .map(|(w, p)| (w.o.reading.clone(), p.clone(), w.runs))
        .collect();
    if readings.is_empty() {
        return Ok((Vec::new(), refused));
    }
    let at = choose(&readings, &coresident::fits)?;
    let mut out = Vec::new();
    for ((w, pts), i) in ok.iter().zip(at) {
        let p = &pts[i];
        compile_point(trace, &w.o, &w.class_rows, p, &dir(&w.o.reading))?;
        out.push((w.o.reading.clone(), p.clone()));
    }
    Ok((out, refused))
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
    let b = crate::compile::floorplan::stream::Bounds::system();
    let t = tiled::program(trace, reading, clip, round16, None, Some((b.w, b.h)));
    let g = grid::program(trace, reading, clip, round16);
    let tc = t
        .as_ref()
        .map_or(f64::MAX, |p| crate::compile::cost::tiled::time(p).cycles);
    let gc = g
        .as_ref()
        .map_or(f64::MAX, |g| crate::compile::cost::grid::time(g).cycles);
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
