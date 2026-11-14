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
pub mod host;
pub mod place;
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
                let width = match trace.values.get(q.0 as usize).map(|d| &d.ty) {
                    Some(poem_ir::Ty::Tensor { shape, .. }) => match shape.last() {
                        Some(poem_ir::Dim::Const(n)) => *n as u32,
                        _ => 0,
                    },
                    _ => 0,
                };
                let spec = JointSpec {
                    q_heads: width / (*head_dim).max(1),
                    kv_heads: *kv_heads,
                    head_dim: *head_dim,
                    sm_scale: *sm_scale,
                    qk_norm: false,
                };
                match &attn {
                    Some(a) if *a != spec => {
                        return Err(refuse(format!(
                            "attn.ragged of {spec:?} beside {a:?}: one attention shape a reading"
                        )));
                    }
                    _ => attn = Some(spec),
                }
            }
            Operation::Elemwise(Elemwise::RopeAxes {
                dims,
                thetas,
                form,
                rotary_dim,
                head_dim,
                ..
            }) => {
                let interleaved = match form {
                    RopeForm::Interleaved => true,
                    RopeForm::Neox => false,
                    other => {
                        return Err(refuse(format!(
                            "elemwise.rope_axes in the {other:?} form: the rotation tables turn \
                             interleaved or NeoX pairs"
                        )));
                    }
                };
                if rotary_dim != head_dim {
                    return Err(refuse(format!(
                        "elemwise.rope_axes over {rotary_dim} of a head's {head_dim} dims"
                    )));
                }
                let spec = AxesRope {
                    dims: *dims,
                    thetas: *thetas,
                    interleaved,
                };
                match &rope {
                    Some(r) if *r != spec => {
                        return Err(refuse("elemwise.rope_axes of two rotations".into()));
                    }
                    _ => rope = Some(spec),
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
            Operation::Linear(Linear::Glu {
                intermediate, act, ..
            }) => {
                let a = crate::target::act(act).map_err(refuse)?;
                match &glu {
                    Some((i, g)) if (*i, *g) != (*intermediate, a) => {
                        return Err(refuse(format!(
                            "linear.glu of {intermediate} wide beside {i}: one MLP a reading"
                        )));
                    }
                    _ => glu = Some((*intermediate, a)),
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
    let mut attn = attn.ok_or_else(|| format!("{name}: the stream step attends nothing"))?;
    attn.qk_norm = qk_norm;
    let (intermediate, act) = glu.ok_or_else(|| format!("{name}: the stream step has no MLP"))?;
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
    })
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

/// Compiles `trace`'s stream reading (`o`) into `out`: the program's CSL (`layout.csl`, the
/// kernels, the printed regions) and, given a step's captured values `gold`, the host's files
/// (`host.json`, `h2d/`). Returns the program.
pub fn compile(
    trace: &Trace,
    o: &Options,
    out: &std::path::Path,
    gold: Option<&std::path::Path>,
) -> R<build::Program> {
    let s = shape_of(trace, &o.reading, o.rows)?;
    let p = build::program_upto(trace, s, o.round16, o.upto, &o.taps)?;
    let e = crate::emit::stream::emit(&p)?;
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
        "cost": crate::cost::stream::time(&p),
    });
    std::fs::write(
        out.join("stream.json"),
        serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if let Some(g) = gold {
        host::write(&p, &e.arrays, g, out)?;
    }
    Ok(p)
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
