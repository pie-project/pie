//! A short causal conv (`ssm.short_conv`: LFM2's, Inkling's on its keys and values) at a unit's
//! roots (`ops/sconv.csl`): each root its chunk's channels, their taps and the history of their
//! last rows (a lane slot's each in a decode step), no activation. It runs where its input is
//! (a root kernel: a region step's interface, its own module).

use std::fmt::Write as _;

use poem_compiler::fuse::chain::{Chain, In, Out};
use poem_ir::{Operation, Ssm, Trace};

use crate::compile::emit::region::{ParamAt, Place, Printed};
use crate::compile::emit::stream::Fill;
use crate::compile::stream::unit::{Lowering, Step, side_of};
use crate::kernel::{Kernel, R, RootArray, RootKernel, Source, UnitKernel};

/// A short causal conv.
pub struct ShortConv;

/// The registry's short conv.
pub static SHORT_CONV: ShortConv = ShortConv;

impl Kernel for ShortConv {
    fn name(&self) -> &'static str {
        "ssm.short_conv"
    }

    fn ops(&self) -> &'static [&'static str] {
        &["ssm.short_conv"]
    }

    fn sources(&self) -> &'static [Source] {
        &[("sconv.csl", include_str!("../../kernels/ssm/sconv.csl"))]
    }

    fn unit(&self) -> Option<&dyn UnitKernel> {
        Some(self)
    }

    fn root(&self) -> Option<&dyn RootKernel> {
        Some(self)
    }
}

/// The node's conv: (input, taps, taps a channel, output).
fn conv(
    trace: &Trace,
    n: u32,
) -> Option<(poem_ir::ValueId, poem_ir::ValueId, u32, poem_ir::ValueId)> {
    match &trace.nodes.get(n as usize)?.op {
        Operation::Ssm(Ssm::ShortConv {
            x,
            weight,
            conv_width,
            y,
            ..
        }) => Some((*x, *weight, *conv_width, *y)),
        _ => None,
    }
}

impl UnitKernel for ShortConv {
    /// A row's conv at the roots its input is at: a region step of one node.
    fn lower(&self, cx: &mut Lowering, at: u32) -> R<bool> {
        let r = cx.r;
        let Some((x, _, taps, y)) = conv(r.trace, at) else {
            return Ok(false);
        };
        if taps < 2 {
            return Err(cx.refuse(at, format!("ssm.short_conv of {taps} taps")));
        }
        let side = side_of(cx.sides, r.cols(x))
            .ok_or_else(|| cx.refuse(at, "ssm.short_conv reads a value no root has".into()))?;
        let dtype = |v| r.dtype(v).unwrap_or(poem_ir::Dtype::Bf16);
        let width = r.width(y);
        let chain = Chain {
            nodes: vec![at],
            ins: vec![In {
                value: x,
                dtype: dtype(x),
                shared: false,
                one: false,
                each: false,
                by: None,
                from: 0,
            }],
            outs: vec![Out {
                value: y,
                dtype: dtype(y),
            }],
            // (its interface alone: a row in, a row out; its code the module's)
            region: kernels_ir::Region {
                width,
                params: vec![kernels_ir::Shape::Row],
                ops: Vec::new(),
                results: vec![0],
                rounds: Vec::new(),
            },
        };
        cx.sides.insert(y, side);
        cx.steps.push(Step::Region { side, chain });
        Ok(true)
    }
}

impl RootKernel for ShortConv {
    fn print(
        &self,
        trace: &Trace,
        chain: &Chain,
        place: &Place,
        rows: u32,
        decode: bool,
    ) -> R<Printed> {
        let (_, _, taps, y) = conv(trace, chain.nodes[0]).ok_or("a short conv of no conv")?;
        if place.params != [ParamAt::Row] {
            return Err("ssm.short_conv: its input is no row".into());
        }
        let c = place.chunk;
        // its output rounded where it is a 16-bit value (in the build's 16-bit type)
        let round = place.round16
            && matches!(
                trace.values.get(y.0 as usize).map(|d| &d.ty),
                Some(poem_ir::Ty::Tensor {
                    dtype: poem_ir::Dtype::Bf16 | poem_ir::Dtype::F16,
                    ..
                })
            );
        let slots = if decode { rows } else { 1 };
        let mut t = String::new();
        let _ = writeln!(
            t,
            "// ssm.short_conv at a root (kernels-cerebras kernel::ssm::short_conv; ops/sconv.csl):\n\
             // {c} channels a root over {} roots, {taps} taps. Interface: in0 (the row), out0 (its\n\
             // conv), w (the taps), init(), row() a row.\n\
             param math: imported_module;\nparam seg: imported_module;\nparam rowp: [*]u32;\n\
             const C: u16 = {c};\n\
             var in0: [C]f32 align(16) = @zeros([C]f32);\n\
             const IN0 = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{C}} -> in0[i] }});\n\
             var out0: [C]f32 align(16) = @zeros([C]f32);\n\
             const OUT0 = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{C}} -> out0[i] }});\n\
             var w = @zeros([C * {taps}]f32);\n\
             const sc = @import_module(\"ops/sconv.csl\", .{{ .C = C, .T = {taps}, .SLOTS = {slots}, .ROWS = {rows}, .DECODE = {decode}, .w = &w, .rowp = rowp }});",
            place.roots
        );
        if round {
            t.push_str(
                "var r16: [C]@fp16() align(16) = @zeros([C]@fp16());\n\
                 const R16 = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{C} -> r16[i] });\n",
            );
        }
        t.push_str("fn init() void { sc.init(); }\n");
        let _ = writeln!(
            t,
            "fn row() void {{\n  sc.row(&in0, &out0);\n{}}}",
            if round {
                "  @fs2h(R16, OUT0); @fh2s(OUT0, R16);\n"
            } else {
                ""
            }
        );
        Ok(Printed {
            csl: t,
            params: vec![("in0".into(), ParamAt::Row)],
            results: vec!["out0".into()],
            tables: Vec::new(),
            all_reduces: 0,
            head_reduces: 0,
        })
    }

    /// Its taps: the chunk's channels', each a row of the taps (`[C][T]`, flat).
    fn arrays(&self, trace: &Trace, chain: &Chain, r: u32, c: u32) -> Vec<RootArray> {
        let Some((_, weight, taps, _)) = conv(trace, chain.nodes[0]) else {
            return Vec::new();
        };
        let width = chain.region.width;
        let first = chain.ins[0].from + r * c;
        let valid = width.saturating_sub(r * c).min(c);
        vec![RootArray {
            name: "w",
            len: c * taps,
            fill: Fill::Slices(vec![(weight.0, first * taps, valid * taps)]),
        }]
    }

    fn rows(&self) -> bool {
        true
    }
}
