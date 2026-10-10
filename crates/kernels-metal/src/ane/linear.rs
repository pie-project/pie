//! The Neural Engine's share of a plain projection `y = x W^T`: its trailing
//! output columns, as one int8 matmul over the activations the GPU rotated
//! and packed, written as an fp16 partial the GPU rescales per token into
//! place. No nonlinearity, so no requantization: one procedure per row
//! count, bound twice over two weight sets.

use std::path::Path;

use super::ffn::{INT8_UNIT, MAX_ROWS, UNIT};
use super::mil::{self, Body, Input};
use super::{Binding, Element, Program, Surface};

#[derive(Clone, Debug)]
pub struct Shape {
    /// The contraction, split into `segment`-wide pieces.
    pub k: u32,
    pub segment: u32,
    /// The projection's output columns.
    pub n: u32,
    /// The output columns the GPU keeps: the leading ones.
    pub keep: u32,
    /// The output columns the Neural Engine takes: the trailing ones.
    pub ane: u32,
}

impl Shape {
    /// `units` of [`UNIT`] output columns go to the Neural Engine.
    pub fn new(k: u32, n: u32, units: u32) -> Result<Shape, String> {
        let Some(segment) = mil::segment_of(k) else {
            return Err(format!(
                "a contraction of {k} does not split into segments the Neural Engine takes"
            ));
        };
        let ane = units * UNIT;
        if units == 0 || ane >= n || !n.is_multiple_of(8) {
            return Err(format!(
                "{units} units leave the GPU or the Neural Engine nothing of {n}"
            ));
        }
        Ok(Shape {
            k,
            segment,
            n,
            keep: n - ane,
            ane,
        })
    }

    #[must_use]
    pub fn segments(&self) -> u32 {
        self.k / self.segment
    }
}

/// One layer's share of the weights: int8 rows per segment, an fp16 scale
/// per row.
pub struct Weights {
    pub w: Vec<Surface>,
    pub scale: Surface,
}

pub struct Memory {
    /// The activations, channel-major int8, one surface per segment.
    pub inputs: Vec<Surface>,
    /// The per-token scale the activations were quantized by; the GPU reads
    /// it back at the join, the program never does.
    pub token_scale: Surface,
    /// What the Neural Engine writes: `ane` rows of partial sums.
    pub partial: Surface,
    pub sets: [Weights; 2],
}

impl Memory {
    pub fn new(shape: &Shape) -> Result<Memory, String> {
        let weights = || -> Result<Weights, String> {
            Ok(Weights {
                w: (0..shape.segments())
                    .map(|_| Surface::new(shape.ane, shape.segment, Element::Int8))
                    .collect::<Result<_, _>>()?,
                scale: Surface::new(shape.ane, 1, Element::Fp16)?,
            })
        };
        Ok(Memory {
            inputs: (0..shape.segments())
                .map(|_| Surface::new(shape.segment, MAX_ROWS, Element::Int8))
                .collect::<Result<_, _>>()?,
            token_scale: Surface::new(1, MAX_ROWS, Element::Fp16)?,
            partial: Surface::new(shape.ane, MAX_ROWS, Element::Fp16)?,
            sets: [weights()?, weights()?],
        })
    }
}

fn inputs<'a>(shape: &Shape, memory: &'a Memory) -> Vec<Input<'a>> {
    let mut list = Vec::new();
    for (k, x) in memory.inputs.iter().enumerate() {
        list.push(Input {
            name: format!("x{k}"),
            surfaces: [x, x],
            rows: shape.segment,
            width: 0,
        });
    }
    let [a, b] = &memory.sets;
    for k in 0..shape.segments() as usize {
        list.push(Input {
            name: format!("w{k}"),
            surfaces: [&a.w[k], &b.w[k]],
            rows: shape.ane,
            width: shape.segment,
        });
    }
    list.push(Input {
        name: "s".into(),
        surfaces: [&a.scale, &b.scale],
        rows: shape.ane,
        width: 1,
    });
    list
}

#[must_use]
pub fn function_name(rows: u32) -> String {
    format!("linear{rows}")
}

/// The procedure specialized for `rows` rows: `y = (sum_k w_k x_k) * s`.
fn function(shape: &Shape, inputs: &[Input<'_>], output: &Surface, rows: u32) -> String {
    let unit = mil::fp16(1.0 / INT8_UNIT);
    let (columns, segment, r) = (
        u64::from(shape.ane),
        u64::from(shape.segment),
        u64::from(rows),
    );
    let mut b = Body::open(inputs, rows);
    let segments = shape.segments() as usize;
    for k in 0..segments {
        b.f16(
            &format!("x{k}_d"),
            segment,
            r,
            format!("dequantize(input = x{k}_t, scale = {unit})"),
        );
        b.matmul(
            &format!("m{k}"),
            &format!("w{k}"),
            &format!("x{k}_d"),
            columns,
            segment,
            r,
        );
    }
    let total = b.sum("m", segments, columns, r);
    b.f16("ys", columns, r, format!("mul(x = {total}, y = s_t)"));
    b.line(format!(
        "{} y = tensor_to_tensor_buffer<ios17>(input = ys, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>({}))",
        output.buffer_type(shape.ane, rows),
        output.strides(shape.ane)
    ));
    b.close(&function_name(rows))
}

pub struct Linear {
    pub program: Program,
    /// Per row count, that procedure bound over each weight set.
    pub evaluations: Vec<(u32, [Binding; 2])>,
}

impl Linear {
    /// Writes and compiles the program for `shape` over `memory`'s surfaces.
    pub fn compile(shape: &Shape, memory: &Memory, cache: &Path) -> Result<Linear, String> {
        let list = inputs(shape, memory);
        let rows = mil::function_rows();
        let mut text = String::from("program(1.3)\n{\n");
        for &count in &rows {
            text.push_str(&function(shape, &list, &memory.partial, count));
        }
        text.push_str("}\n");
        // No constants: an empty blob keeps the program's layout the MLP's.
        let program = Program::compile(&text, &super::constant_blob(&[]), cache)?;
        let mut evaluations = Vec::new();
        for &count in &rows {
            let procedure = program
                .procedure(&function_name(count))
                .ok_or_else(|| format!("the program has no {}", function_name(count)))?;
            let names = program.inputs(procedure);
            let bind = |set: usize| -> Result<Binding, String> {
                let surfaces = names
                    .iter()
                    .map(|name| {
                        list.iter()
                            .find(|input| &input.name == name)
                            .map(|input| input.surfaces[set])
                            .ok_or_else(|| format!("the program has an unknown input {name}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                program.bind(procedure, &surfaces, &memory.partial)
            };
            evaluations.push((count, [bind(0)?, bind(1)?]));
        }
        Ok(Linear {
            program,
            evaluations,
        })
    }

    /// The index into `evaluations` of the smallest procedure that fits
    /// `rows`, if any does.
    #[must_use]
    pub fn evaluation(&self, rows: u32) -> Option<usize> {
        self.evaluations
            .iter()
            .position(|(count, _)| *count >= rows)
    }
}
