//! The surfaces one split MLP runs over: the activations and their scales,
//! two sets of weights that alternate by layer parity, and the partial the
//! Neural Engine writes.

use super::shape::{MAX_ROWS, SEGMENT, Shape};
use crate::ane::{Element, Surface};

/// One layer's share of the weights, int8 rows with an fp16 scale per row.
pub struct Weights {
    pub gate: Vec<Surface>,
    pub up: Vec<Surface>,
    pub down: Vec<Surface>,
    pub gate_scale: Surface,
    pub up_scale: Surface,
    pub down_scale: Surface,
}

pub struct Memory {
    /// The activations, channel-major int8, one surface per hidden segment.
    pub inputs: Vec<Surface>,
    /// The per-token scale the activations were quantized by.
    pub token_scale: Surface,
    /// What the Neural Engine writes: `hidden` rows of partial sums over
    /// its channels, then one row of the intermediate's requantization scale.
    pub partial: Surface,
    /// Two layers' weights, so the GPU stages the next layer's while the
    /// Neural Engine reads this one's.
    pub sets: [Weights; 2],
}

impl Memory {
    pub fn new(shape: &Shape) -> Result<Memory, String> {
        let weights = || -> Result<Weights, String> {
            Ok(Weights {
                gate: (0..shape.segments())
                    .map(|_| Surface::new(shape.ane, SEGMENT, Element::Int8))
                    .collect::<Result<_, _>>()?,
                up: (0..shape.segments())
                    .map(|_| Surface::new(shape.ane, SEGMENT, Element::Int8))
                    .collect::<Result<_, _>>()?,
                down: shape
                    .down
                    .iter()
                    .map(|&width| Surface::new(shape.hidden, width, Element::Int8))
                    .collect::<Result<_, _>>()?,
                gate_scale: Surface::new(shape.ane, 1, Element::Fp16)?,
                up_scale: Surface::new(shape.ane, 1, Element::Fp16)?,
                down_scale: Surface::new(shape.hidden, 1, Element::Fp16)?,
            })
        };
        Ok(Memory {
            inputs: (0..shape.segments())
                .map(|_| Surface::new(SEGMENT, MAX_ROWS, Element::Int8))
                .collect::<Result<_, _>>()?,
            token_scale: Surface::new(1, MAX_ROWS, Element::Fp16)?,
            partial: Surface::new(shape.hidden + 1, MAX_ROWS, Element::Fp16)?,
            sets: [weights()?, weights()?],
        })
    }
}

/// One input of the program: its name, the surface behind it in each weight
/// set, and its tensor shape. A `width` of zero means the row count the
/// procedure is specialized for.
pub(super) struct Input<'a> {
    pub(super) name: String,
    pub(super) surfaces: [&'a Surface; 2],
    pub(super) rows: u32,
    pub(super) width: u32,
}

pub(super) fn inputs<'a>(shape: &Shape, memory: &'a Memory) -> Vec<Input<'a>> {
    let mut list = Vec::new();
    for (k, x) in memory.inputs.iter().enumerate() {
        list.push(Input {
            name: format!("x{k}"),
            surfaces: [x, x],
            rows: SEGMENT,
            width: 0,
        });
    }
    list.push(Input {
        name: "tx".into(),
        surfaces: [&memory.token_scale, &memory.token_scale],
        rows: 1,
        width: 0,
    });
    let [a, b] = &memory.sets;
    for k in 0..shape.segments() as usize {
        list.push(Input {
            name: format!("wg{k}"),
            surfaces: [&a.gate[k], &b.gate[k]],
            rows: shape.ane,
            width: SEGMENT,
        });
        list.push(Input {
            name: format!("wu{k}"),
            surfaces: [&a.up[k], &b.up[k]],
            rows: shape.ane,
            width: SEGMENT,
        });
    }
    list.push(Input {
        name: "sg".into(),
        surfaces: [&a.gate_scale, &b.gate_scale],
        rows: shape.ane,
        width: 1,
    });
    list.push(Input {
        name: "su".into(),
        surfaces: [&a.up_scale, &b.up_scale],
        rows: shape.ane,
        width: 1,
    });
    for (i, &width) in shape.down.iter().enumerate() {
        list.push(Input {
            name: format!("wd{i}"),
            surfaces: [&a.down[i], &b.down[i]],
            rows: shape.hidden,
            width,
        });
    }
    list.push(Input {
        name: "sd".into(),
        surfaces: [&a.down_scale, &b.down_scale],
        rows: shape.hidden,
        width: 1,
    });
    list
}
