//! The surfaces one split MLP runs over: the activations and their scales,
//! two sets of weights that alternate by layer parity, and the partial the
//! Neural Engine writes.

use super::shape::{MAX_ROWS, Shape};
use crate::ane::mil::Input;
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

impl Weights {
    /// The bytes of every surface the set holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.gate
            .iter()
            .chain(&self.up)
            .chain(&self.down)
            .chain([&self.gate_scale, &self.up_scale, &self.down_scale])
            .map(Surface::bytes)
            .sum()
    }

    /// About what one set of `shape` takes, before any is made: its int8
    /// weights and fp16 row scales.
    #[must_use]
    pub fn estimate(shape: &Shape) -> u64 {
        let (ane, hidden) = (u64::from(shape.ane), u64::from(shape.hidden));
        3 * ane * hidden + 2 * (2 * ane + hidden)
    }
}

pub struct Memory {
    /// The activations, channel-major int8, one surface per hidden segment.
    pub inputs: Vec<Surface>,
    /// The per-token scale the activations were quantized by.
    pub token_scale: Surface,
    /// What the Neural Engine writes: `hidden` rows of partial sums over
    /// its channels, then one row of the intermediate's requantization scale.
    pub partial: Surface,
    /// The weight sets the program is bound over: two that alternate by
    /// layer, so the GPU stages the next layer's while the Neural Engine
    /// reads this one's, or one a layer, converted once at load.
    pub sets: Vec<Weights>,
}

impl Memory {
    /// Surfaces for `sets` weight sets, at least two.
    pub fn new(shape: &Shape, sets: usize) -> Result<Memory, String> {
        let weights = || -> Result<Weights, String> {
            Ok(Weights {
                gate: (0..shape.segments())
                    .map(|_| Surface::new(shape.ane, shape.segment, Element::Int8))
                    .collect::<Result<_, _>>()?,
                up: (0..shape.segments())
                    .map(|_| Surface::new(shape.ane, shape.segment, Element::Int8))
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
                .map(|_| Surface::new(shape.segment, MAX_ROWS, Element::Int8))
                .collect::<Result<_, _>>()?,
            token_scale: Surface::new(1, MAX_ROWS, Element::Fp16)?,
            partial: Surface::new(shape.hidden + 1, MAX_ROWS, Element::Fp16)?,
            sets: (0..sets.max(2))
                .map(|_| weights())
                .collect::<Result<_, _>>()?,
        })
    }
}

pub(super) fn inputs<'a>(shape: &Shape, memory: &'a Memory) -> Vec<Input<'a>> {
    let mut list = Vec::new();
    for (k, x) in memory.inputs.iter().enumerate() {
        list.push(Input {
            name: format!("x{k}"),
            surfaces: vec![x; memory.sets.len()],
            rows: shape.segment,
            width: 0,
        });
    }
    list.push(Input {
        name: "tx".into(),
        surfaces: vec![&memory.token_scale; memory.sets.len()],
        rows: 1,
        width: 0,
    });
    let sets = &memory.sets;
    for k in 0..shape.segments() as usize {
        list.push(Input {
            name: format!("wg{k}"),
            surfaces: sets.iter().map(|w| &w.gate[k]).collect(),
            rows: shape.ane,
            width: shape.segment,
        });
        list.push(Input {
            name: format!("wu{k}"),
            surfaces: sets.iter().map(|w| &w.up[k]).collect(),
            rows: shape.ane,
            width: shape.segment,
        });
    }
    list.push(Input {
        name: "sg".into(),
        surfaces: sets.iter().map(|w| &w.gate_scale).collect(),
        rows: shape.ane,
        width: 1,
    });
    list.push(Input {
        name: "su".into(),
        surfaces: sets.iter().map(|w| &w.up_scale).collect(),
        rows: shape.ane,
        width: 1,
    });
    for (i, &width) in shape.down.iter().enumerate() {
        list.push(Input {
            name: format!("wd{i}"),
            surfaces: sets.iter().map(|w| &w.down[i]).collect(),
            rows: shape.hidden,
            width,
        });
    }
    list.push(Input {
        name: "sd".into(),
        surfaces: sets.iter().map(|w| &w.down_scale).collect(),
        rows: shape.hidden,
        width: 1,
    });
    list
}
