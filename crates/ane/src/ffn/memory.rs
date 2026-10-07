use crate::private::Surface;

use super::shape::{MAX_ROWS, SEGMENT, Shape};

pub struct Weights {
    pub gate: Vec<Surface>,
    pub up: Vec<Surface>,
    pub down: Vec<Surface>,
    pub gate_scale: Surface,
    pub up_scale: Surface,
    pub down_scale: Surface,
}

pub struct Memory {
    pub inputs: Vec<Surface>,
    pub token_scale: Surface,
    pub partial: Surface,
    pub sets: [Weights; 2],
}

impl Memory {
    pub fn new(shape: &Shape) -> Result<Memory, String> {
        let weights = || -> Result<Weights, String> {
            Ok(Weights {
                gate: (0..shape.segments())
                    .map(|_| Surface::new(shape.ane, SEGMENT, true))
                    .collect::<Result<_, _>>()?,
                up: (0..shape.segments())
                    .map(|_| Surface::new(shape.ane, SEGMENT, true))
                    .collect::<Result<_, _>>()?,
                down: shape
                    .down
                    .iter()
                    .map(|&width| Surface::new(shape.hidden, width, true))
                    .collect::<Result<_, _>>()?,
                gate_scale: Surface::new(shape.ane, 1, false)?,
                up_scale: Surface::new(shape.ane, 1, false)?,
                down_scale: Surface::new(shape.hidden, 1, false)?,
            })
        };
        Ok(Memory {
            inputs: (0..shape.segments())
                .map(|_| Surface::new(SEGMENT, MAX_ROWS, true))
                .collect::<Result<_, _>>()?,
            token_scale: Surface::new(1, MAX_ROWS, false)?,
            partial: Surface::new(shape.hidden + 1, MAX_ROWS, false)?,
            sets: [weights()?, weights()?],
        })
    }
}

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
