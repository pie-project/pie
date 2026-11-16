//! The guest-program contract every guest-capable backend runs: a program's
//! stage as an inferlet registered it, the seeds and logits a fire is given,
//! and what the host interpreter answers. A backend registers the case's
//! program, binds it with [`Case::wire_seeds`], fires it over
//! [`Case::logits`] and reads [`Case::readers`]: each channel's bytes are
//! [`Case::expected`]'s, exactly.
//!
//! The beam cases are beam_search's step on Qwen3.5-0.8B (a 248,320-token
//! vocabulary): the beams' logits, their log-softmax plus the beams'
//! scores, flattened and cut to the top `width`, split into parent and
//! token, and the parents' mask rows gathered.

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use eta_ir::container::{ChanDType, HostRole, decode};
use eta_ir::registry::{Axis, EngineCaps, OutputDecl};
use eta_ir::types::Dtype;
use eta_ir::validate::{BoundTrace, bind};

use crate::eval::interp::{Instance, NoKernels, PassInputs, Value};

/// Qwen3.5's vocabulary: the logits' width the beam cases bind at.
pub const QWEN35_VOCAB: u32 = 248_320;

/// One case of the contract.
pub struct Case {
    pub name: &'static str,
    pub bound: BoundTrace,
    /// The lanes' rows the fire carries (the beams).
    pub rows: u32,
    /// The width of each row of `logits`.
    pub vocab: u32,
    /// Every seeded channel's cell.
    pub seeds: Vec<(u32, Value)>,
    /// The model's logits, `rows` rows of `vocab`.
    pub logits: Vec<f32>,
    /// The channels the host reads, which the backend's must equal.
    pub readers: Vec<u32>,
}

/// The beam step at widths 2 and 4.
#[must_use]
pub fn beam_stage() -> Vec<Case> {
    vec![
        case(
            "beam w2",
            include_bytes!("../fixtures/beam_stage/qwen35_w2.ptir"),
            QWEN35_VOCAB,
        ),
        case(
            "beam w4",
            include_bytes!("../fixtures/beam_stage/qwen35_w4.ptir"),
            QWEN35_VOCAB,
        ),
    ]
}

fn case(name: &'static str, bytes: &[u8], vocab: u32) -> Case {
    let container = decode(bytes).expect("a fixture decodes");
    let rows = container.channels[0].shape.dims()[0];
    let mut seeds = Vec::new();
    for (index, channel) in container.channels.iter().enumerate() {
        if !channel.seeded {
            continue;
        }
        let cells: u32 = channel
            .shape
            .dims()
            .iter()
            .take(channel.shape.rank() as usize)
            .product();
        let cells = cells as usize;
        let value = match channel.dtype {
            ChanDType::Concrete(Dtype::F32) => {
                Value::F32((0..cells).map(|k| -(k as f32) * 0.75).collect())
            }
            ChanDType::Concrete(Dtype::I32) => Value::I32((0..cells).map(|k| k as i32).collect()),
            ChanDType::Concrete(Dtype::U32) => {
                Value::U32((0..cells).map(|k| (k as u32) % 3).collect())
            }
            ChanDType::Concrete(Dtype::Bool) => Value::Bool((0..cells).map(|k| k % 5 == 0).collect()),
            other => panic!("{name}: channel {index} of {other:?}"),
        };
        seeds.push((index as u32, value));
    }
    let readers = container
        .channels
        .iter()
        .enumerate()
        .filter(|(_, channel)| channel.host_role == HostRole::Reader)
        .map(|(index, _)| index as u32)
        .collect();
    let logits = OutputDecl {
        name: "logits".into(),
        axis: Axis::Row,
        width: vocab,
        dtype: Dtype::F32,
        layers: 0,
        readout: true,
    };
    let caps = EngineCaps {
        lora: true,
        attn_page_mask: true,
    };
    let bound = bind(container, vec![vec![logits]], caps).expect("a fixture binds");
    Case {
        name,
        bound,
        rows,
        vocab,
        seeds,
        logits: plane(rows as usize * vocab as usize),
        readers,
    }
}

/// Logits in [-8, 8): a fixed xorshift stream.
fn plane(len: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 16.0 - 8.0
        })
        .collect()
}

/// A cell as a channel carries it on the wire: a bool a bit, the first in
/// the first byte's lowest bit; any other value its little-endian bytes.
#[must_use]
pub fn wire(value: &Value) -> Vec<u8> {
    match value {
        Value::Bool(bits) => {
            let mut out = vec![0u8; bits.len().div_ceil(8)];
            for (index, &bit) in bits.iter().enumerate() {
                if bit {
                    out[index / 8] |= 1 << (index % 8);
                }
            }
            out
        }
        other => other.to_le_bytes(),
    }
}

impl Case {
    /// The seeds as a bind takes them.
    #[must_use]
    pub fn wire_seeds(&self) -> Vec<(u32, Vec<u8>)> {
        self.seeds
            .iter()
            .map(|(channel, value)| (*channel, wire(value)))
            .collect()
    }

    /// What the host interpreter puts in each of [`Case::readers`], as the
    /// wire carries it.
    #[must_use]
    pub fn expected(&self) -> Vec<Vec<u8>> {
        let mut reference = Instance::new(&self.bound, &self.seeds).expect("the case instantiates");
        let inputs = PassInputs {
            outputs: vec![("logits".to_string(), 0, Value::F32(self.logits.clone()))],
        };
        let stepped = reference
            .step(&self.bound, &inputs, &mut NoKernels)
            .expect("the interpreter steps");
        assert!(stepped.committed, "{}: {:?}", self.name, stepped.missed);
        self.readers
            .iter()
            .map(|&channel| {
                wire(
                    &reference
                        .host_take(&self.bound, channel)
                        .expect("a reader holds a cell"),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The interpreter's answer to the width-2 step, pinned: the top two
    /// scores, their tokens and parents.
    #[test]
    fn the_beam_step_answers_what_it_answered() {
        let cases = beam_stage();
        let w2 = &cases[0];
        assert_eq!((w2.rows, w2.readers.as_slice()), (2, &[2u32, 3, 4, 5][..]));
        let rows = w2.expected();
        let words = |bytes: &[u8]| -> Vec<u32> {
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        assert_eq!(words(&rows[0]), [51_616, 212_655], "tokens");
        assert_eq!(words(&rows[1]), [0, 0], "parents");
        assert_eq!(words(&rows[2]), [0xC11A_7922, 0xC11A_795E], "scores");
    }
}
