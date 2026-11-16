//! The guest-program contract every guest-capable backend runs: a program's
//! stage as an inferlet registered it, the seeds and logits a fire is given,
//! and what the host interpreter answers. A backend registers the case's
//! program, binds it with [`Case::wire_seeds`], fires it over
//! [`Case::buffer`] seated at [`Case::seats`] and reads [`Case::readers`]:
//! each channel's bytes are [`Case::expected`]'s, exactly.
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

/// How a case's logits are seated in the program's output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    /// One lane's seat holds every row, one after another.
    OneLane,
    /// Each row is its own lane's seat, lane `r` the row `r`: an instance
    /// over several lanes of a fire, each lane's readout row wherever the
    /// fire lays it. The rows are laid last first, a row's gap between and
    /// NaN in it, so a backend that read every row from lane 0's seat reads
    /// the wrong ones.
    LaneARow,
}

/// One output seat a case binds: the program's `lane` reads `width` wide
/// rows `row_stride` apart from byte `offset` of [`Case::buffer`], from row
/// `row_offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seat {
    pub lane: u32,
    pub offset: u64,
    pub width: u32,
    pub row_stride: u32,
    pub row_offset: u32,
}

/// One case of the contract.
pub struct Case {
    pub name: &'static str,
    pub binding: Binding,
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

/// The beam step at widths 2 and 4, its logits in one lane's seat and a
/// lane a row.
#[must_use]
pub fn beam_stage() -> Vec<Case> {
    let w2 = include_bytes!("../fixtures/beam_stage/qwen35_w2.ptir");
    let w4 = include_bytes!("../fixtures/beam_stage/qwen35_w4.ptir");
    vec![
        case("beam w2", w2, QWEN35_VOCAB, Binding::OneLane),
        case("beam w4", w4, QWEN35_VOCAB, Binding::OneLane),
        case("beam w2, a lane a row", w2, QWEN35_VOCAB, Binding::LaneARow),
        case("beam w4, a lane a row", w4, QWEN35_VOCAB, Binding::LaneARow),
    ]
}

fn case(name: &'static str, bytes: &[u8], vocab: u32, binding: Binding) -> Case {
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
        binding,
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
    /// Where row `row` of [`Case::logits`] lies in [`Case::buffer`], in
    /// floats.
    fn row_at(&self, row: u32) -> u64 {
        let pitch = u64::from(self.vocab);
        match self.binding {
            Binding::OneLane => u64::from(row) * pitch,
            Binding::LaneARow => u64::from(self.rows - 1 - row) * 2 * pitch,
        }
    }

    /// The f32 buffer a backend uploads and seats the logits in.
    #[must_use]
    pub fn buffer(&self) -> Vec<f32> {
        let vocab = self.vocab as usize;
        let len = (0..self.rows)
            .map(|row| self.row_at(row) as usize + vocab)
            .max()
            .unwrap_or(0);
        let mut out = vec![f32::NAN; len];
        for row in 0..self.rows {
            let (at, from) = (self.row_at(row) as usize, row as usize * vocab);
            out[at..at + vocab].copy_from_slice(&self.logits[from..from + vocab]);
        }
        out
    }

    /// The seats a backend binds the logits to, byte offsets into
    /// [`Case::buffer`].
    #[must_use]
    pub fn seats(&self) -> Vec<Seat> {
        let seat = |lane: u32, row: u32| Seat {
            lane,
            offset: self.row_at(row) * 4,
            width: self.vocab,
            row_stride: self.vocab,
            row_offset: 0,
        };
        match self.binding {
            Binding::OneLane => vec![seat(0, 0)],
            Binding::LaneARow => (0..self.rows).map(|row| seat(row, row)).collect(),
        }
    }

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
        assert_eq!(cases[2].expected(), w2.expected(), "the seats are the backend's alone");
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
