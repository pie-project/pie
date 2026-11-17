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

/// What a platform documents of the f32 functions a guest step lowers
/// to: the most ulps each answers off the exact value. Its basic
/// arithmetic (+, -, *, /) is correctly rounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Accuracy {
    pub exp_ulps: u32,
    pub log_ulps: u32,
}

impl Accuracy {
    /// Metal's precise math (`MTLMathMode::Safe`, precise functions, which
    /// guest libraries compile with): Metal Shading Language Specification
    /// 4.1, Table 8.1, `exp` <= 4 ulp, `log` <= 4 ulp, `x + y`, `x - y`,
    /// `x * y`, `x / y` correctly rounded.
    pub const METAL_PRECISE: Accuracy = Accuracy {
        exp_ulps: 4,
        log_ulps: 4,
    };

    /// The interpreter's: Rust's `f32::exp` and `f32::ln`, the platform's
    /// libm, faithfully rounded (within 1 ulp) on the hosts the contract
    /// runs on.
    pub const HOST: Accuracy = Accuracy {
        exp_ulps: 1,
        log_ulps: 1,
    };
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

    /// The most an f32 cell this step writes may stand off the
    /// interpreter's, a backend whose math is `device`'s answering it:
    /// equal up to stated rounding (the guest-lowering contract: float math
    /// agrees within rounding, while integer, comparison, ordering, mask and
    /// RNG-bit results are exact), derived from what each side documents,
    /// not picked.
    ///
    /// The step is a log-softmax over `vocab` (a score is `x - max -
    /// log(S)`, `S` the sum of `exp(x - max)`, then added to a carried
    /// score). Both sides sum in the canonical order (32-lane trees, five
    /// adds a level), so the sum's own rounding differs only through its
    /// inputs and each add's last bit. Per side:
    /// - each `exp` is off by its ulps, relative to the term;
    /// - each of the tree's adds on the path rounds once (half an ulp,
    ///   relative);
    /// - `log(S)` is off by its ulps of `log(S)`, at most `log(vocab)`
    ///   since `S` lies in `[1, vocab]`, plus the sum's relative error;
    /// - the three correctly rounded adds after it round once each.
    #[must_use]
    pub fn f32_bound(&self, device: Accuracy, cell: f32) -> f32 {
        let host = Accuracy::HOST;
        let ulp = |x: f64| -> f64 { 2f64.powi(x.abs().max(f64::from(f32::MIN_POSITIVE)).log2().floor() as i32 - 23) };
        let unit = 2f64.powi(-24);
        let levels = f64::from(self.vocab.max(2)).log(32.0).ceil();
        let depth = 5.0 * levels;
        let exp = f64::from(device.exp_ulps + host.exp_ulps) * 2f64.powi(-23);
        let sum = exp + 2.0 * depth * unit;
        let log = f64::from(device.log_ulps + host.log_ulps) * ulp(f64::from(self.vocab).ln()) + sum;
        (log + 6.0 * ulp(f64::from(cell))) as f32
    }

    /// Whether `got` answers `want` for channel `channel`, as a backend of
    /// `device`'s math writes it `steps` steps in (each side carrying its own
    /// scores, so the bound adds up a step at a time): an f32 cell within
    /// `steps` times [`Case::f32_bound`] of the interpreter's, every other
    /// cell (ids, parents, masks) byte for byte. The one check every
    /// backend's contract test makes.
    ///
    /// # Errors
    ///
    /// Each cell that disagrees, by index, with both values and the bound.
    pub fn agrees(
        &self,
        channel: u32,
        got: &[u8],
        want: &[u8],
        device: Accuracy,
        steps: u32,
    ) -> Result<(), String> {
        if got.len() != want.len() {
            return Err(format!(
                "{}: channel {channel}: {} bytes, the interpreter's {}",
                self.name,
                got.len(),
                want.len()
            ));
        }
        let floats = matches!(
            self.bound.container.channels.get(channel as usize).map(|c| c.dtype),
            Some(ChanDType::Concrete(Dtype::F32))
        );
        if !floats {
            return if got == want {
                Ok(())
            } else {
                Err(format!("{}: channel {channel}: {got:02x?} != {want:02x?}", self.name))
            };
        }
        let cell = |bytes: &[u8]| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let wrong: Vec<String> = got
            .chunks_exact(4)
            .zip(want.chunks_exact(4))
            .enumerate()
            .filter_map(|(at, (g, w))| {
                let (g, w) = (cell(g), cell(w));
                let bound = self.f32_bound(device, w) * steps.max(1) as f32;
                let off = (g - w).abs();
                (!(off <= bound)).then(|| format!("cell {at}: {g} ({:#010x}) vs {w} ({:#010x}), off {off:e} past {bound:e}", g.to_bits(), w.to_bits()))
            })
            .collect();
        if wrong.is_empty() {
            Ok(())
        } else {
            Err(format!("{}: channel {channel}: {}", self.name, wrong.join("; ")))
        }
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

/// One bf16 step at `x`'s magnitude: the spacing of bf16 values in `x`'s
/// binade, `2^(exponent - 7)`.
#[must_use]
pub fn bf16_step(x: f32) -> f32 {
    let x = x.abs();
    if x == 0.0 || !x.is_finite() {
        return 0.0;
    }
    (2f32).powi(x.log2().floor() as i32 - 7)
}

/// How far two paths' logit rows at one step moved apart: the drift a
/// parting is read against ([`near_tie`]).
#[derive(Clone, Copy, Debug)]
pub enum Drift<'a> {
    /// Both paths' full logit rows at the parting's step, the same history
    /// before it.
    Rows { a: &'a [f32], b: &'a [f32] },
    /// No row to measure: the per-logit drift is stated, `steps` bf16 steps
    /// at the top logit `top`, the depth the logits passed through
    /// propagating each layer's rounding up to them.
    Depth { top: f32, steps: f32 },
}

/// A parting of two runs read under contract A: what was measured and the
/// verdict.
#[derive(Clone, Debug, PartialEq)]
pub struct Parting {
    /// The decision's gap at the parting: its top-1 less its top-2, in
    /// logit units.
    pub gap: f32,
    /// The largest per-logit drift between the two paths' rows (or the
    /// stated one, [`Drift::Depth`]).
    pub drift_max: f32,
    /// The drift's root mean square over the row (the stated drift where no
    /// row was measured).
    pub drift_rms: f32,
    /// Each contested token's rank by its |drift| among the row's: how many
    /// entries moved strictly more (0 is the row's largest move). Empty
    /// where no row was measured.
    pub ranks: Vec<usize>,
    /// The gap's bound: twice the drift.
    pub bound: f32,
    /// Whether the drift is diffuse: no contested token moved more than
    /// every other entry of the row. `None` where no row was measured.
    pub diffuse: Option<bool>,
    /// The verdict: a near-tie contract A admits.
    pub tie: bool,
}

/// Whether two runs that part at one step part on a near-tie contract A
/// admits ("equal up to stated bf16 near-ties"), the decision there going
/// to a different one of `contested` (each path's pick) by `gap` (the
/// decision's top-1 less its top-2, in logit units).
///
/// The rule, as stated for the contract:
/// 1. The gap is within twice the largest per-logit drift between the two
///    paths at that step, measured over the full row: each of the two
///    contested logits may round its own way by up to that drift, so the
///    pair's difference can move by twice it.
/// 2. The drift is diffuse: a contested token's |drift| is an outlier when
///    it exceeds the |drift| of every entry of the row other than the
///    contested pair, i.e. the move that decided the parting is larger than
///    any move rounding made anywhere else in the row. A concentrated move
///    on the contested tokens is a bug even within the gap bound: rounding
///    has no reason to favour the two tokens the decision weighed.
///
/// Without a row ([`Drift::Depth`]) only rule 1 is read, against twice the
/// stated drift (2 x bf16 step x the propagated depth bound), and
/// `diffuse` is `None`.
#[must_use]
pub fn near_tie(gap: f32, contested: &[usize], drift: Drift<'_>) -> Parting {
    match drift {
        Drift::Depth { top, steps } => {
            let drift = bf16_step(top) * steps;
            Parting {
                gap,
                drift_max: drift,
                drift_rms: drift,
                ranks: Vec::new(),
                bound: 2.0 * drift,
                diffuse: None,
                tie: gap <= 2.0 * drift,
            }
        }
        Drift::Rows { a, b } => {
            let moved: Vec<f32> = a.iter().zip(b).map(|(x, y)| (y - x).abs()).collect();
            let drift_max = moved.iter().copied().fold(0.0f32, f32::max);
            let drift_rms = if moved.is_empty() {
                0.0
            } else {
                let squares: f64 = moved.iter().map(|d| f64::from(*d) * f64::from(*d)).sum();
                (squares / moved.len() as f64).sqrt() as f32
            };
            let at = |token: usize| moved.get(token).copied().unwrap_or(f32::INFINITY);
            let ranks = contested
                .iter()
                .map(|&token| moved.iter().filter(|d| **d > at(token)).count())
                .collect();
            let elsewhere = moved
                .iter()
                .enumerate()
                .filter(|(token, _)| !contested.contains(token))
                .map(|(_, d)| *d)
                .fold(0.0f32, f32::max);
            let diffuse = contested.iter().all(|&token| at(token) <= elsewhere);
            let bound = 2.0 * drift_max;
            Parting {
                gap,
                drift_max,
                drift_rms,
                ranks,
                bound,
                diffuse: Some(diffuse),
                tie: gap <= bound && diffuse,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device's score one ulp off the interpreter's agrees, as the
    /// documented accuracy allows (an M5 Pro's precise exp / log answered
    /// 0xc11a7965 where the host answered 0xc11a7966); a score off past the
    /// bound does not, nor does a token id off by one.
    #[test]
    fn a_score_within_the_documented_rounding_agrees_and_an_id_is_exact() {
        let cases = beam_stage();
        let w2 = &cases[2];
        let scores = |words: [u32; 2]| -> Vec<u8> {
            words.iter().flat_map(|w| f32::from_bits(*w).to_le_bytes()).collect()
        };
        let want = scores([0xC11A_7966, 0xC11A_7966]);
        let float = w2
            .bound
            .container
            .channels
            .iter()
            .position(|c| matches!(c.dtype, ChanDType::Concrete(Dtype::F32)))
            .expect("a score channel") as u32;
        let near = scores([0xC11A_7965, 0xC11A_7965]);
        assert_eq!(w2.agrees(float, &near, &want, Accuracy::METAL_PRECISE, 1), Ok(()));
        let bound = w2.f32_bound(Accuracy::METAL_PRECISE, f32::from_bits(0xC11A_7966));
        assert!(bound < 1e-4, "the bound is rounding, not slack: {bound:e}");
        let far = scores([0xC11A_7966, (f32::from_bits(0xC11A_7966) + 4.0 * bound).to_bits()]);
        assert!(w2.agrees(float, &far, &want, Accuracy::METAL_PRECISE, 1).is_err());
        let ids = w2.readers[0];
        let id = 51_616u32.to_le_bytes().to_vec();
        let off = 51_617u32.to_le_bytes().to_vec();
        assert!(w2.agrees(ids, &off, &id, Accuracy::METAL_PRECISE, 1).is_err());
        assert_eq!(w2.agrees(ids, &id, &id, Accuracy::METAL_PRECISE, 1), Ok(()));
    }

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

    /// The 48_6 parting (Qwen3-0.6B on Vulkan, a 48-row tiled prefill
    /// against a 16-row tail over cached pages): a 0.092 gap, the row
    /// moving up to 3 steps everywhere and the contested pair 1 and 2: a
    /// near-tie. The same gap with the pair alone moving is a bug, and a gap
    /// past twice the drift is no tie.
    #[test]
    fn a_parting_is_a_tie_on_diffuse_drift_within_twice_its_largest_move() {
        let step = bf16_step(14.0625);
        assert_eq!(step, 0.0625);
        let mut a = vec![0.0f32; 64];
        let mut b = a.clone();
        for (at, x) in a.iter_mut().enumerate() {
            *x = 10.0 + at as f32 / 8.0;
        }
        for (at, (x, y)) in a.iter().zip(b.iter_mut()).enumerate() {
            *y = x + step * [0.0, 1.0, -1.0, 2.0][at % 4];
        }
        b[40] = a[40] + 3.0 * step;
        // The contested pair: 943 and 729's moves, -1 and +2 steps.
        b[5] = a[5] - step;
        b[7] = a[7] + 2.0 * step;
        let parting = near_tie(0.092, &[5, 7], Drift::Rows { a: &a, b: &b });
        assert!(parting.tie, "{parting:?}");
        assert_eq!(parting.drift_max, 3.0 * step);
        assert_eq!(parting.diffuse, Some(true));
        assert!(parting.ranks.iter().all(|rank| *rank > 0), "{:?}", parting.ranks);
        assert!(!near_tie(0.40, &[5, 7], Drift::Rows { a: &a, b: &b }).tie, "past twice the drift");

        // The decision's pair alone moving past every other entry: structured.
        let mut c = a.clone();
        c[3] = a[3] + step;
        c[5] = a[5] - 2.0 * step;
        c[7] = a[7] + 2.0 * step;
        let concentrated = near_tie(0.092, &[5, 7], Drift::Rows { a: &a, b: &c });
        assert_eq!(concentrated.diffuse, Some(false));
        assert!(!concentrated.tie, "{concentrated:?}");
        assert_eq!(concentrated.ranks, vec![0, 0]);

        let stated = near_tie(0.092, &[5, 7], Drift::Depth { top: 14.0625, steps: 1.0 });
        assert_eq!((stated.bound, stated.diffuse, stated.tie), (0.125, None, true));
    }
}
