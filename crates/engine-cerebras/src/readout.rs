//! A fire's outputs: one plane per output the fire's classes write, as the
//! fire's program returned it. Buffers live on the host here, so a lane's
//! rows are a slice of the words.

use eta_exec::Value;
use poem_ir::{Dtype, InputAxis};

use crate::device::Buffer;

/// Where one lane's rows sit in a plane: `count` rows from `first`, in each
/// of its layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seat {
    pub first: u32,
    pub count: u32,
}

/// One output of a fire: `[layers.max(1) * rows, width]` of `dtype` (f32 or
/// i32; a bf16 plane holds `bits << 16` per word, which reads as the same
/// f32), a layer's `rows` after another.
pub struct Plane {
    pub name: String,
    pub axis: InputAxis,
    pub readout: bool,
    pub buffer: Buffer,
    pub rows: u32,
    pub width: u32,
    pub layers: u32,
    pub dtype: Dtype,
}

impl Plane {
    /// Rows `rows` of every layer, a layer after another.
    fn rows_of(&self, rows: impl Iterator<Item = u32> + Clone) -> Result<Value, String> {
        let width = self.width as usize;
        let words = self.buffer.words();
        let mut picked = Vec::new();
        for layer in 0..self.layers.max(1) {
            for row in rows.clone() {
                let from = (layer * self.rows + row) as usize * width;
                let span = words.get(from..from + width).ok_or_else(|| {
                    format!(
                        "row {row} of layer {layer} of output `{}`, a plane of {} words",
                        self.name,
                        words.len()
                    )
                })?;
                picked.extend_from_slice(span);
            }
        }
        Ok(match self.dtype {
            Dtype::I32 => Value::I32(picked.into_iter().map(|w| w as i32).collect()),
            _ => Value::F32(picked.into_iter().map(f32::from_bits).collect()),
        })
    }
}

pub type Rows<'a> = (&'a str, u32, Vec<f32>);

pub struct Kept {
    pub planes: Vec<Plane>,
    /// Per real lane, per plane: where the lane's rows sit, when its class
    /// writes the plane.
    pub seats: Vec<Vec<Option<Seat>>>,
    /// Per real lane: the plane read back to the host and the rows of it.
    pub readback: Vec<Option<(usize, Vec<u32>)>>,
    /// The output a decoder's pixels are, and each real lane's.
    pub pixels: (String, Vec<Vec<f32>>),
}

impl std::fmt::Debug for Kept {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let planes: Vec<(&str, u32, u32)> = (self.planes.iter())
            .map(|p| (p.name.as_str(), p.rows, p.width))
            .collect();
        f.debug_struct("Kept")
            .field("planes", &planes)
            .field("seats", &self.seats)
            .finish()
    }
}

impl Kept {
    /// The plane lane `lane` reads output `name` from, and its seat there.
    #[must_use]
    pub fn bound(&self, lane: usize, name: &str) -> Option<(usize, Seat)> {
        let seats = self.seats.get(lane)?;
        self.planes
            .iter()
            .zip(seats)
            .enumerate()
            .find_map(|(at, (plane, seat))| (plane.name == name).then_some((at, (*seat)?)))
    }

    /// Lane `lane`'s rows of output `name`, every layer's.
    pub fn lane_value(&self, lane: usize, name: &str) -> Result<Option<Value>, String> {
        let Some((at, seat)) = self.bound(lane, name) else {
            let pixels =
                (self.pixels.1.get(lane)).filter(|v| !v.is_empty() && self.pixels.0 == name);
            return Ok(pixels.map(|values| Value::F32(values.clone())));
        };
        self.planes[at]
            .rows_of(seat.first..seat.first + seat.count)
            .map(Some)
    }

    /// Lane `lane`'s read-back rows: the output they are of, how many, and
    /// their values.
    pub fn lane_rows(&self, lane: usize) -> Result<Option<Rows<'_>>, String> {
        let Some(Some((at, rows))) = self.readback.get(lane) else {
            return Ok(None);
        };
        let plane = &self.planes[*at];
        let values = match plane.rows_of(rows.iter().copied())? {
            Value::F32(values) => values,
            _ => return Ok(None),
        };
        Ok(Some((plane.name.as_str(), rows.len() as u32, values)))
    }

    /// Whether lane `lane` reads back the readout rows.
    #[must_use]
    pub fn reads_readout(&self, lane: usize) -> bool {
        matches!(self.readback.get(lane), Some(Some((at, _))) if self.planes[*at].readout)
    }
}
