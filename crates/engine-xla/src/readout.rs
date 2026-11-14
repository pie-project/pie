//! A fire's outputs kept on the device: one plane per output the fire's
//! classes write, as its executable returned it. Guest stages that lowered
//! read them there; a plane's host copy is made once, and only when
//! something on the host reads it.

use std::sync::{Mutex, OnceLock};

use eta_exec::Value;
use poem_ir::{Dtype, InputAxis};

use crate::pjrt::Buffer;

/// Where one lane's rows sit in a plane: `count` rows from `first`, in each
/// of its layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seat {
    pub first: u32,
    pub count: u32,
}

/// One output of a fire: `[layers.max(1) * rows, width]` of `dtype` (f32 or
/// i32), a layer's `rows` after another.
pub struct Plane {
    pub name: String,
    pub axis: InputAxis,
    pub readout: bool,
    pub buffer: Buffer,
    pub rows: u32,
    pub width: u32,
    pub layers: u32,
    pub dtype: Dtype,
    host: OnceLock<Result<Value, String>>,
}

impl Plane {
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        axis: InputAxis,
        readout: bool,
        buffer: Buffer,
        rows: u32,
        width: u32,
        layers: u32,
        dtype: Dtype,
    ) -> Plane {
        Plane {
            name,
            axis,
            readout,
            buffer,
            rows,
            width,
            layers,
            dtype,
            host: OnceLock::new(),
        }
    }

    fn on_host(&self) -> Result<&Value, String> {
        self.host
            .get_or_init(|| {
                let words = download_words(&self.buffer)?;
                Ok(match self.dtype {
                    Dtype::I32 => Value::I32(words.iter().map(|w| w.to_bits() as i32).collect()),
                    _ => Value::F32(words),
                })
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Rows `rows` of every layer, a layer after another.
    fn rows_of(&self, rows: impl Iterator<Item = u32> + Clone) -> Result<Value, String> {
        let width = self.width as usize;
        let at = |row: u32, layer: u32| (layer * self.rows + row) as usize * width;
        let layers = 0..self.layers.max(1);
        Ok(match self.on_host()? {
            Value::I32(all) => Value::I32(
                layers
                    .flat_map(|l| rows.clone().map(move |r| (r, l)))
                    .flat_map(|(r, l)| all[at(r, l)..at(r, l) + width].iter().copied())
                    .collect(),
            ),
            Value::F32(all) => Value::F32(
                layers
                    .flat_map(|l| rows.clone().map(move |r| (r, l)))
                    .flat_map(|(r, l)| all[at(r, l)..at(r, l) + width].iter().copied())
                    .collect(),
            ),
            _ => unreachable!("a plane is f32 or i32"),
        })
    }
}

impl Drop for Plane {
    fn drop(&mut self) {
        if let Some(Ok(Value::F32(words))) = self.host.take() {
            recycle(words);
        }
    }
}

/// Host buffers of planes dropped, kept for the next copy out: a fresh
/// buffer of a wide plane is pages the kernel zeroes on first touch (v6e
/// qwen35-d0.8b decode of 256 lanes: a 254 MB logits plane a readout).
static SPARE: Mutex<Vec<Vec<f32>>> = Mutex::new(Vec::new());

/// The most spare host buffers kept.
const SPARES: usize = 2;

fn recycle(words: Vec<f32>) {
    if let Ok(mut spare) = SPARE.lock()
        && spare.len() < SPARES
    {
        spare.push(words);
    }
}

/// `buffer` copied out as 32-bit words, into a spare host buffer when one
/// is kept (the largest; its tail cut off or grown).
fn download_words(buffer: &Buffer) -> Result<Vec<f32>, String> {
    let mut words = SPARE
        .lock()
        .ok()
        .and_then(|mut spare| {
            let at = (0..spare.len()).max_by_key(|&i| spare[i].len())?;
            Some(spare.swap_remove(at))
        })
        .unwrap_or_default();
    buffer
        .download_words(&mut words)
        .map_err(|e| e.to_string())?;
    Ok(words)
}

/// The rows of the readout plane the lanes reading back to the host read,
/// gathered by the fire into a short plane of their own: copied out in
/// place of the whole plane when a lane's rows are all in it.
pub struct HostRows {
    /// The readout plane's index in `Kept::planes`.
    pub plane: usize,
    pub buffer: Buffer,
    /// Per real lane: where its readout rows start in `buffer`.
    pub at: Vec<Option<u32>>,
    host: OnceLock<Result<Vec<f32>, String>>,
}

impl HostRows {
    #[must_use]
    pub fn new(plane: usize, buffer: Buffer, at: Vec<Option<u32>>) -> HostRows {
        HostRows {
            plane,
            buffer,
            at,
            host: OnceLock::new(),
        }
    }

    fn on_host(&self) -> Result<&[f32], String> {
        self.host
            .get_or_init(|| download_words(&self.buffer))
            .as_ref()
            .map(Vec::as_slice)
            .map_err(Clone::clone)
    }
}

impl Drop for HostRows {
    fn drop(&mut self) {
        if let Some(Ok(words)) = self.host.take() {
            recycle(words);
        }
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
    /// The output a decoder's pixels are, and each real lane's on the host.
    pub pixels: (String, Vec<Vec<f32>>),
    /// The host-reading lanes' readout rows, when the fire gathered them.
    pub host: Option<HostRows>,
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
    /// The plane lane `lane` reads output `name` from, and its seat there
    /// (coalescing consecutive lanes whose rows form one contiguous run).
    #[must_use]
    pub fn bound(&self, lane: usize, name: &str) -> Option<(usize, Seat)> {
        let seats = self.seats.get(lane)?;
        let (at, first_seat) = self
            .planes
            .iter()
            .zip(seats)
            .enumerate()
            .find_map(|(at, (plane, seat))| (plane.name == name).then_some((at, (*seat)?)))?;
        let mut count = first_seat.count;
        for next_seats in &self.seats[lane + 1..] {
            let Some(next) = next_seats.get(at).copied().flatten() else {
                break;
            };
            if next.first != first_seat.first + count {
                break;
            }
            count += next.count;
        }
        Some((
            at,
            Seat {
                first: first_seat.first,
                count,
            },
        ))
    }

    /// Lane `lane`'s rows of output `name` on the host, every layer's.
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

    /// Lane `lane`'s read-back rows on the host: the output they are of, how
    /// many, and their values.
    pub fn lane_rows(&self, lane: usize) -> Result<Option<Rows<'_>>, String> {
        let Some(Some((at, rows))) = self.readback.get(lane) else {
            return Ok(None);
        };
        let plane = &self.planes[*at];
        if let Some(host) = self.host.as_ref().filter(|host| host.plane == *at)
            && let Some(Some(start)) = host.at.get(lane)
            && plane.layers <= 1
        {
            // The lane's rows are the readout rows of its seat, in order, as
            // the fire laid them in the short plane.
            let width = plane.width as usize;
            let all = host.on_host()?;
            let from = *start as usize * width;
            let values = all[from..from + rows.len() * width].to_vec();
            return Ok(Some((plane.name.as_str(), rows.len() as u32, values)));
        }
        let values = match plane.rows_of(rows.iter().copied())? {
            Value::F32(values) => values,
            other => {
                return Err(format!(
                    "lane {lane} reads back output `{}`, a {:?} plane; a lane's readout carries f32",
                    plane.name,
                    other.dtype()
                ));
            }
        };
        Ok(Some((plane.name.as_str(), rows.len() as u32, values)))
    }

    /// Whether lane `lane` reads back the readout rows.
    #[must_use]
    pub fn reads_readout(&self, lane: usize) -> bool {
        matches!(self.readback.get(lane), Some(Some((at, _))) if self.planes[*at].readout)
    }
}
