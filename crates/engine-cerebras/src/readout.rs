//! A fire's readout: the logits rows (and the draft head's) every lane asked
//! for, as the fire's program returned them. Buffers live on the host here,
//! so a lane's rows are a slice of the words.

use crate::device::Buffer;

/// Where one lane's rows sit in the readout: `count` rows from `first`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seat {
    pub first: u32,
    pub count: u32,
}

pub struct Kept {
    /// `f32 [rows, width]` (a bf16 plane holds `bits << 16` per word, which
    /// reads as the same f32).
    pub logits: Buffer,
    pub rows: u32,
    pub width: u32,
    /// The draft head's rows, `[rows, mtp_width]`, laid out as `logits`.
    pub mtp: Option<(Buffer, u32)>,
    /// Per real lane: its first row and how many.
    pub layout: Vec<(u32, u32)>,
    /// Every plane reads as f32 on this backend.
    pub f32: bool,
}

impl std::fmt::Debug for Kept {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kept")
            .field("rows", &self.rows)
            .field("width", &self.width)
            .field("mtp", &self.mtp.as_ref().map(|(_, w)| *w))
            .field("layout", &self.layout)
            .finish()
    }
}

impl Kept {
    #[must_use]
    pub fn new(
        logits: Buffer,
        rows: u32,
        width: u32,
        mtp: Option<(Buffer, u32)>,
        layout: Vec<(u32, u32)>,
    ) -> Kept {
        Kept {
            f32: true,
            logits,
            rows,
            width,
            mtp,
            layout,
        }
    }

    /// Where lanes `lane..lane + lanes` read, when their rows are one run.
    #[must_use]
    pub fn seat(&self, lane: usize, lanes: usize) -> Option<Seat> {
        let span = self
            .layout
            .get(lane..(lane + lanes.max(1)).min(self.layout.len()))?;
        let (first, _) = *span.first()?;
        let mut next = first;
        for &(at, count) in span {
            if at != next {
                return None;
            }
            next += count;
        }
        (next > first).then_some(Seat {
            first,
            count: next - first,
        })
    }

    /// The readout is always on the host here.
    #[must_use]
    pub fn on_host(&self) -> bool {
        true
    }

    fn rows_of(plane: &Buffer, first: u32, count: u32, width: usize) -> Result<Vec<f32>, String> {
        let words = plane.words();
        let from = first as usize * width;
        let to = from + count as usize * width;
        let span = words.get(from..to).ok_or_else(|| {
            format!(
                "readout rows {first}..{} of a plane holding {} rows",
                first + count,
                words.len() / width.max(1)
            )
        })?;
        Ok(span.iter().map(|w| f32::from_bits(*w)).collect())
    }

    /// Lane `lane`'s logits rows on the host.
    pub fn lane_rows(&self, lane: usize) -> Result<Vec<f32>, String> {
        let Some(&(first, count)) = self.layout.get(lane) else {
            return Ok(Vec::new());
        };
        Self::rows_of(&self.logits, first, count, self.width as usize)
    }

    /// Lane `lane`'s draft rows on the host (empty when there is no draft
    /// head).
    pub fn lane_drafts(&self, lane: usize) -> Result<Vec<f32>, String> {
        let (Some(&(first, count)), Some((plane, width))) = (self.layout.get(lane), &self.mtp)
        else {
            return Ok(Vec::new());
        };
        Self::rows_of(plane, first, count, *width as usize)
    }
}
