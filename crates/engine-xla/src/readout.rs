//! A fire's readout kept on the device: the logits rows (and the draft
//! head's) every lane asked for, as the fire's executable returned them.
//! Guest stages that lowered read them there; the host copy is made once,
//! and only when something on the host reads a row.

use std::sync::OnceLock;

use crate::pjrt::Buffer;

/// Where one lane's rows sit in the readout: `count` rows from `first`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seat {
    pub first: u32,
    pub count: u32,
}

pub struct Kept {
    /// `f32 [rows, width]`.
    pub logits: Buffer,
    pub rows: u32,
    pub width: u32,
    /// The draft head's rows, `f32 [rows, mtp_width]`, laid out as `logits`.
    pub mtp: Option<(Buffer, u32)>,
    /// Per real lane: its first row and how many.
    pub layout: Vec<(u32, u32)>,
    /// Whether the planes hold f32 (else bf16): guest stages read f32 ones.
    pub f32: bool,
    host: OnceLock<Result<Vec<f32>, String>>,
    host_mtp: OnceLock<Result<Vec<f32>, String>>,
}

impl std::fmt::Debug for Kept {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kept")
            .field("rows", &self.rows)
            .field("width", &self.width)
            .field("mtp", &self.mtp.as_ref().map(|(_, w)| *w))
            .field("layout", &self.layout)
            .field("on_host", &self.host.get().is_some())
            .finish()
    }
}

fn floats(raw: &[u8], f32: bool) -> Vec<f32> {
    if f32 {
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        raw.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
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
        let f32 = logits
            .element_type()
            .is_ok_and(|t| t == crate::pjrt::ElementType::F32);
        Kept {
            f32,
            logits,
            rows,
            width,
            mtp,
            layout,
            host: OnceLock::new(),
            host_mtp: OnceLock::new(),
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

    /// Whether the host copy was made.
    #[must_use]
    pub fn on_host(&self) -> bool {
        self.host.get().is_some()
    }

    fn plane(&self) -> Result<&[f32], String> {
        self.host
            .get_or_init(|| {
                self.logits
                    .download()
                    .map(|raw| floats(&raw, self.f32))
                    .map_err(|e| e.to_string())
            })
            .as_deref()
            .map_err(Clone::clone)
    }

    fn mtp_plane(&self) -> Result<Option<&[f32]>, String> {
        let Some((buffer, _)) = &self.mtp else {
            return Ok(None);
        };
        self.host_mtp
            .get_or_init(|| {
                buffer
                    .download()
                    .map(|raw| floats(&raw, self.f32))
                    .map_err(|e| e.to_string())
            })
            .as_deref()
            .map(Some)
            .map_err(Clone::clone)
    }

    /// Lane `lane`'s logits rows on the host.
    pub fn lane_rows(&self, lane: usize) -> Result<Vec<f32>, String> {
        let Some(&(first, count)) = self.layout.get(lane) else {
            return Ok(Vec::new());
        };
        let width = self.width as usize;
        let plane = self.plane()?;
        let from = first as usize * width;
        Ok(plane[from..from + count as usize * width].to_vec())
    }

    /// Lane `lane`'s draft rows on the host (empty when there is no draft
    /// head).
    pub fn lane_drafts(&self, lane: usize) -> Result<Vec<f32>, String> {
        let (Some(&(first, count)), Some((_, width))) = (self.layout.get(lane), &self.mtp) else {
            return Ok(Vec::new());
        };
        let width = *width as usize;
        let Some(plane) = self.mtp_plane()? else {
            return Ok(Vec::new());
        };
        let from = first as usize * width;
        Ok(plane[from..from + count as usize * width].to_vec())
    }
}
