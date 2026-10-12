//! What a still becomes before a tower reads it: framed to the size the
//! model's package states, then cut into patches of raw RGB bytes in block
//! order. Every model-specific choice is the package's; this is the cut.

use std::fmt;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Grid {
    pub t: u32,
    pub h: u32,
    pub w: u32,
}

impl Grid {
    #[must_use]
    pub const fn still(h: u32, w: u32) -> Grid {
        Grid { t: 1, h, w }
    }

    #[must_use]
    pub const fn cells(&self) -> u32 {
        self.t * self.h * self.w
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Budget {
    #[default]
    Still,
    VideoFrame,
}

impl Budget {
    #[must_use]
    pub const fn word(self) -> &'static str {
        match self {
            Budget::Still => "still",
            Budget::VideoFrame => "video",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb8 {
    pub h: u32,
    pub w: u32,
    pub data: Vec<u8>,
}

impl Rgb8 {
    pub fn new(h: u32, w: u32, data: Vec<u8>) -> Result<Rgb8> {
        if w == 0 || h == 0 {
            return Err(Fault::Empty(format!(
                "a frame of {h} x {w} pixels occupies no rows"
            )));
        }
        let owed = h as usize * w as usize * 3;
        if data.len() != owed {
            return Err(Fault::Decode(format!(
                "a {h} x {w} RGB frame is {owed} bytes and {} arrived",
                data.len()
            )));
        }
        Ok(Rgb8 { h, w, data })
    }
}

pub type Resample = fn(&Rgb8, u32, u32) -> Rgb8;

/// A still as the tower reads it: `rows` patches of `payload` bytes each
/// `3 · patch²` long, their `(row, col)` grid `positions`, and how it is
/// spelled among the tokens.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EncodedSpan {
    pub token_count: u32,
    pub position_span: u32,
    pub grid: Grid,
    pub patch_grid: Grid,
    pub uses_mrope: bool,
    pub payload: Vec<u8>,
    pub rows: u32,
    pub positions: Vec<u32>,
    pub prefix: Vec<u32>,
    pub placeholder: u32,
    pub suffix: Vec<u32>,
}

impl EncodedSpan {
    #[must_use]
    pub fn tokens(&self) -> Vec<u32> {
        let mut out =
            Vec::with_capacity(self.prefix.len() + self.token_count as usize + self.suffix.len());
        out.extend_from_slice(&self.prefix);
        out.extend(std::iter::repeat_n(
            self.placeholder,
            self.token_count as usize,
        ));
        out.extend_from_slice(&self.suffix);
        out
    }

    pub fn spell_with(&mut self, prefix: Vec<u32>, placeholder: u32, suffix: Vec<u32>) {
        self.prefix = prefix;
        self.placeholder = placeholder;
        self.suffix = suffix;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fault {
    NoVisionFrontEnd { model: String },
    NoAudioFrontEnd { model: String },
    Frame(String),
    Decode(String),
    Empty(String),
}

impl Fault {
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Fault::NoVisionFrontEnd { .. } => "NoVisionFrontEnd",
            Fault::NoAudioFrontEnd { .. } => "NoAudioFrontEnd",
            Fault::Frame(_) => "Frame",
            Fault::Decode(_) => "Decode",
            Fault::Empty(_) => "Empty",
        }
    }
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fault::NoVisionFrontEnd { model } => {
                write!(f, "NoVisionFrontEnd: `{model}` reads no stills")
            }
            Fault::NoAudioFrontEnd { model } => {
                write!(f, "NoAudioFrontEnd: `{model}` reads no audio")
            }
            Fault::Frame(why) => write!(f, "Frame: {why}"),
            Fault::Decode(why) => write!(f, "Decode: {why}"),
            Fault::Empty(why) => write!(f, "Empty: {why}"),
        }
    }
}

impl std::error::Error for Fault {}

pub type Result<T> = std::result::Result<T, Fault>;

/// How a model cuts a framed still: `patch` × `patch` pixel patches, taken
/// in `block` × `block` blocks so the tower's row folds land on them, each
/// block one token; `mrope` places the tokens on the merged grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Image {
    pub patch: u32,
    pub block: u32,
    pub mrope: bool,
}

impl Image {
    /// `src` resampled to `framed` and cut: each patch row is its pixels
    /// in row-major order, three bytes each.
    pub fn encode(
        &self,
        src: &Rgb8,
        framed: (u32, u32),
        resample: Resample,
    ) -> Result<EncodedSpan> {
        let (p, b) = (self.patch, self.block);
        let (fh, fw) = framed;
        if fh == 0 || fw == 0 || !fh.is_multiple_of(p * b) || !fw.is_multiple_of(p * b) {
            return Err(Fault::Frame(format!(
                "a {} x {} image framed to {fh} x {fw}, which is not a whole number of {} x {} \
                 blocks of {p} x {p} patches",
                src.h,
                src.w,
                p * b,
                p * b
            )));
        }
        let (gh, gw) = (fh / p, fw / p);
        let (bh, bw) = (gh / b, gw / b);
        let token_count = bh * bw;
        if token_count == 0 {
            return Err(Fault::Empty(format!(
                "a {} x {} image framed to a {gh} x {gw} patch grid, fewer than one {b} x {b} \
                 block, so it occupies no token rows",
                src.h, src.w
            )));
        }
        let resized = resample(src, fh, fw);
        let (payload, positions) = cut(&resized.data, fw, p, b, gh, gw);
        Ok(EncodedSpan {
            token_count,
            position_span: if self.mrope { bh.max(bw) } else { token_count },
            grid: Grid::still(bh, bw),
            patch_grid: Grid::still(gh, gw),
            uses_mrope: self.mrope,
            payload,
            rows: gh * gw,
            positions,
            prefix: Vec::new(),
            placeholder: 0,
            suffix: Vec::new(),
        })
    }
}

/// The patches of a `gh` × `gw` grid over an image `w` pixels wide, in
/// block order: block by block row-major, patch by patch row-major within.
fn cut(rgb: &[u8], w: u32, p: u32, b: u32, gh: u32, gw: u32) -> (Vec<u8>, Vec<u32>) {
    let (p, b, w) = (p as usize, b as usize, w as usize);
    let (gh, gw) = (gh as usize, gw as usize);
    let patch_bytes = p * p * 3;
    let mut payload = Vec::with_capacity(gh * gw * patch_bytes);
    let mut positions = Vec::with_capacity(gh * gw * 2);
    for bi in 0..gh / b {
        for bj in 0..gw / b {
            for i in 0..b {
                for j in 0..b {
                    let (pr, pc) = (bi * b + i, bj * b + j);
                    for r in 0..p {
                        let at = ((pr * p + r) * w + pc * p) * 3;
                        payload.extend_from_slice(&rgb[at..at + p * 3]);
                    }
                    #[allow(clippy::cast_possible_truncation)]
                    positions.extend_from_slice(&[pr as u32, pc as u32]);
                }
            }
        }
    }
    (payload, positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixels(h: u32, w: u32) -> Rgb8 {
        let data = (0..h * w)
            .flat_map(|i| {
                let (y, x) = (i / w, i % w);
                #[allow(clippy::cast_possible_truncation)]
                [y as u8, x as u8, 7]
            })
            .collect();
        Rgb8::new(h, w, data).expect("a frame")
    }

    #[test]
    fn front_every_case() {
        patches_land_in_block_order_with_their_positions();
        a_frame_that_is_no_whole_block_is_refused();
    }

    fn patches_land_in_block_order_with_their_positions() {
        let image = Image {
            patch: 2,
            block: 2,
            mrope: true,
        };
        let src = pixels(4, 8);
        let span = image
            .encode(&src, (4, 8), |s, _, _| s.clone())
            .expect("encodes");
        assert_eq!((span.rows, span.token_count), (8, 2));
        assert_eq!(span.grid, Grid::still(1, 2));
        assert_eq!(span.position_span, 2);
        assert_eq!(span.payload.len(), 8 * 2 * 2 * 3);
        // The first block's patches: (0,0), (0,1), (1,0), (1,1); then the
        // second block's: (0,2), (0,3), (1,2), (1,3).
        assert_eq!(
            span.positions,
            vec![0, 0, 0, 1, 1, 0, 1, 1, 0, 2, 0, 3, 1, 2, 1, 3]
        );
        // The third patch is rows 2..4 of columns 0..2: its first pixel is
        // (y=2, x=0).
        assert_eq!(&span.payload[2 * 12..2 * 12 + 3], &[2, 0, 7]);
        let empty = Rgb8::new(0, 4, Vec::new()).expect_err("zero side");
        assert_eq!(empty.name(), "Empty");
    }

    fn a_frame_that_is_no_whole_block_is_refused() {
        let image = Image {
            patch: 2,
            block: 2,
            mrope: false,
        };
        let why = image
            .encode(&pixels(4, 4), (4, 6), |s, _, _| s.clone())
            .expect_err("six is no multiple of four");
        assert_eq!(why.name(), "Frame");
    }
}
