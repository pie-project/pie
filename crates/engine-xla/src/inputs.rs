//! A fire's inputs: host arrays the shell stages, each a root the program
//! reads. Every input lies in one image of u32 words, each array from a
//! whole word: the fire writes the image into a slot of the pool's staging
//! ring (pinned host memory, laid at load) and copies it into the pool's
//! slab at its start (`crate::store::Pools::stage`). Nothing is uploaded per
//! fire.

use crate::handle::Tensor;
use dtype::Dtype;

use crate::trace::{Handles, Root, Source};

/// u32 words a pool row holds.
pub const ROW_WORDS: usize = 128;

/// Rows a u32 tile takes on the device (`T(8,128)`): a staged image is cut
/// from its ring slot in whole tiles.
pub const TILE_ROWS: u32 = 8;

#[derive(Debug, Clone, Default)]
pub struct Inputs {
    /// Every input, back to back, each from a whole word.
    image: Vec<u32>,
    /// Bulk data a maintenance program lands in the pool (a weight atom
    /// run, a parked slot's bytes): written into the bulk ring's next slot
    /// (pinned host memory) when the program runs, not staged, since the
    /// slab is sized for fire inputs.
    bulk: Vec<Vec<u8>>,
    /// What each input is, in staging order: with the fire's windows, what
    /// the traced program's text depends on.
    shapes: Vec<(Dtype, u32, u32)>,
}

impl Inputs {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn root(
        &mut self,
        handles: &Handles,
        dtype: Dtype,
        rows: u32,
        width: u32,
        word: u32,
    ) -> Tensor {
        self.shapes.push((dtype, rows, width));
        handles.root(Root {
            source: Source::Staged { word },
            dtype,
            rows,
            width,
        })
    }

    /// `values` as an i32 `[rows, width]` input (`rows × width` must be its
    /// length); an empty vector lands as one zero row, so every input has a
    /// row.
    pub fn i32s(&mut self, handles: &Handles, values: &[i32], width: u32) -> Tensor {
        let width = width.max(1);
        let word = self.image.len() as u32;
        self.image.extend(values.iter().map(|&v| v as u32));
        if values.is_empty() {
            self.image.extend(std::iter::repeat_n(0, width as usize));
        }
        let len = self.image.len() as u32 - word;
        let rows = len.div_ceil(width);
        self.image.resize((word + rows * width) as usize, 0);
        self.root(handles, Dtype::I32, rows, width, word)
    }

    /// `values` as an f32 input.
    pub fn f32s(&mut self, handles: &Handles, values: &[f32], width: u32) -> Tensor {
        let width = width.max(1);
        let mut v = values.to_vec();
        if v.is_empty() {
            v = vec![0.0; width as usize];
        }
        let rows = (v.len() as u32).div_ceil(width);
        v.resize((rows * width) as usize, 0.0);
        let word = self.image.len() as u32;
        self.image.extend(v.iter().map(|x| x.to_bits()));
        self.root(handles, Dtype::F32, rows, width, word)
    }

    /// Raw rows of `dtype`, `bytes` as its storage form lays them.
    pub fn raw(
        &mut self,
        handles: &Handles,
        dtype: Dtype,
        rows: u32,
        width: u32,
        bytes: Vec<u8>,
    ) -> Tensor {
        let word = self.image.len() as u32;
        let (whole, tail) = bytes.as_chunks::<4>();
        self.image
            .extend(whole.iter().map(|w| u32::from_le_bytes(*w)));
        if !tail.is_empty() {
            let mut last = [0u8; 4];
            last[..tail.len()].copy_from_slice(tail);
            self.image.push(u32::from_le_bytes(last));
        }
        self.root(handles, dtype, rows, width, word)
    }

    /// `bytes` as bulk data, read by its program as a bulk ring slot of
    /// `rows` rows, u32 `[rows, 128]`, the bytes first (see [`Inputs`]'s
    /// `bulk`).
    pub fn bulk(&mut self, handles: &Handles, rows: u32, bytes: Vec<u8>) -> Tensor {
        let input = self.bulk.len() as u32;
        self.shapes.push((Dtype::U32, rows, ROW_WORDS as u32));
        self.bulk.push(bytes);
        handles.root(Root {
            source: Source::Bulk { input },
            dtype: Dtype::U32,
            rows,
            width: ROW_WORDS as u32,
        })
    }

    /// Bulk input `input`'s bytes.
    #[must_use]
    pub fn bulk_bytes(&self, input: u32) -> &[u8] {
        &self.bulk[input as usize]
    }

    /// The image, every input's words.
    #[must_use]
    pub fn image(&self) -> &[u32] {
        &self.image
    }

    /// Pool rows the image takes, whole tiles of [`TILE_ROWS`] (a pinned
    /// host slot is cut in whole tiles).
    #[must_use]
    pub fn rows(&self) -> u32 {
        let rows = self.image.len().div_ceil(ROW_WORDS).max(1) as u32;
        rows.next_multiple_of(TILE_ROWS)
    }

    /// Every input's element type and shape, in staging order: with the
    /// fire's windows, what the traced program's text depends on.
    #[must_use]
    pub fn shapes(&self) -> String {
        let mut out = String::with_capacity(self.shapes.len() * 12 + 8);
        out.push_str(&format!("stage{};", self.rows()));
        for (dtype, rows, width) in &self.shapes {
            out.push_str(&format!("{dtype:?}{rows}x{width};"));
        }
        out
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    /// Bytes the image holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.image.len() as u64 * 4
    }
}
