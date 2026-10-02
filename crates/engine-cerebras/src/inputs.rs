//! A fire's inputs: host arrays the shell stages, each a root the program
//! reads as a parameter. Only what the traced program names is uploaded.

use dtype::Dtype;
use kernels_cerebras::Tensor;

use crate::device::Buffer;
use crate::device::Device;
use crate::error::Result;
use crate::trace::{Handles, Root, Source};

#[derive(Debug, Clone)]
struct Array {
    dtype: Dtype,
    rows: u32,
    width: u32,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct Inputs {
    arrays: Vec<Array>,
    /// Every i32 input, back to back: one upload per fire instead of one per
    /// table.
    pack: Vec<i32>,
}

impl Inputs {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn push(
        &mut self,
        handles: &Handles,
        dtype: Dtype,
        rows: u32,
        width: u32,
        bytes: Vec<u8>,
    ) -> Tensor {
        let input = self.arrays.len() as u32;
        self.arrays.push(Array {
            dtype,
            rows,
            width,
            bytes,
        });
        handles.root(Root {
            source: Source::Input { input },
            dtype,
            rows,
            width,
        })
    }

    /// `values` as an i32 `[rows, width]` input (`rows × width` must be its
    /// length); an empty vector lands as one zero, so every input has a row.
    /// It rides the fire's pack.
    pub fn i32s(&mut self, handles: &Handles, values: &[i32], width: u32) -> Tensor {
        let width = width.max(1);
        let offset = self.pack.len() as u32;
        self.pack.extend_from_slice(values);
        if values.is_empty() {
            self.pack.extend(std::iter::repeat_n(0, width as usize));
        }
        let len = self.pack.len() as u32 - offset;
        let rows = len.div_ceil(width);
        self.pack.resize((offset + rows * width) as usize, 0);
        handles.root(Root {
            source: Source::Packed { offset },
            dtype: Dtype::I32,
            rows,
            width,
        })
    }

    /// Where the next `i32s` input starts in the pack.
    #[must_use]
    pub fn pack_offset(&self) -> u32 {
        self.pack.len() as u32
    }

    /// Elements in the pack.
    #[must_use]
    pub fn pack_len(&self) -> u32 {
        self.pack.len().max(1) as u32
    }

    /// Uploads the pack.
    pub fn upload_pack(&self, device: &Device) -> Result<Buffer> {
        let mut pack = self.pack.clone();
        if pack.is_empty() {
            pack.push(0);
        }
        // `PIE_CEREBRAS_DUMP_PACK=<prefix>`: the pack (`<prefix>.i32`) and
        // each input array (`<prefix>.in<n>`) of the latest fire.
        if let Some(pre) = std::env::var_os("PIE_CEREBRAS_DUMP_PACK") {
            let pre = pre.to_string_lossy().to_string();
            let bytes: Vec<u8> = pack.iter().flat_map(|x| x.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{pre}.i32"), &bytes);
            for (i, a) in self.arrays.iter().enumerate() {
                let _ = std::fs::write(format!("{pre}.in{i}"), &a.bytes);
            }
        }
        device.upload_words(Dtype::I32, pack.iter().map(|x| *x as u32).collect())
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
        let bytes = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.push(handles, Dtype::F32, rows, width, bytes)
    }

    /// Raw rows of `dtype`.
    pub fn raw(
        &mut self,
        handles: &Handles,
        dtype: Dtype,
        rows: u32,
        width: u32,
        bytes: Vec<u8>,
    ) -> Tensor {
        self.push(handles, dtype, rows, width, bytes)
    }

    /// Uploads input `input`.
    pub fn upload(&self, device: &Device, input: u32) -> Result<Buffer> {
        let a = &self.arrays[input as usize];
        device.upload(a.dtype, a.rows, a.width, &a.bytes)
    }

    /// Every input's element type and shape, in staging order: with the
    /// fire's windows, what the traced program's text depends on.
    #[must_use]
    pub fn shapes(&self) -> String {
        let mut out = String::with_capacity(self.arrays.len() * 12);
        out.push_str(&format!("pack{};", self.pack.len()));
        for a in &self.arrays {
            out.push_str(&format!("{:?}{}x{};", a.dtype, a.rows, a.width));
        }
        out
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.arrays.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.arrays.iter().map(|a| a.bytes.len() as u64).sum()
    }
}
