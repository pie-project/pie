//! A fire's inputs: host arrays the shell stages, each a root the program
//! reads as a parameter. Only what the traced program names is uploaded.

use dtype::Dtype;
use kernels_xla::Tensor;

use crate::device::Device;
use crate::error::Result;
use crate::pjrt::Buffer;
use crate::trace::{Handles, Root, Source};

#[derive(Debug, Clone)]
struct Array {
    dtype: Dtype,
    rows: u32,
    width: u32,
    bytes: Vec<u8>,
}

/// A pack word the device supplies: `pack[at]` is word `word` of `src`, a
/// `u32` array still on the device (a guest pass's output).
#[derive(Debug, Clone)]
pub struct Patch {
    pub at: u32,
    pub src: std::sync::Arc<Buffer>,
    pub word: u32,
}

#[derive(Debug, Clone, Default)]
pub struct Inputs {
    arrays: Vec<Array>,
    /// Every i32 input, back to back: one upload per fire instead of one per
    /// table.
    pack: Vec<i32>,
    /// Pack words read off the device after the upload.
    patches: Vec<Patch>,
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

    /// Makes `pack[at]` the device's word `word` of `src`, in place of
    /// whatever the host staged there.
    pub fn patch(&mut self, at: u32, src: std::sync::Arc<Buffer>, word: u32) {
        self.patches.push(Patch { at, src, word });
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
        let bytes: Vec<u8> = pack.iter().flat_map(|x| x.to_le_bytes()).collect();
        // `PIE_XLA_DUMP_PACK=<prefix>`: the pack (`<prefix>.i32`) and each
        // input array (`<prefix>.in<n>`) of the latest fire, for replaying a
        // `PIE_XLA_DUMP` program offline on real inputs.
        if let Some(pre) = std::env::var_os("PIE_XLA_DUMP_PACK") {
            let pre = pre.to_string_lossy().to_string();
            let _ = std::fs::write(format!("{pre}.i32"), &bytes);
            for (i, a) in self.arrays.iter().enumerate() {
                let _ = std::fs::write(format!("{pre}.in{i}"), &a.bytes);
            }
        }
        let mut buffer =
            device.upload_flat(crate::pjrt::ElementType::S32, &bytes, pack.len() as i64)?;
        // Words the device supplies: one small scatter per source, run in
        // the device's order behind the pass that writes the source.
        let mut done = vec![false; self.patches.len()];
        for first in 0..self.patches.len() {
            if done[first] {
                continue;
            }
            let src = &self.patches[first].src;
            let group: Vec<&Patch> = self
                .patches
                .iter()
                .enumerate()
                .filter(|(at, patch)| {
                    let same = std::sync::Arc::ptr_eq(&patch.src, src);
                    if same {
                        done[*at] = true;
                    }
                    same
                })
                .map(|(_, patch)| patch)
                .collect();
            buffer = patch_pack(device, buffer, pack.len(), src, &group)?;
        }
        Ok(buffer)
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

/// `pack` with each patch's word taken from `src` (`u32 [..]`, any rank):
/// a scatter of a gather, compiled once per shape.
fn patch_pack(
    device: &Device,
    pack: Buffer,
    len: usize,
    src: &Buffer,
    patches: &[&Patch],
) -> Result<Buffer> {
    use kernels_xla::hlo::{Combine, Elem, Func, Ty};
    let dims = src.dims()?;
    let words: i64 = dims.iter().product::<i64>().max(1);
    let k = patches.len().next_power_of_two();
    let p = len as i64;
    let mut idx: Vec<i32> = Vec::with_capacity(2 * k);
    idx.extend(patches.iter().map(|patch| patch.at as i32));
    // Padding lands past the pack, which a scatter drops.
    idx.resize(k, p as i32);
    idx.extend(patches.iter().map(|patch| patch.word as i32));
    idx.resize(2 * k, 0);
    let mut f = Func::new("main");
    let pack_v = f.param(Ty::new(Elem::I32, &[p]), None);
    let src_v = f.param(Ty::new(Elem::U32, &dims), None);
    let idx_v = f.param(Ty::new(Elem::I32, &[2 * k as i64]), None);
    let at = f.slice(idx_v, &[0], &[k as i64], &[1])?;
    let from = f.slice(idx_v, &[k as i64], &[2 * k as i64], &[1])?;
    let flat = f.reshape(src_v, &[words, 1])?;
    let flat = f.bitcast(flat, Elem::I32)?;
    let rows = f.take_rows(flat, from)?;
    let table = f.reshape(pack_v, &[p, 1])?;
    let out = f.put_rows(table, at, rows, Combine::Set)?;
    let out = f.reshape(out, &[p])?;
    let text = f.module("patch_pack", &[out]);
    let program = device.program(
        &text,
        crate::trace::Signature {
            params: Vec::new(),
            results: Vec::new(),
        },
    )?;
    let bytes: Vec<u8> = idx.iter().flat_map(|x| x.to_le_bytes()).collect();
    let idx_b = device.upload_flat(crate::pjrt::ElementType::S32, &bytes, 2 * k as i64)?;
    let mut outs = device.run(
        &program,
        vec![
            crate::pjrt::Arg::Keep(&pack),
            crate::pjrt::Arg::Keep(src),
            crate::pjrt::Arg::Keep(&idx_b),
        ],
        false,
    )?;
    outs.pop().ok_or_else(|| crate::error::Fault::Unbound {
        what: "the patched pack, which the patch program did not return".to_string(),
    })
}
