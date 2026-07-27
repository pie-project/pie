//! The one geometry type.
//!
//! A rectangular byte copy is the only thing the loader's middle computes, and
//! it used to be spelled four different ways: `contract::compile::{Dim,
//! BytePiece}` coming out of the affine solver, `ir::{GatherDim, GatherPiece}`
//! in the middle IR, and `load_plan::{Dim, Extent}` in the plan. The
//! three `Dim`s were field-for-field identical and the conversions between them
//! were pure re-encoding — three chances to disagree about what a stride means,
//! and three places to check `spec.md` §3.3's rule that a destination stays
//! dense.
//!
//! Two shapes survive, and they differ in a way that matters:
//!
//! * [`Rect`] is **two-sided** — one copy, source and destination together. It
//!   is what the affine solver produces.
//! * [`Extent`] is **one-sided** — the source *or* the destination of an
//!   instruction, which is how the executor addresses memory.
//!
//! [`Rect::split`] is the only bridge, and it is where the dense-destination
//! rule is enforced, once.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// One level of a rectangular copy's loop nest.
///
/// Reading a `dims` slice from the outside in, the element at loop counters
/// `(i₀ … iₙ)` moves from `src_offset + Σ iₖ·src_strideₖ` to
/// `dst_offset + Σ iₖ·dst_strideₖ`. The innermost level always has unit
/// strides, so every copy ends in a contiguous stretch.
///
/// Counts and strides are in **bytes** once a [`Rect`] has been scaled by its
/// encoding. Before that they are in logical elements; see
/// `contract::compile::Lowering::byte_pieces`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dim {
    pub count: i64,
    pub src_stride: i64,
    pub dst_stride: i64,
}

/// One side of a copy: a base offset and the loop nest over it.
///
/// `element_bytes` is the size of the contiguous inner block, hoisted out of
/// `dims` because every executor wants it as the memcpy length rather than as
/// another loop level.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Extent {
    pub base_offset: u64,
    pub element_bytes: u32,
    pub dims: Vec<Dim>,
}

impl Extent {
    /// One unbroken run of `bytes` bytes.
    pub fn contiguous(bytes: u64) -> Self {
        Self {
            base_offset: 0,
            element_bytes: 1,
            dims: vec![Dim {
                count: i64::try_from(bytes).unwrap_or(i64::MAX),
                src_stride: 1,
                dst_stride: 1,
            }],
        }
    }

    /// Whether this addresses one unbroken block, and so could be aliased
    /// rather than copied.
    pub fn is_contiguous(&self) -> bool {
        self.dims.len() == 1 && self.dims[0].src_stride == 1 && self.dims[0].dst_stride == 1
    }
}

/// One rectangular copy, both sides at once, in bytes.
///
/// `leaf` indexes the owning `Lowering`'s leaves — the tensor these bytes come
/// from. This is what `contract::compile` produces after folding runs, and the
/// last form in which source and destination are described together.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rect {
    pub leaf: usize,
    pub src_offset: u64,
    pub dst_offset: u64,
    pub dims: Vec<Dim>,
}

impl Rect {
    /// A single contiguous byte range.
    pub fn span(leaf: usize, src_offset: u64, dst_offset: u64, bytes: u64) -> Self {
        Self {
            leaf,
            src_offset,
            dst_offset,
            dims: vec![Dim {
                count: i64::try_from(bytes).unwrap_or(i64::MAX),
                src_stride: 1,
                dst_stride: 1,
            }],
        }
    }

    pub fn bytes(&self) -> u64 {
        self.dims.iter().map(|dim| dim.count).product::<i64>() as u64
    }

    /// Whether this moves one unbroken block.
    pub fn is_contiguous(&self) -> bool {
        self.dims.len() == 1 && self.dims[0].src_stride == 1 && self.dims[0].dst_stride == 1
    }

    /// Split into the source and destination extents an instruction carries.
    ///
    /// This is the single place `spec.md` §3.3's third fold rule is enforced:
    /// **the destination must stay dense.** A copy whose destination skips
    /// around is not a rectangle the executor can express as one strided write,
    /// and folding may not produce one — so if the walk below finds a
    /// destination stride that is not the running dense extent, the fold was
    /// wrong and the copy is rejected rather than silently mis-lowered.
    pub fn split(&self) -> Result<(Extent, Extent)> {
        let bytes = self.bytes();
        if self.is_contiguous() {
            return Ok((Extent::contiguous(bytes), Extent::contiguous(bytes)));
        }
        let (inner, outer) = self
            .dims
            .split_last()
            .ok_or_else(|| Error::Contract("copy has no extent".to_string()))?;
        if inner.src_stride != 1 || inner.dst_stride != 1 {
            return Err(Error::Contract(
                "copy has no contiguous inner block".to_string(),
            ));
        }
        let element_bytes = u32::try_from(inner.count)
            .map_err(|_| Error::Overflow("copy inner block exceeds 4 GiB".to_string()))?;

        let mut dense = i64::from(element_bytes);
        let mut source_dims = Vec::with_capacity(outer.len());
        let mut dest_dims = Vec::with_capacity(outer.len());
        for dim in outer.iter().rev() {
            if dim.dst_stride != dense {
                return Err(Error::Contract(
                    "copy writes a non-contiguous destination".to_string(),
                ));
            }
            source_dims.push(Dim {
                count: dim.count,
                src_stride: dim.src_stride,
                dst_stride: dense,
            });
            dest_dims.push(Dim {
                count: dim.count,
                src_stride: dense,
                dst_stride: dense,
            });
            dense = dense
                .checked_mul(dim.count)
                .ok_or_else(|| Error::Overflow("copy extent overflow".to_string()))?;
        }
        source_dims.reverse();
        dest_dims.reverse();
        Ok((
            Extent {
                base_offset: 0,
                element_bytes,
                dims: source_dims,
            },
            Extent {
                base_offset: 0,
                element_bytes,
                dims: dest_dims,
            },
        ))
    }
}
