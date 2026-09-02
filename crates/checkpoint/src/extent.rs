//! The one geometry type for a rectangular byte copy.
//!
//! [`Rect`] is two-sided (one copy, source and destination together, what the
//! affine solver produces); [`Extent`] is one-sided (source or destination of
//! an instruction, how the executor addresses memory). [`Rect::split`] is the
//! only bridge, and the one place the dense-destination rule is enforced.

use serde::{Deserialize, Serialize};

use crate::error::{Error, OrOverflow, Result};

/// One level of a rectangular copy's loop nest. Element at `(i₀ … iₙ)` moves
/// from `src_offset + Σ iₖ·src_strideₖ` to `dst_offset + Σ iₖ·dst_strideₖ`;
/// innermost level always has unit strides. Counts/strides are bytes once a
/// [`Rect`] is scaled by its encoding, logical elements before.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dim {
    pub count: i64,
    pub src_stride: i64,
    pub dst_stride: i64,
}

/// One side of a copy: a base offset and the loop nest over it.
/// `element_bytes` is the contiguous inner block's size, hoisted out of
/// `dims` since every executor wants it as the memcpy length.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Extent {
    pub base_offset: u64,
    pub element_bytes: u32,
    pub dims: Vec<Dim>,
}

impl Extent {
    /// One unbroken run of `bytes` bytes — the shape [`Extent::is_byte_run`]
    /// recognises.
    pub fn byte_run(bytes: u64) -> Self {
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

    /// Whether neither side skips: every stride, innermost out, is the
    /// running dense extent. Asked before merging two copies.
    pub fn is_dense(&self) -> bool {
        self.walk_dense(|dim, stride| dim.src_stride == stride && dim.dst_stride == stride)
    }

    /// The same question, asked of the destination alone.
    pub fn has_dense_destination(&self) -> bool {
        self.walk_dense(|dim, stride| dim.dst_stride == stride)
    }

    /// Whether this is one unbroken run of bytes from offset zero — stronger
    /// than [`Extent::is_dense`]: also needs the base folded in and elements
    /// byte-sized, so the whole extent is a `{offset, len}` pair.
    pub fn is_byte_run(&self) -> bool {
        self.base_offset == 0 && self.element_bytes == 1 && self.dims.len() == 1 && self.is_dense()
    }

    /// The dense row-major layout of `shape`, both sides packed — the shape
    /// [`Extent::is_dense`] recognises.
    pub fn dense(shape: &[i64], element_bytes: u64) -> Self {
        let mut stride = i64::try_from(element_bytes).unwrap_or(i64::MAX);
        let mut dims = Vec::with_capacity(shape.len());
        for dim in shape.iter().rev() {
            dims.push(Dim {
                count: *dim,
                src_stride: stride,
                dst_stride: stride,
            });
            stride = stride.saturating_mul(*dim);
        }
        dims.reverse();
        Self {
            base_offset: 0,
            element_bytes: u32::try_from(element_bytes).unwrap_or(u32::MAX),
            dims,
        }
    }

    fn walk_dense(&self, packed: impl Fn(&Dim, i64) -> bool) -> bool {
        let mut stride = i64::from(self.element_bytes);
        for dim in self.dims.iter().rev() {
            if dim.count < 0 || !packed(dim, stride) {
                return false;
            }
            match stride.checked_mul(dim.count) {
                Some(next) => stride = next,
                None => return false,
            }
        }
        true
    }
}

/// One rectangular copy, both sides at once, in bytes.
/// `leaf` indexes the owning `Lowering`'s leaves — the tensor these bytes
/// come from.
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

    /// Whether this moves one unbroken block. The same question
    /// [`Extent::is_byte_run`] answers, minus the base offsets a [`Rect`]
    /// carries separately.
    pub fn is_byte_run(&self) -> bool {
        self.dims.len() == 1 && self.dims[0].src_stride == 1 && self.dims[0].dst_stride == 1
    }

    /// Split into the source and destination extents an instruction carries.
    /// Enforces that the destination stays dense: a copy whose destination
    /// skips around is rejected rather than silently mis-lowered.
    pub fn split(&self) -> Result<(Extent, Extent)> {
        let bytes = self.bytes();
        if self.is_byte_run() {
            return Ok((Extent::byte_run(bytes), Extent::byte_run(bytes)));
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
        let element_bytes =
            u32::try_from(inner.count).or_overflow("copy inner block exceeds 4 GiB")?;

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
                .or_overflow("copy extent overflow")?;
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

