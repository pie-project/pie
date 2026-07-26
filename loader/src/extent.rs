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
//!
//! The predicates live here for the same reason the type does. "Is this extent
//! simple enough to merge" had four answers in three modules — `is_contiguous`,
//! `is_byte_extent`, `compact_extent_for_copy` and a fourth `compact_extent`
//! that asked only about the destination — and they were not the same question
//! written four times, they were three different questions sharing two names.
//! They are now [`Extent::is_dense`], [`Extent::has_dense_destination`] and
//! [`Extent::is_byte_run`], strictly ordered, each paired with the constructor
//! that produces the shape it recognises.

use serde::{Deserialize, Serialize};

use crate::error::{Error, OrOverflow, Result};

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

    /// Whether neither side skips: walking from the innermost dimension out,
    /// every stride is the running dense extent.
    ///
    /// This is the question a pass asks before merging two copies, because a
    /// dense extent is one the merge can describe as a longer run.
    pub fn is_dense(&self) -> bool {
        self.walk_dense(|dim, stride| dim.src_stride == stride && dim.dst_stride == stride)
    }

    /// The same question, asked of the destination alone.
    ///
    /// An executor writing into an arena cares only that *its* side is packed;
    /// the source may stride however the file does. `spec.md` §3.3's third fold
    /// rule is this property stated over a [`Rect`] instead.
    pub fn has_dense_destination(&self) -> bool {
        self.walk_dense(|dim, stride| dim.dst_stride == stride)
    }

    /// Whether this is one unbroken run of bytes from offset zero — the shape
    /// [`Extent::byte_run`] builds, and the only one a bulk or slab write can
    /// carry.
    ///
    /// Stronger than [`Extent::is_dense`]: density is preserved by scaling, but
    /// a bulk write also needs the base folded in and the elements byte-sized,
    /// so that the whole extent is a `{offset, len}` pair.
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

    /// Whether this moves one unbroken block. The same question
    /// [`Extent::is_byte_run`] answers, minus the base offsets a [`Rect`]
    /// carries separately.
    pub fn is_byte_run(&self) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The three predicates order strictly, and the constructors land where
    /// their names say. Written because the crate used to spell these four
    /// different ways in three modules, and nothing checked they agreed.
    #[test]
    fn the_predicates_are_ordered_by_strength() {
        let run = Extent::byte_run(64);
        assert!(run.is_byte_run() && run.is_dense() && run.has_dense_destination());

        let rows = Extent::dense(&[4, 8], 2);
        assert!(rows.is_dense() && rows.has_dense_destination());
        assert!(!rows.is_byte_run(), "two dims is not one run");
    }

    #[test]
    fn a_strided_source_still_writes_a_dense_destination() {
        let gathered = Extent {
            base_offset: 0,
            element_bytes: 2,
            dims: vec![Dim {
                count: 4,
                src_stride: 64,
                dst_stride: 2,
            }],
        };
        assert!(gathered.has_dense_destination());
        assert!(!gathered.is_dense(), "the source skips");
    }

    #[test]
    fn a_byte_run_needs_its_base_folded_in() {
        let offset = Extent {
            base_offset: 512,
            ..Extent::byte_run(64)
        };
        assert!(offset.is_dense(), "density says nothing about the base");
        assert!(
            !offset.is_byte_run(),
            "a bulk write carries offset plus len"
        );
    }

    #[test]
    fn dense_is_the_layout_a_shape_gets() {
        assert_eq!(
            Extent::dense(&[3, 5], 4).dims,
            vec![
                Dim {
                    count: 3,
                    src_stride: 20,
                    dst_stride: 20
                },
                Dim {
                    count: 5,
                    src_stride: 4,
                    dst_stride: 4
                },
            ]
        );
    }
}
