//! Where one fire's values live in the scratch buffer.
//!
//! A fire's intermediate values are not separate allocations. The host takes
//! one transient buffer, decides an offset for every value in it, and hands the
//! kernels the offset table; the kernels then address their inputs and outputs
//! as slices of that one buffer. This module is that decision, and it is the
//! only thing that knows the buffer's shape:
//!
//! ```text
//! 0                    256                                  temporary
//! +--------------------+-----------+-----------+-- ... --+-------------+
//! | dummy binding      | value 0   | value 1   |         | temporaries |
//! +--------------------+-----------+-----------+-- ... --+-------------+
//! ```
//!
//! The first 256 bytes are a **dummy binding**: emitted kernels declare
//! argument slots they do not always use, and Metal has no null buffer
//! argument, so every unused slot is pointed at this region. Nothing reads it
//! and nothing may write it. Reserving it here rather than allocating a
//! separate buffer costs 256 bytes of a buffer that is already at least that
//! big, and saves a second transient allocation per fire.
//!
//! The tail is scratch the kernels use for reductions and sort keys, sized for
//! the largest value in the fire because a kernel does not know which value it
//! will be run over until it is dispatched.
//!
//! # Why everything is aligned
//!
//! Every offset is a multiple of [`ALIGN`]. That is not a performance
//! preference: a Metal buffer binding takes an offset, and an unaligned one is
//! rejected by the API. Alignment is applied to each value's *size* as well as
//! to its offset, which is redundant given a 256-aligned start -- but it is what
//! keeps the running total aligned without the loop having to reason about it,
//! so the redundancy is the invariant rather than a waste.
//!
//! # Why the arithmetic is checked
//!
//! The C++ accumulated the running total with `+=`, aligned with
//! `(value + alignment - 1) / alignment * alignment`, and only tested the
//! result against the 512 MiB bound *after* the loop had finished. Both steps
//! can wrap on a `size_t`, and a wrapped total passes a bound check that a real
//! one would fail -- which is the worst possible outcome, since the buffer then
//! gets allocated at the wrapped size and the kernels address it at the real
//! offsets. The C++ half-noticed this and added `scratch_bytes < temporary_offset`
//! to catch a wrap in the final addition only. Here every step is checked, so
//! the bound check means what it says.

use super::extent::ValueDesc;

/// The alignment every binding offset must satisfy.
///
/// Metal rejects an unaligned buffer offset outright, so this is a
/// correctness constant, not a tuning one.
pub const ALIGN: u64 = 256;

/// The region at offset zero that unused kernel arguments point at.
///
/// Metal has no null buffer argument. A kernel that declares an argument it
/// does not use on this path still needs something bound, and this is it.
pub const DUMMY_BYTES: u64 = ALIGN;

/// How much scratch one fire may ask for.
///
/// Not a hardware limit: a bound on how badly a single fire can be allowed to
/// go wrong. A layout that wants more than this is far more likely to be a
/// shape that resolved wrongly than a workload anyone intended, and refusing it
/// here turns an allocation failure -- or a several-gibibyte allocation that
/// succeeds and then thrashes -- into a named error.
pub const MAX_BYTES: u64 = 512 << 20;

/// How many `u32` temporaries the kernels want per element of the largest
/// value.
///
/// Four: the reduction and sort paths keep a key, a value, and two ping-pong
/// halves. The number is the kernels', not this module's, and it lives here
/// because this is where the space for it is reserved.
const TEMPORARIES_PER_ELEMENT: u64 = 4;

/// Why a fire's scratch could not be laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TooLarge {
    /// The layout came out larger than [`MAX_BYTES`].
    Bound {
        /// What it came to.
        bytes: u64,
        /// What it may be.
        limit: u64,
    },
    /// The layout did not fit a `u64` at all.
    ///
    /// Unreachable with values whose element counts fit a `u32`, which
    /// [`describe`](super::extent::describe) guarantees -- but the check is
    /// what makes [`TooLarge::Bound`] trustworthy, so it is taken rather than
    /// assumed.
    Overflow,
}

/// Where each of a fire's values sits, and how big the buffer must be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    /// Byte offset of each value, in the order the descriptors were given.
    pub values: Vec<u64>,
    /// Byte offset of the kernels' temporary region.
    pub temporary: u64,
    /// How many bytes that region holds.
    pub temporary_bytes: u64,
    /// How big the whole buffer must be.
    pub total: u64,
}

/// Round up to the next multiple of [`ALIGN`], or `None` if that would not fit.
fn align_up(value: u64) -> Option<u64> {
    value.checked_next_multiple_of(ALIGN)
}

/// Lay out one fire's scratch.
///
/// An empty descriptor list is a legal fire -- a stage whose regions produce
/// nothing addressable -- and it lays out to the dummy binding plus a
/// minimum-sized temporary region. The C++ instead pushed a placeholder
/// descriptor before laying out, so that its `descriptors.size() * sizeof(..)`
/// buffer allocations would not be zero-length, which Metal rejects. That
/// padding is a property of acquiring the buffers, not of the layout, so it
/// belongs at the call site; conflating the two made the layout report a value
/// slot that no value owned.
///
/// # Errors
///
/// [`TooLarge`] if the result exceeds [`MAX_BYTES`] or overflows.
pub fn layout(descriptors: &[ValueDesc]) -> Result<Layout, TooLarge> {
    let mut values = Vec::with_capacity(descriptors.len());
    let mut at = DUMMY_BYTES;
    // One, not zero, so a fire with no values still reserves a usable
    // temporary region rather than a zero-length one.
    let mut widest: u64 = 1;

    for descriptor in descriptors {
        at = align_up(at).ok_or(TooLarge::Overflow)?;
        values.push(at);
        let span = align_up(descriptor.device_bytes()).ok_or(TooLarge::Overflow)?;
        at = at.checked_add(span).ok_or(TooLarge::Overflow)?;
        widest = widest.max(u64::from(descriptor.len));
    }

    let temporary = align_up(at).ok_or(TooLarge::Overflow)?;
    let temporary_bytes = widest
        .checked_mul(size_of::<u32>() as u64)
        .and_then(|bytes| bytes.checked_mul(TEMPORARIES_PER_ELEMENT))
        .and_then(align_up)
        .ok_or(TooLarge::Overflow)?;
    let total = temporary
        .checked_add(temporary_bytes)
        .ok_or(TooLarge::Overflow)?;

    if total > MAX_BYTES {
        return Err(TooLarge::Bound {
            bytes: total,
            limit: MAX_BYTES,
        });
    }
    Ok(Layout {
        values,
        temporary,
        temporary_bytes,
        total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::extent::{Extents, describe};
    use driver_abi::local::PIE_EXTENT_STATIC;
    use driver_abi::plan::LaunchPlanValue;

    fn desc(dtype: u8, len: u32) -> ValueDesc {
        describe(
            &LaunchPlanValue {
                dtype,
                extents: vec![PIE_EXTENT_STATIC],
                dims: vec![len],
            },
            &Extents::default(),
        )
        .expect("a rank-one shape of a legal extent resolves")
    }

    /// Every offset a kernel is handed must be one Metal will accept.
    #[test]
    fn every_offset_is_aligned() {
        let plan = layout(&[desc(0, 1), desc(0, 65), desc(3, 3), desc(0, 1000)])
            .expect("a small fire fits");
        for offset in &plan.values {
            assert_eq!(offset % ALIGN, 0, "offset {offset} is not bindable");
        }
        assert_eq!(plan.temporary % ALIGN, 0);
        assert_eq!(plan.total % ALIGN, 0);
    }

    /// Values may not overlap, and none may start inside the dummy binding --
    /// a value laid over it would be silently clobbered by every kernel that
    /// bound an unused argument.
    #[test]
    fn values_are_disjoint_and_clear_of_the_dummy_binding() {
        let descriptors = [desc(0, 1), desc(0, 65), desc(3, 3), desc(0, 1000)];
        let plan = layout(&descriptors).expect("a small fire fits");
        let mut end = DUMMY_BYTES;
        for (offset, descriptor) in plan.values.iter().zip(&descriptors) {
            assert!(*offset >= end, "value at {offset} overlaps the one before");
            end = offset + descriptor.device_bytes();
        }
        assert!(
            plan.temporary >= end,
            "the temporary region overlaps a value"
        );
    }

    /// A fire that produces nothing addressable still needs the dummy binding
    /// and a temporary region; it does not need a value slot.
    #[test]
    fn a_fire_with_no_values_lays_out_to_the_dummy_binding_and_a_temporary() {
        let plan = layout(&[]).expect("an empty fire fits");
        assert!(plan.values.is_empty(), "no value means no value offset");
        assert_eq!(plan.temporary, DUMMY_BYTES);
        assert_eq!(
            plan.temporary_bytes, ALIGN,
            "one element of four u32 temporaries, rounded up to a binding"
        );
    }

    /// The temporary region is sized for the largest value, not the last one:
    /// a kernel does not know which value it will run over until dispatch.
    #[test]
    fn the_temporary_region_follows_the_widest_value() {
        let wide = layout(&[desc(0, 4096), desc(0, 1)]).expect("fits");
        let narrow = layout(&[desc(0, 1), desc(0, 1)]).expect("fits");
        assert_eq!(wide.temporary_bytes, 4096 * 4 * TEMPORARIES_PER_ELEMENT);
        assert!(wide.temporary_bytes > narrow.temporary_bytes);
    }

    /// A bool is one addressable byte per lane, so it takes a quarter of the
    /// span an `f32` of the same length does -- which is the layout actually
    /// reading `device_bytes` rather than assuming a word per element.
    #[test]
    fn a_bool_value_takes_a_quarter_of_the_span_of_an_f32_of_the_same_length() {
        let bools = layout(&[desc(3, 4096), desc(0, 1)]).expect("fits");
        let floats = layout(&[desc(0, 4096), desc(0, 1)]).expect("fits");
        assert_eq!(bools.values[1] - bools.values[0], 4096);
        assert_eq!(floats.values[1] - floats.values[0], 4096 * 4);
    }

    /// The bound is what stops one bad shape from asking for the machine.
    #[test]
    fn a_fire_past_the_bound_is_refused_and_says_by_how_much() {
        // 2^27 f32 lanes is 512 MiB on its own, before the dummy binding or
        // the temporary region.
        let plan = layout(&[desc(0, 1 << 27)]);
        let Err(TooLarge::Bound { bytes, limit }) = plan else {
            panic!("a fire larger than the bound must be refused: {plan:?}");
        };
        assert!(bytes > limit);
        assert_eq!(limit, MAX_BYTES);
    }

    /// The C++ tested the total against the bound only after accumulating it,
    /// so a total that wrapped passed a check the real one would have failed.
    /// Checked arithmetic makes the wrap its own answer instead.
    #[test]
    fn a_layout_that_would_wrap_is_refused_rather_than_wrapping_under_the_bound() {
        let huge = ValueDesc {
            len: u32::MAX,
            ..ValueDesc::default()
        };
        let many = vec![huge; 64];
        assert!(
            matches!(
                layout(&many),
                Err(TooLarge::Bound { .. } | TooLarge::Overflow)
            ),
            "64 maximal values are 1 TiB; the answer must be a refusal, not a \
             small number that fits"
        );
    }
}
