//! Firing a step whose routed experts do not all fit: the device half.
//!
//! [`batch::paging`](crate::batch::paging) decides the shape — where the
//! segments cut, how many experts one fire can want at once, how a routing
//! readback is renumbered. This module is the twenty lines that drive it: run
//! the segments, and between two of them give the last layer's pins back and
//! page the next layer's experts in.
//!
//! # Why a step becomes several command buffers
//!
//! An Apple Silicon GPU has no demand paging, and `requestResidency` wires
//! every page of a mapping whether a kernel reads it or not. A bank bigger than
//! the working set therefore cannot be mapped — it has to be read through a
//! region that stays wired while its *contents* change, and only the host can
//! change them. So the step is ordered command buffers with the host awake
//! between them:
//!
//! ```text
//!   [ ... router of layer 0 ]  -> host pages layer 0's experts in
//!   [ ... router of layer 1 ]  -> host pages layer 1's experts in
//!   ...
//!   [ the tail ]                  pages nothing
//! ```
//!
//! and the ids the router wrote are rewritten **in place** from expert numbers
//! to slot numbers. That rewrite is why no kernel changed for any of this: a
//! routed matvec does `base += expert_ids[sel] * stride`, and a slot number in
//! a shorter stack is the same instruction against different bytes.
//!
//! The cost is one submit-and-wait per mixture layer. It is a bad trade for a
//! model that fits and the only trade there is for a model that does not.
//!
//! # The ordering that is the whole correctness argument
//!
//! Pins come back **first**, before the page-in. The previous segment's command
//! buffer has completed — which is exactly the condition that makes its slots
//! reusable — and doing it the other way holds two layers at once and needs
//! twice the budget for no benefit. [`Stepper::run_segments`] calls `between`
//! only after the segment it follows has been awaited, which is what makes the
//! host write to shared storage legal at all; it is also the safety condition
//! [`ExpertSlab::ensure_resident`] states.
//!
//! [`ExpertSlab::ensure_resident`]: crate::loader::ExpertSlab::ensure_resident

use crate::batch::{IdsLayout, PagingPlan, RenumberRefused, renumber_routing};
use crate::loader::ExpertSlab;
use crate::region::Region;
use crate::{Error, Result};

use super::encoder::{StepEncoder, Stepper};
use super::handle::Handle;
use super::timing::Timing;

/// Run one step's segments, paging each mixture layer's experts in between.
///
/// `ids[k]` is the buffer mixture layer `k`'s router writes its decision into,
/// host-readable and one per layer — **not** one for the stack. A single handle
/// would assume every layer's routing decision colours onto the same pool slot,
/// which is a property of a greedy colouring rather than a guarantee; the day it
/// stopped holding, the host would rewrite a buffer the next segment does not
/// read, and that is wrong tokens rather than an error.
///
/// `encode(begin, end, step)` encodes the dispatches in `[begin, end)`. The
/// caller recognises the first segment by `begin == 0` and the last by `end`
/// reaching the DAG size, which is where any pre/post hook belongs.
///
/// # Errors
///
/// [`Error::Program`] when the handles do not match the plan, or when a routing
/// readback cannot be renumbered — a wild expert id, a slab with no free slot,
/// or an ids buffer too short for the rows and stride it was promised. Whatever
/// the first failing segment returns, otherwise; a segment that never finished
/// leaves the host holding results that were never computed, so `run_segments`
/// neither pages for it nor encodes the rest.
pub fn fire_paged<R, F>(
    stepper: &mut Stepper,
    plan: &PagingPlan,
    slab: &mut ExpertSlab<'_, R>,
    ids: &[Handle],
    layout: IdsLayout,
    n_experts: u32,
    mut encode: F,
) -> Result<Timing>
where
    R: Region,
    F: FnMut(usize, usize, &mut StepEncoder<'_>) -> Result<()>,
{
    // One ids buffer per mixture layer; the tail segment routes nothing and so
    // owns none. The C++ carried the handle inside its `Cut` and appended a
    // tail cut with a null one, which makes "the tail has no ids" a value that
    // has to be remembered rather than a length that cannot be wrong.
    let layers = plan.mixture_layers();
    if ids.len() != layers {
        return Err(Error::Program {
            message: format!(
                "expert paging: {} routing buffers for {layers} mixture layers",
                ids.len()
            ),
        });
    }

    let segments: Vec<(usize, usize)> = plan.segments().collect();
    stepper.run_segments(
        segments.len(),
        |index, step| {
            let (begin, end) = segments[index];
            encode(begin, end, step)
        },
        |index| {
            // Pins back FIRST. See the module doc: the completed segment's
            // slots are reusable precisely because it completed, and paging in
            // before releasing would hold two layers at once.
            slab.end_batch();
            let Some(handle) = ids.get(index) else {
                // The tail owns no experts.
                return Ok(());
            };
            page_in(slab, handle, index, layout, n_experts)
        },
    )
}

/// Renumber one mixture layer's routing decision from expert ids to slot ids,
/// paging each expert in as it is named.
fn page_in<R: Region>(
    slab: &mut ExpertSlab<'_, R>,
    ids: &Handle,
    layer: usize,
    layout: IdsLayout,
    n_experts: u32,
) -> Result<()> {
    // SAFETY: `run_segments` awaited the completion of the segment this follows
    // before calling us, so no GPU work is reading these bytes; the handle is
    // valid for `len` bytes for as long as it lives, and nothing else holds a
    // reference to them while this borrow is live. That step boundary is also
    // the exact condition `ExpertSlab::ensure_resident` documents as its own.
    let bytes = unsafe {
        core::slice::from_raw_parts_mut(
            ids.contents().cast::<u8>().as_ptr(),
            usize::try_from(ids.len()).unwrap_or(usize::MAX),
        )
    };

    let layer = u32::try_from(layer).unwrap_or(u32::MAX);
    renumber_routing(
        bytes,
        layout.rows,
        layout.experts_per_token,
        layout.row_stride_bytes,
        n_experts,
        // SAFETY: as above -- we are between two segments, which is the
        // boundary this call requires.
        |expert| unsafe { slab.ensure_resident(layer, expert) },
    )
    .map_err(|refused| Error::Program {
        message: match refused {
            RenumberRefused::ShortIds { need, have } => format!(
                "expert paging: mixture layer {layer}'s routing buffer holds {have} bytes, \
                 and {}\u{a0}rows at the promised stride need {need}",
                layout.rows
            ),
            other => format!("expert paging: mixture layer {layer}: {other:?}"),
        },
    })
}
