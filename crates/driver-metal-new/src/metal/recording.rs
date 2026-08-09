//! A fire, recorded once.
//!
//! `encode` walks the dispatch list every step: a pipeline set, a bind per
//! operand, an argument-table set and a dispatch, then a barrier. Measured on
//! `qwen3_0_6b` — 424 dispatches, 3 779 address binds, about 5 000
//! Objective-C messages, **14.8 ms**, which is 47.5 % of a prefill and
//! **76.4 % of a decode**. And it is the same 14.8 ms every step, because the
//! loop is proportional to the dispatch count and the dispatch count is a
//! property of the text rather than of the batch.
//!
//! This records that walk into an [`MTLIndirectCommandBuffer`] instead, so a
//! fire costs one `executeCommandsInBuffer` on the host.
//!
//! # What makes it replayable
//!
//! Only three things differ between two fires of one `(plan, row shape)` —
//! the arena, the params and the fire tables — and `metal::Scratch` pools all
//! three, so their addresses are the previous fire's. Everything else (the
//! dispatch order, the ten pipelines, the twelve grids, the weight addresses)
//! was already stable. See `.wiki/driver/graph-metal.md` §4.
//!
//! # The three preconditions, each of which fails badly
//!
//! Learned from `tests/device_icb.rs`, and none of them is documented by
//! Apple in a place this tree would have found:
//!
//! 1. **Pipelines must be compiled `supportIndirectCommandBuffers`.** Setting
//!    one that was not **faults** — SIGSEGV inside the recording loop, before
//!    anything executes. `Compiler` states it for every pipeline.
//! 2. **The command type must match the dispatch call.** Declaring
//!    `ConcurrentDispatch` and calling `concurrentDispatchThreads` is
//!    **silent**: the command does nothing.
//! 3. **The ICB must itself be resident**, because it is a buffer the GPU
//!    reads its commands out of.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice, MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor,
    MTLIndirectCommandType, MTLIndirectComputeCommand, MTLResidencySet, MTLResourceOptions, MTLSize,
};

use super::context::Context;
use super::regions::Regions;
use crate::error::{Error, Result};
use crate::model::dispatch::Dispatch;
use crate::model::encode::Params;
use crate::model::encode::Pipelines;

/// A fire's dispatches, recorded.
pub struct Recording {
    icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    commands: usize,
}

impl Recording {
    /// The buffer, for [`super::StepEncoder::execute_commands`].
    #[must_use]
    pub fn buffer(&self) -> &ProtocolObject<dyn MTLIndirectCommandBuffer> {
        &self.icb
    }

    /// How many commands it holds.
    #[must_use]
    pub fn commands(&self) -> usize {
        self.commands
    }
}

impl std::fmt::Debug for Recording {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recording")
            .field("commands", &self.commands)
            .finish()
    }
}

/// Record `dispatches` into an indirect command buffer.
///
/// `regions` must resolve every operand address to the allocation holding it:
/// a command binds a **buffer**, not an address, so an unregistered operand
/// is a refusal rather than a bind to nothing.
///
/// # Barriers
///
/// Every command carries one, which is what `encode` does between dispatches
/// and for the same measured reason: Metal does not order two dispatches in
/// one encoder, and three runs of one fire without a barrier gave widest
/// activations of 11.7, 23.1 and 4.5e12. Two of the three looked plausible.
///
/// # Errors
///
/// A dispatch whose pipeline is not compiled, whose scalars were not staged,
/// or whose operand address falls in no registered allocation. Each is drift
/// between the plan and what the caller set up, and each would otherwise be a
/// command reading somebody else's bytes.
pub fn record(
    context: &Context,
    pipelines: &Pipelines,
    params: &Params,
    regions: &Regions,
    dispatches: &[Dispatch<'_>],
) -> Result<Recording> {
    let widest = dispatches
        .iter()
        .map(|d| d.args.len() + d.param_slots.len())
        .max()
        .unwrap_or(1);
    let descriptor = MTLIndirectCommandBufferDescriptor::new();
    // BOTH spellings: the type has to match the call the command makes, and
    // declaring one while calling the other is a command that silently does
    // nothing. This crate dispatches threads, not threadgroups.
    descriptor.setCommandTypes(MTLIndirectCommandType(
        MTLIndirectCommandType::ConcurrentDispatch.0
            | MTLIndirectCommandType::ConcurrentDispatchThreads.0,
    ));
    // FALSE, and it is the property that makes a fire recordable at all: 424
    // dispatches bind DIFFERENT addresses to the same slots, so a command
    // inheriting the encoder's bindings would take whichever was bound last.
    descriptor.setInheritBuffers(false);
    descriptor.setInheritPipelineState(false);
    descriptor.setMaxKernelBufferBindCount(widest.max(1));

    // SAFETY: the descriptor is fully initialised above.
    let icb = unsafe {
        context
            .device()
            .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                &descriptor,
                dispatches.len().max(1),
                MTLResourceOptions::StorageModeShared,
            )
    }
    .ok_or_else(|| Error::Create {
        what: "indirect command buffer",
        message: format!("the device declined {} commands", dispatches.len()),
    })?;

    // RESIDENT: the GPU reads its commands out of this buffer, and this
    // context tracks nothing automatically. Without it the execute faults.
    context
        .residency()
        .addAllocation(ProtocolObject::from_ref(&*icb));
    context.residency().commit();
    context.residency().requestResidency();

    for (index, dispatch) in dispatches.iter().enumerate() {
        let pipeline = pipelines.get(dispatch.symbol).ok_or_else(|| Error::Create {
            what: "recording",
            message: format!("`{}` has no compiled pipeline", dispatch.symbol),
        })?;
        // SAFETY: `index` is below the command count declared above.
        let command = unsafe { icb.indirectComputeCommandAtIndex(index) };
        command.setComputePipelineState(pipeline);

        for (slot, arg) in dispatch.args.iter().enumerate() {
            let (buffer, offset) =
                regions.resolve(arg.slice.address).ok_or_else(|| Error::Create {
                    what: "recording",
                    message: format!(
                        "`{}` operand {slot} is at {:#x}, which is in no registered \
                         allocation -- a recorded command binds a BUFFER, so this \
                         would name the wrong one",
                        dispatch.symbol, arg.slice.address
                    ),
                })?;
            // SAFETY: the buffer outlives the recording (the registry retains
            // it) and the slot is below `maxKernelBufferBindCount`.
            unsafe {
                command.setKernelBuffer_offset_atIndex(buffer, offset as usize, slot);
            }
        }

        if !dispatch.params.is_empty() {
            let base = params.address_of(index).ok_or_else(|| Error::Create {
                what: "recording",
                message: format!("`{}` states scalars but was not staged", dispatch.symbol),
            })?;
            for p in &dispatch.param_slots {
                let at = base + u64::from(p.at);
                let (buffer, offset) = regions.resolve(at).ok_or_else(|| Error::Create {
                    what: "recording",
                    message: format!("`{}` scalars are at {at:#x}, unregistered", dispatch.symbol),
                })?;
                // SAFETY: as above.
                unsafe {
                    command.setKernelBuffer_offset_atIndex(buffer, offset as usize, p.slot);
                }
            }
        }

        command.concurrentDispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: dispatch.grid[0] as usize,
                height: dispatch.grid[1] as usize,
                depth: dispatch.grid[2] as usize,
            },
            MTLSize {
                width: dispatch.threadgroup[0] as usize,
                height: dispatch.threadgroup[1] as usize,
                depth: dispatch.threadgroup[2] as usize,
            },
        );
        // See the doc: a statement reading what the previous one wrote reads
        // whatever was there without this, and the failure is a number.
        command.setBarrier();
    }

    Ok(Recording {
        icb,
        commands: dispatches.len(),
    })
}
