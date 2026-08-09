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
//! the arena, the params and the fire tables — and `gpu::fire::Scratch` pools all
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
    MTLComputePipelineState, MTLDevice, MTLIndirectCommandBuffer,
    MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType, MTLIndirectComputeCommand,
    MTLResidencySet, MTLResourceOptions, MTLSize,
};

use crate::error::{Error, Result};
use crate::device::context::Context;
use crate::device::regions::Regions;

/// One address bound to one kernel slot.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct Bind {
    /// The GPU address, which [`Regions`] turns back into a buffer and an
    /// offset. A recorded command binds a BUFFER, not an address, which is
    /// the whole reason the registry exists.
    pub address: u64,
    /// Which `[[buffer(n)]]` it lands in.
    pub slot: usize,
}

/// One compute command, in the vocabulary a recording is made of.
///
/// # Why this exists
///
/// `record` used to take `&[Dispatch]`, `&Pipelines` and `&Params` — three
/// types from `lowering/` and `gpu/bind/`, both of which are ABOVE this
/// module. `.wiki/driver/real-metal-north-star.md` §9 names the consequence:
/// a cycle, and an ICB path that knows what a fire is.
///
/// What a recording actually needs is a pipeline, some addresses and a grid.
/// That is this, and none of those three words is a fire's. The translation
/// lives in [`crate::bind::encode::commands`], one layer up, where
/// `Dispatch`, `Pipelines` and `Params` are already in scope and where the
/// identical walk is done for encoding.
///
/// The borrow is the pipeline's, and it is deliberate: a `Command` cannot
/// outlive the `Pipelines` that compiled it, so a recording made from stale
/// pipelines is not constructible rather than merely discouraged.
#[derive(Debug)]
pub struct Command<'a> {
    /// Compiled `supportIndirectCommandBuffers`, which is precondition 1 —
    /// setting one that was not **faults**, inside the recording loop.
    pub pipeline: &'a ProtocolObject<dyn MTLComputePipelineState>,
    /// Every operand and scalar address this command binds, already resolved
    /// to slots. Flat, because a recorded command does not distinguish them:
    /// both are `setKernelBuffer_offset_atIndex`.
    pub binds: Vec<Bind>,
    /// Threads, not threadgroups. See precondition 2.
    pub grid: [u32; 3],
    /// Threads per threadgroup.
    pub threadgroup: [u32; 3],
    /// For the error message only, so a refusal names the kernel rather than
    /// an index. Not hashed: two fires that differ only in a symbol string
    /// differ in a pipeline pointer too.
    pub symbol: &'a str,
}

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

/// The most a recorded command can carry as a buffer offset, plus one.
const FOUR_GIB: u64 = 1 << 32;

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
    regions: &Regions,
    commands: &[Command<'_>],
) -> Result<Recording> {
    // The INDEX SPACE, not the count. `maxKernelBufferBindCount` bounds the
    // largest `[[buffer(n)]]` a recorded command may address, and a bind past
    // it is not an error -- Metal drops it and the kernel reads address zero.
    //
    // This took the widest `binds.len()`, which is the same number only when
    // every command binds slots 0..n with no gaps. A plan whose kernels leave
    // a hole -- an optional operand a deployment does not have, a scalar block
    // at a fixed high slot -- has a command that binds three buffers at slots
    // 0, 1 and 5, and `3` bounds it out of its own table.
    //
    // Measured: llama-3.2-1B replays bit-identically and gemma-4-31b and
    // gemma-4-26b came back ALL NaN -- 262144 of 262144 logits, from a
    // recording that reported success, because a dropped bind is silent.
    let widest = commands
        .iter()
        .flat_map(|c| c.binds.iter().map(|b| b.slot + 1))
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
                commands.len().max(1),
                MTLResourceOptions::StorageModeShared,
            )
    }
    .ok_or_else(|| Error::Create {
        what: "indirect command buffer",
        message: format!("the device declined {} commands", commands.len()),
    })?;

    // RESIDENT: the GPU reads its commands out of this buffer, and this
    // context tracks nothing automatically. Without it the execute faults.
    context
        .residency()
        .addAllocation(ProtocolObject::from_ref(&*icb));
    context.residency().commit();
    context.residency().requestResidency();

    for (index, command) in commands.iter().enumerate() {
        // SAFETY: `index` is below the command count declared above.
        let recorded = unsafe { icb.indirectComputeCommandAtIndex(index) };
        recorded.setComputePipelineState(command.pipeline);

        for bind in &command.binds {
            // UNRECORDABLE, not broken. A recorded command binds a BUFFER,
            // and an address in no registered allocation cannot be turned
            // into one -- but the encode path binds addresses and does not
            // care, so a caller that has not registered its regions is
            // un-optimised rather than wrong. Its own variant so the caller
            // swallows THIS and not the three faults beside it.
            let (buffer, offset) =
                regions
                    .resolve(bind.address)
                    .ok_or_else(|| Error::Unrecordable {
                        what: "fire",
                        message: format!(
                            "`{}` binds {:#x} at slot {}, which is in no registered \
                             allocation",
                            command.symbol, bind.address, bind.slot
                        ),
                    })?;
            // FOUR GIBIBYTES, and the device is why.
            // `setKernelBuffer:offset:atIndex:` takes an `NSUInteger` and
            // truncates it to 32 bits on this hardware --
            // `device_icb.rs::a_recorded_commands_buffer_offset_is_truncated_to_thirty_two_bits`
            // binds 4 GiB + 64 into a 5 GiB buffer and reads back byte 64.
            // The encode path binds raw GPU addresses and has no offset to
            // lose, so it is unaffected; only a recording is.
            //
            // It cost a whole debugging round to find, because what it does
            // is bind a weight to the wrong bytes of the right buffer: the
            // fire runs, every launch succeeds, and 262 144 of 262 144 logits
            // come back NaN. Refusing here turns that into an encoded fire --
            // slower, and right.
            //
            // A checkpoint reaches this the moment it is larger than 4 GiB,
            // which is every model this engine is for. Lifting it means
            // staging weights in chunks no larger than 4 GiB rather than the
            // one region `weights/stage.rs` allocates today.
            if offset >= FOUR_GIB {
                return Err(Error::Unrecordable {
                    what: "fire",
                    message: format!(
                        "`{}` binds slot {} at {offset} bytes into its buffer, and a \
                         recorded command truncates that to {}",
                        command.symbol,
                        bind.slot,
                        offset & 0xffff_ffff,
                    ),
                });
            }
            // SAFETY: the buffer outlives the recording (the registry retains
            // it) and the slot is below `maxKernelBufferBindCount`.
            unsafe {
                recorded.setKernelBuffer_offset_atIndex(buffer, offset as usize, bind.slot);
            }
        }

        recorded.concurrentDispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: command.grid[0] as usize,
                height: command.grid[1] as usize,
                depth: command.grid[2] as usize,
            },
            MTLSize {
                width: command.threadgroup[0] as usize,
                height: command.threadgroup[1] as usize,
                depth: command.threadgroup[2] as usize,
            },
        );
        // See the doc: a statement reading what the previous one wrote reads
        // whatever was there without this, and the failure is a number.
        recorded.setBarrier();
    }

    Ok(Recording {
        icb,
        commands: commands.len(),
    })
}
