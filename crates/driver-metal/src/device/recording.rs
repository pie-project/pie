//! A fire, recorded once.
//!
//! `encode` walks the dispatch list every step: a pipeline set, a bind per
//! operand, an argument-table set and a dispatch, then a barrier. Measured on
//! `qwen3_0_6b` that walk is 14.8 ms — 47.5 % of a prefill, 76.4 % of a decode
//! — and it is the same 14.8 ms every step, because it is proportional to the
//! dispatch count and that is a property of the text, not of the batch. This
//! records the walk into an [`MTLIndirectCommandBuffer`] instead, so a fire
//! costs one `executeCommandsInBuffer` on the host.
//!
//! Only three things differ between two fires of one `(plan, row shape)` — the
//! arena, the params and the fire tables — and `gpu::fire::Scratch` pools all
//! three, so their addresses are the previous fire's. See
//! `.wiki/driver/graph-metal.md` §4.
//!
//! # The three preconditions, each of which fails badly
//!
//! 1. **Pipelines must be compiled `supportIndirectCommandBuffers`.** Setting
//!    one that was not **faults**, before anything executes.
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

use crate::device::context::Context;
use crate::device::regions::Regions;
use crate::error::{Error, Result};

/// One address bound to one kernel slot.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct Bind {
    /// The GPU address, which [`Regions`] turns back into a buffer and an
    /// offset. A recorded command binds a BUFFER, not an address.
    pub address: u64,
    /// Which `[[buffer(n)]]` it lands in.
    pub slot: usize,
}

/// One compute command, in the vocabulary a recording is made of.
///
/// A recording needs a pipeline, some addresses and a grid — none of those
/// words is a fire's, which keeps `lowering/` and `gpu/bind/` (both ABOVE this
/// module) out of the cycle. The translation lives in
/// [`crate::bind::encode::commands`], where `Dispatch`, `Pipelines` and
/// `Params` are already in scope and the identical walk is done for encoding.
///
/// The pipeline borrow is deliberate: a `Command` cannot outlive the
/// `Pipelines` that compiled it, so a recording made from stale pipelines is
/// not constructible rather than merely discouraged.
#[derive(Debug)]
pub struct Command<'a> {
    /// Compiled `supportIndirectCommandBuffers`, which is precondition 1.
    pub pipeline: &'a ProtocolObject<dyn MTLComputePipelineState>,
    /// Every operand and scalar address this command binds, already resolved to
    /// slots. Flat, because a recorded command does not distinguish them.
    pub binds: Vec<Bind>,
    /// Threads, not threadgroups. See precondition 2.
    pub grid: [u32; 3],
    /// Threads per threadgroup.
    pub threadgroup: [u32; 3],
    /// For the error message only, so a refusal names the kernel. Not hashed:
    /// two fires differing in a symbol differ in a pipeline pointer too.
    pub symbol: &'a str,
    /// Whether this command must wait for the ones before it.
    ///
    /// The same statement `bind::encode::encode` makes, made once and read by
    /// both paths, because a recording that orders its commands differently
    /// from the encode it stands in for is the drift `device_icb.rs` catches.
    pub barrier: bool,
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
/// a command binds a **buffer**, not an address, so an unregistered operand is
/// a refusal rather than a bind to nothing.
///
/// A command carries a barrier when `bind::encode::commands` said it does.
/// Metal does not order two dispatches in one encoder, and three runs of one
/// fire without any barrier gave widest activations of 11.7, 23.1 and 4.5e12 —
/// two of the three looked plausible. Not every command, because the operand
/// directions say which ones can overlap.
///
/// # Errors
///
/// A dispatch whose pipeline is not compiled, whose scalars were not staged, or
/// whose operand address falls in no registered allocation.
pub fn record(context: &Context, regions: &Regions, commands: &[Command<'_>]) -> Result<Recording> {
    // The INDEX SPACE, not the count. `maxKernelBufferBindCount` bounds the
    // largest `[[buffer(n)]]` a recorded command may address, and a bind past it
    // is not an error -- Metal drops it and the kernel reads address zero.
    // The widest `binds.len()` is the same number only when every command binds
    // slots 0..n with no gaps; a plan whose kernels leave a hole binds three
    // buffers at slots 0, 1 and 5, and `3` bounds it out of its own table.
    // Measured: gemma-4-31b came back ALL NaN from a recording that reported
    // success, because a dropped bind is silent.
    let widest = commands
        .iter()
        .flat_map(|c| c.binds.iter().map(|b| b.slot + 1))
        .max()
        .unwrap_or(1);
    let descriptor = MTLIndirectCommandBufferDescriptor::new();
    // BOTH spellings: the type has to match the call the command makes, and
    // declaring one while calling the other silently does nothing.
    descriptor.setCommandTypes(MTLIndirectCommandType(
        MTLIndirectCommandType::ConcurrentDispatch.0
            | MTLIndirectCommandType::ConcurrentDispatchThreads.0,
    ));
    // FALSE, and it is what makes a fire recordable at all: 424 dispatches bind
    // DIFFERENT addresses to the same slots, so a command inheriting the
    // encoder's bindings would take whichever was bound last.
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

    // RESIDENT: the GPU reads its commands out of this buffer, and this context
    // tracks nothing automatically. Without it the execute faults.
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
            // UNRECORDABLE, not broken: an address in no registered allocation
            // cannot become a buffer, but the encode path binds addresses and
            // does not care, so an unregistered caller is un-optimised rather
            // than wrong. Its own variant so the caller swallows THIS only.
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
            // truncates it to 32 bits on this hardware. The encode path binds
            // raw GPU addresses and has no offset to lose; only a recording is
            // affected. What it does is bind a weight to the wrong bytes of the
            // right buffer: every launch succeeds and every logit is NaN.
            // A checkpoint reaches this the moment it is larger than 4 GiB.
            // Lifting it means staging weights in chunks no larger than 4 GiB
            // rather than the one region `weights/stage.rs` allocates today.
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
        // A statement reading what the previous one wrote reads whatever was
        // there without this, and the failure is a number. `bind::encode`
        // decided; a recording repeats its decision rather than making its own.
        if command.barrier {
            recorded.setBarrier();
        }
    }

    Ok(Recording {
        icb,
        commands: commands.len(),
    })
}
