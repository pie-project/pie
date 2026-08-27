//! The encode sink: the driver side of `kernels_metal::Encode`.
//!
//! A kernel entry hands this module a [`Fire`] — a shader path, an
//! entrypoint, and a grid — and a flat list of [`ArgValue`]s. Three things
//! happen and nothing else: the point is resolved to a compiled pipeline
//! ([`Pipelines`]), each argument is bound at its own index, and one
//! dispatch is encoded into the fire's open compute pass. **Encode only,
//! never sync** — decision #15, and on this plane it is structural rather
//! than a discipline: there is no synchronizing call in this file, and the
//! only one in the shell is [`Frame::commit`](crate::device::ctx::Frame).
//!
//! **THE ARGUMENT SPACE IS ONE FLAT POSITIONAL TABLE, GAPS INCLUDED.** A
//! Metal shader declares every parameter — device pointers and `constant`
//! scalars alike — at its own `[[buffer(n)]]`, and `args[i]` binds at index
//! `i`. That is why `kernels-metal`'s entries push
//! [`absent`](kernels_metal::Encode::absent) into the slots a shader's other
//! variant owns (`attn/kv_write.metal` skips 4 and 6–9) instead of omitting
//! them: an omitted slot would shift every argument after it. The sink
//! honours the same rule — a nil handle binds a nil buffer at its index and
//! the count is unchanged.
//!
//! **A scalar is bound by VALUE, not through a buffer.** `setBytes:length:`
//! copies the four or eight bytes into the encoder's own argument storage,
//! which is what a `const constant int&` parameter reads. Staging scalars
//! through a device buffer would be a second allocation per launch and a
//! second thing to keep alive until the command buffer retires.

use kernels_metal::{ArgValue, Encode, Fire, KernelError};

use crate::device::{Context, Handles, Pipelines, handles::NIL};
use crate::error::Fault;

#[cfg(target_vendor = "apple")]
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

/// One fire's encode sink: everything a dispatch needs, borrowed.
///
/// Built per fire and dropped with it, beside the `Run` it is handed to.
/// It owns nothing — the device, the pass, the pipeline cache and the handle
/// table all outlive it — which is what lets `Encode::fire` take `&self`.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Sink<'a> {
    device: &'a Context,
    frame: &'a crate::device::ctx::Frame,
    pipelines: &'a Pipelines,
    handles: &'a Handles,
}

impl<'a> Sink<'a> {
    /// Bind the four things a dispatch resolves through.
    #[must_use]
    pub fn new(
        device: &'a Context,
        frame: &'a crate::device::ctx::Frame,
        pipelines: &'a Pipelines,
        handles: &'a Handles,
    ) -> Sink<'a> {
        Sink {
            device,
            frame,
            pipelines,
            handles,
        }
    }

    /// A shell fault, restated in the vocabulary a kernel entry's caller
    /// speaks.
    ///
    /// The walk's signature carries [`KernelError`] and nothing else, so a
    /// device refusal discovered mid-encode has to arrive as one. The
    /// entrypoint is the op name — it is what a reader needs to find the
    /// launch — and the fault's own sentence is the detail, so nothing is
    /// lost but the variant.
    fn refuse(fire: Fire, fault: Fault) -> KernelError {
        KernelError::Backend {
            op: fire.entrypoint,
            detail: fault.to_string(),
        }
    }
}

impl Encode for Sink<'_> {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), KernelError> {
        #[cfg(target_vendor = "apple")]
        {
            let pipeline = self
                .pipelines
                .at(self.device.device(), fire)
                .map_err(|fault| Sink::refuse(fire, fault))?;
            let encoder = self.frame.encoder();
            encoder.setComputePipelineState(&pipeline);
            for (at, arg) in args.iter().enumerate() {
                self.bind(encoder, fire, at, *arg)?;
            }
            let lanes = MTLSize {
                width: fire.lanes[0].max(1) as usize,
                height: fire.lanes[1].max(1) as usize,
                depth: fire.lanes[2].max(1) as usize,
            };
            let group = if fire.group == [0, 0, 0] {
                crate::device::ctx::threadgroup(&pipeline, fire.lanes)
            } else {
                MTLSize {
                    width: fire.group[0].max(1) as usize,
                    height: fire.group[1].max(1) as usize,
                    depth: fire.group[2].max(1) as usize,
                }
            };
            encoder.dispatchThreads_threadsPerThreadgroup(lanes, group);
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = args;
            Err(Sink::refuse(fire, Fault::Deviceless))
        }
    }

    fn absent(&self) -> Result<ArgValue, KernelError> {
        Ok(ArgValue::Buffer(NIL))
    }
}

#[cfg(target_vendor = "apple")]
impl Sink<'_> {
    /// One argument at one index.
    fn bind(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        fire: Fire,
        at: usize,
        arg: ArgValue,
    ) -> Result<(), KernelError> {
        match arg {
            ArgValue::Buffer(handle) | ArgValue::BufferMut(handle) => {
                if handle == NIL {
                    // SAFETY: binding nil at an index the shader either does
                    // not declare or does not dereference on this arm — the
                    // `absent` contract, and what keeps the indices aligned.
                    unsafe { encoder.setBuffer_offset_atIndex(None, 0, at) };
                    return Ok(());
                }
                let binding = self.handles.get(handle).ok_or_else(|| {
                    Sink::refuse(
                        fire,
                        Fault::Unbound {
                            what: format!(
                                "handle {handle} at argument {at}, which this fire minted no row for"
                            ),
                        },
                    )
                })?;
                // SAFETY: the row retains its buffer, and its offset was
                // bounds-checked against that buffer when the row was minted.
                unsafe {
                    encoder.setBuffer_offset_atIndex(
                        Some(&*binding.slab().clone()),
                        usize::try_from(binding.offset()).expect("an offset inside a reservation"),
                        at,
                    );
                }
                Ok(())
            }
            ArgValue::I32(v) => self.scalar(encoder, &v, at),
            ArgValue::U32(v) => self.scalar(encoder, &v, at),
            ArgValue::F32(v) => self.scalar(encoder, &v, at),
            // `size_t` is 64 bits in MSL, which is what the pool's stride
            // seats are declared as.
            ArgValue::Usize(v) => self.scalar(encoder, &v, at),
        }
    }

    /// A scalar bound by value into the encoder's argument storage.
    fn scalar<T: Copy>(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        value: &T,
        at: usize,
    ) -> Result<(), KernelError> {
        // SAFETY: `value` is a live local of the caller's frame and
        // `setBytes:length:` copies out of it before returning.
        unsafe {
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::from(value).cast(),
                size_of::<T>(),
                at,
            );
        }
        Ok(())
    }
}
