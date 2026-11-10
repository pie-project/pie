#![allow(unsafe_code)]

use kernels_metal::{ArgValue, Encode, Error, Fire};

use crate::device::ctx::Frame;
use crate::device::{Context, Handles, Pipelines, handles::NIL};
use crate::error::Fault;

#[cfg(target_vendor = "apple")]
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Sink<'a> {
    device: &'a Context,
    frame: &'a Frame,
    pipelines: &'a Pipelines,
    handles: &'a Handles,
    /// The event a `signal` or `wait` fences the frame on; none unless this
    /// load hands work to the Neural Engine.
    #[cfg(target_vendor = "apple")]
    handoff: Option<&'a kernels_metal::ane::Handoff>,
    /// Launches left before the borrowed frame commits its head and goes on
    /// in a fresh buffer; zero never.
    flush_in: std::cell::Cell<u32>,
}

impl<'a> Sink<'a> {
    #[must_use]
    pub fn new(
        device: &'a Context,
        frame: &'a Frame,
        pipelines: &'a Pipelines,
        handles: &'a Handles,
    ) -> Sink<'a> {
        Sink {
            device,
            frame,
            pipelines,
            handles,
            #[cfg(target_vendor = "apple")]
            handoff: None,
            flush_in: std::cell::Cell::new(0),
        }
    }

    fn refuse(fire: Fire, fault: Fault) -> Error {
        Error::Backend {
            op: fire.entrypoint,
            detail: fault.to_string(),
        }
    }
}

static KERNEL_PROFILE: std::sync::Mutex<std::collections::BTreeMap<String, (u64, u64)>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

#[cfg(target_vendor = "apple")]
fn profiling() -> bool {
    crate::diag::on().kernel_profile.on()
}

#[must_use]
pub fn kernel_profile() -> Vec<(String, u64, u64)> {
    let mut rows: Vec<(String, u64, u64)> = KERNEL_PROFILE
        .lock()
        .expect("kernel profile lock poisoned")
        .iter()
        .map(|(name, &(ns, n))| (name.clone(), ns, n))
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    rows
}

pub fn reset_kernel_profile() {
    KERNEL_PROFILE
        .lock()
        .expect("kernel profile lock poisoned")
        .clear();
}

#[cfg(target_vendor = "apple")]
fn profile_key(entrypoint: &str, args: &[ArgValue]) -> String {
    if !crate::diag::on().kernel_profile.shaped() {
        return entrypoint.to_string();
    }
    let scalars: Vec<String> = args
        .iter()
        .filter_map(|arg| match arg {
            ArgValue::I32(v) => Some(v.to_string()),
            ArgValue::U32(v) => Some(v.to_string()),
            _ => None,
        })
        .collect();
    format!("{entrypoint} [{}]", scalars.join(","))
}

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
fn record_kernel(name: &str, seconds: f64) {
    let ns = (seconds * 1e9).max(0.0) as u64;
    let mut table = KERNEL_PROFILE.lock().expect("kernel profile lock poisoned");
    let row = table.entry(name.to_string()).or_insert((0, 0));
    row.0 += ns;
    row.1 += 1;
}

impl Encode for Sink<'_> {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Error> {
        #[cfg(target_vendor = "apple")]
        {
            let pipeline = self
                .pipelines
                .at(self.device.device(), fire)
                .map_err(|fault| Sink::refuse(fire, fault))?;
            if profiling() {
                let refuse = |fault: Fault| Sink::refuse(fire, fault);
                let own = self.device.frame().map_err(refuse)?;
                {
                    let encoder = own.encoder();
                    encoder.setComputePipelineState(&pipeline);
                    for (at, arg) in args.iter().enumerate() {
                        self.bind(&encoder, fire, at, *arg)?;
                    }
                    self.dispatch(&encoder, &pipeline, fire)?;
                }
                let seconds = own.commit_timed().map_err(refuse)?;
                record_kernel(&profile_key(fire.entrypoint, args), seconds);
                return Ok(());
            }
            {
                let encoder = self.frame.encoder();
                encoder.setComputePipelineState(&pipeline);
                for (at, arg) in args.iter().enumerate() {
                    self.bind(&encoder, fire, at, *arg)?;
                }
                self.dispatch(&encoder, &pipeline, fire)?;
            }
            let left = self.flush_in.get();
            if left > 0 {
                self.flush_in.set(left - 1);
                if left == 1 {
                    self.frame
                        .flush()
                        .map_err(|fault| Sink::refuse(fire, fault))?;
                }
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = args;
            Err(Sink::refuse(fire, Fault::Deviceless))
        }
    }

    fn absent(&self) -> Result<ArgValue, Error> {
        Ok(ArgValue::Buffer(NIL))
    }

    fn signal(&self, stamp: u64) -> Result<(), Error> {
        self.fence(stamp, true)
    }

    fn wait(&self, stamp: u64) -> Result<(), Error> {
        self.fence(stamp, false)
    }

    #[cfg(target_vendor = "apple")]
    fn on_commit(&self, then: Box<dyn FnOnce() + Send>) -> Result<(), Error> {
        self.frame.on_commit(then);
        Ok(())
    }
}

impl<'a> Sink<'a> {
    /// Commits the borrowed frame's head once `launches` are encoded, so the
    /// device starts on them while the rest of the fire is still encoding.
    #[must_use]
    pub fn flushing_after(self, launches: u32) -> Self {
        self.flush_in.set(launches);
        self
    }

    /// Arms the hand-off `signal` and `wait` fence on.
    #[cfg(target_vendor = "apple")]
    #[must_use]
    pub fn with_handoff(mut self, handoff: Option<&'a kernels_metal::ane::Handoff>) -> Self {
        self.handoff = handoff;
        self
    }

    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    fn fence(&self, stamp: u64, signal: bool) -> Result<(), Error> {
        let op = if signal {
            "handoff.signal"
        } else {
            "handoff.wait"
        };
        #[cfg(target_vendor = "apple")]
        {
            let handoff = self.handoff.ok_or(Error::Backend {
                op,
                detail: "no Neural Engine hand-off is armed for this encode".to_string(),
            })?;
            // The GPU raises `ready` and waits on `done`; each event has one
            // writer, so neither falls.
            let event = if signal {
                handoff.ready_event()
            } else {
                handoff.done_event()
            };
            let event =
                objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(&**event);
            self.frame.fence(event, stamp, signal)
                .map_err(|fault| Error::Backend {
                    op,
                    detail: fault.to_string(),
                })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Error::Backend {
                op,
                detail: Fault::Deviceless.to_string(),
            })
        }
    }
}

#[cfg(target_vendor = "apple")]
impl Sink<'_> {
    /// Launches `fire` on `encoder`: the threadgroup counts its indirect
    /// buffer holds, where it names one, else its lanes.
    #[cfg(target_vendor = "apple")]
    fn dispatch(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        pipeline: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
        fire: Fire,
    ) -> Result<(), Error> {
        let group = crate::device::ctx::group_of(pipeline, fire)
            .map_err(|fault| Sink::refuse(fire, fault))?;
        if let Some(indirect) = fire.indirect {
            let binding = self.handles.get(indirect.buf).ok_or_else(|| {
                Sink::refuse(
                    fire,
                    Fault::Unbound {
                        what: format!(
                            "handle {}, the fire's grid, which this fire minted no row for",
                            indirect.buf
                        ),
                    },
                )
            })?;
            let offset = usize::try_from(binding.offset() + u64::from(indirect.offset))
                .expect("an offset inside a reservation");
            // SAFETY: the row retains its buffer, and an earlier fire of the
            // frame wrote three threadgroup counts at the offset.
            unsafe {
                encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    &binding.slab().clone(),
                    offset,
                    group,
                );
            }
            return Ok(());
        }
        let lanes = MTLSize {
            width: fire.lanes[0].max(1) as usize,
            height: fire.lanes[1].max(1) as usize,
            depth: fire.lanes[2].max(1) as usize,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(lanes, group);
        Ok(())
    }

    fn bind(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        fire: Fire,
        at: usize,
        arg: ArgValue,
    ) -> Result<(), Error> {
        match arg {
            ArgValue::Buffer(handle) | ArgValue::BufferMut(handle) => {
                if handle == NIL {
                    // SAFETY: binds nil at an index the shader doesn't
                    // dereference on this arm (the `absent` contract).
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
                // SAFETY: the row retains its buffer; its offset was
                // bounds-checked when the row was minted.
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
            ArgValue::Usize(v) => self.scalar(encoder, &v, at),
        }
    }

    fn scalar<T: Copy>(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        value: &T,
        at: usize,
    ) -> Result<(), Error> {
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
