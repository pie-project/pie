//! The persistent arena of an executed `LoadPlan`, living on the DEVICE.
//!
//! `model_loader::executor::host` decides everything about a load — which file
//! extents to read, which transforms to run, where each tensor lands — and
//! bottoms out in three verbs on an [`ArenaBacking`]. This is those three
//! verbs against CUDA global memory.
//!
//! Which is the whole of the "device load plan executor" the C++ tree spent
//! `load_plan_executor.hpp` (629 lines), `weight_copy_engine.hpp` (399) and
//! the host half of `transcode_engine.hpp` on. The executor was never the
//! device-specific part; the *addressing* was.
//!
//! **Host memory stays bounded by the largest single write**, which is what a
//! 39 GB checkpoint needs: the executor reads one file extent at a time and
//! hands it here, and here it goes straight across. Nothing accumulates.
//!
//! # Two things this does beyond addressing
//!
//! **The staging buffer is PINNED.** `cudaMemcpyAsync` out of pageable memory
//! is asynchronous in name only — the runtime stages it internally and the
//! call blocks — which `cuda::PinnedBuf`'s own doc says at length, and which
//! made every byte of a load cross at roughly half the achievable rate with
//! no overlap at all. The executor hands over a borrowed `&[u8]` it reuses,
//! so the bytes have to be copied somewhere before the copy can be left in
//! flight; copying them into pinned memory is that somewhere, and it is what
//! turns the H2D into a real async transfer.
//!
//! Two slots, alternated: one can be in flight while the executor fills the
//! next. A write larger than a slot bypasses staging and goes synchronously,
//! because a 39 GB checkpoint must not be able to make this allocate
//! proportionally to a tensor.
//!
//! **`Cast` and `Scale` run HERE**, on the device, when both operands are
//! already in the arena. The host path for those is a device read that
//! synchronizes, an arithmetic loop, and a device write — a full round trip
//! to compute something `kernels-cuda` has a kernel for. See
//! [`ArenaBacking::tile_map_caps`].

use std::borrow::Cow;

use model_loader::error::Error;
use model_loader::executor::arena::{ArenaBacking, TileMapOp};
use model_loader::plan::TileMapKind;
#[cfg(feature = "bridge")]
use model_loader::plan::{TILE_MAP_CAST, TILE_MAP_ENCODE, TILE_MAP_SCALE};
#[cfg(feature = "bridge")]
use model_loader::types::{DType, Encoding, QuantScheme};

use crate::cuda::{DeviceBuffer, OwnedStream, PinnedBuf};

/// How much pinned host memory one staging slot holds, when the caller
/// states no budget of its own.
///
/// Only a ceiling. [`DeviceArena::new`] takes the plan's `max_tile_bytes` and
/// pins the smaller of the two, because pinned memory is a scarce global
/// resource and a small model must not reserve as if it were a large one.
const STAGING_SLOT_CEILING: usize = 32 * 1024 * 1024;

/// One pinned staging slot and the event that says its copy has landed.
///
/// The event is per SLOT rather than per copy because that is the question
/// asked of it: "may I overwrite these bytes yet". A stream-wide drain
/// answers a stricter question and costs the overlap.
struct StagingSlot {
    buf: PinnedBuf,
    done: crate::cuda::Event,
}

/// A `LoadPlan`'s persistent arena as one CUDA allocation.
pub struct DeviceArena {
    buf: DeviceBuffer,
    stream: OwnedStream,
    /// Pinned staging, alternated so one copy can be in flight while the
    /// executor fills the next slot. Empty when pinning failed — the writes
    /// then take the pageable path they always took, which is slower and
    /// correct.
    staging: Vec<StagingSlot>,
    next_slot: usize,
    /// Whether transforms may run on the device. `false` forces the host
    /// path for every `TileMap`, which is what makes an A/B against the host
    /// executor possible without rebuilding. Read only where there is a
    /// bridge to call through.
    #[cfg_attr(not(feature = "bridge"), allow(dead_code))]
    device_transforms: bool,
    /// Transforms this backing was offered and declined, so they ran on the
    /// host. Reported by [`Self::declined_transforms`] at the end of a load:
    /// a decline is otherwise invisible, and "the loader transforms on the
    /// GPU" would be a claim nothing could contradict.
    declined: u32,
}

impl DeviceArena {
    /// Allocate `bytes` of device memory to execute a plan into.
    ///
    /// `max_write_bytes` is the plan's `target.max_tile_bytes` — the largest
    /// single `write` the executor can make. The staging slots are sized to
    /// the smaller of it and [`STAGING_SLOT_CEILING`], so a small model pins
    /// a small pool and a write that would not fit takes the pageable path
    /// rather than growing one.
    ///
    /// # Errors
    ///
    /// The device could not satisfy the allocation, or a stream to order the
    /// copies on could not be created.
    pub fn new(
        bytes: usize,
        max_write_bytes: usize,
        alloc: &crate::cuda::Allocator,
    ) -> Result<Self, Error> {
        let buf = alloc.alloc(bytes).map_err(device)?;
        let stream = OwnedStream::new(0).map_err(device)?;
        let slot = max_write_bytes.clamp(1, STAGING_SLOT_CEILING);
        // Best effort: a driver that cannot pin two slots still loads. The
        // event is recorded before first use is possible, so a slot's first
        // `synchronize` is on an event that has never been recorded --
        // `cudaEventSynchronize` on such an event returns immediately, which
        // is the answer we want ("nothing is reading this slot yet").
        let staging = (0..2)
            .map(|_| {
                Ok(StagingSlot {
                    buf: PinnedBuf::new(slot)?,
                    done: crate::cuda::Event::new()?,
                })
            })
            .collect::<Result<Vec<_>, crate::Error>>()
            .unwrap_or_default();
        Ok(Self {
            buf,
            stream,
            staging,
            next_slot: 0,
            device_transforms: device_transforms_enabled(),
            declined: 0,
        })
    }

    /// How many transforms ran on the host because this backing declined
    /// them. Zero is the answer a fully device-side load gives.
    #[must_use]
    pub const fn declined_transforms(&self) -> u32 {
        self.declined
    }

    /// The same arena with device transforms forced off.
    ///
    /// The host path is the reference implementation, so this is how a caller
    /// gets the two answers to compare. It is also the honest response to a
    /// checkpoint that trips a kernel: load it on the host and report it.
    #[must_use]
    pub fn host_transforms_only(mut self) -> Self {
        self.device_transforms = false;
        self
    }

    /// The filled arena, once the plan has run.
    ///
    /// Takes `self` because a plan is executed once: handing the buffer back
    /// while the backing still exists would let a second execution write under
    /// the weights the first one published.
    ///
    /// # Errors
    ///
    /// The stream faulted while draining the writes.
    pub fn into_buffer(self) -> Result<DeviceBuffer, Error> {
        self.stream.as_ref().synchronize().map_err(device)?;
        Ok(self.buf)
    }
}

/// Whether device-side load transforms are on. `PIE_LOADER_DEVICE_TRANSFORMS=0`
/// turns them off.
///
/// Defaulted ON: the device path is the one this module exists for, and an
/// env var that must be set to get the intended behaviour is a footgun. The
/// off switch is here for bisecting a numerical disagreement against the host
/// executor without a rebuild.
fn device_transforms_enabled() -> bool {
    !matches!(
        std::env::var("PIE_LOADER_DEVICE_TRANSFORMS").as_deref(),
        Ok("0")
    )
}

fn device(e: crate::Error) -> Error {
    Error::Contract(format!("device arena: {e:?}"))
}

impl ArenaBacking for DeviceArena {
    fn len(&self) -> usize {
        self.buf.len()
    }

    /// A device read is a STAGING COPY, and it synchronizes.
    ///
    /// Both are acceptable here and neither is on the write path: the executor
    /// reads the arena only to feed a transform whose input is a tensor it
    /// already wrote there, while it writes the arena once per file extent.
    fn read(&self, offset: usize, len: usize) -> Result<Cow<'_, [u8]>, Error> {
        let mut out = vec![0u8; len];
        self.buf
            .read_at(offset, &mut out, self.stream.as_ref())
            .map_err(device)?;
        self.stream.as_ref().synchronize().map_err(device)?;
        Ok(Cow::Owned(out))
    }

    /// Staged through PINNED memory, enqueued, not awaited —
    /// [`Self::into_buffer`] drains it.
    ///
    /// `bytes` is host memory the executor owns and reuses, so a copy left in
    /// flight out of it would race the next extent read. Copying into a
    /// pinned slot first is what lets the copy actually stay in flight (see
    /// the module doc); alternating two slots is what makes that useful,
    /// because the executor can fill one while the other is crossing.
    ///
    /// **A slot is only waited on when it is REUSED**, and then only for its
    /// own copy — that is the whole overlap. Draining the stream here instead
    /// would pay the extra `memcpy` and buy nothing.
    ///
    /// The `memcpy` into the slot is not waste either: it replaces the
    /// staging copy the CUDA runtime performs internally for a pageable
    /// source, which it does at a synchronization point rather than off one.
    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        let stream = self.stream.as_ref();
        let slot_bytes = self.staging.first().map_or(0, |slot| slot.buf.len());
        if bytes.len() > slot_bytes {
            // Larger than a slot, or nothing pinned: the pageable path, which
            // is synchronous in effect and is what every write used to be.
            // Ordered on the same stream, so it cannot pass a staged copy.
            return self.buf.write_at(offset, bytes, stream).map_err(device);
        }
        let slot = self.next_slot;
        self.next_slot = (slot + 1) % self.staging.len();
        let staged = &mut self.staging[slot];
        // The copy that last read this slot. With two slots this is the
        // copy-before-last, so the wait is for a transfer that has had a
        // whole extent read to finish in.
        staged.done.synchronize().map_err(device)?;
        staged.buf.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        let src = &staged.buf.as_slice()[..bytes.len()];
        self.buf.write_at(offset, src, stream).map_err(device)?;
        stream.record(&staged.done).map_err(device)
    }

    fn fill(&mut self, offset: usize, len: usize, byte: u8) -> Result<(), Error> {
        let stream = self.stream.as_ref();
        self.buf.memset_at(offset, len, byte, stream).map_err(device)
    }

    /// `Cast`, `Scale` and `Encode`, when the launch bridge is built and
    /// `PIE_LOADER_DEVICE_TRANSFORMS` has not turned them off.
    ///
    /// Without `feature = "bridge"` there is no `launch::ffi` to call, so the
    /// honest answer is zero and every transform takes the host path. That is
    /// the same degradation as `PIE_LOADER_DEVICE_TRANSFORMS=0` and needs no
    /// separate handling anywhere: a backing that claims nothing is asked for
    /// nothing.
    ///
    /// `Encode` was left out of an earlier version of this list on the stated
    /// grounds that `kernels-cuda` "has the dequantizing half of the pair and
    /// not the quantizing half". **That was false.**
    /// `csrc/src/quant/quant_bf16_to_mxfp4.cu` and `quant_bf16_to_fp8.cu`
    /// have implemented it all along — the latter's header even names this
    /// caller — and `transcode.cu` implements the fused FP8→MXFP4 that
    /// `TransformFusion::Fp8ToMxfp4` is the plan's word for. What was missing
    /// was a row in the kernel table, so no `pie_k_*` symbol existed to call.
    /// The rows are there now; the fusion is not yet reached, because
    /// choosing it needs the source scales this op does not carry.
    fn tile_map_caps(&self) -> u32 {
        #[cfg(feature = "bridge")]
        if self.device_transforms {
            return TILE_MAP_CAST | TILE_MAP_SCALE | TILE_MAP_ENCODE;
        }
        0
    }

    fn run_tile_map(&mut self, op: &TileMapOp<'_>) -> Result<bool, Error> {
        #[cfg(feature = "bridge")]
        {
            let ran = match op.kind {
                TileMapKind::Cast => self.cast(op)?,
                TileMapKind::Scale => self.scale(op)?,
                TileMapKind::Encode => self.encode(op)?,
                // Not claimed, so not reached. Declining stays the right
                // answer if `tile_map_caps` and this match ever drift apart.
                _ => false,
            };
            if !ran {
                // A decline is a SILENT performance cliff otherwise: the load
                // completes, the bytes are right, and every transform quietly
                // ran on the host. Counting them is what lets a driver say so
                // at the end of a load instead of leaving "we transform on
                // the GPU now" unfalsifiable.
                self.declined += 1;
            }
            return Ok(ran);
        }
        #[cfg(not(feature = "bridge"))]
        {
            let _ = op;
            Ok(false)
        }
    }
}

#[cfg(feature = "bridge")]
impl DeviceArena {
    /// The device address `bytes` into the arena.
    fn at(&self, offset: usize) -> *mut std::ffi::c_void {
        // SAFETY: every span the executor hands over was resolved against
        // `ArenaBacking::len`, which is this buffer's length.
        unsafe { self.buf.as_ptr().byte_add(offset) }
    }

    /// `quant::cast_fp32_to_bf16`, the one cast the kernel table implements.
    ///
    /// Any other dtype pair is DECLINED, not approximated. A cast this has no
    /// kernel for must never become a copy: the bytes would be the source's
    /// representation under the destination's name, which no later stage can
    /// detect. The host executor casts it instead.
    fn cast(&mut self, op: &TileMapOp<'_>) -> Result<bool, Error> {
        let (Encoding::Raw(src), Encoding::Raw(dst)) = (&op.src_encoding, &op.dst_encoding) else {
            return Ok(false);
        };
        if (*src, *dst) != (DType::F32, DType::BF16) {
            return Ok(false);
        }
        let elems = op.src.len / 4;
        if elems * 4 != op.src.len || elems * 2 != op.dst.len {
            return Ok(false);
        }
        let (src, dst) = (self.at(op.src.offset), self.at(op.dst.offset));
        unsafe {
            crate::launch::ffi::pie_k_quant_cast_fp32_to_bf16(
                src.cast_const(),
                dst,
                elems,
                self.stream.as_ref().as_raw().cast(),
            );
        }
        Ok(true)
    }

    /// `quant::scale_rows_bf16`, the per-row multiply.
    ///
    /// Declines everything the kernel does not cover: a uniform factor (no
    /// operand to read, and the table has no scalar-multiply row), a
    /// non-bf16 operand, a destination that is not the source (the kernel
    /// multiplies IN PLACE), or a shape the plan does not state as 2-D. Each
    /// is a plan the host executor runs correctly.
    fn scale(&mut self, op: &TileMapOp<'_>) -> Result<bool, Error> {
        let (Some(factors), Some((rows, cols))) = (op.factors, op.shape) else {
            return Ok(false);
        };
        if op.src_encoding != Encoding::Raw(DType::BF16)
            || op.dst_encoding != Encoding::Raw(DType::BF16)
            || op.src != op.dst
        {
            return Ok(false);
        }
        let (Ok(rows), Ok(cols)) = (i32::try_from(rows), i32::try_from(cols)) else {
            return Ok(false);
        };
        unsafe {
            crate::launch::ffi::pie_k_quant_scale_rows_bf16(
                self.at(op.dst.offset),
                self.at(factors.offset).cast_const(),
                rows,
                cols,
                self.stream.as_ref().as_raw().cast(),
            );
        }
        Ok(true)
    }

    /// Runtime quantization: `quant::quantize_bf16_to_{mxfp4_e2m1_per_block,
    /// fp8_e4m3_per_channel}`.
    ///
    /// The kernels the loader's `Encode` was written for — `quant_bf16_to_fp8.hpp`
    /// says so in its own header — and which ran on the host until the table
    /// gained a row for them.
    ///
    /// Declines a target with no per-block/per-channel kernel, a non-bf16
    /// source, a missing scale destination, or a shape the plan does not
    /// state as 2-D. The MXFP4 kernel additionally wants `cols % 32 == 0`,
    /// which is its block, and refuses to guess otherwise.
    fn encode(&mut self, op: &TileMapOp<'_>) -> Result<bool, Error> {
        let (Some(scales), Some((rows, cols))) = (op.dst_scales, op.shape) else {
            return Ok(false);
        };
        if op.src_encoding != Encoding::Raw(DType::BF16) {
            return Ok(false);
        }
        let (Ok(rows), Ok(cols)) = (i32::try_from(rows), i32::try_from(cols)) else {
            return Ok(false);
        };
        let stream = self.stream.as_ref().as_raw().cast();
        let (payload, scale_out) = (self.at(op.dst.offset), self.at(scales.offset));
        match op.transform.to {
            Some(QuantScheme::Mxfp4E2M1E8M0) => {
                if cols % 32 != 0 {
                    return Ok(false);
                }
                unsafe {
                    crate::launch::ffi::pie_k_quant_quantize_bf16_to_mxfp4_e2m1_per_block(
                        self.at(op.src.offset).cast_const(),
                        payload.cast::<u8>(),
                        scale_out.cast::<u8>(),
                        rows,
                        cols,
                        stream,
                    );
                }
                Ok(true)
            }
            Some(QuantScheme::Fp8E4M3) => {
                unsafe {
                    crate::launch::ffi::pie_k_quant_quantize_bf16_to_fp8_e4m3_per_channel(
                        self.at(op.src.offset).cast_const(),
                        payload.cast::<u8>(),
                        scale_out.cast::<f32>(),
                        rows,
                        cols,
                        stream,
                    );
                }
                Ok(true)
            }
            // Every other target, including the fused FP8->MXFP4 that
            // `transcode.cu` implements: reaching it needs the SOURCE's block
            // scales, which an `Encode` whose input is already a bf16 buffer
            // does not have. The host path runs those.
            _ => Ok(false),
        }
    }
}
