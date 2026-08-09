//! The multimodal towers' HOST WALKS, in Rust.
//!
//! # What this module is, and what it replaced
//!
//! A tower is a sequence of kernel launches over a scratch arena, with a
//! handful of host decisions between them — a trip count that comes from the
//! image, a group index computed from a patch position, a pooling divisor.
//! `new-horizon.md` §42 measured the only fact that matters about the
//! gemma-4 towers: *"a tower's `.cu` includes nine `.cuh` device headers and
//! every one of them was already in `kernels-cuda-new/csrc/src`. The archive
//! was never holding tower device code; it was compiling a host walk over
//! device code that had already left."*
//!
//! §42 then moved that host walk from `kernels-cuda/csrc` to
//! `driver-cuda/csrc`, which changed the archive it was compiled into and not
//! the language it was written in. This module is the other half: **every
//! line that runs on the CPU is Rust, and the only C++ left is device text
//! NVRTC compiles at run time.** A `<<<grid, block, smem, stream>>>` becomes a
//! [`call`] of the routine that already states that geometry; the arena
//! becomes a [`Scratch`]; the per-image loop becomes a `for`.
//!
//! [`qwen3_vl`] arrived by the same measurement taken a second time:
//! `driver-cuda/csrc/vision/qwen3_vl_tower.cu` held **zero `__global__` and
//! sixteen `<<<>>>`**, all sixteen naming kernels that were already rows. It
//! was a host program with a `.cu` extension, and porting it emptied
//! `driver-cuda/csrc/vision/` — nvcc no longer compiles anything for the
//! vision towers. Its one host callee that is still C++ is the FlashInfer
//! prefill, which north star §5 step 8 retires; see [`qwen3_vl::attn`].
//!
//! # The geometry is not invented here
//!
//! Every tower kernel's grid now lives beside the instantiation it launches,
//! in `kernels_cuda_new::x::vision`, where it was transcribed from the C++
//! launcher it replaces. What is left at a call site is the walk's own
//! vocabulary — which buffer, which extent, which weight — and the `<<<>>>`
//! citation is on the routine.
//!
//! The one numerics divergence this port carries is NOT a tower kernel and so
//! is not covered by that: it is stated at its call site
//! (`gemma4_vision::rms`).
//!
//! # A failure is a refusal
//!
//! The C++ walk ended every CUDA call in `VCK(...)`, which threw
//! `std::runtime_error` — and an exception crossing the C ABI is undefined
//! behaviour that in practice reaches SIGABRT with no message. Every function
//! here returns [`crate::Result`] instead, and [`call`] turns a refused launch
//! into an error naming the routine rather than into a launch of something
//! else. Nothing here substitutes a kernel, retries at another geometry, or
//! treats an empty extent as a no-op.
//!
//! `crate::bind::jit::fire` is deliberately NOT the entry used: it returns
//! `()` because a routed row has nowhere else to go, and swallowing a
//! geometry or an argument refusal is exactly wrong for a walk whose next
//! launch reads the buffer this one was supposed to write.

#![allow(clippy::print_stderr)]

use std::ffi::c_void;

use kernels_cuda_new::ArgValue;
use kernels_cuda_new::jit::Ctx;
use kernels_cuda_new::x::Refusal;

use crate::device::{Allocator, DeviceBuffer, StreamRef};
use crate::{Error, Result};

pub mod gemma4_audio;
pub mod gemma4_vision;
pub mod qwen3_vl;

/// Run one routine on this walk's stream, refusing loudly.
///
/// A routine picks its own instantiation and computes its own grid, so what a
/// call site states is the walk's half — the pointers, the extents — and the
/// name to put in front of a refusal. The name is not derivable: a
/// [`Refusal`] is a `Copy` value with no symbol in it, which is deliberate
/// (the same broken kernel is fired once per layer per token, so the detail
/// goes to the log once and the value stays cheap).
///
/// The closure is what keeps the [`Ctx`] scoped. `Ctx::on` is unsafe because
/// the stream must outlive every launch made through it; taking the body here
/// means the context cannot escape the borrow that carries that.
///
/// # Errors
///
/// Whatever the routine refuses — an empty extent, an extent too large for a
/// grid, or a compile, load or launch the device declined. Each is a stop,
/// never a fallback.
pub(crate) fn call(
    what: &'static str,
    stream: StreamRef<'_>,
    body: impl FnOnce(&Ctx) -> std::result::Result<(), Refusal>,
) -> Result<()> {
    // SAFETY: `stream` is a live `cudaStream_t` for as long as the borrow
    // lasts, which is the assertion `StreamRef` exists to carry; the pointer
    // operands address `Scratch` allocations and published weights, both live
    // until the caller synchronises. The same obligation the C++ walk made
    // implicitly at every `<<<>>>`.
    let ctx = unsafe { Ctx::on(stream.as_raw().cast()) };
    body(&ctx).map_err(|why| Error::invalid(what, format!("{why:?}")))
}

/// The tower's scratch arena — `gemma4_vision.cu`'s `DeviceScratch`.
///
/// A list of allocations freed together when the walk that made them ends,
/// which is what the C++ class was: a `std::vector<void*>` and a destructor
/// full of `cudaFree`. The allocations are handed out as raw pointers because
/// that is what a kernel argument is; they are valid until this value drops,
/// and every walk here drops it after synchronising the stream, in the order
/// the C++ destructor ran.
pub(crate) struct Scratch {
    /// Owns the frees. A fresh one per walk, exactly as the C++ arena was a
    /// fresh object per call — this is not the fire path's pooled allocator
    /// and does not want to be.
    alloc: Allocator,
    /// Every buffer handed out, held so it is not freed early.
    live: Vec<DeviceBuffer>,
}

impl Scratch {
    /// An empty arena.
    pub(crate) fn new() -> Self {
        Self { alloc: Allocator::new(), live: Vec::new() }
    }

    /// `count` elements of `width` bytes, uninitialised.
    fn raw(&mut self, count: usize, width: usize, what: &'static str) -> Result<*mut c_void> {
        let bytes = count
            .checked_mul(width)
            .ok_or_else(|| Error::invalid(what, "allocation size overflowed"))?;
        let buffer = self.alloc.alloc(bytes)?;
        let pointer = buffer.as_ptr();
        self.live.push(buffer);
        Ok(pointer)
    }

    /// `count` bf16 elements — `scratch.alloc<bf>(count)`.
    pub(crate) fn bf16(&mut self, count: usize) -> Result<*mut c_void> {
        self.raw(count, 2, "tower scratch (bf16)")
    }

    /// `count` fp32 elements — `scratch.alloc<float>(count)`.
    pub(crate) fn f32s(&mut self, count: usize) -> Result<*mut c_void> {
        self.raw(count, 4, "tower scratch (f32)")
    }

    /// `count` fp32 elements, zeroed on the stream — the arena allocation
    /// plus the `cudaMemsetAsync` that always followed it.
    pub(crate) fn zeroed_f32s(
        &mut self,
        count: usize,
        stream: StreamRef<'_>,
    ) -> Result<*mut c_void> {
        let bytes = count
            .checked_mul(4)
            .ok_or_else(|| Error::invalid("tower scratch (f32)", "allocation size overflowed"))?;
        let mut buffer = self.alloc.alloc(bytes)?;
        buffer.memset(0, stream)?;
        let pointer = buffer.as_ptr();
        self.live.push(buffer);
        Ok(pointer)
    }

    /// A host `f32` run uploaded on the stream — the arena allocation plus
    /// the `cudaMemcpyAsync(H2D)` that always followed it.
    pub(crate) fn upload_f32s(
        &mut self,
        src: &[f32],
        stream: StreamRef<'_>,
    ) -> Result<*mut c_void> {
        // SAFETY: `f32` has no padding and no invalid bit patterns, so its
        // bytes are readable as `u8` for the length of the slice. The
        // reinterpretation is a read of memory this call already owns.
        let bytes = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), src.len() * 4) };
        self.upload(bytes, stream, "tower scratch (f32 upload)")
    }

    /// A host `i32` run uploaded on the stream. The pooling group index is
    /// computed on the host, as it was in C++, because it is a division by a
    /// pooling kernel the tower knows and the patch grid does not.
    pub(crate) fn upload_i32s(
        &mut self,
        src: &[i32],
        stream: StreamRef<'_>,
    ) -> Result<*mut c_void> {
        // SAFETY: as `upload_f32s` — `i32` is plain data with no padding.
        let bytes = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), src.len() * 4) };
        self.upload(bytes, stream, "tower scratch (i32 upload)")
    }

    /// A host byte run uploaded on the stream, kept as bytes.
    ///
    /// The pixel plane arrives from the plan as `Vec<u8>` cut by a BYTE
    /// indptr; the C++ took a `const float*` and divided the offsets by four,
    /// which named a type it never dereferenced on the host. These are the
    /// same bytes and the division is gone.
    pub(crate) fn upload_bytes(
        &mut self,
        src: &[u8],
        stream: StreamRef<'_>,
    ) -> Result<*mut c_void> {
        self.upload(src, stream, "tower scratch (byte upload)")
    }

    /// The shared body of the three uploads.
    fn upload(
        &mut self,
        bytes: &[u8],
        stream: StreamRef<'_>,
        what: &'static str,
    ) -> Result<*mut c_void> {
        let mut buffer = self.alloc.alloc(bytes.len())?;
        buffer
            .copy_from_host(bytes, stream)
            .map_err(|why| Error::invalid(what, format!("host-to-device copy refused: {why}")))?;
        let pointer = buffer.as_ptr();
        self.live.push(buffer);
        Ok(pointer)
    }
}


