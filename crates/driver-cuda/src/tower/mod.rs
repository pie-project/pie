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
//! [`fire`] of the row that already states that geometry; the arena becomes a
//! [`Scratch`]; the per-image loop becomes a `for`.
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
//! Every fire below names a row whose launch rule was written against the C++
//! launcher it replaces, and each call site carries the `<<<>>>` expression it
//! reproduces in a comment. Where a rule and a launcher disagree, that is a
//! finding and not a rule to add — §10.5 forbids growing the vocabulary for
//! one kernel. There is exactly one such divergence in this module and it is
//! stated at its call site (`gemma4_vision::rms`).
//!
//! # A failure is a refusal
//!
//! The C++ walk ended every CUDA call in `VCK(...)`, which threw
//! `std::runtime_error` — and an exception crossing the C ABI is undefined
//! behaviour that in practice reaches SIGABRT with no message. Every function
//! here returns [`crate::Result`] instead, and [`fire`] turns a refused launch
//! into an error naming the symbol rather than into a launch of something
//! else. Nothing here substitutes a kernel, retries at another geometry, or
//! treats a `Geometry` refusal as a no-op.
//!
//! `crate::bind::jit::fire` is deliberately NOT the entry used: it returns
//! `()` because a routed row has nowhere else to go, and swallowing a
//! `Geometry` or `Args` refusal is exactly wrong for a walk whose next launch
//! reads the buffer this one was supposed to write.

#![allow(clippy::print_stderr)]

use std::ffi::c_void;

use kernels_cuda_new::runtime::{Args, Launch, cache};
use kernels_cuda_new::{ArgValue, Dims, Stream, unit};

use crate::device::{Allocator, DeviceBuffer, StreamRef};
use crate::{Error, Result};

pub mod gemma4_audio;
pub mod gemma4_vision;
pub mod qwen3_vl;

/// Fire the row `symbol` names, refusing loudly.
///
/// `dims` is the fire's rectangle — the rule turns it into the grid and block
/// the C++ launcher spelled by hand. `values` are the row's operands in the
/// row's order, checked against the row by the JIT crate rather than trusted.
///
/// # Errors
///
/// Any refusal from [`kernels_cuda_new::fire`], with the symbol as the call:
/// no unit hosts it, its unit will not compile, the rule has no launch for
/// these dims, the values do not match the row, or the driver refused the
/// launch. Each is a stop, never a fallback.
pub(crate) fn fire(
    symbol: &'static str,
    dims: Dims,
    values: &[ArgValue],
    stream: StreamRef<'_>,
) -> Result<()> {
    // SAFETY: `stream` is a live `cudaStream_t` for as long as the borrow
    // lasts, which is the assertion `StreamRef` exists to carry; the pointer
    // operands address `Scratch` allocations and published weights, both live
    // until the caller synchronises. The same obligation the C++ walk made
    // implicitly at every `<<<>>>`.
    let outcome = unsafe {
        kernels_cuda_new::fire(
            symbol,
            dims,
            values,
            Stream::from_runtime(stream.as_raw().cast()),
        )
    };
    outcome.map_err(|why| Error::invalid(symbol, why.to_string()))
}

/// Fire the row `symbol` names at a grid THIS CALLER states.
///
/// [`fire`] hands a rectangle to the row's `LaunchRule` and lets the rule
/// compute the launch. Some launchers have no rule, and the honest reason is
/// written on their rows: three of gemma-4 audio's SSCP kernels put a channel
/// count on `grid.z`, which `Dims` has no field for, and `k_local_attn` puts
/// a TILE count on `grid.x` where every ported rule puts a count of things.
/// §10.5 forbids growing the vocabulary for one kernel, and the answer under
/// the owner's principle is not a new rule — it is that **the host computes
/// the grid, and the host is Rust**.
///
/// So this takes `grid`, `block` and `smem` from the caller, which quotes the
/// `dim3` it transcribed. `fire/attn_score.rs` has fired
/// `attn::attn_score_fold_heads` exactly this way since before the towers
/// moved; this is that path with the panics turned into refusals.
///
/// # Errors
///
/// No unit hosts `symbol`; the unit will not compile; the values do not match
/// the row; the driver refused the launch. A stop, never a fallback.
pub(crate) fn fire_stated(
    symbol: &'static str,
    grid: [u32; 3],
    block: [u32; 3],
    smem: u32,
    values: &[ArgValue],
    stream: StreamRef<'_>,
) -> Result<()> {
    let Some((index, unit)) = unit::unit_of(symbol) else {
        return Err(Error::invalid(symbol, "no JIT unit hosts this symbol"));
    };
    let Some(sig) = unit.row(symbol).map(|row| row.sig) else {
        return Err(Error::invalid(symbol, "the unit does not hold this row"));
    };
    let module = cache::module(index, unit)
        .map_err(|why| Error::invalid(symbol, format!("unit `{}`: {why}", unit.name)))?;
    let mut args =
        Args::bind(sig, values).map_err(|why| Error::invalid(symbol, why.to_string()))?;
    let launch = Launch { grid, block, smem };
    // SAFETY: as [`fire`] — the stream outlives the borrow and every pointer
    // operand addresses live device memory until the caller synchronises.
    let stream = unsafe { Stream::from_runtime(stream.as_raw().cast()) };
    module
        .fire(sig, launch, &mut args, stream)
        .map_err(|why| Error::invalid(symbol, why.to_string()))
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
        Self {
            alloc: Allocator::new(),
            live: Vec::new(),
        }
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

/// A pointer operand, `const` or not — every `Buf`, `BufMut`, `F32s`,
/// `F32sMut` and `I32s` cell of a row.
///
/// One helper rather than a `.cast_mut()` at forty argument positions, for
/// `gemma4_vision.cu`'s own reason for writing `D()`: a cast spelled out
/// thirty times is thirty places to get the constness wrong, and the compiler
/// cannot tell a wrong one from a right one.
pub(crate) fn p(pointer: *const c_void) -> ArgValue {
    ArgValue::Ptr(pointer.cast_mut())
}

/// A pointer operand from a scratch allocation, which is already `*mut`.
pub(crate) fn pm(pointer: *mut c_void) -> ArgValue {
    ArgValue::Ptr(pointer)
}

/// A rectangle of `rows` by `width` and nothing else — the `Dims` every
/// `Elementwise`, `Tile16`, `PerRow` and `Rms` fire in this module wants.
///
/// The other seven fields stay zero because no rule in this module reads
/// them; the two that do (`AxialRope`'s `kv_heads` and `head_dim`) are spelled
/// at their own call site, where the head count is visible.
pub(crate) fn rect(rows: u32, width: u32) -> Dims {
    Dims {
        rows,
        width,
        ..Dims::default()
    }
}
