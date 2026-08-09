//! `cuLaunchKernel`, and the argument marshalling it needs.
//!
//! # Why the arguments are the whole content of this file
//!
//! `cuLaunchKernel` takes `void**` — an array of pointers to each argument's
//! *storage*, not the argument values. So passing a device pointer means
//! taking the address of the variable that holds it, and passing a `u32` means
//! taking the address of a `u32` that must outlive the call. Getting that one
//! level of indirection wrong does not fail: the driver reads whatever is at
//! the address it was given, and the kernel sees a plausible number.
//!
//! It is also completely unchecked. There is no arity check, no type check,
//! and no diagnostic — a sixteen-parameter kernel handed fifteen arguments
//! reads its sixteenth from uninitialised memory. So the marshalling lives
//! here, in one place, behind [`Args`], and every launcher builds its list
//! through it rather than assembling a `Vec<*mut c_void>` at the call site.
//!
//! # The one-CTA-per-lane rule
//!
//! A generated fused region is launched with `grid.x = lane_count` and
//! `block.x` from the compiled function's own attribute, because the kernel's
//! first line is `dispatch_lane = blockIdx.x`. It is not a tuning choice: two
//! lanes per block would have both write the same `commit_slot`, and a grid
//! smaller than the lane count silently drops the tail lanes.

use cudarc::driver::sys as dr;

use crate::cuda::{Allocator, StreamRef};
use crate::error::{Error, Result};

use super::control::Control;
use super::module::Module;
use super::ring::Rings;

/// A kernel's argument list, kept alive for the launch.
///
/// The storage and the pointer array are one value on purpose. `cuLaunchKernel`
/// dereferences the pointers *during* the call, so the scalars must outlive
/// it; a builder that returned only the `Vec<*mut c_void>` would compile and
/// would be reading freed stack by the time the driver looked.
#[derive(Default)]
pub struct Args {
    /// Boxed so that pushing another scalar cannot move an earlier one and
    /// invalidate a pointer already recorded in `slots`. A `Vec<u64>` would
    /// reallocate and leave every previous entry dangling — and the launch
    /// would still succeed, with the kernel reading whatever now lives there.
    ///
    /// Clippy calls this an unnecessary box, and it is wrong here for a reason
    /// worth stating: its rule is about the indirection being redundant when
    /// the only thing that matters is the value. What matters here is the
    /// ADDRESS, which is precisely what `Vec`'s reallocation does not preserve
    /// and `Box`'s does.
    #[allow(clippy::vec_box)]
    storage: Vec<Box<u64>>,
    slots: Vec<*mut std::ffi::c_void>,
}

impl Args {
    /// An empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a device pointer argument.
    pub fn ptr(&mut self, pointer: *mut std::ffi::c_void) -> &mut Self {
        self.scalar(pointer as u64)
    }

    /// Append a `u32` argument.
    ///
    /// Stored in a `u64` cell and pointed at its first four bytes, which is
    /// correct on every little-endian host — the only kind CUDA runs on, and
    /// the same assumption the ABI's own records make.
    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.scalar(u64::from(value))
    }

    /// Append a raw 64-bit argument.
    fn scalar(&mut self, value: u64) -> &mut Self {
        let mut cell = Box::new(value);
        let at: *mut u64 = &raw mut *cell;
        self.storage.push(cell);
        self.slots.push(at.cast());
        self
    }

    /// How many arguments have been appended.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether nothing has been appended.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The `void**` the driver takes.
    fn as_raw(&mut self) -> *mut *mut std::ffi::c_void {
        self.slots.as_mut_ptr()
    }
}

/// Launch `module`'s entry with `grid` blocks of `block` threads.
///
/// # Errors
///
/// If the driver refuses the launch. Note that a launch is asynchronous, so a
/// fault *inside* the kernel is not reported here — it surfaces at the next
/// synchronization, which is why every caller in this crate synchronizes
/// before it believes a result.
///
/// # Panics
///
/// Never. `expected` is checked against `args.len()` and returns an error.
pub fn launch(
    module: &Module,
    grid: u32,
    block: u32,
    args: &mut Args,
    expected: usize,
    stream: StreamRef<'_>,
) -> Result<()> {
    // The arity check CUDA does not do. A kernel handed too few arguments
    // reads the rest from whatever follows the array, and the failure appears
    // as a wrong answer rather than as an error.
    if args.len() != expected {
        return Err(Error::invalid(
            "cuLaunchKernel",
            format!(
                "'{}' takes {expected} arguments and {} were bound",
                module.entry_name(),
                args.len()
            ),
        ));
    }
    if grid == 0 {
        // A zero grid launches nothing and returns success, so a fire with no
        // lanes would look like a fire that ran.
        return Err(Error::invalid(
            "cuLaunchKernel",
            format!("'{}' launched with an empty grid", module.entry_name()),
        ));
    }
    // SAFETY: `module.function()` came from a loaded module and is live for
    // the borrow; `args` holds every scalar the pointer array points at for
    // the duration of this call; no shared memory is requested and no extra
    // block is passed.
    let code = unsafe {
        dr::cuLaunchKernel(
            module.function(),
            grid,
            1,
            1,
            block,
            1,
            1,
            0,
            stream.as_raw().cast(),
            args.as_raw(),
            std::ptr::null_mut(),
        )
    };
    if code != dr::CUresult::CUDA_SUCCESS {
        return Err(Error::Driver {
            call: "cuLaunchKernel",
            code,
        });
    }
    Ok(())
}

/// The two control kernels, launched.
///
/// A module of its own rather than free functions because the pair share a
/// shape no other launch has: both take the four ring arrays, both take two
/// channel-index lists that have to be uploaded, and both are single-thread.
pub mod launch_control {
    use super::{Allocator, Args, Control, Error, Result, Rings, StreamRef, launch};

    /// Ask whether a pass may commit.
    ///
    /// `need_full` are the channels the pass consumes — their committed cell
    /// must hold a value — and `need_empty` the ones it produces, which need
    /// room. Returns what the kernel decided.
    ///
    /// The verdict is read back rather than left on the device because the
    /// host has to know: a blocked fire is reported to the runtime as a retry,
    /// and that decision cannot be made on the GPU.
    ///
    /// # Errors
    ///
    /// If an upload, the launch, or the readback fails.
    pub fn readiness(
        control: &Control,
        rings: &Rings,
        need_full: &[u32],
        need_empty: &[u32],
        alloc: &Allocator,
        stream: StreamRef<'_>,
    ) -> Result<bool> {
        // The flag starts at 1 and the kernel ANDs into it, which is what lets
        // several stages narrow one pass's verdict without any of them needing
        // to know about the others.
        let mut pass = alloc.alloc(size_of::<u32>())?;
        pass.copy_from_host(&1u32.to_le_bytes(), stream)?;
        let full_list = upload_indices(alloc, need_full, stream)?;
        let empty_list = upload_indices(alloc, need_empty, stream)?;

        let mut args = Args::new();
        args.ptr(rings.full_ptr())
            .ptr(rings.head_ptr())
            .ptr(rings.tail_ptr())
            .ptr(rings.cap1_ptr())
            .ptr(
                full_list
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |b| b.as_ptr()),
            )
            .u32(u32::try_from(need_full.len()).map_err(too_many)?)
            .ptr(
                empty_list
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |b| b.as_ptr()),
            )
            .u32(u32::try_from(need_empty.len()).map_err(too_many)?)
            .ptr(pass.as_ptr());
        launch(control.readiness(), 1, 1, &mut args, 9, stream)?;

        let mut verdict = [0u8; 4];
        pass.copy_to_host(&mut verdict, stream)?;
        stream.synchronize()?;
        Ok(u32::from_le_bytes(verdict) != 0)
    }

    /// Advance the cursors of a pass that ran.
    ///
    /// `committed` is the readiness verdict. When it is false every kernel
    /// still launches — that is the dummy run, and it is what makes a blocked
    /// fire cost the same as a running one instead of branching on the device
    /// — and this call moves nothing.
    ///
    /// # Errors
    ///
    /// If an upload or the launch fails.
    pub fn commit(
        control: &Control,
        rings: &Rings,
        taken: &[u32],
        put: &[u32],
        committed: bool,
        alloc: &Allocator,
        stream: StreamRef<'_>,
    ) -> Result<()> {
        let mut pass = alloc.alloc(size_of::<u32>())?;
        pass.copy_from_host(&u32::from(committed).to_le_bytes(), stream)?;
        let taken_list = upload_indices(alloc, taken, stream)?;
        let put_list = upload_indices(alloc, put, stream)?;

        let mut args = Args::new();
        args.ptr(rings.full_ptr())
            .ptr(rings.head_ptr())
            .ptr(rings.tail_ptr())
            .ptr(rings.cap1_ptr())
            .ptr(
                taken_list
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |b| b.as_ptr()),
            )
            .u32(u32::try_from(taken.len()).map_err(too_many)?)
            .ptr(
                put_list
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |b| b.as_ptr()),
            )
            .u32(u32::try_from(put.len()).map_err(too_many)?)
            .ptr(pass.as_ptr());
        launch(control.commit(), 1, 1, &mut args, 9, stream)?;
        // The flag buffer must outlive the launch, and the launch is
        // asynchronous. Synchronizing here is the cheap correct answer while
        // there is one fire in flight; a pipelined shell would keep the
        // allocation alive against the stream instead.
        stream.synchronize()?;
        Ok(())
    }

    /// Upload a channel-index list, or `None` when it is empty.
    ///
    /// `None` rather than a zero-byte allocation: the kernel reads the array
    /// only `count` times, so a null with a count of zero is exactly correct
    /// and an empty allocation is a pointer the allocator may or may not
    /// return.
    fn upload_indices(
        alloc: &Allocator,
        indices: &[u32],
        stream: StreamRef<'_>,
    ) -> Result<Option<crate::cuda::DeviceBuffer>> {
        if indices.is_empty() {
            return Ok(None);
        }
        let bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        let mut buffer = alloc.alloc(bytes.len())?;
        buffer.copy_from_host(&bytes, stream)?;
        Ok(Some(buffer))
    }

    fn too_many(_: std::num::TryFromIntError) -> Error {
        Error::invalid("ptir::control", "more channels than a u32 can count")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every scalar must still be where its recorded pointer says it is after
    /// more have been appended. A `Vec<u64>` backing would reallocate and
    /// leave the earlier pointers dangling — and the launch would succeed,
    /// with the kernel reading whatever now lives at those addresses.
    #[test]
    fn appending_an_argument_does_not_move_the_ones_already_bound() {
        let mut args = Args::new();
        for value in 0..64u32 {
            args.u32(value);
        }
        assert_eq!(args.len(), 64);
        for (index, slot) in args.slots.iter().enumerate() {
            // SAFETY: each slot points at a `Box<u64>` this `Args` still owns.
            let seen = unsafe { *slot.cast::<u64>() };
            assert_eq!(
                seen, index as u64,
                "argument {index} moved when later ones were appended"
            );
        }
    }

    /// A pointer argument is the ADDRESS OF the pointer, not the pointer. One
    /// level of indirection either way is a plausible number the kernel reads
    /// without complaint.
    #[test]
    fn a_pointer_argument_is_bound_by_address_and_not_by_value() {
        let target = 0xdead_beefu64;
        let mut args = Args::new();
        args.ptr(target as *mut std::ffi::c_void);
        // SAFETY: the slot points at the `Box<u64>` holding the pointer value.
        let stored = unsafe { *args.slots[0].cast::<u64>() };
        assert_eq!(
            stored, target,
            "the slot must hold the pointer, and the slot's own address is \
             what cuLaunchKernel receives"
        );
    }
}
