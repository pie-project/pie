//! Emit the Tier A device typecheck TU on stdout.
//!
//! The generated file is what the offline build compiles: it includes the
//! templates and takes the address of every instantiation the rows name --
//! which both CHECKS them and EMITS them. An example
//! rather than a build-script step because Tier A is a pilot -- the CMake
//! that would emit it as part of `native` is the change this measurement is
//! meant to justify, not one to make ahead of it.
//!
//! ```text
//! cargo run -p kernels-cuda --example emit_device_typecheck > device_typecheck.cu
//! nvcc -std=c++20 -arch=sm_89 -fatbin \
//!      -Xcompiler=-iquote,crates/kernels-cuda-new/csrc/src \
//!      -Xcompiler=-iquote,crates/kernels-cuda-new/csrc/shim \
//!      device_typecheck.cu -o tier_a.fatbin
//! ```
//!
//! # `-Xcompiler=-iquote`, and NOT `-I`, and this is measured
//!
//! An earlier version of this comment said `-I` was right here because *"a
//! generated TU includes nothing angled that a shim could shadow"*. The TU
//! does not — but **shadowing is transitive through the `.cuh` it includes**,
//! and the header set has five shims that shadow real toolkit headers:
//! `cuda_fp16.h`, `cuda_bf16.h`, `cuda_fp8.h`, `cuda_fp4.h` and
//! `cooperative_groups.h`. `-I` is the ANGLE-BRACKET path, so a `.cuh` that
//! reaches `<cuda_fp16.h>` gets the NVRTC shim under nvcc, where `__half`
//! collapses to `device::f16`.
//!
//! The failure is not a compile error. Measured on this box, same source, same
//! flags but for the include spelling:
//!
//! ```text
//! -Xcompiler=-iquote,…   49,480 B   bf16_to_narrow<__half>
//! -I …                   37,816 B   bf16_to_narrow<pie_cuda_driver::kernels::device::f16>
//! ```
//!
//! Both compiled. The `<<<>>>` in that translation unit wrote `<__half>`, and
//! under `-I` the object exports the **other instantiation** — a different
//! mangled symbol, 31% smaller, silently. Since the JIT half of this pair
//! finds a kernel by naming its instantiation to `nvrtcAddNameExpression`, an
//! archive built the `-I` way exports symbols the JIT would name differently,
//! and this file's whole job is to make that kind of drift a build failure.
//!
//! Today's seven entries reach only `norm/altup_aux.cuh` -> `pie_device.cuh`,
//! neither of which includes an angled toolkit header, so both spellings
//! currently produce a **byte-identical 41,504-byte fatbin**. That is why the
//! wrong reason survived: it was true, for a reason that stops holding the
//! moment this file covers a header that does. Extending it over the unit set
//! is exactly the proposal on the table.
//!
//! `nvcc -iquote` is rejected outright (`nvcc fatal: Unknown option
//! '-iquote'`), which is why it must go through `-Xcompiler` and why a probe
//! reaches for `-I` and gets a quiet wrong answer instead of a loud one.
//! `csrc/CMakeLists.txt` spelled the archive's copy the same awkward way, for
//! the same measured reason — and that file is DELETED, with the archive it
//! configured and the `enable_language(CUDA)` that was the last thing in this
//! workspace to look for nvcc. The `-iquote` measurement is kept because it is
//! a fact about nvcc rather than about our build, and because this example's
//! whole subject is which spelling a device compile is handed.
//!
//! # Why this names `kernels_cuda_new::` and not `kernels_cuda::`
//!
//! It used to read `kernels_cuda::abi::emit_device_typecheck(
//! kernels_cuda::norm_device::ENTRIES)`, through two one-line `pub use`
//! shims in this crate's `src/`. Both are deleted, and the reason is the
//! shape `lib.rs` already names four instances of.
//!
//! `lib.rs` records the sweep that emptied it: *"Across the whole workspace
//! there is not one `use kernels_cuda`, not one `kernels_cuda::` outside a
//! doc comment."* That was measured over `src/` and `tests/` and **it did not
//! include `examples/`** — so the one live consumer of the re-exports was the
//! one directory the census did not walk, and `src/lib.rs` was left with no
//! `mod` declaration for the two files this line reached through. The paths
//! it named had already stopped existing.
//!
//! A fifth instance of "a reason that is written down, is checkable, and is
//! false", found the same way as the other four: by running the discriminator
//! across the whole set instead of the part that was in front of us.

fn main() {
    match kernels_cuda_new::abi::emit_device_typecheck(kernels_cuda_new::device::ALTUP_AUX) {
        Ok(text) => print!("{text}"),
        Err(why) => {
            eprintln!("emit_device_typecheck: {why}");
            std::process::exit(1);
        }
    }
}
