//! Emit the Tier A device typecheck TU on stdout.
//!
//! The generated file is what the build compiles: it includes the entry
//! points and checks each one against the row that names it. An example
//! rather than a build-script step because Tier A is a pilot -- the CMake
//! that would emit it as part of `native` is the change this measurement is
//! meant to justify, not one to make ahead of it.
//!
//! ```text
//! cargo run -p kernels-cuda --example emit_device_typecheck > device_typecheck.cu
//! nvcc -std=c++20 -arch=sm_89 -fatbin -I crates/kernels-cuda/csrc/src \
//!      device_typecheck.cu -o tier_a.fatbin
//! ```

fn main() {
    match kernels_cuda::abi::emit_device_typecheck(&[kernels_cuda::norm_device::ENTRIES]) {
        Ok(text) => print!("{text}"),
        Err(why) => {
            eprintln!("emit_device_typecheck: {why}");
            std::process::exit(1);
        }
    }
}
