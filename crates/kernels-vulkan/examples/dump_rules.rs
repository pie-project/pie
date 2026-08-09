//! Every entrypoint and the launch rule its row names, one per line.
//!
//! `driver-vulkan`'s geometry has to agree with the `local_size` baked into
//! each module, and a rule is what connects the two. Printing them is how that
//! agreement gets checked over the whole table rather than by reading.
fn main() {
    for name in kernels_vulkan::entrypoints() {
        let row = kernels::sig_in(kernels_vulkan::KERNELS, &name).expect("resolves");
        println!("{name}\t{:?}", row.launch);
    }
}
