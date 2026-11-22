//! The choice between the kernels that can serve a decode product, by their
//! times measured in place (`[rates] product_ns`): the shared table
//! ([`profile::routes`]) with this backend's candidates, the skinny skeleton
//! and cuBLASLt's first algorithms ([`crate::linear::gemm::candidates`]).
//!
//! A candidate's code is its key in the profile: a hash of the device's
//! arch, its driver and cuBLASLt, the kernel's own code (a JIT kernel's
//! source, headers, options and instantiation; a cuBLASLt algorithm's
//! config) and the measuring method's version, never the pie version, so a
//! profile that holds a shape's candidates measures nothing again whatever
//! build reads it.

pub use profile::routes::{Kind, Routes, Shape, Wanted, product_times};

/// The measuring method's version: a change to how a product is timed
/// measures every shape again (it is folded into each kernel's code).
pub const METHOD: u32 = 1;

/// The code a kernel whose own code fingerprints as `fingerprint` is kept
/// under on `platform` (the device's arch, driver and cuBLASLt).
#[must_use]
pub fn code(platform: &str, fingerprint: &str) -> u32 {
    let hash = crate::source::fnv1a64(format!("{platform}|{fingerprint}|method {METHOD}").as_bytes());
    (hash ^ (hash >> 32)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_code_is_the_platform_s_the_kernel_s_and_the_method_s_alone() {
        let platform = "sm_89|driver 13000|cublaslt 130000";
        let skinny = code(platform, "jit/skinny/r1");
        assert_eq!(skinny, code(platform, "jit/skinny/r1"), "the same kernel, the same key");
        assert_ne!(skinny, code(platform, "jit/skinny/r2"), "changed code, another key");
        assert_ne!(skinny, code("sm_89|driver 13010|cublaslt 130000", "jit/skinny/r1"));
    }
}
