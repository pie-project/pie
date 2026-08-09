//! The measured layouts, asserted against the headers they were measured from.
//!
//! A `__global__` that takes a struct BY VALUE crosses as a byte buffer, and
//! no `Ty` can name it. `by_value!` closes half of that: it asserts the Rust
//! mirror's `size_of`, `align_of` and every `offset_of` against numbers
//! measured out of NVRTC's PTX, in `const` context, so a drifted mirror is a
//! Rust compile error.
//!
//! The other half is this file. The same numbers are emitted as C++
//! `static_assert`s over the HEADER's own declaration, so a drifted header is
//! a C++ compile error. Neither check alone is enough, because the mirror and
//! the header can only drift apart in the direction the other one is watching:
//! `by_value!` would keep passing if `MLAParams` grew a field upstream, and
//! these asserts would keep passing if the Rust mirror were edited to match a
//! header nobody had re-measured.
//!
//! The pairing of a layout to the root that declares its type is written down
//! below rather than derived, because it cannot be derived: a `Layout` says
//! how C++ spells the aggregate and says nothing about which translation unit
//! declares it. Three pairs is not a maintenance burden, and the count is
//! reconciled against the crate's `LAYOUTS` statics so a fourth cannot arrive
//! unnoticed.

#![cfg(feature = "_cuda")]

use kernels_cuda_new::jit::{Root, Toolchain};
use kernels_cuda_new::runtime::nvrtc;
use kernels_cuda_new::x::abi::{TYPECHECK_ENTRY, typecheck_tu};
use kernels_cuda_new::x::{Layout, attn, xqa};

/// One root, and the aggregates its text declares.
fn pairs() -> Vec<(&'static str, &'static Root, &'static [Layout])> {
    vec![
        // `StructuredMaskParams` is `pack_dense_mask.cuh`'s, and this is the
        // only thing in the crate that compiles that root — neither packer has
        // a host program, so it has no instantiation to be asked for.
        ("attn::params", &attn::pack_dense_mask::ROOT, attn::params::LAYOUTS),
        // `::flashinfer::MLAParams<..>` comes in through
        // `attention_mla_fa2.cuh`'s `#include` of `mla_params.cuh`.
        ("attn::mla_params", &attn::mla_fa2::ROOT, attn::mla_params::LAYOUTS),
        // `KVCacheList<true>` is the same in all five lattice members: the
        // `-D` set varies the head group and the page size, and the cache
        // descriptor is a function of neither.
        ("xqa", xqa::ROOTS[0], xqa::LAYOUTS),
    ]
}

#[test]
fn every_measured_layout_matches_its_header() {
    let Ok(have) = nvrtc::version() else {
        eprintln!("SKIPPED: libnvrtc will not load, so nothing here can be compiled");
        return;
    };
    let arch = kernels_cuda_new::jit::cache::arch().unwrap_or("compute_89");
    // `attention_mla_fa2` asks for relocatable device code, so its compile
    // ends in `cuLink` against `libcudadevrt.a` — a driver call, which answers
    // `CUDA_ERROR_INVALID_CONTEXT` on a thread that has not forced the primary
    // context. The other two roots need no device at all.
    if let Err(why) = kernels_cuda_new::jit::cache::bind_context() {
        eprintln!("SKIPPED: no usable context, and one root device-links ({why})");
        return;
    }

    let pairs = pairs();
    let asserted: usize =
        pairs.iter().flat_map(|(.., l)| l.iter()).map(|l| 2 + l.fields.len()).sum();
    eprintln!(
        "nvrtc {have} targeting {arch}: {} aggregates over {} roots, {asserted} assertions",
        pairs.iter().map(|(.., l)| l.len()).sum::<usize>(),
        pairs.len()
    );

    let wanted = [TYPECHECK_ENTRY.to_owned()];
    let mut failed = Vec::new();
    for (what, root, layouts) in &pairs {
        let job = nvrtc::Job {
            name: root.name,
            source: typecheck_tu(root.text, layouts),
            arch,
            options: root.options,
            headers: root.header_set(),
            floor: Toolchain::ANY,
            wanted: &wanted,
            device_link: root.needs_device_runtime(),
        };
        if let Err(why) = nvrtc::compile_text(&job) {
            failed.push(format!("── {what} ({}) ──\n{why}\n", root.name));
        }
    }

    assert!(
        failed.is_empty(),
        "{} of {} typecheck unit(s) would not compile, which means a measured \
         layout no longer describes the header it was measured from:\n\n{}",
        failed.len(),
        pairs.len(),
        failed.join("\n")
    );
}

/// Every `LAYOUTS` static in the crate is compiled by the test above.
///
/// The pairing is hand-written, so the failure mode is a fourth aggregate
/// landing beside a root nobody added here — asserted in Rust and never
/// compiled, which looks exactly like a checked one. Counting the `LAYOUTS`
/// declarations in the source is what makes that loud.
#[test]
fn no_measured_layout_goes_unasserted() {
    let mut declared = 0usize;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/x");
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("a readable directory") {
            let path = entry.expect("a readable entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let text = std::fs::read_to_string(&path).expect("a readable source file");
                declared += text.matches("static LAYOUTS").count();
            }
        }
    }
    assert_eq!(
        declared,
        pairs().len(),
        "the crate declares {declared} `LAYOUTS` statics and this file pairs \
         {} of them with a root, so the difference is asserted in Rust and \
         compiled by nothing",
        pairs().len()
    );
}
