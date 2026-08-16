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
//! declares it. Four pairs is not a maintenance burden, and the count is
//! reconciled against the crate's `LAYOUTS` statics so a fifth cannot arrive
//! unnoticed.

#![cfg(feature = "_cuda")]

use kernels_cuda::jit::{Root, Toolchain};
use kernels_cuda::jit::nvrtc;
use kernels_cuda::jit::abi::{TYPECHECK_ENTRY, typecheck_tu};
use kernels_cuda::attn::{self, xqa};
use kernels_cuda::quant;
use kernels_cuda::jit::Layout;

/// One root, and the aggregates its text declares.
fn pairs() -> Vec<(&'static str, Root, &'static [Layout])> {
    vec![
        // `StructuredMaskParams` is `pack_dense_mask.cuh`'s, and this is the
        // only thing in the crate that compiles that root — neither packer has
        // a host program, so it has no instantiation to be asked for.
        ("attn::params", Root::new("attn/pack_dense_mask.cuh"), attn::params::LAYOUTS),
        // `::flashinfer::MLAParams<..>` comes in through
        // `attention_mla_fa2.cuh`'s `#include` of `mla_params.cuh`.
        ("attn::mla_params", Root::new("attn/attention_mla_fa2.cuh"), attn::mla_params::LAYOUTS),
        // `KVCacheList<true>` is the same in all five lattice members: the
        // `-D` set varies the head group and the page size, and the cache
        // descriptor is a function of neither.
        ("xqa", *xqa::ROOTS[0], xqa::LAYOUTS),
        // The fused transcode's three functors, which the kernel takes BY
        // VALUE -- two Decodes and one Encode, `quant/transcode.cuh:131`,
        // `:143` and `:163`. The pairing is the same shape as the three
        // above and the reason is sharper: these mirrors were written from a
        // measurement of a header nothing in this crate had ever compiled.
        ("quant::transcode", Root::new("quant/transcode.cuh"), quant::transcode::LAYOUTS),
    ]
}

/// Run `f`, answering `None` if it panics, without printing a crash report.
///
/// `cudarc`'s loader is `fallback-dynamic-loading` and nothing here has a
/// `DT_NEEDED` on `libnvrtc` or `libcuda`: the first call `dlopen`s the
/// library and PANICS through `cudarc::panic_no_lib_found` if no candidate
/// name resolves. So none of the `Result`s below can report a library that is
/// simply not installed, and the three skips they guard were unreachable on a
/// box with no CUDA -- which FAILED this test instead. Catching is what makes
/// them reachable. `every_instantiation_compiles.rs` carries the same helper
/// for the same reason.
fn quietly<R>(f: impl FnOnce() -> R + std::panic::UnwindSafe) -> Option<R> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(f);
    std::panic::set_hook(hook);
    out.ok()
}

#[test]
fn every_measured_layout_matches_its_header() {
    let Some(have) = quietly(nvrtc::version).and_then(Result::ok) else {
        eprintln!("SKIPPED: libnvrtc will not load, so nothing here can be compiled");
        return;
    };
    // `sm_XY` and not `compute_XY`: `jit::nvrtc::options` refuses a virtual
    // architecture, so the fallback that used to read `compute_89` failed
    // every root on the arch string alone whenever no device answered.
    let arch = quietly(kernels_cuda::jit::cache::arch).flatten().unwrap_or("sm_89");
    // `attention_mla_fa2` asks for relocatable device code, so its compile
    // ends in `cuLink` against `libcudadevrt.a` — a driver call, which answers
    // `CUDA_ERROR_INVALID_CONTEXT` on a thread that has not forced the primary
    // context. The other three roots need no device at all.
    //
    // Which is why this is a PARTITION and not a return. It used to be a
    // return, and that cost the whole test: on the CI runner this step was
    // built for -- NVRTC from a wheel, no driver -- all four roots were
    // dropped because one of them links, so the step ran in 0.01 s and
    // asserted nothing while reading green. Its sibling
    // `every_instantiation_compiles` had the same shape and was split for the
    // same reason; this one was missed.
    let linkable = matches!(quietly(kernels_cuda::jit::cache::bind_context), Some(Ok(())));

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
    let mut deferred = Vec::new();
    for (what, root, layouts) in &pairs {
        if root.needs_device_runtime() && !linkable {
            deferred.push(format!("{what} ({})", root.name));
            continue;
        }
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

    // Named rather than counted, and after the compiles rather than instead of
    // them: a run that skipped one root and a run that skipped every root must
    // not print the same thing.
    if !deferred.is_empty() {
        eprintln!(
            "SKIPPED {} of {} root(s), which device-link and so need a context this \
             machine has no device to give: {}",
            deferred.len(),
            pairs.len(),
            deferred.join(", ")
        );
    }
    assert!(
        deferred.len() < pairs.len(),
        "every root here device-links, so this run compiled NOTHING"
    );

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
/// The pairing is hand-written, so the failure mode is another aggregate
/// landing beside a root nobody added here — asserted in Rust and never
/// compiled, which looks exactly like a checked one. Counting the `LAYOUTS`
/// declarations in the source is what makes that loud.
#[test]
fn no_measured_layout_goes_unasserted() {
    let mut declared = 0usize;
    // The whole of `src/`, where it used to be `src/x`. A `LAYOUTS` static
    // belongs to whichever family declares a by-value aggregate, and since the
    // dissolution a family is a top-level module rather than a child of one —
    // so scoping this to a subdirectory would be a list of which families are
    // allowed to have one.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
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
