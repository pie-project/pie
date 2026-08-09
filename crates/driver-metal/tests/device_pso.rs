//! Who owns a pipeline, and which GPU its binary is valid on.
//!
//! Two claims, both about things the C++ shell needs machinery for and this
//! crate does not.
//!
//! The first is ownership. `RawMetalContext` keeps every pipeline it built in
//! an `NSMutableArray` plus a parallel `unordered_set<void*>`, and exposes
//! `release_pso` and `retained_pso_count` so a caller can give one back and a
//! test can check the two containers stayed in step. All of that exists
//! because the pipeline crosses an FFI boundary as a bare `void*`, which owns
//! nothing. Here it is a `Retained`, so the test to write is not "does the
//! count go down" but "is there a second owner at all" -- and the answer must
//! be no.
//!
//! The second is identity: [`Context::cache_id`] must actually distinguish
//! GPUs, and must be stable within a run, because it is what keeps one
//! device's compiled binaries out of another's cache.

#![allow(clippy::print_stdout)]

use objc2::rc::{Retained, Weak, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputePipelineState, MTLDevice};

use driver_metal::Error;
use driver_metal::gpu::{Compiler, Context, Math};

const TRIVIAL: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void nothing(device uint* out [[buffer(0)]],
                    uint gid [[thread_position_in_grid]]) {
    out[gid] = gid;
}
";

fn context() -> Option<Context> {
    match Context::new() {
        Ok(c) => Some(c),
        Err(Error::NoDevice) => None,
        Err(e) => panic!("context: {e}"),
    }
}

/// The claim `release_pso` exists to provide, asserted directly.
///
/// If anything in this crate ever starts holding pipelines behind the
/// caller's back -- a cache keyed by function name, a "keep them alive for
/// the archive" array, the C++'s `retained` list ported by reflex -- the weak
/// reference survives the drop and this fails. That is the entire point: the
/// C++ needs `release_pso` because something else is holding on, and the way
/// to not need it is to prove nothing is.
#[test]
fn dropping_a_pipeline_releases_it_because_nothing_else_holds_one() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");

    let weak: Weak<ProtocolObject<dyn MTLComputePipelineState>> = autoreleasepool(|_| {
        let pipeline = compiler
            .compile(&context, TRIVIAL, "nothing")
            .expect("nothing");
        let weak = Weak::from_retained(&pipeline);
        assert!(
            weak.load().is_some(),
            "a weak reference to a live pipeline must load"
        );
        drop(pipeline);
        weak
    });

    // The pool is drained. Anything Metal put in it on the way out is gone,
    // so a surviving object is a strong reference someone kept.
    assert!(
        weak.load().is_none(),
        "the pipeline outlived the only Retained the caller was given, so \
         something in this crate is holding a second reference -- which is \
         exactly the situation that makes a release_pso necessary"
    );

    // The compiler is still usable afterwards: releasing a pipeline is not
    // supposed to disturb the thing that built it. The C++ pair does not have
    // this property for free, since `release_pso` mutates two containers the
    // compile path also writes.
    let again = compiler.compile(&context, TRIVIAL, "nothing");
    assert!(
        again.is_ok(),
        "the compiler survived its pipeline: {again:?}"
    );
}

/// Two `Retained`s of one pipeline are two owners, and the object lives until
/// both are gone.
#[test]
fn a_cloned_handle_is_a_second_owner_and_not_a_second_pipeline() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");

    let weak = autoreleasepool(|_| {
        let first: Retained<ProtocolObject<dyn MTLComputePipelineState>> = compiler
            .compile(&context, TRIVIAL, "nothing")
            .expect("nothing");
        let second = first.clone();
        assert!(
            std::ptr::eq(&raw const *first, &raw const *second),
            "a clone must be the same object, not a second compile"
        );

        let weak = Weak::from_retained(&first);
        drop(first);
        assert!(
            weak.load().is_some(),
            "dropping one of two owners released the object, which would make \
             every handed-out pipeline a dangling one"
        );
        drop(second);
        weak
    });
    assert!(weak.load().is_none(), "the last owner did not release it");
}

/// The cache id has to differ when the GPU differs and not otherwise.
///
/// Only half of that is testable on one machine -- there is one GPU here --
/// so what is checked is the half that would silently rot: that it is stable
/// across contexts rather than, say, folding in a pointer or a creation
/// order. An id that changed per context would invalidate the pipeline
/// archive on every process start and the only symptom would be that startup
/// never got faster.
#[test]
fn the_cache_id_names_the_gpu_and_not_the_context() {
    let Some(first) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let second = Context::new().expect("a second context");

    assert_eq!(
        first.cache_id(),
        second.cache_id(),
        "two contexts on one GPU disagreed about the cache id, so every \
         archive written by one is unreachable to the other"
    );
    assert_ne!(
        first.cache_id(),
        0,
        "a zero id is what an unseeded hash returns, and it would collide \
         with every other device that failed the same way"
    );
    assert_ne!(
        first.cache_id(),
        first.device().registryID(),
        "the id is the registry id unmixed, so the device name -- the only \
         part of it that survives a reboot -- is not in the key"
    );
}

/// Precise and fast math are two different sets of binaries, and an archive
/// keyed without the mode would serve one where the other was asked for.
///
/// The wrong answer here is not slow, it is wrong: a transcode kernel
/// compiled with reassociation on produces quantisation codes that are off by
/// a step, and it would arrive looking like a cache hit.
#[test]
fn the_math_mode_is_part_of_what_a_source_compiles_to() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");

    let fast = compiler
        .compile_with(&context, TRIVIAL, "nothing", Math::Fast)
        .expect("fast");
    let precise = compiler
        .compile_with(&context, TRIVIAL, "nothing", Math::Precise)
        .expect("precise");

    assert!(
        !std::ptr::eq(&raw const *fast, &raw const *precise),
        "the two modes returned the same pipeline object, so the mode was \
         not passed to the compiler at all"
    );
    assert_eq!(
        Math::default(),
        Math::Fast,
        "the default must stay fast: precise-by-default would silently slow \
         every kernel that does not need it"
    );
}
