//! Compiling from several threads at once.
//!
//! This test asserts nothing. It is here because without the gate in
//! `pipeline.rs` it aborts -- not with a failed assertion but with a trap
//! inside `libsystem_malloc`, because concurrent pipeline creation on this
//! driver corrupts the process heap.
//!
//! The churn after the compile is the load-bearing half. Corruption during
//! the compile is invisible to a process that then exits; it surfaces the
//! next time something walks the damaged freelist. Removing the churn takes
//! the failure rate from three runs in six to none, while changing nothing
//! about the damage.

use driver_metal::{Compiler, Context};

const THREADS: usize = 8;

/// Distinct sources, so no thread can be served a cached pipeline.
fn source(thread: usize) -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void fill{thread}(device uint* out [[buffer(0)]], \
         uint gid [[thread_position_in_grid]]) {{ out[gid] = gid + {thread}u; }}\n"
    )
}

/// Enough small allocations to reach a freelist the compile may have damaged.
fn churn() {
    let mut keep: Vec<String> = Vec::new();
    for i in 0..200_000u32 {
        keep.push(format!("{i}"));
        if keep.len() > 64 {
            keep.clear();
        }
    }
}

#[test]
fn compiling_from_several_threads_does_not_corrupt_the_heap() {
    let threads: Vec<_> = (0..THREADS)
        .map(|thread| {
            std::thread::spawn(move || {
                let context = Context::new().expect("context");
                let compiler = Compiler::new(&context).expect("compiler");
                compiler
                    .compile(&context, &source(thread), &format!("fill{thread}"))
                    .expect("compile");
                churn();
            })
        })
        .collect();
    for thread in threads {
        thread.join().expect("thread");
    }
}
