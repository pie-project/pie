//! The shipped kernels, through the shell that will run them.
//!
//! [`real_kernels`] proves the splicer produces the right TEXT. This proves
//! Metal accepts it: every `.metal` under `kernels-metal/kernels/` is spliced
//! and handed to the runtime compiler, under the MSL dialect this driver pins.
//!
//! That pairing is the point. A splice that drops a header still produces
//! plausible text, and the assertion that no quoted include survived would
//! still hold -- it is the compiler that notices the definitions are gone.
//!
//! And then a second test goes one step further than a library, because a
//! library is not a launch: it builds a compute PIPELINE for every entrypoint
//! the kernel table declares, which is where Metal enforces the threadgroup
//! memory budget and the thread count. A shader can compile at a width the
//! device will not run.
//!
//! Requires a Metal 4 GPU, so it skips rather than fails when there is none.
//! The rest of the workspace is developed on Linux and this file has to be a
//! no-op there.
//!
//! [`real_kernels`]: ../real_kernels/index.html

use std::path::PathBuf;

use driver_metal::Error;
// The pipeline's own ceiling, which is the one fact about a built pipeline
// that no `driver-metal` type re-exports: `Encoder` reads it to refuse an
// over-wide dispatch and keeps it private.
use driver_metal::device::Context;
use driver_metal::program::Compiler;
use objc2_metal::MTLComputePipelineState;

/// An entry point no kernel defines.
///
/// The compile is driven to the point of a LIBRARY and no further. Building a
/// pipeline needs an entry point name, and the names differ per file -- so
/// asking for one that cannot exist makes "the source compiled" the answer
/// this test can assert for all of them at once. The missing-entry-point
/// error is reported only after the library was built, which is exactly the
/// fact under test.
const NO_SUCH_ENTRY: &str = "__pie_no_such_entry_point";

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/driver-metal has a parent")
        .join("kernels-metal/kernels")
}

#[test]
fn every_shipped_kernel_compiles_on_this_device() {
    let context = match Context::new() {
        Ok(c) => c,
        Err(Error::NoDevice) => {
            driver_metal::skip::skipped("no Metal 4 device, so no shipped kernel was compiled");
            return;
        }
        Err(e) => panic!("context: {e}"),
    };
    let compiler = Compiler::new(&context).expect("compiler");

    // Recursive: the kernels grew subject directories, and the flat scan
    // this replaces was red for "no .metal found" while thirty shaders sat
    // one level down. `third_party/` is excluded from STANDALONE compiling —
    // the MLX steel fragments there are not translation units (they expect
    // their includer's macros); they are compiled through `quant/qmm_t.metal`,
    // which is in the walk, and their text is still covered by
    // `real_kernels::every_shipped_shader_splices`.
    fn walk(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("kernels dir is readable") {
            let path = entry.expect("kernels dir entry").path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n == "third_party") {
                    continue;
                }
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "metal") {
                out.push(path);
            }
        }
    }
    let mut files: Vec<PathBuf> = Vec::new();
    walk(&kernels_dir(), &mut files);
    files.sort();
    assert!(!files.is_empty(), "no .metal found to compile");

    let mut rejected = Vec::new();
    for path in &files {
        let source = driver_metal::layout::shader::read_source(path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        match compiler.compile(&context, &source, NO_SUCH_ENTRY) {
            // The library built and the entry point does not exist, which is
            // the only outcome that means "this source compiles".
            Err(Error::Compile { message, .. }) if message.contains("exports no such") => {}
            Err(e) => rejected.push(format!("{}: {e}", path.display())),
            Ok(_) => panic!("{}: defines {NO_SUCH_ENTRY}", path.display()),
        }
    }

    assert!(
        rejected.is_empty(),
        "{} of {} shipped kernels do not compile:\n{}",
        rejected.len(),
        files.len(),
        rejected.join("\n")
    );
}

/// **Compiling is not launching, and only this test knows the difference.**
///
/// The test above drives every shader to a LIBRARY and stops, because it asks
/// one question of all of them at once and a library is where that question is
/// answered. This one goes the rest of the way: for every entrypoint the
/// kernel table declares, it builds a COMPUTE PIPELINE STATE, which is where
/// Metal enforces what a library never checks -- the 32 KB of threadgroup
/// memory a threadgroup may declare, and the thread count an entrypoint's
/// `max_total_threads_per_threadgroup` will accept.
///
/// The gap is not hypothetical. `sdpa_paged_mma.metal` carried, for as long as
/// it existed, an argument that a 128-wide head "would be 40 KB, over the 32
/// KB a threadgroup gets" and therefore could not be instantiated. The
/// arithmetic was right for `KT=64` and the file instantiates `KT=16`, where
/// the same three tiles are 16 KB. Nothing contradicted it because nothing
/// asked the device: the shader compiled at every width, and no width but 64
/// was ever built into a pipeline. It is a pipeline now, and so is every other
/// entrypoint the families name.
///
/// It also holds the other direction. A pair that states a file it does not
/// own -- a symbol the shader does not define -- fails here with "exports no
/// such" and passes everywhere else, because the path in `ENTRYPOINTS` is a
/// string nothing else dereferences until a load reaches it.
///
/// And it reads the ceiling back off each pipeline. Building is permission to
/// dispatch at SOME width, not at the width the row's rule will ask for, and
/// Metal does not refuse an over-wide threadgroup -- it skips the dispatch,
/// leaves the output holding whatever it held, and reports success.
/// `device::encoder::dispatch` catches that at serve time; this catches it at
/// build time, for the three rules that fix a width from the row alone.
///
/// 180 of the 481 pipelines admit fewer than 1024 threads on this Mac, down to
/// 576 for the quantised GEMMs, so the ceiling is not a formality. None of
/// them is below what its own rule dispatches -- `qmv_mb` asks for 64 and
/// `qmm_t` for 128 -- which is why the assertion is against the rule and not
/// against a pinned census that would say more about this device than about
/// this tree.
#[test]
#[ignore = "needs a Metal 4 device"]
fn every_declared_entrypoint_builds_a_pipeline_on_this_device() {
    let context = match Context::new() {
        Ok(c) => c,
        Err(Error::NoDevice) => {
            driver_metal::skip::skipped("no Metal 4 device, so no shipped kernel was compiled");
            return;
        }
        Err(e) => panic!("context: {e}"),
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let root = kernels_dir();

    let mut sources: std::collections::HashMap<&'static str, String> = Default::default();
    let mut refused = Vec::new();
    let mut narrow: Vec<String> = Vec::new();
    let mut built = 0usize;

    // The census, not the table. Every family has retired its `kernel!` rows,
    // so `KERNELS` is empty and a sweep keyed on it would build NOTHING and
    // say so with a passing test. `kernels_metal::shaders()` is the same set
    // of `(file, entrypoint)` pairs the rows used to generate, stated by the
    // families instead -- and there is no `fileless` case left to collect,
    // because a pair without a file cannot be written down.
    for (file, entry) in kernels_metal::shaders() {
        let source = sources.entry(file).or_insert_with(|| {
            driver_metal::layout::shader::read_source(root.join(file))
                .unwrap_or_else(|e| panic!("{}: {e}", root.join(file).display()))
        });
        built += 1;
        match compiler.compile(&context, source, entry) {
            Err(e) => refused.push(format!("  {entry} [{file}]: {e}")),
            Ok(pso) => {
                // What the BODY will ask for. Most routines derive a
                // threadgroup from facts this test has none of; the
                // attention families fix theirs outright, and those are
                // what a shader alone can be checked against.
                //
                // The two numbers are the ones `kernels-metal`'s `attn`
                // fires spell at each `Fire::at`: 128 for the matrix-unit
                // tiling, whose shader declares
                // `max_total_threads_per_threadgroup(128)`, and 1024 for
                // every scalar single-pass and tiled form.
                // BY STEM, off the entry point's own spelling.
                // `lowering::routine::crossed` STOOD HERE and answered with
                // the `#[routine]` ROW that declared this entry point, whose
                // `name` was the stem; there are no rows, so the stem is read
                // where it has always been written — at the front of the
                // entry point, which `kernels_metal::shaders()` hands over.
                //
                // The sink forms come FIRST because `sdpa_paged_mma` is a
                // prefix of `sdpa_paged_mma_sink` and a first match on the
                // shorter would answer for both. Here the two want the same
                // width, so the ordering is a guard against the next pair
                // rather than a live fix.
                let Some(stem) = [
                    "sdpa_paged_mma_sink",
                    "sdpa_paged_mma",
                    "sdpa_paged_decode_sink",
                    "sdpa_paged_decode",
                    "sdpa_paged_tiled_sink",
                    "sdpa_paged_tiled",
                    "sdpa_vector_decode_swa",
                    "sdpa_vector_decode",
                ]
                .into_iter()
                .find(|stem| entry.starts_with(stem)) else {
                    continue;
                };
                let wants: u32 = match stem {
                    "sdpa_paged_mma" | "sdpa_paged_mma_sink" => 128,
                    _ => 1024,
                };
                let admits = pso.maxTotalThreadsPerThreadgroup() as u32;
                if admits < wants {
                    narrow.push(format!(
                        "  {entry} [{file}]: admits {admits}, `{stem}` dispatches {wants}"
                    ));
                }
            }
        }
    }

    assert!(
        built > 400,
        "only {built} entrypoints were built, so the census shrank and this \
         test compared almost nothing"
    );
    assert!(
        refused.is_empty(),
        "{} of {built} declared entrypoints compile but the device refuses \
         to make a pipeline for them:\n{}",
        refused.len(),
        refused.join("\n")
    );

    narrow.sort();
    assert!(
        narrow.is_empty(),
        "{} entrypoint(s) build a pipeline the device will not let their own launch rule \
         dispatch:\n{}",
        narrow.len(),
        narrow.join("\n")
    );
}
