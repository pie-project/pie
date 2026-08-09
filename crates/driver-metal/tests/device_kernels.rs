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
//! Requires a Metal 4 GPU, so it skips rather than fails when there is none.
//! The rest of the workspace is developed on Linux and this file has to be a
//! no-op there.
//!
//! [`real_kernels`]: ../real_kernels/index.html

use std::path::PathBuf;

use driver_metal::Error;
use driver_metal::device::Context;
use driver_metal::program::Compiler;

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
        Err(Error::NoDevice) => return,
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
