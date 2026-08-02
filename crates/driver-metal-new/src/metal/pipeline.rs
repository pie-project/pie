//! Turning `.metal` text into a compute pipeline state.
//!
//! Runtime compilation, not a prebuilt `.metallib`. The box this driver was
//! developed on has CommandLineTools and no Xcode, so there is no offline
//! `metal` compiler to produce one -- and the AOT path in the C++ shell's
//! CMake is gated off by default for exactly that reason. Handing
//! `newLibraryWithSource:` a string is the path that is always available.
//!
//! # The language version is a property of the driver
//!
//! `MTLCompileOptions` defaults to an older MSL standard. Under that default
//! `<metal_tensor>` and the MetalPerformancePrimitives tensor ops are simply
//! not visible, so a kernel that uses them fails to compile with an error
//! about an unknown identifier rather than about a dialect. The C++ shell
//! learned to pin `MTLLanguageVersion4_0` in ONE place after setting it at
//! some call sites and not others; [`Compiler::compile`] is that place here,
//! and there is no way to ask for a pipeline that skips it.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4Compiler, MTL4CompilerDescriptor, MTL4ComputePipelineDescriptor,
    MTL4LibraryFunctionDescriptor, MTLCompileOptions, MTLComputePipelineState, MTLDevice,
    MTLLanguageVersion, MTLLibrary,
};

use super::context::{Context, describe};
use crate::error::{Error, Result};

/// The MSL dialect every kernel in this driver is compiled as.
///
/// See the module docs: this is not a default, and a kernel compiled without
/// it fails in a way that does not mention the dialect.
const LANGUAGE_VERSION: MTLLanguageVersion = MTLLanguageVersion::Version4_0;

/// The runtime shader compiler.
///
/// One per context. Kept out of [`Context`] because it is not needed to
/// encode a step -- a driver that loaded prebuilt pipelines would have a
/// context and no compiler -- and because a type that owns a device object
/// and also compiles text is two types.
pub struct Compiler {
    compiler: Retained<ProtocolObject<dyn MTL4Compiler>>,
}

impl Compiler {
    /// Create the compiler for `context`'s device.
    pub fn new(context: &Context) -> Result<Self> {
        let descriptor = MTL4CompilerDescriptor::new();
        let compiler = context
            .device()
            .newCompilerWithDescriptor_error(&descriptor)
            .map_err(|e| Error::Create {
                what: "MTL4Compiler",
                message: describe(&e),
            })?;
        Ok(Self { compiler })
    }

    /// Compile `source` and build the pipeline for its `function` entry point.
    ///
    /// Three failures, kept apart because they have three different remedies:
    /// the source did not compile, it compiled but exports no such entry
    /// point, or the pipeline itself was rejected. The middle one is the one
    /// worth separating -- a misspelled entry point otherwise arrives as
    /// Metal's own message about a nil function, which names neither the
    /// spelling that was asked for nor the ones that exist.
    pub fn compile(
        &self,
        context: &Context,
        source: &str,
        function: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(LANGUAGE_VERSION);

        let library = context
            .device()
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
            .map_err(|e| Error::Compile {
                function: function.to_string(),
                message: describe(&e),
            })?;

        // Asked before the descriptor is built, so the message can list what
        // the library DOES export. After the pipeline call it is too late:
        // the library is still in hand but the error is already Metal's.
        let exported: Vec<String> = library
            .functionNames()
            .iter()
            .map(|name| name.to_string())
            .collect();
        if !exported.iter().any(|name| name == function) {
            return Err(Error::Compile {
                function: function.to_string(),
                message: format!(
                    "the source compiled but exports no such entry point; it exports [{}]",
                    exported.join(", ")
                ),
            });
        }

        let name = NSString::from_str(function);
        let function_descriptor = MTL4LibraryFunctionDescriptor::new();
        function_descriptor.setName(Some(&name));
        function_descriptor.setLibrary(Some(&library));

        let pipeline_descriptor = MTL4ComputePipelineDescriptor::new();
        pipeline_descriptor.setComputeFunctionDescriptor(Some(&function_descriptor));
        // The entry point's name, carried on the pipeline. Per-dispatch
        // tracing has nothing else to report: a pipeline is an opaque object
        // and the DAG ordinal names a position rather than a kernel.
        pipeline_descriptor.setLabel(Some(&name));

        self.compiler
            .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(
                &pipeline_descriptor,
                None,
            )
            .map_err(|e| Error::Compile {
                function: function.to_string(),
                message: describe(&e),
            })
    }
}

impl std::fmt::Debug for Compiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compiler").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIVIAL: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void fill_ones(device float* out [[buffer(0)]],
                      uint gid [[thread_position_in_grid]]) {
    out[gid] = 1.0f;
}
";

    fn compiler() -> Option<(Context, Compiler)> {
        let context = match Context::new() {
            Ok(c) => c,
            Err(Error::NoDevice) => return None,
            Err(e) => panic!("context: {e}"),
        };
        let compiler = Compiler::new(&context).expect("compiler");
        Some((context, compiler))
    }

    #[test]
    fn a_trivial_kernel_compiles() {
        let Some((context, compiler)) = compiler() else {
            return;
        };
        let pso = compiler
            .compile(&context, TRIVIAL, "fill_ones")
            .expect("compiles");
        assert!(
            pso.maxTotalThreadsPerThreadgroup() > 0,
            "a real pipeline reports a threadgroup limit"
        );
    }

    /// The dialect is pinned by the driver, so a source that names an MSL 4.0
    /// header compiles without the caller asking for anything. Under the
    /// default standard this fails at the include.
    #[test]
    fn the_pinned_dialect_reaches_the_msl_4_headers() {
        let Some((context, compiler)) = compiler() else {
            return;
        };
        let source = r"
#include <metal_stdlib>
#include <metal_tensor>
using namespace metal;
kernel void touch(device float* out [[buffer(0)]],
                  uint gid [[thread_position_in_grid]]) {
    out[gid] = 0.0f;
}
";
        compiler
            .compile(&context, source, "touch")
            .expect("<metal_tensor> is visible only under MSL 4.0");
    }

    #[test]
    fn a_syntax_error_names_the_function_and_says_what_metal_said() {
        let Some((context, compiler)) = compiler() else {
            return;
        };
        let err = compiler
            .compile(&context, "kernel void broken( {", "broken")
            .expect_err("that is not MSL");
        match err {
            Error::Compile { function, message } => {
                assert_eq!(function, "broken");
                assert!(!message.is_empty(), "Metal's diagnostic is not dropped");
            }
            other => panic!("expected Compile, got {other}"),
        }
    }

    /// The failure this variant exists for: the source is fine and the name is
    /// not. Metal reports it as a nil function, naming neither the spelling
    /// asked for nor the ones that exist.
    #[test]
    fn a_missing_entry_point_lists_the_ones_that_exist() {
        let Some((context, compiler)) = compiler() else {
            return;
        };
        let err = compiler
            .compile(&context, TRIVIAL, "fill_zeroes")
            .expect_err("no such entry point");
        match err {
            Error::Compile { function, message } => {
                assert_eq!(function, "fill_zeroes");
                assert!(
                    message.contains("fill_ones"),
                    "the message must list what does exist: {message}"
                );
            }
            other => panic!("expected Compile, got {other}"),
        }
    }

    /// A library exports its KERNELS, not its functions. Asking for a plain
    /// helper is therefore the missing-entry-point path rather than a
    /// pipeline failure, and it is answered before Metal is asked.
    #[test]
    fn a_plain_function_is_not_an_entry_point() {
        let Some((context, compiler)) = compiler() else {
            return;
        };
        let source = r"
#include <metal_stdlib>
using namespace metal;
float helper(float x) { return x * 2.0f; }
kernel void real(device float* out [[buffer(0)]],
                 uint gid [[thread_position_in_grid]]) {
    out[gid] = helper(1.0f);
}
";
        let err = compiler
            .compile(&context, source, "helper")
            .expect_err("a helper is not an entry point");
        match err {
            Error::Compile { message, .. } => assert!(
                message.contains("real"),
                "the message lists the kernel that IS exported: {message}"
            ),
            other => panic!("expected Compile, got {other}"),
        }
    }
}
