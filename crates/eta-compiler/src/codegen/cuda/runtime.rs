use alloc::string::String;

const PROLOGUE: &str = include_str!("../../../runtime/cuda/ptir_m1_runtime_prologue.cuh");
const BODY: &str = include_str!("../../../runtime/cuda/ptir_m1_runtime_body.cuh");
const CUDA_CHUNK: &str = include_str!("../../../runtime/cuda/ptir_m1_chunk.cuh");
const HIP_CHUNK: &str = include_str!("../../../runtime/hip/ptir_m1_chunk.hip");

/// Which compiler the emitted source is for. The runtime and every emitted
/// kernel are one text; the dialect picks how the 32-lane chunk the
/// reductions are written over is held (`ptir_m1_chunk`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dialect {
    /// NVRTC: a CUDA warp is the chunk.
    Cuda,
    /// hiprtc: two chunks per 64-lane wavefront.
    Hip,
}

fn rng_preamble() -> String {
    let mut preamble = String::from("\n");
    preamble.push_str(&crate::codegen::rng::cuda_device_functions());
    preamble
}

pub fn singleton_runtime_source() -> String {
    runtime_source(Dialect::Cuda)
}

pub fn runtime_source(dialect: Dialect) -> String {
    let mut source = String::with_capacity(PROLOGUE.len() + BODY.len() + 4096);
    source.push_str(PROLOGUE);
    source.push_str(match dialect {
        Dialect::Cuda => CUDA_CHUNK,
        Dialect::Hip => HIP_CHUNK,
    });
    source.push_str(&rng_preamble());
    source.push_str(BODY);
    source
}
