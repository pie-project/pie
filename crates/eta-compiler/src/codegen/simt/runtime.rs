use alloc::string::String;

const PROLOGUE: &str = include_str!("../../../runtime/cuda/ptir_m1_runtime_prologue.cuh");
const BODY: &str = include_str!("../../../runtime/cuda/ptir_m1_runtime_body.cuh");

/// What the emitted source needs of the compiler it is for. The runtime and
/// every emitted kernel are one text; a dialect gives how the 32-lane chunk
/// the reductions are written over is held (`ptir_m1_chunk`), and how a
/// lane sampler's blocks meet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dialect {
    pub chunk: &'static str,
    pub lane: &'static str,
    /// The line a region carries where it holds the exact cut's state, so
    /// the launcher sizes its dynamic shared memory; empty where the dialect
    /// keeps that state static.
    pub cut_mark: &'static str,
}

fn rng_preamble() -> String {
    let mut preamble = String::from("\n");
    preamble.push_str(&crate::codegen::rng::cuda_device_functions());
    preamble
}

pub fn runtime_source(dialect: Dialect) -> String {
    let mut source = String::with_capacity(PROLOGUE.len() + BODY.len() + 4096);
    source.push_str(PROLOGUE);
    source.push_str(dialect.chunk);
    source.push_str(&rng_preamble());
    source.push_str(BODY);
    source
}
