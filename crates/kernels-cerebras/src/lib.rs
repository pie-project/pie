//! Cerebras kernels: the wafer program's compiler. [`spatial`] plans a model's trace onto the
//! wafer, builds the program (floor, links, kernels, weight bindings of a whole-wafer program,
//! `programs/wafer`), links it with the SDK's toolchain and cuts the checkpoint into the tiles it
//! places, written as an [`spatial::artifact::Artifact`] the engine loads and runs.

pub mod spatial;
