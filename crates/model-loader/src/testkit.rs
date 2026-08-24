//! The part of this crate that exists to *check* the loader rather than to be
//! the loader.
//!
//! One module, one feature gate, one reason: [`mod@reference`] — the
//! differential oracle plans are replayed against — is not on the load path,
//! so a driver that links this crate has no use for it. `worker/Cargo.toml`
//! turns `testkit` off, which is also the check — if the load path ever grows
//! a route into the oracle, the worker stops compiling.
//!
//! Three things used to sit beside it and left for three reasons. The host
//! executor was *promoted*: `pie model import` made it a production caller, so
//! it lives ungated at [`crate::executor`]. The POD contract fixture builder
//! followed the ABI out of the workspace and then out of existence: contracts
//! stopped crossing the ABI when the last C++ author was harvested
//! (`plan/model-in-rust.md` §8-5), so there is no POD graph left to write. And
//! `write_safetensors_fixture` -- a header assembler for tests wanting a real
//! file on disk -- was simply never called, by anything, including the tests it
//! was written for; the ones that need a file on disk build their own header
//! inline. It went with the `pie_loader_open_checkpoint` its doc named, which
//! is a C entry point this tree no longer has.

pub mod reference;
