//! The shared execution-shell substrate — C++ today, Rust eventually.
//!
//! `include/` holds the host C++ vocabulary both shells interpret the direct
//! ABI with: the launch plan, the op table, the program, fire geometry, and
//! `validate`. It sits on `pie_driver_abi.h` and is compiled by whichever
//! shell includes it, never here.
//!
//! There is no Rust in this crate yet, and the bare name is granted on
//! trajectory rather than on today's contents: this is where the driver
//! foundation abstraction grows, and the two shells' shared shape is already
//! here in C++ form. Until then the crate exists to do one job cargo is good
//! at and relative paths are bad at — say where the headers are (see
//! `build.rs`).
