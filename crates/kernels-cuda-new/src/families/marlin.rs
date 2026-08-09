//! Marlin's W4A16/MXFP4 GEMM, as JIT units.
//!
//! Empty until migrated. It has its own module rather than living in
//! [`super::quant`] because it is a vendored third-party library with its own
//! licence and its own shape list, and because it is the clearest case in the
//! tree for what a JIT removes: the ahead-of-time build turns two `__global__`
//! templates into fourteen generated translation units through a Python
//! script, includes all fourteen into one TU because a macro cannot expand to
//! an `#include`, and reconciles the list against `kernels.def` in thirty-eight
//! lines of CMake. Under a JIT a shape is a name expression, and all of that
//! is deleted rather than ported.

use crate::unit::Unit;

/// The units Marlin compiles. Empty: not yet migrated.
pub static UNITS: &[Unit] = &[];
