//! The driver's remaining model knowledge: [`rope`]'s inverse-frequency
//! tables.
//!
//! # NOTHING ON A SERVING PATH CALLS IT, and that is a statement rather than
//! # an omission
//!
//! `serve::load` derived the ladder here at load, from
//! `Deployment::rope_theta`. A `Deployment` states no such field: a rotation
//! is a STATEMENT of the trace (`kernels::rope::{full,partial,partial_q,
//! partial_last}`) and its base rides on that statement, so the plane raises
//! the ladder itself and a rescaled one is a `Const` bank the text names.
//! `baker::stage`'s own note says the same thing from the other side --
//! `FireTable::RopeFrequencies` left the fire tables for exactly this reason.
//!
//! What is kept is the ARITHMETIC, with its five tests, because llama-3's
//! piecewise rescaling and YaRN's ramp are what a text cannot express in a
//! base and something will have to compute them when a rescaled family
//! reaches this plane. Deleting it would delete the measurement with it.
//!
//! `binding` STOOD BESIDE IT and was the legacy load contract's other half:
//! what a Metal load OBSERVED (an affine group and bit width, a router point,
//! an mxfp4 bank) carried as a `MetalBinding`, plus the one door onto a
//! catalog row's Metal text (`row.trace(class, Deployed::metal(&binding))`).
//! Both ends are gone. A plane is chosen by naming
//! `model_ir::kernels::Backend::Metal` when the row is traced, and what a
//! statement rides is the plan's own `repr` column, read at the bank slot by
//! `baker::bound::Bound::form` — so there is nothing left for a driver to
//! observe and hand back.

pub mod rope;
