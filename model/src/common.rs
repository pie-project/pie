//! What every family shares: vocabulary and building blocks, never knowledge
//! about any one family.
//!
//! The test for whether something belongs here: it must be *about models in
//! general* — a decoder that classifies a token stream, a policy type a
//! contract author is parameterized by — and never a fact about one family's
//! checkpoints or templates. Those facts live in the family's own directory.

pub mod decoders;

// The contract aspect's vocabulary and toolkit. Behind the feature so
// chat-only consumers never pull pie-loader.
#[cfg(feature = "contract")]
pub mod builder;
#[cfg(feature = "contract")]
pub mod facts;
#[cfg(feature = "contract")]
pub mod moe;
#[cfg(feature = "contract")]
pub mod policy;
#[cfg(feature = "contract")]
pub mod probe;
