//! Inferlet SDK for Pie
//!
//! This crate provides the core types and traits for building inferlets
//! that run on the Pie inference engine.

// Re-export the macro
pub use inferlet_macros::inferlet;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::inferlet;
}
