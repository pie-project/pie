//! What [`crate::layout`] planned, allocated.
//!
//! `layout::kv::Shape` is arithmetic over nine `u32`s and is correct without
//! a GPU; the pool that shape describes is memory and is not. This is the
//! second half, and the split is rule 2 of `.wiki/driver/north-star.md`
//! drawn on exactly that question.

pub mod kv;
