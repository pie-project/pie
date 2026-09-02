//! Backend lowering rules.
//!
//! These test the *decisions*, not the plumbing: each case pins one branch of
//! the CUDA table so a later change to it has to be deliberate. The end-to-end
//! wiring (that `lower` reaches every `TileMap` and writes the field back) is
//! covered by the plan-level tests in `tests/storage_compiler.rs`.
//!
//! **THIS FILE WAS THE ENCODE LOWERING'S SUITE** and §M-3 shut that door.
//! Fifteen cases here budgeted an encode's scratch rows, fused an FP8 source
//! into MXFP4, or read a fact only `encode_rows_per_tile` consumed; every one
//! of them tested a decision no plan can now reach, because no device mask
//! carries an encode and the transform runs on the host at `pie model
//! import`. What is left is the table that survives — one cast, one row
//! scale — and the statement that each backend resolves to its own mask.


