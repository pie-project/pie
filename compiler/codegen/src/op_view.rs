//! The flat wire record the backend emitters switch on.
//!
//! Both emitters dispatch on a *decoded wire op* — tag, positional args,
//! result count, channel index and a handful of immediates — rather than on
//! the typed [`Op`](pie_ir::op::Op) enum, which is what makes the CUDA and
//! Metal region emitters diffable line by line.
//!
//! The record itself is defined in [`pie_ir::wire`], next to the op table's
//! layout column, so the projection this crate reads and the bytes the
//! container writes come from one declaration. Defining it here instead would
//! make it a third transcription of a byte layout `pie_ir` already encodes and
//! decodes, and the drift that produces is silent: this crate would read a
//! field the container never wrote. It also keeps `OpWire::of` and
//! `OpWire::to_op` being inverses a test in `pie-ir` rather than an invariant
//! this crate hopes for.
//!
//! This module is only the name the emitters know the record by.

pub use pie_ir::wire::{OpWire as OpView, result_bases};
