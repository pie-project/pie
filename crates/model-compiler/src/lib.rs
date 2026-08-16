//! LOWERING — the traced form to the flat launch list a driver executes.
//!
//! ```text
//! declaration  ──trace──▶  forward plan  ──lower──▶  driver executes
//!  `model-dsl`             `model-ir`               THIS CRATE
//! ```
//!
//! * [`lower`] — a [`ForwardPlan`](model_ir::ForwardPlan) plus the engine's
//!   rows and one fire, to rectangles, operands and buffers.
//!
//! # What this crate is NOT
//!
//! It is not the authoring eDSL, and until the toolchain was split it was
//! bundled with one. `model-compiler` used to be the whole toolchain — the
//! surface, the IR and the lowering in five modules — and the reason to cut it
//! was not size but that **no consumer wanted all three**: `crates/model`
//! writes declarations and never lowers one; every driver lowers and never
//! writes one. The two sets did not overlap in a single crate.
//!
//! What that cost was concrete. `driver-metal`, `driver-vulkan` and
//! `driver-wgpu` each compiled `dsl::cuda`'s 4,469 lines to reach this file,
//! and not one of them called a line of it — every `dsl::` path in a driver
//! was a doc link. Now a driver depends on this crate and [`model_ir`], and
//! the authoring surface is not in its graph at all.
//!
//! The name stayed with the lowering rather than following the eDSL because a
//! *compiler* is the thing that turns a representation into what a machine
//! runs, and that is what this does. What a declaration is written in is
//! `model-dsl`; what it becomes is `model-ir`; what a device is handed is
//! here.

pub mod lower;
