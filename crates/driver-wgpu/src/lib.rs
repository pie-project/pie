//! The WebGPU execution shell: what it takes to actually FIRE the modules
//! `kernels-wgpu` states. Pure Rust, no `-sys` crate in the closure
//! (`tests/pure.rs`); `naga` parses WGSL anywhere, so [`reflect`] checks a
//! module against its row with no adapter; `native` gates what must ask one.
//!
//! **WebGPU has no push constants.** A launch's scalars are the fields of ONE
//! uniform buffer the shell writes and binds at `@group(1) @binding(0)`, so
//! [`binding::Params`] carries one variant where `driver-vulkan` carries two,
//! with the slot read as data off `kernels_wgpu::bindings`.
//!
//! **A dispatch may not bind one buffer both readable and writable**, tracked
//! per ALLOCATION, so [`binding::Arena`]'s one buffer trips it on every real
//! plan: `device::Device::run_all` shadows the offending read ranges. WGSL has
//! no 16-bit storage type, so bf16 crosses as `array<u32>` and a decode lane
//! owns the PAIR -- [`geometry::Rule::SdpaVector`] halves the workgroup too.
//!
//! **The seam is exactly this, and nothing else:**
//!
//! ```ignore
//! impl driver_wgpu::binding::Allocation for device::Buffer {
//!     fn size(&self) -> u64 { self.size }
//! }
//! impl driver_wgpu::binding::Resolve for Store {
//!     type Buffer = device::Buffer;
//!     fn weight(&self, name: &str) -> Option<&device::Buffer>;
//!     fn named(&self, value: model_ir::trace::ValueId) -> Option<&device::Buffer>;
//!     // Defaulted to `None`; state them only where the text needs them.
//!     fn kv(&self, layer: u16, values: bool) -> Option<&device::Buffer>;
//!     fn slab(&self, layer: u16, which: &'static str) -> Option<&device::Buffer>;
//!     fn number(&self, which: binding::FireNumber) -> Option<u32>;
//!     fn table(&self, which: binding::FireTable) -> Option<&device::Buffer>;
//! }
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout)]

// Ungated: binding, geometry, lowering, dispatch and reflection are arithmetic
// over a plan, a row and a shader SOURCE, so none of them needs `wgpu`.
pub mod binding;
pub mod dispatch;
pub mod facts;
pub mod geometry;
pub mod lowering;
pub mod runtime;
// Not a module of this crate: `driver::names` is the one table, and the copy
// that stood here was byte-for-byte identical to `driver-vulkan`'s 412 lines.
// Re-exported rather than referred to directly so `driver_wgpu::names` keeps
// answering, which is what this crate's own `tests/checkpoint.rs` asks.
pub use driver::names;
pub mod programs;
pub mod reflect;
pub mod rope;

// The device half: everything below needs an ADAPTER to answer.
#[cfg(feature = "native")]
pub mod device;
// Gated with the serving path because it names `turns::Serving`, though
// nothing in it touches a device.
#[cfg(feature = "native")]
pub mod envelope;
#[cfg(feature = "native")]
pub mod frames;
#[cfg(feature = "native")]
pub mod serve;
// `Encode` over a real adapter. Gated because it names a `Pipeline`.
#[cfg(feature = "native")]
pub mod encode;
#[cfg(feature = "native")]
pub mod shell;
#[cfg(feature = "native")]
pub mod turns;

// UNGATED both: `resources`' `Shape`, `Request` and `Frame` are integer
// arithmetic over page numbers, and `pages` speaks only in those.
pub mod pages;
pub mod resources;

pub use binding::{Allocation, Arena, Bound, Resolve, Unbindable, bind, resolve};
pub use geometry::{Dims, Local, Module, Rule, Tile, Ungeometric, groups, groups_within, lanes};
/// The row a symbol belongs to.
pub use kernels::sig_in;
/// The tier vocabulary this driver was built against.
///
/// `KERNELS` was re-exported beside it, because a caller asking "how wide a
/// fire can this driver take" needed the same table the dispatcher read and a
/// second copy would be a second thing that can drift. There is no table now
/// and no dispatcher that reads one: `engine`'s `every_launch_fits` asks
/// `dispatch::plan_one`, which plans through the arm.
pub use kernels_wgpu::Capability;
pub use reflect::Declared;
