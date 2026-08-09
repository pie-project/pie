//! `store/`: the long-lived device pools and the arithmetic that sizes them.
//!
//! The C++ `store/` is 5,012 lines containing not one `__global__`. It is the
//! largest block of pure host logic in the shell, and it is the reason the
//! rewrite starts where it does: everything here is page arithmetic, budget
//! accounting, and format resolution, none of which needs a GPU to be correct
//! and none of which can currently be tested without one.
//!
//! The device-facing half of this subsystem -- the virtual-memory arena that
//! `store/elastic.{hpp,cpp}` defines -- lives in [`crate::cuda::vmm`] instead,
//! because it is the substrate the rest of the shell allocates out of rather
//! than a store concern. What is left here is what sits on top of it.

mod kv_format;
/// Owns `DeviceTensor`s, so it carries the gate.
#[cfg(feature = "_cuda")]
pub mod dsv4_compress_cache;
pub mod dsv4_geometry;
pub mod dtoa;
pub mod json;
/// Owns `DeviceTensor`s, so it carries the gate. `kv_geometry` next to
/// it is the portable half — the shapes, without the memory.
#[cfg(feature = "_cuda")]
pub mod kv_cache;
/// The one store module that CALLS CUDA — it allocates and copies. Gated
/// so that its eighteen pure-arithmetic neighbours need not be.
#[cfg(feature = "_cuda")]
pub mod kv_cache_live;
pub mod kv_geometry;
pub mod mla_cache;
pub mod mla_geometry;
pub mod memory_planner;
pub mod plan;
pub mod planner_policy;
pub mod calibrate;
pub mod model_costs;
pub mod profile_cache;
pub mod profile_key;
pub mod recurrent_layout;
pub mod recurrent_state_cache;
pub mod swap_plan;
pub mod swap_pool;

pub use kv_format::{KvCacheFormat, KvCacheScaleLayout, KvCacheScheme};
