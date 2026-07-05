//! pie:core/types - error + blob type aliases.
//!
//! The hand-rolled async futures (future-string / future-blob) are gone:
//! the async surface is now native component-model-async (`async func`),
//! so `receive` / `receive-file` / `pull` await directly host-side.

use crate::api::pie;
use crate::instance::InstanceState;

impl pie::core::types::Host for InstanceState {}
