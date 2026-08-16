//! pie:core/types - error + blob type aliases. The async surface is native
//! component-model-async (`async func`), so `receive` / `receive-file` /
//! `pull` await directly host-side.

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;

impl pie::inferlet::types::Host for ProcessCtx {}
