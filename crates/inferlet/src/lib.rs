pub type Result<T> = std::result::Result<T, String>;

pub trait Context<T> {
    fn context(self, what: &str) -> Result<T>;

    fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, what: F) -> Result<T>;
}

impl<T, E: std::fmt::Display> Context<T> for std::result::Result<T, E> {
    fn context(self, what: &str) -> Result<T> {
        self.map_err(|error| format!("{what}: {error}"))
    }

    fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, what: F) -> Result<T> {
        self.map_err(|error| format!("{}: {error}", what()))
    }
}

impl<T> Context<T> for Option<T> {
    fn context(self, what: &str) -> Result<T> {
        self.ok_or_else(|| what.to_string())
    }

    fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, what: F) -> Result<T> {
        self.ok_or_else(|| what().to_string())
    }
}

pub use serde;
pub use serde_json;

pub use inferlet_macros::main;

wit_bindgen::generate!({
    path: "wit",
    world: "inferlet",
    pub_export_macro: true,
    generate_all,
});

pub use wit_bindgen;

pub use pie::inferlet::types;

pub mod working_set {
    pub use crate::pie::inferlet::working_set::*;
}

pub mod eta;
pub mod mask;

pub mod chat;

pub mod latent;

pub mod model {
    pub use crate::pie::inferlet::model::{
        AxisRole, BlockDrafter, CanvasShape, ForwardKind, LaneStream, LatentSpace, PortFact,
        PortKind, PositionConvention, ReadingFact, ReadoutKind, ScheduleFact, ScheduleKind,
        architecture, arena_block_size, canvas, channel_capacity, default_system_speculation,
        draft_block, frame_size, kv_page_size, latent, max_embed_length, max_latent_rows,
        mtp_depth, name, output_vocab_size, pass_kind, prefill_chunk_hint, readings,
        rs_buffer_page_size, rs_fold_granularity, rs_state_size, run_ahead_window, schedule,
        submit_deadline_us,
    };

    pub fn reading(name: &str) -> Option<ReadingFact> {
        readings().into_iter().find(|reading| reading.name == name)
    }
    pub use crate::pie::inferlet::tokenizer::{
        Token, decode, encode, special_tokens, split_regex, token_bytes, tokens_with_prefix, vocabs,
    };
}

pub mod runtime {
    pub use crate::pie::inferlet::system::*;
}

pub async fn sleep(duration: std::time::Duration) {
    let nanos = duration.as_nanos().min(u64::MAX as u128) as u64;
    crate::wasi::clocks::monotonic_clock::wait_for(nanos).await;
}

pub fn monotonic_now_ns() -> u64 {
    crate::wasi::clocks::monotonic_clock::now()
}

pub mod session {
    pub use crate::pie::inferlet::session::*;
}

pub mod frames {
    pub use crate::pie::inferlet::frames::{AudioFormat, Frames, ImageFormat, Pcm};
}

pub mod grammar {
    pub use crate::pie::inferlet::grammar::*;
}

pub mod media {
    pub use crate::pie::inferlet::forward::MediaSpan as Span;
    pub use crate::pie::inferlet::media::{Audio, Image, Video};
}
