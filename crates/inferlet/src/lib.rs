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

pub mod eta;
pub mod mask;
pub mod speculative;

pub mod chat;

pub mod latent;

pub mod model {
    pub use crate::pie::inferlet::model::{
        BlockDrafter, InputAxis, InputDecl, LaneDecl, LatentSpace, OutputDecl, RowCount,
        ScheduleFact, ScheduleKind, SwitchDecl, architecture, channel_capacity, draft_block,
        frame_size, kv_page_size, lane, latent, max_pass_rows, name, output_vocab_size,
        rs_state_size, run_ahead_window, schedule, submit_deadline_us, switches,
    };

    /// What a lane under these `(name, value)` switches binds.
    pub fn lane_of(switches: &[(&str, &str)]) -> Result<LaneDecl, String> {
        let pairs: Vec<(String, String)> = switches
            .iter()
            .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
            .collect();
        lane(&pairs)
    }

    /// The switch `name` the model declares.
    #[must_use]
    pub fn switch(name: &str) -> Option<SwitchDecl> {
        switches().into_iter().find(|decl| decl.name == name)
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

/// The name in a `{"$handle": name}` a client wrote where a handle goes.
pub(crate) fn handle_marker<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<String, D::Error> {
    #[derive(serde::Deserialize)]
    struct Marker {
        #[serde(rename = "$handle")]
        handle: String,
    }
    Ok(<Marker as serde::Deserialize>::deserialize(deserializer)?.handle)
}

pub mod frames {
    pub use crate::pie::inferlet::frames::{AudioFormat, Frames, ImageFormat, Pcm};
    use crate::pie::inferlet::session;

    impl Frames {
        /// The handle the launch moved in under `name`.
        pub fn take(name: &str) -> Option<Frames> {
            session::take_frames(name)
        }

        /// Hands these pictures to the client under the launch's `name`.
        pub fn give(self, name: &str) -> Result<(), String> {
            session::give_frames(name, self)
        }
    }

    impl Pcm {
        /// The handle the launch moved in under `name`.
        pub fn take(name: &str) -> Option<Pcm> {
            session::take_pcm(name)
        }

        /// Hands this sound to the client under the launch's `name`.
        pub fn give(self, name: &str) -> Result<(), String> {
            session::give_pcm(name, self)
        }
    }

    /// `Frames` in a typed input is the handle the client wrote there.
    impl<'de> serde::Deserialize<'de> for Frames {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let handle = crate::handle_marker(deserializer)?;
            Frames::take(&handle).ok_or_else(|| {
                serde::de::Error::custom(format!("the launch moved in no frames `{handle}`"))
            })
        }
    }

    /// `Pcm` in a typed input is the handle the client wrote there.
    impl<'de> serde::Deserialize<'de> for Pcm {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let handle = crate::handle_marker(deserializer)?;
            Pcm::take(&handle).ok_or_else(|| {
                serde::de::Error::custom(format!("the launch moved in no pcm `{handle}`"))
            })
        }
    }
}

pub mod grammar {
    pub use crate::pie::inferlet::grammar::*;

    impl Matcher {
        /// The handle the launch moved in under `name`.
        pub fn take(name: &str) -> Option<Matcher> {
            crate::pie::inferlet::session::take_matcher(name)
        }

        /// Hands this matcher back to the client under the launch's `name`.
        pub fn give(self, name: &str) -> Result<(), String> {
            crate::pie::inferlet::session::give_matcher(name, self)
        }
    }

    /// A `Matcher` in a typed input is the handle the client wrote there.
    impl<'de> serde::Deserialize<'de> for Matcher {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let handle = crate::handle_marker(deserializer)?;
            Matcher::take(&handle).ok_or_else(|| {
                serde::de::Error::custom(format!("the launch moved in no matcher `{handle}`"))
            })
        }
    }
}

pub mod media {
    pub use crate::pie::inferlet::forward::MediaSpan as Span;
    pub use crate::pie::inferlet::media::{Audio, Image, Video};
}
