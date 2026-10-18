//! Worker ↔ worker: the peer link. Every worker serves [`Peer`] on its peer
//! port; a worker dials another directly, not through a gateway.
//!
//! Metadata and requests ride this link; pages' bytes ride the bulk link
//! (`transport::tcp`), one port above it.

use ids::WorkerId;
use serde::{Deserialize, Serialize};
use state::{ModelIdentity, NumericsClass, StateFormat};

/// Largest frame the peer link carries: room for one prefill's pages inline.
pub const PEER_MAX_FRAME_BYTES: usize = 1 << 30;

/// What a worker says about itself when a link opens. Two workers can share
/// state only when their identities agree; whether they can use each
/// other's bytes as they are depends on class and format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hello {
    pub worker: WorkerId,
    /// None: a worker that loads no model.
    pub identity: Option<ModelIdentity>,
    pub class: Option<NumericsClass>,
    pub format: Option<StateFormat>,
}

/// Prefill `tokens` and return the pages from `from_page` on that the run
/// fills completely. The asker already holds pages before `from_page`, and
/// computes the partial page at the end itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prefill {
    pub tokens: Vec<u32>,
    pub from_page: u32,
    /// The asker's bulk-bytes address: the executor fetches from it, by key,
    /// the pages before `from_page` it holds nowhere itself.
    pub home: Option<String>,
}

/// The pages a prefill sealed, left in the executor's cache: `count` from
/// token `start` (a page boundary), to fetch by key from `bulk`, the
/// executor's bulk-bytes address.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pages {
    pub start: u64,
    pub count: u32,
    pub class: NumericsClass,
    pub format: StateFormat,
    pub bulk: String,
}

#[tarpc::service]
pub trait Peer {
    async fn hello(hello: Hello) -> Result<Hello, String>;

    async fn prefill(request: Prefill) -> Result<Pages, String>;
}
