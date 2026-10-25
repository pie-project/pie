//! The CPU engine: host memory for state no device holds.
//!
//! Today it is a pool of sealed pages ([`Pool`]): the bytes of each page as
//! the engine that sealed it lays a host slot out, with that engine's word
//! on the layout, kept by content key. The pool reads no model and
//! computes nothing; any model's pages share it, since a key names its
//! model too. Workers send it pages to keep and fetch them back by key,
//! over the bulk link, as they would from any worker. What memory cannot
//! hold it keeps in a [`Store`]: a directory, or (`nixl`) an object store
//! such as S3.
//!
//! It also runs a poem trace's forward on the host ([`Forward`]), rank by
//! rank of an expert-parallel group whose collectives meet in memory
//! ([`Exchange`]): one rank and `N` simulated ranks answer the same bits,
//! the check a group's lowering is held to without a cluster.

pub mod exchange;
pub mod forward;
pub mod pool;
pub mod store;

pub use exchange::Exchange;
pub use forward::{Forward, Group};
pub use pool::Pool;
pub use store::{Dir, Store};
