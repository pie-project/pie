//! The CPU engine: host memory for state no device holds.
//!
//! Today it is a pool of sealed pages ([`Pool`]): the bytes of each page as
//! the engine that sealed it lays a host slot out, with that engine's word
//! on the layout, kept by content key. It reads no model and computes
//! nothing; any model's pages share it, since a key names its model too.
//! Workers send it pages to keep and fetch them back by key, over the bulk
//! link, as they would from any worker.

pub mod pool;

pub use pool::Pool;
