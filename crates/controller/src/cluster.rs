//! The cluster plane of PLEX 1.1 (`plex:policy@1.1.0`, the `plex-host`
//! crate): where processes run and where KV is copied. Its policies run
//! here, once for the cluster; the gateways send what happens to their
//! processes and apply the rows the policies commit about them.

mod port;

pub use port::{Host, LoadError, call_deadline};
