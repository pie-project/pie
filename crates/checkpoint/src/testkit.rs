//! Exists to check the loader rather than be the loader: [`mod@reference`],
//! the differential oracle plans are replayed against. Not on the load path;
//! `worker/Cargo.toml` turns `testkit` off, which is also the enforcement.

pub mod reference;
