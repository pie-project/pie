//! The engine's on-device end-to-end suite. **There is no code here** — this
//! crate exists to own `tests/`.
//!
//! Thirty-one of these lived in `bin/pie/tests/`, and only two of the tests
//! there ever named a CLI symbol. They were in the CLI crate because it is the
//! one crate that links a driver, not because they test a CLI: `cargo test -p
//! pie-bin` meant "compile the GPU suite", and the four tests that are actually
//! about the CLI were invisible inside it.
//!
//! Almost everything here is `#[ignore]`d and asks for a real device, so a
//! `cargo test --workspace` compiles them and runs none. Run one explicitly:
//!
//! ```text
//! cargo test -p pie-gpu-tests --features driver-cuda \
//!     --test cuda_contention -- --ignored --nocapture
//! ```
//!
//! **One boot per process.** The runtime owns process-global singletons (`auth`
//! panics on a second boot; the driver takes a fixed POSIX shmem), so every
//! test that boots lives in its own test binary — which is why this is thirty
//! files rather than one.
