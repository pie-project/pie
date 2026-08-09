//! A lowered launch becomes a kernel entry, its arguments and its grid.
//!
//! Which symbol, what grid and which addresses are decided in
//! [`crate::lowering`], with no device and under test on a host. What is
//! here is the half that cannot be: compiling the symbols, staging the
//! argument tables, and encoding the dispatch.
//!
//! * [`encode`] — compile the symbols, bind the addresses, dispatch.
//! * [`tables`] — the fire's own tables, staged once, so the seam and the
//!   real-weight gate cannot drift on which ones exist. It allocates, which
//!   is the whole reason it is not host logic.

pub mod encode;
pub mod tables;
