//! `Dispatch*` impls: one arm per variant, each resolving the plan's
//! output/slot kind and calling the matching kernel entry.

mod attn;
mod collective;
pub(crate) mod copy;
mod custom;
mod elemwise;
mod layout;
mod linear;
