//! The six `Dispatch*` impls: every arm is destructure → resolve → call
//! (decision #13), one arm per variant, matches exhaustive.
//!
//! No arm selects a kernel — dtype and variant choice live inside the
//! `kernels-metal` entries — and no arm syncs (#15): a returned `Ok`
//! means the launch is encoded, nothing more. Alias outputs
//! (`#[out(alias = x)]`) bind as `_`: the compiler folded them onto their
//! input's slot, so the input name is the one the in-place kernel reads.
//! Families the metal plane stubs as `Unsupported` still get real arms that
//! forward to the stub, so the typed refusal carries the entry's own name.
//!
//! The six families each absorb several of the old ones; inside a match the
//! arms stay grouped by the family they came from, in the order the merged
//! enum lists them, and a section comment names each absorbed group. The
//! `kernels-metal` module a group calls into is family-first too now
//! (`attn::mla`, `attn::ssm`, `linear::moe`, `elemwise::rope`, …): the kernel
//! plane's tree follows the same six families the IR names, so a call site
//! reads family, then group, then entry.
//!
//! The impls live one per family in the modules below — one file per
//! `Dispatch*` trait, each importing only what its own arms resolve through.

mod attn;
mod collective;
mod custom;
mod elemwise;
mod layout;
mod linear;
