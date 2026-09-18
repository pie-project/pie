//! `wasi:filesystem@0.2`'s resource types, for the bindings only: a tab has
//! no files, so the interfaces are not linked and a guest that imports them
//! is refused at instantiation, by name. The guests pie ships import no
//! filesystem interface at all.

pub struct Descriptor;
pub struct DirectoryEntryStream;
