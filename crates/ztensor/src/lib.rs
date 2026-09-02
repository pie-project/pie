//! zTensor v3: an aligned, verifiable container format for tensor data.
//!
//! zTensor carries what can be *proved* about a checkpoint's bytes: where they
//! are ([`Tensor::locate`]), whether they are intact ([`Tensor::verify`]),
//! whether they can be dropped without disturbing a neighbour
//! ([`Tensor::evict`]), and which model they are
//! ([`Manifest::content_digest`]).
//!
//! ```no_run
//! use ztensor::{Leaf, Source, Writer};
//!
//! let mut w = Writer::create("model.zt")?;
//! w.add("weights", [2u64, 2], Leaf::F32, &[0u8; 16])?;
//! w.finish()?;
//!
//! let src = Source::open("model.zt")?;
//! let t = src.tensor("weights")?;
//! let bytes = t.map()?;      // borrowed, or an error; never a copy
//! # Ok::<(), ztensor::Error>(())
//! ```
//!
//! # The modules are the spec's layers
//!
//! The format is specified in `spec/ztensor-v3-spec.md`, which separates
//! three layers, and this crate is laid out to match:
//!
//! - [`format`](mod@format) — **L0 container and L1 manifest**, frozen. The
//!   magic, the 40-byte footer, the alignment floor, the manifest schema and
//!   its CBOR mapping, the type grammar ([`Term`]) and the canonical layout it
//!   implies, and the rules that decide conformance and canonical form.
//!   Nothing here opens a file.
//! - [`vocab`](mod@vocab) — **L2**, open and registry-managed: named layouts
//!   and encodings, which another crate can extend.
//! - [`read`](mod@read) — opening `.zt` and getting at bytes.
//! - [`write`](mod@write) — producing `.zt`.
//! - [`provide`](mod@provide) — the face turned towards a crate that projects
//!   a *foreign* format into a [`Source`].
//!
//! # Getting bytes
//!
//! A tensor is one blob. [`bytes`](Tensor::bytes) gives the best the source
//! can do as a `Cow`, [`map`](Tensor::map) insists on a borrow, and
//! [`locate`](Tensor::locate) gives the address so the caller can do the I/O
//! itself. Where the type has several planes, [`Tensor::planes`] says where
//! each lies inside those bytes; a `g64_u4_bf16_b_bf16` tensor is its codes,
//! then its scales, then its biases, at offsets derived from the shape.
//!
//! # Two layers of description
//!
//! [`Manifest`] is what one `.zt` file literally says.
//! [`Catalog`](provide::Catalog) is the resolved index a consumer queries,
//! whose addresses are [`StoreId`]s and which can therefore span files that
//! never heard of each other. Foreign projections build catalogs; only a
//! `.zt` root has a manifest, which is what [`Source::provenance`] reports.

mod error;
pub mod format;
pub mod provide;
pub mod read;
pub mod vocab;
pub mod write;

pub use error::{Error, Result, Rule};
pub use format::{
    Blob, Blocks, Digest, DigestAlgorithm, Group, Leaf, Manifest, Object, Offset, Plane, Shard,
    Term,
};
pub use provide::{Location, Store, StoreId};
pub use read::{Caps, Provenance, Source, Tensor, Verified};
pub use vocab::Vocabulary;
pub use write::{ObjectBuilder, Sink, Writer};
