//! Building a [`Source`](crate::Source) out of something that is not a `.zt`.
//!
//! Reading a checkpoint needs nothing from this module. It is the face zTensor
//! turns towards a crate that projects a foreign format — `ztensor-compat` is
//! the one in this workspace — and it is separate so that the reader's own
//! surface is not half full of the machinery behind it.
//!
//! A projection is two things. A [`Store`] is a file: a path, a length, and
//! optionally a mapping of it. A [`Catalog`] is the index a consumer queries:
//! names to [`Entry`]s, each addressed as a [`StoreId`] and a byte range.
//! Hand both to [`Source::from_parts`](crate::Source::from_parts) and the
//! result is an ordinary `Source`, indistinguishable from one that came out of
//! a `.zt` except that it reports [`Provenance::Projection`](crate::Provenance::Projection).
//!
//! ```no_run
//! use ztensor::provide::{Catalog, Entry, Location, Store, StoreId};
//! use ztensor::{Leaf, Source};
//!
//! let store = Store::map("weights.bin", "my-format")?;
//! let mut catalog = Catalog::new();
//! catalog.insert(
//!     "w",
//!     Entry::leaf(
//!         vec![4, 4],
//!         Leaf::F32,
//!         Location { store: StoreId(0), offset: 0, len: 64 },
//!     ),
//! );
//! let source = Source::from_parts(vec![store], catalog)?;
//! # Ok::<(), ztensor::Error>(())
//! ```
//!
//! [`Store`] and [`StoreId`] are also part of the reader's surface, because a
//! consumer that plans I/O has to know which file a tensor is in; they are
//! re-exported at the crate root for that reason. What is only here is how to
//! *build* one.

pub(crate) mod catalog;
pub(crate) mod store;

pub use catalog::{Catalog, Entry, Location, Payload};
pub use store::{page_size, Decode, Store, StoreId};
