//! Foreign tensor formats, projected into the zTensor object model — and the
//! layout profiles that assemble an object once it is read.
//!
//! Every format here builds a [`Catalog`](ztensor::provide::Catalog) of names,
//! shapes, types and byte locations, then hands back an ordinary
//! [`Source`](ztensor::Source). There is no per-format type in the public API,
//! because after projection there is nothing per-format left to say:
//!
//! ```no_run
//! let src = ztensor_compat::open("model.safetensors")?;
//! let t = src.tensor("layer.weight")?;
//! let bytes = t.map()?;  // zero-copy where the file allows it
//! # Ok::<(), ztensor::Error>(())
//! ```
//!
//! Projections are read-only and honest. A tensor's payload says which of the
//! three shapes it has: a raw addressable range, something stored under an
//! encoding, or bytes only the format's own reader can produce, and
//! [`Caps`](ztensor::Caps) is a direct reading of that. Nothing is silently
//! reinterpreted or dequantized, and nothing degrades quietly to a copy.
//!
//! A projected file reports its *actual* offsets; the `.zt` guarantees (the
//! 4096 floor, page exclusivity, digests) hold only for `.zt` files. To
//! upgrade a foreign checkpoint, convert it: `Writer::ingest` copies any
//! source into a canonical `.zt` file.

pub mod csr;

mod detect;
mod project;
mod safe;

#[cfg(feature = "safetensors")]
mod safetensors;

#[cfg(feature = "gguf")]
mod gguf;

#[cfg(feature = "npz")]
mod npz;

#[cfg(feature = "pickle")]
mod pt;

#[cfg(feature = "hdf5")]
mod hdf5;

#[cfg(feature = "onnx")]
mod onnx;

pub use detect::{detect, index, index_all, open, open_all, options, Open, FORMATS};
