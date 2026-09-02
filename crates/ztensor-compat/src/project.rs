//! What a foreign format hands back.
//!
//! A projection reads a file's own metadata and says three things: what
//! tensors are in it and where ([`Catalog`]), which byte ranges the file
//! occupies (so page exclusivity can be decided rather than guessed), and,
//! for formats whose bytes are not simply lying there, how to produce the
//! ones that have no address.
//!
//! It never builds a [`Manifest`](ztensor::format::Manifest). A safetensors
//! file has no manifest; saying otherwise would be inventing a document
//! nobody wrote.

use ztensor::provide::{Catalog, Decode};
use ztensor::{Result, Source, Store, Vocabulary};

pub(crate) struct Projection {
    pub catalog: Catalog,
    /// Every byte range the file is known to use, including its header and
    /// any index. Left empty when the format cannot say, and then the store
    /// never claims page exclusivity.
    pub occupied: Vec<(u64, u64)>,
    pub decoder: Option<Box<dyn Decode>>,
}

impl Projection {
    pub fn new(catalog: Catalog) -> Self {
        Self {
            catalog,
            occupied: Vec::new(),
            decoder: None,
        }
    }

    pub fn occupying(mut self, ranges: Vec<(u64, u64)>) -> Self {
        self.occupied = ranges;
        self
    }

    #[cfg(any(feature = "npz", feature = "pickle", feature = "hdf5", feature = "onnx"))]
    pub fn with_decoder(mut self, decoder: Box<dyn Decode>) -> Self {
        self.decoder = Some(decoder);
        self
    }

    pub fn into_source(self, store: Store, vocab: Option<&Vocabulary>) -> Result<Source> {
        let mut store = store.with_occupied(self.occupied);
        if let Some(decoder) = self.decoder {
            store = store.with_decoder(decoder);
        }
        let mut options = Source::options();
        if let Some(v) = vocab {
            options = options.vocabulary(v);
        }
        options.from_parts(vec![store], self.catalog)
    }
}
