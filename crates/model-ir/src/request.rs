//! One request's shape facts, as the runtime states them per fire; a model's
//! `Classify::of` reads what it declared bits for.

/// One request's shape facts, as the runtime states them per fire: row
/// count, whether it carries a custom mask, an adapter route, a draft head,
/// and score capture. A model's `Classify::of` reads what it declared bits
/// for and ignores the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    query_len: u32,
    custom_mask: bool,
    adapter: bool,
    drafts: bool,
    captures_scores: bool,
    media: bool,
    denoise: bool,
}

impl Request {
    #[must_use]
    pub fn new(query_len: u32, custom_mask: bool) -> Request {
        Request {
            query_len,
            custom_mask,
            adapter: false,
            drafts: false,
            captures_scores: false,
            media: false,
            denoise: false,
        }
    }

    /// The same request, read as a block-diffusion denoiser's canvas: its
    /// rows attend bidirectionally and its input is the denoiser's (the
    /// self-conditioned, post-normed embedding), not the encoder's. A fact
    /// only a diffusion text declares a bit for.
    #[must_use]
    pub fn denoising(mut self, denoise: bool) -> Request {
        self.denoise = denoise;
        self
    }

    /// The same request, routing to an adapter bank.
    #[must_use]
    pub fn adapted(mut self, adapter: bool) -> Request {
        self.adapter = adapter;
        self
    }

    /// The same request, with the model's draft head run over its rows.
    #[must_use]
    pub fn drafting(mut self, drafts: bool) -> Request {
        self.drafts = drafts;
        self
    }

    /// The same request, with its attention's per-key mass kept.
    #[must_use]
    pub fn capturing_scores(mut self, captures_scores: bool) -> Request {
        self.captures_scores = captures_scores;
        self
    }

    /// The same request, carrying images. Without this bit a text-only fire
    /// of a vision load resolves `RuntimeInput::PatchRoutes` and panics.
    #[must_use]
    pub fn with_media(mut self, media: bool) -> Request {
        self.media = media;
        self
    }

    #[must_use]
    pub fn query_len(&self) -> u32 {
        self.query_len
    }

    #[must_use]
    pub fn has_custom_mask(&self) -> bool {
        self.custom_mask
    }

    #[must_use]
    pub fn denoise(&self) -> bool {
        self.denoise
    }

    #[must_use]
    pub fn has_adapter(&self) -> bool {
        self.adapter
    }

    #[must_use]
    pub fn drafts(&self) -> bool {
        self.drafts
    }

    #[must_use]
    pub fn captures_scores(&self) -> bool {
        self.captures_scores
    }

    #[must_use]
    pub fn has_media(&self) -> bool {
        self.media
    }
}

/// How a model packs a request into the fact word a lane carries.
pub type ClassifyFn = fn(&Request) -> u64;
