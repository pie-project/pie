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
    block_draft: bool,
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
            block_draft: false,
        }
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

    /// The same request, carrying a BLOCK DRAFTER's proposal rows rather
    /// than rows of the sequence itself.
    ///
    /// A block drafter (`qwen_3`'s [`Recipe::DFlash`]) proposes many tokens
    /// in one pass over a block whose first row is the correction the target
    /// just made and whose rest is the mask token. Those rows are not the
    /// model's own — the trunk must not run over them — so this is what a
    /// plan guards its trunk against, and it is the guest's to set: only the
    /// inferlet knows the accepted prefix that anchors the block, so only it
    /// can say which fire is a draft.
    ///
    /// Distinct from [`drafting`](Request::drafting), which asks a plan to
    /// run its draft head over rows the trunk ALSO processes.
    #[must_use]
    pub fn drafting_a_block(mut self, block_draft: bool) -> Request {
        self.block_draft = block_draft;
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
    pub fn has_adapter(&self) -> bool {
        self.adapter
    }

    #[must_use]
    pub fn drafts(&self) -> bool {
        self.drafts
    }

    #[must_use]
    pub fn drafts_a_block(&self) -> bool {
        self.block_draft
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
