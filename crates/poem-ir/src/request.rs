#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Stream {
    #[default]
    Text,
    Image,
    Video,
    Audio,
    Context,
    Reference,
}

impl Stream {
    pub const ALL: [Stream; 6] = [
        Stream::Text,
        Stream::Image,
        Stream::Video,
        Stream::Audio,
        Stream::Context,
        Stream::Reference,
    ];

    #[must_use]
    pub fn code(self) -> u8 {
        self as u8
    }

    #[must_use]
    pub fn from_code(code: u8) -> Option<Stream> {
        Stream::ALL.get(usize::from(code)).copied()
    }

    #[must_use]
    pub fn word(self, base: u8) -> u64 {
        1u64 << (base + self.code())
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Stream::Text => "text",
            Stream::Image => "image",
            Stream::Video => "video",
            Stream::Audio => "audio",
            Stream::Context => "context",
            Stream::Reference => "reference",
        }
    }
}

/// What the runtime knows of one lane of a fire: the buffers it carries,
/// from which the builtin facts are derived, and the custom facts its
/// inferlet set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    query_len: u32,
    custom_mask: bool,
    adapter: bool,
    drafts: bool,
    captures_scores: bool,
    media: bool,
    block_draft: bool,
    denoise: bool,
    stream: Stream,
    reading: Option<Name>,
}

impl Request {
    /// The custom flags an inferlet can set today.
    pub const FLAGS: [&'static str; 4] = ["drafts", "block_draft", "scores", "bidirectional"];
    /// The custom choices an inferlet can set today.
    pub const CHOICES: [&'static str; 2] = ["stream", "reading"];

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
            denoise: false,
            stream: Stream::Text,
            reading: None,
        }
    }

    #[must_use]
    pub fn on_stream(mut self, stream: Stream) -> Request {
        self.stream = stream;
        self
    }

    #[must_use]
    pub fn in_reading(mut self, reading: &str) -> Request {
        self.reading = Some(Name::of(reading));
        self
    }

    #[must_use]
    pub fn denoising(mut self, denoise: bool) -> Request {
        self.denoise = denoise;
        self
    }

    #[must_use]
    pub fn adapted(mut self, adapter: bool) -> Request {
        self.adapter = adapter;
        self
    }

    #[must_use]
    pub fn drafting(mut self, drafts: bool) -> Request {
        self.drafts = drafts;
        self
    }

    #[must_use]
    pub fn drafting_a_block(mut self, block_draft: bool) -> Request {
        self.block_draft = block_draft;
        self
    }

    #[must_use]
    pub fn capturing_scores(mut self, captures_scores: bool) -> Request {
        self.captures_scores = captures_scores;
        self
    }

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

    #[must_use]
    pub fn stream(&self) -> Stream {
        self.stream
    }

    #[must_use]
    pub fn reading(&self) -> Option<&str> {
        self.reading.as_ref().map(Name::as_str)
    }

    /// The custom flag `name`, one of [`Request::FLAGS`].
    #[must_use]
    pub fn flag(&self, name: &str) -> bool {
        match name {
            "drafts" => self.drafts,
            "block_draft" => self.block_draft,
            "scores" => self.captures_scores,
            "bidirectional" => self.denoise,
            _ => panic!("`{name}` is no flag a request carries"),
        }
    }

    /// The value the custom choice `name`, one of [`Request::CHOICES`], is
    /// set to.
    #[must_use]
    pub fn choice(&self, name: &str) -> Option<&str> {
        match name {
            "stream" => Some(self.stream.name()),
            "reading" => self.reading(),
            _ => panic!("`{name}` is no choice a request carries"),
        }
    }
}

/// A reading's name, held inline so a request stays `Copy`.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Name {
    len: u8,
    bytes: [u8; Name::MAX],
}

impl Name {
    const MAX: usize = 31;

    fn of(name: &str) -> Name {
        assert!(
            name.len() <= Name::MAX,
            "a reading's name is at most {} bytes, and `{name}` is longer",
            Name::MAX
        );
        let mut bytes = [0; Name::MAX];
        bytes[..name.len()].copy_from_slice(name.as_bytes());
        Name {
            len: name.len() as u8,
            bytes,
        }
    }

    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..usize::from(self.len)])
            .expect("a name is copied from a str")
    }
}

impl std::fmt::Debug for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_str().fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::{Request, Stream};

    #[test]
    fn a_request_defaults_to_the_text_stream_and_the_default_reading() {
        let r = Request::new(4, false);
        assert_eq!(r.stream(), Stream::Text);
        assert_eq!(r.reading(), None);
        let r = r.on_stream(Stream::Audio).in_reading("denoise");
        assert_eq!(r.stream(), Stream::Audio);
        assert_eq!(r.reading(), Some("denoise"));
        assert_eq!(r.choice("stream"), Some("audio"));
        for stream in Stream::ALL {
            assert_eq!(Stream::from_code(stream.code()), Some(stream));
            assert_eq!(stream.word(8), 1 << (8 + stream.code()));
        }
        assert_eq!(Stream::from_code(6), None);
    }
}
