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
    reading: u8,
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
            denoise: false,
            stream: Stream::Text,
            reading: 0,
        }
    }

    #[must_use]
    pub fn on_stream(mut self, stream: Stream) -> Request {
        self.stream = stream;
        self
    }

    #[must_use]
    pub fn in_reading(mut self, reading: u8) -> Request {
        self.reading = reading;
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
    pub fn reading(&self) -> u8 {
        self.reading
    }
}

pub type ClassifyFn = fn(&Request) -> u64;

#[cfg(test)]
mod tests {
    use super::{Request, Stream};

    #[test]
    fn a_request_defaults_to_the_text_stream_and_the_default_reading() {
        let r = Request::new(4, false);
        assert_eq!(r.stream(), Stream::Text);
        assert_eq!(r.reading(), 0);
        let r = r.on_stream(Stream::Audio).in_reading(2);
        assert_eq!(r.stream(), Stream::Audio);
        assert_eq!(r.reading(), 2);
        for stream in Stream::ALL {
            assert_eq!(Stream::from_code(stream.code()), Some(stream));
            assert_eq!(stream.word(8), 1 << (8 + stream.code()));
        }
        assert_eq!(Stream::from_code(6), None);
    }
}
