#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Request {
    query_len: u32,
    custom_mask: bool,
    adapter: bool,
    drafts: bool,
    captures_scores: bool,
    media: bool,
    encode: bool,
    fresh: bool,
    /// One left unset takes its declared default.
    switches: Vec<(String, String)>,
}

impl Request {
    #[must_use]
    pub fn new(query_len: u32, custom_mask: bool) -> Request {
        Request {
            query_len,
            custom_mask,
            ..Request::default()
        }
    }

    #[must_use]
    pub fn encoding(mut self, encode: bool) -> Request {
        self.encode = encode;
        self
    }

    #[must_use]
    pub fn starting(mut self, fresh: bool) -> Request {
        self.fresh = fresh;
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
    pub fn switched(mut self, name: &str, value: &str) -> Request {
        match self.switches.iter_mut().find(|(n, _)| n == name) {
            Some((_, v)) => *v = value.to_string(),
            None => self.switches.push((name.to_string(), value.to_string())),
        }
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
    pub fn captures_scores(&self) -> bool {
        self.captures_scores
    }

    #[must_use]
    pub fn has_media(&self) -> bool {
        self.media
    }

    #[must_use]
    pub fn encodes(&self) -> bool {
        self.encode
    }

    #[must_use]
    pub fn fresh(&self) -> bool {
        self.fresh
    }

    #[must_use]
    pub fn switch(&self, name: &str) -> Option<&str> {
        self.switches
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }

    #[must_use]
    pub fn switches(&self) -> &[(String, String)] {
        &self.switches
    }

    /// Runtime switches on (a multi-token query counting as one) plus guest switches set.
    #[must_use]
    pub fn departures(&self) -> usize {
        [
            self.query_len != 1,
            self.custom_mask,
            self.adapter,
            self.drafts,
            self.captures_scores,
            self.media,
            self.encode,
            self.fresh,
        ]
        .into_iter()
        .filter(|&on| on)
        .count()
            + self.switches.len()
    }

    /// Whether only its query length sets this request apart.
    #[must_use]
    pub fn plain(&self) -> bool {
        self.departures() == usize::from(self.query_len != 1)
    }
}

#[cfg(test)]
mod tests {
    use super::Request;

    #[test]
    fn a_request_holds_the_last_value_of_each_switch() {
        let r = Request::new(4, false);
        assert_eq!(r.switch("stream"), None);
        assert!(r.plain());
        let r = r.switched("stream", "audio").switched("stream", "video");
        assert_eq!(r.switch("stream"), Some("video"));
        assert_eq!(r.switches().len(), 1);
        assert!(!r.plain());
        assert_eq!(r.departures(), 2);
    }
}
