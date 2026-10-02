//! A streaming text filter that never lets a marker's beginning out.
//!
//! Stop sequences and tool-call markers arrive one token at a time, and a
//! token can end in the first half of one. To stream text while still
//! catching the marker whole, the filter holds back the longest tail of what
//! it has seen that is a prefix of some marker, and releases it as soon as
//! the next text proves it was not one.

#[derive(Clone, Debug)]
pub struct HoldBack {
    markers: Vec<String>,
    held: String,
}

/// What one push released.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Released {
    /// Text that is definitely not part of a marker, in order.
    pub text: String,
    /// The marker that completed, if one did. Text before it is in `text`;
    /// text after it is discarded (the caller decides what a hit means, and
    /// it is always the end of something).
    pub hit: Option<usize>,
}

impl HoldBack {
    pub fn new(markers: Vec<String>) -> Self {
        Self {
            markers: markers.into_iter().filter(|m| !m.is_empty()).collect(),
            held: String::new(),
        }
    }

    pub fn push(&mut self, text: &str) -> Released {
        if self.markers.is_empty() {
            return Released {
                text: text.to_string(),
                hit: None,
            };
        }
        self.held.push_str(text);
        // A completed marker: the earliest one wins.
        let mut best: Option<(usize, usize)> = None; // (position, marker index)
        for (index, marker) in self.markers.iter().enumerate() {
            if let Some(at) = self.held.find(marker.as_str())
                && best.is_none_or(|(pos, _)| at < pos)
            {
                best = Some((at, index));
            }
        }
        if let Some((at, index)) = best {
            let text = self.held[..at].to_string();
            self.held.clear();
            return Released {
                text,
                hit: Some(index),
            };
        }
        // Otherwise release everything but the longest tail that could still
        // grow into a marker.
        let keep = self.longest_prefix_tail();
        let cut = self.held.len() - keep;
        let text = self.held[..cut].to_string();
        self.held = self.held[cut..].to_string();
        Released { text, hit: None }
    }

    /// Release whatever is held: the stream ended, so it was never a marker.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    fn longest_prefix_tail(&self) -> usize {
        let held = self.held.as_str();
        let mut longest = 0;
        for marker in &self.markers {
            let max = marker.len().saturating_sub(1).min(held.len());
            for len in (1..=max).rev() {
                if !held.is_char_boundary(held.len() - len) {
                    continue;
                }
                if marker.starts_with(&held[held.len() - len..]) {
                    longest = longest.max(len);
                    break;
                }
            }
        }
        longest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(markers: &[&str], pieces: &[&str]) -> (String, Option<usize>) {
        let mut hb = HoldBack::new(markers.iter().map(|m| m.to_string()).collect());
        let mut out = String::new();
        for piece in pieces {
            let r = hb.push(piece);
            out.push_str(&r.text);
            if r.hit.is_some() {
                return (out, r.hit);
            }
        }
        out.push_str(&hb.flush());
        (out, None)
    }

    #[test]
    fn holdback_every_case() {
        // A marker split across pieces is caught whole.
        assert_eq!(
            run(&["<tool_call>"], &["hi <to", "ol_ca", "ll>{}"]),
            ("hi ".into(), Some(0))
        );
        // A false start is released once disproved.
        assert_eq!(run(&["STOP"], &["a ST", "b"]), ("a STb".into(), None));
        // A tail that could still be a marker is held until the end.
        let mut hb = HoldBack::new(vec!["END".into()]);
        assert_eq!(hb.push("text E").text, "text ");
        assert_eq!(hb.flush(), "E");
        // The earliest of several markers wins.
        assert_eq!(run(&["b", "ab"], &["xab"]), ("x".into(), Some(1)));
        // No markers: pass-through.
        assert_eq!(run(&[], &["a", "b"]), ("ab".into(), None));
        // Multi-byte text around a marker.
        assert_eq!(run(&["끝"], &["안녕 ", "끝!"]), ("안녕 ".into(), Some(0)));
    }
}
