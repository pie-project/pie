//! JSON as the reference templates write it: Jinja's `tojson`, which is
//! Python's `json.dumps` with `", "` and `": "` between items, keys in the
//! order they came, and non-ASCII text as it is.

use std::io;

use serde::Serialize;
use serde_json::ser::Formatter;

struct Spaced;

impl Formatter for Spaced {
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b": ")
    }
}

/// `value` as `tojson` writes it.
#[must_use]
pub fn tojson(value: &serde_json::Value) -> String {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, Spaced);
    value.serialize(&mut ser).expect("a JSON value serializes");
    String::from_utf8(out).expect("serde_json writes UTF-8")
}

/// A JSON document re-written as `tojson` writes it; text that is not JSON
/// stays as it is.
#[must_use]
pub fn retojson(text: &str) -> String {
    serde_json::from_str::<serde_json::Value>(text)
        .map_or_else(|_| text.to_string(), |v| tojson(&v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_every_case() {
        assert_eq!(
            retojson(
                r#"{"type":"function","function":{"name":"f","required":["a","b"],"x":1.5,"é":"ü"}}"#
            ),
            r#"{"type": "function", "function": {"name": "f", "required": ["a", "b"], "x": 1.5, "é": "ü"}}"#
        );
        assert_eq!(retojson("not json"), "not json");
        assert_eq!(retojson("{}"), "{}");
        assert_eq!(retojson("[]"), "[]");
    }
}
