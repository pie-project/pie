//! The ways a family's checkpoints are laid out, and the one a checkpoint is.
//!
//! A container (safetensors, GGUF, a diffusers pipeline) says nothing of how a
//! model's tensors are named or which configuration keys state its shape; a
//! family does. It lists its formats, each with what only its checkpoints hold
//! and the configuration a checkpoint of it states, and [`read_one`] reads a
//! checkpoint by the one format that recognizes it: none recognizing it, or
//! two, is a refusal, not a first-that-works.

use ztensor::format::cbor::Value;

use crate::import::Error;

/// The attribute a snapshot's configuration rides under
/// (`checkpoint::file::snapshot::CONFIG`).
pub const CONFIG: &str = "config";

type Recognizes<'a> = Box<dyn Fn(&ztensor::Source) -> bool + 'a>;

/// A format that recognized a checkpoint: read already where reading was the
/// test of it, or still to read.
enum Recognized<'a, T> {
    Read(String, Result<T, Error>),
    ToRead(Format<'a, T>),
}

/// One way a family's checkpoints are laid out.
pub struct Format<'a, T> {
    name: String,
    recognizes: Option<Recognizes<'a>>,
    states: Vec<Stated>,
    read: Box<dyn FnOnce() -> Result<T, Error> + 'a>,
}

impl<'a, T> Format<'a, T> {
    /// The format `name`: a checkpoint `recognizes` holds is one of it, and
    /// `read` reads it.
    pub fn new(
        name: impl Into<String>,
        recognizes: impl Fn(&ztensor::Source) -> bool + 'a,
        read: impl FnOnce() -> Result<T, Error> + 'a,
    ) -> Format<'a, T> {
        Format {
            name: name.into(),
            recognizes: Some(Box::new(recognizes)),
            states: Vec::new(),
            read: Box::new(read),
        }
    }

    /// The format `name`, recognized by `read` reading a checkpoint: for a
    /// family whose formats share their names, so that reading is the only
    /// test of one. Two readings of a checkpoint still refuse it.
    pub fn reading(
        name: impl Into<String>,
        read: impl FnOnce() -> Result<T, Error> + 'a,
    ) -> Format<'a, T> {
        Format {
            name: name.into(),
            recognizes: None,
            states: Vec::new(),
            read: Box::new(read),
        }
    }

    /// The format's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Reads the checkpoint as this format, whether or not it is one.
    pub fn read(self) -> Result<T, Error> {
        (self.read)()
    }

    /// What a checkpoint of this format states of the model it holds: a
    /// checkpoint stating another value is another model's.
    #[must_use]
    pub fn stating(mut self, states: impl IntoIterator<Item = Stated>) -> Format<'a, T> {
        self.states.extend(states);
        self
    }
}

/// A configuration value a checkpoint states, and the value the model is
/// built with.
#[derive(Clone, Debug, PartialEq)]
pub struct Stated {
    pub key: Key,
    pub want: f64,
    pub at_least: bool,
}

/// Where a checkpoint states a value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Key {
    /// A dotted path into the snapshot's configuration (`config.json`, or a
    /// pipeline's per-component configurations under their prefix).
    Config(String),
    /// A top-level attribute of the container (a GGUF metadata key).
    Attribute(String),
}

/// The configuration value at `path` must be `want`.
pub fn config(path: impl Into<String>, want: impl Into<f64>) -> Stated {
    Stated {
        key: Key::Config(path.into()),
        want: want.into(),
        at_least: false,
    }
}

/// The container attribute `key` must be `want`.
pub fn attribute(key: impl Into<String>, want: impl Into<f64>) -> Stated {
    Stated {
        key: Key::Attribute(key.into()),
        want: want.into(),
        at_least: false,
    }
}

impl Stated {
    /// A depth or a bank's breadth: the checkpoint may hold more than the
    /// model reads, since a miniature reads a prefix of its whole model's layers
    /// or experts; fewer is another model's.
    #[must_use]
    pub fn or_deeper(mut self) -> Stated {
        self.at_least = true;
        self
    }
}

/// Reads `src` as `family` by the one of `formats` that recognizes it, once the
/// configuration it states agrees with the model's.
pub fn read_one<T>(
    family: &str,
    src: &ztensor::Source,
    formats: Vec<Format<'_, T>>,
) -> Result<T, Error> {
    let annotate = |format: &str, why: Error| match why {
        Error::Illegible { name, detail } => Error::Illegible {
            name,
            detail: format!("as {format}, {detail}"),
        },
        other => other,
    };
    let names: Vec<String> = formats.iter().map(|f| f.name.clone()).collect();
    let mut refusals: Vec<String> = Vec::new();
    let mut recognized: Vec<Recognized<'_, T>> = Vec::new();
    for format in formats {
        match &format.recognizes {
            Some(recognizes) => {
                if recognizes(src) {
                    recognized.push(Recognized::ToRead(format));
                }
            }
            None => {
                let Format {
                    name, states, read, ..
                } = format;
                match agrees(src, &states) {
                    Err(detail) => refusals.push(format!("as {name}, {detail}")),
                    Ok(()) => match read() {
                        Ok(read) => recognized.push(Recognized::Read(name, Ok(read))),
                        Err(why) => refusals.push(format!("as {name}, {why}")),
                    },
                }
            }
        }
    }
    match recognized.len() {
        0 => Err(Error::Illegible {
            name: family.to_string(),
            detail: if refusals.is_empty() {
                format!(
                    "no format of `{family}` recognizes this checkpoint; it reads {}",
                    names.join(", ")
                )
            } else {
                format!(
                    "no format of `{family}` reads this checkpoint — {}",
                    refusals.join("; ")
                )
            },
        }),
        1 => match recognized.remove(0) {
            Recognized::Read(_, read) => read,
            Recognized::ToRead(format) => {
                agrees(src, &format.states).map_err(|detail| Error::Illegible {
                    name: family.to_string(),
                    detail: format!("as {}, {detail}", format.name),
                })?;
                let name = format.name;
                (format.read)().map_err(|why| annotate(&name, why))
            }
        },
        _ => Err(Error::Illegible {
            name: family.to_string(),
            detail: format!(
                "{} all recognize this checkpoint; a format recognizes what only its \
                 checkpoints hold",
                recognized
                    .iter()
                    .map(|r| match r {
                        Recognized::Read(name, _) => name.as_str(),
                        Recognized::ToRead(format) => format.name.as_str(),
                    })
                    .collect::<Vec<_>>()
                    .join(" and ")
            ),
        }),
    }
}

/// Whether every value `states` names that `src` states is the one wanted; a
/// value the checkpoint does not state is not held against it.
fn agrees(src: &ztensor::Source, states: &[Stated]) -> Result<(), String> {
    for stated in states {
        let (found, spelled) = match &stated.key {
            Key::Config(path) => (
                src.attributes()
                    .and_then(|a| a.get(CONFIG))
                    .and_then(|c| at_path(c, path)),
                format!("its configuration's `{path}`"),
            ),
            Key::Attribute(key) => (
                src.attributes().and_then(|a| a.get(key)),
                format!("its `{key}`"),
            ),
        };
        let Some(found) = found.and_then(number) else {
            continue;
        };
        let fits = if stated.at_least {
            found >= stated.want
        } else {
            same(found, stated.want)
        };
        if !fits {
            return Err(format!(
                "the checkpoint states {spelled} as {found} and this model is built for {}{}",
                stated.want,
                if stated.at_least { " or more" } else { "" }
            ));
        }
    }
    Ok(())
}

fn at_path<'v>(value: &'v Value, path: &str) -> Option<&'v Value> {
    path.split('.').try_fold(value, |at, key| at.get(key))
}

fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Uint(n) => Some(*n as f64),
        Value::Nint(n) => Some(-1.0 - *n as f64),
        Value::Float(x) => Some(*x),
        Value::Bool(b) => Some(f64::from(u8::from(*b))),
        _ => None,
    }
}

fn same(found: f64, want: f64) -> bool {
    found == want || (found - want).abs() <= 1e-6 * want.abs().max(found.abs())
}

/// Whether `src` holds a tensor named `name`.
#[must_use]
pub fn has(src: &ztensor::Source, name: &str) -> bool {
    src.get(name).is_some()
}

/// Whether `src` holds a tensor whose name ends in `suffix`.
#[must_use]
pub fn has_suffix(src: &ztensor::Source, suffix: &str) -> bool {
    src.names().any(|name| name.ends_with(suffix))
}

/// The container attribute `key`, as text.
#[must_use]
pub fn attribute_text<'s>(src: &'s ztensor::Source, key: &str) -> Option<&'s str> {
    src.attributes()?.get(key)?.as_text()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(dir: &std::path::Path, names: &[&str], layers: u64) -> ztensor::Source {
        std::fs::create_dir_all(dir).expect("a scratch directory");
        let path = dir.join("model.zt");
        let mut writer = ztensor::Writer::create(&path).expect("the container opens");
        for name in names {
            writer
                .add(*name, vec![2], ztensor::Leaf::BF16, &[0u8; 4])
                .expect("the plane lands");
        }
        writer.finish().expect("the container closes");
        let config = Value::Map(vec![(
            Value::Text("num_hidden_layers".into()),
            Value::Uint(layers),
        )]);
        ztensor::Source::open(&path)
            .expect("it reads back")
            .with_attribute(CONFIG, config)
    }

    fn formats(depth: u32) -> Vec<Format<'static, &'static str>> {
        vec![
            Format::new("a", |src| has(src, "a.embed"), || Ok("read as a")).stating([config(
                "num_hidden_layers",
                depth,
            )
            .or_deeper()]),
            Format::new("b", |src| has(src, "b.embed"), || Ok("read as b")),
        ]
    }

    fn refusal(read: Result<&str, Error>) -> String {
        match read {
            Err(Error::Illegible { detail, .. }) => detail,
            other => panic!("a refusal by name, not {other:?}"),
        }
    }

    #[test]
    fn a_checkpoint_is_read_by_the_one_format_that_recognizes_it() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let one = source(dir.path(), &["a.embed"], 24);
        assert_eq!(read_one("f", &one, formats(24)).unwrap(), "read as a");
        assert_eq!(
            read_one("f", &one, formats(6)).unwrap(),
            "read as a",
            "a deeper checkpoint is one a shallower model reads a prefix of"
        );
    }

    #[test]
    fn a_checkpoint_no_format_or_two_recognize_is_refused() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let none = source(&dir.path().join("none"), &["c.embed"], 24);
        assert!(refusal(read_one("f", &none, formats(24))).contains("no format of `f`"));

        let both = source(&dir.path().join("both"), &["a.embed", "b.embed"], 24);
        assert!(refusal(read_one("f", &both, formats(24))).contains("a and b all recognize"));
    }

    #[test]
    fn a_checkpoint_stating_another_model_is_refused() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let shallow = source(dir.path(), &["a.embed"], 6);
        let why = refusal(read_one("f", &shallow, formats(24)));
        assert!(why.contains("`num_hidden_layers` as 6"), "{why}");
        assert!(why.contains("built for 24 or more"), "{why}");
    }
}
